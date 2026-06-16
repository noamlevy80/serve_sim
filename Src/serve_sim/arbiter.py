"""Resource arbitration: co-run several event generators with rescaling.

A single :class:`~serve_sim.events.EventGenerator` assumes it owns its devices
outright. When several generators (e.g. one per concurrently-served batch) share
the same physical compute devices and memory devices, a loaded resource must be
split between the work using it. The PRD calls this *event rescaling*: a memory
device that was serving one compute event and is then asked to serve a second
concurrent event is rescaled to half its bandwidth, prorated for the time
already elapsed; compute and bandwidth are divided equally among concurrent
users.

This module implements that as a fluid (processor-sharing) co-simulation. Each
generator is added as a *job*; the arbiter replays the jobs' events on one
shared timeline, and whenever multiple in-flight events demand the same resource
that resource's rate is divided equally among them. Rates are recomputed every
time an event starts or finishes a demand, which prorates in-progress events
exactly. With a single job, or with jobs that touch disjoint resources, the
result is identical to running each generator on its own.

Resources are identified by object identity: two jobs contend only when they
were handed the *same* :class:`ComputeDevice` / :class:`MemoryDevice` instance.
A compute event loads its device's compute pool (for FLOPs) and its first-tier
memory's bandwidth (for bytes); a transfer event loads the second-tier memory it
streams from; a kernel-launch event is a fixed, unshared wait.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .events import ComputeEvent, EventGenerator, EventSchedule

# Completion tolerance, relative to each demand's total work.
_REL_TOL = 1e-12


@dataclass
class _Demand:
    """One resource demand of an event (compute FLOPs or memory bytes)."""

    resource: tuple
    work: float
    rate: float  # uncontended rate (work / uncontended_time)
    remaining: float = field(init=False)

    def __post_init__(self) -> None:
        self.remaining = self.work

    @property
    def done(self) -> bool:
        return self.remaining <= self.work * _REL_TOL


@dataclass
class _Task:
    """An event under co-simulation, with mutable progress state."""

    job: int
    order: int  # position within the job (for output ordering)
    event: ComputeEvent
    demands: list[_Demand]
    latency: float
    co_start: float = 0.0
    co_end: float | None = None
    preds: list[int] = field(default_factory=list)
    pending: int = 0
    ready_time: float = 0.0
    latency_left: float = field(init=False)

    def __post_init__(self) -> None:
        self.latency_left = self.latency

    @property
    def latency_done(self) -> bool:
        return self.latency_left <= max(self.latency, 1.0) * _REL_TOL

    @property
    def finished(self) -> bool:
        return self.latency_done and all(d.done for d in self.demands)


@dataclass
class ArbiterResult:
    """Per-job rescaled schedules plus the shared makespan."""

    schedules: list[EventSchedule]

    @property
    def makespan(self) -> float:
        return max((s.makespan for s in self.schedules), default=0.0)

    @property
    def total_flops(self) -> float:
        return sum(s.total_flops for s in self.schedules)

    @property
    def total_bytes(self) -> float:
        return sum(s.total_bytes for s in self.schedules)


class ResourceArbiter:
    """Co-runs several event generators, rescaling events under contention."""

    def __init__(self) -> None:
        self._jobs: list[tuple[list[ComputeEvent], list]] = []

    def add_job(
        self,
        generator: EventGenerator,
        shards,
        expert_trace=None,
        expert_cache_capacity=None,
    ) -> None:
        """Add a generator's work as a job.

        The generator is run in isolation to obtain its events (and the device
        objects they reference); the arbiter then re-times those events against
        the other jobs sharing the same resources.
        """

        schedule = generator.run(
            shards,
            expert_trace=expert_trace,
            expert_cache_capacity=expert_cache_capacity,
        )
        self._jobs.append((schedule.events, generator.devices))

    # --- demand extraction --------------------------------------------------

    @staticmethod
    def _demands_for(event: ComputeEvent, devices: list) -> tuple[list[_Demand], float]:
        """Resource demands and fixed latency for one isolated event."""

        if event.phase == "kernel_launch":
            return [], event.duration

        device = devices[event.device_index]
        demands: list[_Demand] = []
        if event.compute_time > 0 and event.flops > 0:
            rate = event.flops / event.compute_time
            demands.append(_Demand(("compute", id(device)), event.flops, rate))
        if event.bandwidth_time > 0 and event.bytes_read > 0:
            if event.phase == "transfer":
                memory = device.second_tier_memory or device.first_tier_memory
            else:
                memory = device.first_tier_memory
            rate = event.bytes_read / event.bandwidth_time
            demands.append(_Demand(("bandwidth", id(memory)), event.bytes_read, rate))
        return demands, 0.0

    # --- co-simulation ------------------------------------------------------

    def run(self) -> ArbiterResult:
        """Replay every job on a shared timeline with equal resource sharing."""

        tasks: list[_Task] = []
        per_job: list[list[_Task]] = []
        for job_index, (events, devices) in enumerate(self._jobs):
            job_tasks: list[_Task] = []
            for order, event in enumerate(events):
                demands, latency = self._demands_for(event, devices)
                task = _Task(job_index, order, event, demands, latency)
                tasks.append(task)
                job_tasks.append(task)
            self._link_dependencies(job_tasks)
            per_job.append(job_tasks)

        self._simulate(tasks)

        schedules: list[EventSchedule] = []
        for job_tasks in per_job:
            schedule = EventSchedule()
            for task in job_tasks:
                schedule.events.append(self._retime(task))
            schedules.append(schedule)
        return ArbiterResult(schedules)

    @staticmethod
    def _link_dependencies(job_tasks: list[_Task]) -> None:
        """Within a job, an event depends on every earlier event on the clock.

        ``EventGenerator`` lays a job's events on one non-decreasing clock, so an
        event's predecessors are exactly those that have already ended when it
        starts; concurrent events (e.g. expert-parallel ranks sharing a start)
        do not depend on one another. Taking the latest predecessor end as the
        earliest start reproduces the isolated schedule when uncontended.
        """

        iso_makespan = max((t.event.end for t in job_tasks), default=0.0)
        eps = iso_makespan * 1e-12 if iso_makespan > 0 else 1e-15
        for task in job_tasks:
            start = task.event.start
            preds = [
                other.order
                for other in job_tasks
                if other is not task and other.event.end <= start + eps
            ]
            task.preds = preds
            task.pending = len(preds)

    def _simulate(self, tasks: list[_Task]) -> None:
        by_order: dict[tuple[int, int], _Task] = {
            (t.job, t.order): t for t in tasks
        }
        successors: dict[int, list[_Task]] = {id(t): [] for t in tasks}
        for task in tasks:
            for pred_order in task.preds:
                pred = by_order[(task.job, pred_order)]
                successors[id(pred)].append(task)

        active: list[_Task] = []
        for task in tasks:
            if task.pending == 0:
                task.co_start = 0.0
                active.append(task)

        time = 0.0
        while active:
            # Flush any zero-work tasks that complete immediately.
            self._complete(active, time, successors)
            if not active:
                break

            sharers = self._count_sharers(active)
            dt = self._next_delta(active, sharers)
            if dt == float("inf"):
                dt = 0.0
            time += dt
            self._advance(active, sharers, dt)
            self._complete(active, time, successors)

    @staticmethod
    def _count_sharers(active: list[_Task]) -> dict[tuple, int]:
        sharers: dict[tuple, int] = {}
        for task in active:
            for demand in task.demands:
                if not demand.done:
                    sharers[demand.resource] = sharers.get(demand.resource, 0) + 1
        return sharers

    @staticmethod
    def _next_delta(active: list[_Task], sharers: dict[tuple, int]) -> float:
        dt = float("inf")
        for task in active:
            for demand in task.demands:
                if demand.done:
                    continue
                eff = demand.rate / sharers[demand.resource]
                if eff > 0:
                    dt = min(dt, demand.remaining / eff)
            if not task.latency_done:
                dt = min(dt, task.latency_left)
        return dt

    @staticmethod
    def _advance(active: list[_Task], sharers: dict[tuple, int], dt: float) -> None:
        if dt <= 0:
            return
        for task in active:
            for demand in task.demands:
                if demand.done:
                    continue
                eff = demand.rate / sharers[demand.resource]
                demand.remaining -= eff * dt
                if demand.remaining < demand.work * _REL_TOL:
                    demand.remaining = 0.0
            if not task.latency_done:
                task.latency_left -= dt
                if task.latency_left < 0:
                    task.latency_left = 0.0

    @staticmethod
    def _complete(active: list[_Task], time: float, successors: dict[int, list[_Task]]) -> None:
        finished = [task for task in active if task.finished]
        for task in finished:
            task.co_end = time
            active.remove(task)
            for succ in successors[id(task)]:
                succ.pending -= 1
                succ.ready_time = max(succ.ready_time, time)
                if succ.pending == 0:
                    succ.co_start = succ.ready_time
                    active.append(succ)

    @staticmethod
    def _retime(task: _Task) -> ComputeEvent:
        event = task.event
        start = task.co_start
        end = task.co_end if task.co_end is not None else start
        return ComputeEvent(
            group_index=event.group_index,
            phase=event.phase,
            device_index=event.device_index,
            flops=event.flops,
            bytes_read=event.bytes_read,
            compute_time=event.compute_time,
            bandwidth_time=event.bandwidth_time,
            duration=end - start,
            start=start,
            end=end,
        )
