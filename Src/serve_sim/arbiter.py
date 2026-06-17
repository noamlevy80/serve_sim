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


class IncrementalArbiter:
    """Fluid co-simulation driven incrementally from an event queue.

    :class:`ResourceArbiter` is a one-shot batch solver: every job is known up
    front and starts at ``t = 0``. The orchestrator instead needs jobs that come
    and go over time -- a new request *arrives* mid-flight, an old one drains --
    while the same equal-share rescaling applies to whatever is in flight. This
    class keeps the active events and clock as mutable state and lets the caller
    :meth:`admit` jobs at the current time and :meth:`advance_to` later instants;
    between steps the in-flight events are rescaled and prorated exactly as in the
    batch solver (the same private fluid helpers are reused).

    Admitting every job at ``t = 0`` and advancing to ``inf`` reproduces
    :meth:`ResourceArbiter.run` event-for-event.
    """

    def __init__(self) -> None:
        self._time = 0.0
        self._active: list[_Task] = []
        self._successors: dict[int, list[_Task]] = {}
        self._jobs: list[list[_Task]] = []

    # --- inspection ---------------------------------------------------------

    @property
    def time(self) -> float:
        """Current simulation clock."""

        return self._time

    @property
    def num_jobs(self) -> int:
        """Number of jobs admitted so far."""

        return len(self._jobs)

    @property
    def active_count(self) -> int:
        """In-flight (released, unfinished) events right now."""

        return len(self._active)

    def is_idle(self) -> bool:
        """Whether nothing is in flight (all admitted work has drained)."""

        return not self._active

    def job_is_done(self, job_index: int) -> bool:
        """Whether every event of a job has finished."""

        return all(task.co_end is not None for task in self._jobs[job_index])

    def job_end_time(self, job_index: int) -> float | None:
        """The job's completion time (max event end), or ``None`` if unfinished."""

        ends = [task.co_end for task in self._jobs[job_index]]
        if not ends or any(end is None for end in ends):
            return None
        return max(ends)

    def job_start_time(self, job_index: int) -> float:
        """The job's admission time on the shared clock."""

        return min((task.co_start for task in self._jobs[job_index]), default=0.0)

    def next_event_time(self) -> float | None:
        """Clock time of the next internal completion, or ``None`` if idle."""

        if not self._active:
            return None
        sharers = ResourceArbiter._count_sharers(self._active)
        dt = ResourceArbiter._next_delta(self._active, sharers)
        if dt == float("inf"):
            dt = 0.0
        return self._time + dt

    # --- driving ------------------------------------------------------------

    def admit(
        self,
        generator: EventGenerator,
        shards,
        expert_trace=None,
        expert_cache_capacity=None,
    ) -> int:
        """Run a generator in isolation and admit its events at the current time."""

        schedule = generator.run(
            shards,
            expert_trace=expert_trace,
            expert_cache_capacity=expert_cache_capacity,
        )
        return self.admit_events(schedule.events, generator.devices)

    def admit_events(self, events: list[ComputeEvent], devices: list) -> int:
        """Admit a pre-computed event list (against ``devices``) at the current time.

        The events are placed on the shared timeline starting now: their relative
        ordering/dependencies (one job's events lie on a non-decreasing clock) are
        preserved, but the whole job is shifted so its first events become ready
        at :attr:`time`.
        """

        job_index = len(self._jobs)
        job_tasks: list[_Task] = []
        for order, event in enumerate(events):
            demands, latency = ResourceArbiter._demands_for(event, devices)
            task = _Task(job_index, order, event, demands, latency)
            task.ready_time = self._time
            job_tasks.append(task)
        ResourceArbiter._link_dependencies(job_tasks)

        by_order = {(t.job, t.order): t for t in job_tasks}
        for task in job_tasks:
            self._successors[id(task)] = []
        for task in job_tasks:
            for pred_order in task.preds:
                pred = by_order[(task.job, pred_order)]
                self._successors[id(pred)].append(task)

        for task in job_tasks:
            if task.pending == 0:
                task.co_start = self._time
                self._active.append(task)

        self._jobs.append(job_tasks)
        # Flush any zero-work roots that complete immediately at the current time.
        ResourceArbiter._complete(self._active, self._time, self._successors)
        return job_index

    def advance_to(self, target: float) -> None:
        """Advance the fluid simulation up to ``target`` (``inf`` runs to idle).

        Completions strictly before ``target`` are processed (and their
        successors released); the clock stops exactly at ``target`` so the caller
        may admit a new job there. In-flight events are prorated for the elapsed
        time at their current shared rate.
        """

        while self._active:
            ResourceArbiter._complete(self._active, self._time, self._successors)
            if not self._active:
                break
            sharers = ResourceArbiter._count_sharers(self._active)
            dt = ResourceArbiter._next_delta(self._active, sharers)
            if dt == float("inf"):
                dt = 0.0
            if self._time + dt > target:
                # ``target`` falls strictly inside the current interval: prorate
                # the in-flight events and stop exactly there.
                dt = target - self._time
                if dt > 0:
                    ResourceArbiter._advance(self._active, sharers, dt)
                    self._time = target
                ResourceArbiter._complete(
                    self._active, self._time, self._successors
                )
                return
            # ``target`` is at or beyond the next natural completion: take the
            # full step so the draining demand is consumed exactly. Clamping to
            # ``target`` here would lose sub-ULP bits at large absolute times
            # (e.g. arrival at t=12.5 with microsecond events), leaving an
            # undrainable residual and an infinite loop.
            self._time += dt
            ResourceArbiter._advance(self._active, sharers, dt)
            ResourceArbiter._complete(self._active, self._time, self._successors)
        # The machine drained before ``target``; still advance the idle clock so
        # a job admitted at ``target`` starts there (no-op for ``inf``).
        if target != float("inf") and target > self._time:
            self._time = target

    def run_to_idle(self) -> None:
        """Advance until all admitted work has drained."""

        self.advance_to(float("inf"))

    # --- output -------------------------------------------------------------

    def result(self) -> ArbiterResult:
        """Per-job rescaled schedules (call once :meth:`is_idle`)."""

        schedules: list[EventSchedule] = []
        for job_tasks in self._jobs:
            schedule = EventSchedule()
            for task in job_tasks:
                schedule.events.append(ResourceArbiter._retime(task))
            schedules.append(schedule)
        return ArbiterResult(schedules)
