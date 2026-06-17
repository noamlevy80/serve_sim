"""Orchestrator v0: a strictly event-driven serving loop.

This is the first end-to-end driver that ties the pieces together: requests
*arrive* over time, are collected into a concurrency window, dispatched as
batches onto a fixed slice of the system's compute devices, co-run on the shared
timeline by the :class:`~serve_sim.arbiter.IncrementalArbiter` (which rescales
events when batches contend for the same resource), and retired on completion.

v0 deliberately uses the *trivial fixed policy* the plan calls for:

- **No PDD:** a request's prefill and decode for its turn run as one job (the
  existing work-shard -> event pipeline), not split across prefill/decode pools.
- **Fixed parallelism:** the engine is the first ``pipeline_parallel x
  expert_parallel`` compute devices of the system; the roofline parallelism
  search comes in a later stage.
- **Target concurrency only:** the one admission knob caps how many sequences
  are in flight; beyond that, ready sequences wait for a completion.
- **Concurrency-window batching:** ready sequences of the same model are grouped
  and dispatched when the window fills (max batch size) or its duration elapses.

Single-turn requests, one-shot batched decode, single-tier devices and no KV /
weight reuse are v0 simplifications; multi-turn conversations, iteration-level
batching, expert streaming and prefix reuse are layered on in later stages. The
loop itself is strictly event driven: it advances to the next arrival, window
deadline or in-flight completion -- never a fixed step.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

from .arbiter import IncrementalArbiter
from .events import EventGenerator
from .parallelism import ParallelismPlanner
from .pdd import context_kv_bytes, kv_transfer_duration, split_work
from .placement import EnginePool, EngineSlot
from .shards import WorkShardGenerator
from .system import System
from .tokenizer import Tokenizer
from .tracker import SequenceWork
from .workload import Workload


@dataclass(frozen=True)
class StrategyConfig:
    """The orchestration knobs (v0 subset).

    Attributes:
        max_batch_size: Sequences per dispatched batch (window fill threshold).
        max_window_duration: Seconds a window stays open before dispatching
            whatever it has collected (``0`` dispatches as soon as work is ready).
        target_concurrency: Max sequences in flight, or ``None`` for unbounded.
        pipeline_parallel: Fixed pipeline-parallel degree of the engine.
        expert_parallel: Fixed expert-parallel degree of the engine.
        auto_parallelism: When ``True``, the orchestrator treats
            ``pipeline_parallel x expert_parallel`` only as the fixed engine size
            and, per batch, searches the ``(pp, ep)`` factorizations of that size
            for the fastest one that fits device memory. When ``False`` (default)
            the fixed ``pp``/``ep`` are used verbatim.
        allow_pdd: When ``True``, prefill and decode run on separate engine pools
            (split by ``prefill_engine_fraction``) with a modeled KV-cache
            transfer between them. When ``False`` (default) each request's prefill
            and decode run as one job.
        prefill_engine_fraction: Fraction of engine slots assigned to the prefill
            pool when ``allow_pdd`` is set (the rest serve decode). A single
            swappable partition point -- dynamic repartitioning is future work.
        prefill_chunk_size: Optional prefill chunking applied to every batch.
        event_random_factor_range: Per-event time is multiplied by
            ``1 + U(-range, range)`` to model system randomness; ``0`` disables it.
        random_seed: Seed for the run's randomness (event-time perturbation);
            ``None`` draws a non-deterministic seed.
    """

    max_batch_size: int = 1
    max_window_duration: float = 0.0
    target_concurrency: int | None = None
    pipeline_parallel: int = 1
    expert_parallel: int = 1
    auto_parallelism: bool = False
    allow_pdd: bool = False
    prefill_engine_fraction: float = 0.5
    prefill_chunk_size: int | None = None
    event_random_factor_range: float = 0.0
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.max_window_duration < 0:
            raise ValueError("max_window_duration must be non-negative")
        if self.target_concurrency is not None and self.target_concurrency < 1:
            raise ValueError("target_concurrency must be >= 1 when set")
        if self.pipeline_parallel < 1 or self.expert_parallel < 1:
            raise ValueError("parallelism degrees must be >= 1")
        if self.prefill_chunk_size is not None and self.prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be >= 1")
        if not 0.0 < self.prefill_engine_fraction < 1.0:
            raise ValueError("prefill_engine_fraction must be in (0, 1)")
        if not 0.0 <= self.event_random_factor_range < 1.0:
            raise ValueError("event_random_factor_range must be in [0, 1)")


@dataclass(frozen=True)
class Request:
    """One serving request: a single sequence to prefill and decode.

    Attributes:
        request_id: Caller-assigned identifier (unique within a run).
        model: The model object serving the request (flat ``Model`` or
            ``LayeredModel``). Requests sharing a model *instance* may batch.
        prompt_tokens: Total prompt length (including any cached prefix).
        output_tokens: Tokens to generate.
        arrival_time: When the request enters the system.
        cached_tokens: Prompt tokens already in cache (prefill skips these).
    """

    request_id: int
    model: object
    prompt_tokens: int
    output_tokens: int
    arrival_time: float = 0.0
    cached_tokens: int = 0

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.output_tokens < 0 or self.cached_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.cached_tokens > self.prompt_tokens:
            raise ValueError("cached_tokens cannot exceed prompt_tokens")
        if self.arrival_time < 0:
            raise ValueError("arrival_time must be non-negative")

    @property
    def work(self) -> SequenceWork:
        """The per-sequence work description for the shard generator."""

        return SequenceWork(
            cached_tokens=self.cached_tokens,
            prefill_tokens=self.prompt_tokens - self.cached_tokens,
            decode_tokens=self.output_tokens,
        )

    @classmethod
    def from_workload(
        cls,
        request_id: int,
        workload: Workload,
        model: object,
        tokenizer: Tokenizer,
        *,
        arrival_time: float = 0.0,
        turn_index: int = 0,
    ) -> "Request":
        """Build a single-sequence request from one turn of a workload."""

        from .tracker import SequenceTracker

        tracker = SequenceTracker.from_turn(workload, turn_index, tokenizer)
        return cls(
            request_id=request_id,
            model=model,
            prompt_tokens=tracker.prompt_tokens,
            output_tokens=tracker.output_tokens,
            arrival_time=arrival_time,
            cached_tokens=tracker.cached_tokens,
        )


@dataclass
class RequestRecord:
    """Per-request timing produced by a run."""

    request_id: int
    arrival_time: float
    dispatch_time: float
    completion_time: float
    prompt_tokens: int
    output_tokens: int
    batch_index: int
    first_token_time: float | None = None

    @property
    def queue_delay(self) -> float:
        """Seconds spent waiting between arrival and dispatch."""

        return self.dispatch_time - self.arrival_time

    @property
    def latency(self) -> float:
        """End-to-end seconds from arrival to completion."""

        return self.completion_time - self.arrival_time

    @property
    def ttft(self) -> float | None:
        """Time-to-first-token: arrival to the first decode step's completion."""

        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def tpot(self) -> float | None:
        """Time-per-output-token after the first (``None`` if not measurable)."""

        if self.first_token_time is None or self.output_tokens <= 1:
            return None
        return (self.completion_time - self.first_token_time) / (self.output_tokens - 1)


@dataclass(frozen=True)
class EventRecord:
    """One simulation event, captured for the raw event log.

    Each event is recorded twice: once as generated in isolation (``rescaled``
    false) and once after the arbiter re-times it for resource contention
    (``rescaled`` true).
    """

    job_index: int
    batch_index: int
    job_phase: str  # "full", "prefill" or "decode" (the dispatch kind)
    request_ids: tuple[int, ...]
    group_index: int
    phase: str  # event phase: prefill / decode / transfer / kernel_launch
    device: str  # device name ("" for the no-device sentinel)
    flops: float
    bytes_read: float
    compute_time: float
    bandwidth_time: float
    duration: float
    start: float
    end: float
    rescaled: bool


@dataclass(frozen=True)
class JobRecord:
    """Per-job placement and reserved-memory footprint, for occupancy reports."""

    job_index: int
    batch_index: int
    job_phase: str
    request_ids: tuple[int, ...]
    devices: tuple[str, ...]
    start: float
    end: float
    per_device_bytes: float  # reserved weights + KV per device (0 if unknown)


@dataclass(frozen=True)
class RunProgress:
    """A progress update emitted as a run retires completed sequences.

    ``sim_time`` is the simulation clock at the moment of the update; ``wall_time``
    is the real time elapsed since the run started.
    """

    completed: int
    total: int
    sim_time: float
    wall_time: float


# Called with a :class:`RunProgress` whenever the completed-sequence count grows.
ProgressCallback = Callable[["RunProgress"], None]


@dataclass
class RunResult:
    """Minimal run state: per-request records and the overall makespan.

    Beyond the per-request timings, a run also captures the raw event log
    (``events``, each event recorded before and after rescaling) and per-job
    placement/footprint records (``jobs``) so the outputs layer can derive
    utilization and memory-occupancy reports.
    """

    records: list[RequestRecord] = field(default_factory=list)
    num_batches: int = 0
    events: list[EventRecord] = field(default_factory=list)
    jobs: list[JobRecord] = field(default_factory=list)

    @property
    def makespan(self) -> float:
        """Wall-clock end of the last request (0 if none)."""

        return max((r.completion_time for r in self.records), default=0.0)

    def record_for(self, request_id: int) -> RequestRecord:
        for record in self.records:
            if record.request_id == request_id:
                return record
        raise KeyError(f"no record for request {request_id}")


class Simulator:
    """Runs a set of requests through a system under a fixed strategy."""

    def __init__(self, system: System, strategy: StrategyConfig | None = None) -> None:
        self.system = system
        self.strategy = strategy or StrategyConfig()
        degree = self.strategy.pipeline_parallel * self.strategy.expert_parallel
        devices = system.compute_devices
        if len(devices) < degree:
            raise ValueError(
                f"system has {len(devices)} compute devices but the engine needs "
                f"{degree} (pipeline_parallel x expert_parallel)"
            )
        # Carve the system into fixed-size engine slots. Concurrent batches land
        # on disjoint slots when free (running independently); when every slot is
        # busy they time-share the least-loaded one (the arbiter then shares that
        # device set between them).
        self._pool = EnginePool(devices, degree)
        self._degree = degree
        # Prefill/decode disaggregation: split the slots into a prefill pool and
        # a decode pool over disjoint device sets. The partition point is a
        # single knob (prefill_engine_fraction); dynamic repartitioning is future
        # work.
        self._prefill_pool: EnginePool | None = None
        self._decode_pool: EnginePool | None = None
        if self.strategy.allow_pdd:
            num_slots = self._pool.num_slots
            if num_slots < 2:
                raise ValueError(
                    f"PDD needs at least 2 engine slots but the system has "
                    f"{num_slots} (degree {degree} over {len(devices)} devices)"
                )
            prefill_slots = round(self.strategy.prefill_engine_fraction * num_slots)
            prefill_slots = max(1, min(num_slots - 1, prefill_slots))
            cut = prefill_slots * degree
            self._prefill_pool = EnginePool(devices[:cut], degree)
            self._decode_pool = EnginePool(devices[cut:], degree)
        # When auto-parallelism is on, a planner re-factors the fixed engine size
        # (degree) into the fastest memory-feasible (pp, ep) per batch; cached per
        # model instance since the layout depends only on the model and device.
        self._planners: dict[int, ParallelismPlanner] = {}
        self._rng = random.Random(self.strategy.random_seed)

    def _planner_for(self, model: object) -> ParallelismPlanner:
        key = id(model)
        planner = self._planners.get(key)
        if planner is None:
            planner = ParallelismPlanner(model, self._pool.slots[0].devices[0])
            self._planners[key] = planner
        return planner

    def run(
        self, requests: list[Request], *, progress: ProgressCallback | None = None
    ) -> RunResult:
        """Event-driven serving loop over ``requests``; returns per-request timing.

        If ``progress`` is given it is called with a :class:`RunProgress` each time
        the run retires one or more sequences (and once when it finishes).
        """

        if self.strategy.allow_pdd:
            return self._run_pdd(requests, progress=progress)

        strategy = self.strategy
        arrivals = sorted(requests, key=lambda r: r.arrival_time)
        arrival_pos = 0
        ready: list[Request] = []
        window_open: float | None = None

        arbiter = IncrementalArbiter()
        in_flight = 0
        batch_index = 0
        total = len(requests)
        start_wall = time.perf_counter()

        def report(now: float) -> None:
            if progress is not None:
                progress(RunProgress(len(result.records), total, now,
                                     time.perf_counter() - start_wall))

        # job_index -> (requests, dispatch_time, batch_index, slot)
        jobs: dict[int, tuple[list[Request], float, int, EngineSlot]] = {}
        # job_index -> (job_phase, requests, slot, batch_index, pp, ep)
        job_meta: dict[int, tuple[str, list[Request], EngineSlot, int, int, int]] = {}
        reported: set[int] = set()
        result = RunResult()

        def next_arrival_time() -> float | None:
            if arrival_pos < len(arrivals):
                return arrivals[arrival_pos].arrival_time
            return None

        while True:
            # Candidate next decision times.
            candidates: list[float] = []
            na = next_arrival_time()
            if na is not None:
                candidates.append(na)
            if window_open is not None:
                deadline = window_open + strategy.max_window_duration
                # Only a *future* deadline is an event to advance to; once the
                # window has elapsed, further dispatch is gated by concurrency /
                # completions, not by the (past) deadline.
                if deadline > arbiter.time:
                    candidates.append(deadline)
            ne = arbiter.next_event_time()
            if ne is not None:
                candidates.append(ne)

            if not candidates:
                # Nothing pending anywhere. If ready work remains it must be
                # dispatchable now (window already accounted for above).
                if not ready:
                    break
                now = arbiter.time
            else:
                now = min(candidates)
                arbiter.advance_to(now)

            # 1) Retire any jobs that have finished by ``now``.
            completed_before = len(result.records)
            for job_index, (reqs, dispatch_time, b_index, slot) in jobs.items():
                if job_index in reported or not arbiter.job_is_done(job_index):
                    continue
                reported.add(job_index)
                end = arbiter.job_end_time(job_index)
                in_flight -= len(reqs)
                self._pool.release(slot)
                for req in reqs:
                    result.records.append(
                        RequestRecord(
                            request_id=req.request_id,
                            arrival_time=req.arrival_time,
                            dispatch_time=dispatch_time,
                            completion_time=end,
                            prompt_tokens=req.prompt_tokens,
                            output_tokens=req.output_tokens,
                            batch_index=b_index,
                        )
                    )
            if len(result.records) != completed_before:
                report(now)

            # 2) Admit arrivals that have occurred by ``now``.
            while arrival_pos < len(arrivals) and arrivals[arrival_pos].arrival_time <= now:
                ready.append(arrivals[arrival_pos])
                arrival_pos += 1
                if window_open is None:
                    window_open = now

            # 3) Dispatch decisions.
            window_elapsed = (
                window_open is not None
                and now >= window_open + strategy.max_window_duration
            )
            progressed = False
            while ready:
                fill = len(ready) >= strategy.max_batch_size
                if not (fill or window_elapsed):
                    break
                batch = self._take_batch(ready, in_flight)
                if batch is None:
                    break  # concurrency full; wait for a completion
                in_flight += len(batch)
                job_index, slot, pp, ep = self._dispatch(arbiter, batch)
                jobs[job_index] = (batch, now, batch_index, slot)
                job_meta[job_index] = ("full", batch, slot, batch_index, pp, ep)
                batch_index += 1
                progressed = True

            if not ready:
                window_open = None

            # Termination: everything arrived, nothing queued, nothing in flight.
            if (
                arrival_pos >= len(arrivals)
                and not ready
                and arbiter.is_idle()
            ):
                break

            # Guard against a stall: no future event and we made no progress.
            if not candidates and not progressed:
                break

        result.num_batches = batch_index
        self._collect_outputs(result, arbiter, job_meta)
        return result

    def _run_pdd(
        self, requests: list[Request], *, progress: ProgressCallback | None = None
    ) -> RunResult:
        """Two-phase PDD loop: prefill pool -> KV transfer -> decode pool.

        A request is prefilled on the prefill pool; on completion its KV cache is
        transferred (a modeled delay derived from the link bandwidth) to the
        decode pool, where it is decoded. Prefill and decode batch independently
        (each with its own window/fill); ``target_concurrency`` counts a sequence
        as in flight from prefill dispatch until decode completion.
        """

        strategy = self.strategy
        arrivals = sorted(requests, key=lambda r: r.arrival_time)
        arrival_pos = 0
        prefill_ready: list[Request] = []
        decode_ready: list[Request] = []
        # Prefill done, KV transfer in flight: (request, decode_ready_time).
        pending: list[tuple[Request, float]] = []
        prefill_window: float | None = None
        decode_window: float | None = None

        arbiter = IncrementalArbiter()
        in_flight = 0  # counted from prefill dispatch to decode completion
        batch_index = 0
        total = len(requests)
        start_wall = time.perf_counter()

        # job_index -> (kind, requests, slot, pool)
        jobs: dict[int, tuple[str, list[Request], EngineSlot, EnginePool]] = {}
        # job_index -> (job_phase, requests, slot, batch_index, pp, ep)
        job_meta: dict[int, tuple[str, list[Request], EngineSlot, int, int, int]] = {}
        reported: set[int] = set()
        # id(request) -> (prefill dispatch time, prefill batch index)
        meta: dict[int, tuple[float, int]] = {}
        result = RunResult()

        prefill_rep = self._prefill_pool.slots[0].devices[0]
        decode_rep = self._decode_pool.slots[0].devices[0]

        def report(now: float) -> None:
            if progress is not None:
                progress(RunProgress(len(result.records), total, now,
                                     time.perf_counter() - start_wall))

        def transfer_ready_time(req: Request, prefill_end: float) -> float:
            kv_bytes = context_kv_bytes(req.model, req.prompt_tokens)
            return prefill_end + kv_transfer_duration(
                kv_bytes, prefill_rep, decode_rep, self.system
            )

        def next_arrival_time() -> float | None:
            if arrival_pos < len(arrivals):
                return arrivals[arrival_pos].arrival_time
            return None

        while True:
            candidates: list[float] = []
            na = next_arrival_time()
            if na is not None:
                candidates.append(na)
            if prefill_window is not None:
                deadline = prefill_window + strategy.max_window_duration
                if deadline > arbiter.time:
                    candidates.append(deadline)
            if decode_window is not None:
                deadline = decode_window + strategy.max_window_duration
                if deadline > arbiter.time:
                    candidates.append(deadline)
            for _, ready_time in pending:
                if ready_time > arbiter.time:
                    candidates.append(ready_time)
            ne = arbiter.next_event_time()
            if ne is not None:
                candidates.append(ne)

            if not candidates:
                if prefill_ready or decode_ready or pending:
                    now = arbiter.time
                else:
                    break
            else:
                now = min(candidates)
                arbiter.advance_to(now)

            # 1) Retire finished jobs.
            completed_before = len(result.records)
            for job_index, (kind, reqs, slot, pool) in jobs.items():
                if job_index in reported or not arbiter.job_is_done(job_index):
                    continue
                reported.add(job_index)
                end = arbiter.job_end_time(job_index)
                pool.release(slot)
                if kind == "prefill":
                    # KV handoff: schedule each sequence for decode after its
                    # transfer completes (the sequence stays "in flight").
                    for req in reqs:
                        pending.append((req, transfer_ready_time(req, end)))
                else:  # decode complete -> request done
                    in_flight -= len(reqs)
                    for req in reqs:
                        dispatch_time, b_index = meta[id(req)]
                        result.records.append(
                            RequestRecord(
                                request_id=req.request_id,
                                arrival_time=req.arrival_time,
                                dispatch_time=dispatch_time,
                                completion_time=end,
                                prompt_tokens=req.prompt_tokens,
                                output_tokens=req.output_tokens,
                                batch_index=b_index,
                            )
                        )
            if len(result.records) != completed_before:
                report(now)

            # 2) Admit arrivals into the prefill queue.
            while arrival_pos < len(arrivals) and arrivals[arrival_pos].arrival_time <= now:
                prefill_ready.append(arrivals[arrival_pos])
                arrival_pos += 1
                if prefill_window is None:
                    prefill_window = now

            # 3) Move transferred sequences into the decode queue.
            still_pending: list[tuple[Request, float]] = []
            for req, ready_time in pending:
                if ready_time <= now:
                    decode_ready.append(req)
                    if decode_window is None:
                        decode_window = now
                else:
                    still_pending.append((req, ready_time))
            pending = still_pending

            progressed = False

            # 4) Dispatch prefill batches (concurrency-gated).
            prefill_elapsed = (
                prefill_window is not None
                and now >= prefill_window + strategy.max_window_duration
            )
            while prefill_ready:
                fill = len(prefill_ready) >= strategy.max_batch_size
                if not (fill or prefill_elapsed):
                    break
                batch = self._take_batch(prefill_ready, in_flight)
                if batch is None:
                    break  # concurrency full; wait for a decode completion
                in_flight += len(batch)
                job_index, slot, pp, ep = self._dispatch(
                    arbiter, batch, pool=self._prefill_pool, phase="prefill"
                )
                for req in batch:
                    meta[id(req)] = (now, batch_index)
                jobs[job_index] = ("prefill", batch, slot, self._prefill_pool)
                job_meta[job_index] = ("prefill", batch, slot, batch_index, pp, ep)
                batch_index += 1
                progressed = True
            if not prefill_ready:
                prefill_window = None

            # 5) Dispatch decode batches (already counted in flight).
            decode_elapsed = (
                decode_window is not None
                and now >= decode_window + strategy.max_window_duration
            )
            while decode_ready:
                fill = len(decode_ready) >= strategy.max_batch_size
                if not (fill or decode_elapsed):
                    break
                batch = self._take_decode_batch(decode_ready)
                job_index, slot, pp, ep = self._dispatch(
                    arbiter, batch, pool=self._decode_pool, phase="decode"
                )
                jobs[job_index] = ("decode", batch, slot, self._decode_pool)
                job_meta[job_index] = ("decode", batch, slot, batch_index, pp, ep)
                batch_index += 1
                progressed = True
            if not decode_ready:
                decode_window = None

            # Termination: everything arrived, drained and idle.
            if (
                arrival_pos >= len(arrivals)
                and not prefill_ready
                and not decode_ready
                and not pending
                and arbiter.is_idle()
            ):
                break

            if not candidates and not progressed:
                break

        result.num_batches = batch_index
        self._collect_outputs(result, arbiter, job_meta)
        return result

    # --- helpers ------------------------------------------------------------

    def _collect_outputs(
        self,
        result: RunResult,
        arbiter: IncrementalArbiter,
        job_meta: dict[int, tuple[str, list[Request], EngineSlot, int, int, int]],
    ) -> None:
        """Attach the raw event log, per-job footprints and first-token times.

        For every job the events are recorded twice -- as generated in isolation
        and after the arbiter rescales them for contention -- and the first decode
        step's completion (from the rescaled events) is the first-token time for
        every request the job serves.
        """

        first_token: dict[int, float] = {}
        for job_index, (job_phase, reqs, slot, b_index, pp, ep) in job_meta.items():
            request_ids = tuple(req.request_id for req in reqs)
            devices = slot.devices
            device_names = tuple(d.name for d in devices)
            original = arbiter.job_original_events(job_index)
            rescaled = arbiter.job_rescaled_events(job_index)

            for events, is_rescaled in ((original, False), (rescaled, True)):
                for ev in events:
                    if 0 <= ev.device_index < len(device_names):
                        dev_name = device_names[ev.device_index]
                    else:
                        dev_name = ""
                    result.events.append(
                        EventRecord(
                            job_index=job_index,
                            batch_index=b_index,
                            job_phase=job_phase,
                            request_ids=request_ids,
                            group_index=ev.group_index,
                            phase=ev.phase,
                            device=dev_name,
                            flops=ev.flops,
                            bytes_read=ev.bytes_read,
                            compute_time=ev.compute_time,
                            bandwidth_time=ev.bandwidth_time,
                            duration=ev.duration,
                            start=ev.start,
                            end=ev.end,
                            rescaled=is_rescaled,
                        )
                    )

            # First-token time: completion of the earliest decode step.
            decode_events = [e for e in rescaled if e.phase == "decode"]
            if decode_events:
                first_group = min(e.group_index for e in decode_events)
                ft = max(e.end for e in decode_events if e.group_index == first_group)
                for rid in request_ids:
                    first_token[rid] = ft

            start = min((e.start for e in rescaled), default=0.0)
            end = max((e.end for e in rescaled), default=start)
            kv_tokens = sum(req.prompt_tokens + req.output_tokens for req in reqs)
            per_device_bytes = self._job_footprint(reqs[0].model, pp, ep, kv_tokens)
            result.jobs.append(
                JobRecord(
                    job_index=job_index,
                    batch_index=b_index,
                    job_phase=job_phase,
                    request_ids=request_ids,
                    devices=device_names,
                    start=start,
                    end=end,
                    per_device_bytes=per_device_bytes,
                )
            )

        for record in result.records:
            if record.request_id in first_token:
                record.first_token_time = first_token[record.request_id]

        result.events.sort(key=lambda e: (e.rescaled, e.start, e.job_index, e.group_index))
        result.jobs.sort(key=lambda j: j.job_index)

    def _job_footprint(self, model: object, pp: int, ep: int, kv_tokens: int) -> float:
        """Reserved per-device bytes (weights + KV) for a job, or 0 if unknown."""

        try:
            return float(self._planner_for(model).footprint(pp, ep, kv_tokens))
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _take_batch(self, ready: list[Request], in_flight: int) -> list[Request] | None:
        """Pull up to ``max_batch_size`` same-model requests, honoring concurrency.

        Returns ``None`` if concurrency is exhausted and work is already in
        flight (so a completion must free a slot first). When nothing is in
        flight a batch is always returned, to guarantee progress.
        """

        strategy = self.strategy
        if strategy.target_concurrency is None:
            slots = strategy.max_batch_size
        else:
            slots = strategy.target_concurrency - in_flight
            if slots <= 0:
                if in_flight > 0:
                    return None
                slots = strategy.max_batch_size  # force progress
        limit = min(strategy.max_batch_size, slots)

        model = ready[0].model
        batch: list[Request] = []
        rest: list[Request] = []
        for req in ready:
            if len(batch) < limit and req.model is model:
                batch.append(req)
            else:
                rest.append(req)
        ready[:] = rest
        return batch

    def _take_decode_batch(self, ready: list[Request]) -> list[Request]:
        """Pull up to ``max_batch_size`` same-model requests for decode.

        Unlike :meth:`_take_batch` this applies no concurrency gate: decode
        sequences were already counted toward ``target_concurrency`` when their
        prefill was dispatched, so gating them again would deadlock.
        """

        limit = self.strategy.max_batch_size
        model = ready[0].model
        batch: list[Request] = []
        rest: list[Request] = []
        for req in ready:
            if len(batch) < limit and req.model is model:
                batch.append(req)
            else:
                rest.append(req)
        ready[:] = rest
        return batch


    def _dispatch(
        self,
        arbiter: IncrementalArbiter,
        batch: list[Request],
        *,
        pool: EnginePool | None = None,
        phase: str = "full",
    ) -> tuple[int, EngineSlot, int, int]:
        """Place a batch on an engine slot and admit its events to the arbiter.

        ``phase`` selects the work generated: ``"full"`` (prefill + decode in one
        job, the default), ``"prefill"`` (prompt forward pass only) or
        ``"decode"`` (generation from a fully-cached prompt). ``pool`` defaults to
        the single engine pool; PDD passes the prefill or decode pool. Returns the
        arbiter job index, the slot the batch occupies (released on completion),
        and the ``(pipeline_parallel, expert_parallel)`` arrangement chosen.
        """

        model = batch[0].model
        pool = pool or self._pool
        slot = pool.place(model).slot
        work = [self._phase_work(req, phase) for req in batch]
        shards = WorkShardGenerator(model).generate(
            work, prefill_chunk_size=self.strategy.prefill_chunk_size
        )
        pp, ep = self._parallelism_for(model, batch, shards)
        generator = EventGenerator(
            model,
            list(slot.devices),
            pipeline_parallel=pp,
            expert_parallel=ep,
            event_random_factor_range=self.strategy.event_random_factor_range,
            rng=self._rng,
        )
        return arbiter.admit(generator, shards), slot, pp, ep

    @staticmethod
    def _phase_work(req: Request, phase: str) -> SequenceWork:
        """The :class:`SequenceWork` for a request in a given dispatch phase."""

        if phase == "full":
            return req.work
        prefill, decode = split_work(
            req.cached_tokens, req.prompt_tokens, req.output_tokens
        )
        if phase == "prefill":
            return prefill
        if phase == "decode":
            return decode
        raise ValueError(f"unknown dispatch phase {phase!r}")

    def _parallelism_for(
        self, model: object, batch: list[Request], shards: list
    ) -> tuple[int, int]:
        """Pick (pipeline_parallel, expert_parallel) for a batch.

        Fixed from the strategy unless ``auto_parallelism`` is set, in which case
        the planner searches the factorizations of the engine size for the
        fastest memory-feasible arrangement of this batch.
        """

        if not self.strategy.auto_parallelism:
            return self.strategy.pipeline_parallel, self.strategy.expert_parallel

        flops_by_dtype: dict[int, float] = {}
        total_bytes = 0.0
        for shard in shards:
            flops_by_dtype[shard.flops_dtype_bytes] = (
                flops_by_dtype.get(shard.flops_dtype_bytes, 0.0) + shard.flops
            )
            total_bytes += shard.bytes_read
        kv_tokens = sum(req.prompt_tokens + req.output_tokens for req in batch)
        choice = self._planner_for(model).plan(
            self._degree,
            kv_tokens=kv_tokens,
            flops_by_dtype=flops_by_dtype,
            total_bytes=total_bytes,
        )
        return choice.pipeline_parallel, choice.expert_parallel
