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
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

from .arbiter import IncrementalArbiter
from .events import ComputeEvent, EventGenerator
from .parallelism import ParallelismPlanner
from .pdd import context_kv_bytes, kv_transfer_duration, split_work
from .placement import EnginePool, EngineSlot
from .kv_store import KVCacheManager, Match
from .shards import WorkShardGenerator
from .system import System
from .tokenizer import Tokenizer
from .tracker import SequenceWork
from .transfer import transfer_duration
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
        model_weight_loading: When ``True``, the first time a model is placed on
            an engine slot its weights are streamed from the system input NVM
            into each device's first-tier memory as a ``transfer`` event that the
            batch's compute waits on; a slot already holding the model reuses its
            resident weights (no reload). When ``False`` (default) weights are
            assumed pre-resident and no load is charged. Config-driven runs turn
            this on (see :func:`serve_sim.runner.run_from_config`).
        event_random_factor_range: Per-event time is multiplied by
            ``1 + U(-range, range)`` to model system randomness; ``0`` disables it.
        random_seed: Seed for the run's randomness (event-time perturbation);
            ``None`` draws a non-deterministic seed.
        global_kv_cache: When ``True`` (default) the orchestrator keeps a
            system-wide record of every non-evicted sequence's KV, reuses the
            longest message-aligned prefix across conversations, offloads
            completed KV to floating (node) memories as arbiter-accounted
            transfers, and evicts least-recently-used whole sequences when the
            floating pool is full. When ``False`` only the previous-turn reuse of
            the same conversation applies. Inert on systems with no node memory.
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
    model_weight_loading: bool = False
    event_random_factor_range: float = 0.0
    random_seed: int | None = None
    global_kv_cache: bool = True

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
        workload_id: Source conversation/workload identifier (``-1`` if the
            request was not built from a workload).
        turn_index: Turn within the workload that produced this request.
        tracker: The :class:`~serve_sim.tracker.SequenceTracker` this request was
            built from, carried so the global KV cache can compare message-aligned
            prefixes across conversations. ``None`` for requests built directly
            from token counts (no cross-sequence reuse is possible for those).
    """

    request_id: int
    model: object
    prompt_tokens: int
    output_tokens: int
    arrival_time: float = 0.0
    cached_tokens: int = 0
    workload_id: int = -1
    turn_index: int = 0
    tracker: object | None = None

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
        workload_id: int = -1,
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
            workload_id=workload_id,
            turn_index=turn_index,
            tracker=tracker,
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

    @property
    def tps(self) -> float | None:
        """Decode tokens-per-second (the reciprocal of ``tpot``)."""

        tpot = self.tpot
        if tpot is None or tpot <= 0:
            return None
        return 1.0 / tpot


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
    memory: str  # name of the memory whose bandwidth this event consumed ("" if none)
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
class DecisionRecord:
    """One high-level orchestration decision, for the decisions report.

    Covers the orchestration acts the PRD calls out -- ``prefill``, ``decode``,
    ``kv_reuse``, ``kv_transfer`` and ``kv_eviction`` -- each tagged with the
    device(s) it mapped to, the model, and the sequence it served (the workload
    id and turn number). KV reuse and KV transfer additionally name the second
    sequence and the device(s) it involves: for reuse, the earlier turn whose
    cached prefix is reused (in place, on the serving devices); for transfer, the
    same sequence's prefill devices the KV is moved from.

    Attributes:
        time: Simulation clock when the decision was taken.
        kind: One of ``prefill``/``decode``/``kv_reuse``/``kv_transfer``/
            ``kv_eviction``.
        request_id: The served request's run-unique id.
        workload_id: The served sequence's workload id (``-1`` if synthetic).
        turn_index: The served sequence's turn within its workload.
        model: Name of the model serving the request.
        devices: Compute device(s) the decision mapped to.
        batch_index: Dispatch batch the decision belongs to.
        tokens: Token count the act concerns (prefilled / decoded / reused /
            transferred context).
        source_request_id: The second sequence's request id, if any.
        source_workload_id: The second sequence's workload id, if any.
        source_turn_index: The second sequence's turn, if any.
        source_devices: The second sequence's device(s), if any.
        batch_members: For batched acts (prefill/decode), the ``(workload_id,
            turn_index)`` of every sequence sharing the batch, in dispatch order;
            empty for per-sequence acts (KV reuse/transfer).
    """

    time: float
    kind: str
    request_id: int
    workload_id: int
    turn_index: int
    model: str
    devices: tuple[str, ...]
    batch_index: int
    tokens: int = 0
    source_request_id: int | None = None
    source_workload_id: int | None = None
    source_turn_index: int | None = None
    source_devices: tuple[str, ...] = ()
    batch_members: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class MemoryRecord:
    """A memory device in the system, for the per-memory utilization report.

    Captured from the system topology (not from events) so the report can list
    every memory -- including idle ones -- with its static spec and the compute
    devices it serves, independent of the compute-device view.

    Attributes:
        name: Identifier for logs/reports.
        capacity_bytes: Total capacity.
        bandwidth_bytes_per_s: Intrinsic (unconstrained) bandwidth ceiling.
        role: ``"input"``, ``"node"``, ``"first_tier"`` or ``"second_tier"``.
        node: Owning node name, or ``""`` for the system-level input NVM.
        attached_devices: Compute devices that use this memory as a tier.
    """

    name: str
    capacity_bytes: float
    bandwidth_bytes_per_s: float
    role: str
    node: str
    attached_devices: tuple[str, ...]


@dataclass(frozen=True)
class RunProgress:
    """A progress update emitted as a run retires completed sequences.

    ``sim_time`` is the simulation clock at the moment of the update; ``wall_time``
    is the real time elapsed since the run started. ``avg_tps`` and ``avg_ttft``
    are the running mean decode tokens-per-second and time-to-first-token over the
    sequences retired so far (``None`` until at least one is measurable).
    """

    completed: int
    total: int
    sim_time: float
    wall_time: float
    avg_tps: float | None = None
    avg_ttft: float | None = None


# Called with a :class:`RunProgress` whenever the completed-sequence count grows.
ProgressCallback = Callable[["RunProgress"], None]


def _running_averages(records: Sequence["RequestRecord"]) -> tuple[float | None, float | None]:
    """Running mean decode TPS and TTFT over the retired records (skip ``None``)."""

    tpss = [r.tps for r in records if r.tps is not None]
    ttfts = [r.ttft for r in records if r.ttft is not None]
    avg_tps = sum(tpss) / len(tpss) if tpss else None
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else None
    return avg_tps, avg_ttft


def _first_token_end(events: Sequence[object]) -> float | None:
    """Completion time of the earliest decode group in a finished job's events."""

    decode = [e for e in events if getattr(e, "phase", None) == "decode"]
    if not decode:
        return None
    first_group = min(e.group_index for e in decode)
    return max(e.end for e in decode if e.group_index == first_group)


def _decision(
    time: float,
    kind: str,
    request: "Request",
    devices: Sequence[str],
    batch_index: int,
    *,
    tokens: int = 0,
    source_request_id: int | None = None,
    source_workload_id: int | None = None,
    source_turn_index: int | None = None,
    source_devices: Sequence[str] = (),
    batch_members: Sequence[tuple[int, int]] = (),
) -> "DecisionRecord":
    """Build a :class:`DecisionRecord` for ``request`` from the dispatch context."""

    return DecisionRecord(
        time=time,
        kind=kind,
        request_id=request.request_id,
        workload_id=request.workload_id,
        turn_index=request.turn_index,
        model=getattr(request.model, "name", "model"),
        devices=tuple(devices),
        batch_index=batch_index,
        tokens=tokens,
        source_request_id=source_request_id,
        source_workload_id=source_workload_id,
        source_turn_index=source_turn_index,
        source_devices=tuple(source_devices),
        batch_members=tuple(batch_members),
    )


def _turn_chains(
    requests: Sequence["Request"],
) -> tuple[list["Request"], dict[tuple[int, int], "Request"]]:
    """Split requests into initial arrivals and a per-workload follow-on map.

    A workload is a multi-turn conversation: turn ``t+1`` is a follow-up that
    cannot be submitted before turn ``t`` has completed. So only a workload's
    first (lowest-``turn_index``) turn arrives externally; every later turn is
    held back and released when its predecessor finishes. Requests with no
    workload (``workload_id < 0``) are independent and all arrive externally.

    Returns the list of externally-arriving requests and a map from a turn's
    ``(workload_id, turn_index)`` to the next turn of the same workload.
    """

    by_workload: dict[int, list["Request"]] = {}
    initial: list["Request"] = []
    for req in requests:
        if req.workload_id < 0:
            initial.append(req)
        else:
            by_workload.setdefault(req.workload_id, []).append(req)

    next_of: dict[tuple[int, int], "Request"] = {}
    for workload_id, turns in by_workload.items():
        turns.sort(key=lambda r: r.turn_index)
        initial.append(turns[0])
        for cur, nxt in zip(turns, turns[1:]):
            next_of[(workload_id, cur.turn_index)] = nxt
    return initial, next_of


@dataclass
class RunResult:
    """Minimal run state: per-request records and the overall makespan.

    Beyond the per-request timings, a run also captures the raw event log
    (``events``, each event recorded before and after rescaling), per-job
    placement/footprint records (``jobs``) and the system's memory-device
    inventory (``memories``) so the outputs layer can derive utilization and
    memory-occupancy reports for both compute and memory devices.
    """

    records: list[RequestRecord] = field(default_factory=list)
    num_batches: int = 0
    events: list[EventRecord] = field(default_factory=list)
    jobs: list[JobRecord] = field(default_factory=list)
    memories: list[MemoryRecord] = field(default_factory=list)
    decisions: list[DecisionRecord] = field(default_factory=list)

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
        # System-wide persistent KV cache (cross-conversation prefix reuse, LRU
        # eviction, migration across floating memories). Inert when disabled or
        # when the system has no floating (node) memory to offload KV into.
        self._kv = KVCacheManager(system) if self.strategy.global_kv_cache else None

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
        initial, next_of = _turn_chains(requests)
        arrivals = sorted(initial, key=lambda r: r.arrival_time)
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
                avg_tps, avg_ttft = _running_averages(result.records)
                progress(RunProgress(len(result.records), total, now,
                                     time.perf_counter() - start_wall,
                                     avg_tps, avg_ttft))

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
                first_token_time = _first_token_end(
                    arbiter.job_rescaled_events(job_index)
                )
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
                            first_token_time=first_token_time,
                        )
                    )
                    # Offload this turn's KV to floating memory so a later
                    # sequence can reuse its prefix (evicting LRU if the floating
                    # pool is full).
                    if self._kv_active:
                        self._store_completed_kv(
                            arbiter, req, slot, end, result, b_index
                        )
                    # Release this workload's next turn now that this one is
                    # done (it arrives the instant its predecessor completes).
                    follow_on = next_of.get((req.workload_id, req.turn_index))
                    if follow_on is not None:
                        ready.append(replace(follow_on, arrival_time=end))
                        if window_open is None:
                            window_open = now
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
                matches: dict[int, Match] = {}
                kv_fetches: list[tuple[object, float]] | None = None
                if self._kv_active:
                    batch, matches = self._resolve_kv(batch, now)
                    kv_fetches = self._kv_fetches(batch, matches)
                job_index, slot, pp, ep = self._dispatch(
                    arbiter, batch, kv_fetches=kv_fetches)
                device_names = tuple(d.name for d in slot.devices)
                members = tuple((r.workload_id, r.turn_index) for r in batch)
                for req in batch:
                    result.decisions.append(_decision(
                        now, "prefill", req, device_names, batch_index,
                        tokens=req.prompt_tokens - req.cached_tokens,
                        batch_members=members))
                    match = matches.get(id(req))
                    if match is not None:
                        self._emit_kv_reuse(
                            result, req, match, device_names, batch_index, now)
                    elif req.cached_tokens > 0:
                        result.decisions.append(_decision(
                            now, "kv_reuse", req, device_names, batch_index,
                            tokens=req.cached_tokens,
                            source_workload_id=req.workload_id,
                            source_turn_index=req.turn_index - 1,
                            source_devices=device_names))
                    result.decisions.append(_decision(
                        now, "decode", req, device_names, batch_index,
                        tokens=req.output_tokens,
                        batch_members=members))
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
        initial, next_of = _turn_chains(requests)
        arrivals = sorted(initial, key=lambda r: r.arrival_time)
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
        # id(request) -> (prefill dispatch time, prefill batch index, prefill devices)
        meta: dict[int, tuple[float, int, tuple[str, ...]]] = {}
        result = RunResult()

        prefill_rep = self._prefill_pool.slots[0].devices[0]
        decode_rep = self._decode_pool.slots[0].devices[0]

        def report(now: float) -> None:
            if progress is not None:
                avg_tps, avg_ttft = _running_averages(result.records)
                progress(RunProgress(len(result.records), total, now,
                                     time.perf_counter() - start_wall,
                                     avg_tps, avg_ttft))

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
                    first_token_time = _first_token_end(
                        arbiter.job_rescaled_events(job_index)
                    )
                    for req in reqs:
                        dispatch_time, b_index, _ = meta[id(req)]
                        result.records.append(
                            RequestRecord(
                                request_id=req.request_id,
                                arrival_time=req.arrival_time,
                                dispatch_time=dispatch_time,
                                completion_time=end,
                                prompt_tokens=req.prompt_tokens,
                                output_tokens=req.output_tokens,
                                batch_index=b_index,
                                first_token_time=first_token_time,
                            )
                        )
                        # Offload this turn's KV to floating memory for later
                        # cross-conversation prefix reuse (evicting LRU if full).
                        if self._kv_active:
                            self._store_completed_kv(
                                arbiter, req, slot, end, result, b_index
                            )
                        # Release this workload's next turn now that this one
                        # is done; it enters the prefill queue for its turn.
                        follow_on = next_of.get((req.workload_id, req.turn_index))
                        if follow_on is not None:
                            prefill_ready.append(replace(follow_on, arrival_time=end))
                            if prefill_window is None:
                                prefill_window = now
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
                matches: dict[int, Match] = {}
                kv_fetches: list[tuple[object, float]] | None = None
                if self._kv_active:
                    batch, matches = self._resolve_kv(batch, now)
                    kv_fetches = self._kv_fetches(batch, matches)
                job_index, slot, pp, ep = self._dispatch(
                    arbiter, batch, pool=self._prefill_pool, phase="prefill",
                    kv_fetches=kv_fetches,
                )
                device_names = tuple(d.name for d in slot.devices)
                members = tuple((r.workload_id, r.turn_index) for r in batch)
                for req in batch:
                    meta[id(req)] = (now, batch_index, device_names)
                    result.decisions.append(_decision(
                        now, "prefill", req, device_names, batch_index,
                        tokens=req.prompt_tokens - req.cached_tokens,
                        batch_members=members))
                    match = matches.get(id(req))
                    if match is not None:
                        self._emit_kv_reuse(
                            result, req, match, device_names, batch_index, now)
                    elif req.cached_tokens > 0:
                        result.decisions.append(_decision(
                            now, "kv_reuse", req, device_names, batch_index,
                            tokens=req.cached_tokens,
                            source_workload_id=req.workload_id,
                            source_turn_index=req.turn_index - 1,
                            source_devices=device_names))
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
                device_names = tuple(d.name for d in slot.devices)
                members = tuple((r.workload_id, r.turn_index) for r in batch)
                for req in batch:
                    _, _, prefill_devices = meta[id(req)]
                    result.decisions.append(_decision(
                        now, "kv_transfer", req, device_names, batch_index,
                        tokens=req.prompt_tokens,
                        source_request_id=req.request_id,
                        source_workload_id=req.workload_id,
                        source_turn_index=req.turn_index,
                        source_devices=prefill_devices))
                    result.decisions.append(_decision(
                        now, "decode", req, device_names, batch_index,
                        tokens=req.output_tokens,
                        batch_members=members))
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
                        mem_name = self._event_memory_name(devices[ev.device_index], ev)
                    else:
                        dev_name = ""
                        mem_name = ""
                    result.events.append(
                        EventRecord(
                            job_index=job_index,
                            batch_index=b_index,
                            job_phase=job_phase,
                            request_ids=request_ids,
                            group_index=ev.group_index,
                            phase=ev.phase,
                            device=dev_name,
                            memory=mem_name,
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
        result.memories = [
            MemoryRecord(
                name=entry["name"],
                capacity_bytes=entry["capacity_bytes"],
                bandwidth_bytes_per_s=entry["bandwidth_bytes_per_s"],
                role=entry["role"],
                node=entry["node"],
                attached_devices=entry["attached_devices"],
            )
            for entry in self.system.memory_inventory()
        ]

    @staticmethod
    def _event_memory_name(device: ComputeDevice, event: ComputeEvent) -> str:
        """Name of the memory whose bandwidth ``event`` consumed on ``device``.

        Mirrors the arbiter's bandwidth attribution: a transfer streams from its
        explicit ``source_memory`` when set (e.g. a weight load from the input
        NVM) else the device's second tier (falling back to the first); compute
        reads the device's first-tier memory; a kernel launch touches no memory.
        """

        if event.phase == "kernel_launch":
            return ""
        if event.phase == "transfer":
            if event.source_memory is not None:
                return event.source_memory.name
            memory = device.second_tier_memory or device.first_tier_memory
            return memory.name
        return device.first_tier_memory.name

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
        kv_fetches: list[tuple[object, float]] | None = None,
    ) -> tuple[int, EngineSlot, int, int]:
        """Place a batch on an engine slot and admit its events to the arbiter.

        ``phase`` selects the work generated: ``"full"`` (prefill + decode in one
        job, the default), ``"prefill"`` (prompt forward pass only) or
        ``"decode"`` (generation from a fully-cached prompt). ``pool`` defaults to
        the single engine pool; PDD passes the prefill or decode pool.
        ``kv_fetches`` lists ``(floating_memory, bytes)`` prefixes to fetch from
        the global KV cache before compute. Returns the arbiter job index, the
        slot the batch occupies (released on completion), and the
        ``(pipeline_parallel, expert_parallel)`` arrangement chosen.
        """

        model = batch[0].model
        pool = pool or self._pool
        placement = pool.place(model)
        slot = placement.slot
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
        loads: list[ComputeEvent] = []
        if self.strategy.model_weight_loading and placement.needs_weight_load:
            loads = self._weight_load_events(model, slot, pp, ep)
        fetches = self._kv_fetch_events(slot, kv_fetches) if kv_fetches else []
        prelude = loads + fetches
        if prelude:
            events = self._prepend_transfers(prelude, generator.run(shards).events)
            return arbiter.admit_events(events, list(slot.devices)), slot, pp, ep
        return arbiter.admit(generator, shards), slot, pp, ep

    @staticmethod
    def _prepend_transfers(
        prelude: list[ComputeEvent], compute_events: list[ComputeEvent]
    ) -> list[ComputeEvent]:
        """Prepend transfer events (weight loads / KV fetches) ahead of compute.

        The compute events are shifted to start once every prelude transfer has
        finished, so the batch waits for its weights and reused prefixes; the
        arbiter then contends the transfers' bytes on their shared memories.
        """

        if not prelude:
            return list(compute_events)
        offset = max(event.end for event in prelude)
        shifted = [
            replace(event, start=event.start + offset, end=event.end + offset)
            for event in compute_events
        ]
        return prelude + shifted

    def _weight_load_events(
        self, model: object, slot: EngineSlot, pp: int, ep: int
    ) -> list[ComputeEvent]:
        """One ``transfer`` event per device: input NVM -> first-tier weights.

        The link latency is folded into ``bandwidth_time`` (as the expert-movement
        transfers do) so the arbiter reproduces the load's duration exactly while
        still contending its bytes on the shared input-NVM bandwidth.
        """

        weight_bytes = float(self._planner_for(model).footprint(pp, ep, 0))
        if weight_bytes <= 0:
            return []
        source = self.system.input_memory
        events: list[ComputeEvent] = []
        for index, device in enumerate(slot.devices):
            destination = device.first_tier_memory
            link = self.system.link_between(source, destination)
            duration = transfer_duration(weight_bytes, source, destination, link)
            events.append(
                ComputeEvent(
                    group_index=-1,
                    phase="transfer",
                    device_index=index,
                    flops=0.0,
                    bytes_read=weight_bytes,
                    compute_time=0.0,
                    bandwidth_time=duration,
                    duration=duration,
                    start=0.0,
                    end=duration,
                    source_memory=source,
                )
            )
        return events

    # --- global KV cache ----------------------------------------------------

    @property
    def _kv_active(self) -> bool:
        """Whether the system-wide KV cache is on and has somewhere to store KV."""

        return self._kv is not None and self._kv.enabled

    def _resolve_kv(
        self, batch: list[Request], now: float
    ) -> tuple[list[Request], dict[int, Match]]:
        """Set each request's cached prefix from the global KV cache.

        Replaces every request's cached-token count with the longest message
        aligned prefix found among non-evicted entries of the same model (zero
        when nothing matches -- the previous turn, if still resident, is itself an
        entry, so a miss means it was evicted or no prefix is shared). Returns the
        replaced batch and a map from each replaced request's ``id`` to its reuse
        match, for decision emission and the prefix fetch.
        """

        resolved: list[Request] = []
        matches: dict[int, Match] = {}
        for req in batch:
            if req.tracker is None:
                # No message structure to prefix-match: leave the request's own
                # cached-token count (e.g. a directly-built request) untouched.
                resolved.append(req)
                continue
            match = self._kv.lookup(req.model, req.tracker, now)
            cached = min(match.prefix_tokens, req.prompt_tokens) if match else 0
            req = replace(req, cached_tokens=cached)
            if match is not None and cached > 0:
                matches[id(req)] = match
            resolved.append(req)
        return resolved, matches

    def _kv_fetches(
        self, batch: list[Request], matches: dict[int, Match]
    ) -> list[tuple[object, float]]:
        """``(floating_memory, bytes)`` to fetch for each reused prefix in a batch."""

        fetches: list[tuple[object, float]] = []
        for req in batch:
            match = matches.get(id(req))
            if match is not None and req.cached_tokens > 0:
                num_bytes = float(context_kv_bytes(req.model, req.cached_tokens))
                fetches.append((match.entry.memory, num_bytes))
        return fetches

    def _kv_fetch_events(
        self, slot: EngineSlot, fetches: list[tuple[object, float]]
    ) -> list[ComputeEvent]:
        """Transfer events that fetch reused prefixes from floating memory.

        Each ``(floating_memory, bytes)`` becomes one ``transfer`` event reading
        the cached prefix into a slot device's first-tier memory (round-robined
        across the slot so several fetches parallelise). The link latency is
        folded into ``bandwidth_time`` so the arbiter reproduces the duration
        while contending the floating memory's bandwidth against all other
        in-flight transfers; the batch's compute waits on the fetch.
        """

        events: list[ComputeEvent] = []
        num_devices = len(slot.devices)
        for index, (source, num_bytes) in enumerate(fetches):
            slot_index = index % num_devices
            destination = slot.devices[slot_index].first_tier_memory
            link = self.system.link_between(source, destination)
            duration = transfer_duration(num_bytes, source, destination, link)
            events.append(
                ComputeEvent(
                    group_index=-1,
                    phase="transfer",
                    device_index=slot_index,
                    flops=0.0,
                    bytes_read=num_bytes,
                    compute_time=0.0,
                    bandwidth_time=duration,
                    duration=duration,
                    start=0.0,
                    end=duration,
                    source_memory=source,
                )
            )
        return events

    def _emit_kv_reuse(
        self,
        result: RunResult,
        req: Request,
        match: Match,
        device_names: tuple[str, ...],
        batch_index: int,
        now: float,
    ) -> None:
        """Record the ``kv_reuse`` + ``kv_transfer`` decisions for a prefix hit.

        The reuse names the source sequence (its conversation/turn) and the
        floating memory holding the prefix; the transfer is the physical fetch of
        that prefix into the serving devices.
        """

        source_memory = (match.entry.memory.name,)
        result.decisions.append(_decision(
            now, "kv_reuse", req, source_memory, batch_index,
            tokens=req.cached_tokens,
            source_workload_id=match.entry.workload_id,
            source_turn_index=match.entry.turn_index,
            source_devices=device_names))
        result.decisions.append(_decision(
            now, "kv_transfer", req, device_names, batch_index,
            tokens=req.cached_tokens,
            source_workload_id=match.entry.workload_id,
            source_turn_index=match.entry.turn_index,
            source_devices=source_memory))

    def _store_completed_kv(
        self,
        arbiter: IncrementalArbiter,
        req: Request,
        slot: EngineSlot,
        now: float,
        result: RunResult,
        batch_index: int,
    ) -> None:
        """Offload a finished sequence's KV to floating memory; record decisions.

        Stores the full context (prompt + generated) KV in the global cache,
        evicting LRU entries if the floating pool is full, and admits an
        arbiter-accounted ``transfer`` for the device->floating move (so its
        bandwidth contends with concurrent work). Emits a ``kv_eviction`` decision
        per evicted entry and a ``kv_transfer`` decision for the offload.
        """

        context_tokens = req.prompt_tokens + req.output_tokens
        store = self._kv.store(
            req.model, req.tracker, req.workload_id, req.turn_index,
            context_tokens, now,
        )
        for victim in store.evicted:
            result.decisions.append(self._eviction_decision(now, victim))
        if store.memory is None:
            return
        device_names = tuple(d.name for d in slot.devices)
        self._admit_kv_move(arbiter, slot.devices[0], store.memory, store.num_bytes)
        result.decisions.append(_decision(
            now, "kv_transfer", req, (store.memory.name,), batch_index,
            tokens=context_tokens, source_request_id=req.request_id,
            source_workload_id=req.workload_id, source_turn_index=req.turn_index,
            source_devices=device_names))

    def _admit_kv_move(
        self,
        arbiter: IncrementalArbiter,
        device,
        floating,
        num_bytes: float,
    ) -> None:
        """Admit a standalone, arbiter-accounted KV offload transfer.

        Moves ``num_bytes`` of KV from ``device``'s first-tier memory to the
        ``floating`` memory. The transfer contends the floating memory's bandwidth
        (the binding resource for offload), so a slow floating tier surfaces as
        contention on concurrent work; it holds no engine slot and does not delay
        the served request, which has already retired.
        """

        if num_bytes <= 0:
            return
        link = self.system.link_between(device.first_tier_memory, floating)
        duration = transfer_duration(
            num_bytes, device.first_tier_memory, floating, link
        )
        event = ComputeEvent(
            group_index=-1,
            phase="transfer",
            device_index=0,
            flops=0.0,
            bytes_read=num_bytes,
            compute_time=0.0,
            bandwidth_time=duration,
            duration=duration,
            start=0.0,
            end=duration,
            source_memory=floating,
        )
        arbiter.admit_events([event], [device])

    @staticmethod
    def _eviction_decision(now: float, entry) -> DecisionRecord:
        """A ``kv_eviction`` decision for an LRU-evicted stored sequence."""

        return DecisionRecord(
            time=now,
            kind="kv_eviction",
            request_id=-1,
            workload_id=entry.workload_id,
            turn_index=entry.turn_index,
            model=getattr(entry.model, "name", "model"),
            devices=(entry.memory.name,),
            batch_index=-1,
            tokens=entry.context_tokens,
        )

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
        fastest memory-feasible arrangement of this batch. Either way the chosen
        arrangement must fit in device memory; a batch that cannot be placed
        raises rather than producing a physically impossible (over-capacity)
        schedule.
        """

        planner = self._planner_for(model)
        kv_tokens = sum(req.prompt_tokens + req.output_tokens for req in batch)

        if not self.strategy.auto_parallelism:
            pp = self.strategy.pipeline_parallel
            ep = self.strategy.expert_parallel
            per_device = planner.footprint(pp, ep, kv_tokens)
            if per_device > planner.capacity:
                raise ValueError(
                    f"fixed parallelism pp={pp}, ep={ep} cannot serve a batch of "
                    f"{kv_tokens} KV tokens: the per-device footprint is "
                    f"{per_device:.0f} bytes but device memory holds only "
                    f"{planner.capacity:.0f} bytes. Raise pipeline_parallel/"
                    f"expert_parallel, enable auto_parallelism, or use a device "
                    f"with more memory."
                )
            return pp, ep

        flops_by_dtype: dict[int, float] = {}
        total_bytes = 0.0
        for shard in shards:
            flops_by_dtype[shard.flops_dtype_bytes] = (
                flops_by_dtype.get(shard.flops_dtype_bytes, 0.0) + shard.flops
            )
            total_bytes += shard.bytes_read
        choice = planner.plan(
            self._degree,
            kv_tokens=kv_tokens,
            flops_by_dtype=flops_by_dtype,
            total_bytes=total_bytes,
        )
        return choice.pipeline_parallel, choice.expert_parallel
