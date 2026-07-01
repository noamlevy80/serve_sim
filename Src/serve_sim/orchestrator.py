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
  expert_parallel x tensor_parallel`` compute devices of the system; the
  roofline parallelism search comes in a later stage.
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
from .device_memory import DeviceHbmResidency, MemoryPolicy
from .events import ComputeEvent, EventGenerator
from .hardware import ComputeDevice
from .parallelism import ParallelismPlanner
from .pdd import context_kv_bytes, split_work
from .placement import EnginePool, EngineSlot
from .kv_store import KVCacheManager, Match
from .shards import WorkShardGenerator
from .system import System
from .tiering import build_activation_trace, peak_active_per_rank
from .tokenizer import Tokenizer
from .tracker import SequenceWork
from .transfer import transfer_duration
from .workload import Workload


class MemoryCapacityExceeded(RuntimeError):
    """Raised when a device's reserved memory footprint exceeds its capacity.

    The simulator pins each job's weights + KV footprint on its serving devices'
    memory. If the concurrent footprint on any device ever exceeds the memory
    available to it (its first tier, plus any second tier it can spill into), the
    configuration is physically infeasible: the run is aborted rather than
    silently reporting an impossible, oversubscribed occupancy.
    """

    def __init__(self, device: str, peak_bytes: float, capacity_bytes: float) -> None:
        self.device = device
        self.peak_bytes = peak_bytes
        self.capacity_bytes = capacity_bytes
        over = peak_bytes / capacity_bytes if capacity_bytes else float("inf")
        super().__init__(
            f"out of memory on device {device!r}: peak reserved footprint "
            f"{peak_bytes / 1e9:.2f} GB exceeds its {capacity_bytes / 1e9:.2f} GB "
            f"capacity ({over:.2f}x oversubscribed). The concurrent weights + KV "
            f"of the jobs placed on this device do not fit in its memory. Reduce "
            f"max_concurrency or max_batch_size, raise the parallelism degree "
            f"(or enable auto_parallelism) to shard the footprint across more "
            f"devices, give the device a second memory tier to spill into, or use "
            f"a smaller model."
        )


@dataclass(frozen=True)
class ParallelismSection:
    """A parallelism scheme scoped to one compute-device *type*.

    A heterogeneous system mixes very different accelerators, and the best way to
    wire an engine group out of one device type rarely suits another. A strategy
    may therefore carry a list of these sections (see
    ``StrategyConfig.parallelism``); the orchestrator partitions the system's
    compute devices by type and builds each type's engine groups with its own
    section. When no sections are given the flat ``StrategyConfig`` fields act as
    a single section spanning every device.

    Attributes:
        compute_device: The compute-device config key this section configures
            (the ``Compute_devices/<key>.json`` stem, e.g. ``"nvidia-b200"``),
            matched against each device's :attr:`~serve_sim.hardware.ComputeDevice.device_key`.
        pipeline_parallel: Fixed pipeline-parallel degree for this device type.
        expert_parallel: Fixed expert-parallel degree for this device type.
        tensor_parallel: Fixed tensor-parallel degree for this device type.
        auto_parallelism: When ``True``, the ``pipeline_parallel x expert_parallel``
            product is treated as a fixed engine size and re-factored per batch
            (``tensor_parallel`` held fixed); see ``StrategyConfig.auto_parallelism``.
        max_parallelism: Optional cap on the parallelism rank to explore (accepted
            for config parity; not yet enforced by the search).
    """

    compute_device: str
    pipeline_parallel: int = 1
    expert_parallel: int = 1
    tensor_parallel: int = 1
    auto_parallelism: bool = False
    max_parallelism: int | None = None

    def __post_init__(self) -> None:
        if not self.compute_device:
            raise ValueError("a parallelism section needs a compute_device key")
        if self.pipeline_parallel < 1 or self.expert_parallel < 1:
            raise ValueError("parallelism degrees must be >= 1")
        if self.tensor_parallel < 1:
            raise ValueError("parallelism degrees must be >= 1")

    @property
    def degree(self) -> int:
        """Devices per engine slot for this section (``pp x ep x tp``)."""

        return self.pipeline_parallel * self.expert_parallel * self.tensor_parallel


@dataclass(frozen=True)
class StrategyConfig:
    """The orchestration knobs (v0 subset).

    Attributes:
        max_batch_size: Sequences per dispatched batch -- the fundamental
            inference knob that sets how wide a single engine slot batches work
            (the window fill threshold).
        max_window_duration: Seconds a window stays open before dispatching
            whatever it has collected (``0`` dispatches as soon as work is ready).
        max_concurrency: High-level orchestration cap on the *total* sequences in
            flight across all batches/slots at once, or ``None`` for unbounded.
            With several engine slots available, lowering ``max_batch_size`` below
            ``max_concurrency`` lets the remaining budget spill into additional
            concurrent batches on other slots.
        pipeline_parallel: Fixed pipeline-parallel degree of the engine.
        expert_parallel: Fixed expert-parallel degree of the engine.
        tensor_parallel: Fixed tensor-parallel degree of the engine. Tensor
            parallelism shards every weight tensor and the KV cache across its
            ranks and splits each rank's compute, so it divides the per-device
            footprint by ``tensor_parallel`` and speeds a batch up by the same
            factor. It is always applied verbatim; ``auto_parallelism`` only
            re-factors the ``pipeline_parallel x expert_parallel`` budget while
            ``tensor_parallel`` is held fixed. The engine occupies
            ``pipeline_parallel x expert_parallel x tensor_parallel`` devices.
        auto_parallelism: When ``True``, the orchestrator treats
            ``pipeline_parallel x expert_parallel`` only as the fixed engine size
            and, per batch, searches the ``(pp, ep)`` factorizations of that size
            for the fastest one that fits device memory. When ``False`` (default)
            the fixed ``pp``/``ep`` are used verbatim.
        parallelism: Optional per-compute-device parallelism schemes (one
            :class:`ParallelismSection` per device *type*). When non-empty the
            orchestrator partitions the system's compute devices by type and
            builds each type's engine groups from its matching section instead of
            the flat ``pipeline_parallel``/``expert_parallel``/``tensor_parallel``/
            ``auto_parallelism`` fields above -- so a heterogeneous system can wire
            each accelerator the way that suits it. When empty (default) the flat
            fields act as a single section spanning every device (the legacy
            homogeneous behaviour).
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
        warm_start: When ``True``, before the run the orchestrator examines the
            suite's models and eagerly pre-loads them onto the engine slots (in
            proportion to demand, round-robin across slots), so the first batch of
            a pre-warmed model finds its weights already resident and skips the
            trivial cold ``weight_load`` -- valuable simulation time is not spent
            on start-up weight transfers. Only meaningful together with
            ``model_weight_loading``; mid-run model offload/switch still loads
            weights by orchestration decision. When ``False`` (default) every slot
            starts empty and the first placement of each model cold loads.
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
        memory_policy: How a completed sequence's KV is retained and where reuse
            fetches it from. ``"node_first"`` (default, legacy) offloads every
            completed KV straight to node memory and re-fetches it from there on
            reuse. The device-first policies instead keep KV resident on the
            serving device's first-tier HBM, so a same-device reuse costs no
            transfer and a cross-device reuse moves the KV device-to-device; node
            memory is used only as a spill tier when HBM is full. ``"global_lru"``
            lets retained KV and resident routed experts compete in one LRU over
            the device's dynamic HBM; ``"partitioned"`` reserves a KV sub-region
            and an expert sub-region (see ``hbm_kv_fraction``), each LRU-managed
            in isolation.
        hbm_kv_fraction: Under ``memory_policy="partitioned"``, the fraction of a
            device's dynamic HBM region reserved for retained KV (the remainder is
            the routed-expert region). Ignored by the other policies.
    """

    max_batch_size: int = 1
    max_window_duration: float = 0.0
    max_concurrency: int | None = None
    pipeline_parallel: int = 1
    expert_parallel: int = 1
    tensor_parallel: int = 1
    auto_parallelism: bool = False
    parallelism: tuple[ParallelismSection, ...] = ()
    allow_pdd: bool = False
    prefill_engine_fraction: float = 0.5
    prefill_chunk_size: int | None = None
    model_weight_loading: bool = False
    warm_start: bool = False
    event_random_factor_range: float = 0.0
    random_seed: int | None = None
    global_kv_cache: bool = True
    memory_policy: str = "node_first"
    hbm_kv_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.max_window_duration < 0:
            raise ValueError("max_window_duration must be non-negative")
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1 when set")
        if self.pipeline_parallel < 1 or self.expert_parallel < 1:
            raise ValueError("parallelism degrees must be >= 1")
        if self.tensor_parallel < 1:
            raise ValueError("parallelism degrees must be >= 1")
        if not isinstance(self.parallelism, tuple):
            object.__setattr__(self, "parallelism", tuple(self.parallelism))
        keys = [s.compute_device for s in self.parallelism]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "each compute_device may appear at most once in parallelism sections"
            )
        if self.prefill_chunk_size is not None and self.prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be >= 1")
        if not 0.0 < self.prefill_engine_fraction < 1.0:
            raise ValueError("prefill_engine_fraction must be in (0, 1)")
        if not 0.0 <= self.event_random_factor_range < 1.0:
            raise ValueError("event_random_factor_range must be in [0, 1)")
        if self.memory_policy not in ("node_first", "global_lru", "partitioned"):
            raise ValueError(
                "memory_policy must be 'node_first', 'global_lru' or 'partitioned'"
            )
        if not 0.0 < self.hbm_kv_fraction < 1.0:
            raise ValueError("hbm_kv_fraction must be in (0, 1)")


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
    workload_id: int = -1
    turn_index: int = 0

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
    model: str  # name of the model this event serves ("" if none)
    flops: float
    bytes_read: float
    compute_time: float
    bandwidth_time: float
    duration: float
    start: float
    end: float
    rescaled: bool
    destination_memory: str = ""  # name of the memory written into ("" if none)


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
    weight_bytes_per_device: float = 0.0  # weight portion of the reservation
    kv_bytes_per_device: float = 0.0  # KV portion of the reservation
    model: str = ""  # name of the model the job serves


@dataclass(frozen=True)
class DecisionRecord:
    """One high-level orchestration decision, for the decisions report.

    Covers the orchestration acts the PRD calls out -- ``weight_load``,
    ``weight_eviction``, ``prefill``, ``decode``, ``kv_reuse``, ``kv_transfer``
    and ``kv_eviction`` -- each tagged with the device(s) it mapped to, the
    model, and the sequence it served (the workload id and turn number). KV reuse
    and KV transfer additionally name the second sequence and the device(s) it
    involves: for reuse, the earlier turn whose cached prefix is reused (in place,
    on the serving devices); for transfer, the same sequence's prefill devices
    the KV is moved from.

    Attributes:
        time: Simulation clock when the decision was taken.
        kind: One of ``weight_load``/``weight_eviction``/``prefill``/``decode``/
            ``kv_reuse``/``kv_transfer``/``kv_eviction``.
        request_id: The served request's run-unique id.
        workload_id: The served sequence's workload id (``-1`` if synthetic).
        turn_index: The served sequence's turn within its workload.
        model: Name of the model serving the request.
        devices: Compute device(s) the decision mapped to.
        batch_index: Dispatch batch the decision belongs to.
        time_started: Simulation clock when the decision's execution began (the
            start of the corresponding compute/transfer events after arbiter
            rescaling), or the decision time for purely bookkeeping acts
            (evictions). Backfilled in :meth:`Simulator._collect_outputs`.
        time_completed: Simulation clock when the decision's execution finished
            (the end of those events), or the decision time for bookkeeping acts.
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
    time_started: float | None = None
    time_completed: float | None = None
    tokens: int = 0
    source_request_id: int | None = None
    source_workload_id: int | None = None
    source_turn_index: int | None = None
    source_devices: tuple[str, ...] = ()
    batch_members: tuple[tuple[int, int], ...] = ()
    bytes_moved: float = 0.0


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
class DeviceRecord:
    """A compute device's static specs, for the per-device utilization report.

    The compute-side companion to :class:`MemoryRecord`: captured from the system
    topology so reports can render absolute compute/bandwidth/capacity values and
    their ``max`` reference lines without re-deriving the hardware.

    Attributes:
        name: Compute device identifier.
        node: Owning node name.
        peak_flops_fp16: Nominal FP16 FLOP/s ceiling.
        first_tier_memory: Name of the first-tier memory.
        first_tier_capacity_bytes: First-tier memory capacity.
        first_tier_bandwidth_bytes_per_s: First-tier memory bandwidth ceiling.
    """

    name: str
    node: str
    peak_flops_fp16: float
    first_tier_memory: str
    first_tier_capacity_bytes: float
    first_tier_bandwidth_bytes_per_s: float



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


@dataclass(frozen=True)
class RunEvent:
    """A single milestone in the run log: a sequence arriving in the queue, a
    batch being issued to an engine group, or a sequence completing.

    Every event carries the simulation clock (``sim_time``) and the real time
    elapsed since the run started (``wall_time``), plus the running done/total
    sequence count. The remaining fields are populated per ``kind``:

    - ``"arrival"``: ``sequence`` (the ``w<workload>t<turn>`` id), ``prompt_tokens``
      and ``output_tokens`` of the arriving sequence.
    - ``"issue"``: ``batch_index`` (the batch id), ``members`` (the sequence ids
      in the batch), ``engine_group`` (first device + rank count, e.g. ``g0x4``)
      and ``phase`` (``"prefill"``/``"decode"`` under PDD, else empty).
    - ``"completion"``: ``sequence`` plus the retired sequence's ``queue_delay``,
      ``ttft`` and decode ``tps``.
    """

    kind: str
    sim_time: float
    wall_time: float
    completed: int
    total: int
    sequence: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    batch_index: int = -1
    members: tuple[str, ...] = ()
    engine_group: str = ""
    phase: str = ""
    queue_delay: float | None = None
    ttft: float | None = None
    tps: float | None = None


# Called with a :class:`RunEvent` at each arrival/issue/completion milestone.
RunEventCallback = Callable[["RunEvent"], None]


def _seq_label(workload_id: int | None, turn_index: int | None, request_id: int) -> str:
    """Sequence id as ``w<workload>t<turn>``, or ``r<request_id>`` if standalone."""

    if workload_id is not None and turn_index is not None and workload_id >= 0:
        return f"w{workload_id}t{turn_index}"
    return f"r{request_id}"


def _engine_group_label(device_names: Sequence[str]) -> str:
    """Engine-group label as first device + rank count (e.g. ``g0x4``)."""

    if not device_names:
        return ""
    if len(device_names) == 1:
        return device_names[0]
    return f"{device_names[0]}x{len(device_names)}"


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
    bytes_moved: float = 0.0,
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
        bytes_moved=bytes_moved,
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
    device_specs: list[DeviceRecord] = field(default_factory=list)

    @property
    def makespan(self) -> float:
        """Wall-clock end of the last request (0 if none)."""

        return max((r.completion_time for r in self.records), default=0.0)

    def record_for(self, request_id: int) -> RequestRecord:
        for record in self.records:
            if record.request_id == request_id:
                return record
        raise KeyError(f"no record for request {request_id}")


@dataclass
class EngineGroup:
    """One device type's engine slots and the parallelism scheme wiring them.

    A heterogeneous system is partitioned into one group per compute-device type
    (or a single group spanning every device in the legacy homogeneous case). The
    group owns an :class:`~serve_sim.placement.EnginePool` carved from *just* that
    type's devices and the ``(pp, ep, tp)`` arrangement used to run batches on it,
    so batches on a B200 group and a Cerebras group are each wired the way their
    silicon prefers. ``prefill_pool``/``decode_pool`` are populated only when PDD
    splits the group's devices into disjoint phase pools.

    Attributes:
        device_key: The compute-device config key the group serves (``""`` for the
            legacy single group spanning every device type).
        pool: Engine pool over this group's devices (used by the non-PDD loop).
        pipeline_parallel: Fixed pipeline-parallel degree for the group.
        expert_parallel: Fixed expert-parallel degree for the group.
        tensor_parallel: Fixed tensor-parallel degree for the group.
        auto_parallelism: Whether the per-batch factorization search is enabled.
        prefill_pool: PDD prefill sub-pool, or ``None`` when PDD is off.
        decode_pool: PDD decode sub-pool, or ``None`` when PDD is off.
    """

    device_key: str
    pool: EnginePool
    pipeline_parallel: int
    expert_parallel: int
    tensor_parallel: int
    auto_parallelism: bool
    prefill_pool: EnginePool | None = None
    decode_pool: EnginePool | None = None

    @property
    def degree(self) -> int:
        """Devices per engine slot for this group (``pp x ep x tp``)."""

        return self.pipeline_parallel * self.expert_parallel * self.tensor_parallel

    @property
    def device(self) -> ComputeDevice:
        """A representative device of this group's type (for the planner)."""

        return self.pool.slots[0].devices[0]


class Simulator:
    """Runs a set of requests through a system under a fixed strategy."""

    def __init__(self, system: System, strategy: StrategyConfig | None = None) -> None:
        self.system = system
        self.strategy = strategy or StrategyConfig()
        # Carve the system into engine groups -- one per compute-device type when
        # the strategy carries per-device parallelism sections, otherwise a single
        # group spanning every device wired by the flat parallelism fields (the
        # legacy homogeneous behaviour). Each group owns an engine pool over just
        # its devices and the (pp, ep, tp) scheme that suits that silicon.
        self._groups: list[EngineGroup] = self._build_groups()
        # Prefill/decode disaggregation operates within a single engine group for
        # now (a heterogeneous PDD split is future work); ``_prefill_pool`` /
        # ``_decode_pool`` mirror that group's phase sub-pools when PDD is on.
        self._prefill_pool: EnginePool | None = None
        self._decode_pool: EnginePool | None = None
        if self.strategy.allow_pdd:
            if len(self._groups) != 1:
                raise NotImplementedError(
                    "PDD is not yet supported with per-compute-device parallelism "
                    "sections; configure a single device type or disable allow_pdd"
                )
            group = self._groups[0]
            num_slots = group.pool.num_slots
            if num_slots < 2:
                raise ValueError(
                    f"PDD needs at least 2 engine slots but the group has "
                    f"{num_slots} (degree {group.degree})"
                )
            group_devices = [d for s in group.pool.slots for d in s.devices]
            prefill_slots = round(self.strategy.prefill_engine_fraction * num_slots)
            prefill_slots = max(1, min(num_slots - 1, prefill_slots))
            cut = prefill_slots * group.degree
            self._prefill_pool = EnginePool(group_devices[:cut], group.degree)
            self._decode_pool = EnginePool(group_devices[cut:], group.degree)
            group.prefill_pool = self._prefill_pool
            group.decode_pool = self._decode_pool
        # A planner re-factors a group's fixed engine size into the fastest
        # memory-feasible (pp, ep) per batch (when auto-parallelism is on); cached
        # by (model instance, representative device) since the layout depends only
        # on the model and the device type it runs on.
        self._planners: dict[tuple[int, int], ParallelismPlanner] = {}
        self._rng = random.Random(self.strategy.random_seed)
        # Representative home node per (MoE model, engine group): a node owning one
        # of the serving devices whose RAM streams routed experts on demand. The
        # model itself is sharded across all the slot's nodes (see ``_home_shards``);
        # this node is only the homogeneous fetch source. Chosen the first time the
        # model is placed on the group and then fixed for the run.
        self._home_nodes: dict[tuple[int, int], object] = {}
        # Per-(model, group) RAM reservation: ``-> {id(node): shard_bytes}``. Each
        # participating node holds the fraction of the model its own devices serve
        # for the life of the run, so co-located models must all fit at once.
        self._home_shards: dict[tuple[int, int], dict[int, float]] = {}
        # (model, node) pairs whose weight shard has already been staged into that
        # node's RAM (NVM -> RAM happens once per node; later placements only stage
        # RAM -> device).
        self._home_loaded: set[tuple[int, int]] = set()
        # Memoized expert-fetch link latencies over the fixed topology. The in-node
        # hop is a pure function of the destination device; the streamed hop is a
        # pure function of (source memory, destination device). Both are cached by
        # identity and reused across the millions of per-dispatch evaluations.
        self._in_node_fetch_latency: dict[int, float] = {}
        self._streamed_fetch_latency: dict[tuple[int, int], float] = {}
        # Per-dispatch streaming reservation (per-device resident bytes) keyed by
        # batch index, so the capacity check uses the working-set footprint rather
        # than pinning every expert.
        self._job_reserve: dict[int, float] = {}
        # Per engine slot (``id(slot)``), the arbiter job index whose in-flight
        # weight/expert transfers are warming that slot's resident model. A later
        # batch reusing the slot is gated behind those transfers (it cannot
        # compute before the weights/experts its ranks need are resident). Reset
        # per run; entries clear naturally once the warm-up finishes.
        self._slot_warming_job: dict[int, int] = {}
        # Memory-aware admission (back-pressure) bookkeeping. ``_slot_reserved``
        # maps ``id(slot)`` to the per-device bytes its in-flight jobs currently
        # pin; ``_job_reserved_bytes`` maps an arbiter job index to the footprint
        # it reserved, so a retiring job releases exactly what it took. Both are
        # reset per run.
        self._slot_reserved: dict[int, float] = {}
        self._job_reserved_bytes: dict[int, float] = {}
        # Cached dispatch plans (work shards + MoE activation trace + chosen
        # parallelism) keyed by ``batch_index``. Generating these is the costly
        # part of a dispatch and depends only on the batch's work, so a batch
        # deferred by memory back-pressure reuses its plan on the next attempt
        # instead of regenerating it. Reset per run; entries clear on commit.
        self._dispatch_plans: dict[int, tuple] = {}
        # System-wide persistent KV cache (cross-conversation prefix reuse, LRU
        # eviction, migration across floating memories). Inert when disabled or
        # when the system has no floating (node) memory to offload KV into.
        # Per-device first-tier HBM residency, shared between retained KV and
        # resident routed experts under the chosen memory policy. Empty (and
        # unused) under the legacy ``node_first`` policy.
        self._hbm: dict[int, DeviceHbmResidency] = self._build_hbm()
        self._kv = (
            KVCacheManager(system, self.strategy.memory_policy, self._hbm)
            if self.strategy.global_kv_cache
            else None
        )
        # Standalone KV-move jobs (offloads/spills and the PDD prefill->decode
        # handoff) admitted to the arbiter outside the batch dispatch loop. They
        # are not in ``job_meta`` (they serve no request and hold no engine
        # slot), so their rescaled events are collected separately:
        # ``(job_index, device, batch_index, job_phase)``. Reset per run by
        # ``run`` / ``_run_pdd``.
        self._aux_jobs: list[tuple[int, ComputeDevice, int, str]] = []

    def _build_groups(self) -> list[EngineGroup]:
        """Partition the system's compute devices into engine groups.

        With per-device parallelism sections, one group is built per section over
        exactly the devices whose ``device_key`` matches it (preserving system
        order); a device type with no section is left unused, and a section that
        matches no device is rejected. Without sections a single group spans every
        device, wired by the flat ``StrategyConfig`` parallelism fields -- the
        legacy homogeneous layout.
        """

        strategy = self.strategy
        all_devices = self.system.compute_devices
        if not strategy.parallelism:
            degree = (
                strategy.pipeline_parallel
                * strategy.expert_parallel
                * strategy.tensor_parallel
            )
            if len(all_devices) < degree:
                raise ValueError(
                    f"system has {len(all_devices)} compute devices but the engine "
                    f"needs {degree} (pipeline_parallel x expert_parallel x "
                    f"tensor_parallel)"
                )
            return [
                EngineGroup(
                    device_key="",
                    pool=EnginePool(all_devices, degree),
                    pipeline_parallel=strategy.pipeline_parallel,
                    expert_parallel=strategy.expert_parallel,
                    tensor_parallel=strategy.tensor_parallel,
                    auto_parallelism=strategy.auto_parallelism,
                )
            ]

        groups: list[EngineGroup] = []
        for section in strategy.parallelism:
            devices = [d for d in all_devices if d.device_key == section.compute_device]
            if not devices:
                raise ValueError(
                    f"parallelism section for compute_device "
                    f"{section.compute_device!r} matches no device in the system"
                )
            if len(devices) < section.degree:
                raise ValueError(
                    f"compute_device {section.compute_device!r} has {len(devices)} "
                    f"devices but its engine needs {section.degree} (pipeline_parallel "
                    f"x expert_parallel x tensor_parallel)"
                )
            groups.append(
                EngineGroup(
                    device_key=section.compute_device,
                    pool=EnginePool(devices, section.degree),
                    pipeline_parallel=section.pipeline_parallel,
                    expert_parallel=section.expert_parallel,
                    tensor_parallel=section.tensor_parallel,
                    auto_parallelism=section.auto_parallelism,
                )
            )
        return groups

    def _build_hbm(self) -> dict[int, "DeviceHbmResidency"]:
        """Per compute device, the HBM residency for retained KV and experts.

        Returns an empty map under the legacy ``node_first`` policy (no device
        retention). Otherwise each device gets a residency sized to its first-tier
        capacity, governed by the configured policy (one shared LRU under
        ``global_lru``; reserved KV/expert sub-regions under ``partitioned``).
        """

        policy = self.strategy.memory_policy
        if policy == "node_first":
            return {}
        mode = (
            MemoryPolicy.PARTITIONED
            if policy == "partitioned"
            else MemoryPolicy.GLOBAL_LRU
        )
        hbm: dict[int, DeviceHbmResidency] = {}
        for device in self.system.compute_devices:
            hbm[id(device)] = DeviceHbmResidency(
                device.first_tier_memory.capacity_bytes,
                mode,
                self.strategy.hbm_kv_fraction,
            )
        return hbm

    def _planner_for(
        self, model: object, device: ComputeDevice | None = None
    ) -> ParallelismPlanner:
        """The planner for ``model`` on ``device`` (defaults to the first group).

        Keyed by ``(model, device)`` so each device *type* gets its own planner:
        the per-device byte footprints are device-independent, but the memory
        ``capacity`` and roofline time estimates that drive the auto-parallelism
        search are not, so a B200 group and a Cerebras group must plan against
        their own silicon. Footprint-only callers may omit ``device`` (any
        group's planner yields the same bytes).
        """

        if device is None:
            device = self._groups[0].device
        key = (id(model), id(device))
        planner = self._planners.get(key)
        if planner is None:
            planner = ParallelismPlanner(model, device)
            self._planners[key] = planner
        return planner

    # --- expert streaming ---------------------------------------------------

    def _is_moe(self, model: object) -> bool:
        """Whether ``model`` has routed-expert (MoE) layers."""

        return self._planner_for(model).model.num_moe_layers > 0

    def _distinct_models_by_demand(self, requests: list["Request"]) -> list[object]:
        """The suite's distinct model instances, most-requested first.

        Ties keep first-arrival order, so the ordering is deterministic. Used to
        decide which models to eagerly pre-load on the engine slots for warm
        start (the leading models win the slots when models outnumber them).
        """

        counts: dict[int, int] = {}
        order: dict[int, int] = {}
        models: dict[int, object] = {}
        for pos, req in enumerate(requests):
            key = id(req.model)
            counts[key] = counts.get(key, 0) + 1
            if key not in order:
                order[key] = pos
                models[key] = req.model
        return [models[k] for k in sorted(models, key=lambda k: (-counts[k], order[k]))]

    def _apply_warm_start(self, requests: list["Request"]) -> None:
        """Pre-load the suite's models onto the engine slots before a run.

        No-op unless ``warm_start`` is set. Marks each pool's slots as already
        hosting the suite's models (in demand order, round-robin), so the first
        batch of a pre-warmed model skips the cold ``weight_load``. Inert without
        ``model_weight_loading`` (no weight loads are charged either way).
        """

        if not self.strategy.warm_start:
            return
        models = self._distinct_models_by_demand(requests)
        if not models:
            return
        for group in self._groups:
            for pool in (group.pool, group.prefill_pool, group.decode_pool):
                if pool is not None:
                    pool.warm_start(models)
                    self._warm_start_experts(group, pool, requests)

    def _warm_start_scheme(
        self, model: object, reqs: list["Request"], group: EngineGroup
    ) -> tuple[int, int, int]:
        """The ``(pp, ep, tp)`` a warm-started MoE model's batches will run under.

        Mirrors the per-batch dispatch: the group's fixed scheme, or -- under
        ``auto_parallelism`` -- the factorization the planner picks for a
        representative batch of this model's requests. Warm start must preload
        experts under this layout because the run keys expert residency by the
        *auto-derived* ``ep`` (on each rank's stage-0 device), not the raw
        configured degrees: using the configured scheme would shard the experts
        the wrong way (e.g. asking one rank to hold every expert) and preload
        nothing. A representative trace is seeded deterministically so warm start
        never perturbs the run's RNG. ``None`` when the scheme cannot be derived.
        """

        if not group.auto_parallelism:
            return group.pipeline_parallel, group.expert_parallel, group.tensor_parallel
        cap = self.strategy.max_batch_size if self.strategy.max_batch_size > 0 else len(reqs)
        batch = reqs[: max(1, cap)]
        work = [req.work for req in batch]
        shards = WorkShardGenerator(model).generate(
            work, prefill_chunk_size=self.strategy.prefill_chunk_size
        )
        expert_trace = build_activation_trace(
            model, work, self.strategy.prefill_chunk_size, seed=0
        )
        return self._parallelism_for(model, batch, shards, expert_trace, group=group)

    def _warm_start_experts(
        self, group: EngineGroup, pool: EnginePool, requests: list["Request"]
    ) -> None:
        """Preload each warm-started MoE model's expert shards onto its slot HBM.

        Routed experts are part of a model's weights, so warm start makes them
        resident on each expert rank's device alongside the model, sparing the
        first batch the expert stream. The shard layout follows the scheme the
        run will actually use (auto-derived per :meth:`_warm_start_scheme`, not
        the raw configured degrees), so the preloaded residency lands on the same
        ranks the run later reads from. The exception is a rank whose full expert
        shard does not fit its HBM region: it is left to stream its working set on
        demand. No-op under the legacy ``node_first`` policy, which keeps no
        device residency.
        """

        if not self._hbm:
            return
        by_model: dict[int, list["Request"]] = {}
        for req in requests:
            by_model.setdefault(id(req.model), []).append(req)
        for slot in pool.slots:
            model = pool.resident_model(slot)
            if model is None or not self._is_moe(model):
                continue
            reqs = by_model.get(id(model))
            if not reqs:
                continue
            try:
                pp, ep, tp = self._warm_start_scheme(model, reqs, group)
            except ValueError:
                # The run will raise its own clear infeasibility error at dispatch;
                # leave this model to stream rather than aborting warm start early.
                continue
            if ep * tp > len(slot.devices):
                continue
            layered = self._planner_for(model).model
            ffns = layered.moe_ffns()
            if not ffns:
                continue
            num_experts = ffns[0].num_experts
            index_bytes = (
                sum(ffn.routed_expert_params for ffn in ffns)
                * layered.param_dtype_bytes
            ) / tp
            for r in range(ep):
                residency = self._hbm[id(slot.devices[r * tp])]
                owned = [e for e in range(num_experts) if e % ep == r]
                residency.preload_experts(owned, index_bytes, 0.0)


    def _home_node_for(self, model: object, slot: EngineSlot, group: EngineGroup):
        """A representative node whose RAM streams ``model``'s experts.

        The model is *sharded across the nodes that own the serving slot's
        devices*: each such node need only hold the fraction of the model its own
        devices serve (``full_model_bytes`` split in proportion to its share of
        the slot's devices), so the tensor/pipeline ranks together shard the whole
        model even when no single node could hold it. When every participating
        node can hold its shard the model is homed -- weights stage NVM -> each
        node's RAM, then RAM -> its own devices, and routed experts stream from the
        devices' own (in-node) RAM. The returned node is a representative (it owns
        a slot device; node RAM is homogeneous) used as the expert-fetch source;
        ``None`` means some node cannot hold its shard, so the model keeps
        streaming experts straight from the shared input NVM. Cached per ``(model,
        group)`` because different device-type groups sit on different nodes.
        """

        key = (id(model), id(group))
        if key in self._home_nodes:
            return self._home_nodes[key]
        full = self._planner_for(model).full_model_bytes
        total = len(slot.devices)
        counts: dict[int, int] = {}
        nodes_by_id: dict[int, object] = {}
        for device in slot.devices:
            node = self.system.node_of(device)
            counts[id(node)] = counts.get(id(node), 0) + 1
            nodes_by_id[id(node)] = node

        def shard_of(node_id: int) -> float:
            return full * counts[node_id] / total if total else 0.0

        homed = total > 0 and all(
            nodes_by_id[nid].node_memory is not None
            and nodes_by_id[nid].node_memory.capacity_bytes >= shard_of(nid)
            for nid in counts
        )
        if homed:
            home = self.system.node_of(slot.devices[0])
            self._home_shards[key] = {nid: shard_of(nid) for nid in counts}
        else:
            home = None
            self._home_shards[key] = {}
        self._home_nodes[key] = home
        return home

    def _expert_fetch_latency(self, source, slot: EngineSlot, home) -> float:
        """One-way fabric latency from the experts' source memory to the slot.

        When the model is homed each device fetches its experts from its *own*
        node's RAM (the cheap in-node link), so the worst such in-node hop bounds
        the group's fetch start. Without a home the experts stream from the shared
        input NVM and the slowest device's link to it bounds the start instead.
        """

        if home is not None:
            in_node = self._in_node_fetch_latency
            worst = 0.0
            for d in slot.devices:
                key = id(d)
                latency = in_node.get(key)
                if latency is None:
                    latency = self.system.link_between(
                        self.system.node_of(d).node_memory, d.first_tier_memory
                    ).latency_s
                    in_node[key] = latency
                if latency > worst:
                    worst = latency
            return worst
        streamed = self._streamed_fetch_latency
        source_id = id(source)
        worst = 0.0
        for d in slot.devices:
            key = (source_id, id(d))
            latency = streamed.get(key)
            if latency is None:
                latency = self.system.link_between(
                    source, d.first_tier_memory
                ).latency_s
                streamed[key] = latency
            if latency > worst:
                worst = latency
        return worst

    def run(
        self,
        requests: list[Request],
        *,
        progress: ProgressCallback | None = None,
        events: RunEventCallback | None = None,
    ) -> RunResult:
        """Event-driven serving loop over ``requests``; returns per-request timing.

        If ``progress`` is given it is called with a :class:`RunProgress` each time
        the run retires one or more sequences (and once when it finishes). If
        ``events`` is given it is called with a :class:`RunEvent` at each
        arrival/issue/completion milestone.
        """

        if self.strategy.allow_pdd:
            return self._run_pdd(requests, progress=progress, events=events)

        strategy = self.strategy
        self._aux_jobs = []
        initial, next_of = _turn_chains(requests)
        arrivals = sorted(initial, key=lambda r: r.arrival_time)
        arrival_pos = 0
        ready: list[Request] = []
        window_open: float | None = None

        arbiter = IncrementalArbiter()
        self._slot_warming_job = {}
        self._slot_reserved = {}
        self._job_reserved_bytes = {}
        self._dispatch_plans = {}
        self._apply_warm_start(requests)
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

        def emit_arrival(req: Request, now: float) -> None:
            if events is not None:
                events(RunEvent(
                    kind="arrival", sim_time=now,
                    wall_time=time.perf_counter() - start_wall,
                    completed=len(result.records), total=total,
                    sequence=_seq_label(req.workload_id, req.turn_index, req.request_id),
                    prompt_tokens=req.prompt_tokens, output_tokens=req.output_tokens))

        def emit_issue(
            now: float, b_index: int, batch: Sequence[Request],
            device_names: Sequence[str], phase: str = "",
        ) -> None:
            if events is not None:
                events(RunEvent(
                    kind="issue", sim_time=now,
                    wall_time=time.perf_counter() - start_wall,
                    completed=len(result.records), total=total,
                    batch_index=b_index,
                    members=tuple(
                        _seq_label(r.workload_id, r.turn_index, r.request_id)
                        for r in batch),
                    engine_group=_engine_group_label(device_names), phase=phase))

        def emit_completion(record: RequestRecord, now: float) -> None:
            if events is not None:
                events(RunEvent(
                    kind="completion", sim_time=now,
                    wall_time=time.perf_counter() - start_wall,
                    completed=len(result.records), total=total,
                    sequence=_seq_label(
                        record.workload_id, record.turn_index, record.request_id),
                    queue_delay=record.queue_delay, ttft=record.ttft, tps=record.tps))

        # job_index -> (requests, dispatch_time, batch_index, slot, pool)
        jobs: dict[
            int, tuple[list[Request], float, int, EngineSlot, EnginePool]
        ] = {}
        # job_index -> (job_phase, requests, slot, batch_index, pp, ep, tp)
        job_meta: dict[
            int, tuple[str, list[Request], EngineSlot, int, int, int, int]
        ] = {}
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
            for job_index, (reqs, dispatch_time, b_index, slot, pool) in jobs.items():
                if job_index in reported or not arbiter.job_is_done(job_index):
                    continue
                reported.add(job_index)
                end = arbiter.job_end_time(job_index)
                in_flight -= len(reqs)
                pool.release(slot)
                self._free_reservation(slot, job_index)
                first_token_time = _first_token_end(
                    arbiter.job_rescaled_events(job_index)
                )
                for req in reqs:
                    record = RequestRecord(
                        request_id=req.request_id,
                        arrival_time=req.arrival_time,
                        dispatch_time=dispatch_time,
                        completion_time=end,
                        prompt_tokens=req.prompt_tokens,
                        output_tokens=req.output_tokens,
                        batch_index=b_index,
                        first_token_time=first_token_time,
                        workload_id=req.workload_id,
                        turn_index=req.turn_index,
                    )
                    result.records.append(record)
                    emit_completion(record, now)
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
                        arrived = replace(follow_on, arrival_time=end)
                        ready.append(arrived)
                        emit_arrival(arrived, now)
                        if window_open is None:
                            window_open = now
            if len(result.records) != completed_before:
                report(now)

            # 2) Admit arrivals that have occurred by ``now``.
            while arrival_pos < len(arrivals) and arrivals[arrival_pos].arrival_time <= now:
                arrived = arrivals[arrival_pos]
                ready.append(arrived)
                emit_arrival(arrived, now)
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
                queued_batch = batch
                matches: dict[int, Match] = {}
                kv_fetches: list[tuple[object, float, object]] | None = None
                if self._kv_active:
                    batch, matches = self._resolve_kv(batch, now)
                    kv_fetches = self._kv_fetches(batch, matches)
                group = self._select_group(batch[0].model)
                job_index, slot, pp, ep, tp = self._dispatch(
                    arbiter, batch, group=group, kv_fetches=kv_fetches,
                    result=result, now=now, batch_index=batch_index)
                if job_index is None:
                    # Memory back-pressure: the slot cannot hold this batch yet.
                    # Return it to the front of the queue and wait for an
                    # in-flight job to complete and free memory.
                    in_flight -= len(queued_batch)
                    ready[:0] = queued_batch
                    break
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
                            result, req, match, slot, batch_index, now)
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
                emit_issue(now, batch_index, batch, device_names)
                jobs[job_index] = (batch, now, batch_index, slot, group.pool)
                job_meta[job_index] = ("full", batch, slot, batch_index, pp, ep, tp)
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
        self,
        requests: list[Request],
        *,
        progress: ProgressCallback | None = None,
        events: RunEventCallback | None = None,
    ) -> RunResult:
        """Two-phase PDD loop: prefill pool -> KV transfer -> decode pool.

        A request is prefilled on the prefill pool; on completion its KV cache is
        transferred (a modeled delay derived from the link bandwidth) to the
        decode pool, where it is decoded. Prefill and decode batch independently
        (each with its own window/fill); ``max_concurrency`` counts a sequence
        as in flight from prefill dispatch until decode completion.
        """

        strategy = self.strategy
        self._aux_jobs = []
        initial, next_of = _turn_chains(requests)
        arrivals = sorted(initial, key=lambda r: r.arrival_time)
        arrival_pos = 0
        prefill_ready: list[Request] = []
        decode_ready: list[Request] = []
        # Prefill done, KV handoff transfer in flight: (request, handoff job).
        pending: list[tuple[Request, int]] = []
        # Round-robins handoff destinations across the decode engines so parallel
        # sequences land on distinct decode hardware (as their decode batches
        # will), rather than all contending one representative memory.
        handoff_rr = 0
        prefill_window: float | None = None
        decode_window: float | None = None

        arbiter = IncrementalArbiter()
        self._slot_warming_job = {}
        self._slot_reserved = {}
        self._job_reserved_bytes = {}
        self._dispatch_plans = {}
        self._apply_warm_start(requests)
        in_flight = 0  # counted from prefill dispatch to decode completion
        batch_index = 0
        total = len(requests)
        start_wall = time.perf_counter()

        # job_index -> (kind, requests, slot, pool)
        jobs: dict[int, tuple[str, list[Request], EngineSlot, EnginePool]] = {}
        # job_index -> (job_phase, requests, slot, batch_index, pp, ep, tp)
        job_meta: dict[
            int, tuple[str, list[Request], EngineSlot, int, int, int, int]
        ] = {}
        reported: set[int] = set()
        # id(request) -> (prefill dispatch time, prefill batch index, prefill devices)
        meta: dict[int, tuple[float, int, tuple[str, ...]]] = {}
        result = RunResult()

        def report(now: float) -> None:
            if progress is not None:
                avg_tps, avg_ttft = _running_averages(result.records)
                progress(RunProgress(len(result.records), total, now,
                                     time.perf_counter() - start_wall,
                                     avg_tps, avg_ttft))

        def emit_arrival(req: Request, now: float) -> None:
            if events is not None:
                events(RunEvent(
                    kind="arrival", sim_time=now,
                    wall_time=time.perf_counter() - start_wall,
                    completed=len(result.records), total=total,
                    sequence=_seq_label(req.workload_id, req.turn_index, req.request_id),
                    prompt_tokens=req.prompt_tokens, output_tokens=req.output_tokens))

        def emit_issue(
            now: float, b_index: int, batch: Sequence[Request],
            device_names: Sequence[str], phase: str,
        ) -> None:
            if events is not None:
                events(RunEvent(
                    kind="issue", sim_time=now,
                    wall_time=time.perf_counter() - start_wall,
                    completed=len(result.records), total=total,
                    batch_index=b_index,
                    members=tuple(
                        _seq_label(r.workload_id, r.turn_index, r.request_id)
                        for r in batch),
                    engine_group=_engine_group_label(device_names), phase=phase))

        def emit_completion(record: RequestRecord, now: float) -> None:
            if events is not None:
                events(RunEvent(
                    kind="completion", sim_time=now,
                    wall_time=time.perf_counter() - start_wall,
                    completed=len(result.records), total=total,
                    sequence=_seq_label(
                        record.workload_id, record.turn_index, record.request_id),
                    queue_delay=record.queue_delay, ttft=record.ttft, tps=record.tps))

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
            for _req, handoff in pending:
                ready_time = arbiter.job_end_time(handoff)
                if ready_time is not None and ready_time > arbiter.time:
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
                self._free_reservation(slot, job_index)
                if kind == "prefill":
                    # KV handoff: move each sequence's KV from the prefill engine
                    # to a decode engine as a real, bandwidth-contended transfer;
                    # the sequence enters the decode queue once it completes.
                    decode_slots = self._decode_pool.slots
                    for req in reqs:
                        _, b_index, _ = meta[id(req)]
                        kv_bytes = context_kv_bytes(req.model, req.prompt_tokens)
                        dst = decode_slots[handoff_rr % len(decode_slots)].devices[0]
                        handoff_rr += 1
                        handoff = self._admit_pdd_handoff(
                            arbiter, slot.devices[0], dst, kv_bytes,
                            batch_index=b_index,
                        )
                        if handoff is None:
                            decode_ready.append(req)
                            if decode_window is None:
                                decode_window = now
                            continue
                        pending.append((req, handoff))
                else:  # decode complete -> request done
                    in_flight -= len(reqs)
                    first_token_time = _first_token_end(
                        arbiter.job_rescaled_events(job_index)
                    )
                    for req in reqs:
                        dispatch_time, b_index, _ = meta[id(req)]
                        record = RequestRecord(
                            request_id=req.request_id,
                            arrival_time=req.arrival_time,
                            dispatch_time=dispatch_time,
                            completion_time=end,
                            prompt_tokens=req.prompt_tokens,
                            output_tokens=req.output_tokens,
                            batch_index=b_index,
                            first_token_time=first_token_time,
                            workload_id=req.workload_id,
                            turn_index=req.turn_index,
                        )
                        result.records.append(record)
                        emit_completion(record, now)
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
                            arrived = replace(follow_on, arrival_time=end)
                            prefill_ready.append(arrived)
                            emit_arrival(arrived, now)
                            if prefill_window is None:
                                prefill_window = now
            if len(result.records) != completed_before:
                report(now)

            # 2) Admit arrivals into the prefill queue.
            while arrival_pos < len(arrivals) and arrivals[arrival_pos].arrival_time <= now:
                arrived = arrivals[arrival_pos]
                prefill_ready.append(arrived)
                emit_arrival(arrived, now)
                arrival_pos += 1
                if prefill_window is None:
                    prefill_window = now

            # 3) Move handed-off sequences into the decode queue.
            still_pending: list[tuple[Request, int]] = []
            for req, handoff in pending:
                if arbiter.job_is_done(handoff):
                    decode_ready.append(req)
                    if decode_window is None:
                        decode_window = now
                else:
                    still_pending.append((req, handoff))
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
                queued_batch = batch
                matches: dict[int, Match] = {}
                kv_fetches: list[tuple[object, float, object]] | None = None
                if self._kv_active:
                    batch, matches = self._resolve_kv(batch, now)
                    kv_fetches = self._kv_fetches(batch, matches)
                job_index, slot, pp, ep, tp = self._dispatch(
                    arbiter, batch, group=self._groups[0],
                    pool=self._prefill_pool, phase="prefill",
                    kv_fetches=kv_fetches,
                    result=result, now=now, batch_index=batch_index,
                )
                if job_index is None:
                    # Memory back-pressure on the prefill pool: requeue and wait.
                    in_flight -= len(queued_batch)
                    prefill_ready[:0] = queued_batch
                    break
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
                            result, req, match, slot, batch_index, now)
                    elif req.cached_tokens > 0:
                        result.decisions.append(_decision(
                            now, "kv_reuse", req, device_names, batch_index,
                            tokens=req.cached_tokens,
                            source_workload_id=req.workload_id,
                            source_turn_index=req.turn_index - 1,
                            source_devices=device_names))
                jobs[job_index] = ("prefill", batch, slot, self._prefill_pool)
                job_meta[job_index] = ("prefill", batch, slot, batch_index, pp, ep, tp)
                emit_issue(now, batch_index, batch, device_names, "prefill")
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
                job_index, slot, pp, ep, tp = self._dispatch(
                    arbiter, batch, group=self._groups[0],
                    pool=self._decode_pool, phase="decode",
                    result=result, now=now, batch_index=batch_index,
                )
                if job_index is None:
                    # Memory back-pressure on the decode pool: requeue and wait
                    # for an in-flight decode to free memory (the sequences are
                    # already counted in flight, so the count is unchanged).
                    decode_ready[:0] = batch
                    break
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
                job_meta[job_index] = ("decode", batch, slot, batch_index, pp, ep, tp)
                emit_issue(now, batch_index, batch, device_names, "decode")
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
        job_meta: dict[
            int, tuple[str, list[Request], EngineSlot, int, int, int, int]
        ],
    ) -> None:
        """Attach the raw event log, per-job footprints and first-token times.

        For every job the events are recorded twice -- as generated in isolation
        and after the arbiter rescales them for contention -- and the first decode
        step's completion (from the rescaled events) is the first-token time for
        every request the job serves.
        """

        first_token: dict[int, float] = {}
        for job_index, (job_phase, reqs, slot, b_index, pp, ep, tp) in job_meta.items():
            request_ids = tuple(req.request_id for req in reqs)
            devices = slot.devices
            device_names = tuple(d.name for d in devices)
            model_name = getattr(reqs[0].model, "name", "")
            original = arbiter.job_original_events(job_index)
            rescaled = arbiter.job_rescaled_events(job_index)

            for events, is_rescaled in ((original, False), (rescaled, True)):
                # Carriers of each routed-expert fetch, keyed by group: the set of
                # slot devices that actually stream a shard (one per source node)
                # plus the group's overall span, used below to mark the remaining
                # devices as waiting.
                expert_groups: dict[int, dict] = {}
                for ev in events:
                    if 0 <= ev.device_index < len(device_names):
                        dev_name = device_names[ev.device_index]
                        mem_name = self._event_memory_name(devices[ev.device_index], ev)
                        dst_name = self._event_destination_memory_name(
                            devices[ev.device_index], ev)
                    else:
                        dev_name = ""
                        mem_name = ""
                        dst_name = ""
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
                            model=model_name,
                            flops=ev.flops,
                            bytes_read=ev.bytes_read,
                            compute_time=ev.compute_time,
                            bandwidth_time=ev.bandwidth_time,
                            duration=ev.duration,
                            start=ev.start,
                            end=ev.end,
                            rescaled=is_rescaled,
                            destination_memory=dst_name,
                        )
                    )
                    # A routed-expert fetch streams to the whole engine (the
                    # ranks share the prefetch and none can start the group until
                    # it lands). One device per source node carries the actual
                    # transfer; record the group's carriers and span so that every
                    # other slot device can be marked as waiting once below.
                    if ev.phase == "expert_transfer" and 0 <= ev.device_index < len(device_names):
                        info = expert_groups.get(ev.group_index)
                        if info is None:
                            expert_groups[ev.group_index] = {
                                "carriers": {ev.device_index},
                                "start": ev.start,
                                "end": ev.end,
                                "memory": mem_name,
                            }
                        else:
                            info["carriers"].add(ev.device_index)
                            info["start"] = min(info["start"], ev.start)
                            info["end"] = max(info["end"], ev.end)

                # Every slot device that is not a carrier for a given fetch is
                # stalled waiting for experts -- not idle. Log a single zero-cost
                # waiting marker per such device spanning the whole fetch (no
                # bytes/bandwidth, so memory and DMA accounting stay attributed to
                # the carrier events) so the per-device state reflects the stall.
                for group_index, info in expert_groups.items():
                    for idx, other_name in enumerate(device_names):
                        if idx in info["carriers"]:
                            continue
                        result.events.append(
                            EventRecord(
                                job_index=job_index,
                                batch_index=b_index,
                                job_phase=job_phase,
                                request_ids=request_ids,
                                group_index=group_index,
                                phase="expert_transfer",
                                device=other_name,
                                memory=info["memory"],
                                model=model_name,
                                flops=0.0,
                                bytes_read=0.0,
                                compute_time=0.0,
                                bandwidth_time=0.0,
                                duration=info["end"] - info["start"],
                                start=info["start"],
                                end=info["end"],
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
            per_device_bytes = self._job_footprint(
                reqs[0].model, pp, ep, tp, kv_tokens, b_index)
            weight_bytes = min(
                per_device_bytes,
                self._weight_footprint(reqs[0].model, pp, ep, tp),
            )
            kv_bytes = max(0.0, per_device_bytes - weight_bytes)
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
                    weight_bytes_per_device=weight_bytes,
                    kv_bytes_per_device=kv_bytes,
                    model=model_name,
                )
            )

        # Standalone KV-move jobs (offloads/spills) serve no request and hold no
        # engine slot, so they never entered ``job_meta``. Record their rescaled
        # (and isolated) events here so the memories they touch -- the device's
        # first tier they read from and the floating memory they write into --
        # show the bandwidth that grows the floating memory's occupancy. They
        # reserve no long-lived footprint, so they produce no ``JobRecord``.
        for job_index, device, b_index, aux_phase in self._aux_jobs:
            device_name = device.name
            original = arbiter.job_original_events(job_index)
            rescaled = arbiter.job_rescaled_events(job_index)
            for events, is_rescaled in ((original, False), (rescaled, True)):
                for ev in events:
                    result.events.append(
                        EventRecord(
                            job_index=job_index,
                            batch_index=b_index,
                            job_phase=aux_phase,
                            request_ids=(),
                            group_index=ev.group_index,
                            phase=ev.phase,
                            device=device_name,
                            memory=self._event_memory_name(device, ev),
                            model="",
                            flops=ev.flops,
                            bytes_read=ev.bytes_read,
                            compute_time=ev.compute_time,
                            bandwidth_time=ev.bandwidth_time,
                            duration=ev.duration,
                            start=ev.start,
                            end=ev.end,
                            rescaled=is_rescaled,
                            destination_memory=self._event_destination_memory_name(
                                device, ev),
                        )
                    )

        for record in result.records:
            if record.request_id in first_token:
                record.first_token_time = first_token[record.request_id]

        result.events.sort(key=lambda e: (e.rescaled, e.start, e.job_index, e.group_index))
        result.jobs.sort(key=lambda j: j.job_index)
        result.decisions = self._attach_decision_times(result.decisions, result.events)
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
        result.device_specs = [
            DeviceRecord(
                name=entry["name"],
                node=entry["node"],
                peak_flops_fp16=entry["peak_flops_fp16"],
                first_tier_memory=entry["first_tier_memory"],
                first_tier_capacity_bytes=entry["first_tier_capacity_bytes"],
                first_tier_bandwidth_bytes_per_s=entry[
                    "first_tier_bandwidth_bytes_per_s"],
            )
            for entry in self.system.device_inventory()
        ]
        self._check_memory_capacity(result)

    def _attach_decision_times(
        self, decisions: list[DecisionRecord], events: list[EventRecord]
    ) -> list[DecisionRecord]:
        """Backfill each decision's execution window from the rescaled events.

        A decision is taken at dispatch (its ``time``); its execution happens
        later, once the arbiter has scheduled and rescaled the batch's events.
        The execution window is the span of the rescaled events in the decision's
        batch that realise it: prefill/decode compute for those acts, the weight
        load (a transfer from the input NVM) for ``weight_load``, and the KV fetch
        (a non-NVM transfer) for ``kv_transfer``/``kv_reuse``. Bookkeeping acts
        that do not run as events (``weight_eviction``/``kv_eviction``, and any
        transfer whose events are not tied to the batch) fall back to the
        decision time for both endpoints.
        """

        spans: dict[tuple[int, str], list[float]] = {}
        for ev in events:
            if not ev.rescaled:
                continue
            if ev.phase == "prefill":
                key = (ev.batch_index, "prefill")
            elif ev.phase == "decode":
                key = (ev.batch_index, "decode")
            elif ev.phase == "transfer":
                key = (ev.batch_index, "kv_transfer")
            elif ev.phase == "weight_transfer":
                key = (ev.batch_index, "weight_transfer")
            elif ev.phase == "expert_transfer":
                key = (ev.batch_index, "expert_transfer")
            else:
                continue
            span = spans.get(key)
            if span is None:
                spans[key] = [ev.start, ev.end]
            else:
                span[0] = min(span[0], ev.start)
                span[1] = max(span[1], ev.end)

        phase_of = {
            "prefill": "prefill",
            "decode": "decode",
            "kv_reuse": "kv_transfer",
            "kv_transfer": "kv_transfer",
            "weight_load": "weight_transfer",
            "expert_load": "expert_transfer",
        }
        timed: list[DecisionRecord] = []
        for d in decisions:
            span = spans.get((d.batch_index, phase_of[d.kind])) if d.kind in phase_of else None
            if span is not None:
                started, completed = span[0], span[1]
            else:
                started = completed = d.time
            timed.append(replace(d, time_started=started, time_completed=completed))
        return timed

    def _check_memory_capacity(self, result: RunResult) -> None:
        """Abort the run if any device's reserved footprint exceeds its memory.

        The reservation model pins each job's weights + KV (``per_device_bytes``)
        on its serving devices; a device with a second memory tier can spill the
        overflow into it. Occupancy is evaluated at every job boundary (the only
        instants it can change), so a transient overlap of concurrent jobs is
        caught -- not merely the steady state. Raises
        :class:`MemoryCapacityExceeded` on the first device that overflows.
        """

        capacity: dict[str, float] = {}
        for device in self.system.compute_devices:
            cap = device.first_tier_memory.capacity_bytes
            if device.second_tier_memory is not None:
                cap += device.second_tier_memory.capacity_bytes
            capacity[device.name] = cap

        for name, cap in capacity.items():
            jobs = [
                j for j in result.jobs
                if name in j.devices and j.per_device_bytes
            ]
            if not jobs:
                continue
            breakpoints = sorted({j.start for j in jobs} | {j.end for j in jobs})
            peak = 0.0
            for t in breakpoints:
                occ = sum(j.per_device_bytes for j in jobs if j.start <= t < j.end)
                peak = max(peak, occ)
            # Tiny relative tolerance so exact-fit footprints are not rejected by
            # floating-point rounding.
            if peak > cap * (1.0 + 1e-9):
                raise MemoryCapacityExceeded(name, peak, cap)

        # Each node holds its shard of every model homed across it for the life of
        # the run (the model is sharded over the nodes that serve it), so the
        # shards co-located on one node must all fit its RAM at once.
        for node in self.system.nodes:
            if node.node_memory is None:
                continue
            reserved = sum(
                shards.get(id(node), 0.0)
                for shards in self._home_shards.values()
            )
            ram_cap = node.node_memory.capacity_bytes
            if reserved > ram_cap * (1.0 + 1e-9):
                raise MemoryCapacityExceeded(node.node_memory.name, reserved, ram_cap)

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
        if event.phase in ("transfer", "weight_transfer", "expert_transfer"):
            if event.source_memory is not None:
                return event.source_memory.name
            memory = device.second_tier_memory or device.first_tier_memory
            return memory.name
        return device.first_tier_memory.name

    @staticmethod
    def _event_destination_memory_name(device: ComputeDevice, event: ComputeEvent) -> str:
        """Name of the memory ``event`` wrote its bytes into, or ``""`` if none.

        Mirrors the arbiter's destination-side bandwidth attribution: a transfer
        writes into its explicit ``destination_memory`` when set, else the
        device's first-tier memory. Compute and kernel-launch events have no
        write destination.
        """

        if event.phase not in ("transfer", "weight_transfer", "expert_transfer"):
            return ""
        if event.destination_memory is not None:
            return event.destination_memory.name
        return device.first_tier_memory.name

    def _job_footprint(
        self, model: object, pp: int, ep: int, tp: int, kv_tokens: int,
        batch_index: int = -1,
    ) -> float:
        """Reserved per-device bytes (weights + KV) for a job, or 0 if unknown.

        A streaming job reserves only its working set (recorded at dispatch in
        ``self._job_reserve``); otherwise every owned expert is pinned.
        """

        reserve = self._job_reserve.get(batch_index)
        if reserve is not None:
            return reserve
        try:
            return float(self._planner_for(model).footprint(pp, ep, kv_tokens, tp))
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _weight_footprint(self, model: object, pp: int, ep: int, tp: int) -> float:
        """Reserved per-device *weight* bytes for a job (KV-free footprint), or 0.

        The weight portion is the planner footprint with no KV tokens; subtracting
        it from the full footprint yields the KV portion, so a job's reservation
        can be split into weights vs KV for the per-memory content breakdown.
        """

        try:
            return float(self._planner_for(model).footprint(pp, ep, 0, tp))
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _device_capacity(self, device: ComputeDevice) -> float:
        """Total memory a device can pin: its first tier plus any spill tier."""

        cap = device.first_tier_memory.capacity_bytes
        if device.second_tier_memory is not None:
            cap += device.second_tier_memory.capacity_bytes
        return cap

    def _admit_on_slot(self, slot: EngineSlot, per_device_bytes: float) -> bool:
        """Whether a job reserving ``per_device_bytes`` may dispatch onto ``slot``.

        Memory-aware back-pressure: the job is admitted when its footprint plus
        the footprints already in flight on the slot fit every serving device's
        memory (first tier plus any spill tier). When it does not fit but the slot
        still holds other in-flight jobs, admission is refused so the caller defers
        the batch until a completion frees memory -- the run serialises rather than
        over-subscribing the devices. A job that overflows an *empty* slot cannot
        be helped by waiting, so it is admitted and left to the post-run capacity
        check, which reports the genuine infeasibility.
        """

        reserved = self._slot_reserved.get(id(slot), 0.0)
        capacity = min(self._device_capacity(d) for d in slot.devices)
        if reserved + per_device_bytes <= capacity * (1.0 + 1e-9):
            return True
        return reserved <= 0.0

    def _record_reservation(
        self, slot: EngineSlot, job_index: int, per_device_bytes: float
    ) -> None:
        """Charge a dispatched job's per-device footprint to its slot."""

        if per_device_bytes <= 0.0:
            return
        self._slot_reserved[id(slot)] = (
            self._slot_reserved.get(id(slot), 0.0) + per_device_bytes
        )
        self._job_reserved_bytes[job_index] = per_device_bytes

    def _free_reservation(self, slot: EngineSlot, job_index: int) -> None:
        """Release a retired job's tracked footprint back to its slot."""

        released = self._job_reserved_bytes.pop(job_index, 0.0)
        if released:
            self._slot_reserved[id(slot)] = (
                self._slot_reserved.get(id(slot), 0.0) - released
            )

    def _take_batch(self, ready: list[Request], in_flight: int) -> list[Request] | None:
        """Pull up to ``max_batch_size`` same-model requests, honoring concurrency.

        Returns ``None`` if concurrency is exhausted and work is already in
        flight (so a completion must free a slot first). When nothing is in
        flight a batch is always returned, to guarantee progress.
        """

        strategy = self.strategy
        if strategy.max_concurrency is None:
            slots = strategy.max_batch_size
        else:
            slots = strategy.max_concurrency - in_flight
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
        sequences were already counted toward ``max_concurrency`` when their
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


    def _select_group(self, model: object) -> EngineGroup:
        """Choose the engine group (device type) to host a batch of ``model``.

        Spreads batches across the system's device-type groups: prefers a group
        with a free slot already holding the model (no weight reload), then any
        group with a free slot (most-free first), and finally the group whose
        least-loaded slot is emptiest so a saturated system time-shares its
        roomiest engine. With a single group (the homogeneous default) this always
        returns that group, so existing behaviour is unchanged.
        """

        best: tuple[tuple[int, int, int, int], EngineGroup] | None = None
        for index, group in enumerate(self._groups):
            pool = group.pool
            slots = pool.slots
            free = pool.free_count
            affinity = any(
                pool.is_free(s) and pool.resident_model(s) is model for s in slots
            )
            min_load = min(pool.slot_load(s) for s in slots)
            key = (0 if (affinity and free > 0) else 1, -free, min_load, index)
            if best is None or key < best[0]:
                best = (key, group)
        assert best is not None
        return best[1]

    def _dispatch(
        self,
        arbiter: IncrementalArbiter,
        batch: list[Request],
        *,
        group: EngineGroup,
        pool: EnginePool | None = None,
        phase: str = "full",
        kv_fetches: list[tuple[object, float, object]] | None = None,
        result: "RunResult | None" = None,
        now: float = 0.0,
        batch_index: int = -1,
    ) -> tuple[int | None, EngineSlot, int, int, int]:
        """Place a batch on an engine slot and admit its events to the arbiter.

        ``group`` is the engine group (device type + parallelism scheme) the batch
        runs on; its ``(pp, ep, tp)`` and planner are used to wire and size the
        job. ``phase`` selects the work generated: ``"full"`` (prefill + decode in
        one job, the default), ``"prefill"`` (prompt forward pass only) or
        ``"decode"`` (generation from a fully-cached prompt). ``pool`` defaults to
        the group's pool; PDD passes the group's prefill or decode pool.
        ``kv_fetches`` lists ``(floating_memory, bytes)`` prefixes to fetch from
        the global KV cache before compute. Returns the arbiter job index, the
        slot the batch occupies (released on completion), and the
        ``(pipeline_parallel, expert_parallel, tensor_parallel)`` arrangement
        chosen. A ``None`` job index signals memory back-pressure: the batch did
        not fit the slot and must be deferred (the placement has been rolled
        back).
        """

        model = batch[0].model
        pool = pool or group.pool
        work = [self._phase_work(req, phase) for req in batch]

        # Plan the batch: work shards, the MoE activation trace, and the chosen
        # parallelism. This is the expensive part of a dispatch (it materialises
        # every work shard and simulates the per-token expert trace) and depends
        # only on the batch's work, not on which slot it lands on. Cache it by
        # ``batch_index`` so a batch deferred by memory back-pressure re-attempts
        # placement without regenerating the plan; an unexpected change in the
        # batch's work (e.g. KV reuse shifting the prefill) invalidates the cache.
        sig = (
            phase,
            tuple(
                (r.request_id, w.prefill_tokens, w.decode_tokens)
                for r, w in zip(batch, work)
            ),
        )
        plan = self._dispatch_plans.get(batch_index)
        if plan is None or plan[0] != sig:
            shards = WorkShardGenerator(model).generate(
                work, prefill_chunk_size=self.strategy.prefill_chunk_size
            )
            expert_trace = (
                build_activation_trace(
                    model, work, self.strategy.prefill_chunk_size,
                    seed=self._rng.randrange(1 << 30),
                )
                if self._is_moe(model)
                else None
            )
            pp, ep, tp = self._parallelism_for(
                model, batch, shards, expert_trace, group=group
            )
            expert_cap = (
                max(1, peak_active_per_rank(expert_trace, ep))
                if expert_trace is not None
                else None
            )
            plan = (sig, shards, expert_trace, pp, ep, tp, expert_cap)
            self._dispatch_plans[batch_index] = plan
        _, shards, expert_trace, pp, ep, tp, expert_cap = plan

        # A real stack stages weights through host RAM: when a node can hold the
        # whole model it becomes the model's home (NVM -> RAM once, then RAM ->
        # device). MoE models additionally keep only their working set resident
        # and stream the rest of the experts on demand -- from the home node's RAM
        # when one fits, otherwise straight from the shared input NVM (so even a
        # model too large for any single node's RAM still streams its experts
        # instead of pinning every one on the serving devices).
        placement = pool.place(model)
        slot = placement.slot
        home = self._home_node_for(model, slot, group)
        expert_source = None
        expert_sources = None
        expert_latency = 0.0
        if expert_trace is not None:
            if home is not None:
                # The model is sharded across the nodes owning the slot's devices;
                # each device streams its experts from its own node's RAM, so the
                # movement spreads across every node the slot spans rather than
                # funneling through one representative memory.
                expert_sources = [
                    self.system.node_of(d).node_memory for d in slot.devices
                ]
            else:
                # No node can home the model: all devices stream from the shared
                # input NVM (one source, so co-streaming contends on it).
                expert_sources = [self.system.input_memory for _ in slot.devices]
            expert_source = expert_sources[0]
            expert_latency = self._expert_fetch_latency(expert_source, slot, home)
            kv_tokens = sum(r.prompt_tokens + r.output_tokens for r in batch)
            self._job_reserve[batch_index] = self._planner_for(
                model, group.device
            ).streaming_footprint(pp, kv_tokens, expert_cap, tp)

        # Memory-aware admission: a real engine cannot pin more than its devices
        # hold, so rather than over-subscribe and crash, refuse a batch whose
        # footprint would not fit the chosen slot yet -- the caller defers it until
        # in-flight work frees memory (back-pressure). Undo the speculative
        # placement and streaming reservation so the pool is untouched.
        kv_tokens_full = sum(r.prompt_tokens + r.output_tokens for r in batch)
        per_device_bytes = self._job_footprint(
            model, pp, ep, tp, kv_tokens_full, batch_index
        )
        if not self._admit_on_slot(slot, per_device_bytes):
            self._job_reserve.pop(batch_index, None)
            pool.unplace(placement)
            return None, slot, pp, ep, tp

        # Committed: the batch will run, so its cached plan is no longer needed.
        self._dispatch_plans.pop(batch_index, None)

        generator = EventGenerator(
            model,
            list(slot.devices),
            pipeline_parallel=pp,
            expert_parallel=ep,
            tensor_parallel=tp,
            event_random_factor_range=self.strategy.event_random_factor_range,
            rng=self._rng,
            scale_up_bandwidth_bytes_per_s=self.system.network.scale_up_bandwidth_bytes_per_s,
            scale_up_latency_s=self.system.network.scale_up_latency_s,
        )
        # Device-first policies keep routed experts resident on each rank's HBM
        # across batches (sharing the pool with retained KV). The residency for
        # ep rank ``r`` lives on that rank's stage-0 device (``r * tp``); a model
        # swap on a device clears its experts (new weights overwrite them).
        expert_residency = None
        expert_index_bytes = None
        if expert_trace is not None and self._hbm:
            if placement.needs_weight_load:
                for device in slot.devices:
                    self._hbm[id(device)].clear_experts()
            expert_index_bytes = generator.routed_expert_bytes_per_index / tp
            expert_residency = [
                self._hbm[id(slot.devices[r * tp])] for r in range(ep)
            ]
        loads: list[ComputeEvent] = []
        if self.strategy.model_weight_loading and placement.needs_weight_load:
            loads = self._weight_load_events(model, slot, pp, ep, tp, home, group)
            if result is not None:
                self._emit_weight_decisions(
                    result, batch, slot, placement.evicted_model, now, batch_index)
        fetches = self._kv_fetch_events(slot, kv_fetches) if kv_fetches else []
        prelude = loads + fetches
        # A batch reusing an engine slot whose cold load is still streaming must
        # wait for that warm-up; a batch that loads its own weights establishes a
        # fresh one and gates on nothing prior.
        after_job = (
            None if placement.needs_weight_load
            else self._slot_warming_job.get(id(slot))
        )
        if prelude or expert_trace is not None:
            schedule = generator.run(
                shards,
                expert_trace=expert_trace,
                expert_cache_capacity=expert_cap,
                expert_source=expert_source,
                expert_sources=expert_sources,
                expert_fetch_latency=expert_latency,
                expert_residency=expert_residency,
                expert_index_bytes=expert_index_bytes,
                expert_now=now,
            )
            if schedule.expert_evicted_kv and self._kv is not None:
                self._reconcile_expert_kv_eviction(
                    arbiter, slot, schedule.expert_evicted_kv,
                    now, result, batch_index)
            if result is not None and expert_trace is not None:
                self._emit_expert_decisions(
                    result, batch, slot, schedule, now, batch_index, expert_source.name)
            events = self._prepend_transfers(prelude, schedule.events)
            job_index = arbiter.admit_events(
                events, list(slot.devices), after_job=after_job)
            # Future reuses of this slot wait for whatever this job is loading.
            if loads or any(e.phase == "expert_transfer" for e in schedule.events):
                self._slot_warming_job[id(slot)] = job_index
            self._record_reservation(slot, job_index, per_device_bytes)
            return job_index, slot, pp, ep, tp
        job_index = arbiter.admit(generator, shards, after_job=after_job)
        self._record_reservation(slot, job_index, per_device_bytes)
        return job_index, slot, pp, ep, tp

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
        self, model: object, slot: EngineSlot, pp: int, ep: int, tp: int, home,
        group: EngineGroup,
    ) -> list[ComputeEvent]:
        """Weight-load (``weight_transfer``) events for a slot (re)load.

        A real serving stack stages a model through host RAM: the weights are
        read from the shared input NVM into the serving nodes' RAM once, then
        streamed from that RAM onto each serving device. The model is sharded
        across the slot's nodes, so when it is homed this emits (1) a one-off
        NVM -> RAM transfer into *each* participating node of that node's shard of
        the model (skipped on later placements -- the RAM keeps it resident) and
        (2) a RAM -> device transfer of the resident *non-expert* weights per
        device, sourced from that device's *own* node RAM (the routed experts
        arrive lazily as ``expert_transfer`` fetches). When no node can home its
        shard but the model is MoE, the device still loads only its resident
        *non-expert* weights -- straight from the NVM -- and streams the
        experts from that same NVM. Only a dense model without a home falls back
        to the legacy single stage: NVM -> each device of the whole resident
        footprint. The link latency is folded into ``bandwidth_time`` so the
        arbiter reproduces each transfer's duration while contending its bytes on
        the source memory's bandwidth.
        """

        planner = self._planner_for(model)
        events: list[ComputeEvent] = []
        if home is not None:
            nvm = self.system.input_memory
            shards = self._home_shards.get((id(model), id(group)), {})
            # Stage 1: NVM -> each participating node's RAM, once per (model, node),
            # of that node's shard of the model (its devices' share of the slot).
            staged_nodes: set[int] = set()
            for index, device in enumerate(slot.devices):
                node = self.system.node_of(device)
                if id(node) in staged_nodes:
                    continue
                staged_nodes.add(id(node))
                marker = (id(model), id(node))
                if marker in self._home_loaded:
                    continue
                self._home_loaded.add(marker)
                shard = float(shards.get(id(node), 0.0))
                if shard <= 0:
                    continue
                ram = node.node_memory
                link = self.system.link_between(nvm, ram)
                d1 = transfer_duration(shard, nvm, ram, link)
                events.append(ComputeEvent(
                    group_index=-1, phase="weight_transfer", device_index=index,
                    flops=0.0, bytes_read=shard, compute_time=0.0,
                    bandwidth_time=d1, duration=d1, start=0.0, end=d1,
                    source_memory=nvm, destination_memory=ram))
            stage1_end = max((e.end for e in events), default=0.0)
            # Stage 2: each node's RAM -> its own devices (resident non-experts).
            dev_bytes = float(planner.streaming_footprint(pp, 0, 0, tp))
            if dev_bytes > 0:
                for index, device in enumerate(slot.devices):
                    ram = self.system.node_of(device).node_memory
                    destination = device.first_tier_memory
                    link = self.system.link_between(ram, destination)
                    d2 = transfer_duration(dev_bytes, ram, destination, link)
                    events.append(ComputeEvent(
                        group_index=-1, phase="weight_transfer", device_index=index,
                        flops=0.0, bytes_read=dev_bytes, compute_time=0.0,
                        bandwidth_time=d2, duration=d2, start=stage1_end,
                        end=stage1_end + d2, source_memory=ram,
                        destination_memory=destination))
            return events

        source = self.system.input_memory
        if self._is_moe(model):
            # No node can home the whole model: stream the resident non-expert
            # weights from the NVM (experts stream lazily from the same NVM).
            dev_bytes = float(planner.streaming_footprint(pp, 0, 0, tp))
            if dev_bytes <= 0:
                return []
            for index, device in enumerate(slot.devices):
                destination = device.first_tier_memory
                link = self.system.link_between(source, destination)
                duration = transfer_duration(dev_bytes, source, destination, link)
                events.append(ComputeEvent(
                    group_index=-1, phase="weight_transfer", device_index=index,
                    flops=0.0, bytes_read=dev_bytes, compute_time=0.0,
                    bandwidth_time=duration, duration=duration, start=0.0,
                    end=duration, source_memory=source,
                    destination_memory=destination))
            return events

        weight_bytes = float(planner.footprint(pp, ep, 0, tp))
        if weight_bytes <= 0:
            return []
        for index, device in enumerate(slot.devices):
            destination = device.first_tier_memory
            link = self.system.link_between(source, destination)
            duration = transfer_duration(weight_bytes, source, destination, link)
            events.append(
                ComputeEvent(
                    group_index=-1,
                    phase="weight_transfer",
                    device_index=index,
                    flops=0.0,
                    bytes_read=weight_bytes,
                    compute_time=0.0,
                    bandwidth_time=duration,
                    duration=duration,
                    start=0.0,
                    end=duration,
                    source_memory=source,
                    destination_memory=destination,
                )
            )
        return events

    def _emit_weight_decisions(
        self,
        result: "RunResult",
        batch: list[Request],
        slot: EngineSlot,
        evicted_model: object | None,
        now: float,
        batch_index: int,
    ) -> None:
        """Record the weight-residency decisions for a slot (re)load.

        When this placement displaced a different model, a ``weight_eviction`` is
        recorded first (the prior resident leaves the slot), then the
        ``weight_load`` that streams this batch's model in from the input NVM.
        """

        device_names = tuple(d.name for d in slot.devices)
        if evicted_model is not None:
            result.decisions.append(self._weight_eviction_decision(
                now, evicted_model, device_names, batch_index))
        members = tuple((r.workload_id, r.turn_index) for r in batch)
        result.decisions.append(_decision(
            now, "weight_load", batch[0], device_names, batch_index,
            source_devices=(self.system.input_memory.name,),
            batch_members=members))

    def _emit_expert_decisions(
        self,
        result: "RunResult",
        batch: list[Request],
        slot: EngineSlot,
        schedule,
        now: float,
        batch_index: int,
        source_name: str,
    ) -> None:
        """Record the routed-expert streaming decisions for a dispatched batch.

        A batch that misses on its working set fetches the absent experts from the
        home node's RAM: that is one ``expert_load`` decision (its execution window
        is later backfilled from the batch's ``expert_transfer`` events). When the
        residency LRU has to evict warm experts to make room, an
        ``expert_eviction`` decision records that bookkeeping act. Batches that hit
        entirely on resident experts emit nothing.
        """

        device_names = tuple(d.name for d in slot.devices)
        members = tuple((r.workload_id, r.turn_index) for r in batch)
        if schedule.expert_experts_loaded > 0:
            result.decisions.append(_decision(
                now, "expert_load", batch[0], device_names, batch_index,
                tokens=schedule.expert_experts_loaded,
                source_devices=(source_name,),
                batch_members=members))
        if schedule.expert_evictions > 0:
            result.decisions.append(_decision(
                now, "expert_eviction", batch[0], device_names, batch_index,
                tokens=schedule.expert_evictions,
                source_devices=(source_name,),
                batch_members=members))

    @staticmethod
    def _weight_eviction_decision(
        now: float, model: object, device_names: tuple[str, ...], batch_index: int
    ) -> DecisionRecord:
        """A ``weight_eviction`` decision for a model displaced from a slot."""

        return DecisionRecord(
            time=now,
            kind="weight_eviction",
            request_id=-1,
            workload_id=-1,
            turn_index=-1,
            model=getattr(model, "name", "model"),
            devices=device_names,
            batch_index=batch_index,
        )

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
    ) -> list[tuple[object, float, object]]:
        """``(source_memory, bytes, source_device)`` for each reused prefix.

        ``source_device`` is the compute device whose HBM holds the prefix under a
        device-first policy, or ``None`` when it lives in node memory.
        """

        fetches: list[tuple[object, float, object]] = []
        for req in batch:
            match = matches.get(id(req))
            if match is not None and req.cached_tokens > 0:
                num_bytes = float(context_kv_bytes(req.model, req.cached_tokens))
                fetches.append((match.entry.memory, num_bytes, match.entry.device))
        return fetches

    def _kv_fetch_events(
        self, slot: EngineSlot, fetches: list[tuple[object, float, object]]
    ) -> list[ComputeEvent]:
        """Transfer events that fetch reused prefixes into the serving slot.

        Each ``(source_memory, bytes, source_device)`` becomes one ``transfer``
        event reading the cached prefix into a slot device's first-tier memory
        (round-robined across the slot so several fetches parallelise). A prefix
        whose KV already resides on one of the slot's own devices needs no fetch
        (it is in place), so it is skipped. The link latency is folded into
        ``bandwidth_time`` so the arbiter reproduces the duration while contending
        the source memory's bandwidth against all other in-flight transfers; the
        batch's compute waits on the fetch.
        """

        events: list[ComputeEvent] = []
        num_devices = len(slot.devices)
        for index, (source, num_bytes, source_device) in enumerate(fetches):
            # Same-slot residency: the prefix is already on a serving device's HBM,
            # so reuse is in place with no transfer.
            if source_device is not None and any(
                source_device is d for d in slot.devices
            ):
                continue
            slot_index = index % num_devices
            destination = slot.devices[slot_index].first_tier_memory
            if id(source) == id(destination):
                continue
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
                    destination_memory=destination,
                )
            )
        return events

    def _emit_kv_reuse(
        self,
        result: RunResult,
        req: Request,
        match: Match,
        slot: EngineSlot,
        batch_index: int,
        now: float,
    ) -> None:
        """Record the ``kv_reuse`` (and, unless in place, ``kv_transfer``) decisions.

        The reuse names the source sequence (its conversation/turn) and the memory
        holding the prefix. A fetch ``kv_transfer`` is recorded only when the
        prefix must be moved into the serving devices; a prefix already resident on
        one of the slot's own devices (device-first in-place reuse) needs no move,
        so only the reuse is recorded.
        """

        device_names = tuple(d.name for d in slot.devices)
        source_memory = (match.entry.memory.name,)
        result.decisions.append(_decision(
            now, "kv_reuse", req, source_memory, batch_index,
            tokens=req.cached_tokens,
            source_workload_id=match.entry.workload_id,
            source_turn_index=match.entry.turn_index,
            source_devices=device_names))
        in_place = match.entry.device is not None and any(
            match.entry.device is d for d in slot.devices
        )
        if in_place:
            return
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
        """Retain a finished sequence's KV; record the resulting decisions.

        Under ``node_first`` the KV is offloaded to a floating node memory and an
        arbiter-accounted device->floating ``transfer`` is admitted (so its
        bandwidth contends with concurrent work). Under a device-first policy the
        KV stays on the serving device's HBM with no offload transfer; instead any
        residents that retention spilled off HBM are moved device->node here. A
        ``kv_eviction`` decision is emitted per entry dropped entirely, and a
        ``kv_transfer`` decision per physical move.
        """

        context_tokens = req.prompt_tokens + req.output_tokens
        store = self._kv.store(
            req.model, req.tracker, req.workload_id, req.turn_index,
            context_tokens, now, device=slot.devices[0],
        )
        for victim in store.evicted:
            result.decisions.append(self._eviction_decision(now, victim))
        # Residents spilled off HBM to make room are physically moved down to node
        # memory (charged against the spilling, not the retired request). They were
        # evicted from the storing device's own KV region, so its first tier is the
        # source of each spill move.
        for spilled in store.spilled:
            self._admit_kv_move(
                arbiter, slot.devices[0], spilled.memory, spilled.num_bytes,
                batch_index)
            result.decisions.append(_decision(
                now, "kv_transfer", req, (spilled.memory.name,), batch_index,
                tokens=spilled.context_tokens,
                source_workload_id=spilled.workload_id,
                source_turn_index=spilled.turn_index,
                source_devices=(slot.devices[0].name,),
                bytes_moved=spilled.num_bytes))
        if store.memory is None:
            return
        device_names = tuple(d.name for d in slot.devices)
        # A device-first retention keeps the KV on the producing device's own HBM
        # -- it is already there, so no offload transfer is charged.
        retained_on_device = store.memory is slot.devices[0].first_tier_memory and (
            id(slot.devices[0]) in self._hbm
        )
        if not retained_on_device:
            self._admit_kv_move(
                arbiter, slot.devices[0], store.memory, store.num_bytes, batch_index)
        result.decisions.append(_decision(
            now, "kv_transfer", req, (store.memory.name,), batch_index,
            tokens=context_tokens, source_request_id=req.request_id,
            source_workload_id=req.workload_id, source_turn_index=req.turn_index,
            source_devices=device_names, bytes_moved=store.num_bytes))

    def _reconcile_expert_kv_eviction(
        self,
        arbiter: IncrementalArbiter,
        slot: EngineSlot,
        residents: list,
        now: float,
        result: "RunResult | None",
        batch_index: int,
    ) -> None:
        """Spill KV that this dispatch's expert admissions evicted off HBM.

        Under ``global_lru`` routed experts and retained KV share a device's pool,
        so making an expert resident can evict the least-recently-used KV. That KV
        is only stored on the slot's lead device, so each spill is a lead-device ->
        node move (or a drop if the node pool is full). Records the resulting
        ``kv_transfer`` / ``kv_eviction`` decisions.
        """

        spilled, dropped = self._kv.spill_residents(residents, now)
        source_device = slot.devices[0]
        for entry in spilled:
            self._admit_kv_move(
                arbiter, source_device, entry.memory, entry.num_bytes, batch_index)
        if result is None:
            return
        for entry in spilled:
            result.decisions.append(DecisionRecord(
                time=now,
                kind="kv_transfer",
                request_id=-1,
                workload_id=entry.workload_id,
                turn_index=entry.turn_index,
                model=getattr(entry.model, "name", "model"),
                devices=(entry.memory.name,),
                batch_index=batch_index,
                tokens=entry.context_tokens,
                source_workload_id=entry.workload_id,
                source_turn_index=entry.turn_index,
                source_devices=(source_device.name,),
                bytes_moved=entry.num_bytes))
        for entry in dropped:
            result.decisions.append(self._eviction_decision(now, entry))

    def _admit_kv_move(
        self,
        arbiter: IncrementalArbiter,
        device,
        floating,
        num_bytes: float,
        batch_index: int = -1,
    ) -> None:
        """Admit a standalone, arbiter-accounted KV offload transfer.

        Moves ``num_bytes`` of KV from ``device``'s first-tier memory to the
        ``floating`` memory. The transfer contends the bandwidth of both ends --
        the device's first tier (source) and the floating memory (destination) --
        so the slower end (typically the floating tier) surfaces as contention on
        concurrent work; it holds no engine slot and does not delay the served
        request, which has already retired. The job is registered in
        ``self._aux_jobs`` so ``_collect_outputs`` records its rescaled event in
        the log -- otherwise the floating memory's occupancy would grow with no
        matching bandwidth.
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
            source_memory=device.first_tier_memory,
            destination_memory=floating,
        )
        job_index = arbiter.admit_events([event], [device])
        self._aux_jobs.append((job_index, device, batch_index, "kv_offload"))

    def _admit_pdd_handoff(
        self,
        arbiter: IncrementalArbiter,
        source: ComputeDevice,
        destination: ComputeDevice,
        num_bytes: float,
        batch_index: int = -1,
    ) -> int | None:
        """Admit the prefill->decode KV handoff as an arbiter-accounted transfer.

        A disaggregated sequence's KV must move from the prefill engine's
        first-tier memory to the decode engine's first-tier memory before decode
        can begin. Modelling it as a real transfer event -- rather than a fixed
        clock delay -- makes it obey the same physics as every other move: it
        contends the bandwidth of both ends, so concurrent handoffs sharing a
        memory slow each other down, and the moved bytes surface in the
        per-memory bandwidth reports. The job holds no engine slot and reserves
        no footprint; the sequence is gated into the decode queue on its
        completion. Returns the job index, or ``None`` when there is nothing to
        move.
        """

        if num_bytes <= 0:
            return None
        src_mem = source.first_tier_memory
        dst_mem = destination.first_tier_memory
        link = self.system.link_between(src_mem, dst_mem)
        duration = transfer_duration(num_bytes, src_mem, dst_mem, link)
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
            source_memory=src_mem,
            destination_memory=dst_mem,
        )
        job_index = arbiter.admit_events([event], [source])
        self._aux_jobs.append((job_index, source, batch_index, "kv_transfer"))
        return job_index

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
        self, model: object, batch: list[Request], shards: list,
        expert_trace: list | None = None, *, group: EngineGroup,
    ) -> tuple[int, int, int]:
        """Pick (pipeline_parallel, expert_parallel, tensor_parallel) for a batch.

        Fixed from the group's scheme unless its ``auto_parallelism`` is set, in
        which case the planner searches the factorizations of the ``pp x ep``
        budget for the fastest memory-feasible arrangement of this batch;
        ``tensor_parallel`` is always taken verbatim and applied to the footprint.
        Either way the chosen arrangement must fit in the group's device memory; a
        batch that cannot be placed raises rather than producing a physically
        impossible (over-capacity) schedule. When ``expert_trace`` is given the
        model streams its experts, so the fit only reserves the per-rank working
        set instead of every expert.
        """

        planner = self._planner_for(model, group.device)
        kv_tokens = sum(req.prompt_tokens + req.output_tokens for req in batch)
        tp = group.tensor_parallel
        cap_for = (
            (lambda ep: peak_active_per_rank(expert_trace, ep))
            if expert_trace is not None
            else None
        )

        if not group.auto_parallelism:
            pp = group.pipeline_parallel
            ep = group.expert_parallel
            if cap_for is not None:
                per_device = planner.streaming_footprint(
                    pp, kv_tokens, cap_for(ep), tp
                )
            else:
                per_device = planner.footprint(pp, ep, kv_tokens, tp)
            if per_device > planner.capacity:
                raise ValueError(
                    f"fixed parallelism pp={pp}, ep={ep}, tp={tp} cannot serve a "
                    f"batch of {kv_tokens} KV tokens: the per-device footprint is "
                    f"{per_device:.0f} bytes but device memory holds only "
                    f"{planner.capacity:.0f} bytes. Raise pipeline_parallel/"
                    f"expert_parallel/tensor_parallel, enable auto_parallelism, or "
                    f"use a device with more memory."
                )
            return pp, ep, tp

        flops_by_dtype: dict[int, float] = {}
        total_bytes = 0.0
        for shard in shards:
            flops_by_dtype[shard.flops_dtype_bytes] = (
                flops_by_dtype.get(shard.flops_dtype_bytes, 0.0) + shard.flops
            )
            total_bytes += shard.bytes_read
        choice = planner.plan(
            group.degree // tp,
            kv_tokens=kv_tokens,
            flops_by_dtype=flops_by_dtype,
            total_bytes=total_bytes,
            tensor_parallel=tp,
            expert_cache_capacity=cap_for,
        )
        return choice.pipeline_parallel, choice.expert_parallel, choice.tensor_parallel
