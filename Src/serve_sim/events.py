"""Event generation: turn work shards into timed compute events.

The event generator maps each work shard onto a grid of ``pipeline_parallel x
expert_parallel`` devices, consolidates the shards of one forward-pass group
that land on the same device into a single event (a roofline ``max`` of
compute-bound and bandwidth-bound time), and lays the events out to produce a
makespan.

Pipeline parallelism partitions the layers across stages. For a single batch
there is no pipeline overlap, so the stages of a group run sequentially and
latency is the sum of the stage events -- which keeps the roofline well-defined
and conserved across the partition.

Expert parallelism partitions the routed experts across the ``expert_parallel``
devices of a stage (expert ``e`` lives on rank ``e % expert_parallel``). The
compute of a stage is split evenly across those ranks (balanced routing, with
the non-expert work tensor-parallel across the group), so the ranks run
concurrently and an evenly balanced stage is ``expert_parallel`` times faster.
Each rank keeps its own LRU residency of the experts it owns; in a two-tier
system those experts are streamed up from the second tier on demand, and when
the second tier is a single shared device (a system NVM) its bandwidth bounds
the aggregate movement. Inter-stage activation transfers are not modeled here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .blocks import LayeredModel
from .hardware import ComputeDevice
from .shards import WorkShard
from .tiering import ExpertResidencyCache, GroupActivation


@dataclass(frozen=True)
class ComputeEvent:
    """A timed unit of compute on one device.

    Attributes:
        group_index: Originating forward-pass group.
        phase: ``"prefill"`` or ``"decode"``.
        device_index: Index into the event generator's device list.
        flops: Total consolidated FLOPs.
        bytes_read: Total consolidated bytes read.
        compute_time: Compute-bound duration.
        bandwidth_time: Bandwidth-bound duration.
        duration: ``max(compute_time, bandwidth_time)``.
        start: Start time on the global clock.
        end: End time on the global clock.
    """

    group_index: int
    phase: str
    device_index: int
    flops: float
    bytes_read: float
    compute_time: float
    bandwidth_time: float
    duration: float
    start: float
    end: float


@dataclass
class EventSchedule:
    """Result of event generation: ordered events and derived totals."""

    events: list[ComputeEvent] = field(default_factory=list)

    @property
    def makespan(self) -> float:
        """Wall-clock time to complete all events."""

        return max((e.end for e in self.events), default=0.0)

    def time_for_phase(self, phase: str) -> float:
        """Summed event durations for a given phase."""

        return sum(e.duration for e in self.events if e.phase == phase)

    @property
    def total_flops(self) -> float:
        return sum(e.flops for e in self.events)

    @property
    def total_bytes(self) -> float:
        return sum(e.bytes_read for e in self.events)


class EventGenerator:
    """Maps work shards to timed compute events across devices."""

    def __init__(
        self,
        model,
        compute_devices: list[ComputeDevice],
        pipeline_parallel: int = 1,
        expert_parallel: int = 1,
    ) -> None:
        model = LayeredModel.from_model(model)
        if not compute_devices:
            raise ValueError("at least one compute device is required")
        if pipeline_parallel < 1 or expert_parallel < 1:
            raise ValueError("parallelism degrees must be >= 1")
        product = pipeline_parallel * expert_parallel
        if len(compute_devices) % product != 0:
            raise ValueError(
                f"number of devices ({len(compute_devices)}) must be divisible by "
                f"the product of parallelism degrees ({product})"
            )
        if model.num_layers % pipeline_parallel != 0:
            raise ValueError(
                f"num_layers ({model.num_layers}) must be divisible by "
                f"pipeline_parallel ({pipeline_parallel})"
            )
        self.model = model
        self.devices = compute_devices
        self.pipeline_parallel = pipeline_parallel
        self.expert_parallel = expert_parallel
        self._layers_per_stage = model.num_layers // pipeline_parallel
        # Bytes moved per expert miss: that expert's routed weights across all
        # MoE layers (selection is shared, so they move together).
        self._moe_routed_bytes_per_miss = sum(
            ffn.routed_expert_params for ffn in model.moe_ffns()
        ) * model.param_dtype_bytes

    def _device_index(self, stage: int, ep_rank: int) -> int:
        """Grid device for a (pipeline stage, expert-parallel rank) pair."""

        return stage * self.expert_parallel + ep_rank

    def _stage_of_layer(self, layer_index: int | None) -> int:
        """Pipeline stage handling a layer; LM head goes to the last stage."""

        if layer_index is None:
            return self.pipeline_parallel - 1
        return layer_index // self._layers_per_stage

    def run(
        self,
        shards: list[WorkShard],
        expert_trace: list[GroupActivation] | None = None,
        expert_cache_capacity: int | None = None,
    ) -> EventSchedule:
        """Consolidate shards into events and time them.

        Within a group the pipeline stages run sequentially (single batch, no
        pipeline overlap) and the expert-parallel ranks of a stage run
        concurrently. When the devices have a second memory tier, the model is
        MoE, and an ``expert_trace`` is supplied, a data-transfer event for the
        routed experts streamed up from the second tier precedes each group's
        compute; each rank moves only the experts it owns (residency reuse keeps
        the working set small), and a shared second tier bounds the aggregate.
        """

        schedule = EventSchedule()
        clock = 0.0
        ep = self.expert_parallel

        two_tier = (
            expert_trace is not None
            and self.model.num_moe_layers > 0
            and self.devices[0].second_tier_memory is not None
        )
        caches: list[ExpertResidencyCache] = []
        trace_by_group: dict[int, GroupActivation] = {}
        shared_tier2 = False
        if two_tier:
            if self.pipeline_parallel != 1:
                raise NotImplementedError(
                    "two-tier expert movement is only supported with "
                    "pipeline_parallel == 1"
                )
            if expert_cache_capacity is None:
                raise ValueError(
                    "expert_cache_capacity is required for two-tier execution"
                )
            caches = [ExpertResidencyCache(expert_cache_capacity) for _ in range(ep)]
            trace_by_group = {g.group_index: g for g in expert_trace}
            tier2_ids = {id(self.devices[r].second_tier_memory) for r in range(ep)}
            shared_tier2 = ep > 1 and len(tier2_ids) == 1

        # Preserve group order; bucket shards by group, then by pipeline stage.
        groups: dict[int, list[WorkShard]] = {}
        order: list[int] = []
        for shard in shards:
            if shard.group_index not in groups:
                groups[shard.group_index] = []
                order.append(shard.group_index)
            groups[shard.group_index].append(shard)

        for group_index in order:
            group_shards = groups[group_index]

            if two_tier and group_index in trace_by_group:
                active = trace_by_group[group_index].active_experts
                rank_bytes = [
                    caches[r].access(frozenset(e for e in active if e % ep == r))
                    * self._moe_routed_bytes_per_miss
                    for r in range(ep)
                ]
                if any(rank_bytes):
                    transfer = self._build_transfer_event(
                        group_index, rank_bytes, shared_tier2,
                        group_shards[0].phase, clock,
                    )
                    clock = transfer.end
                    schedule.events.append(transfer)

            stage_shards: dict[int, list[WorkShard]] = {}
            for shard in group_shards:
                stage = self._stage_of_layer(shard.layer_index)
                stage_shards.setdefault(stage, []).append(shard)

            # Stages run sequentially; the ep ranks of a stage run concurrently.
            for stage in sorted(stage_shards):
                bucket = stage_shards[stage]
                stage_end = clock
                for ep_rank in range(ep):
                    device_index = self._device_index(stage, ep_rank)
                    event = self._build_event(
                        group_index, device_index, bucket, clock, divide=ep
                    )
                    stage_end = max(stage_end, event.end)
                    schedule.events.append(event)
                clock = stage_end

        return schedule

    def _build_transfer_event(
        self,
        group_index: int,
        rank_bytes: list[float],
        shared_tier2: bool,
        phase: str,
        start: float,
    ) -> ComputeEvent:
        """Group expert-movement event across the expert-parallel ranks.

        Ranks move their owned experts concurrently. With a private second tier
        per device the group time is the slowest rank; with a shared second tier
        (system NVM) the aggregate is also bounded by that tier's bandwidth.
        """

        ep = self.expert_parallel
        total_bytes = sum(rank_bytes)
        if shared_tier2:
            tier2_bw = self.devices[0].second_tier_memory.bandwidth_bytes_per_s
            tier1_floor = max(
                (
                    rank_bytes[r] / self.devices[r].first_tier_memory.bandwidth_bytes_per_s
                    for r in range(ep)
                    if rank_bytes[r] > 0
                ),
                default=0.0,
            )
            duration = max(total_bytes / tier2_bw, tier1_floor)
        else:
            duration = max(
                (
                    rank_bytes[r]
                    / min(
                        self.devices[r].first_tier_memory.bandwidth_bytes_per_s,
                        self.devices[r].second_tier_memory.bandwidth_bytes_per_s,
                    )
                    for r in range(ep)
                    if rank_bytes[r] > 0
                ),
                default=0.0,
            )
        return ComputeEvent(
            group_index=group_index,
            phase="transfer",
            device_index=0,
            flops=0.0,
            bytes_read=total_bytes,
            compute_time=0.0,
            bandwidth_time=duration,
            duration=duration,
            start=start,
            end=start + duration,
        )

    def _build_event(
        self,
        group_index: int,
        device_index: int,
        shards: list[WorkShard],
        start: float,
        divide: int = 1,
    ) -> ComputeEvent:
        dtypes = {s.flops_dtype_bytes for s in shards}
        if len(dtypes) != 1:
            raise ValueError(
                "cannot consolidate shards with differing flops dtypes "
                f"in one event: {sorted(dtypes)}"
            )
        dtype_bytes = next(iter(dtypes))
        device = self.devices[device_index]

        total_flops = sum(s.flops for s in shards) / divide
        total_bytes = sum(s.bytes_read for s in shards) / divide

        compute_time = total_flops / device.effective_flops(dtype_bytes)
        bandwidth_time = total_bytes / device.bandwidth_bytes_per_s
        duration = max(compute_time, bandwidth_time)

        return ComputeEvent(
            group_index=group_index,
            phase=shards[0].phase,
            device_index=device_index,
            flops=total_flops,
            bytes_read=total_bytes,
            compute_time=compute_time,
            bandwidth_time=bandwidth_time,
            duration=duration,
            start=start,
            end=start + duration,
        )
