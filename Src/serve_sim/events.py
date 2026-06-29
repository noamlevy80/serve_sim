"""Event generation: turn work shards into timed compute events.

The event generator maps each work shard onto a grid of ``pipeline_parallel x
expert_parallel x tensor_parallel`` devices, consolidates the shards of one
forward-pass group that land on the same device into a single event (a roofline
``max`` of compute-bound and bandwidth-bound time), and lays the events out to
produce a makespan.

Pipeline parallelism partitions the layers across stages. For a single batch
there is no pipeline overlap, so the stages of a group run sequentially and
latency is the sum of the stage events -- which keeps the roofline well-defined
and conserved across the partition.

Expert parallelism partitions the routed experts across the ``expert_parallel``
devices of a stage (expert ``e`` lives on rank ``e % expert_parallel``). The
compute of a stage is split evenly across those ranks (balanced routing), so the
ranks run concurrently and an evenly balanced stage is ``expert_parallel`` times
faster. Each rank keeps its own LRU residency of the experts it owns; in a
two-tier system those experts are streamed up from the second tier on demand,
and when the second tier is a single shared device (a system NVM) its bandwidth
bounds the aggregate movement.

Tensor parallelism shards every tensor (and the per-rank compute) across the
``tensor_parallel`` devices that sit under each ``(stage, expert_rank)`` cell, so
a stage's work is divided evenly across all ``expert_parallel x tensor_parallel``
devices of the stage and runs that many times faster. Tensor parallelism is not
supported alongside two-tier expert movement.

**Communication collectives.** Sharding a forward pass forces the ranks to
exchange activations, and that exchange runs on the scale-up network (the
``scale_up_bandwidth_bytes_per_s`` / ``scale_up_latency_s`` the generator is
given; when no bandwidth is supplied, communication is not modeled and the
output is byte-identical to the pure-roofline path). Each collective is a
*barrier*: every participating rank is occupied for ``latency + volume /
bandwidth`` and the group cannot proceed until it completes. For a group of
``T`` tokens the per-layer activation tensor is ``A = T * hidden_size *
param_dtype_bytes``, and per pipeline stage:

* **Tensor parallelism** adds two ``all-reduce`` collectives per layer (after the
  attention and after the FFN sub-layer), each moving ``2*(tp-1)/tp * A`` across
  the ``tp`` ranks (``phase == "tp_comm"``).
* **Expert parallelism** adds two ``all-to-all`` collectives per MoE layer (the
  dispatch of tokens to their experts and the combine of the results), each
  moving ``(ep-1)/ep * A`` across the ``ep`` ranks (``phase == "ep_comm"``).
* **Pipeline parallelism** adds one point-to-point activation send of ``A`` from
  each stage to the next (``phase == "pp_comm"``).

These barriers are inserted after each stage's compute and serialize with it, so
a sharded forward pass costs its compute plus its collective time.

A group's compute may be preceded by a kernel-launch event: when a work shard
marks a kernel launch, each participating device waits its
``kernel_launch_latency`` before computing (devices with zero launch overhead
emit no such event).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .blocks import LayeredModel
from .device_memory import DeviceHbmResidency
from .hardware import ComputeDevice, MemoryDevice, dtype_compute_scale
from .shards import WorkShard
from .tiering import ExpertResidencyCache, GroupActivation


class _RankExpertResidency:
    """Per-ep-rank view over a device's persistent HBM residency.

    Adapts a shared :class:`DeviceHbmResidency` (which survives across batches and
    competes routed experts against retained KV) to the per-group
    ``access_detail(active) -> (missed, evicted)`` protocol the event generator
    uses for the legacy per-batch :class:`ExpertResidencyCache`. KV evicted by an
    expert admission is accumulated in :attr:`evicted_kv` for the orchestrator to
    spill to node memory.
    """

    def __init__(
        self, residency: DeviceHbmResidency, index_bytes: float, now: float
    ) -> None:
        self._residency = residency
        self._index_bytes = index_bytes
        self._now = now
        self.evicted_kv: list = []

    def access_detail(
        self, active: frozenset[int]
    ) -> tuple[list[int], list[int]]:
        missed, evicted_experts, evicted_kv = self._residency.access_experts(
            active, self._index_bytes, self._now
        )
        self.evicted_kv.extend(evicted_kv)
        return missed, evicted_experts



@dataclass(frozen=True)
class ComputeEvent:
    """A timed unit of compute on one device.

    Attributes:
        group_index: Originating forward-pass group.
        phase: ``"prefill"`` or ``"decode"``.
        device_index: Index into the event generator's device list.
        flops: Total consolidated FP16-equivalent FLOPs. Each shard's raw FLOPs
            are normalized by its dtype's compute-rate scale at consolidation, so
            the device always retires this work at ``peak_flops_fp16``.
        bytes_read: Total consolidated bytes read.
        compute_time: Compute-bound duration.
        bandwidth_time: Bandwidth-bound duration.
        duration: ``max(compute_time, bandwidth_time)``.
        start: Start time on the global clock.
        end: End time on the global clock.
        source_memory: For a transfer that streams from a memory not derivable
            from ``device_index`` (e.g. a weight load from the system input NVM),
            the source memory the arbiter should contend the bandwidth on; left
            ``None`` for compute and for transfers whose source is the device's
            own second/first tier.
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
    source_memory: MemoryDevice | None = None


@dataclass
class EventSchedule:
    """Result of event generation: ordered events and derived totals."""

    events: list[ComputeEvent] = field(default_factory=list)
    #: Total bytes of routed-expert weights fetched on demand (streaming).
    expert_fetch_bytes: float = 0.0
    #: Number of forward-pass groups that triggered an expert fetch.
    expert_fetch_groups: int = 0
    #: Total number of expert indices fetched (summed over groups and ranks).
    expert_experts_loaded: int = 0
    #: Total number of expert indices the residency LRU evicted.
    expert_evictions: int = 0
    #: KV residents evicted from a device's HBM by expert admissions (only under
    #: the shared global-LRU policy); the orchestrator spills these to node memory.
    expert_evicted_kv: list = field(default_factory=list)

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
        tensor_parallel: int = 1,
        event_random_factor_range: float = 0.0,
        rng: random.Random | None = None,
        scale_up_bandwidth_bytes_per_s: float | None = None,
        scale_up_latency_s: float = 0.0,
    ) -> None:
        model = LayeredModel.from_model(model)
        if not compute_devices:
            raise ValueError("at least one compute device is required")
        if pipeline_parallel < 1 or expert_parallel < 1 or tensor_parallel < 1:
            raise ValueError("parallelism degrees must be >= 1")
        if not 0.0 <= event_random_factor_range < 1.0:
            raise ValueError("event_random_factor_range must be in [0, 1)")
        if scale_up_bandwidth_bytes_per_s is not None and scale_up_bandwidth_bytes_per_s <= 0:
            raise ValueError("scale_up_bandwidth_bytes_per_s must be positive")
        if scale_up_latency_s < 0:
            raise ValueError("scale_up_latency_s must be non-negative")
        product = pipeline_parallel * expert_parallel * tensor_parallel
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
        self.tensor_parallel = tensor_parallel
        self.event_random_factor_range = event_random_factor_range
        self._rng = rng if rng is not None else random.Random()
        self._comm_bandwidth = scale_up_bandwidth_bytes_per_s
        self._comm_latency = scale_up_latency_s
        self._layers_per_stage = model.num_layers // pipeline_parallel
        # Bytes moved per expert miss: that expert's routed weights across all
        # MoE layers (selection is shared, so they move together).
        self._moe_routed_bytes_per_miss = sum(
            ffn.routed_expert_params for ffn in model.moe_ffns()
        ) * model.param_dtype_bytes

    @property
    def routed_expert_bytes_per_index(self) -> float:
        """HBM footprint of one routed-expert index (its weights across MoE layers).

        This is the model-wide cost of making one expert index resident; under
        tensor parallelism each device holds a ``1/tensor_parallel`` shard of it.
        """

        return self._moe_routed_bytes_per_miss

    def _device_index(self, stage: int, ep_rank: int, tp_rank: int = 0) -> int:
        """Grid device for a (pipeline stage, expert rank, tensor rank) triple.

        The ``pp x ep x tp`` devices are laid out stage-major, then expert-rank,
        then tensor-rank: ``stage * (ep * tp) + ep_rank * tp + tp_rank``.
        """

        return (
            stage * self.expert_parallel * self.tensor_parallel
            + ep_rank * self.tensor_parallel
            + tp_rank
        )

    def _stage_of_layer(self, layer_index: int | None) -> int:
        """Pipeline stage handling a layer; LM head goes to the last stage."""

        if layer_index is None:
            return self.pipeline_parallel - 1
        return layer_index // self._layers_per_stage

    def _time_scale(self) -> float:
        """Per-event random time multiplier ``1 + U(-range, range)``.

        Models system randomness: the calculated event time is scaled by one
        draw per event (``1.0`` when randomization is disabled). The scale is
        applied to the roofline times, not the FLOPs/bytes, so total work is
        conserved while the effective rate -- and hence the duration the arbiter
        rescales under contention -- carries the perturbation.
        """

        r = self.event_random_factor_range
        if r <= 0.0:
            return 1.0
        return 1.0 + self._rng.uniform(-r, r)

    def run(
        self,
        shards: list[WorkShard],
        expert_trace: list[GroupActivation] | None = None,
        expert_cache_capacity: int | None = None,
        expert_source: MemoryDevice | None = None,
        expert_fetch_latency: float = 0.0,
        expert_residency: list[DeviceHbmResidency] | None = None,
        expert_index_bytes: float | None = None,
        expert_now: float = 0.0,
    ) -> EventSchedule:
        """Consolidate shards into events and time them.

        Within a group the pipeline stages run sequentially (single batch, no
        pipeline overlap) and the expert-parallel ranks of a stage run
        concurrently. When the model is MoE and an ``expert_trace`` is supplied,
        a routed-expert fetch event (``phase == "expert_transfer"``) precedes each
        group whose working set is not already resident on the first tier; each
        rank moves only the experts it owns (residency reuse keeps the set small).

        The source of those experts is either ``expert_source`` -- a single shared
        memory (the home node's RAM) reached over a fabric whose one-way latency
        is ``expert_fetch_latency`` -- or, when ``expert_source`` is ``None``, each
        device's own ``second_tier_memory`` (the legacy per-device two-tier path,
        which requires ``pipeline_parallel == tensor_parallel == 1``). Expert
        indices are assumed identical across the whole model, so a fetch for an
        index moves that expert's weights from every MoE layer at once and the
        per-fetch byte count is independent of the pipeline/tensor split.

        When ``expert_residency`` is given (one persistent
        :class:`DeviceHbmResidency` per ep rank, sharing each device's HBM with
        retained KV under a device-first policy) the routed experts stay resident
        *across batches*, so a later batch on the same device reuses them instead
        of re-fetching; ``expert_index_bytes`` is the per-index HBM footprint and
        ``expert_now`` the recency stamp. Otherwise a fresh per-batch
        :class:`ExpertResidencyCache` of ``expert_cache_capacity`` indices is used
        (the legacy behaviour).
        """

        schedule = EventSchedule()
        clock = 0.0
        ep = self.expert_parallel
        tp = self.tensor_parallel
        divide = ep * tp

        streaming = (
            expert_trace is not None
            and self.model.num_moe_layers > 0
            and (expert_source is not None or self.devices[0].second_tier_memory is not None)
        )
        caches: list = []
        trace_by_group: dict[int, GroupActivation] = {}
        shared_tier2 = False
        if streaming:
            if expert_residency is None and expert_cache_capacity is None:
                raise ValueError(
                    "expert_cache_capacity is required for expert streaming"
                )
            if expert_source is None:
                # Legacy per-device second-tier path: the device layout below
                # assumes the ep ranks occupy devices 0..ep-1, which only holds
                # without a pipeline or tensor split.
                if self.pipeline_parallel != 1:
                    raise NotImplementedError(
                        "per-device second-tier expert movement is only supported "
                        "with pipeline_parallel == 1 (use expert_source instead)"
                    )
                if tp != 1:
                    raise NotImplementedError(
                        "per-device second-tier expert movement is only supported "
                        "with tensor_parallel == 1 (use expert_source instead)"
                    )
                tier2_ids = {id(self.devices[r].second_tier_memory) for r in range(ep)}
                shared_tier2 = ep > 1 and len(tier2_ids) == 1
            if expert_residency is not None:
                if expert_index_bytes is None:
                    raise ValueError(
                        "expert_index_bytes is required with expert_residency"
                    )
                if len(expert_residency) != ep:
                    raise ValueError(
                        "expert_residency must have one entry per expert-parallel "
                        f"rank ({ep}); got {len(expert_residency)}"
                    )
                caches = [
                    _RankExpertResidency(expert_residency[r], expert_index_bytes, expert_now)
                    for r in range(ep)
                ]
            else:
                caches = [ExpertResidencyCache(expert_cache_capacity) for _ in range(ep)]
            trace_by_group = {g.group_index: g for g in expert_trace}

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
            launch = any(s.kind == "kernel_launch" for s in group_shards)
            compute_shards = [s for s in group_shards if s.kind != "kernel_launch"]
            if not compute_shards:
                continue
            phase = compute_shards[0].phase

            if streaming and group_index in trace_by_group:
                active = trace_by_group[group_index].active_experts
                rank_misses: list[int] = []
                evicted_total = 0
                for r in range(ep):
                    missed, evicted = caches[r].access_detail(
                        frozenset(e for e in active if e % ep == r)
                    )
                    rank_misses.append(len(missed))
                    evicted_total += len(evicted)
                total_misses = sum(rank_misses)
                if total_misses:
                    if expert_source is not None:
                        total_bytes = total_misses * self._moe_routed_bytes_per_miss
                        transfer = self._build_shared_expert_transfer_event(
                            group_index, total_bytes, expert_source,
                            expert_fetch_latency, phase, clock,
                        )
                    else:
                        rank_bytes = [
                            m * self._moe_routed_bytes_per_miss for m in rank_misses
                        ]
                        transfer = self._build_transfer_event(
                            group_index, rank_bytes, shared_tier2, phase, clock,
                        )
                    clock = transfer.end
                    schedule.events.append(transfer)
                    schedule.expert_fetch_bytes += transfer.bytes_read
                    schedule.expert_fetch_groups += 1
                    schedule.expert_experts_loaded += total_misses
                schedule.expert_evictions += evicted_total

            stage_shards: dict[int, list[WorkShard]] = {}
            for shard in compute_shards:
                stage = self._stage_of_layer(shard.layer_index)
                stage_shards.setdefault(stage, []).append(shard)

            # Stages run sequentially; the ep*tp ranks of a stage run concurrently.
            ordered_stages = sorted(stage_shards)
            for stage_pos, stage in enumerate(ordered_stages):
                bucket = stage_shards[stage]
                # A new kernel is launched on each rank of this stage; they
                # launch concurrently, so the slowest rank gates the group.
                if launch:
                    latency = max(
                        self.devices[
                            self._device_index(stage, r, t)
                        ].kernel_launch_latency
                        for r in range(ep)
                        for t in range(tp)
                    )
                    if latency > 0:
                        event = self._build_kernel_launch_event(
                            group_index, self._device_index(stage, 0, 0), latency, clock
                        )
                        clock = event.end
                        schedule.events.append(event)
                stage_end = clock
                # Expert parallelism shards the routed experts across the ep
                # ranks; tensor parallelism shards every rank's tensors across
                # its tp ranks. Both run concurrently, so each of the ep*tp
                # devices of a stage does an even 1/(ep*tp) slice of the work.
                for ep_rank in range(ep):
                    for tp_rank in range(tp):
                        device_index = self._device_index(stage, ep_rank, tp_rank)
                        event = self._build_event(
                            group_index, device_index, bucket, clock, divide=divide
                        )
                        stage_end = max(stage_end, event.end)
                        schedule.events.append(event)
                clock = stage_end
                # Sharding the stage forces the ranks to exchange activations
                # over the scale-up network: a tensor-parallel all-reduce after
                # every sub-layer, an expert-parallel all-to-all around every MoE
                # layer, and a point-to-point hand-off of the activations to the
                # next pipeline stage. Each is a barrier that serializes after the
                # stage's compute.
                clock = self._emit_stage_collectives(
                    schedule, group_index, stage, bucket, clock,
                    last_stage=(stage_pos == len(ordered_stages) - 1),
                )

        if expert_residency is not None:
            for cache in caches:
                schedule.expert_evicted_kv.extend(cache.evicted_kv)

        return schedule

    def _build_kernel_launch_event(
        self,
        group_index: int,
        device_index: int,
        latency: float,
        start: float,
    ) -> ComputeEvent:
        """Fixed wait for launching a kernel on a device (no FLOPs/bytes)."""

        latency = latency * self._time_scale()
        return ComputeEvent(
            group_index=group_index,
            phase="kernel_launch",
            device_index=device_index,
            flops=0.0,
            bytes_read=0.0,
            compute_time=0.0,
            bandwidth_time=0.0,
            duration=latency,
            start=start,
            end=start + latency,
        )


    def _emit_stage_collectives(
        self,
        schedule: EventSchedule,
        group_index: int,
        stage: int,
        bucket: list[WorkShard],
        start: float,
        last_stage: bool,
    ) -> float:
        """Append the comm-collective barriers a sharded stage incurs.

        Returns the clock after the barriers. When no scale-up bandwidth was
        configured -- or the stage is unsharded -- nothing is emitted and the
        clock is returned unchanged (byte-identical to the pure-roofline path).

        The activation tensor exchanged by a layer is ``A = tokens * hidden_size
        * param_dtype_bytes``. Tensor parallelism adds two all-reduces per layer,
        expert parallelism two all-to-alls per MoE layer, and pipeline
        parallelism one point-to-point send of ``A`` to the next stage. Each
        barrier occupies every participating rank for its duration; one random
        ``_time_scale`` draw is shared across a barrier's per-rank events so they
        stay aligned.
        """

        if self._comm_bandwidth is None:
            return start
        ep = self.expert_parallel
        tp = self.tensor_parallel
        layer_shards = [s for s in bucket if s.kind == "layer"]
        tokens = max((s.tokens for s in bucket), default=0)
        activation_bytes = tokens * self.model.hidden_size * self.model.param_dtype_bytes
        if activation_bytes <= 0:
            return start
        clock = start
        if tp > 1 and layer_shards:
            # Two all-reduces per layer (post-attention, post-FFN).
            per = self._allreduce_time(activation_bytes, tp)
            total = 2 * len(layer_shards) * per * self._time_scale()
            clock = self._emit_barrier(
                schedule, group_index, "tp_comm", stage, total, clock
            )
        if ep > 1:
            n_moe = sum(
                1 for s in layer_shards if self.model.is_moe_layer(s.layer_index)
            )
            if n_moe:
                # Two all-to-alls per MoE layer (token dispatch, result combine).
                per = self._alltoall_time(activation_bytes, ep)
                total = 2 * n_moe * per * self._time_scale()
                clock = self._emit_barrier(
                    schedule, group_index, "ep_comm", stage, total, clock
                )
        if not last_stage:
            # Hand the stage's activations to the next pipeline stage.
            total = self._p2p_time(activation_bytes) * self._time_scale()
            clock = self._emit_barrier(
                schedule, group_index, "pp_comm", stage + 1, total, clock
            )
        return clock

    def _emit_barrier(
        self,
        schedule: EventSchedule,
        group_index: int,
        phase: str,
        stage: int,
        duration: float,
        start: float,
    ) -> float:
        """Emit a comm barrier of ``duration`` on every rank of ``stage``.

        All ``ep*tp`` ranks of the stage are occupied for the same window (a
        collective completes only when every participant has), so the per-rank
        events share one start and end; the clock advances by ``duration`` once.
        """

        if duration <= 0:
            return start
        end = start + duration
        for ep_rank in range(self.expert_parallel):
            for tp_rank in range(self.tensor_parallel):
                schedule.events.append(
                    ComputeEvent(
                        group_index=group_index,
                        phase=phase,
                        device_index=self._device_index(stage, ep_rank, tp_rank),
                        flops=0.0,
                        bytes_read=0.0,
                        compute_time=0.0,
                        bandwidth_time=0.0,
                        duration=duration,
                        start=start,
                        end=end,
                    )
                )
        return end

    def _allreduce_time(self, num_bytes: float, degree: int) -> float:
        """Ring all-reduce wall time over ``degree`` ranks: ``lat + 2(d-1)/d * B / bw``."""

        if degree <= 1 or self._comm_bandwidth is None:
            return 0.0
        volume = 2.0 * (degree - 1) / degree * num_bytes
        return self._comm_latency + volume / self._comm_bandwidth

    def _alltoall_time(self, num_bytes: float, degree: int) -> float:
        """All-to-all wall time over ``degree`` ranks: ``lat + (d-1)/d * B / bw``."""

        if degree <= 1 or self._comm_bandwidth is None:
            return 0.0
        volume = (degree - 1) / degree * num_bytes
        return self._comm_latency + volume / self._comm_bandwidth

    def _p2p_time(self, num_bytes: float) -> float:
        """Point-to-point send wall time: ``lat + B / bw``."""

        if self._comm_bandwidth is None:
            return 0.0
        return self._comm_latency + num_bytes / self._comm_bandwidth


    def _build_transfer_event(
        self,
        group_index: int,
        rank_bytes: list[float],
        shared_tier2: bool,
        phase: str,
        start: float,
    ) -> ComputeEvent:
        """Legacy per-device expert-movement event across the ep ranks.

        Ranks move their owned experts concurrently from their own second tier.
        With a private second tier per device the group time is the slowest rank;
        with a shared second tier the aggregate is also bounded by that tier's
        bandwidth. The source bandwidth the arbiter contends on is the first
        device's second tier (representative of the shared/per-rank tier).
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
        duration = duration * self._time_scale()
        return ComputeEvent(
            group_index=group_index,
            phase="expert_transfer",
            device_index=0,
            flops=0.0,
            bytes_read=total_bytes,
            compute_time=0.0,
            bandwidth_time=duration,
            duration=duration,
            start=start,
            end=start + duration,
            source_memory=self.devices[0].second_tier_memory,
        )

    def _build_shared_expert_transfer_event(
        self,
        group_index: int,
        total_bytes: float,
        source: MemoryDevice,
        latency: float,
        phase: str,
        start: float,
    ) -> ComputeEvent:
        """Expert-fetch event from a single shared source (the home node RAM).

        Every rank's missed experts stream from the same node memory, so that
        memory's bandwidth bounds the whole group's movement and the fabric's
        one-way ``latency`` is paid once. The arbiter contends ``total_bytes`` on
        ``source`` so concurrent jobs fetching from the same node RAM share it.
        """

        duration = (latency + total_bytes / source.bandwidth_bytes_per_s) * self._time_scale()
        return ComputeEvent(
            group_index=group_index,
            phase="expert_transfer",
            device_index=0,
            flops=0.0,
            bytes_read=total_bytes,
            compute_time=0.0,
            bandwidth_time=duration,
            duration=duration,
            start=start,
            end=start + duration,
            source_memory=source,
        )


    def _build_event(
        self,
        group_index: int,
        device_index: int,
        shards: list[WorkShard],
        start: float,
        divide: int = 1,
    ) -> ComputeEvent:
        device = self.devices[device_index]

        # All compute accounting is in FP16-equivalent FLOPs: a shard's raw FLOPs
        # are normalized by its dtype's compute-rate scale (e.g. FP8 runs at 2x,
        # so its FP16-equivalent work is half the raw FLOPs). Shards of differing
        # dtypes can therefore be consolidated -- each is normalized on the spot --
        # and the device retires the result at ``peak_flops_fp16`` regardless of
        # dtype, so the unit never has to be tracked downstream.
        total_flops = sum(
            s.flops / dtype_compute_scale(s.flops_dtype_bytes) for s in shards
        ) / divide
        total_bytes = sum(s.bytes_read for s in shards) / divide

        scale = self._time_scale()
        compute_time = total_flops / device.peak_flops_fp16 * scale
        bandwidth_time = total_bytes / device.bandwidth_bytes_per_s * scale
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
