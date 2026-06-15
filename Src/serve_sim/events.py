"""Event generation: turn work shards into timed compute events.

The event generator maps each work shard to a compute device, consolidates the
shards of one forward-pass group that land on the same device into a single
event (a roofline ``max`` of compute-bound and bandwidth-bound time), and lays
the events out sequentially to produce a makespan.

Pipeline parallelism partitions the layers across devices. For a single batch
there is no pipeline overlap, so latency is the sum of the stage events -- which
keeps the roofline well-defined and conserved across the partition. Expert
parallelism is accepted as a parameter but requires an MoE model (not yet
supported here). Inter-stage activation transfers are not modeled in this stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .hardware import ComputeDevice
from .model import Model
from .shards import WorkShard


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
        model: Model,
        compute_devices: list[ComputeDevice],
        pipeline_parallel: int = 1,
        expert_parallel: int = 1,
    ) -> None:
        if not compute_devices:
            raise ValueError("at least one compute device is required")
        if pipeline_parallel < 1 or expert_parallel < 1:
            raise ValueError("parallelism degrees must be >= 1")
        if expert_parallel != 1:
            raise NotImplementedError(
                "expert parallelism requires an MoE model (not supported yet)"
            )
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

    def _stage_of_layer(self, layer_index: int | None) -> int:
        """Pipeline stage handling a layer; LM head goes to the last stage."""

        if layer_index is None:
            return self.pipeline_parallel - 1
        return layer_index // self._layers_per_stage

    def run(self, shards: list[WorkShard]) -> EventSchedule:
        """Consolidate shards into events and time them sequentially."""

        schedule = EventSchedule()
        clock = 0.0

        # Preserve group order; within a group, order events by device/stage.
        groups: dict[int, list[WorkShard]] = {}
        order: list[int] = []
        for shard in shards:
            if shard.group_index not in groups:
                groups[shard.group_index] = []
                order.append(shard.group_index)
            groups[shard.group_index].append(shard)

        for group_index in order:
            per_device: dict[int, list[WorkShard]] = {}
            for shard in groups[group_index]:
                device_index = self._stage_of_layer(shard.layer_index)
                per_device.setdefault(device_index, []).append(shard)

            for device_index in sorted(per_device):
                bucket = per_device[device_index]
                event = self._build_event(group_index, device_index, bucket, clock)
                clock = event.end
                schedule.events.append(event)

        return schedule

    def _build_event(
        self,
        group_index: int,
        device_index: int,
        shards: list[WorkShard],
        start: float,
    ) -> ComputeEvent:
        dtypes = {s.flops_dtype_bytes for s in shards}
        if len(dtypes) != 1:
            raise ValueError(
                "cannot consolidate shards with differing flops dtypes "
                f"in one event: {sorted(dtypes)}"
            )
        dtype_bytes = next(iter(dtypes))
        device = self.devices[device_index]

        total_flops = sum(s.flops for s in shards)
        total_bytes = sum(s.bytes_read for s in shards)

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
