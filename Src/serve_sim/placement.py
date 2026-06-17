"""Engine-slot placement: carve a system into device sets for concurrent batches.

A *batch* runs on a *device set* (an engine slot) -- a fixed slice of the
system's compute devices sized to the parallelism degree
(``pipeline_parallel x expert_parallel``). The datacenter can hold several engine
slots at once, so multiple batches can be in flight simultaneously: batches on
*disjoint* slots run independently (they share no device, so the resource arbiter
never makes them contend), while batches forced onto the same slot time-share it.

This module is the bookkeeping for that placement, independent of the
orchestrator: an :class:`EnginePool` partitions the compute devices into slots
and tracks which are busy. Because different models cannot share a batch, serving
several models concurrently means each occupies its own slot, with that model's
weights resident there. The pool models that residency: acquiring a slot for a
model prefers a free slot that already hosts it (no weight reload), and reports
when a reload would be required because the slot last hosted a different model.

Slots hold the actual :class:`~serve_sim.hardware.ComputeDevice` instances, so the
slices handed out preserve object identity -- which is exactly what the event
generator and arbiter key contention on.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import ComputeDevice


@dataclass(frozen=True)
class EngineSlot:
    """A fixed device set a batch can run on.

    Attributes:
        index: Position of the slot within its pool.
        devices: The compute-device instances making up the set (length equals
            the pool's parallelism degree).
    """

    index: int
    devices: tuple[ComputeDevice, ...]


@dataclass(frozen=True)
class Placement:
    """Result of acquiring a slot for a batch.

    Attributes:
        slot: The acquired engine slot.
        needs_weight_load: ``True`` when the slot last hosted a different model
            (or none), so this model's weights must be loaded before it can run;
            ``False`` when the model's weights were already resident.
    """

    slot: EngineSlot
    needs_weight_load: bool


class EnginePool:
    """Partition compute devices into engine slots and track their use.

    The pool splits ``compute_devices`` into ``len(devices) // degree`` contiguous
    slots of ``degree`` devices each; any remainder devices are left unused. Each
    slot is either free or busy, and remembers the last model placed on it so a
    re-acquisition for the same model avoids a weight reload.
    """

    def __init__(self, compute_devices: list[ComputeDevice], degree: int = 1) -> None:
        if degree < 1:
            raise ValueError("degree must be >= 1")
        if not compute_devices:
            raise ValueError("at least one compute device is required")
        num_slots = len(compute_devices) // degree
        if num_slots < 1:
            raise ValueError(
                f"{len(compute_devices)} devices cannot form a slot of degree {degree}"
            )
        self._degree = degree
        self._slots: list[EngineSlot] = [
            EngineSlot(i, tuple(compute_devices[i * degree:(i + 1) * degree]))
            for i in range(num_slots)
        ]
        self._busy: set[int] = set()
        self._resident: dict[int, object] = {}

    # --- introspection ------------------------------------------------------

    @property
    def degree(self) -> int:
        """Devices per slot (the parallelism degree)."""

        return self._degree

    @property
    def num_slots(self) -> int:
        """Total number of engine slots."""

        return len(self._slots)

    @property
    def slots(self) -> list[EngineSlot]:
        """All slots, in index order."""

        return list(self._slots)

    @property
    def free_count(self) -> int:
        return self.num_slots - len(self._busy)

    @property
    def busy_count(self) -> int:
        return len(self._busy)

    def is_busy(self, slot: EngineSlot) -> bool:
        self._check_owned(slot)
        return slot.index in self._busy

    def is_free(self, slot: EngineSlot) -> bool:
        return not self.is_busy(slot)

    def resident_model(self, slot: EngineSlot) -> object | None:
        """The model whose weights currently sit on ``slot`` (or ``None``)."""

        self._check_owned(slot)
        return self._resident.get(slot.index)

    # --- allocation ---------------------------------------------------------

    def acquire(self, model: object | None = None) -> Placement | None:
        """Reserve a free slot for ``model``; ``None`` if every slot is busy.

        When ``model`` is given, a free slot that already hosts that model is
        preferred (so its weights need not be reloaded); otherwise the
        lowest-index free slot is used and its residency is updated to ``model``.
        Passing ``model=None`` reserves a slot without touching residency.
        """

        free = [s for s in self._slots if s.index not in self._busy]
        if not free:
            return None

        slot = None
        if model is not None:
            slot = next((s for s in free if self._resident.get(s.index) is model), None)
        if slot is None:
            slot = free[0]

        needs_load = model is not None and self._resident.get(slot.index) is not model
        if model is not None:
            self._resident[slot.index] = model
        self._busy.add(slot.index)
        return Placement(slot=slot, needs_weight_load=needs_load)

    def release(self, slot: EngineSlot) -> None:
        """Return a busy slot to the pool (its resident model is kept)."""

        self._check_owned(slot)
        if slot.index not in self._busy:
            raise ValueError(f"slot {slot.index} is not currently busy")
        self._busy.discard(slot.index)

    # --- internal -----------------------------------------------------------

    def _check_owned(self, slot: EngineSlot) -> None:
        if not (0 <= slot.index < len(self._slots)) or self._slots[slot.index] is not slot:
            raise ValueError("slot does not belong to this pool")
