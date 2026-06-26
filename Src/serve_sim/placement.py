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
        evicted_model: The model whose weights were displaced from the slot by
            this placement (the slot's previous resident), or ``None`` when the
            slot was empty or already hosted this model.
    """

    slot: EngineSlot
    needs_weight_load: bool
    evicted_model: object | None = None


class EnginePool:
    """Partition compute devices into engine slots and track their use.

    The pool splits ``compute_devices`` into ``len(devices) // degree`` contiguous
    slots of ``degree`` devices each; any remainder devices are left unused. Each
    slot carries an occupancy *load* (how many batches currently run on it) and
    remembers the model last placed on it so a re-placement for the same model
    avoids a weight reload.

    Two allocation styles are offered. :meth:`acquire` is *exclusive*: it only
    takes a free (load-0) slot and returns ``None`` when every slot is busy.
    :meth:`place` is *time-sharing*: it prefers a free slot but, when none is
    free, overlays the batch onto the least-loaded slot (the resource arbiter
    then shares that device set between the co-located batches). Both bump the
    slot's load by one; :meth:`release` drops it by one.
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
        self._load: list[int] = [0] * num_slots
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
        return sum(1 for load in self._load if load == 0)

    @property
    def busy_count(self) -> int:
        return sum(1 for load in self._load if load > 0)

    def is_busy(self, slot: EngineSlot) -> bool:
        self._check_owned(slot)
        return self._load[slot.index] > 0

    def is_free(self, slot: EngineSlot) -> bool:
        return not self.is_busy(slot)

    def slot_load(self, slot: EngineSlot) -> int:
        """Number of batches currently placed on ``slot``."""

        self._check_owned(slot)
        return self._load[slot.index]

    def resident_model(self, slot: EngineSlot) -> object | None:
        """The model whose weights currently sit on ``slot`` (or ``None``)."""

        self._check_owned(slot)
        return self._resident.get(slot.index)

    # --- allocation ---------------------------------------------------------

    def acquire(self, model: object | None = None) -> Placement | None:
        """Exclusively reserve a free slot for ``model``; ``None`` if all busy.

        When ``model`` is given, a free slot that already hosts that model is
        preferred (so its weights need not be reloaded); otherwise the
        lowest-index free slot is used and its residency is updated to ``model``.
        Passing ``model=None`` reserves a slot without touching residency.
        """

        free = [s for s in self._slots if self._load[s.index] == 0]
        if not free:
            return None
        return self._place_on(self._pick(free, model), model)

    def place(self, model: object | None = None) -> Placement:
        """Place ``model`` on a slot, time-sharing when none is free.

        Prefers a free slot (with the same model-affinity rule as
        :meth:`acquire`); when every slot is busy, overlays onto the least-loaded
        slot so the batch co-runs with whatever is already there. Always returns
        a :class:`Placement`.
        """

        min_load = min(self._load)
        candidates = [s for s in self._slots if self._load[s.index] == min_load]
        return self._place_on(self._pick(candidates, model), model)

    def release(self, slot: EngineSlot) -> None:
        """Drop one batch from a slot (its resident model is kept)."""

        self._check_owned(slot)
        if self._load[slot.index] == 0:
            raise ValueError(f"slot {slot.index} is not currently busy")
        self._load[slot.index] -= 1

    def unplace(self, placement: Placement) -> None:
        """Reverse a :meth:`place`/:meth:`acquire` that is not being committed.

        Drops the load increment and, when the placement had loaded a new model,
        restores the slot's prior resident model -- so a batch that is refused
        admission (e.g. by memory back-pressure) leaves the pool exactly as it was
        before the placement.
        """

        slot = placement.slot
        self._check_owned(slot)
        if self._load[slot.index] == 0:
            raise ValueError(f"slot {slot.index} is not currently busy")
        self._load[slot.index] -= 1
        if placement.needs_weight_load:
            if placement.evicted_model is not None:
                self._resident[slot.index] = placement.evicted_model
            else:
                self._resident.pop(slot.index, None)

    def warm_start(self, models: list[object]) -> None:
        """Pre-mark slot residency so an initial placement needs no weight load.

        Distributes ``models`` (most-wanted first) across the slots round-robin
        and records each as the resident of its slot, without changing any slot's
        load. A later :meth:`acquire`/:meth:`place` for one of these models then
        finds a slot already hosting it (``needs_weight_load=False``), so the
        run skips the trivial cold weight transfer that would otherwise stream the
        weights in. With a single model every slot is pre-warmed; with more models
        than slots only the leading ``num_slots`` are preloaded and the rest cold
        load on demand. This is best-effort: any model the orchestrator later
        needs on a slot that does not host it still triggers a normal (re)load.
        """

        if not models:
            return
        for slot in self._slots:
            self._resident[slot.index] = models[slot.index % len(models)]

    # --- internal -----------------------------------------------------------

    def _pick(self, candidates: list[EngineSlot], model: object | None) -> EngineSlot:
        """Choose among equal-priority ``candidates``, preferring model affinity."""

        if model is not None:
            resident = next(
                (s for s in candidates if self._resident.get(s.index) is model), None
            )
            if resident is not None:
                return resident
        return candidates[0]

    def _place_on(self, slot: EngineSlot, model: object | None) -> Placement:
        prior = self._resident.get(slot.index)
        needs_load = model is not None and prior is not model
        evicted = prior if (needs_load and prior is not None) else None
        if model is not None:
            self._resident[slot.index] = model
        self._load[slot.index] += 1
        return Placement(
            slot=slot, needs_weight_load=needs_load, evicted_model=evicted
        )

    def _check_owned(self, slot: EngineSlot) -> None:
        if not (0 <= slot.index < len(self._slots)) or self._slots[slot.index] is not slot:
            raise ValueError("slot does not belong to this pool")
