"""Placement tests: carving a system into engine slots for concurrent batches.

An :class:`EnginePool` partitions compute devices into fixed device sets (engine
slots) of the parallelism degree, hands them out to batches, and tracks which are
busy and which model's weights are resident on each. These tests check the
partition geometry, the free/busy bookkeeping, that handed-out slots are disjoint
by device identity (so the arbiter keeps their batches independent), and the
model-affinity / weight-reload accounting.
"""

from __future__ import annotations

import pytest

from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.model import toy_model
from serve_sim.placement import EnginePool, EngineSlot, Placement


# --- helpers --------------------------------------------------------------------


def make_device(name):
    mem = MemoryDevice(f"{name}-mem", capacity_bytes=80e9, bandwidth_bytes_per_s=1e12)
    return ComputeDevice(name, peak_flops_fp16=100e12, first_tier_memory=mem)


def make_devices(n):
    return [make_device(f"g{i}") for i in range(n)]


# --- partition geometry ---------------------------------------------------------


def test_degree_one_gives_one_slot_per_device():
    devices = make_devices(4)
    pool = EnginePool(devices, degree=1)
    assert pool.num_slots == 4
    assert pool.degree == 1
    for i, slot in enumerate(pool.slots):
        assert slot.index == i
        assert slot.devices == (devices[i],)


def test_partition_groups_contiguous_devices():
    devices = make_devices(8)
    pool = EnginePool(devices, degree=2)
    assert pool.num_slots == 4
    assert [s.devices for s in pool.slots] == [
        (devices[0], devices[1]),
        (devices[2], devices[3]),
        (devices[4], devices[5]),
        (devices[6], devices[7]),
    ]


def test_remainder_devices_are_unused():
    devices = make_devices(5)
    pool = EnginePool(devices, degree=2)
    # 5 // 2 == 2 slots covering the first 4 devices; the 5th is left out.
    assert pool.num_slots == 2
    used = {id(d) for s in pool.slots for d in s.devices}
    assert id(devices[4]) not in used
    assert len(used) == 4


def test_slots_preserve_device_identity():
    devices = make_devices(4)
    pool = EnginePool(devices, degree=2)
    flat = [d for s in pool.slots for d in s.devices]
    for original, placed in zip(devices, flat):
        assert placed is original


# --- validation -----------------------------------------------------------------


def test_rejects_degree_below_one():
    with pytest.raises(ValueError):
        EnginePool(make_devices(2), degree=0)


def test_rejects_empty_devices():
    with pytest.raises(ValueError):
        EnginePool([], degree=1)


def test_rejects_too_few_devices_for_degree():
    with pytest.raises(ValueError):
        EnginePool(make_devices(3), degree=4)


# --- acquire / release bookkeeping ----------------------------------------------


def test_acquire_marks_slot_busy():
    pool = EnginePool(make_devices(2), degree=1)
    assert pool.free_count == 2 and pool.busy_count == 0
    placement = pool.acquire()
    assert isinstance(placement, Placement)
    assert pool.is_busy(placement.slot)
    assert pool.free_count == 1 and pool.busy_count == 1


def test_release_frees_slot():
    pool = EnginePool(make_devices(2), degree=1)
    placement = pool.acquire()
    pool.release(placement.slot)
    assert pool.is_free(placement.slot)
    assert pool.free_count == 2 and pool.busy_count == 0


def test_acquire_uses_lowest_free_index():
    pool = EnginePool(make_devices(3), degree=1)
    a = pool.acquire()
    b = pool.acquire()
    assert a.slot.index == 0
    assert b.slot.index == 1


def test_acquire_returns_none_when_full():
    pool = EnginePool(make_devices(2), degree=1)
    pool.acquire()
    pool.acquire()
    assert pool.acquire() is None


def test_released_slot_can_be_reacquired():
    pool = EnginePool(make_devices(1), degree=1)
    first = pool.acquire()
    assert pool.acquire() is None
    pool.release(first.slot)
    second = pool.acquire()
    assert second is not None
    assert second.slot.index == first.slot.index


# --- disjoint device sets -------------------------------------------------------


def test_concurrent_slots_share_no_device():
    devices = make_devices(8)
    pool = EnginePool(devices, degree=2)
    a = pool.acquire()
    b = pool.acquire()
    ids_a = {id(d) for d in a.slot.devices}
    ids_b = {id(d) for d in b.slot.devices}
    assert ids_a.isdisjoint(ids_b)


# --- model affinity / weight residency ------------------------------------------


def test_first_acquire_for_model_needs_weight_load():
    model = toy_model()
    pool = EnginePool(make_devices(1), degree=1)
    placement = pool.acquire(model)
    assert placement.needs_weight_load is True
    assert pool.resident_model(placement.slot) is model


def test_reacquire_same_model_skips_weight_load():
    model = toy_model()
    pool = EnginePool(make_devices(1), degree=1)
    first = pool.acquire(model)
    pool.release(first.slot)
    second = pool.acquire(model)
    assert second.needs_weight_load is False
    assert second.slot.index == first.slot.index


def test_acquire_prefers_free_slot_already_hosting_model():
    model_a = toy_model(name="a")
    model_b = toy_model(name="b")
    pool = EnginePool(make_devices(2), degree=1)

    a = pool.acquire(model_a)  # slot 0 hosts A
    b = pool.acquire(model_b)  # slot 1 hosts B
    pool.release(a.slot)
    pool.release(b.slot)

    # Re-acquiring B should reuse slot 1 (resident B) over the lower-index slot 0.
    again = pool.acquire(model_b)
    assert again.slot.index == b.slot.index
    assert again.needs_weight_load is False


def test_reusing_slot_for_different_model_needs_load():
    model_a = toy_model(name="a")
    model_b = toy_model(name="b")
    pool = EnginePool(make_devices(1), degree=1)
    first = pool.acquire(model_a)
    pool.release(first.slot)
    second = pool.acquire(model_b)
    assert second.slot.index == first.slot.index
    assert second.needs_weight_load is True
    assert pool.resident_model(second.slot) is model_b


def test_acquire_without_model_does_not_track_residency():
    pool = EnginePool(make_devices(1), degree=1)
    placement = pool.acquire()
    assert placement.needs_weight_load is False
    assert pool.resident_model(placement.slot) is None


def test_different_models_occupy_distinct_slots_concurrently():
    model_a = toy_model(name="a")
    model_b = toy_model(name="b")
    pool = EnginePool(make_devices(4), degree=2)
    a = pool.acquire(model_a)
    b = pool.acquire(model_b)
    assert a.slot.index != b.slot.index
    assert {id(d) for d in a.slot.devices}.isdisjoint({id(d) for d in b.slot.devices})
    assert pool.resident_model(a.slot) is model_a
    assert pool.resident_model(b.slot) is model_b


# --- guard rails ----------------------------------------------------------------


def test_releasing_a_free_slot_raises():
    pool = EnginePool(make_devices(1), degree=1)
    slot = pool.slots[0]
    with pytest.raises(ValueError):
        pool.release(slot)


def test_double_release_raises():
    pool = EnginePool(make_devices(1), degree=1)
    placement = pool.acquire()
    pool.release(placement.slot)
    with pytest.raises(ValueError):
        pool.release(placement.slot)


def test_foreign_slot_is_rejected():
    pool = EnginePool(make_devices(2), degree=1)
    foreign = EngineSlot(0, (make_device("x"),))
    with pytest.raises(ValueError):
        pool.is_busy(foreign)
    with pytest.raises(ValueError):
        pool.release(foreign)
