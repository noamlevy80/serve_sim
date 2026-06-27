"""Device-first KV residency: retain KV on the serving device's HBM, reuse it in
place (same device) or device-to-device, and spill to node memory only under HBM
pressure. Exercises the two device-first policies (``global_lru`` / ``partitioned``)
on top of the tier-aware :class:`~serve_sim.kv_store.KVCacheManager`.
"""

from __future__ import annotations

import pytest

from serve_sim.device_memory import DeviceHbmResidency, MemoryPolicy
from serve_sim.kv_store import KVCacheManager
from serve_sim.model import toy_model
from serve_sim.orchestrator import Simulator, StrategyConfig
from serve_sim.pdd import context_kv_bytes
from serve_sim.placement import EngineSlot

from test_global_kv_cache import (
    SYSTEM,
    USER,
    make_kv_system,
    request_from_messages,
    tracker_from_messages,
)


def make_hbm(system, policy=MemoryPolicy.GLOBAL_LRU, capacity=None, frac=0.5):
    hbm = {}
    for device in system.compute_devices:
        cap = capacity if capacity is not None else device.first_tier_memory.capacity_bytes
        hbm[id(device)] = DeviceHbmResidency(cap, policy, frac)
    return hbm


# --- manager: device retention --------------------------------------------------


def test_store_retains_kv_on_device_hbm():
    model = toy_model()
    system = make_kv_system([80e9])
    device = system.compute_devices[0]
    manager = KVCacheManager(system, "global_lru", make_hbm(system))
    tracker = tracker_from_messages([dict(SYSTEM), dict(USER)])

    res = manager.store(
        model, tracker, 0, 0,
        tracker.prompt_tokens + tracker.output_tokens, now=0.0, device=device,
    )

    assert res.memory is device.first_tier_memory
    assert res.spilled == () and res.evicted == ()
    assert manager.entries[0].device is device
    # The node floating pool is untouched -- KV stays on the device.
    assert manager.used_bytes(system.nodes[0].node_memory) == 0.0


def test_lookup_returns_device_resident_entry():
    model = toy_model()
    system = make_kv_system([80e9])
    device = system.compute_devices[0]
    manager = KVCacheManager(system, "global_lru", make_hbm(system))
    stored = tracker_from_messages([dict(SYSTEM), dict(USER)])
    manager.store(model, stored, 3, 0,
                  stored.prompt_tokens + stored.output_tokens, now=0.0, device=device)

    incoming = tracker_from_messages(
        [dict(SYSTEM), dict(USER),
         {"role": "assistant", "content": "different continuation entirely"}]
    )
    match = manager.lookup(model, incoming, now=1.0)

    assert match is not None
    assert match.entry.workload_id == 3
    assert match.entry.device is device


def test_hbm_pressure_spills_lru_to_node():
    model = toy_model()
    context_tokens = 64
    entry_bytes = float(context_kv_bytes(model, context_tokens))
    system = make_kv_system([entry_bytes * 4])  # node has ample room
    device = system.compute_devices[0]
    # Device KV region holds only one entry, so the second store spills the first.
    manager = KVCacheManager(
        system, "global_lru", make_hbm(system, capacity=entry_bytes * 1.5)
    )
    t0 = tracker_from_messages([{"role": "system", "content": "alpha session one here"}])
    t1 = tracker_from_messages([{"role": "system", "content": "beta session two here"}])
    manager.store(model, t0, 0, 0, context_tokens, now=0.0, device=device)
    res = manager.store(model, t1, 1, 0, context_tokens, now=1.0, device=device)

    assert [e.workload_id for e in res.spilled] == [0]
    assert res.spilled[0].device is None
    assert res.spilled[0].memory is system.nodes[0].node_memory
    by_id = {e.workload_id: e for e in manager.entries}
    assert by_id[1].device is device          # newest retained on HBM
    assert by_id[0].device is None            # evicted one spilled to node
    assert manager.used_bytes(system.nodes[0].node_memory) == pytest.approx(entry_bytes)


def test_partitioned_kv_region_too_small_falls_back_to_node():
    model = toy_model()
    context_tokens = 64
    entry_bytes = float(context_kv_bytes(model, context_tokens))
    system = make_kv_system([entry_bytes * 4])
    device = system.compute_devices[0]
    # KV sub-region is 30% of a region only 1.0x the entry -> cannot hold a block.
    manager = KVCacheManager(
        system, "partitioned",
        make_hbm(system, policy=MemoryPolicy.PARTITIONED, capacity=entry_bytes, frac=0.3),
    )
    tracker = tracker_from_messages([{"role": "system", "content": "alpha session one"}])
    res = manager.store(model, tracker, 0, 0, context_tokens, now=0.0, device=device)

    assert res.memory is system.nodes[0].node_memory
    assert manager.entries[0].device is None


def test_storing_kv_evicting_a_shared_pool_expert_does_not_crash():
    # Under GLOBAL_LRU experts and retained KV share one device pool, so storing
    # a completed sequence's KV can evict a resident expert. The expert carries no
    # stored KV entry; it must be skipped by the spill path rather than have its
    # integer index unpacked as a (workload_id, turn_index) key.
    model = toy_model()
    context_tokens = 64
    entry_bytes = float(context_kv_bytes(model, context_tokens))
    system = make_kv_system([entry_bytes * 4])
    device = system.compute_devices[0]
    hbm = make_hbm(system, capacity=entry_bytes * 1.5)  # holds ~one block
    manager = KVCacheManager(system, "global_lru", hbm)

    residency = hbm[id(device)]
    assert residency.admit_expert(0, entry_bytes, now=0.0) == []
    assert residency.expert_resident(0)

    tracker = tracker_from_messages([dict(SYSTEM), dict(USER)])
    res = manager.store(model, tracker, 0, 0, context_tokens, now=1.0, device=device)

    assert res.memory is device.first_tier_memory   # KV retained on the device
    assert res.spilled == ()                         # nothing migrated for the expert
    assert not residency.expert_resident(0)          # the LRU expert was evicted
    assert manager.entries[0].device is device


# --- orchestrator: fetch routing ------------------------------------------------


def test_fetch_skipped_when_kv_on_same_slot_device():
    system = make_kv_system([80e9, 80e9])
    sim = Simulator(system, StrategyConfig(memory_policy="global_lru"))
    device_a = system.compute_devices[0]
    slot = EngineSlot(0, (device_a,))
    # The prefix already lives on device_a, the serving device -> no fetch event.
    fetches = [(device_a.first_tier_memory, 4096.0, device_a)]

    assert sim._kv_fetch_events(slot, fetches) == []


def test_fetch_builds_device_to_device_transfer():
    system = make_kv_system([80e9, 80e9])
    sim = Simulator(system, StrategyConfig(memory_policy="global_lru"))
    device_a, device_b = system.compute_devices
    slot = EngineSlot(0, (device_b,))
    # The prefix lives on device_a but the batch serves on device_b -> one
    # direct device-to-device transfer sourced from device_a's HBM.
    fetches = [(device_a.first_tier_memory, 4096.0, device_a)]

    events = sim._kv_fetch_events(slot, fetches)

    assert len(events) == 1
    assert events[0].source_memory is device_a.first_tier_memory
    assert events[0].phase == "transfer"


# --- orchestrator: end-to-end ---------------------------------------------------


def test_e2e_same_device_reuse_has_no_node_offload():
    model = toy_model()
    system = make_kv_system([80e9])  # single device + single node
    first = request_from_messages(0, model, [dict(SYSTEM), dict(USER)], workload_id=0)
    second = request_from_messages(
        1, model,
        [dict(SYSTEM), dict(USER),
         {"role": "assistant", "content": "second conversation tail differs"}],
        workload_id=1, arrival_time=1000.0,
    )

    result = Simulator(
        system, StrategyConfig(max_batch_size=1, memory_policy="global_lru")
    ).run([first, second])

    # KV is retained on the device's HBM, so nothing is offloaded to node memory.
    node_offloads = [
        d for d in result.decisions
        if d.kind == "kv_transfer" and d.devices == ("node-0",)
    ]
    assert node_offloads == []
    # The cross-conversation reuse is still recorded.
    reuse = [d for d in result.decisions
             if d.kind == "kv_reuse" and d.workload_id == 1]
    assert reuse and reuse[0].source_workload_id == 0
    # And it needs no fetch transfer (the prefix is already on the serving device).
    fetches = [d for d in result.decisions if d.kind == "kv_transfer"
               and d.workload_id == 1 and d.source_workload_id == 0]
    assert fetches == []


def test_e2e_node_first_still_offloads():
    model = toy_model()
    system = make_kv_system([80e9])
    req = request_from_messages(0, model, [dict(SYSTEM), dict(USER)], workload_id=0)

    result = Simulator(
        system, StrategyConfig(max_batch_size=1, memory_policy="node_first")
    ).run([req])

    offloads = [d for d in result.decisions
                if d.kind == "kv_transfer" and d.devices == ("node-0",)]
    assert offloads, "node_first must still offload completed KV to node memory"
