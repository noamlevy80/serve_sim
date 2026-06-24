"""Persistent, device-first routed-expert residency.

Under a device-first memory policy the routed experts stay resident on each ep
rank's HBM *across batches* (sharing the pool with retained KV), so a later batch
on the same device reuses them instead of re-fetching. These tests cover the
:class:`~serve_sim.device_memory.DeviceHbmResidency` expert API, the event
generator's persistent path, and the end-to-end orchestrator behaviour.
"""

from __future__ import annotations

from serve_sim.device_memory import DeviceHbmResidency, MemoryPolicy
from serve_sim.events import EventGenerator
from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.model import toy_moe_model
from serve_sim.orchestrator import Simulator, StrategyConfig
from serve_sim.shards import WorkShardGenerator
from serve_sim.tiering import build_activation_trace
from serve_sim.tracker import SequenceWork

from test_global_kv_cache import (
    SYSTEM,
    USER,
    make_kv_system,
    request_from_messages,
)


def _device(name="gpu", tier1_cap=80e9):
    tier1 = MemoryDevice(f"{name}-hbm", capacity_bytes=tier1_cap, bandwidth_bytes_per_s=2e12)
    return ComputeDevice(name, peak_flops_fp16=100e12, first_tier_memory=tier1)


def _home_ram():
    return MemoryDevice("home-ram", capacity_bytes=1e12, bandwidth_bytes_per_s=1e12)


# --- DeviceHbmResidency.access_experts ------------------------------------------


def test_access_experts_first_miss_then_resident():
    res = DeviceHbmResidency(1e9, MemoryPolicy.GLOBAL_LRU)
    missed, evicted_experts, evicted_kv = res.access_experts(
        frozenset({0, 1, 2}), index_bytes=10.0, now=0.0
    )
    assert sorted(missed) == [0, 1, 2]
    assert evicted_experts == [] and evicted_kv == []
    # A second access of the same set hits entirely -- the experts stayed resident.
    missed2, _, _ = res.access_experts(frozenset({0, 1, 2}), index_bytes=10.0, now=1.0)
    assert missed2 == []


def test_access_experts_protects_active_set():
    # Capacity holds exactly two indices; the active set of three must not evict
    # its own members mid-group (it stays over capacity instead).
    res = DeviceHbmResidency(20.0, MemoryPolicy.GLOBAL_LRU)
    missed, evicted_experts, _ = res.access_experts(
        frozenset({0, 1, 2}), index_bytes=10.0, now=0.0
    )
    assert sorted(missed) == [0, 1, 2]
    assert all(res.expert_resident(i) for i in (0, 1, 2))


def test_global_lru_expert_admission_evicts_lru_kv():
    res = DeviceHbmResidency(20.0, MemoryPolicy.GLOBAL_LRU)
    res.admit_kv((0, 0), num_bytes=10.0, now=-1.0)  # oldest resident
    _, _, evicted_kv = res.access_experts(
        frozenset({0, 1}), index_bytes=10.0, now=0.0
    )
    # Admitting two experts (20 bytes) into a 20-byte shared pool evicts the KV.
    assert [r.key for r in evicted_kv] == [("kv", (0, 0))]
    assert not res.kv_resident((0, 0))


def test_partitioned_experts_never_evict_kv():
    # Separate sub-regions: experts cannot touch the KV region.
    res = DeviceHbmResidency(80.0, MemoryPolicy.PARTITIONED, kv_fraction=0.5)
    res.admit_kv((0, 0), num_bytes=10.0, now=-1.0)
    _, _, evicted_kv = res.access_experts(
        frozenset({0, 1, 2}), index_bytes=10.0, now=0.0
    )
    assert evicted_kv == []
    assert res.kv_resident((0, 0))


def test_clear_experts_keeps_kv():
    res = DeviceHbmResidency(1e9, MemoryPolicy.GLOBAL_LRU)
    res.admit_kv((0, 0), num_bytes=10.0, now=0.0)
    res.access_experts(frozenset({0, 1}), index_bytes=10.0, now=1.0)
    res.clear_experts()
    assert not res.expert_resident(0) and not res.expert_resident(1)
    assert res.kv_resident((0, 0))


# --- event generator: persistence across run() calls ----------------------------


def _moe_run_inputs(seed=0):
    model = toy_moe_model(num_layers=2)
    work = [SequenceWork(0, 16, 4)]
    shards = WorkShardGenerator(model).generate(work)
    trace = build_activation_trace(model, work, seed=seed)
    return model, shards, trace


def test_residency_persists_experts_between_runs():
    model, shards, trace = _moe_run_inputs()
    gen = EventGenerator(model, [_device()])
    src = _home_ram()
    res = DeviceHbmResidency(80e9, MemoryPolicy.GLOBAL_LRU)
    idx = gen.routed_expert_bytes_per_index

    first = gen.run(
        shards, expert_trace=trace, expert_source=src,
        expert_residency=[res], expert_index_bytes=idx, expert_now=0.0,
    )
    second = gen.run(
        shards, expert_trace=trace, expert_source=src,
        expert_residency=[res], expert_index_bytes=idx, expert_now=1.0,
    )

    assert first.expert_experts_loaded > 0
    # The warm residency means the identical second batch fetches nothing.
    assert second.expert_experts_loaded == 0


def test_fresh_cache_reloads_every_run():
    model, shards, trace = _moe_run_inputs()
    gen = EventGenerator(model, [_device()])
    src = _home_ram()

    first = gen.run(
        shards, expert_trace=trace, expert_source=src, expert_cache_capacity=64,
    )
    second = gen.run(
        shards, expert_trace=trace, expert_source=src, expert_cache_capacity=64,
    )

    # Without a persistent residency each run starts cold and reloads identically.
    assert second.expert_experts_loaded == first.expert_experts_loaded > 0


def test_run_reports_kv_spilled_by_expert_admission():
    model, shards, trace = _moe_run_inputs()
    gen = EventGenerator(model, [_device()])
    src = _home_ram()
    idx = gen.routed_expert_bytes_per_index
    peak = max((len(g.active_experts) for g in trace), default=1)
    # A shared pool just big enough for the working set; a pre-resident KV block
    # is the least-recently-used, so the first full group evicts it.
    res = DeviceHbmResidency(idx * peak, MemoryPolicy.GLOBAL_LRU)
    res.admit_kv((7, 0), num_bytes=idx, now=-1.0)

    schedule = gen.run(
        shards, expert_trace=trace, expert_source=src,
        expert_residency=[res], expert_index_bytes=idx, expert_now=0.0,
    )

    assert any(r.key == ("kv", (7, 0)) for r in schedule.expert_evicted_kv)


# --- orchestrator end-to-end ----------------------------------------------------


def _expert_load_total(result):
    return sum(d.tokens for d in result.decisions if d.kind == "expert_load")


def test_e2e_global_lru_loads_fewer_experts_than_node_first():
    model = toy_moe_model(num_layers=2)

    def make_requests():
        reqs = []
        for i in range(4):
            reqs.append(request_from_messages(
                i, model,
                [dict(SYSTEM), dict(USER),
                 {"role": "assistant", "content": f"reply tail number {i}"}],
                workload_id=i, arrival_time=i * 1000.0,
            ))
        return reqs

    node = Simulator(
        make_kv_system([80e9]),
        StrategyConfig(max_batch_size=1, memory_policy="node_first", random_seed=0),
    ).run(make_requests())
    glru = Simulator(
        make_kv_system([80e9]),
        StrategyConfig(max_batch_size=1, memory_policy="global_lru", random_seed=0),
    ).run(make_requests())

    node_loaded = _expert_load_total(node)
    glru_loaded = _expert_load_total(glru)
    assert node_loaded > 0
    # Persisting experts on the device across batches cuts the experts re-fetched.
    assert glru_loaded < node_loaded


def test_e2e_node_first_reloads_experts_every_batch():
    # The legacy policy keeps no cross-batch residency: every batch that streams
    # experts emits its own expert_load (a fresh per-batch cache).
    model = toy_moe_model(num_layers=2)
    reqs = [
        request_from_messages(
            i, model,
            [dict(SYSTEM), dict(USER),
             {"role": "assistant", "content": f"distinct tail {i}"}],
            workload_id=i, arrival_time=i * 1000.0,
        )
        for i in range(4)
    ]

    result = Simulator(
        make_kv_system([80e9]),
        StrategyConfig(max_batch_size=1, memory_policy="node_first", random_seed=0),
    ).run(reqs)

    load_batches = {d.batch_index for d in result.decisions if d.kind == "expert_load"}
    # Every one of the four batches reloads (no persistence under node_first).
    assert len(load_batches) == 4
