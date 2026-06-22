"""End-to-end two-tier roofline tests for MoE models.

The first tier holds KV cache, non-expert weights, and a working set of routed
experts; the rest of the experts live in the second tier and are moved up on
demand. Total makespan must equal the single-tier compute roofline plus the
expert-movement transfer time, and must keep matching as parameters vary.
"""

from __future__ import annotations

import pytest

from serve_sim.model import toy_moe_model
from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.shards import WorkShardGenerator
from serve_sim.tracker import SequenceWork
from serve_sim.events import EventGenerator
from serve_sim.tiering import build_activation_trace, derive_expert_cache_capacity
from reference import reference_roofline, reference_transfer_time, reference_two_tier


def make_two_tier_device(
    name="gpu",
    peak=100e12,
    tier1_bw=2e12,
    tier1_cap=80e9,
    tier2_bw=4e11,
    tier2_cap=1e12,
):
    tier1 = MemoryDevice(f"{name}-hbm", capacity_bytes=tier1_cap, bandwidth_bytes_per_s=tier1_bw)
    tier2 = MemoryDevice(f"{name}-cxl", capacity_bytes=tier2_cap, bandwidth_bytes_per_s=tier2_bw)
    return ComputeDevice(
        name, peak_flops_fp16=peak, first_tier_memory=tier1, second_tier_memory=tier2
    )


def simulate_two_tier(model, dev, work, capacity, prefill_chunk_size=None, seed=0):
    shards = WorkShardGenerator(model).generate(work, prefill_chunk_size)
    trace = build_activation_trace(model, work, prefill_chunk_size, seed=seed)
    gen = EventGenerator(model, [dev])
    return gen.run(shards, expert_trace=trace, expert_cache_capacity=capacity), trace


def min_capacity(*traces):
    """Smallest capacity that can hold every group's active set."""
    return max((len(g.active_experts) for tr in traces for g in tr), default=1)


# --- matches independent reference ---------------------------------------------


def test_two_tier_matches_reference_single_sequence():
    model = toy_moe_model()
    dev = make_two_tier_device()
    work = [SequenceWork(0, 128, 16)]
    trace = build_activation_trace(model, work, seed=0)
    cap = min_capacity(trace)
    sched, _ = simulate_two_tier(model, dev, work, cap, seed=0)
    expected = reference_two_tier(model, dev, work, trace, cap)
    assert sched.makespan == pytest.approx(expected, rel=1e-9)


def test_two_tier_matches_reference_batch():
    model = toy_moe_model(num_experts=32, num_experts_per_token=4)
    dev = make_two_tier_device()
    work = [SequenceWork(0, 96, 12) for _ in range(4)]
    trace = build_activation_trace(model, work, seed=7)
    cap = min_capacity(trace)
    sched, _ = simulate_two_tier(model, dev, work, cap, seed=7)
    expected = reference_two_tier(model, dev, work, trace, cap)
    assert sched.makespan == pytest.approx(expected, rel=1e-9)


def test_two_tier_matches_reference_with_chunking():
    model = toy_moe_model(num_layers=3)
    dev = make_two_tier_device()
    work = [SequenceWork(0, 200, 8), SequenceWork(0, 120, 4)]
    trace = build_activation_trace(model, work, prefill_chunk_size=32, seed=2)
    cap = min_capacity(trace)
    sched, _ = simulate_two_tier(model, dev, work, cap, prefill_chunk_size=32, seed=2)
    expected = reference_two_tier(model, dev, work, trace, cap, prefill_chunk_size=32)
    assert sched.makespan == pytest.approx(expected, rel=1e-9)


def test_two_tier_matches_with_derived_capacity():
    model = toy_moe_model(num_layers=2, num_experts=32)
    dev = make_two_tier_device()
    work = [SequenceWork(0, 80, 10) for _ in range(2)]
    cap = derive_expert_cache_capacity(model, dev.first_tier_memory.capacity_bytes, work)
    trace = build_activation_trace(model, work, seed=5)
    sched, _ = simulate_two_tier(model, dev, work, cap, seed=5)
    expected = reference_two_tier(model, dev, work, trace, cap)
    assert sched.makespan == pytest.approx(expected, rel=1e-9)


# --- decomposition: total = compute + transfers --------------------------------


def test_total_is_compute_plus_transfers():
    model = toy_moe_model()
    dev = make_two_tier_device()
    work = [SequenceWork(0, 128, 16)]
    trace = build_activation_trace(model, work, seed=0)
    cap = min_capacity(trace)
    sched, _ = simulate_two_tier(model, dev, work, cap)
    compute = sum(e.duration for e in sched.events if e.phase != "expert_transfer")
    transfers = sum(e.duration for e in sched.events if e.phase == "expert_transfer")
    assert compute == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)
    assert compute + transfers == pytest.approx(sched.makespan, rel=1e-9)


def test_compute_time_equals_single_tier_makespan():
    model = toy_moe_model()
    dev = make_two_tier_device()
    work = [SequenceWork(0, 100, 12)]
    trace = build_activation_trace(model, work, seed=0)
    cap = min_capacity(trace)
    sched, _ = simulate_two_tier(model, dev, work, cap)
    compute_only = sum(e.duration for e in sched.events if e.phase != "expert_transfer")
    assert compute_only == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


# --- monotonicity / sensitivity -------------------------------------------------


def test_higher_persistence_reduces_transfer_time():
    dev = make_two_tier_device()
    work = [SequenceWork(0, 256, 0)]
    low = toy_moe_model(expert_persistence_mean=2.0, num_experts=64)
    high = toy_moe_model(expert_persistence_mean=128.0, num_experts=64)
    trace_low = build_activation_trace(low, work, seed=1)
    trace_high = build_activation_trace(high, work, seed=1)
    cap = min_capacity(trace_low, trace_high)
    t_low = reference_transfer_time(low, dev, trace_low, cap)
    t_high = reference_transfer_time(high, dev, trace_high, cap)
    assert t_high < t_low


def test_faster_second_tier_reduces_transfer_time():
    model = toy_moe_model(num_experts=64)
    work = [SequenceWork(0, 200, 8)]
    trace = build_activation_trace(model, work, seed=3)
    cap = min_capacity(trace)
    slow = make_two_tier_device(tier2_bw=1e11)
    fast = make_two_tier_device(tier2_bw=1e12)
    t_slow = reference_transfer_time(model, slow, trace, cap)
    t_fast = reference_transfer_time(model, fast, trace, cap)
    assert t_fast < t_slow


def test_larger_capacity_does_not_increase_transfers():
    model = toy_moe_model(num_experts=64)
    dev = make_two_tier_device()
    work = [SequenceWork(0, 64, 40) for _ in range(2)]
    trace = build_activation_trace(model, work, prefill_chunk_size=16, seed=9)
    small = min_capacity(trace)
    t_small = reference_transfer_time(model, dev, trace, capacity=small)
    t_large = reference_transfer_time(model, dev, trace, capacity=64)
    assert t_large <= t_small


def test_capacity_at_least_num_experts_loads_each_once():
    # With capacity >= E, no expert is ever evicted, so total transfers across
    # all groups equal (distinct experts ever seen) * num_moe_layers.
    model = toy_moe_model(num_experts=16, num_experts_per_token=2)
    dev = make_two_tier_device()
    work = [SequenceWork(0, 300, 0)]
    trace = build_activation_trace(model, work, seed=4)
    ever_seen = set()
    for g in trace:
        ever_seen |= g.active_experts
    total_bytes = len(ever_seen) * model.num_moe_layers * model.routed_expert_bytes
    bw = min(
        dev.first_tier_memory.bandwidth_bytes_per_s,
        dev.second_tier_memory.bandwidth_bytes_per_s,
    )
    expected = total_bytes / bw
    assert reference_transfer_time(model, dev, trace, capacity=16) == pytest.approx(
        expected, rel=1e-9
    )


# --- validation -----------------------------------------------------------------


def test_capacity_too_small_for_active_set_raises():
    model = toy_moe_model(num_experts=64, num_experts_per_token=8)
    dev = make_two_tier_device()
    work = [SequenceWork(0, 4, 0) for _ in range(8)]  # decode-less, big prefill chunk
    # a single big prefill group can touch many distinct experts
    shards = WorkShardGenerator(model).generate(work)
    trace = build_activation_trace(model, work, seed=0)
    gen = EventGenerator(model, [dev])
    max_active = max(len(g.active_experts) for g in trace)
    with pytest.raises(ValueError, match="too small"):
        gen.run(shards, expert_trace=trace, expert_cache_capacity=max(1, max_active - 1))


def test_two_tier_requires_capacity():
    model = toy_moe_model()
    dev = make_two_tier_device()
    work = [SequenceWork(0, 32, 4)]
    shards = WorkShardGenerator(model).generate(work)
    trace = build_activation_trace(model, work)
    gen = EventGenerator(model, [dev])
    with pytest.raises(ValueError, match="expert_cache_capacity"):
        gen.run(shards, expert_trace=trace)


def test_two_tier_rejects_pipeline_parallel():
    model = toy_moe_model(num_layers=4)
    tier1 = MemoryDevice("hbm", 80e9, 2e12)
    tier2 = MemoryDevice("cxl", 1e12, 4e11)
    devs = [
        ComputeDevice(f"gpu{i}", 100e12, tier1, second_tier_memory=tier2)
        for i in range(2)
    ]
    work = [SequenceWork(0, 32, 4)]
    shards = WorkShardGenerator(model).generate(work)
    trace = build_activation_trace(model, work)
    gen = EventGenerator(model, devs, pipeline_parallel=2)
    with pytest.raises(NotImplementedError, match="pipeline_parallel"):
        gen.run(shards, expert_trace=trace, expert_cache_capacity=8)


def test_no_trace_falls_back_to_single_tier():
    # A two-tier device without an expert trace behaves like single-tier.
    model = toy_moe_model()
    dev = make_two_tier_device()
    work = [SequenceWork(0, 64, 8)]
    shards = WorkShardGenerator(model).generate(work)
    sched = EventGenerator(model, [dev]).run(shards)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)
    assert all(e.phase != "expert_transfer" for e in sched.events)
