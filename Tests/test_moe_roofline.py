"""End-to-end roofline tests for MoE models (single tier).

Mirrors the dense roofline suite: the full pipeline (MoE-aware shard generation
-> event generation) must match the independent closed-form roofline, and keep
matching as system parameters vary.
"""

from __future__ import annotations

import pytest

from serve_sim.model import toy_moe_model
from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.shards import WorkShardGenerator
from serve_sim.tracker import SequenceWork
from serve_sim.events import EventGenerator
from reference import reference_roofline


def make_device(name="gpu", peak=100e12, bw=2e12, cap=80e9):
    mem = MemoryDevice(f"{name}-hbm", capacity_bytes=cap, bandwidth_bytes_per_s=bw)
    return ComputeDevice(name, peak_flops_fp16=peak, first_tier_memory=mem)


def simulate(model, devices, batch_work, prefill_chunk_size=None, pp=1):
    shards = WorkShardGenerator(model).generate(batch_work, prefill_chunk_size)
    gen = EventGenerator(model, devices, pipeline_parallel=pp)
    return gen.run(shards)


# --- baseline MoE roofline ------------------------------------------------------


def test_moe_single_sequence_matches_reference():
    model = toy_moe_model()
    dev = make_device()
    work = [SequenceWork(0, 128, 16)]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


def test_moe_batch_of_four_matches_reference():
    model = toy_moe_model()
    dev = make_device()
    work = [SequenceWork(0, 80, 12) for _ in range(4)]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


def test_moe_ragged_batch_matches_reference():
    model = toy_moe_model()
    dev = make_device()
    work = [
        SequenceWork(0, 100, 16),
        SequenceWork(0, 60, 8),
        SequenceWork(50, 30, 20),
        SequenceWork(0, 80, 4),
    ]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


def test_moe_with_dense_prefix_layers_matches_reference():
    model = toy_moe_model(num_layers=6, num_dense_layers=2)
    dev = make_device()
    work = [SequenceWork(0, 64, 8) for _ in range(4)]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


def test_moe_chunked_prefill_matches_reference():
    model = toy_moe_model()
    dev = make_device()
    work = [SequenceWork(0, 100, 4)]
    sched = simulate(model, [dev], work, prefill_chunk_size=32)
    assert sched.makespan == pytest.approx(
        reference_roofline(model, dev, work, prefill_chunk_size=32), rel=1e-9
    )


# --- varying parameters ---------------------------------------------------------


@pytest.mark.parametrize("peak", [25e12, 100e12, 400e12])
@pytest.mark.parametrize("bw", [5e11, 2e12, 8e12])
def test_moe_matches_reference_as_device_varies(peak, bw):
    model = toy_moe_model()
    dev = make_device(peak=peak, bw=bw)
    work = [SequenceWork(0, 96, 12) for _ in range(3)]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_experts": 8, "num_experts_per_token": 1},
        {"num_experts": 64, "num_experts_per_token": 4},
        {"num_shared_experts": 0},
        {"num_shared_experts": 2},
        {"gated": True},
        {"expert_persistence_mean": 4.0},
        {"expert_persistence_mean": 64.0},
        {"shared_expert_intermediate_size": 128},
        {"include_lm_head": False},
    ],
)
def test_moe_matches_reference_as_model_varies(kwargs):
    model = toy_moe_model(**kwargs)
    dev = make_device()
    work = [SequenceWork(0, 96, 12), SequenceWork(20, 40, 8)]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


@pytest.mark.parametrize("param_bytes", [1, 2, 4])
def test_moe_matches_reference_as_dtype_varies(param_bytes):
    model = toy_moe_model(param_dtype_bytes=param_bytes)
    dev = make_device()
    work = [SequenceWork(0, 128, 16)]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


# --- qualitative MoE behaviour --------------------------------------------------


def test_moe_decode_cheaper_than_dense_at_same_total_experts_bandwidth_bound():
    # In a bandwidth-bound decode, MoE reads only the active experts whereas a
    # dense model of equivalent total FFN size reads the whole FFN every step.
    from serve_sim.model import toy_model

    moe = toy_moe_model(
        num_layers=2, num_experts=32, num_experts_per_token=2,
        num_shared_experts=0, intermediate_size=256, include_lm_head=False,
    )
    # dense model whose single FFN equals all 32 experts combined
    dense = toy_model(
        num_layers=2, intermediate_size=32 * 256, include_lm_head=False,
    )
    dev = make_device(peak=1e18, bw=1e12)  # bandwidth-bound
    work = [SequenceWork(0, 8, 8)]
    moe_time = simulate(moe, [dev], work).makespan
    dense_time = simulate(dense, [dev], work).makespan
    assert moe_time < dense_time


def test_moe_higher_persistence_reduces_prefill_weight_traffic():
    # bandwidth-bound prefill: higher persistence -> fewer distinct experts ->
    # less expert weight movement -> shorter time.
    dev = make_device(peak=1e18, bw=1e12)
    work = [SequenceWork(0, 256, 0)]
    low = toy_moe_model(expert_persistence_mean=2.0, num_experts=64)
    high = toy_moe_model(expert_persistence_mean=128.0, num_experts=64)
    t_low = simulate(low, [dev], work).makespan
    t_high = simulate(high, [dev], work).makespan
    assert t_high < t_low


def test_moe_pipeline_parallel_conserves_work():
    model = toy_moe_model(num_layers=4)
    work = [SequenceWork(0, 64, 8)]
    single = simulate(model, [make_device()], work, pp=1)
    dual = simulate(model, [make_device("g0"), make_device("g1")], work, pp=2)
    assert dual.total_flops == pytest.approx(single.total_flops, rel=1e-12)
    assert dual.total_bytes == pytest.approx(single.total_bytes, rel=1e-12)
