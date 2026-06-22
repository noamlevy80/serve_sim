"""Parallelism tests: pipeline + expert parallelism and memory tiering.

Expert parallelism distributes the routed experts across the devices of a stage
and splits the stage compute evenly (balanced routing, non-expert work
tensor-parallel), so a balanced stage is ``expert_parallel`` times faster and
each device keeps its own LRU residency of the experts it owns. Two memory
scenarios are covered: every device holding the whole model (single tier, no
expert movement) and a second tier shared between devices (a system NVM) that
streams experts up on demand and whose bandwidth bounds the aggregate movement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from serve_sim.model import toy_model, toy_moe_model
from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.shards import WorkShardGenerator
from serve_sim.tracker import SequenceWork
from serve_sim.events import EventGenerator
from serve_sim.model_config import load_model_config
from serve_sim.tiering import build_activation_trace, derive_expert_cache_capacity
from reference import reference_roofline, reference_layered, reference_ep_transfer

MODELS_DIR = Path(__file__).resolve().parents[1] / "Models"


def make_device(name="gpu", peak=1e14, bw=2e12, cap=80e9):
    mem = MemoryDevice(f"{name}-hbm", capacity_bytes=cap, bandwidth_bytes_per_s=bw)
    return ComputeDevice(name, peak_flops_fp16=peak, first_tier_memory=mem)


def make_two_tier(
    name="gpu", peak=1e14, tier1_bw=2e12, tier1_cap=80e9, tier2=None,
    tier2_bw=4e11, tier2_cap=1e12,
):
    t1 = MemoryDevice(f"{name}-hbm", capacity_bytes=tier1_cap, bandwidth_bytes_per_s=tier1_bw)
    if tier2 is None:
        tier2 = MemoryDevice(f"{name}-nvm", capacity_bytes=tier2_cap, bandwidth_bytes_per_s=tier2_bw)
    return ComputeDevice(name, peak_flops_fp16=peak, first_tier_memory=t1, second_tier_memory=tier2)


def simulate(model, devices, work, pp=1, ep=1, tp=1, chunk=None):
    shards = WorkShardGenerator(model).generate(work, chunk)
    gen = EventGenerator(
        model, devices, pipeline_parallel=pp, expert_parallel=ep, tensor_parallel=tp
    )
    return gen.run(shards)


def min_capacity_ep(trace, ep):
    """Smallest per-device capacity that holds each rank's per-group active set."""
    return max(
        (sum(1 for e in g.active_experts if e % ep == r) for g in trace for r in range(ep)),
        default=1,
    )


# --- expert-parallel compute (single tier) --------------------------------------


def test_expert_parallel_conserves_work():
    model = toy_moe_model()
    work = [SequenceWork(0, 64, 8)]
    single = simulate(model, [make_device()], work)
    dual = simulate(model, [make_device("a"), make_device("b")], work, ep=2)
    assert dual.total_flops == pytest.approx(single.total_flops, rel=1e-12)
    assert dual.total_bytes == pytest.approx(single.total_bytes, rel=1e-12)


def test_expert_parallel_halves_makespan():
    model = toy_moe_model()
    work = [SequenceWork(0, 64, 8)]
    single = simulate(model, [make_device()], work)
    dual = simulate(model, [make_device("a"), make_device("b")], work, ep=2)
    assert dual.makespan == pytest.approx(single.makespan / 2, rel=1e-9)


def test_expert_parallel_quarters_makespan():
    model = toy_moe_model()
    work = [SequenceWork(0, 48, 6), SequenceWork(0, 32, 4)]
    devs = [make_device(f"g{i}") for i in range(4)]
    single = simulate(model, [make_device()], work)
    quad = simulate(model, devs, work, ep=4)
    assert quad.makespan == pytest.approx(single.makespan / 4, rel=1e-9)


def test_expert_parallel_matches_reference():
    model = toy_moe_model()
    dev = make_device()
    work = [SequenceWork(0, 64, 8), SequenceWork(0, 40, 4)]
    devs = [make_device("a"), make_device("b")]
    sched = simulate(model, devs, work, ep=2)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work) / 2, rel=1e-9)


def test_expert_parallel_works_on_dense_model():
    # the even-split compute model applies to any model, not just MoE
    model = toy_model()
    work = [SequenceWork(0, 64, 8)]
    single = simulate(model, [make_device()], work)
    dual = simulate(model, [make_device("a"), make_device("b")], work, ep=2)
    assert dual.makespan == pytest.approx(single.makespan / 2, rel=1e-9)


# --- tensor-parallel compute ----------------------------------------------------


def test_tensor_parallel_conserves_work():
    model = toy_moe_model()
    work = [SequenceWork(0, 64, 8)]
    single = simulate(model, [make_device()], work)
    dual = simulate(model, [make_device("a"), make_device("b")], work, tp=2)
    assert dual.total_flops == pytest.approx(single.total_flops, rel=1e-12)
    assert dual.total_bytes == pytest.approx(single.total_bytes, rel=1e-12)


def test_tensor_parallel_halves_makespan():
    model = toy_moe_model()
    work = [SequenceWork(0, 64, 8)]
    single = simulate(model, [make_device()], work)
    dual = simulate(model, [make_device("a"), make_device("b")], work, tp=2)
    assert dual.makespan == pytest.approx(single.makespan / 2, rel=1e-9)


def test_tensor_parallel_works_on_dense_model():
    model = toy_model()
    work = [SequenceWork(0, 64, 8)]
    single = simulate(model, [make_device()], work)
    quad = simulate(model, [make_device(f"g{i}") for i in range(4)], work, tp=4)
    assert quad.makespan == pytest.approx(single.makespan / 4, rel=1e-9)


def test_tensor_and_expert_parallel_compose():
    # tp and ep both speed a stage up; together they give an ep*tp speedup.
    model = toy_moe_model()
    work = [SequenceWork(0, 64, 8)]
    single = simulate(model, [make_device()], work)
    devs = [make_device(f"g{i}") for i in range(4)]
    grid = simulate(model, devs, work, ep=2, tp=2)
    assert grid.makespan == pytest.approx(single.makespan / 4, rel=1e-9)
    assert grid.total_flops == pytest.approx(single.total_flops, rel=1e-12)


def test_pipeline_expert_tensor_grid_composes():
    model = toy_moe_model(num_layers=4)
    work = [SequenceWork(0, 64, 8)]
    pp_only = simulate(model, [make_device("s0"), make_device("s1")], work, pp=2)
    devs = [make_device(f"g{i}") for i in range(8)]
    grid = simulate(model, devs, work, pp=2, ep=2, tp=2)
    # ep*tp = 4 concurrent ranks per stage -> 4x faster than pp alone.
    assert grid.makespan == pytest.approx(pp_only.makespan / 4, rel=1e-9)


def test_tensor_parallel_grid_uses_pp_ep_tp_devices():
    model = toy_moe_model(num_layers=4)
    # pp*ep*tp = 8 devices required; 4 is not divisible.
    with pytest.raises(ValueError, match="divisible"):
        EventGenerator(
            model,
            [make_device(f"g{i}") for i in range(4)],
            pipeline_parallel=2,
            expert_parallel=2,
            tensor_parallel=2,
        )


def test_tensor_parallel_event_uses_distinct_devices():
    model = toy_model()
    work = [SequenceWork(0, 32, 4)]
    devs = [make_device("a"), make_device("b")]
    sched = simulate(model, devs, work, tp=2)
    # Each tp rank lands on its own device index (the grid is laid out per rank).
    indices = {e.device_index for e in sched.events if e.phase != "kernel_launch"}
    assert indices == {0, 1}


def test_two_tier_rejects_tensor_parallel():
    model = toy_moe_model()
    work = [SequenceWork(0, 32, 4)]
    devs = [make_two_tier("a"), make_two_tier("b")]
    trace = build_activation_trace(model, work, seed=0)
    shards = WorkShardGenerator(model).generate(work)
    gen = EventGenerator(model, devs, tensor_parallel=2)
    with pytest.raises(NotImplementedError, match="tensor_parallel"):
        gen.run(shards, expert_trace=trace, expert_cache_capacity=8)


# --- pipeline x expert parallelism ----------------------------------------------


def test_pipeline_and_expert_parallel_compose():
    model = toy_moe_model(num_layers=4)
    work = [SequenceWork(0, 64, 8)]
    pp_only = simulate(model, [make_device("s0"), make_device("s1")], work, pp=2)
    devs = [make_device(f"g{i}") for i in range(4)]
    grid = simulate(model, devs, work, pp=2, ep=2)
    assert grid.makespan == pytest.approx(pp_only.makespan / 2, rel=1e-9)


def test_pipeline_and_expert_parallel_conserve_work():
    model = toy_moe_model(num_layers=4)
    work = [SequenceWork(0, 64, 8)]
    single = simulate(model, [make_device()], work)
    devs = [make_device(f"g{i}") for i in range(4)]
    grid = simulate(model, devs, work, pp=2, ep=2)
    assert grid.total_flops == pytest.approx(single.total_flops, rel=1e-12)
    assert grid.total_bytes == pytest.approx(single.total_bytes, rel=1e-12)


# --- validation -----------------------------------------------------------------


def test_expert_parallel_requires_enough_devices():
    model = toy_moe_model()
    with pytest.raises(ValueError, match="divisible"):
        EventGenerator(model, [make_device()], expert_parallel=2)


def test_grid_uses_pipeline_times_expert_devices():
    model = toy_moe_model(num_layers=4)
    with pytest.raises(ValueError, match="divisible"):
        EventGenerator(model, [make_device("a"), make_device("b")], pipeline_parallel=2, expert_parallel=2)


def test_two_tier_rejects_pipeline_parallel():
    model = toy_moe_model(num_layers=4)
    work = [SequenceWork(0, 32, 4)]
    devs = [make_two_tier("a"), make_two_tier("b")]
    trace = build_activation_trace(model, work, seed=0)
    shards = WorkShardGenerator(model).generate(work)
    gen = EventGenerator(model, devs, pipeline_parallel=2)
    with pytest.raises(NotImplementedError, match="pipeline_parallel"):
        gen.run(shards, expert_trace=trace, expert_cache_capacity=8)


# --- scenario A: every device holds the whole model (single tier) ---------------


def test_each_device_holds_full_model_no_movement():
    model = toy_moe_model()
    work = [SequenceWork(0, 64, 8)]
    devs = [make_device("a"), make_device("b")]  # single tier, no second tier
    trace = build_activation_trace(model, work, seed=0)
    shards = WorkShardGenerator(model).generate(work)
    gen = EventGenerator(model, devs, expert_parallel=2)
    sched = gen.run(shards, expert_trace=trace, expert_cache_capacity=999)
    # no second tier -> experts are all resident -> no transfer events
    assert all(e.phase != "transfer" for e in sched.events)
    single = simulate(model, [make_device()], work)
    assert sched.makespan == pytest.approx(single.makespan / 2, rel=1e-9)


def test_expert_parallel_reduces_required_capacity():
    model = toy_moe_model()
    work = [SequenceWork(0, 128, 16)]
    trace = build_activation_trace(model, work, seed=0)
    cap_single = max(len(g.active_experts) for g in trace)
    cap_ep2 = min_capacity_ep(trace, 2)
    # splitting experts across devices never needs more per-device capacity
    assert cap_ep2 <= cap_single


def test_derive_capacity_expert_parallel_gives_more_room():
    model = toy_moe_model()
    work = [SequenceWork(0, 64, 8)]
    cap1 = derive_expert_cache_capacity(model, 60e9, work, expert_parallel=1)
    cap2 = derive_expert_cache_capacity(model, 60e9, work, expert_parallel=2)
    assert cap2 >= cap1


# --- scenario B: shared second tier (system NVM) --------------------------------


def test_shared_tier2_matches_reference():
    model = toy_moe_model()
    work = [SequenceWork(0, 128, 16)]
    nvm = MemoryDevice("sys-nvm", capacity_bytes=1e12, bandwidth_bytes_per_s=4e11)
    devs = [make_two_tier("a", tier2=nvm), make_two_tier("b", tier2=nvm)]
    trace = build_activation_trace(model, work, seed=0)
    cap = min_capacity_ep(trace, 2)
    shards = WorkShardGenerator(model).generate(work)
    gen = EventGenerator(model, devs, expert_parallel=2)
    sched = gen.run(shards, expert_trace=trace, expert_cache_capacity=cap)
    compute = reference_roofline(model, devs[0], work) / 2
    movement = reference_ep_transfer(devs, model, trace, cap, 2, shared_tier2=True)
    assert sched.makespan == pytest.approx(compute + movement, rel=1e-9)
    assert any(e.phase == "transfer" for e in sched.events)


def test_private_tier2_matches_reference():
    model = toy_moe_model()
    work = [SequenceWork(0, 128, 16)]
    devs = [make_two_tier("a"), make_two_tier("b")]  # separate NVM instances
    trace = build_activation_trace(model, work, seed=0)
    cap = min_capacity_ep(trace, 2)
    shards = WorkShardGenerator(model).generate(work)
    gen = EventGenerator(model, devs, expert_parallel=2)
    sched = gen.run(shards, expert_trace=trace, expert_cache_capacity=cap)
    compute = reference_roofline(model, devs[0], work) / 2
    movement = reference_ep_transfer(devs, model, trace, cap, 2, shared_tier2=False)
    assert sched.makespan == pytest.approx(compute + movement, rel=1e-9)


def test_shared_tier2_no_faster_than_private_when_nvm_bound():
    model = toy_moe_model()
    work = [SequenceWork(0, 128, 16)]
    nvm = MemoryDevice("sys-nvm", capacity_bytes=1e12, bandwidth_bytes_per_s=2e11)
    shared_devs = [make_two_tier("a", tier2=nvm), make_two_tier("b", tier2=nvm)]
    priv_devs = [make_two_tier("a", tier2_bw=2e11), make_two_tier("b", tier2_bw=2e11)]
    trace = build_activation_trace(model, work, seed=0)
    cap = min_capacity_ep(trace, 2)
    shards = WorkShardGenerator(model).generate(work)
    shared = EventGenerator(model, shared_devs, expert_parallel=2).run(
        shards, expert_trace=trace, expert_cache_capacity=cap
    )
    private = EventGenerator(model, priv_devs, expert_parallel=2).run(
        shards, expert_trace=trace, expert_cache_capacity=cap
    )
    # a shared NVM funnels both ranks' loads through one pipe -> never faster
    assert shared.makespan >= private.makespan - 1e-9


# --- real models on toy systems -------------------------------------------------


@pytest.fixture(scope="module")
def deepseek():
    return load_model_config(MODELS_DIR / "deepseek-v3.2.json")


def test_real_model_expert_parallel_compute(deepseek):
    work = [SequenceWork(0, 48, 4)]
    dev = make_device()
    devs = [make_device("a"), make_device("b")]
    sched = simulate(deepseek, devs, work, ep=2)
    assert sched.makespan == pytest.approx(reference_layered(deepseek, dev, work) / 2, rel=1e-9)


def test_real_model_shared_tier2_movement(deepseek):
    work = [SequenceWork(0, 64, 6)]
    nvm = MemoryDevice("sys-nvm", capacity_bytes=1e13, bandwidth_bytes_per_s=5e11)
    devs = [
        make_two_tier("a", tier1_cap=300e9, tier2=nvm),
        make_two_tier("b", tier1_cap=300e9, tier2=nvm),
    ]
    trace = build_activation_trace(deepseek, work, seed=0)
    cap = min_capacity_ep(trace, 2)
    shards = WorkShardGenerator(deepseek).generate(work)
    gen = EventGenerator(deepseek, devs, expert_parallel=2)
    sched = gen.run(shards, expert_trace=trace, expert_cache_capacity=cap)
    compute = reference_layered(deepseek, devs[0], work) / 2
    movement = reference_ep_transfer(devs, deepseek, trace, cap, 2, shared_tier2=True)
    assert sched.makespan == pytest.approx(compute + movement, rel=1e-9)
    assert any(e.phase == "transfer" for e in sched.events)
