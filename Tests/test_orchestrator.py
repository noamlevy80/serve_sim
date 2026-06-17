"""Orchestrator v0 tests: the event-driven serving loop.

A `Simulator` runs `Request`s through a `System` under a fixed `StrategyConfig`:
requests arrive over time, are collected into a concurrency window, dispatched as
batches onto a fixed device slice, co-run (and rescaled) by the incremental
arbiter, and retired on completion. These tests check the loop's timing against
the standalone roofline pipeline and the closed-form sharing behaviour, plus the
batching-window and target-concurrency policies.
"""

from __future__ import annotations

import pytest

from serve_sim.events import EventGenerator
from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.model import toy_model
from serve_sim.shards import WorkShardGenerator
from serve_sim.system import Network, Node, System
from serve_sim.orchestrator import (
    Request,
    RunResult,
    Simulator,
    StrategyConfig,
)


# --- helpers --------------------------------------------------------------------


def make_memory(name="mem", bw=1e12, cap=80e9):
    return MemoryDevice(name, capacity_bytes=cap, bandwidth_bytes_per_s=bw)


def make_device(name="gpu", peak=100e12, bw=1e12):
    return ComputeDevice(name, peak_flops_fp16=peak, first_tier_memory=make_memory(f"{name}-mem", bw=bw))


def make_system(num_devices=1):
    network = Network(
        scale_up_bandwidth_bytes_per_s=1e12,
        scale_up_latency_s=1e-6,
        cxl_bandwidth_bytes_per_s=1e11,
        cxl_latency_s=1e-7,
    )
    devices = tuple(make_device(f"g{i}") for i in range(num_devices))
    node = Node(name="node-0", compute_devices=devices, node_memory=make_memory("node", bw=5e11))
    return System(
        name="test", network=network, input_memory=make_memory("nvm", bw=5e9, cap=1e12),
        nodes=(node,),
    )


def solo_makespan(model, device, prompt, output, cached=0, chunk=None):
    work = [Request(0, model, prompt, output, cached_tokens=cached).work]
    shards = WorkShardGenerator(model).generate(work, prefill_chunk_size=chunk)
    return EventGenerator(model, [device]).run(shards).makespan


# --- single request reproduces the standalone pipeline -------------------------


def test_single_request_completion_matches_solo():
    model = toy_model()
    system = make_system(1)
    device = system.compute_devices[0]
    req = Request(0, model, prompt_tokens=64, output_tokens=8, arrival_time=0.0)

    result = Simulator(system, StrategyConfig(max_batch_size=1)).run([req])

    expected = solo_makespan(model, device, 64, 8)
    rec = result.record_for(0)
    assert rec.completion_time == pytest.approx(expected)
    assert rec.dispatch_time == pytest.approx(0.0)
    assert result.makespan == pytest.approx(expected)
    assert result.num_batches == 1


def test_arrival_offsets_completion():
    model = toy_model()
    system = make_system(1)
    device = system.compute_devices[0]
    req = Request(0, model, prompt_tokens=64, output_tokens=8, arrival_time=12.5)

    result = Simulator(system, StrategyConfig(max_batch_size=1)).run([req])

    expected = solo_makespan(model, device, 64, 8)
    rec = result.record_for(0)
    assert rec.dispatch_time == pytest.approx(12.5)
    assert rec.completion_time == pytest.approx(12.5 + expected)
    assert rec.latency == pytest.approx(expected)


def test_sequential_arrivals_do_not_overlap():
    # Second request arrives long after the first has finished: no contention.
    model = toy_model()
    system = make_system(1)
    device = system.compute_devices[0]
    solo = solo_makespan(model, device, 64, 8)

    reqs = [
        Request(0, model, 64, 8, arrival_time=0.0),
        Request(1, model, 64, 8, arrival_time=1000.0),
    ]
    result = Simulator(system, StrategyConfig(max_batch_size=1)).run(reqs)

    assert result.record_for(0).completion_time == pytest.approx(solo)
    assert result.record_for(1).dispatch_time == pytest.approx(1000.0)
    assert result.record_for(1).completion_time == pytest.approx(1000.0 + solo)


# --- batching window -----------------------------------------------------------


def test_fill_triggers_dispatch_as_one_batch():
    model = toy_model()
    system = make_system(1)

    reqs = [
        Request(0, model, 64, 8, arrival_time=0.0),
        Request(1, model, 64, 8, arrival_time=0.0),
    ]
    # Large window, batch size 2 -> both fill one batch immediately at t=0.
    strat = StrategyConfig(max_batch_size=2, max_window_duration=1e9)
    result = Simulator(system, strat).run(reqs)

    assert result.num_batches == 1
    r0, r1 = result.record_for(0), result.record_for(1)
    assert r0.batch_index == r1.batch_index
    assert r0.dispatch_time == pytest.approx(0.0)
    # Both retire together (one batched job).
    assert r0.completion_time == pytest.approx(r1.completion_time)


def test_window_timeout_dispatches_partial_batch():
    model = toy_model()
    system = make_system(1)
    device = system.compute_devices[0]

    # Batch size 5 never fills (one request); window 3.0 forces dispatch at t=3.
    req = Request(0, model, 64, 8, arrival_time=0.0)
    strat = StrategyConfig(max_batch_size=5, max_window_duration=3.0)
    result = Simulator(system, strat).run([req])

    rec = result.record_for(0)
    assert rec.dispatch_time == pytest.approx(3.0)
    assert rec.completion_time == pytest.approx(3.0 + solo_makespan(model, device, 64, 8))


def test_batched_decode_cheaper_than_two_solo_runs():
    model = toy_model()
    system = make_system(1)
    device = system.compute_devices[0]
    solo = solo_makespan(model, device, 64, 8)

    reqs = [Request(0, model, 64, 8, 0.0), Request(1, model, 64, 8, 0.0)]
    strat = StrategyConfig(max_batch_size=2, max_window_duration=1e9)
    result = Simulator(system, strat).run(reqs)

    # One batch sharing weight reads is cheaper than two serialized solo runs.
    assert result.makespan < 2 * solo


# --- concurrency / contention --------------------------------------------------


def test_two_concurrent_batches_share_device():
    # batch size 1 -> two separate single-sequence jobs that co-run on device 0,
    # each rescaled to half the resource, so both finish at ~2x the solo time.
    model = toy_model()
    system = make_system(1)
    device = system.compute_devices[0]
    solo = solo_makespan(model, device, 64, 8)

    reqs = [Request(0, model, 64, 8, 0.0), Request(1, model, 64, 8, 0.0)]
    strat = StrategyConfig(max_batch_size=1, target_concurrency=2)
    result = Simulator(system, strat).run(reqs)

    assert result.num_batches == 2
    assert result.makespan == pytest.approx(2 * solo)


def test_target_concurrency_serializes_batches():
    # Two requests, concurrency 1: the second waits for the first to finish.
    model = toy_model()
    system = make_system(1)
    device = system.compute_devices[0]
    solo = solo_makespan(model, device, 64, 8)

    reqs = [Request(0, model, 64, 8, 0.0), Request(1, model, 64, 8, 0.0)]
    strat = StrategyConfig(max_batch_size=1, target_concurrency=1)
    result = Simulator(system, strat).run(reqs)

    r0, r1 = result.record_for(0), result.record_for(1)
    assert r0.completion_time == pytest.approx(solo)
    # The second dispatches only once the first frees the single slot.
    assert r1.dispatch_time == pytest.approx(solo)
    assert r1.completion_time == pytest.approx(2 * solo)


def test_concurrency_one_with_single_arrival_is_solo():
    model = toy_model()
    system = make_system(1)
    device = system.compute_devices[0]
    req = Request(0, model, 64, 8, 0.0)
    strat = StrategyConfig(max_batch_size=1, target_concurrency=1)
    result = Simulator(system, strat).run([req])
    assert result.record_for(0).completion_time == pytest.approx(
        solo_makespan(model, device, 64, 8)
    )


# --- model grouping ------------------------------------------------------------


def test_different_models_dispatch_in_separate_batches():
    model_a = toy_model(name="a")
    model_b = toy_model(name="b")
    system = make_system(1)

    reqs = [
        Request(0, model_a, 64, 8, 0.0),
        Request(1, model_b, 64, 8, 0.0),
    ]
    strat = StrategyConfig(max_batch_size=4, max_window_duration=1e9)
    result = Simulator(system, strat).run(reqs)

    # Distinct model instances cannot share a batch.
    assert result.num_batches == 2
    assert result.record_for(0).batch_index != result.record_for(1).batch_index


def test_same_model_instance_batches_together():
    model = toy_model()
    system = make_system(1)
    reqs = [Request(i, model, 64, 8, 0.0) for i in range(3)]
    strat = StrategyConfig(max_batch_size=4, max_window_duration=1e9)
    result = Simulator(system, strat).run(reqs)
    assert result.num_batches == 1
    assert len({result.record_for(i).batch_index for i in range(3)}) == 1


def test_batch_size_caps_group():
    model = toy_model()
    system = make_system(1)
    reqs = [Request(i, model, 64, 8, 0.0) for i in range(5)]
    # Batch size 2 -> three batches (2 + 2 + 1).
    strat = StrategyConfig(max_batch_size=2, max_window_duration=1e9, target_concurrency=10)
    result = Simulator(system, strat).run(reqs)
    assert result.num_batches == 3


# --- all requests retire -------------------------------------------------------


def test_all_requests_retire_exactly_once():
    model = toy_model()
    system = make_system(1)
    reqs = [Request(i, model, 32 + i, 4, arrival_time=float(i)) for i in range(6)]
    strat = StrategyConfig(max_batch_size=2, max_window_duration=0.5, target_concurrency=4)
    result = Simulator(system, strat).run(reqs)

    ids = sorted(r.request_id for r in result.records)
    assert ids == list(range(6))
    assert len(result.records) == 6
    for rec in result.records:
        assert rec.completion_time >= rec.dispatch_time >= rec.arrival_time


# --- from_workload convenience -------------------------------------------------


def test_request_from_workload(fake_fetcher):
    from serve_sim.dataset import WorkloadLoader
    from serve_sim.tokenizer import WhitespaceTokenizer

    loader = WorkloadLoader(fake_fetcher, page_size=50)
    workload = loader.load_first()
    model = toy_model()
    req = Request.from_workload(7, workload, model, WhitespaceTokenizer(), arrival_time=2.0)

    assert req.request_id == 7
    assert req.arrival_time == pytest.approx(2.0)
    assert req.prompt_tokens >= 0
    assert req.output_tokens == workload[0].output_length


# --- validation ----------------------------------------------------------------


def test_strategy_validation():
    with pytest.raises(ValueError):
        StrategyConfig(max_batch_size=0)
    with pytest.raises(ValueError):
        StrategyConfig(max_window_duration=-1.0)
    with pytest.raises(ValueError):
        StrategyConfig(target_concurrency=0)
    with pytest.raises(ValueError):
        StrategyConfig(pipeline_parallel=0)


def test_request_validation():
    model = toy_model()
    with pytest.raises(ValueError):
        Request(0, model, prompt_tokens=-1, output_tokens=4)
    with pytest.raises(ValueError):
        Request(0, model, prompt_tokens=10, output_tokens=4, cached_tokens=20)
    with pytest.raises(ValueError):
        Request(0, model, prompt_tokens=10, output_tokens=4, arrival_time=-1.0)


def test_engine_needs_enough_devices():
    model = toy_model()
    system = make_system(1)
    # pipeline_parallel 2 needs 2 devices but the system has 1.
    with pytest.raises(ValueError):
        Simulator(system, StrategyConfig(pipeline_parallel=2))


def test_empty_run_is_empty():
    system = make_system(1)
    result = Simulator(system, StrategyConfig()).run([])
    assert result.records == []
    assert result.makespan == 0.0
    assert result.num_batches == 0
