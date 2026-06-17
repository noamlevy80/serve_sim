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
from serve_sim.parallelism import ParallelismPlanner
from serve_sim.pdd import context_kv_bytes, kv_transfer_duration, split_work
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


def make_device(name="gpu", peak=100e12, bw=1e12, cap=80e9):
    return ComputeDevice(name, peak_flops_fp16=peak, first_tier_memory=make_memory(f"{name}-mem", bw=bw, cap=cap))


def make_system(num_devices=1, cap=80e9):
    network = Network(
        scale_up_bandwidth_bytes_per_s=1e12,
        scale_up_latency_s=1e-6,
        cxl_bandwidth_bytes_per_s=1e11,
        cxl_latency_s=1e-7,
    )
    devices = tuple(make_device(f"g{i}", cap=cap) for i in range(num_devices))
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


# --- engine-slot placement (disjoint vs time-shared) ---------------------------


def test_two_batches_on_disjoint_slots_do_not_contend():
    # A 2-device system has two degree-1 engine slots, so two concurrent
    # single-sequence batches land on separate devices and each runs at solo
    # speed (no rescaling), unlike the single-device shared case.
    model = toy_model()
    system = make_system(2)
    device = system.compute_devices[0]
    solo = solo_makespan(model, device, 64, 8)

    reqs = [Request(0, model, 64, 8, 0.0), Request(1, model, 64, 8, 0.0)]
    strat = StrategyConfig(max_batch_size=1, target_concurrency=2)
    result = Simulator(system, strat).run(reqs)

    assert result.num_batches == 2
    # Both finish at ~solo (concurrent, independent), not 2x solo.
    assert result.record_for(0).completion_time == pytest.approx(solo)
    assert result.record_for(1).completion_time == pytest.approx(solo)
    assert result.makespan == pytest.approx(solo)


def test_more_batches_than_slots_time_share():
    # Three concurrent batches on a 2-slot system: two run disjoint at solo, the
    # third time-shares a slot. The makespan exceeds a single solo run.
    model = toy_model()
    system = make_system(2)
    device = system.compute_devices[0]
    solo = solo_makespan(model, device, 64, 8)

    reqs = [Request(i, model, 64, 8, 0.0) for i in range(3)]
    strat = StrategyConfig(max_batch_size=1, target_concurrency=3)
    result = Simulator(system, strat).run(reqs)

    assert result.num_batches == 3
    assert result.makespan > solo
    # The shared slot's two batches can't both beat solo*2 of shared bandwidth;
    # the disjoint one finishes by solo. Overall bounded by the contended pair.
    assert result.makespan <= 2 * solo * (1 + 1e-9)


def test_single_device_still_time_shares():
    # Regression: with one slot, two concurrent batches still co-run (shared),
    # matching the pre-pool behaviour.
    model = toy_model()
    system = make_system(1)
    device = system.compute_devices[0]
    solo = solo_makespan(model, device, 64, 8)

    reqs = [Request(0, model, 64, 8, 0.0), Request(1, model, 64, 8, 0.0)]
    strat = StrategyConfig(max_batch_size=1, target_concurrency=2)
    result = Simulator(system, strat).run(reqs)
    assert result.record_for(0).completion_time == pytest.approx(2 * solo)
    assert result.record_for(1).completion_time == pytest.approx(2 * solo)


def test_slot_is_released_for_reuse():
    # Four sequential-but-overlapping single-seq batches through a 2-slot system
    # all complete; slots must be released and reused (else later batches stall).
    model = toy_model()
    system = make_system(2)
    reqs = [Request(i, model, 32, 4, arrival_time=0.0) for i in range(4)]
    strat = StrategyConfig(max_batch_size=1, target_concurrency=2)
    result = Simulator(system, strat).run(reqs)
    assert len(result.records) == 4
    assert {r.request_id for r in result.records} == {0, 1, 2, 3}


def test_different_models_run_on_disjoint_slots():
    # Two models can't batch together; on a 2-slot system their batches run
    # concurrently on separate devices, each at solo speed.
    model_a = toy_model(name="a")
    model_b = toy_model(name="b")
    system = make_system(2)
    device = system.compute_devices[0]
    solo = solo_makespan(model_a, device, 64, 8)

    reqs = [Request(0, model_a, 64, 8, 0.0), Request(1, model_b, 64, 8, 0.0)]
    strat = StrategyConfig(max_batch_size=4, target_concurrency=2)
    result = Simulator(system, strat).run(reqs)

    assert result.num_batches == 2
    assert result.record_for(0).completion_time == pytest.approx(solo)
    assert result.record_for(1).completion_time == pytest.approx(solo)


# --- auto parallelism (per-batch pp x ep search) -------------------------------


def _engine_makespan(model, devices, prompt, output, pp, ep):
    """Reference makespan of one batch on an explicit (pp, ep) arrangement."""
    work = [Request(0, model, prompt, output).work]
    shards = WorkShardGenerator(model).generate(work)
    return EventGenerator(
        model, list(devices), pipeline_parallel=pp, expert_parallel=ep
    ).run(shards).makespan


def test_auto_parallelism_uses_chosen_arrangement():
    # Engine size = degree 2; ample memory -> planner picks (pp=1, ep=2).
    model = toy_model()  # 4 layers, dense
    system = make_system(2)
    devices = system.compute_devices
    req = Request(0, model, 64, 8, 0.0)
    strat = StrategyConfig(
        max_batch_size=1, pipeline_parallel=2, expert_parallel=1, auto_parallelism=True
    )
    result = Simulator(system, strat).run([req])

    expected = _engine_makespan(model, devices, 64, 8, pp=1, ep=2)
    assert result.record_for(0).completion_time == pytest.approx(expected)


def test_auto_parallelism_beats_fixed_pipeline():
    # A single batch gets no pipeline overlap, so the EP arrangement the search
    # picks is strictly faster than the fixed pure-pipeline engine.
    model = toy_model()
    system = make_system(2)
    devices = system.compute_devices
    req = Request(0, model, 64, 8, 0.0)

    auto = Simulator(
        system,
        StrategyConfig(
            max_batch_size=1, pipeline_parallel=2, expert_parallel=1,
            auto_parallelism=True,
        ),
    ).run([req])
    fixed_pp = _engine_makespan(model, devices, 64, 8, pp=2, ep=1)

    assert auto.record_for(0).completion_time < fixed_pp
    assert auto.record_for(0).completion_time == pytest.approx(
        _engine_makespan(model, devices, 64, 8, pp=1, ep=2)
    )


def test_auto_parallelism_off_uses_fixed_degrees():
    # Default (auto off) honours the strategy's pp/ep verbatim.
    model = toy_model()
    system = make_system(2)
    devices = system.compute_devices
    req = Request(0, model, 64, 8, 0.0)
    strat = StrategyConfig(max_batch_size=1, pipeline_parallel=2, expert_parallel=1)
    result = Simulator(system, strat).run([req])
    assert result.record_for(0).completion_time == pytest.approx(
        _engine_makespan(model, devices, 64, 8, pp=2, ep=1)
    )


def test_auto_parallelism_falls_back_to_pipeline_under_memory_pressure():
    # Shrink device memory so the EP-only arrangement (pp=1) no longer fits and
    # the search must reach for the pipeline split (pp=2) to place the batch.
    model = toy_model()
    probe = ParallelismPlanner(model, make_device())
    kv_tokens = 64 + 8
    big = probe.footprint(1, 2, kv_tokens)
    small = probe.footprint(2, 1, kv_tokens)
    cap = (big + small) / 2  # fits pp=2, not pp=1
    system = make_system(2, cap=cap)
    devices = system.compute_devices
    req = Request(0, model, 64, 8, 0.0)
    strat = StrategyConfig(
        max_batch_size=1, pipeline_parallel=2, expert_parallel=1, auto_parallelism=True
    )
    result = Simulator(system, strat).run([req])
    assert result.record_for(0).completion_time == pytest.approx(
        _engine_makespan(model, devices, 64, 8, pp=2, ep=1)
    )


def test_auto_parallelism_raises_when_batch_cannot_fit():
    model = toy_model()
    system = make_system(2, cap=1.0)
    req = Request(0, model, 64, 8, 0.0)
    strat = StrategyConfig(
        max_batch_size=1, pipeline_parallel=2, expert_parallel=1, auto_parallelism=True
    )
    with pytest.raises(ValueError, match="no parallelism arrangement"):
        Simulator(system, strat).run([req])


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


# --- prefill/decode disaggregation (PDD) ---------------------------------------


def _phase_makespan(model, device, prompt, output, phase, cached=0):
    prefill_w, decode_w = split_work(cached, prompt, output)
    work = prefill_w if phase == "prefill" else decode_w
    shards = WorkShardGenerator(model).generate([work])
    return EventGenerator(model, [device]).run(shards).makespan


def test_pdd_strategy_validation():
    with pytest.raises(ValueError):
        StrategyConfig(prefill_engine_fraction=0.0)
    with pytest.raises(ValueError):
        StrategyConfig(prefill_engine_fraction=1.0)


def test_pdd_requires_two_slots():
    system = make_system(1)
    with pytest.raises(ValueError):
        Simulator(system, StrategyConfig(allow_pdd=True))


def test_pdd_pools_are_disjoint_and_cover_all_devices():
    system = make_system(4)
    sim = Simulator(system, StrategyConfig(allow_pdd=True, prefill_engine_fraction=0.5))

    prefill_ids = {id(d) for slot in sim._prefill_pool.slots for d in slot.devices}
    decode_ids = {id(d) for slot in sim._decode_pool.slots for d in slot.devices}
    all_ids = {id(d) for d in system.compute_devices}

    assert prefill_ids.isdisjoint(decode_ids)
    assert prefill_ids | decode_ids == all_ids
    # 0.5 of 4 slots -> 2 prefill, 2 decode.
    assert len(prefill_ids) == 2
    assert len(decode_ids) == 2


def test_pdd_fraction_controls_split():
    system = make_system(4)
    sim = Simulator(system, StrategyConfig(allow_pdd=True, prefill_engine_fraction=0.25))
    prefill_ids = {id(d) for slot in sim._prefill_pool.slots for d in slot.devices}
    decode_ids = {id(d) for slot in sim._decode_pool.slots for d in slot.devices}
    assert len(prefill_ids) == 1
    assert len(decode_ids) == 3


def test_pdd_single_request_pipeline_timing():
    model = toy_model()
    system = make_system(2)
    g0, g1 = system.compute_devices
    req = Request(0, model, prompt_tokens=64, output_tokens=8, arrival_time=0.0)

    result = Simulator(
        system, StrategyConfig(allow_pdd=True, max_batch_size=1)
    ).run([req])

    prefill_ms = _phase_makespan(model, g0, 64, 8, "prefill")
    decode_ms = _phase_makespan(model, g1, 64, 8, "decode")
    transfer = kv_transfer_duration(context_kv_bytes(model, 64), g0, g1, system)
    expected = prefill_ms + transfer + decode_ms

    rec = result.record_for(0)
    assert rec.dispatch_time == pytest.approx(0.0)
    assert rec.completion_time == pytest.approx(expected)
    assert result.makespan == pytest.approx(expected)
    # one prefill batch + one decode batch
    assert result.num_batches == 2


def test_pdd_arrival_offsets_pipeline():
    model = toy_model()
    system = make_system(2)
    g0, g1 = system.compute_devices
    req = Request(0, model, prompt_tokens=48, output_tokens=6, arrival_time=9.0)

    result = Simulator(
        system, StrategyConfig(allow_pdd=True, max_batch_size=1)
    ).run([req])

    prefill_ms = _phase_makespan(model, g0, 48, 6, "prefill")
    decode_ms = _phase_makespan(model, g1, 48, 6, "decode")
    transfer = kv_transfer_duration(context_kv_bytes(model, 48), g0, g1, system)

    rec = result.record_for(0)
    assert rec.dispatch_time == pytest.approx(9.0)
    assert rec.completion_time == pytest.approx(9.0 + prefill_ms + transfer + decode_ms)


def test_pdd_pipelines_requests_in_parallel():
    # Two requests, two prefill slots and two decode slots: both pipelines run
    # fully in parallel, so the makespan is a single pipeline, not two.
    model = toy_model()
    system = make_system(4)
    devices = system.compute_devices
    g0, g2 = devices[0], devices[2]  # prefill rep, decode rep (2/2 split)
    reqs = [
        Request(0, model, 64, 8, arrival_time=0.0),
        Request(1, model, 64, 8, arrival_time=0.0),
    ]

    result = Simulator(
        system, StrategyConfig(allow_pdd=True, max_batch_size=1)
    ).run(reqs)

    prefill_ms = _phase_makespan(model, g0, 64, 8, "prefill")
    decode_ms = _phase_makespan(model, g2, 64, 8, "decode")
    transfer = kv_transfer_duration(context_kv_bytes(model, 64), g0, g2, system)
    one_pipeline = prefill_ms + transfer + decode_ms

    assert result.record_for(0).completion_time == pytest.approx(one_pipeline)
    assert result.record_for(1).completion_time == pytest.approx(one_pipeline)
    assert result.makespan == pytest.approx(one_pipeline)


def test_pdd_target_concurrency_serializes_inflight():
    # target_concurrency=1 keeps only one sequence in flight across both phases,
    # so the second request cannot start prefill until the first has decoded.
    model = toy_model()
    system = make_system(4)
    g0, g2 = system.compute_devices[0], system.compute_devices[2]
    reqs = [
        Request(0, model, 64, 8, arrival_time=0.0),
        Request(1, model, 64, 8, arrival_time=0.0),
    ]

    result = Simulator(
        system,
        StrategyConfig(allow_pdd=True, max_batch_size=1, target_concurrency=1),
    ).run(reqs)

    prefill_ms = _phase_makespan(model, g0, 64, 8, "prefill")
    decode_ms = _phase_makespan(model, g2, 64, 8, "decode")
    transfer = kv_transfer_duration(context_kv_bytes(model, 64), g0, g2, system)
    one_pipeline = prefill_ms + transfer + decode_ms

    rec0 = result.record_for(0)
    rec1 = result.record_for(1)
    assert rec0.completion_time == pytest.approx(one_pipeline)
    # req1 waits for req0 to fully complete before its prefill is dispatched.
    assert rec1.dispatch_time == pytest.approx(one_pipeline)
    assert rec1.completion_time == pytest.approx(2 * one_pipeline)


def test_pdd_off_matches_default_loop():
    # allow_pdd defaults off; turning it explicitly off runs the single-phase loop.
    model = toy_model()
    system = make_system(2)
    req = Request(0, model, prompt_tokens=64, output_tokens=8)

    baseline = Simulator(system, StrategyConfig(max_batch_size=1)).run([req])
    explicit = Simulator(
        system, StrategyConfig(max_batch_size=1, allow_pdd=False)
    ).run([req])

    assert explicit.record_for(0).completion_time == pytest.approx(
        baseline.record_for(0).completion_time
    )
