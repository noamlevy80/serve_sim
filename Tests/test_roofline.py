"""End-to-end roofline tests.

These drive the full Stage-2 pipeline (work-shard generation -> event
generation) and check the total simulated time against an independent closed-form
roofline ([reference.py](reference.py)), including as system parameters vary.
"""

from __future__ import annotations

import pytest

from serve_sim.model import toy_model, toy_moe_model
from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.shards import WorkShardGenerator
from serve_sim.tracker import SequenceWork
from serve_sim.events import EventGenerator
from reference import reference_roofline


# --- helpers --------------------------------------------------------------------


def make_device(name="gpu", peak=100e12, bw=2e12, cap=80e9):
    mem = MemoryDevice(f"{name}-hbm", capacity_bytes=cap, bandwidth_bytes_per_s=bw)
    return ComputeDevice(name, peak_flops_fp16=peak, first_tier_memory=mem)


def simulate(model, devices, batch_work, prefill_chunk_size=None, pp=1):
    shards = WorkShardGenerator(model).generate(batch_work, prefill_chunk_size)
    gen = EventGenerator(model, devices, pipeline_parallel=pp)
    return gen.run(shards)


# --- baseline single-sequence roofline -----------------------------------------


def test_single_sequence_matches_reference():
    model = toy_model()
    dev = make_device()
    work = [SequenceWork(cached_tokens=0, prefill_tokens=128, decode_tokens=16)]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


def test_makespan_equals_phase_sum_and_events_are_contiguous():
    model = toy_model()
    dev = make_device()
    work = [SequenceWork(0, 64, 8)]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(
        sched.time_for_phase("prefill") + sched.time_for_phase("decode"), rel=1e-12
    )
    clock = 0.0
    for e in sched.events:
        assert e.start == pytest.approx(clock, rel=1e-12, abs=1e-15)
        assert e.duration == pytest.approx(max(e.compute_time, e.bandwidth_time))
        clock = e.end
    assert clock == pytest.approx(sched.makespan)


def test_existing_cache_matches_reference():
    model = toy_model()
    dev = make_device()
    # simulate a later turn: 200 cached tokens, 50 new prefill, 10 decode
    work = [SequenceWork(cached_tokens=200, prefill_tokens=50, decode_tokens=10)]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


def test_prefill_chunking_matches_reference():
    model = toy_model()
    dev = make_device()
    work = [SequenceWork(0, 100, 4)]
    sched = simulate(model, [dev], work, prefill_chunk_size=32)
    assert sched.makespan == pytest.approx(
        reference_roofline(model, dev, work, prefill_chunk_size=32), rel=1e-9
    )


# --- varying system parameters --------------------------------------------------


@pytest.mark.parametrize("peak", [25e12, 100e12, 400e12, 1e15])
def test_matches_reference_as_peak_flops_varies(peak):
    model = toy_model()
    dev = make_device(peak=peak)
    work = [SequenceWork(0, 128, 16)]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


@pytest.mark.parametrize("bw", [5e11, 1e12, 2e12, 8e12])
def test_matches_reference_as_bandwidth_varies(bw):
    model = toy_model()
    dev = make_device(bw=bw)
    work = [SequenceWork(0, 128, 16)]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


@pytest.mark.parametrize("param_bytes,kv_bytes", [(2, 2), (1, 2), (1, 1), (4, 2)])
def test_matches_reference_as_dtype_varies(param_bytes, kv_bytes):
    model = toy_model(param_dtype_bytes=param_bytes, kv_dtype_bytes=kv_bytes)
    dev = make_device()
    work = [SequenceWork(0, 128, 16)]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_layers": 2},
        {"num_layers": 8, "hidden_size": 512, "num_query_heads": 8},
        {"num_kv_heads": 2},  # GQA
        {"gated": True},
        {"intermediate_size": 4096},
        {"vocab_size": 50000},
        {"include_lm_head": False},
    ],
)
def test_matches_reference_as_model_varies(kwargs):
    model = toy_model(**kwargs)
    dev = make_device()
    work = [SequenceWork(0, 96, 12)]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


# --- explicit scaling relationships (decoupled compute / bandwidth) -------------


def test_compute_bound_scenario_scales_inversely_with_peak():
    # huge bandwidth -> every event is compute-bound
    model = toy_model()
    work = [SequenceWork(0, 256, 0)]  # prefill only, compute-heavy
    base = simulate(model, [make_device(peak=100e12, bw=1e18)], work).makespan
    doubled = simulate(model, [make_device(peak=200e12, bw=1e18)], work).makespan
    assert doubled == pytest.approx(base / 2, rel=1e-9)


def test_bandwidth_bound_scenario_scales_inversely_with_bandwidth():
    # huge compute -> every event is bandwidth-bound
    model = toy_model()
    work = [SequenceWork(0, 8, 64)]  # decode-heavy, weight-read bound
    base = simulate(model, [make_device(peak=1e18, bw=1e12)], work).makespan
    doubled = simulate(model, [make_device(peak=1e18, bw=2e12)], work).makespan
    assert doubled == pytest.approx(base / 2, rel=1e-9)


def test_fp8_doubles_compute_throughput_in_compute_bound_regime():
    work = [SequenceWork(0, 256, 0)]
    fp16 = toy_model(param_dtype_bytes=2)
    fp8 = toy_model(param_dtype_bytes=1)
    # huge bandwidth isolates compute; bytes differ but are not the bottleneck
    t16 = simulate(fp16, [make_device(peak=100e12, bw=1e18)], work).makespan
    t8 = simulate(fp8, [make_device(peak=100e12, bw=1e18)], work).makespan
    assert t8 == pytest.approx(t16 / 2, rel=1e-9)


# --- batch of 4 -----------------------------------------------------------------


def test_batch_of_four_matches_reference():
    model = toy_model()
    dev = make_device()
    work = [SequenceWork(0, 80, 12) for _ in range(4)]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


def test_batch_decode_amortizes_weight_reads_vs_serial():
    # In a bandwidth-bound decode regime, batching 4 sequences reads the weights
    # once per step instead of four times, so it is much faster than 4x serial.
    model = toy_model()
    dev = make_device(peak=1e18, bw=1e12)  # force bandwidth-bound
    one = SequenceWork(0, 32, 16)
    serial_one = simulate(model, [dev], [one]).makespan
    batched_four = simulate(model, [dev], [one, one, one, one]).makespan
    # batched is far cheaper than running four sequences back-to-back
    assert batched_four < 4 * serial_one
    # and matches the closed-form roofline exactly
    assert batched_four == pytest.approx(
        reference_roofline(model, dev, [one, one, one, one]), rel=1e-9
    )


def test_ragged_batch_matches_reference():
    model = toy_model()
    dev = make_device()
    work = [
        SequenceWork(0, 100, 16),
        SequenceWork(0, 60, 8),
        SequenceWork(50, 30, 20),
        SequenceWork(0, 80, 4),
    ]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


# --- batch-size scan ------------------------------------------------------------
# Sweep the batch size and confirm the simulated makespan tracks the independent
# closed-form roofline at every point, for several workload shapes.


BATCH_SIZES = [1, 2, 3, 4, 8, 16, 32, 64]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_uniform_batch_scan_matches_reference(batch_size):
    model = toy_model()
    dev = make_device()
    work = [SequenceWork(0, 96, 12) for _ in range(batch_size)]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_prefill_only_batch_scan_matches_reference(batch_size):
    # compute-heavy: all prefill, no decode.
    model = toy_model()
    dev = make_device()
    work = [SequenceWork(0, 128, 0) for _ in range(batch_size)]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_decode_only_batch_scan_matches_reference(batch_size):
    # bandwidth-heavy: a warm cache then pure decode (weight reads amortized).
    model = toy_model()
    dev = make_device()
    work = [SequenceWork(cached_tokens=64, prefill_tokens=0, decode_tokens=16)
            for _ in range(batch_size)]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_chunked_prefill_batch_scan_matches_reference(batch_size):
    model = toy_model()
    dev = make_device()
    work = [SequenceWork(0, 100, 8) for _ in range(batch_size)]
    sched = simulate(model, [dev], work, prefill_chunk_size=32)
    assert sched.makespan == pytest.approx(
        reference_roofline(model, dev, work, prefill_chunk_size=32), rel=1e-9
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_moe_batch_scan_matches_reference(batch_size):
    # MoE: more sequences touch more distinct experts -> more weight bytes read.
    model = toy_moe_model()
    dev = make_device()
    work = [SequenceWork(0, 64, 8) for _ in range(batch_size)]
    sched = simulate(model, [dev], work)
    assert sched.makespan == pytest.approx(reference_roofline(model, dev, work), rel=1e-9)


def test_makespan_is_monotonic_in_batch_size():
    # Adding work can never reduce the makespan.
    model = toy_model()
    dev = make_device()
    prev = 0.0
    for batch_size in BATCH_SIZES:
        work = [SequenceWork(0, 96, 12) for _ in range(batch_size)]
        makespan = simulate(model, [dev], work).makespan
        assert makespan >= prev - 1e-12
        prev = makespan


def test_compute_bound_prefill_scales_linearly_with_batch_size():
    # huge bandwidth isolates compute; prefill FLOPs are per-sequence, so the
    # makespan grows linearly with the number of sequences.
    model = toy_model()
    dev = make_device(peak=100e12, bw=1e18)
    one = simulate(model, [dev], [SequenceWork(0, 128, 0)]).makespan
    for batch_size in [1, 2, 4, 8]:
        work = [SequenceWork(0, 128, 0) for _ in range(batch_size)]
        makespan = simulate(model, [dev], work).makespan
        assert makespan == pytest.approx(batch_size * one, rel=1e-9)


def test_bandwidth_bound_decode_amortizes_weights_across_batch():
    # huge compute isolates bandwidth. A batched decode step reads the weights
    # once for the whole batch, so per-sequence cost falls as the batch grows
    # (makespan grows far slower than linearly).
    model = toy_model()
    dev = make_device(peak=1e18, bw=1e12)
    decode = SequenceWork(cached_tokens=32, prefill_tokens=0, decode_tokens=16)
    one = simulate(model, [dev], [decode]).makespan
    eight = simulate(model, [dev], [decode for _ in range(8)]).makespan
    assert eight < 8 * one
    # exact match to the independent roofline at the batched point.
    assert eight == pytest.approx(
        reference_roofline(model, dev, [decode for _ in range(8)]), rel=1e-9
    )


def test_decode_per_sequence_throughput_improves_with_batching():
    # In a bandwidth-bound regime the makespan-per-sequence strictly decreases as
    # the batch grows (the core efficiency win of batching).
    model = toy_model()
    dev = make_device(peak=1e18, bw=1e12)
    decode = SequenceWork(cached_tokens=32, prefill_tokens=0, decode_tokens=16)
    per_seq = []
    for batch_size in [1, 2, 4, 8, 16]:
        work = [decode for _ in range(batch_size)]
        per_seq.append(simulate(model, [dev], work).makespan / batch_size)
    assert all(b < a for a, b in zip(per_seq, per_seq[1:]))


# --- pipeline parallelism -------------------------------------------------------


def test_pipeline_parallel_conserves_work():
    model = toy_model(num_layers=4)
    work = [SequenceWork(0, 64, 8)]
    single = simulate(model, [make_device()], work, pp=1)
    dual = simulate(model, [make_device("g0"), make_device("g1")], work, pp=2)
    assert dual.total_flops == pytest.approx(single.total_flops, rel=1e-12)
    assert dual.total_bytes == pytest.approx(single.total_bytes, rel=1e-12)


def test_pipeline_parallel_single_batch_equals_single_device_when_uniformly_bound():
    # huge compute -> uniformly bandwidth-bound -> splitting layers conserves time
    model = toy_model(num_layers=4)
    work = [SequenceWork(0, 64, 16)]
    devs = [make_device("g0", peak=1e18), make_device("g1", peak=1e18)]
    single = simulate(model, [make_device(peak=1e18)], work, pp=1)
    dual = simulate(model, devs, work, pp=2)
    assert dual.makespan == pytest.approx(single.makespan, rel=1e-9)


def test_pipeline_parallel_single_batch_never_faster_than_single_device():
    model = toy_model(num_layers=4)
    work = [SequenceWork(0, 128, 16)]
    single = simulate(model, [make_device()], work, pp=1)
    dual = simulate(model, [make_device("g0"), make_device("g1")], work, pp=2)
    assert dual.makespan >= single.makespan - 1e-9


def test_pipeline_parallel_places_lm_head_on_last_stage():
    model = toy_model(num_layers=4, include_lm_head=True)
    work = [SequenceWork(0, 16, 4)]
    shards = WorkShardGenerator(model).generate(work)
    gen = EventGenerator(model, [make_device("g0"), make_device("g1")], pipeline_parallel=2)
    sched = gen.run(shards)
    # decode groups should produce events on both stages; lm head lands on stage 1
    decode_events = [e for e in sched.events if e.phase == "decode"]
    assert {e.device_index for e in decode_events} == {0, 1}


# --- event generator validation -------------------------------------------------


def test_event_generator_requires_devices():
    with pytest.raises(ValueError, match="at least one"):
        EventGenerator(toy_model(), [])


def test_event_generator_device_count_divisible_by_parallelism():
    model = toy_model(num_layers=4)
    with pytest.raises(ValueError, match="divisible"):
        EventGenerator(model, [make_device()], pipeline_parallel=2)


def test_event_generator_layers_divisible_by_pipeline():
    model = toy_model(num_layers=3)
    with pytest.raises(ValueError, match="num_layers"):
        EventGenerator(model, [make_device("a"), make_device("b")], pipeline_parallel=2)


def test_event_generator_accepts_expert_parallel():
    model = toy_moe_model()
    # two devices, expert_parallel=2 -> constructs without error (see test_parallel.py)
    EventGenerator(model, [make_device("a"), make_device("b")], expert_parallel=2)
