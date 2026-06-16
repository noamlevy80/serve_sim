"""Event-rescaling tests: several event generators sharing resources.

A :class:`ResourceArbiter` co-runs multiple :class:`EventGenerator` jobs on one
shared timeline. When concurrent events demand the same compute device or memory
device, that resource is divided equally and in-progress events are rescaled,
prorated for the time already elapsed (PRD "Event Rescaling"). These tests drive
that with more than one generator and check the rescaled timings against
closed-form expectations, for both compute events and data-transfer events.
"""

from __future__ import annotations

import pytest

from serve_sim.model import toy_model, toy_moe_model
from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.shards import WorkShardGenerator
from serve_sim.tracker import SequenceWork
from serve_sim.events import EventGenerator
from serve_sim.arbiter import ResourceArbiter
from serve_sim.tiering import build_activation_trace


# --- helpers --------------------------------------------------------------------


def make_memory(name="mem", bw=1e12, cap=80e9):
    return MemoryDevice(name, capacity_bytes=cap, bandwidth_bytes_per_s=bw)


def make_device(name="gpu", memory=None, peak=100e12, bw=1e12, cap=80e9, launch=0.0):
    memory = memory or make_memory(f"{name}-mem", bw=bw, cap=cap)
    return ComputeDevice(
        name, peak_flops_fp16=peak, first_tier_memory=memory, kernel_launch_latency=launch
    )


def make_two_tier(name="gpu", nvm=None, peak=1e14, tier1_bw=2e12, nvm_bw=2e11):
    t1 = make_memory(f"{name}-hbm", bw=tier1_bw)
    nvm = nvm or make_memory(f"{name}-nvm", bw=nvm_bw, cap=1e12)
    return ComputeDevice(name, peak_flops_fp16=peak, first_tier_memory=t1, second_tier_memory=nvm)


def gen(model, devices, **kw):
    return EventGenerator(model, devices, **kw)


# --- single job reproduces the standalone generator -----------------------------


def test_single_job_reproduces_run():
    model = toy_model()
    dev = make_device()
    work = [SequenceWork(0, 64, 8)]
    shards = WorkShardGenerator(model).generate(work)
    solo = gen(model, [dev]).run(shards)

    arb = ResourceArbiter()
    arb.add_job(gen(model, [dev]), shards)
    result = arb.run()

    assert result.makespan == pytest.approx(solo.makespan, rel=1e-9)
    assert len(result.schedules[0].events) == len(solo.events)
    for a, b in zip(result.schedules[0].events, solo.events):
        assert a.start == pytest.approx(b.start, rel=1e-9, abs=1e-18)
        assert a.end == pytest.approx(b.end, rel=1e-9, abs=1e-18)


# --- compute-event rescaling ----------------------------------------------------


def test_two_jobs_sharing_device_double_bandwidth_bound_time():
    # decode-heavy, huge compute -> bandwidth-bound; both jobs share one device.
    model = toy_model()
    dev = make_device(peak=1e18, bw=1e12)
    work = [SequenceWork(cached_tokens=64, prefill_tokens=0, decode_tokens=16)]
    shards = WorkShardGenerator(model).generate(work)
    solo = gen(model, [dev]).run(shards).makespan

    arb = ResourceArbiter()
    arb.add_job(gen(model, [dev]), shards)
    arb.add_job(gen(model, [dev]), shards)
    result = arb.run()
    # two identical bandwidth-bound jobs on one memory each get half bandwidth.
    assert result.makespan == pytest.approx(2 * solo, rel=1e-9)


def test_two_jobs_sharing_device_double_compute_bound_time():
    # prefill-only, huge bandwidth -> compute-bound; share one compute device.
    model = toy_model()
    dev = make_device(peak=50e12, bw=1e18)
    work = [SequenceWork(0, 128, 0)]
    shards = WorkShardGenerator(model).generate(work)
    solo = gen(model, [dev]).run(shards).makespan

    arb = ResourceArbiter()
    arb.add_job(gen(model, [dev]), shards)
    arb.add_job(gen(model, [dev]), shards)
    result = arb.run()
    assert result.makespan == pytest.approx(2 * solo, rel=1e-9)


@pytest.mark.parametrize("n_jobs", [1, 2, 3, 4])
def test_n_way_sharing_scales_linearly(n_jobs):
    model = toy_model()
    dev = make_device(peak=1e18, bw=1e12)  # bandwidth-bound
    work = [SequenceWork(64, 0, 8)]
    shards = WorkShardGenerator(model).generate(work)
    solo = gen(model, [dev]).run(shards).makespan

    arb = ResourceArbiter()
    for _ in range(n_jobs):
        arb.add_job(gen(model, [dev]), shards)
    result = arb.run()
    assert result.makespan == pytest.approx(n_jobs * solo, rel=1e-9)


def test_independent_devices_do_not_contend():
    # two jobs on disjoint devices run concurrently: makespan is the max, not sum.
    model = toy_model()
    work = [SequenceWork(64, 0, 8)]
    shards = WorkShardGenerator(model).generate(work)
    da = make_device("a", peak=1e18, bw=1e12)
    db = make_device("b", peak=1e18, bw=1e12)
    solo = gen(model, [da]).run(shards).makespan

    arb = ResourceArbiter()
    arb.add_job(gen(model, [da]), shards)
    arb.add_job(gen(model, [db]), shards)
    result = arb.run()
    assert result.makespan == pytest.approx(solo, rel=1e-9)


def test_shared_memory_only_contends_on_bandwidth_not_compute():
    # two compute devices, one shared memory; compute-bound work does not contend.
    model = toy_model()
    shared = make_memory("shared", bw=1e18)  # huge bw -> compute is the bound
    da = make_device("a", memory=shared, peak=50e12)
    db = make_device("b", memory=shared, peak=50e12)
    work = [SequenceWork(0, 128, 0)]
    shards = WorkShardGenerator(model).generate(work)
    solo = gen(model, [da]).run(shards).makespan

    arb = ResourceArbiter()
    arb.add_job(gen(model, [da]), shards)
    arb.add_job(gen(model, [db]), shards)
    result = arb.run()
    # distinct compute pools, shared memory not the bottleneck -> no slowdown.
    assert result.makespan == pytest.approx(solo, rel=1e-9)


# --- proration ------------------------------------------------------------------


def test_staggered_start_prorates_in_flight_event():
    # PRD example: a memory device serving one compute event is rescaled to half
    # bandwidth when a second event joins, prorated for the elapsed time.
    # One shared memory; two compute devices. Device B launches tau late.
    model = toy_model(num_layers=1, include_lm_head=False)
    shared = make_memory("shared", bw=1e12)
    da = make_device("a", memory=shared, peak=1e18, launch=0.0)
    work = [SequenceWork(cached_tokens=64, prefill_tokens=0, decode_tokens=1)]
    shards = WorkShardGenerator(model).generate(work)

    solo = gen(model, [da]).run(shards).makespan  # == W / R for the single event
    tau = 0.3 * solo
    db = make_device("b", memory=shared, peak=1e18, launch=tau)

    arb = ResourceArbiter()
    arb.add_job(gen(model, [da]), shards)  # job 0: starts at 0
    arb.add_job(gen(model, [db]), shards)  # job 1: compute starts at tau
    result = arb.run()

    # While both overlap each gets W/2 bandwidth; closed form gives makespan 2W/R
    # and the early job's (prorated) end at 2W/R - tau.
    assert result.makespan == pytest.approx(2 * solo, rel=1e-9)
    job0_compute = next(e for e in result.schedules[0].events if e.phase == "decode")
    assert job0_compute.end == pytest.approx(2 * solo - tau, rel=1e-9)


# --- data-transfer rescaling ----------------------------------------------------


def _moe_setup(num_layers=2):
    model = toy_moe_model(num_layers=num_layers)
    work = [SequenceWork(0, 32, 4)]
    trace = build_activation_trace(model, work, seed=0)
    cap = max(len(g.active_experts) for g in trace)
    shards = WorkShardGenerator(model).generate(work)
    return model, work, trace, cap, shards


def test_two_jobs_sharing_nvm_double_transfer_time():
    # two MoE jobs streaming experts from the SAME system NVM contend on it.
    model, work, trace, cap, shards = _moe_setup()
    dev = make_two_tier("a")  # both jobs use this device (shared nvm + tiers)
    solo = gen(model, [dev]).run(shards, expert_trace=trace, expert_cache_capacity=cap)
    solo_first_transfer = next(e for e in solo.events if e.phase == "transfer")

    arb = ResourceArbiter()
    arb.add_job(gen(model, [dev]), shards, expert_trace=trace, expert_cache_capacity=cap)
    arb.add_job(gen(model, [dev]), shards, expert_trace=trace, expert_cache_capacity=cap)
    result = arb.run()

    # the group-0 transfers both start at t=0 on the shared NVM -> each doubles.
    job0_first_transfer = next(
        e for e in result.schedules[0].events if e.phase == "transfer"
    )
    assert job0_first_transfer.duration == pytest.approx(
        2 * solo_first_transfer.duration, rel=1e-9
    )


def test_transfers_on_separate_nvm_do_not_contend():
    model, work, trace, cap, shards = _moe_setup()
    da = make_two_tier("a")  # its own nvm
    db = make_two_tier("b")  # a different nvm instance
    solo = gen(model, [da]).run(shards, expert_trace=trace, expert_cache_capacity=cap)
    solo_first_transfer = next(e for e in solo.events if e.phase == "transfer")

    arb = ResourceArbiter()
    arb.add_job(gen(model, [da]), shards, expert_trace=trace, expert_cache_capacity=cap)
    arb.add_job(gen(model, [db]), shards, expert_trace=trace, expert_cache_capacity=cap)
    result = arb.run()

    job0_first_transfer = next(
        e for e in result.schedules[0].events if e.phase == "transfer"
    )
    # disjoint NVMs -> transfers keep their standalone duration.
    assert job0_first_transfer.duration == pytest.approx(
        solo_first_transfer.duration, rel=1e-9
    )


# --- conservation ---------------------------------------------------------------


def test_rescaling_conserves_total_work():
    model = toy_model()
    dev = make_device(peak=1e18, bw=1e12)
    work = [SequenceWork(0, 96, 12)]
    shards = WorkShardGenerator(model).generate(work)
    solo = gen(model, [dev]).run(shards)

    arb = ResourceArbiter()
    arb.add_job(gen(model, [dev]), shards)
    arb.add_job(gen(model, [dev]), shards)
    result = arb.run()
    # rescaling changes timings, never the total FLOPs/bytes performed.
    assert result.total_flops == pytest.approx(2 * solo.total_flops, rel=1e-12)
    assert result.total_bytes == pytest.approx(2 * solo.total_bytes, rel=1e-12)


def test_each_job_keeps_its_own_event_stream():
    model = toy_model()
    dev = make_device(peak=1e18, bw=1e12)
    work = [SequenceWork(0, 32, 4)]
    shards = WorkShardGenerator(model).generate(work)
    solo = gen(model, [dev]).run(shards)

    arb = ResourceArbiter()
    arb.add_job(gen(model, [dev]), shards)
    arb.add_job(gen(model, [dev]), shards)
    result = arb.run()
    assert len(result.schedules) == 2
    for schedule in result.schedules:
        assert len(schedule.events) == len(solo.events)


def test_sharing_never_faster_than_isolation():
    model = toy_model()
    dev = make_device(peak=2e14, bw=1.5e12)
    work = [SequenceWork(0, 64, 16)]
    shards = WorkShardGenerator(model).generate(work)
    solo = gen(model, [dev]).run(shards).makespan

    arb = ResourceArbiter()
    arb.add_job(gen(model, [dev]), shards)
    arb.add_job(gen(model, [dev]), shards)
    result = arb.run()
    assert result.makespan >= solo - 1e-15
