"""Incremental-arbiter tests: jobs admitted over time, rescaled on the fly.

`IncrementalArbiter` drives the same fluid (processor-sharing) rescaling as the
batch `ResourceArbiter`, but jobs are admitted at the current clock and the
caller steps time forward with `advance_to`. Admitting every job at t=0 and
running to idle must reproduce the batch solver event-for-event; admitting a job
mid-flight must prorate the in-flight events for the time already elapsed. The
mid-flight cases are checked against closed forms derived independently of the
simulator.
"""

from __future__ import annotations

import pytest

from serve_sim.events import ComputeEvent, EventGenerator
from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.model import toy_model, toy_moe_model
from serve_sim.shards import WorkShardGenerator
from serve_sim.tiering import build_activation_trace
from serve_sim.tracker import SequenceWork
from serve_sim.arbiter import IncrementalArbiter, ResourceArbiter


# --- helpers --------------------------------------------------------------------


def make_memory(name="mem", bw=1e12, cap=80e9):
    return MemoryDevice(name, capacity_bytes=cap, bandwidth_bytes_per_s=bw)


def make_device(name="gpu", memory=None, peak=100e12, bw=1e12, cap=80e9):
    memory = memory or make_memory(f"{name}-mem", bw=bw, cap=cap)
    return ComputeDevice(name, peak_flops_fp16=peak, first_tier_memory=memory)


def make_two_tier(name="gpu", nvm=None, peak=1e14, tier1_bw=2e12, nvm_bw=2e11):
    t1 = make_memory(f"{name}-hbm", bw=tier1_bw)
    nvm = nvm or make_memory(f"{name}-nvm", bw=nvm_bw, cap=1e12)
    return ComputeDevice(name, peak_flops_fp16=peak, first_tier_memory=t1, second_tier_memory=nvm)


def gen(model, devices, **kw):
    return EventGenerator(model, devices, **kw)


def compute_event(flops, compute_time):
    """A single compute-bound event: rate = flops / compute_time."""

    return ComputeEvent(
        group_index=0,
        phase="decode",
        device_index=0,
        flops=flops,
        bytes_read=0.0,
        compute_time=compute_time,
        bandwidth_time=0.0,
        duration=compute_time,
        start=0.0,
        end=compute_time,
    )


def assert_schedules_match(a_result, b_result):
    assert len(a_result.schedules) == len(b_result.schedules)
    for sa, sb in zip(a_result.schedules, b_result.schedules):
        assert len(sa.events) == len(sb.events)
        for ea, eb in zip(sa.events, sb.events):
            assert ea.start == pytest.approx(eb.start)
            assert ea.end == pytest.approx(eb.end)
            assert ea.duration == pytest.approx(eb.duration)


# --- batch equivalence (admit all at t=0) --------------------------------------


def test_single_job_reproduces_batch():
    model = toy_model()
    dev = make_device()
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 64, 8)])

    batch = ResourceArbiter()
    batch.add_job(gen(model, [dev]), shards)
    bres = batch.run()

    inc = IncrementalArbiter()
    inc.admit(gen(model, [dev]), shards)
    inc.run_to_idle()

    assert_schedules_match(inc.result(), bres)
    assert inc.is_idle()


def test_two_jobs_shared_compute_match_batch():
    model = toy_model()
    dev = make_device()
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 64, 8)])

    batch = ResourceArbiter()
    batch.add_job(gen(model, [dev]), shards)
    batch.add_job(gen(model, [dev]), shards)
    bres = batch.run()

    inc = IncrementalArbiter()
    inc.admit(gen(model, [dev]), shards)
    inc.admit(gen(model, [dev]), shards)
    inc.run_to_idle()

    assert_schedules_match(inc.result(), bres)


def test_shared_memory_only_matches_batch():
    # Two jobs on distinct compute pools but the SAME first-tier memory.
    shared_mem = make_memory("shared", bw=1e12)
    devA = make_device("a", memory=shared_mem)
    devB = make_device("b", memory=shared_mem)
    model = toy_model()
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 64, 8)])

    batch = ResourceArbiter()
    batch.add_job(gen(model, [devA]), shards)
    batch.add_job(gen(model, [devB]), shards)
    bres = batch.run()

    inc = IncrementalArbiter()
    inc.admit(gen(model, [devA]), shards)
    inc.admit(gen(model, [devB]), shards)
    inc.run_to_idle()

    assert_schedules_match(inc.result(), bres)


def test_transfer_sharing_matches_batch():
    # Two MoE jobs streaming experts from the same shared NVM second tier.
    model = toy_moe_model(num_layers=2, num_experts=8)
    work = [SequenceWork(0, 32, 4)]
    trace = build_activation_trace(model, work, seed=0)
    cap = max(len(g.active_experts) for g in trace)
    shards = WorkShardGenerator(model).generate(work)
    dev = make_two_tier("a")  # both jobs share this device (and its NVM)

    batch = ResourceArbiter()
    batch.add_job(gen(model, [dev]), shards, expert_trace=trace, expert_cache_capacity=cap)
    batch.add_job(gen(model, [dev]), shards, expert_trace=trace, expert_cache_capacity=cap)
    bres = batch.run()

    inc = IncrementalArbiter()
    inc.admit(gen(model, [dev]), shards, expert_trace=trace, expert_cache_capacity=cap)
    inc.admit(gen(model, [dev]), shards, expert_trace=trace, expert_cache_capacity=cap)
    inc.run_to_idle()

    assert_schedules_match(inc.result(), bres)


def test_disjoint_devices_match_solo():
    model = toy_model()
    devA = make_device("a")
    devB = make_device("b")
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 64, 8)])
    solo = gen(model, [devA]).run(shards)

    inc = IncrementalArbiter()
    inc.admit(gen(model, [devA]), shards)
    inc.admit(gen(model, [devB]), shards)
    inc.run_to_idle()
    res = inc.result()

    assert res.schedules[0].makespan == pytest.approx(solo.makespan)
    assert res.schedules[1].makespan == pytest.approx(solo.makespan)


# --- mid-flight admission (the new behaviour) ----------------------------------


def test_midflight_admission_prorates_shared_compute():
    # Job A runs alone until tau, then job B (identical) joins and they share the
    # compute pool equally. Closed form: A ends at 2T - tau, B ends at 2T.
    dev = make_device()
    W, T = 1e12, 1.0  # rate R = W / T = 1e12
    tau = 0.25

    inc = IncrementalArbiter()
    inc.admit_events([compute_event(W, T)], [dev])
    inc.advance_to(tau)
    assert inc.time == pytest.approx(tau)
    inc.admit_events([compute_event(W, T)], [dev])
    inc.run_to_idle()

    res = inc.result()
    a_end = res.schedules[0].events[0].end
    b_end = res.schedules[1].events[0].end
    assert a_end == pytest.approx(2 * T - tau)
    assert b_end == pytest.approx(2 * T)
    assert res.makespan == pytest.approx(2 * T)


def test_midflight_admission_on_disjoint_resource_no_slowdown():
    devA = make_device("a")
    devB = make_device("b")
    W, T = 1e12, 1.0
    tau = 0.3

    inc = IncrementalArbiter()
    inc.admit_events([compute_event(W, T)], [devA])
    inc.advance_to(tau)
    inc.admit_events([compute_event(W, T)], [devB])
    inc.run_to_idle()

    res = inc.result()
    assert res.schedules[0].events[0].end == pytest.approx(T)
    assert res.schedules[1].events[0].end == pytest.approx(tau + T)


def test_admitting_at_zero_equals_batch_share_for_single_events():
    # Two identical single events admitted at t=0 sharing one pool: each 2T.
    dev = make_device()
    W, T = 1e12, 1.0

    inc = IncrementalArbiter()
    inc.admit_events([compute_event(W, T)], [dev])
    inc.admit_events([compute_event(W, T)], [dev])
    inc.run_to_idle()
    res = inc.result()

    assert res.schedules[0].events[0].end == pytest.approx(2 * T)
    assert res.schedules[1].events[0].end == pytest.approx(2 * T)


def test_late_admission_after_first_drains_runs_at_full_rate():
    # B admitted after A has fully finished: B sees an empty machine.
    dev = make_device()
    W, T = 1e12, 1.0

    inc = IncrementalArbiter()
    inc.admit_events([compute_event(W, T)], [dev])
    inc.advance_to(2 * T)  # well past A's finish at T
    assert inc.is_idle()
    inc.admit_events([compute_event(W, T)], [dev])
    inc.run_to_idle()

    res = inc.result()
    assert res.schedules[0].events[0].end == pytest.approx(T)
    assert res.schedules[1].events[0].start == pytest.approx(2 * T)
    assert res.schedules[1].events[0].end == pytest.approx(3 * T)


# --- stepping API --------------------------------------------------------------


def test_next_event_time_and_idle_transitions():
    dev = make_device()
    W, T = 1e12, 1.0

    inc = IncrementalArbiter()
    assert inc.is_idle()
    assert inc.next_event_time() is None

    inc.admit_events([compute_event(W, T)], [dev])
    assert not inc.is_idle()
    assert inc.active_count == 1
    assert inc.next_event_time() == pytest.approx(T)

    inc.advance_to(inc.next_event_time())
    assert inc.is_idle()
    assert inc.next_event_time() is None


def test_advance_to_stops_exactly_at_target():
    dev = make_device()
    inc = IncrementalArbiter()
    inc.admit_events([compute_event(1e12, 1.0)], [dev])
    inc.advance_to(0.4)
    assert inc.time == pytest.approx(0.4)
    assert not inc.is_idle()  # event still in flight


def test_num_jobs_counts_admissions():
    dev = make_device()
    inc = IncrementalArbiter()
    inc.admit_events([compute_event(1e12, 1.0)], [dev])
    inc.admit_events([compute_event(1e12, 1.0)], [dev])
    assert inc.num_jobs == 2


# --- conservation --------------------------------------------------------------


def test_total_work_conserved_under_midflight_sharing():
    model = toy_model()
    dev = make_device()
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 64, 8)])
    solo = gen(model, [dev]).run(shards)

    inc = IncrementalArbiter()
    inc.admit(gen(model, [dev]), shards)
    inc.advance_to(solo.makespan / 3.0)
    inc.admit(gen(model, [dev]), shards)
    inc.run_to_idle()
    res = inc.result()

    assert res.total_flops == pytest.approx(2 * solo.total_flops)
    assert res.total_bytes == pytest.approx(2 * solo.total_bytes)
    # Sharing is never faster than two serialized solo runs and never faster
    # than a single solo run.
    assert res.makespan >= solo.makespan
