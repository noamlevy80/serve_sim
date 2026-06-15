"""Kernel-launch tests: launch work shards and the latency events they incur.

The work-shard generator emits one kernel-launch marker per forward-pass group;
the event generator charges each participating device its
``kernel_launch_latency`` once per group-stage before that stage's compute.
Devices with zero launch overhead emit no kernel-launch event, so a model run on
them is byte-identical to the pure-roofline path.
"""

from __future__ import annotations

import pytest

from serve_sim.model import toy_model, toy_moe_model
from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.shards import WorkShardGenerator
from serve_sim.tracker import SequenceWork
from serve_sim.events import EventGenerator
from reference import reference_roofline


def make_device(name="gpu", peak=100e12, bw=2e12, cap=80e9, launch=0.0):
    mem = MemoryDevice(f"{name}-hbm", capacity_bytes=cap, bandwidth_bytes_per_s=bw)
    return ComputeDevice(
        name, peak_flops_fp16=peak, first_tier_memory=mem, kernel_launch_latency=launch
    )


def num_groups(shards):
    return len({s.group_index for s in shards if s.kind != "kernel_launch"})


# --- kernel-launch work shards --------------------------------------------------


def test_generator_emits_one_kernel_launch_shard_per_group():
    model = toy_model(num_layers=2)
    work = [SequenceWork(0, 10, 3)]
    shards = WorkShardGenerator(model).generate(work, prefill_chunk_size=4)
    launches = [s for s in shards if s.kind == "kernel_launch"]
    assert len(launches) == num_groups(shards)
    # exactly one launch marker per group, ordered with the group.
    assert {s.group_index for s in launches} == {
        s.group_index for s in shards if s.kind != "kernel_launch"
    }


def test_kernel_launch_shard_is_zero_cost():
    model = toy_model(num_layers=2)
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 4, 1)])
    launch = next(s for s in shards if s.kind == "kernel_launch")
    assert launch.phase == "kernel_launch"
    assert launch.flops == 0.0
    assert launch.bytes_read == 0.0
    assert launch.tokens == 0


def test_launch_markers_do_not_disturb_compute_shards():
    # phase/kind filters used elsewhere must still see only compute shards.
    model = toy_model(num_layers=3)
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 8, 2)])
    prefill = [s for s in shards if s.phase == "prefill"]
    decode = [s for s in shards if s.phase == "decode"]
    assert all(s.kind == "layer" for s in prefill)
    assert all(s.kind in ("layer", "lm_head") for s in decode)


# --- kernel-launch events -------------------------------------------------------


def test_zero_latency_device_emits_no_launch_events():
    model = toy_model()
    dev = make_device(launch=0.0)
    work = [SequenceWork(0, 64, 8)]
    shards = WorkShardGenerator(model).generate(work)
    schedule = EventGenerator(model, [dev]).run(shards)
    assert not [e for e in schedule.events if e.phase == "kernel_launch"]
    assert schedule.makespan == pytest.approx(reference_roofline(model, dev, work))


def test_kernel_launch_adds_one_latency_per_group():
    model = toy_moe_model()
    work = [SequenceWork(0, 96, 12)]
    shards = WorkShardGenerator(model).generate(work)
    groups = num_groups(shards)
    base = make_device(launch=0.0)
    delayed = make_device(launch=3e-6)
    base_makespan = EventGenerator(model, [base]).run(shards).makespan
    delayed_makespan = EventGenerator(model, [delayed]).run(shards).makespan
    assert delayed_makespan == pytest.approx(base_makespan + groups * 3e-6)


def test_one_launch_event_per_group_with_expected_duration():
    model = toy_model(num_layers=2)
    dev = make_device(launch=2e-6)
    work = [SequenceWork(0, 16, 4)]
    shards = WorkShardGenerator(model).generate(work)
    schedule = EventGenerator(model, [dev]).run(shards)
    launch_events = [e for e in schedule.events if e.phase == "kernel_launch"]
    assert len(launch_events) == num_groups(shards)
    assert all(e.duration == pytest.approx(2e-6) for e in launch_events)
    assert schedule.time_for_phase("kernel_launch") == pytest.approx(
        num_groups(shards) * 2e-6
    )


def test_launch_latency_matches_reference_plus_overhead():
    model = toy_model()
    work = [SequenceWork(0, 128, 16)]
    shards = WorkShardGenerator(model).generate(work)
    groups = num_groups(shards)
    dev = make_device(launch=5e-6)
    makespan = EventGenerator(model, [dev]).run(shards).makespan
    assert makespan == pytest.approx(
        reference_roofline(model, dev, work) + groups * 5e-6
    )


# --- launches under parallelism -------------------------------------------------


def test_pipeline_launches_once_per_stage_per_group():
    model = toy_model(num_layers=4)
    work = [SequenceWork(0, 32, 4)]
    shards = WorkShardGenerator(model).generate(work)
    groups = num_groups(shards)
    devs0 = [make_device(f"d{i}", launch=0.0) for i in range(2)]
    devsL = [make_device(f"d{i}", launch=4e-6) for i in range(2)]
    base = EventGenerator(model, devs0, pipeline_parallel=2).run(shards)
    delayed = EventGenerator(model, devsL, pipeline_parallel=2).run(shards)
    launch_events = [e for e in delayed.events if e.phase == "kernel_launch"]
    # one launch per (group, stage): 2 stages.
    assert len(launch_events) == groups * 2
    assert delayed.makespan == pytest.approx(base.makespan + groups * 2 * 4e-6)


def test_expert_parallel_launches_once_per_group():
    model = toy_moe_model(num_layers=2)
    work = [SequenceWork(0, 24, 4)]
    shards = WorkShardGenerator(model).generate(work)
    groups = num_groups(shards)
    devs0 = [make_device(f"d{i}", launch=0.0) for i in range(2)]
    devsL = [make_device(f"d{i}", launch=4e-6) for i in range(2)]
    base = EventGenerator(model, devs0, expert_parallel=2).run(shards)
    delayed = EventGenerator(model, devsL, expert_parallel=2).run(shards)
    launch_events = [e for e in delayed.events if e.phase == "kernel_launch"]
    # ranks of a stage launch concurrently -> one launch per group.
    assert len(launch_events) == groups
    assert delayed.makespan == pytest.approx(base.makespan + groups * 4e-6)
