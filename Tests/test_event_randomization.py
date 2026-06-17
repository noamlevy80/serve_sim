"""Event-time randomization tests (PRD "Event time randomization").

To model system randomness the event generator multiplies each event's
calculated roofline time by ``1 + U(-range, range)`` -- one draw per event. The
perturbation is applied to the *times* (and hence the effective rate the arbiter
rescales under contention), never to the FLOPs/bytes, so total work is conserved
and the long-run mean is the unperturbed duration. With ``range == 0`` (the
default) the path is byte-identical to the pure roofline, so every other test in
the suite is unaffected.
"""

from __future__ import annotations

import random
import statistics

import pytest

from serve_sim.model import toy_model, toy_moe_model
from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.shards import WorkShardGenerator
from serve_sim.tracker import SequenceWork
from serve_sim.events import EventGenerator
from serve_sim.arbiter import IncrementalArbiter
from serve_sim.tiering import build_activation_trace
from serve_sim.system import Network, Node, System
from serve_sim.orchestrator import Request, Simulator, StrategyConfig


# --- helpers --------------------------------------------------------------------


def make_memory(name="mem", bw=1e12, cap=80e9):
    return MemoryDevice(name, capacity_bytes=cap, bandwidth_bytes_per_s=bw)


def make_device(name="gpu", memory=None, peak=100e12, bw=1e12, cap=80e9, launch=0.0):
    memory = memory or make_memory(f"{name}-mem", bw=bw, cap=cap)
    return ComputeDevice(
        name, peak_flops_fp16=peak, first_tier_memory=memory, kernel_launch_latency=launch
    )


def make_two_tier(name="gpu", peak=1e14, tier1_bw=2e12, nvm_bw=2e11):
    t1 = make_memory(f"{name}-hbm", bw=tier1_bw)
    nvm = make_memory(f"{name}-nvm", bw=nvm_bw, cap=1e12)
    return ComputeDevice(name, peak_flops_fp16=peak, first_tier_memory=t1, second_tier_memory=nvm)


def make_system(num_devices=1):
    nodes = []
    for i in range(num_devices):
        mem = make_memory(f"hbm-{i}", bw=2e12)
        dev = ComputeDevice(f"gpu-{i}", peak_flops_fp16=100e12, first_tier_memory=mem)
        nodes.append(Node(f"n{i}", (dev,), make_memory(f"node-mem-{i}", bw=5e11)))
    net = Network(1e12, 1e-6, 1e11, 1e-7)
    return System("sys", net, make_memory("nvm", bw=5e9), tuple(nodes))


def compute_events(schedule):
    return [e for e in schedule.events if e.phase in ("prefill", "decode")]


# --- disabled by default is byte-identical --------------------------------------


def test_default_range_is_byte_identical():
    # No randomization argument -> exactly the pure roofline events.
    model = toy_model()
    dev = make_device()
    work = [SequenceWork(0, 64, 8)]
    shards = WorkShardGenerator(model).generate(work)

    base = EventGenerator(model, [dev]).run(shards)
    again = EventGenerator(
        model, [dev], event_random_factor_range=0.0, rng=random.Random(123)
    ).run(shards)

    assert [e.duration for e in base.events] == [e.duration for e in again.events]
    assert [e.compute_time for e in base.events] == [e.compute_time for e in again.events]
    assert [e.bandwidth_time for e in base.events] == [e.bandwidth_time for e in again.events]


# --- perturbation bounds & shape ------------------------------------------------


def test_durations_stay_within_factor_band():
    model = toy_model()
    dev = make_device()
    work = [SequenceWork(0, 64, 8)]
    shards = WorkShardGenerator(model).generate(work)

    base = {e.group_index: e.duration for e in compute_events(
        EventGenerator(model, [dev]).run(shards)
    )}
    r = 0.1
    rnd = compute_events(
        EventGenerator(model, [dev], event_random_factor_range=r, rng=random.Random(1)).run(shards)
    )
    perturbed = False
    for e in rnd:
        lo, hi = base[e.group_index] * (1 - r), base[e.group_index] * (1 + r)
        assert lo <= e.duration <= hi
        if e.duration != base[e.group_index]:
            perturbed = True
    assert perturbed  # at least some events actually moved


def test_roofline_max_shape_preserved():
    # Both compute_time and bandwidth_time scale by the same per-event factor, so
    # duration == max(compute_time, bandwidth_time) still holds.
    model = toy_model()
    dev = make_device()
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 48, 6)])
    sched = EventGenerator(
        model, [dev], event_random_factor_range=0.2, rng=random.Random(7)
    ).run(shards)
    for e in compute_events(sched):
        assert e.duration == pytest.approx(max(e.compute_time, e.bandwidth_time))


# --- conservation ---------------------------------------------------------------


def test_total_work_is_conserved():
    model = toy_model()
    dev = make_device()
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 64, 8)])
    base = EventGenerator(model, [dev]).run(shards)
    rnd = EventGenerator(
        model, [dev], event_random_factor_range=0.3, rng=random.Random(99)
    ).run(shards)
    assert rnd.total_flops == pytest.approx(base.total_flops)
    assert rnd.total_bytes == pytest.approx(base.total_bytes)


# --- reproducibility ------------------------------------------------------------


def test_same_seed_is_reproducible():
    model = toy_model()
    dev = make_device()
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 64, 8)])
    a = EventGenerator(model, [dev], event_random_factor_range=0.1, rng=random.Random(5)).run(shards)
    b = EventGenerator(model, [dev], event_random_factor_range=0.1, rng=random.Random(5)).run(shards)
    assert [e.duration for e in a.events] == [e.duration for e in b.events]


def test_different_seed_differs():
    model = toy_model()
    dev = make_device()
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 64, 8)])
    a = EventGenerator(model, [dev], event_random_factor_range=0.1, rng=random.Random(1)).run(shards)
    b = EventGenerator(model, [dev], event_random_factor_range=0.1, rng=random.Random(2)).run(shards)
    assert [e.duration for e in a.events] != [e.duration for e in b.events]


# --- statistical mean -----------------------------------------------------------


def test_mean_factor_is_one():
    # The symmetric uniform factor averages out: the mean perturbed duration of a
    # single event over many draws is within a couple of std-errors of the base.
    model = toy_model()
    dev = make_device()
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 16, 0)])
    base = compute_events(EventGenerator(model, [dev]).run(shards))[0].duration

    r = 0.4
    rng = random.Random(0)
    samples = []
    for _ in range(4000):
        sched = EventGenerator(model, [dev], event_random_factor_range=r, rng=rng).run(shards)
        samples.append(compute_events(sched)[0].duration)
    mean = statistics.fmean(samples)
    # uniform(-r, r) has std r/sqrt(3); std-error of the mean over N draws.
    std_err = base * (r / (3 ** 0.5)) / (len(samples) ** 0.5)
    assert abs(mean - base) < 4 * std_err


# --- applies to all event kinds -------------------------------------------------


def test_kernel_launch_latency_is_scaled():
    model = toy_model(num_layers=2)
    dev = make_device(launch=1e-6)
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 8, 2)])
    base = EventGenerator(model, [dev]).run(shards)
    rnd = EventGenerator(
        model, [dev], event_random_factor_range=0.25, rng=random.Random(3)
    ).run(shards)
    base_launch = [e.duration for e in base.events if e.phase == "kernel_launch"]
    rnd_launch = [e.duration for e in rnd.events if e.phase == "kernel_launch"]
    assert base_launch  # there are launch events
    assert rnd_launch != base_launch
    for d, b in zip(rnd_launch, base_launch):
        assert b * 0.75 <= d <= b * 1.25


def test_transfer_event_is_scaled():
    model = toy_moe_model(num_layers=2)
    dev = make_two_tier("a")
    work = [SequenceWork(0, 32, 4)]
    trace = build_activation_trace(model, work, seed=0)
    cap = max(len(g.active_experts) for g in trace)
    shards = WorkShardGenerator(model).generate(work)

    base = EventGenerator(model, [dev]).run(
        shards, expert_trace=trace, expert_cache_capacity=cap
    )
    rnd = EventGenerator(
        model, [dev], event_random_factor_range=0.3, rng=random.Random(11)
    ).run(shards, expert_trace=trace, expert_cache_capacity=cap)

    base_xfer = [e for e in base.events if e.phase == "transfer"]
    rnd_xfer = [e for e in rnd.events if e.phase == "transfer"]
    assert base_xfer  # the two-tier MoE streams experts
    # bytes conserved, duration perturbed within band.
    for d, b in zip(rnd_xfer, base_xfer):
        assert d.bytes_read == pytest.approx(b.bytes_read)
        assert b.duration * 0.7 <= d.duration <= b.duration * 1.3
    assert [e.duration for e in rnd_xfer] != [e.duration for e in base_xfer]


# --- arbiter passthrough --------------------------------------------------------


def test_randomization_survives_the_arbiter():
    # A single randomized job retimed by the arbiter keeps its perturbed makespan
    # (the arbiter rescales from the scaled rates, not the raw roofline).
    model = toy_model()
    dev = make_device()
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 64, 8)])

    base_makespan = EventGenerator(model, [dev]).run(shards).makespan

    arb = IncrementalArbiter()
    arb.admit(
        EventGenerator(model, [dev], event_random_factor_range=0.15, rng=random.Random(8)),
        shards,
    )
    arb.run_to_idle()
    perturbed = arb.job_end_time(0)
    assert perturbed != pytest.approx(base_makespan)
    assert base_makespan * 0.85 <= perturbed <= base_makespan * 1.15


# --- simulator integration ------------------------------------------------------


def test_simulator_default_is_unperturbed():
    model = toy_model()
    system = make_system(1)
    req = Request(0, model, prompt_tokens=64, output_tokens=8)
    a = Simulator(system, StrategyConfig()).run([req]).makespan
    b = Simulator(system, StrategyConfig()).run([req]).makespan
    assert a == pytest.approx(b)


def test_simulator_randomization_is_seed_reproducible():
    model = toy_model()
    system = make_system(1)
    reqs = [Request(i, model, prompt_tokens=48, output_tokens=6) for i in range(3)]
    strat = StrategyConfig(
        max_batch_size=1, event_random_factor_range=0.1, random_seed=2024
    )
    m1 = Simulator(system, strat).run(reqs).makespan
    m2 = Simulator(system, strat).run(reqs).makespan
    assert m1 == pytest.approx(m2)


def test_simulator_different_seed_changes_makespan():
    model = toy_model()
    system = make_system(1)
    reqs = [Request(i, model, prompt_tokens=48, output_tokens=6) for i in range(3)]
    base = Simulator(
        system, StrategyConfig(event_random_factor_range=0.1, random_seed=1)
    ).run(reqs).makespan
    other = Simulator(
        system, StrategyConfig(event_random_factor_range=0.1, random_seed=2)
    ).run(reqs).makespan
    assert base != pytest.approx(other)


# --- validation -----------------------------------------------------------------


def test_event_generator_rejects_bad_range():
    model = toy_model()
    dev = make_device()
    with pytest.raises(ValueError):
        EventGenerator(model, [dev], event_random_factor_range=-0.1)
    with pytest.raises(ValueError):
        EventGenerator(model, [dev], event_random_factor_range=1.0)


def test_strategy_rejects_bad_range():
    with pytest.raises(ValueError):
        StrategyConfig(event_random_factor_range=-0.01)
    with pytest.raises(ValueError):
        StrategyConfig(event_random_factor_range=1.0)
