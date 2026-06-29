"""Communication-collective tests: the scale-up-network cost of parallelism.

Sharding a forward pass forces the ranks to exchange activations over the
scale-up network. The event generator models three collectives as fixed-duration
barriers (``latency + volume / bandwidth``), inserted after each stage's compute:

* tensor parallelism -> two ``all-reduce`` per layer (``phase == "tp_comm"``);
* expert parallelism -> two ``all-to-all`` per MoE layer (``phase == "ep_comm"``);
* pipeline parallelism -> one point-to-point hand-off per stage (``phase ==
  "pp_comm"``).

When no scale-up bandwidth is configured, or the engine is unsharded, no
collectives are emitted and the schedule is byte-identical to the pure-roofline
path.
"""

from __future__ import annotations

import pytest

from serve_sim.model import toy_model, toy_moe_model
from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.shards import WorkShardGenerator
from serve_sim.tracker import SequenceWork
from serve_sim.events import EventGenerator
from serve_sim.arbiter import ResourceArbiter
from serve_sim.orchestrator import Simulator, StrategyConfig, Request
from serve_sim.report import device_summaries, sequence_table
from test_orchestrator import make_system

BW = 1e12
LAT = 1e-6

_COMM_PHASES = ("tp_comm", "ep_comm", "pp_comm")


def make_device(name="gpu", peak=1e14, bw=2e12, cap=80e9):
    mem = MemoryDevice(f"{name}-hbm", capacity_bytes=cap, bandwidth_bytes_per_s=bw)
    return ComputeDevice(name, peak_flops_fp16=peak, first_tier_memory=mem)


def devices(n):
    return [make_device(f"gpu{i}") for i in range(n)]


def group_tokens(shards):
    """Tokens per forward-pass group (the activation width its layers exchange)."""
    tokens: dict[int, int] = {}
    for s in shards:
        if s.kind == "layer":
            tokens[s.group_index] = max(tokens.get(s.group_index, 0), s.tokens)
    return tokens


def run(model, devs, shards, *, pp=1, ep=1, tp=1, comm=True):
    gen = EventGenerator(
        model, devs, pipeline_parallel=pp, expert_parallel=ep, tensor_parallel=tp,
        scale_up_bandwidth_bytes_per_s=BW if comm else None,
        scale_up_latency_s=LAT,
    )
    return gen.run(shards)


# --- collectives are absent when they should be ---------------------------------


def test_unsharded_engine_emits_no_collectives():
    model = toy_model()
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 64, 8)])
    schedule = run(model, devices(1), shards, pp=1, ep=1, tp=1)
    assert not [e for e in schedule.events if e.phase in _COMM_PHASES]


def test_no_collectives_without_scale_up_bandwidth():
    # tp=2 but no network configured -> communication is not modeled and the
    # schedule matches the pure-roofline path byte-for-byte.
    model = toy_model()
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 64, 8)])
    bare = run(model, devices(2), shards, tp=2, comm=False)
    assert not [e for e in bare.events if e.phase in _COMM_PHASES]
    with_comm = run(model, devices(2), shards, tp=2, comm=True)
    assert with_comm.makespan > bare.makespan


# --- tensor parallelism: all-reduce ---------------------------------------------


def test_tensor_parallel_adds_two_allreduces_per_layer():
    model = toy_model(num_layers=4)
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 64, 8)])
    bare = run(model, devices(2), shards, tp=2, comm=False)
    full = run(model, devices(2), shards, tp=2, comm=True)

    H, pdb, L = model.hidden_size, model.param_dtype_bytes, model.num_layers

    def allreduce(num_bytes):
        return LAT + (2 * (2 - 1) / 2) * num_bytes / BW

    expected = sum(
        2 * L * allreduce(t * H * pdb) for t in group_tokens(shards).values()
    )
    assert full.makespan - bare.makespan == pytest.approx(expected)
    # one tp_comm barrier event per tp rank, per (group, layer-pass batch).
    tp_events = [e for e in full.events if e.phase == "tp_comm"]
    assert tp_events
    assert all(e.phase == "tp_comm" for e in tp_events)


def test_tp_collective_is_a_barrier_across_all_ranks():
    model = toy_model(num_layers=2)
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 16, 0)])
    schedule = run(model, devices(4), shards, tp=4)
    tp_events = [e for e in schedule.events if e.phase == "tp_comm"]
    # The single prefill group's all-reduce occupies every one of the 4 ranks for
    # an identical window (a collective ends only when all participants arrive).
    assert {e.device_index for e in tp_events} == {0, 1, 2, 3}
    assert len({(e.start, e.end) for e in tp_events}) == 1


# --- expert parallelism: all-to-all ---------------------------------------------


def test_expert_parallel_adds_alltoall_only_on_moe_layers():
    moe = toy_moe_model(num_layers=4, num_dense_layers=0)
    shards = WorkShardGenerator(moe).generate([SequenceWork(0, 32, 4)])
    bare = run(moe, devices(2), shards, ep=2, comm=False)
    full = run(moe, devices(2), shards, ep=2, comm=True)

    H, pdb = moe.hidden_size, moe.param_dtype_bytes
    n_moe = moe.num_layers  # every layer is MoE

    def alltoall(num_bytes):
        return LAT + (2 - 1) / 2 * num_bytes / BW

    expected = sum(
        2 * n_moe * alltoall(t * H * pdb) for t in group_tokens(shards).values()
    )
    assert full.makespan - bare.makespan == pytest.approx(expected)
    assert [e for e in full.events if e.phase == "ep_comm"]
    assert not [e for e in full.events if e.phase in ("tp_comm", "pp_comm")]


def test_dense_model_has_no_expert_collectives():
    model = toy_model()
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 32, 4)])
    schedule = run(model, devices(2), shards, ep=2)
    assert not [e for e in schedule.events if e.phase == "ep_comm"]


# --- pipeline parallelism: point-to-point ---------------------------------------


def test_pipeline_parallel_adds_point_to_point_between_stages():
    model = toy_model(num_layers=4)
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 48, 4)])
    bare = run(model, devices(2), shards, pp=2, comm=False)
    full = run(model, devices(2), shards, pp=2, comm=True)

    H, pdb = model.hidden_size, model.param_dtype_bytes

    def p2p(num_bytes):
        return LAT + num_bytes / BW

    # Two stages -> one hand-off per forward-pass group.
    expected = sum(p2p(t * H * pdb) for t in group_tokens(shards).values())
    assert full.makespan - bare.makespan == pytest.approx(expected)
    pp_events = [e for e in full.events if e.phase == "pp_comm"]
    assert len(pp_events) == len(group_tokens(shards))
    # The hand-off lands on the receiving (second) stage.
    assert all(e.device_index == 1 for e in pp_events)


# --- collectives are fixed-duration (no network congestion modeled) -------------


def test_comm_events_carry_no_compute_or_bytes():
    model = toy_model(num_layers=2)
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 16, 2)])
    schedule = run(model, devices(2), shards, tp=2)
    for e in schedule.events:
        if e.phase in _COMM_PHASES:
            assert e.flops == 0.0
            assert e.bytes_read == 0.0
            assert e.compute_time == 0.0
            assert e.duration > 0.0


def test_collectives_are_not_rescaled_under_contention():
    # Two identical jobs share the same devices: compute contends and stretches,
    # but the comm barriers are fixed network waits and keep their duration.
    model = toy_model(num_layers=2)
    shards = WorkShardGenerator(model).generate([SequenceWork(0, 32, 0)])
    devs = devices(2)
    isolated = run(model, devs, shards, tp=2)
    iso_comm = sorted(
        e.duration for e in isolated.events if e.phase == "tp_comm"
    )

    arbiter = ResourceArbiter()
    for _ in range(2):
        gen = EventGenerator(
            model, devs, tensor_parallel=2,
            scale_up_bandwidth_bytes_per_s=BW, scale_up_latency_s=LAT,
        )
        arbiter.add_job(gen, shards)
    result = arbiter.run()

    co_comm = sorted(
        e.duration
        for sched in result.schedules
        for e in sched.events
        if e.phase == "tp_comm"
    )
    assert co_comm == pytest.approx(iso_comm + iso_comm)


# --- end to end through the simulator -------------------------------------------


def test_simulator_emits_tensor_parallel_collectives():
    model = toy_model()
    system = make_system(2)
    req = Request(0, model, 64, 8, 0.0)
    strat = StrategyConfig(
        max_batch_size=1, pipeline_parallel=1, expert_parallel=1, tensor_parallel=2
    )
    result = Simulator(system, strat).run([req])
    assert "tp_comm" in {e.phase for e in result.events}
    # The communicating device-state surfaces in the per-device report.
    summaries = device_summaries(result)
    assert any(s["communicating_fraction"] > 0 for s in summaries)
    # The per-sequence breakdown reports the time spent in collectives.
    rows = sequence_table(result)
    assert len(rows) == 1
    assert rows[0]["comm_wait_s"] > 0


def test_unsharded_sequence_has_no_comm_wait():
    model = toy_model()
    system = make_system(2)
    req = Request(0, model, 64, 8, 0.0)
    strat = StrategyConfig(
        max_batch_size=1, pipeline_parallel=1, expert_parallel=1, tensor_parallel=1
    )
    result = Simulator(system, strat).run([req])
    rows = sequence_table(result)
    assert len(rows) == 1
    assert rows[0]["comm_wait_s"] == 0.0

