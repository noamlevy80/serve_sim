"""Progress reporting: the run emits sequence/time updates as it retires work.

``Simulator.run(..., progress=cb)`` calls ``cb`` with a :class:`RunProgress` each
time the run completes one or more sequences, reporting how many of the suite's
sequences are done plus the elapsed simulation and wall-clock time. The runner's
:class:`ProgressReporter` formats those updates for the CLI. These tests assert
the monotonic, well-formed progress stream (single-pool and PDD) and the printed
output, all offline.
"""

from __future__ import annotations

import io

import pytest

from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.model import toy_model
from serve_sim.orchestrator import Request, RunEvent, RunProgress, Simulator, StrategyConfig
from serve_sim.runner import (
    BuildProgress,
    BuildProgressReporter,
    EventLogReporter,
    ProgressReporter,
)
from serve_sim.system import Network, Node, System


def make_memory(name="mem", bw=1e12, cap=80e9):
    return MemoryDevice(name, capacity_bytes=cap, bandwidth_bytes_per_s=bw)


def make_device(name="gpu", peak=100e12, bw=1e12, cap=80e9):
    return ComputeDevice(name, peak_flops_fp16=peak,
                         first_tier_memory=make_memory(f"{name}-mem", bw=bw, cap=cap))


def make_system(num_devices=1, cap=80e9):
    network = Network(
        scale_up_bandwidth_bytes_per_s=1e12,
        scale_up_latency_s=1e-6,
        cxl_bandwidth_bytes_per_s=1e11,
        cxl_latency_s=1e-7,
    )
    devices = tuple(make_device(f"g{i}", cap=cap) for i in range(num_devices))
    node = Node(name="node-0", compute_devices=devices,
                node_memory=make_memory("node", bw=5e11))
    return System(name="test", network=network,
                  input_memory=make_memory("nvm", bw=5e9, cap=1e12), nodes=(node,))


def test_progress_reports_each_completion():
    model = toy_model()
    system = make_system(1)
    reqs = [Request(i, model, 32, 4) for i in range(3)]

    updates: list[RunProgress] = []
    result = Simulator(system, StrategyConfig(max_batch_size=1)).run(
        reqs, progress=updates.append
    )

    assert updates, "expected progress updates"
    # Completed count is monotonic non-decreasing and ends at the suite total.
    completed = [u.completed for u in updates]
    assert completed == sorted(completed)
    assert updates[-1].completed == 3
    assert all(u.total == 3 for u in updates)
    assert updates[-1].completed == len(result.records)


def test_progress_times_are_monotonic_and_consistent():
    model = toy_model()
    system = make_system(2)
    reqs = [Request(i, model, 48, 4, arrival_time=i * 0.1) for i in range(4)]

    updates: list[RunProgress] = []
    Simulator(system, StrategyConfig(max_batch_size=1)).run(reqs, progress=updates.append)

    sim_times = [u.sim_time for u in updates]
    wall_times = [u.wall_time for u in updates]
    assert sim_times == sorted(sim_times)
    assert wall_times == sorted(wall_times)
    assert all(u.wall_time >= 0.0 for u in updates)


def test_progress_under_pdd():
    model = toy_model()
    system = make_system(2)
    reqs = [Request(i, model, 48, 4) for i in range(3)]

    updates: list[RunProgress] = []
    Simulator(system, StrategyConfig(allow_pdd=True, max_batch_size=1)).run(
        reqs, progress=updates.append
    )

    assert updates
    assert updates[-1].completed == 3
    # Prefill completions must not be counted as finished sequences.
    assert all(u.completed <= 3 for u in updates)


def test_no_progress_callback_is_fine():
    model = toy_model()
    system = make_system(1)
    reqs = [Request(0, model, 32, 4)]
    # Should not raise without a callback.
    result = Simulator(system, StrategyConfig()).run(reqs)
    assert len(result.records) == 1


def test_progress_reporter_prints_final_line():
    stream = io.StringIO()
    reporter = ProgressReporter(stream=stream, min_interval=0.0)

    reporter(RunProgress(completed=1, total=3, sim_time=1.5, wall_time=0.2))
    reporter(RunProgress(completed=3, total=3, sim_time=4.0, wall_time=0.5))

    out = stream.getvalue()
    assert "1/3 sequences" in out
    assert "3/3 sequences" in out
    assert "sim=" in out and "wall=" in out
    assert out.endswith("\n")  # final (100%) update terminates the line


def test_progress_reporter_throttles_intermediate_updates():
    stream = io.StringIO()
    reporter = ProgressReporter(stream=stream, min_interval=10.0)

    # Two non-final updates within the interval: the second is throttled.
    reporter(RunProgress(completed=1, total=3, sim_time=1.0, wall_time=0.1))
    reporter(RunProgress(completed=2, total=3, sim_time=2.0, wall_time=0.2))
    assert "1/3 sequences" in stream.getvalue()
    assert "2/3 sequences" not in stream.getvalue()

    # The final update is always printed despite throttling.
    reporter(RunProgress(completed=3, total=3, sim_time=3.0, wall_time=0.3))
    assert "3/3 sequences" in stream.getvalue()


def test_build_progress_reporter_prints_final_line():
    stream = io.StringIO()
    reporter = BuildProgressReporter(stream=stream, min_interval=0.0)

    reporter(BuildProgress(workloads_done=1, workloads_total=3,
                           requests_built=2, wall_time=0.2))
    reporter(BuildProgress(workloads_done=3, workloads_total=3,
                           requests_built=7, wall_time=0.5))

    out = stream.getvalue()
    assert "1/3 workloads" in out
    assert "3/3 workloads" in out
    assert "2 requests" in out and "7 requests" in out
    assert "wall=" in out
    assert out.endswith("\n")  # final (100%) update terminates the line


def test_build_progress_reporter_throttles_intermediate_updates():
    stream = io.StringIO()
    reporter = BuildProgressReporter(stream=stream, min_interval=10.0)

    # Two non-final updates within the interval: the second is throttled.
    reporter(BuildProgress(workloads_done=1, workloads_total=3,
                           requests_built=1, wall_time=0.1))
    reporter(BuildProgress(workloads_done=2, workloads_total=3,
                           requests_built=3, wall_time=0.2))
    assert "1/3 workloads" in stream.getvalue()
    assert "2/3 workloads" not in stream.getvalue()

    # The final update is always printed despite throttling.
    reporter(BuildProgress(workloads_done=3, workloads_total=3,
                           requests_built=5, wall_time=0.3))
    assert "3/3 workloads" in stream.getvalue()


def test_run_emits_arrival_issue_completion_events():
    model = toy_model()
    system = make_system(1)
    reqs = [Request(i, model, 32, 4) for i in range(3)]

    seen: list[RunEvent] = []
    result = Simulator(system, StrategyConfig(max_batch_size=1)).run(
        reqs, events=seen.append
    )

    kinds = {e.kind for e in seen}
    assert kinds == {"arrival", "issue", "completion"}
    # Every sequence arrives, is issued and completes.
    arrivals = [e for e in seen if e.kind == "arrival"]
    issues = [e for e in seen if e.kind == "issue"]
    completions = [e for e in seen if e.kind == "completion"]
    assert len(arrivals) == 3
    assert len(issues) == 3  # max_batch_size=1 -> one batch per sequence
    assert len(completions) == len(result.records) == 3
    # Arrival carries token counts; completion carries timing.
    assert all(e.prompt_tokens == 32 and e.output_tokens == 4 for e in arrivals)
    assert all(e.queue_delay is not None for e in completions)
    # Issue carries a batch id, members and an engine-group label.
    assert all(e.engine_group and e.members for e in issues)


def test_run_events_completed_count_is_monotonic():
    model = toy_model()
    system = make_system(2)
    reqs = [Request(i, model, 48, 4, arrival_time=i * 0.1) for i in range(4)]

    seen: list[RunEvent] = []
    Simulator(system, StrategyConfig(max_batch_size=1)).run(reqs, events=seen.append)

    completed = [e.completed for e in seen]
    assert completed == sorted(completed)
    assert seen[-1].completed == 4
    assert all(e.total == 4 for e in seen)


def test_run_events_under_pdd():
    model = toy_model()
    system = make_system(2)
    reqs = [Request(i, model, 48, 4) for i in range(3)]

    seen: list[RunEvent] = []
    Simulator(system, StrategyConfig(allow_pdd=True, max_batch_size=1)).run(
        reqs, events=seen.append
    )

    # PDD issues each sequence twice (prefill then decode) but completes once.
    phases = {e.phase for e in seen if e.kind == "issue"}
    assert phases == {"prefill", "decode"}
    assert len([e for e in seen if e.kind == "completion"]) == 3


def test_event_log_reporter_formats_each_milestone():
    stream = io.StringIO()
    reporter = EventLogReporter(stream=stream)

    reporter(RunEvent(kind="arrival", sim_time=1.0, wall_time=0.1,
                      completed=0, total=3, sequence="w0t0",
                      prompt_tokens=32, output_tokens=4))
    reporter(RunEvent(kind="issue", sim_time=1.5, wall_time=0.2,
                      completed=0, total=3, batch_index=0,
                      members=("w0t0",), engine_group="g0"))
    reporter(RunEvent(kind="completion", sim_time=2.0, wall_time=0.3,
                      completed=1, total=3, sequence="w0t0",
                      queue_delay=0.5, ttft=0.7, tps=12.5))

    out = stream.getvalue()
    lines = out.strip().splitlines()
    assert len(lines) == 3  # one line per milestone (append-only log)
    assert "ARRIVE" in out and "w0t0" in out and "in=32" in out
    assert "ISSUE" in out and "batch#0" in out and "g0" in out
    assert "DONE" in out and "ttft=" in out and "tps=12.5" in out
    assert "sim" in out and "wall" in out


def test_event_log_reporter_handles_missing_metrics():
    stream = io.StringIO()
    reporter = EventLogReporter(stream=stream)

    reporter(RunEvent(kind="completion", sim_time=2.0, wall_time=0.3,
                      completed=1, total=1, sequence="w0t0",
                      queue_delay=0.0, ttft=None, tps=None))

    out = stream.getvalue()
    assert "ttft=-" in out and "tps=-" in out

