"""Outputs stage tests: event capture, first-token metrics and the run report.

A run now captures the raw event log (each event before and after rescaling),
per-job footprints and first-token times, and the :mod:`serve_sim.report` layer
turns those into an aggregate report, per-request metrics, per-device utilization
and the raw event CSVs. These tests check the captured data against the
orchestrator's own timings and that every output file is written and parseable.
"""

from __future__ import annotations

import csv
import json

import pytest

from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.model import toy_model
from serve_sim.orchestrator import Request, Simulator, StrategyConfig
from serve_sim.report import (
    _decision_rows,
    device_summaries,
    device_timeline,
    memory_summaries,
    summarize,
    write_outputs,
)
from serve_sim.system import Network, Node, System


# --- helpers --------------------------------------------------------------------


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


# --- event capture --------------------------------------------------------------


def test_run_captures_events_before_and_after_rescaling():
    model = toy_model()
    system = make_system(1)
    req = Request(0, model, prompt_tokens=32, output_tokens=4)

    result = Simulator(system, StrategyConfig(max_batch_size=1)).run([req])

    before = [e for e in result.events if not e.rescaled]
    after = [e for e in result.events if e.rescaled]
    assert before, "expected isolated events"
    assert len(before) == len(after)
    # Uncontended single job: rescaled timing matches the isolated timing.
    by_group_before = sorted((e.group_index, round(e.start, 9), round(e.end, 9))
                             for e in before)
    by_group_after = sorted((e.group_index, round(e.start, 9), round(e.end, 9))
                            for e in after)
    assert by_group_before == by_group_after


def test_first_token_and_ttft_tpot():
    model = toy_model()
    system = make_system(1)
    req = Request(0, model, prompt_tokens=32, output_tokens=4, arrival_time=1.0)

    result = Simulator(system, StrategyConfig(max_batch_size=1)).run([req])
    rec = result.record_for(0)

    assert rec.first_token_time is not None
    assert rec.arrival_time < rec.first_token_time < rec.completion_time
    assert rec.ttft == pytest.approx(rec.first_token_time - rec.arrival_time)
    assert rec.tpot == pytest.approx(
        (rec.completion_time - rec.first_token_time) / (rec.output_tokens - 1)
    )


def test_single_output_token_has_no_tpot():
    model = toy_model()
    system = make_system(1)
    req = Request(0, model, prompt_tokens=16, output_tokens=1)

    result = Simulator(system, StrategyConfig(max_batch_size=1)).run([req])
    rec = result.record_for(0)

    assert rec.first_token_time is not None
    assert rec.tpot is None


def test_first_token_under_pdd_follows_prefill_and_transfer():
    model = toy_model()
    system = make_system(2)
    req = Request(0, model, prompt_tokens=48, output_tokens=4)

    result = Simulator(system, StrategyConfig(allow_pdd=True, max_batch_size=1)).run([req])
    rec = result.record_for(0)

    assert rec.first_token_time is not None
    # decode (hence first token) only begins after prefill + KV transfer.
    decode_starts = min(e.start for e in result.events
                        if e.rescaled and e.phase == "decode")
    assert rec.first_token_time >= decode_starts
    assert rec.first_token_time < rec.completion_time


# --- aggregate report -----------------------------------------------------------


def test_summarize_basic_counts_and_throughput():
    model = toy_model()
    system = make_system(1)
    reqs = [Request(i, model, prompt_tokens=32, output_tokens=4) for i in range(3)]

    result = Simulator(system, StrategyConfig(max_batch_size=3)).run(reqs)
    report = summarize(result)

    assert report["num_requests"] == 3
    assert report["num_batches"] == result.num_batches
    assert report["total_output_tokens"] == 12
    assert report["makespan_s"] == pytest.approx(result.makespan)
    assert report["throughput_requests_per_s"] == pytest.approx(3 / result.makespan)
    assert report["total_flops"] > 0
    assert report["latency_s"]["count"] == 3


def test_device_summaries_and_timeline_shapes():
    model = toy_model()
    system = make_system(2)
    reqs = [Request(i, model, 32, 4) for i in range(2)]

    result = Simulator(system, StrategyConfig(allow_pdd=True, max_batch_size=1)).run(reqs)

    devices = device_summaries(result)
    assert devices
    for d in devices:
        assert 0.0 <= d["busy_fraction"] <= 1.0 + 1e-9
        assert d["peak_memory_bytes"] >= 0.0

    buckets = 8
    timeline = device_timeline(result, num_buckets=buckets)
    assert len(timeline) == buckets * len(devices)
    assert {row["device"] for row in timeline} == {d["device"] for d in devices}


# --- memory-device report ------------------------------------------------------


def test_memory_summaries_cover_every_memory_device():
    model = toy_model()
    system = make_system(2)
    reqs = [Request(i, model, 32, 4) for i in range(2)]

    result = Simulator(system, StrategyConfig(max_batch_size=1)).run(reqs)
    memories = memory_summaries(result)

    # Every memory in the topology appears: input NVM, node memory, two HBMs.
    names = {m["memory"] for m in memories}
    assert "nvm" in names
    assert "node" in names
    assert {"g0-mem", "g1-mem"} <= names
    roles = {m["memory"]: m["role"] for m in memories}
    assert roles["nvm"] == "input"
    assert roles["node"] == "node"
    assert roles["g0-mem"] == "first_tier"
    for m in memories:
        assert 0.0 <= m["busy_fraction"] <= 1.0 + 1e-9
        assert m["capacity_bytes"] > 0
        assert m["bytes_moved"] >= 0.0


def test_memory_summary_attributes_compute_bandwidth_to_first_tier():
    model = toy_model()
    system = make_system(1)
    req = Request(0, model, 64, 8)

    result = Simulator(system, StrategyConfig(max_batch_size=1)).run([req])
    by_name = {m["memory"]: m for m in memory_summaries(result)}

    # Compute reads land on the device's first-tier memory, which therefore moves
    # bytes and shows attached to the compute device.
    first_tier = by_name["g0-mem"]
    assert first_tier["bytes_moved"] > 0
    assert first_tier["attached_devices"] == "g0"
    # The idle node memory backs no compute and moves nothing.
    assert by_name["node"]["bytes_moved"] == 0.0


def test_memory_summary_captures_weight_load_on_input_nvm():
    model = toy_model()
    system = make_system(1)
    req = Request(0, model, 64, 8)
    strat = StrategyConfig(max_batch_size=1, model_weight_loading=True)

    result = Simulator(system, strat).run([req])
    by_name = {m["memory"]: m for m in memory_summaries(result)}

    # The weight load streams from the input NVM, so it now moves bytes.
    assert by_name["nvm"]["bytes_moved"] > 0
    assert by_name["nvm"]["num_events"] >= 1


# --- orchestration decisions ----------------------------------------------------


def test_decisions_capture_prefill_and_decode():
    model = toy_model()
    system = make_system(1)
    req = Request(0, model, prompt_tokens=32, output_tokens=4)

    result = Simulator(system, StrategyConfig(max_batch_size=1)).run([req])

    kinds = [d.kind for d in result.decisions]
    assert "prefill" in kinds
    assert "decode" in kinds
    assert "kv_transfer" not in kinds  # no PDD, no cross-device transfer

    prefill = next(d for d in result.decisions if d.kind == "prefill")
    assert prefill.request_id == 0
    assert prefill.tokens == 32
    assert prefill.devices  # mapped to at least one device
    decode = next(d for d in result.decisions if d.kind == "decode")
    assert decode.tokens == 4


def test_decisions_record_kv_reuse_with_source_sequence():
    model = toy_model()
    system = make_system(1)
    # A second-turn request whose first 16 prompt tokens are already cached.
    req = Request(0, model, prompt_tokens=48, output_tokens=4,
                  cached_tokens=16, workload_id=7, turn_index=1)

    result = Simulator(system, StrategyConfig(max_batch_size=1)).run([req])

    reuse = [d for d in result.decisions if d.kind == "kv_reuse"]
    assert reuse, "expected a KV reuse decision for a cached prefix"
    r = reuse[0]
    assert r.workload_id == 7
    assert r.turn_index == 1
    assert r.source_workload_id == 7
    assert r.source_turn_index == 0
    # Prefill only covers the uncached suffix.
    prefill = next(d for d in result.decisions if d.kind == "prefill")
    assert prefill.tokens == 32


def test_decisions_list_batch_tenants_in_sequence():
    model = toy_model()
    system = make_system(1)
    reqs = [Request(i, model, 32, 4, workload_id=i, turn_index=0) for i in range(3)]

    result = Simulator(system, StrategyConfig(max_batch_size=3)).run(reqs)

    # The three sequences share one batch, so each prefill/decode decision lists
    # every tenant in its ``batch_members``.
    expected = ((0, 0), (1, 0), (2, 0))
    for kind in ("prefill", "decode"):
        decisions = [d for d in result.decisions if d.kind == kind]
        assert decisions
        for d in decisions:
            assert d.batch_members == expected
    # kv-movement acts stay per-sequence (no batch tenant list).
    for d in result.decisions:
        if d.kind in ("kv_reuse", "kv_transfer"):
            assert d.batch_members == ()

    rows = _decision_rows(result.decisions)
    prefill_rows = [r for r in rows if r["kind"] == "prefill"]
    assert prefill_rows
    for r in prefill_rows:
        assert r["sequence"] == "w0t0 w1t0 w2t0"


def test_decisions_carry_execution_window():
    model = toy_model()
    system = make_system(1)
    reqs = [Request(i, model, 32, 4, workload_id=i, turn_index=0) for i in range(3)]

    result = Simulator(system, StrategyConfig(max_batch_size=3)).run(reqs)

    # Every decision gets a (started, completed) window with started <= completed.
    for d in result.decisions:
        assert d.time_started is not None
        assert d.time_completed is not None
        assert d.time_started <= d.time_completed

    # Prefill/decode windows match the batch's rescaled compute events.
    rescaled = [e for e in result.events if e.rescaled]
    for kind, phase in (("prefill", "prefill"), ("decode", "decode")):
        d = next(x for x in result.decisions if x.kind == kind)
        evs = [e for e in rescaled
               if e.batch_index == d.batch_index and e.phase == phase]
        assert d.time_started == min(e.start for e in evs)
        assert d.time_completed == max(e.end for e in evs)

    # The CSV exposes the two new timeline columns.
    rows = _decision_rows(result.decisions)
    assert {"time_started", "time_completed"} <= set(rows[0].keys())


def test_workload_turns_are_serialized():
    model = toy_model()
    system = make_system(1)
    # Three turns of the SAME conversation, all nominally arriving at t=0.
    reqs = [Request(i, model, 32, 4, workload_id=0, turn_index=i) for i in range(3)]

    result = Simulator(system, StrategyConfig(max_batch_size=8)).run(reqs)

    # A later turn cannot start before its predecessor completes, so no two
    # turns of the workload may share a batch.
    by_batch: dict[int, set[int]] = {}
    for rec in result.records:
        by_batch.setdefault(rec.batch_index, set()).add(rec.request_id)
    for members in by_batch.values():
        assert len(members) == 1, "same-workload turns must not batch together"

    # Each turn dispatches only after the previous one has completed.
    by_id = {rec.request_id: rec for rec in result.records}
    for i in range(1, 3):
        assert by_id[i].dispatch_time >= by_id[i - 1].completion_time - 1e-9


def test_distinct_workloads_still_batch_together():
    model = toy_model()
    system = make_system(1)
    # First turns of three different conversations can batch together.
    reqs = [Request(i, model, 32, 4, workload_id=i, turn_index=0) for i in range(3)]

    result = Simulator(system, StrategyConfig(max_batch_size=3)).run(reqs)

    batch_indices = {rec.batch_index for rec in result.records}
    assert len(batch_indices) == 1


def test_decisions_record_kv_transfer_under_pdd():
    model = toy_model()
    system = make_system(2)
    req = Request(0, model, prompt_tokens=48, output_tokens=4)

    result = Simulator(system, StrategyConfig(allow_pdd=True, max_batch_size=1)).run([req])

    transfers = [d for d in result.decisions if d.kind == "kv_transfer"]
    assert transfers, "expected a KV transfer between prefill and decode devices"
    t = transfers[0]
    assert t.request_id == 0
    assert t.source_devices  # prefill devices recorded as the source
    assert t.devices  # decode devices recorded as the destination


def test_decision_summary_in_report_and_csv(tmp_path):
    model = toy_model()
    system = make_system(2)
    reqs = [Request(i, model, 32, 4) for i in range(2)]
    result = Simulator(system, StrategyConfig(allow_pdd=True, max_batch_size=1)).run(reqs)

    report = summarize(result)
    assert report["num_decisions"] == len(result.decisions)
    assert sum(report["decision_counts"].values()) == len(result.decisions)
    assert report["decision_counts"]["prefill"] >= 2
    assert report["decision_counts"]["decode"] >= 2

    out = write_outputs(result, tmp_path / "dec", run_id="dec", time_buckets=4)
    with open(out / "orchestration_decisions.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result.decisions)
    assert {"time", "kind", "request_id", "sequence", "devices",
            "source_devices"} <= set(rows[0].keys())
    # Rows are ordered by time.
    times = [float(r["time"]) for r in rows]
    assert times == sorted(times)


# --- output files ---------------------------------------------------------------

def test_write_outputs_creates_all_files(tmp_path):
    model = toy_model()
    system = make_system(2)
    reqs = [Request(i, model, 32, 4) for i in range(3)]
    result = Simulator(system, StrategyConfig(allow_pdd=True, max_batch_size=2)).run(reqs)

    out = write_outputs(result, tmp_path / "run-1", run_id="run-1",
                        config={"hello": "world"}, time_buckets=8)

    expected = [
        "run_report.json", "run_report.txt", "requests.csv",
        "orchestration_decisions.csv",
        "events_before_rescaling.csv", "events_after_rescaling.csv",
        "device_summary.csv", "memory_summary.csv", "device_timeline.csv",
        "config.json",
    ]
    for name in expected:
        assert (out / name).exists(), name

    report = json.loads((out / "run_report.json").read_text())
    assert report["run_id"] == "run-1"
    assert report["report"]["num_requests"] == 3

    with open(out / "requests.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert {r["request_id"] for r in rows} == {"0", "1", "2"}

    with open(out / "events_after_rescaling.csv", newline="") as handle:
        after = list(csv.DictReader(handle))
    assert after
    assert all(row["device"] is not None for row in after)
    assert all("memory" in row for row in after)

    with open(out / "memory_summary.csv", newline="") as handle:
        mem_rows = list(csv.DictReader(handle))
    assert mem_rows
    assert {"nvm", "node"} <= {row["memory"] for row in mem_rows}

    echoed = json.loads((out / "config.json").read_text())
    assert echoed == {"hello": "world"}


def test_empty_run_writes_files(tmp_path):
    system = make_system(1)
    result = Simulator(system, StrategyConfig()).run([])

    out = write_outputs(result, tmp_path / "empty", run_id="empty")
    report = json.loads((out / "run_report.json").read_text())
    assert report["report"]["num_requests"] == 0
    assert report["report"]["makespan_s"] == 0.0
    assert (out / "requests.csv").exists()
