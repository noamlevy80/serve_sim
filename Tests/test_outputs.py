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
from serve_sim.orchestrator import (
    EventRecord, Request, RequestRecord, RunResult, Simulator, StrategyConfig,
)
from serve_sim.report import (
    _decision_rows,
    build_viz_payload,
    device_summaries,
    device_timeline,
    memory_summaries,
    memory_timeline,
    summarize,
    workload_graph,
    workload_timeline,
    write_outputs,
)
from serve_sim.report import (
    DEVICE_STATES, WORKLOAD_STATES, _event_state, _state_seconds,
)
from serve_sim.system import Network, Node, System
from serve_sim.tokenizer import WhitespaceTokenizer
from serve_sim.workload import build_workload_from_rows

from conftest import make_row


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


def test_device_timeline_token_throughput_is_shared_across_the_group():
    model = toy_model()
    system = make_system(2)
    reqs = [Request(0, model, 16, 4, 0.0), Request(1, model, 16, 4, 0.0)]

    result = Simulator(system, StrategyConfig(max_batch_size=2, tensor_parallel=2)).run(reqs)

    timeline = device_timeline(result, num_buckets=16)
    by_bucket: dict[int, list[dict]] = {}
    for row in timeline:
        by_bucket.setdefault(row["bucket"], []).append(row)

    decoded = prefilled = False
    for rows in by_bucket.values():
        # Token throughput is a property of the task, so every rank in the
        # engine group reports the same value in a given bucket.
        outs = {round(r["decode_tokens_per_s"], 6) for r in rows}
        ins = {round(r["prefill_tokens_per_s"], 6) for r in rows}
        assert len(outs) == 1
        assert len(ins) == 1
        decoded = decoded or next(iter(outs)) > 0
        prefilled = prefilled or next(iter(ins)) > 0
    assert decoded and prefilled

    # The only batch holds two sequences, so each bucket is idle or shows it.
    sizes = {row["batch_size"] for row in timeline}
    assert sizes <= {0, 2}
    assert 2 in sizes


# --- per-device execution-state breakdown --------------------------------------


def _state_keys():
    return [f"{state}_fraction" for state in DEVICE_STATES]


def _event(phase, start, end, *, compute_time=0.0, bandwidth_time=0.0, device="g0"):
    """A minimal rescaled EventRecord for classifier unit tests."""
    return EventRecord(
        job_index=0, batch_index=0, job_phase="full", request_ids=(0,),
        group_index=0, phase=phase, device=device, memory="", model="m",
        flops=0.0, bytes_read=0.0,
        compute_time=compute_time, bandwidth_time=bandwidth_time,
        duration=end - start, start=start, end=end, rescaled=True,
    )


def test_event_state_classifies_each_phase():
    # Forward passes split on compute_time vs bandwidth_time.
    assert _event_state(
        _event("prefill", 0, 1, compute_time=0.8, bandwidth_time=0.2)
    ) == "compute_bound"
    assert _event_state(
        _event("decode", 0, 1, compute_time=0.2, bandwidth_time=0.8)
    ) == "bandwidth_bound"
    # A tie counts as compute-bound (>= rule).
    assert _event_state(
        _event("prefill", 0, 1, compute_time=0.5, bandwidth_time=0.5)
    ) == "compute_bound"
    # Data-movement phases map to their wait states.
    assert _event_state(_event("transfer", 0, 1)) == "waiting_kv"
    assert _event_state(_event("weight_transfer", 0, 1)) == "waiting_weights"
    assert _event_state(_event("expert_transfer", 0, 1)) == "waiting_experts"
    assert _event_state(_event("kernel_launch", 0, 1)) == "kernel_launch"


def test_state_seconds_partitions_window_with_idle_gap():
    # A compute event [0,1], a KV fetch [1,2], then an idle gap [2,4].
    events = [
        _event("prefill", 0.0, 1.0, compute_time=1.0, bandwidth_time=0.2),
        _event("transfer", 1.0, 2.0),
    ]
    seconds = _state_seconds(events, 0.0, 4.0)

    assert seconds["compute_bound"] == pytest.approx(1.0)
    assert seconds["waiting_kv"] == pytest.approx(1.0)
    assert seconds["idle"] == pytest.approx(2.0)
    # The partition is exhaustive: every second is attributed exactly once.
    assert sum(seconds.values()) == pytest.approx(4.0)


def test_state_seconds_charges_higher_priority_on_overlap():
    # Compute and a prefetch overlap on [0,1]; compute outranks waiting.
    events = [
        _event("decode", 0.0, 1.0, compute_time=0.9, bandwidth_time=0.1),
        _event("transfer", 0.0, 1.0),
    ]
    seconds = _state_seconds(events, 0.0, 1.0)

    assert seconds["compute_bound"] == pytest.approx(1.0)
    assert seconds["waiting_kv"] == pytest.approx(0.0)
    assert sum(seconds.values()) == pytest.approx(1.0)


def test_device_state_breakdown_partitions_the_run():
    model = toy_model()
    system = make_system(1)
    reqs = [Request(i, model, 32, 4, arrival_time=float(i)) for i in range(2)]

    result = Simulator(
        system, StrategyConfig(max_batch_size=1, model_weight_loading=True)
    ).run(reqs)
    devices = device_summaries(result)

    assert devices
    for d in devices:
        fractions = [d[k] for k in _state_keys()]
        for f in fractions:
            assert 0.0 <= f <= 1.0 + 1e-9
        # The states partition the run, so the fractions sum to one.
        assert sum(fractions) == pytest.approx(1.0, abs=1e-6)
        # Some compute happened, and weights were streamed in.
        assert d["compute_bound_fraction"] + d["bandwidth_bound_fraction"] > 0.0
        assert d["waiting_weights_fraction"] > 0.0


def test_device_timeline_state_breakdown_partitions_each_bucket():
    model = toy_model()
    system = make_system(2)
    reqs = [Request(i, model, 32, 4) for i in range(2)]

    result = Simulator(
        system, StrategyConfig(allow_pdd=True, max_batch_size=1)
    ).run(reqs)

    timeline = device_timeline(result, num_buckets=6)
    keys = _state_keys()
    for row in timeline:
        fractions = [row[k] for k in keys]
        for f in fractions:
            assert 0.0 <= f <= 1.0 + 1e-9
        assert sum(fractions) == pytest.approx(1.0, abs=1e-6)


def test_compute_and_bandwidth_states_account_for_all_forward_work():
    model = toy_model()
    system = make_system(1)
    req = Request(0, model, prompt_tokens=64, output_tokens=8)

    result = Simulator(system, StrategyConfig(max_batch_size=1)).run([req])
    g0 = next(d for d in device_summaries(result) if d["device"] == "g0")

    # With no transfers or launch overhead, every busy second is a forward pass,
    # split between the two compute states; together they cover the busy time.
    compute_states = g0["compute_bound_fraction"] + g0["bandwidth_bound_fraction"]
    assert compute_states > 0.0
    assert compute_states == pytest.approx(g0["busy_fraction"], abs=1e-9)
    assert compute_states == pytest.approx(1.0 - g0["idle_fraction"], abs=1e-9)


def test_waiting_experts_state_appears_for_moe_streaming():
    from serve_sim.model import toy_moe_model

    model = toy_moe_model()
    system = make_system(1)
    reqs = [Request(i, model, 32, 4) for i in range(2)]

    result = Simulator(
        system, StrategyConfig(max_batch_size=1, model_weight_loading=True)
    ).run(reqs)
    devices = device_summaries(result)

    # Routed experts stream in on demand, distinct from the base weight load.
    assert any(d["waiting_experts_fraction"] > 0.0 for d in devices)


def test_idle_state_appears_with_a_temporal_gap():
    model = toy_model()
    system = make_system(1)
    # Two requests on one device, the second arriving long after the first
    # retires, so the device sits idle in between.
    reqs = [
        Request(0, model, prompt_tokens=32, output_tokens=4, arrival_time=0.0),
        Request(1, model, prompt_tokens=32, output_tokens=4, arrival_time=100.0),
    ]

    result = Simulator(system, StrategyConfig(max_batch_size=1)).run(reqs)
    g0 = next(d for d in device_summaries(result) if d["device"] == "g0")

    assert g0["idle_fraction"] > 0.0


def test_written_csvs_carry_state_columns(tmp_path):
    model = toy_model()
    system = make_system(1)
    req = Request(0, model, 32, 4)

    result = Simulator(
        system, StrategyConfig(max_batch_size=1, model_weight_loading=True)
    ).run([req])
    write_outputs(result, tmp_path)

    state_cols = set(_state_keys())
    with open(tmp_path / "device_summary.csv", newline="") as f:
        summary_header = set(next(csv.reader(f)))
    with open(tmp_path / "device_timeline.csv", newline="") as f:
        timeline_header = set(next(csv.reader(f)))

    assert state_cols <= summary_header
    assert state_cols <= timeline_header


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
        "memory_timeline.csv", "workload_timeline.csv", "viz.json",
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


# --- visualization timelines ----------------------------------------------------

def test_device_summary_carries_static_specs():
    model = toy_model()
    system = make_system(2)
    reqs = [Request(i, model, 32, 4) for i in range(2)]
    result = Simulator(system, StrategyConfig(max_batch_size=2)).run(reqs)

    devices = device_summaries(result)
    by_name = {d["device"]: d for d in devices}
    for d in devices:
        assert {"node", "peak_flops_fp16", "first_tier_memory",
                "first_tier_capacity_bytes",
                "first_tier_bandwidth_bytes_per_s"} <= set(d)
    g0 = by_name["g0"]
    assert g0["peak_flops_fp16"] == 100e12
    assert g0["first_tier_memory"] == "g0-mem"
    assert g0["first_tier_capacity_bytes"] == 80e9
    assert g0["node"] == "node-0"


def test_device_timeline_carries_compute_and_transfer_columns():
    model = toy_model()
    system = make_system(2)
    reqs = [Request(i, model, 32, 4) for i in range(2)]
    result = Simulator(
        system, StrategyConfig(allow_pdd=True, max_batch_size=2)).run(reqs)

    rows = device_timeline(result, 8)
    assert rows
    assert {"compute_flops_per_s", "compute_seconds", "first_tier_bytes_per_s",
            "bandwidth_seconds", "transfer_source",
            "transfer_object"} <= set(rows[0])
    # Real compute work lands in some bucket, and rates are never negative.
    assert any(r["compute_seconds"] > 0 for r in rows)
    assert all(r["compute_flops_per_s"] >= 0 for r in rows)
    assert all(r["first_tier_bytes_per_s"] >= 0 for r in rows)


def test_memory_timeline_content_decomposes_occupancy():
    model = toy_model()
    system = make_system(1)
    reqs = [Request(i, model, 32, 4) for i in range(2)]
    result = Simulator(system, StrategyConfig(max_batch_size=2)).run(reqs)

    rows = memory_timeline(result, 8)
    assert rows
    assert {"bandwidth_bytes_per_s", "occupancy_bytes", "content",
            "transfer_source", "transfer_object", "eviction_object"} <= set(rows[0])
    # The content breakdown always sums to the reported occupancy.
    for r in rows:
        assert r["occupancy_bytes"] == pytest.approx(sum(r["content"].values()))
    # While weights are resident, the first-tier content carries a "weights" band
    # (KV is broken out per co-resident dispatch batch as "KV B<n>" bands).
    first_tier = [r for r in rows
                  if r["role"] == "first_tier" and r["occupancy_bytes"] > 0]
    assert first_tier
    assert any("weights" in r["content"] for r in first_tier)
    assert all(
        k == "weights" or k == "KV" or k.startswith("KV B")
        for r in rows for k in r["content"]
    )


def test_memory_timeline_tracks_offloaded_kv_residency():
    model = toy_model()
    system = make_system(1)
    sys_msg = {"role": "system", "content": "you are a helpful coding agent always"}
    user_msg = {"role": "user", "content": "please refactor this module for me today"}

    def msg_request(rid, messages, *, workload_id, arrival_time=0.0):
        workload = build_workload_from_rows(
            [make_row(f"s{workload_id}", "m", messages, output_length=4)]
        )
        return Request.from_workload(
            rid, workload, model, WhitespaceTokenizer(),
            arrival_time=arrival_time, turn_index=0, workload_id=workload_id,
        )

    # Conversation 0 completes early; its KV is offloaded to floating memory and
    # stays resident while conversation 1 (arriving much later) runs.
    first = msg_request(0, [dict(sys_msg), dict(user_msg)], workload_id=0)
    second = msg_request(
        1,
        [dict(sys_msg), dict(user_msg),
         {"role": "assistant", "content": "second conversation tail differs here"}],
        workload_id=1, arrival_time=1000.0,
    )
    result = Simulator(system, StrategyConfig(max_batch_size=1)).run([first, second])
    assert any(d.kind == "kv_transfer" and d.source_request_id == 0
               for d in result.decisions), "expected a KV offload to floating memory"

    rows = memory_timeline(result, 16)
    # A floating memory holds the offloaded KV for at least one bucket, recorded
    # under the "KV" content band.
    floating = [r for r in rows if r["role"] in ("node", "second_tier")]
    assert any("KV" in r["content"] and r["content"]["KV"] > 0 for r in floating)


def test_workload_timeline_walks_the_turn_lifecycle():
    model = toy_model()
    system = make_system(1)
    reqs = [Request(i, model, 32, 4, workload_id=0, turn_index=i) for i in range(2)]
    result = Simulator(system, StrategyConfig(max_batch_size=4)).run(reqs)

    rows = workload_timeline(result, 32)
    assert rows
    assert {"workload", "turn", "sequence", "state", "device"} <= set(rows[0])
    states = {r["state"] for r in rows}
    assert states <= set(WORKLOAD_STATES)
    # The single workload's serialized turns both appear, reaching decode on a
    # named device.
    w0 = [r for r in rows if r["workload"] == "w0"]
    assert {r["turn"] for r in w0} >= {0, 1}
    assert any(r["state"] == "decode" for r in w0)
    assert any(r["state"] == "decode" and r["device"] for r in w0)

    # Each row also carries the full engine group: a stable id plus the device
    # list. A decode row's representative device is one of its group's devices.
    assert {"group", "devices"} <= set(rows[0])
    dec = next(r for r in w0 if r["state"] == "decode" and r["device"])
    assert dec["group"]
    assert dec["device"] in dec["devices"]
    # The group id is stable: the same device set always maps to the same id.
    groups = {tuple(r["devices"]): r["group"] for r in rows if r["devices"]}
    for r in rows:
        if r["devices"]:
            assert groups[tuple(r["devices"])] == r["group"]

    # Each row also reports the batch executing the turn: an integer id while the
    # turn is actively computed, and ``None`` otherwise (graph 3.4 "In batch").
    assert "batch" in rows[0]
    for r in rows:
        if r["state"] in {"kv_fetch", "prefill", "decode"} and r["device"]:
            assert isinstance(r["batch"], int)
        elif r["state"] in {"not_arrived", "in_queue", "done"}:
            assert r["batch"] is None


def test_workload_graph_emits_input_and_output_nodes_per_turn():
    model = toy_model()
    system = make_system(1)
    reqs = [Request(i, model, 32, 4, workload_id=i, turn_index=0) for i in range(3)]
    result = Simulator(system, StrategyConfig(max_batch_size=3)).run(reqs)

    g = workload_graph(result)
    assert g["num_lanes"] == 3
    assert {n["lane"] for n in g["nodes"]} == {0, 1, 2}
    kinds = [n["kind"] for n in g["nodes"]]
    assert kinds.count("prefill") == 3
    assert kinds.count("decode") == 3
    for n in g["nodes"]:
        assert n["t1"] >= n["t0"]
        assert {"id", "kind", "sub", "tokens", "group", "sequence"} <= set(n)
    # The input node spans queuing to the first token; the output node runs from
    # the first token to the last.
    by_id = {n["id"]: n for n in g["nodes"]}
    rec0 = next(r for r in result.records if r.request_id == 0)
    assert by_id["w0:t0:in"]["t1"] == pytest.approx(rec0.first_token_time)
    assert by_id["w0:t0:out"]["t0"] == pytest.approx(rec0.first_token_time)
    assert by_id["w0:t0:out"]["t1"] == pytest.approx(rec0.completion_time)
    assert by_id["w0:t0:in"]["tokens"] == rec0.prompt_tokens
    assert by_id["w0:t0:out"]["tokens"] == rec0.output_tokens


def test_workload_graph_adds_tool_node_between_turns():
    # The tool-call wait is the gap between a turn's completion and the next
    # turn's arrival, so build two turns of one conversation with such a gap.
    result = RunResult()
    result.records = [
        RequestRecord(0, 0.0, 0.0, 1.0, 32, 4, 0, first_token_time=0.5,
                      workload_id=0, turn_index=0),
        RequestRecord(1, 5.0, 5.0, 6.0, 32, 4, 1, first_token_time=5.5,
                      workload_id=0, turn_index=1),
    ]

    g = workload_graph(result)
    tools = [n for n in g["nodes"] if n["kind"] == "tool"]
    assert len(tools) == 1
    assert tools[0]["t0"] == pytest.approx(1.0)
    assert tools[0]["t1"] == pytest.approx(5.0)
    # A single conversation is one lane; reuse within it is implied, not drawn.
    assert g["num_lanes"] == 1
    assert g["edges"] == []


def test_workload_graph_links_cross_workload_kv_reuse():
    model = toy_model()
    system = make_system(1)
    sys_msg = {"role": "system", "content": "you are a helpful coding agent always"}
    user_msg = {"role": "user", "content": "please refactor this module for me today"}

    def msg_request(rid, messages, *, workload_id, arrival_time=0.0):
        workload = build_workload_from_rows(
            [make_row(f"s{workload_id}", "m", messages, output_length=4)]
        )
        return Request.from_workload(
            rid, workload, model, WhitespaceTokenizer(),
            arrival_time=arrival_time, turn_index=0, workload_id=workload_id,
        )

    first = msg_request(0, [dict(sys_msg), dict(user_msg)], workload_id=0)
    second = msg_request(
        1,
        [dict(sys_msg), dict(user_msg),
         {"role": "assistant", "content": "second conversation tail differs here"}],
        workload_id=1, arrival_time=1000.0,
    )
    result = Simulator(system, StrategyConfig(max_batch_size=1)).run([first, second])

    assert any(d.kind == "kv_reuse" and d.source_workload_id == 0
               and d.workload_id == 1 for d in result.decisions)

    g = workload_graph(result)
    # The reuse draws one edge from conversation 0's input node to conversation 1's.
    assert {"source": "w0:t0:in", "target": "w1:t0:in"} in g["edges"]


def test_workload_timeline_marks_late_arrivals_not_arrived():
    model = toy_model()
    system = make_system(1)
    reqs = [Request(0, model, 32, 4, arrival_time=0.0),
            Request(1, model, 32, 4, arrival_time=5.0)]
    result = Simulator(system, StrategyConfig(max_batch_size=2)).run(reqs)

    rows = workload_timeline(result, 32)
    # Before its arrival the second request reads as not-arrived in early buckets.
    assert any(r["state"] == "not_arrived" for r in rows if r["time_start"] < 5.0)


def test_build_viz_payload_bundles_every_series():
    model = toy_model()
    system = make_system(2)
    reqs = [Request(i, model, 32, 4) for i in range(2)]
    result = Simulator(
        system, StrategyConfig(allow_pdd=True, max_batch_size=2)).run(reqs)

    payload = build_viz_payload(result, run_id="viz", num_buckets=12)
    assert payload["run_id"] == "viz"
    assert payload["num_buckets"] == 12
    assert payload["makespan_s"] == pytest.approx(result.makespan)
    for key in ("summary", "devices", "memories", "device_timeline",
                "memory_timeline", "workload_timeline"):
        assert key in payload
    # The payload is JSON round-trippable (the GUI consumes it as JSON).
    assert json.loads(json.dumps(payload))["run_id"] == "viz"
    # Timelines are bucketed to the requested resolution.
    assert {r["bucket"] for r in payload["device_timeline"]} == set(range(12))


def test_write_outputs_emits_parseable_viz_files(tmp_path):
    model = toy_model()
    system = make_system(2)
    reqs = [Request(i, model, 32, 4) for i in range(3)]
    result = Simulator(
        system, StrategyConfig(allow_pdd=True, max_batch_size=2)).run(reqs)

    out = write_outputs(result, tmp_path / "viz", run_id="viz",
                        time_buckets=8, viz_buckets=16)

    with open(out / "memory_timeline.csv", newline="") as handle:
        mem_rows = list(csv.DictReader(handle))
    assert mem_rows
    assert "content_json" in mem_rows[0]
    json.loads(mem_rows[0]["content_json"])  # the content column is valid JSON

    with open(out / "workload_timeline.csv", newline="") as handle:
        work_rows = list(csv.DictReader(handle))
    assert work_rows
    assert {"workload", "turn", "state", "device"} <= set(work_rows[0])

    payload = json.loads((out / "viz.json").read_text())
    assert payload["num_buckets"] == 16
    assert payload["device_timeline"]

