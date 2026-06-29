"""Visualization tool tests: the engineering formatter, the view-model
derivation and the Flask app route.

All graph derivation lives in :mod:`serve_sim.viz.graphs` so it is unit-testable
without a browser. These tests build a real run payload via
:func:`serve_sim.report.build_viz_payload`, turn it into a view model and assert
the graph descriptors are well formed; they also smoke-test the Flask route.
"""

from __future__ import annotations

import pytest

from serve_sim.model import toy_model
from serve_sim.orchestrator import Request, Simulator, StrategyConfig
from serve_sim.report import build_viz_payload, write_outputs
from serve_sim.viz import build_view_model, eng_format
from serve_sim.viz.app import create_app
from serve_sim.viz.graphs import build_summary_tables

from test_outputs import make_system


# --- engineering-notation formatter ---------------------------------------------

@pytest.mark.parametrize("value, expected", [
    (2.4e15, "2.4P"),
    (1e13, "10T"),
    (1.2e9, "1.2G"),
    (128e6, "128M"),
    (10100, "10.1K"),
    (0.012, "12m"),
    (75.4e-6, "75.4u"),
    (1.2e-9, "1.2n"),
    (0, "0"),
    (-5e6, "-5M"),
])
def test_eng_format_matches_prd_examples(value, expected):
    assert eng_format(value) == expected


def test_eng_format_handles_non_numbers():
    assert eng_format(None) == ""
    assert eng_format(float("nan")) == "NaN"


# --- view model -----------------------------------------------------------------

def _payload(num_buckets=16):
    model = toy_model()
    system = make_system(2)
    reqs = [Request(i, model, 32, 4) for i in range(3)]
    result = Simulator(
        system, StrategyConfig(allow_pdd=True, max_batch_size=2)).run(reqs)
    return build_viz_payload(result, run_id="viz", num_buckets=num_buckets)


def test_view_model_top_level_shape():
    vm = build_view_model(_payload())
    assert vm["run_id"] == "viz"
    assert vm["makespan_s"] > 0
    assert vm["summary_tables"]
    assert vm["graphs"]
    titles = {t["title"] for t in vm["summary_tables"]}
    assert {"Run", "Throughput", "Devices", "Memories"} <= titles


def test_each_device_has_its_nine_graphs():
    vm = build_view_model(_payload())
    by_group: dict[str, list[str]] = {}
    for g in vm["graphs"]:
        if g["section"] == "compute_device":
            by_group.setdefault(g["group"], []).append(g["id"])
    assert by_group
    for ids in by_group.values():
        suffixes = {gid.rsplit(":", 1)[1] for gid in ids}
        assert {"compute", "bandwidth", "capacity", "reason", "xfer_obj",
                "batch", "out_tps", "in_tps"} <= suffixes
        assert any(gid.endswith("xfer_src") for gid in ids)


def test_value_graphs_carry_buckets_and_static_max():
    vm = build_view_model(_payload())
    compute = next(g for g in vm["graphs"] if g["id"].endswith(":compute"))
    assert compute["kind"] == "value"
    assert compute["unit"] == "FLOP/s"
    assert compute["max_value"] and compute["max_value"] > 0
    assert compute["buckets"]
    for b in compute["buckets"]:
        assert len(b) == 3 and b[0] <= b[1]


def test_reason_graph_is_top2_stacked_fractions():
    vm = build_view_model(_payload())
    reason = next(g for g in vm["graphs"] if g["id"].endswith(":reason"))
    assert reason["kind"] == "stacked"
    assert reason["unit"] == "frac"
    assert reason["max_value"] == 1.0
    # Bands are device-state names drawn in a stable, deduplicated order.
    assert reason["keys"] and len(reason["keys"]) == len(set(reason["keys"]))
    for b in reason["buckets"]:
        assert len(b) == 3 and isinstance(b[2], dict)
        # At most the top-2 states are kept per bucket, and each is a fraction.
        assert len(b[2]) <= 2
        for frac in b[2].values():
            assert 0.0 < frac <= 1.0 + 1e-9


def test_state_legend_covers_every_device_state():
    from serve_sim.report import DEVICE_STATES

    vm = build_view_model(_payload())
    legend = vm["state_legend"]
    assert [item["key"] for item in legend] == list(DEVICE_STATES)
    assert all(item["label"] for item in legend)
    # Every band a reason graph can draw has a legend entry to name its colour.
    keys = {item["key"] for item in legend}
    for g in vm["graphs"]:
        if g["id"].endswith(":reason"):
            assert set(g["keys"]) <= keys


def test_discrete_segments_are_merged_and_within_span():
    vm = build_view_model(_payload())
    makespan = vm["makespan_s"]
    disc = next(g for g in vm["graphs"]
                if g["kind"] == "discrete" and g["segments"])
    for seg in disc["segments"]:
        assert len(seg) == 5
        t0, t1 = seg[0], seg[1]
        assert 0 <= t0 < t1 <= makespan + 1e-9
    # Merging means no two adjacent segments share a colour key at a shared edge.
    for a, b in zip(disc["segments"], disc["segments"][1:]):
        assert not (a[1] == b[0] and a[4] == b[4])


def test_memory_capacity_graph_is_stacked_with_keys():
    vm = build_view_model(_payload())
    stacked = [g for g in vm["graphs"]
               if g["section"] == "memory_device" and g["kind"] == "stacked"]
    assert stacked
    g = stacked[0]
    assert g["id"].endswith(":capacity")
    assert "keys" in g and isinstance(g["keys"], list)
    for b in g["buckets"]:
        assert len(b) == 3 and isinstance(b[2], dict)


def test_each_memory_device_has_its_five_graphs():
    # Independent memory devices carry bandwidth, capacity, transfer source,
    # transfer object and eviction object graphs (PRD section 2).
    vm = build_view_model(_payload())
    by_group: dict[str, list[str]] = {}
    for g in vm["graphs"]:
        if g["section"] == "memory_device":
            by_group.setdefault(g["group"], []).append(g["id"])
    assert by_group
    for ids in by_group.values():
        suffixes = {gid.rsplit(":", 1)[1] for gid in ids}
        assert {"bandwidth", "capacity", "xfer_src", "xfer_obj",
                "evict_obj"} <= suffixes
    obj = next(g for g in vm["graphs"]
               if g["section"] == "memory_device" and g["id"].endswith("xfer_obj"))
    assert obj["kind"] == "discrete"
    for seg in obj["segments"]:
        assert len(seg) == 5
    evict = next(g for g in vm["graphs"]
                 if g["section"] == "memory_device" and g["id"].endswith("evict_obj"))
    assert evict["kind"] == "discrete"
    for seg in evict["segments"]:
        assert len(seg) == 5


def test_capacity_content_is_weights_and_per_batch_kv():
    # Both the compute-device first-tier capacity and the memory-device capacity
    # break occupancy down into a "weights" band plus one "KV B<n>" band per
    # co-resident dispatch batch -- no per-model detail.
    vm = build_view_model(_payload())
    capacity = [g for g in vm["graphs"] if g["id"].endswith(":capacity")]
    assert capacity
    # The compute-device first-tier capacity is now a stacked breakdown too.
    dev_cap = next(g for g in capacity if g["section"] == "compute_device")
    assert dev_cap["kind"] == "stacked"

    def _ok(key):
        return key == "weights" or key == "KV" or key.startswith("KV B")

    for g in capacity:
        assert all(_ok(k) for k in g["keys"])
        for _t0, _t1, content in g["buckets"]:
            assert all(_ok(k) for k in content)


def test_workload_graphs_present():
    vm = build_view_model(_payload())
    wl = [g for g in vm["graphs"] if g["section"] == "workload"]
    assert wl
    suffixes = {g["id"].rsplit(":", 1)[1] for g in wl}
    assert {"device", "turn", "state", "batch"} <= suffixes


def test_view_model_carries_workload_graph():
    vm = build_view_model(_payload())
    wg = vm["workload_graph"]
    assert {"nodes", "edges", "num_lanes", "makespan_s"} <= set(wg)
    assert wg["num_lanes"] >= 1
    assert wg["nodes"]
    kinds = {n["kind"] for n in wg["nodes"]}
    assert {"prefill", "decode"} <= kinds
    for n in wg["nodes"]:
        assert {"id", "lane", "sub", "t0", "t1", "tokens", "group"} <= set(n)
    assert isinstance(wg["edges"], list)


def test_engine_group_graph_labels_groups_and_lists_devices_on_hover():
    # The per-workload device graph shows the engine group id on the bar and the
    # full device list in the hover tooltip (segment[3]).
    vm = build_view_model(_payload())
    dev = next(g for g in vm["graphs"]
               if g["section"] == "workload" and g["id"].endswith(":device"))
    assert dev["kind"] == "discrete"
    labelled = [s for s in dev["segments"] if s[2]]
    assert labelled, "expected at least one labelled engine-group segment"
    for seg in labelled:
        abbrev, full, key = seg[2], seg[3], seg[4]
        assert abbrev.startswith("G")  # group id on the bar
        assert key == f"group:{abbrev}"  # colour keyed by the group
        assert "devices" in full or full  # device list / group name on hover


def test_graph_tree_mirrors_graph_hierarchy():
    vm = build_view_model(_payload())
    tree = vm["graph_tree"]
    labels = [cat["label"] for cat in tree]
    assert labels == ["Compute Devices", "Memory Devices", "Workloads"]

    # Every leaf id refers to a real graph, and every graph appears exactly once.
    graph_ids = {g["id"] for g in vm["graphs"]}
    leaf_ids = [leaf["id"] for cat in tree
                for grp in cat["children"] for leaf in grp["graphs"]]
    assert sorted(leaf_ids) == sorted(graph_ids)
    # The panel groups by graph type first: the middle level is the graph type
    # (title prefix) and each leaf is labelled by its device/workload.
    by_id = {g["id"]: g for g in vm["graphs"]}
    for cat in tree:
        for grp in cat["children"]:
            for leaf in grp["graphs"]:
                graph = by_id[leaf["id"]]
                assert grp["label"] == graph["title"].split(" -- ", 1)[0]
                assert leaf["label"] == graph["group"]


def test_summary_tables_format_distributions():
    payload = _payload()
    tables = build_summary_tables(payload)
    perf = next(t for t in tables if t["title"] == "Performance distributions")
    assert perf["columns"] == ["Metric", "avg", "p50", "p90", "p99", "max"]
    assert perf["rows"]


def test_throughput_table_carries_average_queue_and_latency():
    payload = _payload()
    tables = build_summary_tables(payload)
    throughput = next(t for t in tables if t["title"] == "Throughput")
    metrics = [row[0] for row in throughput["rows"]]
    assert "Avg time in queue (s)" in metrics
    assert "Avg latency (s)" in metrics


def test_sequences_table_has_one_row_per_turn():
    payload = _payload()
    # Three standalone requests => three sequence rows.
    assert len(payload["sequences"]) == 3
    tables = build_summary_tables(payload)
    seq = next(t for t in tables if t["title"] == "Sequences")
    assert seq["columns"] == [
        "Sequence", "Model", "Engine group(s)", "Time in queue (s)",
        "TTFT (prefill, s)", "TPS (decode, tok/s)", "Total idle wait (s)",
        "Comm wait (s)", "Total latency (s)", "Effective TPS (tok/s)"]
    assert len(seq["rows"]) == 3
    for q in payload["sequences"]:
        assert q["queue_s"] >= 0.0
        assert q["idle_wait_s"] >= 0.0
        assert q["comm_wait_s"] >= 0.0
        assert q["latency_s"] >= q["queue_s"]


# --- Flask app ------------------------------------------------------------------

def test_app_serves_view_model(tmp_path):
    model = toy_model()
    system = make_system(2)
    reqs = [Request(i, model, 32, 4) for i in range(2)]
    result = Simulator(system, StrategyConfig(max_batch_size=2)).run(reqs)
    out = write_outputs(result, tmp_path / "run", run_id="served", viz_buckets=8)

    app = create_app(out)
    client = app.test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert b"serve_sim" in page.data

    api = client.get("/api/view-model")
    assert api.status_code == 200
    vm = api.get_json()
    assert vm["run_id"] == "served"
    assert vm["graphs"]


def test_app_errors_without_viz_json(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_app(tmp_path)


def test_app_lists_and_selects_sibling_runs(tmp_path):
    model = toy_model()
    system = make_system(2)
    reqs = [Request(i, model, 32, 4) for i in range(2)]

    outputs = tmp_path / "Outputs"
    runs = {}
    for name in ("run-a", "run-b"):
        result = Simulator(system, StrategyConfig(max_batch_size=2)).run(reqs)
        runs[name] = write_outputs(
            result, outputs / name, run_id=name, viz_buckets=8)

    app = create_app(runs["run-a"])
    client = app.test_client()

    listing = client.get("/api/runs").get_json()
    assert set(listing["runs"]) == {"run-a", "run-b"}
    assert listing["current"] == "run-a"

    # The default run loads, and any sibling can be selected by name.
    assert client.get("/api/view-model").get_json()["run_id"] == "run-a"
    other = client.get("/api/view-model?run=run-b").get_json()
    assert other["run_id"] == "run-b"

    # Unknown / traversal names are rejected.
    assert client.get("/api/view-model?run=does-not-exist").status_code == 404
    assert client.get("/api/view-model?run=../secret").status_code == 404
