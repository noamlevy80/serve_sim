"""Pure view-model derivation for the visualization tool.

:func:`build_view_model` consumes a backend ``viz.json`` payload (as produced by
:func:`serve_sim.report.build_viz_payload`) and returns a declarative *view
model*: a list of summary tables and a list of timeline-graph descriptors. The
browser only renders this structure -- it holds no derivation logic -- so every
graph the tool draws is fully described, and testable, in Python.

Graph descriptors come in three kinds:

- ``value``   -- one bucketed series ``buckets = [[t0, t1, value], ...]`` plus a
  static ``max_value`` (the non-autoscaling ceiling line) and a ``unit``.
- ``stacked`` -- a value graph split into named bands (``keys`` with a
  per-bucket ``{key: value}`` map), for occupancy broken down by content.
- ``discrete``-- merged ``segments = [[t0, t1, abbrev, full, color_key], ...]``;
  gaps are left empty and equal ``color_key`` strings render the same color.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..report import DEVICE_STATES


# --- engineering-notation formatting --------------------------------------------

_MAGNITUDES = (
    (1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K"),
    (1.0, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n"),
)


def eng_format(value: Any) -> str:
    """Format a number with <=3 integer digits, <=1 decimal and a magnitude letter.

    Mirrors the examples in the PRD (``2.4P``, ``10T``, ``1.2G``, ``128M``,
    ``10.1K``, ``12m``, ``75.4u``, ``1.2n``): one decimal is kept below 100 and a
    trailing ``.0`` is dropped; at or above 100 the value shows as an integer.
    """

    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:  # NaN
        return "NaN"
    if v == 0:
        return "0"
    sign = "-" if v < 0 else ""
    a = abs(v)
    for scale, suffix in _MAGNITUDES:
        if a >= scale:
            x = a / scale
            text = f"{x:.0f}" if x >= 100 else f"{x:.1f}".removesuffix(".0")
            return f"{sign}{text}{suffix}"
    # Smaller than 1 nano: still report in nanos.
    x = a / 1e-9
    text = f"{x:.0f}" if x >= 100 else f"{x:.1f}".removesuffix(".0")
    return f"{sign}{text}n"


def _pct(fraction: Any) -> str:
    try:
        return f"{float(fraction) * 100:.1f}%"
    except (TypeError, ValueError):
        return ""


# --- summary tab ----------------------------------------------------------------

def _stat_row(label: str, dist: Mapping[str, Any]) -> list[str]:
    return [label, *(eng_format(dist.get(k)) for k in ("mean", "p50", "p90", "p99", "max"))]


def build_summary_tables(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """AI-Perf style summary tables for the run, workload, system and performance."""

    s = payload.get("summary", {})
    devices = payload.get("devices", [])
    memories = payload.get("memories", [])
    tables: list[dict[str, Any]] = []

    tables.append({
        "title": "Run",
        "columns": ["Metric", "Value"],
        "rows": [
            ["Requests", str(s.get("num_requests", 0))],
            ["Batches", str(s.get("num_batches", 0))],
            ["Events", str(s.get("num_events", 0))],
            ["Decisions", str(s.get("num_decisions", 0))],
            ["Makespan (s)", eng_format(s.get("makespan_s"))],
            ["Peak memory (B)", eng_format(s.get("peak_memory_bytes"))],
            ["Total FLOPs", eng_format(s.get("total_flops"))],
            ["Total bytes read", eng_format(s.get("total_bytes_read"))],
            ["DMA transfers", str(s.get("num_dma_transfers", 0))],
            ["DMA bytes", eng_format(s.get("dma_transfer_bytes"))],
        ],
    })

    tables.append({
        "title": "Throughput",
        "columns": ["Metric", "Value"],
        "rows": [
            ["Requests / s", eng_format(s.get("throughput_requests_per_s"))],
            ["Output tokens / s", eng_format(s.get("throughput_output_tokens_per_s"))],
            ["Avg time in queue (s)", eng_format(s.get("avg_queue_delay_s"))],
            ["Avg latency (s)", eng_format(s.get("avg_latency_s"))],
        ],
    })

    stat_cols = ["Metric", "avg", "p50", "p90", "p99", "max"]
    perf_rows = []
    for label, key in (("Latency (s)", "latency_s"), ("Queue delay (s)", "queue_delay_s"),
                       ("TTFT (s)", "ttft_s"), ("TPOT (s)", "tpot_s"),
                       ("TPS (tok/s)", "tps_tokens_per_s")):
        dist = s.get(key)
        if isinstance(dist, Mapping):
            perf_rows.append(_stat_row(label, dist))
    if perf_rows:
        tables.append({
            "title": "Performance distributions",
            "columns": stat_cols,
            "rows": perf_rows,
        })

    counts = s.get("decision_counts", {})
    if counts:
        tables.append({
            "title": "Orchestration decisions",
            "columns": ["Decision", "Count"],
            "rows": [[k, str(v)] for k, v in counts.items()],
        })

    sequences = payload.get("sequences", [])
    if sequences:
        tables.append({
            "title": "Sequences",
            "columns": ["Sequence", "Model", "Engine group(s)",
                        "Time in queue (s)", "TTFT (prefill, s)",
                        "TPS (decode, tok/s)", "Total idle wait (s)",
                        "Comm wait (s)",
                        "Total latency (s)", "Effective TPS (tok/s)"],
            "rows": [[q.get("sequence", ""),
                      q.get("model", ""),
                      ", ".join(q.get("engine_groups", [])),
                      eng_format(q.get("queue_s")),
                      eng_format(q.get("ttft_prefill_s")),
                      eng_format(q.get("tps_tokens_per_s")),
                      eng_format(q.get("idle_wait_s")),
                      eng_format(q.get("comm_wait_s")),
                      eng_format(q.get("latency_s")),
                      eng_format(q.get("effective_tps_tokens_per_s"))]
                     for q in sequences],
        })

    if devices:
        tables.append({
            "title": "Devices",
            "columns": ["Device", "Peak FLOP/s", "Busy", "Compute", "Bandwidth",
                        "Peak memory"],
            "rows": [[d.get("device", ""), eng_format(d.get("peak_flops_fp16")),
                      _pct(d.get("busy_fraction")), _pct(d.get("compute_util")),
                      _pct(d.get("bandwidth_util")),
                      eng_format(d.get("peak_memory_bytes"))]
                     for d in devices],
        })

    if memories:
        tables.append({
            "title": "Memories",
            "columns": ["Memory", "Role", "Capacity", "Bandwidth util",
                        "Occupancy", "Bytes moved"],
            "rows": [[m.get("memory", ""), m.get("role", ""),
                      eng_format(m.get("capacity_bytes")),
                      _pct(m.get("bandwidth_util")),
                      _pct(m.get("occupancy_fraction")),
                      eng_format(m.get("bytes_moved"))]
                     for m in memories],
        })

    return tables


# --- timeline tab ---------------------------------------------------------------

_STATE_ABBREV = {
    "compute_bound": ("CMP", "Compute bound"),
    "bandwidth_bound": ("BW", "Bandwidth bound"),
    "communicating": ("COM", "Communicating (collectives)"),
    "waiting_kv": ("KV", "Waiting on KV"),
    "waiting_weights": ("WTS", "Waiting on weights"),
    "waiting_experts": ("EXP", "Waiting on experts"),
    "kernel_launch": ("KL", "Kernel launch"),
    "idle": ("IDLE", "Idle"),
}

_WORKLOAD_STATE_ABBREV = {
    "not_arrived": ("--", "Not arrived"),
    "in_queue": ("Q", "In queue"),
    "kv_fetch": ("KVF", "KV fetch"),
    "prefill": ("PRE", "Prefill"),
    "decode": ("DEC", "Decode"),
    "done": ("DONE", "Done"),
}


def _value_graph(gid: str, title: str, section: str, group: str, unit: str,
                 max_value: Any, rows: Sequence[Mapping[str, Any]],
                 field: str) -> dict[str, Any]:
    return {
        "id": gid,
        "title": title,
        "section": section,
        "group": group,
        "kind": "value",
        "unit": unit,
        "max_value": float(max_value) if max_value not in (None, "") else None,
        "buckets": [[r["time_start"], r["time_end"], float(r.get(field) or 0.0)]
                    for r in rows],
    }


def _merge_segments(rows: Sequence[Mapping[str, Any]],
                    classify) -> list[list[Any]]:
    """Merge consecutive buckets with the same label into ``[t0, t1, ...]`` bars.

    ``classify(row)`` returns ``(abbrev, full, color_key)`` or ``None`` to leave
    the bucket empty (a gap). Adjacent buckets sharing a ``color_key`` coalesce.
    """

    segments: list[list[Any]] = []
    for r in rows:
        info = classify(r)
        if info is None:
            continue
        abbrev, full, key = info
        if segments and segments[-1][4] == key and segments[-1][1] == r["time_start"]:
            segments[-1][1] = r["time_end"]
        else:
            segments.append([r["time_start"], r["time_end"], abbrev, full, key])
    return segments


def _discrete_graph(gid: str, title: str, section: str, group: str,
                    segments: list[list[Any]]) -> dict[str, Any]:
    return {
        "id": gid,
        "title": title,
        "section": section,
        "group": group,
        "kind": "discrete",
        "segments": segments,
    }


def _dominant_state(row: Mapping[str, Any]) -> str:
    best_state = "idle"
    best = -1.0
    for state in DEVICE_STATES:
        frac = float(row.get(f"{state}_fraction") or 0.0)
        if frac > best:
            best = frac
            best_state = state
    return best_state


def _object_label(label: str) -> tuple[str, str, str]:
    """(abbrev, full, color_key) for a transfer-object label."""

    if label.startswith("weights:"):
        model = label[len("weights:"):]
        return "W", f"Weights: {model}", f"weights:{model}"
    if label.startswith("experts:"):
        model = label[len("experts:"):]
        return "E", f"Experts: {model}", f"experts:{model}"
    if label.startswith("kv:"):
        seq = label[len("kv:"):]
        return seq, f"KV: {seq}", f"kv:{seq}"
    return label, label, label


def _device_graphs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    graphs: list[dict[str, Any]] = []
    timeline = payload.get("device_timeline", [])
    by_device: dict[str, list[Mapping[str, Any]]] = {}
    for r in timeline:
        by_device.setdefault(r["device"], []).append(r)
    for rows in by_device.values():
        rows.sort(key=lambda r: r["bucket"])

    for spec in payload.get("devices", []):
        name = spec.get("device", "")
        rows = by_device.get(name, [])
        if not rows:
            continue
        graphs.append(_value_graph(
            f"dev:{name}:compute", f"Compute -- {name}", "compute_device", name,
            "FLOP/s", spec.get("peak_flops_fp16"), rows, "compute_flops_per_s"))
        graphs.append(_value_graph(
            f"dev:{name}:bandwidth", f"1st-tier bandwidth -- {name}",
            "compute_device", name, "B/s",
            spec.get("first_tier_bandwidth_bytes_per_s"), rows,
            "first_tier_bytes_per_s"))
        graphs.append(_stacked_graph(
            f"dev:{name}:capacity", f"1st-tier capacity -- {name}",
            "compute_device", name, "B",
            spec.get("first_tier_capacity_bytes"), rows))
        graphs.append(_discrete_graph(
            f"dev:{name}:reason", f"Reason idle -- {name}", "compute_device", name,
            _merge_segments(rows, lambda r: (
                *_STATE_ABBREV.get(_dominant_state(r), ("?", "?")),
                f"state:{_dominant_state(r)}"))))
        graphs.append(_discrete_graph(
            f"dev:{name}:xfer_src", f"Transfer source -- {name}", "compute_device",
            name, _merge_segments(rows, lambda r: (
                (r["transfer_source"], f"Source: {r['transfer_source']}",
                 f"dev:{r['transfer_source']}") if r.get("transfer_source") else None))))
        graphs.append(_discrete_graph(
            f"dev:{name}:xfer_obj", f"Transfer object -- {name}", "compute_device",
            name, _merge_segments(rows, lambda r: (
                _object_label(r["transfer_object"]) if r.get("transfer_object")
                else None))))
        graphs.append(_value_graph(
            f"dev:{name}:batch", f"Task batch size -- {name}", "compute_device",
            name, "seq", None, rows, "batch_size"))
        graphs.append(_value_graph(
            f"dev:{name}:out_tps", f"Output token throughput -- {name}",
            "compute_device", name, "tok/s", None, rows, "decode_tokens_per_s"))
        graphs.append(_value_graph(
            f"dev:{name}:in_tps", f"Input token throughput -- {name}",
            "compute_device", name, "tok/s", None, rows, "prefill_tokens_per_s"))
        graphs.append(_value_graph(
            f"dev:{name}:resident", f"Resident tasks -- {name}", "compute_device",
            name, "tasks", None, rows, "resident_tasks"))
    return graphs


def _stacked_graph(gid: str, title: str, section: str, group: str, unit: str,
                   max_value: Any,
                   rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys: list[str] = []
    for r in rows:
        for k in r.get("content", {}):
            if k not in keys:
                keys.append(k)
    return {
        "id": gid,
        "title": title,
        "section": section,
        "group": group,
        "kind": "stacked",
        "unit": unit,
        "max_value": float(max_value) if max_value not in (None, "") else None,
        "keys": keys,
        "buckets": [[r["time_start"], r["time_end"], dict(r.get("content", {}))]
                    for r in rows],
    }


def _memory_graphs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    graphs: list[dict[str, Any]] = []
    timeline = payload.get("memory_timeline", [])
    by_memory: dict[str, list[Mapping[str, Any]]] = {}
    for r in timeline:
        by_memory.setdefault(r["memory"], []).append(r)
    for rows in by_memory.values():
        rows.sort(key=lambda r: r["bucket"])

    for spec in payload.get("memories", []):
        name = spec.get("memory", "")
        rows = by_memory.get(name, [])
        if not rows:
            continue
        graphs.append(_value_graph(
            f"mem:{name}:bandwidth", f"Bandwidth -- {name}", "memory_device", name,
            "B/s", spec.get("bandwidth_bytes_per_s"), rows, "bandwidth_bytes_per_s"))
        graphs.append(_stacked_graph(
            f"mem:{name}:capacity", f"Capacity by content -- {name}",
            "memory_device", name, "B",
            spec.get("capacity_bytes"), rows))
        graphs.append(_discrete_graph(
            f"mem:{name}:xfer_src", f"Transfer source -- {name}", "memory_device",
            name, _merge_segments(rows, lambda r: (
                (r["transfer_source"], f"Source: {r['transfer_source']}",
                 f"dev:{r['transfer_source']}") if r.get("transfer_source") else None))))
        graphs.append(_discrete_graph(
            f"mem:{name}:xfer_obj", f"Transfer object -- {name}", "memory_device",
            name, _merge_segments(rows, lambda r: (
                _object_label(r["transfer_object"]) if r.get("transfer_object")
                else None))))
    return graphs


def _workload_graphs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    graphs: list[dict[str, Any]] = []
    timeline = payload.get("workload_timeline", [])
    by_workload: dict[str, list[Mapping[str, Any]]] = {}
    order: list[str] = []
    for r in timeline:
        w = r["workload"]
        if w not in by_workload:
            by_workload[w] = []
            order.append(w)
        by_workload[w].append(r)
    for rows in by_workload.values():
        rows.sort(key=lambda r: r["bucket"])

    for w in order:
        rows = by_workload[w]
        label = rows[0].get("sequence") or w
        graphs.append(_discrete_graph(
            f"wl:{w}:device", f"Engine group -- {w}", "workload", w,
            _merge_segments(rows, lambda r: (
                (r["group"],
                 f"{r['group']} ({len(r['devices'])} devices):\n"
                 + "\n".join(r["devices"]) if r.get("devices")
                 else f"Group {r['group']}",
                 f"group:{r['group']}")
                if r.get("group") else None)))) 
        graphs.append(_discrete_graph(
            f"wl:{w}:turn", f"Current turn -- {w}", "workload", w,
            _merge_segments(rows, lambda r: (
                (f"T{r['turn']}", f"Turn {r['turn']}", f"turn:{r['turn']}")))))
        graphs.append(_discrete_graph(
            f"wl:{w}:state", f"State -- {w}", "workload", w,
            _merge_segments(rows, lambda r: (
                *_WORKLOAD_STATE_ABBREV.get(r["state"], (r["state"], r["state"])),
                f"wstate:{r['state']}"))))
        graphs.append(_discrete_graph(
            f"wl:{w}:batch", f"In batch -- {w}", "workload", w,
            _merge_segments(rows, lambda r: (
                (f"B{r['batch']}", f"Batch {r['batch']}", f"batch:{r['batch']}")
                if r.get("batch") is not None else None))))
    return graphs


def build_view_model(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Turn a backend ``viz.json`` payload into the renderer's view model."""

    graphs = _device_graphs(payload) + _memory_graphs(payload) + _workload_graphs(payload)
    return {
        "run_id": payload.get("run_id", "run"),
        "makespan_s": float(payload.get("makespan_s") or 0.0),
        "num_buckets": int(payload.get("num_buckets") or 0),
        "summary_tables": build_summary_tables(payload),
        "graphs": graphs,
        "graph_tree": _build_graph_tree(graphs),
        "workload_graph": payload.get("workload_graph",
                                      {"nodes": [], "edges": [], "num_lanes": 0,
                                       "makespan_s": float(payload.get("makespan_s") or 0.0)}),
    }


_SECTION_LABELS = {
    "compute_device": "Compute Devices",
    "memory_device": "Memory Devices",
    "workload": "Workloads",
}


def _build_graph_tree(graphs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group the graphs into the selection panel's category/type/device hierarchy.

    Three levels, preserving first-seen order: section (Compute Devices / Memory
    Devices / Workloads) -> graph type (Compute, Bandwidth, State, ...) -> the
    individual device or workload it belongs to. Unlike the display -- which keeps
    every graph for one device together -- the panel groups by graph type first so
    a whole type can be hidden at once. Each leaf carries the graph ``id`` (for
    show/hide) and the device/workload label.
    """

    sections: list[str] = []
    types_by_section: dict[str, list[str]] = {}
    leaves: dict[tuple[str, str], list[dict[str, str]]] = {}
    for g in graphs:
        section = g["section"]
        graph_type = g["title"].split(" -- ", 1)[0]
        if section not in types_by_section:
            types_by_section[section] = []
            sections.append(section)
        if graph_type not in types_by_section[section]:
            types_by_section[section].append(graph_type)
        leaves.setdefault((section, graph_type), []).append(
            {"id": g["id"], "label": g["group"]})

    tree: list[dict[str, Any]] = []
    for section in sections:
        tree.append({
            "label": _SECTION_LABELS.get(section, section),
            "children": [
                {"label": graph_type, "graphs": leaves[(section, graph_type)]}
                for graph_type in types_by_section[section]
            ],
        })
    return tree

