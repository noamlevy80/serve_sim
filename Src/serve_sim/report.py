"""Run outputs: aggregate report, per-request metrics and raw event logs.

This turns a :class:`~serve_sim.orchestrator.RunResult` into the raw outputs the
PRD calls for and writes them under one run directory:

- ``run_report.json`` / ``run_report.txt`` -- aggregate report over the suite:
  request/batch counts, total FLOPs and DMA transfers, peak memory, throughput,
  makespan, and latency / TTFT / TPOT distributions.
- ``requests.csv`` -- per-request arrival, dispatch, first-token and completion
  times plus latency, TTFT and TPOT.
- ``orchestration_decisions.csv`` -- the ordered log of orchestration decisions
  (model-weight load/eviction, prefill, decode, KV reuse, KV transfer and KV
  eviction) with the decision time, the execution window (``time_started`` /
  ``time_completed`` from the rescaled events that realise it), the sequence,
  serving devices, token counts and source sequence/devices/memory for weight
  and KV movements.
- ``events_before_rescaling.csv`` / ``events_after_rescaling.csv`` -- the raw
  event log, as generated in isolation and after the arbiter rescales events for
  resource contention.
- ``device_summary.csv`` -- per-device compute/bandwidth utilization, busy
  fraction, peak memory occupancy, DMA transfer totals and the per-device
  execution-state breakdown (fraction of the run spent compute-bound,
  bandwidth-bound, waiting on KV / weights / experts, in kernel-launch overhead,
  or idle).
- ``memory_summary.csv`` -- per-memory-device bandwidth utilization, busy
  fraction, bytes moved, peak occupancy and the compute devices it serves; this
  is the memory-side view of the topology, independent of the compute devices,
  so it stays meaningful if a memory is shared across compute devices.
- ``device_timeline.csv`` -- per-device busy fraction, memory occupancy and the
  execution-state breakdown over time (bucketed).

Memory occupancy is the per-device *reserved* footprint (weights + KV) of the
jobs active at each instant, as sized by the parallelism planner; it is a
first-cut reservation model, not a byte-accurate residency trace.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .orchestrator import (
    DecisionRecord,
    EventRecord,
    JobRecord,
    RequestRecord,
    RunResult,
)

# --- statistics -----------------------------------------------------------------


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated ``q``-th percentile (``q`` in [0, 100])."""

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def _distribution(values: Iterable[float]) -> dict[str, float]:
    """Summary stats (count/mean/min/max/p50/p90/p99) of a value series."""

    series = [v for v in values if v is not None]
    if not series:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0,
                "p50": 0.0, "p90": 0.0, "p99": 0.0}
    return {
        "count": len(series),
        "mean": sum(series) / len(series),
        "min": min(series),
        "max": max(series),
        "p50": _percentile(series, 50),
        "p90": _percentile(series, 90),
        "p99": _percentile(series, 99),
    }


def _union_length(intervals: Sequence[tuple[float, float]]) -> float:
    """Total length covered by a set of (start, end) intervals (merged)."""

    spans = sorted((s, e) for s, e in intervals if e > s)
    if not spans:
        return 0.0
    total = 0.0
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return total


# --- aggregation ----------------------------------------------------------------

_COMPUTE_PHASES = ("prefill", "decode")

# Orchestration-decision kinds, in report order.
_DECISION_KINDS = (
    "weight_load", "weight_eviction", "prefill", "kv_reuse", "kv_transfer",
    "decode", "kv_eviction", "expert_load", "expert_eviction",
)


def _decision_counts(decisions: Sequence[DecisionRecord]) -> dict[str, int]:
    """Count of each orchestration-decision kind (all kinds, including zeros)."""

    counts = {kind: 0 for kind in _DECISION_KINDS}
    for d in decisions:
        counts[d.kind] = counts.get(d.kind, 0) + 1
    return counts


def _rescaled(events: Sequence[EventRecord]) -> list[EventRecord]:
    return [e for e in events if e.rescaled]


def _isolated(events: Sequence[EventRecord]) -> list[EventRecord]:
    return [e for e in events if not e.rescaled]


# Per-device execution-state taxonomy (finer than busy/idle). At any instant a
# device is attributed to exactly one state, so the states partition the run and
# their fractions sum to 1. Listed in *priority* order: when events overlap on a
# device (e.g. a transfer prefetching while a forward pass runs), the
# higher-priority (earlier-listed) state wins that wall-clock interval. ``idle``
# is the time no event covers. Note: a "waiting for tensors" state (inter-stage
# activations / tensor-parallel collectives) is not yet modelled as events, so
# it is not emitted.
DEVICE_STATES = (
    "compute_bound",     # running a forward pass, compute-bound
    "bandwidth_bound",   # running a forward pass, memory-bandwidth-bound
    "waiting_kv",        # stalled fetching KV cache
    "waiting_weights",   # stalled staging (non-expert) model weights
    "waiting_experts",   # stalled streaming routed MoE experts
    "kernel_launch",     # kernel-launch latency overhead
    "idle",              # no work assigned
)

# Priority index per state (lower wins); ``idle`` is handled as the leftover.
_STATE_PRIORITY = {state: i for i, state in enumerate(DEVICE_STATES)
                   if state != "idle"}


def _event_state(event: EventRecord) -> str:
    """The device execution state an event represents."""

    if event.phase in _COMPUTE_PHASES:
        return ("compute_bound" if event.compute_time >= event.bandwidth_time
                else "bandwidth_bound")
    if event.phase == "transfer":
        return "waiting_kv"
    if event.phase == "weight_transfer":
        return "waiting_weights"
    if event.phase == "expert_transfer":
        return "waiting_experts"
    if event.phase == "kernel_launch":
        return "kernel_launch"
    return "idle"


def _state_seconds(
    events: Sequence[EventRecord], window_start: float, window_end: float
) -> dict[str, float]:
    """Partition ``[window_start, window_end]`` across device states by priority.

    Each instant is attributed to the single highest-priority state among the
    events covering it; instants no event covers are ``idle``. Compute/bandwidth
    bound is the event's intrinsic classification (``compute_time`` vs
    ``bandwidth_time``), preserved through arbiter rescaling. Returns seconds per
    state, summing to the window width.
    """

    span = max(0.0, window_end - window_start)
    seconds = {state: 0.0 for state in DEVICE_STATES}
    if span <= 0:
        return seconds

    intervals: list[tuple[float, float, int, str]] = []
    for event in events:
        state = _event_state(event)
        if state == "idle":
            continue
        start = max(window_start, event.start)
        end = min(window_end, event.end)
        if end > start:
            intervals.append((start, end, _STATE_PRIORITY[state], state))

    if not intervals:
        seconds["idle"] = span
        return seconds

    points = sorted({window_start, window_end}
                    | {p for s, e, _, _ in intervals for p in (s, e)})
    covered = 0.0
    for t0, t1 in zip(points, points[1:]):
        if t1 <= t0:
            continue
        active = [(pr, st) for s, e, pr, st in intervals if s <= t0 and e >= t1]
        if active:
            _, state = min(active)
            seconds[state] += t1 - t0
            covered += t1 - t0
    seconds["idle"] = max(0.0, span - covered)
    return seconds


def device_summaries(result: RunResult) -> list[dict[str, Any]]:
    """Per-device utilization, peak memory occupancy and DMA totals."""

    makespan = result.makespan or 0.0
    rescaled = _rescaled(result.events)
    names = sorted({e.device for e in rescaled if e.device} |
                   {d for j in result.jobs for d in j.devices})

    summaries: list[dict[str, Any]] = []
    for name in names:
        dev_events = [e for e in rescaled if e.device == name]
        compute_seconds = sum(e.compute_time for e in dev_events
                              if e.phase in _COMPUTE_PHASES)
        bandwidth_seconds = sum(e.bandwidth_time for e in dev_events
                                if e.phase in _COMPUTE_PHASES)
        busy = _union_length([(e.start, e.end) for e in dev_events
                              if e.phase != "kernel_launch"])
        transfers = [e for e in dev_events
                     if e.phase in ("transfer", "weight_transfer", "expert_transfer")]
        peak_mem = _peak_occupancy(result, name)
        states = _state_seconds(dev_events, 0.0, makespan)
        summary = {
            "device": name,
            "busy_fraction": busy / makespan if makespan else 0.0,
            "compute_util": compute_seconds / makespan if makespan else 0.0,
            "bandwidth_util": bandwidth_seconds / makespan if makespan else 0.0,
            "peak_memory_bytes": peak_mem,
            "num_transfers": len(transfers),
            "transfer_bytes": sum(e.bytes_read for e in transfers),
        }
        for state in DEVICE_STATES:
            summary[f"{state}_fraction"] = (
                states[state] / makespan if makespan else 0.0
            )
        summaries.append(summary)
    return summaries


def _peak_occupancy(result: RunResult, device: str) -> float:
    """Peak reserved bytes on ``device`` across all job-boundary instants."""

    jobs = [j for j in result.jobs if device in j.devices and j.per_device_bytes]
    if not jobs:
        return 0.0
    breakpoints = sorted({j.start for j in jobs} | {j.end for j in jobs})
    peak = 0.0
    for t in breakpoints:
        occ = sum(j.per_device_bytes for j in jobs if j.start <= t < j.end)
        peak = max(peak, occ)
    return peak


def _occupancy_at(jobs: Sequence[JobRecord], device: str, t: float) -> float:
    return sum(j.per_device_bytes for j in jobs
               if device in j.devices and j.start <= t < j.end)


def device_timeline(result: RunResult, num_buckets: int = 64) -> list[dict[str, Any]]:
    """Per-device busy fraction and memory occupancy over time (bucketed)."""

    makespan = result.makespan or 0.0
    if makespan <= 0 or num_buckets < 1:
        return []
    width = makespan / num_buckets
    rescaled = _rescaled(result.events)
    names = sorted({e.device for e in rescaled if e.device} |
                   {d for j in result.jobs for d in j.devices})

    rows: list[dict[str, Any]] = []
    for b in range(num_buckets):
        t0 = b * width
        t1 = (b + 1) * width
        for name in names:
            dev_events = [e for e in rescaled if e.device == name]
            busy_events = [e for e in dev_events if e.phase != "kernel_launch"]
            overlap = sum(min(e.end, t1) - max(e.start, t0)
                          for e in busy_events
                          if e.end > t0 and e.start < t1)
            row = {
                "bucket": b,
                "time_start": t0,
                "time_end": t1,
                "device": name,
                "busy_fraction": overlap / width if width else 0.0,
                "memory_bytes": _occupancy_at(result.jobs, name, t0),
            }
            states = _state_seconds(dev_events, t0, t1)
            for state in DEVICE_STATES:
                row[f"{state}_fraction"] = states[state] / width if width else 0.0
            rows.append(row)
    return rows


def _memory_peak_occupancy(result: RunResult, attached_devices: Sequence[str]) -> float:
    """Peak reserved bytes held in a memory, summed over the devices it serves.

    The footprint model reserves ``per_device_bytes`` (weights + KV) on each
    compute device's first-tier memory; a memory shared by several devices holds
    the sum. Evaluated at every job boundary to find the peak.
    """

    attached = set(attached_devices)
    jobs = [
        j for j in result.jobs
        if j.per_device_bytes and any(d in attached for d in j.devices)
    ]
    if not jobs:
        return 0.0
    breakpoints = sorted({j.start for j in jobs} | {j.end for j in jobs})
    peak = 0.0
    for t in breakpoints:
        occ = sum(
            j.per_device_bytes * sum(1 for d in j.devices if d in attached)
            for j in jobs
            if j.start <= t < j.end
        )
        peak = max(peak, occ)
    return peak


def memory_summaries(result: RunResult) -> list[dict[str, Any]]:
    """Per-memory-device bandwidth utilization, busy fraction and occupancy.

    Keyed off the memory devices in the system inventory (so idle memories such
    as a node's CPU memory still appear), with bandwidth attributed to whichever
    memory each event actually streamed from -- not assumed from its compute
    device -- so the view stays correct when a memory backs several devices.
    """

    makespan = result.makespan or 0.0
    rescaled = _rescaled(result.events)
    by_memory: dict[str, list[EventRecord]] = {}
    for event in rescaled:
        if event.memory and event.bandwidth_time > 0:
            by_memory.setdefault(event.memory, []).append(event)

    summaries: list[dict[str, Any]] = []
    for mem in result.memories:
        mem_events = by_memory.get(mem.name, [])
        bandwidth_seconds = sum(e.bandwidth_time for e in mem_events)
        bytes_moved = sum(e.bytes_read for e in mem_events)
        busy = _union_length([(e.start, e.end) for e in mem_events])
        peak_mem = _memory_peak_occupancy(result, mem.attached_devices)
        summaries.append({
            "memory": mem.name,
            "role": mem.role,
            "node": mem.node,
            "attached_devices": " ".join(mem.attached_devices),
            "capacity_bytes": mem.capacity_bytes,
            "bandwidth_bytes_per_s": mem.bandwidth_bytes_per_s,
            "busy_fraction": busy / makespan if makespan else 0.0,
            "bandwidth_util": bandwidth_seconds / makespan if makespan else 0.0,
            "num_events": len(mem_events),
            "bytes_moved": bytes_moved,
            "peak_memory_bytes": peak_mem,
            "occupancy_fraction": (
                peak_mem / mem.capacity_bytes if mem.capacity_bytes else 0.0
            ),
        })
    return summaries


def summarize(result: RunResult) -> dict[str, Any]:
    """Aggregate run report over the whole suite."""

    records = result.records
    makespan = result.makespan or 0.0
    isolated = _isolated(result.events)
    transfers = [e for e in isolated
                 if e.phase in ("transfer", "weight_transfer", "expert_transfer")]
    devices = device_summaries(result)
    memories = memory_summaries(result)

    total_output = sum(r.output_tokens for r in records)
    total_prompt = sum(r.prompt_tokens for r in records)

    return {
        "num_requests": len(records),
        "num_batches": result.num_batches,
        "num_events": len(isolated),
        "makespan_s": makespan,
        "total_flops": sum(e.flops for e in isolated),
        "total_bytes_read": sum(e.bytes_read for e in isolated),
        "num_dma_transfers": len(transfers),
        "dma_transfer_bytes": sum(e.bytes_read for e in transfers),
        "peak_memory_bytes": max((d["peak_memory_bytes"] for d in devices),
                                 default=0.0),
        "num_memory_devices": len(memories),
        "total_memory_capacity_bytes": sum(m["capacity_bytes"] for m in memories),
        "total_prompt_tokens": total_prompt,
        "total_output_tokens": total_output,
        "throughput_requests_per_s": len(records) / makespan if makespan else 0.0,
        "throughput_output_tokens_per_s": total_output / makespan if makespan else 0.0,
        "throughput_processed_tokens_per_s":
            (total_prompt + total_output) / makespan if makespan else 0.0,
        "latency_s": _distribution(r.latency for r in records),
        "queue_delay_s": _distribution(r.queue_delay for r in records),
        "ttft_s": _distribution(r.ttft for r in records),
        "tpot_s": _distribution(r.tpot for r in records),
        "tps_tokens_per_s": _distribution(r.tps for r in records),
        "num_decisions": len(result.decisions),
        "decision_counts": _decision_counts(result.decisions),
    }


# --- writing --------------------------------------------------------------------


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _request_rows(records: Sequence[RequestRecord]) -> list[dict[str, Any]]:
    rows = []
    for r in sorted(records, key=lambda x: x.request_id):
        rows.append({
            "request_id": r.request_id,
            "arrival_time": r.arrival_time,
            "dispatch_time": r.dispatch_time,
            "first_token_time": r.first_token_time,
            "completion_time": r.completion_time,
            "prompt_tokens": r.prompt_tokens,
            "output_tokens": r.output_tokens,
            "batch_index": r.batch_index,
            "queue_delay": r.queue_delay,
            "latency": r.latency,
            "ttft": r.ttft,
            "tpot": r.tpot,
            "tps": r.tps,
        })
    return rows


def _sequence_id(workload_id: int | None, turn_index: int | None) -> str:
    """Human-readable sequence id (``w<workload>t<turn>``), or empty if unknown."""

    if workload_id is None or turn_index is None or workload_id < 0:
        return ""
    return f"w{workload_id}t{turn_index}"


def _batch_sequence_ids(members: Sequence[tuple[int, int]]) -> str:
    """Space-joined sequence ids for every tenant of a batch (skips synthetic)."""

    ids = [s for (w, t) in members if (s := _sequence_id(w, t))]
    return " ".join(ids)


_DECISION_FIELDS = (
    "time", "time_started", "time_completed", "kind", "request_id", "sequence",
    "model", "devices", "batch_index", "tokens", "source_sequence",
    "source_request_id", "source_devices",
)


def _decision_rows(decisions: Sequence[DecisionRecord]) -> list[dict[str, Any]]:
    """Flatten orchestration decisions into CSV rows, ordered by time."""

    rows = []
    for d in sorted(decisions, key=lambda x: (x.time, x.request_id, x.kind)):
        # Batched acts (prefill/decode) list every batch tenant in ``sequence``.
        sequence = (
            _batch_sequence_ids(d.batch_members) if d.batch_members
            else _sequence_id(d.workload_id, d.turn_index)
        )
        rows.append({
            "time": d.time,
            "time_started": "" if d.time_started is None else d.time_started,
            "time_completed": "" if d.time_completed is None else d.time_completed,
            "kind": d.kind,
            "request_id": d.request_id,
            "sequence": sequence,
            "model": d.model,
            "devices": " ".join(d.devices),
            "batch_index": d.batch_index,
            "tokens": d.tokens,
            "source_sequence": _sequence_id(d.source_workload_id, d.source_turn_index),
            "source_request_id": (
                "" if d.source_request_id is None else d.source_request_id
            ),
            "source_devices": " ".join(d.source_devices),
        })
    return rows


_EVENT_FIELDS = (
    "job_index", "batch_index", "job_phase", "request_ids", "group_index",
    "phase", "device", "memory", "flops", "bytes_read", "compute_time",
    "bandwidth_time", "duration", "start", "end",
)


def _event_rows(events: Sequence[EventRecord]) -> list[dict[str, Any]]:
    rows = []
    for e in events:
        row = asdict(e)
        row.pop("rescaled")
        row["request_ids"] = " ".join(str(r) for r in e.request_ids)
        rows.append(row)
    return rows


def _format_distribution(name: str, dist: Mapping[str, Any]) -> str:
    return (f"  {name:<14} n={dist['count']:<6} mean={dist['mean']:.6g} "
            f"p50={dist['p50']:.6g} p90={dist['p90']:.6g} "
            f"p99={dist['p99']:.6g} max={dist['max']:.6g}")


def _report_text(report: Mapping[str, Any], devices: Sequence[Mapping[str, Any]],
                 memories: Sequence[Mapping[str, Any]], run_id: str) -> str:
    lines = [
        f"serve_sim run report: {run_id}",
        "=" * 48,
        f"  requests        {report['num_requests']}",
        f"  batches         {report['num_batches']}",
        f"  events          {report['num_events']}",
        f"  makespan (s)    {report['makespan_s']:.6g}",
        f"  total FLOPs     {report['total_flops']:.6g}",
        f"  total bytes     {report['total_bytes_read']:.6g}",
        f"  DMA transfers   {report['num_dma_transfers']} "
        f"({report['dma_transfer_bytes']:.6g} bytes)",
        f"  peak memory     {report['peak_memory_bytes']:.6g} bytes",
        f"  req throughput  {report['throughput_requests_per_s']:.6g} req/s",
        f"  tok throughput  {report['throughput_output_tokens_per_s']:.6g} out-tok/s",
        f"  avg TPS         {report['tps_tokens_per_s']['mean']:.6g} tok/s",
        f"  avg TTFT        {report['ttft_s']['mean']:.6g} s",
        "",
        "Distributions (seconds):",
        _format_distribution("latency", report["latency_s"]),
        _format_distribution("queue_delay", report["queue_delay_s"]),
        _format_distribution("ttft", report["ttft_s"]),
        _format_distribution("tpot", report["tpot_s"]),
        _format_distribution("tps", report["tps_tokens_per_s"]),
        "",
        "Orchestration decisions:",
        f"  total           {report['num_decisions']}",
    ]
    for kind in _DECISION_KINDS:
        lines.append(f"  {kind:<14} {report['decision_counts'].get(kind, 0)}")
    lines += [
        "",
        "Per-device:",
    ]
    for d in devices:
        lines.append(
            f"  {d['device']:<16} busy={d['busy_fraction']:.3f} "
            f"compute={d['compute_util']:.3f} bw={d['bandwidth_util']:.3f} "
            f"peak_mem={d['peak_memory_bytes']:.6g} "
            f"transfers={d['num_transfers']}"
        )
        lines.append(
            f"  {'':<16} states: cmp={d['compute_bound_fraction']:.3f} "
            f"bw={d['bandwidth_bound_fraction']:.3f} "
            f"kv={d['waiting_kv_fraction']:.3f} "
            f"wts={d['waiting_weights_fraction']:.3f} "
            f"exp={d['waiting_experts_fraction']:.3f} "
            f"klaunch={d['kernel_launch_fraction']:.3f} "
            f"idle={d['idle_fraction']:.3f}"
        )
    lines.append("")
    lines.append("Per-memory:")
    for m in memories:
        lines.append(
            f"  {m['memory']:<20} [{m['role']:<11}] busy={m['busy_fraction']:.3f} "
            f"bw={m['bandwidth_util']:.3f} moved={m['bytes_moved']:.6g} "
            f"peak_mem={m['peak_memory_bytes']:.6g}/{m['capacity_bytes']:.6g} "
            f"({m['occupancy_fraction']:.3f})"
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    result: RunResult,
    out_dir: str | Path,
    *,
    run_id: str = "run",
    config: Mapping[str, Any] | None = None,
    time_buckets: int = 64,
) -> Path:
    """Write all raw outputs for ``result`` under ``out_dir`` and return the path."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = summarize(result)
    devices = device_summaries(result)
    memories = memory_summaries(result)
    timeline = device_timeline(result, time_buckets)

    with open(out / "run_report.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"run_id": run_id, "report": report, "devices": devices,
             "memories": memories},
            handle, indent=2)
    with open(out / "run_report.txt", "w", encoding="utf-8") as handle:
        handle.write(_report_text(report, devices, memories, run_id))

    _write_csv(
        out / "requests.csv",
        ["request_id", "arrival_time", "dispatch_time", "first_token_time",
         "completion_time", "prompt_tokens", "output_tokens", "batch_index",
         "queue_delay", "latency", "ttft", "tpot", "tps"],
        _request_rows(result.records),
    )
    _write_csv(out / "orchestration_decisions.csv", _DECISION_FIELDS,
               _decision_rows(result.decisions))
    _write_csv(out / "events_before_rescaling.csv", _EVENT_FIELDS,
               _event_rows(_isolated(result.events)))
    _write_csv(out / "events_after_rescaling.csv", _EVENT_FIELDS,
               _event_rows(_rescaled(result.events)))
    _write_csv(
        out / "device_summary.csv",
        ["device", "busy_fraction", "compute_util", "bandwidth_util",
         "peak_memory_bytes", "num_transfers", "transfer_bytes",
         *(f"{state}_fraction" for state in DEVICE_STATES)],
        devices,
    )
    _write_csv(
        out / "memory_summary.csv",
        ["memory", "role", "node", "attached_devices", "capacity_bytes",
         "bandwidth_bytes_per_s", "busy_fraction", "bandwidth_util",
         "num_events", "bytes_moved", "peak_memory_bytes", "occupancy_fraction"],
        memories,
    )
    _write_csv(
        out / "device_timeline.csv",
        ["bucket", "time_start", "time_end", "device", "busy_fraction",
         "memory_bytes", *(f"{state}_fraction" for state in DEVICE_STATES)],
        timeline,
    )
    if config is not None:
        with open(out / "config.json", "w", encoding="utf-8") as handle:
            json.dump(dict(config), handle, indent=2)

    return out
