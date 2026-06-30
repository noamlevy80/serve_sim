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
- ``device_timeline.csv`` -- per-device busy fraction, memory occupancy, the
  first-tier content breakdown (``content_json``: KV vs weights bytes), the
  execution-state breakdown, achieved compute and first-tier bandwidth, and the
  dominant transfer source/object over time (bucketed).
- ``memory_timeline.csv`` -- per-memory-device bandwidth, occupancy, the content
  breakdown (``content_json``: KV vs weights bytes) and the dominant
  transfer source/object over time (bucketed).
- ``workload_timeline.csv`` -- per-workload current turn, lifecycle state
  (not-arrived / in-queue / KV-fetch / prefill / decode / done) and serving
  device over time (bucketed).
- ``viz.json`` -- a single GUI-ready payload bundling the summary, the device and
  memory specs/aggregates, and the device, memory and workload timelines at a
  finer bucket resolution; built by :func:`build_viz_payload` so the
  visualization tool stays a pure renderer.

Memory occupancy is the per-device *reserved* footprint (weights + KV) of the
jobs active at each instant, as sized by the parallelism planner; it is a
first-cut reservation model, not a byte-accurate residency trace.
"""

from __future__ import annotations

import bisect
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .orchestrator import (
    DecisionRecord,
    DeviceRecord,
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

# Scale-up-network parallelism collectives (tensor-parallel all-reduce,
# expert-parallel all-to-all, pipeline-parallel activation hand-off).
_COMM_PHASES = ("tp_comm", "ep_comm", "pp_comm")

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
# is the time no event covers. The ``communicating`` state covers the scale-up
# network collectives a sharded forward pass incurs (tensor-parallel all-reduce,
# expert-parallel all-to-all and pipeline-parallel activation hand-off).
DEVICE_STATES = (
    "compute_bound",     # running a forward pass, compute-bound
    "bandwidth_bound",   # running a forward pass, memory-bandwidth-bound
    "communicating",     # stalled in a parallelism comm collective
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
    if event.phase in _COMM_PHASES:
        return "communicating"
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

    # Sweep line: each clipped interval contributes a +1 at its start and a -1
    # at its end. Between consecutive boundaries the covering interval with the
    # lowest priority index wins (matching the previous min() tie-break, since
    # priorities are unique per state). This is O(I log I) versus the former
    # O(points * intervals) scan.
    starts: dict[float, list[int]] = {}
    ends: dict[float, list[int]] = {}
    for event in events:
        state = _event_state(event)
        if state == "idle":
            continue
        start = max(window_start, event.start)
        end = min(window_end, event.end)
        if end > start:
            pr = _STATE_PRIORITY[state]
            starts.setdefault(start, []).append(pr)
            ends.setdefault(end, []).append(pr)

    if not starts:
        seconds["idle"] = span
        return seconds

    points = sorted(set(starts) | set(ends) | {window_start, window_end})
    active_counts: dict[int, int] = {}
    covered = 0.0
    for t0, t1 in zip(points, points[1:]):
        for pr in ends.get(t0, ()):  # intervals ending at t0 stop covering [t0, t1)
            n = active_counts[pr] - 1
            if n:
                active_counts[pr] = n
            else:
                del active_counts[pr]
        for pr in starts.get(t0, ()):  # intervals starting at t0 cover [t0, t1)
            active_counts[pr] = active_counts.get(pr, 0) + 1
        if t1 <= t0:
            continue
        if active_counts:
            state = DEVICE_STATES[min(active_counts)]
            seconds[state] += t1 - t0
            covered += t1 - t0
    seconds["idle"] = max(0.0, span - covered)
    return seconds


def device_summaries(
    result: RunResult, *, progress: Callable[[float, float], None] | None = None
) -> list[dict[str, Any]]:
    """Per-device utilization, peak memory occupancy and DMA totals."""

    makespan = result.makespan or 0.0
    rescaled = _rescaled(result.events)
    specs = {d.name: d for d in result.device_specs}
    names = sorted({e.device for e in rescaled if e.device} |
                   {d for j in result.jobs for d in j.devices} |
                   set(specs))

    summaries: list[dict[str, Any]] = []
    for name in names:
        spec = specs.get(name)
        dev_events = [e for e in rescaled if e.device == name]
        compute_seconds = sum(e.compute_time for e in dev_events
                              if e.phase in _COMPUTE_PHASES)
        bandwidth_seconds = sum(e.bandwidth_time for e in dev_events
                                if e.phase in _COMPUTE_PHASES)
        busy = _union_length([(e.start, e.end) for e in dev_events
                              if e.phase != "kernel_launch"])
        transfers = [e for e in dev_events
                     if e.phase in ("transfer", "weight_transfer", "expert_transfer")
                     and e.bytes_read > 0]
        peak_mem = _peak_occupancy(result, name)
        states = _state_seconds(dev_events, 0.0, makespan)
        summary = {
            "device": name,
            "node": spec.node if spec else "",
            "peak_flops_fp16": spec.peak_flops_fp16 if spec else 0.0,
            "first_tier_memory": spec.first_tier_memory if spec else "",
            "first_tier_capacity_bytes":
                spec.first_tier_capacity_bytes if spec else 0.0,
            "first_tier_bandwidth_bytes_per_s":
                spec.first_tier_bandwidth_bytes_per_s if spec else 0.0,
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
        if progress is not None:
            progress(len(summaries), len(names))
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


def _resident_tasks_at(jobs: Sequence[JobRecord], device: str, t: float) -> int:
    """How many jobs (running batches) are resident on ``device`` at instant ``t``.

    A job pins its footprint on its serving devices for its whole ``[start, end)``
    span, so this counts every batch concurrently resident -- not just the one
    dominating compute -- and tracks the memory-occupancy series instant for
    instant.
    """

    return sum(1 for j in jobs
               if device in j.devices and j.start <= t < j.end)


_TRANSFER_PHASES = ("transfer", "weight_transfer", "expert_transfer")


def _overlap(event: EventRecord, t0: float, t1: float) -> float:
    """Wall-clock seconds of ``event`` falling inside ``[t0, t1]``."""

    return max(0.0, min(event.end, t1) - max(event.start, t0))


def _overlap_fraction(event: EventRecord, t0: float, t1: float) -> float:
    """Fraction of ``event``'s duration that falls inside ``[t0, t1]``."""

    if event.duration <= 0:
        return 1.0 if event.start >= t0 and event.start < t1 else 0.0
    return _overlap(event, t0, t1) / event.duration


def _sequence_by_request(records: Sequence[RequestRecord]) -> dict[int, str]:
    """Map each request id to its ``w<workload>t<turn>`` sequence label."""

    return {
        r.request_id: _sequence_id(r.workload_id, r.turn_index)
        for r in records
    }


def _transfer_object_label(event: EventRecord, seq_by_request: Mapping[int, str]) -> str:
    """Human label for what a transfer event moves (weights / experts / KV)."""

    if event.phase == "weight_transfer":
        return f"weights:{event.model}" if event.model else "weights"
    if event.phase == "expert_transfer":
        return f"experts:{event.model}" if event.model else "experts"
    if event.phase == "transfer":
        seq = next((seq_by_request.get(rid, "") for rid in event.request_ids
                    if seq_by_request.get(rid)), "")
        return f"kv:{seq}" if seq else "kv"
    return ""


def _dominant_transfer(
    events: Sequence[EventRecord], t0: float, t1: float
) -> EventRecord | None:
    """The transfer-family event covering the most of ``[t0, t1]`` (or None)."""

    best: EventRecord | None = None
    best_overlap = 0.0
    for e in events:
        if e.phase not in _TRANSFER_PHASES:
            continue
        ov = _overlap(e, t0, t1)
        if ov > best_overlap:
            best, best_overlap = e, ov
    return best


def _phase_token_throughput(
    events: Sequence[EventRecord],
    result: RunResult,
    num_buckets: int,
    width: float,
) -> tuple[dict[tuple[str, int], float], dict[tuple[str, int], float]]:
    """Effective per-(device, bucket) decode and prefill token throughput.

    A batch's token rate is a property of the whole task, not of any single rank,
    so every device in the batch's engine group is credited the same value. For
    each job the phase's token count (output tokens for decode, prompt tokens for
    prefill) is spread evenly over that phase's wall-clock window -- ``[min start,
    max end]`` of its rescaled events -- and the rate is prorated into the buckets
    the window covers and attributed to every device the job ran on. Returns two
    ``{(device, bucket): tokens_per_second}`` maps (decode, prefill).
    """

    jobs_by_index = {j.job_index: j for j in result.jobs}
    tokens = {r.request_id: (r.prompt_tokens, r.output_tokens) for r in result.records}

    def windows(phase: str) -> dict[int, tuple[float, float]]:
        spans: dict[int, list[float]] = {}
        for e in events:
            if e.phase != phase:
                continue
            span = spans.get(e.job_index)
            if span is None:
                spans[e.job_index] = [e.start, e.end]
            else:
                span[0] = min(span[0], e.start)
                span[1] = max(span[1], e.end)
        return {idx: (s, e) for idx, (s, e) in spans.items()}

    def distribute(phase: str, token_index: int) -> dict[tuple[str, int], float]:
        acc: dict[tuple[str, int], float] = {}
        if width <= 0:
            return acc
        for job_index, (start, end) in windows(phase).items():
            job = jobs_by_index.get(job_index)
            if job is None or end <= start:
                continue
            total = sum(tokens.get(rid, (0, 0))[token_index] for rid in job.request_ids)
            if total <= 0:
                continue
            rate = total / (end - start)
            first = max(0, int(start // width))
            last = min(num_buckets - 1, int((end - 1e-12) // width))
            for b in range(first, last + 1):
                ov = max(0.0, min(end, (b + 1) * width) - max(start, b * width))
                if ov <= 0:
                    continue
                contrib = rate * ov / width
                for device in job.devices:
                    acc[(device, b)] = acc.get((device, b), 0.0) + contrib
        return acc

    return distribute("decode", 1), distribute("prefill", 0)


def device_timeline(
    result: RunResult, num_buckets: int = 64, *,
    progress: Callable[[float, float], None] | None = None,
) -> list[dict[str, Any]]:
    """Per-device busy fraction, occupancy, state and throughput over time.

    Each row is one device in one time bucket. Beyond the busy fraction, memory
    occupancy and execution-state breakdown, it carries the achieved compute
    (FLOP/s) and first-tier bandwidth (bytes/s) in the bucket, the first-tier
    occupancy split into weights and per-task KV (``content``: a ``weights``
    band plus one ``KV B<n>`` band per co-resident dispatch batch), a discrete
    label for any
    incoming transfer (its source memory and what it moves), the current task's
    batch size, the number of tasks (running batches) concurrently resident on
    the device, the ids of the sequences resident on the device for the whole
    job lifetime (``running_sequences`` -- spanning the wait for weights/KV,
    prefill and decode; a stack when more than one), and the effective decode
    (output) and prefill (input) token throughput -- the same for every device in
    the task's engine group -- so the visualization can plot absolute values
    against the device's static ceilings.
    """

    makespan = result.makespan or 0.0
    if makespan <= 0 or num_buckets < 1:
        return []
    width = makespan / num_buckets
    rescaled = _rescaled(result.events)
    specs = {d.name: d for d in result.device_specs}
    seq_by_request = _sequence_by_request(result.records)
    names = sorted({e.device for e in rescaled if e.device} |
                   {d for j in result.jobs for d in j.devices} |
                   set(specs))
    events_by_device: dict[str, list[EventRecord]] = {}
    for e in rescaled:
        if e.device:
            events_by_device.setdefault(e.device, []).append(e)
    # Jobs resident on each device, so the per-(device, bucket) occupancy/content
    # helpers scan only the handful of jobs on that device rather than all jobs.
    jobs_by_device: dict[str, list[JobRecord]] = {}
    for j in result.jobs:
        for d in j.devices:
            jobs_by_device.setdefault(d, []).append(j)

    # Per-(device, bucket) effective token throughput. A task's token rate is a
    # property of the whole batch, so every device in its engine group reports
    # the same value: the batch's decode (output) or prefill (input) tokens
    # spread over that phase's wall-clock window and prorated into each bucket.
    decode_tps, prefill_tps = _phase_token_throughput(
        rescaled, result, num_buckets, width
    )

    rows: list[dict[str, Any]] = []
    for b in range(num_buckets):
        t0 = b * width
        t1 = (b + 1) * width
        for name in names:
            spec = specs.get(name)
            first_tier = spec.first_tier_memory if spec else ""
            dev_events = [e for e in events_by_device.get(name, [])
                          if e.end > t0 and e.start < t1]
            busy_events = [e for e in dev_events if e.phase != "kernel_launch"]
            overlap = sum(_overlap(e, t0, t1) for e in busy_events)
            compute_events = [e for e in dev_events if e.phase in _COMPUTE_PHASES]
            # 1.7 Current task batch size: the dominant compute event's batch.
            batch_size = (
                len(max(compute_events, key=lambda e: _overlap(e, t0, t1)).request_ids)
                if compute_events else 0
            )
            bucket_flops = sum(e.flops * _overlap_fraction(e, t0, t1)
                               for e in compute_events)
            bucket_compute_s = sum(e.compute_time * _overlap_fraction(e, t0, t1)
                                   for e in compute_events)
            first_tier_events = [e for e in dev_events if e.memory == first_tier]
            bucket_bytes = sum(e.bytes_read * _overlap_fraction(e, t0, t1)
                               for e in first_tier_events)
            bucket_bw_s = sum(e.bandwidth_time * _overlap_fraction(e, t0, t1)
                              for e in first_tier_events)
            dom = _dominant_transfer(dev_events, t0, t1)
            dev_jobs = jobs_by_device.get(name, [])
            # 1.11 Running sequence(s): every sequence resident on the device in
            # the bucket -- the whole job lifetime, so it spans the wait for
            # weights/KV, prefill and decode, not just active compute. More than
            # one renders as a stack.
            running_seqs = sorted({
                seq_by_request.get(rid) or f"r{rid}"
                for j in dev_jobs if j.start < t1 and j.end > t0
                for rid in j.request_ids
            })
            dev_weights, dev_kv_by_batch = _first_tier_content_at(dev_jobs, {name}, t0)
            content: dict[str, float] = {}
            weight_bytes = sum(dev_weights.values())
            if weight_bytes > 0:
                content["weights"] = weight_bytes
            for batch in sorted(dev_kv_by_batch):
                if dev_kv_by_batch[batch] > 0:
                    content[f"KV B{batch}"] = dev_kv_by_batch[batch]
            row = {
                "bucket": b,
                "time_start": t0,
                "time_end": t1,
                "device": name,
                "busy_fraction": overlap / width if width else 0.0,
                "memory_bytes": _occupancy_at(dev_jobs, name, t0),
                "content": content,
                "batch_size": batch_size,
                "resident_tasks": _resident_tasks_at(dev_jobs, name, t0),
                "running_sequences": running_seqs,
                "decode_tokens_per_s": decode_tps.get((name, b), 0.0),
                "prefill_tokens_per_s": prefill_tps.get((name, b), 0.0),
                "compute_flops_per_s": bucket_flops / width if width else 0.0,
                "compute_seconds": bucket_compute_s,
                "first_tier_bytes_per_s": bucket_bytes / width if width else 0.0,
                "bandwidth_seconds": bucket_bw_s,
                "transfer_source": dom.memory if dom else "",
                "transfer_object": (
                    _transfer_object_label(dom, seq_by_request) if dom else ""
                ),
            }
            states = _state_seconds(dev_events, t0, t1)
            for state in DEVICE_STATES:
                row[f"{state}_fraction"] = states[state] / width if width else 0.0
            rows.append(row)
        if progress is not None:
            progress(b + 1, num_buckets)
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
        if event.bandwidth_time <= 0:
            continue
        # A transfer occupies both ends' bandwidth, so it counts against the
        # memory it streamed from (``memory``) and the one it wrote into
        # (``destination_memory``); a compute read has only a source memory.
        names = set()
        if event.memory:
            names.add(event.memory)
        if event.destination_memory:
            names.add(event.destination_memory)
        for name in names:
            by_memory.setdefault(name, []).append(event)

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


def _floating_kv_residency(
    result: RunResult,
) -> dict[str, list[tuple[float, float, str, float]]]:
    """Reconstruct KV residency in floating memories from the decision log.

    A completed sequence's KV is offloaded to a floating memory (a ``kv_transfer``
    decision targeting that single memory, tagged with the bytes moved) and later
    evicted (a ``kv_eviction`` decision naming the same memory and sequence). The
    pair bounds a residency interval ``[store, evict)`` (open to the makespan if
    never evicted). Returns, per memory name, the list of
    ``(start, end, sequence_label, bytes)`` intervals.
    """

    makespan = result.makespan or 0.0
    stores: dict[tuple[str, int, int], tuple[float, float]] = {}
    evicts: dict[tuple[str, int, int], float] = {}
    for d in result.decisions:
        if len(d.devices) != 1:
            continue
        key = (d.devices[0], d.workload_id, d.turn_index)
        if d.kind == "kv_transfer" and d.bytes_moved > 0:
            stores[key] = (d.time, d.bytes_moved)
        elif d.kind == "kv_eviction":
            evicts[key] = d.time

    intervals: dict[str, list[tuple[float, float, str, float]]] = {}
    for (mem, w, t), (start, num_bytes) in stores.items():
        end = evicts.get((mem, w, t), makespan)
        label = _sequence_id(w, t) or f"w{w}t{t}"
        intervals.setdefault(mem, []).append((start, max(start, end), label, num_bytes))
    return intervals


def _first_tier_content_at(
    jobs: Sequence[JobRecord], attached: set[str], t: float
) -> tuple[dict[str, float], dict[int, float]]:
    """Reserved first-tier content at instant ``t`` across ``attached`` devices.

    Returns ``(weights_by_model, kv_by_batch)`` summed over the jobs active at
    ``t`` on the attached devices, splitting each job's per-device reservation
    into its weight portion (keyed by model) and its KV portion (keyed by the
    dispatch ``batch_index`` of the task holding it, so co-resident parallel
    tasks stay separable).
    """

    weights: dict[str, float] = {}
    kv_by_batch: dict[int, float] = {}
    for j in jobs:
        if not (j.start <= t < j.end):
            continue
        count = sum(1 for d in j.devices if d in attached)
        if count == 0:
            continue
        weights[j.model] = weights.get(j.model, 0.0) + j.weight_bytes_per_device * count
        kv_by_batch[j.batch_index] = (
            kv_by_batch.get(j.batch_index, 0.0) + j.kv_bytes_per_device * count
        )
    return weights, kv_by_batch


def memory_timeline(
    result: RunResult, num_buckets: int = 64, *,
    progress: Callable[[float, float], None] | None = None,
) -> list[dict[str, Any]]:
    """Per-memory bandwidth, occupancy-by-content and incoming transfer over time.

    One row per memory per time bucket. Carries the achieved bandwidth (bytes/s)
    in the bucket, the occupancy split by content (``content``: on a first-tier
    memory a ``weights`` band plus one ``KV B<n>`` band per co-resident dispatch
    batch; on a floating memory the offloaded ``KV``), a discrete label for
    the dominant incoming transfer (its source and object), and the object most
    recently *removed* from the memory (``eviction_object`` -- see
    :func:`_assign_eviction_objects`), so the visualization can stack occupancy
    by content, plot bandwidth against the memory's static ceiling and show what
    was last evicted.
    """

    makespan = result.makespan or 0.0
    if makespan <= 0 or num_buckets < 1:
        return []
    width = makespan / num_buckets
    rescaled = _rescaled(result.events)
    seq_by_request = _sequence_by_request(result.records)
    first_tier_of = {d.name: d.first_tier_memory for d in result.device_specs}
    residency = _floating_kv_residency(result)

    events_by_memory: dict[str, list[EventRecord]] = {}
    for e in rescaled:
        # Charge a transfer's bandwidth against both ends (source ``memory`` and
        # ``destination_memory``); a compute read has only its source memory.
        if e.memory:
            events_by_memory.setdefault(e.memory, []).append(e)
        if e.destination_memory and e.destination_memory != e.memory:
            events_by_memory.setdefault(e.destination_memory, []).append(e)
    events_by_device: dict[str, list[EventRecord]] = {}
    for e in rescaled:
        if e.device:
            events_by_device.setdefault(e.device, []).append(e)
    # Jobs resident on each device, so the per-(memory, bucket) first-tier content
    # helper scans only the handful of jobs on the memory's attached devices
    # rather than every job in the run.
    jobs_by_device: dict[str, list[JobRecord]] = {}
    for j in result.jobs:
        for d in j.devices:
            jobs_by_device.setdefault(d, []).append(j)
    # Per-memory attached-device set and the jobs scoped to those devices,
    # deduplicated by identity (a job spanning several attached devices appears
    # once; ``_first_tier_content_at`` still counts its per-device shares).
    attached_by_memory: dict[str, set[str]] = {}
    jobs_by_memory: dict[str, list[JobRecord]] = {}
    for mem in result.memories:
        attached = set(mem.attached_devices)
        attached_by_memory[mem.name] = attached
        if mem.role == "first_tier":
            seen: dict[int, JobRecord] = {}
            for d in attached:
                for j in jobs_by_device.get(d, []):
                    seen.setdefault(id(j), j)
            jobs_by_memory[mem.name] = list(seen.values())

    rows: list[dict[str, Any]] = []
    for b in range(num_buckets):
        t0 = b * width
        t1 = (b + 1) * width
        for mem in result.memories:
            attached = attached_by_memory[mem.name]
            mem_events = [e for e in events_by_memory.get(mem.name, [])
                          if e.end > t0 and e.start < t1]
            bucket_bytes = sum(e.bytes_read * _overlap_fraction(e, t0, t1)
                               for e in mem_events)
            bucket_bw_s = sum(e.bandwidth_time * _overlap_fraction(e, t0, t1)
                              for e in mem_events)

            content: dict[str, float] = {}
            if mem.role == "first_tier":
                weights, kv_by_batch = _first_tier_content_at(
                    jobs_by_memory[mem.name], attached, t0
                )
                weight_bytes = sum(weights.values())
                if weight_bytes > 0:
                    content["weights"] = weight_bytes
                for batch in sorted(kv_by_batch):
                    if kv_by_batch[batch] > 0:
                        content[f"KV B{batch}"] = kv_by_batch[batch]
            elif mem.role in ("node", "second_tier"):
                kv_bytes = sum(
                    num_bytes
                    for start, end, label, num_bytes in residency.get(mem.name, [])
                    if start <= t0 < end
                )
                if kv_bytes > 0:
                    content["KV"] = kv_bytes
            occupancy = sum(content.values())

            # Incoming transfer: into a first-tier memory it is the dominant
            # transfer on its attached devices (the source is the event's
            # memory); into a floating memory it is the dominant transfer that
            # streamed *to* it (the source is the moving device).
            source = ""
            obj = ""
            if mem.role == "first_tier":
                dev_events = [e for d in attached
                              for e in events_by_device.get(d, [])
                              if e.end > t0 and e.start < t1]
                dom = _dominant_transfer(dev_events, t0, t1)
                if dom is not None:
                    source = dom.memory
                    obj = _transfer_object_label(dom, seq_by_request)
            else:
                dom = _dominant_transfer(mem_events, t0, t1)
                if dom is not None:
                    source = dom.device
                    obj = _transfer_object_label(dom, seq_by_request)

            rows.append({
                "bucket": b,
                "time_start": t0,
                "time_end": t1,
                "memory": mem.name,
                "role": mem.role,
                "node": mem.node,
                "bandwidth_bytes_per_s": bucket_bytes / width if width else 0.0,
                "bandwidth_seconds": bucket_bw_s,
                "bandwidth_util": bucket_bw_s / width if width else 0.0,
                "occupancy_bytes": occupancy,
                "content": content,
                "transfer_source": source,
                "transfer_object": obj,
                "eviction_object": "",
            })
        if progress is not None:
            progress(b + 1, num_buckets)
    _assign_eviction_objects(rows, residency, makespan)
    return rows


def _assign_eviction_objects(
    rows: list[dict[str, Any]],
    residency: Mapping[str, Sequence[tuple[float, float, str, float]]],
    makespan: float,
) -> None:
    """Fill each memory row's ``eviction_object`` with the last band it lost.

    For a first-tier memory the evictable objects are the occupancy bands the
    capacity graph shows: a band present in one bucket and gone the next (a
    completed batch's ``KV B<n>`` or displaced ``weights``) is a removal, labelled
    by that band key so it keeps the colour it had in the capacity stack. A
    floating memory aggregates every offloaded sequence under one ``KV`` band, so
    its per-sequence evictions come from the residency intervals instead: an
    interval that ends before the makespan is an eviction of ``kv:<sequence>``.
    Each row then holds the most recent removal at or before its bucket end (a
    step function), or empty before the first removal.
    """

    by_memory: dict[str, list[dict[str, Any]]] = {}
    role_of: dict[str, str] = {}
    for r in rows:
        by_memory.setdefault(r["memory"], []).append(r)
        role_of[r["memory"]] = r["role"]

    for mem, mrows in by_memory.items():
        mrows.sort(key=lambda r: r["bucket"])
        events: list[tuple[float, str]] = []
        if role_of[mem] == "first_tier":
            prev: set[str] | None = None
            for r in mrows:
                cur = set(r["content"])
                if prev is not None:
                    removed = prev - cur
                    if removed:
                        events.append((r["time_start"], max(removed)))
                prev = cur
        else:
            for start, end, label, _bytes in residency.get(mem, []):
                if end < makespan:
                    events.append((end, f"kv:{label}"))
        if not events:
            continue
        events.sort(key=lambda e: e[0])
        times = [t for t, _ in events]
        labels = [lbl for _, lbl in events]
        for r in mrows:
            idx = bisect.bisect_right(times, r["time_end"]) - 1
            if idx >= 0:
                r["eviction_object"] = labels[idx]



# State of a workload's current turn at an instant, in lifecycle order.
WORKLOAD_STATES = (
    "not_arrived", "in_queue", "kv_fetch", "prefill", "decode", "done",
)


def _events_by_request(events: Sequence[EventRecord]) -> dict[int, list[EventRecord]]:
    by_request: dict[int, list[EventRecord]] = {}
    for e in events:
        for rid in e.request_ids:
            by_request.setdefault(rid, []).append(e)
    return by_request


def _turn_state_at(
    record: RequestRecord, events: Sequence[EventRecord], t: float
) -> tuple[str, str]:
    """The (state, device) of a single turn at instant ``t``.

    State follows the turn lifecycle: not-arrived -> in-queue -> (KV fetch ->)
    prefill -> decode -> done. While dispatched, the active event phase covering
    ``t`` decides the sub-state; between phases it falls back to prefill before
    first token and decode after. ``device`` is the representative device serving
    the turn at ``t`` (the full engine group is reported separately).
    """

    if t < record.arrival_time:
        return "not_arrived", ""
    if t < record.dispatch_time:
        return "in_queue", ""
    if t >= record.completion_time:
        return "done", ""
    covering = [e for e in events if e.start <= t < e.end]
    decode = next((e for e in covering if e.phase == "decode"), None)
    if decode is not None:
        return "decode", decode.device
    prefill = next((e for e in covering if e.phase == "prefill"), None)
    if prefill is not None:
        return "prefill", prefill.device
    transfer = next((e for e in covering if e.phase in _TRANSFER_PHASES), None)
    if transfer is not None:
        return "kv_fetch", transfer.device
    ft = record.first_token_time
    if ft is not None and t >= ft:
        return "decode", ""
    return "prefill", ""


def _turn_batch_at(
    events: Sequence[EventRecord], t: float
) -> int | None:
    """The id of the batch actively computing this turn at instant ``t``.

    Returns the ``batch_index`` of the event covering ``t`` (compute phases take
    precedence over transfers), or ``None`` when no event covers ``t`` -- i.e.
    the sequence is not currently being executed by any batch.
    """

    covering = [e for e in events if e.start <= t < e.end]
    compute = next((e for e in covering if e.phase in _COMPUTE_PHASES), None)
    if compute is not None:
        return compute.batch_index
    transfer = next((e for e in covering if e.phase in _TRANSFER_PHASES), None)
    if transfer is not None:
        return transfer.batch_index
    return None


def _per_tenant_decode_tps(
    result: RunResult, events: Sequence[EventRecord]
) -> dict[int, float]:
    """Per-tenant decode token throughput for every request, keyed by id.

    A decode batch produces all its tenants' tokens together over one wall-clock
    window (``[min start, max end]`` of the job's decode events); the per-tenant
    rate the PRD's "Current TPS" graph plots is that batch rate divided by the
    batch size, so every sequence in the batch shares the same value. Sequences
    that never decode are absent (the caller treats them as ``0``).
    """

    jobs_by_index = {j.job_index: j for j in result.jobs}
    out_tokens = {r.request_id: r.output_tokens for r in result.records}
    spans: dict[int, list[float]] = {}
    for e in events:
        if e.phase != "decode":
            continue
        span = spans.get(e.job_index)
        if span is None:
            spans[e.job_index] = [e.start, e.end]
        else:
            span[0] = min(span[0], e.start)
            span[1] = max(span[1], e.end)

    per_tenant: dict[int, float] = {}
    for job_index, (start, end) in spans.items():
        job = jobs_by_index.get(job_index)
        if job is None or end <= start:
            continue
        batch_size = len(job.request_ids)
        if batch_size <= 0:
            continue
        total = sum(out_tokens.get(rid, 0) for rid in job.request_ids)
        if total <= 0:
            continue
        rate = (total / (end - start)) / batch_size
        for rid in job.request_ids:
            per_tenant[rid] = rate
    return per_tenant


def _workload_key(record: RequestRecord) -> tuple[int, str]:
    """A stable (sort-key, label) for a record's workload (or itself if standalone)."""

    if record.workload_id >= 0:
        return record.workload_id, f"w{record.workload_id}"
    return 1_000_000 + record.request_id, f"r{record.request_id}"


def workload_timeline(result: RunResult, num_buckets: int = 64) -> list[dict[str, Any]]:
    """Per-workload current-turn, serving device and lifecycle state over time.

    One row per workload per time bucket. A workload is a multi-turn conversation
    (or a standalone request); at each instant exactly one turn is current. The
    row reports that turn's index, its serving device, the full engine group it
    runs on (a stable ``group`` id plus the ``devices`` list), the id of the
    batch currently executing it (``batch``, ``None`` when not being computed),
    its lifecycle state (not-arrived / in-queue / KV-fetch / prefill / decode /
    done) and, while actively decoding, the per-tenant decode throughput
    (``decode_tps`` -- the batch's token rate divided by its batch size; ``0``
    otherwise).
    """

    makespan = result.makespan or 0.0
    if makespan <= 0 or num_buckets < 1:
        return []
    width = makespan / num_buckets
    rescaled = _rescaled(result.events)
    events_by_request = _events_by_request(rescaled)
    per_tenant_decode_tps = _per_tenant_decode_tps(result, rescaled)

    workloads: dict[str, list[RequestRecord]] = {}
    labels: dict[str, int] = {}
    for r in result.records:
        sort_key, label = _workload_key(r)
        workloads.setdefault(label, []).append(r)
        labels[label] = sort_key
    for turns in workloads.values():
        turns.sort(key=lambda r: r.turn_index)

    group_ids: dict[tuple[str, ...], str] = {}

    def _group_id(devices: tuple[str, ...]) -> str:
        if not devices:
            return ""
        if devices not in group_ids:
            group_ids[devices] = f"G{len(group_ids)}"
        return group_ids[devices]

    def _slot(request_id: int) -> tuple[str, ...]:
        evs = events_by_request.get(request_id, [])
        return tuple(sorted({e.device for e in evs if e.device}))

    # A turn runs on one engine replica (a fixed slot of devices); the slot is the
    # union of every device that serves the turn across its lifetime.
    slots: dict[int, tuple[str, ...]] = {}

    _ACTIVE = {"kv_fetch", "prefill", "decode"}

    rows: list[dict[str, Any]] = []
    for b in range(num_buckets):
        t0 = b * width
        for label in sorted(workloads, key=lambda x: labels[x]):
            turns = workloads[label]
            current = _current_turn(turns, t0)
            events = events_by_request.get(current.request_id, [])
            state, device = _turn_state_at(current, events, t0)
            if state in _ACTIVE:
                devices = slots.setdefault(
                    current.request_id, _slot(current.request_id)
                )
                batch = _turn_batch_at(events, t0)
            else:
                devices = ()
                batch = None
            rows.append({
                "bucket": b,
                "time_start": t0,
                "time_end": (b + 1) * width,
                "workload": label,
                "turn": current.turn_index,
                "sequence": _sequence_id(current.workload_id, current.turn_index)
                            or label,
                "state": state,
                "device": device,
                "group": _group_id(devices),
                "devices": list(devices),
                "batch": batch,
                "decode_tps": (per_tenant_decode_tps.get(current.request_id, 0.0)
                               if state == "decode" else 0.0),
            })
    return rows


def _current_turn(turns: Sequence[RequestRecord], t: float) -> RequestRecord:
    """The turn of a workload that is current at instant ``t``.

    Turns are serialized, so at most one covers ``[arrival, completion)``; before
    the first arrival the first turn is current (not-arrived) and after the last
    completion the last turn is current (done).
    """

    for r in turns:
        if r.arrival_time <= t < r.completion_time:
            return r
    if t < turns[0].arrival_time:
        return turns[0]
    return turns[-1]


def workload_graph(result: RunResult) -> dict[str, Any]:
    """Node-link model of the run's workloads for the workload tab.

    Each turn contributes an *input* (prefill) node -- spanning queuing to its
    first output token -- and an *output* (decode) node -- spanning the first to
    the last output token. A *tool-call* node fills the client-side gap between a
    turn's completion and the next turn's arrival. Edges run from the sequence
    whose cached KV prefix was reused to the sequence that reused it; reuse within
    one conversation is implied and left undrawn, so only cross-workload links
    appear. A cross-conversation link is anchored at the source's turn-0 input
    node (the leading-message prefix it already holds from its start) and at the
    turn where the reuse actually fired in the reusing conversation, and is drawn
    once at its earliest occurrence -- never earlier than the borrow, so arrows
    never point backwards in time. Workloads occupy stacked horizontal lanes
    ordered by arrival, and each node carries logical ``t0``/``t1`` (time) and
    ``lane``/``sub`` (vertical) coordinates the renderer maps to pixels.
    """

    makespan = result.makespan or 0.0

    workloads: dict[str, list[RequestRecord]] = {}
    keys: dict[str, int] = {}
    for r in result.records:
        sort_key, label = _workload_key(r)
        workloads.setdefault(label, []).append(r)
        keys[label] = sort_key
    for turns in workloads.values():
        turns.sort(key=lambda r: r.turn_index)
    order = sorted(workloads, key=lambda x: keys[x])
    lane_of = {label: i for i, label in enumerate(order)}

    # Stable engine-group id per turn (the union of devices that served it).
    events_by_request = _events_by_request(_rescaled(result.events))
    group_ids: dict[tuple[str, ...], str] = {}

    def _group(request_id: int) -> str:
        devices = tuple(sorted({
            e.device for e in events_by_request.get(request_id, []) if e.device
        }))
        if not devices:
            return ""
        if devices not in group_ids:
            group_ids[devices] = f"G{len(group_ids)}"
        return group_ids[devices]

    nodes: list[dict[str, Any]] = []
    node_by_request: dict[int, str] = {}        # request id -> its input node id
    node_by_seq: dict[tuple[int, int], str] = {}  # (workload, turn) -> input node id

    for label in order:
        lane = lane_of[label]
        turns = workloads[label]
        for i, r in enumerate(turns):
            seq = _sequence_id(r.workload_id, r.turn_index) or f"{label}t{r.turn_index}"
            group = _group(r.request_id)
            first_token = (r.first_token_time if r.first_token_time is not None
                           else r.completion_time)
            in_id = f"{label}:t{r.turn_index}:in"
            out_id = f"{label}:t{r.turn_index}:out"
            node_by_request[r.request_id] = in_id
            if r.workload_id >= 0:
                node_by_seq[(r.workload_id, r.turn_index)] = in_id
            nodes.append({
                "id": in_id, "kind": "prefill", "workload": label, "lane": lane,
                "sub": 0, "t0": r.arrival_time, "t1": first_token,
                "turn": r.turn_index, "tokens": r.prompt_tokens, "group": group,
                "ttft_s": r.ttft, "tps": None, "text": f"T{r.turn_index} in",
                "sequence": seq, "desc": "",
            })
            nodes.append({
                "id": out_id, "kind": "decode", "workload": label, "lane": lane,
                "sub": 1, "t0": first_token, "t1": r.completion_time,
                "turn": r.turn_index, "tokens": r.output_tokens, "group": group,
                "ttft_s": None, "tps": r.tps, "text": f"T{r.turn_index} out",
                "sequence": seq, "desc": "",
            })
            if i + 1 < len(turns):
                nxt = turns[i + 1]
                if nxt.arrival_time > r.completion_time + 1e-12:
                    nodes.append({
                        "id": f"{label}:t{r.turn_index}:tool", "kind": "tool",
                        "workload": label, "lane": lane, "sub": 1,
                        "t0": r.completion_time, "t1": nxt.arrival_time,
                        "turn": r.turn_index, "tokens": 0, "group": "",
                        "ttft_s": None, "tps": None, "text": "tool",
                        "sequence": seq,
                        "desc": f"Tool call before turn {nxt.turn_index}",
                    })

    # Edges: KV-prefix reuse, directed existing-KV -> reusing sequence. Within one
    # conversation reuse is implied, so only cross-workload links are drawn.
    seen: set[tuple[int, int]] = set()
    edges: list[dict[str, str]] = []
    for d in result.decisions:
        if d.kind != "kv_reuse":
            continue
        if d.source_workload_id is None or d.source_workload_id < 0:
            continue
        if d.source_workload_id == d.workload_id:
            continue
        # The borrow can only happen once the source has run, so anchor the link
        # at the turn where the reuse actually fired (never earlier, or the arrow
        # would point backwards in time). The source supplies a leading-message
        # prefix it already holds from its first turn, so anchor that end at the
        # source's turn 0. Decisions are chronological, so keeping the first per
        # conversation pair draws the link once, at its earliest occurrence.
        src = (node_by_seq.get((d.source_workload_id, 0))
               or node_by_seq.get((d.source_workload_id, d.source_turn_index)))
        dst = node_by_request.get(d.request_id)
        if not src or not dst or src == dst:
            continue
        key = (d.source_workload_id, d.workload_id)
        if key in seen:
            continue
        seen.add(key)
        edges.append({"source": src, "target": dst})

    return {
        "nodes": nodes,
        "edges": edges,
        "num_lanes": len(order),
        "makespan_s": makespan,
    }


def _idle_wait(
    events: Sequence[EventRecord], dispatch: float, completion: float
) -> float:
    """Seconds a dispatched sequence spent with no active event before completing.

    The idle wait is the part of the ``[dispatch, completion]`` window not
    covered by any of the sequence's events (compute or transfer): time the
    sequence was admitted but sitting without active work (batching, contention,
    transfer queueing).
    """

    window = completion - dispatch
    if window <= 0:
        return 0.0
    spans = sorted(
        (max(e.start, dispatch), min(e.end, completion))
        for e in events
        if min(e.end, completion) > max(e.start, dispatch)
    )
    busy = 0.0
    if spans:
        cur_lo, cur_hi = spans[0]
        for lo, hi in spans[1:]:
            if lo > cur_hi:
                busy += cur_hi - cur_lo
                cur_lo, cur_hi = lo, hi
            else:
                cur_hi = max(cur_hi, hi)
        busy += cur_hi - cur_lo
    return max(0.0, window - busy)


def _comm_wait(
    events: Sequence[EventRecord], dispatch: float, completion: float
) -> float:
    """Seconds a dispatched sequence spent in parallelism comm collectives.

    The wall-clock time inside ``[dispatch, completion]`` covered by the
    sequence's communication events (tensor-parallel all-reduce, expert-parallel
    all-to-all, pipeline-parallel hand-off). Collectives are barriers replicated
    across every rank with identical spans, so the spans are merged to avoid
    double-counting the parallel ranks; sequential collectives add up.
    """

    spans = [
        (max(e.start, dispatch), min(e.end, completion))
        for e in events
        if e.phase in _COMM_PHASES
        and min(e.end, completion) > max(e.start, dispatch)
    ]
    return _union_length(spans)


def sequence_table(result: RunResult) -> list[dict[str, Any]]:
    """Per-sequence timings: one row per turn of every workload (or request).

    Each row carries the model that served the turn, the engine group(s) it
    executed on (a list when prefill and decode run on different groups, e.g.
    under prefill/decode disaggregation), the time in queue (arrival to
    dispatch), TTFT counted only over prefill (dispatch to first token), TPS
    counted only over decode, the total idle wait while dispatched, the time
    spent in parallelism communication collectives, the end-to-end latency
    (arrival to answer), and the effective TPS (output tokens over that
    end-to-end latency). Rows are ordered by workload then turn, with standalone
    requests last.
    """

    by_request = _events_by_request(_rescaled(result.events))

    def _key(r: RequestRecord) -> tuple[int, int, int]:
        if r.workload_id >= 0:
            return (0, r.workload_id, r.turn_index)
        return (1, r.request_id, 0)

    rows: list[dict[str, Any]] = []
    for r in sorted(result.records, key=_key):
        ttft_prefill = (r.first_token_time - r.dispatch_time
                        if r.first_token_time is not None else None)
        events = by_request.get(r.request_id, ())
        rows.append({
            "sequence": _sequence_id(r.workload_id, r.turn_index)
                        or f"r{r.request_id}",
            "model": _request_model(events),
            "engine_groups": _engine_groups(events),
            "queue_s": r.queue_delay,
            "ttft_prefill_s": ttft_prefill,
            "tps_tokens_per_s": r.tps,
            "idle_wait_s": _idle_wait(
                events, r.dispatch_time, r.completion_time),
            "comm_wait_s": _comm_wait(
                events, r.dispatch_time, r.completion_time),
            "latency_s": r.latency,
            "effective_tps_tokens_per_s": (
                r.output_tokens / r.latency if r.latency > 0 else None),
        })
    return rows


def _request_model(events: Sequence[EventRecord]) -> str:
    """Name of the model serving a request (space-joined if it spans several)."""

    seen: list[str] = []
    for e in events:
        if e.model and e.model not in seen:
            seen.append(e.model)
    return " ".join(seen)


def _engine_groups(events: Sequence[EventRecord]) -> list[str]:
    """Engine group(s) a request executed on, in prefill-then-decode order.

    Each group is abbreviated as ``{first device} + {count of others}`` over the
    set of devices that ran one compute phase. Prefill and decode collapse to a
    single entry when they share devices, and yield two entries when they run on
    disjoint groups (prefill/decode disaggregation).
    """

    groups: list[tuple[str, ...]] = []
    for phase in _COMPUTE_PHASES:
        devices = tuple(sorted(
            {e.device for e in events if e.phase == phase and e.device}))
        if devices and devices not in groups:
            groups.append(devices)
    return [_abbreviate_devices(g) for g in groups]


def _abbreviate_devices(devices: Sequence[str]) -> str:
    """``{first device} + {count of others}`` (just the device when alone)."""

    if not devices:
        return ""
    if len(devices) == 1:
        return devices[0]
    return f"{devices[0]} + {len(devices) - 1}"


def summarize(
    result: RunResult, *, progress: Callable[[float, float], None] | None = None
) -> dict[str, Any]:
    """Aggregate run report over the whole suite."""

    records = result.records
    makespan = result.makespan or 0.0
    isolated = _isolated(result.events)
    transfers = [e for e in isolated
                 if e.phase in ("transfer", "weight_transfer", "expert_transfer")
                 and e.bytes_read > 0]
    devices = device_summaries(result, progress=progress)
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
        "avg_latency_s": (sum(r.latency for r in records) / len(records)
                          if records else 0.0),
        "avg_queue_delay_s": (sum(r.queue_delay for r in records) / len(records)
                              if records else 0.0),
        "ttft_s": _distribution(r.ttft for r in records),
        "tpot_s": _distribution(r.tpot for r in records),
        "tps_tokens_per_s": _distribution(r.tps for r in records),
        "num_decisions": len(result.decisions),
        "decision_counts": _decision_counts(result.decisions),
    }


# --- writing --------------------------------------------------------------------


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]],
    *, progress: Callable[[float, float], None] | None = None, total: int = 0,
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames),
                                extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            writer.writerow(row)
            if progress is not None and total and (i % 8192 == 0 or i == total):
                progress(i, total)


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
    "model", "devices", "batch_index", "tokens", "bytes_moved", "source_sequence",
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
            "bytes_moved": d.bytes_moved,
            "source_sequence": _sequence_id(d.source_workload_id, d.source_turn_index),
            "source_request_id": (
                "" if d.source_request_id is None else d.source_request_id
            ),
            "source_devices": " ".join(d.source_devices),
        })
    return rows


_EVENT_FIELDS = (
    "job_index", "batch_index", "job_phase", "request_ids", "group_index",
    "phase", "device", "memory", "model", "flops", "bytes_read", "compute_time",
    "bandwidth_time", "duration", "start", "end",
)


def _event_rows(events: Sequence[EventRecord]) -> Iterable[dict[str, Any]]:
    for e in events:
        row = asdict(e)
        row.pop("rescaled")
        row["request_ids"] = " ".join(str(r) for r in e.request_ids)
        yield row


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


class _Progress:
    """A single-line, throttled percentage indicator for a long epilogue step.

    ``update(done, total)`` advances the bar in place (overwriting the same line);
    ``done()`` finalizes it with the elapsed wall time and a newline. Everything
    is silent unless ``verbose`` is set. ``done``/``total`` may be floats so a
    composite step can report a continuous fraction.
    """

    def __init__(self, label: str, verbose: bool, *, min_interval: float = 0.2):
        self.label = label
        self.verbose = verbose
        self.min_interval = min_interval
        self.start = time.perf_counter()
        self._last_t = 0.0
        self._last_pct = -1

    def update(self, done: float, total: float) -> None:
        if not self.verbose or total <= 0:
            return
        pct = int(max(0.0, min(1.0, done / total)) * 100)
        now = time.perf_counter()
        if pct <= self._last_pct or (
            pct < 100 and now - self._last_t < self.min_interval
        ):
            return
        self._last_pct = pct
        self._last_t = now
        print(f"\r  [epilogue] {self.label}: {pct:3d}%",
              file=sys.stderr, end="", flush=True)

    def done(self) -> None:
        if not self.verbose:
            return
        elapsed = time.perf_counter() - self.start
        print(f"\r  [epilogue] {self.label}: 100% ({elapsed:.3f}s)",
              file=sys.stderr, flush=True)


def write_outputs(
    result: RunResult,
    out_dir: str | Path,
    *,
    run_id: str = "run",
    config: Mapping[str, Any] | None = None,
    time_buckets: int = 64,
    viz_buckets: int = 256,
    verbose: bool = False,
) -> Path:
    """Write all raw outputs for ``result`` under ``out_dir`` and return the path.

    When ``verbose`` is set, each derivation/write step that is expensive enough
    to matter shows a live progress percentage on stderr (so a slow epilogue can
    be attributed to a specific table and the process is visibly alive); cheap
    sub-second steps are written silently.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    overall = time.perf_counter()

    def _tracked(label: str, fn):
        """Run ``fn(progress)`` under a progress bar and return its result."""

        bar = _Progress(label, verbose)
        value = fn(bar.update)
        bar.done()
        return value

    # --- derivations (the heavy ones report progress; cheap ones stay silent) ---
    report = _tracked("summarize", lambda p: summarize(result, progress=p))
    devices = _tracked("device_summaries",
                       lambda p: device_summaries(result, progress=p))
    memories = memory_summaries(result)
    timeline = _tracked(
        f"device_timeline ({time_buckets} buckets)",
        lambda p: device_timeline(result, time_buckets, progress=p))
    mem_timeline = _tracked(
        f"memory_timeline ({time_buckets} buckets)",
        lambda p: memory_timeline(result, time_buckets, progress=p))
    work_timeline = workload_timeline(result, time_buckets)

    # --- writes (cheap tables written silently) ---------------------------------
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

    num_events = len(result.events)
    _tracked(
        "write events_before_rescaling.csv",
        lambda p: _write_csv(
            out / "events_before_rescaling.csv", _EVENT_FIELDS,
            _event_rows(_isolated(result.events)), progress=p, total=num_events))
    _tracked(
        "write events_after_rescaling.csv",
        lambda p: _write_csv(
            out / "events_after_rescaling.csv", _EVENT_FIELDS,
            _event_rows(_rescaled(result.events)), progress=p, total=num_events))

    _write_csv(
        out / "device_summary.csv",
        ["device", "node", "peak_flops_fp16", "first_tier_memory",
         "first_tier_capacity_bytes", "first_tier_bandwidth_bytes_per_s",
         "busy_fraction", "compute_util", "bandwidth_util",
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
         "memory_bytes", "content_json", "batch_size", "resident_tasks",
         "running_sequences_json", "decode_tokens_per_s",
         "prefill_tokens_per_s", "compute_flops_per_s", "compute_seconds",
         "first_tier_bytes_per_s", "bandwidth_seconds", "transfer_source",
         "transfer_object", *(f"{state}_fraction" for state in DEVICE_STATES)],
        [{**row, "content_json": json.dumps(row["content"], sort_keys=True),
          "running_sequences_json": json.dumps(row.get("running_sequences", []))}
         for row in timeline],
    )
    _write_csv(
        out / "memory_timeline.csv",
        ["bucket", "time_start", "time_end", "memory", "role", "node",
         "bandwidth_bytes_per_s", "bandwidth_seconds", "bandwidth_util",
         "occupancy_bytes", "content_json", "transfer_source", "transfer_object",
         "eviction_object"],
        [{**row, "content_json": json.dumps(row["content"], sort_keys=True)}
         for row in mem_timeline],
    )
    _write_csv(
        out / "workload_timeline.csv",
        ["bucket", "time_start", "time_end", "workload", "turn", "sequence",
         "state", "device", "group", "devices_json", "decode_tps"],
        [{**row, "devices_json": json.dumps(row.get("devices", []))}
         for row in work_timeline],
    )

    payload = _tracked(
        f"build+write viz.json ({viz_buckets} buckets)",
        lambda p: build_viz_payload(
            result, run_id=run_id, num_buckets=viz_buckets, progress=p))
    with open(out / "viz.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    if config is not None:
        with open(out / "config.json", "w", encoding="utf-8") as handle:
            json.dump(dict(config), handle, indent=2)

    if verbose:
        print(f"  [epilogue] total: {time.perf_counter() - overall:.3f}s",
              file=sys.stderr, flush=True)
    return out


def build_viz_payload(
    result: RunResult, *, run_id: str = "run", num_buckets: int = 256,
    progress: Callable[[float, float], None] | None = None,
) -> dict[str, Any]:
    """Bundle every GUI-ready series into one JSON-serializable payload.

    The visualization tool consumes only this payload, so all derivation lives
    here (testable in Python) and the GUI is a pure renderer. Carries the run
    summary, the static device/memory specs and aggregates, and the bucketed
    device, memory and workload timelines.
    """

    def _sub(base: float, weight: float) -> Callable[[float, float], None] | None:
        # Scale a sub-step's own progress into ``[base, base + weight]`` of the
        # whole payload, so the caller sees one continuous percentage.
        if progress is None:
            return None
        return lambda done, total: progress(
            base + weight * (done / total if total else 1.0), 1.0)

    summary = summarize(result, progress=_sub(0.00, 0.35))
    devices = device_summaries(result, progress=_sub(0.35, 0.30))
    memories = memory_summaries(result)
    dev_timeline = device_timeline(result, num_buckets, progress=_sub(0.65, 0.23))
    mem_timeline = memory_timeline(result, num_buckets, progress=_sub(0.88, 0.10))
    work_timeline = workload_timeline(result, num_buckets)
    work_graph = workload_graph(result)
    if progress is not None:
        progress(1.0, 1.0)

    return {
        "run_id": run_id,
        "makespan_s": result.makespan or 0.0,
        "num_buckets": num_buckets,
        "summary": summary,
        "sequences": sequence_table(result),
        "devices": devices,
        "memories": memories,
        "device_timeline": dev_timeline,
        "memory_timeline": mem_timeline,
        "workload_timeline": work_timeline,
        "workload_graph": work_graph,
    }
