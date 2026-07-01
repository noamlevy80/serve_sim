# serve_sim arbiter (Src/serve_sim/arbiter.py)

- Decode emits ONE group per output token (shards.py `_emit_decode`), so a
  long-output job has tens of thousands of events.
- `_link_dependencies` (static, used by ResourceArbiter.run and
  IncrementalArbiter.admit_events) must keep only the FRONTIER of predecessors,
  not every earlier event. Full all-pairs preds is O(N^2) memory/CPU per job and
  caused a 20GB+ blowup (~180MB/sequence). Frontier reduction makes preds ~O(N*W)
  where W = barrier width (ep ranks). Verified: 32k events -> ~512k edges (avg 16)
  vs ~512M before.
- Frontier correctness: an event starts at the latest predecessor end, so a pred
  `p` ending at/before another pred `q`'s start is dominated by `q` (q depends on
  p). Always include the greatest-start pooled pred so the set is non-empty and
  covers the max-start branch. Reduced preds are always a subset of full preds.
- Arbiter retains ALL jobs/events (`self._jobs`) to build the final event log;
  that growth is linear and expected, not the bug.

# Transfer bandwidth = BOTH ends (2026-06-30)
- A transfer occupies the bandwidth of BOTH the source memory and the destination
  memory simultaneously. `ComputeEvent.destination_memory` (events.py) carries the
  write target; arbiter `_demands_for` adds a bandwidth `_Demand` per end via
  `_bandwidth_memories` (source = source_memory|second/first tier; dest =
  destination_memory|device.first_tier). Each demand work=bytes_read, rate=
  bytes_read/bandwidth_time (same effective rate both ends -> uncontended duration
  UNCHANGED; only multi-job contention sharing a dest memory changes). `_retime`
  copies destination_memory.
- KV offload `_admit_kv_move` now sets source=device.first_tier, dest=floating
  (was source=floating as a contention hack). Cerebras first-tier SRAM bw is huge
  so it adds ~0 contention; floating still binds.
- Report: EventRecord.destination_memory (orchestrator, set via
  `_event_destination_memory_name`); report.py memory_summaries + memory_timeline
  attribute an event's bytes/bandwidth to BOTH `memory` and `destination_memory`
  buckets (dedup if equal). Fixed floating/node memories under-reporting write bw.
- device_summary first_tier bandwidth still source-based (e.memory==first_tier),
  intentionally left unchanged.
- All 844 tests pass (test_integration live-download deselected).

# KV offload events were MISSING from log (2026-06-30, "many rats")
- BUG: `_collect_outputs` only records events for jobs in `job_meta` (the dispatch
  loop). `_admit_kv_move` (orchestrator ~2830) admits an arbiter job OUTSIDE the
  dispatch loop -> its transfer event was never collected. Floating node memory
  occupancy grew (rebuilt from kv_transfer DECISIONS via `_floating_kv_residency`)
  but bytes_moved/bandwidth stayed ~0. Class of bug: modeled move that bypasses the
  event log.
- FIX: `self._aux_jobs: list[(job_index, device, batch_index)]` populated in
  `_admit_kv_move`, reset at top of run()/_run_pdd(). `_collect_outputs` has an
  aux-job loop building EventRecords (job_phase="kv_offload"). Regression test:
  Tests/test_global_kv_cache.py::test_kv_offload_is_accounted_as_bandwidth_on_the_node_memory.
- Invariant: every byte landing on a node memory must be backed by a logged transfer
  event. Only `_admit_kv_move` among admit sites bypassed job_meta; all 3 of its
  callers now register aux jobs.
- (FIXED -- see next section) PDD prefill->decode KV handoff was a PURE CLOCK DELAY
  (orchestrator ~1437, `transfer_ready_time` = prefill_end + kv_transfer_duration):
  no arbiter event, no link/HBM contention, invisible in bandwidth reports. Same
  class as the offload bug.

# PDD handoff is now REAL physics (2026-06-30, user: "no event bypasses physics")
- FIXED the rat above. `_admit_pdd_handoff` admits the prefill->decode KV move as an
  arbiter aux job (phase="kv_transfer"): source=prefill slot device first_tier,
  dest=a decode engine first_tier. Gated decode on `arbiter.job_is_done(handoff)`
  (pending list now holds (req, handoff_job_index), not (req, ready_time)).
- `_aux_jobs` tuple grew to (job_index, device, batch_index, job_phase); collection
  loop + `_admit_kv_move` updated. Removed unused `transfer_ready_time`,
  `kv_transfer_duration` import, `prefill_rep`/`decode_rep`.
- CONTENTION GOTCHA: using one representative decode device (decode_rep) as dest for
  ALL handoffs created FALSE contention (two parallel sequences -> two transfers to
  the same memory -> 2x slower), broke test_pdd_pipelines_requests_in_parallel. FIX:
  round-robin dest across `self._decode_pool.slots` (`handoff_rr` counter) so parallel
  handoffs hit distinct decode engines, matching real decode placement. Uncontended
  duration unchanged (homogeneous devices) so all exact-timing PDD tests still pass.
- Regression test: Tests/test_outputs.py::test_pdd_handoff_is_a_bandwidth_accounted_transfer.
- All 846 tests pass (test_integration live-download deselected).

# Floating-memory occupancy for homed weights/experts (2026-06-30, "same as KV")
- BUG (occupancy, NOT bandwidth): node memories holding homed model shards showed
  peak_memory_bytes=0 / occupancy_fraction=0 in memory_summary.csv even though the
  shard (weights + the experts it later streams) sits in node RAM for the whole run.
  Root cause: report.py `_memory_peak_occupancy` only sums device-attached
  per_device_bytes; node memories have NO attached devices -> 0. Same shape as the
  floating-KV occupancy gap (KV was already reconstructed via `_floating_kv_residency`).
- BANDWIDTH WAS FINE here: weight staging (NVM->RAM->device) + expert streams already
  set BOTH memories and are logged (node bytes_moved>0, bandwidth_util~0.08). No
  KV-class bypass bug for experts/weights. Verified in run-...-nvl72-mm-b16.
- FIX (report.py): new `_floating_weight_residency(result)` scans rescaled
  phase=="weight_transfer" events whose destination_memory has role in
  ("node","second_tier") and bytes_read>0, dedup by (dst_memory, model) [stage1 is
  once per (model,node)], emits band [event.end, makespan) of bytes_read labeled by
  model. `_residency_peak(bands)` = peak simultaneous bytes via sweep over band
  endpoints. memory_summaries: peak_mem += _residency_peak(kv_res[mem]+weight_res[mem])
  (computed ONCE before the loop). memory_timeline node/second_tier branch adds a
  "weights" content band alongside KV. First_tier memories aren't in the residency
  dicts so the add is clean/additive.
- Feasible by construction: `_check_memory_capacity` already caps sum of home shards
  <= node RAM, so reconstructed occupancy <= capacity (e2e occupancy_fraction<=1 holds).
- FIXED (was: `_build_shared_expert_transfer_event` funneled ALL ranks' expert-stream
  bandwidth to ONE representative home node RAM). Now events.py
  `_build_expert_transfer_events(group, rank_misses, source_for, latency, phase, start)`
  spreads each ep rank's missed-expert bytes across its pp*tp devices, buckets by
  id(source memory) and emits ONE parallel transfer per distinct source node (dest =
  a representative device on that node). orchestrator passes `expert_sources` list
  (one MemoryDevice per slot device: homed -> node_of(d).node_memory; non-homed ->
  input_memory) to generator.run() alongside representative `expert_source` (kept for
  decision source_name + latency). Single-source/single-node collapses to one event
  (byte-identical, one _time_scale() draw) so old 1-node tests unchanged.
  Marker expansion in _collect_outputs made group-aware (carriers set + span per group)
  to avoid duplicate waiting markers when a group has multiple carrier events.
  Result on multi_model_nvl72: expert bytes spread evenly over all 36 node mems
  (~0.96 TB each vs old 11 TB on node#0); makespan 26.7s->10.7s, TTFT p99 20.7s->6.0s.
  Regression test: Tests/test_orchestrator.py::test_moe_expert_stream_spreads_across_home_nodes.
- Regression test: Tests/test_e2e.py::test_homed_weights_show_occupancy_on_the_floating_node_memory
  (auto_parallel.json homes tiny-moe -> node mem peak>0). All 853 tests pass.

# Performance (2026-06-27 optimization pass, zero behavior change, 819 tests pass)
- IncrementalArbiter.job_is_done is O(1): self._job_unfinished[] counter decremented
  in _complete via on_complete callback (was 349M genexpr iters rescanning all tasks).
- _Task/_Demand precompute thresholds: done_threshold=work*_REL_TOL,
  latency_threshold=max(latency,1)*_REL_TOL (killed 30M+ max() calls).
- _Task.open_demands + live_demands list: finished = open_demands==0 and latency_done
  (no all(d.done) scan); _count_sharers/_next_delta/_advance iterate live_demands only.
  _advance rebuilds live_demands only when a demand crosses done_threshold.
- _complete is a FIXPOINT (flushes zero-work successors transitively in one call);
  advance_to dropped its redundant top-of-loop _complete.
- report.py _state_seconds: O(I log I) sweep line (was O(points*intervals)).
- report.py device_timeline pre-groups jobs_by_device so per-(device,bucket) helpers
  (_occupancy_at/_resident_tasks_at/_first_tier_content_at) scan only that device's jobs.
  _first_tier_content_at now takes a jobs iterable (not RunResult).
- Result: large_model_expert_traffic 310.8s->76s (4x), small_kv 32.7->12.6s, pdd 24->10s.
- Remaining hotspots if more needed: JSON dump (~9s, viz.json), device/memory_timeline
  per-bucket event filtering O(buckets*events), csv writerow.

# Performance (2026-06-29 pass, cerebras_kimi26.json, tp=32 on 32-node rack)
- Symptom: slow + low CPU% = single-thread Python saturating ~1 core. Hotspot was
  topology linear scans, NOT the arbiter.
- system.py System (frozen dataclass): node_of + _node_index_of_memory were O(devices)
  identity scans (id()/any()), called millions of times. Memoized via lazy @property
  caches (_node_by_device {id(dev)->Node}, _node_index_by_memory {id(mem)->idx}) built
  with setdefault (first-owner-wins == original first-match). Frozen => write cache with
  object.__setattr__, read via self.__dict__.get("..._cache"). O(devices)->O(1).
- orchestrator._expert_fetch_latency: per-dispatch max() over slot.devices of
  link_between(...).latency_s. Terms are topology constants: in-node hop = f(device);
  streamed hop = f(source_mem, device). Memoized in __init__ dicts
  _in_node_fetch_latency[id(d)] and _streamed_fetch_latency[(id(src),id(d))].
  link_between calls 2.35M -> 36. _expert_fetch_latency cum 14.7s -> 1.3s.
- report.memory_timeline: scoped _first_tier_content_at to jobs_by_memory (union of
  jobs on attached devices, dedup by id) instead of result.jobs. Mirrors device_timeline.
  NOTE: zero benefit when tp spans all devices (every job touches every memory) but
  helps smaller-tp configs.
- Result: cerebras_kimi26 clean wall 94.5s -> 55s (1.7x). Byte-identical outputs
  (git stash baseline diff = 0). All 835 tests pass.
- Still-remaining: memory_timeline/device_timeline per-bucket event comprehensions +
  _first_tier_content_at scanning all active jobs per (bucket,mem) when tp=full-rack.
  A per-entity content-timeline sweep (step fn at job start/end) could fix it but risks
  float-boundary output drift; left alone to preserve byte-identical guarantee.
