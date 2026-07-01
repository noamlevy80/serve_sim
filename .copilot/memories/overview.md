# serve_sim orientation

- Read `ENGINEERING.md` (repo root) FIRST for the codebase map, change playbooks
  (models/devices/systems/orchestration/KV), config schema, outputs, gotchas.
- `Project.md` = PRD/behavioural spec; keep it in sync with code changes.
- Two scheduling paths in orchestrator.py: `run()` (non-PDD) and `_run_pdd()`.
  Orchestration changes usually go in BOTH.
- Decision kinds registered in report.py `_DECISION_KINDS`:
  weight_load, weight_eviction, prefill, kv_reuse, kv_transfer, decode, kv_eviction.
- Memory OOM: TWO checks. (1) per-batch fit: `_parallelism_for` raises ValueError
  if one batch's per-device footprint > device mem. (2) aggregate: `_check_memory_capacity`
  (end of `_collect_outputs`) raises `MemoryCapacityExceeded` if peak concurrent
  per_device_bytes on any device > first_tier(+second_tier) capacity. example_heavier
  is intentionally oversubscribed (~1.85x); example.json fails the per-batch check (pp=ep=1).
- DecisionRecord (frozen) has time_started/time_completed: execution window from
  rescaled events, attached by `_attach_decision_times` (end of `_collect_outputs`).
  prefill/decode/kv_transfer/weight_load matched by (batch_index, phase); evictions &
  unmatched fall back to decision time. CSV cols in report.py `_DECISION_FIELDS`.
- Parallelism: 3 axes. PP (pipeline, shards layers, relieves mem, no single-batch
  speedup), EP (expert, shards routed experts, ep x speedup, replicates dense/attn/KV),
  TP (tensor, shards EVERY tensor + KV by tp, tp x speedup, divides per-device footprint).
  Engine = pp*ep*tp devices. TP is a FIXED degree, NOT auto-searched (no comm cost modeled,
  so it would dominate EP); auto_parallelism re-factors only the pp*ep budget (= degree//tp)
  while tp rides along. `factorizations(degree)` still returns (pp,ep) pairs.
  Key signatures keyword-default tp=1: footprint(pp,ep,kv,tensor_parallel=1),
  estimate_time(ep,flops,bytes,tensor_parallel=1), plan(degree,...,tensor_parallel=1),
  EventGenerator(..., tensor_parallel=1), StrategyConfig.tensor_parallel, ParallelismChoice
  .tensor_parallel. EventGenerator._device_index(stage,ep_rank,tp_rank)=stage*ep*tp+ep_rank*tp+tp_rank.
  Two-tier expert movement requires tp==1 (and pp==1) -> NotImplementedError otherwise.
  Config key `tensor_parallel` parsed in runner._strategy_from_config.
- Per-device-type parallelism: optional config `parallelism` list -> StrategyConfig.parallelism
  tuple[ParallelismSection]. Each section has compute_device (matches ComputeDevice.device_key,
  set from node `device` key in system._build_node) + pp/ep/tp/auto. Simulator._build_groups()
  partitions devices into EngineGroup per section (one group/type, slots span ONE type only);
  no `parallelism` -> single legacy group over all devices. _select_group(model) load-balances
  across groups (NOT feasibility-aware: a model that can't fit a group's scheme crashes in
  _parallelism_for). PDD requires len(_groups)==1. Group threaded through _planner_for(model,device),
  _home_node_for(model,slot,group), _home_shards keyed (id(model),id(group)), _dispatch(...,group=).
  Cerebras 44GB SRAM can't host glm-5.2/deepseek-v4-pro/nemotron-3-ultra; fits gemma-4-31b,
  qwen3.6-27b, deepseek-v3.2/v4-flash. Tests: test_parallelism_sections.py.
- Global KV cache: kv_store.py `KVCacheManager`; floating memory = node memory only.
  KV offload spreads across node memories: `_node_with_room(num_bytes, key)` probes
  starting at `_home_index((workload_id,turn_index))` (explicit int mix, PYTHONHASHSEED-
  independent) and wraps; one sequence => one memory (never split). Honors capacity
  (probe others if home full), LRU-evicts only when whole pool full. Callers: `_store_on_node`,
  `_spill_evicted`. Tests: test_global_kv_cache.py spread/deterministic-home tests.
- KV/prefix sharing is MODEL-RESTRICTED, correctly: kv_store.lookup skips entries via
  `entry.model is not model` (identity). Robust because runner.py (~418) loads one
  LayeredModel object per name and shares it across all that model's requests (~348);
  batching also gates on `req.model is model`. Verified on run-...-nvl72-mm-b16: 305
  kv_reuse decisions + 68 workload-graph edges, ALL same-model (0 cross-model).
  Viz confusion: report.py workload_graph lanes interleave models by arrival, so a
  same-model reuse edge visually crosses other models' lanes. Fixed by adding `model`
  to each workload-graph node (report.py _model helper) + showing it in app.js lane
  label and hover tooltip. NOTE: existing runs' viz.json must be regenerated (re-run
  sim) to pick up the node `model` field.
- Run: `python run_sim.py Configs/<x>.json`. Test deselect:
  `pytest Tests/ -q --deselect Tests/test_integration.py::test_live_multi_turn_session_is_downloadable`.
