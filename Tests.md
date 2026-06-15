# Tests

Test suite for `serve_sim`, written with **pytest**. Source lives in `Src/`,
tests in `Tests/`. Configuration is in [pyproject.toml](pyproject.toml)
(`pythonpath = ["Src"]`, `testpaths = ["Tests"]`, and a `network` marker).

## Running

```pwsh
python -m pytest -m "not network"   # offline, deterministic (no network)
python -m pytest                     # everything, including live dataset calls
python -m pytest -m network          # only the live dataset integration tests
```

Network tests auto-skip if the dataset API is unreachable.

## Coverage summary

- **Stage 1 — Workloads:** workload model + dataset loader. 54 tests (49
  offline + 5 live).
- **Stage 2 — Event generation:** roofline simulation path (model sizing,
  devices, trackers, work shards, events). 75 offline tests.
- **Stage 3 — MoE:** expert-usage model, MoE sizing/shards, MoE roofline, and
  two-tier expert movement. 90 offline tests.
- **Stage 4 — Model mechanisms:** per-layer block model (GQA/MHA + sliding
  window, MLA + DSA, Mamba-2, dense/MoE incl. shared + LatentMoE) and the three
  reference model configs loaded from JSON and run end-to-end. 22 offline tests.
- **Stage 5 — Parallelism:** expert parallelism (distributed experts + even-split
  compute) composed with pipeline parallelism, and the two memory scenarios
  (every device holds the model; a shared system-NVM second tier). 18 offline
  tests.

Totals: 259 tests (254 offline + 5 live).

## Stage 1: Workloads

Stage 1 covers the **workload model** and the **loader** that downloads a
multi-turn session from the source dataset. 54 tests total: 49 offline + 5 live.

### Shared fixtures — [Tests/conftest.py](Tests/conftest.py)

- `make_row` / `make_session_rows` — build dataset rows matching the source
  schema, with the prefix-growth property (turn `N` messages extend turn `N-1`).
- `FakeRowFetcher` — in-memory stand-in for the row API; records every
  `(offset, length)` request so paging behaviour can be asserted.
- `fake_dataset` / `fake_fetcher` — three contiguous sessions (3, 1, 4 turns;
  8 rows) used across loader tests.

### Workload model — [Tests/test_workload.py](Tests/test_workload.py)

Message / tool-call parsing:
- Minimal message parsing; `role` is required (raises otherwise); `null`
  content is allowed.
- Tool-call parsing, tool-result messages (`tool_call_id`, `name`), and a
  tool call with a missing `function` block.
- Messages are hashable and value-comparable.

Turns and workload assembly:
- `Turn` length counts its messages.
- `build_workload_from_rows` sets session/model fields, turn indices, and
  per-turn `output_length` / `pre_gap`.
- Type coercion of `output_length` → `int` and `pre_gap` → `float`.
- Errors on empty rows and on rows spanning multiple sessions.

Workload behaviour:
- Construction requires at least one turn.
- Indexing and iteration over turns.
- `new_messages`: first turn returns the full prefix; later turns return only
  the appended delta; concatenating all deltas reconstructs the final history.
- `validate_prefix_growth`: passes on valid data; detects a shrinking turn and
  a divergent (mutated) prefix.

### Loader — [Tests/test_loader.py](Tests/test_loader.py)

Construction and totals:
- Rejects invalid `page_size` (`0` or `> 100`).
- `num_rows` reports the split size.

`iter_workloads` (grouping contiguous sessions):
- Groups three sessions correctly with the right turn counts.
- Handles a single session that spans multiple pages (forces re-fetching).
- Handles session boundaries that fall mid-page.
- `start` offset skips earlier rows.
- `load_first` returns the first workload.

`load_session_at` (expand a full session around any row):
- Parametrised over every row offset (`0–7`) with a large page size.
- Same offsets with `page_size=1` to stress left/right expansion.
- Middle-of-a-long-session lookup; turns are re-indexed from 0.
- Out-of-range offsets raise `IndexError`.
- Loaded workloads satisfy `validate_prefix_growth`.

`HttpRowFetcher` (no network, via a stubbed session):
- Builds the correct request URL and params, and returns parsed JSON.
- Works end-to-end as a `WorkloadLoader` backend.

### Live integration — [Tests/test_integration.py](Tests/test_integration.py)

Marked `network`; the module skips if the dataset API is unreachable.
- Split has a plausible row count (> 1000).
- First workload has a `session_id`, `model`, ≥ 1 turn, and a leading `system`
  message.
- First workload satisfies `validate_prefix_growth`.
- `load_session_at(0)` matches `load_first`.
- A session with ≥ 3 turns is downloadable whole, validates, and grows in
  message count across turns.

## Stage 2: Event generation (roofline)

Stage 2 turns one turn of a sequence (or a batch of sequences) into timed
compute events and checks the total simulated time against an **independent
closed-form roofline**. 75 offline tests.

### Roofline model (what the tests assume)

- Per work-shard: `compute_time = flops / (peak_fp16 * 2/dtype_bytes)` and
  `bandwidth_time = bytes / first_tier_bandwidth`; an event takes the `max`.
- Shards of one forward-pass group on the same device are consolidated (roofline
  `max` of summed FLOPs and summed bytes). Makespan is the sum of events
  (single batch → sequential).
- Prefill is emitted per sequence (optionally chunked); decode is batched
  lockstep, so a step reads the layer weights once regardless of batch size.

### Independent reference — [Tests/reference.py](Tests/reference.py)

`reference_roofline` re-derives the expected makespan from the model dimensions,
device specs and batch work, separately from the shard/event code, so the tests
verify the simulator rather than mirror it.

### Model sizing — [Tests/test_model.py](Tests/test_model.py)

- `toy_model` defaults (`head_dim`, `num_kv_heads`) and divisibility guard.
- Attention weight params for MHA and GQA; gated vs ungated FFN params.
- Layer params = attention + FFN; LM head params toggle; KV bytes per token.
- FLOPs/byte helpers and constructor validation.

### Devices — [Tests/test_hardware.py](Tests/test_hardware.py)

- `dtype_compute_scale` (fp32 0.5×, fp16 1×, fp8 2×, fp4 4×) and rejection of
  non-positive sizes.
- Memory/compute device validation; `effective_flops` scaling and exposed
  bandwidth.

### Trackers + tokenizer — [Tests/test_tracker.py](Tests/test_tracker.py)

- `SequenceWork.base_tokens` and validation.
- `SequenceTracker` derived indices (cache / prefill / decode) with and without
  a cached prefix; `cached ≤ prompt` guard; `to_work`.
- `from_turn` tokenizes a workload turn (per-message counts), caches the
  previous turn, and preserves prefix growth.
- `BatchTracker.work()` and empty-batch rejection.
- Real `TiktokenTokenizer` token counting (skipped if tiktoken is unavailable).

### Work shards — [Tests/test_shards.py](Tests/test_shards.py)

- Prefill emits one shard per layer per chunk; chunking creates ordered groups.
- Decode emits layers + LM head per step (and layers-only when the LM head is
  disabled); ragged decode lengths shrink the active batch.
- Decode weight bytes are amortized over the batch (only KV scales with `B`).
- Empty-batch and bad-chunk-size rejection; fully-cached sequences skip prefill.

### End-to-end roofline — [Tests/test_roofline.py](Tests/test_roofline.py)

- Single sequence matches the reference; makespan equals the per-phase sum and
  events are contiguous in time.
- Existing-cache (later-turn) and chunked-prefill cases match the reference.
- Matches the reference as **peak FLOPs**, **bandwidth**, **dtype**
  (param + KV), and **model dimensions** (layers, hidden, GQA, gating,
  intermediate, vocab, LM head) vary.
- Decoupled scaling: compute-bound time ∝ 1/peak and 1/dtype-scale;
  bandwidth-bound time ∝ 1/bandwidth.
- Batch of 4 matches the reference; batched decode amortizes weight reads (much
  cheaper than 4× serial); ragged batch matches the reference.
- Pipeline parallelism (PP=2): conserves total FLOPs/bytes; equals single-device
  latency when uniformly bandwidth-bound; never faster than single device for a
  single batch; places the LM head on the last stage.
- Event-generator validation: requires devices, device count divisible by the
  parallelism product, layers divisible by PP, and expert parallelism rejected.

## Stage 3: MoE

Stage 3 adds Mixture-of-Experts: a statistical expert-usage model, MoE-aware
sizing and shards, and a two-tier system where routed experts move from a second
memory tier into the first on demand. 90 offline tests.

### Expert usage — [Tests/test_experts.py](Tests/test_experts.py)

- `ExpertUsageModel` construction and `from_model`.
- Closed-form `expected_distinct`: monotone in tokens, bounded by `E` and by the
  number of picks; the consecutive (prefill) regime sees fewer distinct experts
  than the independent (decode) regime for the same token count; depends only on
  the persistence **mean**, not its variance.
- Persistence sampling is clamped to `≥ 1` and rounds a Gaussian about the mean.
- Monte-Carlo `sample_distinct` approximates the closed form in the independent
  regime (`rel=0.1`) and is in the right ballpark for consecutive (`rel=0.3`).

### MoE sizing + shards — [Tests/test_moe.py](Tests/test_moe.py)

- `toy_moe_model` builds an MoE model; `is_moe_layer` / `num_moe_layers` respect
  `num_dense_layers`.
- Routed/shared expert params and bytes; MoE layer weight params; MoE
  constructor validation.
- Shard generation: routed FFN FLOPs scale with `tokens · k_E`; routed bytes
  scale with the distinct active experts; shared-expert work is always present;
  dense layers in a hybrid stack still use the dense FFN cost.

### MoE roofline — [Tests/test_moe_roofline.py](Tests/test_moe_roofline.py)

`reference_roofline` is extended with the MoE FFN cost (routed + shared) using an
independent re-derivation of expected distinct experts. The full pipeline matches
it for single sequences, batches, ragged batches, and chunked prefill, and keeps
matching as MoE parameters (`num_experts`, `num_experts_per_token`,
`num_shared_experts`, persistence, dense-layer count) and the usual system
parameters vary.

### Two-tier residency — [Tests/test_tiering.py](Tests/test_tiering.py)

- `ExpertResidencyCache`: first access all-miss, reuse hits, LRU eviction order,
  rejection when the active set exceeds capacity, capacity validation.
- `build_activation_trace`: empty for dense models; group indices match the work
  shards (with and without chunking); seed-reproducible; active sets stay within
  `[0, E)` and within the per-group top-k bound; higher persistence yields fewer
  distinct experts per prefill group.
- `derive_expert_cache_capacity`: positive for a generous tier, grows with tier
  size, raises when the tier cannot hold the reserved bytes plus one expert, and
  rejects dense models.

### Two-tier roofline — [Tests/test_two_tier.py](Tests/test_two_tier.py)

`reference_two_tier` = compute roofline + an independent LRU replay of the same
activation trace. Tests assert:

- Total makespan matches the reference for single sequences, batches, chunked
  prefill, and a derived cache capacity.
- The total decomposes exactly into compute (= single-tier makespan) plus
  transfer time.
- Transfer time falls with higher expert persistence and with faster second-tier
  bandwidth, and never rises as capacity grows; with capacity `≥ E` each expert
  is loaded exactly once.
- Validation: capacity smaller than a group's active set raises; two-tier
  execution requires a capacity and rejects pipeline parallelism; a two-tier
  device with no expert trace falls back to the single-tier path.

## Stage 4: Model mechanisms

Stage 4 represents a model as an ordered list of heterogeneous per-layer
**blocks** (`LayeredModel`) so real architectures can be assembled from a JSON
config and run through the existing shard/event path. The flat `Model` converts
to a `LayeredModel`, so every earlier stage keeps passing unchanged. 22 offline
tests in [Tests/test_models.py](Tests/test_models.py).

### Independent reference — [Tests/reference.py](Tests/reference.py)

`reference_layered` re-derives the makespan from the block fields directly (not
by calling the block cost methods), covering GQA/MHA + sliding window, MLA + DSA,
Mamba-2, and dense/MoE (shared experts + LatentMoE). On homogeneous models it
agrees with the flat `reference_roofline`, anchoring the new path to the old one.

### Building blocks

- **Sliding window:** decode attention is capped to the window; within the
  window it is identical to full attention.
- **MLA:** caches a single compressed latent (`kv_lora_rank + qk_rope_head_dim`)
  rather than per-head K and V.
- **DSA:** a lightweight indexer scores the windowed candidate set and the main
  attention is capped to `sparse_topk`; at long context this is far cheaper than
  dense attention in both FLOPs and bytes, while within top-k it adds only the
  indexer term.
- **Mamba-2:** holds no KV cache and its per-step cost is independent of
  sequence length (fixed-size recurrent state).
- Synthetic mixed-block models (MLA+DSA, Mamba+LatentMoE+GQA) match
  `reference_layered` end-to-end.

### Reference models (loaded from `Models/*.json`)

- **Gemma-4-31B** — 60 composite GQA layers alternating sliding/full attention
  with dense gated FFN and tied embeddings. Loads, generates events matching the
  reference (plain and chunked), and emits an LM-head shard at decode.
- **DeepSeek-V3.2** — 61 layers (3 dense + 58 MoE) of MLA + DSA attention.
  Loads with the expected expert counts and matches the reference.
- **Nemotron-3-Ultra** — 108 standalone blocks interleaving Mamba, GQA
  attention, and LatentMoE FFNs. Loads with one mixer-or-FFN per layer and
  matches the reference.

(Tensor parallelism beyond the expert-parallel group is not modeled; MTP heads
are documented but not yet costed.)

## Stage 5: Parallelism

Stage 5 lays a batch's work onto a `pipeline_parallel x expert_parallel` device
grid. Pipeline stages of a group run sequentially (single batch); the
expert-parallel ranks of a stage run concurrently with the stage compute split
evenly across them, and each rank keeps its own LRU residency of the experts it
owns. 18 offline tests in [Tests/test_parallel.py](Tests/test_parallel.py).

### Independent reference — [Tests/reference.py](Tests/reference.py)

`reference_ep_transfer` re-derives expert-movement time under expert
parallelism: experts are partitioned by `e mod expert_parallel`, each rank
replays its own LRU residency, and a group's movement runs the ranks
concurrently (a private second tier charges the slowest rank; a shared second
tier also bounds the aggregate by its bandwidth). The compute side reuses the
Stage-2/4 references divided by `expert_parallel`.

### Expert-parallel compute

- Even-split compute conserves total FLOPs/bytes and gives an exact `1/EP`
  makespan (a balanced stage is `expert_parallel` times faster); verified
  against the single-device reference and across two and four devices.
- Applies to dense models too (the split is of the whole stage).
- Composes with pipeline parallelism: a `PP x EP` grid equals the `PP`-only
  makespan divided by `EP` and conserves work.
- Validation: the device count must be divisible by `PP x EP`; two-tier expert
  movement still requires `pipeline_parallel == 1`.

### Scenario A — every device holds the whole model

Single-tier devices: experts are all resident, so no transfer events are emitted
and the makespan is the distributed compute (`single / EP`). Splitting experts
across devices never needs more per-device residency capacity, and
`derive_expert_cache_capacity` leaves more room as `expert_parallel` grows.

### Scenario B — shared second tier (system NVM)

The devices share one second-tier `MemoryDevice` (same instance). Expert
movement matches `reference_ep_transfer`, and because a shared NVM funnels every
rank's loads through one pipe it is never faster than an equivalent per-device
tier. Both the flat toy MoE model and the real DeepSeek config (on a toy system)
are checked end-to-end.
