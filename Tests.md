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
- **Stage 6 — Devices & kernel launch:** compute/memory device JSON configs
  loaded into hardware objects, and kernel-launch work shards + latency events
  (one launch per group-stage, concurrent across expert-parallel ranks). 30
  offline tests.
- **Stage 7 — Event rescaling:** a resource arbiter co-runs several event
  generators on one timeline, dividing a loaded compute device or memory device
  equally and prorating in-flight compute and transfer events. 15 offline tests.
- **Stage 8 — Multi-turn conversations:** a conversation driver stitches a
  workload's turns end-to-end on one timeline and inserts tool-call wait events
  (scaled by `tool_calling_speedup`) between turns. 14 offline tests.
- **Stage 9 — System configs:** a `System` loader expands a `Systems/*.json`
  (network, input memory, nodes) into live hardware objects, instantiating each
  `count` device as a distinct instance with its own first-tier memory. 20
  offline tests.
- **Stage 10 — Test suites:** a randomized suite draws N random dataset
  workloads and binds each to a model chosen at random from a configured list,
  reproducibly from a seed. 15 offline tests.
- **Stage 11 — KV-cache tracking:** a per-layer `KVCacheTracker` records each
  token's generated state and device residency (by identity) and supports
  prefix links to another sequence; `SequenceTracker` finds the message-aligned
  common prefix and links its KV trackers for reuse. 24 offline tests.
- **Stage 12 — Model weights & transfers:** a `ModelWeightsTracker` decomposes a
  model into weight shards (per-layer attention/Mamba, dense FFN or per-expert +
  shared/latent for MoE, plus the LM head) and records their device residency by
  identity; a network-aware transfer cost model bounds bandwidth by the slower of
  the two memories and the link, and `System.link_between` classifies a pair of
  memories as intra-package, CXL or scale-up. 34 offline tests.
- **Stage 13 — Incremental arbiter:** an `IncrementalArbiter` drives the same
  fluid rescaling as the batch `ResourceArbiter`, but jobs are admitted at the
  current clock and the caller steps time forward; admitting all jobs at t=0 and
  running to idle reproduces the batch solver, while mid-flight admissions
  prorate in-flight events for the elapsed time. 13 offline tests.
- **Stage 14 — Orchestrator v0:** a `Simulator` runs an event-driven serving
  loop over arriving `Request`s, batching same-model requests under the strategy
  knobs (max batch size, window timeout, target concurrency) and dispatching each
  batch through the `IncrementalArbiter`; a single request reproduces its solo
  makespan, batching shares the device, and concurrency/window limits gate
  dispatch as configured. 18 offline tests.

Totals: 487 tests (482 offline + 5 live).

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

## Stage 6: Devices & kernel launch

Stage 6 loads hardware from JSON and models kernel-launch overhead.

### Device configs — [Tests/test_devices.py](Tests/test_devices.py)

Compute devices (`Compute_devices/*.json`) and memory devices
(`Memory_devices/*.json`) load into `ComputeDevice` / `MemoryDevice`. A compute
config names its first-tier memory by file stem; the loader resolves it from the
sibling `Memory_devices/` directory. A second tier is never part of a compute
config — it is a system-configuration choice attached at load time — so devices
load with no second tier unless one is supplied. Tests cover every shipped
config, dtype-scaled effective FLOPs, and that named first tiers exist.

### Kernel launch — [Tests/test_kernel_launch.py](Tests/test_kernel_launch.py)

The work-shard generator emits one zero-cost kernel-launch marker per
forward-pass group (with `phase == "kernel_launch"`, so the prefill/decode
filters used elsewhere are unaffected). The event generator charges a device its
`kernel_launch_latency` once per group-stage before that stage's compute:

- A zero-latency device emits no launch events and is byte-identical to the pure
  roofline (`reference_roofline`).
- A non-zero device adds exactly one latency per group: makespan equals the
  roofline plus `groups x kernel_launch_latency`.
- Under pipeline parallelism there is one launch per group per stage; under
  expert parallelism the ranks of a stage launch concurrently, so one launch per
  group. Both are checked by differencing against the zero-latency run.

## Stage 7: Event rescaling

A single event generator assumes it owns its devices. Stage 7 adds a
`ResourceArbiter` that co-runs several generators (jobs) on one shared timeline:
whenever concurrent events demand the same compute device or memory device, that
resource's rate is split equally and in-flight events are rescaled, prorated for
the elapsed time. The arbiter is a fluid (processor-sharing) co-simulation —
rates are recomputed each time an event starts or finishes a demand. Resources
contend only when the jobs were handed the *same* device/memory instance. 15
offline tests in [Tests/test_arbiter.py](Tests/test_arbiter.py).

- **Identity:** a single job, or jobs on disjoint resources, reproduce the
  standalone `EventGenerator.run` timings exactly.
- **Compute events:** two identical jobs sharing one device take twice as long
  in both the bandwidth-bound and compute-bound regimes; `N` jobs scale `N x`.
  Sharing only a memory device (distinct compute pools) does not slow
  compute-bound work.
- **Proration (PRD example):** with one shared memory and a second job that
  starts `tau` late, the closed form gives makespan `2W/R` and the early job's
  rescaled end at `2W/R - tau` — both matched exactly.
- **Transfer events:** two MoE jobs streaming experts from the same system NVM
  each see their transfer time double; separate NVMs do not contend.
- **Conservation:** rescaling changes only timings — total FLOPs/bytes are
  preserved and sharing is never faster than isolation.

## Stage 8: Multi-turn conversations

A workload is a multi-turn agentic conversation whose turns are separated by
tool-call waits — the client-side gap (`pre_gap`) spent waiting for a tool/function
call to return. `run_conversation` drives the existing single-turn pipeline
(tokenize a turn → work shards → timed compute events) once per turn and lays the
per-turn schedules out end-to-end on one clock, inserting a tool-call wait event
(`phase == "tool_call"`) in front of every turn with a positive `pre_gap`. We
model only the wait, not the tool computation, so those events carry no FLOPs or
bytes. 14 offline tests in [Tests/test_conversation.py](Tests/test_conversation.py).

- **Tool-call events:** exactly one wait per non-zero `pre_gap` (the first turn
  has none); each wait's duration equals its `pre_gap`, the total equals the sum
  of gaps, and the events carry no compute or bandwidth work.
- **`tool_calling_speedup`:** divides every wait (a 2x speedup halves total tool
  time) without changing any compute; a non-positive value is rejected.
- **Timeline:** the stitched events are contiguous and non-overlapping, a wait
  sits between the surrounding turns' compute, and the conversation makespan
  equals the sum of the independently-timed per-turn makespans plus the waits.
- **Tokenization:** each later turn caches exactly the previous turn's tokens, so
  the appended tool-call request and tool response become that turn's prefill;
  the compute events reproduce the single-turn runs, and a single-turn
  conversation has no waits.

## Stage 9: System configs

A system config (`Systems/*.json`) describes a whole machine: a scale-up + CXL
`Network`, a designated *input memory* (the shared NVM holding all weights at
init), and a list of nodes. `load_system` expands each node's
`{"device": ..., "count": N}` entries into N **distinct**
`ComputeDevice` instances — each with its own first-tier memory instance, because
the event generator and arbiter contend on object identity. 20 offline tests in
[Tests/test_system.py](Tests/test_system.py).

- **Loading:** every shipped system loads into live objects; the two-node B200
  system has 8 GPUs across two nodes, the heterogeneous system pairs a B200 node
  with a Cerebras WSE-3 node, and each device keeps its own first-tier memory.
- **Network/memory:** the network bandwidth/latency fields parse, the input
  memory resolves to the datacenter NVM, and each node's node memory resolves to
  the Grace LPDDR5X device.
- **Identity:** every compute device and every first-tier memory is a distinct
  instance; `node_of` finds a device's owning node and rejects foreign devices.
- **Validation:** `count` expands to distinct instances and defaults to one,
  omitted node memory is `None`, second tiers are never auto-attached, and zero
  counts / empty node lists / bad network parameters are rejected.

## Stage 10: Test suites

A suite is what the simulator runs: a list of `SuiteEntry` pairs, each binding a
multi-turn workload to the name of the model that serves it. A *randomized* suite
(`Suites/*.json`) draws `num_workloads` random sessions from the dataset and
assigns each a model chosen uniformly from a configured list. Dataset access is
injected through a `WorkloadLoader`, so the tests run offline over an in-memory
fetcher, and randomness comes from a caller-supplied seed. 15 offline tests in
[Tests/test_suite.py](Tests/test_suite.py).

- **Construction:** the suite has exactly `num_workloads` entries; every entry is
  a valid whole-session `Workload` bound to a model from the configured list.
- **Determinism:** a fixed seed reproduces the suite; different seeds (with
  several sessions and models) produce different draws.
- **Config dispatch:** the JSON parses its fields, defaults to the randomized
  type, and rejects the not-yet-implemented `directed` type and unknown types.
- **Validation:** zero workloads, an empty model list, and an empty suite are
  rejected; the shipped sample suite loads and builds.

## Stage 11: KV-cache tracking

A `KVCacheTracker` is the per-`(model, conversation, layer)` bookkeeping the PRD
calls for: per token index it records whether the KV is generated, which memory
devices hold it (by identity, so value-equal device instances stay distinct), and
an optional link to the same index in another tracker for cross-sequence prefix
reuse. A `SequenceTracker` built from a turn remembers its messages and can find
the token-length of the message-aligned common prefix with another sequence, then
link its per-layer KV trackers over that prefix. 24 offline tests in
[Tests/test_kv_cache.py](Tests/test_kv_cache.py).

- **Residency:** `place` marks a range generated and resident; a token may live
  on several devices; `evict` removes one and the last eviction leaves the token
  generated but unavailable; value-equal devices are distinct locations.
- **Prefix length:** `cached_prefix_length` counts the leading run of available
  tokens and stops at the first gap.
- **Linking:** a linked range delegates generated/residency to its source and
  follows the source's evictions; local tokens may be appended after the link;
  over-long links are rejected.
- **Sequence integration:** the common prefix is message-aligned and symmetric,
  zero when the first message differs, and requires trackers built from a turn;
  `link_kv_prefix` links every layer and requires matching layer counts.

## Stage 12: Model weights & transfers

The static counterpart of KV tracking. A `ModelWeightsTracker` decomposes a model
into **weight shards** — per layer the attention or Mamba matrices and either a
dense FFN shard or one shard per routed expert (plus shared-expert and
latent-projection shards when present), and the global LM head — and records, per
shard, the memory devices it resides on (by identity, so value-equal memories
stay distinct). Weight shards are never "ungenerated": they exist from init,
typically parked on the system NVM, and are moved up to a device's first tier on
demand. The transfer cost model bounds a move's bandwidth by the slower of the
two memories and the connecting link, and adds the link's latency. 34 offline
tests in [Tests/test_weights.py](Tests/test_weights.py) and
[Tests/test_transfer.py](Tests/test_transfer.py).

- **Shard enumeration:** a dense model emits one attention + one FFN shard per
  layer plus a single LM head; total shard bytes match the block arithmetic. A
  MoE model expands each routed expert into its own shard and adds a shared-expert
  (and, for LatentMoE, a latent-projection) shard; dense layers in a hybrid stack
  emit a single FFN shard; a Mamba layer emits a Mamba shard; the LM head is
  omitted when disabled.
- **Residency:** `place`/`evict` track copies by identity, so two value-equal
  memories are distinct locations; `place_all` then `bytes_on` equals the total
  weight bytes; eviction removes only the named copy; foreign or fabricated
  shards are rejected; `shard_for` resolves a descriptor and flags missing or
  ambiguous lookups.
- **Transfer cost:** an intra-package access is bounded by the slower memory; a
  link can bound the transfer; latency is added on top of the byte time; zero
  bytes costs just the latency; negative bytes and non-positive link bandwidth /
  negative latency are rejected; `make_transfer_event` fills a `phase="transfer"`
  event.
- **Link classification:** `System.link_between` returns intra-package for the
  same device, CXL within a node (including node memory), and scale-up across
  nodes or to the system NVM, carrying the right bandwidth and latency.

## Stage 13: Incremental arbiter

The batch `ResourceArbiter` is a one-shot solve over a fixed job set that all
start at t=0. The orchestrator instead needs jobs that arrive and drain over
time while the same equal-share rescaling applies to whatever is in flight. An
`IncrementalArbiter` keeps the active events and clock as mutable state: the
caller `admit`s jobs at the current time and `advance_to`s later instants,
stopping exactly at each target so a new job can be injected there. The same
private fluid (processor-sharing) helpers are reused, so admitting every job at
t=0 and running to idle reproduces the batch solver event-for-event. 13 offline
tests in [Tests/test_incremental_arbiter.py](Tests/test_incremental_arbiter.py).

- **Batch equivalence:** a single job, two jobs sharing one compute device, two
  jobs sharing only a memory device, two MoE jobs sharing one NVM (transfer
  contention), and two jobs on disjoint devices each match the batch
  `ResourceArbiter` (or the standalone generator) event-for-event when admitted
  at t=0.
- **Mid-flight proration:** a job admitted at `tau` into another's single shared
  compute event splits the rate equally from `tau` on — the running job ends at
  `2T - tau` and the newcomer at `2T`, matching a closed form derived
  independently of the simulator. Admission on a disjoint resource leaves the
  running job untouched; a job admitted after the machine has drained starts at
  the (advanced) idle clock and runs at full rate.
- **Stepping API:** `next_event_time` reports the next completion (or `None`
  when idle), `advance_to` stops exactly at its target (clock jumps to the target
  even when the machine drains early), `is_idle`/`active_count`/`num_jobs` expose
  state, and total FLOPs/bytes are conserved while sharing is never faster than
  isolation.

## Stage 14: Orchestrator v0

The orchestrator closes the loop from a request stream to per-request timing. A
`Simulator` consumes `Request`s in arrival order and runs a strictly
event-driven loop: at each step it advances the clock to the nearest of the next
arrival, the open batch window's deadline, or the arbiter's next completion,
then retires finished jobs, admits new arrivals, and dispatches batches. A batch
groups consecutive same-model requests (by instance identity) up to the strategy
max batch size, dispatching when the batch fills or the concurrency window times
out; `target_concurrency` caps how many sequences may be in flight so excess
batches serialize behind completions. Each batch is turned into work shards and
events and admitted to the shared `IncrementalArbiter`, so concurrent batches
contend on the engine devices exactly as the fluid solver dictates. 18 offline
tests in [Tests/test_orchestrator.py](Tests/test_orchestrator.py).

- **Timing fidelity:** a lone request retires at its standalone solo makespan;
  a non-zero arrival time shifts completion by exactly that offset; strictly
  sequential arrivals never overlap. A floating-point regression where an
  arrival at a large absolute time (microsecond events at t=12.5) left an
  undrainable sub-ULP residual in `advance_to` is covered by these tests.
- **Batching triggers:** filling the batch dispatches all requests as one batch;
  a partial batch waits for the window timeout before dispatching; a batched
  decode is cheaper than two solo runs because the batch shares the device.
- **Concurrency control:** two concurrent batches share the device, while
  `target_concurrency` serializes batches behind completions and a concurrency
  of one with a single arrival reduces to the solo case.
- **Grouping:** different models dispatch in separate batches, the same model
  instance batches together, and the batch-size knob caps each group; every
  request retires exactly once.
- **Construction & validation:** `Request.from_workload` builds a request from a
  workload turn, and invalid strategy/request parameters or an engine that needs
  more devices than the system has are rejected; an empty run is empty.


