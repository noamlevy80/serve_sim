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

Totals: 129 tests (124 offline + 5 live).

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
