---
name: add-model
description: 'Add a new LLM architecture to the serve_sim framework. USE WHEN the user wants to add, define, or onboard a new model (dense, MoE, MLA/DeepSeek, GQA, Mamba/hybrid, linear-attention) into Models/ so it can be referenced by suites and run through the simulator. Covers the Models/*.json schema (global/blocks/layer_pattern), every supported block and attention/FFN type, validation, and wiring the model into a config/suite. DO NOT USE for editing hardware (Compute_devices/Memory_devices), systems, or run configs.'
---

# Adding a new model to serve_sim

A model is **one JSON file** in [Models/](../../../Models/), named by its stem
(e.g. `Models/kimi-k2.6.json` → referenced as `"kimi-k2.6"`). No code change is
needed for standard architectures — the parser in
[model_config.py](../../../Src/serve_sim/model_config.py) maps the JSON onto the
architecture primitives in [blocks.py](../../../Src/serve_sim/blocks.py).

## Workflow

The canonical way every existing model was added: **start from the HuggingFace
model root**, fetch its architecture config, and derive the schema from it.

1. **Get a pointer to the HF model root.** Ask the user for the HuggingFace repo
   if not given (e.g. `https://huggingface.co/moonshotai/Kimi-K2.6` or just the
   `org/model` id). Record it as the `source` field in the output JSON.
2. **Fetch the architecture config** from that root and analyze it. The
   authoritative file is `config.json`:
   - Raw URL: `https://huggingface.co/<org>/<model>/resolve/main/config.json`
     (fallback `.../raw/main/config.json`). Use the `fetch_webpage` tool.
   - If a field is missing or the architecture is non-standard, also check the
     model card README and `configuration_*.py` / `modeling_*.py` in the repo.
3. **Map the HF config onto the serve_sim schema** (see **HF config → schema**
   below). Identify: layer count, hidden size, vocab, dtype bytes, attention
   type, FFN type (dense/MoE), and any per-layer pattern (sliding/full,
   dense/moe, mamba/attention interleave).
4. **Create `Models/<stem>.json`** following the schema below. Reuse the closest
   existing model as a template:
   - Dense + GQA + sliding/full pattern → [gemma-4-31b.json](../../../Models/gemma-4-31b.json)
   - MoE + MLA (DeepSeek/Kimi style) → [kimi-k2.6.json](../../../Models/kimi-k2.6.json)
   - Hybrid Mamba/attention/MoE → [nemotron-3-ultra.json](../../../Models/nemotron-3-ultra.json)
   - Linear-attention (gated deltanet) → [qwen3.6-27b.json](../../../Models/qwen3.6-27b.json)
5. **Validate it loads** (see **Validate** below).
6. **Reference it from a suite/config** to run it (see **Use the model**).

## HF config → schema

Read these from the HF `config.json` (HF key names vary by model; common ones
shown). Cross-check against the model card if anything is ambiguous.

| serve_sim field | HuggingFace `config.json` key(s) |
|---|---|
| `global.num_layers` | `num_hidden_layers` |
| `global.hidden_size` | `hidden_size` |
| `global.vocab_size` | `vocab_size` |
| `global.tie_word_embeddings` | `tie_word_embeddings` |
| `global.param_dtype_bytes` | derive from `torch_dtype`/`quantization_config` (bf16/fp16→2, fp8→1) |
| `global.kv_dtype_bytes` | usually 2 (bf16); 1 if KV is quantized |
| attention `num_query_heads` | `num_attention_heads` |
| attention `num_kv_heads` | `num_key_value_heads` (GQA; equal to query heads → MHA) |
| attention `head_dim` | `head_dim` (else `hidden_size / num_attention_heads`) |
| attention `sliding_window` | `sliding_window` (+ `layer_types`/`sliding_window_pattern` for the per-layer pattern) |
| MLA fields | `q_lora_rank`, `kv_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim` |
| FFN `intermediate_size` | `intermediate_size` (MoE: `moe_intermediate_size`) |
| MoE `num_experts` | `n_routed_experts` / `num_experts` / `num_local_experts` |
| MoE `num_experts_per_token` | `num_experts_per_tok` / `num_experts_per_token` |
| MoE `num_shared_experts` | `n_shared_experts` / `num_shared_experts` |
| `gated` | true for SwiGLU/GeGLU MLPs (most modern models) |
| `layer_pattern` | infer from `layer_types`, `first_k_dense_replace`/`moe_layer_freq`, or hybrid block lists |

Notes:
- **MLA** (DeepSeek/Kimi) is indicated by the presence of `kv_lora_rank` etc.
- **MoE layer placement:** many configs put the first N layers dense via
  `first_k_dense_replace` and the rest MoE, or use `moe_layer_freq` — expand
  this into the explicit `layer_pattern`.
- **Hybrid** (Nemotron/Qwen-Next) configs list per-layer block kinds (e.g.
  `layer_types` / `hybrid_override_pattern`); map each to `mamba` /
  `gated_deltanet` / `attention` / `moe` block names.

## Schema

```jsonc
{
  "name": "Display-Name",                 // optional, defaults to "model"
  "source": "https://...",                // optional provenance
  "global": {
    "num_layers": 61,                     // MUST equal len(layer_pattern)
    "hidden_size": 7168,
    "vocab_size": 163840,
    "tie_word_embeddings": false,         // default false
    "param_dtype_bytes": 1,               // weight bytes/param (1=fp8, 2=bf16)
    "kv_dtype_bytes": 2,                  // KV-cache bytes/elem
    "layer_pattern": ["dense", "moe", "moe", ...]  // one block name per layer
  },
  "blocks": {
    "dense": { ... },                     // each name used in layer_pattern
    "moe": { ... }
  }
}
```

`layer_pattern` lists a block **name** (key into `blocks`) per layer, so
repeating/interleaving patterns are spelled out explicitly. Its length MUST
equal `num_layers` or loading raises `ValueError`.

## Block types

Each entry in `blocks` has a `block_type`:

- `composite` — a mixer (attention/mamba/linear) **plus** an FFN. Most
  transformer layers. Provide an `attention` (or `mamba` / `linear_attention`)
  sub-object and an `ffn` sub-object:
  ```jsonc
  { "block_type": "composite", "attention": { ... }, "ffn": { ... } }
  ```
- `attention` — standalone attention layer (no FFN). Fields are the attention
  fields below, inline.
- `mamba` / `gated_deltanet` — standalone recurrent mixer (no FFN, no KV cache).
- `ffn` — standalone FFN layer (dense or MoE), no mixer.

(`composite` needs at least the mixer; `ffn` is optional inside a composite.)

### Attention sub-object

```jsonc
{
  "block_type": "attention",     // omit when nested inside a composite's "attention"
  "attention_type": "GQA",       // "MHA" | "GQA" | "MLA"
  ...
}
```

- **MHA / GQA** require `head_dim`. Set `num_query_heads` and `num_kv_heads`
  (GQA: kv < query, must divide evenly; MHA: equal/omit kv). Optional
  `sliding_window` (int, or `null` for full attention).
- **MLA** (DeepSeek/Kimi compressed-latent) requires ALL of: `q_lora_rank`,
  `kv_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim` (plus
  `num_query_heads`).
- **DSA sparse overlay** (optional, any type): `sparse_attention: true` requires
  `sparse_topk`, `index_n_heads`, `index_head_dim`. `indexer_shared: true`
  (GLM IndexShare) reuses a neighbour's index and requires `sparse_attention`.
- **V4 extras** (optional): `kv_compression_ratio` (>=1, one compressed KV per N
  tokens), `o_lora_rank` / `o_groups` (low-rank output projection).

### FFN sub-object

Dense:
```jsonc
{ "block_type": "ffn", "ffn_type": "dense", "intermediate_size": 18432, "gated": true }
```

MoE (`ffn_type: "MoE"`) requires `intermediate_size`, `num_experts`,
`num_experts_per_token`. Optional: `gated`, `num_shared_experts`,
`shared_expert_intermediate_size`, `moe_latent_size` (LatentMoE down-projection
width). Example:
```jsonc
{
  "block_type": "ffn", "ffn_type": "MoE",
  "intermediate_size": 2048, "gated": true,
  "num_experts": 384, "num_experts_per_token": 8,
  "num_shared_experts": 1, "shared_expert_intermediate_size": 2048
}
```

### Mamba sub-object

```jsonc
{
  "block_type": "mamba",
  "mamba_d_state": 128, "mamba_d_conv": 4, "mamba_expand": 2,
  "mamba_num_heads": 256, "mamba_head_dim": 64, "mamba_n_groups": 8
}
```

### Gated DeltaNet (linear attention) sub-object

```jsonc
{
  "block_type": "gated_deltanet",
  "num_key_heads": 16, "num_value_heads": 32,
  "key_head_dim": 128, "value_head_dim": 128,
  "conv_kernel_dim": 4
}
```

Mamba and gated_deltanet hold **no KV cache** — their state size is fixed and
independent of context length.

## Validate

After writing the file, confirm it parses into a `LayeredModel` (the package
lives under `Src/`, so put it on `PYTHONPATH`):

```powershell
$env:PYTHONPATH="Src"; python -c "from serve_sim.model_config import load_model_config; m = load_model_config('Models/<stem>.json'); print(m.name, len(m.layers), 'layers')"
```

Errors to expect if the JSON is wrong:
- `layer_pattern length N != num_layers M` — fix the pattern/count.
- `unknown attention_type` — must be MHA/GQA/MLA.
- `MLA attention requires <field>` / `sparse (DSA) attention requires <field>` —
  add the missing required field.
- `KeyError` on a `mamba_*` / MoE field — the block is missing a required key.

Run the model-config test suite to roofline-check it end to end:

```powershell
python -m pytest Tests/test_models.py -q
```

## Use the model

Reference the stem from a suite's `models` list (in a config or `Suites/*.json`):

```jsonc
"suite": { "type": "randomized", "num_workloads": 40, "models": ["<stem>"] }
```

Then run: `python run_sim.py Configs/<your-config>.json`. Ensure the target
system/devices have enough memory to host the model (see `ENGINEERING.md` on
parallelism and memory OOM checks).
