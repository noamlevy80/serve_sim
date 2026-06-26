"""Per-layer building blocks for a roofline-sizable model.

A real model is heterogeneous: layers differ in attention mechanism (MHA/GQA/MLA,
optionally sliding-window or sparse), some layers are Mamba (linear attention),
and FFNs are dense or MoE (optionally latent). This module expresses each layer
as a small block with pure-arithmetic cost methods, and assembles them into a
:class:`LayeredModel` that the shard/event generators consume.

The flat :class:`serve_sim.model.Model` (used by the toy roofline tests) is a
homogeneous special case; :meth:`LayeredModel.from_model` converts it so a single
generation path serves both. Block arithmetic for GQA/MHA + dense/MoE reproduces
the flat model's formulas exactly.

Cost conventions (matching the roofline in Project.md):
- A matmul over ``t`` tokens with ``p`` weight params costs ``2 * t * p`` FLOPs
  and reads ``p * param_dtype_bytes`` weight bytes (once per group).
- Attention score+value over one query-key pair costs ``2 * width`` per head for
  the QK^T and ``2 * width`` for the value sum.
- KV cache read bytes scale with the number of (cached) keys attended.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# --- FFN blocks ----------------------------------------------------------------


@dataclass(frozen=True)
class DenseFFN:
    """A dense (optionally gated) feed-forward block."""

    hidden_size: int
    intermediate_size: int
    gated: bool = False

    is_moe: bool = False

    @property
    def _matrices(self) -> int:
        return 3 if self.gated else 2

    @property
    def weight_params(self) -> int:
        return self._matrices * self.hidden_size * self.intermediate_size

    def cost(self, tokens: int, param_dtype_bytes: int) -> tuple[float, float]:
        """(flops, weight bytes) for this FFN over ``tokens`` tokens."""

        return 2.0 * tokens * self.weight_params, float(self.weight_params * param_dtype_bytes)


@dataclass(frozen=True)
class MoEFFN:
    """A mixture-of-experts FFN (routed experts + optional shared experts).

    When ``moe_latent_size`` is set the routed experts operate on a down-projected
    latent (LatentMoE): each layer adds a hidden->latent and latent->hidden
    projection applied to every token, and routed experts are sized at the latent
    width. Shared experts operate at ``hidden_size``.
    """

    hidden_size: int
    intermediate_size: int
    num_experts: int
    num_experts_per_token: int
    gated: bool = False
    num_shared_experts: int = 0
    shared_expert_intermediate_size: int | None = None
    moe_latent_size: int | None = None

    is_moe: bool = True

    @property
    def _matrices(self) -> int:
        return 3 if self.gated else 2

    @property
    def _expert_width(self) -> int:
        """Width the routed experts operate at (latent if set, else hidden)."""

        return self.moe_latent_size or self.hidden_size

    @property
    def routed_expert_params(self) -> int:
        """Weights in one routed expert FFN."""

        return self._matrices * self._expert_width * self.intermediate_size

    @property
    def shared_expert_params(self) -> int:
        """Total weights across all always-active shared experts."""

        if self.num_shared_experts == 0:
            return 0
        width = self.shared_expert_intermediate_size or self.intermediate_size
        return self.num_shared_experts * self._matrices * self.hidden_size * width

    @property
    def latent_proj_params(self) -> int:
        """Per-layer hidden<->latent projection weights (0 if not latent)."""

        if self.moe_latent_size is None:
            return 0
        return 2 * self.hidden_size * self.moe_latent_size

    @property
    def weight_params(self) -> int:
        """Stored weights in one MoE layer (all experts + shared + latent)."""

        return (
            self.num_experts * self.routed_expert_params
            + self.shared_expert_params
            + self.latent_proj_params
        )

    def cost(
        self, tokens: int, distinct_experts: float, param_dtype_bytes: int
    ) -> tuple[float, float]:
        """(flops, bytes) for the MoE FFN over ``tokens`` tokens.

        ``distinct_experts`` is the expected number of distinct routed experts
        touched by the group; only those experts' weights are read.
        """

        routed_flops = 2.0 * tokens * self.num_experts_per_token * self.routed_expert_params
        shared_flops = 2.0 * tokens * self.shared_expert_params
        latent_flops = 2.0 * tokens * self.latent_proj_params
        flops = routed_flops + shared_flops + latent_flops
        bytes_read = (
            distinct_experts * self.routed_expert_params * param_dtype_bytes
            + self.shared_expert_params * param_dtype_bytes
            + self.latent_proj_params * param_dtype_bytes
        )
        return flops, float(bytes_read)


# --- attention blocks ----------------------------------------------------------


@dataclass(frozen=True)
class Attention:
    """Self-attention block.

    Supports MHA/GQA and MLA (DeepSeek-style compressed-latent attention), each
    with an optional sliding window and an optional sparse (DSA) indexer overlay
    that caps the number of attended keys to ``sparse_topk``.

    DeepSeek-V4 adds two further knobs: a per-layer ``kv_compression_ratio`` for
    the hybrid compressed attention (Compressed Sparse Attention / Heavily
    Compressed Attention), which stores a single compressed KV entry per
    ``ratio`` tokens; and a low-rank output projection (``o_lora_rank`` /
    ``o_groups``).

    GLM-5.2 adds IndexShare: a sparse layer with ``indexer_shared=True`` reuses
    the top-k selection computed by an earlier layer, so it still runs the
    capped main attention but pays no indexer projection/scoring cost.
    """

    hidden_size: int
    attention_type: str = "GQA"
    num_query_heads: int = 1
    num_kv_heads: int | None = None
    head_dim: int | None = None
    sliding_window: int | None = None
    # MLA fields:
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    qk_nope_head_dim: int | None = None
    v_head_dim: int | None = None
    # DSA overlay:
    sparse_attention: bool = False
    sparse_topk: int | None = None
    index_n_heads: int | None = None
    index_head_dim: int | None = None
    # GLM-5.2 IndexShare: reuse a neighbouring layer's index (skip indexer cost).
    indexer_shared: bool = False
    # V4 compressed attention (CSA/HCA): cache one compressed KV entry per
    # ``kv_compression_ratio`` tokens; the sliding window and DSA top-k are then
    # measured in compressed entries.
    kv_compression_ratio: int = 1
    # V4 low-rank output projection (factored over ``o_groups`` groups):
    o_lora_rank: int | None = None
    o_groups: int | None = None

    has_kv: bool = True

    def __post_init__(self) -> None:
        if self.attention_type not in ("MHA", "GQA", "MLA"):
            raise ValueError(f"unknown attention_type {self.attention_type!r}")
        if self.attention_type in ("MHA", "GQA"):
            if self.head_dim is None:
                raise ValueError("MHA/GQA attention requires head_dim")
            kv_heads = self.num_kv_heads if self.num_kv_heads is not None else self.num_query_heads
            object.__setattr__(self, "num_kv_heads", kv_heads)
            if self.num_query_heads % kv_heads != 0:
                raise ValueError("num_query_heads must be a multiple of num_kv_heads")
        if self.attention_type == "MLA":
            for field in ("q_lora_rank", "kv_lora_rank", "qk_rope_head_dim",
                          "qk_nope_head_dim", "v_head_dim"):
                if getattr(self, field) is None:
                    raise ValueError(f"MLA attention requires {field}")
        if self.sparse_attention:
            for field in ("sparse_topk", "index_n_heads", "index_head_dim"):
                if getattr(self, field) is None:
                    raise ValueError(f"sparse (DSA) attention requires {field}")
        if self.kv_compression_ratio < 1:
            raise ValueError("kv_compression_ratio must be >= 1")
        if self.o_lora_rank is not None and self.o_lora_rank <= 0:
            raise ValueError("o_lora_rank must be positive")
        if self.indexer_shared and not self.sparse_attention:
            raise ValueError("indexer_shared requires sparse (DSA) attention")

    # --- dimensions ---

    @property
    def is_mla(self) -> bool:
        return self.attention_type == "MLA"

    @property
    def q_dim(self) -> int:
        return self.num_query_heads * (self.head_dim or 0)

    @property
    def kv_dim(self) -> int:
        return (self.num_kv_heads or 0) * (self.head_dim or 0)

    @property
    def _attn_output_dim(self) -> int:
        """Width of the concatenated head outputs feeding the output projection."""

        if self.is_mla:
            return self.num_query_heads * (self.v_head_dim or 0)
        return self.q_dim

    @property
    def _output_proj_params(self) -> int:
        """Output projection weights (low-rank when ``o_lora_rank`` is set)."""

        out_dim = self._attn_output_dim
        if self.o_lora_rank is None:
            return out_dim * self.hidden_size
        groups = self.o_groups or 1
        return out_dim * self.o_lora_rank + groups * self.o_lora_rank * self.hidden_size

    @property
    def weight_params(self) -> int:
        d = self.hidden_size
        if self.is_mla:
            h = self.num_query_heads
            return (
                d * self.q_lora_rank
                + self.q_lora_rank * h * (self.qk_nope_head_dim + self.qk_rope_head_dim)
                + d * (self.kv_lora_rank + self.qk_rope_head_dim)
                + self.kv_lora_rank * h * self.qk_nope_head_dim
                + self.kv_lora_rank * h * self.v_head_dim
                + self._output_proj_params
            )
        return d * self.q_dim + 2 * d * self.kv_dim + self._output_proj_params

    def kv_bytes_per_token(self, kv_dtype_bytes: int) -> float:
        """Bytes of KV cache stored/read per token.

        GQA/MHA cache K and V (``2 * kv_dim``); MLA caches a single compressed
        latent plus the decoupled rope key (``kv_lora_rank + qk_rope_head_dim``).
        DeepSeek-V4's compressed attention stores one compressed entry per
        ``kv_compression_ratio`` tokens, scaling the footprint down accordingly.
        """

        if self.is_mla:
            base = (self.kv_lora_rank + self.qk_rope_head_dim) * kv_dtype_bytes
        else:
            base = 2 * self.kv_dim * kv_dtype_bytes
        return base / self.kv_compression_ratio

    @property
    def _per_pair_flops(self) -> int:
        """FLOPs per attended query-key pair (QK^T + attention-weighted value)."""

        if self.is_mla:
            h = self.num_query_heads
            return 2 * h * ((self.qk_nope_head_dim + self.qk_rope_head_dim) + self.v_head_dim)
        return 4 * self.q_dim

    # --- DSA indexer ---

    @property
    def _indexer_proj_params(self) -> int:
        return self.hidden_size * self.index_n_heads * self.index_head_dim

    @property
    def _indexer_per_pair(self) -> int:
        return 2 * self.index_n_heads * self.index_head_dim

    def _index_kv_bytes_per_token(self, kv_dtype_bytes: int) -> int:
        return self.index_head_dim * kv_dtype_bytes

    def _window(self) -> float:
        return math.inf if self.sliding_window is None else float(self.sliding_window)

    def _main_and_candidate(self, keys: float) -> tuple[float, float]:
        """For ``keys`` causal keys, return (main-attended, indexer-candidate).

        Keys are first compressed by ``kv_compression_ratio`` (CSA/HCA), so the
        sliding window and DSA top-k below are measured in compressed KV entries.
        ``candidate`` is the window-limited set the indexer scores; ``main`` is
        that set further capped to ``sparse_topk`` when DSA is enabled.
        """

        compressed = float(keys) / self.kv_compression_ratio
        candidate = min(compressed, self._window())
        if self.sparse_attention:
            return min(candidate, float(self.sparse_topk)), candidate
        return candidate, candidate

    def _sum_clamped(self, a: int, b: int, cap: float) -> float:
        """Sum of ``min(k / kv_compression_ratio, cap)`` for integer ``k`` in [a, b].

        Closed form of the per-token accumulation in :meth:`prefill_cost` (which
        otherwise loops once per prefill position and turns long-prompt batches
        into an O(prefill_tokens) Python loop per layer). ``a`` and ``b`` are
        inclusive; callers guarantee ``a <= b``.

        ``min(k / ratio, cap)`` is linear in ``k`` until ``k / ratio`` reaches the
        cap, then constant. ``kt`` is the last integer key still in the linear
        region, so the sum splits into an arithmetic-series part and a flat part.
        """

        ratio = self.kv_compression_ratio
        if cap == math.inf:
            return (a + b) * (b - a + 1) / 2.0 / ratio
        kt = math.floor(cap * ratio)
        total = 0.0
        lin_hi = min(b, kt)
        if lin_hi >= a:
            n_lin = lin_hi - a + 1
            total += (a + lin_hi) * n_lin / 2.0 / ratio
        lo = max(a, kt + 1)
        if b >= lo:
            total += cap * (b - lo + 1)
        return total

    # --- costs ---

    def prefill_cost(
        self,
        new_tokens: int,
        cached: int,
        start: int,
        param_dtype_bytes: int,
        kv_dtype_bytes: int,
    ) -> tuple[float, float]:
        """(flops, bytes) for a prefill chunk of ``new_tokens`` query positions.

        Query position ``cached + start + j`` attends to ``cached + start + j + 1``
        keys (causal), capped by the sliding window and DSA top-k if set.
        """

        main_pairs = 0.0
        cand_pairs = 0.0
        if new_tokens > 0:
            a = cached + start + 1
            b = cached + start + new_tokens
            window = self._window()
            cand_pairs = self._sum_clamped(a, b, window)
            if self.sparse_attention:
                main_pairs = self._sum_clamped(a, b, min(window, float(self.sparse_topk)))
            else:
                main_pairs = cand_pairs
        prior_main, prior_cand = self._main_and_candidate(cached + start)
        kv_bpt = self.kv_bytes_per_token(kv_dtype_bytes)
        flops = 2.0 * new_tokens * self.weight_params + self._per_pair_flops * main_pairs
        bytes_read = self.weight_params * param_dtype_bytes + prior_main * kv_bpt
        if self.sparse_attention and not self.indexer_shared:
            flops += 2.0 * new_tokens * self._indexer_proj_params + self._indexer_per_pair * cand_pairs
            bytes_read += (
                self._indexer_proj_params * param_dtype_bytes
                + prior_cand * self._index_kv_bytes_per_token(kv_dtype_bytes)
            )
        return flops, float(bytes_read)

    def decode_cost(
        self,
        contexts: list[int],
        param_dtype_bytes: int,
        kv_dtype_bytes: int,
    ) -> tuple[float, float]:
        """(flops, bytes) for one decode step over sequences with ``contexts``."""

        main_sum = 0.0
        cand_sum = 0.0
        for c in contexts:
            main, cand = self._main_and_candidate(c)
            main_sum += main
            cand_sum += cand
        batch = len(contexts)
        kv_bpt = self.kv_bytes_per_token(kv_dtype_bytes)
        flops = 2.0 * batch * self.weight_params + self._per_pair_flops * main_sum
        bytes_read = self.weight_params * param_dtype_bytes + main_sum * kv_bpt
        if self.sparse_attention and not self.indexer_shared:
            flops += 2.0 * batch * self._indexer_proj_params + self._indexer_per_pair * cand_sum
            bytes_read += (
                self._indexer_proj_params * param_dtype_bytes
                + cand_sum * self._index_kv_bytes_per_token(kv_dtype_bytes)
            )
        return flops, float(bytes_read)


# --- mamba block ---------------------------------------------------------------


@dataclass(frozen=True)
class MambaBlock:
    """Mamba-2 state-space mixer (Nemotron).

    Roofline costs cover the input/output projections, the depthwise causal
    conv1d, and the selective-scan recurrence. The recurrent state has a fixed
    size (it does not grow with context), so a Mamba layer holds **no KV cache**
    and its per-step cost is independent of sequence length.
    """

    hidden_size: int
    d_state: int
    d_conv: int
    expand: int
    num_heads: int
    head_dim: int
    n_groups: int
    has_kv: bool = False

    @property
    def d_inner(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def _conv_channels(self) -> int:
        return self.d_inner + 2 * self.n_groups * self.d_state

    @property
    def _in_proj_params(self) -> int:
        # z, x, B, C and per-head dt
        return self.hidden_size * (
            2 * self.d_inner + 2 * self.n_groups * self.d_state + self.num_heads
        )

    @property
    def _conv_params(self) -> int:
        return self._conv_channels * self.d_conv

    @property
    def _out_proj_params(self) -> int:
        return self.d_inner * self.hidden_size

    @property
    def weight_params(self) -> int:
        return self._in_proj_params + self._conv_params + self._out_proj_params

    @property
    def _ssm_flops_per_token(self) -> int:
        # selective-scan: state update + output readout, each ~ d_inner * d_state
        return 4 * self.d_inner * self.d_state

    def kv_bytes_per_token(self, kv_dtype_bytes: int) -> int:
        return 0

    def prefill_cost(
        self,
        new_tokens: int,
        cached: int,
        start: int,
        param_dtype_bytes: int,
        kv_dtype_bytes: int,
    ) -> tuple[float, float]:
        flops = 2.0 * new_tokens * self.weight_params + new_tokens * self._ssm_flops_per_token
        return flops, float(self.weight_params * param_dtype_bytes)

    def decode_cost(
        self,
        contexts: list[int],
        param_dtype_bytes: int,
        kv_dtype_bytes: int,
    ) -> tuple[float, float]:
        batch = len(contexts)
        flops = 2.0 * batch * self.weight_params + batch * self._ssm_flops_per_token
        return flops, float(self.weight_params * param_dtype_bytes)

    @classmethod
    def from_spec(cls, spec: dict, hidden_size: int) -> "MambaBlock":
        return cls(
            hidden_size=hidden_size,
            d_state=spec["mamba_d_state"],
            d_conv=spec["mamba_d_conv"],
            expand=spec["mamba_expand"],
            num_heads=spec["mamba_num_heads"],
            head_dim=spec["mamba_head_dim"],
            n_groups=spec["mamba_n_groups"],
        )


# --- gated delta-net block -----------------------------------------------------


@dataclass(frozen=True)
class GatedDeltaNet:
    """Gated DeltaNet linear-attention mixer (Qwen3.5/3.6, Qwen3-Next).

    A linear-attention layer with separate query/key heads (``num_key_heads``)
    and value heads (``num_value_heads``), a short depthwise causal conv1d, a
    gated delta-rule recurrence, and an output (swish) gate. Like Mamba, the
    recurrent state has a fixed size, so the layer holds **no KV cache** and its
    per-step cost is independent of sequence length.
    """

    hidden_size: int
    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_dim: int = 4
    has_kv: bool = False

    @property
    def _q_dim(self) -> int:
        return self.num_key_heads * self.key_head_dim

    @property
    def _k_dim(self) -> int:
        return self.num_key_heads * self.key_head_dim

    @property
    def _v_dim(self) -> int:
        return self.num_value_heads * self.value_head_dim

    @property
    def _qkv_proj_params(self) -> int:
        return self.hidden_size * (self._q_dim + self._k_dim + self._v_dim)

    @property
    def _conv_channels(self) -> int:
        return self._q_dim + self._k_dim + self._v_dim

    @property
    def _conv_params(self) -> int:
        return self._conv_channels * self.conv_kernel_dim

    @property
    def _gate_params(self) -> int:
        # output (swish) gate over the value projection, plus tiny per-value-head
        # decay (a) and beta gates.
        return self.hidden_size * (self._v_dim + 2 * self.num_value_heads)

    @property
    def _out_proj_params(self) -> int:
        return self._v_dim * self.hidden_size

    @property
    def weight_params(self) -> int:
        return (
            self._qkv_proj_params
            + self._conv_params
            + self._gate_params
            + self._out_proj_params
        )

    @property
    def _recurrence_flops_per_token(self) -> int:
        # delta-rule state update + readout (~state size per value head), plus the
        # depthwise causal conv.
        state = 4 * self.num_value_heads * self.key_head_dim * self.value_head_dim
        conv = 2 * self._conv_channels * self.conv_kernel_dim
        return state + conv

    def kv_bytes_per_token(self, kv_dtype_bytes: int) -> int:
        return 0

    def prefill_cost(
        self,
        new_tokens: int,
        cached: int,
        start: int,
        param_dtype_bytes: int,
        kv_dtype_bytes: int,
    ) -> tuple[float, float]:
        flops = (
            2.0 * new_tokens * self.weight_params
            + new_tokens * self._recurrence_flops_per_token
        )
        return flops, float(self.weight_params * param_dtype_bytes)

    def decode_cost(
        self,
        contexts: list[int],
        param_dtype_bytes: int,
        kv_dtype_bytes: int,
    ) -> tuple[float, float]:
        batch = len(contexts)
        flops = (
            2.0 * batch * self.weight_params
            + batch * self._recurrence_flops_per_token
        )
        return flops, float(self.weight_params * param_dtype_bytes)

    @classmethod
    def from_spec(cls, spec: dict, hidden_size: int) -> "GatedDeltaNet":
        return cls(
            hidden_size=hidden_size,
            num_key_heads=spec["num_key_heads"],
            num_value_heads=spec["num_value_heads"],
            key_head_dim=spec["key_head_dim"],
            value_head_dim=spec["value_head_dim"],
            conv_kernel_dim=spec.get("conv_kernel_dim", 4),
        )


# --- layer + model -------------------------------------------------------------

Mixer = Attention | MambaBlock | GatedDeltaNet
FFN = DenseFFN | MoEFFN


@dataclass(frozen=True)
class Layer:
    """One model layer: an optional sequence mixer and an optional FFN.

    Composite layers (DeepSeek, Gemma) have both; standalone layers (Nemotron's
    interleaved mamba / attention / moe blocks) have exactly one.
    """

    mixer: Attention | MambaBlock | GatedDeltaNet | None = None
    ffn: DenseFFN | MoEFFN | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.mixer is None and self.ffn is None:
            raise ValueError("a layer must have a mixer and/or an FFN")

    @property
    def is_moe(self) -> bool:
        return self.ffn is not None and self.ffn.is_moe

    def kv_bytes_per_token(self, kv_dtype_bytes: int) -> int:
        if self.mixer is not None and self.mixer.has_kv:
            return self.mixer.kv_bytes_per_token(kv_dtype_bytes)
        return 0


@dataclass(frozen=True)
class LayeredModel:
    """A model as an ordered list of heterogeneous layers plus globals.

    This is the universal representation consumed by the shard and event
    generators; the flat :class:`serve_sim.model.Model` converts to it.
    """

    layers: tuple[Layer, ...]
    hidden_size: int
    vocab_size: int
    param_dtype_bytes: int = 2
    kv_dtype_bytes: int = 2
    tie_word_embeddings: bool = False
    include_lm_head: bool = True
    expert_persistence_mean: float = 16.0
    expert_persistence_variance: float = 4.0
    name: str = "layered-model"

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("model must have at least one layer")

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def layer_at(self, index: int) -> Layer:
        return self.layers[index]

    def is_moe_layer(self, index: int) -> bool:
        return self.layers[index].is_moe

    @property
    def num_moe_layers(self) -> int:
        return sum(1 for layer in self.layers if layer.is_moe)

    def moe_ffns(self) -> list[MoEFFN]:
        """The MoE FFN blocks across the stack (one per MoE layer)."""

        return [layer.ffn for layer in self.layers if layer.is_moe]  # type: ignore[misc]

    @property
    def lm_head_params(self) -> int:
        if not self.include_lm_head:
            return 0
        return self.hidden_size * self.vocab_size

    @property
    def lm_head_bytes(self) -> int:
        return self.lm_head_params * self.param_dtype_bytes

    def lm_head_flops(self, tokens: int) -> float:
        return 2.0 * tokens * self.lm_head_params

    @classmethod
    def from_model(cls, model) -> "LayeredModel":
        """Build a homogeneous :class:`LayeredModel` from a flat ``Model``."""

        if isinstance(model, cls):
            return model
        layers: list[Layer] = []
        for index in range(model.num_layers):
            mixer = Attention(
                hidden_size=model.hidden_size,
                attention_type="GQA",
                num_query_heads=model.num_query_heads,
                num_kv_heads=model.num_kv_heads,
                head_dim=model.head_dim,
            )
            if model.is_moe_layer(index):
                ffn: DenseFFN | MoEFFN = MoEFFN(
                    hidden_size=model.hidden_size,
                    intermediate_size=model.intermediate_size,
                    num_experts=model.num_experts,
                    num_experts_per_token=model.num_experts_per_token,
                    gated=model.gated,
                    num_shared_experts=model.num_shared_experts,
                    shared_expert_intermediate_size=model.shared_expert_intermediate_size,
                )
            else:
                ffn = DenseFFN(
                    hidden_size=model.hidden_size,
                    intermediate_size=model.intermediate_size,
                    gated=model.gated,
                )
            layers.append(Layer(mixer=mixer, ffn=ffn))
        return cls(
            layers=tuple(layers),
            hidden_size=model.hidden_size,
            vocab_size=model.vocab_size,
            param_dtype_bytes=model.param_dtype_bytes,
            kv_dtype_bytes=model.kv_dtype_bytes,
            tie_word_embeddings=model.tie_word_embeddings,
            include_lm_head=model.include_lm_head,
            expert_persistence_mean=model.expert_persistence_mean,
            expert_persistence_variance=model.expert_persistence_variance,
            name=model.name,
        )
