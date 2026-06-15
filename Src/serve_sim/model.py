"""Model description and forward-pass cost sizing.

This module provides a roofline-oriented view of an LLM: given the layer
dimensions it can size the weight parameters, the per-token compute (FLOPs) and
the bytes that must be read for a forward pass. The full PRD model space (MLA,
MoE, Mamba, sparse attention) is not all implemented yet; this stage supports
dense / GQA-or-MHA attention with a dense (optionally gated) FFN, which is
enough for the toy development model and the roofline tests.

All cost methods are pure arithmetic so a test can independently reproduce them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    """A roofline-sizable transformer model.

    Attributes are the minimal set needed to size every matmul in a forward
    pass for MHA/GQA attention with a dense (optionally gated) FFN.
    """

    num_layers: int
    hidden_size: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    gated: bool = False
    param_dtype_bytes: int = 2
    kv_dtype_bytes: int = 2
    tie_word_embeddings: bool = False
    include_lm_head: bool = True
    # --- MoE FFN (ffn_type == "moe") ---
    ffn_type: str = "dense"
    num_experts: int = 0
    num_experts_per_token: int = 0
    num_shared_experts: int = 0
    shared_expert_intermediate_size: int | None = None
    num_dense_layers: int = 0
    expert_persistence_mean: float = 16.0
    expert_persistence_variance: float = 4.0
    name: str = "model"

    def __post_init__(self) -> None:
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_query_heads must be a multiple of num_kv_heads "
                f"(got {self.num_query_heads} and {self.num_kv_heads})"
            )
        for field_name in (
            "num_layers",
            "hidden_size",
            "num_query_heads",
            "num_kv_heads",
            "head_dim",
            "intermediate_size",
            "vocab_size",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.ffn_type not in ("dense", "moe"):
            raise ValueError(f"ffn_type must be 'dense' or 'moe', got {self.ffn_type!r}")
        if self.ffn_type == "moe":
            if self.num_experts <= 0:
                raise ValueError("MoE model requires num_experts > 0")
            if not 1 <= self.num_experts_per_token <= self.num_experts:
                raise ValueError(
                    "num_experts_per_token must be in [1, num_experts] "
                    f"(got {self.num_experts_per_token} of {self.num_experts})"
                )
            if self.num_shared_experts < 0:
                raise ValueError("num_shared_experts must be non-negative")
            if not 0 <= self.num_dense_layers <= self.num_layers:
                raise ValueError(
                    "num_dense_layers must be in [0, num_layers] "
                    f"(got {self.num_dense_layers} of {self.num_layers})"
                )
            if self.expert_persistence_mean <= 0:
                raise ValueError("expert_persistence_mean must be positive")

    # --- dimensions ---------------------------------------------------------

    @property
    def q_dim(self) -> int:
        """Total query projection width (``num_query_heads * head_dim``)."""

        return self.num_query_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        """Total key (or value) projection width (``num_kv_heads * head_dim``)."""

        return self.num_kv_heads * self.head_dim

    # --- parameter counts ---------------------------------------------------

    @property
    def attention_weight_params(self) -> int:
        """Weights in one attention block: Q, K, V and output projections."""

        d = self.hidden_size
        return d * self.q_dim + 2 * d * self.kv_dim + self.q_dim * d

    def _ffn_params(self, intermediate: int) -> int:
        """Weights in one FFN with the given hidden width (3 if gated, else 2)."""

        matrices = 3 if self.gated else 2
        return matrices * self.hidden_size * intermediate

    @property
    def ffn_weight_params(self) -> int:
        """Weights in one dense FFN block (3 matrices if gated, else 2)."""

        return self._ffn_params(self.intermediate_size)

    @property
    def layer_weight_params(self) -> int:
        """Total weights in one dense transformer layer (attention + FFN)."""

        return self.attention_weight_params + self.ffn_weight_params

    # --- MoE sizing ---------------------------------------------------------

    def is_moe_layer(self, layer_index: int) -> bool:
        """Whether the layer at ``layer_index`` uses a MoE FFN."""

        return self.ffn_type == "moe" and layer_index >= self.num_dense_layers

    @property
    def num_moe_layers(self) -> int:
        """Number of layers that use a MoE FFN."""

        if self.ffn_type != "moe":
            return 0
        return self.num_layers - self.num_dense_layers

    @property
    def routed_expert_params(self) -> int:
        """Weights in one routed expert FFN."""

        return self._ffn_params(self.intermediate_size)

    @property
    def shared_expert_params(self) -> int:
        """Total weights across all always-active shared experts in a layer."""

        if self.num_shared_experts == 0:
            return 0
        width = self.shared_expert_intermediate_size or self.intermediate_size
        return self.num_shared_experts * self._ffn_params(width)

    def moe_layer_weight_params(self) -> int:
        """Stored weights in one MoE layer (attention + all experts + shared)."""

        return (
            self.attention_weight_params
            + self.num_experts * self.routed_expert_params
            + self.shared_expert_params
        )

    @property
    def lm_head_params(self) -> int:
        """Weights read by the LM head matmul (``hidden * vocab``)."""

        if not self.include_lm_head:
            return 0
        return self.hidden_size * self.vocab_size

    # --- byte sizes ---------------------------------------------------------

    @property
    def layer_weight_bytes(self) -> int:
        return self.layer_weight_params * self.param_dtype_bytes

    @property
    def attention_weight_bytes(self) -> int:
        return self.attention_weight_params * self.param_dtype_bytes

    @property
    def dense_ffn_bytes(self) -> int:
        return self.ffn_weight_params * self.param_dtype_bytes

    @property
    def routed_expert_bytes(self) -> int:
        return self.routed_expert_params * self.param_dtype_bytes

    @property
    def shared_expert_bytes(self) -> int:
        return self.shared_expert_params * self.param_dtype_bytes

    @property
    def lm_head_bytes(self) -> int:
        return self.lm_head_params * self.param_dtype_bytes

    @property
    def kv_bytes_per_token(self) -> int:
        """Bytes of KV cache produced/read per token in one layer (K and V)."""

        return 2 * self.kv_dim * self.kv_dtype_bytes

    # --- FLOPs --------------------------------------------------------------

    def linear_flops(self, tokens: int) -> int:
        """FLOPs for the dense projections (QKVO + FFN) over ``tokens`` tokens."""

        return 2 * tokens * self.layer_weight_params

    def attention_proj_flops(self, tokens: int) -> int:
        """FLOPs for the QKVO projections over ``tokens`` tokens."""

        return 2 * tokens * self.attention_weight_params

    def dense_ffn_flops(self, tokens: int) -> int:
        """FLOPs for one dense FFN over ``tokens`` tokens."""

        return 2 * tokens * self.ffn_weight_params

    def routed_expert_flops(self, token_expert_pairs: int) -> int:
        """FLOPs for ``token_expert_pairs`` (token, routed-expert) activations."""

        return 2 * token_expert_pairs * self.routed_expert_params

    def shared_expert_flops(self, tokens: int) -> int:
        """FLOPs for the always-active shared experts over ``tokens`` tokens."""

        return 2 * tokens * self.shared_expert_params

    def attention_flops(self, query_key_pairs: int) -> int:
        """FLOPs for attention scores+values over ``query_key_pairs`` pairs.

        Counts the QK^T and the attention-weighted value sum, each a multiply
        accumulate over the query-head width: ``4 * q_dim * pairs``.
        """

        return 4 * self.q_dim * query_key_pairs

    def lm_head_flops(self, tokens: int) -> int:
        """FLOPs for projecting ``tokens`` hidden states to vocab logits."""

        return 2 * tokens * self.lm_head_params


def toy_model(
    num_layers: int = 4,
    hidden_size: int = 256,
    num_query_heads: int = 8,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    intermediate_size: int = 1024,
    vocab_size: int = 2048,
    gated: bool = False,
    param_dtype_bytes: int = 2,
    kv_dtype_bytes: int = 2,
    include_lm_head: bool = True,
    name: str = "toy",
) -> Model:
    """Build a small, fully roofline-sizable model for development and tests."""

    if head_dim is None:
        if hidden_size % num_query_heads != 0:
            raise ValueError("hidden_size must be divisible by num_query_heads")
        head_dim = hidden_size // num_query_heads
    if num_kv_heads is None:
        num_kv_heads = num_query_heads
    return Model(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        gated=gated,
        param_dtype_bytes=param_dtype_bytes,
        kv_dtype_bytes=kv_dtype_bytes,
        include_lm_head=include_lm_head,
        name=name,
    )


def toy_moe_model(
    num_layers: int = 4,
    hidden_size: int = 256,
    num_query_heads: int = 8,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    intermediate_size: int = 512,
    vocab_size: int = 2048,
    num_experts: int = 16,
    num_experts_per_token: int = 2,
    num_shared_experts: int = 1,
    shared_expert_intermediate_size: int | None = None,
    num_dense_layers: int = 0,
    expert_persistence_mean: float = 16.0,
    expert_persistence_variance: float = 4.0,
    gated: bool = False,
    param_dtype_bytes: int = 2,
    kv_dtype_bytes: int = 2,
    include_lm_head: bool = True,
    name: str = "toy-moe",
) -> Model:
    """Build a small MoE model for development and tests."""

    if head_dim is None:
        if hidden_size % num_query_heads != 0:
            raise ValueError("hidden_size must be divisible by num_query_heads")
        head_dim = hidden_size // num_query_heads
    if num_kv_heads is None:
        num_kv_heads = num_query_heads
    return Model(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        gated=gated,
        param_dtype_bytes=param_dtype_bytes,
        kv_dtype_bytes=kv_dtype_bytes,
        include_lm_head=include_lm_head,
        ffn_type="moe",
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        num_shared_experts=num_shared_experts,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        num_dense_layers=num_dense_layers,
        expert_persistence_mean=expert_persistence_mean,
        expert_persistence_variance=expert_persistence_variance,
        name=name,
    )
