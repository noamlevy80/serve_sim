"""Load JSON model configs (``Models/*.json``) into a :class:`LayeredModel`.

The JSON schema (see ``Models/``) describes a model as a ``global`` section, a
named ``blocks`` dictionary, and a ``layer_pattern`` naming the block for each
layer. A block is either ``composite`` (a mixer + an FFN), or a standalone
``attention`` / ``mamba`` / ``ffn`` block (Nemotron interleaves these as separate
layers). This loader maps each block onto the architecture blocks in
:mod:`serve_sim.blocks`.

Mechanisms not yet implemented (MLA, DSA, Mamba) raise ``NotImplementedError``
from their block constructors, so configs using them fail loudly until the
corresponding step lands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .blocks import Attention, DenseFFN, Layer, LayeredModel, MoEFFN


def _build_attention(spec: dict[str, Any], hidden_size: int) -> Attention:
    return Attention(
        hidden_size=hidden_size,
        attention_type=spec.get("attention_type", "GQA"),
        num_query_heads=spec.get("num_query_heads", 1),
        num_kv_heads=spec.get("num_kv_heads"),
        head_dim=spec.get("head_dim"),
        sliding_window=spec.get("sliding_window"),
        q_lora_rank=spec.get("q_lora_rank"),
        kv_lora_rank=spec.get("kv_lora_rank"),
        qk_rope_head_dim=spec.get("qk_rope_head_dim"),
        qk_nope_head_dim=spec.get("qk_nope_head_dim"),
        v_head_dim=spec.get("v_head_dim"),
        sparse_attention=spec.get("sparse_attention", False),
        sparse_topk=spec.get("sparse_topk"),
        index_n_heads=spec.get("index_n_heads"),
        index_head_dim=spec.get("index_head_dim"),
    )


def _build_ffn(spec: dict[str, Any], hidden_size: int) -> DenseFFN | MoEFFN:
    if spec.get("ffn_type", "dense").lower() == "moe":
        return MoEFFN(
            hidden_size=hidden_size,
            intermediate_size=spec["intermediate_size"],
            num_experts=spec["num_experts"],
            num_experts_per_token=spec["num_experts_per_token"],
            gated=spec.get("gated", False),
            num_shared_experts=spec.get("num_shared_experts", 0),
            shared_expert_intermediate_size=spec.get("shared_expert_intermediate_size"),
            moe_latent_size=spec.get("moe_latent_size"),
        )
    return DenseFFN(
        hidden_size=hidden_size,
        intermediate_size=spec["intermediate_size"],
        gated=spec.get("gated", False),
    )


def _build_mixer(spec: dict[str, Any], hidden_size: int):
    block_type = spec.get("block_type")
    if block_type == "mamba":
        from .blocks import MambaBlock  # imported lazily until implemented

        return MambaBlock.from_spec(spec, hidden_size)
    return _build_attention(spec, hidden_size)


def _build_layer(spec: dict[str, Any], hidden_size: int, name: str) -> Layer:
    block_type = spec.get("block_type")
    if block_type == "composite":
        mixer_spec = spec.get("attention") or spec.get("mamba")
        if mixer_spec is None:
            raise ValueError(f"composite block {name!r} needs an attention/mamba mixer")
        mixer = _build_mixer(mixer_spec, hidden_size)
        ffn = _build_ffn(spec["ffn"], hidden_size) if "ffn" in spec else None
        return Layer(mixer=mixer, ffn=ffn, name=name)
    if block_type == "attention":
        return Layer(mixer=_build_attention(spec, hidden_size), ffn=None, name=name)
    if block_type == "mamba":
        return Layer(mixer=_build_mixer(spec, hidden_size), ffn=None, name=name)
    if block_type == "ffn":
        return Layer(mixer=None, ffn=_build_ffn(spec, hidden_size), name=name)
    raise ValueError(f"unknown block_type {block_type!r} for block {name!r}")


def model_from_config(config: dict[str, Any]) -> LayeredModel:
    """Build a :class:`LayeredModel` from a parsed config dict."""

    g = config["global"]
    hidden = g["hidden_size"]
    blocks = config["blocks"]
    pattern = g["layer_pattern"]
    if len(pattern) != g["num_layers"]:
        raise ValueError(
            f"layer_pattern length {len(pattern)} != num_layers {g['num_layers']}"
        )
    layers = tuple(_build_layer(blocks[name], hidden, name) for name in pattern)
    return LayeredModel(
        layers=layers,
        hidden_size=hidden,
        vocab_size=g["vocab_size"],
        param_dtype_bytes=g.get("param_dtype_bytes", 2),
        kv_dtype_bytes=g.get("kv_dtype_bytes", 2),
        tie_word_embeddings=g.get("tie_word_embeddings", False),
        name=config.get("name", "model"),
    )


def load_model_config(path: str | Path) -> LayeredModel:
    """Load and build a :class:`LayeredModel` from a JSON config file."""

    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return model_from_config(config)
