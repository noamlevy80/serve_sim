"""Tests for Model sizing arithmetic and the toy_model factory."""

from __future__ import annotations

import pytest

from serve_sim.model import Model, toy_model


def test_toy_model_defaults_head_dim_and_kv_heads():
    m = toy_model(hidden_size=256, num_query_heads=8)
    assert m.head_dim == 32
    assert m.num_kv_heads == 8
    assert m.q_dim == 256
    assert m.kv_dim == 256


def test_toy_model_requires_divisible_hidden():
    with pytest.raises(ValueError, match="divisible"):
        toy_model(hidden_size=250, num_query_heads=8)


def test_attention_weight_params_mha():
    m = toy_model(hidden_size=256, num_query_heads=8)  # MHA, q_dim=kv_dim=256
    # Q + K + V + O = 4 * d * d
    assert m.attention_weight_params == 4 * 256 * 256


def test_attention_weight_params_gqa():
    # GQA: fewer KV heads -> smaller K, V projections
    m = toy_model(hidden_size=256, num_query_heads=8, num_kv_heads=2)
    d, q_dim, kv_dim = 256, 256, 64
    expected = d * q_dim + 2 * d * kv_dim + q_dim * d
    assert m.attention_weight_params == expected


def test_ffn_weight_params_gated_vs_ungated():
    ungated = toy_model(hidden_size=256, intermediate_size=1024, gated=False)
    gated = toy_model(hidden_size=256, intermediate_size=1024, gated=True)
    assert ungated.ffn_weight_params == 2 * 256 * 1024
    assert gated.ffn_weight_params == 3 * 256 * 1024


def test_layer_weight_params_is_sum():
    m = toy_model()
    assert m.layer_weight_params == m.attention_weight_params + m.ffn_weight_params


def test_lm_head_params_toggle():
    with_head = toy_model(hidden_size=256, vocab_size=2048, include_lm_head=True)
    without = toy_model(hidden_size=256, vocab_size=2048, include_lm_head=False)
    assert with_head.lm_head_params == 256 * 2048
    assert without.lm_head_params == 0


def test_kv_bytes_per_token():
    m = toy_model(hidden_size=256, num_query_heads=8, num_kv_heads=2, kv_dtype_bytes=2)
    # 2 (K and V) * kv_dim * bytes
    assert m.kv_bytes_per_token == 2 * 64 * 2


def test_flops_helpers():
    m = toy_model()
    assert m.linear_flops(10) == 2 * 10 * m.layer_weight_params
    assert m.attention_flops(7) == 4 * m.q_dim * 7
    assert m.lm_head_flops(3) == 2 * 3 * m.lm_head_params


def test_byte_helpers():
    m = toy_model(param_dtype_bytes=2)
    assert m.layer_weight_bytes == m.layer_weight_params * 2
    assert m.lm_head_bytes == m.lm_head_params * 2


def test_model_validation():
    with pytest.raises(ValueError, match="multiple of num_kv_heads"):
        Model(
            num_layers=2,
            hidden_size=256,
            num_query_heads=7,
            num_kv_heads=2,
            head_dim=32,
            intermediate_size=512,
            vocab_size=128,
        )
    with pytest.raises(ValueError, match="positive"):
        Model(
            num_layers=0,
            hidden_size=256,
            num_query_heads=8,
            num_kv_heads=8,
            head_dim=32,
            intermediate_size=512,
            vocab_size=128,
        )
