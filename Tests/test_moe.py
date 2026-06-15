"""Tests for MoE model sizing and MoE-aware work-shard generation."""

from __future__ import annotations

import pytest

from serve_sim.model import Model, toy_moe_model
from serve_sim.experts import ExpertUsageModel
from serve_sim.shards import WorkShardGenerator
from serve_sim.tracker import SequenceWork


# --- model sizing ---------------------------------------------------------------


def test_moe_model_validation():
    with pytest.raises(ValueError, match="num_experts"):
        Model(
            num_layers=2, hidden_size=128, num_query_heads=4, num_kv_heads=4,
            head_dim=32, intermediate_size=256, vocab_size=128, ffn_type="moe",
            num_experts=0, num_experts_per_token=1,
        )
    with pytest.raises(ValueError, match="num_experts_per_token"):
        toy_moe_model(num_experts=8, num_experts_per_token=9)
    with pytest.raises(ValueError, match="num_dense_layers"):
        toy_moe_model(num_layers=4, num_dense_layers=5)


def test_is_moe_layer_respects_dense_prefix():
    model = toy_moe_model(num_layers=4, num_dense_layers=2)
    assert not model.is_moe_layer(0)
    assert not model.is_moe_layer(1)
    assert model.is_moe_layer(2)
    assert model.is_moe_layer(3)


def test_dense_model_has_no_moe_layers():
    from serve_sim.model import toy_model

    model = toy_model()
    assert not any(model.is_moe_layer(i) for i in range(model.num_layers))


def test_routed_and_shared_expert_params():
    model = toy_moe_model(
        hidden_size=256, intermediate_size=512, num_shared_experts=2, gated=False
    )
    assert model.routed_expert_params == 2 * 256 * 512
    assert model.shared_expert_params == 2 * (2 * 256 * 512)  # 2 shared experts


def test_shared_expert_custom_intermediate():
    model = toy_moe_model(
        hidden_size=256,
        intermediate_size=512,
        num_shared_experts=1,
        shared_expert_intermediate_size=128,
    )
    assert model.shared_expert_params == 2 * 256 * 128


def test_shared_expert_zero_when_none():
    model = toy_moe_model(num_shared_experts=0)
    assert model.shared_expert_params == 0
    assert model.shared_expert_bytes == 0


def test_moe_layer_weight_params_includes_all_experts():
    model = toy_moe_model(num_experts=8, num_shared_experts=1)
    expected = (
        model.attention_weight_params
        + 8 * model.routed_expert_params
        + model.shared_expert_params
    )
    assert model.moe_layer_weight_params() == expected


def test_expert_flops_helpers():
    model = toy_moe_model()
    assert model.routed_expert_flops(10) == 2 * 10 * model.routed_expert_params
    assert model.shared_expert_flops(4) == 2 * 4 * model.shared_expert_params


# --- MoE shard generation -------------------------------------------------------


def test_moe_prefill_one_shard_per_layer():
    model = toy_moe_model(num_layers=3)
    gen = WorkShardGenerator(model)
    work = [SequenceWork(0, 16, 0)]
    shards = gen.generate(work)
    prefill = [s for s in shards if s.phase == "prefill"]
    assert len(prefill) == 3


def test_moe_shard_bytes_match_attention_plus_experts():
    model = toy_moe_model(num_layers=1, include_lm_head=False, num_dense_layers=0)
    gen = WorkShardGenerator(model)
    # single decode step, batch of 1 sequence
    work = [SequenceWork(0, 8, 1)]
    shards = gen.generate(work)
    decode = [s for s in shards if s.phase == "decode"]
    shard = decode[0]
    usage = ExpertUsageModel.from_model(model)
    distinct = usage.expected_distinct(1, consecutive=False)
    total_context = 8 + 1
    expected_bytes = (
        model.attention_weight_bytes
        + total_context * model.kv_bytes_per_token
        + distinct * model.routed_expert_bytes
        + model.shared_expert_bytes
    )
    assert shard.bytes_read == pytest.approx(expected_bytes)


def test_moe_decode_bytes_grow_with_batch_via_distinct_experts():
    model = toy_moe_model(num_layers=1, include_lm_head=False, num_experts=64)
    gen = WorkShardGenerator(model)
    small = gen.generate([SequenceWork(0, 4, 1)])
    large = gen.generate([SequenceWork(0, 4, 1) for _ in range(8)])
    s_small = next(s for s in small if s.phase == "decode")
    s_large = next(s for s in large if s.phase == "decode")
    # more sequences -> more distinct experts -> more expert weight bytes
    small_expert_bytes = s_small.bytes_read - (
        model.attention_weight_bytes + 5 * model.kv_bytes_per_token + model.shared_expert_bytes
    )
    large_expert_bytes = s_large.bytes_read - (
        model.attention_weight_bytes + 8 * 5 * model.kv_bytes_per_token + model.shared_expert_bytes
    )
    assert large_expert_bytes > small_expert_bytes


def test_moe_compute_flops_scale_with_topk_not_total_experts():
    # routed compute depends on k_E, not E: two models with same k but different E
    a = toy_moe_model(num_layers=1, num_experts=16, num_experts_per_token=2, include_lm_head=False)
    b = toy_moe_model(num_layers=1, num_experts=64, num_experts_per_token=2, include_lm_head=False)
    work = [SequenceWork(0, 0, 1)]
    sa = next(s for s in WorkShardGenerator(a).generate(work) if s.phase == "decode")
    sb = next(s for s in WorkShardGenerator(b).generate(work) if s.phase == "decode")
    assert sa.flops == pytest.approx(sb.flops)


def test_mixed_dense_moe_layers_have_different_costs():
    model = toy_moe_model(num_layers=4, num_dense_layers=2, num_experts=32)
    gen = WorkShardGenerator(model)
    # large batch so MoE touches many experts -> heavier than dense FFN bytes
    work = [SequenceWork(0, 0, 1) for _ in range(8)]
    shards = gen.generate(work)
    decode_layer_shards = {
        s.layer_index: s for s in shards if s.phase == "decode" and s.kind == "layer"
    }
    dense_bytes = decode_layer_shards[0].bytes_read
    moe_bytes = decode_layer_shards[2].bytes_read
    assert dense_bytes != moe_bytes
