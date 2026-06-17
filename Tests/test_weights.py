"""Tests for the model-weights tracker (``serve_sim.weights``).

The tracker decomposes a model into weight shards (per-layer attention/Mamba,
dense FFN or per-expert + shared-expert + latent-projection for MoE, plus the
global LM head) and records, per shard, the memory devices it resides on -- by
object identity, so two value-equal memories stay distinct locations.
"""

from __future__ import annotations

import pytest

from serve_sim.blocks import (
    Attention,
    DenseFFN,
    Layer,
    LayeredModel,
    MambaBlock,
    MoEFFN,
)
from serve_sim.hardware import MemoryDevice
from serve_sim.model import toy_model, toy_moe_model
from serve_sim.weights import ModelWeightsTracker, WeightShard


def mem(name: str) -> MemoryDevice:
    return MemoryDevice(name=name, capacity_bytes=1e12, bandwidth_bytes_per_s=1e12)


# --- shard enumeration ---------------------------------------------------------


def test_dense_model_shards_attention_ffn_per_layer_plus_lm_head() -> None:
    model = toy_model(num_layers=3)
    tracker = ModelWeightsTracker.from_model(model)

    attn = [s for s in tracker.shards if s.component == "attention"]
    ffn = [s for s in tracker.shards if s.component == "ffn"]
    heads = [s for s in tracker.shards if s.component == "lm_head"]
    assert len(attn) == 3
    assert len(ffn) == 3
    assert len(heads) == 1
    # One attention + one ffn per layer, plus a single global LM head.
    assert len(tracker) == 3 + 3 + 1


def test_dense_model_total_bytes_matches_block_arithmetic() -> None:
    model = toy_model(num_layers=3)
    layered = LayeredModel.from_model(model)
    tracker = ModelWeightsTracker.from_model(model)

    pdb = layered.param_dtype_bytes
    expected = layered.lm_head_bytes
    for layer in layered.layers:
        expected += layer.mixer.weight_params * pdb
        expected += layer.ffn.weight_params * pdb
    assert tracker.total_bytes == pytest.approx(expected)


def test_shard_indices_are_contiguous_and_stable() -> None:
    tracker = ModelWeightsTracker.from_model(toy_model(num_layers=2))
    assert [s.index for s in tracker.shards] == list(range(len(tracker)))


def test_moe_model_expands_routed_experts_and_shared_expert() -> None:
    model = toy_moe_model(
        num_layers=2, num_experts=8, num_shared_experts=1, num_dense_layers=0
    )
    layered = LayeredModel.from_model(model)
    tracker = ModelWeightsTracker.from_model(model)

    experts = [s for s in tracker.shards if s.component == "expert"]
    shared = [s for s in tracker.shards if s.component == "shared_expert"]
    # 8 routed experts per MoE layer, both layers MoE.
    assert len(experts) == 8 * 2
    assert {s.expert_index for s in experts if s.layer_index == 0} == set(range(8))
    assert len(shared) == 2
    # Per-expert shard size matches one routed expert's weights.
    moe = layered.layers[0].ffn
    assert experts[0].bytes == pytest.approx(
        moe.routed_expert_params * layered.param_dtype_bytes
    )


def test_moe_total_bytes_matches_block_arithmetic() -> None:
    model = toy_moe_model(num_layers=3, num_experts=8, num_dense_layers=1)
    layered = LayeredModel.from_model(model)
    tracker = ModelWeightsTracker.from_model(model)

    pdb = layered.param_dtype_bytes
    expected = layered.lm_head_bytes
    for layer in layered.layers:
        if layer.mixer is not None:
            expected += layer.mixer.weight_params * pdb
        expected += layer.ffn.weight_params * pdb
    assert tracker.total_bytes == pytest.approx(expected)


def test_dense_layers_emit_a_single_ffn_shard() -> None:
    model = toy_moe_model(num_layers=3, num_experts=8, num_dense_layers=1)
    layered = LayeredModel.from_model(model)
    tracker = ModelWeightsTracker.from_model(model)

    dense_layer = next(
        i for i, layer in enumerate(layered.layers) if not layer.is_moe
    )
    layer_shards = tracker.shards_for_layer(dense_layer)
    assert sorted(s.component for s in layer_shards) == ["attention", "ffn"]


def test_latent_proj_shard_emitted_for_latent_moe() -> None:
    layer = Layer(
        mixer=Attention(
            hidden_size=256, attention_type="GQA", num_query_heads=8,
            num_kv_heads=8, head_dim=32,
        ),
        ffn=MoEFFN(
            hidden_size=256, intermediate_size=512, num_experts=4,
            num_experts_per_token=2, moe_latent_size=128,
        ),
    )
    model = LayeredModel(layers=(layer,), hidden_size=256, vocab_size=1024)
    tracker = ModelWeightsTracker.from_model(model)

    latent = [s for s in tracker.shards if s.component == "latent_proj"]
    assert len(latent) == 1
    assert latent[0].bytes == pytest.approx(
        layer.ffn.latent_proj_params * model.param_dtype_bytes
    )


def test_mamba_layer_emits_mamba_shard() -> None:
    layer = Layer(
        mixer=MambaBlock(
            hidden_size=256, d_state=16, d_conv=4, expand=2, num_heads=8,
            head_dim=32, n_groups=1,
        ),
        ffn=DenseFFN(hidden_size=256, intermediate_size=512),
    )
    model = LayeredModel(layers=(layer,), hidden_size=256, vocab_size=1024)
    tracker = ModelWeightsTracker.from_model(model)

    assert any(s.component == "mamba" for s in tracker.shards)
    assert not any(s.component == "attention" for s in tracker.shards)


def test_no_lm_head_shard_when_disabled() -> None:
    model = toy_model(num_layers=2, include_lm_head=False)
    tracker = ModelWeightsTracker.from_model(model)
    assert not any(s.component == "lm_head" for s in tracker.shards)


# --- residency -----------------------------------------------------------------


def test_place_and_is_resident() -> None:
    tracker = ModelWeightsTracker.from_model(toy_model(num_layers=1))
    nvm = mem("nvm")
    shard = tracker.shards[0]
    assert not tracker.is_resident(shard, nvm)
    tracker.place(shard, nvm)
    assert tracker.is_resident(shard, nvm)


def test_value_equal_devices_are_distinct_locations() -> None:
    tracker = ModelWeightsTracker.from_model(toy_model(num_layers=1))
    a = mem("hbm")
    b = mem("hbm")  # value-equal but a distinct instance
    assert a == b
    shard = tracker.shards[0]
    tracker.place(shard, a)
    assert tracker.is_resident(shard, a)
    assert not tracker.is_resident(shard, b)


def test_place_all_then_bytes_on_equals_total() -> None:
    tracker = ModelWeightsTracker.from_model(toy_moe_model(num_layers=2))
    nvm = mem("nvm")
    tracker.place_all(nvm)
    assert tracker.bytes_on(nvm) == pytest.approx(tracker.total_bytes)
    assert len(tracker.resident_shards(nvm)) == len(tracker)


def test_evict_removes_only_that_copy() -> None:
    tracker = ModelWeightsTracker.from_model(toy_model(num_layers=1))
    nvm = mem("nvm")
    hbm = mem("hbm")
    shard = tracker.shards[0]
    tracker.place(shard, nvm)
    tracker.place(shard, hbm)
    tracker.evict(shard, nvm)
    assert not tracker.is_resident(shard, nvm)
    assert tracker.is_resident(shard, hbm)
    assert {id(d) for d in tracker.devices_of(shard)} == {id(hbm)}


def test_evict_absent_copy_is_noop() -> None:
    tracker = ModelWeightsTracker.from_model(toy_model(num_layers=1))
    shard = tracker.shards[0]
    tracker.evict(shard, mem("nvm"))  # no error
    assert tracker.devices_of(shard) == []


def test_bytes_on_counts_only_resident_shards() -> None:
    tracker = ModelWeightsTracker.from_model(toy_model(num_layers=2))
    hbm = mem("hbm")
    first, second = tracker.shards[0], tracker.shards[1]
    tracker.place(first, hbm)
    tracker.place(second, hbm)
    assert tracker.bytes_on(hbm) == pytest.approx(first.bytes + second.bytes)


def test_shard_from_other_tracker_rejected() -> None:
    a = ModelWeightsTracker.from_model(toy_model(num_layers=1))
    b = ModelWeightsTracker.from_model(toy_model(num_layers=1))
    foreign = b.shards[0]
    with pytest.raises(ValueError):
        a.place(foreign, mem("nvm"))


def test_fabricated_shard_rejected() -> None:
    tracker = ModelWeightsTracker.from_model(toy_model(num_layers=1))
    bogus = WeightShard(index=999, component="ffn", bytes=1.0)
    with pytest.raises(ValueError):
        a = mem("nvm")
        tracker.is_resident(bogus, a)


# --- descriptor lookup ---------------------------------------------------------


def test_shard_for_descriptor() -> None:
    tracker = ModelWeightsTracker.from_model(
        toy_moe_model(num_layers=1, num_experts=4, num_dense_layers=0)
    )
    expert2 = tracker.shard_for("expert", layer_index=0, expert_index=2)
    assert expert2.component == "expert"
    assert expert2.expert_index == 2


def test_shard_for_missing_raises() -> None:
    tracker = ModelWeightsTracker.from_model(toy_model(num_layers=1))
    with pytest.raises(KeyError):
        tracker.shard_for("mamba", layer_index=0)


def test_shard_for_ambiguous_raises() -> None:
    tracker = ModelWeightsTracker.from_model(
        toy_moe_model(num_layers=1, num_experts=4, num_dense_layers=0)
    )
    # Many "expert" shards share component+layer without an expert index.
    with pytest.raises(KeyError):
        tracker.shard_for("expert", layer_index=0)
