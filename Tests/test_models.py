"""Event-generation tests for the real model configs in ``Models/``.

Each model is loaded from its JSON config into a :class:`LayeredModel`, run
through the shard + event generators, and checked against the independent
``reference_layered`` roofline. Mechanisms are added step by step; this file
grows as DeepSeek (MLA/DSA) and Nemotron (Mamba/Latent-MoE) come online.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from serve_sim.blocks import Attention, DenseFFN, Layer, LayeredModel, MambaBlock, MoEFFN
from serve_sim.model import toy_model, toy_moe_model
from serve_sim.model_config import load_model_config
from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.shards import WorkShardGenerator
from serve_sim.tracker import SequenceWork
from serve_sim.events import EventGenerator
from reference import reference_layered, reference_roofline

MODELS_DIR = Path(__file__).resolve().parents[1] / "Models"


def make_device(peak=1e15, bw=4e12, cap=192e9):
    mem = MemoryDevice("hbm", capacity_bytes=cap, bandwidth_bytes_per_s=bw)
    return ComputeDevice("gpu", peak_flops_fp16=peak, first_tier_memory=mem)


def simulate(model, device, work, prefill_chunk_size=None):
    shards = WorkShardGenerator(model).generate(work, prefill_chunk_size)
    return EventGenerator(model, [device]).run(shards)


# --- layered reference agrees with the flat reference on homogeneous models -----


def test_layered_reference_matches_flat_dense():
    model = toy_model()
    dev = make_device(peak=1e14, bw=2e12)
    work = [SequenceWork(0, 64, 8)]
    assert reference_layered(model, dev, work) == pytest.approx(
        reference_roofline(model, dev, work), rel=1e-9
    )


def test_layered_reference_matches_flat_moe():
    model = toy_moe_model()
    dev = make_device(peak=1e14, bw=2e12)
    work = [SequenceWork(0, 48, 6) for _ in range(3)]
    assert reference_layered(model, dev, work) == pytest.approx(
        reference_roofline(model, dev, work), rel=1e-9
    )


def test_simulator_matches_layered_reference_dense():
    model = toy_model()
    dev = make_device(peak=1e14, bw=2e12)
    work = [SequenceWork(0, 40, 5), SequenceWork(0, 24, 3)]
    sched = simulate(model, dev, work)
    assert sched.makespan == pytest.approx(reference_layered(model, dev, work), rel=1e-9)


# --- sliding window attention ---------------------------------------------------


def test_sliding_window_caps_decode_attention():
    full = Attention(hidden_size=256, num_query_heads=8, num_kv_heads=8, head_dim=32)
    sliding = Attention(
        hidden_size=256, num_query_heads=8, num_kv_heads=8, head_dim=32, sliding_window=16
    )
    contexts = [100]  # far beyond the window
    f_full, b_full = full.decode_cost(contexts, 2, 2)
    f_slide, b_slide = sliding.decode_cost(contexts, 2, 2)
    # weight term identical; attention term capped to the window
    assert f_slide < f_full
    assert b_slide < b_full


def test_sliding_window_no_effect_within_window():
    full = Attention(hidden_size=256, num_query_heads=8, num_kv_heads=8, head_dim=32)
    sliding = Attention(
        hidden_size=256, num_query_heads=8, num_kv_heads=8, head_dim=32, sliding_window=1024
    )
    contexts = [64, 32]
    assert sliding.decode_cost(contexts, 2, 2) == full.decode_cost(contexts, 2, 2)


# --- Gemma (GQA + sliding window + dense gated + tied embeddings) ----------------


@pytest.fixture(scope="module")
def gemma() -> LayeredModel:
    return load_model_config(MODELS_DIR / "gemma-4-31b.json")


def test_gemma_loads(gemma):
    assert gemma.num_layers == 60
    assert gemma.tie_word_embeddings is True
    assert gemma.num_moe_layers == 0
    # every layer is composite: GQA attention + dense gated FFN
    for layer in gemma.layers:
        assert isinstance(layer.mixer, Attention)
        assert isinstance(layer.ffn, DenseFFN)
        assert layer.ffn.gated is True
    # mix of sliding and full attention layers
    windows = {layer.mixer.sliding_window for layer in gemma.layers}
    assert windows == {1024, None}
    # sliding layers and full layers differ in head_dim / kv heads
    sliding = next(layer for layer in gemma.layers if layer.mixer.sliding_window == 1024)
    full = next(layer for layer in gemma.layers if layer.mixer.sliding_window is None)
    assert sliding.mixer.head_dim == 256 and sliding.mixer.num_kv_heads == 16
    assert full.mixer.head_dim == 512 and full.mixer.num_kv_heads == 4


def test_gemma_event_generation_matches_reference(gemma):
    dev = make_device()
    work = [SequenceWork(0, 64, 4), SequenceWork(0, 32, 2)]
    sched = simulate(gemma, dev, work)
    assert sched.makespan == pytest.approx(reference_layered(gemma, dev, work), rel=1e-9)


def test_gemma_event_generation_matches_reference_chunked(gemma):
    dev = make_device()
    work = [SequenceWork(0, 200, 3)]
    sched = simulate(gemma, dev, work, prefill_chunk_size=48)
    assert sched.makespan == pytest.approx(
        reference_layered(gemma, dev, work, prefill_chunk_size=48), rel=1e-9
    )


def test_gemma_decode_emits_lm_head(gemma):
    dev = make_device()
    work = [SequenceWork(0, 16, 2)]
    shards = WorkShardGenerator(gemma).generate(work)
    assert any(s.kind == "lm_head" for s in shards)


# --- MLA + DSA building blocks ---------------------------------------------------


def _mla(sparse_topk=None, **overrides):
    kwargs = dict(
        hidden_size=1024,
        attention_type="MLA",
        num_query_heads=8,
        q_lora_rank=256,
        kv_lora_rank=128,
        qk_rope_head_dim=16,
        qk_nope_head_dim=32,
        v_head_dim=32,
    )
    if sparse_topk is not None:
        kwargs.update(
            sparse_attention=True,
            sparse_topk=sparse_topk,
            index_n_heads=4,
            index_head_dim=32,
        )
    kwargs.update(overrides)
    return Attention(**kwargs)


def test_mla_caches_compressed_latent():
    mla = _mla()
    # latent + decoupled rope key, single copy (not 2x like MHA)
    assert mla.kv_bytes_per_token(2) == (128 + 16) * 2


def test_dsa_caps_main_attention_for_long_context():
    dense = _mla()
    sparse = _mla(sparse_topk=64)
    contexts = [4096]  # far beyond top-k, so the indexer overhead is amortized
    f_dense, b_dense = dense.decode_cost(contexts, 1, 2)
    f_sparse, b_sparse = sparse.decode_cost(contexts, 1, 2)
    # main attention is capped to top-k; at long context sparse is far cheaper
    assert f_sparse < f_dense
    assert b_sparse < b_dense


def test_dsa_no_cap_within_topk():
    dense = _mla()
    sparse = _mla(sparse_topk=4096)
    contexts = [128, 64]
    # within top-k, main attention matches dense plus a positive indexer term
    f_dense, _ = dense.decode_cost(contexts, 1, 2)
    f_sparse, _ = sparse.decode_cost(contexts, 1, 2)
    assert f_sparse > f_dense


def test_synthetic_mla_dsa_matches_reference():
    layers = (
        Layer(mixer=_mla(sparse_topk=64), ffn=DenseFFN(1024, 4096, gated=True), name="mla"),
        Layer(
            mixer=_mla(sparse_topk=64),
            ffn=MoEFFN(1024, 1024, num_experts=16, num_experts_per_token=4, gated=True,
                       num_shared_experts=1, shared_expert_intermediate_size=1024),
            name="mla-moe",
        ),
    )
    model = LayeredModel(layers=layers, hidden_size=1024, vocab_size=8192, param_dtype_bytes=1)
    dev = make_device(peak=1e14, bw=2e12)
    work = [SequenceWork(0, 128, 4), SequenceWork(0, 96, 2)]
    sched = simulate(model, dev, work)
    assert sched.makespan == pytest.approx(reference_layered(model, dev, work), rel=1e-9)


# --- DeepSeek (MLA + DSA + dense/MoE) -------------------------------------------


@pytest.fixture(scope="module")
def deepseek() -> LayeredModel:
    return load_model_config(MODELS_DIR / "deepseek-v3.2.json")


def test_deepseek_loads(deepseek):
    assert deepseek.num_layers == 61
    assert deepseek.num_moe_layers == 58
    dense_layers = [l for l in deepseek.layers if isinstance(l.ffn, DenseFFN)]
    moe_layers = [l for l in deepseek.layers if isinstance(l.ffn, MoEFFN)]
    assert len(dense_layers) == 3 and len(moe_layers) == 58
    for layer in deepseek.layers:
        assert layer.mixer.is_mla
        assert layer.mixer.sparse_attention is True
        assert layer.mixer.sparse_topk == 2048
    moe = moe_layers[0].ffn
    assert moe.num_experts == 256 and moe.num_experts_per_token == 8
    assert moe.num_shared_experts == 1


def test_deepseek_event_generation_matches_reference(deepseek):
    dev = make_device()
    work = [SequenceWork(0, 64, 4), SequenceWork(0, 32, 2)]
    sched = simulate(deepseek, dev, work)
    assert sched.makespan == pytest.approx(reference_layered(deepseek, dev, work), rel=1e-9)


def test_deepseek_event_generation_matches_reference_chunked(deepseek):
    dev = make_device()
    work = [SequenceWork(0, 160, 3)]
    sched = simulate(deepseek, dev, work, prefill_chunk_size=48)
    assert sched.makespan == pytest.approx(
        reference_layered(deepseek, dev, work, prefill_chunk_size=48), rel=1e-9
    )


# --- Mamba building block --------------------------------------------------------


def _mamba(**overrides):
    kwargs = dict(
        hidden_size=2048,
        d_state=128,
        d_conv=4,
        expand=2,
        num_heads=32,
        head_dim=64,
        n_groups=8,
    )
    kwargs.update(overrides)
    return MambaBlock(**kwargs)


def test_mamba_has_no_kv_cache():
    mamba = _mamba()
    assert mamba.has_kv is False
    assert mamba.kv_bytes_per_token(2) == 0


def test_mamba_decode_cost_independent_of_context():
    mamba = _mamba()
    short = mamba.decode_cost([8], 2, 2)
    long = mamba.decode_cost([100000], 2, 2)
    assert short == long  # fixed-size recurrent state


def test_synthetic_mamba_moe_matches_reference():
    layers = (
        Layer(mixer=_mamba(), ffn=None, name="mamba"),
        Layer(
            mixer=None,
            ffn=MoEFFN(2048, 1024, num_experts=16, num_experts_per_token=4, gated=False,
                       num_shared_experts=1, shared_expert_intermediate_size=2048,
                       moe_latent_size=512),
            name="latent-moe",
        ),
        Layer(mixer=Attention(2048, num_query_heads=16, num_kv_heads=2, head_dim=128), ffn=None,
              name="attn"),
    )
    model = LayeredModel(layers=layers, hidden_size=2048, vocab_size=8192)
    dev = make_device(peak=1e14, bw=2e12)
    work = [SequenceWork(0, 96, 4), SequenceWork(0, 48, 2)]
    sched = simulate(model, dev, work)
    assert sched.makespan == pytest.approx(reference_layered(model, dev, work), rel=1e-9)


# --- Nemotron (Mamba + GQA + Latent-MoE) ----------------------------------------


@pytest.fixture(scope="module")
def nemotron() -> LayeredModel:
    return load_model_config(MODELS_DIR / "nemotron-3-ultra.json")


def test_nemotron_loads(nemotron):
    assert nemotron.num_layers == 108
    mamba_layers = [l for l in nemotron.layers if isinstance(l.mixer, MambaBlock)]
    attn_layers = [l for l in nemotron.layers if isinstance(l.mixer, Attention)]
    moe_layers = [l for l in nemotron.layers if isinstance(l.ffn, MoEFFN)]
    assert mamba_layers and attn_layers and moe_layers
    # standalone blocks: a layer has either a mixer or an FFN, not both
    for layer in nemotron.layers:
        assert (layer.mixer is None) != (layer.ffn is None)
    moe = moe_layers[0].ffn
    assert moe.num_experts == 512 and moe.num_experts_per_token == 22
    assert moe.moe_latent_size == 2048  # LatentMoE


def test_nemotron_event_generation_matches_reference(nemotron):
    dev = make_device()
    work = [SequenceWork(0, 64, 4), SequenceWork(0, 32, 2)]
    sched = simulate(nemotron, dev, work)
    assert sched.makespan == pytest.approx(reference_layered(nemotron, dev, work), rel=1e-9)


def test_nemotron_event_generation_matches_reference_chunked(nemotron):
    dev = make_device()
    work = [SequenceWork(0, 160, 3)]
    sched = simulate(nemotron, dev, work, prefill_chunk_size=48)
    assert sched.makespan == pytest.approx(
        reference_layered(nemotron, dev, work, prefill_chunk_size=48), rel=1e-9
    )
