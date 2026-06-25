"""Event-generation tests for the real model configs in ``Models/``.

Each model is loaded from its JSON config into a :class:`LayeredModel`, run
through the shard + event generators, and checked against the independent
``reference_layered`` roofline. Mechanisms are added step by step; this file
grows as DeepSeek (MLA/DSA) and Nemotron (Mamba/Latent-MoE) come online.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from serve_sim.blocks import Attention, DenseFFN, GatedDeltaNet, Layer, LayeredModel, MambaBlock, MoEFFN
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


# --- Gated DeltaNet building block -----------------------------------------------


def _gdn(**overrides):
    kwargs = dict(
        hidden_size=2048,
        num_key_heads=8,
        num_value_heads=24,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel_dim=4,
    )
    kwargs.update(overrides)
    return GatedDeltaNet(**kwargs)


def test_gated_deltanet_has_no_kv_cache():
    gdn = _gdn()
    assert gdn.has_kv is False
    assert gdn.kv_bytes_per_token(2) == 0


def test_gated_deltanet_decode_cost_independent_of_context():
    gdn = _gdn()
    short = gdn.decode_cost([8], 2, 2)
    long = gdn.decode_cost([100000], 2, 2)
    assert short == long  # fixed-size recurrent state


def test_gated_deltanet_weight_params():
    gdn = _gdn()
    q = k = gdn.num_key_heads * gdn.key_head_dim
    v = gdn.num_value_heads * gdn.value_head_dim
    expected = (
        gdn.hidden_size * (q + k + v)  # qkv proj
        + (q + k + v) * gdn.conv_kernel_dim  # conv
        + gdn.hidden_size * (v + 2 * gdn.num_value_heads)  # gates
        + v * gdn.hidden_size  # out proj
    )
    assert gdn.weight_params == expected


def test_synthetic_gated_deltanet_matches_reference():
    layers = (
        Layer(mixer=_gdn(), ffn=DenseFFN(2048, 5504, gated=True), name="linear"),
        Layer(
            mixer=Attention(2048, num_query_heads=16, num_kv_heads=2, head_dim=128),
            ffn=DenseFFN(2048, 5504, gated=True),
            name="full",
        ),
    )
    model = LayeredModel(layers=layers, hidden_size=2048, vocab_size=8192)
    dev = make_device(peak=1e14, bw=2e12)
    work = [SequenceWork(0, 96, 4), SequenceWork(0, 48, 2)]
    sched = simulate(model, dev, work)
    assert sched.makespan == pytest.approx(reference_layered(model, dev, work), rel=1e-9)


# --- Qwen3.6-27B (Gated DeltaNet + GQA + dense) ---------------------------------


@pytest.fixture(scope="module")
def qwen36() -> LayeredModel:
    return load_model_config(MODELS_DIR / "qwen3.6-27b.json")


def test_qwen36_loads(qwen36):
    assert qwen36.num_layers == 64
    assert qwen36.hidden_size == 5120
    assert qwen36.vocab_size == 248320
    assert qwen36.tie_word_embeddings is False
    linear_layers = [l for l in qwen36.layers if isinstance(l.mixer, GatedDeltaNet)]
    full_layers = [l for l in qwen36.layers if isinstance(l.mixer, Attention)]
    assert len(linear_layers) == 48 and len(full_layers) == 16
    # the [linear x3, full] x16 pattern
    assert [isinstance(l.mixer, Attention) for l in qwen36.layers[:4]] == [
        False, False, False, True
    ]
    # every layer has its own dense gated FFN
    for layer in qwen36.layers:
        assert isinstance(layer.ffn, DenseFFN)
        assert layer.ffn.gated is True
        assert layer.ffn.intermediate_size == 17408
    gdn = linear_layers[0].mixer
    assert gdn.num_key_heads == 16 and gdn.num_value_heads == 48
    assert gdn.key_head_dim == 128 and gdn.value_head_dim == 128
    assert gdn.has_kv is False
    full = full_layers[0].mixer
    assert full.attention_type == "GQA"
    assert full.num_query_heads == 24 and full.num_kv_heads == 4
    assert full.head_dim == 256


def test_qwen36_event_generation_matches_reference(qwen36):
    dev = make_device()
    work = [SequenceWork(0, 64, 4), SequenceWork(0, 32, 2)]
    sched = simulate(qwen36, dev, work)
    assert sched.makespan == pytest.approx(reference_layered(qwen36, dev, work), rel=1e-9)


def test_qwen36_event_generation_matches_reference_chunked(qwen36):
    dev = make_device()
    work = [SequenceWork(0, 160, 3)]
    sched = simulate(qwen36, dev, work, prefill_chunk_size=48)
    assert sched.makespan == pytest.approx(
        reference_layered(qwen36, dev, work, prefill_chunk_size=48), rel=1e-9
    )


# --- DeepSeek-V4 compressed attention (CSA / HCA) + output LoRA ------------------


def _mla_v4(**overrides):
    kwargs = dict(
        hidden_size=512,
        attention_type="MLA",
        num_query_heads=8,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_rope_head_dim=16,
        qk_nope_head_dim=32,
        v_head_dim=32,
    )
    kwargs.update(overrides)
    return Attention(**kwargs)


def test_kv_compression_scales_cache_footprint():
    base = _mla_v4()
    hca = _mla_v4(kv_compression_ratio=128)
    assert base.kv_bytes_per_token(2) == (64 + 16) * 2
    assert hca.kv_bytes_per_token(2) == base.kv_bytes_per_token(2) / 128


def test_kv_compression_reduces_attention_cost():
    base = _mla_v4()
    compressed = _mla_v4(kv_compression_ratio=8)
    contexts = [4096]
    f_base, b_base = base.decode_cost(contexts, 1, 2)
    f_comp, b_comp = compressed.decode_cost(contexts, 1, 2)
    # the weight term is unchanged but the attention (per-pair) + KV-read terms shrink
    assert f_comp < f_base
    assert b_comp < b_base


def test_output_lora_weight_params():
    full = _mla_v4()
    lora = _mla_v4(o_lora_rank=64, o_groups=2)
    out_dim = full.num_query_heads * full.v_head_dim
    # full output projection -> low-rank factored projection
    assert lora.weight_params - lora._output_proj_params == (
        full.weight_params - out_dim * full.hidden_size
    )
    assert lora._output_proj_params == out_dim * 64 + 2 * 64 * full.hidden_size


def test_synthetic_compressed_mla_matches_reference():
    hca = _mla_v4(kv_compression_ratio=16, sliding_window=8, o_lora_rank=64, o_groups=2)
    csa = _mla_v4(
        kv_compression_ratio=4,
        sparse_attention=True,
        sparse_topk=16,
        index_n_heads=4,
        index_head_dim=16,
        o_lora_rank=64,
        o_groups=2,
    )
    layers = (
        Layer(mixer=hca, ffn=DenseFFN(512, 1024, gated=True), name="hca"),
        Layer(mixer=csa, ffn=DenseFFN(512, 1024, gated=True), name="csa"),
    )
    model = LayeredModel(layers=layers, hidden_size=512, vocab_size=4096)
    dev = make_device(peak=1e14, bw=2e12)
    work = [SequenceWork(0, 96, 4), SequenceWork(0, 48, 2)]
    sched = simulate(model, dev, work)
    assert sched.makespan == pytest.approx(reference_layered(model, dev, work), rel=1e-9)


# --- DeepSeek-V4-Pro (MLA + CSA/HCA hybrid + output LoRA + MoE) ------------------


@pytest.fixture(scope="module")
def deepseek_v4() -> LayeredModel:
    return load_model_config(MODELS_DIR / "deepseek-v4-pro.json")


def test_deepseek_v4_loads(deepseek_v4):
    assert deepseek_v4.num_layers == 61
    assert deepseek_v4.num_moe_layers == 58
    dense_layers = [l for l in deepseek_v4.layers if isinstance(l.ffn, DenseFFN)]
    assert len(dense_layers) == 3  # first_k_dense_replace
    hca = [l for l in deepseek_v4.layers if l.mixer.kv_compression_ratio == 128]
    csa = [l for l in deepseek_v4.layers if l.mixer.kv_compression_ratio == 4]
    assert len(hca) == 31 and len(csa) == 29  # + 1 full = 61
    for layer in deepseek_v4.layers:
        assert layer.mixer.is_mla
        assert layer.mixer.o_lora_rank == 1024 and layer.mixer.o_groups == 16
    # HCA layers are heavily compressed + locally windowed; CSA layers run the indexer
    hca_attn = hca[0].mixer
    assert hca_attn.sliding_window == 128 and hca_attn.sparse_attention is False
    csa_attn = csa[0].mixer
    assert csa_attn.sparse_attention is True and csa_attn.sparse_topk == 1024
    # compression slashes the KV cache footprint
    uncompressed = (csa_attn.kv_lora_rank + csa_attn.qk_rope_head_dim) * 2
    assert hca_attn.kv_bytes_per_token(2) == uncompressed / 128
    assert csa_attn.kv_bytes_per_token(2) == uncompressed / 4
    moe = next(l.ffn for l in deepseek_v4.layers if isinstance(l.ffn, MoEFFN))
    assert moe.num_experts == 384 and moe.num_experts_per_token == 6
    assert moe.num_shared_experts == 1


def test_deepseek_v4_event_generation_matches_reference(deepseek_v4):
    dev = make_device()
    work = [SequenceWork(0, 64, 4), SequenceWork(0, 32, 2)]
    sched = simulate(deepseek_v4, dev, work)
    assert sched.makespan == pytest.approx(
        reference_layered(deepseek_v4, dev, work), rel=1e-9
    )


def test_deepseek_v4_event_generation_matches_reference_chunked(deepseek_v4):
    dev = make_device()
    work = [SequenceWork(0, 160, 3)]
    sched = simulate(deepseek_v4, dev, work, prefill_chunk_size=48)
    assert sched.makespan == pytest.approx(
        reference_layered(deepseek_v4, dev, work, prefill_chunk_size=48), rel=1e-9
    )


# --- DeepSeek-V4-Flash (smaller MLA + CSA/HCA hybrid, full-attn endpoints) -------


@pytest.fixture(scope="module")
def deepseek_v4_flash() -> LayeredModel:
    return load_model_config(MODELS_DIR / "deepseek-v4-flash.json")


def test_deepseek_v4_flash_loads(deepseek_v4_flash):
    assert deepseek_v4_flash.num_layers == 43
    assert deepseek_v4_flash.num_moe_layers == 40
    dense_layers = [l for l in deepseek_v4_flash.layers if isinstance(l.ffn, DenseFFN)]
    assert len(dense_layers) == 3  # first_k_dense_replace
    full = [l for l in deepseek_v4_flash.layers if l.mixer.kv_compression_ratio == 1]
    hca = [l for l in deepseek_v4_flash.layers if l.mixer.kv_compression_ratio == 128]
    csa = [l for l in deepseek_v4_flash.layers if l.mixer.kv_compression_ratio == 4]
    assert len(full) == 3 and len(hca) == 20 and len(csa) == 20  # 43 total
    # the first two layers and the last layer run uncompressed full attention
    assert [l.mixer.kv_compression_ratio for l in deepseek_v4_flash.layers[:2]] == [1, 1]
    assert deepseek_v4_flash.layers[-1].mixer.kv_compression_ratio == 1
    for layer in deepseek_v4_flash.layers:
        assert layer.mixer.is_mla
        assert layer.mixer.num_query_heads == 64
        assert layer.mixer.o_lora_rank == 1024 and layer.mixer.o_groups == 8
    hca_attn = hca[0].mixer
    assert hca_attn.sliding_window == 128 and hca_attn.sparse_attention is False
    csa_attn = csa[0].mixer
    assert csa_attn.sparse_attention is True and csa_attn.sparse_topk == 512
    uncompressed = (csa_attn.kv_lora_rank + csa_attn.qk_rope_head_dim) * 2
    assert hca_attn.kv_bytes_per_token(2) == uncompressed / 128
    assert csa_attn.kv_bytes_per_token(2) == uncompressed / 4
    moe = next(l.ffn for l in deepseek_v4_flash.layers if isinstance(l.ffn, MoEFFN))
    assert moe.num_experts == 256 and moe.num_experts_per_token == 6
    assert moe.num_shared_experts == 1


def test_deepseek_v4_flash_event_generation_matches_reference(deepseek_v4_flash):
    dev = make_device()
    work = [SequenceWork(0, 64, 4), SequenceWork(0, 32, 2)]
    sched = simulate(deepseek_v4_flash, dev, work)
    assert sched.makespan == pytest.approx(
        reference_layered(deepseek_v4_flash, dev, work), rel=1e-9
    )


def test_deepseek_v4_flash_event_generation_matches_reference_chunked(deepseek_v4_flash):
    dev = make_device()
    work = [SequenceWork(0, 160, 3)]
    sched = simulate(deepseek_v4_flash, dev, work, prefill_chunk_size=48)
    assert sched.makespan == pytest.approx(
        reference_layered(deepseek_v4_flash, dev, work, prefill_chunk_size=48), rel=1e-9
    )


# --- GLM-5.2 IndexShare (DSA indexer reused across every 4 layers) ---------------


def _dsa(**overrides):
    kwargs = dict(
        hidden_size=512,
        attention_type="MLA",
        num_query_heads=8,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_rope_head_dim=16,
        qk_nope_head_dim=32,
        v_head_dim=32,
        sparse_attention=True,
        sparse_topk=16,
        index_n_heads=4,
        index_head_dim=16,
    )
    kwargs.update(overrides)
    return Attention(**kwargs)


def test_indexer_shared_skips_indexer_cost():
    full = _dsa()
    shared = _dsa(indexer_shared=True)
    contexts = [256, 128]
    f_full, b_full = full.decode_cost(contexts, 1, 2)
    f_shared, b_shared = shared.decode_cost(contexts, 1, 2)
    # the shared layer keeps the (identical) top-k main attention but drops the
    # indexer projection + candidate-scoring FLOPs and the index-KV reads
    assert f_shared < f_full
    assert b_shared < b_full
    # the saving is exactly the indexer terms (no window -> candidates = full context)
    batch = len(contexts)
    cand = sum(contexts)
    idx_flops = 2.0 * batch * full._indexer_proj_params + full._indexer_per_pair * cand
    assert f_full - f_shared == pytest.approx(idx_flops)


def test_indexer_shared_requires_sparse_attention():
    with pytest.raises(ValueError):
        Attention(hidden_size=512, attention_type="GQA", num_query_heads=8,
                  num_kv_heads=8, head_dim=64, indexer_shared=True)


def test_synthetic_indexshare_matches_reference():
    full = _dsa()
    shared = _dsa(indexer_shared=True)
    layers = tuple(
        Layer(
            mixer=full if i % 4 == 0 else shared,
            ffn=DenseFFN(512, 1024, gated=True),
            name=f"l{i}",
        )
        for i in range(8)
    )
    model = LayeredModel(layers=layers, hidden_size=512, vocab_size=4096)
    dev = make_device(peak=1e14, bw=2e12)
    work = [SequenceWork(0, 96, 4), SequenceWork(0, 48, 2)]
    sched = simulate(model, dev, work)
    assert sched.makespan == pytest.approx(reference_layered(model, dev, work), rel=1e-9)


@pytest.fixture(scope="module")
def glm52() -> LayeredModel:
    return load_model_config(MODELS_DIR / "glm-5.2.json")


def test_glm52_loads(glm52):
    assert glm52.num_layers == 78
    assert glm52.num_moe_layers == 75
    dense_layers = [l for l in glm52.layers if isinstance(l.ffn, DenseFFN)]
    assert len(dense_layers) == 3  # first_k_dense_replace
    full_idx = [l for l in glm52.layers if not l.mixer.indexer_shared]
    shared_idx = [l for l in glm52.layers if l.mixer.indexer_shared]
    # IndexShare: ~1 in 4 layers computes its own index (here 3 dense + 19 MoE)
    assert len(full_idx) == 22 and len(shared_idx) == 56
    for layer in glm52.layers:
        assert layer.mixer.is_mla
        assert layer.mixer.sparse_attention is True
        assert layer.mixer.sparse_topk == 2048
        assert layer.mixer.num_query_heads == 64
        assert layer.mixer.qk_nope_head_dim == 192 and layer.mixer.v_head_dim == 256
        assert layer.mixer.o_lora_rank is None  # GLM-5.2 keeps a full output proj
    moe = next(l.ffn for l in glm52.layers if isinstance(l.ffn, MoEFFN))
    assert moe.num_experts == 256 and moe.num_experts_per_token == 8
    assert moe.num_shared_experts == 1


def test_glm52_event_generation_matches_reference(glm52):
    dev = make_device()
    work = [SequenceWork(0, 64, 4), SequenceWork(0, 32, 2)]
    sched = simulate(glm52, dev, work)
    assert sched.makespan == pytest.approx(reference_layered(glm52, dev, work), rel=1e-9)


def test_glm52_event_generation_matches_reference_chunked(glm52):
    dev = make_device()
    work = [SequenceWork(0, 160, 3)]
    sched = simulate(glm52, dev, work, prefill_chunk_size=48)
    assert sched.makespan == pytest.approx(
        reference_layered(glm52, dev, work, prefill_chunk_size=48), rel=1e-9
    )




