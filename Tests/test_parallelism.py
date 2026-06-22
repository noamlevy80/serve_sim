"""Unit tests for the parallelism planner (Stage 5, step 2).

The planner factors a fixed engine ``degree`` into a ``(pp, ep)`` arrangement,
estimates each option's roofline time, and rejects arrangements whose per-device
footprint exceeds memory. The reference below re-derives the footprint partition
by hand from the trusted block-level weight/KV quantities, so it exercises the
planner's *partition and selection* logic independently of the byte formulas.
"""

from __future__ import annotations

import math

import pytest

from serve_sim import ParallelismChoice, ParallelismPlanner
from serve_sim.blocks import (
    Attention,
    DenseFFN,
    Layer,
    LayeredModel,
    MoEFFN,
)
from serve_sim.hardware import ComputeDevice, MemoryDevice


# --- helpers -------------------------------------------------------------------


def make_device(cap=80e9, bw=1e12, peak=100e12):
    mem = MemoryDevice("hbm", capacity_bytes=cap, bandwidth_bytes_per_s=bw)
    return ComputeDevice("gpu", peak_flops_fp16=peak, first_tier_memory=mem)


def _attn():
    return Attention(
        hidden_size=64, attention_type="MHA", num_query_heads=4, head_dim=16
    )


def dense_model(num_layers=8, hidden=64, vocab=512):
    layers = tuple(
        Layer(mixer=_attn(), ffn=DenseFFN(hidden, hidden * 4, gated=False))
        for _ in range(num_layers)
    )
    return LayeredModel(layers=layers, hidden_size=hidden, vocab_size=vocab)


def moe_model(num_layers=4, hidden=64, vocab=512, num_experts=8):
    layers = tuple(
        Layer(
            mixer=_attn(),
            ffn=MoEFFN(
                hidden,
                hidden * 2,
                num_experts=num_experts,
                num_experts_per_token=2,
                gated=False,
            ),
        )
        for _ in range(num_layers)
    )
    return LayeredModel(layers=layers, hidden_size=hidden, vocab_size=vocab)


def ref_footprint(model, pp, ep, kv_tokens):
    """Independent re-derivation of the planner's per-device peak footprint."""
    model = LayeredModel.from_model(model)
    pdt = model.param_dtype_bytes
    kdt = model.kv_dtype_bytes
    lps = model.num_layers // pp
    peak = 0.0
    for stage in range(pp):
        stage_layers = model.layers[stage * lps : (stage + 1) * lps]
        replicated = 0.0
        experts = 0.0
        for layer in stage_layers:
            mixer_p = layer.mixer.weight_params if layer.mixer is not None else 0
            if isinstance(layer.ffn, MoEFFN):
                nonexp = layer.ffn.shared_expert_params + layer.ffn.latent_proj_params
                experts += (
                    math.ceil(layer.ffn.num_experts / ep)
                    * layer.ffn.routed_expert_params
                    * pdt
                )
            elif isinstance(layer.ffn, DenseFFN):
                nonexp = layer.ffn.weight_params
            else:
                nonexp = 0
            replicated += (mixer_p + nonexp) * pdt
            replicated += layer.kv_bytes_per_token(kdt) * kv_tokens
        if stage == pp - 1:
            replicated += model.lm_head_bytes
        peak = max(peak, replicated + experts)
    return peak


# --- factorization enumeration -------------------------------------------------


def test_factorizations_sorted_fastest_first():
    planner = ParallelismPlanner(dense_model(num_layers=8), make_device())
    # degree 4 -> pp in {1,2,4} (all divide 8 layers), ep descending.
    assert planner.factorizations(4) == [(1, 4), (2, 2), (4, 1)]


def test_factorizations_filtered_by_layer_divisibility():
    # 4 layers: pp must divide 4. degree 6 divisors are 1,2,3,6 -> keep 1,2.
    planner = ParallelismPlanner(dense_model(num_layers=4), make_device())
    assert planner.factorizations(6) == [(1, 6), (2, 3)]


def test_factorizations_degree_one():
    planner = ParallelismPlanner(dense_model(num_layers=4), make_device())
    assert planner.factorizations(1) == [(1, 1)]


def test_factorizations_rejects_bad_degree():
    planner = ParallelismPlanner(dense_model(), make_device())
    with pytest.raises(ValueError):
        planner.factorizations(0)


# --- footprint -----------------------------------------------------------------


@pytest.mark.parametrize("pp,ep", [(1, 4), (2, 2), (4, 1)])
def test_footprint_matches_reference_dense(pp, ep):
    model = dense_model(num_layers=8)
    planner = ParallelismPlanner(model, make_device())
    assert planner.footprint(pp, ep, kv_tokens=100) == pytest.approx(
        ref_footprint(model, pp, ep, 100)
    )


@pytest.mark.parametrize("pp,ep", [(1, 8), (2, 4), (4, 2)])
def test_footprint_matches_reference_moe(pp, ep):
    model = moe_model(num_layers=4, num_experts=8)
    planner = ParallelismPlanner(model, make_device())
    assert planner.footprint(pp, ep, kv_tokens=50) == pytest.approx(
        ref_footprint(model, pp, ep, 50)
    )


def test_ep_does_not_relieve_dense_footprint():
    # Pure expert parallelism replicates dense/attention/KV across ranks, so for
    # a dense model the footprint depends only on pp, not ep.
    model = dense_model(num_layers=8)
    planner = ParallelismPlanner(model, make_device())
    assert planner.footprint(1, 2, 100) == planner.footprint(1, 4, 100)
    assert planner.footprint(2, 1, 100) == planner.footprint(2, 2, 100)


def test_pp_relieves_footprint():
    # More pipeline stages -> fewer layers per device -> smaller footprint.
    model = dense_model(num_layers=8)
    planner = ParallelismPlanner(model, make_device())
    assert planner.footprint(2, 1, 100) < planner.footprint(1, 1, 100)
    assert planner.footprint(4, 1, 100) < planner.footprint(2, 1, 100)


def test_ep_relieves_moe_expert_footprint():
    # Routed experts ARE sharded by ep, so more ep shrinks an MoE footprint.
    model = moe_model(num_layers=4, num_experts=8)
    planner = ParallelismPlanner(model, make_device())
    assert planner.footprint(1, 4, 0) < planner.footprint(1, 2, 0)
    assert planner.footprint(1, 2, 0) < planner.footprint(1, 1, 0)


def test_more_kv_tokens_increase_footprint():
    model = dense_model(num_layers=4)
    planner = ParallelismPlanner(model, make_device())
    assert planner.footprint(1, 1, 1000) > planner.footprint(1, 1, 10)


def test_footprint_rejects_indivisible_pp():
    planner = ParallelismPlanner(dense_model(num_layers=6), make_device())
    with pytest.raises(ValueError):
        planner.footprint(4, 1, 10)  # 4 does not divide 6


# --- speed estimate ------------------------------------------------------------


def test_estimate_scales_inversely_with_ep():
    device = make_device(peak=100e12, bw=1e12)
    planner = ParallelismPlanner(dense_model(), device)
    flops = {2: 200e12}
    base = planner.estimate_time(1, flops, total_bytes=0.0)
    assert planner.estimate_time(2, flops, 0.0) == pytest.approx(base / 2)
    assert planner.estimate_time(4, flops, 0.0) == pytest.approx(base / 4)


def test_estimate_takes_roofline_max():
    device = make_device(peak=100e12, bw=1e12)
    planner = ParallelismPlanner(dense_model(), device)
    # compute-bound: 200e12 / 100e12 = 2s ; bandwidth tiny -> max is compute.
    assert planner.estimate_time(1, {2: 200e12}, total_bytes=1.0) == pytest.approx(2.0)
    # bandwidth-bound: 2e12 / 1e12 = 2s ; compute tiny -> max is bandwidth.
    assert planner.estimate_time(1, {2: 1.0}, total_bytes=2e12) == pytest.approx(2.0)


def test_estimate_uses_dtype_scaled_flops():
    device = make_device(peak=100e12, bw=1e12)
    planner = ParallelismPlanner(dense_model(), device)
    # fp8 (1 byte) runs at 2x: 200e12 / 200e12 = 1s.
    assert planner.estimate_time(1, {1: 200e12}, total_bytes=0.0) == pytest.approx(1.0)


# --- planning ------------------------------------------------------------------


def test_plan_picks_max_ep_when_everything_fits():
    model = dense_model(num_layers=8)
    planner = ParallelismPlanner(model, make_device(cap=80e9))
    choice = planner.plan(4, kv_tokens=100, flops_by_dtype={2: 1e12}, total_bytes=1e9)
    assert isinstance(choice, ParallelismChoice)
    assert (choice.pipeline_parallel, choice.expert_parallel) == (1, 4)
    assert choice.per_device_bytes == pytest.approx(ref_footprint(model, 1, 4, 100))


def test_plan_falls_back_to_more_pipeline_when_tight():
    model = dense_model(num_layers=8)
    # Size capacity so pp=1 overflows but pp=2 (half the layers) fits.
    big = ParallelismPlanner(model, make_device()).footprint(1, 4, 100)
    small = ParallelismPlanner(model, make_device()).footprint(2, 2, 100)
    cap = (big + small) / 2
    planner = ParallelismPlanner(model, make_device(cap=cap))
    choice = planner.plan(4, kv_tokens=100, flops_by_dtype={2: 1e12}, total_bytes=1e9)
    assert (choice.pipeline_parallel, choice.expert_parallel) == (2, 2)


def test_plan_raises_when_nothing_fits():
    model = dense_model(num_layers=8)
    planner = ParallelismPlanner(model, make_device(cap=1.0))
    with pytest.raises(ValueError, match="no parallelism arrangement"):
        planner.plan(4, kv_tokens=100, flops_by_dtype={2: 1e12}, total_bytes=1e9)


def test_plan_reports_estimated_time():
    model = dense_model(num_layers=8)
    device = make_device(cap=80e9, peak=100e12, bw=1e12)
    planner = ParallelismPlanner(model, device)
    choice = planner.plan(
        4, kv_tokens=100, flops_by_dtype={2: 200e12}, total_bytes=0.0
    )
    # ep=4 chosen -> 2s compute / 4.
    assert choice.estimated_time == pytest.approx(0.5)


def test_plan_degree_one_returns_trivial():
    model = dense_model(num_layers=4)
    planner = ParallelismPlanner(model, make_device())
    choice = planner.plan(1, kv_tokens=10, flops_by_dtype={2: 1e12}, total_bytes=1e9)
    assert (choice.pipeline_parallel, choice.expert_parallel) == (1, 1)


# --- tensor parallelism --------------------------------------------------------


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_tensor_parallel_divides_footprint(tp):
    # TP shards every tensor AND the KV cache, so the whole per-device footprint
    # scales as 1/tp (unlike ep, which shards only routed experts).
    model = moe_model(num_layers=4, num_experts=8)
    planner = ParallelismPlanner(model, make_device())
    base = planner.footprint(1, 1, kv_tokens=50)
    assert planner.footprint(1, 1, kv_tokens=50, tensor_parallel=tp) == pytest.approx(
        base / tp
    )


def test_tensor_parallel_relieves_dense_footprint_unlike_ep():
    # A dense model: ep does nothing, but tp still shrinks the footprint.
    model = dense_model(num_layers=8)
    planner = ParallelismPlanner(model, make_device())
    base = planner.footprint(1, 1, 100)
    assert planner.footprint(1, 4, 100) == pytest.approx(base)  # ep inert on dense
    assert planner.footprint(1, 1, 100, tensor_parallel=2) == pytest.approx(base / 2)


def test_tensor_parallel_rejects_zero():
    planner = ParallelismPlanner(dense_model(num_layers=4), make_device())
    with pytest.raises(ValueError):
        planner.footprint(1, 1, 10, tensor_parallel=0)


def test_estimate_scales_inversely_with_ep_times_tp():
    device = make_device(peak=100e12, bw=1e12)
    planner = ParallelismPlanner(dense_model(), device)
    flops = {2: 200e12}
    base = planner.estimate_time(1, flops, total_bytes=0.0)
    assert planner.estimate_time(2, flops, 0.0, tensor_parallel=2) == pytest.approx(
        base / 4
    )
    assert planner.estimate_time(1, flops, 0.0, tensor_parallel=4) == pytest.approx(
        base / 4
    )


def test_plan_holds_tensor_parallel_fixed_and_refactors_pp_ep():
    model = dense_model(num_layers=8)
    planner = ParallelismPlanner(model, make_device(cap=80e9))
    # The pp*ep budget here is 4; tp=2 is held fixed and applied to the footprint.
    choice = planner.plan(
        4, kv_tokens=100, flops_by_dtype={2: 1e12}, total_bytes=1e9, tensor_parallel=2
    )
    assert isinstance(choice, ParallelismChoice)
    assert (choice.pipeline_parallel, choice.expert_parallel) == (1, 4)
    assert choice.tensor_parallel == 2
    assert choice.per_device_bytes == pytest.approx(
        ref_footprint(model, 1, 4, 100) / 2
    )


def test_plan_tensor_parallel_makes_a_tight_batch_fit():
    model = dense_model(num_layers=8)
    # The smallest tp=1 footprint over the degree-4 budget is the most-pipelined
    # pp=4 arrangement; a cap below it cannot place the batch at any (pp, ep).
    min_fp = ParallelismPlanner(model, make_device()).footprint(4, 1, 100)
    planner = ParallelismPlanner(model, make_device(cap=min_fp * 0.75))
    with pytest.raises(ValueError, match="no parallelism arrangement"):
        planner.plan(4, kv_tokens=100, flops_by_dtype={2: 1e12}, total_bytes=1e9)
    # tp=2 halves every arrangement's footprint, so the batch now fits.
    choice = planner.plan(
        4, kv_tokens=100, flops_by_dtype={2: 1e12}, total_bytes=1e9, tensor_parallel=2
    )
    assert choice.tensor_parallel == 2
    assert choice.per_device_bytes <= planner.capacity

