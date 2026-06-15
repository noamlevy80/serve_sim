"""Independent closed-form roofline reference for the toy model.

This re-derives the expected makespan from first principles (separate from the
shard/event machinery) so the roofline tests verify the simulator rather than
mirror it. Keep these formulas aligned with the documented model in Project.md.

Supports dense and MoE FFNs. For MoE, the expected number of distinct routed
experts touched by a group is re-derived here independently of
``ExpertUsageModel``.
"""

from __future__ import annotations

from serve_sim.model import Model
from serve_sim.hardware import ComputeDevice
from serve_sim.tracker import SequenceWork
from serve_sim.tiering import GroupActivation


def _expected_distinct(model: Model, num_tokens: int, consecutive: bool) -> float:
    """Independent re-derivation of expected distinct routed experts."""

    if num_tokens <= 0:
        return 0.0
    k = model.num_experts_per_token
    e = model.num_experts
    if consecutive:
        picks = k * (1.0 + (num_tokens - 1) / model.expert_persistence_mean)
    else:
        picks = float(k * num_tokens)
    return e * (1.0 - (1.0 - 1.0 / e) ** picks)


def _ffn_cost(model: Model, layer_index: int, tokens: int, consecutive: bool):
    """(flops, bytes) for one layer's FFN, dense or MoE."""

    if not model.is_moe_layer(layer_index):
        return float(model.dense_ffn_flops(tokens)), float(model.dense_ffn_bytes)
    distinct = _expected_distinct(model, tokens, consecutive)
    routed_flops = 2 * tokens * model.num_experts_per_token * model.routed_expert_params
    routed_bytes = distinct * model.routed_expert_bytes
    shared_flops = 2 * tokens * model.shared_expert_params
    shared_bytes = float(model.shared_expert_bytes)
    return routed_flops + shared_flops, routed_bytes + shared_bytes


def reference_roofline(
    model: Model,
    device: ComputeDevice,
    batch_work: list[SequenceWork],
    prefill_chunk_size: int | None = None,
) -> float:
    """Total roofline makespan on a single device for one turn."""

    scale = 2.0 / model.param_dtype_bytes
    eff_peak = device.peak_flops_fp16 * scale
    bw = device.first_tier_memory.bandwidth_bytes_per_s
    L = model.num_layers
    q_dim = model.q_dim
    kv_per_tok = model.kv_bytes_per_token
    attn_bytes = model.attention_weight_bytes

    total = 0.0

    # Prefill: per sequence, chunked.
    for seq in batch_work:
        if seq.prefill_tokens == 0:
            continue
        chunk = prefill_chunk_size or seq.prefill_tokens
        start = 0
        while start < seq.prefill_tokens:
            stop = min(start + chunk, seq.prefill_tokens)
            tokens = stop - start
            triangular = stop * (stop + 1) // 2 - start * (start + 1) // 2
            pairs = tokens * seq.cached_tokens + triangular
            prior_kv = seq.cached_tokens + start
            flops = 0.0
            bytes_read = 0.0
            for layer in range(L):
                ffn_flops, ffn_bytes = _ffn_cost(model, layer, tokens, consecutive=True)
                flops += (
                    2 * tokens * model.attention_weight_params
                    + 4 * q_dim * pairs
                    + ffn_flops
                )
                bytes_read += attn_bytes + prior_kv * kv_per_tok + ffn_bytes
            total += max(flops / eff_peak, bytes_read / bw)
            start = stop

    # Decode: batched lockstep steps.
    max_steps = max(seq.decode_tokens for seq in batch_work)
    for step in range(1, max_steps + 1):
        active = [seq for seq in batch_work if seq.decode_tokens >= step]
        if not active:
            continue
        batch_size = len(active)
        total_context = sum(seq.base_tokens + step for seq in active)
        flops = 0.0
        bytes_read = 0.0
        for layer in range(L):
            ffn_flops, ffn_bytes = _ffn_cost(model, layer, batch_size, consecutive=False)
            flops += (
                2 * batch_size * model.attention_weight_params
                + 4 * q_dim * total_context
                + ffn_flops
            )
            bytes_read += attn_bytes + total_context * kv_per_tok + ffn_bytes
        flops += 2 * batch_size * model.lm_head_params
        bytes_read += model.lm_head_bytes
        total += max(flops / eff_peak, bytes_read / bw)

    return total


def reference_transfer_time(
    model: Model,
    device: ComputeDevice,
    trace: list[GroupActivation],
    capacity: int,
) -> float:
    """Independent re-derivation of total expert-movement transfer time.

    Replays the activation trace through a simple LRU residency cache (written
    independently of ``ExpertResidencyCache``) and charges each group's misses
    at the slower of the two tiers' bandwidths.
    """

    tier2 = device.second_tier_memory
    assert tier2 is not None
    bandwidth = min(device.first_tier_memory.bandwidth_bytes_per_s, tier2.bandwidth_bytes_per_s)
    moe_layers = model.num_moe_layers

    resident: list[int] = []  # LRU order, oldest first
    total_time = 0.0
    for group in trace:
        active = sorted(group.active_experts)
        if len(active) > capacity:
            raise ValueError("first tier too small for active set")
        misses = 0
        for idx in active:
            if idx in resident:
                resident.remove(idx)
                resident.append(idx)
            else:
                misses += 1
                resident.append(idx)
        while len(resident) > capacity:
            resident.pop(0)
        if misses:
            bytes_moved = misses * moe_layers * model.routed_expert_bytes
            total_time += bytes_moved / bandwidth
    return total_time


def reference_two_tier(
    model: Model,
    device: ComputeDevice,
    batch_work: list[SequenceWork],
    trace: list[GroupActivation],
    capacity: int,
    prefill_chunk_size: int | None = None,
) -> float:
    """Total two-tier makespan: compute roofline plus expert-movement time."""

    return reference_roofline(model, device, batch_work, prefill_chunk_size) + (
        reference_transfer_time(model, device, trace, capacity)
    )


# --- independent reference for heterogeneous (layered) models -------------------

from serve_sim.blocks import Attention, DenseFFN, LayeredModel, MambaBlock, MoEFFN  # noqa: E402

_INF = float("inf")


def _ref_distinct(ffn: MoEFFN, num_tokens: int, consecutive: bool, mean: float) -> float:
    if num_tokens <= 0:
        return 0.0
    k = ffn.num_experts_per_token
    e = ffn.num_experts
    picks = k * (1.0 + (num_tokens - 1) / mean) if consecutive else float(k * num_tokens)
    return e * (1.0 - (1.0 - 1.0 / e) ** picks)


def _ref_ffn_cost(
    ffn, tokens: int, consecutive: bool, pdb: int, mean: float
) -> tuple[float, float]:
    matrices = 3 if ffn.gated else 2
    if isinstance(ffn, DenseFFN):
        wp = matrices * ffn.hidden_size * ffn.intermediate_size
        return 2.0 * tokens * wp, float(wp * pdb)
    # MoE
    expert_width = ffn.moe_latent_size or ffn.hidden_size
    routed_ep = matrices * expert_width * ffn.intermediate_size
    shared_width = ffn.shared_expert_intermediate_size or ffn.intermediate_size
    shared = ffn.num_shared_experts * matrices * ffn.hidden_size * shared_width
    latent = 2 * ffn.hidden_size * ffn.moe_latent_size if ffn.moe_latent_size else 0
    distinct = _ref_distinct(ffn, tokens, consecutive, mean)
    flops = 2.0 * tokens * ffn.num_experts_per_token * routed_ep + 2.0 * tokens * shared + 2.0 * tokens * latent
    bytes_read = distinct * routed_ep * pdb + shared * pdb + latent * pdb
    return flops, float(bytes_read)


def _ref_attn_dims(attn: Attention) -> tuple[int, int]:
    """(q_dim, kv_dim) for GQA/MHA."""

    q_dim = attn.num_query_heads * attn.head_dim
    kv_dim = attn.num_kv_heads * attn.head_dim
    return q_dim, kv_dim


def _ref_attn_weight(attn) -> int:
    d = attn.hidden_size
    if attn.attention_type == "MLA":
        h = attn.num_query_heads
        return (
            d * attn.q_lora_rank
            + attn.q_lora_rank * h * (attn.qk_nope_head_dim + attn.qk_rope_head_dim)
            + d * (attn.kv_lora_rank + attn.qk_rope_head_dim)
            + attn.kv_lora_rank * h * attn.qk_nope_head_dim
            + attn.kv_lora_rank * h * attn.v_head_dim
            + h * attn.v_head_dim * d
        )
    q_dim, kv_dim = _ref_attn_dims(attn)
    return d * q_dim + 2 * d * kv_dim + q_dim * d


def _ref_attn_per_pair(attn) -> int:
    if attn.attention_type == "MLA":
        h = attn.num_query_heads
        return 2 * h * ((attn.qk_nope_head_dim + attn.qk_rope_head_dim) + attn.v_head_dim)
    q_dim, _ = _ref_attn_dims(attn)
    return 4 * q_dim


def _ref_attn_kv_bpt(attn, kvdb) -> int:
    if attn.attention_type == "MLA":
        return (attn.kv_lora_rank + attn.qk_rope_head_dim) * kvdb
    _, kv_dim = _ref_attn_dims(attn)
    return 2 * kv_dim * kvdb


def _ref_main_cand(attn, keys, window) -> tuple[float, float]:
    cand = min(float(keys), window)
    if attn.sparse_attention:
        return min(cand, float(attn.sparse_topk)), cand
    return cand, cand


def _ref_attn_prefill(attn, tokens, cached, start, pdb, kvdb) -> tuple[float, float]:
    wp = _ref_attn_weight(attn)
    per_pair = _ref_attn_per_pair(attn)
    kv_bpt = _ref_attn_kv_bpt(attn, kvdb)
    window = _INF if attn.sliding_window is None else float(attn.sliding_window)
    main_pairs = 0.0
    cand_pairs = 0.0
    for j in range(tokens):
        m, c = _ref_main_cand(attn, cached + start + j + 1, window)
        main_pairs += m
        cand_pairs += c
    prior_main, prior_cand = _ref_main_cand(attn, cached + start, window)
    flops = 2.0 * tokens * wp + per_pair * main_pairs
    bytes_read = wp * pdb + prior_main * kv_bpt
    if attn.sparse_attention:
        idx_proj = attn.hidden_size * attn.index_n_heads * attn.index_head_dim
        flops += 2.0 * tokens * idx_proj + 2 * attn.index_n_heads * attn.index_head_dim * cand_pairs
        bytes_read += idx_proj * pdb + prior_cand * attn.index_head_dim * kvdb
    return flops, float(bytes_read)


def _ref_attn_decode(attn, contexts, pdb, kvdb) -> tuple[float, float]:
    wp = _ref_attn_weight(attn)
    per_pair = _ref_attn_per_pair(attn)
    kv_bpt = _ref_attn_kv_bpt(attn, kvdb)
    window = _INF if attn.sliding_window is None else float(attn.sliding_window)
    main_sum = 0.0
    cand_sum = 0.0
    for c in contexts:
        m, cand = _ref_main_cand(attn, c, window)
        main_sum += m
        cand_sum += cand
    batch = len(contexts)
    flops = 2.0 * batch * wp + per_pair * main_sum
    bytes_read = wp * pdb + main_sum * kv_bpt
    if attn.sparse_attention:
        idx_proj = attn.hidden_size * attn.index_n_heads * attn.index_head_dim
        flops += 2.0 * batch * idx_proj + 2 * attn.index_n_heads * attn.index_head_dim * cand_sum
        bytes_read += idx_proj * pdb + cand_sum * attn.index_head_dim * kvdb
    return flops, float(bytes_read)


def _ref_mamba_cost(mixer, tokens, pdb) -> tuple[float, float]:
    d = mixer.hidden_size
    d_inner = mixer.num_heads * mixer.head_dim
    in_proj = d * (2 * d_inner + 2 * mixer.n_groups * mixer.d_state + mixer.num_heads)
    conv = (d_inner + 2 * mixer.n_groups * mixer.d_state) * mixer.d_conv
    out_proj = d_inner * d
    wp = in_proj + conv + out_proj
    ssm = 4 * d_inner * mixer.d_state
    flops = 2.0 * tokens * wp + tokens * ssm
    return flops, float(wp * pdb)


def reference_layered(
    model,
    device: ComputeDevice,
    batch_work: list[SequenceWork],
    prefill_chunk_size: int | None = None,
) -> float:
    """Independent roofline for a heterogeneous :class:`LayeredModel`.

    Re-derives per-layer costs from the block fields (not the block methods),
    supporting GQA/MHA attention (optionally sliding-window) and dense/MoE FFNs.
    """

    model = LayeredModel.from_model(model)
    pdb = model.param_dtype_bytes
    kvdb = model.kv_dtype_bytes
    mean = model.expert_persistence_mean
    scale = 2.0 / pdb
    eff_peak = device.peak_flops_fp16 * scale
    bw = device.first_tier_memory.bandwidth_bytes_per_s
    total = 0.0

    for seq in batch_work:
        if seq.prefill_tokens == 0:
            continue
        chunk = prefill_chunk_size or seq.prefill_tokens
        start = 0
        while start < seq.prefill_tokens:
            stop = min(start + chunk, seq.prefill_tokens)
            tokens = stop - start
            flops = 0.0
            bytes_read = 0.0
            for layer in model.layers:
                if layer.mixer is not None:
                    if isinstance(layer.mixer, MambaBlock):
                        f, b = _ref_mamba_cost(layer.mixer, tokens, pdb)
                    else:
                        f, b = _ref_attn_prefill(
                            layer.mixer, tokens, seq.cached_tokens, start, pdb, kvdb
                        )
                    flops += f
                    bytes_read += b
                if layer.ffn is not None:
                    f, b = _ref_ffn_cost(layer.ffn, tokens, True, pdb, mean)
                    flops += f
                    bytes_read += b
            total += max(flops / eff_peak, bytes_read / bw)
            start = stop

    max_steps = max(seq.decode_tokens for seq in batch_work)
    for step in range(1, max_steps + 1):
        active = [seq for seq in batch_work if seq.decode_tokens >= step]
        if not active:
            continue
        batch_size = len(active)
        contexts = [seq.base_tokens + step for seq in active]
        flops = 0.0
        bytes_read = 0.0
        for layer in model.layers:
            if layer.mixer is not None:
                if isinstance(layer.mixer, MambaBlock):
                    f, b = _ref_mamba_cost(layer.mixer, batch_size, pdb)
                else:
                    f, b = _ref_attn_decode(layer.mixer, contexts, pdb, kvdb)
                flops += f
                bytes_read += b
            if layer.ffn is not None:
                f, b = _ref_ffn_cost(layer.ffn, batch_size, False, pdb, mean)
                flops += f
                bytes_read += b
        flops += 2.0 * batch_size * model.lm_head_params
        bytes_read += model.lm_head_bytes
        total += max(flops / eff_peak, bytes_read / bw)

    return total


# --- independent reference for expert-parallel expert movement ------------------


def _moe_bytes_per_miss(model) -> int:
    """Bytes moved when one expert index misses: its routed weights across all
    MoE layers (works for both the flat ``Model`` and a ``LayeredModel``)."""

    m = LayeredModel.from_model(model)
    return sum(ffn.routed_expert_params for ffn in m.moe_ffns()) * m.param_dtype_bytes


def reference_ep_transfer(
    devices: list[ComputeDevice],
    model,
    trace: list[GroupActivation],
    capacity: int,
    expert_parallel: int,
    shared_tier2: bool,
) -> float:
    """Total expert-movement time under expert parallelism, re-derived.

    Expert ``e`` is owned by rank ``e % expert_parallel``; each rank replays its
    own LRU residency (capacity in expert indices) over the experts it owns. A
    group's movement runs the ranks concurrently: with a private second tier the
    group time is the slowest rank, with a shared second tier the aggregate is
    also bounded by that tier's bandwidth.
    """

    ep = expert_parallel
    bpm = _moe_bytes_per_miss(model)
    resident: list[list[int]] = [[] for _ in range(ep)]
    total_time = 0.0
    for group in trace:
        rank_bytes = [0.0] * ep
        for r in range(ep):
            active = sorted(e for e in group.active_experts if e % ep == r)
            if len(active) > capacity:
                raise ValueError("first tier too small for active set")
            misses = 0
            for idx in active:
                if idx in resident[r]:
                    resident[r].remove(idx)
                    resident[r].append(idx)
                else:
                    misses += 1
                    resident[r].append(idx)
            while len(resident[r]) > capacity:
                resident[r].pop(0)
            rank_bytes[r] = misses * bpm
        if not any(rank_bytes):
            continue
        if shared_tier2:
            tier2_bw = devices[0].second_tier_memory.bandwidth_bytes_per_s
            tier1_floor = max(
                (
                    rank_bytes[r] / devices[r].first_tier_memory.bandwidth_bytes_per_s
                    for r in range(ep)
                    if rank_bytes[r] > 0
                ),
                default=0.0,
            )
            total_time += max(sum(rank_bytes) / tier2_bw, tier1_floor)
        else:
            total_time += max(
                (
                    rank_bytes[r]
                    / min(
                        devices[r].first_tier_memory.bandwidth_bytes_per_s,
                        devices[r].second_tier_memory.bandwidth_bytes_per_s,
                    )
                    for r in range(ep)
                    if rank_bytes[r] > 0
                ),
                default=0.0,
            )
    return total_time
