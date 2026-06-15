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
