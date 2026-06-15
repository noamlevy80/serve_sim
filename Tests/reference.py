"""Independent closed-form roofline reference for the toy model.

This re-derives the expected makespan from first principles (separate from the
shard/event machinery) so the roofline tests verify the simulator rather than
mirror it. Keep these formulas aligned with the documented model in Project.md.
"""

from __future__ import annotations

from serve_sim.model import Model
from serve_sim.hardware import ComputeDevice
from serve_sim.tracker import SequenceWork


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
    lw_params = model.layer_weight_params
    lw_bytes = lw_params * model.param_dtype_bytes
    q_dim = model.q_dim
    kv_per_tok = model.kv_bytes_per_token

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
            flops = L * (2 * tokens * lw_params + 4 * q_dim * pairs)
            bytes_read = L * (lw_bytes + prior_kv * kv_per_tok)
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
        flops = (
            L * (2 * batch_size * lw_params + 4 * q_dim * total_context)
            + 2 * batch_size * model.lm_head_params
        )
        bytes_read = L * (lw_bytes + total_context * kv_per_tok) + model.lm_head_bytes
        total += max(flops / eff_peak, bytes_read / bw)

    return total
