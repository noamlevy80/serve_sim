"""Work shards: atomic, device-executable pieces of a forward pass.

A work shard carries the FLOPs and bytes-read needed for one layer of one
phase-unit (a prefill chunk or a single decode step), plus a ``group_index`` so
the event generator knows which shards belong to the same forward pass and may
be consolidated when they land on the same device.

Shard sizing is pure model arithmetic; see :mod:`serve_sim.model`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Model
from .tracker import SequenceWork


@dataclass(frozen=True)
class WorkShard:
    """One atomic unit of compute work.

    Attributes:
        group_index: Forward-pass group; shards sharing it run "together" and
            may be consolidated per device. Groups execute in ascending order.
        phase: ``"prefill"`` or ``"decode"``.
        layer_index: Transformer layer index, or ``None`` for the LM head
            (which is placed on the last pipeline stage).
        flops: Floating point operations for this shard.
        bytes_read: Bytes that must be read from first-tier memory.
        flops_dtype_bytes: Operand element size (sets the compute-rate scale).
        kind: ``"layer"`` or ``"lm_head"`` (informational).
        tokens: Number of tokens processed (informational).
    """

    group_index: int
    phase: str
    layer_index: int | None
    flops: float
    bytes_read: float
    flops_dtype_bytes: int
    kind: str
    tokens: int


def _causal_pairs(start: int, stop: int, cached: int) -> int:
    """Query-key pairs for new query positions ``[start, stop)`` over ``cached``.

    Query position ``j`` (0-based within the new prompt tokens) attends to
    ``cached + j + 1`` keys (the cached prefix, earlier new tokens, and itself).
    """

    count = stop - start
    triangular = stop * (stop + 1) // 2 - start * (start + 1) // 2
    return count * cached + triangular


class WorkShardGenerator:
    """Turns per-sequence work into ordered work shards for one turn."""

    def __init__(self, model: Model) -> None:
        self.model = model

    def generate(
        self,
        batch_work: list[SequenceWork],
        prefill_chunk_size: int | None = None,
    ) -> list[WorkShard]:
        """Generate all shards for a batch's turn.

        Prefill is emitted per sequence (optionally chunked); decode is emitted
        as batched lockstep steps that share a single weight read per step.
        """

        if not batch_work:
            raise ValueError("batch_work must contain at least one sequence")
        if prefill_chunk_size is not None and prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be >= 1")

        shards: list[WorkShard] = []
        group_index = 0
        group_index = self._emit_prefill(shards, batch_work, prefill_chunk_size, group_index)
        self._emit_decode(shards, batch_work, group_index)
        return shards

    # --- prefill ------------------------------------------------------------

    def _emit_prefill(
        self,
        shards: list[WorkShard],
        batch_work: list[SequenceWork],
        prefill_chunk_size: int | None,
        group_index: int,
    ) -> int:
        model = self.model
        for seq in batch_work:
            if seq.prefill_tokens == 0:
                continue
            chunk = prefill_chunk_size or seq.prefill_tokens
            start = 0
            while start < seq.prefill_tokens:
                stop = min(start + chunk, seq.prefill_tokens)
                tokens = stop - start
                pairs = _causal_pairs(start, stop, seq.cached_tokens)
                # KV already materialized before this chunk and read by attention.
                prior_kv_tokens = seq.cached_tokens + start
                for layer in range(model.num_layers):
                    flops = model.linear_flops(tokens) + model.attention_flops(pairs)
                    bytes_read = (
                        model.layer_weight_bytes
                        + prior_kv_tokens * model.kv_bytes_per_token
                    )
                    shards.append(
                        WorkShard(
                            group_index=group_index,
                            phase="prefill",
                            layer_index=layer,
                            flops=flops,
                            bytes_read=bytes_read,
                            flops_dtype_bytes=model.param_dtype_bytes,
                            kind="layer",
                            tokens=tokens,
                        )
                    )
                group_index += 1
                start = stop
        return group_index

    # --- decode -------------------------------------------------------------

    def _emit_decode(
        self,
        shards: list[WorkShard],
        batch_work: list[SequenceWork],
        group_index: int,
    ) -> int:
        model = self.model
        max_steps = max(seq.decode_tokens for seq in batch_work)
        for step in range(1, max_steps + 1):
            active = [seq for seq in batch_work if seq.decode_tokens >= step]
            if not active:
                continue
            batch_size = len(active)
            # context length seen by each active sequence at this step
            contexts = [seq.base_tokens + step for seq in active]
            total_context = sum(contexts)
            for layer in range(model.num_layers):
                flops = model.linear_flops(batch_size) + model.attention_flops(total_context)
                bytes_read = (
                    model.layer_weight_bytes
                    + total_context * model.kv_bytes_per_token
                )
                shards.append(
                    WorkShard(
                        group_index=group_index,
                        phase="decode",
                        layer_index=layer,
                        flops=flops,
                        bytes_read=bytes_read,
                        flops_dtype_bytes=model.param_dtype_bytes,
                        kind="layer",
                        tokens=batch_size,
                    )
                )
            if model.lm_head_params > 0:
                shards.append(
                    WorkShard(
                        group_index=group_index,
                        phase="decode",
                        layer_index=None,
                        flops=model.lm_head_flops(batch_size),
                        bytes_read=model.lm_head_bytes,
                        flops_dtype_bytes=model.param_dtype_bytes,
                        kind="lm_head",
                        tokens=batch_size,
                    )
                )
            group_index += 1
        return group_index
