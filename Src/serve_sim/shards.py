"""Work shards: atomic, device-executable pieces of a forward pass.

A work shard carries the FLOPs and bytes-read needed for one layer of one
phase-unit (a prefill chunk or a single decode step), plus a ``group_index`` so
the event generator knows which shards belong to the same forward pass and may
be consolidated when they land on the same device.

Shard sizing is pure model arithmetic; each layer's mixer (attention/Mamba) and
FFN block size themselves (see :mod:`serve_sim.blocks`).
"""

from __future__ import annotations

from dataclasses import dataclass

from .blocks import Layer, LayeredModel, MoEFFN
from .experts import ExpertUsageModel
from .tracker import SequenceWork


@dataclass(frozen=True)
class WorkShard:
    """One atomic unit of compute work.

    Attributes:
        group_index: Forward-pass group; shards sharing it run "together" and
            may be consolidated per device. Groups execute in ascending order.
        phase: ``"prefill"``, ``"decode"``, or ``"kernel_launch"`` (a launch
            marker that precedes a group's compute; carries no FLOPs/bytes).
        layer_index: Transformer layer index, or ``None`` for the LM head
            (which is placed on the last pipeline stage) and launch markers.
        flops: Floating point operations for this shard.
        bytes_read: Bytes that must be read from first-tier memory.
        flops_dtype_bytes: Operand element size (sets the compute-rate scale).
        kind: ``"layer"``, ``"lm_head"``, or ``"kernel_launch"`` (informational).
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


class WorkShardGenerator:
    """Turns per-sequence work into ordered work shards for one turn."""

    def __init__(self, model) -> None:
        self.model: LayeredModel = LayeredModel.from_model(model)
        self._expert_usage: dict[int, ExpertUsageModel] = {}

    def _kernel_launch_shard(self, group_index: int) -> WorkShard:
        """A zero-cost marker that a new kernel is launched for this group.

        The event generator charges the launching device's
        ``kernel_launch_latency`` once per group-stage when such a marker is
        present.
        """

        return WorkShard(
            group_index=group_index,
            phase="kernel_launch",
            layer_index=None,
            flops=0.0,
            bytes_read=0.0,
            flops_dtype_bytes=self.model.param_dtype_bytes,
            kind="kernel_launch",
            tokens=0,
        )

    def _distinct(self, ffn: MoEFFN, tokens: int, consecutive: bool) -> float:
        usage = self._expert_usage.get(id(ffn))
        if usage is None:
            usage = ExpertUsageModel(
                num_experts=ffn.num_experts,
                num_experts_per_token=ffn.num_experts_per_token,
                persistence_mean=self.model.expert_persistence_mean,
                persistence_variance=self.model.expert_persistence_variance,
            )
            self._expert_usage[id(ffn)] = usage
        return usage.expected_distinct(tokens, consecutive)

    def _ffn_cost(self, layer: Layer, tokens: int, consecutive: bool) -> tuple[float, float]:
        """FLOPs and bytes for one layer's FFN (dense or MoE), if any."""

        ffn = layer.ffn
        if ffn is None:
            return 0.0, 0.0
        if isinstance(ffn, MoEFFN):
            distinct = self._distinct(ffn, tokens, consecutive)
            return ffn.cost(tokens, distinct, self.model.param_dtype_bytes)
        return ffn.cost(tokens, self.model.param_dtype_bytes)

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
        group_index = self._emit_prefill(shards, batch_work, prefill_chunk_size, 0)
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
        pdb = model.param_dtype_bytes
        kvdb = model.kv_dtype_bytes
        for seq in batch_work:
            if seq.prefill_tokens == 0:
                continue
            chunk = prefill_chunk_size or seq.prefill_tokens
            start = 0
            while start < seq.prefill_tokens:
                stop = min(start + chunk, seq.prefill_tokens)
                tokens = stop - start
                shards.append(self._kernel_launch_shard(group_index))
                for layer_index, layer in enumerate(model.layers):
                    flops = 0.0
                    bytes_read = 0.0
                    if layer.mixer is not None:
                        m_flops, m_bytes = layer.mixer.prefill_cost(
                            tokens, seq.cached_tokens, start, pdb, kvdb
                        )
                        flops += m_flops
                        bytes_read += m_bytes
                    ffn_flops, ffn_bytes = self._ffn_cost(layer, tokens, consecutive=True)
                    flops += ffn_flops
                    bytes_read += ffn_bytes
                    shards.append(
                        WorkShard(
                            group_index=group_index,
                            phase="prefill",
                            layer_index=layer_index,
                            flops=flops,
                            bytes_read=bytes_read,
                            flops_dtype_bytes=pdb,
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
        pdb = model.param_dtype_bytes
        kvdb = model.kv_dtype_bytes
        max_steps = max(seq.decode_tokens for seq in batch_work)
        for step in range(1, max_steps + 1):
            active = [seq for seq in batch_work if seq.decode_tokens >= step]
            if not active:
                continue
            batch_size = len(active)
            contexts = [seq.base_tokens + step for seq in active]
            shards.append(self._kernel_launch_shard(group_index))
            for layer_index, layer in enumerate(model.layers):
                flops = 0.0
                bytes_read = 0.0
                if layer.mixer is not None:
                    m_flops, m_bytes = layer.mixer.decode_cost(contexts, pdb, kvdb)
                    flops += m_flops
                    bytes_read += m_bytes
                ffn_flops, ffn_bytes = self._ffn_cost(layer, batch_size, consecutive=False)
                flops += ffn_flops
                bytes_read += ffn_bytes
                shards.append(
                    WorkShard(
                        group_index=group_index,
                        phase="decode",
                        layer_index=layer_index,
                        flops=flops,
                        bytes_read=bytes_read,
                        flops_dtype_bytes=pdb,
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
                        flops_dtype_bytes=pdb,
                        kind="lm_head",
                        tokens=batch_size,
                    )
                )
            group_index += 1
        return group_index
