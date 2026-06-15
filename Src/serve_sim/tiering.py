"""Two-tier expert residency: activation traces and an LRU residency cache.

In a two-tier system the first-tier memory holds the KV cache, the non-expert
weights, and a working set of routed experts; all routed experts live in the
second tier. When a forward-pass group needs an expert that is not currently
resident in the first tier, its weights must be moved up -- a data-transfer
event. Experts kept live by the persistence model are reused across groups,
which is what makes the working set small.

This module provides:

- :func:`build_activation_trace` -- the concrete set of routed experts touched
  by each forward-pass group, in the same group order the work-shard generator
  uses. Expert selection is sampled once (seeded) and is shared by both the event
  generator and the test reference so they agree without coupling their RNGs.
- :class:`ExpertResidencyCache` -- an LRU cache over expert indices that counts
  misses (transfers) per group.
- :func:`derive_expert_cache_capacity` -- first-tier expert capacity (in expert
  indices) derived from device/memory sizes.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass

from .experts import ExpertUsageModel
from .model import Model
from .tracker import SequenceWork


@dataclass(frozen=True)
class GroupActivation:
    """Routed experts touched by one forward-pass group."""

    group_index: int
    phase: str
    active_experts: frozenset[int]


def build_activation_trace(
    model: Model,
    batch_work: list[SequenceWork],
    prefill_chunk_size: int | None = None,
    seed: int = 0,
) -> list[GroupActivation]:
    """Sample the routed experts each group touches, in shard group order.

    Each sequence has ``num_experts_per_token`` persistent expert slots whose
    runs follow the model's persistence distribution; slots advance through the
    sequence's prefill tokens then its decode tokens as a single stream (so
    decode reuses what prefill warmed). Selection is identical across layers, so
    a single set of expert indices describes every MoE layer of the group.
    """

    if model.ffn_type != "moe":
        return []

    usage = ExpertUsageModel.from_model(model)
    rng = random.Random(seed)
    k = model.num_experts_per_token
    e = model.num_experts
    slots: dict[int, list[list[int]]] = {}

    def advance_token(seq_id: int) -> set[int]:
        state = slots.setdefault(seq_id, [[-1, 0] for _ in range(k)])
        used: set[int] = set()
        for slot in state:
            if slot[1] <= 0:
                slot[0] = rng.randrange(e)
                slot[1] = usage._sample_persistence(rng)
            used.add(slot[0])
            slot[1] -= 1
        return used

    groups: list[GroupActivation] = []
    group_index = 0

    # Prefill: per sequence, chunked (mirrors WorkShardGenerator).
    for seq_id, seq in enumerate(batch_work):
        if seq.prefill_tokens == 0:
            continue
        chunk = prefill_chunk_size or seq.prefill_tokens
        start = 0
        while start < seq.prefill_tokens:
            stop = min(start + chunk, seq.prefill_tokens)
            active: set[int] = set()
            for _ in range(stop - start):
                active |= advance_token(seq_id)
            groups.append(GroupActivation(group_index, "prefill", frozenset(active)))
            group_index += 1
            start = stop

    # Decode: batched lockstep steps.
    max_steps = max(seq.decode_tokens for seq in batch_work)
    for step in range(1, max_steps + 1):
        active_seqs = [i for i, seq in enumerate(batch_work) if seq.decode_tokens >= step]
        if not active_seqs:
            continue
        active = set()
        for seq_id in active_seqs:
            active |= advance_token(seq_id)
        groups.append(GroupActivation(group_index, "decode", frozenset(active)))
        group_index += 1

    return groups


class ExpertResidencyCache:
    """LRU cache over routed-expert indices resident in the first tier.

    Capacity is measured in expert indices; one index occupies the weights of
    that expert in every MoE layer (they move together since selection is shared
    across layers).
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._resident: "OrderedDict[int, None]" = OrderedDict()

    @property
    def resident(self) -> frozenset[int]:
        return frozenset(self._resident)

    def access(self, active_experts: frozenset[int] | set[int]) -> int:
        """Touch ``active_experts``; return the number of misses (transfers).

        Requires ``capacity >= len(active_experts)`` so the current working set
        is never evicted mid-group.
        """

        if len(active_experts) > self.capacity:
            raise ValueError(
                f"first tier too small: active set {len(active_experts)} exceeds "
                f"expert cache capacity {self.capacity}"
            )
        misses = 0
        for idx in sorted(active_experts):
            if idx in self._resident:
                self._resident.move_to_end(idx)
            else:
                misses += 1
                self._resident[idx] = None
        while len(self._resident) > self.capacity:
            self._resident.popitem(last=False)
        return misses


def _peak_kv_bytes(model: Model, batch_work: list[SequenceWork]) -> int:
    total_tokens = sum(seq.base_tokens + seq.decode_tokens for seq in batch_work)
    return total_tokens * model.num_layers * model.kv_bytes_per_token


def _nonexpert_weight_bytes(model: Model) -> int:
    moe_layers = model.num_moe_layers
    dense_layers = model.num_layers - moe_layers
    return (
        model.num_layers * model.attention_weight_bytes
        + dense_layers * model.dense_ffn_bytes
        + moe_layers * model.shared_expert_bytes
        + model.lm_head_bytes
    )


def derive_expert_cache_capacity(
    model: Model,
    first_tier_capacity_bytes: float,
    batch_work: list[SequenceWork],
) -> int:
    """First-tier routed-expert capacity, in expert indices.

    Reserves space for the peak KV cache and the always-resident non-expert
    weights; the remainder is divided by the per-index expert footprint
    (``num_moe_layers * routed_expert_bytes``).

    Raises:
        ValueError: If the first tier cannot even hold the reserved bytes plus
            one expert index.
    """

    moe_layers = model.num_moe_layers
    if moe_layers == 0:
        raise ValueError("model has no MoE layers")
    per_index_bytes = moe_layers * model.routed_expert_bytes
    reserved = _peak_kv_bytes(model, batch_work) + _nonexpert_weight_bytes(model)
    budget = first_tier_capacity_bytes - reserved
    capacity = int(budget // per_index_bytes)
    if capacity < 1:
        raise ValueError(
            "first tier too small to hold reserved bytes plus one routed expert"
        )
    return capacity
