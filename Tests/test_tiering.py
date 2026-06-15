"""Tests for two-tier expert residency: trace, cache, and capacity."""

from __future__ import annotations

import pytest

from serve_sim.model import toy_moe_model, toy_model
from serve_sim.tracker import SequenceWork
from serve_sim.shards import WorkShardGenerator
from serve_sim.tiering import (
    ExpertResidencyCache,
    build_activation_trace,
    derive_expert_cache_capacity,
)


# --- ExpertResidencyCache -------------------------------------------------------


def test_cache_first_access_all_miss():
    cache = ExpertResidencyCache(capacity=4)
    assert cache.access(frozenset({0, 1, 2})) == 3


def test_cache_reuse_hits():
    cache = ExpertResidencyCache(capacity=4)
    cache.access(frozenset({0, 1}))
    assert cache.access(frozenset({0, 1})) == 0  # fully resident
    assert cache.access(frozenset({1, 2})) == 1  # only 2 is new


def test_cache_lru_eviction():
    cache = ExpertResidencyCache(capacity=2)
    cache.access(frozenset({0}))
    cache.access(frozenset({1}))
    # access 0 to make it most-recent, then add 2 -> evicts 1
    cache.access(frozenset({0}))
    cache.access(frozenset({2}))
    assert cache.resident == frozenset({0, 2})
    assert cache.access(frozenset({1})) == 1  # 1 was evicted


def test_cache_rejects_active_larger_than_capacity():
    cache = ExpertResidencyCache(capacity=2)
    with pytest.raises(ValueError, match="too small"):
        cache.access(frozenset({0, 1, 2}))


def test_cache_capacity_validation():
    with pytest.raises(ValueError):
        ExpertResidencyCache(capacity=0)


# --- build_activation_trace -----------------------------------------------------


def test_trace_empty_for_dense_model():
    model = toy_model()
    trace = build_activation_trace(model, [SequenceWork(0, 8, 2)])
    assert trace == []


def test_trace_group_indices_match_shards():
    model = toy_moe_model(num_layers=3)
    work = [SequenceWork(0, 20, 5), SequenceWork(0, 12, 3)]
    shards = WorkShardGenerator(model).generate(work)
    shard_groups = sorted({s.group_index for s in shards})
    trace = build_activation_trace(model, work)
    trace_groups = sorted(g.group_index for g in trace)
    assert trace_groups == shard_groups


def test_trace_group_indices_match_with_chunking():
    model = toy_moe_model(num_layers=2)
    work = [SequenceWork(0, 30, 4)]
    shards = WorkShardGenerator(model).generate(work, prefill_chunk_size=8)
    trace = build_activation_trace(model, work, prefill_chunk_size=8)
    assert sorted(g.group_index for g in trace) == sorted({s.group_index for s in shards})


def test_trace_is_seed_reproducible():
    model = toy_moe_model()
    work = [SequenceWork(0, 40, 8)]
    a = build_activation_trace(model, work, seed=42)
    b = build_activation_trace(model, work, seed=42)
    assert [g.active_experts for g in a] == [g.active_experts for g in b]


def test_trace_active_experts_within_bounds():
    model = toy_moe_model(num_experts=16, num_experts_per_token=2)
    work = [SequenceWork(0, 50, 10) for _ in range(3)]
    trace = build_activation_trace(model, work, seed=1)
    for g in trace:
        assert all(0 <= e < 16 for e in g.active_experts)
        assert len(g.active_experts) <= 16


def test_trace_decode_step_active_bounded_by_batch_topk():
    model = toy_moe_model(num_experts=100, num_experts_per_token=2)
    work = [SequenceWork(0, 4, 3) for _ in range(4)]
    trace = build_activation_trace(model, work, seed=0)
    decode = [g for g in trace if g.phase == "decode"]
    # each decode step has <= batch * k distinct experts
    for g in decode:
        assert len(g.active_experts) <= 4 * 2


def test_trace_high_persistence_fewer_distinct_in_prefill():
    work = [SequenceWork(0, 200, 0)]
    low = build_activation_trace(
        toy_moe_model(expert_persistence_mean=2.0, num_experts=64), work, seed=3
    )
    high = build_activation_trace(
        toy_moe_model(expert_persistence_mean=128.0, num_experts=64), work, seed=3
    )
    assert len(high[0].active_experts) < len(low[0].active_experts)


# --- derive_expert_cache_capacity ----------------------------------------------


def test_derive_capacity_basic():
    model = toy_moe_model(num_layers=2, num_experts=16)
    work = [SequenceWork(0, 32, 8)]
    # generous first tier -> capacity capped by budget / per-index footprint
    cap = derive_expert_cache_capacity(model, first_tier_capacity_bytes=1e9, batch_work=work)
    assert cap >= 1


def test_derive_capacity_raises_when_tier_too_small():
    model = toy_moe_model(num_layers=2, num_experts=16)
    work = [SequenceWork(0, 32, 8)]
    with pytest.raises(ValueError, match="too small"):
        derive_expert_cache_capacity(model, first_tier_capacity_bytes=1.0, batch_work=work)


def test_derive_capacity_grows_with_tier_size():
    model = toy_moe_model(num_layers=2, num_experts=64)
    work = [SequenceWork(0, 32, 8)]
    small = derive_expert_cache_capacity(model, 5e7, work)
    large = derive_expert_cache_capacity(model, 5e8, work)
    assert large > small


def test_derive_capacity_dense_model_raises():
    model = toy_model()
    with pytest.raises(ValueError, match="no MoE"):
        derive_expert_cache_capacity(model, 1e9, [SequenceWork(0, 8, 2)])
