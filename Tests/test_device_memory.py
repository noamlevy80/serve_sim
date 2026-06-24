"""Unit tests for per-device HBM residency (experts + retained KV)."""

from __future__ import annotations

import pytest

from serve_sim.device_memory import (
    DeviceHbmResidency,
    MemoryPolicy,
    _LruPool,
)


# --- _LruPool -------------------------------------------------------------------


def test_pool_admits_until_full_then_evicts_lru():
    pool = _LruPool(capacity=100.0)
    ok, ev = pool.admit("a", "kv", 40.0, now=1.0)
    assert ok and ev == []
    ok, ev = pool.admit("b", "kv", 40.0, now=2.0)
    assert ok and ev == []
    # 'a' is older; admitting 'c' (40) overflows and evicts the LRU 'a'.
    ok, ev = pool.admit("c", "kv", 40.0, now=3.0)
    assert ok
    assert [r.key for r in ev] == ["a"]
    assert pool.contains("b") and pool.contains("c") and not pool.contains("a")


def test_pool_touch_changes_eviction_order():
    pool = _LruPool(capacity=100.0)
    pool.admit("a", "kv", 40.0, now=1.0)
    pool.admit("b", "kv", 40.0, now=2.0)
    pool.touch("a", now=5.0)  # 'a' now newer than 'b'
    _, ev = pool.admit("c", "kv", 40.0, now=6.0)
    assert [r.key for r in ev] == ["b"]


def test_pool_rejects_item_larger_than_capacity():
    pool = _LruPool(capacity=50.0)
    ok, ev = pool.admit("big", "kv", 60.0, now=1.0)
    assert not ok and ev == []
    assert not pool.contains("big")


def test_pool_refresh_resizes_and_reuses_slot():
    pool = _LruPool(capacity=100.0)
    pool.admit("a", "kv", 30.0, now=1.0)
    pool.admit("b", "kv", 30.0, now=2.0)
    # Growing 'a' in place to 80 forces eviction of the other resident 'b'.
    ok, ev = pool.admit("a", "kv", 80.0, now=3.0)
    assert ok
    assert [r.key for r in ev] == ["b"]
    assert pool.contains("a") and pool.used == pytest.approx(80.0)


def test_pool_remove_frees_bytes():
    pool = _LruPool(capacity=100.0)
    pool.admit("a", "kv", 40.0, now=1.0)
    removed = pool.remove("a")
    assert removed is not None and removed.key == "a"
    assert pool.used == 0.0
    assert pool.remove("a") is None


# --- DeviceHbmResidency: global LRU ---------------------------------------------


def test_global_lru_experts_and_kv_compete():
    dev = DeviceHbmResidency(100.0, MemoryPolicy.GLOBAL_LRU)
    assert dev.admit_expert(0, 40.0, now=1.0) == []
    admitted, ev = dev.admit_kv("k0", 40.0, now=2.0)
    assert admitted and ev == []
    # A third 40-byte item overflows; the LRU resident is expert 0 -> evicted,
    # demonstrating KV and experts share one pool.
    ev = dev.admit_expert(1, 40.0, now=3.0)
    assert [(r.kind, r.key) for r in ev] == [("expert", ("expert", 0))]
    assert dev.kv_resident("k0")
    assert dev.expert_resident(1) and not dev.expert_resident(0)


def test_global_lru_protected_experts_not_evicted():
    dev = DeviceHbmResidency(100.0, MemoryPolicy.GLOBAL_LRU)
    dev.admit_kv("k0", 50.0, now=1.0)
    dev.admit_expert(0, 30.0, now=2.0)
    # Admitting expert 1 while protecting expert 0 must evict the KV instead.
    ev = dev.admit_expert(1, 40.0, now=3.0, protected_experts=frozenset({0}))
    assert [r.kind for r in ev] == ["kv"]
    assert dev.expert_resident(0) and dev.expert_resident(1)
    assert not dev.kv_resident("k0")


# --- DeviceHbmResidency: partitioned --------------------------------------------


def test_partitioned_isolates_kv_and_experts():
    dev = DeviceHbmResidency(100.0, MemoryPolicy.PARTITIONED, kv_fraction=0.5)
    # Each sub-region is 50 bytes. Filling KV never evicts experts.
    dev.admit_expert(0, 40.0, now=1.0)
    dev.admit_kv("k0", 40.0, now=2.0)
    admitted, ev = dev.admit_kv("k1", 40.0, now=3.0)
    assert admitted
    # Only the KV sub-region's own LRU ('k0') is evicted; expert 0 survives.
    assert [r.kind for r in ev] == ["kv"]
    assert dev.expert_resident(0)
    assert dev.kv_resident("k1") and not dev.kv_resident("k0")


def test_partitioned_respects_fraction_capacity():
    dev = DeviceHbmResidency(100.0, MemoryPolicy.PARTITIONED, kv_fraction=0.3)
    # KV sub-region is 30 bytes: a 40-byte block cannot be retained at all.
    admitted, ev = dev.admit_kv("k0", 40.0, now=1.0)
    assert not admitted and ev == []


def test_kv_fraction_validation():
    with pytest.raises(ValueError):
        DeviceHbmResidency(100.0, MemoryPolicy.PARTITIONED, kv_fraction=0.0)
    with pytest.raises(ValueError):
        DeviceHbmResidency(100.0, MemoryPolicy.PARTITIONED, kv_fraction=1.0)


def test_kv_bytes_and_remove():
    dev = DeviceHbmResidency(100.0, MemoryPolicy.GLOBAL_LRU)
    dev.admit_kv("k0", 25.0, now=1.0)
    assert dev.kv_bytes("k0") == pytest.approx(25.0)
    assert dev.kv_bytes("absent") is None
    dev.remove_kv("k0")
    assert not dev.kv_resident("k0")
