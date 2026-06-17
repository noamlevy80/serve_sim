"""Tests for KV-cache residency tracking and cross-sequence prefix linking.

A :class:`KVCacheTracker` records, per token index, whether the token's KV is
generated, which memory devices hold it (by identity), and an optional link to
the same index in another tracker (prefix reuse). A :class:`SequenceTracker` owns
one KV tracker per layer and can link a freshly admitted sequence to an existing
one over their common message-aligned prefix.
"""

from __future__ import annotations

import pytest

from serve_sim.kv_cache import KVCacheTracker
from serve_sim.hardware import MemoryDevice
from serve_sim.tracker import SequenceTracker
from serve_sim.tokenizer import WhitespaceTokenizer
from serve_sim.workload import build_workload_from_rows
from conftest import make_row, make_session_rows


def mem(name: str) -> MemoryDevice:
    return MemoryDevice(name, capacity_bytes=1e12, bandwidth_bytes_per_s=1e12)


# --- length / growth ------------------------------------------------------------


def test_new_tracker_is_empty():
    t = KVCacheTracker()
    assert len(t) == 0
    assert t.cached_prefix_length() == 0


def test_extend_adds_ungenerated_unplaced_tokens():
    t = KVCacheTracker()
    t.extend(5)
    assert len(t) == 5
    for i in range(5):
        assert not t.is_generated(i)
        assert t.devices_at(i) == []
        assert not t.is_available(i)


def test_ensure_length_only_grows():
    t = KVCacheTracker()
    t.extend(3)
    t.ensure_length(2)  # no shrink
    assert len(t) == 3
    t.ensure_length(6)
    assert len(t) == 6


def test_extend_rejects_negative():
    with pytest.raises(ValueError):
        KVCacheTracker().extend(-1)


# --- generated / residency ------------------------------------------------------


def test_mark_generated_sets_flag_without_residency():
    t = KVCacheTracker()
    t.extend(4)
    t.mark_generated(0, 2)
    assert t.is_generated(0) and t.is_generated(1)
    assert not t.is_generated(2)
    # generated but resident nowhere -> not yet available for reuse.
    assert not t.is_available(0)


def test_place_marks_generated_and_resident():
    t = KVCacheTracker()
    t.extend(4)
    d = mem("hbm")
    t.place(0, 3, d)
    for i in range(3):
        assert t.is_generated(i)
        assert t.is_resident(i, d)
        assert t.is_available(i)
    assert t.devices_at(0) == [d]


def test_token_can_reside_on_multiple_devices():
    t = KVCacheTracker()
    t.extend(2)
    a, b = mem("a"), mem("b")
    t.place(0, 2, a)
    t.place(0, 2, b)
    assert {id(x) for x in t.devices_at(0)} == {id(a), id(b)}


def test_value_equal_devices_are_distinct_locations():
    # two MemoryDevice with identical fields are value-equal but different
    # physical locations; residency must keep them apart.
    t = KVCacheTracker()
    t.extend(1)
    a, b = mem("same"), mem("same")
    assert a == b  # value equality
    t.place(0, 1, a)
    t.place(0, 1, b)
    assert len(t.devices_at(0)) == 2


def test_evict_removes_one_device():
    t = KVCacheTracker()
    t.extend(2)
    a, b = mem("a"), mem("b")
    t.place(0, 2, a)
    t.place(0, 2, b)
    t.evict(0, 2, a)
    assert not t.is_resident(0, a)
    assert t.is_resident(0, b)
    assert t.is_generated(0)  # still generated, just moved


def test_evicting_last_device_leaves_generated_but_unavailable():
    t = KVCacheTracker()
    t.extend(1)
    d = mem("d")
    t.place(0, 1, d)
    t.evict(0, 1, d)
    assert t.is_generated(0)
    assert t.devices_at(0) == []
    assert not t.is_available(0)


def test_range_checks():
    t = KVCacheTracker()
    t.extend(3)
    with pytest.raises(IndexError):
        t.place(0, 4, mem("d"))
    with pytest.raises(IndexError):
        t.mark_generated(2, 1)


# --- cached prefix length -------------------------------------------------------


def test_cached_prefix_length_counts_leading_available_run():
    t = KVCacheTracker()
    t.extend(5)
    d = mem("d")
    t.place(0, 3, d)  # tokens 0,1,2 available; 3,4 not
    assert t.cached_prefix_length() == 3


def test_cached_prefix_length_stops_at_first_gap():
    t = KVCacheTracker()
    t.extend(4)
    d = mem("d")
    t.place(1, 4, d)  # token 0 missing -> prefix is 0
    assert t.cached_prefix_length() == 0


# --- prefix linking -------------------------------------------------------------


def test_link_prefix_delegates_state_to_source():
    source = KVCacheTracker()
    source.extend(4)
    d = mem("shared")
    source.place(0, 4, d)

    follower = KVCacheTracker()
    follower.link_prefix(source, 3)
    assert len(follower) == 3
    for i in range(3):
        assert follower.is_linked(i)
        assert follower.is_generated(i)
        assert follower.is_resident(i, d)
        assert follower.is_available(i)
    assert follower.cached_prefix_length() == 3


def test_linked_tokens_follow_source_eviction():
    source = KVCacheTracker()
    source.extend(2)
    d = mem("d")
    source.place(0, 2, d)
    follower = KVCacheTracker()
    follower.link_prefix(source, 2)
    source.evict(0, 2, d)
    # follower sees the eviction through the link.
    assert not follower.is_available(0)


def test_link_then_extend_local_tokens():
    source = KVCacheTracker()
    source.extend(2)
    source.place(0, 2, mem("d"))
    follower = KVCacheTracker()
    follower.link_prefix(source, 2)
    follower.extend(2)  # local, un-shared tokens after the prefix
    assert len(follower) == 4
    assert follower.is_linked(0)
    assert not follower.is_linked(2)
    assert not follower.is_available(2)


def test_link_rejects_length_beyond_source():
    source = KVCacheTracker()
    source.extend(2)
    with pytest.raises(ValueError):
        KVCacheTracker().link_prefix(source, 3)


# --- SequenceTracker prefix integration ----------------------------------------


def shared_prefix_workloads():
    """Two sessions sharing a system+user prefix, diverging afterwards."""

    system = {"role": "system", "content": "you are a helpful agent here"}
    user = {"role": "user", "content": "please help me with my task today"}
    a = build_workload_from_rows([
        make_row("a", "m", [dict(system), dict(user)], output_length=4),
    ])
    b = build_workload_from_rows([
        make_row("b", "m", [dict(system), dict(user),
                            {"role": "assistant", "content": "different tail here"}],
                 output_length=4),
    ])
    return a, b


def test_common_prefix_length_is_message_aligned():
    a, b = shared_prefix_workloads()
    tok = WhitespaceTokenizer()
    ta = SequenceTracker.from_turn(a, 0, tok)
    tb = SequenceTracker.from_turn(b, 0, tok)
    # the two leading messages are identical; their summed token count is the prefix.
    expected = sum(tok.count(m.content) for m in a[0].messages)
    assert ta.common_prefix_length(tb) == expected
    assert ta.common_prefix_length(tb) == tb.common_prefix_length(ta)


def test_common_prefix_zero_when_first_message_differs():
    tok = WhitespaceTokenizer()
    a = build_workload_from_rows([
        make_row("a", "m", [{"role": "system", "content": "alpha beta"}], output_length=1),
    ])
    b = build_workload_from_rows([
        make_row("b", "m", [{"role": "system", "content": "gamma delta"}], output_length=1),
    ])
    ta = SequenceTracker.from_turn(a, 0, tok)
    tb = SequenceTracker.from_turn(b, 0, tok)
    assert ta.common_prefix_length(tb) == 0


def test_common_prefix_requires_messages():
    plain = SequenceTracker(prompt_tokens=10, output_tokens=2)
    other = SequenceTracker(prompt_tokens=10, output_tokens=2)
    with pytest.raises(ValueError):
        plain.common_prefix_length(other)


def test_link_kv_prefix_links_every_layer():
    a, b = shared_prefix_workloads()
    tok = WhitespaceTokenizer()
    num_layers = 3
    # existing sequence: prefill its prefix onto a device.
    existing = SequenceTracker.from_turn(a, 0, tok)
    existing.attach_kv_trackers(num_layers)
    prefix = existing.prompt_tokens
    d = mem("hbm")
    for layer in existing.kv_trackers:
        layer.extend(prefix)
        layer.place(0, prefix, d)

    # new sequence reuses the shared prefix.
    incoming = SequenceTracker.from_turn(b, 0, tok)
    incoming.attach_kv_trackers(num_layers)
    linked = incoming.link_kv_prefix(existing)

    expected = existing.common_prefix_length(incoming)
    assert linked == expected
    for layer in incoming.kv_trackers:
        assert layer.cached_prefix_length() == expected
        assert layer.is_resident(0, d)


def test_link_kv_prefix_requires_matching_layer_counts():
    a, b = shared_prefix_workloads()
    tok = WhitespaceTokenizer()
    ta = SequenceTracker.from_turn(a, 0, tok)
    tb = SequenceTracker.from_turn(b, 0, tok)
    ta.attach_kv_trackers(2)
    tb.attach_kv_trackers(3)
    with pytest.raises(ValueError):
        tb.link_kv_prefix(ta)


def test_attach_kv_trackers_validates_count():
    a, _ = shared_prefix_workloads()
    t = SequenceTracker.from_turn(a, 0, WhitespaceTokenizer())
    with pytest.raises(ValueError):
        t.attach_kv_trackers(0)


# --- existing from_turn behavior is unchanged ----------------------------------


def test_from_turn_still_tracks_counts():
    rows = make_session_rows("s", "m", num_turns=2)
    wl = build_workload_from_rows(rows)
    tok = WhitespaceTokenizer()
    t = SequenceTracker.from_turn(wl, 1, tok)
    prev = sum(tok.count(m.content) for m in wl[0].messages)
    assert t.cached_tokens == prev
    assert t.messages == wl[1].messages
