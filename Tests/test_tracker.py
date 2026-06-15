"""Tests for SequenceTracker, BatchTracker and tokenizer integration."""

from __future__ import annotations

import pytest

from serve_sim.tracker import BatchTracker, SequenceTracker, SequenceWork
from serve_sim.tokenizer import WhitespaceTokenizer
from serve_sim.workload import build_workload_from_rows
from conftest import make_session_rows


class CharTokenizer:
    """Deterministic tokenizer: one token per character (test only)."""

    def count(self, text: str) -> int:
        return len(text or "")


# --- SequenceWork ---------------------------------------------------------------


def test_sequence_work_base_tokens():
    work = SequenceWork(cached_tokens=10, prefill_tokens=5, decode_tokens=3)
    assert work.base_tokens == 15


def test_sequence_work_rejects_negative():
    with pytest.raises(ValueError):
        SequenceWork(cached_tokens=-1, prefill_tokens=0, decode_tokens=0)


# --- SequenceTracker indices ----------------------------------------------------


def test_sequence_tracker_prefill_tokens():
    t = SequenceTracker(prompt_tokens=100, output_tokens=20, cached_tokens=30)
    assert t.prefill_tokens == 70


def test_sequence_tracker_indices_with_cache():
    t = SequenceTracker(prompt_tokens=100, output_tokens=20, cached_tokens=30)
    assert t.last_cache_index == 29
    assert t.last_prefill_index == 99
    assert t.last_decode_index == 119


def test_sequence_tracker_indices_without_cache():
    t = SequenceTracker(prompt_tokens=10, output_tokens=4)
    assert t.last_cache_index is None
    assert t.last_prefill_index == 9
    assert t.last_decode_index == 13


def test_sequence_tracker_cached_cannot_exceed_prompt():
    with pytest.raises(ValueError, match="cannot exceed"):
        SequenceTracker(prompt_tokens=10, output_tokens=1, cached_tokens=11)


def test_sequence_tracker_to_work():
    t = SequenceTracker(prompt_tokens=100, output_tokens=20, cached_tokens=30)
    work = t.to_work()
    assert work == SequenceWork(cached_tokens=30, prefill_tokens=70, decode_tokens=20)


# --- from_turn (tokenizing a workload turn) ------------------------------------


def test_from_turn_first_turn_has_no_cache():
    rows = make_session_rows("s", "m", num_turns=3, messages_per_turn=2)
    wl = build_workload_from_rows(rows)
    tok = CharTokenizer()
    t = SequenceTracker.from_turn(wl, 0, tok)
    assert t.cached_tokens == 0
    # prompt tokens == sum of char counts of the two prefix messages
    expected = sum(tok.count(m.content) for m in wl[0].messages)
    assert t.prompt_tokens == expected
    assert t.output_tokens == wl[0].output_length


def test_from_turn_later_turn_caches_previous():
    rows = make_session_rows("s", "m", num_turns=3, messages_per_turn=2)
    wl = build_workload_from_rows(rows)
    tok = CharTokenizer()
    t = SequenceTracker.from_turn(wl, 1, tok)
    prev = sum(tok.count(m.content) for m in wl[0].messages)
    cur = sum(tok.count(m.content) for m in wl[1].messages)
    assert t.cached_tokens == prev
    assert t.prompt_tokens == cur
    assert t.prefill_tokens == cur - prev


def test_from_turn_prefix_growth_makes_cache_a_true_prefix():
    # because previous turn's messages are a prefix, cached <= prompt always
    rows = make_session_rows("s", "m", num_turns=4, messages_per_turn=3)
    wl = build_workload_from_rows(rows)
    tok = WhitespaceTokenizer()
    for i in range(wl.num_turns):
        t = SequenceTracker.from_turn(wl, i, tok)
        assert t.cached_tokens <= t.prompt_tokens
        assert t.prefill_tokens >= 0


# --- BatchTracker ---------------------------------------------------------------


def test_batch_tracker_work_list():
    seqs = [
        SequenceTracker(prompt_tokens=10, output_tokens=2),
        SequenceTracker(prompt_tokens=20, output_tokens=5, cached_tokens=4),
    ]
    batch = BatchTracker(seqs)
    assert len(batch) == 2
    work = batch.work()
    assert work[0] == SequenceWork(0, 10, 2)
    assert work[1] == SequenceWork(4, 16, 5)


def test_batch_tracker_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        BatchTracker([])


# --- real tokenizer integration -------------------------------------------------


def test_tiktoken_tokenizer_counts_tokens():
    pytest.importorskip("tiktoken")
    from serve_sim.tokenizer import TiktokenTokenizer

    tok = TiktokenTokenizer()
    assert tok.count("") == 0
    assert tok.count("hello world") >= 1
    # longer text yields at least as many tokens
    assert tok.count("hello world this is a longer sentence") > tok.count("hello")
