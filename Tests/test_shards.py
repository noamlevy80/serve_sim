"""Tests for the work-shard generator structure and sizing."""

from __future__ import annotations

import pytest

from serve_sim.model import toy_model
from serve_sim.shards import WorkShardGenerator
from serve_sim.tracker import SequenceWork


def test_prefill_emits_one_shard_per_layer():
    model = toy_model(num_layers=4)
    gen = WorkShardGenerator(model)
    work = [SequenceWork(cached_tokens=0, prefill_tokens=8, decode_tokens=0)]
    shards = gen.generate(work)
    prefill = [s for s in shards if s.phase == "prefill"]
    assert len(prefill) == 4
    assert {s.layer_index for s in prefill} == {0, 1, 2, 3}
    assert all(s.group_index == 0 for s in prefill)


def test_prefill_chunking_creates_groups():
    model = toy_model(num_layers=2)
    gen = WorkShardGenerator(model)
    work = [SequenceWork(cached_tokens=0, prefill_tokens=10, decode_tokens=0)]
    shards = gen.generate(work, prefill_chunk_size=4)  # chunks: 4, 4, 2 -> 3 groups
    groups = sorted({s.group_index for s in shards})
    assert len(groups) == 3
    # each group has one shard per layer
    for g in groups:
        assert sum(1 for s in shards if s.group_index == g) == 2


def test_decode_emits_layers_plus_lm_head_per_step():
    model = toy_model(num_layers=3, include_lm_head=True)
    gen = WorkShardGenerator(model)
    work = [SequenceWork(cached_tokens=0, prefill_tokens=4, decode_tokens=5)]
    shards = gen.generate(work)
    decode = [s for s in shards if s.phase == "decode"]
    steps = sorted({s.group_index for s in decode})
    assert len(steps) == 5  # 5 decode steps
    for g in steps:
        group_shards = [s for s in decode if s.group_index == g]
        assert len(group_shards) == 4  # 3 layers + 1 lm head
        assert sum(1 for s in group_shards if s.kind == "lm_head") == 1


def test_decode_without_lm_head():
    model = toy_model(num_layers=3, include_lm_head=False)
    gen = WorkShardGenerator(model)
    work = [SequenceWork(cached_tokens=0, prefill_tokens=4, decode_tokens=2)]
    shards = gen.generate(work)
    decode = [s for s in shards if s.phase == "decode"]
    assert all(s.kind == "layer" for s in decode)
    assert len({s.group_index for s in decode}) == 2


def test_ragged_decode_lengths_shrink_active_batch():
    model = toy_model(num_layers=1, include_lm_head=False)
    gen = WorkShardGenerator(model)
    work = [
        SequenceWork(0, 4, 3),  # decodes 3 steps
        SequenceWork(0, 4, 1),  # decodes 1 step
    ]
    shards = gen.generate(work)
    decode = [s for s in shards if s.phase == "decode"]
    # step 1: both active (tokens=2), steps 2,3: only one active (tokens=1)
    by_group = {}
    for s in decode:
        by_group.setdefault(s.group_index, []).append(s)
    token_counts = sorted(s[0].tokens for s in by_group.values())
    assert token_counts == [1, 1, 2]


def test_decode_weight_bytes_amortized_over_batch():
    model = toy_model(num_layers=1, include_lm_head=False)
    gen = WorkShardGenerator(model)
    single = gen.generate([SequenceWork(0, 4, 1)])
    quad = gen.generate([SequenceWork(0, 4, 1) for _ in range(4)])
    # first decode-step, layer-0 shard for each
    s1 = next(s for s in single if s.phase == "decode")
    s4 = next(s for s in quad if s.phase == "decode")
    # weight bytes (constant part) are shared; only KV bytes scale with batch.
    kv = model.kv_bytes_per_token
    weight_bytes = model.layer_weight_bytes
    # single: context = base(4)+1 = 5
    assert s1.bytes_read == weight_bytes + 5 * kv
    # quad: weights once, KV x4 sequences (each context 5)
    assert s4.bytes_read == weight_bytes + 4 * 5 * kv


def test_generate_rejects_empty_batch():
    gen = WorkShardGenerator(toy_model())
    with pytest.raises(ValueError, match="at least one"):
        gen.generate([])


def test_generate_rejects_bad_chunk_size():
    gen = WorkShardGenerator(toy_model())
    with pytest.raises(ValueError, match="prefill_chunk_size"):
        gen.generate([SequenceWork(0, 4, 1)], prefill_chunk_size=0)


def test_prefill_skipped_when_fully_cached():
    model = toy_model(num_layers=2)
    gen = WorkShardGenerator(model)
    work = [SequenceWork(cached_tokens=8, prefill_tokens=0, decode_tokens=2)]
    shards = gen.generate(work)
    assert not any(s.phase == "prefill" for s in shards)
    assert any(s.phase == "decode" for s in shards)
