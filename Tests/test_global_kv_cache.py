"""Global (system-wide) KV cache: cross-conversation reuse, LRU eviction,
migration across floating memories, and the orchestration decisions that record
KV transfers and evictions.

The :class:`~serve_sim.kv_store.KVCacheManager` keeps every non-evicted
sequence's KV; a device's first-tier memory holds KV only while computing, so
completed turns are offloaded to *floating* (node) memories where later
sequences of any conversation can reuse their prefix. These tests exercise the
manager directly and through a :class:`~serve_sim.orchestrator.Simulator` run,
asserting that ``kv_reuse``, ``kv_transfer`` and ``kv_eviction`` decisions are
emitted.
"""

from __future__ import annotations

from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.kv_store import KVCacheManager
from serve_sim.model import toy_model
from serve_sim.orchestrator import Request, Simulator, StrategyConfig
from serve_sim.pdd import context_kv_bytes
from serve_sim.system import Network, Node, System
from serve_sim.tokenizer import WhitespaceTokenizer
from serve_sim.tracker import SequenceTracker
from serve_sim.workload import build_workload_from_rows

from conftest import make_row


# --- helpers --------------------------------------------------------------------


def _network() -> Network:
    return Network(
        scale_up_bandwidth_bytes_per_s=1e12,
        scale_up_latency_s=1e-6,
        cxl_bandwidth_bytes_per_s=1e11,
        cxl_latency_s=1e-7,
    )


def _memory(name: str, cap: float, bw: float = 5e11) -> MemoryDevice:
    return MemoryDevice(name, capacity_bytes=cap, bandwidth_bytes_per_s=bw)


def make_kv_system(node_caps, device_cap: float = 80e9) -> System:
    """A system with one node per entry in ``node_caps`` (each a floating cap)."""

    nodes = []
    for index, cap in enumerate(node_caps):
        device = ComputeDevice(
            f"g{index}",
            peak_flops_fp16=100e12,
            first_tier_memory=_memory(f"g{index}-mem", device_cap, bw=1e12),
        )
        node_memory = None if cap is None else _memory(f"node-{index}", cap)
        nodes.append(Node(name=f"node-{index}", compute_devices=(device,),
                          node_memory=node_memory))
    return System(name="kv-test", network=_network(),
                  input_memory=_memory("nvm", 1e12, bw=5e9), nodes=tuple(nodes))


def tracker_from_messages(messages, output_length: int = 4) -> SequenceTracker:
    workload = build_workload_from_rows(
        [make_row("s", "m", messages, output_length=output_length)]
    )
    return SequenceTracker.from_turn(workload, 0, WhitespaceTokenizer())


def request_from_messages(rid, model, messages, *, workload_id, arrival_time=0.0,
                          output_length=4) -> Request:
    workload = build_workload_from_rows(
        [make_row(f"s{workload_id}", "m", messages, output_length=output_length)]
    )
    return Request.from_workload(
        rid, workload, model, WhitespaceTokenizer(),
        arrival_time=arrival_time, turn_index=0, workload_id=workload_id,
    )


SYSTEM = {"role": "system", "content": "you are a helpful coding agent always"}
USER = {"role": "user", "content": "please refactor this module for me today"}


# --- KVCacheManager unit tests --------------------------------------------------


def test_lookup_finds_cross_conversation_prefix():
    model = toy_model()
    manager = KVCacheManager(make_kv_system([80e9]))
    stored = tracker_from_messages([dict(SYSTEM), dict(USER)])
    manager.store(model, stored, workload_id=3, turn_index=0,
                  context_tokens=stored.prompt_tokens + stored.output_tokens, now=0.0)

    incoming = tracker_from_messages(
        [dict(SYSTEM), dict(USER),
         {"role": "assistant", "content": "totally different continuation"}]
    )
    match = manager.lookup(model, incoming, now=1.0)

    assert match is not None
    assert match.entry.workload_id == 3
    tok = WhitespaceTokenizer()
    assert match.prefix_tokens == sum(tok.count(m["content"]) for m in (SYSTEM, USER))


def test_lookup_misses_when_no_shared_prefix():
    model = toy_model()
    manager = KVCacheManager(make_kv_system([80e9]))
    a = tracker_from_messages([{"role": "system", "content": "alpha beta gamma"}])
    manager.store(model, a, workload_id=0, turn_index=0,
                  context_tokens=a.prompt_tokens + a.output_tokens, now=0.0)
    b = tracker_from_messages([{"role": "system", "content": "delta epsilon zeta"}])

    assert manager.lookup(model, b, now=1.0) is None


def test_lookup_ignores_other_models():
    model_a = toy_model()
    model_b = toy_model()
    manager = KVCacheManager(make_kv_system([80e9]))
    stored = tracker_from_messages([dict(SYSTEM), dict(USER)])
    manager.store(model_a, stored, workload_id=0, turn_index=0,
                  context_tokens=stored.prompt_tokens + stored.output_tokens, now=0.0)
    incoming = tracker_from_messages([dict(SYSTEM), dict(USER)])

    assert manager.lookup(model_b, incoming, now=1.0) is None


def test_lru_eviction_when_floating_pool_full():
    model = toy_model()
    context_tokens = 64
    entry_bytes = float(context_kv_bytes(model, context_tokens))
    # Floating memory that holds exactly two entries.
    manager = KVCacheManager(make_kv_system([entry_bytes * 2 + 1]))

    t0 = tracker_from_messages([{"role": "system", "content": "first session prompt"}])
    t1 = tracker_from_messages([{"role": "system", "content": "second session prompt"}])
    t2 = tracker_from_messages([{"role": "system", "content": "third session prompt"}])
    manager.store(model, t0, 0, 0, context_tokens, now=0.0)
    manager.store(model, t1, 1, 0, context_tokens, now=1.0)
    # Touch entry 0 so entry 1 becomes the least-recently-used.
    manager.lookup(model, t0, now=2.0)
    result = manager.store(model, t2, 2, 0, context_tokens, now=3.0)

    assert len(result.evicted) == 1
    assert result.evicted[0].workload_id == 1
    resident = {e.workload_id for e in manager.entries}
    assert resident == {0, 2}


def test_migration_to_other_floating_avoids_eviction():
    model = toy_model()
    context_tokens = 64
    entry_bytes = float(context_kv_bytes(model, context_tokens))
    # Two nodes, each floating memory holds exactly one entry.
    manager = KVCacheManager(make_kv_system([entry_bytes + 1, entry_bytes + 1]))

    t0 = tracker_from_messages([{"role": "system", "content": "alpha session one"}])
    t1 = tracker_from_messages([{"role": "system", "content": "beta session two"}])
    first = manager.store(model, t0, 0, 0, context_tokens, now=0.0)
    second = manager.store(model, t1, 1, 0, context_tokens, now=1.0)

    # The second entry lands on the *other* node's memory rather than evicting.
    assert second.evicted == ()
    assert second.memory is not first.memory
    assert len(manager.entries) == 2


def test_manager_inert_without_floating_memory():
    model = toy_model()
    manager = KVCacheManager(make_kv_system([None]))  # node with no node memory

    assert manager.enabled is False
    tracker = tracker_from_messages([dict(SYSTEM), dict(USER)])
    result = manager.store(model, tracker, 0, 0, 32, now=0.0)
    assert result.memory is None
    assert manager.lookup(model, tracker, now=1.0) is None


# --- Simulator integration ------------------------------------------------------


def test_completion_records_kv_offload_transfer():
    model = toy_model()
    system = make_kv_system([80e9])
    req = request_from_messages(0, model, [dict(SYSTEM), dict(USER)], workload_id=0)

    result = Simulator(system, StrategyConfig(max_batch_size=1)).run([req])

    offloads = [d for d in result.decisions
                if d.kind == "kv_transfer" and d.source_request_id == 0]
    assert offloads, "expected a KV offload transfer to floating memory"
    # The offload targets a floating (node) memory, sourced from the serving GPU.
    assert offloads[0].devices == ("node-0",)
    assert offloads[0].source_devices == ("g0",)


def test_cross_conversation_reuse_is_recorded():
    model = toy_model()
    system = make_kv_system([80e9])
    first = request_from_messages(0, model, [dict(SYSTEM), dict(USER)], workload_id=0)
    # A later conversation that shares the system+user prefix.
    second = request_from_messages(
        1, model,
        [dict(SYSTEM), dict(USER),
         {"role": "assistant", "content": "second conversation tail differs"}],
        workload_id=1, arrival_time=1000.0,
    )

    result = Simulator(system, StrategyConfig(max_batch_size=1)).run([first, second])

    reuse = [d for d in result.decisions
             if d.kind == "kv_reuse" and d.workload_id == 1]
    assert reuse, "expected a cross-conversation KV reuse decision"
    assert reuse[0].source_workload_id == 0
    assert reuse[0].source_turn_index == 0
    assert reuse[0].tokens > 0
    # The reuse is accompanied by a physical fetch transfer from floating memory.
    fetch = [d for d in result.decisions if d.kind == "kv_transfer"
             and d.workload_id == 1 and d.source_workload_id == 0]
    assert fetch, "expected a KV fetch transfer for the reused prefix"


def test_eviction_is_recorded_when_floating_pool_fills():
    model = toy_model()
    # Size the floating pool to hold only one completed sequence, so admitting
    # later ones forces LRU eviction.
    sample = tracker_from_messages(
        [{"role": "system", "content": "distinct session number 0 here"}],
        output_length=8,
    )
    entry_bytes = float(
        context_kv_bytes(model, sample.prompt_tokens + sample.output_tokens)
    )
    system = make_kv_system([entry_bytes * 1.5])
    requests = [
        request_from_messages(
            i, model,
            [{"role": "system", "content": f"distinct session number {i} here"}],
            workload_id=i, arrival_time=float(i) * 1000.0, output_length=8,
        )
        for i in range(3)
    ]

    result = Simulator(system, StrategyConfig(max_batch_size=1)).run(requests)

    evictions = [d for d in result.decisions if d.kind == "kv_eviction"]
    assert evictions, "expected at least one LRU eviction decision"
    assert evictions[0].devices == ("node-0",)


def test_memory_timeline_records_the_last_evicted_object():
    from serve_sim.report import memory_timeline

    model = toy_model()
    sample = tracker_from_messages(
        [{"role": "system", "content": "distinct session number 0 here"}],
        output_length=8,
    )
    entry_bytes = float(
        context_kv_bytes(model, sample.prompt_tokens + sample.output_tokens)
    )
    system = make_kv_system([entry_bytes * 1.5])
    requests = [
        request_from_messages(
            i, model,
            [{"role": "system", "content": f"distinct session number {i} here"}],
            workload_id=i, arrival_time=float(i) * 1000.0, output_length=8,
        )
        for i in range(3)
    ]

    result = Simulator(system, StrategyConfig(max_batch_size=1)).run(requests)
    evictions = [d for d in result.decisions if d.kind == "kv_eviction"]
    assert evictions, "expected at least one LRU eviction decision"

    rows = memory_timeline(result, 64)
    floating = [r for r in rows if r["memory"] == "node-0"]
    # A floating memory aggregates every offloaded sequence under one "KV" band,
    # so its evictions come from the residency intervals: the eviction-object
    # label names the evicted sequence ("kv:<seq>") and holds until the next one.
    labelled = [r for r in floating if r["eviction_object"]]
    assert labelled, "expected the eviction object to be recorded for node-0"
    assert all(r["eviction_object"].startswith("kv:") for r in labelled)
    first_evict_time = min(d.time for d in evictions)
    assert all(r["time_end"] > first_evict_time for r in labelled)
