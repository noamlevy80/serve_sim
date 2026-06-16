"""Multi-turn conversation tests: tool-call wait events across turns.

``run_conversation`` lays a workload's turns out end-to-end on one timeline. Each
turn is tokenized (its predecessor's messages form the cached prefix), turned into
work shards and timed by the same event generator the single-turn path uses; a
tool-call wait event (``phase == "tool_call"``) precedes every turn whose
``pre_gap`` is positive. We model only the *wait* for a tool call, not its
computation, so those events carry no FLOPs or bytes -- just a duration scaled by
the global ``tool_calling_speedup``.
"""

from __future__ import annotations

import pytest

from serve_sim.model import toy_model
from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.tokenizer import WhitespaceTokenizer
from serve_sim.workload import build_workload_from_rows
from serve_sim.tracker import SequenceTracker
from serve_sim.shards import WorkShardGenerator
from serve_sim.events import EventGenerator
from serve_sim.conversation import TOOL_CALL_PHASE, run_conversation
from conftest import make_row


def make_device(name="gpu", peak=100e12, bw=2e12, cap=80e9):
    mem = MemoryDevice(f"{name}-hbm", capacity_bytes=cap, bandwidth_bytes_per_s=bw)
    return ComputeDevice(name, peak_flops_fp16=peak, first_tier_memory=mem)


def conversation_rows(pre_gaps, outputs, session="conv", model="m"):
    """Build prefix-growing rows for a tool-using agentic conversation.

    Turn 0 is the system+user prompt. Every later turn appends the assistant's
    tool-call request and the tool's response (so tool-call argument text and the
    tool result are tokenized into that turn's prefill), and carries the
    client-side ``pre_gap`` spent waiting for the tool.
    """

    assert len(pre_gaps) == len(outputs)
    base = [
        {"role": "system", "content": "you are a helpful agent"},
        {"role": "user", "content": "what is the weather and the latest news"},
    ]
    messages = list(base)
    rows = []
    for t, (gap, out) in enumerate(zip(pre_gaps, outputs)):
        if t > 0:
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{t}",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": f"city tel aviv query number {t} here",
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{t}",
                    "content": f"tool result for turn {t} with several extra tokens",
                }
            )
        rows.append(
            make_row(
                session_id=session,
                model=model,
                input_messages=[dict(m) for m in messages],
                output_length=out,
                pre_gap=gap,
            )
        )
    return rows


def make_conversation(pre_gaps, outputs):
    return build_workload_from_rows(conversation_rows(pre_gaps, outputs))


def run(workload, *, tool_calling_speedup=1.0, prefill_chunk_size=None):
    return run_conversation(
        workload,
        toy_model(num_layers=2),
        [make_device()],
        WhitespaceTokenizer(),
        prefill_chunk_size=prefill_chunk_size,
        tool_calling_speedup=tool_calling_speedup,
    )


def tool_events(schedule):
    return [e for e in schedule.events if e.phase == TOOL_CALL_PHASE]


# --- tool-call wait events ------------------------------------------------------


def test_one_tool_wait_event_per_nonzero_pre_gap():
    wl = make_conversation(pre_gaps=[0.0, 0.5, 0.3], outputs=[8, 6, 4])
    waits = tool_events(run(wl))
    assert len(waits) == 2  # turn 0 has no pre_gap


def test_first_turn_has_no_tool_wait():
    wl = make_conversation(pre_gaps=[0.0, 0.5], outputs=[8, 6])
    schedule = run(wl)
    first_event = min(schedule.events, key=lambda e: e.start)
    assert first_event.phase != TOOL_CALL_PHASE
    assert first_event.start == 0.0


def test_tool_wait_durations_match_pre_gaps():
    pre_gaps = [0.0, 0.5, 0.3, 0.0, 0.2]
    wl = make_conversation(pre_gaps, outputs=[5, 5, 5, 5, 5])
    durations = sorted(e.duration for e in tool_events(run(wl)))
    assert durations == pytest.approx(sorted(g for g in pre_gaps if g > 0))


def test_tool_wait_events_carry_no_work():
    wl = make_conversation(pre_gaps=[0.0, 0.7], outputs=[8, 6])
    for event in tool_events(run(wl)):
        assert event.flops == 0.0
        assert event.bytes_read == 0.0
        assert event.compute_time == 0.0
        assert event.bandwidth_time == 0.0


def test_total_tool_wait_equals_sum_of_pre_gaps():
    pre_gaps = [0.0, 0.5, 0.3, 0.4]
    wl = make_conversation(pre_gaps, outputs=[6, 6, 6, 6])
    schedule = run(wl)
    assert schedule.time_for_phase(TOOL_CALL_PHASE) == pytest.approx(sum(pre_gaps))


# --- tool_calling_speedup -------------------------------------------------------


def test_speedup_scales_tool_waits():
    wl = make_conversation(pre_gaps=[0.0, 0.6, 0.4], outputs=[6, 6, 6])
    base = run(wl).time_for_phase(TOOL_CALL_PHASE)
    fast = run(wl, tool_calling_speedup=2.0).time_for_phase(TOOL_CALL_PHASE)
    assert fast == pytest.approx(base / 2.0)


def test_speedup_does_not_change_compute_time():
    wl = make_conversation(pre_gaps=[0.0, 0.6, 0.4], outputs=[6, 6, 6])
    slow = run(wl, tool_calling_speedup=1.0)
    fast = run(wl, tool_calling_speedup=10.0)
    assert fast.total_flops == pytest.approx(slow.total_flops)
    assert fast.total_bytes == pytest.approx(slow.total_bytes)


def test_non_positive_speedup_rejected():
    wl = make_conversation(pre_gaps=[0.0, 0.5], outputs=[8, 6])
    with pytest.raises(ValueError):
        run(wl, tool_calling_speedup=0.0)


# --- timeline composition -------------------------------------------------------


def test_events_form_a_contiguous_non_overlapping_timeline():
    wl = make_conversation(pre_gaps=[0.0, 0.5, 0.3], outputs=[8, 6, 4])
    events = sorted(run(wl).events, key=lambda e: e.start)
    clock = 0.0
    for event in events:
        assert event.start == pytest.approx(clock)
        assert event.end >= event.start
        clock = event.end


def test_tool_wait_sits_between_turns_compute():
    wl = make_conversation(pre_gaps=[0.0, 0.5], outputs=[8, 6])
    events = sorted(run(wl).events, key=lambda e: e.start)
    wait_index = next(i for i, e in enumerate(events) if e.phase == TOOL_CALL_PHASE)
    # compute precedes and follows the single wait.
    assert any(e.phase != TOOL_CALL_PHASE for e in events[:wait_index])
    assert any(e.phase != TOOL_CALL_PHASE for e in events[wait_index + 1:])


def test_makespan_is_turn_compute_plus_tool_waits():
    pre_gaps = [0.0, 0.5, 0.3]
    outputs = [8, 6, 4]
    wl = make_conversation(pre_gaps, outputs)
    schedule = run(wl)

    # Independently time each turn through the public single-turn path.
    model = toy_model(num_layers=2)
    tokenizer = WhitespaceTokenizer()
    turn_makespans = 0.0
    for turn in wl:
        tracker = SequenceTracker.from_turn(wl, turn.index, tokenizer)
        shards = WorkShardGenerator(model).generate([tracker.to_work()])
        turn_makespans += EventGenerator(model, [make_device()]).run(shards).makespan

    assert schedule.makespan == pytest.approx(turn_makespans + sum(pre_gaps))


def test_compute_events_reproduce_single_turn_runs():
    wl = make_conversation(pre_gaps=[0.0, 0.5], outputs=[8, 6])
    schedule = run(wl)
    compute = [e for e in schedule.events if e.phase != TOOL_CALL_PHASE]

    model = toy_model(num_layers=2)
    tokenizer = WhitespaceTokenizer()
    expected = 0
    for turn in wl:
        tracker = SequenceTracker.from_turn(wl, turn.index, tokenizer)
        shards = WorkShardGenerator(model).generate([tracker.to_work()])
        expected += len(EventGenerator(model, [make_device()]).run(shards).events)
    assert len(compute) == expected


# --- tool-call content is tokenized into the conversation -----------------------


def test_later_turns_cache_previous_messages():
    # Each turn's cached prefix is exactly the previous turn's token count, so the
    # tool-call request and tool response added in a turn become its prefill.
    wl = make_conversation(pre_gaps=[0.0, 0.5, 0.3], outputs=[6, 6, 6])
    tokenizer = WhitespaceTokenizer()
    trackers = [
        SequenceTracker.from_turn(wl, turn.index, tokenizer) for turn in wl
    ]
    assert trackers[0].cached_tokens == 0
    for i in range(1, len(trackers)):
        assert trackers[i].cached_tokens == trackers[i - 1].prompt_tokens
        assert trackers[i].prefill_tokens > 0  # appended tool messages add work


def test_single_turn_conversation_has_no_tool_waits():
    wl = make_conversation(pre_gaps=[0.0], outputs=[8])
    schedule = run(wl)
    assert tool_events(schedule) == []
    assert schedule.makespan > 0.0
