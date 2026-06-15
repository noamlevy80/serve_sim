"""Tests for the workload data model (parsing, turns, prefix growth, deltas)."""

from __future__ import annotations

import pytest

from serve_sim.workload import (
    Message,
    ToolCall,
    Turn,
    Workload,
    build_workload_from_rows,
)
from conftest import make_row, make_session_rows


# --- Message / ToolCall parsing -------------------------------------------------


def test_message_from_raw_minimal():
    msg = Message.from_raw({"role": "user", "content": "hello"})
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.tool_calls == ()
    assert msg.tool_call_id is None
    assert msg.name is None


def test_message_from_raw_requires_role():
    with pytest.raises(ValueError, match="role"):
        Message.from_raw({"content": "no role"})


def test_message_from_raw_null_content_allowed():
    msg = Message.from_raw({"role": "assistant", "content": None})
    assert msg.content is None


def test_message_from_raw_parses_tool_calls():
    raw = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
            }
        ],
    }
    msg = Message.from_raw(raw)
    assert len(msg.tool_calls) == 1
    tc = msg.tool_calls[0]
    assert tc == ToolCall(
        id="call_1",
        type="function",
        function_name="read_file",
        function_arguments='{"path": "a.py"}',
    )


def test_message_from_raw_tool_result():
    raw = {"role": "tool", "content": "result", "tool_call_id": "call_1", "name": "read_file"}
    msg = Message.from_raw(raw)
    assert msg.role == "tool"
    assert msg.tool_call_id == "call_1"
    assert msg.name == "read_file"


def test_tool_call_missing_function():
    tc = ToolCall.from_raw({"id": "x", "type": "function"})
    assert tc.function_name is None
    assert tc.function_arguments is None


def test_message_is_hashable_and_comparable():
    a = Message.from_raw({"role": "user", "content": "hi"})
    b = Message.from_raw({"role": "user", "content": "hi"})
    assert a == b
    assert hash(a) == hash(b)


# --- Turn -----------------------------------------------------------------------


def test_turn_len_counts_messages():
    turn = Turn(
        index=0,
        messages=(Message("system"), Message("user")),
        output_length=5,
        pre_gap=0.0,
    )
    assert len(turn) == 2


# --- build_workload_from_rows ---------------------------------------------------


def test_build_workload_basic_fields():
    rows = make_session_rows("sess", "model-z", num_turns=3)
    wl = build_workload_from_rows(rows)
    assert wl.session_id == "sess"
    assert wl.model == "model-z"
    assert wl.num_turns == 3
    assert len(wl) == 3


def test_build_workload_turn_metadata():
    rows = make_session_rows("sess", "model-z", num_turns=3)
    wl = build_workload_from_rows(rows)
    assert [t.index for t in wl] == [0, 1, 2]
    assert [t.output_length for t in wl] == [10, 20, 30]
    assert [t.pre_gap for t in wl] == [0.0, 1.0, 2.0]


def test_build_workload_coerces_types():
    row = make_row("s", "m", [{"role": "user", "content": "x"}], output_length=7, pre_gap=3)
    # simulate values arriving as strings/ints from a loosely typed source
    row["output_length"] = "7"
    row["pre_gap"] = 3
    wl = build_workload_from_rows([row])
    assert wl.turns[0].output_length == 7
    assert isinstance(wl.turns[0].output_length, int)
    assert wl.turns[0].pre_gap == 3.0
    assert isinstance(wl.turns[0].pre_gap, float)


def test_build_workload_empty_rows_raises():
    with pytest.raises(ValueError, match="zero rows"):
        build_workload_from_rows([])


def test_build_workload_multiple_sessions_raises():
    rows = [
        make_row("a", "m", [{"role": "user", "content": "x"}]),
        make_row("b", "m", [{"role": "user", "content": "y"}]),
    ]
    with pytest.raises(ValueError, match="multiple sessions"):
        build_workload_from_rows(rows)


# --- Workload behaviour ---------------------------------------------------------


def test_workload_requires_turns():
    with pytest.raises(ValueError, match="no turns"):
        Workload(session_id="s", model="m", turns=[])


def test_workload_indexing_and_iteration():
    rows = make_session_rows("sess", "m", num_turns=2)
    wl = build_workload_from_rows(rows)
    assert wl[0].index == 0
    assert wl[1].index == 1
    assert list(wl) == wl.turns


def test_new_messages_first_turn_is_full_prefix():
    rows = make_session_rows("sess", "m", num_turns=3, messages_per_turn=2)
    wl = build_workload_from_rows(rows)
    first = wl.new_messages(0)
    assert len(first) == 2  # system + user
    assert first[0].role == "system"
    assert first[1].role == "user"


def test_new_messages_delta_only():
    rows = make_session_rows("sess", "m", num_turns=3, messages_per_turn=2)
    wl = build_workload_from_rows(rows)
    delta1 = wl.new_messages(1)
    assert len(delta1) == 2
    assert all(m.role == "assistant" for m in delta1)
    assert delta1[0].content == "turn1-msg0"
    assert delta1[1].content == "turn1-msg1"


def test_new_messages_reconstructs_full_history():
    rows = make_session_rows("sess", "m", num_turns=4, messages_per_turn=3)
    wl = build_workload_from_rows(rows)
    reconstructed: list[Message] = []
    for t in range(wl.num_turns):
        reconstructed.extend(wl.new_messages(t))
    assert tuple(reconstructed) == wl.turns[-1].messages


def test_validate_prefix_growth_passes_for_valid_data():
    rows = make_session_rows("sess", "m", num_turns=5)
    wl = build_workload_from_rows(rows)
    wl.validate_prefix_growth()  # must not raise


def test_validate_prefix_growth_detects_shrink():
    rows = make_session_rows("sess", "m", num_turns=2)
    rows[1]["input"] = rows[1]["input"][:1]  # shorter than turn 0
    wl = build_workload_from_rows(rows)
    with pytest.raises(ValueError, match="fewer messages"):
        wl.validate_prefix_growth()


def test_validate_prefix_growth_detects_divergence():
    rows = make_session_rows("sess", "m", num_turns=2)
    # same length but mutated prefix -> not a superset
    rows[1]["input"][0] = {"role": "system", "content": "DIFFERENT"}
    wl = build_workload_from_rows(rows)
    with pytest.raises(ValueError, match="prefix-superset"):
        wl.validate_prefix_growth()
