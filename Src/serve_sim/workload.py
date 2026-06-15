"""Workload data model.

A *workload* is a single multi-turn agentic conversation (one ``session_id`` in
the source dataset). Each turn corresponds to one LLM iteration: the cumulative
list of input messages, the number of tokens generated, and the client-side gap
that preceded the request.

The source dataset guarantees that the input of turn ``N`` is a strict
prefix-superset of the input of turn ``N-1`` (messages are only appended at the
end). This module preserves that structure and exposes the per-turn message
delta, which is what the KV-cache machinery downstream needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ToolCall:
    """A single tool/function call requested by an assistant message."""

    id: str | None
    type: str | None
    function_name: str | None
    function_arguments: str | None

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "ToolCall":
        function = raw.get("function") or {}
        return cls(
            id=raw.get("id"),
            type=raw.get("type"),
            function_name=function.get("name"),
            function_arguments=function.get("arguments"),
        )


@dataclass(frozen=True)
class Message:
    """A single chat message in OpenAI message format."""

    role: str
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "Message":
        if "role" not in raw:
            raise ValueError("message is missing required 'role' field")
        raw_tool_calls = raw.get("tool_calls") or ()
        tool_calls = tuple(ToolCall.from_raw(tc) for tc in raw_tool_calls)
        return cls(
            role=raw["role"],
            content=raw.get("content"),
            tool_calls=tool_calls,
            tool_call_id=raw.get("tool_call_id"),
            name=raw.get("name"),
        )


@dataclass(frozen=True)
class Turn:
    """One LLM iteration within a workload.

    Attributes:
        index: Zero-based position of the turn within the workload.
        messages: Full cumulative input messages for this iteration.
        output_length: Completion tokens generated for this iteration.
        pre_gap: Seconds between the previous iteration completing and this
            request being sent (tool-execution / think time). ``0.0`` for the
            first turn.
    """

    index: int
    messages: tuple[Message, ...]
    output_length: int
    pre_gap: float

    def __len__(self) -> int:
        return len(self.messages)


@dataclass
class Workload:
    """A multi-turn agentic conversation ready for simulation."""

    session_id: str
    model: str
    turns: list[Turn] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError(f"workload '{self.session_id}' has no turns")

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    def __len__(self) -> int:
        return len(self.turns)

    def __iter__(self):
        return iter(self.turns)

    def __getitem__(self, index: int) -> Turn:
        return self.turns[index]

    def new_messages(self, turn_index: int) -> tuple[Message, ...]:
        """Messages appended in ``turn_index`` relative to the previous turn.

        For the first turn this is the entire message list. Assumes the
        prefix-growth property holds; call :meth:`validate_prefix_growth` first
        if the data source is untrusted.
        """

        turn = self.turns[turn_index]
        if turn_index == 0:
            return turn.messages
        previous = self.turns[turn_index - 1].messages
        return turn.messages[len(previous):]

    def validate_prefix_growth(self) -> None:
        """Verify each turn's messages extend the previous turn's messages.

        Raises:
            ValueError: If any turn is not a prefix-superset of its predecessor.
        """

        for i in range(1, len(self.turns)):
            previous = self.turns[i - 1].messages
            current = self.turns[i].messages
            if len(current) < len(previous):
                raise ValueError(
                    f"workload '{self.session_id}' turn {i} has fewer messages "
                    f"({len(current)}) than turn {i - 1} ({len(previous)})"
                )
            if current[: len(previous)] != previous:
                raise ValueError(
                    f"workload '{self.session_id}' turn {i} is not a "
                    f"prefix-superset of turn {i - 1}"
                )


def _parse_messages(raw_messages: Sequence[Mapping[str, Any]]) -> tuple[Message, ...]:
    return tuple(Message.from_raw(m) for m in raw_messages)


def build_workload_from_rows(rows: Sequence[Mapping[str, Any]]) -> Workload:
    """Assemble a :class:`Workload` from ordered dataset rows of one session.

    Args:
        rows: Dataset row dicts (each with ``session_id``, ``model``, ``input``,
            ``output_length``, ``pre_gap``) belonging to a single session, in
            turn order.

    Raises:
        ValueError: If ``rows`` is empty or contains more than one session.
    """

    if not rows:
        raise ValueError("cannot build a workload from zero rows")

    session_ids = {row["session_id"] for row in rows}
    if len(session_ids) != 1:
        raise ValueError(
            f"rows span multiple sessions: {sorted(session_ids)}"
        )

    first = rows[0]
    turns = [
        Turn(
            index=i,
            messages=_parse_messages(row["input"]),
            output_length=int(row["output_length"]),
            pre_gap=float(row["pre_gap"]),
        )
        for i, row in enumerate(rows)
    ]
    return Workload(
        session_id=first["session_id"],
        model=first["model"],
        turns=turns,
    )
