"""Shared pytest fixtures: an in-memory fake of the datasets-server row API."""

from __future__ import annotations

from typing import Any, Mapping

import pytest


def make_row(
    session_id: str,
    model: str,
    input_messages: list[dict[str, Any]],
    output_length: int = 0,
    pre_gap: float = 0.0,
) -> dict[str, Any]:
    """Build a dataset row dict matching the source schema."""

    return {
        "session_id": session_id,
        "model": model,
        "input": input_messages,
        "output_length": output_length,
        "pre_gap": pre_gap,
    }


def make_session_rows(
    session_id: str,
    model: str,
    num_turns: int,
    messages_per_turn: int = 2,
) -> list[dict[str, Any]]:
    """Build ``num_turns`` rows that satisfy the prefix-growth property.

    Turn ``t`` contains ``2 + t * messages_per_turn`` messages: a fixed system +
    user prefix followed by appended assistant/tool messages.
    """

    base = [
        {"role": "system", "content": f"system::{session_id}"},
        {"role": "user", "content": f"user::{session_id}"},
    ]
    rows: list[dict[str, Any]] = []
    messages = list(base)
    for t in range(num_turns):
        if t > 0:
            for m in range(messages_per_turn):
                messages.append(
                    {"role": "assistant", "content": f"turn{t}-msg{m}"}
                )
        rows.append(
            make_row(
                session_id=session_id,
                model=model,
                input_messages=[dict(msg) for msg in messages],
                output_length=10 * (t + 1),
                pre_gap=0.0 if t == 0 else float(t),
            )
        )
    return rows


class FakeRowFetcher:
    """In-memory :class:`RowFetcher` over a flat list of rows.

    Records every requested ``(offset, length)`` so tests can assert on paging.
    """

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
        self.calls: list[tuple[int, int]] = []

    def __call__(self, offset: int, length: int) -> Mapping[str, Any]:
        self.calls.append((offset, length))
        window = self.rows[offset : offset + length]
        return {
            "rows": [{"row_idx": offset + i, "row": row} for i, row in enumerate(window)],
            "num_rows_total": len(self.rows),
        }


@pytest.fixture
def fake_dataset() -> list[dict[str, Any]]:
    """Three contiguous sessions of differing lengths (8 rows total)."""

    rows: list[dict[str, Any]] = []
    rows += make_session_rows("sess-a", "model-x", num_turns=3)
    rows += make_session_rows("sess-b", "model-y", num_turns=1)
    rows += make_session_rows("sess-c", "model-x", num_turns=4)
    return rows


@pytest.fixture
def fake_fetcher(fake_dataset) -> FakeRowFetcher:
    return FakeRowFetcher(fake_dataset)
