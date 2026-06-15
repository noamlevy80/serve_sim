"""Tokenization.

The simulator uses a single tokenizer only to obtain token *counts/indices* from
conversation text -- the actual token values are never used for inference. The
default is tiktoken's ``cl100k_base``. A :class:`Tokenizer` protocol allows
injecting a deterministic fake in tests so the core logic stays offline.

Messages are tokenized per-message and counts are summed, which guarantees that
a prefix-superset message list (turn N extends turn N-1) yields a prefix-superset
token sequence -- exactly the property the KV-cache reuse logic relies on.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Anything that can count the tokens in a piece of text."""

    def count(self, text: str) -> int:
        ...


class TiktokenTokenizer:
    """Real tokenizer backed by tiktoken (default ``cl100k_base``)."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken  # imported lazily so offline tests need no dependency

        self.encoding_name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))


class WhitespaceTokenizer:
    """Deterministic, dependency-free tokenizer: one token per whitespace word.

    Intended for tests and offline development, not for accurate sizing.
    """

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())
