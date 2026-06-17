"""Sequence and batch tracking.

A :class:`SequenceTracker` represents one conversation's token bookkeeping for a
single turn: how many tokens are already cached (from previous turns), how many
new prompt tokens must be prefilled, and how many tokens will be decoded. A
:class:`BatchTracker` groups several sequences that are executed together.

These trackers translate a tokenized workload turn into the index information the
work-shard generator consumes (last cached / prefill / decode token indices).
"""

from __future__ import annotations

from dataclasses import dataclass

from .kv_cache import KVCacheTracker
from .tokenizer import Tokenizer
from .workload import Workload


@dataclass(frozen=True)
class SequenceWork:
    """Per-sequence work description for one turn.

    Attributes:
        cached_tokens: Tokens already present in the KV cache before this turn.
        prefill_tokens: New prompt tokens to prefill this turn.
        decode_tokens: Tokens to generate (decode) this turn.
    """

    cached_tokens: int
    prefill_tokens: int
    decode_tokens: int

    def __post_init__(self) -> None:
        if self.cached_tokens < 0 or self.prefill_tokens < 0 or self.decode_tokens < 0:
            raise ValueError("token counts must be non-negative")

    @property
    def base_tokens(self) -> int:
        """Context length once prefill completes (cached + prefilled)."""

        return self.cached_tokens + self.prefill_tokens


class SequenceTracker:
    """Token bookkeeping for one sequence's turn.

    Construct directly with token counts, or via :meth:`from_turn` to tokenize a
    workload turn with a supplied tokenizer. When built from a turn the tracker
    also remembers the turn's messages and their per-message token counts, which
    lets two trackers find the token-length of their common (message-aligned)
    prefix and link their per-layer KV trackers for reuse.
    """

    def __init__(
        self,
        prompt_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        messages: tuple | None = None,
        message_token_counts: tuple[int, ...] | None = None,
    ) -> None:
        if prompt_tokens < 0 or output_tokens < 0 or cached_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if cached_tokens > prompt_tokens:
            raise ValueError(
                f"cached_tokens ({cached_tokens}) cannot exceed prompt_tokens "
                f"({prompt_tokens})"
            )
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.cached_tokens = cached_tokens
        self.messages = messages
        self._message_token_counts = message_token_counts
        self.kv_trackers: list[KVCacheTracker] | None = None

    @property
    def prefill_tokens(self) -> int:
        """New prompt tokens to prefill this turn (prompt minus cached prefix)."""

        return self.prompt_tokens - self.cached_tokens

    # --- token index helpers (as the shard generator expects) ---------------

    @property
    def last_cache_index(self) -> int | None:
        """Index of the last cached token, or ``None`` if nothing is cached."""

        return self.cached_tokens - 1 if self.cached_tokens > 0 else None

    @property
    def last_prefill_index(self) -> int:
        """Index of the last prompt token after prefill (cached + new - 1)."""

        return self.prompt_tokens - 1

    @property
    def last_decode_index(self) -> int:
        """Index of the last decoded token for this turn."""

        return self.prompt_tokens + self.output_tokens - 1

    def to_work(self) -> SequenceWork:
        return SequenceWork(
            cached_tokens=self.cached_tokens,
            prefill_tokens=self.prefill_tokens,
            decode_tokens=self.output_tokens,
        )

    @classmethod
    def from_turn(
        cls,
        workload: Workload,
        turn_index: int,
        tokenizer: Tokenizer,
    ) -> "SequenceTracker":
        """Build a tracker by tokenizing one turn of a workload.

        Tokens are counted per message and summed, so the previous turn's tokens
        form an exact prefix of this turn's tokens.
        """

        turn = workload[turn_index]
        counts = tuple(_message_token_count(m, tokenizer) for m in turn.messages)
        prompt_tokens = sum(counts)
        if turn_index == 0:
            cached = 0
        else:
            cached = _count_messages(workload[turn_index - 1].messages, tokenizer)
        return cls(
            prompt_tokens=prompt_tokens,
            output_tokens=turn.output_length,
            cached_tokens=cached,
            messages=turn.messages,
            message_token_counts=counts,
        )

    # --- cross-sequence prefix sharing --------------------------------------

    def common_prefix_length(self, other: "SequenceTracker") -> int:
        """Token-length of the leading messages identical to ``other``.

        The prefix is message-aligned (matching how prompts are tokenized): it
        sums the token counts of the leading messages that are equal between the
        two sequences, stopping at the first message that differs. Both trackers
        must have been built from a turn (so they carry messages).
        """

        if self.messages is None or other.messages is None:
            raise ValueError(
                "common_prefix_length requires trackers built from a turn"
            )
        total = 0
        for (m_a, c_a), m_b in zip(
            zip(self.messages, self._message_token_counts), other.messages
        ):
            if m_a == m_b:
                total += c_a
            else:
                break
        return total

    def attach_kv_trackers(self, num_layers: int) -> list[KVCacheTracker]:
        """Create one :class:`KVCacheTracker` per layer, owned by this tracker."""

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.kv_trackers = [KVCacheTracker() for _ in range(num_layers)]
        return self.kv_trackers

    def link_kv_prefix(self, other: "SequenceTracker") -> int:
        """Link this sequence's KV trackers to ``other`` over their common prefix.

        Both trackers must already have the same number of attached KV trackers.
        Returns the linked prefix length in tokens.
        """

        if self.kv_trackers is None or other.kv_trackers is None:
            raise ValueError("both sequences must have attached KV trackers")
        if len(self.kv_trackers) != len(other.kv_trackers):
            raise ValueError("trackers have different layer counts")
        length = self.common_prefix_length(other)
        for mine, theirs in zip(self.kv_trackers, other.kv_trackers):
            theirs.ensure_length(length)
            mine.link_prefix(theirs, length)
        return length


def _message_token_count(message, tokenizer: Tokenizer) -> int:
    total = 0
    if message.content:
        total += tokenizer.count(message.content)
    for tool_call in message.tool_calls:
        if tool_call.function_arguments:
            total += tokenizer.count(tool_call.function_arguments)
    return total


def _count_messages(messages, tokenizer: Tokenizer) -> int:
    return sum(_message_token_count(m, tokenizer) for m in messages)


class BatchTracker:
    """A batch of sequences executed together in one turn."""

    def __init__(self, sequences: list[SequenceTracker]):
        if not sequences:
            raise ValueError("a batch must contain at least one sequence")
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self):
        return iter(self.sequences)

    def work(self) -> list[SequenceWork]:
        """Per-sequence work descriptions for the work-shard generator."""

        return [seq.to_work() for seq in self.sequences]
