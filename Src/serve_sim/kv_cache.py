"""KV-cache residency tracking.

A :class:`KVCacheTracker` is the per-``(model, conversation, layer)`` bookkeeping
the PRD calls for: for every token index in a sequence it records whether that
token's KV has been *generated* yet, the set of memory devices the KV currently
*resides* on, and -- to model prefix reuse across sequences -- an optional *link*
to the same index in another tracker.

Residency is tracked by object identity (``id(device)``), matching the resource
arbiter: two value-equal :class:`~serve_sim.hardware.MemoryDevice` instances are
distinct locations. A *linked* token owns no local state; it inherits the
generated flag and residency of the tracker it points at, which is how a freshly
admitted sequence reuses the KV a previous sequence already prefilled for a shared
prefix.

This module is pure bookkeeping: it stores no tensors and does no timing. The
event generator and orchestrator consume it to decide what must be (re)computed or
moved.
"""

from __future__ import annotations

from .hardware import MemoryDevice


class KVCacheTracker:
    """Per-layer KV residency for one sequence's tokens.

    Token indices are ``0 .. len(self) - 1`` in sequence order. New tokens start
    *ungenerated* and resident nowhere. Generated tokens may reside on zero or
    more devices (zero means generated but since evicted everywhere). A linked
    token delegates all of its state to the source tracker at the same index.
    """

    def __init__(self) -> None:
        self._generated: list[bool] = []
        # Per token: id(device) -> device, so value-equal instances stay distinct.
        self._devices: list[dict[int, MemoryDevice]] = []
        # Per token: a source tracker this index is shared from (same index).
        self._links: list[KVCacheTracker | None] = []

    # --- length -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._generated)

    def extend(self, count: int) -> None:
        """Append ``count`` ungenerated, unplaced tokens to the sequence."""

        if count < 0:
            raise ValueError("count must be non-negative")
        for _ in range(count):
            self._generated.append(False)
            self._devices.append({})
            self._links.append(None)

    def ensure_length(self, n: int) -> None:
        """Grow the sequence to at least ``n`` tokens (no-op if already longer)."""

        if n > len(self):
            self.extend(n - len(self))

    # --- state mutation -----------------------------------------------------

    def _check_range(self, start: int, end: int) -> None:
        if start < 0 or end > len(self) or start > end:
            raise IndexError(f"range [{start}, {end}) out of bounds for length {len(self)}")

    def mark_generated(self, start: int, end: int) -> None:
        """Mark tokens ``[start, end)`` as generated (KV computed)."""

        self._check_range(start, end)
        for i in range(start, end):
            self._generated[i] = True

    def place(self, start: int, end: int, device: MemoryDevice) -> None:
        """Record that tokens ``[start, end)`` reside on ``device``.

        Storing KV implies it was generated, so this also marks the range
        generated.
        """

        self._check_range(start, end)
        for i in range(start, end):
            self._generated[i] = True
            self._devices[i][id(device)] = device

    def evict(self, start: int, end: int, device: MemoryDevice) -> None:
        """Remove ``device`` from the residency of tokens ``[start, end)``."""

        self._check_range(start, end)
        for i in range(start, end):
            self._devices[i].pop(id(device), None)

    def link_prefix(self, source: "KVCacheTracker", length: int) -> None:
        """Share the first ``length`` tokens from ``source`` (same indices).

        The linked range must exist in ``source``; this tracker is grown to cover
        it if needed. Linked tokens delegate their state to ``source``.
        """

        if length < 0:
            raise ValueError("length must be non-negative")
        if length > len(source):
            raise ValueError(
                f"cannot link {length} tokens from a source of length {len(source)}"
            )
        self.ensure_length(length)
        for i in range(length):
            self._links[i] = source

    # --- queries ------------------------------------------------------------

    def is_linked(self, index: int) -> bool:
        return self._links[index] is not None

    def is_generated(self, index: int) -> bool:
        link = self._links[index]
        if link is not None:
            return link.is_generated(index)
        return self._generated[index]

    def devices_at(self, index: int) -> list[MemoryDevice]:
        """Memory devices holding this token's KV (following a link if any)."""

        link = self._links[index]
        if link is not None:
            return link.devices_at(index)
        return list(self._devices[index].values())

    def is_resident(self, index: int, device: MemoryDevice) -> bool:
        link = self._links[index]
        if link is not None:
            return link.is_resident(index, device)
        return id(device) in self._devices[index]

    def is_available(self, index: int) -> bool:
        """Generated and resident on at least one device (locally or via link)."""

        return self.is_generated(index) and bool(self.devices_at(index))

    def cached_prefix_length(self) -> int:
        """Number of leading tokens that are available for reuse without compute."""

        count = 0
        for i in range(len(self)):
            if not self.is_available(i):
                break
            count += 1
        return count
