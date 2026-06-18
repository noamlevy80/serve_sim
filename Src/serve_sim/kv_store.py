"""System-wide persistent KV cache.

The :class:`KVCacheManager` is the orchestrator's single record of every
sequence's KV cache that has not yet been evicted. It exists so that:

* a prefix can be reused *across conversations* (not just between consecutive
  turns of one conversation), by message-aligned prefix comparison against every
  resident entry of the same model;
* the KV may live anywhere in the memory hierarchy, not only on the device that
  last served the sequence. A device's first-tier memory holds KV only while the
  device is computing it; once a turn completes its KV is offloaded to a
  *floating* memory (any node's CPU-managed node memory). The system NVM (the
  weight input device) never holds KV.

Storage competes for floating-memory capacity: as long as some floating memory
has room the entry is kept (migrating across nodes if needed to avoid eviction);
only when the whole floating pool is full are entries evicted, least-recently-used
first, at the granularity of a whole stored sequence.

This module only does the bookkeeping (which prefix to reuse, where KV resides,
what to evict). The orchestrator turns its decisions into modeled, arbiter
accounted transfer events and into ``kv_reuse`` / ``kv_transfer`` / ``kv_eviction``
decision records.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import MemoryDevice
from .pdd import context_kv_bytes
from .system import System


@dataclass
class KVEntry:
    """One stored sequence's KV cache, resident in a floating memory.

    Attributes:
        workload_id: Source conversation identifier.
        turn_index: Turn within that conversation.
        model: The model the KV belongs to (entries of other models never match).
        tracker: The sequence tracker, used for message-aligned prefix matching.
        context_tokens: Full context length (prompt + generated) the KV covers.
        num_bytes: Floating-memory footprint of the stored KV.
        memory: The floating memory currently holding the entry.
        last_use: Simulation time of the most recent store or reuse (for LRU).
    """

    workload_id: int
    turn_index: int
    model: object
    tracker: object
    context_tokens: int
    num_bytes: float
    memory: MemoryDevice
    last_use: float


@dataclass(frozen=True)
class Match:
    """A prefix-reuse hit: ``prefix_tokens`` of ``entry`` are reusable."""

    entry: KVEntry
    prefix_tokens: int


@dataclass(frozen=True)
class StoreResult:
    """Outcome of storing one sequence's KV.

    ``memory`` is the floating memory the entry landed on (``None`` if the KV
    could not be stored at all, e.g. no floating memory or it exceeds capacity);
    ``num_bytes`` is the entry's floating footprint; ``evicted`` lists the entries
    dropped (LRU) to make room.
    """

    memory: MemoryDevice | None
    num_bytes: float
    evicted: tuple[KVEntry, ...]


class KVCacheManager:
    """System-wide persistent KV cache with prefix reuse, migration and LRU."""

    def __init__(self, system: System) -> None:
        self.system = system
        # The floating pool: every node's CPU-managed memory (deduplicated by
        # identity). Device first-tier memory and the input NVM are excluded.
        floating: list[MemoryDevice] = []
        seen: set[int] = set()
        for node in system.nodes:
            memory = node.node_memory
            if memory is not None and id(memory) not in seen:
                seen.add(id(memory))
                floating.append(memory)
        self._floating = floating
        self._used: dict[int, float] = {id(m): 0.0 for m in floating}
        self._entries: list[KVEntry] = []

    @property
    def enabled(self) -> bool:
        """Whether there is any floating memory to hold KV (else inert)."""

        return bool(self._floating)

    @property
    def entries(self) -> tuple[KVEntry, ...]:
        return tuple(self._entries)

    def used_bytes(self, memory: MemoryDevice) -> float:
        """Stored-KV bytes currently resident on ``memory``."""

        return self._used.get(id(memory), 0.0)

    # --- admission ----------------------------------------------------------

    def lookup(self, model: object, tracker: object, now: float) -> Match | None:
        """Longest message-aligned prefix reusable for ``tracker``, if any.

        Compares ``tracker`` against every resident entry of the same model and
        returns the entry sharing the longest leading-message prefix (refreshing
        that entry's recency, since it is about to be reused). Returns ``None`` if
        nothing shares a prefix or KV reuse is inert.
        """

        if not self._floating or tracker is None or getattr(tracker, "messages", None) is None:
            return None
        best: KVEntry | None = None
        best_len = 0
        for entry in self._entries:
            if entry.model is not model or entry.tracker.messages is None:
                continue
            length = tracker.common_prefix_length(entry.tracker)
            if length > best_len:
                best_len = length
                best = entry
        if best is None or best_len <= 0:
            return None
        best.last_use = now
        return Match(entry=best, prefix_tokens=best_len)

    # --- completion ---------------------------------------------------------

    def store(
        self,
        model: object,
        tracker: object,
        workload_id: int,
        turn_index: int,
        context_tokens: int,
        now: float,
    ) -> StoreResult:
        """Offload a completed sequence's KV into a floating memory.

        Picks any floating memory with room (migrating across nodes by choice of
        target rather than evicting), evicting least-recently-used entries only
        when the whole floating pool cannot otherwise fit the new entry. Returns
        the chosen memory and the entries evicted to make room.
        """

        if not self._floating or tracker is None:
            return StoreResult(memory=None, num_bytes=0.0, evicted=())
        num_bytes = float(context_kv_bytes(model, context_tokens))
        if num_bytes <= 0:
            return StoreResult(memory=None, num_bytes=0.0, evicted=())

        evicted: list[KVEntry] = []
        memory = self._memory_with_room(num_bytes)
        while memory is None:
            victim = self._lru_entry()
            if victim is None:
                # Nothing left to evict and still no room: the entry is larger
                # than any single floating memory; give up storing it.
                return StoreResult(
                    memory=None, num_bytes=num_bytes, evicted=tuple(evicted)
                )
            self._remove(victim)
            evicted.append(victim)
            memory = self._memory_with_room(num_bytes)

        entry = KVEntry(
            workload_id=workload_id,
            turn_index=turn_index,
            model=model,
            tracker=tracker,
            context_tokens=context_tokens,
            num_bytes=num_bytes,
            memory=memory,
            last_use=now,
        )
        self._entries.append(entry)
        self._used[id(memory)] += num_bytes
        return StoreResult(memory=memory, num_bytes=num_bytes, evicted=tuple(evicted))

    # --- internals ----------------------------------------------------------

    def _memory_with_room(self, num_bytes: float) -> MemoryDevice | None:
        """First floating memory whose free capacity holds ``num_bytes``."""

        for memory in self._floating:
            free = memory.capacity_bytes - self._used[id(memory)]
            if free >= num_bytes:
                return memory
        return None

    def _lru_entry(self) -> KVEntry | None:
        if not self._entries:
            return None
        return min(self._entries, key=lambda e: e.last_use)

    def _remove(self, entry: KVEntry) -> None:
        self._entries.remove(entry)
        self._used[id(entry.memory)] -= entry.num_bytes
