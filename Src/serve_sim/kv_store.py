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

from .device_memory import DeviceHbmResidency, MemoryPolicy
from .hardware import MemoryDevice
from .pdd import context_kv_bytes
from .system import System


@dataclass
class KVEntry:
    """One stored sequence's KV cache, resident in a floating or device memory.

    Attributes:
        workload_id: Source conversation identifier.
        turn_index: Turn within that conversation.
        model: The model the KV belongs to (entries of other models never match).
        tracker: The sequence tracker, used for message-aligned prefix matching.
        context_tokens: Full context length (prompt + generated) the KV covers.
        num_bytes: Memory footprint of the stored KV.
        memory: The memory currently holding the entry (a device first-tier HBM
            under a device-first policy, else a floating node memory).
        device: The compute device whose HBM holds the entry, or ``None`` when it
            lives in a floating node memory.
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
    device: object | None = None


@dataclass(frozen=True)
class Match:
    """A prefix-reuse hit: ``prefix_tokens`` of ``entry`` are reusable."""

    entry: KVEntry
    prefix_tokens: int


@dataclass(frozen=True)
class StoreResult:
    """Outcome of storing one sequence's KV.

    ``memory`` is the memory the entry landed on (``None`` if the KV could not be
    stored at all, e.g. no floating memory or it exceeds capacity); ``num_bytes``
    is the entry's footprint; ``evicted`` lists entries dropped entirely (the node
    pool was full); ``spilled`` lists entries pushed off device HBM down to a node
    memory to make room (each carries its new memory in ``KVEntry.memory``).
    """

    memory: MemoryDevice | None
    num_bytes: float
    evicted: tuple[KVEntry, ...]
    spilled: tuple[KVEntry, ...] = ()


class KVCacheManager:
    """System-wide persistent KV cache with prefix reuse, migration and LRU."""

    def __init__(
        self,
        system: System,
        policy: str = "node_first",
        hbm: dict[int, DeviceHbmResidency] | None = None,
    ) -> None:
        self.system = system
        self.policy = policy
        # Per-device first-tier HBM residency, shared with the expert path so
        # retained KV and resident experts contend under the chosen policy. Only
        # populated for the device-first policies; ``node_first`` leaves it empty
        # and behaves exactly as the legacy offload-to-node cache.
        self._hbm = hbm if hbm is not None else {}
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
    def _device_first(self) -> bool:
        return self.policy != "node_first" and bool(self._hbm)

    @property
    def enabled(self) -> bool:
        """Whether there is anywhere to hold KV (node pool or device HBM)."""

        return bool(self._floating) or bool(self._hbm)

    @property
    def entries(self) -> tuple[KVEntry, ...]:
        return tuple(self._entries)

    def used_bytes(self, memory: MemoryDevice) -> float:
        """Stored-KV bytes currently resident on ``memory``."""

        return self._used.get(id(memory), 0.0)

    def spill_residents(self, residents, now: float) -> tuple[tuple[KVEntry, ...], tuple[KVEntry, ...]]:
        """Spill KV that another tenant (e.g. experts) evicted off a device.

        ``residents`` are the device-pool residents the expert path already
        removed from HBM; each is migrated down to node memory (evicting node LRU
        if the pool is full). Returns ``(spilled, dropped)`` :class:`KVEntry`
        tuples -- those relocated to node and those that could not fit and were
        dropped entirely.
        """

        spilled, dropped = self._spill_evicted(residents, now)
        return tuple(spilled), tuple(dropped)

    # --- admission ----------------------------------------------------------

    def lookup(self, model: object, tracker: object, now: float) -> Match | None:
        """Longest message-aligned prefix reusable for ``tracker``, if any.

        Compares ``tracker`` against every resident entry of the same model and
        returns the entry sharing the longest leading-message prefix (refreshing
        that entry's recency, since it is about to be reused). Returns ``None`` if
        nothing shares a prefix or KV reuse is inert.
        """

        if not self.enabled or tracker is None or getattr(tracker, "messages", None) is None:
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
        if best.device is not None and id(best.device) in self._hbm:
            self._hbm[id(best.device)].touch_kv((best.workload_id, best.turn_index), now)
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
        device: object | None = None,
    ) -> StoreResult:
        """Offload a completed sequence's KV; return where it landed.

        Under ``node_first`` the KV goes straight to a floating node memory,
        evicting least-recently-used entries only when the whole floating pool
        cannot otherwise fit it. Under a device-first policy the KV is retained on
        the serving ``device``'s HBM (no transfer -- it is already there); residents
        the device's KV region evicts to make room are *spilled* down to node
        memory, and only evicted entirely when the node pool is full too.
        """

        if not self.enabled or tracker is None:
            return StoreResult(memory=None, num_bytes=0.0, evicted=())
        num_bytes = float(context_kv_bytes(model, context_tokens))
        if num_bytes <= 0:
            return StoreResult(memory=None, num_bytes=0.0, evicted=())

        # A re-store of the same sequence replaces any prior copy.
        self._drop_key(workload_id, turn_index)

        if self._device_first and device is not None and id(device) in self._hbm:
            return self._store_on_device(
                model, tracker, workload_id, turn_index, context_tokens,
                num_bytes, device, now,
            )
        return self._store_on_node(
            model, tracker, workload_id, turn_index, context_tokens, num_bytes, now,
        )

    def _store_on_device(
        self, model, tracker, workload_id, turn_index, context_tokens,
        num_bytes, device, now,
    ) -> StoreResult:
        residency = self._hbm[id(device)]
        admitted, evicted = residency.admit_kv(
            (workload_id, turn_index), num_bytes, now
        )
        if not admitted:
            # The block is larger than the device's whole KV region: fall back to
            # holding it in node memory rather than dropping it.
            return self._store_on_node(
                model, tracker, workload_id, turn_index, context_tokens,
                num_bytes, now,
            )
        entry = KVEntry(
            workload_id=workload_id,
            turn_index=turn_index,
            model=model,
            tracker=tracker,
            context_tokens=context_tokens,
            num_bytes=num_bytes,
            memory=device.first_tier_memory,
            last_use=now,
            device=device,
        )
        self._entries.append(entry)
        spilled, dropped = self._spill_evicted(evicted, now)
        return StoreResult(
            memory=device.first_tier_memory,
            num_bytes=num_bytes,
            evicted=tuple(dropped),
            spilled=tuple(spilled),
        )

    def _store_on_node(
        self, model, tracker, workload_id, turn_index, context_tokens,
        num_bytes, now,
    ) -> StoreResult:
        if not self._floating:
            return StoreResult(memory=None, num_bytes=num_bytes, evicted=())
        evicted: list[KVEntry] = []
        memory = self._node_with_room(num_bytes)
        while memory is None:
            victim = self._lru_node_entry()
            if victim is None:
                return StoreResult(
                    memory=None, num_bytes=num_bytes, evicted=tuple(evicted)
                )
            self._remove(victim)
            evicted.append(victim)
            memory = self._node_with_room(num_bytes)

        entry = KVEntry(
            workload_id=workload_id,
            turn_index=turn_index,
            model=model,
            tracker=tracker,
            context_tokens=context_tokens,
            num_bytes=num_bytes,
            memory=memory,
            last_use=now,
            device=None,
        )
        self._entries.append(entry)
        self._used[id(memory)] += num_bytes
        return StoreResult(memory=memory, num_bytes=num_bytes, evicted=tuple(evicted))

    # --- internals ----------------------------------------------------------

    def _spill_evicted(self, evicted, now) -> tuple[list[KVEntry], list[KVEntry]]:
        """Move HBM residents evicted by an admission down to node memory.

        Each evicted resident names a stored entry; it is migrated to a floating
        node memory (evicting node LRU entries if needed). Entries that cannot fit
        the node pool at all are dropped. Returns ``(spilled, dropped)`` lists of
        the affected :class:`KVEntry`.
        """

        spilled: list[KVEntry] = []
        dropped: list[KVEntry] = []
        for resident in evicted:
            _, key = resident.key  # ("kv", (workload_id, turn_index))
            entry = self._entry_for_key(key)
            if entry is None:
                continue
            memory = self._node_with_room(entry.num_bytes)
            while memory is None:
                victim = self._lru_node_entry()
                if victim is None or victim is entry:
                    break
                self._remove(victim)
                dropped.append(victim)
                memory = self._node_with_room(entry.num_bytes)
            if memory is None:
                self._entries.remove(entry)
                dropped.append(entry)
                continue
            entry.memory = memory
            entry.device = None
            self._used[id(memory)] += entry.num_bytes
            spilled.append(entry)
        return spilled, dropped

    def _entry_for_key(self, key) -> KVEntry | None:
        workload_id, turn_index = key
        for entry in self._entries:
            if entry.workload_id == workload_id and entry.turn_index == turn_index:
                return entry
        return None

    def _drop_key(self, workload_id: int, turn_index: int) -> None:
        entry = self._entry_for_key((workload_id, turn_index))
        if entry is not None:
            self._remove(entry)

    def _node_with_room(self, num_bytes: float) -> MemoryDevice | None:
        """First floating memory whose free capacity holds ``num_bytes``."""

        for memory in self._floating:
            free = memory.capacity_bytes - self._used[id(memory)]
            if free >= num_bytes:
                return memory
        return None

    def _lru_node_entry(self) -> KVEntry | None:
        node_entries = [e for e in self._entries if e.device is None]
        if not node_entries:
            return None
        return min(node_entries, key=lambda e: e.last_use)

    def _remove(self, entry: KVEntry) -> None:
        self._entries.remove(entry)
        if entry.device is not None and id(entry.device) in self._hbm:
            self._hbm[id(entry.device)].remove_kv((entry.workload_id, entry.turn_index))
        else:
            self._used[id(entry.memory)] -= entry.num_bytes
