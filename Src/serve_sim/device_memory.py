"""Per-device HBM residency for routed experts and retained KV.

A compute device's first-tier memory (HBM) holds three kinds of bytes:

* **pinned weights** -- the non-expert weights of the model placed on the device,
  always resident while that model occupies the slot;
* **routed experts** -- the working set of MoE experts streamed up on demand and
  kept resident so later groups/batches reuse them;
* **retained KV** -- completed sequences' KV kept on the device so a later turn
  (the same conversation, or another sharing its prefix) can reuse it *in place*
  rather than re-fetching it from node memory.

The experts and the retained KV compete for whatever HBM the pinned weights leave
free (the *dynamic region*). This module models that contention under two
selectable policies:

* :attr:`MemoryPolicy.GLOBAL_LRU` -- experts and KV share one least-recently-used
  pool over the whole dynamic region; admitting either kind may evict the
  globally least-recently-used resident of *either* kind.
* :attr:`MemoryPolicy.PARTITIONED` -- the dynamic region is split into a KV
  sub-region and an expert sub-region by a fixed fraction; each runs its own LRU
  and never evicts the other kind.

The module is pure bookkeeping: it stores no tensors and does no timing. It only
decides what stays resident and what is evicted (so the orchestrator can turn an
eviction into a spill transfer to node memory, and a miss into a fetch). Eviction
is by ``last_use`` recency; the caller may *protect* keys (e.g. the working set of
the group currently executing) so they are never evicted to admit their peers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MemoryPolicy(str, Enum):
    """How routed experts and retained KV share a device's dynamic HBM region."""

    GLOBAL_LRU = "global_lru"
    PARTITIONED = "partitioned"


@dataclass
class Resident:
    """One resident item in a device HBM pool.

    Attributes:
        key: Caller-supplied identity (unique within its kind).
        kind: ``"expert"`` or ``"kv"``.
        num_bytes: First-tier footprint of the item.
        last_use: Simulation time of the most recent admit/touch (for LRU).
    """

    key: object
    kind: str
    num_bytes: float
    last_use: float


class _LruPool:
    """A fixed-capacity byte pool that evicts least-recently-used residents.

    Admitting an item that does not fit evicts the lowest-``last_use`` residents
    (excluding any the caller protects) until it does, or reports failure when the
    item is larger than the whole pool. The pool is agnostic to item kind, so the
    same class backs both the shared global pool and each partitioned sub-pool.
    """

    def __init__(self, capacity: float) -> None:
        self.capacity = max(0.0, float(capacity))
        self._items: dict[object, Resident] = {}
        self._used = 0.0

    @property
    def used(self) -> float:
        return self._used

    @property
    def free(self) -> float:
        return self.capacity - self._used

    def contains(self, key: object) -> bool:
        return key in self._items

    def get(self, key: object) -> Resident | None:
        return self._items.get(key)

    def touch(self, key: object, now: float) -> None:
        """Refresh an item's recency without changing its size (no-op if absent)."""

        item = self._items.get(key)
        if item is not None:
            item.last_use = now

    def admit(
        self,
        key: object,
        kind: str,
        num_bytes: float,
        now: float,
        protected: frozenset[object] = frozenset(),
    ) -> tuple[bool, list[Resident]]:
        """Insert or refresh ``key``; evict LRU to fit. Returns (admitted, evicted).

        An already-resident key is resized and its recency refreshed. A new key
        that exceeds the whole pool capacity is rejected (``admitted=False``) with
        no eviction. Otherwise the key is inserted and the least-recently-used
        residents not in ``protected`` are evicted until the pool fits; the
        admitted key is always protected from its own admission.
        """

        num_bytes = float(num_bytes)
        existing = self._items.get(key)
        if existing is not None:
            self._used += num_bytes - existing.num_bytes
            existing.num_bytes = num_bytes
            existing.last_use = now
            return True, self._evict_until_fit(protected | {key})
        if num_bytes > self.capacity + 1e-9:
            return False, []
        self._items[key] = Resident(key, kind, num_bytes, now)
        self._used += num_bytes
        return True, self._evict_until_fit(protected | {key})

    def remove(self, key: object) -> Resident | None:
        """Drop ``key`` (no-op returning ``None`` if absent)."""

        item = self._items.pop(key, None)
        if item is not None:
            self._used -= item.num_bytes
        return item

    def residents(self) -> tuple[Resident, ...]:
        return tuple(self._items.values())

    def _evict_until_fit(self, protected: frozenset[object]) -> list[Resident]:
        evicted: list[Resident] = []
        while self._used > self.capacity + 1e-9:
            victim = self._lru(protected)
            if victim is None:
                break
            del self._items[victim.key]
            self._used -= victim.num_bytes
            evicted.append(victim)
        return evicted

    def _lru(self, protected: frozenset[object]) -> Resident | None:
        candidates = [r for r in self._items.values() if r.key not in protected]
        if not candidates:
            return None
        return min(candidates, key=lambda r: r.last_use)


class DeviceHbmResidency:
    """The resident routed experts and retained KV of one device's first tier.

    The dynamic region is the first-tier capacity left after the model's pinned
    (non-expert) weights. Under :attr:`MemoryPolicy.GLOBAL_LRU` experts and KV
    share that whole region in a single LRU; under
    :attr:`MemoryPolicy.PARTITIONED` it is split into a KV sub-region of
    ``kv_fraction`` and an expert sub-region of the remainder, each LRU-managed in
    isolation. Expert keys and KV keys live in separate namespaces, so a caller's
    expert index and KV identifier never collide.
    """

    def __init__(
        self,
        dynamic_capacity: float,
        policy: MemoryPolicy = MemoryPolicy.GLOBAL_LRU,
        kv_fraction: float = 0.5,
    ) -> None:
        if not 0.0 < kv_fraction < 1.0:
            raise ValueError("kv_fraction must be in (0, 1)")
        capacity = max(0.0, float(dynamic_capacity))
        self.policy = policy
        self.capacity = capacity
        if policy is MemoryPolicy.PARTITIONED:
            self._kv = _LruPool(capacity * kv_fraction)
            self._experts = _LruPool(capacity * (1.0 - kv_fraction))
        else:
            shared = _LruPool(capacity)
            self._kv = shared
            self._experts = shared

    @property
    def used(self) -> float:
        """Total resident bytes (experts + KV) across the region(s)."""

        if self._kv is self._experts:
            return self._kv.used
        return self._kv.used + self._experts.used

    # --- experts ------------------------------------------------------------

    def expert_resident(self, index: int) -> bool:
        return self._experts.contains(("expert", index))

    def touch_expert(self, index: int, now: float) -> None:
        self._experts.touch(("expert", index), now)

    def clear_experts(self) -> None:
        """Drop every resident expert (e.g. when a new model's weights load).

        Retained KV in the (shared or partitioned) region is left untouched.
        """

        for resident in self._experts.residents():
            if resident.kind == "expert":
                self._experts.remove(resident.key)

    def admit_expert(
        self,
        index: int,
        num_bytes: float,
        now: float,
        protected_experts: frozenset[int] = frozenset(),
    ) -> list[Resident]:
        """Make expert ``index`` resident; return residents evicted to fit it.

        ``protected_experts`` are expert indices that must not be evicted (the
        active set of the executing group); they are protected alongside the
        admitted index.
        """

        protected = frozenset(("expert", i) for i in protected_experts)
        _, evicted = self._experts.admit(
            ("expert", index), "expert", num_bytes, now, protected
        )
        return evicted

    def preload_experts(
        self, indices: list[int], index_bytes: float, now: float
    ) -> bool:
        """Pin a rank's full expert shard as part of its model's weights.

        Experts are part of the weights, so they are made resident together with
        the model. Every index in ``indices`` (each costing ``index_bytes``) is
        admitted, but only when the whole shard fits the expert region; the set is
        admitted atomically so no preloaded expert evicts another. Returns
        ``True`` when the shard was pinned, ``False`` when it cannot fit -- the
        documented exception, leaving the rank to stream its working set on
        demand. Any KV already retained in a shared region is left untouched.
        """

        index_bytes = float(index_bytes)
        if index_bytes <= 0.0 or not indices:
            return False
        if len(indices) * index_bytes > self._experts.free + 1e-9:
            return False
        for index in indices:
            self._experts.admit(("expert", index), "expert", index_bytes, now)
        return True

    def access_experts(
        self,
        active: frozenset[int],
        index_bytes: float,
        now: float,
    ) -> tuple[list[int], list[int], list[Resident]]:
        """Touch a group's active experts; admit the misses, evicting LRU to fit.

        ``active`` are the routed-expert indices the executing group needs (the
        working set); they are protected from eviction for the duration of the
        admission so the set is never dropped mid-group. Returns
        ``(missed, evicted_experts, evicted_kv)``: the expert indices that were not
        resident and had to be fetched, the expert indices the LRU dropped, and the
        :class:`Resident` records of any *KV* the admissions evicted (only possible
        under :attr:`MemoryPolicy.GLOBAL_LRU`, where experts and KV share a pool) so
        the caller can spill that KV to node memory.
        """

        missed: list[int] = []
        for index in sorted(active):
            if self.expert_resident(index):
                self.touch_expert(index, now)
            else:
                missed.append(index)
        protected = frozenset(active)
        evicted_experts: list[int] = []
        evicted_kv: list[Resident] = []
        for index in missed:
            for victim in self.admit_expert(index, index_bytes, now, protected):
                if victim.kind == "expert":
                    evicted_experts.append(victim.key[1])
                else:
                    evicted_kv.append(victim)
        return missed, evicted_experts, evicted_kv


    # --- KV -----------------------------------------------------------------

    def kv_resident(self, key: object) -> bool:
        return self._kv.contains(("kv", key))

    def kv_bytes(self, key: object) -> float | None:
        item = self._kv.get(("kv", key))
        return item.num_bytes if item is not None else None

    def touch_kv(self, key: object, now: float) -> None:
        self._kv.touch(("kv", key), now)

    def admit_kv(
        self, key: object, num_bytes: float, now: float
    ) -> tuple[bool, list[Resident]]:
        """Retain KV ``key`` on the device; return (admitted, evicted residents).

        ``admitted`` is ``False`` only when the block is larger than its whole
        (sub-)region, in which case it cannot be retained on the device at all and
        the caller must keep it in node memory.
        """

        return self._kv.admit(("kv", key), "kv", num_bytes, now)

    def remove_kv(self, key: object) -> Resident | None:
        return self._kv.remove(("kv", key))
