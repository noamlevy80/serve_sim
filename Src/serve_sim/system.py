"""Parse a system configuration JSON into live hardware objects.

A *system* config (see ``Systems/*.json``) describes a whole simulated machine:
a scale-up + CXL :class:`Network`, a designated *input memory* (the shared NVM
that holds every model's weights at init), and a list of :class:`Node` s. Each
node owns an optional CPU-managed node memory and some compute devices, each of
which is an *instance* of a device config in ``Compute_devices/`` (whose
first-tier memory in turn resolves from ``Memory_devices/``).

Instances matter: the event generator and resource arbiter contend on object
identity, so a node with ``{"device": "nvidia-b200", "count": 4}`` is expanded
into four *distinct* :class:`~serve_sim.hardware.ComputeDevice` objects, each with
its own distinct first-tier memory and a unique, node-qualified name (e.g.
``"NVIDIA B200 [node-0 #2]"``) so the instances stay distinguishable in
per-device reports. The single input memory is one shared instance; each node's
node memory is one instance per node, given a node-qualified name (e.g.
``"NVIDIA Grace LPDDR5X [node-0]"``) so same-type node memories stay
distinguishable in per-memory reports.

A whole node can likewise be replicated with a node-level ``"count"``: a node
entry with ``"count": 8`` is expanded into eight *distinct* :class:`Node` s,
each given a unique name (the base name qualified with a per-copy index, e.g.
``"pod #2"``) so the node memory and every compute-device instance -- both of
which qualify their names on the owning node -- stay conflict-free across copies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from .device_config import load_compute_device, load_memory_device
from .hardware import ComputeDevice, MemoryDevice
from .transfer import INTRA_PACKAGE, TransferLink


@dataclass(frozen=True)
class Network:
    """Point-to-point fabric parameters shared by the whole system.

    Attributes:
        scale_up_bandwidth_bytes_per_s: Cross-device scale-up bandwidth.
        scale_up_latency_s: Scale-up per-transfer latency.
        cxl_bandwidth_bytes_per_s: In-node CXL bandwidth (to node memory).
        cxl_latency_s: In-node CXL per-transfer latency.
    """

    scale_up_bandwidth_bytes_per_s: float
    scale_up_latency_s: float
    cxl_bandwidth_bytes_per_s: float
    cxl_latency_s: float

    def __post_init__(self) -> None:
        if self.scale_up_bandwidth_bytes_per_s <= 0:
            raise ValueError("scale_up_bandwidth_bytes_per_s must be positive")
        if self.cxl_bandwidth_bytes_per_s <= 0:
            raise ValueError("cxl_bandwidth_bytes_per_s must be positive")
        if self.scale_up_latency_s < 0 or self.cxl_latency_s < 0:
            raise ValueError("network latencies must be non-negative")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Network":
        return cls(
            scale_up_bandwidth_bytes_per_s=config["scale_up_bandwidth_bytes_per_s"],
            scale_up_latency_s=config["scale_up_latency_s"],
            cxl_bandwidth_bytes_per_s=config["cxl_bandwidth_bytes_per_s"],
            cxl_latency_s=config["cxl_latency_s"],
        )


@dataclass(frozen=True)
class Node:
    """One node: a CPU-managed node memory and its compute devices.

    Attributes:
        name: Identifier for logs/reports.
        compute_devices: Distinct compute-device instances in this node.
        node_memory: Optional CPU-managed memory reached over CXL in-node (or the
            scale-up network from outside the node). ``None`` if unmodeled.
    """

    name: str
    compute_devices: tuple[ComputeDevice, ...]
    node_memory: MemoryDevice | None = None

    def __len__(self) -> int:
        return len(self.compute_devices)


@dataclass(frozen=True)
class System:
    """A fully instantiated simulated machine.

    Attributes:
        name: Identifier for logs/reports.
        network: Scale-up + CXL fabric parameters.
        input_memory: Shared NVM holding all model weights at init.
        nodes: The system's nodes.
        description: Optional free-text description from the config.
    """

    name: str
    network: Network
    input_memory: MemoryDevice
    nodes: tuple[Node, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("a system must have at least one node")

    @property
    def compute_devices(self) -> list[ComputeDevice]:
        """Every compute device across all nodes, in node order."""

        return [d for node in self.nodes for d in node.compute_devices]

    @property
    def num_compute_devices(self) -> int:
        return sum(len(node) for node in self.nodes)

    def node_of(self, device: ComputeDevice) -> Node:
        """The node that owns ``device`` (by identity)."""

        node = self._node_by_device.get(id(device))
        if node is not None:
            return node
        raise ValueError(f"device {device.name!r} is not part of this system")

    @property
    def _node_by_device(self) -> dict[int, Node]:
        """``id(compute device) -> owning Node``, built once (first owner wins)."""

        cache = self.__dict__.get("_node_by_device_cache")
        if cache is None:
            cache = {}
            for node in self.nodes:
                for device in node.compute_devices:
                    cache.setdefault(id(device), node)
            object.__setattr__(self, "_node_by_device_cache", cache)
        return cache

    def memory_inventory(self) -> list[dict[str, Any]]:
        """Every distinct memory device with its role, node and attached devices.

        Returns one entry per memory *instance* (deduplicated by identity, so a
        memory shared by several compute devices -- or by a whole node -- appears
        once with all the compute devices it serves listed in
        ``attached_devices``). This is the memory-side view of the topology the
        reports key utilization on, independent of the compute devices, so it
        stays correct if the one-compute-to-one-memory assumption is broken.

        Each entry has: ``name``, ``capacity_bytes``, ``bandwidth_bytes_per_s``,
        ``role`` (``"input"``, ``"node"``, ``"first_tier"`` or ``"second_tier"``),
        ``node`` (owning node name, or ``""`` for the system-level input NVM) and
        ``attached_devices`` (compute devices that use it as a tier, in order).
        """

        order: list[int] = []
        by_id: dict[int, dict[str, Any]] = {}

        def record(memory: MemoryDevice, role: str, node: str) -> dict[str, Any]:
            key = id(memory)
            if key not in by_id:
                by_id[key] = {
                    "name": memory.name,
                    "capacity_bytes": memory.capacity_bytes,
                    "bandwidth_bytes_per_s": memory.bandwidth_bytes_per_s,
                    "role": role,
                    "node": node,
                    "attached_devices": [],
                }
                order.append(key)
            return by_id[key]

        record(self.input_memory, "input", "")
        for node in self.nodes:
            if node.node_memory is not None:
                record(node.node_memory, "node", node.name)
            for device in node.compute_devices:
                first = record(device.first_tier_memory, "first_tier", node.name)
                first["attached_devices"].append(device.name)
                second = device.second_tier_memory
                if second is not None:
                    entry = record(second, "second_tier", node.name)
                    entry["attached_devices"].append(device.name)

        return [
            {**by_id[key], "attached_devices": tuple(by_id[key]["attached_devices"])}
            for key in order
        ]

    def device_inventory(self) -> list[dict[str, Any]]:
        """Every compute device with its static specs, in node order.

        The compute-side companion to :meth:`memory_inventory`: one entry per
        compute device with its FLOP ceiling and the static capacity/bandwidth of
        its first-tier memory, so reports can render absolute values and the
        ``max`` reference lines without re-deriving topology. Each entry has:
        ``name``, ``node``, ``peak_flops_fp16``, ``first_tier_memory``,
        ``first_tier_capacity_bytes`` and ``first_tier_bandwidth_bytes_per_s``.
        """

        entries: list[dict[str, Any]] = []
        for node in self.nodes:
            for device in node.compute_devices:
                first = device.first_tier_memory
                entries.append({
                    "name": device.name,
                    "node": node.name,
                    "peak_flops_fp16": device.peak_flops_fp16,
                    "first_tier_memory": first.name,
                    "first_tier_capacity_bytes": first.capacity_bytes,
                    "first_tier_bandwidth_bytes_per_s": first.bandwidth_bytes_per_s,
                })
        return entries

    def _node_index_of_memory(self, memory: MemoryDevice) -> int | None:
        """Index of the node that hosts ``memory`` (by identity), else ``None``.

        Node-local memories are a node's CPU memory and each compute device's
        first- and second-tier memory. The shared input memory (system NVM) is
        not node-local and yields ``None``.
        """

        return self._node_index_by_memory.get(id(memory))

    @property
    def _node_index_by_memory(self) -> dict[int, int]:
        """``id(node-local memory) -> node index``, built once (first owner wins)."""

        cache = self.__dict__.get("_node_index_by_memory_cache")
        if cache is None:
            cache = {}
            for index, node in enumerate(self.nodes):
                if node.node_memory is not None:
                    cache.setdefault(id(node.node_memory), index)
                for device in node.compute_devices:
                    cache.setdefault(id(device.first_tier_memory), index)
                    second = device.second_tier_memory
                    if second is not None:
                        cache.setdefault(id(second), index)
            object.__setattr__(self, "_node_index_by_memory_cache", cache)
        return cache

    def link_between(self, src: MemoryDevice, dst: MemoryDevice) -> TransferLink:
        """Classify the link a transfer between ``src`` and ``dst`` traverses.

        Same device -> intra-package (own bandwidth, no latency); same node ->
        CXL; anything else (different nodes, or either end is the system NVM) ->
        the scale-up network.
        """

        if src is dst:
            return INTRA_PACKAGE
        node_a = self._node_index_of_memory(src)
        node_b = self._node_index_of_memory(dst)
        if node_a is not None and node_a == node_b:
            return TransferLink(
                "cxl",
                self.network.cxl_bandwidth_bytes_per_s,
                self.network.cxl_latency_s,
            )
        return TransferLink(
            "scale_up",
            self.network.scale_up_bandwidth_bytes_per_s,
            self.network.scale_up_latency_s,
        )



def _name_instance(
    device: ComputeDevice, node_name: str, index: int
) -> ComputeDevice:
    """Give a freshly-loaded device a unique, node-qualified name.

    Several instances of one device config share the same base name (the type),
    which makes them indistinguishable in per-device reports. Append the node and
    a per-node index (e.g. ``"NVIDIA B200 [node-0 #2]"``), and qualify the
    instance's first-tier memory name to match. Identity is unchanged.
    """

    suffix = f" [{node_name} #{index}]"
    first_tier = replace(
        device.first_tier_memory,
        name=device.first_tier_memory.name + suffix,
    )
    return replace(
        device,
        name=device.name + suffix,
        first_tier_memory=first_tier,
    )


def _build_node(
    raw_node: Mapping[str, Any],
    node_name: str,
    compute_dir: Path,
    memory_dir: Path,
) -> Node:
    """Instantiate one :class:`Node` from its config under a given name.

    ``node_name`` is the (already uniquified) name to give this node; it is used
    to qualify the node memory and every compute-device instance so all instances
    stay distinguishable in per-node and per-device reports.
    """

    node_memory = None
    node_memory_stem = raw_node.get("node_memory")
    if node_memory_stem is not None:
        node_memory = load_memory_device(memory_dir / f"{node_memory_stem}.json")
        # Each node owns a distinct node-memory instance; qualify its name
        # with the node so same-type node memories stay distinguishable in
        # per-memory reports (events key utilization by memory name).
        node_memory = replace(node_memory, name=f"{node_memory.name} [{node_name}]")

    devices: list[ComputeDevice] = []
    for entry in raw_node["compute_devices"]:
        count = entry.get("count", 1)
        if count < 1:
            raise ValueError("compute device 'count' must be >= 1")
        device_path = compute_dir / f"{entry['device']}.json"
        # A fresh load per instance gives each device its own first-tier
        # memory instance (identity matters for resource contention). Each
        # instance is also given a unique, node-qualified name so it is
        # distinguishable in logs and per-device reports.
        for _ in range(count):
            device = load_compute_device(device_path, memory_dir=memory_dir)
            # Tag the instance with its config key (the file stem) so the
            # orchestrator can group devices by *type* independent of the
            # node-qualified instance name.
            device = replace(device, device_key=entry["device"])
            index = len(devices)
            device = _name_instance(device, node_name, index)
            devices.append(device)

    if not devices:
        raise ValueError(f"node {node_name!r} has no compute devices")

    return Node(
        name=node_name,
        compute_devices=tuple(devices),
        node_memory=node_memory,
    )


def system_from_config(
    config: Mapping[str, Any],
    compute_dir: str | Path,
    memory_dir: str | Path,
) -> System:
    """Build a :class:`System` from a parsed config and device directories.

    Args:
        config: Parsed system JSON.
        compute_dir: Directory holding ``<device>.json`` compute configs.
        memory_dir: Directory holding ``<memory>.json`` memory configs.
    """

    compute_dir = Path(compute_dir)
    memory_dir = Path(memory_dir)

    network = Network.from_config(config["network"])
    input_memory = load_memory_device(memory_dir / f"{config['input_memory']}.json")

    nodes: list[Node] = []
    for raw_node in config["nodes"]:
        node_count = raw_node.get("count", 1)
        if node_count < 1:
            raise ValueError("node 'count' must be >= 1")
        # A node with ``count`` > 1 is expanded into that many *distinct* nodes.
        # Each copy is given a unique name (the base name qualified with a
        # per-copy index, e.g. ``"pod #2"``) so the node memory and every
        # compute-device instance -- which qualify their names on the owning
        # node's name -- stay distinct across copies. A single node (``count``
        # == 1) keeps its plain name for backward compatibility.
        base_name = raw_node["name"]
        for copy_index in range(node_count):
            node_name = base_name if node_count == 1 else f"{base_name} #{copy_index}"
            nodes.append(
                _build_node(raw_node, node_name, compute_dir, memory_dir)
            )

    return System(
        name=config["name"],
        network=network,
        input_memory=input_memory,
        nodes=tuple(nodes),
        description=config.get("description", ""),
    )


def load_system(
    path: str | Path,
    compute_dir: str | Path | None = None,
    memory_dir: str | Path | None = None,
) -> System:
    """Load a :class:`System` from a system JSON file.

    Args:
        path: Path to the system JSON.
        compute_dir: Directory of compute configs. Defaults to a sibling
            ``Compute_devices`` directory next to the system file's parent.
        memory_dir: Directory of memory configs. Defaults to a sibling
            ``Memory_devices`` directory next to the system file's parent.
    """

    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    if compute_dir is None:
        compute_dir = path.parent.parent / "Compute_devices"
    if memory_dir is None:
        memory_dir = path.parent.parent / "Memory_devices"

    return system_from_config(config, compute_dir=compute_dir, memory_dir=memory_dir)
