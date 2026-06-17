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
per-device reports. The single input memory and each node's node memory are one
shared instance apiece.
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

        for node in self.nodes:
            if any(d is device for d in node.compute_devices):
                return node
        raise ValueError(f"device {device.name!r} is not part of this system")

    def _node_index_of_memory(self, memory: MemoryDevice) -> int | None:
        """Index of the node that hosts ``memory`` (by identity), else ``None``.

        Node-local memories are a node's CPU memory and each compute device's
        first- and second-tier memory. The shared input memory (system NVM) is
        not node-local and yields ``None``.
        """

        target = id(memory)
        for index, node in enumerate(self.nodes):
            if node.node_memory is not None and id(node.node_memory) == target:
                return index
            for device in node.compute_devices:
                if id(device.first_tier_memory) == target:
                    return index
                second = device.second_tier_memory
                if second is not None and id(second) == target:
                    return index
        return None

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
        node_memory = None
        node_memory_stem = raw_node.get("node_memory")
        if node_memory_stem is not None:
            node_memory = load_memory_device(memory_dir / f"{node_memory_stem}.json")

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
                index = len(devices)
                device = _name_instance(device, raw_node["name"], index)
                devices.append(device)

        if not devices:
            raise ValueError(f"node {raw_node['name']!r} has no compute devices")

        nodes.append(
            Node(
                name=raw_node["name"],
                compute_devices=tuple(devices),
                node_memory=node_memory,
            )
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
