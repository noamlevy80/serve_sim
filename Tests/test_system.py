"""Tests for parsing system configs (``Systems/*.json``) into hardware objects.

A system config names a network, an input memory and a list of nodes; each node
expands ``{"device": ..., "count": N}`` entries into N distinct compute-device
instances. The repo ships two sample systems we assert against, plus synthetic
configs for the edge cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from serve_sim.system import (
    Network,
    Node,
    System,
    load_system,
    system_from_config,
)
from serve_sim.hardware import ComputeDevice, MemoryDevice

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEMS_DIR = REPO_ROOT / "Systems"
COMPUTE_DIR = REPO_ROOT / "Compute_devices"
MEMORY_DIR = REPO_ROOT / "Memory_devices"

SYSTEM_FILES = sorted(SYSTEMS_DIR.glob("*.json"))


def load(name: str) -> System:
    return load_system(SYSTEMS_DIR / name)


# --- every shipped system loads ------------------------------------------------


@pytest.mark.parametrize("path", SYSTEM_FILES, ids=lambda p: p.stem)
def test_every_system_config_loads(path: Path) -> None:
    system = load_system(path)
    assert isinstance(system, System)
    assert system.name
    assert system.num_compute_devices > 0
    assert isinstance(system.input_memory, MemoryDevice)
    for device in system.compute_devices:
        assert isinstance(device, ComputeDevice)


# --- dual-node B200 ------------------------------------------------------------


def test_dual_node_b200_topology() -> None:
    system = load("dual-node-b200.json")
    assert len(system.nodes) == 2
    assert [len(n) for n in system.nodes] == [4, 4]
    assert system.num_compute_devices == 8
    # Each instance keeps the device type as a name prefix but is uniquely named.
    assert all(d.name.startswith("NVIDIA B200") for d in system.compute_devices)
    names = [d.name for d in system.compute_devices]
    assert len(set(names)) == len(names)


def test_dual_node_b200_input_memory_is_the_nvm() -> None:
    system = load("dual-node-b200.json")
    assert system.input_memory.name == "Datacenter NVM Pool"


def test_dual_node_b200_has_grace_node_memory() -> None:
    system = load("dual-node-b200.json")
    for node in system.nodes:
        assert isinstance(node.node_memory, MemoryDevice)
        assert "Grace" in node.node_memory.name


def test_network_parameters_are_parsed() -> None:
    system = load("dual-node-b200.json")
    net = system.network
    assert isinstance(net, Network)
    assert net.scale_up_bandwidth_bytes_per_s > 0
    assert net.cxl_bandwidth_bytes_per_s > 0
    assert net.scale_up_latency_s >= 0
    assert net.cxl_latency_s >= 0


# --- heterogeneous B200 + Cerebras ---------------------------------------------


def test_heterogeneous_system_has_two_device_types() -> None:
    system = load("heterogeneous-b200-cerebras.json")
    node0, node1 = system.nodes
    assert all(d.name.startswith("NVIDIA B200") for d in node0.compute_devices)
    assert all(d.name.startswith("Cerebras WSE-3") for d in node1.compute_devices)
    assert len(node0) == 4 and len(node1) == 4


def test_heterogeneous_devices_keep_their_own_first_tier() -> None:
    system = load("heterogeneous-b200-cerebras.json")
    node0, node1 = system.nodes
    assert node0.compute_devices[0].first_tier_memory.name.startswith(
        "NVIDIA B200 HBM3e"
    )
    assert "SRAM" in node1.compute_devices[0].first_tier_memory.name


# --- instance identity (matters for the resource arbiter) ----------------------


def test_each_compute_device_is_a_distinct_instance() -> None:
    system = load("dual-node-b200.json")
    devices = system.compute_devices
    ids = {id(d) for d in devices}
    assert len(ids) == len(devices)


def test_each_compute_device_has_a_unique_node_qualified_name() -> None:
    system = load("dual-node-b200.json")
    names = [d.name for d in system.compute_devices]
    # Distinct names so per-device reports don't collapse the 8 GPUs into one.
    assert len(set(names)) == len(names)
    # Names are node-qualified and keep the device type as a prefix.
    assert "NVIDIA B200 [node-0 #0]" in names
    assert "NVIDIA B200 [node-1 #3]" in names
    # The instance's first-tier memory name is qualified to match.
    for device in system.compute_devices:
        assert device.first_tier_memory.name.endswith(device.name[len("NVIDIA B200"):])


def test_each_device_has_its_own_first_tier_memory_instance() -> None:
    system = load("dual-node-b200.json")
    memories = [d.first_tier_memory for d in system.compute_devices]
    assert len({id(m) for m in memories}) == len(memories)


def test_node_of_finds_owning_node() -> None:
    system = load("dual-node-b200.json")
    first = system.nodes[0].compute_devices[0]
    last = system.nodes[1].compute_devices[-1]
    assert system.node_of(first) is system.nodes[0]
    assert system.node_of(last) is system.nodes[1]


def test_node_of_rejects_foreign_device() -> None:
    system = load("dual-node-b200.json")
    stranger = ComputeDevice(
        "stranger",
        peak_flops_fp16=1e14,
        first_tier_memory=MemoryDevice("m", capacity_bytes=1e9, bandwidth_bytes_per_s=1e12),
    )
    with pytest.raises(ValueError):
        system.node_of(stranger)


# --- second tier is NOT auto-attached (it's an orchestration choice) -----------


def test_loaded_devices_have_no_second_tier_by_default() -> None:
    system = load("dual-node-b200.json")
    assert all(d.second_tier_memory is None for d in system.compute_devices)


# --- synthetic configs / validation --------------------------------------------


def _base_config(**overrides):
    config = {
        "name": "Synthetic",
        "network": {
            "scale_up_bandwidth_bytes_per_s": 1.0e12,
            "scale_up_latency_s": 1.0e-6,
            "cxl_bandwidth_bytes_per_s": 6.0e10,
            "cxl_latency_s": 3.0e-7,
        },
        "input_memory": "datacenter-nvm",
        "nodes": [
            {
                "name": "n0",
                "node_memory": None,
                "compute_devices": [{"device": "nvidia-b200", "count": 2}],
            }
        ],
    }
    config.update(overrides)
    return config


def build(config):
    return system_from_config(config, compute_dir=COMPUTE_DIR, memory_dir=MEMORY_DIR)


def test_count_expands_to_distinct_instances() -> None:
    system = build(_base_config())
    assert system.num_compute_devices == 2
    a, b = system.compute_devices
    assert a is not b
    assert a.first_tier_memory is not b.first_tier_memory


def test_count_defaults_to_one() -> None:
    config = _base_config()
    config["nodes"][0]["compute_devices"] = [{"device": "nvidia-b200"}]
    system = build(config)
    assert system.num_compute_devices == 1


def test_omitted_node_memory_is_none() -> None:
    system = build(_base_config())
    assert system.nodes[0].node_memory is None


def test_zero_count_is_rejected() -> None:
    config = _base_config()
    config["nodes"][0]["compute_devices"] = [{"device": "nvidia-b200", "count": 0}]
    with pytest.raises(ValueError):
        build(config)


# --- node-level count (replicating whole nodes) --------------------------------


def test_node_count_expands_to_distinct_nodes() -> None:
    config = _base_config()
    config["nodes"][0]["count"] = 3
    system = build(config)
    assert len(system.nodes) == 3
    # Replicated nodes get unique, index-qualified names.
    assert [n.name for n in system.nodes] == ["n0 #0", "n0 #1", "n0 #2"]
    # Every compute device across every copy is a distinct instance...
    devices = system.compute_devices
    assert system.num_compute_devices == 6
    assert len({id(d) for d in devices}) == len(devices)
    # ...with a unique name and a distinct first-tier memory instance.
    assert len({d.name for d in devices}) == len(devices)
    assert len({id(d.first_tier_memory) for d in devices}) == len(devices)


def test_node_count_qualifies_node_memory_names() -> None:
    config = _base_config()
    config["nodes"][0]["node_memory"] = "nvidia-grace-lpddr5x"
    config["nodes"][0]["count"] = 2
    system = build(config)
    mem_names = [n.node_memory.name for n in system.nodes]
    assert mem_names[0] != mem_names[1]
    assert mem_names[0].endswith("[n0 #0]")
    assert mem_names[1].endswith("[n0 #1]")


def test_node_count_defaults_to_one_keeps_plain_name() -> None:
    system = build(_base_config())
    assert len(system.nodes) == 1
    assert system.nodes[0].name == "n0"


def test_zero_node_count_is_rejected() -> None:
    config = _base_config()
    config["nodes"][0]["count"] = 0
    with pytest.raises(ValueError):
        build(config)


def test_system_requires_at_least_one_node() -> None:
    config = _base_config(nodes=[])
    with pytest.raises(ValueError):
        build(config)


def test_network_rejects_non_positive_bandwidth() -> None:
    with pytest.raises(ValueError):
        Network(
            scale_up_bandwidth_bytes_per_s=0.0,
            scale_up_latency_s=1e-6,
            cxl_bandwidth_bytes_per_s=6e10,
            cxl_latency_s=3e-7,
        )


def test_network_rejects_negative_latency() -> None:
    with pytest.raises(ValueError):
        Network(
            scale_up_bandwidth_bytes_per_s=1e12,
            scale_up_latency_s=-1.0,
            cxl_bandwidth_bytes_per_s=6e10,
            cxl_latency_s=3e-7,
        )
