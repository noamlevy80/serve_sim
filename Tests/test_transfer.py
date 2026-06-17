"""Tests for the network-aware transfer cost model (``serve_sim.transfer``).

A transfer's bandwidth is the minimum across the two memories and the link; its
latency is the link's. The system classifies a pair of memories into the link a
move would traverse: intra-package (same device), CXL (same node) or scale-up
(different nodes / the system NVM).
"""

from __future__ import annotations

import pytest

from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.system import Network, Node, System
from serve_sim.transfer import (
    INTRA_PACKAGE,
    TransferLink,
    make_transfer_event,
    transfer_duration,
)


def mem(name: str, bw: float) -> MemoryDevice:
    return MemoryDevice(name=name, capacity_bytes=1e12, bandwidth_bytes_per_s=bw)


# --- transfer_duration ---------------------------------------------------------


def test_intra_package_bounded_by_slower_memory() -> None:
    src = mem("fast", 1e12)
    dst = mem("slow", 4e11)
    # No link bandwidth or latency: bound by the slower of the two ends.
    assert transfer_duration(8e11, src, dst, INTRA_PACKAGE) == pytest.approx(2.0)


def test_link_bandwidth_can_bound_the_transfer() -> None:
    src = mem("hbm-a", 1e12)
    dst = mem("hbm-b", 1e12)
    link = TransferLink("scale_up", bandwidth_bytes_per_s=2e11, latency_s=0.0)
    # 1e12 bytes over the 2e11 link = 5 s.
    assert transfer_duration(1e12, src, dst, link) == pytest.approx(5.0)


def test_latency_is_added_on_top() -> None:
    src = mem("hbm-a", 1e12)
    dst = mem("hbm-b", 1e12)
    link = TransferLink("cxl", bandwidth_bytes_per_s=1e12, latency_s=3e-7)
    # 1e12 / 1e12 = 1 s of bytes plus 3e-7 link latency.
    assert transfer_duration(1e12, src, dst, link) == pytest.approx(1.0 + 3e-7)


def test_zero_bytes_is_just_latency() -> None:
    src = mem("a", 1e12)
    dst = mem("b", 1e12)
    link = TransferLink("scale_up", bandwidth_bytes_per_s=2e11, latency_s=1e-6)
    assert transfer_duration(0.0, src, dst, link) == pytest.approx(1e-6)


def test_negative_bytes_rejected() -> None:
    src = mem("a", 1e12)
    dst = mem("b", 1e12)
    with pytest.raises(ValueError):
        transfer_duration(-1.0, src, dst, INTRA_PACKAGE)


def test_non_positive_link_bandwidth_rejected() -> None:
    with pytest.raises(ValueError):
        TransferLink("scale_up", bandwidth_bytes_per_s=0.0)


def test_negative_link_latency_rejected() -> None:
    with pytest.raises(ValueError):
        TransferLink("cxl", bandwidth_bytes_per_s=1e12, latency_s=-1.0)


# --- make_transfer_event -------------------------------------------------------


def test_transfer_event_fields() -> None:
    src = mem("a", 1e12)
    dst = mem("b", 1e12)
    link = TransferLink("scale_up", bandwidth_bytes_per_s=5e11, latency_s=1e-6)
    event = make_transfer_event(
        1e12, src, dst, link, start=10.0, group_index=3, device_index=2
    )
    assert event.phase == "transfer"
    assert event.group_index == 3
    assert event.device_index == 2
    assert event.flops == 0.0
    assert event.compute_time == 0.0
    assert event.bytes_read == pytest.approx(1e12)
    assert event.bandwidth_time == pytest.approx(2.0)  # 1e12 / 5e11
    assert event.duration == pytest.approx(2.0 + 1e-6)
    assert event.start == pytest.approx(10.0)
    assert event.end == pytest.approx(10.0 + 2.0 + 1e-6)


def test_transfer_event_negative_bytes_rejected() -> None:
    src = mem("a", 1e12)
    dst = mem("b", 1e12)
    with pytest.raises(ValueError):
        make_transfer_event(-1.0, src, dst, INTRA_PACKAGE, start=0.0)


# --- System.link_between -------------------------------------------------------


def build_system() -> System:
    """Two nodes: node 0 has two compute devices, node 1 has one."""

    network = Network(
        scale_up_bandwidth_bytes_per_s=1.8e12,
        scale_up_latency_s=1e-6,
        cxl_bandwidth_bytes_per_s=6.4e10,
        cxl_latency_s=3e-7,
    )
    input_memory = mem("nvm", 5e9)

    def compute(name: str) -> ComputeDevice:
        return ComputeDevice(
            name=name, peak_flops_fp16=1e15, first_tier_memory=mem(f"{name}-hbm", 8e12)
        )

    node0 = Node(
        name="node-0",
        compute_devices=(compute("g0"), compute("g1")),
        node_memory=mem("node0-mem", 5e11),
    )
    node1 = Node(
        name="node-1",
        compute_devices=(compute("g2"),),
        node_memory=mem("node1-mem", 5e11),
    )
    return System(
        name="test", network=network, input_memory=input_memory,
        nodes=(node0, node1),
    )


def test_link_same_device_is_intra_package() -> None:
    system = build_system()
    hbm = system.nodes[0].compute_devices[0].first_tier_memory
    assert system.link_between(hbm, hbm) is INTRA_PACKAGE


def test_link_within_node_is_cxl() -> None:
    system = build_system()
    hbm0 = system.nodes[0].compute_devices[0].first_tier_memory
    hbm1 = system.nodes[0].compute_devices[1].first_tier_memory
    link = system.link_between(hbm0, hbm1)
    assert link.kind == "cxl"
    assert link.bandwidth_bytes_per_s == pytest.approx(6.4e10)
    assert link.latency_s == pytest.approx(3e-7)


def test_link_node_memory_within_node_is_cxl() -> None:
    system = build_system()
    hbm0 = system.nodes[0].compute_devices[0].first_tier_memory
    node_mem = system.nodes[0].node_memory
    assert system.link_between(hbm0, node_mem).kind == "cxl"


def test_link_across_nodes_is_scale_up() -> None:
    system = build_system()
    hbm0 = system.nodes[0].compute_devices[0].first_tier_memory
    hbm2 = system.nodes[1].compute_devices[0].first_tier_memory
    link = system.link_between(hbm0, hbm2)
    assert link.kind == "scale_up"
    assert link.bandwidth_bytes_per_s == pytest.approx(1.8e12)
    assert link.latency_s == pytest.approx(1e-6)


def test_link_to_system_nvm_is_scale_up() -> None:
    system = build_system()
    hbm0 = system.nodes[0].compute_devices[0].first_tier_memory
    # The shared input memory is not node-local: reached over the scale-up net.
    assert system.link_between(hbm0, system.input_memory).kind == "scale_up"
