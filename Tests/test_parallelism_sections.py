"""Per-device-type parallelism sections.

A :class:`StrategyConfig` may carry a list of :class:`ParallelismSection`s, one
per compute-device *type* (matched against each device's ``device_key``). The
orchestrator then partitions the cluster into one *engine group* per section,
each spanning only its device type and sliced into slots of its own degree. With
no sections the flat fields drive a single group over every device (the legacy
homogeneous layout). These tests pin that partitioning and its validation.
"""

from __future__ import annotations

import pytest

from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.model import toy_model
from serve_sim.system import Network, Node, System
from serve_sim.orchestrator import (
    ParallelismSection,
    Request,
    Simulator,
    StrategyConfig,
)


# --- helpers --------------------------------------------------------------------


def make_memory(name="mem", bw=1e12, cap=80e9):
    return MemoryDevice(name, capacity_bytes=cap, bandwidth_bytes_per_s=bw)


def make_device(name, device_key, cap=80e9, bw=1e12, peak=100e12):
    return ComputeDevice(
        name,
        peak_flops_fp16=peak,
        first_tier_memory=make_memory(f"{name}-mem", bw=bw, cap=cap),
        device_key=device_key,
    )


def make_hetero_system(counts):
    """Build a system with one node per device type.

    ``counts`` maps a ``device_key`` to how many devices of that type to create.
    """

    network = Network(
        scale_up_bandwidth_bytes_per_s=1e12,
        scale_up_latency_s=1e-6,
        cxl_bandwidth_bytes_per_s=1e11,
        cxl_latency_s=1e-7,
    )
    nodes = []
    for i, (key, n) in enumerate(counts.items()):
        devices = tuple(make_device(f"{key}-{j}", key) for j in range(n))
        nodes.append(
            Node(
                name=f"node-{i}",
                compute_devices=devices,
                node_memory=make_memory(f"node-{i}-mem", bw=5e11),
            )
        )
    return System(
        name="hetero",
        network=network,
        input_memory=make_memory("nvm", bw=5e9, cap=1e12),
        nodes=tuple(nodes),
    )


# --- section validation --------------------------------------------------------


def test_section_requires_compute_device():
    with pytest.raises(ValueError):
        ParallelismSection(compute_device="")


def test_section_rejects_zero_degree():
    with pytest.raises(ValueError):
        ParallelismSection(compute_device="gpu", pipeline_parallel=0)


def test_section_degree_is_product():
    section = ParallelismSection(
        compute_device="gpu", pipeline_parallel=2, expert_parallel=3, tensor_parallel=4
    )
    assert section.degree == 24


def test_strategy_normalizes_parallelism_to_tuple():
    strat = StrategyConfig(
        parallelism=[ParallelismSection(compute_device="gpu")]
    )
    assert isinstance(strat.parallelism, tuple)


def test_strategy_rejects_duplicate_device_sections():
    with pytest.raises(ValueError):
        StrategyConfig(
            parallelism=[
                ParallelismSection(compute_device="gpu"),
                ParallelismSection(compute_device="gpu"),
            ]
        )


# --- group partitioning --------------------------------------------------------


def test_flat_config_builds_single_group_over_all_devices():
    system = make_hetero_system({"type-a": 4, "type-b": 4})
    sim = Simulator(system, StrategyConfig(max_batch_size=1, pipeline_parallel=2))

    assert len(sim._groups) == 1
    group = sim._groups[0]
    assert group.device_key == ""
    assert group.pool.num_slots == 4  # 8 devices / degree 2
    assert group.degree == 2


def test_sections_partition_groups_by_device_type():
    system = make_hetero_system({"type-a": 4, "type-b": 6})
    strategy = StrategyConfig(
        max_batch_size=1,
        parallelism=[
            ParallelismSection(compute_device="type-a", pipeline_parallel=2),
            ParallelismSection(compute_device="type-b", tensor_parallel=3),
        ],
    )
    sim = Simulator(system, strategy)

    assert [g.device_key for g in sim._groups] == ["type-a", "type-b"]

    group_a, group_b = sim._groups
    # Each group spans only its own device type.
    assert all(d.device_key == "type-a" for s in group_a.pool.slots for d in s.devices)
    assert all(d.device_key == "type-b" for s in group_b.pool.slots for d in s.devices)
    # ...sliced into slots of its own configured degree.
    assert group_a.degree == 2 and group_a.pool.num_slots == 2  # 4 / 2
    assert group_b.degree == 3 and group_b.pool.num_slots == 2  # 6 / 3


def test_section_matching_no_device_is_rejected():
    system = make_hetero_system({"type-a": 2})
    strategy = StrategyConfig(
        parallelism=[ParallelismSection(compute_device="type-missing")]
    )
    with pytest.raises(ValueError):
        Simulator(system, strategy)


def test_group_too_small_for_degree_is_rejected():
    system = make_hetero_system({"type-a": 2})
    strategy = StrategyConfig(
        parallelism=[
            ParallelismSection(compute_device="type-a", pipeline_parallel=4)
        ]
    )
    with pytest.raises(ValueError):
        Simulator(system, strategy)


# --- end to end ---------------------------------------------------------------


def test_batches_run_on_their_device_type_group():
    # A two-type cluster serves a batch entirely within one device-type group;
    # the run completes and every served device belongs to a single type.
    system = make_hetero_system({"type-a": 2, "type-b": 2})
    model = toy_model()
    strategy = StrategyConfig(
        max_batch_size=1,
        parallelism=[
            ParallelismSection(compute_device="type-a"),
            ParallelismSection(compute_device="type-b"),
        ],
    )
    reqs = [Request(i, model, 64, 8, arrival_time=0.0) for i in range(4)]

    result = Simulator(system, strategy).run(reqs)

    assert len(result.records) == 4
    for decision in result.decisions:
        keys = {
            d.split("-")[0] + "-" + d.split("-")[1] for d in decision.devices
        }
        assert len(keys) == 1  # a batch never spans two device types
