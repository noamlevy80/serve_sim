"""Load JSON device configs into :class:`ComputeDevice` / :class:`MemoryDevice`.

Devices live in two sibling directories (see ``Memory_devices/`` and
``Compute_devices/``). A memory config is a flat record with a ``capacity_bytes``
and ``bandwidth_bytes_per_s``. A compute config carries a ``peak_flops_fp16`` and
names its first-tier memory by file stem (``"first_tier_memory": "groq-lpu-sram"``).

A second-tier memory is deliberately *not* part of a compute device's own config:
whether a device is backed by a shared pool (e.g. a datacenter NVM) is a property
of a system configuration, so it is supplied here as an explicit argument.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .hardware import ComputeDevice, MemoryDevice


def memory_device_from_config(config: Mapping[str, Any]) -> MemoryDevice:
    """Build a :class:`MemoryDevice` from a parsed memory config."""

    return MemoryDevice(
        name=config["name"],
        capacity_bytes=config["capacity_bytes"],
        bandwidth_bytes_per_s=config["bandwidth_bytes_per_s"],
    )


def compute_device_from_config(
    config: Mapping[str, Any],
    first_tier_memory: MemoryDevice,
    second_tier_memory: MemoryDevice | None = None,
) -> ComputeDevice:
    """Build a :class:`ComputeDevice` from a parsed compute config.

    The first-tier memory is resolved by the caller (the config only names it);
    the optional second tier is a system-configuration choice supplied here.
    """

    return ComputeDevice(
        name=config["name"],
        peak_flops_fp16=config["peak_flops_fp16"],
        first_tier_memory=first_tier_memory,
        second_tier_memory=second_tier_memory,
        kernel_launch_latency=config.get("kernel_launch_latency", 0.0),
    )


def load_memory_device(path: str | Path) -> MemoryDevice:
    """Load and build a :class:`MemoryDevice` from a JSON config file."""

    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return memory_device_from_config(config)


def load_compute_device(
    path: str | Path,
    memory_dir: str | Path | None = None,
    second_tier_memory: MemoryDevice | None = None,
) -> ComputeDevice:
    """Load a :class:`ComputeDevice`, resolving its first-tier memory by stem.

    Args:
        path: Path to the compute device JSON.
        memory_dir: Directory holding ``<first_tier_memory>.json``. Defaults to a
            sibling ``Memory_devices`` directory next to the compute file's parent.
        second_tier_memory: Optional shared/second-tier memory to attach.
    """

    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    if memory_dir is None:
        memory_dir = path.parent.parent / "Memory_devices"
    memory_path = Path(memory_dir) / f"{config['first_tier_memory']}.json"
    first_tier_memory = load_memory_device(memory_path)

    return compute_device_from_config(
        config,
        first_tier_memory=first_tier_memory,
        second_tier_memory=second_tier_memory,
    )
