"""Tests for loading compute/memory devices from the JSON device configs.

The configs live in ``Compute_devices/`` and ``Memory_devices/`` at the repo
root. A compute config names its first-tier memory by file stem; a second tier
is a system-configuration choice supplied at load time, never in the config.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from serve_sim.device_config import (
    compute_device_from_config,
    load_compute_device,
    load_memory_device,
    memory_device_from_config,
)
from serve_sim.hardware import ComputeDevice, MemoryDevice

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPUTE_DIR = REPO_ROOT / "Compute_devices"
MEMORY_DIR = REPO_ROOT / "Memory_devices"

COMPUTE_FILES = sorted(COMPUTE_DIR.glob("*.json"))
MEMORY_FILES = sorted(MEMORY_DIR.glob("*.json"))


# --- memory devices ---------------------------------------------------------

@pytest.mark.parametrize("path", MEMORY_FILES, ids=lambda p: p.stem)
def test_load_memory_device(path: Path) -> None:
    device = load_memory_device(path)
    assert isinstance(device, MemoryDevice)
    assert device.name
    assert device.capacity_bytes > 0
    assert device.bandwidth_bytes_per_s > 0


def test_memory_device_from_config_matches_values() -> None:
    config = {
        "name": "Toy SRAM",
        "capacity_bytes": 1.0e9,
        "bandwidth_bytes_per_s": 2.0e12,
    }
    device = memory_device_from_config(config)
    assert device.name == "Toy SRAM"
    assert device.capacity_bytes == 1.0e9
    assert device.bandwidth_bytes_per_s == 2.0e12


def test_datacenter_nvm_is_available_as_a_memory_device() -> None:
    nvm = load_memory_device(MEMORY_DIR / "datacenter-nvm.json")
    assert nvm.capacity_bytes > 0
    assert nvm.bandwidth_bytes_per_s > 0


# --- compute devices --------------------------------------------------------

@pytest.mark.parametrize("path", COMPUTE_FILES, ids=lambda p: p.stem)
def test_load_compute_device(path: Path) -> None:
    device = load_compute_device(path)
    assert isinstance(device, ComputeDevice)
    assert device.name
    assert device.peak_flops_fp16 > 0
    # first tier resolved from the sibling Memory_devices directory.
    assert isinstance(device.first_tier_memory, MemoryDevice)
    assert device.first_tier_memory.bandwidth_bytes_per_s > 0
    # no second tier unless a system configuration attaches one.
    assert device.second_tier_memory is None


def test_compute_device_resolves_named_first_tier() -> None:
    b200 = load_compute_device(COMPUTE_DIR / "nvidia-b200.json")
    assert b200.name == "NVIDIA B200"
    assert b200.first_tier_memory.name == "NVIDIA B200 HBM3e"
    # the compute device's bounding bandwidth comes from its first tier.
    assert b200.bandwidth_bytes_per_s == b200.first_tier_memory.bandwidth_bytes_per_s


def test_effective_flops_scales_with_dtype() -> None:
    wse3 = load_compute_device(COMPUTE_DIR / "cerebras-wse3.json")
    # fp16 (2 bytes) is the 1x baseline; fp8 (1 byte) is 2x.
    assert wse3.effective_flops(2) == pytest.approx(wse3.peak_flops_fp16)
    assert wse3.effective_flops(1) == pytest.approx(2 * wse3.peak_flops_fp16)


def test_second_tier_is_attached_at_load_time_not_in_config() -> None:
    nvm = load_memory_device(MEMORY_DIR / "datacenter-nvm.json")
    groq = load_compute_device(
        COMPUTE_DIR / "groq-lpu.json", second_tier_memory=nvm
    )
    assert groq.second_tier_memory is nvm
    # the config itself does not mention a second tier.
    config = json.loads((COMPUTE_DIR / "groq-lpu.json").read_text(encoding="utf-8"))
    assert "second_tier_memory" not in config


def test_compute_device_from_config_uses_supplied_memories() -> None:
    tier1 = MemoryDevice("T1", capacity_bytes=1e9, bandwidth_bytes_per_s=1e12)
    tier2 = MemoryDevice("T2", capacity_bytes=1e12, bandwidth_bytes_per_s=1e11)
    device = compute_device_from_config(
        {"name": "Toy", "peak_flops_fp16": 5e14},
        first_tier_memory=tier1,
        second_tier_memory=tier2,
    )
    assert device.first_tier_memory is tier1
    assert device.second_tier_memory is tier2


def test_custom_memory_dir_is_respected() -> None:
    b200 = load_compute_device(
        COMPUTE_DIR / "nvidia-b200.json", memory_dir=MEMORY_DIR
    )
    assert b200.first_tier_memory.name == "NVIDIA B200 HBM3e"


def test_all_named_first_tiers_exist() -> None:
    for path in COMPUTE_FILES:
        config = json.loads(path.read_text(encoding="utf-8"))
        assert (MEMORY_DIR / f"{config['first_tier_memory']}.json").exists()
