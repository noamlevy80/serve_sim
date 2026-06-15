"""Tests for device models and dtype scaling."""

from __future__ import annotations

import pytest

from serve_sim.hardware import ComputeDevice, MemoryDevice, dtype_compute_scale


def test_dtype_compute_scale():
    assert dtype_compute_scale(4) == 0.5  # fp32: 2x slower
    assert dtype_compute_scale(2) == 1.0  # fp16: nominal
    assert dtype_compute_scale(1) == 2.0  # fp8: 2x faster
    assert dtype_compute_scale(0.5) == 4.0  # fp4: 4x faster


def test_dtype_compute_scale_rejects_nonpositive():
    with pytest.raises(ValueError):
        dtype_compute_scale(0)


def test_memory_device_validation():
    with pytest.raises(ValueError, match="bandwidth"):
        MemoryDevice("m", capacity_bytes=1, bandwidth_bytes_per_s=0)
    with pytest.raises(ValueError, match="capacity"):
        MemoryDevice("m", capacity_bytes=-1, bandwidth_bytes_per_s=1)


def test_compute_device_effective_flops():
    mem = MemoryDevice("hbm", capacity_bytes=1e9, bandwidth_bytes_per_s=1e12)
    dev = ComputeDevice("gpu", peak_flops_fp16=100e12, first_tier_memory=mem)
    assert dev.effective_flops(2) == 100e12
    assert dev.effective_flops(1) == 200e12
    assert dev.effective_flops(4) == 50e12
    assert dev.bandwidth_bytes_per_s == 1e12


def test_compute_device_validation():
    mem = MemoryDevice("hbm", capacity_bytes=1e9, bandwidth_bytes_per_s=1e12)
    with pytest.raises(ValueError, match="peak_flops"):
        ComputeDevice("gpu", peak_flops_fp16=0, first_tier_memory=mem)
