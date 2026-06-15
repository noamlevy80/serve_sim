"""Compute and memory device models.

These are deliberately thin: a memory device exposes a capacity and an intrinsic
bandwidth; a compute device exposes a nominal FP16 FLOP rate and links to a
first-tier memory (and optional second tier). The event generator uses these to
turn a work shard's FLOPs/bytes into a duration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryDevice:
    """Volatile or non-volatile memory.

    Attributes:
        name: Identifier for logs/reports.
        capacity_bytes: Total capacity.
        bandwidth_bytes_per_s: Intrinsic (unconstrained) bandwidth ceiling.
    """

    name: str
    capacity_bytes: float
    bandwidth_bytes_per_s: float

    def __post_init__(self) -> None:
        if self.capacity_bytes < 0:
            raise ValueError("capacity_bytes must be non-negative")
        if self.bandwidth_bytes_per_s <= 0:
            raise ValueError("bandwidth_bytes_per_s must be positive")


# dtype byte size -> compute speed multiplier relative to FP16 (2 bytes = 1x).
# 8-bit is 2x faster, 4-bit is 4x faster, FP32 is 2x slower.
def dtype_compute_scale(dtype_bytes: float) -> float:
    """Compute-rate multiplier for a given element size, relative to FP16."""

    if dtype_bytes <= 0:
        raise ValueError("dtype_bytes must be positive")
    return 2.0 / dtype_bytes


@dataclass(frozen=True)
class ComputeDevice:
    """A GPU-like inference device tied to a first-tier memory.

    Attributes:
        name: Identifier for logs/reports.
        peak_flops_fp16: Nominal FP16 FLOP/s.
        first_tier_memory: Memory whose bandwidth bounds compute events.
        second_tier_memory: Optional slower/larger memory (requires a transfer
            into the first tier before compute can use it).
        kernel_launch_latency: Fixed wait incurred each time a new kernel is
            launched on this device (seconds). Zero for devices with no launch
            overhead (e.g. statically scheduled dataflow chips).
    """

    name: str
    peak_flops_fp16: float
    first_tier_memory: MemoryDevice
    second_tier_memory: MemoryDevice | None = None
    kernel_launch_latency: float = 0.0

    def __post_init__(self) -> None:
        if self.peak_flops_fp16 <= 0:
            raise ValueError("peak_flops_fp16 must be positive")
        if self.kernel_launch_latency < 0:
            raise ValueError("kernel_launch_latency must be non-negative")

    def effective_flops(self, dtype_bytes: float) -> float:
        """Peak FLOP/s adjusted for the operand data type."""

        return self.peak_flops_fp16 * dtype_compute_scale(dtype_bytes)

    @property
    def bandwidth_bytes_per_s(self) -> float:
        """First-tier memory bandwidth that bounds compute events."""

        return self.first_tier_memory.bandwidth_bytes_per_s
