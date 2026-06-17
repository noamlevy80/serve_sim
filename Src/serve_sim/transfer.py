"""Network-aware data-transfer cost model.

A data transfer moves bytes between two memory devices over a link. Per the PRD,
the achievable bandwidth is the minimum across both ends of the transfer and the
link connecting them, and the latency is the maximum across the two ends and the
link. Memory devices contribute a bandwidth ceiling but no latency of their own,
so the latency reduces to the link's latency.

The link is the scale-up network when the two ends sit on different nodes, the
in-node CXL fabric when they share a node, and a degenerate *intra-package* link
(the memory's own bandwidth, no link latency) when a device reads its own memory.
:meth:`serve_sim.system.System.link_between` classifies a pair of memories into
the right link for a given system.

    duration = link.latency + num_bytes / min(src.bw, dst.bw, link.bw)
"""

from __future__ import annotations

from dataclasses import dataclass

from .events import ComputeEvent
from .hardware import MemoryDevice


@dataclass(frozen=True)
class TransferLink:
    """A link between two memory devices.

    Attributes:
        kind: ``"intra_package"``, ``"cxl"`` or ``"scale_up"``.
        bandwidth_bytes_per_s: Link bandwidth ceiling, or ``None`` for an
            unconstrained intra-package access (bounded only by the memories).
        latency_s: Fixed per-transfer latency added before the byte transfer.
    """

    kind: str
    bandwidth_bytes_per_s: float | None = None
    latency_s: float = 0.0

    def __post_init__(self) -> None:
        if self.bandwidth_bytes_per_s is not None and self.bandwidth_bytes_per_s <= 0:
            raise ValueError("bandwidth_bytes_per_s must be positive")
        if self.latency_s < 0:
            raise ValueError("latency_s must be non-negative")


# An access with no connecting fabric: own bandwidth, no link latency.
INTRA_PACKAGE = TransferLink("intra_package", None, 0.0)


def transfer_duration(
    num_bytes: float,
    src: MemoryDevice,
    dst: MemoryDevice,
    link: TransferLink = INTRA_PACKAGE,
) -> float:
    """Seconds to move ``num_bytes`` from ``src`` to ``dst`` over ``link``.

    Bandwidth is the minimum across the two memories and the link; latency is the
    link's (the memories contribute none).
    """

    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")
    bandwidths = [src.bandwidth_bytes_per_s, dst.bandwidth_bytes_per_s]
    if link.bandwidth_bytes_per_s is not None:
        bandwidths.append(link.bandwidth_bytes_per_s)
    bandwidth = min(bandwidths)
    return link.latency_s + num_bytes / bandwidth


def make_transfer_event(
    num_bytes: float,
    src: MemoryDevice,
    dst: MemoryDevice,
    link: TransferLink,
    start: float,
    *,
    group_index: int = 0,
    device_index: int = -1,
) -> ComputeEvent:
    """Build a ``phase="transfer"`` :class:`ComputeEvent` for a network move.

    The byte-transfer time is the bandwidth-bound part; the link latency is added
    on top to form the total duration (compute time is zero).
    """

    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")
    bandwidths = [src.bandwidth_bytes_per_s, dst.bandwidth_bytes_per_s]
    if link.bandwidth_bytes_per_s is not None:
        bandwidths.append(link.bandwidth_bytes_per_s)
    bandwidth = min(bandwidths)
    bandwidth_time = num_bytes / bandwidth
    duration = link.latency_s + bandwidth_time
    return ComputeEvent(
        group_index=group_index,
        phase="transfer",
        device_index=device_index,
        flops=0.0,
        bytes_read=num_bytes,
        compute_time=0.0,
        bandwidth_time=bandwidth_time,
        duration=duration,
        start=start,
        end=start + duration,
    )
