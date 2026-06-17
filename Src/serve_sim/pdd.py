"""Prefill / decode disaggregation (PDD) primitives.

PDD serves a request in two phases on two separate engine pools: the prompt is
*prefilled* on a prefill engine, the resulting KV cache is *transferred* over the
network to a decode engine, and the tokens are *decoded* there. This module holds
the pure, orchestrator-independent pieces of that split:

* :func:`split_work` cuts one sequence's work into a prefill-only and a
  decode-only :class:`~serve_sim.tracker.SequenceWork`. The prefill half runs the
  prompt forward pass and generates nothing; the decode half treats the whole
  prompt as already cached and only generates.
* :func:`context_kv_bytes` sizes the KV cache that prefill produces and the
  handoff must move (the per-token KV across every layer, times the context
  length).
* :func:`kv_transfer_duration` times that move over the link between the two
  engines' memories, reusing the network-aware transfer cost model.

The engine-pool split and the two-phase scheduling live in the orchestrator; this
module is just arithmetic so it can be unit-tested on its own.
"""

from __future__ import annotations

from .blocks import LayeredModel
from .hardware import ComputeDevice
from .system import System
from .tracker import SequenceWork
from .transfer import transfer_duration


def split_work(
    cached_tokens: int, prompt_tokens: int, output_tokens: int
) -> tuple[SequenceWork, SequenceWork]:
    """Split one sequence into (prefill-only, decode-only) work.

    The prefill job prefills the new prompt tokens and decodes nothing; the
    decode job treats the full prompt as cached and only decodes.
    """

    if cached_tokens > prompt_tokens:
        raise ValueError("cached_tokens cannot exceed prompt_tokens")
    prefill = SequenceWork(
        cached_tokens=cached_tokens,
        prefill_tokens=prompt_tokens - cached_tokens,
        decode_tokens=0,
    )
    decode = SequenceWork(
        cached_tokens=prompt_tokens,
        prefill_tokens=0,
        decode_tokens=output_tokens,
    )
    return prefill, decode


def kv_bytes_per_token(model) -> int:
    """KV cache bytes produced per context token, summed across all layers."""

    model = LayeredModel.from_model(model)
    return sum(
        layer.kv_bytes_per_token(model.kv_dtype_bytes) for layer in model.layers
    )


def context_kv_bytes(model, context_tokens: int) -> float:
    """Total KV cache bytes for ``context_tokens`` tokens of context."""

    if context_tokens < 0:
        raise ValueError("context_tokens must be non-negative")
    return context_tokens * kv_bytes_per_token(model)


def kv_transfer_duration(
    num_bytes: float,
    src: ComputeDevice,
    dst: ComputeDevice,
    system: System,
) -> float:
    """Seconds to move ``num_bytes`` of KV from a prefill to a decode engine.

    The move runs between the two engines' first-tier memories over the link the
    system places between them (intra-package, CXL or scale-up).
    """

    src_mem = src.first_tier_memory
    dst_mem = dst.first_tier_memory
    link = system.link_between(src_mem, dst_mem)
    return transfer_duration(num_bytes, src_mem, dst_mem, link)
