"""Unit tests for the PDD primitives (Stage 5, step 3).

These cover the orchestrator-independent pieces of prefill/decode disaggregation:
splitting a sequence into prefill-only and decode-only work, sizing the KV cache
the handoff moves, and timing that move over the system's links. Timing and byte
references are derived independently of the simulator.
"""

from __future__ import annotations

import pytest

from serve_sim import (
    context_kv_bytes,
    kv_bytes_per_token,
    kv_transfer_duration,
    split_work,
)
from serve_sim.blocks import Attention, Layer, LayeredModel
from serve_sim.hardware import ComputeDevice, MemoryDevice
from serve_sim.model import toy_model
from serve_sim.system import Network, Node, System
from serve_sim.transfer import transfer_duration


# --- helpers -------------------------------------------------------------------


def make_memory(name="mem", bw=1e12, cap=80e9):
    return MemoryDevice(name, capacity_bytes=cap, bandwidth_bytes_per_s=bw)


def make_device(name="gpu", bw=1e12):
    return ComputeDevice(
        name, peak_flops_fp16=100e12, first_tier_memory=make_memory(f"{name}-mem", bw=bw)
    )


def make_system(devices, cxl_bw=1e11, su_bw=1e10):
    network = Network(
        scale_up_bandwidth_bytes_per_s=su_bw,
        scale_up_latency_s=1e-6,
        cxl_bandwidth_bytes_per_s=cxl_bw,
        cxl_latency_s=1e-7,
    )
    node = Node(name="n0", compute_devices=tuple(devices), node_memory=make_memory("node"))
    return System(
        name="s", network=network, input_memory=make_memory("nvm", bw=5e9, cap=1e12),
        nodes=(node,),
    )


def two_node_system(dev_a, dev_b):
    network = Network(
        scale_up_bandwidth_bytes_per_s=1e10,
        scale_up_latency_s=1e-6,
        cxl_bandwidth_bytes_per_s=1e11,
        cxl_latency_s=1e-7,
    )
    n0 = Node(name="n0", compute_devices=(dev_a,), node_memory=make_memory("node0"))
    n1 = Node(name="n1", compute_devices=(dev_b,), node_memory=make_memory("node1"))
    return System(
        name="s", network=network, input_memory=make_memory("nvm", bw=5e9, cap=1e12),
        nodes=(n0, n1),
    )


# --- work split ----------------------------------------------------------------


def test_split_work_partitions_phases():
    prefill, decode = split_work(cached_tokens=10, prompt_tokens=100, output_tokens=20)
    # Prefill: prefills the new prompt tokens, decodes nothing.
    assert prefill.cached_tokens == 10
    assert prefill.prefill_tokens == 90
    assert prefill.decode_tokens == 0
    # Decode: whole prompt is cached, only generates.
    assert decode.cached_tokens == 100
    assert decode.prefill_tokens == 0
    assert decode.decode_tokens == 20


def test_split_work_base_context_matches():
    prefill, decode = split_work(0, 64, 8)
    # Decode starts from exactly the context prefill leaves behind.
    assert decode.cached_tokens == prefill.base_tokens


def test_split_work_no_cache():
    prefill, decode = split_work(0, 50, 5)
    assert prefill.prefill_tokens == 50
    assert decode.cached_tokens == 50


def test_split_work_rejects_overcache():
    with pytest.raises(ValueError):
        split_work(cached_tokens=120, prompt_tokens=100, output_tokens=10)


# --- KV sizing -----------------------------------------------------------------


def test_kv_bytes_per_token_sums_layers():
    attn = Attention(hidden_size=64, attention_type="MHA", num_query_heads=4, head_dim=16)
    model = LayeredModel(
        layers=(Layer(mixer=attn), Layer(mixer=attn), Layer(mixer=attn)),
        hidden_size=64,
        vocab_size=512,
        kv_dtype_bytes=2,
    )
    per_layer = attn.kv_bytes_per_token(2)
    assert kv_bytes_per_token(model) == 3 * per_layer


def test_context_kv_bytes_scales_with_tokens():
    model = toy_model()
    per_token = kv_bytes_per_token(model)
    assert context_kv_bytes(model, 100) == pytest.approx(100 * per_token)
    assert context_kv_bytes(model, 0) == 0.0


def test_context_kv_bytes_rejects_negative():
    with pytest.raises(ValueError):
        context_kv_bytes(toy_model(), -1)


# --- transfer timing -----------------------------------------------------------


def test_transfer_intra_package_when_same_device():
    # A device "transferring" to itself uses its own bandwidth, no link latency.
    dev = make_device(bw=1e12)
    system = make_system([dev])
    dur = kv_transfer_duration(1e9, dev, dev, system)
    assert dur == pytest.approx(1e9 / 1e12)


def test_transfer_cxl_same_node():
    a = make_device("a", bw=1e12)
    b = make_device("b", bw=1e12)
    system = make_system([a, b], cxl_bw=1e11)
    # Link bandwidth (1e11) is the min, plus CXL latency.
    expected = 1e-7 + 1e9 / 1e11
    assert kv_transfer_duration(1e9, a, b, system) == pytest.approx(expected)


def test_transfer_scale_up_across_nodes():
    a = make_device("a", bw=1e12)
    b = make_device("b", bw=1e12)
    system = two_node_system(a, b)
    expected = 1e-6 + 1e9 / 1e10  # scale-up latency + bandwidth-bound
    assert kv_transfer_duration(1e9, a, b, system) == pytest.approx(expected)


def test_transfer_matches_transfer_duration_helper():
    a = make_device("a", bw=2e12)
    b = make_device("b", bw=1e12)
    system = make_system([a, b], cxl_bw=5e11)
    link = system.link_between(a.first_tier_memory, b.first_tier_memory)
    expected = transfer_duration(
        2e9, a.first_tier_memory, b.first_tier_memory, link
    )
    assert kv_transfer_duration(2e9, a, b, system) == pytest.approx(expected)
