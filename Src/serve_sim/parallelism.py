"""Per-batch parallelism planning for a fixed-size engine.

An engine slot has a fixed device count -- the *parallelism degree*
``pipeline_parallel x expert_parallel``. When auto-parallelism is enabled the
orchestrator does not take the strategy's ``pp``/``ep`` verbatim; instead, for
each batch it searches the ways to factor that fixed ``degree`` into a
``(pp, ep)`` arrangement and picks the fastest one that still fits in device
memory. The device count never changes -- only how the engine is wired.

Two ingredients drive the choice:

* **Speed** (a lightweight roofline estimate). A single batch has no pipeline
  overlap, so the stages of a forward pass run sequentially and ``pp`` does not
  shorten the critical path. Expert parallelism splits each stage's work across
  its ``ep`` ranks, so the batch runs roughly ``ep`` times faster. The estimate
  is therefore ``max(compute_bound, bandwidth_bound) / ep`` -- enough to rank
  the candidates without building the full event schedule.

* **Memory feasibility** (pure expert parallelism). ``pp`` shards the layers
  across stages, so each device holds only ``num_layers / pp`` layers' weights
  and KV. ``ep`` shards *only the routed experts* (expert ``e`` lives on rank
  ``e % ep``); the attention/dense/shared/LM-head weights and the KV cache are
  replicated across the ``ep`` ranks. Hence ``pp`` -- not ``ep`` -- is what
  relieves dense-weight and KV pressure, and the search will reach for more
  pipeline stages when a batch will not fit otherwise.

This module is pure arithmetic: it stores no tensors and runs no events. The
orchestrator feeds it the conserved work (FLOPs by dtype, bytes) and the KV
token count of a batch and receives a :class:`ParallelismChoice`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .blocks import DenseFFN, LayeredModel, MoEFFN
from .hardware import ComputeDevice


@dataclass(frozen=True)
class ParallelismChoice:
    """The arrangement chosen for one batch on a fixed-size engine.

    Attributes:
        pipeline_parallel: Pipeline stages (shards layers).
        expert_parallel: Expert-parallel ranks per stage (shards routed experts).
        per_device_bytes: Peak resident footprint on the busiest device.
        estimated_time: Lightweight roofline time estimate used to rank options.
    """

    pipeline_parallel: int
    expert_parallel: int
    per_device_bytes: float
    estimated_time: float


@dataclass(frozen=True)
class _LayerFootprint:
    """Pre-computed per-layer byte quantities for the footprint model."""

    replicated_bytes: float  # mixer + non-expert FFN weights (replicated over ep)
    routed_expert_bytes: float  # bytes of ONE routed expert (0 if not MoE)
    num_experts: int  # routed experts in this layer (0 if not MoE)
    kv_bytes_per_token: float  # KV cache bytes per token (replicated over ep)


class ParallelismPlanner:
    """Chooses a ``(pp, ep)`` arrangement for batches on a fixed-size engine."""

    def __init__(self, model, device: ComputeDevice) -> None:
        self.model = LayeredModel.from_model(model)
        self.device = device
        self._capacity = device.first_tier_memory.capacity_bytes
        pdtype = self.model.param_dtype_bytes
        kvdtype = self.model.kv_dtype_bytes

        self._layers: list[_LayerFootprint] = []
        for layer in self.model.layers:
            mixer_params = layer.mixer.weight_params if layer.mixer is not None else 0
            routed_bytes = 0.0
            num_experts = 0
            non_expert_ffn_params = 0
            if isinstance(layer.ffn, MoEFFN):
                non_expert_ffn_params = (
                    layer.ffn.shared_expert_params + layer.ffn.latent_proj_params
                )
                routed_bytes = layer.ffn.routed_expert_params * pdtype
                num_experts = layer.ffn.num_experts
            elif isinstance(layer.ffn, DenseFFN):
                non_expert_ffn_params = layer.ffn.weight_params
            self._layers.append(
                _LayerFootprint(
                    replicated_bytes=(mixer_params + non_expert_ffn_params) * pdtype,
                    routed_expert_bytes=routed_bytes,
                    num_experts=num_experts,
                    kv_bytes_per_token=layer.kv_bytes_per_token(kvdtype),
                )
            )
        self._lm_head_bytes = self.model.lm_head_bytes

    @property
    def capacity(self) -> float:
        """First-tier memory capacity (bytes) of the engine's device."""

        return self._capacity

    # --- candidate enumeration ----------------------------------------------

    def factorizations(self, degree: int) -> list[tuple[int, int]]:
        """``(pp, ep)`` pairs with ``pp*ep == degree`` and ``pp | num_layers``.

        Ordered fastest-first (descending ``ep``), since a single batch gets no
        pipeline overlap and only expert parallelism shortens it.
        """

        if degree < 1:
            raise ValueError("degree must be >= 1")
        pairs: list[tuple[int, int]] = []
        for pp in range(1, degree + 1):
            if degree % pp != 0:
                continue
            if self.model.num_layers % pp != 0:
                continue
            pairs.append((pp, degree // pp))
        pairs.sort(key=lambda pe: pe[1], reverse=True)
        return pairs

    # --- footprint ----------------------------------------------------------

    def footprint(
        self, pipeline_parallel: int, expert_parallel: int, kv_tokens: int
    ) -> float:
        """Peak resident bytes on the busiest device for an arrangement.

        ``pp`` splits the layers across stages; ``ep`` replicates the
        non-expert weights and KV across ranks while sharding the routed experts
        (the busiest rank owns ``ceil(num_experts / ep)`` of them).
        """

        if self.model.num_layers % pipeline_parallel != 0:
            raise ValueError("pipeline_parallel must divide num_layers")
        layers_per_stage = self.model.num_layers // pipeline_parallel
        peak = 0.0
        for stage in range(pipeline_parallel):
            stage_layers = self._layers[
                stage * layers_per_stage : (stage + 1) * layers_per_stage
            ]
            replicated = sum(
                lf.replicated_bytes + lf.kv_bytes_per_token * kv_tokens
                for lf in stage_layers
            )
            if stage == pipeline_parallel - 1:
                replicated += self._lm_head_bytes
            experts = sum(
                math.ceil(lf.num_experts / expert_parallel) * lf.routed_expert_bytes
                for lf in stage_layers
                if lf.num_experts
            )
            peak = max(peak, replicated + experts)
        return peak

    # --- speed estimate -----------------------------------------------------

    def estimate_time(
        self,
        expert_parallel: int,
        flops_by_dtype: Mapping[int, float],
        total_bytes: float,
    ) -> float:
        """Lightweight roofline time: ``max(compute, bandwidth) / ep``."""

        compute = sum(
            flops / self.device.effective_flops(dtype)
            for dtype, flops in flops_by_dtype.items()
        )
        bandwidth = total_bytes / self.device.bandwidth_bytes_per_s
        return max(compute, bandwidth) / expert_parallel

    # --- planning -----------------------------------------------------------

    def plan(
        self,
        degree: int,
        *,
        kv_tokens: int,
        flops_by_dtype: Mapping[int, float],
        total_bytes: float,
    ) -> ParallelismChoice:
        """Pick the fastest ``(pp, ep)`` that fits; raise if none do."""

        candidates = self.factorizations(degree)
        best_unfit: tuple[float, int, int] | None = None
        for pp, ep in candidates:  # fastest (max ep) first
            per_device = self.footprint(pp, ep, kv_tokens)
            if per_device <= self._capacity:
                return ParallelismChoice(
                    pipeline_parallel=pp,
                    expert_parallel=ep,
                    per_device_bytes=per_device,
                    estimated_time=self.estimate_time(ep, flops_by_dtype, total_bytes),
                )
            if best_unfit is None or per_device < best_unfit[0]:
                best_unfit = (per_device, pp, ep)

        assert best_unfit is not None
        smallest, pp, ep = best_unfit
        raise ValueError(
            f"no parallelism arrangement of degree {degree} fits a batch of "
            f"{kv_tokens} KV tokens in {self._capacity:.0f} bytes; the smallest "
            f"footprint was {smallest:.0f} bytes at pp={pp}, ep={ep}"
        )
