"""Drive a full simulation run from a single ``config.json``.

This is the end-to-end entry point behind the CLI: it reads one config file,
loads the system architecture, builds the test suite (drawing workloads through a
:class:`~serve_sim.dataset.WorkloadLoader`), resolves each referenced model,
turns the suite's conversation turns into :class:`~serve_sim.orchestrator.Request`
objects, runs the :class:`~serve_sim.orchestrator.Simulator`, and writes the raw
outputs under ``<output_root>/<run_id>/``.

Paths in the config are resolved relative to the config file's directory. The
dataset and tokenizer default to the live Hugging Face source and tiktoken; both
are injectable so the wiring can be exercised offline.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .dataset import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CONFIG,
    DEFAULT_DATASET,
    DEFAULT_SPLIT,
    HttpRowFetcher,
    LocalRowFetcher,
    WorkloadLoader,
)
from .model_config import load_model_config
from .orchestrator import Request, RunResult, Simulator, StrategyConfig
from .report import write_outputs
from .suite import Suite, build_suite_from_config
from .system import load_system
from .tokenizer import TiktokenTokenizer, Tokenizer, WhitespaceTokenizer


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _strategy_from_config(cfg: Mapping[str, Any]) -> StrategyConfig:
    """Map the PRD config parameters onto a :class:`StrategyConfig`."""

    return StrategyConfig(
        max_batch_size=int(cfg.get("max_concurrency", 8)),
        max_window_duration=float(cfg.get("concurrency_window_sec", 1.0)),
        target_concurrency=cfg.get("target_concurrency"),
        pipeline_parallel=int(cfg.get("pipeline_parallel", 1)),
        expert_parallel=int(cfg.get("expert_parallel", 1)),
        auto_parallelism=bool(cfg.get("auto_parallelism", False)),
        allow_pdd=bool(cfg.get("allow_pdd", True)),
        prefill_engine_fraction=float(cfg.get("prefill_engine_fraction", 0.5)),
        prefill_chunk_size=cfg.get("prefill_chunk_size"),
        event_random_factor_range=float(cfg.get("event_random_factor_range", 0.05)),
        random_seed=cfg.get("random_seed"),
    )


def _make_tokenizer(name: str) -> Tokenizer:
    if name == "whitespace":
        return WhitespaceTokenizer()
    if name == "tiktoken":
        return TiktokenTokenizer()
    raise ValueError(f"unknown tokenizer {name!r} (expected 'tiktoken' or 'whitespace')")


def _make_loader(cfg: Mapping[str, Any], base: Path) -> WorkloadLoader:
    """Build the workload loader, preferring a local cache when one is present.

    The optional ``dataset`` config block selects the dataset/config/split and a
    ``cache_dir`` (resolved relative to the config file, default ``Dataset``). If
    a populated cache exists there it is used (offline, reproducible); otherwise
    the loader falls back to the live HTTP API.
    """

    dataset = cfg.get("dataset") or {}
    name = dataset.get("dataset", DEFAULT_DATASET)
    config = dataset.get("config", DEFAULT_CONFIG)
    split = dataset.get("split", DEFAULT_SPLIT)
    cache_dir = _resolve(base, dataset.get("cache_dir", DEFAULT_CACHE_DIR))

    local = LocalRowFetcher(cache_dir, dataset=name, config=config, split=split)
    if local.exists():
        return WorkloadLoader(local)
    if dataset.get("require_cache"):
        raise FileNotFoundError(
            f"no cached dataset at {local.dir}; run `python cache_dataset.py` to "
            f"populate it (config sets dataset.require_cache)"
        )
    return WorkloadLoader(
        HttpRowFetcher(dataset=name, config=config, split=split)
    )


def _build_requests(
    suite: Suite,
    models: Mapping[str, Any],
    tokenizer: Tokenizer,
    arrival_interval: float,
    max_turns: int | None,
) -> list[Request]:
    """Turn each suite conversation's turns into single-sequence requests.

    Requests are spaced ``arrival_interval`` seconds apart (0 admits them all at
    once). A turn with no work at all (empty prompt and no output) is skipped.
    """

    requests: list[Request] = []
    rid = 0
    for entry in suite:
        model = models[entry.model]
        num_turns = len(entry.workload)
        if max_turns is not None:
            num_turns = min(num_turns, max_turns)
        for turn_index in range(num_turns):
            req = Request.from_workload(
                rid,
                entry.workload,
                model,
                tokenizer,
                arrival_time=rid * arrival_interval,
                turn_index=turn_index,
            )
            if req.prompt_tokens == 0 and req.output_tokens == 0:
                continue
            requests.append(req)
            rid += 1
    return requests


def run_from_config(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    loader: WorkloadLoader | None = None,
    tokenizer: Tokenizer | None = None,
) -> tuple[RunResult, Path]:
    """Run a simulation described by ``config_path`` and write its outputs.

    Returns the :class:`RunResult` and the output directory that was written.
    """

    config_path = Path(config_path)
    base = config_path.parent
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    system = load_system(_resolve(base, cfg["system"]))
    strategy = _strategy_from_config(cfg)

    suite_cfg = cfg["suite"]
    if isinstance(suite_cfg, str):
        with open(_resolve(base, suite_cfg), "r", encoding="utf-8") as handle:
            suite_cfg = json.load(handle)
    loader = loader or _make_loader(cfg, base)
    suite = build_suite_from_config(suite_cfg, loader, rng=cfg.get("random_seed"))

    tokenizer = tokenizer or _make_tokenizer(cfg.get("tokenizer", "tiktoken"))
    models_dir = _resolve(base, cfg.get("models_dir", "Models"))
    models = {name: load_model_config(models_dir / f"{name}.json")
              for name in suite.models}

    requests = _build_requests(
        suite,
        models,
        tokenizer,
        float(cfg.get("arrival_interval_sec", 0.0)),
        cfg.get("max_turns_per_workload"),
    )

    result = Simulator(system, strategy).run(requests)

    run_id = run_id or cfg.get("run_id") or f"run-{datetime.now():%Y%m%d-%H%M%S}"
    root = Path(output_root or cfg.get("output_root", "Outputs"))
    out_dir = write_outputs(
        result,
        root / run_id,
        run_id=run_id,
        config=cfg,
        time_buckets=int(cfg.get("report_time_buckets", 64)),
    )
    return result, out_dir
