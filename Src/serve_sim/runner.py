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
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

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
from .orchestrator import (
    ProgressCallback,
    Request,
    RunProgress,
    RunResult,
    Simulator,
    StrategyConfig,
)
from .report import write_outputs
from .suite import Suite, build_suite_from_config
from .system import load_system
from .tokenizer import TiktokenTokenizer, Tokenizer, WhitespaceTokenizer


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


class ProgressReporter:
    """A :data:`ProgressCallback` that prints run progress to a stream.

    Each update reports sequences completed out of the suite total, the elapsed
    simulation time and the elapsed wall-clock time, refreshed in place. Updates
    are throttled to at most one per ``min_interval`` seconds (the final, 100%,
    update is always printed).
    """

    def __init__(self, stream: TextIO | None = None, min_interval: float = 0.25) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._min_interval = min_interval
        self._last_wall = -1.0

    def __call__(self, progress: RunProgress) -> None:
        done = progress.completed >= progress.total
        first = self._last_wall < 0.0
        if not done and not first and (
            progress.wall_time - self._last_wall < self._min_interval
        ):
            return
        self._last_wall = progress.wall_time
        line = (
            f"  {progress.completed}/{progress.total} sequences  "
            f"sim={progress.sim_time:8.3f}s  wall={progress.wall_time:7.3f}s"
        )
        self._stream.write("\r" + line.ljust(60))
        if done:
            self._stream.write("\n")
        self._stream.flush()


@dataclass(frozen=True)
class BuildProgress:
    """A progress update emitted while turning suite turns into requests.

    Attributes:
        workloads_done: Suite workloads (conversations) processed so far.
        workloads_total: Total workloads in the suite.
        requests_built: Requests produced so far (turns may be skipped).
        wall_time: Real seconds elapsed since request building started.
    """

    workloads_done: int
    workloads_total: int
    requests_built: int
    wall_time: float


# Called with a :class:`BuildProgress` as each suite workload is turned to requests.
BuildProgressCallback = Callable[["BuildProgress"], None]


class BuildProgressReporter:
    """A :data:`BuildProgressCallback` that prints request-building progress.

    Tokenizing every conversation turn of a large suite can take a while; this
    reports workloads processed out of the suite total, requests built so far and
    the elapsed wall-clock time, refreshed in place. Updates are throttled to at
    most one per ``min_interval`` seconds (the final, 100%, update always prints).
    """

    def __init__(self, stream: TextIO | None = None, min_interval: float = 0.25) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._min_interval = min_interval
        self._last_wall = -1.0

    def __call__(self, progress: BuildProgress) -> None:
        done = progress.workloads_done >= progress.workloads_total
        first = self._last_wall < 0.0
        if not done and not first and (
            progress.wall_time - self._last_wall < self._min_interval
        ):
            return
        self._last_wall = progress.wall_time
        line = (
            f"  building requests: {progress.workloads_done}/"
            f"{progress.workloads_total} workloads  "
            f"{progress.requests_built} requests  "
            f"wall={progress.wall_time:7.3f}s"
        )
        self._stream.write("\r" + line.ljust(60))
        if done:
            self._stream.write("\n")
        self._stream.flush()



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
        model_weight_loading=bool(cfg.get("model_weight_loading", True)),
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
    progress: BuildProgressCallback | None = None,
) -> list[Request]:
    """Turn each suite conversation's turns into single-sequence requests.

    Requests are spaced ``arrival_interval`` seconds apart (0 admits them all at
    once). A turn with no work at all (empty prompt and no output) is skipped.
    If ``progress`` is given it is called with a :class:`BuildProgress` after each
    workload is processed (tokenizing a large suite can be slow).
    """

    requests: list[Request] = []
    rid = 0
    total = len(suite)
    start_wall = time.perf_counter()
    for index, entry in enumerate(suite):
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
        if progress is not None:
            progress(BuildProgress(
                workloads_done=index + 1,
                workloads_total=total,
                requests_built=len(requests),
                wall_time=time.perf_counter() - start_wall,
            ))
    return requests


def run_from_config(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    loader: WorkloadLoader | None = None,
    tokenizer: Tokenizer | None = None,
    progress: ProgressCallback | None = None,
    build_progress: BuildProgressCallback | None = None,
) -> tuple[RunResult, Path]:
    """Run a simulation described by ``config_path`` and write its outputs.

    If ``progress`` is given it is called with a
    :class:`~serve_sim.orchestrator.RunProgress` as sequences complete (see
    :class:`ProgressReporter` for a printing implementation). If ``build_progress``
    is given it is called with a :class:`BuildProgress` as the suite's turns are
    tokenized into requests (see :class:`BuildProgressReporter`).

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
        progress=build_progress,
    )

    result = Simulator(system, strategy).run(requests, progress=progress)

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
