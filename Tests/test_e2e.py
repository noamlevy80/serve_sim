"""End-to-end tests: run the whole pipeline a user runs, then check correctness.

Each test drives :func:`serve_sim.runner.run_from_config` over a committed
``config.json`` under ``Tests/e2e/Configs`` -- parsing the config, loading a
``System`` from JSON, building a suite, tokenizing it into requests, simulating,
and writing every output file -- and asserts the simulator behaves correctly for
one config-reachable feature. The fixtures (tiny devices, memories, models and
systems) live under ``Tests/e2e`` so the runs are self-contained, fast,
deterministic and feasible (the models are megabytes, the memories gigabytes).

The workload source and tokenizer are injected (an in-memory fetcher and the
dependency-free whitespace tokenizer), so the runs need no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from serve_sim.dataset import WorkloadLoader
from serve_sim.report import memory_summaries
from serve_sim.runner import run_from_config
from serve_sim.tokenizer import WhitespaceTokenizer

E2E_DIR = Path(__file__).resolve().parent / "e2e"
CONFIG_DIR = E2E_DIR / "Configs"
ALL_CONFIGS = sorted(p.name for p in CONFIG_DIR.glob("*.json"))

OUTPUT_FILES = (
    "run_report.json",
    "run_report.txt",
    "requests.csv",
    "events_before_rescaling.csv",
    "events_after_rescaling.csv",
    "device_summary.csv",
    "device_timeline.csv",
    "memory_summary.csv",
    "config.json",
)


# --- offline workload source ------------------------------------------------


def _msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def _session(session_id: str, model: str, num_turns: int) -> list[dict[str, Any]]:
    """A prefix-growing multi-turn session with multi-word (multi-token) content."""

    messages = [
        _msg("system", "you are a careful helpful assistant ready to help"),
        _msg("user", "please summarize the quick brown fox story for me now"),
    ]
    rows: list[dict[str, Any]] = []
    for turn in range(num_turns):
        if turn > 0:
            messages.append(
                _msg("assistant", f"answer number {turn} the quick brown fox jumps high")
            )
            messages.append(
                _msg("user", f"a follow up question {turn} about the lazy dog please")
            )
        rows.append(
            {
                "session_id": session_id,
                "model": model,
                "input": [dict(m) for m in messages],
                "output_length": 6,
                "pre_gap": 0.0 if turn == 0 else float(turn),
            }
        )
    return rows


def _dataset() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows += _session("sess-0", "model-x", num_turns=3)
    rows += _session("sess-1", "model-y", num_turns=2)
    rows += _session("sess-2", "model-x", num_turns=4)
    return rows


class _ListFetcher:
    """In-memory row fetcher mirroring the datasets-server page shape."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __call__(self, offset: int, length: int) -> Mapping[str, Any]:
        window = self.rows[offset : offset + length]
        return {
            "rows": [
                {"row_idx": offset + i, "row": row} for i, row in enumerate(window)
            ],
            "num_rows_total": len(self.rows),
        }


def _run(config_name: str, tmp_path: Path, run_id: str | None = None):
    """Run one e2e config offline; returns ``(RunResult, output_dir)``."""

    loader = WorkloadLoader(_ListFetcher(_dataset()))
    return run_from_config(
        CONFIG_DIR / config_name,
        output_root=tmp_path / "Outputs",
        run_id=run_id,
        loader=loader,
        tokenizer=WhitespaceTokenizer(),
    )


def _input_memory(result) -> dict[str, Any]:
    """The single input-NVM entry from the memory report."""

    inputs = [m for m in memory_summaries(result) if m["role"] == "input"]
    assert len(inputs) == 1, "expected exactly one input memory"
    return inputs[0]


def _nvm_loads(result) -> list:
    """Rescaled weight-load transfer events streamed from the input NVM."""

    nvm_name = _input_memory(result)["memory"]
    return [
        e
        for e in result.events
        if e.rescaled and e.phase == "weight_transfer" and e.memory == nvm_name
    ]


# --- every config: completes, is feasible, deterministic --------------------


@pytest.mark.parametrize("config_name", ALL_CONFIGS)
def test_e2e_config_runs_completes_and_is_feasible(config_name, tmp_path):
    result, out_dir = _run(config_name, tmp_path, run_id=f"{Path(config_name).stem}-a")

    # Every request is retired and the run advances real time.
    assert result.records, f"{config_name}: no completed requests"
    assert all(r.completion_time >= r.arrival_time for r in result.records)
    assert result.makespan > 0.0

    # All output files were written and the report echoes the run.
    for name in OUTPUT_FILES:
        assert (out_dir / name).exists(), f"{config_name}: missing {name}"
    report = json.loads((out_dir / "run_report.json").read_text())
    assert report["report"]["num_requests"] == len(result.records)
    assert report["report"]["throughput_requests_per_s"] > 0.0

    # Every event is attributed to a real device (or the no-device sentinel),
    # and no memory is asked to hold more than its capacity (the run is feasible).
    slot_devices = {d for job in result.jobs for d in job.devices}
    for ev in result.events:
        if ev.device:
            assert ev.device in slot_devices, f"{config_name}: phantom device {ev.device}"
    for mem in memory_summaries(result):
        assert mem["occupancy_fraction"] <= 1.0 + 1e-9, (
            f"{config_name}: {mem['memory']} over capacity "
            f"({mem['occupancy_fraction']:.3f})"
        )


@pytest.mark.parametrize("config_name", ALL_CONFIGS)
def test_e2e_config_is_deterministic(config_name, tmp_path):
    first, _ = _run(config_name, tmp_path, run_id=f"{Path(config_name).stem}-1")
    second, _ = _run(config_name, tmp_path, run_id=f"{Path(config_name).stem}-2")

    assert len(first.records) == len(second.records)
    assert first.makespan == pytest.approx(second.makespan)


# --- per-feature correctness ------------------------------------------------


def test_weight_loading_on_streams_from_input_nvm(tmp_path):
    result, _ = _run("basic.json", tmp_path)

    assert _input_memory(result)["bytes_moved"] > 0
    loads = _nvm_loads(result)
    assert loads, "expected weight-load transfers from the input NVM"

    # A weight load precedes (gates) the model's first compute.
    first_load = min(e.start for e in loads)
    first_compute = min(
        e.start
        for e in result.events
        if e.rescaled and e.phase in ("prefill", "decode")
    )
    assert first_load <= first_compute


def test_weight_loading_off_has_no_input_nvm_traffic(tmp_path):
    result, _ = _run("weight_loading_off.json", tmp_path)

    assert result.records
    assert _input_memory(result)["bytes_moved"] == 0
    assert not _nvm_loads(result), "weights should be assumed resident when off"


def test_pdd_splits_prefill_and_decode_across_disjoint_pools(tmp_path):
    result, _ = _run("pdd.json", tmp_path)

    rescaled = [e for e in result.events if e.rescaled and e.device]
    prefill_devices = {e.device for e in rescaled if e.job_phase == "prefill"}
    decode_devices = {e.device for e in rescaled if e.job_phase == "decode"}

    assert prefill_devices, "expected prefill events"
    assert decode_devices, "expected decode events"
    assert prefill_devices.isdisjoint(decode_devices), (
        "prefill and decode pools must be disjoint device sets"
    )
    # Every request gets a first token (decode began after prefill + KV transfer).
    assert all(r.first_token_time is not None for r in result.records)


def test_fixed_parallelism_runs_each_job_on_two_devices(tmp_path):
    result, _ = _run("fixed_parallel.json", tmp_path)

    compute_jobs = [j for j in result.jobs if j.request_ids]
    assert compute_jobs
    assert all(len(j.devices) == 2 for j in compute_jobs), (
        "pipeline_parallel=2 should place every job on a two-device slot"
    )


def test_auto_parallelism_runs_moe_feasibly(tmp_path):
    result, _ = _run("auto_parallel.json", tmp_path)

    assert result.records
    # The fixed two-device budget is wired through the planner.
    assert any(len(j.devices) == 2 for j in result.jobs)
    for mem in memory_summaries(result):
        assert mem["occupancy_fraction"] <= 1.0 + 1e-9


def test_chunked_prefill_adds_prefill_groups(tmp_path):
    unchunked, _ = _run("basic.json", tmp_path, run_id="twin-unchunked")
    chunked, _ = _run("chunked_prefill.json", tmp_path, run_id="twin-chunked")

    def prefill_events(result):
        return [e for e in result.events if e.rescaled and e.phase == "prefill"]

    assert len(prefill_events(chunked)) > len(prefill_events(unchunked)), (
        "a small prefill_chunk_size should split prompts into more prefill groups"
    )


def test_heterogeneous_system_uses_both_device_types(tmp_path):
    result, _ = _run("heterogeneous.json", tmp_path)

    memory_names = {m["memory"] for m in memory_summaries(result)}
    assert any("Tiny HBM" in n for n in memory_names)
    assert any("Tiny SRAM" in n for n in memory_names)

    used_devices = {e.device for e in result.events if e.rescaled and e.device}
    assert any("Tiny GPU" in d for d in used_devices), "node-0 GPUs should run work"
    assert any("Tiny Accel" in d for d in used_devices), "node-1 accelerators should run work"


def test_multi_model_suite_serves_both_models(tmp_path):
    result, _ = _run("multi_model.json", tmp_path)

    assert result.records
    assert _input_memory(result)["bytes_moved"] > 0
    loads = _nvm_loads(result)
    # Two models have different weight footprints, so loading both shows up as at
    # least two distinct transfer sizes on the input NVM.
    distinct_sizes = {round(e.bytes_read) for e in loads}
    assert len(distinct_sizes) >= 2, "expected weight loads for two distinct models"


def test_concurrency_cap_serializes_the_pipeline(tmp_path):
    serial, _ = _run("serial_concurrency.json", tmp_path, run_id="serial")
    parallel, _ = _run("parallel_concurrency.json", tmp_path, run_id="parallel")

    assert len(serial.records) == len(parallel.records)
    assert serial.makespan > parallel.makespan, (
        "max_concurrency=1 should finish later than a high concurrency cap"
    )
