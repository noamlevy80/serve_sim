"""End-to-end runner test: drive a full run from a config.json offline.

Uses the in-memory :class:`FakeRowFetcher` as the workload source and the
dependency-free whitespace tokenizer, so the whole config -> system -> suite ->
requests -> simulation -> outputs path runs without network access while still
loading the real system and model JSON files.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from serve_sim.dataset import WorkloadLoader
from serve_sim.runner import _arrival_times, _sample_arrival_gap, run_from_config
from serve_sim.tokenizer import WhitespaceTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_arrival_times_zero_variance_is_evenly_spaced():
    times = _arrival_times(5, mean=0.25, variance=0.0, rng=random.Random(0))
    assert times == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_arrival_times_zero_mean_admits_all_at_once():
    times = _arrival_times(4, mean=0.0, variance=1.0, rng=random.Random(0))
    assert times == [0.0, 0.0, 0.0, 0.0]


def test_arrival_times_are_monotonic_and_reproducible():
    a = _arrival_times(50, mean=0.25, variance=0.0625, rng=random.Random(7))
    b = _arrival_times(50, mean=0.25, variance=0.0625, rng=random.Random(7))
    assert a == b  # same seed -> identical draws
    assert a[0] == 0.0
    assert all(later >= earlier for earlier, later in zip(a, a[1:]))
    # Randomized gaps differ from the deterministic cadence.
    assert a != [i * 0.25 for i in range(50)]


def test_sample_arrival_gap_matches_requested_moments():
    rng = random.Random(123)
    mean, variance = 0.4, 0.09
    gaps = [_sample_arrival_gap(mean, variance, rng) for _ in range(20000)]
    sample_mean = sum(gaps) / len(gaps)
    sample_var = sum((g - sample_mean) ** 2 for g in gaps) / len(gaps)
    assert abs(sample_mean - mean) < 0.02
    assert abs(sample_var - variance) < 0.02
    assert all(g >= 0.0 for g in gaps)



def _write_config(path: Path) -> None:
    config = {
        "run_id": "test-run",
        "system": str(REPO_ROOT / "Systems" / "dual-node-b200.json"),
        "models_dir": str(REPO_ROOT / "Models"),
        "tokenizer": "whitespace",
        "suite": {
            "name": "runner-test",
            "type": "randomized",
            "num_workloads": 2,
            "models": ["gemma-4-31b"],
        },
        "max_concurrency": 4,
        "concurrency_window_sec": 0.5,
        "allow_pdd": False,
        "arrival_interval_sec": 0.0,
        "max_turns_per_workload": 1,
        "report_time_buckets": 8,
        "random_seed": 1,
    }
    path.write_text(json.dumps(config), encoding="utf-8")


def test_run_from_config_offline(tmp_path, fake_fetcher):
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    result, out_dir = run_from_config(
        config_path,
        output_root=tmp_path / "Outputs",
        loader=WorkloadLoader(fake_fetcher),
        tokenizer=WhitespaceTokenizer(),
    )

    assert out_dir == tmp_path / "Outputs" / "test-run"
    assert result.records, "expected at least one completed request"
    assert len(result.records) == 2

    for name in ("run_report.json", "requests.csv", "events_after_rescaling.csv",
                 "device_summary.csv", "device_timeline.csv", "config.json"):
        assert (out_dir / name).exists(), name

    report = json.loads((out_dir / "run_report.json").read_text())
    assert report["run_id"] == "test-run"
    assert report["report"]["num_requests"] == 2
    assert report["report"]["makespan_s"] > 0.0
    assert report["report"]["throughput_requests_per_s"] > 0.0

    with open(out_dir / "requests.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2

    echoed = json.loads((out_dir / "config.json").read_text())
    assert echoed["run_id"] == "test-run"


def test_run_id_override_and_default_output_root(tmp_path, fake_fetcher):
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    result, out_dir = run_from_config(
        config_path,
        output_root=tmp_path / "elsewhere",
        run_id="override-id",
        loader=WorkloadLoader(fake_fetcher),
        tokenizer=WhitespaceTokenizer(),
    )

    assert out_dir == tmp_path / "elsewhere" / "override-id"
    assert (out_dir / "run_report.json").exists()


def test_run_from_config_reports_build_progress(tmp_path, fake_fetcher):
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    updates = []
    result, _ = run_from_config(
        config_path,
        output_root=tmp_path / "Outputs",
        loader=WorkloadLoader(fake_fetcher),
        tokenizer=WhitespaceTokenizer(),
        build_progress=updates.append,
    )

    assert updates, "expected build-progress callbacks"
    # One update per suite workload, ending at the full count.
    assert updates[-1].workloads_done == updates[-1].workloads_total == 2
    assert updates[-1].requests_built == len(result.records)
    # workloads_done is monotonic non-decreasing and wall time non-negative.
    assert [u.workloads_done for u in updates] == sorted(u.workloads_done for u in updates)
    assert all(u.wall_time >= 0.0 for u in updates)
