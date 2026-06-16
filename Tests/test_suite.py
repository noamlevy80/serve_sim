"""Tests for randomized test-suite construction.

A randomized suite draws N random workloads from the dataset and binds each to a
model chosen at random from a configured list. Dataset access is injected via a
:class:`WorkloadLoader` over an in-memory fetcher, so these tests run offline and
deterministically (a fixed seed reproduces the suite).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from serve_sim.dataset import WorkloadLoader
from serve_sim.suite import (
    RandomizedSuiteConfig,
    Suite,
    SuiteEntry,
    build_randomized_suite,
    build_suite_from_config,
    load_suite,
)
from serve_sim.workload import Workload
from conftest import FakeRowFetcher, make_session_rows

MODELS = ("model-x", "model-y", "model-z")


@pytest.fixture
def loader() -> WorkloadLoader:
    """A loader over a small, fixed multi-session dataset (4 sessions)."""

    rows = []
    rows += make_session_rows("sess-a", "src-model", num_turns=3)
    rows += make_session_rows("sess-b", "src-model", num_turns=1)
    rows += make_session_rows("sess-c", "src-model", num_turns=4)
    rows += make_session_rows("sess-d", "src-model", num_turns=2)
    return WorkloadLoader(FakeRowFetcher(rows), page_size=100)


def config(n: int) -> RandomizedSuiteConfig:
    return RandomizedSuiteConfig(num_workloads=n, models=MODELS)


# --- basic construction ---------------------------------------------------------


def test_suite_has_requested_number_of_entries(loader: WorkloadLoader) -> None:
    suite = build_randomized_suite(config(8), loader, rng=0)
    assert isinstance(suite, Suite)
    assert len(suite) == 8


def test_every_entry_is_a_workload_and_a_listed_model(loader: WorkloadLoader) -> None:
    suite = build_randomized_suite(config(10), loader, rng=1)
    for entry in suite:
        assert isinstance(entry, SuiteEntry)
        assert isinstance(entry.workload, Workload)
        assert entry.model in MODELS


def test_drawn_workloads_are_real_sessions(loader: WorkloadLoader) -> None:
    valid_sessions = {"sess-a", "sess-b", "sess-c", "sess-d"}
    suite = build_randomized_suite(config(12), loader, rng=2)
    for entry in suite:
        assert entry.workload.session_id in valid_sessions
        # the loader returns whole sessions, so each workload is internally valid.
        entry.workload.validate_prefix_growth()


# --- determinism ----------------------------------------------------------------


def test_same_seed_reproduces_the_suite(loader: WorkloadLoader) -> None:
    a = build_randomized_suite(config(10), loader, rng=42)
    b = build_randomized_suite(config(10), loader, rng=42)
    assert [(e.workload.session_id, e.model) for e in a] == [
        (e.workload.session_id, e.model) for e in b
    ]


def test_different_seeds_can_differ(loader: WorkloadLoader) -> None:
    a = build_randomized_suite(config(10), loader, rng=1)
    b = build_randomized_suite(config(10), loader, rng=2)
    # not a strict guarantee, but with 4 sessions x 3 models over 10 draws the
    # two seeds are overwhelmingly likely to produce different sequences.
    assert [(e.workload.session_id, e.model) for e in a] != [
        (e.workload.session_id, e.model) for e in b
    ]


def test_models_are_drawn_from_the_configured_list(loader: WorkloadLoader) -> None:
    suite = build_randomized_suite(config(30), loader, rng=7)
    assert suite.models.issubset(set(MODELS))


# --- config parsing + dispatch --------------------------------------------------


def test_from_config_parses_fields() -> None:
    cfg = RandomizedSuiteConfig.from_config(
        {"num_workloads": 5, "models": ["a", "b"]}
    )
    assert cfg.num_workloads == 5
    assert cfg.models == ("a", "b")


def test_build_from_config_dispatches_randomized(loader: WorkloadLoader) -> None:
    suite = build_suite_from_config(
        {"type": "randomized", "num_workloads": 4, "models": list(MODELS)},
        loader,
        rng=3,
    )
    assert len(suite) == 4


def test_build_from_config_defaults_to_randomized(loader: WorkloadLoader) -> None:
    suite = build_suite_from_config(
        {"num_workloads": 3, "models": list(MODELS)}, loader, rng=3
    )
    assert len(suite) == 3


def test_directed_suite_is_not_implemented(loader: WorkloadLoader) -> None:
    with pytest.raises(NotImplementedError):
        build_suite_from_config({"type": "directed"}, loader)


def test_unknown_suite_type_rejected(loader: WorkloadLoader) -> None:
    with pytest.raises(ValueError):
        build_suite_from_config({"type": "bogus"}, loader)


# --- validation -----------------------------------------------------------------


def test_zero_workloads_rejected() -> None:
    with pytest.raises(ValueError):
        RandomizedSuiteConfig(num_workloads=0, models=MODELS)


def test_empty_model_list_rejected() -> None:
    with pytest.raises(ValueError):
        RandomizedSuiteConfig(num_workloads=3, models=())


def test_empty_suite_rejected() -> None:
    with pytest.raises(ValueError):
        Suite(())


# --- loading the shipped sample suite ------------------------------------------


def test_sample_suite_json_loads_and_builds(loader: WorkloadLoader) -> None:
    path = Path(__file__).resolve().parents[1] / "Suites" / "randomized-sample.json"
    suite = load_suite(path, loader, rng=0)
    assert len(suite) == 16
    assert suite.models.issubset(
        {"deepseek-v3.2", "gemma-4-31b", "nemotron-3-ultra"}
    )
