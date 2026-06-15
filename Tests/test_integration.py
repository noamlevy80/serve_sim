"""Integration tests against the live dataset.

Run with ``pytest -m network``. Skipped automatically when the dataset API is
unreachable so the default suite stays offline and deterministic.
"""

from __future__ import annotations

import pytest

from serve_sim.dataset import HttpRowFetcher, WorkloadLoader

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def live_loader():
    fetcher = HttpRowFetcher()
    try:
        # cheap probe; skip the whole module if the service is unavailable
        fetcher(0, 1)
    except Exception as exc:  # noqa: BLE001 - network/transport errors of any kind
        pytest.skip(f"dataset API unreachable: {exc}")
    return WorkloadLoader(fetcher=fetcher)


def test_live_num_rows(live_loader):
    assert live_loader.num_rows() > 1000


def test_live_load_first_workload(live_loader):
    wl = live_loader.load_first()
    assert wl.session_id
    assert wl.model
    assert wl.num_turns >= 1
    # first turn always starts with a system message in this dataset
    assert wl.turns[0].messages[0].role == "system"


def test_live_workload_has_prefix_growth(live_loader):
    wl = live_loader.load_first()
    wl.validate_prefix_growth()


def test_live_load_session_at_matches_iteration(live_loader):
    # the workload covering row 0 must equal the first iterated workload
    by_offset = live_loader.load_session_at(0)
    first = live_loader.load_first()
    assert by_offset.session_id == first.session_id
    assert by_offset.num_turns == first.num_turns


def test_live_multi_turn_session_is_downloadable(live_loader):
    # scan forward to find a session with several turns and download it whole
    target = None
    for wl in live_loader.iter_workloads():
        if wl.num_turns >= 3:
            target = wl
            break
    assert target is not None
    target.validate_prefix_growth()
    # later turns add messages on top of earlier ones
    assert len(target.turns[-1].messages) > len(target.turns[0].messages)
