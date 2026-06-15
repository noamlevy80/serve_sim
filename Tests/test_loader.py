"""Tests for WorkloadLoader paging, grouping and session expansion."""

from __future__ import annotations

import pytest

from serve_sim.dataset import HttpRowFetcher, WorkloadLoader
from conftest import FakeRowFetcher, make_session_rows


# --- construction / validation --------------------------------------------------


def test_loader_rejects_bad_page_size():
    with pytest.raises(ValueError, match="page_size"):
        WorkloadLoader(fetcher=FakeRowFetcher([]), page_size=0)
    with pytest.raises(ValueError, match="page_size"):
        WorkloadLoader(fetcher=FakeRowFetcher([]), page_size=101)


def test_num_rows(fake_fetcher, fake_dataset):
    loader = WorkloadLoader(fetcher=fake_fetcher)
    assert loader.num_rows() == len(fake_dataset)


# --- iter_workloads -------------------------------------------------------------


def test_iter_workloads_groups_contiguous_sessions(fake_fetcher):
    loader = WorkloadLoader(fetcher=fake_fetcher, page_size=100)
    workloads = list(loader.iter_workloads())
    assert [w.session_id for w in workloads] == ["sess-a", "sess-b", "sess-c"]
    assert [w.num_turns for w in workloads] == [3, 1, 4]


def test_iter_workloads_handles_session_spanning_pages():
    # single 5-turn session, small page size forces multiple fetches
    rows = make_session_rows("solo", "m", num_turns=5)
    fetcher = FakeRowFetcher(rows)
    loader = WorkloadLoader(fetcher=fetcher, page_size=2)
    workloads = list(loader.iter_workloads())
    assert len(workloads) == 1
    assert workloads[0].num_turns == 5
    # paging happened
    assert len(fetcher.calls) >= 3


def test_iter_workloads_across_pages_with_boundaries():
    rows = (
        make_session_rows("a", "m", num_turns=3)
        + make_session_rows("b", "m", num_turns=2)
        + make_session_rows("c", "m", num_turns=3)
    )
    fetcher = FakeRowFetcher(rows)
    loader = WorkloadLoader(fetcher=fetcher, page_size=2)
    workloads = list(loader.iter_workloads())
    assert [w.session_id for w in workloads] == ["a", "b", "c"]
    assert [w.num_turns for w in workloads] == [3, 2, 3]


def test_iter_workloads_start_offset_skips_rows(fake_fetcher):
    loader = WorkloadLoader(fetcher=fake_fetcher, page_size=100)
    # sess-a has 3 turns (rows 0-2); starting at row 3 begins at sess-b
    workloads = list(loader.iter_workloads(start=3))
    assert [w.session_id for w in workloads] == ["sess-b", "sess-c"]


def test_load_first(fake_fetcher):
    loader = WorkloadLoader(fetcher=fake_fetcher, page_size=100)
    first = loader.load_first()
    assert first.session_id == "sess-a"
    assert first.num_turns == 3


# --- load_session_at ------------------------------------------------------------


@pytest.mark.parametrize(
    "offset,expected_sid,expected_turns",
    [
        (0, "sess-a", 3),
        (1, "sess-a", 3),
        (2, "sess-a", 3),
        (3, "sess-b", 1),
        (4, "sess-c", 4),
        (5, "sess-c", 4),
        (6, "sess-c", 4),
        (7, "sess-c", 4),
    ],
)
def test_load_session_at_full_window(fake_fetcher, offset, expected_sid, expected_turns):
    loader = WorkloadLoader(fetcher=fake_fetcher, page_size=100)
    wl = loader.load_session_at(offset)
    assert wl.session_id == expected_sid
    assert wl.num_turns == expected_turns


@pytest.mark.parametrize("offset", [0, 1, 2, 3, 4, 5, 6, 7])
def test_load_session_at_small_pages(fake_dataset, offset):
    # tiny page size stresses left/right expansion across many fetches
    fetcher = FakeRowFetcher(fake_dataset)
    loader = WorkloadLoader(fetcher=fetcher, page_size=1)
    wl = loader.load_session_at(offset)
    expected_sid = fake_dataset[offset]["session_id"]
    expected_turns = sum(1 for r in fake_dataset if r["session_id"] == expected_sid)
    assert wl.session_id == expected_sid
    assert wl.num_turns == expected_turns


def test_load_session_at_middle_of_long_session():
    rows = make_session_rows("long", "m", num_turns=10)
    fetcher = FakeRowFetcher(rows)
    loader = WorkloadLoader(fetcher=fetcher, page_size=3)
    wl = loader.load_session_at(5)
    assert wl.session_id == "long"
    assert wl.num_turns == 10
    # turns reindexed from 0 regardless of absolute offset
    assert [t.index for t in wl] == list(range(10))


def test_load_session_at_out_of_range(fake_fetcher):
    loader = WorkloadLoader(fetcher=fake_fetcher, page_size=100)
    with pytest.raises(IndexError):
        loader.load_session_at(-1)
    with pytest.raises(IndexError):
        loader.load_session_at(999)


def test_loaded_workload_has_valid_prefix_growth(fake_fetcher):
    loader = WorkloadLoader(fetcher=fake_fetcher, page_size=2)
    for wl in loader.iter_workloads():
        wl.validate_prefix_growth()


# --- HttpRowFetcher (no network) ------------------------------------------------


class _StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _StubSession:
    def __init__(self, payload):
        self.payload = payload
        self.last_url = None
        self.last_params = None

    def get(self, url, params=None, timeout=None):
        self.last_url = url
        self.last_params = params
        return _StubResponse(self.payload)


def test_http_row_fetcher_builds_request_and_parses():
    payload = {"rows": [], "num_rows_total": 42}
    session = _StubSession(payload)
    fetcher = HttpRowFetcher(
        dataset="ds", config="cfg", split="train", base_url="https://x/", session=session
    )
    result = fetcher(offset=5, length=10)
    assert result == payload
    assert session.last_url == "https://x/rows"
    assert session.last_params == {
        "dataset": "ds",
        "config": "cfg",
        "split": "train",
        "offset": 5,
        "length": 10,
    }


def test_http_row_fetcher_works_as_loader_backend():
    rows = make_session_rows("s", "m", num_turns=2)
    payload = {
        "rows": [{"row_idx": i, "row": r} for i, r in enumerate(rows)],
        "num_rows_total": len(rows),
    }
    fetcher = HttpRowFetcher(session=_StubSession(payload))
    loader = WorkloadLoader(fetcher=fetcher, page_size=100)
    wl = loader.load_first()
    assert wl.session_id == "s"
    assert wl.num_turns == 2
