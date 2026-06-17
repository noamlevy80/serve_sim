"""Local dataset cache: download into ./Dataset and serve rows offline.

``download_dataset`` pages through a row fetcher (the live HTTP one by default)
and writes a local cache; ``LocalRowFetcher`` serves that cache with the same
page shape as the HTTP fetcher, so a :class:`WorkloadLoader` works offline and
reproducibly. These tests drive the whole round-trip with the in-memory
``FakeRowFetcher`` so no network is touched.
"""

from __future__ import annotations

import json

import pytest

from serve_sim.dataset import (
    LocalRowFetcher,
    WorkloadLoader,
    cache_subdir,
    download_dataset,
)


def test_download_then_local_fetcher_round_trips(tmp_path, fake_fetcher, fake_dataset):
    out_dir = download_dataset(tmp_path, fetcher=fake_fetcher)

    assert out_dir == cache_subdir(tmp_path)
    assert (out_dir / "rows.jsonl").exists()
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["num_rows_total"] == len(fake_dataset)
    assert meta["num_rows_cached"] == len(fake_dataset)

    local = LocalRowFetcher(tmp_path)
    assert local.exists()
    page = local(0, len(fake_dataset))
    assert page["num_rows_total"] == len(fake_dataset)
    rows = [item["row"] for item in page["rows"]]
    assert [r["session_id"] for r in rows] == [r["session_id"] for r in fake_dataset]


def test_local_fetcher_matches_fake_paging(tmp_path, fake_fetcher, fake_dataset):
    download_dataset(tmp_path, fetcher=fake_fetcher)
    local = LocalRowFetcher(tmp_path)

    for offset, length in [(0, 1), (2, 3), (1, 100), (len(fake_dataset), 5)]:
        got = local(offset, length)
        want = fake_fetcher(offset, length)
        assert [i["row"] for i in got["rows"]] == [i["row"] for i in want["rows"]]
        assert got["num_rows_total"] == want["num_rows_total"]


def test_loader_over_local_cache_offline(tmp_path, fake_fetcher):
    download_dataset(tmp_path, fetcher=fake_fetcher)
    loader = WorkloadLoader(LocalRowFetcher(tmp_path))

    workloads = list(loader.iter_workloads())
    # fake_dataset has three contiguous sessions.
    assert len(workloads) == 3


def test_max_rows_caps_cache_and_total(tmp_path, fake_fetcher, fake_dataset):
    out_dir = download_dataset(tmp_path, fetcher=fake_fetcher, max_rows=3, page_size=2)

    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["num_rows_cached"] == 3
    assert meta["num_rows_total"] == 3  # cache is self-consistent within its range

    local = LocalRowFetcher(tmp_path)
    page = local(0, 100)
    assert len(page["rows"]) == 3
    assert page["num_rows_total"] == 3


def test_download_skips_when_cache_present(tmp_path, fake_fetcher):
    download_dataset(tmp_path, fetcher=fake_fetcher)
    calls_before = len(fake_fetcher.calls)

    download_dataset(tmp_path, fetcher=fake_fetcher)  # no overwrite
    assert len(fake_fetcher.calls) == calls_before  # fetcher not called again

    download_dataset(tmp_path, fetcher=fake_fetcher, overwrite=True)
    assert len(fake_fetcher.calls) > calls_before


def test_fetch_retries_on_transient_error(tmp_path, fake_dataset):
    class FlakyFetcher:
        def __init__(self, rows):
            self.rows = rows
            self.attempts = 0

        def __call__(self, offset, length):
            self.attempts += 1
            if self.attempts == 1:
                err = Exception("rate limited")
                err.response = type("R", (), {"status_code": 429})()
                raise err
            window = self.rows[offset : offset + length]
            return {
                "rows": [{"row_idx": offset + i, "row": r} for i, r in enumerate(window)],
                "num_rows_total": len(self.rows),
            }

    flaky = FlakyFetcher(fake_dataset)
    out_dir = download_dataset(tmp_path, fetcher=flaky, backoff=0.0)

    assert flaky.attempts >= 2
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["num_rows_cached"] == len(fake_dataset)


def test_local_fetcher_missing_cache_raises(tmp_path):
    local = LocalRowFetcher(tmp_path)
    assert not local.exists()
    with pytest.raises(FileNotFoundError):
        local(0, 1)
