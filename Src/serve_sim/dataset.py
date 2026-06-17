"""Download multi-turn workloads from the source dataset.

The workloads come from the ``sammshen/lmcache-agentic-traces`` dataset, served
through the Hugging Face *datasets-server* REST API. Each row is one turn; all
rows of a session are stored contiguously and ordered by turn. We therefore
group contiguous runs of equal ``session_id`` into workloads.

Network access is isolated behind the :class:`RowFetcher` protocol so the
grouping/paging logic can be unit-tested with an in-memory fake. The same
protocol lets a run read from a *local cache* (see :func:`download_dataset` and
:class:`LocalRowFetcher`) instead of the live API, so runs are reproducible and
do not depend on network availability or rate limits.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

from .workload import Workload, build_workload_from_rows

DEFAULT_DATASET = "sammshen/lmcache-agentic-traces"
DEFAULT_CONFIG = "default"
DEFAULT_SPLIT = "train"
DEFAULT_BASE_URL = "https://datasets-server.huggingface.co"
DEFAULT_CACHE_DIR = "Dataset"



class RowFetcher(Protocol):
    """Fetches a contiguous block of dataset rows.

    Implementations return a mapping with at least:
        ``rows``: list of ``{"row": {...}}`` items starting at ``offset`` and in
            dataset order (item ``i`` is absolute index ``offset + i``).
        ``num_rows_total``: total number of rows in the split.
    """

    def __call__(self, offset: int, length: int) -> Mapping[str, Any]:
        ...


class HttpRowFetcher:
    """A :class:`RowFetcher` backed by the datasets-server ``/rows`` endpoint."""

    def __init__(
        self,
        dataset: str = DEFAULT_DATASET,
        config: str = DEFAULT_CONFIG,
        split: str = DEFAULT_SPLIT,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        session: Any | None = None,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.split = split
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session

    def _get_session(self):
        if self._session is None:
            import requests  # imported lazily so offline tests need no network deps

            self._session = requests.Session()
        return self._session

    def __call__(self, offset: int, length: int) -> Mapping[str, Any]:
        params = {
            "dataset": self.dataset,
            "config": self.config,
            "split": self.split,
            "offset": offset,
            "length": length,
        }
        response = self._get_session().get(
            f"{self.base_url}/rows", params=params, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()


def _dataset_slug(dataset: str) -> str:
    """Filesystem-safe directory name for a dataset id (``org/name``)."""

    return dataset.replace("/", "__")


def cache_subdir(
    cache_dir: str | Path,
    dataset: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
) -> Path:
    """Directory under ``cache_dir`` holding one cached dataset/config/split."""

    return Path(cache_dir) / _dataset_slug(dataset) / config / split


class LocalRowFetcher:
    """A :class:`RowFetcher` backed by a local cache written by :func:`download_dataset`.

    The cache is a directory (see :func:`cache_subdir`) containing ``rows.jsonl``
    (one JSON row per line, in dataset order) and ``meta.json`` (with
    ``num_rows_total``). Rows are loaded lazily on first access and kept in
    memory, mirroring the page shape returned by the HTTP fetcher.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        dataset: str = DEFAULT_DATASET,
        config: str = DEFAULT_CONFIG,
        split: str = DEFAULT_SPLIT,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.split = split
        self.dir = cache_subdir(cache_dir, dataset, config, split)
        self._rows: list[Mapping[str, Any]] | None = None
        self._total: int | None = None

    @property
    def rows_path(self) -> Path:
        return self.dir / "rows.jsonl"

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    def exists(self) -> bool:
        return self.rows_path.exists() and self.meta_path.exists()

    def _load(self) -> list[Mapping[str, Any]]:
        if self._rows is None:
            if not self.exists():
                raise FileNotFoundError(
                    f"no cached dataset at {self.dir}; run download_dataset (e.g. "
                    f"`python cache_dataset.py`) to populate it"
                )
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self._total = int(meta["num_rows_total"])
            with open(self.rows_path, "r", encoding="utf-8") as handle:
                self._rows = [json.loads(line) for line in handle if line.strip()]
        return self._rows

    def __call__(self, offset: int, length: int) -> Mapping[str, Any]:
        rows = self._load()
        window = rows[offset : offset + length]
        return {
            "rows": [
                {"row_idx": offset + i, "row": row} for i, row in enumerate(window)
            ],
            "num_rows_total": self._total,
        }


def _fetch_with_retry(
    fetcher: RowFetcher,
    offset: int,
    length: int,
    *,
    max_retries: int,
    backoff: float,
) -> Mapping[str, Any]:
    """Call ``fetcher`` with exponential backoff on transient HTTP errors (429/5xx)."""

    attempt = 0
    while True:
        try:
            return fetcher(offset, length)
        except Exception as exc:  # noqa: BLE001 - inspect status to decide retry
            status = getattr(getattr(exc, "response", None), "status_code", None)
            transient = status == 429 or (status is not None and 500 <= status < 600)
            if not transient or attempt >= max_retries:
                raise
            wait = backoff * (2 ** attempt)
            time.sleep(wait)
            attempt += 1


def download_dataset(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    dataset: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
    fetcher: RowFetcher | None = None,
    page_size: int = 100,
    max_rows: int | None = None,
    max_retries: int = 6,
    backoff: float = 2.0,
    request_pause: float = 0.0,
    overwrite: bool = False,
) -> Path:
    """Download dataset rows into a local cache for offline, reproducible runs.

    Pages through ``fetcher`` (defaulting to the live :class:`HttpRowFetcher`),
    with exponential backoff on rate-limit/server errors, and writes
    ``rows.jsonl`` + ``meta.json`` under :func:`cache_subdir`. ``max_rows`` caps
    how many rows are stored (the cache then reports that many as the total, so
    sampling stays inside the cached range); ``None`` downloads the whole split.

    Returns the directory the cache was written to. If a cache already exists and
    ``overwrite`` is False, the existing directory is returned without fetching.
    """

    out_dir = cache_subdir(cache_dir, dataset, config, split)
    rows_path = out_dir / "rows.jsonl"
    meta_path = out_dir / "meta.json"
    if rows_path.exists() and meta_path.exists() and not overwrite:
        return out_dir

    if fetcher is None:
        fetcher = HttpRowFetcher(dataset=dataset, config=config, split=split)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[Mapping[str, Any]] = []
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        length = page_size
        if max_rows is not None:
            length = min(length, max_rows - len(rows))
            if length <= 0:
                break
        response = _fetch_with_retry(
            fetcher, offset, length, max_retries=max_retries, backoff=backoff
        )
        if total is None:
            total = int(response["num_rows_total"])
        block = [item["row"] for item in response.get("rows", [])]
        if not block:
            break
        rows.extend(block)
        offset += len(block)
        if request_pause and (max_rows is None or len(rows) < max_rows):
            time.sleep(request_pause)

    num_total = len(rows) if max_rows is not None else (total or len(rows))
    with open(rows_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "num_rows_total": num_total,
        "num_rows_cached": len(rows),
        "source_num_rows_total": total,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out_dir


class WorkloadLoader:
    """Downloads workloads (full multi-turn sessions) from the dataset.

    Args:
        fetcher: Row fetcher. Defaults to :class:`HttpRowFetcher` against the
            public dataset.
        page_size: Number of rows requested per network call (the datasets-server
            caps a page at 100 rows).
    """

    MAX_PAGE_SIZE = 100

    def __init__(self, fetcher: RowFetcher | None = None, page_size: int = 100) -> None:
        if page_size < 1 or page_size > self.MAX_PAGE_SIZE:
            raise ValueError(
                f"page_size must be between 1 and {self.MAX_PAGE_SIZE}, got {page_size}"
            )
        self.fetcher = fetcher if fetcher is not None else HttpRowFetcher()
        self.page_size = page_size

    def num_rows(self) -> int:
        """Total number of turn-rows in the dataset split."""

        response = self.fetcher(0, 1)
        return int(response["num_rows_total"])

    def _rows_in_block(self, response: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        return [item["row"] for item in response.get("rows", [])]

    def _iter_rows(self, start: int = 0) -> Iterator[tuple[int, Mapping[str, Any]]]:
        """Yield ``(absolute_index, row)`` pairs from ``start`` to the end."""

        offset = start
        total: int | None = None
        while total is None or offset < total:
            response = self.fetcher(offset, self.page_size)
            if total is None:
                total = int(response["num_rows_total"])
            rows = self._rows_in_block(response)
            if not rows:
                break
            for i, row in enumerate(rows):
                yield offset + i, row
            offset += len(rows)

    def iter_workloads(self, start: int = 0) -> Iterator[Workload]:
        """Yield workloads sequentially by grouping contiguous session ids."""

        current_rows: list[Mapping[str, Any]] = []
        current_sid: str | None = None
        for _, row in self._iter_rows(start):
            sid = row["session_id"]
            if sid != current_sid:
                if current_rows:
                    yield build_workload_from_rows(current_rows)
                current_rows = []
                current_sid = sid
            current_rows.append(row)
        if current_rows:
            yield build_workload_from_rows(current_rows)

    def load_first(self) -> Workload:
        """Load the first workload in the dataset."""

        return next(self.iter_workloads())

    def load_session_at(self, offset: int) -> Workload:
        """Load the full workload whose session contains row ``offset``.

        Expands left and right from ``offset`` to cover the entire contiguous
        run of rows sharing the same ``session_id``.
        """

        total = self.num_rows()
        if offset < 0 or offset >= total:
            raise IndexError(f"offset {offset} out of range [0, {total})")

        anchor = self.fetcher(offset, 1)
        anchor_rows = self._rows_in_block(anchor)
        if not anchor_rows:
            raise IndexError(f"no row at offset {offset}")
        sid = anchor_rows[0]["session_id"]

        start = self._expand_left(offset, sid)
        end = self._expand_right(offset, sid, total)

        rows = [row for _, row in self._collect_range(start, end)]
        return build_workload_from_rows(rows)

    def _expand_left(self, offset: int, sid: str) -> int:
        start = offset
        while start > 0:
            block_start = max(0, start - self.page_size)
            length = start - block_start
            rows = self._rows_in_block(self.fetcher(block_start, length))
            new_start = start
            matched_to_block_start = False
            for i in range(len(rows) - 1, -1, -1):
                if rows[i]["session_id"] == sid:
                    new_start = block_start + i
                    matched_to_block_start = i == 0
                else:
                    break
            if new_start == start:
                break  # the immediately preceding row is a different session
            start = new_start
            if not matched_to_block_start:
                break  # found the boundary inside this block
        return start

    def _expand_right(self, offset: int, sid: str, total: int) -> int:
        end = offset  # inclusive
        while end < total - 1:
            block_start = end + 1
            length = min(self.page_size, total - block_start)
            rows = self._rows_in_block(self.fetcher(block_start, length))
            new_end = end
            matched_to_block_end = False
            for i, row in enumerate(rows):
                if row["session_id"] == sid:
                    new_end = block_start + i
                    matched_to_block_end = i == len(rows) - 1
                else:
                    break
            if new_end == end:
                break  # the immediately following row is a different session
            end = new_end
            if not matched_to_block_end:
                break  # found the boundary inside this block
        return end

    def _collect_range(
        self, start: int, end: int
    ) -> Iterator[tuple[int, Mapping[str, Any]]]:
        """Yield ``(absolute_index, row)`` for the inclusive range ``[start, end]``."""

        offset = start
        while offset <= end:
            length = min(self.page_size, end - offset + 1)
            rows = self._rows_in_block(self.fetcher(offset, length))
            if not rows:
                break
            for i, row in enumerate(rows):
                yield offset + i, row
            offset += len(rows)
