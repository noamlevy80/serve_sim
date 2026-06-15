"""Download multi-turn workloads from the source dataset.

The workloads come from the ``sammshen/lmcache-agentic-traces`` dataset, served
through the Hugging Face *datasets-server* REST API. Each row is one turn; all
rows of a session are stored contiguously and ordered by turn. We therefore
group contiguous runs of equal ``session_id`` into workloads.

Network access is isolated behind the :class:`RowFetcher` protocol so the
grouping/paging logic can be unit-tested with an in-memory fake.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping, Protocol

from .workload import Workload, build_workload_from_rows

DEFAULT_DATASET = "sammshen/lmcache-agentic-traces"
DEFAULT_CONFIG = "default"
DEFAULT_SPLIT = "train"
DEFAULT_BASE_URL = "https://datasets-server.huggingface.co"


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
