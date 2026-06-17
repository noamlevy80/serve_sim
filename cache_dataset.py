"""Download the source dataset into a local cache for offline, reproducible runs.

Populates ``./Dataset/`` (by default) with the dataset rows so that
``run_sim.py`` does not need to hit the live Hugging Face datasets-server on
every run (avoiding rate limits / network flakiness). Commit the cache, or
re-run this script on another machine to reproduce it.

Usage::

    python cache_dataset.py                 # full default dataset into ./Dataset
    python cache_dataset.py --max-rows 500  # cache only the first 500 rows
    python cache_dataset.py --cache-dir Dataset --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "Src"))

from serve_sim.dataset import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_CONFIG,
    DEFAULT_DATASET,
    DEFAULT_SPLIT,
    download_dataset,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cache_dataset",
        description="Download the source dataset into a local cache directory.",
    )
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                        help="Cache root directory (default: Dataset).")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Hugging Face dataset id.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Dataset config.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split.")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cache at most this many rows (default: the whole split).")
    parser.add_argument("--request-pause", type=float, default=0.2,
                        help="Seconds to wait between page requests (politeness).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download even if a cache already exists.")
    args = parser.parse_args(argv)

    out_dir = download_dataset(
        args.cache_dir,
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        max_rows=args.max_rows,
        request_pause=args.request_pause,
        overwrite=args.overwrite,
    )
    print(f"Cached dataset to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
