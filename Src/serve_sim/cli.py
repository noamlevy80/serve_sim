"""Command-line entry point: run a simulation from a ``config.json``.

Usage::

    python -m serve_sim CONFIG [--output-root DIR] [--run-id ID]
                               [--tokenizer {tiktoken,whitespace}] [--quiet]

Writes the raw outputs under ``<output-root>/<run-id>/``, reports progress as
sequences complete, and prints a short summary of the run.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .report import summarize
from .runner import BuildProgressReporter, ProgressReporter, run_from_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="serve_sim",
        description="Run a serve_sim simulation from a JSON config and write outputs.",
    )
    parser.add_argument("config", help="Path to the run config JSON.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory to write <run_id>/ outputs under (default: Outputs).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Override the run id (default: config value or a timestamp).",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["tiktoken", "whitespace"],
        default=None,
        help="Override the tokenizer named in the config.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-progress sequence/time updates.",
    )
    args = parser.parse_args(argv)

    tokenizer = None
    if args.tokenizer is not None:
        from .runner import _make_tokenizer

        tokenizer = _make_tokenizer(args.tokenizer)

    progress = None if args.quiet else ProgressReporter()
    build_progress = None if args.quiet else BuildProgressReporter()
    result, out_dir = run_from_config(
        args.config,
        output_root=args.output_root,
        run_id=args.run_id,
        tokenizer=tokenizer,
        progress=progress,
        build_progress=build_progress,
    )

    report = summarize(result)
    print(f"Wrote outputs to {out_dir}")
    print(f"  requests={report['num_requests']} batches={report['num_batches']} "
          f"makespan={report['makespan_s']:.6g}s")
    print(f"  throughput={report['throughput_requests_per_s']:.6g} req/s, "
          f"{report['throughput_output_tokens_per_s']:.6g} out-tok/s")
    print(f"  avg TPS={report['tps_tokens_per_s']['mean']:.6g} tok/s, "
          f"avg TTFT={report['ttft_s']['mean']:.6g}s")
    print(f"  latency p50={report['latency_s']['p50']:.6g}s "
          f"p99={report['latency_s']['p99']:.6g}s; "
          f"ttft p50={report['ttft_s']['p50']:.6g}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
