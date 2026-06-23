"""Repo-root launcher for the serve_sim visualization tool.

Serves the web UI for one run's outputs (its ``viz.json``). With no path it
picks the most recently modified run directory under ``Outputs/``::

    python run_viz.py                         # latest run under Outputs/
    python run_viz.py Outputs/run-2026...     # a specific run directory
    python run_viz.py Outputs/foo --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "Src"))

from serve_sim.viz.app import create_app  # noqa: E402


def _latest_run(outputs: Path) -> Path | None:
    runs = [p for p in outputs.iterdir() if p.is_dir() and (p / "viz.json").is_file()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="serve_sim visualization tool")
    parser.add_argument("run_dir", nargs="?", default=None,
                        help="run output directory (defaults to the latest under Outputs/)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args(argv)

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        run_dir = _latest_run(Path("Outputs"))
        if run_dir is None:
            parser.error("no run with a viz.json found under Outputs/ -- run the simulator first")
    if not (run_dir / "viz.json").is_file():
        parser.error(f"no viz.json under {run_dir}")

    app = create_app(run_dir)
    print(f"serve_sim visualization: {run_dir}  ->  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
