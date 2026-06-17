"""Repo-root launcher for the serve_sim CLI.

Adds ``Src`` to the import path and runs the CLI, so the simulator can be driven
without installing the package::

    python run_sim.py Configs/example.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "Src"))

from serve_sim.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
