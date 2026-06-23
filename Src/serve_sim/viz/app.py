"""Flask app for the serve_sim visualization tool.

The app is a thin shell: it loads a run's ``viz.json``, derives the view model
with :func:`serve_sim.viz.graphs.build_view_model`, and exposes it at
``/api/view-model`` for the single-page renderer served at ``/``. All derivation
is done in Python; the page only draws the returned structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from flask import Flask, jsonify, render_template

from .graphs import build_view_model


def _load_payload(run_dir: Path) -> Mapping[str, Any]:
    viz_path = run_dir / "viz.json"
    if not viz_path.is_file():
        raise FileNotFoundError(
            f"no viz.json under {run_dir} -- run the simulator to produce one")
    with open(viz_path, encoding="utf-8") as handle:
        return json.load(handle)


def create_app(run_dir: str | Path) -> Flask:
    """Build the Flask app serving the run under ``run_dir``."""

    run_dir = Path(run_dir)
    app = Flask(__name__)
    # Derive once at startup: the payload is immutable for a finished run.
    view_model = build_view_model(_load_payload(run_dir))
    view_model["run_dir"] = str(run_dir)

    @app.get("/")
    def index() -> str:
        return render_template("index.html", run_id=view_model["run_id"])

    @app.get("/api/view-model")
    def api_view_model():
        return jsonify(view_model)

    return app
