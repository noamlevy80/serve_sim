"""Flask app for the serve_sim visualization tool.

The app is a thin shell: it loads a run's ``viz.json``, derives the view model
with :func:`serve_sim.viz.graphs.build_view_model`, and exposes it at
``/api/view-model`` for the single-page renderer served at ``/``. All derivation
is done in Python; the page only draws the returned structure.

The run passed to :func:`create_app` is the default selection, but every sibling
run under the same ``Outputs/`` directory is discoverable via ``/api/runs`` and
selectable through ``/api/view-model?run=<name>``, so the page can offer a run
picker. View models are derived lazily and cached (a finished run is immutable).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from flask import Flask, jsonify, render_template, request

from .graphs import build_view_model


def _load_payload(run_dir: Path) -> Mapping[str, Any]:
    viz_path = run_dir / "viz.json"
    if not viz_path.is_file():
        raise FileNotFoundError(
            f"no viz.json under {run_dir} -- run the simulator to produce one")
    with open(viz_path, encoding="utf-8") as handle:
        return json.load(handle)


def create_app(run_dir: str | Path) -> Flask:
    """Build the Flask app serving the run under ``run_dir`` (the default run).

    Sibling runs under the same parent directory are also selectable.
    """

    run_dir = Path(run_dir).resolve()
    outputs_root = run_dir.parent
    default_run = run_dir.name
    app = Flask(__name__)
    cache: dict[str, Mapping[str, Any]] = {}

    def _available_runs() -> list[str]:
        """Run directory names that have a ``viz.json``, newest first."""

        runs = [p for p in outputs_root.iterdir()
                if p.is_dir() and (p / "viz.json").is_file()]
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        names = [p.name for p in runs]
        if default_run not in names:  # an explicit dir outside Outputs/
            names.insert(0, default_run)
        return names

    def _dir_for(name: str) -> Path:
        return run_dir if name == default_run else outputs_root / name

    def _view_model_for(name: str) -> Mapping[str, Any]:
        if name not in cache:
            vm = build_view_model(_load_payload(_dir_for(name)))
            vm["run_dir"] = str(_dir_for(name))
            cache[name] = vm
        return cache[name]

    @app.get("/")
    def index() -> str:
        return render_template("index.html", run_id=_view_model_for(default_run)["run_id"])

    @app.get("/api/runs")
    def api_runs():
        return jsonify({"runs": _available_runs(), "current": default_run})

    @app.get("/api/view-model")
    def api_view_model():
        # Only known run names are served -- this also blocks path traversal.
        name = request.args.get("run", default_run)
        if name not in _available_runs():
            return jsonify({"error": f"unknown run {name!r}"}), 404
        return jsonify(_view_model_for(name))

    _view_model_for(default_run)  # fail fast if the default run has no viz.json
    return app

