"""Visualization tool for serve_sim runs.

A Flask web app with its own entry point (``run_viz.py``) that renders a run's
``viz.json`` payload. All derivation lives in :mod:`serve_sim.viz.graphs`
(:func:`build_view_model`), which turns the backend payload into a declarative
*view model* -- a list of summary tables and timeline graph descriptors. The
browser is a pure renderer of that view model, so the visualization can be
debugged entirely from its textual (JSON) description.
"""

from __future__ import annotations

from .graphs import build_view_model, eng_format

__all__ = ["build_view_model", "eng_format"]
