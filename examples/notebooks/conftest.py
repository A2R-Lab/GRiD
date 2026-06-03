"""Collection-level skip guards for the example-notebook smoke tests.

The notebooks under ``examples/notebooks/`` double as CI smoke tests: they are executed
end-to-end via ``pytest --nbval-lax examples/notebooks/`` and validate numbers in-cell
(asserts against ``RBDReference`` / closed-form checks), so a green run == passing
docs. They require a real CUDA GPU + ``nvcc`` on PATH + ``grid_rbd`` installed —
the same triad as the python-wrapper smokes. nbval is required to collect ``.ipynb``
files as tests; if it's absent the notebooks are skipped (so a plain ``pytest``
without the dev extras doesn't error).

This conftest applies the ``notebooks`` marker to every collected ``.ipynb`` and
skips the whole directory when the preconditions aren't met. Per-notebook optional
deps (torch) are guarded inside the notebook's own preamble cell.
"""
from __future__ import annotations

import importlib.util
import shutil

import pytest


def _missing():
    reasons = []
    if importlib.util.find_spec("grid_rbd") is None:
        reasons.append("grid_rbd not installed (pip install bindings/)")
    if importlib.util.find_spec("nbval") is None:
        reasons.append("nbval not installed (pip install -r requirements-dev.txt)")
    if shutil.which("nvcc") is None:
        reasons.append("nvcc not on PATH")
    return reasons


_SKIP_REASONS = _missing()


def pytest_collectstart(collector):
    if _SKIP_REASONS and str(getattr(collector, "fspath", "")).endswith(".ipynb"):
        pytest.skip("notebook smoke tests require: " + "; ".join(_SKIP_REASONS))


def pytest_collection_modifyitems(config, items):
    mark = pytest.mark.notebooks
    for item in items:
        if str(getattr(item, "fspath", "")).endswith(".ipynb"):
            item.add_marker(mark)
