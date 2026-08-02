"""Staleness gate for test/dynamics_fingerprint.json (the model-alignment table).

The committed fingerprint is a PUBLISHED artifact: downstream consumers (GATO,
MPCGPU, PDDP, external sims, hardware pipelines) pin it to verify their model is
the same robot as GRiD's reference dynamics. This gate regenerates the table
in-process (CPU-only, RBDReference oracle) and fails loudly if the committed
file drifted — a changed robot_assets URDF, changed reference-dynamics
semantics, or a hand-edited table all trip it. Fix = rerun
`tools/gen_dynamics_fingerprint.py` and commit the result (and tell consumers:
a value change means THE MODEL changed).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TABLE = _REPO / "test" / "dynamics_fingerprint.json"


def _load_generator():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gen_dynamics_fingerprint", _REPO / "tools" / "gen_dynamics_fingerprint.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dynamics_fingerprint_current():
    assert _TABLE.exists(), "test/dynamics_fingerprint.json missing — run tools/gen_dynamics_fingerprint.py"
    committed = json.loads(_TABLE.read_text())
    gen = _load_generator()
    fresh, failures = gen.generate_table()

    assert committed["schema"] == fresh["schema"] == 1
    assert committed["semantics"] == fresh["semantics"], "semantics metadata drifted — regenerate"
    assert sorted(committed["plants"]) == sorted(fresh["plants"]), (
        f"robot set drifted (committed {sorted(committed['plants'])} vs fresh "
        f"{sorted(fresh['plants'])}; exclusions now: {failures}) — regenerate")

    for rid, fresh_plant in fresh["plants"].items():
        plant = committed["plants"][rid]
        assert plant["urdf_sha256"] == fresh_plant["urdf_sha256"], (
            f"{rid}: URDF changed on disk but the fingerprint was not regenerated — "
            "rerun tools/gen_dynamics_fingerprint.py and notify consumers")
        assert plant["nq"] == fresh_plant["nq"] and plant["nv"] == fresh_plant["nv"]
        assert [p["name"] for p in plant["probes"]] == [p["name"] for p in fresh_plant["probes"]]
        for got, want in zip(plant["probes"], fresh_plant["probes"]):
            for key in ("q", "qd", "u"):
                np.testing.assert_array_equal(got[key], want[key],
                                              err_msg=f"{rid}/{got['name']}: probe state drifted")
            # tight-but-not-bitwise: tolerate BLAS/numpy build noise, catch model changes
            np.testing.assert_allclose(
                got["qdd"], want["qdd"], rtol=1e-9, atol=1e-12,
                err_msg=(f"{rid}/{got['name']}: reference qdd drifted — the MODEL or the "
                         "reference dynamics changed; regenerate + notify consumers"))


def test_dynamics_fingerprint_guidance_math():
    """The published guidance divides consumer qdd by table qdd per joint on the
    inertia probes; make sure no committed inertia-probe response is exactly zero
    on its OWN driven joint (a zero would make the ratio check vacuous there)."""
    committed = json.loads(_TABLE.read_text())
    for rid, plant in committed["plants"].items():
        for j, probe in enumerate(p for p in plant["probes"] if p["name"].startswith("inertia_")):
            assert probe["qdd"][j] != pytest.approx(0.0, abs=1e-12), (
                f"{rid}: inertia_{j} has ~zero self-response; probe design assumption broken")
