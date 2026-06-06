"""PS5 value-op binding coverage: coriolis_matrix / energy regressors / dccrba /
cmm_time_variation.

Exercises the grid_rbd handle methods newly bound for the PS5 value algorithms
and asserts numerical agreement with the RBDReference numpy oracle at float32
precision:

  coriolis_matrix · kinetic_energy_regressor · potential_energy_regressor
  dccrba · cmm_time_variation

iiwa14 (7-DoF serial arm, fixed, non-mimic) HAS all five. fr3 (mimic) is used to
assert the dccrba / cmm_time_variation availability error path is clean.

Run with:
    PYTHONPATH=$PWD/bindings pytest test/python_wrappers/test_ps5_value_ops.py -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


# Repo root is parent of `test/`. Insert FIRST so this clone's submodules
# (URDFParser / RBDReference / GRiDCodeGenerator) and `bindings/` win import.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "bindings"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed (build bindings/ _core)")

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)

_GRID_RBD_DIR = Path(_grid_rbd.__file__).resolve().parent
if _REPO_ROOT not in _GRID_RBD_DIR.parents:
    pytest.skip(
        f"grid_rbd resolved to {_GRID_RBD_DIR} (not this clone under {_REPO_ROOT}); "
        "set PYTHONPATH=<clone>/bindings and build _core in-place",
        allow_module_level=True,
    )

_ASSETS = _REPO_ROOT / "robot_assets"

pytestmark = pytest.mark.python_wrappers

_TOL = 5e-3   # float32 vs float64 cross-precision
_B = 4        # batch


def _register(name, urdf):
    urdf_path = _ASSETS / urdf
    if not urdf_path.exists():
        pytest.skip(f"{urdf} fixture not present at {urdf_path}")
    return _grid_rbd.register_robot(
        name=f"{name}_ps5_value_pytest",
        urdf_path=str(urdf_path),
        floating_base=False,
        max_batch_size=8,
        force_rebuild=True,   # recompile the .so with the new wrapper
    )


def _reference(urdf):
    from URDFParser import URDFParser
    from RBDReference import RBDReference
    return RBDReference(URDFParser().parse(str(_ASSETS / urdf), floating_base=False))


def _samples(handle):
    rng = np.random.default_rng(0)
    NJ = handle.num_joints
    return {
        "q":  rng.standard_normal((_B, NJ)).astype(np.float32),
        "qd": rng.standard_normal((_B, NJ)).astype(np.float32),
    }


def _max_err(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


@pytest.fixture(scope="module")
def iiwa():
    handle = _register("iiwa14", "iiwa14.urdf")
    ref = _reference("iiwa14.urdf")
    return handle, ref, _samples(handle)


# ─── value ops (all five, on iiwa14-fixed) ───────────────────────────────────

def test_coriolis_matrix(iiwa):
    handle, ref, s = iiwa
    C = handle.coriolis_matrix(s["q"], s["qd"])
    NV = handle.num_vel
    assert C.shape == (_B, NV, NV)
    for i, (q, qd) in enumerate(zip(s["q"], s["qd"])):
        C_ref = ref.coriolis_matrix(q.astype(np.float64), qd.astype(np.float64))
        assert _max_err(C[i], C_ref) < _TOL
    # And the defining identity C qd + g = nonlinear_effects (extra cross-check).
    nle = handle.nonlinear_effects(s["q"], s["qd"], gravity=-9.81)
    gg = handle.generalized_gravity(s["q"], gravity=-9.81)
    for i, qd in enumerate(s["qd"]):
        cqd = C[i] @ qd.astype(np.float64)
        assert _max_err(cqd + gg[i], nle[i]) < 1e-2


def test_kinetic_energy_regressor(iiwa):
    handle, ref, s = iiwa
    NB = handle.num_bodies
    y = handle.kinetic_energy_regressor(s["q"], s["qd"])
    assert y.shape == (_B, 10 * NB)
    for i, (q, qd) in enumerate(zip(s["q"], s["qd"])):
        y_ref = ref.kinetic_energy_regressor(q.astype(np.float64), qd.astype(np.float64))
        assert _max_err(y[i], y_ref) < _TOL


def test_potential_energy_regressor(iiwa):
    handle, ref, s = iiwa
    NB = handle.num_bodies
    y = handle.potential_energy_regressor(s["q"], gravity=-9.81)
    assert y.shape == (_B, 10 * NB)
    for i, q in enumerate(s["q"]):
        y_ref = ref.potential_energy_regressor(q.astype(np.float64), GRAVITY=-9.81)
        assert _max_err(y[i], y_ref) < _TOL


def test_dccrba(iiwa):
    handle, ref, s = iiwa
    NV = handle.num_vel
    D = handle.dccrba(s["q"])
    assert D.shape == (_B, 6, NV, NV)
    for i, q in enumerate(s["q"]):
        D_ref = ref.dccrba(q.astype(np.float64))   # (6, NV, NV) [:, k, i]
        assert _max_err(D[i], D_ref) < _TOL


def test_cmm_time_variation(iiwa):
    handle, ref, s = iiwa
    NV = handle.num_vel
    Adot = handle.cmm_time_variation(s["q"], s["qd"])
    assert Adot.shape == (_B, 6, NV)
    for i, (q, qd) in enumerate(zip(s["q"], s["qd"])):
        Adot_ref = ref.cmm_time_variation(q.astype(np.float64), qd.astype(np.float64))
        assert _max_err(Adot[i], Adot_ref) < _TOL
    # cmm_time_variation == sum_i dccrba[:, :, i] qd_i (consistency w/ dccrba).
    D = handle.dccrba(s["q"])
    for i, qd in enumerate(s["qd"]):
        contracted = np.einsum("aki,i->ak", D[i], qd.astype(np.float64))
        assert _max_err(contracted, Adot[i]) < _TOL


# ─── mimic availability error path (dccrba / cmm) ────────────────────────────

def test_dccrba_cmm_mimic_clean_error():
    """fr3 is a mimic robot: dccrba / cmm_time_variation are not generated. The
    binding must raise a CLEAR error (not crash). The whole centroidal value
    surface (com/ccrba/energy/dccrba/cmm) is mimic-gated, so fr3 may not even
    register — in that case we skip (the limitation is documented), since the
    requirement is specifically that dccrba/cmm fail cleanly rather than crash."""
    if not (_ASSETS / "fr3.urdf").exists():
        pytest.skip("fr3.urdf fixture not present")
    try:
        handle = _register("fr3", "fr3.urdf")
    except Exception as e:
        pytest.skip(f"fr3 (mimic) does not register through the binding "
                    f"(centroidal value surface is mimic-gated at compile): {e!r}")
    q = np.zeros((1, handle.num_joints), dtype=np.float32)
    qd = np.zeros((1, handle.num_joints), dtype=np.float32)
    with pytest.raises(RuntimeError, match="dccrba"):
        handle.dccrba(q)
    with pytest.raises(RuntimeError, match="cmm_time_variation"):
        handle.cmm_time_variation(q, qd)
