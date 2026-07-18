"""End-to-end g1 (fixed-base) `plant_step_hessian` smoke via the grid_rbd binding.

The CUDA `plant_step_hessian` codegen carries spill TIERS so big robots fit in
smem. iiwa14 validates SHARED + forced-spill; g1's smem macro confirms it FITS,
but g1 had not been run end-to-end. The cuda_equivalents smoke runner can't
compile g1 (a pre-existing sibling `plant_step_kernel` uses static `__shared__`
that overflows the 48 KB default). The bindings are a separate TU with
all-dynamic-smem kernels + `cudaFuncSetAttribute`, so g1's plant_step_hessian
launches here. This proves g1 numerically matches the numpy oracle through the
binding for euler + semi_implicit_euler.

The oracle is the project RBDReference adapter (build_project_adapter(..,
base_mode='fixed').reference); its `plant_step_hessian` calls `fdsva_so` (slow
pure-Python second order), so we keep the sample count low (B=2).

Output shape: (B, 2*NV, 3*NV, 3*NV).

Run with:
    pytest test/python_wrappers/test_g1_plant_hessian_smoke.py -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from config import robot_urdf

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed (pip install -e bindings)")

_URDF = robot_urdf("g1")
if not _URDF.exists():
    pytest.skip(f"g1 URDF fixture not present at {_URDF}", allow_module_level=True)
if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)


pytestmark = pytest.mark.python_wrappers

# Per-robot magnitude-relative bucket for the g1 float32 2nd-order tensor. The
# Hessian is mostly structural zeros, so a |ref|+eps relative metric blows up on
# float32 roundoff at the zero cells; compare with a magnitude-relative atol
# (atol scaled by the array's max magnitude) plus a small rtol. Mirrors the cuda
# integrator test's big-robot bucket. Do NOT loosen this globally.
_RTOL = 5e-3
_ATOL_SCALE = 5e-3


@pytest.fixture(scope="module")
def handle():
    # g1 fixed-base ~29 DOF; this build is ~20-40 min on first run, then cached.
    return _grid_rbd.register_robot(
        name="g1_plant_hessian_smoke",
        urdf_path=str(_URDF),
        floating_base=False,
        max_batch_size=8,
    )


@pytest.fixture(scope="module")
def ref():
    # Project oracle via the same adapter path the cuda_equivalents tests use.
    from RBDReference.tests import MANIFEST_PATH
    from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
    from RBDReference.equivalents.reference_backend import build_project_adapter

    spec = None
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="fixed"):
        if case["spec"].robot_id == "g1":
            spec = case["spec"]
            break
    if spec is None:
        pytest.skip("g1 not in RBDReference manifest")
    resolved = resolve_robot_spec(spec)
    return build_project_adapter(spec, resolved, base_mode="fixed").reference


@pytest.fixture(scope="module")
def samples(handle):
    # Keep B small: the g1 oracle plant_step_hessian calls fdsva_so (slow pure
    # Python 2nd order); 2 samples is enough to exercise the kernel.
    rng = np.random.default_rng(0)
    NQ, NV = handle.num_joints, handle.num_vel
    B = 2
    q = rng.standard_normal((B, NQ)).astype(np.float32)
    qd = rng.standard_normal((B, NV)).astype(np.float32)
    u = rng.standard_normal((B, NV)).astype(np.float32)
    return {"q": q, "qd": qd, "u": u,
            "x": np.concatenate([q, qd], axis=1).astype(np.float32), "B": B}


@pytest.mark.parametrize("integrator_type", ["euler", "semi_implicit_euler"])
def test_plant_step_hessian(handle, ref, samples, integrator_type, record_property):
    x, u, B = samples["x"], samples["u"], samples["B"]
    NQ, NV = handle.num_joints, handle.num_vel
    dt = 0.01
    H = handle.plant_step_hessian(x, u, dt, integrator_type=integrator_type)
    assert H.shape == (B, 2 * NV, 3 * NV, 3 * NV), H.shape

    max_rel = 0.0
    for b in range(B):
        r = ref.plant_step_hessian(x[b, :NQ], x[b, NQ:], u[b], dt,
                                   integrator_type=integrator_type)
        a = np.asarray(H[b], np.float64)
        rr = np.asarray(r, np.float64)
        scale = max(float(np.abs(rr).max()), 1e-6)
        # Magnitude-relative error for reporting: |a-rr| / (max|rr| + eps).
        max_rel = max(max_rel, float(np.max(np.abs(a - rr)) / scale))
        np.testing.assert_allclose(
            a, rr, rtol=_RTOL, atol=_ATOL_SCALE * scale,
            err_msg=f"g1 plant_step_hessian {integrator_type} sample {b}",
        )
    record_property("max_magnitude_relative_error", max_rel)
    print(f"\n[g1 plant_step_hessian] {integrator_type}: "
          f"max magnitude-relative error = {max_rel:.3e}")
