"""Validation for the runtime_transform feature (mutable joint-frame <origin>).

runtime_transform lets a caller change link origin (xyz+rpy) parameters at
runtime WITHOUT recompiling the generated CUDA, by mirroring the runtime_inertia
path: the raw [x,y,z,r,p,y] per joint live in a mutable d_transform_params table,
each joint's constant Xfixed 6x6 is rebuilt on-device once per launch, and the
hot sin/cos(q) loop reads that scratch instead of inline origin literals.

Three checks (mirroring the runtime_inertia suite's structure):

  CHECK1  default == baked      runtime_transform(original URDF params) ~= plain baked .so.
          Tolerance is RELATIVE (~float32): the on-device sincos(rpy) rebuild does
          NOT reproduce the baked sympy-FOLDED origin constants bit-for-bit, but the
          relative error is float32 noise (~1e-7). Not bit-identical; float-identical.
  CHECK2  perturb == recodegen  set_transform_params(perturbed) == a freshly codegen'd
          robot whose URDF origins ARE those perturbed values (proves mutation == recompile).
  CHECK3  thread-invariance     bit-identical across {1,32,256} threads (single-block core).

Requires a CUDA GPU + the grid_rbd binding build toolchain (nvcc). iiwa14 fixed base.

Run with:
    pytest test/python_wrappers/test_runtime_transform.py -m python_wrappers -v
"""
from __future__ import annotations

import contextlib
import io
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest


# Repo root is parent of `test/`; add it (URDFParser/config) and bindings/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "bindings"))
from config import robot_urdf, ROBOT_ASSETS_DIR  # noqa: E402


# ─── skip preconditions ─────────────────────────────────────────────────────

import shutil  # noqa: E402

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it",
                allow_module_level=True)

_IIWA = robot_urdf("iiwa14")
if not _IIWA.exists():
    pytest.skip(f"iiwa14 URDF not present at {_IIWA}", allow_module_level=True)


pytestmark = pytest.mark.python_wrappers

# Tolerances (see the per-check docstrings above).
CHECK1_REL_TOL = 1e-4   # float32 rebuild-vs-folded-constant noise (observed ~1e-7)
CHECK2_REL_TOL = 5e-3
CHECK3_ABS_TOL = 1e-5


# ─── helpers ────────────────────────────────────────────────────────────────


def _reg(name, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return _grid_rbd.register_robot(name=name, urdf_path=str(_IIWA),
                                        floating_base=False, max_batch_size=8, **kw)


def _relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))


# ─── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rt():
    return _reg("iiwa_rt_xform_pytest", runtime_transform=True, force_rebuild=True)


@pytest.fixture(scope="module")
def baked():
    return _reg("iiwa_baked_xform_pytest")


@pytest.fixture(scope="module")
def samples(rt):
    nj = rt.num_joints
    rng = np.random.default_rng(1)
    return {
        "q":  rng.standard_normal((4, nj)).astype(np.float32),
        "qd": rng.standard_normal((4, nj)).astype(np.float32),
        "u":  rng.standard_normal((4, nj)).astype(np.float32),
    }


@pytest.fixture(scope="module")
def perturbed(rt):
    """Deterministic perturbation of the baked origin params (xyz shift + rpy tweak)."""
    nj = rt.num_joints
    baked_params = rt.transform_params.astype(np.float64)
    rng = np.random.default_rng(7)
    p = baked_params.copy()
    p[:, 0:3] += 0.03 * rng.standard_normal((nj, 3))   # xyz
    p[:, 3:6] += 0.05 * rng.standard_normal((nj, 3))   # rpy
    return p


@pytest.fixture(scope="module")
def recodegen(rt, perturbed):
    """A robot codegen'd from a URDF whose joint <origin>s ARE the perturbed params."""
    with contextlib.redirect_stdout(io.StringIO()):
        robot = __import__("URDFParser", fromlist=["URDFParser"]).URDFParser().parse(
            str(_IIWA), floating_base=False)
    nj = rt.num_joints
    names = [robot.get_joint_by_id(j).get_name() for j in range(robot.get_num_joints())]
    pert_by_name = {names[j]: perturbed[j] for j in range(nj)}

    tree = ET.parse(str(_IIWA))
    for joint in tree.getroot().findall("joint"):
        if joint.get("name") in pert_by_name:
            x, y, z, r, p, yw = pert_by_name[joint.get("name")]
            origin = joint.find("origin")
            if origin is None:  # explicit None check: an attr-only <origin> is falsy
                origin = ET.SubElement(joint, "origin")
            origin.set("xyz", f"{x} {y} {z}")
            origin.set("rpy", f"{r} {p} {yw}")
    pert_urdf = str(ROBOT_ASSETS_DIR / ".iiwa14_perturbed_rt_pytest.urdf")
    tree.write(pert_urdf)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            handle = _grid_rbd.register_robot(
                name="iiwa_recodegen_pert_pytest", urdf_path=pert_urdf,
                floating_base=False, max_batch_size=8, force_rebuild=True)
        yield handle
    finally:
        Path(pert_urdf).unlink(missing_ok=True)


# ─── tests ───────────────────────────────────────────────────────────────────


def test_runtime_transform_metadata(rt):
    assert rt.runtime_transform is True
    # transform_params is (num_joints, 6): [x, y, z, r, p, y] per joint origin.
    assert rt.transform_params.shape == (rt.num_joints, 6)


def test_default_params_match_baked(rt, baked, samples):
    """CHECK1: runtime_transform fed its ORIGINAL baked params must reproduce the
    plain baked .so to float32 relative tolerance (the on-device sincos(rpy)
    rebuild is not bit-identical to the sympy-folded origin constants, but the
    relative error is float32 noise ~1e-7)."""
    rt.set_transform_params(rt.transform_params)
    q, qd, u = samples["q"], samples["qd"], samples["u"]
    checks = {
        "inverse_dynamics": (rt.inverse_dynamics(q, qd), baked.inverse_dynamics(q, qd)),
        "crba": (rt.crba(q), baked.crba(q)),
        "minv": (rt.minv(q), baked.minv(q)),
        "forward_dynamics": (rt.forward_dynamics(q, qd, u), baked.forward_dynamics(q, qd, u)),
    }
    for name, (a, b) in checks.items():
        e = _relerr(a, b)
        assert e < CHECK1_REL_TOL, f"{name}: runtime_transform(default) != baked, relerr={e:.3e}"


def test_perturbed_matches_recodegen(rt, recodegen, perturbed, samples):
    """CHECK2: mutating the transform table (set_transform_params) must equal a
    robot recompiled with those same origins baked into the URDF — proving a
    runtime mutation is equivalent to a full recodegen."""
    rt.set_transform_params(perturbed.astype(np.float32))
    q, qd, u = samples["q"], samples["qd"], samples["u"]
    checks = {
        "inverse_dynamics": (rt.inverse_dynamics(q, qd), recodegen.inverse_dynamics(q, qd)),
        "crba": (rt.crba(q), recodegen.crba(q)),
        "forward_dynamics": (rt.forward_dynamics(q, qd, u), recodegen.forward_dynamics(q, qd, u)),
    }
    for name, (a, b) in checks.items():
        e = _relerr(a, b)
        assert e < CHECK2_REL_TOL, f"{name}: perturbed != recodegen, relerr={e:.3e}"

    # And the perturbation genuinely changed the output (guards against a no-op
    # set_transform_params that would trivially "match" a baked-equals-baked case).
    rt.set_transform_params(rt.transform_params)
    baked_c = rt.inverse_dynamics(q, qd)
    rt.set_transform_params(perturbed.astype(np.float32))
    pert_c = rt.inverse_dynamics(q, qd)
    assert _relerr(pert_c, baked_c) > 1e-2, "perturbation had no effect"


def test_transform_poke_seen_across_surfaces(rt, perturbed, samples):
    """A poke through the numpy handle mutates the single device-resident
    d_transform_params table, so jax/torch (sharing the same dlopen'd .so __device__
    global) see the poked joint origins: numpy == jax == torch AFTER
    set_transform_params. End-to-end check that the jax/torch runtime_transform
    exposure threads the shared table. Skips if jax/torch unavailable."""
    gj = pytest.importorskip("grid_rbd.jax")
    pytest.importorskip("jax")
    gt = pytest.importorskip("grid_rbd.torch")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available for torch")
    # jax/torch handles over the SAME cached .so as the numpy `rt` fixture.
    common = dict(urdf_path=str(_IIWA), floating_base=False, max_batch_size=8,
                  runtime_transform=True)
    hj = gj.register_robot(name=rt.name, **common)
    ht = gt.register_robot(name=rt.name, **common)
    rt.set_transform_params(perturbed.astype(np.float32))   # poke via numpy
    q, qd = samples["q"], samples["qd"]
    qt = torch.tensor(q, device="cuda", dtype=torch.float32)
    qdt = torch.tensor(qd, device="cuda", dtype=torch.float32)
    cn = np.asarray(rt.inverse_dynamics(q, qd))
    cj = np.asarray(hj.inverse_dynamics(q, qd))
    ct = ht.inverse_dynamics(qt, qdt).detach().cpu().numpy()
    assert _relerr(cn, cj) < 2e-3, "jax did not see the numpy transform poke"
    assert _relerr(cn, ct) < 2e-3, "torch did not see the numpy transform poke"
    rt.set_transform_params(rt.transform_params)            # restore


def test_thread_invariance(rt, perturbed, samples):
    """CHECK3: the runtime_transform path is single-block and thread-count
    invariant — bit-identical (to abs tol) across {1,32,256} threads."""
    rt.set_transform_params(perturbed.astype(np.float32))
    q, qd = samples["q"], samples["qd"]
    ref = None
    for nthreads in (1, 32, 256):
        rt.set_threads_per_block(nthreads)
        out = np.asarray(rt.inverse_dynamics(q, qd))
        if ref is None:
            ref = out
        else:
            d = float(np.max(np.abs(out - ref)))
            assert d < CHECK3_ABS_TOL, f"threads={nthreads} vs 1: maxabsdiff={d:.3e}"
