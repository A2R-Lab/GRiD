"""External-force (f_ext) handle coverage for `grid-rbd` (E6).

Exercises the optional ``f_ext=`` kwarg threaded through the numpy ``RobotHandle``
and (when torch is available) the ``TorchRobotHandle`` for inverse_dynamics / forward_dynamics
/ aba on iiwa14 (fixed-base), and asserts numerical agreement with the
``RBDReference`` external-force reference (``inverse_dynamics(..., f_ext=...)`` /
``aba(..., f_ext=...)``, which subtract the per-body local-frame wrench
``f[:, i] -= f_ext[i]``).

f_ext layout (handle side): (B, 6*num_bodies) float32, body-major, each per-body
wrench ordered [angular(3); linear(3)] in that link's LOCAL frame — identical to
``RBDReference.apply_external_forces`` and to the CUDA kernel's ``d_f_ext``.

Also asserts the no-f_ext path is byte-identical (f_ext=None == omitting it ==
zeros), so this surface is a pure superset of the previous behavior.

Run with:
    pytest test/python_wrappers/test_iiwa14_f_ext.py -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")

_URDF = _REPO_ROOT / "robot_assets" / "iiwa14.urdf"
if not _URDF.exists():
    pytest.skip(f"iiwa14 URDF fixture not present at {_URDF}", allow_module_level=True)
if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)


pytestmark = pytest.mark.python_wrappers

_TOL = 5e-3
_GRAVITY = -9.81  # RBDReference convention; handle takes +9.81 magnitude.


# ─── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def handle():
    return _grid_rbd.register_robot(
        name="iiwa14_fext_smoke", urdf_path=str(_URDF),
        floating_base=False, max_batch_size=8)


@pytest.fixture(scope="module")
def ref():
    from URDFParser import URDFParser
    from RBDReference import RBDReference
    return RBDReference(URDFParser().parse(str(_URDF), floating_base=False))


@pytest.fixture(scope="module")
def samples(handle):
    rng = np.random.default_rng(7)
    NJ = handle.num_joints
    NB = handle.num_bodies
    B = 4
    # A known nonzero local-frame f_ext: a wrench on every body (body-major,
    # [angular; linear]). Per-body distinct so a wrong body-ordering would fail.
    f_ext = np.zeros((B, 6 * NB), dtype=np.float32)
    for b in range(NB):
        f_ext[:, 6 * b: 6 * b + 6] = (0.1 * (b + 1)) * np.array(
            [0.3, -0.5, 0.2, 1.1, -0.7, 0.9], dtype=np.float32)
    return {
        "q":  rng.standard_normal((B, NJ)).astype(np.float32),
        "qd": rng.standard_normal((B, NJ)).astype(np.float32),
        "u":  rng.standard_normal((B, NJ)).astype(np.float32),
        "f_ext": f_ext,
        "B": B,
    }


def _f_ext_list(f_ext_row, NB):
    """(6*NB,) handle row -> list of NB length-6 vectors (RBDReference form)."""
    return [f_ext_row[6 * b: 6 * b + 6].astype(np.float64) for b in range(NB)]


def _max_err(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# ─── metadata ───────────────────────────────────────────────────────────────


def test_num_bodies(handle):
    assert handle.num_bodies == 7  # iiwa14 fixed-base: 7 moving links


# ─── numpy handle vs RBDReference ───────────────────────────────────────────


def test_inverse_dynamics_f_ext(handle, ref, samples):
    NB = handle.num_bodies
    grid = handle.inverse_dynamics(samples["q"], samples["qd"], f_ext=samples["f_ext"])
    for i, (q, qd) in enumerate(zip(samples["q"], samples["qd"])):
        fe = _f_ext_list(samples["f_ext"][i], NB)
        c_ref, *_ = ref.inverse_dynamics(q.astype(np.float64), qd.astype(np.float64),
                             GRAVITY=_GRAVITY, f_ext=fe)
        assert _max_err(grid[i], c_ref) < _TOL


def test_forward_dynamics_f_ext(handle, ref, samples):
    NB = handle.num_bodies
    grid = handle.forward_dynamics(samples["q"], samples["qd"], samples["u"],
                                   f_ext=samples["f_ext"])
    for i, (q, qd, u) in enumerate(zip(samples["q"], samples["qd"], samples["u"])):
        fe = _f_ext_list(samples["f_ext"][i], NB)
        # qdd = Minv (u - inverse_dynamics(q,qd,0; f_ext)); reference forward_dynamics has no
        # f_ext arg, so recompute via the bias from inverse_dynamics(..., f_ext).
        c_ref, *_ = ref.inverse_dynamics(q.astype(np.float64), qd.astype(np.float64),
                             GRAVITY=_GRAVITY, f_ext=fe)
        Minv = ref.minv(q.astype(np.float64))
        qdd_ref = Minv @ (u.astype(np.float64) - c_ref)
        assert _max_err(grid[i], qdd_ref) < _TOL


def test_aba_f_ext(handle, ref, samples):
    NB = handle.num_bodies
    grid = handle.aba(samples["q"], samples["qd"], samples["u"], f_ext=samples["f_ext"])
    for i, (q, qd, u) in enumerate(zip(samples["q"], samples["qd"], samples["u"])):
        fe = _f_ext_list(samples["f_ext"][i], NB)
        qdd_ref = ref.aba(q.astype(np.float64), qd.astype(np.float64),
                          u.astype(np.float64), f_ext=fe, GRAVITY=_GRAVITY)
        assert _max_err(grid[i], qdd_ref) < _TOL


# ─── no-f_ext path unchanged (defaults to zeros / identical to before) ──────


def test_no_f_ext_path_unchanged(handle, samples):
    """f_ext=None == omitting f_ext == zeros: byte-identical."""
    NB = handle.num_bodies
    zeros = np.zeros((samples["B"], 6 * NB), dtype=np.float32)
    base = handle.inverse_dynamics(samples["q"], samples["qd"])
    none = handle.inverse_dynamics(samples["q"], samples["qd"], f_ext=None)
    zero = handle.inverse_dynamics(samples["q"], samples["qd"], f_ext=zeros)
    assert np.array_equal(base, none)
    assert np.array_equal(base, zero)
    # And a nonzero f_ext must actually change the answer (sanity).
    nz = handle.inverse_dynamics(samples["q"], samples["qd"], f_ext=samples["f_ext"])
    assert _max_err(base, nz) > 1e-3
    # The singleton buffer is reset after each call: a no-f_ext call AFTER a
    # nonzero one returns the no-f_ext answer (no stale f_ext leakage).
    after = handle.inverse_dynamics(samples["q"], samples["qd"])
    assert np.array_equal(base, after)


def test_f_ext_bad_shape_raises(handle, samples):
    bad = np.zeros((samples["B"], 5), dtype=np.float32)  # wrong last dim
    with pytest.raises(ValueError):
        handle.inverse_dynamics(samples["q"], samples["qd"], f_ext=bad)


# ─── torch handle (optional) ────────────────────────────────────────────────


def test_torch_inverse_dynamics_f_ext(handle, ref, samples):
    torch = pytest.importorskip("torch", reason="torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for torch")
    import grid_rbd.torch as gt
    th = gt.get_robot("iiwa14_fext_smoke")
    NB = th.num_bodies

    q = torch.tensor(samples["q"], device="cuda", dtype=torch.float32)
    qd = torch.tensor(samples["qd"], device="cuda", dtype=torch.float32)
    fe = torch.tensor(samples["f_ext"], device="cuda", dtype=torch.float32)

    out = th.inverse_dynamics(q, qd, f_ext=fe).detach().cpu().numpy()
    for i in range(samples["B"]):
        fel = _f_ext_list(samples["f_ext"][i], NB)
        c_ref, *_ = ref.inverse_dynamics(samples["q"][i].astype(np.float64),
                             samples["qd"][i].astype(np.float64),
                             GRAVITY=_GRAVITY, f_ext=fel)
        assert _max_err(out[i], c_ref) < _TOL

    # parity with the numpy handle's f_ext path
    npy = handle.inverse_dynamics(samples["q"], samples["qd"], f_ext=samples["f_ext"])
    assert _max_err(out, npy) < _TOL

    # no-f_ext torch path unchanged + autograd still flows with f_ext
    base = th.inverse_dynamics(q, qd).detach().cpu().numpy()
    none = th.inverse_dynamics(q, qd, f_ext=None).detach().cpu().numpy()
    assert np.allclose(base, none, atol=0, rtol=0)

    qg = q.clone().requires_grad_(True)
    th.inverse_dynamics(qg, qd, f_ext=fe).sum().backward()
    assert qg.grad is not None and torch.isfinite(qg.grad).all()
