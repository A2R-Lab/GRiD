"""Subset-build tests for the torch op surface of ``grid-rbd``.

Mirrors ``test_subset_build.py`` (the numpy subset suite) for the torch backend.
``grid_rbd.torch.register_robot(algorithm_list=[...])`` now builds only a SUBSET of
algorithms into the per-robot ``.so`` AND into the torch op surface: each
``torch_<algo>`` impl + its ``m.def`` / ``m.impl`` (and ``_mujoco`` twins) sits
inside ``#if GRID_HAS_<ALGO>``, mirroring the numpy C-ABI bodies. So a reduced
profile compiles cleanly, the requested cores + their transitive deps run (forward
+ autograd VJP), and an un-requested core's op is simply not registered — the
Python wrapper maps the resulting ``getattr`` ``AttributeError`` to the same clean
"not built into this robot .so — add to algorithm_list and rebuild" error the numpy
rc=3 path raises.

This suite covers, for the torch surface:
  (1) SUBSET build — requested algos + their transitive deps run + match the numpy
      oracle (forward), and forward_dynamics supports .backward() (VJP).
  (2) An un-requested algorithm raises the CLEAN subset error (naming the algo +
      algorithm_list), NOT a bare AttributeError / segfault / generic rc.
  (3) The subset ``.so`` is meaningfully smaller than the default build.
  (4) A default (no algorithm_list) register stays an instant cache hit (no rekey).

Every subset/default fixture passes ``force_rebuild=True`` to exercise the real
codegen+nvcc path (the build cache is content-addressed on inputs, NOT on the
generated CUDA — see docs/agent_debugging_guide.md §7).

Run with:
    pytest test/python_wrappers/test_subset_build_torch.py -m python_wrappers -v
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


# ─── skip preconditions ─────────────────────────────────────────────────────

_grid_rbd       = pytest.importorskip("grid_rbd",       reason="grid-rbd not installed (pip install bindings/)")
_torch          = pytest.importorskip("torch",          reason="torch not installed (pip install grid-rbd[torch])")
_grid_rbd_torch = pytest.importorskip("grid_rbd.torch", reason="grid_rbd.torch import failed")

if not _torch.cuda.is_available():
    pytest.skip("CUDA not available for torch", allow_module_level=True)

# In-repo iiwa14 URDF (always present alongside the codegen submodules).
_URDF = robot_urdf("iiwa14")
if not _URDF.exists():
    pytest.skip(f"iiwa14 URDF fixture not present at {_URDF}", allow_module_level=True)

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid_rbd.torch register_robot requires it", allow_module_level=True)


pytestmark = pytest.mark.python_wrappers

_TOL = 5e-3  # float32 vs float64 cross-precision


# ─── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    # Isolated cache so these (force_rebuild) builds never collide with the shared
    # ~/.cache/grid-rbd/ entries used by the other suites.
    return tmp_path_factory.mktemp("subset_build_torch_cache")


@pytest.fixture(scope="module")
def default_handle(cache_dir):
    """The full default profile (algorithm_list=None) — must stay fully working on
    the torch surface. force_rebuild to exercise the real codegen+nvcc path."""
    return _grid_rbd_torch.register_robot(
        name="iiwa14_subset_torch_default",
        urdf_path=str(_URDF),
        floating_base=False,
        max_batch_size=8,
        cache_dir=str(cache_dir),
        force_rebuild=True,
    )


@pytest.fixture(scope="module")
def subset_handle(cache_dir):
    """A reduced profile. Requesting inverse_dynamics + forward_dynamics +
    forward_dynamics_gradient: the normalizer expands fd_gradient's deps
    (inverse_dynamics, minv, forward_dynamics, inverse_dynamics_gradient), so minv
    and the gradients are available WITHOUT being named — proving dep expansion.
    crba / idsva_so / fdsva_so / end_effector_pose / integrator are NOT pulled in,
    so their torch ops are gated out → the wrapper raises the clean subset error."""
    return _grid_rbd_torch.register_robot(
        name="iiwa14_subset_torch_reduced",
        urdf_path=str(_URDF),
        floating_base=False,
        max_batch_size=8,
        cache_dir=str(cache_dir),
        force_rebuild=True,
        algorithm_list=["inverse_dynamics", "forward_dynamics",
                        "forward_dynamics_gradient"],
    )


@pytest.fixture(scope="module")
def ref():
    from URDFParser import URDFParser
    from RBDReference import RBDReference
    return RBDReference(URDFParser().parse(str(_URDF), floating_base=False))


@pytest.fixture(scope="module")
def samples(default_handle):
    rng = np.random.default_rng(7)
    NJ = default_handle.num_joints
    B = 4
    return {
        "q":  rng.standard_normal((B, NJ)).astype(np.float32),
        "qd": rng.standard_normal((B, NJ)).astype(np.float32),
        "u":  rng.standard_normal((B, NJ)).astype(np.float32),
    }


def _t(a):
    return _torch.tensor(a, device="cuda")


def _max_err(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# ─── (1) subset requested algos + transitive deps run + match the oracle ────


def test_subset_requested_algos_match_oracle(subset_handle, ref, samples):
    """The explicitly-requested algorithms (id, fd) run on the torch surface and
    match the numpy oracle."""
    q, qd, u = _t(samples["q"]), _t(samples["qd"]), _t(samples["u"])
    h = subset_handle
    id_out = h.inverse_dynamics(q, qd).cpu().numpy()
    fd_out = h.forward_dynamics(q, qd, u).cpu().numpy()
    qn, qdn, un = samples["q"], samples["qd"], samples["u"]
    for i in range(qn.shape[0]):
        qi, qdi, ui = qn[i].astype(np.float64), qdn[i].astype(np.float64), un[i].astype(np.float64)
        assert _max_err(id_out[i], ref.inverse_dynamics(qi, qdi, GRAVITY=-9.81)[0]) < _TOL
        assert _max_err(fd_out[i], ref.forward_dynamics(qi, qdi, ui)) < _TOL


def test_subset_transitive_deps_work(subset_handle, ref, samples):
    """minv + inverse_dynamics_gradient were NOT named, but forward_dynamics_gradient
    pulls them in — the codegen dep expansion makes them available (macro=1), so
    their torch ops are emitted, run, and match the oracle."""
    q, qd = _t(samples["q"]), _t(samples["qd"])
    h = subset_handle
    minv_out = h.minv(q).cpu().numpy()
    idg_out = h.inverse_dynamics_gradient(q, qd).cpu().numpy()
    qn, qdn = samples["q"], samples["qd"]
    for i in range(qn.shape[0]):
        qi, qdi = qn[i].astype(np.float64), qdn[i].astype(np.float64)
        assert _max_err(minv_out[i], ref.minv(qi)) < _TOL
        assert _max_err(idg_out[i], ref.inverse_dynamics_gradient(qi, qdi, GRAVITY=-9.81)) < _TOL


def test_subset_forward_dynamics_backward_vjp(subset_handle, samples):
    """A built subset op is autograd-aware: forward_dynamics(q, qd, u).sum() must
    .backward() and populate finite, nonzero grads for the differentiated inputs —
    the analytic VJP routes through the (transitively-built) fd_gradient + minv ops."""
    h = subset_handle
    q  = _t(samples["q"]).requires_grad_(True)
    qd = _t(samples["qd"]).requires_grad_(True)
    u  = _t(samples["u"]).requires_grad_(True)
    out = h.forward_dynamics(q, qd, u)
    out.sum().backward()
    for name, g in (("q", q.grad), ("qd", qd.grad), ("u", u.grad)):
        assert g is not None, f"no grad for {name}"
        assert _torch.isfinite(g).all(), f"non-finite grad for {name}"
    # ∂qdd/∂u = M⁻¹ is full-rank → the u-grad must be nonzero.
    assert float(u.grad.abs().max()) > 0, "u VJP is identically zero"


# ─── (2) un-requested algorithm raises the CLEAN subset error ───────────────


@pytest.mark.parametrize("method,call", [
    ("idsva_so", lambda h, s: h.idsva_so(_t(s["q"]), _t(s["qd"]), _t(s["u"]))),
    ("fdsva_so", lambda h, s: h.fdsva_so(_t(s["q"]), _t(s["qd"]), _t(s["u"]))),
    ("crba", lambda h, s: h.crba(_t(s["q"]))),
    ("end_effector_pose", lambda h, s: h.end_effector_pose(_t(s["q"]))),
    ("integrator", lambda h, s: h.integrator(_t(s["q"]), _t(s["qd"]), _t(s["u"]), 0.01, integrator_type="euler")),
])
def test_subset_unrequested_raises_clean_error(subset_handle, samples, method, call):
    """An algorithm NOT in the subset (and not a transitive dep) has its torch op
    gated out; calling it raises a clean runtime error naming the algorithm + how to
    fix it — never a bare AttributeError, segfault, or generic rc."""
    with pytest.raises(RuntimeError) as ei:
        call(subset_handle, samples)
    msg = str(ei.value)
    assert "not built into this robot .so" in msg, f"unclear error for {method}: {msg}"
    assert "algorithm_list" in msg, f"error for {method} doesn't point at the fix: {msg}"
    # Must be the dedicated SUBSET message, not the whole-surface-missing fallthrough
    # ("compiled with GRID_RBD_WITH_TORCH?") and not a generic "failed: rc=" path.
    assert "failed: rc=" not in msg, f"generic rc error leaked for {method}: {msg}"
    assert "GRID_RBD_WITH_TORCH" not in msg, f"whole-surface error leaked for {method}: {msg}"


def test_default_built_algos_run(default_handle, ref, samples):
    """Control: the un-requested algos that the subset gates out DO run on the
    DEFAULT build (proving the subset error is about the subset, not a regression)."""
    q, qd, u = _t(samples["q"]), _t(samples["qd"]), _t(samples["u"])
    h = default_handle
    crba = h.crba(q).cpu().numpy()
    for i in range(samples["q"].shape[0]):
        assert _max_err(crba[i], ref.crba(samples["q"][i].astype(np.float64))) < _TOL
    so = h.idsva_so(q, qd)
    assert all(_torch.isfinite(t).all() for t in so)
    assert _torch.isfinite(h.integrator(q, qd, u, 0.01, integrator_type="euler")).all()


# ─── (3) subset payoff: smaller .so ─────────────────────────────────────────


def test_subset_so_is_smaller_than_default(default_handle, subset_handle, cache_dir):
    """The reduced .so is meaningfully smaller than the full build (the
    un-requested heavy second-order / integrator inner kernels — and their torch
    ops — are simply not emitted)."""
    from grid_rbd._cache import store_dir
    entries = {e["name"]: e for e in _grid_rbd.list_registered(str(cache_dir))}
    d = store_dir(Path(cache_dir), entries["iiwa14_subset_torch_default"]["cache_key"]) / "robot.so"
    s = store_dir(Path(cache_dir), entries["iiwa14_subset_torch_reduced"]["cache_key"]) / "robot.so"
    if not d.exists() or not s.exists():
        pytest.skip("could not resolve .so paths via the manifest")
    assert s.stat().st_size < d.stat().st_size, (
        f"subset .so ({s.stat().st_size}) not smaller than default ({d.stat().st_size})")


# ─── (4) cache non-drift for the default profile ────────────────────────────


def test_subset_does_not_rekey_default(cache_dir):
    """A default register_robot (no algorithm_list) keeps the SAME cache key as a
    pre-subset build — re-registering is an instant cache hit (no nvcc), proving
    the subset plumbing is inject-only-when-set."""
    import time
    t0 = time.time()
    h = _grid_rbd_torch.register_robot(
        name="iiwa14_subset_torch_default",
        urdf_path=str(_URDF),
        floating_base=False,
        max_batch_size=8,
        cache_dir=str(cache_dir),
        force_rebuild=False,
    )
    assert h.num_joints == 7
    assert time.time() - t0 < 10.0, "default re-register recompiled — cache key drifted"
