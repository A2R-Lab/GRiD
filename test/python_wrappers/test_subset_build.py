"""Subset-build tests for the `grid-rbd` package.

`register_robot(algorithm_list=[...])` builds only a SUBSET of algorithms into the
per-robot `.so` (cuts nvcc wall-time / RAM / `.so` size for big robots). Codegen
already supported subsets via `_normalize_codegen_algorithms` (transitive-dep
expansion) + per-algorithm emission in grid.cuh; this suite exercises the binding
stack that plumbs the request and rc=3-gates the wrapper bodies.

The model (rc=3, low blast-radius): the codegen emits `#define GRID_HAS_<ALGO>
{0,1}` per core algorithm, the wrapper guards each `extern "C"` body with `#if
GRID_HAS_<ALGO>` (real body when 1, `(void)args; return 3;` when 0), and the
pybind Runner maps rc==3 to a clean "<algo> not built — add to algorithm_list and
rebuild" error. For the default "all" profile every macro is 1, so the wrapper
selects the real body verbatim and the default `.so` is byte-identical / fully
functional.

This suite covers:
  (1) DEFAULT build unchanged — every method runs + matches the oracle (the
      blast-radius gate: a default register_robot must behave exactly as before).
  (2) SUBSET build — requested algos + their transitive deps work vs the oracle
      (proving dep expansion), and an un-requested algorithm raises the clean
      "not built — add to algorithm_list" error (NOT a segfault, NOT a generic rc).

Run with:
    pytest test/python_wrappers/test_subset_build.py -m python_wrappers -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


# Repo root is parent of `test/`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


# ─── skip preconditions ─────────────────────────────────────────────────────

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed (pip install bindings/)")

# Use the in-repo iiwa14 URDF (always present alongside the codegen submodules),
# so the suite doesn't depend on an external robot_descriptions cache.
_URDF = _REPO_ROOT / "config/robot_assets" / "iiwa14.urdf"
if not _URDF.exists():
    pytest.skip(f"iiwa14 URDF fixture not present at {_URDF}", allow_module_level=True)

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)


pytestmark = pytest.mark.python_wrappers

_TOL = 5e-3  # float32 vs float64 cross-precision


# ─── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    # Isolated cache so these (force_rebuild) builds never collide with the shared
    # ~/.cache/grid-rbd/ entries used by the other suites.
    return tmp_path_factory.mktemp("subset_build_cache")


@pytest.fixture(scope="module")
def default_handle(cache_dir):
    """The full default profile (algorithm_list=None) — must stay byte-identical /
    fully working. force_rebuild to exercise the real codegen+nvcc path."""
    return _grid_rbd.register_robot(
        name="iiwa14_subset_default",
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
    so they must rc=3-stub with a clean error."""
    return _grid_rbd.register_robot(
        name="iiwa14_subset_reduced",
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


def _max_err(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# ─── (1) default unchanged ──────────────────────────────────────────────────


def test_default_metadata(default_handle):
    assert default_handle.num_joints == 7
    assert default_handle.num_vel == 7


def test_default_value_methods_match_oracle(default_handle, ref, samples):
    """The full default build runs every value method and matches the oracle —
    the blast-radius gate (a default register_robot must behave as before)."""
    q, qd, u = samples["q"], samples["qd"], samples["u"]
    h = default_handle
    for i in range(q.shape[0]):
        qi, qdi, ui = q[i].astype(np.float64), qd[i].astype(np.float64), u[i].astype(np.float64)
        assert _max_err(h.inverse_dynamics(q, qd)[i],
                        ref.inverse_dynamics(qi, qdi, GRAVITY=-9.81)[0]) < _TOL
        assert _max_err(h.forward_dynamics(q, qd, u)[i],
                        ref.forward_dynamics(qi, qdi, ui)) < _TOL
        assert _max_err(h.aba(q, qd, u)[i],
                        ref.aba(qi, qdi, ui, GRAVITY=-9.81)) < _TOL
        assert _max_err(h.minv(q)[i], ref.minv(qi)) < _TOL
        assert _max_err(h.crba(q)[i], ref.crba(qi)) < _TOL


def test_default_derivative_methods_match_oracle(default_handle, ref, samples):
    q, qd, u = samples["q"], samples["qd"], samples["u"]
    h = default_handle
    NJ = h.num_joints
    g_id = h.inverse_dynamics_gradient(q, qd)
    g_fd = h.forward_dynamics_gradient(q, qd, u)
    g_ee = h.end_effector_pose_gradient(q)
    for i in range(q.shape[0]):
        qi, qdi, ui = q[i].astype(np.float64), qd[i].astype(np.float64), u[i].astype(np.float64)
        assert _max_err(g_id[i], ref.inverse_dynamics_gradient(qi, qdi, GRAVITY=-9.81)) < _TOL
        dq, dqd = ref.forward_dynamics_gradient(qi, qdi, ui)
        assert _max_err(g_fd[i][:, :NJ], dq) < _TOL
        assert _max_err(g_fd[i][:, NJ:], dqd) < _TOL
        assert _max_err(g_ee[i], ref.end_effector_pose_gradient(qi)[0]) < _TOL


def test_default_second_order_and_integrator_run(default_handle, samples):
    """idsva_so / fdsva_so / ee_hessian / integrator(+gradient) all run + finite in
    the default build (numerical parity is covered by the broader smoke suite)."""
    q, qd, u = samples["q"], samples["qd"], samples["u"]
    h = default_handle
    for out in (h.idsva_so(q, qd, u), h.fdsva_so(q, qd, u)):
        assert isinstance(out, tuple) and len(out) == 4
        assert all(np.all(np.isfinite(t)) for t in out)
    assert np.all(np.isfinite(h.end_effector_pose_hessian(q)))
    assert np.all(np.isfinite(h.integrator(q, qd, u, 0.01, integrator_type="euler")))
    assert np.all(np.isfinite(h.integrator_gradient(q, qd, u, 0.01, integrator_type="euler")))


# ─── (2) subset build ───────────────────────────────────────────────────────


def test_subset_requested_algos_match_oracle(subset_handle, ref, samples):
    """The explicitly-requested algorithms (id, fd) run + match the oracle."""
    q, qd, u = samples["q"], samples["qd"], samples["u"]
    h = subset_handle
    for i in range(q.shape[0]):
        qi, qdi, ui = q[i].astype(np.float64), qd[i].astype(np.float64), u[i].astype(np.float64)
        assert _max_err(h.inverse_dynamics(q, qd)[i],
                        ref.inverse_dynamics(qi, qdi, GRAVITY=-9.81)[0]) < _TOL
        assert _max_err(h.forward_dynamics(q, qd, u)[i],
                        ref.forward_dynamics(qi, qdi, ui)) < _TOL


def test_subset_transitive_deps_work(subset_handle, ref, samples):
    """minv + inverse_dynamics_gradient were NOT named, but forward_dynamics_gradient
    pulls them in — the codegen dep expansion makes them available (macro=1), so they
    run + match the oracle. This proves the subset request threads through the
    normalizer's transitive-dependency expansion."""
    q, qd = samples["q"], samples["qd"]
    h = subset_handle
    for i in range(q.shape[0]):
        qi, qdi = q[i].astype(np.float64), qd[i].astype(np.float64)
        assert _max_err(h.minv(q)[i], ref.minv(qi)) < _TOL
        assert _max_err(h.inverse_dynamics_gradient(q, qd)[i],
                        ref.inverse_dynamics_gradient(qi, qdi, GRAVITY=-9.81)) < _TOL


@pytest.mark.parametrize("method,call", [
    ("idsva_so", lambda h, s: h.idsva_so(s["q"], s["qd"], s["u"])),
    ("fdsva_so", lambda h, s: h.fdsva_so(s["q"], s["qd"], s["u"])),
    ("crba", lambda h, s: h.crba(s["q"])),
    ("end_effector_pose", lambda h, s: h.end_effector_pose(s["q"])),
    ("integrator", lambda h, s: h.integrator(s["q"], s["qd"], s["u"], 0.01, integrator_type="euler")),
])
def test_subset_unrequested_raises_clean_error(subset_handle, samples, method, call):
    """An algorithm NOT in the subset (and not a transitive dep) raises a clean
    runtime error naming the algorithm + how to fix it — never a segfault, never a
    bare 'rc=3'."""
    with pytest.raises(RuntimeError) as ei:
        call(subset_handle, samples)
    msg = str(ei.value)
    assert "not built into this robot .so" in msg, f"unclear error for {method}: {msg}"
    assert "algorithm_list" in msg, f"error for {method} doesn't point at the fix: {msg}"
    # Must be the dedicated subset message, not the generic "failed: rc=3" fallthrough.
    assert "failed: rc=" not in msg, f"generic rc error leaked for {method}: {msg}"


def test_subset_so_is_smaller_than_default(default_handle, subset_handle, cache_dir):
    """The reduced .so is meaningfully smaller than the full build (the un-requested
    heavy second-order / integrator inner kernels are simply not emitted)."""
    from grid_rbd._cache import store_dir
    entries = {e["name"]: e for e in _grid_rbd.list_registered(str(cache_dir))}
    d = store_dir(Path(cache_dir), entries["iiwa14_subset_default"]["cache_key"]) / "robot.so"
    s = store_dir(Path(cache_dir), entries["iiwa14_subset_reduced"]["cache_key"]) / "robot.so"
    if not d.exists() or not s.exists():
        pytest.skip("could not resolve .so paths via the manifest")
    assert s.stat().st_size < d.stat().st_size


def test_subset_does_not_rekey_default(cache_dir):
    """A default register_robot (no algorithm_list) keeps the SAME cache key as
    before the subset feature — re-registering is an instant cache hit (no nvcc),
    proving the subset plumbing is inject-only-when-set."""
    import time
    t0 = time.time()
    h = _grid_rbd.register_robot(
        name="iiwa14_subset_default",
        urdf_path=str(_URDF),
        floating_base=False,
        max_batch_size=8,
        cache_dir=str(cache_dir),
        force_rebuild=False,
    )
    assert h.num_joints == 7
    assert time.time() - t0 < 5.0, "default re-register recompiled — cache key drifted"
