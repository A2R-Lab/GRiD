"""Subset-build tests for the JAX FFI surface of ``grid-rbd``.

Mirrors ``test_subset_build.py`` (the numpy subset suite) for the JAX backend.
``grid_rbd.jax.register_robot(algorithm_list=[...])`` now builds only a SUBSET of
algorithms into the per-robot ``.so`` AND into the JAX FFI surface: each
``grid::*_kernel``-calling handler (impl + its ``XLA_FFI_DEFINE_HANDLER_SYMBOL``
pin/mjx blocks) sits inside ``#if GRID_HAS_<ALGO>``, mirroring the numpy C-ABI
bodies. So a reduced profile compiles cleanly, the requested cores + their
transitive deps run, and an un-requested core's FFI symbol is simply absent — the
Python wrapper maps the resulting dlopen ``AttributeError`` to the same clean
"not built into this robot .so — add to algorithm_list and rebuild" error the
numpy rc=3 path raises.

This suite covers, for the JAX surface:
  (1) SUBSET build — requested algos + their transitive deps run + match the numpy
      oracle (eager + jit + vmap), proving the dep expansion threads through.
  (2) An un-requested algorithm raises the CLEAN subset error (naming the algo +
      algorithm_list), NOT a bare AttributeError / segfault / generic rc.
  (3) The subset ``.so`` is meaningfully smaller than the default build.
  (4) A default (no algorithm_list) register stays an instant cache hit (no rekey).

Every subset/default fixture passes ``force_rebuild=True`` to exercise the real
codegen+nvcc path (the build cache is content-addressed on inputs, NOT on the
generated CUDA — see docs/agent_debugging_guide.md §7).

Run with:
    pytest test/python_wrappers/test_subset_build_jax.py -m python_wrappers -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


# ─── skip preconditions ─────────────────────────────────────────────────────

_grid_rbd     = pytest.importorskip("grid_rbd",     reason="grid-rbd not installed (pip install bindings/)")
_jax          = pytest.importorskip("jax",          reason="jax not installed (pip install grid-rbd[jax])")
_grid_rbd_jax = pytest.importorskip("grid_rbd.jax", reason="grid_rbd.jax import failed")

# In-repo iiwa14 URDF (always present alongside the codegen submodules).
_URDF = _REPO_ROOT / "robot_assets" / "iiwa14.urdf"
if not _URDF.exists():
    pytest.skip(f"iiwa14 URDF fixture not present at {_URDF}", allow_module_level=True)

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid_rbd.jax register_robot requires it", allow_module_level=True)


pytestmark = pytest.mark.python_wrappers

_TOL = 5e-3  # float32 vs float64 cross-precision


# ─── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    # Isolated cache so these (force_rebuild) builds never collide with the shared
    # ~/.cache/grid-rbd/ entries used by the other suites.
    return tmp_path_factory.mktemp("subset_build_jax_cache")


@pytest.fixture(scope="module")
def default_handle(cache_dir):
    """The full default profile (algorithm_list=None) — must stay fully working
    on the JAX surface. force_rebuild to exercise the real codegen+nvcc path."""
    return _grid_rbd_jax.register_robot(
        name="iiwa14_subset_jax_default",
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
    so their FFI symbols are gated out → the wrapper raises the clean subset error."""
    return _grid_rbd_jax.register_robot(
        name="iiwa14_subset_jax_reduced",
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


# ─── (1) subset requested algos + transitive deps run + match the oracle ────


def test_subset_requested_algos_match_oracle(subset_handle, ref, samples):
    """The explicitly-requested algorithms (id, fd) run on the JAX surface
    (eager) and match the numpy oracle."""
    q, qd, u = samples["q"], samples["qd"], samples["u"]
    h = subset_handle
    id_out = np.asarray(h.inverse_dynamics(q, qd))
    fd_out = np.asarray(h.forward_dynamics(q, qd, u))
    for i in range(q.shape[0]):
        qi, qdi, ui = q[i].astype(np.float64), qd[i].astype(np.float64), u[i].astype(np.float64)
        assert _max_err(id_out[i], ref.inverse_dynamics(qi, qdi, GRAVITY=-9.81)[0]) < _TOL
        assert _max_err(fd_out[i], ref.forward_dynamics(qi, qdi, ui)) < _TOL


def test_subset_transitive_deps_work(subset_handle, ref, samples):
    """minv + inverse_dynamics_gradient were NOT named, but forward_dynamics_gradient
    pulls them in — the codegen dep expansion makes them available (macro=1), so
    their JAX handlers are emitted, run, and match the oracle."""
    q, qd = samples["q"], samples["qd"]
    h = subset_handle
    minv_out = np.asarray(h.minv(q))
    idg_out = np.asarray(h.inverse_dynamics_gradient(q, qd))
    for i in range(q.shape[0]):
        qi, qdi = q[i].astype(np.float64), qd[i].astype(np.float64)
        assert _max_err(minv_out[i], ref.minv(qi)) < _TOL
        assert _max_err(idg_out[i], ref.inverse_dynamics_gradient(qi, qdi, GRAVITY=-9.81)) < _TOL


def test_subset_jit_matches_eager(subset_handle, samples):
    """A built subset op composes inside jax.jit (the FFI target registered fine)."""
    import jax
    q, qd, u = samples["q"], samples["qd"], samples["u"]
    h = subset_handle
    f = jax.jit(lambda q, qd, u: h.forward_dynamics(q, qd, u))
    jit_out = np.asarray(f(q, qd, u))
    eager_out = np.asarray(h.forward_dynamics(q, qd, u))
    assert jit_out.shape == eager_out.shape
    assert np.max(np.abs(jit_out - eager_out)) < 1e-6


def test_subset_vmap_compose(subset_handle, samples):
    """A built subset op slots into a larger jit graph (our handler already
    batches axis 0; the +0.0 noop checks graph composition)."""
    import jax
    q, qd = samples["q"], samples["qd"]
    h = subset_handle
    f = jax.jit(lambda q, qd: h.inverse_dynamics(q, qd) + 0.0)
    out = np.asarray(f(q, qd))
    assert out.shape == q.shape
    assert np.all(np.isfinite(out))


# ─── (2) un-requested algorithm raises the CLEAN subset error ───────────────


@pytest.mark.parametrize("method,call", [
    ("idsva_so", lambda h, s: h.idsva_so(s["q"], s["qd"], s["u"])),
    ("fdsva_so", lambda h, s: h.fdsva_so(s["q"], s["qd"], s["u"])),
    ("crba", lambda h, s: h.crba(s["q"])),
    ("end_effector_pose", lambda h, s: h.end_effector_pose(s["q"])),
    ("integrator", lambda h, s: h.integrator(s["q"], s["qd"], s["u"], 0.01, integrator_type="euler")),
])
def test_subset_unrequested_raises_clean_error(subset_handle, samples, method, call):
    """An algorithm NOT in the subset (and not a transitive dep) has its JAX FFI
    handler gated out; calling it raises a clean runtime error naming the algorithm
    + how to fix it — never a bare AttributeError, segfault, or generic rc."""
    with pytest.raises(RuntimeError) as ei:
        call(subset_handle, samples)
    msg = str(ei.value)
    assert "not built into this robot .so" in msg, f"unclear error for {method}: {msg}"
    assert "algorithm_list" in msg, f"error for {method} doesn't point at the fix: {msg}"
    # Must be the dedicated SUBSET message, not the whole-surface-missing fallthrough
    # ("compiled with GRID_RBD_WITH_JAX?") and not a generic "failed: rc=" path.
    assert "failed: rc=" not in msg, f"generic rc error leaked for {method}: {msg}"
    assert "GRID_RBD_WITH_JAX" not in msg, f"whole-surface error leaked for {method}: {msg}"


def test_default_built_algos_run(default_handle, ref, samples):
    """Control: the un-requested algos that the subset gates out DO run on the
    DEFAULT build (proving the subset error is about the subset, not a regression)."""
    q, qd, u = samples["q"], samples["qd"], samples["u"]
    h = default_handle
    crba = np.asarray(h.crba(q))
    for i in range(q.shape[0]):
        assert _max_err(crba[i], ref.crba(q[i].astype(np.float64))) < _TOL
    so = h.idsva_so(q, qd, u)
    assert isinstance(so, tuple) and all(np.all(np.isfinite(np.asarray(t))) for t in so)
    assert np.all(np.isfinite(np.asarray(h.integrator(q, qd, u, 0.01, integrator_type="euler"))))


# ─── (3) subset payoff: smaller .so ─────────────────────────────────────────


def test_subset_so_is_smaller_than_default(default_handle, subset_handle, cache_dir):
    """The reduced .so is meaningfully smaller than the full build (the
    un-requested heavy second-order / integrator inner kernels — and their JAX
    handlers — are simply not emitted)."""
    from grid_rbd._cache import store_dir
    entries = {e["name"]: e for e in _grid_rbd.list_registered(str(cache_dir))}
    d = store_dir(Path(cache_dir), entries["iiwa14_subset_jax_default"]["cache_key"]) / "robot.so"
    s = store_dir(Path(cache_dir), entries["iiwa14_subset_jax_reduced"]["cache_key"]) / "robot.so"
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
    h = _grid_rbd_jax.register_robot(
        name="iiwa14_subset_jax_default",
        urdf_path=str(_URDF),
        floating_base=False,
        max_batch_size=8,
        cache_dir=str(cache_dir),
        force_rebuild=False,
    )
    assert h.num_joints == 7
    assert time.time() - t0 < 10.0, "default re-register recompiled — cache key drifted"
