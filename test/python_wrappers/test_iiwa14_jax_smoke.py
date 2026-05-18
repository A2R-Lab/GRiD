"""JAX FFI smoke tests for grid_rbd.jax.

Validates that the JAX FFI integration:
  * Compiles the per-robot .so with the JAX FFI handler block included.
  * Resolves the FFI handler symbol via dlopen and registers it with JAX.
  * Returns numerically-identical results to the plain Python wrapper
    (which exercises the same underlying CUDA kernels).
  * Works under ``jax.jit`` and accepts both numpy + JAX arrays as input
    (JAX moves CPU→GPU transparently before the FFI handler runs).

Skips if jax, grid_rbd, or the URDF fixture aren't available.

Run with:
    pytest test/python_wrappers/test_iiwa14_jax_smoke.py -m python_wrappers -v
or as part of the full python_wrappers tier:
    pytest test/python_wrappers/ -m python_wrappers -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


# ─── preconditions ───────────────────────────────────────────────────────────

_grid_rbd      = pytest.importorskip("grid_rbd",      reason="grid-rbd not installed")
_jax           = pytest.importorskip("jax",           reason="jax not installed (pip install grid-rbd[jax])")
_grid_rbd_jax  = pytest.importorskip("grid_rbd.jax",  reason="grid_rbd.jax import failed")

_URDF = (
    Path.home()
    / ".cache/robot_descriptions/drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
)
if not _URDF.exists():
    pytest.skip(f"iiwa14 URDF fixture not present at {_URDF}", allow_module_level=True)

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid_rbd.jax.register_robot requires it", allow_module_level=True)


pytestmark = pytest.mark.python_wrappers
_TOL = 5e-3


# ─── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def jax_handle():
    return _grid_rbd_jax.register_robot(
        name="iiwa14_jax_pytest",
        urdf_path=str(_URDF),
        floating_base=False,
        max_batch_size=8,
    )


@pytest.fixture(scope="module")
def plain_handle():
    # Same cache key, same .so — picked up by name from the manifest.
    return _grid_rbd.get_robot("iiwa14_jax_pytest")


@pytest.fixture(scope="module")
def samples(jax_handle):
    rng = np.random.default_rng(0)
    NJ = jax_handle.num_joints
    B = 4
    return {
        "q":  rng.standard_normal((B, NJ)).astype(np.float32),
        "qd": rng.standard_normal((B, NJ)).astype(np.float32),
        "u":  rng.standard_normal((B, NJ)).astype(np.float32),
    }


# ─── tests ───────────────────────────────────────────────────────────────────


def test_metadata(jax_handle):
    assert jax_handle.num_joints == 7
    assert jax_handle.num_vel == 7
    assert jax_handle.num_ees == 1
    assert jax_handle.floating_base is False


def test_rnea_eager_matches_plain(jax_handle, plain_handle, samples):
    """JAX FFI rnea (eager) must produce identical results to the plain
    Python wrapper, since both ultimately dispatch the same CUDA kernel."""
    c_jax   = np.asarray(jax_handle.rnea(samples["q"], samples["qd"]))
    c_plain = plain_handle.rnea(samples["q"], samples["qd"])
    assert c_jax.shape == c_plain.shape
    assert np.max(np.abs(c_jax - c_plain)) < _TOL


def test_rnea_jit_matches_eager(jax_handle, samples):
    """jax.jit compilation must not change the result."""
    import jax
    @jax.jit
    def f(q, qd):
        return jax_handle.rnea(q, qd)
    c_jit  = np.asarray(f(samples["q"], samples["qd"]))
    c_eager = np.asarray(jax_handle.rnea(samples["q"], samples["qd"]))
    assert c_jit.shape == c_eager.shape
    assert np.max(np.abs(c_jit - c_eager)) < 1e-6  # should be bitwise identical


def test_rnea_accepts_numpy_input(jax_handle, samples):
    """JAX accepts CPU numpy arrays and moves them to GPU transparently;
    the FFI handler sees device-resident buffers either way."""
    q  = samples["q"]
    qd = samples["qd"]
    # Numpy in:
    c1 = np.asarray(jax_handle.rnea(q, qd))
    # JAX device array in:
    import jax.numpy as jnp
    c2 = np.asarray(jax_handle.rnea(jnp.asarray(q), jnp.asarray(qd)))
    assert np.allclose(c1, c2, atol=1e-6)


def test_rnea_under_vmap(jax_handle, samples):
    """jax.vmap doesn't apply here (our handler already batches on axis 0),
    but composing under jit + asarray should be fine."""
    import jax
    @jax.jit
    def f(q, qd):
        # Add a noop transformation around the FFI call to test that the
        # call slots into a larger JIT graph.
        return jax_handle.rnea(q, qd) + 0.0
    c = np.asarray(f(samples["q"], samples["qd"]))
    assert c.shape == samples["q"].shape
    assert np.all(np.isfinite(c))


def test_minv_eager_matches_plain(jax_handle, plain_handle, samples):
    m_jax   = np.asarray(jax_handle.minv(samples["q"]))
    m_plain = plain_handle.minv(samples["q"])
    assert m_jax.shape == m_plain.shape
    assert np.max(np.abs(m_jax - m_plain)) < _TOL


def test_minv_is_symmetric(jax_handle, samples):
    """The handler returns the lower triangle; JaxRobotHandle.minv symmetrizes."""
    m = np.asarray(jax_handle.minv(samples["q"]))
    assert np.allclose(m, np.swapaxes(m, -1, -2), atol=1e-6)


def test_forward_dynamics_eager_matches_plain(jax_handle, plain_handle, samples):
    qdd_jax   = np.asarray(jax_handle.forward_dynamics(samples["q"], samples["qd"], samples["u"]))
    qdd_plain = plain_handle.forward_dynamics(samples["q"], samples["qd"], samples["u"])
    assert qdd_jax.shape == qdd_plain.shape
    assert np.max(np.abs(qdd_jax - qdd_plain)) < _TOL


def test_aba_eager_matches_plain(jax_handle, plain_handle, samples):
    qdd_jax   = np.asarray(jax_handle.aba(samples["q"], samples["qd"], samples["u"]))
    qdd_plain = plain_handle.aba(samples["q"], samples["qd"], samples["u"])
    assert qdd_jax.shape == qdd_plain.shape
    assert np.max(np.abs(qdd_jax - qdd_plain)) < _TOL


def test_aba_matches_forward_dynamics(jax_handle, samples):
    """ABA and forward_dynamics solve the same problem via different paths."""
    qdd_aba = np.asarray(jax_handle.aba(samples["q"], samples["qd"], samples["u"]))
    qdd_fd  = np.asarray(jax_handle.forward_dynamics(samples["q"], samples["qd"], samples["u"]))
    assert np.max(np.abs(qdd_aba - qdd_fd)) < _TOL


def test_crba_eager_matches_plain(jax_handle, plain_handle, samples):
    m_jax   = np.asarray(jax_handle.crba(samples["q"]))
    m_plain = plain_handle.crba(samples["q"])
    assert m_jax.shape == m_plain.shape
    assert np.max(np.abs(m_jax - m_plain)) < _TOL


def test_crba_is_symmetric(jax_handle, samples):
    m = np.asarray(jax_handle.crba(samples["q"]))
    assert np.allclose(m, np.swapaxes(m, -1, -2), atol=5e-4)


def test_minv_and_crba_invert(jax_handle, samples):
    """Minv·M ≈ I — exercises both kernels end-to-end."""
    M    = np.asarray(jax_handle.crba(samples["q"]))
    Minv = np.asarray(jax_handle.minv(samples["q"]))
    NJ   = jax_handle.num_joints
    eye  = np.eye(NJ, dtype=np.float32)[None].repeat(M.shape[0], axis=0)
    prod = Minv @ M
    # float32 + non-trivial conditioning — keep this loose.
    assert np.max(np.abs(prod - eye)) < 5e-3


def test_all_methods_jit(jax_handle, samples):
    """Every method must slot into a single jax.jit graph."""
    import jax
    @jax.jit
    def f(q, qd, u):
        c    = jax_handle.rnea(q, qd)
        qdd1 = jax_handle.forward_dynamics(q, qd, u)
        qdd2 = jax_handle.aba(q, qd, u)
        Minv = jax_handle.minv(q)
        M    = jax_handle.crba(q)
        return c, qdd1, qdd2, Minv, M
    c, qdd1, qdd2, Minv, M = f(samples["q"], samples["qd"], samples["u"])
    for x in (c, qdd1, qdd2, Minv, M):
        assert np.all(np.isfinite(np.asarray(x)))


def test_register_idempotent(jax_handle):
    """Re-registering under the same name reuses the cached .so."""
    import time
    t0 = time.time()
    h2 = _grid_rbd_jax.register_robot(
        name="iiwa14_jax_pytest",
        urdf_path=str(_URDF),
        floating_base=False,
        max_batch_size=8,
    )
    elapsed = time.time() - t0
    assert h2.num_joints == jax_handle.num_joints
    assert elapsed < 5.0, f"cache hit took {elapsed:.1f}s — should be <1s"
