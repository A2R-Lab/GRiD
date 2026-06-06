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


def test_inverse_dynamics_eager_matches_plain(jax_handle, plain_handle, samples):
    """JAX FFI inverse_dynamics (eager) must produce identical results to the plain
    Python wrapper, since both ultimately dispatch the same CUDA kernel."""
    c_jax   = np.asarray(jax_handle.inverse_dynamics(samples["q"], samples["qd"]))
    c_plain = plain_handle.inverse_dynamics(samples["q"], samples["qd"])
    assert c_jax.shape == c_plain.shape
    assert np.max(np.abs(c_jax - c_plain)) < _TOL


def test_inverse_dynamics_jit_matches_eager(jax_handle, samples):
    """jax.jit compilation must not change the result."""
    import jax
    @jax.jit
    def f(q, qd):
        return jax_handle.inverse_dynamics(q, qd)
    c_jit  = np.asarray(f(samples["q"], samples["qd"]))
    c_eager = np.asarray(jax_handle.inverse_dynamics(samples["q"], samples["qd"]))
    assert c_jit.shape == c_eager.shape
    assert np.max(np.abs(c_jit - c_eager)) < 1e-6  # should be bitwise identical


def test_inverse_dynamics_honors_qdd(jax_handle, plain_handle, samples):
    """JAX inverse_dynamics must USE qdd and stay numerically equal to the numpy
    handle's full-RNEA τ at a nonzero qdd; qdd=None == bias; qdd shifts τ."""
    q, qd, qdd = samples["q"], samples["qd"], samples["u"]
    tau_jax   = np.asarray(jax_handle.inverse_dynamics(q, qd, qdd))
    tau_plain = plain_handle.inverse_dynamics(q, qd, qdd)
    assert np.max(np.abs(tau_jax - tau_plain)) < _TOL, "jax RNEA(qdd) != numpy"
    bias = np.asarray(jax_handle.inverse_dynamics(q, qd, None))
    bias_plain = plain_handle.inverse_dynamics(q, qd)
    assert np.max(np.abs(bias - bias_plain)) < _TOL
    assert np.max(np.abs(tau_jax - bias)) > 1e-2, "jax inverse_dynamics ignored qdd"


def test_inverse_dynamics_honors_qdd_under_jit(jax_handle, plain_handle, samples):
    """qdd flows through jax.jit too."""
    import jax
    q, qd, qdd = samples["q"], samples["qd"], samples["u"]
    f = jax.jit(lambda q, qd, a: jax_handle.inverse_dynamics(q, qd, a))
    tau = np.asarray(f(q, qd, qdd))
    assert np.max(np.abs(tau - plain_handle.inverse_dynamics(q, qd, qdd))) < _TOL


def test_jax_f_ext_parity_vs_numpy(jax_handle, plain_handle, samples):
    """JAX f_ext parity (F8): jax inverse_dynamics/forward_dynamics/aba with an
    explicit f_ext must match the numpy handle (the established f_ext oracle),
    and f_ext=None must equal the no-f_ext path."""
    q, qd, u = samples["q"], samples["qd"], samples["u"]
    NB = jax_handle.num_bodies
    rng = np.random.default_rng(11)
    f_ext = (0.2 * rng.standard_normal((q.shape[0], 6 * NB))).astype(np.float32)
    for name, jax_out, np_out in [
        ("inverse_dynamics",
         jax_handle.inverse_dynamics(q, qd, f_ext=f_ext),
         plain_handle.inverse_dynamics(q, qd, f_ext=f_ext)),
        ("forward_dynamics",
         jax_handle.forward_dynamics(q, qd, u, f_ext=f_ext),
         plain_handle.forward_dynamics(q, qd, u, f_ext=f_ext)),
        ("aba",
         jax_handle.aba(q, qd, u, f_ext=f_ext),
         plain_handle.aba(q, qd, u, f_ext=f_ext)),
    ]:
        assert np.max(np.abs(np.asarray(jax_out) - np_out)) < _TOL, f"jax f_ext {name} != numpy"
    # f_ext=None == no external force (byte-identical path).
    a = np.asarray(jax_handle.forward_dynamics(q, qd, u, f_ext=None))
    b = np.asarray(jax_handle.forward_dynamics(q, qd, u))
    assert np.max(np.abs(a - b)) < 1e-6


def test_inverse_dynamics_accepts_numpy_input(jax_handle, samples):
    """JAX accepts CPU numpy arrays and moves them to GPU transparently;
    the FFI handler sees device-resident buffers either way."""
    q  = samples["q"]
    qd = samples["qd"]
    # Numpy in:
    c1 = np.asarray(jax_handle.inverse_dynamics(q, qd))
    # JAX device array in:
    import jax.numpy as jnp
    c2 = np.asarray(jax_handle.inverse_dynamics(jnp.asarray(q), jnp.asarray(qd)))
    assert np.allclose(c1, c2, atol=1e-6)


def test_inverse_dynamics_under_vmap(jax_handle, samples):
    """jax.vmap doesn't apply here (our handler already batches on axis 0),
    but composing under jit + asarray should be fine."""
    import jax
    @jax.jit
    def f(q, qd):
        # Add a noop transformation around the FFI call to test that the
        # call slots into a larger JIT graph.
        return jax_handle.inverse_dynamics(q, qd) + 0.0
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


def test_end_effector_pose_eager_matches_plain(jax_handle, plain_handle, samples):
    e_jax   = np.asarray(jax_handle.end_effector_pose(samples["q"]))
    e_plain = plain_handle.end_effector_pose(samples["q"])
    assert e_jax.shape == e_plain.shape
    assert np.max(np.abs(e_jax - e_plain)) < _TOL


def test_end_effector_pose_jit_matches_eager(jax_handle, samples):
    import jax
    @jax.jit
    def f(q): return jax_handle.end_effector_pose(q)
    a = np.asarray(f(samples["q"]))
    b = np.asarray(jax_handle.end_effector_pose(samples["q"]))
    assert np.max(np.abs(a - b)) < 1e-6


def test_end_effector_pose_gradient_eager_matches_plain(jax_handle, plain_handle, samples):
    g_jax   = np.asarray(jax_handle.end_effector_pose_gradient(samples["q"]))
    g_plain = plain_handle.end_effector_pose_gradient(samples["q"])
    assert g_jax.shape == g_plain.shape
    assert np.max(np.abs(g_jax - g_plain)) < _TOL


def test_end_effector_pose_gradient_jit_matches_eager(jax_handle, samples):
    import jax
    @jax.jit
    def f(q): return jax_handle.end_effector_pose_gradient(q)
    a = np.asarray(f(samples["q"]))
    b = np.asarray(jax_handle.end_effector_pose_gradient(samples["q"]))
    assert np.max(np.abs(a - b)) < 1e-6


def test_end_effector_pose_hessian_eager_matches_plain(jax_handle, plain_handle, samples):
    h_jax   = np.asarray(jax_handle.end_effector_pose_hessian(samples["q"]))
    h_plain = plain_handle.end_effector_pose_hessian(samples["q"])
    assert h_jax.shape == h_plain.shape
    assert np.max(np.abs(h_jax - h_plain)) < _TOL


def test_end_effector_pose_hessian_jit_matches_eager(jax_handle, samples):
    import jax
    @jax.jit
    def f(q): return jax_handle.end_effector_pose_hessian(q)
    a = np.asarray(f(samples["q"]))
    b = np.asarray(jax_handle.end_effector_pose_hessian(samples["q"]))
    assert np.max(np.abs(a - b)) < 1e-6


def test_inverse_dynamics_gradient_eager_matches_plain(jax_handle, plain_handle, samples):
    dc_jax   = np.asarray(jax_handle.inverse_dynamics_gradient(samples["q"], samples["qd"]))
    dc_plain = plain_handle.inverse_dynamics_gradient(samples["q"], samples["qd"])
    assert dc_jax.shape == dc_plain.shape
    assert np.max(np.abs(dc_jax - dc_plain)) < _TOL


def test_inverse_dynamics_gradient_jit_matches_eager(jax_handle, samples):
    import jax
    @jax.jit
    def f(q, qd): return jax_handle.inverse_dynamics_gradient(q, qd)
    a = np.asarray(f(samples["q"], samples["qd"]))
    b = np.asarray(jax_handle.inverse_dynamics_gradient(samples["q"], samples["qd"]))
    assert np.max(np.abs(a - b)) < 1e-6


def test_forward_dynamics_grad_eager_matches_plain(jax_handle, plain_handle, samples):
    df_jax   = np.asarray(jax_handle.forward_dynamics_gradient(samples["q"], samples["qd"], samples["u"]))
    df_plain = plain_handle.forward_dynamics_gradient(samples["q"], samples["qd"], samples["u"])
    assert df_jax.shape == df_plain.shape
    assert np.max(np.abs(df_jax - df_plain)) < _TOL


def test_forward_dynamics_grad_jit_matches_eager(jax_handle, samples):
    import jax
    @jax.jit
    def f(q, qd, u): return jax_handle.forward_dynamics_gradient(q, qd, u)
    a = np.asarray(f(samples["q"], samples["qd"], samples["u"]))
    b = np.asarray(jax_handle.forward_dynamics_gradient(samples["q"], samples["qd"], samples["u"]))
    assert np.max(np.abs(a - b)) < 1e-6


def test_idsva_so_eager_matches_plain(jax_handle, plain_handle, samples):
    so_jax   = tuple(np.asarray(t) for t in jax_handle.idsva_so(samples["q"], samples["qd"]))
    so_plain = plain_handle.idsva_so(samples["q"], samples["qd"])
    assert len(so_jax) == 4 == len(so_plain)
    for i, (a, b) in enumerate(zip(so_jax, so_plain)):
        assert a.shape == b.shape, f"idsva_so tuple[{i}] shape mismatch"
        assert np.max(np.abs(a - b)) < _TOL, f"idsva_so tuple[{i}] disagrees"


def test_idsva_so_jit_matches_eager(jax_handle, samples):
    import jax
    @jax.jit
    def f(q, qd): return jax_handle.idsva_so(q, qd)
    aj = tuple(np.asarray(t) for t in f(samples["q"], samples["qd"]))
    be = tuple(np.asarray(t) for t in jax_handle.idsva_so(samples["q"], samples["qd"]))
    for a, b in zip(aj, be):
        assert np.max(np.abs(a - b)) < 1e-6


def test_fdsva_so_eager_matches_plain(jax_handle, plain_handle, samples):
    so_jax   = tuple(np.asarray(t) for t in jax_handle.fdsva_so(samples["q"], samples["qd"], samples["u"]))
    so_plain = plain_handle.fdsva_so(samples["q"], samples["qd"], samples["u"])
    assert len(so_jax) == 4 == len(so_plain)
    for i, (a, b) in enumerate(zip(so_jax, so_plain)):
        assert a.shape == b.shape, f"fdsva_so tuple[{i}] shape mismatch"
        assert np.max(np.abs(a - b)) < _TOL, f"fdsva_so tuple[{i}] disagrees"


def test_fdsva_so_jit_matches_eager(jax_handle, samples):
    import jax
    @jax.jit
    def f(q, qd, u): return jax_handle.fdsva_so(q, qd, u)
    aj = tuple(np.asarray(t) for t in f(samples["q"], samples["qd"], samples["u"]))
    be = tuple(np.asarray(t) for t in jax_handle.fdsva_so(samples["q"], samples["qd"], samples["u"]))
    for a, b in zip(aj, be):
        assert np.max(np.abs(a - b)) < 1e-6


_INTEGRATOR_TYPES = ("euler", "semi_implicit_euler", "midpoint", "rk3", "rk4")


@pytest.mark.parametrize("it_name", _INTEGRATOR_TYPES)
def test_integrator_eager_matches_plain(jax_handle, plain_handle, samples, it_name):
    dt = 0.01
    x_jax   = np.asarray(jax_handle.integrator(samples["q"], samples["qd"], samples["u"], dt,
                                               integrator_type=it_name))
    x_plain = plain_handle.integrator(samples["q"], samples["qd"], samples["u"], dt,
                                      integrator_type=it_name)
    assert x_jax.shape == x_plain.shape
    assert np.max(np.abs(x_jax - x_plain)) < _TOL


@pytest.mark.parametrize("it_name", _INTEGRATOR_TYPES)
def test_integrator_gradient_eager_matches_plain(jax_handle, plain_handle, samples, it_name):
    dt = 0.01
    g_jax   = np.asarray(jax_handle.integrator_gradient(samples["q"], samples["qd"], samples["u"], dt,
                                                        integrator_type=it_name))
    g_plain = plain_handle.integrator_gradient(samples["q"], samples["qd"], samples["u"], dt,
                                               integrator_type=it_name)
    assert g_jax.shape == g_plain.shape
    assert np.max(np.abs(g_jax - g_plain)) < _TOL


def test_integrator_jit_matches_eager(jax_handle, samples):
    import jax
    @jax.jit
    def f(q, qd, u):
        return jax_handle.integrator(q, qd, u, 0.01, integrator_type="rk4")
    a = np.asarray(f(samples["q"], samples["qd"], samples["u"]))
    b = np.asarray(jax_handle.integrator(samples["q"], samples["qd"], samples["u"], 0.01,
                                         integrator_type="rk4"))
    assert np.max(np.abs(a - b)) < 1e-6


def test_integrator_gradient_jit_matches_eager(jax_handle, samples):
    import jax
    @jax.jit
    def f(q, qd, u):
        return jax_handle.integrator_gradient(q, qd, u, 0.01, integrator_type="euler")
    a = np.asarray(f(samples["q"], samples["qd"], samples["u"]))
    b = np.asarray(jax_handle.integrator_gradient(samples["q"], samples["qd"], samples["u"], 0.01,
                                                  integrator_type="euler"))
    assert np.max(np.abs(a - b)) < 1e-6


def test_all_methods_jit(jax_handle, samples):
    """Every method must slot into a single jax.jit graph."""
    import jax
    @jax.jit
    def f(q, qd, u):
        c    = jax_handle.inverse_dynamics(q, qd)
        qdd1 = jax_handle.forward_dynamics(q, qd, u)
        qdd2 = jax_handle.aba(q, qd, u)
        Minv = jax_handle.minv(q)
        M    = jax_handle.crba(q)
        ee   = jax_handle.end_effector_pose(q)
        dee  = jax_handle.end_effector_pose_gradient(q)
        d2ee = jax_handle.end_effector_pose_hessian(q)
        dc   = jax_handle.inverse_dynamics_gradient(q, qd)
        df   = jax_handle.forward_dynamics_gradient(q, qd, u)
        return c, qdd1, qdd2, Minv, M, ee, dee, d2ee, dc, df
    outs = f(samples["q"], samples["qd"], samples["u"])
    for x in outs:
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
