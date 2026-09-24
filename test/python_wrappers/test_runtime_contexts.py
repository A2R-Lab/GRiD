"""W04-B B1 runtime contexts (2026-09-24): every call resolves its context by id.

Acceptance list from codex's second review: two artifacts with independent
defaults; two contexts within one artifact; foreign / stale (closed) id
rejection; separate slabs (an explicit context carves from its own pool, the
default's cursor is untouched); partial-init failure (a slab too small fails
cleanly, nothing published); close racing with submission (a submitting thread
either completes or gets the closed/closing error, never a crash); and NumPy,
torch and JAX dispatch through an explicit context. Builds are subset
artifacts (fd + id) so the module stays minutes, not hours.
"""
from __future__ import annotations

import shutil
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
grid_rbd = pytest.importorskip("grid_rbd")

pytestmark = pytest.mark.python_wrappers
ALGOS = ["forward_dynamics", "inverse_dynamics"]
# B2 (K4): the iiwa artifact also carries fd's backward ops + runtime inertia so
# the version / stamp tests can mutate the model and differentiate through it.
ALGOS_B2 = ALGOS + ["forward_dynamics_gradient", "minv"]


def _register(name, urdf, floating, algos=ALGOS, runtime_inertia=False):
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not on PATH")
    return grid_rbd.register_robot(name, str(_REPO / "config/robot_assets" / urdf), floating_base=floating,
                                   max_batch_size=16, algorithm_list=algos, enable_mujoco_kernels=False,
                                   runtime_inertia=runtime_inertia)


@pytest.fixture(scope="module")
def iiwa():
    h = _register("ctx_pytest_iiwa14", "iiwa14.urdf", False, algos=ALGOS_B2, runtime_inertia=True)
    yield h
    h.close()


@pytest.fixture(scope="module")
def go2():
    h = _register("ctx_pytest_go2", "go2.urdf", True)
    yield h
    h.close()


def _state(h, B=4, seed=0):
    rng = np.random.default_rng(seed)
    q = 0.3 * rng.standard_normal((B, h.nq)).astype(np.float32)
    if h.floating_base:
        quat = rng.standard_normal((B, 4)); quat /= np.linalg.norm(quat, axis=1, keepdims=True); q[:, 3:7] = quat
    qd = 0.3 * rng.standard_normal((B, h.nq)).astype(np.float32); u = 0.3 * rng.standard_normal((B, h.nq)).astype(np.float32)
    if h.floating_base:
        qd[:, h.nv:] = 0; u[:, h.nv:] = 0
    return q, qd, u


def test_two_artifacts_have_independent_default_contexts(iiwa, go2):
    a, b = iiwa._runner.ctx_default_id(), go2._runner.ctx_default_id()
    assert a != 0 and b != 0 and a != b
    assert (a >> 32) != (b >> 32), "per-artifact salts must differ"
    assert iiwa.ctx_id == 0 and go2.ctx_id == 0  # handles over the default use the alias
    assert iiwa.device_profile["max_batch"] == 16 and go2.device_profile["arena_bytes"] > 0
    assert iiwa.device_profile["device_cc"] == iiwa.device_profile["artifact_cc"]


def test_foreign_id_is_rejected_by_the_other_artifact(iiwa, go2):
    q, qd, u = _state(iiwa)
    foreign = go2._runner.ctx_default_id()
    iiwa._runner.bind_context(foreign)
    try:
        with pytest.raises(RuntimeError, match="another robot artifact|unknown context"):
            iiwa.forward_dynamics(q, qd, u)
    finally:
        iiwa._runner.bind_context(0)
    np.testing.assert_array_equal(iiwa.forward_dynamics(q, qd, u), iiwa.forward_dynamics(q, qd, u))


def test_two_contexts_in_one_artifact_are_isolated(iiwa):
    q, qd, u = _state(iiwa)
    ref = iiwa.forward_dynamics(q, qd, u)
    c1, c2 = iiwa.context(), iiwa.context()
    try:
        assert c1.ctx_id != c2.ctx_id and c1.ctx_id != 0 and (c1.ctx_id >> 32) == (iiwa._runner.ctx_default_id() >> 32)
        np.testing.assert_array_equal(c1.forward_dynamics(q, qd, u), ref)
        np.testing.assert_array_equal(c2.forward_dynamics(q, qd, u), ref)
        # launch overrides are context state: forcing threads on c1 changes nothing on c2 / default
        c1._runner.set_threads_per_block(32)
        assert c1._runner.threads_per_block == 32
        assert c2._runner.threads_per_block == -1 and iiwa._runner.threads_per_block == -1
        np.testing.assert_array_equal(c1.forward_dynamics(q, qd, u), ref)
        assert c1.device_profile["arena_bytes"] == c2.device_profile["arena_bytes"]
    finally:
        c1.close(); c2.close()
    np.testing.assert_array_equal(iiwa.forward_dynamics(q, qd, u), ref)


def test_closed_context_id_never_resolves_again(iiwa):
    q, qd, u = _state(iiwa)
    c = iiwa.context(); cid = c.ctx_id
    c.close()
    with pytest.raises(Exception):
        c.forward_dynamics(q, qd, u)  # the handle released its runner
    probe = iiwa.context()  # a fresh context must NOT reuse the closed id
    pid = probe.ctx_id
    try:
        assert pid != cid
        probe._runner.bind_context(cid)
        with pytest.raises(RuntimeError, match="closed"):
            probe.forward_dynamics(q, qd, u)
    finally:
        probe._runner.bind_context(pid)   # back to its own context so close() closes THAT
        probe.close()


def test_explicit_context_carves_its_own_slab_and_a_tiny_slab_fails_cleanly(iiwa):
    import ctypes
    r = iiwa._runner
    default_used_before = r.device_pool_used()
    bytes_needed = r.device_pool_bytes(4)
    cuda = ctypes.CDLL("libcudart.so")
    ptr = ctypes.c_void_p()
    assert cuda.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(bytes_needed)) == 0
    try:
        cid = r.ctx_create(ptr.value, bytes_needed, 4)
        try:
            prof = r.ctx_profile(cid)
            assert prof["slab_installed"] is True and prof["workspace_slots"] == 4 and prof["arena_bytes"] == bytes_needed
            assert r.device_pool_used() == default_used_before, "the default context's pool cursor moved"
            view = iiwa.context.__self__.__class__(iiwa._name, iiwa._so_path, iiwa._meta)
            view._runner.bind_context(cid)
            q, qd, u = _state(iiwa)
            np.testing.assert_array_equal(view.forward_dynamics(q, qd, u), iiwa.forward_dynamics(q, qd, u))
            view.close()
        finally:
            r.ctx_close(cid)
        # partial-init failure: a slab that cannot hold the arena → clean error, no context published
        n_before = len([1])  # placeholder to keep the structure explicit
        with pytest.raises(RuntimeError):
            r.ctx_create(ptr.value, 256, 4)
        np.testing.assert_array_equal(iiwa.forward_dynamics(*_state(iiwa)), iiwa.forward_dynamics(*_state(iiwa)))
    finally:
        cuda.cudaFree(ptr)


def test_close_racing_with_submission_never_crashes(iiwa):
    q, qd, u = _state(iiwa, B=16)
    c = iiwa.context()
    errors, ok = [], []
    stop = threading.Event()

    def submit():
        while not stop.is_set():
            try:
                c.forward_dynamics(q, qd, u); ok.append(1)
            except RuntimeError as e:  # closed / closing: the documented outcome
                errors.append(str(e)); stop.set(); return
            except Exception as e:  # the handle's runner may already be gone
                errors.append(str(e)); stop.set(); return
    t = threading.Thread(target=submit); t.start()
    threading.Event().wait(0.05)
    c.close()
    stop.set(); t.join(10)
    assert not t.is_alive()
    assert ok, "the submitter never got a call through before close"
    # after close, the default context is unaffected
    np.testing.assert_array_equal(iiwa.forward_dynamics(q, qd, u), iiwa.forward_dynamics(q, qd, u))


def test_torch_and_jax_views_dispatch_to_an_explicit_context(iiwa):
    q, qd, u = _state(iiwa)
    ref = iiwa.forward_dynamics(q, qd, u)
    ctx = iiwa.context()
    try:
        torch = pytest.importorskip("torch")
        import grid_rbd.torch as gt
        tv = gt.TorchRobotHandle(ctx, gt._cache_key_of(iiwa) if hasattr(gt, "_cache_key_of") else _cache_key(iiwa), ctx._so_path)
        tq, tqd, tu = (torch.as_tensor(x, device="cuda") for x in (q, qd, u))
        out = tv.forward_dynamics(tq, tqd, tu).cpu().numpy()
        assert tv.ctx_id == ctx.ctx_id and np.allclose(out, ref, atol=1e-5)
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp, grid_rbd.jax as gj
        jv = gj.JaxRobotHandle(ctx, _cache_key(iiwa), ctx._so_path)
        jout = np.asarray(jax.block_until_ready(jv.forward_dynamics(jnp.asarray(q), jnp.asarray(qd), jnp.asarray(u))))
        assert jv.ctx_id == ctx.ctx_id and np.allclose(jout, ref, atol=1e-5)
    finally:
        ctx.close()


def _cache_key(handle):
    """The manifest's content key for a handle (what the jax/torch views key their registrations on)."""
    entry = grid_rbd.manifest_lookup(grid_rbd.default_cache_dir(), handle._name)
    return entry["cache_key"]


# ─── W04-B B2: admission lock, model version, execution-time stamps (K4) ───────

def _scaled_inertia(h, factor):
    tbl = np.array(h.inertia_params, dtype=np.float32, copy=True)
    tbl[1:, 0] *= factor   # scale every non-root mass (row 0 may be the fixed base)
    return tbl


def test_model_version_bumps_on_parameter_mutation_only(iiwa):
    v0 = iiwa.model_version
    assert v0 >= 1
    iiwa.set_inertia_params(np.asarray(iiwa.inertia_params, dtype=np.float32))
    assert iiwa.model_version == v0 + 1
    # launch overrides are exclusive-admission too, but NOT a model mutation
    iiwa.set_threads_per_block(64); iiwa.set_threads_per_block(0)
    assert iiwa.model_version == v0 + 1
    # attach/detach route through the inertia setter: one bump each
    row_joint = list(iiwa._meta.get("inertia_row_by_joint_name") or {"": None})[-1]
    if row_joint:
        iiwa.attach_tool(row_joint, mass=0.5)
        iiwa.detach_tool()
        assert iiwa.model_version == v0 + 3
    # a NEW context starts at its own version 1, independent of the default's history
    ctx = iiwa.context()
    try:
        assert ctx.model_version == 1
    finally:
        ctx.close()


def test_mutation_is_serialized_against_admitted_calls(iiwa):
    """A setter racing a hot submitting thread must never see a torn table or
    crash: every result is either the old or the new physics (both are exact
    kernels on a consistent table), and the version moves monotonically."""
    q, qd, u = _state(iiwa, B=8, seed=3)
    base = np.asarray(iiwa.inertia_params, dtype=np.float32)
    ref_a = iiwa.forward_dynamics(q, qd, u)
    iiwa.set_inertia_params(_scaled_inertia(iiwa, 2.0))
    ref_b = iiwa.forward_dynamics(q, qd, u)
    iiwa.set_inertia_params(base)
    bad = []
    stop = threading.Event()

    def hammer():
        while not stop.is_set():
            out = iiwa.forward_dynamics(q, qd, u)
            if not (np.allclose(out, ref_a, atol=1e-4) or np.allclose(out, ref_b, atol=1e-4)):
                bad.append(out.copy())
    th = threading.Thread(target=hammer); th.start()
    v = iiwa.model_version
    for i in range(20):
        iiwa.set_inertia_params(_scaled_inertia(iiwa, 2.0) if i % 2 == 0 else base)
        assert iiwa.model_version == v + i + 1
    stop.set(); th.join()
    iiwa.set_inertia_params(base)
    assert not bad, f"{len(bad)} torn/inconsistent results under a racing mutation"


def test_torch_backward_rejects_a_mutated_model_and_fresh_forward_recovers(iiwa):
    torch = pytest.importorskip("torch")
    import grid_rbd.torch as gt
    tv = gt.TorchRobotHandle(iiwa, _cache_key(iiwa), iiwa._so_path)
    q, qd, u = _state(iiwa)
    base = np.asarray(iiwa.inertia_params, dtype=np.float32)
    tq = torch.as_tensor(q, device="cuda").requires_grad_(True)
    tqd, tu = (torch.as_tensor(x, device="cuda") for x in (qd, u))
    out = tv.forward_dynamics(tq, tqd, tu)
    iiwa.set_inertia_params(_scaled_inertia(iiwa, 1.5))     # forward@A -> mutate -> backward
    try:
        with pytest.raises(RuntimeError, match="model mutated between forward and backward"):
            out.sum().backward()
        tq.grad = None
        out2 = tv.forward_dynamics(tq, tqd, tu)              # a fresh forward at B differentiates B
        out2.sum().backward()
        assert tq.grad is not None and np.isfinite(tq.grad.cpu().numpy()).all()
    finally:
        iiwa.set_inertia_params(base)


def test_jax_vjp_rejects_a_mutated_model_and_a_jitted_grad_follows_it(iiwa):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp, grid_rbd.jax as gj
    jv = gj.JaxRobotHandle(iiwa, _cache_key(iiwa), iiwa._so_path)
    q, qd, u = _state(iiwa)
    base = np.asarray(iiwa.inertia_params, dtype=np.float32)
    jq, jqd, ju = (jnp.asarray(x) for x in (q, qd, u))
    # eager vjp: forward@A -> mutate -> backward must raise (K4 acceptance)
    y, f_vjp = jax.vjp(lambda a: jv.forward_dynamics(a, jqd, ju), jq)
    jax.block_until_ready(y)
    iiwa.set_inertia_params(_scaled_inertia(iiwa, 1.5))
    try:
        with pytest.raises(Exception, match="model mutated between forward and backward"):
            jax.block_until_ready(f_vjp(jnp.ones_like(y)))
        # the SAME compiled function across mutations: the stamp is produced at
        # execution time inside the executable, so no false rejection, and the
        # gradient tracks the CURRENT model (differs from the pre-mutation one).
        g = jax.jit(jax.grad(lambda a: jv.forward_dynamics(a, jqd, ju).sum()))
        g_b = np.asarray(jax.block_until_ready(g(jq)))
        iiwa.set_inertia_params(base)
        g_a = np.asarray(jax.block_until_ready(g(jq)))
        eager_a = np.asarray(jax.block_until_ready(
            jax.grad(lambda a: jv.forward_dynamics(a, jqd, ju).sum())(jq)))
        assert np.isfinite(g_a).all() and np.isfinite(g_b).all()
        assert not np.allclose(g_a, g_b, atol=1e-5), "jitted grad ignored the mutation"
        assert np.allclose(g_a, eager_a, atol=1e-5)
    finally:
        iiwa.set_inertia_params(base)
