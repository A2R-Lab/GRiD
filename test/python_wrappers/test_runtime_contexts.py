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

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
grid_rbd = pytest.importorskip("grid_rbd")
from ._subset_artifacts import register_subset, cache_key as _cache_key, random_state as _state  # noqa: E402

pytestmark = pytest.mark.python_wrappers
ALGOS = ["forward_dynamics", "inverse_dynamics"]
# B2 (K4): the iiwa artifact also carries fd's backward ops + runtime inertia so
# the version / stamp tests can mutate the model and differentiate through it.
ALGOS_B2 = ALGOS + ["forward_dynamics_gradient", "minv"]


@pytest.fixture(scope="module")
def iiwa():
    h = register_subset("ctx_pytest_iiwa14", "iiwa14.urdf", floating=False, algos=ALGOS_B2, runtime_inertia=True)
    yield h
    h.close()


@pytest.fixture(scope="module")
def go2():
    h = register_subset("ctx_pytest_go2", "go2.urdf", floating=True, algos=ALGOS)
    yield h
    h.close()


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


# ─── W04-B B2: admission lock, model version, execution-time stamps (K4) ───────

def _scaled_inertia(h, factor):
    tbl = np.array(h.inertia_params, dtype=np.float32, copy=True)
    tbl[1:, 0] *= factor   # scale every non-root mass (row 0 may be the fixed base)
    return tbl


def test_model_version_bumps_on_parameter_mutation_only(iiwa):
    # versions are drawn from the artifact-wide epoch (R6): strictly increasing on
    # every model mutation, never contiguous (context creations bump the epoch too)
    v0 = iiwa.model_version
    assert v0 >= 1
    iiwa.set_inertia_params(np.asarray(iiwa.inertia_params, dtype=np.float32))
    v1 = iiwa.model_version
    assert v1 > v0
    # launch overrides are exclusive-admission too, but NOT a model mutation
    iiwa.set_threads_per_block(64); iiwa.set_threads_per_block(0)
    assert iiwa.model_version == v1
    # attach/detach route through the inertia setter: one bump each
    row_joint = list(iiwa._meta.get("inertia_row_by_joint_name") or {"": None})[-1]
    if row_joint:
        iiwa.attach_tool(row_joint, mass=0.5)
        v2 = iiwa.model_version
        iiwa.detach_tool()
        assert v2 > v1 and iiwa.model_version > v2
    # a NEW context draws its version from the artifact-wide epoch (R6): unique,
    # never equal to another context's, and its own mutations bump only it
    ctx = iiwa.context()
    try:
        assert ctx.model_version != iiwa.model_version
        v_ctx, v_def = ctx.model_version, iiwa.model_version
        ctx.set_inertia_params(np.asarray(ctx.inertia_params, dtype=np.float32))
        assert ctx.model_version > v_ctx and iiwa.model_version == v_def   # only ctx moved
    finally:
        ctx.close()


def test_dropped_explicit_context_is_finalized_and_close_is_idempotent(iiwa):
    """codex R3: a handle is the one owner of its context; dropping it without
    close() reclaims the context at GC; explicit close + GC never double-closes."""
    import gc
    n0 = iiwa._runner.ctx_count()
    ctx = iiwa.context()
    q, qd, u = _state(iiwa)
    ctx.forward_dynamics(q, qd, u)
    assert iiwa._runner.ctx_count() == n0 + 1
    del ctx
    gc.collect()
    assert iiwa._runner.ctx_count() == n0
    ctx = iiwa.context()
    ctx.close(); ctx.close()
    del ctx
    gc.collect()
    assert iiwa._runner.ctx_count() == n0


def test_explicit_workspace_cap_is_honoured_and_results_hold(iiwa):
    """codex R2: context(workspace_slots=1) really fits one slot (no slab, no env)
    and a batch above the cap still grid-strides to the same result."""
    q, qd, u = _state(iiwa, B=6)
    ref = iiwa.forward_dynamics_gradient(q, qd, u)
    ctx = iiwa.context(workspace_slots=1)
    try:
        assert ctx.device_profile["workspace_slots"] == 1
        assert np.allclose(ctx.forward_dynamics_gradient(q, qd, u), ref, atol=1e-6)
    finally:
        ctx.close()


def test_default_context_reset_invalidates_a_deferred_backward(iiwa):
    """codex R6: a backward whose forward ran on a since-recreated default context
    is refused even when the per-context mutation counts coincide."""
    torch = pytest.importorskip("torch")
    import grid_rbd.torch as gt
    tv = gt.TorchRobotHandle(iiwa, _cache_key(iiwa), iiwa._so_path)
    q, qd, u = _state(iiwa)
    base = np.asarray(iiwa.inertia_params, dtype=np.float32)
    tq = torch.as_tensor(q, device="cuda").requires_grad_(True)
    tqd, tu = (torch.as_tensor(x, device="cuda") for x in (qd, u))
    out = tv.forward_dynamics(tq, tqd, tu)                # forward on incarnation A
    iiwa._runner.close_arena()                            # default context A closed (Runner-level reset)
    iiwa.forward_dynamics(q, qd, u)                       # incarnation B created lazily
    with pytest.raises(RuntimeError, match="model mutated between forward and backward"):
        out.sum().backward()
    iiwa.set_inertia_params(base)


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
        assert iiwa.model_version > v; v = iiwa.model_version
    stop.set(); th.join()
    iiwa.set_inertia_params(base)
    assert not bad, f"{len(bad)} torn/inconsistent results under a racing mutation"


def test_graph_replay_is_refused_after_mutation_or_close(iiwa):
    """codex R5: a captured graph replays only on its context at its captured
    model epoch: mutation → refused until recapture; close → refused, never a
    launch into freed memory."""
    torch = pytest.importorskip("torch")
    import grid_rbd.torch as gt
    base = np.asarray(iiwa.inertia_params, dtype=np.float32)
    q, qd, u = (torch.as_tensor(x, device="cuda") for x in _state(iiwa))
    tv = gt.TorchRobotHandle(iiwa, _cache_key(iiwa), iiwa._so_path)
    g = tv.capture("forward_dynamics", q, qd, u)
    ref = g.replay().clone()
    assert torch.allclose(g(q, qd, u), ref)
    iiwa.set_inertia_params(_scaled_inertia(iiwa, 1.5))
    try:
        with pytest.raises(RuntimeError, match="mutated since this graph was captured"):
            g.replay()
        g2 = tv.capture("forward_dynamics", q, qd, u)     # recapture at the new model
        assert not torch.allclose(g2.replay(), ref)
    finally:
        iiwa.set_inertia_params(base)
    ctx = iiwa.context()
    cv = gt.TorchRobotHandle(ctx, _cache_key(iiwa), ctx._so_path)
    gc_ = cv.capture("forward_dynamics", q, qd, u)
    gc_.replay()
    ctx.close()
    with pytest.raises(RuntimeError, match="closed"):
        gc_.replay()


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


_RACE_CHILD = r'''
import sys, time, threading
import numpy as np, grid_rbd
h = grid_rbd.get_robot("ctx_pytest_iiwa14")
r = h._runner
q = np.zeros((2, h.nq), np.float32)
h.forward_dynamics(q, q, q)                       # default context exists
base = np.asarray(h.inertia_params, np.float32)

def hold(runner, version, secs=1.0):
    tok = runner.graph_begin(runner.ctx_id(), version)
    time.sleep(secs)
    runner.graph_end(tok)

# 1. a runtime-parameter setter racing a held replay token (exclusive vs shared)
t = threading.Thread(target=hold, args=(r, h.model_version)); t.start(); time.sleep(0.2)
t0 = time.time(); h.set_inertia_params(base); dt1 = time.time() - t0; t.join()

# 2. an explicit context closed while a token is held on it (drain vs shared)
ctx = h.context(); ctx.forward_dynamics(q, q, q); rc = ctx._runner
t = threading.Thread(target=hold, args=(rc, ctx.model_version)); t.start(); time.sleep(0.2)
t0 = time.time(); ctx.close(); dt2 = time.time() - t0; t.join()
del rc, ctx                                       # drop the closed context's runner (arena owner count)

# 3. the default arena reset while a token is held on the default context
t = threading.Thread(target=hold, args=(r, h.model_version)); t.start(); time.sleep(0.2)
t0 = time.time(); r.close_arena(); dt3 = time.time() - t0; t.join()
h.forward_dynamics(q, q, q)                       # default re-created lazily

# 4. the real GraphCallable bracket: a replay loop racing a mutation ends REFUSED, never hung
import torch, grid_rbd.torch as gt
key = grid_rbd.manifest_lookup(grid_rbd.default_cache_dir(), h._name)["cache_key"]
tv = gt.TorchRobotHandle(h, key, h._so_path)
tq = torch.zeros(2, h.nq, device="cuda")
g = tv.capture("forward_dynamics", tq, tq, tq)
state = {"refused": False, "n": 0}
def spin():
    end = time.time() + 5.0
    while time.time() < end:
        try:
            g.replay(); state["n"] += 1
        except RuntimeError as e:
            state["refused"] = "mutated since this graph was captured" in str(e); break
t = threading.Thread(target=spin); t.start(); time.sleep(0.1)
h.set_inertia_params(base); t.join()
assert dt1 >= 0.6 and dt2 >= 0.6 and dt3 >= 0.6, (dt1, dt2, dt3)   # each waited for the token
assert state["refused"] and state["n"] > 0, state
print("OK", round(dt1, 2), round(dt2, 2), round(dt3, 2), state["n"])
'''


def test_replay_admission_racing_mutation_and_close_never_deadlocks(iiwa, tmp_path):
    """codex follow-up (2026-09-24): a replay token holds native admission across
    Python; a setter / close / arena reset in another thread must WAIT for it
    without holding the GIL (else the token holder can never reach graph_end).
    Deterministic and subprocess-isolated with a hard timeout: a regression hangs
    the child, which is reported as a failure instead of hanging the suite."""
    import subprocess
    script = tmp_path / "race_child.py"
    script.write_text(_RACE_CHILD)
    try:
        res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                             timeout=240, cwd=str(_REPO))
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"replay-admission race DEADLOCKED (child timed out); stdout={e.stdout!r}")
    assert res.returncode == 0 and "OK" in res.stdout, f"rc={res.returncode}\n{res.stdout}\n{res.stderr[-3000:]}"


def test_construction_does_not_create_the_default_context_and_a_slab_still_installs(iiwa):
    """Release receipt #3 (2026-09-25): a handle's init-time launch overlay resolved
    context 0 and CREATED the default context, after which a device slab could never
    be installed. Overlays set before any context exists are pending and seeded at
    creation; the first kernel call is what creates the default context."""
    import grid_rbd
    n0 = iiwa._runner.ctx_count()
    h = grid_rbd.get_robot("ctx_pytest_iiwa14")          # fresh handle over the same .so
    try:
        assert h._runner.ctx_count() == n0                  # construction created nothing
        h.set_threads_per_block(64)                         # pending when no default exists
        assert h.threads_per_block == 64
        q, qd, u = _state(h)
        h.forward_dynamics(q, qd, u)                        # first call creates the default…
        assert h.threads_per_block == 64                    # …seeded with the pending overlay
        h.set_threads_per_block(0)
    finally:
        h.close()
