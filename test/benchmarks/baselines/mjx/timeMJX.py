#!/usr/bin/env python3
"""Time MJX algorithms for one robot.

Prints results in the same format as timeGRiD so parse_grid_output can be reused.
Available algorithms: id (inverse), fd (forward), ee_pose (kinematics), id_du (Jacobian).
All others (minv, aba, crba, fd_du, idsva_so, fdsva_so) are null for MJX.

Usage:
    python timeMJX.py <mjcf_path> [T/F] [ee_body_name]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

# Configurable via env var (set by mjx/run.py's --test-iters flag).
TEST_ITERS  = int(os.environ.get("BENCH_TEST_ITERS", "500"))
BATCH_SIZES = [16, 32, 64, 128, 256, 1024]
# Derivative (vmap(jacobian)) timing builds a SEPARATE giant XLA graph per batch
# size; compiling it at N=1024 is intractable (minutes-to-wedged). The competitive
# figures only need up to batch_256, so cap the derivative sweep here (override via
# MJX_DERIV_BATCH_SIZES="16,256"). The value-function sweep still uses BATCH_SIZES.
DERIV_BATCH_SIZES = [int(x) for x in
                     os.environ.get("MJX_DERIV_BATCH_SIZES", "16,32,64,128,256").split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Output helpers (same format as timeGRiD)
# ---------------------------------------------------------------------------

def _print_stats(label: str, n: int, times: np.ndarray) -> None:
    print(
        f"[N:{n}]: {label}: "
        f"Average[{np.mean(times):.4f}us] "
        f"Std Dev [{np.std(times):.4f}us] "
        f"Min [{np.min(times):.4f}us] "
        f"Max [{np.max(times):.4f}us]"
    )


# ---------------------------------------------------------------------------
# JAX timing primitives
#
# Discipline (mirrors what the GRiD benchmark does on the C++ side):
#   1. JIT-compile the function with a representative input. First call is
#      always slow (XLA HLO lowering + ptx); ignore it.
#   2. Run N_WARMUP_PASSES additional warmup calls. These cache device buffers,
#      stabilize the GPU clock, and drain any pending compilation. Ignore.
#   3. Time the next N_ITERS calls. JIT cache is guaranteed warm and shapes are
#      fixed, so any variance reflects the kernel + sync cost, not compilation.
#
# For with-memory timings, the JIT'd function takes raw numpy arrays as inputs
# so that the host->device transfer is included in the timed region. The numpy
# arrays are regenerated per-iter (outside the timer) to defeat device-side
# buffer caching across calls.
# ---------------------------------------------------------------------------

N_WARMUP_PASSES = 3


def _jit_and_warmup(fn, sample_args, n_warmup: int = N_WARMUP_PASSES) -> None:
    """Trigger JIT compile + run warmup passes. Discards results."""
    import jax
    # First call: XLA lowering + GPU compilation. Slow.
    jax.block_until_ready(fn(*sample_args))
    # Additional warmup: GPU clock stabilization + device buffer caching.
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*sample_args))


def _time_device(fn, *args, n_iters: int = TEST_ITERS) -> np.ndarray:
    """Time a JIT-compiled, warmed-up JAX function with args already on device."""
    import jax
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append((time.perf_counter() - t0) * 1e6)
    return np.array(times)


def _time_with_mem(fn, make_args_fn, n_iters: int) -> np.ndarray:
    """Time fn where make_args_fn() produces numpy arrays that get transferred each call.

    Caller must have already JIT'd + warmed up `fn` against a numpy-input sample.
    """
    import jax
    times = []
    for _ in range(n_iters):
        np_args = make_args_fn()           # outside the timer
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*np_args))
        times.append((time.perf_counter() - t0) * 1e6)
    return np.array(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <mjcf_path> [T/F] [ee_body_name]", file=sys.stderr)
        sys.exit(1)

    mjcf_path     = sys.argv[1]
    floating_base = len(sys.argv) > 2 and sys.argv[2].upper() == "T"
    ee_body_name  = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        import jax
        import jax.numpy as jnp
        import mujoco
        import mujoco.mjx as mjx
    except ImportError as e:
        print(f"MJX import failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(mjcf_path)

    # Disable collision detection on every geom. MJX 3.8 rejects
    # certain collision-pair types (e.g. CYLINDER vs BOX/MESH) at put_model
    # time, even though we never call mjx.step()/mjx.collision() — we only
    # use mjx.inverse(), mjx.forward(), mjx.kinematics() which don't need
    # contacts. Zeroing contype/conaffinity disables all geom-geom pair
    # generation so the model loads on any robot MJCF.
    model.geom_contype[:]     = 0
    model.geom_conaffinity[:] = 0

    # Disable the constraint solver too. mjx.forward()/inverse() always invoke
    # solver.solve(); on jax >= 0.10 the (constraint-less) solver path indexes a
    # float32 array (mjx/_src/solver.py _update_constraint) which newer jax rejects
    # ("Indexer must have integer or boolean type ... float32"). With no contacts
    # and no equality/limit constraints there is nothing to solve, and disabling
    # the constraint pass is exactly the unconstrained articulated-body dynamics
    # GRiD computes — so this is both the apples-to-apples comparison and the fix.
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)

    data  = mujoco.MjData(model)
    mx    = mjx.put_model(model)
    dx0   = mjx.put_data(model, data)

    nq, nv = model.nq, model.nv

    # Look up EE body id
    ee_body_id: int | None = None
    if ee_body_name:
        _id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
        if _id >= 0:
            ee_body_id = _id
        else:
            print(f"  [mjx] warning: body '{ee_body_name}' not found", file=sys.stderr)

    # ------------------------------------------------------------------
    # Metadata block (parsed by timing_parser)
    # ------------------------------------------------------------------
    try:
        backend = jax.default_backend()
    except Exception:
        backend = "unknown"

    print("=== BEGIN MJX METADATA ===")
    print(f"mujoco_version: {mujoco.__version__}")
    print(f"jax_version: {jax.__version__}")
    print(f"jax_backend: {backend}")
    print("ID codegen: false")
    print("FD codegen: false")
    print("EE_POSE codegen: false")
    print("ID_DU codegen: false")
    print("FD_DU codegen: false")
    print("=== END MJX METADATA ===")

    # ------------------------------------------------------------------
    # State factory (jnp — on device)
    # ------------------------------------------------------------------
    rng = jax.random.PRNGKey(0)

    def _make_dx_jnp(key):
        k1, k2, k3 = jax.random.split(key, 3)
        qpos = jax.random.normal(k1, (nq,))
        qvel = jax.random.normal(k2, (nv,))
        qacc = jax.random.normal(k3, (nv,))
        if floating_base and nq >= 7:
            quat = qpos[3:7]
            qpos = qpos.at[3:7].set(quat / (jnp.linalg.norm(quat) + 1e-8))
        return dx0.replace(qpos=qpos, qvel=qvel, qacc=qacc)

    dx_single = _make_dx_jnp(rng)

    # ------------------------------------------------------------------
    # GPU warmup — a few forward passes, discarded
    # ------------------------------------------------------------------
    _fd_jit = jax.jit(lambda d: mjx.forward(mx, d))
    for _ in range(5):
        jax.block_until_ready(_fd_jit(dx_single))

    # ------------------------------------------------------------------
    # Single-call timing
    # ------------------------------------------------------------------

    # ID (inverse dynamics)
    try:
        _id_jit = jax.jit(lambda d: mjx.inverse(mx, d))
        _jit_and_warmup(_id_jit, (dx_single,))
        t = _time_device(_id_jit, dx_single)
        print(f"Single Call INVERSE_DYNAMICS {np.median(t):.4f}us")
    except Exception as e:
        print(f"# Single Call INVERSE_DYNAMICS skipped: {e}", file=sys.stderr)

    # FD (forward dynamics)
    try:
        _jit_and_warmup(_fd_jit, (dx_single,))
        t = _time_device(_fd_jit, dx_single)
        print(f"Single Call FORWARD_DYNAMICS {np.median(t):.4f}us")
    except Exception as e:
        print(f"# Single Call FORWARD_DYNAMICS skipped: {e}", file=sys.stderr)

    # EE_POSE (kinematics)
    try:
        _ee_jit = jax.jit(lambda d: mjx.kinematics(mx, d))
        _jit_and_warmup(_ee_jit, (dx_single,))
        t = _time_device(_ee_jit, dx_single)
        print(f"Single Call END_EFFECTOR_POSE {np.median(t):.4f}us")
    except Exception as e:
        print(f"# Single Call END_EFFECTOR_POSE skipped: {e}", file=sys.stderr)

    # ID_DU (Jacobian of inverse dynamics w.r.t. q, v, a)
    try:
        @jax.jit
        def _id_du_jit(d):
            return jax.jacobian(
                lambda qpos, qvel, qacc: mjx.inverse(
                    mx, d.replace(qpos=qpos, qvel=qvel, qacc=qacc)
                ).qfrc_inverse,
                argnums=(0, 1, 2),
            )(d.qpos, d.qvel, d.qacc)

        _jit_and_warmup(_id_du_jit, (dx_single,))
        t = _time_device(_id_du_jit, dx_single, n_iters=max(1, TEST_ITERS // 10))
        print(f"Single Call INVERSE_DYNAMICS_GRADIENT {np.median(t):.4f}us")
    except Exception as e:
        print(f"# Single Call INVERSE_DYNAMICS_GRADIENT skipped: {e}", file=sys.stderr)

    # FD_DU (Jacobian of forward dynamics qacc w.r.t. q, v, qfrc_applied).
    # d(qacc)/d(q,v,tau) via jax autodiff through mjx.forward — the mjx analogue of
    # GRiD's analytic forward_dynamics_gradient (differentiate w.r.t. qfrc_applied,
    # the generalized force = GRiD's tau, NOT ctrl which goes through the actuator).
    try:
        @jax.jit
        def _fd_du_jit(d):
            return jax.jacobian(
                lambda qpos, qvel, qfrc: mjx.forward(
                    mx, d.replace(qpos=qpos, qvel=qvel, qfrc_applied=qfrc)
                ).qacc,
                argnums=(0, 1, 2),
            )(d.qpos, d.qvel, d.qfrc_applied)

        _jit_and_warmup(_fd_du_jit, (dx_single,))
        t = _time_device(_fd_du_jit, dx_single, n_iters=max(1, TEST_ITERS // 10))
        print(f"Single Call FORWARD_DYNAMICS_GRADIENT {np.median(t):.4f}us")
    except Exception as e:
        print(f"# Single Call FORWARD_DYNAMICS_GRADIENT skipped: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Batch timing via vmap
    # ------------------------------------------------------------------
    for N in BATCH_SIZES:
        keys      = jax.random.split(rng, N)
        dx_batch  = jax.vmap(_make_dx_jnp)(keys)   # on device

        # Pre-build vmapped JIT functions
        _batch_id_fn = jax.jit(jax.vmap(lambda d: mjx.inverse(mx, d)))
        _batch_fd_fn = jax.jit(jax.vmap(lambda d: mjx.forward(mx, d)))
        _batch_ee_fn = jax.jit(jax.vmap(lambda d: mjx.kinematics(mx, d)))

        # JIT compile + warmup all three (compile + N_WARMUP_PASSES extra passes)
        for _fn in (_batch_id_fn, _batch_fd_fn, _batch_ee_fn):
            _jit_and_warmup(_fn, (dx_batch,))

        # Helper: build a batch dx from numpy arrays (simulates host→device transfer)
        def _make_batch_from_np(qs_np: np.ndarray, vs_np: np.ndarray, as_np: np.ndarray):
            qs  = jnp.array(qs_np)
            vs  = jnp.array(vs_np)
            as_ = jnp.array(as_np)
            return jax.vmap(lambda q, v, a: dx0.replace(qpos=q, qvel=v, qacc=a))(qs, vs, as_)

        _make_batch_jit = jax.jit(_make_batch_from_np)

        n_batch_iters = max(10, TEST_ITERS // 5)

        def _make_np_args():
            return (
                np.random.randn(N, nq).astype(np.float32),
                np.random.randn(N, nv).astype(np.float32),
                np.random.randn(N, nv).astype(np.float32),
            )

        for label, _fn in [("INVERSE_DYNAMICS", _batch_id_fn), ("FORWARD_DYNAMICS", _batch_fd_fn), ("END_EFFECTOR_POSE", _batch_ee_fn)]:
            # WITH MEMORY
            #
            # FAIRNESS FIX (was mistiming JIT compile inside the timed region —
            # mjx batch_256_with_mem came out ~41,500us, ~3000x slower than the
            # COMPUTE-ONLY path and physically impossible for batched throughput).
            # Root cause: the previous code warmed `_make_batch_jit` against raw
            # NUMPY args but TIMED the composite `_fn(_make_batch_jit(jnp.array(...)))`
            # closure, which was never warmed end-to-end. The first timed iter
            # therefore traced+compiled `_make_batch_jit` against jnp-array inputs
            # AND lowered the `_fn(...)`-of-builder dispatch — and since the report
            # uses the MEAN over only n_batch_iters reps, that one compile dominated.
            # Fix: build the EXACT composite closure once, then `_jit_and_warmup` it
            # (block_until_ready, >=3 passes) so the timed region is pure execution
            # + the intended host->device transfer (jnp.array(qs/vs/as)).
            try:
                _wm_fn = (lambda qs, vs, as_, fn=_fn: fn(_make_batch_jit(
                    jnp.array(qs), jnp.array(vs), jnp.array(as_)
                )))
                _jit_and_warmup(_wm_fn, _make_np_args())
                wm = _time_with_mem(
                    _wm_fn,
                    _make_np_args,
                    n_iters=n_batch_iters,
                )
                _print_stats(f"{label} WITH MEMORY", N, wm)
            except Exception as e:
                print(f"# [N:{N}] {label} WITH MEMORY skipped: {e}", file=sys.stderr)

            # COMPUTE ONLY
            try:
                co = _time_device(_fn, dx_batch, n_iters=n_batch_iters)
                _print_stats(f"{label} COMPUTE ONLY", N, co)
            except Exception as e:
                print(f"# [N:{N}] {label} COMPUTE ONLY skipped: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Derivative (gradient) batch timing: id_du + fd_du, BOTH compute-only and
    # with-memory — the mjx autodiff analogue of GRiD's analytic gradients. We
    # emit WITH MEMORY too (was compute-only only) because the competitive
    # analysis/plots read batch_256_with_mem_us; without it the gradient bars
    # were null. vmap(jacobian(...)) is expensive, so fewer timed iters.
    # ------------------------------------------------------------------
    def _id_du_one(d):
        return jax.jacobian(
            lambda qpos, qvel, qacc: mjx.inverse(
                mx, d.replace(qpos=qpos, qvel=qvel, qacc=qacc)
            ).qfrc_inverse,
            argnums=(0, 1, 2),
        )(d.qpos, d.qvel, d.qacc)

    def _fd_du_one(d):
        return jax.jacobian(
            lambda qpos, qvel, qfrc: mjx.forward(
                mx, d.replace(qpos=qpos, qvel=qvel, qfrc_applied=qfrc)
            ).qacc,
            argnums=(0, 1, 2),
        )(d.qpos, d.qvel, d.qfrc_applied)

    # (label, per-sample jacobian fn, third-input field name) — id_du differentiates
    # w.r.t. qacc, fd_du w.r.t. qfrc_applied (= GRiD's tau). The with-memory builder
    # writes that third field from numpy so the H2D transfer is timed fairly.
    _DERIVS = [
        ("INVERSE_DYNAMICS_GRADIENT", _id_du_one, "qacc"),
        ("FORWARD_DYNAMICS_GRADIENT", _fd_du_one, "qfrc_applied"),
    ]
    for N in DERIV_BATCH_SIZES:
        keys     = jax.random.split(rng, N)
        dx_batch = jax.vmap(_make_dx_jnp)(keys)
        n_du_iters = max(1, TEST_ITERS // 20)

        for label, _one, third in _DERIVS:
            try:
                _batch_du_fn = jax.jit(lambda bd, f=_one: jax.vmap(f)(bd))
                _jit_and_warmup(_batch_du_fn, (dx_batch,))

                # WITH MEMORY: build the batch dx from numpy (q,v,third) inside the
                # timed region, mirroring the value-function with-mem path.
                def _build_du(qs, vs, ts, field=third):
                    return jax.vmap(
                        lambda q, v, t: dx0.replace(**{"qpos": q, "qvel": v, field: t})
                    )(jnp.array(qs), jnp.array(vs), jnp.array(ts))
                _build_du_jit = jax.jit(_build_du)
                _wm_fn = (lambda qs, vs, ts, bf=_build_du_jit, ff=_batch_du_fn: ff(bf(qs, vs, ts)))
                _du_np = lambda: (
                    np.random.randn(N, nq).astype(np.float32),
                    np.random.randn(N, nv).astype(np.float32),
                    np.random.randn(N, nv).astype(np.float32),
                )
                _jit_and_warmup(_wm_fn, _du_np())
                wm = _time_with_mem(_wm_fn, _du_np, n_iters=n_du_iters)
                _print_stats(f"{label} WITH MEMORY", N, wm)

                # COMPUTE ONLY: device-resident batch.
                co = _time_device(_batch_du_fn, dx_batch, n_iters=n_du_iters)
                _print_stats(f"{label} COMPUTE ONLY", N, co)
            except Exception as e:
                print(f"# [N:{N}] {label} skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
