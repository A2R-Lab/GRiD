"""grid-rbd × JAX on a FLOATING-BASE robot (Unitree Go2): stay on the GPU, integrate
the quaternion state correctly, differentiate analytically.

The go2 twin of ``jax_gpu_resident.py`` (iiwa14, fixed base). What changes with a
floating base:

  * the configuration is nq = 7 + 12 = 19 wide: ``q = [base_pos(3), base_quat_xyzw(4),
    joint_angles(12)]`` with a NORMALIZED quaternion;
  * the tangent space is nv = 6 + 12 = 18 (nv != nq): velocity/torque buffers are passed
    nq-wide with the trailing quaternion-padding slot zeroed; only the leading nv entries
    of nq-wide outputs are meaningful;
  * a hand-rolled ``q + dt*qd`` Euler step is WRONG for the quaternion — the resident
    rollout therefore drives GRiD's own ``integrator`` kernel (on-manifold base retract)
    inside ``jax.lax.scan``, which is exactly the nq != nv path this example exercises.

The .so is built pin-convention-only (``enable_mujoco_kernels=False``) to keep the
one-time nvcc build light. The ``[6]`` rollout timing line matches the iiwa14 example
format so ``test/benchmarks/gpu_resident_timing.py`` can drive both with the same parser.
(donate_argnums / dlpack handoff are demonstrated in the iiwa14 file and not repeated.)

Run:  python bindings/examples/jax_gpu_resident_go2.py [--urdf PATH] [--batch 256] [--steps 50]
Needs: pip install -e .[jax]   ·   nvcc on PATH   ·   go2 URDF (auto-resolved via
       robot_descriptions if installed)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_DEFAULT_URDF = (
    Path.home()
    / ".cache/robot_descriptions/unitree_ros/robots/go2_description/urdf/go2_description.urdf"
)


def _resolve_urdf(arg: str) -> Path:
    p = Path(arg).expanduser()
    if p.exists():
        return p
    try:  # robot_descriptions downloads/caches on first use
        from robot_descriptions import go2_description
        return Path(go2_description.URDF_PATH)
    except Exception:
        sys.exit(f"URDF not found: {p} (pass --urdf, or pip install robot_descriptions)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", default=str(_DEFAULT_URDF))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=50, help="rollout horizon K")
    args = ap.parse_args()

    urdf = _resolve_urdf(args.urdf)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import grid_rbd
    import grid_rbd.jax as grid_jax

    print(f"grid_rbd v{grid_rbd.__version__} · jax {jax.__version__} · backend={jax.default_backend()}")
    if jax.default_backend() != "gpu":
        print("  !! JAX is not on the GPU backend — install jax[cuda12]; this demo is about GPU residency.")

    grid_rbd.precompile("go2_resident", str(urdf), floating_base=True,
                        max_batch_size=max(args.batch, 256), backends=("jax",),
                        tiers=[{"enable_mujoco_kernels": False}])
    h = grid_jax.get_robot("go2_resident")
    nq, nv, B = h.num_joints, h.num_vel, args.batch
    print(f"  go2 (floating): nq={nq} nv={nv} (nv != nq: quaternion base)  batch B={B}  "
          f"max_batch={h.max_batch}")

    key = jax.random.PRNGKey(0)
    kq, kquat, kv, ku = jax.random.split(key, 4)

    # ── 1. place a VALID floating state on the GPU once ──────────────────────
    # q = [pos(3), quat_xyzw(4) NORMALIZED, joints(12)]; qd/u nq-wide with the
    # trailing quaternion-padding slot zeroed (leading nv entries = tangent data).
    pos = jax.random.uniform(kq, (B, 3), jnp.float32, -1.0, 1.0)
    quat = jax.random.normal(kquat, (B, 4), jnp.float32)
    quat = quat / jnp.linalg.norm(quat, axis=1, keepdims=True)
    joints = jax.random.uniform(kq, (B, nq - 7), jnp.float32, -1.0, 1.0)
    q = jax.device_put(jnp.concatenate([pos, quat, joints], axis=1))
    pad = jnp.zeros((B, nq - nv), jnp.float32)
    qd = jax.device_put(jnp.concatenate(
        [jax.random.uniform(kv, (B, nv), jnp.float32, -1.0, 1.0), pad], axis=1))
    u = jax.device_put(jnp.concatenate(
        [jax.random.uniform(ku, (B, nv), jnp.float32, -1.0, 1.0), pad], axis=1))
    print(f"\n[1] inputs resident on: {q.devices()}  (q normalized quaternion, "
          f"qd/u padded {nq - nv} slot)")

    # ── 2. single call → device-resident output (no host hop) ────────────────
    qdd = h.forward_dynamics(q, qd, u)
    jax.block_until_ready(qdd)
    print(f"[2] forward_dynamics output {tuple(qdd.shape)} lives on: {qdd.devices()} "
          f"(nq-wide; leading nv={nv} = tangent qdd)")

    # ── 3. compose under ONE jit: dynamics → quadratic effort+accel cost ──────
    @jax.jit
    def step_cost(q, qd, u):
        qdd = h.forward_dynamics(q, qd, u)                 # GRiD FFI call, on device
        return jnp.mean(qdd[:, :nv] ** 2) + 1e-3 * jnp.mean(u**2)
    c = step_cost(q, qd, u); jax.block_until_ready(c)
    print(f"[3] fused jit(forward_dynamics→cost) = {float(c):.4f}  (single GPU program)")

    # ── 4. vmap: batch a per-sample closure with no Python loop ───────────────
    per_sample = lambda q1, v1, u1: h.forward_dynamics(q1, v1, u1)
    vmapped = jax.jit(jax.vmap(per_sample))
    out = vmapped(q, qd, u); jax.block_until_ready(out)
    print(f"[4] vmap over B={B}: {tuple(out.shape)} on {out.devices()}")

    # ── 5. analytic grad through the custom_vjp (tangent-space Jacobians) ─────
    grad_u = jax.jit(jax.grad(step_cost, argnums=2))(q, qd, u)
    jax.block_until_ready(grad_u)
    print(f"[5] jax.grad(cost, u) via GRiD's analytic Jacobian-matvec FFI: "
          f"|∂cost/∂u|={float(jnp.linalg.norm(grad_u)):.4f}")

    # ── 6. RESIDENT ROLLOUT: K steps, quaternion-correct, state never leaves GPU ─
    # GRiD's integrator kernel does the on-manifold base retract (nq != nv), so
    # the scan carries the full floating state device-to-device. The kernel
    # returns (B, nq+nv): [q_next (nq) | qd_next (nv)] — re-pad qd for the next call.
    dt = 0.01
    def rollout_resident(q0, qd0, us):
        def body(carry, uk):
            q, qd = carry
            x = h.integrator(q, qd, uk, dt)               # GRiD call inside the scan
            q = x[:, :nq]
            qd = jnp.concatenate([x[:, nq:], pad], axis=1)
            return (q, qd), jnp.mean(x[:, nq:] ** 2)
        (qK, qdK), traj = jax.lax.scan(body, (q0, qd0), us)
        return qK, qdK, traj
    rollout_jit = jax.jit(rollout_resident)
    us = jax.device_put(jnp.concatenate(
        [jax.random.uniform(ku, (args.steps, B, nv), jnp.float32, -0.2, 0.2),
         jnp.zeros((args.steps, B, nq - nv), jnp.float32)], axis=2))
    qK, qdK, traj = rollout_jit(q, qd, us); jax.block_until_ready(qK)   # compile + warm
    t0 = time.perf_counter()
    for _ in range(20):
        qK, qdK, traj = rollout_jit(q, qd, us); jax.block_until_ready(qK)
    t_res = (time.perf_counter() - t0) / 20 * 1e3

    # sanity: the resident rollout kept the base quaternion on the unit sphere
    quat_norm_drift = float(jnp.max(jnp.abs(jnp.linalg.norm(qK[:, 3:7], axis=1) - 1.0)))

    # anti-pattern: round-trip to host every step (what NOT to do)
    def rollout_host_roundtrip(q0, qd0, us):
        q, qd = np.asarray(q0), np.asarray(qd0)
        step = jax.jit(lambda q, qd, uk: h.integrator(q, qd, uk, dt))
        np_pad = np.zeros((B, nq - nv), np.float32)
        for k in range(us.shape[0]):
            x = np.asarray(step(jnp.asarray(q), jnp.asarray(qd), jnp.asarray(us[k])))  # D2H+H2D each step
            q = x[:, :nq]
            qd = np.concatenate([x[:, nq:], np_pad], axis=1)
        return q, qd
    _ = rollout_host_roundtrip(q, qd, np.asarray(us))   # warm
    t0 = time.perf_counter()
    for _ in range(3):
        rollout_host_roundtrip(q, qd, np.asarray(us))
    t_host = (time.perf_counter() - t0) / 3 * 1e3
    print(f"[6] {args.steps}-step rollout (B={B}):  RESIDENT (lax.scan) {t_res:7.2f} ms"
          f"   vs   host-roundtrip-per-step {t_host:8.2f} ms   →  {t_host/t_res:.1f}× faster staying on GPU")
    print(f"    base quaternion after {args.steps} on-manifold steps: max |1-||quat||| = "
          f"{quat_norm_drift:.2e} (integrator retract keeps the state valid)")

    print("\nTakeaway: a floating base changes the STATE handling (normalized quaternion,"
          " nq != nv padding, integrator-kernel retract) but not the residency story —"
          " place data on the GPU once and compose jit/vmap/grad/scan exactly as fixed-base.")


if __name__ == "__main__":
    main()
