"""grid-rbd × JAX: keep everything on the GPU, compose freely, differentiate.

THE POINT (the fast path): the `.jax` handle returns ``jax.Array`` from every method via
``jax.ffi.ffi_call`` wrapped in ``jax.custom_vjp``. That means a GRiD call is just another
JAX op — it stays on the device, composes under ``jax.jit`` / ``jax.vmap`` / ``jax.grad``,
and chains into the NEXT GRiD call with NO host round-trip. Data placed on the GPU once
lives there across an entire control/learning pipeline. This is what an adopter should reach
for; the numpy handle (host arrays in/out) is the convenience surface, not the speed surface.

What this file demonstrates, in order:
  1. Inputs placed on device once (``jax.device_put``); outputs come back device-resident.
  2. Single GRiD call — confirm the result never left the GPU (``.devices()``).
  3. Composition under one ``@jax.jit``: forward_dynamics → cost, fused, all on-device.
  4. ``jax.vmap`` — batch the same closure with no Python loop.
  5. ``jax.grad`` — GRiD emits ANALYTIC gradients, so ``grad`` flows through the custom_vjp
     (a Jacobian-matvec FFI call), not a finite-difference or autodiff tape.
  6. A RESIDENT ROLLOUT (``jax.lax.scan``): K dynamics steps where the state NEVER touches the
     host. Timed against the anti-pattern (a Python loop that round-trips to numpy each step)
     to show why staying resident is the whole game.
  7. donate_argnums — let XLA reuse the input buffer in-place (no extra allocation).
  8. (optional) zero-copy dlpack handoff JAX → PyTorch — share the literal GPU pointer.

Run:  python bindings/examples/jax_gpu_resident.py [--urdf PATH] [--batch 256] [--steps 50]
Needs: pip install -e .[jax]   ·   nvcc on PATH   ·   an iiwa14 URDF
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_DEFAULT_URDF = (
    Path.home()
    / ".cache/robot_descriptions/drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", default=str(_DEFAULT_URDF))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=50, help="rollout horizon K")
    args = ap.parse_args()

    urdf = Path(args.urdf).expanduser()
    if not urdf.exists():
        sys.exit(f"URDF not found: {urdf} (pass --urdf)")

    import jax
    import jax.numpy as jnp
    import numpy as np
    import grid_rbd
    import grid_rbd.jax as grid_jax

    print(f"grid_rbd v{grid_rbd.__version__} · jax {jax.__version__} · backend={jax.default_backend()}")
    if jax.default_backend() != "gpu":
        print("  !! JAX is not on the GPU backend — install jax[cuda12]; this demo is about GPU residency.")

    # ── register (or hit the cache) and grab the JAX handle ──────────────────
    # precompile() bakes max_batch + builds the .so once; get_robot() returns a
    # JaxRobotHandle whose methods are jittable and return jax.Array.
    grid_rbd.precompile("iiwa14_resident", str(urdf),
                        max_batch_size=max(args.batch, 256), backends=("jax",))
    h = grid_jax.get_robot("iiwa14_resident")
    nq, nv, B = h.num_joints, h.num_vel, args.batch
    print(f"  iiwa14: nq={nq} nv={nv}  batch B={B}  max_batch={h.max_batch}")

    key = jax.random.PRNGKey(0)
    kq, kv, ku = jax.random.split(key, 3)

    # ── 1. place inputs on the GPU ONCE ──────────────────────────────────────
    q  = jax.device_put(jax.random.uniform(kq, (B, nq), jnp.float32, -1.0, 1.0))
    qd = jax.device_put(jax.random.uniform(kv, (B, nv), jnp.float32, -1.0, 1.0))
    u  = jax.device_put(jax.random.uniform(ku, (B, nv), jnp.float32, -1.0, 1.0))
    print(f"\n[1] inputs resident on: {q.devices()}")

    # ── 2. single call → device-resident output (no host hop) ────────────────
    qdd = h.forward_dynamics(q, qd, u)
    jax.block_until_ready(qdd)
    print(f"[2] forward_dynamics output {tuple(qdd.shape)} lives on: {qdd.devices()} "
          f"(never copied to host)")

    # ── 3. compose under ONE jit: dynamics → quadratic effort+accel cost ──────
    @jax.jit
    def step_cost(q, qd, u):
        qdd = h.forward_dynamics(q, qd, u)          # GRiD FFI call, on device
        return jnp.mean(qdd**2) + 1e-3 * jnp.mean(u**2)   # fused into the same XLA program
    c = step_cost(q, qd, u); jax.block_until_ready(c)
    print(f"[3] fused jit(forward_dynamics→cost) = {float(c):.4f}  (single GPU program)")

    # ── 4. vmap: batch a per-sample closure with no Python loop ───────────────
    # Per-sample args are 1-D — the FFI calls carry vmap_method="broadcast_all",
    # so vmap re-adds the mapped batch axis and GRiD sees one (B, n) call. (Do
    # NOT wrap with [None]: that stacks to (B, 1, n), which the 2-D FFI rejects.)
    per_sample = lambda q1, v1, u1: h.forward_dynamics(q1, v1, u1)
    vmapped = jax.jit(jax.vmap(per_sample))
    out = vmapped(q, qd, u); jax.block_until_ready(out)
    print(f"[4] vmap over B={B}: {tuple(out.shape)} on {out.devices()}")

    # ── 5. analytic grad through the custom_vjp (NOT autodiff/finite-diff) ────
    grad_u = jax.jit(jax.grad(step_cost, argnums=2))(q, qd, u)
    jax.block_until_ready(grad_u)
    print(f"[5] jax.grad(cost, u) via GRiD's analytic Jacobian-matvec FFI: "
          f"|∂cost/∂u|={float(jnp.linalg.norm(grad_u)):.4f}")

    # ── 6. RESIDENT ROLLOUT: K steps, state never leaves the GPU ─────────────
    # Semi-implicit Euler around GRiD forward_dynamics, driven through lax.scan so
    # the whole K-step horizon is ONE GPU program; q/qd are carried device-to-device.
    dt = 0.01
    def rollout_resident(q0, qd0, us):
        def body(carry, uk):
            q, qd = carry
            qdd = h.forward_dynamics(q, qd, uk)     # GRiD call inside the scan
            qd = qd + dt * qdd
            q = q + dt * qd
            return (q, qd), jnp.mean(qd**2)
        (qK, qdK), traj = jax.lax.scan(body, (q0, qd0), us)
        return qK, qdK, traj
    rollout_jit = jax.jit(rollout_resident)
    us = jax.device_put(jax.random.uniform(ku, (args.steps, B, nv), jnp.float32, -0.2, 0.2))
    qK, qdK, traj = rollout_jit(q, qd, us); jax.block_until_ready(qK)   # compile + warm
    t0 = time.perf_counter()
    for _ in range(20):
        qK, qdK, traj = rollout_jit(q, qd, us); jax.block_until_ready(qK)
    t_res = (time.perf_counter() - t0) / 20 * 1e3

    # anti-pattern: round-trip to host every step (what NOT to do)
    def rollout_host_roundtrip(q0, qd0, us):
        q, qd = np.asarray(q0), np.asarray(qd0)
        fd = jax.jit(h.forward_dynamics)
        for k in range(us.shape[0]):
            qdd = np.asarray(fd(jnp.asarray(q), jnp.asarray(qd), jnp.asarray(us[k])))  # D2H+H2D each step
            qd = qd + dt * qdd; q = q + dt * qd
        return q, qd
    _ = rollout_host_roundtrip(q, qd, np.asarray(us))   # warm
    t0 = time.perf_counter()
    for _ in range(3):
        rollout_host_roundtrip(q, qd, np.asarray(us))
    t_host = (time.perf_counter() - t0) / 3 * 1e3
    print(f"[6] {args.steps}-step rollout (B={B}):  RESIDENT (lax.scan) {t_res:7.2f} ms"
          f"   vs   host-roundtrip-per-step {t_host:8.2f} ms   →  {t_host/t_res:.1f}× faster staying on GPU")

    # ── 7. donate_argnums: reuse the input buffer in place ───────────────────
    bumped = jax.jit(lambda q, qd, u: h.forward_dynamics(q, qd, u),
                     donate_argnums=(0,))(q, qd, u)
    jax.block_until_ready(bumped)
    print(f"[7] donate_argnums=(0,): XLA reused q's buffer in place (no extra alloc)")

    # ── 8. zero-copy dlpack handoff JAX → PyTorch (share the GPU pointer) ─────
    try:
        import torch
        t = torch.utils.dlpack.from_dlpack(qdd)         # same device memory, no copy
        print(f"[8] dlpack JAX→torch: torch.Tensor on {t.device}, shares qdd's GPU buffer "
              f"(data_ptr=0x{t.data_ptr():x})")
    except Exception as e:
        print(f"[8] dlpack handoff skipped ({type(e).__name__}: {e})")

    print("\nTakeaway: place data on the GPU once, call GRiD as a JAX op, and keep composing — "
          "jit/vmap/grad/scan all stay resident. Reach for host arrays only at the I/O boundary.")


if __name__ == "__main__":
    main()
