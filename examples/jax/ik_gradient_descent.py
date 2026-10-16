#!/usr/bin/env python3
"""Inverse kinematics by plain gradient descent through GRiD's JAX handle.

GRiD provides one differentiable primitive here: ``end_effector_pose(q)``, a
batched FFI call whose backward is GRiD's analytic end-effector Jacobian. JAX
provides everything else: ``jax.grad`` to turn the position error into a
gradient, ``jax.jit`` and ``jax.lax.fori_loop`` to run the whole descent as one
GPU program, and the batch axis to solve several targets in one call. No
optimiser library is involved.

The targets are end-effector positions of known joint configurations, so each
one is reachable and the final error is a direct measure of convergence.

    .venv/bin/python examples/jax/ik_gradient_descent.py [--steps 300] [--batch 8]

Needs ``pip install "grid-rbd[jax]"`` with a CUDA jax wheel, nvcc on PATH and
a GPU. The first call compiles the iiwa14 ``.so`` once (the cold/warm table on
the Fast Robot Setup page has the per-robot numbers); later runs load it from
the cache in seconds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DEFAULT_URDF = Path(__file__).resolve().parents[2] / "grid_assets" / "robots" / "iiwa14.urdf"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", default=str(_DEFAULT_URDF))
    ap.add_argument("--steps", type=int, default=300, help="gradient-descent iterations")
    ap.add_argument("--batch", type=int, default=8, help="targets solved in one call")
    ap.add_argument("--lr", type=float, default=1.5, help="step size")
    ap.add_argument("--tol", type=float, default=1e-3, help="pass threshold on the final error [m]")
    args = ap.parse_args()
    urdf = Path(args.urdf).expanduser()
    if not urdf.exists():
        sys.exit(f"URDF not found: {urdf} (pass --urdf)")

    import jax
    import jax.numpy as jnp
    import grid_rbd

    robot = grid_rbd.load_robot(str(urdf), backend="jax")
    nq, B = robot.nq, args.batch
    print(f"iiwa14 through grid_rbd.jax: nq={nq}, batch={B}, jax backend={jax.default_backend()}")

    # Deterministic start configurations and targets. The targets are the
    # end-effector positions at q_goal, which sits within 0.5 rad of q0 per joint.
    # q_nom is a bent-elbow posture: the iiwa14 zero configuration is fully
    # stretched, where the position Jacobian is singular and descent stalls.
    q_nom = jnp.array([0.0, 0.6, 0.0, -1.2, 0.0, 0.9, 0.0], jnp.float32)
    joint = jnp.arange(nq, dtype=jnp.float32)
    item = jnp.arange(B, dtype=jnp.float32)[:, None]
    q0 = q_nom + 0.2 * jnp.sin(0.9 * joint + 0.3 * item)
    q_goal = q0 + 0.5 * jnp.sin(1.7 * joint + 0.8 * item + 1.0)
    p_goal = robot.end_effector_pose(q_goal)[:, :3]        # [xyz, rpy] per EE; keep xyz

    def loss(q):
        p = robot.end_effector_pose(q)[:, :3]
        return 0.5 * jnp.sum((p - p_goal) ** 2)             # sums over the batch; the gradient separates per item

    grad = jax.grad(loss)

    @jax.jit
    def descend(q, steps):
        return jax.lax.fori_loop(0, steps, lambda _, q: q - args.lr * grad(q), q)

    def position_error(q):
        return jnp.linalg.norm(robot.end_effector_pose(q)[:, :3] - p_goal, axis=1)

    q = q0
    chunk = max(args.steps // 10, 1)
    done = 0
    print(f"step {done:4d}  max error {float(position_error(q).max()):.3e} m")
    while done < args.steps:
        n = min(chunk, args.steps - done)
        q = descend(q, n)
        done += n
        print(f"step {done:4d}  max error {float(position_error(q).max()):.3e} m")

    err = position_error(q)
    jax.block_until_ready(err)
    worst = float(err.max())
    print(f"final error per target [m]: {[f'{e:.2e}' for e in err.tolist()]}")
    print(f"final max error {worst:.3e} m after {args.steps} steps (threshold {args.tol:.0e})")
    assert worst < args.tol, f"IK did not converge: max error {worst:.3e} >= {args.tol:.0e}"
    return 0


if __name__ == "__main__":
    sys.exit(main())
