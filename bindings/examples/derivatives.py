"""grid-rbd derivatives: analytic first- + second-order, and the autodiff idioms.

GRiD emits ANALYTIC derivatives (not finite-difference, not an autodiff tape). There are
two ways to reach them and this file shows both:

  A. CALL the derivative methods DIRECTLY — you want the full Jacobian / Hessian tensor
     (e.g. to assemble a DDP/iLQR Riccati pass yourself):
        inverse_dynamics_gradient(q, qd, qdd)  -> (B, NV, 2*NV)   [ ∂tau/∂q | ∂tau/∂qd ]
        forward_dynamics_gradient(q, qd, u)     -> (B, NV, 2*NV)   [ ∂qdd/∂q | ∂qdd/∂qd ]
        end_effector_pose_gradient(q)           -> (B, 6*num_ees, NV)
        idsva_so(q, qd, qdd)  -> SecondOrderID NamedTuple of 4 x (B, NV, NV, NV)
        fdsva_so(q, qd, u)    -> SecondOrderFD NamedTuple of 4 x (B, NV, NV, NV)
     (id_du / fd_du in the literature == inverse/forward_dynamics_gradient here.)

  B. Let jax.grad / loss.backward() PULL the gradient through a cost — the value methods
     (id / fd / aba / end_effector_pose / integrator) carry a custom_vjp / autograd backward
     that contracts your cotangent with the SAME analytic Jacobian. You write physics + cost;
     grad does the chain rule with no finite differencing.

Run:  python bindings/examples/derivatives.py [--urdf PATH] [--batch 64]
Needs: pip install -e .[jax]   ·   nvcc on PATH   ·   an iiwa14 URDF
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_DEFAULT_URDF = (
    Path.home()
    / ".cache/robot_descriptions/drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", default=str(_DEFAULT_URDF))
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    urdf = Path(args.urdf).expanduser()
    if not urdf.exists():
        sys.exit(f"URDF not found: {urdf} (pass --urdf)")

    import grid_rbd

    # ── numpy surface: the FULL derivative tensors, batched ──────────────────
    h = grid_rbd.register_robot("iiwa14_deriv", str(urdf),
                                max_batch_size=max(args.batch, 64))
    nq, nv, B = h.num_joints, h.num_vel, args.batch
    print(f"iiwa14: nq={nq} nv={nv}  batch B={B}  num_ees={h.num_ees}")

    rng = np.random.default_rng(0)
    q   = rng.standard_normal((B, nq)).astype(np.float32)
    qd  = rng.standard_normal((B, nv)).astype(np.float32)
    qdd = rng.standard_normal((B, nv)).astype(np.float32)
    u   = rng.standard_normal((B, nv)).astype(np.float32)

    # ── A. first-order analytic Jacobians (id_du / fd_du) ────────────────────
    # Both stack [d/dq | d/dqd] along the last axis -> (B, NV, 2*NV). The id
    # gradient is qdd-AWARE: pass qdd to include the d(M*qdd)/dq term; omit it
    # (qdd=None) for the bias gradient d(c)/d[q,qd].
    id_du = h.inverse_dynamics_gradient(q, qd, qdd)     # (B, NV, 2*NV)
    fd_du = h.forward_dynamics_gradient(q, qd, u)        # (B, NV, 2*NV)
    dtau_dq,  dtau_dqd = id_du[..., :nv], id_du[..., nv:]
    dqdd_dq, dqdd_dqd  = fd_du[..., :nv], fd_du[..., nv:]
    print(f"\n[A] inverse_dynamics_gradient -> {id_du.shape}  "
          f"(dtau/dq {dtau_dq.shape}, dtau/dqd {dtau_dqd.shape})")
    print(f"    forward_dynamics_gradient -> {fd_du.shape}  "
          f"(dqdd/dq {dqdd_dq.shape}, dqdd/dqd {dqdd_dqd.shape})")

    # end-effector pose Jacobian: d[xyz; rpy]/dv, stacked over the EE targets
    ee_J = h.end_effector_pose_gradient(q)               # (B, 6*num_ees, NV)
    print(f"    end_effector_pose_gradient -> {ee_J.shape}  (6*num_ees x NV)")

    # ── B. second-order analytic tensors (idsva_so / fdsva_so) ───────────────
    # Each returns a NamedTuple of four (B, NV, NV, NV) tensors (the d2/dq2,
    # d2/dq dqd, ... blocks). Unpack by field name or positionally.
    so_id = h.idsva_so(q, qd, qdd)                        # SecondOrderID
    so_fd = h.fdsva_so(q, qd, u)                          # SecondOrderFD
    print(f"\n[B] idsva_so -> {type(so_id).__name__}{tuple(so_id._fields)}, "
          f"each {so_id[0].shape}")
    print(f"    fdsva_so -> {type(so_fd).__name__}{tuple(so_fd._fields)}, "
          f"each {so_fd[0].shape}")

    # ── C. the autodiff idioms on the JAX fast path ──────────────────────────
    # The same analytic Jacobians, but PULLED through a cost by jax.grad — and
    # composed under jit + vmap. This is what an RL/MPC loss reaches for.
    try:
        import jax
        import jax.numpy as jnp
        import grid_rbd.jax as grid_jax
    except Exception as e:
        print(f"\n[C] JAX surface skipped ({type(e).__name__}: {e}). pip install -e .[jax]")
        return

    grid_rbd.precompile("iiwa14_deriv_jax", str(urdf),
                        max_batch_size=max(B, 64), backends=("jax",))
    hj = grid_jax.get_robot("iiwa14_deriv_jax")
    qj  = jax.device_put(jnp.asarray(q))
    qdj = jax.device_put(jnp.asarray(qd))
    uj  = jax.device_put(jnp.asarray(u))

    # (i) jit + grad: an effort-regularized acceleration cost, grad wrt u.
    @jax.jit
    def cost(q, qd, u):
        qdd = hj.forward_dynamics(q, qd, u)              # analytic-diff'd FFI call
        return jnp.mean(qdd ** 2) + 1e-3 * jnp.mean(u ** 2)
    g_u = jax.jit(jax.grad(cost, argnums=2))(qj, qdj, uj)
    print(f"\n[C.i]  jax.grad(cost, u) via custom_vjp: |dC/du|={float(jnp.linalg.norm(g_u)):.4f}  "
          f"on {g_u.devices()}")

    # (ii) jit + grad through end_effector_pose: a Cartesian reach cost wrt q.
    p_des = jnp.zeros((6 * hj.num_ees,), jnp.float32)
    @jax.jit
    def ee_cost(q):
        pose = hj.end_effector_pose(q)                  # (B, 6*num_ees)
        return jnp.mean((pose - p_des) ** 2)
    g_q = jax.jit(jax.grad(ee_cost))(qj)
    print(f"[C.ii] jax.grad(ee_cost, q): |dC/dq|={float(jnp.linalg.norm(g_q)):.4f}")

    # (iii) vmap a PER-SAMPLE gradient (Jacobian-of-cost over the batch axis).
    per_sample_grad = jax.vmap(jax.grad(lambda q1, v1, u1:
        jnp.sum(hj.forward_dynamics(q1[None], v1[None], u1[None])[0] ** 2), argnums=2))
    G = jax.jit(per_sample_grad)(qj, qdj, uj)            # (B, NV)
    print(f"[C.iii] vmap(grad) per-sample wrt u -> {tuple(G.shape)} (no Python loop)")

    print("\nTakeaway: call *_gradient / idsva_so / fdsva_so for the full analytic tensors; "
          "let jax.grad / .backward() pull them through a cost. Same math, never finite-diff.")


if __name__ == "__main__":
    main()
