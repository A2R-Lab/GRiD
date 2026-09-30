"""System identification with the joint-torque regressor (tau = Y . pi).

The classic inertial-parameter pipeline, end to end on the GPU surface:

  1. collect (q, qd, qdd, tau) "measurements" from the robot's own inverse
     dynamics (in the wild: your torque sensors / motor currents);
  2. stack the joint-torque regressors Y(q, qd, qdd) — tau is exactly LINEAR
     in the 10-per-link inertial parameters pi = [m, m*c(3), I_O(6)], so the
     fit is one least-squares solve, no iteration;
  3. verify the fit in TORQUE space (Y @ pi_hat reproduces tau to machine
     precision). NOTE the classic caveat: only the IDENTIFIABLE base
     parameters are pinned down by torque data — pi_hat is lstsq's
     minimum-norm solution and need not equal the true pi entrywise, but it
     predicts identical torques (that's what an identified model is for);
  4. push the identified parameters INTO the running model with
     set_inertia_params (runtime_inertia .so — no recompile) and show the
     device dynamics now match the identified model;
  5. (optional, if jax is installed) the differentiable outer loop:
     inverse_dynamics_wrt_params flows the analytic dc/dpi = Y through
     jax.grad — gradient-based sysID / adaptive control without FD.

Run:  .venv/bin/python bindings/examples/system_identification.py [--urdf PATH]
"""
import argparse
from pathlib import Path

import numpy as np

import grid_rbd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_URDF = _REPO_ROOT / "config" / "robot_assets" / "iiwa14.urdf"


def main(urdf: str):
    # runtime_inertia=True builds the mutable device inertia table (step 4).
    h = grid_rbd.register_robot(
        "iiwa14_sysid_example", urdf, floating_base=False,
        runtime_inertia=True,
    )
    nq, nv, nb = h.num_joints, h.num_vel, h.num_bodies
    rng = np.random.default_rng(7)

    # ── 1. "measurements": random excitation trajectories ────────────────────
    B = 64  # 64 samples * nv equations >> 10*nb unknowns
    q = rng.uniform(-1.5, 1.5, size=(B, nq)).astype(np.float32)
    qd = rng.uniform(-1.0, 1.0, size=(B, nv)).astype(np.float32)
    qdd = rng.uniform(-2.0, 2.0, size=(B, nv)).astype(np.float32)
    tau = np.asarray(h.inverse_dynamics(q, qd, qdd), dtype=np.float64)
    print(f"[1] collected {B} samples ({B * nv} torque equations, "
          f"{10 * nb} unknown parameters)")

    # ── 2. stack the regressors and solve ────────────────────────────────────
    Y = np.asarray(h.inverse_dynamics_regressor(q, qd, qdd), dtype=np.float64)
    A = Y.reshape(B * nv, 10 * nb)          # (B, nv, 10*nb) -> stacked rows
    b = tau.reshape(B * nv)
    pi_hat, *_ = np.linalg.lstsq(A, b, rcond=None)
    print(f"[2] least-squares fit: pi_hat shape {pi_hat.shape} "
          f"(minimum-norm over the unidentifiable subspace)")

    # ── 3. verify in torque space ────────────────────────────────────────────
    resid = float(np.max(np.abs(A @ pi_hat - b)))
    pi_true = np.asarray(h.inertia_params, dtype=np.float64).reshape(-1)
    resid_true = float(np.max(np.abs(A @ pi_true - b)))
    print(f"[3] torque residual: fit {resid:.3e}   baked-truth {resid_true:.3e}")
    assert resid < 1e-3, "identified model fails to reproduce the torques"

    # ── 4. push the identified model into the device table ──────────────────
    h.set_inertia_params(pi_hat.astype(np.float32))
    tau_ident = np.asarray(h.inverse_dynamics(q, qd, qdd), dtype=np.float64)
    print(f"[4] set_inertia_params(pi_hat): device-vs-data torque gap "
          f"{float(np.max(np.abs(tau_ident - tau))):.3e} (no recompile)")
    h.set_inertia_params(np.asarray(h.inertia_params, dtype=np.float32).reshape(-1))

    # ── 5. optional: the differentiable outer loop (jax) ─────────────────────
    # Simulate the real sysID setting: "measurements" come from the REAL robot
    # (here: the device model with an extra payload on the last link), while
    # the compiled model is our (payload-free) guess. The vjp flows the
    # analytic dc/dpi = Y, so jax.grad points from the guess toward the truth.
    try:
        import jax
        import jax.numpy as jnp

        pi_payload = pi_true.copy()
        pi_payload[10 * (nb - 1)] += 0.5           # +0.5 kg on the last link
        h.set_inertia_params(pi_payload.astype(np.float32))
        tau_meas = np.asarray(
            h.inverse_dynamics(q[:8], qd[:8], np.zeros_like(qdd[:8])))
        h.set_inertia_params(pi_true.astype(np.float32))  # restore the model

        hj = grid_rbd.register_robot(
            "iiwa14_sysid_example", urdf, floating_base=False,
            runtime_inertia=True, backend="jax",
        )
        target = jnp.asarray(tau_meas)

        def torque_loss(params):
            c = hj.inverse_dynamics_wrt_params(q[:8], qd[:8], jnp.tile(params, (8, 1)))
            return jnp.mean((c - target) ** 2)

        # NOTE the op's contract: its FORWARD value is the compiled model's
        # bias (params-independent — the .so carries the baked inertia); what
        # params buys you is the analytic pullback dc/dpi = Y(q, qd, qdd=0).
        # An outer sysID loop therefore takes jax.grad steps and APPLIES each
        # update with set_inertia_params (as in step 4). Here we validate the
        # gradient itself against the closed form (2/N) * Y^T (c - tau_meas)
        # built from the numpy regressor:
        pi0 = jnp.asarray(pi_true, dtype=jnp.float32)
        g = jax.grad(torque_loss)(pi0)
        c_model = np.asarray(
            h.inverse_dynamics(q[:8], qd[:8], np.zeros_like(qdd[:8])), dtype=np.float64)
        Y_bias = np.asarray(
            h.inverse_dynamics_regressor(q[:8], qd[:8], np.zeros_like(qdd[:8])),
            dtype=np.float64).reshape(8 * nv, 10 * nb)
        resid_v = (c_model - tau_meas).astype(np.float64).reshape(8 * nv)
        g_expected = (2.0 / (8 * nv)) * (Y_bias.T @ resid_v)
        gap = float(np.max(np.abs(np.asarray(g, dtype=np.float64) - g_expected)))
        print(f"[5] jax.grad through inverse_dynamics_wrt_params vs payload-world "
              f"measurements: |grad| {float(jnp.max(jnp.abs(g))):.3e}; matches the "
              f"closed form (2/N)·Y^T(c - tau_meas) to {gap:.3e} "
              f"(analytic dc/dpi = Y — no FD)")
        assert gap < 1e-3, "vjp gradient departs the closed-form regressor pullback"
    except ImportError:
        print("[5] jax not installed — skipping the differentiable outer loop")
    h.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", default=str(_DEFAULT_URDF))
    args = ap.parse_args()
    main(args.urdf)
