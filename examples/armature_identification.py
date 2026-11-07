"""Synthetic armature identification with known link inertias.

Run with a URDF path, optionally --floating-base. This example assumes exact
acceleration and torque measurements and no unknown friction or external loads.
It estimates effective reduced-coordinate armature, not individual motor inertias.
It is not a general identification solver or a claim of joint inertial identifiability.
"""
import argparse
import json

import numpy as np


def fit_armature(regressors, residual_torques):
    """Fit a full-rank compact block; refuse unexcited or negative solutions."""
    y = np.asarray(regressors, dtype=np.float64)
    residual = np.asarray(residual_torques, dtype=np.float64)
    if y.ndim != 3 or residual.shape != y.shape[:2] or y.shape[-1] == 0:
        raise ValueError("expected nonempty (samples,nv,n_a) and (samples,nv)")
    if not np.isfinite(y).all() or not np.isfinite(residual).all():
        raise ValueError("identification data must be finite")
    design = y.reshape(-1, y.shape[-1])
    alpha, _, rank, singular = np.linalg.lstsq(design, residual.reshape(-1), rcond=None)
    if rank != y.shape[-1]:
        raise ValueError("rank-deficient excitation; armature is not identifiable")
    if np.any(alpha < 0):
        raise ValueError("negative fitted armature; inspect offsets, noise and model mismatch")
    return alpha, singular


def run(urdf, *, floating_base=False, samples=32):
    import grid_rbd

    if samples < 8:
        raise ValueError("at least eight samples are needed for train/held-out split")
    options = dict(floating_base=floating_base, max_batch_size=samples,
                   enable_mujoco_kernels=False,
                   algorithm_list=["inverse_dynamics", "forward_dynamics_armature_gradient"])
    zero = grid_rbd.register_robot("sysid_known_inertias", urdf, **options)
    truth = None
    try:
        slots = list(zero.armature_slots)
        if not slots:
            raise ValueError("model has no supported scalar armature parameters")
        known = np.zeros(zero.num_vel, dtype=np.float32)
        known[slots] = np.linspace(.2, .5, len(slots))
        truth = grid_rbd.register_robot("sysid_synthetic", urdf, armature=known,
                                        runtime_armature=True, **options)
        rng = np.random.default_rng(704)
        q = rng.uniform(-.4, .4, (samples, zero.num_joints)).astype(np.float32)
        # Use configuration metadata so spherical joints and floating roots are valid.
        for kind, qi, vi, nq, nv in zero.configuration_layout:
            if kind in ("spherical", "floating"):
                start = qi + (3 if kind == "floating" else 0)
                q[:, start:start+4] /= np.linalg.norm(q[:, start:start+4], axis=1, keepdims=True)
        v, acc = [rng.uniform(-1, 1, (samples, zero.num_vel)).astype(np.float32) for _ in range(2)]
        measured = truth.inverse_dynamics(q, v, acc)
        # This known offset includes any baked friction; add known external loads
        # to BOTH calls when adapting the example to a loaded experiment.
        baseline = zero.inverse_dynamics(q, v, acc)
        y = zero.armature_regressor(acc)
        split = samples * 3 // 4
        fitted, singular = fit_armature(y[:split], (measured-baseline)[:split])
        installed = np.zeros_like(known)
        installed[slots] = fitted
        truth.set_armature(installed)
        predicted = truth.inverse_dynamics(q[split:], v[split:], acc[split:])
        np.testing.assert_allclose(fitted, known[slots], rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(predicted, measured[split:], rtol=2e-4, atol=2e-4)
        # A successful least-squares call alone does not establish identifiability.
        try:
            fit_armature(np.zeros_like(y[:split]), np.zeros_like(measured[:split]))
        except ValueError as error:
            assert "rank-deficient" in str(error)
        else:
            raise AssertionError("unexcited negative control was accepted")
        return dict(armature_slots=slots, fitted=fitted.tolist(), singular_values=singular.tolist(),
                    condition=float(singular[0]/singular[-1]),
                    held_out_max_abs_error=float(np.max(np.abs(predicted-measured[split:]))),
                    noise="none; synthetic exact-state measurements", rank_deficient_control="refused")
    finally:
        if truth is not None:
            truth.close()
        zero.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf")
    parser.add_argument("--floating-base", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.urdf, floating_base=args.floating_base), indent=2))
