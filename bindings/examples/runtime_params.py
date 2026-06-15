"""grid-rbd runtime params: mutate the model WITHOUT recompiling.

Two opt-in, runtime-mutable parameter tables let you change the robot model after compile,
with NO new .so build — the entry points for system-identification, domain randomization,
payload changes, and kinematic calibration on the GPU.

  * runtime_inertia=True  -> handle.set_inertia_params(table)
        per-body 10-param inertia [m, hx,hy,hz, Ixx,Ixy,Ixz, Iyy,Iyz, Izz], shape (num_bodies, 10).
        Every subsequent id / fd / aba / crba / minv / gradient rebuilds the spatial inertia
        from the updated table. Use it for sysID and payload / domain-randomization sweeps.

  * runtime_transform=True -> handle.set_transform_params(table)
        per-joint <origin> [x, y, z, roll, pitch, yaw], shape (num_joints, 6).
        Every subsequent DYNAMICS call rebuilds each joint's constant Xfixed from the table.
        Use it for kinematic calibration / link-length domain randomization.
        (v1 scope: this affects the DYNAMICS; end_effector_pose still uses the baked origin.)

Both are NUMPY-backend only today (the jax/torch FFI surfaces don't thread the mutable tables
yet — register_robot raises a clear error if you ask). Both are byte-identical to a baked build
when you pass the baked values straight back. Each re-keys the cache, so the runtime-mutable .so
coexists with the plain one.

Run:  python bindings/examples/runtime_params.py [--urdf PATH]
Needs: pip install -e bindings/   ·   nvcc on PATH   ·   an iiwa14 URDF
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
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    urdf = Path(args.urdf).expanduser()
    if not urdf.exists():
        sys.exit(f"URDF not found: {urdf} (pass --urdf)")

    import grid_rbd

    B = args.batch

    # ── 1. runtime-mutable INERTIA (sysID / payload / domain randomization) ──
    # Register with runtime_inertia=True: the .so carries a d_inertia_params
    # table + an on-device 6x6 rebuild + a host mutator. numpy backend only.
    h = grid_rbd.register_robot("iiwa14_runtime_inertia", str(urdf),
                                runtime_inertia=True, max_batch_size=max(B, 8))
    nq, nv = h.num_joints, h.num_vel
    print(f"runtime_inertia={h.runtime_inertia}  nq={nq} nv={nv} num_bodies={h.num_bodies}")

    rng = np.random.default_rng(0)
    q  = rng.standard_normal((B, nq)).astype(np.float32)
    qd = rng.standard_normal((B, nv)).astype(np.float32)
    u  = rng.standard_normal((B, nv)).astype(np.float32)

    # baked values: (num_bodies, 10), rows [m, hx,hy,hz, Ixx,Ixy,Ixz, Iyy,Iyz, Izz]
    baked_I = h.inertia_params.copy()
    print(f"  baked inertia_params {baked_I.shape}, body-0 mass m={baked_I[0, 0]:.4f}")

    # passing the baked table back is a no-op (byte-identical to the baked build)
    h.set_inertia_params(baked_I)
    qdd_baked = h.forward_dynamics(q, qd, u)

    # sysID / payload: bump the LAST link's mass by +0.5 kg (a tool / payload),
    # scaling its first moment h = m*c so the COM stays put. No recompile.
    perturbed = baked_I.copy()
    m0 = perturbed[-1, 0]
    m1 = m0 + 0.5
    perturbed[-1, 1:4] *= (m1 / m0)     # h = m*c scales with mass at fixed COM
    perturbed[-1, 0] = m1
    h.set_inertia_params(perturbed)
    qdd_payload = h.forward_dynamics(q, qd, u)

    dmax = float(np.max(np.abs(qdd_payload - qdd_baked)))
    print(f"  +0.5 kg payload on last link -> max |d qdd| = {dmax:.4e} (no rebuild)")
    h.set_inertia_params(baked_I)        # restore

    # domain-randomization sweep idiom: jitter all masses +-10%, no rebuild per draw
    for k in range(3):
        draw = baked_I.copy()
        scale = (1.0 + 0.1 * rng.standard_normal(draw.shape[0])).astype(np.float32)
        draw[:, 0:4] *= scale[:, None]   # scale m and h=m*c together
        h.set_inertia_params(draw)
        qdd_k = h.forward_dynamics(q, qd, u)
        print(f"  domain-rand draw {k}: |qdd|={np.linalg.norm(qdd_k):.4f}")
    h.set_inertia_params(baked_I)

    # ── 2. runtime-mutable joint-frame TRANSFORM (kinematic calibration) ─────
    # Register with runtime_transform=True: the .so carries a d_transform_params
    # table (per-joint <origin>) + a host mutator. numpy backend only.
    ht = grid_rbd.register_robot("iiwa14_runtime_transform", str(urdf),
                                 runtime_transform=True, max_batch_size=max(B, 8))
    print(f"\nruntime_transform={ht.runtime_transform}")
    baked_T = ht.transform_params.copy()     # (num_joints, 6) = [x,y,z,roll,pitch,yaw]
    print(f"  baked transform_params {baked_T.shape}, joint-1 origin xyz={baked_T[1, :3]}")

    ht.set_transform_params(baked_T)         # baked values -> baked result (no-op)
    tau_baked = ht.inverse_dynamics(q, qd)

    # kinematic calibration / link-length randomization: nudge joint-2's z origin
    calibrated = baked_T.copy()
    calibrated[2, 2] += 0.01                  # +1 cm along the link
    ht.set_transform_params(calibrated)
    tau_calib = ht.inverse_dynamics(q, qd)
    dmax_t = float(np.max(np.abs(tau_calib - tau_baked)))
    print(f"  +1 cm on joint-2 origin -> max |d tau| = {dmax_t:.4e} (DYNAMICS only; "
          f"end_effector_pose still uses the baked origin in v1)")
    ht.set_transform_params(baked_T)          # restore

    print("\nTakeaway: build once with runtime_inertia / runtime_transform, then set_*_params() "
          "as often as you like — sysID, payloads, calibration, domain-rand — with no recompile.")


if __name__ == "__main__":
    main()
