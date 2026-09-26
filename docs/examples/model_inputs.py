"""CPU-only packing and inertial-parameter examples; run from the repo root."""
from pathlib import Path

import numpy as np
from RBDReference import RBDReference
from URDFParser import URDFParser

ROOT = Path(__file__).resolve().parents[2]


def model_inputs(name, batch=2):
    """Neutral inputs for these two scalar-joint fixtures, not arbitrary URDFs."""
    floating = {"iiwa14": False, "go2": True}[name]
    robot = URDFParser().parse(
        str(ROOT / "config/robot_assets" / f"{name}.urdf"),
        floating_base=floating,
    )
    if robot is None:
        raise ValueError("URDF parsing failed")
    nq, nv = robot.get_num_pos(), robot.get_num_vel()
    q = np.zeros((batch, nq), dtype=np.float32)
    if floating:
        q[:, 6] = 1  # Pinocchio base [xyz, qx, qy, qz, qw].
    # Velocities, accelerations and torques are NV wide on every surface.
    v = np.zeros((batch, nv), dtype=np.float32)
    u = np.zeros_like(v)
    # plant_step takes the state x = [q, v] and the NV-wide control.
    x = np.concatenate((q, v), axis=1)
    return robot, q, v, u, x


def inertial_parameters(robot):
    """Stack GRiD parameters from parsed, fixed-joint-merged body inertias."""
    params = []
    for body in range(robot.get_num_bodies()):
        I = np.asarray(robot.get_Imat_by_id(body), dtype=np.float64)
        # Angular-linear spatial inertia: top-right block is skew(m*c).
        params.extend([
            I[3, 3], I[2, 4], I[0, 5], I[1, 3],
            I[0, 0], I[0, 1], I[0, 2], I[1, 1], I[1, 2], I[2, 2],
        ])
    return np.asarray(params)


def regressor_residual(robot):
    """Check Y*pi=tau on fixed iiwa14 using the CPU reference only."""
    rbd = RBDReference(robot)
    q = np.linspace(-0.2, 0.3, robot.get_num_pos())
    v = np.linspace(-0.3, 0.2, robot.get_num_vel())
    a = np.linspace(0.1, 0.4, robot.get_num_vel())
    Y = rbd.inverse_dynamics_regressor(q, v, a)
    tau, *_ = rbd.inverse_dynamics(q, v, a)
    return float(np.max(np.abs(Y @ inertial_parameters(robot) - tau)))


if __name__ == "__main__":
    for name in ("iiwa14", "go2"):
        robot, q, qd, u, x = model_inputs(name)
        print(name, "NQ/NV:", robot.get_num_pos(), robot.get_num_vel(),
              "q/qd/u/x shapes:", q.shape, qd.shape, u.shape, x.shape)
        if name == "iiwa14":
            print("max |Y*pi - tau|:", regressor_residual(robot))
