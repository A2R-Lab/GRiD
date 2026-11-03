"""A polynomial-coupled robot on the Python surface (item E of 0.8).

The chain of test/cuda_equivalents/polynomial_coupling_model.py has two
followers whose fold multiplier varies with the leader. The generator refuses
the algorithms whose folds still bake a constant multiplier, so the handle is
registered with the supported subset; the bound methods must match the
reference's projection of the full tree, and the refused ones must raise the
generator's message rather than build a wrong kernel.

Run with:
    pytest test/python_wrappers/test_polynomial_coupling_handle.py -v
"""
from __future__ import annotations

import contextlib
import io
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "bindings"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed (pip install -e . from the repo root)")
if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)

from test.cuda_equivalents.polynomial_coupling_model import write_polynomial_chain

pytestmark = pytest.mark.python_wrappers

_SUPPORTED = ["inverse_dynamics", "crba", "minv", "forward_dynamics", "aba",
              "inverse_dynamics_gradient", "forward_dynamics_gradient",
              "end_effector_pose", "end_effector_pose_gradient",
              "frame_jacobian", "frame_jacobian_dot", "frame_velocity", "frame_acceleration",
              "frame_velocity_gradient", "frame_acceleration_gradient",
              "integrator", "integrator_gradient", "generalized_gravity", "nonlinear_effects",
              "inverse_dynamics_regressor", "com", "ccrba", "energy"]
_B = 4
_TOL = 3e-4


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    from URDFParser import URDFParser
    from RBDReference import RBDReference
    d = tmp_path_factory.mktemp("poly_chain")
    urdf = write_polynomial_chain(d)
    handle = _grid_rbd.register_robot(name="poly_chain_pytest", urdf_path=str(urdf), floating_base=False,
                                      max_batch_size=8, algorithm_list=_SUPPORTED, enable_mujoco_kernels=False)
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(urdf), floating_base=False)
    ref = RBDReference(robot)
    assert ref._embedding is not None
    rng = np.random.default_rng(21)
    nq, nv = robot.get_num_pos(), robot.get_num_vel()
    s = {"q": rng.standard_normal((_B, nq)).astype(np.float32),
         "qd": rng.standard_normal((_B, nv)).astype(np.float32),
         "qdd": rng.standard_normal((_B, nv)).astype(np.float32),
         "u": rng.standard_normal((_B, nv)).astype(np.float32)}
    return handle, ref, robot, s


def _close(got, exp, what):
    got = np.asarray(got, np.float64); exp = np.asarray(exp, np.float64)
    scale = max(1.0, float(np.max(np.abs(exp))))
    np.testing.assert_allclose(got.reshape(exp.shape), exp, rtol=0, atol=_TOL * scale, err_msg=what)


def test_handle_matches_projected_reference(chain):
    handle, ref, robot, s = chain
    assert handle.num_vel == robot.get_num_vel() == 2
    leaf = robot.get_joint_by_id(int(robot.get_leaf_nodes()[0])).get_name()
    c = handle.inverse_dynamics(s["q"], s["qd"], s["qdd"])
    M = handle.crba(s["q"])
    qdd = handle.forward_dynamics(s["q"], s["qd"], s["u"])
    J = handle.frame_jacobian(s["q"], reference_frame="WORLD")
    v = handle.frame_velocity(s["q"], s["qd"], reference_frame="WORLD")
    for i in range(_B):
        q, qd, qdd_i, u = (s[k][i].astype(np.float64) for k in ("q", "qd", "qdd", "u"))
        _close(c[i], ref.inverse_dynamics(q, qd, qdd_i)[0], f"inverse_dynamics {i}")
        _close(M[i], ref.crba(q), f"crba {i}")
        _close(qdd[i], ref.forward_dynamics(q, qd, u), f"forward_dynamics {i}")
        _close(J[i], ref.frame_jacobian(q, leaf, "WORLD"), f"frame_jacobian {i}")
        _close(v[i], ref.frame_velocity(q, qd, leaf, "WORLD"), f"frame_velocity {i}")
    dc = handle.inverse_dynamics_gradient(s["q"], s["qd"], s["qdd"])
    for i in range(_B):
        q, qd, qdd_i = (s[k][i].astype(np.float64) for k in ("q", "qd", "qdd"))
        _close(dc[i], ref.inverse_dynamics_gradient(q, qd, qdd_i), f"inverse_dynamics_gradient {i}")


def test_refused_algorithms_name_the_coupling(tmp_path):
    """The default algorithm list pulls in folds that still bake a constant
    multiplier; the generator refuses them with a message naming them."""
    urdf = write_polynomial_chain(tmp_path)
    with pytest.raises(NotImplementedError, match="polynomial couplings"):
        _grid_rbd.register_robot(name="poly_chain_refused_pytest", urdf_path=str(urdf), floating_base=False,
                                 max_batch_size=4, algorithm_list=["coriolis_matrix"], enable_mujoco_kernels=False)
