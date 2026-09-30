"""Contact-frame world positions + tangent Jacobians (GATO ask 2026-09-20).

`register_robot(contact_frames=[...])` / `gen_all_code(contact_frames=...)` bakes a
contact set; this family exposes the world positions of the contact ORIGINS (the
same points f_ext_body takes the wrench about) and their 3 x NV tangent Jacobians
(`[3*NV*f + 3*vi + row]`, tangent `[v_lin; omega; joints]` in the pin LOCAL chart),
as `grid::contact_frame_positions[_gradient]_device` + the caller-scratch
`grid_plant::contact_frame_positions[_gradient]` wrappers, all riding the
multi-target emitters with the contact spec as the batch.

Gates: positions vs the RBDReference world-FK oracle (Xw[jid] @ [offset, 1]);
Jacobian vs central differences of that oracle under the local/local retraction;
thread invariance (1/32/256); TIER_MINIMAL spill bit-identical; plant wrappers ==
device functions; and the build is a DYNAMICS-ONLY subset (no kinematics
algorithm), proving the family is self-contained.
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from grid_codegen import GRiDCodeGenerator
from grid_codegen.algorithms._f_ext_contact import contact_frames_from_urdf
from RBDReference import RBDReference
from URDFParser import URDFParser
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch

REPO = Path(__file__).resolve().parents[2]
RUNNER_SOURCE = Path(__file__).with_name("cuda_contact_frame_positions_runner.cu")
CELLS = [
    ("iiwa14", False, ["iiwa_joint_ee", "tool0_joint"]),
    ("go2", True, ["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"]),
]


def _robot(name, floating):
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(str(REPO / "config" / "robot_assets" / f"{name}.urdf"), floating_base=floating)


def _q_for(robot, floating, seed=20260920):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1.0, 1.0, robot.get_num_pos())
    if floating:
        quat = rng.standard_normal(4)
        q[3:7] = quat / np.linalg.norm(quat)
    return q


def _qmul(a, b):
    x1, y1, z1, w1 = a; x2, y2, z2, w2 = b
    return np.array([w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2, w1*w2 - x1*x2 - y1*y2 - z1*z2])


def _qexp(v):
    th = np.linalg.norm(v)
    return np.array([0, 0, 0, 1.0]) if th < 1e-14 else np.concatenate([v / th * np.sin(th / 2), [np.cos(th / 2)]])


def _R(qu):
    x, y, z, w = qu
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def _retract(q, d, floating):
    """q (+) d in the pin chart: floating root linear LOCAL + angular LOCAL, joints additive."""
    out = q.copy()
    if not floating:
        return out + d
    out[:3] = q[:3] + _R(q[3:7]) @ d[:3]
    out[3:7] = _qmul(q[3:7], _qexp(d[3:6]))
    out[7:] = q[7:] + d[6:]
    return out


def _oracle_positions(robot, contacts, q):
    Xw, _ = RBDReference(robot)._frame_world_placement_and_chain(q)
    out = np.zeros((len(contacts), 3))
    for f, c in enumerate(contacts):
        out[f] = (np.asarray(Xw[int(c["jid"])], dtype=np.float64) @ np.array([*c["offset"], 1.0]))[:3]
    return out


def _build(robot, contacts, build_dir):
    gen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid")
    with contextlib.redirect_stdout(io.StringIO()):
        gen.gen_all_code(algorithm_list=["inverse_dynamics", "forward_dynamics"],   # NO kinematics algorithm
                         contact_frames=contacts, output_path=str(build_dir / "grid.cuh"),
                         enable_mujoco_kernels=False)
    runner = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner)
    exe = build_dir / "runner.exe"
    arch = _detect_cuda_arch()
    r = subprocess.run(["nvcc", "-std=c++17", "-O2", f"-arch=sm_{arch}", "-I", str(build_dir),
                        "-o", str(exe), str(runner)], capture_output=True, text=True)
    assert r.returncode == 0, f"runner compilation failed:\n{r.stderr[-4000:]}"
    return exe


def _parse(stdout, nf, nv):
    P = np.zeros((nf, 3)); J = np.zeros((nf, nv, 3)); checks = {}
    for line in stdout.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "P":
            P[int(p[1])] = [float(x) for x in p[2:5]]
        elif p[0] == "J":
            J[int(p[1]), int(p[2])] = [float(x) for x in p[3:6]]
        elif p[0] in ("THREADINV", "SPILLDIFF", "PLANTDIFF"):
            checks[p[0]] = float(p[1].split("=")[1])
    return P, J, checks


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("name,floating,frames", CELLS, ids=[f"{c[0]}-{'floating' if c[1] else 'fixed'}" for c in CELLS])
def test_contact_frame_positions_and_jacobian(tmp_path, name, floating, frames):
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not on PATH")
    robot = _robot(name, floating)
    contacts = contact_frames_from_urdf(robot, frames)
    nf, nv = len(contacts), robot.get_num_vel()
    build_dir = tmp_path / f"{name}_contact_positions"
    build_dir.mkdir()
    exe = _build(robot, contacts, build_dir)

    q = _q_for(robot, floating)
    (build_dir / "q.txt").write_text("\n".join(f"{v:.17g}" for v in q) + "\n")
    r = subprocess.run([str(exe)], capture_output=True, text=True, cwd=build_dir)
    assert r.returncode == 0, f"runner self-checks failed:\n{r.stdout}\n{r.stderr}"
    P, J, checks = _parse(r.stdout, nf, nv)
    assert checks["THREADINV"] == 0.0 and checks["SPILLDIFF"] == 0.0 and checks["PLANTDIFF"] == 0.0, checks

    np.testing.assert_allclose(P, _oracle_positions(robot, contacts, q), rtol=1e-10, atol=1e-10,
                               err_msg=f"{name}: contact-frame world positions != RBDReference host map")
    eps = 1e-6
    for vi in range(nv):
        d = np.zeros(nv); d[vi] = eps
        fd = (_oracle_positions(robot, contacts, _retract(q, d, floating))
              - _oracle_positions(robot, contacts, _retract(q, -d, floating))) / (2 * eps)
        np.testing.assert_allclose(J[:, vi, :], fd, rtol=1e-6, atol=1e-7,
                                   err_msg=f"{name}: d(contact pos)/d(tangent {vi}) != central differences")
