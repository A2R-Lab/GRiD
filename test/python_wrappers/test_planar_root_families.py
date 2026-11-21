"""Receipt coverage for a planar root with continuous and mimic joints.

The fixture (test/python_wrappers/planar_root_model.py) is a mobile manipulator: a planar root,
a continuous wheel, a prismatic torso, two limited revolute arm joints, a
gripper finger with a mimic twin and a fixed tool frame. The planar root is
obtained in two ways, from a planar joint declared in the URDF and from
register_robot(planar_base=True) on the same body, and both must give the same
model descriptor and the same results.

The float32 handle is compared with the float64 RBDReference oracle at 1, 32
and 128 threads per block, on states whose base block ranges from zero to
translations of tens of metres and yaw beyond one turn. The collision family
is not on the handle; it is driven as the existing complete-collision gate
drives it, through grid-generate (with -p for the option variant) and the
shared runner, against the oracle's sphere positions.
"""
import contextlib
import io
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import grid_rbd
from grid_codegen.algorithms._collision import build_sphere_tiers
from grid_codegen.algorithms._spherize import spherize_urdf
from grid_codegen.model_descriptor import planar_base_block
from grid_rbd._model import parse_model
from RBDReference import RBDReference
from test.python_wrappers import planar_root_model as model
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.executable_cache import cached_nvcc_executable

pytestmark = pytest.mark.python_wrappers
BATCH = 6
DT = 0.002
THREADS = (1, 32, 128)
TOOL = [model.TOOL_FRAME]
ROOT = Path(__file__).resolve().parents[2]


def close(actual, expected):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-4,
                               atol=2e-4 * max(1., float(np.abs(expected).max())))


def close_pose(actual, expected):
    """[xyz, rpy] comparison; an angle and the same angle plus a turn are equal."""
    actual = np.array(actual, dtype=float).reshape(6)
    expected = np.asarray(expected, dtype=float).reshape(6)
    actual[3:] = expected[3:] + (actual[3:] - expected[3:] + np.pi) % (2 * np.pi) - np.pi
    close(actual, expected)


def _parse(path, option):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return parse_model(path, planar_base=option)


@pytest.fixture(scope="module")
def directory(tmp_path_factory):
    return tmp_path_factory.mktemp("planar_root")


@pytest.fixture(scope="module", params=["declared", "option"])
def planar(request, directory):
    variant = request.param
    path, option = model.write_planar_root_model(directory, variant)
    ref = RBDReference(_parse(path, option))
    h = grid_rbd.register_robot(f"planar_root_{variant}", str(path), planar_base=option,
                                ee_joint_names=TOOL, max_batch_size=BATCH)
    q, v, u = model.sample_states(BATCH, 20261121, dtype=np.float32)
    assert np.abs(q[:, :2]).max() > 50 and np.abs(q[:, 2]).max() > np.pi
    yield h, ref, q, v, u, variant
    h.close()


@pytest.fixture(scope="module")
def results():
    """Outputs of the first variant, for the bitwise comparison of the second."""
    return {}


def _same_across_variants(results, variant, key, value):
    value = np.array(value)
    if key in results and results[key][0] != variant:
        np.testing.assert_array_equal(value, results[key][1], err_msg=key)
    results.setdefault(key, (variant, value))


def test_model_identity(planar, results):
    h, ref, *_ , variant = planar
    descriptor = h.model_descriptor
    assert planar_base_block(descriptor) == (0, 0, 3)
    assert h.planar_base is True and h.floating_base is False
    assert (h.num_joints, h.num_vel, h.num_ees) == (model.NQ, model.NV, 1)
    assert list(h.joint_names) == model.JOINT_ORDER
    assert h.model_descriptor["floating_base"] is False
    limits = h.joint_pos_limits
    assert limits[:3] == [[None, None], [None, None], None]
    assert limits[model.JOINT_ORDER.index("shoulder")] == [-1.2, 1.5]
    assert limits[model.JOINT_ORDER.index("wheel_spin")] is None
    results.setdefault("descriptor_id", (variant, descriptor["descriptor_id"]))
    assert results["descriptor_id"][1] == descriptor["descriptor_id"]
    assert ref.robot.planar_base is (variant == "option")


@pytest.mark.parametrize("threads", THREADS)
def test_dynamics(planar, results, threads):
    h, ref, q, v, u, variant = planar
    h.set_threads_per_block(threads)
    a = (0.7 * u).astype(np.float32)
    out = dict(crba=h.crba(q), minv=h.minv(q), inverse_dynamics=h.inverse_dynamics(q, v, a),
               forward_dynamics=h.forward_dynamics(q, v, u),
               inverse_dynamics_gradient=h.inverse_dynamics_gradient(q, v, a),
               forward_dynamics_gradient=h.forward_dynamics_gradient(q, v, u))
    for key, value in out.items():
        _same_across_variants(results, variant, f"{key}@{threads}", value)
    for k in range(BATCH):
        qr, vr, ur, ar = [x[k].astype(float) for x in (q, v, u, a)]
        close(out["crba"][k], ref.crba(qr))
        close(out["minv"][k], ref.minv(qr))
        close(out["inverse_dynamics"][k], ref.inverse_dynamics(qr, vr, ar)[0])
        close(out["forward_dynamics"][k], ref.forward_dynamics(qr, vr, ur))
        close(out["inverse_dynamics_gradient"][k], ref.inverse_dynamics_gradient(qr, vr, ar))
        close(out["forward_dynamics_gradient"][k],
              np.hstack(ref.forward_dynamics_gradient(qr, vr, ur)))


@pytest.mark.parametrize("integrator_type", ["euler", "rk4"])
@pytest.mark.parametrize("threads", THREADS)
def test_integrator_and_plant_step(planar, results, threads, integrator_type):
    h, ref, q, v, u, variant = planar
    h.set_threads_per_block(threads)
    x = np.concatenate([q, v], axis=1)
    kw = dict(integrator_type=integrator_type)
    out = dict(integrator=h.integrator(q, v, u, DT, **kw),
               integrator_gradient=h.integrator_gradient(q, v, u, DT, **kw),
               plant_step=h.plant_step(x, u, DT, **kw),
               plant_step_gradient=h.plant_step_gradient(x, u, DT, **kw))
    for key, value in out.items():
        _same_across_variants(results, variant, f"{key}:{integrator_type}@{threads}", value)
    for k in range(BATCH):
        qr, vr, ur = [x_[k].astype(float) for x_ in (q, v, u)]
        close(out["integrator"][k], ref.integrator(qr, vr, ur, DT, **kw))
        close(out["integrator_gradient"][k], ref.integrator_gradient(qr, vr, ur, DT, **kw))
        close(out["plant_step"][k], ref.plant_step(qr, vr, ur, DT, **kw))
        close(out["plant_step_gradient"][k], ref.plant_step_gradient(qr, vr, ur, DT, **kw))


@pytest.mark.parametrize("threads", THREADS)
def test_end_effector_family(planar, results, threads):
    h, ref, q, v, u, variant = planar
    h.set_threads_per_block(threads)
    rng = np.random.default_rng(7)
    p_des = (q[:, :3] * [1, 1, 0] + rng.uniform(-0.5, 0.5, (BATCH, 3))).astype(np.float32)
    weight = rng.uniform(0.5, 2.0, (BATCH, 3)).astype(np.float32)
    pose, gradient, hessian = (h.end_effector_pose(q), h.end_effector_pose_gradient(q),
                               h.end_effector_pose_hessian(q))
    value, cost_gradient, cost_hessian = h.ee_pos_cost(q, p_des, weight)
    for key, item in dict(pose=pose, gradient=gradient, hessian=hessian, value=value,
                          cost_gradient=cost_gradient, cost_hessian=cost_hessian).items():
        _same_across_variants(results, variant, f"ee_{key}@{threads}", item)
    nv = model.NV
    for k in range(BATCH):
        qr = q[k].astype(float)
        expected_pose = np.asarray(ref.end_effector_pose(qr, ee_joint_names=TOOL)[0], float).reshape(6)
        # rpy and its derivatives are singular at pitch = +-pi/2; the samples stay clear.
        assert abs(abs(expected_pose[4]) - np.pi / 2) > 0.1
        jacobian = np.asarray(ref.end_effector_pose_gradient(qr, ee_joint_names=TOOL)[0], float)
        close_pose(pose[k], expected_pose)
        close(gradient[k], jacobian)
        close(hessian[k], ref.end_effector_pose_hessian_analytic(qr, ee_joint_names=TOOL)[0])
        residual = expected_pose[:3] - p_des[k].astype(float)
        w = weight[k].astype(float)
        expected_gradient = np.zeros(2 * nv)
        expected_gradient[:nv] = jacobian[:3].T @ (w * residual)
        expected_hessian = np.zeros((2 * nv, 2 * nv))
        expected_hessian[:nv, :nv] = jacobian[:3].T @ (w[:, None] * jacobian[:3])
        close(value[k], 0.5 * np.sum(w * residual * residual))
        close(cost_gradient[k], expected_gradient)
        close(cost_hessian[k], expected_hessian)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_backend_handles(planar, backend):
    h, ref, q, v, u, variant = planar
    handle = grid_rbd.get_robot(h.name, backend=backend)
    try:
        assert handle.planar_base is True and handle.floating_base is False
        if backend == "torch":
            import torch
            to = lambda x: torch.as_tensor(x, dtype=torch.float32, device="cuda")
            out = handle.forward_dynamics(to(q), to(v), to(u)).cpu().numpy()
        else:
            import jax.numpy as jnp
            to = lambda x: jnp.asarray(x, dtype=jnp.float32)
            out = np.asarray(handle.forward_dynamics(to(q), to(v), to(u)))
        for k in range(BATCH):
            close(out[k], ref.forward_dynamics(*[x[k].astype(float) for x in (q, v, u)]))
    finally:
        handle.close()


# ---- collision family -------------------------------------------------------

@pytest.fixture(scope="module", params=["declared", "option"])
def collision(request, directory):
    from grid_codegen.cli import main
    variant = request.param
    path, option = model.write_planar_root_model(directory, variant)
    out = directory / f"collision_{variant}"
    out.mkdir()
    spheres = spherize_urdf(str(path), 0.06, str(out / "spheres.urdf"))
    argv = ["grid-generate", str(path), "--spherized-urdf", str(spheres),
            "--algorithm-list", "kinematics", "-o", str(out / "grid.cuh")] + (["-p"] if option else [])
    with pytest.MonkeyPatch.context() as patch, contextlib.redirect_stdout(io.StringIO()):
        patch.setattr(sys, "argv", argv)
        main()
    robot = _parse(path, option)
    spec = build_sphere_tiers(robot, {"fine": spheres})["fine"]
    arch = _detect_cuda_arch()
    exe, _ = cached_nvcc_executable(
        [ROOT / "test/cuda_equivalents/cuda_collision_complete_runner.cu", out / "grid.cuh"],
        ["-std=c++17", "-O2", "-gencode", f"arch=compute_{arch},code=sm_{arch}"],
        exe_name="planar_root_collision.exe", fallback_dir=out,
        include_dirs=[ROOT / "grid_codegen/collision"], what="planar root collision model")
    return exe, RBDReference(robot), spec, (out / "grid.cuh").read_bytes()


def test_collision_header_is_the_same_for_both_variants(collision, results):
    header = collision[3]
    assert b"inline constexpr bool planar_base = true;" in header
    assert b"namespace grid_collision {" in header
    assert results.setdefault("collision_header", header) == header


@pytest.mark.parametrize("threads", [1, 32, 100])
def test_collision_spheres_and_verdict(collision, threads):
    exe, ref, spec, _ = collision
    radius = np.asarray(spec["radius"], float)
    offsets = np.asarray(spec["offset"], float).reshape(-1, 3)
    assert len(radius) > 20
    q_all = model.sample_states(BATCH, 20261121, dtype=np.float32)[0].astype(float)
    for q in q_all:
        world, _ = ref._frame_world_placement_and_chain(q)
        expected = np.array([(world[a] @ np.r_[p, 1])[:3] for a, p in zip(spec["anchor"], offsets)])
        gaps = [np.sum((expected[i] - expected[j])**2) - (radius[i] + radius[j])**2
                for i, start, end in spec["self_cc_ranges"] for j in range(start, end + 1)]
        # The float32 verdict is only defined away from a grazing pair.
        assert min(abs(g) for g in gaps) > 1e-5
        self_hit = min(gaps) < 0
        for obstacle in [np.array([1000., 1000., 1000., .1]), np.r_[expected[0], .1],
                         np.r_[expected[-1] + [0.0, 0.0, 0.5], .1]]:
            obstacle = obstacle.astype(np.float32).astype(float)
            env_gaps = np.sum((expected - obstacle[:3])**2, axis=1) - (radius + obstacle[3])**2
            assert np.abs(env_gaps).min() > 1e-5
            result = subprocess.run([str(exe), str(threads)],
                                    input=" ".join("%.17g" % x for x in np.r_[q, obstacle]),
                                    capture_output=True, text=True, check=True)
            lines = result.stdout.strip().splitlines()
            actual = np.array(lines[-1].split(), dtype=float).reshape(-1, 3)
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
            assert int(lines[-2]) == int(not (self_hit or np.any(env_gaps < 0)))
