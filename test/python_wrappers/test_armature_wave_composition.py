"""Cross-branch contracts: reduced armature, variable coupling and frame twins.

Small fixtures keep this merge regression inexpensive. The public default
registry remains covered separately by test_armature_default_registration.
"""
import contextlib
import io
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import grid_rbd
from grid_codegen.armature import supported_slots
from RBDReference import RBDReference
from URDFParser import URDFParser
from test.cuda_equivalents.polynomial_coupling_model import write_polynomial_chain

pytestmark = pytest.mark.python_wrappers
ALGOS = ["inverse_dynamics", "crba", "minv", "aba", "forward_dynamics",
         "inverse_dynamics_gradient", "forward_dynamics_gradient", "energy",
         "frame_velocity", "frame_acceleration", "frame_velocity_gradient",
         "frame_acceleration_gradient", "integrator", "integrator_gradient"]
DT = .002


def close(actual, expected):
    expected = np.asarray(expected, dtype=float)
    np.testing.assert_allclose(actual, expected, rtol=2e-4,
                               atol=2e-4 * max(1., float(np.abs(expected).max())))


@pytest.fixture(scope="module", params=[("polynomial", False, False),
    ("polynomial", False, True), ("linear", True, False), ("linear", True, True),
    ("ordinary", True, False), ("ordinary", True, True)],
    ids=["poly-baked", "poly-runtime", "floating-mimic-baked", "floating-mimic-runtime",
         "floating-baked", "floating-runtime"])
def composed(request, tmp_path_factory):
    kind, floating, runtime = request.param
    directory = tmp_path_factory.mktemp("armature_wave")
    path = write_polynomial_chain(directory, kind=kind, joint_dynamics=floating)
    if kind == "ordinary":
        tree = ET.parse(path)
        tree.getroot().set("name", "ordinary_chain_damped")
        for joint in tree.getroot().findall("joint"):
            mimic = joint.find("mimic")
            if mimic is not None:
                joint.remove(mimic)
        tree.write(path)
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(path), floating_base=floating)
    ref = RBDReference(robot, use_joint_dynamics=True,
                       friction_model="tanh", friction_width=.1)
    a = np.zeros(robot.get_num_vel(), np.float32)
    slots = supported_slots(robot)
    a[slots] = np.linspace(.2, .4, len(slots))
    ref.set_armature(a)
    h = grid_rbd.register_robot(f"armature_wave_{kind}_{runtime}", str(path),
        floating_base=floating, armature=a, runtime_armature=runtime,
        use_joint_dynamics=True, runtime_joint_dynamics=floating,
        friction_model="tanh", friction_width=.1, algorithm_list=ALGOS,
        enable_mujoco_kernels=True, max_batch_size=2)
    rng = np.random.default_rng(11071)
    q = rng.uniform(-.4, .4, (2, h.num_joints)).astype(np.float32)
    if floating:
        q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
        assert np.linalg.norm(ref._joint_dynamics_bias(np.ones(h.num_vel))) > .1
    v, u = [rng.uniform(-.4, .4, (2, h.num_vel)).astype(np.float32) for _ in range(2)]
    if runtime:
        # A live update must reach every composed algorithm, not just initialization.
        a = 1.25 * a
        h.set_armature(a)
        ref.set_armature(a)
    yield h, ref, q, v, u
    h.close()


def test_values_frames_and_integration(composed):
    h, ref, q, v, u = composed
    leaf = ref.robot.get_joint_by_id(int(ref.robot.get_leaf_nodes()[0])).get_name()
    for threads in (1, 32, 128):
        h.set_threads_per_block(threads)
        mass, inv = h.crba(q), h.minv(q)
        fd, aba = h.forward_dynamics(q, v, u), h.aba(q, v, u)
        tau, energy = h.inverse_dynamics(q, v, u), h.energy(q, v)
        step = h.integrator(q, v, u, DT, integrator_type="euler")
        grad = h.integrator_gradient(q, v, u, DT, integrator_type="euler")
        for k in range(2):
            qr, vr, ur = [x[k].astype(float) for x in (q, v, u)]
            close(mass[k], ref.crba(qr))
            close(inv[k], ref.minv(qr))
            close(fd[k], ref.forward_dynamics(qr, vr, ur))
            close(aba[k], ref.forward_dynamics(qr, vr, ur))
            close(tau[k], ref.inverse_dynamics(qr, vr, ur)[0])
            ke, pe = ref.kinetic_energy(qr, vr), ref.potential_energy(qr)
            close(energy[k], [ke, pe, ke + pe])
            close(step[k], ref.integrator(qr, vr, ur, DT, integrator_type="euler"))
            close(grad[k], ref.integrator_gradient(qr, vr, ur, DT, integrator_type="euler"))
        for frame in ("LOCAL", "WORLD", "LOCAL_WORLD_ALIGNED"):
            vel = h.frame_velocity(q, v, reference_frame=frame)
            acc = h.frame_acceleration(q, v, fd, reference_frame=frame)
            vg = h.frame_velocity_gradient(q, v, reference_frame=frame)
            ag = h.frame_acceleration_gradient(q, v, fd, reference_frame=frame)
            for k in range(2):
                qr, vr, ar = [x[k].astype(float) for x in (q, v, fd)]
                close(vel[k], ref.frame_velocity(qr, vr, leaf, frame))
                close(acc[k], ref.frame_acceleration(qr, vr, ar, leaf, frame))
                close(vg[k], np.hstack(ref.frame_velocity_derivatives(qr, vr, leaf, frame)))
                close(ag[k], np.hstack(ref.frame_acceleration_derivatives(qr, vr, ar, leaf, frame)[1:]))


def _reference_step(ref, q, v, u, mjx):
    """Independent double-precision Euler step, not finite differences of CUDA."""
    if not mjx or not ref.robot.floating_base:
        q = q.copy()
        if ref.robot.floating_base:
            q[3:7] /= np.linalg.norm(q[3:7])
        return ref.integrator(q, v, u, DT, integrator_type="euler")
    # MuJoCo stores wxyz, world linear velocity, local angular velocity.
    pin_q = q.copy()
    pin_q[3:7] = q[[4, 5, 6, 3]] / np.linalg.norm(q[3:7])
    x, y, z, w = pin_q[3:7]
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    vp, up = v.copy(), u.copy()
    vp[:3], up[:3] = R.T @ v[:3], R.T @ u[:3]
    ap = ref.forward_dynamics(pin_q, vp, up)
    am = ap.copy()
    am[:3] = R @ (ap[:3] + np.cross(vp[3:6], vp[:3]))
    qn = ref.integrate(pin_q, DT * vp)
    qn[:3] = q[:3] + DT * v[:3]
    qn[3:7] = qn[[6, 3, 4, 5]]
    return np.r_[qn, v + DT * am]


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("mjx", [False, True], ids=["native", "mujoco"])
def test_integrator_backward_and_torque_cotangent(composed, backend, mjx):
    h, ref, q, v, u = composed
    if mjx and h.floating_base and any(getattr(j, "is_mimic", False)
                                      for j in ref.robot.get_joints_ordered_by_id()):
        # This existing support boundary is not broadened by the merge.
        with pytest.raises(ValueError, match="mjx kernel twins"):
            grid_rbd.get_robot(h.name, backend=backend, output_convention="mujoco")
        return
    qr, vr, ur = [x[0].astype(float) for x in (q, v, u)]
    if mjx and h.floating_base:
        qr[3:7] = qr[[6, 3, 4, 5]]
    handle = grid_rbd.get_robot(h.name, backend=backend,
                               output_convention="mujoco" if mjx else "pinocchio")
    weights = np.linspace(.3, 1.1, h.num_joints + h.num_vel)
    eps = 1e-5
    numeric = np.array([(weights @ _reference_step(ref, qr + eps*e, vr, ur, mjx)
                        - weights @ _reference_step(ref, qr - eps*e, vr, ur, mjx))/(2*eps)
                       for e in np.eye(h.num_joints)])
    try:
        if backend == "torch":
            import torch
            t = lambda x: torch.as_tensor(x, dtype=torch.float32, device="cuda")
            tq = t(qr[None]).requires_grad_(True)
            value = handle.integrator(tq, t(vr[None]), t(ur[None]), DT, integrator_type="euler")
            (value * t(weights)).sum().backward()
            actual = tq.grad.cpu().numpy()[0]
            actual_value = value.detach().cpu().numpy()[0]
            tu = t(ur[None]).requires_grad_(True)
            handle.forward_dynamics(t(qr[None]), t(vr[None]), tu).sum().backward()
            torque = tu.grad.cpu().numpy()[0]
        else:
            import jax
            import jax.numpy as jnp
            def loss(x):
                return (handle.integrator(x, jnp.asarray(vr[None], dtype=jnp.float32),
                    jnp.asarray(ur[None], dtype=jnp.float32), DT, integrator_type="euler") * weights).sum()
            actual = np.asarray(jax.grad(loss)(jnp.asarray(qr[None], dtype=jnp.float32)))[0]
            actual_value = np.asarray(handle.integrator(jnp.asarray(qr[None], dtype=jnp.float32),
                jnp.asarray(vr[None], dtype=jnp.float32), jnp.asarray(ur[None], dtype=jnp.float32),
                DT, integrator_type="euler"))[0]
            torque = np.asarray(jax.grad(lambda x: handle.forward_dynamics(
                jnp.asarray(qr[None], dtype=jnp.float32), jnp.asarray(vr[None], dtype=jnp.float32),
                x).sum())(jnp.asarray(ur[None], dtype=jnp.float32)))[0]
        close(actual, numeric)
        close(actual_value, _reference_step(ref, qr, vr, ur, mjx))
        # Torque pullback must use symmetric Minv once, not double its off-diagonal.
        if not mjx or not h.floating_base:
            close(torque, ref.minv(qr).sum(axis=0))
        else:
            def acceleration(tau):
                return (_reference_step(ref, qr, vr, tau, True)[h.num_joints:] - vr) / DT
            expected = np.array([(acceleration(ur+eps*e).sum()-acceleration(ur-eps*e).sum())/(2*eps)
                                 for e in np.eye(h.num_vel)])
            close(torque, expected)
    finally:
        handle.close()
