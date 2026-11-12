"""Armature composed with mutable inertia, friction, integration and curvature."""
import contextlib
import io
from pathlib import Path

import numpy as np
import pytest
import grid_rbd
from grid_assets import robot_urdf
from grid_codegen.armature import supported_slots
from URDFParser import URDFParser
from RBDReference import RBDReference

pytestmark = pytest.mark.python_wrappers


@pytest.fixture(scope="module", params=[("fr3", False, "float32"),
    ("go2", True, "float32"), ("damping_friction_arm",False,"float32"),
    pytest.param(("iiwa14", False, "float64"), marks=pytest.mark.fp64_alpha)],
    ids=["mimic", "floating", "nonzero-tanh", "fixed-fp64"])
def composed(request):
    name, floating, dtype = request.param
    path = (Path(__file__).resolve().parents[2]/"external/URDFParser/tests/fixtures/damping_friction_arm.urdf"
            if name == "damping_friction_arm" else robot_urdf(name))
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(path), floating_base=floating)
    ref = RBDReference(robot, use_joint_dynamics=True, friction_model="tanh", friction_width=.1)
    a = np.zeros(robot.get_num_vel(), dtype=dtype)
    a[supported_slots(robot)] = .3
    ref.set_armature(a)
    h = grid_rbd.register_robot("composed_armature_"+name, str(path),
        dtype=dtype, floating_base=floating, armature=a, runtime_armature=True,
        runtime_inertia=True, use_joint_dynamics=True, runtime_joint_dynamics=True,
        friction_model="tanh", friction_width=.1, enable_mujoco_kernels=False,
        max_batch_size=2, algorithm_list=["forward_dynamics_armature_gradient",
          "forward_dynamics_gradient", "fdsva_so", "integrator_gradient", "energy",
          "inverse_dynamics_regressor", "kinetic_energy_regressor", "com", "ccrba", "dccrba"])
    # Scale every body's spatial inertia, independently on the CPU oracle.
    scales=np.linspace(1.1,1.3,h.num_bodies)
    h.set_inertia_params(h.inertia_params * scales[:,None])
    for link, scale in zip(robot.get_links_ordered_by_id()[1:],scales):
        link.set_spatial_inertia(link.get_spatial_inertia() * scale)
    rng = np.random.default_rng(927)
    q = rng.uniform(-.3, .3, (2, h.num_joints)).astype(dtype)
    if floating:
        q[:,3:7] /= np.linalg.norm(q[:,3:7], axis=1, keepdims=True)
    v, u = [rng.uniform(-.3,.3,(2,h.num_vel)).astype(dtype) for _ in range(2)]
    if name == "damping_friction_arm":
        assert np.linalg.norm(ref._joint_dynamics_bias(v[0])) > .01
    yield h, ref, a, q, v, u
    h.close()


def close(actual, expected, dtype):
    expected = np.asarray(expected)
    tol = 3e-4 if dtype == "float32" else 2e-9
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol*max(1.,float(np.abs(expected).max())))


def test_mutable_inertia_armature_and_friction(composed):
    h, ref, a, q, v, u = composed
    fe=np.random.default_rng(97).uniform(-.2,.2,(2,6*h.num_bodies)).astype(h.dtype)
    fd = h.forward_dynamics(q,v,u,f_ext=fe)
    dg = h.forward_dynamics_armature_gradient(q,v,u,f_ext=fe)
    energy = h.energy(q,v)
    for k in range(2):
        qr, vr, ur = [x[k].astype(float) for x in (q,v,u)]
        force=fe[k].reshape(-1,6).astype(float)
        close(fd[k], ref.forward_dynamics(qr,vr,ur,f_ext=force), h.dtype)
        close(dg[k], ref.forward_dynamics_armature_gradient(qr,vr,ur,f_ext=force), h.dtype)
        ke,pe=ref.kinetic_energy(qr,vr),ref.potential_energy(qr)
        close(energy[k], [ke,pe,ke+pe], h.dtype)


def test_centroidal_reads_the_same_inertia_table(composed):
    h, ref, a, q, v, u = composed
    com,jcom=h.com(q)
    matrix,momentum=h.ccrba(q,v)
    derivative=h.dccrba(q)
    for k in range(2):
        qr,vr=q[k].astype(float),v[k].astype(float)
        close(com[k],ref.com(qr),h.dtype)
        close(jcom[k],ref.jacobian_com(qr),h.dtype)
        ar,hr=ref.ccrba(qr,vr)
        close(matrix[k],ar,h.dtype)
        close(momentum[k],hr,h.dtype)
        close(derivative[k],ref.dccrba(qr),h.dtype)


def test_parameter_blocks_reconstruct_torque_and_energy(composed):
    h,ref,a,q,v,acc=composed
    forces=np.random.default_rng(407).uniform(-.2,.2,(2,6*h.num_bodies)).astype(h.dtype)
    inertia=h.inverse_dynamics_regressor(q,v,acc)
    rotor=h.armature_regressor(acc)
    energy=h.kinetic_energy_regressor(q,v)
    torque=h.inverse_dynamics(q,v,acc,f_ext=forces)
    pi=np.asarray(ref.robot.get_inertia_params_ordered_by_id()[1:]).ravel()
    assert inertia.shape==(2,h.num_vel,10*h.num_bodies)
    for k in range(2):
        qr,vr,ar=[x[k].astype(float) for x in (q,v,acc)]
        force=forces[k].reshape(-1,6).astype(float)
        zero=np.zeros(h.num_vel)
        loaded=ref.inverse_dynamics(qr,zero,zero,f_ext=force)[0]
        unloaded=ref.inverse_dynamics(qr,zero,zero)[0]
        known_offset=loaded-unloaded+ref._joint_dynamics_bias(vr)
        close(inertia[k],ref.inverse_dynamics_regressor(qr,vr,ar),h.dtype)
        close(torque[k],inertia[k]@pi+rotor[k]@a[list(h.armature_slots)]+known_offset,h.dtype)
        close(energy[k]@pi+.5*np.sum(a*vr**2),ref.kinetic_energy(qr,vr),h.dtype)


def test_native_energy_device_uses_live_tables(composed, tmp_path):
    import subprocess
    import torch
    h,ref,a,q,v,u=composed
    root=Path(__file__).resolve().parents[2]
    binary=tmp_path/"native_energy"
    cmd=["nvcc","-std=c++17","-O3","-arch=sm_%d%d" % torch.cuda.get_device_capability(),
         "-I"+str(Path(h._so_path).parent),"-I"+str(root/"external/GLASS"),
         "-I"+str(root/"external/GLASS/src"),
         str(root/"test/cuda_equivalents/cuda_runtime_energy_runner.cu"),"-o",str(binary)]
    if h.dtype=="float64":
        cmd.append("-DTEST_DOUBLE")
    built=subprocess.run(cmd,text=True,capture_output=True,timeout=180)
    assert built.returncode==0,built.stdout+built.stderr
    pi=np.asarray(ref.robot.get_inertia_params_ordered_by_id()[1:]).ravel()
    for qr,vr in zip(q,v):
        data=" ".join(map(str,np.concatenate([qr,vr,pi,a])))
        result=subprocess.run([str(binary)],input=data,text=True,capture_output=True,timeout=30)
        assert result.returncode==0,result.stdout+result.stderr
        ke,pe=ref.kinetic_energy(qr.astype(float),vr.astype(float)),ref.potential_energy(qr.astype(float))
        close(np.fromstring(result.stdout,sep=" "),[ke,pe,ke+pe],h.dtype)


@pytest.mark.parametrize("scheme", ["euler", "rk4"])
def test_integrator_and_plant(composed, scheme):
    h, ref, a, q, v, u = composed
    dt = .002
    step = h.integrator(q,v,u,dt,integrator_type=scheme)
    grad = h.integrator_gradient(q,v,u,dt,integrator_type=scheme)
    x = np.concatenate([q,v],axis=1)
    close(h.plant_step(x,u,dt,integrator_type=scheme), step, h.dtype)
    close(h.plant_step_gradient(x,u,dt,integrator_type=scheme), grad, h.dtype)
    for k in range(2):
        qr, vr, ur = [x[k].astype(float) for x in (q,v,u)]
        close(step[k], ref.integrator(qr,vr,ur,dt,integrator_type=scheme), h.dtype)
        close(grad[k], ref.integrator_gradient(qr,vr,ur,dt,integrator_type=scheme), h.dtype)


def test_second_order_with_runtime_model_pointer(composed):
    h, ref, a, q, v, u = composed
    got = h.fdsva_so(q,v,u)
    for k in range(2):
        oracle = ref.fdsva_so(*(x[k].astype(float) for x in (q,v,u)))
        # FD slab order matches RBDReference: qq, vq, vv, uq (not ID order).
        for actual, expected in zip(got, oracle):
            close(actual[k], expected, h.dtype)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_configuration_cotangent(composed, backend):
    h, ref, a, q, v, u = composed
    handle = grid_rbd.get_robot(h.name, backend=backend)
    qr, vr, ur = [x[0].astype(float) for x in (q,v,u)]
    # Ambient-q finite differences normalize the quaternion just as the wrappers do.
    def loss(x):
        x = x.copy()
        if h.floating_base:
            x[3:7] /= np.linalg.norm(x[3:7])
        return ref.forward_dynamics(x,vr,ur).sum()
    eps = 1e-5
    numeric = np.array([(loss(qr+eps*e)-loss(qr-eps*e))/(2*eps) for e in np.eye(h.num_joints)])
    if backend == "torch":
        import torch
        t = lambda x: torch.as_tensor(x, device="cuda", dtype=getattr(torch,h.dtype))
        tq = t(q[:1]).requires_grad_(True)
        handle.forward_dynamics(tq,t(v[:1]),t(u[:1])).sum().backward()
        got = tq.grad.cpu().numpy()[0]
    else:
        import jax
        import jax.numpy as jnp
        with jax.enable_x64():
            got = np.asarray(jax.grad(lambda x: handle.forward_dynamics(x,jnp.asarray(v[:1]),jnp.asarray(u[:1])).sum())(jnp.asarray(q[:1])))[0]
    close(got,numeric,h.dtype)
