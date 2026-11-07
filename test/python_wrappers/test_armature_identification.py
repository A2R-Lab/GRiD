"""Independent parameter finite differences and native/backend ownership."""
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


@pytest.fixture(scope="module", params=[("iiwa14", False, "float32"),
    ("go2", True, "float32"), pytest.param(("fr3", True, "float64"), marks=pytest.mark.fp64_alpha)], ids=["fixed", "floating", "floating-mimic-fp64"])
def case(request):
    name, floating, dtype = request.param
    with contextlib.redirect_stdout(io.StringIO()):
        ref = RBDReference(URDFParser().parse(str(robot_urdf(name)), floating_base=floating))
    slots = supported_slots(ref.robot)
    a = np.zeros(ref.robot.get_num_vel(), dtype=dtype)
    a[slots] = np.linspace(.2, .5, len(slots))
    h = grid_rbd.register_robot("identification_"+name, str(robot_urdf(name)),
        floating_base=floating, dtype=dtype, armature=a, runtime_armature=True,
        algorithm_list=["forward_dynamics_armature_gradient", "inverse_dynamics_regressor", "crba"],
        max_batch_size=3, enable_mujoco_kernels=False)
    ref.set_armature(a)
    rng = np.random.default_rng(173)
    q = rng.uniform(-.5, .5, (3, h.num_joints)).astype(dtype)
    if floating:
        q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    v, u, acc = [rng.uniform(-.6, .6, (3, h.num_vel)).astype(dtype) for _ in range(3)]
    fe = rng.uniform(-.3, .3, (3, 6*h.num_bodies)).astype(dtype)
    yield h, ref, a, q, v, u, acc, fe
    h.close()


def close(x, y, dtype):
    tol = 2e-4 if dtype == "float32" else 2e-9
    np.testing.assert_allclose(x, y, rtol=tol, atol=tol * max(1., float(np.abs(y).max())))


@pytest.mark.parametrize("threads", [1, 32, 128])
def test_parameter_gradient_and_torque_identity(case, threads):
    h, ref, a, q, v, u, acc, fe = case
    h.set_threads_per_block(threads)
    got = h.forward_dynamics_armature_gradient(q, v, u, f_ext=fe)
    reg = h.armature_regressor(acc)
    for k in range(3):
        qr, vr, ur, ar = [x[k].astype(float) for x in (q, v, u, acc)]
        force = fe[k].reshape(-1, 6).astype(float)
        expected = ref.forward_dynamics_armature_gradient(qr, vr, ur, f_ext=force)
        close(got[k], expected, h.dtype)
        close(reg[k], ref.armature_regressor(ar), h.dtype)
        # Independently perturb the installed physical model, not the formula.
        finite = []
        for slot in h.armature_slots:
            step = 1e-5
            plus, minus = a.astype(float).copy(), a.astype(float).copy()
            plus[slot] += step; minus[slot] -= step
            ref.set_armature(plus)
            fp = ref.forward_dynamics(qr, vr, ur, f_ext=force)
            ref.set_armature(minus)
            fm = ref.forward_dynamics(qr, vr, ur, f_ext=force)
            finite.append((fp-fm)/(2*step))
        ref.set_armature(a)
        close(got[k], np.stack(finite, axis=1), h.dtype)
        assert np.linalg.norm(expected) > 1e-3
        # Known non-inertial offset includes external loads, not fitted inertia.
        tau, *_ = ref.inverse_dynamics(qr, vr, ar, f_ext=force)
        ref.set_armature(np.zeros_like(a))
        bare, *_ = ref.inverse_dynamics(qr, vr, ar, f_ext=force)
        ref.set_armature(a)
        close(tau-bare, reg[k] @ a[list(h.armature_slots)], h.dtype)


def test_backend_and_context_values(case):
    import jax
    import jax.numpy as jnp
    import torch
    from grid_rbd.jax import JaxRobotHandle
    from grid_rbd.torch import TorchRobotHandle
    h, ref, a, q, v, u, acc, fe = case
    h.set_threads_per_block(32)
    want = h.forward_dynamics_armature_gradient(q, v, u, f_ext=fe)
    th = TorchRobotHandle(h, Path(h._so_path).parent.name, h._so_path)
    tensors = [torch.as_tensor(x, device="cuda") for x in (q,v,u,acc,fe)]
    close(th.forward_dynamics_armature_gradient(*tensors[:3], f_ext=tensors[4]).cpu().numpy(), want, h.dtype)
    close(th.armature_regressor(tensors[3]).cpu().numpy(), h.armature_regressor(acc), h.dtype)
    jh = JaxRobotHandle(h, Path(h._so_path).parent.name, h._so_path)
    with jax.enable_x64():
        arrays = [jnp.asarray(x) for x in (q,v,u,acc,fe)]
        call = jax.jit(lambda q,v,u,f: jh.forward_dynamics_armature_gradient(q,v,u,f_ext=f))
        close(np.asarray(call(*arrays[:3], arrays[4])), want, h.dtype)
        close(np.asarray(jh.armature_regressor(arrays[3])), h.armature_regressor(acc), h.dtype)
    child = h.context()
    try:
        child.set_armature(a*2)
        other = child.forward_dynamics_armature_gradient(q,v,u,f_ext=fe)
        assert not np.allclose(other, want)
        with jax.enable_x64():
            close(np.asarray(call(*arrays[:3], arrays[4])), want, h.dtype)
    finally:
        child.close()


def test_native_collectives(case, tmp_path):
    import subprocess
    import torch
    h, ref, a, q, v, u, acc, fe = case
    root = Path(__file__).resolve().parents[2]
    source = root / "test/cuda_equivalents/cuda_armature_identification_runner.cu"
    binary = tmp_path / "native_identification"
    arch = "sm_%d%d" % torch.cuda.get_device_capability()
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch="+arch, "-I"+str(Path(h._so_path).parent),
           "-I"+str(root/"external/GLASS"), "-I"+str(root/"external/GLASS/src"),
           str(source), "-o", str(binary)]
    if h.dtype == "float64":
        cmd.append("-DTEST_DOUBLE")
    built = subprocess.run(cmd, text=True, capture_output=True, timeout=180)
    assert built.returncode == 0, built.stdout+built.stderr
    for k in range(len(q)):
        data = " ".join(map(str,np.concatenate([q[k],v[k],u[k],acc[k],fe[k]])))
        expected = np.concatenate([ref.armature_regressor(acc[k].astype(float)).ravel(),
            ref.forward_dynamics_armature_gradient(q[k].astype(float),v[k].astype(float),
                u[k].astype(float),f_ext=fe[k].reshape(-1,6).astype(float)).ravel()])
        for dims in [(1,1),(32,1),(8,4),(64,2)]:
            run = subprocess.run([str(binary),*map(str,dims)], input=data,
                                 text=True,capture_output=True,timeout=30)
            assert run.returncode == 0, run.stdout+run.stderr
            close(np.fromstring(run.stdout,sep=" "),expected,h.dtype)


def test_regressor_acceleration_derivative(case):
    import jax
    import jax.numpy as jnp
    import torch
    h,ref,a,q,v,u,acc,fe=case
    expected=np.zeros_like(acc)
    expected[:,list(h.armature_slots)]=1
    th=grid_rbd.get_robot(h.name,backend="torch")
    tensor=torch.as_tensor(acc,device="cuda").requires_grad_(True)
    th.armature_regressor(tensor).sum().backward()
    np.testing.assert_array_equal(tensor.grad.cpu().numpy(),expected)
    jh=grid_rbd.get_robot(h.name,backend="jax")
    with jax.enable_x64():
        grad=jax.jit(jax.grad(lambda x:jh.armature_regressor(x).sum()))(jnp.asarray(acc))
        np.testing.assert_array_equal(grad,expected)


def test_sensitivity_mutation_interleavings(case):
    """Barrier-coordinated rounds, not a sleep-based race or timing assertion."""
    import threading
    h, ref, a, q, v, u, acc, fe = case
    first = h.forward_dynamics_armature_gradient(q,v,u,f_ext=fe)
    h.set_armature(a*2)
    second = h.forward_dynamics_armature_gradient(q,v,u,f_ext=fe)
    assert not np.allclose(first,second)
    h.set_armature(a)
    gate = threading.Barrier(2,timeout=30)
    results, errors = [], []
    def submit():
        try:
            for _ in range(12):
                gate.wait()
                results.append(h.forward_dynamics_armature_gradient(q,v,u,f_ext=fe))
                gate.wait()
        except BaseException as error:
            errors.append(error)
            gate.abort()
    worker=threading.Thread(target=submit)
    worker.start()
    try:
        for i in range(12):
            gate.wait()
            h.set_armature(a*(1+i%2))
            gate.wait()
    finally:
        worker.join(timeout=30)
        h.set_armature(a)
    assert not worker.is_alive() and not errors
    assert len(results)==12
    for result in results:
        # Exact repeatability at either full model; a mixed Minv/qdd is neither.
        assert np.array_equal(result,first) or np.array_equal(result,second)
    old=h.model_version
    h.set_armature(a*2)
    try:
        with pytest.raises(RuntimeError,match="mutat|version|changed"):
            h._runner.graph_begin(h._runner.ctx_id(),old)
    finally:
        h.set_armature(a)


@pytest.mark.parametrize("empty", [False, True], ids=["baked", "empty-columns"])
def test_baked_and_empty_parameter_blocks(empty, tmp_path):
    import jax.numpy as jnp
    import torch
    root = Path(__file__).resolve().parents[2]
    path = root / "external/URDFParser/tests/fixtures/spherical_arm.urdf" if empty else robot_urdf("iiwa14")
    if empty:
        import xml.etree.ElementTree as ET
        tree=ET.parse(path)
        tree.find(".//joint[@name='joint_2']").set("type","fixed")
        path=tmp_path/"only_spherical.urdf"
        tree.write(path)
    with contextlib.redirect_stdout(io.StringIO()):
        ref = RBDReference(URDFParser().parse(str(path)))
    a = np.zeros(ref.robot.get_num_vel(),np.float32)
    a[supported_slots(ref.robot)] = .3
    ref.set_armature(a)
    h = grid_rbd.register_robot("identification_boundary_"+str(empty),str(path),armature=a,
        algorithm_list=["forward_dynamics_armature_gradient"],enable_mujoco_kernels=False,max_batch_size=2)
    try:
        assert h.num_armature_params == (0 if empty else 7)
        q = np.full((2,h.num_joints),.2,np.float32)
        if empty:
            q[:,:4] /= np.linalg.norm(q[:,:4],axis=1,keepdims=True)
        v = np.full((2,h.num_vel),.3,np.float32)
        want = np.stack([ref.forward_dynamics_armature_gradient(row.astype(float),v[0],v[0]) for row in q])
        for backend in ("numpy","torch","jax"):
            handle = h if backend == "numpy" else grid_rbd.get_robot(h.name,backend=backend)
            to = (lambda x:x) if backend == "numpy" else (lambda x:torch.as_tensor(x,device="cuda")) if backend == "torch" else jnp.asarray
            result = handle.forward_dynamics_armature_gradient(to(q),to(v),to(v))
            reg = handle.armature_regressor(to(v))
            if backend == "torch":
                result,reg=result.cpu().numpy(),reg.cpu().numpy()
            assert result.shape == (2,h.num_vel,h.num_armature_params)
            assert reg.shape == result.shape
            np.testing.assert_allclose(result,want,rtol=2e-4,atol=2e-4)
        with pytest.raises(RuntimeError,match="runtime_armature"):
            h.set_armature(a)
    finally:
        h.close()


def test_workspace_backed_sensitivity(monkeypatch, tmp_path):
    monkeypatch.setenv("GRID_CUDA_TARGET_SHARED_MEM_BYTES","4096")
    path=str(robot_urdf("iiwa14"))
    with contextlib.redirect_stdout(io.StringIO()):
        ref=RBDReference(URDFParser().parse(path))
    a=np.full(ref.robot.get_num_vel(),.25,np.float32)
    ref.set_armature(a)
    h=grid_rbd.register_robot("identification_forced_spill",path,armature=a,
        algorithm_list=["forward_dynamics_armature_gradient"],enable_mujoco_kernels=False,
        cache_dir=tmp_path/"cache",max_batch_size=2)
    try:
        header=(Path(h._so_path).parent/"grid.cuh").read_text()
        assert "FD_MINV_F_IN_SMEM() { return (TIER == TIER_SHARED) ? false" in header
        q=np.asarray([[.1]*h.num_joints,[.3]*h.num_joints],np.float32)
        v,u=q*.7,q*-.4
        expected=np.stack([ref.forward_dynamics_armature_gradient(*[x[k].astype(float) for x in (q,v,u)]) for k in range(2)])
        close(h.forward_dynamics_armature_gradient(q,v,u),expected,h.dtype)
    finally:
        h.close()
