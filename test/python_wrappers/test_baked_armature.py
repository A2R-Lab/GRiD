"""Baked/runtime parity and initializer ownership, without timing claims."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import grid_rbd
from grid_assets import robot_urdf
from grid_codegen.armature import supported_slots
from URDFParser import URDFParser
from RBDReference import RBDReference

pytestmark = pytest.mark.python_wrappers
ALGOS = ["inverse_dynamics", "crba", "minv", "aba", "forward_dynamics",
         "forward_dynamics_gradient", "energy"]


def close(actual, expected):
    expected = np.asarray(expected, dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=2e-4,
                               atol=2e-4 * max(1., float(np.abs(expected).max())))


@pytest.fixture(scope="module", params=[("iiwa14", False), ("go2", True), ("fr3", False)],
                ids=["fixed", "floating", "mimic"])
def models(request):
    robot, floating = request.param
    with contextlib.redirect_stdout(io.StringIO()):
        ref = RBDReference(URDFParser().parse(str(robot_urdf(robot)), floating_base=floating))
    a = np.zeros(ref.robot.get_num_vel(), dtype=np.float32)
    a[supported_slots(ref.robot)] = np.linspace(.15, .45, len(supported_slots(ref.robot)))
    common = dict(urdf_path=str(robot_urdf(robot)), floating_base=floating, armature=a,
                  algorithm_list=ALGOS, enable_mujoco_kernels=False, max_batch_size=3)
    # Rebind the name to a different artifact; the old handle must keep its owner.
    baked = grid_rbd.register_robot(f"baked_armature_{robot}", **common)
    runtime = grid_rbd.register_robot(baked.name, runtime_armature=True, **common)
    ref.set_armature(a)
    rng = np.random.default_rng(23)
    q = rng.uniform(-.5, .5, (3, baked.num_joints)).astype(np.float32)
    if floating:
        q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    v, acc, u = [rng.uniform(-.7, .7, (3, baked.num_vel)).astype(np.float32) for _ in range(3)]
    yield baked, runtime, ref, a, q, v, acc, u
    baked.close()
    runtime.close()


@pytest.mark.parametrize("threads", [1, 32, 128])
def test_values_derivatives_and_energy(models, threads):
    baked, runtime, ref, a, q, v, acc, u = models
    outputs = []
    for h in (baked, runtime):
        h.set_threads_per_block(threads)
        outputs.append(dict(id=h.inverse_dynamics(q, v, acc), crba=h.crba(q),
                            minv=h.minv(q), aba=h.aba(q, v, u),
                            fd=h.forward_dynamics(q, v, u),
                            fdg=h.forward_dynamics_gradient(q, v, u), energy=h.energy(q, v)))
    for key in outputs[0]:
        close(outputs[0][key], outputs[1][key])
    for row in range(3):
        qr, vr, ar, ur = [x[row].astype(float) for x in (q, v, acc, u)]
        tau, *_ = ref.inverse_dynamics(qr, vr, ar)
        close(outputs[0]["id"][row], tau)
        close(outputs[0]["crba"][row], ref.crba(qr))
        close(outputs[0]["minv"][row], ref.minv(qr))
        fd = ref.forward_dynamics(qr, vr, ur)
        close(outputs[0]["fd"][row], fd)
        close(outputs[0]["aba"][row], fd)
        dq, dv = ref.forward_dynamics_gradient(qr, vr, ur)
        close(outputs[0]["fdg"][row], np.concatenate((dq, dv), axis=1))
        ke, pe = ref.kinetic_energy(qr, vr), ref.potential_energy(qr)
        close(outputs[0]["energy"][row], [ke, pe, ke + pe])


def test_initializers_contexts_and_name_rebinding(models, monkeypatch):
    baked, runtime, ref, a, q, v, acc, u = models
    assert baked._so_path != runtime._so_path
    assert baked.armature_slots == runtime.armature_slots == tuple(supported_slots(ref.robot))
    for h in (baked, runtime):
        np.testing.assert_array_equal(h.armature, a)
        initial = h.initial_armature
        initial[:] = 0
        np.testing.assert_array_equal(h.initial_armature, a)
    with pytest.raises(RuntimeError, match="runtime_armature"):
        baked.set_armature(a)
    runtime.set_armature(a * 2)
    def forbidden(*args, **kwargs):
        raise AssertionError("identical construction attempted recompilation")
    monkeypatch.setattr(grid_rbd, "compile_sources", forbidden)
    alias = grid_rbd.register_robot(
        runtime.name, str(robot_urdf(runtime.name.removeprefix("baked_armature_"))),
        floating_base=runtime.floating_base, armature=a, runtime_armature=True,
        algorithm_list=ALGOS, enable_mujoco_kernels=False, max_batch_size=3)
    np.testing.assert_array_equal(alias.armature, a * 2)
    alias.close()
    child = runtime.context()
    try:
        np.testing.assert_array_equal(baked.armature, a)
        np.testing.assert_array_equal(child.armature, a)
        np.testing.assert_array_equal(runtime.initial_armature, a)
        child.set_armature(a * 3)
        np.testing.assert_array_equal(runtime.armature, a * 2)
    finally:
        child.close()
        runtime.set_armature(a)


def test_torch_state_autograd_and_mutation_refusal(models):
    import torch
    from grid_rbd.torch import TorchRobotHandle
    baked, runtime, ref, a, q, v, acc, u = models
    for base in (baked, runtime):
        h = TorchRobotHandle(base, Path(base._so_path).parent.name, base._so_path)
        tq = torch.as_tensor(q, device="cuda")
        tv = torch.as_tensor(v, device="cuda").requires_grad_(True)
        tu = torch.as_tensor(u, device="cuda").requires_grad_(True)
        out = h.forward_dynamics(tq, tv, tu)
        close(out.detach().cpu().numpy(), base.forward_dynamics(q, v, u))
        out.sum().backward()
        gradient = base.forward_dynamics_gradient(q, v, u)
        close(tv.grad.cpu().numpy(), gradient[:, :, base.num_vel:2 * base.num_vel].sum(axis=1))
        close(tu.grad.cpu().numpy(), np.linalg.inv(base.crba(q)).sum(axis=1))
        if base.runtime_armature:
            tv2 = torch.as_tensor(v, device="cuda").requires_grad_(True)
            pending = h.forward_dynamics(tq, tv2, tu.detach())
            base.set_armature(a * 2)
            try:
                with pytest.raises(RuntimeError, match="model mutated between forward and backward"):
                    pending.sum().backward()
            finally:
                base.set_armature(a)


def test_jax_state_autograd(models):
    import jax
    import jax.numpy as jnp
    from grid_rbd.jax import JaxRobotHandle
    baked, runtime, ref, a, q, v, acc, u = models
    for base in (baked, runtime):
        h = JaxRobotHandle(base, Path(base._so_path).parent.name, base._so_path)
        tq, tv, tu = map(jnp.asarray, (q, v, u))
        close(np.asarray(h.forward_dynamics(tq, tv, tu)), base.forward_dynamics(q, v, u))
        dv, du = jax.grad(lambda vel, force: jnp.sum(h.forward_dynamics(tq, vel, force)),
                          argnums=(0, 1))(tv, tu)
        gradient = base.forward_dynamics_gradient(q, v, u)
        close(np.asarray(dv), gradient[:, :, base.num_vel:2 * base.num_vel].sum(axis=1))
        close(np.asarray(du), np.linalg.inv(base.crba(q)).sum(axis=1))


def test_backend_views_use_artifact_identity_not_rebound_name(models):
    import jax.numpy as jnp
    import torch
    from grid_rbd.jax import JaxRobotHandle
    from grid_rbd.torch import TorchRobotHandle
    baked, runtime, ref, a, q, v, acc, u = models
    runtime.set_armature(a * 2)
    try:
        assert not np.allclose(baked.minv(q), runtime.minv(q), rtol=1e-6, atol=1e-6)
        for base in (baked, runtime):
            key = Path(base._so_path).parent.name
            jh = JaxRobotHandle(base, key, base._so_path)
            th = TorchRobotHandle(base, key, base._so_path)
            expected = np.linalg.inv(base.crba(q).astype(float))
            close(np.asarray(jh.minv(jnp.asarray(q))), expected)
            close(th.minv(torch.as_tensor(q, device="cuda")).cpu().numpy(), expected)
    finally:
        runtime.set_armature(a)


def test_prepared_fresh_process_restores_initializer_without_compilation(models):
    baked, runtime, ref, a, q, v, acc, u = models
    runtime.set_armature(a * 2)
    try:
        subprocess.run([sys.executable, "-c", '''
import json, sys
import numpy as np
import grid_rbd
import grid_rbd._compile as compiler
def forbidden(*a, **kw):
    raise AssertionError("prepared loading attempted compilation")
compiler.find_nvcc = forbidden
compiler.compile_so = forbidden
compiler.compile_sources = forbidden
grid_rbd.compile_sources = forbidden
grid_rbd.warm_robot = forbidden
h = grid_rbd.get_robot(sys.argv[1])
np.testing.assert_array_equal(h.armature, json.loads(sys.argv[2]))
np.testing.assert_array_equal(h.initial_armature, h.armature)
q = np.asarray(json.loads(sys.argv[3]), dtype=np.float32)
np.testing.assert_allclose(h.minv(q), np.asarray(json.loads(sys.argv[4])), rtol=2e-4, atol=2e-4)
h.close()
''', runtime.name, json.dumps(a.tolist()), json.dumps(q.tolist()),
                        json.dumps(baked.minv(q).tolist())], check=True,
                       capture_output=True, text=True)
    finally:
        runtime.set_armature(a)
