"""Runtime rotor armature (``register_robot(..., runtime_armature=True)``), 2026-11-03.

The device table ``d_armature_params`` is zero after registration, so an
untouched runtime build must equal the plain build bit for bit on every
algorithm that reads the joint-space inertia. After ``set_armature`` the
kernels follow Pinocchio's ``model.armature``: inverse dynamics gains
``a*qdd``, CRBA and Minv see ``M + diag(a)``, ABA solves with it, and forward
dynamics, its gradient and the second-order forward dynamics follow. The
oracle is the float64 RBDReference with the same armature installed (it is
checked against Pinocchio in RBDReference/tests/test_armature_equivalence.py).
iiwa14 fixed and go2 floating (the root's six slots carry no armature).
"""
from __future__ import annotations

import contextlib
import io
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from grid_assets import robot_urdf  # noqa: E402

grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")
if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH", allow_module_level=True)

pytestmark = pytest.mark.python_wrappers

_ALGOS = ["inverse_dynamics", "crba", "minv", "forward_dynamics", "aba", "forward_dynamics_gradient",
          "inverse_dynamics_gradient", "fdsva_so"]
# float32 kernels against the float64 oracle: the CUDA harness float class.
_RTOL, _ATOL = 2e-4, 2e-4


def _reference(name, floating):
    from URDFParser import URDFParser
    from RBDReference import RBDReference
    with contextlib.redirect_stdout(io.StringIO()):
        return RBDReference(URDFParser().parse(str(robot_urdf(name)), floating_base=floating))


def _sample(ref, floating, seed=3):
    rng = np.random.default_rng(seed)
    nq, nv = ref.robot.get_num_pos(), ref.robot.get_num_vel()
    q = rng.uniform(-0.8, 0.8, nq)
    if floating:
        quat = rng.standard_normal(4); q[3:7] = quat / np.linalg.norm(quat)
    qd = rng.uniform(-1.0, 1.0, nv); qdd = rng.uniform(-1.0, 1.0, nv); u = rng.uniform(-2.0, 2.0, nv)
    return q, qd, qdd, u


def _armature(ref, floating, seed=5):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.05, 0.4, ref.robot.get_num_vel())
    if floating:
        a[:6] = 0.0
    return a


def _close(actual, expected, msg):
    expected = np.asarray(expected, dtype=np.float64)
    scale = max(1.0, float(np.abs(expected).max()))
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float64), expected,
                               rtol=_RTOL, atol=_ATOL * scale, err_msg=msg)


@pytest.fixture(scope="module", params=[("iiwa14", False), ("go2", True)], ids=["iiwa14-fixed", "go2-floating"])
def builds(request):
    name, floating = request.param
    common = dict(urdf_path=str(robot_urdf(name)), floating_base=floating, max_batch_size=4,
                  algorithm_list=_ALGOS, enable_mujoco_kernels=False)
    h_rt = grid_rbd.register_robot(name=f"{name}_rtarm_pytest", runtime_armature=True, **common)
    h_plain = grid_rbd.register_robot(name=f"{name}_plainarm_pytest", **common)
    yield name, floating, h_rt, h_plain, _reference(name, floating)
    h_rt.close(); h_plain.close()


def _outputs(h, q, qd, qdd, u):
    B = lambda x: np.asarray(x, dtype=np.float32)[None]
    out = {
        "inverse_dynamics": h.inverse_dynamics(B(q), B(qd), B(qdd)),
        "crba": h.crba(B(q)),
        "minv": h.minv(B(q)),
        "forward_dynamics": h.forward_dynamics(B(q), B(qd), B(u)),
        "aba": h.aba(B(q), B(qd), B(u)),
        "forward_dynamics_gradient": h.forward_dynamics_gradient(B(q), B(qd), B(u)),
        "inverse_dynamics_gradient": h.inverse_dynamics_gradient(B(q), B(qd), B(qdd)),
    }
    for k, t in enumerate(h.fdsva_so(B(q), B(qd), B(u))):
        out[f"fdsva_so[{k}]"] = t
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def test_metadata_and_zero_table_equals_plain_bitwise(builds):
    name, floating, h_rt, h_plain, ref = builds
    assert h_rt.runtime_armature is True and h_plain.runtime_armature is False
    assert np.array_equal(h_rt.armature, np.zeros(h_rt.num_vel))
    with pytest.raises(RuntimeError):
        h_plain.set_armature(np.zeros(h_plain.num_vel))
    q, qd, qdd, u = _sample(ref, floating)
    rt, plain = _outputs(h_rt, q, qd, qdd, u), _outputs(h_plain, q, qd, qdd, u)
    for k in plain:
        assert np.array_equal(rt[k], plain[k]), f"{name}: {k} untouched runtime != plain bit-for-bit"


def test_armature_matches_reference_and_restores(builds):
    name, floating, h_rt, h_plain, ref = builds
    q, qd, qdd, u = _sample(ref, floating)
    a = _armature(ref, floating)
    h_rt.set_armature(a)
    assert np.allclose(h_rt.armature, a.astype(np.float32))
    ref.set_armature(a)
    try:
        got = _outputs(h_rt, q, qd, qdd, u)
        tau, *_ = ref.inverse_dynamics(q, qd, qdd, GRAVITY=-9.81)
        _close(got["inverse_dynamics"][0], tau, f"{name}: inverse_dynamics with armature")
        _close(got["crba"][0], ref.crba(q), f"{name}: crba with armature")
        _close(got["minv"][0], ref.minv(q), f"{name}: minv with armature")
        qdd_ref = ref.forward_dynamics(q, qd, u)
        _close(got["forward_dynamics"][0], qdd_ref, f"{name}: forward_dynamics with armature")
        _close(got["aba"][0], qdd_ref, f"{name}: aba with armature")
        dq, dqd = ref.forward_dynamics_gradient(q, qd, u)
        fdg = got["forward_dynamics_gradient"][0]
        nv = h_rt.num_vel
        _close(fdg.reshape(nv, -1)[:, :nv], np.asarray(dq), f"{name}: dqdd/dq with armature")
        _close(fdg.reshape(nv, -1)[:, nv:2 * nv], np.asarray(dqd), f"{name}: dqdd/dqd with armature")
        # a plain build does not carry the term: the armature changes the answer
        plain = _outputs(h_plain, q, qd, qdd, u)
        assert not np.allclose(got["minv"], plain["minv"], rtol=1e-6, atol=1e-6)
        # second order through M + diag(a): the four tensors against the oracle
        so = ref.fdsva_so(q, qd, u)
        for k, t in enumerate(so):
            _close(got[f"fdsva_so[{k}]"][0], np.asarray(t), f"{name}: fdsva_so[{k}] with armature")
    finally:
        h_rt.set_armature(np.zeros(h_rt.num_vel))
        ref.set_armature(np.zeros(ref.robot.get_num_vel()))
    # zero again: back to the plain build bit for bit
    rt, plain = _outputs(h_rt, q, qd, qdd, u), _outputs(h_plain, q, qd, qdd, u)
    for k in plain:
        assert np.array_equal(rt[k], plain[k]), f"{name}: {k} restored runtime != plain bit-for-bit"


def test_root_slots_are_refused(builds):
    name, floating, h_rt, _, _ = builds
    nv = h_rt.num_vel
    with pytest.raises(ValueError):
        h_rt.set_armature(np.zeros(nv + 1))
    if floating:
        bad = np.zeros(nv); bad[2] = 0.1
        with pytest.raises(ValueError):
            h_rt.set_armature(bad)


def test_poke_seen_across_surfaces(builds):
    name, floating, h_rt, _, ref = builds
    q, qd, qdd, u = _sample(ref, floating, seed=9)
    a = _armature(ref, floating, seed=12)
    try:
        h_rt.set_armature(a)
        want = np.asarray(h_rt.minv(np.asarray(q, np.float32)[None]), np.float64)
        for backend in ("jax", "torch"):
            fw = pytest.importorskip(backend)
            h = grid_rbd.get_robot(h_rt.name, backend=backend)
            try:
                assert h.runtime_armature is True
                np.testing.assert_allclose(np.asarray(h.armature, np.float64), a, rtol=1e-6, atol=1e-6)
                if backend == "jax":
                    got = np.asarray(h.minv(fw.numpy.asarray(q, dtype=fw.numpy.float32)[None]), np.float64)
                else:
                    got = h.minv(fw.as_tensor(q, dtype=fw.float32, device="cuda")[None]).cpu().numpy().astype(np.float64)
                assert np.array_equal(got, want), f"{name}: {backend} minv does not see the armature poke"
            finally:
                h.close()
    finally:
        h_rt.set_armature(np.zeros(h_rt.num_vel))
