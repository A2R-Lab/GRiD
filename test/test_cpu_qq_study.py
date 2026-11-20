"""Independent CPU-only referees for the private numerical experiment."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from tools.cpu_qq_study import add_terms, cross, local_qq
from tools.run_cpu_qq_study import entry_attribution


def model(mimic=False):
    class Robot:
        floating_base = True
        using_quaternion = False
        def coupling_records(self): return []
        def get_num_bodies(self): return 4
        def get_parent_id(self, i): return [-1, 0, 0, 1][i]
        def get_joint_index_v(self, i): return list(range(6)) if i == 0 else ([6, 7, 6 if mimic else 8][i-1])
        def get_S_by_id(self, i):
            if i == 0: return np.eye(6)
            return np.eye(6)[:, [2, 1, 0][i-1]]
        def get_Imat_by_id(self, i): return np.diag([0.2+i*.05, 0.3+i*.05, 0.4+i*.05, 2+i, 2+i, 2+i])
        def q_for_joint(self, i, q):
            return 0 if i == 0 else (-0.7 if i == 3 and mimic else 1) * q[self.get_joint_index_v(i)]
        def get_Xmat_Func_by_id(self, i):
            def transform(q):
                if i == 0: return np.eye(6)
                offset = np.eye(6)
                offset[3:, :3] = cross(np.array([0., 0., 0., .1*i, -.04, .02]))[3:, :3]
                return expm(-cross(self.get_S_by_id(i)) * q) @ offset
            return transform
    robot = Robot()
    ref = SimpleNamespace(_as_index_list=lambda v: v if isinstance(v, list) else [v],
                          _mimic_multiplier=lambda i: -.7 if i == 3 and mimic else 1.)
    return SimpleNamespace(robot=robot, reference=ref, nv=8 if mimic else 9)


def rnea(m, q, qd, qdd):
    """Ordinary local RNEA for scalar-joint finite differences, no derivative jets."""
    r = m.robot; v = []; a = []; f = []; xs = []
    tau = np.zeros(m.nv)
    for i in range(r.get_num_bodies()):
        p = r.get_parent_id(i); vi = m.reference._as_index_list(r.get_joint_index_v(i))
        s = np.asarray(r.get_S_by_id(i)).reshape(6, len(vi)); alpha = m.reference._mimic_multiplier(i)
        x = r.get_Xmat_Func_by_id(i)(r.q_for_joint(i, q)); xs.append(x)
        vj, aj = s @ (alpha * qd[vi]), s @ (alpha * qdd[vi])
        v.append(vj if p < 0 else x @ v[p] + vj)
        a.append(np.array([0, 0, 0, 0, 0, 9.81]) + aj if p < 0 else x @ a[p] + aj + cross(v[i]) @ vj)
        inertia = r.get_Imat_by_id(i)
        f.append(inertia @ a[i] - cross(v[i]).T @ inertia @ v[i])
    for i in reversed(range(r.get_num_bodies())):
        vi = m.reference._as_index_list(r.get_joint_index_v(i))
        s = np.asarray(r.get_S_by_id(i)).reshape(6, len(vi))
        tau[vi] += m.reference._mimic_multiplier(i) * s.T @ f[i]
        p = r.get_parent_id(i)
        if p >= 0: f[p] += xs[i].T @ f[i]
    return tau


@pytest.mark.parametrize('mimic', [False, True])
@pytest.mark.parametrize('order', ['forward', 'reverse'])
def test_local_hessian_against_independent_scalar_joint_finite_difference(mimic, order):
    m = model(mimic); rng = np.random.default_rng(19)
    q, qd, qdd = [rng.normal(size=m.nv) for _ in range(3)]
    h = 2e-4
    actual = local_qq(m, q, qd, qdd, np.float64, order)
    for j in range(6, m.nv):
        ej = np.eye(m.nv)[j] * h
        diagonal = (rnea(m, q+ej, qd, qdd) - 2*rnea(m, q, qd, qdd) + rnea(m, q-ej, qd, qdd)) / h**2
        np.testing.assert_allclose(actual[:, j, j], diagonal, atol=2e-6, rtol=2e-6)
        for k in range(j+1, m.nv):
            ek = np.eye(m.nv)[k] * h
            mixed = (rnea(m, q+ej+ek, qd, qdd) - rnea(m, q+ej-ek, qd, qdd)
                     - rnea(m, q-ej+ek, qd, qdd) + rnea(m, q-ej-ek, qd, qdd)) / (4*h*h)
            np.testing.assert_allclose(actual[:, j, k], mixed, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('mimic', [False, True])
def test_float32_and_input_immutability(mimic):
    m = model(mimic); q = np.linspace(-.3, .4, m.nv); original = q.copy()
    ref = local_qq(m, q, q, q, np.float64)
    for order in ('forward', 'reverse'):
        actual = local_qq(m, q, q, q, order=order)
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, ref, atol=3e-5, rtol=3e-5)
    np.testing.assert_array_equal(q, original)


def test_cross_batch_and_dual_identity():
    a = np.arange(12, dtype=np.float32).reshape(2, 6)
    b = np.array([.1, .4, -.2, -.6, .2, 1], dtype=np.float32)
    out = cross(a)
    assert out.dtype == a.dtype
    for i in range(2):
        expected = np.r_[np.cross(a[i, :3], b[:3]), np.cross(a[i, 3:], b[:3]) + np.cross(a[i, :3], b[3:])]
        np.testing.assert_allclose(out[i] @ b, expected, atol=1e-6)


def test_refuse_unmodeled_coupling_and_invalid_order():
    m = model(); m.robot.coupling_records = lambda: [{'multiplier': None}]
    with pytest.raises(ValueError, match='nonlinear'): local_qq(m, np.zeros(9), np.zeros(9), np.zeros(9))
    with pytest.raises(ValueError, match='order'): add_terms([np.ones(1)], 'random')


def test_entry_attribution_sign_and_cancellation():
    error = np.zeros((2, 2, 2)); error[:, 0, 1] = [2, -1]
    inverse = np.array([[3., 5.], [0., 1.]])
    row = entry_attribution(error, inverse)
    assert row['worst_output'] == -1
    assert row['contribution_abs_sum'] == 11
    assert row['cancellation_ratio'] == 11


def test_ordered_lie_root_hessian_is_not_symmetrized():
    m = model(); q = np.linspace(-.2, .3, m.nv)
    result = local_qq(m, q, q, q, np.float64)
    assert np.max(np.abs(result[:, :6, :6] - result[:, :6, :6].swapaxes(1, 2))) > 1
