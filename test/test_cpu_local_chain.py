"""CPU-only mathematical and dtype checks for the experimental full chain."""
from types import SimpleNamespace

import numpy as np
import pytest

from test.test_cpu_qq_study import model, rnea
from tools.cpu_qq_study import cross
from tools.cpu_local_chain import rnea_jets, mass_derivative, typed_transform, factor


def data_for(m, q, dtype):
    r, ref = m.robot, m.reference
    data = []
    for i in range(r.get_num_bodies()):
        vi = ref._as_index_list(r.get_joint_index_v(i))
        s = np.asarray(r.get_S_by_id(i), dtype=dtype).reshape(6, len(vi))
        alpha = dtype(ref._mimic_multiplier(i))
        x = np.asarray(r.get_Xmat_Func_by_id(i)(r.q_for_joint(i, q)), dtype=dtype)
        dx = np.zeros_like(x) if i == 0 else (-alpha * cross(s[:, 0])) @ x
        ddx = np.zeros_like(x) if i == 0 else (-alpha * cross(s[:, 0])) @ dx
        data.append(dict(parent=r.get_parent_id(i), vi=vi, s=s, alpha=alpha, x=x, dx=dx, ddx=ddx,
                         inertia=np.asarray(r.get_Imat_by_id(i), dtype=dtype)))
    return data


@pytest.mark.parametrize('mimic', [False, True])
def test_all_local_jet_blocks_against_independent_finite_differences(mimic):
    m = model(mimic); n = m.nv; rng = np.random.default_rng(203)
    q, v, a = [rng.normal(size=n) for _ in range(3)]
    data = data_for(m, q, np.float64)
    tau, gradient, (qq, vv, vq) = rnea_jets(data, v, a, np.zeros(n), np.float64)
    np.testing.assert_allclose(tau, rnea(m, q, v, a), atol=1e-11)
    h = 2e-4; eye = np.eye(n) * h
    for j in range(n):
        ev = eye[j]
        numeric_v = (rnea(m, q, v+ev, a)-rnea(m, q, v-ev, a))/(2*h)
        np.testing.assert_allclose(gradient[:, n+j], numeric_v, atol=2e-7, rtol=2e-7)
        vv_diag = (rnea(m, q, v+ev, a)-2*tau+rnea(m, q, v-ev, a))/h**2
        np.testing.assert_allclose(vv[:, j, j], vv_diag, atol=4e-6, rtol=4e-6)
        for k in range(6, n):
            eq = eye[k]
            numeric = (rnea(m, q+eq, v+ev, a)-rnea(m, q-eq, v+ev, a)
                       -rnea(m, q+eq, v-ev, a)+rnea(m, q-eq, v-ev, a))/(4*h*h)
            np.testing.assert_allclose(vq[:, j, k], numeric, atol=4e-6, rtol=4e-6)
    for j in range(6, n):
        numeric_q = (rnea(m, q+eye[j], v, a)-rnea(m, q-eye[j], v, a))/(2*h)
        np.testing.assert_allclose(gradient[:, j], numeric_q, atol=2e-6, rtol=2e-7)


@pytest.mark.parametrize('mimic', [False, True])
def test_mass_and_derivative_match_rnea_and_scalar_difference(mimic):
    m = model(mimic); n = m.nv; q = np.linspace(-.4, .5, n)
    armature = np.linspace(0, .1, n)
    mass, dm = mass_derivative(data_for(m, q, np.float64), armature, n, np.float64)
    bias = rnea(m, q, np.zeros(n), np.zeros(n))
    columns = np.column_stack([rnea(m, q, np.zeros(n), e)-bias for e in np.eye(n)])
    np.testing.assert_allclose(mass, columns+np.diag(armature), atol=2e-13)
    for j in range(6, n):
        step = np.eye(n)[j]*1e-5
        mp = mass_derivative(data_for(m, q+step, np.float64), armature, n, np.float64)[0]
        mm = mass_derivative(data_for(m, q-step, np.float64), armature, n, np.float64)[0]
        np.testing.assert_allclose(dm[:, :, j], (mp-mm)/2e-5, atol=1e-9, rtol=1e-8)
    np.testing.assert_array_equal(dm[:, :, :6], 0)


def test_float32_jets_and_factor_do_not_promote_or_mutate_inputs():
    m = model(True); n = m.nv; q = np.linspace(-.3, .6, n, dtype=np.float32)
    data = data_for(m, q, np.float32)
    copies = [d['x'].copy() for d in data]; original = q.copy()
    mass, dm = mass_derivative(data, np.zeros(n), n)
    solve = factor(mass)
    inv = solve(np.eye(n, dtype=np.float32))
    assert inv.dtype == np.float32
    np.testing.assert_allclose(mass @ inv, np.eye(n), atol=3e-6)
    for order in ('forward', 'reverse'):
        tau, gradient, tensors = rnea_jets(data, q, q, np.zeros(n), order=order)
        assert all(x.dtype == np.float32 for x in (mass, dm, tau, gradient, *tensors))
    for d, before in zip(data, copies): np.testing.assert_array_equal(d['x'], before)
    np.testing.assert_array_equal(q, original)


def _expression(theta):
    return array([[1.7*cos(theta) + 1, theta**2/3.1], [-sin(theta), 0]])


def _unsupported(theta):
    return np.asarray([theta])


def test_transform_typed_intermediates_and_rejection():
    fn = typed_transform(_expression, np.float32)
    result = fn(np.float32(.7))
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, typed_transform(_expression, np.float64)(np.float64(.7)), atol=1e-7)
    with pytest.raises(TypeError, match='promoted'):
        fn(np.float64(.7))
    with pytest.raises(ValueError, match='unsupported'):
        typed_transform(_unsupported, np.float32)


def test_root_gravity_uses_float32_rotation_without_numpy_solve(monkeypatch):
    from scipy.linalg import expm
    m = model(); n = m.nv
    data = data_for(m, np.zeros(n), np.float32)
    rotation = expm(cross(np.array([.3, -.4, .2, 0., 0., 0.]))[:3, :3]).astype(np.float32)
    data[0]['x'][:3, :3] = rotation
    data[0]['x'][3:, 3:] = rotation
    def forbidden(*args, **kwargs):
        raise AssertionError('NumPy solve silently promotes float32')
    monkeypatch.setattr(np.linalg, 'solve', forbidden)
    tau, gradient, tensors = rnea_jets(data, np.zeros(n), np.zeros(n), np.zeros(n))
    assert all(v.dtype == np.float32 for v in (tau, gradient, *tensors))
    # Whole-tree gravity force in the floating root equals total mass times
    # the rotated gravity; child translations change torque, not net force.
    total_mass = sum(float(d['inertia'][3, 3]) for d in data)
    np.testing.assert_allclose(tau[3:6], total_mass * 9.81 * rotation[2], rtol=2e-6)


def test_scipy_factorization_dispatches_single_precision():
    from scipy.linalg import get_lapack_funcs
    matrix = np.eye(3, dtype=np.float32)
    for name in ('potrf', 'potrs'):
        assert get_lapack_funcs(name, (matrix,)).typecode == 's'
