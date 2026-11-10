"""CPU regressions for per-joint quaternion pullbacks, including spherical joints."""
import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

from URDFParser import URDFParser
from grid_codegen.abi_specs import ABI_SPECS
from grid_rbd._compile import configuration_layout_from_robot
from grid_rbd._configuration import configuration_layout_from_meta
from grid_rbd._vjp_common import _configuration_cotangent, integrator_state_cotangent, vjp_backward

FIXTURES = Path(__file__).resolve().parents[1] / "external/URDFParser/tests/fixtures"


def _robot(name, floating):
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(str(FIXTURES / f"{name}.urdf"), floating_base=floating)


def _mul(a, b):
    # Independent Hamilton product, xyzw; cross/dot formulation.
    return np.r_[a[3]*b[:3] + b[3]*a[:3] + np.cross(a[:3], b[:3]),
                 a[3]*b[3] - a[:3] @ b[:3]]


def _rotation(p):
    p = p / np.linalg.norm(p)
    inv = np.r_[-p[:3], p[3]]
    return np.column_stack([_mul(_mul(p, np.r_[e, 0.]), inv)[:3] for e in np.eye(3)])


def _fd(f, x):
    eps = 2e-6
    return np.array([(f(x + eps*e) - f(x - eps*e))/(2*eps) for e in np.eye(x.size)])


@pytest.mark.parametrize("name", ["spherical_arm", "mixed_spherical_arm"])
@pytest.mark.parametrize("floating", [False, True])
@pytest.mark.parametrize("scale", [1., 1.3, -1.3])
def test_each_quaternion_pullback_matches_ambient_difference(name, floating, scale):
    robot = _robot(name, floating)
    layout = configuration_layout_from_robot(robot)
    nq, nv = robot.get_num_pos(), robot.get_num_vel()
    rng = np.random.default_rng(19)
    q = rng.normal(size=nq)
    weights = rng.normal(size=nq)
    matrices = {qi: rng.normal(size=(3, 3)) for kind, qi, _, _, _ in layout if kind != "euclidean"}
    for kind, qi, vi, np_, nv_ in layout:
        if kind != "euclidean":
            start = qi + (3 if kind == "floating" else 0)
            q[start:start+4] *= scale / np.linalg.norm(q[start:start+4])

    def value(x):
        out = 0.
        for kind, qi, vi, np_, nv_ in layout:
            if kind == "euclidean":
                out += weights[qi:qi+np_] @ x[qi:qi+np_]
            else:
                start = qi + (3 if kind == "floating" else 0)
                out += np.sum(matrices[qi] * _rotation(x[start:start+4]))
                if kind == "floating":
                    out += weights[qi:qi+3] @ x[qi:qi+3]
        return out

    def retract(delta):
        out = q.copy()
        for kind, qi, vi, np_, nv_ in layout:
            if kind == "euclidean":
                out[qi:qi+np_] += delta[vi:vi+nv_]
            else:
                start = qi + (3 if kind == "floating" else 0)
                angle = delta[vi + (3 if kind == "floating" else 0):vi+nv_]
                theta = np.linalg.norm(angle)
                dq = np.r_[angle * (np.sin(theta/2)/theta if theta else .5), np.cos(theta/2)]
                out[start:start+4] = _mul(q[start:start+4], dq)
                if kind == "floating":
                    out[qi:qi+3] += _rotation(q[start:start+4]) @ delta[vi:vi+3]
        return out

    tangent = _fd(lambda d: value(retract(d)), np.zeros(nv))
    ambient = _fd(value, q)
    result = _configuration_cotangent(tangent[None], q[None], layout=layout)[0]
    assert result.shape == q.shape
    np.testing.assert_allclose(result, ambient, rtol=2e-7, atol=2e-8)
    # Exercise the actual recipe dispatch too: nq>nv is not a floating flag.
    G = np.zeros((1, nv, 2*nv)); G[0, 0, :nv] = tangent
    ct = np.zeros((1, nv)); ct[0, 0] = 1
    got = vjp_backward(ABI_SPECS["inverse_dynamics"].vjp, ct, {"grad": lambda: G},
                       nv=nv, nq=nq, q=q[None], configuration_layout=layout)
    np.testing.assert_allclose(got["q"][0], ambient, rtol=2e-7, atol=2e-8)


def test_multiple_spherical_blocks_are_not_treated_as_scalar_tail():
    layout = (("spherical", 0, 0, 4, 3), ("spherical", 4, 3, 4, 3))
    q = np.array([[0., 0., 0., 1., 0., 0., 0., 1.]])
    g = np.array([[1., 2., 3., 4., 5., 6.]])
    np.testing.assert_array_equal(_configuration_cotangent(g, q, layout=layout),
                                  [[2., 4., 6., 0., 8., 10., 12., 0.]])


def test_old_metadata_is_only_inferred_when_unambiguous():
    assert configuration_layout_from_meta({}, 7, 7) == (("euclidean", 0, 0, 7, 7),)
    assert len(configuration_layout_from_meta({"floating_base": True}, 19, 18)) == 2
    for meta, nq, nv in (({}, 5, 4), ({"floating_base": True}, 12, 10)):
        with pytest.raises(ValueError, match="re-register"):
            configuration_layout_from_meta(meta, nq, nv)


def test_force_residuals_match_the_public_backward_contract():
    for key in ("inverse_dynamics", "forward_dynamics", "aba"):
        assert "f_ext" in ABI_SPECS[key].vjp.residuals


def _exp_quat(delta):
    # Independent SO(3) exponential as an xyzw unit quaternion.
    angle = np.linalg.norm(delta)
    if angle < 1e-12:
        return np.r_[0.5 * delta, 1.]
    return np.r_[np.sin(angle / 2) * delta / angle, np.cos(angle / 2)]


def _retract_state(x, delta, layout, nq, mjx):
    """The chart's output map: move the ambient state x = [q; qd] by a tangent
    delta = [dq (nv); dqd (nv)], block by block, the way the integrator's
    q_{k+1} is defined (local linear tangent through R on the pin root, a
    world add on the mjx root; local angular on every quaternion block)."""
    y = np.array(x, dtype=np.float64)
    for kind, qi, vi, bnq, bnv in layout:
        d = delta[vi:vi + bnv]
        if kind == "euclidean":
            y[qi:qi + bnq] += d
            continue
        if kind == "floating":
            quat = x[qi + 3:qi + 7]
            if mjx:                                # wxyz, world-linear add
                qx = np.r_[quat[1:4], quat[0]]
                y[qi:qi + 3] += d[:3]
                new = _mul(qx, _exp_quat(d[3:6]))
                y[qi + 3:qi + 7] = np.r_[new[3], new[:3]]
            else:                                  # xyzw, local linear through R
                y[qi:qi + 3] += _rotation(quat) @ d[:3]
                y[qi + 3:qi + 7] = _mul(quat, _exp_quat(d[3:6]))
        else:                                      # spherical, xyzw
            y[qi:qi + 4] = _mul(x[qi:qi + 4], _exp_quat(d))
    y[nq:] += delta[len(delta) // 2:]
    return y


@pytest.mark.parametrize("name", ["spherical_arm", "mixed_spherical_arm"])
@pytest.mark.parametrize(("floating", "mjx"), [(False, False), (True, False), (True, True)],
                         ids=["fixed", "floating", "floating-mjx"])
def test_integrator_state_cotangent_is_the_transpose_of_the_output_chart(name, floating, mjx):
    """The integrator's state cotangent (nq+nv) lands on the gradient kernel's
    tangent rows (2nv) through the transpose of the chart's output map
    (2026-11-03): <c, retract(x, delta)> differentiated at delta = 0 equals
    integrator_state_cotangent(c, x) for every direction."""
    robot = _robot(name, floating)
    layout = configuration_layout_from_robot(robot)
    nq, nv = robot.get_num_pos(), robot.get_num_vel()
    rng = np.random.default_rng(23)
    x = rng.normal(size=nq + nv)
    for kind, qi, vi, bnq, bnv in layout:
        if kind != "euclidean":
            start = qi + (3 if kind == "floating" else 0)
            x[start:start + 4] /= np.linalg.norm(x[start:start + 4])
    c = rng.normal(size=nq + nv)
    expected = _fd(lambda d: c @ _retract_state(x, d, layout, nq, mjx), np.zeros(2 * nv))
    actual = integrator_state_cotangent(c[None], x[None], nq=nq, nv=nv, mjx=mjx, layout=layout)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
    # scalar models pass through unchanged
    assert integrator_state_cotangent(c[None], x[None], nq=nq + nv, nv=nq + nv) is not None
