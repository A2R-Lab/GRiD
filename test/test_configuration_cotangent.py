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
from grid_rbd._vjp_common import _configuration_cotangent, vjp_backward

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
    ct = np.zeros((1, nq)); ct[0, 0] = 1
    got = vjp_backward(ABI_SPECS["inverse_dynamics"].vjp, ct, {"grad": lambda: G},
                       nv=nv, nj=nq, q=q[None], configuration_layout=layout)
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
