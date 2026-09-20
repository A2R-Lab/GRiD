"""A4-1 offline proof: _vjp_common.vjp_backward must reproduce the HAND
backward formulas (the pre-collapse jax einsum / torch bmm chains) on random
buffers, for every recipe row. CPU/numpy; the GPU half of the gate is the
bit-comparison against the captured pre-collapse gradients on real kernels.
"""
import numpy as np
import pytest
from functools import partial

from grid_codegen.abi_specs import ABI_SPECS
from grid_rbd._vjp_common import _contract, _pad_tail, _configuration_cotangent, vjp_backward as _vjp_backward

B, NV, NJ, NEE, NB = 3, 8, 9, 2, 9  # floating-style nj = nv + 1 (6 root + 2 joints)
LAYOUT = (("floating", 0, 0, 7, 6), ("euclidean", 7, 6, 2, 2))
vjp_backward = partial(_vjp_backward, configuration_layout=LAYOUT)


# ── independent numpy oracle for the floating-base q pullback (audit W01) ──
# Built from quaternion products (J = 2·vec(q̂⁻¹ ⊗ e_i)) and the explicit
# rotation matrix, NOT from the driver's closed-form expressions.
def _qmul_xyzw(a, b):
    x1, y1, z1, w1 = a; x2, y2, z2, w2 = b
    return np.array([w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2, w1*w2 - x1*x2 - y1*y2 - z1*z2])


def _R_xyzw(q):
    x, y, z, w = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def _pullback_np(g, q, mjx=False):
    out = np.zeros(g.shape[:-1] + (g.shape[-1] + 1,))
    for b in range(g.shape[0]):
        p = q[b, 3:7]; n = np.linalg.norm(p); qn = p / n
        qx = qn[[1, 2, 3, 0]] if mjx else qn                       # to xyzw
        qinv = np.array([-qx[0], -qx[1], -qx[2], qx[3]])
        J = np.stack([2 * _qmul_xyzw(qinv, np.eye(4)[i])[:3] for i in range(4)], axis=1)  # (3, 4)
        g_pos = g[b, :3] if mjx else _R_xyzw(qx) @ g[b, :3]
        g_quat = J.T @ g[b, 3:6] / n
        if mjx:
            g_quat = g_quat[[3, 0, 1, 2]]
        out[b] = np.concatenate([g_pos, g_quat, g[b, 6:]])
    return out


def _q(unit=True, mjx=False):
    q = _r(B, NJ)
    for b in range(B):
        quat = np.random.default_rng(100 + b).standard_normal(4)
        q[b, 3:7] = quat / np.linalg.norm(quat) * (1.0 if unit else 1.3)
    return q
rng = np.random.default_rng(5)


def _r(*shape):
    return rng.standard_normal(shape).astype(np.float64)


def test_contract_equals_einsum():
    ct, G = _r(B, NV), _r(B, NV, 2 * NV)
    ref = np.einsum('...o,...oi->...i', ct, G)
    np.testing.assert_allclose(_contract(ct, G), ref, rtol=1e-14, atol=0)


def test_pad_tail_equals_np_pad():
    g = _r(B, NV)
    np.testing.assert_array_equal(
        _pad_tail(g, NJ - NV), np.pad(g, [(0, 0), (0, NJ - NV)]))
    assert _pad_tail(g, 0) is g


def test_fd_recipe_matches_hand_formulas():
    v = ABI_SPECS["forward_dynamics"].vjp
    ct, G, M = _r(B, NJ), _r(B, NV, 2 * NV), _r(B, NV, NV)
    q = _q(unit=False)                                       # non-unit: exercises the 1/|p|
    g = vjp_backward(v, ct, {"grad": lambda: G, "minv": lambda: M}, nv=NV, nj=NJ, q=q)
    ctv = ct[..., :NV]
    pad = lambda a: np.pad(a, [(0, 0), (0, NJ - NV)])
    np.testing.assert_allclose(g["q"], _pullback_np(np.einsum('bo,boi->bi', ctv, G[..., :NV]), q), rtol=1e-13)
    np.testing.assert_allclose(g["qd"], pad(np.einsum('bo,boi->bi', ctv, G[..., NV:])), rtol=1e-14)
    np.testing.assert_allclose(g["u"], pad(np.einsum('bo,boi->bi', ctv, M)), rtol=1e-14)
    assert g["f_ext"] is None


def test_id_recipe_nondiff_slots():
    v = ABI_SPECS["inverse_dynamics"].vjp
    ct, G = _r(B, NJ), _r(B, NV, 2 * NV)
    g = vjp_backward(v, ct, {"grad": lambda: G}, nv=NV, nj=NJ, q=_q())
    assert g["qdd"] is None and g["f_ext"] is None
    assert g["q"].shape == (B, NJ)


def test_floating_q_needs_the_saved_position():
    v = ABI_SPECS["inverse_dynamics"].vjp
    ct, G = _r(B, NJ), _r(B, NV, 2 * NV)
    with pytest.raises(ValueError, match="saved q"):
        vjp_backward(v, ct, {"grad": lambda: G}, nv=NV, nj=NJ)
    # fixed base (nj == nv): no q needed, plain block split
    g = vjp_backward(v, ct[..., :NV], {"grad": lambda: G}, nv=NV, nj=NV)
    np.testing.assert_allclose(g["q"], np.einsum('bo,boi->bi', ct[..., :NV], G[..., :NV]), rtol=1e-14)


def test_configuration_cotangent_mjx_chart():
    g, q = _r(B, NV), _q(mjx=True)                                # q = [pos, quat_wxyz, joints]
    np.testing.assert_allclose(_configuration_cotangent(g, q, mjx=True, layout=LAYOUT), _pullback_np(g, q, mjx=True), rtol=1e-13)
    np.testing.assert_allclose(_configuration_cotangent(g, q, layout=LAYOUT), _pullback_np(g, q), rtol=1e-13)


def test_ee_recipe_no_ct_slice():
    v = ABI_SPECS["end_effector_pose"].vjp
    ct, J = _r(B, 6 * NEE), _r(B, 6 * NEE, NV)
    q = _q()
    g = vjp_backward(v, ct, {"grad": lambda: J}, nv=NV, nj=NJ, q=q)
    np.testing.assert_allclose(g["q"], _pullback_np(np.einsum('bo,boi->bi', ct, J), q), rtol=1e-13)


def test_integrator_recipe_thirds_full_ct():
    v = ABI_SPECS["integrator"].vjp
    # fixed base: nj == nv; ct is the FULL 2NV state cotangent (no slice)
    ct, dAB = _r(B, 2 * NV), _r(B, 2 * NV, 3 * NV)
    g = vjp_backward(v, ct, {"grad": lambda: dAB}, nv=NV, nj=NV)
    whole = np.einsum('bo,boi->bi', ct, dAB)
    np.testing.assert_allclose(g["q"], whole[:, :NV], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(g["qd"], whole[:, NV:2 * NV], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(g["u"], whole[:, 2 * NV:], rtol=1e-12, atol=1e-12)


def test_wrt_params_recipes():
    npar = 10 * NB
    for key, has_minv in (("inverse_dynamics_wrt_params", False),
                          ("forward_dynamics_wrt_params", True)):
        v = ABI_SPECS[key].vjp
        ct, G, P = _r(B, NJ), _r(B, NV, 2 * NV), _r(B, NV, npar)
        ops = {"grad": lambda: G, "param_grad": lambda: P}
        if has_minv:
            ops["minv"] = lambda: _r(B, NV, NV)
        g = vjp_backward(v, ct, ops, nv=NV, nj=NJ, q=_q())
        ctv = ct[..., :NV]
        np.testing.assert_allclose(g["params"], np.einsum('bo,bop->bp', ctv, P), rtol=1e-12, atol=1e-12)
        assert g["params"].shape == (B, npar)  # π is npar-wide: never padded
        assert ("u" in g) == has_minv or not has_minv
