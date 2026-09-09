"""A4-1 offline proof: _vjp_common.vjp_backward must reproduce the HAND
backward formulas (the pre-collapse jax einsum / torch bmm chains) on random
buffers, for every recipe row. CPU/numpy; the GPU half of the gate is the
bit-comparison against the captured pre-collapse gradients on real kernels.
"""
import numpy as np

from grid_codegen.abi_specs import ABI_SPECS
from grid_rbd._vjp_common import _contract, _pad_tail, vjp_backward

B, NV, NJ, NEE, NB = 3, 6, 7, 2, 9  # floating-style nj = nv + 1
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
    g = vjp_backward(v, ct, {"grad": lambda: G, "minv": lambda: M}, nv=NV, nj=NJ)
    ctv = ct[..., :NV]
    pad = lambda a: np.pad(a, [(0, 0), (0, NJ - NV)])
    np.testing.assert_allclose(g["q"], pad(np.einsum('bo,boi->bi', ctv, G[..., :NV])), rtol=1e-14)
    np.testing.assert_allclose(g["qd"], pad(np.einsum('bo,boi->bi', ctv, G[..., NV:])), rtol=1e-14)
    np.testing.assert_allclose(g["u"], pad(np.einsum('bo,boi->bi', ctv, M)), rtol=1e-14)
    assert g["f_ext"] is None


def test_id_recipe_nondiff_slots():
    v = ABI_SPECS["inverse_dynamics"].vjp
    ct, G = _r(B, NJ), _r(B, NV, 2 * NV)
    g = vjp_backward(v, ct, {"grad": lambda: G}, nv=NV, nj=NJ)
    assert g["qdd"] is None and g["f_ext"] is None
    assert g["q"].shape == (B, NJ)


def test_ee_recipe_no_ct_slice():
    v = ABI_SPECS["end_effector_pose"].vjp
    ct, J = _r(B, 6 * NEE), _r(B, 6 * NEE, NV)
    g = vjp_backward(v, ct, {"grad": lambda: J}, nv=NV, nj=NJ)
    ref = np.pad(np.einsum('bo,boi->bi', ct, J), [(0, 0), (0, NJ - NV)])
    np.testing.assert_allclose(g["q"], ref, rtol=1e-14)


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
        g = vjp_backward(v, ct, ops, nv=NV, nj=NJ)
        ctv = ct[..., :NV]
        np.testing.assert_allclose(g["params"], np.einsum('bo,bop->bp', ctv, P), rtol=1e-12, atol=1e-12)
        assert g["params"].shape == (B, npar)  # π is npar-wide: never padded
        assert ("u" in g) == has_minv or not has_minv
