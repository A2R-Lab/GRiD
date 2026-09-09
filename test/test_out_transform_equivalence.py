"""A3 / C4 slice-4 offline proof: bindings/grid_rbd/_out_transform.py must
reproduce the HAND transform chains byte-for-byte on random buffers.

The hand chains below are transcribed VERBATIM from _handle.py (the oracle
surface, per the 2026-09-09 layout audit — jax/torch copy it). When the
5-site swap lands, this test is what guarantees the shared module didn't
change a single element; the end-to-end numeric round-trip on real kernels
is the GPU half of the gate.

CPU-only, no GPU/jax/torch needed (numpy proves the chain algebra; the
module is framework-agnostic by construction).
"""
import numpy as np
import pytest

from grid_rbd._out_transform import apply_out_layout

B, NV, NEE, NB = 3, 5, 2, 7
rng = np.random.default_rng(42)


def _rand(n):
    return rng.standard_normal((B, n)).astype(np.float32)


def test_reshape_row_major():
    raw = _rand(NV * NV)
    got = apply_out_layout(raw, ("reshape", None), (NV, NV), nv=NV)
    np.testing.assert_array_equal(got, raw.reshape(B, NV, NV))


def test_colmajor_matches_hand_chain():
    # hand (frame_jacobian): raw.reshape(B, NV, 6).transpose(0, 2, 1)
    raw = _rand(6 * NV)
    got = apply_out_layout(raw, ("colmajor", None), (6, NV), nv=NV)
    np.testing.assert_array_equal(got, raw.reshape(B, NV, 6).transpose(0, 2, 1))


def test_colmajor_whole_integrator_gradient():
    # hand: raw.reshape(B, 3NV, 2NV).transpose(0, 2, 1)
    raw = _rand(2 * NV * 3 * NV)
    got = apply_out_layout(raw, ("colmajor_whole", None), (2 * NV, 3 * NV), nv=NV)
    np.testing.assert_array_equal(got, raw.reshape(B, 3 * NV, 2 * NV).transpose(0, 2, 1))


def test_vec_then_colmajor_com():
    # hand: p = raw[:, :3]; J = raw[:, 3:].reshape(B, NV, 3).transpose(0, 2, 1)
    raw = _rand(3 + 3 * NV)
    p, J = apply_out_layout(raw, ("vec_then_colmajor", None, None), (3, (3, NV)), nv=NV)
    np.testing.assert_array_equal(p, raw[:, :3])
    np.testing.assert_array_equal(J, raw[:, 3:].reshape(B, NV, 3).transpose(0, 2, 1))


def test_colmajor_then_vec_ccrba():
    # hand: A = raw[:, :6NV].reshape(B, NV, 6).transpose(0, 2, 1); h = raw[:, 6NV:]
    raw = _rand(6 * NV + 6)
    A, h = apply_out_layout(raw, ("colmajor_then_vec", None, None), ((6, NV), 6), nv=NV)
    np.testing.assert_array_equal(A, raw[:, :6 * NV].reshape(B, NV, 6).transpose(0, 2, 1))
    np.testing.assert_array_equal(h, raw[:, 6 * NV:])


def test_grad_concat_id_du():
    # hand: raw.reshape(B, 2, NV, NV).transpose(0, 1, 3, 2) then
    #       np.concatenate((x[:, 0], x[:, 1]), axis=-1) -> (B, NV, 2NV)
    raw = _rand(2 * NV * NV)
    got = apply_out_layout(raw, ("grad_concat",), None, nv=NV)
    x = raw.reshape(B, 2, NV, NV).transpose(0, 1, 3, 2)
    np.testing.assert_array_equal(got, np.concatenate((x[:, 0], x[:, 1]), axis=-1))
    assert got.shape == (B, NV, 2 * NV)


def test_ee_grad():
    # hand: raw.reshape(B, NEE, NV, 6).transpose(0, 1, 3, 2).reshape(B, 6*NEE, NV)
    raw = _rand(6 * NEE * NV)
    got = apply_out_layout(raw, ("ee_grad",), (NEE,), nv=NV)
    np.testing.assert_array_equal(
        got, raw.reshape(B, NEE, NV, 6).transpose(0, 1, 3, 2).reshape(B, 6 * NEE, NV))


def test_dccrba():
    # hand: raw.reshape(B, NV, NV, 6).transpose(0, 3, 2, 1)
    raw = _rand(6 * NV * NV)
    got = apply_out_layout(raw, ("dccrba",), None, nv=NV)
    np.testing.assert_array_equal(got, raw.reshape(B, NV, NV, 6).transpose(0, 3, 2, 1))
    assert got.shape == (B, 6, NV, NV)


def test_so_slabs():
    # hand: flat[:, i*NV**3:(i+1)*NV**3].reshape(B, NV, NV, NV) for i in 0..3
    raw = _rand(4 * NV ** 3)
    got = apply_out_layout(raw, ("so_slabs",), None, nv=NV)
    assert isinstance(got, tuple) and len(got) == 4
    for i, g in enumerate(got):
        np.testing.assert_array_equal(
            g, raw[:, i * NV ** 3:(i + 1) * NV ** 3].reshape(B, NV, NV, NV))


def test_minv_pin_symmetrize_and_mjx_dense():
    # hand pin: m = raw.reshape(B, NV, NV); m + m.swapaxes(-1,-2) - m*eye
    raw = _rand(NV * NV)
    eye = np.eye(NV, dtype=np.float32)
    got = apply_out_layout(raw, ("minv",), None, nv=NV, eye=eye)
    m = raw.reshape(B, NV, NV)
    np.testing.assert_array_equal(got, m + m.swapaxes(-1, -2) - m * eye)
    got_mjx = apply_out_layout(raw, ("minv",), None, nv=NV, mjx=True)
    np.testing.assert_array_equal(got_mjx, m)


def test_flat_passthrough_and_leading_axes():
    raw = _rand(NV)
    assert apply_out_layout(raw, "flat", None, nv=NV) is raw
    # extra leading axis (vmap-style) flows through the colmajor chain too
    raw3 = rng.standard_normal((2, B, 6 * NV)).astype(np.float32)
    got = apply_out_layout(raw3, ("colmajor", None), (6, NV), nv=NV)
    np.testing.assert_array_equal(got, raw3.reshape(2, B, NV, 6).swapaxes(-1, -2))


def test_every_out_layout_row_is_a_known_class():
    from grid_codegen.abi_specs import ABI_SPECS
    KNOWN = {"flat", "reshape", "colmajor", "colmajor_whole", "vec_then_colmajor",
             "colmajor_then_vec", "grad_concat", "ee_grad", "dccrba", "so_slabs",
             "minv"}
    missing, unknown = [], []
    for key, spec in ABI_SPECS.items():
        if spec.py_out_dims and spec.out_layout is None:
            missing.append(key)
        if spec.out_layout is not None:
            kind = spec.out_layout if isinstance(spec.out_layout, str) else spec.out_layout[0]
            if kind not in KNOWN:
                unknown.append((key, kind))
    assert not missing, f"rows with py_out_dims but no out_layout: {missing}"
    assert not unknown, f"rows with unknown layout class: {unknown}"
