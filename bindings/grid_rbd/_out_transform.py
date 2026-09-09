"""ONE implementation of the raw-buffer → python-shape output transforms
(A3 / C4 slice 4, from the 2026-09-09 layout audit).

Every method's flat per-batch-item kernel buffer is reshaped/reordered
identically on the numpy, jax, and torch surfaces — previously in FIVE
hand-copied sites (_handle.py, the jax public methods, the jax custom-VJP
bodies, the torch public methods, the torch autograd backwards). The chains
live here now, keyed by ``AbiSpec.out_layout``:

- ``"flat"``                       pass-through (vector outputs).
- ``("reshape", dims)``            row-major reshape (crba full-dense,
                                   coriolis_matrix, osc_inertia, regressor,
                                   ee_pose_hessian).
- ``("colmajor", (r, c))``         a column-major (r, c) matrix: stored as c
                                   rows of r → reshape(c, r) + swap(-1, -2)
                                   (frame_jacobian[_dot], cmm_time_variation,
                                   runtime EE gradient per-target).
- ``("colmajor_whole", (r, c))``   same, whole-output (integrator_gradient's
                                   (2nv, 3nv) dAB).
- ``("vec_then_colmajor", h, (r, c))``   com: [p(h); J col-major] → 2-tuple.
- ``("colmajor_then_vec", (r, c), t)``   ccrba: [A col-major; h(t)] → 2-tuple.
- ``("grad_concat",)``             id/fd gradient: two col-major nv×nv halves
                                   → swap each → concat to (nv, 2nv).
- ``("ee_grad",)``                 ee_pose_gradient: col-major (6, nee*nv) ≡
                                   C-order (nee, nv, 6) → per-EE swap →
                                   (6*nee, nv).
- ``("dccrba",)``                  (nv, nv, 6) buffer → (6, nv, nv) tensor.
- ``("so_slabs",)``                idsva_so/fdsva_so: 4 concatenated nv³
                                   slabs → 4-tuple of (nv, nv, nv).
- ``("minv",)``                    pin: UPPER-triangle buffer → M + Mᵀ − diag
                                   (symmetrize); mjx twin writes FULL DENSE →
                                   pass-through. The ONLY convention-dependent
                                   transform (pass ``mjx=True`` for the twin).

All ops work on the LAST axes so arrays may carry any number of leading batch
axes (2-D FFI batches, vmap leading axes, or a single sample). Framework
dispatch is duck-typed: torch tensors lack ``.swapaxes`` pre-2.x spellings we
avoid — everything here uses reshape / transpose-by-move / concat primitives
available identically on numpy, jax.numpy arrays, and torch tensors
(``reshape``, ``swapaxes``-equivalent via the adapter, slicing).

⚠Keep this module dependency-free: it must import NONE of numpy/jax/torch —
the arrays themselves carry their namespace.
"""
from __future__ import annotations


def _swap_last2(a):
    """swap the last two axes — same spelling everywhere: torch.Tensor has
    .transpose(dim0, dim1); numpy/jax have .swapaxes."""
    if hasattr(a, "swapaxes"):
        return a.swapaxes(-1, -2)
    return a.transpose(-1, -2)  # torch


def _swap(a, i, j):
    """swap two axes — numpy/jax .swapaxes, torch .transpose."""
    if hasattr(a, "swapaxes"):
        return a.swapaxes(i, j)
    return a.transpose(i, j)  # torch


def _concat_last(parts):
    mod = type(parts[0]).__module__
    if mod.startswith("torch"):
        import torch
        return torch.cat(parts, dim=-1)
    if mod.startswith("jax"):
        import jax.numpy as jnp
        return jnp.concatenate(parts, axis=-1)
    import numpy as np
    return np.concatenate(parts, axis=-1)


def _reshape_tail(a, lead_nd, dims):
    """reshape keeping the first ``lead_nd`` axes, tail becomes ``dims``."""
    return a.reshape(a.shape[:lead_nd] + tuple(dims))


def apply_out_layout(raw, layout, dims, *, nv, mjx=False, eye=None):
    """Transform the raw buffer ``raw`` (leading batch axes + ONE flat trailing
    axis, or the _core-allocated shape) per ``layout``.

    ``dims`` are the RESOLVED logical ints for the layout's dim tokens
    (use grid_codegen.abi_specs.py_dim_tokens / expand at the call site).
    ``nv`` is needed by grad_concat/so_slabs/minv/ee_grad. ``eye`` is a
    framework identity (nv, nv) for the minv symmetrize (built by the caller
    in its own framework so this module stays import-free).

    Returns the transformed array, a tuple of arrays (vec/colmajor splits,
    so_slabs), matching what the hand chains produced.
    """
    kind = layout if isinstance(layout, str) else layout[0]
    lead = raw.ndim - 1  # raw arrives flat: batch axes + one trailing axis

    if kind == "flat":
        return raw
    if kind == "reshape":
        return _reshape_tail(raw, lead, dims)
    if kind == "colmajor":
        r, c = dims
        return _swap_last2(_reshape_tail(raw, lead, (c, r)))
    if kind == "colmajor_whole":
        r, c = dims
        return _swap_last2(_reshape_tail(raw, lead, (c, r)))
    if kind == "vec_then_colmajor":
        h, (r, c) = dims
        head = raw[..., :h]
        tail = _swap_last2(_reshape_tail(raw[..., h:], lead, (c, r)))
        return head, tail
    if kind == "colmajor_then_vec":
        (r, c), t = dims
        mat = _swap_last2(_reshape_tail(raw[..., : r * c], lead, (c, r)))
        return mat, raw[..., r * c:]
    if kind == "grad_concat":
        # two col-major nv x nv halves [d?_dq | d?_dqd] -> (nv, 2nv)
        halves = _reshape_tail(raw, lead, (2, nv, nv))
        halves = _swap_last2(halves)
        return _concat_last([halves[..., 0, :, :], halves[..., 1, :, :]])
    if kind == "ee_grad":
        nee = dims[0]
        g = _swap_last2(_reshape_tail(raw, lead, (nee, nv, 6)))  # (.., nee, 6, nv)
        return _reshape_tail(g, lead, (6 * nee, nv))
    if kind == "dccrba":
        t = _reshape_tail(raw, lead, (nv, nv, 6))
        # trailing (m, k, row) -> (row, k, m): the numpy chain's
        # transpose(0, 3, 2, 1) on (B, m, k, row) is exactly swapaxes(-3, -1).
        return _swap(t, -3, -1)
    if kind == "so_slabs":
        n3 = nv ** 3
        return tuple(
            _reshape_tail(raw[..., i * n3:(i + 1) * n3], lead, (nv, nv, nv))
            for i in range(4))
    if kind == "minv":
        m = _reshape_tail(raw, lead, (nv, nv))
        if mjx:
            return m  # the twin writes full dense (congruence forces it)
        return m + _swap_last2(m) - m * eye
    raise ValueError(f"unknown out_layout {layout!r}")
