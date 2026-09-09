"""ONE implementation of the analytic VJP recipes (A4-1 vjp_ops, approved
2026-09-09). The jax custom_vjp closures and the torch autograd.Function
backwards previously duplicated the same five facts per differentiable op —
which analytic gradient backs it, the saved residuals, the per-input
cotangent recipes (Jacobian blocks / M⁻¹ / regressor), the floating-base
nv-slice/nj-pad pattern, and the non-diff set. Those facts now live in
``grid_codegen.abi_specs`` (``AbiSpec.vjp``: a :class:`VjpSpec`); this module
is the one driver both surfaces call. The surfaces keep only their
registration shells (``defvjp`` / the ``ctx`` protocol) plus callables that
return SHAPED gradient-op outputs (their ``out_layout`` applied via
``_shape_out`` / ``apply_out_layout``).

Framework-agnostic like ``_out_transform`` (imports NO numpy/jax/torch):

- The contraction is a broadcast matmul ``(ct[..., None, :] @ G)[..., 0, :]``,
  which spells identically on jax arrays and torch tensors and replaces both
  ``jnp.einsum('...o,...oi->...i')`` and ``torch.bmm`` — gated bit-identical
  against the pre-collapse gradients on real kernels.
- Contracting the whole block matrix ``[A|B|…]`` and splitting the result is
  EXACTLY contracting each block separately (each output element reduces over
  the same rows either way), so the fd/id halves, the EE single block, and
  the integrator thirds all flow through one path.
- The nj-pad is spelled as concat-with-zeros through
  :func:`_out_transform._concat_last` so no framework pad API is needed.
"""
from __future__ import annotations

from ._out_transform import _concat_last


def _contract(ct, G):
    """Row-cotangent × matrix over the last two axes: (…, o) × (…, o, i) →
    (…, i). Broadcast matmul; identical spelling for jax and torch."""
    return (ct[..., None, :] @ G)[..., 0, :]


def _pad_tail(g, n):
    """Append ``n`` zeros on the last axis (the nj-wide quaternion-padding slot
    of a floating base's input buffers); no-op for n == 0. n <= g's width by
    construction (n = nj - nv = 1 on a floating base)."""
    if n <= 0:
        return g
    return _concat_last([g, g[..., :n] * 0])


def vjp_backward(vjp, ct, ops, *, nv, nj):
    """Run one recipe: returns ``{input_name: cotangent-or-None}``.

    ``vjp`` is the :class:`grid_codegen.abi_specs.VjpSpec` row. ``ct`` is the
    value cotangent. ``ops`` maps recipe roles to zero-arg callables returning
    SHAPED arrays (built by the calling surface so this module stays
    framework- and dispatch-free):

    - ``"grad"``       → the grad_op output, last dim ``len(wrt) * nv``
                          (grad_concat halves / ee_grad / colmajor_whole dAB —
                          for ee_grad the contraction runs over its 6*NEE rows).
    - ``"minv"``       → (…, nv, nv), required when ``u_via_minv``.
    - ``"param_grad"`` → (…, nv, 10*NB), required when ``param_grad_op``.

    The floating-base bridge: dynamics VALUE cotangents are nj-wide → take the
    leading nv tangent rows (``ct_slice_nv``); input cotangents for the
    nj-wide q/qd/u buffers are padded back to nj. Fixed base: both no-ops.
    """
    out = {}
    ctv = ct[..., :nv] if (vjp.ct_slice_nv and nj != nv) else ct
    pad = nj - nv
    G = ops["grad"]()
    g_all = _contract(ctv, G)
    w = G.shape[-1] // len(vjp.wrt)
    for i, name in enumerate(vjp.wrt):
        out[name] = _pad_tail(g_all[..., i * w:(i + 1) * w], pad)
    if vjp.u_via_minv:
        out["u"] = _pad_tail(_contract(ctv, ops["minv"]()), pad)
    if vjp.param_grad_op:
        out["params"] = _contract(ctv, ops["param_grad"]())  # π is npar-wide: no pad
    for name in vjp.nondiff:
        out[name] = None
    return out
