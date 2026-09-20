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
    (…, i). Identical spelling for jax and torch.

    Spelled as broadcast multiply + sum, NOT ``(ct[..., None, :] @ G)``:
    under ``jax.jacobian`` the batched matmul lowers to an XLA gemm that is
    TF32-eligible at the DEFAULT matmul precision (10-bit mantissa → ~5e-4
    relative error; repro 2026-09-10: jacrev dc/dq off by 0.013 on O(64)
    entries while the eager per-one-hot vjp was bit-exact, and
    ``jax.default_matmul_precision("highest")`` restored 0.0). Multiply+sum
    lowers to an exact fp32 multiply/reduce on both frameworks — the same
    class the pre-collapse einsum spelling used — and these contractions are
    tiny (nv × len(wrt)·nv), so gemm throughput is irrelevant."""
    return (ct[..., :, None] * G).sum(-2)


def _pad_tail(g, n):
    """Append ``n`` zeros on the last axis (the nj-wide quaternion-padding slot
    of a floating base's VELOCITY-like input buffers qd/qdd/u — their tangent
    cotangent is already in transport order, only the trailing pad slot is
    missing); no-op for n == 0. n <= g's width by construction (one extra
    slot per quaternion joint). NEVER use this for ``q``: its quaternion
    blocks may occur anywhere in the chain and its cotangent needs
    :func:`_configuration_cotangent`."""
    if n <= 0:
        return g
    return _concat_last([g, g[..., :n] * 0])


def _quaternion_cotangent(g_ang, p, scalar_first=False):
    """Pull a local SO(3) cotangent through quaternion normalization.

    Pin kernels evaluate R(p/|p|); 2*vec(qhat^-1 * dp)/|p| maps ambient
    perturbations to their local angular tangent. The radial derivative is zero.
    MuJoCo's unit-quaternion contract uses the same tangential pullback.
    """
    pn = (p * p).sum(-1, keepdims=True) ** 0.5
    qn = p / pn
    if scalar_first:
        w, x, y, z = qn[..., 0:1], qn[..., 1:2], qn[..., 2:3], qn[..., 3:4]
    else:
        x, y, z, w = qn[..., 0:1], qn[..., 1:2], qn[..., 2:3], qn[..., 3:4]
    a0, a1, a2 = g_ang[..., 0:1], g_ang[..., 1:2], g_ang[..., 2:3]
    gx = w * a0 - z * a1 + y * a2
    gy = z * a0 + w * a1 - x * a2
    gz = -y * a0 + x * a1 + w * a2
    gw = -x * a0 - y * a1 - z * a2
    return _concat_last([gw, gx, gy, gz] if scalar_first else [gx, gy, gz, gw]) * (2 / pn)


def _configuration_cotangent(g, q, mjx=False, *, layout):
    """Map every independent joint's tangent block to its public q block.

    ``layout`` is validated model metadata: (kind, q_start, v_start, nq, nv).
    Spherical joints have their own xyzw quaternion at any position in the
    chain; nq>nv alone does NOT imply a free-flyer at the start. MuJoCo changes
    only the floating root to world-linear / local-angular / wxyz coordinates.
    """
    parts = []
    for kind, qi, vi, nq, nv in layout:
        block = g[..., vi:vi + nv]
        if kind == "euclidean":
            parts.append(block)
        elif kind == "spherical":
            parts.append(_quaternion_cotangent(block, q[..., qi:qi + nq]))
        elif kind == "floating":
            p = q[..., qi + 3:qi + 7]
            g_lin = block[..., :3]
            if not mjx:
                pn = (p * p).sum(-1, keepdims=True) ** 0.5
                qn = p / pn
                x, y, z, w = (qn[..., i:i + 1] for i in range(4))
                l0, l1, l2 = (g_lin[..., i:i + 1] for i in range(3))
                # R(qhat) maps the local linear tangent to world positions.
                g_lin = _concat_last([
                    (1 - 2 * (y*y + z*z))*l0 + 2*(x*y - z*w)*l1 + 2*(x*z + y*w)*l2,
                    2*(x*y + z*w)*l0 + (1 - 2*(x*x + z*z))*l1 + 2*(y*z - x*w)*l2,
                    2*(x*z - y*w)*l0 + 2*(y*z + x*w)*l1 + (1 - 2*(x*x + y*y))*l2,
                ])
            parts.extend([g_lin, _quaternion_cotangent(block[..., 3:6], p, mjx)])
        else:
            raise ValueError(f"Unknown configuration block {kind!r}")
    result = _concat_last(parts)
    if result.shape[-1] != q.shape[-1]:
        raise ValueError("Configuration cotangent layout does not match q")
    return result


def vjp_backward(vjp, ct, ops, *, nv, nj, q=None, mjx=False, configuration_layout=None):
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

    The configuration/tangent bridge: dynamics VALUE cotangents are nj-wide → take the
    leading nv tangent rows (``ct_slice_nv``); the qd/qdd/u cotangents are
    tail-padded back to nj, and the ``q`` cotangent is pulled back to the
    per-joint position layout by :func:`_configuration_cotangent`
    (needs saved ``q`` and ``configuration_layout``). Scalar joints: no-ops.
    """
    out = {}
    ctv = ct[..., :nv] if (vjp.ct_slice_nv and nj != nv) else ct
    pad = nj - nv
    G = ops["grad"]()
    g_all = _contract(ctv, G)
    w = G.shape[-1] // len(vjp.wrt)
    for i, name in enumerate(vjp.wrt):
        block = g_all[..., i * w:(i + 1) * w]
        if name == "q" and pad > 0:
            if q is None:
                raise ValueError("vjp_backward: a quaternion-joint 'q' cotangent needs the saved q "
                                 "(pass q=) and its joint layout")
            if configuration_layout is None:
                raise ValueError("vjp_backward: quaternion joints need configuration_layout metadata")
            out[name] = _configuration_cotangent(block, q, mjx=mjx, layout=configuration_layout)
        else:
            out[name] = _pad_tail(block, pad)
    if vjp.u_via_minv:
        out["u"] = _pad_tail(_contract(ctv, ops["minv"]()), pad)
    if vjp.param_grad_op:
        out["params"] = _contract(ctv, ops["param_grad"]())  # π is npar-wide: no pad
    for name in vjp.nondiff:
        out[name] = None
    return out
