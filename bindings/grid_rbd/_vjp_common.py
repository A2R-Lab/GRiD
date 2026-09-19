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
    missing); no-op for n == 0. n <= g's width by construction (n = nj - nv = 1
    on a floating base). NEVER use this for ``q``: the position buffer is
    ``[pos(3), quat_xyzw(4), joints]`` and its cotangent needs
    :func:`_configuration_cotangent`."""
    if n <= 0:
        return g
    return _concat_last([g, g[..., :n] * 0])


def _configuration_cotangent(g, q, mjx=False):
    """Pull a floating-base TANGENT cotangent back to the public ``q`` layout.

    ``g`` is ``(…, nv)`` = ``ct · ∂out/∂δ`` where δ is the Pinocchio free-flyer
    tangent ``[v_lin(3) LOCAL, ω(3) LOCAL, joints]`` (the chart every GRiD
    analytic gradient kernel differentiates in — verified 2026-09-19 against
    central differences of the kernels under the four lin/ang local/world
    retractions: ID and EE-pose match ONLY lin=local, ang=local). ``q`` is the
    saved ``(…, nj)`` position ``[pos(3), quat_xyzw(4), joints]`` (nj = nv+1).

    The forward kernels evaluate the rotation as R(p/|p|) (their quaternion
    formulas divide by |p|²), i.e. the public function is ``f ∘ normalize`` on
    the ambient quaternion, so the returned cotangent is the exact pullback of
    THAT extension (finite differences over any ambient q component agree):

    - ``pos``:   δpos_world = R(q) δ_lin      →  g_pos  = R(q) · g_lin
    - ``quat``:  δθ_local  = 2·vec(q̂⁻¹ ⊗ dp)/|p| (radial dp is a null direction)
                 →  g_quat = 2·[w g0 − z g1 + y g2,  z g0 + w g1 − x g2,
                                −y g0 + x g1 + w g2,  −x g0 − y g1 − z g2] / |p|
                 with (x,y,z,w) = q̂ = p/|p| and (g0,g1,g2) = g_ang
    - ``joints``: identity, shifted by the one extra quaternion slot.

    ``mjx=True`` (the MuJoCo output convention, floating base): the twins
    differentiate in the mjx free-joint chart — linear WORLD, angular LOCAL
    (G = blockdiag(R, I) w.r.t. the pin chart; probed 2026-09-19) — and ``q``
    is ``[pos(3), quat_wxyz(4), joints]``: so ``g_pos = g_lin`` and the same
    quaternion pullback is read/written in wxyz order.

    Before 2026-09-19 (audit W01) ``q`` went through :func:`_pad_tail`, which
    shifted every joint cotangent one slot early, dropped the last joint, and
    left raw ω components in the quaternion slots. Framework-agnostic:
    slicing, products and :func:`_concat_last` only.
    """
    g_lin, g_ang, g_j = g[..., 0:3], g[..., 3:6], g[..., 6:]
    p = q[..., 3:7]
    pn = (p * p).sum(-1, keepdims=True) ** 0.5
    qn = p / pn
    if mjx:
        w, x, y, z = qn[..., 0:1], qn[..., 1:2], qn[..., 2:3], qn[..., 3:4]
    else:
        x, y, z, w = qn[..., 0:1], qn[..., 1:2], qn[..., 2:3], qn[..., 3:4]
    l0, l1, l2 = g_lin[..., 0:1], g_lin[..., 1:2], g_lin[..., 2:3]
    a0, a1, a2 = g_ang[..., 0:1], g_ang[..., 1:2], g_ang[..., 2:3]
    if mjx:
        g_pos = g_lin                                   # world-frame linear tangent
    else:
        # R(q̂) · g_lin, R in the xyzw convention the kernels use
        g_pos = _concat_last([
            (1 - 2 * (y * y + z * z)) * l0 + 2 * (x * y - z * w) * l1 + 2 * (x * z + y * w) * l2,
            2 * (x * y + z * w) * l0 + (1 - 2 * (x * x + z * z)) * l1 + 2 * (y * z - x * w) * l2,
            2 * (x * z - y * w) * l0 + 2 * (y * z + x * w) * l1 + (1 - 2 * (x * x + y * y)) * l2,
        ])
    gx = w * a0 - z * a1 + y * a2
    gy = z * a0 + w * a1 - x * a2
    gz = -y * a0 + x * a1 + w * a2
    gw = -x * a0 - y * a1 - z * a2
    g_quat = _concat_last([gw, gx, gy, gz] if mjx else [gx, gy, gz, gw]) * (2 / pn)
    return _concat_last([g_pos, g_quat, g_j])


def vjp_backward(vjp, ct, ops, *, nv, nj, q=None, mjx=False):
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
    leading nv tangent rows (``ct_slice_nv``); the qd/qdd/u cotangents are
    tail-padded back to nj, and the ``q`` cotangent is pulled back to the
    ``[pos, quat_xyzw, joints]`` layout by :func:`_configuration_cotangent`
    (needs the saved ``q``). Fixed base: all no-ops.
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
                raise ValueError("vjp_backward: a floating-base 'q' cotangent needs the saved q "
                                 "(pass q=) — the position layout is [pos, quat_xyzw, joints]")
            out[name] = _configuration_cotangent(block, q, mjx=mjx)
        else:
            out[name] = _pad_tail(block, pad)
    if vjp.u_via_minv:
        out["u"] = _pad_tail(_contract(ctv, ops["minv"]()), pad)
    if vjp.param_grad_op:
        out["params"] = _contract(ctv, ops["param_grad"]())  # π is npar-wide: no pad
    for name in vjp.nondiff:
        out[name] = None
    return out
