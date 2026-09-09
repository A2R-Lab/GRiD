"""grid_rbd.torch — PyTorch backend (D.3).

Mirrors the standard grid_rbd API but returns a :py:class:`TorchRobotHandle`
whose methods are autograd-aware torch ops returning ``torch.Tensor`` and
running on the current torch CUDA stream. The differentiable algorithms
(inverse_dynamics / forward_dynamics / aba / integrator, plus the inertial-param
sysID ops inverse_dynamics_wrt_params / forward_dynamics_wrt_params) carry
analytic backward passes that reuse the existing ``*_gradient`` / regressor
kernels; the rest are forward-only ops.

The forward-only centroidal / energy / kinematics surface mirrors the numpy
handle: ``generalized_gravity``, ``nonlinear_effects``, ``energy``, ``com``,
``ccrba``, ``dccrba``, ``cmm_time_variation``, ``coriolis_matrix``,
``kinetic_energy_regressor``, ``potential_energy_regressor``, ``frame_jacobian``,
``frame_jacobian_dot``, ``osc_inertia``, ``end_effector_pose_runtime``,
``end_effector_pose_gradient_runtime`` (the gated ones raise an actionable
"not generated for this robot" error when their kernel is absent from the .so).

The grid_plant cost / barrier / plant-step surface is also exposed
(``plant_step``, ``plant_step_gradient``, ``quadratic_state_cost``,
``quadratic_input_cost``, ``ee_pos_cost``, ``joint_position_barrier``,
``joint_velocity_barrier``, ``joint_torque_barrier``, ``com_cost``,
``momentum_cost``) as forward-only ops matching the numpy handle's shapes; the
ops needing a gated kernel (plant_step[_gradient], ee/com/momentum cost) are
registered only when the per-robot ``.so`` was built with that kernel.

Usage:

    import grid_rbd
    h = grid_rbd.register_robot("iiwa14", urdf_path="iiwa.urdf", backend="torch")
    qdd = h.forward_dynamics(q, qd, u)   # torch.Tensor, autograd-aware
    qdd.sum().backward()                 # gradients flow to q, qd, u

The underlying ``.so`` is shared with the plain / JAX surfaces (same content-
addressed cache); the torch op block is compiled into it behind
``-DGRID_RBD_WITH_TORCH`` when torch is installed at register time.

CUDA-Graphs: ``h.capture(method, *example_inputs, **kw)`` returns a
``GraphCallable`` for fixed-batch, low-launch-overhead replay (MPC / training).
A mandatory off-graph warmup runs the one-time device-global setup (the >48 KB
dynamic-smem opt-in via grid_rbd_init) BEFORE capture, since those calls are
illegal during stream capture.

GPU/torch compatibility: the backward VJP contractions (bmm / eye) and tensor
allocation run torch's own CUDA kernels, so the installed torch must support the
GPU's compute capability. On an RTX 5090 (sm_120) you need a cu128 (or newer)
torch build — a cu124 wheel (max sm_90) cannot launch any CUDA kernel on
sm_120 ("no kernel image is available for execution on the device"). The
grid_plant/grid kernels themselves are always nvcc-built for the detected arch
and are unaffected; only torch's own kernels carry this requirement.
"""
from __future__ import annotations

import ctypes
import threading
from pathlib import Path
from typing import Any

import grid_rbd as _grid_rbd
from grid_rbd._handle import RobotHandle, SecondOrderID, SecondOrderFD, _integrator_code
from .._out_transform import apply_out_layout
from .._surface_common import BaseDelegateMixin, MujocoDerivativeViewMixin, MujocoViewBase


# ─── op-library registry (process-global, idempotent) ───────────────────────

# Maps cache_key → the torch op namespace (e.g. "grid_rbd_torch_k03e0c8d05d7e").
# torch.ops.load_library is idempotent per .so path, but we guard so a later
# notebook cell calling a method never re-loads.
_LOADED: dict[str, str] = {}
_LOCK = threading.Lock()


def _require_torch():
    """Import + return the ``torch`` module, or raise a clear, actionable error.

    Routed through the surface entry points (register_robot / get_robot /
    _load_ops) so a missing optional dep gives install guidance instead of a
    bare ``ModuleNotFoundError`` from deep inside a method."""
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "grid_rbd.torch requires PyTorch, which isn't installed. Install "
            "the optional extra:  pip install -e 'grid-rbd[torch]'  (or "
            "`pip install torch`; on an RTX 5090 / sm_120 you need a cu128+ "
            "build). The numpy and jax backends do not need torch."
        ) from e
    return torch


def _torch_op_namespace(cache_key: str) -> str:
    """Mirror _compile.compile_sources's torch_op_key = 'k' + key[:12]."""
    return f"grid_rbd_torch_k{cache_key[:12]}"


# A torch op that is ALWAYS registered when the torch surface compiles (an ungated
# plant cost op — not behind any per-algo `#if GRID_HAS_*`). Used to tell a
# SUBSET-omitted CORE algo (this present, the requested op absent) apart from a
# never-built op, so the clean "not built — add to algorithm_list" error only fires
# for a genuine subset gap.
_TORCH_SURFACE_SENTINEL = "quadratic_input_cost"


def _resolve_core_op(ops, name: str):
    """Resolve a CORE torch op (id/fd/minv/.../regressor/integrator/...), mapping a
    missing op to the clean subset error the numpy rc=3 path raises. A core op is
    absent only on a SUBSET build (algorithm_list) that omitted it — distinguished
    from a whole-surface-missing .so via the always-present sentinel op."""
    try:
        return getattr(ops, name)
    except AttributeError as e:
        # mjx-suffixed names that resolve through here strip the suffix for the msg.
        base = name[:-len("_mujoco")] if name.endswith("_mujoco") else name
        if hasattr(ops, _TORCH_SURFACE_SENTINEL):
            raise RuntimeError(
                f"{base!r} not built into this robot .so — add {base!r} to "
                f"algorithm_list in register_robot() and rebuild (force_rebuild=True). "
                f"The torch op surface is present but this algorithm was excluded by "
                f"the subset build."
            ) from e
        raise


def _load_ops(so_path: Path, cache_key: str) -> str:
    """Load the .so's torch ops (once per cache_key); return the op namespace."""
    torch = _require_torch()
    ns = _torch_op_namespace(cache_key)
    with _LOCK:
        if _LOADED.get(cache_key):
            return ns
        # Sanity: the .so must export the torch op block. If it was built
        # without torch (torch installed after register), raise a clear error.
        try:
            lib = ctypes.CDLL(str(so_path))
            del lib
        except OSError as e:
            raise RuntimeError(f"failed to dlopen {so_path}: {e}") from e
        torch.ops.load_library(str(so_path))
        # torch.ops.<ns> is created lazily, so its mere existence proves nothing;
        # probe for the ALWAYS-PRESENT sentinel op (an ungated plant cost op) to
        # confirm the TORCH_LIBRARY block registered. Probing a core op (e.g.
        # inverse_dynamics) would FALSELY fail a SUBSET .so that omits that core,
        # so use the sentinel which every torch surface emits.
        try:
            getattr(getattr(torch.ops, ns), _TORCH_SURFACE_SENTINEL)
        except AttributeError as e:
            raise RuntimeError(
                f"torch op {ns}.{_TORCH_SURFACE_SENTINEL} not found in {so_path}; was the .so "
                f"compiled with GRID_RBD_WITH_TORCH (torch installed at register time)? "
                f"Re-register with force_rebuild=True."
            ) from e
        _LOADED[cache_key] = ns
    return ns


# ─── autograd Functions (backward = batched VJP via the *_gradient kernels) ──
#
# GRiD emits the FULL analytic per-timestep Jacobian; torch autograd needs vᵀJ.
# Each backward calls the gradient op (forward of the gradient), reshapes to the
# row-major convention, then bmm's the upstream grad against it — on-GPU and
# itself graph-capturable.


def _make_autograd(ns, nv, mujoco=False):
    import torch

    ops = getattr(torch.ops, ns)

    def _op(name):
        # In mjx mode every forward/backward op dispatches to its _mujoco
        # variant (kernel launched with MUJOCO_OUTPUT=true): mjx-convention
        # forward + mjx-convention analytic Jacobian, so backward stays
        # self-consistent. mjx is FLOATING-base only (the _mujoco symbols are
        # #ifdef'd out of fixed .so). A CORE op absent on a SUBSET build maps to
        # the clean "not built — add to algorithm_list" error (not a bare
        # AttributeError) via _resolve_core_op.
        return _resolve_core_op(ops, (name + "_mujoco") if mujoco else name)

    # A4-1 vjp_ops: each backward is a SHELL — the recipe (grad op, per-input
    # cotangents, the floating-base nv-slice/nj-pad bridge) lives in
    # ABI_SPECS[key].vjp and runs through the shared _vjp_common.vjp_backward
    # driver, identical to the jax surface.
    from grid_codegen.abi_specs import ABI_SPECS
    from .._vjp_common import vjp_backward

    class InverseDynamicsFn(torch.autograd.Function):
        # forward args mirror the op schema order (q, qd, gravity, qdd, f_ext);
        # qdd/gravity/f_ext are non-differentiated (backward returns None for them).
        @staticmethod
        def forward(ctx, q, qd, gravity, qdd, f_ext):
            ctx.save_for_backward(q, qd)
            ctx.gravity = gravity
            ctx.qdd = qdd
            ctx.f_ext = f_ext
            ctx.nv = nv
            return _op("inverse_dynamics")(q, qd, gravity, qdd, f_ext)

        @staticmethod
        def backward(ctx, grad_c):
            q, qd = ctx.saved_tensors
            # f_ext is affine in RNEA (passed through for bias consistency);
            # qdd threads into the USE_QDD grad overload so ∂(M·qdd)/∂q is
            # included. Recipe + nv-slice/nj-pad bridge: the shared vjp table.
            g = vjp_backward(ABI_SPECS["inverse_dynamics"].vjp, grad_c, {
                "grad": lambda: apply_out_layout(
                    _op("inverse_dynamics_gradient")(q, qd, ctx.gravity, ctx.qdd, ctx.f_ext),
                    ("grad_concat",), None, nv=nv),
            }, nv=nv, nj=q.shape[1])
            # grads for (q, qd, gravity, qdd, f_ext)
            return g["q"], g["qd"], None, g["qdd"], g["f_ext"]

    def _make_fd_like(fwd_op):
        # forward_dynamics & aba share the qdd output and the fd-grad backward
        # (∂qdd/∂(q,qd) from fd_grad; ∂qdd/∂u = M⁻¹). One factory, two ops.
        class FDLikeFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q, qd, u, gravity, f_ext):
                ctx.save_for_backward(q, qd, u)
                ctx.gravity = gravity
                ctx.f_ext = f_ext
                return fwd_op(q, qd, u, gravity, f_ext)

            @staticmethod
            def backward(ctx, grad_qdd):
                q, qd, u = ctx.saved_tensors
                g = vjp_backward(ABI_SPECS["forward_dynamics"].vjp, grad_qdd, {
                    "grad": lambda: apply_out_layout(
                        _op("forward_dynamics_gradient")(q, qd, u, ctx.gravity, ctx.f_ext),
                        ("grad_concat",), None, nv=nv),
                    # ∂qdd/∂u = M⁻¹ via the shared minv layout (pin symmetrize
                    # / mjx full-dense — _out_transform).
                    "minv": lambda: apply_out_layout(
                        _op("minv")(q), ("minv",), None, nv=nv, mjx=mujoco,
                        eye=torch.eye(nv, dtype=q.dtype, device=q.device)),
                }, nv=nv, nj=q.shape[1])
                return g["q"], g["qd"], g["u"], None, g["f_ext"]
        return FDLikeFn

    FDFn = _make_fd_like(lambda q, qd, u, g, fe: _op("forward_dynamics")(q, qd, u, g, fe))
    AbaFn = _make_fd_like(lambda q, qd, u, g, fe: _op("aba")(q, qd, u, g, fe))

    # ── inertial-parameter (sysID) VJPs ──
    # The forward op is independent of the `params` (π) VALUE (the compiled .so
    # carries the baked-in inertia); π exists so autograd can flow the analytic
    # ∂(·)/∂π to it — the linearization of the bias / qdd around the compiled
    # model (mirrors the JAX idyn_pi / fd_pi custom_vjp). q/qd[/u] cotangents
    # flow exactly as the plain id / fd VJPs above.

    class IDWrtParamsFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, qd, params, gravity, f_ext):
            # forward value ignores `params`; baked-in inertia → c = ID(q, qd).
            ctx.save_for_backward(q, qd)
            ctx.gravity = gravity
            ctx.f_ext = f_ext
            return ops.inverse_dynamics(q, qd, gravity, None, f_ext)

        @staticmethod
        def backward(ctx, grad_c):
            q, qd = ctx.saved_tensors
            # sysID is the bias linearization (qdd=0): None/zeros thread into
            # the grad op's qdd slot and the regressor.
            g = vjp_backward(ABI_SPECS["inverse_dynamics_wrt_params"].vjp, grad_c, {
                "grad": lambda: apply_out_layout(
                    ops.inverse_dynamics_gradient(q, qd, ctx.gravity, None, ctx.f_ext),
                    ("grad_concat",), None, nv=nv),
                "param_grad": lambda: ops.inverse_dynamics_regressor(
                    q, qd, torch.zeros_like(q), ctx.gravity).reshape(q.shape[0], nv, -1),
            }, nv=nv, nj=q.shape[1])
            return g["q"], g["qd"], g["params"], None, None

    class FDWrtParamsFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, qd, u, params, gravity, f_ext):
            ctx.save_for_backward(q, qd, u)
            ctx.gravity = gravity
            ctx.f_ext = f_ext
            return ops.forward_dynamics(q, qd, u, gravity, f_ext)

        @staticmethod
        def backward(ctx, grad_qdd):
            q, qd, u = ctx.saved_tensors
            g = vjp_backward(ABI_SPECS["forward_dynamics_wrt_params"].vjp, grad_qdd, {
                "grad": lambda: apply_out_layout(
                    ops.forward_dynamics_gradient(q, qd, u, ctx.gravity, ctx.f_ext),
                    ("grad_concat",), None, nv=nv),
                # wrt_params is pin-only (mjx omits it) → always the pin symmetrize.
                "minv": lambda: apply_out_layout(
                    ops.minv(q), ("minv",), None, nv=nv,
                    eye=torch.eye(nv, dtype=q.dtype, device=q.device)),
                "param_grad": lambda: ops.forward_dynamics_parameter_gradient(
                    q, qd, u, ctx.gravity).reshape(q.shape[0], nv, -1),
            }, nv=nv, nj=q.shape[1])
            return g["q"], g["qd"], g["u"], g["params"], None, None

    class IntegratorFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, qd, u, dt, it, gravity):
            ctx.save_for_backward(q, qd, u)
            ctx.dt, ctx.it, ctx.gravity = dt, it, gravity
            return _op("integrator")(q, qd, u, dt, it, gravity)

        @staticmethod
        def backward(ctx, grad_x):
            q, qd, u = ctx.saved_tensors
            # A3-audit fix (2026-09-09): this used to shadow the closure's nv
            # with q.shape[1] (= nq). Fixed base: nq == nv, worked by luck.
            # Floating base: nq = nv+1, so the reshape below mis-sized AND the
            # tangent (2nv-wide) Jacobian cannot be chained to the (nq+nv)-wide
            # x cotangent without the quaternion-chart VJP — unimplemented, so
            # say so instead of crashing on a confusing reshape.
            if q.shape[1] != nv:
                raise NotImplementedError(
                    "integrator backward on a FLOATING base needs the SE(3) "
                    "chart VJP (tangent 2*nv Jacobian vs nq+nv state cotangent) "
                    "— differentiate integrator_gradient outputs directly, or "
                    "use a fixed-base robot.")
            g = vjp_backward(ABI_SPECS["integrator"].vjp, grad_x, {
                # h_dAB is (2*NV x 3*NV) column-major per ts → shared out-layout
                # (colmajor_whole → row-major (B, 2NV, 3NV)); thirds = gq/gqd/gu.
                "grad": lambda: apply_out_layout(
                    _op("integrator_gradient")(q, qd, u, ctx.dt, ctx.it, ctx.gravity),
                    ("colmajor_whole", None), (2 * nv, 3 * nv), nv=nv),
            }, nv=nv, nj=q.shape[1])
            return g["q"], g["qd"], g["u"], None, None, None

    fns = {"inverse_dynamics": InverseDynamicsFn, "fd": FDFn, "aba": AbaFn,
           "integrator": IntegratorFn}
    if not mujoco:
        # sysID (inverse/forward_dynamics_wrt_params) has NO _mujoco kernel; omit
        # in mjx mode so a caller hitting it gets a clean KeyError, not a
        # missing-symbol crash.
        fns["id_wrt_params"] = IDWrtParamsFn
        fns["fd_wrt_params"] = FDWrtParamsFn
    return fns


# ─── CUDA-Graphs callable ───────────────────────────────────────────────────


class GraphCallable:
    """A CUDA-Graphs-captured op for fixed-batch replay.

    ``static_in`` are the captured input tensors (``.copy_()`` new data in);
    ``static_out`` is the captured output; ``replay()`` re-runs the graph.

    Two usage modes, with different costs:

    * ``replay()`` — re-runs the captured graph on the CURRENT contents of
      ``static_in`` (write new data into those buffers in place, or reuse the
      captured inputs). One ``cudaGraphLaunch``: CPU submission cost collapses
      to ~2 us regardless of how many kernels/memcpys were captured.
    * ``__call__(*inputs)`` — convenience: copies each input D->D into
      ``static_in`` first, then replays. The copy-in adds one fused foreach
      launch (or one per tensor on older torch) of REAL GPU + CPU work per
      call, so for identical repeated inputs prefer ``replay()``.

    Performance note: a CUDA graph removes CPU launch overhead; it does not
    speed up the kernels themselves. If the captured op is GPU-execution-bound
    (large batch and/or a heavy kernel), replay wall time ~= eager wall time
    and the win is the freed CPU time (submission drops from ~10-15 us of
    per-op dispatch to ~2 us), which matters when the python thread has other
    work (RL/training loops) or when many small ops are captured together.
    """

    def __init__(self, op, example_inputs, kwargs):
        import torch
        self._torch = torch
        self.static_in = [t.clone() for t in example_inputs]
        self._kwargs = kwargs
        # 1. WARMUP off-graph: forces grid_rbd_init (>48KB smem opt-in) + first
        #    allocs. These device-global registrations are illegal during capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out = op(*self.static_in, **kwargs)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        # 2. CAPTURE: only the memcpy-repack + kernel launch + memcpy-out remain,
        #    all stream-ordered / capturable.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = op(*self.static_in, **kwargs)
        self._op = op

    def replay(self):
        self.graph.replay()
        return self.static_out

    def __call__(self, *inputs):
        if len(inputs) != len(self.static_in):
            raise ValueError(f"expected {len(self.static_in)} inputs, got {len(inputs)}")
        # Fused copy-in: one dispatcher hop for all inputs (falls back to a
        # per-tensor loop on torch builds without _foreach_copy_).
        foreach = getattr(self._torch, "_foreach_copy_", None)
        if foreach is not None:
            foreach(self.static_in, list(inputs))
        else:
            for dst, src in zip(self.static_in, inputs):
                dst.copy_(src)
        return self.replay()


# ─── mjx view ────────────────────────────────────────────────────────────────


class _TorchMujocoView(MujocoDerivativeViewMixin, MujocoViewBase):
    """MuJoCo-native, autograd-aware view over a :class:`TorchRobotHandle`
    (``handle.mujoco``). Method surface shared with the jax view via
    grid_rbd._surface_common; the differentiable methods (inverse_dynamics/
    forward_dynamics/aba/integrator) stay autograd-aware (the backward uses the
    mjx-convention analytic Jacobian)."""

    __slots__ = ()


# ─── TorchRobotHandle ───────────────────────────────────────────────────────


class TorchRobotHandle(BaseDelegateMixin):
    """Torch-flavored wrapper. Methods return ``torch.Tensor`` (autograd-aware
    for inverse_dynamics / forward_dynamics / aba / integrator)."""

    def __init__(self, base: RobotHandle, cache_key: str, so_path: str,
                 output_convention: str = "pinocchio"):
        self._base = base
        self._cache_key = cache_key
        self._so_path = Path(so_path)
        self._ns = _load_ops(self._so_path, cache_key)
        import torch
        self._ops = getattr(torch.ops, self._ns)
        # Per-convention registry of autograd Functions, built lazily. mjx mode
        # binds every op to its _mujoco variant (floating-base only).
        self._fns_cache: dict[str, dict] = {}
        # Route through the BaseDelegateMixin setter: validates the value AND,
        # on a floating base, mjx-twin presence (fixed base: "mujoco" = no-op).
        self.output_convention = output_convention
        self._mjx_view = None

    # ─── metadata (delegated) ────────────────────────────────────────────
    @property
    def name(self) -> str:        return self._base.name
    @property
    def num_joints(self) -> int:  return self._base.num_joints
    @property
    def num_vel(self) -> int:     return self._base.num_vel
    @property
    def num_ees(self) -> int:     return self._base.num_ees
    @property
    def num_bodies(self) -> int:  return self._base.num_bodies
    @property
    def floating_base(self) -> bool: return self._base.floating_base
    @property
    def max_batch(self) -> int:   return self._base.max_batch

    # ─── output convention (mjx parity) ──────────────────────────────────
    # output_convention property/setter + _mjx_active come from BaseDelegateMixin
    # (numpy semantics: "mujoco" on a fixed base is a uniform-interface no-op).

    def _resolve_convention(self, convention):
        """None → the handle default; else the explicit per-call convention."""
        return self._output_convention if convention is None else convention

    def _shape_out(self, key, raw, *, mjx: bool = False):
        """Apply the spec's out_layout (A3 slice 4) via the shared transform
        module — replaces the reshape chains copied verbatim from _handle.py."""
        import torch
        from .._out_transform import shape_out_for
        nv = self.num_vel
        return shape_out_for(key, raw, nq=self.num_joints, nv=nv,
                             nee=self.num_ees, nb=int(self.num_bodies),
                             mjx=mjx,
                             eye=torch.eye(nv, dtype=raw.dtype, device=raw.device))

    def _fns_for(self, convention):
        """Per-convention autograd Function registry (lazily built + cached).
        ``convention="mujoco"`` builds the SAME closures but every op is the
        ``_mujoco`` variant; mjx is floating-base only."""
        conv = self._resolve_convention(convention)
        if conv not in ("pinocchio", "mujoco"):
            raise ValueError(
                f"output_convention must be 'pinocchio' or 'mujoco'; got {conv!r}")
        cache = self._fns_cache
        if conv not in cache:
            # mjx closures only when ACTIVE (floating base) — on a fixed base the
            # pin closures ARE the mjx closures (the conventions coincide).
            cache[conv] = _make_autograd(self._ns, self._base.num_vel,
                                         mujoco=self._mjx_active(conv))
        return cache[conv]

    @property
    def _fns(self):
        """The pinocchio-convention autograd registry (existing pin call sites)."""
        return self._fns_for("pinocchio")

    def _op(self, conv, name):
        """Resolve a DIRECT (non-autograd) op, dispatching to the ``_mujoco``
        variant when the mjx convention is ACTIVE (resolved "mujoco" AND floating
        base — fixed base: the pin op is the correct no-op).

        Missing-op handling is driven by ``ABI_SPECS[name].gate_form`` (the same
        field the C emitter uses) instead of a per-call-site chooser:
        - ``"if"`` (core subset-gated): resolve via ``_resolve_core_op`` → the
          clean "not built — add to algorithm_list" error.
        - ``"ifdef"`` (emitted-when-present: the centroidal / kinematics family
          com / ccrba / energy / dccrba / cmm_time_variation /
          frame_jacobian[_dot] / osc_inertia / runtime-EE pair): resolve the raw
          op and, when absent, raise the per-algo advice from
          ``ABI_SPECS.py_rc3_msg`` (same table the numpy rc==3 path uses)."""
        resolved = (name + "_mujoco") if self._mjx_active(conv) else name
        try:
            from grid_codegen.abi_specs import ABI_SPECS
            spec = ABI_SPECS.get(name)
        except Exception:
            spec = None
        if spec is None or spec.gate_form != "ifdef":
            return _resolve_core_op(self._ops, resolved)
        try:
            return getattr(self._ops, resolved)
        except AttributeError as e:
            raise AttributeError(
                spec.py_rc3_msg
                or (f"{name} not generated for this robot .so "
                    f"(subset algorithm_list, or unsupported for this robot class)")
            ) from e

    @property
    def mujoco(self) -> "_TorchMujocoView":
        """MuJoCo-native view (``handle.mujoco.inverse_dynamics(qpos, qvel, qacc)``):
        forwards a per-call ``_convention="mujoco"`` WITHOUT mutating the shared
        ``output_convention`` default, so it is safe alongside pinocchio calls."""
        v = self._mjx_view
        if v is None:
            v = self._mjx_view = _TorchMujocoView(self)
        return v

    # ─── differentiable algorithms ───────────────────────────────────────

    def inverse_dynamics(self, q, qd, qdd=None, *, gravity: float = -9.81, f_ext=None,
                         _convention=None):
        """Inverse dynamics (RNEA) τ = M·qdd + h − g (B, NJ). Autograd-aware wrt (q, qd).

        ``qdd`` (optional): joint acceleration, CUDA float32 ``(B, NJ)``. With
        ``qdd=None`` (default) returns the bias c = h − g; a nonzero ``qdd`` adds
        the M·qdd inertial term (USE_QDD overload). The autograd backward threads
        the saved qdd through, so the q/qd Jacobian includes ∂(M·qdd)/∂q for a
        nonzero-qdd call; qdd itself is not differentiated.

        ``f_ext`` (optional): per-body external forces, a CUDA float32 tensor
        ``(B, 6*num_bodies)``, body-major, each ``[angular; linear]`` in the
        body's local frame (subtracted from the per-body force; matches the
        numpy handle and ``RBDReference.inverse_dynamics(..., f_ext=...)``).

        With ``output_convention="mujoco"`` (floating base) inputs/outputs are
        MuJoCo-convention and the autograd VJP uses the mjx-convention Jacobian."""
        self._refuse_mjx_f_ext("inverse_dynamics", f_ext, _convention)
        return self._fns_for(_convention)["inverse_dynamics"].apply(
            q, qd, float(gravity), qdd, f_ext)

    def forward_dynamics(self, q, qd, u, *, gravity: float = -9.81, f_ext=None,
                         _convention=None):
        """qdd = M⁻¹(τ − c) (B, NJ). Autograd-aware wrt (q, qd, u).

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``
        CUDA float32 (see :py:meth:`inverse_dynamics`).

        With ``output_convention="mujoco"`` (floating base) inputs/outputs are
        MuJoCo-convention and the autograd VJP uses the mjx-convention Jacobian."""
        self._refuse_mjx_f_ext("forward_dynamics", f_ext, _convention)
        return self._fns_for(_convention)["fd"].apply(q, qd, u, float(gravity), f_ext)

    def aba(self, q, qd, u, *, gravity: float = -9.81, f_ext=None, _convention=None):
        """qdd via ABA (B, NJ). Autograd-aware wrt (q, qd, u).

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``
        CUDA float32 (see :py:meth:`inverse_dynamics`).

        With ``output_convention="mujoco"`` (floating base) inputs/outputs are
        MuJoCo-convention."""
        self._refuse_mjx_f_ext("aba", f_ext, _convention)
        return self._fns_for(_convention)["aba"].apply(q, qd, u, float(gravity), f_ext)

    def inverse_dynamics_wrt_params(self, q, qd, params, *, gravity: float = -9.81, f_ext=None):
        """Inverse-dynamics bias c = ID(q, qd, qdd=0) (B, NJ), differentiable wrt
        the per-link inertial parameters ``params`` (π) AND ``q``/``qd``.

        ``params``: (B, 10*num_bodies) — per-link [m, m*c(3), I_O(6)] in the
        parser's origin-frame basis (same as ``inverse_dynamics_regressor`` Y and
        ``RBDReference._regressor``). The FORWARD value is independent of
        ``params`` (the compiled ``.so`` carries the baked-in inertia); the op
        exists so ``torch.autograd`` flows the analytic ``∂c/∂π = Y(q,qd,qdd=0)``
        (regressor) to ``params`` — the outer-loop system-ID gradient.
        ``q``/``qd`` gradients are unchanged. Mirrors the JAX
        ``inverse_dynamics_wrt_params`` custom_vjp."""
        return self._fns["id_wrt_params"].apply(q, qd, params, float(gravity), f_ext)

    def forward_dynamics_wrt_params(self, q, qd, u, params, *, gravity: float = -9.81, f_ext=None):
        """Forward dynamics qdd = FD(q, qd, u) (B, NJ), differentiable wrt the
        per-link inertial parameters ``params`` (π) AND ``q``/``qd``/``u``.

        ``params``: (B, 10*num_bodies) — see :py:meth:`inverse_dynamics_wrt_params`.
        The forward value is independent of ``params`` (baked-in inertia); the VJP
        flows the analytic ``∂qdd/∂π = -M⁻¹·Y`` (the
        ``forward_dynamics_parameter_gradient`` kernel) to ``params``.
        ``q``/``qd``/``u`` gradients are unchanged. Mirrors the JAX
        ``forward_dynamics_wrt_params`` custom_vjp."""
        return self._fns["fd_wrt_params"].apply(q, qd, u, params, float(gravity), f_ext)

    def integrator(self, q, qd, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81,
                   _convention=None):
        """x_{k+1} (B, NP+NV). Autograd-aware wrt (q, qd, u).

        With ``output_convention="mujoco"`` (floating base) inputs/outputs are
        MuJoCo-convention (global-additive base retract + reframed base velocity)."""
        it = _integrator_code(integrator_type)
        return self._fns_for(_convention)["integrator"].apply(
            q, qd, u, float(dt), it, float(gravity))

    # ─── forward-only algorithms (raw kernel ops; reshapes mirror _handle) ──

    def minv(self, q, *, _convention=None):
        """Minv(q) (B, NV, NV), symmetrized (the pin kernel writes the UPPER triangle, lower zero).

        Tangent-space (pinocchio) inverse mass matrix. FIXED base: NV == NJ
        (shape unchanged); FLOATING base: NV < NJ.

        With ``output_convention="mujoco"`` (floating base) the returned Minv is the
        mjx-frame inverse mass matrix (G^-T Minv G^-1); the ``minv_mujoco`` kernel
        writes it FULL DENSE so no host symmetrize is applied."""
        import torch
        conv = self._resolve_convention(_convention)
        nv = self.num_vel
        raw = self._op(conv, "minv")(q)
        # "minv" in the shared out-layout module: pin = UPPER-triangle
        # symmetrize, mjx twin = full dense pass-through.
        return self._shape_out("minv", raw, mjx=self._mjx_active(conv))

    def crba(self, q, *, gravity: float = -9.81, _convention=None):
        """Mass matrix M(q) (B, NV, NV), tangent-space (pinocchio) convention.
        FIXED base: NV == NJ (unchanged); FLOATING base: NV < NJ.

        With ``output_convention="mujoco"`` (floating base) the returned M is the
        mjx-frame mass matrix (G M G^T congruence, written full dense)."""
        return self._shape_out("crba", self._op(_convention, "crba")(q, float(gravity)))

    # ─── centroidal / energy / kinematics value methods (numpy-handle parity) ──
    #
    # Mirror the numpy RobotHandle's centroidal / energy / regressor / frame
    # methods exactly (same torch op names, same gravity/int args, shared
    # out-layout module). Forward-only (no autograd). Every method calls `_op`;
    # missing-op behavior (subset "not built" vs the actionable "not generated
    # for this robot" advice) is chosen by ABI_SPECS[name].gate_form — the same
    # field that gates the C emit.

    def generalized_gravity(self, q, *, gravity: float = -9.81, _convention=None):
        """Generalized gravity torque g(q) = RNEA(q, 0, 0). Returns (B, NV).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned g is in the mjx frame (base rows rotated in-kernel)."""
        return self._op(_convention, "generalized_gravity")(q, float(gravity))

    def nonlinear_effects(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Nonlinear (bias) effects c(q,qd) = RNEA(q, qd, 0). Returns (B, NV).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and the returned bias matches MuJoCo's ``qfrc_bias``."""
        return self._op(_convention, "nonlinear_effects")(q, qd, float(gravity))

    def energy(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Kinetic / potential / mechanical energy. Returns (B, 3) = [KE, PE, KE+PE].

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention; the energies are frame-INVARIANT (inputs converted in-kernel)."""
        return self._op(_convention, "energy")(q, qd, float(gravity))

    def com(self, q, *, _convention=None):
        """Center-of-mass world position p_com and CoM Jacobian J_com.

        Returns ``(p_com, J_com)`` where ``p_com`` is ``(B, 3)`` and ``J_com`` is
        ``(B, 3, NV)``. The op returns one flat ``(B, 3 + 3*NV)`` buffer
        ``[p_com(3); J_com(3 x NV col-major)]`` (split mirrors the numpy handle).

        With ``output_convention="mujoco"`` (floating base) ``p_com`` is invariant
        and the ``J_com`` columns are reframed (computed in-kernel)."""
        nv = self.num_vel
        raw = self._op(_convention, "com")(q)
        return self._shape_out("com", raw)

    def ccrba(self, q, qd, *, _convention=None):
        """Centroidal momentum matrix A (6 x NV) and momentum h = A·qd (6,).

        Returns ``(A, h)`` where ``A`` is ``(B, 6, NV)`` and ``h`` is ``(B, 6)``.
        The op returns one flat ``(B, 6*NV + 6)`` buffer ``[A(6 x NV col-major); h(6)]``.

        With ``output_convention="mujoco"`` (floating base) ``h`` is invariant and
        the ``A`` columns are reframed (computed in-kernel)."""
        nv = self.num_vel
        raw = self._op(_convention, "ccrba")(q, qd)
        return self._shape_out("ccrba", raw)

    def dccrba(self, q, *, _convention=None):
        """dCCRBA tensor ∂A/∂q, shape ``(B, 6, NV, NV)`` indexed
        ``[:, :, k, i] = ∂A[:, k]/∂q_i`` (Pinocchio centroidal convention).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned tensor is the dA/dq of the mjx CMM (same layout)."""
        nv = self.num_vel
        raw = self._op(_convention, "dccrba")(q)
        return self._shape_out("dccrba", raw)

    def cmm_time_variation(self, q, qd, *, _convention=None):
        """Centroidal-momentum-matrix time variation Ȧ = dA(q(t))/dt, shape
        ``(B, 6, NV)`` = ``Σ_i (∂A/∂q_i)·qd_i``.

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and Ȧ has its columns reframed (computed in-kernel)."""
        nv = self.num_vel
        raw = self._op(_convention, "cmm_time_variation")(q, qd)
        return self._shape_out("cmm_time_variation", raw)

    def coriolis_matrix(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Coriolis matrix C(q,qd). Returns ``(B, NV, NV)`` row-major, with
        ``C·qd + g(q) = nonlinear_effects(q, qd)`` (gravity unused by C; the
        kwarg mirrors the host wrapper signature).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and the returned ``C`` is the mjx-frame Coriolis matrix."""
        nv = self.num_vel
        raw = self._op(_convention, "coriolis_matrix")(q, qd, float(gravity))
        return self._shape_out("coriolis_matrix", raw)

    def kinetic_energy_regressor(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Kinetic-energy regressor y_KE, length ``10*num_bodies``, with
        ``KE = y_KE · π``. Returns ``(B, 10*num_bodies)`` (gravity unused).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention; the regressor is frame-INVARIANT (inputs converted in-kernel)."""
        return self._op(_convention, "kinetic_energy_regressor")(q, qd, float(gravity))

    def potential_energy_regressor(self, q, *, gravity: float = -9.81, _convention=None):
        """Potential-energy regressor y_PE, length ``10*num_bodies``, with
        ``PE = y_PE · π`` (PE uses ``gravity``). Returns ``(B, 10*num_bodies)``.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention;
        the regressor is frame-INVARIANT (input converted in-kernel)."""
        return self._op(_convention, "potential_energy_regressor")(q, float(gravity))

    def frame_jacobian(self, q, *, target_jid=None, reference_frame=None, _convention=None):
        """Geometric Jacobian (6 x NV, ``[linear; angular]``) of a frame.
        Returns ``(B, 6, NV)``.

        ``target_jid`` selects the frame's joint id (default: the leaf
        end-effector joint baked at codegen time). ``reference_frame`` is
        ``'LOCAL'`` (0), ``'WORLD'`` (1), or ``'LOCAL_WORLD_ALIGNED'`` (2, the
        default), or the equivalent int — passed as int op args.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned Jacobian is column-reframed ``J G⁻¹`` (mjx frame)."""
        from .._handle import _resolve_frame_args
        nv = self.num_vel
        tj, rf = _resolve_frame_args(self._base._meta, target_jid, reference_frame)
        raw = self._op(_convention, "frame_jacobian")(q, int(tj), int(rf))
        return self._shape_out("frame_jacobian", raw)

    def frame_jacobian_dot(self, q, qd, *, target_jid=None, reference_frame=None, _convention=None):
        """Time derivative Jdot of :py:meth:`frame_jacobian` along v = qd
        (6 x NV, ``[linear; angular]``). Returns ``(B, 6, NV)``.

        ``target_jid`` / ``reference_frame`` are runtime int op args (default:
        leaf-EE joint / ``LOCAL_WORLD_ALIGNED``); see :py:meth:`frame_jacobian`.

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and Jdot is column-reframed (mjx frame)."""
        from .._handle import _resolve_frame_args
        nv = self.num_vel
        tj, rf = _resolve_frame_args(self._base._meta, target_jid, reference_frame)
        raw = self._op(_convention, "frame_jacobian_dot")(q, qd, int(tj), int(rf))
        return self._shape_out("frame_jacobian_dot", raw)

    def osc_inertia(self, q, *, _convention=None):
        """Operational-space (task) inertia Lambda = (J·M⁻¹·Jᵀ)⁻¹ (6 x 6) for
        the leaf-EE frame (LWA). Returns ``(B, 6, 6)``.

        With ``output_convention="mujoco"`` the MuJoCo ``q`` is reordered before the
        kinematics build (Lambda is otherwise frame-INVARIANT)."""
        return self._shape_out("osc_inertia", self._op(_convention, "osc_inertia")(q))

    def end_effector_pose(self, q, *, _convention=None):
        """EE pose [xyz, rpy] per EE (B, 6*NUM_EES).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention."""
        return self._op(_convention, "end_effector_pose")(q)

    def end_effector_pose_gradient(self, q, *, _convention=None):
        """EE pose Jacobian d/dv (B, 6*NEE, NV), pinocchio tangent convention.

        With ``output_convention="mujoco"`` (floating base) the base-velocity
        Jacobian columns are reframed to the mjx free-joint tangent (J·G^-1)."""
        nee, nv = self.num_ees, self.num_vel
        raw = self._op(_convention, "end_effector_pose_gradient")(q)
        return self._shape_out("end_effector_pose_gradient", raw)

    def end_effector_pose_hessian(self, q, *, _convention=None):
        """EE pose Hessian d²/dv² (B, 6*NEE, NV, NV).

        With ``output_convention="mujoco"`` (floating base) the base-tangent indices
        are reframed to the mjx free-joint convention."""
        nee, nv = self.num_ees, self.num_vel
        return self._shape_out("end_effector_pose_hessian", self._op(_convention, "end_effector_pose_hessian")(q))

    def end_effector_pose_runtime(self, q, ee_joint_names=None, ee_offsets=None,
                                  *, _convention=None):
        """Runtime-target multi-EE pose ``[xyz; rpy]`` at an offset point.

        Mirrors :py:meth:`grid_rbd.RobotHandle.end_effector_pose_runtime`:
        ``ee_joint_names`` (None => all leaf joints, or a str / list of joint
        names) selects the EE frames; ``ee_offsets`` (None => frame origin, or one
        ``[x,y,z]`` / ``[x,y,z,1]`` per EE) shifts the measurement point. The
        single-target op is looped over the resolved jid list and stacked.

        Returns ``(B, NUM_EE, 6)``. Forward-only (no autograd).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention;
        the world pose VALUE is frame-invariant."""
        import torch
        jids = self._base._resolve_ee_jids(ee_joint_names)
        offsets = self._base._normalize_ee_offsets(ee_offsets, len(jids))
        op = self._op(_convention, "end_effector_pose_runtime")
        per_ee = []
        for jid, off in zip(jids, offsets):
            # off is the 16-float col-major X_tool from _normalize_ee_offsets;
            # the op wants ALL 16 (translation lives at [12..14], not [0..2]).
            off_t = torch.as_tensor(off, dtype=q.dtype, device=q.device).reshape(-1).contiguous()
            per_ee.append(op(q, int(jid), off_t))  # (B, 6)
        return torch.stack(per_ee, dim=1)  # (B, NUM_EE, 6)

    def end_effector_pose_gradient_runtime(self, q, ee_joint_names=None, ee_offsets=None,
                                           *, _convention=None):
        """Runtime-target multi-EE pose gradient ``d[xyz; rpy]/dv`` (6 x NV) at an
        offset point. Same ``ee_joint_names`` / ``ee_offsets`` semantics as
        :py:meth:`end_effector_pose_runtime`; mirrors
        :py:meth:`grid_rbd.RobotHandle.end_effector_pose_gradient_runtime`.

        The kernel writes column-major ``(B, 6*NV)`` per target; we reshape
        ``(B, NV, 6)`` then transpose to ``(B, 6, NV)`` and stack.

        Returns ``(B, NUM_EE, 6, NV)``. Forward-only (no autograd).

        With ``output_convention="mujoco"`` (floating base) the base-linear columns
        are reframed by R^T in-kernel (column-reframe class)."""
        import torch
        nv = self.num_vel
        jids = self._base._resolve_ee_jids(ee_joint_names)
        offsets = self._base._normalize_ee_offsets(ee_offsets, len(jids))
        op = self._op(_convention, "end_effector_pose_gradient_runtime")
        per_ee = []
        for jid, off in zip(jids, offsets):
            off_t = torch.as_tensor(off, dtype=q.dtype, device=q.device).reshape(-1).contiguous()
            raw = op(q, int(jid), off_t)  # (B, 6*NV) col-major
            per_ee.append(self._shape_out("end_effector_pose_gradient_runtime", raw))
        return torch.stack(per_ee, dim=1)  # (B, NUM_EE, 6, NV)

    def inverse_dynamics_gradient(self, q, qd, qdd=None, *, gravity: float = -9.81, f_ext=None,
                                  _convention=None):
        """∂c/∂(q,qd) (B, NV, 2*NV) = [dc_dq | dc_dqd], tangent-space (pinocchio)
        convention. FIXED base: NV == NJ (unchanged); FLOATING base: NV < NJ.

        ``qdd`` (optional): joint acceleration ``(B, NJ)``. With ``qdd=None``
        (default) this is the bias gradient ∂(h−g)/∂(q,qd); a nonzero ``qdd``
        adds ∂(M·qdd)/∂q (USE_QDD overload).

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``;
        affine in f_ext so a constant f_ext leaves this Jacobian unchanged.

        With ``output_convention="mujoco"`` (floating base) the gradient is the
        mjx-convention Jacobian (rows base-rotated, columns base-reframed)."""
        self._refuse_mjx_f_ext("inverse_dynamics_gradient", f_ext, _convention)
        nv = self.num_vel
        raw = self._op(_convention, "inverse_dynamics_gradient")(q, qd, float(gravity), qdd, f_ext)
        return self._shape_out("inverse_dynamics_gradient", raw)

    def forward_dynamics_gradient(self, q, qd, u, *, gravity: float = -9.81, f_ext=None,
                                  _convention=None):
        """∂qdd/∂(q,qd) (B, NV, 2*NV), tangent-space (pinocchio) convention.
        FIXED base: NV == NJ (unchanged); FLOATING base: NV < NJ.

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``;
        affine in f_ext so a constant f_ext leaves this Jacobian unchanged.

        With ``output_convention="mujoco"`` (floating base) the gradient is the
        mjx-convention Jacobian."""
        self._refuse_mjx_f_ext("forward_dynamics_gradient", f_ext, _convention)
        nv = self.num_vel
        raw = self._op(_convention, "forward_dynamics_gradient")(q, qd, u, float(gravity), f_ext)
        return self._shape_out("forward_dynamics_gradient", raw)

    def inverse_dynamics_regressor(self, q, qd, qdd=None, *, gravity: float = -9.81,
                                   _convention=None):
        """Joint-torque regressor Y with τ = Y·π (∂τ/∂π). Returns
        (B, NV, 10*num_bodies), row-major (NV, 10*NB) per sample. ``qdd=None`` ⇒
        zeros (the bias regressor used by :py:meth:`inverse_dynamics_wrt_params`).
        Per-link basis [m, m*c(3), I_O(6)].

        With ``output_convention="mujoco"`` (floating base) the base-linear rows are
        rotated to the mjx frame (same covector transform as the τ value)."""
        import torch
        nv, npar = self.num_vel, 10 * self.num_bodies
        if qdd is None:
            q = torch.as_tensor(q)
            qdd = torch.zeros_like(q)
        return self._op(_convention, "inverse_dynamics_regressor")(
            q, qd, qdd, float(gravity)).reshape(-1, nv, npar)

    def forward_dynamics_parameter_gradient(self, q, qd, u, *, gravity: float = -9.81):
        """FD inertial-parameter gradient ∂qdd/∂π = -M⁻¹·Y. Returns
        (B, NV, 10*num_bodies), row-major (NV, 10*NB) per sample."""
        return self._shape_out("forward_dynamics_parameter_gradient",
                               self._ops.forward_dynamics_parameter_gradient(q, qd, u, float(gravity)))

    def idsva_so(self, q, qd, qdd=None, *, gravity: float = -9.81, _convention=None):
        """Second-order ID at joint acceleration ``qdd``. Returns a
        :class:`grid_rbd.SecondOrderID` NamedTuple of 4 tensors each
        (B, NV, NV, NV) (a plain tuple — positional unpacking / indexing work).

        ``qdd=None`` ⇒ zero acceleration (explicit zeros are passed so the
        result never depends on a stale device buffer from a prior call).

        With ``output_convention="mujoco"`` (floating base) all four tensors are in
        the mjx convention (the kernel transforms every slab)."""
        import torch
        nv = self.num_vel
        if qdd is None:
            q = torch.as_tensor(q)
            qdd = torch.zeros_like(q)
        flat = self._op(_convention, "idsva_so")(q, qd, qdd, float(gravity))
        return SecondOrderID(*self._shape_out("idsva_so", flat))

    def fdsva_so(self, q, qd, u, *, gravity: float = -9.81, _convention=None):
        """Second-order FD. Returns a :class:`grid_rbd.SecondOrderFD` NamedTuple
        of 4 tensors each (B, NV, NV, NV) (a plain tuple, positional-compatible).

        With ``output_convention="mujoco"`` (floating base) all four tensors are in
        the mjx convention."""
        nv = self.num_vel
        # mjx fdsva_so LANDED: _op routes to fdsva_so_mujoco (MUJOCO_OUTPUT=true kernel;
        # epilogue recomputes Minv/qdd/dqdd_du fresh into a disjoint scratch band — the
        # §1g/§1h liveness bug is fixed).
        flat = self._op(_convention, "fdsva_so")(q, qd, u, float(gravity))
        return SecondOrderFD(*self._shape_out("fdsva_so", flat))

    def integrator_gradient(self, q, qd, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81,
                            _convention=None):
        """dAB (B, 2*NV, 3*NV) = [d/dq | d/dqd | d/du] tangent.

        With ``output_convention="mujoco"`` (floating base) the state-transition
        Jacobian is in the mjx convention (retract tangent + reframed velocity)."""
        nv = self.num_vel
        it = _integrator_code(integrator_type)
        raw = self._op(_convention, "integrator_gradient")(q, qd, u, float(dt), it, float(gravity))
        return self._shape_out("integrator_gradient", raw)

    # ─── field-standard short aliases ────────────────────────────────────────
    # `rnea`/`fd` are the names roboticists reach for (pinocchio / frax / bard);
    # bind them to the long-named methods (aba / crba / minv already match).
    rnea = inverse_dynamics
    fd = forward_dynamics

    # ─── grid_plant surface (cost / barrier / plant-step) ────────────────────
    #
    # Mirror the numpy RobotHandle plant methods (shapes / fields / gravity /
    # integrator_type). Cost methods return (value, grad, hess); barriers return
    # (value, grad, hess_diag). value is squeezed to (B,) to match numpy.

    def quadratic_state_cost(self, x, x_des, Q, *, _convention=None):
        """1/2 sum_i Q_i (x_i - x_des_i)^2 over x=[q;qd]. Returns
        (value (B,), grad (B, NX), hess=diag(Q) (B, NX, NX)).

        With ``output_convention="mujoco"`` (floating base) ``x`` is MuJoCo-convention;
        the value is invariant, the grad base-rotates (covector) and the GN hess is
        the mjx congruence."""
        nx = self.num_joints + self.num_vel
        out, grad, hess = self._op(_convention, "quadratic_state_cost")(x, x_des, Q)
        return out[:, 0], grad, hess.reshape(-1, nx, nx)

    def quadratic_input_cost(self, u, u_des, R):
        """1/2 sum_i R_i (u_i - u_des_i)^2 over u (NV). Returns
        (value (B,), grad (B, NV), hess=diag(R) (B, NV, NV))."""
        nv = self.num_vel
        out, grad, hess = self._ops.quadratic_input_cost(u, u_des, R)
        return out[:, 0], grad, hess.reshape(-1, nv, nv)

    def joint_position_barrier(self, var, lower, upper, mu):
        """Log-barrier over NUM_POS positions. Returns
        (value (B,), grad (B, NUM_POS), hess_diag (B, NUM_POS))."""
        out, grad, hdiag = self._ops.joint_position_barrier(var, lower, upper, float(mu))
        return out[:, 0], grad, hdiag

    def joint_velocity_barrier(self, var, lower, upper, mu):
        """Log-barrier over NUM_VEL velocities. See joint_position_barrier."""
        out, grad, hdiag = self._ops.joint_velocity_barrier(var, lower, upper, float(mu))
        return out[:, 0], grad, hdiag

    def joint_torque_barrier(self, var, lower, upper, mu):
        """Log-barrier over NUM_VEL torques. See joint_position_barrier."""
        out, grad, hdiag = self._ops.joint_torque_barrier(var, lower, upper, float(mu))
        return out[:, 0], grad, hdiag

    def plant_step(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81,
                   _convention=None):
        """x_{k+1} = integrator(x_k, u_k, dt). x (B, NX); u (B, NV). Returns (B, NX).

        With ``output_convention="mujoco"`` (floating base) ``x`` is MuJoCo-convention
        and the returned next state is mjx-convention (global-additive base retract)."""
        it = _integrator_code(integrator_type)
        return self._op(_convention, "plant_step")(x, u, float(dt), it, float(gravity))

    def plant_step_gradient(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81,
                            _convention=None):
        """[A|B] = d x_{k+1}/d(x,u). x (B, NX); u (B, NV). Returns (B, 2*NV, 3*NV)
        with column blocks [d/dq | d/dqd | d/du] (tangent space).

        With ``output_convention="mujoco"`` (floating base) the state-transition
        Jacobian is in the mjx convention."""
        nv = self.num_vel
        it = _integrator_code(integrator_type)
        raw = self._op(_convention, "plant_step_gradient")(x, u, float(dt), it, float(gravity))
        # (2*NV x 3*NV) column-major per timestep → shared out-layout.
        return self._shape_out("plant_step_gradient", raw)

    def ee_pos_cost(self, q, p_des, W, *, _convention=None):
        """End-effector position cost (EE 0). q (B, NQ); p_des/W (B, 3). Returns
        (value (B,), grad_x (B, NX), GN hess_x (B, NX, NX)).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention;
        value invariant, grad covector-rotated, GN hess the mjx congruence."""
        nx = self.num_joints + self.num_vel
        out, grad, hess = self._op(_convention, "ee_pos_cost")(q, p_des, W)
        return out[:, 0], grad, hess.reshape(-1, nx, nx)

    def com_cost(self, q, p_des, W, *, _convention=None):
        """Center-of-mass tracking cost. q (B, NQ); p_des/W (B, 3). Returns
        (value (B,), grad_x (B, NX), GN hess_x (B, NX, NX)).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention;
        value invariant, grad covector-rotated, GN hess the mjx congruence."""
        nx = self.num_joints + self.num_vel
        out, grad, hess = self._op(_convention, "com_cost")(q, p_des, W)
        return out[:, 0], grad, hess.reshape(-1, nx, nx)

    def momentum_cost(self, q, qd, h_des, W, *, _convention=None):
        """Centroidal-momentum tracking cost. q (B, NQ); qd (B, NV); h_des/W (B, 6).
        Returns (value (B,), grad_x (B, NX), GN hess_x (B, NX, NX)).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention; value invariant, grad covector-rotated, GN hess congruence."""
        nx = self.num_joints + self.num_vel
        out, grad, hess = self._op(_convention, "momentum_cost")(q, qd, h_des, W)
        return out[:, 0], grad, hess.reshape(-1, nx, nx)

    # ─── CUDA-Graphs capture ─────────────────────────────────────────────

    def capture(self, method: str, *example_inputs, **kwargs) -> GraphCallable:
        """Capture ``method`` at the example inputs' fixed batch into a
        replayable CUDA graph. A mandatory off-graph warmup runs the >48 KB
        dynamic-smem opt-in (grid_rbd_init) before capture.

        Returns a :py:class:`GraphCallable` — call it with new inputs (same
        shapes) to ``.copy_()`` + ``.replay()``, or use ``.static_in`` /
        ``.replay()`` directly.
        """
        op = getattr(self, method)
        return GraphCallable(op, example_inputs, kwargs)

    # ─── lifecycle ───────────────────────────────────────────────────────────


    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"TorchRobotHandle(name={self.name!r}, num_joints={self.num_joints}, "
                f"num_vel={self.num_vel}, num_ees={self.num_ees}, "
                f"floating_base={self.floating_base})")


def _concat_blocks(blocks):
    import torch
    return torch.cat([blocks[:, 0], blocks[:, 1]], dim=-1)


# ─── public API ─────────────────────────────────────────────────────────────


def register_robot(
    name: str,
    urdf_path: str | None = None,
    *,
    urdf_string: str | None = None,
    floating_base: bool = False,
    ee_joint_names: list[str] | tuple[str, ...] | None = None,
    max_batch_size: int = 256,
    cache_dir: str | Path | None = None,
    force_rebuild: bool = False,
    cuda_arch: int | None = None,
    output_convention: str = "pinocchio",
    algorithm_list: list[str] | str | None = None,
    use_joint_dynamics: bool = False,
    runtime_joint_dynamics: bool = False,
    runtime_inertia: bool = False,
    runtime_transform: bool = False,
    enable_tool: bool = False,
    enable_mujoco_kernels: bool = True,
    dtype: str = "float32",
) -> TorchRobotHandle:
    """Register a robot for the torch backend (same cache as the plain/JAX
    surfaces). Returns a :py:class:`TorchRobotHandle`.

    ``dtype="float64"`` (Wave 2a) builds/loads the fp64 .so — the torch ops then
    take and return float64 CUDA tensors (fp32 tensors are rejected by the op's
    dtype check, and vice versa for the default fp32 build).

    ``algorithm_list`` (subset build) is supported: only the requested cores + their
    transitive deps are compiled into the torch op surface; calling a method that
    was excluded raises a clean "not built into this robot .so — add to
    algorithm_list and rebuild" error. ``None`` ⇒ the full default profile."""
    _require_torch()  # fail early with install guidance if torch is missing
    base = _grid_rbd.register_robot(
        name=name, urdf_path=urdf_path, urdf_string=urdf_string,
        floating_base=floating_base, ee_joint_names=ee_joint_names,
        max_batch_size=max_batch_size, cache_dir=cache_dir,
        force_rebuild=force_rebuild, cuda_arch=cuda_arch,
        algorithm_list=algorithm_list, use_joint_dynamics=use_joint_dynamics,  # C5
        runtime_joint_dynamics=runtime_joint_dynamics,  # C5 mutable damping/friction table
        runtime_inertia=runtime_inertia,  # D.4: mutable inertia table (torch op reads same device global)
        runtime_transform=runtime_transform,  # mutable joint-origin table (shared device global)
        enable_tool=enable_tool,  # tool welding: attach_tool/detach_tool/tool_fext surface
        enable_mujoco_kernels=enable_mujoco_kernels,  # pin-only builds (RAM/compile-time)
        dtype=dtype,  # Wave 2a: fp64 .so carries an fp64 torch surface
        _profile_overlay="torch",  # E6 torch threads overlay
    )
    _install_torch_device_pool(base)  # arena carved from torch's caching allocator
    cache_key, so_path = _lookup(name, cache_dir)
    return TorchRobotHandle(base, cache_key, so_path,
                            output_convention=output_convention)


def _install_torch_device_pool(base):
    """Carve GRiD's gridData device arena out of torch's caching allocator (a
    uint8 CUDA tensor held alive on the handle) instead of raw cudaMalloc —
    same framework-allocator integration as the jax surface (see
    RobotHandle.install_device_pool). Falls back to the cudaMalloc path when
    the arena is already initialized or the allocation does not fit."""
    import torch

    def _alloc(nbytes):
        buf = torch.empty(int(nbytes), dtype=torch.uint8, device="cuda")
        return buf, buf.data_ptr()

    return base.install_device_pool(_alloc)


def get_robot(name: str, cache_dir: str | Path | None = None, *,
              output_convention: str = "pinocchio") -> TorchRobotHandle:
    """Look up a previously-registered robot (same cache as grid_rbd.get_robot).
    ``output_convention`` ('pinocchio' or 'mujoco') mirrors register_robot and
    can also be set later via the handle's ``output_convention`` property."""
    _require_torch()  # fail early with install guidance if torch is missing
    base = _grid_rbd.get_robot(name, cache_dir=cache_dir, _profile_overlay="torch")  # E6
    _install_torch_device_pool(base)
    cache_key, so_path = _lookup(name, cache_dir)
    return TorchRobotHandle(base, cache_key, so_path,
                            output_convention=output_convention)


def _lookup(name, cache_dir):
    from grid_rbd._cache import default_cache_dir, manifest_lookup, store_dir
    cd = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    entry = manifest_lookup(cd, name)
    if entry is None:
        raise RuntimeError(f"{name!r} isn't in manifest; cache may be corrupted")
    so_path = store_dir(cd, entry["cache_key"]) / "robot.so"
    return entry["cache_key"], str(so_path)


__all__ = ["TorchRobotHandle", "GraphCallable", "register_robot", "get_robot"]
