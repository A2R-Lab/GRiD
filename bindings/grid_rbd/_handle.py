"""RobotHandle — Python-facing wrapper around a compiled per-robot .so.

The handle is what users actually interact with after register_robot()
returns. It bridges the pybind11 Runner (which dlopens the .so and calls
its C ABI) to numpy/Python conventions, and adds shape validation +
helpful error messages.

All algorithm methods take and return 2D arrays where axis 0 is the
batch dimension. Single-sample calls are supported too: ``_accept_1d``
lets each method accept 1D (unbatched) inputs and return unbatched
outputs (internally computed as batch=1 with negligible overhead).

Gravity convention
------------------
`gravity` is the **signed** gravitational acceleration along world +z, default
``-9.81`` (standard downward gravity) — the same convention as pinocchio and
``RBDReference`` (``GRAVITY=-9.81``). Pass the same value to both for matching
results; the default already matches.
"""
from __future__ import annotations

import functools
from typing import Any, NamedTuple

import numpy as np
from ._surface_common import MujocoDerivativeViewMixin, MujocoViewBase


def _unbatch(out):
    """Squeeze a leading batch dim of size 1 off a method result — an array,
    ``None``, or a (possibly named) tuple of arrays. Used by ``_accept_1d`` to
    return single-sample outputs for single-sample inputs."""
    if out is None:
        return None
    if isinstance(out, tuple):
        vals = [_unbatch(o) for o in out]
        try:
            return type(out)(*vals)          # namedtuple (e.g. SecondOrderID)
        except TypeError:
            return tuple(vals)               # plain tuple
    arr = np.asarray(out)
    return arr[0] if (arr.ndim >= 1 and arr.shape[0] == 1) else out


def _accept_1d(method):
    """Friction 6: let a numpy value/kinematic method take a SINGLE unbatched
    sample — a 1-D ``q`` of shape ``(nq,)`` (and matching 1-D ``qd``/``qdd``/``u``
    /``f_ext``) — and return correspondingly unbatched outputs. An already-batched
    ``(B, nq)`` input passes through completely unchanged (purely additive). Only
    the leading positional DOF arrays and the ``f_ext`` kwarg are reshaped to
    ``(1, -1)``; scalars (``dt``, ``gravity``), strings (``_convention``), and any
    ndim>=2 input are untouched. The reshape happens BEFORE the method body, so
    ``_check_nq_width`` still fires on an nv-wide floating-base velocity (the
    Friction 3 guard is preserved). The jax/torch surfaces already accept 1-D via
    ``vmap``; this brings the numpy surface to parity."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not args or np.asarray(args[0]).ndim != 1:
            return method(self, *args, **kwargs)
        new_args = []
        for a in args:
            if a is None:
                new_args.append(a)
                continue
            aa = np.asarray(a)
            new_args.append(aa.reshape(1, -1) if aa.ndim == 1 else a)
        fe = kwargs.get("f_ext")
        if fe is not None and np.asarray(fe).ndim == 1:
            kwargs = {**kwargs, "f_ext": np.asarray(fe).reshape(1, -1)}
        return _unbatch(method(self, *new_args, **kwargs))
    return wrapper


# ─── structured second-order return types (shared across numpy/jax/torch) ─────
#
# idsva_so / fdsva_so return four rank-3 tensors each shape (B, NV, NV, NV).
# A NamedTuple gives them names while staying a plain tuple — positional
# unpacking (`a, b, c, d = h.idsva_so(...)`) and indexing still work, so this
# is backward-compatible. The component names match RBDReference's
# ``idsva_so_*`` return order (d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq).


class SecondOrderID(NamedTuple):
    """Second-order inverse-dynamics tensors, each shape ``(B, NV, NV, NV)``.

    Matches ``RBDReference.idsva_so_body_frame`` / ``idsva_so_world_frame``:
    ``(d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq)``.
    """
    d2tau_dq: Any
    d2tau_dqd: Any
    d2tau_cross: Any
    dM_dq: Any


class SecondOrderFD(NamedTuple):
    """Second-order forward-dynamics tensors, each shape ``(B, NV, NV, NV)``:
    ``(d2qdd_dq, d2qdd_dqd, d2qdd_cross, d2qdd_du)`` (the FD analogue of
    :class:`SecondOrderID`; consult ``RBDReference.fdsva_so`` for the exact
    Singh/Wensing tensor semantics)."""
    d2qdd_dq: Any
    d2qdd_dqd: Any
    d2qdd_cross: Any
    d2qdd_du: Any


# Integrator-type name -> the int code the C ABI dispatches onto IntegratorType.
_INTEGRATOR_CODES = {
    "euler": 0,
    "semi_implicit_euler": 1,
    "si_euler": 1,
    "midpoint": 2,
    "rk3": 3,
    "rk4": 4,
    "trapezoidal": 5,
}


def _integrator_code(integrator_type: str) -> int:
    try:
        return _INTEGRATOR_CODES[integrator_type.lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"unknown integrator_type {integrator_type!r}; expected one of "
            f"{sorted(set(_INTEGRATOR_CODES))}"
        )


# Pinocchio reference-frame ordering (matches RBDReference / the CUDA enum):
# LOCAL=0, WORLD=1, LOCAL_WORLD_ALIGNED=2.
_REFERENCE_FRAME_CODES = {"local": 0, "world": 1, "local_world_aligned": 2}


def _frame_args(target_jid, reference_frame):
    """Normalize the frame_jacobian[_dot] runtime frame kwargs to the C ABI's
    (int target_jid, int reference_frame), where -1 means "use the codegen
    leaf-EE / LWA default baked into the host wrapper". ``reference_frame`` may
    be an int (0/1/2) or one of LOCAL / WORLD / LOCAL_WORLD_ALIGNED."""
    tj = -1 if target_jid is None else int(target_jid)
    if reference_frame is None:
        rf = -1
    elif isinstance(reference_frame, str):
        key = reference_frame.lower()
        if key not in _REFERENCE_FRAME_CODES:
            raise ValueError(
                f"unknown reference_frame {reference_frame!r}; expected one of "
                "LOCAL / WORLD / LOCAL_WORLD_ALIGNED (or 0/1/2)"
            )
        rf = _REFERENCE_FRAME_CODES[key]
    else:
        rf = int(reference_frame)
    return tj, rf


def _resolve_frame_args(meta, target_jid, reference_frame):
    """Like :func:`_frame_args` but resolves the ``-1`` "use default" sentinels
    to concrete values for the jax/torch DIRECT-kernel FFI path.

    The numpy C-ABI goes through the ``grid::frame_jacobian`` host launcher,
    which resolves ``target_jid<0`` -> leaf-EE (codegen ``get_leaf_nodes()[0]``)
    and ``reference_frame<0`` -> ``LOCAL_WORLD_ALIGNED`` (2) before the launch.
    The FFI/torch handlers launch the kernel directly and do NOT resolve, so the
    Python wrapper must substitute the same defaults the launcher bakes in."""
    tj, rf = _frame_args(target_jid, reference_frame)
    if tj < 0:
        leaf = meta.get("leaf_jids")
        if not leaf:
            raise RuntimeError(
                "this .so predates the runtime-target API (no leaf_jids in "
                "meta.json); re-register with force_rebuild=True")
        tj = int(leaf[0])  # mirrors codegen default_tjid = get_leaf_nodes()[0]
    if rf < 0:
        rf = 2  # LOCAL_WORLD_ALIGNED default
    return tj, rf


class _MujocoView(MujocoDerivativeViewMixin, MujocoViewBase):
    """MuJoCo-native view over a :class:`RobotHandle` (``handle.mujoco``).

    MuJoCo parameter names (``qpos``/``qvel``/``qacc``/``qfrc``) with the mjx
    output convention applied PER CALL — it forwards an explicit
    ``_convention="mujoco"`` rather than mutating the handle's shared
    ``output_convention``, so it is thread-safe alongside pinocchio-convention
    calls. On a fixed base the convention is a no-op (no free-flyer).

    A4 roster unification (2026-09-09): the full shared surface — values,
    centroidal/energy (now on MujocoViewBase), and the derivative / kinematics
    / integrator / plant methods (MujocoDerivativeViewMixin) — matching the
    jax/torch views method-for-method. Derivative methods need a .so built
    with the mjx kernel twins on a floating base (clear error otherwise)."""

    __slots__ = ()



class RobotHandle:
    """Opaque handle to a compiled per-robot GRiD library.

    Created by `grid_rbd.register_robot(...)` and `grid_rbd.get_robot(...)`.
    Don't construct directly; the constructor wires up the pybind11 Runner
    plus the metadata loaded from the cache's meta.json.

    Precision: float32 by default — methods cast inputs to ``float32`` and
    compute in single precision. A **true fp64 tier** (Phase 8) is available by
    registering with ``dtype="float64"``: the handle then drives a double-
    precision .so (``_core.RunnerF64``), casts inputs to ``float64``, and returns
    ``float64`` arrays computed end-to-end in double precision (``handle.dtype ==
    "float64"``). The legacy ``allow_fp64=True`` is only an fp32-compute upcast
    convenience (compute in fp32, cast i/o to fp64, single-precision accuracy
    caveat); it is off by default and ignored for a true-fp64 handle. The
    jax/torch handles follow the .so dtype too (``dtype="float64"`` builds carry
    fp64 jax/torch surfaces; jax additionally needs x64 mode).

    Method index (all take/return ``(B, …)`` arrays, batch axis first)::

        dynamics    inverse_dynamics (rnea) · forward_dynamics (fd) · aba ·
                    crba · minv · generalized_gravity · nonlinear_effects
        gradients   inverse_dynamics_gradient · forward_dynamics_gradient
        2nd-order   idsva_so → SecondOrderID · fdsva_so → SecondOrderFD
        kinematics  end_effector_pose[_gradient|_hessian] · fk_batched ·
                    frame_jacobian[_dot] · com · ccrba · osc_inertia ·
                    dccrba · cmm_time_variation
        energy      energy · coriolis_matrix ·
                    kinetic_energy_regressor · potential_energy_regressor
        integration integrator[_gradient]
        plant/cost  plant_step[_gradient|_hessian] · quadratic_state_cost ·
                    quadratic_input_cost · ee_pos_cost · com_cost ·
                    momentum_cost · joint_{position,velocity,torque}_barrier
        sysID/param inverse_dynamics_regressor · inertia_params ·
                    set_inertia_params (runtime-mutable inertia, no recompile)
        runtime-EE  end_effector_pose_runtime[_gradient] (arbitrary target joints)

    ``handle.mujoco`` is the MuJoCo-convention view of the differentiable methods
    (mjx free-joint frame, applied per-call; safe alongside pinocchio-convention calls).

    Short aliases: ``rnea`` → :py:meth:`inverse_dynamics`,
    ``fd`` → :py:meth:`forward_dynamics` (``aba`` / ``crba`` / ``minv`` already
    use their field-standard names).
    """

    def __init__(self, name: str, so_path: str, meta: dict[str, Any],
                 *, allow_fp64: bool = False) -> None:
        from . import _core  # pybind11 extension; built at pip install time

        self._name = name
        self._meta = dict(meta)
        # fp64 (Phase 8): a .so built with dtype="float64" has a double-precision
        # C ABI; it must be driven through RunnerF64 (which declares its buffers
        # + fn-pointers as double) and fed/returned float64 numpy arrays. fp32
        # (default / pre-Phase-8 meta lacking a dtype field) uses Runner. _dt is
        # the host-side numpy element dtype every method casts inputs to.
        self._dtype = str(meta.get("dtype", "float32"))
        if self._dtype == "float64":
            self._dt = np.float64
            self._runner = _core.RunnerF64(so_path)
        else:
            self._dt = np.float32
            self._runner = _core.Runner(so_path)
        # fp64-in / fp64-out convenience (compute stays fp32). Off by default.
        # Ignored for a true-fp64 build (outputs are already float64).
        self.allow_fp64 = bool(allow_fp64) and self._dtype != "float64"
        # Output convention: "pinocchio" (default, native) or "mujoco" (mjx parity).
        # Only affects FLOATING-base robots; a no-op (byte-identical) on fixed base.
        self._output_convention = "pinocchio"

        # Sanity-check that the .so's reported constants match meta.json.
        # A mismatch implies the cache is corrupted.
        for key, runner_val in [
            ("num_joints", self._runner.num_joints),
            ("num_vel", self._runner.num_vel),
            ("num_ees", self._runner.num_ees),
        ]:
            cached_val = meta.get(key)
            if cached_val is not None and cached_val != runner_val:
                raise RuntimeError(
                    f"Cache inconsistency: meta.json says {key}={cached_val} "
                    f"but the .so reports {runner_val}. Re-register with "
                    f"force_rebuild=True."
                )

    # ─── metadata ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_joints(self) -> int:
        return self._runner.num_joints

    @property
    def num_vel(self) -> int:
        return self._runner.num_vel

    @property
    def num_ees(self) -> int:
        return self._runner.num_ees

    @property
    def dtype(self) -> str:
        """Compute precision of this handle's .so: ``"float32"`` (default) or
        ``"float64"`` (Phase 8 true fp64 tier). Inputs are cast to / outputs are
        returned in this numpy dtype."""
        return self._dtype

    @property
    def floating_base(self) -> bool:
        return bool(self._meta.get("floating_base", False))

    @property
    def output_convention(self) -> str:
        """Output/IO convention: ``"pinocchio"`` (default, GRiD-native — xyzw quat,
        spatial-local free-joint velocity) or ``"mujoco"`` (mjx parity — wxyz quat,
        global-linear free-joint velocity). Setting ``"mujoco"`` makes the value
        methods take and return MuJoCo-convention ``q``/``qd``/``qdd``/``M``/... for
        a FLOATING base; it is a byte-identical no-op for a fixed base. See
        ``external/RBDReference/equivalents/mujoco_convention.md`` for the exact transforms.
        Applies to the VALUE methods (inverse_dynamics, forward_dynamics, aba,
        crba, minv, ...) AND to the gradient / second-order / regressor surfaces
        (served by the mjx kernel twins when the .so is built with them)."""
        return self._output_convention

    @output_convention.setter
    def output_convention(self, value: str) -> None:
        if value not in ("pinocchio", "mujoco"):
            raise ValueError(
                f"output_convention must be 'pinocchio' or 'mujoco', got {value!r}")
        if (value == "mujoco" and self.floating_base
                and not getattr(self._runner, "has_inverse_dynamics_mujoco", False)):
            raise ValueError(
                "output_convention='mujoco' needs a floating-base .so built with "
                "the mjx kernel twins (this one was built with "
                "enable_mujoco_kernels=False, or the robot is mimic/skew) — "
                "re-register with enable_mujoco_kernels=True.")
        self._output_convention = value

    @property
    def mujoco(self) -> "_MujocoView":
        """MuJoCo-native view (``handle.mujoco.inverse_dynamics(qpos, qvel, qacc)``):
        the supported value methods with MuJoCo parameter names and the mjx output
        convention applied PER CALL — thread-safe and independent of the handle's
        ``output_convention`` default (it never mutates shared state). Cached."""
        view = getattr(self, "_mujoco_view", None)
        if view is None:
            view = _MujocoView(self)
            self._mujoco_view = view
        return view

    def _resolve_convention(self, convention=None) -> str:
        """Per-call output convention: an explicit ``convention`` override (used by
        the thread-safe :py:attr:`mujoco` view) wins over the handle's mutable
        ``output_convention`` default. Validated."""
        conv = self._output_convention if convention is None else convention
        if conv not in ("pinocchio", "mujoco"):
            raise ValueError(f"convention must be 'pinocchio' or 'mujoco', got {conv!r}")
        return conv

    def _mjx_active(self, convention=None) -> bool:
        """True when mjx-convention transforms should actually run (mujoco AND a
        floating base — fixed base has no free-flyer, so the flag is a no-op).
        ``convention`` is an optional per-call override (thread-safe; the
        :py:attr:`mujoco` view passes it instead of mutating shared state)."""
        return self._resolve_convention(convention) == "mujoco" and self.floating_base

    _MJX_TWINS_ADVICE = (
        "needs a floating-base .so built with the mjx kernel twins (this .so "
        "was built with enable_mujoco_kernels=False, or the robot is mimic/skew "
        "— twins are never emitted there) — re-register with "
        "enable_mujoco_kernels=True.")

    def _shape_out(self, key: str, raw, *, mjx: bool = False):
        """Apply the spec's out_layout (A3 slice 4): the ONE shared transform
        implementation (bindings/grid_rbd/_out_transform.py) replacing the
        per-method hand reshape/transpose chains. ``raw`` may arrive in the
        _core allocation shape — it is flattened per batch item first."""
        from grid_codegen.abi_specs import ABI_SPECS, py_dim_tokens
        from ._out_transform import apply_out_layout, resolve_dims
        spec = ABI_SPECS[key]
        nv = self.num_vel
        raw = np.asarray(raw)
        raw = raw.reshape(raw.shape[0], -1)
        toks = py_dim_tokens(self.num_joints, nv, self.num_ees,
                             int(getattr(self._runner, "num_bodies", 0) or 0))
        eye = np.eye(nv, dtype=raw.dtype)
        return apply_out_layout(raw, spec.out_layout,
                                resolve_dims(spec.out_layout, toks),
                                nv=nv, mjx=mjx, eye=eye)

    def _require_mjx_twin(self, key: str, method: str | None = None) -> None:
        """Raise the standard actionable error unless the mjx twin symbol for
        ``key`` is present in the dlopen'd .so (A4 probe dedup, 2026-09-09 —
        one message, one probe idiom, instead of 15 hand copies)."""
        if not getattr(self._runner, f"has_{key}_mujoco", False):
            raise NotImplementedError(
                f"{method or key}(output_convention='mujoco') "
                + self._MJX_TWINS_ADVICE)

    # ─── runtime-mutable inertia (D.4 / Phase 5) ─────────────────────────────

    @property
    def runtime_inertia(self) -> bool:
        """True if this robot was registered with ``runtime_inertia=True`` (the
        .so carries a mutable inertia table + :py:meth:`set_inertia_params`)."""
        return bool(self._meta.get("runtime_inertia", False))

    @property
    def meta(self):
        """A COPY of the .so's persisted metadata (``meta.json``): sizes, joint
        names/limits, build facts (``algorithm_list`` / ``generated_algorithms``
        / ``dtype`` / ``max_batch`` / ``cuda_arch`` / ``enable_mujoco_kernels``),
        and the launch-config robot key. Mutating the returned dict never
        affects the handle. Keys added over time; a .so cached by an older
        grid-rbd simply lacks the newer ones."""
        import copy
        return copy.deepcopy(self._meta)

    @property
    def joint_names(self):
        """Joint names, index == joint id (the input-vector ordering). ``None``
        on a .so registered before names were persisted."""
        names = self._meta.get("joint_names")
        return list(names) if names else None

    def capabilities(self):
        """What this .so can do, per python-surface method key:
        ``{key: {"built", "mjx", "out_shape", "tier", "threads", "max_threads"}}``.

        - ``built``: the algorithm was in the build's post-dep-expansion emit
          set (``None`` on a .so cached before ``generated_algorithms`` was
          persisted — re-register with ``force_rebuild=True`` to populate).
          Best-effort: a robot-CLASS-limited surface (centroidal family on a
          mimic robot, ``fk_batched`` on floating/mimic) can still raise its
          clean rc=3 error at call time even when requested at build time.
        - ``mjx``: the MuJoCo-convention twin symbol is REALLY present in the
          dlopen'd .so (ground truth), for methods that have a twin.
        - ``out_shape``: trailing per-batch-item output dims (ints).
        - ``tier``/``threads``: the tuned launch config baked for this robot/
          GPU (``None`` without a config/launch_configs entry).
        - ``max_threads``: the compiled kernel's real ``__launch_bounds__``
          ceiling via :py:meth:`kernel_max_threads` (``None`` where the probe
          does not apply).
        - ``batch_threshold``/``threads_small``: the ARMED E6 batch-switch state
          (see :py:meth:`apply_batch_overlay`) — calls with batch <=
          ``batch_threshold`` launch at ``threads_small`` instead of
          ``threads``. Both ``None`` when the switch is not armed for the algo.
        """
        from grid_codegen.abi_specs import ABI_SPECS, expand_py_out_dims
        gen = self._meta.get("generated_algorithms")
        gen_set = set(gen) if gen is not None else None
        # spec key -> the algorithm_list key whose membership gates it (default:
        # the spec key itself).
        algo_key_of = {
            "idsva_so": "idsva_so_body_frame",
            "energy": "kinetic_energy_regressor",
            "fk_batched": "end_effector_pose",
            # the plant surface builds as ONE algorithm_list entry
            "plant_step": "plant",
            "plant_step_gradient": "plant",
            "plant_step_hessian": "plant",
        }
        num_bodies = int(getattr(self._runner, "num_bodies", 0) or 0)
        # tuned launch config (best-effort)
        baked = {}
        try:
            from grid_codegen.GRiDCodeGenerator import load_launch_config, LAUNCH_CONFIG_DEFAULT_GPU
            from grid_codegen.algo_registry import build_launch_config_algo_to_symbol
            robot_key = self._meta.get("launch_config_robot")
            if robot_key:
                gpu = self._meta.get("launch_config_gpu", LAUNCH_CONFIG_DEFAULT_GPU)
                by_sym = load_launch_config(robot_key, self.floating_base, gpu, profile="ffi") or {}
                sym_of = build_launch_config_algo_to_symbol()
                baked = {k: by_sym.get(s) for k, s in sym_of.items() if by_sym.get(s)}
        except Exception:
            baked = {}

        def _shape(spec):
            return expand_py_out_dims(spec, self.num_joints, self.num_vel,
                                      self.num_ees, num_bodies)

        try:
            from grid_codegen.algo_registry import descriptor_for
        except Exception:
            descriptor_for = None

        # armed E6 batch-switch state, keyed by SHORT autotune key (best-effort;
        # same enum-index derivation as apply_batch_overlay).
        switch = {}
        try:
            ctx = self._overlay_context()
            if ctx is not None:
                _, _, algo_to_symbol, algo_index, _, _ = ctx
                for akey, sym in algo_to_symbol.items():
                    idx = algo_index.get(sym)
                    if idx is None:
                        continue
                    thr, n_small = self._runner.get_batch_switch(idx)
                    if thr and thr > 0 and n_small and n_small > 0:
                        switch[akey] = (int(thr), int(n_small))
        except Exception:
            switch = {}

        out = {}
        for key, spec in ABI_SPECS.items():
            if not spec.py_out_dims:
                continue
            akey = algo_key_of.get(key, key)
            built = (akey in gen_set) if gen_set is not None else None
            if spec.surface_class == "plant" and gen_set is not None and akey not in gen_set:
                built = None  # the plant layer isn't tracked in generated_algorithms
            # the ceiling probe + launch-config tables key on the SHORT autotune
            # key ("id", "fd", "idsva_so", ...), not the surface method name.
            short = key
            if descriptor_for is not None:
                try:
                    ak = descriptor_for(key).autotune_keys
                    if ak:
                        short = ak[0]
                except Exception:
                    pass
            try:
                mt = self.kernel_max_threads(short)
                max_threads = mt if mt and mt > 0 else None
            except Exception:
                max_threads = None
            cfg = baked.get(short) or {}
            sw = switch.get(short)
            out[key] = {
                "built": built,
                # the has_*_mujoco bindings are pybind PROPERTIES, not methods
                "mjx": (bool(getattr(self._runner, f"has_{key}_mujoco", False))
                        if spec.has_mjx_twin else None),
                "out_shape": _shape(spec),
                "tier": cfg.get("tier"),
                "threads": cfg.get("threads"),
                "max_threads": max_threads,
                "batch_threshold": sw[0] if sw else None,
                "threads_small": sw[1] if sw else None,
            }
        return out

    @property
    def joint_pos_limits(self):
        """Per-joint position limits ``[lower, upper]`` (index == joint id), from
        the URDF ``<limit lower= upper=>``. ``None`` for a joint with no position
        limit (fixed / continuous / floating / unspecified); an unbounded side is
        ``None``. Metadata only — not enforced by any kernel. Returns ``None`` on
        a .so registered before limits were persisted (re-register to populate)."""
        return self._meta.get("joint_pos_limits")

    @property
    def joint_vel_limits(self):
        """Per-joint velocity limits (index == joint id) from the URDF
        ``<limit velocity=>``; ``None`` where unspecified. Metadata only."""
        return self._meta.get("joint_vel_limits")

    @property
    def joint_effort_limits(self):
        """Per-joint effort (torque) limits (index == joint id) from the URDF
        ``<limit effort=>``; ``None`` where unspecified. Metadata only."""
        return self._meta.get("joint_effort_limits")

    @property
    def inertia_params(self):
        """The BAKED 10-param-per-body inertia table, shape ``(num_bodies, 10)``.

        Each row is ``[m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]`` (mass,
        first moment ``h = m*c``, then the 6 upper-triangle entries of the
        link's inertia about its frame origin) in the frozen GRiD/URDF regressor
        basis — body-indexed, bodies 1..N (the synthetic world-frame link is
        dropped, mirroring the device table layout). For a FIXED base
        ``num_bodies == num_joints``; for a FLOATING base the floating trunk is
        row 0 and ``num_bodies < num_joints (== num_pos)``. Fetch this, mutate
        it, and pass it to
        :py:meth:`set_inertia_params`. Only available on a ``runtime_inertia``
        build (raises otherwise; the values aren't persisted for a baked .so).
        """
        params = self._meta.get("inertia_params")
        if params is None:
            raise RuntimeError(
                "inertia_params is only available on a robot registered with "
                "runtime_inertia=True. Re-register with "
                "register_robot(..., runtime_inertia=True, force_rebuild=True).")
        return np.asarray(params, dtype=self._dt)

    def set_inertia_params(self, params) -> None:
        """Update the device-resident inertia table at runtime (no recompile).

        ``params`` is the 10-param-per-body table — either flat
        ``(10*num_bodies,)`` or ``(num_bodies, 10)`` — in the same layout /
        basis as :py:attr:`inertia_params` (bodies 1..N, each ``[m, h(3),
        I_O(6)]``). All subsequent algorithm calls (inverse_dynamics, crba, …)
        reconstruct the per-link spatial inertia from the updated table. The
        sysID / domain-randomization / payload entry point.

        The table is body-indexed by ``num_bodies`` (the inertia-body count, ==
        :py:attr:`inertia_params` rows), NOT ``num_joints``: for a FIXED base
        they coincide, but for a FLOATING base (or a mimic robot)
        ``num_joints == num_pos > num_bodies`` and the device table is
        ``10*num_bodies`` long.

        Only valid on a robot registered with ``runtime_inertia=True``; raises a
        clear error otherwise. Passing the baked :py:attr:`inertia_params` back
        reproduces the baked result.
        """
        if not self.runtime_inertia:
            raise RuntimeError(
                "set_inertia_params requires a robot registered with "
                "runtime_inertia=True. Re-register with "
                "register_robot(..., runtime_inertia=True, force_rebuild=True).")
        nb = self.num_bodies
        arr = np.ascontiguousarray(params, dtype=self._dt)
        if arr.shape == (nb, 10):
            arr = arr.reshape(-1)
        elif arr.shape != (10 * nb,):
            raise ValueError(
                f"params must be ({nb}, 10) or ({10 * nb},) = 10*num_bodies "
                f"(bodies 1..N, [m, h(3), I_O(6)] each); got shape {arr.shape}.")
        arr = np.ascontiguousarray(arr, dtype=self._dt)
        self._runner.set_inertia_params(arr)

    # ─── welded tool / payload (attach_tool) ─────────────────────────────────
    def attach_tool(self, joint, *, mass, com=(0.0, 0.0, 0.0), inertia=None,
                    tip_transform=None):
        """Weld a rigid tool/payload to the link moved by ``joint`` at runtime.

        This is the "the robot now has a tool" entry point, and it needs no
        recompile. It does two things:

        1. **Inertia** — composes the payload's spatial inertia (``mass`` [kg],
           ``com`` [m, in the link frame], ``inertia`` [3x3 or 6-vector about the
           payload CoM]) into that link's spatial inertia and pokes it via
           :py:meth:`set_inertia_params`. The whole dynamics stack (inverse/forward
           dynamics + gradients) then sees the composite rigid body.
        2. **Tip frame** — if ``tip_transform`` (a 4x4 SE(3) matrix in the joint
           frame) is given, stores it so subsequent :py:meth:`end_effector_pose_runtime`
           / :py:meth:`end_effector_pose_gradient_runtime` calls (with no explicit
           target) report the SE(3) tool-tip frame. Omit it for a pure payload.

        ``joint`` is a joint NAME; the tool welds to that joint's child link, and
        the tip frame hangs off that joint's frame — so a tool can be attached
        ANYWHERE in the chain, not just a leaf. Requires ``runtime_inertia=True``
        (use ``register_robot(..., enable_tool=True)``). One tool at a time;
        :py:meth:`detach_tool` restores the baked robot.
        """
        if not self.runtime_inertia:
            raise RuntimeError(
                "attach_tool requires a robot registered with enable_tool=True "
                "(or runtime_inertia=True). Re-register with "
                "register_robot(..., enable_tool=True, force_rebuild=True).")
        j2row = self._meta.get("inertia_row_by_joint_name") or {}
        if joint not in j2row:
            raise ValueError(
                f"attach joint '{joint}' has no movable child link. Known attach "
                f"joints: {sorted(j2row.keys())}")
        row = j2row[joint]
        from ._payload import compose_payload_inertia
        baked = np.asarray(self.inertia_params, dtype=self._dt)  # (num_bodies, 10) BAKED
        tbl = np.array(baked, dtype=self._dt, copy=True)
        tbl[row] = compose_payload_inertia(baked[row], mass, com, inertia)
        self.set_inertia_params(tbl)
        Xtool = None
        if tip_transform is not None:
            Xtool = np.asarray(tip_transform, dtype=np.float64)
            if Xtool.shape != (4, 4):
                raise ValueError("tip_transform must be a 4x4 SE(3) matrix.")
        self._tool = {"joint": joint, "row": int(row), "Xtool": Xtool}
        return self._tool

    def detach_tool(self):
        """Remove the attached tool: restore the baked inertia for its link and
        clear the stored tip frame. A no-op if nothing is attached."""
        if getattr(self, "_tool", None) is None:
            return
        # inertia_params is the immutable BAKED table, so re-poking it restores the
        # single-tool link exactly (composition was baked + payload).
        self.set_inertia_params(np.asarray(self.inertia_params, dtype=self._dt))
        self._tool = None

    @property
    def tool(self):
        """The currently attached tool dict ``{joint, row, Xtool}`` or ``None``."""
        return getattr(self, "_tool", None)

    def tool_fext(self, q, wrench, *, joint=None, offset=None):
        """Map a world-aligned tool-tip wrench to a joint-local ``f_ext`` array.

        ``wrench`` is ``(B, 6)`` = ``[n_w; f_w]`` (moment about the tool tip; world axes)
        per timestep. Returns ``(B, 6*num_bodies)`` joint-local f_ext ([angular;linear]
        per body) that feeds straight into :py:meth:`inverse_dynamics` /
        :py:meth:`forward_dynamics` / :py:meth:`aba` as ``f_ext=...`` — i.e. the effect
        of the tool pushing on the world (grinding, pushing, a second gripper finger).

        With a tool attached (via :py:meth:`attach_tool`) the contact body + tip offset
        default to that tool; otherwise pass ``joint`` (a joint name) and ``offset``
        (the 3-vector tip point in the joint frame). Needs an ``enable_tool`` .so."""
        if not getattr(self._runner, "has_tool_fext", False):
            raise NotImplementedError(
                "tool_fext needs an enable_tool .so (re-register with "
                "register_robot(..., enable_tool=True, force_rebuild=True)).")
        tool = getattr(self, "_tool", None)
        if joint is None:
            if tool is None:
                raise ValueError("tool_fext: no tool attached; pass joint= and offset=.")
            joint = tool["joint"]
            if offset is None:
                X = tool.get("Xtool")
                offset = (X[:3, 3] if X is not None else np.zeros(3))
        if offset is None:
            offset = np.zeros(3)
        jid = int(self._resolve_ee_jids(joint)[0])
        q = np.ascontiguousarray(q, dtype=self._dt)
        w = np.ascontiguousarray(wrench, dtype=self._dt)
        rc = np.ascontiguousarray(np.asarray(offset, dtype=self._dt).reshape(-1)[:3])
        raw = self._runner.tool_fext(q, w, jid, rc)     # (B, 6*num_bodies)
        return self._cast_out(raw)

    def _tool_tip_default(self, ee_joint_names, ee_offsets):
        """If a tool with a tip frame is attached and the caller gave no explicit
        target/offset, default the runtime EE query to the tool tip frame."""
        tool = getattr(self, "_tool", None)
        if (ee_joint_names is None and ee_offsets is None
                and tool is not None and tool.get("Xtool") is not None):
            return tool["joint"], [tool["Xtool"]]
        return ee_joint_names, ee_offsets

    # ─── runtime-mutable joint-frame transform (runtime_transform) ───────────

    @property
    def runtime_transform(self) -> bool:
        """True if this robot was registered with ``runtime_transform=True`` (the
        .so carries a mutable joint-origin table + :py:meth:`set_transform_params`)."""
        return bool(self._meta.get("runtime_transform", False))

    @property
    def transform_params(self):
        """The BAKED 6-param-per-joint origin table, shape ``(num_joints, 6)``.

        Each row is ``[x, y, z, roll, pitch, yaw]`` — the raw URDF ``<origin>``
        translation + rpy of joint ``i`` (joint-indexed, ALL joints 0..NB-1).
        Fetch this, mutate it, and pass it to :py:meth:`set_transform_params`.
        Only available on a ``runtime_transform`` build (raises otherwise).
        """
        params = self._meta.get("transform_params")
        if params is None:
            raise RuntimeError(
                "transform_params is only available on a robot registered with "
                "runtime_transform=True. Re-register with "
                "register_robot(..., runtime_transform=True, force_rebuild=True).")
        return np.asarray(params, dtype=self._dt)

    def set_transform_params(self, params) -> None:
        """Update the device-resident joint-origin table at runtime (no recompile).

        ``params`` is the 6-param-per-joint table — either flat
        ``(6*num_joints,)`` or ``(num_joints, 6)`` — in the same layout / basis
        as :py:attr:`transform_params` (joints 0..NB-1, each
        ``[x, y, z, roll, pitch, yaw]``). All subsequent algorithm calls rebuild
        each joint's constant ``Xfixed`` from the updated table. The kinematic
        calibration / domain-randomization entry point for joint frames.

        Only valid on a robot registered with ``runtime_transform=True``; raises a
        clear error otherwise. Passing the baked :py:attr:`transform_params` back
        reproduces the baked result.

        v1 scope: this mutates the spatial X transforms used by the DYNAMICS
        (inverse_dynamics, crba, forward_dynamics, Minv, gradients, …). The
        END-EFFECTOR / homogeneous-transform kinematics (end_effector_pose and its
        gradient/hessian) still use the BAKED origin and are NOT affected — making
        them mutable is a planned v2 follow-up.
        """
        if not self.runtime_transform:
            raise RuntimeError(
                "set_transform_params requires a robot registered with "
                "runtime_transform=True. Re-register with "
                "register_robot(..., runtime_transform=True, force_rebuild=True).")
        # The device d_transform_params table is 6*grid::NUM_JOINTS, where
        # NUM_JOINTS is the PARSER body count (get_num_joints()) — NOT the handle's
        # num_joints metadata (which is get_num_pos(), differing on a floating
        # base, e.g. go2: 13 origins vs num_pos==19). Drive the row count off the
        # persisted baked table, whose length is exactly that parser count.
        nj = len(self._meta.get("transform_params", []))
        arr = np.ascontiguousarray(params, dtype=self._dt)
        if arr.shape == (nj, 6):
            arr = arr.reshape(-1)
        elif arr.shape != (6 * nj,):
            raise ValueError(
                f"params must be ({nj}, 6) or ({6 * nj},) = 6*num_joints "
                f"([x,y,z,roll,pitch,yaw] each); got shape {arr.shape}.")
        arr = np.ascontiguousarray(arr, dtype=self._dt)
        self._runner.set_transform_params(arr)

    @property
    def runtime_joint_dynamics(self) -> bool:
        """True if registered with runtime_joint_dynamics=True (the mutable
        damping/friction table backing :py:meth:`set_joint_dynamics`)."""
        return bool(self._meta.get("runtime_joint_dynamics", False))

    @property
    def joint_damping(self):
        """Baked per-v-slot viscous damping (length nv, alpha-folded). Mutate a copy
        and pass to :py:meth:`set_joint_dynamics`. Only on a runtime_joint_dynamics
        build (raises otherwise)."""
        vals = self._meta.get("joint_damping")
        if vals is None:
            raise RuntimeError(
                "joint_damping is only available on a robot registered with "
                "runtime_joint_dynamics=True. Re-register with "
                "register_robot(..., runtime_joint_dynamics=True, force_rebuild=True).")
        return np.asarray(vals, dtype=self._dt)

    @property
    def joint_friction(self):
        """Baked per-v-slot Coulomb friction (length nv, alpha-folded). Only on a
        runtime_joint_dynamics build (raises otherwise)."""
        vals = self._meta.get("joint_friction")
        if vals is None:
            raise RuntimeError(
                "joint_friction is only available on a robot registered with "
                "runtime_joint_dynamics=True. Re-register with "
                "register_robot(..., runtime_joint_dynamics=True, force_rebuild=True).")
        return np.asarray(vals, dtype=self._dt)

    def set_joint_dynamics(self, damping=None, friction=None) -> None:
        """Update the device-resident damping/friction table at runtime (no recompile).

        Pass either/both as length-nv arrays (v-slot indexed, same basis as
        :py:attr:`joint_damping` / :py:attr:`joint_friction`). An omitted side keeps
        its current baked value. This is the sysID / domain-randomization entry point
        for joint dynamics; passing the baked values back reproduces the baked result
        bit-for-bit. Only on a runtime_joint_dynamics build (raises otherwise).
        """
        if not self.runtime_joint_dynamics:
            raise RuntimeError(
                "set_joint_dynamics requires a robot registered with "
                "runtime_joint_dynamics=True. Re-register with "
                "register_robot(..., runtime_joint_dynamics=True, force_rebuild=True).")
        nv = self.num_vel
        b = np.asarray(self.joint_damping  if damping  is None else damping,  dtype=self._dt)
        f = np.asarray(self.joint_friction if friction is None else friction, dtype=self._dt)
        if b.shape != (nv,) or f.shape != (nv,):
            raise ValueError(
                f"damping and friction must each be length nv={nv} (v-slot indexed); "
                f"got {b.shape} and {f.shape}.")
        # [damping(nv) || friction(nv)], matching grid_rbd_set_joint_dynamics_params.
        arr = np.ascontiguousarray(np.concatenate([b, f]), dtype=self._dt)
        self._runner.set_joint_dynamics_params(arr)

    @property
    def max_batch(self) -> int:
        return self._runner.max_batch

    @property
    def max_perf_level_threads(self) -> int:
        """Codegen-time thread-count hint (DOF-aware, warp-rounded).

        The default block size for kernel launches. Since v2.0 it is a
        recommendation, not an enforced floor — callers can override
        via :py:meth:`set_threads_per_block`.
        """
        return self._runner.max_perf_level_threads

    @property
    def threads_per_block(self) -> int:
        """Active global threads-per-block override.

        Returns ``-1`` when no override is set, meaning each algorithm
        launches at its own autotuned ``launch_cfg<ALGO>::THREADS`` baked
        into ``grid.cuh`` (the per-algo default that fixes the FFI
        thread-default pathology). A value ``>= 1`` is a global override
        forced via :py:meth:`set_threads_per_block` that applies to every
        algorithm. Do not assume this is a positive block size.
        """
        return self._runner.threads_per_block

    def set_threads_per_block(self, n: int) -> None:
        """Force a single global per-block thread count for all subsequent
        kernel launches issued through this handle.

        The codegen does block-cooperative compute: each block handles one
        timestep with its threads cooperating via block-stride loops.
        Batching across timesteps is grid-stride at the block level. Any
        block size ``n >= 1`` (up to the per-block max, 1024 on current
        GPUs) is valid; smaller sizes are correct but slower.

        By default (no override) each algorithm uses its own autotuned
        per-algo thread count baked into ``grid.cuh``. Passing ``n >= 1``
        overrides that for every algorithm.
        """
        self._runner.set_threads_per_block(int(n))

    def kernel_max_threads(self, algo: str) -> int:
        """Real compiled ``__launch_bounds__`` ceiling of the baked kernel for the
        short autotune key (``id``, ``fd``, ``minv``, ``id_du``, ``ee_pose``,
        ``idsva_so``, …): ``cudaFuncGetAttributes().maxThreadsPerBlock`` for
        ``grid::<algo>_kernel`` instantiated at its baked ``launch_cfg<ALGO>::TIER``.

        This is the E1 tier-contract read: the FFI autotune uses it to record a
        self-consistent ``{tier, threads}`` instead of guessing a host tier and
        clamping to it. Returns ``-1`` when the key is unknown/not-built or the .so
        predates the introspection symbol (the autotune then infers the tier from
        the swept ceiling — it never crashes on a stale .so).
        """
        return self._runner.kernel_max_threads(str(algo))

    def apply_profile_overlay(self, profile: str) -> int:
        """Overlay this profile's per-algo threads (``torch_bases`` / ``pybind_bases``)
        onto the baked .so at runtime (E6). The torch/pybind surfaces share the baked
        ffi .so but want different per-algo block sizes; when a profile's tier matches
        the baked (ffi) tier the only difference is the block size, settable here with
        no rebuild.

        SKIPS any algo whose profile tier != the baked ffi tier (a tier mismatch needs
        a profile build, not a runtime overlay) so it never launches a kernel at a tier
        it wasn't compiled for. Returns the number of algos overlaid. No-op (returns 0)
        if the robot has no ``<profile>_bases``.
        """
        ctx = self._overlay_context()
        if ctx is None:
            return 0
        doc, base, algo_to_symbol, algo_index, baked, tier_symbol = ctx
        # PROFILE-ONLY block (NOT load_launch_config — that falls back to host bases;
        # an absent <profile>_bases must mean "no overlay", i.e. keep the baked ffi
        # default, not silently apply host threads). Keyed by the short json algo key.
        prof = (doc.get(str(profile) + "_bases") or {}).get(base) or {}
        if not prof:
            return 0
        n = 0
        for key, cfg in prof.items():
            sym = algo_to_symbol.get(key)
            idx = algo_index.get(sym)
            tier_sym = tier_symbol.get(str(cfg.get("tier", "")).lower())
            threads = cfg.get("threads")
            if idx is None or tier_sym is None or not isinstance(threads, int) or threads < 1:
                continue
            if tier_sym != (baked.get(sym) or {}).get("tier"):
                continue  # tier mismatch -> needs a profile build, not an overlay
            self._runner.set_threads_for(idx, int(threads))
            n += 1
        return n

    def _overlay_context(self):
        """Shared prologue of the E6 overlay appliers (apply_profile_overlay /
        apply_batch_overlay): the robot's launch-config JSON, the descriptor-table
        GridAlgo enum index, and the deployed ffi bake. Returns
        ``(doc, base, algo_to_symbol, algo_index, baked, tier_symbol)`` or None
        when no overlay is applicable (no launch_config_robot key, no JSON, or a
        codegen/binding enum-count drift — refuse to index rather than overlay
        the wrong algo)."""
        from grid_codegen.GRiDCodeGenerator import (
            LAUNCH_CONFIG_TIER_SYMBOL,
            LAUNCH_CONFIG_DEFAULT_GPU, load_launch_config, _launch_configs_dir)
        from grid_codegen.algo_registry import build_launch_config_algo_to_symbol
        import json, os
        robot_key = self._meta.get("launch_config_robot")
        if not robot_key:
            return None
        gpu = self._meta.get("launch_config_gpu", LAUNCH_CONFIG_DEFAULT_GPU)
        floating = self.floating_base
        base = "floating" if floating else "fixed"
        path = os.path.join(_launch_configs_dir(), str(robot_key), str(gpu) + ".json")
        try:
            with open(path) as f:
                doc = json.load(f)
        except (OSError, ValueError):
            return None
        # {short json algo key -> grid symbol}, derived from the descriptor table;
        # index of each grid symbol = its position in the GridAlgo enum, emitted
        # from dict.fromkeys(algo_to_symbol.values()) — the SAME descriptor-table
        # single source of truth the C-ABI uses. Check the count vs the .so
        # before indexing.
        algo_to_symbol = build_launch_config_algo_to_symbol()
        enum_syms = list(dict.fromkeys(algo_to_symbol.values()))
        algo_index = {sym: i for i, sym in enumerate(enum_syms)}
        n_algo = self._runner.algo_count()
        if n_algo and n_algo != len(enum_syms):
            return None
        baked = load_launch_config(robot_key, floating, gpu, profile="ffi")  # {sym:{tier,threads}}
        return doc, base, algo_to_symbol, algo_index, baked, LAUNCH_CONFIG_TIER_SYMBOL

    def install_device_pool(self, alloc_fn) -> int:
        """Framework-allocator integration (2026-09-09): carve GRiD's entire
        gridData device arena out of a slab the embedding framework allocates
        (jax: an XLA-pool ``jnp`` buffer; torch: a caching-allocator tensor)
        instead of raw ``cudaMalloc`` — so framework preallocation (XLA's 75%)
        and GRiD's allocations stop fighting (the h1_2 "launch failed" class).

        ``alloc_fn(nbytes) -> (buffer, device_ptr)`` allocates ``nbytes`` on
        the CUDA device via the framework and returns the keep-alive object
        plus its raw device address. Must run BEFORE the first kernel call
        (the arena init is lazy). Honors ``GRID_WORKSPACE_TIMESTEP_SLOTS``;
        when the framework cannot fit the full-slot slab the slot count is
        halved down to 1 (mirroring the auto-fit) before giving up. Returns
        the installed slab size in bytes, or 0 when the pool stays off (the
        ``cudaMalloc`` path with its own VRAM auto-fit remains the fallback).
        The slab is held alive on the handle until close/GC."""
        import os
        import warnings
        try:
            if int(self._runner.device_pool_used()) > 0:
                # a sibling surface (jax/torch handle on the SAME .so) already
                # installed a pool and the arena carved from it — nothing to do,
                # and closing a live arena here would drop tools/runtime tables.
                return 0
        except Exception:
            pass
        env = os.environ.get("GRID_WORKSPACE_TIMESTEP_SLOTS", "")
        slots = int(env) if env.strip().isdigit() and int(env) > 0 else 0
        full = slots if slots > 0 else int(self.max_batch)
        try_slots = full
        while True:
            n = int(self._runner.device_pool_bytes(try_slots))
            if n <= 0:
                return 0
            try:
                buf, ptr = alloc_fn(n)
            except Exception:
                if try_slots <= 1:
                    warnings.warn(
                        "install_device_pool: framework allocator could not fit "
                        "even a 1-slot slab — staying on the cudaMalloc path")
                    return 0
                try_slots = max(1, try_slots // 2)
                continue
            try:
                self._runner.set_device_pool(int(ptr), n, try_slots)
            except RuntimeError:
                # The Runner constructor eagerly grid_rbd_init()s as a load
                # sanity check, so a fresh handle's arena already exists on the
                # cudaMalloc path — close it (unconfigured at this point) and
                # let the lazy re-init carve from the slab. ⚠A LATE install
                # resets attached tools / runtime parameter tables the same
                # way close() does.
                self._runner.close_arena()
                self._runner.set_device_pool(int(ptr), n, try_slots)
            self._device_pool_keepalive = buf
            return n

    def apply_batch_overlay(self, profile: str = "ffi") -> int:
        """Arm the E6 batch-switch from this profile's ``<profile>_bases_by_n``
        block: for each algo with a small-batch winner recorded there, calls with
        batch <= the threshold launch at that winner's thread count instead of the
        large-batch default. The pick is a stateless per-call threshold compare in
        the .so (no hysteresis; CUDA-graph capture sees one stable pick since a
        graph freezes its batch).

        Bucket keys are the sweep batch sizes (today just ``"16"``); the threshold
        is ``<profile>_by_n_threshold`` when present, else the geometric midpoint
        of the smallest bucket and the bake batch (16/256 -> 64). Same safety
        rules as apply_profile_overlay: enum-index derivation from the descriptor
        table, algo_count drift refusal, and SKIP on any entry whose tier differs
        from the baked ffi tier. Returns the number of algos armed; 0 if no by-n
        block exists.
        """
        ctx = self._overlay_context()
        if ctx is None:
            return 0
        doc, base, algo_to_symbol, algo_index, baked, tier_symbol = ctx
        by_n = doc.get(str(profile) + "_bases_by_n") or {}
        buckets = sorted(int(k) for k in by_n.keys() if str(k).isdigit())
        if not buckets:
            return 0
        bucket = buckets[0]  # the small-batch regime (one bucket today)
        prof = (by_n.get(str(bucket)) or {}).get(base) or {}
        if not prof:
            return 0
        thr = doc.get(str(profile) + "_by_n_threshold")
        if not isinstance(thr, int) or thr < 1:
            bake_n = doc.get("autotune_N") if isinstance(doc.get("autotune_N"), int) else 256
            thr = max(bucket, int((bucket * bake_n) ** 0.5))  # 16/256 -> 64
        n = 0
        for key, cfg in prof.items():
            sym = algo_to_symbol.get(key)
            idx = algo_index.get(sym)
            tier_sym = tier_symbol.get(str(cfg.get("tier", "")).lower())
            threads = cfg.get("threads")
            if idx is None or tier_sym is None or not isinstance(threads, int) or threads < 1:
                continue
            if tier_sym != (baked.get(sym) or {}).get("tier"):
                continue  # tier mismatch -> needs a profile build, not an overlay
            self._runner.set_threads_for_n(idx, int(thr), int(threads))
            n += 1
        return n

    # ─── algorithms ──────────────────────────────────────────────────────────
    #
    # All methods take 2D float32 arrays of shape (B, num_joints) for the q/qd/qdd
    # /u inputs (GRiD's kernels consume q, qd, qdd and u all at the NUM_JOINTS
    # (== num_pos == nq) stride — for a floating base the 6-dof base velocity
    # occupies the first slots and the +1 quaternion offset is a padded slot).
    # VALUE vector outputs (torque c, qdd) are likewise (B, num_joints).
    # MATRIX / Jacobian outputs are tangent-space (pinocchio convention) and are
    # nv-dimensioned: crba/minv -> (B, num_vel, num_vel); the dynamics gradients
    # -> (B, num_vel, 2*num_vel). For a FIXED base num_vel == num_joints so all
    # shapes coincide; for a FLOATING base num_vel < num_joints.

    @property
    def num_bodies(self) -> int:
        """Number of bodies/links (incl. the base for floating-base). The
        external-force array ``f_ext`` is shaped ``(B, 6*num_bodies)``."""
        return self._runner.num_bodies

    def _prep_f_ext(self, f_ext):
        """Validate + coerce the optional external-force argument.

        ``f_ext`` is ``(B, 6*num_bodies)`` float32, body-major, each per-body
        wrench ordered ``[angular(3); linear(3)]`` in that link's LOCAL frame.
        This matches ``RBDReference.apply_external_forces`` (which subtracts the
        local wrench from the per-body force, ``f[:, i] -= f_ext[i]``) and the
        GATO/CUDA ``f -= f_ext`` convention. Returns None (no-op) if f_ext is
        None, keeping the no-f_ext path identical to before.
        """
        if f_ext is None:
            return None
        fe = np.ascontiguousarray(f_ext, dtype=self._dt)
        nb = self.num_bodies
        if fe.ndim != 2 or fe.shape[1] != 6 * nb:
            raise ValueError(
                f"f_ext must be (batch, 6*num_bodies) = (batch, {6 * nb}); "
                f"got shape {fe.shape}. Layout is body-major, each body a "
                f"length-6 [angular; linear] wrench in the body's local frame."
            )
        return fe

    def _check_nq_width(self, arr, kind: str):
        """Guard floating-base velocity/force/acceleration inputs against the
        ``nq`` vs ``nv`` footgun (Friction 3).

        GRiD consumes ``q``/``qd``/``qdd``/``u`` all at the ``NUM_JOINTS == nq``
        (== num_pos) stride — for a FLOATING base the +1 quaternion slot is a
        padded position; the 6-DOF base velocity occupies the FIRST slots and the
        velocity vector is still passed at the ``nq`` width (last slot a pad).
        Matrix/gradient OUTPUTS are ``nv``-wide, so a caller porting from
        mjx/pinocchio naturally reaches for an ``nv``-wide ``qd``/``u`` — which
        ``_core`` silently accepts (it only checks last-dim ``== num_joints``)
        and reads into the quaternion pad, returning subtly WRONG dynamics with
        no error.

        This raises a precise ``ValueError`` in exactly that case. It is a no-op
        on a FIXED base (``nq == nv`` ⇒ the widths coincide, nothing to confuse)
        and a no-op for any already-``nq``-wide input. v1 deliberately does NOT
        auto-pad — an explicit error is safer than guessing the base-velocity
        layout. ``arr`` must already be a numpy array (post-``asarray``); ``kind``
        names the offending argument for the message (e.g. ``"qd"``).
        """
        nj = self.num_joints
        nv = self.num_vel
        # Only a floating base can confuse the two widths (fixed base: nj == nv).
        if nj == nv:
            return
        last = arr.shape[-1] if arr.ndim else 0
        if last == nv:
            raise ValueError(
                f"{kind} last dim is {last}, which equals num_vel (nv={nv}); "
                f"but this is a FLOATING-base robot and GRiD expects {kind} at "
                f"the num_joints (nq={nj}) stride — the base DOFs occupy the "
                f"leading slots and the +1 quaternion slot is a pad. Pass an "
                f"nq-wide ({nj}) {kind} (velocity/force in the first nv={nv} "
                f"slots, the trailing slot a 0 pad), not an nv-wide one. "
                f"(Matrix/gradient OUTPUTS are nv-wide; the kinematic INPUTS are "
                f"nq-wide.)"
            )

    def _cast_out(self, *arrays):
        """fp64-out convenience: upcast results to float64 when ``allow_fp64``
        is set (compute already ran in fp32; this is a pure host-side cast with
        the obvious precision caveat). A no-op otherwise. Returns a single array
        for one input, else a tuple — mirroring the wrapped method's return."""
        if not self.allow_fp64:
            return arrays[0] if len(arrays) == 1 else arrays
        out = tuple(np.asarray(a, dtype=np.float64) for a in arrays)
        return out[0] if len(out) == 1 else out

    # ─── mjx (MuJoCo) output-convention transforms (floating base only) ──────
    #
    # When ``output_convention="mujoco"`` the value methods accept and return
    # MuJoCo-convention quantities. Inputs are converted mjx->pin before the
    # kernel, outputs pin->mjx after. The transforms touch only the free-flyer
    # block (quat reorder + the G=blockdiag(R,I) root basis change + the omega x v
    # acceleration term); internal joints are untouched. See `_mujoco.py` and
    # `external/RBDReference/equivalents/mujoco_convention.md`. Velocity-space inputs are
    # nq-wide (tangent in the first NV slots), so the slice-based transforms apply
    # unchanged. Done in float64 then cast back to the handle dtype.

    def _mjx_inputs(self, q, qd=None, qdd=None, u=None):
        from . import _mujoco
        q_pin = _mujoco.q_mjx_to_pin(np.asarray(q, dtype=np.float64), True)
        R = _mujoco.base_rotation(q_pin)
        qd_pin = None if qd is None else _mujoco.v_mjx_to_pin(np.asarray(qd, np.float64), R, True)
        qdd_pin = None if qdd is None else _mujoco.accel_mjx_to_pin(np.asarray(qdd, np.float64), qd_pin, R, True)
        u_pin = None if u is None else _mujoco.force_mjx_to_pin(np.asarray(u, np.float64), R, True)
        return q_pin, qd_pin, qdd_pin, u_pin, R

    def inverse_dynamics(self, q, qd, qdd=None, *, gravity: float = -9.81, f_ext=None, _convention=None) -> np.ndarray:
        """Inverse dynamics (RNEA): τ = M(q)·qdd + h(q,qd) − g(q). Returns ``(B, NJ)``.

        With ``qdd=None`` (default) this is the **bias** c = h(q,qd) − g(q)
        (= ``RBDReference.inverse_dynamics(q, qd, qdd=0)``). Pass a nonzero
        ``qdd`` to get the full RNEA torque including the inertial term M·qdd
        — the acceleration is now plumbed through (USE_QDD_FLAG=true).

        ``f_ext`` (optional): per-body external forces, shape
        ``(B, 6*num_bodies)``, body-major, each ``[angular; linear]`` in the
        body's local frame (subtracted from the per-body force, matching
        ``RBDReference.inverse_dynamics(..., f_ext=...)``). Default None ⇒ no external force.

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd``/``qdd`` are
        MuJoCo-convention and the returned ``τ`` is in the mjx frame.
        """
        # mjx (MuJoCo output convention): prefer the NATIVE mjx kernel when available.
        # It bakes the convention transform into the kernel (raw mjx inputs in, mjx
        # tau out) — no host-side input convert or tau rotation. Requires an explicit
        # qdd (the qdd=0 bias path is nonlinear_effects) and currently no f_ext (the
        # kernel input-convert doesn't reframe external wrenches yet); otherwise fall
        # back to the validated pin-kernel + _mujoco.py post-process path below.
        if (self._mjx_active(_convention) and qdd is not None and f_ext is None
                and getattr(self._runner, "has_inverse_dynamics_mujoco", False)):
            q   = np.ascontiguousarray(q,   dtype=self._dt)
            qd  = np.ascontiguousarray(qd,  dtype=self._dt)
            qdd_arr = np.ascontiguousarray(qdd, dtype=self._dt)
            c = self._runner.inverse_dynamics_mujoco(q, qd, qdd_arr, gravity, None)
            return self._cast_out(c)

        R = None
        if self._mjx_active(_convention):
            q, qd, qdd, _, R = self._mjx_inputs(q, qd, qdd)
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        self._check_nq_width(qd, "qd")
        qdd_arr = None
        if qdd is not None:
            qdd_arr = np.ascontiguousarray(qdd, dtype=self._dt)
            self._check_nq_width(qdd_arr, "qdd")
        c = self._runner.inverse_dynamics(q, qd, qdd_arr, gravity, self._prep_f_ext(f_ext))
        if R is not None:
            from . import _mujoco
            c = _mujoco.id_tau_pin_to_mjx(np.asarray(c, np.float64), R, True).astype(self._dt)
        return self._cast_out(c)

    def minv(self, q, *, _convention=None) -> np.ndarray:
        """Direct mass-matrix inverse Minv(q). Returns shape (B, NV, NV).

        Minv is the tangent-space (pinocchio-convention) inverse mass matrix:
        ``NV x NV``. For a FIXED base ``NV == NJ`` (== num_pos) so the shape is
        unchanged; for a FLOATING base ``NV = 6 + n_joints < NJ = 7 + n_joints``
        (the +1 is the quaternion offset in q only). GRiD's `minv` kernel writes
        only the upper triangle (lower zero); we symmetrize on the host before
        returning so the matrix matches `RBDReference.minv(..., output_dense=True)`.
        The symmetrization is a single numpy op per call — negligible cost.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned ``Minv`` is the mjx-frame inverse mass matrix.
        """
        # mjx: native kernel only (raw mjx q in, full DENSE SYMMETRIC mjx Minv
        # out — the G^-T Minv G^-1 congruence is baked into the kernel; no host
        # symmetrize or post-process).
        if self._mjx_active(_convention):
            self._require_mjx_twin("minv", "minv")
            q = np.ascontiguousarray(q, dtype=self._dt)
            return self._cast_out(self._runner.minv_mujoco(q))

        q = np.ascontiguousarray(q, dtype=self._dt)
        # A3 slice 4: the UPPER-triangle symmetrize (M + Mᵀ − diag) now lives in
        # the shared out-layout module.
        return self._cast_out(self._shape_out("minv", self._runner.minv(q)))

    def forward_dynamics(self, q, qd, u, *, gravity: float = -9.81, f_ext=None, _convention=None) -> np.ndarray:
        """Forward dynamics qdd = M⁻¹·(τ − c). Returns shape (B, NJ).

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``,
        body-major, ``[angular; linear]`` local-frame (see :py:meth:`inverse_dynamics`).
        With ``output_convention="mujoco"`` (floating base) ``q``/``qd``/``u`` are
        MuJoCo-convention and the returned ``qdd`` is in the mjx frame."""
        # mjx: prefer the native kernel (raw mjx in, mjx qdd out — accel_out baked in);
        # fall back to the validated host path. f_ext isn't reframed by the kernel.
        if (self._mjx_active(_convention) and f_ext is None
                and getattr(self._runner, "has_forward_dynamics_mujoco", False)):
            q  = np.ascontiguousarray(q,  dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            u  = np.ascontiguousarray(u,  dtype=self._dt)
            return self._cast_out(self._runner.forward_dynamics_mujoco(q, qd, u, gravity, None))

        R = None; qd_pin = None
        if self._mjx_active(_convention):
            q, qd_pin, _, u, R = self._mjx_inputs(q, qd, u=u); qd = qd_pin
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        u  = np.ascontiguousarray(u,  dtype=self._dt)
        self._check_nq_width(qd, "qd")
        self._check_nq_width(u, "u")
        acc = self._runner.forward_dynamics(q, qd, u, gravity, self._prep_f_ext(f_ext))
        if R is not None:
            from . import _mujoco
            acc = _mujoco.fd_qdd_pin_to_mjx(np.asarray(acc, np.float64), qd_pin, R, True).astype(self._dt)
        return self._cast_out(acc)

    def aba(self, q, qd, u, *, gravity: float = -9.81, f_ext=None, _convention=None) -> np.ndarray:
        """Recursive forward dynamics via Articulated Body Algorithm.
        Returns shape (B, NJ). Alternative to forward_dynamics() with the
        same output but a different implementation.

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``,
        body-major, ``[angular; linear]`` local-frame (see :py:meth:`inverse_dynamics`).
        With ``output_convention="mujoco"`` (floating base) the IO is mjx-convention."""
        if (self._mjx_active(_convention) and f_ext is None
                and getattr(self._runner, "has_aba_mujoco", False)):
            q  = np.ascontiguousarray(q,  dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            u  = np.ascontiguousarray(u,  dtype=self._dt)
            return self._cast_out(self._runner.aba_mujoco(q, qd, u, gravity, None))

        R = None; qd_pin = None
        if self._mjx_active(_convention):
            q, qd_pin, _, u, R = self._mjx_inputs(q, qd, u=u); qd = qd_pin
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        u  = np.ascontiguousarray(u,  dtype=self._dt)
        self._check_nq_width(qd, "qd")
        self._check_nq_width(u, "u")
        acc = self._runner.aba(q, qd, u, gravity, self._prep_f_ext(f_ext))
        if R is not None:
            from . import _mujoco
            acc = _mujoco.fd_qdd_pin_to_mjx(np.asarray(acc, np.float64), qd_pin, R, True).astype(self._dt)
        return self._cast_out(acc)

    def crba(self, q, *, gravity: float = -9.81, _convention=None) -> np.ndarray:
        """Joint-space mass matrix M(q) via Composite Rigid Body Algorithm.
        Returns shape (B, NV, NV) — the tangent-space (pinocchio-convention)
        mass matrix. FIXED base: NV == NJ (unchanged); FLOATING base: NV < NJ
        (the kernel writes NUM_VEL x NUM_VEL). Pass `gravity` only because the
        host wrapper takes it; the result doesn't depend on gravity.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned ``M`` is the mjx-frame mass matrix."""
        # mjx: native kernel only (raw mjx q in, mjx M out — the G M G^T
        # congruence is baked into the kernel).
        if self._mjx_active(_convention):
            self._require_mjx_twin("crba", "crba")
            q = np.ascontiguousarray(q, dtype=self._dt)
            return self._cast_out(self._runner.crba_mujoco(q, gravity))

        q = np.ascontiguousarray(q, dtype=self._dt)
        return self._cast_out(self._runner.crba(q, gravity))

    def end_effector_pose(self, q, *, _convention=None) -> np.ndarray:
        """End-effector pose [xyz, rpy] per EE. Returns shape (B, 6*NUM_EES).
        For multi-EE robots, reshape to (B, NUM_EES, 6) at the caller side.

        With ``output_convention="mujoco"`` (floating base) ``q`` is
        MuJoCo-convention; the pose itself is frame-INVARIANT, but the native
        kernel routes ``q`` through the mjx quaternion reorder (so the output
        equals feeding the pin kernel the pin-converted ``q``)."""
        if self._mjx_active(_convention):
            self._require_mjx_twin("end_effector_pose", "end_effector_pose")
            q = np.ascontiguousarray(q, dtype=self._dt)
            return self._runner.end_effector_pose_mujoco(q)
        q = np.ascontiguousarray(q, dtype=self._dt)
        return self._runner.end_effector_pose(q)

    def fk_batched(self, q, *, use_warp: bool = False) -> np.ndarray:
        """Large-batch forward kinematics, one block (thread variant) or warp
        (warp variant) per sample.

        Input  q:     (B, NUM_POS)  joint positions (batch-major).
        Output pose7: (B, 7) = [tx, ty, tz, qw, qx, qy, qz] for the leaf EE
        frame, where the last four are the unit quaternion (w, x, y, z).

        `use_warp=True` runs the warp-cooperative per-sample inner; both
        variants return identical poses. Fixed AND floating base supported
        (floating q = [x,y,z, qx,qy,qz,qw, joints], pin convention); mimic
        joints fold inline. Raises only for spherical-joint or >32-joint
        robots (those route through end_effector_pose)."""
        q = np.ascontiguousarray(q, dtype=self._dt)
        return self._runner.fk_batched(q, use_warp)

    def end_effector_pose_gradient(self, q, *, _convention=None) -> np.ndarray:
        """End-effector pose Jacobian d/dv (TANGENT, pinocchio convention).

        Returns shape (B, 6*NUM_EES, NV). Floating-base produces the
        spatial Jacobian (omega; v) base block, not the older non-standard
        quaternion-derivative columns. Fixed-base shape unchanged (NV == NJ).

        GRiD's `h_end_effector_pose_gradient` is stored column-major as (6, NUM_EES*NV) per
        timestep; we re-orient to (6*NUM_EES, NV) per timestep.

        With ``output_convention="mujoco"`` (floating base) ``q`` is
        MuJoCo-convention and the returned Jacobian has its base-linear columns
        reframed into the mjx frame (computed natively in the kernel).
        """
        NEE = self.num_ees
        NV = self.num_vel
        if self._mjx_active(_convention):
            self._require_mjx_twin("end_effector_pose_gradient", "end_effector_pose_gradient")
            q = np.ascontiguousarray(q, dtype=self._dt)
            raw = self._runner.end_effector_pose_gradient_mujoco(q)
            return self._shape_out("end_effector_pose_gradient", raw)
        q = np.ascontiguousarray(q, dtype=self._dt)
        raw = self._runner.end_effector_pose_gradient(q)
        return self._shape_out("end_effector_pose_gradient", raw)

    def inverse_dynamics_gradient(self, q, qd, qdd=None, *, gravity: float = -9.81, f_ext=None, _convention=None) -> np.ndarray:
        """∂τ/∂(q, qd). Returns shape (B, NV, 2*NV) — concatenated
        [dc_dq | dc_dqd], tangent-space (pinocchio) convention. Slice with
        `[..., :NV]` / `[..., NV:]`. FIXED base: NV == NJ (unchanged); FLOATING
        base: NV < NJ (the kernel writes nv x 2nv).

        ``qdd`` (optional): joint acceleration. The gradient depends on it
        (through the M·qdd term); ``qdd=None`` (default) ⇒ the bias gradient at
        qdd=0. Now plumbed through (USE_QDD_FLAG=true) matching
        :py:meth:`inverse_dynamics`.

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``.
        f_ext enters RNEA affinely, so for a CONSTANT f_ext the Jacobian
        ∂c/∂(q,qd) is unchanged; the kwarg is for consistency with inverse_dynamics().

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd``/``qdd`` are
        MuJoCo-convention and the returned gradient is in the mjx frame (the full
        convention transform — reframe + base-row rotate + ω×v couplings — is baked
        into the kernel). Requires an explicit ``qdd`` (and no ``f_ext``)."""
        NV = self.num_vel
        if (self._mjx_active(_convention) and qdd is not None and f_ext is None
                and getattr(self._runner, "has_inverse_dynamics_gradient_mujoco", False)):
            q   = np.ascontiguousarray(q,   dtype=self._dt)
            qd  = np.ascontiguousarray(qd,  dtype=self._dt)
            qdd_arr = np.ascontiguousarray(qdd, dtype=self._dt)
            raw = self._runner.inverse_dynamics_gradient_mujoco(q, qd, qdd_arr, gravity, None)
            return self._shape_out("inverse_dynamics_gradient", raw)
        if self._mjx_active(_convention):
            raise NotImplementedError(
                "inverse_dynamics_gradient(output_convention='mujoco') needs an explicit "
                "qdd, no f_ext, and a floating-base .so built with the mjx kernel twins (this .so was built with enable_mujoco_kernels=False, or the robot is mimic/skew — twins are never emitted there) — re-register with enable_mujoco_kernels=True.")
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        self._check_nq_width(qd, "qd")
        qdd_arr = None
        if qdd is not None:
            qdd_arr = np.ascontiguousarray(qdd, dtype=self._dt)
            self._check_nq_width(qdd_arr, "qdd")
        raw = self._runner.inverse_dynamics_gradient(q, qd, qdd_arr, gravity, self._prep_f_ext(f_ext))
        # A3 slice 4: the two-col-major-halves -> (NV, 2NV) hstack chain lives
        # in the shared out-layout module ("grad_concat").
        return self._shape_out("inverse_dynamics_gradient", raw)

    def forward_dynamics_gradient(self, q, qd, u, *, gravity: float = -9.81, f_ext=None, _convention=None) -> np.ndarray:
        """∂qdd/∂(q, qd). Returns shape (B, NV, 2*NV), tangent-space (pinocchio)
        convention. FIXED base: NV == NJ (unchanged); FLOATING base: NV < NJ.

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``;
        affine in f_ext so a constant f_ext leaves this Jacobian unchanged.

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd``/``u`` are
        MuJoCo-convention and the returned gradient is in the mjx frame (the full
        convention transform — reframe + base-row rotate + ω×v couplings — is baked
        into the kernel). Requires no ``f_ext``."""
        NV = self.num_vel
        if (self._mjx_active(_convention) and f_ext is None
                and getattr(self._runner, "has_forward_dynamics_gradient_mujoco", False)):
            q  = np.ascontiguousarray(q,  dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            u  = np.ascontiguousarray(u,  dtype=self._dt)
            raw = self._runner.forward_dynamics_gradient_mujoco(q, qd, u, gravity, None)
            return self._shape_out("forward_dynamics_gradient", raw)
        if self._mjx_active(_convention):
            raise NotImplementedError(
                "forward_dynamics_gradient(output_convention='mujoco') needs no f_ext "
                "and a floating-base .so built with " + self._MJX_TWINS_ADVICE.split("built with ", 1)[-1])
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        u  = np.ascontiguousarray(u,  dtype=self._dt)
        self._check_nq_width(qd, "qd")
        self._check_nq_width(u, "u")
        raw = self._runner.forward_dynamics_gradient(q, qd, u, gravity, self._prep_f_ext(f_ext))
        # Same layout as dc_du ("grad_concat" in the shared out-layout module).
        return self._shape_out("forward_dynamics_gradient", raw)

    def end_effector_pose_hessian(self, q, *, _convention=None) -> np.ndarray:
        """End-effector pose Hessian ∂²(pose)/∂v² (tangent-space, pinocchio convention).
        Returns shape (B, 6*NUM_EES, NV, NV). For fixed-base NV == NJ; for
        floating-base the (NV, NV) block indexes spatial twist components.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned Hessian is the symmetric coordinate Hessian along the mjx
        retract (double column-reframe + symmetrized base-rotation frame term, baked
        in-kernel)."""
        if self._mjx_active(_convention) and getattr(self._runner, "has_end_effector_pose_hessian_mujoco", False):
            q = np.ascontiguousarray(q, dtype=self._dt)
            return self._runner.end_effector_pose_hessian_mujoco(q)
        if self._mjx_active(_convention):
            raise NotImplementedError(
                "end_effector_pose_hessian(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
        q = np.ascontiguousarray(q, dtype=self._dt)
        return self._runner.end_effector_pose_hessian(q)

    def idsva_so(self, q, qd, qdd=None, *, gravity: float = -9.81, _convention=None) -> SecondOrderID:
        """Second-order inverse dynamics. Returns a :class:`SecondOrderID`
        NamedTuple ``(d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq)``, each tensor
        shape ``(B, NV, NV, NV)``. (NamedTuple is a plain tuple — positional
        unpacking and indexing still work.)

        Uses the codegen-time dispatcher: body-frame for fixed-base,
        world-frame for floating-base.

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd``/``qdd`` are
        MuJoCo-convention and all four 2nd-order tensors are returned in the mjx frame
        (the explicit-analytic SO transform + dM_dq closed form, baked in-kernel)."""
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        # qdd is packed into the device acceleration slot; pass explicit zeros for
        # the default (qdd=None ⇒ zero acceleration) so the result never depends on
        # a stale device buffer from a previous call.
        qdd_in = qdd if qdd is not None else np.zeros_like(q)
        qdd_arr = np.ascontiguousarray(qdd_in, dtype=self._dt)
        NV = self.num_vel
        if self._mjx_active(_convention) and getattr(self._runner, "has_idsva_so_mujoco", False):
            flat = self._runner.idsva_so_mujoco(q, qd, qdd_arr, 4 * NV ** 3, gravity)
        elif self._mjx_active(_convention):
            raise NotImplementedError(
                "idsva_so(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
        else:
            self._check_nq_width(qd, "qd")
            self._check_nq_width(qdd_arr, "qdd")
            flat = self._runner.idsva_so(q, qd, qdd_arr, 4 * NV ** 3, gravity)
        # 4 NV^3 slabs ("so_slabs" in the shared out-layout module).
        return SecondOrderID(*self._cast_out(*self._shape_out("idsva_so", flat)))

    def inverse_dynamics_regressor(self, q, qd, qdd=None, *, gravity: float = -9.81, _convention=None) -> np.ndarray:
        """Inverse-dynamics inertial-parameter regressor ``Y`` with
        ``tau = Y . pi`` (``pi`` = the stacked 10-param spatial inertia of each link).
        Returns ``(B, NV, 10*NUM_BODIES)``. Mirrors
        ``RBDReference.inverse_dynamics_regressor(q, qd, qdd)``.

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd``/``qdd`` are
        MuJoCo-convention and the base-linear ROWS (0:3) of ``Y`` are rotated by ``R``
        in-kernel (the rows transform like a generalized force, ``Y_mjx = G Y_pin``).
        """
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        qdd_in = qdd if qdd is not None else np.zeros_like(q)
        qdd_arr = np.ascontiguousarray(qdd_in, dtype=self._dt)
        NV = self.num_vel
        ncol = 10 * self.num_bodies
        if self._mjx_active(_convention) and getattr(self._runner, "has_inverse_dynamics_regressor_mujoco", False):
            flat = self._runner.inverse_dynamics_regressor_mujoco(q, qd, qdd_arr, gravity)
        elif self._mjx_active(_convention):
            raise NotImplementedError(
                "inverse_dynamics_regressor(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
        else:
            self._check_nq_width(qd, "qd")
            self._check_nq_width(qdd_arr, "qdd")
            flat = self._runner.inverse_dynamics_regressor(q, qd, qdd_arr, gravity)
        B = flat.shape[0]
        return self._cast_out(flat.reshape(B, NV, ncol))  # (B, NV, 10*NUM_BODIES)

    def fdsva_so(self, q, qd, u, *, gravity: float = -9.81, _convention=None) -> SecondOrderFD:
        """Second-order forward dynamics. Returns a :class:`SecondOrderFD`
        NamedTuple of 4 tensors each shape ``(B, NV, NV, NV)`` (a plain tuple,
        so positional unpacking / indexing still work).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd``/``u`` are
        MuJoCo-convention and all four 2nd-order tensors are returned in the mjx frame
        (explicit-analytic SO transform, contravector output-map, baked in-kernel)."""
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        u  = np.ascontiguousarray(u,  dtype=self._dt)
        NV = self.num_vel
        if self._mjx_active(_convention) and getattr(self._runner, "has_fdsva_so_mujoco", False):
            # The fdsva_so mjx epilogue recomputes Minv / qdd / dqdd_du FRESH into a
            # disjoint d_mjx_scratch band (mirrors idsva_so) so it never reads the
            # possibly-spilled in-flight buffers — the §1g/§1h liveness bug is fixed.
            flat = self._runner.fdsva_so_mujoco(q, qd, u, 4 * NV ** 3, gravity)
        elif self._mjx_active(_convention):
            raise NotImplementedError(
                "fdsva_so(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
        else:
            self._check_nq_width(qd, "qd")
            self._check_nq_width(u, "u")
            flat = self._runner.fdsva_so(q, qd, u, 4 * NV ** 3, gravity)
        return SecondOrderFD(*self._cast_out(*self._shape_out("fdsva_so", flat)))

    def integrator(self, q, qd, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81, _convention=None):
        """One integration step x_{k+1} = integrator(x_k, u, dt).

        Returns shape (B, NUM_POS + NUM_VEL) — concatenated [q_new, v_new].
        `dt` is the runtime timestep; gravity is the signed gravitational acceleration (default -9.81).
        `integrator_type` is one of euler / semi_implicit_euler / midpoint /
        rk3 / rk4.

        With ``output_convention="mujoco"`` (floating base) the free-joint base
        position takes a GLOBAL additive step (the MuJoCo retract) rather than
        pinocchio's SE(3) update; ``q``/``qd`` are MuJoCo-convention and the returned
        ``q_new`` is in the mjx frame (quaternion wxyz). Baked into the kernel."""
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        u  = np.ascontiguousarray(u,  dtype=self._dt)
        it = _integrator_code(integrator_type)
        if self._mjx_active(_convention):
            self._require_mjx_twin("integrator", "integrator")
            return self._runner.integrator_mujoco(q, qd, u, float(dt), it, gravity=float(gravity))
        self._check_nq_width(qd, "qd")
        self._check_nq_width(u, "u")
        return self._runner.integrator(q, qd, u, float(dt), it, gravity=float(gravity))

    def integrator_gradient(self, q, qd, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81, _convention=None):
        """Gradient of the integrator step. Returns shape (B, 2*NV, 3*NV) —
        column blocks [d/dq | d/dqd | d/du] in tangent space.

        `dt` is the runtime timestep; gravity is the signed gravitational acceleration (default -9.81).

        With ``output_convention="mujoco"`` (floating base, EULER/SI-EULER) ``q``/``qd``/``u``
        are MuJoCo-convention and the returned state-transition Jacobian is in the mjx
        tangent (global-add retract rows + G velocity reframe + input couplings, in-kernel)."""
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        u  = np.ascontiguousarray(u,  dtype=self._dt)
        it = _integrator_code(integrator_type)
        if self._mjx_active(_convention) and getattr(self._runner, "has_integrator_gradient_mujoco", False):
            raw = self._runner.integrator_gradient_mujoco(q, qd, u, float(dt), it, gravity=float(gravity))
        elif self._mjx_active(_convention):
            raise NotImplementedError(
                "integrator_gradient(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
        else:
            self._check_nq_width(qd, "qd")
            self._check_nq_width(u, "u")
            raw = self._runner.integrator_gradient(q, qd, u, float(dt), it, gravity=float(gravity))
        # (2NV x 3NV) col-major dAB ("colmajor_whole" in the shared module).
        return self._shape_out("integrator_gradient", raw)

    # ─── grid_plant surface (cost / barrier / plant-step) ────────────────────
    #
    # Composed over the grid:: device surface (integrator + EE pose/Jacobian).
    # Validated against RBDReference._PlantMixin. All take/return 2D arrays with
    # axis 0 = batch. Cost methods return (value, grad, hess); barriers return
    # (value, grad, hess_diag). Conventions mirror RBDReference/_plant.py.

    def quadratic_state_cost(self, x, x_des, Q, *, _convention=None):
        """1/2 * sum_i Q_i (x_i - x_des_i)^2 over the full state x = [q; qd].

        x / x_des / Q are (B, NUM_POS + NUM_VEL). Returns:
          value (B,), grad (B, NX), hess = diag(Q) (B, NX, NX).

        With ``output_convention="mujoco"`` (floating base) ``x`` is MuJoCo-convention:
        the kernel input-converts the velocity base-linear block (global->local) before
        differencing against the mjx-frame ``x_des``/``Q``, so the VALUE is
        convention-DEPENDENT; the qd-block grad/hess are reframed in-kernel.
        """
        x = np.ascontiguousarray(x, dtype=self._dt)
        x_des = np.ascontiguousarray(x_des, dtype=self._dt)
        Q = np.ascontiguousarray(Q, dtype=self._dt)
        if self._mjx_active(_convention) and getattr(self._runner, "has_quadratic_state_cost_mujoco", False):
            return self._runner.quadratic_state_cost_mujoco(x, x_des, Q)
        if self._mjx_active(_convention):
            raise NotImplementedError(
                "quadratic_state_cost(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
        return self._runner.quadratic_state_cost(x, x_des, Q)

    def quadratic_input_cost(self, u, u_des, R):
        """1/2 * sum_i R_i (u_i - u_des_i)^2 over the input u (size NUM_VEL).

        u / u_des / R are (B, NUM_VEL). Returns:
          value (B,), grad (B, NV), hess = diag(R) (B, NV, NV).
        """
        u = np.ascontiguousarray(u, dtype=self._dt)
        u_des = np.ascontiguousarray(u_des, dtype=self._dt)
        R = np.ascontiguousarray(R, dtype=self._dt)
        return self._runner.quadratic_input_cost(u, u_des, R)

    def ee_pos_cost(self, q, p_des, W, *, _convention=None):
        """End-effector position cost over the 3 position axes (EE 0).

        q is (B, NUM_POS); p_des / W are (B, 3). Returns:
          value (B,), grad_x (B, NX) = [J_p^T (W·r); 0], GN hess_x (B, NX, NX)
          with the top-left NV×NV q-block = J_p^T diag(W) J_p.

        The hessian is returned in the kernel's column-major layout; since the
        GN hessian J_p^T W J_p is symmetric the row/col-major distinction is
        immaterial.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention;
        the EE position (and thus the value) is invariant, and the q-block grad/hess are
        reframed to the mjx tangent (covector / congruence) in-kernel."""
        q = np.ascontiguousarray(q, dtype=self._dt)
        p_des = np.ascontiguousarray(p_des, dtype=self._dt)
        W = np.ascontiguousarray(W, dtype=self._dt)
        if self._mjx_active(_convention) and getattr(self._runner, "has_ee_pos_cost_mujoco", False):
            return self._runner.ee_pos_cost_mujoco(q, p_des, W)
        if self._mjx_active(_convention):
            raise NotImplementedError(
                "ee_pos_cost(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
        return self._runner.ee_pos_cost(q, p_des, W)

    def joint_position_barrier(self, var, lower, upper, mu):
        """Log-barrier b = -mu·(log(x-lo)+log(hi-x)) over NUM_POS positions.

        var / lower / upper are (B, NUM_POS); an ±inf bound contributes zero.
        Returns (value (B,), grad (B, NUM_POS), hess_diag (B, NUM_POS)).
        """
        return self._barrier("joint_position_barrier", var, lower, upper, mu)

    def joint_velocity_barrier(self, var, lower, upper, mu):
        """Log-barrier over NUM_VEL velocities. See joint_position_barrier."""
        return self._barrier("joint_velocity_barrier", var, lower, upper, mu)

    def joint_torque_barrier(self, var, lower, upper, mu):
        """Log-barrier over NUM_VEL torques. See joint_position_barrier."""
        return self._barrier("joint_torque_barrier", var, lower, upper, mu)

    def _barrier(self, method, var, lower, upper, mu):
        var = np.ascontiguousarray(var, dtype=self._dt)
        lower = np.ascontiguousarray(lower, dtype=self._dt)
        upper = np.ascontiguousarray(upper, dtype=self._dt)
        return getattr(self._runner, method)(var, lower, upper, float(mu))

    def plant_step(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81, _convention=None):
        """x_{k+1} = integrator(x_k, u_k, dt). Thin wrapper over grid::integrator.

        x is (B, NUM_POS + NUM_VEL); u is (B, NUM_VEL). Returns (B, NX).
        `integrator_type` is one of euler / semi_implicit_euler / midpoint /
        rk3 / rk4 (same codes as :py:meth:`integrator`).

        With ``output_convention="mujoco"`` (floating base, EULER/SI-EULER) ``x``/``u``
        are MuJoCo-convention and the returned next state uses the mjx global-add
        retract (base position) + reordered quaternion."""
        x = np.ascontiguousarray(x, dtype=self._dt)
        u = np.ascontiguousarray(u, dtype=self._dt)
        it = _integrator_code(integrator_type)
        if self._mjx_active(_convention) and getattr(self._runner, "has_plant_step_mujoco", False):
            return self._runner.plant_step_mujoco(x, u, float(dt), it, float(gravity))
        if self._mjx_active(_convention):
            raise NotImplementedError(
                "plant_step(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
        return self._runner.plant_step(x, u, float(dt), it, float(gravity))

    def plant_step_gradient(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81, _convention=None):
        """[A | B] = d x_{k+1}/d(x,u) = the integrator-gradient s_dAB surface.

        x is (B, NUM_POS + NUM_VEL); u is (B, NUM_VEL). Returns (B, 2*NV, 3*NV)
        with column blocks [d/dq | d/dqd | d/du] in tangent space. Pass-through
        to grid::integrator_gradient (the value is byte-identical to
        :py:meth:`integrator_gradient`). Matches ``RBDReference.plant_step_gradient``
        (= ``integrator_gradient``). ``integrator_type`` is one of euler /
        semi_implicit_euler / midpoint / rk3 / rk4.
        """
        x = np.ascontiguousarray(x, dtype=self._dt)
        u = np.ascontiguousarray(u, dtype=self._dt)
        it = _integrator_code(integrator_type)
        if self._mjx_active(_convention) and getattr(self._runner, "has_plant_step_gradient_mujoco", False):
            raw = self._runner.plant_step_gradient_mujoco(x, u, float(dt), it, float(gravity))
        elif self._mjx_active(_convention):
            raise NotImplementedError(
                "plant_step_gradient(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
        else:
            raw = self._runner.plant_step_gradient(x, u, float(dt), it, float(gravity))
        # (2*NV x 3*NV) column-major dAB → shared out-layout (colmajor_whole).
        return self._shape_out("plant_step_gradient", raw)

    def plant_step_hessian(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81, _convention=None):
        """Second-order sensitivity of the integrator step x_{k+1} = [q; v].

        x is (B, NUM_POS + NUM_VEL); u is (B, NUM_VEL). Returns the s_d2AB
        surface of shape (B, 2*NV, 3*NV, 3*NV):

            H[b, o, a, b'] = d^2 x_{k+1}[o] / dz[a] dz[b'],  z = [dq; dqd; du]

        with output rows split as [position-tangent (NV); velocity (NV)].
        Matches ``RBDReference.plant_step_hessian``. Pass-through to
        grid::integrator_hessian_device (composes fdsva_so + dt-scaled assembly).

        Scope (first landing): euler / semi_implicit_euler on a FIXED base.
        Floating-base and multi-stage RK are deferred (the C-ABI returns rc=3 /
        raises for any other ``integrator_type``).
        """
        x = np.ascontiguousarray(x, dtype=self._dt)
        u = np.ascontiguousarray(u, dtype=self._dt)
        it = _integrator_code(integrator_type)
        if self._mjx_active(_convention) and getattr(self._runner, "has_plant_step_hessian_mujoco", False):
            raw = self._runner.plant_step_hessian_mujoco(x, u, float(dt), it, float(gravity))
        elif self._mjx_active(_convention):
            raise NotImplementedError(
                "plant_step_hessian(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
        else:
            raw = self._runner.plant_step_hessian(x, u, float(dt), it, float(gravity))
        # row-major per timestep (C-order, no transpose) → shared out-layout
        # (reshape (2*NV, 3*NV, 3*NV)).
        return self._shape_out("plant_step_hessian", raw)

    def com_cost(self, q, p_des, W, *, _convention=None):
        """Center-of-mass tracking cost over the 3 CoM axes.

        q is (B, NUM_POS); p_des / W are (B, 3). Returns:
          value (B,), grad_x (B, NX) = [J_com^T (W·r); 0], GN hess_x (B, NX, NX)
          with the top-left NV×NV q-block = J_com^T diag(W) J_com.

        Matches ``RBDReference.com_cost(q, p_des, W)``.

        With ``output_convention="mujoco"`` (floating base) the CoM position (and value)
        is invariant; the q-block grad/hess are reframed to the mjx tangent in-kernel."""
        q = np.ascontiguousarray(q, dtype=self._dt)
        p_des = np.ascontiguousarray(p_des, dtype=self._dt)
        W = np.ascontiguousarray(W, dtype=self._dt)
        if self._mjx_active(_convention) and getattr(self._runner, "has_com_cost_mujoco", False):
            return self._runner.com_cost_mujoco(q, p_des, W)
        if self._mjx_active(_convention):
            raise NotImplementedError(
                "com_cost(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
        return self._runner.com_cost(q, p_des, W)

    def momentum_cost(self, q, qd, h_des, W, *, _convention=None):
        """Centroidal-momentum tracking cost over the 6 momentum components.

        q is (B, NUM_POS); qd is (B, NUM_VEL); h_des / W are (B, 6). Returns:
          value (B,), grad_x (B, NX) = [0; A^T (W·r)], GN hess_x (B, NX, NX)
          with the bottom-right NV×NV qd-block = A^T diag(W) A.

        Matches ``RBDReference.momentum_cost(q, qd, h_des, W)``.

        With ``output_convention="mujoco"`` (floating base) the centroidal momentum h
        (and value) is invariant; the qd-block grad/hess are reframed in-kernel."""
        q = np.ascontiguousarray(q, dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        h_des = np.ascontiguousarray(h_des, dtype=self._dt)
        W = np.ascontiguousarray(W, dtype=self._dt)
        if self._mjx_active(_convention) and getattr(self._runner, "has_momentum_cost_mujoco", False):
            return self._runner.momentum_cost_mujoco(q, qd, h_des, W)
        if self._mjx_active(_convention):
            raise NotImplementedError(
                "momentum_cost(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
        return self._runner.momentum_cost(q, qd, h_des, W)

    # ─── centroidal / energy / general-frame kinematics (F2) ─────────────────
    #
    # Convenience compositions over the grid:: kinematics/dynamics surface,
    # validated against RBDReference's centroidal / energy / frame mixins. All
    # take 2D float32 (B, NUM_JOINTS) inputs. The frame_jacobian family takes the
    # target frame at RUNTIME: `target_jid` / `reference_frame` kwargs (defaults:
    # the leaf end-effector joint, LOCAL_WORLD_ALIGNED) are passed straight to
    # the GPU surface — no codegen-time baking.

    def com(self, q, *, _convention=None):
        """Center-of-mass world position p_com (3,) and CoM Jacobian J_com.

        Returns ``(p_com, J_com)`` where ``p_com`` is ``(B, 3)`` and ``J_com``
        is ``(B, 3, NV)`` = ``d(p_com)/dv``. Matches ``RBDReference.com(q)``
        (= ``p_com``) and ``RBDReference.jacobian_com(q)`` (= ``J_com``).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention;
        ``p_com`` is invariant and the ``J_com`` columns are reframed (computed
        natively in the kernel)."""
        NV = self.num_vel
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_com_mujoco", False):
                raise NotImplementedError(
                "com(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
            q = np.ascontiguousarray(q, dtype=self._dt)
            raw = self._runner.com_mujoco(q)
        else:
            q = np.ascontiguousarray(q, dtype=self._dt)
            raw = self._runner.com(q)  # (B, 3 + 3*NV): [p_com(3); J_com(3 x NV col-major)]
        # [p(3); J col-major] ("vec_then_colmajor" in the shared module).
        p_com, j_com = self._shape_out("com", raw)
        return self._cast_out(p_com), self._cast_out(j_com)

    def ccrba(self, q, qd, *, _convention=None):
        """Centroidal momentum matrix A (6 x NV) and momentum h = A·qd (6,).

        Returns ``(A, h)`` where ``A`` is ``(B, 6, NV)`` and ``h`` is ``(B, 6)``,
        in the Pinocchio convention (``[linear; angular]`` at the CoM, world
        aligned). Matches ``RBDReference.ccrba(q, qd)``.

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention; ``h`` is invariant and the ``A`` columns are reframed
        (computed natively in the kernel)."""
        NV = self.num_vel
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_ccrba_mujoco", False):
                raise NotImplementedError(
                "ccrba(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
            q = np.ascontiguousarray(q, dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            raw = self._runner.ccrba_mujoco(q, qd)
        else:
            q = np.ascontiguousarray(q, dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            self._check_nq_width(qd, "qd")
            raw = self._runner.ccrba(q, qd)  # (B, 6*NV + 6): [A(6 x NV col-major); h(6)]
        A, h = self._shape_out("ccrba", raw)
        return self._cast_out(A), self._cast_out(h)

    def energy(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Kinetic / potential / mechanical energy. Returns ``(B, 3)`` =
        ``[KE, PE, KE+PE]``. Matches ``RBDReference.kinetic_energy`` /
        ``potential_energy`` / ``mechanical_energy`` (PE uses ``gravity``).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention; the energies are frame-INVARIANT, so the native kernel
        only converts the inputs and the output equals the pin result."""
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_energy_mujoco", False):
                raise NotImplementedError(
                "energy(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
            q = np.ascontiguousarray(q, dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            return self._cast_out(self._runner.energy_mujoco(q, qd, float(gravity)))
        q = np.ascontiguousarray(q, dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        self._check_nq_width(qd, "qd")
        return self._runner.energy(q, qd, float(gravity))

    def generalized_gravity(self, q, *, gravity: float = -9.81, _convention=None):
        """Generalized gravity torque g(q) = RNEA(q, 0, 0). Returns ``(B, NV)``.
        Matches ``RBDReference.generalized_gravity(q, GRAVITY=gravity)``.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned g is in the mjx frame (the base rows are rotated in-kernel).
        Output shape is invariant ``(B, NV)``."""
        if self._mjx_active(_convention) and getattr(self._runner, "has_generalized_gravity_mujoco", False):
            q = np.ascontiguousarray(q, dtype=self._dt)
            return self._runner.generalized_gravity_mujoco(q, float(gravity))
        if self._mjx_active(_convention):
            raise NotImplementedError(
                "generalized_gravity(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
        q = np.ascontiguousarray(q, dtype=self._dt)
        return self._runner.generalized_gravity(q, float(gravity))

    def nonlinear_effects(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Nonlinear (bias) effects c(q,qd) = RNEA(q, qd, 0) = C(q,qd)·qd + g(q).
        Returns ``(B, NV)``. Matches ``RBDReference.nonlinear_effects(q, qd,
        GRAVITY=gravity)``.

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and the returned bias matches MuJoCo's ``qfrc_bias`` (the
        floating-root accel-couple −ω×v is injected and the base rows rotated
        in-kernel). Output shape is invariant ``(B, NV)``."""
        if self._mjx_active(_convention) and getattr(self._runner, "has_nonlinear_effects_mujoco", False):
            q = np.ascontiguousarray(q, dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            return self._runner.nonlinear_effects_mujoco(q, qd, float(gravity))
        if self._mjx_active(_convention):
            raise NotImplementedError(
                "nonlinear_effects(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
        q = np.ascontiguousarray(q, dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        self._check_nq_width(qd, "qd")
        return self._runner.nonlinear_effects(q, qd, float(gravity))

    def coriolis_matrix(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Coriolis matrix C(q,qd). Returns ``(B, NV, NV)`` row-major, with
        ``C·qd + g(q) = nonlinear_effects(q, qd)``. Matches
        ``RBDReference.coriolis_matrix(q, qd)`` (gravity is unused by C; the
        kwarg mirrors the host wrapper signature).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and the returned ``C`` is the mjx-frame Coriolis matrix
        (congruence ``G C Gᵀ``, computed natively in the kernel)."""
        NV = self.num_vel
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_coriolis_matrix_mujoco", False):
                raise NotImplementedError(
                "coriolis_matrix(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
            q = np.ascontiguousarray(q, dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            raw = self._runner.coriolis_matrix_mujoco(q, qd, float(gravity))
            return self._cast_out(self._shape_out("coriolis_matrix", raw))
        q = np.ascontiguousarray(q, dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        self._check_nq_width(qd, "qd")
        raw = self._runner.coriolis_matrix(q, qd, float(gravity))  # (B, NV*NV) row-major
        return self._cast_out(self._shape_out("coriolis_matrix", raw))

    def kinetic_energy_regressor(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Kinetic-energy regressor y_KE, length ``10*num_bodies``, with
        ``KE = y_KE · π`` (π = stacked per-link inertial parameters, body-major,
        10 params/body). Returns ``(B, 10*num_bodies)``. Matches
        ``RBDReference.kinetic_energy_regressor(q, qd)`` (gravity unused).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention; the regressor is frame-INVARIANT, so the native kernel
        only converts the inputs and the output equals the pin result."""
        if self._mjx_active(_convention):
            self._require_mjx_twin("kinetic_energy_regressor", "kinetic_energy_regressor")
            q = np.ascontiguousarray(q, dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            return self._cast_out(self._runner.kinetic_energy_regressor_mujoco(q, qd, float(gravity)))
        q = np.ascontiguousarray(q, dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        self._check_nq_width(qd, "qd")
        return self._cast_out(self._runner.kinetic_energy_regressor(q, qd, float(gravity)))

    def potential_energy_regressor(self, q, *, gravity: float = -9.81, _convention=None):
        """Potential-energy regressor y_PE, length ``10*num_bodies``, with
        ``PE = y_PE · π``. Returns ``(B, 10*num_bodies)``. Matches
        ``RBDReference.potential_energy_regressor(q, GRAVITY=gravity)`` (PE uses
        ``gravity``; only the mass + first-moment columns are nonzero).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention;
        the regressor is frame-INVARIANT, so the native kernel only converts the
        input and the output equals the pin result."""
        if self._mjx_active(_convention):
            self._require_mjx_twin("potential_energy_regressor", "potential_energy_regressor")
            q = np.ascontiguousarray(q, dtype=self._dt)
            return self._cast_out(self._runner.potential_energy_regressor_mujoco(q, float(gravity)))
        q = np.ascontiguousarray(q, dtype=self._dt)
        return self._cast_out(self._runner.potential_energy_regressor(q, float(gravity)))

    def dccrba(self, q, *, _convention=None):
        """dCCRBA tensor ∂A/∂q, shape ``(B, 6, NV, NV)`` indexed
        ``[:, :, k, i] = ∂A[:, k]/∂q_i`` (Pinocchio centroidal convention,
        ``[linear; angular]`` at the CoM, world-aligned). Matches
        ``RBDReference.dccrba(q)`` (which returns ``(6, NV, NV)`` per sample).

        Runs on all robots — including mimic and big floating-base (the
        centroidal sweep pool spills to global memory at the spilled tiers). A
        clear ``RuntimeError`` is raised only in the rare case where even the
        most-spilled tier's centroidal pool exceeds this GPU's shared-memory cap.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned tensor is the dA/dq of the mjx CMM (double G^{-1} reframe of
        the qd-column and q-tangent indices + base-rotation frame term, baked
        in-kernel). Same ``(B, 6, NV, NV)`` layout."""
        if self._mjx_active(_convention) and getattr(self._runner, "has_dccrba_mujoco", False):
            q = np.ascontiguousarray(q, dtype=self._dt)
            raw = self._runner.dccrba_mujoco(q)
        elif self._mjx_active(_convention):
            raise NotImplementedError(
                "dccrba(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
        else:
            q = np.ascontiguousarray(q, dtype=self._dt)
            raw = self._runner.dccrba(q)  # (B, 6*NV*NV) flat, dA[row + 6*k + 6*NV*m]
        # flat dA[row + 6*k + 6*NV*m] -> (B, 6, NV, NV) ("dccrba" shared class).
        return self._cast_out(self._shape_out("dccrba", raw))

    def cmm_time_variation(self, q, qd, *, _convention=None):
        """Centroidal-momentum-matrix time variation Ȧ = dA(q(t))/dt, shape
        ``(B, 6, NV)`` (Pinocchio convention, ``[linear; angular]`` at the CoM,
        world-aligned) = ``Σ_i (∂A/∂q_i)·qd_i``. Matches
        ``RBDReference.cmm_time_variation(q, qd)``.

        Runs on all robots (mimic + big floating-base via centroidal-pool spill);
        raises a clear ``RuntimeError`` only on the rare oversized-pool case (see
        :py:meth:`dccrba`).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and the returned Ȧ has its columns reframed into the mjx
        frame (computed natively in the kernel)."""
        NV = self.num_vel
        if self._mjx_active(_convention):
            self._require_mjx_twin("cmm_time_variation", "cmm_time_variation")
            q = np.ascontiguousarray(q, dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            raw = self._runner.cmm_time_variation_mujoco(q, qd)  # (B, 6*NV) col-major A[r + 6*c]
            return self._cast_out(self._shape_out("cmm_time_variation", raw))
        q = np.ascontiguousarray(q, dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        self._check_nq_width(qd, "qd")
        raw = self._runner.cmm_time_variation(q, qd)  # (B, 6*NV) col-major A[r + 6*c]
        return self._cast_out(self._shape_out("cmm_time_variation", raw))

    def frame_jacobian(self, q, *, target_jid=None, reference_frame=None, _convention=None):
        """Geometric Jacobian (6 x NV, ``[linear; angular]``) of a frame.
        Returns ``(B, 6, NV)``. Matches ``RBDReference.frame_jacobian(q,
        frame_name, reference_frame)``.

        ``target_jid`` selects the frame's joint id (default: the leaf
        end-effector joint baked at codegen time). ``reference_frame`` is
        ``'LOCAL'`` (0), ``'WORLD'`` (1), or ``'LOCAL_WORLD_ALIGNED'`` (2, the
        default), or the equivalent int. Both are now RUNTIME parameters of the
        GPU surface.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned Jacobian is column-reframed ``J G⁻¹`` (mjx frame)."""
        NV = self.num_vel
        tj, rf = _frame_args(target_jid, reference_frame)
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_frame_jacobian_mujoco", False):
                raise NotImplementedError(
                "frame_jacobian(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
            q = np.ascontiguousarray(q, dtype=self._dt)
            raw = self._runner.frame_jacobian_mujoco(q, tj, rf)
            return self._shape_out("frame_jacobian", raw)
        q = np.ascontiguousarray(q, dtype=self._dt)
        raw = self._runner.frame_jacobian(q, tj, rf)  # (B, 6*NV) col-major: J[r + 6*c]
        return self._shape_out("frame_jacobian", raw)

    def frame_jacobian_dot(self, q, qd, *, target_jid=None, reference_frame=None, _convention=None):
        """Time derivative Jdot of :py:meth:`frame_jacobian` along v = qd
        (6 x NV, ``[linear; angular]``). Returns ``(B, 6, NV)``. Matches
        ``RBDReference.frame_jacobian_dot(q, qd, frame_name, reference_frame)``.

        ``target_jid`` / ``reference_frame`` are RUNTIME parameters (default:
        leaf-EE joint / ``LOCAL_WORLD_ALIGNED``); see :py:meth:`frame_jacobian`.

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and Jdot is column-reframed (mjx frame)."""
        NV = self.num_vel
        tj, rf = _frame_args(target_jid, reference_frame)
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_frame_jacobian_dot_mujoco", False):
                raise NotImplementedError(
                "frame_jacobian_dot(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
            q = np.ascontiguousarray(q, dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            raw = self._runner.frame_jacobian_dot_mujoco(q, qd, tj, rf)
            return self._shape_out("frame_jacobian_dot", raw)
        q = np.ascontiguousarray(q, dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        self._check_nq_width(qd, "qd")
        raw = self._runner.frame_jacobian_dot(q, qd, tj, rf)  # (B, 6*NV) col-major
        return self._shape_out("frame_jacobian_dot", raw)

    def osc_inertia(self, q, *, _convention=None):
        """Operational-space (task) inertia Lambda = (J·M⁻¹·Jᵀ)⁻¹ (6 x 6) for
        the leaf-EE frame (LWA). Returns ``(B, 6, 6)``. Matches
        ``RBDReference.osc_inertia(q)``.

        Lambda is frame-INVARIANT, but with ``output_convention="mujoco"`` the
        MuJoCo ``q`` (wxyz quaternion) must be reordered before the kinematics
        build — the mjx kernel does that, so mjx ``q`` routes through it."""
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_osc_inertia_mujoco", False):
                raise NotImplementedError(
                "osc_inertia(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
            q = np.ascontiguousarray(q, dtype=self._dt)
            raw = self._runner.osc_inertia_mujoco(q)
            return self._shape_out("osc_inertia", raw)
        q = np.ascontiguousarray(q, dtype=self._dt)
        raw = self._runner.osc_inertia(q)  # (B, 36) row/col-major (symmetric)
        return self._shape_out("osc_inertia", raw)

    # ─── runtime arbitrary multi-EE pose / pose-gradient (target + offset) ─────

    def _resolve_ee_jids(self, ee_joint_names):
        """Resolve ee_joint_names -> joint ids, mirroring
        RBDReference.select_end_effector_joints: ``None`` => all leaf joints; a
        str / list of names => their joint ids (from the cached joint_names map)."""
        names = self._meta.get("joint_names")
        if ee_joint_names is None:
            jids = self._meta.get("leaf_jids")
            if jids is None:
                raise RuntimeError(
                    "this .so predates the runtime-target API (no leaf_jids in "
                    "meta.json); re-register with force_rebuild=True")
            return list(jids)
        if names is None:
            raise RuntimeError(
                "this .so predates the runtime-target API (no joint_names in "
                "meta.json); re-register with force_rebuild=True")
        if isinstance(ee_joint_names, str):
            ee_joint_names = [ee_joint_names]
        jids = []
        for name in ee_joint_names:
            try:
                jids.append(names.index(name))
            except ValueError:
                raise ValueError(f"Could not find joint named: {name}")
        return jids

    def _normalize_ee_offsets(self, ee_offsets, num_ees):
        """Normalize ee_offsets to a list of 16-float COLUMN-MAJOR 4x4 SE(3) tool
        transforms (one per EE), matching the device ``s_Xtool`` layout.

        ``None`` => identity (frame origin). Each entry may be a point (``[x,y,z]``
        or homogeneous ``[x,y,z,1]`` -> pure translation, ``R_tool = I``) or a full
        4x4 SE(3) transform (rotation + translation of the tool/tip frame in the
        target joint frame). A single offset is broadcast to all EEs."""
        def _to_xtool(o):
            A = np.asarray(o, dtype=self._dt)
            if A.shape == (4, 4):
                X = A
            elif A.size in (3, 4):
                X = np.eye(4, dtype=self._dt)
                X[:3, 3] = A.reshape(-1)[:3]
            else:
                raise ValueError(
                    "ee offset must be [x,y,z], [x,y,z,1], or a 4x4 SE(3) transform")
            return np.ascontiguousarray(X.reshape(-1, order="F"), dtype=self._dt)
        identity = np.ascontiguousarray(np.eye(4, dtype=self._dt).reshape(-1, order="F"))
        if ee_offsets is None:
            return [identity] * num_ees
        # a bare 4x4 is a SINGLE offset, not an iterable of rows.
        if isinstance(ee_offsets, np.ndarray) and ee_offsets.shape == (4, 4):
            ee_offsets = [ee_offsets]
        offs = [_to_xtool(o) for o in ee_offsets]
        if len(offs) == 1:
            offs = offs * num_ees
        if len(offs) != num_ees:
            raise ValueError(
                f"ee_offsets length {len(offs)} != number of EEs {num_ees}")
        return offs

    def end_effector_pose_runtime(self, q, ee_joint_names=None, ee_offsets=None, *, _convention=None):
        """Runtime-target end-effector pose ``[xyz; rpy]`` at an offset point.

        Mirrors ``RBDReference.end_effector_pose(q, ee_joint_names, ee_offsets)``:
        ``ee_joint_names`` (None => all leaf joints, or a str / list of joint
        names) selects the EE frames, ``ee_offsets`` (None => frame origin, or one
        ``[x,y,z]`` / ``[x,y,z,1]`` per EE) shifts the measurement point. The
        single-target GPU kernel is looped over the resolved jid list and the
        results stacked.

        Returns ``(B, NUM_EE, 6)`` where each row is ``[x, y, z, roll, pitch, yaw]``.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        (quat reordered in-kernel); the pose VALUE is frame-invariant.
        """
        ee_joint_names, ee_offsets = self._tool_tip_default(ee_joint_names, ee_offsets)
        q = np.ascontiguousarray(q, dtype=self._dt)
        jids = self._resolve_ee_jids(ee_joint_names)
        offsets = self._normalize_ee_offsets(ee_offsets, len(jids))
        mjx = self._mjx_active(_convention)
        if mjx:
            self._require_mjx_twin("end_effector_pose_runtime")
        per_ee = []
        for jid, off in zip(jids, offsets):
            off_arr = np.ascontiguousarray(off, dtype=self._dt)
            if mjx:
                raw = self._runner.end_effector_pose_runtime_mujoco(q, int(jid), off_arr)  # (B, 6)
            else:
                raw = self._runner.end_effector_pose_runtime(q, int(jid), off_arr)  # (B, 6)
            per_ee.append(raw)
        return self._cast_out(np.stack(per_ee, axis=1))  # (B, NUM_EE, 6)

    def end_effector_pose_gradient_runtime(self, q, ee_joint_names=None, ee_offsets=None, *, _convention=None):
        """Runtime-target end-effector pose gradient ``d[xyz; rpy]/dv`` (6 x NV)
        at an offset point. Same ``ee_joint_names`` / ``ee_offsets`` semantics as
        :py:meth:`end_effector_pose_runtime`; mirrors
        ``RBDReference.end_effector_pose_gradient(q, ee_joint_names, ee_offsets)``.

        Returns ``(B, NUM_EE, 6, NV)``.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention;
        the base-linear columns are reframed by R^T in-kernel (column-reframe class).
        """
        ee_joint_names, ee_offsets = self._tool_tip_default(ee_joint_names, ee_offsets)
        q = np.ascontiguousarray(q, dtype=self._dt)
        NV = self.num_vel
        jids = self._resolve_ee_jids(ee_joint_names)
        offsets = self._normalize_ee_offsets(ee_offsets, len(jids))
        mjx = self._mjx_active(_convention)
        if mjx and not getattr(self._runner, "has_end_effector_pose_gradient_runtime_mujoco", False):
            raise NotImplementedError(
                "end_effector_pose_gradient_runtime(output_convention='mujoco') " + self._MJX_TWINS_ADVICE)
        per_ee = []
        for jid, off in zip(jids, offsets):
            off_arr = np.ascontiguousarray(off, dtype=self._dt)
            if mjx:
                raw = self._runner.end_effector_pose_gradient_runtime_mujoco(q, int(jid), off_arr)  # (B, 6*NV) col-major
            else:
                raw = self._runner.end_effector_pose_gradient_runtime(q, int(jid), off_arr)  # (B, 6*NV) col-major
            per_ee.append(self._shape_out("end_effector_pose_gradient_runtime", raw))  # (B, 6, NV)
        return self._cast_out(np.stack(per_ee, axis=1))  # (B, NUM_EE, 6, NV)

    # ─── field-standard short aliases ────────────────────────────────────────
    # `rnea`/`fd` are the names roboticists (pinocchio / frax / bard) reach for;
    # bind them to the long-named methods (aba / crba / minv already match).
    rnea = inverse_dynamics
    fd = forward_dynamics

    # ─── lifecycle ───────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release the underlying .so handle. After close(), method calls
        will fail. Idempotent."""
        if self._runner is not None:
            del self._runner
            self._runner = None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"RobotHandle(name={self._name!r}, num_joints={self.num_joints}, "
            f"num_vel={self.num_vel}, num_ees={self.num_ees}, "
            f"floating_base={self.floating_base})"
        )


# Friction 6: accept a single unbatched (nq,) sample on the core numpy value /
# kinematic methods (jax/torch already accept 1-D via vmap). Wrapped post-class so
# the method bodies stay batch-only and the single-sample plumbing lives in ONE
# visible place. The cost/integrator methods (quadratic_state_cost, integrator,
# …) are intentionally excluded — their dt / Q / x_des args need separate care.
for _m in ("inverse_dynamics", "forward_dynamics", "aba", "crba", "minv",
           "end_effector_pose", "end_effector_pose_gradient", "end_effector_pose_hessian",
           "fk_batched", "inverse_dynamics_gradient", "forward_dynamics_gradient",
           "idsva_so", "fdsva_so", "inverse_dynamics_regressor"):
    setattr(RobotHandle, _m, _accept_1d(getattr(RobotHandle, _m)))
del _m
# Re-bind the short aliases to the now-wrapped methods (the class-body
# `rnea = inverse_dynamics` captured the un-wrapped functions).
RobotHandle.rnea = RobotHandle.inverse_dynamics
RobotHandle.fd = RobotHandle.forward_dynamics
