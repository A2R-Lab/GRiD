"""Feature-macro emission for grid.cuh (the GRID_HAS_* / GRID_RBD_WITH_MUJOCO wall).

Emits the file-scope preprocessor gates that the binding wrapper's `#if`-guarded
C-ABI bodies key on: second-order availability, integrator availability, the
per-core-algorithm subset-build (rc=3) macros, and the mjx-twins gate. Moved
verbatim out of gen_all_code (C3 fold, 2026-09-08); coverage-gated by
test/test_feature_macro_coverage.py. `gen` is the GRiDCodeGenerator instance;
`algorithms` is the POST-dep-expansion emit set.
"""
import warnings

from .algorithms._idsva_so import _idsva_so_use_world_frame

CORE_HAS_MACROS = {
    "INVERSE_DYNAMICS": "inverse_dynamics",
    "MINV": "minv",
    "FORWARD_DYNAMICS": "forward_dynamics",
    "ABA": "aba",
    "CRBA": "crba",
    "INVERSE_DYNAMICS_GRADIENT": "inverse_dynamics_gradient",
    "FORWARD_DYNAMICS_GRADIENT": "forward_dynamics_gradient",
    # f_ext gradient family — big output buffers (dtau/dqdd_dfext each nv*6NB;
    # f_ext_gradient_dq is nv*6NB*nv, the single largest per-timestep buffer).
    # Gated so init_gridData (B2) skips their alloc when the algo wasn't generated.
    # BOTH map to the parent key "f_ext_gradient": gen_f_ext_gradient emits the
    # first-order kernels AND the A.3 (-dJ^T/dq) child UNCONDITIONALLY for both
    # base modes (C.2), so d_f_ext_gradient_dq must be allocated whenever
    # f_ext_gradient is requested. Keying the DQ macro to the (never-present)
    # child key "f_ext_gradient_dq" left the buffer unallocated -> null-write crash.
    "F_EXT_GRADIENT": "f_ext_gradient",
    "F_EXT_GRADIENT_DQ": "f_ext_gradient",
    "INVERSE_DYNAMICS_REGRESSOR": "inverse_dynamics_regressor",
    # FD parameter gradient (dqdd/dpi = -Minv . Y): its grid::kernel is
    # emitted only when 'forward_dynamics_parameter_gradient' is in the
    # (post-dep-expansion) algorithm set, so the jax/torch handler that
    # calls it must rc=3-stub (clean subset error) when it's omitted. In
    # the default "all" profile it IS requested → macro=1 → real body,
    # byte-identical to the pre-subset build.
    "FORWARD_DYNAMICS_PARAMETER_GRADIENT": "forward_dynamics_parameter_gradient",
    "END_EFFECTOR_POSE": "end_effector_pose",
    "END_EFFECTOR_POSE_GRADIENT": "end_effector_pose_gradient",
    "END_EFFECTOR_POSE_HESSIAN": "end_effector_pose_hessian",
    # RNEA-bias + PS5 surfaces whose wrapper bodies were also UNgated and
    # call grid::<fn> directly (so a subset that omits them must rc=3-stub).
    "GENERALIZED_GRAVITY": "generalized_gravity",
    "NONLINEAR_EFFECTS": "nonlinear_effects",
    "CORIOLIS_MATRIX": "coriolis_matrix",
    "KINETIC_ENERGY_REGRESSOR": "kinetic_energy_regressor",
    "POTENTIAL_ENERGY_REGRESSOR": "potential_energy_regressor",
}


def emit_feature_macros(gen, algorithms):
    # File-scope preprocessor mirrors of the second-order codegen gates.
    # We also emit `const int GRID_GENERATES_*_SO` inside `namespace grid`
    # below (for C++ runtime / test consumers), but `#if X` requires a
    # preprocessor macro — a namespaced const reads as the undefined-symbol
    # zero in `#if`, silently turning off any consumer that gates on it
    # (notably the bench's timeGRiD_{single,batch}.cu measure_* blocks).
    # Different names from the namespaced const avoid macro/const collision.
    gen.gen_add_code_line(
        "#define GRID_HAS_IDSVA_SO_BODY_FRAME " + str(int(getattr(gen, "generate_idsva_so_body_frame", True)))
    )
    gen.gen_add_code_line(
        "#define GRID_HAS_FDSVA_SO " + str(int(getattr(gen, "generate_fdsva_so", True)))
    )
    gen.gen_add_code_line(
        "#define GRID_HAS_IDSVA_SO_WORLD_FRAME " + str(int(getattr(gen, "generate_idsva_so_world_frame", False)))
    )
    # GRID_HAS_IDSVA_SO gates the dispatching `grid::idsva_so` host wrapper.
    # Emitted whenever the chosen variant for this robot's base type is
    # available: body_frame for fixed-base (always), world_frame for
    # floating-base (opt-in via enable_idsva_so_world_frame).
    has_idsva_so = getattr(gen, "generate_idsva_so_body_frame", True) and (
        (not gen.robot.floating_base) or getattr(gen, "generate_idsva_so_world_frame", False)
    )
    gen.gen_add_code_line("#define GRID_HAS_IDSVA_SO " + str(int(has_idsva_so)))
    # The `grid::idsva_so` dispatcher forwards at CODEGEN time (world frame for
    # floating / spherical / high-DOF fixed — _idsva_so_use_world_frame). A
    # consumer probing launchability (e.g. the bench smem-skip guard) must
    # check the DISPATCHED kernel's smem bytes, not a fixed frame: on floating
    # the body frame is a no-ladder diagnostic (242 KB on h2_plus) while the
    # dispatched world frame fits at every tier.
    gen.gen_add_code_line(
        "#define GRID_IDSVA_SO_DISPATCHES_WORLD_FRAME "
        + str(int(_idsva_so_use_world_frame(gen))))
    # Integrator availability gates. Both value and gradient kernels are
    # emitted whenever requested, for fixed- and floating-base. (Floating
    # gradient is currently Euler-only and inherits the upstream
    # forward_dynamics_gradient dqdd/dqd bug; see _normalize_codegen_algorithms.)
    # Consumers must #if-guard gradient calls so a build that omits
    # integrator_gradient still compiles.
    gen.gen_add_code_line(
        "#define GRID_HAS_INTEGRATOR " + str(int("integrator" in algorithms)))
    gen.gen_add_code_line(
        "#define GRID_HAS_INTEGRATOR_GRADIENT " + str(int("integrator_gradient" in algorithms)))
    # Subset-build (rc=3 model): per-CORE-algo availability macros. These gate the
    # binding wrapper's currently-UNGATED `extern "C"` core bodies so a reduced
    # codegen profile (subset algorithm_list) ships a clean rc=3 stub for the
    # un-requested cores instead of failing to link on a missing grid:: inner.
    # The set read is the POST-dep-expansion `algorithms`, so a transitively-pulled
    # dep (e.g. minv via forward_dynamics) gets macro=1 and its real body. For the
    # default "all" profile every one of these is 1 (every core requested), so the
    # wrapper's `#if GRID_HAS_X` selects the real body verbatim and the header diff
    # vs a pre-subset build is EXACTLY these added `#define`s (no body change).
    # The wrapper uses `#if GRID_HAS_X` (not `#ifdef`), so an UNDEFINED macro reads
    # as 0 and fails LOUD (rc=3) rather than silently shipping a stub — but we emit
    # every core macro unconditionally here so "undefined" can only happen if this
    # list and the wrapper's `#if` sites ever drift.
    #
    # `end_effector_pose` is a transitive dep of nearly everything kinematic; its
    # gradient/hessian are the algorithm keys that map to the
    # grid_rbd_end_effector_pose{,_gradient,_hessian} wrapper symbols.
    for macro_suffix, algo_key in CORE_HAS_MACROS.items():
        gen.gen_add_code_line(
            "#define GRID_HAS_" + macro_suffix + " "
            + str(int(algo_key in gen.generated_algorithms)))
    # GRID_RBD_WITH_MUJOCO gates the binding's mjx (MuJoCo output-convention)
    # C-ABI entry points: the `grid::*<...,MUJOCO_OUTPUT=true>` template overloads
    # are EMITTED only for a floating-base robot WITHOUT mimic joints or skew
    # axes (a mimic/skew robot skips the mjx variants — see the `mjx_*`/`mjx_inner`
    # gates in the algorithm emitters, all `floating and not (mimic or skew)`).
    # So the wrapper's mjx symbols must compile ONLY when those overloads exist;
    # otherwise a floating+mimic robot (e.g. h1_2, 12 mimic joints) fails to build
    # on `grid::*<...,true>` "no matching function". Defined (to 1) iff the mjx
    # overloads were emitted; absent otherwise so `#ifdef GRID_RBD_WITH_MUJOCO`
    # is a clean no-op on fixed-base AND mimic/skew headers.
    # enable_mujoco_kernels=False suppresses this define -> the binding's mjx
    # entry points compile out -> the mjx twins are never instantiated (trigger 2
    # of 2; see gen_all_code). getattr keeps legacy callers that never set it.
    if getattr(gen, "enable_mujoco_kernels", True) and gen.robot.floating_base and not (
            gen.robot_has_mimic_joints() or gen.robot.robot_has_skew_axis()):
        gen.gen_add_code_line("#define GRID_RBD_WITH_MUJOCO 1")
        # Warn BEFORE the multi-hour compile rather than after the OOM. The mjx twins
        # dominate a big floating-base build (~2.1M of ~2.4M SASS lines on
        # g1-floating; the idsva_so_world_frame twin alone is 28x its pin kernel), so
        # a humanoid built with them can exhaust a 62 GB box, while the same robot
        # built pin-only lands in ~33 min at ~11 GB peak. Threshold is empirical:
        # go2-floating (nv=18) builds fine with mjx; g1-floating (nv=36) did not
        # build at all. Only fires where the twins actually exist (floating,
        # non-mimic, non-skew) and only when they are switched ON.
        if gen.robot.get_num_vel() >= 30:
            warnings.warn(
                f"Generating MuJoCo-convention (mjx) kernel twins for a large "
                f"floating-base robot (nv={gen.robot.get_num_vel()}). These twins "
                f"dominate the build and can exhaust host RAM during nvcc/cicc. If "
                f"you do not need the MuJoCo output convention, build pin-only with "
                f"enable_mujoco_kernels=False (register_robot / gen_all_code), or set "
                f"GRID_ENABLE_MUJOCO_KERNELS=0 for a whole codegen session.",
                stacklevel=2,
            )
    gen.gen_add_code_line("")
