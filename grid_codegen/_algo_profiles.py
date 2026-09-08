"""Algorithm-selection profiles for GRiD codegen.

The single source of truth for which algorithm keys exist, the curated profile
sets ("all", "dynamics", ...), and the transitive dependency expansion that
turns a user selection into the exact emit set. Moved verbatim out of
GRiDCodeGenerator (C3 fold, 2026-09-08); `gen` is the GRiDCodeGenerator
instance (robot topology drives the mimic/spherical/floating dep rules).
"""


def normalize_codegen_algorithms(gen, codegen_profile = "all", algorithm_list = None):
    all_algorithms = {
        "inverse_dynamics", "minv", "forward_dynamics", "inverse_dynamics_gradient", "forward_dynamics_gradient", "aba", "crba",
        "idsva_so_body_frame", "fdsva_so", "end_effector_pose", "end_effector_pose_gradient", "end_effector_pose_hessian",
        "integrator", "integrator_gradient", "integrator_with_gradient",
        "f_ext_gradient", "inverse_dynamics_regressor", "forward_dynamics_parameter_gradient",
        "kinetic_energy_regressor", "potential_energy_regressor",
        # G2 centroidal quick-wins: each is its OWN first-class key (R6). They
        # expand to their real deps below (com/ccrba/energy -> ee_pose kin
        # machinery; generalized_gravity/nonlinear_effects -> id RNEA-bias), but
        # requesting a SIBLING dep no longer silently emits them.
        "com", "ccrba", "energy", "generalized_gravity", "nonlinear_effects",
        # PS5: full Coriolis matrix C(q,qd) (closed-form spatial recursion; C qd+g=nle).
        "coriolis_matrix",
        # PS5: analytic dCCRBA (∂A/∂q tensor, 6×NV×NV) + its qd-contraction Adot.
        "dccrba", "cmm_time_variation",
    }
    # E2 (additive, opt-in only): frame_jacobian is NOT part of the default
    # `all` profile so the default-profile header stays byte-identical. It is
    # a recognized key for explicit algorithm_list requests and has its own
    # `frame-jacobian` profile. Requires `ee_pose` (world-transform machinery).
    # E2 CUDA parity (additive, opt-in only): frame_jacobian_dot (Jdot) and
    # osc_inertia (Lambda) join frame_jacobian as recognized-but-not-default
    # keys so the default `all` header stays byte-identical.
    # idsva_so_world_frame is a first-class, recognized algorithm_list key
    # (mirrors idsva_so_body_frame), but it is NOT in the default `all`
    # profile: on fixed-base the default emits only body_frame, and on
    # floating-base world_frame is already pulled in by the
    # enable_idsva_so_world_frame default (True). Keeping it opt-in (not in
    # `all`) preserves the default-profile header byte-for-byte.
    # F1: integrator_hessian (the plant_step_hessian s_d2AB surface) is a
    # recognized opt-in key. It is NOT in the default `all` profile (keeping
    # the default header byte-identical for non-fdsva_so callers is moot since
    # `all` already pulls fdsva_so, but the hessian device fn + plant wrapper
    # are gated on fdsva_so membership, which `all` satisfies). Requesting it
    # explicitly pulls in fdsva_so + integrator below.
    opt_in_algorithms = {"frame_jacobian", "frame_jacobian_dot", "osc_inertia",
                         "end_effector_pose_runtime", "end_effector_pose_gradient_runtime",
                         "idsva_so_world_frame", "integrator_hessian"}
    profile_algorithms = {
        "all": all_algorithms,
        "frame-jacobian": {"end_effector_pose", "minv", "frame_jacobian",
                           "frame_jacobian_dot", "osc_inertia"},
        "dynamics": {"inverse_dynamics", "minv", "forward_dynamics", "inverse_dynamics_gradient", "forward_dynamics_gradient", "aba", "crba", "idsva_so_body_frame", "fdsva_so",
                     "integrator", "integrator_gradient", "integrator_with_gradient"},
        "dynamics-core": {"inverse_dynamics", "minv", "forward_dynamics"},
        "dynamics-gradients": {"inverse_dynamics", "minv", "forward_dynamics", "inverse_dynamics_gradient", "forward_dynamics_gradient", "f_ext_gradient"},
        "regressor": {"inverse_dynamics", "inverse_dynamics_regressor",
                      "kinetic_energy_regressor", "potential_energy_regressor"},
        "energy-regressor": {"inverse_dynamics", "end_effector_pose",
                             "kinetic_energy_regressor", "potential_energy_regressor"},
        "fd-param-gradient": {"inverse_dynamics", "minv", "forward_dynamics", "inverse_dynamics_regressor", "forward_dynamics_parameter_gradient"},
        "f-ext-gradient": {"inverse_dynamics", "minv", "f_ext_gradient"},
        "kinematics": {"end_effector_pose"},
        "kinematics-derivatives": {"end_effector_pose", "end_effector_pose_gradient", "end_effector_pose_hessian"},
        "second-order": {"inverse_dynamics", "minv", "forward_dynamics", "inverse_dynamics_gradient", "forward_dynamics_gradient", "idsva_so_body_frame", "fdsva_so"},
        "integrators": {"inverse_dynamics", "minv", "forward_dynamics", "inverse_dynamics_gradient", "forward_dynamics_gradient", "integrator", "integrator_gradient",
                        "integrator_with_gradient"},
    }
    # Clean break: NO algorithm-name aliases. Every algorithm is requested by its
    # ONE canonical verbose key (== emitted grid:: symbol == bench key == printf
    # label); canonicalize() below tolerates dash-vs-underscore spelling only.
    # The only entries here are PROFILE-selection conveniences (curated algo SETS,
    # a distinct concept from algo naming) — they alias to profile_algorithms keys.
    aliases = {
        "all-dynamics": "dynamics",
        "dynamics-only": "dynamics",
        "kinematics-only": "kinematics",
    }

    def canonicalize(name):
        return aliases.get(str(name).strip().lower().replace("_", "-"), str(name).strip().lower().replace("-", "_"))

    if algorithm_list is None:
        profile_key = str(codegen_profile or "all").strip().lower().replace("_", "-")
        profile_key = aliases.get(profile_key, profile_key)
        if profile_key not in profile_algorithms:
            raise ValueError("Unknown GRiD codegen profile: " + str(codegen_profile))
        algorithms = set(profile_algorithms[profile_key])
    else:
        if isinstance(algorithm_list, str):
            requested = [item for item in algorithm_list.replace(";", ",").split(",") if item.strip()]
        else:
            requested = list(algorithm_list)
        algorithms = set()
        for item in requested:
            key = canonicalize(item)
            if key in profile_algorithms:
                algorithms.update(profile_algorithms[key])
            elif key in all_algorithms or key in opt_in_algorithms:
                algorithms.add(key)
            else:
                raise ValueError("Unknown GRiD algorithm selection: " + str(item))

    # E2: frame_jacobian (+ its Jdot/Lambda siblings) need the world-transform
    # machinery (ee_pose) and, for OSC composition, minv. Pull them in plus the
    # base frame_jacobian emit (the siblings reuse frame_jacobian_inner).
    if "frame_jacobian_dot" in algorithms or "osc_inertia" in algorithms:
        algorithms.add("frame_jacobian")
    if "frame_jacobian" in algorithms:
        algorithms.update({"end_effector_pose", "minv"})
    # Runtime-target pose / pose-gradient (additive, opt-in): need ee_pose's
    # world-transform (s_XmatsHom) machinery only -- same dep as frame_jacobian
    # but WITHOUT minv (no OSC composition).
    if ("end_effector_pose_runtime" in algorithms
            or "end_effector_pose_gradient_runtime" in algorithms):
        algorithms.add("end_effector_pose")
    if "forward_dynamics_gradient" in algorithms:
        algorithms.update({"inverse_dynamics", "minv", "forward_dynamics", "inverse_dynamics_gradient"})
    if "inverse_dynamics_gradient" in algorithms:
        algorithms.add("inverse_dynamics")
    # f_ext gradient: dtau/dfext reuses the RNEA spatial-transform load (id),
    # dqdd/dfext reuses minv's inner (minv).
    if "f_ext_gradient" in algorithms:
        algorithms.update({"inverse_dynamics", "minv"})
    if "aba" in algorithms and gen.robot.floating_base:
        algorithms.update({"inverse_dynamics", "minv", "forward_dynamics"})
    # F1: integrator_hessian (plant_step_hessian) composes fdsva_so_device for
    # the four 2nd-order FD blocks; gravity/integrator value share the same RBD
    # chain. Pull in fdsva_so (which expands to the full 2nd-order dep set below)
    # plus integrator.
    if "integrator_hessian" in algorithms:
        algorithms.update({"fdsva_so", "integrator"})
    if "fdsva_so" in algorithms:
        algorithms.update({"inverse_dynamics", "minv", "forward_dynamics", "inverse_dynamics_gradient", "forward_dynamics_gradient", "idsva_so_body_frame"})
    # Mimic AND spherical (Tier-C) Minv route through crba_inner (see _minv.py:
    # the reduced-space M is built via CRBA then inverted — the per-body scalar
    # ABA-recursion minv does not generalize to multi-DoF / folded joints), so a
    # mimic OR spherical robot emitting `minv` has a hidden dependency on `crba`
    # for the crba_inner definition. Declare it so the forward-decl'd crba_inner is
    # actually emitted (else nvlink: unresolved extern crba_inner). This matters in
    # particular when `minv` is pulled in transitively (e.g. by frame_jacobian)
    # WITHOUT `crba` being requested directly. Non-mimic/non-spherical minv doesn't
    # touch crba, so this is additive — byte-identical for cardinal robots.
    if "minv" in algorithms and (gen.robot_has_mimic_joints()
                                 or gen.robot.robot_has_spherical()):
        algorithms.add("crba")
    # idsva_so_world_frame is emitted alongside idsva_so_body_frame (it
    # reuses the body-frame inner scaffolding + dispatcher); requesting it
    # pulls in body_frame so the world-frame emit block (gated under
    # body_frame) actually runs.
    if "idsva_so_world_frame" in algorithms:
        algorithms.add("idsva_so_body_frame")
    if "idsva_so_body_frame" in algorithms:
        algorithms.add("inverse_dynamics")
        if gen.robot.floating_base:
            algorithms.add("inverse_dynamics_gradient")
            # FLOATING idsva_so (both the body-frame floating-reference inner and
            # the world-frame inner) builds the dense mass matrix M via crba_inner
            # (M into the dead gradient pool), same as floating forward_dynamics_gradient
            # above. A non-mimic/non-spherical floating robot is NOT caught by the
            # mimic/spherical minv->crba guards, so it would emit the crba_inner CALL
            # without the DEFINITION -> nvlink: unresolved extern crba_inner (surfaces
            # in subset codegen requesting idsva_so WITHOUT crba). Fixed-base idsva_so
            # never calls crba_inner, so gate on floating (cardinal byte-identical).
            algorithms.add("crba")
        # Spherical (ball) joints route the idsva_so dispatcher to the WORLD
        # frame (the body-frame inner's single-DoF S contractions are wrong
        # for a 3-DoF joint). Force the world-frame emit so the routed
        # dispatcher call links. Gated on spherical => cardinal byte-identical.
        if gen.robot.robot_has_spherical():
            algorithms.add("idsva_so_world_frame")
    # integrator value needs forward dynamics; gradient needs FD + FD-gradient.
    # Floating-base integrator gradients are emitted for all five types
    # (Euler / SI-Euler / Midpoint / RK3 / RK4) and validated against
    # RBDReference. The earlier "forward_dynamics_gradient drops floating
    # linear<->angular velocity coupling" story was a MISDIAGNOSIS: the real
    # defect was a CUDA thread-count race in inverse_dynamics_gradient
    # (missing __syncthreads + a 6-way root accumulation), correct at 32
    # threads and racing above one warp. Fixed; see HANDOFF.md §3.
    if "integrator" in algorithms:
        algorithms.update({"inverse_dynamics", "minv", "forward_dynamics"})
    if "integrator_gradient" in algorithms or "integrator_with_gradient" in algorithms:
        algorithms.update({"inverse_dynamics", "minv", "forward_dynamics", "inverse_dynamics_gradient", "forward_dynamics_gradient"})
    # FLOATING forward_dynamics_gradient rebuilds the mass matrix M via crba_inner
    # (the floating FD-gradient path composes M explicitly, unlike the fixed-base
    # per-body ABA-recursion which never calls crba_inner). A NON-mimic/non-spherical
    # floating robot is NOT caught by the mimic/spherical minv->crba guards above, so
    # it would emit the crba_inner CALL (in the floating FD-gradient) without the
    # DEFINITION -> nvlink: unresolved extern crba_inner (breaks every floating
    # integrator gradient: euler/si/rk + trapezoidal). Declare the crba dependency
    # whenever a floating robot emits forward_dynamics_gradient. Fixed-base FD-gradient
    # does not call crba_inner, so this is gated on floating (cardinal byte-identical).
    if "forward_dynamics_gradient" in algorithms and gen.robot.floating_base:
        algorithms.add("crba")
    # The minv->crba spherical/mimic expansion above runs BEFORE integrator
    # pulls in `minv` (line ordering), so a spherical robot requesting ONLY
    # `integrator` would add minv here WITHOUT crba_inner being emitted ->
    # nvlink: unresolved extern crba_inner. Re-assert the crba dependency
    # after the integrator (and any other late minv-adders) have run. Additive
    # for cardinal robots (the guard is false), byte-identical.
    if "minv" in algorithms and (gen.robot_has_mimic_joints()
                                 or gen.robot.robot_has_spherical()):
        algorithms.add("crba")
    # R6 centroidal deps: com/ccrba/energy reuse the ee_pose homogeneous-
    # transform world-frame machinery; generalized_gravity/nonlinear_effects
    # are RNEA-bias wrappers around the id inner. Expanding the DEP is correct;
    # the R6 fix is that requesting only the dep no longer pulls the centroidal
    # OUTPUT (gen_centroidal_quickwins now gates on these own keys).
    if algorithms & {"com", "ccrba", "energy"}:
        algorithms.add("end_effector_pose")
    if algorithms & {"generalized_gravity", "nonlinear_effects"}:
        algorithms.add("inverse_dynamics")
    # Energy regressors (PS5): KE reuses the RNEA forward sweep (inverse_dynamics
    # inner); PE reuses the ee_pose world-transform homogeneous machinery.
    if "kinetic_energy_regressor" in algorithms:
        algorithms.add("inverse_dynamics")
    if "potential_energy_regressor" in algorithms:
        algorithms.add("end_effector_pose")
    # PS5 Coriolis matrix: closed-form world-frame spatial recursion. Reuses the
    # XImats spatial-transform load + the cross/icrf spatial helpers (the same dep
    # set as nonlinear_effects -> inverse_dynamics). It does NOT call the RNEA inner,
    # but pulling inverse_dynamics guarantees the spatial-algebra helper emit.
    if "coriolis_matrix" in algorithms:
        algorithms.add("inverse_dynamics")
    # PS5 dCCRBA: both surfaces reuse the centroidal kinematics world-sweep
    # (centroidal_inner), so they pull the ee_pose homogeneous-transform machinery
    # exactly like com/ccrba.
    if algorithms & {"dccrba", "cmm_time_variation"}:
        algorithms.add("end_effector_pose")
    return algorithms
