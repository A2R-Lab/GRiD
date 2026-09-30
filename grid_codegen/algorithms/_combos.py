"""Combination wrappers (dynamics_core/all_dynamics/kinematics_only) and the
centroidal quick-win dispatch. H4 move from GRiDCodeGenerator.py (2026-08-27,
verbatim)."""


def gen_combination_functions(self, algorithms, fixed_target_name = ""):
    kinematics_suffix = "" if fixed_target_name == "" else "_" + fixed_target_name

    def has_all(names):
        return all(name in algorithms for name in names)

    def emit_dynamics_combo(name, description, calls):
        self.gen_add_func_doc(description, [], [], None)
        self.gen_add_code_line("template <typename T, gridDataKind KIND = GRID_DATA_ALL>")
        self.gen_add_code_line("__host__")
        self.gen_add_code_line("void " + name + "(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,", False)
        self.gen_add_code_line("                   const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {", True)
        self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, \"" + name + " requires all-data or dynamics gridData\");")
        self.gen_add_code_lines(calls)
        self.gen_add_end_function()

    def emit_kinematics_combo(name, description, calls):
        self.gen_add_func_doc(description, [], [], None)
        self.gen_add_code_line("template <typename T, gridDataKind KIND = GRID_DATA_ALL>")
        self.gen_add_code_line("__host__")
        self.gen_add_code_line("void " + name + "(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,", False)
        self.gen_add_code_line("                     const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {", True)
        self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, \"" + name + " requires all-data or kinematics gridData\");")
        self.gen_add_code_lines(calls)
        self.gen_add_end_function()

    id_call = "inverse_dynamics<T,false,false,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);"
    minv_call = "minv<T,false,KIND>(hd_data,d_robotModel,num_timesteps,block_dimms,thread_dimms,streams);"
    fd_call = "forward_dynamics<T,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);"
    inverse_dynamics_gradient_call = "inverse_dynamics_gradient<T,false,false,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);"
    forward_dynamics_gradient_call = "forward_dynamics_gradient<T,false,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);"
    aba_call = "aba<T,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);"
    crba_call = "crba<T,false,KIND>(hd_data,d_robotModel,gravity,num_timesteps,block_dimms,thread_dimms,streams);"

    if has_all(("inverse_dynamics", "minv", "forward_dynamics")):
        core_calls = [id_call, minv_call, fd_call]
        emit_dynamics_combo("dynamics_core", "Run inverse dynamics, Minv, and forward dynamics in sequence", core_calls)
        emit_dynamics_combo("id_minv_fd", "Run inverse dynamics, Minv, and forward dynamics in sequence", core_calls)

    if has_all(("inverse_dynamics", "inverse_dynamics_gradient")):
        emit_dynamics_combo("id_and_id_gradient", "Run inverse dynamics and its first derivative in sequence", [id_call, inverse_dynamics_gradient_call])

    if has_all(("forward_dynamics", "forward_dynamics_gradient")):
        emit_dynamics_combo("fd_and_fd_gradient", "Run forward dynamics and its first derivative in sequence", [fd_call, forward_dynamics_gradient_call])

    if has_all(("inverse_dynamics_gradient", "forward_dynamics_gradient")):
        emit_dynamics_combo("dynamics_gradients", "Run inverse and forward dynamics gradients in sequence", [inverse_dynamics_gradient_call, forward_dynamics_gradient_call])

    if has_all(("inverse_dynamics", "minv", "forward_dynamics", "inverse_dynamics_gradient", "forward_dynamics_gradient")):
        calls = [id_call, minv_call, fd_call, inverse_dynamics_gradient_call, forward_dynamics_gradient_call]
        if "aba" in algorithms:
            calls.append(aba_call)
        if "crba" in algorithms:
            calls.append(crba_call)
        emit_dynamics_combo("all_dynamics", "Run all generated non-second-order dynamics wrappers in sequence", calls)
        emit_dynamics_combo("dynamics_only", "Run all generated non-second-order dynamics wrappers in sequence", calls)

    kinematics_calls = []
    if "end_effector_pose" in algorithms:
        kinematics_calls.append("end_effector_pose" + kinematics_suffix + "<T,false,KIND>(hd_data,d_robotModel,num_timesteps,block_dimms,thread_dimms,streams);")
    if "end_effector_pose_gradient" in algorithms:
        kinematics_calls.append("end_effector_pose_gradient" + kinematics_suffix + "<T,false,KIND>(hd_data,d_robotModel,num_timesteps,block_dimms,thread_dimms,streams);")
    if "end_effector_pose_hessian" in algorithms:
        kinematics_calls.append("end_effector_pose_hessian" + kinematics_suffix + "<T,false,KIND>(hd_data,d_robotModel,num_timesteps,block_dimms,thread_dimms,streams);")
    if kinematics_calls:
        emit_kinematics_combo("kinematics_only", "Run all generated kinematics wrappers in sequence", kinematics_calls)


def gen_centroidal_quickwins(self, algorithms):
    """Emit the G2 centroidal quick-win families (R1-R3). Each is gated on
    the grid:: deps it composes being present; a missing dep emits a comment
    instead of an undefined call (mirrors gen_grid_plant's gating). Mimic
    robots are skipped for the kinematics-domain centroidal families (the
    per-body Jacobian fold isn't mimic-reduced yet) — gravity/bias still
    emit since they reuse the mimic-aware RNEA inner."""
    # R1 generalized_gravity / nonlinear_effects: RNEA bias wrappers. R6: each
    # emits when ITS OWN key is requested (its 'id' dep is auto-expanded by
    # _normalize_codegen_algorithms); requesting only 'id' no longer emits them.
    want_gg = "generalized_gravity" in algorithms
    want_nle = "nonlinear_effects" in algorithms
    if want_gg or want_nle:
        # gen_id_bias depends on grid::inverse_dynamics_inner (auto-pulled).
        if want_gg:
            self.gen_id_bias(gravity_only=True)
        if want_nle:
            self.gen_id_bias(gravity_only=False)
    else:
        self.gen_add_code_line("// [centroidal] generalized_gravity/nonlinear_effects skipped: not requested (request 'generalized_gravity'/'nonlinear_effects').")
    # R3/R2/energy: kinematics-domain centroidal families. R6: each emits on
    # its OWN key (com/ccrba/energy); the ee_pose homogeneous-transform world-
    # frame machinery is auto-expanded as their dep. Mimic robots are SUPPORTED
    # (centroidal_inner's per-body Jacobian + the dccrba per-unit phi are
    # alpha-folded, mirroring the mimic-aware RBDReference oracle).
    want_com = "com" in algorithms
    want_ccrba = "ccrba" in algorithms
    want_energy = "energy" in algorithms
    # PS5 dCCRBA surfaces also reuse centroidal_inner (the shared world sweep).
    want_dccrba = "dccrba" in algorithms
    want_cmm = "cmm_time_variation" in algorithms
    want_centroidal = want_com or want_ccrba or want_energy or want_dccrba or want_cmm
    if want_centroidal:
        self.gen_centroidal_inner()
        if want_com:
            self.gen_com()
            # Signal to the binding layer that grid::com is emitted (the
            # grid_rbd C-ABI gates each centroidal wrapper on its own define).
            self.gen_add_code_line("#define GRID_HAS_COM 1")
        if want_ccrba:
            self.gen_ccrba()
            self.gen_add_code_line("#define GRID_HAS_CCRBA 1")
        if want_energy:
            self.gen_energy()
            self.gen_add_code_line("#define GRID_HAS_ENERGY 1")
        # PS5 dCCRBA: emit the qd-contraction Adot first (lighter), then the
        # full 6*nv*nv tensor (with per-tier output spill).
        if want_cmm:
            self.gen_cmm_time_variation()
        if want_dccrba:
            self.gen_dccrba()
        # Signal to the binding layer (wrapper_template.cu) that the centroidal
        # tensor host wrappers (dccrba / cmm_time_variation) were emitted; the
        # grid_rbd C-ABI gates its dccrba / cmm_time_variation symbols on these
        # defines and returns rc=3 otherwise.
        if want_dccrba:
            self.gen_add_code_line("#define GRID_HAS_DCCRBA 1")
        if want_cmm:
            self.gen_add_code_line("#define GRID_HAS_CMM_TIME_VARIATION 1")
    else:
        self.gen_add_code_line("// [centroidal] com/ccrba/energy/dccrba/cmm_time_variation skipped: not requested.")
    # PS5 full Coriolis matrix C(q,qd): closed-form world-frame spatial recursion.
    # Self-contained (reuses only the XImats spatial-transform load + cross/icrf
    # helpers); emits on its OWN key. Mimic-safe (alpha-folded column assembly).
    if "coriolis_matrix" in algorithms:
        self.gen_coriolis_matrix()
    else:
        self.gen_add_code_line("// [coriolis] coriolis_matrix skipped: not requested (request 'coriolis_matrix').")

# finally generate all of the code
