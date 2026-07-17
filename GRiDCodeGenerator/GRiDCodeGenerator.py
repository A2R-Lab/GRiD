import os
import json
import numpy as np

# Codegen-time frame-selection predicate shared with the idsva_so dispatcher /
# device wrapper. The emission gate (enable_idsva_so_world_frame) MUST agree with
# it so the world-frame inner is emitted exactly when a wrapper forwards there
# (else high-DOF fixed-base robots route to an undefined idsva_so_world_frame_inner).
from .algorithms._idsva_so import _idsva_so_use_world_frame

# Per-algo DESCRIPTOR table (item M) — single source of truth for the launch-config
# enum/symbol metadata that used to live in the two module-level dicts below, and
# (Step 2) for the KERNEL_ATTR_MANIFEST / mujoco_manifest per-entry HEAD fields
# (algo_label, algo_short, gate_attr, bytes_macro) — only the irregular kernel
# overload SIGNATURES stay as co-located payload data.
from .algo_registry import (ALGO_DESCRIPTORS, build_launch_config_algo_to_symbol, descriptor_for,
                            arena_ctx_from_codegen, compose_arena_full, compose_arena_rungs,
                            ARENA_COMPOSED_KEYS, ARENA_RUNG_KEYS)

# ─── A1b launch-config bake (single source of truth) ─────────────────────────
# The autotuned per-(robot,base,algo) {tier,threads} live in
# launch_configs/<robot>/<DEFAULT_GPU>.json. At codegen time we read the
# matching entry and bake it into the generated header as host-side constants
# (grid::launch_cfg<ALGO_*>). This is purely ADDITIVE host content — kernel
# bodies are untouched (Gate-A). A missing robot / GPU / algo falls back to the
# conservative MAX_PERF_LEVEL_THREADS default so un-tuned robots are unaffected.

# DEFAULT GPU whose autotuned config ships baked-in.
LAUNCH_CONFIG_DEFAULT_GPU = "rtx5090_sm120"

# JSON tier name -> emitted TIER_* enum symbol. Global (not per-algo), kept inline.
LAUNCH_CONFIG_TIER_SYMBOL = {
    "shared":  "TIER_SHARED",
    "lite":    "TIER_LITE",
    "minimal": "TIER_MINIMAL",
}

# JSON (bench-abbreviated) algo key -> canonical grid:: host-launcher symbol.
# DERIVED from the per-algo descriptor table (algo_registry.ALGO_DESCRIPTORS):
# {autotune_key: descriptor.key} for every descriptor carrying a launch_cfg. The
# launch_configs JSON inherits the autotune-sweep's short keys; this maps them to
# the host-launcher / kernel base names so the baked enum is unambiguous. Algos
# present in the registry but absent from the autotune sweep simply get no baked
# override (they fall back to the conservative default). Built once at import.
_ALGO_TO_SYMBOL = build_launch_config_algo_to_symbol()


def _launch_configs_dir():
    """Absolute path to the repo's launch_configs/ dir (sibling of GRiDCodeGenerator)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "launch_configs")


def load_launch_config(robot_id, floating_base, gpu = LAUNCH_CONFIG_DEFAULT_GPU, profile = "host"):
    """Return {grid_symbol: {"tier": TIER_*, "threads": int}} for (robot_id, base).

    Reads launch_configs/<robot_id>/<gpu>.json and picks the matching base
    ("floating" or "fixed"). Returns {} (-> full fallback) when the file is
    absent / unreadable / lacks the base. Unknown algo keys / tiers are skipped
    individually (so a partially-populated config still bakes what it can).

    `profile` selects WHICH autotune the binding/build wants — the optimal
    thread count is launch-path AND use-case dependent (see autotune_ffi.py):
      * "host" (default): the C++/host launch path, throughput-optimal — the
        `bases` block, baked by the C++ run.py autotune.
      * "ffi":  the jax/torch FFI launch path, batch-to-land (single batched
        launch, wait for all N to land) — the `ffi_bases` block, baked by
        autotune_ffi.py. The SAME kernel has a DIFFERENT thread optimum under
        the FFI launch path (e.g. iiwa14 fd: host best=128, FFI best=768), so a
        binding that inherits the host pick runs ~1.6x slow. Per-algo fallback
        to `bases` when an algo has no FFI entry (partial FFI tuning is fine)."""
    if not robot_id:
        return {}
    path = os.path.join(_launch_configs_dir(), str(robot_id), str(gpu) + ".json")
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return {}
    base = "floating" if floating_base else "fixed"
    host_block = (doc.get("bases") or {}).get(base) or {}
    # Any non-host profile overlays its <profile>_bases on top of the host bases
    # (per-algo fallback): "ffi" -> ffi_bases (jax), "torch" -> torch_bases,
    # "pybind" -> pybind_bases. An unknown profile or a missing block leaves the
    # host bases untouched -> byte-identical to an un-tuned robot. (ffi behavior is
    # unchanged since "ffi" -> "ffi_bases".)
    if profile != "host":
        overlay = (doc.get(str(profile) + "_bases") or {}).get(base) or {}
        base_block = dict(host_block)
        base_block.update(overlay)
    else:
        base_block = host_block
    out = {}
    for algo_key, cfg in base_block.items():
        symbol = _ALGO_TO_SYMBOL.get(algo_key)
        if symbol is None:
            continue
        tier_sym = LAUNCH_CONFIG_TIER_SYMBOL.get(str(cfg.get("tier", "")).lower())
        threads = cfg.get("threads")
        if tier_sym is None or not isinstance(threads, int) or threads < 1:
            continue
        out[symbol] = {"tier": tier_sym, "threads": int(threads)}
    return out


class GRiDCodeGenerator:
    # first import helpers to write code generation, spatial algebra, and opology helpers (parent, child, Sind, XImats) and the robotModel object wrapepr
    from .helpers import gen_add_code_line, gen_add_code_lines, gen_add_end_control_flow, gen_add_end_function, \
                         gen_add_func_doc, gen_add_serial_ops, gen_add_parallel_loop, gen_minv_apply, gen_add_sync, gen_var_in_list, \
                         gen_var_not_in_list, gen_add_multi_threaded_select, gen_kernel_load_inputs, gen_kernel_save_result, \
                         gen_anti_licm_input_reload, gen_anti_licm_output_write, \
                         gen_static_array_ind_2d, gen_static_array_ind_3d, gen_add_debug_print_code_lines, \
                         gen_mx_func_call_for_cpp, gen_add_shared_memory_helpers, gen_declare_shared_arena, \
                         gen_shared_arena_t_count, gen_device_wrapper, gen_tier_dispatch, gen_spatial_algebra_helpers, \
                         gen_get_XI_size, gen_init_XImats, gen_get_inertia_params_size, gen_init_inertia_params, gen_set_inertia_params, \
                         gen_get_transform_params_size, gen_init_transform_params, gen_set_transform_params, \
                         _joint_dynamics_folded_by_vslot, gen_get_joint_dynamics_params_size, \
                         gen_init_joint_dynamics_params, gen_set_joint_dynamics_params, \
                         gen_load_update_XImats_helpers_temp_mem_size, gen_load_update_XImats_helpers_function_call, \
                         gen_XImats_helpers_temp_shared_memory_code, gen_load_update_XImats_helpers, gen_topology_helpers_size, \
                         gen_get_Xhom_size, gen_load_update_XmatsHom_helpers, gen_load_update_XmatsHom_helpers_function_call, gen_XmatsHom_helpers_temp_shared_memory_code, gen_load_topology_helpers, \
                         gen_topology_sparsity_helpers_python, gen_init_topology_helpers, gen_topology_helpers_pointers_for_cpp, \
                         gen_topology_S_sign_for_cpp, gen_insert_helpers_function_call, gen_insert_helpers_func_def_params, gen_init_robotModel, gen_free_robotModel, gen_joint_limits_size, gen_init_joint_limits, \
                         gen_grid_linalg_backend_helpers, gen_linalg_smem_setup, gen_invert_matrix, gen_matmul, gen_matmul_trans, gen_crm_mul, gen_crm, gen_mxS_general, gen_outer_product, custom_is_constant, \
                         gen_mjx_input_convert, gen_mjx_quat_reorder, gen_mjx_base_rotate, gen_mjx_base_rotate_rows, gen_mjx_symmetrize_full, gen_mjx_accel_out, gen_mjx_congruence, gen_mjx_column_reframe, gen_mjx_retract, \
                         robot_has_mimic_joints, _v_slot_cpp, _alpha_for_jid, _alpha_prefix_cpp, _id_S_desc

    # then import all of the algorithms
    from .algorithms import gen_inverse_dynamics_inner_temp_mem_size, gen_inverse_dynamics_inner_function_call, \
                            gen_inverse_dynamics_device_temp_mem_size, gen_inverse_dynamics_inner, gen_inverse_dynamics_device, \
                            gen_inverse_dynamics_joint_dynamics_bias, \
                            gen_inverse_dynamics_kernel, gen_inverse_dynamics_host, gen_inverse_dynamics, \
                            gen_inverse_dynamics_regressor_inner_temp_mem_size, gen_inverse_dynamics_regressor_inner_function_call, \
                            gen_inverse_dynamics_regressor_inner, gen_inverse_dynamics_regressor_device_temp_mem_size, \
                            gen_inverse_dynamics_regressor_device, gen_inverse_dynamics_regressor_kernel, \
                            gen_inverse_dynamics_regressor_host, gen_inverse_dynamics_regressor, \
                            gen_kinetic_energy_regressor_inner_temp_mem_size, gen_kinetic_energy_regressor_inner_function_call, \
                            gen_kinetic_energy_regressor_inner, gen_kinetic_energy_regressor_device, gen_kinetic_energy_regressor_kernel, \
                            gen_kinetic_energy_regressor_host, gen_kinetic_energy_regressor, \
                            gen_potential_energy_regressor_inner_function_call, gen_potential_energy_regressor_inner, \
                            gen_potential_energy_regressor_device, gen_potential_energy_regressor_kernel, \
                            gen_potential_energy_regressor_host, gen_potential_energy_regressor, \
                            gen_forward_dynamics_parameter_gradient_inner_temp_mem_size, gen_forward_dynamics_parameter_gradient_inner_function_call, \
                            gen_forward_dynamics_parameter_gradient_inner, gen_forward_dynamics_parameter_gradient_device_temp_mem_size, \
                            gen_forward_dynamics_parameter_gradient_device, gen_forward_dynamics_parameter_gradient_kernel, \
                            gen_forward_dynamics_parameter_gradient_host, gen_forward_dynamics_parameter_gradient, \
                            gen_minv_inner_temp_mem_size, gen_minv_inner_F_size, gen_minv_inner_no_F_size, gen_minv_inner_function_call, gen_minv_inner, \
                            gen_minv_device, gen_minv_kernel, gen_minv_host, gen_minv, \
                            gen_forward_dynamics_inner_temp_mem_size, gen_forward_dynamics_inner_F_size, gen_forward_dynamics_finish_function_call, gen_forward_dynamics_finish, \
                            gen_forward_dynamics_inner_function_call, gen_forward_dynamics_inner, gen_forward_dynamics_device, \
                            gen_forward_dynamics_kernel, gen_forward_dynamics_host, gen_forward_dynamics, \
                            gen_inverse_dynamics_gradient_inner_temp_mem_size, gen_inverse_dynamics_gradient_temp_layout, _emit_fb_bfs_level_indexing, \
                            gen_inverse_dynamics_gradient_kernel_max_temp_mem_size, \
                            gen_inverse_dynamics_gradient_inner_function_call, gen_inverse_dynamics_gradient_inner, \
                            gen_inverse_dynamics_gradient_device, gen_inverse_dynamics_gradient_device_function_call, \
                            gen_inverse_dynamics_gradient_kernel, gen_inverse_dynamics_gradient_host, gen_inverse_dynamics_gradient, \
                            gen_forward_dynamics_gradient_inner_temp_mem_size, gen_forward_dynamics_gradient_kernel_max_temp_mem_size, \
                            gen_forward_dynamics_gradient_inner_python, gen_forward_dynamics_gradient_kernel, \
                            gen_forward_dynamics_gradient_device, gen_forward_dynamics_gradient_device_function_call, \
                            gen_forward_dynamics_gradient_host, gen_forward_dynamics_gradient, \
                            gen_f_ext_gradient_inner_temp_mem_size, gen_f_ext_gradient_inner_function_call, \
                            gen_f_ext_gradient_jacobianT_inner, gen_f_ext_gradient_output_size, gen_f_ext_gradient_device, \
                            gen_f_ext_gradient_dq_kernel, gen_f_ext_gradient_dq_host, \
                            gen_f_ext_gradient_kernel, gen_f_ext_gradient_host, gen_f_ext_gradient, \
                            gen_end_effector_pose_inner_temp_mem_size, gen_end_effector_pose_inner_function_call, gen_end_effector_pose_inner, \
                            gen_end_effector_pose_device_temp_mem_size, gen_end_effector_pose_device, gen_end_effector_pose_kernel, \
                            gen_end_effector_pose_host, gen_end_effector_pose_gradient_inner_temp_mem_size, gen_end_effector_pose_gradient_inner_function_call, \
                            gen_end_effector_pose_gradient_inner, gen_end_effector_pose_gradient_device, gen_end_effector_pose_gradient_kernel, \
                            gen_end_effector_pose_gradient_host, gen_end_effector_pose_hessian_output_count, gen_end_effector_pose_hessian_inner_temp_mem_size, gen_end_effector_pose_hessian_inner_function_call, \
                            gen_end_effector_pose_hessian_inner, gen_end_effector_pose_hessian_device, gen_end_effector_pose_hessian_kernel, gen_ee_pose_inner_thread, gen_ee_pose_inner_warp, \
                            gen_ee_pose_inner_xform_from_q_lines, gen_ee_pose_inner_parent_lookup, gen_update_XmatHom_joint, \
                            gen_ee_pose_fk_batched_kernel, gen_ee_pose_fk_batched_host, \
                            gen_end_effector_pose_hessian_host, gen_eepose_and_derivatives, gen_ee_target_aliases, \
                            gen_aba, gen_aba_inner, gen_aba_host, \
                            gen_aba_inner_function_call, gen_aba_kernel, gen_aba_device, gen_aba_inner_temp_mem_size, gen_aba_inner_cold_mem_size, \
                            gen_crba, gen_crba_inner_temp_mem_size, gen_crba_inner_function_call, gen_crba_inner, gen_crba_device_temp_mem_size, \
                            gen_crba_device, gen_crba_kernel, gen_crba_host, \
                            gen_idsva_so_xdown_plucker_inverse, gen_idsva_so_reference_order_rt_rp_assembly, \
                            gen_idsva_so_body_frame_inner_temp_mem_size, gen_idsva_so_body_frame_inner_function_call, idsva_so_needs_reference_order_output_repair, \
                            gen_idsva_so_body_frame_reference_order_output_repair, gen_idsva_so_body_frame_floating_reference_inner, gen_idsva_so_body_frame_public_dvdq_layout_repair, gen_idsva_so_body_frame_inner, \
                            gen_idsva_so_body_frame_kernel, gen_idsva_so_body_frame_host, gen_idsva_so_body_frame, \
                            gen_idsva_so_world_frame_temp_mem_size, gen_idsva_so_world_frame_inner, gen_idsva_so_world_cold_floats, \
                            gen_idsva_so_world_frame_inner_function_call, gen_idsva_so_world_frame_kernel, \
                            gen_idsva_so_world_frame_host, gen_idsva_so_world_frame, \
                            gen_idsva_so_device, gen_idsva_so_dispatcher_host, gen_idsva_so_dispatcher, \
                            gen_floating_gravity_d2tau_dq_temp_mem_size, gen_floating_gravity_d2tau_dq_shared_count, \
                            gen_floating_gravity_d2tau_dq_spill_count, gen_floating_gravity_d2tau_dq_lie_inline, \
                            gen_fdsva_so, gen_fdsva_so_contract_temp_mem_size, gen_fdsva_so_fd_gradient_inline_temp_mem_size, gen_fdsva_so_fd_gradient_inline_temp_mem_size_spilled, gen_fdsva_so_fd_gradient_inline, gen_fdsva_so_contract_function_call, gen_fdsva_so_contract, \
                            gen_fdsva_so_device, gen_fdsva_so_device_function_call, gen_fdsva_so_kernel, gen_fdsva_so_host, \
                            gen_integrator_inner_temp_mem_size, gen_integrator_finish_function_call, gen_integrator_finish, \
                            _spherical_retract_index_tables, _emit_q_update, gen_integrate_spherical_helper, \
                            gen_integrator_inner_function_call, gen_integrator_inner, gen_integrator_device, \
                            gen_integrator_kernel, gen_integrator_host, gen_integrator, gen_lie_group_helpers, \
                            gen_integrator_gradient_inner_temp_mem_size, gen_integrator_gradient_dAB_assembly, \
                            gen_integrator_gradient_inner_python, gen_integrator_gradient_multistage, \
                            gen_integrator_gradient_device, gen_integrator_gradient_device_function_call, \
                            gen_integrator_gradient_kernel, gen_integrator_gradient_host, gen_integrator_gradient, \
                            gen_integrator_hessian_device, gen_integrator_hessian_device_function_call, \
                            gen_plant_step, gen_plant_step_gradient, gen_plant_step_hessian, gen_plant_step_hessian_kernel, gen_quadratic_state_cost, gen_quadratic_input_cost, \
                            gen_ee_pos_cost, gen_plant_barriers, gen_grid_plant, \
                            gen_plant_step_kernel, gen_quadratic_cost_kernel, gen_ee_pos_cost_kernel, gen_plant_kernels, \
                            gen_id_bias_device, gen_id_bias_kernel, gen_id_bias_host, gen_id_bias, \
                            gen_centroidal_inner, gen_com_device, gen_ccrba_device, gen_energy_device, \
                            _gen_kin_centroidal_kernel, _gen_kin_centroidal_host, gen_com, gen_ccrba, gen_energy, \
                            gen_frame_jacobian_inner, gen_frame_jacobian_device, \
                            gen_frame_jacobian_kernel, gen_frame_jacobian_host, gen_frame_jacobian, \
                            gen_frame_jacobian_dot_device, gen_frame_jacobian_dot_kernel, \
                            gen_frame_jacobian_dot_host, gen_frame_jacobian_dot, \
                            gen_osc_inertia_device, gen_osc_inertia_kernel, \
                            gen_osc_inertia_host, gen_osc_inertia, \
                            gen_end_effector_pose_runtime_inner, gen_end_effector_pose_runtime_device, \
                            gen_end_effector_pose_runtime_kernel, gen_end_effector_pose_runtime_host, \
                            gen_end_effector_pose_runtime, \
                            gen_end_effector_pose_gradient_runtime_inner, gen_end_effector_pose_gradient_runtime_device, \
                            gen_end_effector_pose_gradient_runtime_kernel, gen_end_effector_pose_gradient_runtime_host, \
                            gen_end_effector_pose_gradient_runtime, _gen_runtime_host, \
                            gen_coriolis_matrix_inner_temp_mem_size, gen_coriolis_matrix_inner_function_call, \
                            gen_coriolis_matrix_inner, gen_coriolis_matrix_device, \
                            gen_coriolis_matrix_kernel, gen_coriolis_matrix_host, gen_coriolis_matrix, \
                            gen_f_ext_contact, gen_f_ext_contact_inner_temp_mem_size, \
                            build_target_batch, gen_multi_target_position_inner_temp_mem_size, \
                            gen_multi_target_position_inner_function_call, gen_multi_target_position_inner, \
                            gen_multi_target_position_device, gen_multi_target_position, \
                            gen_multi_target_position_gradient_inner_temp_mem_size, gen_multi_target_position_gradient_inner, \
                            gen_multi_target_position_gradient_inner_function_call, gen_multi_target_position_gradient_device, \
                            gen_multi_target_position_gradient, \
                            gen_multi_target_position_kernel, gen_multi_target_position_host, \
                            gen_multi_target_position_gradient_kernel, gen_multi_target_position_gradient_host, \
                            gen_multi_target_position_bench
    from .algorithms._collision import gen_collision_namespace
    from .algorithms._dccrba import _dccrba_inner_temp_mem_size, _dccrba_sweep_J_count, gen_cmm_time_variation, gen_dccrba

    # finally import the test code
    from ._test import test_rnea_fpass, test_rnea_bpass, test_rnea, test_minv_bpass, test_minv_fpass, test_densify_Minv, test_minv, test_rnea_grad_inner, \
                      test_rnea_grad, test_fd_grad, mx0, mx1, mx2, mx3, mx4, mx5, mx, mxS, mxv, fx, fxS, fxv

    # initialize the object
    def __init__(self, robotObj, DEBUG_MODE = False, NEED_PRINT_MAT = False, USE_DYNAMIC_SHARED_MEM = True, FILE_NAMESPACE = "grid", USE_JOINT_DYNAMICS = False, dtype = "float", MUJOCO_OUTPUT = False, LAUNCH_CONFIG_ROBOT = None, LAUNCH_CONFIG_PROFILE = "host", runtime_joint_dynamics = False):
        self.robot = robotObj
        # runtime_joint_dynamics: when True, the id/fd/aba/*_gradient bias reads
        # the per-DOF damping/friction coefficients from a mutable device table
        # (set_joint_dynamics_params) instead of baking them as literals, so they
        # can be poked at runtime (sysID / domain randomization) with no recompile.
        # The table is initialized from the URDF (the alpha-FOLDED per-v-slot
        # coefficient), so the runtime path is BIT-IDENTICAL to the baked path
        # until poked. DEFAULT False keeps the baked path byte-identical (the
        # struct field + the device reads are entirely flag-gated). Set here so the
        # bias/gradient emitters (which read getattr(self,"runtime_joint_dynamics"))
        # see it; gen_all_code can override it per call.
        self.runtime_joint_dynamics = runtime_joint_dynamics
        # Which autotune profile to bake into launch_cfg<ALGO>::{TIER,THREADS}.
        # "host" = the C++/host throughput-optimal `bases` (default; used by the
        # C++ harness + numpy/pybind). "ffi" = the jax/torch FFI batch-to-land
        # optimum `ffi_bases` (used by the python bindings — the same kernel has a
        # different thread optimum under the FFI launch path; see autotune_ffi.py
        # + load_launch_config). Falls back per-algo to host when ffi is absent.
        self.launch_config_profile = LAUNCH_CONFIG_PROFILE
        # A1b launch-config bake: the robot id used to locate the autotuned
        # launch_configs/<robot>/<DEFAULT_GPU>.json (per-algo {tier,threads}).
        # The launch_configs dir is keyed by the URDF FILENAME stem (e.g.
        # "iiwa14", "go2", "g1"), which does NOT always equal the URDF
        # <robot name=...> (e.g. go2_description, g1_29dof). Callers that know
        # the canonical robot id (the bindings + the bench/test harness) pass it
        # explicitly; otherwise we fall back to robotObj.name. None / no match
        # falls back to the conservative MAX_PERF_LEVEL_THREADS default so
        # un-tuned robots keep compiling exactly as before (additive, Gate-A safe).
        self.launch_config_robot = LAUNCH_CONFIG_ROBOT
        # MUJOCO_OUTPUT: when True AND the robot is floating-base, the generator
        # additionally INSTANTIATES the `MUJOCO_OUTPUT=true` variant of every
        # convention-sensitive floating kernel/host wrapper (the mjx output
        # convention: quaternion wxyz + global base-linear velocity; see
        # RBDReference/equivalents/mujoco_convention.py). The `template <..., bool
        # MUJOCO_OUTPUT = false>` parameter and its `if constexpr` epilogue are
        # ALWAYS emitted (so the default pin instantiation is byte-identical PTX);
        # this flag only controls whether the extra `true` instantiation is also
        # emitted, to avoid binary bloat on robots that never use mjx mode. No-op on
        # a fixed base (the epilogue is gated out at Python time, flag is inert).
        self.MUJOCO_OUTPUT = MUJOCO_OUTPUT and robotObj.floating_base
        # USE_JOINT_DYNAMICS: when True, the RNEA/FD value path emits the
        # joint-local <dynamics damping>/<dynamics friction> bias
        # (tau += damping*qd + friction*sign(qd)). DEFAULT False so the emitted
        # kernels stay consistent with bare Pinocchio's pin.rnea/pin.aba (which
        # ignore model.damping/friction in the value path) — the authoritative
        # CUDA-equivalence oracle. With the flag off the bias emit is skipped
        # entirely, so EVERY robot (damped or not) stays byte-identical to the
        # historical emit. Pair with RBDReference(use_joint_dynamics=True) to get
        # a matching oracle when the term is enabled.
        self.USE_JOINT_DYNAMICS = USE_JOINT_DYNAMICS
        # planar joints PARSE (URDFParser groundwork) but the CUDA codegen
        # transform chain does not yet emit them — fail loudly here rather than
        # silently mis-generating. (Planar is decomposed into cardinal sub-joints
        # at parse time, so a surviving type=="planar" means the decomposition
        # was bypassed.) SPHERICAL is now supported for inverse_dynamics ONLY
        # (Tier-C, first slice); the per-algorithm guard in gen_all_code rejects
        # spherical for not-yet-ported algorithms (crba/aba/fd/gradients/SO/...).
        # See docs/open-tasks/joint_types_plan.md (backlog A2).
        _unsupported = {
            jt for jt in robotObj.get_joint_types_by_id().values()
            if jt in ("planar",)
        }
        if _unsupported:
            raise NotImplementedError(
                f"GRiD codegen does not yet support joint type(s) {sorted(_unsupported)}; "
                "they parse as groundwork but have no CUDA transform / RBDReference model. "
                "See docs/open-tasks/joint_types_plan.md."
            )
        self.code_str = ""
        self.indent_level = 0
        self.DEBUG_MODE = DEBUG_MODE
        self.gen_print_mat = DEBUG_MODE or NEED_PRINT_MAT
        # even if dynamic shared mem is not requested for large robots we need to use it
        self.use_dynamic_shared_mem_flag = USE_DYNAMIC_SHARED_MEM or (self.robot.get_num_pos() > 12)
        # check for the file/namespace name
        self.file_namespace = FILE_NAMESPACE
        self.cuda_target_shared_mem_bytes = int(os.environ.get("GRID_CUDA_TARGET_SHARED_MEM_BYTES", "98304"))
        # LITE tier: pick the lowest-spill level whose arena ≤ this target.
        # 48 KB is roughly half the sm_120 ~100 KB per-block cap, so inline-CUDA
        # callers retain ~48 KB for their own outer-kernel scratch.
        self.cuda_target_lite_shared_mem_bytes = int(os.environ.get("GRID_CUDA_TARGET_LITE_SHARED_MEM_BYTES", "49152"))
        # fp64 (Phase 8): dtype="double" doubles every shared-arena T-region, so the
        # codegen-time spill-tier pick (py_arena_bytes / select_shared_tier_3way) must
        # size T at sizeof(double)=8. The fp32 default stays at 4 -> byte-identical.
        # The emitted grid_shared_arena_bytes<T>() / device-cap fit-check / cudaFuncSetAttribute
        # are already sizeof(T)-correct, so this is the ONLY codegen byte constant to flip.
        # The env var still overrides (e.g. to force a spill tier on a small robot); dtype
        # sets the default. ints stay 32-bit regardless (int_count term is *4 below).
        if dtype not in ("float", "double"):
            raise ValueError(f"GRiDCodeGenerator dtype must be 'float' or 'double', got {dtype!r}")
        self.codegen_dtype = dtype
        _default_t_bytes = "8" if dtype == "double" else "4"
        self.cuda_shared_mem_type_size_bytes = int(os.environ.get("GRID_CUDA_SHARED_MEM_TYPE_SIZE_BYTES", _default_t_bytes))

    def get_joint_dynamics_baked(self):
        """Return the baked (alpha-FOLDED, per-v-slot) damping/friction the
        runtime_joint_dynamics device table is initialized with.

        Returns (damping, friction): two length-nv python float lists, v-slot
        indexed, IDENTICAL to what init_joint_dynamics_params writes into
        d_joint_dynamics_params (both call _joint_dynamics_folded_by_vslot). The
        bindings (_compile.py) persist these as meta so handle.joint_damping /
        .joint_friction echo EXACTLY the device init, and the codegen + meta agree
        by construction.
        """
        return self._joint_dynamics_folded_by_vslot()

    def _normalize_codegen_algorithms(self, codegen_profile = "all", algorithm_list = None):
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
        if "aba" in algorithms and self.robot.floating_base:
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
        if "minv" in algorithms and (self.robot_has_mimic_joints()
                                     or self.robot.robot_has_spherical()):
            algorithms.add("crba")
        # idsva_so_world_frame is emitted alongside idsva_so_body_frame (it
        # reuses the body-frame inner scaffolding + dispatcher); requesting it
        # pulls in body_frame so the world-frame emit block (gated under
        # body_frame) actually runs.
        if "idsva_so_world_frame" in algorithms:
            algorithms.add("idsva_so_body_frame")
        if "idsva_so_body_frame" in algorithms:
            algorithms.add("inverse_dynamics")
            if self.robot.floating_base:
                algorithms.add("inverse_dynamics_gradient")
            # Spherical (ball) joints route the idsva_so dispatcher to the WORLD
            # frame (the body-frame inner's single-DoF S contractions are wrong
            # for a 3-DoF joint). Force the world-frame emit so the routed
            # dispatcher call links. Gated on spherical => cardinal byte-identical.
            if self.robot.robot_has_spherical():
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
        if "forward_dynamics_gradient" in algorithms and self.robot.floating_base:
            algorithms.add("crba")
        # The minv->crba spherical/mimic expansion above runs BEFORE integrator
        # pulls in `minv` (line ordering), so a spherical robot requesting ONLY
        # `integrator` would add minv here WITHOUT crba_inner being emitted ->
        # nvlink: unresolved extern crba_inner. Re-assert the crba dependency
        # after the integrator (and any other late minv-adders) have run. Additive
        # for cardinal robots (the guard is false), byte-identical.
        if "minv" in algorithms and (self.robot_has_mimic_joints()
                                     or self.robot.robot_has_spherical()):
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
    
    # add generic code needs and helpers (includes, memory initialization, constants, kernel settings etc.)
    def gen_add_includes(self):
        # first all of the includes
        self.gen_add_code_line("")
        self.gen_add_code_line("#include <assert.h>")
        self.gen_add_code_line("#include <cstddef>")
        self.gen_add_code_line("#include <stdint.h>")
        self.gen_add_code_line("#include <stddef.h>")
        self.gen_add_code_line("#include <math.h>")
        self.gen_add_code_line("#include <stdio.h>")
        self.gen_add_code_line("#include <stdlib.h>")
        self.gen_add_code_line("#include <string.h>")
        self.gen_add_code_line("#include <time.h>")
        self.gen_add_code_line("#include <cuda_runtime.h>")
        self.gen_add_code_lines([
            "",
            "// SIMT GLASS is the only linalg backend (cuBLASDx was removed in v2.0;",
            "// see docs/source/user_guide/concepts/cublasdx_removal_design.rst).",
            "#if defined(__has_include)",
            "#if __has_include(<cub/cub.cuh>)",
            "#define GRID_CUB_HEADER_AVAILABLE 1",
            "#else",
            "#define GRID_CUB_HEADER_AVAILABLE 0",
            "#endif",
            "#else",
            "#define GRID_CUB_HEADER_AVAILABLE 0",
            "#endif",
        ])
        # then any namespaces
        # then any #defines
        self.gen_add_code_lines(["// single kernel timing helper code", \
            "#define time_delta_us_timespec(start,end) (1e6*static_cast<double>(end.tv_sec - start.tv_sec)+1e-3*static_cast<double>(end.tv_nsec - start.tv_nsec))"])
        self.gen_add_code_line("")
        self.gen_add_code_line("#define XIMAT_SIZE 36")

    def gen_add_launch_config_helpers(self):
        """Emit the A1b baked launch-config table (single source of truth).

        Reads launch_configs/<robot>/<DEFAULT_GPU>.json for this robot+base and
        emits per-algo `grid::launch_cfg<GRID_ALGO_*>` specializations carrying
        the autotuned {TIER, THREADS}. The primary template falls back to the
        conservative (GRID_DEFAULT_RESOURCE_TIER, MAX_PERF_LEVEL_THREADS) default,
        so any algo without a baked entry — and any robot/GPU without a config —
        compiles exactly as before. Purely additive host-side content: no kernel
        body is emitted here (Gate-A: kernel `__global__` bodies byte-identical).

        HOST launchers / bindings read grid::launch_cfg<ALGO>::{TIER,THREADS} to
        default their launch config, fixing the FFI thread-default pathology at
        the C++ root. Explicit caller-supplied threads still override."""
        robot_id = self.launch_config_robot if self.launch_config_robot is not None else self.robot.get_name()
        profile = getattr(self, "launch_config_profile", "host")
        cfg = load_launch_config(robot_id, self.robot.floating_base, profile=profile)
        # Stable, declaration-ordered list of every algo that COULD carry a config
        # (the canonical grid:: symbols). Emit one enumerator per algo so the
        # table is complete regardless of which algos this robot tuned.
        # Driven from the descriptor table: every algo carrying a baked launch_cfg,
        # in descriptor order (which is aligned to the launch-config enum order).
        launch_descriptors = [d for d in ALGO_DESCRIPTORS if d.carries_launch_cfg]
        algo_symbols = [d.key for d in launch_descriptors]
        enum_names = {d.key: d.enum_name for d in launch_descriptors}
        base_name = "floating" if self.robot.floating_base else "fixed"
        if cfg:
            src = "launch_configs/" + str(robot_id) + "/" + LAUNCH_CONFIG_DEFAULT_GPU + ".json (" + base_name + ", profile=" + profile + ")"
        else:
            src = "NONE found for robot=" + str(robot_id) + " base=" + base_name + " gpu=" + LAUNCH_CONFIG_DEFAULT_GPU + " profile=" + profile + " -> conservative fallback"
        self.gen_add_code_lines([
            "",
            "// ─── A1b baked launch config (single source of truth) ───────────────",
            "// Autotuned per-algo {resource tier, threads-per-block} for THIS robot",
            "// + base, baked from " + src + ".",
            "// GRiD kernels are single-block + thread-count-invariant, so (tier,threads)",
            "// is a pure PERFORMANCE choice; HOST launchers / python-jax-torch bindings",
            "// default their launch config from grid::launch_cfg<GRID_ALGO_*>. An algo",
            "// with no autotuned entry (or a robot/GPU with no launch_configs file)",
            "// falls back to the conservative (GRID_DEFAULT_RESOURCE_TIER,",
            "// MAX_PERF_LEVEL_THREADS) default -> un-tuned robots are unaffected.",
            "enum GridAlgo {",
        ])
        for sym in algo_symbols:
            self.gen_add_code_line("    " + enum_names[sym] + ",", )
        self.gen_add_code_lines([
            "    GRID_ALGO_COUNT",
            "};",
            "// Primary template = conservative fallback (matches the historical default).",
            "template <int ALGO> struct launch_cfg {",
            "    static constexpr int TIER    = GRID_DEFAULT_RESOURCE_TIER;",
            "    static constexpr int THREADS = MAX_PERF_LEVEL_THREADS;",
            "};",
        ])
        # Per-algo specializations (only for algos with a baked entry).
        # THREADS is CLAMPED to the tier's __launch_bounds__ (tier_max_threads<TIER>):
        # an autotune may report a count above the tier's launch_bounds because the
        # jax/torch FFI path clamps at launch (grid_clamp_threads_for), but the
        # numpy/pybind host-wrapper path passes the count RAW — an unclamped value >
        # launch_bounds is cudaErrorInvalidValue ("invalid argument"). launch_bounds
        # guarantees tier_max_threads threads are always launchable, so the clamp is
        # exact for the FFI path (a no-op there) and correct for the host path.
        for sym in algo_symbols:
            entry = cfg.get(sym)
            if entry is None:
                continue
            n = str(entry["threads"])
            tmax = "tier_max_threads<" + entry["tier"] + ">()"
            self.gen_add_code_line(
                "template <> struct launch_cfg<" + enum_names[sym] + "> { "
                "static constexpr int TIER = " + entry["tier"] + "; "
                "static constexpr int THREADS = ((" + n + ") < " + tmax + ") ? (" + n + ") : " + tmax + "; };"
            )
        self.gen_add_code_line("")

    def gen_add_constants_helpers(self, include_base_inertia = False, include_homogenous_transforms = False):
        # first add constants
        n = self.robot.get_num_pos()
        nv = self.robot.get_num_vel()
        NJ = self.robot.get_num_joints()
        # Dynamics kernels only need the spatial X/I storage. Homogeneous transforms
        # are accounted separately for kinematics kernels.
        XI_size = self.gen_get_XI_size(include_base_inertia,include_homogenous_transforms=False)
        # runtime_transform: the load_update_XImats helper rebuilds each joint's
        # constant 6x6 Xfixed into s_temp at offset _runtime_transform_xfixed_offset
        # (= 2*num_pos non-mimic / 3*NB mimic), occupying 36*NB extra floats. That
        # block is consumed ENTIRELY within the helper (the hot loop reads it to
        # build s_XImats) and is DEAD after the helper returns, so each algorithm's
        # inner-temp scratch may reuse the region afterward. A purely ADDITIVE
        # reservation of 36*NB floats in every XImats-domain (s_temp-backed) arena
        # t_count is therefore sufficient to stop the OOB AND keep correctness; the
        # baked path keeps rt_xfixed_reserve == 0 so its arena/header stays
        # byte-identical. Only the s_temp-backed (dynamics, XImats) arenas need it;
        # the XmatsHom/kinematics arenas don't invoke the Xfixed rebuild. NOT added
        # to XI_size itself (that sizes the s_XImats array + DYNAMICS_XI_T_COUNT and
        # would corrupt the XImats layout) — it is appended to the temp region.
        rt_xfixed_reserve = (36 * NJ) if getattr(self, "runtime_transform", False) else 0
        XHom_size, dXhom_size, d2Xhom_size = self.gen_get_Xhom_size()
        # Descriptor-table Step 3: build the ArenaCtx snapshot from the exact sizing
        # locals so the per-algo `arena_full_fn` closures (algo_registry) can drive the
        # FULL/rung-0 arena t_counts. Each fold keeps the hand-written legacy expression
        # behind `assert composed == legacy` (the parity shim, deleted in 3.6). The
        # inner-temp helpers ArenaCtx reads are pure robot-shape functions, so building
        # the snapshot here (before the arena math) matches their mid-function values —
        # proven by test/test_algo_descriptor_arena_parity.py.
        self._arena_ctx = arena_ctx_from_codegen(self, xi=XI_size, xhom=XHom_size, rt=rt_xfixed_reserve)
        dva_cols_per_partial = self.robot.get_total_ancestor_count() + self.robot.get_num_joints()
        max_threads_in_comp_loop = 6*2*dva_cols_per_partial
        max_perf_level_threads = 32 * int(np.ceil(max_threads_in_comp_loop/32.0))
        # cap to 512 mirrors the constant we emit further down; expose on self so
        # _lin_alg_helpers can pin cuBLASDx's BlockDim<TC,1,1> to the same value.
        self.max_perf_level_threads = min(max_perf_level_threads, 512)
        topology_count = self.gen_topology_helpers_size()
        def py_align_up(offset, alignment):
            return ((offset + alignment - 1) // alignment) * alignment

        def py_arena_bytes(t_count, int_count = topology_count):
            offset = 0
            t_align = max(1, min(self.cuda_shared_mem_type_size_bytes, 8))
            offset = py_align_up(offset, t_align)
            offset += self.cuda_shared_mem_type_size_bytes * int(t_count)
            if int_count > 0:
                offset = py_align_up(offset, 4)
                offset += 4 * int(int_count)
            return py_align_up(offset, 16)

        def select_shared_tier_3way(*t_counts):
            """Pick the (perf, lite, minimal) spill-level indices for one algo.
            `t_counts` is the ordered list of arena t_counts at each spill level,
            least-spill first. PERF picks the lowest index whose arena fits
            cuda_target_shared_mem_bytes; LITE picks the lowest index whose
            arena fits cuda_target_lite_shared_mem_bytes (clamped to be ≥ PERF
            pick — LITE can't be less spill than PERF); MINIMAL is always the
            last (most-spill) index."""
            last = len(t_counts) - 1
            perf = next((i for i, t in enumerate(t_counts)
                         if py_arena_bytes(t) <= self.cuda_target_shared_mem_bytes), last)
            lite = next((i for i, t in enumerate(t_counts)
                         if py_arena_bytes(t) <= self.cuda_target_lite_shared_mem_bytes), last)
            lite = max(perf, lite)
            return (perf, lite, last)

        # The ID kernel's s_vaf band is body-indexed (18*NJ); for mimic robots
        # (NJ > n) size it 18*NJ so the inner's body f-writes don't overflow into
        # the XImats region. Non-mimic keeps the legacy 18*n byte-identical.
        _id_vaf = 18 * (self.robot.get_num_joints() if self.robot_has_mimic_joints() else n)
        id_t_count = compose_arena_full("inverse_dynamics", self._arena_ctx)   # Step 3.1 fold
        # joint-torque regressor (E1): kernel smem = XI + s_q_qd_qdd(NUM_POS+2nv)
        # + s_Y (nv x 10*NUM_BODIES) + s_vaf(18*NUM_POS) + RNEA forward scratch.
        # n == get_num_pos() here. Additive.
        regressor_t_count = (n + 2*nv) + nv*10*self.robot.get_num_bodies() + 18*n \
            + self.gen_inverse_dynamics_regressor_inner_temp_mem_size() + XI_size + rt_xfixed_reserve
        self.regressor_t_count = regressor_t_count
        # g1-spill: 2-rung s_Y output-spill ladder (mirror fdpg :795-802). The s_Y output
        # (nv*10*NB, ~277KB on h2_plus -> unlaunchable) routes to the L2-pinned d_workspace
        # SO section at any tier whose full arena overflows the target, keeping s_vaf +
        # inputs + XImats + RNEA scratch in smem. Small robots stay rung 0 (s_Y in smem).
        _idr_Y_count          = nv*10*self.robot.get_num_bodies()
        (_idr_t_count_full, _idr_t_count_surgical) = compose_arena_rungs("inverse_dynamics_regressor", self._arena_ctx)   # Step 3.5d fold
        self.inverse_dynamics_regressor_spill_tier_3way = select_shared_tier_3way(_idr_t_count_full, _idr_t_count_surgical)
        self.inverse_dynamics_regressor_t_count_per_tier = tuple(
            (_idr_t_count_full, _idr_t_count_surgical)[i] for i in self.inverse_dynamics_regressor_spill_tier_3way)
        self.inverse_dynamics_regressor_spill_Y_ws_count = _idr_Y_count
        # PS5 energy regressors. Both outputs are 10*NUM_BODIES (no DoF sweep).
        # KE (spatial / XImats domain): s_q_qd(n+nv) + s_y_ke(10NB) + s_vaf(18n)
        #   + RNEA forward scratch + XI_size.
        self.kinetic_energy_regressor_t_count = compose_arena_full("kinetic_energy_regressor", self._arena_ctx)   # Step 3.1 fold
        # PE (kinematics / XmatsHom domain): s_q(n) + s_y_pe(10NB)
        #   + world-transform BFS scratch (16*NUM_JOINTS) + XHom_size.
        self.potential_energy_regressor_t_count = compose_arena_full("potential_energy_regressor", self._arena_ctx)   # Step 3.1 fold
        # PS5 Coriolis matrix C(q,qd): kernel smem = XI + s_q_qd(NUM_POS+nv) + s_coriolis(nv*nv)
        #   + the inner spatial-recursion scratch (per-body NB bands + per-column n_int bands).
        # 3-rung surgical ladder (mirror crba): full | s_coriolis(nv*nv output) -> L2-pinned SO
        # band, hot inner band stays smem | whole-band (inner scratch -> d_workspace GRAD section,
        # output -> SO band). On h2_plus the inner band (18330 t ~73KB) dominates and the full
        # arena is ~121KB (UNLAUNCHABLE); the output-spill rung is ~95KB (PERF-launchable) and the
        # whole-band rung ~22KB (LITE/MINIMAL).
        _coriolis_inner   = self.gen_coriolis_matrix_inner_temp_mem_size()
        _coriolis_no_out  = (n + nv) + XI_size + rt_xfixed_reserve            # input + XI, no s_coriolis
        (_coriolis_t_full, _coriolis_t_output_spill, _coriolis_t_workspace) = \
            compose_arena_rungs("coriolis_matrix", self._arena_ctx)   # Step 3.3 fold
        self.coriolis_matrix_spill_tier_3way = select_shared_tier_3way(_coriolis_t_full, _coriolis_t_output_spill, _coriolis_t_workspace)
        # whole-band spill (inner scratch -> d_workspace) fires only at the deepest rung (index 2).
        self.coriolis_matrix_use_workspace_temp = self.coriolis_matrix_spill_tier_3way[0] == 2
        self.coriolis_matrix_t_count_per_tier = tuple(
            (_coriolis_t_full, _coriolis_t_output_spill, _coriolis_t_workspace)[i] for i in self.coriolis_matrix_spill_tier_3way)
        self.coriolis_matrix_t_count = self.coriolis_matrix_t_count_per_tier[0]
        # PS5 dCCRBA (kinematics / XmatsHom domain). The shared inner pool is the
        # SHRUNK (no-J) centroidal_inner pool + 6*n_int per-unit phi band
        # (== _dccrba_inner_temp_mem_size). The Jw sweep band (6*nv*NB) is carved as a
        # SEPARATE tier-routed buffer s_J (in-smem at L0/L1, d_workspace at the
        # J-spilled tier) -- DE-GATE #2: this is the cold/large quadratic buffer whose
        # spill de-gates big floating robots (g1/h1_2-floating).
        _dccrba_inner_temp = self._dccrba_inner_temp_mem_size()
        _dccrba_sJ = self._dccrba_sweep_J_count()   # 6*nv*NB
        # cmm_time_variation (Adot, 6*nv output). 2-rung ladder (the only lever is the
        # Jw band; its tiny 6*nv output never spills): L0 keeps s_J in smem, L1 spills
        # it to the d_workspace SO band.
        #   base = s_q_qd(2n) + s_out(6nv) + s_A(6nv) + s_com(3) + s_extra(4) + inner + XHom.
        _cmm_base = 2*n + 6*nv + 6*nv + 3 + 4 + _dccrba_inner_temp + XHom_size
        (_cmm_t_count_full, _cmm_t_count_Jspill) = compose_arena_rungs("cmm_time_variation", self._arena_ctx)   # Step 3.3 fold
        self.cmm_time_variation_spill_tier_3way = select_shared_tier_3way(_cmm_t_count_full, _cmm_t_count_Jspill)
        self.cmm_time_variation_t_count_per_tier = tuple(
            (_cmm_t_count_full, _cmm_t_count_Jspill)[i] for i in self.cmm_time_variation_spill_tier_3way
        )
        self.cmm_time_variation_spill_J_ws_count = _dccrba_sJ
        # dccrba (full 6*nv*nv tensor). 3-rung ladder: L0 keeps the s_dccrba output
        # (6*nv*nv) AND s_J in smem; L1 spills the output to d_workspace but keeps s_J
        # in smem (today's surgical rung on robots that fit it); L2 (NEW) spills BOTH
        # output and s_J -> d_workspace at distinct SO sub-offsets (the de-gating rung).
        #   base = s_q(n) + s_A(6nv) + s_com(3) + s_extra(4) + inner + XHom (no out, no s_J).
        _dccrba_out = 6 * nv * nv
        _dccrba_base = n + 6*nv + 3 + 4 + _dccrba_inner_temp + XHom_size
        (_dccrba_L0, _dccrba_L1, _dccrba_L2) = compose_arena_rungs("dccrba", self._arena_ctx)   # Step 3.3 fold
        self.dccrba_spill_tier_3way = select_shared_tier_3way(_dccrba_L0, _dccrba_L1, _dccrba_L2)
        self.dccrba_t_count_per_tier = tuple(
            (_dccrba_L0, _dccrba_L1, _dccrba_L2)[i] for i in self.dccrba_spill_tier_3way
        )
        self.dccrba_spill_out_ws_count = _dccrba_out
        self.dccrba_spill_J_ws_count = _dccrba_sJ
        # FD param gradient: kernel smem = XI + s_q_qd_u(NUM_POS+2nv) + s_dqdd_dpi
        # + s_Minv(nv*nv) + s_Y(nv x 10*NB) + s_qdd(nv) + s_vaf(18*NUM_POS) + s_c(nv)
        # + the (max) inner forward scratch. n == get_num_pos() here. Additive.
        forward_dynamics_parameter_gradient_t_count = (n + 2*nv) + nv*10*self.robot.get_num_bodies() \
            + nv*nv + nv*10*self.robot.get_num_bodies() + nv + 18*n + nv \
            + self.gen_forward_dynamics_parameter_gradient_inner_temp_mem_size() + XI_size + rt_xfixed_reserve
        self.forward_dynamics_parameter_gradient_t_count = forward_dynamics_parameter_gradient_t_count
        # FD-param-gradient g1-spill: 2-level surgical ladder. Level 0 keeps every
        # buffer in smem (current behavior on robots that fit). Level 1 spills the
        # s_Y regressor scratch (nv*10*NB, write-once / consumed-once in the final
        # -Minv.Y GEMM) to the L2-pinned d_workspace SO section -- the hot Minv +
        # vaf + inner-RNEA path stays in smem. On g1-floating this drops the arena
        # from ~135 KB to ~94 KB, under the sm_120 ~99 KB cap. The picker selects
        # level 1 for any tier whose level-0 arena overflows the smem target.
        _fpg_Y_count = nv * 10 * self.robot.get_num_bodies()
        (_fpg_t_count_full, _fpg_t_count_surgical) = compose_arena_rungs("forward_dynamics_parameter_gradient", self._arena_ctx)   # Step 3.5d fold
        self.forward_dynamics_parameter_gradient_spill_tier_3way = select_shared_tier_3way(_fpg_t_count_full, _fpg_t_count_surgical)
        self.forward_dynamics_parameter_gradient_t_count_per_tier = tuple(
            (_fpg_t_count_full, _fpg_t_count_surgical)[i] for i in self.forward_dynamics_parameter_gradient_spill_tier_3way
        )
        self.forward_dynamics_parameter_gradient_spill_Y_ws_count = _fpg_Y_count
        # f_ext gradient (section A): kernel smem = XI + s_q + the two nv x (6*NB)
        # outputs + temp (nv*nv s_Minv + max(J^T-inner, minv-inner) scratch).
        _n_pos = self.robot.get_num_pos()
        _NB = self.robot.get_num_bodies()
        _feg_out = nv * 6 * _NB
        _feg_temp = nv*nv + max(self.gen_f_ext_gradient_inner_temp_mem_size(),
                                self.gen_minv_inner_temp_mem_size())
        f_ext_gradient_t_count = _n_pos + 2*_feg_out + _feg_temp + XI_size + rt_xfixed_reserve
        # f_ext-gradient (first-order) g1/h2_plus-spill: 3-rung surgical ladder.
        #   rung 0 (full): both outputs (s_dtau_dfext, s_dqdd_dfext) + minv-F in smem.
        #   rung 1 (out-spill): spill s_dqdd_dfext (the SECOND output, written write-once
        #     by the final -Minv@s_dtau GEMM) to the L2-pinned d_workspace SO section;
        #     s_dtau_dfext (read by that GEMM) + s_Minv + minv-F stay in smem. On
        #     g1-floating the full arena is ~99.3 KB (272 B over the sm_120 cap), so this
        #     ~74.6 KB rung is what lets it run; mid robots stop here.
        #   rung 2 (deep): ALSO spill s_dtau_dfext (-> SO section) and route minv's
        #     6*nv*nv F-region to the GRAD-section minv-F workspace offset (F_IN_SMEM=
        #     false). On h2_plus (nv=81) rung 0/1 are ~506/361 KB (UNLAUNCHABLE); the deep
        #     rung is ~68 KB. s_Minv (nv*nv) + the J^T-inner + minv no-F scratch stay hot
        #     in smem. select picks the least-spill rung that fits, so small robots keep 0.
        _feg_temp_deep = nv*nv + max(self.gen_f_ext_gradient_inner_temp_mem_size(),
                                     self.gen_minv_inner_no_F_size())
        (_feg_t_count_full, _feg_t_count_out_spill, _feg_t_count_deep) = compose_arena_rungs("f_ext_gradient", self._arena_ctx)   # Step 3.5d fold
        self.f_ext_gradient_spill_tier_3way = select_shared_tier_3way(
            _feg_t_count_full, _feg_t_count_out_spill, _feg_t_count_deep)
        self.f_ext_gradient_t_count_per_tier = tuple(
            (_feg_t_count_full, _feg_t_count_out_spill, _feg_t_count_deep)[i] for i in self.f_ext_gradient_spill_tier_3way
        )
        # SO-section reservation: rung 1 holds 1 output, rung 2 holds BOTH (s_dqdd at
        # SO base, s_dtau at SO base + _feg_out). Reserve 2x when any tier reaches deep.
        self.f_ext_gradient_spill_out_ws_count = (2*_feg_out if any(p >= 2 for p in self.f_ext_gradient_spill_tier_3way) else _feg_out)
        # rung 2 routes minv's 6*nv*nv F-region to the GRAD-section minv-F offset.
        self.f_ext_gradient_spill_minv_F_count = self.gen_minv_inner_F_size() if any(p >= 2 for p in self.f_ext_gradient_spill_tier_3way) else 0
        # A.3 (-dJ^T/dq) FD kernel (both base modes): arena = XI + s_q + the FD
        # scratch (s_qpert | [floating: s_dv(nv)] | 2x J^T buffers | J^T-inner temp
        # | XImats reload). Floating base adds an nv-sized velocity-perturbation
        # buffer for the SE(3) Lie-group root retract (grid_integrate_floating_q).
        _feg_dq_dv = nv if self.robot.floating_base else 0
        _feg_dq_jt = 2*nv*6*_NB   # s_JTp | s_JTm: the two -J^T FD buffers (spillable pair)
        _feg_dq_extra = (_n_pos + _feg_dq_dv + _feg_dq_jt
                         + self.gen_f_ext_gradient_inner_temp_mem_size()
                         + self.gen_load_update_XImats_helpers_temp_mem_size())
        f_ext_gradient_dq_t_count = _n_pos + _feg_dq_extra + XI_size
        # h2_plus-spill: 2-rung ladder for the -dJ^T/dq FD kernel. rung 0 keeps both
        # nv x (6*NB) J^T buffers (s_JTp, s_JTm) in smem. rung 1 spills the PAIR to the
        # L2-pinned d_workspace SO section (s_JTp at SO base, s_JTm at SO base + nv*6NB);
        # s_qpert/s_dv/the J^T-inner scratch/XImats reload stay hot in smem. On h2_plus
        # the pair is ~288 KB (full arena ~311 KB, UNLAUNCHABLE); spilling it lands ~22 KB.
        (_feg_dq_t_count_full, _feg_dq_t_count_spill) = compose_arena_rungs("f_ext_gradient_dq", self._arena_ctx)   # Step 3.5d fold
        self.f_ext_gradient_dq_spill_tier_3way = select_shared_tier_3way(_feg_dq_t_count_full, _feg_dq_t_count_spill)
        self.f_ext_gradient_dq_t_count_per_tier = tuple(
            (_feg_dq_t_count_full, _feg_dq_t_count_spill)[i] for i in self.f_ext_gradient_dq_spill_tier_3way
        )
        self.f_ext_gradient_dq_spill_jt_ws_count = _feg_dq_jt
        # Minv Phase 3a: per-tier spill picks. Level 0 = F in smem (6*NV*NV
        # bytes); Level 1 = surgical F to L2-pinned workspace.
        _minv_F_count = self.gen_minv_inner_F_size()
        _minv_no_F_count = self.gen_minv_inner_no_F_size()
        (_minv_t_count_full, _minv_t_count_surgical) = compose_arena_rungs("minv", self._arena_ctx)   # Step 3.2 fold
        self.minv_spill_tier_3way = select_shared_tier_3way(_minv_t_count_full, _minv_t_count_surgical)
        self.minv_use_workspace_F = self.minv_spill_tier_3way[0] == 1
        minv_t_count = _minv_t_count_full if not self.minv_use_workspace_F else _minv_t_count_surgical
        self.minv_t_count_per_tier = tuple(
            (_minv_t_count_full, _minv_t_count_surgical)[i] for i in self.minv_spill_tier_3way
        )
        # FD inner-controlled placement: forward_dynamics_inner slices its own
        # Minv-F. The inner-temp size now bundles F (or not) per MINV_F_IN_SMEM,
        # so the full vs surgical kernel arenas come straight from the sized
        # helper (no separate F term — avoids double-counting). Level 0 = F in
        # smem; Level 1 = F in L2-pinned workspace.
        # Canonical input slot: q/qd/u each NUM_JOINTS(=nq=n here)-wide -> s_q_qd_u
        # is 3*n (matches _emit_fd_kernel_body_for_flags' ("s_q_qd_u", 3*nq)); qdd is
        # nv. For a FIXED base n==nv so 3*n == old 3*nv+fb byte-identical; FLOATING
        # n>nv so the arena must reserve the wider 3*n slot the kernel slices (the
        # old 3*nv+fb under-reserved by 3*(n-nv)-fb floats -> smem overrun).
        _fd_base = 3*n + nv + XI_size + rt_xfixed_reserve
        (_fd_t_count_full, _fd_t_count_surgical) = compose_arena_rungs("forward_dynamics", self._arena_ctx)   # Step 3.2 fold
        self.fd_spill_tier_3way = select_shared_tier_3way(_fd_t_count_full, _fd_t_count_surgical)
        self.fd_use_workspace_F = self.fd_spill_tier_3way[0] == 1
        fd_t_count = _fd_t_count_full if not self.fd_use_workspace_F else _fd_t_count_surgical
        self.fd_t_count_per_tier = tuple(
            (_fd_t_count_full, _fd_t_count_surgical)[i] for i in self.fd_spill_tier_3way
        )
        # Integrator: kernel-shared t-count layout is
        #   s_q_qd_u (3*nq) + s_qdd (nv) + s_stage_qdd ((max_stages-1)*nv)
        #   + s_stage_point ((max_stages-1)*(nq+nv)) + s_x_kp1 (nq+nv)
        #   + s_temp (= FD inner)
        # Canonical input slot: q/qd/u each NUM_JOINTS(=nq=n here)-wide -> 3*n; the
        # per-stage point + next state are [q (nq); qd (nv)] = n+nv. Must match
        # _emit_integrator_kernel_body_for_flags exactly. For a FIXED base n==nv so
        # 3*n == old 3*nv+fb and n+nv == old 2*nv+fb (byte-identical); FLOATING n>nv
        # so this reserves the wider input slot (old 3*nv+fb under-reserved -> overrun).
        # max_stages = 4 (RK4) — see _integrator._max_stages_in_use().
        _max_stages = 4
        _fb = int(self.robot.floating_base)
        _integrator_base = ((3*n) + nv
                            + (_max_stages - 1) * nv
                            + (_max_stages - 1) * (n + nv)
                            + (n + nv) + XI_size + rt_xfixed_reserve)
        integrator_t_count = _integrator_base + self.gen_forward_dynamics_inner_temp_mem_size()
        # Integrator VALUE surgical spill. The dominant inner buffer is the FD
        # inner's Minv F-region (6*NV*NV). Level 0 keeps it in smem; level 1
        # spills ONLY F to d_workspace (the hot FD path stays in smem), mirroring
        # the standalone forward_dynamics kernel's MINV_F_IN_SMEM lever. For
        # h1_2 the value arena overflows by only a few KB, so the surgical F
        # spill is enough — no whole-arena dump.
        (_integrator_t_count_full, _integrator_t_count_Fspill) = compose_arena_rungs("integrator", self._arena_ctx)   # Step 3.3 fold
        self.integrator_spill_tier_3way = select_shared_tier_3way(_integrator_t_count_full, _integrator_t_count_Fspill)
        self.integrator_t_count_per_tier = tuple(
            (_integrator_t_count_full, _integrator_t_count_Fspill)[i] for i in self.integrator_spill_tier_3way)
        # F float-count the value path spills (for grad-section sizing); 0 if no tier spills.
        self.integrator_minv_F_workspace_count = (self.gen_minv_inner_F_size()
                                                  if any(p == 1 for p in self.integrator_spill_tier_3way) else 0)
        # Integrator gradient: kernel-shared t-count layout is
        #   s_q_qd_u (3nv+fb) + s_dAB (2nv*3nv) + s_df_du (nv*2nv) + s_dc_du (nv*2nv) +
        #   s_vaf (18nv) + s_Minv (nv*nv) + s_qdd (nv)
        #   + multi-stage scratch: s_q_orig (nv+fb) + s_qd_orig (nv)
        #     + s_stage_grad_qdd (max_stages*nv) + s_D_qdd_stage (max_stages*nv*3nv)
        #   + s_temp (= FD-grad inner)
        # s_q_orig holds the FULL nq pose (floating-base adds the quaternion slot),
        # so it is nv+fb — must match _emit_body's ("s_q_orig", n+fb) exactly, else
        # the launched dynamic-smem (this t_count) is fb floats short of the arena
        # the kernel slices and the tail buffer overruns shared memory (floating only).
        # max_stages = 4 (RK4) — see _integrator._max_stages_in_use().
        # The multi-stage scratch is always allocated even for single-stage IT;
        # cost is small relative to total (~12*nv² for iiwa14 ≈ 588 floats).
        _max_stages = 4
        # s_vaf is body-indexed (stride 6 over NB bodies). For a MIMIC robot
        # (fixed base) NB > nv, so the composed FD-grad inner writes 18*NB — size
        # the arena's s_vaf term 18*NB to match _emit_body's ("s_vaf", 18*NB)
        # exactly (else the launched dynamic-smem is short and the kernel overruns
        # smem). Non-mimic keeps 18*nv (byte-identical; floating nv > NB).
        _vaf_count = 18 * (self.robot.get_num_joints() if self.robot_has_mimic_joints() else nv)
        # +72 for the two 6x6 SE(3) dIntegrate blocks (floating-base gradient;
        # allocated for fixed-base too but unused there).
        # Canonical input slot: q/qd/u each NUM_JOINTS(=nq=n here)-wide -> s_q_qd_u
        # is 3*n (matches _emit_body's ("s_q_qd_u", 3*nq)). The (n+nv) term is
        # s_q_orig(nq) + s_qd_orig(nv). For a FIXED base n==nv so 3*n == old 3*nv+fb
        # and n+nv == old 2*nv+fb (byte-identical); FLOATING n>nv so this reserves
        # the wider input slot (old 3*nv+fb under-reserved -> smem overrun).
        integrator_gradient_t_count = ((3*n) + 2*nv*3*nv + 2*(nv*2*nv)
                                 + _vaf_count + nv*nv + nv
                                 + (n + nv) + _max_stages * nv + _max_stages * nv * 3*nv
                                 + 72
                                 + self.gen_forward_dynamics_gradient_inner_temp_mem_size() + XI_size + rt_xfixed_reserve)
        # The "with x_kp1" variant adds s_x_kp1 ([q (nq); qd (nv)] = n+nv) on top.
        integrator_gradient_with_x_kp1_t_count = integrator_gradient_t_count + (n + nv)
        # Integrator-gradient surgical spill ladder (4 rungs, least-spill first).
        # Each rung spills only cold / output / coalesced matrices to d_workspace,
        # keeping the hot path (s_vaf + the FD-grad scaffold) in smem as long as
        # it fits. The integrator-gradient kernel never runs concurrently with
        # inverse_dynamics_gradient/forward_dynamics_gradient/fdsva_so, so its spilled buffers safely reuse those sections.
        #   rung 0: everything in smem.
        #   rung 1: s_D_qdd_stage (max_stages*nv*3nv) -> d_workspace. (g1_fixed)
        #   rung 2: + s_dAB output (2nv*3nv) -> d_workspace, + inverse_dynamics_gradient da_df band
        #           SELECTIVE spill (the FD-grad inner shrinks to the selective
        #           shared count; only the da_df band leaves smem). (g1_floating)
        #   rung 3: + the WHOLE FD-grad inner s_temp -> d_workspace (inverse_dynamics_gradient
        #           global_temp; the inner can't fit a 100 KB box on h1_2). The
        #           gradient scaffold (s_dc_du / s_vaf / s_Minv) stays in smem.
        # PERF picks the lowest fitting rung; MINIMAL is the last (always fits).
        _integrator_gradient_D_qdd_count = _max_stages * nv * 3 * nv
        _integrator_gradient_dAB_count = 2 * nv * 3 * nv
        _integrator_gradient_inner_full = self.gen_forward_dynamics_gradient_inner_temp_mem_size()
        # The FD-grad inner's da_df-band SELECTIVE spill only exists on the SPARSE
        # (non-mimic) inverse_dynamics_gradient layout. The MIMIC inner is a DENSE
        # serial fold that ignores USE_DA_DF_SPILL and always writes its full pool,
        # so it cannot shrink — size the "selective" rung to the FULL inner there.
        # (Otherwise rung 2 claims a false shrink: the arena is sized for the sparse
        # selective_shared_count but the dense inner writes its full pool -> smem OOB.
        # This is the design_principles §7 "arena sized for one rung, inner flag for
        # another" trap.) Mimic robots therefore only ever use rung 0 (full smem) or
        # rung 3 (whole pool -> d_workspace); the selective rungs 1/2 collapse onto
        # the full-inner size so the picker never lands a mimic robot on a rung that
        # under-sizes the dense pool.
        _integrator_gradient_inner_selective = (
            _integrator_gradient_inner_full if self.robot_has_mimic_joints()
            else max(self.gen_minv_inner_temp_mem_size(),
                     self.gen_inverse_dynamics_gradient_temp_layout()["selective_shared_count"]))
        _integrator_gradient_full = max(integrator_gradient_t_count, integrator_gradient_with_x_kp1_t_count)
        _integrator_gradient_arenas = compose_arena_rungs("integrator_gradient", self._arena_ctx)   # Step 3.5b fold
        self.integrator_gradient_spill_tier_3way = select_shared_tier_3way(*_integrator_gradient_arenas)
        self.integrator_gradient_t_count_per_tier = tuple(_integrator_gradient_arenas[i] for i in self.integrator_gradient_spill_tier_3way)
        _picks = self.integrator_gradient_spill_tier_3way
        # Placement booleans per rung index: which buffers leave smem.
        self.integrator_gradient_dqdd_in_smem_per_tier  = tuple(p < 1 for p in _picks)  # spilled at rungs >=1
        self.integrator_gradient_dab_in_smem_per_tier    = tuple(p < 2 for p in _picks)  # spilled at rungs >=2
        # FD-grad inner level per tier: 0 full smem, 1 selective (da_df band), 2 global_temp (whole inner).
        # The selective level (1) only exists for the SPARSE (non-mimic) inner; the
        # MIMIC dense inner ignores USE_DA_DF_SPILL, so it must never be emitted at
        # level 1 (that would set USE_DA_DF_SPILL=true against a dense pool that does
        # not honour it). For mimic, rung 2 collapses to level 0 (full smem; its arena
        # equals rung-1's-minus-dAB because the inner can't shrink) and only rung 3
        # routes the whole pool to global (level 2).
        if self.robot_has_mimic_joints():
            self.integrator_gradient_inner_level_per_tier = tuple((0 if p < 3 else 2) for p in _picks)
        else:
            self.integrator_gradient_inner_level_per_tier = tuple((0 if p < 2 else (1 if p == 2 else 2)) for p in _picks)
        # d_workspace floats the gradient needs when ANY tier spills: Dqdd + dAB +
        # the whole inner (rung-3 worst case; the rungs reuse the same regions).
        self.integrator_gradient_workspace_count = (
            (_integrator_gradient_D_qdd_count + _integrator_gradient_dAB_count + _integrator_gradient_inner_full)
            if any(p >= 1 for p in _picks) else 0)
        # da_df selective spill is a sparse-inner-only mechanism (level 1); mimic
        # never uses it (it has no selective level).
        self.integrator_gradient_uses_da_df_spill = (
            (not self.robot_has_mimic_joints()) and any(p == 2 for p in _picks))
        self._integrator_gradient_dqdd_count = _integrator_gradient_D_qdd_count
        self._integrator_gradient_dAB_count = _integrator_gradient_dAB_count
        inverse_dynamics_gradient_temp_layout = self.gen_inverse_dynamics_gradient_temp_layout()
        # Mimic robots emit a DENSE serial inverse_dynamics_gradient inner (no sparse-band spill), so
        # its inner scratch is the dense count; the selective-spill tier doesn't
        # apply (its band offsets are meaningless for the dense layout). Use the
        # dense count for full AND selective so the kernel always sizes smem for
        # the dense buffers and never selects a too-small selective arena.
        _inverse_dynamics_gradient_has_mimic = self.robot_has_mimic_joints()
        inverse_dynamics_gradient_temp_count = self.gen_inverse_dynamics_gradient_inner_temp_mem_size()
        inverse_dynamics_gradient_selective_temp_count = (
            inverse_dynamics_gradient_temp_count if _inverse_dynamics_gradient_has_mimic
            else inverse_dynamics_gradient_temp_layout["selective_shared_count"]
        )
        forward_dynamics_gradient_temp_count = self.gen_forward_dynamics_gradient_inner_temp_mem_size()
        forward_dynamics_gradient_selective_temp_count = max(self.gen_minv_inner_temp_mem_size(), inverse_dynamics_gradient_selective_temp_count)
        # s_vaf is body-indexed (NB bodies). For a MIMIC robot NB > nv so size
        # 18*NB; non-mimic keeps 18*nv/18*n (byte-identical; floating non-mimic has
        # nv > NB so 18*nv already covers the body writes). The id_device path uses
        # the get_num_pos() (==n) flavour for non-mimic to stay byte-identical with
        # the legacy 18*n; mimic robots route through 18*NB so the inner's
        # body-indexed f writes never overflow s_vaf into the XImats region.
        _vaf_cnt = 18 * (self.robot.get_num_joints() if self.robot_has_mimic_joints() else nv)
        _vaf_cnt_id = 18 * (self.robot.get_num_joints() if self.robot_has_mimic_joints() else n)
        id_device_t_count = _vaf_cnt_id + self.gen_inverse_dynamics_inner_temp_mem_size() + XI_size + rt_xfixed_reserve
        minv_device_t_count = self.gen_minv_inner_temp_mem_size() + XI_size + rt_xfixed_reserve
        fd_device_t_count = self.gen_forward_dynamics_inner_temp_mem_size() + XI_size + rt_xfixed_reserve
        inverse_dynamics_gradient_device_t_count = _vaf_cnt + inverse_dynamics_gradient_temp_count + XI_size + rt_xfixed_reserve
        forward_dynamics_gradient_device_t_count = (2*nv*nv) + (_vaf_cnt) + nv + (nv*nv) + forward_dynamics_gradient_temp_count + XI_size + rt_xfixed_reserve
        inverse_dynamics_gradient_t_count_full = (nv + n) + (2*nv*nv) + (_vaf_cnt) + nv + inverse_dynamics_gradient_temp_count + XI_size + rt_xfixed_reserve
        # Canonical input slot: q/qd/u each NUM_JOINTS(=nq=n here)-wide -> s_q_qd_u
        # is 3*n (matches _emit_forward_dynamics_gradient_kernel_body_for_flags'
        # ("s_q_qd_u", 3*nq)). For a FIXED base n==nv so 3*n == old 3*nv+fb
        # byte-identical; FLOATING n>nv reserves the wider input slot (old 3*nv+fb
        # under-reserved -> smem overrun on the s_q_qd_u load).
        forward_dynamics_gradient_t_count_full = (3*n) + (2*nv*nv) + (_vaf_cnt) + nv + (nv*nv) + forward_dynamics_gradient_temp_count + XI_size + rt_xfixed_reserve
        inverse_dynamics_gradient_t_count_selective = inverse_dynamics_gradient_t_count_full - inverse_dynamics_gradient_temp_count + inverse_dynamics_gradient_selective_temp_count
        forward_dynamics_gradient_t_count_selective = forward_dynamics_gradient_t_count_full - forward_dynamics_gradient_temp_count + forward_dynamics_gradient_selective_temp_count
        inverse_dynamics_gradient_t_count_emergency = inverse_dynamics_gradient_t_count_full - inverse_dynamics_gradient_temp_count
        forward_dynamics_gradient_t_count_emergency = forward_dynamics_gradient_t_count_full - forward_dynamics_gradient_temp_count
        # fd_du OUTPUT-spill rung (h2_plus de-gate): the emergency rung routes the
        # whole inner s_temp pool to d_workspace but keeps the OUTPUT bands s_dc_du
        # (2*nv*nv id-gradient band) + s_Minv (nv*nv mass matrix) in smem -> on a big
        # floating robot (h2_plus) those 3*nv*nv floats leave the arena at ~105KB,
        # still over sm_120's ~99KB. This rung additionally repoints s_dc_du/s_Minv
        # to the L2-pinned SO band (crba/fdsva_so output-spill style), shrinking the
        # arena by 3*nv*nv (~77KB) so h2_plus launches (~28KB). Inner pool stays in
        # the GRAD section (offset 0); the outputs sit in the disjoint SO section.
        forward_dynamics_gradient_t_count_output_spill = forward_dynamics_gradient_t_count_emergency - (2*nv*nv + nv*nv)
        # Per-tier picks (perf, lite, minimal). The existing single-pick flags
        # (inverse_dynamics_gradient_spill_tier etc.) are kept = perf pick so today's emit paths
        # are byte-for-byte unchanged; the lite/minimal indices are exposed
        # only as metadata until the per-tier emit work lands.
        _inverse_dynamics_gradient_arenas = compose_arena_rungs("inverse_dynamics_gradient", self._arena_ctx)   # Step 3.5c fold
        _forward_dynamics_gradient_arenas = compose_arena_rungs("forward_dynamics_gradient", self._arena_ctx)   # Step 3.5c fold
        self.inverse_dynamics_gradient_spill_tier_3way = select_shared_tier_3way(*_inverse_dynamics_gradient_arenas)
        self.forward_dynamics_gradient_spill_tier_3way = select_shared_tier_3way(*_forward_dynamics_gradient_arenas)
        self.inverse_dynamics_gradient_spill_tier = self.inverse_dynamics_gradient_spill_tier_3way[0]
        self.forward_dynamics_gradient_spill_tier = self.forward_dynamics_gradient_spill_tier_3way[0]
        self.inverse_dynamics_gradient_use_selective_spill = self.inverse_dynamics_gradient_spill_tier == 1
        self.forward_dynamics_gradient_use_selective_spill = self.forward_dynamics_gradient_spill_tier == 1
        self.inverse_dynamics_gradient_use_global_temp = self.inverse_dynamics_gradient_spill_tier == 2
        # global-temp (whole inner pool -> d_workspace) is used at BOTH the emergency
        # rung (2) and the output-spill rung (3, which also spills s_dc_du/s_Minv).
        self.forward_dynamics_gradient_use_global_temp = self.forward_dynamics_gradient_spill_tier >= 2
        self.forward_dynamics_gradient_use_output_spill = self.forward_dynamics_gradient_spill_tier == 3
        inverse_dynamics_gradient_t_count = _inverse_dynamics_gradient_arenas[self.inverse_dynamics_gradient_spill_tier]
        forward_dynamics_gradient_t_count = _forward_dynamics_gradient_arenas[self.forward_dynamics_gradient_spill_tier]
        # Per-tier t_counts exposed for tier-aware constexpr metadata.
        self.inverse_dynamics_gradient_t_count_per_tier = tuple(_inverse_dynamics_gradient_arenas[i] for i in self.inverse_dynamics_gradient_spill_tier_3way)
        self.forward_dynamics_gradient_t_count_per_tier = tuple(_forward_dynamics_gradient_arenas[i] for i in self.forward_dynamics_gradient_spill_tier_3way)
        # §1e: the aba kernel body reserves a 3*nq-wide per-timestep input slot
        # ("s_q_qd_tau", 3*nq in _emit_aba_kernel_body_for_flags; n == get_num_pos()
        # == nq here); the matching ABA_DYNAMIC_SHARED_MEM_BYTES arena count must use
        # the SAME 3*n, not n+2*nv. For a fixed-base cardinal robot nq==nv==n so
        # 3*n == n+2*nv (byte-identical), but on a spherical/floating (nq>nv) base the
        # n+2*nv form under-sizes the launch smem by 3*(nq-nv) floats -> the aba batch
        # kernel OOBs in load_update_XImats_helpers. Mirrors the fd/minv 3*n input slots.
        aba_input_t_count = 3 * n
        crba_input_t_count = n + nv
        # ABA surgical-spill ladder, 3 rungs. The 140*NJ+138 inner scratch band
        # keeps its hot recursion in smem and spills only the cold sub-band when
        # possible:
        #   level 0 (full)     : whole inner arena in smem (PERF, byte-identical).
        #   level 1 (surgical) : hot band in smem, cold sub-band -> d_cold. The
        #                        smem arena shrinks to the hot region only.
        #   level 2 (workspace): whole inner arena -> L2-pinned workspace (blunt
        #                        MINIMAL fallback).
        _aba_inner_temp_count = self.gen_aba_inner_temp_mem_size()
        _aba_inner_cold_count = self.gen_aba_inner_cold_mem_size()
        # Hot smem arena at the surgical rung: FIXED reclaims the whole 42*n cold
        # tail (hot ends at 98*n); FLOATING reclaims only the 138-float fb* tail
        # above tempVec (the interior vcross slot still relocates to d_cold but
        # cannot be byte-identically compacted out of smem).
        _aba_surgical_inner_count = (_aba_inner_temp_count - 138) if self.robot.floating_base else (98 * NJ)
        _aba_base_count = nv + aba_input_t_count + 12*NJ + XI_size + rt_xfixed_reserve
        _aba_arenas = compose_arena_rungs("aba", self._arena_ctx)   # Step 3.5c fold
        self.aba_spill_tier_3way = select_shared_tier_3way(*_aba_arenas)
        self.aba_use_workspace_temp = self.aba_spill_tier_3way[0] == 2
        aba_t_count = _aba_arenas[self.aba_spill_tier_3way[0]]
        self.aba_t_count_per_tier = tuple(_aba_arenas[i] for i in self.aba_spill_tier_3way)
        self._aba_inner_cold_count = _aba_inner_cold_count
        # CRBA: the inner scratch band is spilled as one band to L2-pinned
        # workspace at LITE/MINIMAL. Level 0 = scratch in smem (current);
        # Level 1 = scratch redirected to workspace.
        # An intermediate surgical-spill rung (keep hot band in smem, spill only a
        # cold sub-band) was investigated for K-crbarung and DEFERRED: post-I-crba
        # the whole inner band (42*NJ / 36*NJ+slab) is hot with no cold sub-band,
        # and the full arena (<=~19 KB) already fits smem at every default tier.
        # See _crba.py header (gen_crba_inner_temp_mem_size) for the full rationale.
        # 3-rung ladder: full | s_M->d_workspace (surgical OUTPUT_SPILL of the nv*nv mass
        # matrix, the dominant write-once buffer, read only by the optional mjx congruence;
        # the HOT inner band + XI stay in smem) | whole-band-to-workspace (MINIMAL fallback).
        # s_M routes to the L2-pinned SO band exactly like dccrba's output. The surgical
        # rung keeps the hot band resident at LITE (vs the blunt whole-band spill it replaces),
        # raising occupancy on big floating robots (h2_plus crba LITE ~47.6KB -> ~34KB).
        _crba_inner   = self.gen_crba_inner_temp_mem_size()
        _crba_no_M    = crba_input_t_count + XI_size + rt_xfixed_reserve   # input + XI, no s_M
        (_crba_t_count_full, _crba_t_count_output_spill, _crba_t_count_workspace) = \
            compose_arena_rungs("crba", self._arena_ctx)   # Step 3.2 fold
        self.crba_spill_tier_3way = select_shared_tier_3way(_crba_t_count_full, _crba_t_count_output_spill, _crba_t_count_workspace)
        # whole-band spill (inner band -> d_workspace) fires only at the deepest rung (index 2).
        self.crba_use_workspace_temp = self.crba_spill_tier_3way[0] == 2
        self.crba_t_count_per_tier = tuple(
            (_crba_t_count_full, _crba_t_count_output_spill, _crba_t_count_workspace)[i] for i in self.crba_spill_tier_3way
        )
        crba_t_count = self.crba_t_count_per_tier[0]
        # osc_inertia (Lambda = (J Minv J^T)^-1): SELF-CONTAINED, composes Minv on
        # device via minv_inner with the heavy F-region kept in a dedicated SHARED
        # s_F buffer (6*nv*nv). On h2_plus that F-region is ~153KB and the full arena
        # ~228KB (UNLAUNCHABLE). 2-rung ladder: full (s_F in smem) | spill-F (s_F ->
        # the L2-pinned minv-F workspace offset, keeping s_Minv + everything else in
        # smem). h2_plus picks spill-F (~74KB) -> LAUNCHES; all other robots fit full.
        _osc_XI = self.gen_get_XI_size(False, False)
        _osc_Xhom, _, _ = self.gen_get_Xhom_size()
        _osc_F = self.gen_minv_inner_F_size()                          # 6*nv*nv minv F-region
        _osc_temp = max(self.gen_minv_inner_no_F_size(), 16 * self.robot.get_num_joints())
        (_osc_t_full, _osc_t_spill_F) = compose_arena_rungs("osc_inertia", self._arena_ctx)   # Step 3.5e fold
        self.osc_inertia_spill_tier_3way = select_shared_tier_3way(_osc_t_full, _osc_t_spill_F)
        self.osc_inertia_t_count_per_tier = tuple(
            (_osc_t_full, _osc_t_spill_F)[i] for i in self.osc_inertia_spill_tier_3way)
        ee_t_count = compose_arena_full("end_effector_pose", self._arena_ctx)   # Step 3.1 fold
        # Phase 3d (EE_POSE_GRAD): two-tier spill, mirrors D2EE's (full, spill, spill).
        # Level 0 = full smem (inner_temp + s_end_effector_pose_gradient). Level 1 =
        # inner_temp + s_end_effector_pose_gradient -> workspace/global. The old dXmatsHom
        # spill rung is RETIRED: the geometric-Jacobian gradient inner reads only s_Xhom
        # and (void)s_dXhom, so dXmatsHom was never allocated in smem -> dropping the dead
        # + dXhom_size reservation shrinks the PERF/LITE arenas fleet-wide and pushes the
        # spill threshold outward. inner_temp is recursion-hot but L2-pinned at the host
        # wrapper for spill tiers; s_end_effector_pose_gradient is write-once output.
        _end_effector_pose_gradient_num_ees = self.robot.get_total_leaf_nodes()
        _end_effector_pose_gradient_inner_temp_count = self.gen_end_effector_pose_gradient_inner_temp_mem_size()
        _end_effector_pose_gradient_full_t_count        = n + 6*n*_end_effector_pose_gradient_num_ees + _end_effector_pose_gradient_inner_temp_count + XHom_size
        _end_effector_pose_gradient_spill_temp_t_count  = n                                                    + XHom_size
        # Degenerate 3rd arena == 2nd (MINIMAL = last index always; collapses to spill).
        _end_effector_pose_gradient_arenas = compose_arena_rungs("end_effector_pose_gradient", self._arena_ctx)   # Step 3.5e fold
        self.end_effector_pose_gradient_spill_tier_3way = select_shared_tier_3way(*_end_effector_pose_gradient_arenas)
        self.end_effector_pose_gradient_spill_tier = self.end_effector_pose_gradient_spill_tier_3way[0]
        self.end_effector_pose_gradient_use_workspace_temp = self.end_effector_pose_gradient_spill_tier >= 1
        self.end_effector_pose_gradient_use_workspace_dxhom = False  # dXhom never in smem; mirror d2ee_use_workspace_d2xhom=False
        dee_t_count = _end_effector_pose_gradient_arenas[self.end_effector_pose_gradient_spill_tier]
        self.end_effector_pose_gradient_t_count_per_tier = tuple(_end_effector_pose_gradient_arenas[i] for i in self.end_effector_pose_gradient_spill_tier_3way)
        # D2EE (FD-on-d/dv-Jacobian): two spill levels (the nv^2 output is the only
        # large buffer that can move out of smem). dXhom/d2Xhom are no longer used
        # by the geometric-Jacobian gradient inner the d2ee inner runs internally.
        _d2ee_num_ees = self.robot.get_total_leaf_nodes()
        d2ee_inner_temp_count = self.gen_end_effector_pose_hessian_inner_temp_mem_size()
        d2ee_output_count = self.gen_end_effector_pose_hessian_output_count()
        d2ee_grad_count = 6 * nv * _d2ee_num_ees
        # full smem: q + grad + d2ee_output + inner_temp + Xhom
        d2ee_full_t_count   = n + d2ee_grad_count + d2ee_output_count + d2ee_inner_temp_count + XHom_size
        # output spilled: drop d2ee_output from smem (still need q + grad + inner_temp + Xhom)
        d2ee_spill_t_count  = n + d2ee_grad_count                     + d2ee_inner_temp_count + XHom_size
        _d2ee_arenas = compose_arena_rungs("end_effector_pose_hessian", self._arena_ctx)   # Step 3.5e fold
        if "end_effector_pose_hessian" in getattr(self, "generated_algorithms", set()):
            self.d2ee_spill_tier_3way = select_shared_tier_3way(*_d2ee_arenas)
        else:
            self.d2ee_spill_tier_3way = (0, 0, 0)
        self.d2ee_spill_tier = self.d2ee_spill_tier_3way[0]
        self.d2ee_use_workspace_output = self.d2ee_spill_tier >= 1
        # Legacy aliases (older surfaces / tests still read these names; both now
        # mean "output spilled to workspace"). d2xhom flag is permanently false:
        # the FD inner never uses d2Xhom.
        self.d2ee_use_workspace_temp = self.d2ee_use_workspace_output
        self.d2ee_use_workspace_d2xhom = False
        d2ee_t_count = _d2ee_arenas[self.d2ee_spill_tier]
        self.d2ee_t_count_per_tier = tuple(_d2ee_arenas[i] for i in self.d2ee_spill_tier_3way)
        # G2 centroidal quick-wins smem t-counts (no tier spill — new, low perf
        # priority families use the full smem arena).
        NB = self.robot.get_num_bodies()
        # generalized_gravity (the larger of the two ID-bias kernels: + s_qd0):
        #   s_q_qd(2n) + s_out(nv) + s_vaf(18*nb_vaf) + s_qd0(nv) + inner_temp(6n) + XI
        # s_vaf is body-indexed by RAW body id inside inverse_dynamics_inner; for mimic
        # robots get_num_joints() (NB) > get_num_pos() (NV) so it MUST be 18*NB here to
        # match the device wrapper's arena (_centroidal.py:68/96) — else the host arena
        # macro under-budgets and the high-body f-writes overflow shared mem (h1_2:fixed
        # NB=51>NV=39 crashed). Non-mimic (nb_vaf==n) is byte-identical to the old 18*n.
        nb_vaf = self.robot.get_num_joints() if self.robot_has_mimic_joints() else n
        # generalized_gravity / nonlinear_effects share the INVERSE_DYNAMICS_BIAS arena.
        self.id_bias_t_count = compose_arena_full("generalized_gravity", self._arena_ctx)   # Step 3.1 fold
        # com/ccrba/energy (DE-GATE #2): the Jw band (6*nv*NB) is carved as a SEPARATE
        # tier-routed buffer s_J (in-smem tail at the J-in-smem tier, d_workspace at the
        # J-spilled tier), mirroring cmm/dccrba. Each family gets a 2-rung ladder:
        #   L0 keeps s_J in smem (arena byte-equivalent to the old single rung),
        #   L1 spills it -> d_workspace at the shared GRID_DCCRBA_J_OFFSET_BYTES.
        # base = s_in(<=2n) + s_out(<=6nv+6) + s_A(6nv) + s_com(3) + s_extra(4)
        #        + centroidal_inner(no-J) + XHom_size  (the Jw band is added per-tier).
        _centroidal_inner_noJ = 16*self.robot.get_num_joints() + 36*NB + 6*nv + 36
        _centroidal_sJ = 6*nv*NB
        _com_base    = n     + (3 + 3*nv) + 6*nv + 3 + 4 + _centroidal_inner_noJ + XHom_size
        _ccrba_base  = 2*n   + (6*nv + 6) + 6*nv + 3 + 4 + _centroidal_inner_noJ + XHom_size
        _energy_base = 2*n   + 3          + 6*nv + 3 + 4 + _centroidal_inner_noJ + XHom_size
        _com_rungs    = compose_arena_rungs("com", self._arena_ctx)      # Step 3.3 fold
        _ccrba_rungs  = compose_arena_rungs("ccrba", self._arena_ctx)    # Step 3.3 fold
        _energy_rungs = compose_arena_rungs("energy", self._arena_ctx)   # Step 3.3 fold
        self.com_spill_tier_3way    = select_shared_tier_3way(*_com_rungs)
        self.ccrba_spill_tier_3way  = select_shared_tier_3way(*_ccrba_rungs)
        self.energy_spill_tier_3way = select_shared_tier_3way(*_energy_rungs)
        self.com_t_count_per_tier    = tuple(_com_rungs[i] for i in self.com_spill_tier_3way)
        self.ccrba_t_count_per_tier  = tuple(_ccrba_rungs[i] for i in self.ccrba_spill_tier_3way)
        self.energy_t_count_per_tier = tuple(_energy_rungs[i] for i in self.energy_spill_tier_3way)
        self.centroidal_spill_J_ws_count = _centroidal_sJ
        # Size-triggered gravity-shim full-spill. Default OFF; if shim total shared
        # would exceed the target, set self.idsva_so_body_frame_grav_full_spill and
        # let gen_idsva_so_body_frame_inner_temp_mem_size() return the smaller value
        # (the gravity shim's dX/a/da/f/df spill into d_workspace alongside
        # d2X/d2a/d2f). Saves 50-60 KB on g1-class robots without changing
        # iiwa14/go2 behavior.
        def _compute_idsva_body_t_count():
            inner = self.gen_idsva_so_body_frame_inner_temp_mem_size()
            base = (3*n) + inner + XI_size + rt_xfixed_reserve
            full = base + 4*nv**3
            use_global_output = py_arena_bytes(full) > self.cuda_target_shared_mem_bytes
            return base if use_global_output else full, use_global_output

        self.idsva_so_body_frame_grav_full_spill = False
        idsva_so_body_frame_t_count, self.idsva_so_body_frame_use_global_output = _compute_idsva_body_t_count()
        if self.robot.floating_base and py_arena_bytes(idsva_so_body_frame_t_count) > self.cuda_target_shared_mem_bytes:
            self.idsva_so_body_frame_grav_full_spill = True
            idsva_so_body_frame_t_count, self.idsva_so_body_frame_use_global_output = _compute_idsva_body_t_count()
        # Capture the inner temp count for use downstream (world-frame fallback for
        # fixed-base + FDSVA-SO inner sizing) and for the per-tier spill ladder below.
        idsva_so_body_frame_inner_temp_count = self.gen_idsva_so_body_frame_inner_temp_mem_size()

        # ----- idsva_so BODY-frame per-tier spill ladder -----
        # Rungs least->most spill. Flags = (use_global_output, s_temp_in_global, bc_in_global, tp_in_global).
        #   rung0 full:          output + s_temp + BC all in smem
        #   rung1 global_output: 4*NV^3 output tensor -> d_workspace (cheap; coalesced one-shot)
        #   rung2 output_bc:     + BC (36*NB cold buffer, dead before hot loops) -> d_workspace (surgical)
        #   rung3 output_tp:     + ancestor-pair scratch t/p1..p6 (36*len(jids_a), 30-45% of the
        #                          body arena; DEAD through the whole recursion-hot forward sweep,
        #                          live only in the final block-parallel output assembly) ->
        #                          d_workspace. BC stays in smem (slides down to fill the vacated
        #                          t/p span). This surgical rung keeps the entire recursion-hot
        #                          chain in smem and is the highest-payoff cold sub-band on
        #                          humanoid-scale robots.
        #   rung4 output_temp:   + whole s_temp inner arena -> d_workspace (guaranteed-fit fallback)
        # rungs 2 (BC) and 3 (t/p) are mutually exclusive surgical levers (inner enforces it).
        # Fixed-base is the production overflow case. Floating-base BODY is diagnostic
        # (the dispatcher routes floating to the WORLD frame) so it keeps the legacy
        # single-body emit with the gravity shim; its picks are (0,0,0) and unused.
        _idsva_bf_BC = 36 * self.robot.get_num_bodies()
        _idsva_bf_jids_a = len(self.robot.get_jid_ancestor_ids(include_joint=True)[0])
        _idsva_bf_TP = 36 * _idsva_bf_jids_a
        _idsva_bf_base_smem = (3*n) + XI_size                                  # whole s_temp -> global
        _idsva_bf_full     = (3*n) + idsva_so_body_frame_inner_temp_count + XI_size + 4*nv**3 + rt_xfixed_reserve
        _idsva_bf_out      = (3*n) + idsva_so_body_frame_inner_temp_count + XI_size + rt_xfixed_reserve
        _idsva_so_body_tiers = [
            ("full",          _idsva_bf_full,                False, False, False, False),
            ("global_output", _idsva_bf_out,                 True,  False, False, False),
            ("output_bc",     _idsva_bf_out - _idsva_bf_BC,  True,  False, True,  False),
            ("output_tp",     _idsva_bf_out - _idsva_bf_TP,  True,  False, False, True),
            ("output_temp",   _idsva_bf_base_smem,           True,  True,  False, False),
        ]
        self._idsva_so_body_tier_table = _idsva_so_body_tiers
        if self.robot.floating_base:
            # Diagnostic path: keep legacy single-body emit + grav shim (no ladder).
            # The body ladder is NOT composed here: floating body inner is grav_full_spill-
            # dependent (picker-driven), so it is not ctx-pure — the single-value override
            # stays in-gen and the body rungs are asserted/captured only on fixed base.
            self.idsva_so_body_frame_spill_tier_3way = (0, 0, 0)
            self.idsva_so_body_frame_t_count_per_tier = (idsva_so_body_frame_t_count,) * 3
            self.idsva_so_body_frame_use_ladder = False
        else:
            _idsva_so_body_arenas = compose_arena_rungs("idsva_so_body_frame", self._arena_ctx)   # Step 3.5 fold (fixed-base ladder)
            self.idsva_so_body_frame_spill_tier_3way = select_shared_tier_3way(*_idsva_so_body_arenas)
            self.idsva_so_body_frame_t_count_per_tier = tuple(_idsva_so_body_arenas[i] for i in self.idsva_so_body_frame_spill_tier_3way)
            self.idsva_so_body_frame_use_ladder = True
            # Keep the const flag accurate to the PERF-tier pick.
            self.idsva_so_body_frame_use_global_output = _idsva_so_body_tiers[self.idsva_so_body_frame_spill_tier_3way[0]][2]

        # world-frame path has its own (smaller) scratch — no gravity-shim shared, no
        # main-sweep extras. Sized via gen_idsva_so_world_frame_temp_mem_size.
        idsva_so_world_frame_inner_temp_count = self.gen_idsva_so_world_frame_temp_mem_size() if self.robot.floating_base else idsva_so_body_frame_inner_temp_count
        idsva_so_world_frame_base_t_count = (3*n) + idsva_so_world_frame_inner_temp_count + XI_size + rt_xfixed_reserve
        idsva_so_world_frame_full_t_count = idsva_so_world_frame_base_t_count + 4*nv**3
        # ----- idsva_so WORLD-frame per-tier spill ladder -----
        # Flags = (use_global_output, s_temp_in_global, cold_in_global). The world inner
        # is UN-aliased, so a surgical rung is landed: the cold QUAD Xup (36*NB, dead after
        # the Step-4 IC build) + Xdown (36*NB, dead after Step 3) + v_w/a_w (6*NB each, dead
        # after Step 4's f_w build) = 84*NB can move to d_workspace while the hot arena stays
        # in smem (inner COLD_IN_SMEM=false).
        # Rungs least->most spill:
        #   full:               output + whole s_temp arena in smem
        #   global_output:      4*NV^3 output tensor -> d_idsva_so global (coalesced one-shot)
        #   output_cold:        + surgical cold quad (Xup 36*NB + Xdown 36*NB + v_w/a_w 12*NB = 84*NB) -> d_workspace
        #   output_temp:        + whole s_temp inner arena -> d_workspace (guaranteed-fit fallback)
        _idsva_wf_cold = self.gen_idsva_so_world_cold_floats()
        _idsva_wf_base_smem = (3*n) + XI_size
        _idsva_so_world_tiers = [
            ("full",          idsva_so_world_frame_full_t_count,                  False, False, False),
            ("global_output", idsva_so_world_frame_base_t_count,                  True,  False, False),
            ("output_cold",   idsva_so_world_frame_base_t_count - _idsva_wf_cold, True,  False, True),
            ("output_temp",   _idsva_wf_base_smem,                                True,  True,  False),
        ]
        self._idsva_so_world_tier_table = _idsva_so_world_tiers
        _idsva_so_world_arenas = compose_arena_rungs("idsva_so_world_frame", self._arena_ctx)   # Step 3.5 fold
        self.idsva_so_world_frame_spill_tier_3way = select_shared_tier_3way(*_idsva_so_world_arenas)
        self.idsva_so_world_frame_t_count_per_tier = tuple(_idsva_so_world_arenas[i] for i in self.idsva_so_world_frame_spill_tier_3way)
        self.idsva_so_world_frame_use_global_output = _idsva_so_world_tiers[self.idsva_so_world_frame_spill_tier_3way[0]][2]
        idsva_so_world_frame_t_count = self.idsva_so_world_frame_t_count_per_tier[0]
        # d_workspace floats needed per timestep by the idsva_so spill rungs (for so_workspace sizing).
        # Body rungs: 4=output_temp (whole inner arena), 3=output_tp (36*len(jids_a) ancestor-pair
        # scratch), 2=output_bc (36*NB cold slab); 0/1 spill nothing into d_workspace.
        def _idsva_body_ws_floats(pick):
            if pick == 4:
                # whole s_temp routed to workspace; the XImats helper still rebuilds
                # Xfixed into it (offset 2*num_pos) so the workspace must reserve it.
                return idsva_so_body_frame_inner_temp_count + rt_xfixed_reserve
            if pick == 3:
                return _idsva_bf_TP
            if pick == 2:
                return _idsva_bf_BC
            return 0
        def _idsva_world_ws_floats(pick):
            # pick 3 (output_temp) spills the whole inner arena; pick 2 (output_cold)
            # spills just the surgical cold quad (Xup 36*NB + Xdown 36*NB + v_w/a_w 12*NB).
            if pick == 3:
                # whole s_temp routed to workspace; XImats helper rebuilds Xfixed
                # into it (offset 2*num_pos) so the workspace must reserve it.
                return idsva_so_world_frame_inner_temp_count + rt_xfixed_reserve
            if pick == 2:
                return self.gen_idsva_so_world_cold_floats()
            return 0
        idsva_so_spill_ws_t_count = max(
            [_idsva_body_ws_floats(p) for p in self.idsva_so_body_frame_spill_tier_3way] +
            [_idsva_world_ws_floats(p) for p in self.idsva_so_world_frame_spill_tier_3way] + [0])

        # ----- FDSVA_SO shared-mem tier selection -----
        # Four nested tiers, ordered from least-spill to most-spill. Pick the
        # lowest-spill tier whose shared-arena bytes fit cuda_target_shared_mem.
        # Each tier sets three orthogonal state flags read by gen_fdsva_so_*:
        #   - use_global_tensors:  s_idsva_so + s_df2 (8*nv³ outputs) -> d_workspace
        #   - use_workspace_temp:  s_fdsva_temp (4*nv³ inner) -> d_workspace
        #   - fd_grad_use_spill:   fd_grad_inline's da_dq..fxvi band -> d_workspace grad section
        fdsva_so_base_t_count = 4*nv + nv*nv + nv + 2*nv*nv + XI_size
        fdsva_so_contract_temp_count = 4*nv**3
        fdsva_so_fd_gradient_inline_temp_count = self.gen_fdsva_so_fd_gradient_inline_temp_mem_size()
        fdsva_so_fd_gradient_inline_spilled_count = self.gen_fdsva_so_fd_gradient_inline_temp_mem_size_spilled()
        # fdsva_so dispatches to world_frame_inner for floating-base (smaller
        # footprint + no grav-shim spill) and body_frame_inner for fixed-base.
        fdsva_so_inner_idsva_so_temp_count = (
            idsva_so_world_frame_inner_temp_count if self.robot.floating_base
            else idsva_so_body_frame_inner_temp_count
        )
        # runtime_transform: the composed XImats helper rebuilds Xfixed into this
        # shared s_temp pool (offset 2*num_pos), so every pool variant must reserve
        # the 36*NB Xfixed band on top of its inner peak (dead after the helper).
        _temp_full     = max(fdsva_so_inner_idsva_so_temp_count, fdsva_so_contract_temp_count, fdsva_so_fd_gradient_inline_temp_count) + rt_xfixed_reserve
        _temp_no_contract = max(fdsva_so_inner_idsva_so_temp_count, fdsva_so_fd_gradient_inline_temp_count) + rt_xfixed_reserve
        _temp_spilled  = max(fdsva_so_inner_idsva_so_temp_count, fdsva_so_fd_gradient_inline_spilled_count) + rt_xfixed_reserve
        # Phase 3e: extend to 6 levels. Each level pushes an additional buffer
        # to L2-pinned workspace. Tuple is
        # (name, shared_count, use_global_tensors, use_workspace_temp,
        #  fd_grad_use_spill, use_workspace_df_du, use_workspace_Minv).
        # Levels 0-3 unchanged from pre-Phase-3e. Level 4 pushes s_df_du
        # (2*NV²); Level 5 also pushes s_Minv (NV²).
        fdsva_so_base_no_df_du = fdsva_so_base_t_count - 2*nv*nv
        fdsva_so_base_no_df_du_no_Minv = fdsva_so_base_no_df_du - nv*nv
        # Level 6: pool -> global. fdsva_so_device runs with SCRATCH_IN_SMEM=false,
        # routing the WHOLE shared s_temp pool (helper sincos + minv + fd + fd_grad +
        # idsva) to d_workspace (reusing the non-concurrent contraction SO-temp region).
        # Smem then holds only the base: inputs + s_qdd + s_Minv + s_df_du + XI
        # (~46-54 KB on h1_2 -> fits the ~99 KB cap). Outputs->device arrays and
        # contraction->global as in levels >=2. Works for BOTH bases because the full
        # inner repoints s_temp and hands the placed pool to the idsva inner (body or
        # world) — the sub-inner just uses the pointer it is given (inner-owns-placement).
        # A4: idsva_cold rung (between global_tensors and workspace_temp). Keeps outputs
        # in global (like global_tensors) but additionally spills the embedded WORLD
        # idsva_so inner's cold quad (Xup 36*NB + Xdown 36*NB + v_w/a_w 12*NB = 84*NB floats,
        # dead before the hot triple-walk) to the SO-temp d_workspace region via the inner's
        # COLD_IN_SMEM=false, while the hot pool stays in smem. Its smem arena = the
        # global_tensors arena minus the cold-quad span. Only EFFECTIVE when the composed
        # idsva inner is the WORLD frame (floating OR spherical) — the body-frame inner
        # has no exposed cold quad, so for body-frame robots this rung's arena is set
        # EQUAL to global_tensors (no fit advantage -> the picker never distinguishes it,
        # and the kernel's COLD flag is inert there). The reduction matches the standalone
        # idsva_so world cold rung (_idsva_wf_cold).
        _fdsva_so_uses_world_idsva = self.robot.floating_base or self.robot.robot_has_spherical()
        _fdsva_so_cold_floats = self.gen_idsva_so_world_cold_floats() if _fdsva_so_uses_world_idsva else 0
        # idsva_cold rung pool: the idsva world inner spills its cold quad to GLOBAL, so ONLY the
        # idsva term shrinks -- the reduction goes INSIDE the max. The pool is a max over three
        # INDEPENDENT consumers and the CONTRACTION (4nv^3) dominates on a floating quadruped
        # (go2-floating: idsva=3030, contraction=23328, fdg=10494), so shrinking idsva changes the
        # max by NOTHING. The old `_temp_full - _fdsva_so_cold_floats` cut the TOTAL instead,
        # under-reserving the launch by 1077 elems (4308 B) while the kernel still carved the full
        # pool -> fdsva_so_kernel wrote past shared memory ("illegal memory access", go2-floating @
        # TIER_SHARED, every thread count). See docs/agent_debugging_guide.md §1t.
        # NOTE the arena values in the 9-tuple below are DEAD (the composer has supplied arenas
        # since Step 3.4) -- but they must stay CORRECT so they can't mislead a future reader.
        _temp_idsva_cold = max(fdsva_so_inner_idsva_so_temp_count - _fdsva_so_cold_floats,
                               fdsva_so_contract_temp_count,
                               fdsva_so_fd_gradient_inline_temp_count) + rt_xfixed_reserve
        # 9-tuple: (..., use_workspace_idsva_temp == pool->global, idsva_cold_in_global). Levels keep pool in smem except pool_global.
        _fdsva_so_tiers = [
            ("full",                 fdsva_so_base_t_count + 8*nv**3 + _temp_full,  False, False, False, False, False, False, False),
            ("global_tensors",       fdsva_so_base_t_count + _temp_full,            True,  False, False, False, False, False, False),
            ("idsva_cold",           fdsva_so_base_t_count + _temp_idsva_cold,      True,  False, False, False, False, False, True),
            ("workspace_temp",       fdsva_so_base_t_count + _temp_no_contract,        True,  True,  False, False, False, False, False),
            ("workspace_temp_spill", fdsva_so_base_t_count + _temp_spilled,         True,  True,  True,  False, False, False, False),
            ("spill_df_du",          fdsva_so_base_no_df_du + _temp_spilled,        True,  True,  True,  True,  False, False, False),
            ("spill_Minv",           fdsva_so_base_no_df_du_no_Minv + _temp_spilled,True,  True,  True,  True,  True,  False, False),
            # pool->global: smem = base (inputs + qdd + Minv + df_du + XI), no pool/outputs/contraction.
            ("pool_global",          fdsva_so_base_t_count,                         True,  True,  False, False, False, True,  False),
        ]
        _fdsva_so_arenas = compose_arena_rungs("fdsva_so", self._arena_ctx)   # Step 3.4 fold
        self.fdsva_so_spill_tier_3way = select_shared_tier_3way(*_fdsva_so_arenas)
        # arena counts now come from the composer; the 9-tuple STATE FLAGS stay in-gen.
        _chosen = _fdsva_so_tiers[self.fdsva_so_spill_tier_3way[0]]
        (_, _, self.fdsva_so_use_global_tensors,
         self.fdsva_so_use_workspace_temp, self.fdsva_so_fd_grad_use_spill,
         self.fdsva_so_use_workspace_df_du, self.fdsva_so_use_workspace_Minv,
         self.fdsva_so_use_workspace_idsva_temp, self.fdsva_so_idsva_cold_in_global) = _chosen
        fdsva_so_t_count = _fdsva_so_arenas[self.fdsva_so_spill_tier_3way[0]]
        self.fdsva_so_t_count_per_tier = tuple(_fdsva_so_arenas[i] for i in self.fdsva_so_spill_tier_3way)

        # ----- F1: plant_step_hessian shared-mem tier selection (fixed-base only) -----
        # The hessian kernel composes fdsva_so_device and stages its 18*nv^3 output
        # band s_d2AB. Two tiers (mirrors _PLANT_HESSIAN_PICK_FLAGS in _plant.py):
        #   tier 0 (full smem): base + s_d2AB(18nv^3) + s_df2+s_idsva_so(8nv^3) + pool
        #   tier 1 (deep spill): base only in smem; d2AB + fdsva tensors + pool -> global
        # PERF picks the lowest tier whose arena fits cuda_target_shared_mem; LITE
        # clamps >= PERF; MINIMAL is always the deep-spill tier. Only meaningful on a
        # fixed base (floating + RK static_assert out in the device fn), but compute
        # it unconditionally — the kernel/macros are only emitted on fixed-base anyway.
        _psh_d2ab = 2 * nv * (3 * nv) * (3 * nv)
        _psh_base = (nv + nv) + nv + nv*nv + 2*nv*nv + nv + XI_size  # s_x(nx==2nv fixed) + s_u + s_qdd + Minv + df_du
        _psh_pool = max(fdsva_so_inner_idsva_so_temp_count, fdsva_so_contract_temp_count, fdsva_so_fd_gradient_inline_temp_count) + rt_xfixed_reserve
        (_psh_t_full, _psh_t_spill) = compose_arena_rungs("integrator_hessian", self._arena_ctx)   # Step 3.5e fold
        self.plant_step_hessian_spill_tier_3way = select_shared_tier_3way(_psh_t_full, _psh_t_spill)

        # ── Descriptor-table Step 3 parity net (item M) ──────────────────────
        # Ground-truth capture of each algo's FULL (least-spill / rung-0) arena
        # t_count, keyed by ALGO_DESCRIPTORS key. This EMITS NOTHING — it snapshots
        # the hand-written arena math above so test/test_algo_descriptor_arena_parity.py
        # can assert the descriptor `arena_regions` composer reproduces it exactly on
        # every matrix robot (the Step-0-style safety net that de-risks driving the
        # arena sites from the table). See docs/open-tasks/design_descriptor_table_spec.md.
        self._arena_full_t_counts = {
            "inverse_dynamics":                    id_t_count,
            "inverse_dynamics_regressor":          _idr_t_count_full,
            "kinetic_energy_regressor":            self.kinetic_energy_regressor_t_count,
            "potential_energy_regressor":          self.potential_energy_regressor_t_count,
            "coriolis_matrix":                     _coriolis_t_full,
            "cmm_time_variation":                  _cmm_t_count_full,
            "dccrba":                              _dccrba_L0,
            "forward_dynamics_parameter_gradient": _fpg_t_count_full,
            "f_ext_gradient":                      _feg_t_count_full,
            "f_ext_gradient_dq":                   _feg_dq_t_count_full,
            "minv":                                _minv_t_count_full,
            "forward_dynamics":                    _fd_t_count_full,
            "integrator":                          _integrator_t_count_full,
            "integrator_gradient":                 _integrator_gradient_full,
            "integrator_with_gradient":            _integrator_gradient_full,
            "inverse_dynamics_gradient":           inverse_dynamics_gradient_t_count_full,
            "forward_dynamics_gradient":           forward_dynamics_gradient_t_count_full,
            "aba":                                 _aba_arenas[0],
            "crba":                                _crba_t_count_full,
            "osc_inertia":                         _osc_t_full,
            "end_effector_pose":                   ee_t_count,
            "end_effector_pose_gradient":          _end_effector_pose_gradient_full_t_count,
            "end_effector_pose_hessian":           d2ee_full_t_count,
            "generalized_gravity":                 self.id_bias_t_count,
            "nonlinear_effects":                   self.id_bias_t_count,
            "com":                                 _com_base + _centroidal_sJ,
            "ccrba":                               _ccrba_base + _centroidal_sJ,
            "energy":                              _energy_base + _centroidal_sJ,
            # SO-dispatch monsters (per_base_override / workspace / dispatch aliases):
            # captured here for the 3.4/3.5 fold commits, NOT composed in Step 3.0.
            "idsva_so_body_frame":                 (_idsva_bf_full if not self.robot.floating_base
                                                    else idsva_so_body_frame_t_count),
            "idsva_so_world_frame":                _idsva_so_world_arenas[0],
            "fdsva_so":                            _fdsva_so_arenas[0],
            "integrator_hessian":                  _psh_t_full,
        }
        # Spill-ladder rung arenas (least-spill first). The generator DRIVES every
        # `select_shared_tier_3way` call above from `compose_arena_rungs`; this snapshot
        # is the descriptor table's canonical arena record + what the invariant parity
        # test reads (test/test_algo_descriptor_arena_parity.py). The body ladder is
        # fixed-base only (floating body uses the grav_full_spill picker override, which
        # is NOT ctx-pure), so it is captured only there.
        self._arena_rung_t_counts = {
            k: compose_arena_rungs(k, self._arena_ctx)
            for k in sorted(ARENA_RUNG_KEYS)
            if not (k == "idsva_so_body_frame" and self.robot.floating_base)
        }
        # In-generation descriptor-arena invariants (runs on EVERY robot generated): every
        # composed rung is positive, and — for keys whose FULL is composed — rung[0] equals
        # the full arena. Cheap structural guard against a future closure edit producing a
        # negative/absurd/inconsistent arena (the §2 under-size bug class). rt-reservation
        # correctness is checked in the parity test (needs a no-rt ctx variant).
        for _k, _rungs in self._arena_rung_t_counts.items():
            assert all(r > 0 for r in _rungs), f"descriptor arena {_k}: non-positive rung {_rungs}"
            if _k in ARENA_COMPOSED_KEYS:
                _full = compose_arena_full(_k, self._arena_ctx)
                assert _rungs[0] == _full, f"descriptor arena {_k}: rung[0] {_rungs[0]} != full {_full}"

        # Phase 3a: include Minv-F count if Minv is spilling (collisions are OK
        # because Minv runs before inverse_dynamics_gradient / forward_dynamics_gradient in any kernel that
        # composes both — they sequentially reuse the same workspace bytes).
        _minv_F_workspace_count = self.gen_minv_inner_F_size() if any(p == 1 for p in self.minv_spill_tier_3way) else 0
        # osc_inertia spill-F rung routes its 6*nv*nv minv F-region (s_F) to the same
        # minv-F workspace offset (offset 0 of the GRAD section); cover it in the max.
        _osc_F_workspace_count = self.gen_minv_inner_F_size() if any(p >= 1 for p in self.osc_inertia_spill_tier_3way) else 0
        # f_ext_gradient deep rung (2) routes minv's 6*nv*nv F-region to the SAME
        # GRAD-section minv-F offset (offset 0); never co-runs with minv/osc, so the
        # max-fold is free. (self.f_ext_gradient_spill_minv_F_count is 0 unless deep.)
        _feg_minv_F_workspace_count = self.f_ext_gradient_spill_minv_F_count
        # CRBA whole-arena spill: when crba_inner's scratch band is redirected to
        # d_workspace (LITE/MINIMAL, or a forced deep-spill tier), the per-timestep
        # workspace must be able to back the full 140*NJ-class band. Include it in
        # the grad-section max so the allocation always covers it regardless of the
        # tier the kernel template is instantiated with.
        _crba_inner_temp_count = self.gen_crba_inner_temp_mem_size()
        # coriolis whole-band spill: at the deepest rung the inner spatial-recursion
        # scratch redirects to the GRAD section, so it must back the full inner band.
        _coriolis_inner_temp_count = self.gen_coriolis_matrix_inner_temp_mem_size() if any(p >= 2 for p in self.coriolis_matrix_spill_tier_3way) else 0
        grad_spill_workspace_t_count = max(inverse_dynamics_gradient_temp_layout["spill_count"],
                                           inverse_dynamics_gradient_temp_count,
                                           forward_dynamics_gradient_temp_count,
                                           2*nv*nv,
                                           _crba_inner_temp_count,
                                           _coriolis_inner_temp_count,
                                           _minv_F_workspace_count,
                                           _osc_F_workspace_count,
                                           _feg_minv_F_workspace_count,
                                           self.integrator_minv_F_workspace_count,
                                           self.integrator_gradient_workspace_count)
        # D2EE needs no d_workspace: under the FD-on-Jacobian inner the only large
        # buffer is the nv^2 output, and when spilled the inner writes directly into
        # d_end_effector_pose_hessian (the persistent output) -- not a per-timestep workspace slice.
        d2ee_workspace_t_count = 0
        # Phase 3d: max workspace required by EE_POSE_GRAD across any tier (PERF
        # may pick 0, but the workspace allocation must cover what LITE/MINIMAL
        # need at runtime when the user switches tier via the kernel template).
        end_effector_pose_gradient_workspace_t_count = 0
        if any(p >= 1 for p in self.end_effector_pose_gradient_spill_tier_3way):
            end_effector_pose_gradient_workspace_t_count += _end_effector_pose_gradient_inner_temp_count
        # dXhom workspace clause RETIRED: the dxhom-spill rung is gone (pick 2 == pick 1),
        # so no tier materializes s_dXmatsHom -> the phantom dXhom reservation is dropped.
        # Include the floating-base gravity-shim spill (Phase D): the d2X/d2a/d2f
        # tensors live in d_workspace instead of shared memory for larger robots.
        idsva_so_body_frame_grav_spill_t_count = self.gen_floating_gravity_d2tau_dq_spill_count() if self.robot.floating_base else 0
        # g1-spill: fd_parameter_gradient (s_Y) and f_ext_gradient (s_dqdd_dfext)
        # surgically spill into this same SO workspace section when their tier picks
        # level >= 1. They never run concurrently with the SO/d2ee/end_effector_pose_gradient kernels,
        # so reuse is safe and costs no new allocation. Fold their spill counts into
        # the max so the per-timestep workspace always covers them.
        _fpg_spill_ws = self.forward_dynamics_parameter_gradient_spill_Y_ws_count if any(p >= 1 for p in self.forward_dynamics_parameter_gradient_spill_tier_3way) else 0
        _idr_spill_ws = self.inverse_dynamics_regressor_spill_Y_ws_count if any(p >= 1 for p in self.inverse_dynamics_regressor_spill_tier_3way) else 0
        _feg_spill_ws = self.f_ext_gradient_spill_out_ws_count if any(p >= 1 for p in self.f_ext_gradient_spill_tier_3way) else 0
        # f_ext_gradient_dq spills its s_JTp/s_JTm pair (2*nv*6NB) into the same SO band.
        _feg_dq_spill_ws = self.f_ext_gradient_dq_spill_jt_ws_count if any(p >= 1 for p in self.f_ext_gradient_dq_spill_tier_3way) else 0
        # PS5 dccrba: its 6*nv*nv output (L1+) and the Jw sweep band (L2, DE-GATE #2)
        # surgically spill into this same SO band at DISTINCT sub-offsets, so at L2 the
        # band must hold BOTH simultaneously (out at the base, s_J at base+6*nv*nv).
        # cmm spills only s_J (L1). Never runs concurrently with the SO kernels, so the
        # max-fold is free (8*nv^3 dwarfs out+s_J on the big robots this de-gates).
        _dccrba_spill_ws = 0
        if any(p >= 2 for p in self.dccrba_spill_tier_3way):
            _dccrba_spill_ws = self.dccrba_spill_out_ws_count + self.dccrba_spill_J_ws_count
        elif any(p >= 1 for p in self.dccrba_spill_tier_3way):
            _dccrba_spill_ws = self.dccrba_spill_out_ws_count
        # cmm places its Jw band at GRID_DCCRBA_J_OFFSET_BYTES = SO_TEMP_OFFSET +
        # 6*nv*nv*sizeof(T) (shared with dccrba's J sub-offset), so the band must span
        # that offset region (6*nv*nv) plus the Jw band itself, even though cmm never
        # writes the output region.
        _cmm_spill_ws = (6 * nv * nv + self.cmm_time_variation_spill_J_ws_count) if any(p >= 1 for p in self.cmm_time_variation_spill_tier_3way) else 0
        # com/ccrba/energy place their Jw band at the SAME GRID_DCCRBA_J_OFFSET_BYTES
        # sub-offset (SO_TEMP_OFFSET + 6*nv*nv), so the band must span that offset region
        # plus the Jw band itself when any of them spills (same robots cmm spills).
        _centroidal_spill_ws = (6 * nv * nv + self.centroidal_spill_J_ws_count) if any(
            p >= 1 for p in (self.com_spill_tier_3way + self.ccrba_spill_tier_3way + self.energy_spill_tier_3way)) else 0
        so_workspace_t_count = max(8*max(nv**3, 1), d2ee_workspace_t_count, end_effector_pose_gradient_workspace_t_count, idsva_so_body_frame_grav_spill_t_count, idsva_so_spill_ws_t_count, _fpg_spill_ws, _idr_spill_ws, _feg_spill_ws, _feg_dq_spill_ws, _dccrba_spill_ws, _cmm_spill_ws, _centroidal_spill_ws)
        # Deprecated launch-count constants remain for external callers that still
        # pass COUNT*sizeof(T).  Make them conservative aliases for the byte arena
        # layouts so those callers do not under-allocate int topology helpers or
        # 16-byte alignment padding.
        legacy_count_pad = topology_count + 8
        legacy_arena_count = lambda t_count: int(t_count + legacy_count_pad)
        _b = lambda flag: "true" if flag else "false"
        # GRID_LINALG_NVIDIA_MAX_HELPER_BYTES is a stub returning 0 (v2.0+).
        # Forward-declared here so the *_DYNAMIC_SHARED_MEM_BYTES helpers
        # below compile before _lin_alg_helpers emits the definition.
        self.gen_add_code_line("template <typename T> __host__ __device__ constexpr size_t GRID_LINALG_NVIDIA_MAX_HELPER_BYTES();")
        self.gen_add_code_lines(["const int NUM_JOINTS = " + str(self.robot.get_num_pos()) + ";", \
                                 "const int NUM_POS = " + str(self.robot.get_num_pos()) + ";", \
                                 "const int NUM_VEL = " + str(self.robot.get_num_vel()) + ";", \
                                 "const int NUM_BODIES = " + str(self.robot.get_num_bodies()) + ";", \
                                 "const int SECOND_ORDER_COORDS = " + str(self.robot.get_num_vel()) + ";", \
                                 "const int SECOND_ORDER_TENSOR_SIZE = " + str(4 * self.robot.get_num_vel()**3) + ";", \
                                 "const int Q_QD_U_STRIDE = " + str(3 * self.robot.get_num_pos()) + ";", \
                                 "const int NUM_EES = " + str(self.robot.get_total_leaf_nodes()) + ";", \
                                 "const int TOPOLOGY_HELPERS_COUNT = " + str(topology_count) + ";", \
                                 "const int DYNAMICS_XI_T_COUNT = " + str(XI_size) + ";", \
                                 "const int XHOM_T_COUNT = " + str(XHom_size) + ";", \
                                 "const int DXHOM_T_COUNT = " + str(dXhom_size) + ";", \
                                 "const int D2XHOM_T_COUNT = " + str(d2Xhom_size) + ";", \
                                 "const int GRID_INVERSE_DYNAMICS_GRADIENT_USES_GLOBAL_TEMP = " + str(int(self.inverse_dynamics_gradient_use_global_temp)) + ";", \
                                 "const int GRID_INVERSE_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER = " + str(1 if any(p >= 1 for p in self.inverse_dynamics_gradient_spill_tier_3way) else 0) + ";", \
                                 "const int GRID_FORWARD_DYNAMICS_GRADIENT_USES_GLOBAL_TEMP = " + str(int(self.forward_dynamics_gradient_use_global_temp)) + ";", \
                                 "const int GRID_FORWARD_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER = " + str(1 if any(p >= 1 for p in self.forward_dynamics_gradient_spill_tier_3way) else 0) + ";", \
                                 "const int GRID_INVERSE_DYNAMICS_GRADIENT_USES_DA_DF_SPILL = " + str(int(self.inverse_dynamics_gradient_use_selective_spill)) + ";", \
                                 "const int GRID_FORWARD_DYNAMICS_GRADIENT_USES_DA_DF_SPILL = " + str(int(self.forward_dynamics_gradient_use_selective_spill)) + ";", \
                                 "const int GRID_INTEGRATOR_USES_WORKSPACE = " + str(int(any(p == 1 for p in self.integrator_spill_tier_3way))) + ";", \
                                 "const int GRID_INTEGRATOR_GRADIENT_USES_WORKSPACE = " + str(int(any(p >= 1 for p in self.integrator_gradient_spill_tier_3way))) + ";", \
                                 "const int GRID_INTEGRATOR_GRADIENT_USES_DA_DF_SPILL = " + str(int(self.integrator_gradient_uses_da_df_spill)) + ";", \
                                 "const int GRID_GENERATES_IDSVA_SO_BODY_FRAME = " + str(int(getattr(self, "generate_idsva_so_body_frame", True))) + ";", \
                                 "const int GRID_GENERATES_FDSVA_SO = " + str(int(getattr(self, "generate_fdsva_so", True))) + ";", \
                                 "const int GRID_GENERATES_D2EE = " + str(int(getattr(self, "generate_end_effector_pose_hessian", True))) + ";", \
                                 "const int GRID_IDSVA_SO_USES_GLOBAL_OUTPUT = " + str(int(self.idsva_so_body_frame_use_global_output)) + ";", \
                                 "const int GRID_FDSVA_SO_USES_GLOBAL_TENSORS = " + str(int(self.fdsva_so_use_global_tensors)) + ";", \
                                 # Single-bool kept for inline-CUDA back-compat (reflects PERF-pick only).
                                 "const int GRID_FDSVA_SO_USES_WORKSPACE_TEMP = " + str(int(self.fdsva_so_use_workspace_temp)) + ";", \
                                 # Per-tier-aware gate (true if ANY of PERF/LITE/MINIMAL spills any band
                                 # of the s_temp pool to d_workspace; picks >= 2 cover spill_temp,
                                 # spill_fd_grad_band, spill_df_du, spill_Minv, pool_global). The host
                                 # uses this to pin L2 persistence on d_workspace for the per-tier path.
                                 "const int GRID_FDSVA_SO_USES_WORKSPACE_ANY_TIER = " + str(1 if any(p >= 2 for p in self.fdsva_so_spill_tier_3way) else 0) + ";", \
                                 # GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_TEMP: 1 if the PERF tier spills the d2ee output
                                 # (the only large buffer in the new FD-on-Jacobian path) to global memory.
                                 # When spilled, the inner writes directly into d_end_effector_pose_hessian (the persistent
                                 # output buffer) -- no extra per-timestep workspace slice is used. The old
                                 # d2xhom-spill bit is permanently 0 (the FD inner never touches d2Xhom).
                                 "const int GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_TEMP = " + str(int(self.d2ee_use_workspace_output)) + ";", \
                                 "const int GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_D2XHOM = 0;", \
                                 "const int GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_TEMP_ANY = " + str(1 if any(p >= 1 for p in self.d2ee_spill_tier_3way) else 0) + ";", \
                                 "const int GRID_END_EFFECTOR_POSE_HESSIAN_SHARED_TIER_VALUE = " + str(self.d2ee_spill_tier) + ";", \
                                 "const int GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP = " + str(int(self.end_effector_pose_gradient_use_workspace_temp)) + ";", \
                                 # Same per-tier gate for the EE_POSE_GRAD chain workspace.
                                 "const int GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP_ANY = " + str(1 if any(p >= 1 for p in self.end_effector_pose_gradient_spill_tier_3way) else 0) + ";", \
                                 "const int GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_DXHOM = " + str(int(self.end_effector_pose_gradient_use_workspace_dxhom)) + ";", \
                                 "const int GRID_END_EFFECTOR_POSE_GRADIENT_SHARED_TIER_VALUE = " + str(self.end_effector_pose_gradient_spill_tier) + ";", \
                                 # DE-GATE #2: 1 if the DEFAULT tier spills the dccrba output or Jw band / cmm Jw band
                                 # into d_workspace (so init_gridData must allocate d_workspace in the kinematics path).
                                 "const int GRID_DCCRBA_USES_WORKSPACE_TEMP = " + str(1 if (self.dccrba_spill_tier_3way[0] >= 1 or self.cmm_time_variation_spill_tier_3way[0] >= 1 or self.com_spill_tier_3way[0] >= 1 or self.ccrba_spill_tier_3way[0] >= 1 or self.energy_spill_tier_3way[0] >= 1) else 0) + ";", \
                                 # osc_inertia (kinematics-path) spill-F rung needs d_workspace; the dynamics
                                 # block does not allocate it for a kinematics-only codegen -> signal the lazy-alloc.
                                 "const int GRID_OSC_INERTIA_USES_WORKSPACE = " + str(1 if self.osc_inertia_spill_tier_3way[0] >= 1 else 0) + ";", \
                                 "const int GRID_INVERSE_DYNAMICS_GRADIENT_SHARED_TIER_VALUE = " + str(self.inverse_dynamics_gradient_spill_tier) + ";", \
                                 "const int GRID_FORWARD_DYNAMICS_GRADIENT_SHARED_TIER_VALUE = " + str(self.forward_dynamics_gradient_spill_tier) + ";", \
                                 "const int ID_DU_TEMP_SPILL_START = " + str(inverse_dynamics_gradient_temp_layout["spill_start"]) + ";", \
                                 "const int ID_DU_TEMP_SPILL_END = " + str(inverse_dynamics_gradient_temp_layout["spill_end"]) + ";", \
                                 "const int ID_DU_TEMP_SPILL_COUNT = " + str(inverse_dynamics_gradient_temp_layout["spill_count"]) + ";", \
                                 "const int INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(id_t_count)) + ";", \
                                 "const int MINV_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(minv_t_count)) + ";", \
                                 "const int FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(fd_t_count)) + ";", \
                                 "const int INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(inverse_dynamics_gradient_t_count)) + ";", \
                                 "const int FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(forward_dynamics_gradient_t_count)) + ";", \
                                 "const int INTEGRATOR_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(integrator_t_count)) + ";", \
                                 "const int INTEGRATOR_DU_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(max(integrator_gradient_t_count, integrator_gradient_with_x_kp1_t_count))) + ";", \
                                 "const int ABA_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(aba_t_count)) + ";", \
                                 "const int CRBA_SHARED_MEM_COUNT = " + str(legacy_arena_count(crba_t_count)) + ";", \
                                 "const int ID_DU_MAX_SHARED_MEM_COUNT = " + str(legacy_arena_count(inverse_dynamics_gradient_t_count_full)) + ";", \
                                 "const int FD_DU_MAX_SHARED_MEM_COUNT = " + str(legacy_arena_count(forward_dynamics_gradient_t_count_full)) + ";", \
                                 "const int END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(ee_t_count)) + ";", \
                                 "const int END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(dee_t_count)) + ";", \
                                 "const int END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_COUNT = " + str(legacy_arena_count(d2ee_t_count)) + ";", \
                                 f"const int IDSVA_SO_DYNAMIC_SHARED_MEM_COUNT = {legacy_arena_count(idsva_so_body_frame_t_count)};", \
                                 f"const int FDSVA_SO_DYNAMIC_SHARED_MEM_COUNT = {legacy_arena_count(fdsva_so_t_count)};", \
                                 "const int MAX_PERF_LEVEL_THREADS = " + str(self.max_perf_level_threads) + ";", \
                                 "",
                                 "// Resource-tier API (v2.0): each emitted kernel/_device/_inner takes a",
                                 "// `RESOURCE_TIER` template parameter that picks the (launch_bounds, smem,",
                                 "// register-footprint) profile. TIER_SHARED is the default and is the",
                                 "// current-best perf; TIER_LITE keeps the same launch_bounds but reduces",
                                 "// smem footprint (some intermediates moved to workspace global mem);",
                                 "// TIER_MINIMAL drops launch_bounds to 1024 for maximum block-size flexibility",
                                 "// at the cost of register slack. Inline-CUDA power users with tight outer",
                                 "// kernels pick LITE/MINIMAL to fit GRiD primitives in their resource budget.",
                                 "constexpr int TIER_SHARED    = 0;",
                                 "constexpr int TIER_LITE    = 1;",
                                 "constexpr int TIER_MINIMAL = 2;",
                                 "",
                                 "// Compile-time override for the default RESOURCE_TIER baked into every",
                                 "// emitted kernel template. Defaults to TIER_SHARED so existing callsites",
                                 "// (kernel<T>, host wrappers that call kernel<T><<<...>>>) keep their",
                                 "// current best-perf semantics. Bench harness sets this via",
                                 "// -DGRID_DEFAULT_RESOURCE_TIER=TIER_LITE (or TIER_MINIMAL) to sweep",
                                 "// per-tier perf without modifying host-wrapper template signatures.",
                                 "#ifndef GRID_DEFAULT_RESOURCE_TIER",
                                 "#define GRID_DEFAULT_RESOURCE_TIER TIER_SHARED",
                                 "#endif",
                                 "",
                                 "// Per-tier launch_bounds upper-bound (= max threads per block nvcc must",
                                 "// budget registers for). sm_120 has 65536 regs/block; nvcc enforces",
                                 "// regs_per_thread * max_threads <= regs_per_block, so a larger max_threads",
                                 "// directly caps regs_per_thread. PERF=SUGGESTED keeps current best perf;",
                                 "// LITE=min(2*SUGGESTED, 768) gives ~85 regs/thread cap (mid-budget);",
                                 "// MINIMAL=1024 gives ~64 regs/thread cap (maximum block-size flexibility).",
                                 "template <int TIER> __host__ __device__ constexpr int tier_max_threads() {",
                                 "    return (TIER == TIER_MINIMAL) ? 1024",
                                 "         : (TIER == TIER_LITE)    ? ((MAX_PERF_LEVEL_THREADS * 2 < 768) ? MAX_PERF_LEVEL_THREADS * 2 : 768)",
                                 "         :                          MAX_PERF_LEVEL_THREADS;",
                                 "}"])
        # A1b: baked autotuned per-algo launch config (single source of truth).
        self.gen_add_launch_config_helpers()
        self.gen_add_code_lines([
                                 "#define GRID_GENERATED_NUM_JOINTS " + str(n),
                                 "#define GRID_GENERATED_NUM_EES " + str(self.robot.get_total_leaf_nodes()),
                                 ""])
        self.gen_add_code_lines([
                                 "template <typename T> __host__ __device__ inline size_t INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(" + str(id_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t INVERSE_DYNAMICS_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.inverse_dynamics_regressor_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.inverse_dynamics_regressor_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.inverse_dynamics_regressor_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "}",
                                 # g1-spill: per-tier placement of s_Y -- true => smem, false => d_workspace.
                                 "template <int TIER> __host__ __device__ constexpr bool INVERSE_DYNAMICS_REGRESSOR_Y_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.inverse_dynamics_regressor_spill_tier_3way[0] == 0 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.inverse_dynamics_regressor_spill_tier_3way[1] == 0 else "false") + " : " + ("true" if self.inverse_dynamics_regressor_spill_tier_3way[2] == 0 else "false") + "; }",
                                 # PS5 energy regressors (each output 10*NUM_BODIES, fits every tier -> no spill).
                                 "template <typename T> __host__ __device__ inline size_t KINETIC_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(" + str(self.kinetic_energy_regressor_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }",
                                 # PS5 Coriolis matrix C(q,qd) (nv x nv; fits smem at FULL -> no spill).
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t CORIOLIS_MATRIX_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.coriolis_matrix_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.coriolis_matrix_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.coriolis_matrix_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "}",
                                 # g1-spill: tier-aware. At a spilled tier the s_Y regressor
                                 # scratch moves to d_workspace, shrinking the smem arena. Default
                                 # TIER = TIER_SHARED keeps every existing single-arg call site working.
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t FORWARD_DYNAMICS_PARAMETER_GRADIENT_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.forward_dynamics_parameter_gradient_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.forward_dynamics_parameter_gradient_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.forward_dynamics_parameter_gradient_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "}",
                                 # g1-spill: per-tier placement of s_Y -- true => smem, false => d_workspace.
                                 "template <int TIER> __host__ __device__ constexpr bool FD_PARAMETER_GRADIENT_Y_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.forward_dynamics_parameter_gradient_spill_tier_3way[0] == 0 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.forward_dynamics_parameter_gradient_spill_tier_3way[1] == 0 else "false") + " : " + ("true" if self.forward_dynamics_parameter_gradient_spill_tier_3way[2] == 0 else "false") + "; }",
                                 # g1-spill: tier-aware. At a spilled tier s_dqdd_dfext (2nd output) moves to d_workspace.
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t F_EXT_GRADIENT_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.f_ext_gradient_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.f_ext_gradient_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.f_ext_gradient_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "}",
                                 # g1/h2_plus-spill: per-tier placement. DQDD in smem only at rung 0 (full);
                                 # DTAU in smem at rungs 0-1 (spilled only at the deep rung 2, which also
                                 # routes minv-F to the GRAD workspace, F_IN_SMEM == DTAU_IN_SMEM).
                                 "template <int TIER> __host__ __device__ constexpr bool F_EXT_GRADIENT_DQDD_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.f_ext_gradient_spill_tier_3way[0] == 0 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.f_ext_gradient_spill_tier_3way[1] == 0 else "false") + " : " + ("true" if self.f_ext_gradient_spill_tier_3way[2] == 0 else "false") + "; }",
                                 "template <int TIER> __host__ __device__ constexpr bool F_EXT_GRADIENT_DTAU_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.f_ext_gradient_spill_tier_3way[0] <= 1 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.f_ext_gradient_spill_tier_3way[1] <= 1 else "false") + " : " + ("true" if self.f_ext_gradient_spill_tier_3way[2] <= 1 else "false") + "; }",
                                 # h2_plus-spill: dq kernel arena is tier-aware; the s_JTp/s_JTm pair spills at rung 1.
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t F_EXT_GRADIENT_DQ_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.f_ext_gradient_dq_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.f_ext_gradient_dq_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.f_ext_gradient_dq_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "}",
                                 # h2_plus-spill: per-tier placement of the s_JTp/s_JTm pair -- true => smem, false => d_workspace.
                                 "template <int TIER> __host__ __device__ constexpr bool F_EXT_GRADIENT_DQ_JT_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.f_ext_gradient_dq_spill_tier_3way[0] == 0 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.f_ext_gradient_dq_spill_tier_3way[1] == 0 else "false") + " : " + ("true" if self.f_ext_gradient_dq_spill_tier_3way[2] == 0 else "false") + "; }"] + [
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t MINV_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.minv_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.minv_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.minv_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "}",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.fd_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.fd_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.fd_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "}",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.inverse_dynamics_gradient_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.inverse_dynamics_gradient_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.inverse_dynamics_gradient_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "}",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.forward_dynamics_gradient_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.forward_dynamics_gradient_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.forward_dynamics_gradient_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "}",
                                 # Tier-aware: at LITE/MINIMAL the FD inner's Minv F-region (6*nv*nv)
                                 # spills to d_workspace, so the smem arena shrinks. Default TIER keeps
                                 # the existing single-arg call sites working.
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.integrator_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.integrator_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.integrator_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "}",
                                 # Per-robot tier->placement map for the integrator VALUE path's Minv F-region:
                                 # in smem at spill level 0, in d_workspace (grad section) at level 1.
                                 "template <int TIER> __host__ __device__ constexpr bool INTEGRATOR_MINV_F_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.integrator_spill_tier_3way[0] == 0 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.integrator_spill_tier_3way[1] == 0 else "false") + " : " + ("true" if self.integrator_spill_tier_3way[2] == 0 else "false") + "; }",
                                 # Tier-aware: at LITE/MINIMAL the s_D_qdd_stage buffer (max_stages*nv*3nv)
                                 # spills to d_workspace, so the smem arena shrinks. Default TIER keeps the
                                 # existing single-arg call sites working.
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.integrator_gradient_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.integrator_gradient_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.integrator_gradient_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "}",
                                 # Per-robot tier->placement map for the integrator gradient's s_D_qdd_stage
                                 # buffer: in smem at spill level 0, in d_workspace (grad section) at level 1.
                                 # Per-robot tier->placement maps for the integrator gradient's surgical
                                 # spill ladder. Each buffer's IN_SMEM bool is keyed on RESOURCE_TIER;
                                 # INNER_LEVEL gives the FD-grad inner spill (0 full smem, 1 da_df-band
                                 # selective, 2 whole inner -> d_workspace).
                                 "template <int TIER> __host__ __device__ constexpr bool INTEGRATOR_DU_D_QDD_IN_SMEM() { return (TIER == TIER_SHARED) ? " + _b(self.integrator_gradient_dqdd_in_smem_per_tier[0]) + " : (TIER == TIER_LITE) ? " + _b(self.integrator_gradient_dqdd_in_smem_per_tier[1]) + " : " + _b(self.integrator_gradient_dqdd_in_smem_per_tier[2]) + "; }",
                                 "template <int TIER> __host__ __device__ constexpr bool INTEGRATOR_DU_DAB_IN_SMEM() { return (TIER == TIER_SHARED) ? " + _b(self.integrator_gradient_dab_in_smem_per_tier[0]) + " : (TIER == TIER_LITE) ? " + _b(self.integrator_gradient_dab_in_smem_per_tier[1]) + " : " + _b(self.integrator_gradient_dab_in_smem_per_tier[2]) + "; }",
                                 "template <int TIER> __host__ __device__ constexpr int INTEGRATOR_DU_INNER_LEVEL() { return (TIER == TIER_SHARED) ? " + str(self.integrator_gradient_inner_level_per_tier[0]) + " : (TIER == TIER_LITE) ? " + str(self.integrator_gradient_inner_level_per_tier[1]) + " : " + str(self.integrator_gradient_inner_level_per_tier[2]) + "; }",
                                 # d_workspace sub-offsets (within the per-timestep slot): Dqdd at 0, then dAB, then the inner-spill region.
                                 "template <typename T> __host__ __device__ inline size_t GRID_INTEGRATOR_GRADIENT_DAB_OFFSET_BYTES() { return sizeof(T) * static_cast<size_t>(" + str(self._integrator_gradient_dqdd_count) + "); }",
                                 "template <typename T> __host__ __device__ inline size_t GRID_INTEGRATOR_GRADIENT_INNER_OFFSET_BYTES() { return sizeof(T) * static_cast<size_t>(" + str(self._integrator_gradient_dqdd_count + self._integrator_gradient_dAB_count) + "); }",
                                 "template <typename T> __host__ __device__ inline size_t INVERSE_DYNAMICS_DEVICE_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(" + str(id_device_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }",
                                 "template <typename T> __host__ __device__ inline size_t MINV_DEVICE_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(" + str(minv_device_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }",
                                 "template <typename T> __host__ __device__ inline size_t FORWARD_DYNAMICS_DEVICE_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(" + str(fd_device_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }",
                                 # Per-tier sizes for forward_dynamics_device (inline-CUDA users only).
                                 # At TIER_SHARED the FD inner s_temp lives in the smem arena; at
                                 # TIER_LITE/MINIMAL the whole arena moves to d_workspace (this is the
                                 # device-path analog of the FD kernel's MINV_F_IN_SMEM lever, which
                                 # surgically spills only the F tail; the device path takes the
                                 # whole-arena route to keep the inline call's smem footprint minimal).
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t FORWARD_DYNAMICS_DEVICE_INLINE_SMEM_BYTES() {",
                                 "    return (TIER == TIER_SHARED)",
                                 "        ? grid_shared_arena_bytes<T>(" + str(fd_device_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>())",
                                 "        : grid_shared_arena_bytes<T>(" + str(fd_device_t_count - self.gen_forward_dynamics_inner_temp_mem_size(minv_f_in_smem=True)) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());",
                                 "}",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t FORWARD_DYNAMICS_DEVICE_INLINE_WORKSPACE_BYTES() { return (TIER == TIER_SHARED) ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(self.gen_forward_dynamics_inner_temp_mem_size(minv_f_in_smem=True)) + "); }",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t ABA_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.aba_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.aba_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.aba_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "}",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t CRBA_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.crba_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.crba_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.crba_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
                                 "}",
                                 "template <typename T> __host__ __device__ constexpr size_t GRID_EE_LINALG_SHARED_BYTES() { return static_cast<size_t>(0); }",
                                 # PS5 potential-energy regressor (kinematics / XmatsHom domain; uses the ee linalg helper bytes).
                                 "template <typename T> __host__ __device__ inline size_t POTENTIAL_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(" + str(self.potential_energy_regressor_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }",
                                 "template <typename T> __host__ __device__ inline size_t END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(" + str(ee_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }",
                                 # Phase 3d: tier-aware. PERF/LITE/MINIMAL each report the smem
                                 # bytes their picked spill level needs. Collapsed picks (small
                                 # robots) return identical values across branches.
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.end_effector_pose_gradient_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.end_effector_pose_gradient_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.end_effector_pose_gradient_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                                 "}",
                                 # Tier-aware: TIER_SHARED/LITE/MINIMAL each report the smem bytes their
                                 # picked spill level needs. When the picks collapse (small robots) the
                                 # three branches return identical values. Default TIER = TIER_SHARED
                                 # preserves all existing single-arg call sites.
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.d2ee_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.d2ee_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.d2ee_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                                 "}",
                                 # G2 centroidal quick-wins shared-mem macros (no tier spill).
                                 "template <typename T> __host__ __device__ inline size_t INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES() { return grid_shared_arena_bytes<T>(" + str(self.id_bias_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }",
                                 # com/ccrba/energy: 2-rung J-spill ladder (DE-GATE #2). L0 keeps the Jw
                                 # band in smem (== old single rung), L1 spills it -> d_workspace.
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t COM_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.com_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.com_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.com_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t CCRBA_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.ccrba_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.ccrba_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.ccrba_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t ENERGY_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.energy_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.energy_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.energy_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }",
                                 "template <int TIER> __host__ __device__ constexpr bool COM_J_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.com_spill_tier_3way[0] == 0 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.com_spill_tier_3way[1] == 0 else "false") + " : " + ("true" if self.com_spill_tier_3way[2] == 0 else "false") + "; }",
                                 "template <int TIER> __host__ __device__ constexpr bool CCRBA_J_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.ccrba_spill_tier_3way[0] == 0 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.ccrba_spill_tier_3way[1] == 0 else "false") + " : " + ("true" if self.ccrba_spill_tier_3way[2] == 0 else "false") + "; }",
                                 "template <int TIER> __host__ __device__ constexpr bool ENERGY_J_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.energy_spill_tier_3way[0] == 0 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.energy_spill_tier_3way[1] == 0 else "false") + " : " + ("true" if self.energy_spill_tier_3way[2] == 0 else "false") + "; }",
                                 # PS5 dCCRBA (kinematics domain, uses the EE linalg scratch like ccrba):
                                 # cmm_time_variation (Adot, 6*nv; no spill) + dccrba (6*nv*nv; per-tier
                                 # surgical spill of the output to the d_workspace SO band).
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t CMM_TIME_VARIATION_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.cmm_time_variation_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.cmm_time_variation_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.cmm_time_variation_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                                 "}",
                                 # DE-GATE #2: per-tier placement of the cmm Jw sweep band -- true => smem, false => d_workspace.
                                 "template <int TIER> __host__ __device__ constexpr bool CMM_J_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.cmm_time_variation_spill_tier_3way[0] == 0 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.cmm_time_variation_spill_tier_3way[1] == 0 else "false") + " : " + ("true" if self.cmm_time_variation_spill_tier_3way[2] == 0 else "false") + "; }",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t DCCRBA_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.dccrba_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.dccrba_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.dccrba_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                                 "}",
                                 # per-tier placement of the s_dccrba output -- true => smem, false => d_workspace.
                                 "template <int TIER> __host__ __device__ constexpr bool DCCRBA_OUTPUT_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.dccrba_spill_tier_3way[0] == 0 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.dccrba_spill_tier_3way[1] == 0 else "false") + " : " + ("true" if self.dccrba_spill_tier_3way[2] == 0 else "false") + "; }",
                                 # DE-GATE #2: per-tier placement of the dccrba Jw sweep band -- in smem at L0/L1 (pick<=1), spilled at L2.
                                 "template <int TIER> __host__ __device__ constexpr bool DCCRBA_J_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.dccrba_spill_tier_3way[0] <= 1 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.dccrba_spill_tier_3way[1] <= 1 else "false") + " : " + ("true" if self.dccrba_spill_tier_3way[2] <= 1 else "false") + "; }",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.idsva_so_body_frame_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.idsva_so_body_frame_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.idsva_so_body_frame_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT); "
                                 "}",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.idsva_so_world_frame_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.idsva_so_world_frame_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.idsva_so_world_frame_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT); "
                                 "}",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES() { "
                                 "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(self.fdsva_so_t_count_per_tier[0]) + ", TOPOLOGY_HELPERS_COUNT); "
                                 "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(self.fdsva_so_t_count_per_tier[1]) + ", TOPOLOGY_HELPERS_COUNT); "
                                 "else                                 return grid_shared_arena_bytes<T>(" + str(self.fdsva_so_t_count_per_tier[2]) + ", TOPOLOGY_HELPERS_COUNT); "
                                 "}",
                                 "// Per-tier scratch sizes for fdsva_so_contract (inline-CUDA users only — the host launchers always use TIER_SHARED).",
                                 "// At TIER_SHARED the 4*NV^3 inner scratch lives in s_temp; at TIER_LITE/MINIMAL it moves to d_workspace, freeing shared memory for the caller's outer kernel.",
                                 "// fdsva_so_contract scratch sizing, keyed on the INNER's placement choice",
                                 "// (SCRATCH_IN_SMEM) rather than a tier — the inner decides placement, the",
                                 "// caller sizes both arenas from these. FDSVA_SO_SCRATCH_IN_SMEM<TIER>()",
                                 "// gives the placement codegen assigned to each tier for THIS robot.",
                                 "template <typename T, bool SCRATCH_IN_SMEM = true> __host__ __device__ constexpr size_t FDSVA_SO_INNER_SMEM_BYTES() { return SCRATCH_IN_SMEM ? sizeof(T) * static_cast<size_t>(" + str(4*nv**3) + ") : static_cast<size_t>(0); }",
                                 "template <typename T, bool SCRATCH_IN_SMEM = true> __host__ __device__ constexpr size_t FDSVA_SO_INNER_WORKSPACE_BYTES() { return SCRATCH_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(4*nv**3) + "); }",
                                 # Per-robot tier->placement map for the fdsva_so_contract 4*NV^3 scratch:
                                 # it stays in smem (CONTRACT_IN_SMEM=true) at every rung that does NOT
                                 # set use_workspace_temp (the contract-spill flag). Derived from the per-rung
                                 # use_workspace_temp flag (tuple field index 3) so the A4 idsva_cold rung
                                 # (which keeps the contract in smem) is classified correctly without a
                                 # hardcoded index that the rung insertion would have shifted.
                                 "// Per-robot tier->placement map: contraction scratch stays in smem at any rung that doesn't set use_workspace_temp.",
                                 "template <int TIER> __host__ __device__ constexpr bool FDSVA_SO_SCRATCH_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if not _fdsva_so_tiers[self.fdsva_so_spill_tier_3way[0]][3] else "false") + " : (TIER == TIER_LITE) ? " + ("true" if not _fdsva_so_tiers[self.fdsva_so_spill_tier_3way[1]][3] else "false") + " : " + ("true" if not _fdsva_so_tiers[self.fdsva_so_spill_tier_3way[2]][3] else "false") + "; }",
                                 "// Inner-controlled placement API (design rollout): each inline inner is keyed on a",
                                 "// placement bool and decides arena pointers itself. *_INNER_{SMEM,WORKSPACE}_BYTES<T, IN_SMEM>",
                                 "// give the two arena sizes; *_<...>_IN_SMEM<TIER>() give the per-robot tier->placement",
                                 "// map codegen assigned (multiple tiers may share a placement on small robots).",
                                 "// --- minv_inner (F-region) ---",
                                 "template <typename T, bool F_IN_SMEM = true> __host__ __device__ constexpr size_t MINV_INNER_SMEM_BYTES() { return sizeof(T) * static_cast<size_t>(" + str(self.gen_minv_inner_no_F_size()) + (" + " + str(6*nv*nv) + " * (F_IN_SMEM ? 1 : 0)") + "); }",
                                 "template <typename T, bool F_IN_SMEM = true> __host__ __device__ constexpr size_t MINV_INNER_WORKSPACE_BYTES() { return F_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(6*nv*nv) + "); }",
                                 "template <int TIER> __host__ __device__ constexpr bool MINV_F_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.minv_spill_tier_3way[0] == 0 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.minv_spill_tier_3way[1] == 0 else "false") + " : " + ("true" if self.minv_spill_tier_3way[2] == 0 else "false") + "; }",
                                 "// --- forward_dynamics_inner (internal Minv F-region) ---",
                                 "template <typename T, bool MINV_F_IN_SMEM = true> __host__ __device__ constexpr size_t FD_INNER_SMEM_BYTES() { return MINV_F_IN_SMEM ? sizeof(T) * static_cast<size_t>(" + str(self.gen_forward_dynamics_inner_temp_mem_size(minv_f_in_smem=True)) + ") : sizeof(T) * static_cast<size_t>(" + str(self.gen_forward_dynamics_inner_temp_mem_size(minv_f_in_smem=False)) + "); }",
                                 "template <typename T, bool MINV_F_IN_SMEM = true> __host__ __device__ constexpr size_t FD_INNER_WORKSPACE_BYTES() { return MINV_F_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(6*nv*nv) + "); }",
                                 "template <int TIER> __host__ __device__ constexpr bool FD_MINV_F_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.fd_spill_tier_3way[0] == 0 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.fd_spill_tier_3way[1] == 0 else "false") + " : " + ("true" if self.fd_spill_tier_3way[2] == 0 else "false") + "; }",
                                 # The integrator value path's only inner scratch is the FD inner itself,
                                 # so its arena sizes mirror FD_INNER_* exactly: when MINV_F_IN_SMEM the
                                 # 6*NV*NV F-region is in s_temp, else it spills to d_workspace. The
                                 # per-robot tier->placement map is INTEGRATOR_MINV_F_IN_SMEM<TIER> (above).
                                 "// --- integrator_inner (forwards the FD inner's Minv F-region lever) ---",
                                 "template <typename T, bool MINV_F_IN_SMEM = true> __host__ __device__ constexpr size_t INTEGRATOR_INNER_SMEM_BYTES() { return MINV_F_IN_SMEM ? sizeof(T) * static_cast<size_t>(" + str(self.gen_integrator_inner_temp_mem_size(minv_f_in_smem=True)) + ") : sizeof(T) * static_cast<size_t>(" + str(self.gen_integrator_inner_temp_mem_size(minv_f_in_smem=False)) + "); }",
                                 "template <typename T, bool MINV_F_IN_SMEM = true> __host__ __device__ constexpr size_t INTEGRATOR_INNER_WORKSPACE_BYTES() { return MINV_F_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(6*nv*nv) + "); }",
                                 "// --- aba_inner (scratch band, surgical-spill ladder) ---",
                                 "// Levels: 0=full (smem), 1=surgical (hot smem + cold d_cold), 2=workspace (whole band global).",
                                 "// TEMP_IN_SMEM is false only at the level-2 (workspace) rung; COLD_IN_SMEM is false only at the level-1 (surgical) rung.",
                                 "template <typename T, bool TEMP_IN_SMEM = true> __host__ __device__ constexpr size_t ABA_INNER_SMEM_BYTES() { return TEMP_IN_SMEM ? sizeof(T) * static_cast<size_t>(" + str(self.gen_aba_inner_temp_mem_size()) + ") : static_cast<size_t>(0); }",
                                 "template <typename T, bool TEMP_IN_SMEM = true> __host__ __device__ constexpr size_t ABA_INNER_WORKSPACE_BYTES() { return TEMP_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(self.gen_aba_inner_temp_mem_size()) + "); }",
                                 "template <typename T> __host__ __device__ constexpr size_t ABA_INNER_COLD_BYTES() { return sizeof(T) * static_cast<size_t>(" + str(self._aba_inner_cold_count) + "); }",
                                 "template <int TIER> __host__ __device__ constexpr bool ABA_TEMP_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("false" if self.aba_spill_tier_3way[0] == 2 else "true") + " : (TIER == TIER_LITE) ? " + ("false" if self.aba_spill_tier_3way[1] == 2 else "true") + " : " + ("false" if self.aba_spill_tier_3way[2] == 2 else "true") + "; }",
                                 "template <int TIER> __host__ __device__ constexpr bool ABA_COLD_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("false" if self.aba_spill_tier_3way[0] == 1 else "true") + " : (TIER == TIER_LITE) ? " + ("false" if self.aba_spill_tier_3way[1] == 1 else "true") + " : " + ("false" if self.aba_spill_tier_3way[2] == 1 else "true") + "; }",
                                 "// --- crba_inner (scratch band) ---",
                                 "template <typename T, bool TEMP_IN_SMEM = true> __host__ __device__ constexpr size_t CRBA_INNER_SMEM_BYTES() { return TEMP_IN_SMEM ? sizeof(T) * static_cast<size_t>(" + str(self.gen_crba_inner_temp_mem_size()) + ") : static_cast<size_t>(0); }",
                                 "template <typename T, bool TEMP_IN_SMEM = true> __host__ __device__ constexpr size_t CRBA_INNER_WORKSPACE_BYTES() { return TEMP_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(self.gen_crba_inner_temp_mem_size()) + "); }",
                                 # Inner band stays in smem at rung0 AND rung1 (surgical output-spill); spills only at rung2 (whole-band). Mirror DCCRBA_J_IN_SMEM's <=1.
                                 "template <int TIER> __host__ __device__ constexpr bool CRBA_TEMP_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.crba_spill_tier_3way[0] <= 1 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.crba_spill_tier_3way[1] <= 1 else "false") + " : " + ("true" if self.crba_spill_tier_3way[2] <= 1 else "false") + "; }",
                                 # s_M output in smem ONLY at rung0; spilled to the SO band at rung1/rung2. Mirror DCCRBA_OUTPUT_IN_SMEM's ==0.
                                 "template <int TIER> __host__ __device__ constexpr bool CRBA_M_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.crba_spill_tier_3way[0] == 0 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.crba_spill_tier_3way[1] == 0 else "false") + " : " + ("true" if self.crba_spill_tier_3way[2] == 0 else "false") + "; }",
                                 "// --- end_effector_pose_gradient_inner (chain workspace) ---",
                                 "template <typename T, bool TEMP_IN_SMEM = true> __host__ __device__ constexpr size_t EE_GRAD_INNER_SMEM_BYTES() { return TEMP_IN_SMEM ? sizeof(T) * static_cast<size_t>(" + str(self.gen_end_effector_pose_gradient_inner_temp_mem_size()) + ") : static_cast<size_t>(0); }",
                                 "template <typename T, bool TEMP_IN_SMEM = true> __host__ __device__ constexpr size_t EE_GRAD_INNER_WORKSPACE_BYTES() { return TEMP_IN_SMEM ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(self.gen_end_effector_pose_gradient_inner_temp_mem_size()) + "); }",
                                 "template <int TIER> __host__ __device__ constexpr bool EE_GRAD_TEMP_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.end_effector_pose_gradient_spill_tier_3way[0] == 0 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.end_effector_pose_gradient_spill_tier_3way[1] == 0 else "false") + " : " + ("true" if self.end_effector_pose_gradient_spill_tier_3way[2] == 0 else "false") + "; }",
                                 "// --- end_effector_pose_hessian_inner (large nv^2 end_effector_pose_hessian output) ---",
                                 "// Per-tier placement of the d2ee inner's OUTPUT s_end_effector_pose_hessian: true => smem, false => d_workspace (which the kernel sets to d_end_effector_pose_hessian directly).",
                                 "template <int TIER> __host__ __device__ constexpr bool D2EE_OUT_IN_SMEM() { return (TIER == TIER_SHARED) ? " + ("true" if self.d2ee_spill_tier_3way[0] == 0 else "false") + " : (TIER == TIER_LITE) ? " + ("true" if self.d2ee_spill_tier_3way[1] == 0 else "false") + " : " + ("true" if self.d2ee_spill_tier_3way[2] == 0 else "false") + "; }",
                                 "// Per-tier sizes for forward_dynamics_gradient_device (inline-CUDA users only). At TIER_SHARED the temp scratch arena lives in s_temp; at TIER_LITE/MINIMAL it moves to d_workspace, freeing roughly " + str(forward_dynamics_gradient_temp_count) + "*sizeof(T) bytes of smem.",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t FORWARD_DYNAMICS_GRADIENT_DEVICE_INLINE_SMEM_BYTES() {",
                                 "    return (TIER == TIER_SHARED)",
                                 "        ? grid_shared_arena_bytes<T>(" + str(forward_dynamics_gradient_device_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>())",
                                 "        : grid_shared_arena_bytes<T>(" + str(forward_dynamics_gradient_device_t_count - forward_dynamics_gradient_temp_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());",
                                 "}",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t FORWARD_DYNAMICS_GRADIENT_DEVICE_INLINE_WORKSPACE_BYTES() { return (TIER == TIER_SHARED) ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(forward_dynamics_gradient_temp_count) + "); }",
                                 "// Per-tier sizes for end_effector_pose_hessian_device (inline-CUDA users only). At TIER_SHARED the smem arena keeps only the FD scratch + s_Xhom; at TIER_LITE/MINIMAL the device contract is unchanged (smem arena is the same -- the caller-provided s_end_effector_pose_hessian is what shifts), and the inner writes its " + str(d2ee_output_count) + "*sizeof(T) output bytes to d_workspace instead.",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t END_EFFECTOR_POSE_HESSIAN_DEVICE_INLINE_SMEM_BYTES() {",
                                 "    return grid_shared_arena_bytes<T>(" + str(d2ee_inner_temp_count + XHom_size) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>());",
                                 "}",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t END_EFFECTOR_POSE_HESSIAN_DEVICE_INLINE_WORKSPACE_BYTES() { return (TIER == TIER_SHARED) ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(d2ee_output_count) + "); }",
                                 "// Per-tier sizes for inverse_dynamics_gradient_device (inline-CUDA users only). At TIER_SHARED temp lives in s_temp; at TIER_LITE/MINIMAL it moves to d_workspace, freeing " + str(inverse_dynamics_gradient_temp_count) + "*sizeof(T) bytes of smem.",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t INVERSE_DYNAMICS_GRADIENT_DEVICE_INLINE_SMEM_BYTES() {",
                                 "    return (TIER == TIER_SHARED)",
                                 "        ? grid_shared_arena_bytes<T>(" + str(inverse_dynamics_gradient_device_t_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>())",
                                 "        : grid_shared_arena_bytes<T>(" + str(inverse_dynamics_gradient_device_t_count - inverse_dynamics_gradient_temp_count) + ", TOPOLOGY_HELPERS_COUNT, GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());",
                                 "}",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t INVERSE_DYNAMICS_GRADIENT_DEVICE_INLINE_WORKSPACE_BYTES() { return (TIER == TIER_SHARED) ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(inverse_dynamics_gradient_temp_count) + "); }",
                                 "// Per-tier sizes for idsva_so_device (inline-CUDA users only). At TIER_SHARED temp lives in s_temp; at TIER_LITE/MINIMAL it moves to d_workspace, freeing " + str(idsva_so_world_frame_inner_temp_count if self.robot.floating_base else idsva_so_body_frame_inner_temp_count) + "*sizeof(T) bytes of smem. Frame picked at codegen time: " + ("world_frame" if self.robot.floating_base else "body_frame") + ".",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t IDSVA_SO_DEVICE_INLINE_SMEM_BYTES() {",
                                 "    return (TIER == TIER_SHARED)",
                                 "        ? grid_shared_arena_bytes<T>(" + str((idsva_so_world_frame_inner_temp_count if self.robot.floating_base else idsva_so_body_frame_inner_temp_count) + XI_size) + ", TOPOLOGY_HELPERS_COUNT)",
                                 "        : grid_shared_arena_bytes<T>(" + str(XI_size) + ", TOPOLOGY_HELPERS_COUNT);",
                                 "}",
                                 "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t IDSVA_SO_DEVICE_INLINE_WORKSPACE_BYTES() { return (TIER == TIER_SHARED) ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(idsva_so_world_frame_inner_temp_count if self.robot.floating_base else idsva_so_body_frame_inner_temp_count) + "); }",
                                 "template <typename T> __host__ __device__ inline size_t GRID_GRAD_WORKSPACE_BYTES_PER_TIMESTEP() { return sizeof(T) * static_cast<size_t>(" + str(grad_spill_workspace_t_count) + "); }",
                                 "template <typename T> __host__ __device__ inline size_t GRID_SO_WORKSPACE_BYTES_PER_TIMESTEP() { return sizeof(T) * static_cast<size_t>(" + str(so_workspace_t_count) + "); }",
                                 # Phase 3e: sized for MINIMAL tier's spill (max across PERF/LITE/MINIMAL).
                                 # Even if PERF doesn't spill df_du/Minv, MINIMAL might — the workspace
                                 # allocation has to cover MINIMAL's needs at all times.
                                 # A4: the idsva_cold rung insertion shifted spill_df_du/spill_Minv/pool_global
                                 # from old indices 4/5/6 to 5/6/7, so the legacy `p >= 4` threshold (= "any rung
                                 # at or past spill_df_du") is now `p >= 5` to reproduce the EXACT same per-robot
                                 # value (Gate-A byte-identical for cardinals). The 3*NV^2 span backs
                                 # s_df_du(2*NV^2) + s_Minv(NV^2) at GRID_FDSVA_SO_SPILL_OFFSET_BYTES.
                                 "template <typename T> __host__ __device__ inline size_t GRID_FDSVA_SO_SPILL_BYTES_PER_TIMESTEP() { return sizeof(T) * static_cast<size_t>(" + str(3*nv*nv if any(p >= 5 for p in getattr(self, 'fdsva_so_spill_tier_3way', (0, 0, 0))) else 0) + "); }",
                                 "template <typename T> __host__ __device__ inline size_t GRID_FDSVA_SO_SPILL_OFFSET_BYTES() { return GRID_GRAD_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_SO_WORKSPACE_BYTES_PER_TIMESTEP<T>(); }",
                                 "template <typename T> __host__ __device__ inline size_t GRID_WORKSPACE_BYTES_PER_TIMESTEP() { return GRID_GRAD_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_SO_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_FDSVA_SO_SPILL_BYTES_PER_TIMESTEP<T>(); }",
                                 "template <typename T> __host__ __device__ inline gridSharedTier GRID_INVERSE_DYNAMICS_GRADIENT_SHARED_TIER() { return static_cast<gridSharedTier>(GRID_INVERSE_DYNAMICS_GRADIENT_SHARED_TIER_VALUE); }",
                                 "template <typename T> __host__ __device__ inline gridSharedTier GRID_FORWARD_DYNAMICS_GRADIENT_SHARED_TIER() { return static_cast<gridSharedTier>(GRID_FORWARD_DYNAMICS_GRADIENT_SHARED_TIER_VALUE); }",
                                 "template <typename T> __host__ __device__ inline size_t GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES() { return GRID_GRAD_WORKSPACE_BYTES_PER_TIMESTEP<T>(); }",
                                 # DE-GATE #2: the dccrba Jw sweep band (and the cmm Jw band) spill to the SO
                                 # band at a DISTINCT sub-offset so they never alias the dccrba output (which
                                 # sits at GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES, size 6*nv*nv*sizeof(T)). For cmm
                                 # the output region is unused so the overlap is harmless.
                                 "template <typename T> __host__ __device__ inline size_t GRID_DCCRBA_J_OFFSET_BYTES() { return GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>() + sizeof(T) * static_cast<size_t>(" + str(6 * nv * nv) + "); }",
                                 # Phase 3a: Minv-F lives at offset 0 of the grad section when spilled.
                                 # Safe to overlap with inverse_dynamics_gradient spill region because Minv finishes before
                                 # inverse_dynamics_gradient starts in any kernel that composes both.
                                 "template <typename T> __host__ __device__ inline size_t GRID_MINV_F_WORKSPACE_OFFSET_BYTES() { return static_cast<size_t>(0); }",
                                 # ABA surgical cold sub-buffer reuses the SO/grad workspace band base (ABA
                                 # never runs concurrently with SO/grad). The cold band (ABA_INNER_COLD_BYTES)
                                 # is far smaller than GRID_GRAD_WORKSPACE_BYTES_PER_TIMESTEP, so it fits at
                                 # offset 0 without growing GRID_WORKSPACE_BYTES_PER_TIMESTEP.
                                 "template <typename T> __host__ __device__ inline size_t GRID_ABA_COLD_OFFSET_BYTES() { return static_cast<size_t>(0); }",
                                 # D2EE no longer uses a per-timestep d_workspace slice (the spilled
                                 # s_end_effector_pose_hessian is written directly into d_end_effector_pose_hessian); these offset macros are
                                 # retained as 0 for backward compatibility with any inline-CUDA caller
                                 # pattern that still references them. New code should not use them.
                                 "template <typename T> __host__ __device__ inline size_t GRID_END_EFFECTOR_POSE_HESSIAN_WORKSPACE_TEMP_OFFSET_BYTES() { return static_cast<size_t>(0); }",
                                 "template <typename T> __host__ __device__ inline size_t GRID_END_EFFECTOR_POSE_HESSIAN_WORKSPACE_D2XHOM_OFFSET_BYTES() { return static_cast<size_t>(0); }",
                                 "template <typename T> __host__ __device__ inline size_t GRID_END_EFFECTOR_POSE_HESSIAN_WORKSPACE_D2EETEMP_OFFSET_BYTES() { return static_cast<size_t>(0); }",
                                 # Phase 3d: EE_POSE_GRAD reuses the SO section (the kernels don't
                                 # run concurrently — d_workspace bytes are safely repurposed). When
                                 # the MINIMAL tier spills dXmatsHom, it sits before the temp arena.
                                 "template <typename T> __host__ __device__ inline size_t GRID_END_EFFECTOR_POSE_GRADIENT_WORKSPACE_DXHOM_OFFSET_BYTES() { return GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>(); }",
                                 "template <typename T> __host__ __device__ inline size_t GRID_END_EFFECTOR_POSE_GRADIENT_WORKSPACE_TEMP_OFFSET_BYTES() { return GRID_END_EFFECTOR_POSE_GRADIENT_WORKSPACE_DXHOM_OFFSET_BYTES<T>() + (GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_DXHOM ? sizeof(T) * static_cast<size_t>(DXHOM_T_COUNT) : 0); }",
                                 "template <typename T> __host__ __device__ inline bool grid_selected_shared_memory_fits() { return INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>() <= GRID_CUDA_TARGET_SHARED_MEM_BYTES && FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>() <= GRID_CUDA_TARGET_SHARED_MEM_BYTES && (!GRID_GENERATES_D2EE || END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T>() <= GRID_CUDA_TARGET_SHARED_MEM_BYTES) && (!GRID_GENERATES_IDSVA_SO_BODY_FRAME || IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES<T>() <= GRID_CUDA_TARGET_SHARED_MEM_BYTES) && (!GRID_GENERATES_FDSVA_SO || FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES<T>() <= GRID_CUDA_TARGET_SHARED_MEM_BYTES); }",
                                 "// __forceinline__ used throughout the xhom helper chain so ptxas folds these into the",
                                 "// inner kernels at all opt levels. For fixed-base the body of grid_q_index_affects_joint is",
                                 "// the trivial `q_index == joint_id` check that pre-GLASS callsites used directly.",
                                 "__host__ __device__ __forceinline__ bool grid_q_index_affects_joint(const int q_index, const int joint_id) {",
                                 ("    if (joint_id == 0) { return q_index >= 0 && q_index < 7; } return q_index == joint_id + 6;" if self.robot.floating_base else "    return q_index == joint_id;"),
                                 "}",
                                 "__host__ __device__ __forceinline__ int grid_d2xhom_offset(const int q_index_i, [[maybe_unused]] const int q_index_j) {",
                                 ("    return 16 * (q_index_i * NUM_JOINTS + q_index_j);" if self.robot.floating_base else "    return 16 * q_index_i;"),
                                 "}",
                                 "template <typename T>",
                                 "__device__ __forceinline__ const T *grid_xhom_or_dxhom_ptr(const T *s_Xhom, const T *s_dXhom, const int q_index, const int joint_id) {",
                                 "    return grid_q_index_affects_joint(q_index, joint_id) ? &s_dXhom[16 * q_index] : &s_Xhom[16 * joint_id];",
                                 "}",
                                 "template <typename T>",
                                 "__device__ __forceinline__ const T *grid_xhom_or_dxhom_or_d2xhom_ptr(const T *s_Xhom, const T *s_dXhom, const T *s_d2Xhom, const int q_index_i, const int q_index_j, const int joint_id) {",
                                 "    const bool i_affects = grid_q_index_affects_joint(q_index_i, joint_id);",
                                 "    const bool j_affects = grid_q_index_affects_joint(q_index_j, joint_id);",
                                 "    if (i_affects && j_affects) { return &s_d2Xhom[grid_d2xhom_offset(q_index_i, q_index_j)]; }",
                                 "    if (i_affects) { return &s_dXhom[16 * q_index_i]; }",
                                 "    if (j_affects) { return &s_dXhom[16 * q_index_j]; }",
                                 "    return &s_Xhom[16 * joint_id];",
                                 "}",
                                 "template <typename T, bool USE_DA_DF_SPILL>",
                                 "__device__ inline T *grid_id_du_temp_ptr(T *s_temp, T *d_temp_spill, int index) {",
                                 "    if (!USE_DA_DF_SPILL) { return &s_temp[index]; }",
                                 "    if (index >= ID_DU_TEMP_SPILL_START && index < ID_DU_TEMP_SPILL_END) {",
                                 "        return &d_temp_spill[index - ID_DU_TEMP_SPILL_START];",
                                 "    }",
                                 "    if (index >= ID_DU_TEMP_SPILL_END) {",
                                 "        return &s_temp[index - ID_DU_TEMP_SPILL_COUNT];",
                                 "    }",
                                 "    return &s_temp[index];",
                                 "}",
                                 ""])
        # then the structs
        # first add the struct
        self.gen_add_code_line("// Define custom structs")
        struct_lines = ["template <typename T>", \
                        "struct robotModel {", \
                        "    T *d_XImats;", \
                        "    int *d_topology_helpers;"]
        if getattr(self, "runtime_inertia", False):
            # D.4 / Phase 5: flag-gated mutable inertia table (10*NB or 10*(NB+1)
            # floats, body-indexed, in the frozen regressor basis). Emitted ONLY
            # under runtime_inertia so the baked struct stays byte-identical.
            struct_lines.append("    T *d_inertia_params;")
        if getattr(self, "runtime_transform", False):
            # runtime_transform: flag-gated mutable joint-origin table (6*NB
            # floats, joint-indexed, raw [x,y,z,r,p,y] basis). Emitted ONLY under
            # runtime_transform so the baked struct stays byte-identical.
            struct_lines.append("    T *d_transform_params;")
        if getattr(self, "runtime_joint_dynamics", False):
            # runtime_joint_dynamics: flag-gated mutable joint-dynamics table
            # (2*nv floats, v-slot indexed, [damping||friction], alpha-folded).
            # Emitted ONLY under runtime_joint_dynamics so the baked struct stays
            # byte-identical.
            struct_lines.append("    T *d_joint_dynamics_params;")
        struct_lines.append("};")
        self.gen_add_code_lines(struct_lines)
        self.gen_add_code_lines(["template <typename T, gridDataKind KIND = GRID_DATA_ALL>", \
                                 "struct gridData {", \
                                 "    // GPU INPUTS", \
                                 "    T *d_q_qd_u;", \
                                 "    T *d_q_qd;", \
                                 "    T *d_q;", \
                                 # external forces: body-major 6*NUM_BODIES local-frame, per timestep (zeroed by default)
                                 "    T *d_f_ext;", \
                                 "    // CPU INPUTS", \
                                 "    T *h_q_qd_u;", \
                                 "    T *h_q_qd;", \
                                 "    T *h_q;", \
                                 "    T *h_f_ext;", \
                                 "    // GPU OUTPUTS", \
                                 "    T *d_c;", \
                                 "    T *d_Minv;", \
                                 "    T *d_qdd;", \
                                 "    T *d_M;", \
                                 "    T *d_dc_du;", \
                                 "    T *d_df_du;", \
                                 # f_ext gradient column (section A): dtau/dfext = -J^T,
                                 # dqdd/dfext = M^-1 J^T; each nv x (6*NB), body-major.
                                 "    T *d_dtau_dfext;",
                                 "    T *d_dqdd_dfext;",
                                 "    T *d_f_ext_gradient_dq;  // -dJ^T/dq = d(inverse_dynamics_gradient)/dfext, nv*6NB*nv (both base modes)",
                                 # R2: regressor + FD param-gradient outputs (each nv x 10*NUM_BODIES)
                                 "    T *d_Y;          // inverse_dynamics_regressor (tau = Y . pi), nv*10NB",
                                 "    T *d_dqdd_dpi;   // forward_dynamics_parameter_gradient (-Minv . Y), nv*10NB",
                                 # PS5 energy regressors (each 10*NUM_BODIES; KE=y_KE.pi, PE=y_PE.pi)
                                 "    T *d_ke_regressor;   // kinetic_energy_regressor (KE = y_KE . pi), 10NB",
                                 "    T *d_pe_regressor;   // potential_energy_regressor (PE = y_PE . pi), 10NB",
                                 # PS5 Coriolis matrix C(q,qd), row-major nv x nv (C qd+g = nonlinear_effects)
                                 "    T *d_coriolis;       // coriolis_matrix C(q,qd), nv*nv",
                                 # PS5 dCCRBA: dccrba tensor (6*nv*nv) + cmm_time_variation Adot (6*nv)
                                 "    T *d_dccrba;             // dccrba dA_dq[:,k,m], 6*nv*nv",
                                 "    T *d_cmm_time_variation; // cmm_time_variation Adot, 6*nv"]
                                 + [
                                 "    T *d_end_effector_pose;", \
                                 "    T *d_end_effector_pose_gradient;", \
                                 "    T *d_end_effector_pose_hessian;", \
                                 # E2/S1: general-frame Jacobian outputs (frame_jacobian / frame_jacobian_dot:
                                 # 6 x NUM_VEL each; osc_inertia: 6 x 6 task inertia Lambda)
                                 "    T *d_frame_jacobian;       // frame_jacobian (6 x NUM_VEL)", \
                                 "    T *d_frame_jacobian_dot;   // frame_jacobian_dot (6 x NUM_VEL)", \
                                 "    T *d_osc_inertia;          // osc_inertia Lambda (6 x 6)", \
                                 # runtime-target pose/pose-gradient (additive, opt-in). The 3-vector
                                 # runtime offset is a single device buffer, host-init to {0,0,0}.
                                 "    T *d_eePose;               // end_effector_pose_runtime (6 = [xyz;rpy])", \
                                 "    T *d_eePoseGrad;           // end_effector_pose_gradient_runtime (6 x NUM_VEL)", \
                                 "    T *d_eepose_runtime_offset; // runtime 3-vector point offset (target frame)"] \
                                 # W1b.3 batched multi-target world positions / position-gradient (opt-in via
                                 # multi_target_batch). Emitted ONLY for an MT robot -- a Python-side condition,
                                 # not a #if, so a non-MT header stays BYTE-IDENTICAL (no inert preprocessor
                                 # text). NUM_MULTI_TARGETS (the malloc size) is emitted before gen_init_gridData.
                                 + ([
                                 "    T *d_multi_target_position;          // multi_target_position (3 x NUM_MULTI_TARGETS)",
                                 "    T *d_multi_target_position_gradient; // multi_target_position_gradient (3 x NUM_VEL x NUM_MULTI_TARGETS)",
                                 ] if getattr(self, "_has_multi_target_position", False) else []) \
                                 + [
                                 "    unsigned char *d_workspace;", \
                                 # idsva_so - d2tau_dq2, d2tau_dqd2, d2tau_dvdq, dM_dq
                                 "    T *d_idsva_so;", \
                                 # fdsva_so - d2a_dq2, d2a_dv2, d2a_dvdq, d2a_dtdq
                                 "    T *d_df2;", \
                                 # integrator outputs
                                 "    T *d_x_kp1;", \
                                 "    T *d_dAB;", \
                                 # G2 centroidal quick-wins outputs
                                 "    T *d_com;", \
                                 "    T *d_ccrba;", \
                                 "    T *d_energy;", \
                                 "    // CPU OUTPUTS", \
                                 "    T *h_c;", \
                                 "    T *h_Minv;", \
                                 "    T *h_qdd;", \
                                 "    T *h_M;", \
                                 "    T *h_dc_du;", \
                                 "    T *h_df_du;", \
                                 "    T *h_dtau_dfext;",
                                 "    T *h_dqdd_dfext;",
                                 "    T *h_f_ext_gradient_dq;  // -dJ^T/dq, nv*6NB*nv (both base modes)",
                                 # R2: regressor + FD param-gradient outputs (each nv x 10*NUM_BODIES)
                                 "    T *h_Y;",
                                 "    T *h_dqdd_dpi;",
                                 # PS5 energy regressors
                                 "    T *h_ke_regressor;",
                                 "    T *h_pe_regressor;",
                                 # PS5 Coriolis matrix C(q,qd)
                                 "    T *h_coriolis;",
                                 # PS5 dCCRBA host buffers
                                 "    T *h_dccrba;",
                                 "    T *h_cmm_time_variation;"]
                                 + [
                                 "    T *h_end_effector_pose;", \
                                 "    T *h_end_effector_pose_gradient;", \
                                 "    T *h_end_effector_pose_hessian;", \
                                 # E2/S1: general-frame Jacobian host buffers
                                 "    T *h_frame_jacobian;", \
                                 "    T *h_frame_jacobian_dot;", \
                                 "    T *h_osc_inertia;", \
                                 "    T *h_eePose;", \
                                 "    T *h_eePoseGrad;"] \
                                 # W1b.3 batched multi-target host buffers (opt-in; Python-conditional like d_ above)
                                 + ([
                                 "    T *h_multi_target_position;",
                                 "    T *h_multi_target_position_gradient;",
                                 ] if getattr(self, "_has_multi_target_position", False) else []) \
                                 + [
                                 # idsva_so - d2tau_dq2, d2tau_dqd2, d2tau_dvdq, dM_dq
                                 "    T *h_idsva_so;", \
                                 # fdsva_so - d2a_dq2, d2a_dv2, d2a_dvdq, d2a_dtdq
                                 "    T *h_df2;", \
                                 # integrator outputs
                                 "    T *h_x_kp1;", \
                                 "    T *h_dAB;", \
                                 # G2 centroidal quick-wins outputs
                                 "    T *h_com;", \
                                 "    T *h_ccrba;", \
                                 "    T *h_energy;", \
                                 "};"])

    def gen_init_gridData(self):
        code_lines = (["gridData<T, KIND> *hd_data = (gridData<T, KIND> *)calloc(1, sizeof(gridData<T, KIND>));",
                      "const bool needs_dynamics = KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS;",
                      "const bool needs_kinematics = KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS;",
                      "// input variables used by dynamics and/or kinematics",
                      "if (needs_dynamics || needs_kinematics) {", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_q_qd_u, 3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_q, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                      "    hd_data->h_q_qd_u = (T *)malloc(3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                      "    hd_data->h_q = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                      "    // external forces (body-major 6*NUM_BODIES local-frame); zeroed so the", \
                      "    // default (no-fext) path subtracts nothing. Users overwrite h_f_ext and", \
                      "    // copy to d_f_ext to apply external forces.", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_f_ext, 6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T)));", \
                      "    gpuErrchk(cudaMemset(hd_data->d_f_ext, 0, 6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T)));", \
                      "    hd_data->h_f_ext = (T *)calloc(6*NUM_BODIES*NUM_TIMESTEPS, sizeof(T));", \
                      "}", \
                      "if (needs_dynamics) {", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_q_qd, 2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                      "    hd_data->h_q_qd = (T *)malloc(2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                      "}", \
                      "// dynamics outputs and fallback workspace", \
                      "if (needs_dynamics) {", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_c, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_Minv, NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_qdd, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_M, NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                      "    #if GRID_HAS_INVERSE_DYNAMICS_GRADIENT", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_dc_du, 2*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                      "    #endif", \
                      "    #if GRID_HAS_FORWARD_DYNAMICS_GRADIENT", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_df_du, 2*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                      "    #endif", \
                      "    // f_ext gradient column (section A): dtau/dfext, dqdd/dfext are each nv x (6*NB)", \
                      "    #if GRID_HAS_F_EXT_GRADIENT", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_dtau_dfext, NUM_VEL*6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T)));", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_dqdd_dfext, NUM_VEL*6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T)));", \
                      "    #endif", \
                      "    hd_data->h_dtau_dfext = (T *)malloc(NUM_VEL*6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));",
                      "    hd_data->h_dqdd_dfext = (T *)malloc(NUM_VEL*6*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));",
                      "    // f_ext A.3: -dJ^T/dq = d(inverse_dynamics_gradient)/dfext, nv*6NB*nv (fixed base only; the largest per-timestep buffer)",
                      "    #if GRID_HAS_F_EXT_GRADIENT_DQ",
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_f_ext_gradient_dq, NUM_VEL*6*NUM_BODIES*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));",
                      "    hd_data->h_f_ext_gradient_dq = (T *)malloc(NUM_VEL*6*NUM_BODIES*NUM_VEL*NUM_TIMESTEPS*sizeof(T));",
                      "    #endif",
                      "    // R2: regressor Y and FD param-gradient dqdd/dpi (each nv x 10*NUM_BODIES)",
                      "    #if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR",
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_Y, NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T)));",
                      "    hd_data->h_Y = (T *)malloc(NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));",
                      "    #endif",
                      "    #if GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT",
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_dqdd_dpi, NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T)));",
                      "    hd_data->h_dqdd_dpi = (T *)malloc(NUM_VEL*10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));",
                      "    #endif"]
                      + [
                      "    #if GRID_HAS_IDSVA_SO", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_idsva_so, SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS*sizeof(T)));", \
                      "    #endif", \
                      "    #if GRID_HAS_FDSVA_SO", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_df2, SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS*sizeof(T)));", \
                      "    #endif", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_workspace, GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*GRID_WORKSPACE_SLOTS*NUM_TIMESTEPS));", \
                      "    // Phase 3a/b/c/e: L2-pin d_workspace for its lifetime. Spilled buffers", \
                      "    // (Minv-F, FD's Minv-F, ABA's inner scratch, FDSVA_SO's df_du/Minv) are", \
                      "    // recursion-hot — L2 pinning narrows the smem→HBM gap to smem→L2.", \
                      "    gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*GRID_WORKSPACE_SLOTS*NUM_TIMESTEPS));", \
                      "    hd_data->h_c = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                      "    hd_data->h_Minv = (T *)malloc(NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                      "    hd_data->h_M = (T *)malloc(NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                      "    hd_data->h_qdd = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                      "    #if GRID_HAS_INVERSE_DYNAMICS_GRADIENT", \
                      "    hd_data->h_dc_du = (T *)malloc(2*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                      "    #endif", \
                      "    #if GRID_HAS_FORWARD_DYNAMICS_GRADIENT", \
                      "    hd_data->h_df_du = (T *)malloc(2*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                      "    #endif", \
                      "    #if GRID_HAS_IDSVA_SO", \
                      "    hd_data->h_idsva_so = (T *)malloc(SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS*sizeof(T));", \
                      "    #endif", \
                      "    #if GRID_HAS_FDSVA_SO", \
                      "    hd_data->h_df2 = (T *)malloc(SECOND_ORDER_TENSOR_SIZE*NUM_TIMESTEPS*sizeof(T));", \
                      "    #endif", \
                      "    #if GRID_HAS_INTEGRATOR", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_x_kp1, 2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                      "    hd_data->h_x_kp1 = (T *)malloc(2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                      "    #endif", \
                      "    #if GRID_HAS_INTEGRATOR_GRADIENT", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_dAB, 2*NUM_JOINTS*3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                      "    hd_data->h_dAB = (T *)malloc(2*NUM_JOINTS*3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                      "    #endif", \
                      "}", \
                      "// kinematics outputs", \
                      "if (needs_kinematics) {", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_end_effector_pose, 6*NUM_EES*NUM_TIMESTEPS*sizeof(T)));", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_end_effector_pose_gradient, 6*NUM_EES*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_end_effector_pose_hessian, 6*NUM_EES*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                      "    if ((GRID_END_EFFECTOR_POSE_HESSIAN_USES_WORKSPACE_TEMP || GRID_END_EFFECTOR_POSE_GRADIENT_USES_WORKSPACE_TEMP || GRID_DCCRBA_USES_WORKSPACE_TEMP || GRID_OSC_INERTIA_USES_WORKSPACE) && hd_data->d_workspace == nullptr) {gpuErrchk(cudaMalloc((void**)&hd_data->d_workspace, GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*GRID_WORKSPACE_SLOTS*NUM_TIMESTEPS));}", \
                      "    hd_data->h_end_effector_pose = (T *)malloc(6*NUM_EES*NUM_TIMESTEPS*sizeof(T));", \
                      "    hd_data->h_end_effector_pose_gradient = (T *)malloc(6*NUM_EES*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                      "    hd_data->h_end_effector_pose_hessian = (T *)malloc(6*NUM_EES*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                      # E2/S1: general-frame Jacobian outputs (J / Jdot: 6*NV each ; Lambda: 36)
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_frame_jacobian, 6*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_frame_jacobian_dot, 6*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_osc_inertia, 36*NUM_TIMESTEPS*sizeof(T)));", \
                      "    hd_data->h_frame_jacobian = (T *)malloc(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                      "    hd_data->h_frame_jacobian_dot = (T *)malloc(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                      "    hd_data->h_osc_inertia = (T *)malloc(36*NUM_TIMESTEPS*sizeof(T));", \
                      # runtime-target pose / pose-gradient (additive, opt-in). The runtime
                      # 3-vector offset is a single device buffer init to {0,0,0} (frame origin);
                      # a binding overwrites it before the call to request a nonzero offset.
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_eePose, 6*NUM_TIMESTEPS*sizeof(T)));", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_eePoseGrad, 6*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_eepose_runtime_offset, 3*sizeof(T)));", \
                      "    gpuErrchk(cudaMemset(hd_data->d_eepose_runtime_offset, 0, 3*sizeof(T)));", \
                      "    hd_data->h_eePose = (T *)malloc(6*NUM_TIMESTEPS*sizeof(T));", \
                      "    hd_data->h_eePoseGrad = (T *)malloc(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T));"] \
                      # W1b.3 batched multi-target. Emitted ONLY for an MT robot (Python-conditional,
                      # not a #if) so a non-MT header is byte-identical AND never references
                      # NUM_MULTI_TARGETS, which only exists when the batch is emitted.
                      #
                      # The DEVICE buffers carry a 1024-element FLOOR. The single_timing kernels'
                      # anti-LICM feedback indexes d_<out>[(rep + 0x3FF) & 0x3FF] -> up to slot 1023,
                      # so gen_anti_licm_{input_reload,output_write} REQUIRE >= 1024 output slots.
                      # Every other algo meets that naturally (ee_pose: 6*NUM_EES*256 = 1536), but
                      # multi_target scales with the BATCH: a small batch under-allocates (a 1-target
                      # batch is only 3*1*256 = 768 < 1024) and the timing kernel reads OOB (caught by
                      # compute-sanitizer memcheck). The floor costs a few KB and makes it unbreakable;
                      # the D2H copy still moves only the natural 3*NUM_MULTI_TARGETS*n elements.
                      + ([
                      "    const int MT_POS_SLOTS  = 3*NUM_MULTI_TARGETS*NUM_TIMESTEPS;",
                      "    const int MT_GRAD_SLOTS = 3*NUM_VEL*NUM_MULTI_TARGETS*NUM_TIMESTEPS;",
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_multi_target_position, (MT_POS_SLOTS > 1024 ? MT_POS_SLOTS : 1024)*sizeof(T)));",
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_multi_target_position_gradient, (MT_GRAD_SLOTS > 1024 ? MT_GRAD_SLOTS : 1024)*sizeof(T)));",
                      "    hd_data->h_multi_target_position = (T *)malloc(MT_POS_SLOTS*sizeof(T));",
                      "    hd_data->h_multi_target_position_gradient = (T *)malloc(MT_GRAD_SLOTS*sizeof(T));",
                      ] if getattr(self, "_has_multi_target_position", False) else []) \
                      + [
                      "}", \
                      "// G2 centroidal quick-wins outputs (com: 3+3*NV ; ccrba: 6*NV+6 ; energy: 3)", \
                      "if (needs_dynamics || needs_kinematics) {", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_com, (3+3*NUM_VEL)*NUM_TIMESTEPS*sizeof(T)));", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_ccrba, (6*NUM_VEL+6)*NUM_TIMESTEPS*sizeof(T)));", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_energy, 3*NUM_TIMESTEPS*sizeof(T)));", \
                      "    hd_data->h_com = (T *)malloc((3+3*NUM_VEL)*NUM_TIMESTEPS*sizeof(T));", \
                      "    hd_data->h_ccrba = (T *)malloc((6*NUM_VEL+6)*NUM_TIMESTEPS*sizeof(T));", \
                      "    hd_data->h_energy = (T *)malloc(3*NUM_TIMESTEPS*sizeof(T));", \
                      "    // PS5 energy regressors (each 10*NUM_BODIES): KE (dynamics) + PE (kinematics)", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_ke_regressor, 10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T)));", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_pe_regressor, 10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T)));", \
                      "    hd_data->h_ke_regressor = (T *)malloc(10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));", \
                      "    hd_data->h_pe_regressor = (T *)malloc(10*NUM_BODIES*NUM_TIMESTEPS*sizeof(T));", \
                      "    // PS5 Coriolis matrix C(q,qd) (nv x nv)", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_coriolis, NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                      "    hd_data->h_coriolis = (T *)malloc(NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                      "    // PS5 dCCRBA: dccrba tensor (6*nv*nv) + cmm_time_variation Adot (6*nv)", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_dccrba, 6*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                      "    gpuErrchk(cudaMalloc((void**)&hd_data->d_cmm_time_variation, 6*NUM_VEL*NUM_TIMESTEPS*sizeof(T)));", \
                      "    hd_data->h_dccrba = (T *)malloc(6*NUM_VEL*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                      "    hd_data->h_cmm_time_variation = (T *)malloc(6*NUM_VEL*NUM_TIMESTEPS*sizeof(T));", \
                      "}", \
                      "return hd_data;"])
        # generate as templated or not function
        self.gen_add_func_doc("Allocated device and host memory for all computations",
                              [], [], "A pointer to the gridData struct of pointers")
        self.gen_add_code_line("template <typename T, int NUM_TIMESTEPS, gridDataKind KIND = GRID_DATA_ALL>")
        self.gen_add_code_line("__host__")
        self.gen_add_code_line("gridData<T, KIND> *init_gridData(){", True)
        self.gen_add_code_lines(code_lines)
        self.gen_add_end_function()
        self.gen_add_func_doc("Allocated device and host memory for all computations",
                              [], ["Max number of timesteps in the trajectory"], "A pointer to the gridData struct of pointers")
        self.gen_add_code_line("template <typename T, gridDataKind KIND = GRID_DATA_ALL>")
        self.gen_add_code_line("__host__")
        self.gen_add_code_line("gridData<T, KIND> *init_gridData(int NUM_TIMESTEPS){", True)
        self.gen_add_code_lines(code_lines)
        self.gen_add_end_function()

    # Manifest of every algorithm kernel that needs cudaFuncSetAttribute
    # (MaxDynamicSharedMemorySize). Each entry:
    #   (algo_label, algo_short_name, gate_attr, bytes_macro, [(kernel_name, signature), ...])
    #
    # algo_short_name matches the keys used in self.generated_algorithms (see
    # _normalize_codegen_algorithms). gate_attr is the `self.generate_*` bool —
    # honored when present, else we rely on algo_short_name membership.
    #
    # Applied to EVERY kernel, not just the historically-large ones: without
    # cudaFuncSetAttribute(MaxDynamicSharedMemorySize, BYTES), a kernel whose
    # runtime dynamic smem exceeds the device default per-block limit (48 KB on
    # most consumer NVIDIA GPUs incl. sm_8x) fails to launch silently with
    # cudaErrorInvalidConfiguration — the error doesn't propagate through
    # cudaDeviceSynchronize() reliably, so timings come back as bogus ~0 us (hit
    # on g1 floating where ABA/FD/MINV need 52-57 KB). No-op when BYTES is already
    # under the device default.
    # Default False: the f_ext A.3 (-dJ^T/dq) kernel is registered only when
    # gen_f_ext_gradient actually emitted it (set True whenever f_ext_gradient runs).
    _f_ext_gradient_dq_emitted = False

    # ── Step 2: per-algo kernel-overload SIGNATURES (the irregular C++ payload).
    # Keyed by algo_short, INSERTION ORDER == the historical KERNEL_ATTR_MANIFEST
    # order (byte-identity: the emitted cudaFuncSetAttribute block + its alias
    # counter follow this order). The per-entry HEAD (algo_label, gate_attr,
    # bytes_macro) is NO LONGER restated here — it is derived from the descriptor
    # table (algo_registry.ALGO_DESCRIPTORS) below, so gate/bytes live in exactly
    # one place. algo_label == algo_short for every pin entry (registry invariant).
    KERNEL_OVERLOADS = {
        "inverse_dynamics": [
            ("inverse_dynamics_kernel<T>",
             "void (*)(T *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)"),
            ("inverse_dynamics_kernel<T>",
             "void (*)(T *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
            ("inverse_dynamics_kernel_single_timing<T>",
             "void (*)(T *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)"),
            ("inverse_dynamics_kernel_single_timing<T>",
             "void (*)(T *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
        ],
        "minv": [
            ("minv_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
            ("minv_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ],
        "forward_dynamics": [
            ("forward_dynamics_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
            ("forward_dynamics_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
        ],
        "aba": [
            ("aba_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
            ("aba_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
        ],
        "crba": [
            ("crba_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
            ("crba_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ],
        "end_effector_pose": [
            ("end_effector_pose_kernel<T>",
             "void (*)(T *, const T *, const int, const robotModel<T> *, const int)"),
            ("end_effector_pose_kernel_single_timing<T>",
             "void (*)(T *, const T *, const int, const robotModel<T> *, const int)"),
        ],
        "end_effector_pose_gradient": [
            ("end_effector_pose_gradient_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
            ("end_effector_pose_gradient_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ],
        "inverse_dynamics_gradient": [
            ("inverse_dynamics_gradient_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)"),
            ("inverse_dynamics_gradient_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
            ("inverse_dynamics_gradient_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)"),
            ("inverse_dynamics_gradient_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
        ],
        "forward_dynamics_gradient": [
            ("forward_dynamics_gradient_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const T *, const T *, T *, const robotModel<T> *, const T, const int)"),
            ("forward_dynamics_gradient_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
            ("forward_dynamics_gradient_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const T *, const T *, T *, const robotModel<T> *, const T, const int)"),
            ("forward_dynamics_gradient_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
        ],
        # g1-spill: f_ext_gradient_kernel gained `unsigned char *d_workspace` as its
        # 3rd arg (after the two outputs) so s_dqdd_dfext can spill there at LITE/MINIMAL.
        "f_ext_gradient": [
            ("f_ext_gradient_kernel<T>",
             "void (*)(T *, T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
            ("f_ext_gradient_kernel_single_timing<T>",
             "void (*)(T *, T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ],
        # A.3 (-dJ^T/dq): own kernel + smem macro, fixed-base only. Gated (descriptor
        # gate_attr=_f_ext_gradient_dq_emitted) so the floating-base header — which has
        # neither the kernel nor the F_EXT_GRADIENT_DQ_* macro — never references them.
        # h2_plus-spill: f_ext_gradient_dq_kernel gained `unsigned char *d_workspace` as
        # its 2nd arg (after the output) so the s_JTp/s_JTm pair can spill there at rung 1.
        "f_ext_gradient_dq": [
            ("f_ext_gradient_dq_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
            ("f_ext_gradient_dq_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ],
        # E1 joint-torque regressor: Y is nv x 10*NUM_BODIES, can exceed the 48 KB
        # default dynamic-smem cap on big robots (g1: ~55 KB), so it MUST opt in.
        "inverse_dynamics_regressor": [
            ("inverse_dynamics_regressor_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
            ("inverse_dynamics_regressor_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ],
        # PS5 energy regressors (each output 10*NUM_BODIES; opt-in like the joint-torque
        # regressor so big-robot launches set the dynamic-smem attr).
        "kinetic_energy_regressor": [
            ("kinetic_energy_regressor_kernel<T>",
             "void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)"),
            ("kinetic_energy_regressor_kernel_single_timing<T>",
             "void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ],
        "potential_energy_regressor": [
            ("potential_energy_regressor_kernel<T>",
             "void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)"),
            ("potential_energy_regressor_kernel_single_timing<T>",
             "void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ],
        # FD param gradient dqdd/dpi = -Minv . Y: output is nv x 10*NUM_BODIES (same
        # size class as the regressor), can exceed the 48 KB default cap; opt in.
        # g1-spill: forward_dynamics_parameter_gradient_kernel gained `unsigned char *d_workspace`
        # as its 2nd arg (after d_dqdd_dpi) so s_Y can spill there at LITE/MINIMAL.
        "forward_dynamics_parameter_gradient": [
            ("forward_dynamics_parameter_gradient_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
            ("forward_dynamics_parameter_gradient_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ],
        "idsva_so_body_frame": [
            ("idsva_so_body_frame_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
            ("idsva_so_body_frame_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ],
        # world-frame single-pass alternative (opt-in via enable_idsva_so_world_frame).
        # Now takes d_workspace (unified signature; cold buffers spill there at LITE/MINIMAL).
        # Uses its own shared-mem macro (~25 KB for g1 vs shim's ~162 KB).
        "idsva_so_world_frame": [
            ("idsva_so_world_frame_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
            ("idsva_so_world_frame_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ],
        "fdsva_so": [
            ("fdsva_so_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
            ("fdsva_so_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
        ],
        # Integrator kernels are templated on IntegratorType IT (a non-type param
        # that does not change the function signature). Each IT is a distinct
        # __global__ instantiation, so cudaFuncSetAttribute must run for ALL of
        # them — otherwise a non-Euler IT whose floating-base arena exceeds the
        # 48 KB device default launches with cudaErrorInvalidValue while Euler
        # (the only one historically registered) succeeds.
        "integrator": [
            (f"integrator_kernel{suffix}<T, IntegratorType::{it}>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, T *, const T, const T, const int)")
            for suffix in ("", "_single_timing")
            for it in ("EULER", "SEMI_IMPLICIT_EULER", "MIDPOINT", "RK3", "RK4", "TRAPEZOIDAL")
        ],
        "integrator_gradient": [
            (f"integrator_gradient_kernel{suffix}<T, IntegratorType::{it}>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, T *, const T, const T, const int)")
            for suffix in ("", "_single_timing")
            for it in ("EULER", "SEMI_IMPLICIT_EULER", "MIDPOINT", "RK3", "RK4", "TRAPEZOIDAL")
        ],
        "integrator_with_gradient": [
            (f"integrator_with_gradient_kernel{suffix}<T, IntegratorType::{it}>",
             "void (*)(T *, T *, unsigned char *, const T *, const int, const robotModel<T> *, T *, const T, const T, const int)")
            for suffix in ("", "_single_timing")
            for it in ("EULER", "SEMI_IMPLICIT_EULER", "MIDPOINT", "RK3", "RK4", "TRAPEZOIDAL")
        ],
        # ee_pose_hessian is special: only emitted when its shared-mem fits the
        # GRID_CUDA_TARGET_SHARED_MEM_BYTES budget at compile time. The runtime
        # guard wraps the cudaFuncSetAttribute call.
        "end_effector_pose_hessian": [
            ("end_effector_pose_hessian_kernel<T>",
             "void (*)(T *, T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
            ("end_effector_pose_hessian_kernel_single_timing<T>",
             "void (*)(T *, T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ],
        # G2 centroidal quick-wins. R6: each registers on its OWN key (matching
        # gen_centroidal_quickwins' per-key emit) — algo_short keys an entry that
        # is in generated_algorithms exactly when THAT centroidal fn was emitted.
        "generalized_gravity": [
            ("generalized_gravity_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
            ("generalized_gravity_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ],
        "nonlinear_effects": [
            ("nonlinear_effects_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
            ("nonlinear_effects_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ],
        # PS5 Coriolis matrix C(q,qd): nv x nv output can exceed the 48 KB default
        # dynamic-smem cap on big robots, so it MUST opt in. Kernel takes d_workspace
        # as its 2nd arg (reserved for a future big-robot spill; unused at FULL).
        "coriolis_matrix": [
            ("coriolis_matrix_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
            ("coriolis_matrix_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ],
        # PS5 dCCRBA: cmm_time_variation (Adot 6*nv; no spill, no d_workspace arg) and
        # dccrba (6*nv*nv tensor; spill -> d_workspace as the 2nd arg). REQUIRED so
        # init_grid runs cudaFuncSetAttribute (the 6*nv*nv output blows the 48 KB cap).
        "cmm_time_variation": [
            ("cmm_time_variation_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
            ("cmm_time_variation_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ],
        "dccrba": [
            ("dccrba_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
            ("dccrba_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ],
        # E2/S1 general-frame Jacobian family (opt-in; gated on membership in
        # generated_algorithms via algo_short). The kernels take the target frame
        # at RUNTIME: (T *out, const T *q, const int stride_q, const int target_jid,
        # const int reference_frame, robotModel, int N).
        "frame_jacobian": [
            ("frame_jacobian_kernel<T>",
             "void (*)(T *, const T *, const int, const int, const int, const robotModel<T> *, const int)"),
            ("frame_jacobian_kernel_single_timing<T>",
             "void (*)(T *, const T *, const int, const int, const int, const robotModel<T> *, const int)"),
        ],
        "frame_jacobian_dot": [
            ("frame_jacobian_dot_kernel<T>",
             "void (*)(T *, const T *, const int, const int, const int, const robotModel<T> *, const int)"),
            ("frame_jacobian_dot_kernel_single_timing<T>",
             "void (*)(T *, const T *, const int, const int, const int, const robotModel<T> *, const int)"),
        ],
        "osc_inertia": [
            ("osc_inertia_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
            ("osc_inertia_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ],
        # Runtime-target pose / pose-gradient (opt-in). Kernels take target_jid +
        # the runtime offset pointer: (T *out, const T *q, const int stride_q,
        # const int target_jid, const T *offset, robotModel, int N).
        "end_effector_pose_runtime": [
            ("end_effector_pose_runtime_kernel<T>",
             "void (*)(T *, const T *, const int, const int, const T *, const robotModel<T> *, const int)"),
            ("end_effector_pose_runtime_kernel_single_timing<T>",
             "void (*)(T *, const T *, const int, const int, const T *, const robotModel<T> *, const int)"),
        ],
        "end_effector_pose_gradient_runtime": [
            ("end_effector_pose_gradient_runtime_kernel<T>",
             "void (*)(T *, const T *, const int, const int, const T *, const robotModel<T> *, const int)"),
            ("end_effector_pose_gradient_runtime_kernel_single_timing<T>",
             "void (*)(T *, const T *, const int, const int, const T *, const robotModel<T> *, const int)"),
        ],
        # W1b.3 batched multi-target world positions / position-gradient (opt-in via
        # multi_target_batch; gated on _has_multi_target_position). Same launch shape as
        # end_effector_pose: (T *out, const T *q, const int stride_q, robotModel, int N).
        "multi_target_position": [
            ("multi_target_position_kernel<T>",
             "void (*)(T *, const T *, const int, const robotModel<T> *, const int)"),
            ("multi_target_position_kernel_single_timing<T>",
             "void (*)(T *, const T *, const int, const robotModel<T> *, const int)"),
        ],
        "multi_target_position_gradient": [
            ("multi_target_position_gradient_kernel<T>",
             "void (*)(T *, const T *, const int, const robotModel<T> *, const int)"),
            ("multi_target_position_gradient_kernel_single_timing<T>",
             "void (*)(T *, const T *, const int, const robotModel<T> *, const int)"),
        ],
        "com": [
            ("com_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
            ("com_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ],
        "ccrba": [
            ("ccrba_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
            ("ccrba_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ],
        "energy": [
            ("energy_kernel<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
            ("energy_kernel_single_timing<T>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ],
    }

    # KERNEL_ATTR_MANIFEST head (algo_label, algo_short, gate_attr, bytes_macro)
    # DERIVED from the descriptor rows — single source of truth for gate/bytes. The
    # `kernels` payload comes from KERNEL_OVERLOADS. Order follows KERNEL_OVERLOADS
    # (== the historical manifest order) so the emitted attr block is byte-identical.
    KERNEL_ATTR_MANIFEST = [
        (short, short, descriptor_for(short).gate_attr, descriptor_for(short).bytes_macro, kernels)
        for short, kernels in KERNEL_OVERLOADS.items()
    ]

    # ── Step 2: floating mjx-twin (MUJOCO_OUTPUT=true) kernel-overload SIGNATURES.
    # The mjx twin is a SUBSET of the pin kernels with the trailing `, GRID_DEFAULT_
    # RESOURCE_TIER, true` template flag; it has NO single_timing twin. Payload only;
    # the mujoco_manifest head (label=short+"(mjx)", gate_attr, bytes_macro) is derived
    # from the descriptor rows in gen_init_close_grid. Insertion order == historical
    # mujoco_manifest order (byte-identity of the mjx portion of the attr block). Some
    # sigs differ from the pin base: id / id-grad use the qdd overload; fd-grad uses the
    # single-output overload; integrator families fan out over their mjx IntegratorType set.
    MJX_KERNEL_OVERLOADS = {
        "inverse_dynamics": [("inverse_dynamics_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
             "void (*)(T *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)")],
        "minv": [("minv_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)")],
        "forward_dynamics": [("forward_dynamics_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
             "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)")],
        "aba": [("aba_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
             "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)")],
        "crba": [("crba_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)")],
        "end_effector_pose": [("end_effector_pose_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
             "void (*)(T *, const T *, const int, const robotModel<T> *, const int)")],
        # osc_inertia(mjx): Lambda is frame-invariant; quat reorder on input. Needs its
        # >48KB dynamic-smem opt-in (passes on go2 where arena<48KB; fails on big floating).
        "osc_inertia": [("osc_inertia_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)")],
        "end_effector_pose_gradient": [("end_effector_pose_gradient_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)")],
        "end_effector_pose_hessian": [("end_effector_pose_hessian_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
             "void (*)(T *, T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)")],
        "inverse_dynamics_gradient": [("inverse_dynamics_gradient_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
             "void (*)(T *, unsigned char *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)")],
        "forward_dynamics_gradient": [("forward_dynamics_gradient_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
             "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)")],
        "inverse_dynamics_regressor": [("inverse_dynamics_regressor_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)")],
        "idsva_so_world_frame": [("idsva_so_world_frame_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)")],
        "fdsva_so": [("fdsva_so_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
             "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)")],
        "integrator": [(f"integrator_kernel<T, IntegratorType::{it}, GRID_DEFAULT_RESOURCE_TIER, true>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, T *, const T, const T, const int)")
             for it in ("EULER", "SEMI_IMPLICIT_EULER", "MIDPOINT", "RK3", "RK4")],
        # integrator_gradient mjx is single-stage (EULER / SI-Euler) only.
        "integrator_gradient": [(f"integrator_gradient_kernel<T, IntegratorType::{it}, GRID_DEFAULT_RESOURCE_TIER, true>",
             "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, T *, const T, const T, const int)")
             for it in ("EULER", "SEMI_IMPLICIT_EULER")],
    }

    def gen_init_close_grid(self):
        # set the max shared mem to account for large robots and allocate streams
        MAX_STREAMS = 3 # max needed in any of our functions
        # ----- init_grid_kernel_attrs<T>(): cudaFuncSetAttribute for every kernel --
        # Split out from init_grid so per-algo TU callers can invoke ONLY the
        # attribute-setting part (without stream allocation), needed because
        # cudaFuncSetAttribute operates on the TU-local host stub. The per-algo
        # TU split (P6-7b) calls this from a static initializer in each
        # measure_X_*_entry so its stubs get the attribute set.
        self.gen_add_func_doc("Set MaxDynamicSharedMemorySize for every algorithm kernel "
                              "(callable from any TU; idempotent). __forceinline__ is "
                              "REQUIRED so the &kernel<T> expressions resolve to the "
                              "CALLING TU's host stubs — otherwise the linker merges this "
                              "function across TUs and we set the attribute on one TU's "
                              "stubs while the launch goes through a different TU's.",
                              [], [], None)
        self.gen_add_code_line("template <typename T>")
        self.gen_add_code_line("__host__ __forceinline__")
        self.gen_add_code_line("void init_grid_kernel_attrs(){", True)
        attr_lines = ["// enable opt-in dynamic shared memory for every algorithm kernel",
                      "// Gate registration on the DEVICE opt-in max (not the codegen target):",
                      "// grid_check_dynamic_shared_memory_bytes and the bench's",
                      "// grid_kernel_fits_device both use the device cap, so registering only up",
                      "// to the smaller GRID_CUDA_TARGET_SHARED_MEM_BYTES left kernels in",
                      "// (target, device-max] checkable+launchable but UNregistered -> launching",
                      "// them failed with cudaErrorInvalidValue (e.g. the floating idsva_so",
                      "// body-frame diagnostic ~101 KB on g1). Keying on the device max keeps",
                      "// registration, the fit-check, and the launch-skip in lockstep.",
                      "size_t _grid_smem_max = 0; gpuErrchk(grid_get_max_dynamic_shared_memory_bytes(&_grid_smem_max));"]
        generated_set = getattr(self, "generated_algorithms", None)
        alias_counter = 0
        # mjx (floating MUJOCO_OUTPUT) emits a SECOND __global__ instantiation of every
        # mjx-capable kernel with the trailing MUJOCO_OUTPUT=true template flag. Those are
        # DISTINCT device functions, so cudaFuncSetAttribute must run for them too — else a
        # big mjx kernel (fdsva_so / idsva_so-world / integrator_gradient / id-grad on a
        # humanoid) whose dynamic smem exceeds the 48 KB default launches with
        # cudaErrorInvalidValue while its pin twin succeeds. The signature is identical
        # (MUJOCO_OUTPUT is a non-type param). For the qdd-overloaded kernels (inverse_dynamics
        # / inverse_dynamics_gradient) only the qdd overload carries MUJOCO_OUTPUT, so the
        # qdd-overload signature is used; integrator_gradient mjx is single-stage (EULER/SI) only.
        # Gate on floating_base, NOT self.MUJOCO_OUTPUT: the mjx kernel template flag is
        # emitted for EVERY floating-base robot (the C-ABI/jax/torch mjx wrappers are
        # #ifdef GRID_FLOATING_BASE), independent of the MUJOCO_OUTPUT constructor arg
        # (which the .so build does not set). So the MUJOCO_OUTPUT=true instantiations
        # exist for any floating .so and need their dynamic-smem attribute registered.
        # mujoco_manifest head derived from the descriptor rows (label=short+"(mjx)",
        # gate_attr/bytes_macro from the descriptor — single source of truth); the mjx
        # kernel-overload signatures come from MJX_KERNEL_OVERLOADS. Order follows that
        # dict (== the historical mujoco_manifest order) for a byte-identical attr block.
        mujoco_manifest = []
        if self.robot.floating_base:
            mujoco_manifest = [
                (short + "(mjx)", short, descriptor_for(short).gate_attr,
                 descriptor_for(short).bytes_macro, kernels)
                for short, kernels in self.MJX_KERNEL_OVERLOADS.items()
            ]
        for entry in self.KERNEL_ATTR_MANIFEST + mujoco_manifest:
            algo_label, algo_short, gate_attr, bytes_macro, kernels = entry
            # Honor the legacy generate_* gate when present; otherwise fall
            # back to membership in generated_algorithms; if neither is set
            # (legacy callers), assume the algo is generated.
            if gate_attr is not None and not getattr(self, gate_attr, True):
                continue
            if gate_attr is None and generated_set is not None and algo_short not in generated_set:
                continue
            # Wrap EVERY kernel's attribute registration in a compile-time-
            # resolvable size guard so init_grid never hard-aborts when a kernel
            # literally can't fit a device even with cudaFuncSetAttribute (e.g.
            # the floating-base idsva_so_body_frame *diagnostic* frame on h1_2 at
            # ~168 KB, or the integrator value/gradient kernels at ~103-228 KB on
            # big floating-base robots). Such kernels simply go unregistered;
            # they aren't the dispatched production path, and any code that DOES
            # launch them still runs `grid_check_dynamic_shared_memory_bytes` at
            # the host wrapper, so the fit check + attribute setup stay in
            # lockstep at the actual use site. Small kernels are always under the
            # target, so the guard is a no-op for them.
            attr_lines.append(f"if ({bytes_macro} <= _grid_smem_max) {{")
            attr_lines.append(f"    gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"{algo_label}\", {bytes_macro}));")
            for kernel_name, signature in kernels:
                alias = f"_grid_kern_alias_{alias_counter}"
                alias_counter += 1
                attr_lines.append(f"    auto {alias} = static_cast<{signature}>(&{kernel_name});")
                attr_lines.append(f"    gpuErrchk(cudaFuncSetAttribute({alias}, cudaFuncAttributeMaxDynamicSharedMemorySize, {bytes_macro}));")
            attr_lines.append("}")
        self.gen_add_code_lines(attr_lines)
        self.gen_add_end_function()

        # ----- init_grid<T>(): full init = attrs + streams (the original API) ----
        self.gen_add_func_doc("Sets MaxDynamicSharedMemorySize for every algorithm kernel and initializes streams for host functions", \
                              [], [], "A pointer to the array of streams")
        self.gen_add_code_line("template <typename T>")
        self.gen_add_code_line("__host__")
        self.gen_add_code_line("cudaStream_t *init_grid(){", True)
        init_lines = ["init_grid_kernel_attrs<T>();",
                      "gpuErrchk(cudaDeviceSynchronize());",
                      "// allocate streams",
                      "cudaStream_t *streams = (cudaStream_t *)malloc(" + str(MAX_STREAMS) + "*sizeof(cudaStream_t));",
                      "int priority, minPriority, maxPriority;",
                      "gpuErrchk(cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority));",
                      "for(int i=0; i<" + str(MAX_STREAMS) + "; i++){",
                      "    int adjusted_max = maxPriority - i; priority = adjusted_max > minPriority ? adjusted_max : minPriority;",
                      "    gpuErrchk(cudaStreamCreateWithPriority(&(streams[i]),cudaStreamNonBlocking,priority));",
                      "}", "return streams;"]
        self.gen_add_code_lines(init_lines)
        self.gen_add_end_function()
        # free the streams and all allocated data
        self.gen_add_func_doc("Frees the memory used by grid", [], ["streams allocated by init_grid", "robotModel allocated by init_robotModel", "data allocated by init_gridData"], None)
        self.gen_add_code_line("template <typename T, gridDataKind KIND = GRID_DATA_ALL>")
        self.gen_add_code_line("__host__")
        self.gen_add_code_line("void close_grid(cudaStream_t *streams, robotModel<T> *d_robotModel, gridData<T, KIND> *hd_data){", True)
        self.gen_add_code_lines(["free_robotModel(d_robotModel); // frees nested d_XImats/d_topology_helpers(+runtime tables)+struct (bare cudaFree would leak the nested arrays)", \
                                 "gpuErrchk(cudaFree(hd_data->d_q_qd_u)); gpuErrchk(cudaFree(hd_data->d_q_qd)); gpuErrchk(cudaFree(hd_data->d_q));", \
                                 "gpuErrchk(cudaFree(hd_data->d_f_ext)); free(hd_data->h_f_ext);", \
                                 "gpuErrchk(cudaFree(hd_data->d_c)); gpuErrchk(cudaFree(hd_data->d_Minv)); gpuErrchk(cudaFree(hd_data->d_qdd)); gpuErrchk(cudaFree(hd_data->d_M));", \
                                 "gpuErrchk(cudaFree(hd_data->d_dc_du)); gpuErrchk(cudaFree(hd_data->d_df_du));", \
                                 "gpuErrchk(cudaFree(hd_data->d_dtau_dfext)); gpuErrchk(cudaFree(hd_data->d_dqdd_dfext));",
                                 "free(hd_data->h_dtau_dfext); free(hd_data->h_dqdd_dfext);",
                                 "gpuErrchk(cudaFree(hd_data->d_f_ext_gradient_dq)); free(hd_data->h_f_ext_gradient_dq);",
                                 # R2: regressor Y + FD param-gradient dqdd/dpi outputs
                                 "gpuErrchk(cudaFree(hd_data->d_Y)); gpuErrchk(cudaFree(hd_data->d_dqdd_dpi));",
                                 "free(hd_data->h_Y); free(hd_data->h_dqdd_dpi);",
                                 # PS5 energy regressors
                                 "gpuErrchk(cudaFree(hd_data->d_ke_regressor)); gpuErrchk(cudaFree(hd_data->d_pe_regressor));",
                                 "free(hd_data->h_ke_regressor); free(hd_data->h_pe_regressor);",
                                 # PS5 Coriolis matrix
                                 "gpuErrchk(cudaFree(hd_data->d_coriolis)); free(hd_data->h_coriolis);",
                                 # PS5 dCCRBA
                                 "gpuErrchk(cudaFree(hd_data->d_dccrba)); gpuErrchk(cudaFree(hd_data->d_cmm_time_variation));",
                                 "free(hd_data->h_dccrba); free(hd_data->h_cmm_time_variation);"]
                                 + [
                                 "gpuErrchk(cudaFree(hd_data->d_end_effector_pose)); gpuErrchk(cudaFree(hd_data->d_end_effector_pose_gradient)); gpuErrchk(cudaFree(hd_data->d_end_effector_pose_hessian));", \
                                 # Phase 3a/b/c/e: end the L2 persisting window opened at init.
                                 "gpuErrchk(grid_end_l2_persisting(0));", \
                                 "gpuErrchk(cudaFree(hd_data->d_workspace));", \
                                # idsva_so - d2tau_dq, d2tau_dqd, d2tau_dvdq, dM_dq
                                 "gpuErrchk(cudaFree(hd_data->d_idsva_so));", \
                                 # fdsva_so - d2fd_dq2, d2fd_cross, d2fd_dqd2, d2fd_dtaudq
                                 "gpuErrchk(cudaFree(hd_data->d_df2));", \
                                 "free(hd_data->h_idsva_so); free(hd_data->h_df2);", \
                                 "free(hd_data->h_q_qd_u); free(hd_data->h_q_qd); free(hd_data->h_q);", \
                                 "free(hd_data->h_c); free(hd_data->h_Minv); free(hd_data->h_qdd); free(hd_data->h_M);", \
                                 "free(hd_data->h_dc_du); free(hd_data->h_df_du);",\
                                 "free(hd_data->h_end_effector_pose); free(hd_data->h_end_effector_pose_gradient); free(hd_data->h_end_effector_pose_hessian);", \
                                 # E2/S1: general-frame Jacobian outputs (frame_jacobian / frame_jacobian_dot / osc_inertia)
                                 "gpuErrchk(cudaFree(hd_data->d_frame_jacobian)); gpuErrchk(cudaFree(hd_data->d_frame_jacobian_dot)); gpuErrchk(cudaFree(hd_data->d_osc_inertia));", \
                                 "free(hd_data->h_frame_jacobian); free(hd_data->h_frame_jacobian_dot); free(hd_data->h_osc_inertia);", \
                                 "gpuErrchk(cudaFree(hd_data->d_eePose)); gpuErrchk(cudaFree(hd_data->d_eePoseGrad)); gpuErrchk(cudaFree(hd_data->d_eepose_runtime_offset));", \
                                 "free(hd_data->h_eePose); free(hd_data->h_eePoseGrad);"] \
                                 # W1b.3 batched multi-target (opt-in; Python-conditional, mirrors the alloc)
                                 + ([
                                 "gpuErrchk(cudaFree(hd_data->d_multi_target_position)); gpuErrchk(cudaFree(hd_data->d_multi_target_position_gradient));",
                                 "free(hd_data->h_multi_target_position); free(hd_data->h_multi_target_position_gradient);",
                                 ] if getattr(self, "_has_multi_target_position", False) else []) \
                                 + [
                                 "gpuErrchk(cudaFree(hd_data->d_x_kp1)); gpuErrchk(cudaFree(hd_data->d_dAB));", \
                                 "free(hd_data->h_x_kp1); free(hd_data->h_dAB);", \
                                 "for(int i=0; i<" + str(MAX_STREAMS) + "; i++){gpuErrchk(cudaStreamDestroy(streams[i]));} free(streams);"])
        self.gen_add_end_function()
        
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

    def gen_add_gpu_err(self):
        # add the GPU error check code
        self.gen_add_func_doc("Check for runtime errors using the CUDA API", \
                ["Adapted from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api"], \
                [],None)
        self.gen_add_code_line("__host__")
        # `inline` is required so the per-algo TU split (multiple .o files all
        # including grid.cuh) doesn't trip ODR multiple-definition errors at link.
        self.gen_add_code_line("inline void gpuAssert(cudaError_t code, const char *file, const int line, bool abort=true){", True)
        self.gen_add_code_line("if (code != cudaSuccess){", True)
        # note that below we need to escape the \n and "" to get it to print to a string or file correctly
        self.gen_add_code_line("fprintf(stderr,\"GPUassert: %s %s %d\\n\", cudaGetErrorString(code), file, line);")
        self.gen_add_code_line("if (abort){cudaDeviceReset(); exit(code);}")
        self.gen_add_end_control_flow()
        self.gen_add_end_control_flow() # end of function but don't want spacing
        self.gen_add_code_line("#define gpuErrchk(err) {gpuAssert(err, __FILE__, __LINE__);}")
        # gpuErrchkKernel catches BOTH (a) synchronous launch-time errors via
        # cudaPeekAtLastError — e.g. cudaErrorLaunchOutOfResources (code 701)
        # when the kernel asks for more registers than the SM can give — and
        # (b) asynchronous execution-time errors via cudaDeviceSynchronize.
        # Use this after every <<<>>> kernel launch. Plain cudaDeviceSynchronize
        # alone does NOT propagate launch-time errors: a launch can fail before
        # work is queued, leaving the stream empty, so sync returns success and
        # the next call clears the error. That's why the overnight bench was
        # silently reporting failed launches as ~2us "compute time."
        self.gen_add_code_line("#define gpuErrchkKernel() {gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());}")
        self.gen_add_code_line("")

        # also add printMat for debug if requested
        if self.gen_print_mat:
            self.gen_add_code_line("template <typename T, int M, int N>")
            self.gen_add_code_line("__host__ __device__")
            self.gen_add_code_line("void printMat(T *A, int lda){", True)
            self.gen_add_code_line("for(int i=0; i<M; i++){", True)
            self.gen_add_code_line("for(int j=0; j<N; j++){printf(\"%.4f \",A[i + lda*j]);}")
            self.gen_add_code_line("printf(\"\\n\");")
            self.gen_add_end_control_flow()
            self.gen_add_end_function()
            self.gen_add_code_line("template <typename T, int M, int N>")
            self.gen_add_code_line("__host__ __device__")
            self.gen_add_code_line("void printMat(const T *A, int lda){", True)
            self.gen_add_code_line("for(int i=0; i<M; i++){", True)
            self.gen_add_code_line("for(int j=0; j<N; j++){printf(\"%.4f \",A[i + lda*j]);}")
            self.gen_add_code_line("printf(\"\\n\");")
            self.gen_add_end_control_flow()
            self.gen_add_end_function()

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
    def gen_all_code(self, include_base_inertia = False, include_homogenous_transforms = False, fixed_target_name = "", output_path = None,
                     codegen_profile = "all", algorithm_list = None, enable_floating_second_order = True,
                     enable_idsva_so_world_frame = None, runtime_inertia = False, runtime_transform = False,
                     runtime_joint_dynamics = None, multi_target_batch = None, collision_spec = None,
                     contact_frames = None):
        # Default-pick the SO variant that wins per the 2026-05 perf sweep
        # (see test/benchmarks/benchmark_multi_version_sm120_5090_full.md
        # § IDSVA_SO_BODY_FRAME vs IDSVA_SO_WORLD_FRAME):
        #
        #   Robot          body-frame µs   world-frame µs   winner   margin
        #   iiwa14_fixed       26.5             805         body      30.3×
        #   go2_fixed          36.0            1338         body      37.1×
        #   g1_fixed         1301              5804         body       4.5×
        #   iiwa14_floating  2642              1652         world      1.6×
        #   go2_floating     3951              2830         world      1.4×
        #   g1_floating     28222              8451         world      3.3×
        #
        # body-frame multi-pass amortizes well for fixed-base; world-frame's
        # single-pass + no gravity shim wins floating-base. Callers can
        # override with enable_idsva_so_world_frame=True/False to force a
        # specific variant (the bench harness exercises both for comparison).
        if enable_idsva_so_world_frame is None:
            # Match the dispatcher/device frame predicate EXACTLY: world-frame for
            # floating-base, spherical, OR high-DOF fixed-base (NV >= threshold).
            # If this drifts from _idsva_so_use_world_frame, a fixed-base robot the
            # dispatcher routes to world (g1/h1_2/h2_plus) gets no world_frame inner
            # emitted -> undefined idsva_so_world_frame_inner at compile time.
            enable_idsva_so_world_frame = _idsva_so_use_world_frame(self)
        # idsva_so_world_frame is also a first-class algorithm_list key: an
        # explicit request enables the world-frame emit even on fixed-base
        # (where the kwarg default is False). Resolve the algorithm set up front
        # so the request can be OR'd into enable_idsva_so_world_frame below.
        algorithms = self._normalize_codegen_algorithms(codegen_profile, algorithm_list)
        # SPHERICAL (Tier-C) slices: inverse_dynamics + crba + minv + forward_dynamics
        # (dynamics) plus end_effector_pose + frame_jacobian (kinematics) are ported.
        # forward_dynamics routes through minv (= inv(CRBA(q))) + the compute_c RNEA,
        # not the standalone ABA (which keeps its scalar single-DoF articulated-body
        # recursion — a separate later slice). The EE-pose / frame-jacobian FK chain-up
        # consumes the joint's HOMOGENEOUS quaternion transform (R, not Rᵀ), built via
        # the shared quaternion XmatsHom substitution on the joint's own 4-wide q-block.
        # Reject any other requested algorithm for a robot with a spherical joint so we
        # fail loudly instead of emitting a wrong aba/gradient/SO kernel.
        if self.robot.robot_has_spherical():
            _SPHERICAL_OK = {"inverse_dynamics", "crba", "minv", "forward_dynamics",
                             "end_effector_pose", "frame_jacobian", "integrator",
                             "inverse_dynamics_gradient", "forward_dynamics_gradient",
                             "aba", "idsva_so_body_frame", "idsva_so_world_frame",
                             "fdsva_so"}
            _unported = sorted(a for a in algorithms if a not in _SPHERICAL_OK)
            if _unported:
                raise NotImplementedError(
                    "Spherical (ball) joint CUDA codegen currently supports "
                    "inverse_dynamics + crba + minv + forward_dynamics + "
                    f"end_effector_pose + frame_jacobian + integrator + "
                    f"inverse_dynamics_gradient + forward_dynamics_gradient + aba + "
                    f"idsva_so + fdsva_so; requested unsupported algorithm(s) "
                    f"{_unported}. Remaining algorithms are follow-on slices "
                    "(see docs/open-tasks/joint_types_plan.md)."
                )
        if "idsva_so_world_frame" in algorithms:
            enable_idsva_so_world_frame = True
        self.include_fixed_kinematic_targets = fixed_target_name != ""
        # GATO Ask-4: the single named kinematic target, resolved ONCE here so every
        # consumer agrees. The generic end_effector_pose* family evaluates the last
        # MOVING joint -- it DROPS the terminal fixed joint's <origin> (indy7 "EE":
        # 6cm z; iiwa14: 4cm z), so grid_plant's EE costs were tracking the wrong
        # frame. "" and "all" have no single canonical target -> fall back to the
        # generic family (same rule as the GRID_RBD_* macros below). Read by
        # gen_ee_target_aliases (the stable end_effector_pose_target_* symbols) and
        # by _plant.py's ee_pos_cost family, so neither hard-codes a joint name.
        self._ee_target_name = fixed_target_name if fixed_target_name not in ("", "all") else ""
        self._ee_target_sfx = ("_" + self._ee_target_name) if self._ee_target_name else ""
        # D.4 / Phase 5: runtime-mutable inertia table. When True, the per-link
        # spatial inertia is reconstructed on-device from a mutable d_inertia_params
        # table (set_inertia_params) instead of streamed from the baked d_XImats.
        # Default False keeps the BAKED path byte-identical (the field + device
        # branch are entirely flag-gated).
        self.runtime_inertia = runtime_inertia
        # runtime_transform: runtime-mutable joint-frame <origin> (xyz+rpy). When
        # True, each joint's constant Xfixed is rebuilt on-device once per launch
        # from a mutable d_transform_params table (set_transform_params) and the
        # general-rpy DENSE X pattern is baked so rpy can move freely. Default
        # False keeps the BAKED path byte-identical (field + branches flag-gated).
        self.runtime_transform = runtime_transform
        # runtime_joint_dynamics: runtime-mutable per-DOF damping/friction. When
        # True the id/fd/aba/*_gradient bias reads the coefficients from a mutable
        # device table instead of baking them as literals; the table is URDF-
        # initialized so the runtime path is BIT-IDENTICAL until set_joint_dynamics.
        # gen_all_code default None means "keep the ctor value" (so the ctor arg and
        # the gen_all_code arg compose); pass True/False to override per call.
        if runtime_joint_dynamics is not None:
            self.runtime_joint_dynamics = runtime_joint_dynamics
        self.generated_algorithms = algorithms
        # MIMIC GRADIENTS — fully supported, no refusal. All first/second-order mimic
        # gradients emit correctly for BOTH bases via the alpha-weighted reduced-v-slot
        # column fold: inverse_dynamics_gradient / forward_dynamics_gradient, ee_pose grad/hess, f_ext_gradient, idsva_so/fdsva_so
        # (the SO world inner runs a per-column INTERNAL-coordinate sweep then folds to
        # the reduced 4*NV^3 output; the floating root's 6 DoF emerge as 6 distinct
        # internal slots, alpha=1, so no separate root fold). The last gated case, the
        # floating-base + multi-stage RK + mimic integrator gradient, was RESOLVED
        # 2026-06-02 (B3): it composes the (correct, B1) floating-mimic FD gradient in
        # reduced tangent space — the mimic alpha-fold lives entirely inside the FD inner,
        # and the RK chain-rule + SE(3) dIntegrate projection act on already-reduced
        # columns — so no separate refusal exists. Verified vs the RBDReference oracle on
        # fr3-floating (Euler/SI-Euler/Midpoint/RK3/RK4, value + gradient + both-at-once);
        # the only large residuals are float32 cancellation through the ill-conditioned
        # reduced finger Minv (cond ~1.3e4) at extreme energetic+large-dt samples (the same
        # conditioning floor the fr3-floating FD-gradient tolerance documents), absorbed by
        # the equivalence test's norm-relative guard — not a code bug.
        self.generate_inverse_dynamics_gradient = "inverse_dynamics_gradient" in algorithms
        self.generate_forward_dynamics_gradient = "forward_dynamics_gradient" in algorithms
        self.generate_end_effector_pose_hessian = "end_effector_pose_hessian" in algorithms
        self.enable_floating_second_order = enable_floating_second_order
        allow_second_order = (not self.robot.floating_base) or enable_floating_second_order
        self.generate_idsva_so_body_frame = ("idsva_so_body_frame" in algorithms) and allow_second_order
        self.generate_fdsva_so = ("fdsva_so" in algorithms) and allow_second_order
        self.generate_idsva_so_world_frame = bool(enable_idsva_so_world_frame) and self.generate_idsva_so_body_frame
        # fdsva_so on floating-base's body calls idsva_so_world_frame_inner<T>(...)
        # unconditionally — without world_frame emission we'd produce a header
        # that fails at link time. Catch the misconfiguration early so the
        # error points at the cause, not a downstream nvcc undefined-symbol.
        if self.generate_fdsva_so and self.robot.floating_base and not self.generate_idsva_so_world_frame:
            raise ValueError(
                "Cannot emit fdsva_so on floating-base without idsva_so_world_frame. "
                "fdsva_so's floating-base body calls idsva_so_world_frame_inner. "
                "Either keep enable_idsva_so_world_frame at its default (None → "
                "True for floating-base) or remove fdsva_so from algorithms."
            )
        include_any_kinematics = any(name in algorithms for name in ("end_effector_pose", "end_effector_pose_gradient", "end_effector_pose_hessian"))
        include_homogenous_transforms = include_homogenous_transforms or include_any_kinematics
        # first generate the file info
        file_notes = [ "Interface is:", \
            "    __host__   robotModel<T> *d_robotModel = init_robotModel<T>()", \
            "    __host__   cudaStream_t streams = init_grid<T>()", \
            "    __host__   gridData<T> *hd_ata = init_gridData<T,NUM_TIMESTEPS>();"
            "    __host__   close_grid<T>(cudaStream_t *streams, robotModel<T> *d_robotModel, gridData<T> *hd_data)", \
            "",\
            "    __device__ inverse_dynamics_inner<T>(T *s_c,  T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity)",\
            "    __device__ inverse_dynamics_inner<T>(T *s_c,  T *s_vaf, const T *s_q, const T *s_qd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity)",\
            "    __device__ inverse_dynamics_device<T>(T *s_c, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity)", \
            "    __device__ inverse_dynamics_device<T>(T *s_c, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity)", \
            "    __global__ inverse_dynamics_kernel<T>(T *d_c, const T *d_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)", \
            "    __global__ inverse_dynamics_kernel<T>(T *d_c, const T *d_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)", \
            "    __host__   inverse_dynamics<T,USE_QDD_FLAG=false,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)", \
            "",\
            "    __device__ inverse_dynamics_inner_vaf<T>(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity)",\
            "    __device__ inverse_dynamics_inner_vaf<T>(T *s_vaf, const T *s_q, const T *s_qd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity)",\
            "    __device__ inverse_dynamics_vaf_device<T>(T *s_vaf, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity)", \
            "    __device__ inverse_dynamics_vaf_device<T>(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity)", \
            "",\
            "    __device__ minv_inner<T>(T *s_Minv, T *s_F, const T *s_q, T *s_XImats, int *s_topology_helpers, T *s_temp)",\
            "    __device__ minv_device<T>(T *s_Minv, const T *s_q, const robotModel<T> *d_robotModel)", \
            "    __global__ minv_Kernel<T>(T *d_Minv, unsigned char *d_workspace, const T *d_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)", \
            "    __host__   minv<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)", \
            "",\
            "    __device__ forward_dynamics_inner<T>(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, T *s_minv_F, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity)",\
            "    __device__ forward_dynamics_device<T, RESOURCE_TIER=TIER_SHARED>(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity, T *d_workspace = nullptr)", \
            "    __global__ forward_dynamics_kernel<T>(T *d_qdd, unsigned char *d_workspace, const T *d_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)", \
            "    __host__   forward_dynamics<T>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)", \
            "",\
            "    __device__ inverse_dynamics_gradient_inner<T>(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_vaf, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity)",\
            "    __device__ inverse_dynamics_gradient_device<T>(T *s_dc_du, const T *s_q, const T *s_qd, const T *robotModel<T> *d_robotModel, const T gravity)", \
            "    __device__ inverse_dynamics_gradient_device<T>(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity)", \
            "    __global__ inverse_dynamics_gradient_kernel<T>(T *d_dc_du, const T *d_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)", \
            "    __global__ inverse_dynamics_gradient_kernel<T>(T *d_dc_du, const T *d_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)", \
            "    __host__   inverse_dynamics_gradient<T,USE_QDD_FLAG=false,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)", \
            "",\
            "    __device__ forward_dynamics_gradient_device<T>(T *s_df_du, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity)",\
            "    __device__ forward_dynamics_gradient_device<T>(T *s_df_du, const T *s_q, const T *s_qd, const T *s_qdd, const T *s_Minv, const robotModel<T> *d_robotModel, const T gravity)", \
            "    __global__ forward_dynamics_gradient_kernel<T>(T *d_df_du, const T *d_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)", \
            "    __global__ forward_dynamics_gradient_kernel<T>(T *d_df_du, const T *d_q_qd, const T *d_qdd, const T *d_Minv, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)", \
            "    __host__   forward_dynamics_gradient<T,USE_QDD_MINV_FLAG=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)", \
            "",\
            "    __device__ end_effector_pose_inner<T,TEMP_IN_SMEM=true>(T *s_end_effector_pose, const T *s_q, const T *s_Xhom, int *s_topology_helpers, T *s_temp, T *d_workspace, unsigned char *s_linalg_smem)", \
            "    __device__ end_effector_pose_device<T>(T *s_end_effector_pose, const T *s_q, const robotModel<T> *d_robotModel)", \
            "    __global__ end_effector_pose_kernel<T>(T *d_end_effector_pose, const T *d_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)", \
            "    __host__   end_effector_pose<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)", \
            "",\
            "    __device__ end_effector_pose_gradient_inner<T>(T *s_end_effector_pose_gradient, const T *s_q, const T *s_Xhom, const T *s_dXhom, int *s_topology_helpers, T *s_temp)", \
            "    __device__ end_effector_pose_gradient_device<T>(T *s_end_effector_pose_gradient, const T *s_q, const robotModel<T> *d_robotModel)", \
            "    __global__ end_effector_pose_gradient_kernel<T>(T *d_end_effector_pose_gradient, unsigned char *d_workspace, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)", \
            "    __host__   end_effector_pose_gradient<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)", \
            "",\
            "    __device__ end_effector_pose_hessian_inner<T>(T *s_end_effector_pose_gradient, const T *s_q, const T *s_Xhom, const T *s_dXhom, int *s_topology_helpers, T *s_temp)", \
            "    __device__ end_effector_pose_hessian_device<T>(T *s_end_effector_pose_gradient, const T *s_q, const robotModel<T> *d_robotModel)", \
            "    __global__ end_effector_pose_hessian_kernel<T>(T *d_end_effector_pose_gradient, const T *d_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)", \
            "    __host__   end_effector_pose_hessian<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)", \
            "",\
            "    __device__ idsva_so_body_frame_inner(T *s_idsva_so, const T *s_q, const T *s_qd, T *s_qdd, T *s_XImats, T *s_mem, const T gravity)",\
            "    __global__ idsva_so_body_frame_kernel(T *d_idsva_so, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)", \
            "    __host__   idsva_so_body_frame<T>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)", \
            "",\
            "    __device__ fdsva_so_contract(T *s_df2, T *s_idsva_so, T *s_Minv, T *s_df_du, T *s_q, T *s_qd, const T *s_qdd, const T *s_tau, T *s_XImats, T *s_temp, const T gravity)",\
            "    __device__ fdsva_so_device(T *s_df2, T *s_df_du, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity)", \
            "    __global__ fdsva_so_kernel(T *d_df2, const T *d_q_qd_qdd_tau, const int stride_q_qd_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)", \
            "    __host__   fdsva_so<T>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)", \
            "",\
            "","Suggested Type T is float",\
            "","Additional helper functions and ALGORITHM_inner functions which take in __shared__ memory temp variables exist -- see function descriptions in the file",\
            "","By default device and kernels need to be launched with dynamic shared mem of size <FUNC_CODE>_DYNAMIC_SHARED_MEM_COUNT where <FUNC_CODE> = [INVERSE_DYNAMICS, MINV, FORWARD_DYNAMICS, INVERSE_DYNAMICS_GRADIENT, FORWARD_DYNAMICS_GRADIENT]"]
        file_notes += ["", "Codegen profile: " + str(codegen_profile), "Generated algorithms: " + ", ".join(sorted(algorithms))]
        if self.include_fixed_kinematic_targets:
            file_notes += ["", "Additional EEPose Functions Included for Fixed Kinematic Target: " + fixed_target_name,""]
        self.gen_add_func_doc("This instance of grid.cuh is optimized for the urdf: " + self.robot.name,file_notes)
        # then all of the includes (and namespaces and defines)
        self.gen_add_includes()
        # then add the gpu error macro
        self.gen_add_gpu_err()
        # File-scope preprocessor mirrors of the second-order codegen gates.
        # We also emit `const int GRID_GENERATES_*_SO` inside `namespace grid`
        # below (for C++ runtime / test consumers), but `#if X` requires a
        # preprocessor macro — a namespaced const reads as the undefined-symbol
        # zero in `#if`, silently turning off any consumer that gates on it
        # (notably the bench's timeGRiD_{single,batch}.cu measure_* blocks).
        # Different names from the namespaced const avoid macro/const collision.
        self.gen_add_code_line(
            "#define GRID_HAS_IDSVA_SO_BODY_FRAME " + str(int(getattr(self, "generate_idsva_so_body_frame", True)))
        )
        self.gen_add_code_line(
            "#define GRID_HAS_FDSVA_SO " + str(int(getattr(self, "generate_fdsva_so", True)))
        )
        self.gen_add_code_line(
            "#define GRID_HAS_IDSVA_SO_WORLD_FRAME " + str(int(getattr(self, "generate_idsva_so_world_frame", False)))
        )
        # GRID_HAS_IDSVA_SO gates the dispatching `grid::idsva_so` host wrapper.
        # Emitted whenever the chosen variant for this robot's base type is
        # available: body_frame for fixed-base (always), world_frame for
        # floating-base (opt-in via enable_idsva_so_world_frame).
        has_idsva_so = getattr(self, "generate_idsva_so_body_frame", True) and (
            (not self.robot.floating_base) or getattr(self, "generate_idsva_so_world_frame", False)
        )
        self.gen_add_code_line("#define GRID_HAS_IDSVA_SO " + str(int(has_idsva_so)))
        # Integrator availability gates. Both value and gradient kernels are
        # emitted whenever requested, for fixed- and floating-base. (Floating
        # gradient is currently Euler-only and inherits the upstream
        # forward_dynamics_gradient dqdd/dqd bug; see _normalize_codegen_algorithms.)
        # Consumers must #if-guard gradient calls so a build that omits
        # integrator_gradient still compiles.
        self.gen_add_code_line(
            "#define GRID_HAS_INTEGRATOR " + str(int("integrator" in algorithms)))
        self.gen_add_code_line(
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
        _core_has = {
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
        for macro_suffix, algo_key in _core_has.items():
            self.gen_add_code_line(
                "#define GRID_HAS_" + macro_suffix + " "
                + str(int(algo_key in self.generated_algorithms)))
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
        if self.robot.floating_base and not (
                self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis()):
            self.gen_add_code_line("#define GRID_RBD_WITH_MUJOCO 1")
        self.gen_add_code_line("")
        # then open our namespace
        self.gen_add_func_doc("All functions are kept in this namespace")
        self.gen_add_code_line("namespace " + self.file_namespace + " {", True)
        self.gen_add_shared_memory_helpers()
        # W1b.3 (Inc4b): multi_target PRESENCE marker emitted here — BEFORE the gridData
        # struct (in gen_add_constants_helpers just below) — so the struct's
        # #if GRID_HAS_MULTI_TARGET_POSITION-guarded d_/h_multi_target_position fields
        # resolve. Non-MT robots never see the #define -> #if reads 0 -> byte-identical
        # struct. NUM_MULTI_TARGETS (the malloc size) is emitted later, just before
        # gen_init_gridData. Collision is exclusive (each owns the single batch).
        self._has_multi_target_position = multi_target_batch is not None
        if multi_target_batch is not None:
            assert collision_spec is None, "multi_target_batch and collision_spec are exclusive (each defines the single multi_target batch / NUM_MULTI_TARGETS)"
            self._mt_batch = self.build_target_batch(multi_target_batch)
            self.gen_add_code_line("#define GRID_HAS_MULTI_TARGET_POSITION 1")
        # then generate any constants and other helpers
        self.gen_add_constants_helpers(include_base_inertia, include_homogenous_transforms)
        # then the linear algebra related helpers
        # Emit GLASS before spatial algebra because dot_prod is a compatibility
        # shim over glass::dot_strided.
        self.gen_grid_linalg_backend_helpers()
        # then the spatial algebra related helpers
        self.gen_spatial_algebra_helpers()
        self.gen_crm()
        self.gen_crm_mul()
        # Tier-B (skew/general axis) helper: additive, emitted ONLY when the
        # model carries a non-cardinal motion column. All-cardinal robots get
        # byte-identical headers (no extra device function).
        if self.robot.robot_has_skew_axis():
            self.gen_mxS_general()
        self.gen_invert_matrix()
        self.gen_matmul()
        self.gen_matmul_trans() 
        self.gen_outer_product()
        # then generate the robot specific transformation and inertia matricies
        self.gen_init_topology_helpers()
        self.gen_init_XImats(include_base_inertia, include_homogenous_transforms)
        # D.4 / Phase 5: flag-gated mutable inertia table init + mutator. Emitted
        # only under runtime_inertia. The device XImats helper rebuilds the
        # I-region from this table; both use include_base_inertia=False to mirror
        # the I-region layout the helper streams (Imats[1:], bodies 1..N).
        if getattr(self, "runtime_inertia", False):
            self.gen_init_inertia_params()
            self.gen_set_inertia_params()
        # runtime_transform: flag-gated mutable joint-origin table init + mutator.
        # The device XImats helper rebuilds each joint's Xfixed scratch from this
        # table once per launch and hoists it out of the hot sin/cos(q) loop.
        if getattr(self, "runtime_transform", False):
            self.gen_init_transform_params()
            self.gen_set_transform_params()
        # runtime_joint_dynamics: flag-gated mutable damping/friction table init +
        # mutator. The id/fd/aba/*_gradient bias reads this table (URDF-initialized,
        # alpha-folded per v-slot) instead of the baked literals when the flag is set.
        if getattr(self, "runtime_joint_dynamics", False):
            self.gen_init_joint_dynamics_params()
            self.gen_set_joint_dynamics_params()
        self.gen_init_robotModel()
        self.gen_free_robotModel()
        # W1b.3 (Inc4b): emit the multi_target CONSTANTS *before* gen_init_gridData so its
        # #if GRID_HAS_MULTI_TARGET_POSITION-guarded d_/h_multi_target_position mallocs can
        # size on NUM_MULTI_TARGETS. The batch's functions (inner/device/kernel/host) are
        # emitted later in the kinematics dispatch (they need the world-FK machinery); only
        # the count + presence marker are hoisted here. Build the batch ONCE and cache it on
        # self so the dispatch reuses the exact same descriptor. Collision is exclusive.
        # NUM_MULTI_TARGETS (the malloc size) emitted here — before gen_init_gridData —
        # so the #if GRID_HAS_MULTI_TARGET_POSITION-guarded d_/h_multi_target_position
        # mallocs can size on it. The presence #define + cached batch were set right after
        # the namespace opened (above the gridData struct); reuse the cached batch.
        if multi_target_batch is not None:
            self.gen_add_code_line("const int NUM_MULTI_TARGETS = " + str(self._mt_batch["n"]) + ";")
        self.gen_init_gridData()
        self.gen_joint_limits_size()
        self.gen_init_joint_limits()
        self.gen_load_update_XImats_helpers()
        # Standalone topology-helper filler for external/inline-CUDA callers of the
        # *_inner functions (uniform interface; no-op for serial chains).
        self.gen_load_topology_helpers()
        if include_homogenous_transforms and include_any_kinematics:
            self.gen_load_update_XmatsHom_helpers(include_base_inertia)
            if "end_effector_pose_gradient" in algorithms or "end_effector_pose_hessian" in algorithms:
                self.gen_load_update_XmatsHom_helpers(include_base_inertia,include_gradients = True)
            if "end_effector_pose_hessian" in algorithms:
                self.gen_load_update_XmatsHom_helpers(include_base_inertia,include_gradients = True, include_hessians = True)
        # then generate kinematic algorithms.
        # SE(3) Lie-group helpers (grid_integrate_floating_q, grid_so3_*, grid_quat_*)
        # are needed by the FD-on-Jacobian d2ee inner on floating base. Emit them
        # here too so callers that skip the integrator codegen still get them; track
        # the emission so gen_integrator skips its own emit (avoiding redefinitions).
        self._lie_helpers_emitted = False
        if include_any_kinematics:
            if self.robot.floating_base and "end_effector_pose_hessian" in algorithms:
                self.gen_lie_group_helpers()
                self._lie_helpers_emitted = True
            self.gen_eepose_and_derivatives(fixed_target_name = fixed_target_name,
                                            include_pose = "end_effector_pose" in algorithms,
                                            include_gradient = "end_effector_pose_gradient" in algorithms,
                                            include_hessian = "end_effector_pose_hessian" in algorithms)
            # GATO Ask-4: stable end_effector_pose_target_* aliases (forward to the named
            # target's family when one is baked, else to the generic one). Emitted right
            # after the family they alias, so both symbols are in scope.
            self.gen_ee_target_aliases(include_pose = "end_effector_pose" in algorithms,
                                       include_gradient = "end_effector_pose_gradient" in algorithms,
                                       include_hessian = "end_effector_pose_hessian" in algorithms)
            # C4: grid_rbd EE binding entry-point aliases. The C-ABI wrappers
            # (grid_rbd_end_effector_pose[_gradient/_hessian]) and the A1 kernel-threads
            # introspection call THROUGH these macros rather than the bare unsuffixed
            # symbols. When a SINGLE named fixed target is baked (grid_rbd
            # ee_joint_names=[name]), they route to the _<name> launchers anchored at
            # that flange (matching pinocchio) and report a 1-EE output; "" / "all" keep
            # the all-leaf default (the unsuffixed symbols, NUM_EES leaves). gen_all_code
            # always emits exactly one of these blocks so the fixed wrapper compiles.
            _ee_named = fixed_target_name not in ("", "all")
            _ee_sfx = ("_" + fixed_target_name) if _ee_named else ""
            self.gen_add_code_lines([
                "// ---- grid_rbd EE binding entry-point aliases (single named target routes here) ----",
                "#define GRID_RBD_NUM_EES " + ("1" if _ee_named else "grid::NUM_EES"),
                "#define GRID_RBD_EE_POSE_FN end_effector_pose" + _ee_sfx,
                "#define GRID_RBD_EE_POSE_GRADIENT_FN end_effector_pose_gradient" + _ee_sfx,
                "#define GRID_RBD_EE_POSE_HESSIAN_FN end_effector_pose_hessian" + _ee_sfx,
                "#define GRID_RBD_EE_POSE_KERNEL end_effector_pose_kernel" + _ee_sfx,
                "#define GRID_RBD_EE_POSE_GRADIENT_KERNEL end_effector_pose_gradient_kernel" + _ee_sfx,
                "#define GRID_RBD_EE_POSE_HESSIAN_KERNEL end_effector_pose_hessian_kernel" + _ee_sfx,
                ""])
            # W1b: batched multi-target world positions. Opt-in via multi_target_batch
            # (a list of {anchor_jid, offset[, group]}); default None -> NOT emitted so
            # every existing robot's grid.cuh is byte-identical. Reuses the shared FK
            # (emit_world_fk_chainup) + XmatsHom machinery set up above.
            if multi_target_batch is not None:
                # NUM_MULTI_TARGETS + GRID_HAS_MULTI_TARGET_POSITION were emitted EARLY
                # (before gen_init_gridData); reuse the cached batch and skip the duplicate
                # const here. gen_multi_target_position_bench emits the launchable kernel +
                # 3-mode host for BOTH position and gradient (public batch only).
                _mt_batch = self._mt_batch
                self.gen_multi_target_position(_mt_batch, emit_num_const=False)
                self.gen_multi_target_position_gradient(_mt_batch)
                self.gen_multi_target_position_bench(_mt_batch)
            # C.2 (GATO ask 1): contact-frame wrench -> joint-local f_ext + d/dq. Opt-in: default None
            # emits nothing, so every existing header is byte-identical. Reuses the SAME world-FK
            # chain-up as multi_target (emit_world_fk_chainup) — no second copy.
            if contact_frames:
                self.gen_f_ext_contact(contact_frames)
            # W3: collision. Each sphere-density tier IS a multi_target batch — build it in the
            # tier's own order (NO group re-sort) so the baked radii/self_cc_ranges stay
            # index-aligned. Emit a POSITION extractor per tier (config_free's broad-phase needs
            # the coarse positions); the expensive GRADIENT only for the FINEST tier (the sole
            # consumer is the differentiable cost, which binds the finest/public batch). The
            # grid_collision namespace itself is emitted AFTER grid closes (below), like grid_plant.
            if collision_spec is not None:
                from .algorithms._collision import normalize_collision_tiers
                _cc_tiers = normalize_collision_tiers(collision_spec)
                for _t in _cc_tiers:
                    _cc_batch = {"n": _t["n"], "anchor": _t["anchor"],
                                 "offset": _t["offset"], "groups": {"all": (0, _t["n"])}}
                    self.gen_multi_target_position(_cc_batch, suffix=_t["suffix"])
                    if _t["suffix"] == "":  # finest / public tier -> the differentiable path
                        self.gen_multi_target_position_gradient(_cc_batch, suffix="")
        if self.robot.floating_base and not enable_floating_second_order:
            print('floating-base second order dynamics are still under development')
        # then generate the dynamics algorithms
        if "inverse_dynamics" in algorithms:
            self.gen_inverse_dynamics()
        # E1: joint-torque regressor Y (tau = Y . pi). Additive; reuses the RNEA
        # forward sweep emitted by gen_inverse_dynamics (requires "inverse_dynamics").
        if "inverse_dynamics_regressor" in algorithms:
            self.gen_inverse_dynamics_regressor()
        # PS5 energy regressors. KE reuses the RNEA forward sweep (requires
        # "inverse_dynamics"); PE reuses the ee_pose world-transform machinery
        # (requires "end_effector_pose", emitted above in the kinematics block).
        if "kinetic_energy_regressor" in algorithms:
            self.gen_kinetic_energy_regressor()
        if "potential_energy_regressor" in algorithms:
            self.gen_potential_energy_regressor()
        if "minv" in algorithms:
            self.gen_minv()
        if "forward_dynamics" in algorithms:
            self.gen_forward_dynamics()
        # FD parameter gradient dqdd/dpi = -Minv . Y. Additive; composes the
        # regressor (Y), minv (Minv) and inverse_dynamics/forward_dynamics
        # inners, so it requires "inverse_dynamics", "minv", "forward_dynamics" and "inverse_dynamics_regressor" co-emitted.
        if "forward_dynamics_parameter_gradient" in algorithms:
            self.gen_forward_dynamics_parameter_gradient()
        if "inverse_dynamics_gradient" in algorithms:
            self.gen_inverse_dynamics_gradient()
        if "forward_dynamics_gradient" in algorithms:
            self.gen_forward_dynamics_gradient()
        if "f_ext_gradient" in algorithms:
            self.gen_f_ext_gradient()
        if "aba" in algorithms:
            self.gen_aba()
        if "crba" in algorithms:
            self.gen_crba()
        if "integrator" in algorithms:
            self.gen_integrator()
        if ("integrator_gradient" in algorithms) or ("integrator_with_gradient" in algorithms):
            self.gen_integrator_gradient()
        if not self.robot.floating_base or enable_floating_second_order:
            if "idsva_so_body_frame" in algorithms:
                self.gen_idsva_so_body_frame()
                # Optional: emit the world-frame single-pass alternative path alongside
                # the existing emission. Co-exists with `idsva_so_body_frame_kernel`/`idsva_so_body_frame` (host);
                # the new entry point is `idsva_so_world_frame_kernel`/`idsva_so_world_frame` (host).
                if enable_idsva_so_world_frame:
                    self.gen_idsva_so_world_frame()
                # Emit the dispatching `idsva_so` host wrapper. It forwards to
                # world_frame for floating-base / spherical / high-DOF fixed-base
                # (NV >= threshold) and to body_frame otherwise; the world-frame
                # cases require world_frame to have been enabled above (now kept in
                # lockstep via _idsva_so_use_world_frame).
                if (not self.robot.floating_base) or enable_idsva_so_world_frame:
                    self.gen_idsva_so_dispatcher()
            if "fdsva_so" in algorithms:
                self.gen_fdsva_so()
                # F1: integrator_hessian device (the plant_step_hessian s_d2AB
                # surface) composes fdsva_so_device, so emit it alongside fdsva_so.
                # Fixed-base (dt-scaled assembly) and floating-base (the SE(3) retract
                # Hessian) both emit; only multi-stage RK static_asserts out. Gated on
                # fdsva_so membership to keep non-fdsva_so headers byte-identical.
                self.gen_integrator_hessian_device()
        # G2 centroidal quick-wins (R1-R3): additive families gated on their
        # grid:: deps. generalized_gravity / nonlinear_effects are RNEA bias
        # wrappers (need `id`); com / ccrba / energy live in the kinematics
        # (homogeneous-transform) domain and reuse the world-transform machinery
        # (need `ee_pose`). All are NEW emitters appended after the existing
        # algorithms, so existing emission is byte-identical.
        self.gen_centroidal_quickwins(algorithms)
        # E2 (additive, opt-in): general-frame geometric Jacobian. Only emitted
        # when the `frame_jacobian` key is explicitly selected, so every existing
        # profile's header is byte-identical. Needs ee_pose's world-transform
        # machinery (pulled in by _normalize_codegen_algorithms).
        # FLAG (shared GCG.py edit, additive — minimal gate relax): mimic robots
        # are now supported for the frame_jacobian family. The geometric Jacobian J
        # gained the alpha-weighted shared-v-slot fold (mirrors the ee_pose_gradient
        # Step 3b mimic accumulate); J-dot reuses frame_jacobian_inner at perturbed
        # q (mimic q-fold baked into s_XmatsHom by the XmatsHom helper) and osc_inertia
        # routes mimic Minv through crba_inner (the `minv`->`crba` dep above). The
        # ee_pose dependency itself is mimic-gated elsewhere (kin_ok), but the
        # frame_jacobian family only needs ee_pose's world-transform machinery, which
        # is emitted whenever the frame_jacobian key is selected.
        if "frame_jacobian" in algorithms and "end_effector_pose" in algorithms:
            NJ_fj = self.robot.get_num_joints()
            nv_fj = self.robot.get_num_vel()
            n_pos_fj = self.robot.get_num_pos()
            Xhom_size_fj, _, _ = self.gen_get_Xhom_size()
            # Arena sized for the launchable KERNEL (the widest consumer): it adds
            # s_q (NUM_POS) + s_frame_jacobian (6*NV) to the device arena
            # (s_XmatsHom(Xhom_size) + inner_temp(16*NJ)). The *_device wrapper takes
            # s_J as a caller param so it needs less, but over-allocating the shared
            # arena for it is harmless; mirrors ee_t_count = n + 6*ees + inner + XHom.
            fj_t_count = Xhom_size_fj + (16 * NJ_fj) + n_pos_fj + (6 * nv_fj)
            self.gen_add_code_line(
                "template <typename T> __host__ __device__ inline size_t FRAME_JACOBIAN_DYNAMIC_SHARED_MEM_BYTES() "
                "{ return grid_shared_arena_bytes<T>(" + str(fj_t_count) +
                ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }")
            # S1: surface-presence marker so host runners/bindings can detect the
            # launchable frame_jacobian host surface (kernel + 3-mode host).
            self.gen_add_code_line("#define GRID_HAS_FRAME_JACOBIAN 1")
            self.gen_frame_jacobian()
            # E2 CUDA parity (opt-in siblings). Jdot/Lambda reuse frame_jacobian_inner.
            if "frame_jacobian_dot" in algorithms:
                # Jdot arena = s_XmatsHom + extras(s_qpert[n_pos] + s_Jp[6nv] + s_Jm[6nv])
                # + inner_temp(16*NJ). The launchable kernel keeps its input (s_q_qd)
                # and output (s_frame_jacobian_dot) in STATIC __shared__ — NOT this
                # dynamic arena, which the frame_jacobian_dot_device wrapper owns
                # entirely — so this size matches the wrapper exactly.
                fjd_t_count = Xhom_size_fj + n_pos_fj + (2 * 6 * nv_fj) + (16 * NJ_fj)
                self.gen_add_code_line(
                    "template <typename T> __host__ __device__ inline size_t FRAME_JACOBIAN_DOT_DYNAMIC_SHARED_MEM_BYTES() "
                    "{ return grid_shared_arena_bytes<T>(" + str(fjd_t_count) +
                    ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }")
                self.gen_add_code_line("#define GRID_HAS_FRAME_JACOBIAN_DOT 1")
                # Floating-base Jdot integrates q on the SE(3) group; emit the Lie
                # helpers if no other kinematics path already did.
                if self.robot.floating_base and not getattr(self, "_lie_helpers_emitted", False):
                    self.gen_lie_group_helpers()
                    self._lie_helpers_emitted = True
                self.gen_frame_jacobian_dot()
            # Lambda (osc_inertia) is emitted for mimic robots too. It composes
            # Minv on device via minv_inner, whose mimic path routes
            # through crba_inner -> invert_matrix (== RBDReference.minv's mimic
            # fast path inv(CRBA(q))). That compose is correct in this arena: the
            # fr3-fixed CUDA crba/minv equivalence tests already prove it, and the
            # fr3 Lambda matches the numpy oracle to float32 across all 3 reference
            # frames. (The earlier all-zero Lambda was a SMOKE-RUNNER artifact: the
            # heavy osc kernel overflowed the register budget at 512 threads and
            # silently failed; the runner now clamps to the kernel's
            # maxThreadsPerBlock and checks the launch.)  (FLAGGED: un-gated mimic.)
            if "osc_inertia" in algorithms:
                # Lambda is SELF-CONTAINED: it composes Minv on device via
                # minv_inner, so the arena carries BOTH transform families
                # (spatial s_XImats for minv + homogeneous s_XmatsHom for J) plus
                # the minv buffers (s_Minv + the spilled F-region passed as
                # d_workspace) and the J*Minv*J^T compose scratch.
                # arena = s_XImats(XI) + extras + s_temp(max(no_F, 16*NJ)) where
                # extras = s_XmatsHom + s_Minv + s_F + s_Jfj + s_MJt + s_task + s_taskinv.
                # 2-rung spill ladder (full | spill-F): the per-tier t-counts +
                # spill_tier picks were computed in initialize_constants_helpers. The
                # OSC_INERTIA_F_IN_SMEM<TIER> predicate (true at the rungs whose pick
                # keeps s_F in smem) is the single source of truth the device reads.
                _osc_per = self.osc_inertia_t_count_per_tier
                _osc_F_in_smem = tuple(p == 0 for p in self.osc_inertia_spill_tier_3way)
                self.gen_add_code_line(
                    "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t OSC_INERTIA_DYNAMIC_SHARED_MEM_BYTES() { "
                    "if constexpr (TIER == TIER_SHARED)    return grid_shared_arena_bytes<T>(" + str(_osc_per[0]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                    "else if constexpr (TIER == TIER_LITE) return grid_shared_arena_bytes<T>(" + str(_osc_per[1]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
                    "else                                 return grid_shared_arena_bytes<T>(" + str(_osc_per[2]) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }")
                self.gen_add_code_line(
                    "template <int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr bool OSC_INERTIA_F_IN_SMEM() { return (TIER == TIER_SHARED) ? "
                    + ("true" if _osc_F_in_smem[0] else "false") + " : (TIER == TIER_LITE) ? "
                    + ("true" if _osc_F_in_smem[1] else "false") + " : "
                    + ("true" if _osc_F_in_smem[2] else "false") + "; }")
                self.gen_add_code_line("#define GRID_HAS_OSC_INERTIA 1")
                self.gen_osc_inertia()
        # Mimic-only marker: signal to consumers (e.g. the frame_jacobian smoke
        # runner) that this is a mimic header where osc_inertia (Lambda) was NOT
        # emitted -- which now only happens when osc_inertia was not selected at
        # all (mimic Lambda IS emitted otherwise). Emitted ONLY for mimic robots
        # so non-mimic headers stay byte-identical; the runner gates its Lambda
        # machinery on #ifndef GRID_FRAME_JAC_MIMIC.
        if ("frame_jacobian" in algorithms and "end_effector_pose" in algorithms
                and self.robot_has_mimic_joints() and "osc_inertia" not in algorithms):
            self.gen_add_code_line("#define GRID_FRAME_JAC_MIMIC 1")
        # Runtime-target pose / pose-gradient (additive, opt-in). Emitted only when
        # their key is selected, so every existing profile's header is byte-identical.
        # Both need ee_pose's world-transform machinery (pulled in above). The arena
        # mirrors frame_jacobian: s_XmatsHom(Xhom) + inner_temp(16*NJ) + s_q(NUM_POS)
        # + the output band (6 for pose, 6*NV for gradient). The runtime offset[3]
        # lives in STATIC __shared__ inside the kernel, not this dynamic arena.
        if (("end_effector_pose_runtime" in algorithms
                or "end_effector_pose_gradient_runtime" in algorithms)
                and "end_effector_pose" in algorithms):
            NJ_rt = self.robot.get_num_joints()
            nv_rt = self.robot.get_num_vel()
            n_pos_rt = self.robot.get_num_pos()
            Xhom_size_rt, _, _ = self.gen_get_Xhom_size()
            if "end_effector_pose_runtime" in algorithms:
                eprt_t_count = Xhom_size_rt + (16 * NJ_rt) + n_pos_rt + 6
                self.gen_add_code_line(
                    "template <typename T> __host__ __device__ inline size_t END_EFFECTOR_POSE_RUNTIME_DYNAMIC_SHARED_MEM_BYTES() "
                    "{ return grid_shared_arena_bytes<T>(" + str(eprt_t_count) +
                    ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }")
                self.gen_add_code_line("#define GRID_HAS_END_EFFECTOR_POSE_RUNTIME 1")
                self.gen_end_effector_pose_runtime()
            if "end_effector_pose_gradient_runtime" in algorithms:
                epgrt_t_count = Xhom_size_rt + (16 * NJ_rt) + n_pos_rt + (6 * nv_rt)
                self.gen_add_code_line(
                    "template <typename T> __host__ __device__ inline size_t END_EFFECTOR_POSE_GRADIENT_RUNTIME_DYNAMIC_SHARED_MEM_BYTES() "
                    "{ return grid_shared_arena_bytes<T>(" + str(epgrt_t_count) +
                    ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }")
                self.gen_add_code_line("#define GRID_HAS_END_EFFECTOR_POSE_GRADIENT_RUNTIME 1")
                self.gen_end_effector_pose_gradient_runtime()
        self.gen_combination_functions(algorithms, fixed_target_name)
        # then finally the master init and close the namespace
        self.gen_init_close_grid()
        self.gen_add_end_control_flow()
        # T6: emit the sibling `grid_plant` namespace (cost/constraint/plant-step
        # primitives composed over the grid:: surface). Additive: this runs AFTER
        # the grid namespace closes and makes ZERO edits to any grid:: emit path.
        self.gen_grid_plant(algorithms)
        # W3: sibling `grid_collision` namespace (baked sphere radii + self_cc_ranges + config_free
        # over grid::multi_target_position + the static SDF header). Emitted like grid_plant, after
        # the grid namespace closes; gated on collision_spec (the sphere batch was emitted above).
        if collision_spec is not None:
            from .algorithms._collision import normalize_collision_tiers
            self.gen_collision_namespace(normalize_collision_tiers(collision_spec))
        # then output to a file
        if output_path is None:
            output_path = self.file_namespace + ".cuh"
        file = open(output_path, "w")
        file.write(self.code_str)
        file.close()
