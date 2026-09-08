import os
import warnings

# Codegen-time frame-selection predicate shared with the idsva_so dispatcher /
# device wrapper. The emission gate (enable_idsva_so_world_frame) MUST agree with
# it so the world-frame inner is emitted exactly when a wrapper forwards there
# (else high-DOF fixed-base robots route to an undefined idsva_so_world_frame_inner).
from .algorithms._idsva_so import _idsva_so_use_world_frame

from .helpers._host_clamp import _apply_host_thread_clamp_pass

# Launch-config bake moved to _launch_config.py (H4); re-exported here because
# bindings/_compile/_handle/autotune_ffi and the parity goldens import these
# names from grid_codegen.GRiDCodeGenerator.
from ._launch_config import (LAUNCH_CONFIG_DEFAULT_GPU, LAUNCH_CONFIG_TIER_SYMBOL,
                             _ALGO_TO_SYMBOL, _launch_configs_dir, load_launch_config)



class GRiDCodeGenerator:
    # first import helpers to write code generation, spatial algebra, and opology helpers (parent, child, Sind, XImats) and the robotModel object wrapepr
    from .helpers import gen_add_code_line, gen_add_code_lines, gen_bake_const_array, gen_add_end_control_flow, gen_add_end_function, \
                         gen_add_func_doc, gen_add_serial_ops, gen_add_parallel_loop, gen_minv_apply, gen_add_sync, gen_var_in_list, \
                         gen_var_not_in_list, gen_add_multi_threaded_select, gen_kernel_load_inputs, gen_kernel_save_result, \
                         gen_anti_licm_input_reload, gen_anti_licm_output_write, \
                         gen_static_array_ind_3d, \
                         gen_mx_func_call_for_cpp, gen_add_shared_memory_helpers, gen_add_workspace_slot_count, gen_add_workspace_clamped_launch, gen_declare_shared_arena, _resolve_arena_layout, gen_arena_carve_struct, \
                         gen_device_wrapper, gen_tier_dispatch, gen_spatial_algebra_helpers, \
                         gen_get_XI_size, gen_init_XImats, gen_get_inertia_params_size, gen_init_inertia_params, gen_set_inertia_params, \
                         gen_get_transform_params_size, gen_init_transform_params, gen_set_transform_params, \
                         _joint_dynamics_folded_by_vslot, gen_get_joint_dynamics_params_size, \
                         gen_init_joint_dynamics_params, gen_set_joint_dynamics_params, \
                         gen_load_update_XImats_helpers_temp_mem_size, gen_load_update_XImats_helpers_function_call, \
                         gen_XImats_helpers_temp_shared_memory_code, gen_load_update_XImats_helpers, gen_topology_helpers_size, \
                         gen_get_Xhom_size, gen_load_update_XmatsHom_helpers, gen_load_update_XmatsHom_helpers_function_call, gen_XmatsHom_helpers_temp_shared_memory_code, gen_load_topology_helpers, \
                         gen_topology_sparsity_helpers_python, gen_init_topology_helpers, gen_topology_helpers_pointers_for_cpp, \
                         gen_topology_S_sign_for_cpp, gen_insert_helpers_function_call, gen_insert_helpers_func_def_params, gen_init_robotModel, gen_free_robotModel, gen_joint_limits_size, gen_init_joint_limits, \
                         gen_grid_linalg_backend_helpers, gen_linalg_smem_setup, gen_invert_matrix, gen_matmul, gen_matmul_trans, gen_crm_mul, gen_crm, gen_mxS_general, custom_is_constant, \
                         gen_mjx_input_convert, gen_mjx_quat_reorder, gen_mjx_base_rotate, gen_mjx_symmetrize_full, gen_mjx_accel_out, gen_mjx_congruence, gen_mjx_column_reframe, gen_mjx_retract, \
                         robot_has_mimic_joints, _v_slot_cpp, _alpha_for_jid, _id_S_desc

    # then import all of the algorithms
    from .algorithms import gen_inverse_dynamics_inner_temp_mem_size, gen_inverse_dynamics_inner_function_call, \
                            gen_inverse_dynamics_inner, gen_inverse_dynamics_device, \
                            gen_inverse_dynamics_joint_dynamics_bias, \
                            gen_inverse_dynamics_kernel, gen_inverse_dynamics_host, gen_inverse_dynamics, \
                            gen_inverse_dynamics_regressor_inner_temp_mem_size, gen_inverse_dynamics_regressor_inner_function_call, \
                            gen_inverse_dynamics_regressor_inner, \
                            gen_inverse_dynamics_regressor_device, gen_inverse_dynamics_regressor_kernel, \
                            gen_inverse_dynamics_regressor_host, gen_inverse_dynamics_regressor, \
                            gen_kinetic_energy_regressor_inner_temp_mem_size, gen_kinetic_energy_regressor_inner_function_call, \
                            gen_kinetic_energy_regressor_inner, gen_kinetic_energy_regressor_device, gen_kinetic_energy_regressor_kernel, \
                            gen_kinetic_energy_regressor_host, gen_kinetic_energy_regressor, \
                            gen_potential_energy_regressor_inner_function_call, gen_potential_energy_regressor_inner, \
                            gen_potential_energy_regressor_device, gen_potential_energy_regressor_kernel, \
                            gen_potential_energy_regressor_host, gen_potential_energy_regressor, \
                            gen_forward_dynamics_parameter_gradient_inner_temp_mem_size, gen_forward_dynamics_parameter_gradient_inner_function_call, \
                            gen_forward_dynamics_parameter_gradient_inner, \
                            gen_forward_dynamics_parameter_gradient_device, gen_forward_dynamics_parameter_gradient_kernel, \
                            gen_forward_dynamics_parameter_gradient_host, gen_forward_dynamics_parameter_gradient, \
                            gen_minv_inner_temp_mem_size, gen_minv_inner_F_size, gen_minv_inner_no_F_size, gen_minv_inner_function_call, gen_minv_inner, \
                            gen_minv_device, gen_minv_kernel, gen_minv_host, gen_minv, \
                            gen_forward_dynamics_inner_temp_mem_size, gen_forward_dynamics_finish_function_call, gen_forward_dynamics_finish, \
                            gen_forward_dynamics_inner_function_call, gen_forward_dynamics_inner, gen_forward_dynamics_device, \
                            gen_forward_dynamics_kernel, gen_forward_dynamics_host, gen_forward_dynamics, \
                            gen_inverse_dynamics_gradient_inner_temp_mem_size, gen_inverse_dynamics_gradient_temp_layout, _emit_fb_bfs_level_indexing, \
                            \
                            gen_inverse_dynamics_gradient_inner_function_call, gen_inverse_dynamics_gradient_inner, \
                            gen_inverse_dynamics_gradient_device, gen_inverse_dynamics_gradient_device_function_call, \
                            gen_inverse_dynamics_gradient_kernel, gen_inverse_dynamics_gradient_host, gen_inverse_dynamics_gradient, \
                            gen_forward_dynamics_gradient_inner_temp_mem_size, \
                            gen_forward_dynamics_gradient_inner_python, gen_forward_dynamics_gradient_kernel, \
                            gen_forward_dynamics_gradient_device, gen_forward_dynamics_gradient_device_function_call, \
                            gen_forward_dynamics_gradient_host, gen_forward_dynamics_gradient, \
                            gen_f_ext_gradient_inner_temp_mem_size, gen_f_ext_gradient_inner_function_call, \
                            gen_f_ext_gradient_jacobianT_inner, gen_f_ext_gradient_device, \
                            gen_f_ext_gradient_dq_kernel, gen_f_ext_gradient_dq_host, gen_f_ext_gradient_dq_num_jobs, \
                            gen_f_ext_gradient_kernel, gen_f_ext_gradient_host, gen_f_ext_gradient, \
                            gen_end_effector_pose_inner_temp_mem_size, gen_end_effector_pose_inner_function_call, gen_end_effector_pose_inner, \
                            gen_end_effector_pose_device, gen_end_effector_pose_kernel, \
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
                            gen_floating_gravity_d2tau_dq_shared_count, \
                            gen_floating_gravity_d2tau_dq_spill_count, gen_floating_gravity_d2tau_dq_lie_inline, \
                            gen_fdsva_so, gen_fdsva_so_contract_temp_mem_size, gen_fdsva_so_fd_gradient_inline_temp_mem_size, gen_fdsva_so_fd_gradient_inline_temp_mem_size_spilled, gen_fdsva_so_fd_gradient_inline, gen_fdsva_so_contract_function_call, gen_fdsva_so_contract, \
                            gen_fdsva_so_device, gen_fdsva_so_device_function_call, gen_fdsva_so_kernel, gen_fdsva_so_host, \
                            gen_integrator_inner_temp_mem_size, gen_integrator_finish_function_call, gen_integrator_finish, \
                            _spherical_retract_index_tables, _emit_q_update, gen_integrate_spherical_helper, \
                            gen_spherical_dintegrate_helpers, \
                            gen_integrator_inner_function_call, gen_integrator_inner, gen_integrator_device, \
                            gen_integrator_kernel, gen_integrator_host, gen_integrator, gen_lie_group_helpers, \
                            gen_integrator_arena_carve_struct, gen_integrator_du_arena_carve_struct, \
                            gen_integrator_gradient_inner_temp_mem_size, gen_integrator_gradient_dAB_assembly, \
                            gen_integrator_gradient_inner_python, gen_integrator_gradient_multistage, \
                            gen_integrator_gradient_device, gen_integrator_gradient_device_function_call, \
                            gen_integrator_gradient_kernel, gen_integrator_gradient_host, gen_integrator_gradient, \
                            gen_integrator_hessian_device, \
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
                            gen_f_ext_contact, gen_f_ext_contact_runtime, gen_f_ext_contact_inner_temp_mem_size, \
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
    from .algorithms._combos import gen_combination_functions, gen_centroidal_quickwins
    from ._launch_config import gen_add_launch_config_helpers
    from ._constants_arena import gen_add_constants_helpers, gen_init_gridData
    from ._kernel_attrs import (_f_ext_gradient_dq_emitted, KERNEL_OVERLOADS,
                                KERNEL_ATTR_MANIFEST, MJX_KERNEL_OVERLOADS,
                                gen_init_close_grid)
    from .helpers._gpu_err import gen_add_gpu_err
    from .algorithms._dccrba import _dccrba_inner_temp_mem_size, _dccrba_sweep_J_count, gen_cmm_time_variation, gen_dccrba

    # finally import the test code
    from ._reference_impl import test_rnea_fpass, test_rnea_bpass, test_rnea, test_minv_bpass, test_minv_fpass, test_densify_Minv, test_minv, test_rnea_grad_inner, \
                      test_rnea_grad, mx0, mx1, mx2, mx3, mx4, mx5, mxS, fxv

    # initialize the object
    def __init__(self, robotObj, DEBUG_MODE = False, NEED_PRINT_MAT = False, FILE_NAMESPACE = "grid", USE_JOINT_DYNAMICS = False, dtype = "float", MUJOCO_OUTPUT = False, LAUNCH_CONFIG_ROBOT = None, LAUNCH_CONFIG_PROFILE = "host", runtime_joint_dynamics = False):
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
        # config/launch_configs/<robot>/<DEFAULT_GPU>.json (per-algo {tier,threads}).
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
        from ._algo_profiles import normalize_codegen_algorithms
        return normalize_codegen_algorithms(self, codegen_profile, algorithm_list)
    
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

        
    def gen_all_code(self, include_base_inertia = False, include_homogenous_transforms = False, fixed_target_name = "", output_path = None,
                     codegen_profile = "all", algorithm_list = None, enable_floating_second_order = True,
                     enable_idsva_so_world_frame = None, runtime_inertia = False, runtime_transform = False,
                     runtime_joint_dynamics = None, multi_target_batch = None, collision_spec = None,
                     contact_frames = None, enable_contact_runtime = False, enable_mujoco_kernels = None,
                     emit_alloc_gating = False):
        # enable_mujoco_kernels=False builds a PIN-ONLY header: the mjx
        # (MUJOCO_OUTPUT=true) template overloads are still EMITTED (they are
        # templates -- uninstantiated they cost nothing; a bare #include is 2 s /
        # 0.3 GB), but nothing INSTANTIATES them, because the only two triggers are
        # gated off: (1) the aggregate init_grid_kernel_attrs' mjx registration block
        # and (2) `#define GRID_RBD_WITH_MUJOCO`, which is what compiles the binding's
        # mjx C-ABI entry points. That matters because the mjx twin of a
        # derivative/second-order kernel is larger than its pin counterpart. Those
        # twins used to be enormous (idsva_so_world_frame was 28x pin raw); the epilogue
        # is now block-parallel (go2-floating: idsva_so 2.42x, fdsva_so 1.41x pin -- see
        # docs/agent_debugging_guide.md 1u), but the second-order twins are still the
        # largest kernels and the bulk of a big-humanoid build. Pin-only consumers
        # (GATO/PDDP 2nd-order DDP, and anyone not using the MuJoCo output convention)
        # can opt out and pay none of it. Default True = existing behavior, byte-identical.
        #
        # None (the default) means "consult GRID_ENABLE_MUJOCO_KERNELS, else True", so a
        # whole session can go pin-only without touching each of the ~50 gen_all_code
        # call sites (the CUDA equivalence suite does exactly this -- see
        # test/cuda_equivalents/conftest.py). An EXPLICIT True/False always wins over the
        # env var, so a test that genuinely exercises mjx can opt back in locally.
        if enable_mujoco_kernels is None:
            enable_mujoco_kernels = os.environ.get("GRID_ENABLE_MUJOCO_KERNELS", "1") != "0"
        self.enable_mujoco_kernels = enable_mujoco_kernels
        # 2a: opt-in per-algo alloc gating in init_gridData (see gen_init_gridData).
        # Default False emits the header BYTE-IDENTICAL to before the feature; the
        # per-algo bench turns it on so its solo exes can -D away other algos'
        # (multi-GB at nv=81) buffers.
        self.emit_alloc_gating = emit_alloc_gating
        # (The former bench-only GRID_WORKSPACE_CHUNK seam is now an always-on
        # RUNTIME feature: init_gridData auto-fits gridData.workspace_timestep_slots
        # from cudaMemGetInfo, kernels index the workspace arena per-BLOCK slot,
        # and workspace-using host wrappers clamp their launch grid to the slot
        # count. See gen_add_shared_memory_helpers / gen_add_workspace_clamped_launch.)
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
                             "fdsva_so",
                             # integrator gradient slice (2026-07-30): per-joint SO(3)
                             # dIntegrate blocks; single-stage IT only (multi-stage RK
                             # static_asserts in the device — follow-on slice).
                             "integrator_gradient", "integrator_with_gradient",
                             # ee pose gradient/hessian slice (2026-08-02): the
                             # geometric-Jacobian gradient inner and the analytic
                             # chain-composition hessian inner are S-column driven,
                             # so a spherical joint's 3 tangent columns (d/dv, local
                             # body-frame omega, pinocchio JointModelSpherical order)
                             # flow through the same per-(ee, S-col) fill jobs and
                             # intra-joint rev-rev pair blocks the floating root
                             # uses. Requires the URDFParser forward hom composition
                             # (origin ∘ exp(q), pinocchio placement convention).
                             "end_effector_pose_gradient", "end_effector_pose_hessian"}
            _unported = sorted(a for a in algorithms if a not in _SPHERICAL_OK)
            if _unported:
                raise NotImplementedError(
                    "Spherical (ball) joint CUDA codegen currently supports "
                    "inverse_dynamics + crba + minv + forward_dynamics + "
                    f"end_effector_pose + its gradient/hessian + frame_jacobian + integrator + "
                    f"integrator_gradient/with_gradient (single-stage IT) + "
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
        # The `grid::idsva_so` dispatcher forwards at CODEGEN time (world frame for
        # floating / spherical / high-DOF fixed — _idsva_so_use_world_frame). A
        # consumer probing launchability (e.g. the bench smem-skip guard) must
        # check the DISPATCHED kernel's smem bytes, not a fixed frame: on floating
        # the body frame is a no-ladder diagnostic (242 KB on h2_plus) while the
        # dispatched world frame fits at every tier.
        self.gen_add_code_line(
            "#define GRID_IDSVA_SO_DISPATCHES_WORLD_FRAME "
            + str(int(_idsva_so_use_world_frame(self))))
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
        # enable_mujoco_kernels=False suppresses this define -> the binding's mjx
        # entry points compile out -> the mjx twins are never instantiated (trigger 2
        # of 2; see gen_all_code). getattr keeps legacy callers that never set it.
        if getattr(self, "enable_mujoco_kernels", True) and self.robot.floating_base and not (
                self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis()):
            self.gen_add_code_line("#define GRID_RBD_WITH_MUJOCO 1")
            # Warn BEFORE the multi-hour compile rather than after the OOM. The mjx twins
            # dominate a big floating-base build (~2.1M of ~2.4M SASS lines on
            # g1-floating; the idsva_so_world_frame twin alone is 28x its pin kernel), so
            # a humanoid built with them can exhaust a 62 GB box, while the same robot
            # built pin-only lands in ~33 min at ~11 GB peak. Threshold is empirical:
            # go2-floating (nv=18) builds fine with mjx; g1-floating (nv=36) did not
            # build at all. Only fires where the twins actually exist (floating,
            # non-mimic, non-skew) and only when they are switched ON.
            if self.robot.get_num_vel() >= 30:
                warnings.warn(
                    f"Generating MuJoCo-convention (mjx) kernel twins for a large "
                    f"floating-base robot (nv={self.robot.get_num_vel()}). These twins "
                    f"dominate the build and can exhaust host RAM during nvcc/cicc. If "
                    f"you do not need the MuJoCo output convention, build pin-only with "
                    f"enable_mujoco_kernels=False (register_robot / gen_all_code), or set "
                    f"GRID_ENABLE_MUJOCO_KERNELS=0 for a whole codegen session.",
                    stacklevel=2,
                )
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
            # Tool-use: runtime single-contact f_ext (the welded-tool tip). Body id + local
            # offset are RUNTIME args. Opt-in (default False => byte-identical header).
            if enable_contact_runtime:
                self.gen_f_ext_contact_runtime()
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
                    if "pb" in _t:
                        # capsule tier: each row contributes TWO targets (endpoint a then b),
                        # so row i's world endpoints land at s_out_pos[6i..6i+5].
                        _off2 = []
                        for _i in range(_t["n"]):
                            _off2.extend(_t["offset"][3 * _i:3 * _i + 3])
                            _off2.extend(_t["pb"][3 * _i:3 * _i + 3])
                        _cc_batch = {"n": 2 * _t["n"],
                                     "anchor": [_a for _a in _t["anchor"] for _ in (0, 1)],
                                     "offset": _off2, "groups": {"all": (0, 2 * _t["n"])}}
                    else:
                        _cc_batch = {"n": _t["n"], "anchor": _t["anchor"],
                                     "offset": _t["offset"], "groups": {"all": (0, _t["n"])}}
                    self.gen_multi_target_position(_cc_batch, suffix=_t["suffix"])
                    if _t["suffix"] == "":  # finest / public tier -> the differentiable path
                        self.gen_multi_target_position_gradient(_cc_batch, suffix="")
        if self.robot.floating_base and not enable_floating_second_order:
            warnings.warn('floating-base second order dynamics are still under development')
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
            # GATO ASK6: namespace-scope carve struct mirroring the integrator
            # kernel's TIER_SHARED arena (additive; external-caller surface).
            self.gen_integrator_arena_carve_struct()
        if ("integrator_gradient" in algorithms) or ("integrator_with_gradient" in algorithms):
            self.gen_integrator_gradient()
            # GATO ASK6: the du-kernel twin (with-x_kp1 shape, TIER_SHARED rung).
            self.gen_integrator_du_arena_carve_struct()
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
                # ANALYTIC Jdot arena = s_XmatsHom + extras(s_wvel[3*NJ] + s_vvel[3*NJ] +
                # s_Jval[6nv]) + inner_temp(16*NJ, s_Xworld). The launchable kernel keeps
                # its input (s_q_qd) and output (s_frame_jacobian_dot) in STATIC __shared__
                # — NOT this dynamic arena, which the frame_jacobian_dot_device wrapper owns
                # entirely — so this size matches the wrapper exactly.
                fjd_t_count = Xhom_size_fj + (6 * NJ_fj) + (6 * nv_fj) + (16 * NJ_fj)
                self.gen_add_code_line(
                    "template <typename T> __host__ __device__ inline size_t FRAME_JACOBIAN_DOT_DYNAMIC_SHARED_MEM_BYTES() "
                    "{ return grid_shared_arena_bytes<T>(" + str(fjd_t_count) +
                    ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }")
                self.gen_add_code_line("#define GRID_HAS_FRAME_JACOBIAN_DOT 1")
                # Floating-base Jdot integrates q on the SE(3) group; emit the Lie
                # helpers if no other kinematics path already did.
                if self.robot.floating_base:
                    self.gen_lie_group_helpers()
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
        # Host-side thread-count clamp pass (2026-08-10): every host-wrapper
        # kernel launch that takes the caller's `thread_dimms` gets it clamped
        # against the LAUNCHED kernel's own cudaFuncAttributes cap first. A
        # caller passing a thread count above the kernel's register/launch_bounds
        # cap used to get a silently-rejected launch (empty stream, stale output
        # buffers — the fr3 cmm_time_variation zeros, same class as the plant
        # 2026-06-19 incident). One pass fixes all ~170 launch sites for every
        # consumer (bindings, GATO/MPCGPU-style direct callers, bench exes).
        self.code_str = _apply_host_thread_clamp_pass(self.code_str)
        # then output to a file
        if output_path is None:
            output_path = self.file_namespace + ".cuh"
        file = open(output_path, "w")
        file.write(self.code_str)
        file.close()
