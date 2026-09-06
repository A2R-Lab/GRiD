// grid_rbd._core — pybind11 Runner that dlopens a per-robot .so and
// dispatches numpy arrays through its C ABI.
//
// The per-robot .so is built at register_robot() time by
// grid_rbd._compile.compile_sources() from a generated grid.cuh plus
// the robot-agnostic wrapper.cu (see grid_rbd/wrapper_template.cu). It
// exports `extern "C"` symbols like:
//
//   int grid_rbd_init();
//   int grid_rbd_num_joints();
//   int grid_rbd_inverse_dynamics(const CT* q, const CT* qd, const CT* qdd_opt,
//                     CT* c_out, int batch, CT gravity,
//                     const CT* f_ext_opt);  // f_ext_opt may be nullptr
//   ... etc ...
//
// The Runner constructor dlopens the .so and resolves every symbol it
// knows about. Algorithms whose symbols are missing from a given .so
// (e.g. an old build before a method was added) raise a clear error
// at call time.
//
// Single-process / single-robot assumption: the wrapper.cu holds device
// buffers as file-scope statics keyed by NUM_JOINTS / NUM_VEL, so each
// .so manages one robot. If two Runners are constructed against the same
// .so in the same process, they share state — fine for now, future-v2
// concern.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <dlfcn.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>

namespace py = pybind11;


// ─── C ABI function signatures (must match wrapper_template.cu) ──────────────
//
// fp64 (Phase 8): the per-robot .so's `extern "C"` symbols take `const CT*` /
// `CT*` where CT == float (default) or CT == double (a .so built with
// -DGRID_WRAPPER_T_DOUBLE). dlsym carries no type, so the Runner must declare
// its function-pointer typedefs AND its numpy buffers in the SAME element type
// as the .so it dlopen'd. We therefore template the whole signature set + Runner
// on the buffer C-type CT and register one pybind class per dtype (Runner =
// float, RunnerF64 = double). The float path is unchanged.
// NOTE on scalar arg types: in wrapper_template.cu `gravity` and `dt` are `T`
// (so they become double in an fp64 .so) but `mu` is ALWAYS `float`. The
// function-pointer signatures below must match EXACTLY (a by-value scalar
// passed at the wrong width corrupts the ABI), so gravity/dt use CT and mu
// stays float. The pybind11 METHOD parameters that receive these from Python
// must ALSO be CT (not float): a Python float is a C double, so declaring the
// param `float` would round -9.81 to fp32 BEFORE the (double) ABI call and cap
// an fp64 .so's id/fd accuracy at ~4e-8 (the fp32 path is byte-identical either
// way since CT==float there). So gravity/dt params are CT; mu stays float.
template <class CT>
struct CAbi {
    using fn_int_v_t        = int (*)();
    using fn_int_i_t        = int (*)(int);
    using fn_int_s_t        = int (*)(const char*);   // kernel_max_threads(algo)
    using fn_int_ii_t       = int (*)(int, int);      // set_threads_for(algo, n)
    using fn_int_iii_t      = int (*)(int, int, int); // set_threads_for_n(algo, threshold, n_small)
    using fn_int_ipp_t      = int (*)(int, int*, int*); // get_batch_switch(algo, &threshold, &n_small)
    using fn_dyn_t         = int (*)(const CT*, const CT*, const CT*,
                                      CT*, int, CT, const CT*);
    using fn_minv_t         = int (*)(const CT*, CT*, int);
    using fn_fd_t           = int (*)(const CT*, const CT*, const CT*,
                                      CT*, int, CT, const CT*);
    using fn_dyn_no_fext_t = int (*)(const CT*, const CT*, const CT*,
                                      CT*, int, CT);
    using fn_fd_no_fext_t   = int (*)(const CT*, const CT*, const CT*,
                                      CT*, int, CT);
    using fn_crba_t         = int (*)(const CT*, CT*, int, CT);
    using fn_ee_t           = int (*)(const CT*, CT*, int);
    using fn_fk_batched_t   = int (*)(const CT*, CT*, int, int);
    using fn_integrator_t   = int (*)(const CT*, const CT*, const CT*,
                                      CT*, int, CT, CT, int);
    using fn_plant_cost_t   = int (*)(const CT*, const CT*, const CT*,
                                      CT*, CT*, CT*, int);
    using fn_plant_barrier_t = int (*)(const CT*, const CT*, const CT*, float,
                                       CT*, CT*, CT*, int);
    using fn_plant_step_t   = int (*)(const CT*, const CT*, CT*,
                                      int, CT, CT, int);
    using fn_plant_ee_t     = int (*)(const CT*, const CT*, const CT*,
                                      CT*, CT*, CT*, int);
    using fn_plant_mom_t    = int (*)(const CT*, const CT*, const CT*, const CT*,
                                      CT*, CT*, CT*, int);
    using fn_plant_step_grad_t = int (*)(const CT*, const CT*, CT*,
                                         int, CT, CT, int);
    using fn_plant_step_hess_t = int (*)(const CT*, const CT*, CT*,
                                         int, CT, CT, int);
    using fn_q_out_t        = int (*)(const CT*, CT*, int);
    using fn_q_qd_out_t     = int (*)(const CT*, const CT*, CT*, int);
    using fn_frame_jac_t    = int (*)(const CT*, CT*, int, int, int);
    using fn_frame_jac_dot_t = int (*)(const CT*, const CT*, CT*, int, int, int);
    using fn_ee_runtime_t   = int (*)(const CT*, CT*, int, int, const CT*);
    using fn_tool_fext_t    = int (*)(const CT*, const CT*, int, const CT*, CT*, int);  // grid_rbd_tool_fext
    using fn_q_qd_out_grav_t = int (*)(const CT*, const CT*, CT*, int, CT);
    using fn_q_out_grav_t   = int (*)(const CT*, CT*, int, CT);
    using fn_set_inertia_t  = int (*)(const CT*);   // grid_rbd_set_inertia_params
    using fn_set_transform_t = int (*)(const CT*);  // grid_rbd_set_transform_params
    using fn_set_jd_t       = int (*)(const CT*);   // grid_rbd_set_joint_dynamics_params
};


// ─── Runner ──────────────────────────────────────────────────────────────────

template <class CT>
class RunnerT {
    // C-ABI function-pointer typedefs (parameterized on the buffer dtype CT).
    using fn_int_v_t = typename CAbi<CT>::fn_int_v_t;
    using fn_int_i_t = typename CAbi<CT>::fn_int_i_t;
    using fn_int_s_t = typename CAbi<CT>::fn_int_s_t;
    using fn_int_ii_t = typename CAbi<CT>::fn_int_ii_t;
    using fn_int_iii_t = typename CAbi<CT>::fn_int_iii_t;
    using fn_int_ipp_t = typename CAbi<CT>::fn_int_ipp_t;
    using fn_dyn_t = typename CAbi<CT>::fn_dyn_t;
    using fn_minv_t = typename CAbi<CT>::fn_minv_t;
    using fn_fd_t = typename CAbi<CT>::fn_fd_t;
    using fn_dyn_no_fext_t = typename CAbi<CT>::fn_dyn_no_fext_t;
    using fn_fd_no_fext_t = typename CAbi<CT>::fn_fd_no_fext_t;
    using fn_crba_t = typename CAbi<CT>::fn_crba_t;
    using fn_ee_t = typename CAbi<CT>::fn_ee_t;
    using fn_fk_batched_t = typename CAbi<CT>::fn_fk_batched_t;
    using fn_integrator_t = typename CAbi<CT>::fn_integrator_t;
    using fn_plant_cost_t = typename CAbi<CT>::fn_plant_cost_t;
    using fn_plant_barrier_t = typename CAbi<CT>::fn_plant_barrier_t;
    using fn_plant_step_t = typename CAbi<CT>::fn_plant_step_t;
    using fn_plant_ee_t = typename CAbi<CT>::fn_plant_ee_t;
    using fn_plant_mom_t = typename CAbi<CT>::fn_plant_mom_t;
    using fn_plant_step_grad_t = typename CAbi<CT>::fn_plant_step_grad_t;
    using fn_plant_step_hess_t = typename CAbi<CT>::fn_plant_step_hess_t;
    using fn_q_out_t = typename CAbi<CT>::fn_q_out_t;
    using fn_q_qd_out_t = typename CAbi<CT>::fn_q_qd_out_t;
    using fn_frame_jac_t = typename CAbi<CT>::fn_frame_jac_t;
    using fn_frame_jac_dot_t = typename CAbi<CT>::fn_frame_jac_dot_t;
    using fn_ee_runtime_t = typename CAbi<CT>::fn_ee_runtime_t;
    using fn_tool_fext_t = typename CAbi<CT>::fn_tool_fext_t;
    using fn_q_qd_out_grav_t = typename CAbi<CT>::fn_q_qd_out_grav_t;
    using fn_q_out_grav_t = typename CAbi<CT>::fn_q_out_grav_t;
    using fn_set_inertia_t = typename CAbi<CT>::fn_set_inertia_t;
    using fn_set_transform_t = typename CAbi<CT>::fn_set_transform_t;
    using fn_set_jd_t = typename CAbi<CT>::fn_set_jd_t;
    // Per-dtype numpy array alias: an input is force-cast to CT, outputs are CT.
    using arr_t = py::array_t<CT, py::array::c_style | py::array::forcecast>;
public:
    explicit RunnerT(const std::string& so_path) {
        handle_ = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle_) {
            throw std::runtime_error(
                std::string("dlopen failed for ") + so_path + ": " + dlerror());
        }

        // Metadata symbols — required.
        fn_num_joints_       = reinterpret_cast<fn_int_v_t>(require_sym("grid_rbd_num_joints"));
        fn_num_vel_          = reinterpret_cast<fn_int_v_t>(require_sym("grid_rbd_num_vel"));
        fn_num_ees_          = reinterpret_cast<fn_int_v_t>(require_sym("grid_rbd_num_ees"));
        // num_bodies — OPTIONAL (older .so built before the f_ext surface lacks
        // it). Used to size/validate the optional f_ext arg; fall back to 0
        // (f_ext then rejected with a clear error) if absent.
        fn_num_bodies_       = reinterpret_cast<fn_int_v_t>(opt_sym("grid_rbd_num_bodies"));
        fn_max_batch_        = reinterpret_cast<fn_int_v_t>(require_sym("grid_rbd_max_batch"));
        fn_max_perf_level_threads_ = reinterpret_cast<fn_int_v_t>(require_sym("grid_rbd_max_perf_level_threads"));
        fn_threads_per_block_ = reinterpret_cast<fn_int_v_t>(require_sym("grid_rbd_threads_per_block"));
        fn_set_threads_per_block_ = reinterpret_cast<fn_int_i_t>(require_sym("grid_rbd_set_threads_per_block"));
        // OPTIONAL (E1): pre-this-patch .so lacks it -> kernel_max_threads returns -1
        // and the FFI autotune falls back to swept-ceiling inference (never crashes).
        fn_kernel_max_threads_ = reinterpret_cast<fn_int_s_t>(opt_sym("grid_rbd_kernel_max_threads"));
        // OPTIONAL (E6 per-algo overlay): pre-this-patch .so lacks these -> overlay is a
        // graceful no-op (has_per_algo_threads() == false).
        fn_set_threads_for_  = reinterpret_cast<fn_int_ii_t>(opt_sym("grid_rbd_set_threads_for"));
        fn_algo_count_       = reinterpret_cast<fn_int_v_t>(opt_sym("grid_rbd_algo_count"));
        fn_set_threads_for_n_ = reinterpret_cast<fn_int_iii_t>(opt_sym("grid_rbd_set_threads_for_n"));
        fn_get_batch_switch_ = reinterpret_cast<fn_int_ipp_t>(opt_sym("grid_rbd_get_batch_switch"));
        fn_init_             = reinterpret_cast<fn_int_v_t>(require_sym("grid_rbd_init"));
        fn_close_            = reinterpret_cast<fn_int_v_t>(require_sym("grid_rbd_close"));

        // Algorithm symbols — required for v1 surface.
        fn_inverse_dynamics_             = reinterpret_cast<fn_dyn_t>(require_sym("grid_rbd_inverse_dynamics"));
        // MuJoCo output-convention ID kernel: optional symbol — present ONLY in a
        // mjx-capable .so (floating, non-mimic, non-skew; gated on GRID_RBD_WITH_MUJOCO
        // in the wrapper). nullptr on fixed-base / mimic / older .so, in which case
        // the mjx method raises.
        fn_inverse_dynamics_mujoco_      = reinterpret_cast<fn_dyn_t>(opt_sym("grid_rbd_inverse_dynamics_mujoco"));
        fn_minv_             = reinterpret_cast<fn_minv_t>(require_sym("grid_rbd_minv"));
        fn_fd_               = reinterpret_cast<fn_fd_t>  (require_sym("grid_rbd_forward_dynamics"));
        fn_aba_              = reinterpret_cast<fn_fd_t>  (require_sym("grid_rbd_aba"));
        fn_crba_             = reinterpret_cast<fn_crba_t>(require_sym("grid_rbd_crba"));
        fn_crba_mujoco_      = reinterpret_cast<fn_crba_t>(opt_sym("grid_rbd_crba_mujoco"));  // floating only
        // floating-base mjx value kernels (optional; present only on a floating .so)
        fn_fd_mujoco_        = reinterpret_cast<fn_fd_t>(opt_sym("grid_rbd_forward_dynamics_mujoco"));
        fn_aba_mujoco_       = reinterpret_cast<fn_fd_t>(opt_sym("grid_rbd_aba_mujoco"));
        fn_coriolis_matrix_mujoco_ = reinterpret_cast<fn_q_qd_out_grav_t>(opt_sym("grid_rbd_coriolis_matrix_mujoco"));
        fn_frame_jacobian_mujoco_  = reinterpret_cast<fn_frame_jac_t>(opt_sym("grid_rbd_frame_jacobian_mujoco"));
        fn_frame_jacobian_dot_mujoco_ = reinterpret_cast<fn_frame_jac_dot_t>(opt_sym("grid_rbd_frame_jacobian_dot_mujoco"));
        fn_osc_inertia_mujoco_     = reinterpret_cast<fn_q_out_t>(opt_sym("grid_rbd_osc_inertia_mujoco"));
        // floating-base mjx value kernels (optional; present only on a floating .so)
        fn_minv_mujoco_      = reinterpret_cast<fn_minv_t>(opt_sym("grid_rbd_minv_mujoco"));
        fn_com_mujoco_       = reinterpret_cast<fn_q_out_t>(opt_sym("grid_rbd_com_mujoco"));
        fn_ccrba_mujoco_     = reinterpret_cast<fn_q_qd_out_t>(opt_sym("grid_rbd_ccrba_mujoco"));
        fn_energy_mujoco_    = reinterpret_cast<fn_q_qd_out_grav_t>(opt_sym("grid_rbd_energy_mujoco"));
        fn_kinetic_energy_regressor_mujoco_   = reinterpret_cast<fn_q_qd_out_grav_t>(opt_sym("grid_rbd_kinetic_energy_regressor_mujoco"));
        fn_potential_energy_regressor_mujoco_ = reinterpret_cast<fn_q_out_grav_t>(opt_sym("grid_rbd_potential_energy_regressor_mujoco"));
        fn_ee_pose_          = reinterpret_cast<fn_ee_t>  (require_sym("grid_rbd_end_effector_pose"));
        fn_ee_pose_grad_     = reinterpret_cast<fn_ee_t>  (require_sym("grid_rbd_end_effector_pose_gradient"));
        // floating-base mjx EE kernels (optional; present only on a floating .so)
        fn_ee_pose_mujoco_      = reinterpret_cast<fn_ee_t>(opt_sym("grid_rbd_end_effector_pose_mujoco"));
        fn_ee_pose_grad_mujoco_ = reinterpret_cast<fn_ee_t>(opt_sym("grid_rbd_end_effector_pose_gradient_mujoco"));
        fn_inverse_dynamics_gradient_        = reinterpret_cast<fn_dyn_t>(require_sym("grid_rbd_inverse_dynamics_gradient"));
        fn_inverse_dynamics_gradient_mujoco_ = reinterpret_cast<fn_dyn_t>(opt_sym("grid_rbd_inverse_dynamics_gradient_mujoco"));  // floating only
        fn_fd_grad_          = reinterpret_cast<fn_fd_t>  (require_sym("grid_rbd_forward_dynamics_gradient"));
        fn_fd_grad_mujoco_   = reinterpret_cast<fn_fd_t>  (opt_sym("grid_rbd_forward_dynamics_gradient_mujoco"));  // floating only
        // Phase-C extension: hessian + SO. Required for v0.1+ .so files.
        fn_ee_pose_hessian_  = reinterpret_cast<fn_ee_t>  (require_sym("grid_rbd_end_effector_pose_hessian"));
        fn_ee_pose_hessian_mujoco_ = reinterpret_cast<fn_ee_t>(opt_sym("grid_rbd_end_effector_pose_hessian_mujoco"));  // floating only
        fn_idsva_so_         = reinterpret_cast<fn_dyn_no_fext_t>(require_sym("grid_rbd_idsva_so"));
        fn_idsva_so_mujoco_  = reinterpret_cast<fn_dyn_no_fext_t>(opt_sym("grid_rbd_idsva_so_mujoco"));  // floating only
        fn_id_regressor_        = reinterpret_cast<fn_dyn_no_fext_t>(opt_sym("grid_rbd_inverse_dynamics_regressor"));
        fn_id_regressor_mujoco_ = reinterpret_cast<fn_dyn_no_fext_t>(opt_sym("grid_rbd_inverse_dynamics_regressor_mujoco"));  // floating only
        fn_fdsva_so_         = reinterpret_cast<fn_fd_no_fext_t>  (require_sym("grid_rbd_fdsva_so"));
        fn_fdsva_so_mujoco_  = reinterpret_cast<fn_fd_no_fext_t>  (opt_sym("grid_rbd_fdsva_so_mujoco"));  // floating only
        fn_integrator_       = reinterpret_cast<fn_integrator_t>(require_sym("grid_rbd_integrator"));
        fn_integrator_mujoco_ = reinterpret_cast<fn_integrator_t>(opt_sym("grid_rbd_integrator_mujoco"));  // floating only
        fn_integrator_grad_  = reinterpret_cast<fn_integrator_t>(require_sym("grid_rbd_integrator_gradient"));
        fn_integrator_grad_mujoco_ = reinterpret_cast<fn_integrator_t>(opt_sym("grid_rbd_integrator_gradient_mujoco"));  // floating only

        // grid_plant C ABI (G1) — OPTIONAL: resolve if present (older .so files
        // built before the plant surface won't have them; the handle methods
        // raise a clear error at call time if the symbol is null).
        fn_plant_state_cost_ = reinterpret_cast<fn_plant_cost_t>(opt_sym("grid_plant_quadratic_state_cost"));
        fn_plant_input_cost_ = reinterpret_cast<fn_plant_cost_t>(opt_sym("grid_plant_quadratic_input_cost"));
        fn_plant_pos_barrier_ = reinterpret_cast<fn_plant_barrier_t>(opt_sym("grid_plant_joint_position_barrier"));
        fn_plant_vel_barrier_ = reinterpret_cast<fn_plant_barrier_t>(opt_sym("grid_plant_joint_velocity_barrier"));
        fn_plant_tor_barrier_ = reinterpret_cast<fn_plant_barrier_t>(opt_sym("grid_plant_joint_torque_barrier"));
        fn_plant_step_       = reinterpret_cast<fn_plant_step_t>(opt_sym("grid_plant_step"));
        fn_plant_step_mujoco_ = reinterpret_cast<fn_plant_step_t>(opt_sym("grid_plant_step_mujoco"));  // floating only
        fn_plant_ee_cost_    = reinterpret_cast<fn_plant_ee_t>(opt_sym("grid_plant_ee_pos_cost"));
        fn_plant_com_cost_   = reinterpret_cast<fn_plant_ee_t>(opt_sym("grid_plant_com_cost"));
        fn_plant_mom_cost_   = reinterpret_cast<fn_plant_mom_t>(opt_sym("grid_plant_momentum_cost"));
        fn_plant_ee_cost_mujoco_  = reinterpret_cast<fn_plant_ee_t>(opt_sym("grid_rbd_ee_pos_cost_mujoco"));   // floating only
        fn_plant_com_cost_mujoco_ = reinterpret_cast<fn_plant_ee_t>(opt_sym("grid_rbd_com_cost_mujoco"));      // floating only
        fn_plant_mom_cost_mujoco_ = reinterpret_cast<fn_plant_mom_t>(opt_sym("grid_rbd_momentum_cost_mujoco")); // floating only
        fn_plant_state_cost_mujoco_ = reinterpret_cast<fn_plant_cost_t>(opt_sym("grid_rbd_quadratic_state_cost_mujoco")); // floating only
        fn_plant_step_grad_  = reinterpret_cast<fn_plant_step_grad_t>(opt_sym("grid_plant_step_gradient"));
        fn_plant_step_grad_mujoco_ = reinterpret_cast<fn_plant_step_grad_t>(opt_sym("grid_plant_step_gradient_mujoco"));  // floating only
        fn_plant_step_hess_  = reinterpret_cast<fn_plant_step_hess_t>(opt_sym("grid_plant_step_hessian"));
        fn_plant_step_hess_mujoco_ = reinterpret_cast<fn_plant_step_hess_t>(opt_sym("grid_plant_step_hessian_mujoco"));  // floating only

        // G2 batched FK (pos+quat) — OPTIONAL: only present in newer .so files
        // (and only non-null for fixed-base/non-mimic robots).
        fn_fk_batched_      = reinterpret_cast<fn_fk_batched_t>(opt_sym("grid_rbd_fk_batched"));

        // F2 centroidal / energy / general-frame kinematics — OPTIONAL: present
        // in newer .so files. com/ccrba/energy/gg/nle are always emitted with
        // the "all" profile; frame_jacobian* / osc_inertia are opt-in codegen
        // (the C-ABI symbol returns rc=3 if the family wasn't generated).
        fn_com_                = reinterpret_cast<fn_q_out_t>(opt_sym("grid_rbd_com"));
        fn_ccrba_              = reinterpret_cast<fn_q_qd_out_t>(opt_sym("grid_rbd_ccrba"));
        fn_energy_             = reinterpret_cast<fn_q_qd_out_grav_t>(opt_sym("grid_rbd_energy"));
        fn_generalized_gravity_ = reinterpret_cast<fn_q_out_grav_t>(opt_sym("grid_rbd_generalized_gravity"));
        fn_generalized_gravity_mujoco_ = reinterpret_cast<fn_q_out_grav_t>(opt_sym("grid_rbd_generalized_gravity_mujoco"));  // floating only
        fn_nonlinear_effects_  = reinterpret_cast<fn_q_qd_out_grav_t>(opt_sym("grid_rbd_nonlinear_effects"));
        fn_nonlinear_effects_mujoco_ = reinterpret_cast<fn_q_qd_out_grav_t>(opt_sym("grid_rbd_nonlinear_effects_mujoco"));  // floating only
        fn_frame_jacobian_     = reinterpret_cast<fn_frame_jac_t>(opt_sym("grid_rbd_frame_jacobian"));
        fn_frame_jacobian_dot_ = reinterpret_cast<fn_frame_jac_dot_t>(opt_sym("grid_rbd_frame_jacobian_dot"));
        fn_osc_inertia_        = reinterpret_cast<fn_q_out_t>(opt_sym("grid_rbd_osc_inertia"));
        fn_ee_pose_runtime_      = reinterpret_cast<fn_ee_runtime_t>(opt_sym("grid_rbd_end_effector_pose_runtime"));
        fn_ee_pose_grad_runtime_ = reinterpret_cast<fn_ee_runtime_t>(opt_sym("grid_rbd_end_effector_pose_gradient_runtime"));
        fn_ee_pose_runtime_mujoco_      = reinterpret_cast<fn_ee_runtime_t>(opt_sym("grid_rbd_end_effector_pose_runtime_mujoco"));            // floating only
        fn_ee_pose_grad_runtime_mujoco_ = reinterpret_cast<fn_ee_runtime_t>(opt_sym("grid_rbd_end_effector_pose_gradient_runtime_mujoco")); // floating only
        fn_tool_fext_            = reinterpret_cast<fn_tool_fext_t>(opt_sym("grid_rbd_tool_fext"));  // enable_tool only

        // PS5 value ops — OPTIONAL: present in newer .so files.
        // coriolis_matrix / kinetic_energy_regressor / potential_energy_regressor
        // are always emitted with the "all" profile (mimic-safe). dccrba /
        // cmm_time_variation are skipped for mimic robots (the per-body Jacobian
        // fold isn't mimic-reduced), so their C-ABI symbol returns rc=3 there.
        fn_coriolis_matrix_    = reinterpret_cast<fn_q_qd_out_grav_t>(opt_sym("grid_rbd_coriolis_matrix"));
        fn_kinetic_energy_regressor_   = reinterpret_cast<fn_q_qd_out_grav_t>(opt_sym("grid_rbd_kinetic_energy_regressor"));
        fn_potential_energy_regressor_ = reinterpret_cast<fn_q_out_grav_t>(opt_sym("grid_rbd_potential_energy_regressor"));
        fn_dccrba_             = reinterpret_cast<fn_q_out_t>(opt_sym("grid_rbd_dccrba"));
        fn_dccrba_mujoco_      = reinterpret_cast<fn_q_out_t>(opt_sym("grid_rbd_dccrba_mujoco"));  // floating only
        fn_cmm_time_variation_ = reinterpret_cast<fn_q_qd_out_t>(opt_sym("grid_rbd_cmm_time_variation"));
        // floating-base mjx variant (optional; present only on a floating .so)
        fn_cmm_time_variation_mujoco_ = reinterpret_cast<fn_q_qd_out_t>(opt_sym("grid_rbd_cmm_time_variation_mujoco"));

        // D.4 / Phase 5 runtime-mutable inertia — OPTIONAL: present only in a .so
        // built with runtime_inertia=True (compiled with -DGRID_RBD_RUNTIME_INERTIA).
        // set_inertia_params() raises a clear error if this symbol is null.
        fn_set_inertia_params_ = reinterpret_cast<fn_set_inertia_t>(opt_sym("grid_rbd_set_inertia_params"));

        // runtime_transform — OPTIONAL: present only in a .so built with
        // runtime_transform=True (-DGRID_RBD_RUNTIME_TRANSFORM). set_transform_params()
        // raises a clear error if this symbol is null.
        fn_set_transform_params_ = reinterpret_cast<fn_set_transform_t>(opt_sym("grid_rbd_set_transform_params"));

        // runtime_joint_dynamics — OPTIONAL: present only in a .so built with
        // runtime_joint_dynamics=True (-DGRID_RBD_RUNTIME_JOINT_DYNAMICS).
        // set_joint_dynamics_params() raises a clear error if this symbol is null.
        fn_set_jd_params_ = reinterpret_cast<fn_set_jd_t>(opt_sym("grid_rbd_set_joint_dynamics_params"));

        // Cache constants (avoid the indirect-function-call cost on every read).
        num_joints_ = fn_num_joints_();
        num_vel_    = fn_num_vel_();
        num_ees_    = fn_num_ees_();
        max_batch_  = fn_max_batch_();
        num_bodies_ = fn_num_bodies_ ? fn_num_bodies_() : 0;

        // Initialize device buffers eagerly. wrapper_template.cu does this
        // lazily on first algo call too, but eager init surfaces CUDA errors
        // at register_robot time rather than first inference.
        if (fn_init_() != 0) {
            throw std::runtime_error(std::string("grid_rbd_init() failed in ") + so_path);
        }
    }

    ~RunnerT() {
        if (handle_) {
            if (fn_close_) fn_close_();
            dlclose(handle_);
            handle_ = nullptr;
        }
    }

    int num_joints() const { return num_joints_; }
    int num_vel()    const { return num_vel_; }
    int num_ees()    const { return num_ees_; }
    int num_bodies() const { return num_bodies_; }
    int max_batch()  const { return max_batch_; }
    int max_perf_level_threads() const { return fn_max_perf_level_threads_(); }
    // E1: real compiled __launch_bounds__ ceiling of the baked kernel for `algo`
    // (cudaFuncGetAttributes maxThreadsPerBlock). -1 if the symbol is absent (old
    // .so) or the key is unknown/not-built; the FFI autotune treats -1 as "infer".
    int kernel_max_threads(const std::string& algo) const {
        return fn_kernel_max_threads_ ? fn_kernel_max_threads_(algo.c_str()) : -1;
    }
    // E6 per-algo threads overlay: force `n` threads for the GridAlgo at index `algo`
    // (n==0 clears it back to the baked launch_cfg<ALGO>::THREADS). The global
    // set_threads_per_block override still wins when set.
    void set_threads_for(int algo, int n) {
        if (!fn_set_threads_for_)
            throw std::runtime_error("set_threads_for: this .so predates the per-algo "
                "threads overlay (rebuild to use profile overlays)");
        if (n < 0) throw std::invalid_argument("set_threads_for: n must be >= 0");
        int rc = fn_set_threads_for_(algo, n);
        if (rc != 0) throw std::runtime_error(
            "grid_rbd_set_threads_for failed: rc=" + std::to_string(rc));
    }
    int algo_count() const { return fn_algo_count_ ? fn_algo_count_() : 0; }
    bool has_per_algo_threads() const { return fn_set_threads_for_ != nullptr; }
    // E6 batch-switch: when a call's batch <= threshold, launch `algo` with
    // n_small threads (threshold==0 clears the switch for that algo).
    void set_threads_for_n(int algo, int threshold, int n_small) {
        if (!fn_set_threads_for_n_)
            throw std::runtime_error("set_threads_for_n: this .so predates the "
                "batch-regime overlay (rebuild to use the batch switch)");
        int rc = fn_set_threads_for_n_(algo, threshold, n_small);
        if (rc != 0) throw std::runtime_error(
            "grid_rbd_set_threads_for_n failed: rc=" + std::to_string(rc));
    }
    py::tuple get_batch_switch(int algo) const {
        if (!fn_get_batch_switch_)
            throw std::runtime_error("get_batch_switch: this .so predates the "
                "batch-regime overlay");
        int threshold = 0, n_small = -1;
        int rc = fn_get_batch_switch_(algo, &threshold, &n_small);
        if (rc != 0) throw std::runtime_error(
            "grid_rbd_get_batch_switch failed: rc=" + std::to_string(rc));
        return py::make_tuple(threshold, n_small);
    }
    bool has_batch_switch() const { return fn_set_threads_for_n_ != nullptr; }
    int threads_per_block() const { return fn_threads_per_block_(); }
    void set_threads_per_block(int n) {
        // Override the per-block thread count for all subsequent kernel
        // launches. Default is the per-algo autotuned launch_cfg<ALGO>::THREADS;
        // n==0 resets to that autotuned default, n>=1 forces one count for all
        // algos. The codegen no longer pins launch_bounds (cuBLASDx removed in
        // v2.0), so any n that fits per-block (≤1024 on current GPUs) is valid.
        if (n < 0) {
            throw std::invalid_argument(
                "set_threads_per_block: n must be >= 0 (0 resets to autotuned default), got " + std::to_string(n));
        }
        int rc = fn_set_threads_per_block_(n);
        if (rc != 0) {
            throw std::runtime_error(
                "grid_rbd_set_threads_per_block failed: rc=" + std::to_string(rc));
        }
    }

    // ─── inverse_dynamics ────────────────────────────────────────────────────────────────
    //
    // q, qd, qdd: (batch, num_joints) float32, C-contiguous. GRiD's kernels read
    //         q, qd AND qdd all at the NUM_JOINTS (== num_pos == nq) stride; for a
    //         FLOATING base the base 6-dof velocity lives in the leading slots and
    //         the +1 quaternion offset is a padded slot (so qd/qdd stay nq-wide,
    //         NOT nv-wide, on this surface).
    // returns c (generalized force): (batch, num_joints) float32 — likewise nq-wide.
    py::array_t<CT> inverse_dynamics(
        arr_t q,
        arr_t qd,
        py::object qdd_opt,
        CT gravity,
        py::object f_ext_opt)
    {
        int batch = check_inputs_2d(q, qd, /*last_dim=*/num_joints_);
        const CT* qdd_ptr = nullptr;
        if (!qdd_opt.is_none()) {
            auto qdd = qdd_opt.cast<
                arr_t>();
            check_array_2d(qdd, batch, num_joints_, "qdd");
            qdd_ptr = qdd.data();
        }
        arr_t fe_hold;
        const CT* fe_ptr = f_ext_ptr(f_ext_opt, fe_hold, batch);

        py::array_t<CT> out({batch, num_joints_});
        int rc = fn_inverse_dynamics_(q.data(), qd.data(), qdd_ptr,
                          out.mutable_data(), batch, gravity, fe_ptr);
        if (rc == 3) throw std::runtime_error(
            "inverse_dynamics not built into this robot .so — add 'inverse_dynamics' "
            "to algorithm_list in register_robot() and rebuild");
        if (rc != 0) {
            throw std::runtime_error("grid_rbd_inverse_dynamics failed: rc=" + std::to_string(rc));
        }
        return out;
    }

    // ─── inverse_dynamics_mujoco ─────────────────────────────────────────────
    //
    // MuJoCo output-convention ID (floating base only). q/qd/qdd are MuJoCo-native
    // and the returned tau is in the mjx frame — the convention transform is baked
    // into the kernel (MUJOCO_OUTPUT=true), so NO host pre/post-process is applied.
    // qdd is REQUIRED (the qdd=0 bias path can't represent mjx; use nonlinear_effects).
    // Raises if the .so doesn't export the symbol (fixed-base / older build).
    bool has_inverse_dynamics_mujoco() const { return fn_inverse_dynamics_mujoco_ != nullptr; }

    py::array_t<CT> inverse_dynamics_mujoco(
        arr_t q,
        arr_t qd,
        arr_t qdd,
        CT gravity,
        py::object f_ext_opt)
    {
        if (!fn_inverse_dynamics_mujoco_) {
            throw std::runtime_error(
                "inverse_dynamics_mujoco unavailable: this .so has no mjx ID kernel "
                "(only floating-base robots export grid_rbd_inverse_dynamics_mujoco)");
        }
        int batch = check_inputs_2d(q, qd, /*last_dim=*/num_joints_);
        check_array_2d(qdd, batch, num_joints_, "qdd");
        arr_t fe_hold;
        const CT* fe_ptr = f_ext_ptr(f_ext_opt, fe_hold, batch);

        py::array_t<CT> out({batch, num_joints_});
        int rc = fn_inverse_dynamics_mujoco_(q.data(), qd.data(), qdd.data(),
                          out.mutable_data(), batch, gravity, fe_ptr);
        if (rc != 0) {
            throw std::runtime_error("grid_rbd_inverse_dynamics_mujoco failed: rc=" + std::to_string(rc));
        }
        return out;
    }

    // ─── minv ────────────────────────────────────────────────────────────────
    py::array_t<CT> minv(
        arr_t q)
    {
        int batch = q.ndim() == 2 ? q.shape(0) : 0;
        if (q.ndim() != 2 || q.shape(1) != num_joints_) {
            throw std::invalid_argument(
                "minv: q must be (batch, " + std::to_string(num_joints_) + ")");
        }
        if (batch > max_batch_) {
            throw std::invalid_argument(
                "minv: batch=" + std::to_string(batch) + " > max_batch=" + std::to_string(max_batch_));
        }
        // Minv is nv x nv (tangent-space, pinocchio convention). For a FIXED base
        // nv == nq == num_joints_; for a FLOATING base nv = num_vel_ < num_joints_
        // (the kernel writes NUM_VEL*NUM_VEL, not NUM_JOINTS*NUM_JOINTS).
        py::array_t<CT> out({batch, num_vel_, num_vel_});
        int rc = fn_minv_(q.data(), out.mutable_data(), batch);
        if (rc == 3) throw std::runtime_error(
            "minv not built into this robot .so — add 'minv' to algorithm_list "
            "in register_robot() and rebuild");
        if (rc != 0) {
            throw std::runtime_error("grid_rbd_minv failed: rc=" + std::to_string(rc));
        }
        return out;
    }

    // ─── forward_dynamics ────────────────────────────────────────────────────
    py::array_t<CT> forward_dynamics(
        arr_t q,
        arr_t qd,
        arr_t u,
        CT gravity,
        py::object f_ext_opt)
    {
        int batch = check_inputs_2d(q, qd, /*last_dim=*/num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        arr_t fe_hold;
        const CT* fe_ptr = f_ext_ptr(f_ext_opt, fe_hold, batch);

        py::array_t<CT> out({batch, num_joints_});
        int rc = fn_fd_(q.data(), qd.data(), u.data(),
                        out.mutable_data(), batch, gravity, fe_ptr);
        if (rc == 3) throw std::runtime_error(
            "forward_dynamics not built into this robot .so — add 'forward_dynamics' "
            "to algorithm_list in register_robot() and rebuild");
        if (rc != 0) {
            throw std::runtime_error("grid_rbd_forward_dynamics failed: rc=" + std::to_string(rc));
        }
        return out;
    }

    // ─── aba ─────────────────────────────────────────────────────────────────
    py::array_t<CT> aba(
        arr_t q,
        arr_t qd,
        arr_t u,
        CT gravity,
        py::object f_ext_opt)
    {
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        arr_t fe_hold;
        const CT* fe_ptr = f_ext_ptr(f_ext_opt, fe_hold, batch);
        py::array_t<CT> out({batch, num_joints_});
        int rc = fn_aba_(q.data(), qd.data(), u.data(),
                         out.mutable_data(), batch, gravity, fe_ptr);
        if (rc == 3) throw std::runtime_error(
            "aba not built into this robot .so — add 'aba' to algorithm_list "
            "in register_robot() and rebuild");
        if (rc != 0) throw std::runtime_error("grid_rbd_aba failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── crba ────────────────────────────────────────────────────────────────
    py::array_t<CT> crba(
        arr_t q,
        CT gravity)
    {
        if (q.ndim() != 2 || q.shape(1) != num_joints_) {
            throw std::invalid_argument(
                "crba: q must be (batch, " + std::to_string(num_joints_) + ")");
        }
        int batch = (int)q.shape(0);
        if (batch > max_batch_) {
            throw std::invalid_argument(
                "crba: batch=" + std::to_string(batch) + " > max_batch=" + std::to_string(max_batch_));
        }
        // M is nv x nv (tangent-space, pinocchio convention). FIXED base: nv == nq
        // == num_joints_; FLOATING base: nv = num_vel_ < num_joints_ (the kernel
        // writes NUM_VEL*NUM_VEL).
        py::array_t<CT> out({batch, num_vel_, num_vel_});
        int rc = fn_crba_(q.data(), out.mutable_data(), batch, gravity);
        if (rc == 3) throw std::runtime_error(
            "crba not built into this robot .so — add 'crba' to algorithm_list "
            "in register_robot() and rebuild");
        if (rc != 0) throw std::runtime_error("grid_rbd_crba failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── crba_mujoco ─────────────────────────────────────────────────────────
    // MuJoCo-convention mass matrix M_mjx = G M_pin G^T (floating base only). q is
    // MuJoCo-native; the congruence is baked into the kernel — no host transform.
    bool has_crba_mujoco() const { return fn_crba_mujoco_ != nullptr; }

    py::array_t<CT> crba_mujoco(
        arr_t q,
        CT gravity)
    {
        if (!fn_crba_mujoco_) {
            throw std::runtime_error(
                "crba_mujoco unavailable: this .so has no mjx CRBA kernel "
                "(only floating-base robots export grid_rbd_crba_mujoco)");
        }
        if (q.ndim() != 2 || q.shape(1) != num_joints_) {
            throw std::invalid_argument(
                "crba_mujoco: q must be (batch, " + std::to_string(num_joints_) + ")");
        }
        int batch = (int)q.shape(0);
        if (batch > max_batch_) {
            throw std::invalid_argument(
                "crba_mujoco: batch=" + std::to_string(batch) + " > max_batch=" + std::to_string(max_batch_));
        }
        py::array_t<CT> out({batch, num_vel_, num_vel_});
        int rc = fn_crba_mujoco_(q.data(), out.mutable_data(), batch, gravity);
        if (rc != 0) throw std::runtime_error("grid_rbd_crba_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── mjx value kernels (floating base only; raw mjx in, mjx-frame out) ─────
    bool has_forward_dynamics_mujoco() const { return fn_fd_mujoco_ != nullptr; }
    py::array_t<CT> forward_dynamics_mujoco(
        arr_t q, arr_t qd, arr_t u, CT gravity, py::object f_ext_opt)
    {
        if (!fn_fd_mujoco_) throw std::runtime_error(
            "forward_dynamics_mujoco unavailable: floating-base .so only");
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        arr_t fe_hold;
        const CT* fe_ptr = f_ext_ptr(f_ext_opt, fe_hold, batch);
        py::array_t<CT> out({batch, num_joints_});
        int rc = fn_fd_mujoco_(q.data(), qd.data(), u.data(),
                               out.mutable_data(), batch, gravity, fe_ptr);
        if (rc != 0) throw std::runtime_error("grid_rbd_forward_dynamics_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    bool has_aba_mujoco() const { return fn_aba_mujoco_ != nullptr; }
    py::array_t<CT> aba_mujoco(
        arr_t q, arr_t qd, arr_t u, CT gravity, py::object f_ext_opt)
    {
        if (!fn_aba_mujoco_) throw std::runtime_error(
            "aba_mujoco unavailable: floating-base .so only");
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        arr_t fe_hold;
        const CT* fe_ptr = f_ext_ptr(f_ext_opt, fe_hold, batch);
        py::array_t<CT> out({batch, num_joints_});
        int rc = fn_aba_mujoco_(q.data(), qd.data(), u.data(),
                                out.mutable_data(), batch, gravity, fe_ptr);
        if (rc != 0) throw std::runtime_error("grid_rbd_aba_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    bool has_coriolis_matrix_mujoco() const { return fn_coriolis_matrix_mujoco_ != nullptr; }
    py::array_t<CT> coriolis_matrix_mujoco(arr_t q, arr_t qd, CT gravity)
    {
        if (!fn_coriolis_matrix_mujoco_) throw std::runtime_error(
            "coriolis_matrix_mujoco unavailable: floating-base .so only");
        int batch = check_inputs_2d(q, qd, num_joints_);
        py::array_t<CT> out({batch, num_vel_ * num_vel_});
        int rc = fn_coriolis_matrix_mujoco_(q.data(), qd.data(), out.mutable_data(), batch, gravity);
        if (rc != 0) throw std::runtime_error("grid_rbd_coriolis_matrix_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    bool has_frame_jacobian_mujoco() const { return fn_frame_jacobian_mujoco_ != nullptr; }
    py::array_t<CT> frame_jacobian_mujoco(arr_t q, int target_jid, int reference_frame)
    {
        if (!fn_frame_jacobian_mujoco_) throw std::runtime_error(
            "frame_jacobian_mujoco unavailable: floating-base .so with frame_jacobian only");
        int batch = check_q(q, "frame_jacobian_mujoco");
        py::array_t<CT> out({batch, 6 * num_vel_});
        int rc = fn_frame_jacobian_mujoco_(q.data(), out.mutable_data(), batch, target_jid, reference_frame);
        if (rc == 3) throw std::runtime_error("frame_jacobian not generated for this robot .so");
        if (rc != 0) throw std::runtime_error("grid_rbd_frame_jacobian_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    bool has_frame_jacobian_dot_mujoco() const { return fn_frame_jacobian_dot_mujoco_ != nullptr; }
    py::array_t<CT> frame_jacobian_dot_mujoco(arr_t q, arr_t qd, int target_jid, int reference_frame)
    {
        if (!fn_frame_jacobian_dot_mujoco_) throw std::runtime_error(
            "frame_jacobian_dot_mujoco unavailable: floating-base .so with frame_jacobian only");
        int batch = check_inputs_2d(q, qd, num_joints_);
        py::array_t<CT> out({batch, 6 * num_vel_});
        int rc = fn_frame_jacobian_dot_mujoco_(q.data(), qd.data(), out.mutable_data(), batch, target_jid, reference_frame);
        if (rc == 3) throw std::runtime_error("frame_jacobian_dot not generated for this robot .so");
        if (rc != 0) throw std::runtime_error("grid_rbd_frame_jacobian_dot_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    bool has_osc_inertia_mujoco() const { return fn_osc_inertia_mujoco_ != nullptr; }
    py::array_t<CT> osc_inertia_mujoco(arr_t q)
    {
        if (!fn_osc_inertia_mujoco_) throw std::runtime_error(
            "osc_inertia_mujoco unavailable: floating-base .so with frame_jacobian only");
        int batch = check_q(q, "osc_inertia_mujoco");
        py::array_t<CT> out({batch, 36});
        int rc = fn_osc_inertia_mujoco_(q.data(), out.mutable_data(), batch);
        if (rc == 3) throw std::runtime_error("osc_inertia not generated for this robot .so");
        if (rc != 0) throw std::runtime_error("grid_rbd_osc_inertia_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── minv_mujoco ─────────────────────────────────────────────────────────
    // MuJoCo-convention direct mass-matrix inverse (floating base only). q is
    // MuJoCo-native; the kernel reorders the quaternion + applies the congruence
    // and writes a FULL DENSE SYMMETRIC mjx Minv — no host symmetrize / transform.
    bool has_minv_mujoco() const { return fn_minv_mujoco_ != nullptr; }
    py::array_t<CT> minv_mujoco(arr_t q)
    {
        if (!fn_minv_mujoco_) throw std::runtime_error(
            "minv_mujoco unavailable: this .so has no mjx Minv kernel "
            "(only floating-base robots export grid_rbd_minv_mujoco)");
        if (q.ndim() != 2 || q.shape(1) != num_joints_) {
            throw std::invalid_argument(
                "minv_mujoco: q must be (batch, " + std::to_string(num_joints_) + ")");
        }
        int batch = (int)q.shape(0);
        if (batch > max_batch_) {
            throw std::invalid_argument(
                "minv_mujoco: batch=" + std::to_string(batch) + " > max_batch=" + std::to_string(max_batch_));
        }
        py::array_t<CT> out({batch, num_vel_, num_vel_});
        int rc = fn_minv_mujoco_(q.data(), out.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error("grid_rbd_minv_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── com_mujoco ──────────────────────────────────────────────────────────
    // MuJoCo-convention com(q) -> (batch, 3 + 3*NV): [p_com(3); J_com(3 x NV)].
    // p_com is invariant, J_com columns reframed by the kernel; q is mjx-native.
    bool has_com_mujoco() const { return fn_com_mujoco_ != nullptr; }
    py::array_t<CT> com_mujoco(arr_t q)
    {
        if (!fn_com_mujoco_) throw std::runtime_error(
            "com_mujoco unavailable: floating-base .so with com only "
            "(re-register with force_rebuild=True)");
        int batch = check_q(q, "com_mujoco");
        py::array_t<CT> out({batch, 3 + 3 * num_vel_});
        int rc = fn_com_mujoco_(q.data(), out.mutable_data(), batch);
        if (rc == 3) throw std::runtime_error("com not generated for this robot (reduced codegen profile)");  // defensive: the twin has no rc=3 stub
        if (rc != 0) throw std::runtime_error("grid_rbd_com_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── ccrba_mujoco ────────────────────────────────────────────────────────
    // MuJoCo-convention ccrba(q, qd) -> (batch, 6*NV + 6): [A(6 x NV); h(6)].
    // h is invariant, A columns reframed by the kernel; q/qd are mjx-native.
    bool has_ccrba_mujoco() const { return fn_ccrba_mujoco_ != nullptr; }
    py::array_t<CT> ccrba_mujoco(arr_t q, arr_t qd)
    {
        if (!fn_ccrba_mujoco_) throw std::runtime_error(
            "ccrba_mujoco unavailable: floating-base .so with ccrba only "
            "(re-register with force_rebuild=True)");
        int batch = check_inputs_2d(q, qd, num_joints_);
        py::array_t<CT> out({batch, 6 * num_vel_ + 6});
        int rc = fn_ccrba_mujoco_(q.data(), qd.data(), out.mutable_data(), batch);
        if (rc == 3) throw std::runtime_error("ccrba not generated for this robot (reduced codegen profile)");  // defensive: the twin has no rc=3 stub
        if (rc != 0) throw std::runtime_error("grid_rbd_ccrba_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── energy_mujoco ───────────────────────────────────────────────────────
    // MuJoCo-convention energy(q, qd, gravity) -> (batch, 3): [KE, PE, KE+PE].
    // The energies are frame-invariant; the kernel only converts mjx-native inputs.
    bool has_energy_mujoco() const { return fn_energy_mujoco_ != nullptr; }
    py::array_t<CT> energy_mujoco(arr_t q, arr_t qd, CT gravity)
    {
        if (!fn_energy_mujoco_) throw std::runtime_error(
            "energy_mujoco unavailable: floating-base .so with energy only "
            "(re-register with force_rebuild=True)");
        int batch = check_inputs_2d(q, qd, num_joints_);
        py::array_t<CT> out({batch, 3});
        int rc = fn_energy_mujoco_(q.data(), qd.data(), out.mutable_data(), batch, gravity);
        if (rc == 3) throw std::runtime_error("energy not generated for this robot (reduced codegen profile)");  // defensive: the twin has no rc=3 stub
        if (rc != 0) throw std::runtime_error("grid_rbd_energy_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── kinetic_energy_regressor_mujoco ─────────────────────────────────────
    // MuJoCo-convention y_KE -> (batch, 10*NUM_BODIES). Frame-invariant regressor;
    // the kernel only converts the mjx-native inputs (quat reorder + qd reframe).
    bool has_kinetic_energy_regressor_mujoco() const { return fn_kinetic_energy_regressor_mujoco_ != nullptr; }
    py::array_t<CT> kinetic_energy_regressor_mujoco(arr_t q, arr_t qd, CT gravity)
    {
        if (!fn_kinetic_energy_regressor_mujoco_) throw std::runtime_error(
            "kinetic_energy_regressor_mujoco unavailable: floating-base .so only "
            "(re-register with force_rebuild=True)");
        int batch = check_inputs_2d(q, qd, num_joints_);
        py::array_t<CT> out({batch, 10 * num_bodies_});
        int rc = fn_kinetic_energy_regressor_mujoco_(q.data(), qd.data(), out.mutable_data(), batch, gravity);
        if (rc != 0) throw std::runtime_error("grid_rbd_kinetic_energy_regressor_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── potential_energy_regressor_mujoco ───────────────────────────────────
    // MuJoCo-convention y_PE -> (batch, 10*NUM_BODIES). Frame-invariant regressor;
    // the kernel only converts the mjx-native q (quaternion reorder).
    bool has_potential_energy_regressor_mujoco() const { return fn_potential_energy_regressor_mujoco_ != nullptr; }
    py::array_t<CT> potential_energy_regressor_mujoco(arr_t q, CT gravity)
    {
        if (!fn_potential_energy_regressor_mujoco_) throw std::runtime_error(
            "potential_energy_regressor_mujoco unavailable: floating-base .so only "
            "(re-register with force_rebuild=True)");
        int batch = check_q(q, "potential_energy_regressor_mujoco");
        py::array_t<CT> out({batch, 10 * num_bodies_});
        int rc = fn_potential_energy_regressor_mujoco_(q.data(), out.mutable_data(), batch, gravity);
        if (rc != 0) throw std::runtime_error("grid_rbd_potential_energy_regressor_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── end_effector_pose ───────────────────────────────────────────────────
    py::array_t<CT> end_effector_pose(
        arr_t q)
    {
        if (q.ndim() != 2 || q.shape(1) != num_joints_) {
            throw std::invalid_argument(
                "end_effector_pose: q must be (batch, " + std::to_string(num_joints_) + ")");
        }
        int batch = (int)q.shape(0);
        if (batch > max_batch_) {
            throw std::invalid_argument(
                "end_effector_pose: batch=" + std::to_string(batch) + " > max_batch=" + std::to_string(max_batch_));
        }
        py::array_t<CT> out({batch, 6 * num_ees_});
        int rc = fn_ee_pose_(q.data(), out.mutable_data(), batch);
        if (rc == 3) throw std::runtime_error(
            "end_effector_pose not built into this robot .so — add 'end_effector_pose' "
            "to algorithm_list in register_robot() and rebuild");
        if (rc != 0) throw std::runtime_error("grid_rbd_end_effector_pose failed: rc=" + std::to_string(rc));
        return out;
    }

    // MuJoCo-convention end_effector_pose(q) -> (batch, 6*NUM_EES). Pose is
    // frame-INVARIANT; the native kernel reorders the mjx quaternion (latent-bug
    // path like osc_inertia). Floating-base only.
    bool has_end_effector_pose_mujoco() const { return fn_ee_pose_mujoco_ != nullptr; }
    py::array_t<CT> end_effector_pose_mujoco(arr_t q)
    {
        if (!fn_ee_pose_mujoco_) throw std::runtime_error(
            "end_effector_pose_mujoco unavailable: floating-base .so only");
        if (q.ndim() != 2 || q.shape(1) != num_joints_) {
            throw std::invalid_argument(
                "end_effector_pose_mujoco: q must be (batch, " + std::to_string(num_joints_) + ")");
        }
        int batch = (int)q.shape(0);
        if (batch > max_batch_) {
            throw std::invalid_argument(
                "end_effector_pose_mujoco: batch=" + std::to_string(batch) + " > max_batch=" + std::to_string(max_batch_));
        }
        py::array_t<CT> out({batch, 6 * num_ees_});
        int rc = fn_ee_pose_mujoco_(q.data(), out.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error("grid_rbd_end_effector_pose_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── fk_batched (large-batch FK, pos+quat) ───────────────────────────────
    // Input  q:     (batch, NUM_POS)
    // Output pose7: (batch, 7) = [tx,ty,tz, qw,qx,qy,qz]
    // use_warp selects the warp-cooperative per-sample inner.
    py::array_t<CT> fk_batched(
        arr_t q,
        bool use_warp)
    {
        if (!fn_fk_batched_) {
            throw std::runtime_error(
                "fk_batched not available in this robot .so (rebuild after adding "
                "the G2 batched-FK surface)");
        }
        if (q.ndim() != 2 || q.shape(1) != num_joints_) {
            throw std::invalid_argument(
                "fk_batched: q must be (batch, " + std::to_string(num_joints_) + ")");
        }
        int batch = (int)q.shape(0);
        if (batch > max_batch_) {
            throw std::invalid_argument(
                "fk_batched: batch=" + std::to_string(batch) + " > max_batch=" + std::to_string(max_batch_));
        }
        py::array_t<CT> out({batch, 7});
        int rc = fn_fk_batched_(q.data(), out.mutable_data(), batch, use_warp ? 1 : 0);
        if (rc == 3) throw std::runtime_error(
            "fk_batched: not supported for this robot (floating-base / mimic)");
        if (rc != 0) throw std::runtime_error("grid_rbd_fk_batched failed: rc=" + std::to_string(rc));
        return out;
    }

    py::array_t<CT> end_effector_pose_gradient(
        arr_t q)
    {
        if (q.ndim() != 2 || q.shape(1) != num_joints_) {
            throw std::invalid_argument(
                "end_effector_pose_gradient: q must be (batch, " + std::to_string(num_joints_) + ")");
        }
        int batch = (int)q.shape(0);
        if (batch > max_batch_) {
            throw std::invalid_argument(
                "end_effector_pose_gradient: batch=" + std::to_string(batch) + " > max_batch=" + std::to_string(max_batch_));
        }
        // d/dv tangent (pinocchio convention): (batch, 6*NUM_EES, NV)
        py::array_t<CT> out({batch, 6 * num_ees_, num_vel_});
        int rc = fn_ee_pose_grad_(q.data(), out.mutable_data(), batch);
        if (rc == 3) throw std::runtime_error(
            "end_effector_pose_gradient not built into this robot .so — add "
            "'end_effector_pose_gradient' to algorithm_list in register_robot() and rebuild");
        if (rc != 0) throw std::runtime_error("grid_rbd_end_effector_pose_gradient failed: rc=" + std::to_string(rc));
        return out;
    }

    // MuJoCo-convention end_effector_pose Jacobian (q) -> (batch, 6*NUM_EES, NV)
    // in the mjx frame (base-linear column reframe baked into the kernel).
    // Floating-base only.
    bool has_end_effector_pose_gradient_mujoco() const { return fn_ee_pose_grad_mujoco_ != nullptr; }
    py::array_t<CT> end_effector_pose_gradient_mujoco(arr_t q)
    {
        if (!fn_ee_pose_grad_mujoco_) throw std::runtime_error(
            "end_effector_pose_gradient_mujoco unavailable: floating-base .so only");
        if (q.ndim() != 2 || q.shape(1) != num_joints_) {
            throw std::invalid_argument(
                "end_effector_pose_gradient_mujoco: q must be (batch, " + std::to_string(num_joints_) + ")");
        }
        int batch = (int)q.shape(0);
        if (batch > max_batch_) {
            throw std::invalid_argument(
                "end_effector_pose_gradient_mujoco: batch=" + std::to_string(batch) + " > max_batch=" + std::to_string(max_batch_));
        }
        py::array_t<CT> out({batch, 6 * num_ees_, num_vel_});
        int rc = fn_ee_pose_grad_mujoco_(q.data(), out.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error("grid_rbd_end_effector_pose_gradient_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // MuJoCo-convention end_effector_pose Hessian (q) -> (batch, 6*NUM_EES, NV, NV).
    // Double column-reframe + symmetrized base-rotation frame term baked in-kernel.
    // Floating-base only.
    bool has_end_effector_pose_hessian_mujoco() const { return fn_ee_pose_hessian_mujoco_ != nullptr; }
    py::array_t<CT> end_effector_pose_hessian_mujoco(arr_t q)
    {
        if (!fn_ee_pose_hessian_mujoco_) throw std::runtime_error(
            "end_effector_pose_hessian_mujoco unavailable: floating-base .so only");
        if (q.ndim() != 2 || q.shape(1) != num_joints_) {
            throw std::invalid_argument(
                "end_effector_pose_hessian_mujoco: q must be (batch, " + std::to_string(num_joints_) + ")");
        }
        int batch = (int)q.shape(0);
        if (batch > max_batch_) {
            throw std::invalid_argument(
                "end_effector_pose_hessian_mujoco: batch=" + std::to_string(batch) + " > max_batch=" + std::to_string(max_batch_));
        }
        py::array_t<CT> out({batch, 6 * num_ees_, num_vel_, num_vel_});
        int rc = fn_ee_pose_hessian_mujoco_(q.data(), out.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error("grid_rbd_end_effector_pose_hessian_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── inverse_dynamics_gradient / forward_dynamics_gradient ───────────────────────────────────
    py::array_t<CT> inverse_dynamics_gradient(
        arr_t q,
        arr_t qd,
        py::object qdd_opt,
        CT gravity,
        py::object f_ext_opt)
    {
        int batch = check_inputs_2d(q, qd, num_joints_);
        const CT* qdd_ptr = nullptr;
        if (!qdd_opt.is_none()) {
            auto qdd = qdd_opt.cast<
                arr_t>();
            check_array_2d(qdd, batch, num_joints_, "qdd");
            qdd_ptr = qdd.data();
        }
        arr_t fe_hold;
        const CT* fe_ptr = f_ext_ptr(f_ext_opt, fe_hold, batch);
        // dc/d(q,qd) is nv x 2nv (tangent-space). FIXED base: nv == num_joints_;
        // FLOATING base: nv = num_vel_ (the kernel writes 2*NUM_VEL*NUM_VEL).
        py::array_t<CT> out({batch, num_vel_, 2 * num_vel_});
        int rc = fn_inverse_dynamics_gradient_(q.data(), qd.data(), qdd_ptr,
                               out.mutable_data(), batch, gravity, fe_ptr);
        if (rc == 3) throw std::runtime_error(
            "inverse_dynamics_gradient not built into this robot .so — add "
            "'inverse_dynamics_gradient' to algorithm_list in register_robot() and rebuild");
        if (rc != 0) throw std::runtime_error("grid_rbd_inverse_dynamics_gradient failed: rc=" + std::to_string(rc));
        return out;
    }

    bool has_inverse_dynamics_gradient_mujoco() const { return fn_inverse_dynamics_gradient_mujoco_ != nullptr; }
    py::array_t<CT> inverse_dynamics_gradient_mujoco(
        arr_t q, arr_t qd, arr_t qdd, CT gravity, py::object f_ext_opt)
    {
        if (!fn_inverse_dynamics_gradient_mujoco_) throw std::runtime_error(
            "inverse_dynamics_gradient_mujoco unavailable: floating-base .so only");
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(qdd, batch, num_joints_, "qdd");
        arr_t fe_hold;
        const CT* fe_ptr = f_ext_ptr(f_ext_opt, fe_hold, batch);
        py::array_t<CT> out({batch, num_vel_, 2 * num_vel_});
        int rc = fn_inverse_dynamics_gradient_mujoco_(q.data(), qd.data(), qdd.data(),
                               out.mutable_data(), batch, gravity, fe_ptr);
        if (rc != 0) throw std::runtime_error("grid_rbd_inverse_dynamics_gradient_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    py::array_t<CT> forward_dynamics_gradient(
        arr_t q,
        arr_t qd,
        arr_t u,
        CT gravity,
        py::object f_ext_opt)
    {
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        arr_t fe_hold;
        const CT* fe_ptr = f_ext_ptr(f_ext_opt, fe_hold, batch);
        // dqdd/d(q,qd) is nv x 2nv (tangent-space). FIXED base: nv == num_joints_;
        // FLOATING base: nv = num_vel_ (the kernel writes 2*NUM_VEL*NUM_VEL).
        py::array_t<CT> out({batch, num_vel_, 2 * num_vel_});
        int rc = fn_fd_grad_(q.data(), qd.data(), u.data(),
                             out.mutable_data(), batch, gravity, fe_ptr);
        if (rc == 3) throw std::runtime_error(
            "forward_dynamics_gradient not built into this robot .so — add "
            "'forward_dynamics_gradient' to algorithm_list in register_robot() and rebuild");
        if (rc != 0) throw std::runtime_error("grid_rbd_forward_dynamics_gradient failed: rc=" + std::to_string(rc));
        return out;
    }

    bool has_forward_dynamics_gradient_mujoco() const { return fn_fd_grad_mujoco_ != nullptr; }
    py::array_t<CT> forward_dynamics_gradient_mujoco(
        arr_t q, arr_t qd, arr_t u, CT gravity, py::object f_ext_opt)
    {
        if (!fn_fd_grad_mujoco_) throw std::runtime_error(
            "forward_dynamics_gradient_mujoco unavailable: floating-base .so only");
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        arr_t fe_hold;
        const CT* fe_ptr = f_ext_ptr(f_ext_opt, fe_hold, batch);
        py::array_t<CT> out({batch, num_vel_, 2 * num_vel_});
        int rc = fn_fd_grad_mujoco_(q.data(), qd.data(), u.data(),
                                    out.mutable_data(), batch, gravity, fe_ptr);
        if (rc != 0) throw std::runtime_error("grid_rbd_forward_dynamics_gradient_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── end_effector_pose_hessian ───────────────────────────────────────────
    py::array_t<CT> end_effector_pose_hessian(
        arr_t q)
    {
        if (q.ndim() != 2 || q.shape(1) != num_joints_) {
            throw std::invalid_argument(
                "end_effector_pose_hessian: q must be (batch, " + std::to_string(num_joints_) + ")");
        }
        int batch = (int)q.shape(0);
        if (batch > max_batch_) {
            throw std::invalid_argument(
                "end_effector_pose_hessian: batch=" + std::to_string(batch) + " > max_batch=" + std::to_string(max_batch_));
        }
        py::array_t<CT> out({batch, 6 * num_ees_, num_vel_, num_vel_});
        int rc = fn_ee_pose_hessian_(q.data(), out.mutable_data(), batch);
        if (rc == 3) throw std::runtime_error(
            "end_effector_pose_hessian not built into this robot .so — add "
            "'end_effector_pose_hessian' to algorithm_list in register_robot() and rebuild");
        if (rc != 0) throw std::runtime_error("grid_rbd_end_effector_pose_hessian failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── idsva_so / fdsva_so (raw second-order tensor surface) ───────────────
    // Returns shape (B, SECOND_ORDER_TENSOR_SIZE) — flat, 4 * NV^3 floats per
    // timestep. The Python side slices into the four NV^3 tensors.
    py::array_t<CT> idsva_so(
        arr_t q,
        arr_t qd,
        py::object qdd_opt,
        int second_order_tensor_size,
        CT gravity)
    {
        int batch = check_inputs_2d(q, qd, num_joints_);
        const CT* qdd_ptr = nullptr;
        if (!qdd_opt.is_none()) {
            auto qdd = qdd_opt.cast<
                arr_t>();
            check_array_2d(qdd, batch, num_joints_, "qdd");
            qdd_ptr = qdd.data();
        }
        py::array_t<CT> out({batch, second_order_tensor_size});
        int rc = fn_idsva_so_(q.data(), qd.data(), qdd_ptr,
                              out.mutable_data(), batch, gravity);
        if (rc == 3) throw std::runtime_error(
            "idsva_so not built into this robot .so — add 'idsva_so_body_frame' "
            "to algorithm_list in register_robot() and rebuild");
        if (rc != 0) throw std::runtime_error("grid_rbd_idsva_so failed: rc=" + std::to_string(rc));
        return out;
    }

    // MuJoCo-convention idsva_so -> (B, SECOND_ORDER_TENSOR_SIZE). Floating-base only.
    bool has_idsva_so_mujoco() const { return fn_idsva_so_mujoco_ != nullptr; }
    py::array_t<CT> idsva_so_mujoco(
        arr_t q,
        arr_t qd,
        py::object qdd_opt,
        int second_order_tensor_size,
        CT gravity)
    {
        if (!fn_idsva_so_mujoco_) throw std::runtime_error(
            "idsva_so_mujoco unavailable: floating-base .so only");
        int batch = check_inputs_2d(q, qd, num_joints_);
        const CT* qdd_ptr = nullptr;
        if (!qdd_opt.is_none()) {
            auto qdd = qdd_opt.cast<arr_t>();
            check_array_2d(qdd, batch, num_joints_, "qdd");
            qdd_ptr = qdd.data();
        }
        py::array_t<CT> out({batch, second_order_tensor_size});
        int rc = fn_idsva_so_mujoco_(q.data(), qd.data(), qdd_ptr,
                                     out.mutable_data(), batch, gravity);
        if (rc != 0) throw std::runtime_error("grid_rbd_idsva_so_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── inverse_dynamics_regressor: Y (NV x 10*NUM_BODIES), tau = Y . pi ─────
    // Returns shape (B, NV*10*NUM_BODIES) flat, row-major NV x (10*NUM_BODIES) per
    // timestep. The Python side reshapes into (B, NV, 10*NUM_BODIES).
    bool has_inverse_dynamics_regressor() const { return fn_id_regressor_ != nullptr; }
    py::array_t<CT> inverse_dynamics_regressor(
        arr_t q,
        arr_t qd,
        py::object qdd_opt,
        CT gravity)
    {
        if (!fn_id_regressor_) throw std::runtime_error(
            "inverse_dynamics_regressor not available in this .so (re-register with force_rebuild=True)");
        int batch = check_inputs_2d(q, qd, num_joints_);
        const CT* qdd_ptr = nullptr;
        if (!qdd_opt.is_none()) {
            auto qdd = qdd_opt.cast<arr_t>();
            check_array_2d(qdd, batch, num_joints_, "qdd");
            qdd_ptr = qdd.data();
        }
        py::array_t<CT> out({batch, num_vel_ * 10 * num_bodies_});
        int rc = fn_id_regressor_(q.data(), qd.data(), qdd_ptr,
                                  out.mutable_data(), batch, gravity);
        if (rc == 3) throw std::runtime_error(
            "inverse_dynamics_regressor not built into this robot .so — add "
            "'inverse_dynamics_regressor' to algorithm_list in register_robot() and rebuild");
        if (rc != 0) throw std::runtime_error("grid_rbd_inverse_dynamics_regressor failed: rc=" + std::to_string(rc));
        return out;
    }

    // MuJoCo-convention inverse_dynamics_regressor. Floating-base only. The base-linear
    // ROWS (0:3) rotate by R in-kernel (the rows are tangent-indexed generalized forces).
    bool has_inverse_dynamics_regressor_mujoco() const { return fn_id_regressor_mujoco_ != nullptr; }
    py::array_t<CT> inverse_dynamics_regressor_mujoco(
        arr_t q,
        arr_t qd,
        py::object qdd_opt,
        CT gravity)
    {
        if (!fn_id_regressor_mujoco_) throw std::runtime_error(
            "inverse_dynamics_regressor_mujoco unavailable: floating-base .so only");
        int batch = check_inputs_2d(q, qd, num_joints_);
        const CT* qdd_ptr = nullptr;
        if (!qdd_opt.is_none()) {
            auto qdd = qdd_opt.cast<arr_t>();
            check_array_2d(qdd, batch, num_joints_, "qdd");
            qdd_ptr = qdd.data();
        }
        py::array_t<CT> out({batch, num_vel_ * 10 * num_bodies_});
        int rc = fn_id_regressor_mujoco_(q.data(), qd.data(), qdd_ptr,
                                         out.mutable_data(), batch, gravity);
        if (rc != 0) throw std::runtime_error("grid_rbd_inverse_dynamics_regressor_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    py::array_t<CT> fdsva_so(
        arr_t q,
        arr_t qd,
        arr_t u,
        int second_order_tensor_size,
        CT gravity)
    {
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        py::array_t<CT> out({batch, second_order_tensor_size});
        int rc = fn_fdsva_so_(q.data(), qd.data(), u.data(),
                              out.mutable_data(), batch, gravity);
        if (rc == 3) throw std::runtime_error(
            "fdsva_so not built into this robot .so — add 'fdsva_so' "
            "to algorithm_list in register_robot() and rebuild");
        if (rc != 0) throw std::runtime_error("grid_rbd_fdsva_so failed: rc=" + std::to_string(rc));
        return out;
    }

    // MuJoCo-convention fdsva_so -> (B, SECOND_ORDER_TENSOR_SIZE). Floating-base only.
    bool has_fdsva_so_mujoco() const { return fn_fdsva_so_mujoco_ != nullptr; }
    py::array_t<CT> fdsva_so_mujoco(
        arr_t q,
        arr_t qd,
        arr_t u,
        int second_order_tensor_size,
        CT gravity)
    {
        if (!fn_fdsva_so_mujoco_) throw std::runtime_error(
            "fdsva_so_mujoco unavailable: floating-base .so only");
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        py::array_t<CT> out({batch, second_order_tensor_size});
        int rc = fn_fdsva_so_mujoco_(q.data(), qd.data(), u.data(),
                                     out.mutable_data(), batch, gravity);
        if (rc != 0) throw std::runtime_error("grid_rbd_fdsva_so_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // integrator(q, qd, u, dt, it) -> x_kp1 (batch, NUM_POS + NUM_VEL).
    // gravity is the signed gravitational acceleration (default -9.81) (baked in the wrapper).
    py::array_t<CT> integrator(
        arr_t q,
        arr_t qd,
        arr_t u,
        CT dt, int it, CT gravity)
    {
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        py::array_t<CT> out({batch, num_joints_ + num_vel_});
        int rc = fn_integrator_(q.data(), qd.data(), u.data(),
                                out.mutable_data(), batch, gravity, dt, it);
        // rc=3 here = integrator not built into this .so (subset profile). The
        // Python surface validates integrator_type into a 0-4 code before this
        // call, so the dispatch-default rc=3 (unknown it) is unreachable.
        if (rc == 3) throw std::runtime_error(
            "integrator not built into this robot .so — add 'integrator' "
            "to algorithm_list in register_robot() and rebuild");
        if (rc != 0) throw std::runtime_error("grid_rbd_integrator failed: rc=" + std::to_string(rc));
        return out;
    }

    bool has_integrator_mujoco() const { return fn_integrator_mujoco_ != nullptr; }
    py::array_t<CT> integrator_mujoco(
        arr_t q, arr_t qd, arr_t u, CT dt, int it, CT gravity)
    {
        if (!fn_integrator_mujoco_) throw std::runtime_error(
            "integrator_mujoco unavailable: floating-base .so only");
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        py::array_t<CT> out({batch, num_joints_ + num_vel_});
        int rc = fn_integrator_mujoco_(q.data(), qd.data(), u.data(),
                                       out.mutable_data(), batch, gravity, dt, it);
        if (rc == 3) throw std::runtime_error("integrator_mujoco: unsupported integrator_type for this build");
        if (rc != 0) throw std::runtime_error("grid_rbd_integrator_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // integrator_gradient(q, qd, u, dt, it) -> flat dAB (batch, 2*NV*3*NV),
    // column-major per timestep ([d/dq | d/dqd | d/du]); reshaped Python-side.
    py::array_t<CT> integrator_gradient(
        arr_t q,
        arr_t qd,
        arr_t u,
        CT dt, int it, CT gravity)
    {
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        py::array_t<CT> out({batch, 2 * num_vel_ * 3 * num_vel_});
        int rc = fn_integrator_grad_(q.data(), qd.data(), u.data(),
                                     out.mutable_data(), batch, gravity, dt, it);
        if (rc == 3) throw std::runtime_error(
            "integrator_gradient not built into this robot .so — add 'integrator_gradient' "
            "to algorithm_list in register_robot() and rebuild");
        if (rc != 0) throw std::runtime_error("grid_rbd_integrator_gradient failed: rc=" + std::to_string(rc));
        return out;
    }

    // MuJoCo-convention integrator_gradient -> (batch, 2*NV*3*NV). Floating-base only;
    // EULER/SI-EULER only (rc=3 otherwise).
    bool has_integrator_gradient_mujoco() const { return fn_integrator_grad_mujoco_ != nullptr; }
    py::array_t<CT> integrator_gradient_mujoco(
        arr_t q,
        arr_t qd,
        arr_t u,
        CT dt, int it, CT gravity)
    {
        if (!fn_integrator_grad_mujoco_) throw std::runtime_error(
            "integrator_gradient_mujoco unavailable: floating-base .so only");
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        py::array_t<CT> out({batch, 2 * num_vel_ * 3 * num_vel_});
        int rc = fn_integrator_grad_mujoco_(q.data(), qd.data(), u.data(),
                                            out.mutable_data(), batch, gravity, dt, it);
        if (rc == 3) throw std::runtime_error("integrator_gradient_mujoco: only EULER/SI-EULER supported");
        if (rc != 0) throw std::runtime_error("grid_rbd_integrator_gradient_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── grid_plant surface (G1 binding layer) ───────────────────────────────
    //
    // Each returns a tuple (value, grad, hess[/hess_diag]). value is (batch,);
    // grad/hess shapes depend on the cost. The plant kernels are emitted in the
    // grid_plant namespace; symbols are optional (raise if the .so lacks them).

    void require_plant(void* fn, const char* name) const {
        if (!fn) throw std::runtime_error(
            std::string("this robot .so does not export ") + name +
            " (grid_plant surface not generated for it). Re-register with a "
            "build that includes the plant namespace.");
    }

    // quadratic cost (state or input). var/des/w are (batch, N).
    std::tuple<py::array_t<CT>, py::array_t<CT>, py::array_t<CT>>
    plant_quadratic_cost(fn_plant_cost_t fn, const char* name,
        arr_t var,
        arr_t des,
        arr_t w,
        int N)
    {
        require_plant((void*)fn, name);
        if (var.ndim() != 2 || var.shape(1) != N)
            throw std::invalid_argument(std::string(name) + ": var must be (batch, " + std::to_string(N) + ")");
        int batch = (int)var.shape(0);
        if (batch > max_batch_) throw std::invalid_argument(std::string(name) + ": batch > max_batch");
        check_array_2d(des, batch, N, "des");
        check_array_2d(w, batch, N, "weight");
        py::array_t<CT> out({batch});
        py::array_t<CT> grad({batch, N});
        py::array_t<CT> hess({batch, N, N});
        int rc = fn(var.data(), des.data(), w.data(),
                    out.mutable_data(), grad.mutable_data(), hess.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error(std::string(name) + " failed: rc=" + std::to_string(rc));
        return {out, grad, hess};
    }

    std::tuple<py::array_t<CT>, py::array_t<CT>, py::array_t<CT>>
    quadratic_state_cost(
        arr_t x,
        arr_t x_des,
        arr_t Q)
    { return plant_quadratic_cost(fn_plant_state_cost_, "quadratic_state_cost", x, x_des, Q, num_joints_ + num_vel_); }

    std::tuple<py::array_t<CT>, py::array_t<CT>, py::array_t<CT>>
    quadratic_input_cost(
        arr_t u,
        arr_t u_des,
        arr_t R)
    { return plant_quadratic_cost(fn_plant_input_cost_, "quadratic_input_cost", u, u_des, R, num_vel_); }

    // MuJoCo-convention quadratic_state_cost. Floating-base only. x = [q; qd] is
    // mjx-native; the kernel input-converts the qd base-linear block and reframes
    // the qd-block grad/hess. The value is convention-DEPENDENT.
    bool has_quadratic_state_cost_mujoco() const { return fn_plant_state_cost_mujoco_ != nullptr; }
    std::tuple<py::array_t<CT>, py::array_t<CT>, py::array_t<CT>>
    quadratic_state_cost_mujoco(
        arr_t x,
        arr_t x_des,
        arr_t Q)
    { return plant_quadratic_cost(fn_plant_state_cost_mujoco_, "quadratic_state_cost_mujoco", x, x_des, Q, num_joints_ + num_vel_); }

    // barrier (position/velocity/torque). var/lower/upper are (batch, N).
    std::tuple<py::array_t<CT>, py::array_t<CT>, py::array_t<CT>>
    plant_barrier(fn_plant_barrier_t fn, const char* name,
        arr_t var,
        arr_t lower,
        arr_t upper,
        float mu, int N)
    {
        require_plant((void*)fn, name);
        if (var.ndim() != 2 || var.shape(1) != N)
            throw std::invalid_argument(std::string(name) + ": var must be (batch, " + std::to_string(N) + ")");
        int batch = (int)var.shape(0);
        if (batch > max_batch_) throw std::invalid_argument(std::string(name) + ": batch > max_batch");
        check_array_2d(lower, batch, N, "lower");
        check_array_2d(upper, batch, N, "upper");
        py::array_t<CT> out({batch});
        py::array_t<CT> grad({batch, N});
        py::array_t<CT> hess_diag({batch, N});
        int rc = fn(var.data(), lower.data(), upper.data(), mu,
                    out.mutable_data(), grad.mutable_data(), hess_diag.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error(std::string(name) + " failed: rc=" + std::to_string(rc));
        return {out, grad, hess_diag};
    }

    std::tuple<py::array_t<CT>, py::array_t<CT>, py::array_t<CT>>
    joint_position_barrier(
        arr_t var,
        arr_t lower,
        arr_t upper, float mu)
    { return plant_barrier(fn_plant_pos_barrier_, "joint_position_barrier", var, lower, upper, mu, num_joints_); }

    std::tuple<py::array_t<CT>, py::array_t<CT>, py::array_t<CT>>
    joint_velocity_barrier(
        arr_t var,
        arr_t lower,
        arr_t upper, float mu)
    { return plant_barrier(fn_plant_vel_barrier_, "joint_velocity_barrier", var, lower, upper, mu, num_vel_); }

    std::tuple<py::array_t<CT>, py::array_t<CT>, py::array_t<CT>>
    joint_torque_barrier(
        arr_t var,
        arr_t lower,
        arr_t upper, float mu)
    { return plant_barrier(fn_plant_tor_barrier_, "joint_torque_barrier", var, lower, upper, mu, num_vel_); }

    // plant_step: x (batch, NX), u (batch, NV) -> x_kp1 (batch, NX).
    py::array_t<CT> plant_step(
        arr_t x,
        arr_t u,
        CT dt, int it, CT gravity)
    {
        require_plant((void*)fn_plant_step_, "plant_step");
        int nx = num_joints_ + num_vel_;
        if (x.ndim() != 2 || x.shape(1) != nx)
            throw std::invalid_argument("plant_step: x must be (batch, " + std::to_string(nx) + ")");
        int batch = (int)x.shape(0);
        if (batch > max_batch_) throw std::invalid_argument("plant_step: batch > max_batch");
        check_array_2d(u, batch, num_vel_, "u");
        py::array_t<CT> out({batch, nx});
        int rc = fn_plant_step_(x.data(), u.data(), out.mutable_data(), batch, gravity, dt, it);
        if (rc != 0) throw std::runtime_error("plant_step failed: rc=" + std::to_string(rc));
        return out;
    }

    // MuJoCo-convention plant_step -> (batch, NX). Floating-base only; EULER/SI-EULER.
    bool has_plant_step_mujoco() const { return fn_plant_step_mujoco_ != nullptr; }
    py::array_t<CT> plant_step_mujoco(arr_t x, arr_t u, CT dt, int it, CT gravity)
    {
        require_plant((void*)fn_plant_step_mujoco_, "plant_step_mujoco");
        int nx = num_joints_ + num_vel_;
        if (x.ndim() != 2 || x.shape(1) != nx)
            throw std::invalid_argument("plant_step_mujoco: x must be (batch, " + std::to_string(nx) + ")");
        int batch = (int)x.shape(0);
        if (batch > max_batch_) throw std::invalid_argument("plant_step_mujoco: batch > max_batch");
        check_array_2d(u, batch, num_vel_, "u");
        py::array_t<CT> out({batch, nx});
        int rc = fn_plant_step_mujoco_(x.data(), u.data(), out.mutable_data(), batch, gravity, dt, it);
        if (rc == 3) throw std::runtime_error("plant_step_mujoco: only EULER/SI-EULER supported");
        if (rc != 0) throw std::runtime_error("plant_step_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // ee_pos_cost: q (batch, NQ), p_des (batch, 3), W (batch, 3)
    // -> (value (batch,), grad_x (batch, NX), hess_x (batch, NX, NX)).
    std::tuple<py::array_t<CT>, py::array_t<CT>, py::array_t<CT>>
    ee_pos_cost(
        arr_t q,
        arr_t p_des,
        arr_t W)
    {
        require_plant((void*)fn_plant_ee_cost_, "ee_pos_cost");
        int nx = num_joints_ + num_vel_;
        if (q.ndim() != 2 || q.shape(1) != num_joints_)
            throw std::invalid_argument("ee_pos_cost: q must be (batch, " + std::to_string(num_joints_) + ")");
        int batch = (int)q.shape(0);
        if (batch > max_batch_) throw std::invalid_argument("ee_pos_cost: batch > max_batch");
        check_array_2d(p_des, batch, 3, "p_des");
        check_array_2d(W, batch, 3, "W");
        py::array_t<CT> out({batch});
        py::array_t<CT> grad({batch, nx});
        py::array_t<CT> hess({batch, nx, nx});
        int rc = fn_plant_ee_cost_(q.data(), p_des.data(), W.data(),
                                   out.mutable_data(), grad.mutable_data(), hess.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error("ee_pos_cost failed: rc=" + std::to_string(rc));
        return {out, grad, hess};
    }

    // com_cost: q (batch, NQ), p_des (batch, 3), W (batch, 3)
    // -> (value (batch,), grad_x (batch, NX), hess_x (batch, NX, NX)). CoM-tracking.
    std::tuple<py::array_t<CT>, py::array_t<CT>, py::array_t<CT>>
    com_cost(
        arr_t q,
        arr_t p_des,
        arr_t W)
    {
        require_plant((void*)fn_plant_com_cost_, "com_cost");
        int nx = num_joints_ + num_vel_;
        if (q.ndim() != 2 || q.shape(1) != num_joints_)
            throw std::invalid_argument("com_cost: q must be (batch, " + std::to_string(num_joints_) + ")");
        int batch = (int)q.shape(0);
        if (batch > max_batch_) throw std::invalid_argument("com_cost: batch > max_batch");
        check_array_2d(p_des, batch, 3, "p_des");
        check_array_2d(W, batch, 3, "W");
        py::array_t<CT> out({batch});
        py::array_t<CT> grad({batch, nx});
        py::array_t<CT> hess({batch, nx, nx});
        int rc = fn_plant_com_cost_(q.data(), p_des.data(), W.data(),
                                    out.mutable_data(), grad.mutable_data(), hess.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error("com_cost failed: rc=" + std::to_string(rc));
        return {out, grad, hess};
    }

    // momentum_cost: q (batch, NQ), qd (batch, NV), h_des (batch, 6), W (batch, 6)
    // -> (value (batch,), grad_x (batch, NX), hess_x (batch, NX, NX)). Centroidal-momentum tracking.
    std::tuple<py::array_t<CT>, py::array_t<CT>, py::array_t<CT>>
    momentum_cost(
        arr_t q,
        arr_t qd,
        arr_t h_des,
        arr_t W)
    {
        require_plant((void*)fn_plant_mom_cost_, "momentum_cost");
        int nx = num_joints_ + num_vel_;
        if (q.ndim() != 2 || q.shape(1) != num_joints_)
            throw std::invalid_argument("momentum_cost: q must be (batch, " + std::to_string(num_joints_) + ")");
        int batch = (int)q.shape(0);
        if (batch > max_batch_) throw std::invalid_argument("momentum_cost: batch > max_batch");
        check_array_2d(qd, batch, num_vel_, "qd");
        check_array_2d(h_des, batch, 6, "h_des");
        check_array_2d(W, batch, 6, "W");
        py::array_t<CT> out({batch});
        py::array_t<CT> grad({batch, nx});
        py::array_t<CT> hess({batch, nx, nx});
        int rc = fn_plant_mom_cost_(q.data(), qd.data(), h_des.data(), W.data(),
                                    out.mutable_data(), grad.mutable_data(), hess.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error("momentum_cost failed: rc=" + std::to_string(rc));
        return {out, grad, hess};
    }

    // MuJoCo-convention tracking costs (floating base only): value invariant; the
    // active grad block reframes as a covector (G·) and the GN hess block by
    // congruence (G·G^T), all baked in-kernel (q/qd input-converted internally).
    bool has_ee_pos_cost_mujoco() const { return fn_plant_ee_cost_mujoco_ != nullptr; }
    std::tuple<py::array_t<CT>, py::array_t<CT>, py::array_t<CT>>
    ee_pos_cost_mujoco(arr_t q, arr_t p_des, arr_t W)
    {
        require_plant((void*)fn_plant_ee_cost_mujoco_, "ee_pos_cost_mujoco");
        int nx = num_joints_ + num_vel_;
        if (q.ndim() != 2 || q.shape(1) != num_joints_)
            throw std::invalid_argument("ee_pos_cost_mujoco: q must be (batch, " + std::to_string(num_joints_) + ")");
        int batch = (int)q.shape(0);
        if (batch > max_batch_) throw std::invalid_argument("ee_pos_cost_mujoco: batch > max_batch");
        check_array_2d(p_des, batch, 3, "p_des");
        check_array_2d(W, batch, 3, "W");
        py::array_t<CT> out({batch}); py::array_t<CT> grad({batch, nx}); py::array_t<CT> hess({batch, nx, nx});
        int rc = fn_plant_ee_cost_mujoco_(q.data(), p_des.data(), W.data(),
                                          out.mutable_data(), grad.mutable_data(), hess.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error("ee_pos_cost_mujoco failed: rc=" + std::to_string(rc));
        return {out, grad, hess};
    }

    bool has_com_cost_mujoco() const { return fn_plant_com_cost_mujoco_ != nullptr; }
    std::tuple<py::array_t<CT>, py::array_t<CT>, py::array_t<CT>>
    com_cost_mujoco(arr_t q, arr_t p_des, arr_t W)
    {
        require_plant((void*)fn_plant_com_cost_mujoco_, "com_cost_mujoco");
        int nx = num_joints_ + num_vel_;
        if (q.ndim() != 2 || q.shape(1) != num_joints_)
            throw std::invalid_argument("com_cost_mujoco: q must be (batch, " + std::to_string(num_joints_) + ")");
        int batch = (int)q.shape(0);
        if (batch > max_batch_) throw std::invalid_argument("com_cost_mujoco: batch > max_batch");
        check_array_2d(p_des, batch, 3, "p_des");
        check_array_2d(W, batch, 3, "W");
        py::array_t<CT> out({batch}); py::array_t<CT> grad({batch, nx}); py::array_t<CT> hess({batch, nx, nx});
        int rc = fn_plant_com_cost_mujoco_(q.data(), p_des.data(), W.data(),
                                           out.mutable_data(), grad.mutable_data(), hess.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error("com_cost_mujoco failed: rc=" + std::to_string(rc));
        return {out, grad, hess};
    }

    bool has_momentum_cost_mujoco() const { return fn_plant_mom_cost_mujoco_ != nullptr; }
    std::tuple<py::array_t<CT>, py::array_t<CT>, py::array_t<CT>>
    momentum_cost_mujoco(arr_t q, arr_t qd, arr_t h_des, arr_t W)
    {
        require_plant((void*)fn_plant_mom_cost_mujoco_, "momentum_cost_mujoco");
        int nx = num_joints_ + num_vel_;
        if (q.ndim() != 2 || q.shape(1) != num_joints_)
            throw std::invalid_argument("momentum_cost_mujoco: q must be (batch, " + std::to_string(num_joints_) + ")");
        int batch = (int)q.shape(0);
        if (batch > max_batch_) throw std::invalid_argument("momentum_cost_mujoco: batch > max_batch");
        check_array_2d(qd, batch, num_vel_, "qd");
        check_array_2d(h_des, batch, 6, "h_des");
        check_array_2d(W, batch, 6, "W");
        py::array_t<CT> out({batch}); py::array_t<CT> grad({batch, nx}); py::array_t<CT> hess({batch, nx, nx});
        int rc = fn_plant_mom_cost_mujoco_(q.data(), qd.data(), h_des.data(), W.data(),
                                           out.mutable_data(), grad.mutable_data(), hess.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error("momentum_cost_mujoco failed: rc=" + std::to_string(rc));
        return {out, grad, hess};
    }

    // plant_step_gradient: x (batch, NX), u (batch, NV) -> dAB (batch, 2*NV, 3*NV).
    py::array_t<CT> plant_step_gradient(
        arr_t x,
        arr_t u,
        CT dt, int it, CT gravity)
    {
        require_plant((void*)fn_plant_step_grad_, "plant_step_gradient");
        int nx = num_joints_ + num_vel_;
        int nv = num_vel_;
        if (x.ndim() != 2 || x.shape(1) != nx)
            throw std::invalid_argument("plant_step_gradient: x must be (batch, " + std::to_string(nx) + ")");
        int batch = (int)x.shape(0);
        if (batch > max_batch_) throw std::invalid_argument("plant_step_gradient: batch > max_batch");
        check_array_2d(u, batch, nv, "u");
        py::array_t<CT> out({batch, 2 * nv, 3 * nv});
        int rc = fn_plant_step_grad_(x.data(), u.data(), out.mutable_data(), batch, gravity, dt, it);
        if (rc != 0) throw std::runtime_error("plant_step_gradient failed: rc=" + std::to_string(rc));
        return out;
    }

    // MuJoCo-convention plant_step_gradient -> (batch, 2*NV, 3*NV). Floating; EULER/SI.
    bool has_plant_step_gradient_mujoco() const { return fn_plant_step_grad_mujoco_ != nullptr; }
    py::array_t<CT> plant_step_gradient_mujoco(arr_t x, arr_t u, CT dt, int it, CT gravity)
    {
        require_plant((void*)fn_plant_step_grad_mujoco_, "plant_step_gradient_mujoco");
        int nx = num_joints_ + num_vel_;
        int nv = num_vel_;
        if (x.ndim() != 2 || x.shape(1) != nx)
            throw std::invalid_argument("plant_step_gradient_mujoco: x must be (batch, " + std::to_string(nx) + ")");
        int batch = (int)x.shape(0);
        if (batch > max_batch_) throw std::invalid_argument("plant_step_gradient_mujoco: batch > max_batch");
        check_array_2d(u, batch, nv, "u");
        py::array_t<CT> out({batch, 2 * nv, 3 * nv});
        int rc = fn_plant_step_grad_mujoco_(x.data(), u.data(), out.mutable_data(), batch, gravity, dt, it);
        if (rc == 3) throw std::runtime_error("plant_step_gradient_mujoco: only EULER/SI-EULER supported");
        if (rc != 0) throw std::runtime_error("plant_step_gradient_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // plant_step_hessian: x (batch, NX), u (batch, NV) -> d2AB (batch, 2*NV, 3*NV*3*NV).
    // The C-ABI fills a row-major (2*NV x 3*NV x 3*NV) Hessian per timestep; this
    // surface returns it as (batch, 2*NV, 3*NV*3*NV) and the Python handle reshapes
    // the trailing 9*NV^2 into (3*NV, 3*NV). Only EULER / SI-EULER (rc=3 otherwise).
    py::array_t<CT> plant_step_hessian(
        arr_t x,
        arr_t u,
        CT dt, int it, CT gravity)
    {
        require_plant((void*)fn_plant_step_hess_, "plant_step_hessian");
        int nx = num_joints_ + num_vel_;
        int nv = num_vel_;
        if (x.ndim() != 2 || x.shape(1) != nx)
            throw std::invalid_argument("plant_step_hessian: x must be (batch, " + std::to_string(nx) + ")");
        int batch = (int)x.shape(0);
        if (batch > max_batch_) throw std::invalid_argument("plant_step_hessian: batch > max_batch");
        check_array_2d(u, batch, nv, "u");
        py::array_t<CT> out({batch, 2 * nv, 3 * nv * 3 * nv});
        int rc = fn_plant_step_hess_(x.data(), u.data(), out.mutable_data(), batch, gravity, dt, it);
        if (rc != 0) throw std::runtime_error("plant_step_hessian failed: rc=" + std::to_string(rc));
        return out;
    }

    // MuJoCo-convention plant_step_hessian -> (batch, 2*NV, 3*NV*3*NV). Floating; EULER/SI.
    bool has_plant_step_hessian_mujoco() const { return fn_plant_step_hess_mujoco_ != nullptr; }
    py::array_t<CT> plant_step_hessian_mujoco(arr_t x, arr_t u, CT dt, int it, CT gravity)
    {
        require_plant((void*)fn_plant_step_hess_mujoco_, "plant_step_hessian_mujoco");
        int nx = num_joints_ + num_vel_;
        int nv = num_vel_;
        if (x.ndim() != 2 || x.shape(1) != nx)
            throw std::invalid_argument("plant_step_hessian_mujoco: x must be (batch, " + std::to_string(nx) + ")");
        int batch = (int)x.shape(0);
        if (batch > max_batch_) throw std::invalid_argument("plant_step_hessian_mujoco: batch > max_batch");
        check_array_2d(u, batch, nv, "u");
        py::array_t<CT> out({batch, 2 * nv, 3 * nv * 3 * nv});
        int rc = fn_plant_step_hess_mujoco_(x.data(), u.data(), out.mutable_data(), batch, gravity, dt, it);
        if (rc == 3) throw std::runtime_error("plant_step_hessian_mujoco: only EULER/SI-EULER supported");
        if (rc != 0) throw std::runtime_error("plant_step_hessian_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── centroidal / energy / general-frame kinematics (F2) ─────────────────
    //
    // Each takes q (or q,qd) of shape (batch, NUM_JOINTS) and returns the flat
    // per-timestep gridData output buffer (the Python handle reshapes). The
    // frame_jacobian family is opt-in codegen; its C-ABI symbol returns rc=3 if
    // the family wasn't generated for this robot's .so.

    int check_q(const py::array_t<CT>& q, const char* name) const {
        if (q.ndim() != 2 || q.shape(1) != num_joints_)
            throw std::invalid_argument(
                std::string(name) + ": q must be (batch, " + std::to_string(num_joints_) + ")");
        int batch = (int)q.shape(0);
        if (batch > max_batch_)
            throw std::invalid_argument(std::string(name) + ": batch > max_batch");
        return batch;
    }

    // com(q) -> (batch, 3 + 3*NUM_VEL): [p_com(3); J_com(3 x NV, col-major)].
    py::array_t<CT> com(
        arr_t q)
    {
        if (!fn_com_) throw std::runtime_error("com not available in this .so (re-register with force_rebuild=True)");
        int batch = check_q(q, "com");
        py::array_t<CT> out({batch, 3 + 3 * num_vel_});
        int rc = fn_com_(q.data(), out.mutable_data(), batch);
        if (rc == 3) throw std::runtime_error(
            "com not available for this robot: it is not generated for mimic "
            "robots (the per-body Jacobian fold is not yet mimic-reduced)");
        if (rc != 0) throw std::runtime_error("grid_rbd_com failed: rc=" + std::to_string(rc));
        return out;
    }

    // ccrba(q, qd) -> (batch, 6*NUM_VEL + 6): [A(6 x NV, col-major); h(6)].
    py::array_t<CT> ccrba(
        arr_t q,
        arr_t qd)
    {
        if (!fn_ccrba_) throw std::runtime_error("ccrba not available in this .so (re-register with force_rebuild=True)");
        int batch = check_inputs_2d(q, qd, num_joints_);
        py::array_t<CT> out({batch, 6 * num_vel_ + 6});
        int rc = fn_ccrba_(q.data(), qd.data(), out.mutable_data(), batch);
        if (rc == 3) throw std::runtime_error(
            "ccrba not available for this robot: it is not generated for mimic "
            "robots (the per-body Jacobian fold is not yet mimic-reduced)");
        if (rc != 0) throw std::runtime_error("grid_rbd_ccrba failed: rc=" + std::to_string(rc));
        return out;
    }

    // energy(q, qd, gravity) -> (batch, 3): [KE, PE, KE+PE].
    py::array_t<CT> energy(
        arr_t q,
        arr_t qd,
        CT gravity)
    {
        if (!fn_energy_) throw std::runtime_error("energy not available in this .so (re-register with force_rebuild=True)");
        int batch = check_inputs_2d(q, qd, num_joints_);
        py::array_t<CT> out({batch, 3});
        int rc = fn_energy_(q.data(), qd.data(), out.mutable_data(), batch, gravity);
        if (rc == 3) throw std::runtime_error(
            "energy not available for this robot: it is not generated for mimic "
            "robots (the per-body Jacobian fold is not yet mimic-reduced)");
        if (rc != 0) throw std::runtime_error("grid_rbd_energy failed: rc=" + std::to_string(rc));
        return out;
    }

    // generalized_gravity(q, gravity) -> (batch, NUM_VEL): g(q) = RNEA(q,0,0).
    py::array_t<CT> generalized_gravity(
        arr_t q,
        CT gravity)
    {
        if (!fn_generalized_gravity_) throw std::runtime_error("generalized_gravity not available in this .so (re-register with force_rebuild=True)");
        int batch = check_q(q, "generalized_gravity");
        py::array_t<CT> out({batch, num_vel_});
        int rc = fn_generalized_gravity_(q.data(), out.mutable_data(), batch, gravity);
        if (rc != 0) throw std::runtime_error("grid_rbd_generalized_gravity failed: rc=" + std::to_string(rc));
        return out;
    }

    bool has_generalized_gravity_mujoco() const { return fn_generalized_gravity_mujoco_ != nullptr; }
    py::array_t<CT> generalized_gravity_mujoco(
        arr_t q,
        CT gravity)
    {
        if (!fn_generalized_gravity_mujoco_) throw std::runtime_error(
            "generalized_gravity_mujoco unavailable: floating-base .so only");
        int batch = check_q(q, "generalized_gravity_mujoco");
        py::array_t<CT> out({batch, num_vel_});
        int rc = fn_generalized_gravity_mujoco_(q.data(), out.mutable_data(), batch, gravity);
        if (rc != 0) throw std::runtime_error("grid_rbd_generalized_gravity_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // nonlinear_effects(q, qd, gravity) -> (batch, NUM_VEL): c(q,qd) = RNEA(q,qd,0).
    py::array_t<CT> nonlinear_effects(
        arr_t q,
        arr_t qd,
        CT gravity)
    {
        if (!fn_nonlinear_effects_) throw std::runtime_error("nonlinear_effects not available in this .so (re-register with force_rebuild=True)");
        int batch = check_inputs_2d(q, qd, num_joints_);
        py::array_t<CT> out({batch, num_vel_});
        int rc = fn_nonlinear_effects_(q.data(), qd.data(), out.mutable_data(), batch, gravity);
        if (rc != 0) throw std::runtime_error("grid_rbd_nonlinear_effects failed: rc=" + std::to_string(rc));
        return out;
    }

    // MuJoCo-convention nonlinear_effects(q, qd, gravity) -> (batch, NUM_VEL): mjx
    // qfrc_bias. Floating-base .so only (the accel-couple is a floating-root effect).
    bool has_nonlinear_effects_mujoco() const { return fn_nonlinear_effects_mujoco_ != nullptr; }
    py::array_t<CT> nonlinear_effects_mujoco(
        arr_t q,
        arr_t qd,
        CT gravity)
    {
        if (!fn_nonlinear_effects_mujoco_) throw std::runtime_error(
            "nonlinear_effects_mujoco unavailable: floating-base .so only");
        int batch = check_inputs_2d(q, qd, num_joints_);
        py::array_t<CT> out({batch, num_vel_});
        int rc = fn_nonlinear_effects_mujoco_(q.data(), qd.data(), out.mutable_data(), batch, gravity);
        if (rc != 0) throw std::runtime_error("grid_rbd_nonlinear_effects_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // frame_jacobian(q) -> (batch, 6*NUM_VEL): leaf-EE frame Jacobian (col-major,
    // [linear;angular], LOCAL_WORLD_ALIGNED). Opt-in codegen: rc=3 if absent.
    py::array_t<CT> frame_jacobian(
        arr_t q,
        int target_jid, int reference_frame)
    {
        if (!fn_frame_jacobian_) throw std::runtime_error("frame_jacobian not available in this .so (frame_jacobian family not generated; re-register with force_rebuild=True)");
        int batch = check_q(q, "frame_jacobian");
        py::array_t<CT> out({batch, 6 * num_vel_});
        // target_jid < 0 / reference_frame < 0 => the C ABI uses the codegen
        // leaf-EE / LWA defaults baked into the host wrapper.
        int rc = fn_frame_jacobian_(q.data(), out.mutable_data(), batch, target_jid, reference_frame);
        if (rc == 3) throw std::runtime_error("frame_jacobian not generated for this robot .so");
        if (rc != 0) throw std::runtime_error("grid_rbd_frame_jacobian failed: rc=" + std::to_string(rc));
        return out;
    }

    // frame_jacobian_dot(q, qd, target_jid, reference_frame) -> (batch, 6*NUM_VEL).
    // Opt-in codegen: rc=3 if absent.
    py::array_t<CT> frame_jacobian_dot(
        arr_t q,
        arr_t qd,
        int target_jid, int reference_frame)
    {
        if (!fn_frame_jacobian_dot_) throw std::runtime_error("frame_jacobian_dot not available in this .so (frame_jacobian family not generated; re-register with force_rebuild=True)");
        int batch = check_inputs_2d(q, qd, num_joints_);
        py::array_t<CT> out({batch, 6 * num_vel_});
        int rc = fn_frame_jacobian_dot_(q.data(), qd.data(), out.mutable_data(), batch, target_jid, reference_frame);
        if (rc == 3) throw std::runtime_error("frame_jacobian_dot not generated for this robot .so");
        if (rc != 0) throw std::runtime_error("grid_rbd_frame_jacobian_dot failed: rc=" + std::to_string(rc));
        return out;
    }

    // osc_inertia(q) -> (batch, 36): 6x6 task inertia Lambda = (J Minv J^T)^-1
    // at the leaf-EE frame (LWA). Opt-in codegen: rc=3 if absent.
    py::array_t<CT> osc_inertia(
        arr_t q)
    {
        if (!fn_osc_inertia_) throw std::runtime_error("osc_inertia not available in this .so (frame_jacobian family not generated; re-register with force_rebuild=True)");
        int batch = check_q(q, "osc_inertia");
        py::array_t<CT> out({batch, 36});
        int rc = fn_osc_inertia_(q.data(), out.mutable_data(), batch);
        if (rc == 3) throw std::runtime_error("osc_inertia not generated for this robot .so");
        if (rc != 0) throw std::runtime_error("grid_rbd_osc_inertia failed: rc=" + std::to_string(rc));
        return out;
    }

    // end_effector_pose_runtime(q, target_jid, offset) -> (batch, 6) = [xyz; rpy]
    // of target_jid at a runtime offset point. target_jid<0 => leaf-EE default;
    // offset is a length-3 array or empty (=> frame origin). Single-target; the
    // Python list API loops it over a jid list. Opt-in codegen: rc=3 if absent.
    py::array_t<CT> end_effector_pose_runtime(
        arr_t q,
        int target_jid,
        arr_t offset)
    {
        if (!fn_ee_pose_runtime_) throw std::runtime_error("end_effector_pose_runtime not available in this .so (re-register with force_rebuild=True)");
        int batch = check_q(q, "end_effector_pose_runtime");
        const CT* off_ptr = nullptr;
        if (offset.size() == 16) off_ptr = offset.data();
        else if (offset.size() != 0) throw std::invalid_argument("end_effector_pose_runtime: offset must be length-16 (4x4 col-major) or empty");
        py::array_t<CT> out({batch, 6});
        int rc = fn_ee_pose_runtime_(q.data(), out.mutable_data(), batch, target_jid, off_ptr);
        if (rc == 3) throw std::runtime_error("end_effector_pose_runtime not generated for this robot .so");
        if (rc != 0) throw std::runtime_error("grid_rbd_end_effector_pose_runtime failed: rc=" + std::to_string(rc));
        return out;
    }

    // tool_fext(q, wrench, jid, rc) -> (batch, 6*NUM_BODIES) joint-local f_ext from a
    // world-aligned tool-tip wrench at runtime (body jid, offset rc). Feed to
    // inverse_dynamics(f_ext=...). Present only on an enable_tool .so.
    bool has_tool_fext() const { return fn_tool_fext_ != nullptr; }
    py::array_t<CT> tool_fext(arr_t q, arr_t wrench, int jid, arr_t rc)
    {
        if (!fn_tool_fext_) throw std::runtime_error("tool_fext not available in this .so (register with enable_tool=True, force_rebuild=True)");
        int batch = check_q(q, "tool_fext");
        if (wrench.size() != (py::ssize_t)6 * batch)
            throw std::invalid_argument("tool_fext: wrench must be (batch, 6)");
        if (rc.size() != 3) throw std::invalid_argument("tool_fext: rc must be length-3");
        py::array_t<CT> out({batch, 6 * num_bodies_});
        int rc0 = fn_tool_fext_(q.data(), wrench.data(), jid, rc.data(), out.mutable_data(), batch);
        if (rc0 == 3) throw std::runtime_error("tool_fext not generated for this robot .so");
        if (rc0 != 0) throw std::runtime_error("grid_rbd_tool_fext failed: rc=" + std::to_string(rc0));
        return out;
    }

    // end_effector_pose_gradient_runtime(q, target_jid, offset) -> (batch, 6*NUM_VEL)
    // col-major d[xyz; rpy]/dv of target_jid at a runtime offset point. Same
    // conventions as end_effector_pose_runtime. Opt-in codegen: rc=3 if absent.
    py::array_t<CT> end_effector_pose_gradient_runtime(
        arr_t q,
        int target_jid,
        arr_t offset)
    {
        if (!fn_ee_pose_grad_runtime_) throw std::runtime_error("end_effector_pose_gradient_runtime not available in this .so (re-register with force_rebuild=True)");
        int batch = check_q(q, "end_effector_pose_gradient_runtime");
        const CT* off_ptr = nullptr;
        if (offset.size() == 16) off_ptr = offset.data();
        else if (offset.size() != 0) throw std::invalid_argument("end_effector_pose_gradient_runtime: offset must be length-16 (4x4 col-major) or empty");
        py::array_t<CT> out({batch, 6 * num_vel_});
        int rc = fn_ee_pose_grad_runtime_(q.data(), out.mutable_data(), batch, target_jid, off_ptr);
        if (rc == 3) throw std::runtime_error("end_effector_pose_gradient_runtime not generated for this robot .so");
        if (rc != 0) throw std::runtime_error("grid_rbd_end_effector_pose_gradient_runtime failed: rc=" + std::to_string(rc));
        return out;
    }

    // MuJoCo-convention end_effector_pose_runtime -> (batch, 6). Floating-base only.
    // Pose value is frame-INVARIANT; only the input quat is reordered in-kernel.
    bool has_end_effector_pose_runtime_mujoco() const { return fn_ee_pose_runtime_mujoco_ != nullptr; }
    py::array_t<CT> end_effector_pose_runtime_mujoco(
        arr_t q, int target_jid, arr_t offset)
    {
        if (!fn_ee_pose_runtime_mujoco_) throw std::runtime_error(
            "end_effector_pose_runtime_mujoco unavailable: floating-base .so only");
        int batch = check_q(q, "end_effector_pose_runtime_mujoco");
        const CT* off_ptr = nullptr;
        if (offset.size() == 16) off_ptr = offset.data();
        else if (offset.size() != 0) throw std::invalid_argument("end_effector_pose_runtime_mujoco: offset must be length-16 (4x4 col-major) or empty");
        py::array_t<CT> out({batch, 6});
        int rc = fn_ee_pose_runtime_mujoco_(q.data(), out.mutable_data(), batch, target_jid, off_ptr);
        if (rc == 3) throw std::runtime_error("end_effector_pose_runtime_mujoco not generated for this robot .so");
        if (rc != 0) throw std::runtime_error("grid_rbd_end_effector_pose_runtime_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // MuJoCo-convention end_effector_pose_gradient_runtime -> (batch, 6*NUM_VEL)
    // col-major. Floating-base only; base-linear columns reframe by R^T in-kernel.
    bool has_end_effector_pose_gradient_runtime_mujoco() const { return fn_ee_pose_grad_runtime_mujoco_ != nullptr; }
    py::array_t<CT> end_effector_pose_gradient_runtime_mujoco(
        arr_t q, int target_jid, arr_t offset)
    {
        if (!fn_ee_pose_grad_runtime_mujoco_) throw std::runtime_error(
            "end_effector_pose_gradient_runtime_mujoco unavailable: floating-base .so only");
        int batch = check_q(q, "end_effector_pose_gradient_runtime_mujoco");
        const CT* off_ptr = nullptr;
        if (offset.size() == 16) off_ptr = offset.data();
        else if (offset.size() != 0) throw std::invalid_argument("end_effector_pose_gradient_runtime_mujoco: offset must be length-16 (4x4 col-major) or empty");
        py::array_t<CT> out({batch, 6 * num_vel_});
        int rc = fn_ee_pose_grad_runtime_mujoco_(q.data(), out.mutable_data(), batch, target_jid, off_ptr);
        if (rc == 3) throw std::runtime_error("end_effector_pose_gradient_runtime_mujoco not generated for this robot .so");
        if (rc != 0) throw std::runtime_error("grid_rbd_end_effector_pose_gradient_runtime_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── PS5 value ops (coriolis / energy regressors / dccrba / cmm) ──────────

    // coriolis_matrix(q, qd, gravity) -> (batch, NUM_VEL*NUM_VEL) row-major C(q,qd).
    py::array_t<CT> coriolis_matrix(
        arr_t q,
        arr_t qd,
        CT gravity)
    {
        if (!fn_coriolis_matrix_) throw std::runtime_error("coriolis_matrix not available in this .so (re-register with force_rebuild=True)");
        int batch = check_inputs_2d(q, qd, num_joints_);
        py::array_t<CT> out({batch, num_vel_ * num_vel_});
        int rc = fn_coriolis_matrix_(q.data(), qd.data(), out.mutable_data(), batch, gravity);
        if (rc != 0) throw std::runtime_error("grid_rbd_coriolis_matrix failed: rc=" + std::to_string(rc));
        return out;
    }

    // kinetic_energy_regressor(q, qd, gravity) -> (batch, 10*NUM_BODIES) y_KE.
    py::array_t<CT> kinetic_energy_regressor(
        arr_t q,
        arr_t qd,
        CT gravity)
    {
        if (!fn_kinetic_energy_regressor_) throw std::runtime_error("kinetic_energy_regressor not available in this .so (re-register with force_rebuild=True)");
        int batch = check_inputs_2d(q, qd, num_joints_);
        py::array_t<CT> out({batch, 10 * num_bodies_});
        int rc = fn_kinetic_energy_regressor_(q.data(), qd.data(), out.mutable_data(), batch, gravity);
        if (rc != 0) throw std::runtime_error("grid_rbd_kinetic_energy_regressor failed: rc=" + std::to_string(rc));
        return out;
    }

    // potential_energy_regressor(q, gravity) -> (batch, 10*NUM_BODIES) y_PE.
    py::array_t<CT> potential_energy_regressor(
        arr_t q,
        CT gravity)
    {
        if (!fn_potential_energy_regressor_) throw std::runtime_error("potential_energy_regressor not available in this .so (re-register with force_rebuild=True)");
        int batch = check_q(q, "potential_energy_regressor");
        py::array_t<CT> out({batch, 10 * num_bodies_});
        int rc = fn_potential_energy_regressor_(q.data(), out.mutable_data(), batch, gravity);
        if (rc != 0) throw std::runtime_error("grid_rbd_potential_energy_regressor failed: rc=" + std::to_string(rc));
        return out;
    }

    // dccrba(q) -> (batch, 6*NUM_VEL*NUM_VEL) dA/dq tensor. Not emitted for mimic
    // robots (rc=3): the per-body Jacobian fold isn't mimic-reduced.
    py::array_t<CT> dccrba(
        arr_t q)
    {
        if (!fn_dccrba_) throw std::runtime_error(
            "dccrba not available in this .so (re-register with force_rebuild=True)");
        int batch = check_q(q, "dccrba");
        py::array_t<CT> out({batch, 6 * num_vel_ * num_vel_});
        int rc = fn_dccrba_(q.data(), out.mutable_data(), batch);
        if (rc == 3) throw std::runtime_error(
            "dccrba not available for this robot: it is not generated for mimic "
            "robots (the per-body Jacobian fold is not yet mimic-reduced)");
        if (rc != 0) throw std::runtime_error("grid_rbd_dccrba failed: rc=" + std::to_string(rc));
        return out;
    }

    // MuJoCo-convention dccrba(q) -> (batch, 6*NUM_VEL*NUM_VEL) dA/dq tensor. Double
    // G^{-1} reframe + base-rotation frame term baked in-kernel. Floating-base only
    // (and non-mimic, like the base dccrba).
    bool has_dccrba_mujoco() const { return fn_dccrba_mujoco_ != nullptr; }
    py::array_t<CT> dccrba_mujoco(
        arr_t q)
    {
        if (!fn_dccrba_mujoco_) throw std::runtime_error(
            "dccrba_mujoco unavailable: floating-base non-mimic .so only");
        int batch = check_q(q, "dccrba_mujoco");
        py::array_t<CT> out({batch, 6 * num_vel_ * num_vel_});
        int rc = fn_dccrba_mujoco_(q.data(), out.mutable_data(), batch);
        if (rc == 3) throw std::runtime_error(
            "dccrba_mujoco not available for this robot: not generated for mimic robots");
        if (rc != 0) throw std::runtime_error("grid_rbd_dccrba_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // cmm_time_variation(q, qd) -> (batch, 6*NUM_VEL) Adot. Not emitted for mimic
    // robots (rc=3), same caveat as dccrba.
    py::array_t<CT> cmm_time_variation(
        arr_t q,
        arr_t qd)
    {
        if (!fn_cmm_time_variation_) throw std::runtime_error(
            "cmm_time_variation not available in this .so (re-register with force_rebuild=True)");
        int batch = check_inputs_2d(q, qd, num_joints_);
        py::array_t<CT> out({batch, 6 * num_vel_});
        int rc = fn_cmm_time_variation_(q.data(), qd.data(), out.mutable_data(), batch);
        if (rc == 3) throw std::runtime_error(
            "cmm_time_variation not available for this robot: it is not generated "
            "for mimic robots (the per-body Jacobian fold is not yet mimic-reduced)");
        if (rc != 0) throw std::runtime_error("grid_rbd_cmm_time_variation failed: rc=" + std::to_string(rc));
        return out;
    }

    // MuJoCo-convention cmm_time_variation(q, qd) -> (batch, 6*NUM_VEL) Adot in
    // the mjx frame (column reframe baked into the kernel). Floating-base only.
    bool has_cmm_time_variation_mujoco() const { return fn_cmm_time_variation_mujoco_ != nullptr; }
    py::array_t<CT> cmm_time_variation_mujoco(arr_t q, arr_t qd)
    {
        if (!fn_cmm_time_variation_mujoco_) throw std::runtime_error(
            "cmm_time_variation_mujoco unavailable: floating-base .so only");
        int batch = check_inputs_2d(q, qd, num_joints_);
        py::array_t<CT> out({batch, 6 * num_vel_});
        int rc = fn_cmm_time_variation_mujoco_(q.data(), qd.data(), out.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error("grid_rbd_cmm_time_variation_mujoco failed: rc=" + std::to_string(rc));
        return out;
    }

    // set_inertia_params(params) — D.4 / Phase 5 runtime-mutable inertia.
    // params is a flat (10*num_bodies,) array, body-indexed bodies 1..N, each a
    // length-10 [m, h(3), I_O(6)] vector. Copies it into the device d_inertia_params
    // table; all subsequent kernel calls rebuild the per-link spatial inertia from
    // it (no recompile). Only available on a .so built with runtime_inertia=True.
    //
    // The table is sized by NUM_BODIES (the inertia-body count), NOT NUM_JOINTS:
    // for a FIXED base they coincide, but for a FLOATING base (or a mimic robot)
    // NUM_JOINTS == NUM_POS > NUM_BODIES, and the device d_inertia_params table is
    // 10*NUM_BODIES. Validating against 10*NUM_JOINTS used to reject the only
    // correct (NUM_BODIES,10) table on floating-base robots.
    void set_inertia_params(arr_t params) {
        if (!fn_set_inertia_params_) throw std::runtime_error(
            "set_inertia_params not available in this .so: register the robot with "
            "runtime_inertia=True (and force_rebuild=True) to enable the mutable "
            "inertia table.");
        // num_bodies_ is the device-table body count (grid::NUM_BODIES). It is
        // resolved from the optional grid_rbd_num_bodies symbol; for any
        // runtime_inertia .so it is present (that surface postdates num_bodies).
        const int n_bodies = num_bodies_ > 0 ? num_bodies_ : num_joints_;
        const int want = 10 * n_bodies;
        if (params.ndim() != 1 || (int)params.shape(0) != want) {
            throw std::runtime_error(
                "set_inertia_params: params must be a flat (" + std::to_string(want) +
                ",) array = 10 * num_bodies (bodies 1..N, [m, h(3), I_O(6)] each); got "
                "ndim=" + std::to_string(params.ndim()) +
                ", size=" + std::to_string(params.size()));
        }
        int rc = fn_set_inertia_params_(params.data());
        if (rc != 0) throw std::runtime_error(
            "grid_rbd_set_inertia_params failed: rc=" + std::to_string(rc));
    }

    // set_transform_params(params) — runtime-mutable joint-frame transform.
    // params is a flat (6*num_joints,) array, joint-indexed ALL joints 0..NB-1,
    // each a [x,y,z,roll,pitch,yaw] raw URDF <origin> vector. Copies it into the
    // device d_transform_params table; all subsequent kernel calls rebuild each
    // joint's constant Xfixed from it (no recompile). Only available on a .so
    // built with runtime_transform=True. The table is sized by NUM_JOINTS (one
    // origin per joint), NOT NUM_BODIES.
    void set_transform_params(arr_t params) {
        if (!fn_set_transform_params_) throw std::runtime_error(
            "set_transform_params not available in this .so: register the robot with "
            "runtime_transform=True (and force_rebuild=True) to enable the mutable "
            "joint-origin table.");
        const int want = 6 * num_joints_;
        if (params.ndim() != 1 || (int)params.shape(0) != want) {
            throw std::runtime_error(
                "set_transform_params: params must be a flat (" + std::to_string(want) +
                ",) array = 6 * num_joints ([x,y,z,roll,pitch,yaw] each); got "
                "ndim=" + std::to_string(params.ndim()) +
                ", size=" + std::to_string(params.size()));
        }
        int rc = fn_set_transform_params_(params.data());
        if (rc != 0) throw std::runtime_error(
            "grid_rbd_set_transform_params failed: rc=" + std::to_string(rc));
    }

    // set_joint_dynamics_params(params) — runtime-mutable damping/friction (C5).
    // params is a flat (2*num_vel,) array = [damping(nv) || friction(nv)], v-slot
    // indexed and alpha-folded (matching init_joint_dynamics_params). Copies it into
    // the device d_joint_dynamics_params table; all subsequent id/fd/aba/*_gradient
    // calls read the biased coefficients from it (no recompile). Bit-identical to the
    // baked literal until poked. Only available on a .so built with
    // runtime_joint_dynamics=True.
    void set_joint_dynamics_params(arr_t params) {
        if (!fn_set_jd_params_) throw std::runtime_error(
            "set_joint_dynamics_params not available in this .so: register the robot with "
            "runtime_joint_dynamics=True (and force_rebuild=True) to enable the mutable "
            "damping/friction table.");
        const int want = 2 * num_vel_;
        if (params.ndim() != 1 || (int)params.shape(0) != want) {
            throw std::runtime_error(
                "set_joint_dynamics_params: params must be a flat (" + std::to_string(want) +
                ",) array = 2 * num_vel ([damping(nv) || friction(nv)]); got "
                "ndim=" + std::to_string(params.ndim()) +
                ", size=" + std::to_string(params.size()));
        }
        int rc = fn_set_jd_params_(params.data());
        if (rc != 0) throw std::runtime_error(
            "grid_rbd_set_joint_dynamics_params failed: rc=" + std::to_string(rc));
    }

private:
    void* require_sym(const char* name) {
        dlerror();  // clear errors
        void* sym = dlsym(handle_, name);
        const char* err = dlerror();
        if (err) {
            throw std::runtime_error(
                std::string("missing symbol ") + name + " in robot .so: " + err);
        }
        return sym;
    }

    // Optional symbol: returns nullptr if absent (no throw). Used for the
    // grid_plant ABI, which an older .so may not export.
    void* opt_sym(const char* name) {
        dlerror();
        void* sym = dlsym(handle_, name);
        (void)dlerror();
        return sym;
    }

    int check_inputs_2d(const py::array_t<CT>& q,
                        const py::array_t<CT>& qd, int last_dim) const
    {
        if (q.ndim() != 2 || qd.ndim() != 2) {
            throw std::invalid_argument(
                "inputs must be 2D (batch, " + std::to_string(last_dim) + ")");
        }
        if (q.shape(0) != qd.shape(0) || q.shape(1) != last_dim || qd.shape(1) != last_dim) {
            throw std::invalid_argument(
                "inputs must be (batch, " + std::to_string(last_dim)
                + ") with matching batch dim");
        }
        int batch = (int)q.shape(0);
        if (batch > max_batch_) {
            throw std::invalid_argument(
                "batch=" + std::to_string(batch) + " > max_batch=" + std::to_string(max_batch_)
                + " (compiled-in limit; pass max_batch_size= at register_robot time to raise it)");
        }
        return batch;
    }

    void check_array_2d(const py::array_t<CT>& a, int batch, int last_dim,
                        const char* name) const
    {
        if (a.ndim() != 2 || a.shape(0) != batch || a.shape(1) != last_dim) {
            throw std::invalid_argument(
                std::string(name) + " must be (batch=" + std::to_string(batch)
                + ", " + std::to_string(last_dim) + ")");
        }
    }

    // Validate the optional f_ext kwarg and return its data pointer (or nullptr
    // if None). f_ext is (batch, 6*NUM_BODIES) float32 C-contiguous, body-major,
    // [angular; linear] in each body's LOCAL frame — same layout as the kernel's
    // d_f_ext / RBDReference.apply_external_forces. The caller must keep the
    // py::array alive across the C-ABI call (hold it in a local).
    const CT* f_ext_ptr(py::object f_ext_opt,
                           arr_t& hold,
                           int batch) const
    {
        if (f_ext_opt.is_none()) return nullptr;
        if (num_bodies_ <= 0) {
            throw std::runtime_error(
                "f_ext: this robot .so does not export grid_rbd_num_bodies "
                "(built before the external-force surface). Re-register with "
                "force_rebuild=True.");
        }
        hold = f_ext_opt.cast<
            arr_t>();
        check_array_2d(hold, batch, 6 * num_bodies_, "f_ext");
        return hold.data();
    }

    void* handle_ = nullptr;

    fn_int_v_t fn_num_joints_ = nullptr;
    fn_int_v_t fn_num_vel_    = nullptr;
    fn_int_v_t fn_num_ees_    = nullptr;
    fn_int_v_t fn_num_bodies_ = nullptr;
    fn_int_v_t fn_max_batch_  = nullptr;
    fn_int_v_t fn_max_perf_level_threads_      = nullptr;
    fn_int_v_t fn_threads_per_block_      = nullptr;
    fn_int_i_t fn_set_threads_per_block_  = nullptr;
    fn_int_s_t fn_kernel_max_threads_     = nullptr;
    fn_int_ii_t fn_set_threads_for_       = nullptr;
    fn_int_v_t fn_algo_count_             = nullptr;
    fn_int_iii_t fn_set_threads_for_n_    = nullptr;
    fn_int_ipp_t fn_get_batch_switch_     = nullptr;
    fn_int_v_t fn_init_       = nullptr;
    fn_int_v_t fn_close_      = nullptr;
    fn_dyn_t  fn_inverse_dynamics_           = nullptr;
    fn_dyn_t  fn_inverse_dynamics_mujoco_    = nullptr;  // floating-base mjx ID (optional)
    fn_minv_t  fn_minv_           = nullptr;
    fn_fd_t    fn_fd_             = nullptr;
    fn_fd_t    fn_aba_            = nullptr;
    fn_crba_t  fn_crba_           = nullptr;
    fn_crba_t  fn_crba_mujoco_    = nullptr;  // floating-base mjx CRBA (optional)
    // floating-base mjx value kernels (optional symbols; nullptr on fixed base)
    fn_fd_t    fn_fd_mujoco_      = nullptr;
    fn_fd_t    fn_aba_mujoco_     = nullptr;
    fn_q_qd_out_grav_t fn_coriolis_matrix_mujoco_ = nullptr;
    fn_frame_jac_t     fn_frame_jacobian_mujoco_  = nullptr;
    fn_frame_jac_dot_t fn_frame_jacobian_dot_mujoco_ = nullptr;
    fn_q_out_t         fn_osc_inertia_mujoco_     = nullptr;
    fn_minv_t          fn_minv_mujoco_            = nullptr;
    fn_q_out_t         fn_com_mujoco_             = nullptr;
    fn_q_qd_out_t      fn_ccrba_mujoco_           = nullptr;
    fn_q_qd_out_grav_t fn_energy_mujoco_          = nullptr;
    fn_q_qd_out_grav_t fn_kinetic_energy_regressor_mujoco_   = nullptr;
    fn_q_out_grav_t    fn_potential_energy_regressor_mujoco_ = nullptr;
    fn_ee_t    fn_ee_pose_        = nullptr;
    fn_ee_t    fn_ee_pose_grad_   = nullptr;
    fn_ee_t    fn_ee_pose_mujoco_      = nullptr;  // floating-base mjx EE pose (optional)
    fn_ee_t    fn_ee_pose_grad_mujoco_ = nullptr;  // floating-base mjx EE-pose grad (optional)
    fn_dyn_t  fn_inverse_dynamics_gradient_      = nullptr;
    fn_dyn_t  fn_inverse_dynamics_gradient_mujoco_ = nullptr;  // floating mjx (optional)
    fn_fd_t    fn_fd_grad_        = nullptr;
    fn_fd_t    fn_fd_grad_mujoco_ = nullptr;  // floating mjx (optional)
    fn_ee_t    fn_ee_pose_hessian_ = nullptr;
    fn_ee_t    fn_ee_pose_hessian_mujoco_ = nullptr;  // floating-base mjx EE-pose hessian (optional)
    fn_fk_batched_t fn_fk_batched_ = nullptr;
    fn_dyn_no_fext_t fn_idsva_so_ = nullptr;
    fn_dyn_no_fext_t fn_idsva_so_mujoco_ = nullptr;  // floating mjx (optional)
    fn_dyn_no_fext_t fn_id_regressor_ = nullptr;         // (q, qd, qdd) -> Y (optional)
    fn_dyn_no_fext_t fn_id_regressor_mujoco_ = nullptr;  // floating mjx (optional)
    fn_fd_no_fext_t   fn_fdsva_so_ = nullptr;
    fn_fd_no_fext_t   fn_fdsva_so_mujoco_ = nullptr;  // floating mjx (optional)
    fn_integrator_t fn_integrator_      = nullptr;
    fn_integrator_t fn_integrator_mujoco_ = nullptr;  // floating-base mjx integrator (optional)
    fn_integrator_t fn_integrator_grad_ = nullptr;
    fn_integrator_t fn_integrator_grad_mujoco_ = nullptr;  // floating mjx (optional)
    // grid_plant surface (optional symbols)
    fn_plant_cost_t    fn_plant_state_cost_  = nullptr;
    fn_plant_cost_t    fn_plant_input_cost_  = nullptr;
    fn_plant_barrier_t fn_plant_pos_barrier_ = nullptr;
    fn_plant_barrier_t fn_plant_vel_barrier_ = nullptr;
    fn_plant_barrier_t fn_plant_tor_barrier_ = nullptr;
    fn_plant_step_t    fn_plant_step_        = nullptr;
    fn_plant_step_t    fn_plant_step_mujoco_ = nullptr;  // floating mjx (optional)
    fn_plant_ee_t      fn_plant_ee_cost_     = nullptr;
    fn_plant_ee_t      fn_plant_com_cost_    = nullptr;
    fn_plant_mom_t     fn_plant_mom_cost_    = nullptr;
    fn_plant_ee_t      fn_plant_ee_cost_mujoco_  = nullptr;  // floating mjx (optional)
    fn_plant_ee_t      fn_plant_com_cost_mujoco_ = nullptr;  // floating mjx (optional)
    fn_plant_mom_t     fn_plant_mom_cost_mujoco_ = nullptr;  // floating mjx (optional)
    fn_plant_cost_t    fn_plant_state_cost_mujoco_ = nullptr;  // floating mjx (optional)
    fn_plant_step_grad_t fn_plant_step_grad_ = nullptr;
    fn_plant_step_grad_t fn_plant_step_grad_mujoco_ = nullptr;  // floating mjx (optional)
    fn_plant_step_hess_t fn_plant_step_hess_ = nullptr;
    fn_plant_step_hess_t fn_plant_step_hess_mujoco_ = nullptr;  // floating mjx (optional)
    // F2 centroidal / energy / general-frame kinematics (optional symbols)
    fn_q_out_t         fn_com_                 = nullptr;
    fn_q_qd_out_t      fn_ccrba_               = nullptr;
    fn_q_qd_out_grav_t fn_energy_              = nullptr;
    fn_q_out_grav_t    fn_generalized_gravity_ = nullptr;
    fn_q_out_grav_t    fn_generalized_gravity_mujoco_ = nullptr;  // floating mjx (optional)
    fn_q_qd_out_grav_t fn_nonlinear_effects_   = nullptr;
    fn_q_qd_out_grav_t fn_nonlinear_effects_mujoco_ = nullptr;  // floating mjx (optional)
    fn_frame_jac_t     fn_frame_jacobian_      = nullptr;
    fn_frame_jac_dot_t fn_frame_jacobian_dot_  = nullptr;
    fn_q_out_t         fn_osc_inertia_         = nullptr;
    fn_ee_runtime_t    fn_ee_pose_runtime_      = nullptr;
    fn_ee_runtime_t    fn_ee_pose_grad_runtime_ = nullptr;
    fn_tool_fext_t     fn_tool_fext_            = nullptr;
    fn_ee_runtime_t    fn_ee_pose_runtime_mujoco_      = nullptr;  // floating mjx (optional)
    fn_ee_runtime_t    fn_ee_pose_grad_runtime_mujoco_ = nullptr;  // floating mjx (optional)
    // PS5 value ops (optional symbols)
    fn_q_qd_out_grav_t fn_coriolis_matrix_            = nullptr;
    fn_q_qd_out_grav_t fn_kinetic_energy_regressor_   = nullptr;
    fn_q_out_grav_t    fn_potential_energy_regressor_ = nullptr;
    fn_q_out_t         fn_dccrba_                      = nullptr;
    fn_q_out_t         fn_dccrba_mujoco_               = nullptr;  // floating mjx (optional)
    fn_q_qd_out_t      fn_cmm_time_variation_          = nullptr;
    fn_q_qd_out_t      fn_cmm_time_variation_mujoco_   = nullptr;  // floating-base mjx (optional)
    fn_set_inertia_t   fn_set_inertia_params_          = nullptr;
    fn_set_transform_t fn_set_transform_params_        = nullptr;
    fn_set_jd_t        fn_set_jd_params_               = nullptr;

    int num_joints_ = 0;
    int num_vel_    = 0;
    int num_ees_    = 0;
    int num_bodies_ = 0;
    int max_batch_  = 0;
};


// Register a Runner specialization (float -> "Runner", double -> "RunnerF64").
// Both classes expose the IDENTICAL Python surface; the only difference is the
// numpy element type of inputs/outputs (float32 vs float64) and the dtype of
// the per-robot .so each one dlopens (the build flag -DGRID_WRAPPER_T_DOUBLE).
template <class CT>
static void register_runner(py::module_& m, const char* cls_name) {
    using R = RunnerT<CT>;
    py::class_<R>(m, cls_name)
        .def(py::init<const std::string&>(), py::arg("so_path"),
             "Open the per-robot .so at so_path and resolve its C ABI symbols.")
        .def_property_readonly("num_joints", &R::num_joints)
        .def_property_readonly("num_vel",    &R::num_vel)
        .def_property_readonly("num_ees",    &R::num_ees)
        .def_property_readonly("num_bodies", &R::num_bodies,
            "Number of bodies/links (incl. base for floating-base). f_ext is "
            "(batch, 6*num_bodies). 0 if the .so predates the f_ext surface.")
        .def_property_readonly("max_batch",  &R::max_batch)
        .def_property_readonly("max_perf_level_threads", &R::max_perf_level_threads,
            "Codegen-time thread-count hint (DOF-aware, warp-rounded). "
            "The default block size for kernel launches; not enforced since v2.0.")
        .def_property_readonly("threads_per_block", &R::threads_per_block,
            "Current per-block thread count used by kernel launches.")
        .def("set_threads_per_block", &R::set_threads_per_block,
            py::arg("n"),
            "Override the per-block thread count. Default is max_perf_level_threads. "
            "Smaller block sizes work (SIMT helpers use block-stride loops) but may be slower; "
            "larger sizes are valid up to the per-block max (1024 on current GPUs).")
        .def("kernel_max_threads", &R::kernel_max_threads,
            py::arg("algo"),
            "Real compiled __launch_bounds__ ceiling (cudaFuncGetAttributes "
            "maxThreadsPerBlock) of the baked kernel for the short autotune key "
            "(id, minv, fd, aba, crba, id_du, fd_du, ee_pose, ee_pose_gradient, "
            "ee_pose_hessian, idsva_so, fdsva_so). -1 if the key is unknown/not-built "
            "or the .so predates this symbol; the FFI autotune treats -1 as 'infer'.")
        .def("set_threads_for", &R::set_threads_for, py::arg("algo"), py::arg("n"),
            "E6 per-algo threads overlay: force n threads for the GridAlgo at index "
            "`algo` (n=0 clears to the baked default). Raises if the .so predates the "
            "overlay. The global set_threads_per_block override still takes precedence.")
        .def("algo_count", &R::algo_count,
            "GridAlgo enum size (per-algo overlay index bound); 0 if the .so predates it.")
        .def("has_per_algo_threads", &R::has_per_algo_threads,
            "True if the .so exposes the E6 per-algo threads overlay (set_threads_for).")
        .def("set_threads_for_n", &R::set_threads_for_n,
            py::arg("algo"), py::arg("threshold"), py::arg("n_small"),
            "E6 batch-switch: launch `algo` with n_small threads whenever a call's "
            "batch is <= threshold (threshold=0 clears). Stateless per call; the "
            "global override and the switch both beat the per-algo overlay.")
        .def("get_batch_switch", &R::get_batch_switch, py::arg("algo"),
            "(threshold, n_small) for the batch-switch on `algo`; threshold 0 = unarmed.")
        .def("has_batch_switch", &R::has_batch_switch,
            "True if the .so exposes the E6 batch-regime overlay (set_threads_for_n).")
        .def("inverse_dynamics", &R::inverse_dynamics,
             py::arg("q"), py::arg("qd"),
             py::arg("qdd") = py::none(),
             py::arg("gravity") = -9.81f,
             py::arg("f_ext") = py::none())
        .def_property_readonly("has_inverse_dynamics_mujoco", &R::has_inverse_dynamics_mujoco,
            "True if this .so exports the native MuJoCo-convention ID kernel "
            "(floating-base robots only).")
        .def("inverse_dynamics_mujoco", &R::inverse_dynamics_mujoco,
             py::arg("q"), py::arg("qd"), py::arg("qdd"),
             py::arg("gravity") = -9.81f,
             py::arg("f_ext") = py::none())
        .def("minv", &R::minv,
             py::arg("q"))
        .def("forward_dynamics", &R::forward_dynamics,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("gravity") = -9.81f,
             py::arg("f_ext") = py::none())
        .def("aba", &R::aba,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("gravity") = -9.81f,
             py::arg("f_ext") = py::none())
        .def("crba", &R::crba,
             py::arg("q"), py::arg("gravity") = -9.81f)
        .def_property_readonly("has_crba_mujoco", &R::has_crba_mujoco,
            "True if this .so exports the native MuJoCo-convention CRBA kernel "
            "(floating-base robots only).")
        .def("crba_mujoco", &R::crba_mujoco,
             py::arg("q"), py::arg("gravity") = -9.81f)
        // mjx value kernels (floating base only)
        .def_property_readonly("has_forward_dynamics_mujoco", &R::has_forward_dynamics_mujoco)
        .def("forward_dynamics_mujoco", &R::forward_dynamics_mujoco,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("gravity") = -9.81f, py::arg("f_ext") = py::none())
        .def_property_readonly("has_aba_mujoco", &R::has_aba_mujoco)
        .def("aba_mujoco", &R::aba_mujoco,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("gravity") = -9.81f, py::arg("f_ext") = py::none())
        .def_property_readonly("has_coriolis_matrix_mujoco", &R::has_coriolis_matrix_mujoco)
        .def("coriolis_matrix_mujoco", &R::coriolis_matrix_mujoco,
             py::arg("q"), py::arg("qd"), py::arg("gravity") = -9.81f)
        .def_property_readonly("has_frame_jacobian_mujoco", &R::has_frame_jacobian_mujoco)
        .def("frame_jacobian_mujoco", &R::frame_jacobian_mujoco,
             py::arg("q"), py::arg("target_jid") = -1, py::arg("reference_frame") = -1)
        .def_property_readonly("has_frame_jacobian_dot_mujoco", &R::has_frame_jacobian_dot_mujoco)
        .def("frame_jacobian_dot_mujoco", &R::frame_jacobian_dot_mujoco,
             py::arg("q"), py::arg("qd"), py::arg("target_jid") = -1, py::arg("reference_frame") = -1)
        .def_property_readonly("has_osc_inertia_mujoco", &R::has_osc_inertia_mujoco)
        .def("osc_inertia_mujoco", &R::osc_inertia_mujoco, py::arg("q"))
        .def_property_readonly("has_minv_mujoco", &R::has_minv_mujoco,
            "True if this .so exports the native MuJoCo-convention Minv kernel "
            "(floating-base robots only).")
        .def("minv_mujoco", &R::minv_mujoco, py::arg("q"))
        .def_property_readonly("has_com_mujoco", &R::has_com_mujoco)
        .def("com_mujoco", &R::com_mujoco, py::arg("q"))
        .def_property_readonly("has_ccrba_mujoco", &R::has_ccrba_mujoco)
        .def("ccrba_mujoco", &R::ccrba_mujoco, py::arg("q"), py::arg("qd"))
        .def_property_readonly("has_energy_mujoco", &R::has_energy_mujoco)
        .def("energy_mujoco", &R::energy_mujoco,
             py::arg("q"), py::arg("qd"), py::arg("gravity") = -9.81f)
        .def_property_readonly("has_kinetic_energy_regressor_mujoco", &R::has_kinetic_energy_regressor_mujoco)
        .def("kinetic_energy_regressor_mujoco", &R::kinetic_energy_regressor_mujoco,
             py::arg("q"), py::arg("qd"), py::arg("gravity") = -9.81f)
        .def_property_readonly("has_potential_energy_regressor_mujoco", &R::has_potential_energy_regressor_mujoco)
        .def("potential_energy_regressor_mujoco", &R::potential_energy_regressor_mujoco,
             py::arg("q"), py::arg("gravity") = -9.81f)
        .def_property_readonly("has_end_effector_pose_mujoco", &R::has_end_effector_pose_mujoco)
        .def("end_effector_pose_mujoco", &R::end_effector_pose_mujoco, py::arg("q"))
        .def_property_readonly("has_end_effector_pose_gradient_mujoco", &R::has_end_effector_pose_gradient_mujoco)
        .def("end_effector_pose_gradient_mujoco", &R::end_effector_pose_gradient_mujoco, py::arg("q"))
        .def_property_readonly("has_end_effector_pose_hessian_mujoco", &R::has_end_effector_pose_hessian_mujoco)
        .def("end_effector_pose_hessian_mujoco", &R::end_effector_pose_hessian_mujoco, py::arg("q"))
        .def("end_effector_pose", &R::end_effector_pose,
             py::arg("q"))
        .def("fk_batched", &R::fk_batched,
             py::arg("q"), py::arg("use_warp") = false)
        .def("end_effector_pose_gradient", &R::end_effector_pose_gradient,
             py::arg("q"))
        .def("inverse_dynamics_gradient", &R::inverse_dynamics_gradient,
             py::arg("q"), py::arg("qd"), py::arg("qdd") = py::none(),
             py::arg("gravity") = -9.81f,
             py::arg("f_ext") = py::none())
        .def_property_readonly("has_inverse_dynamics_gradient_mujoco", &R::has_inverse_dynamics_gradient_mujoco)
        .def("inverse_dynamics_gradient_mujoco", &R::inverse_dynamics_gradient_mujoco,
             py::arg("q"), py::arg("qd"), py::arg("qdd"),
             py::arg("gravity") = -9.81f, py::arg("f_ext") = py::none())
        .def("forward_dynamics_gradient", &R::forward_dynamics_gradient,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("gravity") = -9.81f,
             py::arg("f_ext") = py::none())
        .def_property_readonly("has_forward_dynamics_gradient_mujoco", &R::has_forward_dynamics_gradient_mujoco)
        .def("forward_dynamics_gradient_mujoco", &R::forward_dynamics_gradient_mujoco,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("gravity") = -9.81f, py::arg("f_ext") = py::none())
        .def("end_effector_pose_hessian", &R::end_effector_pose_hessian,
             py::arg("q"))
        .def("idsva_so", &R::idsva_so,
             py::arg("q"), py::arg("qd"), py::arg("qdd") = py::none(),
             py::arg("second_order_tensor_size"),
             py::arg("gravity") = -9.81f)
        .def_property_readonly("has_idsva_so_mujoco", &R::has_idsva_so_mujoco)
        .def("idsva_so_mujoco", &R::idsva_so_mujoco,
             py::arg("q"), py::arg("qd"), py::arg("qdd") = py::none(),
             py::arg("second_order_tensor_size"),
             py::arg("gravity") = -9.81f)
        .def_property_readonly("has_inverse_dynamics_regressor", &R::has_inverse_dynamics_regressor)
        .def("inverse_dynamics_regressor", &R::inverse_dynamics_regressor,
             py::arg("q"), py::arg("qd"), py::arg("qdd") = py::none(),
             py::arg("gravity") = -9.81f)
        .def_property_readonly("has_inverse_dynamics_regressor_mujoco", &R::has_inverse_dynamics_regressor_mujoco)
        .def("inverse_dynamics_regressor_mujoco", &R::inverse_dynamics_regressor_mujoco,
             py::arg("q"), py::arg("qd"), py::arg("qdd") = py::none(),
             py::arg("gravity") = -9.81f)
        .def("fdsva_so", &R::fdsva_so,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("second_order_tensor_size"),
             py::arg("gravity") = -9.81f)
        .def_property_readonly("has_fdsva_so_mujoco", &R::has_fdsva_so_mujoco)
        .def("fdsva_so_mujoco", &R::fdsva_so_mujoco,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("second_order_tensor_size"),
             py::arg("gravity") = -9.81f)
        .def("integrator", &R::integrator,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("dt"), py::arg("it") = 0, py::arg("gravity") = -9.81f)
        .def_property_readonly("has_integrator_mujoco", &R::has_integrator_mujoco)
        .def("integrator_mujoco", &R::integrator_mujoco,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("dt"), py::arg("it") = 0, py::arg("gravity") = -9.81f)
        .def("integrator_gradient", &R::integrator_gradient,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("dt"), py::arg("it") = 0, py::arg("gravity") = -9.81f)
        .def_property_readonly("has_integrator_gradient_mujoco", &R::has_integrator_gradient_mujoco)
        .def("integrator_gradient_mujoco", &R::integrator_gradient_mujoco,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("dt"), py::arg("it") = 0, py::arg("gravity") = -9.81f)
        // ─── grid_plant surface (G1) ──────────────────────────────────────
        .def("quadratic_state_cost", &R::quadratic_state_cost,
             py::arg("x"), py::arg("x_des"), py::arg("Q"))
        .def("quadratic_input_cost", &R::quadratic_input_cost,
             py::arg("u"), py::arg("u_des"), py::arg("R"))
        .def_property_readonly("has_quadratic_state_cost_mujoco", &R::has_quadratic_state_cost_mujoco)
        .def("quadratic_state_cost_mujoco", &R::quadratic_state_cost_mujoco,
             py::arg("x"), py::arg("x_des"), py::arg("Q"))
        .def("joint_position_barrier", &R::joint_position_barrier,
             py::arg("var"), py::arg("lower"), py::arg("upper"), py::arg("mu"))
        .def("joint_velocity_barrier", &R::joint_velocity_barrier,
             py::arg("var"), py::arg("lower"), py::arg("upper"), py::arg("mu"))
        .def("joint_torque_barrier", &R::joint_torque_barrier,
             py::arg("var"), py::arg("lower"), py::arg("upper"), py::arg("mu"))
        .def("plant_step", &R::plant_step,
             py::arg("x"), py::arg("u"), py::arg("dt"),
             py::arg("it") = 0, py::arg("gravity") = -9.81f)
        .def_property_readonly("has_plant_step_mujoco", &R::has_plant_step_mujoco)
        .def("plant_step_mujoco", &R::plant_step_mujoco,
             py::arg("x"), py::arg("u"), py::arg("dt"),
             py::arg("it") = 0, py::arg("gravity") = -9.81f)
        .def("plant_step_gradient", &R::plant_step_gradient,
             py::arg("x"), py::arg("u"), py::arg("dt"),
             py::arg("it") = 0, py::arg("gravity") = -9.81f)
        .def_property_readonly("has_plant_step_gradient_mujoco", &R::has_plant_step_gradient_mujoco)
        .def("plant_step_gradient_mujoco", &R::plant_step_gradient_mujoco,
             py::arg("x"), py::arg("u"), py::arg("dt"),
             py::arg("it") = 0, py::arg("gravity") = -9.81f)
        .def("plant_step_hessian", &R::plant_step_hessian,
             py::arg("x"), py::arg("u"), py::arg("dt"),
             py::arg("it") = 0, py::arg("gravity") = -9.81f)
        .def_property_readonly("has_plant_step_hessian_mujoco", &R::has_plant_step_hessian_mujoco)
        .def("plant_step_hessian_mujoco", &R::plant_step_hessian_mujoco,
             py::arg("x"), py::arg("u"), py::arg("dt"),
             py::arg("it") = 0, py::arg("gravity") = -9.81f)
        .def("ee_pos_cost", &R::ee_pos_cost,
             py::arg("q"), py::arg("p_des"), py::arg("W"))
        .def("com_cost", &R::com_cost,
             py::arg("q"), py::arg("p_des"), py::arg("W"))
        .def("momentum_cost", &R::momentum_cost,
             py::arg("q"), py::arg("qd"), py::arg("h_des"), py::arg("W"))
        .def_property_readonly("has_ee_pos_cost_mujoco", &R::has_ee_pos_cost_mujoco)
        .def("ee_pos_cost_mujoco", &R::ee_pos_cost_mujoco,
             py::arg("q"), py::arg("p_des"), py::arg("W"))
        .def_property_readonly("has_com_cost_mujoco", &R::has_com_cost_mujoco)
        .def("com_cost_mujoco", &R::com_cost_mujoco,
             py::arg("q"), py::arg("p_des"), py::arg("W"))
        .def_property_readonly("has_momentum_cost_mujoco", &R::has_momentum_cost_mujoco)
        .def("momentum_cost_mujoco", &R::momentum_cost_mujoco,
             py::arg("q"), py::arg("qd"), py::arg("h_des"), py::arg("W"))
        // ─── centroidal / energy / general-frame kinematics (F2) ───────────
        .def("com", &R::com, py::arg("q"))
        .def("ccrba", &R::ccrba, py::arg("q"), py::arg("qd"))
        .def("energy", &R::energy,
             py::arg("q"), py::arg("qd"), py::arg("gravity") = -9.81f)
        .def("generalized_gravity", &R::generalized_gravity,
             py::arg("q"), py::arg("gravity") = -9.81f)
        .def_property_readonly("has_generalized_gravity_mujoco", &R::has_generalized_gravity_mujoco)
        .def("generalized_gravity_mujoco", &R::generalized_gravity_mujoco,
             py::arg("q"), py::arg("gravity") = -9.81f)
        .def("nonlinear_effects", &R::nonlinear_effects,
             py::arg("q"), py::arg("qd"), py::arg("gravity") = -9.81f)
        .def_property_readonly("has_nonlinear_effects_mujoco", &R::has_nonlinear_effects_mujoco)
        .def("nonlinear_effects_mujoco", &R::nonlinear_effects_mujoco,
             py::arg("q"), py::arg("qd"), py::arg("gravity") = -9.81f)
        .def("frame_jacobian", &R::frame_jacobian,
             py::arg("q"), py::arg("target_jid") = -1, py::arg("reference_frame") = -1)
        .def("frame_jacobian_dot", &R::frame_jacobian_dot,
             py::arg("q"), py::arg("qd"),
             py::arg("target_jid") = -1, py::arg("reference_frame") = -1)
        .def("osc_inertia", &R::osc_inertia, py::arg("q"))
        .def("end_effector_pose_runtime", &R::end_effector_pose_runtime,
             py::arg("q"), py::arg("target_jid") = -1,
             py::arg("offset") = py::array_t<float>())
        .def("end_effector_pose_gradient_runtime", &R::end_effector_pose_gradient_runtime,
             py::arg("q"), py::arg("target_jid") = -1,
             py::arg("offset") = py::array_t<float>())
        .def_property_readonly("has_tool_fext", &R::has_tool_fext)
        .def("tool_fext", &R::tool_fext,
             py::arg("q"), py::arg("wrench"), py::arg("jid"), py::arg("rc"))
        .def_property_readonly("has_end_effector_pose_runtime_mujoco", &R::has_end_effector_pose_runtime_mujoco)
        .def("end_effector_pose_runtime_mujoco", &R::end_effector_pose_runtime_mujoco,
             py::arg("q"), py::arg("target_jid") = -1,
             py::arg("offset") = py::array_t<float>())
        .def_property_readonly("has_end_effector_pose_gradient_runtime_mujoco", &R::has_end_effector_pose_gradient_runtime_mujoco)
        .def("end_effector_pose_gradient_runtime_mujoco", &R::end_effector_pose_gradient_runtime_mujoco,
             py::arg("q"), py::arg("target_jid") = -1,
             py::arg("offset") = py::array_t<float>())
        .def("coriolis_matrix", &R::coriolis_matrix,
             py::arg("q"), py::arg("qd"), py::arg("gravity") = -9.81f)
        .def("kinetic_energy_regressor", &R::kinetic_energy_regressor,
             py::arg("q"), py::arg("qd"), py::arg("gravity") = -9.81f)
        .def("potential_energy_regressor", &R::potential_energy_regressor,
             py::arg("q"), py::arg("gravity") = -9.81f)
        .def("dccrba", &R::dccrba, py::arg("q"))
        .def_property_readonly("has_dccrba_mujoco", &R::has_dccrba_mujoco)
        .def("dccrba_mujoco", &R::dccrba_mujoco, py::arg("q"))
        .def("cmm_time_variation", &R::cmm_time_variation,
             py::arg("q"), py::arg("qd"))
        .def_property_readonly("has_cmm_time_variation_mujoco", &R::has_cmm_time_variation_mujoco)
        .def("cmm_time_variation_mujoco", &R::cmm_time_variation_mujoco,
             py::arg("q"), py::arg("qd"))
        .def("set_inertia_params", &R::set_inertia_params,
             py::arg("params"),
             "Update the device-resident mutable inertia table (D.4 / Phase 5). "
             "params is a flat (10*num_joints,) array, bodies 1..N, each a length-10 "
             "[m, h(3), I_O(6)] vector. Only available on a .so built with "
             "runtime_inertia=True; raises otherwise.")
        .def("set_transform_params", &R::set_transform_params,
             py::arg("params"),
             "Update the device-resident mutable joint-origin transform table "
             "(runtime_transform). params is a flat (6*num_joints,) array, joints "
             "0..NB-1, each a [x,y,z,roll,pitch,yaw] raw URDF <origin> vector. Only "
             "available on a .so built with runtime_transform=True; raises otherwise.")
        .def("set_joint_dynamics_params", &R::set_joint_dynamics_params,
             py::arg("params"),
             "Update the device-resident mutable damping/friction table (C5 "
             "runtime_joint_dynamics). params is a flat (2*num_vel,) array = "
             "[damping(nv) || friction(nv)], v-slot indexed (alpha-folded). Only "
             "available on a .so built with runtime_joint_dynamics=True; raises otherwise.");
}


PYBIND11_MODULE(_core, m) {
    m.doc() = "grid-rbd internal: pybind11 Runner that dlopens a per-robot "
              "compiled .so and dispatches numpy calls through its C ABI. "
              "Runner = fp32 buffers; RunnerF64 = fp64 buffers (loads a "
              ".so built with -DGRID_WRAPPER_T_DOUBLE).";
    register_runner<float>(m, "Runner");
    register_runner<double>(m, "RunnerF64");
}
