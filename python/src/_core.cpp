// grid_rbd._core — pybind11 Runner that dlopens a per-robot .so and
// dispatches numpy arrays through its C ABI.
//
// The per-robot .so is built at register_robot() time by
// grid_rbd._compile.generate_and_compile() from a generated grid.cuh plus
// the robot-agnostic wrapper.cu (see grid_rbd/wrapper_template.cu). It
// exports `extern "C"` symbols like:
//
//   int grid_rbd_init();
//   int grid_rbd_num_joints();
//   int grid_rbd_rnea(const float* q, const float* qd, const float* qdd_opt,
//                     float* c_out, int batch, float gravity,
//                     const float* f_ext_opt);  // f_ext_opt may be nullptr
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

extern "C" {
    using fn_int_v_t        = int (*)();
    using fn_int_i_t        = int (*)(int);
    // q, qd, qdd_opt, out, batch, gravity, f_ext_opt   — rnea, rnea_grad, idsva_so
    //   f_ext_opt: (batch, 6*NUM_BODIES) local-frame body wrenches, or null
    using fn_rnea_t         = int (*)(const float*, const float*, const float*,
                                      float*, int, float, const float*);
    // q, out, batch
    using fn_minv_t         = int (*)(const float*, float*, int);
    // q, qd, u, out, batch, gravity, f_ext_opt   — fd, aba, fd_grad
    //   f_ext_opt: (batch, 6*NUM_BODIES) local-frame body wrenches, or null
    using fn_fd_t           = int (*)(const float*, const float*, const float*,
                                      float*, int, float, const float*);
    // q, qd, qdd_opt, out, batch, gravity   — idsva_so (no f_ext; 2nd-order surface)
    using fn_rnea_no_fext_t = int (*)(const float*, const float*, const float*,
                                      float*, int, float);
    // q, qd, u, out, batch, gravity   — fdsva_so (no f_ext; second-order surface)
    using fn_fd_no_fext_t   = int (*)(const float*, const float*, const float*,
                                      float*, int, float);
    // q, out, batch, gravity          — crba
    using fn_crba_t         = int (*)(const float*, float*, int, float);
    // q, out, batch                   — ee_pose, ee_pose_gradient, ee_pose_hessian
    using fn_ee_t           = int (*)(const float*, float*, int);
    // q, pose7_out, batch, use_warp    — fk_batched (pos+quat, one block/warp per sample)
    using fn_fk_batched_t   = int (*)(const float*, float*, int, int);
    // q, qd, qdd_opt, out, batch, gravity   — idsva_so (same as rnea)
    // q, qd, u, out, batch, gravity         — fdsva_so (same as fd)
    // q, qd, u, out, batch, dt, it          — integrator, integrator_gradient
    //   (dt is the runtime timestep; it selects the IntegratorType; gravity is
    //    the standard 9.81 constant baked in the wrapper)
    using fn_integrator_t   = int (*)(const float*, const float*, const float*,
                                      float*, int, float, float, int);
    // grid_plant C ABI (G1 binding layer)
    // quadratic cost: (var, des, w, out, grad, hess, batch)
    using fn_plant_cost_t   = int (*)(const float*, const float*, const float*,
                                      float*, float*, float*, int);
    // barrier: (var, lower, upper, mu, out, grad, hess_diag, batch)
    using fn_plant_barrier_t = int (*)(const float*, const float*, const float*, float,
                                       float*, float*, float*, int);
    // plant_step: (x, u, x_kp1, batch, gravity, dt, it)
    using fn_plant_step_t   = int (*)(const float*, const float*, float*,
                                      int, float, float, int);
    // ee_pos_cost: (q, p_des, W, out, grad, hess, batch)
    using fn_plant_ee_t     = int (*)(const float*, const float*, const float*,
                                      float*, float*, float*, int);
}


// ─── Runner ──────────────────────────────────────────────────────────────────

class Runner {
public:
    explicit Runner(const std::string& so_path) {
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
        fn_init_             = reinterpret_cast<fn_int_v_t>(require_sym("grid_rbd_init"));
        fn_close_            = reinterpret_cast<fn_int_v_t>(require_sym("grid_rbd_close"));

        // Algorithm symbols — required for v1 surface.
        fn_rnea_             = reinterpret_cast<fn_rnea_t>(require_sym("grid_rbd_rnea"));
        fn_minv_             = reinterpret_cast<fn_minv_t>(require_sym("grid_rbd_minv"));
        fn_fd_               = reinterpret_cast<fn_fd_t>  (require_sym("grid_rbd_forward_dynamics"));
        fn_aba_              = reinterpret_cast<fn_fd_t>  (require_sym("grid_rbd_aba"));
        fn_crba_             = reinterpret_cast<fn_crba_t>(require_sym("grid_rbd_crba"));
        fn_ee_pose_          = reinterpret_cast<fn_ee_t>  (require_sym("grid_rbd_end_effector_pose"));
        fn_ee_pose_grad_     = reinterpret_cast<fn_ee_t>  (require_sym("grid_rbd_end_effector_pose_gradient"));
        fn_rnea_grad_        = reinterpret_cast<fn_rnea_t>(require_sym("grid_rbd_rnea_grad"));
        fn_fd_grad_          = reinterpret_cast<fn_fd_t>  (require_sym("grid_rbd_forward_dynamics_grad"));
        // Phase-C extension: hessian + SO. Required for v0.1+ .so files.
        fn_ee_pose_hessian_  = reinterpret_cast<fn_ee_t>  (require_sym("grid_rbd_end_effector_pose_hessian"));
        fn_idsva_so_         = reinterpret_cast<fn_rnea_no_fext_t>(require_sym("grid_rbd_idsva_so"));
        fn_fdsva_so_         = reinterpret_cast<fn_fd_no_fext_t>  (require_sym("grid_rbd_fdsva_so"));
        fn_integrator_       = reinterpret_cast<fn_integrator_t>(require_sym("grid_rbd_integrator"));
        fn_integrator_grad_  = reinterpret_cast<fn_integrator_t>(require_sym("grid_rbd_integrator_gradient"));

        // grid_plant C ABI (G1) — OPTIONAL: resolve if present (older .so files
        // built before the plant surface won't have them; the handle methods
        // raise a clear error at call time if the symbol is null).
        fn_plant_state_cost_ = reinterpret_cast<fn_plant_cost_t>(opt_sym("grid_plant_quadratic_state_cost"));
        fn_plant_input_cost_ = reinterpret_cast<fn_plant_cost_t>(opt_sym("grid_plant_quadratic_input_cost"));
        fn_plant_pos_barrier_ = reinterpret_cast<fn_plant_barrier_t>(opt_sym("grid_plant_joint_position_barrier"));
        fn_plant_vel_barrier_ = reinterpret_cast<fn_plant_barrier_t>(opt_sym("grid_plant_joint_velocity_barrier"));
        fn_plant_tor_barrier_ = reinterpret_cast<fn_plant_barrier_t>(opt_sym("grid_plant_joint_torque_barrier"));
        fn_plant_step_       = reinterpret_cast<fn_plant_step_t>(opt_sym("grid_plant_step"));
        fn_plant_ee_cost_    = reinterpret_cast<fn_plant_ee_t>(opt_sym("grid_plant_ee_pos_cost"));

        // G2 batched FK (pos+quat) — OPTIONAL: only present in newer .so files
        // (and only non-null for fixed-base/non-mimic robots).
        fn_fk_batched_      = reinterpret_cast<fn_fk_batched_t>(opt_sym("grid_rbd_fk_batched"));

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

    ~Runner() {
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
    int threads_per_block() const { return fn_threads_per_block_(); }
    void set_threads_per_block(int n) {
        // Override the per-block thread count for all subsequent kernel
        // launches. Default is MAX_PERF_LEVEL_THREADS; the codegen no longer
        // pins launch_bounds (cuBLASDx removed in v2.0), so any positive
        // n that fits per-block (≤1024 on current GPUs) is valid.
        if (n < 1) {
            throw std::invalid_argument(
                "set_threads_per_block: n must be >= 1, got " + std::to_string(n));
        }
        int rc = fn_set_threads_per_block_(n);
        if (rc != 0) {
            throw std::runtime_error(
                "grid_rbd_set_threads_per_block failed: rc=" + std::to_string(rc));
        }
    }

    // ─── rnea ────────────────────────────────────────────────────────────────
    //
    // q, qd:  (batch, num_joints) float32, C-contiguous
    // qdd:    optional (batch, num_joints) — currently ignored
    //         (USE_QDD_FLAG=false in wrapper); future v2 will plumb through.
    // returns c: (batch, num_joints) float32
    py::array_t<float> rnea(
        py::array_t<float, py::array::c_style | py::array::forcecast> q,
        py::array_t<float, py::array::c_style | py::array::forcecast> qd,
        py::object qdd_opt,
        float gravity,
        py::object f_ext_opt)
    {
        int batch = check_inputs_2d(q, qd, /*last_dim=*/num_joints_);
        const float* qdd_ptr = nullptr;
        if (!qdd_opt.is_none()) {
            auto qdd = qdd_opt.cast<
                py::array_t<float, py::array::c_style | py::array::forcecast>>();
            check_array_2d(qdd, batch, num_joints_, "qdd");
            qdd_ptr = qdd.data();
        }
        py::array_t<float, py::array::c_style | py::array::forcecast> fe_hold;
        const float* fe_ptr = f_ext_ptr(f_ext_opt, fe_hold, batch);

        py::array_t<float> out({batch, num_joints_});
        int rc = fn_rnea_(q.data(), qd.data(), qdd_ptr,
                          out.mutable_data(), batch, gravity, fe_ptr);
        if (rc != 0) {
            throw std::runtime_error("grid_rbd_rnea failed: rc=" + std::to_string(rc));
        }
        return out;
    }

    // ─── minv ────────────────────────────────────────────────────────────────
    py::array_t<float> minv(
        py::array_t<float, py::array::c_style | py::array::forcecast> q)
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
        py::array_t<float> out({batch, num_joints_, num_joints_});
        int rc = fn_minv_(q.data(), out.mutable_data(), batch);
        if (rc != 0) {
            throw std::runtime_error("grid_rbd_minv failed: rc=" + std::to_string(rc));
        }
        return out;
    }

    // ─── forward_dynamics ────────────────────────────────────────────────────
    py::array_t<float> forward_dynamics(
        py::array_t<float, py::array::c_style | py::array::forcecast> q,
        py::array_t<float, py::array::c_style | py::array::forcecast> qd,
        py::array_t<float, py::array::c_style | py::array::forcecast> u,
        float gravity,
        py::object f_ext_opt)
    {
        int batch = check_inputs_2d(q, qd, /*last_dim=*/num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        py::array_t<float, py::array::c_style | py::array::forcecast> fe_hold;
        const float* fe_ptr = f_ext_ptr(f_ext_opt, fe_hold, batch);

        py::array_t<float> out({batch, num_joints_});
        int rc = fn_fd_(q.data(), qd.data(), u.data(),
                        out.mutable_data(), batch, gravity, fe_ptr);
        if (rc != 0) {
            throw std::runtime_error("grid_rbd_forward_dynamics failed: rc=" + std::to_string(rc));
        }
        return out;
    }

    // ─── aba ─────────────────────────────────────────────────────────────────
    py::array_t<float> aba(
        py::array_t<float, py::array::c_style | py::array::forcecast> q,
        py::array_t<float, py::array::c_style | py::array::forcecast> qd,
        py::array_t<float, py::array::c_style | py::array::forcecast> u,
        float gravity,
        py::object f_ext_opt)
    {
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        py::array_t<float, py::array::c_style | py::array::forcecast> fe_hold;
        const float* fe_ptr = f_ext_ptr(f_ext_opt, fe_hold, batch);
        py::array_t<float> out({batch, num_joints_});
        int rc = fn_aba_(q.data(), qd.data(), u.data(),
                         out.mutable_data(), batch, gravity, fe_ptr);
        if (rc != 0) throw std::runtime_error("grid_rbd_aba failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── crba ────────────────────────────────────────────────────────────────
    py::array_t<float> crba(
        py::array_t<float, py::array::c_style | py::array::forcecast> q,
        float gravity)
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
        py::array_t<float> out({batch, num_joints_, num_joints_});
        int rc = fn_crba_(q.data(), out.mutable_data(), batch, gravity);
        if (rc != 0) throw std::runtime_error("grid_rbd_crba failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── end_effector_pose ───────────────────────────────────────────────────
    py::array_t<float> end_effector_pose(
        py::array_t<float, py::array::c_style | py::array::forcecast> q)
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
        py::array_t<float> out({batch, 6 * num_ees_});
        int rc = fn_ee_pose_(q.data(), out.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error("grid_rbd_end_effector_pose failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── fk_batched (large-batch FK, pos+quat) ───────────────────────────────
    // Input  q:     (batch, NUM_POS)
    // Output pose7: (batch, 7) = [tx,ty,tz, qw,qx,qy,qz]
    // use_warp selects the warp-cooperative per-sample inner.
    py::array_t<float> fk_batched(
        py::array_t<float, py::array::c_style | py::array::forcecast> q,
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
        py::array_t<float> out({batch, 7});
        int rc = fn_fk_batched_(q.data(), out.mutable_data(), batch, use_warp ? 1 : 0);
        if (rc == 3) throw std::runtime_error(
            "fk_batched: not supported for this robot (floating-base / mimic)");
        if (rc != 0) throw std::runtime_error("grid_rbd_fk_batched failed: rc=" + std::to_string(rc));
        return out;
    }

    py::array_t<float> end_effector_pose_gradient(
        py::array_t<float, py::array::c_style | py::array::forcecast> q)
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
        py::array_t<float> out({batch, 6 * num_ees_, num_vel_});
        int rc = fn_ee_pose_grad_(q.data(), out.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error("grid_rbd_end_effector_pose_gradient failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── rnea_grad / forward_dynamics_grad ───────────────────────────────────
    py::array_t<float> rnea_grad(
        py::array_t<float, py::array::c_style | py::array::forcecast> q,
        py::array_t<float, py::array::c_style | py::array::forcecast> qd,
        py::object qdd_opt,
        float gravity,
        py::object f_ext_opt)
    {
        int batch = check_inputs_2d(q, qd, num_joints_);
        const float* qdd_ptr = nullptr;
        if (!qdd_opt.is_none()) {
            auto qdd = qdd_opt.cast<
                py::array_t<float, py::array::c_style | py::array::forcecast>>();
            check_array_2d(qdd, batch, num_joints_, "qdd");
            qdd_ptr = qdd.data();
        }
        py::array_t<float, py::array::c_style | py::array::forcecast> fe_hold;
        const float* fe_ptr = f_ext_ptr(f_ext_opt, fe_hold, batch);
        py::array_t<float> out({batch, num_joints_, 2 * num_joints_});
        int rc = fn_rnea_grad_(q.data(), qd.data(), qdd_ptr,
                               out.mutable_data(), batch, gravity, fe_ptr);
        if (rc != 0) throw std::runtime_error("grid_rbd_rnea_grad failed: rc=" + std::to_string(rc));
        return out;
    }

    py::array_t<float> forward_dynamics_grad(
        py::array_t<float, py::array::c_style | py::array::forcecast> q,
        py::array_t<float, py::array::c_style | py::array::forcecast> qd,
        py::array_t<float, py::array::c_style | py::array::forcecast> u,
        float gravity,
        py::object f_ext_opt)
    {
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        py::array_t<float, py::array::c_style | py::array::forcecast> fe_hold;
        const float* fe_ptr = f_ext_ptr(f_ext_opt, fe_hold, batch);
        py::array_t<float> out({batch, num_joints_, 2 * num_joints_});
        int rc = fn_fd_grad_(q.data(), qd.data(), u.data(),
                             out.mutable_data(), batch, gravity, fe_ptr);
        if (rc != 0) throw std::runtime_error("grid_rbd_forward_dynamics_grad failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── end_effector_pose_hessian ───────────────────────────────────────────
    py::array_t<float> end_effector_pose_hessian(
        py::array_t<float, py::array::c_style | py::array::forcecast> q)
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
        py::array_t<float> out({batch, 6 * num_ees_, num_vel_, num_vel_});
        int rc = fn_ee_pose_hessian_(q.data(), out.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error("grid_rbd_end_effector_pose_hessian failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── idsva_so / fdsva_so (raw second-order tensor surface) ───────────────
    // Returns shape (B, SECOND_ORDER_TENSOR_SIZE) — flat, 4 * NV^3 floats per
    // timestep. The Python side slices into the four NV^3 tensors.
    py::array_t<float> idsva_so(
        py::array_t<float, py::array::c_style | py::array::forcecast> q,
        py::array_t<float, py::array::c_style | py::array::forcecast> qd,
        py::object qdd_opt,
        int second_order_tensor_size,
        float gravity)
    {
        int batch = check_inputs_2d(q, qd, num_joints_);
        const float* qdd_ptr = nullptr;
        if (!qdd_opt.is_none()) {
            auto qdd = qdd_opt.cast<
                py::array_t<float, py::array::c_style | py::array::forcecast>>();
            check_array_2d(qdd, batch, num_joints_, "qdd");
            qdd_ptr = qdd.data();
        }
        py::array_t<float> out({batch, second_order_tensor_size});
        int rc = fn_idsva_so_(q.data(), qd.data(), qdd_ptr,
                              out.mutable_data(), batch, gravity);
        if (rc != 0) throw std::runtime_error("grid_rbd_idsva_so failed: rc=" + std::to_string(rc));
        return out;
    }

    py::array_t<float> fdsva_so(
        py::array_t<float, py::array::c_style | py::array::forcecast> q,
        py::array_t<float, py::array::c_style | py::array::forcecast> qd,
        py::array_t<float, py::array::c_style | py::array::forcecast> u,
        int second_order_tensor_size,
        float gravity)
    {
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        py::array_t<float> out({batch, second_order_tensor_size});
        int rc = fn_fdsva_so_(q.data(), qd.data(), u.data(),
                              out.mutable_data(), batch, gravity);
        if (rc != 0) throw std::runtime_error("grid_rbd_fdsva_so failed: rc=" + std::to_string(rc));
        return out;
    }

    // integrator(q, qd, u, dt, it) -> x_kp1 (batch, NUM_POS + NUM_VEL).
    // gravity is the standard 9.81 constant (baked in the wrapper).
    py::array_t<float> integrator(
        py::array_t<float, py::array::c_style | py::array::forcecast> q,
        py::array_t<float, py::array::c_style | py::array::forcecast> qd,
        py::array_t<float, py::array::c_style | py::array::forcecast> u,
        float dt, int it, float gravity)
    {
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        py::array_t<float> out({batch, num_joints_ + num_vel_});
        int rc = fn_integrator_(q.data(), qd.data(), u.data(),
                                out.mutable_data(), batch, gravity, dt, it);
        if (rc != 0) throw std::runtime_error("grid_rbd_integrator failed: rc=" + std::to_string(rc));
        return out;
    }

    // integrator_gradient(q, qd, u, dt, it) -> flat dAB (batch, 2*NV*3*NV),
    // column-major per timestep ([d/dq | d/dqd | d/du]); reshaped Python-side.
    py::array_t<float> integrator_gradient(
        py::array_t<float, py::array::c_style | py::array::forcecast> q,
        py::array_t<float, py::array::c_style | py::array::forcecast> qd,
        py::array_t<float, py::array::c_style | py::array::forcecast> u,
        float dt, int it, float gravity)
    {
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        py::array_t<float> out({batch, 2 * num_vel_ * 3 * num_vel_});
        int rc = fn_integrator_grad_(q.data(), qd.data(), u.data(),
                                     out.mutable_data(), batch, gravity, dt, it);
        if (rc != 0) throw std::runtime_error("grid_rbd_integrator_gradient failed: rc=" + std::to_string(rc));
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
    std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
    plant_quadratic_cost(fn_plant_cost_t fn, const char* name,
        py::array_t<float, py::array::c_style | py::array::forcecast> var,
        py::array_t<float, py::array::c_style | py::array::forcecast> des,
        py::array_t<float, py::array::c_style | py::array::forcecast> w,
        int N)
    {
        require_plant((void*)fn, name);
        if (var.ndim() != 2 || var.shape(1) != N)
            throw std::invalid_argument(std::string(name) + ": var must be (batch, " + std::to_string(N) + ")");
        int batch = (int)var.shape(0);
        if (batch > max_batch_) throw std::invalid_argument(std::string(name) + ": batch > max_batch");
        check_array_2d(des, batch, N, "des");
        check_array_2d(w, batch, N, "weight");
        py::array_t<float> out({batch});
        py::array_t<float> grad({batch, N});
        py::array_t<float> hess({batch, N, N});
        int rc = fn(var.data(), des.data(), w.data(),
                    out.mutable_data(), grad.mutable_data(), hess.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error(std::string(name) + " failed: rc=" + std::to_string(rc));
        return {out, grad, hess};
    }

    std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
    quadratic_state_cost(
        py::array_t<float, py::array::c_style | py::array::forcecast> x,
        py::array_t<float, py::array::c_style | py::array::forcecast> x_des,
        py::array_t<float, py::array::c_style | py::array::forcecast> Q)
    { return plant_quadratic_cost(fn_plant_state_cost_, "quadratic_state_cost", x, x_des, Q, num_joints_ + num_vel_); }

    std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
    quadratic_input_cost(
        py::array_t<float, py::array::c_style | py::array::forcecast> u,
        py::array_t<float, py::array::c_style | py::array::forcecast> u_des,
        py::array_t<float, py::array::c_style | py::array::forcecast> R)
    { return plant_quadratic_cost(fn_plant_input_cost_, "quadratic_input_cost", u, u_des, R, num_vel_); }

    // barrier (position/velocity/torque). var/lower/upper are (batch, N).
    std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
    plant_barrier(fn_plant_barrier_t fn, const char* name,
        py::array_t<float, py::array::c_style | py::array::forcecast> var,
        py::array_t<float, py::array::c_style | py::array::forcecast> lower,
        py::array_t<float, py::array::c_style | py::array::forcecast> upper,
        float mu, int N)
    {
        require_plant((void*)fn, name);
        if (var.ndim() != 2 || var.shape(1) != N)
            throw std::invalid_argument(std::string(name) + ": var must be (batch, " + std::to_string(N) + ")");
        int batch = (int)var.shape(0);
        if (batch > max_batch_) throw std::invalid_argument(std::string(name) + ": batch > max_batch");
        check_array_2d(lower, batch, N, "lower");
        check_array_2d(upper, batch, N, "upper");
        py::array_t<float> out({batch});
        py::array_t<float> grad({batch, N});
        py::array_t<float> hess_diag({batch, N});
        int rc = fn(var.data(), lower.data(), upper.data(), mu,
                    out.mutable_data(), grad.mutable_data(), hess_diag.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error(std::string(name) + " failed: rc=" + std::to_string(rc));
        return {out, grad, hess_diag};
    }

    std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
    joint_position_barrier(
        py::array_t<float, py::array::c_style | py::array::forcecast> var,
        py::array_t<float, py::array::c_style | py::array::forcecast> lower,
        py::array_t<float, py::array::c_style | py::array::forcecast> upper, float mu)
    { return plant_barrier(fn_plant_pos_barrier_, "joint_position_barrier", var, lower, upper, mu, num_joints_); }

    std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
    joint_velocity_barrier(
        py::array_t<float, py::array::c_style | py::array::forcecast> var,
        py::array_t<float, py::array::c_style | py::array::forcecast> lower,
        py::array_t<float, py::array::c_style | py::array::forcecast> upper, float mu)
    { return plant_barrier(fn_plant_vel_barrier_, "joint_velocity_barrier", var, lower, upper, mu, num_vel_); }

    std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
    joint_torque_barrier(
        py::array_t<float, py::array::c_style | py::array::forcecast> var,
        py::array_t<float, py::array::c_style | py::array::forcecast> lower,
        py::array_t<float, py::array::c_style | py::array::forcecast> upper, float mu)
    { return plant_barrier(fn_plant_tor_barrier_, "joint_torque_barrier", var, lower, upper, mu, num_vel_); }

    // plant_step: x (batch, NX), u (batch, NV) -> x_kp1 (batch, NX).
    py::array_t<float> plant_step(
        py::array_t<float, py::array::c_style | py::array::forcecast> x,
        py::array_t<float, py::array::c_style | py::array::forcecast> u,
        float dt, int it, float gravity)
    {
        require_plant((void*)fn_plant_step_, "plant_step");
        int nx = num_joints_ + num_vel_;
        if (x.ndim() != 2 || x.shape(1) != nx)
            throw std::invalid_argument("plant_step: x must be (batch, " + std::to_string(nx) + ")");
        int batch = (int)x.shape(0);
        if (batch > max_batch_) throw std::invalid_argument("plant_step: batch > max_batch");
        check_array_2d(u, batch, num_vel_, "u");
        py::array_t<float> out({batch, nx});
        int rc = fn_plant_step_(x.data(), u.data(), out.mutable_data(), batch, gravity, dt, it);
        if (rc != 0) throw std::runtime_error("plant_step failed: rc=" + std::to_string(rc));
        return out;
    }

    // ee_pos_cost: q (batch, NQ), p_des (batch, 3), W (batch, 3)
    // -> (value (batch,), grad_x (batch, NX), hess_x (batch, NX, NX)).
    std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
    ee_pos_cost(
        py::array_t<float, py::array::c_style | py::array::forcecast> q,
        py::array_t<float, py::array::c_style | py::array::forcecast> p_des,
        py::array_t<float, py::array::c_style | py::array::forcecast> W)
    {
        require_plant((void*)fn_plant_ee_cost_, "ee_pos_cost");
        int nx = num_joints_ + num_vel_;
        if (q.ndim() != 2 || q.shape(1) != num_joints_)
            throw std::invalid_argument("ee_pos_cost: q must be (batch, " + std::to_string(num_joints_) + ")");
        int batch = (int)q.shape(0);
        if (batch > max_batch_) throw std::invalid_argument("ee_pos_cost: batch > max_batch");
        check_array_2d(p_des, batch, 3, "p_des");
        check_array_2d(W, batch, 3, "W");
        py::array_t<float> out({batch});
        py::array_t<float> grad({batch, nx});
        py::array_t<float> hess({batch, nx, nx});
        int rc = fn_plant_ee_cost_(q.data(), p_des.data(), W.data(),
                                   out.mutable_data(), grad.mutable_data(), hess.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error("ee_pos_cost failed: rc=" + std::to_string(rc));
        return {out, grad, hess};
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

    int check_inputs_2d(const py::array_t<float>& q,
                        const py::array_t<float>& qd, int last_dim) const
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

    void check_array_2d(const py::array_t<float>& a, int batch, int last_dim,
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
    const float* f_ext_ptr(py::object f_ext_opt,
                           py::array_t<float, py::array::c_style | py::array::forcecast>& hold,
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
            py::array_t<float, py::array::c_style | py::array::forcecast>>();
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
    fn_int_v_t fn_init_       = nullptr;
    fn_int_v_t fn_close_      = nullptr;
    fn_rnea_t  fn_rnea_           = nullptr;
    fn_minv_t  fn_minv_           = nullptr;
    fn_fd_t    fn_fd_             = nullptr;
    fn_fd_t    fn_aba_            = nullptr;
    fn_crba_t  fn_crba_           = nullptr;
    fn_ee_t    fn_ee_pose_        = nullptr;
    fn_ee_t    fn_ee_pose_grad_   = nullptr;
    fn_rnea_t  fn_rnea_grad_      = nullptr;
    fn_fd_t    fn_fd_grad_        = nullptr;
    fn_ee_t    fn_ee_pose_hessian_ = nullptr;
    fn_fk_batched_t fn_fk_batched_ = nullptr;
    fn_rnea_no_fext_t fn_idsva_so_ = nullptr;
    fn_fd_no_fext_t   fn_fdsva_so_ = nullptr;
    fn_integrator_t fn_integrator_      = nullptr;
    fn_integrator_t fn_integrator_grad_ = nullptr;
    // grid_plant surface (optional symbols)
    fn_plant_cost_t    fn_plant_state_cost_  = nullptr;
    fn_plant_cost_t    fn_plant_input_cost_  = nullptr;
    fn_plant_barrier_t fn_plant_pos_barrier_ = nullptr;
    fn_plant_barrier_t fn_plant_vel_barrier_ = nullptr;
    fn_plant_barrier_t fn_plant_tor_barrier_ = nullptr;
    fn_plant_step_t    fn_plant_step_        = nullptr;
    fn_plant_ee_t      fn_plant_ee_cost_     = nullptr;

    int num_joints_ = 0;
    int num_vel_    = 0;
    int num_ees_    = 0;
    int num_bodies_ = 0;
    int max_batch_  = 0;
};


PYBIND11_MODULE(_core, m) {
    m.doc() = "grid-rbd internal: pybind11 Runner that dlopens a per-robot "
              "compiled .so and dispatches numpy calls through its C ABI.";

    py::class_<Runner>(m, "Runner")
        .def(py::init<const std::string&>(), py::arg("so_path"),
             "Open the per-robot .so at so_path and resolve its C ABI symbols.")
        .def_property_readonly("num_joints", &Runner::num_joints)
        .def_property_readonly("num_vel",    &Runner::num_vel)
        .def_property_readonly("num_ees",    &Runner::num_ees)
        .def_property_readonly("num_bodies", &Runner::num_bodies,
            "Number of bodies/links (incl. base for floating-base). f_ext is "
            "(batch, 6*num_bodies). 0 if the .so predates the f_ext surface.")
        .def_property_readonly("max_batch",  &Runner::max_batch)
        .def_property_readonly("max_perf_level_threads", &Runner::max_perf_level_threads,
            "Codegen-time thread-count hint (DOF-aware, warp-rounded). "
            "The default block size for kernel launches; not enforced since v2.0.")
        .def_property_readonly("threads_per_block", &Runner::threads_per_block,
            "Current per-block thread count used by kernel launches.")
        .def("set_threads_per_block", &Runner::set_threads_per_block,
            py::arg("n"),
            "Override the per-block thread count. Default is max_perf_level_threads. "
            "Smaller block sizes work (SIMT helpers use block-stride loops) but may be slower; "
            "larger sizes are valid up to the per-block max (1024 on current GPUs).")
        .def("rnea", &Runner::rnea,
             py::arg("q"), py::arg("qd"),
             py::arg("qdd") = py::none(),
             py::arg("gravity") = 9.81f,
             py::arg("f_ext") = py::none())
        .def("minv", &Runner::minv,
             py::arg("q"))
        .def("forward_dynamics", &Runner::forward_dynamics,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("gravity") = 9.81f,
             py::arg("f_ext") = py::none())
        .def("aba", &Runner::aba,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("gravity") = 9.81f,
             py::arg("f_ext") = py::none())
        .def("crba", &Runner::crba,
             py::arg("q"), py::arg("gravity") = 9.81f)
        .def("end_effector_pose", &Runner::end_effector_pose,
             py::arg("q"))
        .def("fk_batched", &Runner::fk_batched,
             py::arg("q"), py::arg("use_warp") = false)
        .def("end_effector_pose_gradient", &Runner::end_effector_pose_gradient,
             py::arg("q"))
        .def("rnea_grad", &Runner::rnea_grad,
             py::arg("q"), py::arg("qd"), py::arg("qdd") = py::none(),
             py::arg("gravity") = 9.81f,
             py::arg("f_ext") = py::none())
        .def("forward_dynamics_grad", &Runner::forward_dynamics_grad,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("gravity") = 9.81f,
             py::arg("f_ext") = py::none())
        .def("end_effector_pose_hessian", &Runner::end_effector_pose_hessian,
             py::arg("q"))
        .def("idsva_so", &Runner::idsva_so,
             py::arg("q"), py::arg("qd"), py::arg("qdd") = py::none(),
             py::arg("second_order_tensor_size"),
             py::arg("gravity") = 9.81f)
        .def("fdsva_so", &Runner::fdsva_so,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("second_order_tensor_size"),
             py::arg("gravity") = 9.81f)
        .def("integrator", &Runner::integrator,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("dt"), py::arg("it") = 0, py::arg("gravity") = 9.81f)
        .def("integrator_gradient", &Runner::integrator_gradient,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("dt"), py::arg("it") = 0, py::arg("gravity") = 9.81f)
        // ─── grid_plant surface (G1) ──────────────────────────────────────
        .def("quadratic_state_cost", &Runner::quadratic_state_cost,
             py::arg("x"), py::arg("x_des"), py::arg("Q"))
        .def("quadratic_input_cost", &Runner::quadratic_input_cost,
             py::arg("u"), py::arg("u_des"), py::arg("R"))
        .def("joint_position_barrier", &Runner::joint_position_barrier,
             py::arg("var"), py::arg("lower"), py::arg("upper"), py::arg("mu"))
        .def("joint_velocity_barrier", &Runner::joint_velocity_barrier,
             py::arg("var"), py::arg("lower"), py::arg("upper"), py::arg("mu"))
        .def("joint_torque_barrier", &Runner::joint_torque_barrier,
             py::arg("var"), py::arg("lower"), py::arg("upper"), py::arg("mu"))
        .def("plant_step", &Runner::plant_step,
             py::arg("x"), py::arg("u"), py::arg("dt"),
             py::arg("it") = 0, py::arg("gravity") = 9.81f)
        .def("ee_pos_cost", &Runner::ee_pos_cost,
             py::arg("q"), py::arg("p_des"), py::arg("W"));
}
