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
//                     float* c_out, int batch, float gravity);
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

namespace py = pybind11;


// ─── C ABI function signatures (must match wrapper_template.cu) ──────────────

extern "C" {
    using fn_int_v_t        = int (*)();
    using fn_int_i_t        = int (*)(int);
    // q, qd, qdd_opt, out, batch, gravity
    using fn_rnea_t         = int (*)(const float*, const float*, const float*,
                                      float*, int, float);
    // q, out, batch
    using fn_minv_t         = int (*)(const float*, float*, int);
    // q, qd, u, out, batch, gravity   — fd, aba, fd_grad
    using fn_fd_t           = int (*)(const float*, const float*, const float*,
                                      float*, int, float);
    // q, out, batch, gravity          — crba
    using fn_crba_t         = int (*)(const float*, float*, int, float);
    // q, out, batch                   — ee_pose, ee_pose_gradient, ee_pose_hessian
    using fn_ee_t           = int (*)(const float*, float*, int);
    // q, qd, qdd_opt, out, batch, gravity   — idsva_so (same as rnea)
    // q, qd, u, out, batch, gravity         — fdsva_so (same as fd)
    // q, qd, u, out, batch, dt, it          — integrator, integrator_gradient
    //   (dt is the runtime timestep; it selects the IntegratorType; gravity is
    //    the standard 9.81 constant baked in the wrapper)
    using fn_integrator_t   = int (*)(const float*, const float*, const float*,
                                      float*, int, float, float, int);
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
        fn_idsva_so_         = reinterpret_cast<fn_rnea_t>(require_sym("grid_rbd_idsva_so"));
        fn_fdsva_so_         = reinterpret_cast<fn_fd_t>  (require_sym("grid_rbd_fdsva_so"));
        fn_integrator_       = reinterpret_cast<fn_integrator_t>(require_sym("grid_rbd_integrator"));
        fn_integrator_grad_  = reinterpret_cast<fn_integrator_t>(require_sym("grid_rbd_integrator_gradient"));

        // Cache constants (avoid the indirect-function-call cost on every read).
        num_joints_ = fn_num_joints_();
        num_vel_    = fn_num_vel_();
        num_ees_    = fn_num_ees_();
        max_batch_  = fn_max_batch_();

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
        float gravity)
    {
        int batch = check_inputs_2d(q, qd, /*last_dim=*/num_joints_);
        const float* qdd_ptr = nullptr;
        if (!qdd_opt.is_none()) {
            auto qdd = qdd_opt.cast<
                py::array_t<float, py::array::c_style | py::array::forcecast>>();
            check_array_2d(qdd, batch, num_joints_, "qdd");
            qdd_ptr = qdd.data();
        }

        py::array_t<float> out({batch, num_joints_});
        int rc = fn_rnea_(q.data(), qd.data(), qdd_ptr,
                          out.mutable_data(), batch, gravity);
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
        float gravity)
    {
        int batch = check_inputs_2d(q, qd, /*last_dim=*/num_joints_);
        check_array_2d(u, batch, num_joints_, "u");

        py::array_t<float> out({batch, num_joints_});
        int rc = fn_fd_(q.data(), qd.data(), u.data(),
                        out.mutable_data(), batch, gravity);
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
        float gravity)
    {
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        py::array_t<float> out({batch, num_joints_});
        int rc = fn_aba_(q.data(), qd.data(), u.data(),
                         out.mutable_data(), batch, gravity);
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
        py::array_t<float> out({batch, 6 * num_ees_, num_joints_});
        int rc = fn_ee_pose_grad_(q.data(), out.mutable_data(), batch);
        if (rc != 0) throw std::runtime_error("grid_rbd_end_effector_pose_gradient failed: rc=" + std::to_string(rc));
        return out;
    }

    // ─── rnea_grad / forward_dynamics_grad ───────────────────────────────────
    py::array_t<float> rnea_grad(
        py::array_t<float, py::array::c_style | py::array::forcecast> q,
        py::array_t<float, py::array::c_style | py::array::forcecast> qd,
        py::object qdd_opt,
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
        py::array_t<float> out({batch, num_joints_, 2 * num_joints_});
        int rc = fn_rnea_grad_(q.data(), qd.data(), qdd_ptr,
                               out.mutable_data(), batch, gravity);
        if (rc != 0) throw std::runtime_error("grid_rbd_rnea_grad failed: rc=" + std::to_string(rc));
        return out;
    }

    py::array_t<float> forward_dynamics_grad(
        py::array_t<float, py::array::c_style | py::array::forcecast> q,
        py::array_t<float, py::array::c_style | py::array::forcecast> qd,
        py::array_t<float, py::array::c_style | py::array::forcecast> u,
        float gravity)
    {
        int batch = check_inputs_2d(q, qd, num_joints_);
        check_array_2d(u, batch, num_joints_, "u");
        py::array_t<float> out({batch, num_joints_, 2 * num_joints_});
        int rc = fn_fd_grad_(q.data(), qd.data(), u.data(),
                             out.mutable_data(), batch, gravity);
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
        py::array_t<float> out({batch, 6 * num_ees_, num_joints_, num_joints_});
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

    void* handle_ = nullptr;

    fn_int_v_t fn_num_joints_ = nullptr;
    fn_int_v_t fn_num_vel_    = nullptr;
    fn_int_v_t fn_num_ees_    = nullptr;
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
    fn_rnea_t  fn_idsva_so_       = nullptr;
    fn_fd_t    fn_fdsva_so_       = nullptr;
    fn_integrator_t fn_integrator_      = nullptr;
    fn_integrator_t fn_integrator_grad_ = nullptr;

    int num_joints_ = 0;
    int num_vel_    = 0;
    int num_ees_    = 0;
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
             py::arg("gravity") = 9.81f)
        .def("minv", &Runner::minv,
             py::arg("q"))
        .def("forward_dynamics", &Runner::forward_dynamics,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("gravity") = 9.81f)
        .def("aba", &Runner::aba,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("gravity") = 9.81f)
        .def("crba", &Runner::crba,
             py::arg("q"), py::arg("gravity") = 9.81f)
        .def("end_effector_pose", &Runner::end_effector_pose,
             py::arg("q"))
        .def("end_effector_pose_gradient", &Runner::end_effector_pose_gradient,
             py::arg("q"))
        .def("rnea_grad", &Runner::rnea_grad,
             py::arg("q"), py::arg("qd"), py::arg("qdd") = py::none(),
             py::arg("gravity") = 9.81f)
        .def("forward_dynamics_grad", &Runner::forward_dynamics_grad,
             py::arg("q"), py::arg("qd"), py::arg("u"),
             py::arg("gravity") = 9.81f)
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
             py::arg("dt"), py::arg("it") = 0, py::arg("gravity") = 9.81f);
}
