#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "grid.cuh" // Path to the grid.cuh header
#include <cuda_runtime.h>

namespace py = pybind11;

// Helper: convert numpy array to raw ptr
template <typename T>
T* get_data_ptr(py::array_t<T, py::array::c_style> array) {
    return static_cast<T*>(array.request().ptr);
}

// Python wrapper class for GRiD
template <typename T>
class PyGRidData {
private:
    T gravity;
    dim3 dimms;
    grid::gridData<T>* grid_data;
    cudaStream_t* streams;
    grid::robotModel<T>* d_robot_model;

public:
    PyGRidData(py::array_t<T, py::array::c_style> q,
                py::array_t<T, py::array::c_style> qd,
                py::array_t<T, py::array::c_style> u,
                T g = static_cast<T>(9.81)) {
        gravity = g;
        dimms = dim3(grid::SUGGESTED_THREADS, 1, 1);
        streams = grid::init_grid<T>();
        d_robot_model = grid::init_robotModel<T>();
        grid_data = grid::init_gridData<T, 1>();

        load_joint_info(q, qd, u);
    }

    ~PyGRidData() {
        grid::close_grid<T>(streams, d_robot_model, grid_data);
    }

    void load_joint_info(py::array_t<T, py::array::c_style> q,
                          py::array_t<T, py::array::c_style> qd,
                          py::array_t<T, py::array::c_style> u) {
        if (q.size() != grid::NUM_JOINTS || qd.size() != grid::NUM_JOINTS || u.size() != grid::NUM_JOINTS) {
            throw std::runtime_error("Input arrays must be of size NUM_JOINTS");
        }
        
        T* q_ptr = get_data_ptr(q);
        T* qd_ptr = get_data_ptr(qd);
        T* u_ptr = get_data_ptr(u);
        
        for (int i = 0; i < grid::NUM_JOINTS; i++) {
            grid_data->h_q_qd_u[i] = q_ptr[i];
            grid_data->h_q_qd_u[i + grid::NUM_JOINTS] = qd_ptr[i];
            grid_data->h_q_qd_u[i + 2 * grid::NUM_JOINTS] = u_ptr[i];
        }
        
        gpuErrchk(cudaMemcpy(grid_data->d_q_qd_u, grid_data->h_q_qd_u,
                             3 * grid::NUM_JOINTS * sizeof(T), cudaMemcpyHostToDevice));
        gpuErrchk(cudaDeviceSynchronize());
    }

    // Get end effector positions
    py::array_t<T> get_end_effector_positions() {
        grid::end_effector_positions<T, false>(grid_data, d_robot_model, 1, dim3(1, 1, 1), dimms, streams);
        
        // Create numpy array from our data
        auto result = py::array_t<T>(6 * grid::NUM_EES);
        py::buffer_info buf = result.request();
        T* ptr = static_cast<T*>(buf.ptr);
        
        // Copy data from h_eePos to numpy array
        std::memcpy(ptr, grid_data->h_eePos, 6 * grid::NUM_EES * sizeof(T));
        
        return result;
    }

    // Get end effector position gradients
    py::array_t<T> get_end_effector_position_gradients() {
        grid::end_effector_positions_gradient<T, false>(grid_data, d_robot_model, 1, dim3(1, 1, 1), dimms, streams);
        
        // Create numpy array with shape (6, NUM_EES*NUM_JOINTS)
        std::vector<size_t> shape = {6, static_cast<size_t>(grid::NUM_EES * grid::NUM_JOINTS)};
        std::vector<size_t> strides = {sizeof(T), 6 * sizeof(T)};
        
        auto result = py::array_t<T>(shape, strides);
        py::buffer_info buf = result.request();
        T* ptr = static_cast<T*>(buf.ptr);
        
        // Copy data from h_deePos to numpy array
        std::memcpy(ptr, grid_data->h_deePos, 6 * grid::NUM_EES * grid::NUM_JOINTS * sizeof(T));
        
        return result;
    }

    py::array_t<T> inverse_dynamics() {
        grid::inverse_dynamics<T, false, false>(
            grid_data, d_robot_model, gravity, 1,
            dim3(1, 1, 1), dimms, streams
        );
        
        auto result = py::array_t<T>(grid::NUM_JOINTS);
        py::buffer_info buf = result.request();
        T* ptr = static_cast<T*>(buf.ptr);
        std::memcpy(ptr, grid_data->h_c, grid::NUM_JOINTS * sizeof(T));
        return result;
    }

    // Compute mass matrix inverse
    py::array_t<T> minv() {
        grid::direct_minv<T, false>(grid_data, d_robot_model, 1, dim3(1, 1, 1), dimms, streams);
        
        std::vector<size_t> shape = {
            static_cast<size_t>(grid::NUM_JOINTS), 
            static_cast<size_t>(grid::NUM_JOINTS)
        };
        auto result = py::array_t<T>(shape);
        py::buffer_info buf = result.request();
        T* ptr = static_cast<T*>(buf.ptr);
        
        std::memcpy(ptr, grid_data->h_Minv, grid::NUM_JOINTS * grid::NUM_JOINTS * sizeof(T));
        
        return result;
    }

    // Compute forward dynamics 
    py::array_t<T> forward_dynamics() {
        grid::forward_dynamics<T>(grid_data, d_robot_model, gravity, 1, dim3(1, 1, 1), dimms, streams);
        
        auto result = py::array_t<T>(grid::NUM_JOINTS);
        py::buffer_info buf = result.request();
        T* ptr = static_cast<T*>(buf.ptr);
        
        std::memcpy(ptr, grid_data->h_qdd, grid::NUM_JOINTS * sizeof(T));
        
        return result;
    }

    // Compute inverse dynamics gradient
    py::array_t<T> inverse_dynamics_gradient() {
        grid::inverse_dynamics_gradient<T, true, false>(grid_data, d_robot_model, gravity, 1, dim3(1, 1, 1), dimms, streams);
        
        // Create array for both dc_dq and dc_dqd
        std::vector<size_t> shape = {
            2,  // Two matrices: dc_dq and dc_dqd
            static_cast<size_t>(grid::NUM_JOINTS), 
            static_cast<size_t>(grid::NUM_JOINTS)
        };
        auto result = py::array_t<T>(shape);
        py::buffer_info buf = result.request();
        T* ptr = static_cast<T*>(buf.ptr);
        
        // Copy dc_dq (first NUM_JOINTS*NUM_JOINTS elements)
        std::memcpy(ptr, grid_data->h_dc_du, grid::NUM_JOINTS * grid::NUM_JOINTS * sizeof(T));
        
        // Copy dc_dqd (next NUM_JOINTS*NUM_JOINTS elements)
        std::memcpy(
            ptr + grid::NUM_JOINTS * grid::NUM_JOINTS, 
            &grid_data->h_dc_du[grid::NUM_JOINTS * grid::NUM_JOINTS], 
            grid::NUM_JOINTS * grid::NUM_JOINTS * sizeof(T)
        );
        
        return result;
    }

    // Compute forward dynamics gradient
    py::array_t<T> forward_dynamics_gradient() {
        grid::forward_dynamics_gradient<T, false>(grid_data, d_robot_model, gravity, 1, dim3(1, 1, 1), dimms, streams);
        
        // Create array for both df_dq and df_dqd
        std::vector<size_t> shape = {
            2,  // Two matrices: df_dq and df_dqd
            static_cast<size_t>(grid::NUM_JOINTS), 
            static_cast<size_t>(grid::NUM_JOINTS)
        };
        auto result = py::array_t<T>(shape);
        py::buffer_info buf = result.request();
        T* ptr = static_cast<T*>(buf.ptr);
        
        // Copy df_dq (first NUM_JOINTS*NUM_JOINTS elements)
        std::memcpy(ptr, grid_data->h_df_du, grid::NUM_JOINTS * grid::NUM_JOINTS * sizeof(T));
        
        // Copy df_dqd (next NUM_JOINTS*NUM_JOINTS elements)
        std::memcpy(
            ptr + grid::NUM_JOINTS * grid::NUM_JOINTS, 
            &grid_data->h_df_du[grid::NUM_JOINTS * grid::NUM_JOINTS], 
            grid::NUM_JOINTS * grid::NUM_JOINTS * sizeof(T)
        );
        
        return result;
    }
};

// Pybind11 module
PYBIND11_MODULE(gridCuda, m) {
    m.doc() = "Python bindings for CUDA GRiD dynamics";
    
    py::class_<PyGRidData<float>>(m, "GRidDataFloat")
        .def(py::init<py::array_t<float, py::array::c_style>,
                      py::array_t<float, py::array::c_style>,
                      py::array_t<float, py::array::c_style>,
                      float>(),
             py::arg("q"), py::arg("qd"), py::arg("u"), py::arg("gravity") = 9.81f)
        .def("load_joint_info", &PyGRidData<float>::load_joint_info,
             "Load joint positions, velocities, torques",
             py::arg("q"), py::arg("qd"), py::arg("u"))
        .def("get_end_effector_positions", &PyGRidData<float>::get_end_effector_positions,
            "Calculate end effector positions")
        .def("get_end_effector_position_gradients", &PyGRidData<float>::get_end_effector_position_gradients,
            "Calculate end effector position gradients")
        .def("inverse_dynamics", &PyGRidData<float>::inverse_dynamics,
             "Compute inverse dynamics")
        .def("minv", &PyGRidData<float>::minv,
            "Calculate mass matrix inverse")
        .def("forward_dynamics", &PyGRidData<float>::forward_dynamics,
            "Calculate forward dynamics (joint accelerations)")
        .def("inverse_dynamics_gradient", &PyGRidData<float>::inverse_dynamics_gradient,
            "Calculate inverse dynamics gradient")
        .def("forward_dynamics_gradient", &PyGRidData<float>::forward_dynamics_gradient,
            "Calculate forward dynamics gradient");
    
    // double compatability was breaking code.
    // py::class_<PyGRidData<double>>(m, "GRidDataDouble")
    //     .def(py::init<double>(), py::arg("gravity") = 9.81)
    //     .def("load_joint_info", &PyGRidData<double>::load_joint_info,
    //          "Load joint positions, velocities, torques",
    //          py::arg("q"), py::arg("qd"), py::arg("u"))
    //     .def("inverse_dynamics", &PyGRidData<double>::inverse_dynamics,
    //          "Compute inverse dynamics");
    
    m.attr("NUM_JOINTS") = grid::NUM_JOINTS;
    m.attr("NUM_EES") = grid::NUM_EES;
}
