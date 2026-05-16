// Python binding for pinocchio::ComputeRNEASecondOrderDerivatives.
//
// Exposes a single function `compute_rnea_second_order(urdf_path, floating, q, v, a)`
// returning the four second-order RNEA tensors as (nv, nv, nv) numpy arrays.
// Pinocchio's internal Tensor3x is column-major (Eigen default); we return arrays
// already laid out as numpy default (row-major) by transposing on copy.

#include <pinocchio/algorithm/rnea-second-order-derivatives.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/multibody/joint/joint-free-flyer.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

namespace py = pybind11;

static py::array_t<double>
tensor_to_numpy(const pinocchio::Data::Tensor3x &t)
{
    const auto &dims = t.dimensions();
    const ssize_t n0 = dims[0];
    const ssize_t n1 = dims[1];
    const ssize_t n2 = dims[2];
    py::array_t<double> out({n0, n1, n2});
    auto *src = t.data();
    auto buf = out.mutable_unchecked<3>();
    for (ssize_t i = 0; i < n0; ++i)
        for (ssize_t j = 0; j < n1; ++j)
            for (ssize_t k = 0; k < n2; ++k)
                buf(i, j, k) = src[i + j * n0 + k * n0 * n1];
    return out;
}

static py::tuple
compute_rnea_second_order(const std::string &urdf_path,
                          bool floating_base,
                          const Eigen::VectorXd &q,
                          const Eigen::VectorXd &v,
                          const Eigen::VectorXd &a)
{
    pinocchio::Model model;
    if (floating_base) {
        pinocchio::JointModelFreeFlyer root_joint;
        pinocchio::urdf::buildModel(urdf_path, root_joint, model);
    } else {
        pinocchio::urdf::buildModel(urdf_path, model);
    }

    if (q.size() != model.nq)
        throw std::invalid_argument("q size mismatch: expected " + std::to_string(model.nq));
    if (v.size() != model.nv)
        throw std::invalid_argument("v size mismatch: expected " + std::to_string(model.nv));
    if (a.size() != model.nv)
        throw std::invalid_argument("a size mismatch: expected " + std::to_string(model.nv));

    pinocchio::Data data(model);
    pinocchio::ComputeRNEASecondOrderDerivatives(model, data, q, v, a);

    return py::make_tuple(
        tensor_to_numpy(data.d2tau_dqdq),
        tensor_to_numpy(data.d2tau_dvdv),
        tensor_to_numpy(data.d2tau_dqdv),
        tensor_to_numpy(data.d2tau_dadq));
}

PYBIND11_MODULE(pin_so_ext, m)
{
    m.doc() = "Pinocchio second-order RNEA derivatives binding.";
    m.def("compute_rnea_second_order", &compute_rnea_second_order,
          py::arg("urdf_path"), py::arg("floating_base"),
          py::arg("q"), py::arg("v"), py::arg("a"),
          "Compute (d2tau/dqdq, d2tau/dvdv, d2tau/dqdv, d2tau/dadq) tensors of shape (nv, nv, nv).");
}
