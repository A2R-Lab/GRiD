#include <pinocchio/codegen/code-generator-algo.hpp>

// Derived class to expose protected members of CodeGenRNEADerivatives
template<typename T>
class DerivedCodeGenRNEADerivatives : public pinocchio::CodeGenRNEADerivatives<T> {
public:
    // Inherit constructor from base class
    using pinocchio::CodeGenRNEADerivatives<T>::CodeGenRNEADerivatives;


    // Getter for dtau_dq
    const typename pinocchio::CodeGenRNEADerivatives<T>::MatrixXs & getDtauDq() const {
        return this->dtau_dq;
    }

    // Getter for dtau_dv
    const typename pinocchio::CodeGenRNEADerivatives<T>::MatrixXs & getDtauDv() const {
        return this->dtau_dv;
    }
};
