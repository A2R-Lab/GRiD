#ifndef CODEGENRNEAWITHGETRES_HPP
#define CODEGENRNEAWITHGETRES_HPP

#include <pinocchio/codegen/code-generator-algo.hpp>

template<typename Scalar>
class CodeGenRNEAWithGetRes : public pinocchio::CodeGenRNEA<Scalar>
{
public:
  // Inherit the constructor from the base class
  using pinocchio::CodeGenRNEA<Scalar>::CodeGenRNEA;

  // Public getter function to access the protected 'res' member
  const typename pinocchio::CodeGenRNEA<Scalar>::VectorXs & getRes() const
  {
    return this->res;
  }
};

#endif // CODEGENRNEAWITHGETRES_HPP
