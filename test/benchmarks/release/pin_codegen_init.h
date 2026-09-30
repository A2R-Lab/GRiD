// CppADCodeGen settings that determine the CONTENT of the generated Pinocchio
// libraries (cg_rnea_eval_*.so, cg_partial_rnea_eval_*.so, cg_minv_eval_*.so).
//
// Kept in its own header so the JIT-library cache key hashes exactly the
// inputs that change generated code (this file, the generator classes in
// timePinocchio.cpp, the compiler, the Pinocchio version) and NOT the timing
// bridge around them — a change to how batches are dispatched must not throw
// away minutes of G1 code generation.
#pragma once
#include <limits>

template <class Generator>
void init_release_codegen(Generator &gen) {
    gen.initLib();
    // CppADCodeGen defaults to digits10 (six for float), which is NOT enough
    // to round-trip model constants. Preserve fp32 values exactly; this
    // changes decimal serialization, not the arithmetic precision.
    gen.codeGenerator().setParameterPrecision(std::numeric_limits<float>::max_digits10);
    gen.loadLib(true, "/usr/bin/gcc", "-O3");
}
