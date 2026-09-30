#ifndef GRID_RUNNER_SELECT_CUH
#define GRID_RUNNER_SELECT_CUH
//
// Per-algorithm COMPILE selection for the correctness runners (split, not monolith).
//
// The harness compiles ONE algorithm per translation unit: it passes
//   -DGRID_RUN_SPLIT  -DRUN_<ALGO>=1
// for the single selected algo. In that mode GRID_RUN_DEFAULT is 0, so every
// other RUN_<ALGO> defaults to 0 and only the selected algo's launch/host/dump
// blocks compile and run. A build break (or missing codegen dependency) in algo Y
// can therefore never void validation of algo X: X's TU never references Y.
// This retires the "one TU compiles every algorithm" coverage void (Bug A,
// 2026-06-17) where a crba_inner build break masked a forward_dynamics VALUE bug.
//
// With NO -DGRID_RUN_SPLIT (default; local all-in-one runs) GRID_RUN_DEFAULT is 1,
// so every RUN_<ALGO> defaults to 1 and the runner builds its full algo set exactly
// as before — back-compat, byte-identical behavior.
//
// Each runner, right after `#include "grid.cuh"`, includes this header and then
// declares a default for each algorithm token it owns:
//     #include "grid_runner_select.cuh"
//     #ifndef RUN_INVERSE_DYNAMICS
//     #  define RUN_INVERSE_DYNAMICS GRID_RUN_DEFAULT
//     #endif
// and wraps that algo's blocks in `#if RUN_INVERSE_DYNAMICS ... #endif`.
//
#ifdef GRID_RUN_SPLIT
#  define GRID_RUN_DEFAULT 0
#else
#  define GRID_RUN_DEFAULT 1
#endif

#endif // GRID_RUNNER_SELECT_CUH
