# GLASS RFC: round 2 — auto-dispatch gaps + new feature wishes

**Author:** Claude (GRiD-A2R integration agent)
**Date:** 2026-05-14
**Predecessor work:** `glass-rfc-batched-1d.md` (delivered as GLASS commit
`0861724` — `gemm_batched_1d` + `gemm_strided_batched_1d` + auto-dispatch
for `gemm<>` via `should_use_cublasdx<>` + `tuning_table.cuh`).

## TL;DR

The first RFC's `gemm<>` auto-dispatch (`should_use_cublasdx<>` +
SIMT fallback) is exactly the right pattern. **GRiD now has 4 other
glass::nvidia APIs that still need the same treatment plus 2-3 genuinely
new features that would simplify our codegen.** Bundling everything
into one round.

## Auto-dispatch gaps (mirror the gemm<> work)

After the batched-1d PR, only `glass::nvidia::gemm<>` auto-dispatches.
The following also need it — same pattern (`if constexpr (!should_use_cublasdx*<>) → ::glass::*`,
else require `DEFINE_NVIDIA_*` macro).

### Gap A — `glass::nvidia::gemv<>` (P0)

Used implicitly via the GRiD `grid_linalg_gemv` wrapper. No GRiD
shapes today exceed where cuBLASDx would win, but we'd like to drop
the codegen-time backend hardcode and let GLASS pick.

```cpp
template <typename T, uint32_t M, uint32_t N,
          uint32_t BLOCK_THREADS = 0, ..., uint32_t SM_VAL = SMS>
__device__ void gemv(T alpha, T* A, T* x, T beta, T* y, char* smem) {
    if constexpr (!should_use_cublasdx_gemv<T, M, N, SM_VAL>()) {
        ::glass::gemv<T, M, N>(alpha, A, x, beta, y);
        return;
    }
    // cuBLASDx specialization (or static_assert if no DEFINE_NVIDIA_GEMV)
}
```

Needs `should_use_cublasdx_gemv<T, M, N, SM_VAL>` (mirror
`should_use_cublasdx<>`). Fallback heuristic suggestion:
`max(M, N) >= 32` (GEMV's compute density is lower than GEMM so the
SIMT-vs-cuBLASDx tipping shape should be larger).

### Gap B — `glass::nvidia::row_strided_gemv<>` (P0)

Same as Gap A but with the additional `ROW_STRIDE` template arg.
SIMT fallback calls `::glass::row_strided_gemv<T, M, N, ROW_STRIDE>`.

**This is the one GRiD uses every codegen run today** (6×6 GEMV in
ID and ABA inner loops). Currently the GRiD wrapper hardcodes the
SIMT path; with this auto-dispatch we can collapse to a one-line wrapper.

### Gap C — `glass::nvidia::row_strided_gemm<>` (P0)

GRiD uses this with shape (6,6,6) via `grid_linalg_row_strided_gemm`.
Same auto-dispatch pattern needed. Fallback heuristic same as GEMM
(`max(M,N,K) >= 16 AND min(M,N,K) >= 4`).

### Gap D — `glass::nvidia::gemm<>` with TRANSPOSE_B=true (P1)

The auto-dispatching `gemm<>` primary template currently has
`/*TRANSPOSE_B=*/false` hardcoded in the SIMT call. For
`TRANSPOSE_B=true`, the template falls through to cuBLASDx
unconditionally — no SIMT fallback path.

GRiD has 6 call sites that use TRANSPOSE_B=true (all 6×6×6,
in `_crba.py` and `_aba.py`):

```cpp
grid_linalg_gemm<T,6,6,6,/*TRANSPOSE_B=*/true,false>(...);
```

These currently always pay the cuBLASDx tax. Want: SIMT fallback
for transposed-B too, using `::glass::gemm<T,M,N,K, /*TRANSPOSE_B=*/true, ...>`.

## New features (not just auto-dispatch)

These don't exist yet — would be wins to add while you're in there.

### NEW-1 — `gemm_batched_1d` with per-pointer A AND cuBLASDx fallback (P1)

`gemm_batched_1d` (P0-1 from previous RFC) is SIMT-only today. For
larger shapes / larger BATCH it would benefit from cuBLASDx. Pattern:
mirror the auto-dispatch in `gemm<>` to `gemm_batched_1d<>` using a
new `should_use_cublasdx_batched<T,M,N,K,BATCH,SM>` decision (which
may have different tipping shape than single-GEMM `should_use_cublasdx<>`
since per-element compute density × BATCH changes the calculus).

GRiD use case: the ee_pose_gradient gradient chain (`s_deeTemp`)
currently keeps a SIMT parallel_loop because A is djid-dependent
(per-batch A pointer). If `gemm_batched_1d` (pointer-array variant)
got auto-dispatch + reasonable perf for larger shapes, we could batch
that path too. Today the SIMT loop ships everywhere.

### NEW-2 — `required_smem_for_dispatch<T,M,N,K,...>()` constexpr (P1)

Returns 0 if `should_use_cublasdx<...>()` returns false (SIMT needs
no smem); else returns `gemm_smem_size<...>()` (cuBLASDx scratch size).

GRiD codegen currently allocates `s_linalg_smem` (cuBLASDx scratch)
in every kernel for glass-nvidia builds — even kernels where every
GEMM call routes to SIMT. With this helper, codegen could right-size
or skip the smem allocation per kernel.

### NEW-3 — `print_dispatch_full<T,M,N,K>()` host helper (P2)

Existing `print_dispatch<T,M,N,K>()` says "SIMT" or "cuBLASDx".
Extended version dumps: which decision rule fired (tuning_table
specialization vs heuristic), what the heuristic would have said,
what shape the tuning_table has the nearest measurement for, expected
relative speedup. Useful for `bench/autotune.py` consumers debugging
why a specific shape went one way.

### NEW-4 — `chol_solve<T,N,NRHS>` SIMT path (P2)

`glass::nvidia::chol_inplace<>` (used by GRiD's optional cuSOLVERDx
path) has no SIMT fallback. For the planned FD-solve A/B benchmark
(direct_minv+gemv vs crba+posv) we want to test small-N Cholesky on
GPU; cuSOLVERDx is heavyweight to compile and tuned for large N. A
SIMT `chol_solve<T,N,NRHS>` for N≤16 would unblock that A/B without
forcing every consumer to pull in cuSOLVERDx.

(Lower priority — only matters if we revive the FD-solve A/B work.)

## Out of scope for this RFC

- Tensor-core paths for fp16/bf16 (GRiD is float32-only today).
- Multi-stream / async APIs (GRiD is single-stream).
- Warp-resident GEMM without `__syncthreads()` (the P2-6 you already
  evaluated and dropped — your analysis showed SIMT wins; agreed).

## How GRiD consumes round 2 once landed

After Gap A-D + NEW-1, the GRiD `_lin_alg_helpers.py` wrapper becomes
near-trivial: every wrapper is a 3-line `#if GRID_CUDA_USE_GLASS_NVIDIA`
glass::nvidia::* call, `#else` ::glass::* SIMT call, `#endif`. No
runtime dispatch, no codegen-time shape thresholds (those live in
GLASS's tuning_table).

GRiD delete-list after round 2:
- The "Phase 5a workaround" comments in `grid_linalg_row_strided_gemv`
  (always-SIMT hardcode).
- The `grid_linalg_packed_gemm_nvidia_transb` separate function
  (subsumed by `gemm<TRANSPOSE_B=true>` auto-dispatch — Gap D).
- The codegen-time per-call backend choice (already removed in
  Phase 5a — `linalg_smem_for()` was deleted).

After NEW-2:
- The unconditional `s_linalg_smem` setup in every glass-nvidia kernel
  becomes conditional — only emitted when at least one call in the
  kernel actually needs cuBLASDx scratch.

## Validation criteria (per item)

For each of A/B/C/D:
1. `glass::nvidia::<api><T,M,N,K,...>` for a small shape compiles
   without a `DEFINE_NVIDIA_*` macro and routes to SIMT.
2. Same for a larger shape that's in the tuning table → routes to
   cuBLASDx, requires the appropriate DEFINE macro.
3. Numerical equivalence between SIMT and cuBLASDx paths on iiwa14 ID
   or ABA outputs.
4. `print_dispatch*<T,M,N,K>()` reports correctly.

For NEW-1: bench `gemm_batched_1d<float,8,8,8,BATCH=32>` SIMT vs
cuBLASDx on sm_120 and sm_86 to validate the tipping point.

For NEW-2: GRiD's `init_grid()` for iiwa14_fixed glass-nvidia should
allocate ~0 bytes of `s_linalg_smem` (every shape routes to SIMT on
sm_120); for go2_fixed should allocate the cuBLASDx requirement (some
shapes route to cuBLASDx per tuning_table).

## Effort estimate

- Gap A/B/C: each is a near-copy of the GEMM auto-dispatch pattern in
  `query_simt.cuh` + `l3.cuh` + `tuning_table.cuh`. Should be quick.
- Gap D: smaller change to existing `gemm<>` template (lift the
  `TRANSPOSE_B=false` hardcode in the SIMT branch).
- NEW-1: largest item — touches `l3_simt.cuh`, adds cuBLASDx-batched
  specialization. Probably comparable to original P0-1.
- NEW-2: small constexpr helper.
- NEW-3: small.
- NEW-4: largest if pursued; skip if FD-solve A/B isn't on roadmap.

## What's still NOT a feature request

- The cicc -O3 hang on sm_8x (worked around via `--cicc-opt-level 2`
  at the GRiD nvcc invocation; investigated, may be Ampere-only nvcc
  bug — separate from GLASS).
- The ee_pose_gradient single-call cicc -O2 LICM elision (anti-LICM
  machinery is in place, batch path is correct; documented limitation).
