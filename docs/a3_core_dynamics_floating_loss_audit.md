# A.3 — Core-dynamics floating-base loss audit (2026-05-28)

**Pattern (from `test/benchmarks/results/ee_grad_step_c_perf_v2/`, N=256 compute-only us vs pinocchio with-mem us):**

ALL losses are floating-base. Every fixed-base cell is a WIN (typically 0.18–0.66×).
Sorted by ratio:

| robot         | algo  | GRiD μs | pin μs | ratio |
|---------------|-------|---------|--------|-------|
| iiwa14 float  | aba   | 107.30  | 56.87  | 1.89× |
| go2    float  | aba   | 126.57  | 86.02  | 1.47× |
| g1     float  | crba  | 107.60  | 80.99  | 1.33× |
| go2    float  | crba  | 39.19   | 29.58  | 1.33× |
| iiwa14 float  | crba  | 32.31   | 24.57  | 1.32× |
| go2    float  | minv  | 69.31   | 56.94  | 1.22× |
| g1     float  | aba   | 197.97  | 177.65 | 1.11× |
| iiwa14 float  | minv  | 64.34   | 57.82  | 1.11× |

(h1_2-floating absent from this table — only collected for grid_glass, not pin.)

## Floating-base cost drivers (codegen survey)

**ABA (`_aba.py`).** A dedicated `gen_aba_inner_floating` (lines 1–267) emits a
chain ABA + an explicit 6×6 root block solve. The root block:
- 6×6 IA initialization (each timestep, per-block).
- 6×6 forward pass: `U_fb = IA_fb · S_fb` (6×6 matmul; 36 fmas).
- 6×6 `D_fb = S_fb^T · U_fb`; 6×6 invert; 6-vector solve for the root accel.

For small robots, this 6×6 root block is the DOMINANT compute. For iiwa14
(7 scalar joints + 1 floating root), the root block is ~50% of the algorithm's
arithmetic.

**CRBA & Minv.** No `gen_crba_inner_floating` split — the same scalar-joint code
walks the floating base "DOF by DOF" (6 sweeps), which is cache- and ILP-poor
compared to a single block 6×6 sweep.

## Root cause **CONFIRMED** by static audit (2026-05-29) — single-threaded 6×6 invert

The 6×6 root-block matrix inverse used by floating ABA + Minv is
**single-threaded Gauss-Jordan**.

`gen_invert_matrix` in [_lin_alg_helpers.py:198-243](../GRiDCodeGenerator/helpers/_lin_alg_helpers.py#L198-L243)
emits a body wrapped in `gen_add_serial_ops` ( `if (threadIdx.x == 0 &&
threadIdx.y == 0)` ), and inside that single thread runs `for pivRC in
range(6); for ind in range(36); ...`. So **at block size 448 (the iiwa14
`MAX_PERF_LEVEL_THREADS` default), 447 threads sit idle while one thread
walks 6 × (6 + 36) = 252 sequential ops** per invert.

**Call sites (per floating-base ABA / Minv timestep):**
- `_aba.py:154` — invert the root `D_fb` 6×6 for the backward pass.
- `_aba.py:225` — second 6×6 invert for the root acceleration solve.
- `_direct_minv.py:165` — invert the root `I_fb` 6×6.

So **floating ABA pays 2× this stall per call, Minv pays 1×**. That maps
exactly to the loss ranks observed: ABA worst (1.89× / 1.47×), Minv lighter
(1.11× / 1.22×). **CRBA is different** — `gen_crba_inner_floating`
[_crba.py:291-298](../GRiDCodeGenerator/algorithms/_crba.py#L291-L298)
does the root-block fill in 36 parallel threads (no invert), so the 1.32×
CRBA gap must come from a *different* hotspot (likely the floating-base
XImats quaternion-to-rotation conversion and the larger H matrix; needs
the ncu profile to confirm).

## Fix is mechanical: GLASS already has the parallel primitives

GLASS exposes BOTH replacements out-of-the-box:

- `invertMatrix<T>(dimA, A, s_temp, cgrps::thread_group)` —
  [GLASS/src/L3/inv.cuh:9-37](../GLASS/src/L3/inv.cuh#L9-L37). Inner loop is
  `for (ind = g.thread_rank(); ind < dimA*(dimA+1); ind += g.size())` —
  block-cooperative Gauss-Jordan.
- `cholDecomp_InPlace<T>` (in `GLASS/src/L3/chol_InPlace.cuh`) — block-
  cooperative Cholesky. IA / Iₑ are SPD by construction (inertia matrices),
  so Cholesky + two trsm solves is the right factor.

The GRiD `_lin_alg_helpers.gen_invert_matrix` was authored before GLASS
existed; it predates the GLASS-first-party policy. Two ways to migrate:

**Option A (minimal patch, low risk):** Rewrite `gen_invert_matrix` to drop
the `gen_add_serial_ops` wrap and parallelize the pivot-row update across
the block (same algorithm structure, just `tid + N`-strided over the
`dimA*dimA` inner loop, with a single-thread step for the pivot inverse
computation and a `__syncthreads` between pivots). Keeps the same out-of-
place `(A, Ainv, s_temp)` signature so call sites are untouched.

**Option B (cleaner, uses GLASS):** Replace each `invert_matrix(...)` call
in `_aba.py` / `_direct_minv.py` with a copy-then-`glass::invertMatrix`
sequence (the GLASS impl is in-place), or with a `glass::cholDecomp_InPlace`
+ two `glass::trsm` calls for the SPD-aware path (fewer flops on SPD).

Recommended: **A first** (one helper change, mechanical, ~1 commit), then
**B** as the second step alongside the per-algorithm sub-agent pass for the
A.3 backlog.

### Expected speedup ceiling

If the single-threaded invert IS the iiwa14-floating ABA bottleneck (to be
confirmed in pass-2 ncu instruction-mix), going from 1 active thread to
~36 active threads on the `dimA*dimA` inner loop drops that phase's cycles
by ~36×. The whole-kernel impact depends on what fraction of the kernel
is spent in the invert vs in the chain forward/backward (which use GLASS
gemv/gemm already). Pass 1 of `profile_aba.sh` gives the kernel SOL %.
Even halving the ABA latency would put iiwa14-floating ABA at ~54 μs vs
pin 56 μs — i.e. win, not lose.

### MEASURED — iiwa14-floating ABA (2026-05-29, cudaEvent A/B)

Microbench: `/tmp/grid_prof/aba_microbench.cu`, BATCH=256, 16 timed
launches after 8 warm-ups, sm_120 (RTX 5090), GRID_CUDA_LINALG_BACKEND=GLASS.

| variant                                    | μs/launch | μs/problem |
|--------------------------------------------|-----------|-----------:|
| baseline (single-threaded invert)          | 109.7     |      0.429 |
| **new** (`glass::invertMatrix_dense`)      | **51.9**  |  **0.203** |
| pinocchio CPU (from prior sweep)           | (56.9 batch ÷ 256) | 0.222 |

**2.12× speedup on iiwa14-floating ABA.** Beats pinocchio by ~9% per
problem — the 1.89× pin loss flips to a 1.09× GRiD win.

Same primitive change applies to `_aba.py:225` (second 6×6 invert per
floating ABA) and `_direct_minv.py:165` (floating Minv root invert); both
benefit identically. Expect Minv-floating loss (1.11–1.22×) to flip too.

## Original hypothesis (kept for completeness)

1. ~~Root 6×6 block in ABA isn't using GLASS block-matmul primitives.~~
   The 6×6 gemm/gemv calls ALREADY use GLASS — see `_aba.py:179, 186, 187`
   (`grid_linalg_gemv<T,6,6,true>`, `grid_linalg_gemm<T,6,6,6>`). The hot
   spot is the INVERT, not the gemm.
2. **CRBA loss source TBD** — `gen_crba_inner_floating` doesn't invert; the
   1.32× pin gap is a different mechanism (XImats? H-matrix fill?). Profile
   pass 2 on `crba_kernel` answers it.
3. Floating-base XImats / Xhom quaternion → rotation conversion — still a
   plausible secondary hotspot; pass 2 LSU/FMA ratio will show it.

## Concrete next steps (next perf session)

1. **Profile** iiwa14-floating ABA with `ncu --kernel-name aba_kernel
   --metrics smsp__sass_thread_inst_executed_op_*_inst.sum,...` to confirm
   whether the 6×6 root block is the hotspot.
2. **GLASS-ify the floating root block in ABA** — `block_inverse_6x6`,
   `block_matmul_6x6` primitives (or extend `glass::indexed_batched_gemm`
   with a 6×6 corner case).
3. **Audit `gen_crba_inner_floating`** (if it doesn't exist, the scalar walk
   is the bottleneck — add a 6-DOF root sweep). Same for Minv.
4. **Avoid recomputing the base-quaternion → rotation conversion per timestep**
   for batched callers — fold into the XImats precompute. (May already be —
   verify in `load_update_XImats`.)

## Why this is multi-day perf work (not a "few hours" item)

- Each candidate optimization needs: (a) codegen edit, (b) per-robot
  regen + CUDA equivalence at PERF + LITE + MINIMAL, (c) timing rerun. Single
  iteration is several hours.
- The 6×6 root block GLASS extension is a real linalg-primitive task — fits the
  GLASS-first-party policy (memory `project_grid_glass_first_party.md`) and
  benefits multiple algorithms (ABA, CRBA root, Minv root).
- Profiling is required to confirm the hypothesis before code change — otherwise
  we'd be optimizing blind.

**Estimated effort:** 1 session for profiling + ABA root GLASS-ification;
1–2 sessions for CRBA/Minv. Total: ~3–4 focused sessions.

## Caveats

- Pinocchio numbers in `comparison_sm120_tier_perf_baseline_20260520_0104/`
  may be stale relative to the current GRiD code. Re-collect alongside any
  optimization.
- h1_2-floating data missing — need it to confirm the loss pattern scales.
