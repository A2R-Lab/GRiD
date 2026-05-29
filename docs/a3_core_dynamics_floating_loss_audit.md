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

## Hypothesis (needs profiling to confirm)

1. **Root 6×6 block in ABA isn't using GLASS block-matmul primitives** — it's
   coded as scalar element fmas. Pinocchio's Eigen-emitted root block uses LDLT
   with unrolled fixed-size kernels.
2. **CRBA/Minv floating root walks the 6 DOFs serially** instead of treating the
   floating joint as a single 6-DOF unit. Each scalar walk pays full chain-walk
   overhead.
3. **Floating-base XImats / Xhom load** computes the floating base's body-to-
   world quaternion-to-rotation map; this is more expensive than a single-axis
   joint and may dominate for small robots.

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
