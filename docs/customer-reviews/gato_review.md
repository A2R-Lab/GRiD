# GRiD → GATO consulting review

**Customer:** GATO (`/home/plancher/Desktop/GATO`, branch `main`) — a batched-SQP ("bsqp")
GPU MPC solver for manipulators (iiwa14, indy7).
**Reviewer scope:** read-only. Assess how GATO consumes GRiD today, whether GRiD's *current*
algorithm + `grid_plant` surfaces drop in, where `grid_plant` can delete GATO code, where
GLASS can speed up GATO's linear algebra, and other perf wins.
**Date:** 2026-06-01. GRiD reference revision: `modernizing-tests`.

---

## Executive summary (ranked by impact / effort)

1. **GATO is pinned to a STALE, hand-vendored GRiD header and a hand-rolled plant.** Its
   `gato/dynamics/iiwa14/iiwa14_grid.cuh` is an old `grid.cuh` that predates the current
   `grid_plant` namespace, `IntegratorType`, `integrator[_gradient]_device`, `com_device`,
   `ccrba_device`, `frame_jacobian`, and `osc_inertia`. GATO's entire
   `iiwa14_plant.cuh` (457 lines) + `dynamics/integrator.cuh` (259 lines) is hand-written
   against the *low-level* `grid::*_inner` API with **hardcoded magic offsets**
   (`s_XITemp[504]`, `s_vaf[126]`, `s_dc_du[98]`, `s_Minv[49]`). **Highest-value, lowest-risk
   move: regenerate the header with current GRiD and adopt the `grid_plant` primitives.**
   This is the biggest code-deletion opportunity (item 3) and removes a class of
   silently-wrong-on-robot-change bugs. *Impact: high. Effort: low–medium.*

2. **Adopt `grid_plant`'s cost/barrier/step primitives to delete ~600 LoC of hand-rolled
   plant + cost machinery.** GATO's `trackingcost`, `trackingCostGradientAndHessian`,
   `jointBarrier{,Gradient,Hessian}`, `forwardDynamics`, `forwardDynamicsAndGradient`,
   `integrator_*`, `sim_step`, `compute_linearized_dynamics` are near-exact analogues of
   `grid_plant`'s `quadratic_*_cost`, `ee_pos_cost`, `joint_*_barrier`, `plant_step`,
   `plant_step_gradient`. The convention matches (state `x=[q;qd]`, GN Hessian, ½-quadratic,
   log-barrier). *Impact: high. Effort: medium (it's a real cutover, see §3 gaps).*

3. **Replace GATO's `block::matMul` family + `block::invertMatrix` with GLASS primitives**
   in `schur_linsys.cuh` (17 call sites). GATO's GEMMs are the naive
   one-thread-per-output-element, serial-K-loop form; GLASS ships the same semantics plus a
   tiled/compile-time-dim path, strided/batched variants, and Cholesky/TRSM. The Schur
   `theta_k` inversion (STATE_SIZE×STATE_SIZE, SPD) is a textbook fit for
   `glass::cholDecomp_InPlace` + `trsm` instead of full Gauss-Jordan inverse. *Impact:
   medium–high (Schur is on the hot SQP path). Effort: medium.*

4. **GATO leaves GRiD's analytic 2nd-order + OSC/centroidal surfaces entirely on the table.**
   `idsva_so`/`fdsva_so` are even *present in the vendored header* but **no GATO code calls
   them** — the bsqp Hessian is Gauss-Newton only. Exact-Newton steps via `fdsva_so`, and
   contact/task-space costs via `frame_jacobian`/`osc_inertia`/`ccrba`, are available now.
   *Impact: medium (solver quality). Effort: medium–high (research, not just plumbing).*

5. **Several concrete perf/correctness pitfalls** (see §5): `block::reduce` serial tail,
   merit kernel launched at `grid::SUGGESTED_THREADS` (a constant GRiD is actively
   deprecating), `matMul<1,N,N>` for matrix-vector products that should be `gemv`, the
   recompute of `direct_minv` inside `forwardDynamicsAndGradient`, and **no
   `cudaGetLastError()` after the opt-in kernels** — exactly the silent-launch-failure class
   GRiD's debugging guide calls out.

---

## Q1 — How GATO uses GRiD today

### Touchpoints

| GATO file | GRiD surface consumed | How |
|---|---|---|
| `gato/dynamics/iiwa14/iiwa14_grid.cuh` (9123 lines) | the whole `namespace grid` | **vendored copy of an old generated `grid.cuh`** |
| `gato/dynamics/iiwa14/iiwa14_fext.cuh` (510) | f_ext RNEA variants | hand-written external-wrench inverse dynamics in `namespace grid` |
| `gato/dynamics/iiwa14/iiwa14_plant.cuh` (457) | `grid::load_update_XImats_helpers`, `direct_minv_inner`, `inverse_dynamics_inner[_vaf]`, `inverse_dynamics_gradient_inner`, `forward_dynamics_finish`, `forward_dynamics_inner`, `end_effector_pose[_gradient]_device`, `init/free_robotModel` | hand-written plant composing the **low-level `_inner`** API |
| `gato/dynamics/integrator.cuh` | composes `forwardDynamics[AndGradient]` from the plant | hand-written integrators + linearized-dynamics |
| `gato/bsqp/kernels/setup_kkt.cuh` | `compute_linearized_dynamics`, `trackingCostGradientAndHessian[_lastblock]` | builds per-knot Q,R,q,r,A,B,c |
| `gato/bsqp/kernels/merit.cuh` | `plant::trackingcost`, `compute_integrator_error` | merit = cost + μ·‖defect‖ |
| `gato/bsqp/kernels/sim.cuh` | `plant::sim_step` | rollout |
| `gato/types.cuh`, `gato/constants.h` | `grid::NUM_JOINTS`, `EE_POS_SIZE`, `NQ/NX/NU/NEE`, `*_DYNAMIC_SHARED_MEM_COUNT`, `SUGGESTED_THREADS`, `robotModel<T>` | dimension + handle plumbing |

### Conventions consumed
- **State layout `x=[q; qd]`**, control `u=τ`, `STATE_SIZE=2·NUM_JOINTS`, `CONTROL_SIZE=NUM_JOINTS`
  (`constants.h:8-11`). Fixed-base only (`STATE_SIZE/2` assumed == NUM_POS == NUM_VEL throughout
  `integrator.cuh`).
- **Gravity hardcoded** to `9.81` in `iiwa14_plant.cuh:25-28`, passed into every `grid::*_inner`.
- **`robotModel<T>` handle** allocated by `grid::init_robotModel<T>()` and carried as an opaque
  `void* d_GRiD_mem` (`bsqp.cuh:205`, `types.cuh ProblemInputs`).
- **EE tracking** via `grid::end_effector_pose[_gradient]_device`, Gauss-Newton Hessian
  `J_pᵀ W J_p` assembled by hand (`iiwa14_plant.cuh:388-423`).

### Key finding: GATO consumes the *low-level inner* API, not host wrappers
GATO does its own shared-memory arena layout and calls `grid::*_inner` with **hand-computed byte
offsets** (`iiwa14_plant.cuh:161-205`). This is exactly the surface GRiD's codegen-architecture doc
warns is unstable — those `504 / 126 / 98 / 49` constants are robot-specific and will silently
desync if the model or codegen layout changes. The whole `forwardDynamicsAndGradient` even
re-derives the FD-grad pipeline by hand (Minv → c → finish → vaf → id-gradient → Minv·dc/du fold),
duplicating what current GRiD emits as `forward_dynamics_gradient_device` /
`integrator_gradient_device`.

---

## Q2 — Will GRiD's CURRENT algorithms drop in?

**Convention compatibility: excellent. API/version compatibility: needs a regen + a cutover.**

What matches cleanly (no GATO change needed semantically):
- State `x=[q;qd]`, control `u`, gravity-as-arg, `robotModel<T>` handle, EE-pose layout, the
  ½-quadratic cost + GN-Hessian + log-barrier conventions — `grid_plant` was explicitly designed
  to match the GATO/PDDP references (see `_plant.py` docstring lines 9-34). **This is a deliberate
  fit, not a coincidence.**

Friction / mismatches to resolve:

1. **The vendored header is too old to expose the new surfaces.** Greps confirm GATO's
   `iiwa14_grid.cuh` has **zero** occurrences of `IntegratorType`, `integrator_gradient_device`,
   `com_device`, `ccrba_device`, `grid_plant`, `frame_jacobian`, `osc_inertia`. So "drop in the
   current `grid_plant`" first requires **regenerating the header** (`grid-generate iiwa14.urdf`
   with `integrator`, `ee_pose`, `ee_pose_gradient`, etc. enabled). Until then there's nothing to
   call.

2. **Integrator family mismatch.** GATO's `INTEGRATOR_TYPE` is an `unsigned` with
   `0=euler,1=semi-implicit,2=trapezoidal`, **default 2** (`integrator.cuh`). GRiD's `plant_step`
   takes `grid::IntegratorType IT = EULER`. The default differs (2 vs Euler) and GATO's
   *trapezoidal* `q_{k+1}=q+v·dt+½a·dt²` with its specific A/B blocks (`integrator_gradient_inner`,
   `integrator.cuh:78-200`) must exist as a matching `IntegratorType` enumerator in GRiD or GATO
   keeps its own integrator. **Action for GRiD: confirm the emitted `IntegratorType` set includes
   semi-implicit-Euler and trapezoidal with byte-matching A/B, or this is a gap (see §6).**

3. **`grid_plant::plant_step_gradient` has a very wide, caller-placed-scratch signature**
   (`_plant.py:90-139`: `s_df_du, s_dc_du, s_vaf, s_Minv, s_qdd, s_q_orig, …, d_workspace,
   d_temp_spill`). GATO's `compute_linearized_dynamics` hides all scratch behind one `s_temp`
   arena and a single `forwardDynamicsAndGradient`. Adopting the GRiD gradient means GATO's
   `setup_kkt` must allocate/placement those buffers. Workable, but it's the main porting cost.

4. **f_ext convention.** GATO threads a **single 6-vector wrench per solve**
   (`getOffsetWrench` → `batch + solve_idx*6`, applied in `iiwa14_fext.cuh`). GRiD's f_ext is
   **per-body** `6*NUM_BODIES`, body-local. If GATO moves to GRiD's f_ext path it must broadcast
   its single EE wrench into the per-body buffer (zeros elsewhere). Minor, but a real
   convention adapter.

5. **EE-pose size.** GATO sets `EE_POS_SIZE=6` and tracks only **position rows 0..2** in the cost.
   `grid_plant::ee_pos_cost` also tracks the 3 position axes — matches. (If GATO later wants
   orientation tracking, GRiD has the full 6-pose + `ee_pose_hessian`.)

**Net:** semantics drop in; the work is (a) regen header, (b) wire the wider gradient scratch,
(c) reconcile the integrator enum. No fundamental convention rewrite.

---

## Q3 — Can `grid_plant` support GATO and simplify it? (biggest deletion opportunity)

**Yes — `grid_plant` is essentially a cleaned-up, robot-generic re-implementation of GATO's
hand-rolled plant.** Concrete file/function map of what GATO could delete:

| GATO code (delete / thin) | `grid_plant` replacement | Notes |
|---|---|---|
| `iiwa14_plant.cuh:276-327` `trackingcost` (EE term) | `ee_pos_cost` (value) | GATO fuses EE + qd-reg + 3 barriers in one reduction; `grid_plant` splits them but all exist |
| `iiwa14_plant.cuh:338-424` `trackingCostGradientAndHessian` | `ee_pos_cost_gradient` + `ee_pos_cost_hessian` + `quadratic_state/input_cost_*` | GN Hessian `J_pᵀWJ_p` is **identical math** to `_plant.py:345-376` |
| `iiwa14_plant.cuh:103-155` `jointBarrier{,Gradient,Hessian}` | `joint_position/velocity/torque_barrier[_gradient/_hessian]` | GRiD adds `isfinite` per-side guard + accumulate-offset templating — strictly better |
| `iiwa14_plant.cuh:157-185` `forwardDynamics` | `grid::forward_dynamics_device` (regen) | deletes the `s_XITemp[504]` magic-offset hand-wiring |
| `iiwa14_plant.cuh:187-273` `forwardDynamicsAndGradient` | `grid::forward_dynamics_gradient_device` / `integrator_gradient_device` | deletes the entire hand-coded Minv·dc/du fold |
| `dynamics/integrator.cuh:23-46` `integrator_inner` | `grid::integrator_device` via `plant_step` | per-type math matches (modulo enum, §2.2) |
| `dynamics/integrator.cuh:206-258` `compute_linearized_dynamics` | `plant_step_gradient_and_value` | the `[A|B]` surface is exactly `s_dAB` |
| `dynamics/integrator.cuh:160-184` `sim_step` | `plant_step` | trivial |

**Estimated deletion:** most of `iiwa14_plant.cuh` (≈350 of 457 lines) + most of
`dynamics/integrator.cuh` (≈200 of 259), replaced by thin calls into the regenerated header. The
two `indy7_*` clones get the same treatment for free (today they are a hand-maintained second copy
— that duplication *is* the cost GRiD's codegen removes).

### Where `grid_plant` falls SHORT of GATO (so we know what to extend)

These are real gaps that block a 1:1 cutover and should feed the GRiD backlog (§6):

- **No single fused tracking-cost-with-barriers kernel.** GATO's `trackingcost` computes EE
  tracking + qd-reg + position/vel/torque barriers **in one block reduction** over `threadsNeeded+3`
  terms (`iiwa14_plant.cuh:300-326`). `grid_plant` offers the pieces but the caller must call 4–5
  device fns + manage `ACCUMULATE`. Functionally equivalent, slightly more launches/syncs.
- **No "last-block" terminal-cost variant.** GATO has `trackingCostGradientAndHessian_lastblock`
  (terminal `N_cost` + `Q_kp1`/`q_kp1` for the final knot, `iiwa14_plant.cuh:426-450`). `grid_plant`
  has no terminal/stage distinction — GATO would template that at the call site (it already keys on
  `blockIdx.x==KNOT_POINTS-1`).
- **Per-knot weight scalars vs weight vectors.** GATO passes scalar `q_cost,qd_cost,u_cost,N_cost`;
  `grid_plant` takes diagonal weight **vectors** `s_Q/s_R/s_W`. Adapter is trivial (broadcast), but
  it's a signature mismatch.
- **`plant_step_gradient` exposes spill/scratch internals** the caller must place (§2.3). A
  GATO-friendly "auto-allocating" overload (like the existing `*_device` wrappers) would remove the
  porting friction.
- **Barriers need explicit bound pointers.** GRiD parses only position limits; GATO's vel/torque
  limits are baked as `constexpr` tables (`iiwa14_plant.cuh:48-70`). GATO must pass those as
  `s_lower/s_upper`. Fine, but means the URDF doesn't carry them.

---

## Q4 — Can GLASS improve GATO's linear algebra? (biggest perf opportunity outside plant)

GATO uses **none** of GLASS (grep: zero hits). It re-implements block linear algebra in
`gato/utils/linalg.cuh` (`namespace block`). Concrete swaps:

1. **`block::matMul / matMulSum / matMulTranspose[Sum]` (`linalg.cuh:101-172`) → `glass::gemm`
   family.** GATO's form is one-thread-per-C-element with a serial K-loop and **no tiling, no
   shared-memory staging, no compile-time-dim specialization**. GLASS's `gemm_impl_ct`
   (`src/base/L3/gemm.cuh`) takes M,N,K as template params (cheap magic-number index math instead
   of `MUFU.RCP` div/mod), and `glass.cuh` ships a tiled dispatch with a host smem helper
   (`glass_gemm_dispatch_smem`). The Schur kernels call these 17 times on STATE_SIZE×STATE_SIZE
   blocks (`schur_linsys.cuh:111-123, 244-246`) — the hottest matmuls in the SQP loop.
   *Replace with the compile-time `glass::gemm` (dims are known: STATE_SIZE/CONTROL_SIZE).*

2. **`block::matMul<T,1,STATE_SIZE,STATE_SIZE>` and `<1,N,N>` used as matrix-vector products**
   (`schur_linsys.cuh:351,414`; `computeDz`) → **`glass::gemv`**. Using a GEMM with `m=1` wastes
   the row dimension of parallelism; GLASS `gemv` (`src/base/L2/gemv.cuh`) is the right primitive
   and also has strided/segmented variants for the batched layout.

3. **`block::invertMatrix` Gauss-Jordan (`linalg.cuh:364-519`) → `glass::inv` / Cholesky+TRSM.**
   - The multi-matrix `invertMatrix(DIMA,DIMB,DIMC,…)` fusion in `schur_linsys.cuh:96` (invert
     Q_k, Q_kp1, R_k together) is clever but bespoke. GLASS `invertMatrix` is the maintained,
     block-cooperative equivalent.
   - **`theta_k` and `Q_0` are SPD** (Schur complements / regularized Hessians). Inverting them
     with full Gauss-Jordan (`schur_linsys.cuh:154,191`) is ~2× the flops of
     `glass::cholDecomp_InPlace` + two `glass::trsm` solves. For STATE_SIZE=14 this is a real
     constant-factor win on the per-knot critical path, and Cholesky is more numerically stable
     under the `rho`-regularization GATO already adds.

4. **`block::btdMatrixVectorProduct` (`linalg.cuh:174-273`)** — the PCG block-tridiagonal SpMV — is
   actually a *good*, warp-cooperative kernel (per-block-row warp + shuffle reduction). GLASS
   doesn't have a BTD-specific primitive, so **keep this one**; it's a candidate to *contribute back
   to GLASS* (§6) rather than replace.

5. **`block::dot` (`linalg.cuh:290-327`)** is a solid warp-shuffle reduction — GLASS has `dot` /
   `reduce` equivalents; low priority to swap, but standardizing on GLASS reduces GATO's
   maintenance surface.

**Why this matters:** GATO's matmuls are correct but are the "naive parallel" pattern GRiD's perf
campaigns specifically moved away from. The Schur form/solve is the dominant non-RBD cost in bsqp,
so GLASS GEMM + Cholesky there is the highest-leverage linalg change.

---

## Q5 — Other performance / correctness improvements

1. **Silent launch failures — GRiD's #1 documented pitfall.** GATO's opt-in/heavy kernels (KKT
   setup, merit) launch with large dynamic smem and the merit kernel uses
   `grid::SUGGESTED_THREADS` (`merit.cuh:279`). There is **no `cudaGetLastError()`** after these
   `<<<>>>` launches — only `gpuErrchk` around the *memcpy/malloc* calls and a final
   `cudaDeviceSynchronize`. Per `docs/agent_debugging_guide.md §1c`, a register/smem-cap launch
   failure then yields **zeroed outputs that look like a real (wrong) solve**. *Add
   `gpuErrchk(cudaPeekAtLastError())` immediately after every kernel launch in the bsqp loop, and
   clamp threads to `cudaFuncGetAttributes().maxThreadsPerBlock`.*

2. **`grid::SUGGESTED_THREADS` is being deprecated in GRiD** (memory: "SUGGESTED_THREADS→perf
   cap+autotune"). GATO's merit kernel depends on it (`merit.cuh:279`); the other kernels use fixed
   `*_THREADS` from `settings.h`. When GATO regenerates the header, `SUGGESTED_THREADS` may change
   semantics or vanish — pin merit to an explicit, autotuned thread count instead.

3. **Redundant `direct_minv` recompute.** `forwardDynamicsAndGradient` (`iiwa14_plant.cuh:200`)
   recomputes `direct_minv_inner` and re-runs `inverse_dynamics_inner` even though `s_v` doesn't
   change between the value and gradient passes (the code even has a `// TODO: there is a slightly
   faster way as s_v does not change`). Current GRiD's `forward_dynamics_gradient_device` /
   `integrator_gradient_device` already fuse this — another reason to adopt the generated path.

4. **`block::reduce` serial tail (`linalg.cuh:329-353`)** finishes the reduction on **thread 0
   alone** for the last ≤3 elements and is used in the merit/cost/integrator-error hot paths
   (`compute_integrator_error`, `trackingcost`). GLASS's tree/warp reduce avoids the serial tail;
   the merit kernel also does an `atomicAdd` per (solve,alpha) across knots (`merit.cuh:249`) — fine,
   but the per-block reduce is the serial part to fix.

5. **`addScaledIdentity` only regularizes the top-left half-block** (`linalg.cuh:84-96`:
   `x<dim/2 && y<dim/2`) — i.e. it adds `rho` only to the **position** diagonal of the
   STATE_SIZE×STATE_SIZE Q. That's an intentional choice (velocity states differently
   regularized), but it's subtle and undocumented; worth a comment so a future regen/refactor
   doesn't "fix" it.

6. **Memory layout / fusion:** `setup_kkt` recomputes EE pose **and** EE-pose-gradient inside
   `trackingCostGradientAndHessian` (`iiwa14_plant.cuh:363-364`) while `merit` recomputes EE pose
   yet again (`trackingcost`, `iiwa14_plant.cuh:298`). Across the SQP iteration the EE FK is
   evaluated many times per knot. GRiD's `ee_pose_gradient` already returns pose+Jacobian together;
   a fused cost device fn (cost+grad+hess from one FK sweep) would cut redundant FK. This is also a
   `grid_plant` extension candidate (§6).

7. **`compute_linearized_dynamics` scratch sizing** uses `forwardDynamicsAndGradient_TempMemSize_Shared()
   == grid::FD_DU_MAX_SHARED_MEM_COUNT` (`iiwa14_plant.cuh:270-273`). When GATO regenerates with a
   newer GRiD, **re-derive these `*_SHARED_MEM_COUNT` constants from the header** — they're baked
   per-robot and a stale value silently under/over-allocates (the mimic-overflow / undersized-scratch
   bug class in the debugging guide §1a). Not a mimic robot here, but the same failure mode.

---

## What GRiD should add / extend to better serve GATO (feeds OUR backlog)

1. **A fused, terminal-aware tracking-cost device fn.** `grid_plant` should offer a
   `tracking_cost_value_grad_hess` that does EE-pos tracking + state/input quadratic + the three
   barriers **in one FK sweep + one reduction**, with a `TERMINAL` template (N_cost / no-R / Q_kp1)
   — matching GATO's `trackingcost` + `trackingCostGradientAndHessian_lastblock`. This is the single
   biggest "make the cutover 1:1" item.
2. **Auto-allocating `plant_step_gradient` overload.** Mirror the `*_device` pattern so the caller
   doesn't placement-manage `s_df_du/s_vaf/s_Minv/d_workspace/d_temp_spill`. Removes the main §3
   porting friction.
3. **Confirm/extend `IntegratorType`** to include **semi-implicit Euler** and **trapezoidal** with
   A/B Jacobian blocks byte-matching GATO's `integrator_gradient_inner`. If only Euler+RK are
   emitted today, GATO can't drop its integrator.
4. **A single-wrench-at-frame f_ext adapter** (or doc) bridging GATO's per-solve EE wrench to GRiD's
   per-body `6*NUM_BODIES` convention.
5. **Per-knot diagonal-weight *scalar* fast path** (or accept scalar weights) so callers don't build
   weight vectors for the common uniform-weight case.
6. **Adopt GATO's `btdMatrixVectorProduct` into GLASS** as a block-tridiagonal SpMV primitive —
   it's a genuinely good warp-cooperative kernel GLASS lacks, and it's exactly the structure GRiD's
   own trajectory-optimization users (PDDP/GATO) need.
7. **Ship a maintained Schur/KKT linear-algebra recipe** (Cholesky+TRSM on the regularized
   Hessian/Schur blocks) as a GLASS example, so customers stop hand-rolling Gauss-Jordan inverses.

---

## Open questions

- Is GATO's vendored `iiwa14_grid.cuh` a *frozen* fork (hand-edited beyond codegen, e.g. the
  `iiwa14_fext.cuh` RNEA looks hand-written) or a clean old generation? If hand-edited, a regen
  needs a diff/merge, not a drop-in replace. (The fext file in particular looks bespoke.)
- Does GATO require the **trapezoidal** integrator specifically (default `INTEGRATOR_TYPE=2`), or
  was that just inherited? Determines whether §6.3 is blocking.
- What batch sizes / knot counts does GATO run in production? GLASS's tiled-GEMM vs compile-time-dim
  path choice (and whether Cholesky beats Gauss-Jordan) depends on STATE_SIZE (14 here) and the
  occupancy at that batch size.
- Is exact-Newton (via `fdsva_so`) actually wanted, or is Gauss-Newton sufficient for GATO's
  convergence targets? Determines whether item 4 is worth the research effort.
- Floating-base on the roadmap for GATO? Everything in `integrator.cuh`/`constants.h` assumes
  `NUM_POS==NUM_VEL` (fixed base); GRiD's floating path would need the SE(3) integrator + 7/6 split.
