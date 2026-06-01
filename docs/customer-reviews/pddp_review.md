# GRiD ↔ PDDP consulting review

**Customer:** PDDP (parallel DDP / trajectory-optimization solver), `/home/plancher/Desktop/PDDP`, branch `GBP3` (clean).
**Reviewer scope:** read-only assessment of how PDDP consumes GRiD today, drop-in fit of GRiD's *current* algorithm surface, `grid_plant` simplification potential, GLASS linear-algebra wins, and other perf. No PDDP edits; no builds. GRiD baseline read from `README.md`, `GRiDCodeGenerator/algorithms/_plant.py`, `GLASS/`, `docs/agent_debugging_guide.md`.

PDDP is a chunked/parallel multiple-shooting DDP for the 7-DoF IIWA (`STATE_SIZE=14`, `CONTROL_SIZE=7`, fixed base). It runs one CUDA block per trajectory chunk, does a Riccati backward pass per chunk, an 8-alpha parallel line search, and full second-order DDP (`USE_SECOND_ORDER 1`) — so it consumes GRiD's forward-dynamics gradient **and** the forward-dynamics Hessian.

---

## Executive summary — top 5 ranked recommendations

1. **Adopt GRiD's current transpose-aware GLASS `gemm_ex` and kill the hand-rolled transposes in the backward pass.** PDDP literally has two `// TODO: remove transposes, support transpose of first matrix in gemm` markers (`backwardPassChunk.cuh:126-128` and `:596`) and serially transposes `AB→ABk_T` and `Huxk→HuxkT` in single-thread `for(i)for(j)` loops every knot. GRiD's shipped GLASS `gemm_ex<T, TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_A, ROW_MAJOR_B, ROW_MAJOR_C>` (`GLASS/src/L3/gemm.cuh`) removes both the scratch buffers and the serial transpose loops. **This is the single highest-leverage, lowest-risk change.**

2. **PDDP's GLASS submodule is uninitialized — it is pinned to a stale vendored GLASS.** `PDDP/GLASS/` is an empty directory (submodule not checked out), and PDDP calls `glass::invertMatrixA` + `glass::gemm<T,false>`, neither of which is the current GRiD GLASS API (`invertMatrix`, `gemm<T,TRANSPOSE_B,ROW_MAJOR>`). PDDP is running months-old linalg. Re-pointing PDDP's GLASS submodule at current GRiD GLASS unlocks #1, the batched/strided gemm, `trsm`, and the row/col-major-correct `gemm` — but is an API-break that needs a compat shim or call-site sweep.

3. **PDDP rolls its own second-order term by hand (`compute_fxx_entry` in `integratorGradientKern.cuh`) on top of GRiD's `forwardDynamicsHessian`. This is exactly `fdsva_so`, which GRiD now ships analytically.** PDDP already calls `gato_plant::forwardDynamicsHessian` for the `d²a/dq²,d²a/dqdv,d²a/dv²,d²a/dtdq` blocks, then re-scatters them into the `fxx`/`fux` integrator tensors itself. GRiD's `fdsva_so` (fixed + floating) is the validated, Pinocchio-checked version of this. Migrating removes PDDP's hand-scatter and its index-arithmetic risk.

4. **The `grid_plant` cost/barrier layer can delete most of PDDP's hand-written CUDA cost machinery** (`costGradKern_*`, `costKern_*`, the `gato_plant::COST_*` constants, the EE-position-cost Gauss-Newton assembly in `costGradKern_wEEpose_sequence.cuh`). `grid_plant` emits exactly the quadratic-state/input cost (GN-diag Hessian), the EE-position cost (`J_p^T W J_p` GN Hessian), and joint position/velocity/torque log-barriers PDDP's reference solvers want — with the *same* conventions (½ scaling, `x=[q;qd]`, GN Hessian). See §3.

5. **Replace the catastrophic element-by-element `cudaMemcpy` loops in the host driver (`pddp.cu:221-235`, `:363-368`).** The Vxx/Vx initialization copies *one scalar at a time* across `NUM_TIME_STEPS × STATE_SIZE × STATE_SIZE` (~46·14·14 ≈ 9000) `cudaMemcpyDeviceToDevice` calls per outer iteration. This is pure launch-latency waste; it should be a single strided copy or a tiny kernel. Low risk, real wall-clock win even though it's "only" setup.

---

## Q1 — How PDDP uses GRiD's algorithms today

PDDP wraps GRiD through a `gato_plant::` namespace defined in `include/cuda-include/cg_v4_iiwaplant.cuh`, which `#include`s the **generated** `iiwa14_grid.cuh` (8449 lines — a checked-in GRiD codegen output, the `grid::` namespace). Touchpoints:

**Model handle / init.** `gato_plant::initializeDynamicsConstMem<T>()` → `grid::init_robotModel<T>()`; freed via `grid::free_robotModel`. The opaque `void *d_dynMem_const` carried through `PDDPVariables` (`pddp_variables.cuh:161`) **is** a `grid::robotModel<T>*` (cast at `pddp_utils.cu:244`, `:335`). PDDP also reaches into `robotModel->d_XImats` directly in debug paths (`integratorGradientKern.cuh:153-158`) — a private-layout dependency, fragile across GRiD versions.

**First-order dynamics gradient (the A|B blocks).** `integratorGradientKern` (`include/cuda-include/integratorGradientKern.cuh`) calls `gato_plant::forwardDynamicsGradient<T>(s_dqdd, s_q, s_qd, s_u, d_dynMem_const)` → GRiD's fixed-base `fd_du`, then `sgrutils::_integratorGradient` folds `dqdd/dq, dqdd/dqd, Minv` into the discretized `[A|B]` (2n×3n, column-major, Euler `× dt`). This is what GRiD's `grid_plant::plant_step_gradient` now does as a thin wrapper.

**Second-order (full DDP).** Under `#if USE_SECOND_ORDER` (`integratorGradientKern.cuh:163-198`): `gato_plant::forwardDynamicsHessian<T>(s_df2, s_dqdd, ...)` produces the four `NUM_POS³` blocks; PDDP's own `compute_fxx_entry` (same file, lines 27-81) and an inline `fux` scatter assemble the `fxx` (2n×2n×2n) and `fux` (2n×nu×2n) integrator-Hessian tensors. Those feed the backward pass's `Hxx += Vx·fxx`, `Hux += Vx·fux` contraction (`backwardPassChunk.cuh:245-285`).

**End-effector pose + Jacobian.** `costKern_wEEpose_sequence` / `costGradKern_wEEpose_sequence.cuh` call `grid::end_effector_positions_device` and `grid::end_effector_positions_gradient_device`, then hand-assemble an EE-position tracking cost gradient `J_p^T W (p−p_des)` (`costGradKern_wEEpose_sequence.cuh:91-118`) with `atomicAdd` into `gk`.

**Cost values.** `costKern_sequence` / `costKern_wEEpose_sequence` + `gato_plant::costGrad_regularization` + the `gato_plant::COST_Q1/Q2/QF1/R/Qee...` constant family (`cg_v4_iiwaplant.cuh:50-110`).

**Host driver.** `cuda/pddp.cu :: parallel_ddp` orchestrates init → (backward → forward/line-search → next-iter-setup) loop. `nextIterationSetup_sequence` / `integratorGradientKern` recompute `AB/H/g` each iteration.

**`reference/` Python.** A clean, modular CPU reference: `TrajoptPlant.py` (Pendulum / DoubleIntegrator / URDF/IIWA via Pinocchio), `TrajoptCost.py`, `TrajoptConstraint.py` (ACTIVE_SET / QP / QUADRATIC_PENALTY / AUGMENTED_LAGRANGIAN / **LOG_BARRIER**), and `reference/solvers/{unconstrained, interior_point, log_barrier, augmented_langrangian, quadratic_program}`. This is the design intent the CUDA side implements: an unconstrained DDP core plus an interior-point/log-barrier outer layer.

**GBP3 = "scaling variables".** `pddp_variables_old_08212025.cuh` vs `pddp_variables.cuh`: the new struct adds DARE `d_P`, `d_Qdg`, regularization-index bookkeeping, and the chunk-offset machinery. The matching scaling design is `reference/scaling.py` + `reference/docs/scaling.md` (multiplicative state/control/cost-matrix/Jacobian scaling). So GBP3 is mid-migration: the Python reference has the scaler; the CUDA path carries the new variable struct but the scaler is **not yet wired into the device gradients** (the A|B from `integratorGradientKern` are physical-unit, unscaled). **Open question for the customer:** is on-device scaling intended to fold into the plant-step gradient, or stay a host-side pre/post transform?

---

## Q2 — Will GRiD's current algorithms drop in well?

**Conventions line up.** PDDP's state is `x=[q;qd]` (`config.cuh:16-22`, NUM_POS=7), control `u=torque(7)`, gradient block order `[A|B]=[dx'/dx | dx'/du]` 2n×3n column-major — **identical** to `grid_plant`'s documented convention (`_plant.py` header: "x=[q;qd]; [A|B] = s_dAB surface, 2n×3n, column-major"). Fixed-base IIWA ⇒ NUM_POS==NUM_VEL, no mimic, no floating-base quaternion tangent-space mismatch. The integrator is explicit Euler `× dt`, which is `grid::IntegratorType::EULER` (the plant-step default). So the gridData/robotModel handle, dims, and tangent convention are a clean match.

**First-order: drop-in.** `grid_plant::plant_step_gradient[_and_value]` is a pass-through wrapper over the exact `grid::integrator_gradient_device` surface PDDP already folds by hand. PDDP could swap `integratorGradientKern`'s body for a `grid_plant::plant_step_gradient` call and delete `sgrutils::_integratorGradient` — same A|B bytes.

**Second-order: GRiD now offers what PDDP hand-rolls, with one caveat.** PDDP builds the *integrator* Hessian (`fxx`,`fux`) from `forwardDynamicsHessian`. GRiD's `fdsva_so` gives the analytic forward-dynamics second-order directly. **But note `_plant.py` is explicit that GRiD does NOT yet emit a `grid_plant` analytic 2nd-order *integrator* Hessian** — it ships only the Gauss-Newton cost Hessian and leaves `// TODO(plant-2nd-order): grid has no analytic 2nd-order integrator`. So today PDDP can adopt `fdsva_so` for the RBD Hessian blocks, but the `dt`-scaling + state-lift scatter into the 2n×2n×2n `fxx` tensor (PDDP's `compute_fxx_entry`) still has to live somewhere. **This is the cleanest single backlog item for GRiD** (see §"what GRiD should add"): emit a `grid_plant::plant_step_hessian` that does exactly PDDP's `compute_fxx_entry`/`fux` scatter from `fdsva_so`, validated.

**Friction / gaps.**
- **Private `robotModel` field access** (`d_XImats`) in PDDP debug code couples to GRiD internals; should go through public device fns.
- **`gato_plant` is a fork, not a thin shim.** PDDP's `gato_plant::` predates GRiD's `grid_plant::` namespace and duplicates its intent (cost constants, EE cost, init). Re-basing `gato_plant` onto `grid_plant` is desirable but is a real refactor, not a recompile.
- **Checked-in generated header.** `iiwa14_grid.cuh` is a vendored codegen snapshot; adopting current GRiD algorithms means regenerating it (and re-pinning GLASS). Worth a one-time `grid-generate iiwa.urdf -t <ee_joint>` + diff.
- **No launch-error checking around the GRiD-derived kernels** (see §5) — exactly the silent-zero-launch-failure class GRiD's own `agent_debugging_guide.md §1c` calls out.

---

## Q3 — Can `grid_plant` support PDDP's needs and simplify its code?

**Yes, substantially — for the cost/barrier surface.** Mapping PDDP's hand-rolled CUDA to `grid_plant` (`GRiDCodeGenerator/algorithms/_plant.py`):

| PDDP today | `grid_plant` replacement | Notes |
|---|---|---|
| `gato_plant::COST_Q1/Q2/QF1/QF2/R/...` constants + `costKern_sequence` quadratic state/control cost (`cg_v4_iiwaplant.cuh:50-101`) | `quadratic_state_cost` / `quadratic_input_cost` `_value_grad_hess` | Same ½ scaling, GN-diag Hessian = diag(W). PDDP passes weights as a vector instead of compiled-in constants — a usability *gain*. |
| `costGradKern_wEEpose_sequence.cuh:91-118` hand EE cost `J_p^T W r` + `atomicAdd` | `ee_pos_cost` / `ee_pos_cost_gradient` / `ee_pos_cost_hessian` (GN `J_p^T W J_p`) | `grid_plant` writes the q-block of the NX×NX Hessian directly; PDDP currently only assembles the gradient and leans on GN elsewhere. |
| `reference/solvers/log_barrier` + `interior_point` (Python only; no CUDA barrier kernels found) | `joint_position_barrier` / `joint_velocity_barrier` / `joint_torque_barrier` (value/grad/hess-diag, isfinite-guarded) | **This is the biggest *new* capability.** PDDP's constrained solvers exist only in the Python reference; `grid_plant` gives PDDP ready, on-device log-barriers in the right `[q;qd]`/`u` slices to bring interior-point to the GPU. |
| `plant_step` (forward sim) in `computeNextState_FP.cuh` / forward pass | `grid_plant::plant_step` | thin integrator wrapper. |

**Biggest single simplification:** deleting the bespoke EE-cost gradient assembly in `costGradKern_wEEpose_sequence.cuh` and the `gato_plant::COST_*` constant family in favor of `grid_plant::{ee_pos_cost*, quadratic_*_cost*}` — that's a few hundred lines of error-prone, atomically-accumulated, debug-print-laden CUDA replaced by validated device fns with the same math.

**Where `grid_plant` falls short of PDDP (→ GRiD backlog):**
- **No analytic 2nd-order plant (integrator) Hessian** — the one thing PDDP's *full DDP* most needs on the dynamics side (it builds `fxx`/`fux` by hand). `_plant.py` documents this as a deliberate omission. PDDP is precisely the customer that justifies adding it.
- **No augmented-Lagrangian / quadratic-penalty constraint primitive** — `grid_plant` ships only log-barriers; PDDP's `TrajoptConstraint.py` supports QUADRATIC_PENALTY and AUGMENTED_LAGRANGIAN modes. A `grid_plant` AL/penalty term (value/grad/GN-Hess of `‖c(x)‖²` with a multiplier/`μ`) would let PDDP's AL solver go on-device too.
- **No EE-*orientation* / 6-DoF pose cost** — PDDP currently zeros out the orientation error (`costGradKern_wEEpose_sequence.cuh:109-111` commented out), but the variable carries `EE_POSE_SIZE=6`. `grid_plant::ee_pos_cost` is position-only (3). A 6-DoF pose-error cost (with the known d2ee orientation-hessian caveat from GRiD's own backlog) is a future ask.
- **No scaling hook** — GBP3's multiplicative scaler has no `grid_plant` counterpart; the plant gradient is physical-unit only.

---

## Q4 — Can GLASS learnings improve PDDP's linear algebra?

The backward pass (`backwardPassChunk.cuh`) is the linalg-heavy core and the clearest win. It is *already* GLASS-based (`glass::gemm`, `glass::gemv`, `glass::axpy`, `glass::invertMatrixA`, `glass::cholDecomp_check_multiThread`), but against a **stale, uninitialized GLASS submodule** (`PDDP/GLASS/` is empty; API names `invertMatrixA` / `gemm<T,false>` don't exist in current GRiD GLASS). Concrete improvements with current GRiD GLASS:

1. **Transpose-aware gemm eliminates two serial transpose loops per knot.**
   - `backwardPassChunk.cuh:126-133` transposes `AB→ABk_T` (single-thread double loop) to form `AB^T·Vxx·AB`. PDDP's own comment: *"TODO: remove this extra computation, improve matrix multiplication function to be able to handle transposed matrices via smarter indexing."*
   - `:596-604` transposes `Huxk→HuxkT` (single-thread double loop) for `Hux^T·K`. Comment: *"TODO: remove transposes, support transpose of first matrix in gemm."*
   - **GRiD GLASS already has this:** `gemm_ex<T, TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_A, ROW_MAJOR_B, ROW_MAJOR_C>` (`GLASS/src/L3/gemm.cuh:135-163`, plus the inner `gemm_impl` with both `TRANSPOSE_A`/`TRANSPOSE_B`, lines 209+). Replacing the transpose-then-gemm with a single `gemm_ex<T,true,false,...>` removes the `ABk_T`/`HuxkT` scratch buffers (per-knot global memory) **and** the serial loops (currently O(2n·3n) and O(n·nu) single-thread work, fully unparallelized) on every knot of every chunk.

2. **`Huu` inverse via explicit Gauss-Jordan is the wrong primitive for an SPD matrix.** `invHuu` (`backwardPassChunk.cuh:1-36`) builds `[Huu | I]` and runs `glass::invertMatrixA` (Gauss-Jordan), then PDDP *also* runs `glass::cholDecomp_check_multiThread` separately just to test PD-ness. For an SPD `Huu` the textbook path is **one** Cholesky (`glass::cholDecomp_InPlace`, `GLASS/src/L3/chol_InPlace.cuh`) which (a) is the PD test for free — failure ⇔ non-PD — and (b) feeds `glass::trsm` (`GLASS/src/L3/trsm.cuh`, present in current GLASS, absent in PDDP's) to solve `K = −Huu⁻¹Hux` and `κ = −Huu⁻¹gu` by triangular solves instead of forming the explicit inverse and a follow-up gemm. That replaces *inverse + 2 gemms + a separate chol-check* with *chol + 2 trsm*, which is both fewer flops and better-conditioned.

3. **Strided/batched gemm for the per-knot sweep.** Current GLASS adds `gemm_strided` and `gemm_batched_indexed` (`glass.cuh:36-37`). PDDP's backward pass loops knots serially within a chunk calling scalar gemms; the `VxxAB`, `AB^T·VxxAB`, `AB^T·Vxx` products at independent knots inside the line-search/forward stages (and the per-alpha `_lro` buffers, `pddp.cu:288-297`) are batched-gemm-shaped. The 8-alpha line search in particular (`NUM_ALPHA=8` independent trajectories) is the canonical batched workload.

4. **Block-cooperative reductions over hand loops.** The expected-cost-reduction and norm computations and the `compute_tracking_error` host loop are fine on host, but the device-side `Hxx += Vx·fxx` contraction (`backwardPassChunk.cuh:252-282`) is a hand grid-stride double loop; it's correct but is exactly the shape `glass::gemv`/a small batched contraction handles, and it currently strides on `blockDim.x*gridDim.x` inside a single-block-per-chunk kernel (the `gridDim.x` stride is a latent bug-smell — see §5).

---

## Q5 — Other performance improvements

- **Element-wise `cudaMemcpy` storms (highest wall-clock waste).** `pddp.cu:221-235` copies `d_H`→`d_Vxx` and `d_g`→`d_Vx` **one scalar per `cudaMemcpy`** across all knots — thousands of API calls per iteration. `:363-368` similarly. Replace with one strided `cudaMemcpy2D`/`cudaMemcpyAsync` or a 1-line copy kernel. Pure latency elimination.

- **Silent launch-failure / no error checks around GRiD-derived kernels.** `initialization_sequence` (`pddp_utils.cu:254-260`) and `integratorGradientKern` launches `cudaDeviceSynchronize()` but never `cudaGetLastError()`. Under `USE_SECOND_ORDER`, `integratorGradientKern` declares large `__shared__` arrays — `s_df2 = 4·NUM_POS³` (=1372 T) + `s_dqdd = 3·NUM_POS²` + the FD-hessian temp — and is launched with `NEXT_ITER_SETUP_THREADS = 17` threads and **no `__launch_bounds__`**. This is the exact "zeros masquerading as results / register-cap launch failure" class in GRiD's `agent_debugging_guide.md §1c`. **Add `cudaGetLastError()` + fail-loud after every launch**, and clamp threads to `cudaFuncGetAttributes().maxThreadsPerBlock`.

- **`gridDim.x` in a block-per-chunk grid-stride.** `backwardPassChunk.cuh:252` / `:268` stride `idx += blockDim.x * gridDim.x` and offset `+ blockIdx.x * blockDim.x` for the `fxx`/`fux` contraction — but this device fn runs per-chunk-block writing a *chunk-local* tensor. Mixing `blockIdx.x` into an index that's meant to be block-local looks wrong for `BACKPASS_BLOCKS=NUM_CHUNKS>1`; worth verifying it isn't reading/writing across chunk boundaries.

- **Occupancy / thread counts.** `NEXT_ITER_SETUP_THREADS=17`, `BACKPASS_THREADS=32`, `FORWARDPASS_THREADS=128` are hand-tuned magic numbers. 17 threads on a kernel doing `NUM_POS³`-sized shared work massively under-occupies; the GLASS block-cooperative primitives want a full warp-multiple. GRiD's autotune/`SUGGESTED_THREADS` (already surfaced as `grid::SUGGESTED_THREADS` and aliased in `cg_v4_iiwaplant.cuh:37`) should drive these.

- **Redundant recompute.** `nextIterationSetup_sequence` recomputes `AB/H/g` from scratch each iteration even when the line search accepted a small step; the `with_value` fused integrator-gradient (`grid_plant::plant_step_gradient_and_value`) returns `x_{k+1}` and `[A|B]` in one RBD pass, saving the separate forward-sim cost evaluation.

- **`std::default_random_engine randEng(time(0))` at namespace scope** in a header (`cg_v4_iiwaplant.cuh:15`) — a global with a static initializer pulled into every TU that includes the plant. Harmless functionally, but it's a header-hygiene smell and a non-deterministic seed in a solver you may want reproducible.

---

## What GRiD should add / extend to better serve PDDP (our backlog)

1. **`grid_plant::plant_step_hessian` (analytic 2nd-order integrator Hessian).** Emit the `dt`-scaled lift of `fdsva_so` into the `fxx` (2n×2n×2n) and `fux` (2n×nu×2n) tensors — i.e. productize PDDP's hand-written `compute_fxx_entry` + `fux` scatter (`integratorGradientKern.cuh:27-198`), validated against the numpy reference. This is the #1 ask from a *full-DDP* customer and the documented gap in `_plant.py`.
2. **Constraint primitives beyond log-barrier:** a quadratic-penalty / augmented-Lagrangian `grid_plant` term (value/grad/GN-Hess of `‖c(x)‖²` with `μ`/multiplier inputs) to match `TrajoptConstraint.py`'s AL/penalty modes and let PDDP's AL+IP solvers run on-device.
3. **6-DoF EE-pose cost** (position + orientation error) layered on `ee_pose` + the d2ee path, with the orientation-hessian caveat flagged.
4. **A scaling hook** in the plant gradient (or documented host-side recipe) so GBP3's multiplicative scaler composes with `plant_step_gradient` without a separate unscale pass.
5. **A blessed `riccati`/backward-pass GLASS recipe** (chol + trsm + transpose-aware gemm) as a documented example — PDDP is reimplementing the standard DDP backward pass that every GRiD trajopt customer will need; a reference kernel would prevent the stale-GLASS + hand-transpose pattern recurring.
6. **Public accessor for `robotModel` internals** (or doc that `d_XImats` is private) so customers stop reaching into the struct.

---

## Open questions for the customer

- Is GBP3's scaler meant to fold into on-device gradients, or remain a host pre/post transform? (Drives backlog item 4.)
- Is the `gato_plant::` namespace intended to be re-based onto `grid_plant::`, or kept as a PDDP-owned fork? (Drives how much of §3 is "delete" vs "wrap".)
- Is the checked-in `iiwa14_grid.cuh` a frozen snapshot, or do you want to track GRiD `main` (and thus re-pin GLASS)? (Gates every "adopt current GRiD" recommendation.)
- The EE cost currently zeros orientation error — is 6-DoF pose tracking on the roadmap (backlog 3)?
- `backwardPassChunk.cuh:252/:268` `blockIdx.x` in the `fxx` index with `NUM_CHUNKS>1`: intended, or a latent cross-chunk read? (Correctness, not perf.)
