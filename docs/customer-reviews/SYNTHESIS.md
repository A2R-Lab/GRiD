# Customer code-review synthesis — GATO / PDDP / HJCD-IK (2026-06-01)

Three deep read-only reviews of GRiD's core customers (during the C.7 sweep window). Per-customer
detail: [`gato_review.md`](gato_review.md), [`pddp_review.md`](pddp_review.md),
[`hjcdik_review.md`](hjcdik_review.md). This is the consolidated signal + the GRiD roadmap it implies.

## The headline: all three are on STALE GRiD and underusing what we already ship
Every customer is pinned to an old GRiD/GLASS and hand-rolls capabilities GRiD now provides. The
single highest-leverage action for each is **regenerate the header / re-pin GLASS** — it unlocks
every other win below with near-zero integration risk.

| Customer | How it uses GRiD | Stale-pin symptom |
|---|---|---|
| **GATO** | vendors a 9123-line `grid.cuh`; hand-writes its whole plant against low-level `grid::*_inner` with magic offsets (`s_XITemp[504]`…) | header predates `grid_plant`, `IntegratorType`, `integrator_gradient`, `frame_jacobian`, `osc_inertia`; uses **none of GLASS** |
| **PDDP** | `pddp.cuh` consumes dynamics + hand-rolls the 2nd-order term; Riccati backward pass | **`PDDP/GLASS/` submodule is empty**; calls GLASS APIs that no longer exist (`invertMatrixA`, `gemm<T,false>`) |
| **HJCD-IK** | uses GRiD as a URDF→device-model compiler only; **hand-patches panda FK + joint limits INTO the generated header** | frozen on pin `0a6c18e` (vs HEAD `501adee`) precisely to preserve those hand-patches |

## Four findings that appeared in ALL THREE (strongest signal)
1. **They hand-roll what `grid_plant` / GRiD algorithms now provide.** `grid_plant` cost/barrier/step
   primitives match their conventions by design (`x=[q;qd]`, ½-quadratic, GN-Hessian, log-barrier).
   Concrete deletion: GATO ≈**350/457 lines** of `iiwa14_plant.cuh` + ≈200/259 of `integrator.cuh`
   + the duplicated `indy7` copy; PDDP its bespoke `costKern_*`/EE-cost-gradient assembly; HJCD its
   inline geometric Jacobian + hand-FK.
2. **They underuse GRiD's analytic 2nd-order + new kinematics — our differentiators.** PDDP hand-rolls
   `forwardDynamicsHessian` (= our `fdsva_so`); GATO leaves `idsva_so`/`fdsva_so` (exact Newton) +
   OSC + centroidal unused *though they're already in its header*; HJCD does GN/dogleg where
   `ee_pose_hessian` gives a true Newton step + `frame_jacobian` replaces its inline J.
3. **They underuse GLASS for their heavy linalg.** GATO: 17 naive `matMul`/Gauss-Jordan-invert in
   `schur_linsys.cuh`. PDDP: Gauss-Jordan `invHuu` + separate Cholesky in the Riccati pass. HJCD: LM
   Cholesky serialized to `lane==0`. **All three want the same swap:** `glass::cholDecomp_InPlace`
   (PD test free) + `trsm` solves instead of explicit-inverse-then-gemm for their SPD systems, plus
   `glass::gemm`/`gemv` for the matmuls. ~2× fewer flops + more stable under regularization.
4. **All three have the silent-launch-failure pitfall in the wild** — they launch large-smem,
   no-`__launch_bounds__` kernels and discard/skip `cudaGetLastError()` (GATO merit; PDDP backward;
   HJCD `lm_tuner`). This is EXACTLY `agent_debugging_guide.md §1c` — a zeroed output masquerading as
   a result. Validates that the guard we added to our own timing harness is a real-world trap, and is
   worth a one-pager "integrating GRiD kernels safely" for customers.

Also seen twice: the soon-deprecated `SUGGESTED_THREADS` symbol leaking into customer headers (GATO
merit kernel) — our F2 rename has external blast radius, do it carefully.

## The GRiD roadmap this implies (customer-driven "extend GRiD" asks, ranked by demand)
1. **Plant ergonomics + 2nd-order (PDDP + GATO — the control customers):**
   - `plant_step_hessian` — analytic 2nd-order plant/integrator Hessian. `_plant.py` explicitly omits
     it; **PDDP is the customer that justifies it** (its `compute_fxx`), and it makes GATO's exact-Newton
     path 1:1. **Highest-value single addition.**
   - **Auto-allocating `plant_step_gradient`** overload (GATO) — current one forces callers to
     placement-manage `s_df_du/s_vaf/s_Minv/d_workspace/spill`. An ergonomic wrapper removes the
     biggest cutover friction.
   - **Fused terminal-aware `cost_value_grad_hess`** (one FK sweep: EE + quadratic + barriers, with a
     TERMINAL/N_cost variant) — GATO needs this for a 1:1 plant cutover.
   - **Constraint primitives beyond log-barriers** — augmented-Lagrangian / quadratic-penalty (PDDP's
     interior-point + GBP3 need more than the log-barriers we ship).
2. **Parallelize `frame_jacobian_inner` (HJCD — near-term perf).** We shipped frame_jacobian J as
   "correctness-first serial," and HJCD reports it is **slower than their per-joint-parallel hand
   version**. Our newest kinematics needs the column-parallelism treatment (the d2ee/id_du pattern)
   before customers will adopt it for perf. Quick + high-credibility.
3. **frame_jacobian extras (HJCD):** a fixed-tool/`X_fixed`-frame target; **mimic CUDA frame-Jacobian
   (grippers) — DONE this session (mimic-frame-jac landed)**; joint-limits as a first-class emitted
   helper (HJCD had to patch `init_joint_limits` into `grid.cuh` — a recurring "patch the header"
   smell that also freezes their pin).
4. **Integrator coverage (GATO):** confirm/ship `IntegratorType` semi-implicit-Euler + **trapezoidal**
   with byte-matching A/B (GATO defaults to trapezoidal).
5. **Lean codegen profiles (HJCD):** a kinematics-only profile to shrink the ~10k-line header
   (customers vendoring the whole header is part of why they freeze pins).
6. **Upstream GATO's `btdMatrixVectorProduct`** (block-tridiagonal SpMV) into GLASS — GLASS lacks it,
   GATO has a good one; mutual win.
7. **Small adapters (GATO):** single-wrench→per-body f_ext adapter; scalar-weight cost fast path.

## Cross-cutting recommendation
The recurring "patch the generated header" anti-pattern (HJCD FK + joint limits, GATO magic offsets)
is what freezes customers on stale pins and blocks them from every improvement. Two structural fixes
would compound: (a) the lean/extensible codegen profiles + first-class helpers above so customers
never need to hand-edit the header; (b) a short **"integrating + regenerating GRiD safely"** guide
(launch-error checks, re-pin GLASS, regen cadence) — the customer-facing companion to our internal
`agent_debugging_guide.md`. Lead every customer engagement with **"regenerate against current GRiD"**;
it's the unlock.
