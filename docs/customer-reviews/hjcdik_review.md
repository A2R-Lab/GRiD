# GRiD Consulting Review — HJCD-IK

**Customer:** HJCD-IK (`/home/plancher/Desktop/HJCD-IK`, branch `main`, commit `5817ec9`)
**Reviewer:** GRiD perf/architecture consulting
**Date:** 2026-06-01
**Scope:** Read-only review. No edits, builds, or GPU runs. Recommendations only.

HJCD-IK is a batched GPU inverse-kinematics solver for the Franka Panda (7-DOF
fixed base). It runs a two-phase pipeline: a **coarse** greedy cyclic-coordinate
search (`coarse_search`, float) followed by a **refine** damped Gauss-Newton /
Levenberg-Marquardt + dogleg tuner (`lm_tuner` → `solve_lm_batched`, double),
with an optional pRRTC collision-cost layer. It already vendors GRiD as a
submodule (`external/GRiD`) and generates `grid.cuh` via
`scripts/generate_grid.py`.

---

## Executive summary — top recommendations (ranked)

1. **Stop hand-patching forward kinematics into the generated header.** The two
   hottest device routines, `grid::X_single_thread` and `grid::X_warp`
   (`include/test_cuh/grid.cuh:4126` and `:4288`), are **hand-written,
   panda-specific FK** (joint sin/cos patterns hardcoded for 7 joints, a baked
   `FLANGE_IDX=8`/`EE_IDX=7` tool offset) injected *into* the GRiD-generated
   file. They are not GRiD codegen output, so **regenerating `grid.cuh` deletes
   them** — the repo is pinned to a stale GRiD precisely because regeneration is
   unsafe. This is the central architectural debt and it blocks every other
   improvement below. GRiD's stock `end_effector_pose` family + the new
   `frame_jacobian` family cover this functionality from codegen.

2. **Adopt GRiD's new `frame_jacobian` kernel and delete the hand-rolled
   geometric Jacobian.** `solve_lm_batched` hand-assembles the 6×N geometric
   Jacobian inline (`src/hjcd_kernel.cu:729-750`) from joint z-axes and EE
   offsets. GRiD now emits exactly this (`frame_jacobian`, LOCAL_WORLD_ALIGNED
   matches HJCD's world-frame convention) — validated against Pinocchio. This is
   a near-drop-in replacement that removes hand-derived Jacobian math.

3. **Replace the per-iteration FK-difference cost probes with an analytic
   Jacobian/Hessian path.** The refine loop recomputes full FK (`X_warp` /
   `X_single_thread`) **many times per LM iteration** — once per residual, once
   per backtracking try (up to 4×, `:861`), and again inside `try_dogleg_step`
   and `try_coord_linesearch` (each calls `recompute_cost_scaled` →
   `X_single_thread`, and `try_coord_linesearch` does it `2` times,
   `src/hjcd_kernel.cu:492-504`). For a Newton/2nd-order step GRiD's
   `ee_pose_hessian` gives the analytic pose Hessian directly, removing the
   finite-difference-style line-search FK storm.

4. **Move the normal-equations solve onto GLASS block-cooperative primitives.**
   The LM solve is single-warp Cholesky on one lane (`chol_solve`,
   `warp_cholesky_solve_inplace` do forward/back-substitution on `lane==0` only,
   `src/hjcd_kernel.cu:564-581`). GLASS `invertMatrix` / `chol_InPlace` are
   block-cooperative and would parallelize the N×N factor/solve across the block.

5. **Regenerate against current GRiD once FK is de-patched.** The submodule is
   pinned at `0a6c18e`; current GRiD HEAD is `501adee` and includes the entire
   new kinematics surface (`frame_jacobian`, `frame_jacobian_dot`,
   `osc_inertia`, `ee_pose_hessian`) plus parallelized gradients and the
   silent-launch-failure hardening. HJCD gets these "for free" on regeneration —
   *if* the FK patch is first lifted out of the generated file (rec. 1).

---

## Q1. How HJCD-IK uses GRiD today

### Touchpoints

| File | GRiD usage |
|------|-----------|
| `scripts/generate_grid.py` | Thin wrapper that shells out to `external/GRiD/generateGRiD.py <urdf> [-t target] [-n ns] [-d] [-f]`. Generates `grid.cuh`. |
| `include/test_cuh/grid.cuh` | The generated header (10066 lines, "optimized for the urdf: panda"). **Hand-patched** with `X_single_thread`, `X_warp`, `init_joint_limits` (see Q1 findings). |
| `include/hjcd_kernel.h` | Forward-declares `grid::robotModel<T>`, `grid::init_robotModel<T>()`. Public `Result<T>` / `generate_ik_solutions` API. |
| `src/hjcd_kernel.cu` | The solver. Consumes `grid::NUM_JOINTS`, `grid::init_robotModel<T>`, `grid::init_joint_limits<double>`, `grid::load_update_XmatsHom_helpers`, `grid::X_warp`, `grid::X_single_thread`. |
| `src/main.cpp` / `src/pybind_module.cpp` | Host entry points; only touch `init_robotModel`, `grid_num_joints`, `init_joint_limits_from_grid`. |
| `include/math.cuh` / `include/device_utils.cuh` | Hand-rolled quaternion / vec / RNG helpers. No GRiD dependency. |
| `benchmark/ik_benchmark.py` | Python-level timing harness; calls the pybind module, not GRiD directly. |

### What GRiD code HJCD actually consumes

- **`grid::NUM_JOINTS`** (=7) — used as `hjcd::N` / `N` everywhere.
- **`grid::robotModel<T>` + `init_robotModel<T>()`** — the device model struct
  carrying the per-joint homogeneous transform constants. This is the real
  GRiD value-add HJCD relies on.
- **`grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, s_q, d_robotModel, s_tmp)`**
  — stock GRiD (confirmed: it's emitted by `_topology_helpers.py` /
  `GRiDCodeGenerator.py`). Loads the per-joint **local** homogeneous transforms
  into shared memory. HJCD calls this, then runs its **own** chain-up.
- **`grid::init_joint_limits<double>()`** — a custom helper added to the
  generated header (`grid.cuh:2321`), copied to a `__constant__ double2
  c_joint_limits[N]` by `init_joint_limits_from_grid()` (`hjcd_kernel.cu:121`).

### What HJCD hand-rolls (and shouldn't have to)

- **Forward kinematics chain-up + EE/flange transforms** — `X_single_thread`
  (`grid.cuh:4126`) and `X_warp` (`grid.cuh:4288`). These take GRiD's local
  `s_XmatsHom`, write the q-dependent rotation entries themselves (a hardcoded
  `cos/sin` block per joint 0..6), accumulate `T_j = T_{j-1}·X_j`, then build
  flange (slot 8) and EE (slot 7 = flange·X_fixed) transforms. **Entirely
  hand-derived and panda-specific.** GRiD's `end_effector_pose_inner`
  (`grid.cuh:2729`) already does the chain-up; the new `frame_jacobian_inner`
  does the same world-transform BFS chain-up generically.
- **Geometric Jacobian** — assembled inline in `solve_lm_batched`
  (`hjcd_kernel.cu:729-750`): `J_lin = z_i × (p_ee − p_i)`, `J_ang = z_i`. This
  is the textbook revolute-joint geometric Jacobian in the world frame — exactly
  GRiD's `frame_jacobian` in `LOCAL_WORLD_ALIGNED`.
- **Quaternion error / rotvec / `mat_to_quat`** (`math.cuh`) — orientation
  residual machinery; reasonable to keep local (GRiD's EE pose uses a different
  orientation parameterization), but see Q2.
- **Normal-equations build + Cholesky solve** — `build_ne_and_solve_warp`,
  `warp_cholesky_solve_inplace`, `chol_solve` (`hjcd_kernel.cu:292-635`).
- **Coarse cyclic-coordinate search** — `coarse_search` and `solve_pos` /
  `solve_ori` (`hjcd_kernel.cu:188-288, 1020`). This is HJCD's own algorithm and
  is appropriately bespoke.

**Net:** HJCD uses GRiD as (a) a URDF→device-model compiler and (b) a loader for
local joint transforms. Everything downstream of `load_update_XmatsHom_helpers`
— the FK chain-up, the Jacobian, the linear algebra — is hand-rolled, including
two routines surgically inserted into the generated header.

---

## Q2. Will GRiD's current algorithms (esp. the new kinematics) drop in?

**Yes for the geometric Jacobian and FK; partially for the rest.** GRiD's new
kinematics surface maps onto HJCD's needs closely.

### 2a. Geometric Jacobian `J` — strong drop-in

HJCD's inline Jacobian (`hjcd_kernel.cu:729-750`) is the 6×N revolute geometric
Jacobian with rows ordered `[linear(3); angular(3)]` evaluated at the EE origin
in world axes. GRiD's `frame_jacobian` (`_frame_jacobian.py:53`) emits **exactly
this layout** (6×NV column-major, `[linear; angular]`) and the
`LOCAL_WORLD_ALIGNED` (reference_frame=2) convention is "at the frame origin,
world-aligned axes" — the same convention HJCD hand-derives. Validated against
Pinocchio's `getFrameJacobian`.

- **Convention fit:** ✅ row order, ✅ frame (LWA = HJCD's world-at-EE).
- **`target_jid`:** HJCD targets a fixed tool frame (`FLANGE_IDX`/EE = flange ·
  X_fixed). GRiD's `frame_jacobian` targets a *joint* frame; the fixed
  tool-offset rotation X_fixed would need to be folded in. Since the X_fixed
  offset is a pure constant transform, the Jacobian at the tool frame equals the
  flange Jacobian with the linear rows shifted by the (rotated) offset lever arm
  — straightforward, but **a real convention gap**: GRiD's `frame_jacobian`
  currently keys on a joint id selected at codegen via `-t`, matching how HJCD
  already passes `-t panda_grasptarget_hand`. If the `-t` target lands on the
  tool frame, the drop-in is exact. **Confirm the `-t` target frame matches
  HJCD's `X_fixed` slot.**
- **Before:** ~22 lines of hand-derived cross products + the entire bespoke
  `X_warp` FK feeding them.
  **After:** `grid::frame_jacobian_device<T>(s_J, target_jid, 2 /*LWA*/, s_q,
  d_robotModel)` — one call, codegen-maintained, Pinocchio-validated.

**Caveat (perf):** GRiD's `frame_jacobian_inner` is explicitly "correctness-first
single-block, serial inner assembly" (Step 3 accumulates columns in a
`gen_add_serial_ops()` block, `_frame_jacobian.py:160-196`). HJCD's inline
version is **fully parallel across N joints** (one thread per joint,
`hjcd_kernel.cu:729`). So a naive swap could be *slower* per call. The right move
is: adopt GRiD's J for correctness/maintainability **and** push GRiD to
parallelize `frame_jacobian_inner`'s column assembly (see backlog). This is also
flagged in GRiD's own codegen-parallelism audit memo.

### 2b. Pose Hessian `H` — enables a real Newton IK (not yet used)

HJCD is Gauss-Newton (it forms `JᵀJ`, never the true Hessian). GRiD now emits
`ee_pose_hessian` (the 2nd-order EE Jacobian, `grid.cuh:56` lists
`end_effector_pose_gradient_hessian`). Adopting it would let HJCD take true
Newton steps near convergence instead of the current GN + dogleg + coordinate
line-search stack (`try_dogleg_step` / `try_coord_linesearch`,
`hjcd_kernel.cu:402-514`), which exists largely to compensate for GN's lack of
curvature. **Opportunity, not a drop-in** — it changes the step math.

### 2c. `J̇` (frame_jacobian_dot) — not needed for static IK

HJCD solves static pose IK (no velocity tracking), so `J̇` and `Λ` aren't on the
critical path **today**. They become relevant only if HJCD adds OSC/nullspace
posture IK (Q3). Worth noting GRiD's `frame_jacobian_dot` is a finite-difference
of J along the integrator flow (`_frame_jacobian.py:268`), not analytic.

### 2d. OSC inertia `Λ` — relevant only if HJCD adds nullspace IK

`osc_inertia_device` self-composes `Λ = (J·M⁻¹·Jᵀ)⁻¹` on device (direct_minv +
J + a 6×6 GLASS `invert_matrix`, `_frame_jacobian.py:350`). HJCD does plain
damped-least-squares pose IK with no secondary task, so Λ is unused now. If a
posture/joint-centering secondary objective is ever added (HJCD already zeros a
"joint-center prior" at `hjcd_kernel.cu:822-824` — a vestige of exactly this
idea), Λ-weighted nullspace projection is the principled version.

### 2e. Mimic support — useful if a gripper is modeled

The Panda's two finger joints are a classic mimic pair. GRiD's non-gradient and
most gradient algorithms support mimic; however **the CUDA `frame_jacobian` path
does not yet support mimic** (`frame_jacobian.rst:86`: "Mimic-joint robots are
not yet supported on the CUDA frame-Jacobian path"). HJCD's current URDFs
(`panda.urdf`, `panda_ext_*dof.urdf`) appear to drive only arm joints, so this
is not blocking today — but it's a gap if grippers enter the IK chain.

---

## Q3. Can GLASS improve HJCD's linear algebra?

HJCD's linalg is **single-lane / single-warp** in the hottest spot. Concrete
opportunities:

### 3a. Normal-equations solve — biggest GLASS win

`build_ne_and_solve_warp` (`hjcd_kernel.cu:362`) and
`build_ne_and_solve_warp`'s sibling `build_ne_and_solve_warp` /
`warp_cholesky_solve_inplace` (`:517-635`):
- The `JᵀJ` build is warp-parallel over rows (good), but the **Cholesky
  factorization is serialized to `lane==0`'s forward/back substitution**
  (`:564-581`: `if (lane == 0) { ... }`), and `chol_solve` (`:292`, used by the
  dogleg path's `recompute_cost_scaled` callers) is **fully serial on a single
  thread**.
- `lm_tuner` launches with **`TPB_lm = 32`** (`hjcd_kernel.cu:2047`) — one warp
  per problem. So the block-cooperative win is bounded by the warp here, but
  even within a warp GLASS's `chol_InPlace` + `trsm` parallelize the
  triangular solves across lanes instead of one lane.

**Recommendation:** Replace `warp_cholesky_solve_inplace` / `chol_solve` with
GLASS `chol_InPlace` (`GLASS/src/L3/chol_InPlace.cuh`) + `trsm`
(`GLASS/src/L3/trsm.cuh`), or — for the damped-LS step — GLASS `invertMatrix`
(`GLASS/src/L3/inv.cuh:10`, block-cooperative Gauss-Jordan, the same primitive
`osc_inertia` uses for its 6×6 invert). For N=7 the gain is modest per solve, but
it removes the serial tail and the bespoke factorization code. **The win grows
with DOF**: HJCD ships `panda_ext_12dof/18dof/24dof.urdf` — at N=24 a serial
single-lane back-substitution is a real bottleneck, and `TPB_lm=32` no longer
covers the matrix.

### 3b. `JᵀJ` and `J·M⁻¹·Jᵀ` style contractions

The `JᵀJ` build (`hjcd_kernel.cu:602-616`) is a hand-rolled GEMM-like reduction.
GLASS `gemm` (`GLASS/src/L3/gemm.cuh`) / `gemv` (`GLASS/src/L2/gemv.cuh`) are the
block-cooperative equivalents. Marginal at N=7; meaningful at N=24 and if HJCD
adopts `osc_inertia`'s `J·M⁻¹·Jᵀ` composition for nullspace IK.

### 3c. Precision note

HJCD already does the `JᵀJ` accumulation in `double` even in the float coarse
path (`(double)J[...]` casts, `hjcd_kernel.cu:606-608`) — good. GLASS is
templated on `T`, so this carries over.

---

## Q4. Other performance improvements

### 4a. Redundant FK recompute across the LM iteration (biggest perf issue)

In each LM iteration, full FK (`X_warp` over the whole chain) is recomputed:
- once to refresh transforms at the top (`hjcd_kernel.cu:872`),
- up to **4× in the backtracking loop** (`:861-914`, each `tries` calls
  `X_warp` at `:872`),
- again in `try_dogleg_step` → `recompute_cost_scaled` → `X_single_thread`
  (`hjcd_kernel.cu:343`, single-thread FK),
- and **2× in `try_coord_linesearch`** (`:492-504`, the `sgn=-1,+1` loop, each
  body calls `recompute_cost_scaled`),
- plus a final post-step `X_warp` (`:978`) and a drift-guard `X_warp` (`:997`).

Several of these are *single-thread* FK (`X_single_thread`) inside a 32-thread
block — 31 lanes idle. **Recommendations:** (1) make every FK call warp-cooperative
(use the `X_warp` path, never `X_single_thread`, inside a warp-sized block); (2)
fuse the residual+cost evaluation so a single FK pass feeds both the pose error
and the scaled residual instead of recomputing; (3) the coordinate line-search
perturbs **one joint** — FK only needs to re-chain from that joint forward, not
from the base (a partial-chain FK), a big saving the hand-rolled code can't
express but is natural if FK is structured per-joint.

### 4b. Launch config / occupancy

- `lm_tuner` runs `<<<Krep, 32>>>` (`hjcd_kernel.cu:2051`) — **one warp per
  block**. With heavy `__shared__` usage (`solve_lm_batched` declares
  `s_XmatsHom[NX*16]`, `s_jointX[NX*16]`, `Ad_sh[N*N]` doubles, etc.,
  `:654-661`) occupancy is shared-mem-bound, but 32 threads/block leaves the SM
  underutilized. Consider 64–128 threads with the spare lanes doing the FK
  chain-up and the `JᵀJ` build cooperatively (which GLASS enables).
- `coarse_search` already does adaptive warps-per-block with a **launch-retry on
  `cudaErrorLaunchOutOfResources`** (`hjcd_kernel.cu:1895-1912`) — good defensive
  pattern. `lm_tuner` has **no such guard**.

### 4c. Silent-launch-failure pitfall (GRiD just learned this)

Per GRiD's `docs/agent_debugging_guide.md` §1c: a heavy kernel with no
`__launch_bounds__` can silently fail to launch at high thread counts
("too many resources requested"), and if you only `cudaDeviceSynchronize()` you
get **zeros masquerading as results**. HJCD's pattern is `cudaGetLastError();`
then `CUDA_OK(cudaDeviceSynchronize());` — but note **many launches call only
`cudaGetLastError()` and discard the return** (e.g. `:1453, :1478, :1840,
:1940, :1968`), which does *not* fail loudly. The collision and sampling kernels
in particular swallow launch errors. Recommend wrapping every launch in
`CUDA_OK(cudaGetLastError())` (check, don't discard) and clamping TPB to
`cudaFuncGetAttributes().maxThreadsPerBlock` (the register cap, which can be
below the nominal thread cap). HJCD does this for `coarse_search` but not
elsewhere.

### 4d. `lm_tuner` `max_iters` vs `solve_lm_batched` `k_max`

`lm_tuner` is launched with `max_iters = 40` (`hjcd_kernel.cu:2048`) passed as
`k_max`, while `HJCDSettings::k_max = 20` is used by `coarse_search`. Not a bug,
but two unrelated iteration caps named alike — worth a comment.

### 4e. `pow` in the inner backtracking loop

`(T)0.5 * pow((T)0.5, tries-1)` (`hjcd_kernel.cu:862`) calls `pow` per try; for
`tries∈{0,1,2,3}` this is a fixed `{1, 0.5, 0.25, 0.125}` — replace with a shift
or a tiny lookup to avoid the transcendental in the hot loop.

---

## Q5. Codegen leverage — stale pin vs regenerate

- **Pin:** `external/GRiD` is at `0a6c18e`; GRiD HEAD is `501adee`. The pinned
  commit **predates the entire new kinematics surface** (`frame_jacobian`,
  `frame_jacobian_dot`, `osc_inertia`, `ee_pose_hessian`) and the recent perf
  work (parallelized gradients/Hessian, ABA floating fusion, per-tier spill,
  silent-launch-failure bench hardening).
- **Free wins on regenerate:** the new `frame_jacobian` / `ee_pose_hessian`
  kernels, plus any improvements to the `end_effector_pose` /
  `load_update_XmatsHom_helpers` machinery HJCD already calls.
- **Blocker:** regeneration **overwrites `grid.cuh` and deletes the hand-patched
  `X_single_thread` / `X_warp` / `init_joint_limits`.** The repo is effectively
  *frozen on a stale pin to protect those patches.* This is the crux. The fix is
  to **move the hand-patched routines out of the generated header** — either (a)
  adopt GRiD's stock FK + `frame_jacobian` and delete the patches entirely
  (preferred), or (b) keep any still-needed custom helper in a separate
  `hjcd_grid_ext.cuh` that `#include`s after `grid.cuh`, so regeneration is
  idempotent. `init_joint_limits` is a good candidate to upstream into GRiD as a
  first-class emitted helper (see backlog).

---

## What GRiD should add/extend to better serve HJCD-IK (our backlog)

1. **Parallelize `frame_jacobian_inner` column assembly.** Today it's
   "correctness-first single-block, serial" (`_frame_jacobian.py:24,160`).
   HJCD's hand-rolled J is one-thread-per-joint parallel; to win the drop-in we
   must match that. Compute independent columns in parallel (aligns with GRiD's
   own codegen-parallelism audit directive).
2. **Emit joint limits as a first-class helper.** HJCD had to hand-add
   `grid::init_joint_limits` to the generated header. GRiD already parses limits
   from the URDF; emit a stock `joint_limits` accessor (host + a `__constant__`
   loader) so customers don't patch the generated file.
3. **Tool/fixed-frame target in `frame_jacobian`.** HJCD's EE is `flange ·
   X_fixed` (a fixed tool offset). Let `frame_jacobian` / `ee_pose` accept a
   fixed child-frame offset (the `-t` fixed-joint target already exists for
   `ee_pose`; ensure `frame_jacobian` honors the same fixed-target so the J is
   computed at the tool frame, not the last joint).
4. **Mimic on the CUDA `frame_jacobian` path** — for grippers (the numpy
   reference already alpha-folds mimic columns; the CUDA path is gated off).
5. **A "kinematics-only" codegen profile.** HJCD needs `robotModel` + local
   transforms + FK + J + (optionally) pose Hessian, and nothing else (no ID/FD/
   CRBA/SO). A lean profile would shrink `grid.cuh` from 10k lines and cut
   compile time. The opt-in key system already supports this; document the
   minimal IK key set (`ee_pose`, `frame_jacobian`, `ee_pose_hessian`).
6. **A `__launch_bounds__` / TPB-clamp recipe in the customer docs.** HJCD
   re-derived the adaptive-warp launch-retry by hand for one kernel and omitted
   it for others. Ship the pattern (it's already in
   `docs/agent_debugging_guide.md` §1c) as customer-facing guidance.

---

## Open questions

1. **Does the `-t` target HJCD passes to `generate_grid.py`
   (`panda_grasptarget_hand`) land on the same frame as the hand-coded
   `X_fixed` (slot 8)?** If yes, GRiD's `frame_jacobian` / `ee_pose` at that
   target are exact drop-ins. If the tool offset is applied *after* GRiD's EE
   frame, the offset must be folded in.
2. **Why was FK hand-patched in the first place** — a perf gap (GRiD's
   `end_effector_pose` was deemed too slow / not warp-cooperative at the time of
   the pin), a missing feature (the flange+tool two-slot output), or just
   convenience? This determines whether stock GRiD FK is acceptable now or needs
   the parallelization in backlog item 1.
3. **Will the 12/18/24-DOF panda variants ship?** They change the cost/benefit
   sharply: the serial Cholesky tail and `TPB_lm=32` that are fine at N=7 become
   real bottlenecks at N=24, raising the priority of the GLASS solve (Q3a) and a
   wider `lm_tuner` block.
4. **Is there appetite for a true Newton step** (adopting `ee_pose_hessian`), or
   is the GN + dogleg + coordinate line-search stack considered tuned-and-frozen?
   The Hessian could simplify that stack substantially.
5. **Will an OSC/nullspace secondary objective** (joint centering, the
   currently-zeroed prior at `hjcd_kernel.cu:822`) be added? If so, `osc_inertia`
   (Λ) becomes directly relevant.
