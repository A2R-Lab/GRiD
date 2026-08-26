# GRiD `humanoid-tier-spill` — Handoff

Status as of 2026-05-23 (branch `humanoid-tier-spill`; RBDReference + URDFParser
submodules on `modernizing-tests`). This is the single handoff for the branch.
It covers three intertwined efforts:

1. **Time-integrator codegen** (Euler / Semi-Implicit Euler / Midpoint / RK3 /
   RK4) — value + gradient. Detailed below (§1-6).
2. **Resource-tier shared-memory spill rollout** — every overflowing kernel now
   fits the sm_120 ~100 KB cap at all tiers via surgical per-tier spill. The
   architecture + per-algo details live in the concepts doc
   `docs/source/user_guide/concepts/resource_tier_system.rst`.
3. **Warning cleanup** — the RBDReference `mxS` NumPy `ndim>0` deprecation is
   fixed at the source.

See **Backlog / open items** at the end for what's left. The floating Euler
gradient bug that earlier revisions flagged as a "CRITICAL OPEN QUESTION" is
**RESOLVED** — see §3.

> **🛠 Debugging / doing a refactor / hunting a mimic or perf bug? READ FIRST:**
> [`docs/agent_debugging_guide.md`](docs/agent_debugging_guide.md) — the validation checklist +
> recurring bug classes (NV-vs-NB scratch sizing, shared-helper NB/NJ index bugs, silent CUDA
> launch failures), debugging methodology, refactor traps, the optimization patterns that worked,
> merge discipline, and oracle gotchas. Distilled from ~15 agent operations. Open systemic-bug
> worklist: [`docs/open-tasks/f2_audit_findings.md`](docs/open-tasks/f2_audit_findings.md).

---

## 1. What was built

GRiD emits time-integrator kernels alongside the existing dynamics algorithms,
in the usual inner / device / kernel / host layering, dispatched at compile time
on `template <typename T, IntegratorType IT>`.

Per integrator, three entry points:
- `integrator<T,IT>`                       — value only, `x_{k+1}`
- `integrator_gradient<T,IT>`              — gradient only, `dAB = [A|B]`
- `integrator_gradient_with_x_kp1<T,IT>`   — both at once

State / shape conventions:
- `q` size `nq` (= `nv` fixed-base; `= nv+1` floating-base, xyzw quaternion).
- `v` size `nv`, GRiD internal order `[ω(3), v_lin(3), joint_v…]` (angular first).
- `x_{k+1}` size `nq + nv` (NOT `2·nv`); host wrappers truncate the
  over-allocated `d_x_kp1` tail.
- `dAB` shape `(2·nv) × (3·nv)`, column-major, columns `[∂/∂q | ∂/∂v | ∂/∂u]`
  all in **tangent space** (so the shape is identical fixed vs floating).

Canonical Python lives in the `RBDReference` submodule (not the adapter):
- `RBDReference.integrator(q,qd,u,dt,integrator_type)` — [RBDReference/RBDReference.py:333](RBDReference/RBDReference.py#L333)
- `RBDReference.integrator_grad(...)`                  — [RBDReference/RBDReference.py:372](RBDReference/RBDReference.py#L372)
- SE(3)/SO(3) Lie-group helpers (`_quat_mul_xyzw`, `_quat_exp_from_half_omega`,
  `_so3_right_jacobian`, `_se3_Q_block`, `integrate`, `dIntegrate`) —
  [RBDReference/RBDReference.py:98-330](RBDReference/RBDReference.py#L98).

`ProjectModelAdapter.integrator / integrator_gradient` are thin pass-throughs
that wrap RBDReference and apply `normalize_vector` / `normalize_matrix`
([reference_backend.py:112-119](RBDReference/equivalents/reference_backend.py#L112)).
`normalize_matrix` is the IDENTITY; RBDReference already emits in GRiD internal
order, so CUDA-vs-ProjectModelAdapter is an apples-to-apples internal-order
comparison.

CUDA codegen:
- `grid_codegen/algorithms/_integrator.py` — value path incl. floating
  Lie-group retract helpers for the q-update.
- `grid_codegen/algorithms/_integrator_gradient.py` — gradient assembly.
  Floating Euler reads precomputed `s_dInt_q_6x6` / `s_dInt_v_6x6` for the top
  `nv` rows; bottom rows are `dt·J_qq | I+dt·J_qv | dt·Minv`.
- `GRID_HAS_INTEGRATOR` / `GRID_HAS_INTEGRATOR_GRADIENT` macros gate consumer
  code so a value-only build still compiles.
- Kernel tier: `RESOURCE_TIER` defaults to `GRID_DEFAULT_RESOURCE_TIER` (so the
  bench `-DGRID_DEFAULT_RESOURCE_TIER=…` macro reaches integrators; TIER_PERF
  behavior unchanged). Python keeps its `TIER_PERF` default.

---

## 2. What is validated (green, trusted)

- **Fixed-base, all 5 integrators, value + gradient + both-at-once**: CUDA ↔
  ProjectModelAdapter(RBDReference) passes at rtol=atol=5e-4 on iiwa14-fixed and
  go2-fixed across dt ∈ {1e-3, 1e-2, 1e-1} and corner + random samples.
- **Floating-base, all 5 integrators, VALUE path**: passes (~5e-7) on
  go2-floating; q-update is the Lie-group retract, cross-checked vs
  `pin.integrate` (`test_integrator_pinocchio_equivalence.py`, 20/20).
- **Floating-base GRADIENT, all 5 integrators** (Euler / SI-Euler / Midpoint /
  RK3 / RK4): value + gradient + both-at-once pass on iiwa14-floating and
  go2-floating at rtol=atol=5e-4, at 32 and 448 threads. The floating multi-stage
  gradients project each stage's `J_qq` columns through the SE(3) `dIntegrate`
  6x6 blocks (`_integrator_gradient.py`); SI-Euler evaluates `dIntegrate` at
  `dt*v_new` and adds the `dInt_v @ dv/dX` top-row matmul.
- The CUDA equivalence suite now **sweeps block thread counts** (1 warp +
  multi-warp + a session-random non-multiple-of-32) so any future thread-count
  race fails the suite. See `test/TESTING_STRATEGY.md`. The integrator smoke
  runner clamps the requested count to `SUGGESTED_THREADS` because the kernels
  are `__launch_bounds__(tier_max_threads<TIER>())` and launching above that
  bound is a hard `cudaErrorInvalidValue` (hit on iiwa14-fixed, bound 352 < 448).
- All five `IntegratorType` instantiations are registered for
  `cudaFuncSetAttribute` (init_grid_kernel_attrs); previously only the default
  (Euler) was, so a non-Euler floating gradient whose arena exceeds the 48 KB
  device default failed to launch.

Test files:
- CUDA equivalence: `test/cuda_equivalents/test_cuda_integrator_equivalence.py`
  (smoke runner `cuda_integrator_smoke_runner.cu`).
- Pinocchio equivalence: `RBDReference/tests/test_integrator_pinocchio_equivalence.py`.
- FD sanity (analytical ↔ finite diff): `RBDReference/tests/test_integrator_gradient_fd_sanity.py`
  (fixed-base only).

---

## 3. The floating Euler gradient bug — RESOLVED 2026-05-22

It was a **CUDA thread-count race**, not a math/convention defect. The whole CUDA
equivalence suite previously launched every kernel at `<<<1,32>>>` (one warp),
while the integrator runner uses `SUGGESTED_THREADS` (448). RBDReference matches
Pinocchio; the CUDA side was correct *at 32 threads* and raced above one warp, so
the error grew with operand magnitude — which earlier debugging misread as a
"velocity-scaled dropped-coupling term."

Root causes, both CUDA-side, floating-base only, in
`_inverse_dynamics_gradient.py` (`gen_inverse_dynamics_gradient_inner`):
1. The `da/du` init zeroed `s_temp[Offset_da_dq]` in one parallel loop and then
   `+=`-accumulated into it in the next with **no `__syncthreads` between** —
   correct only within a single warp. Fixed by adding the barrier.
2. The `MxS(dv/du)·qd` accumulation for the **floating root** had all 6 root axes
   (`dof_id` 0–5 → `jid` 0) `mxX_peq_scaled` into the *same* `da/du` column; that
   helper assumes one writer per destination, so the 6-way `+=` raced. Fixed by
   serializing the root sum onto a single lane.

A **separate, real** bug fixed alongside it: the SE(3) right-Jacobian `Q`-block
had a sign-flipped `c3` coefficient in BOTH `RBDReference._se3_Q_block` and the
CUDA `grid_se3_Q_block` (verified vs `pin.dIntegrate(ARG1)` to ~1e-14). It
affected the SI-Euler floating gradient and any non-tiny `dIntegrate` increment;
it matched Python↔CUDA before only because both were wrong.

After both fixes, standalone `forward_dynamics_gradient_qd`,
`inverse_dynamics_gradient_qd`, and the integrator `dAB` all match the reference
to float32 noise at **32 and 448 threads**.

Fix commits: GRiDCodeGenerator `501501b`, RBDReference `df76001`.

---

## 4. Open / next (nothing blocking; pick up here)

In rough priority order:

1. **Floating FD-sanity test — WON'T DO (covered elsewhere).** A floating
   version of `test_integrator_gradient_fd_sanity.py` would need SE(3)-log
   machinery for the tangent-space q perturbation and output difference. Not
   worth it: the floating integrator gradient is already validated by
   `test_integrator_pinocchio_equivalence.py` (vs `pin.dIntegrate`) and the CUDA
   equivalence suite (g1_floating passes). The FD-sanity test stays fixed-base.
2. **Integrator VALUE-path spill — DONE 2026-05-23.** The value kernel now
   threads the FD inner's `MINV_F_IN_SMEM` lever: at LITE/MINIMAL (and PERF on
   h1_2) the Minv F-region (`6·NV²`) spills to `d_workspace`, keeping the hot FD
   path in smem. h1_2 fixed/floating drop from 103/124 KB to ~41/46 KB.
   `integrator_kernel` gained `unsigned char *d_workspace` (2nd arg);
   `INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T, TIER>` is tier-aware. Validated:
   iiwa14 equivalence passes at TIER_MINIMAL.
3. **Bench the integrator algos — DONE 2026-05-23.** `integrator`,
   `integrator_gradient`, and `integrator_with_gradient` are now timed by the
   bench: `PER_ALGO_SPECS` rows (per-algo path) + measures in the monolithic
   `timeGRiD_{single,batch}.cu` (the default path). Their host signatures take an
   extra `dt` (a fixed bench dt is used; IntegratorType defaults to EULER) and
   `gridData` already allocates the I/O (`d_x_kp1`, `d_dAB`). Verified on iiwa14:
   integrator 15.6 us, gradient 48 us @ N=256 (compute-only).

Out of scope here but on the longer roadmap: **JAX FFI bindings** to replace the
stale Pybind11 layer (generate-compile-run-fast fit; see project memory).

---

## 5. Integrator gradient tier spill (done 2026-05-22)

The integrator-gradient kernel is inner-controlled-placement-aware like
FD/Minv/ABA. The dominant cold buffer `s_D_qdd_stage` (`max_stages·nv·3nv`,
≈59 KB float on g1-floating) lives in smem at TIER_PERF and spills to the
L2-pinned `d_workspace` grad section at TIER_LITE/MINIMAL. Mechanism:
- `select_shared_tier_3way` picks per-tier placement; the kernel emits per-tier
  bodies (like `_emit_fd_du_kernel_body_for_flags`) gated on `RESOURCE_TIER`,
  with `s_D_qdd_stage` conditionally in `extra_t_buffers` (smem) vs pointed into
  `&d_workspace[k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()]`.
- `INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T,TIER>()` is tier-aware;
  `INTEGRATOR_DU_D_QDD_IN_SMEM<TIER>()` gives the placement; `d_workspace` is
  threaded through the kernel + host (L2-pinned when `GRID_INTEGRATOR_DU_USES_WORKSPACE`).
- All five `IntegratorType` instantiations × 3 tiers compile in
  `tier_instantiation_smoke.py` (iiwa14/go2/h1_2). Verified: go2-floating
  gradient matches the reference at TIER_PERF (smem) AND TIER_MINIMAL (spilled),
  32 + 448 threads. Exercise the spill via `GRID_CUDA_INTEGRATOR_TIER=TIER_MINIMAL`.

Note: the arena helper (`gen_declare_shared_arena`) only natively spills the
`s_temp` inner slot via `tier_workspace_expr`; `s_D_qdd_stage` is a cold
extra-buffer, hence the per-tier-body approach rather than a single placement bool.

## 6. Known limitations

- Floating FD-sanity deferred (§4 item 1).
- Integrator value-path is full-smem only (§4 item 2) — fits all current robots.
- **rdc + launch_bounds (FIXED 2026-05-23).** Under `-rdc=true` (the bench's
  single-call build), the integrator kernels failed ptxas at LITE/MINIMAL for
  medium+ robots: `__launch_bounds__(tier_max_threads<TIER>())` budgeted ~64-85
  regs but the (non-inlined) RBD callees need 86-99 (`load_update_XImats`,
  `direct_minv_inner`, `inverse_dynamics_gradient_inner`). The integrator is
  register-bound by those callees, so launch_bounds is now pinned to
  `SUGGESTED_THREADS` (PERF cap, ≤512 → ≥128 regs) at ALL tiers — tier behavior
  remains the s_D_qdd_stage smem spill. Verified: compiles under rdc at all 3
  tiers (go2-floating) + MINIMAL equivalence still passes (iiwa14). The earlier
  "verified at MINIMAL" only covered the inlined (no-rdc) equivalence build.
- **Integrator smem overflow on big floating robots — RESOLVED 2026-05-23.**
  Both kernels now have surgical per-tier spill ladders (see §4 items 2-3 and
  `docs/source/user_guide/concepts/resource_tier_system.rst` → "Integrator
  surgical spill"). Value: spills the Minv F-region. Gradient: 4-rung ladder
  (Dqdd → +dAB+id_du-selective → +whole FD-grad inner) composing the existing
  fd_du levers. All robots/tiers now fit the sm_120 ~100 KB cap — g1_floating
  PERF keeps the hot path in smem via the selective rung; h1_2 (both) use the
  whole-inner rung (the FD-grad inner alone is 160-441 KB, physically can't fit
  smem, so the spill is forced). Verified: all integrator kernels compile at all
  tiers (incl. g1_floating selective rung + h1_2 whole-inner); iiwa14 numerical
  equivalence passes at TIER_MINIMAL (value F-spill + gradient whole-inner) and
  g1_floating passes at PERF (the selective rung). tier_instantiation_smoke
  passes for iiwa14/go2/h1_2 at all 3 tiers.

---

## History

Earlier revisions of this doc carried a long "CRITICAL OPEN QUESTION" thread that
attributed the floating Euler gradient mismatch to a structural dropped
linear↔angular velocity-coupling term in floating-base `forward_dynamics_gradient`,
then went back and forth on which side carried it. **That entire narrative was a
misdiagnosis** — the real cause was the 32-thread launch hiding a multi-warp race
(§3). The lesson is now encoded as a regression guard: the CUDA equivalence tests
sweep thread counts including a random non-multiple-of-32 (see
`test/TESTING_STRATEGY.md`, Principle 2). Do not reintroduce fixed-32-thread
launches.

---

## Backlog / open items — CURRENT TODO LIST

Last updated 2026-05-25. Grouped by theme; rough priority within each.

## CURRENT PLAN — re-prioritized 2026-05-26 (perf-cleanup)

### C.7 OVERNIGHT PERF SWEEP RESULTS — 2026-05-30 (PICK UP HERE)

**Sweep:** `test/benchmarks/results/perf_cleanup_20260530_000808/` (11h40m,
24/24 GRiD cells GREEN, zero crashes/errors). Full matrix: iiwa14/go2/g1/h1_2 ×
fixed/floating × {PERF, LITE, MINIMAL} × {N=1, 16, 256} with `--autotune-threads`.
Pinocchio column merged in from `tier_sweep_20260523_2200/` (pin CPU baseline
doesn't move). Comparison summary file: `grid_vs_pin_summary.md` in the same dir.

**Headline score @ N=256 with-mem:** GRiD-best-tier wins **47 / 80** comparable
cells; pinocchio wins **30**; **3** ties.

**Big GRiD wins (≥2×, sorted):**
- h1_2.fixed ee_pose 6.44× · iiwa14.fixed id_du 5.46× · iiwa14.floating crba
  3.90× · h1_2.fixed id 3.85× · go2.fixed fd_du 3.77× · iiwa14.fixed id 3.20×
- h1_2.fixed aba 2.60× · iiwa14.fixed aba 2.56× · h1_2.floating fd 2.41× ·
  go2.fixed aba 2.35× · h1_2.fixed fd 2.27× · iiwa14.floating aba 2.16× ·
  h1_2.floating aba 2.10× · go2.fixed ee_pose 2.08×

**Tier story:** **MIN tier wins most batch-256 cells on big robots** — confirms
the spill-to-L2 occupancy hypothesis. Examples: g1.fixed id PERF 52.7 → MIN
31.0 µs (40% faster); h1_2.floating id PERF 94.4 → MIN 56.4 µs. **One LITE
regression:** h1_2.fixed id LITE 82.8 µs vs PERF 50.0 µs / MIN 46.9 µs — likely
a bad LITE smem-target on this specific (robot, base, algo) cell (backlog item
A.7).

**Three loss-cluster backlog items filed below** (A.5–A.7):
- **A.5 idsva_so big-robot regression** — pin 3.2–10.5× faster on every
  big robot. Worst: g1.fixed 0.09×, h1_2.fixed 0.11×, g1.floating 0.13×.
- **A.6 ee_pose_gradient big-robot tail** — pin 2.2–6.2× faster on h1_2 /
  big floating. Worst: h1_2.fixed 0.16×, h1_2.floating 0.18×.
- **A.7 crba big-robot floating tail + h1_2.fixed LITE id regression** — pin
  2.1–2.7× faster on g1.floating (0.36×), h1_2.fixed (0.47×), h1_2.floating
  (0.56×).

### ee_pose_gradient GEOMETRIC-JACOBIAN REWRITE — 2026-05-27 (LATEST; pick up here)

**STEP A + B + C LANDED + COMPREHENSIVELY VALIDATED + PUSHED (2026-05-27).**
Branch HEAD: parent `5dde921`, GRiDCodeGenerator `df70675`, RBDReference `0e71d06`.

**Validation matrix (CUDA equivalence vs pinocchio oracle):**
- iiwa14 fixed: 10/10 samples GREEN
- iiwa14 floating: 1/1 zero GREEN + 6/6 non-degenerate (conservative, high_velocity,
  high_acceleration, energetic_random_0, floating_quat_positive, floating_quat_mixed)
- go2 fixed: GREEN (multi-EE branched, 4 leaves)
- go2 floating: 1/1 zero GREEN + 6/6 non-degenerate (branched + floating + non-zero
  pitch — the *key* validation that catches sign errors like the row5 `(sp/cp)*(cy*Jw0 +
  sy*Jw1) + Jw2` formula)
- g1 fixed: 10/10 (29 DOF, 4 EEs)
- g1 floating: 1/1 zero GREEN
- h1_2 fixed: 10/10 (51 DOF, 12 EEs — biggest serial-equivalent we can run)
- h1_2 floating: SKIPPED on pre-existing `forward_dynamics` smem-cap (120 KB > 101 KB,
  the deferred ancestor-scratch de-alias issue) — NOT a regression from this work.
  ee_pose_gradient kernel itself never crashes; FD does, before ee_grad runs.

Floating-zero is degenerate (pitch=0 zeros the `sp/cp` term in row5, identity rotation
hides axis-mapping bugs). The non-degenerate sample set on iiwa14+go2 floating is the
load-bearing validation — both passed. Algorithm is solid.

**Important validation-gap fix:** `FLOATING_CUDA_ALGORITHMS` (the default for floating
equivalence tests) excludes `end_effector_pose_gradient` / `end_effector_pose_hessian`
(they sit in `FLOATING_CUDA_CANDIDATE_ALGORITHMS` instead, opt-in via
`GRID_CUDA_FLOATING_ALGORITHMS=all`). Earlier "iiwa14/go2 floating GREEN" runs were
therefore not actually comparing ee_pose_gradient on CUDA — they tested every other
algo. Fixed with focused re-runs:
- iiwa14-floating `GRID_CUDA_FLOATING_ALGORITHMS=end_effector_pose,end_effector_pose_gradient`
  + 6 non-degenerate samples: GREEN
- go2-floating same args: GREEN (branched + multi-EE + floating + non-zero pitch ⇒
  exercises every codegen path that matters)
The CUDA d/dv ee_pose_gradient is now genuinely validated.

### PERF WIN — Step C closes the floating ee_pose_gradient gap (2026-05-28)

Sweep `test/benchmarks/results/ee_grad_step_c_perf_v2/` (parent `1200115` =
shared-chain geometric Jacobian + dxhom-skip). The dxhom-skip was the load-bearing
change: the new inner doesn't use `s_dXhom` (`(void)`-ed) but the device+kernel were
still computing the per-joint LOCAL d-transforms — for floating base that includes an
expensive quaternion-derivative of the base transform. Setting `include_gradients=False`
in the helpers + passing `s_dXhom=nullptr` to the inner killed that wasted work.

ee_pose_gradient batch (N=256 compute-only µs/prob), GRiD vs pinocchio:
| robot  | base     | OLD-GRiD | new-GRiD | pin     | OLD ratio (vs pin) | new ratio          |
|--------|----------|----------|----------|---------|--------------------|--------------------|
| iiwa14 | fixed    | 0.058    | 0.057    | 0.138   | GRiD 2.38x         | GRiD 2.43x         |
| iiwa14 | floating | 2.116    | **0.178**| 0.133   | pin 15.9x          | **pin 1.34x**      |
| go2    | fixed    | 0.064    | 0.058    | 0.133   | GRiD 2.08x         | GRiD 2.28x         |
| go2    | floating | (n/a)    | **0.179**| 0.151   | (pin 14.0x est.)   | **pin 1.18x**      |
| g1     | fixed    | 0.166    | 0.102    | 0.193   | GRiD 1.16x         | GRiD 1.89x         |
| g1     | floating | 2.200    | **0.219**| 0.288   | pin 7.52x          | **GRiD 1.31x WIN** |
| h1_2   | fixed    | 0.333    | 0.157    | 0.245   | pin 1.16x          | GRiD 1.56x WIN     |
| h1_2   | floating | 2.434    | **0.286**| 0.315   | pin 7.14x          | **GRiD 1.10x WIN** |

Single-call ee_pose_gradient (floating) went 260-312µs → **16-29µs** (10-16× faster);
pinocchio still wins single-call (sub-µs codegen-CPU) but the gap is reasonable now.

### Floating-base first-order vs pinocchio (snapshot from same sweep)

Same `ee_grad_step_c_perf_v2` data — N=256 compute-only µs/prob, GRiD vs pin:
- **Big robots win across the board (h1_2):** aba 1.31x, crba 1.22x, minv 3.83x, fd 2.43x, id 1.34x.
- **iiwa14 (small floating) GRiD loses:** aba pin 1.08x, minv pin 1.51x, fd pin 1.29x. (crba 4.48x WIN, id 1.44x WIN.)
- **go2 (small floating, branched) GRiD loses several:** aba pin 1.75x, crba pin 1.01x ~tie, minv pin 1.11x, fd pin 1.15x. (id 1.35x WIN.)
- **g1 mixed:** aba pin 1.40x, crba **pin 1.73x (biggest concrete loss)**, minv/fd/id WIN.

Top floating-base first-order targets if/when this becomes the next priority:
1. **g1-floating crba** (pin 1.73x) — biggest concrete miss.
2. **go2-floating aba** (pin 1.75x) — second biggest.
3. **iiwa14-floating minv/fd** (pin 1.51x/1.29x) — small robot launch overhead.

### ✅ GPU d2ee floating-base d/dv rewrite — DONE 2026-05-29 (audit-verified)

**Closed via analytic, not the FD-on-Jacobian recipe below.** The original section
described an FD-on-Jacobian approach. The actual implementation went analytic:
- **RBDReference `843a302` (A.1 Python):** closed-form analytic d²(pose)/dv² via
  direct 2nd-order Taylor expansion of the chain world transform. No FD.
- **GRiDCodeGenerator `2bf6d53` (A.1 GPU, parent `080debd`):** analytic chain-
  composition Hessian replaces the older FD-on-Jacobian; iiwa14 fixed+floating +
  go2 floating CUDA equivalence GREEN; matches `RBDReference.end_effector_pose_
  hessian_analytic` to ~1e-13 numpy sim.
- **CUDA emits 6×nv×nv** (the tangent-space output, pinocchio convention).
- **ee_pose_hessian d/dv:** Python `342465d`, GPU codegen `10282e9` (FD-on-
  Jacobian d/dv rewrite, 6*nv*nv output; iiwa14/go2/g1/h1_2 fixed + iiwa14/go2/g1
  floating CUDA equivalence GREEN). The d2ee-vs-pin oracle uses `pin_so_ext`
  (composed `getJointKinematicHessian(LOCAL_WORLD_ALIGNED)`).
- **C.7 sweep will measure analytic-vs-pin speedup at N=256** (deferred until
  the sweep runs).

The original FD-on-Jacobian pickup recipe is preserved below for historical
reference but is no longer the active plan.

---

**ORIGINAL (HISTORICAL) PICKUP RECIPE — superseded by the analytic path above:**

1. **Architectural change.** d2ee inner currently takes pre-computed `s_Xhom`, `s_dXhom`,
   `s_d2Xhom` and assumes one fixed q. FD-on-Jacobian needs to recompute Xhom for each
   perturbed q, which means the inner has to call `load_update_XmatsHom_helpers` + the
   new `end_effector_pose_gradient_inner` internally — requires `d_robotModel` and
   `s_topology_helpers` in the inner signature. Look at the canonical pattern in
   `_eepose_gradient_hessian.py::gen_end_effector_pose_gradient_hessian_inner`
   (line 1362) to see what's changing.

2. **Algorithm.** Mirror `RBDReference.end_effector_pose_hessian` (the FD-on-d/dv-J
   approach): for each `vi ∈ [0, nv)`:
     - SE(3) integrate `q_plus = integrate(q, +h*e_vi)`, `q_minus = integrate(q, -h*e_vi)`.
     - Recompute Xhom for each perturbed q via `load_update_XmatsHom_helpers`.
     - Compute J(q_plus) and J(q_minus) via `end_effector_pose_gradient_inner`.
     - `H[:, :, vi] = (J_plus - J_minus) / (2h)`.
   Then symmetrize: `H[a, i, j] = 0.5 * (H[a, i, j] + H[a, j, i])`.

3. **SE(3) integrate codegen** for the perturbation. Codegen-time specialized per vi:
     - **Fixed-base:** trivially `q_pert = q + h*e_vi` (just add h to q[vi]).
     - **Floating vi ∈ [0, 3):** linear base — `q_pert[0..3] = q[0..3] + h*R(q[3:7])[:, vi]`;
       quat and joints unchanged.
     - **Floating vi ∈ [3, 6):** angular base — quaternion multiplication by small
       axis quat `(h/2 * e_{vi-3}; sqrt(1 - h²/4))`; xyz and joints unchanged.
     - **Floating vi ≥ 6:** arm joint — `q_pert[vi+1] = q[vi+1] + h`; rest unchanged
       (assumes single-DOF arm joints; check `get_joint_index_q` if any multi-DOF).
   Reference implementation: `RBDReference.RBDReference.integrate(q, v_dt)`.

4. **Output shape & consumer ripple.** `d_d2eePos` allocator changes from
   `6*NUM_JOINTS*NUM_JOINTS*NUM_EES` → `6*NUM_VEL*NUM_VEL*NUM_EES`. Update:
     - `GRiDCodeGenerator.py:1157,1161` (gridData allocator).
     - `test/cuda_equivalents/cuda_equivalence_runner.cu` (h_d2ee/d_d2ee sizing,
       memcpy, print_vector). 5+ sites.
     - `bindings/grid_rbd/wrapper_template.cu` (C extern + JAX FFI memcpy + buffer
       validation; the d2ee handler around line 875+).
     - `bindings/src/_core.cpp` (`py::array_t<float> out({batch, 6*nees, nv, nv})`).
     - `bindings/grid_rbd/_handle.py` + `bindings/grid_rbd/jax/__init__.py`
       (d2ee reshape: use `num_vel` instead of `num_joints`).
     - `printGRiD.cu` (`printMat<T,NUM_VEL,NUM_VEL>` for d2eePos block, offset
       calculation uses NUM_VEL).
     - `test/cuda_equivalents/test_cuda_executable_equivalence.py:1041-1054`
       (the d2ee oracle path; already calls `reference_model.end_effector_pose_hessian`
       which now returns 6×nv×nv — so this might be OK already, just verify).

5. **Smem budget.** For h1_2 floating, s_d2eePos = 6*57*57*12 ≈ 900 KB — way over the
   ~101 KB cap. Will need workspace spilling like the existing d2ee (existing macros
   like `GRID_D2EE_USES_WORKSPACE_TEMP` may need re-keying for the new layout). Scratch
   for the per-iter J_plus + J_minus + q_pert + Xhom_pert + inner_temp is ~14K floats
   (~57 KB) for h1_2-floating — fits in smem at PERF tier; LITE/MINIMAL spill the
   inner workspace as the current d2ee already does.

6. **Validation.** After codegen + consumer ripple, run iiwa14-floating CUDA
   equivalence with `GRID_CUDA_FLOATING_ALGORITHMS=end_effector_pose,end_effector_pose_gradient,end_effector_pose_hessian`
   and a non-degenerate sample set (e.g. conservative, high_velocity,
   high_acceleration, floating_quat_positive, floating_quat_mixed). Once GREEN, expand
   to go2-floating then g1/h1_2-floating.

**Independent cleanup tasks the user flagged for later (2026-05-29 EVENING-2 audit
re-verified):**
- Remove orphaned helper `_emit_eepose_grad_compacted_nonserial` in
  `_eepose_gradient_hessian.py:667` (zero callers, ~18 lines). **NOTE: the sibling
  `_emit_eepose_grad_extraction` at line 632 is NOT orphaned — it still has 1 caller
  at line 853** (the eepose_grad_hessian path). Earlier HANDOFF note conflated the
  two; only `_compacted_nonserial` is safe to drop.
- Drop `s_dXhom` param from `end_effector_pose_gradient_inner` signature (still
  `(void)`-marked dead arg at `_eepose_gradient_hessian.py:467`) — propagate through
  device/kernel call sites.
- Per-tier smem allocator can also drop the dxhom-shared logic (the spilling-to-
  workspace path for dxhom is never taken on the gradient path).
- Documentation: add the d/dv convention note + the geometric-Jacobian explanation
  in `docs/source/user_guide/concepts/`, and a section in `docs/open-tasks/archive/python_wrappers_plan.md`.

### ee_pose_hessian d/dv (RBDReference + pinocchio_backend ONLY) — 2026-05-28

Python d²(pose)/dv² landed (RBDReference `342465d`, parent `2748c4f`):
- `RBDReference.end_effector_pose_hessian` rewritten as central-difference FD of the
  (now-correct) d/dv Jacobian on `self.integrate(q, h*e_i)`, then symmetrized. Replaces
  ~285 lines of analytic d²/dq² (whose per-row atan2 second derivatives had known
  orientation-row errors at non-small angles).
- `pinocchio_backend.end_effector_pose_hessian` symmetrized FD of d/dv Jacobian on
  `pin.integrate` — matches the project adapter.
- iiwa14 fixed + floating Python equivalence GREEN.

**CUDA d2ee codegen STILL emits d²/dq² (6×nq×nq).** Default floating equivalence tests
do not include d2ee so they still pass. Opt-in `GRID_CUDA_FLOATING_ALGORITHMS=all` would
shape-mismatch floating because RBDReference now returns 6×nv×nv. The GPU d2ee rewrite
is a separate (large) ripple deferred for now — would need either FD-on-d/dv-gradient on
device or analytic d²/dv² (spatial second derivatives, pinocchio-frame-Hessian style).
Recommendation: defer until a real consumer needs d/dv on GPU.

- **Step C (`df70675` in GRiDCodeGenerator, `d8b1bcb` in parent):**
  - `_eepose_gradient_hessian.py::gen_end_effector_pose_gradient_inner` rewritten as
    shared-chain geometric Jacobian: one FK pass builds world transforms for every joint
    via BFS-level chain-up; per-(ee, chain joint, S-col) compile-time-unrolled column
    fills compute `J_v = aw x (p_ee - p_j)` (revolute) or `J_v = aw` (prismatic) with
    `aw = R_j_world * S_local`; per-ee `(cy,sy,cp,sp)` cached for `E(rpy)^{-1}`; closed-
    form rpy rows: row3 `(cy*Jw0 + sy*Jw1)/cp`, row4 `-sy*Jw0 + cy*Jw1`, row5
    `(sp/cp)*(cy*Jw0 + sy*Jw1) + Jw2`. (Sign on row5's sy term took one iteration to
    catch — adj(E)[2,1] = sy*sp, plus not minus.) The old per-(djid,ee) re-chain + the
    floating compacted nonserial path are dead code (helpers `_emit_eepose_grad_*`
    still in the file as harmless orphans; cleanup pass later).
  - Scratch layout (in s_temp): `Xworld[16*n_joints] | Jv[3*nv*ee] | Jw[3*nv*ee] | E_sc[4*ee]`.
    Much smaller than the old `2*2*16*nq*ee` arena.
  - Output buffer flips `6*nq*num_ees` -> `6*nv*num_ees`. For fixed-base nq==nv so no
    functional change; for floating the base block is now the spatial Jacobian (omega; v)
    matching pinocchio's tangent convention.
  - Ripple: `GRiDCodeGenerator.py` allocator (`d_deePos`/`h_deePos` use `NUM_VEL`);
    `cuda_equivalence_runner.cu` (h_dee/d_dee sized to NUM_VEL); `wrapper_template.cu`
    (FFI memcpy uses NUM_VEL for both C extern + JAX); `_core.cpp` pybind output array
    `(batch, 6*NUM_EES, NV)`; `_handle.py` + `jax/__init__.py` reshape to NV; `printGRiD.cu`
    uses `printMat<T,6,NUM_VEL>`.
  - Fixed-target gradient (`fixed_target_name` != "") raises NotImplementedError in the
    rewrite; the only caller is `examples/quickstart_iiwa14.py` (unused by bench /
    equivalence). Re-add when a real consumer surfaces.

- **Step A (`087d458` in RBDReference):** `RBDReference.end_effector_pose_gradient` rewritten as
- **Step A (`087d458` in RBDReference):** `RBDReference.end_effector_pose_gradient` rewritten as
  shared-chain geometric Jacobian producing d/dv (tangent), 6×nv per ee. Validated 96/96 ee×scale
  combos vs the proven prototype on 5 robots × 2 bases, worst max-abs-err 2.4e-15 (machine
  precision); FD-validated on non-zero offset (~1e-9, FD floor).
- **Step B (`0e71d06`):** `pinocchio_backend.end_effector_pose_gradient` rewritten to d/dv via
  central-difference FD on `pin.integrate(q, h*e_i)` (Lie-group tangent). All 9 robots × 2 bases
  GREEN in `test_kinematics_derivatives_equivalence.py` — RBDReference d/dv ≡ pinocchio d/dv. The
  "match pinocchio" property is proven.
- **Step C — GPU codegen rewrite — PENDING.** Multi-file ripple:
  (a) `_eepose_gradient_hessian.py::gen_end_effector_pose_gradient_inner` — replace per-(djid,ee)
  re-chain with shared-chain FK + per-chain-joint J_v/J_w via cross-products + E(rpy)⁻¹·J_w.
  (b) Output buffer size for floating: `6 * NUM_EES * NUM_JOINTS` → `6 * NUM_EES * NUM_VEL` in the
  `gridData<T>` allocator (`GRiDCodeGenerator.py:1156,1160`), all `6*n*num_ees` strides in the
  codegen, kernel save-result sizes, and the host-copy memcpy bytes.
  (c) Consumers: `wrapper_template.cu` (FFI), JAX bindings reshape, CUDA equivalence harness
  expected shape, `python_wrappers` smoke tests, `printGRiD.cu`.
  (d) For fixed-base nq==nv so no buffer-size change; only floating shape flips. C++ constant
  `NUM_VEL` is already emitted (`GRiDCodeGenerator.py:715`) so no new constant needed.
  Per-robot regen + CUDA equivalence (fixed + floating) is the validation gate.

**PROVEN in Python (`/tmp/geom_jac_proto.py` → `test/benchmarks/_scratch_geom_jac_proto.py`),
ready to implement.** The floating ee_pose_gradient
"outlier" is ALGORITHMIC, not a bug: GRiD re-chains a full 4×4 transform PER Jacobian column
(`O(nq·depth)`); the fix is the shared-chain geometric (spatial) Jacobian (`O(nq+depth)`) — FK once,
then `J_v=â×(p_ee−p_j)`, `J_w=â` (revolute) / `J_v=â` (prismatic), pose-grad `=[J_v; E(rpy)⁻¹ J_w]`.
(The earlier "100× vs fixed" was inflated — fixed ee_pose_gradient is partially LICM-elided in the
bench; real gap ≈16× vs pinocchio.) PROOF: matches the existing analytic gradient to MACHINE
PRECISION (fixed iiwa14/go2/baxter incl. branched; floating arm cols); **3–22.7× fewer 4×4-mults**
(grows with size: h1_2 22.7×). E(rpy) for GRiD's RPY convention (R=Rz(yaw)Ry(pitch)Rx(roll))
verified: E=[[cy·cp,−sy,0],[sy·cp,cy,0],[−sp,0,1]].

**DECISION (user 2026-05-27): output d/dv (tangent, 6×nv) to MATCH PINOCCHIO; document in
RBDReference + GPU code + docs. d/dv ONLY** (no d/dq — no current consumers; d/dq is non-standard
quaternion-component derivs; trivially derivable later via `d/dq = d/dv · base quaternion-rate map`).
Fixed base: nq==nv, unchanged. Floating: switches current d/dq(6×nq) → d/dv(6×nv tangent).

**IMPL PLAN (ordered, with checkpoints):**
1. **RBDReference.end_effector_pose_gradient** → geometric shared-chain Jacobian (d/dv). Preserve
   interface (q, ee_joint_names, ee_offsets → list of 6×nv per ee) + offset + fixed-joint handling
   (offset only shifts p_ee; fixed-joint EE: X_ee=X_world[parent]@fixed_T, chain=parent's). Use
   `get_S_by_id` + `get_joint_index_v` (both exist). Validate new==prototype (machine precision).
2. **pinocchio_backend.end_effector_pose_gradient** → d/dv (pin frame Jacobian, LOCAL_WORLD_ALIGNED,
   mapped to [xyz,rpy] via E⁻¹). Validate RBDReference d/dv == pinocchio d/dv ALL robots = the
   "match pinocchio" proof. CHECKPOINT before GPU.
3. **GPU codegen** `gen_end_effector_pose_gradient*` → shared-chain d/dv (kills the per-column
   re-chain + the floating non-serial compacted path). Re-validate equivalence + perf (expect the
   16× floating gap to close + the fixed path to drop too).
4. Docs + JAX bindings note the d/dv (tangent, pinocchio-matching) convention + the floating
   shape change (6×nv).
5. **RIPPLE (separate task):** ee_pose_hessian (d2ee) needs the same d/dv treatment for consistency.
Proof + scaling + E(rpy) in memory `project_grid_competitive_analysis.md`.

### COMPETITIVE ANALYSIS + crba FIX + SO DATA GAP — 2026-05-27

**crba regression FIXED (code; perf re-measure pending):** root cause = `6cdba85`'s depth-stepped
M-fill (3 `__syncthreads`/chain-depth → sync storm; regressed crba 2.5–6×, even unbranched iiwa14).
Reverted to the sync-free one-thread-per-jid ancestor walk, KEEPING the S-index correctness (M
entry now indexes by the PARENT's S index/sign via compile-time tables). Correctness validating
(b9g9c2kvg): go2 (branched) + iiwa14 PASS; baxter pending. NOT yet committed (gated on baxter).

**Limb-parallelism diagnostic (user asked if go2>iiwa means lost BFS parallelism):** NO — at
single-call go2/iiwa is 0.88–1.39× across all algos (limb parallelism works); go2>iiwa is purely a
BATCH effect (saturated GPU → work-bound, 12 vs 7 DOF). id_du/fd_du clean (1.07–1.08×), disconfirming
a systemic regression. Only crba regressed; minv/aba/crba share a pre-existing ~1.8× go2/iiwa
batch-occupancy gap (not a regression).

**Competitive vs Pinocchio (see memory `project_grid_competitive_analysis.md`):** throughput-vs-
latency split — pinocchio codegen-CPU wins single-call EVERYWHERE (sub-µs); GRiD wins batch N=256
(1.2–11×/prob), except crba (regression artifact → ~parity post-fix). GRiD numbers are compute-only
(GPU-resident assumption). pre_glass id/minv/id_du (iiwa14+go2 fixed, reused): glass HELD/beat it.

**SECOND-ORDER pinocchio data GAP (user-flagged, collect next sweep):** no pinocchio SO timing yet.
The pinocchio harness DOES support `idsva_so_body_frame` (`PINOCCHIO_ALGOS`); `fdsva_so` skipped (no
equivalent). Add the pinocchio column to the next sweep. Caveat: pinocchio nv³ SO codegen for g1/h1_2
may be very slow / hit the 1500s/algo timeout.

**crba FIX COMMITTED + COLLECTION DONE (`crba_so_collection`, EXIT 0):** crba recovered to baseline
(iiwa14 N=256 27.3→11.05µs, go2→13.0, g1→60.4, h1_2→103.7) and now BEATS pinocchio at batch
(2.5×/8.9×/1.2×/1.2×). Pinocchio second-order timing collected for ALL robots (no timeout).
**KEY FINDING — GRiD second-order LOSES to pinocchio on big robots:** idsva_so N=256/prob iiwa14
GRiD 6.8× / go2 4.5× WIN, but g1 pin 2.5× / h1_2 pin 2.7× LOSS. GRiD idsva_so scales badly with DOF
(nv³, compute+smem-bound) → **TOP optimization target** (see memory `project_grid_competitive_analysis.md`).

**NEXT: P4 — crba regression resolved, so the branch is mergeable.** Decide P4 merge
`perf-cleanup → modernizing-tests`. Open perf items (not merge-blockers): idsva_so big-robot SO
scaling (top target), minv/aba/crba batch-occupancy, single-call launch-overhead, idsva_so de-alias.

### P3 PERF SWEEP RESULTS — 2026-05-27
Ran the first trustworthy sweep on the P1-B-parallelized tooling: glass × {iiwa14,go2,g1,h1_2}
× {fixed,floating} × {perf,lite,minimal}, 24/24 cases produced (EXIT 0, ~4h21m). Output:
`test/benchmarks/results/perf_cleanup_overnight/` + report `test/benchmarks/perf_cleanup_overnight.md`.
Compared vs baseline `tier_sweep_20260525_002438` on N=256 compute-only (script in /tmp/compare_sweep.py).

**Headline: 101 wins, 71 regressions (>5%). Net = big-robot dynamics got ~2× faster, but crba
regressed hard.**
- **WINS (the GLASS/inner-owns payoff):** h1_2 (and g1) dynamics roughly HALVED vs baseline —
  h1_2-fixed minv 95 vs 178µs (0.54×), fd 147 vs 285 (0.51×), ee_pose_gradient 85 vs 167
  (0.51×), integrator 0.53×. ~100 cases >5% faster.
- **REGRESSION #1 — crba, systemic, MOST ACTIONABLE:** crba is 2.5–6× slower than baseline on
  EVERY robot, and the gap GROWS with batch (iiwa14 1.6×→2.5×, g1 1.6×→4.9× single→N=256;
  h1_2 ~6×, g1-fixed-minimal 8.5×). It hits even UNBRANCHED iiwa14, so it is NOT just the
  branched S-index correctness fix — it's a systemic crba codegen perf issue introduced
  relative to the 05-25 baseline (prime suspect: the crba P1 "depth-stepped fill" GLASS pass —
  verify whether the baseline pre/post-dates it via git archaeology on `_crba.py`). NEEDS
  root-cause; decide correctness-vs-perf tradeoff with user.
- **REGRESSION #2 — ee_pose_gradient on iiwa14 ~1.9× slower** (all tiers). Minor vs crba.
- **idsva_so whole-arena spill cost (the deferred-de-alias signal): CONFIRMED a real bottleneck
  on big robots** — g1-fixed idsva_so N=256: perf 3001µs → lite 6162 (2.05×) → minimal 5475
  (1.82×); h1_2-fixed: perf 21687 → lite 29455 (1.36×) → minimal 25887 (1.19×). Small robots
  noisy/flat. So the design-notes gate is MET: the deferred surgical ancestor-scratch de-alias
  IS warranted (would avoid the ~2× lite penalty on g1). See `docs/idsva_so_inner_refactor_notes.md`.

**Caveats:** glass-vs-glass only (pre_glass/pinocchio/mjx/frax columns not run); some small-robot
(<100µs) tier numbers are measurement noise (lite occasionally "faster" than perf).

**Suggested next (AM):** (1) root-cause the crba regression (git-bisect `_crba.py` vs baseline;
likely the depth-stepped-fill pass — may need to revert/retune it); (2) given the confirmed
lite/minimal spill cost, schedule the idsva_so surgical de-alias; (3) P4 merge once crba is
understood (don't merge a 6× crba regression unexamined).

### STATUS SNAPSHOT — 2026-05-26 (pick up at the P3 results above)
Branch `perf-cleanup`: **parent HEAD `d5be248`, RBDReference submodule `3e012b7`, codegen
`0b3cd29`, GLASS `3e910e1` — all pushed.** (Uncommitted working-tree noise NOT ours and
left alone: `URDFParser` t8.txt deletion, `test/dev_notes/` deletions [vestigial, to fold
into the Stage-5 sweep], `benchmark_multi_version.md`, an untracked tier matrix md.)

**d2ee BUG IS FIXED (2026-05-26, committed+pushed)** — see "KNOWN BUG — d2ee" below, now
resolved. Root cause: the pitch-row sqrt-term 2nd derivative used `s'_i = T_i/s` as the
quotient-rule numerator where it must use `T_i` (dropped a `1/s` factor), so the error
vanished near home (`s≈1`) and grew with joint angle. Fixed in BOTH the RBDReference
analytic hessian (both joint-loop copies, `3e012b7`) and the GRiD `ee_pose_hessian` codegen
(`_eepose_gradient_hessian.py`, `0b3cd29`). Validated: CPU finite-diff of the verified
gradient on iiwa14/go2/g1-fixed (~1e-11), and CUDA hard-pass vs the pinocchio oracle on
iiwa14 **fixed and floating** (KNOWN_FAILING now empty → d2ee is a hard requirement again).

**SO expansion RESULTS (2026-05-26):** g1 fallback (fixed+floating) GREEN (~12m, compile-
bound), g1 world-frame GREEN (35s), h1_2 world-frame GREEN (60s) — all via the pinocchio
oracle = the SO speed-win proof (was hours of pure-Python reference, now nvcc-compile-bound).
**h1_2 fallback (fixed + floating diag) FAILS**: `idsva_so` requests **168784 B** of shared
mem but the device caps at **101376 B/block** → `GPUassert: invalid configuration argument`.
This is the known device-path smem cap (P2 / perf-cleanup agent-11): `idsva_so` body+world
are the two kernels that still lack the per-tier surgical spill the other 9 kernels got, so
the largest robot overflows even on the forced-fallback path. NOT a regression, NOT from
d2ee. **Decision: SO test defaults stay `iiwa14`** (g1 fallback is a ~12m compile; the
`GRID_CUDA_*_ROBOTS` env vars already make g1/h1_2 zero-code for nightly/broad runs).
g1 is validated-green for permanent inclusion if a slower default is ever wanted; h1_2
fallback must wait for the `idsva_so` spill.

**DONE this session (committed + pushed):**
- **P1-A restructure**: the whole pinocchio-equivalence layer moved into the RBDReference
  submodule. `RBDReference/equivalents/` = reusable lib (conventions + `reference_backend`
  + `pinocchio_backend` + `pin_so_ext`); `RBDReference/tests/` = the suite + its infra
  (manifest/sampling/tolerances/comparators/model_sources/source_lock). Base vs dev
  requirements; README documents the swap. All GRiD importers + `install/developer_install.sh`
  rewired; 936 tests collect.
- **crba S-index fix CONFIRMED** on baxter+fetch fixed+floating.
- **Independent pinocchio oracle wired** as the default (`GRID_REFERENCE_BACKEND=pinocchio`):
  `build_adapter(backend=…)`; project_model (pure-Python URDFParser) is ONLY the codegen /
  model-under-test input, never a value oracle. Executable harness validated green on
  iiwa14 + go2 (fixed+floating). SO tests (idsva_so body+world, fdsva_so) wired to the EXACT
  `pin_so_ext` oracle, validated green on iiwa14.
- **Real d2ee bug found + filed** (see KNOWN BUG below); d2ee compared vs the independent
  oracle + listed in `KNOWN_FAILING_ALGORITHMS` (tracked, loud, non-fatal, NOT masked).

**DONE since the snapshot (committed+pushed):**
- **Vestigial sweep (Stage-5)** — `test/dev_notes/` removed (`c88153b`); restructure audited
  clean (no dead code/dups/orphans/shims); stale doc link fixed (`8c4e9d5`).
- **P1-B perf tooling FIXED** (`27ebfc7`): the sweep serialized CPU-bound nvcc compiles on the
  false premise "GPU work is serial" (iiwa14-fixed alone = 460s compile; big-robot floating =
  the 25-min cases). Split into a parallel **BUILD** phase + serial isolated **MEASURE** phase:
  `run.py` gained `--compile-only` (compile + populate the content-keyed binary cache, skip
  timing) and `--build-dir` (isolate per-case working grid.cuh/.o/.exe — the shared `results/`
  dir was the only real collision risk; caches are content-keyed). `run_multi_version.py`
  gained `--build-jobs` (auto from cores+free RAM, =8 on this box) that fans all GRiD compiles
  across cores, then the existing measure loop runs `--no-recompile` = pure cache-hit + GPU
  timing. `runner_key` hashes only compile inputs (not build_dir/compile_only), so the build
  phase produces the EXACT binary the measure phase loads; timing stays strictly serial →
  numbers unaffected. Validated: cold parallel build of iiwa14-fixed×{perf,lite,minimal} =
  ~499s wall (max, not the 919s sum), 3/3 no collision, measure phase hit every cache (0.0s).

**P2 RESOLVED/RE-SCOPED (2026-05-26):** the idsva_so surgical de-alias spill + inner-owns
placement + fdsva pool→global are ALREADY committed (`a0f3df0`, `86ad706`). The residual gap
— h1_2 body-frame IDSVA-SO requests 168784 B > the ~101 KB device cap *even at the forced
10 KB budget* (both `test_fixed_second_order_forced_fallback[h1_2-fixed]` and
`test_floating_second_order_diagnostic[h1_2-floating]`) — is the DEEP ancestor-pair-scratch
de-alias that `docs/idsva_so_inner_refactor_notes.md` explicitly DEFERS until a perf sweep
proves whole-arena spill is a real bottleneck. Per user (2026-05-26): **sweep-first, defer
the deep de-alias.** Interim: a self-healing device-cap SKIP landed in `_run_runner`
(`308f950`) — any kernel whose most-spilled smem request exceeds the per-block cap now skips
honestly (h1_2 body-frame SO; world-frame is h1_2's production path and passes). Skip
auto-disappears if the de-alias later makes it fit.

**P1-C DONE (2026-05-26) — broad correctness, all 9 robots × fixed+floating vs pinocchio:**
5 clean pass (iiwa14, go2, g1, gen3, fetch). The 4 "failures" were ALL first-time coverage
of robots beyond the old gate, and NONE were core GRiD dynamics-math bugs:
- **Comparator gaps (fixed, `f2c73e9`)** — not GRiD bugs: (A) added a magnitude-scaled atol
  floor `max(atol, rtol·max|expected|)` (mirrors `RBDReference/tests/comparators.py`) so
  large-`Minv` robots (h1_2 near-singular M, entries ~1e6–1e8, rel err ~1e-7) stop
  false-failing; does NOT re-mask d2ee (O(1)-scale). (B) EE-pose orientation rows fold ±π
  (atan2 branch) to exact agreement + a scoped rpy gimbal-lock/derivative-blowup skip;
  POSITION rows always stay strict. → baxter now green.
- **rizon4 = broken upstream asset (fixed, `0e58f74`)**: resolved URDF has 0 `<inertial>`
  blocks (flexiv xacro emits bare inertia tags) → zero-mass model → crba NaN. Honest
  `_model_inertia_is_degenerate` skip (self-heals if a fixed asset resolves). NOT a GRiD bug.
- **fr3 finger-2 position — CLOSED (2026-08-26 note)**: mimic-joint support landed end-to-end
  (URDFParser `resolve_mimic_targets` incl. chained mimic, alpha-fold across all algos); fr3
  is now an active validation robot in the CUDA equivalence suite.
- **h1_2 `direct_minv` dynamic-smem crash — CLOSED (2026-08-26 note)**: the symbol was renamed
  to `minv`, and kernel-attr registration is now uniformly guarded (`init_grid_kernel_attrs`
  checks each algo's smem need vs the device opt-in max before `cudaFuncSetAttribute`, then
  the graceful runtime guard) — the flaky class is gone.
See memory `project_grid_broad_coverage_findings.md`.

**BACKLOG / FEATURE REQUEST (user-filed 2026-05-26):** GRiD doesn't support all URDF
features. Do a single AUDIT of URDF features (`URDFParser` + codegen) vs the spec and add the
missing ones in ONE clean pass later. (2026-08-26: the audit ran — mimic incl. chained, planar incl. skew normals, and
`<dynamics>` damping/friction ALL landed; see docs/open-tasks/urdf_feature_matrix.md for the
current matrix. Remaining candidates: `<limit>` enforcement, `<transmission>`,
`<safety_controller>`/`<calibration>` metadata.)

**h1_2-floating over-cap crash FIXED (`d4d601f`):** the runner's floating block re-registered
kernels with UNGUARDED `cudaFuncSetAttribute` (the generated `init_grid` path is guarded; this
block isn't, for its runner-local kernels). h1_2-floating `forward_dynamics` (120624 B) /
`direct_minv` exceed the ~101 KB cap at PERF tier → hard GPUassert. New
`grid_runner_set_smem_or_skip` helper emits the standard "shared-memory request … device
supports" message + exits cleanly → harness SKIPS (whole-case). Validated: iiwa14-floating
8 passed (no regression); h1_2-fixed 4 passed; h1_2-floating 4 skipped (clear reason). Whole-
case skip is coarse (loses h1_2-floating's fitting algos) — finer per-kernel skip / tier-aware
big-robot testing is future refinement.

**REMAINING TODOS (current):**
1. **OPEN from P1-C:** fr3 mimic-joint → folded into the URDF-feature audit backlog above.
   Also: auto-parallel test sizing (P1-C used a manual 6-way shell shard); finer per-kernel
   over-cap skip + tier-aware h1_2-floating testing.
2. **P3** full sweep vs `tier_sweep_20260525_002438` (now parallelized by P1-B; MUST run with
   the GPU idle — no concurrent compiles/tests — so timing isn't skewed). If it shows
   whole-arena idsva_so spill is a real LITE/MINIMAL bottleneck, THEN do the deferred deep
   de-alias (refactor idea 1 in the notes). Also naming/warnings/docs sweep.
3. **P4** merge `perf-cleanup → modernizing-tests` once validated.

---

Branch `perf-cleanup` (parent HEAD d414915, codegen 6cdba85, GLASS 3e910e1).
Key re-prioritization: **validation + measurement infrastructure (P1) is now ABOVE
further perf work**, driven by two hard learnings this effort — (1) we are optimizing
**unmeasured** (the sweep tooling stalls), and (2) the crba pass uncovered a **latent
branched-robot bug** that the 4-robot equivalence gate did not expose (coverage gaps
hide bugs). Both point to the same fix: a fast reference + broader, faster validation.

**DONE (committed/pushed on perf-cleanup):**
- Inner-owns + surgical-spill correctness wave, all 9 algorithms (validated 6/8 gate
  cases incl. g1 fixed+floating spill rungs; pure-relocation/byte-identical).
- GLASS primitives (segmented_gemv, indexed_batched_gemm, dot_strided_coalesced) wired
  into codegen. 5 perf passes: **id P1** (BFS-level GEMV fusion) + **minv P1**
  (forward-pass sync fusion) — green, racecheck-clean; **crba P1** (depth-stepped fill)
  — green + **fixed a latent parent-vs-jid S-index bug** on branched robots
  (matches `RBDReference H[ind,j]=S_j^T*fh`; numpy old=1.6err/fixed=0); **ee/d2ee P2**
  (floating dense->in-chain compaction, the ~535us outlier) — numpy byte-identical,
  CUDA re-validation DEFERRED to fast-reference; **fdsva_so P3** — documented NO-OP
  (coalesced dot doesn't fit the n^2-stride contraction; SO-spill cost is inherent
  global-mem latency, not coalescing).

**P0 — finish current wave:** DONE. The **crba S-index fix is CONFIRMED GREEN**
(2026-05-26) on the branched robots that exercise the path: baxter + fetch, fixed +
floating, all pass CUDA equivalence (crba never appears in any failure). One incidental
finding: baxter-**fixed** `q=0` is an rpy/atan2 **gimbal-lock singularity** — the EE
pose Jacobian/Hessian is non-finite in BOTH the reference and CUDA (not a codegen bug).
Handled by a harness guard that skips a comparison only when the *reference* itself is
non-finite (a finite reference is always asserted, so no real NaN is masked).

**KNOWN BUG — d2ee orientation rows (found 2026-05-26 — ✅ FIXED 2026-05-26, see STATUS
SNAPSHOT above for the fix + validation):** the
`end_effector_pose_hessian` (d2ee) **orientation (roll/pitch/yaw) rows are wrong at
non-small joint angles in BOTH the GRiD CUDA codegen AND the RBDReference analytic
hessian** — they match each other (so CUDA-vs-analytic equivalence passed and hid it),
but both diverge from the truth as joint angles grow (iiwa14-fixed: orientation-row
error 3e-6 near q=0 → 1.1e-3 at ~0.8 rad → **4.789 at large angle**, analytic max ~8.6).
Position (xyz) rows are correct everywhere. Found by pointing d2ee at the INDEPENDENT
finite-diff/pinocchio oracle (step-convergent, position rows exact, worst q not at
gimbal lock → the FD is the truth). Locus: the d²(roll/pitch/yaw)/dq² atan2 chain-rule
(`RBDReference.py:~1159-1165` + the matching GRiD `ee_pose_hessian` codegen). The CUDA
executable harness now compares d2ee against the independent oracle and lists it in
`KNOWN_FAILING_ALGORITHMS`, so mismatches are reported as a tracked known bug (loud, NOT
masked) but non-fatal. **TODO: fix the orientation-hessian formula in both, then drop
d2ee from `KNOWN_FAILING_ALGORITHMS`.** Note this also moots the "d2ee 72-min pole" plan:
the analytic d2ee was both slow AND wrong; the pinocchio finite-diff d2ee is fast
(~seconds) and correct, so it is the right oracle.

**P1 — Validation & measurement infra (TOP PRIORITY, unblocks everything):**
- **P1-A · Pinocchio + C++ fast reference** (keystone — see §7 item): sub-call
  composition + thin C++ glue, pinocchio kinematics/derivatives for EE, C++ finite-diff
  for the d2ee hessian, a `RBDReference` hooks file (one shared interface both sides),
  hybrid pure-Python fallback. Turns hours-long references into seconds. Then run the
  deferred P0 broad re-validation (now fast) to confirm crba + the whole wave.
  - **DONE 2026-05-26 — relocation + shared interface + one-line swap.** The whole
    pinocchio-equivalence layer moved OUT of `test/pinocchio_equivalents/` INTO the
    `RBDReference` submodule, so the reference is self-contained + self-testing:
    `RBDReference/equivalents/` is the reusable lib (`conventions.py` ← normalization;
    `reference_backend.py` ← the pure-Python adapter; `pinocchio_backend.py` ← the
    pinocchio+`pin_so_ext` adapter — both expose the IDENTICAL surface; plus
    `model_sources`/`source_lock`/`state_sampling`/`tolerances`/`comparators`, the two
    JSON manifests, and the `pin_so_ext` C++ ext). `RBDReference/tests/` holds the 17
    equivalence tests + conftest (817 cases collect green). The **one-line swap** is
    `RBDReference.equivalents.build_adapter(..., backend="reference"|"pinocchio")` /
    env `GRID_REFERENCE_BACKEND` — wired into the CUDA harness at the
    `build_adapter(...)` call. Requirements split: base `requirements.txt`
    (numpy+sympy) vs `requirements-dev.txt` (pin/robot_descriptions/bs4/pybind11/
    pytest). All 7 GRiD importers + `install/developer_install.sh` pin_so_ext path rewired;
    validated both backends build + agree on rnea (iiwa14, 7e-15) and iiwa14 rnea
    equivalence passes through the moved suite. **Committed + pushed** on `perf-cleanup`
    (submodule `abdbd95`, parent `3da85d2`, gitlink bumped).
  - **Pinocchio-as-oracle scope, settled 2026-05-26 (iiwa14-fixed end-to-end test):**
    the pinocchio backend is an **EXACT** CUDA oracle for ALL dynamics
    (rnea/aba/fd/minv/crba/rnea_grad/fd_grad) **and** `ee_pose` **and**
    `ee_pose_gradient`. It is **NOT** a valid oracle for **`d2ee`**: pinocchio's d2ee is
    finite-difference and blows up order-1 near the rpy/atan2 wraps (iiwa14 high_velocity:
    CUDA-analytic −2.08 vs finite-diff +0.11), so the CUDA *executable* harness — which
    compares analytic d2ee — stays a **single pure-Python oracle** (no half-pinocchio
    mix; `build_project_adapter`, documented inline). This revises the earlier
    "C++ finite-diff hessian" plan: keep analytic d2ee; a fast d2ee oracle, if ever
    needed, must be analytic, not finite-diff.
  - **Two distinct multi-hour reference poles (profiled h1_2-fixed, nv=51, 2026-05-26):**
    every dynamics algo + ee_pose/ee_pose_gradient is <1.1s; the poles are (1) the SO
    refs (`idsva_so`/`fdsva_so`) — solved cleanly by the exact `pin_so_ext` oracle; and
    (2) **`ee_pose_hessian` = ~4347s (≈72 min) for ONE evaluation** (12 leaves, analytic
    pure-Python). The executable harness runs d2ee per sample (~10), so h1_2-fixed
    executable equivalence is many hours, dominated entirely by analytic d2ee. Pinocchio
    CANNOT fix this (finite-diff invalid). Options for the d2ee pole (separate task):
    derive an analytic fast d2ee (pinocchio frame 2nd derivatives / C++), down-sample
    d2ee for the largest robots, or accept the cost. Distinct from the SO win.
  - **TODO (still P1-A) — the real pinocchio win:** wire the **EXACT** `pin_so_ext`
    oracle into the **second-order CUDA tests** (`idsva_so`/`fdsva_so`), the actual
    documented multi-hour pole, via the same project(codegen)/reference(oracle) split
    proven on the executable harness; expand those SO tests beyond iiwa14 to the big
    robots (g1/h1_2) where the pure-Python SO reference takes hours.
- **P1-B · Perf-measurement tooling:** `run_multi_version.py` stalls / 25-min compiles /
  can't parallelize (it's a bench). Fix (parallel compiles, isolate only the timing
  window) or build a light per-algorithm timing harness. *We have ZERO perf numbers.*
- **P1-C · Coverage + parallel testing:** once P1-A is fast, validate all 9 manifest
  robots x bases x tiers; add cores+RAM-sized parallel test launch (pytest-xdist `-n`
  or a (robot,base) sharding runner). See "Parallel equivalence testing" item below.

**P2 — Correctness/cleanup debts (from the wave):**
- **h1_2-floating device-path smem cap**: make inline `*_device` paths tier-aware/
  spillable (also completes the deferred **fd audit**) OR guard the runner setattr;
  then validate h1_2-floating kernels. (See §6 item.)
- **`gen_idsva_so_device` dispatcher reconcile**: drop the now-redundant XImats reload,
  thread `scratch_in_smem` by tier (body+world inners now own the load + take d_robotModel).
- **Stage 5**: naming/uniformity sweep; nvcc `-Werror` warnings sweep; code-bloat/
  streamlining audit (consolidate dup helpers across the 14 algos); design-doc
  reconciliation + encode friction corrections (crba NOT called by fd; idsva world cold
  buffer is `Xdown` not `f_w`; `integrator_with_gradient` lives in `_integrator_gradient.py`;
  add base `end_effector_pose_inner` conformance-table row; stale inner-signature docstrings).

**P3 — Data-driven perf (ONLY after P1-B tooling exists):**
- Full sweep (wave + GLASS passes) vs baseline `tier_sweep_20260525_002438`, incl.
  lite/minimal tiers. Then *measured* refinements: id backward distinct-parent grouping;
  aba floating interior-cold-hole; **idsva body-frame ~7x slower than world** (seen in the
  partial sweep block: body single ~3945us vs world ~547us) — investigate.

**P4 — Longer-term:** branch merge perf-cleanup -> modernizing-tests once validated;
remaining naming (§6) + pinocchio-alignment (§7) items.

### Resolved 2026-05-24/25 (no longer backlog)
- **LITE/MINIMAL spill crashes fixed**: (a) null-`s_temp` passed to the
  XImats/XmatsHom load helper in whole-arena spill rungs — `aba`,
  `ee_pose_gradient`; (b) a bug *class* — a single-valued PERF-pick macro
  (`GRID_*_USES_DA_DF_SPILL`) used as a **per-rung inner template flag** —
  `id_du`, `fd_du`, `integrator_gradient`. All committed; generated C++
  verified (LITE=`true`, PERF/MIN=`false`).
- **fdsva_so / idsva_so big-robot fit** via inner-owns-placement
  (`fdsva_so_full_inner` + world-inner `SCRATCH_IN_SMEM` + pool→global tier).
  h1_2 fdsva 198/142 KB → 53.8/45.8 KB; overnight gate + SO equivalence PASSED.
- **Full tier-validation sweep** ran (overnight + targeted re-sweep);
  iiwa/go2 12/12 all tiers; matrix in `test/benchmarks/results/tier_sweep_20260525_002438/`.
- **`ee_pose_hessian` (d2ee) promoted to a first-class algo**: registered in
  `ALGO_REGISTRY`; fixed the codegen single-timing printf label (was
  `EE_POSE_GRADIENT`); added to bench `PER_ALGO_SPECS`; equivalence runner
  standardized (real `hd_data->d_workspace`, removed the bespoke
  `GRID_CUDA_RUN_FLOATING_EEPOSE_HESSIAN` gate). FLOATING accuracy validated
  (iiwa14). Timing recapture + FIXED accuracy are backlogged (§4, §5).

### A. Pinocchio competitive gaps (perf work)
1. ✅ **d2ee analytic d/dv Hessian on GPU DONE (codegen `2bf6d53`, 2026-05-28
   shipped + 2026-05-29 PM verified).** Closed-form analytic via direct
   2nd-order Taylor expansion of the chain world transform; handles per-joint
   Δ + intra-joint multi-DOF (revolute / prismatic / SE(3) free-flyer)
   uniformly via chain composition `L_a · A^{local}_i · P_{a→b} · A^{local}_j ·
   R_b`. Python reference: `RBDReference.end_effector_pose_hessian_analytic`
   (RBDReference `843a302`). Validated to ~1e-9 on iiwa14/floating +
   go2/floating; FD-noise-floor (1.6e-7) on iiwa14/fixed near pitch=π/2. GPU
   path validated via the analytic-vs-pinocchio-analytic oracle (`pin_so_ext`):
   iiwa14 fixed + iiwa14 floating GREEN at machine precision. Full derivation
   in `docs/d2ee_analytic_derivation.md`. **Open speedup measurement
   (analytic-vs-pin at N=256) → C.7 sweep will capture.** Side benefit:
   surfaced an fr3 mimic-joint bug in URDFParser (D.2; partial Python fix
   landed but parked, see Done list).
2. **idsva_so big-robot scaling.** g1/h1_2 lose 2.5–2.7× at N=256 batch; nv³
   kernel is compute+smem-bound. Same family as A.1; ties to ancestor-scratch
   de-alias (B.1).
3. **Core-dynamics batch losses (floating-base only).**
   **2026-05-29 — ABA + Minv FIXED, CRBA per-jid Phase 2 + BFS-parallel
   Phase 1 BOTH LANDED.** ALL losses are/were floating-base. Root cause was
   the single-threaded 6×6 root invert. GLASS `invertMatrix_dense` swap
   landed (GLASS 773cff5, codegen 0e0a9e2, call-site cleanup 617a057, CRBA
   refactor `9eb51a6` per-jid Phase 2 + `809b145` BFS-parallel Phase 1).
   **Measured wins on iiwa14-floating** (pre-`809b145`): ABA 2.12×
   (109.7→51.9 µs, 1.89× pin loss → 1.09× win); Minv 1.77× (66.7→37.6 µs,
   1.11× pin loss → 1.55× win); CRBA 1.26× (35.0→27.8 µs, 1.42× pin
   loss → 1.14× pin loss). **Post-`809b145` + `b140319` surgical XImats
   (2026-05-29 PM):** iiwa14 stays byte-identical (1-wide BFS);
   go2/g1/h1_2 floating expected to pick up additional wins from sync
   reduction (go2 Phase 1 ~24→6 syncs ≈ ~1.8 µs/launch; g1 ~38 syncs ≈
   ~3.8 µs/launch). Surgical XImats `b140319` adds `SKIP_FLOATING_BASE_X`
   template to elide the per-call recomputation of the floating root
   `s_XImats[0..35]` (CRBA never reads X[0]) — closes more of the iiwa14
   residual gap. C.7 sweep will measure all of these. See
   `docs/a3_core_dynamics_floating_loss_audit.md` for the full audit +
   measured A/B table.
4. ✅ **`fdsva_so` pinocchio baseline DONE (parent `c6ab64c`, 2026-05-29).**
   Pinocchio has no direct fdsva_so; synthesized via chain rule:
   `ComputeRNEASecondOrderDerivatives` (RNEA SO tensors) +
   `computeABADerivatives` (Minv + fd_dq + fd_dqd) → apply chain rule. Wired
   into `timing_parser.py` + `run.py` + `generate_report.py` so C.7 sweep
   picks it up. iiwa14 fixed single 22.4 µs/iter, floating 67.4 µs/iter.
   **Open:** numerical cross-check vs GRiD output deferred (formula matches
   Python `pinocchio_backend.fdsva_so` exactly).
5. **idsva_so big-robot regression** — C.7 (2026-05-30, N=256 batch with-mem).
   GRiD loses to pin 3.2–10.5× on every big robot. Worst cells:
   - g1.fixed body_frame: 998 µs pin vs 10,518 µs GRiD (**0.09×**)
   - h1_2.fixed body_frame: 6,632 µs pin vs 61,234 µs GRiD (**0.11×**)
   - g1.floating world_frame: 2,055 µs pin vs 16,021 µs GRiD (**0.13×**)
   - h1_2.floating world_frame: 16,122 µs pin vs 65,740 µs GRiD (**0.25×**)
   - iiwa14.floating world_frame: 404 µs pin vs 1,280 µs GRiD (**0.32×**)
   - go2.fixed body_frame: 325 µs pin vs 641 µs GRiD (**0.51×**)
   - go2.floating world_frame: 495 µs pin vs 2,547 µs GRiD (**0.19×**)
   iiwa14.fixed body_frame is the lone competitive cell (**1.09×**).
   Same family as A.1 / A.2 (compute+smem-bound nv³ kernel); ties to
   ancestor-scratch de-alias / parallelism work. Body and world frames are
   algorithmically equivalent — the dispatcher picks body for fixed, world
   for floating, and we compare against pin's body-frame baseline (pin only
   exposes one variant).
6. **ee_pose_gradient big-robot tail** — C.7 (2026-05-30, N=256 batch
   with-mem). Step-C geometric-Jacobian rewrite closed the floating gap on
   small robots, but big-robot floating and the h1_2 cluster regressed:
   - h1_2.fixed: 62.7 µs pin vs 389.8 µs GRiD (**0.16×**, pin 6.2× faster)
   - h1_2.floating: 80.7 µs pin vs 451.0 µs GRiD (**0.18×**, pin 5.6× faster)
   - go2.floating: 38.8 µs pin vs 91.9 µs GRiD (**0.42×**)
   - g1.fixed: 49.4 µs pin vs 107.5 µs GRiD (**0.46×**)
   - g1.floating: 73.7 µs pin vs 139.7 µs GRiD (**0.53×**)
   The h1_2 regression is the standout — likely an inner-loop bottleneck
   that scales poorly with NEE=12 (h1_2 has 12 end-effectors, more than
   any other robot in the sweep). Audit per-EE accumulation and shared
   chain-walk reuse.
7. **CRBA floating-base tail + h1_2.fixed LITE id regression** — C.7
   (2026-05-30, N=256 batch with-mem). CRBA losses on big floating-base
   robots (consistent with the longstanding "CRBA weakest" memory note):
   - g1.floating: 62.2 µs pin vs 170.9 µs GRiD (**0.36×**)
   - h1_2.fixed: 129.6 µs pin vs 273.0 µs GRiD (**0.47×**)
   - h1_2.floating: 233.9 µs pin vs 415.9 µs GRiD (**0.56×**)
   - g1.fixed: 73.6 µs pin vs 130.7 µs GRiD (**0.56×**)
   - go2.floating: 39.0 µs pin vs 60.4 µs GRiD (**0.65×**)
   Plus a tier-sizing bug: **h1_2.fixed id LITE = 82.8 µs vs PERF 50.0 µs
   / MIN 46.9 µs** — LITE smem-target on this cell is mis-tuned. Should be
   a quick LITE-target re-tune for that (robot, base, algo) triple.

### B. Architecture cleanup (gated on perf data)
1. ✅ **De-alias `idsva_so` / `fdsva_so` inners — DONE in prior commits,
   verified 2026-05-29 (codegen `124be77` docs only).** The body inner now
   has `SCRATCH_IN_SMEM × BC_IN_SMEM` (whole-arena + surgical BC); world
   inner has `SCRATCH_IN_SMEM × COLD_IN_SMEM` (whole-arena + surgical cold
   trio Xdown/v_w/a_w); fdsva_so_device has the canonical 3-lever pattern.
   Conformance audit in `docs/idsva_so_inner_refactor_notes.md` shows ALL
   algorithms now conform to the inner-owns-placement design.
2. ~~**ABA whole-arena → surgical retrofit** (Minv-F sub-split pattern).~~
   **DONE (pre-2026-05-29 audit):** `gen_aba_inner_floating`
   (`_aba.py:7-60`) already exposes the `TEMP_IN_SMEM` × `COLD_IN_SMEM`
   selective-spill levers — the cold `vcross` slab `[36*NJ, 72*NJ)` and the
   floating-base root tail `[140*NJ, 140*NJ+138)` spill via `s_vcross_cold`
   / `s_fb_cold` to packed `d_cold` (`= d_workspace`); the hot recursion
   stays in `s_temp`. This backlog entry was stale.
3. ✅ **idsva body kernel `output_temp` → body-`_device` `SCRATCH_IN_SMEM` —
   ALREADY DONE in prior commits, verified 2026-05-29 (codegen `124be77`
   docs only).** The body kernel at `_idsva_so.py:2390` maps
   `s_temp_in_global=True` → `SCRATCH_IN_SMEM=false` and calls the body
   inner via `gen_idsva_so_body_frame_inner_function_call(... scratch_in_
   smem_expr = ...)`; the inner does the repoint at line 1372.
4. ✅ **h1_2-floating inline DEVICE path smem cap — DONE 2026-05-29 PM**
   (codegen `8f39604` + parent `27c2f0b`). `forward_dynamics_device` is
   now tier-aware: at `TIER_LITE`/`TIER_MINIMAL` the whole FD inner arena
   routes to L2-pinned `d_workspace` (freeing ~120 KB smem). Runner uses
   `TIER_MINIMAL`. Mirrors the established tier pattern from
   `idsva_so_device` / `d2ee_device` / `id_du_device`. **NOT end-to-end
   validated on h1_2-floating yet** (blocked by `idsva_so_body_frame_inner`
   `t_index_map` pre-existing NJ vs NV bug when mimic URDFParser is used;
   not blocking C.7 since parent's URDFParser is mimic-unaware).
5. *(deferred B.1 follow-ups)*
   ✅ **B.5.b cosmetic rename DONE 2026-05-29 (codegen f433d9c, parent
   124e690):** `fdsva_so_inner` → `fdsva_so_contract` across 20 refs in
   `_fdsva_so.py` + `GRiDCodeGenerator.py` + `tier_instantiation_smoke.py`;
   ``_temp_no_inner`` local var also renamed to `_temp_no_contract`.
   iiwa14 fixed + floating regen verified.
   - **REMAINING:** collapse the simple-algo auto-alloc `_device` wrappers
     (id / minv / fd / aba / crba / ee_pose* / integrator / idsva_so_*).
     Touches the equivalence-runner test kernels that still consume them
     (~2-4 hour refactor).
6. **Codegen interface cleanup.** Pays back on every future algorithm
   addition. Sub-items:
   (a) ✅ **Drop thread-group plumbing — FULLY DONE 2026-05-29 (codegen
   `b228756` partial, then `75089f8`/parent `1ff8d54` complete).**
   `b228756` stripped 191 lines of dead `if use_thread_group:` branches
   across 15 codegen files. `75089f8` dropped the `use_thread_group`
   parameter itself across 1079 scrubs in 16 files. Audit 2026-05-29
   EVENING-2: grep `use_thread_group` across `grid_codegen/**.py`
   returns 0 hits. iiwa14 fixed+floating + go2 floating equivalence GREEN.
   (b) ✅ **Consolidate emitter helpers — MAIN PASS DONE 2026-05-29 (codegen
   `75089f8`):** consolidated `gen_kernel_load_inputs` / `gen_kernel_save_result`
   with their `_single_timing` variants (single fn each via optional stride
   kwarg, 74 call sites rewritten). 2026-05-29 EVENING-2 polish also dropped
   `select_shared_tier` (singular, shadowed by 3way), `gen_add_debug_print_code_line`
   (singular), and several dead `_temp_mem_size` shells (codegen `bd1bdd3`).
   Residual: one-off `gen_add_*` shims in `_code_generation_helpers.py` could
   still collapse into a smaller canonical set (parallel-loop helper, workspace
   pointer-carve helper) — fold this into the next emitter rewrite.
   (c) **Dedup repeated branches** — algorithm emitters fan out on
   `compute_c` / `use_qdd_input` / `use_qdd_Minv_input`; factor into
   table-driven helpers. *2026-05-29 scoping note:* surveyed
   `_inverse_dynamics.py`, `_forward_dynamics.py`,
   `_inverse_dynamics_gradient.py`, `_forward_dynamics_gradient.py` (~78
   flag refs). These flags gate conditionally-augmented emission inside
   one body (function-name string, extra param, extra code line) — NOT
   duplicate parallel blocks. Real dedup requires a new design layer
   (emit-spec dataclass per variant, then a single shared traversal) —
   non-mechanical, multi-hour. Deferred until paired with a wider
   emitter rewrite.
   (d) ✅ **Codegen author guide DONE 2026-05-29 (parent 4394343):**
   `docs/source/user_guide/tutorials/adding_an_algorithm.rst` — step-by-
   step worked example using fdsva_so, common pitfalls (caller-side
   s_temp repoint, single-thread invert, PERF-pick macro misuse), helper
   cheat sheet. Linked from the tutorials index.
   Large surgery but high leverage — directly attacks the code-bloat the user
   has flagged repeatedly.

### C. Cleanup + comprehensive perf re-sweep (do as one phase)
1. **Validation matrix completion** — PERF green for all 4×2 (pre-existing);
   **LITE on big robots NOW GREEN 2026-05-29** (g1-fixed + g1-floating +
   h1_2-fixed all PASSED at `GRID_CUDA_TARGET_SHARED_MEM_BYTES=49152`).
   h1_2-floating SKIPPED on the pre-existing `forward_dynamics` smem cap
   (120 KB > 101 KB) → tracks as B.4. MINIMAL on big robots (target=16384
   or similar) still uncollected; ~1 hr run if wanted before C.7.
2. ✅ **Auto-parallel equivalence harness DONE 2026-05-29 (parent
   084af68):** `test/cuda_equivalents/run_parallel.sh` sizes
   `pytest-xdist -n` from `free -g / GB_PER_JOB` (default 5 GB), clamps
   to `nproc`, uses `--dist loadgroup` to keep header cache hot per
   (robot, base). `pytest-xdist` added to `requirements-dev.txt`.
3. ✅ **nvcc/ptxas `-Werror`-style warnings sweep DONE 2026-05-29:** iiwa14
   floating compile clean under `-Wall -Wextra` (no #177-D). The
   `(void)dof_id;` casts at all 5 emit sites in
   `_inverse_dynamics_gradient.py:365` already suppress the
   `inverse_dynamics_gradient_device_qdd` warning class; the earlier
   HANDOFF entry was stale. **Bigger robots / non-PERF tiers still TBD**;
   would re-probe after the overnight sweep.
4. ✅ **Autotune `performance_threads` DONE 2026-05-29 (parent `32ac963`).**
   New `--autotune-threads` flag in `test/benchmarks/baselines/grid/run.py` +
   `run_multi_version.py`. Coarse sweep + midpoint refinement, picks land
   in `algo_picks` of benchmark JSON. ~14 s/(robot, base) wall. Backward-
   compatible. Follow-ups: codegen-cap-aware grid clipping, floating +
   branched untested, single-call path not yet tuned.
5. ~~**`fdsva_so` single_us recapture** — was dropped on -rdc regcount
   error (now fixed).~~ **DONE — verified 2026-05-29:** iiwa14-fixed
   `fdsva_so` reports `single_us` = 49.5 μs in the latest sweep
   (`ee_grad_step_c_perf_v2`); idsva_so + aba also present. The single_us
   pipeline is healthy on all collected algos.
6. ~~**Vendor URDFs instead of pulling `robot_descriptions`.**~~
   **DONE 2026-05-29 (parent 34492f7, RBDReference 990786a):** vendored
   the 9 smoke-tier URDFs (~255 KB total) at `config/robot_assets/` with SHA
   provenance in `URDF_SOURCES.md`. Added `"vendored"` source_kind to
   `model_sources.py`; manifest now lists it FIRST with the
   `robot_descriptions` candidate as fallback. Equivalence resolution
   verified vendored-first; the multi-GB `robot_descriptions` cache is
   now only needed for new robots not yet vendored.
7. **→ Full perf re-sweep** with everything cleaned up. Captures fdsva_so
   single_us + d2ee timing under the standard pipeline (today both required
   manual binary runs).

### D. Reference + runtime modernization (separate effort, post-perf)
1. **URDFParser/RBDReference pinocchio-alignment** — strict-parse API, structured
   diagnostics, `nq`/`nv`/quat-order helpers, mark fixed-base-only methods. See
   `RBDReference/tests/PINOCCHIO_ALIGNMENT_BACKLOG.md`.
2. **URDF feature support — PARTIAL LANDED on submodule branches, NOT yet
   in parent.** Mimic joints (the urgent missing feature) parsed +
   kinematic-Jacobian / analytic-Hessian aware + CRBA/MINV/RNEA-grad/
   FD-grad/ABA mimic-aware (Python) as of 2026-05-29. Pin backend mimic-
   bypass via implicit-function decomposition for FD-grad. Plan for
   CUDA codegen propagation in `docs/d2_codegen_mimic_plan.md`. URDFParser
   `511e398` (on `modernizing-tests` branch) + RBDReference `0b1a89d` (on
   `perf-cleanup` branch). fr3 analytic d²(pose)/dv² now matches FD oracle
   to ~2e-11 (was the A.1 false-positive) and matches pinocchio's
   `getJointKinematicHessian(LOCAL_WORLD_ALIGNED)`. h1_2 (12 mimics,
   multipliers 1.0/1.6/2.4) kinematics + RNEA tests GREEN (173/173 across
   kinematics+RNEA+metadata).
   **Why parked, not bumped into parent:**
   (a) **Dynamics ripple OPEN.** `aba`, `crba`, `minv`, RNEA-grad, fd-grad
   still ASSIGN to mimicked column instead of `+= mimic_scale * …`. The
   Robot.py nq/nv reduction surfaces this; tests
   `test_aba_equivalence[fr3]`, `test_crba_equivalence[fr3,h1_2]`,
   `test_minv_equivalence[fr3]`, `test_integrator_pinocchio_equivalence[fr3]`
   now fail (were passing by coincidence with naive-on-naive). Fix mirrors
   the RNEA pattern.
   (b) **CUDA codegen has NO mimic awareness.** Kernel signatures use
   `NUM_POS`/`NUM_VEL` macros that don't drop mimic; S/Xmat unpack treats
   each joint independently. Bumping submodules into parent without
   codegen propagation makes CUDA-equiv vs RBDReference mismatch on
   fr3/h1_2 across multiple algos.
   (c) **Pre-existing FK orientation bug surfaces here.**
   `URDFParser/Joint.set_type` composes `Xmat_sp_hom` by directly summing
   `t_free + t_origin` without rotating `t_free` through the origin's
   rotation. Invisible on every robot whose origin rpy is zero (iiwa, go2,
   g1, fr3_finger_joint1 — most cases). Mis-places fr3_finger_joint2
   (origin rpy = π around Z). Independent of mimic but blocks fr3
   selection of finger_joint2 as an EE target.
   **Pickup recipe:** finish dynamics ripple in RBDReference; propagate
   mimic awareness into CUDA codegen (NUM_VEL macros, S/Xmat for mimic);
   fix the FK orientation bug; then bump submodules into parent and
   un-skip fr3/h1_2 in CUDA equivalence. Estimated: 1 focused day for the
   dynamics ripple, 1+ day for codegen propagation, ~hour for the FK bug.
3. **PyTorch in-memory compile + re-link → CUDA-Graphs callable.** Use
   `torch.utils.cpp_extension.load_inline` (or equivalent JIT path) to compile
   a per-robot generated header *in memory* and expose the resulting kernels
   as PyTorch ops. The CUDA-Graphs angle is the load-bearing motivation: ops
   that survive `torch.cuda.graph(...)` stream-capture let downstream MPC /
   training loops capture a whole step into a graph, avoiding per-launch
   overhead. Today GRiD generates a `.cuh` that has to be compiled offline
   into a `.so` then loaded — slow iteration loop for users specializing per
   robot. Pair with the grid-rbd binding so PyTorch users can `import` a
   newly-codegen'd robot without touching nvcc.
4. **Runtime mass / inertia parameters (two-variant emit).** Today XImats and
   robot constants are baked into the codegen (consts folded into PTX → fast).
   Add a runtime-parameter variant: kernels read inertias from a per-robot
   parameter struct passed at launch, enabling system-ID, adaptive control,
   parameter sweeps, and online adaptation without re-codegen. Keep the
   baked-constant fast path as the default; the runtime path is opt-in and
   expected to cost ~10–30% (more on small robots where constant folding
   matters most, less on big robots where memory bandwidth dominates). Two
   templates per algorithm: emit both, measure the gap on the standard sweep.
5. **RBDReference file split — user-filed 2026-05-30.** `RBDReference.py` has
   grown SOOOO long (~3000+ lines) — split it the way `grid_codegen/` is
   organized:
   - `helpers.py` — spatial-algebra primitives, quaternion utilities, small
     utility math (the `mx0…mx6` / cross-product / Lie-group helpers).
   - `kinematics.py` — all pose-related: `end_effector_pose`,
     `end_effector_pose_gradient`, `end_effector_pose_hessian` (+ analytic
     variants).
   - `dynamics.py` — `id` (rnea), `fd`, `aba`, `minv`, `crba`.
   - `gradients_and_hessians.py` — 1st and 2nd-order dynamics gradients:
     `rnea_grad`, `forward_dynamics_grad`, `idsva_so`, `fdsva_so`, etc.
   - Top-level `RBDReference.py` (or `__init__.py`) imports each module and
     either assembles a single class via mixins or re-exports a thin facade
     so existing call sites (`from RBDReference import RBDReference`)
     continue to work unchanged.
   **Why:** the file is currently the dominant readability bottleneck —
   browsing for a specific algorithm is painful. The codegen split (one
   `_algo.py` per algorithm under `algorithms/`) is a proven model.
   **Risk:** import / mixin shim has to preserve method-resolution-order so
   `self.X(...)` cross-calls (e.g. `fd` calling `aba`, `aba` calling `crba`)
   keep working. Tests must run unchanged. Not urgent.

### E. Branch merge
- ✅ **`perf-cleanup` → `modernizing-tests` DONE 2026-05-30.** Parent
  `c29cae7..44ba497`. Submodules: URDFParser merge commit `e2a0c9a` (perf-cleanup
  rpy-snap + modernizing-tests D.2 mimic — FK fix landed on both branches as
  cherry-pick); GRiDCodeGenerator FF to `bd1bdd3`; RBDReference FF to `d0e552a`.
  All four pushed to origin/modernizing-tests. Submodule pointers in the parent
  match each submodule's HEAD exactly. GLASS stays on `main` (intentional
  exception per user).

### F. Parallel work batch — 2026-05-30

**✅ MERGE COMPLETE 2026-05-30.** All six tasks merged into `modernizing-tests`
(parent `c9af492`, GRiDCodeGenerator `819328a`, RBDReference `5dcabf4`,
URDFParser `b74e6bc`). Merge order T6→T1→T4→T3→T5→T2 went with only ONE conflict
(a stale T1 comment vs T4's implemented f_ext, resolved to T4); T3/T5/T2 auto-merged
clean. Consolidated post-merge equivalence validation run separately. NOT yet pushed.

**⚠️ DEFERRED FROM F — backlog (the incomplete parts; some are footguns):**
- **T3 mimic CUDA codegen is PARTIAL (P1+P2 only).** Done: id/crba/minv/fd/aba
  mimic-aware, gated byte-identical for non-mimic; fr3-fixed fully passes. NOT done:
  - **P3/P4 mimic GRADIENTS are safe-stubbed to ZEROS and gated out of tests** —
    `inverse_dynamics_gradient`/`forward_dynamics_gradient`/ee_pose-grad/idsva_so/
    fdsva_so return zeros for MIMIC robots (fr3/h1_2). **Footgun:** silently wrong,
    not an error. Fix forward: implement P3/P4 mimic folds, and meanwhile consider
    making the mimic-gradient path emit a clear "unsupported" error vs silent zeros.
  - **h1_2 branched-root `inverse_dynamics` fails** — root-caused: topology-helper
    sizing uses `get_num_pos()`/`get_num_vel()` but sections are built NJ-wide; for
    mimic (NJ>nv) the device reads wrong parent/S-index at multi-joint BFS levels.
    Fix must be **gated fixed-base** (floating non-mimic legitimately has NJ≠nv — a
    blanket swap breaks Gate A; verified + reverted).
  - **`vel_to_body` latent bug** (floating mimic SO path, `_idsva_so_floating_velocity_metadata`)
    — overwrites on shared v-slot; not exercised by the runner (no idsva_so in it) but
    real for P4.
  - Add h1_2 `norm_rtol` override for fd/aba (float32-on-1e6 noise; mirror go2/g1).
  → A **T3-finisher** is the natural next cascade launch (GPU now free).
- **T2 left crba unmodified** (its Phase-2 is already parallel; the gap is batch
  occupancy/tier) → handled by T5's autotune, not a codegen restructure.
- **T5 propose-only follow-ups** (`docs/open-tasks/tier_autotune_followups.md`):
  feed autotuned best-tier back into codegen per-robot defaults; deeper A.7 fix
  (LITE aliases SHARED launch_bounds for no-smem-spill algos) — belongs in B+C.

Six tasks in parallel, each in its own clone + branch off `modernizing-tests`
(except T2 and T6 which run in main repo on their own branches). Per-agent
contract: read `docs/source/user_guide/concepts/{design_principles,
codegen_architecture, resource_tier_system}.rst` +
`docs/idsva_so_inner_refactor_notes.md` + relevant HANDOFF/memory sections,
then produce a written plan that the main agent reviews **before** any
implementation (Phase 1 is read-only).

**Phasing:** Phase 1 (planning, all 6 agents in parallel, read-only) →
synthesis gate (main agent reviews plans + surfaces overlaps to user) →
Phase 2 (parallel implementation with GPU serialized for perf measurement) →
Phase 3 (sequential merge back, lowest-risk first).

| # | Task | Folder | Branch | Sub-agents | Notes |
|---|---|---|---|---|---|
| **T1** | Pinocchio-alignment + RBDReference split + URDF feature audit + comment→notes cleanup | `~/Desktop/GRiD-T1-rbd-align` (new clone) | `rbd-align-split` | 1 | D.1 + D.5 + URDF audit folded. Touches RBDReference + URDFParser. Sub-step: audit pinocchio joint/feature support gaps; propose adds in RBDReference first with notes on CUDA propagation. Move stale "TODO/FIXME/concern" comments out of code into `docs/open-tasks/notes.md` with line refs. |
| **T2** | A.5 idsva_so + A.6 ee_pose_gradient + A.7 crba perf gaps (+ A.4 fdsva_so sanity) | main repo `~/Desktop/GRiD` | `perf-gaps` | 3 (one per algo file) | Different files (`_idsva_so.py`, `_eepose_gradient_hessian.py`, `_crba.py`) — parallel-safe edits, serialize on PERF measurement. |
| **T3** | D.2 CUDA codegen mimic-joint propagation | `~/Desktop/GRiD-T3-mimic-codegen` (new clone) | `mimic-codegen` | 1 (orchestrator) | Follow `docs/d2_codegen_mimic_plan.md`. **MUST NOT perf-regress non-mimic.** Validate iiwa14/go2/g1/h1_2 stay byte-identical at PERF; un-skip fr3/h1_2 mimic CUDA-equivalence at end. |
| **T4** | External-forces threading | `~/Desktop/GRiD-T4-fext` (new clone) | `external-forces` | 1 | Reference GATO `iiwa14_fext.cuh` + spatial_v2_extended; cross-check pinocchio `fext` plumbing. Add `d_f_ext` to RNEA / FD / ABA / ID-grad / FD-grad where appropriate (direct add, in-frame). Python (RBDReference) first, then CUDA codegen. |
| **T5** | PERF-tier rethink + dynamic autotune | `~/Desktop/GRiD-T5-tier-autotune` (new clone) | `tier-autotune` | 1 | Investigate why MIN/LITE beat PERF on big-robot N=256 (smem-occupancy hypothesis from C.7). Rename current PERF→SHARED; add post-codegen autotune that picks PERF = whichever-tier-best-at-current-launch-config. Folds A.7 h1_2.fixed LITE id mis-tune + C.4 autotune follow-ups. |
| **T6** | Plant file + cost/constraint primitives | main repo `~/Desktop/GRiD` | `plant-namespace` | 1 | `namespace grid_plant` over existing `grid`. Expose integrator/grad/hessian + cost/grad/hessian (quadratic state-deviation, quadratic input, EE-position) + constraint barriers (joint pos/vel/torque bounds, mirror GATO `iiwa14_plant.cuh`). Additive, no collisions. |

**Watchpoints flagged at the synthesis gate:**
- T2 + T5 both touch perf-measurement infrastructure; coordinate on `algo_picks`
  / autotune wiring.
- T3 + T1 both touch URDFParser + RBDReference Python layer; coordinate on
  any shared signature changes.
- T4 + T1 both audit pinocchio features; cross-check feature-list outputs.

**Merge-back order (sequential, lowest-risk first):**
T6 (plant additions) → T1 (RBD audit/docs only — split deferred) → T4 (fext) →
T3 (mimic codegen) → T5 (tier autotune) → T2 (perf gaps).

**SYNTHESIS GATE OUTCOME — 2026-05-30 (Phase-1 plans reviewed, user-ratified):**
- **Reference repos located on disk** (resolves T4 + T6 "no GATO/spatial_v2" blocker):
  `~/Desktop/GATO/gato/dynamics/iiwa14/{iiwa14_fext.cuh, iiwa14_plant.cuh}` +
  `integrator.cuh`; `~/Desktop/PDDP/include/cuda-include/{cg_v4_iiwaplant.cuh,
  costGradKern_wEEpose_sequence.cuh, dynamics_arm.cuh}` + `reference/TrajoptCost.py`.
  `spatial_v2_extended` NOT present — pinocchio is the authoritative fext cross-check.
- **Merge danger is smaller than feared:** T4 is the ONLY signature-mutator (trailing
  `d_f_ext`/`s_f_ext`). T5 is a pure value-rename `TIER_PERF→TIER_SHARED` (keeps the
  `RESOURCE_TIER` template param) shipped with a `TIER_PERF=TIER_SHARED` alias so T2
  (merges last, still says `TIER_PERF`) compiles untouched. T2/T3 change ZERO
  signatures (T3 constant-folds all mimic logic; T2 is loop-restructuring). T6 is
  additive (new `namespace grid_plant`). `GRiDCodeGenerator.py` is the one shared
  hotspot (5 tasks, different regions) — MAIN AGENT owns/serializes all edits to it.
- **DECISION 1 (ratified): defer T1's RBDReference file-split (D.5) to B+C.** T1 ships
  only the D.1 pinocchio-alignment audit + URDF feature matrix + comment→`docs/open-tasks/notes.md`
  relocation this batch. The mixin file-split (method→file map is in T1's plan) moves
  to the B+C pass, AFTER T3/T4 land, to avoid rewriting every line they edit.
- **DECISION 2 (ratified): drop T3's `NUM_JOINTS→NUM_POS` macro rename** from
  `d2_codegen_mimic_plan.md`. The macro already == reduced `get_num_pos()`; rename is
  unnecessary and would break the byte-identical non-mimic PERF guarantee. Do all
  raw-NJ mimic work via constant-folded Python integers inside the inners. (T3 also
  found: "size-57 vs 45" HANDOFF note is STALE; real bug is raw-NJ inner loop bounds.
  Genuine latent bug surfaced: idsva_so world-frame `vel_to_body` overwrites for
  shared v-slots — fix in P4, may want its own sub-PR.)
- **DECISION 3 (ratified): T6 plant/integrator hessian = Gauss-Newton outer-product
  of the gradient** (faster/simpler); NO finite-diff hessian; leave the true analytic
  2nd-order hessian as a TODO. (grid has no analytic 2nd-order integrator; adding one
  would violate "additive".)
- **Convention defaults (main-agent-decided, agents reconcile against GATO+pinocchio):**
  T4 replaces the buggy `apply_external_forces` (uninitialized `Xa`, joint-id-as-q) with
  the local-frame subtract convention; T4 f_ext is constant-local-frame so id_du/fd_du
  get it only through the corrected `vaf` (no new gradient term); T4 passes `d_f_ext`
  global straight to the inner (read once at subtract) → `*_DYNAMIC_SHARED_MEM_BYTES`
  byte-identical, no tier-pick perturbation. T6 constraint barriers take explicit bound
  pointers (URDF parses only position limits today — no URDFParser change).

**Backlogged (not assigned this batch):**
- D.3 PyTorch in-memory compile + CUDA-Graphs callable.
- D.4 Runtime mass/inertia parameters (two-variant emit).
- Combined **B+C**: architecture cleanup (`_device` wrapper collapse, table-driven
  emitter dedup, `s_temp_spill` rename, tier-name macros, etc.) + warnings sweep
  (RBDReference `mxS` NumPy `ndim>0` deprecation + nvcc/ptxas on bigger robots) +
  comprehensive perf re-sweep — fold the naming/uniformity residuals from
  [[project-grid-naming-audit-backlog]] in too. Do as ONE phase after F lands.

### G. Round-2 batch plan — 2026-05-30 (PICK UP HERE)

**STATUS 2026-05-30:**
- **F-batch:** ✅ MERGED + validated GREEN (tier smoke / iiwa14 all-algo / fext 3-way /
  fr3-fixed mimic; floating+mimic guarded+test-skipped). Reference-oracle numpy layer
  (plant/energy/centroidal/regressor, 72/0 vs pinocchio) ✅ MERGED.
- **G0 + G1:** ✅ MERGED + validated GREEN (parent `164b655`, GRiDCodeGenerator `85e239c`,
  RBDReference `fe5af8c`). G0 footgun fixed (mimic gradients → clear `NotImplementedError`,
  NOT silent zeros). B+C consolidation: `gen_device_wrapper` (×7 device emitters),
  `gen_tier_dispatch` (×12 tier ladders), byte-identical all robots, ~182 lines removed.
  `grid_plant` now Python-callable (C-ABI + kernels + `grid_rbd` handle). D.3 PyTorch +
  CUDA-Graphs backend (`backend="torch"`, autograd on 4 algos, `urdf_string=`, 1.69× graph
  replay). Fixed a real bug: T4's `d_f_ext` had broken the JAX FFI compile.
- **G2:** ✅ ALL 4 MERGED into `modernizing-tests` (existing emit byte-identical; new
  functionality additive):
  - **warp/thread batched FK** — `X_*`→`ee_pose_inner_{thread,warp}`, de-hardcoded from
    iiwa14, batched `ee_pose_fk_batched` kernel/host + `grid_rbd.fk_batched` ((B×N)→(B×7)
    pos+quat). Validated iiwa14/gen3/go2 B=64.
  - **f_ext gradients** — `∂tau/∂f_ext=−Jᵀ`, `∂q̈/∂f_ext=M⁻¹Jᵀ` (wired+validated vs pin).
    `−∂Jᵀ/∂q` GPU emit DEFERRED (FD-loop bug; numpy/pin oracle ships).
  - **centroidal + R1** — generalized_gravity / nonlinear_effects / energy / com+J_com /
    ccrba(A,h) new additive algos + plant `com_cost`/`momentum_cost`. Validated vs the
    numpy+pin oracles. Centroidal DERIVATIVES (`∂h/∂q`) DEFERRED (value layer complete).
  - **T3-finisher (mimic)** — **fr3 mimic now works FULLY** (fixed + floating, incl.
    id_du/fd_du gradients) vs pinocchio. Fixed: latent `s_vaf` `18·NB`-vs-`18·nv` overflow,
    topology NJ-consistency (`max(nq,NJ)` sizing), `vel_to_body` shared-slot collision —
    all byte-identical non-mimic. Removed id_du/fd_du from G0 refusal (fixed-base mimic);
    un-skipped floating+mimic.

  **G2 DEFERRED (top of next round):**
  - ❌ **h1_2-fixed/floating branched-multi-root ID *value* bug** (norm_rel ~69, worst c[2]).
    Topology indexing now verified correct; the bug is in the branched multi-root (roots
    0/6/12) force-propagation VALUE path (fr3 single-root passes; RBDReference+pin agree
    c[2]=−0.413). h1_2 mimic equivalence is **re-gated to SKIP** with this reason (keeps the
    suite green) until device-level debugged. Needs device-level debugging.
  - **P4 mimic:** `ee_pose_gradient` α-accumulate IS implemented but still G0-refused (ships
    paired with `ee_pose_hessian`, not yet folded); `idsva_so`/`fdsva_so` mimic not done.
  - **floating+mimic gradients** (id_du/fd_du still refuse floating mimic — dense inner
    asserts fixed-base).
  - **`−∂Jᵀ/∂q` GPU emit** (f_ext §A.3) + **centroidal derivatives** (above).
- **NOT yet pushed** to origin (whole F+G stack local on `modernizing-tests`).

---

## OPEN ITEMS — canonical prioritized backlog (2026-05-30)

Single source of truth; supersedes the scattered "deferred" notes above. Detail in
`docs/open-tasks/` (esp. `coverage_parity_matrix.md`, `library_capability_roadmap.md`).

**NEXT-ROUND PRIORITY (user, 2026-05-31):** A1, A2, B2, F1, F2 first → then E1, E4 → then
F3. Any agent deferral triggers a fresh focused explore round to resolve it.

**Round status (2026-05-31):** Round H landed E5/E6 (notebooks + python/torch f_ext) + the
batched-FK cuda test; h-fext (C1) / h-centroidal (C2/C3) / h-mimic (A1+B2) still in flight.
Round I in flight: I-crba (perf de-alias tail), I-regressor (E1), I-urdf (E4). NOTE the
`perf-cleanup` surgical-spill campaign ALREADY LANDED + merged (branch 0 ahead of
modernizing-tests) — Minv/FD/fd_du/aba/integrator/ee/idsva inner-owns + surgical spill +
L2-pin default-on. Deferred TAIL only: crba 2.5-6x regression (I-crba) + idsva_so body/world
deep de-alias (recursion-hot, no clean split — after h-mimic) + the comprehensive perf
re-sweep (= the 4pm sweep). That tail folds into F2; it is NOT a separate pending campaign.

### CLOSED on the 2026-05-31 H/I/J/K agent campaign (merged @ 6a2a20c)
- **A1 DONE** (h-mimic): h1_2 ID value bug = mimic scratch arena under-sized by `get_num_pos()`
  not `get_num_joints()` (`s_vaf` overflowed `s_XImats`) + `id_du`/`fd_du` dropped `alpha*s_sign`.
  h1_2 UN-GATED.
- **A2 DONE** (J-d2ee): the "orientation-hessian bug" lived in a RETIRED d²/dq² path; the
  production tangent d²/dv² (kinematic Hessian) is correct — validated fleet-wide vs pin
  `getJointKinematicHessian`. d2ee UN-GATED to a HARD requirement, NO codegen change.
- **B1 DONE** (K-b1): floating+mimic `id_du`/`fd_du` via per-root-DoF subspace fold.
- **B2 PARTIAL** (K-ee): fixed-base mimic `ee_pose_gradient`+`ee_pose_hessian` un-refused (fr3 green).
- **C1 DONE fixed-base** (h-fext): `−∂Jᵀ/∂q` GPU emit + `f_ext_gradient_dq` kernel/host + test (D1a).
- **C2 DONE** (h-centroidal): centroidal `∂h/∂q` + `ḣ` + pin oracle.
- **D1 DONE**: dedicated f_ext_gradient, 5 centroidal kernels, com/momentum cost, batched-FK cuda tests.
- **E1 DONE** (I-regressor): sysID joint-torque regressor CUDA emit + test.
- **E4 DONE** (I-urdf): URDFParser typed exceptions + continuous-joint verify + planar/spherical
  PARSER groundwork.  **E5/E6 DONE** (H-roadmap): notebooks + python/torch f_ext handle.
- **PERF**: crba dead-buffer removal (3.3× smaller smem, I-crba); idsva_so body `output_tp`
  de-alias rung (K-idsva). Both part of the F2 perf-cleanup tail.

### CLOSED on the 2026-05-31 post-compaction wave (merged @ `182ef91`)
Nine file-isolated agents merged onto `modernizing-tests`; tree green. Each landed with the
hardened checklist (clean-cache equivalence + Gate-A byte-identical / opt-in + floating+fixed
codegen smoke + additive-GCG reconcile).
- **B2-ee-float DONE** (floating-mimic-ee): floating+mimic `ee_pose_gradient`/`ee_pose_hessian`
  un-refused. KEY FINDING: needs NO new fold — the free-flyer root's 6 DoF decompose into 6
  independent v-slot singleton columns, orthogonal to the 1-DoF mimic alpha fold; pure ungate,
  `_eepose_gradient_hessian.py` byte-identical. fr3-floating + h1_2-floating (3-way thumb folds) green.
- **B2-SO DONE (fixed-base)** + **shared `matmul` bug fix** (B2-SO mimic): fixed-base mimic
  `idsva_so`/`fdsva_so` un-refused via the internal-NUM_BODIES sweep + alpha-fold. ROOT CAUSE of the
  long-deferred NB>NV value bug = a SHARED helper: `gen_matmul` used `36*((index/num)%NUM_JOINTS)`;
  for mimic robots (NB>NJ) the last mimic body wrapped to block 0 and read body-0's inertia,
  corrupting the whole composite-inertia chain (no-op for non-mimic NB==NJ). Fixed `%NUM_JOINTS`→
  `%NUM_BODIES`. fr3-fixed SO green PERF+spilled. Floating-base mimic SO still refused.
- **E2 DONE (numpy + CUDA)**: general-frame Jacobian J (LOCAL/WORLD/LOCAL_WORLD_ALIGNED) + J̇ +
  OSC Λ=(JM⁻¹Jᵀ)⁻¹. numpy ref vs pin (getFrameJacobian/getJointJacobian/computeJointJacobiansTimeVariation);
  CUDA `frame_jacobian`/`frame_jacobian_dot`/`osc_inertia` (opt-in, validated on-device on
  iiwa14/go2/g1 × 3 frames). Λ kernel takes precomputed M⁻¹ (on-device compose = follow-up).
- **g1-spill DONE**: g1-floating now runs `f_ext_gradient` + `fd_parameter_gradient` via surgical
  smem-cap spill (135→94 KB, 99→75 KB) to the L2-pinned SO workspace; Gate-A PERF arena byte-identical.
- **PERF (SO audit G4 partial)**: `id_du` branched-fixed + mimic-dense column-parallel (K-iddu);
  world-frame `idsva_so` forward-sweep parallelized + Xdown/rt-rp dedup (SO-idsva); `fdsva_so`
  timed/untimed emitter dedup byte-identical + A2 Minv-apply hotspot profiling PLAN (SO-fdsva).
- **TEST/DOCS**: `docs/open-tasks/test_coverage_matrix.md` (per-algo × robot both-layer matrix) +
  `test_cuda_matmul_blockwrap_regression.py` (NB>NV pin, passes/fails-on-revert); README + CHANGELOG +
  new `frame_jacobian.rst` + stale-mimic corrections (docs-sweep); coverage-fill (robot-default
  widenings + the continuous-joint CUDA test = the one algo-family that had no CUDA coverage).

**Second backlog batch (post green-gate, while holding for the ~9pm sweep):**
- **floating-mimic SO LANDED** (8547ea2): floating-base mimic idsva_so/fdsva_so via the world-frame
  inner's internal-coordinate scatter-fold → MIMIC SECOND-ORDER now COMPLETE (fixed + floating).
- **E2 on-device Λ Minv-compose LANDED** (516c15e): `osc_inertia` is self-contained.
- **F1 RBDReference split — DEFERRED (broken).** The agent's split (4006→45-line shell + mixins) had an
  IDENTICAL public surface (126/126) but the full numpy suite diverged: **131 failed / 824 passed** vs the
  main baseline's **38 failed / 917 passed** → ~93 regressions (mis-wired mixin MRO / cross-`self` helper
  references hidden behind the identical surface). NOT merged; main stays single-file. LESSON: a mechanical
  4006-line class→mixin split must verify cross-method `self.` resolution + shared-helper inclusion against a
  FULL before/after suite diff, not just the public-name set. Future redo: split incrementally (one mixin,
  re-run suite, repeat) rather than all-at-once. (RBDReference baseline itself has 38 pre-existing fails:
  h1_2 minv, plant-floating, floating-quaternion d2tau — known pinocchio-alignment gaps.)
- Pre-sweep green-gate PASSED earlier: **31 passed / 1 skipped / 0 failed** (iiwa14/go2/g1/fr3 full
  matrix, fresh-compile @ 49f2208). CUDA tree validated for the sweep at the current tip.

### REMAINING backlog (prioritized)
**A. Correctness — open/deferred**
- A-d2ee-gate. **RESOLVED 2026-05-31 (`5ff1ce1`).** Integrated green-gate caught h1_2-fixed d2ee
  emitting exact-0 columns — the fixed-base mimic d2ee fold (B2-ee, fr3-validated) didn't sum over
  MULTI-BLOCK v-slots (h1_2 thumb = proximal + 2 mimics → 3 blocks on one slot; last-writer-win
  dropped terms). Fix: emit the alpha-weighted block-pair SUM per shared v-slot. h1_2-fixed +
  fr3-fixed d2ee GREEN on clean rebuild; gate removed; d2ee HARD for all robots. (Lesson: the
  integrated gate caught a coverage gap no isolated-clone validation could — fr3 has 1 mimic
  joint, h1_2 has 12. Also: clear the generated-header cache before re-validating, or stale
  headers give phantom id_du/fd_du failures.)

**B. Mimic codegen — remaining deferrals**
- B2-ee-float. **CLOSED 2026-05-31** (floating-mimic-ee) — see post-compaction closures above.
- B2-SO. **CLOSED — FIXED + FLOATING 2026-05-31.** Fixed-base = internal-NB body-frame sweep +
  alpha-fold (root cause of the old NB>NV bug = shared `matmul` `%NUM_JOINTS`→`%NUM_BODIES` block-wrap).
  Floating-base = the world-frame inner's internal-coordinate scatter-fold (per-root-DoF treatment
  emerges from per-column internal slotting); fr3-floating green. h1_2 SO stays world-frame non-mimic
  scope (NB=51/52 internal slab impractical) — fr3 is the landed mimic case (fixed + floating).
  Only mimic refusals left: integrator gradients + `f_ext_grad`.

**C. Partial-feature remainders**
- C1-float. **ALREADY DONE** (the A.3 `−∂Jᵀ/∂q` block emits for BOTH bases — floating perturbs via
  `grid_integrate_floating_q`; tested go2/g1-floating in `test_cuda_f_ext_gradient_equivalence.py`).
- C3. **ALREADY DONE** (`coriolis_matrix` handles fixed+floating via the Bcrb recursion; tested in
  `test_energy_equivalence.py`). [Both were stale REMAINING entries — corrected 2026-05-31.]

**E. Roadmap — not started**
- E1-rem. mimic regressor `Y` CUDA validation **ALREADY DONE** (fr3-fixed case landed at `2b5afbe`,
  confirmed green on fresh compile 2026-05-31 — stale entry). FD param-gradient `∂q̈/∂π=−M⁻¹Y` also
  DONE (g1-spill validated). REMAINING: runtime-inertia two-variant emit (plan only, `d4_*` doc); CRBA regressor.
- E2. **CLOSED 2026-05-31** (numpy + CUDA J/J̇/Λ). On-device `osc_inertia` M⁻¹ compose **also DONE**
  (self-contained kernel folds `direct_minv_inner` in; iiwa14/go2/g1 green). REMAINING follow-up:
  mimic-robot frame-Jacobian CUDA (gated ¬mimic — needs the multiplier/effective-angle fold + the
  crba-based mimic Minv branch validated).
- E3. **Contact/constraint dynamics** (constraint Jac, KKT/Delassus, constrained FD, impulse) — biggest moat vs frax.
- E4-cuda. planar/spherical CUDA multi-column-S emit (parser groundwork landed; needs multi-column-S
  + NV≠NQ codegen — guard raises `UnsupportedJointTypeError` today). Other joint types (helical/translation/composite).
- **E-customer. CUSTOMER-DRIVEN roadmap** (from the GATO/PDDP/HJCD-IK reviews —
  [`docs/customer-reviews/SYNTHESIS.md`](docs/customer-reviews/SYNTHESIS.md)). All 3 customers are on
  STALE GRiD/GLASS and hand-roll plant/2nd-order/linalg we now ship. Demand-ranked asks:
  (1) **`plant_step_hessian`** (analytic 2nd-order plant Hessian — `_plant.py` omits it; PDDP+GATO
  justify it = highest-value addition); (2) **auto-allocating `plant_step_gradient`** overload (GATO
  ergonomics — biggest cutover friction); (3) **parallelize `frame_jacobian_inner`** (HJCD: it's
  correctness-first SERIAL, slower than their hand version — our newest kinematics needs the
  column-parallel treatment); (4) fused terminal-aware `cost_value_grad_hess` (GATO); (5)
  constraint primitives beyond log-barriers — aug-Lagrangian/penalty (PDDP); (6) frame_jacobian
  fixed-tool-frame target + joint-limits emitted helper (HJCD); (7) IntegratorType SI-Euler +
  trapezoidal byte-match (GATO); (8) lean kinematics-only codegen profile (HJCD); (9) upstream GATO's
  block-tridiagonal SpMV into GLASS. Cross-cutting: a customer "integrating/regenerating GRiD safely"
  guide (the silent-launch-check pitfall hit ALL 3 in the wild) + kill the "patch-the-header"
  anti-pattern that freezes their pins.

**F. Cleanup**
- F1. D.5 RBDReference file-split (mixin map: `rbdreference_split_plan.md`). **ATTEMPTED + DEFERRED
  2026-05-31** — all-at-once split broke ~93 tests (mis-wired mixin MRO behind an identical public
  surface). Redo INCREMENTALLY: one mixin at a time, full numpy suite green after each.
- F2. Naming/uniformity + warnings sweep + **two systemic bug-class audits** — full worklist in
  [`docs/open-tasks/f2_audit_findings.md`](docs/open-tasks/f2_audit_findings.md):
  (A) **silent CUDA launch-failure pattern** — smoke runners that launch without a
  `cudaGetLastError` check (osc-Λ zero-output was one; `cuda_centroidal_smoke_runner.cu` +
  `cuda_plant_smoke_runner.cu` still unguarded). (B) **per-body scratch sized by NV not NB**
  (mimic overflow class — 3 fixed this session; `_centroidal.py` `s_vaf=18*n` is a latent suspect).
  Plus the `grid::SUGGESTED_THREADS`→perf-cap printGRiD.cu rename, nvcc/ptxas warnings, naming
  uniformity. (`mxS` ndim>0 ALREADY fixed — dropped.) Perf tail: floating idsva_so de-alias,
  crba MINIMAL 3rd-rung, idsva_so big-robot gap. Do AFTER the sweep (needs codegen/compiles).
- F3. T5 propose-only: autotuned best-tier → per-robot codegen defaults; LITE-aliases-SHARED launch_bounds.

**G. Housekeeping**
- G1. NOT pushed to origin (whole H/I/J/K stack local on `modernizing-tests`).
- G2. Deliberate skips: fr3-floating max-threads; h1_2 body-frame SO 168KB smem-cap (world-frame is production).
- G3. **Merge-checklist lesson (2026-05-31):** after any URDFParser/Robot.py or shared-GCG.py merge,
  run a FLOATING+FIXED codegen smoke (`cli <urdf>` and `cli <urdf> -f`), not just py_compile — a
  floating-codegen regression (E4 single-axis-S guard hitting the 6-DoF root) slipped past py_compile.
- G4. **SO audit (user-requested, later) — full plan: [`docs/open-tasks/so_audit_plan.md`](docs/open-tasks/so_audit_plan.md).**
  Dedicated second-order (`idsva_so` body+world, `fdsva_so`) pass: A parallelism (world-frame
  thread-0 6-vec chains; the twice-rejected fdsva_so Minv-apply hotspot — Nsight+transpose first),
  B dedup (Xdown Plücker-inverse 3× [P-dedup may land this]; reference-order assembly 2×; fdsva_so
  timing/non-timing emitters), C mimic-SO completion (the NB>NV internal-sweep VALUE bug J-idsva
  built+reverted), D the floating-reference fallback decision. The fallback
  `gen_idsva_so_body_frame_floating_reference_inner` (~`_idsva_so.py:959`, ~700 lines) is confirmed
  NOT-emitted-in-production + NOT a live oracle but a DELIBERATELY-KEPT fallback ("world-frame
  co-exists with" it, ~`_idsva_so.py:2606`). Decision (user, 2026-05-31): KEEP for now; flagged
  inline at the def. Do NOT delete without the audit.

**API STABILITY NOTE:** the core/benchmarked algorithm signatures are STABLE now — the only
signature churn (T4 `d_f_ext`, T5 `TIER_PERF`→`TIER_SHARED`+alias) is DONE+landed, and the B+C
consolidation was byte-identical. Remaining work is additive (new algos: centroidal/f_ext-grad/
frame-Jac/OSC/contact), internal bug-fixes (no signature change), or — the ONE deliberate
API-changer — the F2 naming/uniformity pass (renames for consistency, perf-neutral). So a perf
sweep on the current green tree stays valid through the remaining work.

---

Scope chosen by user: all four feature tasks + **B+C consolidation FIRST**. Sequencing
resolves "T3-finisher mandatory-first" vs "consolidate-first" by splitting the footgun
fix (immediate) from the full mimic-gradient implementation (after consolidation).

- **G0 — footgun hotfix (immediate, tiny):** make the MIMIC gradient codegen path emit
  a clear "unsupported" compile/runtime error instead of silently returning ZEROS
  (`inverse_dynamics_gradient`/`forward_dynamics_gradient`/ee-grad/idsva_so/fdsva_so on
  mimic robots). Neutralizes the F-deferred footgun without waiting for P3/P4.
- **G1 — consolidate (serial on codegen core) + bindings (parallel):**
  - **B+C architecture consolidation:** device-`_device`-wrapper collapse +
    table-driven tier-dispatch dedup (`docs/open-tasks/bc_cleanup_plan.md` items 1–2).
    Byte-identical validation. New G2 algos emit against the deduped base.
  - **Bindings track (parallel, `bindings/grid_rbd/` — independent of codegen core):**
    D.3 PyTorch in-memory compile + autograd + CUDA-Graphs + notebook UX
    (`d3_pytorch_cudagraphs_plan.md`) AND the `grid_plant` Python/handle surface
    (CUDA-only today). + reference-oracle numpy layer finishing (merges at back).
- **G2 — feature wave on the deduped base (parallel among themselves, different algos):**
  - **T3-finisher:** mimic P3/P4 gradients (id_du/fd_du/ee-grad/idsva_so/fdsva_so) +
    h1_2 branched-root ID topology fix (NJ/nv, fixed-base-gated) + `vel_to_body` fix +
    h1_2 fd/aba `norm_rtol`. Replaces the G0 error path with real support.
  - **Centroidal + R1 quick wins:** energy/g/Coriolis + CCRBA A(q)/h + CoM/J_com
    (`centroidal_quickwins_plan.md`). Oracle = reference-oracle layer.
  - **f_ext gradients:** ∂tau/∂f_ext=−Jᵀ, ∂q̈/∂f_ext=M⁻¹Jᵀ, ∂(id_du)/∂f_ext
    (`differentiability_extensions_plan.md §A`).
  - **Warp/thread FK fix+rename (customer):** generalize beyond hardcoded iiwa14 +
    fix broken `mat4_mul` branch + rename `X_{single_thread,warp}`→`ee_pose_inner_{thread,warp}`
    (`kinematics_warp_thread_plan.md`).
- **Deferred to a later round:** D.4 runtime inertia + regressor; joint types
  (continuous-first); notebook examples (needs bindings); R4 frame Jacobians; R5 OSC/Λ;
  rest of B+C (D.5 split, naming, warnings, comprehensive perf re-sweep).

### Future directions / roadmap (planned during the F-batch downtime — 2026-05-30)

Full implementer-ready plans live in `docs/open-tasks/`. `library_capability_roadmap.md`
is the umbrella (GRiD vs Pinocchio vs **frax** = arXiv 2604.04310, the direct JAX
competitor; GRiD's moat = analytic 2nd-order + contact/constraint + control/MPC, which
frax/Brax lack). Index:
- `library_capability_roadmap.md` — capability matrix + R1–R5 prioritized adds.
- `centroidal_quickwins_plan.md` — R1 energy/g(q)/Coriolis (near-free), R2 centroidal
  CCRBA `A(q)`/`h`/`ḣ` + derivatives (flagship), R3 CoM + CoM-Jacobian. Quick wins.
- `kinematics_warp_thread_plan.md` — **customer-driven, near-term:** verify + rename the
  no-derivative single-thread/warp FK path (`ee_pose_inner_{thread,warp}`) for
  sampling-based kinematics. EXECUTE AFTER T2+T3 merge (they touch `_eepose_gradient_hessian.py`).
- `joint_types_plan.md` — more joint types (continuous cheap; planar/spherical reuse the
  floating-base + mimic NV≠NQ machinery), ripple-rated, cheap-first.
- `d3_pytorch_cudagraphs_plan.md` (PyTorch + CUDA-Graphs + notebook UX),
  `d4_runtime_inertia_params_plan.md` (runtime inertia + sysID regressor),
  `differentiability_extensions_plan.md` (∂/∂f_ext, ∂(du)/∂π derivative matrix),
  `notebook_examples_plan.md`, `rbdreference_split_plan.md` (D.5 split + grid_plant
  numpy ref), `urdf_feature_matrix.md`, `notes.md`.
- Follow-ons noted: expose `grid_plant` via a `grid_rbd` Python/handle surface (it's
  CUDA-only today); add RBDReference numpy refs for plant + regressor.

### Done — historical log archived

The detailed completed-work narrative (pre-2026-05-31 batches) moved to
[`docs/HANDOFF_ARCHIVE.md`](docs/HANDOFF_ARCHIVE.md) to keep this tracker lean.
Recent closures are summarized in OPEN ITEMS above; per-merge detail is in `git log`.

### Reference docs (kept separate, linked from here)
- `docs/source/user_guide/concepts/resource_tier_system.rst` — tier/spill architecture.
- `docs/idsva_so_inner_refactor_notes.md` — deferred inner de-alias design.
- `docs/open-tasks/archive/python_wrappers_plan.md` — grid-rbd bindings (historical plan; v0.3 shipped; archived).
- `test/benchmarks/overnight_tier_sweep.md` — partial sweep results (pre-fix run).
- `RBDReference/tests/PINOCCHIO_ALIGNMENT_BACKLOG.md` — Pinocchio alignment.

