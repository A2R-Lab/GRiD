# GRiD — Backlog to Full Correctness & Completeness (2026-06-01)

The shared work-list to get every supported feature/algorithm to {correct, complete, uniformly named,
fully tested}, single-block kept, no back-compat. Status: ✅ done · 🔵 in progress · ⬜ open.
Detail/evidence: `api_completeness_audit.md`, `rename_mapping.md`.

## 1. CORRECTNESS — known wrong results (must-fix bugs)
- ✅ **B1 / C1 — `ee_pose_hessian` orientation-hessian** — STALE backlog item: already fixed by the prior
  chain-composition `d²/dv²` rewrite on BOTH surfaces. Verified empirically (2026-06-01): analytic-vs-FD
  ~1e-11; the residual vs pin (~1e-5) is pinocchio's own `getJointKinematicHessian` FD floor (FD-vs-pin ==
  analytic-vs-pin), NOT a GRiD error. Numpy test already at full-fleet scope (63 pass incl. 18 hessian-vs-pin
  fixed+floating); CUDA confirmed fresh (non-cached) on iiwa14-fixed + go2-floating. Only stale docstrings
  corrected.
- ✅ **B2 / C6b — `fdsva_so` floating-base `daba_dqdq` block** FIXED 2026-06-01. The scratch/repair lead was a
  RED HERRING; the real bug was a **jk-transpose in `gen_fdsva_so_contract`'s daba_dqdq assembly** — the final
  `-Minv` reduction reads `inner_dq` jk-transposed, harmless for the symmetric `dM_dq*da_dq` term but DROPPING the
  jk-asymmetry of `d2tau_dqdq` on the 6-DoF floating root q-q columns (1-DoF fixed joints are jk-symmetric, so
  fixed-base was always correct). Fix: floating branch only adds `d2tau_dqdq` jk-transposed; fixed-base
  BYTE-IDENTICAL (iiwa14+fr3 verified). go2-floating (non-mimic control) now passes the strict floating SO
  diagnostic (norm_rel ~1e-6, was 4.4e-2); double-precision emulation matches pin to ~3e-16. Gate cleared
  (`MIMIC_FLOATING_UNSUPPORTED_GRADIENTS` now empty). NOTE: fr3-floating mimic fdsva residual ~6e-3 is float32
  amplification through the ill-conditioned reduced Minv (NOT this bug) → needs a per-robot floating conditioning
  bucket in `_fdsva_so_tolerance` (deferred, tolerance-policy owner); fr3 diagnostic also blocked upstream by a
  separate pre-existing `idsva_so_body_frame` mimic-column failure. See C6b in `api_completeness_audit.md`.
- ⬜ **B3 — floating-mimic `integrator_gradient`/`integrator_with_gradient` multi-stage RK bug** (stage
  projection at floating∩multistage∩mimic). numpy ref is correct; CUDA refused. The only remaining true
  mimic refusal.
- ✅ **B5 — grid_rbd FLOATING `end_effector_pose_gradient`** RESOLVED. Root cause: the in-tree precompiled
  `python/grid_rbd/_core*.so` was STALE (predated commit `d8b1bcb`, the d/dv-tangent convention ripple). The
  codegen kernel, `wrapper_template.cu`, `src/_core.cpp`, and `_handle.py` had ALL already been updated to the
  NV-column (`6*NUM_EES*NUM_VEL`) tangent convention in source, but the loaded `_core.so` still used the old
  NUM_POS-column ABI → it allocated/read a 19-col (NUM_POS) output for go2 while the freshly-built per-robot `.so`
  filled 18 NV cols, surfacing as BOTH the `(B,NEE,NV,6)` reshape `ValueError` (1824 vs 432) AND the
  garbage/uninitialized tail (~1e31..1e35, launch-dependent reads past the filled region). FIX: rebuilt `_core`
  from `python/src/_core.cpp` in place — no source edit needed; the convention is settled (public surface returns
  the d/dv TANGENT spatial Jacobian, NV columns, pinocchio convention, base block (omega; v)). go2-floating now
  matches the RBDReference/pin oracle to **2.0e-7** norm-rel (iiwa14-fixed 1.3e-7) across threads {1,2,16,32,64,
  128,256}; raw output finite + thread-invariant. V6 floating cells un-skipped (40/40 pass, 0 skipped).
  *(Note: `_core*.so` is GITIGNORED (`.gitignore` `*.so`), not checked-in — this was a STALE LOCAL build (May 23,
  predating `d8b1bcb`), not a repo artifact. `pip install`/`build_ext --inplace` rebuild it from source, so wheel
  users were never affected. Residual dev hazard: a stale local `_core.so` mis-reports ABI ripples → consider a
  test-time freshness guard or a documented rebuild step in CONTRIBUTING. Low priority.)*
- ✅ **B4 — `idsva_so_body_frame` fr3 mimic-column bug** FIXED 2026-06-02. The `last_two_axis_transpose=1.16`
  diagnostic was a red herring (just the natural j,k asymmetry of dvdq; the real failure was O(magnitude) on
  slot 13 = the mimic body's SHARED reduced velocity slot). TWO distinct NB-vs-NV mimic bugs: (1) the **floating
  body-frame inner** (`gen_idsva_so_body_frame_floating_reference_inner`) keyed its whole velocity-indexed sweep
  AND output on the REDUCED slot (`body_v_index`/`subtree_v_index` carrying the duplicate `...,13,13`) with plain
  `=` writes, so a mimic joint's contribution CLOBBERED its target's at the shared slot instead of alpha-folding.
  Ported it to the (already-correct) world-frame scheme: UNIQUE per-column INTERNAL slots (n_int) → 4*n_int^3
  internal slab → alpha-fold to the reduced 4*NV^3 output. The **gravity shim**
  (`gen_floating_gravity_d2tau_dq_lie_inline` + `_floating_gravity_lie_metadata`) had the same reduced-slot
  overwrite, so it too now runs in internal slots and ADDS into the internal slab BEFORE the single fold (matching
  the oracle's internal-coord gravity Hessian + R-fold). (2) the **fixed-base mimic** repair-path Pass-1 zero
  (`gen_idsva_so_body_frame_reference_order_output_repair`) zeroed only `SECOND_ORDER_TENSOR_SIZE` (=4*NV^3) of the
  NB-strided internal slab (4*NB^3), leaving the dM_dq block + dvdq tail stale (latent: harmless on the current
  fixed-base sample because the optimized-assembly write-set was re-covered by the repair, but a genuine
  undersize) → now zeroes `4*NB^3` for mimic. All gated mimic-only; non-mimic emission BYTE-IDENTICAL (fixed-base
  g1+iiwa14 diff empty; floating iiwa14+go2 SO diagnostics still pass). fr3-floating body-frame SO now asserts
  green vs pin_so_ext (all 4 blocks ~1e-7, was O(0.2) on slot 13). B4 skip guard removed; fr3 added to the
  fixed+floating SO test sets. Fix in `GRiDCodeGenerator/algorithms/_idsva_so.py`.

## 2. CORRECTNESS — verification gaps (untested code that could hide bugs)
- ✅ **C2 — `fd_parameter_gradient` numpy** now tested vs −M⁻¹Y pin oracle (ref was correct). *(committed)*
- ✅ **idsva_so floating-mimic** verified correct vs pin (world-frame), gate enabled. *(C6, committing)*
- ✅ **V1 / C3 — `com_cost`/`momentum_cost` CUDA equivalence test added** (iiwa14-fixed + go2-floating);
  EXPOSED + FIXED a real emitter bug: `com_cost_gradient` zeroed `[nq, nq+nv)` but the meaningful gradient
  lives in `[0, nv)`, leaving `[nv, nq)` UNINITIALIZED on floating-base (nq>nv) → stale shared mem (go2
  `s_grad[18]`). Fixed in `_plant.py` to zero the full `[nv, nx)` tail (byte-identical fixed-base; matches the
  GN-hessian `[0,nv)` convention). `momentum_cost_gradient` was already correct. Both cells green. *(committed)*
- ✅ **V2 / C4 + D — mimic-safe centroidal CUDA runner added** (`cuda_centroidal_mimic_smoke_runner.cu` +
  test, codegens `algorithm_list=["id"]` so only generalized_gravity/nonlinear_effects emit; non-mimic runner
  untouched). fr3-fixed (mimic, NB=9>NV=8) + iiwa14-fixed control PASS ~5e-7 — runtime confirms the device
  `s_vaf=18*NB` fix. EXPOSED + FIXED a second mimic-overflow bug: the HOST arena macro `id_bias_t_count`
  (`GRiDCodeGenerator.py:662`) still sized `s_vaf` by `18*n` (NV) while the device wrapper uses `18*NB` →
  h1_2-fixed (NB=51>NV=39) crashed with an illegal shared write (under-budget by ~99 floats). Fixed to
  `18*nb_vaf` (NB for mimic; byte-identical non-mimic). Pending h1_2 validation → then commit. *(committing)*
- 🔵 **V3 / C5 — integrator CUDA breadth to g1/h1_2 + PERF/LITE tier sweep** (committed `6aaad70`). Test infra
  done: 5 robots × 2 base × 2 tiers = 20 cells; floating-mimic runs value-only (B3-refused gradient); mimic
  `s_vaf=18*NB` sizing confirmed. VALIDATED subset green (g1-fixed SHARED+LITE, h1_2-fixed-LITE mimic spill —
  the hardest cell). REMAINING (reasoned, not run, due to ~8-way contention): h1_2-fixed-PERF, g1-floating×2,
  h1_2-floating value-only → **must get a full-matrix green run in the pre-sweep gate** (see I1 area).
- ⬜ **V4 / C7 — `integrator_with_gradient` no standalone numpy test**; **FK-batched** has no equivalence
  diff vs `end_effector_pose` (binding coverage ≠ correctness).
- ⬜ **V5 — widen d2ee + SO tests to big/floating robots** once B1/B2 land.
- ⬜ **V6 — kinematics thread-count-invariance matrix** (user-requested, before the sweep): the single-block
  kinematics kernels (`end_effector_pose`, `ee_pose_gradient`, `frame_jacobian`) must give identical results
  at ANY thread count — test single-thread (1) and warp-sized (32) explicitly, plus a sweep of thread counts
  {1,2,16,32,64,128,256} × batch sizes {1,16,256}, asserting equivalence vs the numpy/pin reference at every
  cell. Catches reduction/sync bugs that only surface at extreme thread counts. Any divergence = a new
  backlog bug to fix BEFORE the sweep.

## 3. COMPLETENESS — missing surfaces
- ✅ **S1 — `frame_jacobian` / `frame_jacobian_dot` / `osc_inertia` were DEVICE-ONLY → now full surface.**
  Each gained `*_kernel` (+`_single_timing`) + 3-mode host (`<algo>`/`_single_timing`/`_compute_only`) +
  `gridData` output buffers (`d_/h_frame_jacobian` 6×NV, `d_/h_frame_jacobian_dot` 6×NV, `d_/h_osc_inertia`
  6×6) → benchmarkable + bindable. All three registered in `KERNEL_ATTR_MANIFEST` + bench `PER_ALGO_SPECS`
  (gates `GRID_HAS_FRAME_JACOBIAN[_DOT]` / `GRID_HAS_OSC_INERTIA`; the bench codegen now requests the opt-in
  keys via `algorithm_list=["all",...]`). The launchable surface bakes a fixed target (leaf-EE joint) +
  `LOCAL_WORLD_ALIGNED` frame; the `_device` functions stay the arbitrary-target/-frame entry points.
  `frame_jacobian` calls its `_inner` directly (kernel owns the dynamic arena, I/O in the arena);
  `frame_jacobian_dot` + `osc_inertia` call their auto-smem `_device` wrappers (which own the dynamic arena),
  so those kernels keep I/O in static `__shared__`; `osc_inertia_kernel` carries `__launch_bounds__` to
  register-fit the heavy minv/J/invert path. Host surfaces validated vs the numpy oracle in
  `test/cuda_equivalents/test_cuda_frame_jacobian_host.py` (+ `cuda_frame_jacobian_host_runner.cu`):
  iiwa14-fixed J rel err ~1.8e-7, go2-floating ~7.5e-8; FJD (5e-2 FD tol) + Λ (5e-3, non-singular) also
  checked. Byte-identical for non-frame_jac profiles (only the additive gridData buffers differ). Also fixed
  a latent `_plant.py` bug: `com_cost`/`momentum_cost` were emitted whenever `end_effector_pose` was present
  (referencing undefined `grid::com_device`/`ccrba_device`) — now gated on `com`+`ccrba` keys. Unblocks P2
  (parallelize `frame_jacobian_inner`) + F2 bindings.
- ⬜ **S2 — uniform `_inner` missing** for `fd_du` (reuses id_du band), `f_ext_gradient_dq` (kernel-only),
  `integrator_gradient` (uses `_multistage`).
- ✅ **S3 — CUDA-level USAGE EXAMPLES (the "write your own kernel against grid.cuh" gap).** Added
  `examples/cuda/`: the flagship `inverse_dynamics_kernel_example.cu` (one-line codegen invocation →
  `#include "grid.cuh"` → single-block kernels hitting BOTH the auto-scratch `inverse_dynamics_device` and
  the caller-scratch `inverse_dynamics_inner` + `load_update_XImats_helpers`, with the
  `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` + `cudaPeekAtLastError` discipline, then a batched
  one-block-per-timestep grid-stride launch); a second-order `idsva_so_host_example.cu` showing the heavy
  `_host` surface; `README.md` (the step-by-step walkthrough) + `wrapper_types.md` (the
  `_inner`→`_device`→`_kernel`→`_host`→batch layering + when to use each). nvcc-compiles on sm_120 and is
  validated end-to-end (`build_and_validate.sh` + `validate.py`/`validate_so.py`): ID worst rel err 2.7e-7,
  idsva_so worst 2.7e-6 (float32) vs `RBDReference`. The emitted smem macros now match the verbose function
  names: `INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES` / `INVERSE_DYNAMICS_DEVICE_DYNAMIC_SHARED_MEM_BYTES`
  (macro-rename A5b done — see below). Generated headers stay build artifacts (gitignored).

## 4. COMPLETENESS — missing features
- ⬜ **F1 — `plant_step_hessian`** (true analytic 2nd-order plant/integrator Hessian) is absent; needs an
  analytic 2nd-order integrator (`_plant.py:82-87 TODO`). **#1 customer ask** (PDDP `compute_fxx`, GATO).
- 🟡 **F2 — bindings (8 of 11 bound; 3 blocked on codegen).** Bound `frame_jacobian`, `frame_jacobian_dot`,
  `osc_inertia`, `com`, `ccrba`, `energy`, `generalized_gravity`, `nonlinear_effects` through the full C-ABI
  + `_core` Runner + numpy `RobotHandle` stack (`python/grid_rbd/wrapper_template.cu`, `python/src/_core.cpp`,
  `python/grid_rbd/_handle.py`). `_compile.py` now requests the opt-in `frame_jacobian` family in
  `gen_all_code` (the codegen emits `#define GRID_HAS_FRAME_JACOBIAN`, which gates the wrapper's
  frame_jacobian C-ABI symbols; com/ccrba/energy/gg/nle ride the default `all` profile). Validated vs
  `RBDReference` on iiwa14 + go2 (fixed) in `test/python_wrappers/test_centroidal_energy_frame.py`. NOT bound:
  `com_cost`, `momentum_cost`, `plant_step_gradient` — the codegen (`_plant.py gen_plant_kernels`) emits only
  their per-timestep `__device__` fns, NOT the launchable `grid_plant::*_kernel<<<>>>` wrappers the C-ABI
  binding requires (and no `GRID_PLANT_HAS_*` define). Binding these needs codegen work: add
  `gen_com_cost_kernel`/`gen_momentum_cost_kernel`/`gen_plant_step_gradient_kernel` (+ HAS defines) mirroring
  `gen_ee_pos_cost_kernel`/`gen_plant_step_kernel`. Also fixed a pre-existing build-blocker: the JAX/torch
  `fdsva_so` FFI handlers in `wrapper_template.cu` passed `fdsva_so_kernel` args in the wrong order
  (`d_df2,d_q_qd_u,stride,d_workspace,…` vs the header's `d_df2,d_workspace,d_q_qd_u,stride,d_idsva_so,…`),
  which broke every jax+torch `.so` compile; corrected to match the header. JAX/torch backends not extended
  for the 8 new methods (numpy-handle only, mirroring the plant surface).

## 5. READABILITY — clean-break rename + uniformity (scheme LOCKED: rename_mapping.md)
- ⬜ **R1 — verbose rename** (id→inverse_dynamics, etc.; keep aba/crba/ccrba/minv/idsva_so/fdsva_so;
  rnea→docstring not alias). One name as key==symbol==bench==token==label.
- ✅ **R2 — outputs into `gridData`** for `inverse_dynamics_regressor` (`d_Y`/`h_Y`) + `forward_dynamics_parameter_gradient` (`d_dqdd_dpi`/`h_dqdd_dpi`) done: added to gridData struct + alloc/free, hosts write `hd_data->d_*` and copy back to `hd_data->h_*` (uniform `(hd_data, model, ...)` sig; dropped explicit param). Bench drops TU-static malloc; runners read from gridData. Byte-identity confined to struct/init/free + the two hosts; regressor + fd_param equivalence green (g1-floating fd `@ zero` is the pre-existing float32 conditioning floor, fails identically pre-R2).
- ✅ **R3 — reorder `fdsva_so_kernel` args (`d_workspace`→2nd)** done (kernel sig + host launchers + func-ptr attr cast; byte-identical otherwise; iiwa14-fixed/go2-floating equivalence green).
- ⬜ **R4 — drop** `_host`/`_with_x_kp1`/`direct_` + `rnea`/`eepos`/`deepos` labels + `SUGGESTED_THREADS` alias.
- ⬜ **R5 — unify gravity-sign convention** (GRiD `+9.81` vs RBDReference `-9.81`).
- ✅ **R6 — emit centroidal family on OWN keys** done: com/ccrba/energy/generalized_gravity/nonlinear_effects are now first-class `algorithm_list` keys (each auto-expands its real dep: com/ccrba/energy→ee_pose, gg/nle→id); requesting a sibling key no longer silently emits them. `all` byte-identical (banner list only). iiwa14/go2 + mimic-safe centroidal equivalence green.

## 6. PERFORMANCE — max-perf completion (single-block ONLY)
- ⬜ **P1 — big-robot `crba`/`aba`/`minv`/`fd` in-block parallelism.** They lose to Pinocchio CPU on h1_2;
  `id` wins 5× there → the in-block-parallelism pattern to follow. The only place GRiD loses.
- ⬜ **P2 — parallelize `frame_jacobian_inner`** (HJCD; after S1 wrappers exist to measure it).

## 7. INFRA (mostly done)
- ✅ Sweep efficiency: collapse + narrow + aliasing-fix + SASS tier-dedup (~6× faster, correct per-tier).
- ⬜ **PRE-SWEEP VALIDATION GATE** (run once the backlog is otherwise green, in a GPU-idle/overnight window,
  BEFORE I1): a single clean full-matrix equivalence run with cleared header cache covering every robot × base
  × representative tiers — INCLUDING the big-robot integrator cells C5 only reasoned (h1_2-fixed-PERF,
  g1-floating×2, h1_2-floating value-only) and V6's kinematics thread×batch matrix. No known-open bug at sweep
  time (user directive). Any red cell = a bug to fix before the sweep, not after.
- ⬜ **I1 — re-sweep on the fixed harness** to set real autotune defaults + true best-tier numbers (later).

## AUDIT FINDINGS (submodule features / docs / consistency — 2026-06-02, ongoing)
- ⬜ **A1 — `f_ext_gradient_dq` has NO RBDReference oracle.** GRiD emits the kernel (∂(id_du)/∂f_ext,
  fixed-base) but RBDReference lacks a `f_ext_gradient_dq` method → the kernel is unverifiable vs a numpy
  reference (only `f_ext_gradient` is checked). Add the reference method + an equivalence test.
- ⬜ **A2 — planar/spherical are PHANTOM support (interface inconsistency).** URDFParser PARSES them
  (Joint.py px_pl/py_pl/theta_pl) and `errors.py` advertises them as "supported", but the CODEGEN transform
  chain (`_topology_helpers.py`) does NOT handle them, RBDReference does NOT model them, and NO fixture uses
  them. So the parser advertises joint types the rest of the stack can't codegen/validate. FIX (pick one):
  (a) DEMOTE — make URDFParser raise a clear "planar/spherical not yet codegen-supported" error + drop them
  from the advertised-supported list + doc note (honest, small); or (b) IMPLEMENT full support (parse +
  codegen transforms + RBDReference reference + planar/spherical test URDFs) — a real feature. Lean (a) now.
- ✅ **A3 — user-facing docs were STALE post-rename** (old names across `docs/source/**` + submodule
  `RBDReference/README.md` + a `rnea.rst` page; gravity docs said +9.81). DONE: verbose names + signed
  gravity applied across Sphinx + READMEs; `rnea.rst` → `inverse_dynamics.rst` (toctree fixed); sphinx
  builds clean.
- ✅ **A4 — stale completed planning docs** → archive to `docs/open-tasks/archive/`. DONE for the one
  confirmed-historical doc (`python_wrappers_plan.md`, "v0.3 SHIPPED, historical"). KEPT LIVE (not
  archived): `d2_codegen_mimic_plan.md` ("scoping doc, NOT yet implemented" + T3-tracked),
  `a3_…audit` (referenced by `_crba.py` code as the refactor plan), `idsva_so_inner_refactor_notes`
  (cited as source-of-truth by live Sphinx concept pages + CUDA-equivalence tests; names fixed in place).
- 🟡 **A5 — internal identifiers the rename MISSED** (public API + docs are consistent; these are internal):
  - ✅ **A5b-macro (DONE)** — the user-facing emitted macros now match the verbose function names:
    `ID_*`→`INVERSE_DYNAMICS_*`, `FD_*`→`FORWARD_DYNAMICS_*`, `ID_DU_*`→`INVERSE_DYNAMICS_GRADIENT_*`,
    `FD_DU_*`→`FORWARD_DYNAMICS_GRADIENT_*`, `EE_POS_*`→`END_EFFECTOR_POSE_*`,
    `DEE_POS_*`→`END_EFFECTOR_POSE_GRADIENT_*`, `D2EE_*`/`D2EE_POS_*`→`END_EFFECTOR_POSE_HESSIAN_*`,
    `ID_BIAS_*`→`INVERSE_DYNAMICS_BIAS_*`, `FD_PARAMETER_GRADIENT_*`→`FORWARD_DYNAMICS_PARAMETER_GRADIENT_*`
    (across the `*_DYNAMIC_SHARED_MEM_{BYTES,COUNT}`, `*_DEVICE_DYNAMIC_SHARED_MEM_BYTES`, `*_DEVICE_INLINE*`
    families). Kept proper names: `ABA_`/`CRBA_`/`CCRBA_`/`MINV_`/`IDSVA_SO_*`/`FDSVA_SO_`/`COM_`/`ENERGY_`/etc.
    Emitters + every consumer (tests, benchmarks, `examples/cuda/*`, `wrapper_template.cu`, docs) updated; regen
    byte-identical except the renamed identifiers; examples + regressor/fd_param/centroidal equivalence +
    python_wrappers smoke all green.
  - ⬜ **A5b-residual (DEFERRED)** — internal-only, tangled naming left as-is: (a) `GRiDCodeGenerator/_test.py`
    still has `rnea`/`rnea_grad`/`fd_grad` method names (dev script); (b) the spill-decision flags
    `GRID_ID_DU_*`/`GRID_FD_DU_*`/`GRID_D2EE_*`/`GRID_EE_GRAD_*` (`_USES_*`, `_SHARED_TIER*`,
    `_WORKSPACE_*_OFFSET_BYTES`) + the `*_INNER_{SMEM,WORKSPACE}_BYTES`/`*_F_IN_SMEM`/`*_TEMP_IN_SMEM` tier
    bools + `*_spill_tier_3way` attrs still use old short segments. NOT done because each short segment spans a
    LARGER macro family beyond the `_USES_` flags (renaming only `_USES_` would split `GRID_ID_DU_*`/`GRID_D2EE_*`
    internally), and the `test_cuda_codegen_layout.py` / `test_cuda_executable_equivalence.py` assertions track
    them — verbosify the whole internal family in one structural pass.
  `perf_cleanup_overnight.md` does not exist under `docs/` (only a stale mention in a bench-result file).
- ⬜ **A6 — RBDReference test suite is PATHOLOGICALLY SLOW (infra).** The 1026-test suite runs >2h even at
  `-n 12` (serial was killed at 2h); a long tail of a few big-robot pinocchio comparisons (likely h1_2/g1
  SO + param-grad numpy refs) dominates. This blocks full-suite validation after any RBDReference change.
  Profile (`--durations=20`), mark the slowest as `slow`/`developer_only` so the default suite is fast, and
  cache/vectorize the worst numpy refs. Until then, validate RBDReference changes with a representative subset.
- (Non-gaps confirmed: `integrator_with_gradient` covered via the integrator+gradient pairing (C7);
  `plant_step_hessian` absence == F1, already tracked.)

## Proposed sequence (for discussion)
1. **Correctness bugs first** — B1, B2, B3. (Don't ship known-wrong results.)
2. **Verification** — V1–V5 lock the rest with tests (cheap, high-confidence).
3. **Clean-break rename** — R1–R6 as ONE coordinated pass over the now-correct code (so new work lands on final names).
4. **Surface + feature completion** — S1/S2, then F1/F2.
5. **Perf** — P1, then P2.
6. **I1 re-sweep** whenever a GPU-idle window opens (sets defaults; non-blocking).
