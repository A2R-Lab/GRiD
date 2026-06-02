# F2 audit findings (2026-06-01, during the C.7 sweep)

Read-only greps surfaced during the post-mimic-campaign session. These are the concrete
worklist for the F2 cleanup pass (each needs codegen/compile work → do AFTER the perf sweep
finishes, so it doesn't skew timing). Two systemic bug classes + the standing naming items.

## A. Silent CUDA launch-failure pattern (smoke runners missing error checks)
Origin: the mimic `osc_inertia` Λ "zero output" was a SILENT launch failure — the heavy
`osc_kernel` (no `__launch_bounds__`, ~100+ regs) failed to launch at 512 threads
("too many resources requested"), and the runner never checked `cudaGetLastError`, so the
zeroed output masqueraded as a result. Fix that landed: clamp the osc launch to
`cudaFuncGetAttributes().maxThreadsPerBlock` + hard-fail on a bad launch.

**Remaining unguarded runners (launches with 0 `cudaGetLastError`/sync checks):**
- `test/cuda_equivalents/cuda_centroidal_smoke_runner.cu` — 4 launches, 0 checks.
- `test/cuda_equivalents/cuda_plant_smoke_runner.cu` — 2 launches, 0 checks.
- `test/cuda_equivalents/cuda_frame_jacobian_smoke_runner.cu` — 3 launches, 1 check (osc got the
  check; J/J̇ launches may still be unchecked — verify).
- (`cuda_f_ext_gradient_runner.cu`, `cuda_integrator_smoke_runner.cu` launch via host wrappers;
  grep didn't count `<<<` — audit them directly.)
- Well-guarded reference: `cuda_equivalence_runner.cu` (104 checks) — mirror its pattern.

FIX: after each kernel launch (or once after the launch batch), `cudaGetLastError()` +
`cudaDeviceSynchronize()`, and FAIL LOUDLY (nonzero exit / assert) rather than compare zeros.
Also worth: a shared `GRID_CHECK_LAUNCH(...)` helper so every runner uses one idiom.
(The perf timing harness `timeGRiD_common.h` already got a log-only `[GRID_LAUNCH_ERROR]`
guard — see commit b9e592e.)

## B. Per-body scratch sized by NV/num_vel instead of NB/num_bodies (mimic overflow class)
Origin: THREE instances fixed this session — h1_2 RNEA `s_vaf` (A1), B2-SO body-inner scratch,
and the integrator `s_vaf` (sized `18*NV`, overflowed `s_Minv` for NB>NV mimic). The rule:
any per-BODY scratch consumed by an inner that writes body-indexed (stride over NB) must be
sized by `get_num_joints()`/NB, not `get_num_pos()`/`get_num_vel()`/NV — they differ for mimic
robots (mimic joints carry 0 DoF, so NB > NV).

**Latent suspect found:**
- `GRiDCodeGenerator/algorithms/_centroidal.py:64` and `:88` — `extra = [("s_vaf", 18 * n)]`
  (n = num_vel). Centroidal reuses `inverse_dynamics_inner(compute_c, qdd=0)`, which writes
  `s_vaf` body-indexed over NB. For a mimic robot this UNDERSIZES `s_vaf` → overflow, exactly
  like the integrator bug. Centroidal is not currently exercised on mimic robots (coverage-fill
  deferred fr3 centroidal), so it's latent — but FIX it (size by NB when mimic) AND add an fr3
  centroidal equivalence cell to catch it.
- Broader sweep: grep every `algorithms/*.py` for per-body scratch (`s_vaf`, `s_XImats`-adjacent
  temps, `18*n`, `6*n` body buffers) and verify NB-vs-NV. crba uses both `36*NB` and `36*n` in
  different functions — crba is GREEN on fr3 (validated), so likely fine, but worth a conscious
  pass to confirm none are NV-where-NB-needed.

## C. Standing naming/warnings items (the original F2 scope)
- `grid::SUGGESTED_THREADS` → perf-cap rename: **RESOLVED (F2, 2026-06-01).** Status when picked up:
  the codegen had ALREADY renamed the emitted symbol to `MAX_PERF_LEVEL_THREADS`
  (`GRiDCodeGenerator.py:967`); `SUGGESTED_THREADS` no longer appeared in generated `grid.cuh` at
  all. The only in-repo straggler was `printGRiD.cu:19` (`grid::SUGGESTED_THREADS`), which the audit
  correctly flagged as failing to compile against a fresh header.

  **External blast radius (the reason a blind rename was the wrong move):** customers hold older
  generated headers and reference the OLD symbol — GATO `merit.cuh:279` and PDDP
  `cg_v4_iiwaplant.cuh:37` both use `grid::SUGGESTED_THREADS`. Because the rename had already
  happened in codegen, a customer regen would make the symbol *vanish* and break them.

  **Path taken (deprecation-alias, customer-safe):**
  1. `printGRiD.cu:19` updated to `grid::MAX_PERF_LEVEL_THREADS` (our own example file — fine to move).
  2. Re-added a backward-compat alias to the emitted header so the old name keeps resolving:
     `const int SUGGESTED_THREADS = MAX_PERF_LEVEL_THREADS;` (emitted right after the
     `MAX_PERF_LEVEL_THREADS` def in `GRiDCodeGenerator.py`). This UN-breaks GATO/PDDP on their next
     regen instead of leaving them broken.

  **Migration note for customers (old → new):**
  - `grid::SUGGESTED_THREADS`  →  `grid::MAX_PERF_LEVEL_THREADS` (same value; the PERF-tier
    launch-bounds cap / autotune ceiling).
  - The alias is a deprecation shim, scheduled for removal in a future major version. New customer
    code (GATO merit kernel, PDDP backpass/forwardpass `*_THREADS`) should adopt
    `MAX_PERF_LEVEL_THREADS` (or an explicit autotuned thread count) and add a
    `gpuErrchk(cudaPeekAtLastError())` after the launch + clamp to
    `cudaFuncGetAttributes().maxThreadsPerBlock` (see §A / `agent_debugging_guide §1c`).
- nvcc/ptxas warning sweep (`-Werror`-style) across generated code. *(Still open — not in F2 scope.)*
- Naming uniformity residuals (`s_*`-named-but-device pointers; `*_DYNAMIC_SHARED_MEM_BYTES`
  macro shape). The `mxS` NumPy `ndim>0` deprecation is ALREADY fixed (stale item — dropped).
  *(Still open — not in F2 scope.)*

## D. Backlog-accuracy note
This session found 5 stale "open" backlog entries that were already done (C1-float, C3, mxS
warning, mimic regressor-Y, FD param-grad). The HANDOFF REMAINING list had drifted; corrected
inline. A periodic "is this actually still open?" pass is cheap insurance against wasted agent
effort.

## (D) Mimic centroidal CUDA validation gap (opened 2026-06-01)
The `s_vaf` NB-sizing fix in `_centroidal.py` (item B) is validated by codegen sizing only
(h1_2 `s_vaf[918]`=18·51, fr3 `s_vaf[162]`=18·9) + it matches `inverse_dynamics_inner`'s
documented `18*NUM_JOINTS` contract. It is NOT runtime-validated on a mimic robot, because the
existing `cuda_centroidal_smoke_runner.cu` drives `grid_plant::com_cost`/`momentum_cost` (+ com/
ccrba device fns) which codegen emits for NON-MIMIC robots only (`_plant.py:907`). A naive fr3
cell therefore fails to compile (`grid_plant has no member momentum_cost_gradient`).
**TODO:** add a dedicated mimic-safe runner that exercises ONLY `generalized_gravity` +
`nonlinear_effects` (the RNEA-bias path that carries `s_vaf`, and which DOES emit for mimic) on
fr3/h1_2, diffed against the RBDReference oracle. That closes the mimic centroidal-dynamics
coverage hole and gives the B fix a runtime check. (Keep the existing runner non-mimic.)
