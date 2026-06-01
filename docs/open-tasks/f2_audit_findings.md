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
- `grid::SUGGESTED_THREADS` → perf-cap rename: `printGRiD.cu` fails to compile on the stale symbol
  (flagged by SO-idsva; pre-existing, absent from baseline). Fix the rename end-to-end.
- nvcc/ptxas warning sweep (`-Werror`-style) across generated code.
- Naming uniformity residuals (`s_*`-named-but-device pointers; `*_DYNAMIC_SHARED_MEM_BYTES`
  macro shape). The `mxS` NumPy `ndim>0` deprecation is ALREADY fixed (stale item — dropped).

## D. Backlog-accuracy note
This session found 5 stale "open" backlog entries that were already done (C1-float, C3, mxS
warning, mimic regressor-Y, FD param-grad). The HANDOFF REMAINING list had drifted; corrected
inline. A periodic "is this actually still open?" pass is cheap insurance against wasted agent
effort.
