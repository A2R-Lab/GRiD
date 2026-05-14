# Session status handoff — GRiD-A2R + GLASS v2

Living handoff doc for the GRiD-A2R / GLASS v2 trajectory. Replaces the
dated `handoff-session-status-2026-05-14.md` and folds in the resolved
content from `handoff-ptxas-hang.md` and the original
`glass-rfc-batched-1d.md`. Active GLASS asks live in
`handoff-glass-rfc-gemv-auto-dispatch.md`.

## Where we are (as of 2026-05-14 + integration)

PR1 (GLASS v2 integration) and PR-multi-version-bench shipped before this
session. Two upstream PRs then landed via commits `b104098 sm_8x fixes`,
`8f9b607 merge`, `7f38efc updating GLASS`:

1. **GLASS PR (1D-batched + auto-fallback).** Delivered the
   `gemm_batched_1d` and `gemm_strided_batched_1d` APIs from the original
   RFC, plus compile-time `should_use_cublasdx<>` auto-dispatch in
   `glass::nvidia::gemm<>` via `tuning_table.cuh`. P0-1 through P2-7
   landed (P2-6 deferred).
2. **sm_8x agent fixes.** Found two real bugs the prior ptxas-hang
   handoff had misdiagnosed (details in "Resolved-upstream notes" below).

This session's work integrated the GLASS PR into GRiD's codegen, did
Phase 5a cleanup of the runtime-dispatch wrapper, and validated end-to-end.

## What's committed

- All PR1 work (GLASS v2 backend + multi-version harness skeleton).
- Multi-version 6-column harness, robustness flags, Pinocchio threading
  hardening (`effective_thread_count` ≥16 work per thread heuristic).
- MJX collision-pair strip (go2/g1 geometry now loads).
- Phase 7d: ccache split into compile/link stages (~14× iteration speedup).
- Phase 7a: `custom_is_constant` memoization in `_topology_helpers.py`
  (g1_floating codegen 22min → 6.4min).
- sm_8x: `--cicc-opt-level` flag + `KERNEL_ATTR_MANIFEST` in
  GRiDCodeGenerator.

## What's in this session (uncommitted at time of writing)

### Step 1 — reinstated cuBLASDx in ee_pose_gradient

[`GRiDCodeGenerator/algorithms/_eepose_gradient_hessian.py`](../GRiDCodeGenerator/algorithms/_eepose_gradient_hessian.py)
now emits `glass::nvidia::gemm_strided_batched_1d<T,4,4,4,BATCH,TC>`
calls for the eepose position chain on glass-nvidia builds. The
gradient chain stays SIMT (djid-dependent A pointer can't share).
BATCH = n*run_len; TC = max(1, SUGGESTED_THREADS // BATCH).

Vendored 3 new GLASS headers into the codegen output:
[`GRiDCodeGenerator/helpers/_lin_alg_helpers.py`](../GRiDCodeGenerator/helpers/_lin_alg_helpers.py)
`_GLASS_NVIDIA_GLOBAL_SCOPE_FILES` += `tuning_table.cuh`;
`_GLASS_NVIDIA_FILES` += `query_simt.cuh`, `l3_simt.cuh`.

### Step 2 — Phase 5a cleanup + DEFINE filter

- Deleted `linalg_smem_for()` helper and `GRID_BENCH_NVIDIA_MIN_DIM`
  env-var (no longer needed; GLASS auto-dispatches via
  `should_use_cublasdx<>`).
- Simplified `grid_linalg_gemm` wrapper: compile-time `if constexpr`
  dispatch only, no runtime nullptr branch.
- Simplified `grid_linalg_row_strided_gemv` to always-SIMT (the only
  consumer is 6×6 which SIMT wins; GLASS doesn't have GEMV auto-dispatch
  yet — see `handoff-glass-rfc-gemv-auto-dispatch.md`).
- Filtered `_nvidia_gemm_sizes()` / `_nvidia_gemv_sizes()` by the
  cuBLASDx-wins heuristic (`max(M,N,K) >= 16 AND min(M,N,K) >= 4`) so
  we DON'T emit `DEFINE_NVIDIA_GEMM_BLOCKDIM_SM` for small shapes.
  Pre-fix, the unfiltered DEFINEs created explicit specializations
  that bypassed auto-dispatch and forced cuBLASDx even when SIMT wins —
  caused `cudaErrorIllegalAddress` on iiwa14 fixed glass-nvidia.
- Removed dead `_ee_gradient_packed_gemm_k_values()` (collected K
  dims for an older packed-K codegen path now replaced by
  `gemm_strided_batched_1d`).

### Step 3 — validation (in-flight at time of writing)

Full multi-version sweep on sm_120:
`comparison_sm120_post_glass_rfc/`. Pass conditions: no
`cudaErrorIllegalAddress`, numerical equivalence vs the SIMT-only
fallback in eepose, and the go2 ee_pose_grad N=256 regression I had
flagged (149µs Step 1 vs the SIMT-only stub) recovers — confirmed
locally to 79µs on the post-filter compile (vs pre-GLASS-PR May-13
baseline of 56µs; cuBLASDx packed-K had crash-prone OOBs that
masqueraded as fast on go2/g1).

## Resolved-upstream notes (preserved historical findings)

### From the old ptxas-hang handoff (now-resolved bugs the sm_8x agent root-caused)

Original claim: "ptxas hangs at 100% CPU on floating-base GRiD compiles
on sm_8x; `--no-licm-barrier` is the workaround." **The sm_8x agent
found that claim was wrong on two counts:**

- **It's `cicc` (NVVM optimizer) that hangs, not `ptxas`.** Verified
  via `pgrep -ax cicc` showing 100% CPU while `pgrep -ax ptxas` was
  empty. The hang is in cicc's -O3 optimization pipeline on the
  floating-base codegen output.
- **`--no-licm-barrier` does NOT eliminate the hang** (verified by
  clean-cache reproduction). The trigger isn't `-rdc=true` and isn't
  the anti-LICM emission as previously believed.

**Real fix shipped:** `--cicc-opt-level` flag in [grid/run.py](../test/benchmarks/baselines/grid/run.py)
and [run_multi_version.py](../test/benchmarks/run_multi_version.py).
On sm_8x, pass `--cicc-opt-level 2` — leaves ptxas at -O3 so SASS
quality is preserved; cicc bisection showed -O0 (41s), -O1 (10s),
-O2 (122s) all finish for iiwa14 floating, -O3 hangs > 5 min.
On newer GPUs (sm_120+) the flag is optional and not forwarded by the
multi-version harness to fixed-base inner calls (avoids a fixed-base
30-70% regression where cicc -O2 produced slower SASS than -O3).

**Status:** root-cause investigation deferred — could be Ampere-only
nvcc bug, or could be specific to the bench's heavy template surface
(`timeGRiD.cu` instantiates every algorithm's `_single_timing` +
`_compute_only` + batch wrapper with multiple template combos +
TEST_ITERS=10000 rep loops). A minimal user-style `.cu` compiled
clean at cicc -O3 in 2m39s, so general GRiD users on sm_8x are
unaffected — only the bench surface trips it. See deferred follow-up
#3a in the plan file.

### From the old ptxas-hang handoff — cudaFuncSetAttribute fix

ABA/FD/MINV on g1_floating need >48 KB dynamic shared memory; without
`cudaFuncSetAttribute(MaxDynamicSharedMemorySize, BYTES)` the kernels
silently failed and the harness measured launch overhead (~0.5 µs N=256,
which previously looked like "free speedup" — it was the bug). Fix
shipped: class-level `KERNEL_ATTR_MANIFEST` in
[GRiDCodeGenerator.py](../GRiDCodeGenerator/GRiDCodeGenerator.py)
that emits `cudaFuncSetAttribute(...)` for every algorithm kernel.

### From the old GLASS RFC (delivered)

The original RFC requested 7 features. **6 of 7 landed** in GLASS
commit `0861724`:

- **P0-1 `gemm_batched_1d`** — 1D-launch batched GEMM (pointer arrays).
- **P0-2 `gemm_strided_batched_1d`** — 1D-launch batched GEMM (shared A
  + strided B/C). The API GRiD uses for ee_pose_gradient.
- **P1-3 `should_use_cublasdx<>` constexpr** — compile-time dispatch.
- **P1-4 small-GEMM SIMT fast path** — auto-fallback for shapes where
  cuBLASDx loses.
- **P1-5 GEMV audit** — doc-only (no API change; the GEMV auto-dispatch
  GAP is now the subject of `handoff-glass-rfc-gemv-auto-dispatch.md`).
- **P2-7 diagnostics** — `print_dispatch<>`.
- **P2-6 warp-resident no-sync GEMM** — analyzed and dropped (GLASS
  team's measurements showed SIMT wins).

## Active asks of GLASS (round 2)

See [`handoff-glass-rfc-gemv-auto-dispatch.md`](handoff-glass-rfc-gemv-auto-dispatch.md)
for full details. Summary:

- **Auto-dispatch gaps (P0):** mirror the `gemm<>` auto-dispatch for
  `gemv<>`, `row_strided_gemv<>`, `row_strided_gemm<>`, and
  `gemm<TRANSPOSE_B=true>`.
- **NEW features:** `gemm_batched_1d` cuBLASDx fallback,
  `required_smem_for_dispatch<>` constexpr,
  `print_dispatch_full<>` diagnostic.

## Productive without dependencies (PR3 — compile speed)

Most of Phase 7 still pending. See deferred follow-ups in the plan
file `let-s-make-a-plan-foamy-sunbeam.md`.

## Validation TODO: autotune regression test

The README now suggests `python GLASS/bench/autotune.py --sm AUTO`
as a recommended one-time install step. But we haven't actually
validated that the autotuned `tuning_table.cuh` produces *faster*
GRiD benchmarks than the shipped sm_120-only defaults on each
target GPU. Possible regression: if the autotuner's measurement
methodology differs from how GRiD actually invokes the kernels
(e.g., different launch geom, different surrounding workload), the
"win" recorded in the table might not translate to a real GRiD win.

Action item before promoting autotune to "required" in the
README: run the multi-version sweep twice on a non-sm_120 GPU
(e.g., 5090 or the sm_8x machine) — once with shipped defaults,
once with `bench/autotune.py` output — and compare:

- Time to compile per robot (should be similar; autotune affects
  dispatch only).
- Per-algo single-call and N=256 batch numbers (should match or
  improve everywhere; flag any algo that regresses ≥10%).

If autotune wins consistently, leave the README recommendation as
"recommended for production." If it has mixed results, demote to
"experimental; A/B compare per machine."

## Loose ends

- **MJX go2/g1 broken** — `_update_constraint` rejects float-indexed
  array access on current JAX/MJX pins. iiwa14 only on current pins.
- **ee_pose_gradient single-call cicc -O2 LICM elision** — anti-LICM
  machinery is in place but cicc -O2 still defeats `grid_licm_barrier`
  for this specific kernel (sm_8x FYI 2). Batch path is correct.
- **Pre-existing glass_nv ee_pose_grad regression** noted by sm_8x
  agent: 5-10× slower than glass on go2/g1 fixed. **Step 3 validation
  will show if reinstated batched_1d resolves this** — early go2 test
  shows 79µs at N=256 vs 149µs with the SIMT-only stub. Closer to
  the May-13 56µs baseline but not exactly there.
- **idsva_so / fdsva_so floating-base gate** — gated by
  `enable_floating_second_order` in `_normalize_codegen_algorithms`.

## Strategic

Once post-integration sm_120 + 5090 + sm_86 data are all in, decide
PR2/PR3 direction. Sweep is running now.

## Key files (quick reference)

| File | Purpose |
|---|---|
| `test/benchmarks/run_multi_version.py` | 6-column orchestrator |
| `test/benchmarks/generate_report.py` | `--mode multi_version` |
| `test/benchmarks/baselines/grid/run.py` | GRiD harness — ccache split, `--cicc-opt-level`, safeguards |
| `test/benchmarks/baselines/pinocchio/run.py` | Pinocchio harness — ccache, `effective_thread_count` |
| `test/benchmarks/baselines/util/experiment_helpers.h` | `effective_thread_count(N, MAX)` C++ |
| `GRiDCodeGenerator/helpers/_code_generation_helpers.py` | `_no_licm_barrier()`, anti-LICM |
| `GRiDCodeGenerator/helpers/_lin_alg_helpers.py` | GLASS vendoring, wrappers, DEFINE filter |
| `GRiDCodeGenerator/helpers/_topology_helpers.py` | `custom_is_constant` memoization |
| `GRiDCodeGenerator/algorithms/_eepose_gradient_hessian.py` | New `gemm_strided_batched_1d` emission |
| `GRiDCodeGenerator/GRiDCodeGenerator.py` | `KERNEL_ATTR_MANIFEST` |
| `GLASS/src/nvidia/tuning_table.cuh` | Per-SM cuBLASDx-vs-SIMT lookup table; regenerate per-machine via `bench/autotune.py --sm AUTO` |
| `docs/handoff-glass-rfc-gemv-auto-dispatch.md` | Active asks of GLASS (round 2) |

## How to use this doc

- Update this file at session boundaries — keep it as the living
  source of truth.
- When a handoff doc (like the GLASS RFC) gets delivered upstream,
  fold its historical content into a "Resolved-upstream notes" section
  here and delete the source doc.
- Active asks live in their own docs; this file just indexes them.
