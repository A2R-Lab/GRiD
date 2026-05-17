# Handoff: full benchmark sweep on the 5090

**Audience**: Claude agent running on the user's new 5090 (sm_120, no thermal-throttling)
workstation. This doc is self-contained — read it, then run the sweep.

**Source repo**: GRiD-A2R, branch `modernizing-tests`.
**Latest commit at handoff**: `cdb4f4e bench: ungate fdsva_so floating + symmetric SO display labels`

---

## Why we're handing this off

The current dev box runs the sweep at ~25 min/robot/base/backend (slow CPU + thermal
throttling on the GPU). The 5090 box is faster, thermally stable, and Blackwell-class
(same arch family as this box, so directly comparable). Bottom line: better numbers,
faster turn-around — worth doing the canonical sweep there.

You (the 5090 agent) will:
1. Sync the branch
2. Smoke-verify the bench compiles + runs end-to-end on your machine
3. Launch the 12-combo sweep
4. Sanity-check the resulting JSONs / report
5. Commit the report + push back

The bench is set up to produce **complete data on every cell** as of `cdb4f4e`. The
sweep should not need any code changes from you. If something doesn't compile or a
cell crashes, **stop and report back to the user** — that's a regression I missed.

---

## Context — what the branch is and why

### The high-level goal

`GRiD-A2R/modernizing-tests` is an active refactor pass over the GRiD CUDA codegen
pipeline. Three concurrent threads:

1. **Codegen modernization** — clean shared-memory arena, opt-in dynamic shared mem
   via `cudaFuncSetAttribute`, proper anti-LICM machinery for single-call timings,
   the GLASS round-2 linalg backend wired through.
2. **Pinocchio-grounded equivalence** — `test/pinocchio_equivalents/` now uses the
   Pinocchio C++ implementation as the golden oracle for every RBDReference public
   algorithm. 320 pass / 12 narrow singular-Minv skips across all 8 manifest robots
   × fixed+floating.
3. **Second-order ungated** — `idsva_so` / `fdsva_so` now generate + run cleanly
   for floating-base, not just fixed. A second variant (world-frame single-pass)
   sits alongside the original body-frame multi-pass impl.

The sweep you're about to run captures the timing data that justifies these changes.

### Naming convention you'll see in the bench

Two IDSVA-SO variants exist now — they differ only in **reference frame**:

| Variant | Reference frame | Best for |
|---|---|---|
| `idsva_so_body_frame` | Body-frame propagation, multi-pass | **Fixed-base** robots |
| `idsva_so_world_frame` | World-frame propagation, single-pass | **Floating-base** robots |

Smokes on iiwa14_fixed showed body-frame 30× faster than world-frame on fixed; on
iiwa14_floating world-frame is 1.9× faster than body-frame; on g1_floating it's
3.6× faster. Both are kept in the default sweep — the crossover is the point of
the comparison.

---

## What's resolved (do not re-do anything in this list)

Folded from the previous handoff docs (deleted). The plan file at
`~/.claude/plans/let-s-make-a-plan-foamy-sunbeam.md` (on the previous dev box)
has the deeper history; this is the short version you need on the 5090 side.

**Submodules** (both pinned via main repo commits):
- `GRiDCodeGenerator` at submodule HEAD `9a8a595` (fdsva_so OOB fix)
- `RBDReference` at submodule HEAD `5c47061` (README polish)

**Done in the last 24 hours of work (chronological)**:
- GLASS round-2 integration: `gemm_batched_1d`, `gemm_strided_batched_1d`,
  `should_use_cublasdx<>` auto-dispatch via `tuning_table.cuh`.
- `_eepose_gradient_hessian.py` cuBLASDx path reinstated via batched GEMM.
- `linalg_smem_for()` + `GRID_BENCH_NVIDIA_MIN_DIM` env knob removed
  (Phase 5a cleanup); linalg wrappers now compile-time `#if` only.
- `timeGRiD.cu` split into `_single.cu` (`-rdc=true`, anti-LICM correct) +
  `_batch.cu` (no `-rdc`, recovers 3× SIMT batch perf on chain-heavy algos).
- `--cicc-opt-level` flag (sm_86 cicc-O3 hang workaround) — plumbed everywhere.
- `--ptxas-opt-level` flag (sm_86 ptxas wedge workaround) — plumbed everywhere
  (including `run_multi_version.py` forwarding). Both flags are sm_86-specific;
  the 5090 (sm_120) won't need either.
- `KERNEL_ATTR_MANIFEST` in `GRiDCodeGenerator.py`: emits
  `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, ...)` for every algorithm
  kernel. Fixes silent kernel launch failures on g1 floating ABA/FD/MINV
  (>48 KB dynamic shared mem). Compile-time size guards wrap `fdsva_so` and
  `d2ee` registrations so init_grid doesn't fail when those kernels can't fit
  even with the attribute (happens on g1_floating for fdsva_so).
- XmatsHom helper serial-section split (reg pressure 140 → ~50). Required to
  fit `__launch_bounds__(SUGGESTED_THREADS)` cleanly on ee_pose_gradient_hessian.
- Anti-LICM rewrite: `__noinline__ grid_licm_barrier` (brittle, was being
  optimized away as no-op self-stores) replaced with **rep-stomp + output→input
  feedback**. Every `_single_timing` rep now reads previous reps' outputs back
  into shared inputs, creating a loop-carried data dependency ptxas cannot fold.
  8 of 9 algos measure correctly on sm_86 floating after this. sm_120 has all
  9 working including ee_pose_gradient.
- Dead `grid_licm_barrier` helper deleted (was unreferenced after the rewrite).
- 3 stale handoff docs deleted (folded into the plan file): GLASS round-2 RFC
  + ptxas-hang + dated session status.
- SO ungate wired in the bench: `enable_floating_second_order=True` passed to
  `gen_all_code` from `test/benchmarks/baselines/grid/run.py`. Smoke-verified
  on iiwa14_floating + g1_floating.
- **IDSVA-SO rename by reference frame**: `idsva_so` → `idsva_so_body_frame`,
  `idsva_so_spatial_v2` → `idsva_so_world_frame`. Wired through codegen + bench
  + parser + report + RBDReference + tests. Smoke-verified iiwa14_fixed: body
  28 µs, world 862 µs (matches pre-rename within noise).
- **`fdsva_so` floating-base OOB fix**: `gen_fdsva_so_inner` correctly used
  NUM_VEL throughout, but `gen_fdsva_so_kernel` / `_device` / `_host` used
  `n = NUM_POS`. For floating-base where NUM_POS = NUM_VEL + 1 (quaternion),
  this caused:
  - `s_df2` / `s_idsva_so` write of `4*NUM_POS^3` to a `d_df2` allocated for
    `4*NUM_VEL^3` → out-of-bounds write → `cudaErrorIllegalAddress`.
  - `s_u = &s_q_qd_u[2*NUM_POS]` off by one (should be `[NUM_POS+NUM_VEL]`)
    → corrupted u → corrupted qdd → poisoned everything downstream.
  - Host wrapper used `stride = 3*NUM_JOINTS` instead of `Q_QD_U_STRIDE`.

  Fixed in commit `9a8a595` (codegen) + `cdb4f4e` (bench ungate). Verified
  iiwa14_floating: `fdsva_so single = 4053 µs` real.
- Pinocchio C++ binding install docs added to `test/benchmarks/README.md`.
- `RBDReference/README.md` refreshed with full algorithm table + body/world-frame
  guidance with measured numbers.

### Outstanding from the dev box at handoff time

- **g1_floating smoke is in flight** to confirm the `fdsva_so` fix scales to the
  worst-case robot. If when you sync the repo the previous agent already finished
  this, look for `/tmp/g1_fdsva_fix_smoke.json` on the previous box — but you
  should just re-smoke on your own hardware.
- **`TODO(licm-eepose-grad)` is sm_86-specific**, deprioritized.
  ee_pose_gradient single_timing on sm_86 floating elides. sm_120 confirmed
  clean (1.78 µs iiwa14_fixed, 306 µs iiwa14_floating, 322 µs g1_floating).
  Inline TODO at
  [`GRiDCodeGenerator/algorithms/_eepose_gradient_hessian.py:687`](../GRiDCodeGenerator/algorithms/_eepose_gradient_hessian.py).
  You shouldn't hit this on 5090.

---

## What we want you to do — the sweep

### 1. Sync + machine check

```bash
git checkout modernizing-tests
git pull --recurse-submodules
git submodule update --init --recursive
# Confirm submodule heads match this doc:
git -C GRiDCodeGenerator log --oneline -1   # → 9a8a595 fix fdsva_so ...
git -C RBDReference     log --oneline -1   # → 5c47061 README: body_frame vs world_frame ...

# Confirm CUDA toolchain + GPU:
nvcc --version       # CUDA 12.x expected
nvidia-smi           # confirm RTX 5090 + driver
```

Then create the venv if it doesn't exist:
```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pip install cmeel-cppadcodegen   # for the Pinocchio codegen path
```

### 2. Smoke first (one cell, ~5 min)

Confirm the bench end-to-end works on your box before launching the long sweep:

```bash
rm -rf .pytest_cache/grid_cuda/headers .pytest_cache/grid_cuda/grid_benchmarks
.venv/bin/python test/benchmarks/baselines/grid/run.py \
    --robot iiwa14 --base fixed \
    --single-call-iters 5000 --batch-iters 50 \
    --output /tmp/smoke.json
```

Expected on sm_120 (from the previous dev box for reference — yours should be
similar or faster):

```
aba                single= ~10 us    N=256 compute= ~80 us
crba               single=  ~5 us    N=256 compute= ~65 us
ee_pose            single=  ~1.5 us  N=256 compute= ~80 us
ee_pose_gradient   single=  ~1.8 us  N=256 compute= ~50 us
fd                 single= ~10 us    N=256 compute= ~90 us
fd_du              single= ~19 us    N=256 compute= ~170 us
fdsva_so           single= ~53 us    N=256 compute= ~410 us
id                 single=  ~5.5 us  N=256 compute= ~55 us
id_du              single=  ~8.5 us  N=256 compute= ~95 us
idsva_so_body_frame  single= ~28 us  N=256 compute= ~190 us
idsva_so_world_frame single= ~860 us N=256 compute= ~5400 us  ← yes, expected
minv               single=  ~7.5 us  N=256 compute= ~75 us
```

The world-frame on iiwa14_fixed being ~30× slower than body-frame is **correct
and expected** — it's the multi-pass-vs-single-pass crossover. Floating-base
reverses the ranking.

If the smoke crashes or has any `null` values, **stop and report back**.

### 3. Launch the full 12-combo sweep

```bash
.venv/bin/python test/benchmarks/run_multi_version.py \
    --columns glass glass_nvidia \
    --robots iiwa14 go2 g1 \
    --bases fixed floating \
    --single-call-iters 50000 --batch-iters 500 \
    --output-dir test/benchmarks/results/comparison_sm120_5090_full \
    --report test/benchmarks/benchmark_multi_version_sm120_5090_full.md
```

What this does:
- 3 robots (iiwa14, go2, g1) × 2 bases (fixed, floating) × 2 GRiD backends
  (glass = pure SIMT, glass_nvidia = cuBLASDx-backed) = 12 GRiD cells.
- 50000 single-call iters / 500 batch iters per cell (~10× tighter than the
  default; gives stable medians).
- Writes per-cell JSON under `test/benchmarks/results/comparison_sm120_5090_full/`
  (gitignored) + a unified markdown report at the `--report` path.

Estimated wall time:
- With ccache warm: ~1.5-2 hr on the 5090.
- Cold: ~3-4 hr (each robot/base needs codegen + nvcc compile).
- If you have ccache, set `CCACHE_DIR` to a fast disk; bench prints when ccache
  is engaged.

**Run it in the background or via tmux** — don't tie up your terminal:
```bash
nohup .venv/bin/python test/benchmarks/run_multi_version.py \
    --columns glass glass_nvidia \
    --robots iiwa14 go2 g1 \
    --bases fixed floating \
    --single-call-iters 50000 --batch-iters 500 \
    --output-dir test/benchmarks/results/comparison_sm120_5090_full \
    --report test/benchmarks/benchmark_multi_version_sm120_5090_full.md \
    > /tmp/sweep.log 2>&1 &
```

Check progress with `tail -f /tmp/sweep.log`.

### 4. What "success" looks like

When the sweep finishes, you should have:
- 24 JSON files in `comparison_sm120_5090_full/` (one per robot/base/column,
  for `grid_glass` and `grid_glass_nvidia` only — Pinocchio/MJX/Frax columns
  are excluded by `--columns`).
- 1 markdown report at `benchmark_multi_version_sm120_5090_full.md`.

**Zero-cost spot checks against the JSONs**:

a. **No unexpected `null` cells** — every algo should have populated
   `single_us`, `batch_{16,32,64,128,256}_with_mem_us`,
   `batch_{...}_compute_only_us`. The one expected `null` is fdsva_so
   on g1_floating (see (b) below). Any other null is a regression: stop
   and report.

b. **`fdsva_so` floating populates for iiwa14 + go2** — the fix from
   commit `9a8a595` produces real numbers, not crashes.
   **`fdsva_so` on g1_floating is hardware-blocked, expected `null`** —
   its kernel needs ~197 KB of dynamic shared memory, but the sm_120
   device cap is ~100 KB. The bench gracefully skips it at runtime with
   a clear "SKIPPED" message in stdout (look for
   `[N:K]: FDSVA_SO SKIPPED` or `Single Call FDSVA_SO SKIPPED`). Note
   this in the report writeup as a hardware-blocked cell, not a
   regression. Deeper fix (moving more fdsva_so intermediate state to
   global memory) is queued as a follow-up.

c. **`ee_pose_gradient` single-call > 0.5 µs on every floating-base cell**
   (sanity check that anti-LICM is working on sm_120 floating). Previous
   measurements:
   - iiwa14_floating: ~306 µs ✓
   - g1_floating: ~322 µs ✓

d. **world-frame vs body-frame relative perf**:
   - Fixed: body-frame should win by 20-40×
   - Floating: world-frame should win by 1.5-4×
   If either ranking flips on any robot, flag it — that's interesting either
   way.

e. **glass_nv vs glass** — historically on go2_fixed glass_nv was 5-10×
   slower than glass for ee_pose_gradient + ee_pose_hessian. The
   reinstated cuBLASDx batched-GEMM path (commit `cdb4f4e` lineage) may have
   resolved this. Spot check those cells specifically.

### 5. Commit + push

The per-cell JSONs are gitignored — they only live on disk. The markdown
report is **also gitignored** (per `test/benchmarks/.gitignore` pattern
`benchmark_multi_version_sm*.md`) so the canonical place to share it is to
either:

- Paste the report text back to the user directly (it's ~few hundred lines)
- Copy it to a shared location outside the repo

If the user wants it tracked, they can move it under a tracked name (e.g.
the canonical `benchmark_multi_version.md` which isn't gitignored). Default:
**don't commit the report**, just produce it and share back.

If you made any code changes during smoke or sweep (you shouldn't have had to):
```bash
git add -p   # be selective
git commit -m "..."
git push
```

---

## Outstanding follow-up work (context, not your job right now)

These are queued for after this sweep. Mentioned only so you know what the
sweep data feeds into next.

### Compile-speed: per-algo TU split (P6 7b)

`timeGRiD.cu` currently splits into single + batch (2 TUs). Next step: split
each into per-algorithm TUs (`timeGRiD_id.cu`, `timeGRiD_fd.cu`, ...) with
parallel `nvcc -c` under a thread pool. Expected to cut sweep wall time by
~3-5× because cicc is the bottleneck and per-algo TUs are 10× smaller.
**Has not been done yet** — if you have spare cycles after the sweep, this
is the next biggest win.

### Phase 6 baseline expansion

- 6a regression check vs pre-glass ref `d2c0d18` (worktree + rerun).
- 6b MJX pin downgrade — go2/g1 currently fail on current JAX/MJX pins
  (`_update_constraint` rejects float-indexed array access); iiwa14 works.
- FD-solve A/B microbench: `direct_minv + gemv` vs `crba + posv`
  (cuSOLVERDx Cholesky).
- CuRobo deferred (doesn't expose the API we'd need).

### Phase 7 codegen perf (measurement-gated)

- 7-SO-a: batched GEMM for IDSVA-SO / FDSVA-SO inner 6×6×6, using the new
  GLASS `gemm_batched_1d`. Retain current SIMT path as fallback; benchmark
  both.
- 7-SO-b: CUB-backed L1 reductions in IDSVA-SO outer pass.

### Architecture: any-thread-count library functions

Carry-over design question: GRiD kernels currently pin `__launch_bounds__(
SUGGESTED_THREADS)`. People calling GRiD from inside their own kernels may
want different thread counts. Need perf-mode (current) + compat-mode (no
launch_bounds, fallback inner variants) per algorithm. Open design;
1-2 days to scope.

### sm_86 LICM follow-up

`TODO(licm-eepose-grad)` — sm_86 floating ee_pose_gradient single_timing
still elides despite the rewrite. sm_120 confirmed clean (so won't affect
your sweep). Parked until someone runs on sm_86 and cares.

### Pinocchio-equivalence cleanup (P4)

- Audit comparison plumbing (shapes, value scales, semantics) →
  `test/pinocchio_equivalents/COVERAGE.md` summary table.
- Spot-check `test/cuda_equivalents/` routes through validated Python algs.
- Dropped: sample-diversity sweep, rnea∘aba identity, apply_external_forces
  test, binding-bootstrap hardening.

### Helper rename cleanup (low-priority)

A few body-frame-specific codegen helpers didn't get renamed for naming
consistency in the IDSVA-SO rename pass:
`gen_idsva_so_device_temp_mem_size`,
`gen_idsva_so_floating_reference_inner`,
`gen_idsva_so_reference_order_output_repair`. Add `_body_frame` for
consistency next time someone touches `_idsva_so.py`.

---

## Repo layout cheat-sheet

```
GRiD-A2R/
├── GRiDCodeGenerator/                # codegen submodule
│   ├── GRiDCodeGenerator.py          # entry point + init_grid + KERNEL_ATTR_MANIFEST
│   ├── algorithms/
│   │   ├── _idsva_so.py              # body_frame + world_frame variants (single file)
│   │   ├── _fdsva_so.py              # fdsva_so (uses idsva_so_body_frame_inner)
│   │   ├── _eepose_gradient_hessian.py
│   │   └── ...
│   └── helpers/_code_generation_helpers.py  # anti-LICM machinery
├── RBDReference/                     # Python reference impl submodule
│   └── RBDReference.py               # idsva_so_body_frame / idsva_so_world_frame / fdsva_so methods
├── test/
│   ├── benchmarks/
│   │   ├── baselines/grid/run.py     # GRiD per-cell harness (uses gen_all_code)
│   │   ├── baselines/grid/timeGRiD_{single,batch}.cu  # bench TUs
│   │   ├── run_multi_version.py      # multi-column sweep driver
│   │   ├── timing_parser.py          # parses stdout → JSON schema
│   │   └── generate_report.py        # JSON → markdown report
│   ├── pinocchio_equivalents/        # Python ↔ Pinocchio golden tests
│   └── cuda_equivalents/             # CUDA ↔ Python tests
└── docs/
    └── sweep-on-5090.md              # this file
```

## Verification commands you can run anytime

These print CPU reference values + CUDA kernel outputs for manual eyeball check
(both use `MATCH_CPP_RANDOM=True` so values match between runs):

```bash
# Generate grid.cuh only (no compile):
.venv/bin/grid-generate /path/to/iiwa14.urdf      # fixed-base
.venv/bin/grid-generate /path/to/iiwa14.urdf -f   # floating

# Compile + run CUDA kernels:
.venv/bin/python examples/print_grid.py <urdf>    # fixed
.venv/bin/python examples/print_grid.py <urdf> -f # floating

# Print CPU reference (RBDReference, Pinocchio-grounded):
.venv/bin/python examples/print_reference_values.py <urdf>    # fixed
.venv/bin/python examples/print_reference_values.py <urdf> -f # floating
```

URDF paths via `robot_descriptions` (auto-cached):
```
iiwa14: /root/.cache/robot_descriptions/drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf
go2:    /root/.cache/robot_descriptions/unitree_ros/robots/go2_description/urdf/go2_description.urdf
g1:     /root/.cache/robot_descriptions/unitree_ros/robots/g1_description/g1_29dof.urdf
```

(Adjust path prefix for your user.)

---

## If something breaks

The bench tries hard to give you good error messages. The two error patterns
to recognize:

1. **`GPUassert: an illegal memory access ...` (code 188)** — kernel did an
   OOB write. If it's `fdsva_so` floating-base, my fix may have a bug; report
   the robot + base + kernel line number.
2. **`GPUassert: invalid argument ...` (code 1)** — usually means
   `cudaFuncSetAttribute` wasn't called for that kernel (size exceeds the
   default 48 KB and we forgot to opt in). If it's a fresh failure on a new
   robot, the KERNEL_ATTR_MANIFEST may need an entry for that kernel.

If both bench TUs (single + batch) compile but one crashes at runtime, the
crash will name a specific `.cuh` line — look there for the kernel launch.
The launch's host wrapper (just above the kernel call) tells you which
algo is firing.

If nvcc itself hangs at cicc or ptxas (sm_86 problem, shouldn't happen on
sm_120), the `--cicc-opt-level 2` and `--ptxas-opt-level 2` flags exist
as the documented workaround.

---

## Questions / unclear bits — report back to the user

If anything in this doc is unclear, the numbers don't match my expectations,
or the sweep produces surprising results, write them up and surface back.
This is a research codebase — surprising data is the point.
