# GRiD release collection

This is a new, validation-gated collector for the proposed release figures. It
does not reuse historical timing values or the old drivers' unmatched `with_mem`
measurements. Collision work is deliberately excluded. Nothing here changes
GPU/CPU clocks, deploys website data, or approves a performance claim.

See [BUG_TRIAGE.md](BUG_TRIAGE.md) for the latest fixes and retests, and
[SMOKE_STATUS.md](SMOKE_STATUS.md) for the original smoke audit. The original
Pinocchio core blocker is fixed. The reviewed `fp32-fd-warnings` policy now
allows the expanded FD table's bounded fp32 discrepancies to be timed and
included with explicit accuracy warnings, not strict validation-pass labels.

**Local readiness, 2026-09-25:** policy v2 is tested; all 580 selected
core/wrapper/table cells are prepared at B=16–256. Fresh cached checks retain
all 10 G1 MJX FD/grad-FD cells with disclosed warnings, and all 16 representative
wrapper checks pass strictly. The focused suite passes 133 tests plus two
subtests. This is preparation/smoke evidence, not release benchmark data; the
full collection still validates every cell. See the readiness notes for captures.

Run from the repository root, using the existing development environment:

```bash
cd /home/plancher/Desktop/GRiD
# No GPU work and no files written. Inspect the exact job matrix first.
.venv/bin/python -m test.benchmarks.release.collect --stage core

# Real numerical + timing-path smoke checks. Fresh output directory required.
.venv/bin/python -m test.benchmarks.release.collect --stage core --smoke \
  --execute --output test/benchmarks/results/release-core-smoke
.venv/bin/python -m test.benchmarks.release.collect --stage wrappers --smoke \
  --execute --output test/benchmarks/results/release-wrapper-smoke

# Export an auditable table and draft clustered-bar figures, not release claims.
.venv/bin/python -m test.benchmarks.release.report \
  test/benchmarks/results/release-core-smoke \
  --output test/benchmarks/results/release-core-smoke-report
```

Open the resulting `index.html` directly, or serve its containing directory
with `python -m http.server 8011 --bind 127.0.0.1 --directory PATH_TO_REPORT`.
The actual website remains separate and its benchmark placeholders remain in
place. All generated reports explicitly say **DRAFT** or **SMOKE TEST**.

## Prepare now, collect later

Preparation builds/loads each implementation and exercises both available API
paths at B=16,32,64,128,256. It writes **no timing samples**, performs only a
finite-output check (not oracle validation), and is explicitly rejected by the
performance report exporter. It does not change GPU clocks or baseline math.

```bash
.venv/bin/python -m test.benchmarks.release.collect --stage core \
  --prepare-only --execute --output test/benchmarks/results/prepare-core
.venv/bin/python -m test.benchmarks.release.collect --stage wrappers \
  --prepare-only --execute --output test/benchmarks/results/prepare-wrappers
.venv/bin/python -m test.benchmarks.release.collect --stage table \
  --operations minv forward_dynamics forward_dynamics_gradient fdsva_so \
    end_effector_pose end_effector_pose_gradient end_effector_pose_hessian \
  --prepare-only --execute --output test/benchmarks/results/prepare-table
```

Run sequentially. The collector enables an owner-controlled persistent JAX
cache in `test/benchmarks/results/release-jax-cache` for both preparation and
later collection. GRiD's build cache, the content/toolchain-keyed Pinocchio and
native bridges in `release-build-cache`, and Warp's usual kernel cache are
reused too. Do not import executable caches from untrusted sources. Keep the
same checkout, environment, GPU, operations, and maximum batch size; changing
them may require recompilation. Preparation is optional, never an accuracy gate.

Later collection still performs process/model initialization, oracle checks,
warmups and timed repetitions. JAX tracing can recur even on an executable
cache hit. Cache misses compile before timing; preparation reduces startup work,
not the amount of statistically useful measurement. Do not delete these caches
between preparation and collection.

## Collection after reviewing smoke failures and the protocol

Omit `--smoke` to run B=16,32,64,128,256,1024, three isolated warmed repetitions,
five warmups sustained for at least `--warm-seconds` (default 1.5 s) and 30
timed samples per boundary. A displayed value is the median of the three **run
means**, not a pooled single-call median. Whiskers show the range of those
means. Adjust repetitions/iterations explicitly when needed.

The warm-up is time-based on purpose: a handful of microsecond calls never
leaves the idle clock on a GPU that cannot be clock-locked without root (this
box idles far below its sustained boost; `timeGRiD_common.h` records the
measured range), so every backend, CPU or GPU, is driven for the same wall time
before its samples are taken. `nvidia-smi -lgc` by the operator remains the
stronger control and is recorded in provenance when used.

```bash
.venv/bin/python -m test.benchmarks.release.collect --stage core \
  --execute --output test/benchmarks/results/release-core
.venv/bin/python -m test.benchmarks.release.collect --stage wrappers \
  --execute --output test/benchmarks/results/release-wrappers
# Deferred table work; selecting only the non-core operations avoids recollecting core ops.
.venv/bin/python -m test.benchmarks.release.collect --stage table \
  --accuracy-policy fp32-fd-warnings \
  --operations minv forward_dynamics forward_dynamics_gradient fdsva_so \
    end_effector_pose end_effector_pose_gradient end_effector_pose_hessian \
    crba nonlinear_effects generalized_gravity ccrba coriolis_matrix \
  --execute --output test/benchmarks/results/release-table
# The CPU/tensor baselines on the core operations are not in the core stage
# (PRIMARY selects the headline comparators only); collect them separately.
.venv/bin/python -m test.benchmarks.release.collect --stage table \
  --operations inverse_dynamics inverse_dynamics_gradient idsva_so \
  --backends mujoco_cpu bard frax \
  --execute --output test/benchmarks/results/release-table-core-baselines
```

`--robots`, `--operations`, `--backends`, and `--batches` restrict work before
compilation, not just plotting. `--smoke --batches 16 256` exercises both endpoints
without becoming release evidence. `--timeout` is per job, including build.
Subprocesses run sequentially. Do not run other GPU experiments concurrently
during a real capture. Idle-device status and thermal/clock stability still need
human review; one `nvidia-smi` snapshot cannot certify an idle collection.

Fresh directories prevent overwriting an earlier capture. On failure, the
collector continues to record other jobs and exits nonzero. An interrupted run
can still be exported: unrecorded planned cells become `not_collected`, not zero.
There is no automatic resume; collect a narrowly selected retry in a fresh
directory and choose which version to retain. Do not merge duplicate repeats.

## Measurement contract and implementation choices

- **Core plot configuration:** GRiD's CUDA host call (`grid_cuda`) and its JAX
  resident API, Pinocchio CPU codegen AND the standard Pinocchio API
  (`pinocchio_plain`, the same fp32 algorithms without CppADCodeGen — the
  stacked figure draws it as a cap over the codegen bar, exactly like the
  memory/wrapper caps over GRiD's kernel), MJX, and MuJoCo Warp for RNEA; omit Warp
  for the gradient; GRiD and analytical Pinocchio for the Hessian. GRiD JAX is
  explicitly labeled, not presented as raw native kernel latency; the CUDA host
  call IS that latency (see below). The wrapper figure includes the CUDA host
  call, the native C ABI, NumPy/pybind, JAX, and PyTorch, and the report writes
  an overhead decomposition (compute, memory traffic, C-ABI staging, Python,
  framework dispatch, framework round trip) as differences of the same cells.
  Review which bar is the headline before publishing.
- **`grid_cuda` — GRiD's own C++ host calls.** `kernel_bridge.cu` is compiled
  per robot/operation against the SAME `grid.cuh` the wrapper `.so` was built
  from (same nvcc flags and arch; content-keyed on the header, the bridge, the
  flags and the compiler). Its `resident` boundary is the generated
  `<op>_compute_only` host function (inputs already in the gridData arena,
  kernel launch, device sync); its full-call boundary is the generated `<op>`
  host function (H2D copies, kernel, D2H copy). The batch is one block per
  sample at the artifact's baked per-algorithm thread count (recorded). Both
  outputs, and the two boundaries against each other, must be bitwise equal to
  the NumPy wrapper's result before any timing is kept. No Python in the loop.
- Full-call wall time includes host inputs, necessary H2D copies, evaluation,
  synchronization, and **all selected outputs copied to host**. Resident wall
  time includes ordinary API dispatch and synchronization. Input generation,
  oracle conversion, code generation/JIT, setup, and warmups are excluded.
- Gray hatched caps are full-call minus resident wall time from the same job.
  They are not separately measured PCIe or Python costs (for `grid_cuda` the
  cap is exactly the H2D/D2H traffic of one host call). Negative differences
  are flagged for recollection, never silently clamped. C ABI/NumPy and CPU
  baselines have full-call measurements only and are unstacked.
- MuJoCo Warp is timed as a captured CUDA graph replay (its own benchmark
  path) for both boundaries after one eager call has loaded every kernel; the
  eager launch is recorded as a secondary `resident_eager` series and never
  plotted as the library's cost.
- The native timing loop calls the same C ABI and compiled artifact used by
  the wrappers, with no Python inside the loop. Its output is checked bitwise
  against the NumPy wrapper. Native output allocation is inside the timer;
  the final validation copy is outside it. Native bridge coverage currently
  includes RNEA and grad RNEA, exactly the wrapper-study operations.
- Each robot/backend/operation/repetition is a fresh process. Builds and JITs
  are not timed. Pinocchio runs on a persistent C++ pool (`release_pool.h`)
  with an independent model/data/codegen context per thread; the batch is
  split into contiguous slices, slice 0 on the calling thread. Every cell is
  timed at each candidate thread count — 1, `max(1, B//16)` and the
  `--cpu-threads` ceiling (default eight) — and the BEST run mean is the
  reported full-call time; every variant and the selected count are kept in
  the capture and the report's `threads` column. BLAS/OpenMP threads are
  pinned to one. This is still **not** a claim of the optimal possible CPU
  threading implementation, but no Python executor is inside the timed call.
- Full Jacobians are validated as two blocks (d/dq and d/dv) so the gross-error
  backstop cannot hide a wrong velocity block under a large position block.
- Bias (`nonlinear_effects`) and gravity are each library's inverse dynamics at
  zero acceleration (and zero velocity) in ITS convention; the oracle transport
  carries the pin-frame correction ("qacc = 0" is not frame-invariant on a
  floating base). `crba` is the dense mass matrix (MJX `crb` + `full_m`,
  MuJoCo `mj_crb` + `mj_fullM`, BARD `crba`, Frax `mass_matrix`, Pinocchio
  codegen `CodeGenCRBA` / standard `crba`); Warp's dense `qM` layout is not
  validated yet. The centroidal momentum and Coriolis matrices are GRiD vs the
  standard Pinocchio API only (no codegen class, no simulator output).
- Each GRiD capture records the fitted `workspace_slots`: a value below the
  batch means the kernels grid-stride with fewer blocks in flight (still
  correct); B=1024 cells should be read with that column.
- Default arithmetic is fp32. JAX matrix products use `highest` precision
  (no reduced-precision TF32 default), x64 is disabled, and BARD TF32 is disabled.
  Pinocchio codegen uses `-O3`, without `-Ofast`, and nine significant digits
  (`float::max_digits10`) when serializing fp32 constants, rather than the
  upstream six-digit default that loses model precision. Current Pinocchio analytical
  Hessian and direct FK paths use fp64; installed MuJoCo CPU uses double too.
  Those exceptions are recorded and starred in figures. Pinocchio materializes
  its output in float64 storage even for fp32 codegen; storage casts are timed.
- Pinocchio FD/grad FD are explicitly Minv/RNEA compositions, not mislabeled
  native ABA codegen. Analytical FDSVA-SO is composed from second-order RNEA
  and first-order ABA derivatives. No baseline Hessian timing uses finite
  differences or nested automatic differentiation.

## Matching and validation

`fixtures.py` resolves the curated iiwa14-fixed, go2-floating and g1-floating
URDFs. Every implemented adapter uses those exact URDFs, not a same-named
Menagerie model. Simulator import strips visual/collision geometry, preserves
fixed bodies, adds the floating root when necessary, and disables constraints,
contacts, damping, friction, armature and springs. Gravity is [0,0,-9.81].

Inputs are deterministic, bounded joint states and normalized free-base
quaternions, with nonzero velocity/acceleration/torque samples. Each worker saves
the actual fp32 arrays and both file and value hashes. MuJoCo adapters reorder
joints and transport the independent oracle into MuJoCo tangent coordinates
outside timing. Full Jacobians differentiate q in nv-dimensional tangent space
and velocity, with acceleration/torque held fixed. Four full SO tensor blocks are
materialized; no VJP/JVP substitutes. FK is one named endpoint's xyz+RPY pose,
not an all-body transform dump.

The independent numerical oracle is the Pinocchio adapter in RBDReference,
running in fp64 on the saved fp32 values. Pinocchio baseline validation therefore
checks its native/codegen bridge against the reference adapter, not an independent
library. FK coordinate derivatives in the oracle use finite differences; those
are validation-only and never timed as an analytical comparator.

The strict gate remains entrywise `atol=1e-3, rtol=2e-4`. This is **not** a uniform
0.02% accuracy guarantee. The default `--accuracy-policy strict` still blocks a
cell that exceeds it. Following review, `--accuracy-policy fp32-fd-warnings`
retains such cells as `accuracy_warning` for fp32 Minv, FD, grad FD and FDSVA only,
equally for all backends. A gross-error backstop requires every output block's
relative L2 error to be at most `1e-3` (0.1%); this is not an entrywise bound.
Shape/nonfinite failures, fp64 discrepancies and other operations still block.
Policy **v2** applies that same rule to bounded host/resident and repeatability
differences, reflecting the diagnosed stock-MJX reduction-order variation.
Host and resident outputs are independently checked against the oracle before
and after timing; their variations are recorded separately. A close pair of
wrong outputs cannot bypass the oracle. Native C++/NumPy comparison stays exact.
These are sampled pre/post checks, not a guarantee for every timed invocation or
proof that no small state mutation exists. Captures record the policy version;
v1 captures retain their original strict variation gates when exported.
Old failed captures are not retroactively granted timings; recollect them.

Caption/table footnote:

> FP32 forward-dynamics computations can amplify rounding and cancellation,
> particularly in high-velocity cases. Selected entries show percent-level
> discrepancies from the fp64 reference; relative errors can be larger near
> zero. GPU reduction order can also cause small run-to-run differences.
> Accuracy-warning timings are retained with measured errors and entrywise
> exceedance counts, not labeled strict validation passes.

Do not describe single-digit percentages as an upper bound: the pinned G1
Hessian has about 17% relative error in one entry with reference magnitude above
0.1, and larger percentages closer to zero. Output scales/units differ, so these
are not a universal robotics application error bound or proof that every backend
discrepancy is caused solely by rounding.

## Availability is not inferred capability

`protocol.py` is the executable matrix. Planned gaps are explicit:

- Pinocchio FK pose-coordinate gradient/Hessian adapters remain pending; its
  spatial derivatives are not automatically the same as xyz+RPY derivatives.
- MuJoCo Warp dense Minv and full derivative adapters are pending. MuJoCo CPU
  derivative adapters are pending. MJX Minv and FK-gradient adapters are pending.
- BARD currently has RNEA and FD adapters. Frax covers fixed-base RNEA/FD/Minv;
  its six-coordinate free base still needs a validated quaternion conversion.
- Nonanalytical Hessian approaches are excluded from this study, not labeled
  unsupported. Collision data and protocols remain deferred.

Use `adapter_pending`, `model_mismatch`, `excluded_method`, `validation_failed`,
`error`, or `timeout` accurately. An unwritten adapter is not evidence that a
competing library cannot support the operation.

## Artifacts, dependencies, and tests

Each capture contains `plan.json`, incremental `results.json`, job logs,
per-job JSON, input NPZs, and a completed-capture SHA-256 `manifest.json`.
Metadata includes commit/dirty diff/submodule state, collector source hashes,
package versions, CPU/GPU identity, arithmetic/thread policy, robot hashes,
and GRiD build metadata or baseline-library hashes. Raw timed samples are kept.
The report checks recorded hashes before exporting CSV/JSON/HTML/PNG/SVG.
No speedup claims or automatic website updates are generated.

The existing `.venv` supplies the optional backends. Native compilation needs
g++, GRiD needs its usual CUDA toolchain, and Pinocchio codegen needs the installed
CppADCodeGen headers/libraries. Missing dependencies are errors, not unsupported
capability claims. The first Pinocchio bridge compilation can take several
minutes and approximately 10 GB of host RAM. Its content-keyed library cache is
under the ignored `test/benchmarks/results/release-build-cache/`; JIT libraries
are keyed on the code generators and their settings (`pin_codegen_init.h`,
`timePinocchio.cpp`, the util headers, compiler flags, gcc and the Pinocchio
version — the `codegen_cache_key` in adapter metadata), URDF and base mode,
with distinct operation library names, protected by a build lock, and hashed
in the capture; a change to the dispatch code in `pin_bridge.cpp` rebuilds the
bridge only. The CUDA host-call harnesses (`kernel-<key>.so`, one per robot and
operation) live in the same cache, keyed on the artifact header. Full G1 codegen setup may take minutes
per op on the first run; repetitions reuse the compiled library but create
fresh contexts and repeat all warmups.

```bash
.venv/bin/python -m pytest test/benchmarks/test_release_collection.py \
  test/benchmarks/test_timing_contracts.py test/benchmarks/test_autotune_picker.py -q
```

These CPU tests cover the matrix, bad selections, numerical gates, timer
synchronization, native ABI bridge with a fake CPU library, missing/negative
overhead, timeouts, hash checks, and strict repeat aggregation. Real GPU smoke
captures are a separate requirement and are not a replacement for the final
release's full GPU validation receipt.
