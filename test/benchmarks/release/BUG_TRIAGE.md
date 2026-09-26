# Release benchmark bug triage — 2026-09-25

All work is local on `modernizing-tests`. These are correctness and timing-path
smokes, not performance evidence. Some compilation and other workloads overlapped.
The bug investigation did not relax validation tolerances, change inputs,
promote release arithmetic, or change compiler optimization policy to obtain a
pass. After reviewing the remaining errors, the user approved collecting and
including fp32 FD discrepancies with disclosure. The new opt-in
`--accuracy-policy fp32-fd-warnings` preserves strict-check results while
retaining bounded discrepancies as `accuracy_warning`, never `validated`.

## Fixed: Pinocchio codegen truncates model constants

Installed CppADCodeGen's `ModelCSourceGen` defaults to `digits10`, which is six
significant decimal digits for float. This is insufficient to round-trip an
fp32 value. The release bridge now calls
`setParameterPrecision(std::numeric_limits<float>::max_digits10)` after
`initLib()` and before `loadLib()`, for RNEA, its gradient, and Minv.

Arithmetic remains fp32, compilation remains `-O3` without fast-math, and
the source-keyed caches force new generated libraries. The setting is captured
as `codegen_constant_digits: 9` in adapter metadata.

- Original G1 grad-RNEA gate: two failing entries at B=16.
- Corrected G1 grad-RNEA: zero failing entries, relative L2 `9.67e-7`
  (formerly `7.34e-6`).
- **45/45 Pinocchio core cells pass:** all three robots, all three core
  operations, and all five requested batch sizes. The same eight-worker ceiling
  and batch-dependent 1/2/4/8/8 dispatch are exercised.
- This fixes the core blocker, not every Minv/FD numerical discrepancy.

Captures:

- `results/release-investigation-pin-constants-20260925/` (under `test/benchmarks/`)
- `results/release-investigation-pin-core-20260925/`

## Fixed: Hessian oracle extension rebuild loop and race

The RBDReference extension loader detects changes in both C++ and `setup.py`,
but setuptools could copy an old binary when only `setup.py` was newer.
Consequently every oracle evaluation attempted another build. Concurrent
workers could delete/copy the same `.so` and fail intermittently.

The loader now locks its check/build/import, forces a relink when stale, and
checks that the result is current. It imports Pinocchio first to initialize the
wheel's shared-library loading; standalone loader calls no longer depend on
an earlier caller import. Four simultaneous fresh Python processes each loaded
the extension twice successfully, without rebuilding.

**This fix is inside the `external/RBDReference` submodule.** Its local changes
must be committed/pushed there and the GRiD gitlink updated when preparing the
release; a parent-repository commit alone will not transport this fix.

## Fixed: URDF transform simplification imposed a precision floor

`external/URDFParser/Joint.py` rationalized spatial/homogeneous transforms with
`nsimplify(..., tolerance=1e-6)`. On G1, the shoulder rotation coefficients
differed from the URDF model by about `2e-8`, and `R R^T - I` reached `3.6e-8`
even in double precision. Algorithms that assume rigid transforms then amplify
this preprocessing error. This is distinct from fp32 rounding during a kernel.

The parser now uses a shared `_simplify_transform` helper at `1e-15` for all
five transform-simplification sites. Existing explicit quarter-turn angle
cleanup remains unchanged. Six new CPU regression cases cover coefficient
preservation, orthogonality, and quarter-turn cleanup; the full 66-test parser
suite passes.

The diagnostic established causality before changing the parser:

- Promoting inputs to fp64 and renormalizing quaternions did **not** resolve
  the remaining 35 G1 FDSVA entry failures.
- The GPU FDSVA contraction agreed with a CPU contraction of its intermediate
  outputs within `3.6e-11`; replacing only the RNEA-Hessian intermediates with
  the independent oracle made the composition pass.
- Tightening only parser rationalization in a CPU replay brought the worst
  sample's world-frame RNEA Hessian error from about `1.06e-4` to `5.57e-12`.

The generated-source cache detects the parser change. After rebuilding, all
**45/45 GRiD core cells** and **120/120 wrapper cells** pass. The iiwa14/go2
generated core artifacts deduplicated to the same binaries. All 24 wrapper jobs
retain byte-identical input values across the fix. The fp64 G1 FDSVA GPU replay
now passes all entries: maximum error falls from `0.01545` to `4.25e-9`, with
maximum block relative L2 `3.14e-13`. This confirms the preprocessing bug was
responsible for the fp64 error floor.

**This is a second submodule change**, in `external/URDFParser`; include its
commit and updated GRiD gitlink when transporting the fix.

## FD investigation: precision sensitivity, not a cleared fp32 gate

The validation-only `diagnose_precision` module replays deterministic fp32
input values through fp32/fp64 GRiD NumPy kernels against the existing fp64
oracle. It does not collect timing or change the release collector's dtype.
All four FDSVA output blocks remain checked.

| Replay at B=16 | fp32 failing entries | fp64 failing entries | fp64 max absolute error |
|---|---:|---:|---:|
| iiwa14 grad FD | 2 | 0 | `5.87e-11` |
| iiwa14 FD Hessian | 404 | 0 | `4.60e-10` |
| go2 FD Hessian | 1 | 0 | `7.63e-12` |
| G1 grad FD | 140 | 0 | `2.88e-5` |
| G1 FD Hessian, before parser fix | 10,185 | 35 | `1.55e-2` |
| G1 FD Hessian, after parser fix | 10,397 | 0 | `4.25e-9` |

G1 grad-FD relative L2 drops from `2.41e-5` to `6.75e-9` before the parser fix.
The remaining G1 Hessian discrepancy was traced to parser precision as above,
not dismissed as unavoidable fp32 arithmetic error.

For the original 16 samples, maximum mass-matrix condition numbers are roughly
6,513 (iiwa14), 12,220 (go2), and 138,973 (G1). Maximum absolute generalized
accelerations are about 1,146, 748, and 3,137 respectively. Inverse-mass and
derivative composition can amplify fp32 rounding/cancellation on these inputs.
The matched fp64 results strongly support numerical sensitivity for the replayed
cases, but do not establish that every other backend's failing cell is correct.

The Pinocchio constant fix still leaves seven B=16 Minv/FD/grad-FD checks failing.
After the parser fix, G1 fp32 grad FD still has 147 failing entries (relative L2
`3.11e-5`); preserving model constants does not eliminate fp32 cancellation.
The final G1 fp32 FDSVA replay has 10,397 failing entries, maximum absolute error
`1.1593` and maximum block relative L2 `7.50e-5`. Its four blocks contain
8,762 / 1,105 / 0 / 530 failing entries. The fp64 replay on the same saved input
values has zero. The diagnostic processes have finished; no release timing
sweep was started.
The entrywise gate remains `atol=1e-3, rtol=2e-4`. Under default strict policy,
failed cells are excluded. Under the reviewed opt-in policy, fp32 Minv/FD-family
cells may be retained if every output block's relative L2 error is at most
`1e-3` (0.1%). Shape/nonfinite, gross-error, repeatability and API-boundary
failures remain excluded. The rule is backend-neutral. Error metrics and
exceedance counts accompany retained timings; old failed captures are not
retroactively relabeled. No input or arithmetic-precision changes are made.

Reproduce a diagnostic in a fresh directory:

```bash
.venv/bin/python -m test.benchmarks.release.diagnose_precision \
  --robot iiwa14 --operations forward_dynamics_gradient fdsva_so \
  --output test/benchmarks/results/precision-replay
```

Saved arrays and per-block metrics are in:

- `results/release-investigation-iiwa-precision-20260925/` (gradient only;
  interrupted at Hessian setup by the now-fixed oracle race)
- `results/release-investigation-iiwa-hessian-precision-20260925/`
- `results/release-investigation-g1-precision-20260925/`
- `results/release-investigation-go2-precision-20260925/`
- `results/release-investigation-g1-parser-fixed-20260925/`
- `results/release-investigation-g1-parser-fixed-fp32-20260925/`

## Readiness

All **135/135 core cells** passed across robots and batches; after the parser
fix, the affected GRiD subset was rechecked with **45/45 passing** and all
**120/120 wrapper cells** passed again. The CPU suite passes **124 tests plus
two subtests**. Fresh datasets contain source/build/input hashes; these separate
diagnostic runs must not be spliced together into performance figures.

The expanded-table policy is now reviewed and implemented. Collect it with
`--accuracy-policy fp32-fd-warnings`, while retaining the core/wrapper strict
checks. The new seven-operation, all-robot/backend B=16 smoke sweep completed
in `results/release-table-warning-smoke-20260925/`: 47 strict passes, 16 retained
accuracy warnings, two validation failures, and 82 explicit unavailable jobs.
All 21 GRiD cells were retained (16 strict passes and five accuracy warnings).
The two failures are G1 MJX FD and grad FD: resident/host output comparisons
exceed the unchanged entrywise gate. The investigation below supports bounded
reduction-order numerical variation, not a missing frame adapter. These captures
remain excluded under the current strict repeatability/boundary policy; the
recommendation to retain this variation was subsequently implemented as policy
v2 (see below). The old captures themselves have not been relabeled.
The report exported all 63 eligible rows with their statuses and errors intact
to `results/release-table-warning-smoke-report-20260925/index.html`; excluded
rows have no exported timing. A strict-policy iiwa14 Pinocchio Minv negative
control still rejects before timing, as intended.

The footnote deliberately says **percent-level discrepancies, potentially
larger near zero**, not a single-digit upper bound. The saved G1 Hessian has
up to 17% relative discrepancy even among entries with reference magnitude
at least 0.1, versus about 3.05% for entries with reference magnitude at least
1.0; near-zero references make percentage errors much larger or ill-conditioned.
These thresholds have operation-specific units and are descriptive only.
Do not quote smoke timings. Real collection must run without overlapping GPU
experiments or heavy CPU compilation; use fresh directories and the commands
in [README.md](README.md).

## G1 MJX FD / grad-FD investigation

The existing simulator adapter already handles quaternion order, world-linear
versus body-linear velocity, force transport, and the acceleration correction
`R (a + omega cross v)`. The gradient reference includes the tangent-basis and
input/output frame derivatives while holding MuJoCo velocity/force fixed.
Joint ordering is also mapped. This is not a missing frame adapter.

Validation-only replay on G1, B=16, six host calls and six resident calls per
operation, using identical converted input values:

| Stock MJX result | FD | grad FD |
|---|---:|---:|
| Maximum fp32 relative L2 error vs oracle, across 12 calls | `2.997e-5` | `9.028e-5` |
| Largest fp32 same-path entry drift vs that path's first call | `0.303467` | `0.238134` |
| fp32 same-path repeatability comparisons passing the strict gate | 2/10 | 0/10 |
| fp64 oracle comparisons passing the strict gate | 12/12 | 12/12 |
| Largest fp64 same-path entry drift | `7.30e-10` | `5.88e-10` |

The fp32 relative L2 errors are about 0.0030% and 0.0090%, respectively; these
are aggregate errors, not bounds on individual entries. The largest observed FD
drift is sample 4 (`high_velocity_accel`), right ankle roll acceleration. The
largest grad-FD drift is sample 3 (`high_acceleration`), left ankle roll output
with respect to left hip pitch position. Here sample acceleration is not an FD
input; the label identifies the shared fixture, not a causal explanation.
Grad-FD exceedances also occur in ordinary/random states, not only high velocity.

Both default models use MJX's dense mass-matrix path. Installed MJX's reverse
body-tree scans combine child contributions with `jax.ops.segment_sum`
(`mujoco/mjx/_src/scan.py:464`), including subtree CoM, composite inertia and
force computations. Installed JAX implements this with scatter-add and documents
that conflicting update order can be nondeterministic.

**Controlled diagnostic intervention:** replacing only `jax.ops.segment_sum`
in the diagnostic process with explicit masked sums makes both operations
bit-for-bit identical across all 12 fp32 calls, including host versus resident.
Their oracle relative L2 errors remain `2.745e-5` and `8.773e-5`: removing
run-to-run variation does not remove ordinary fp32 accuracy loss. This strongly
implicates reduction order, with numerical sensitivity amplifying the resulting
rounding differences. It does not identify a particular GPU instruction or prove
that every future discrepancy has this cause.

The convention suite passes all **24 tests**. An additional G1 check compares
the transformed grad-FD oracle to MuJoCo CPU central differences on samples 2
and 4, across all 70 tangent/velocity columns and step sizes `1e-4`, `1e-5`,
`1e-6`; all six comparisons pass the original entrywise tolerance. Promoting
MJX arithmetic does not undo the already-rounded converted inputs, so its fp64
oracle residual is not expected to be machine zero.

Recommendation: keep stock fp32 MJX for fair performance comparisons and
consider explicitly retaining this bounded numerical variation under the same
backend-neutral warning budget, while preserving finite/shape checks and
checking each sampled output against the oracle. Do not benchmark the masked-sum
diagnostic replacement as stock MJX, silently promote the baseline to fp64,
or retroactively relabel the failed captures. No further production adapter or
validation-policy change was made during this investigation. No timing was
collected by these diagnostic replays.

### Follow-up: policy v2 and preparation

After user approval, `fp32-fd-warnings` policy v2 now accepts bounded cross-path
and repeated-call discrepancies using the same backend-neutral norm budget.
Both host and resident outputs must independently satisfy the oracle gate before
and after timing; variation checks cannot substitute for oracle checks. Shape,
nonfinite, missing checks, unsupported precision/operations and over-budget
discrepancies still fail. Native/NumPy comparison remains exact. Policy versions
are captured and exported; v1 retains its original strict variation checks.
These sampled checks do not prove absence of small state mutation or establish
an error guarantee for every timed invocation.

Fresh G1 MJX B=16 FD and grad-FD smokes both completed and exported as
`accuracy_warning`, with no failed jobs, in
`results/release-mjx-variation-policy-smoke-20260925/` and
`results/release-mjx-variation-policy-report-20260925/`. Stock MJX code and fp32
arithmetic are unchanged. These are smoke timings, not performance evidence.

`--prepare-only` now populates build/JIT caches at all requested batch sizes.
It performs finite-output checks only, saves `prepared` (not `validated`) cells,
and never writes timing samples. The report refuses preparation captures.
JAX executables are stored in an owner-controlled persistent local cache shared
by preparation and collection. Native timing bridges now reuse a
source/toolchain-keyed cache instead of rebuilding in every capture directory.
Preparation completed sequentially for core, wrappers and the seven extra table
operations in `results/release-prepared-{core,wrappers,table}-20260925/`:
**135 + 120 + 325 = 580 prepared cells**, with no failed jobs. All requested
batch sizes (16, 32, 64, 128, 256) were exercised. The table plan also records
82 explicit unavailable jobs, not preparation failures. All completed manifests
and collector-source fingerprints were verified. Preparation is not full oracle
validation or performance evidence.

Fresh-process verification after preparation:

- G1 MJX FD and grad FD, all five batches: **10/10 retained with accuracy
  warnings**, zero failed jobs, and **10/10 main-executable JAX cache hits**.
  Largest checked reference relative L2 errors were `4.706e-5` (FD) and
  `1.194e-4` (grad FD), below the `1e-3` warning budget.
- iiwa14 native/NumPy/JAX/PyTorch RNEA and grad RNEA, B=16 and 256:
  **16/16 strict passes**, including exact native/NumPy comparisons.
- During wrapper preparation, fresh JAX workers reused **30/30** core
  executables; all 30 native-wrapper preparation cells shared one bridge hash.
- A fresh policy-v2 strict iiwa14 Pinocchio Minv negative control still rejects
  its 12 entrywise exceedances before collecting any timing.
- Focused CPU/website/parser regressions: **133 tests plus two subtests pass**.

Verification captures are `results/release-mjx-variation-cached-smoke-20260925/`,
`results/release-wrapper-cached-smoke-20260925/`, and
`results/release-policy-v2-strict-control-20260925/`. These are smoke/control
data only. No real release timing sweep was started, and no preparatory GPU
job remains running. Later collection must still perform its full accuracy
checks and warmups; use the commands in README on the same checkout/environment
and an otherwise idle GPU. Keep the local caches. Changes remain uncommitted,
including the two submodule fixes noted above.

Artifacts under `test/benchmarks/results/`:

- `grid_mjx_diagnostic_20260925.py` (validation-only reproducer; SHA-256
  `42134d90f0df091695f1c38798a2cc68847deb28a67766dda06df5d833c17e08`).
- `release-mjx-g1-fp32-audit-20260925/` (stock fp32).
- `release-mjx-g1-fp64-audit-20260925/` (stock fp64 diagnostic).
- `release-mjx-g1-fp32-masked-audit-20260925/` (modified sums, diagnostic only).

Each directory includes all 12 outputs per operation, oracle arrays, per-call
checks and a manifest. Run the reproducer with `--dtype float32` or `float64`,
`--output NEW_DIRECTORY`, and optionally `--segment-sum-mode masked`.

The post-fix wrapper capture also exported successfully to
`results/release-investigation-wrappers-parser-fixed-report-20260925/index.html`,
with all 120 rows validated. It remains a watermarked smoke artifact, not
performance evidence.

Latest capture manifest SHA-256 values (paths under `test/benchmarks/results/`):

| Capture | `manifest.json` SHA-256 |
|---|---|
| `release-investigation-pin-core-20260925` | `b25cb7684e30661cf47779b309eca3456c04ebb57d842462262aa258befa485e` |
| `release-investigation-gpu-core-20260925` | `9aad989c9d82714a463d51c5bb324d08e9ab1e4b520bf51fb0d2a949f55a8a7f` |
| `release-investigation-grid-parser-fixed-20260925` | `185b71e81b4da785a0cec079b0b2e586625c2a90622f34d7185f5af6b8121d1a` |
| `release-investigation-wrappers-parser-fixed-20260925` | `e8a7b71ee91c0d852a3847344a8619216b1c1b6344477e7d48b6f47f8594e9b8` |
| `release-table-warning-smoke-20260925` | `1f9c61ea0e26a4731b7b85c7befff612b6ca75f7e3d83113bcb2f87f40d06487` |
| `release-warning-strict-control-20260925` | `e4a9cc9f23c6eedd58086485e587b2aee02a6852b028bc005412a76eb6da3728` |
| `release-mjx-g1-fp32-audit-20260925` | `f873a10f983203e128c4f2264f844cd8f4bbd2cfcf98e8e4f5d0c3c323de4761` |
| `release-mjx-g1-fp64-audit-20260925` | `b166a1ffe88c024dcb055819468b8d8edd4c58ccb8b23d33d5de1d5b7a5b8fcc` |
| `release-mjx-g1-fp32-masked-audit-20260925` | `010823541847a786d18fda0f1b130a2ace282b143337ee7f8a2135f238ae2064` |
| `release-prepared-core-20260925` | `191a55687aa09fd8a589469573a149a0edd23da50ead94e319099e7b9b0bfd1e` |
| `release-prepared-wrappers-20260925` | `667852dcd9865cd8067e86f429741fb5920d7b357d8073a90f8cce8b9501a3e5` |
| `release-prepared-table-20260925` | `4f7b6f7cd28489d0f8666d919dc5751c5d890391929c5448d580e76d146a275e` |

## Fixed (2026-09-25 evening review): three measurement defects, before any collection

Found by the pre-launch review of the collector (three read-only audits over
the collector, the website and the baseline/submodule edits, cross-checked
against the smoke samples). None of the runs had been launched.

- **Pinocchio was timed through a Python `ThreadPoolExecutor`** that switched
  on at B=32: iiwa14 RNEA 10 us at B=16, 234 us at B=32, 968 us at B=256, with
  3x swings inside one cell — the executor, not Pinocchio. Replaced by a
  persistent C++ pool (`release_pool.h`) with an independent context per
  thread and slice 0 on the caller; every cell is now timed at 1, B//16 and
  the ceiling and the best run mean is reported with all variants kept.
  The JIT cache key now hashes the code generators and their settings only
  (`pin_codegen_init.h`), so the existing G1/go2/iiwa14 libraries were reused
  by renaming their directories, not regenerated.
- **MuJoCo Warp was timed eager**: g1 RNEA flat 1.2 ms at every batch size
  and resident slower than full-call at B=16 (Python launch overhead).
  Replaced by captured CUDA graph replay for both boundaries (mujoco_warp's
  own benchmark path); the eager series is kept as `resident_eager`.
- **Five short warm-up calls never leave the idle clock.** `timed()` and both
  native bridges now sustain warm-up calls for `--warm-seconds` (1.5 s) on
  every backend; the value is recorded in `plan.json` and each capture.

Also: gradients validated as two blocks; the native C ABI loop no longer
zero-fills its output inside the timer; new `grid_cuda` backend (GRiD's own
C++ host calls, compute-only and with-memory, bitwise-checked against NumPy)
so the report can decompose GRiD's wrapper costs; CPU/tensor baselines on the
core operations get their own collection command. CPU suite: 62 tests pass.
