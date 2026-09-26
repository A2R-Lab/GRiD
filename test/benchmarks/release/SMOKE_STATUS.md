# Release collector smoke audit — 2026-09-25

**Historical first-pass audit.** The core Pinocchio failure below has since
been fixed. See [BUG_TRIAGE.md](BUG_TRIAGE.md) for the follow-up investigation,
new captures, and current timing-readiness recommendation.

Implemented locally on `modernizing-tests`, base commit `bf78e43`, with the
working-tree source/build hashes saved in each capture. These are functional
checks on an RTX 5090, **not release performance measurements**. Some diagnostic
jobs overlapped compilation/other smoke work. Do not quote their timing ratios.
Nothing was pushed, merged, or published to the website.

## Result

The collection, numerical validation, provenance, table export, and draft
plotting paths work. **Do not launch the final release sweep yet:** one core
baseline cell and several expanded-table cells fail the current numerical gate.
There were no relaxed tolerances, omitted tensor blocks, or fabricated timings.

| Check | Outcome |
|---|---|
| CPU collector/reporting, timing, picker and website regressions | 49 tests plus two subtests pass |
| Core matrix, all three robots, B=16 | 26/27 pass; Pinocchio G1 grad RNEA fails two entries |
| GRiD RNEA / grad RNEA / Hessian RNEA | All nine robot/operation cells pass |
| Four wrappers × two operations × three robots, B=16 | 24/24 pass |
| Four wrappers × two operations × five batches, iiwa14 | 40/40 pass |
| Pinocchio core, iiwa14 B=16,32,256 | 9/9 pass, including 1/2/8-worker dispatch |
| GRiD FK, FK gradient and FK Hessian | All nine robot/operation cells pass |
| MuJoCo CPU RNEA, Minv, FD, FK | All 12 cells pass, explicitly fp64 |
| MJX RNEA / grad RNEA | All six cells pass with highest-precision fp32 matrix products |
| MuJoCo Warp RNEA / FK | All six cells pass, including corrected endpoint-output kernel |
| BARD RNEA / FD | All six cells pass after quaternion conversion |
| Frax fixed-base RNEA | Pass after supplying gravity explicitly |
| Pinocchio analytical FD Hessian | All three robots pass after fixing the inverse-mass scratch source |

The two earlier failures in Warp's FK adapter and the early iiwa14 IDSVA
algorithm-selection error were fixed and retested. Their original failed logs
remain in the archive; the successful retries supersede those implementation
failures. Earlier compiler-policy experiments do not supersede the final pinned
core result below.

## Core gate that still needs review

Current Pinocchio codegen uses fp32 arithmetic and `-O3` without fast-math.
For G1 grad RNEA, **2 of 39,200 entries** fail `atol=1e-3, rtol=2e-4`:

- Both entries are in sample 4, column 19, output rows 5 and 18 (zero-based).
- Actual: `2.63714599609375`; oracle: `2.6387784857085186`.
- Absolute difference: `0.00163248961477`; allowed difference: `0.00152775569714`.
- Whole-output relative L2 error: `7.3375351e-6`.

This is a narrow accuracy-gate miss, not evidence that Pinocchio lacks the
operation. The historical Pinocchio driver defaults to `-Ofast`; the final
release compiler policy should be reviewed and held fixed across cells, not
selected opportunistically to pass individual tests. Any acceptance-policy
change must likewise be applied consistently and disclosed.

Reproduce just this check in a fresh directory:

```bash
.venv/bin/python -m test.benchmarks.release.collect --stage core \
  --robots g1 --backends pinocchio --operations inverse_dynamics_gradient \
  --smoke --execute --output test/benchmarks/results/recheck-g1-pin-gradient
```

## Expanded-table numerical gates

All listed paths were executed. These are **validation failures**, not N/A or
adapter gaps. Full per-block metrics and, in the newer captures, actual/oracle
arrays are retained. The number below counts entries outside the mixed
absolute/relative tolerance, summed across returned blocks.

| Backend | Robot and operation | Failing entries at B=16 |
|---|---|---:|
| GRiD fp32 | iiwa14 grad FD / FD Hessian | 2 / 404 |
| GRiD fp32 | go2 FD Hessian | 1 |
| GRiD fp32 | G1 grad FD / FD Hessian | 140 / 10,185 |
| Pinocchio codegen fp32 | iiwa14 Minv / FD / grad FD | 14 / 2 / 23 |
| Pinocchio codegen fp32 | go2 grad FD | 1 |
| Pinocchio codegen fp32 | G1 Minv / FD / grad FD | 65 / 9 / 1,189 |
| MJX fp32 | iiwa14 grad FD | 58 |
| MJX fp32 | G1 FD / grad FD | 6 / 1,527 |
| MuJoCo Warp fp32 | G1 FD | 7 |
| Frax fp32 | iiwa14 Minv / FD | 12 / 1 |

Relative-L2 errors are small (roughly `1e-6`–`1e-4` across these blocks), but
that does not establish an entrywise bound or justify automatically loosening
the gate. The next step is numerical triage of these saved outputs and a
reviewed operation-specific accuracy policy, before collecting release timings.
Do not present failed-validation cells as a competitor coverage disadvantage.

## Fixes made during the smoke work

- Matched URDFs, free-base conventions, endpoint selection and gravity across
  adapters; complete selected-output D2H copies inside full-call timers.
- Correct algorithm key for the GRiD IDSVA build; avoid naming one endpoint for
  dynamics-only builds, which caused a metadata/compiled endpoint-count mismatch.
- Correct JAX/PyTorch handle metadata and NumPy/PyTorch output normalization.
- MuJoCo model import handles existing URDF compiler tags; dense-M API call is
  correct for the installed version; Warp endpoint kernel compiles.
- Highest-precision JAX fp32 matrix products; explicit Frax RNEA gravity;
  BARD xyzw→wxyz quaternion conversion.
- Pinocchio analytical FDSVA helper reads its returned `ddq_dtau` inverse mass,
  not stale `data.Minv`; SO scratch tensors are zeroed before reuse.
- Pinocchio codegen Minv is sliced to its defined `nv × nv` block. The installed
  API allocates `nv × nq`; using the full matrix on a free base was incorrect.
- Bounded subprocess timeouts/interruption cleanup, locked content-keyed builds,
  explicit missing/failure states, hash checking, and no negative-overhead clamp.

## Local artifacts

All paths below are relative to the repository root and are gitignored:

- `test/benchmarks/results/release-smoke-core-20260925/`
- `test/benchmarks/results/release-smoke-core-report-20260925/index.html`
- `test/benchmarks/results/release-smoke-wrapper-batches-20260925/`
- `test/benchmarks/results/release-smoke-wrapper-report-20260925/index.html`
- `test/benchmarks/results/release-smoke-secondary-20260925/` retains the
  all-robot wrapper, GRiD table, CPU, simulator, tensor and Pinocchio diagnostics.
- `test/benchmarks/results/release-smoke-grid-table-report-20260925/index.html`
- `test/benchmarks/results/release-smoke-pin-table-report-20260925/index.html`

SHA-256 of the core capture's `manifest.json`:
`98ed818dfe8d0d34337f3dd2512b349011717ebdbbcf0d15aa73d84b7710f948`.
Wrapper five-batch capture manifest:
`07aa424624cdd0465a041a1e69fb6cad1801742456aeceea5db25b19ebf281ee`.

Use [README.md](README.md) for commands and the exact measurement contract.
No smoke plot belongs on the public site. Full B=16–256, three-repeat release
collection, a final-tip GPU validation receipt, review of baseline threading and
compiler policy, and publication approval remain separate gates.
