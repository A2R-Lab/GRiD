# GRiD diagnostics

Empirical resource-and-perf characterization scripts. Not pytest tests —
these are *diagnostics* that drive codegen-time decisions and validate
that the emitted code fits target GPUs.

Each script writes a markdown report into `results/`. Reports under
`results/` are committed when they represent a useful reference baseline
for a specific GPU (filename includes hardware tag, e.g.
`tier_baseline_sm120_rtx5090.md`).

## Scripts

### `tier_baseline.py` — current
Empirical resource matrix for the v2.0 resource-tier design. For each
algorithm × robot config, captures:

- Per-thread register count + spill stores at `__launch_bounds__(MAX_PERF_LEVEL_THREADS)`
  (perf tier) and `__launch_bounds__(1024)` (minimal tier).
- Per-block dynamic shared memory bytes (from the codegen-emitted
  `*_DYNAMIC_SHARED_MEM_BYTES<float>()` constexpr).
- Decision predicate per cell: does it need real lite/minimal downgrade
  work, or can it free-alias to perf?

Drives the codegen's `if smem >= 80KB` and `if algo in heavy_set`
predicates that decide tier emission. Re-run whenever codegen changes
might affect resource footprint.

```bash
PYTHONPATH=. .venv/bin/python test/diagnostics/tier_baseline.py
```

Wall time: ~10-30 min depending on which robots are cached.

### `glass_vs_pre_glass_ptxas.py` — archival
Compares GRiD HEAD (GLASS-vendored SIMT linalg) against the pre-GLASS
reference at commit `d2c0d18`. Used during the 2026-05 GLASS rollout to
isolate per-kernel register/spill regressions. Kept as a template for
future "compare HEAD vs <some-baseline>" investigations.

Requires a sibling worktree at `$GRID_PRE_GLASS_REPO` (defaults to a
sibling directory `../GRiD-A2R-pre-glass/`). See
`test/benchmarks/run_multi_version.py --columns pre_glass` for how to
set up the worktree.

## Results

| File | What |
|---|---|
| `results/tier_baseline_sm120_rtx5090.md` | Tier baseline matrix on RTX 5090 / sm_120 / CUDA 13. Reference data for the v2.0 design. |
| `results/glass_vs_pre_glass_sm120_rtx5090.md` | Pre-vs-post GLASS register / spill diff (2026-05). |

Each results file is gpu-tagged because resource footprints (register
file size, smem cap) are hardware-specific.

## When to add a new diagnostic

Add a `<descriptive-name>.py` script here when you want to characterize
the codegen's output against a hardware constraint or compare two
codegen variants. Conventions:

- Take env-var overrides for output paths (so CI / agents can target
  alternate locations).
- Write to `results/<descriptive-name>.md` by default.
- Print a final one-line summary so the script is useful in a
  background-task pipeline.
- Use `Path(__file__).resolve().parents[2]` for repo root, not
  hardcoded absolute paths.
- Document what hardware your reference data was collected on.

## Relationship to other test/ subdirs

- `test/benchmarks/` — perf benchmarks (wall-time measurements). Tier
  validation lives here, not in diagnostics.
- `test/cuda_equivalents/` — correctness tests against `RBDReference`.
  Strict numerical equivalence.
- `test/python_wrappers/` — pytest suites for `grid_rbd` Python +
  JAX APIs.
- `RBDReference/tests/` — correctness tests against Pinocchio.

Diagnostics sit alongside these as a fourth category: tooling that
informs design decisions, not assertion suites.
