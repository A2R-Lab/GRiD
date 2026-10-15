# GRiD diagnostics

Empirical resource-and-perf characterization scripts. They are not pytest
tests; they are diagnostics that drive codegen-time decisions and validate
that the emitted code fits target GPUs.

Each script writes a markdown report into `results/`. That directory is
gitignored (`test/diagnostics/.gitignore`), so reports stay local to the box
that produced them; tag the filename with the hardware (e.g.
`tier_baseline_sm120_rtx5090.md`) so a report shared out-of-band is
self-describing.

## Scripts

### `tier_baseline.py` (current)
Empirical resource matrix for the resource-tier design
(`docs/source/user_guide/concepts/resource_tier_system.rst`). For each
algorithm × robot config, it captures:

- Per-thread register count + spill stores at `__launch_bounds__(MAX_PERF_LEVEL_THREADS)`
  (perf tier) and `__launch_bounds__(1024)` (minimal tier).
- Per-block dynamic shared memory bytes (from the codegen-emitted
  `*_DYNAMIC_SHARED_MEM_BYTES<float>()` constexpr).
- Decision predicate per cell: does it need real lite/minimal downgrade
  work, or can it free-alias to perf.

It drives the codegen's `if smem >= 80KB` and `if algo in heavy_set`
predicates that decide tier emission. Re-run it whenever codegen changes
might affect resource footprint.

```bash
PYTHONPATH=. .venv/bin/python test/diagnostics/tier_baseline.py
```

Wall time: ~10-30 min depending on which robots are cached.

### `archive/glass_vs_pre_glass_ptxas.py` (archival)
Compares GRiD HEAD (GLASS-vendored SIMT linalg) against the pre-GLASS
reference at commit `d2c0d18`. It was used during the GLASS rollout to
isolate per-kernel register/spill regressions and is kept as a template for
future "compare HEAD vs <some-baseline>" investigations.

It requires a sibling worktree at `$GRID_PRE_GLASS_REPO` (defaults to a
sibling directory `../GRiD-A2R-pre-glass/`). See
`test/benchmarks/run_multi_version.py --columns pre_glass` for how to
set up the worktree.

## Results (local, not tracked)

Reports produced so far on the development box, under the ignored `results/`:

| File | What |
|---|---|
| `results/tier_baseline_sm120_rtx5090.md` | Tier baseline matrix on RTX 5090 / sm_120 / CUDA 13. Reference data for the tier design. |
| `results/glass_vs_pre_glass_sm120_rtx5090.md` | Pre-vs-post GLASS register / spill diff (2026-05). |

Each results file is gpu-tagged because resource footprints (register
file size, smem cap) are hardware-specific.

## Adding a new diagnostic

Add a `<descriptive-name>.py` script here to characterize the codegen's
output against a hardware constraint or to compare two codegen variants.
Conventions:

- Take env-var overrides for output paths (so CI / agents can target
  alternate locations).
- Write to `results/<descriptive-name>.md` by default.
- Print a final one-line summary so the script is useful in a
  background-task pipeline.
- Use `Path(__file__).resolve().parents[2]` for repo root, not
  hardcoded absolute paths.
- Document what hardware your reference data was collected on.

## Relationship to other test/ subdirs

- `test/benchmarks/`: perf benchmarks (wall-time measurements). Tier
  validation lives there, not in diagnostics.
- `test/cuda_equivalents/`: correctness tests against `RBDReference`.
  Strict numerical equivalence.
- `test/python_wrappers/`: pytest suites for `grid_rbd` Python +
  JAX APIs.
- `external/RBDReference/tests/`: correctness tests against Pinocchio.

Diagnostics sit alongside these as a fourth category: tooling that
informs design decisions, not assertion suites.
