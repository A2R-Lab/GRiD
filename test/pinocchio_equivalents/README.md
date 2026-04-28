# Pinocchio Equivalence Suite

This developer-only suite validates GRiD's CPU-side Python reference path against
Pinocchio Python bindings on the same URDF inputs. The goal is to make Pinocchio
the comparison authority for parse behavior, metadata sanity, and core dynamics
algorithms before extending that trust boundary to CUDA and generated GPU code.

## What This Suite Validates

- Robot acquisition is manifest-driven and reproducible.
- The same resolved URDF can be loaded by both GRiD and Pinocchio.
- Fixed-base default-tier robots have sane metadata and can be compared numerically.
- Fixed-base `rnea` and `minv` match Pinocchio within central epsilon tolerances.
- Fixed-base default-tier robots also exercise `crba`, `aba`,
  `forward_dynamics`, `forward_dynamics_grad`, `rnea_grad`, and selected pose
  targets against Pinocchio when the resolved robot parses cleanly on both sides.
- The current fixed-base verified set includes `iiwa14`, `go2`, `g1`, `fetch`,
  `baxter`, `fr3`, and `gen3`, while `rizon4` remains included with explicit
  skips for inverse-mass and ABA-style checks because the resolved source model
  is singular on both the GRiD and Pinocchio sides.
- Floating-base parse and metadata coverage exists for the same smoke robots.
- Floating-base convention gaps are surfaced explicitly instead of being hidden by
  loose tolerances or ad hoc test logic.

## What This Suite Does Not Validate

- CUDA kernels, generated GPU code, or accelerator paths.
- Every function in `RBDReference`.
- Broad nightly robot corpora in the default developer path.
- Floating-base numerical equivalence before free-flyer convention alignment is
  verified well enough to trust the comparison.

## Robot Sourcing

The default suite is controlled by `robot_manifest.json` and currently includes:

- `iiwa14`
- `go2`
- `g1`
- `fr3`
- `rizon4`
- `gen3`
- `fetch`
- `baxter`

Robot acquisition is `robot_descriptions`-first in v1. Each manifest entry can
carry a source-candidate chain, so the default developer flow tries
`robot_descriptions` first and leaves room for later `direct_git`,
`example_robot_data`, or `local_path` fallbacks when upstream coverage is missing.

## Installation

Base install:

```bash
./base_install.sh
```

Developer install for the Pinocchio equivalence suite:

```bash
./developer_install.sh
```

The developer install adds `pytest`, `pin`, and `robot_descriptions>=1.23.0`,
then prewarms only the default tier unless `PINOCCHIO_EQUIVALENCE_TIER` is
overridden. Both install scripts create and reuse a repo-local `.venv` so the
workflow does not depend on global `pip` writes.

Top-level runner and suite entrypoints:

- `test/run_tests.py` is the main command-line entrypoint for listing models,
  preparing assets, and running the suite.
- `test/pinocchio_equivalents/test_all.py` is the suite-level pytest target used
  by the top-level runner.

## Running The Default Suite

```bash
.venv/bin/python test/run_tests.py
```

To focus on fixed-base coverage first:

```bash
.venv/bin/python test/run_tests.py -- -m "pinocchio_equivalence and not floating_base"
```

To list the manifest-controlled default robots:

```bash
.venv/bin/python test/run_tests.py --list-tests
```

## Provenance And Lock Files

- `ROBOT_SOURCE_LOCK.json` is the checked-in source/provenance note for the suite.
- `test/run_tests.py --prepare-models` can generate a fresh machine-readable lock under
  `.external_test_assets/robot_source_lock.generated.json`.

If you want to refresh the checked-in lock after verifying dependencies and robot
resolution locally, run:

```bash
.venv/bin/python test/run_tests.py --prepare-models --update-lock
```

## Adding A New Robot Safely

1. Add a new manifest entry with robot id, embodiment, source kind, source
   descriptor, tier, and base modes.
2. Resolve it with `test/run_tests.py --prepare-models`.
3. Refresh the source lock.
4. Run parse and metadata coverage first.
5. Only add numerical coverage for algorithms whose mapping to Pinocchio is clear.

## Interpreting Failures

- Parse failures mean GRiD or Pinocchio could not load the resolved URDF cleanly.
  The failure message should identify the robot id, source kind, URDF path, and
  base mode.
- Metadata failures usually indicate a naming, indexing, or model-structure
  mismatch that should be triaged before trusting numerical results.
- Numerical mismatches indicate either an algorithm bug or a remaining convention
  mismatch. Check `PINOCCHIO_MAPPING_INVENTORY.md` and
  `PINOCCHIO_ALIGNMENT_BACKLOG.md` before loosening tolerances.
