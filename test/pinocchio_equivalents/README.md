# Pinocchio Equivalence Suite

This developer-only suite validates GRiD's CPU-side Python reference path against
Pinocchio Python bindings on the same URDF inputs. The goal is to make Pinocchio
the comparison authority for parse behavior, metadata sanity, and core dynamics
algorithms before extending that trust boundary to CUDA and generated GPU code.

## What This Suite Validates

- Robot acquisition is manifest-driven and reproducible.
- The same resolved URDF can be loaded by both GRiD and Pinocchio.
- Fixed-base smoke robots have sane metadata and can be compared numerically.
- Fixed-base `rnea` and `minv` match Pinocchio within central epsilon tolerances.
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

The smoke suite is controlled by `robot_manifest.json` and defaults to:

- `iiwa14`
- `go2`
- `g1`

Robot acquisition uses `robot_descriptions` in v1. The resolver structure also
reserves `example_robot_data`, `direct_git`, and `local_path` for later use, but
those are not part of the default smoke manifest.

## Installation

Base install:

```bash
./base_install.sh
```

Developer install for the Pinocchio equivalence suite:

```bash
./developer_install.sh
```

The developer install adds `pytest`, `pin`, and `robot_descriptions`, then
prewarms only the default smoke-tier assets unless `PINOCCHIO_EQUIVALENCE_TIER`
is overridden. Both install scripts create and reuse a repo-local `.venv` so the
workflow does not depend on global `pip` writes.

## Running The Smoke Suite

```bash
.venv/bin/pytest -m pinocchio_equivalence test/pinocchio_equivalents
```

To focus on fixed-base coverage first:

```bash
.venv/bin/pytest -m "pinocchio_equivalence and not floating_base" test/pinocchio_equivalents
```

## Provenance And Lock Files

- `ROBOT_SOURCE_LOCK.json` is the checked-in source/provenance note for the suite.
- `scripts/fetch_test_models.py` can generate a fresh machine-readable lock under
  `.external_test_assets/robot_source_lock.generated.json`.

If you want to refresh the checked-in lock after verifying dependencies and robot
resolution locally, run:

```bash
python3 scripts/fetch_test_models.py --update-checked-in-lock
```

## Adding A New Robot Safely

1. Add a new manifest entry with robot id, embodiment, source kind, source
   descriptor, tier, and base modes.
2. Resolve it with `scripts/fetch_test_models.py`.
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
