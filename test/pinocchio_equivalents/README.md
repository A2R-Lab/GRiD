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
- Floating-base `rnea` now matches Pinocchio for the floating-enabled smoke
  robots after aligning the floating root velocity/acceleration convention and
  correcting the root gravity transport in `RBDReference`.
- Floating-base `minv` matches Pinocchio for the floating-enabled smoke robots
  under the same native Pinocchio free-flyer ordering now used by GRiD.
- Floating-base `crba` now also matches Pinocchio for the current
  floating-enabled set wherever the resolved source model has an invertible
  mass matrix.
- Floating-base `aba` now also matches Pinocchio for the current
  floating-enabled set wherever the resolved source model has an invertible
  mass matrix.
- Floating-base `forward_dynamics` now also matches Pinocchio for the current
  floating-enabled set: `iiwa14`, `go2`, `g1`, `fr3`, `fetch`, `baxter`, and
  `gen3`, with singular-model skips scoped narrowly where needed.
- Floating-base `rnea_grad` and `forward_dynamics_grad` now also match
  Pinocchio for the current floating-enabled set wherever the resolved source
  model has an invertible mass matrix.
- Floating-base selected pose targets now also match Pinocchio for the current
  floating-enabled set.
- End-effector pose gradients now match Pinocchio on the current fixed-base and
  floating-base default robots using articulated leaf-joint targets shared by
  both sides.
- End-effector pose Hessians now match Pinocchio on a focused `iiwa14`
  fixed-base and floating-base slice, with second-order checks using a
  dedicated finite-difference tolerance policy.
- The second-order top-level helpers `idsva_so` and `fdsva_so` are now covered
  on fixed-base `iiwa14` and floating-base smoke robots. Floating-base coverage
  is validated against Pinocchio's bound C++
  `ComputeRNEASecondOrderDerivatives` (exposed through the `pin_so_ext`
  pybind11 extension), which is the golden second-order oracle rather than
  finite-differencing the first-order paths.
- Floating-base parse, metadata, `rnea`, `minv`, `crba`, `aba`,
  `forward_dynamics`, `rnea_grad`, `forward_dynamics_grad`, and selected pose
  targets are now exercised on the broader floating-enabled set `iiwa14`,
  `go2`, `g1`, `fr3`, `fetch`, `baxter`, `gen3`, and `rizon4`, with
  singular-model skips scoped narrowly where needed.
- Remaining floating-base convention gaps are surfaced explicitly instead of
  being hidden by loose tolerances or ad hoc test logic.

## What This Suite Does Not Validate

- CUDA kernels, generated GPU code, or accelerator paths.
- Every function in `RBDReference`.
- Broad nightly robot corpora in the default developer path.
- Broad end-effector Hessian coverage beyond the focused `iiwa14`
  fixed/floating slice until the current higher-runtime finite-difference path
  is either widened confidently or replaced with a broader analytic
  implementation.

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

The developer install adds `pytest`, `pin`, `robot_descriptions>=1.23.0`, and
`pybind11`, then builds the `pin_so_ext` C++ extension (which wraps Pinocchio's
`ComputeRNEASecondOrderDerivatives` for use as the second-order golden oracle)
and prewarms only the default tier unless `PINOCCHIO_EQUIVALENCE_TIER` is
overridden. Both install scripts create and reuse a repo-local `.venv` so the
workflow does not depend on global `pip` writes.

### pin_so_ext (Pinocchio second-order RNEA binding)

The extension lives at `test/pinocchio_equivalents/pin_so_ext/`. Its loader
(`pin_so_ext/__init__.py`) auto-builds via `setup.py build_ext --inplace` on
first import if the compiled `.so` is missing, so a fresh checkout can be
exercised without manual setup. Prerequisites:

- A C++17 compiler (`g++` ≥ 7 is sufficient).
- `pkg-config` configured for Pinocchio (`pkg-config --cflags --libs pinocchio`
  must succeed).
- Pinocchio development headers, typically installed via the OpenRobotPkg
  binaries that ship under `/opt/openrobots/`.
- `pybind11` (in `requirements-dev.txt`).

To rebuild manually after editing the C++ source:

```bash
cd test/pinocchio_equivalents/pin_so_ext
.venv/bin/python setup.py build_ext --inplace
```

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
