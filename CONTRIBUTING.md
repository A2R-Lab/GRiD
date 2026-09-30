# Contributing to GRiD

Thanks for your interest in contributing! GRiD is a GPU-accelerated rigid body
dynamics library that **generates** per-robot CUDA C++ from a URDF.

## Getting set up

```bash
git clone --recursive https://github.com/A2R-Lab/GRiD.git
cd GRiD
bash install/base_install.sh && source .venv/bin/activate   # single `pip install -e .`
```

Add a backend extra (`pip install -e ".[jax]"` / `".[torch]"` / `".[all]"`) for
the Python-wrapper surface, or `".[dev]"` to run the bindings' tests. Developer
tooling (Pinocchio, docs, comparators) installs via
`bash install/developer_install.sh`.

## Before you open a PR

- **Read [`CLAUDE.md`](CLAUDE.md)** — it documents the durable engineering
  conventions (single-block/thread-invariant kernels, byte-identical codegen
  discipline, fix-don't-guard, Pinocchio-authoritative physics).
- **Keep the two surfaces in agreement.** Every algorithm exists as a numpy
  oracle in `RBDReference` (validated against Pinocchio) and the generated CUDA
  (validated against that oracle). Preserve that invariant.
- **Byte-identical codegen:** a refactor that shouldn't change emitted code must
  produce a byte-identical `grid.cuh` (regenerate before/after and `diff`).
  Never land a non-identical diff without a CUDA-equivalence sign-off.
- **Run the tests** (`.venv/bin/python -m pytest -q`; GPU markers need a GPU —
  see the [testing guide](docs/source/user_guide/tutorials/cuda_validation.rst)).
  GPU outcomes are captured in a signed `gpu-proof.json` receipt so CPU-only CI
  can verify them.
- Full contributor guidelines (style, PRs, docs): see
  [`docs/source/contribution_guidelines.rst`](docs/source/contribution_guidelines.rst)
  (published at https://a2r-lab.github.io/GRiD/).

## Developer Testing

The Pinocchio-side floating convention regression suite exercises both public
floating-base orderings across the current floating robot manifest:

```bash
.venv/bin/python -m pytest external/RBDReference/tests/test_floating_base_conventions.py -q
```

The CUDA executable equivalence suite defaults to the Pinocchio-facing
floating convention and is an established suite with broad floating-base
algorithm coverage (40+ equivalence modules across the robot manifest).

For floating CUDA development, the pytest harness also accepts optional env
overrides:

+ `GRID_CUDA_FLOATING_ALGORITHMS=all` to try the broader floating candidate set
+ `GRID_CUDA_FLOATING_ALGORITHMS=inverse_dynamics,forward_dynamics` to request a subset
+ `GRID_CUDA_FLOATING_SAMPLE_NAMES=all` to run every deterministic/random sample instead of only `zero`

Generated CUDA defaults to a 96 KiB dynamic shared-memory target
(`GRID_CUDA_TARGET_SHARED_MEM_BYTES=98304`) and selects spill fallbacks only
when the generated arena would exceed that target. See
[CUDA validation guide](https://a2r-lab.github.io/GRiD/user_guide/tutorials/cuda_validation.html)
for shared-memory target overrides, L2 controls, and the recommended
ptxas/register-pressure analysis workflow for tuning a specific robot/GPU.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating you are expected to uphold it.
