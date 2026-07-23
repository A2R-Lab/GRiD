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

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating you are expected to uphold it.
