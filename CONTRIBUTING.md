# Contributing to GRiD

The contributor guide (code and test expectations, the docs build pipeline,
licensing) is the site page
[Contributing to GRiD](https://a2r-lab.org/GRiD/docs/contribution_guidelines.html)
(source: [`docs/source/contribution_guidelines.rst`](docs/source/contribution_guidelines.rst)).
This file covers only the GitHub-side mechanics.

## Setup

```bash
git clone --recursive https://github.com/A2R-Lab/GRiD.git
cd GRiD
bash install/base_install.sh && source .venv/bin/activate   # single `pip install -e .`
```

Add a backend extra (`pip install -e ".[jax]"` / `".[torch]"` / `".[all]"`) for
the Python-wrapper surface; `bash install/developer_install.sh` adds the
developer tooling (Pinocchio, docs, comparators). Tests, the signed
`gpu-proof.json` receipt and the receipt CI job are documented in the
[CUDA validation guide](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/cuda_validation.html).

## Issues

Open a [GitHub issue](https://github.com/A2R-Lab/GRiD/issues) to report a bug
(include the robot URDF or its name, the backend, the GPU/CUDA/driver versions
and the exact command) or to discuss a substantial change before you start on
it. Algorithm or feature requests are welcome there too, as is a Python
reference implementation in `RBDReference` for any algorithm you would like to
see on the GPU.

## Pull requests

- Keep a PR focused: one change, with its motivation, implementation and
  validation results. The [PR template](.github/PULL_REQUEST_TEMPLATE.md) is
  the checklist reviewers apply.
- Read [`CLAUDE.md`](CLAUDE.md) first. It holds the durable engineering
  conventions (single-block / thread-invariant kernels, byte-identical codegen
  discipline, fix-don't-guard, Pinocchio-authoritative physics) that both AI
  agents and humans follow in this repo.
- Keep the two surfaces in agreement: every algorithm exists as a numpy
  oracle in `RBDReference` (validated against Pinocchio) and as generated CUDA
  (validated against that oracle). Preserve that invariant.
- Keep codegen byte-identical: a refactor that should not change emitted code
  must produce a byte-identical `grid.cuh` (regenerate before/after and
  `diff`). Never land a non-identical diff without a CUDA-equivalence sign-off.
- Respect receipt CI: a push that touches the correctness inputs turns the
  verify-receipt job red until a refreshed `gpu-proof.json` is committed; the
  [CUDA validation guide](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/cuda_validation.html)
  explains the everyday refresh.
- Peer libraries (`GLASS`, `RBDReference`, `URDFParser`) change in their own
  repositories; a GRiD PR then bumps the submodule pin to a commit available
  from those remotes.
- Commits: short single-line messages, no trailer footers; stage paths
  explicitly rather than `git add -A`.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating you are expected to uphold it.
