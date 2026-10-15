# Releasing `grid-rbd`

Maintainer playbook: what the package contains, the gates before a release, and how the
tag, TestPyPI and PyPI workflow (`.github/workflows/publish.yml`) runs.

`grid-rbd` is one distribution with NumPy, JAX and PyTorch adapters. Extras
select framework dependencies; they do not install the CUDA Toolkit or driver.
The small `_core` extension is CPU C++; robot CUDA is compiled explicitly by
`register_robot`. A wheel install requires neither a compiler nor GPU.

## Pinned contents

`_build_resources.py` bundles exact clean gitlink versions of URDFParser and
RBDReference Python sources, GLASS headers, launch profiles and peer licenses.
`grid_codegen/_data/provenance.json` records peer commits plus per-file hashes.
The sdist contains these bytes and rebuilds without Git or submodule downloads.
Do not separately install conflicting distributions that own the `URDFParser`
or `RBDReference` import namespaces in the same environment.

`grid_codegen.resources` resolves editable versus bundled resources. The cache
hashes installed generator/parser source and GLASS header contents; it never
assumes a precompiled robot library is shipped in a wheel. Example robot URDFs,
mesh collections, benchmark fixtures and the optional foam checkout are not
part of the package; applications supply their own URDF and referenced assets.

## Build and inspect

Use a clean committed candidate with initialized, clean pinned submodules.
The standard build command builds the wheel from the generated sdist:

```sh
.venv/bin/python -m pip install build auditwheel patchelf twine
.venv/bin/python -m build
PATH="$PWD/.venv/bin:$PATH" .venv/bin/auditwheel show dist/*.whl
PATH="$PWD/.venv/bin:$PATH" .venv/bin/auditwheel repair dist/*.whl --wheel-dir wheelhouse
.venv/bin/python -m twine check dist/*.tar.gz wheelhouse/*.whl
```

Keep native `linux_x86_64` wheels for local diagnostics; upload only wheels whose
portable tags have been verified. The October 2026 CPython 3.12 host build meets
`manylinux_2_34_x86_64` (glibc 2.34 or newer), not an older manylinux baseline.
The artifact workflow additionally builds and checks CPython 3.10 and 3.11
wheels on a CPU runner; do not advertise an untested interpreter/toolkit/
framework combination. There are no Windows,
macOS, Jetson or precompiled robot wheels in this release path.

The release figures under `docs/source/_static/release` are regenerated only from a
reviewed capture by `docs/plot_release_figures.py <report> --approve`, followed by
`docs/export_release_tables.py --approve`; they are never edited by hand. `manifest.json`
lists every file there with its hash and `docs/check_site.py` refuses a site whose assets
differ from it.

## Installed-artifact gates

The fresh-install check is required before freezing. On the lab box, from the
candidate checkout, run

```sh
make release-check          # bash tools/release_fresh_install.sh [--work DIR] [--keep] [--python PATH]
```

It builds the sdist and the wheel from that sdist as above, installs the
repaired wheel into a throwaway virtual environment under a work directory
outside the checkout (no editable install; the checkout is asserted absent
from `sys.path`), runs gates 1 to 3 below, then exercises both README entry
points from outside the checkout: `load_robot` on iiwa14 with
`forward_dynamics` checked against the bundled RBDReference oracle, and
`grid-generate` on that URDF followed by `nvcc` on
`examples/cuda/inverse_dynamics_kernel_example.cu` and
`examples/cuda/validate.py`. It prints the Python, nvcc, jax and torch
versions and the wheel name and SHA-256 it used, and ends with one line,
`release fresh install: PASS` or `release fresh install: FAIL`. Keep the work
directory's `SHA256SUMS.txt` and `release_fresh_install.log` with the release
evidence (gate 5). The script takes minutes and downloads `jax[cuda12]` and a
torch CUDA wheel into the venv. Set `GRID_RBD_BUILD_JOBS=1` to keep one `nvcc`
process at a time when the box is shared.

1. Create a fresh virtual environment, install the wheel, then run from outside
   the checkout with the GPU hidden:
   `python -I /path/to/GRiD/tools/check_installed_package.py --work /tmp/package-check`.
   This verifies imports, all provenance hashes, launch profiles and actual
   generation without Git, nvcc or GPU calls. It must not use an editable install.
   `release-check` runs this step.
2. In the tested GPU/framework environment, install the wheel and run the same
   checker with `--gpu`. It registers a small robot and checks NumPy against
   the bundled CPU oracle, all three backend values, and JAX/PyTorch autograd.
   `release-check` runs this step.
3. Run again in a new process with `--gpu --reload` and the same work directory.
   Generation and compilation are forbidden, so a cache miss fails the gate.
   `release-check` runs this step.
4. For a wheel overlay on an existing framework environment, use
   `--package-root /absolute/unpacked-wheel-directory`. All GRiD/peer imports
   are checked to originate there. This complements, not replaces, the clean
   virtual-environment installation gate.
5. Preserve artifact SHA-256 hashes, build logs, exact versions and installed
   GPU results alongside the fresh full source receipt. Artifact checks do not
   substitute for the full GPU test scope.

## Freeze and publish

Set the agreed version in `pyproject.toml` before freezing. Run the CPU/docs/artifact
gates, `make release-check`, and focused GPU/sanitizer checks, then commit all
correctness inputs. Coordinate
other projects' quiet timing before starting `test/run_validation.py --full`. After
recording, commit only the generated evidence and use `--finalize` to verify its
signature, complete scope and release policy on the clean descendant. No carried
shards, policy bypass or tolerance relaxation: a release receipt is one fresh full
pass at the release tip, verified under `test/gpu-proof-policy-release.yaml`.

Inspect final package provenance and verify the exact frozen artifacts before the
TestPyPI rehearsal and the production upload. Package ownership, publisher setup,
tagging, pushes and uploads need maintainer approval; the artifact workflow
(`packages.yml`) has no publishing credentials and never uploads to a package index.
TestPyPI is a distribution rehearsal, not a replacement for numerical evidence.

### The publish workflow

`.github/workflows/publish.yml` is the only upload path. It uses PyPI trusted
publishing (OIDC), so no token is stored anywhere, and nothing in it runs on an
ordinary push. Its jobs, in order:

1. `receipt` (tags `v*` only; skipped on a `workflow_dispatch` rehearsal) checks
   out the tagged commit with submodules, installs the pinned `pytest-gpu-proof` from
   `install/requirements-dev.txt`, and runs
   `gpu-proof verify --receipt gpu-proof.json --repo . --policy test/gpu-proof-policy-release.yaml --expected-skips test/gpu-proof-expected-skips.txt`.
   A tag whose committed receipt is not a fresh release-policy receipt at that exact
   commit (carried shards, stale fingerprints, missing scope) fails here and nothing is
   built.
2. `build` (CPython 3.10 / 3.11 / 3.12) refuses a tag that does not equal
   `v<pyproject version>`; builds the sdist and the wheel from that sdist under
   `SOURCE_DATE_EPOCH` (reproducible artifacts), `auditwheel repair`s, `twine check`s,
   installs the wheel into a fresh venv outside the checkout with no GPU or build tools
   and runs `tools/check_installed_package.py`; uploads one sdist (the 3.12 job's) plus
   every wheel and their `SHA256SUMS-cp*.txt` as the `release-cp*` artifacts.
3. `testpypi` is gated by the protected `testpypi` GitHub environment (a required
   reviewer approves by hand); it uploads the artifacts to TestPyPI (`skip-existing`).
4. `pypi` (tags only) is gated by the protected `pypi` environment (second manual
   approval); it uploads the same artifacts to PyPI.

One-time set-up:

1. On pypi.org and test.pypi.org, add a *pending* trusted publisher for the
   project name `grid-rbd`: owner `A2R-Lab`, repository `GRiD`, workflow
   `publish.yml`, environment `pypi` (resp. `testpypi`). The project is
   created by its first upload.
2. In the GitHub repository settings create environments `testpypi` and
   `pypi`, each with a required reviewer; restrict `pypi` to tags `v*`.

Per release:

1. Rehearse: run the workflow by hand (`workflow_dispatch`; the receipt gate is
   skipped and only TestPyPI is reachable), approve the `testpypi` environment, then
   install from TestPyPI into a clean virtual environment and run
   `tools/check_installed_package.py --gpu` on a GPU box.
2. Release: push the tag `v<version>` at the commit that carries the fresh release
   receipt. The `receipt` job verifies it, `build` runs, then approve `testpypi`,
   re-check the installed package, and approve `pypi`.
3. Write the GitHub release from `CHANGELOG.md` (the release notes; add the
   `[<version>]:` compare link at its bottom) and attach the `SHA256SUMS` files
   from the build artifacts.

Build-hook behavior follows the [setuptools extension protocol](https://setuptools.pypa.io/en/stable/userguide/extension.html);
the artifact sequence follows the [PyPA packaging flow](https://packaging.python.org/en/latest/flow/).
