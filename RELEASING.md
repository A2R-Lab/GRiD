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
The artifact workflow additionally builds and checks CPython 3.10, 3.11 and 3.13
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
`release fresh install: PASS` or `release fresh install: FAIL`. It also runs
`tools/check_installed_numerics.py` on supplied iiwa14 and floating-base go2 URDFs
against the bundled FP64 oracle. Keep the work
directory's `SHA256SUMS.txt` and `release_fresh_install.log` with the release
evidence (gate 5). Compilation time depends on cache state; the script downloads `jax[cuda12]` and a
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

### Numerical release matrix

The tiny-robot smoke is necessary but not sufficient. The mandatory
`tools/check_installed_numerics.py --work DIR --urdf-dir DIR` checks real fixed-
and floating-base robots, several batch sizes, dynamics and gradients, NumPy/
JAX/PyTorch values and autograd, pinned outputs and isolated contexts. The fixed-
base case additionally checks dynamics Hessians, pose derivatives and all six
integrators. The same matrix runs in the signed wrapper suite. CUDA tests cover
batched integrator values and gradients directly, including null force pointers.

Every batched operand must vary independently across rows, including external
forces and optional accelerations. Include distinct, repeated and zero values;
compare against per-row CPU reference results, concatenated single calls and
permuted batches. Vary the batch/block relationship for direct CUDA tests.
Identical force rows previously masked a kernel reading row zero for every sample.
Cross-backend agreement alone cannot detect a shared generator bug.

Retain failing probes and their outputs. Fix the cause, add a failing-before/
passing-after regression, and refresh the source receipt; do not label a release
ready because its minimal smoke passed. Neither malformed-input checks nor
sanitizers replace numerical comparisons.

## Freeze and publish

Set the agreed version in `pyproject.toml` before freezing. Run the CPU/docs/artifact
gates, `make release-check`, and focused GPU/sanitizer checks, then commit all
correctness inputs. Coordinate
other projects' quiet timing before starting `test/run_validation.py --full`. After
recording, commit only the generated evidence and use `--finalize` to verify its
signature, complete scope and release policy on the clean descendant. No carried
shards, policy bypass or tolerance relaxation: a release receipt is one fresh full
pass at the release tip, verified under `test/gpu-proof-policy-release.yaml`.

One receipt per architecture. The primary receipt, `gpu-proof.json`, is recorded
on the RTX 5090 (sm_120); it gates CI, defines the collection manifest and is the
only receipt the release workflow verifies. Secondary receipts document other
architectures and live under `receipts/` (`receipts/gpu-proof-sm87.json` for the
Jetson Orin); `test/gpu-proof-receipts.yaml` maps each compute capability to its
receipt path, its header-key ledger and, when it needs one, its own expected-skips
baseline. To record a secondary receipt, install the dev requirements on the
Jetson and run `SPLIT=1 test/run_gpu_proof.sh` on a clean checkout of the release
tip; the driver detects the architecture, writes that row's path and never
touches `gpu-proof.json`. Commit the receipt with its header-key ledger.
`make verify-all` verifies every receipt on disk, one line per architecture, and
`test/receipt_integrity.py --table` prints their rows for the compatibility page.
A release does not wait for secondary receipts; they are documentation of the
other architectures, recorded when those machines are available.

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
2. `build` (CPython 3.10 / 3.11 / 3.12 / 3.13) refuses a tag that does not equal
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
   download that exact version from TestPyPI with `--no-deps`. Match **every**
   downloaded wheel and sdist SHA-256 to that run's artifacts, not a prior run.
   Install the exact wheel in a new virtual environment outside the checkout;
   resolve ordinary dependencies from production PyPI separately (no mixed
   `--extra-index-url` resolution). Run CPU provenance/codegen, `--gpu`, fresh-
   process `--gpu --reload`, and the numerical matrix above. Independently build
   and install the downloaded sdist and check CPU provenance/codegen. Audit all
   wheel RECORD entries and the common bundled payload across Python versions;
   preserve logs, versions and hashes. Local GPU validation of one interpreter
   does not establish GPU coverage for every wheel ABI.
   TestPyPI filenames are immutable: a corrected package needs a new version.
   An upload skipped by `skip-existing` is not evidence that new bytes were tested;
   any hash mismatch blocks the rehearsal even if the workflow is green.
2. Release: push the tag `v<version>` at the commit that carries the fresh release
   receipt. The `receipt` job verifies it, `build` runs, then approve `testpypi`,
   re-check the installed package, and approve `pypi`.
3. Write the GitHub release from `CHANGELOG.md` (the release notes; add the
   `[<version>]:` compare link at its bottom) and attach the `SHA256SUMS` files
   from the build artifacts.

### Published container acceptance

A successful `.github/workflows/docker.yml` build/push does not establish public
pull access or GPU runtime correctness. Before announcing the container:

1. Check the workflow's source revision, version and successful push. First-time
   GHCR packages default to private; a maintainer must explicitly approve public
   visibility in the package settings. Do not work around failed anonymous access
   with a personal token and then call the public release verified.
2. On an exclusively available GPU host with Docker and NVIDIA Container Toolkit,
   run `sudo bash tools/check_container_image.sh 0.5.3` (substitute the released
   version). Omit sudo if Docker is already accessible. No host packages, drivers,
   group permissions or registry settings are changed by this checker.
   It uses an empty Docker CLI configuration for the anonymous pull, resolves
   the tag to a digest, and uses that immutable digest for every container.
3. The checker requires the existing shared `/tmp/a2rlab-timing.lock` and refuses
   a busy lock or detected compute/compiler/test processes. On another host,
   establish an agreed user-owned coordination file first, then pass its path
   through `GRID_CONTAINER_TIMING_LOCK`; never use a different lock to bypass an
   active workload. The lock is opened read-only, but held exclusively, so sudo
   does not need to overwrite a user-owned file in sticky `/tmp`.
4. Acceptance includes GPU visibility, precompiled iiwa14 values versus the
   bundled CPU oracle on all three backends without recompilation, installed
   provenance, new-robot compilation, values/autograd, and a second container's
   cache-only reload with generation/compilation forbidden. Tests run offline
   after the pull, with only the checker scripts mounted read-only and a fresh
   writable scratch directory. The source checkout is not imported.
5. Preserve the printed work directory, `smoke.log`, image digest, versions and
   `PASS` marker. Stop and investigate any failure without weakening checks.
   Report the tested GPU explicitly: a smoke on sm_120 does not validate the
   embedded sm_86/89/90 variants, replace the real-robot numerical matrix, or
   constitute a fresh full source receipt. Conda is a separate unvalidated recipe.

Build-hook behavior follows the [setuptools extension protocol](https://setuptools.pypa.io/en/stable/userguide/extension.html);
the artifact sequence follows the [PyPA packaging flow](https://packaging.python.org/en/latest/flow/).
