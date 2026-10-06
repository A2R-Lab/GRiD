# Distribution and release checks

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
The artifact workflow additionally checks CPython 3.11; do not advertise an
untested interpreter/toolkit/framework combination. There are no Windows,
macOS, Jetson or precompiled robot wheels in this release path.

## Installed-artifact gates

1. Create a fresh virtual environment, install the wheel, then run from outside
   the checkout with the GPU hidden:
   `python -I /path/to/GRiD/tools/check_installed_package.py --work /tmp/package-check`.
   This verifies imports, all provenance hashes, launch profiles and actual
   generation without Git, nvcc or GPU calls. It must not use an editable install.
2. In the tested GPU/framework environment, install the wheel and run the same
   checker with `--gpu`. It registers a small robot and checks NumPy against
   the bundled CPU oracle, all three backend values, and JAX/PyTorch autograd.
3. Run again in a new process with `--gpu --reload` and the same work directory.
   Generation and compilation are forbidden, so a cache miss fails the gate.
4. For a wheel overlay on an existing framework environment, use
   `--package-root /absolute/unpacked-wheel-directory`. All GRiD/peer imports
   are checked to originate there. This complements, not replaces, the clean
   virtual-environment installation gate.
5. Preserve artifact SHA-256 hashes, build logs, exact versions and installed
   GPU results alongside the fresh full source receipt. Artifact checks do not
   substitute for the full GPU test scope.

## Freeze and publish

Set the agreed version before freezing. Run the CPU/docs/artifact gates and
focused GPU/sanitizer checks, then commit all correctness inputs. Coordinate
other projects' quiet timing before starting `test/run_validation.py --full`.
After recording, commit only the generated evidence and use `--finalize` to
verify its signature, complete scope and release policy on the clean descendant.
No carried shards, policy bypass or tolerance relaxation.

Inspect final package provenance and verify the exact frozen artifacts before
TestPyPI rehearsal and production upload. Package ownership, publisher setup,
tagging, pushes and uploads need maintainer approval; the artifact workflow
has no publishing credentials and never uploads to a package index. TestPyPI
is a distribution rehearsal, not a replacement for numerical evidence.

Build-hook behavior follows the [setuptools extension protocol](https://setuptools.pypa.io/en/stable/userguide/extension.html);
the artifact sequence follows the [PyPA packaging flow](https://packaging.python.org/en/latest/flow/).
