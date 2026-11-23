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
   The GPU pair also checks the optional armature identification API against the
   bundled double-precision oracle on NumPy, JAX and PyTorch, including the
   regressor's acceleration derivative. The first process mutates the live
   armature and hides the original URDF. The second forbids parsing and compiler
   lookup and must restore the construction initializer, not the last mutation.
   Keep both logs. Running only the construction half does not close this gate.
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

## Branch stages

`dev` collects reviewed pull requests and requires the CPU lane, the packaging
jobs and the docs build. `main` moves only when a maintainer merges `dev`
(or a `hotfix/` or `release/` branch) with a merge commit, after the
stale-shard refresh on the primary GPU; the everyday policy applies and
carried shards are allowed. PyPI moves from a tag on `main` under the release
policy below. `dev` may be ahead of `main`, and `main` ahead of PyPI.

To promote `dev`: check out `dev`, run `make receipt-refresh`, commit the
refreshed `gpu-proof.json` and its header-key ledger on `dev`, open the pull
request from `dev` into `main`, and merge it with a merge commit once the
receipt job is green. A hotfix branches from `main`, receives the same
refresh, merges into `main` and is then merged into `dev`.

A secondary receipt under `receipts/` attests the commit it was recorded at.
CI verifies it against that commit and does not require it to match the tip;
the compatibility page states which source each architecture's latest receipt
covers.

### Hardware receipt tiers

`test/gpu-proof-receipts.yaml` is the policy inventory. `make verify-all`
checks each existing receipt at its recorded ancestor, including signatures,
scope, skips and clean-source fingerprints. Historical age is evaluated at
recording; history is not fresh evidence for changed source.
Historical verification uses the current everyday policy against the recorded
source, not a snapshot of that policy from the recording date. Future policy
changes can therefore reject historical receipts. Review this effect explicitly
when changing the everyday policy; do not silently reinterpret past evidence.

Before a stable X.Y.Z release, run
`.venv/bin/python test/receipt_integrity.py --release X.Y.Z`. Publication runs
this same check. Primary evidence must be fresh and full for every release;
minor-tier evidence for X.Y.0; major-tier evidence for X.0.0. Freshness uses
the current release policy with no carried shards. A missing required receipt
fails. The current rows are RTX 5090 (primary), Orin (minor), and Xavier
(major, not yet recorded).

A reviewed secondary waiver is an explicit `waivers` entry with `version`
and a nonempty `reason`; it applies only to that version. Primary waivers are
forbidden. A waiver does not excuse invalid evidence already on disk.
Retirement requires `tier: retired`, `retired_in: X.Y.0` and a changelog entry;
retain the receipt as history. Never use retirement retroactively.

Regenerate the compatibility policy inventory with
`test/receipt_integrity.py --tier-table` after changing rows or receipts.
The generated source version is read from the recorded commit, not inferred
from the latest release tag. These checks do not approve a hardware campaign.

### Smaller validation units

Native shards no longer combine test modules. Integrator, FD-gradient output
spill and kinematics-invariance modules also partition by robot/base, keeping
their compilation groups and thread/batch variants together. Scope and
tolerances are unchanged. Refreshes shadow all stale shard names and may add
new names; carried names are never overwritten. An indivisible expensive
test remains expensive: partitioning improves stale-run granularity and
failure recovery, not the algorithm's runtime or the total work by itself.
The default target is 30 estimated minutes, not a hard deadline. A fresh clone
seeds scheduling durations from the committed receipt; valid local rolling
measurements override them. These costs authorize neither evidence reuse nor
lower memory admission. Indivisible compilation atoms can exceed the target.
The 30-minute target is provisional until a refresh measures the overhead of
the extra sessions. Recycled stale shard names are merge identities, not current
scope descriptions. The planner logs their mapping to the proposed partitions;
the explicit node IDs remain authoritative. When the new plan has fewer shards
than stale names, it reports the name-count-constrained fallback. A fresh full
run is needed to adopt fewer identities under the current merge contract.

### Trusted contributor evidence CI

`contributor-evidence.yml` uses `pull_request_target` only to read receipts:
it executes verifier code, dependencies and policies from the current tip of
the PR's base branch in the base repository and
never installs or imports the candidate. Both checkouts have credentials
disabled; permissions are read-only. The checked-out immutable SHA is printed
in the run log and job summary. The PR event's `base.sha` is not the policy
source: a long-lived PR must receive later verifier hardening. Each run keeps
its checked-out tree fixed even if the branch advances during verification.
The base authorizes the fixed smoke,
optional small-robot groups and fingerprint paths. Stage the verifier and
policy on `dev` before relying on this new base-owned workflow; a workflow
introduced only in a PR cannot attest its own implementation. Changes to
branch protection or GitHub settings require a separate maintainer action.

Contributor recording reserves 1 GiB for the exact four mandatory smoke IDs:
twice the rounded-up 0.39 GiB cold sampled process-tree peak, plus 0.22 GiB
absolute headroom. The measurement used a 64 GiB host, RTX 5090, CUDA 13.2 and
one compiler unit on November 15; see
`docs/design_history/low_memory_validation_plan_2026-11-15.md` for provenance.
The provisional lane uses a 2 GiB reserve, making smoke require 3 GiB available;
the target of 8 GiB available therefore has 5 GiB additional admission headroom.
The full lab lane retains its 14 GiB reserve. Unmeasured optional small-native
groups retain 8 GiB times 1.3 (12.4 GiB total in the contributor lane); unknown
groups retain 16 GiB times 1.3 (22.8 GiB total). On a constrained machine the
mandatory smoke can finish while optional groups are refused cleanly.
Changing the exact mandatory ID set does not inherit the measured allowance.
These are estimates, not memory caps or promises for every compiler/GPU. A
smaller machine does not inherently increase a process's RAM consumption;
live available RAM determines admission. Contributor admission uses the smaller
of host MemAvailable and remaining cgroup v1/v2 memory headroom at every visible
ancestor in the memory controller's own hierarchy (v1 on hybrid hosts when
memory is attached there). Undelegated v2 leaves still inherit the visible
ancestor's accounting. `memory.high` is a conservative admission ceiling to
avoid reclaim and throttling, not a hard kernel limit. Cache counts as used, swap is not extra
capacity, and unreadable accounting refuses admission. Limits outside a mounted
cgroup namespace cannot be observed; these estimates are not isolation.
The whole pytest process tree
is sampled, including inline compilers. Pressure drains the current unit and
prevents further admission; it is not a hard RSS limit. The 2 GiB reserve is
policy headroom for a contributor workstation, not a measured OS requirement.

The optional-evidence verifier loads its trusted sibling directly, without
importing the robotics test package or recording driver. It needs only the
standard library, PyYAML and gpu-proof in the base checkout, with no submodules.
Symlink receipts (including dangling links) fail closed. Local isolated tests
exercise trusted dispatch, not cryptographic acceptance of a signed fixture.
Claude's November 18 scratch-repository rerun exercised same-repository PRs,
not forks: forged full scope and both symlink forms fail for the intended
reasons; control and planted-code cases use only trusted code. A genuine signed
four-test provisional receipt also passed the latest base-tip workflow on
November 18. These were same-repository exercises; a real fork exercise is
still pending. The private scratch repository has since been deleted.

Optional evidence has a lifecycle: CI inspects the Git object type first and
rejects symlinks, even inherited ones. A regular-file `receipts/provisional.json`
whose blob is identical to the current trusted base's blob is inherited and
neutral, including on source-changing descendants. It makes no new validation
claim. New or changed evidence is verified in full, including the signer,
signature, scope and the unchanged-source-since-recording rule.

## Freeze and publish

CUDA partitions now persist `partition-map.json` schema 1 alongside
`partition.json`. Each row separates the immutable receipt merge identity from
the planned display label, records exact node IDs and indivisible compile atoms,
and records predecessor identities with their old node sets. The union is the
exact fresh CUDA scope; nodes and atoms cannot overlap shards. Resume validates
both artifacts and replays the saved partition without repacking it from newer
duration estimates. Older outputs without this map require a fresh full run.
This is an audit map, not a cross-receipt merge protocol. If the budget plan has
fewer shards than stale identities, retain the complete name-count-constrained
partition and record its fallback labels and predecessors. Fewer fresh atoms
than stale identities still requires a fresh full pass. Never relabel a carried
receipt to make a smaller partition fit.

Strip optional `receipts/provisional.json` when promoting `dev` to `main`, before
the release freeze. It is contributor evidence, never release evidence; do not
copy it into the release branch. Review this removal explicitly in the promotion
diff. Historical primary and secondary receipts remain subject to their policies.

Set the agreed version in `pyproject.toml` before freezing. Run the CPU/docs/artifact
gates, `make release-check`, and focused GPU/sanitizer checks, then commit all
correctness inputs. Coordinate
other projects' quiet timing before starting `test/run_validation.py --full`. After
recording, commit only the generated evidence and use `--finalize` to verify its
signature, complete scope and release policy on the clean descendant. No carried
shards, policy bypass or tolerance relaxation: a release receipt is one fresh full
pass at the release tip, verified under `test/gpu-proof-policy-release.yaml`.

Before that freeze commit, plan the manifest from authoritative collection:

```bash
.venv/bin/python test/run_split_suite.py --plan-manifest test/gpu-proof-tests.txt
```

This does not compile, execute tests or record evidence. It uses the split
driver's wrapper modules and CUDA partition, with the same `gpu_proof` marker
selection as the recording. Review additions and removals, then commit the
manifest with the correctness inputs. The full launcher checks canonical
collection equality before starting builds; any partial or duplicate collection
is an error. The empty expected-skips baseline stays empty. FP64 alpha tests
remain in their separate lane, not as skips in the receipt. This one-pass
protocol supersedes the October 26 two-pass manifest procedure; the receipt
writer and collection writer produce identical manifest bytes for identical IDs.
The old receipt is stale during this transition. CPU pre-freeze tests exercise
exact-scope rejection with controlled missing, extra and duplicate outcomes;
the launcher applies that same check to the completed receipt before promotion
and again at finalization. Both external verification policies still require
the current manifest. No old receipt is accepted for the expanded scope.

### Primary receipt resource profile

The tracked launcher defaults to `GRID_SPLIT_RESOURCE_PROFILE=serial`: one prewarm
worker, then one GPU shard at a time, with two compiler units per wrapper build.
The complete prewarm phase drains before any shard starts. A cap on prewarm
workers alone does not bound memory: tests can compile additional wrappers and
temporary executables inline. Serial phases prevent that overlap without
omitting tests, changing numerical thresholds or changing compiled algorithms.
Serialization is between workers and shards, not inside a test: the consumer
artifact concurrency regression deliberately starts two builds. The whole-shard
allowance includes descendants; the compiler-unit cap applies per build, not
globally across those descendants.

Every prewarm admission, including the first, requires its predicted process-tree
peak plus a 14 GiB host reserve to fit the live `MemAvailable` and pool budget.
The allowance is the larger of identity-qualified measured peaks and the
class floor in `test/resource_history.py`, before a 1.3 margin. With at most
two compiler units, measured native classes have a 6 GiB floor, ordinary
wrappers 8 GiB, and full floating/MJX wrappers 12 GiB. Unknown recipes,
unpiloted robots, humanoid wrappers and wider compiler builds retain 16 GiB.
Whole shards reserve 8 GiB, 12 GiB for default armature registration, or
16 GiB for the unpiloted large-model classes because they may compile inline. These are
planning allowances, not enforced process RSS limits: monitor actual peaks and
other workloads, and do not assume an unmeasured model is bounded by the default.
The primary launcher's profile overrides inherited concurrency settings.
Direct split-driver runs retain the older overlap profile unless explicitly
configured; the secondary/Jetson recipe below is separate.

Both tracked profiles disable JAX GPU preallocation and record that setting,
matching the correctness gates and pilot when JAX and PyTorch share a shard.
The optional tracked `--resource-profile bounded-overlap` selects ceilings of
two pool builds, one GPU shard and two compiler units per build. One atomic
admission authority reserves memory for both the pool and the entire shard,
including inline compilers. Only the unspent part of a reservation is deducted
from live available memory: resident pages have already reduced that value.
Both profiles use those same class floors and retain larger matching measured
peaks. Resource history is keyed by source bytes, recipe, toolchain,
architecture and build options; cache-hit samples cannot lower a cold floor.
It predicts resources only, never reuses test outcomes. The 14 GiB host reserve is unchanged. Sampled
process-tree peaks and headroom are logged in `memory-admission.jsonl`; short
peaks between samples can still be missed. Admission is not a hard RSS limit.
Pressure stops new work and drains owned compilers rather than killing peers.

Three launcher options widen or narrow bounded overlap for the machine at hand.
They are accepted only with `--resource-profile bounded-overlap`; with the
serial profile they are an error, and without them bounded overlap keeps its
two pool builds and two compiler units.

- `--compile-slots N` sets the number of pool builds that may run at once
  (`GRID_SPLIT_COMPILE_JOBS`).
- `--build-jobs N` sets the compiler units per wrapper build
  (`GRID_RBD_BUILD_JOBS`). Above two units every pool build is charged the
  16 GiB class floor, 20.8 GiB with the margin, because the smaller floors were
  measured at two units. Wider builds therefore admit fewer jobs at once.
- `--memory-cap GIB` sets a ceiling on the memory allowances held at once by
  pool builds and the shard together (`GRID_SPLIT_MEMORY_CAP_GIB`). The pool
  budget becomes the smaller of its usual value and the cap, and the admission
  authority refuses a reservation that would take the held allowances past the
  cap. The cap never raises the budget and does not change the 14 GiB reserve
  or the live `MemAvailable` check. The launcher rejects a cap below 20.8 GiB,
  the largest single allowance, since such a run could only pause.

Slots are a ceiling, not a promise: admission still decides how many builds
run, so extra slots are used only when their allowances fit. The three values
are recorded in `state.json` with the profile, and a resume must repeat them
exactly. The one GPU shard at a time is not configurable.

`test/run_validation.py --print-resources` reports what the current machine
offers: cores, `MemTotal`, `MemAvailable`, the reserve, the resulting pool
budget, how many largest-class allowances fit now, the largest compile peak in
the resource history ledger when one exists, and the slots and units of both
profiles. Add `--resource-profile bounded-overlap --memory-cap GIB` to see the
capped budget. It reads `/proc/meminfo` and the ledger only. It starts no
compiler, uses no GPU and takes no lock, and the figures describe that moment,
not the memory available hours into a run.

Select bounded overlap only after a cold-build, inline-cache-miss pilot on the
same scope as serial demonstrates safe, useful throughput. This is validation
throughput, not runtime performance evidence. A small pilot cannot certify the
largest receipt models; those must still fit admission or stop. Any setting
wider than the default two slots and two units needs that same pilot at that
setting on the machine that will record. The resource
profile does not change collection, numerical checks or expected skips.
Resume requires the recorded profile. To fall back, let the old run exit and
use a new output directory with `--full --resource-profile serial
--fallback-from <old-launcher-directory>`. This records the transition and runs
the full scope without carrying outcomes; the source must still match exactly.
Never edit a frozen harness to obtain fallback.

Insufficient admission writes `resource-blocked.json` and `PAUSE` in the split
run directory and returns nonzero without merging receipts. Completed shard
results remain intact. After inspecting memory pressure, remove that run's
`PAUSE` marker and resume at the same clean commit with the same full/refresh
mode. A manual PAUSE during prewarm stops queued builds and lets the current
build finish before returning; resume reuses its content cache. A paused or
unattempted shard is never reported as passing. Estimate both cold and warm
passes from the intended cache and worker configuration, not a past run's
elapsed time. Receipt recording still requires its separate approval.

### Architecture receipts

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
The signing key stays on the desktop, so a Jetson records with
`PYTEST_ARGS="--gpu-proof-signing-backend none"`. The driver merges the unsigned
shards into an unsigned run-directory receipt, verifies it under the everyday
policy with `--allow-unsigned` (scope, expected skips, fingerprint, ancestry),
writes the durations and the header-key ledger at the row's path, and the
launcher refuses to publish the unsigned receipt. Copy the run directory's
`receipts/` and the ledger to the desktop, branch from the release tip, merge and
sign there with `gpu-proof merge --out receipts/gpu-proof-sm87.json --repo .
<shard receipts>`, commit the receipt with the ledger, then run `make
verify-all`: the policy refuses a dirty verification tree, and the receipt
itself counts as dirt until it is committed. CI's verify job runs the same
`--verify-all` over every committed secondary receipt.
`make verify-all` verifies every receipt on disk, one line per architecture, and
`test/receipt_integrity.py --table` prints their rows for the compatibility page.
A release does not wait for secondary receipts; they are documentation of the
other architectures, recorded when those machines are available.

Jetson resource budget for a secondary receipt (Orin, 64 GB, twelve cores). The
10-24 attempt rebooted the box out of memory inside the compile pool (guide
7.z63), so launch with the budget stated, not the defaults: `GRID_SPLIT_COMPILE_JOBS=2`
(two full wrapper builds fit with room to spare; the default of five does not),
`GRID_SPLIT_SHARD_JOBS=1` (one GPU shard at a time, the default), and
`GRID_RBD_BUILD_JOBS=4` per wrapper build (the default is `min(8, cores,
MemAvailable / 3 GB)`, which with two concurrent humanoid builds at 12 to 15 GB
each exceeds the box). Disk preflight before launch: the pass writes wrapper
objects to `~/.cache/grid-rbd` and runner executables to `.grid_build_cache/cuda`
in the checkout (37 GB and 47 GB respectively on the 5090 box after a month of
runs); check `df -h` on both volumes, clear both caches if the volume is under
30 GB free, and do not start at the 6 GB free the 10-24 launch had. Set
`GRID_SPLIT_STALL_SECS=7200` so a long humanoid compile is not mistaken for a
hang, and `GRID_CUDA_ARCH=87` only if a test's architecture probe reports
otherwise. Everything else follows the shell path above unchanged.

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
   run `sudo bash tools/check_container_image.sh 0.7.0` (substitute the released
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
