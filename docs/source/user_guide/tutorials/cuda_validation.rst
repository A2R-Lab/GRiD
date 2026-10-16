CUDA Validation And Performance Reporting
=========================================

This page summarizes the developer CUDA validation flow for generated GRiD
headers. The checks are staged so long GPU runs provide useful progress and
can be resumed with cached generated artifacts.

``test/run_validation.py`` is the launcher: plan, then ``--execute``, then
commit the generated evidence, then ``--finalize`` (add ``--full`` for a
release receipt). Underneath it, ``test/run_split_suite.py`` is the
crash-isolated driver that shards and runs the suites, and
``test/run_gpu_proof.sh`` is a thin wrapper over that driver for ad-hoc
runs. All three write the receipt that ``test/gpu-proof-receipts.yaml``
assigns to the detected GPU architecture: ``gpu-proof.json`` on the primary
(sm_120), which CI verifies, and a file under ``receipts/`` on a secondary
architecture such as a Jetson. ``make verify-all`` checks every receipt on
disk.

The launcher
------------

Plan without collecting tests or starting any compilation (the ``--out``
directory holds the run's logs and shards; any directory works)::

   .venv/bin/python test/run_validation.py --out validation-runs/next

Add ``--execute`` only on a clean committed candidate in a coordinated quiet
window. The launcher holds ``/tmp/a2rlab-timing.lock`` throughout the run,
records logs, validates fresh/carried header-key provenance, verifies the
merged receipt under the everyday policy (exact test manifest, expected-skip
baseline, fingerprint, signature) and stages the generated receipt and header
manifest. Its ``awaiting-evidence-commit`` state is successful recording, not
a failed GPU run. Commit only those generated evidence files, then verify the
clean descendant without rerunning any tests::

   git add gpu-proof.json test/gpu-proof-header-keys.json
   git commit -m "Record GPU correctness evidence"
   .venv/bin/python test/run_validation.py --finalize validation-runs/next

Finalization rejects altered evidence, dirty trees and source changes since
the recorded candidate. It does not commit files or start GPU work.
``--resume <split-directory>`` reuses completed shards;
choose a new ``--out`` directory for the resumed launch's logs.

For a formal release, add ``--full`` to both the dry plan and execution. This
disables carried shards and selects the strict release policy at finalization.
Resuming a full run also requires ``--full``. Freeze implementation and package
inputs before starting this long run; installed-artifact checks complement,
but do not replace, the source correctness receipt.

Main requires a receipt whose collection equals ``test/gpu-proof-tests.txt``
exactly (the policies' ``required_test_manifest``). A smoke receipt cannot
replace full-scope evidence. Carried shards remain permitted by the everyday
policy; this does not require a fresh release-policy run. After a reviewed
full collection adds, removes or renames tests, regenerate the manifest with
``test/receipt_integrity.py --receipt <run>/gpu-proof.json --write-manifest test/gpu-proof-tests.txt``
and commit it before recording; the manifest diff is reviewed in the PR.

The driver and the thin wrapper
-------------------------------

The launcher above runs the GPU pass through the crash-isolated split driver,
which partitions the ``cuda_equivalence`` and ``python_wrappers`` suites into
bounded shards (one module's abort does not take the rest down) and is
pausable/resumable. ``test/run_gpu_proof.sh`` drives the same driver without
the launcher's lock, logs and provenance checks:

.. code-block:: bash

   # everything, sharded + receipt-signed, every shard fresh (prefer
   # test/run_validation.py above for the locked, logged refresh):
   SPLIT=1 test/run_gpu_proof.sh

   # only the shards whose inputs changed vs the committed receipt:
   SPLIT=1 SPLIT_REFRESH=1 test/run_gpu_proof.sh

   # the driver directly (no receipt), e.g. wrappers only:
   .venv/bin/python test/run_split_suite.py --domains wrappers
   .venv/bin/python test/run_split_suite.py --domains wrappers,cuda --changed-only

``touch <out>/PAUSE`` stops cleanly between shards; ``--resume <out>`` continues
an interrupted run without re-running completed shards. See ``CLAUDE.md`` and
``test/run_split_suite.py --help`` for the full contract (RAM-aware compile
pool, shard bin-packing, receipt merge/carry).

For a quick targeted check, plain pytest still works:

.. code-block:: bash

   .venv/bin/python -m pytest -m cuda_equivalence -q   # CUDA vs numpy oracle
   .venv/bin/python -m pytest -m python_wrappers -q    # jax/torch handles

The ``run_staged_cuda_checks.py`` stage runner is retired; its stages map
onto the shard partition and ``--changed-only``.

GPU-Proof Signed Receipts
-------------------------

Because the CUDA equivalence suite needs a real GPU (and a full cold run is
hours), GRiD records correctness in a signed receipt so a merge can be gated
CPU-only, without re-running GPU tests in CI. This uses the ``pytest-gpu-proof``
plugin, installed from PyPI at the exact version pinned in
``install/requirements-dev.txt`` (CI installs the same pin):

.. code-block:: bash

   .venv/bin/python -m pip install -r install/requirements-dev.txt   # brings in pytest-gpu-proof

Generate the receipt on a quiet GPU box with the launcher (``--execute``).
The receipt's scope is fixed: main accepts only a receipt whose
collection equals ``test/gpu-proof-tests.txt`` exactly (the policies'
``required_test_manifest``). A narrowed run of the wrapper
(``PYTEST_ARGS="-k go2" test/run_gpu_proof.sh``) is a local diagnostic that
exercises the plumbing, never a candidate receipt.

The fp64 tier (``register_robot(..., dtype="float64")``) is an alpha feature
and its tests are outside the receipt by collection: they carry the
``fp64_alpha`` marker and run in their own lane, ``make test-fp64``, which
is not part of the signed receipt. One canary stays in the receipt, the numpy
``iiwa14`` fp64 parity test, so the receipt still shows that the double
``.so`` builds and agrees with the oracle on one robot. Everything else the
receipt validates ran in fp32.

The ``gpu_proof`` marker is auto-applied to every ``cuda_equivalence`` and
``python_wrappers`` item by ``test/conftest.py`` (no per-test annotation), so
receipt membership tracks the existing marker taxonomy. Running the script signs
``gpu-proof.json`` in place with the local SSH key and refuses a dirty tree
(``allow_dirty:false`` in ``test/gpu-proof-policy.yaml``): the fingerprint
hashes the tracked generator, bindings, configuration and GPU-test sources and
pins the peer submodules (GLASS, RBDReference, URDFParser) by gitlink, so a
clean tree makes the receipt's commit SHA a faithful pin of the code under
test.

CI (``.github/workflows/verify-gpu-proof.yml``) verifies whatever receipt is
committed (signature via ``github.com/{signer}.keys``, fingerprint, commit
SHA, freshness, exact test manifest) with no GPU and no secrets, and fails
when the receipt is missing or stale: main always carries current evidence. A
second always-on CPU lane runs the no-GPU tests (descriptor parity, kernel-attr
manifest, plant launch hygiene).

The two-tier receipt policy
---------------------------

The committed ``gpu-proof.json`` goes stale, and CI's verify-receipt job
fails, the moment a push touches a fingerprinted path: the generator
(``grid_codegen/``), the bindings, the peer submodule pins, ``grid_assets/``, the
packaging files, the GPU suites (``test/cuda_equivalents/``,
``test/python_wrappers/``) and the receipt harness itself (the exact list is
``tool.gpu_proof.fingerprint_paths`` in ``pyproject.toml``; documentation and
the website are outside it). The remedy is a refresh of the receipt, not a
revert:

- The everyday policy (``test/gpu-proof-policy.yaml``, ``allow_carried: true``):
  ``SPLIT=1 SPLIT_REFRESH=1 test/run_gpu_proof.sh`` re-runs only the shards
  whose narrow fingerprints changed vs the committed receipt and carries the
  rest; commit the refreshed receipt and CI goes green in minutes to hours
  rather than a full pass. A carried shard attests "these tests, whose files
  are unchanged, passed at an ancestor commit ≤30 days old"; cross-cutting
  codegen/bindings changes are not re-proven by carry.
  ``GRID_SPLIT_REFRESH_DRY=1`` previews the stale/carried plan (clean tree
  required).
- The release policy (``test/gpu-proof-policy-release.yaml``,
  ``allow_carried: false``): refuses carried shards outright; a release
  receipt requires one fresh full ``SPLIT=1`` pass at the release tip.
  Verify with ``--policy test/gpu-proof-policy-release.yaml``.

Maintainer internals: stale shard names are recycled into the fresh partition
automatically (the carry contract has no "superseded" state); a deleted test
module forces a full pass; and both compile caches are content-keyed
(header/source bytes), so byte-identical codegen edits cost seconds of
regeneration, never an nvcc rebuild. Two further carry gates: the per-shard
header-key replay (``test/header_key_replay.py``, records in
``test/gpu-proof-header-keys.json``) byte-precisely re-checks that each
carried shard's generated headers regenerate identically; and the
robot-asset gate (``codegen_neutrality.changed_robot_assets``) demotes
the covering shards when an oracle-side robot asset changes. Every
emission-shaping env var is folded into the header cache key, so that a
variable such as ``GRID_ENABLE_MUJOCO_KERNELS`` cannot change emission
without changing the key and serving stale headers.

How the equivalence tests are designed (the two reference layers,
``RBDReference`` against Pinocchio, then generated CUDA against
``RBDReference``, and the test-design principles) is written up in
``test/TESTING_STRATEGY.md``; a bucket map of everything in ``test/``'s root
is ``test/README.md``. What makes the receipt trustworthy is the policy
described on this page: signature, fingerprint, commit SHA and the exact
manifest.

CUDA Artifact Cache
-------------------

The CUDA equivalence tests cache both generated headers and compiled runners.
This avoids regenerating and recompiling the same robot/configuration between
test stages.

Environment variables:

* ``GRID_CUDA_CACHE_DIR=.grid_build_cache/cuda`` controls the cache root.
* ``GRID_CUDA_DISABLE_CACHE=1`` disables header and runner reuse.
* ``GRID_CUDA_VERBOSE_CACHE=1`` prints cache hit/miss details.
* ``GRID_CUDA_ARCH=120`` overrides architecture detection when needed.

The header cache key includes robot identity, base mode, codegen source hash,
target shared-memory bytes, profile/algorithm selection, and cache schema
version. The runner cache key also includes CUDA architecture, ``nvcc`` version,
L2 mode, compile flags, and runner source hash.

Progress Output
---------------

CUDA equivalence tests print live progress through pytest's terminal reporter.
This keeps long GPU runs from looking stuck under normal pytest capture.

Environment variables:

* ``GRID_CUDA_PROGRESS=0`` disables progress output.
* ``GRID_CUDA_VERBOSE_PROGRESS=1`` enables per-algorithm detail.
* ``GRID_CUDA_VERBOSE_CACHE=1`` includes cache details in progress output.

Sample Selection
----------------

Use ``GRID_CUDA_SAMPLE_NAMES`` to choose deterministic samples:

.. code-block:: bash

   GRID_CUDA_RANDOM_SAMPLES=0 \
   GRID_CUDA_SAMPLE_NAMES=zero,positive,negative,mixed_sign,tiny \
   .venv/bin/python -m pytest test/cuda_equivalents/test_cuda_executable_equivalence.py -q -s

Recognized deterministic sample names include:

* ``zero``
* ``positive``
* ``negative``
* ``mixed_sign``
* ``tiny``
* ``velocity_only``
* ``accel_or_torque_only``
* ``near_limit``
* ``floating_quat_identity``
* ``floating_quat_positive``
* ``floating_quat_mixed``

Set ``GRID_CUDA_SAMPLE_NAMES=all`` to include all deterministic corner samples.
Set ``GRID_CUDA_RANDOM_SAMPLES=N`` to append ``N`` seeded random samples.

Shared-Memory And L2 Controls
-----------------------------

Generated CUDA kernels use target-aware shared-memory tiers. The generator keeps
the most shared-memory-heavy, all-shared implementation when it fits, and only
selects spill tiers when the generated dynamic-shared arena exceeds the target
byte budget. The current default target is ``98304`` bytes (96 KiB), which
matches the opt-in dynamic shared-memory size available on many CUDA GPUs.

Use ``GRID_CUDA_TARGET_SHARED_MEM_BYTES`` at code-generation time to choose a
different target:

.. code-block:: bash

   # More portable target; useful for testing low-shared-memory fallback paths.
   GRID_CUDA_TARGET_SHARED_MEM_BYTES=49152 grid-generate path/to/robot.urdf

   # Current default: prefer all-shared kernels up to 96 KiB.
   GRID_CUDA_TARGET_SHARED_MEM_BYTES=98304 grid-generate path/to/robot.urdf

   # Experimental only: use a larger target if the deployment GPU supports it.
   GRID_CUDA_TARGET_SHARED_MEM_BYTES=120000 grid-generate path/to/robot.urdf

At runtime, generated host initialization checks the actual device's per-block
shared-memory limit, including CUDA's opt-in limit when available, and reports a
clear launch/configuration error if the generated request is too large for the
device.

Useful environment variables for fallback testing:

* ``GRID_CUDA_TARGET_SHARED_MEM_BYTES=10000`` forces low-target fallback paths.
* ``GRID_CUDA_ENABLE_L2_PERSISTING=1`` opts back in to the persisting-L2 hint.
  The generated default is ``0``: a controlled A/B measured the pin never
  helping and hurting by up to 23%. The harness forwards the env var to the
  compile as ``-DGRID_CUDA_ENABLE_L2_PERSISTING=<n>``.

The shared-memory target is a legality and first-pass performance heuristic, not
a full occupancy model. It does not currently account for register pressure,
instruction-cache effects, or topology-specific compiler behavior. For example,
on one robot a 96 KiB all-shared tier may be fastest, while on another robot a
selective-spill tier can win because it uses fewer registers or allows better
occupancy.

Register-Pressure And Tier Analysis
-----------------------------------

For performance work, compare a small set of generated variants on the target
GPU instead of assuming one target is best for every robot/topology. A practical
local sweep is:

1. Generate the same robot/profile with several shared-memory targets, usually
   ``49152``, ``98304``, and optionally the device's opt-in maximum.
2. Compile with ptxas verbosity enabled, for example by adding
   ``-Xptxas=-v`` to the compile command used for the timing binary.
3. Record ptxas output for each kernel: registers per thread, local memory, and
   any compiler-reported spills.
4. Run the generated timing kernels with GPU warmup and internal repeats.
5. Prefer the fastest correct tier on that GPU, while keeping the 96 KiB default
   for generated code that has not been locally tuned.

The non-failing performance reporter can capture ptxas lines and timing
summaries around an existing timing command:

.. code-block:: bash

   .venv/bin/python test/benchmarks/perf_regression_report.py \
     --robot g1 \
     --base-mode floating \
     --profile all \
     --fallback-tier auto \
     --precision float \
     -- ./path/to/generated_timing_binary

If the compile command emits ptxas lines, the report prints a ``ptxas summary``
section. Use this to look for tier choices that trade shared memory for
substantially higher register count or local-memory spills. Save baselines only
from stable, representative GPU machines:

.. code-block:: bash

   .venv/bin/python test/benchmarks/perf_regression_report.py \
     --robot g1 --base-mode floating --profile all \
     --fallback-tier GRID_SPILL_DA_DF_OUTPUT --precision float --save \
     -- ./path/to/generated_timing_binary

The second-order forced-fallback diagnostics
(``test/cuda_equivalents/test_cuda_second_order_fallback.py``) are selected by
their markers only (``cuda_equivalence``, ``developer_only``, ``robot_smoke``);
no environment variable opts them in. Narrow the robot set and the runner's
thread count when reproducing one cell:

.. code-block:: bash

   GRID_CUDA_SECOND_ORDER_SMOKE_ROBOTS=iiwa14 \
   .venv/bin/python -m pytest test/cuda_equivalents/test_cuda_second_order_fallback.py -q -s

Use ``GRID_CUDA_SECOND_ORDER_TEST_THREADS`` to override the diagnostic runner's
thread count.

Linear Algebra Backend
----------------------

Generated headers include a GRiD-owned linear algebra backend adapter
that vendors the SIMT subset of `GLASS
<https://github.com/A2R-Lab/GLASS>`_ (``L1/dot``, ``L2/gemv``,
``L3/gemm``) at codegen time. Generated headers are self-contained, with no
external SDK dependencies. The cuBLASDx-backed ``glass-nvidia`` path was
removed before the first PyPI release (0.5.0); the historical design record is
kept in ``docs/design_history/`` in the repository.

Performance Reporting
---------------------

Performance reports are informational and non-failing. They are intended for a
stable, fast GPU machine after correctness is green.

Run the standard benchmark suite with:

.. code-block:: bash

   .venv/bin/python test/benchmarks/run_benchmarks.py

or run one GRiD benchmark slice directly:

.. code-block:: bash

   .venv/bin/python test/benchmarks/baselines/grid/run.py \
     --robot g1 --base floating

Use ``test/benchmarks/perf_regression_report.py`` for a non-failing
delta report around a timing command or a local JSON baseline. The reporter
records:

* GPU name
* compute capability
* CUDA version
* robot
* fixed/floating base mode
* codegen profile
* fallback tier
* precision

It reports min, median or mean, max, spread, and percent delta from matching
JSON baselines. Use ``--save`` only when the run should become the local
baseline for that GPU class.
