CUDA Validation And Performance Reporting
=========================================

This page summarizes the developer CUDA validation flow for generated GRiD
headers. The checks are intentionally staged so long GPU runs provide useful
progress and can be resumed with cached generated artifacts.

Staged CUDA Correctness Checks
------------------------------

Run the staged CUDA checker from the repository root:

.. code-block:: bash

   .venv/bin/python test/cuda_equivalents/run_staged_cuda_checks.py

By default this runs the shorter stages:

* ``toolchain``
* ``layout``
* ``fixed``
* ``fixed-fallback``
* ``floating``
* ``floating-stress``
* ``l2``

Select stages explicitly with repeated ``--stage`` flags:

.. code-block:: bash

   .venv/bin/python test/cuda_equivalents/run_staged_cuda_checks.py \
     --stage toolchain \
     --stage layout \
     --stage fixed \
     --stage floating-stress

For an overnight correctness run, include the deterministic corner samples and
seeded random sweep:

.. code-block:: bash

   GRID_CUDA_PROGRESS=1 \
   GRID_CUDA_VERBOSE_CACHE=1 \
   /usr/bin/timeout 10h \
   .venv/bin/python test/cuda_equivalents/run_staged_cuda_checks.py \
     --stage toolchain \
     --stage layout \
     --stage fixed \
     --stage fixed-fallback \
     --stage floating \
     --stage floating-stress \
     --stage l2 \
     --stage corner-samples \
     --stage long-random \
     --timeout-per-command 7200

CUDA Artifact Cache
-------------------

The CUDA equivalence tests cache both generated headers and compiled runners.
This avoids regenerating and recompiling the same robot/configuration between
test stages.

Environment variables:

* ``GRID_CUDA_CACHE_DIR=.pytest_cache/grid_cuda`` controls the cache root.
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
* ``GRID_CUDA_ENABLE_L2_PERSISTING=1`` enables the optional persisting-L2 hint.
* ``GRID_CUDA_FORCE_SHARED_TIER=GRID_SPILL_DA_DF_OUTPUT`` forces a gradient
  shared-memory tier when generated support is available.

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

Second-order diagnostics are opt-in:

.. code-block:: bash

   GRID_CUDA_RUN_SECOND_ORDER_FALLBACK_SMOKE=1 \
   GRID_CUDA_SECOND_ORDER_SMOKE_ROBOT=iiwa14 \
   .venv/bin/python -m pytest test/cuda_equivalents/test_cuda_second_order_fallback.py -q -s

Use ``GRID_CUDA_SECOND_ORDER_TEST_THREADS`` to override the diagnostic runner's
thread count.

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

Use ``test/benchmarks/perf_regression_report.py`` when you want a non-failing
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
