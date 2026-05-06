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

Useful environment variables for fallback testing:

* ``GRID_CUDA_TARGET_SHARED_MEM_BYTES=10000`` forces low-target fallback paths.
* ``GRID_CUDA_ENABLE_L2_PERSISTING=1`` enables the optional persisting-L2 hint.
* ``GRID_CUDA_FORCE_SHARED_TIER=GRID_SPILL_DA_DF_OUTPUT`` forces a gradient
  shared-memory tier when generated support is available.

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

One-shot wrapper:

.. code-block:: bash

   bash GRiDBenchmarks/run_fast_gpu_perf_report.sh \
     /path/to/robot.urdf fixed iiwa14 all auto float --save

   bash GRiDBenchmarks/run_fast_gpu_perf_report.sh \
     /path/to/robot.urdf floating g1 all GRID_SPILL_DA_DF_OUTPUT float --save

The wrapper invokes ``GRiDBenchmarks/perf_regression_report.py`` around the
existing timing binary flow. The reporter records:

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
