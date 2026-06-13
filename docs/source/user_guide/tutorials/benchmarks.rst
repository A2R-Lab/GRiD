Benchmarks
==========

GRiD ships a benchmark harness that compares GRiD against CPU and GPU
baselines (Pinocchio, MJX, Frax, BARD) and against historical GRiD
reference points. BARD (PyTorch) is timed on both torch CPU and CUDA, so
it contributes ``bard_cpu`` / ``bard_gpu`` columns to the report. The
harness lives under
`test/benchmarks/ <https://github.com/A2R-Lab/GRiD/tree/main/test/benchmarks>`_.

Quick start
-----------

The full multi-baseline sweep:

.. code-block:: shell

   .venv/bin/python test/benchmarks/run_benchmarks.py

A single GRiD cell for fast iteration:

.. code-block:: shell

   .venv/bin/python test/benchmarks/baselines/grid/run.py \
       --robot iiwa14 --base fixed \
       --single-call-iters 50000 --batch-iters 500

A multi-version comparison sweep (multiple GRiD configurations + baselines)
producing the canonical markdown report:

.. code-block:: shell

   .venv/bin/python test/benchmarks/run_multi_version.py \
       --columns glass --robots iiwa14 go2 g1 --bases fixed floating \
       --single-call-iters 50000 --batch-iters 500 \
       --output-dir test/benchmarks/results/comparison_<tag> \
       --report test/benchmarks/benchmark_multi_version_<tag>.md

The full sweep takes ~1.5 hours on an RTX 5090. cuBLASDx (``glass_nvidia``)
was removed in v2.0 of the codegen; see
:doc:`../concepts/cublasdx_removal_design` for the rationale and the
``archive/last-cublasdx`` git tag for the historical comparison data.

Algorithms measured
-------------------

The bench reports 14 rows per cell:

**First-order set (9 algorithms)**:

* ``inverse_dynamics`` — inverse dynamics (RNEA).
* ``minv`` — direct mass-matrix inverse.
* ``forward_dynamics`` — forward dynamics via ``Minv·(τ − c)``.
* ``aba`` — articulated body algorithm (independent forward dynamics).
* ``crba`` — composite rigid body algorithm (mass matrix).
* ``inverse_dynamics_gradient`` — ∂(inverse_dynamics)/∂(q, qd).
* ``forward_dynamics_gradient`` — ∂(forward_dynamics)/∂(q, qd).
* ``end_effector_pose`` — end-effector pose.
* ``end_effector_pose_gradient`` — end-effector Jacobian.

**Second-order set (4 algorithms)**:

* ``idsva_so`` — second-order inverse dynamics, codegen-time-dispatched
  to ``idsva_so_body_frame`` (fixed-base) or ``idsva_so_world_frame``
  (floating-base). The dispatcher row reports whichever variant the
  codegen picked for that cell.
* ``idsva_so_body_frame`` — body-frame propagation, multi-pass.
* ``idsva_so_world_frame`` — world-frame propagation, single-pass.
* ``fdsva_so`` — second-order forward dynamics.

The two IDSVA-SO variants are mathematically equivalent and ship
side-by-side so the report shows the body-vs-world crossover; see
:doc:`../concepts/algorithms/idsva` for the dispatch rule details.

Output schema
-------------

Each cell produces a JSON file at
``<output-dir>/<robot>_<base>_grid_<column>.json``. Each row in the JSON
captures three measurements per algorithm:

* **single-call** — wall time for one host-level invocation (one timestep).
  Anti-LICM machinery prevents the compiler from hoisting the call out of
  the rep-loop, so this is a real per-call cost, not amortized.
* **N=16 (compute)** — kernel-only time for a 16-timestep batch
  (excludes host→device / device→host transfer).
* **N=256 (compute)** — kernel-only time for a 256-timestep batch.

Pinocchio numbers are batch with-memory (CPU codegen makes
compute/transfer inseparable).

Stability and compile flags
---------------------------

A few flags exist for working around hardware/toolchain quirks; the
defaults are correct on sm_120 (RTX 5090) and likely fine on sm_8x:

* ``--cicc-opt-level 2`` — forwards ``-Xcicc -O2`` to floating-base
  GRiD compiles. Workaround for an nvcc cicc -O3 hang seen on sm_8x +
  CUDA 12.6 with the heavy template-surface bench harness. ptxas stays
  at -O3. Default OFF (-O3 cicc).
* ``--ptxas-opt-level 2`` — analogous knob for ptxas, used when nvcc
  ptxas itself crashes (rare).
* ``--per-algo-tus`` / ``--no-per-algo-tus`` — split the bench's
  monolithic ``timeGRiD_{single,batch}.cu`` into per-algorithm TUs
  for parallel nvcc compilation. Currently OFF by default because the
  per-algo split doesn't actually reduce wall time at the small bench
  cell sizes (parse-bound rather than compile-bound on
  ``grid.cuh``); revisit if ``grid.cuh`` ever gets split per-algo too.

Per-host autotuning (recommended for production)
------------------------------------------------

For best performance on a given GPU, run the GLASS autotuner once per
host. It writes a per-host override file under
``GLASS/bench/tuning/<hostname>.cuh`` (the shipped table is left
untouched):

.. code-block:: shell

   cd GLASS
   python3 bench/autotune.py --sm AUTO

Wall time: ~10–15 minutes. Consume the override on subsequent builds
with ``-DGLASS_TUNING_TABLE_LOCAL='"GLASS/bench/tuning/<hostname>.cuh"'``.
See :doc:`../getting_started/installation` for details.

.. _autotune-launch-config:

Autotune launch config for your robot / GPU
-------------------------------------------

GRiD kernels are single-block and **thread-count-invariant** (same result
at any block size), so the optimal ``(resource_tier, threads_per_block)``
for each algorithm is a pure *performance* choice that depends on the
**robot** (DoF / topology) and the **GPU**. GRiD ships measured-optimal
launch configs under ``launch_configs/<robot>/<gpu>.json``; codegen bakes
the matching file into ``grid_launch_config.cuh`` and the host launchers
(and therefore the python / jax / torch bindings) default their launch
config from it. With no entry for your (robot, GPU), GRiD falls back to a
conservative — still correct, just slower — default.

To autotune **your** robot on **your** GPU and write the override:

.. code-block:: shell

   bash tools/autotune_robot.sh <robot> [fixed floating]

   # examples
   bash tools/autotune_robot.sh iiwa14             # fixed + floating
   bash tools/autotune_robot.sh go2 floating        # floating only
   GPU_KEY=a40_sm86 bash tools/autotune_robot.sh g1  # force the GPU key

This:

#. Detects your GPU (``nvidia-smi`` name + compute capability) and derives
   the GPU key ``<model>_sm<arch>`` (e.g. ``rtx5090_sm120``). If detection
   fails, set ``GPU_KEY=<model>_sm<arch>`` and re-run.
#. Runs the GRiD autotune sweep (``run.py --autotune-threads``) for that
   robot + bases. The build is **RAM-safe serial**
   (``GRID_COMPILE_WORKERS=1 --build-jobs 1``) so the big-robot second-order
   TUs — which need ~24–36 GB of ``cicc`` each — never OOM the box.
#. Converts the swept winners into ``launch_configs/<robot>/<gpu>.json`` in
   the documented schema (``gpu``, ``cuda_arch``, ``gpu_name``,
   ``autotune_N``, ``source``, ``bases``).

**Then rebuild to pick it up:** re-run codegen for the robot (codegen
auto-discovers the new ``launch_configs`` file) and rebuild GRiD / the
bindings as usual. The host launchers will default to your tuned
``(tier, threads)``.

.. note::

   **Single-call timing is OFF by default** (it is hard to time and needs
   the ``-rdc`` shim). The autotuner tunes on **batch** timing (``N=256`` by
   default; override with ``AUTOTUNE_N``), which is what the host launch
   config should optimize for. Opt single-call timing back in with
   ``--single-timing`` / ``GRID_BENCH_SINGLE_TIMING=1`` only if you
   specifically need it.

Run the sweep on a **quiet GPU** — timing must be isolated, so close other
GPU workloads first.

**Crowdsource it (please do!):** once you have a good config, open a PR
adding ``launch_configs/<robot>/<gpu>.json`` (one file per (robot, GPU)) so
everyone defaults to fast launches on your hardware. No code changes are
needed — codegen auto-discovers the file. See
`launch_configs/README.md
<https://github.com/A2R-Lab/GRiD/blob/main/launch_configs/README.md>`_
for the full contribution checklist (GPU model, driver / CUDA version,
robot DoF / base to include in the PR description).

Pre-GLASS regression check
--------------------------

The multi-version harness supports an automatic regression check
against the pre-GLASS reference commit (``d2c0d18``). Add ``pre_glass``
to ``--columns`` and the harness will:

#. Create a worktree at ``../GRiD-A2R-pre-glass`` (or
   ``$GRID_PRE_GLASS_WORKTREE``).
#. Initialize submodules pinned to their pre-GLASS revisions.
#. Run the same cell list using that frozen harness.

The ``pre_glass`` column is fixed-base only — the pre-GLASS bench
harness predates floating-base support.

See also
--------

* `test/benchmarks/README.md
  <https://github.com/A2R-Lab/GRiD/blob/main/test/benchmarks/README.md>`_
  — full bench-harness reference (Pinocchio / MJX / Frax / BARD setup, etc.).
* To run a full sweep yourself, use the harness under
  `test/benchmarks/
  <https://github.com/A2R-Lab/GRiD/tree/main/test/benchmarks>`_;
  results are written under ``test/benchmarks/results/`` locally
  (hardware-specific, e.g. sm_120).
* :doc:`cuda_support_status` — currently exercised CUDA paths.
* :doc:`../concepts/algorithms/idsva` and :doc:`../concepts/algorithms/fdsva_so`
  — algorithm details.
