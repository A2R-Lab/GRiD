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

A single GRiD cell for fast iteration (per-exe path: one TU / exe /
process per algorithm, RAM-safe and crash-isolated):

.. code-block:: shell

   .venv/bin/python test/benchmarks/per_algo_bench.py --robot iiwa14 --base fixed

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

A flag exists for working around hardware/toolchain quirks; the
default is correct on sm_120 (RTX 5090) and likely fine on sm_8x:

* ``--ptxas-opt-level 2`` — drops ptxas from -O3 to -O2, used when nvcc
  ptxas itself crashes (rare).

Per-host autotuning (recommended for production)
------------------------------------------------

For best performance on a given GPU, run the GLASS autotuner once per
host. It writes a per-host override file under
``external/GLASS/bench/tuning/<hostname>.cuh`` (the shipped table is left
untouched):

.. code-block:: shell

   cd external/GLASS
   python3 bench/autotune.py --sm AUTO

Wall time: ~10–15 minutes. Consume the override on subsequent builds
with ``-DGLASS_TUNING_TABLE_LOCAL='"external/GLASS/bench/tuning/<hostname>.cuh"'``.
See :doc:`../getting_started/installation` for details.

.. _autotune-launch-config:

Autotune launch config for your robot / GPU
-------------------------------------------

GRiD kernels are single-block and **thread-count-invariant** (same result
at any block size), so the optimal ``(resource_tier, threads_per_block)``
for each algorithm is a pure *performance* choice that depends on the
**robot** (DoF / topology) and the **GPU**. GRiD ships measured-optimal
launch configs under ``config/launch_configs/<robot>/<gpu>.json``; codegen bakes
the matching file into ``grid_launch_config.cuh`` and the host launchers
(and therefore the python / jax / torch bindings) default their launch
config from it. With no entry for your (robot, GPU), GRiD falls back to a
conservative — still correct, just slower — default.

To autotune **your** robot on **your** GPU and write the override:

.. code-block:: shell

   bash config/autotune_robot.sh <robot> [fixed floating]

   # examples
   bash config/autotune_robot.sh iiwa14             # fixed + floating
   bash config/autotune_robot.sh go2 floating        # floating only
   GPU_KEY=a40_sm86 bash config/autotune_robot.sh g1  # force the GPU key

This:

#. Detects your GPU (``nvidia-smi`` name + compute capability) and derives
   the GPU key ``<model>_sm<arch>`` (e.g. ``rtx5090_sm120``). If detection
   fails, set ``GPU_KEY=<model>_sm<arch>`` and re-run.
#. Runs the GRiD autotune sweep (``per_algo_bench.py --mode autotune
   --stage sweep``) for that robot + bases. The build is **RAM-safe
   serial** (``--compile-jobs 1``, one per-algo TU at a time) so the
   big-robot second-order TUs never OOM the box.
#. Converts the swept winners into ``config/launch_configs/<robot>/<gpu>.json`` in
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
adding ``config/launch_configs/<robot>/<gpu>.json`` (one file per (robot, GPU)) so
everyone defaults to fast launches on your hardware. No code changes are
needed — codegen auto-discovers the file. See
`config/launch_configs/README.md
<https://github.com/A2R-Lab/GRiD/blob/main/config/launch_configs/README.md>`_
for the full contribution checklist (GPU model, driver / CUDA version,
robot DoF / base to include in the PR description).

Cheap re-bake: ``refresh_launch_configs.sh``
--------------------------------------------

``config/refresh_launch_configs.sh`` re-bakes the live
``config/launch_configs/<robot>/<gpu>.json`` picks from an existing
tier-sweep **without** the day-long recompile: it harvests the latest
``results/tier_sweep_phased_*`` sweep (or a ``--sweep-dir`` you pass),
merges the picks, and bakes the per-robot ``bases`` (optionally
``ffi_bases`` with ``--with-ffi``), printing a ``git diff`` of
``config/launch_configs/`` at the end (it never commits).

**Which one when:** ``autotune_robot.sh`` builds a **full monolithic
per-tier binary**, so big robots (g1 / h2_plus) hit the ~hour
second-order recompile — use it for a robot/GPU with no prior sweep.
``refresh_launch_configs.sh`` reuses the cached split (noSO/SO)
binaries from a phased tier sweep
(``bash test/benchmarks/run_tier_sweep_phased.sh``), so re-baking after
a sweep costs no recompile at all.

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

GPU-resident pipelines (no-transfer timing)
-------------------------------------------

GRiD's jax/torch handles can keep an entire control/rollout loop on the GPU —
inputs, dynamics calls, and downstream math never round-trip through host
memory. Two example pairs demonstrate and time this
(``bindings/examples/jax_gpu_resident.py`` / ``torch_cuda_graphs.py`` for the
iiwa14, and ``..._go2.py`` twins for the floating-base go2), driven by
``test/benchmarks/gpu_resident_timing.py``.

Measured on the RTX 5090 (sm_120, 2026-08-09,
``results/overnight_20260809/legC_gpu_resident.json``):

* **JAX resident rollout vs host round-trip** (a ``lax.scan`` rollout calling
  GRiD's dynamics each step): iiwa14 **24.3x / 14.5x / 7.3x** faster at batch
  64 / 256 / 1024 (1.5 ms vs 36.6 ms at B=64); go2-floating **6.8x / 6.9x /
  3.6x** (the floating rollout drives GRiD's own ``integrator`` kernel for the
  on-manifold base retract inside the scan).
* **Torch CUDA graphs**: replay wall-time is roughly break-even with eager
  (0.91-0.99x) — the win is in **CPU submission cost**, ~13 us eager vs ~1.9 us
  replay (**~6.6x**), which is what matters when the CPU is the bottleneck
  feeding a real-time control loop. (An earlier "replay slower than eager"
  reading was a harness artifact — it timed a per-iteration device copy-in;
  replay-only and copy-in variants are now recorded separately.)

Reproduce with ``.venv/bin/python test/benchmarks/gpu_resident_timing.py``
(quiet GPU; results land under ``test/benchmarks/results/``).

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
