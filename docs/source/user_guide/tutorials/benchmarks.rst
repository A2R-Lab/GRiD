Benchmarks
==========

GRiD ships a benchmark harness that compares GRiD against CPU and GPU
baselines (Pinocchio, MJX, Frax) and against historical GRiD reference
points. The harness lives under
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
  — full bench-harness reference (Pinocchio / MJX / Frax setup, etc.).
* `docs/sweep-on-5090.md
  <https://github.com/A2R-Lab/GRiD/blob/main/docs/sweep-on-5090.md>`_
  — canonical sweep on sm_120 + interpretation notes.
* :doc:`cuda_support_status` — currently exercised CUDA paths.
* :doc:`../concepts/algorithms/idsva` and :doc:`../concepts/algorithms/fdsva_so`
  — algorithm details.
