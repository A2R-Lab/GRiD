Release measurements
====================

.. note::

   **Data from the 27 September 2026 run:** 210 worker processes,
   1,260 measurements, all passing the existing strict numerical checks.

These measurements cover **iiwa14** (fixed base, 7 velocities), **go2** (floating
base, 18 velocities), and **G1** (floating base, 35 velocities) on one NVIDIA
RTX 5090 and Intel Core Ultra 9 285K system. Batch sizes are 16, 32, 64, 128,
256, and 1024. The main figures compare RNEA, its analytical gradient, and its
analytical Hessian. The wrapper study covers RNEA and its gradient through
CUDA C++, the C ABI, NumPy, JAX, and PyTorch.

The main collection used commit ``e92477857b0ca53fb307a8fa4799a4afd5757177``.
It includes the floating-base JAX Hessian packing correction and preserves
fp64 mapped inputs for Pinocchio's analytical paths. No numerical tolerance
was relaxed. Subsequent figure and documentation edits do not alter timed code.

What the measurements show
--------------------------

GRiD's strongest performance is available to solvers and libraries that keep
robot data on the GPU. The measurements also show that this advantage can
survive transfers and framework dispatch, rather than being limited to the
compute-only boundary.

* **Fast robot-specific CUDA for GPU-resident applications.** GRiD's native
  compute-only call has a lower median than every evaluated GPU baseline on
  its matched core RNEA/gradient cells. Against MuJoCo Warp and MJX, the
  ratios span 4.3–38.6×. These are resident-call comparisons, not isolated
  kernel comparisons: GRiD includes native launch and synchronization;
  the competitors also include their framework dispatch.
* **GRiD's JAX API is faster than MJX on all 36 matched RNEA/gradient cells.**
  The ratios are 1.8–4.7× for complete host-to-host calls and 2.2–14.7× with
  inputs and outputs resident on the GPU. Every one of these comparisons
  remains above 1× across the observed ranges of the three process means.
  These are measured ranges, not statistical confidence intervals.
* **Large-batch gradients show substantial compute and host-call gains.**
  At batch 1024, GRiD's compute-only CUDA calls for RNEA gradients are
  11.7× faster on iiwa14 and 5.6× faster on go2 than Pinocchio codegen's
  CPU calls. GRiD takes 30.8 µs and 116.6 µs, respectively, versus
  361.3 µs and 657.8 µs for Pinocchio codegen. The compute-only comparison
  excludes GRiD's transfers but includes native launch and synchronization.
  Including transfers, GRiD's C++ host calls take 59.4 µs and 257.3 µs,
  retaining 6.1× and 2.6× speedups at the matched host-to-host boundary.
  These host-call wins remain separated across the observed repeat ranges.
* **Larger batches turn compute gains into host-call wins.** Including
  transfers, GRiD's CUDA host call has a lower median than the Pinocchio
  codegen-mode adapter in 14 of 45 core cells at batches 16–256; 13 of those
  wins remain separated across observed ranges. At batch 1024, it wins 8 of
  9 cells across those ranges; the remaining comparison overlaps. Both
  Pinocchio modes use the same standard analytical fp64 path for Hessians,
  not a code-generated Hessian. Pinocchio wins many small-batch comparisons,
  particularly the lighter RNEA workload when GRiD's transfers are included.
  The batch-1024 wins include all three robots' Hessians. There are also
  selective wins with JAX overhead included: iiwa14's RNEA gradient at batch
  1024 takes 303.6 µs through GRiD JAX versus 361.4 µs through Pinocchio
  codegen, a 1.2× median speedup for complete host-to-host calls.
* **Wrapper costs matter.** GRiD JAX resident calls beat MuJoCo Warp on all
  18 matched RNEA cells across observed ranges, but the host-to-host results
  are mixed. On iiwa14 RNEA at batch 256, median full calls are 23.3 µs in
  CUDA, 24.7 µs through the C ABI, 26.8 µs through NumPy, 61.0 µs through
  PyTorch, and 224.8 µs through JAX. This is one illustrative cell, not a
  universal wrapper overhead.

Figure 1 — Where the time goes
------------------------------

Keeping data on the GPU makes the most of GRiD's compute performance.
Larger batches can amortize transfers, with host-call wins extending to
analytical Hessians on all three robots at batch 1024. Full JAX calls can
also win, as the iiwa14 gradient example above illustrates; the crossover
depends on the operation, batch size, robot, and baseline.
Pinocchio wins many small-batch host-call comparisons, especially for RNEA;
GPU execution is not the fastest choice for every workload.

.. image:: _static/release/stacked_core.svg
   :alt: Clustered bars for RNEA, its gradient and Hessian on three robots, with separate Pinocchio API bars and gray hatched boundary increments.
   :target: _static/release/stacked_core.svg

Absolute microseconds per complete batch, on a log axis. GRiD's JAX full-call
bar is decomposed into its native CUDA compute-only call, the CUDA transfer
increment, and the additional JAX API increment. Green denotes the CUDA call,
gray diagonal hatching the GPU–CPU I/O increment, and gray dots on white the
JAX wrapper increment.
These are **differences of measured call times**, not isolated
measurements of individual wrapper components. The CUDA compute-only call
includes native host launch and synchronization; it is not CUDA-event timing
of a bare kernel.

GPU competitors show resident calls plus full-call increments when the
decomposition is consistent. Pinocchio's two API modes are separate bars for
RNEA and its gradient. Only its standard analytical fp64 API is shown for the
Hessian; there is no codegen Hessian bar. BARD and Frax remain in the tables
but are omitted from this figure for clarity.
A red triangle means
the decomposition is unavailable and the **measured full-call total** is shown
without a stack. Three MJX gradient cells have this flag because at least one
repeat's resident time exceeded its full-call time, even though the median
difference is positive. Neither boundary is discarded or clamped.

Figure 2 — Speedup against Pinocchio (CPU)
--------------------------------------------

.. image:: _static/release/speedup_pinocchio.svg
   :alt: Core-operation speedups against both Pinocchio modes, separately for CUDA compute-only and CUDA host calls including copies.
   :target: _static/release/speedup_pinocchio.svg

Ratios are baseline time divided by GRiD time. Above 1× favors GRiD; below 1×
favors the baseline. The top row excludes GRiD's host–device transfers and is
therefore a different workload boundary from Pinocchio's host-array call.
The bottom row includes GRiD's transfers and compares host arrays in and out
on both sides. Pinocchio uses a persistent C++ thread pool, choosing the best
recorded candidate from ``{1, max(1, batch//16), 8}``, excluding counts above
eight or the batch size. Eight is this study's configured worker ceiling,
not a Pinocchio limit; counts above eight were not evaluated. All 24 logical
CPUs were available to the processes. All tested variants are retained in
the raw captures.

Pinocchio's CPU paths are strong at small batches, particularly for RNEA.
Transfers can reverse a compute-only advantage: for iiwa14's RNEA gradient
at batch 32, GRiD's compute-only call takes 14.8 µs versus 15.8 µs for
Pinocchio codegen, but GRiD's full C++ host call takes 24.6 µs.

Figure 3 — Speedup against the GPU libraries
--------------------------------------------

.. image:: _static/release/speedup_gpu_resident.svg
   :alt: CUDA compute-only calls against GPU-library resident calls.
   :target: _static/release/speedup_gpu_resident.svg

.. image:: _static/release/speedup_gpu_jax_resident.svg
   :alt: GRiD JAX resident calls against GPU-library resident calls, including each framework's dispatch and synchronization.
   :target: _static/release/speedup_gpu_jax_resident.svg

.. image:: _static/release/speedup_gpu_full.svg
   :alt: Complete host-array calls through GRiD JAX and each GPU baseline.
   :target: _static/release/speedup_gpu_full.svg

**Top:** native CUDA compute-only calls against competitors' resident API
calls. Both include launch and synchronization, but only the competitors pay
framework dispatch. **Middle:** resident API calls with framework dispatch on
both sides. **Bottom:** full host-to-host calls on both sides. These boundaries
answer different application questions and must not be combined into one
unqualified speedup claim.

Heatmaps show ratios of medians, not guarantees of separation across repeat
ranges. ``~`` marks a side whose run means span more than 1.5×. Colors are
clipped at 100×; printed cell values retain the measured ratios. Missing cells
reflect adapter coverage, model mismatch, or study scope, not library-wide
incapability. No finite-difference or nested-autodiff Hessian sweep was added
to this analytical-Hessian comparison.

Figure 4 — Wrapper costs
------------------------

.. image:: _static/release/wrappers.svg
   :alt: RNEA and gradient call wall times for CUDA Device, C++ Host, NumPy, PyTorch and JAX.
   :target: _static/release/wrappers.svg

Five solid bars show the measured wall time at each call boundary, ordered
CUDA Device, C++ Host, NumPy, PyTorch, JAX. CUDA Device is the native
compute-only call with resident data, including launch and synchronization;
it is not bare device-event timing. C++ Host includes transfers with prepared
host buffers. The Python bars include their complete host-to-host API calls.
The C ABI remains available
in the downloadable data but is omitted from this application-facing figure.

Protocol
--------

* Three independent worker-process repeats per robot, backend and operation.
  Each boundary warms for at least five calls and 1.5 seconds, then records
  **300 samples for the main and wrapper figures**. Reported times are medians
  of the three process means, with every repeat retained.
* CPU governors were ``performance`` on all 24 CPUs, with all CPUs available
  to the processes. Raw EPP was ``default`` under active ``intel_pstate``;
  the complete policy is captured. A diagnostic P-core-only pilot did not
  consistently improve results and is not included in release timing data.
* The box was reserved for serial measurement, with quiet checks between
  workers. Native CUDA, C ABI, NumPy, PyTorch, Warp and the other stable paths
  were not exempted from variability checks. Of 420 supported groups, 17
  full-call groups span more than 1.5× across process means; 23 of 222
  resident groups do so. These are marked, not selectively rerun.
* Resident means inputs already on the GPU and outputs left there, including
  synchronization and any device-side copies. Full call includes host input
  upload and output download. CUDA boundaries use the generated host functions
  through a C++ timing harness.
* fp32 arithmetic except marked ``*`` cells: Pinocchio analytical Hessians
  and MuJoCo CPU in this core study. Secondary tables also include fp64
  Pinocchio end-effector pose. Input precision is recorded separately.
* Identical seeded states, normalized quaternions, URDF hashes and input-value
  hashes are used across backends. Every timed cell is checked before and
  after timing against RBDReference's Pinocchio-backed fp64 oracle, plus
  repeatability and boundary agreement (entrywise ``rtol=2e-4, atol=1e-3``).
  This is a separate reference path, not an independent library when checking
  Pinocchio itself. All 1,260 main-study measurements pass the strict checks.

Downloads and reproduction
--------------------------

* `Browse both complete tables <_static/release/tables.html>`_.
* :download:`Core and wrapper CSV <_static/release/table.csv>`:
  594 planned cells, including 420 validated and 174 explicit N/A cells.
* :download:`Secondary-operation CSV <_static/release/secondary_table.csv>`:
  the remaining operations, retained from the earlier **30-sample** collection.
  This is a separate population, not mixed with 300-sample core comparisons.
* :download:`Decomposition CSV <_static/release/decomposition.csv>`;
  :download:`observed comparison ranges <_static/release/comparisons.json>`.
* :download:`Audit summary <_static/release/audit.json>`,
  :download:`figure manifest <_static/release/manifest.json>`, and
  :download:`secondary provenance <_static/release/secondary_provenance.json>`.

Every table row includes status, reason, dtype, times, run ranges and numerical
error information. ``adapter_pending`` means our adapter is not wired;
``excluded_method`` means outside the analytical study; ``model_mismatch``
identifies Frax's unvalidated floating-base conversion. No unavailable result
is treated as zero.

Secondary fp32 forward-dynamics-family cells may carry ``accuracy_warning``:
they exceed the strict entrywise gate while every output block remains within
0.1% relative L2 error under the explicitly accepted policy. Componentwise
relative errors can be percent-level or larger near zero. Those measurements
retain their error metrics and are **not** labeled strict passes. The previously
failing G1 Pinocchio forward-dynamics Hessians pass after preserving fp64 inputs;
the secondary table includes their replacement captures.

From the repository root, plan without launching GPU work:

.. code-block:: shell

   .venv/bin/python -m test.benchmarks.release.collect --stage core --iterations 300
   .venv/bin/python -m test.benchmarks.release.collect --stage wrappers --iterations 300
   .venv/bin/python -m test.benchmarks.release.report <matched-capture> --output <report>
   .venv/bin/python docs/plot_release_figures.py <report>

Add ``--execute --output <fresh-capture>`` only in a coordinated quiet window.
Raw captures remain in ``test/benchmarks/results/``; published assets contain
their hashes and provenance.

Measurement scope
-----------------

* One desktop CPU/GPU system and three robots do not establish performance on
  every robot or on Jetson. Some JAX, MJX and Pinocchio cells remain variable;
  these measurements do not establish a root cause.
* Pinocchio's candidate thread counts are capped at eight. This is not a claim
  of optimal CPU threading. Its fp64 Hessians are a precision exception, not
  an equal-precision comparison with GRiD's fp32 Hessians.
* Boundary increments do not separately identify framework dispatch, staging,
  or large-output costs.
* This dataset does not measure collision performance.
