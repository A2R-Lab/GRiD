Release measurements
====================

Every number on this page comes from one audited collection on one machine
(NVIDIA RTX 5090, fp32 everywhere except where marked, 26 September 2026,
``modernizing-tests`` branch). Three robots — **iiwa14** (fixed base, 7 joints),
**go2** (floating base, 18 velocities) and **G1** (floating base, 43 velocities) —
fifteen operations, batch sizes 16 to 1024, GRiD through every one of its
surfaces beside Pinocchio, MJX, MuJoCo Warp, MuJoCo CPU, BARD and Frax. Each
cell was validated against an independent fp64 Pinocchio oracle before and after
timing; a cell that failed validation has no bar. The collector, its protocol
and the raw captures are in ``test/benchmarks/release/`` (see its ``README.md``).

In short:

* **GRiD's kernel leads every GPU library.** With no host–device transfers on
  either side, GRiD's CUDA compute-only call is faster than MJX on every
  measured cell (1–30×, most cells 5–20×) and faster than MuJoCo Warp on the
  dynamics operations (2–17×). It is faster than BARD by 14–160× and than Frax
  by 4–7×. This holds at every batch size up to 1024. The library's resident
  call includes its framework dispatch. With GRiD's own JAX dispatch included
  too (the middle panel of Figure 3) the ratios shrink, and two MJX cells turn
  into losses: floating-base end-effector pose at batch 1024, 0.7× on go2 and
  0.9× on G1.
* **GRiD does not win everywhere.** End-effector pose on the floating-base
  robots is a tiny kernel and MuJoCo Warp's is faster (GRiD at 0.4–0.9× of
  Warp kernel to kernel, 0.3–0.6× through the JAX API). On G1 at batch 1024 the
  ABA and mass-matrix kernels are at parity with Warp. Through the JAX API's
  complete call from host arrays GRiD still beats MJX and BARD on nearly every
  cell. On that boundary it is at parity or behind Warp on the first-order
  operations, because Warp's host round trip is cheaper than JAX's, and it
  trades cells with Frax.
* **Against Pinocchio, the boundary and the batch decide.** Pinocchio's
  code-generated C++ (best of 1, batch/16 and 8 threads per cell) wins nearly
  every cell below batch 64. With the data already on the GPU, GRiD's kernel
  is 1.2–14× faster at batch 1024 on every operation Pinocchio code-generates.
  Including the host–device copies, GRiD's CUDA host call reaches parity around
  batch 128–256 and 0.7–6× at 1024. Most of those cells are above 1×. The
  floating-base bias vector and G1's mass matrix stay just below parity.
  GRiD's JAX full call adds a host round trip of roughly 150–200 µs, loses to
  Pinocchio's codegen at small batch and reaches parity only at the largest
  batches. The PyTorch full call is much cheaper, about 35 µs over the CUDA
  host call on iiwa14 RNEA at batch 256, and the NumPy call is within 1 µs of
  it. Most users call Pinocchio's standard templated API, which measured about
  twice the code-generated time. Both are shown. Pinocchio's centroidal
  momentum matrix beats GRiD's host call on the floating robots at every batch
  (0.1–0.6×).

Figure 1 — Where the time goes
------------------------------

.. image:: _static/release/stacked_core.svg
   :alt: Nine panels (RNEA, grad RNEA, Hessian RNEA by iiwa14, go2, G1). GRiD is a three-segment stack of kernel compute, memory traffic and JAX wrapper overhead; each competitor is its resident time plus a hatched host round trip; Pinocchio is its code-generated time plus a hatched standard-API cap. Log axis, batches 16 to 1024.
   :target: _static/release/stacked_core.svg

Absolute microseconds per complete batch, log axis. GRiD is the blue stack:
the generated kernel's compute-only time (CUDA host call), the host↔device
memory traffic on top (with-memory host call minus compute), and the JAX API's
own overhead on top of that (full call minus with-memory). Each GPU competitor
is its resident evaluation plus a hatched cap for its host round trip.
Pinocchio CPU is its code-generated time plus a hatched cap for the standard
API. Where Pinocchio has no code-generated path (the Hessian) both modes run
the same standard analytical code and the whole bar is hatched. Every segment
is a difference of two measured medians on the same cell. A negative
difference is marked rather than clamped. ``*`` marks fp64 arithmetic and
``†`` a retained fp32 accuracy warning (see below). Whiskers show the range of
the three run means. The same figure for all fifteen operations:
:download:`stacked_all.svg <_static/release/stacked_all.svg>`.

Figure 2 — Speedup against Pinocchio (CPU)
-------------------------------------------

.. image:: _static/release/speedup_pinocchio.svg
   :alt: Four heatmaps, rows are robot and operation, columns are batch sizes. Top row: GRiD kernel compute-only against Pinocchio code-generated and standard APIs. Bottom row: GRiD CUDA host call including copies against the same two. Blue cells are GRiD faster, red cells are Pinocchio faster.
   :target: _static/release/speedup_pinocchio.svg

Ratio of Pinocchio's warmed batch time to GRiD's, blue when GRiD is faster,
red when Pinocchio is. The top row is the kernel with the data already on the
GPU, the situation inside a GPU optimiser. The bottom row includes the copies
in and out. Pinocchio runs a persistent C++ thread pool. For every cell the
best of the recorded thread counts (1, batch/16 and 8) is used, so small
batches often run on one or two threads.

Figure 3 — Speedup against the GPU libraries
--------------------------------------------

.. image:: _static/release/speedup_gpu_resident.svg
   :alt: Heatmaps of GRiD kernel compute-only time against MJX, MuJoCo Warp, BARD and Frax resident calls, rows are robot and operation, columns are batch sizes.
   :target: _static/release/speedup_gpu_resident.svg

.. image:: _static/release/speedup_gpu_jax_resident.svg
   :alt: Heatmaps of GRiD JAX resident call against MJX, MuJoCo Warp, BARD and Frax resident calls: no memory traffic on either side, each framework's own dispatch included.
   :target: _static/release/speedup_gpu_jax_resident.svg

.. image:: _static/release/speedup_gpu_full.svg
   :alt: Heatmaps of GRiD JAX full call against MJX, MuJoCo Warp, BARD and Frax full calls from host arrays to host arrays.
   :target: _static/release/speedup_gpu_full.svg

Three boundaries, each measured directly on both sides. **Top:** the
competitor's warmed device-to-device evaluation (including its framework
dispatch) over GRiD's bare CUDA compute-only call (no framework on GRiD's
side). **Middle:** no host–device transfers on either side — GRiD's JAX
resident call against the competitor's resident call, each paying its own
framework dispatch and any device-side copies it makes; this is the strictest
like-for-like view.
**Bottom:** the complete call from host arrays to host arrays on both sides,
through GRiD's JAX API; this is where Warp's cheaper host round trip shows, and
where GRiD's own JAX overhead on large outputs shows (on G1 at batch 1024 the
JAX resident mass-matrix call is 3× the kernel, a device copy of a 7.6 MB
output). Second-order competitor cells are absent by design:
this is an analytical-Hessian study and no finite-difference or nested-autodiff
Hessians were built to fill them.

Protocol
--------

* One worker process per (robot, backend, operation) and three independent
  repeats. Each repeat warms the exact closure for at least 1.5 s of sustained
  calls, then takes 5 warm-ups and 30 timed samples. The reported value is the
  **median of the three run means**, with the range recorded: whiskers on the
  stacked bars, and ``~`` on a heatmap cell when either side's three run means
  spread by more than 1.5×.
* **Run-to-run variability is real on the Python-driven paths.** GRiD's CUDA
  host-call boundaries (kernel and with-memory) repeat within a few percent.
  The Python-driven paths (GRiD's JAX API, MJX, Frax and Pinocchio's thread
  pool), at both the resident and the full-call boundary, show occasional
  repeats in which every sample of the window runs 3–5× slower while the
  C++-timed kernel in the same process is unchanged; this reproduces on
  an otherwise idle machine, the GPU clock stays at its maximum throughout,
  and a run pinned to the efficiency cores was slower but free of it, which
  points at CPU power management (``intel_pstate`` in ``powersave`` with cores
  idling to 800 MHz between synchronised calls) rather than the GPU. Those cells are marked;
  their medians are indicative, not precise. A governor-locked re-collection of
  the affected cells is a listed follow-up.
* Boundaries measured directly, never stacked from unrelated runs:
  *resident* = inputs pre-placed on the device, outputs left on the device,
  synchronised, no host–device transfers (any device-side copies a framework
  makes are inside it; the same rule for every GPU library and for GRiD's JAX
  and PyTorch surfaces); *full call* = fresh upload, call, download to host arrays. GRiD's CUDA host call is the generated
  ``<op>_compute_only`` (resident) and ``<op>`` (with memory) host functions
  called from a C++ harness, checked bitwise against the NumPy wrapper.
* fp32 for every backend, including GRiD's Hessians, with two exceptions
  marked ``*``: Pinocchio's analytical second derivatives and every MuJoCo CPU
  cell (the installed binary computes in double). MuJoCo Warp is graph-captured;
  its eager launch time is in the table as ``resident_eager_us``.
* Identical inputs for every backend (seeded legal states, normalised
  quaternions), a shared fixture per robot, hashed into every capture.
* Every cell validated against RBDReference's Pinocchio-backed fp64 oracle
  before timing, after timing and across repeats (``rtol 2e-4, atol 1e-3``
  entrywise). Under the ``fp32-fd-warnings`` policy, fp32 forward-dynamics-family
  cells that exceed the entrywise gate but keep every output block within 0.1 %
  relative L2 error are **retained with their errors reported** (status
  ``accuracy_warning``, ``†``); this applies to every backend equally.

Coverage and statuses
---------------------

A missing bar is never a zero. Every planned cell carries a status and a
reason in the table:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Status
     - Meaning
   * - ``validated``
     - Timed; agreed with the fp64 oracle before and after timing.
   * - ``accuracy_warning``
     - Timed; fp32 forward-dynamics-family cell retained under the policy above with
       its measured error (GRiD JAX grad/Hessian ABA, Pinocchio/MJX/Warp/Frax ABA
       family, Pinocchio M⁻¹).
   * - ``adapter_pending``
     - The library may support the operation but this collection's adapter does
       not wire it (MJX and Warp M⁻¹; centroidal momentum and Coriolis matrices on
       every competitor; end-effector derivatives on BARD, Frax and Pinocchio, whose
       spatial kinematic derivatives are not the RPY pose-coordinate derivatives GRiD
       returns). Not a capability claim.
   * - ``excluded_method``
     - Second-order cells that would need finite differences or nested autodiff.
   * - ``model_mismatch``
     - Frax on the floating-base robots (six-coordinate base against the shared
       quaternion fixture, no validated conversion); Frax appears on iiwa14 only.
   * - ``validation_failed``
     - G1 Pinocchio Hessian ABA (both APIs) at batch 256 and 1024: one tensor entry
       of one sample disagrees with the oracle by about 1 %, identically in both
       Pinocchio modes. Not timed; under investigation.

Downloads and reproduction
--------------------------

:download:`table.csv <_static/release/table.csv>` — every planned cell:
status, reason, host and resident medians, run range, threads, dtype, oracle
errors and exceedance counts.
:download:`decomposition.csv <_static/release/decomposition.csv>` — GRiD's
kernel / memory / C ABI / NumPy / JAX / PyTorch terms per cell, and Pinocchio's
standard-API overhead over its codegen.
:download:`manifest.json <_static/release/manifest.json>` — the report identity
(table hash, commits, capture order, accepted source drift, status counts) and
the hash of every figure.

From the repository root, with a GPU and the ``[all]`` extras installed:

.. code-block:: shell

   # plan without launching GPU work (add --execute to collect)
   .venv/bin/python -m test.benchmarks.release.collect --stage core
   .venv/bin/python -m test.benchmarks.release.collect --stage table --accuracy-policy fp32-fd-warnings
   # combine captures (narrow re-collections first: the first listed capture supersedes)
   .venv/bin/python -m test.benchmarks.release.report <capture>... --output <report>
   # website figures (DRAFT banner unless --approve, after the audit)
   python docs/plot_release_figures.py <report>

Known limitations and open items
--------------------------------

* End-effector pose on go2 and G1 is slower than MuJoCo Warp at every
  boundary. The kernel itself is 10–50 % slower, and JAX dispatch (~44 µs)
  then dominates a ~30 µs kernel. This is both a kernel item and a wrapper
  item.
* GRiD's JAX resident path carries 240–290 µs beyond the kernel on G1's ABA
  and mass matrix (batch 256–1024): a device copy of the large output and
  the ABA composition in the wrapper. The C++ host call does not pay it.
* The NumPy and PyTorch surfaces spend over a millisecond staging G1 gradient
  outputs at batch 1024 (pageable host memory); pinned output buffers are a
  backlog item.
* Full-call variability (Protocol): re-collect the JAX/MJX/Pinocchio host
  cells with the CPU governor set to ``performance`` and record the setting in
  the provenance; until then the marked cells carry their range.
* Pinocchio's CPU numbers are the best of three thread counts (at most eight)
  on a 24-thread desktop CPU; they are not a claim of optimal CPU threading.
* Collision-checking timings are a separate follow-up with their own protocol
  (matched geometry, coverage and agreement beside latency).
