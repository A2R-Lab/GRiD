Release measurement plan
========================

**Release protocol for review. Functional GPU smoke tests have been run, but
no new release-quality performance sweep has been collected.** Collect the comparison
plots first, the remaining operations for a full results table second, and
collisions later. The previous resident-rollout figure is no longer in scope.
See :doc:`plot_designs` for a CPU-generated layout preview, not release evidence.

Collection implementation
-------------------------

The validation-gated implementation is in ``test/benchmarks/release/``. Its
``README.md`` documents exact commands, adapter coverage, precision exceptions,
and measurement boundaries. Plan without launching GPU work::

   .venv/bin/python -m test.benchmarks.release.collect --stage core

Run a functional check in a fresh directory::

   .venv/bin/python -m test.benchmarks.release.collect --stage core --smoke \
       --execute --output test/benchmarks/results/release-core-smoke

Export draft figures and the complete table::

   .venv/bin/python -m test.benchmarks.release.report \
       test/benchmarks/results/release-core-smoke \
       --output test/benchmarks/results/release-core-smoke-report

Smoke reports are prominently marked and must not replace website placeholders.
The collector retains actual shared inputs, per-iteration samples, numerical
errors, source/build identities, and capture hashes. Failed or missing cells
never become zero-valued bars. Some expanded FD-table cells exceed the current
entrywise accuracy gate. Following review, ``--accuracy-policy fp32-fd-warnings``
retains bounded fp32 Minv/FD discrepancies with explicit error metrics, not strict
pass labels; shape/nonfinite failures and gross errors remain blocked. All 135
core cells passed, and the affected 45 GRiD cells and all 120 wrapper cells
passed again after the parser fix. See ``test/benchmarks/release/BUG_TRIAGE.md``
and the collector README for details. The
default core comparison currently labels GRiD's JAX API explicitly; it is not
a claim of pure native-kernel latency. C++/NumPy/JAX/PyTorch full-call comparisons
are collected in the separate wrapper stage.

Collection matrix and order
---------------------------

Use **iiwa14 fixed-base, go2 floating-base, and G1 floating-base**, with
**B = 16, 32, 64, 128, 256**. Confirm these base choices before collection.
Default to **fp32**, including GRiD Hessians. Use one GPU/host initially and existing documented launch settings, not a new
autotuning sweep. Pin the exact G1 model and joint count; historical baselines
have used different G1 configurations. Never compare just by robot name.

.. list-table::
   :header-rows: 1
   :widths: 18 57 25

   * - Stage
     - Operations
     - GRiD configurations
   * - 1. Core plot
     - RNEA, grad RNEA, Hessian RNEA (IDSVA-SO)
     - 45, before repeats and timing boundaries
   * - 2. Interface plot
     - RNEA and grad RNEA through NumPy/pybind, JAX, and PyTorch,
       plus a matched CUDA C++ reference
     - 90 wrapper configurations; 30 native references reusable only if matched
   * - 3. Full table
     - Add Minv, FD, grad FD, Hessian FD (FDSVA-SO), FK, grad FK, Hessian FK
     - 105 additional native configurations; 150 total over ten operations
   * - 4. Follow-up
     - Collision latency, coverage, and agreement
     - Separate protocol below; not a gate for the first two plots

These counts are not a runtime estimate and exclude comparator cells. G1
second-order generation/compilation may dominate preparation. Preflight one
build and record time/peak memory before launching the full stage.

Recommended comparison set
--------------------------

**Main plot:** GRiD, Pinocchio, MJX, and MuJoCo Warp for RNEA; GRiD,
Pinocchio, and MJX for grad RNEA; GRiD and analytical Pinocchio for Hessian
RNEA. This keeps clusters readable while representing CPU code generation,
GPU simulation, autodiff, and analytical derivatives. Label each method and
device explicitly. The gradient row is not restricted to analytical methods.

**Full table:** add standalone MuJoCo CPU, BARD, and Frax where their supported
operations match. Include the MuJoCo Warp finite-difference gradient as an
explicitly labeled secondary comparison if feasible, not a headline full-
Jacobian competitor silently mixed with analytical/autodiff methods. Do not
build finite-difference Hessians merely to fill missing cells. Keep cuRobo as
optional until its model and full-output contract match this collection.

.. list-table:: Adapter audit, not a certified library support matrix
   :header-rows: 1
   :widths: 18 47 35

   * - Baseline
     - Existing path and applicable comparison
     - Preparation or scope limit
   * - Pinocchio CPU
     - Codegen paths for RNEA, Minv, FD, grad RNEA, grad FD; direct analytical
       RNEA second derivatives and additional direct/composed operations
     - Hessian is not codegen. Current direct second-order path uses double;
       footnote that mixed-precision comparison. Pin CPU threads/batch policy.
   * - MJX GPU
     - RNEA, FD, pose and autodiff dynamics gradients are wired
     - Match derivative variables and output materialization. Exclude optional
       second-order autodiff from this deliberately analytical Hessian study.
   * - MuJoCo Warp GPU
     - RNEA, FD, pose and CRBA are wired
     - Current gradient adapter is finite differences, not analytical/autodiff.
       Not applicable to the selected headline gradient-method set.
   * - MuJoCo CPU
     - Useful familiar simulator baseline for supported first-order operations
     - No standalone CPU timing column found in the multi-version driver.
       Adapter pending, not evidence of an unsupported library operation.
   * - BARD GPU
     - PyTorch RNEA, FD and CRBA paths are wired
     - Add applicable timings to the table. Derivative comparison is not wired;
       do not infer that the library lacks differentiability.
   * - Frax GPU
     - Dynamics and inverse-inertia paths are wired; CPU results also exist
     - Select GPU results explicitly; verify model and operation semantics.
       Wider derivatives are not covered by the current adapter.
   * - cuRobo GPU, optional
     - Separate fixed-base G1 RNEA/FK driver exists
     - Historical model/base differ; its backward timing is a VJP, not a full
       Jacobian. Not a comparable grad RNEA cell without additional work.

Adapter sources are under ``test/benchmarks/baselines/``. The main coordinator
is ``test/benchmarks/run_multi_version.py``; cuRobo uses a separate path.
Availability of adapter code is not proof it runs at the release tip.

Use **N/A — unsupported** only after checking the selected version cannot
produce the agreed output through an applicable supported API. Distinguish
**adapter pending**, **not collected**, **excluded method**, **model mismatch**,
**failed validation**, and **OOM/error**. Record reasons in the table rather
than leaving blanks. A coverage advantage may be stated for verified support;
an unwritten adapter is not a competitor capability limitation.

.. _figure-core:
.. _figure-derivatives:

Figure 1 — Clustered operation comparisons
------------------------------------------

Rows are RNEA, grad RNEA, Hessian RNEA; columns are the three robots. Each panel
has five batch clusters and baseline-colored bars. Use absolute microseconds
per complete batch, log y axes if needed, and matching row scales where legible.
Label ratios from the same full-call boundary, not across different boundaries.

The **bar top is a warmed host-input to host-output call**, including required
transfers and synchronization. The colored portion for a GPU baseline is its
warmed resident-input/resident-output evaluation wall time, including ordinary
API dispatch. The **gray hatched cap is the paired total-minus-resident delta**,
not a separately measured pure PCIe transfer or pure Python overhead. CPU
Pinocchio has no GPU-transfer cap. Resident costs already include Python launch
overhead where applicable; do not call them pure kernel latency.

Measure both boundaries directly, using the same computation/configuration,
allocation policy and inputs. If a total is below its resident measurement,
flag/repeat that cell rather than silently clamping overhead to zero. Retain
raw paired results. The stack is a descriptive difference of timings, not a
claim that independently measured components have an exact additive execution
trace. Error bars describe total-call run variability, not fabricated component
uncertainty. Publish resident and full-call values separately in the table.

* [ ] Match model assets, base convention, gravity, input state and external
  forces, output components, frames, quaternion/tangent convention, and variables
  differentiated. Full Jacobians/Hessians are not interchangeable with VJPs or
  directional derivatives. FK means the agreed endpoint pose, not all-link poses.
* [ ] Default to fp32 for every operation. Retain the current fp64 Pinocchio
  analytical Hessian path as an explicitly footnoted exception, not a
  same-precision speedup. Label its bars ``Pinocchio CPU (fp64)*``. Audit direct
  Pinocchio FK and other table paths too; their current implementations also
  use double internally. Mark every actual exception, not just the headline one.
  Do not infer arithmetic precision from the input array dtype.
* [ ] Validate outputs before timing, with dtype-appropriate recorded tolerances.
  Use legal joint states and normalized floating-base quaternions. Confirm
  Hessian blocks and symmetry conventions, not merely matching array sizes.
* [ ] Repair/adapt timing boundaries before claiming host-to-host comparisons.
  Current MJX, MuJoCo Warp and BARD ``with_mem`` paths do not include equivalent
  output copies to host. Current native compute-only timings and Python wall
  timings also cannot simply be stacked as if they were identical boundaries.
* [ ] Add an explicit five-batch execution filter where missing. Several drivers
  currently include 1024 or hard-code templates; hiding that batch in the plot
  does not avoid its compilation/run cost. Subset selected operations as well.
* [ ] Warm exact closures/builds, exclude compilation, use a fixed disclosed CPU
  thread policy, time on an idle GPU, and run three independent warmed repeats.
  Report the median of the three run means and their range, named precisely.

Precision footnote and accuracy evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accuracy-warning footnote for the expanded table: **"FP32 forward-dynamics
computations can amplify rounding and cancellation, particularly in high-velocity
cases. Selected entries show percent-level discrepancies from the fp64 reference;
relative errors can be larger near zero. GPU reduction order can also cause
small run-to-run differences. Accuracy-warning timings are retained
with measured errors and entrywise exceedance counts, not labeled strict
validation passes."** Do not claim a single-digit-percentage upper bound.
The opt-in policy requires each output block's relative L2 error to remain at
most 0.1%, while preserving the original entrywise exceedance counts. It applies
equally to every backend's fp32 Minv/FD-family operations and does not override
shape or nonfinite failures. Policy v2 additionally permits bounded fp32
repeatability/API-boundary discrepancies under the same budget, with both paths
checked against the oracle before and after timing. Larger discrepancies still
block; native C++/NumPy agreement remains exact. Pre/post sampling is not a
guarantee about every timed call. Use the preparation-only commands in
``test/benchmarks/release/README.md`` to populate reusable caches before timing.

Proposed caption footnote: **"GRiD uses fp32. The Pinocchio analytical Hessian
baseline uses fp64 in our current benchmark implementation; this comparison
therefore includes a precision difference."** Do not say Pinocchio cannot
compute fp32 Hessians: its installed C++ API is scalar-templated, while our
``timePinocchio.cpp::idsvaSoThreaded_inner`` explicitly uses ``pinocchio::Model``,
``pinocchio::Data`` and casts to double. Whether adapting that path is worthwhile
is separate from choosing the practical baseline for this collection.

The existing fp32 CUDA RNEA Hessian tests compare all selected tensor entries
at **rtol = 2e-4, atol = 1e-3**, with the Pinocchio extension as the default
independent oracle. Archived JUnit records include iiwa14-fixed passing in
``test/.split_suite/receipt_20260920_195310/cuda_04_kinematics_thread__mjx_tier_inva.xml``
and go2/G1 floating world-frame passes in
``test/.split_suite/receipt_20260924_145347/cuda_01_gen3_fetch_baxter_f_ext_contact_.xml``.
This supports historical agreement within the test criteria, **not equal
precision, a measured 0.02% error bound, or validation of every benchmark cell**.
The mixed absolute/relative threshold matters for near-zero entries. A later
fixed-base archive had C++ compilation failures, not numerical disagreements;
neither historical passing nor failing runs certify the final release tip.

During collection, record maximum absolute error and relative tensor-norm error
per Hessian block against the double oracle on the same inputs, plus failure
counts. Compare equal represented inputs (fp32 values promoted for the oracle),
and separately document model-parameter rounding. Include seeded legal random
states beyond the existing zero/conservative smoke samples. Report actual
errors beside timings before claiming "close agreement with the fp64 reference
on the benchmark inputs." Do not generalize RNEA-Hessian tolerances to FD or FK
Hessians, which have different output scales and test criteria.

.. _figure-workflows:

Figure 2 — GRiD interface costs
-------------------------------

Two rows (RNEA, grad RNEA), three robot columns, five batch clusters. Compare
CUDA C++, NumPy/pybind, JAX and PyTorch using the **same generated kernels,
dtype, launch configuration, inputs and full outputs**. Primary comparison is
warmed host-to-host synchronized wall time for every interface. Record JAX and
PyTorch resident-call timings separately in the table; do not pretend NumPy's
host-array entry point provides the same resident interface.

Use colored matched native full-call reference plus a gray hatched
**incremental interface cost** for wrapper bars, only when the measured
difference is nonnegative and the underlying work matches. The cap includes
dispatch, conversions, and any extra allocation/copy behavior; it is not pure
Python execution time. The native reference already includes the transfers
required by this host-to-host contract. Unlike Figure 1, this hatch does not
represent the host-versus-resident delta; captions must state the distinction.

* [ ] Extend the existing ``baselines/grid/timeGRiD_bindings.py`` approach into
  a matched three-interface harness. It currently times JAX only, and its
  ``with_mem`` path does not establish a full host-output boundary.
* [ ] Keep JIT/compilation outside warm timing; force completion and materialize
  every promised output. Document allocations, cache policy and graph use.
* [ ] Call GRiD's explicit gradient API in each wrapper, not a scalar-loss VJP.
* [ ] Save numerical agreement, native/reference timings and raw wrapper totals;
  never fabricate a wrapper cap from unrelated historical kernel captures.

Full results table
------------------

Place it below the figures after stage 3. Include all ten operations, robots,
batches and selected baselines, with searchable/filterable rows and a download
of the underlying records. Columns: operation, robot/base/joints, batch, dtype,
baseline/version, differentiation method, CPU threads/GPU, resident mean,
host-to-host mean, run variability, matched-boundary speedup, validation/status,
and capture reference. Aggregate values must identify their population; never
count N/A as a win or infinity. FK/grad FK/Hessian FK require endpoint-frame and
output-parameterization agreement before any ratio is meaningful.

.. _figure-collisions:

Deferred collision collection
-----------------------------

No collision timing runs in stages 1–3. Start later with go2, frozen resolved
geometry, a fixed obstacle scene and self-pair table, and the same five batch
sizes. Compare single-fine-tier against coarse-to-fine GRiD checks; collect
latency, free/colliding/near-contact counts, fine-tier verdict agreement, and
complete geometry coverage. An external collision baseline is a separate scope
decision requiring matched geometry and distance/contact semantics.

The generic registry excludes the composite collision operation. Adapt and
validate a current timing wrapper first; the archived
``test/benchmarks/archive/collision_configfree_timing.cu`` is a starting point,
not a current ready benchmark. Reject unresolved assets. Agreement with GRiD's
fine representation does not establish exact-mesh accuracy or an independent
false-negative guarantee. Freeze seeds and representation resolution; report
them next to the figure. See the layout-only sketch in :doc:`plot_designs`.

Bug fixes and release gates
---------------------------

The timing parser no longer copies an average into a fictional median, and a
single summary no longer invents a zero-variance distribution. Python baseline
single-call medians are explicitly preserved as medians. The legacy latency
plot no longer substitutes zero for missing data or clamps negative overhead;
it marks unstackable totals and uses the actual B=256 reference index. Existing
captures are unchanged and must still be read as recorded means. The two JAX
resident examples now match host/resident final-state outputs, fence both
outputs, use equal repetition counts, and check agreement outside timing.
Those fixes are retained even though the rollout plot is dropped. CPU regression
tests exercise parsing and workload structure; GPU validation is still needed.

* [ ] Agree on comparison/precision/output contracts before GPU collection.
* [ ] Pin source/submodule SHAs, asset hashes, hardware/software versions, dtype,
  CPU threads, power/clocks, allocation policy, build/launch options, exact commands,
  warmups/repetitions and tolerances. Save logs and failure records.
* [ ] Publish immutable captures with SHA-256s and plotting revision, not just
  ignored local results. No historical preview becomes a release claim.
* [ ] Review stage 1 before widening collection; do not silently drop expensive
  or unfavorable cells. Time all selected baselines under the agreed protocol.
* [ ] Require a fresh full GPU validation receipt at the final release tip.
  Rerun affected timing cells after numerical or tuning changes.
* [ ] Replace homepage placeholders only with reviewed data. Once merged to
  ``main``, update the development clone command/banner and verify installation.
  The original paper stays tied to the archival repository.
