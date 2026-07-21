cuBLASDx Removal & Any-Thread-Count Library Functions
======================================================

**Status**: design — not yet executed.
**Archive tag**: ``archive/last-cublasdx`` (commit
``5177070``) marks the final state with full ``glass_nvidia``
(cuBLASDx-backed) support in mainline.

This document captures the rationale, scope, testing strategy, and
revert path for removing cuBLASDx from GRiD and making the generated
library functions thread-count-agnostic.

Why we are removing cuBLASDx
-----------------------------

Two user personas drive the change:

1. **Python-pinned users** call GRiD through ``grid_rbd.RobotHandle`` or
   ``grid_rbd.jax.JaxRobotHandle``. They never touch ``nvcc`` after
   ``register_robot``. For them GRiD is a sealed product and the
   thread-count question is invisible.
2. **CUDA-inline users** consume the generated ``grid.cuh`` directly,
   ``#include``-ing it in their own kernels (MPC solvers, OCP shooting
   pipelines, neural-network policy code with embedded dynamics). For
   them, the fact that every GRiD kernel pins
   ``__launch_bounds__(MAX_PERF_LEVEL_THREADS)`` and the cuBLASDx call sites
   ``static_assert`` on a minimum block size is a constant source of
   friction — it forces their outer kernel to either match GRiD's
   block shape or pay a host round-trip via the host wrappers.

We expect the CUDA-inline persona to be substantially more common
than the Python-pinned one in research settings. The decision to
optimize for them is the structural argument behind this change.

The empirical case for cuBLASDx ≪ SIMT on GRiD's call patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 2026-05-18 sweep with per-host-autotuned cuBLASDx (run via
``GLASS/bench/autotune.py --sm AUTO``) showed:

* Across **every** algorithm × robot × base in the iiwa14 / go2 / g1
  bench (24 cells total), ``glass_nv/glass = 0.98–1.01×``. Tuned
  cuBLASDx delivers no net benefit on GRiD's call mix.
* On the 4×4×4 batched GEMM inside ``end_effector_pose_hessian``,
  SIMT wins **by 2.6×** (see comment at
  ``test/benchmarks/run_multi_version.py:50``). cuBLASDx tile setup
  cost dominates at small shapes.

This finding is already encoded in the bench:

.. code-block:: python

   # All columns the sweep knows how to run. `glass_nvidia` (cuBLASDx-backed)
   # is intentionally NOT in DEFAULT_COLUMNS — the 2026-05-18 sweep + autotune
   # showed cuBLASDx loses to SIMT at every GEMM shape GRiD currently calls.

The structural argument
~~~~~~~~~~~~~~~~~~~~~~~~

GRiD's GEMM shapes are bounded by spatial algebra, not by robot DOF:

* 6×6 spatial inertia × spatial motion (composite-rigid-body recursion)
* 4×4×4 batched (EE-pose Hessian inner)
* 3×3 rotation composition

Higher-DOF robots do **more** small GEMMs, not bigger ones. The chain
propagation never accumulates into a single ``NJ×NJ`` GEMM — CRBA builds
the mass matrix entry-by-entry via inner products of 6-vectors. RNEA-grad
emits the ``M⁻¹ · dC/dU`` finishing step as a hand-rolled SIMT loop
(see ``_forward_dynamics_gradient.py:54-64``), not as a GEMM call.

The one genuine exception
~~~~~~~~~~~~~~~~~~~~~~~~~~

``fdsva_so_inner``'s final contraction is structurally an ``iL,Ljk → ijk``
tensor operation with FMA count growing as ``4·NV⁴``:

================  ==========  ========================
Robot             NV          fdsva_so contraction
================  ==========  ========================
iiwa14_fixed       7          ~10K FMAs
g1_fixed          29          ~2.8M FMAs
g1_floating       35          ~6M FMAs
h1_2_fixed        51          ~27M FMAs
h1_2_floating     57          ~42M FMAs
================  ==========  ========================

Standalone autotune says cuBLASDx wins 2.4–5.2× on this shape at
``n = 24–48``. In-kernel, the comment at ``_fdsva_so.py:75-93`` lists
four reasons the standalone win has not been realized: register
pressure with all SO state live, ``iL,Ljk`` strides that aren't
GEMM-friendly, tier pressure (``g1_floating`` is already in spill
tier), and per-block-per-timestep sync overhead. Realizing the
standalone gain would require a substantial layout refactor with
uncertain in-kernel payoff.

Inline-CUDA users are not going to do that layout dance inside their
own kernels. Keeping cuBLASDx alive to serve a code path that no
inline user will exercise — and that the Python users don't see —
is dead weight. Section *Open research questions* below preserves
this as a future-self target.

Limb-parallel opportunities (also SIMT-friendly)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Humanoid and quadruped topologies have independent limb subtrees. The
gradient and SO output tensors are block-sparse: ``∂c_i/∂q_j = 0`` when
joints ``i`` and ``j`` live in different limb subtrees without a
shared dependency path. For h1_2 at NV=51, naive dense ``(NV, NV)`` =
2601 entries; structurally non-zero is closer to ~700–900.

Two follow-on codegen opportunities sit here, both SIMT-friendly:

* **Output sparsity**: skip zero-block writes in gradient / SO emitters.
* **Limb-batched dispatch**: pack 4–6 independent limbs' spatial ops
  into batched-small-matrix SIMT primitives.

Both are research-grade codegen refactors. Neither motivates cuBLASDx —
the resulting shapes (5 × 6×6 spatial ops, or 5 × 10×10 inner-products)
are squarely in the small-batch small-matrix regime where SIMT remains
competitive or winning.

What's changing
----------------

The change splits into two coupled sub-rips:

A. **cuBLASDx dispatch removal** — strip the ``GRID_LINALG_GLASS_NVIDIA``
   backend everywhere it's invoked from GRiD codegen. **Leave GLASS itself
   intact** — GLASS as a library keeps its cuBLASDx support, we just stop
   calling into it. Smaller blast radius, preserves GLASS's value as a
   standalone library.
B. **Any-thread-count emission** — drop the ``__launch_bounds__``
   *attribute* from every emitted kernel while keeping the
   ``MAX_PERF_LEVEL_THREADS`` *constant* alive as a true caller hint (the
   value still encodes the codegen's preferred DOF-aware,
   warp-rounded block size; it just stops being enforced). Parameterize
   ``g_thread_dimms`` in the wrapper, and convert ``threadIdx.x < N``
   guards in ``_inner`` functions to block-stride loops via the
   existing helper at
   ``grid_codegen/helpers/_code_generation_helpers.py:87-88``.

(B) becomes mechanical once (A) is done — without the cuBLASDx ``static_assert``
floor, there's no second mode to emit, and the only remaining work is the
``_inner`` refactor + wrapper parameterization.

Codegen layer
~~~~~~~~~~~~~~

* :file:`grid_codegen/GRiDCodeGenerator.py` — drop
  ``enable_cublasdx`` / linalg-backend kwargs from ``__init__``; drop
  emission of ``GRID_LINALG_GLASS_NVIDIA`` / ``GRID_CUBLASDX_HEADER_AVAILABLE``
  / ``GRID_CUSOLVERDX_HEADER_AVAILABLE`` macros into ``grid.cuh``.
  Simplify the ``MAX_PERF_LEVEL_THREADS`` computation (no longer pinned to
  cuBLASDx minimum).
* :file:`grid_codegen/helpers/_lin_alg_helpers.py` — single-path
  SIMT emission. Drop the ``DEFINE_NVIDIA_GEMM_BLOCKDIM`` /
  ``DEFINE_NVIDIA_GEMV_BLOCKDIM`` machinery and the ``if constexpr``
  dispatch at every call site. This file gets substantially smaller.
* :file:`grid_codegen/helpers/_code_generation_helpers.py` — drop any
  backend-conditional branches.
* Per-algorithm files that branch on backend:
  ``_forward_dynamics.py``, ``_eepose_gradient_hessian.py``, ``_fdsva_so.py``.
  Each gets a focused drop of the backend-conditional code paths.

Generated code (effect, not direct edits)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the codegen-layer change, regenerated ``grid.cuh`` files no
longer carry the ``GRID_CUDA_LINALG_BACKEND`` switch nor any cuBLASDx
includes. Files using ``__launch_bounds__`` keep the attribute at first;
the second sub-rip (any-thread-count) drops it.

Wrapper layer (Python + JAX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :file:`bindings/grid_rbd/_compile.py` — remove libmathdx / MATHDX_ROOT
  discovery, drop the ``-DGRID_CUDA_LINALG_BACKEND`` ``nvcc`` flag, drop
  the ``-rdc=true`` requirement (cuBLASDx is the only reason for it).
* :file:`bindings/grid_rbd/wrapper_template.cu` — replace the hardcoded
  ``g_thread_dimms = dim3(MAX_PERF_LEVEL_THREADS, 1, 1)`` with either a
  caller-provided ``threads_per_block`` parameter (preferred) or a
  named alias preserving current behavior.
* :file:`bindings/grid_rbd/__init__.py`, :file:`bindings/grid_rbd/jax/__init__.py` —
  surface the simplification if exposed in the public API (currently
  hidden; no change expected).

The JAX FFI handlers (the 12 we just landed in v0.3) launch kernels
directly without going through the host wrappers. They read
``g_thread_dimms`` for their launch shape — same change as above.

Bench harness
~~~~~~~~~~~~~~

* :file:`test/benchmarks/baselines/grid/run.py` — drop
  ``--linalg-backend`` choice, ``--mathdx-root``, ``cublasdx_sm_from_arch``,
  ``resolve_mathdx_root``, ``with_cusolverdx``.
* :file:`test/benchmarks/run_multi_version.py` — drop ``glass_nvidia``
  from ``COLUMNS``, ``COLUMN_TO_BASELINE_KEY``, ``DEFAULT_COLUMNS``
  (no-op since it's already excluded by default). Drop the
  ``--cicc-opt-level`` workaround (was for cuBLASDx-induced cicc hangs).
* :file:`test/benchmarks/generate_report.py` — drop the ``glass_nv``
  column rendering and the ``glass_nv/glass`` ratio.
* :file:`test/benchmarks/run_benchmarks.py`,
  :file:`test/benchmarks/run_overnight_sweep.sh` — sweep for stale CLI flags.
* Existing reports in ``test/benchmarks/`` (especially
  ``benchmark_multi_version_sm120_5090_full.md``) — **archive, do not
  delete**. They are the final record of glass_nv numbers.

Documentation
~~~~~~~~~~~~~~

* :file:`docs/source/user_guide/getting_started/installation.rst` — drop
  libmathdx install section, drop MATHDX_ROOT environment-variable docs,
  drop cuBLASDx EULA notes.
* :file:`docs/source/user_guide/getting_started/docker_setup.rst` — sweep
  for libmathdx references.
* :file:`docs/source/user_guide/concepts/codegen_architecture.rst` —
  update the thread-count constraint section (no longer cuBLASDx-bound;
  fully thread-flexible after any-thread-count lands).
* :file:`docs/source/user_guide/tutorials/benchmarks.rst`,
  :file:`docs/source/user_guide/tutorials/cuda_validation.rst`,
  :file:`docs/source/user_guide/tutorials/python_wrappers.rst` —
  remove ``glass_nvidia`` mentions; update install line.
* :file:`README.md`, :file:`bindings/README.md` — sweep.
* Any local hardware-specific sweep-results notes — historical, leave
  intact; add a pointer to this design doc.

Install / build system
~~~~~~~~~~~~~~~~~~~~~~~

* :file:`install/base_install.sh`, :file:`install/developer_install.sh` — drop libmathdx
  setup steps if present.
* :file:`pyproject.toml`, :file:`bindings/pyproject.toml`,
  :file:`setup.py` (repo root) — drop cuBLASDx-related extras if any.
* GitHub Actions / CI — drop ``--mathdx-root`` or
  ``GRID_CUDA_LINALG_BACKEND`` from any workflow.

Tests
~~~~~~

* :file:`test/cuda_equivalents/test_cuda_executable_equivalence.py`,
  :file:`test/cuda_equivalents/test_cuda_codegen_layout.py` — drop
  parametrize entries pinning ``glass_nvidia``; otherwise no change
  (the SIMT path is what the tests exercise by default).
* :file:`test/cuda_equivalents/test_cuda_second_order_fallback.py` — the
  ``GRID_CUDA_SECOND_ORDER_TEST_THREADS`` env-var scaffold at
  lines 135-147 is reused for any-thread-count coverage in (B).
* :file:`test/python_wrappers/*` — no changes expected; already
  SIMT-only and thread-count-agnostic at the test level.
* :file:`RBDReference/equivalents/* + RBDReference/tests/*` — no backend dependency; no
  change.

GLASS submodule
~~~~~~~~~~~~~~~~

**Not touched.** GLASS as a library keeps its cuBLASDx code paths
(``GLASS/glass-nvidia.cuh``, ``GLASS/src/nvidia/``,
``GLASS/bench/autotune.py``, the tuning tables). External users of
GLASS for non-RBD GEMM workloads may legitimately want cuBLASDx.
Removing the integration at the GRiD side is the right separation.

Testing strategy
-----------------

The change must produce zero regression in **accuracy** and zero
regression in **timing** versus the existing ``glass`` (SIMT) column.

Accuracy
~~~~~~~~~

Existing suites cover this thoroughly. Re-run after each phase:

.. code-block:: bash

   PYTHONPATH=. .venv/bin/pytest \
       test/python_wrappers/ \
       RBDReference/tests/ \
       test/cuda_equivalents/ \
       -q

Expected: identical pass set to mainline. The Python wrapper tests
(60 tests in ``test_iiwa14_smoke.py`` + ``test_iiwa14_jax_smoke.py``)
already cross-check every method against ``RBDReference`` at float32
precision. The Pinocchio equivalence suite covers floating-base
conventions and SO algorithms.

For added confidence: re-run the **same** test inputs against the
``archive/last-cublasdx`` tag and diff outputs byte-for-byte. Any
deviation outside float32 noise (>5e-5 absolute) is a regression.

Timing
~~~~~~~

The ``glass`` column from the existing
``benchmark_multi_version_sm120_5090_full.md`` is **the SIMT-only
path**. The post-rip timings must match it within sweep noise (typically
±2-3% per cell). Re-run after the codegen + wrapper sub-rip lands:

.. code-block:: bash

   .venv/bin/python test/benchmarks/run_multi_version.py \
       --columns glass --robots iiwa14 go2 g1 \
       --bases fixed floating \
       --single-call-iters 50000 --batch-iters 500 \
       --output-dir test/benchmarks/results/comparison_post_cublasdx_removal \
       --report test/benchmarks/benchmark_post_cublasdx_removal.md

Diff against the archived full report:

* Per-cell ratio within ±3% across all 24 (robot × base × algorithm) cells.
* No new SKIPPED cells.
* No new "compile failed" cells.

Any-thread-count validation (after sub-rip B lands)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extend the existing
``test/cuda_equivalents/test_cuda_second_order_fallback.py`` scaffold
to cover every algorithm at block sizes
``{64, 128, 256, MAX_PERF_LEVEL_THREADS, 512}``. Output must match
``glass``-column reference at all block sizes within float32 precision.

Add a microbench at ``test/benchmarks/any_thread_count_microbench.py``
that times RNEA at each block size for iiwa14, go2, g1, h1_2 and
reports the perf trajectory. Saved as a committed artifact (not /tmp)
so future-self can re-run after changes.

Revert path
------------

If we ever need to restore cuBLASDx support:

1. The archive tag ``archive/last-cublasdx`` (commit ``5177070``)
   points at the final pre-rip state. ``git show
   archive/last-cublasdx -- <path>`` gives the exact pre-rip content
   of any file.
2. The list of touched files is captured in the "What's changing"
   section above. Restoration is a re-apply, not a ``git revert`` (the
   intervening codegen will have evolved).
3. The autotune data lives at
   ``GLASS/bench/tuning/plancher-omen-26.cuh`` and
   ``GLASS/bench/tuning/plancher-omen-26_results.md`` (sm_120 / 5090
   RTX), kept in-repo as historical record.
4. The fdsva_so contraction analysis at
   ``grid_codegen/algorithms/_fdsva_so.py:75-93`` (archived) is
   the most useful starting point for anyone reviving cuBLASDx for the
   one shape where it might pay off.

Cache invalidation
-------------------

No external users to migrate (no PyPI publish, no CI dependents since
the GLASS rollout), so the rip is **clean — no deprecation cycle, no
version bump**:

* Hard-remove ``GRID_CUDA_LINALG_BACKEND``, ``GRID_LINALG_GLASS_NVIDIA``,
  ``GRID_CUBLASDX_HEADER_AVAILABLE``, and ``GRID_CUSOLVERDX_HEADER_AVAILABLE``
  from the emitted ``grid.cuh``.
* Package version stays at ``1.0.0`` (it has never been published).
* Any locally-cached ``.so`` files become stale. The
  :py:func:`grid_rbd._cache.compute_cache_key` formula already mixes in
  a hash of ``wrapper_template.cu``, so wrapper edits invalidate
  naturally. For the codegen-output change in A2, the only affected
  user is the maintainer, who can ``rm -rf ~/.cache/grid-rbd`` once.
* No no-op alias, no deprecation warning.

Open research questions (deferred backlog)
-------------------------------------------

These were considered and excluded from the current rip. Captured so
the next person doesn't have to re-derive them.

1. **fdsva_so iL,Ljk contraction layout refactor.** The one place
   where re-introducing cuBLASDx might pay off, at high DOF only.
   Standalone autotune says 2.4-5.2× win at n=24-48; in-kernel
   constraints (layout, register pressure, tier pressure, sync) need
   resolving. ~1-2 weeks of focused work; uncertain in-kernel payoff.
   Likely worth it only with a humanoid-scale paper target.
2. **Limb-parallel output sparsity in gradient / SO tensors.** Skip
   zero-block writes in dC/dQ, d²c/dq dq', d²qdd/dq dq' for branched
   topologies. SIMT-only. ~2-3 weeks across all gradient + SO
   emitters. More certain payoff at high DOF (h1_2: estimated 2-3×
   reduction in output writes for SO methods).
3. **Limb-batched SIMT dispatch.** Pack per-limb spatial ops into
   batched-small-matrix SIMT primitives. Architectural shift in the
   codegen propagation patterns; speculative payoff.
4. **Per-host autotune integration for glass_nvidia.** Already built
   (``GLASS/bench/autotune.py``), produced the data motivating this
   removal. Archived dormant.

Execution phases
-----------------

Each phase is a self-contained commit with passing tests. Phase 1
(this commit) is already in.

1. **Tag the archive** — ``archive/last-cublasdx`` at the pre-rip head.
   *Done at commit 5177070.*
2. **Codegen rip (A1)** — strip backend dispatch from ``_lin_alg_helpers.py``,
   ``GRiDCodeGenerator.py``, and the per-algorithm files. Regenerate
   iiwa14 ``grid.cuh`` to verify clean SIMT output. Run accuracy suite.
3. **Wrapper rip (A2)** — drop libmathdx detection from ``_compile.py``,
   drop ``g_thread_dimms`` hardcode in ``wrapper_template.cu`` (use
   ``MAX_PERF_LEVEL_THREADS`` for now; parameterize in B2). Run wrapper +
   JAX tests.
4. **Bench rip (A3)** — drop ``glass_nvidia`` column from harness,
   drop ``--mathdx-root``, drop ``--cicc-opt-level``. Re-run
   ``--columns glass`` and confirm parity with archived sweep.
5. **Docs rip (A4)** — sweep installation.rst, codegen_architecture.rst,
   READMEs, tutorials. Sphinx ``make html`` clean.
6. **Install scripts rip (A5)** — strip libmathdx from install/base_install.sh /
   install/developer_install.sh.
7. **Install scripts (A6)** — strip libmathdx setup from base/developer
   install scripts; no version bump (never published).
8. **Any-thread-count emission (B1)** — drop the ``__launch_bounds__``
   attribute from every kernel emitter (keep the ``MAX_PERF_LEVEL_THREADS``
   constant emission as a documented hint). Parameterize
   ``threads_per_block`` in ``wrapper_template.cu`` defaulting to
   ``grid::MAX_PERF_LEVEL_THREADS``. Convert ``threadIdx.x < N`` guards in
   ``_inner`` functions to block-stride loops (use the existing
   helper).
9. **Any-thread-count tests (B2)** — extend
   ``test_cuda_second_order_fallback.py`` to cover all algorithms at
   block sizes {64, 128, 256, MAX_PERF_LEVEL_THREADS, 512}. Add the
   microbench artifact.
10. **Release notes (C)** — short changelog entry noting the rip and
    pointing at the archive tag + this design doc. No migration story
    needed (never published).

Estimated total: 2-3 days of focused work plus ~1 day of bench /
validation. Phases 2-5 are largely mechanical (file-by-file deletion);
phases 8-9 are the only ones with new code.

Adjacent considerations
------------------------

* **Branch hygiene.** This sub-rip lives on a dedicated
  ``cublasdx-removal`` branch off ``modernizing-tests``. When the
  phase sequence is complete and tests pass, merge
  ``cublasdx-removal → modernizing-tests``, then
  ``modernizing-tests → main``. The archive tag is reachable from any
  of those branches.
* **No public-API breakage for Python users.** ``RobotHandle`` and
  ``JaxRobotHandle`` keep their full method surface. The only visible
  change is faster ``register_robot`` times (no libmathdx discovery,
  no ``-rdc=true``).
* **MAX_PERF_LEVEL_THREADS keeps its name and value.** It moves from
  "enforced launch bound" to "recommended block size hint", which is
  what the name promises. Generated ``grid.cuh`` keeps the constant,
  the host wrappers still default to it, and external CUDA-inline
  users get a useful starting point if they don't have a reason to
  pick something else.

See also
---------

* :doc:`codegen_architecture` — the four emission layers; thread-count
  constraint section gets updated as part of phase 5.
* ``test/benchmarks/run_multi_version.py:50-56`` — the original
  comment that captured the empirical case.
* ``grid_codegen/algorithms/_fdsva_so.py:75-93`` — the one place
  where the open research question lives.
