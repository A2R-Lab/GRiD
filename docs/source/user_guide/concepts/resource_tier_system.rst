Resource-Tier System (v2.0)
============================

**Status**: shipped in v2.0, with one explicitly deferred follow-up
(``TIER_LITE`` partial smem target, see "Deferred work" below).

**Audience**: inline-CUDA users (``#include "grid.cuh"`` from their own
kernel). The Python wrappers (``grid_rbd.RobotHandle``,
``grid_rbd.jax.JaxRobotHandle``) always use ``TIER_PERF`` by design.

What the tier system is
------------------------

Every emitted ``__global__`` kernel and every inline-callable
``_device``/``_inner`` function takes a non-type template parameter
``int RESOURCE_TIER`` (defaulting to ``TIER_PERF``). The tier picks a
``(launch_bounds, smem footprint, register cap)`` profile so an
inline-CUDA caller can fit a GRiD primitive into their outer kernel's
resource budget.

The three tiers:

.. list-table:: Tier semantics
   :header-rows: 1
   :widths: 18 22 22 38

   * - Tier
     - ``launch_bounds``
     - Register cap (sm_120)
     - Smem behavior
   * - ``TIER_PERF`` (default)
     - ``SUGGESTED_THREADS`` (288-512 per robot)
     - ~128-186 regs/thread
     - Full inner scratch lives in shared memory; current best perf.
   * - ``TIER_LITE``
     - ``min(2*SUGGESTED, 768)``
     - ~85 regs/thread
     - **Currently same as MINIMAL**: inner scratch routes to caller-
       provided ``s_workspace``. Distinct from MINIMAL only on the
       register axis. **Planned upgrade**: partial-spill with 48 KB
       smem target (see "Deferred work").
   * - ``TIER_MINIMAL``
     - ``1024`` (hardware cap)
     - ~64 regs/thread
     - Inner scratch routes entirely to ``s_workspace``. Smallest
       smem footprint; maximum block-size flexibility for tight
       outer kernels.

The register cap follows from ``regs_per_thread * max_threads <=
65536`` on sm_120: a tighter ``launch_bounds`` lets nvcc allocate
more registers per thread, a looser one forces it to budget for more
threads and use fewer registers each.

What's plumbed today
---------------------

The tier knob is exposed at the kernel level on every emitted
``*_kernel<T, RESOURCE_TIER>``. At the inline-CUDA ``_device`` /
``_inner`` level, these functions accept ``RESOURCE_TIER`` + a
caller-provided ``T *s_workspace`` argument:

* ``fdsva_so_inner<T, RESOURCE_TIER>(s_df2, s_idsva_so, s_Minv,
  s_df_du, s_XImats, s_temp, s_workspace, gravity)`` — 4*nv³ inner
  scratch routes between ``s_temp`` (PERF) and ``s_workspace``
  (LITE/MINIMAL).
* ``forward_dynamics_gradient_device<T, RESOURCE_TIER>(s_df_du,
  s_q, s_qd, [s_qdd, s_Minv | s_u], d_robotModel, gravity,
  s_workspace)`` — whole s_temp arena routes per tier.
* ``inverse_dynamics_gradient_device<T, RESOURCE_TIER>(s_dc_du,
  s_q, s_qd, [s_qdd], d_robotModel, gravity, s_workspace)`` — whole
  s_temp arena routes per tier.
* ``end_effector_pose_gradient_hessian_device<T, RESOURCE_TIER>
  (s_d2eePos, s_deePos, s_q, d_robotModel, s_workspace)`` —
  ``s_d2eeTemp`` slot (the 2*16*num_ees*n² portion) routes per tier;
  inner_no_d2 stays in smem at all tiers.
* ``idsva_so_device<T, RESOURCE_TIER>(s_idsva_so, s_q, s_qd, s_qdd,
  d_robotModel, gravity, s_workspace)`` — codegen-time frame
  dispatcher: ``body_frame_inner`` for fixed-base, ``world_frame_inner``
  for floating-base. Inner temp arena routes per tier.

Sizing constants — call these from host code to allocate the right
buffers:

.. code-block:: cpp

   template <typename T, int TIER = TIER_PERF>
   constexpr size_t FDSVA_SO_INNER_SMEM_BYTES();           // bytes for s_temp at TIER
   template <typename T, int TIER = TIER_PERF>
   constexpr size_t FDSVA_SO_INNER_WORKSPACE_BYTES();      // bytes for s_workspace at TIER

   // Same pattern: FD_DU_DEVICE_INLINE_*, ID_DU_DEVICE_INLINE_*,
   //               D2EE_DEVICE_INLINE_*, IDSVA_SO_DEVICE_INLINE_*

At ``TIER_PERF`` the SMEM_BYTES value matches current behavior
(the temp is in shared); at ``TIER_LITE``/``TIER_MINIMAL`` the
SMEM_BYTES value drops (temp moved out) and the WORKSPACE_BYTES
value covers the moved temp.

Inline-CUDA usage example::

   __global__ void my_outer_kernel(...) {
       extern __shared__ unsigned char s_arena[];
       // ... slice s_arena for your own buffers ...
       T *s_grid_temp = /* slice for GRiD primitive */;

       // At launch site we passed sizeof(s_grid_temp) =
       //   grid::FDSVA_SO_INNER_SMEM_BYTES<T, grid::TIER_MINIMAL>()
       //   == 0 at MINIMAL — no smem reserved for GRiD temp.
       // Workspace was malloc'd to:
       //   grid::FDSVA_SO_INNER_WORKSPACE_BYTES<T, grid::TIER_MINIMAL>()
       //   == 4 * NV^3 * sizeof(T) at MINIMAL.

       grid::fdsva_so_inner<T, grid::TIER_MINIMAL>(
           s_df2, s_idsva_so, s_Minv, s_df_du, s_XImats,
           /* s_temp */ nullptr,       // unused at MINIMAL
           /* s_workspace */ workspace, // global mem
           gravity);
   }

Design choices
---------------

**Why not three completely independent bodies per tier?**
Numerical equivalence: the math is identical at every tier;
``if constexpr`` branches only differ in pointer routing
(s_temp vs s_workspace). One body per algo, with up to two
pointer-routing branches. Less code duplication, fewer drift bugs.

**Why is JAX/Python locked to TIER_PERF?**
The Python wrapper persona is "sealed product, never touches
nvcc". They aren't fighting outer-kernel register/smem pressure
because they don't have an outer kernel. Exposing tier switching
through Python would add API surface without clear demand. The
host wrappers always launch ``*_kernel<T, TIER_PERF>`` (= current
behavior).

**Why no LITE smem target between PERF and MINIMAL today?**
Honest answer: implementation cost. The existing per-algo multi-
tier spill machinery (``fdsva_so`` has 4 levels, ``d2ee``/``id_du``/
``fd_du`` have 3, ``idsva_so_body_frame`` has 2) picks **one** spill
level at codegen time based on ``cuda_target_shared_mem_bytes``. To
make LITE pick a different level than PERF/MINIMAL we need to
emit three code paths and have codegen compute three picks per
algo. That work is deferred to the humanoid follow-up because:

1. On the current robot manifest (DOF ≤ 35), PERF already fits in
   ≤100 KB and MINIMAL fits in ≤workspace; a binary LITE/MINIMAL
   collapse is acceptable.
2. h1_2 (DOF ≥ 50) needs additional spill levels added to the
   existing machinery anyway — some algos overflow even at the
   current most-aggressive tier. Tuning new spill levels + tier
   targets together avoids re-doing the work.

What's in the framework but not yet exercised at LITE-distinct-from-MINIMAL:

* The ``gen_declare_shared_arena(tier_workspace_expr=...)``
  mechanism in ``GRiDCodeGenerator/helpers/_code_generation_helpers.py``
  is binary (PERF in smem / non-PERF in workspace). The follow-up
  extends it to ternary picks.

Deferred work — LITE 48 KB smem target
---------------------------------------

**Target design** (to ship alongside humanoid-scale support):

.. code-block:: cpp

   // Three smem targets the codegen picks against:
   constexpr size_t GRID_TIER_PERF_SMEM_TARGET    = /* cuda_target_shared_mem_bytes, ~100 KB */;
   constexpr size_t GRID_TIER_LITE_SMEM_TARGET    = 48 * 1024;       // configurable, default 48 KB
   constexpr size_t GRID_TIER_MINIMAL_SMEM_TARGET = /* ~XImats only */;

**Codegen-time per-algo selection**:

For algos with multi-level spill machinery
(``fdsva_so_use_global_tensors`` / ``fdsva_so_use_workspace_temp`` /
``fdsva_so_fd_grad_use_spill``, ``d2ee_use_workspace_temp`` /
``d2ee_use_workspace_d2xhom``, ``id_du_spill_tier`` /
``fd_du_spill_tier``, ``idsva_so_body_frame_use_global_output`` /
``idsva_so_body_frame_grav_full_spill``):

* For each tier in {PERF, LITE, MINIMAL}, pick the lowest-spill
  level whose ``py_arena_bytes`` is ≤ that tier's target.
* If three picks collapse to one (small robot, all fit PERF): emit
  one body, alias LITE/MINIMAL to PERF via ``using``.
* Else: emit per-tier ``if constexpr`` branches selecting the
  appropriate spill flags.

**Code paths affected** (rough):

* ``GRiDCodeGenerator/GRiDCodeGenerator.py:240-350`` — compute three
  spill picks per tiered algo instead of one.
* ``GRiDCodeGenerator/algorithms/_fdsva_so.py``,
  ``_eepose_gradient_hessian.py``, ``_forward_dynamics_gradient.py``,
  ``_inverse_dynamics_gradient.py``, ``_idsva_so.py`` — convert
  current single-tier ``if codegen_flag:`` blocks to per-tier
  ``if constexpr (RESOURCE_TIER == TIER_X)`` blocks.
* ``GRiDCodeGenerator/helpers/_code_generation_helpers.py`` —
  extend ``gen_declare_shared_arena(tier_workspace_expr=...)`` from
  binary to ternary (separate PERF / LITE / MINIMAL allocations).

**Co-design with humanoid (DOF ≥ 50)**:

For h1_2 (NV=51 fixed, 57 floating), several algos overflow the
sm_120 100 KB cap even at the current most-aggressive spill tier:

* ``h1_2_floating IDSVA_SO_B`` = 168 KB
* ``h1_2_floating FDSVA_SO`` = 244 KB
* ``h1_2_fixed FDSVA_SO`` = 178 KB
* ``h1_2_fixed EE_POSE_GRAD`` = 179 KB

These need new spill levels added (e.g. partial output-tensor
streaming, recompute-vs-cache trade-offs in inner functions, or
algorithmic recursion refactoring). When that work happens, the
LITE 48 KB target falls out as a natural intermediate level.

Inline-CUDA inner functions still needing tier plumbing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``idsva_so_body_frame_inner`` and ``idsva_so_world_frame_inner`` —
  caller currently controls ``s_temp`` location directly (no
  template). For humanoid use, callers may want a partial-spill
  knob the way ``fdsva_so_inner`` has one. Will need symmetric
  ``s_workspace`` plumbing.
* Some intermediates inside the SO body emitters that have
  heavily-aliased lifetimes — partial spill requires lifetime
  analysis to decide which buffers can safely go to workspace
  during which phase.

How it relates to other v2.0 work
----------------------------------

* :doc:`cublasdx_removal_design` — v2.0 set the stage by removing
  cuBLASDx and adding ``set_threads_per_block`` (up to
  SUGGESTED_THREADS). The tier system extends this to **above**
  SUGGESTED_THREADS via TIER_MINIMAL's ``launch_bounds=1024``.
* :doc:`codegen_architecture` — describes the four-layer emission
  (``_inner`` / ``_device`` / ``_kernel`` / host); tier templates
  live at the ``_inner`` / ``_device`` / ``_kernel`` layers.

References
-----------

* P1 baseline matrix:
  ``test/diagnostics/results/tier_baseline_sm120_rtx5090.md`` —
  per-(algo, robot) register + smem + spill data driving tier
  decisions.
* Smoke test: ``test/diagnostics/tier_instantiation_smoke.py`` —
  verifies all 9 single-overload kernels compile at all 3 tiers.
* Reusable arena helper:
  ``GRiDCodeGenerator/helpers/_code_generation_helpers.py:504-583``
  (``gen_declare_shared_arena``, ``tier_workspace_expr``).
