Resource-Tier System (v2.0)
============================

**Status**: shipped in v2.0 + Phase 3a/b/c/d/e spill machinery + L2 pinning
default-on. Inner-controlled placement refactor (below) implemented and
numerically validated for fdsva_so/Minv/FD/ABA/EE_GRAD. Per-tier surgical
spill now also lands for **idsva_so (body + world frame)** and the
**time-integrator value + gradient** kernels — see
:doc:`resource_tier_changelog`. The global scratch arena is named ``d_workspace`` (device memory);
earlier revisions of this doc called it ``d_global_temp``.

**Audience**: inline-CUDA users (``#include "grid.cuh"`` from their own
kernel). The Python wrappers (``grid_rbd.RobotHandle``,
``grid_rbd.jax.JaxRobotHandle``) launch each kernel at its PER-ALGO BAKED
tier — ``grid::launch_cfg<GRID_ALGO_*>::TIER``, autotuned into
``config/launch_configs/<robot>/<gpu>.json`` and baked at codegen. On a
tuned robot many algos run at ``TIER_LITE``/``TIER_MINIMAL``; an untuned
robot (or algo) falls back to ``TIER_SHARED``.



.. note::
   This page is the REFERENCE (definition, memory model, exposed constants,
   coverage). The rationale essays live in :doc:`resource_tier_design_notes`
   and the shipped-work log in :doc:`resource_tier_changelog` (split from this
   page 2026-09-09; it used to bury the definition 400 lines deep).

What the tier system is
------------------------

Every emitted ``__global__`` kernel and every inline-callable
``_device``/``_inner`` function takes a non-type template parameter
``int RESOURCE_TIER`` (defaulting to ``TIER_SHARED``). The tier picks a
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
   * - ``TIER_SHARED`` (default)
     - ``MAX_PERF_LEVEL_THREADS`` (288-512 per robot)
     - ~128-186 regs/thread
     - Full inner scratch lives in shared memory; current best perf.
   * - ``TIER_LITE``
     - ``min(2*SUGGESTED, 768)``
     - ~85 regs/thread
     - Picks the lowest spill rung that fits the ~48 KB LITE smem target
       (``cuda_target_lite_shared_mem_bytes``), clamped to be ≥ the SHARED
       rung. On robots/algos with multi-rung ladders this is now a
       genuinely intermediate spill level, not an alias of MINIMAL.
   * - ``TIER_MINIMAL``
     - ``1024`` (hardware cap)
     - ~64 regs/thread
     - Inner scratch routes entirely to ``d_workspace``. Smallest
       smem footprint; maximum block-size flexibility for tight
       outer kernels.

The register cap follows from ``regs_per_thread * max_threads <=
65536`` on sm_120: a tighter ``launch_bounds`` lets nvcc allocate
more registers per thread, a looser one forces it to budget for more
threads and use fewer registers each.

Who this is for (read this first)
---------------------------------

GRiD is, at its core, a **code generator for power users** — people who want
to call hand-tuned, robot-specialized rigid-body-dynamics kernels directly
from their own CUDA code and squeeze every cycle and byte out of the GPU.
Everything below the convenience layer is built for that person.

But you do **not** have to be that person to use GRiD. We deliberately ship a
ladder of entry points, from "one line, no GPU knowledge required" up to
"hand me the raw block-parallel device routine and I'll manage the shared
memory myself." Pick the rung that matches how much control you need:

.. list-table:: Entry points, easiest to most powerful
   :header-rows: 1
   :widths: 22 20 58

   * - You want…
     - Use…
     - You manage…
   * - Just the answer, from Python
     - ``grid_rbd.RobotHandle`` / ``grid_rbd.jax.JaxRobotHandle``
     - Nothing. Arrays in, arrays out. Tier comes from the per-algo baked
       launch config (``TIER_SHARED`` fallback when untuned).
   * - The answer, from C++/CUDA host code
     - ``grid::<algo>(hd_data, ...)`` **host** wrapper
     - Nothing on-device. The wrapper does H2D/D2H copies, picks
       launch dims, sets shared-mem attributes, launches the kernel.
   * - A kernel to drop into your own launch
     - ``grid::<algo>_kernel<T, TIER>`` **__global__**
     - The launch (grid/block dims, dynamic-smem bytes, streams) and
       the per-trajectory batch loop is done for you inside.
   * - A block-parallel routine to call **inside** your own kernel
     - ``grid::<algo>_inner<T, PLACEMENT>`` / ``_device`` **__device__**
     - Everything: shared-memory arenas, scratch placement, syncs.
       This is the real engine; the layers above are conveniences.

If you are new, start at the top of that table and ignore the rest of this
document — the ``RobotHandle`` tutorial is all you need. If you are here to
fight for occupancy inside a fused planning/MPC/learning kernel, read on: the
rest of this page documents the full machinery so you can drive it directly.


Smart memory: shared vs. global, spills, and L2 pinning
--------------------------------------------------------

The other half of the inner's intelligence is the **memory hierarchy**. An
inner's working set is a mix of:

* **inputs/outputs** — supplied by the caller (you decide where these live);
* **persistent scratch** — needed across the whole routine;
* **transient scratch** — needed only within a phase, freely reused.

On a GPU these can live in shared memory (fast, scarce — ~48 KB default /
~100 KB opt-in per block on sm_120) or global memory (abundant, slower, but
**L2-pinnable**). The inner decides, per buffer, where each goes — and that
decision is exposed as a **compile-time placement parameter** so the caller
can pick a profile that fits *their* outer kernel's pressure.

**Placement is the inner's job, not the kernel's.** Each inline-callable inner
is keyed on a placement template parameter and chooses ``s_temp`` (shared) vs.
``d_workspace`` (global) for each spillable buffer *at the top of the
function*. The caller (kernel, device wrapper, or your own code) is a thin
shim: it sizes both arenas from the exposed constants, hands both pointers in,
and passes the placement. A surgical-spill change — moving one more buffer to
global, or splitting a buffer hot/cold — is therefore **local to the inner**:
repoint a sub-buffer and update its size constant, with no kernel edit. This
is what makes the spill machinery tractable to evolve.

Placement parameters currently emitted (all default to "in shared memory" so
existing call sites are unchanged):

.. list-table:: Inner placement parameters
   :header-rows: 1
   :widths: 26 24 50

   * - Inner
     - Placement param
     - Buffer it routes
   * - ``minv_inner``
     - ``bool F_IN_SMEM``
     - the 6·NV² articulated-body F-region
   * - ``forward_dynamics_inner``
     - ``bool MINV_F_IN_SMEM``
     - the internal Minv F-region (FD no longer takes it as a param)
   * - ``aba_inner``
     - ``bool TEMP_IN_SMEM``
     - the 140·NJ+ recursion scratch band
   * - ``end_effector_pose_gradient_inner``
     - ``bool TEMP_IN_SMEM``
     - the double-buffered kinematic-chain workspace
   * - ``fdsva_so_inner``
     - ``bool SCRATCH_IN_SMEM``
     - the 4·NV³ contraction scratch
   * - ``integrator_inner``
     - ``bool MINV_F_IN_SMEM``
     - the FD inner's Minv F-region (value path); see "Integrator surgical spill"
   * - ``integrator_gradient_kernel``
     - per-tier rung (Dqdd / dAB / inner level)
     - composes inverse_dynamics_gradient selective/global_temp + spills Dqdd & the dAB output
   * - ``*_device`` (inverse_dynamics_gradient / forward_dynamics_gradient / idsva_so / end_effector_pose_hessian)
     - ``int RESOURCE_TIER``
     - whole inner ``s_temp`` arena (via the ``tier_workspace_expr`` helper)

**Spilled global memory is L2-pinned.** Most spilled buffers are
recursion-hot (touched every BFS step), so a naive spill to HBM would be a
perf cliff. With L2 persistence enabled by default
(``GRID_CUDA_ENABLE_L2_PERSISTING=1``), the generated workspace is pinned in
L2 for the kernel's lifetime, so a spilled access costs roughly an L2 hit
rather than an HBM round-trip. The cost of a spill is then bounded by the
shared-vs-L2 latency gap, not the shared-vs-HBM gap.

**Spill levels and the per-robot tier→level map.** Each algorithm has a fixed
*menu* of spill levels (level 0 = everything in shared memory; higher levels
progressively move buffers to ``d_workspace``). Which level a given
``RESOURCE_TIER`` maps to is decided **at code-generation time, per robot**,
based on what actually fits the smem budget for that robot. Small robots
(e.g. iiwa14, go2) keep every tier at level 0 — there is nothing to spill, so
the tiers share the **same spill placement** and emit a single kernel *body*.
Note this does **not** make them byte-identical SASS: each tier still carries
its own ``__launch_bounds__(tier_max_threads<TIER>())`` (SHARED =
``MAX_PERF_LEVEL_THREADS``, LITE = ``min(2×, 768)``, MINIMAL = ``1024``), so
ptxas budgets a different register cap per tier and can produce different
register counts / SASS even when the placement is identical. For algorithms
with **no** smem spill at all (e.g. ``inverse_dynamics``) the launch_bounds is
in fact the *only* thing that differs between tiers — which is exactly why a
looser-bounds tier can run *slower* via register starvation (see the autotune
A.7 note). Large robots (e.g.
h1_2) map the lower tiers to deeper spill levels. Because the mapping is
per-robot, **multiple tiers can share a level**, and the generated
``*_IN_SMEM<TIER>()`` / ``*_SCRATCH_IN_SMEM<TIER>()`` constexprs expose
exactly which placement each tier resolves to for the robot you generated.

This is the reconciliation of two goals that look opposed: the inner stays
self-contained and decidable from a single placement flag (good for inline
reuse and for evolving spills), while the *choice* of flag per tier is a
per-robot, fit-driven decision made once at codegen time (good for not paying
for a spill you don't need).

**This is the required standard for every algorithm, not a per-kernel option.**
New algorithms and refactors must put the spill decision *in the inner* (a
placement template + ``if constexpr`` selecting ``s_temp`` vs ``d_workspace`` at
the top), never repoint the inner's scratch from the kernel/host caller. A
caller that aliases or reassigns ``s_temp`` from the outside is a smell to
migrate. Two consequences worth calling out:

* **Composing kernels inherit spills for free.** ``fdsva_so`` embeds the
  ``idsva_so`` inner; once that inner owns its placement, ``fdsva_so`` spills the
  (dominant) idsva_so scratch just by passing ``SCRATCH_IN_SMEM=false`` — no
  pointer surgery in ``fdsva_so`` — and any future *surgical* (cold-only) idsva_so
  spill propagates to ``fdsva_so`` automatically.
* **The XImats/XmatsHom load helper is a separate scratch user.** It is called
  from the kernel *outside* the inner and dereferences ``s_temp`` for its
  ``2*num_pos`` sincos scratch, so a spilled (null) ``s_temp`` crashes it. Handle
  this uniformly: either keep the tiny sincos scratch in smem always, or repoint
  ``s_temp`` at the workspace before the helper call. (See the null-``s_temp``
  fix history for ``aba``/``end_effector_pose_gradient``.)

Conformance audit and the remaining migration list (``idsva_so`` world inner,
``inverse_dynamics_gradient``/``forward_dynamics_gradient``/``integrator_gradient``
kernel-side repoints) live in
``docs/idsva_so_inner_refactor_notes.md`` — that table is the source of truth for
propagating this pattern across the project.


Block-wide parallelism in the inners
------------------------------------

The inners are written to saturate the **whole thread block**, not a fixed
lane count. Every parallel region is emitted as a **block-stride loop** of the
form::

    for (int i = threadIdx.x + threadIdx.y * blockDim.x;
         i < WORK_ITEMS;
         i += blockDim.x * blockDim.y) { ... }

This has several deliberate consequences that power users rely on:

* **Correct at any block size.** The same generated routine runs correctly
  whether you launch it with 32 threads or 1024. Work items are distributed
  across however many threads the block has; there are no hard-coded lane
  assumptions and no out-of-bounds writes when ``blockDim`` does not divide
  the work evenly. (Audited: every parallel write in the emitted code is
  inside a block-stride loop — there are no bare ``threadIdx``-indexed stores.)
* **You choose the occupancy/latency trade.** Because the routine adapts to
  the launch, you can tune block size for your fused kernel's occupancy
  without regenerating anything. The tier system's ``__launch_bounds__`` only
  bounds the *maximum* threads (to control the register budget), it does not
  fix the launch.
* **Maximal parallelism by construction.** Each algorithm exposes its
  natural parallel width (e.g. per-(i,j,k) tensor elements, per-DOF columns,
  per-body 6×6 blocks) directly as the loop bound, so a large block fills with
  useful work rather than idling. Where the recursion structure forces
  seriality (e.g. the BFS sweeps), the parallel regions sit between syncs and
  still use the full block.

The practical upshot: the inner is the unit of parallelism. You bring the
threads; it uses all of them.


Exposed sizing + placement constants (power-user reference)
-----------------------------------------------------------

For every spillable inner, codegen emits a matched trio so you can allocate
correctly and know what codegen chose. Using fdsva_so as the template::

    // bytes to reserve in shared memory for this placement
    template <typename T, bool SCRATCH_IN_SMEM = true>
    constexpr size_t FDSVA_SO_INNER_SMEM_BYTES();

    // bytes to reserve in (L2-pinned) global memory for this placement
    template <typename T, bool SCRATCH_IN_SMEM = true>
    constexpr size_t FDSVA_SO_INNER_WORKSPACE_BYTES();

    // the placement codegen assigned to each tier, for THIS robot
    template <int TIER>
    constexpr bool FDSVA_SO_SCRATCH_IN_SMEM();

The same trio is emitted for the other converted algorithms, with the
placement bool named for the buffer it controls:

* ``MINV_INNER_{SMEM,WORKSPACE}_BYTES<T, F_IN_SMEM>`` + ``MINV_F_IN_SMEM<TIER>``
* ``FD_INNER_{SMEM,WORKSPACE}_BYTES<T, MINV_F_IN_SMEM>`` + ``FD_MINV_F_IN_SMEM<TIER>``
* ``ABA_INNER_{SMEM,WORKSPACE}_BYTES<T, TEMP_IN_SMEM>`` + ``ABA_TEMP_IN_SMEM<TIER>``
* ``EE_GRAD_INNER_{SMEM,WORKSPACE}_BYTES<T, TEMP_IN_SMEM>`` + ``EE_GRAD_TEMP_IN_SMEM<TIER>``

The ``*_device`` inline entry points (inverse_dynamics_gradient / forward_dynamics_gradient / idsva_so / end_effector_pose_hessian) still
expose their sizing as ``*_DEVICE_INLINE_{SMEM,WORKSPACE}_BYTES<T, TIER>``
(keyed on tier rather than a placement bool); they decide placement internally
via the ``tier_workspace_expr`` arena helper, and their kernels inline + spill
at the kernel level by design. Converting those kernels to call
placement-deciding inners is a tracked follow-up.

Rule of thumb for an inline call:

#. Pick a ``TIER`` (or call ``<ALGO>_..._IN_SMEM<TIER>()`` to see the
   placement it resolves to for your robot).
#. Reserve ``..._INNER_SMEM_BYTES<T, placement>()`` in your block's dynamic
   shared memory for the primitive's ``s_temp``.
#. ``cudaMalloc`` (once) ``..._INNER_WORKSPACE_BYTES<T, placement>()`` per
   concurrently-resident block for ``d_workspace`` (0 when the placement keeps
   everything in shared). Pin it in L2 if you spill (see ``grid_begin_l2_persisting``).
#. Call ``<algo>_inner<T, placement>(..., s_temp, d_workspace, ...)``.

**Testing status of the refactor**: all five converted algos
(fdsva_so / Minv / FD / ABA / EE_GRAD) compile clean at all three tiers across
iiwa14 / go2 / h1_2 (fixed) + g1 (floating); the tier-instantiation smoke
passes. Numerical equivalence (``cuda_equivalence``) and the per-tier perf
sweep are pending a joint testing session — the Minv/FD arena layout changed
(no_F-then-F instead of F-then-no_F), so equivalence is the gating check.

What's plumbed today
---------------------

The tier knob is exposed at the kernel level on every emitted
``*_kernel<T, RESOURCE_TIER>``. At the inline-CUDA ``_device`` /
``_inner`` level, these functions accept ``RESOURCE_TIER`` + a
caller-provided ``T *d_workspace`` argument:

* ``fdsva_so_inner<T, RESOURCE_TIER>(s_df2, s_idsva_so, s_Minv,
  s_df_du, s_XImats, s_temp, d_workspace, gravity)`` — 4*nv³ inner
  scratch routes between ``s_temp`` (SHARED) and ``d_workspace``
  (LITE/MINIMAL).
* ``forward_dynamics_gradient_device<T, RESOURCE_TIER>(s_df_du,
  s_q, s_qd, [s_qdd, s_Minv | s_u], d_robotModel, gravity,
  d_workspace)`` — whole s_temp arena routes per tier.
* ``inverse_dynamics_gradient_device<T, RESOURCE_TIER>(s_dc_du,
  s_q, s_qd, [s_qdd], d_robotModel, gravity, d_workspace)`` — whole
  s_temp arena routes per tier.
* ``end_effector_pose_hessian_device<T, RESOURCE_TIER>
  (s_d2eePos, s_deePos, s_q, d_robotModel, d_workspace)`` —
  ``s_d2eeTemp`` slot (the 2*16*num_ees*n² portion) routes per tier;
  inner_no_d2 stays in smem at all tiers.
* ``idsva_so_device<T, RESOURCE_TIER>(s_idsva_so, s_q, s_qd, s_qdd,
  d_robotModel, gravity, d_workspace)`` — codegen-time frame
  dispatcher: ``body_frame_inner`` for fixed-base, ``world_frame_inner``
  for floating-base. Inner temp arena routes per tier.

Sizing constants — call these from host code to allocate the right
buffers:

.. code-block:: cpp

   template <typename T, int TIER = TIER_SHARED>
   constexpr size_t FDSVA_SO_INNER_SMEM_BYTES();           // bytes for s_temp at TIER
   template <typename T, int TIER = TIER_SHARED>
   constexpr size_t FDSVA_SO_INNER_WORKSPACE_BYTES();      // bytes for d_workspace at TIER

   // Same pattern: FORWARD_DYNAMICS_GRADIENT_DEVICE_INLINE_*, INVERSE_DYNAMICS_GRADIENT_DEVICE_INLINE_*,
   //               END_EFFECTOR_POSE_HESSIAN_DEVICE_INLINE_*, IDSVA_SO_DEVICE_INLINE_*

At ``TIER_SHARED`` the SMEM_BYTES value matches current behavior
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
           /* d_workspace */ workspace, // global mem
           gravity);
   }

References
-----------

* P1 baseline matrix:
  ``test/diagnostics/results/tier_baseline_sm120_rtx5090.md`` —
  per-(algo, robot) register + smem + spill data driving tier
  decisions.
* Smoke test: ``test/diagnostics/tier_instantiation_smoke.py`` —
  verifies all 9 single-overload kernels compile at all 3 tiers
  AND 17 static_asserts validate per-tier SMEM/WORKSPACE invariants.
* Reusable arena helper:
  ``grid_codegen/helpers/_code_generation_helpers.py:504-583``
  (``gen_declare_shared_arena``, ``tier_workspace_expr``).
* Existing bench harness:
  ``test/benchmarks/run_multi_version.py`` (multi-column driver),
  ``test/benchmarks/baselines/{grid,pinocchio,frax,mjx}/`` (per-
  baseline runners), ``test/benchmarks/generate_report.py`` (output
  renderer that already understands ``frax_cpu``/``frax_gpu`` columns
  and gracefully renders missing cells as ``—``).
