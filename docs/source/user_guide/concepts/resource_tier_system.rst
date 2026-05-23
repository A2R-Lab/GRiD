Resource-Tier System (v2.0)
============================

**Status**: shipped in v2.0 + Phase 3a/b/c/d/e spill machinery + L2 pinning
default-on. Inner-controlled placement refactor (below) implemented for
fdsva_so/Minv/FD/ABA/EE_GRAD — pending numerical-equivalence testing. One
surgical-spill follow-up deferred (IDSVA_SO_B/W; see "Immediate next steps").

**Audience**: inline-CUDA users (``#include "grid.cuh"`` from their own
kernel). The Python wrappers (``grid_rbd.RobotHandle``,
``grid_rbd.jax.JaxRobotHandle``) always use ``TIER_PERF`` by design.


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
     - Nothing. Arrays in, arrays out. Always ``TIER_PERF``.
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


Design philosophy: smart inners, thin wrappers
----------------------------------------------

The organizing principle of the generated code is:

  **Put all the intelligence in the inner functions. Make everything above
  them a thin convenience wrapper.**

Concretely, an emitted algorithm is four layers, and the value is concentrated
entirely in the bottom one:

``<algo>_inner`` — *the engine.*
  A ``__device__`` routine that does the actual rigid-body-dynamics math. It
  is written to use **as much block-wide parallelism as possible** (see
  below), and it is **smart about memory**: it owns the decision of what lives
  in shared memory vs. global memory, how scratch is laid out, what gets
  spilled under resource pressure, and what gets recomputed vs. cached. It
  takes the caller's input/output pointers plus a shared scratch arena
  (``s_temp``) and a global scratch arena (``s_workspace``), and decides
  internally — via a compile-time placement parameter — which buffers go
  where. Nothing above this layer needs to understand the algorithm's memory
  layout.

``<algo>_device`` — *convenience: "call the engine without thinking about arenas."*
  A ``__device__`` wrapper for inline-CUDA users who want a single call rather
  than managing the scratch arena themselves. It declares the shared-memory
  arena (sized for the default placement), loads/updates the per-configuration
  helper tables (``XImats`` etc.), and calls ``_inner``. Use it when you want
  to call a GRiD primitive from your kernel but don't need to micro-manage
  where its scratch lives.

``<algo>_kernel`` — *convenience: "a ready-to-launch batch entry point."*
  A ``__global__`` entry point that loops over a trajectory/batch of inputs,
  loads each timestep's inputs into shared memory, dispatches on
  ``RESOURCE_TIER``, and calls the engine. This is what you launch if you want
  GRiD to own the whole kernel. It is templated on ``<T, RESOURCE_TIER>`` and
  carries the ``__launch_bounds__`` for the tier.

``<algo>`` (host) — *convenience: "I never want to touch device code."*
  A ``__host__`` wrapper that does the host↔device memory transfers, chooses
  block/grid dimensions, sets the kernel's dynamic-shared-memory attribute,
  and launches the kernel. This is what the Python/JAX handles call under the
  hood, and what a C++ host-only user calls.

Why this shape? Because the audience that cares about performance is calling
``_inner`` (or ``_device``) and composing it into a larger fused kernel. For
that user, **the kernel and host layers are noise** — they want the raw
block-parallel routine and full control of the memory hierarchy. The
convenience layers exist so that the *other* 90% of users never have to see
any of it. Keeping the layers thin also means there is exactly one place where
the hard decisions live (the inner), so there is one place to audit, tune, and
get right.


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
``s_workspace`` (global) for each spillable buffer *at the top of the
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
   * - ``direct_minv_inner``
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
   * - ``*_device`` (id_du / fd_du / idsva_so / d2ee)
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
progressively move buffers to ``s_workspace``). Which level a given
``RESOURCE_TIER`` maps to is decided **at code-generation time, per robot**,
based on what actually fits the smem budget for that robot. Small robots
(e.g. iiwa14, go2) keep every tier at level 0 — there is nothing to spill, so
``TIER_PERF``/``LITE``/``MINIMAL`` are byte-identical. Large robots (e.g.
h1_2) map the lower tiers to deeper spill levels. Because the mapping is
per-robot, **multiple tiers can share a level**, and the generated
``*_IN_SMEM<TIER>()`` / ``*_SCRATCH_IN_SMEM<TIER>()`` constexprs expose
exactly which placement each tier resolves to for the robot you generated.

This is the reconciliation of two goals that look opposed: the inner stays
self-contained and decidable from a single placement flag (good for inline
reuse and for evolving spills), while the *choice* of flag per tier is a
per-robot, fit-driven decision made once at codegen time (good for not paying
for a spill you don't need).


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

The ``*_device`` inline entry points (id_du / fd_du / idsva_so / d2ee) still
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
   concurrently-resident block for ``s_workspace`` (0 when the placement keeps
   everything in shared). Pin it in L2 if you spill (see ``grid_begin_l2_persisting``).
#. Call ``<algo>_inner<T, placement>(..., s_temp, s_workspace, ...)``.

**Testing status of the refactor**: all five converted algos
(fdsva_so / Minv / FD / ABA / EE_GRAD) compile clean at all three tiers across
iiwa14 / go2 / h1_2 (fixed) + g1 (floating); the tier-instantiation smoke
passes. Numerical equivalence (``cuda_equivalence``) and the per-tier perf
sweep are pending a joint testing session — the Minv/FD arena layout changed
(no_F-then-F instead of F-then-no_F), so equivalence is the gating check.

Immediate next steps
---------------------

One algorithm still needs per-algorithm surgical-spill design before
it can be unlocked at h1_2-scale humanoids:

1. **IDSVA_SO_B / IDSVA_SO_W (Phase 3f)** — overflow at 146-168 KB
   on h1_2. ``use_global_output`` (Level 1) already spills the 4*NV³
   output tensor; ``grav_full_spill`` (Level 2, floating-only) already
   spills the d2X/d2a/d2f gravity-Hessian helper tensors. What's left
   is the recursion-hot inner band (per-body 6x6 spatial matrices,
   per-velocity 6-vectors). A naïve full-band spill (analogous to
   ABA Phase 3c) would be a perf cliff because the band is touched
   every BFS step — needs per-sub-buffer hot/cold analysis to find a
   cold/write-once slice that's safe to spill.

   Entry points for the analysis:
   ``GRiDCodeGenerator/algorithms/_idsva_so.py:gen_idsva_so_body_frame_inner_temp_mem_size``
   shows the smem layout (``body_mat_count``, ``body_vec_count``,
   ``vel_vec_count``, ``vel_mat_count``); the inner function body
   (``gen_idsva_so_body_frame_inner``) is where access patterns live.

What's *not* a concern (already addressed by L2 pinning):

* Phase 3a/3b/3c spilled buffers (``s_F``, ABA's 140*NJ band) ARE
  recursion-hot, but L2 pinning (default-on) means access is
  ~smem→L2 cost, not ~smem→HBM. The perf cost relative to keeping
  them in smem is bounded by the L2-vs-shared latency gap. Measure
  via the post-Phase-3 baseline sweep before treating as a problem.

* Phase 3e (FDSVA_SO ``s_df_du``, ``s_Minv``) is more naturally
  output-like (write-then-read in disjoint phases) — less perf-
  sensitive than 3a-3c.

Immediate next steps
---------------------

One algorithm still needs per-algorithm surgical-spill design before
it can be unlocked at h1_2-scale humanoids:

1. **IDSVA_SO_B / IDSVA_SO_W (Phase 3f)** — overflow at 146-168 KB
   on h1_2. ``use_global_output`` (Level 1) already spills the 4*NV³
   output tensor; ``grav_full_spill`` (Level 2, floating-only) already
   spills the d2X/d2a/d2f gravity-Hessian helper tensors. What's left
   is the recursion-hot inner band (per-body 6x6 spatial matrices,
   per-velocity 6-vectors). A naïve full-band spill (analogous to
   ABA Phase 3c) would be a perf cliff because the band is touched
   every BFS step — needs per-sub-buffer hot/cold analysis to find a
   cold/write-once slice that's safe to spill.

   Entry points for the analysis:
   ``GRiDCodeGenerator/algorithms/_idsva_so.py:gen_idsva_so_body_frame_inner_temp_mem_size``
   shows the smem layout (``body_mat_count``, ``body_vec_count``,
   ``vel_vec_count``, ``vel_mat_count``); the inner function body
   (``gen_idsva_so_body_frame_inner``) is where access patterns live.

What's *not* a concern (already addressed by L2 pinning):

* Phase 3a/3b/3c spilled buffers (``s_F``, ABA's 140*NJ band) ARE
  recursion-hot, but L2 pinning (default-on) means access is
  ~smem→L2 cost, not ~smem→HBM. The perf cost relative to keeping
  them in smem is bounded by the L2-vs-shared latency gap. Measure
  via the post-Phase-3 baseline sweep before treating as a problem.

* Phase 3e (FDSVA_SO ``s_df_du``, ``s_Minv``) is more naturally
  output-like (write-then-read in disjoint phases) — less perf-
  sensitive than 3a-3c.

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
     - ``MAX_PERF_LEVEL_THREADS`` (288-512 per robot)
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

In-flight humanoid follow-up (``humanoid-tier-spill`` branch)
-------------------------------------------------------------

The next bundle, branching off ``modernizing-tests``, is staged in three
chunks:

**Chunk 1: bench harness h1_2 enablement + failure tolerance** (shipped)
  - ``h1_2`` (Unitree H1.2, NV=51 fixed, 57 floating) added to the
    multi-version bench's ``ROBOTS`` tuple + EE-frame maps in
    ``run_multi_version.py`` and all four baseline runners
    (``baselines/{grid,pinocchio,mjx,frax}/run.py``).
  - **Per-algo runtime skip**: ``timeGRiD_{single,batch}.cu`` and
    ``run.py``'s ``PER_ALGO_SPECS`` now wire ``GRID_SKIP_*`` macros for
    every measured kernel. When ``grid_kernel_fits_device(SHARED_BYTES)``
    is false, the measure function prints a parseable ``... SKIPPED``
    line and returns. ``timing_parser.py`` ignores the SKIPPED line and
    ``fill_nulls`` populates the algo with null —
    ``generate_report.py`` renders missing cells as ``—``.
  - Net result: a baseline sweep including h1_2 now produces a real row
    for every (algo, robot, base) cell that fits the sm_120 ~100 KB
    per-block cap, and graceful ``—`` placeholders for cells that
    overflow. No more "one overflowing kernel kills the whole binary."

**Chunk 2: 3-way spill picker infrastructure** (shipped, dormant)
  - ``cuda_target_lite_shared_mem_bytes`` (default 48 KB, env-overridable)
    added to ``GRiDCodeGenerator.__init__``.
  - ``select_shared_tier_3way(*t_counts)`` returns
    ``(perf_pick, lite_pick, minimal_pick)`` indices into the algorithm's
    spill-level list. PERF picks the lowest-spill fitting
    ``cuda_target_shared_mem_bytes`` (~98 KB); LITE picks the lowest-spill
    fitting ``cuda_target_lite_shared_mem_bytes`` (~48 KB), clamped to
    ``≥`` PERF; MINIMAL is always the most-spill index.
  - Five algos now populate ``self.<algo>_spill_tier_3way`` plus
    ``self.<algo>_t_count_per_tier`` (3-tuple of arena t_counts): ID_DU,
    FD_DU, D2EE, FDSVA_SO, IDSVA_SO_BODY_FRAME.
  - **No emit-path change yet** — existing single-body emission uses
    the PERF pick (= today's behavior). The picks are available for
    introspection by tests + future per-tier emit work.

**Chunk 3: per-tier ``if constexpr`` emission per algo** (4 of 5 shipped)
  - Each kernel now dispatches on its 3-way picks: collapsed picks emit a
    single body (current behavior), divergent picks emit
    ``if constexpr (RESOURCE_TIER == TIER_X)`` branches with per-tier
    spill flags. The tier-aware ``*_DYNAMIC_SHARED_MEM_BYTES<T, TIER>``
    constexpr reports per-tier smem requirements (default ``TIER = TIER_PERF``
    preserves all existing single-arg call sites).
  - **Shipped**: ``d2ee``, ``id_du``, ``fd_du``, ``fdsva_so`` (commit
    ``8e5ff50``). Verified via nvcc compile of go2_fixed (FULL 3-way
    divergence on d2ee + fdsva_so picks) and h1_2_fixed (PERF=1, LITE/MIN=2
    divergence on d2ee + id_du). Smoke test passes on iiwa14 (picks
    collapse).
  - **Deferred**: ``idsva_so_body_frame``. Its current spill machinery is
    asymmetric (``grav_full_spill`` only applies to floating-base, and is
    auto-triggered only when ``use_global_output`` already exceeds the
    target). The 3-way picks would need a per-base spill-level enumeration.
    Better to restructure this in tandem with Phase 3 (which will add new
    spill levels for h1_2 anyway).

**Where Phase 2b divergence shows up empirically** (from the per-tier picks
survey across 4 robots × 2 bases):

.. list-table:: 3-way picks per (algo × robot × base) — *(perf, lite, minimal)*
   :header-rows: 1
   :widths: 22 18 18 18 18

   * - Robot
     - fdsva_so
     - d2ee
     - id_du
     - fd_du
   * - iiwa14_fixed
     - (0,0,3) divergent
     - (0,0,2) divergent
     - (0,0,2) divergent
     - (0,0,2) divergent
   * - iiwa14_floating
     - (1,1,3) divergent
     - (0,0,2) divergent
     - (0,0,2) divergent
     - (0,0,2) divergent
   * - go2_fixed
     - (0,1,3) **FULL 3-way**
     - (0,1,2) **FULL 3-way**
     - (0,0,2) divergent
     - (0,0,2) divergent
   * - go2_floating
     - (2,3,3) divergent
     - (1,1,2) divergent
     - (0,0,2) divergent
     - (0,0,2) divergent
   * - g1_fixed
     - (2,3,3) divergent
     - (1,1,2) divergent
     - (0,1,2) **FULL 3-way**
     - (0,1,2) **FULL 3-way**
   * - g1_floating
     - (3,3,3) collapsed
     - (2,2,2) collapsed
     - (1,2,2) divergent
     - (1,2,2) divergent
   * - h1_2_fixed
     - (3,3,3) collapsed
     - (1,2,2) divergent
     - (1,2,2) divergent
     - (2,2,2) collapsed
   * - h1_2_floating
     - (3,3,3) collapsed
     - (2,2,2) collapsed
     - (2,2,2) collapsed
     - (2,2,2) collapsed

For robots where picks collapse, the kernel emits a single body (current
behavior, byte-identical to pre-Phase-2b). For divergent rows, the kernel
emits 2 or 3 specialized bodies inside ``if constexpr`` branches.

**Chunk 4: new spill levels for h1_2-overflowing kernels** (deferred — design fixed)
  - With Chunk 1's failure tolerance, h1_2's overflowing kernels (FDSVA_SO,
    IDSVA_SO_B, IDSVA_SO_W, EE_POSE_GRAD, Minv/FD/ABA on floating-base)
    SKIP cleanly at runtime. To actually *run* them, the codegen needs
    new spill levels.

  **Design — two-level surgical + full per algo** (matches the v2.0 tier
  philosophy: surgical wins are the default, full-spill is a backstop):

  * **Level 1 — surgical**: push the *single largest contributor* in
    ``inner_temp`` to L2-pinned workspace, keep everything else (and small
    hot buffers) in shared memory. For ``direct_minv`` on h1_2_fixed the
    target is ``s_F`` (6\*nv² = ~62 KB on NV=51), which alone is enough to
    drop Minv from 100 KB to ~38 KB and clear the 99 KB sm_120 opt-in cap.
    The pattern repeats for ``forward_dynamics`` (its ``s_F``-equivalent
    inner buffer), ``aba`` (the 12\*NJ partial-tree storage), and
    ``ee_pose_gradient`` (per-EE Jacobian column workspace).
  * **Level 2 — full-spill backstop**: push the entire inner-temp arena
    to workspace. Coarse and slow but guaranteed-correct fallback when
    Level 1 still overflows (e.g. ``fdsva_so`` on h1_2_floating at 244 KB
    needs more than just the largest buffer).
  * The codegen's ``select_shared_tier_3way`` picks the lowest spill level
    fitting each tier's target: PERF prefers Level 0 (no spill), then
    Level 1; LITE adds Level 1 at 48 KB target; MINIMAL is always
    most-spill.

  **L2 cache pinning** is on-by-default whenever any kernel in the
  generated header is at Level ≥ 1: workspace bytes get read/written
  every timestep and are HBM-cold without pinning. The codegen flips
  ``GRID_CUDA_ENABLE_L2_PERSISTING`` to 1 in the emitted header when any
  algo has ``use_workspace_temp`` (or surgical equivalent) set; the
  existing ``grid_begin_l2_persisting`` helper covers the runtime mechanics.

  **Implementation cost per algo** (estimate from Minv inspection):

  * ``gen_direct_minv_inner`` has 21 ``FOffset`` references; surgical
    refactor splits ``s_F`` out as a separate parameter (default-routes
    to ``&s_temp[FOffset]`` when not spilled, to a workspace pointer
    when spilled). Other offsets re-base to 0.
  * Every caller that composes ``direct_minv_inner`` (forward dynamics,
    forward_dynamics_gradient, fdsva_so) also passes the new ``s_F``
    pointer.
  * ``gen_direct_minv_inner_temp_mem_size`` splits into
    ``gen_direct_minv_inner_temp_mem_size`` (rest of s_temp, minus F) and
    ``gen_direct_minv_inner_F_size`` (the F-region size).
  * ``GRiDCodeGenerator.py`` arena math + 3-way pick tables for Minv.
  * Tier-aware ``MINV_DYNAMIC_SHARED_MEM_BYTES<T, TIER>`` constexpr
    reports the per-tier smem footprint.
  * Workspace sizing in ``GRID_WORKSPACE_BYTES_PER_TIMESTEP`` extended to
    cover Minv's F-region when Level 1 active.

  Per algo this is ~150 lines of careful refactor + smoke + nvcc compile
  test on h1_2. For 5-6 h1_2-overflowing algos that's multi-day work
  best done in a focused follow-up session, not bundled with Phase 1-2b.

**Phase 3f IDSVA_SO_B + W — surgical spill needs design (no clean win)**

The IDSVA_SO_B/W kernels overflow on h1_2 (146-168 KB) but the design
needs to be surgical for it to be worth landing:

* **Already spilled at Level 1** (``use_global_output``): the 4*NV³ output
  tensor (``s_idsva_so``). Write-once + read-at-end → ideal spill target.
* **Already spilled at Level 2** (floating-base only, ``grav_full_spill``):
  the d2X/d2a/d2f gravity-Hessian helper tensors. Moderate access.
* **What's left in smem** is the recursion-hot inner working set
  (per-body 6x6 spatial matrices, per-velocity 6-vectors, etc.) — every
  step of the BFS recursion touches multiple sub-buffers per thread.
  *Pushing this band to workspace is a perf cliff*, not a surgical spill.

The naïve "redirect entire ``s_temp`` to L2-pinned workspace" approach
(analogous to ABA Phase 3c) would tank perf on every divergent (algo, robot)
cell — and ABA only got away with it because its band is small relative
to its dispatch overhead. For IDSVA_SO_B the inner is the main cost.

**Surgical Phase 3f needs**: identify which sub-buffers in the IDSVA_SO_B
inner are cold-or-write-once vs. recursion-hot. Then split a separate
parameter for the cold ones (similar to ``s_F``/``s_minv_F`` in Phase 3a/3b).

This is a per-algorithm design exercise requiring access to the inner's
recursion structure. **Deferred to a focused next session** until we have
that analysis — for now h1_2 IDSVA_SO_B/W SKIP cleanly via the
failure-tolerant bench (Phase 1).

**L2 cache pinning (default-ON in v2.0)**

``GRID_CUDA_ENABLE_L2_PERSISTING`` defaults to 1 in the generated header.
The ``init_gridData`` wrapper calls ``grid_begin_l2_persisting`` on
``d_workspace`` once at allocation time and pairs it with
``grid_end_l2_persisting`` in ``close_grid``. This means spilled buffers
(Minv-F at Phase 3a, FD's Minv-F at 3b, ABA's interleaved scratch at 3c,
FDSVA_SO's df_du/Minv at 3e) live in L2 cache for the kernel's lifetime —
the perf hit relative to keeping them in shared memory is ~smem→L2 latency
(~a few cycles), not the smem→HBM gap (~100s of cycles).

Why this matters: most Phase 3 spills target recursion-hot buffers
(touched many times per kernel), not write-once outputs. Without L2
pinning, spilled hot data would hit HBM repeatedly and the perf cliff
would be sharp. With L2 pinning, the kernel still mostly hits L2.

If your workload requires the L2 cache for other concurrent kernels and
you want to opt out, compile with ``-DGRID_CUDA_ENABLE_L2_PERSISTING=0``.

**Phase 3a + 3b + 3c + 3d + 3e shipped — Minv + FD + ABA + EE_POSE_GRAD + FDSVA_SO L4-5 spill landed**

* **Phase 3d (EE_POSE_GRAD)**: mirrors the D2EE 3-tier spill pattern. PERF
  keeps the full inner_temp + s_deePos + dXmatsHom in smem; LITE pushes the
  recursion-hot inner_temp (2*2*16*num_ees*n T = ~52 KB on humanoid-scale)
  to L2-pinned workspace and writes ``s_deePos`` directly into global
  output; MINIMAL also pushes ``s_dXmatsHom`` (16*n T) to workspace.
  ``end_effector_pose_gradient_kernel`` now takes ``unsigned char *d_workspace``
  as its new 2nd argument. ``DEE_POS_DYNAMIC_SHARED_MEM_BYTES<T, TIER>()``
  is tier-aware. The workspace section reuses the SO offset (EE_POSE_GRAD
  and SO algos don't run concurrently). Per-(robot) picks:

  - iiwa14_fixed/floating: (0, 0, 2) — PERF/LITE alias to full smem;
    MINIMAL spills inner_temp + dxhom
  - go2_fixed/floating: (0, 0, 2) — same
  - h1_2_fixed/floating: (1, 1, 2) — PERF/LITE both already spill
    inner_temp + s_deePos; MINIMAL additionally spills dxhom

  Smoke (nvcc -gencode arch=compute_120,code=sm_120, all 9 emitted kernels
  × 3 tiers per robot): iiwa14_fixed/go2_fixed/h1_2_fixed all 27/27 PASS.
  h1_2_fixed EE_POSE_GRAD compiles clean at 40/40/50 registers (PERF/LITE/MINIMAL).

* **Phase 3e (FDSVA_SO Level 4 + 5)**: extends the existing 4-level spill machinery
  with two new top levels. Level 4 pushes ``s_df_du`` (2*NV²) to a new
  ``GRID_FDSVA_SO_SPILL_OFFSET_BYTES`` workspace section past grad + SO;
  Level 5 also pushes ``s_Minv`` (NV²). The new workspace section is
  sized only when MINIMAL (or any tier) picks ≥ 4 (so iiwa14 doesn't pay
  the allocation). Per-(algo, robot) picks:

  - iiwa14_fixed: (0, 0, 5) — MINIMAL spills max
  - go2_fixed: (0, 1, 5) — full 3-way divergence
  - g1_fixed: (2, 5, 5) — LITE/MINIMAL aggressive
  - g1_floating: (3, 5, 5)
  - h1_2_fixed: (5, 5, 5) — all tiers max-spill (still doesn't fit 99 KB;
    XI tables are the dominant cost on humanoid-scale; defer to a future
    XI-streaming refactor)
  - h1_2_floating: (5, 5, 5)



Status (commits ``da831dd`` + ``0795442`` + (3c-tbd)):

* **Phase 3c (ABA)** uses a different spill pattern than 3a/3b. ABA's 140*NJ+138
  interleaved scratch band has no natural surgical sub-split — it's all one
  tightly-coupled recursion. So Level 1 redirects the *entire* ``s_temp``
  arena to L2-pinned workspace (analogous to the existing ``id_du``
  ``use_global_temp`` pattern). A side effect of Phase 3b: ABA's
  ``inner_temp_mem_size`` decreased on floating-base because the defensive
  ``max(140*NJ+138, fd_inner_size)`` formula now sees a smaller FD inner
  (post-F-removal). On h1_2_floating ABA's Level 0 arena dropped enough
  that it now fits the 99 KB cap without spill — picks are
  ``aba=(0, 1, 1)``: PERF/LITE use full smem on most robots, LITE on
  h1_2_floating spills (~48KB target).



Status (commits ``da831dd`` + ``0795442``):

* ``direct_minv_inner`` now takes ``T *s_F`` as a separate 6*NV*NV scratch
  parameter; ``forward_dynamics_inner`` analogously takes ``T *s_minv_F``.
  Callers decide whether the F-region lives in extra smem (Level 0,
  preserves current behavior on small robots) or L2-pinned workspace
  (Level 1, frees ~62 KB smem on humanoid-scale robots).
* ``direct_minv_kernel`` and ``forward_dynamics_kernel`` now both take
  ``unsigned char *d_workspace`` as their new 2nd argument. The per-tier
  ``select_shared_tier_3way`` picks Level 0 vs Level 1 based on the
  ``cuda_target_shared_mem_bytes`` (PERF, 98 KB), ``cuda_target_lite_shared_mem_bytes``
  (LITE, 48 KB), and "always max spill" (MINIMAL) targets.
* ``MINV_DYNAMIC_SHARED_MEM_BYTES<T, TIER>`` and
  ``FD_DYNAMIC_SHARED_MEM_BYTES<T, TIER>`` are now tier-aware constexprs
  reporting per-tier smem footprints (default ``TIER = TIER_PERF`` preserves
  every existing single-arg call site).
* Verified via nvcc compile of h1_2_fixed at all 3 tiers:

  - **h1_2_fixed Minv**: 100 KB → 37 KB smem (PERF picks surgical at h1_2-scale)
  - **h1_2_fixed FD**: 106 KB → 37 KB smem
  - 40-64 registers/thread per tier; all three tiers instantiate cleanly.

* External call sites updated to pass ``d_workspace``:

  - ``python/grid_rbd/wrapper_template.cu`` (Python FFI surface)
  - ``test/cuda_equivalents/cuda_equivalence_runner.cu`` (CUDA equivalence harness)

* Composition: FDSVA_SO's device + kernel paths internally compose Minv
  and FD inner. Both call sites updated to pass ``minv_s_F`` (the local
  slot at the start of ``s_temp``) through to the new signatures.

**Phase 3a — Minv surgical spill (concrete implementation plan, executed above)**

The Minv inner function has ~15 references to ``s_temp[FOffset + X]`` across
the backward pass (lines ~71, 173, 212-216, 226), debug prints (314, 362,
412, 438, 457), and the forward pass (402-404, 422, 448). All other
references (IAOffset, UOffset, DinvOffset, IaOffset, IaTempOffset) are
self-contained within s_temp.

**Implementation steps (in order, each independently testable):**

1. **Refactor ``gen_direct_minv_inner`` signature**:

   .. code-block:: python

      # Old:
      func_def_start = "void direct_minv_inner(T *s_Minv, const T *s_q, "
      # New:
      func_def_start = "void direct_minv_inner(T *s_Minv, T *s_F, const T *s_q, "
      # And add template <typename T, bool SPILL_F = false> at the top.

2. **Unified pointer-alias setup at top of body** (replaces lines 52-60):

   .. code-block:: cpp

      // Before existing offset declarations, emit:
      constexpr int F_in_temp = SPILL_F ? 0 : 6 * NUM_VEL * NUM_VEL;
      // Offsets re-base to 0 when F is spilled out
      // (Existing FOffset/IAOffset/UOffset constants get adjusted accordingly)

   In Python (codegen-side):

   .. code-block:: python

      FOffset = 0  # always — F refs use F_ptr below
      IAOffset = 0 if spill_F else 6*n*n
      UOffset = IAOffset + 36*n
      # ... etc, all shifted

3. **Body F-reference substitution** (~15 sites):
   Replace ``s_temp[FOffset + X]`` → ``s_F[X]`` everywhere F is accessed.
   The non-F references (IA, U, Dinv, Ia, IaTemp) automatically pick up
   the new offsets via the Python variables — no body edit needed.

4. **Update ``gen_direct_minv_inner_function_call``**:
   Add ``s_F_name`` parameter (default ``"s_F"``), thread it through the
   emitted call site.

5. **Update ``gen_direct_minv_inner_temp_mem_size``**:
   Return ``6*n*n + 36*n + 6*n + d_inv_count + 36*2*max_bfs_width``
   (current), OR the same minus 6*n*n when SPILL_F. Add a new helper
   ``gen_direct_minv_inner_F_size()`` returning ``6*n*n``.

6. **Update ``gen_direct_minv_kernel``** to dispatch on the 3-way
   ``minv_spill_tier_3way`` pick (Level 0 = full smem; Level 1 = surgical
   F-to-workspace). At Level 1: allocate ``s_F = reinterpret_cast<T *>(&d_workspace[...])``
   and ``s_temp`` arena from smem (sized for everything except F).

7. **Update callers**: ``gen_forward_dynamics_inner`` (line 99),
   ``gen_forward_dynamics_gradient_inner_python`` (line 18 of
   ``_forward_dynamics_gradient.py``), and the 3 calls in
   ``_fdsva_so.py`` (lines 265, 329, 360). Each passes ``s_F = &s_temp[0]``
   so their existing smem layout is preserved (Level 0 behavior).

8. **GRiDCodeGenerator.py arena math**:

   .. code-block:: python

      _minv_inner_temp_count = self.gen_direct_minv_inner_temp_mem_size()
      _minv_F_count = self.gen_direct_minv_inner_F_size()  # = 6*n*n
      # Level 0: full smem
      _minv_t_count_full = n + n*n + _minv_F_count + _minv_inner_temp_count + XI_size
      # Level 1: F to workspace
      _minv_t_count_surgical = n + n*n + _minv_inner_temp_count + XI_size
      self.minv_spill_tier_3way = select_shared_tier_3way(_minv_t_count_full, _minv_t_count_surgical)

9. **Tier-aware ``MINV_DYNAMIC_SHARED_MEM_BYTES<T, TIER>``**:
   Emit if-constexpr branch picking the per-tier t_count.

10. **Workspace sizing**: extend ``GRID_WORKSPACE_BYTES_PER_TIMESTEP`` to
    account for Minv-F when ``self.minv_spill_tier_3way`` has any
    non-zero pick.

11. **L2 pinning default-on**: in ``GRiDCodeGenerator.py``, when any algo
    has spill flag set, flip the ``#define GRID_CUDA_ENABLE_L2_PERSISTING``
    default to 1 in the emitted header.

12. **Smoke + nvcc compile** on h1_2_fixed: ``Minv`` should compile at
    all 3 tiers; TIER_PERF arena bytes should drop from ~100 KB to ~38 KB
    (= subtract 6 × 51² × sizeof(float) = ~62 KB).

13. **Numerical correctness**: extend
    ``test/pinocchio_equivalents/test_direct_minv_equivalence.py`` to
    cover the per-tier paths on h1_2_fixed (PERF only, since the
    Python wrapper is locked to PERF; inline-CUDA users at LITE/MINIMAL
    are covered by the smoke-only "compiles" guarantee until a
    dedicated test_resource_tiers harness exists).

Once Phase 3a is in, Phase 3b (FD), 3c (ABA), 3d (EE_POSE_GRAD), and
3e (FDSVA_SO level 4) follow the same pattern — each per-algo identifies
its largest inner-temp buffer, splits it as a separate parameter, and
plumbs through callers.

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
  MAX_PERF_LEVEL_THREADS). The tier system extends this to **above**
  MAX_PERF_LEVEL_THREADS via TIER_MINIMAL's ``launch_bounds=1024``.
* :doc:`codegen_architecture` — describes the four-layer emission
  (``_inner`` / ``_device`` / ``_kernel`` / host); tier templates
  live at the ``_inner`` / ``_device`` / ``_kernel`` layers.

Deferred validation sweep (P7 / P8)
------------------------------------

The tier-system perf characterization is deferred to land alongside
the LITE-48KB and humanoid follow-ups. When that work happens, the
benchmark sweep should produce one comprehensive matrix in a single
run:

**Coverage**

* **GRiD across tiers**: PERF, LITE (post-48KB-target), MINIMAL.
  Each tier × each algo × each robot.
* **Baselines**:
    - Pinocchio (CPU, cppadcodegen-accelerated, multi-threaded — the
      existing ``baselines/pinocchio/run.py`` harness already drives
      this).
    - Frax CPU + Frax GPU (JAX reference at
      https://github.com/danielpmorton/frax — already wired in
      ``baselines/frax/``; emits ``frax_cpu`` and ``frax_gpu`` columns
      that ``generate_report.py`` knows how to render).
* **Timing modes**: single-call AND multi-call (batch) sweeps. Both
  modes already supported by ``run_multi_version.py`` via
  ``--single-call-iters`` and ``--batch-iters``.
* **Base modes**: fixed AND floating per robot.

**Robustness — collect, don't crash**

The sweep should be failure-tolerant: a single (column × algo × robot
× tier × batch_size) cell failing must NOT abort the script. The goal
is to capture as much data as possible in one overnight run. Each
cell that fails should leave a ``—`` (or NaN) entry in the output
JSON; ``generate_report.py`` already renders missing cells gracefully.

Existing entry points to extend:

* ``test/benchmarks/run_multi_version.py`` — multi-column driver;
  add a ``--tiers perf lite minimal`` argument that fans out the
  GRiD column 3-way. Each tier is a separate run of the GRiD
  harness with the appropriate template-arg-specifying compile flag
  (TIER_PERF default, TIER_LITE/MINIMAL via a new ``--resource-tier``
  passthrough on the GRiD harness).
* ``test/benchmarks/run_overnight_sweep.sh`` — already wraps the
  big runs; add the tiers parameter.
* Each cell's ``try`` block in the runner needs to catch all
  ``Exception`` (including ``cudaError`` surfacing as Python
  exceptions, OOM, codegen failures, timeout) and write a placeholder
  entry instead of re-raising.

**Output artifact**

The result lands as ``test/benchmarks/tier_validation_matrix.md``
(committed). Same row × column structure as the existing
``benchmark_multi_version_sm120_5090_full.md`` but with GRiD split
into three tier columns (``grid_perf``, ``grid_lite``,
``grid_minimal``).

**Threshold tuning** (the reason this is a sweep, not just
correctness verification):

* Was 48 KB the right LITE smem target? Maybe 64 KB or 32 KB fits
  the actual perf cliff better. Adjust the codegen target.
* Are there ``if constexpr`` branches whose perf cost is too high?
  E.g. on iiwa14 where everything fits PERF, LITE/MINIMAL aliases
  should be byte-equivalent — verify no regression.
* Pinocchio absolute baseline: GRiD-PERF / Pinocchio-CPU and
  GRiD-MINIMAL / Pinocchio-CPU ratios. Even at MINIMAL, GRiD on GPU
  should beat Pinocchio CPU for batch ≥ ~16. If MINIMAL drops below
  Pinocchio at small batches, the downgrade design is too aggressive.
* Frax comparison: with GPU acceleration available on both sides,
  GRiD should beat Frax GPU at the dynamics kernels GRiD is
  specialized for (RNEA, FD, gradients, SO). Frax GPU may win on
  end-effector pose (no SIMT specialization). Use this to calibrate
  expectations.

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
  ``GRiDCodeGenerator/helpers/_code_generation_helpers.py:504-583``
  (``gen_declare_shared_arena``, ``tier_workspace_expr``).
* Existing bench harness:
  ``test/benchmarks/run_multi_version.py`` (multi-column driver),
  ``test/benchmarks/baselines/{grid,pinocchio,frax,mjx}/`` (per-
  baseline runners), ``test/benchmarks/generate_report.py`` (output
  renderer that already understands ``frax_cpu``/``frax_gpu`` columns
  and gracefully renders missing cells as ``—``).
