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

**Phase 3a + 3b shipped — Minv + FD surgical spill landed**

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
  SUGGESTED_THREADS). The tier system extends this to **above**
  SUGGESTED_THREADS via TIER_MINIMAL's ``launch_bounds=1024``.
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
