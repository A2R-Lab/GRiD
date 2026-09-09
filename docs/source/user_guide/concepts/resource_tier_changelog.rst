Resource-Tier Changelog
=======================

The shipped-work log for the tier/spill system, split out of
:doc:`resource_tier_system` on 2026-09-09. Newest-relevant first is NOT
guaranteed — these are historical records in their original order.

Integrator surgical spill (value + gradient)
--------------------------------------------

The time-integrator kernels follow the same "spill the cold buffers, keep the
hot path in shared memory" philosophy. Both compose **existing** placement
levers from their callees rather than introducing a whole-arena dump.

**Value path** (``integrator_kernel``). The kernel runs forward dynamics per
stage; its dominant inner buffer is the FD inner's Minv F-region (``6·NV²``).
It threads the existing ``forward_dynamics_inner<T, MINV_F_IN_SMEM>`` lever:

* Level 0 (SHARED on robots that fit): F stays in ``s_temp`` (shared).
* Level 1 (LITE/MINIMAL, or SHARED on h1_2): F spills to ``d_workspace`` while the
  hot FD path stays in smem. The overflow on h1_2 is only a few KB, so this
  single surgical lever is enough — h1_2 fixed/floating drop from 103/124 KB to
  ~41/46 KB. ``integrator_kernel`` gained ``unsigned char *d_workspace`` as its
  2nd argument; ``INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T, TIER>`` and
  ``INTEGRATOR_MINV_F_IN_SMEM<TIER>`` are tier-aware.

**Gradient path** (``integrator_gradient_kernel`` / ``..._with_x_kp1``). A
4-rung ladder, least-spill first, spilling only cold / output / coalesced
matrices to **distinct, non-aliasing** ``d_workspace`` sub-offsets (the
gradient never runs concurrently with
inverse_dynamics_gradient/forward_dynamics_gradient/fdsva_so, so it reuses those
sections):

.. list-table:: Integrator-gradient spill ladder
   :header-rows: 1
   :widths: 8 52 40

   * - Rung
     - Spills (all in ``d_workspace``)
     - Example tier/robot
   * - 0
     - nothing (full smem)
     - small robots at SHARED
   * - 1
     - ``s_D_qdd_stage`` (``max_stages·NV·3NV``)
     - g1_fixed SHARED
   * - 2
     - rung 1 + ``s_dAB`` output (``2NV·3NV``) + inverse_dynamics_gradient **selective** (da_df band only;
       the FD-grad inner stays in smem, just smaller)
     - g1_floating SHARED — hot path stays in smem
   * - 3
     - rung 2 + the **whole** FD-grad inner ``s_temp`` (inverse_dynamics_gradient global_temp)
     - h1_2 fixed/floating — the inner is 160-441 KB, physically can't fit a
       100 KB box, so this is unavoidable

The sub-offsets are ``GRID_INTEGRATOR_DU_DAB_OFFSET_BYTES`` and
``GRID_INTEGRATOR_DU_INNER_OFFSET_BYTES`` (Dqdd sits at offset 0); the
placement per tier is exposed via ``INTEGRATOR_DU_{D_QDD,DAB}_IN_SMEM<TIER>``
and ``INTEGRATOR_DU_INNER_LEVEL<TIER>``. The gradient scaffold that feeds the
final dAB assembly (``s_dc_du`` / ``s_vaf`` / ``s_Minv``) always stays in smem.

**Why a whole-inner rung exists here but not for, e.g., the value path.** For
the value path the overflow is tiny, so one surgical lever closes it. For the
gradient on the biggest *floating* humanoid (h1_2), the FD-gradient inner
scratch *alone* exceeds the per-block smem cap, so some of the hot path must go
to L2-pinned global at MINIMAL — there is no surgical decomposition that keeps
it in smem. Rung 3 is therefore a physically-forced backstop used only where
rung 2 can't fit, not the default.

idsva_so (body + world frame) — done
------------------------------------

The IDSVA-SO surgical-spill follow-up that earlier revisions of this doc listed
as deferred is **implemented**: both frames use a ``select_shared_tier_3way``
ladder. Body rungs are {full, output→global, +BC→global (surgical),
+whole-s_temp→global}; world rungs are {full, output→global,
+whole-s_temp→global}. The fixed body inner is monolithic/aliased, so the
surgical BC spill is small relative to a humanoid's smem gap and the whole-arena
rung is what makes h1_2 fit (at a perf cost); a finer hot/cold de-alias of that
inner remains a tracked refactor (``docs/idsva_so_inner_refactor_notes.md``).

Humanoid-scale spill (``humanoid-tier-spill``, landed)
------------------------------------------------------

This bundle (branched off ``modernizing-tests``) brought every overflowing
kernel under the sm_120 ~100 KB cap on humanoid-scale robots. It was staged in
chunks; all are now landed (the integrator + idsva_so surgical spills described
above were the final pieces):

**Chunk 1: bench harness h1_2 enablement + failure tolerance** (shipped)
  - ``h1_2`` (Unitree H1.2, NV=51 fixed, 57 floating) added to the
    multi-version bench's ``ROBOTS`` tuple + EE-frame maps in
    ``run_multi_version.py`` and all four baseline runners
    (``baselines/{grid,pinocchio,mjx,frax}/run.py``).
  - **Per-algo runtime skip**: ``baselines/grid/run.py``'s
    ``PER_ALGO_SPECS`` (the single source of truth for each algo's bench
    call) wires ``GRID_SKIP_*`` macros for every measured kernel. When ``grid_kernel_fits_device(SHARED_BYTES)``
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
    spill-level list. SHARED picks the lowest-spill fitting
    ``cuda_target_shared_mem_bytes`` (~98 KB); LITE picks the lowest-spill
    fitting ``cuda_target_lite_shared_mem_bytes`` (~48 KB), clamped to
    ``≥`` SHARED; MINIMAL is always the most-spill index.
  - Five algos now populate ``self.<algo>_spill_tier_3way`` plus
    ``self.<algo>_t_count_per_tier`` (3-tuple of arena t_counts):
    ``inverse_dynamics_gradient``, ``forward_dynamics_gradient``,
    ``end_effector_pose_hessian``, ``fdsva_so``, ``idsva_so_body_frame``.
  - **No emit-path change yet** — existing single-body emission uses
    the SHARED pick (= today's behavior). The picks are available for
    introspection by tests + future per-tier emit work.

**Chunk 3: per-tier ``if constexpr`` emission per algo** (4 of 5 shipped)
  - Each kernel now dispatches on its 3-way picks: collapsed picks emit a
    single body (current behavior), divergent picks emit
    ``if constexpr (RESOURCE_TIER == TIER_X)`` branches with per-tier
    spill flags. The tier-aware ``*_DYNAMIC_SHARED_MEM_BYTES<T, TIER>``
    constexpr reports per-tier smem requirements (default ``TIER = TIER_SHARED``
    preserves all existing single-arg call sites).
  - **Shipped**: ``end_effector_pose_hessian``, ``inverse_dynamics_gradient``,
    ``forward_dynamics_gradient``, ``fdsva_so`` (commit
    ``8e5ff50``). Verified via nvcc compile of go2_fixed (FULL 3-way
    divergence on end_effector_pose_hessian + fdsva_so picks) and h1_2_fixed
    (SHARED=1, LITE/MIN=2 divergence on end_effector_pose_hessian +
    inverse_dynamics_gradient). Smoke test passes on iiwa14 (picks
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
     - end_effector_pose_hessian
     - inverse_dynamics_gradient
     - forward_dynamics_gradient
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

**Chunk 4: new spill levels for h1_2-overflowing kernels** (landed)
  All h1_2-overflowing kernels now have surgical spill ladders that bring them
  under the sm_120 ~100 KB cap. The design that landed matches the v2.0
  philosophy — push the cold / output / coalesced buffers first, keep the hot
  recursion in smem, and fall back to a whole-inner spill only where the inner
  alone exceeds the cap (physically forced, e.g. fdsva_so / idsva_so / the
  integrator gradient on h1_2). ``select_shared_tier_3way`` picks the lowest
  fitting rung per tier. Per-algo specifics:

  * **Minv / FD**: split the ``6·NV²`` ``s_F`` region out as a separate
    ``s_F`` / ``d_workspace`` parameter (Level 1 surgical). On h1_2_fixed this
    alone drops Minv 100 KB → ~38 KB.
  * **ABA**: the 140·NJ recursion band has no clean sub-split, so its Level 1
    redirects the whole inner ``s_temp`` to L2-pinned workspace.
  * **end_effector_pose_gradient / end_effector_pose_hessian / inverse_dynamics_gradient / forward_dynamics_gradient / fdsva_so**: 3-6 level ladders
    spilling inner_temp, then the output, then (fdsva_so) ``s_df_du`` / ``s_Minv``.
  * **idsva_so (body + world)**: ladders spilling the 4·NV³ output, then BC
    (body, surgical), then the whole inner. See "idsva_so — done" above.
  * **integrator (value + gradient)**: see "Integrator surgical spill" above.

  Where a surgical sub-split exists it is preferred; the whole-inner rung is the
  guaranteed-fit backstop. The remaining finer-grained win (de-aliasing the
  monolithic idsva_so / fdsva_so inners so even MINIMAL keeps more of the hot
  band in smem) is tracked in ``docs/idsva_so_inner_refactor_notes.md``.

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

**Phase 3a + 3b + 3c + 3d + 3e shipped — Minv + FD + ABA + END_EFFECTOR_POSE_GRADIENT + FDSVA_SO L4-5 spill landed**

* **Phase 3d (END_EFFECTOR_POSE_GRADIENT)**: mirrors the END_EFFECTOR_POSE_HESSIAN 3-tier spill pattern. SHARED
  keeps the full inner_temp + s_deePos + dXmatsHom in smem; LITE pushes the
  recursion-hot inner_temp (2*2*16*num_ees*n T = ~52 KB on humanoid-scale)
  to L2-pinned workspace and writes ``s_deePos`` directly into global
  output; MINIMAL also pushes ``s_dXmatsHom`` (16*n T) to workspace.
  ``end_effector_pose_gradient_kernel`` now takes ``unsigned char *d_workspace``
  as its new 2nd argument. ``END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, TIER>()``
  is tier-aware. The workspace section reuses the SO offset (END_EFFECTOR_POSE_GRADIENT
  and SO algos don't run concurrently). Per-(robot) picks:

  - iiwa14_fixed/floating: (0, 0, 2) — SHARED/LITE alias to full smem;
    MINIMAL spills inner_temp + dxhom
  - go2_fixed/floating: (0, 0, 2) — same
  - h1_2_fixed/floating: (1, 1, 2) — SHARED/LITE both already spill
    inner_temp + s_deePos; MINIMAL additionally spills dxhom

  Smoke (nvcc -gencode arch=compute_120,code=sm_120, all 9 emitted kernels
  × 3 tiers per robot): iiwa14_fixed/go2_fixed/h1_2_fixed all 27/27 PASS.
  h1_2_fixed END_EFFECTOR_POSE_GRADIENT compiles clean at 40/40/50 registers (SHARED/LITE/MINIMAL).

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
  arena to L2-pinned workspace (analogous to the existing
  ``inverse_dynamics_gradient`` ``use_global_temp`` pattern). A side effect of Phase 3b: ABA's
  ``inner_temp_mem_size`` decreased on floating-base because the defensive
  ``max(140*NJ+138, fd_inner_size)`` formula now sees a smaller FD inner
  (post-F-removal). On h1_2_floating ABA's Level 0 arena dropped enough
  that it now fits the 99 KB cap without spill — picks are
  ``aba=(0, 1, 1)``: SHARED/LITE use full smem on most robots, LITE on
  h1_2_floating spills (~48KB target).



Status (commits ``da831dd`` + ``0795442``):

* ``minv_inner`` now takes ``T *s_F`` as a separate 6*NV*NV scratch
  parameter; ``forward_dynamics_inner`` analogously takes ``T *s_minv_F``.
  Callers decide whether the F-region lives in extra smem (Level 0,
  preserves current behavior on small robots) or L2-pinned workspace
  (Level 1, frees ~62 KB smem on humanoid-scale robots).
* ``minv_kernel`` and ``forward_dynamics_kernel`` now both take
  ``unsigned char *d_workspace`` as their new 2nd argument. The per-tier
  ``select_shared_tier_3way`` picks Level 0 vs Level 1 based on the
  ``cuda_target_shared_mem_bytes`` (SHARED, 98 KB), ``cuda_target_lite_shared_mem_bytes``
  (LITE, 48 KB), and "always max spill" (MINIMAL) targets.
* ``MINV_DYNAMIC_SHARED_MEM_BYTES<T, TIER>`` and
  ``FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T, TIER>`` are now tier-aware constexprs
  reporting per-tier smem footprints (default ``TIER = TIER_SHARED`` preserves
  every existing single-arg call site).
* Verified via nvcc compile of h1_2_fixed at all 3 tiers:

  - **h1_2_fixed Minv**: 100 KB → 37 KB smem (SHARED picks surgical at h1_2-scale)
  - **h1_2_fixed FD**: 106 KB → 37 KB smem
  - 40-64 registers/thread per tier; all three tiers instantiate cleanly.

* External call sites updated to pass ``d_workspace``:

  - ``bindings/grid_rbd/wrapper_template.cu`` (Python FFI surface)
  - ``test/cuda_equivalents/cuda_equivalence_runner.cu`` (CUDA equivalence harness)

* Composition: FDSVA_SO's device + kernel paths internally compose Minv
  and FD inner. Both call sites updated to pass ``minv_s_F`` (the local
  slot at the start of ``s_temp``) through to the new signatures.

The surgical-spill pattern each of these algos used (split the largest
inner-temp buffer — e.g. Minv's ``s_F`` — into a separate ``s_F`` /
``d_workspace`` parameter, re-base the other offsets to 0, and pick the
placement per tier via ``select_shared_tier_3way``) is the same one the
integrator and idsva_so now follow. See the per-algo ``gen_*`` functions in
``grid_codegen/algorithms/`` for the concrete signatures.

LITE 48 KB smem target — shipped; value tuning remains
-------------------------------------------------------

The machinery this section once described as deferred is **landed**:
``cuda_target_lite_shared_mem_bytes`` (default 48 KB, env-overridable),
``select_shared_tier_3way`` picking a per-tier rung against the SHARED / LITE /
MINIMAL targets, per-tier ``if constexpr`` emission, the ternary
``gen_declare_shared_arena`` arena helper, and the new spill levels that bring
every previously-overflowing h1_2 kernel (IDSVA_SO, FDSVA_SO, END_EFFECTOR_POSE_GRADIENT,
Minv/FD/ABA, and the integrator value+gradient) under the sm_120 cap.

What remains is **tuning, not plumbing**: is 48 KB the right LITE cliff, or
would 32/64 KB fit the real perf curve better? That is a knob to sweep, not a
feature to build — see the deferred validation sweep below.

Remaining inner-plumbing refinement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``idsva_so_body_frame_inner`` / ``idsva_so_world_frame_inner`` now take a
unified ``(s_temp, d_workspace)`` signature and spill per tier (whole-arena at
the deepest rung). The remaining refinement is a *finer-grained* partial spill:
the SO body emitters have heavily-aliased intermediate lifetimes, so keeping
more of the hot band in smem even at MINIMAL needs per-sub-buffer lifetime
analysis. Tracked in ``docs/idsva_so_inner_refactor_notes.md``.

Deferred validation sweep (P7 / P8)
------------------------------------

The tier-system perf characterization is deferred to land alongside
the LITE-48KB and humanoid follow-ups. When that work happens, the
benchmark sweep should produce one comprehensive matrix in a single
run:

**Coverage**

* **GRiD across tiers**: SHARED, LITE (post-48KB-target), MINIMAL.
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
  (TIER_SHARED default, TIER_LITE/MINIMAL via a new ``--resource-tier``
  passthrough on the GRiD harness).
* Each cell's ``try`` block in the runner needs to catch all
  ``Exception`` (including ``cudaError`` surfacing as Python
  exceptions, OOM, codegen failures, timeout) and write a placeholder
  entry instead of re-raising.

**Output artifact**

The result lands as a dated, committed snapshot
``test/benchmarks/tier_validation_matrix_<ts>.md`` (e.g.
``tier_validation_matrix_20260523_2200.md``). Same row × column structure as the existing
``benchmark_multi_version_sm120_5090_full.md`` but with GRiD split
into three tier columns (``grid_shared``, ``grid_lite``,
``grid_minimal``).

**Threshold tuning** (the reason this is a sweep, not just
correctness verification):

* Was 48 KB the right LITE smem target? Maybe 64 KB or 32 KB fits
  the actual perf cliff better. Adjust the codegen target.
* Are there ``if constexpr`` branches whose perf cost is too high?
  E.g. on iiwa14 where everything fits SHARED, LITE/MINIMAL aliases
  should be byte-equivalent — verify no regression.
* Pinocchio absolute baseline: GRiD-SHARED / Pinocchio-CPU and
  GRiD-MINIMAL / Pinocchio-CPU ratios. Even at MINIMAL, GRiD on GPU
  should beat Pinocchio CPU for batch ≥ ~16. If MINIMAL drops below
  Pinocchio at small batches, the downgrade design is too aggressive.
* Frax comparison: with GPU acceleration available on both sides,
  GRiD should beat Frax GPU at the dynamics kernels GRiD is
  specialized for (RNEA, FD, gradients, SO). Frax GPU may win on
  end-effector pose (no SIMT specialization). Use this to calibrate
  expectations.

