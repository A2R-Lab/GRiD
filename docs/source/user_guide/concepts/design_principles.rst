Algorithm design principles
===========================

This page is the shared model for anyone writing or modifying a GRiD
algorithm. It states the reasoning behind the generated-code structure so
that changes match the existing structure. The mechanics live in
:doc:`codegen_architecture` (the emission layers) and
:doc:`resource_tier_system` (tiers and spill); this page states the rules
that tie them together.

Summary
-------

GRiD is a code generator for power users who call hand-tuned,
robot-specialized rigid-body-dynamics kernels directly from their own CUDA
and want every cycle and byte. As such, all the intelligence is placed in the
innermost device function, everything above it is a thin shim, and the inner
owns its own memory placement. Big robots overflow shared memory and must
spill to global; the inner decides what spills, and the caller only sizes the
arenas and passes a flag.


1. Smart inners, thin wrappers
------------------------------

The emission layers (``X_device`` / ``X_kernel`` / host ``X``) and the
``_inner`` composition contract are documented in :doc:`codegen_architecture`;
this section states the rules those layers must obey.

* Value is concentrated in ``X_device`` and its ``_inner`` building blocks.
  The kernel and host layers are noise to the performance user; they stay
  mechanical so there is exactly one place where the hard decisions
  (placement, spill, orchestration) live and get fixed.
* ``_device`` is block-cooperative, not fixed-lane. Every parallel region is
  a block-stride loop (``for (i = tid; i < N; i += blockDim...)``), so any block
  size that fits is correct. Batching is the kernel's grid-stride loop over
  ``blockIdx``. Never assume a specific thread count.
* Higher-level algorithms compose by calling other algorithms' ``_inner``.
  ``fdsva_so``, which embeds ``idsva_so``, loads ``XImats`` once at the top of
  its ``_device`` and calls placement-free ``_inner`` variants of
  sub-algorithms so the load is paid once, not per sub-algorithm.
* Scratch signatures are algorithm-specific. Orchestrators accept
  caller-owned scratch; simpler device functions such as RNEA allocate scratch
  internally. Use the generated signature and resource constants, not a
  universal prototype.


2. The inner owns its memory placement
--------------------------------------

When an inner's scratch does not fit shared memory, the decision of where it
lives belongs to the inner, never the caller.

Spill-capable helpers use a placement flag (``SCRATCH_IN_SMEM``, or
equivalently the ``RESOURCE_TIER``) and take both pointers, ``s_temp``
(shared) and ``d_workspace`` (global). At the very top of the body the helper
selects::

    template <typename T, bool SCRATCH_IN_SMEM = true>
    __device__ void foo_inner(..., T *s_temp, T *d_workspace, ...) {
        if constexpr (!SCRATCH_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }
        // ... the rest of the body is UNCHANGED; it just uses s_temp ...
    }

The caller (a kernel, a device wrapper, or a composing kernel) is a thin
shim: it sizes both arenas from the inner's exposed ``*_SMEM_BYTES`` /
``*_WORKSPACE_BYTES`` constants, hands both pointers in, and passes the
per-tier flag. It never hard-repoints or aliases the inner's scratch from
outside.

This is a rule rather than a preference for three reasons:

* The inner sizes and places its own scratch, so the shared-mem-bytes macro,
  the workspace-bytes macro, and the pointer arithmetic cannot drift apart
  across callers.
* Improvements propagate. If an inner later learns to spill only its cold
  buffers (keeping the hot loop in smem), every caller, the standalone kernel
  and every composing kernel, inherits it by passing the flag.
* ``SCRATCH_IN_SMEM = true`` keeps small robots byte-identical; only robots
  that overflow flip it to ``false``.

Composing kernels inherit spills. ``fdsva_so`` wraps its whole orchestration
in ``fdsva_so_device`` and spills the dominant embedded ``idsva_so`` scratch
by passing ``SCRATCH_IN_SMEM=false``, with no pointer surgery in the caller.

The ``XImats`` / ``XmatsHom`` load helper is also a scratch consumer. It is
called near the top and writes its sincos scratch into ``s_temp``. Put that
call inside the inner, after the repoint, so it follows the placement, or
otherwise guarantee it a valid pointer; never pass it a ``nullptr``
``s_temp`` (see the anti-patterns below).

Reference points that follow this: ``aba_inner``,
``forward_dynamics_inner``, ``minv_inner``, ``fdsva_so_inner`` (the
rank-3 contraction sub-step), ``fdsva_so_device``,
``inverse_dynamics_gradient_device``, ``forward_dynamics_gradient_device``,
``integrator_gradient_device``, ``end_effector_pose_gradient_inner``, and the
``idsva_so`` world inner. Every orchestrator owns its placement inside the
``_device`` body; no kernel-side ``s_temp`` repoint remains.

Project-wide conformance
~~~~~~~~~~~~~~~~~~~~~~~~

Every algorithm's inner owns its scratch placement, not only the second-order
ones. The placement template each inner carries is listed below; a new inner
joins this list:

* ``aba_inner``: ``TEMP_IN_SMEM`` (whole arena) + ``COLD_IN_SMEM`` (surgical cold sub-band)
* ``forward_dynamics_inner``, ``minv_inner``, ``integrator_inner``: ``MINV_F_IN_SMEM`` / ``F_IN_SMEM`` (the F region)
* ``fdsva_so_contract``: ``SCRATCH_IN_SMEM`` (4·NV³)
* ``fdsva_so_device`` (orchestrator): ``SCRATCH_IN_SMEM × FD_GRAD_USE_SPILL × CONTRACT_IN_SMEM`` (the canonical three-lever pattern)
* ``crba_inner``, ``end_effector_pose_gradient_inner``: ``TEMP_IN_SMEM``
* ``idsva_so_body_frame_inner``: ``SCRATCH_IN_SMEM × BC_IN_SMEM`` (whole arena + surgical BC; mutually exclusive per the body tier table)
* ``idsva_so_world_frame_inner``: ``SCRATCH_IN_SMEM × COLD_IN_SMEM`` (whole arena + surgical cold trio Xdown/v_w/a_w; mutually exclusive)
* ``inverse_dynamics_gradient_device``, ``forward_dynamics_gradient_device``: ``SCRATCH_IN_SMEM`` (whole arena via the ``_device`` orchestrator)
* ``integrator_gradient_device``: per-tier rung (Dqdd / dAB / inverse_dynamics_gradient level) via the ``_device`` orchestrator



3. Memory-hierarchy design
--------------------------

* Shared memory is fast but scarce. On sm_120 / RTX 5090 the opt-in
  dynamic-smem cap is ~99 KB, not 227 KB. Query it
  (``cudaDevAttrMaxSharedMemoryPerBlockOptin`` /
  ``grid_get_max_dynamic_shared_memory_bytes``); never hardcode an assumed cap.
* Global ``d_workspace`` is abundant and can be pinned in L2. Spilled buffers
  are recursion-hot, so the generated workspace supports L2 pinning for the
  kernel's lifetime; pinning is off by default because measurements showed no
  consistent gain (see :doc:`resource_tier_system`). With pinning on, a spilled
  access costs roughly an L2 hit rather than an HBM access.
* ``s_temp`` is a reused pool, not one buffer per quantity. Phases overwrite
  it sequentially, so its size is the maximum live footprint across phases, and
  aliasing (``f = vJ = Xdown``) keeps the high-water mark low.
* The ``d_workspace`` layout is a per-timestep slot indexed
  ``&d_workspace[k * GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + <band offset>]``,
  partitioned into the GRAD, SO and FDSVA_SO_SPILL bands. Buffers that are not
  live at the same time may reuse the same region.
* The naming convention carries meaning: ``s_`` is shared memory and ``d_`` is
  device (global) memory. A pointer's prefix must match where it lives.


4. The spill ladder and per-tier picks
--------------------------------------

Each algorithm has a fixed menu of spill rungs (level 0 = everything in smem;
higher levels progressively move buffers to ``d_workspace``). At codegen time,
per robot, ``select_shared_tier_3way(*rung_arenas)`` maps each
``RESOURCE_TIER`` to a rung:

* PERF picks the lowest rung that fits the occupancy-friendly target; if
  nothing fits, it falls through to the most-spilled rung, so big robots
  spill even at PERF. There is no larger smem to use on a ~99 KB device.
* MINIMAL is always the last (most-spilled) rung, for maximum occupancy
  headroom.
* Small robots keep every tier at rung 0, so PERF/LITE/MINIMAL are
  byte-identical, with no spill to pay for.

The arena size and the emitted layout must agree, per tier. The
``*_DYNAMIC_SHARED_MEM_BYTES<TIER>()`` macro, the rung's ``t_count`` in
``select_shared_tier_3way``, and what the kernel writes must all match
for every tier. A mismatch is a silent out-of-bounds. When adding a rung,
prove that the most-spilled rung fits the device cap.


5. Adding or modifying an algorithm
-----------------------------------

#. Write ``X_inner`` block-cooperatively, templated on ``SCRATCH_IN_SMEM``
   (and any surgical sub-flags). Repoint ``s_temp`` at the top; expose
   ``X_*_SMEM_BYTES`` / ``X_*_WORKSPACE_BYTES``.
#. Keep ``X_kernel`` thin: load inputs, set up the output and spill-region
   pointers per tier, call the inner with the per-tier flags, save outputs. No
   compute, no scratch ownership.
#. Register the tier picks with ``select_shared_tier_3way`` over the rung
   arenas; make the per-tier smem-bytes macro emit each rung's size.
#. Pass the per-rung flag to the inner, computed from the rung being
   emitted (for example ``"true" if use_selective_spill else "false"``), not a
   single-valued ``GRID_*_USES_*`` macro, which is only the PERF pick.
#. Add a CUDA equivalence test against the verified ``RBDReference`` Python,
   and exercise it at PERF and at a forced deep spill.


6. Validation
-------------

* Check equivalence against ``RBDReference`` at PERF and at forced spill.
  PERF validates the math; a forced deep spill (set
  ``GRID_CUDA_TARGET_SHARED_MEM_BYTES`` small) validates the spill path, which
  is otherwise never instantiated on small robots and can ship untested.
* Gate before a long timing sweep. A per-algorithm TU that fails to compile
  empties an entire robot/tier column (the per-algorithm TUs link into one
  binary). Compile and equivalence-check a small robot first; only then launch
  the multi-hour sweep.
* Keep numerical gates strict. Investigate fp32 near-misses and record the
  failing inputs. A reviewed benchmark accuracy-warning policy is separate
  from a correctness pass; never silently downgrade a failed assertion.
* Never run CPU-heavy work during a GPU timing sweep; it skews the numbers.
* Spill rungs 0..N-1 that only relocate code must be numerically identical to
  the unspilled path; the only difference is where a pointer points.


7. Anti-patterns
----------------

* Caller-side ``s_temp`` repoint or alias. Placement is the inner's job; a
  kernel that reassigns ``s_temp`` from outside must be migrated.
* Null ``s_temp`` passed to the load helper. In a whole-arena spill rung the
  smem ``s_temp`` slot is ``nullptr``; the ``XImats``/``XmatsHom`` helper
  still dereferences it for sincos scratch and crashes. Repoint first, or keep
  the helper inside the inner.
* A single-valued PERF-pick macro used as a per-rung flag.
  ``GRID_*_USES_DA_DF_SPILL`` equals the PERF pick. Using it as the inner's
  template arg inside an ``if constexpr (RESOURCE_TIER==...)`` branch gives a
  non-PERF tier the wrong flag: it writes the full band into a
  selective-sized arena, a shared-memory out-of-bounds write on big robots.
  Pass the actual per-rung value.
* An arena sized for one rung with the inner flag for another. They must
  match per tier.
* An assumed 227 KB smem cap. sm_120 / RTX 5090 opt-in is ~99 KB. Query it.


Further reading
---------------

* :doc:`codegen_architecture`: the three emission layers and the ``_device``
  / ``_inner`` composition contract.
* :doc:`resource_tier_system`: tiers, spill levels,
  ``select_shared_tier_3way``, L2 pinning, the placement-parameter table, and
  the per-robot tier-to-level map.
* ``docs/design_history/idsva_so_inner_refactor_notes.md``: the second-order
  inner internals behind the rule above and the deferred surgical de-alias
  work.
