Algorithm Design Principles & Best Practices
============================================

**Read this first.** This page is the shared mental model for anyone — human or
agent — writing or modifying a GRiD algorithm. It captures the *why* behind the
generated-code structure so your changes match the existing grain instead of
fighting it. The mechanics live in :doc:`codegen_architecture` (the four emission
layers) and :doc:`resource_tier_system` (tiers + spill); this page is the ethos
that ties them together. If you internalize the rules below, your code will look
like the code already here.

The one-paragraph version
-------------------------

GRiD is a code generator for power users who call hand-tuned, robot-specialized
rigid-body-dynamics kernels directly from their own CUDA and want every cycle and
byte. So: **put all the intelligence in the innermost device function, keep
everything above it a thin shim, and let the inner own its own memory placement.**
Big robots overflow shared memory and must spill to global; the *inner* decides
what spills, the *caller* only sizes the arenas and passes a flag.


1. Smart inners, thin wrappers
------------------------------

Each algorithm ``X`` is emitted in **three layers** (see :doc:`codegen_architecture`):
``X_device`` (the canonical ``__device__`` orchestrator — *all the value is
here*; takes caller-supplied ``s_temp`` + ``d_workspace``, owns its placement),
``X_kernel`` (grid-stride loop over timesteps; allocates smem and calls
``_device``), and ``X`` (host launcher).

Internal ``X_inner`` helpers (and role-specific sub-step helpers like
``fdsva_so_contract``) still exist where useful — they are the placement-free
math building blocks that one ``_device`` may call into another to amortize
shared work (e.g. an XImats load).

* **Value is concentrated in ``X_device`` (and its ``_inner`` building blocks).**
  The kernel and host layers are noise to the performance user; keep them
  mechanical so there is exactly one place where the hard decisions (placement,
  spill, orchestration) live, audit, and get fixed.
* **``_device`` is block-cooperative, not fixed-lane.** Every parallel region is
  a block-stride loop (``for (i = tid; i < N; i += blockDim...)``), so any block
  size that fits is correct. Batching is the kernel's grid-stride loop over
  ``blockIdx``. Never assume a specific thread count.
* **Compose by calling other algorithms' ``_inner``.** A higher-level algorithm
  (e.g. ``fdsva_so`` embeds ``idsva_so``) loads ``XImats`` once at the top of its
  ``_device`` and calls placement-free ``_inner`` variants of sub-algorithms so
  the load is paid once, not per sub-algorithm. (Historically a separate
  auto-allocating ``_device`` wrapper existed for orchestrators; that
  training-wheels layer was dropped in 2026-05 — orchestrators are now a clean
  three layers like every other algorithm.)


2. The inner owns its memory placement (the central rule)
---------------------------------------------------------

When an inner's scratch does not fit shared memory, **the decision of where it
lives belongs to the inner, never the caller.**

Every ``*_inner`` is templated on a placement flag (``SCRATCH_IN_SMEM`` — or,
equivalently, the ``RESOURCE_TIER``) and takes **both** pointers, ``s_temp``
(shared) and ``d_workspace`` (global). At the very top of the body it selects::

    template <typename T, bool SCRATCH_IN_SMEM = true>
    __device__ void foo_inner(..., T *s_temp, T *d_workspace, ...) {
        if constexpr (!SCRATCH_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }
        // ... the rest of the body is UNCHANGED; it just uses s_temp ...
    }

The **caller** (a kernel, a device wrapper, or a composing kernel) is a thin
shim: it sizes both arenas from the inner's exposed ``*_SMEM_BYTES`` /
``*_WORKSPACE_BYTES`` constants, hands both pointers in, and passes the per-tier
flag. **It must never hard-repoint or alias the inner's scratch from outside.**

Why this is a rule, not a preference:

* **Single source of truth.** The inner sizes *and* places its own scratch, so
  the shared-mem-bytes macro, the workspace-bytes macro, and the pointer
  arithmetic can never drift apart across callers.
* **Improvements propagate for free.** If you later teach an inner to spill only
  its *cold* buffers (keeping the hot loop in smem), every caller — the standalone
  kernel *and* every composing kernel — inherits it just by passing the flag.
* **Backward-compatible by default.** ``SCRATCH_IN_SMEM = true`` keeps small
  robots byte-identical; only robots that overflow flip it to ``false``.

**Composing kernels inherit spills.** ``fdsva_so`` wraps its whole orchestration
in ``fdsva_so_device`` and spills the (dominant) embedded ``idsva_so`` scratch
purely by passing ``SCRATCH_IN_SMEM=false`` — no pointer surgery in the caller.

**The ``XImats`` / ``XmatsHom`` load helper is also a scratch consumer.** It is
called near the top and writes its sincos scratch into ``s_temp``. Put that call
*inside* the inner (after the repoint) so it follows the placement, or otherwise
guarantee it a valid pointer — **never pass it a ``nullptr`` ``s_temp``** (see the
anti-patterns below; this was a real crash class).

Reference points that already follow this: ``aba_inner``,
``forward_dynamics_inner``, ``direct_minv_inner``, ``fdsva_so_inner`` (the
rank-3 contraction sub-step), ``fdsva_so_device``,
``inverse_dynamics_gradient_device``, ``forward_dynamics_gradient_device``,
``integrator_gradient_device``, ``end_effector_pose_gradient_inner``, and the
``idsva_so`` world inner. As of 2026-05-28 every orchestrator owns its
placement inside the ``_device`` body — no kernel-side ``s_temp`` repoint
remains.


3. Memory-hierarchy design
--------------------------

* **Shared memory is fast but scarce, and smaller than you think.** On
  sm_120 / RTX 5090 the *opt-in* dynamic-smem cap is only **~99 KB**, not 227 KB.
  **Query it** (``cudaDevAttrMaxSharedMemoryPerBlockOptin`` /
  ``grid_get_max_dynamic_shared_memory_bytes``); never hardcode an assumed cap.
* **Global ``d_workspace`` is abundant and L2-pinnable.** Spilled buffers are
  recursion-hot, so the generated workspace is pinned in L2 for the kernel's
  lifetime — a spilled access then costs roughly an L2 hit, not an HBM round-trip.
* **``s_temp`` is a reused pool**, not one buffer per quantity — phases overwrite
  it sequentially, so its size is the *max* live footprint across phases, and
  aliasing (``f = vJ = Xdown``) is deliberate to keep the high-water mark low.
* **``d_workspace`` layout** is a per-timestep slot indexed
  ``&d_workspace[k * GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + <band offset>]``,
  partitioned into the GRAD / SO / FDSVA_SO_SPILL bands. Buffers not live at the
  same time may safely reuse the same region.
* **Naming convention is load-bearing:** ``s_`` = shared, ``d_`` = device/global.
  A pointer's prefix must match where it actually lives. (Renaming a stray
  device pointer that was named ``s_`` is exactly the kind of fix we make.)


4. The spill ladder and per-tier picks
--------------------------------------

Each algorithm has a fixed *menu* of spill rungs (level 0 = everything in smem;
higher levels progressively move buffers to ``d_workspace``). At codegen time,
per robot, ``select_shared_tier_3way(*rung_arenas)`` maps each
``RESOURCE_TIER`` to a rung:

* **PERF** picks the lowest rung that fits the (occupancy-friendly) target; if
  *nothing* fits, it falls through to the most-spilled rung — so **big robots
  spill even at PERF.** That is the whole point: there is no "just use more smem"
  on a ~99 KB device.
* **MINIMAL** is always the last (most-spilled) rung — maximum occupancy headroom.
* **Small robots** keep every tier at rung 0, so PERF/LITE/MINIMAL are
  byte-identical (no spill to pay for).

**The arena size and the emitted layout must agree, per tier.** The
``*_DYNAMIC_SHARED_MEM_BYTES<TIER>()`` macro, the rung's ``t_count`` in
``select_shared_tier_3way``, and what the kernel actually writes must all match
for every tier. A mismatch is a silent out-of-bounds. When you add a rung, prove
the most-spilled rung fits the device cap.


5. Recipe: adding or modifying an algorithm
-------------------------------------------

#. **Write ``X_inner``** block-cooperatively, templated on ``SCRATCH_IN_SMEM``
   (and any surgical sub-flags). Repoint ``s_temp`` at the top; expose
   ``X_*_SMEM_BYTES`` / ``X_*_WORKSPACE_BYTES``.
#. **Keep ``X_kernel`` thin:** load inputs, set up the output and spill-region
   pointers per tier, call the inner with the per-tier flags, save outputs. No
   compute, no scratch ownership.
#. **Register the tier picks** with ``select_shared_tier_3way`` over the rung
   arenas; make the per-tier smem-bytes macro emit each rung's size.
#. **Pass the *per-rung* flag to the inner**, computed from the rung you are
   emitting (e.g. ``"true" if use_selective_spill else "false"``) — **not** a
   single-valued ``GRID_*_USES_*`` macro (that macro is only the PERF pick).
#. **Add a CUDA equivalence test** vs the verified ``RBDReference`` Python, and
   exercise it at PERF *and* at a forced deep spill.


6. Validation discipline
------------------------

* **Equivalence vs ``RBDReference`` at PERF *and* forced spill.** PERF validates
  the math; a forced deep spill (set ``GRID_CUDA_TARGET_SHARED_MEM_BYTES`` small)
  validates the spill path — which is otherwise *never instantiated* on small
  robots and can ship untested.
* **Gate before a long timing sweep.** A per-algo TU that fails to compile empties
  an entire robot/tier column (the per-algo TUs link into one binary). Compile +
  equivalence-check a small robot first; only then launch the multi-hour sweep.
* **Use non-blocking gates for near-misses.** Float32 second-order derivatives on
  big robots accumulate error; let a borderline equivalence *log* rather than
  abort an overnight run — but **never loosen a tolerance to mask a true
  divergence.** Investigate near-misses; don't paper over them.
* **Never run CPU-heavy work during a GPU timing sweep** — it skews the numbers.
* **Spill rungs 0..N-1 that only relocate code must be numerically identical** to
  the unspilled path; the only difference is where a pointer points.


7. Anti-patterns to avoid (each one bit us once)
------------------------------------------------

* **Caller-side ``s_temp`` repoint / alias.** Placement is the inner's job; a
  kernel that reassigns ``s_temp`` from outside is a smell to migrate.
* **Null ``s_temp`` to the load helper.** In a whole-arena spill rung the smem
  ``s_temp`` slot is ``nullptr``; the ``XImats``/``XmatsHom`` helper still
  dereferences it for sincos scratch → crash. Repoint first (or keep the helper
  inside the inner).
* **A single-valued PERF-pick macro as a per-rung flag.** ``GRID_*_USES_DA_DF_SPILL``
  equals the *PERF* pick. Using it as the inner's template arg inside an
  ``if constexpr (RESOURCE_TIER==...)`` branch gives a non-PERF tier the wrong
  flag → it writes the full band into a selective-sized arena → shared OOB on big
  robots. Pass the actual per-rung value.
* **Arena sized for one rung, inner flag for another.** They must match per tier.
* **Assuming a 227 KB smem cap.** sm_120 / RTX 5090 opt-in is ~99 KB. Query it.


Go deeper
---------

* :doc:`codegen_architecture` — the three emission layers and ``_device`` /
  ``_inner`` composition contract.
* :doc:`resource_tier_system` — tiers, spill levels, ``select_shared_tier_3way``,
  L2 pinning, the placement-parameter table, and the per-robot tier→level map.
* ``docs/idsva_so_inner_refactor_notes.md`` — the inner-owns-placement standard,
  the project-wide conformance audit, and the deferred surgical de-alias work.
* ``HANDOFF.md`` — the live backlog / current todo list.
