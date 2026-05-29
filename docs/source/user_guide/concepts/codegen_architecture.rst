Codegen Architecture
====================

Every GRiD algorithm is emitted in three layers (``_host`` / ``_kernel`` /
``_device``). Knowing the layering helps when you want to compose generated
functions, call kernels from your own CUDA host code, or read the emitter
source in ``GRiDCodeGenerator/algorithms/``.

.. note::

   New here (human or agent)? Read :doc:`design_principles` first — it is the
   shared mental model (smart inners / thin wrappers, *the inner owns its memory
   placement*, the spill ladder, validation discipline, and the anti-patterns to
   avoid). This page covers the *mechanics* of the three layers; that page covers
   the *ethos* behind them.

The three emission layers
-------------------------

For each algorithm ``X`` (e.g. ``inverse_dynamics``, ``forward_dynamics``,
``crba``, ``fdsva_so``), the codegen emits:

* ``X_device`` — a ``__device__`` function with the **canonical caller-supplied
  buffer contract**. The caller passes pointers to inputs, outputs, ``s_temp``
  (shared scratch pool), and ``d_workspace`` (global scratch). The ``_device``
  function owns its **scratch placement**: a single ``if constexpr (!SCRATCH_IN_SMEM)
  { s_temp = d_workspace; }`` at the top routes the whole pool to global for
  spilled tiers. After that repoint, every consumer below (XImats helper, sub-
  inners, etc.) follows the placement, so the kernel never repoints ``s_temp``
  from the outside. This is the *inner-owns-placement* discipline; see
  :doc:`design_principles`.

* ``X_kernel`` — a ``__global__`` entry point that handles **batch scheduling**
  (``blockIdx.x`` loops over timesteps) and **global ↔ shared memory transfer**.
  It allocates ``__shared__`` smem for inputs/outputs and the ``s_temp`` pool,
  loads inputs, calls ``X_device``, and writes outputs back. Per-tier dispatch
  ( ``RESOURCE_TIER`` template ) picks the spill flags; the kernel never decides
  placement itself.

* ``X`` (no suffix) — a **host function** that wraps ``X_kernel`` and handles
  H↔D copies for inputs and outputs.

Inner helpers (``X_inner``, sub-step helpers like ``fdsva_so_contract``) still
exist where useful, but they are **internal to ``X_device``** — not part of the
external surface. Sub-algorithm composition routes through other algorithms'
``_inner`` helpers when they are placement-free building blocks (e.g.
``fdsva_so_device`` calls ``direct_minv_inner`` and ``forward_dynamics_inner``
to reuse one XImats load across all of them).

Why orchestration moved into ``_device`` (history)
--------------------------------------------------

Pre-2026 the emitter shipped *four* layers: ``_inner`` (math),
``_full_inner`` (orchestrator + placement), ``_device`` (auto-allocating
training-wheels wrapper), ``_kernel``. The auto-allocating ``_device`` had
exactly one consumer (the equivalence runner) and its existence forced two
confusing things:

1. **Two functions with overlapping roles** — orchestration was duplicated in
   ``_full_inner`` (called from the kernel) and in the auto-allocating
   ``_device`` (which essentially re-emitted the same orchestration under
   ``SCRATCH_IN_SMEM=true``).
2. **Inconsistent placement contract** — some algorithms repointed ``s_temp``
   from the kernel (around the now-removed ``_device``), others repointed it
   inside ``_full_inner``. Reviewers had to chase which.

The 2026 rename collapses the two: ``_full_inner`` becomes the canonical
``_device`` (caller-supplied ``s_temp`` + ``d_workspace`` + spill flags;
``__device__ __forceinline__``; owns its placement), and the old auto-
allocating ``_device`` is gone. Inline-CUDA users either embed ``_device``
inside their own kernel (passing their own ``s_temp``) or call ``_kernel``
directly for batches. The host wrapper is unchanged.

Nested composition
------------------

The split exists for **nested composition**. Higher-level algorithms call
other algorithms' ``_inner`` (the placement-free math) directly to reuse one
expensive ``XImats`` load.

For example, second-order forward dynamics
(`_fdsva_so.py <https://github.com/A2R-Lab/GRiD/blob/main/GRiDCodeGenerator/algorithms/_fdsva_so.py>`_)
needs both forward dynamics and direct inverse-mass-matrix outputs internally.
``fdsva_so_device`` loads XImats once at the top, then calls the placement-free
``_inner`` variants:

.. code-block:: text

   fdsva_so_device
     ├── [s_temp repoint based on SCRATCH_IN_SMEM]
     ├── load_update_XImats()       # once
     ├── direct_minv_inner()        # reuses s_XImats
     ├── forward_dynamics_inner()   # reuses s_XImats
     ├── fd_gradient_inline()       # may surgically spill to d_fd_grad_spill
     ├── idsva_so_{world,body}_inner()
     └── fdsva_so_inner()           # the rank-3 contraction step

If the ``_inner`` building blocks were collapsed into their owning ``_device``,
the compositional algorithms would pay the XImats load three times instead of
once. The separation is a performance contract, not a stylistic preference.

Concrete signatures (RNEA / inverse_dynamics)
---------------------------------------------

.. code-block:: cuda

   // pure math; assumes s_XImats already loaded; placement-free
   template <typename T>
   __device__ void inverse_dynamics_inner(
       T *s_vaf, const T *s_q, const T *s_qd,
       /* topology + scratch */ T *s_temp,
       const T gravity);

   // canonical _device: caller-supplied buffers + scratch; owns placement
   template <typename T>
   __device__ void inverse_dynamics_device(
       T *s_c, const T *s_q, const T *s_qd,
       const robotModel<T> *d_robotModel, const T gravity);

   // global entry, batched over timesteps; per-tier RESOURCE_TIER dispatch
   template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
   __global__ void inverse_dynamics_kernel(
       T *d_c, const T *d_q_qd, const int stride_q_qd,
       const robotModel<T> *d_robotModel, const T gravity,
       const int NUM_TIMESTEPS);

   // CPU launcher with H↔D copies
   template <typename T, bool USE_QDD_FLAG=false, bool USE_COMPRESSED_MEM=false>
   __host__ void inverse_dynamics(
       gridData<T> *hd_data, const robotModel<T> *d_robotModel,
       const T gravity, const int num_timesteps,
       dim3 block_dimms, dim3 thread_dimms, cudaStream_t *streams);

Orchestrator signature (fdsva_so / id_du / fd_du / integrator_gradient)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The orchestrators take the placement flags + a ``d_workspace`` pointer in
addition to the standard inputs:

.. code-block:: cuda

   template <typename T,
             bool SCRATCH_IN_SMEM = true,        // s_temp pool location
             bool FD_GRAD_USE_SPILL = false,     // selective spill bits
             bool CONTRACT_IN_SMEM = true>       // (fdsva_so only)
   __device__ __forceinline__
   void fdsva_so_device(
       T *s_df2, T *s_idsva_so, T *s_Minv, T *s_df_du, T *s_qdd,
       const T *s_q, const T *s_qd, const T *s_u,
       /* XImats helpers */
       T *s_temp,              // smem pool; ignored when !SCRATCH_IN_SMEM
       T *d_workspace,         // global pool; ignored when SCRATCH_IN_SMEM
       T *d_fd_grad_spill,     // band-spill region; nullptr unless FD_GRAD_USE_SPILL
       T *s_fdsva_temp,        // contraction scratch; per CONTRACT_IN_SMEM
       const robotModel<T> *d_robotModel, const T gravity);

The kernel emitter passes per-tier ``true``/``false`` literals for the
template flags and threads the right ``d_workspace`` offsets in. Inline-CUDA
users size their smem from ``*_DEVICE_INLINE_SMEM_BYTES<T, TIER>()`` and
``d_workspace`` from ``*_DEVICE_INLINE_WORKSPACE_BYTES<T, TIER>()``.

Thread-count assumptions
------------------------

GRiD emits a ``MAX_PERF_LEVEL_THREADS`` constant per generated header, computed
from the robot's DVA parallelism (rounded up to a warp, capped at 512) —
for iiwa14 it is 352, for go2_fixed it is 288, and so on. The host
wrappers default to launching with ``dim3(MAX_PERF_LEVEL_THREADS, 1, 1)``.

After the v2.0 cuBLASDx removal, ``MAX_PERF_LEVEL_THREADS`` is **a hint, not
an enforced floor**. Every emitted ``X_inner`` does block-cooperative
compute on one timestep — threads within a block split work via
*block-stride loops* (the ``gen_add_parallel_loop`` helper emits
``for (int i = threadIdx.x + threadIdx.y*blockDim.x; i < max_val;
i += blockDim.x*blockDim.y)``). Any block size that fits per-block
covers the work correctly. Batching across timesteps is handled by
the outer ``X_kernel`` via a *grid-stride loop* over ``blockIdx`` —
each block processes one or more timesteps.

CUDA-inline users can launch GRiD kernels with any block size that
suits their outer kernel. See :doc:`cublasdx_removal_design` for the
rationale.

Design sweet spot: tens-to-hundreds of parallel computations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The one-timestep-per-block layout — block-cooperative compute inside,
grid-stride over batch outside — is **tuned for batch sizes in the
tens to hundreds**. Many batch robotics workloads (MPC shooting nodes,
trajectory optimization horizons, behavioral cloning rollouts, real-time
control with N robots) sit squarely in that range, which is the design
target.

For very small batches (N=1-8) the per-block fixed overhead dominates,
and a design that packed multiple timesteps per block could be faster;
for very large batches (N=10k+) a design that splits one timestep
across multiple blocks could expose more parallelism. Neither is what
GRiD optimizes for. If your workload sits at one of those extremes, a
codegen layered on a different parallelism map (or an entirely
different library) will likely beat GRiD; for the tens-to-hundreds
range, GRiD's layout is the right tool.

The ``MAX_PERF_LEVEL_THREADS`` constant is what the codegen picks as the
best block-cooperative thread count for *this robot* (DOF-aware,
warp-rounded). External callers are free to override (see
:py:meth:`grid_rbd.RobotHandle.set_threads_per_block` or
``grid_rbd_set_threads_per_block`` in the C ABI), but smaller block
sizes will be slower at the same batch size (work-per-block stays
constant; fewer threads cover it).

The codegen currently still emits ``__launch_bounds__(MAX_PERF_LEVEL_THREADS)``
on each ``X_kernel``. That attribute drops in phase B1 (any-thread-count
emission); see the design doc for the rollout sequence.

See also
--------

* :doc:`algorithms/index` — algorithm-level docs.
* ``GRiDCodeGenerator/README.md`` (repo root) — codegen-helper
  reference for emitter authors.
