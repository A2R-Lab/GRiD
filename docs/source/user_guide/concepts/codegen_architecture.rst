Codegen Architecture
====================

Every GRiD algorithm is emitted in four layers. Knowing the layering
helps when you want to compose generated functions, call kernels from
your own CUDA host code, or read the emitter source in
``GRiDCodeGenerator/algorithms/``.

The four emission layers
------------------------

For each algorithm ``X`` (e.g. ``inverse_dynamics``, ``forward_dynamics``,
``crba``), the codegen emits:

* ``X_inner`` — a device function that does the **core math**. Inputs
  are assumed to be in shared memory; XImats and topology helpers must
  already be loaded; a pointer to scratch shared memory is passed in.
  Many threads cooperate via parallel loops with intra-block
  synchronization.

* ``X_device`` — a device function that handles **scratch allocation
  and matrix setup** for ``X_inner``. Inputs are assumed to be in
  shared memory; outputs are returned in shared memory.

* ``X_kernel`` — a ``__global__`` entry point that handles **batch
  scheduling** (``blockIdx.x`` loops over timesteps) and **global ↔
  shared memory transfer**. Wraps ``X_device``.

* ``X`` (no suffix) — a **host function** that wraps ``X_kernel`` and
  handles H↔D copies for inputs and outputs.

Why ``_inner`` and ``_device`` are separate
-------------------------------------------

The split exists for **nested composition**. Higher-level algorithms
call ``X_inner`` directly to reuse one expensive ``XImats`` load.

For example, second-order forward dynamics
([_fdsva_so.py](https://github.com/A2R-Lab/GRiD/blob/main/GRiDCodeGenerator/algorithms/_fdsva_so.py))
needs both forward dynamics and direct inverse-mass-matrix outputs
internally. Rather than calling the ``_device`` wrappers (which would
each redo the XImats load), ``fdsva_so_device`` loads XImats once and
calls the ``_inner`` variants:

.. code-block:: text

   fdsva_so_device
     ├── load_update_XImats()       # once
     ├── direct_minv_inner()        # reuses s_XImats
     ├── forward_dynamics_inner()   # reuses s_XImats
     └── fdsva_so_inner()           # reuses s_XImats, Minv, qdd

If ``_inner`` and ``_device`` were collapsed into a single function,
the compositional algorithms would pay the XImats load three times
instead of once. The two-layer separation is a peformance contract,
not a stylistic preference.

Concrete signatures (RNEA / inverse_dynamics)
---------------------------------------------

.. code-block:: cuda

   // pure math; assumes s_XImats already loaded
   template <typename T>
   __device__ void inverse_dynamics_inner(
       T *s_vaf, const T *s_q, const T *s_qd,
       /* topology + scratch */ T *s_temp,
       const T gravity);

   // allocates scratch, loads XImats, calls _inner
   template <typename T>
   __device__ void inverse_dynamics_device(
       const T *s_q, const T *s_qd,
       const robotModel<T> *d_robotModel, const T gravity);

   // global entry, batched over timesteps
   template <typename T>
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

Thread-count assumptions
------------------------

GRiD emits a ``SUGGESTED_THREADS`` constant per generated header, computed
from the robot's DVA parallelism (rounded up to a warp, capped at 512) —
for iiwa14 it is 352, for go2_fixed it is 288, and so on. The host
wrappers default to launching with ``dim3(SUGGESTED_THREADS, 1, 1)``.

After the v2.0 cuBLASDx removal, ``SUGGESTED_THREADS`` is **a hint, not
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

The ``SUGGESTED_THREADS`` constant is what the codegen picks as the
best block-cooperative thread count for *this robot* (DOF-aware,
warp-rounded). External callers are free to override (see
:py:meth:`grid_rbd.RobotHandle.set_threads_per_block` or
``grid_rbd_set_threads_per_block`` in the C ABI), but smaller block
sizes will be slower at the same batch size (work-per-block stays
constant; fewer threads cover it).

The codegen currently still emits ``__launch_bounds__(SUGGESTED_THREADS)``
on each ``X_kernel``. That attribute drops in phase B1 (any-thread-count
emission); see the design doc for the rollout sequence.

See also
--------

* :doc:`algorithms/index` — algorithm-level docs.
* ``GRiDCodeGenerator/README.md`` (repo root) — codegen-helper
  reference for emitter authors.
