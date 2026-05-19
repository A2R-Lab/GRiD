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

Each emitted ``X_kernel`` carries
``__launch_bounds__(SUGGESTED_THREADS)``. ``SUGGESTED_THREADS`` is
computed at codegen time from the robot's DVA parallelism (rounded up
to a warp, capped at 512) — for iiwa14 it is 352, for go2_fixed it is
288, and so on. The generated host wrappers also hardcode
``thread_dimms = dim3(SUGGESTED_THREADS, 1, 1)`` when launching.

This pinning has two consequences for users calling GRiD primitives
from inside their own kernels:

* **cuBLASDx call sites** (in
  ``GRiDCodeGenerator/helpers/_lin_alg_helpers.py``) ``static_assert``
  that the launching block has at least ``gemm_min_block_threads<T,
  M, N, K>()`` threads. cuBLASDx is what enforces the floor.

* **GLASS SIMT paths** (the L1/L2/L3 helpers in ``GLASS/src/``) are
  thread-count-agnostic — they already use grid-stride loops. The
  ``__launch_bounds__`` pinning is an occupancy hint, not a
  correctness requirement, for SIMT-only code paths.

External callers wanting to launch GRiD kernels with different block
sizes (e.g. to fit GRiD primitives inside a larger user kernel) are
the motivation for **compat-mode** codegen — see the *Any-thread-count
design* page for the proposed two-mode emission scheme that drops
``__launch_bounds__`` and falls back to GLASS SIMT when below the
cuBLASDx threshold.

See also
--------

* :doc:`algorithms/index` — algorithm-level docs.
* ``GRiDCodeGenerator/README.md`` (repo root) — codegen-helper
  reference for emitter authors.
