Library-safe initialization and cleanup
=======================================

Every generated ``grid.cuh`` builds the per-robot device tables once, on the
host: the spatial transforms and inertias (``d_XImats``), the topology helpers,
the joint limits and, when opted in, the mutable runtime tables (inertia,
transform, joint dynamics). Historically these initializers used the
``gpuErrchk`` policy: on any CUDA error they printed, called
``cudaDeviceReset()`` and ``exit()``-ed the process. That is right for a
benchmark binary and wrong for a library embedded in a Python interpreter or a
long-lived service. Since 2026-09-22 the header emits a second, nonterminating
spelling of the same construction and destruction logic.

The contract
------------

.. code-block:: cpp

   #include "grid.cuh"

   grid::robotModel<float> *model = nullptr;
   const char *failed_op = nullptr;
   cudaError_t e = grid::init_robotModel_checked<float>(&model, &failed_op);
   if (e != cudaSuccess) {
       // model == nullptr; nothing acquired by this attempt is still allocated
       throw std::runtime_error(std::string(failed_op) + ": " + cudaGetErrorString(e));
   }
   float *limits = nullptr;
   e = grid::init_joint_limits_checked<float>(&limits, &failed_op);
   ...
   e = grid::free_robotModel_checked<float>(model, &failed_op);   // reportable cleanup

* ``init_robotModel_checked`` / ``init_joint_limits_checked`` /
  ``init_XImats_checked`` / ``init_topology_helpers_checked`` and the
  ``init_*_params_checked`` runtime-table initializers return ``cudaError_t``,
  never ``exit``, ``abort`` or ``cudaDeviceReset``. Every owned pointer is
  null before the first fallible call; construction stops at the first failed
  operation; every resource acquired by *that attempt* is released in reverse
  order; ``*out`` is written on complete success only. Host allocations are
  checked too (a failed ``calloc`` reports ``cudaErrorMemoryAllocation``);
  the joint-limit table no longer touches the heap at all.
* ``failed_op`` (optional) receives a static string naming the operation that
  failed (``"cudaMalloc(d_XImats)"``, ``"cudaMemcpy(d_robotModel)"``,
  ``"calloc(h_inertia_params)"``, ...). The **primary** error is what you get
  back: a cleanup failure during rollback never overwrites it.
* ``free_robotModel_checked(model, &failed_op)``: ``nullptr`` is a no-op
  returning ``cudaSuccess``. The struct is copied back to recover the nested
  device pointers; if that copy fails, nothing else is touched, the copy error
  is returned and the nested arrays leak (documented limitation, it means the
  context is already unusable). Cleanup continues past a failed ``cudaFree``
  and returns the first error.
* **Device affinity.** A model is bound to the device that was current when
  it was built. ``free_robotModel_checked`` validates that with
  ``cudaPointerGetAttributes`` and returns ``cudaErrorInvalidDevice`` (freeing
  nothing) if another device is current. Ownership is never mixed across
  devices silently.
* **Lost contexts.** After a sticky/context-invalidating error
  (``cudaErrorIllegalAddress`` and friends) every CUDA call fails; the checked
  functions report that faithfully and clean up best-effort. They do not, and
  cannot, recover the context.
* **Threading.** The checked API keeps no shared state: each call reports
  through its own return value and out-parameters, so concurrent construction
  of different models on one device is well-defined (CUDA runtime calls are
  thread-safe). The legacy sticky slot (below) is a single non-atomic
  process-wide first-error slot and is *not* part of this contract.

Owning handle
-------------

``grid::robotModel_owner<T>`` is a noncopyable, movable RAII wrapper:

.. code-block:: cpp

   grid::robotModel_owner<float> owner;
   if (owner.init(&failed_op) != cudaSuccess) { ... }
   use(owner.get());
   cudaError_t ce = owner.free(&failed_op);   // explicit, reportable
   // or let the destructor free best-effort; release() hands the raw pointer back

Legacy spellings
----------------

``init_robotModel()``, ``init_joint_limits()``, ``init_XImats()``,
``init_topology_helpers()``, ``init_*_params()`` and ``free_robotModel()`` are
thin wrappers over the checked functions that apply the historical policy:
print the failed operation, then ``gpuAssert`` (``cudaDeviceReset``+``exit``
by default; under ``-DGRID_GPUERRCHK_NO_EXIT`` record the first error in the
sticky slot readable through ``grid_last_error()`` /
``grid_consume_last_error()`` and return ``nullptr``). Their signatures and
behaviour are unchanged, and there is exactly one construction/destruction
implementation underneath. The Python bindings (``grid_rbd``) construct the
model through the checked path and translate a failure into a Python
exception naming the operation.

Testing the failure paths
-------------------------

Two host-only macros are the fault-injection seams: ``GRID_CUDA_CALL(expr)``
wraps every CUDA runtime call the checked initializers make and
``GRID_HOST_ALLOC(expr)`` every host allocation. They default to the bare
expression (no runtime cost, never used in device code). A test translation
unit defines them *before* including ``grid.cuh`` to fail the N-th call
deterministically; ``test/cuda_equivalents/test_cuda_safe_init.py`` does this
for every call in the construction sequence, for host allocation failure, for
copy-back and nested-free failures during destruction, and for the legacy
exit path (in a subprocess), on a serial arm and on a floating-base robot with
every runtime table enabled.
