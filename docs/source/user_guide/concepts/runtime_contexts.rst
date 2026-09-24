Runtime contexts
================

A compiled robot artifact (the per-robot ``.so`` the bindings build and cache)
is immutable: it never owns device memory. Everything a call needs at run time
lives in a **runtime context**: the device arena (``gridData``) and the
allocator pool it was carved from, the robot tables (inertias, transforms,
joint dynamics, attached tools), the CUDA streams, the plant staging buffers,
the per-algorithm launch overrides and a **device profile** captured at
creation. Every native entry point, on every surface (the C ABI the numpy
handle drives, every JAX FFI handler, every torch op), names its context by id
and resolves it at call time. The design note is
``docs/open-tasks/w04b_runtime_split_design_2026-09-22.md`` (increment B1).

The default context
-------------------

A handle obtained from ``register_robot`` / ``get_robot`` / ``load_robot``
dispatches to the artifact's **default context** (``handle.ctx_id == 0``, an
alias). It is created lazily on the first call, shared by every handle over the
same artifact that did not ask for its own, closed by the last owner (or by
``close_arena()``), and re-created on the next call. This is the historical
lifecycle, unchanged.

Explicit contexts
-----------------

``handle.context(workspace_slots=...)`` creates a **new** context on the same
artifact and returns a handle bound to it (numpy); ``jax_handle.context()`` and
``torch_handle.context()`` return the corresponding views. An explicit context
has its own arena and allocator pool, its own tables and launch overrides:
mutating one context (``set_inertia_params``, ``attach_tool``,
``set_threads_for``...) is invisible to the others, and an allocation failure in
one leaves the others usable. Closing the handle closes the context: no new
calls are admitted, the calls already admitted drain, the device completes,
then the memory is released. Contexts are the unit of isolation for concurrent
pipelines on the single GPU; they are not a multi-GPU mechanism.

Identity rules
--------------

* Ids are 64-bit and salted per loaded artifact. An id minted by another
  robot's artifact is rejected (``unknown context id, or a context of another
  robot artifact``).
* A closed id never resolves again (``context is closed``), and never aliases
  a newly created context; a context in the middle of closing refuses new
  admissions (``context is closing``).
* Lookup takes a strong execution reference atomically with the open check,
  so a call that was admitted always runs against a live context even if
  another thread closes it concurrently.

Device profile
--------------

``handle.device_profile`` is the record captured when the handle's context was
created: the device compute capability and the one the artifact was compiled
for (a mismatch refuses to create the context), total and free device memory at
creation, the arena bytes and workspace slot count that were fitted, the
compiled-in ``max_batch``, the opt-in shared-memory cap and whether the arena
was carved from a caller-owned slab. A deployment can assert it; the
constrained-memory behaviour (a slab too small for the arena) fails cleanly at
creation with nothing published.

What stays for later increments
-------------------------------

B1 keeps today's synchronisation: setters still fence with a device-wide
synchronise and the numpy input pack still drains the device. Per-call leases
(B3) and the mutation version check in autograd backward (B2) follow, each as
its own gated increment.

Inline-CUDA consumers of ``grid.cuh`` are untouched: ``init_gridData`` /
``init_gridData_checked`` / ``close_grid`` keep their signatures (the checked
initializer merely gained an optional trailing allocator-pool argument), and
the pool-less ``grid_device_alloc`` / ``grid_device_free`` overloads remain the
default-pool spellings.
