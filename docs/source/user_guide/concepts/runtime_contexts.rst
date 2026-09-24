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
and resolves it at call time. This page is the durable contract; the
internal design history stays in the maintainers' local notes.

The default context
-------------------

A handle obtained from ``register_robot`` / ``get_robot`` / ``load_robot``
dispatches to the artifact's **default context** (``handle.ctx_id == 0``, an
alias). It is created lazily on the first call, shared by every handle over the
same artifact that did not ask for its own, closed by the last owner (or by
the runner's ``close_arena()``), and re-created on the next call. This is the
historical lifecycle, unchanged. Every re-creation is a new incarnation with a
new model epoch (below), so work captured against the old one is refused.

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
then the memory is released. The handle is the one owner of its context: a
handle dropped without ``close()`` is finalized at garbage collection, an
explicit ``close()`` is idempotent, and a jax/torch view references the
handle, never the context. ``workspace_slots=N`` caps the per-block workspace
slot count on that context (``0`` = auto-fit; the explicit cap beats the
``GRID_WORKSPACE_TIMESTEP_SLOTS`` environment override, which beats the
auto-fit; a batch above the cap grid-strides). Contexts are the unit of
isolation for concurrent pipelines on the single GPU; they are not a
multi-GPU mechanism.

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

Mutation, model versions and autograd
-------------------------------------

Every call is **admitted** against its context: compute calls hold the
context's admission lock *shared* for their whole submission, and every
**model mutation** — ``set_inertia_params``, ``set_transform_params``,
``set_joint_dynamics``, hence ``attach_tool`` / ``detach_tool`` — and every
launch override (``set_threads_per_block``, ``set_threads_for``...) holds it
*exclusive*. A mutation is therefore ordered after every admitted call and
before every later one: a thread hammering ``forward_dynamics`` while another
swaps the inertia table sees either the old or the new physics, never a torn
table, and a launch override is never read half-written.

Each model mutation gives the context a new **model version**
(``handle.model_version``): versions are drawn from one artifact-wide,
strictly increasing epoch, bumped on every context creation and every mutation
of any context, so no two (context incarnation, model state) pairs ever share
a value — every mutation gives its context a strictly larger value (values are
not contiguous: creations elsewhere consume epochs too), a new context or a
re-created default gets a fresh value, launch overrides do not bump it. The
version is what makes autograd honest across mutations:

* a torch / JAX **forward** stamps the version it ran under **on the device**,
  as part of its own launch (``stamp_out`` on the torch op, a second output of
  the ``_stamped`` FFI twin), so under ``jax.jit`` or a captured graph the stamp
  is produced when the forward *executes*, not when it was traced;
* the matching **backward** hands the stamp to each gradient op
  (``stamp_expect`` / the ``_checked`` FFI twin), which reads it inside its own
  admission scope and refuses to run if the model has moved on:
  ``model mutated between forward and backward (version A -> B); recompute the
  forward``. A backward never silently differentiates a model its forward did
  not see.

The same compiled function keeps working across mutations — a jitted
``jax.grad`` called after ``set_inertia_params`` differentiates the new model,
because its forward and backward stamp and check within one execution. Only a
forward whose backward is deferred across a mutation (or across a re-creation
of the default context) is refused. The check costs the gradient op one 4-byte
device-to-host read (a stream sync) per backward; direct (non-autograd)
gradient calls do not pay it. A captured CUDA graph is different: its stamp
kernel argument is frozen at capture, so a graph is never a way to observe a
later model — see the replay rule below.

Runtime end-effector offsets (``end_effector_pose_runtime`` and its gradient)
are per-call inputs, not model state: they are admitted like any compute call
(shared) and bump no version.

Captured graphs (torch ``capture`` / ``GraphCallable``)
--------------------------------------------------------

A captured graph bakes device addresses and the captured model epoch. Every
``replay()`` (and ``__call__``) therefore takes a **replay admission** on its
context — shared, like a compute call, so it is ordered against a concurrent
mutator or close — and is refused with a ``RuntimeError`` if the model was
mutated since capture (recapture) or the context was closed (never a launch
into freed memory). A ``GraphCallable`` holds its handle, so the context
cannot be garbage-collected under a live graph; explicit ``close()`` of the
context still wins and later replays are refused. Replays of one graph are
serialized. ``static_out`` is an owned tensor whose value is overwritten by
the next replay: clone it to keep a value, and order a reader on another
stream (``wait_stream``) before the next replay overwrites it. Capturing a
backward is not supported (the backward's stamp check synchronises).

What stays for later increments
-------------------------------

B1/B2 keep today's synchronisation: setters still fence with a device-wide
synchronise and the numpy input pack still drains the device. The admission
lock orders *submissions*; it does not wait for outstanding GPU work, so the
setter fence is what guarantees a mutation never lands under a kernel still
reading the table — it stays until per-call leases and completion events
(B3) supply that ordering. Concurrent asynchronous calls on ONE context still
share its scratch buffers — one pipeline per context; use a context per
pipeline. Setter and numpy fences and the per-backward stamp check are
synchronisation points: do not read this page as fully asynchronous execution.

Inline-CUDA consumers of ``grid.cuh`` are untouched: ``init_gridData`` /
``init_gridData_checked`` / ``close_grid`` keep their signatures (the checked
initializer merely gained an optional trailing allocator-pool argument), and
the pool-less ``grid_device_alloc`` / ``grid_device_free`` overloads remain the
default-pool spellings.
