Composing generated workspaces
==============================

Generated ``grid_plant`` views describe storage for device calls inside a caller's
kernel. They do not allocate memory, launch kernels, initialize numerical inputs
or choose a CUDA stream. Use ``float`` or ``double`` consistently with the model
and the caller. A view belongs to the generated header that defines it; neither
its layout nor its byte count is portable to another model or generation option.

Preparation and lifetime
------------------------------

Byte and alignment queries are host/device ``constexpr`` functions. The checked
``bind`` functions are noncollective and return false for null, short or misaligned
storage without changing the output view. A successful bind establishes capacity
and alignment only. The caller must supply a real backing allocation of that
capacity, with disjoint live regions and the appropriate address space.

Before launching, check the *enclosing* kernel's compiled static shared memory
plus requested dynamic shared memory against both the target budget and the
device opt-in limit. Set the kernel's opt-in attribute and respect its thread
limit. The binders do not perform this admission check: a full-shared view can be
well formed even when no kernel on the current device can launch it.

Numerical evaluators are block collective. Every participating thread must take
the same call path. Fence input staging, output consumption and scratch reuse
with block synchronization. Do not let concurrent blocks, streams or graph
replays share a mutable spill slot. Keep the model and all input/output/workspace
allocations alive until the asynchronous work completes. Read-only model sharing
does not make concurrent model mutation safe.

Cost menus
----------

``tracking_cost_menu_arena<T, MX, MU>`` is emitted when
``GRID_PLANT_HAS_COST_MENU`` is enabled. It owns two views, ``s_frame`` and
``s_scratch``, used by the menu value, gradient and Hessian calls. Its queries are
``alignment()``, ``scratch_offset()``, ``scratch_bytes()`` and ``bytes()``.

``MX`` and ``MU`` are the nonnegative linear-feature row counts used to instantiate
the corresponding menu calls. They are not the stored state width, tangent state
width, feature input width or control width. Zero rows are supported. Size the
view for the largest instantiated row counts when one allocation is reused.

``carve(base)`` is the unchecked hot-path equivalent. Use it only after preparation
has established the full allocation capacity and the caller has constructed a
16-byte-aligned address. It performs pointer arithmetic only. Retain checked
``bind`` at untrusted boundaries; unchecked carving is not a resource-check bypass.

For example, inside a block whose dynamic arena has already been admitted:

.. code-block:: cuda

   using Menu = grid_plant::tracking_cost_menu_arena<float, MX, MU>;
   extern __shared__ __align__(16) unsigned char storage[];
   Menu menu;
   if (!Menu::bind(storage, Menu::bytes(), menu)) return; // uniform result
   // Stage valid x, u and params, then synchronize the block.
   __syncthreads();
   grid_plant::tracking_cost_menu<float, MX, MU>(
       value, x, u, params, menu.s_frame, menu.s_scratch, model);
   __syncthreads();

Bounds, activation parameters, outputs, contact-force scatter and solver storage
remain caller owned. In particular, the menu's barrier bound arrays must remain
valid even when their weight is zero. The view does not replace a weighted
dynamics-Hessian contraction workspace. If an outer arena is only ``alignof(T)``
aligned, account for padding when aligning the menu's starting address to 16
bytes; rounding an inner offset alone is insufficient.

Plant gradients
---------------

``plant_step_gradient_arena<T>`` exposes ``bytes()``, ``alignment()`` and
``bind(base, capacity, out)`` in addition to the existing unchecked ``carve``.
The query describes its full-shared layout, not the tier-selected integrator
kernel layout. Generated element counts such as ``s_vaf_count`` are part of the
initialization contract; they count elements, not bytes.

For a raw composed ``plant_step_gradient`` call, stage ``s_x`` and ``s_u`` and
zero all ``s_vaf_count`` entries before the block enters the evaluator. Reinitialize
that band when reusing poisoned or unrelated storage. Pass all other fields from
the matching view, including topology and runtime-transform scratch. Do not derive
these extents from ``nv``: retained bodies, mimic topology and runtime transforms
can require different storage. The plant gradient may mutate its staged state.

Plant Hessians
--------------

``GRID_PLANT_HAS_HESSIAN_ARENA`` is one for the current fixed-base ``nq == nv > 0``
composition interface, and zero otherwise. A zero capability does not remove or
redefine the existing tiered floating-base Hessian kernels.

.. code-block:: cuda

   using Layout = grid_plant::plant_hessian_layout;
   using Hessian = grid_plant::plant_step_hessian_arena<double, Layout::deep_spill>;
   Hessian h;
   const bool ok = Hessian::bind(shared, shared_capacity,
                                 spill_slot, spill_capacity, h);
   // Handle failure uniformly before entering a collective operation.
   if (!ok) return;
   // Stage h.s_x and h.s_u cooperatively, then synchronize.
   __syncthreads();
   h.evaluate<grid::IntegratorType::EULER>(model, gravity, dt);
   __syncthreads();
   // Consume h.s_d2AB before reusing either region.

``full_shared`` and ``deep_spill`` are explicit storage layouts, not resource-tier
labels. Use ``shared_bytes()`` and ``workspace_bytes()`` for the selected layout;
both regions have 16-byte alignment. Full-shared uses no global workspace and
permits a null spill pointer. Deep-spill places the Hessian, second-order tensors
and contraction pool in the invocation's global slot while retaining the small
hot bands in shared memory. The view's ``evaluate`` selects the matching storage
flags; do not substitute flags inferred from a tier name.

Euler and semi-implicit Euler are supported by this interface. Output
``s_d2AB`` has the existing row-major ``[2*nv, 3*nv, 3*nv]`` layout, with perturbation
axis before gradient-column axis. Controls here are generalized torques of width
``nv``; coordinate metadata is not an actuator map. No multistage Hessian,
manifold-provider or custom-actuation contract is introduced by these views.

End-effector inner scratch
------------------------------

``grid::EE_POSE_INNER_SCRATCH_BYTES<T>()`` sizes the all-leaf pose inner.
The existing ``EE_GRAD_INNER_SMEM_BYTES<T>()`` sizes the generic all-leaf gradient
inner. When target aliases are generated, use
``EE_TARGET_POSE_INNER_SCRATCH_BYTES<T>()`` and
``EE_TARGET_GRAD_INNER_SCRATCH_BYTES<T>()`` for their selected target. These queries
describe only the inner's temporary pool, not its input, output, XI tables or a
complete enclosing kernel. A fixed named target can have a smaller pool than the
generic all-leaf call. Scratch from one entry point must not be used to size a
differently carved wrapper.

Pose scratch queries include internal 16-byte alignment headroom. The caller
still supplies only a ``T``-aligned pointer and exactly the queried capacity;
the inner aligns its two matrix banks within that allocation. Do not hard-code
``32 * number_of_targets`` elements or round the supplied pointer yourself
without accounting for the lost capacity. The same rule applies when the pose
scratch is placed in global workspace. Regenerate consumer headers and rebuild
their allocations together when updating GRiD.
