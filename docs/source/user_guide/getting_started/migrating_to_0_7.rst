Migrating from 0.6 to 0.7
=========================

Version 0.7.0 changes how a compiled robot is identified, how a consumer
artifact is built and reloaded, how collision margins are specified, and what
the plant's second derivative means on a manifold. Python users who call
``load_robot`` and the batched handle methods need only rebuild their caches.
Consumers who compile the generated header into their own kernels should walk
the steps below in order. Each step names the failure a user meets without it.

1. Pin and keep the old environment
-----------------------------------

Install 0.7.0 into a new environment or a new cache directory
(``cache_dir=`` or ``GRID_RBD_CACHE_DIR``) and keep the previous deployment
until the steps below pass. Do not purge a shared cache: entries are
content-keyed, so old and new builds coexist, and a purge removes artifacts
another process may still be using.

2. Rebuild prepared artifacts once
----------------------------------

Consumer artifact schema 2 stores the model descriptor beside the header and
binds the three together. An artifact built by 0.6 has no descriptor, so
``load_artifact`` refuses it rather than guessing. Rebuild with
``build_artifact`` once; do not copy a descriptor file next to an old binary,
because the association check compares hashes and will reject the pair.

.. code-block:: python

   from grid_rbd import describe_model, build_artifact, load_artifact

   model = describe_model("robot.urdf")
   art = build_artifact("myapp", "robot", "robot.urdf",
                        sources=["kernel.cu"],
                        algorithm_list=["end_effector_pose"],
                        expected_descriptor_id=model["descriptor_id"])
   # later, without a compiler or the URDF:
   art = load_artifact("myapp", "robot",
                       expected_descriptor_id=model["descriptor_id"])

``expected_descriptor_id`` is optional, but an application that inspected the
model earlier should pass it so a changed URDF cannot silently pair with a
stale binary. The descriptor identity covers coordinates, names, limits and
placements; it does not cover inertias, so two physically different robots can
share one identity. Build reuse is decided by the artifact content key, not by
the descriptor.

3. Declare Python extensions
----------------------------

A consumer that compiles a pybind11 module through ``build_artifact`` now
passes ``python_extension=True``. The build then records the interpreter ABI in
the artifact identity, exports only the module entry point and keeps the
module's static state private, so two revisions or two robots loaded in one
process do not share symbols. Without the flag a build whose object
exports a ``PyInit_*`` symbol is refused before publication, so a pybind module
cannot be recorded as a plain native library. There is no stable-ABI promise: a module built for CPython 3.12 is
reused by CPython 3.12 only.

Loading a second revision of a module under the same name in one process is
not supported by the interpreter; switch revisions in a fresh process.

4. Build lookup and prepared execution are different paths
-----------------------------------------------------------

At build and lookup time GRiD runs the compiler's dependency pass
(``nvcc -MM``) over every compilation unit, so an edit to any header the unit
actually includes, including a transitive or shadowing one, invalidates the
artifact. Manual ``headers=`` lists remain available as a supplement. This
means a warm *lookup* needs the compiler and the sources. A prepared
*application* that calls ``load_artifact`` needs neither; it checks the saved
descriptor, header hash and binary hash and loads. Containers that ship
without a toolchain must use the prepared path. System headers excluded by the
compiler's dependency output are an environment assumption: the toolkit and
host compiler identities are recorded, their contents are not.

5. Collision margins
--------------------

The checkers take a ``CollisionMargins<T>`` with ``environment_margin`` and
``self_margin`` where 0.6 took one scalar. A scalar still compiles and applies
the same value to both checks, so an unchanged call keeps its 0.6 behaviour.
An application that wants clearance against the environment but none between
its own links passes the two values explicitly:

.. code-block:: cuda

   auto margins = grid_collision::make_collision_margins<float>(0.02f, std::nullopt, 0.0f);

The host factory rejects non-finite or negative values. See
:doc:`../tutorials/collisions`.

6. Dimensions
-------------

The descriptor distinguishes ``nq`` (stored configuration entries), ``nv``
(tangent entries), ``nj`` (effective joints) and ``nb`` (effective bodies).
On a floating base or a spherical joint ``nq`` exceeds ``nv``; the difference
does not identify a free-flyer, and neither count is an actuator count. Stored
state has ``nq+nv`` entries and tangent state ``2*nv``. A solver's control
width, its linear feature rows (``MX``, ``MU`` on the cost menu) and its
actuation map are the solver's own definitions. See
:doc:`../../api_reference/model_coordinates`.

7. Defects, duals and the second derivative
-------------------------------------------

The native difference ``D(a, b)`` is reference-first, and its two Jacobians
reduce to ``-I`` and ``+I`` only on scalar coordinates. A shooting solver
that linearises a nonzero floating or spherical defect must compose the
step gradient with the difference Jacobians at the actual anchors and
transport its multipliers accordingly; a sign change is not a substitute.
``plant_step_hessian`` is the derivative of the moving step gradient, not a
fixed-chart Hessian and not the curvature of a defect. Scalar fixed-base
models are unaffected. The maps, the row normalisation and the covector
relation are specified in :doc:`../../api_reference/integration_conventions`.

8. Workspaces inside your kernel
--------------------------------

``grid_plant`` now exposes checked ``bind`` views for the plant gradient, the
plant Hessian and the cost menu, sized by the same layout the kernels use.
A view establishes capacity and alignment only; the enclosing kernel's static
plus dynamic shared memory must still be admitted against the device limit by
the caller. The Hessian view covers fixed-base ``nq == nv`` models with Euler
and semi-implicit Euler; other configurations keep using the tiered kernels.
See :doc:`../../api_reference/workspace_composition`.

9. Validate
-----------

Run your own regression on the rebuilt artifacts, compare a handful of
outputs against the 0.6 deployment, and exercise the prepared path in a
process without a compiler before switching production. The example
``bindings/examples/consumer_artifact_migration.py`` builds a native library
and a declared extension, reuses them, and loads them again from a fresh
process that cannot see the sources or the toolchain.
