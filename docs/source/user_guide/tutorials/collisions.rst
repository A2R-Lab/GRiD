Collision code generation
=========================

GRiD generates CUDA routines for self- and environment-collision checks from
URDF collision geometry. This is collision checking, not a contact-dynamics
simulator. The collision interface is generated CUDA C++, not a promise that
every collision operation is exposed through the NumPy, JAX, or PyTorch handles.

Generate collision routines
---------------------------

After the :doc:`installation <../getting_started/installation>`, run from the
repository root with a URDF whose collision assets resolve on your machine:

.. code-block:: shell

   grid-generate path/to/robot.urdf --collision --collision-res 0.10,0.05

This enables collision code generation and a coarse-to-fine covering-sphere
representation. The generated ``grid_collision::config_free`` device routine
checks a configuration against the configured self-collision pairs and obstacle
environment. It is called from your CUDA code, with device model/environment
data and the required scratch storage.

``--collision-native`` selects a representation using native sphere/capsule
rows where supported; remaining geometry uses covering spheres. It is a
different approximation and must be labeled separately in a comparison.

The repository example ``examples/codegen/generate_collision.py`` shows the
programmatic two-tier flow. See ``grid_codegen/cli.py`` for the current CLI and
``grid_codegen/collision/grid_collision_geometry.cuh`` for geometry types.
The integration notes in ``grid_codegen/collision/HANDOFF.md`` describe scratch
layouts and device calls; older module-name examples there predate the current
``grid-generate`` entry point.

Device API surfaces
-------------------

Two device entry points share the same baked sphere batch (``NUM_COLLISION_SPHERES`` spheres,
``sphere_anchor`` / ``sphere_offset`` / ``sphere_radius`` in ``namespace grid_collision``,
``_broad``-suffixed twins for the broad tier of a two-tier model):

* ``grid_collision::config_free<T>(s_q, d_robotModel, env, ...)`` — the block-cooperative check
  from joint positions: it runs the batched sphere extractor, then checks self pairs and the
  environment thread-per-range / thread-per-sphere and returns one block-uniform verdict. With a
  two-tier model the broad tier runs first and the fine tier only for the links it flags; the
  verdict equals a fine-only check.
* ``grid_collision::warp::config_free<T>(s_Xworld, env, w_scratch)`` and
  ``grid_collision::warp::collision_distance<T>(s_Xworld, env, s_dist, s_normal, w_scratch)`` — the
  same verdict / per-sphere clearance for a caller that already holds the joint world transforms
  (column-major 4x4 per movable joint, e.g. from ``ee_pose_inner_warp``), evaluated by ONE warp
  with lanes striding the spheres (``w_scratch`` = ``grid_collision::warp::W_SCRATCH_FLOATS``
  floats per warp). This is the entry point for multi-warp solvers that need a verdict inside one
  warp of a block (one IK candidate per warp); sphere positions are formed in ``float`` even for
  ``T = double``. The two surfaces are gated against each other on random configurations and
  environments (``test/cuda_equivalents/test_cuda_collision_warp.py``).

Geometry coverage and correctness
---------------------------------

* Sphere spacing controls geometry resolution and work. Coarse-to-fine checks
  are intended to match the fine-only verdict for the same representation.
* Covering geometry is an approximation of the original robot. Agreement with
  a fine-tier sphere check is not proof of exact mesh collision detection.
* Missing or unusable meshes fail generation. A failed voxel fill may use an
  explicitly warned bounding-box cover, but geometry is never dropped to
  produce a partial model. Self-collision pair exclusions must also match your application.
* Correctness tests live under ``test/cuda_equivalents/test_cuda_collision_*``;
  they cover geometry, pairs, tiering, costs, and native representations.

The :doc:`release measurements <../../release_measurements>` do not include
collision timings. When benchmarking collisions, report latency alongside
geometry coverage and fine-tier verdict agreement.

Sphere models and fidelity reports
------------------------------------------------------------

Generate an interchangeable sphere URDF without compiling CUDA::

   python -m grid_codegen.spherize robot.urdf --out spheres.urdf --mesh-mode bounded-bulge --report
   grid-generate robot.urdf --spherized-urdf spheres.urdf

``grid-spherize`` is the installed entry point for the same module. The
bounded-bulge mode uses one fixed mesh policy: 2.5 mm voxels and a 10 mm bulge
budget relative to the voxel approximation. Coarse candidate centres fall
back to the full lattice for thin geometry. Primitive geometry retains its
analytic sphere cover. The input is the URDF's **collision** geometry, not
its visual meshes. Unresolved assets and incomplete fits are errors.

For ``package://`` meshes, the importer checks paths relative to the URDF and
explicit ``ROS_PACKAGE_PATH`` roots (either a package directory or its parent).
It never downloads assets. Drake's collision meshes for iiwa14 links 6 and 7
are bundled under ``config/robot_assets/drake`` with their upstream licenses,
revision and hashes; the bundled URDF resolves them without environment setup.
The historical iiwa14 sphere asset omits those meshes and is not a complete
replacement for the source collision model.

The complete-model CUDA gate checks the supplied-model CLI, every transformed
sphere against CPU forward kinematics, and collision verdicts at 1, 32 and
100 threads (``test_cuda_collision_complete.py``).

To evaluate an existing model against the original collision geometry::

   python -m grid_codegen.spherize robot.urdf --report spheres.urdf

The per-link report gives sphere count, voxel-estimated maximum/p99 bulge,
and the uncovered fraction of 2,000 seeded mesh-surface samples. Bulge uses
400 samples per sphere. These are reproducible diagnostics, **not a proof
of continuous mesh coverage or a certified bulge bound**. Primitive meshes
used for this report are tessellated. Inspect the report alongside the model.

``--spherized-urdf`` rejects missing collision links, non-sphere geometry,
invalid spheres, and mismatched joint frames. It cannot prove that an
arbitrary supplied sphere model covers its source mesh. The sole named
lookup is ``foam``; it requires an optional local
``external/foam/assets/panda/smaller_panda_spherized.urdf``. GRiD does not
download that asset or substitute another model when absent. Pass an explicit
path to a model from your own checkout instead. No cuRobo or fairness-study
assets are bundled by this mechanism.
