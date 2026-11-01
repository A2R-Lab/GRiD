Model coordinates and named frames
==================================

A model descriptor records the coordinates and frames used by a generated
model. Applications can use it to interpret states and attach tools without
parsing generated functions or keeping a second joint table. It does not
define actuation, select a solver, or certify every operation for every model.

Inspecting a model
------------------

.. code-block:: python

   from grid_rbd import describe_model

   model = describe_model("robot.urdf", floating_base=True)
   print(model["nq"], model["nv"], model["descriptor_id"])

This call uses the installed parser but needs neither a GPU nor a compiler.
Unknown options and schema versions are errors. Schema 1 uses quaternion
coordinates. The raw generator's legacy Euler-coordinate path is unchanged
but does not emit schema-1 metadata.

The generated header exposes the same data through ``grid::model``. Library
generation saves ``model_descriptor.json`` beside that header. A Python
handle's ``model_descriptor`` property returns an independent copy of its
baked description. This is always GRiD-native metadata, including when the
handle's output adapter is set to MuJoCo conventions.

Coordinate ownership
--------------------

``nq`` counts configuration entries, ``nv`` counts velocity/tangent entries,
``nj`` counts effective joints, and ``nb`` counts effective bodies. The root
reference has body ID -1 and is excluded from ``nb``. Effective joint and
body IDs follow the parsed tree, not XML tag order. Fixed joints are merged;
planar and non-root floating joints may become scalar joints. The descriptor
reports those effective joints rather than inventing a native representation.

Each joint records its parent and child body. Use those mappings rather than
coordinate offsets to index bodies. A floating base is an effective body.
A decomposition body remains a body even when it has no mass.

Independent joints own disjoint ``q_start`` and ``v_start`` blocks. Their
``local_nq`` and ``local_nv`` widths partition the input vectors exactly.
``configuration_layout`` is a derived view of these blocks, sorted by q start.
The historical Python ``num_joints`` property means nq; ``joint_names`` is
indexed by effective joint ID, not by q or v slot.

.. list-table:: Local parameterizations
   :header-rows: 1

   * - Kind
     - Configuration
     - Velocity
   * - Revolute / continuous
     - Angle in radians
     - Radians per second
   * - Prismatic
     - Displacement in metres
     - Metres per second
   * - Helical / screw
     - Angle; pitch in metres per radian
     - Radians per second
   * - Spherical
     - Unit quaternion [x,y,z,w]
     - Three local angular components
   * - Floating root
     - [x,y,z,qx,qy,qz,qw]
     - Local [vx,vy,vz,wx,wy,wz]

Internal spatial vectors often use angular-before-linear ordering. They are
not the public floating velocity vector. Quaternion velocity is a tangent
vector, not four quaternion rates. Invalid quaternions are not normalized
by metadata inspection.

Mimic joints retain their bodies but own no configuration block: both starts
are null. Their flattened relation is
``theta = multiplier * q[leader.q_start] + offset``. The local velocity is
``multiplier * v[leader.v_start]``. A null leader denotes a constant joint:
its multiplier is zero, and it reads no input slot. Local widths describe
the joint motion, not additional independent coordinates. No multi-DoF mimic
semantics are introduced by the descriptor.

Limits remain attached to each effective joint. Null means no finite bound,
not zero; continuous joints have no position interval. No quaternion box or
sampling policy is fabricated. Ownership and effort bounds do not define a
control dimension, transmission or underactuation policy.

Transforms and frames
---------------------

``T_AB`` maps coordinates in B into A: ``p_A = T_AB p_B``. JSON stores a
4-by-4 nested row array; generated C++ records use column-major arrays with
index ``r + 4*c``. For each joint,
``T_parent_child(q) = T_parent_joint_zero * M_joint(q)``.
The zero placement includes merged fixed ancestors, evaluated before any
mimic offset. It is not necessarily the original URDF origin matrix.

The signed unit axis is expressed in the local zero-motion joint frame.
Spherical and floating joints have no single axis. Floating motion is
``[R(quaternion), xyz; 0, 1]``. World composition is parent world pose times
local pose. A fixed named frame F on body B has
``T_world_frame = T_world_body * T_body_frame``.

Frame IDs have their own namespace: root first, effective joints in ID order,
then retained fixed joints in parser order. IDs are stable only within one
descriptor identity. Joint frames have identity offsets; fixed frames retain
their accumulated placement. This is not an alias table for every original
URDF link discarded during reduction. Names are opaque UTF-8 strings.
Use frame kind as well as name when identifying the typed root entry.

Generated access
----------------

``grid::model::joint(id, joint_record&)`` and
``grid::model::frame(id, frame_record&)`` are constexpr host/device accessors.
They return false for invalid IDs without changing the destination. Numeric
device access does not index host-only arrays. ``joint_name`` and ``frame_name``
are host constexpr accessors; invalid IDs return null. ``descriptor_json`` and
``descriptor_id`` are host-readable constants.

``compose_frame_from_body(frame_id, body_id, world_body, world_frame)`` accepts
one already-computed body-world pose and writes one frame-world pose. The two
16-element column-major buffers must be disjoint. This is a thread-scope
operation: no allocation, hidden synchronization, second FK, or world-pose
array is needed. Caller-managed fences still govern data shared across threads.
The function checks the attachment ID and leaves output unchanged on failure.

With ``runtime_transform=True``, placements describe the nominal baked model,
not current mutable origin parameters. The static composer refuses such models.
Existing runtime-transform operations remain available through their own APIs.

Identity and prepared deployment
--------------------------------

The descriptor ID covers coordinate geometry, names, limits and ownership.
It deliberately excludes inertias, runtime values, algorithms, precision,
compiler options and artifact bytes. Different physical dynamics and binaries
can share a descriptor ID. Use the artifact content identity for build reuse.

Schema 1 hashes a typed canonical encoding of all fields except
``descriptor_id``. Null, bool, integer, float, string, list and map have separate
tags. Integer payloads are decimal strings. Floats are finite IEEE binary64
big-endian lowercase hex, with signed zero normalized to positive zero.
Strings retain their Unicode spelling and are UTF-8 encoded; unpaired
surrogates are rejected. Maps sort keys by Unicode code point. The tagged
tree uses compact JSON without ASCII escaping, then SHA256. The implementation
and golden vectors live in ``grid_codegen/model_descriptor.py`` and
``test/test_model_descriptor.py``. Pretty-printing descriptor JSON does not
change its identity. Readers also validate structural invariants; rehashing an
invalid ownership table does not make it valid.

The exact tag forms are ``["null"]``, ``["bool", value]``,
``["int", decimal_string]``, ``["f64", hex_string]``, ``["str", value]``,
``["list", encoded_elements]`` and
``["map", [[key, encoded_value], ...]]``. A map's keys remain plain strings;
its values are recursively tagged. The top-level digest field alone is omitted.

``build_artifact(..., expected_descriptor_id=model["descriptor_id"])`` checks
a previously inspected model on cold and reused builds. The returned
``ConsumerArtifact.model_descriptor`` is validated metadata. Prepared
``load_artifact(..., expected_descriptor_id=...)`` checks the saved descriptor,
its generated-header representation, and the header/binary hashes without
reading the original source or invoking a compiler. These are consistency
checks, not authentication against an attacker replacing all files and hashes.

Consumer artifact schema 2 requires this association. Rebuild older prepared
artifacts once; do not copy a new descriptor onto an old binary. Production
consumer pins are not changed by this migration mechanism.

Support boundary
----------------

A model with zero independent coordinates has valid empty ownership metadata.
That does not promise every wrapper, empty-batch operation, integrator or timing
tool supports it. Existing operation-specific validation remains authoritative.
This interface supplies geometry and coordinate meaning, not general dynamics
providers, contact differentiation, incremental FK caches or solver policies.
