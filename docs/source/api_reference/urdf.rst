URDFParser
==========

A small parser library for URDF files. It returns a ``robot`` object that
exposes links, joints, motion subspaces, spatial inertias and the joint-frame
transformation matrices that the dynamics algorithms and the code generator
consume.

Usage
-----

.. code:: python

   from URDFParser import URDFParser

   parser = URDFParser()
   robot = parser.parse(urdf_filepath, floating_base=False, joint_ordering="pinocchio_order")

The complete signature is::

   parse(filename, floating_base=False, using_quaternion=True,
         joint_ordering="pinocchio_order", strict_inertial=False,
         reduce_couplings=True, planar_base=False)

Keep ``using_quaternion=True`` for the documented floating-base GRiD paths.

``joint_ordering`` controls how sibling joints under the same parent link are
ordered in the depth-first walk that assigns joint ids:

.. code:: python

   joint_ordering="pinocchio_order"    # DFS with Pinocchio-style sibling sorting (default)
   joint_ordering="urdf_order"         # DFS preserving the raw URDF sibling order
   joint_ordering="alphabetical_order" # DFS sorting sibling joints by joint name

A floating base adds a free-flyer root joint. Its layout is pinocchio's:
``q = [x, y, z, qx, qy, qz, qw, joints]`` and ``v = [vx, vy, vz, wx, wy, wz,
joints]`` on every surface (the parser, RBDReference and the generated code).

Supported joint types
---------------------

Revolute, continuous, prismatic, fixed, helical (screw), planar and spherical
joints are supported, as are mimic joints (chained mimics are flattened when
the model is resolved; a mimic cycle raises ``MimicResolutionError``). An
arbitrary ``<axis>`` direction is parsed into a dense 6-vector motion subspace
``S``; robots with only cardinal axes keep the compact form.
Planar joints expand into two prismatic joints and one continuous joint;
``translation`` (alias ``cartesian``) expands into three prismatic joints.
Fixed joints are merged into the moving-body model. Mimic bodies remain in
the tree, but their coordinates are tied to the independent driver.

A mimic may also be a polynomial follower, GRiD's reading of the MuJoCo
joint equality: ``<mimic joint="l" polycoef="c0 c1 c2 c3 c4" q0="..."/>``
sets ``q = c0 + c1 (q_l - q0) + c2 (q_l - q0)^2 + c3 (q_l - q0)^3 + c4
(q_l - q0)^4`` (at most four coefficients beyond the constant; trailing
zeros may be omitted; ``multiplier`` and ``offset`` are excluded, since
they are the degree-one polynomial, which is stored as a plain mimic). Its
fold multiplier ``dq/dq_l`` varies with the leader, so the joint's constant
accessors raise and consumers read ``coupling_alpha(q_l)`` with its first
and second derivatives; chains through a polynomial follower are not
supported. ``parse(..., reduce_couplings=False)`` returns the full spanning
tree, every follower independent, with the couplings on ``robot.couplings``;
the numpy reference projects that tree onto the independent coordinates for
such robots. The generated code evaluates the multiplier on the device; the
algorithms whose folds read it at run time are listed in the changelog, the
others are refused for a polynomial robot.

Helical (screw) joints are a one-degree-of-freedom joint whose single
coordinate drives coupled rotation about and translation along the same axis,
``S = [axis; pitch·axis]``. URDF has no native helical type, so the pitch is
carried as a ``pitch`` attribute on ``<axis>``, in metres per radian, matching
Pinocchio's ``JointModelHelical``:

.. code:: xml

   <joint type="helical">
     <axis xyz="0 0 1" pitch="0.05"/>
   </joint>

A spherical joint is a second source of ``NQ != NV`` (quaternion position,
3-wide tangent) in addition to the floating base; ``joint_is_spherical(jid)``
and ``robot_has_spherical()`` report it. Closed kinematic loops are not
supported.

Planar joints and the planar base
---------------------------------

A URDF ``planar`` joint named ``J`` is decomposed when the file is parsed. It
becomes three one-coordinate joints, ``J__sub0`` (prismatic), ``J__sub1``
(prismatic) and ``J__sub2`` (continuous), joined by two massless links named
``J__dummy0`` and ``J__dummy1``. The decomposition is exact, and every
algorithm sees three ordinary scalar joints.

* The coordinates are ``[px, py, theta]`` in that order, with ``nq = nv = 3``.
  ``px`` and ``py`` are translations along the first and second in-plane axes
  of the parent frame and ``theta`` is the rotation about the plane normal.
* The velocity is ``[d px/dt, d py/dt, d theta/dt]``. The linear part is
  expressed in the parent frame, so it does not rotate with ``theta``.
* ``<axis>`` is the plane normal and defaults to ``0 0 1``. For a normal along
  a coordinate axis the in-plane axes are the other two coordinate axes in
  the order x, y, z, and the rotation is about the positive coordinate axis
  whatever the sign written in the file. For any other normal ``n`` the
  in-plane axes are ``R x`` and ``R y``, where ``R`` is the smallest rotation
  that takes ``z`` to ``n``, and the rotation is about ``n``.
* The joint ``<origin>`` is placed on ``J__sub0``. The other two sub-joints
  have an identity origin.
* ``<limit>`` on a planar joint is not read. All three coordinates are
  unbounded. ``<dynamics>`` damping and friction apply to each sub-joint.
* A planar joint cannot carry ``<mimic>``.

Pinocchio's ``JointModelPlanar`` describes the same mechanism with different
coordinates: ``q_pin = [x, y, cos(theta), sin(theta)]`` and a velocity
``[vx, vy, omega]`` expressed in the moving joint frame. With ``c = cos(theta)``
and ``s = sin(theta)`` the velocities are related by
``v_pin = [c vx + s vy, -s vx + c vy, d theta/dt]``, written ``v_pin = G v``
with ``G = blockdiag(Rz(theta)^T, 1)``. The accelerations then satisfy
``a_pin = G a + (dG/dt) v``, the generalized forces ``tau = G^T tau_pin`` and
the mass matrices ``M = G^T M_pin G``. Pinocchio's URDF reader ignores the
``<axis>`` of a planar joint and always uses the XY plane.

``parse(..., planar_base=True)`` attaches the root link of the URDF to a
``world`` link through a planar joint named ``planar_base_joint`` with normal
``0 0 1`` and identity origin. The result is the same model as a URDF that
declares that joint: the first three joints are ``planar_base_joint__sub0``,
``planar_base_joint__sub1`` and ``planar_base_joint__sub2`` and the first three
coordinates are the base pose ``[px, py, theta]`` in the world frame. A planar
base is not a floating base. ``robot.floating_base`` stays false, there is no
quaternion, and no MuJoCo-convention kernels are generated; ``robot.planar_base``
records that the option was used. ``planar_base`` and ``floating_base`` cannot
both be set. When the URDF root is already a link named ``world``, it must have
exactly one child joint and that joint must be fixed; the fixed joint is
replaced and its origin is kept on the first sub-joint. Any other explicit
``world`` root raises ``URDFParseError``, as does a URDF that already uses the
``planar_base_joint`` names.

The option is available as ``register_robot(..., planar_base=True)`` on the
numpy, jax and torch backends, as ``describe_model(..., planar_base=True)``,
as ``build_artifact(..., planar_base=True)`` and as ``grid-generate -p``
(``--planar-base``). It enters the build cache key only when set, so a planar
base build and the fixed-base build of the same URDF are separate cache
entries. The handle property ``planar_base`` and the generated constant
``grid::model::planar_base`` report a planar base found either way; see
:doc:`model_coordinates`.

Parse options and errors
------------------------

* ``strict_inertial=True`` rejects a missing or degenerate ``<inertial>``
  on a real moving body with a ``URDFParseError``. Root/base and dummy links
  are exempt; the default keeps parsing and warns.
* The typed exceptions live in ``errors.py``: ``URDFParseError``,
  ``UnsupportedJointTypeError`` and ``MimicResolutionError``.
  Missing files, malformed model fields, and invalid options also raise
  ``URDFParseError``; wrapped failures preserve their cause in ``__cause__``.
* Joint limits from the URDF ``<limit>`` tags are available through
  ``get_joint_limits_by_id``, ``get_velocity_limit_by_id`` and
  ``get_effort_limit_by_id``.
* ``get_origin_params_ordered_by_id()`` returns the per-joint
  ``[x, y, z, r, p, y]`` origin table used by the runtime-transform feature.

Installation
------------

The parser is installed with GRiD's editable install. Standalone, it needs
``beautifulsoup4``, ``lxml``, ``numpy`` and ``sympy``:

.. code:: shell

   pip install -r requirements.txt

Robot API
---------

The accessor families below take ``XXX`` as one of:

* ``joint``: a joint object (see the joint API below)
* ``link``: a link object (see the link API below)
* ``Xmat``: a sympy spatial transform with coordinates defined by its
  joint (a 4x4 homogeneous version and its first and second derivatives also
  exist, for example ``d2Xmat_hom``)
* ``Xmat_Func``: a function returning a numpy matrix for a value of the free
  variable (again with homogeneous and derivative variants)
* ``Imat``: a numpy 6x6 spatial inertia matrix
* ``S``: a numpy motion subspace, shape ``(6,)`` for a scalar joint and
  multiple columns for a multi-DoF joint

.. code:: python

   # A single object by its ID or by its name as defined in the URDF
   get_XXX_by_id(lid) # jid for joints 
   get_XXX_by_name(name)
   # A list of the objects that occur in the given bfs level
   get_XXX_by_bfs_level(name)
   # A list of the object ordered by their IDs or by their names as defined in the URDF
   # Note: The base link/inertia exists at index -1 and so will appear at the beginning of the list
   get_XXXs_ordered_by_id(reverse = False)
   get_XXXs_ordered_by_name(reverse = False)
   # A dictionary of objects by their ID or by their name as defined in the URDF
   # Note: The base link/inertia exists at index -1
   get_XXXs_dict_by_id()
   get_XXXs_dict_by_name()

The API also includes the following functions:

.. code:: python

   # get the robot name
   get_name()
   # get the robot type (if applicable)
   is_serial_chain()
   # get the number of positions and velocities in the robot state as well as numbers of links and joints
   # Do not infer coordinate widths from the number of joints or bodies.
   # Each independent quaternion joint adds one position coordinate over NV.
   # Mimic joints add bodies, not independent coordinates.
   get_num_pos()
   get_num_vel()
   get_num_bodies() # effective moving bodies; excludes the fixed world base
   get_num_joints()
   get_num_links()
   get_num_links_effective() # num_links - 1 (base link is not used in many RBD algorithms when fixed)
   get_joint_index_q(jid) # scalar index or list of position indices
   get_joint_index_v(jid) # scalar index or list of tangent indices
   # get the max bfs_level
   get_max_bfs_level()
   # get the IDs at a given bfs level and the bfs level for a given id
   get_ids_by_bfs_level(level)
   get_bfs_level_by_id(jid)
   # get the ID of the parent(s) of a given link(s) by id
   get_parent_id(lid)
   get_parent_ids(lids)
   get_unique_parent_ids(lids) # remove duplicates
   # get the full list of parents ordered by id
   get_parent_id_array()
   # test if there is a repeated parent by ids
   has_repeated_parents(jids)
   # get the subtree IDs for a given id and total count and test if in a subtree
   get_subtree_by_id(jid)
   get_total_subtree_count()
   get_is_in_subtree_of(jid,jid_of)
   # get the ancestor IDs for a given id and total count and test if an ancestor
   get_ancestors_by_id(jid)
   get_total_ancestor_count()
   get_is_ancestor_of(jid,jid_of)
   # get all joints that have parent link name as the parent or child link name as the child
   get_joints_by_parent_name(parent_name)
   get_joints_by_child_name(child_name)
   # get the joint that has parent link name as the parent and child link name as the child
   get_joint_by_parent_child_name(parent_name,child_name)
   # see if the following joints have the same S (useful for codegen)
   are_Ss_identical(jids)

Joint API
---------

.. code:: python

   # get the name, id, and bfs of the joint
   get_name()
   get_id()
   get_bfs_id()
   get_bfs_level()
   # get the parent and child link name
   get_parent()
   get_child()
   # get the Xmat or Xmat_Func for this joint as defined above
   get_transformation_matrix()
   get_transformation_matrix_function()
   # get the S for this joint as defined above
   get_joint_subspace()
   # get the velocity damping coefficent for this joint
   get_damping()

Link API
--------

.. code:: python

   # get the name, id, and bfs of the link
   get_name()
   get_id()
   get_bfs_id()
   get_bfs_level()
   # get the link's spatial inertia matrix
   get_spatial_inertia()
