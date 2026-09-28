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
         alpha_tie_breaker=None, joint_ordering="pinocchio_order",
         floating_base_convention="pinocchio", strict_inertial=False)

Keep ``using_quaternion=True`` for the documented floating-base GRiD paths.

``joint_ordering`` controls how sibling joints under the same parent link are
ordered in the depth-first walk that assigns joint ids:

.. code:: python

   joint_ordering="pinocchio_order"    # DFS with Pinocchio-style sibling sorting (default)
   joint_ordering="urdf_order"         # DFS preserving the raw URDF sibling order
   joint_ordering="alphabetical_order" # DFS sorting sibling joints by joint name

The older ``alpha_tie_breaker`` argument is still accepted: ``False`` means
``urdf_order`` and ``True`` means ``alphabetical_order``.
When supplied, it takes precedence over ``joint_ordering``.

A floating base adds a free-flyer root joint. The public input convention is
selected with ``floating_base_convention``:

.. code:: python

   robot = parser.parse(urdf_filepath, floating_base=True, floating_base_convention="pinocchio")

   floating_base_convention="pinocchio"  # default: q = [x, y, z, qx, qy, qz, qw], v = [vx, vy, vz, wx, wy, wz]
   floating_base_convention="legacy"     # q = [x, y, z, qw, qx, qy, qz], v = [wx, wy, wz, vx, vy, vz]

The robot's normalization helpers convert legacy vectors to the internal
Pinocchio ordering; RBDReference uses these helpers. This parser option is
not a legacy-layout switch for the Python/CUDA bindings: use their documented
Pinocchio or MuJoCo interfaces and buffer layouts.

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

The accessor families below take **XXX** as one of:

* **joint**: a joint object (see the joint API below)
* **link**: a link object (see the link API below)
* **Xmat**: a sympy spatial transform with coordinates defined by its
  joint (a 4x4 homogeneous version and its first and second derivatives also
  exist, for example ``d2Xmat_hom``)
* **Xmat_Func**: a function returning a numpy matrix for a value of the free
  variable (again with homogeneous and derivative variants)
* **Imat**: a numpy 6x6 spatial inertia matrix
* **S**: a numpy motion subspace, shape ``(6,)`` for a scalar joint and
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
