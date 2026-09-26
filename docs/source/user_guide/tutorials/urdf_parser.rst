Inspecting a robot model with URDFParser
========================================

You rarely call the parser yourself: ``grid-generate`` and
``grid_rbd.register_robot`` parse the URDF for you. Reach for it directly when
you want to see what the generated code will assume, when you write or check a
reference algorithm in RBDReference, or when a URDF fails to parse.

Parse and look around
---------------------

.. code:: python

   from URDFParser import URDFParser

   robot = URDFParser().parse("config/robot_assets/iiwa14.urdf", floating_base=False)

   robot.get_num_pos(), robot.get_num_vel(), robot.get_num_bodies()
   [j.get_name() for j in robot.get_joints_ordered_by_id()]   # the input-vector order
   robot.get_parent_id_array()                                 # tree structure by joint id
   robot.get_S_by_id(3)                                        # motion subspace of joint 3
   robot.get_Imat_by_id(3)                                     # spatial inertia of body 3
   robot.get_Xmat_Func_by_id(3)(0.7)                           # joint transform at q3 = 0.7

The joint order returned here is the order of every ``q``, ``qd`` and ``tau``
vector the generated code takes, and it matches Pinocchio's by default
(``joint_ordering="pinocchio_order"``). A floating base
(``floating_base=True``) prepends the free-flyer root: seven position
coordinates (position plus a unit quaternion, ``xyzw``) and six velocities.

Checking a model before generating code
---------------------------------------

* Parse with ``strict_inertial=True`` once. A link without a valid
  ``<inertial>`` block raises instead of warning, which is what you want before
  spending a compile on it.
* Confirm the joint types you expect with ``robot.get_joints_ordered_by_id()``;
  mimic joints are flattened to their driving joint, and a spherical joint
  makes ``get_num_pos()`` and ``get_num_vel()`` differ.
* Joint limits are available as ``get_joint_limits_by_id(jid)`` and friends and
  are what the Python handles expose as ``joint_pos_limits`` and
  ``joint_effort_limits``.

The complete method list, the joint-type notes (including helical joints and
arbitrary axes) and the parse options are on the
:doc:`URDFParser API page <../../api_reference/urdf>`.
