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
   if robot is None:
       raise ValueError("URDF parsing failed; inspect the file and parser warnings")

   robot.get_num_pos(), robot.get_num_vel(), robot.get_num_bodies()
   [j.get_name() for j in robot.get_joints_ordered_by_id()]   # the input-vector order
   robot.get_parent_id_array()                                 # tree structure by joint id
   robot.get_S_by_id(3)                                        # motion subspace of joint 3
   robot.get_Imat_by_id(3)                                     # spatial inertia of body 3
   robot.get_Xmat_Func_by_id(3)(0.7)                           # joint transform at q3 = 0.7

For this scalar-joint, non-mimic robot, joint order is coordinate order and
matches Pinocchio's by default (``joint_ordering="pinocchio_order"``).
For general models use ``get_joint_index_q(jid)`` and
``get_joint_index_v(jid)``: spherical joints occupy multiple coordinates,
planar/translation joints decompose, and mimic bodies share driver coordinates.
A floating base
(``floating_base=True``) prepends the free-flyer root: seven position
coordinates (position plus a unit quaternion, ``xyzw``) and six velocities.

Checking a model before generating code
---------------------------------------

* Parse with ``strict_inertial=True`` once. A real moving body without a valid
  ``<inertial>`` block raises (root/base and dummy links are exempt), before
  spending a compile on it.
* Confirm the joint types you expect with ``robot.get_joints_ordered_by_id()``;
  chained mimic relations resolve to their driving joint without removing
  the dependent bodies, and a spherical joint
  makes ``get_num_pos()`` and ``get_num_vel()`` differ.
* Joint limits are available as ``get_joint_limits_by_id(jid)`` and friends and
  are what the Python handles expose as ``joint_pos_limits`` and
  ``joint_effort_limits``.

The complete method list, the joint-type notes (including helical joints and
arbitrary axes) and the parse options are on the
:doc:`URDFParser API page <../../api_reference/urdf>`.

Parser coordinates are not the packed CUDA ABI. Before constructing floating
or spherical dynamics inputs, read :doc:`../concepts/input_output_abi` and
:doc:`python_wrappers`: ``q`` is ``nq`` wide and velocities, accelerations and
torques are ``nv`` wide on every wrapper surface.
