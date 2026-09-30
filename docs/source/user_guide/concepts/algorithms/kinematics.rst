Kinematics (end-effector pose, Jacobian, Hessian)
=================================================

.. currentmodule:: RBDReference.RBDReference

Overview
--------
The kinematics family maps a configuration to the pose of one or more
end-effector frames and to the first and second derivatives of that pose. The
end effectors are chosen at generation time (the ``-t`` option of
``grid-generate`` or the ``ee_joint_names`` argument of ``register_robot``); a
runtime-target variant takes the target joint and an offset as call
arguments instead.

Signature
---------
.. code-block:: python

   pose = h.end_effector_pose(q)                 # (B, 6*NUM_EES): [xyz, rpy] per end effector
   J    = h.end_effector_pose_gradient(q)        # (B, 6*NUM_EES, NV)
   H    = h.end_effector_pose_hessian(q)         # (B, 6*NUM_EES, NV, NV)
   pose = h.end_effector_pose_runtime(q, ee_joint_names, ee_offsets)  # (B, NEE, 6)
   J    = h.end_effector_pose_gradient_runtime(q, ee_joint_names, ee_offsets)  # (B, NEE, 6, NV)
   pose7 = h.fk_batched(q, use_warp=False)       # NumPy only: (B, 7), first leaf pose

The pose is ``[x, y, z, roll, pitch, yaw]`` for each end effector, and the
derivatives are taken with respect to the tangent (velocity) coordinates, so
the Jacobian is ``6·NUM_EES × NV`` and the Hessian ``6·NUM_EES × NV × NV``.
For a floating base the six base columns are the spatial twist components,
in Pinocchio order (local linear velocity, then local angular velocity).
For a fixed base with independent scalar joints, ``NV`` equals the joint count;
use ``h.num_vel`` for spherical or mimic models. Inputs are ``(B, h.nq)``.
Note that these are derivatives of the pose *coordinates* (position and RPY
angles); they are not the same object as Pinocchio's spatial frame Jacobian,
which is available separately as :doc:`frame_jacobian`.
RPY coordinates have chart singularities; their derivatives should not be
treated as a globally nonsingular orientation representation.

Implementation
--------------
The Python references are ``RBDReference.end_effector_pose``,
``end_effector_pose_gradient`` and ``end_effector_pose_hessian_analytic``
(`RBDReference <https://github.com/A2R-Lab/RBDReference>`__). The CUDA
generators are ``grid_codegen/algorithms/_eepose_gradient_hessian.py`` (pose
value, gradient, Hessian and batched FK) and ``_eepose_runtime.py`` (runtime
targets). The pose-coordinate derivatives are not direct substitutes for
Pinocchio's spatial frame derivatives.

In GRiD
-------
Dispatch can be a substantial part of short pose evaluations. The release
collection includes floating-base pose losses against MuJoCo Warp; consult
:doc:`../../../release_measurements` for the measured API boundary and batch
size rather than inferring pure device-kernel speed from resident API timings.

The CUDA host entries are ``grid::end_effector_pose``,
``grid::end_effector_pose_gradient`` and ``grid::end_effector_pose_hessian``,
each with a ``_compute_only`` variant. With ``output_convention="mujoco"`` the
input configuration is MuJoCo-convention; the pose itself is frame-invariant,
the Jacobian's base columns are reframed, and the Hessian is the symmetric
coordinate Hessian along the MuJoCo retract, all computed in the kernel.
Mimic robots fold the derivatives to the reduced coordinates.

The runtime-target variants let one compiled robot serve any leaf or
intermediate frame: the target joint names and per-target offsets are call
arguments rather than baked into the artifact. Each offset may be a local
point ``[x, y, z]``, a homogeneous point ``[x, y, z, 1]``, or a full 4×4
SE(3) tool transform expressed in the target joint frame. The transform's
rotation affects the returned tool-frame orientation; a single offset is
broadcast to all targets. Supply points as a list of offsets, for example
``ee_offsets=[[0, 0, 0.1]]``. ``None`` selects the frame origin; the target names default to
all leaf joints. Runtime results retain a separate target axis, unlike the
flattened baked-pose outputs.
``fk_batched`` instead returns the first leaf's ``[tx, ty, tz, qw, qx, qy, qz]``
pose, not a transform for every body. It uses Pinocchio-convention inputs,
is NumPy-only, and is generated only for non-spherical models with at most
32 joints. Use ``end_effector_pose`` when that helper is unavailable.

Building and selecting targets
------------------------------

Load a model with the operations you need before calling the examples above::

   import grid_rbd

   h = grid_rbd.load_robot(
       "config/robot_assets/iiwa14.urdf",
       algorithm_list=["end_effector_pose", "end_effector_pose_gradient",
                       "end_effector_pose_hessian"],
   )

Without named targets, the baked pose family uses the robot's leaf joints.
Request the runtime-target operations explicitly in ``algorithm_list`` if
you need them; a runtime target does not add an operation to an existing
artifact. Close the handle when finished. See :doc:`../../tutorials/python_wrappers`
for registration, available-operation inspection and framework backends.

See Also
--------
* :doc:`frame_jacobian` — the geometric (spatial) Jacobian of an arbitrary
  frame, its time derivative and the operational-space inertia.
* :doc:`integrators_and_plant` — the end-effector position cost built on the
  pose and its Jacobian.
* :doc:`../../tutorials/cuda_support_status` — per-robot coverage.
