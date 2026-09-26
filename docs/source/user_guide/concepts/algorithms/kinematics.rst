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
   pose = h.end_effector_pose_runtime(q, ee_joint_names, ee_offsets)
   J    = h.end_effector_pose_gradient_runtime(q, ee_joint_names, ee_offsets)
   X    = h.fk_batched(q)                        # every body's transform, large batches

The pose is ``[x, y, z, roll, pitch, yaw]`` for each end effector, and the
derivatives are taken with respect to the tangent (velocity) coordinates, so
the Jacobian is ``6·NUM_EES × NV`` and the Hessian ``6·NUM_EES × NV × NV``.
For a floating base the six base columns are the spatial twist components,
the Pinocchio convention, and for a fixed base ``NV`` equals the joint count.
Note that these are derivatives of the pose *coordinates* (position and RPY
angles); they are not the same object as Pinocchio's spatial frame Jacobian,
which is available separately as :doc:`frame_jacobian`.

Implementation
--------------
The Python reference is RBDReference's end-effector pose family, validated
against Pinocchio's frame placement and its derivatives
(`RBDReference <https://github.com/A2R-Lab/RBDReference>`__). The CUDA
generators are ``grid_codegen/algorithms/_eepose_gradient_hessian.py`` (pose
gradient and Hessian) and ``_eepose_runtime.py`` (runtime targets); the pose
value is emitted with the kinematics helpers.

In GRiD
-------
The pose kernel is the smallest kernel GRiD generates (tens of microseconds
for a batch of a thousand states on the largest robot), so its cost through a
Python surface is dominated by dispatch. The release benchmarks show this
directly: on the floating-base robots the kernel is slower than MuJoCo Warp's
by 10–50 % and the JAX call is slower by 2–3×. If pose is all you need at high
rate, call it from the C++ host entry (``grid::end_effector_pose`` /
``_compute_only``) or batch it with the dynamics call that follows.

The CUDA host entries are ``grid::end_effector_pose``,
``grid::end_effector_pose_gradient`` and ``grid::end_effector_pose_hessian``,
each with a ``_compute_only`` variant. With ``output_convention="mujoco"`` the
input configuration is MuJoCo-convention; the pose itself is frame-invariant,
the Jacobian's base columns are reframed, and the Hessian is the symmetric
coordinate Hessian along the MuJoCo retract, all computed in the kernel.
Mimic robots fold the derivatives to the reduced coordinates.

The runtime-target variants let one compiled robot serve any leaf or
intermediate frame: the target joint names and per-target offsets are call
arguments rather than baked into the artifact. ``fk_batched`` returns every
body's transform and is intended for large batches (thread or warp variant).

See Also
--------
* :doc:`frame_jacobian` — the geometric (spatial) Jacobian of an arbitrary
  frame, its time derivative and the operational-space inertia.
* :doc:`integrators_and_plant` — the end-effector position cost built on the
  pose and its Jacobian.
* :doc:`../../tutorials/cuda_support_status` — per-robot coverage.
