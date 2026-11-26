Coordinate conventions across libraries
=======================================

GRiD, Pinocchio and MuJoCo describe the same robot state with different
vectors. The physical state is the same. What differs is how each joint type
stores its configuration, and in which frame the velocity of a moving base is
expressed. This page states the three conventions side by side, gives the
maps between them, and records why each library made its choice.

GRiD's own layout is specified in :doc:`../../api_reference/model_coordinates`.
The MuJoCo output convention that GRiD can serve natively is described in
:doc:`mjx_convention`.

Configuration and velocity per joint type
-----------------------------------------

``nq`` is the number of configuration entries of the joint and ``nv`` the
number of velocity entries.

.. list-table::
   :header-rows: 1
   :widths: 16 28 28 28

   * - Joint
     - GRiD
     - Pinocchio
     - MuJoCo
   * - Revolute with limits
     - Angle. ``nq = nv = 1``.
     - Angle. ``nq = nv = 1``.
     - Hinge: angle. ``nq = nv = 1``.
   * - Continuous (unbounded revolute)
     - Angle. ``nq = nv = 1``.
     - ``(cos, sin)``. ``nq = 2``, ``nv = 1``.
     - Hinge: angle. ``nq = nv = 1``.
   * - Prismatic
     - Displacement. ``nq = nv = 1``.
     - Displacement. ``nq = nv = 1``.
     - Slide: displacement. ``nq = nv = 1``.
   * - Spherical
     - Unit quaternion ``[x, y, z, w]``, local angular velocity.
       ``nq = 4``, ``nv = 3``.
     - Unit quaternion ``[x, y, z, w]``, local angular velocity.
       ``nq = 4``, ``nv = 3``.
     - Ball: unit quaternion ``[w, x, y, z]``, local angular velocity.
       ``nq = 4``, ``nv = 3``.
   * - Planar joint or planar base
     - ``[px, py, theta]``, velocity ``[vx, vy, omega]`` with the
       translation rates in the parent frame. ``nq = nv = 3``.
     - ``[x, y, cos, sin]``, velocity ``[vx, vy, omega]`` with the
       translation rates in the body frame. ``nq = 4``, ``nv = 3``.
     - No planar joint. Two slides and a hinge give ``[x, y, theta]`` with
       parent-frame rates. ``nq = nv = 3``.
   * - Floating base
     - ``[x, y, z, qx, qy, qz, qw]``, velocity ``[v; omega]`` with both
       parts in the body frame. ``nq = 7``, ``nv = 6``.
     - ``[x, y, z, qx, qy, qz, qw]``, velocity ``[v; omega]`` with both
       parts in the body frame. ``nq = 7``, ``nv = 6``.
     - Free joint: ``[x, y, z, qw, qx, qy, qz]``, velocity ``[v; omega]``
       with ``v`` in the world frame and ``omega`` in the body frame.
       ``nq = 7``, ``nv = 6``.

GRiD follows Pinocchio for the quaternion joints and differs from it for the
two joints that Pinocchio stores as ``(cos, sin)``. GRiD and MuJoCo agree on
every scalar joint and differ on the floating base.

The maps
--------

Each difference is a map on the configuration and a linear map on the
velocity. Forces transform by the inverse transpose of the velocity map, and
each derivative index contracts with the matching factor.

Continuous joint, GRiD to Pinocchio
   The configuration map is ``theta`` to ``(cos theta, sin theta)``, with
   Jacobian ``[-sin theta, cos theta]``. The velocity is unchanged. The
   inverse is ``atan2(sin, cos)``, which returns the angle in
   ``(-pi, pi]``: the number of turns is not recoverable from Pinocchio's
   storage.

Planar joint, GRiD to Pinocchio
   The configuration map is ``[px, py, theta]`` to
   ``[px, py, cos theta, sin theta]``. The translation rates rotate from the
   parent frame into the body frame:

   .. code-block:: text

      [vx_body]   [ cos theta   sin theta ] [vx_parent]
      [vy_body] = [-sin theta   cos theta ] [vy_parent]

   and ``omega`` is unchanged. The map depends on ``theta``, so
   accelerations and derivatives pick up its time derivative.

Floating base, GRiD to MuJoCo
   The quaternion is reordered from ``[x, y, z, w]`` to ``[w, x, y, z]``.
   The linear velocity rotates from the body frame into the world frame,
   ``v_world = R v_body`` with ``R`` the base orientation, and the angular
   velocity is unchanged. On the six base velocity entries this is
   ``G = blockdiag(R, I)``, which is orthogonal. :doc:`mjx_convention` lists
   the resulting transform of every GRiD output, including the
   ``omega x v`` term that the acceleration picks up.

Fixed-base robots built from revolute joints with limits and prismatic
joints need no conversion in any direction.

Where the maps are implemented
------------------------------

- Floating base, GRiD and MuJoCo. The generated kernels have MuJoCo-convention
  twins, selected with ``handle.mujoco`` or ``output_convention="mujoco"``.
  This is public API and needs no conversion in user code.
- Continuous joints and mimic joints, GRiD and Pinocchio.
  ``RBDReference.equivalents.conventions`` holds the expand, collapse and
  Jacobian functions (``normalize_project_q_for_pin``,
  ``collapse_pin_q_to_project``, ``project_q_to_pin_q_jacobian``). The
  validation suite uses them to compare against Pinocchio. They are test
  support today and their signatures may change.
- Floating base, values, gradients and second derivatives, as numpy
  functions. ``RBDReference.equivalents.mujoco_convention``. Also test
  support.
- Planar joints, GRiD and Pinocchio. The map above is used by the planar
  validation tests and has no library function yet.

Why the conventions differ
--------------------------

Each choice gives something up.

Angle or ``(cos, sin)`` for an unbounded rotation
   An angle is a minimal coordinate: ``nq = nv``, the state is a plain
   vector, and a difference of two configurations is a subtraction. It is
   not unique, since ``theta`` and ``theta + 2 pi`` are the same pose, so an
   interpolation or a distance has to decide whether to wrap. ``(cos, sin)``
   is unique and never wraps, which is what Pinocchio's Lie group operations
   want. It costs an extra entry, a unit-norm constraint that integration
   must preserve, and a configuration that can no longer be differenced by
   subtraction. GRiD keeps the angle because its kernels and its users'
   optimizers treat the scalar joints as a flat vector, and it documents
   that configuration differences on continuous joints are not wrapped
   (:doc:`../../api_reference/integration_conventions`).

Body-frame or parent-frame planar velocity
   Pinocchio treats the planar joint as the group SE(2) and uses the body
   velocity, as it does for every joint. GRiD decomposes a planar joint into
   two prismatic joints and a continuous joint, so its rates are those of
   the scalar joints, which are expressed in the parent frame. The
   decomposition reuses the scalar joint kernels unchanged. MuJoCo models
   built from two slides and a hinge make the same choice.

World-frame or body-frame linear velocity of a floating base
   With body-frame velocity the base is an ordinary joint of the spatial
   algebra: the six entries are one twist in one frame, and the recursive
   algorithms need no special case. Pinocchio and GRiD do this. MuJoCo
   stores the linear part in the world frame, so the position derivative is
   the velocity itself and integrating the position is an addition. The
   price is a rotation between the stored velocity and the twist the
   dynamics use.

Quaternion order
   ``[x, y, z, w]`` is the order of Eigen's coefficient storage, which
   Pinocchio inherits, and of ROS messages. ``[w, x, y, z]`` is the order
   of the mathematical notation, which MuJoCo uses. Neither has a numerical
   advantage. The conversion is a relabel and the mistake it invites is
   silent, since a misordered unit quaternion is still a unit quaternion.

GRiD's position is to keep its conventions and convert at the boundary. Its
choices are the ones that keep every scalar joint minimal and every base
velocity a twist, and changing them would move every output of the library.
