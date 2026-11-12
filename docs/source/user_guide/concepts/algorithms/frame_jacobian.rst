Frame Jacobian (general-frame geometric Jacobian)
=================================================

.. currentmodule:: RBDReference.RBDReference

Overview
--------
The frame-Jacobian family computes the :math:`6 \times n_v` geometric
Jacobian :math:`J` of an arbitrary target frame (typically an
end-effector or any named joint frame), relating generalized velocity
:math:`\dot q` to the spatial velocity (twist) of that frame. Rows are
ordered ``[linear(3); angular(3)]`` to match Pinocchio's
``getFrameJacobian`` / ``getJointJacobian``.

The same family provides two derived quantities in the reference and
generated CUDA interfaces:

* :math:`\dot J`, the Jacobian time-variation (so that the frame
  acceleration is :math:`J\,\ddot q + \dot J\,\dot q`).
* :math:`\Lambda = (J\,M^{-1}\,J^{\top})^{-1}`, the :math:`6 \times 6`
  operational-space (OSC) inertia of the task frame. :math:`\Lambda`
  is only defined when the frame is reachable by at least six
  independent DOFs (the task matrix :math:`J\,M^{-1}\,J^{\top}` must be
  well-conditioned).
* the frame velocity :math:`v = J\,\dot q` (the twist in the chosen
  reference frame) and the frame acceleration
  :math:`a = J\,\ddot q + \dot J\,\dot q`, the time derivative of that
  twist: the spatial acceleration in ``LOCAL`` and ``WORLD`` and, in
  ``LOCAL_WORLD_ALIGNED``, the classical acceleration of the frame origin
  in world axes.

Reference frames
----------------
:math:`J` (and :math:`\dot J`, :math:`\Lambda`) can be expressed in any
of the three Pinocchio reference-frame conventions, selected by a
``reference_frame`` argument:

* ``LOCAL`` (``0``): twist in the frame's own body axes.
* ``WORLD`` (``1``): spatial Jacobian at the world origin.
* ``LOCAL_WORLD_ALIGNED`` (``2``): at the frame origin, with
  world-aligned axes (the default).

Signature
---------
.. code-block:: python

   # general-frame geometric Jacobian (6 x nv)
   J      = rbd.frame_jacobian(q, frame_name, reference_frame="LOCAL_WORLD_ALIGNED")
   # Jacobian time-variation (6 x nv)
   Jdot   = rbd.frame_jacobian_dot(q, qd, frame_name, reference_frame="LOCAL_WORLD_ALIGNED")
   # operational-space (OSC) inertia (6 x 6)
   Lambda = rbd.osc_inertia(q, frame_name, reference_frame="LOCAL_WORLD_ALIGNED")
   # frame velocity (6) and frame acceleration (6)
   v      = rbd.frame_velocity(q, qd, frame_name, reference_frame="LOCAL_WORLD_ALIGNED")
   a      = rbd.frame_acceleration(q, qd, qdd, frame_name, reference_frame="LOCAL_WORLD_ALIGNED")
   # their derivatives (each 6 x nv): dv/dq, dv/dqd and dv/dq, da/dq, da/dqd, da/dqdd
   dv_dq, dv_dqd = rbd.frame_velocity_derivatives(q, qd, frame_name, reference_frame="LOCAL_WORLD_ALIGNED")
   dv_dq, da_dq, da_dqd, da_dqdd = rbd.frame_acceleration_derivatives(q, qd, qdd, frame_name,
                                                                      reference_frame="LOCAL_WORLD_ALIGNED")

Validation (Pinocchio oracle)
-----------------------------
The numpy reference is validated against Pinocchio across all three
reference frames on both a fixed-base (iiwa14) and a floating-base
(go2) robot:

* :math:`J` vs ``getFrameJacobian`` / ``getJointJacobian``.
* :math:`\dot J` vs ``getFrameJacobianTimeVariation`` /
  ``getJointJacobianTimeVariation`` (via
  ``computeJointJacobiansTimeVariation``).
* :math:`\Lambda` vs ``(J\,M^{-1}\,J^{\top})^{-1}`` built from
  Pinocchio's ``computeMinverse``.
* :math:`v` vs ``getFrameVelocity``; :math:`a` vs ``getFrameAcceleration``
  in ``LOCAL`` and ``WORLD`` and ``getFrameClassicalAcceleration`` in
  ``LOCAL_WORLD_ALIGNED`` (iiwa14, go2 and gen3 on both bases, a leaf and
  a mid-chain target).
* The derivatives vs ``getFrameVelocityDerivatives`` /
  ``getFrameAccelerationDerivatives`` (``computeForwardKinematicsDerivatives``)
  in ``LOCAL`` and ``WORLD``. In ``LOCAL_WORLD_ALIGNED`` Pinocchio's
  q-derivatives hold the frame origin fixed; the reference differentiates
  the function itself, so its q-derivatives carry the moving origin
  (``w x dp_f/dq``) and, for the acceleration, the classical term, and the
  Pinocchio values are converted before the comparison. The same
  derivatives are checked against central differences of ``frame_velocity``
  and ``frame_acceleration`` through the ``integrate`` chart on every robot
  of the manifest, mimic robots included.

See ``external/RBDReference/tests/test_frame_jacobian_equivalence.py``,
``test_frame_velocity_acceleration_equivalence.py`` and
``test_frame_kinematics_derivatives_equivalence.py``. The generated CUDA
``frame_velocity_device`` and ``frame_acceleration_device`` (opt-in keys
``frame_velocity`` and ``frame_acceleration``, with batched kernels and
three-mode hosts) are checked against the numpy reference across the three
frames and a thread-count sweep in
``test/cuda_equivalents/test_cuda_frame_kinematics.py``, as are the
derivative devices ``frame_velocity_gradient_device`` (``[dv/dq | dv/dqd]``,
6 x 2nv, key ``frame_velocity_gradient``) and
``frame_acceleration_gradient_device`` (``[da/dq | da/dqd | da/dqdd]``,
6 x 3nv, key ``frame_acceleration_gradient``).

The derivatives follow from the world-origin Plücker form. With
:math:`A_j` the column of joint :math:`j` (parent :math:`p`), the target
body :math:`i`, :math:`v` the body twists and :math:`a` the spatial
accelerations, and :math:`\times` the motion cross product,

.. math::

   \frac{\partial v_i}{\partial q_j} = A_j \times (v_i - v_p), \qquad
   \frac{\partial a_i}{\partial q_j} = A_j \times (a_i - a_p)
     + (v_p \times A_j) \times (v_i - v_p), \qquad
   \frac{\partial a_i}{\partial \dot q_j} = v_j \times A_j + A_j \times (v_i - v_p),

since every column at or below :math:`j` has :math:`\partial A_k /
\partial q_j = A_j \times A_k` (the root's own columns included, which
gives the local tangent chart). The CUDA inner runs one serial sweep over
the bodies for the columns, twists and accelerations, then one thread per
velocity slot forms its columns and applies the reference-frame map.

Implementation
--------------
The Python reference is ``RBDReference.frame_jacobian`` /
``.frame_jacobian_dot`` / ``.osc_inertia`` in
`RBDReference/RBDReference.py
<https://github.com/A2R-Lab/RBDReference>`__.

CUDA codegen lives in
`grid_codegen/algorithms/_frame_jacobian.py
<https://github.com/A2R-Lab/GRiD/tree/main/grid_codegen>`__. It is an
opt-in, additive family: the surfaces are only emitted when the
``frame_jacobian`` key is explicitly selected (it pulls in the
``end_effector_pose`` world-transform machinery), so existing profiles' headers
are byte-identical. The CUDA path emits the geometric Jacobian
:math:`J` (all three reference frames) via ``frame_jacobian_inner`` /
``frame_jacobian_device``, the Jacobian time-variation :math:`\dot J`
via ``frame_jacobian_dot_device`` (opt-in ``frame_jacobian_dot`` key),
and the OSC inertia :math:`\Lambda` via ``osc_inertia_device`` (opt-in
``osc_inertia`` key). The :math:`\Lambda` device is self-contained: it
composes :math:`M^{-1}` on-device via ``minv_inner`` (no external
:math:`M^{-1}` feed). Mimic-joint robots are supported: the geometric-Jacobian
column fold is alpha-weighted onto the shared velocity slot, and :math:`\Lambda`
routes the mimic :math:`M^{-1}` through ``crba_inner``.

``frame_jacobian`` has the same launchable surface set as the other
benchmarkable kinematics algorithms: a batched ``__global__``
``frame_jacobian_kernel`` (plus a ``_single_timing`` variant) and a 3-mode
``__host__`` launcher ``frame_jacobian`` / ``frame_jacobian_single_timing`` /
``frame_jacobian_compute_only`` that reads/writes the ``gridData`` output buffer
``hd_data->d_frame_jacobian`` (6 × NUM_VEL, copied back into
``h_frame_jacobian``). The host/kernel surface accepts runtime
``target_jid`` and ``reference_frame`` arguments, defaulting to the leaf-EE
joint and ``LOCAL_WORLD_ALIGNED``. The
host surface is validated end-to-end against the numpy oracle in
``test/cuda_equivalents/test_cuda_frame_jacobian_host.py`` (the device functions
are covered by ``test_cuda_frame_jacobian.py``).

``frame_jacobian_dot`` (opt-in ``frame_jacobian_dot`` key) has the same
launchable set: ``frame_jacobian_dot_kernel`` (+ ``_single_timing``) and the
3-mode host ``frame_jacobian_dot`` / ``_single_timing`` / ``_compute_only``,
reading the packed ``[q; qd]`` input and writing
``hd_data->d_frame_jacobian_dot`` (6 × NUM_VEL). The kernel keeps its input /
output in static ``__shared__`` because ``frame_jacobian_dot_device`` owns the
whole dynamic-smem arena; runtime target/frame selection has the same
leaf-EE / ``LOCAL_WORLD_ALIGNED`` defaults. It is covered by the same host-surface test (validated for
iiwa14-fixed + go2-floating).

``osc_inertia`` (opt-in ``osc_inertia`` key) has
``osc_inertia_kernel`` (+ ``_single_timing``) and the 3-mode host
``osc_inertia`` / ``_single_timing`` / ``_compute_only``, writing
``hd_data->d_osc_inertia`` (6 × 6). It is self-contained (q-only input;
composes :math:`M^{-1}` on-device), keeps its q input + Λ output in static
``__shared__``, and carries ``__launch_bounds__`` so nvcc fits the heavy
``minv``/``J``/``invert`` register footprint to the tier thread cap (the
un-annotated device smoke runner instead clamps threads manually). It has the same
host-surface test coverage (Λ checked at non-singular configs).

Python handle calls
-------------------

Select ``frame_jacobian``, ``frame_jacobian_dot``, ``frame_velocity``,
``frame_acceleration``, ``frame_velocity_gradient``,
``frame_acceleration_gradient`` and/or ``osc_inertia`` in ``algorithm_list``
when building the handle (the default list has all seven). The NumPy calls
are::

   J = h.frame_jacobian(q, target_jid=jid, reference_frame="LOCAL")  # (B, 6, NV)
   Jdot = h.frame_jacobian_dot(q, qd, target_jid=jid,
                              reference_frame="LOCAL_WORLD_ALIGNED")  # (B, 6, NV)
   v = h.frame_velocity(q, qd, target_jid=jid, reference_frame="WORLD")  # (B, 6)
   a = h.frame_acceleration(q, qd, qdd, target_jid=jid,
                            reference_frame="WORLD")  # (B, 6)
   dv = h.frame_velocity_gradient(q, qd, target_jid=jid)  # (B, 6, 2 NV): [dv/dq | dv/dqd]
   da = h.frame_acceleration_gradient(q, qd, qdd, target_jid=jid)  # (B, 6, 3 NV): [da/dq | da/dqd | da/dqdd]
   Lambda = h.osc_inertia(q)  # (B, 6, 6), baked leaf-EE / LWA target

The JAX and PyTorch handles expose the same seven methods with the same
arguments (forward only). Unlike the other methods, the ``osc_inertia``
handle does not expose runtime target/frame arguments. Its CUDA device
function does. Do not use an undamped OSC inverse at a singular task
configuration.

See Also
--------
* :doc:`crba`: joint-space mass matrix :math:`M` (used to form
  :math:`\Lambda`).
* :doc:`minv`: direct :math:`M^{-1}` (the inverse used inside
  :math:`J\,M^{-1}\,J^{\top}`).
* :doc:`inverse_dynamics`: inverse dynamics (RNEA).
