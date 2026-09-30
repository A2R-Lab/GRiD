RBDReference
============

RBDReference is a CPU reference implementation for inspecting and validating
rigid-body algorithms. It consumes the same parsed model as GRiD's generator;
it is not the batched GPU handle.

Run a reference calculation
---------------------------

After installing GRiD from its recursive source checkout, run this from the
repository root:

.. code-block:: python

   import numpy as np
   from URDFParser import URDFParser
   from RBDReference import RBDReference

   robot = URDFParser().parse("config/robot_assets/iiwa14.urdf")
   if robot is None:
       raise ValueError("URDF parsing failed")
   rbd = RBDReference(robot)
   q = np.zeros(robot.get_num_pos())
   qd = np.zeros(robot.get_num_vel())
   qdd = np.zeros_like(qd)
   tau, v, a, f = rbd.inverse_dynamics(q, qd, qdd, GRAVITY=-9.81)
   M = rbd.crba(q)
   Minv = rbd.minv(q, output_dense=True)
   print(tau.shape, M.shape, Minv.shape)

This example is fixed-base and uses one state, not a batch. For quaternion
models initialize a valid configuration (a zero quaternion is not a valid
orientation) and use the model's position/tangent coordinate maps. The
reference's ``crba`` takes ``q``; a second positional argument is a
normalization flag, not joint velocity.

Methods and validation
-----------------------

The reference includes inverse dynamics and its derivatives, CRBA, Minv,
ABA, forward-dynamics derivatives, kinematics, centroidal quantities and
plant/cost operations. The :doc:`algorithm pages <../concepts/algorithms/index>`
distinguish reference calls from the generated GPU handle calls.

For the maintained correctness workflows, see :doc:`cuda_validation` and
``external/RBDReference/tests``. Historical root-level scripts such as
``printReferenceValues.py`` are not part of this checkout's supported workflow.

Dependencies
------------

The pure-Python reference requires NumPy, SymPy and URDFParser. Optional
Pinocchio-backed validation and second-order extensions have additional
dependencies; follow the repository installation and validation instructions
rather than treating NumPy alone as a complete oracle environment.
