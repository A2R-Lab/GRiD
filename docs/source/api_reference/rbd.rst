RBDReference index
===================

A Python reference implementation of rigid body dynamics algorithms.

This package is designed to enable rapid prototyping and testing of new
algorithms and algorithmic optimizations. If your favorite rigid body
dynamics algorithm is not yet implemented please submit a PR with the
implementation. We'll then try to get a GPU, FPGA, and/or accelerator
implementation designed as soon as possible.

Usage and API:
--------------

This package relies on an already parsed ``robot`` object from our
`URDFParser <https://github.com/robot-acceleration/URDFParser>`__
package.

.. code:: python

   RBDReference = RBDReference(robot)
   outputs = RBDReference.ALGORITHM(inputs)

Currently implemented algorithms include:

**Core dynamics**

* Inverse dynamics (``inverse_dynamics``; the Recursive Newton-Euler Algorithm, RNEA):
  ``(c, v, a, f) = rbd.inverse_dynamics(q, qd, qdd=None, GRAVITY=-9.81)``
* Articulated Body Algorithm (ABA, forward dynamics):
  ``qdd = rbd.aba(q, qd, tau, f_ext=[], GRAVITY=-9.81)``
* Composite Rigid Body Algorithm (mass matrix): ``M = rbd.crba(q)``
* Direct Inverse Mass Matrix: ``Minv = rbd.minv(q, output_dense=True)``
* Forward Dynamics (``Minv (τ − c)``): ``qdd = rbd.forward_dynamics(q, qd, u)``

**Gradients**

* ``dc_du = rbd.inverse_dynamics_gradient(q, qd, qdd=None, GRAVITY=-9.81)`` returning
  ``np.hstack((dc_dq, dc_dqd))``
* ``(dqdd_dq, dqdd_dqd) = rbd.forward_dynamics_gradient(q, qd, u)``

**End-effector kinematics**

* Pose: ``ee = rbd.end_effector_pose(q, ee_joint_names=None, ee_offsets=None)``
* Pose Jacobian: ``dee = rbd.end_effector_pose_gradient(q, ...)`` —
  ``d(pose)/dv`` in **tangent space** (6 × nv per EE), matching pinocchio's
  convention. Fixed-base ``nv == nq`` so the shape is unchanged from earlier
  versions; floating-base outputs the spatial Jacobian (``[omega; v]`` base
  block, ``nv = 6 + n_joints``) rather than the older non-standard
  quaternion-derivative columns.
* Pose Hessian: ``d2ee = rbd.end_effector_pose_hessian(q, ...)`` — still ``d²(pose)/dq²``
  for now; a follow-up will switch it to ``d²(pose)/dv²`` for consistency.

**Second-order**

* IDSVA-SO meta dispatcher (auto-picks body/world frame by base type):
  ``(d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq) = rbd.idsva_so(q, qd, qdd, GRAVITY=-9.81)``
* Body-frame variant (fastest on fixed-base):
  ``... = rbd.idsva_so_body_frame(q, qd, qdd, GRAVITY=-9.81)``
* World-frame variant (fastest on floating-base):
  ``... = rbd.idsva_so_world_frame(q, qd, qdd, GRAVITY=-9.81)``
* Second-order forward dynamics:
  ``... = rbd.fdsva_so(q, qd, u, GRAVITY=-9.81)``

The two IDSVA-SO variants are mathematically equivalent — they differ
only in reference frame. Either is correct on either base type; the
dispatch is purely a performance optimization (see
:doc:`../user_guide/concepts/algorithms/idsva`).

We also include functions that break these algorithms down into their
different passes and by their output types (dq vs dqd) to enable easier
testing of downstream GPU, FPGA, and accelerator implementations.

Testing Algorithms on URDFs
---------------------------

Run the following in terminal from forked repo here at base of the GRiD
working directory in the floating base branch. Run any URDF tests by
replacing the desired urdf with the terminal scripts below.

iiwa14.urdf testing ``python printReferenceValues.py iiwa.urdf -f``

This runs the print printReferenceValues.py script located
https://github.com/A2R-Lab/GRiD/blob/floating-base/printReferenceValues.py.
This will output values for ``RNEA``, ``Minv``, ``CRBA``, ``RNEA_grad``,
and more.

Once can test algorithm outputs using Matlab Spatial V2 using the
following sequence: 
1. Run ``python compare_rnea_fb_spatial.py INSERT_URDF.urdf -f`` which will
print the exact q, qd, qdd inputs for Spatial V2 floating base
algorithms. (Matlab expects floating base joint to input quaternion xyzw
and xyz location of Fb joint) 

2. After generating compatible inputs with
Matlab, create a floating base URDF.m file by running
``python generate_spatial_model.py INSERT_URDF.urdf -f``. 

3. Place the
newly generated ``URDF.m`` file in the working directory of
spatial_v2_extended. 

4. Run the script after ensuring spatial
directories are sourced correctly. 

5. Suggested testing includes
``RNEA (ID)``, ``HandC``, ``Hinverse``, and more.


Dependencies and Installation
-----------------------------

The only external dependency is ``numpy``, which can be automatically installed by running:

.. code:: shell

   pip3 install -r requirements.txt

This package also depends on our 
`URDFParser <https://github.com/robot-acceleration/URDFParser>`__ package.

API Reference
--------------

The following sections provide an overview of each function, including its purpose, parameters, return values, and example usage.

Functions
---------

.. function:: __init__(robotObj)

   Description
   ^^^^^^^^^^^

   Initializes the `RBDReference` class, associating it with a given robot object.

   Parameters
   ^^^^^^^^^^

   - **robotObj** : *object*
      The robot object to associate with this reference. It should represent a multibody system.

   Returns
   ^^^^^^^

   - **None** : *None*
      This function does not return a value.

   Example
   ^^^^^^^

   .. code-block:: python

      from your_module import RBDReference

      robot_obj = SomeRobotObject()
      rbd_ref = RBDReference(robot_obj)

----

.. function:: cross_operator(v)

   Description
   ^^^^^^^^^^^

   Computes the cross product operator for a given vector `v`.

   Parameters
   ^^^^^^^^^^

   - **v** : *ndarray*
      A vector representing the spatial velocity (or any other relevant quantity) to compute the cross product operator.

   Returns
   ^^^^^^^

   - **result** : *ndarray*
      A 6x6 matrix representing the cross product operator of the input vector.

   Example
   ^^^^^^^

   .. code-block:: python

      from your_module import RBDReference

      v = np.array([1, 2, 3, 4, 5, 6])
      rbd_ref = RBDReference(robot_obj)
      result = rbd_ref.cross_operator(v)
      print(result)

----

.. function:: dual_cross_operator(v)

   Description
   ^^^^^^^^^^^

   Computes the dual cross product operator for a given vector `v`.

   Parameters
   ^^^^^^^^^^

   - **v** : *ndarray*
      A vector representing the spatial velocity (or any other relevant quantity) to compute the dual cross product operator.

   Returns
   ^^^^^^^

   - **result** : *ndarray*
      A 6x6 matrix representing the dual cross product operator of the input vector.

   Example
   ^^^^^^^

   .. code-block:: python

      from your_module import RBDReference

      v = np.array([1, 2, 3, 4, 5, 6])
      rbd_ref = RBDReference(robot_obj)
      result = rbd_ref.dual_cross_operator(v)
      print(result)

----

.. function:: icrf(v)

   Description
   ^^^^^^^^^^^

   Computes the inverse cross product matrix for a vector `v`.

   Parameters
   ^^^^^^^^^^

   - **v** : *ndarray*
      A vector representing the spatial velocity (or any other relevant quantity).

   Returns
   ^^^^^^^

   - **result** : *ndarray*
      A 6x6 matrix representing the inverse cross product matrix for the input vector.

   Example
   ^^^^^^^

   .. code-block:: python

      from your_module import RBDReference

      v = np.array([1, 2, 3, 4, 5, 6])
      rbd_ref = RBDReference(robot_obj)
      result = rbd_ref.icrf(v)
      print(result)

----

.. function:: factor_functions(I, v, number=3)

   Description
   ^^^^^^^^^^^

   Computes a factor function for the system, which is used in various multibody dynamics algorithms.

   Parameters
   ^^^^^^^^^^

   - **I** : *ndarray*
      The inertia matrix for the system.

   - **v** : *ndarray*
      The velocity vector for the system.

   - **number** : *int, optional*
      A number that selects which formula to use (default is 3).

   Returns
   ^^^^^^^

   - **result** : *ndarray*
      The resulting factor matrix for the system.

   Example
   ^^^^^^^

   .. code-block:: python

      from your_module import RBDReference

      I = np.eye(6)
      v = np.array([1, 2, 3, 4, 5, 6])
      rbd_ref = RBDReference(robot_obj)
      result = rbd_ref.factor_functions(I, v)
      print(result)

----

.. function:: _mxS(S, vec, alpha=1.0)

   Description
   ^^^^^^^^^^^

   Computes the spatial cross product between vectors `S` and `vec`.

   Parameters
   ^^^^^^^^^^

   - **S** : *ndarray*
      The first vector (spatial motion vector).

   - **vec** : *ndarray*
      The second vector (spatial force or other relevant vector).

   - **alpha** : *float, optional*
      A scaling factor for the operation (default is 1.0).

   Returns
   ^^^^^^^

   - **result** : *ndarray*
      The result of the spatial cross product.

   Example
   ^^^^^^^

   .. code-block:: python

      from your_module import RBDReference

      S = np.array([1, 2, 3, 4, 5, 6])
      vec = np.array([6, 5, 4, 3, 2, 1])
      rbd_ref = RBDReference(robot_obj)
      result = rbd_ref._mxS(S, vec)
      print(result)

----

.. function:: mxS(S, vec)

   Description
   ^^^^^^^^^^^

   Computes the spatial cross product for a given set of vectors `S` and `vec`.

   Parameters
   ^^^^^^^^^^

   - **S** : *ndarray*
      A vector representing the spatial motion.

   - **vec** : *ndarray*
      A vector representing the spatial velocity or force.

   Returns
   ^^^^^^^

   - **result** : *ndarray*
      The resulting spatial cross product.

   Example
   ^^^^^^^

   .. code-block:: python

      from your_module import RBDReference

      S = np.array([1, 2, 3, 4, 5, 6])
      vec = np.array([6, 5, 4, 3, 2, 1])
      rbd_ref = RBDReference(robot_obj)
      result = rbd_ref.mxS(S, vec)
      print(result)

----

.. function:: fxv(fxVec, timesVec)

   Description
   ^^^^^^^^^^^

   Computes the result of multiplying two vectors `fxVec` and `timesVec` using spatial operations.

   Parameters
   ^^^^^^^^^^

   - **fxVec** : *ndarray*
      The first vector to be used in the operation.

   - **timesVec** : *ndarray*
      The second vector to be used in the operation.

   Returns
   ^^^^^^^

   - **result** : *ndarray*
      The resulting vector after applying the spatial cross product.

   Example
   ^^^^^^^

   .. code-block:: python

      from your_module import RBDReference

      fxVec = np.array([1, 2, 3, 4, 5, 6])
      timesVec = np.array([6, 5, 4, 3, 2, 1])
      rbd_ref = RBDReference(robot_obj)
      result = rbd_ref.fxv(fxVec, timesVec)
      print(result)

----

.. function:: fxS(S, vec, alpha=1.0)

   Description
   ^^^^^^^^^^^

   Computes the spatial cross product between a matrix `S` and a vector `vec`.

   Parameters
   ^^^^^^^^^^

   - **S** : *ndarray*
      The matrix to apply the spatial cross product on.

   - **vec** : *ndarray*
      The vector to apply the spatial cross product with.

   - **alpha** : *float, optional*
      A scaling factor for the operation (default is 1.0).

   Returns
   ^^^^^^^

   - **result** : *ndarray*
      The resulting spatial cross product.

   Example
   ^^^^^^^

   .. code-block:: python

      from your_module import RBDReference

      S = np.array([[1, 0], [0, 1]])
      vec = np.array([1, 2, 3, 4, 5, 6])
      rbd_ref = RBDReference(robot_obj)
      result = rbd_ref.fxS(S, vec)
      print(result)

----

.. function:: vxIv(vec, Imat)

   Description
   ^^^^^^^^^^^

   Computes the result of multiplying a vector `vec` by an inertia matrix `Imat` using spatial operations.

   Parameters
   ^^^^^^^^^^

   - **vec** : *ndarray*
      A vector to be multiplied by the inertia matrix.

   - **Imat** : *ndarray*
      The inertia matrix to multiply the vector with.

   Returns
   ^^^^^^^

   - **result** : *ndarray*
      The resulting vector after the multiplication.

   Example
   ^^^^^^^

   .. code-block:: python

      from your_module import RBDReference

      vec = np.array([1, 2, 3, 4, 5, 6])
      Imat = np.eye(6)
      rbd_ref = RBDReference(robot_obj)
      result = rbd_ref.vxIv(vec, Imat)
      print(result)

----

.. function:: apply_external_forces(self, q, f_in, f_ext)

   Description
   ^^^^^^^^^^^

   Subtracts external forces from the input forces, applying the external forces to the system. This method takes the structure of either a 6/3xNB matrix or a shortened planar vector with length == NB, where `f[i]` corresponds to the force applied to body `i`.

   Parameters
   ^^^^^^^^^^

   - **f_in** : *ndarray*
      The initial forces applied to links.

   - **f_ext** : *ndarray*
      The external forces to subtract.

   Returns
   ^^^^^^^

   - **f_out** : *ndarray*
      The updated forces after applying the external forces.

   Example
   ^^^^^^^

   .. code-block:: python

      from your_module import RBDReference

      q = np.array([1, 2, 3])
      f_in = np.array([1, 1, 1, 0, 0, 0])
      f_ext = np.array([0, 0, 0, 0, 0, 0])
      rbd_ref = RBDReference(robot_obj)
      result = rbd_ref.apply_external_forces(q, f_in, f_ext)
      print(result)

----

.. function:: inverse_dynamics_fpass(self, q, qd, qdd=None, GRAVITY=-9.81)

   Description
   ^^^^^^^^^^^

   Performs the forward pass of the Recursive Newton-Euler Algorithm (RNEA) for computing spatial velocities, accelerations, and forces.

   Parameters
   ^^^^^^^^^^

   - **q** : *ndarray*
      The joint positions of the robot.

   - **qd** : *ndarray*
      The joint velocities.

   - **qdd** : *ndarray*, optional
      The joint accelerations.

   - **GRAVITY** : *float*, optional
      The gravitational constant. Default is -9.81.

   Returns
   ^^^^^^^

   - **v** : *ndarray*
      The spatial velocities.

   - **a** : *ndarray*
      The spatial accelerations.

   - **f** : *ndarray*
      The spatial forces.

   Example
   ^^^^^^^

   .. code-block:: python

      from your_module import RBDReference

      q = np.array([1, 2, 3])
      qd = np.array([0, 1, 0])
      rbd_ref = RBDReference(robot_obj)
      v, a, f = rbd_ref.inverse_dynamics_fpass(q, qd)
      print(v, a, f)

----

.. function:: inverse_dynamics_bpass(self, q, f)

   Description
   ^^^^^^^^^^^

   Performs the backward pass of the Recursive Newton-Euler Algorithm (RNEA), computing joint forces.

   Parameters
   ^^^^^^^^^^

   - **q** : *ndarray*
      The joint positions.

   - **f** : *ndarray*
      The spatial forces.

   Returns
   ^^^^^^^

   - **c** : *ndarray*
      The computed joint forces.

   - **f** : *ndarray*
      The updated spatial forces.

   Example
   ^^^^^^^

   .. code-block:: python

      from your_module import RBDReference

      q = np.array([1, 2, 3])
      f = np.array([0, 0, 0, 0, 0, 0])
      rbd_ref = RBDReference(robot_obj)
      c, f = rbd_ref.inverse_dynamics_bpass(q, f)
      print(c, f)

----

.. function:: inverse_dynamics(self, q, qd, qdd=None, GRAVITY=-9.81, f_ext=None)

   Description
   ^^^^^^^^^^^

   Computes the Recursive Newton-Euler Algorithm (RNEA) by performing both the forward and backward passes.

   Parameters
   ^^^^^^^^^^

   - **q** : *ndarray*
      The joint positions of the robot.

   - **qd** : *ndarray*
      The joint velocities.

   - **qdd** : *ndarray*, optional
      The joint accelerations.

   - **GRAVITY** : *float*, optional
      The gravitational constant. Default is -9.81.

   - **f_ext** : *ndarray*, optional
      The external forces to apply.

   Returns
   ^^^^^^^

   - **c** : *ndarray*
      The computed joint forces.

   - **v** : *ndarray*
      The spatial velocities.

   - **a** : *ndarray*
      The spatial accelerations.

   - **f** : *ndarray*
      The spatial forces.

   Example
   ^^^^^^^

   .. code-block:: python

      from your_module import RBDReference

      q = np.array([1, 2, 3])
      qd = np.array([0, 1, 0])
      rbd_ref = RBDReference(robot_obj)
      c, v, a, f = rbd_ref.inverse_dynamics(q, qd)
      print(c, v, a, f)

----

.. function:: inverse_dynamics_gradient_fpass_dq(self, q, qd, v, a, GRAVITY=-9.81)

   Description
   ^^^^^^^^^^^

   Computes the forward pass gradient of the Recursive Newton-Euler Algorithm (RNEA) with respect to joint positions.

   Parameters
   ^^^^^^^^^^

   - **q** : *ndarray*
      The joint positions.

   - **qd** : *ndarray*
      The joint velocities.

   - **v** : *ndarray*
      The spatial velocities.

   - **a** : *ndarray*
      The spatial accelerations.

   - **GRAVITY** : *float*, optional
      The gravitational constant. Default is -9.81.

   Returns
   ^^^^^^^

   - **dv_dq** : *ndarray*
      The derivative of spatial velocities with respect to joint positions.

   - **da_dq** : *ndarray*
      The derivative of spatial accelerations with respect to joint positions.

   - **df_dq** : *ndarray*
      The derivative of spatial forces with respect to joint positions.

   Example
   ^^^^^^^

   .. code-block:: python

      from your_module import RBDReference

      q = np.array([1, 2, 3])
      qd = np.array([0, 1, 0])
      v = np.array([0, 0, 0])
      a = np.array([0, 0, 0])
      rbd_ref = RBDReference(robot_obj)
      dv_dq, da_dq, df_dq = rbd_ref.inverse_dynamics_gradient_fpass_dq(q, qd, v, a)
      print(dv_dq, da_dq, df_dq)

----

.. function:: inverse_dynamics_gradient_fpass_dqd(self, q, qd, v)

   Description
   ^^^^^^^^^^^

   Computes the forward pass gradient of the Recursive Newton-Euler Algorithm (RNEA) with respect to joint velocities.

   Parameters
   ^^^^^^^^^^

   - **q** : *ndarray*
      The joint positions.

   - **qd** : *ndarray*
      The joint velocities.

   - **v** : *ndarray*
      The spatial velocities.

   Returns
   ^^^^^^^

   - **dv_dqd** : *ndarray*
      The gradient of spatial velocities with respect to joint velocities.

   - **da_dqd** : *ndarray*
      The gradient of spatial accelerations with respect to joint velocities.

   - **df_dqd** : *ndarray*
      The gradient of spatial forces with respect to joint velocities.

   Example
   ^^^^^^^

   .. code-block:: python

      from your_module import RBDReference

      q = np.array([1, 2, 3])
      qd = np.array([0, 1, 0])
      v = np.array([0, 0, 0])
      rbd_ref = RBDReference(robot_obj)
      dv_dqd, da_dqd, df_dqd = rbd_ref.inverse_dynamics_gradient_fpass_dqd(q, qd, v)
      print(dv_dqd, da_dqd, df_dqd)

----

.. function:: inverse_dynamics_gradient_bpass_dq(self, q, f, df_dq)

   Description
   ^^^^^^^^^^^

   Computes the backward pass gradient of the Recursive Newton-Euler Algorithm (RNEA) with respect to joint positions.

   Parameters
   ^^^^^^^^^^

   - **q** : *ndarray*
      The joint positions.

   - **f** : *ndarray*
      The spatial forces.

   - **df_dq** : *ndarray*
      The gradient of joint forces with respect to joint positions.

   Returns
   ^^^^^^^

   - **dc_dq** : *ndarray*
      The gradient of RNEA with respect to joint positions.

   Example
   ^^^^^^^

   .. code-block:: python

      from your_module import RBDReference

      q = np.array([1, 2, 3])
      f = np.array([0, 0, 0, 0, 0, 0])
      df_dq = np.array([0, 0, 0])
      rbd_ref = RBDReference(robot_obj)
      dc_dq = rbd_ref.inverse_dynamics_gradient_bpass_dq(q, f,

     

Additional Resources
--------------------

For more information on how to use this package, please see:
- `Sphinx Documentation <https://www.sphinx-doc.org/en/master/>`_