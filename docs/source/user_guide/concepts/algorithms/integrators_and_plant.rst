Integrators and the plant layer
===============================

.. currentmodule:: RBDReference.RBDReference

Overview
--------
A trajectory optimizer needs more than dynamics: it needs the discrete step
``x_{k+1} = f(x_k, u_k)``, the derivatives of that step, and the costs and
constraints of the problem, all batched over the knot points. GRiD generates
these alongside the dynamics so that a whole shooting or collocation
iteration can stay on the GPU. The integrator family lives in the ``grid``
namespace; the costs and barriers are emitted as a sibling ``grid_plant``
namespace and mirrored on the Python handles.

Signature
---------
.. code-block:: python

   x_next = h.integrator(q, qd, u, dt, integrator_type="rk4")   # (B, NQ + NV)
   AB     = h.integrator_gradient(q, qd, u, dt, integrator_type="rk4")   # (B, 2*NV, 3*NV)

   x_next = h.plant_step(x, u, dt, integrator_type="euler")     # x = [q; qd], (B, NX)
   AB     = h.plant_step_gradient(x, u, dt, integrator_type="euler")    # [A | B]
   H      = h.plant_step_hessian(x, u, dt, integrator_type="euler")     # second-order sensitivity

   c = h.quadratic_state_cost(x, x_des, Q)      # 1/2 Σ Q_i (x_i − x_des_i)²
   c = h.quadratic_input_cost(u, u_des, R)
   c = h.ee_pos_cost(q, p_des, W)               # end-effector position, Gauss-Newton Hessian
   c = h.com_cost(q, p_des, W)                  # centre-of-mass tracking
   c = h.momentum_cost(q, qd, h_des, W)         # centroidal-momentum tracking
   b = h.joint_position_barrier(q, lower, upper, mu)   # log barriers; also velocity and torque

``integrator_type`` is one of ``euler``, ``semi_implicit_euler``,
``midpoint``, ``rk3`` and ``rk4`` (the generator also knows a trapezoidal
scheme); ``dt`` and the signed gravity are runtime arguments. The gradient is ``[A | B] = ∂x_{k+1}/∂(x, u)`` in the tangent
space, ``2·NV`` rows by ``3·NV`` columns, and the Hessian is the second-order
sensitivity of the same step.

Implementation
--------------
The Python references are ``plant_step``, ``plant_step_gradient``,
``plant_step_hessian`` and the cost and barrier functions of the
``_plant.py`` mixin of RBDReference. The CUDA generators are
``grid_codegen/algorithms/_integrator.py``, ``_integrator_gradient.py`` and
``_plant.py``, which emits the ``grid_plant`` namespace.

In GRiD
-------
The integrator composes forward dynamics (the mass-matrix-inverse path) with
the chosen scheme inside one kernel, so a Runge–Kutta step does not pay four
launches. Its gradient uses the analytical forward-dynamics gradient, and the
Hessian uses the second-order :doc:`fdsva_so`; the fixed-base Hessian is the
one the release test suite gates, and the floating-base variants are listed
on the :doc:`support matrix <../../tutorials/cuda_support_status>`.

On a floating base the integrator applies Pinocchio's SE(3) update to the
base pose. With ``output_convention="mujoco"`` the free-joint base position
takes MuJoCo's global additive step instead, the quaternion is reordered, and
the returned state is in the MuJoCo frame; this is baked into the kernel, not
patched on the host.

The plant layer is what the sibling trajectory-optimization projects consume
through the C ABI and the JAX and PyTorch handles: quadratic state and input
costs, an end-effector position cost with a Gauss–Newton Hessian, centre-of-
mass and centroidal-momentum costs, and log-barriers on joint positions,
velocities and torques with a runtime ``mu``. Each returns its value, its
gradient and a Hessian (exact for the quadratic costs, Gauss–Newton for the
end-effector cost), batched over the knot points.

See Also
--------
* :doc:`aba` — the forward-dynamics paths the integrator steps.
* :doc:`fdsva_so` — the second-order forward dynamics behind
  ``plant_step_hessian``.
* :doc:`kinematics` and :doc:`centroidal_and_bias` — the quantities behind
  the end-effector, centre-of-mass and momentum costs.
* :doc:`../../tutorials/python_wrappers` — the JAX and PyTorch handles, where
  these operations plug into autodiff.
