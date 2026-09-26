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
   H      = h.plant_step_hessian(x, u, dt, integrator_type="euler")     # NumPy: (B, 2*NV, 3*NV, 3*NV)

   value, grad, Hcost = h.quadratic_state_cost(x, x_des, Q)
   value, grad, Hcost = h.quadratic_input_cost(u, u_des, R)
   value, grad, Hcost = h.ee_pos_cost(q, p_des, W)       # first EE only
   value, grad, Hcost = h.com_cost(q, p_des, W)
   value, grad, Hcost = h.momentum_cost(q, qd, h_des, W) # velocity-only derivative blocks
   value, grad, Hdiag = h.joint_position_barrier(q, lower, upper, mu)

``integrator_type`` is one of ``euler``, ``semi_implicit_euler``,
``midpoint``, ``rk3``, ``rk4`` and ``trapezoidal`` (``si_euler`` is an alias
for semi-implicit Euler); the default is ``euler``. ``dt`` and
``gravity=-9.81`` are runtime arguments. The gradient is ``[A | B] = ∂x_{k+1}/∂(x, u)`` in the tangent
space, ``2·NV`` rows by ``3·NV`` columns, and the Hessian is the second-order
sensitivity of the same step.

.. important::

   The following describes the generated GPU implementation. The CPU
   reference has been updated to full-state midpoint, explicit Heun
   (``trapezoidal``), and RK4; the previous one-evaluation ``trapezoidal``
   is now ``constant_acceleration``, and ``rk3``/``si_euler`` are removed
   there. Reference tests do not certify the GPU implementation. The reference
   uses base-point retractions, with second-order rotational accuracy even
   for RK4, rather than Munthe-Kaas corrections.

   The names ``midpoint``, ``rk3`` and ``rk4`` refer to GRiD's current
   TrajoptPlant-style schemes. Intermediate configurations use the original
   velocity, and the final position is ``integrate(q, dt*qd)``; only the
   velocity update combines the intermediate acceleration evaluations.
   This is not textbook Runge–Kutta integration of the full ``[q, qd]``
   state, and the name ``rk4`` is not a claim of fourth-order state accuracy.
   ``trapezoidal`` likewise uses one acceleration evaluation with a
   half-acceleration position term, not an implicit trapezoidal solve.

Inputs, outputs and scope
---------------------------

A runnable :doc:`CPU diagnostic <../../tutorials/verified_inputs>` illustrates
the current RK position-update semantics on constant acceleration.

These examples require a handle built with the relevant algorithms. The
``integrator`` calls take ``q`` at ``(B, NQ)`` and ``qd``, ``u`` at ``(B, NV)``;
the ``plant_step`` calls take ``x`` of shape ``(B, NQ+NV)`` and ``u`` of shape
``(B, NV)``. Both return the same position-plus-velocity state.
``NX = NQ + NV``; derivative outputs use the ``2*NV`` tangent state, not
the ambient quaternion coordinates.

``plant_step_hessian`` is a NumPy/CUDA surface, not a JAX or PyTorch handle
method. It supports Euler and semi-implicit Euler on fixed and floating
bases; multi-stage RK Hessians are not implemented. Multi-stage integrator
gradients are not available for spherical joints or MuJoCo-output twins.
See :doc:`../../tutorials/python_wrappers` for backend coverage.

Weights are diagonal, supplied as vectors: state ``x_des`` and ``Q`` have
shape ``(B, NX)``, input ``u_des`` and ``R`` have ``(B, NV)``, and tracking
targets and weights have ``(B, 3)`` for position or ``(B, 6)`` for momentum.
Cost outputs are a tuple of value ``(B,)``, gradient and Hessian; the
state/tracking costs use ``NX``-sized outputs, and input costs use ``NV``.
Position barriers take ``(B, NQ)`` bounds and velocity/torque barriers take
``(B, NV)`` bounds. Their third output is the Hessian diagonal, not a dense
matrix. Finite log-barrier bounds require strictly interior inputs;
infinite bounds contribute no term.

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
the chosen scheme inside one kernel, so the four acceleration stages of
``rk4`` do not require four kernel launches. Its gradient uses the analytical forward-dynamics gradient, and the
step Hessian composes the second-order :doc:`fdsva_so` with the integration
map (including the floating-base retract derivatives). This fused integrator
kernel does not make an arbitrary sequence of Python calls a single launch.

On a floating base the integrator applies Pinocchio's SE(3) update to the
base pose. With ``output_convention="mujoco"`` the free-joint base position
takes MuJoCo's global additive step instead, the quaternion is reordered, and
the returned state is in the MuJoCo frame; this is baked into the kernel, not
patched on the host.

The plant layer exposes quadratic state and input
costs, an end-effector position cost with a Gauss–Newton Hessian, centre-of-
mass and centroidal-momentum costs, and log-barriers on joint positions,
velocities and torques with a runtime ``mu``. Quadratic costs have exact
ambient-coordinate Hessians; end-effector-position and CoM tracking use
Gauss–Newton Hessians. The current momentum cost deliberately drops the
configuration derivative: it returns ``[0; A.T @ (W*r)]`` and only the
velocity–velocity Hessian block ``A.T @ diag(W) @ A``. These are not the
full derivatives of ``h(q, qd)`` with respect to the state. On quaternion
models, geometric tracking derivatives occupy tangent blocks embedded in
the ``NX``-sized outputs; they are not ambient quaternion Hessians.

The updated CPU ``RBDReference.momentum_cost`` instead returns the full
gradient and full Gauss–Newton Hessian in ``2*NV`` tangent-state coordinates,
including configuration and cross blocks, using
``J = [(dA/dq)*qd | A]``. Its Hessian is ``J.T @ diag(W) @ J``, not the exact
cost Hessian at a general nonzero residual. This reference update must not
be mistaken for availability in a previously generated GPU artifact.

Select ``integrator`` / ``integrator_gradient`` for step values / gradients
and ``fdsva_so`` for the step Hessian dependency; the plant generator only
emits operations whose dependencies exist. CoM and momentum costs require
``com`` and ``ccrba``. Use the wrapper guide's available-operation checks
after registration rather than assuming every handle includes the full
plant layer.

See Also
--------
* :doc:`aba` — the forward-dynamics paths the integrator steps.
* :doc:`fdsva_so` — the second-order forward dynamics behind
  ``plant_step_hessian``.
* :doc:`kinematics` and :doc:`centroidal_and_bias` — the quantities behind
  the end-effector, centre-of-mass and momentum costs.
* :doc:`../../tutorials/python_wrappers` — the JAX and PyTorch handles, where
  these operations plug into autodiff.
