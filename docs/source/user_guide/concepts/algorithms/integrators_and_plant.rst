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
   value, grad, Hcost = h.momentum_cost(q, qd, h_des, W) # tangent [dq | dv] derivatives
   value, grad, Hdiag = h.joint_position_barrier(q, lower, upper, mu)

``integrator_type`` is one of ``euler``, ``semi_implicit_euler``,
``midpoint``, ``rk4``, ``trapezoidal`` and ``constant_acceleration``;
the default is ``euler``. ``rk3`` and the ``si_euler`` alias are removed. ``dt`` and
``gravity=-9.81`` are runtime arguments. The gradient is ``[A | B] = ∂x_{k+1}/∂(x, u)`` in the tangent
space, ``2·NV`` rows by ``3·NV`` columns, and the Hessian is the second-order
sensitivity of the same step.

.. important::

   Regenerate previously built GPU artifacts after this contract change.
   ``trapezoidal`` means explicit two-stage Heun, not an implicit solve.
   The old one-evaluation formula is named ``constant_acceleration``:
   ``q_next = integrate(q, dt*v + 0.5*dt**2*a)``, ``v_next = v + dt*a``.

   Midpoint, Heun and RK4 advance the full state. Each stage retracts from
   the initial configuration using the preceding stage velocity, evaluates
   forward dynamics at that stage configuration and velocity, and combines
   both velocities and accelerations with the scheme's weights. Controls
   and body-local external forces stay constant throughout the step.
   Midpoint/Heun have second order and RK4 fourth order on Euclidean
   configurations. The base-point retraction limits rotational convergence
   to second order even for RK4; no Munthe–Kaas correction is applied.

Inputs, outputs and scope
---------------------------

A runnable :doc:`CPU diagnostic <../../tutorials/verified_inputs>` illustrates
the full-state position update on constant acceleration.

These examples require a handle built with the relevant algorithms. The
``integrator`` calls take ``q`` at ``(B, NQ)`` and ``qd``, ``u`` at ``(B, NV)``;
the ``plant_step`` calls take ``x`` of shape ``(B, NQ+NV)`` and ``u`` of shape
``(B, NV)``. Both return the same position-plus-velocity state.
``NX = NQ + NV``; derivative outputs use the ``2*NV`` tangent state, not
the ambient quaternion coordinates.

``plant_step_hessian`` is a NumPy/CUDA surface, not a JAX or PyTorch handle
method. It supports Euler and semi-implicit Euler on fixed and floating
bases; other scheme Hessians are not implemented. Multi-stage integrator
gradients are not available for spherical joints. MuJoCo-output integration
values and gradients currently support Euler and semi-implicit Euler only.
See :doc:`../../tutorials/python_wrappers` for backend coverage.

.. note::

   **Planned after this release.** Step Hessians for the multi-stage schemes,
   multi-stage gradients on spherical joints, MuJoCo-convention multi-stage and
   constant-acceleration integration, and the momentum cost on the largest
   humanoids (its fused ``dccrba`` evaluation runs through the spill tiers).
   Each of these currently raises a clear error rather than silently degrading.

Weights are diagonal, supplied as vectors: state ``x_des`` and ``Q`` have
shape ``(B, NX)``, input ``u_des`` and ``R`` have ``(B, NV)``, and tracking
targets and weights have ``(B, 3)`` for position or ``(B, 6)`` for momentum.
Cost outputs are a tuple of value ``(B,)``, gradient and Hessian; the
state/position-tracking costs use ``NX``-sized outputs, and input costs use ``NV``.
Momentum returns a ``2*NV`` gradient and ``(2*NV, 2*NV)`` Gauss–Newton Hessian
in tangent ``[dq | dv]`` coordinates, including configuration and cross blocks.
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
Gauss–Newton Hessians. Momentum uses ``r = A(q)*qd - h_des`` and the full
residual Jacobian ``J = [(dA/dq)*qd | A]`` from ``dccrba``. Its gradient is
``J.T @ (W*r)`` and Hessian ``J.T @ diag(W) @ J``. This omits residual
curvature, not configuration dependence, and is not the exact cost Hessian
at a general nonzero residual. MuJoCo momentum derivatives include the
configuration dependence of the velocity-frame conversion before forming
the gradient and Hessian. On quaternion models, position-tracking
derivatives retain tangent blocks embedded in ``NX``-sized outputs;
momentum uses the compact ``2*NV`` tangent layout.

Select ``integrator`` / ``integrator_gradient`` for step values / gradients
and ``fdsva_so`` for the step Hessian dependency; the plant generator only
emits operations whose dependencies exist. CoM costs require ``com`` and
``ccrba``; full-state momentum costs require ``dccrba``. CUDA momentum
callers supply a tier-sized dccrba arena and a workspace slot for spills.
Use the wrapper guide's available-operation checks
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
