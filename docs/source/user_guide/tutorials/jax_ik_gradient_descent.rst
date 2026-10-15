Gradient descent through GRiD with plain JAX
============================================

This page solves inverse kinematics for the iiwa14 by gradient descent on the
end-effector position error, with ``jax.grad`` running through GRiD's
end-effector kinematics. It needs ``pip install "grid-rbd[jax]"``, a CUDA
``jax`` wheel, ``nvcc`` on the path and a GPU, and no optimiser library. The
complete script is ``examples/jax/ik_gradient_descent.py`` in the repository.

What GRiD provides and what JAX provides
-----------------------------------------

GRiD provides the differentiable primitive. On the JAX handle,
``robot.end_effector_pose(q)`` is a batched FFI call into the compiled
per-robot kernels, wrapped in ``jax.custom_vjp`` so that its backward is
GRiD's analytic end-effector Jacobian, another FFI call. Nothing is traced
through the dynamics and nothing is finite-differenced. The forward and the
backward run on the device and return device arrays.

JAX provides the rest. ``jax.grad`` turns a scalar loss on that primitive into
a gradient, ``jax.jit`` with ``jax.lax.fori_loop`` compiles the whole descent
into one GPU program, and the leading batch axis solves several targets in one
call. The update rule is written out by hand.

Load the robot
--------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import grid_rbd

   robot = grid_rbd.load_robot("grid_assets/robots/iiwa14.urdf", backend="jax")
   nq = robot.nq

The first call generates and compiles the robot once; later calls load the
cached ``.so`` in seconds (:doc:`../getting_started/fast_robot_setup`).

Targets
-------

The targets are end-effector positions of known joint configurations, so each
one is reachable and the final error measures convergence alone. The start
and goal configurations are offsets from a bent-elbow posture. The iiwa14
zero configuration is fully stretched, where the position Jacobian is
singular and gradient descent stalls, so the example stays away from it.

.. code-block:: python

   B = 8
   q_nom = jnp.array([0.0, 0.6, 0.0, -1.2, 0.0, 0.9, 0.0], jnp.float32)
   joint = jnp.arange(nq, dtype=jnp.float32)
   item = jnp.arange(B, dtype=jnp.float32)[:, None]
   q0 = q_nom + 0.2 * jnp.sin(0.9 * joint + 0.3 * item)
   q_goal = q0 + 0.5 * jnp.sin(1.7 * joint + 0.8 * item + 1.0)
   p_goal = robot.end_effector_pose(q_goal)[:, :3]

``end_effector_pose`` returns ``[xyz, rpy]`` per end effector, so ``[:, :3]``
keeps the position.

Loss and gradient
-----------------

.. code-block:: python

   def loss(q):
       p = robot.end_effector_pose(q)[:, :3]
       return 0.5 * jnp.sum((p - p_goal) ** 2)

   grad = jax.grad(loss)

The loss sums over the batch. Each target's error depends only on its own
row of ``q``, so the gradient of the sum is the per-target gradient
:math:`J_k^\top (p_k - p^*_k)`, with :math:`J_k` the position Jacobian GRiD
supplies in the backward pass. One ``grad`` call differentiates all ``B``
problems.

The descent as one GPU program
------------------------------

.. code-block:: python

   lr = 1.5

   @jax.jit
   def descend(q, steps):
       return jax.lax.fori_loop(0, steps, lambda _, q: q - lr * grad(q), q)

   q = descend(q0, 300)
   err = jnp.linalg.norm(robot.end_effector_pose(q)[:, :3] - p_goal, axis=1)

Each iteration is one forward and one backward FFI call. Under ``jit`` and
``fori_loop`` the 300 iterations are a single XLA program and ``q`` never
leaves the device. The script runs the loop in chunks to print the error as
it falls, and asserts that the final error is below 1 mm for every target.

Run it
------

.. code-block:: shell

   .venv/bin/python examples/jax/ik_gradient_descent.py --steps 300 --batch 8

The result is deterministic: the inputs are fixed and the kernels sum in a
fixed order. The step size is the one tuning knob. The error decreases
monotonically up to about ``lr = 2`` for these postures and diverges beyond
that, as plain gradient descent does once the step exceeds the inverse of
the Jacobian's largest squared singular value. Momentum, a line search or a
Gauss-Newton step with ``robot.end_effector_pose_gradient`` all fit the same
pattern; the handle supplies the Jacobian either way.

The same structure extends to dynamics. ``robot.forward_dynamics`` and
``robot.inverse_dynamics`` carry analytic backward passes as well
(:doc:`python_wrappers`), so a short trajectory-optimisation loop replaces
``end_effector_pose`` with a rollout and the position error with a running
cost. ``bindings/examples/jax_gpu_resident.py`` shows the rollout under
``jax.lax.scan``.
