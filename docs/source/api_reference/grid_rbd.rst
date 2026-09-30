grid_rbd (Python bindings)
==========================

The ``grid-rbd`` package is the surface most users actually call: it turns a
URDF into a cached per-robot ``.so`` and hands back a batched handle on the
numpy, JAX, or PyTorch backend.

See :doc:`../user_guide/tutorials/backend_coverage` for a source-checked
method inventory and :doc:`../user_guide/tutorials/verified_inputs` for
CPU-tested packing and inertial-parameter examples.

.. code:: python

   from grid_rbd import register_robot

   h = register_robot("iiwa", "path/to/iiwa14.urdf")     # numpy handle
   tau = h.inverse_dynamics(q, qd, qdd)                  # (batch, nq) float32

   hj = register_robot("iiwa", "path/to/iiwa14.urdf", backend="jax")
   ht = register_robot("iiwa", "path/to/iiwa14.urdf", backend="torch")

See :doc:`../user_guide/tutorials/python_wrappers` for the guided tour and
``bindings/examples/AGENT_INTEGRATION_GUIDE.md`` for the agent-facing
lifecycle notes.

Registration
------------

.. autofunction:: grid_rbd.register_robot

.. autofunction:: grid_rbd.load_robot

The numpy handle
----------------

.. autoclass:: grid_rbd.RobotHandle
   :members:
   :undoc-members:
   :exclude-members: __init__, plant_step_hessian

Step Hessian
~~~~~~~~~~~~

.. py:method:: grid_rbd.RobotHandle.plant_step_hessian(x, u, dt, *, integrator_type="euler", gravity=-9.81)

   Return the explicit step Hessian with shape ``(B, 2*NV, 3*NV, 3*NV)``
   for ``x`` of shape ``(B, NQ+NV)`` and ``u`` of shape ``(B, NV)``.
   Input derivative axes are ``[dq; dqd; du]`` and output rows are
   position tangent followed by velocity. Euler and semi-implicit Euler
   are supported on fixed and floating bases; multi-stage RK Hessians are
   unsupported. This method is available on the NumPy handle, not the
   JAX/PyTorch handles. The generated artifact must include its ``fdsva_so``
   dependency. See :doc:`../user_guide/concepts/algorithms/integrators_and_plant`.

JAX and torch handles
---------------------

``JaxRobotHandle`` (``grid_rbd.jax``) and ``TorchRobotHandle``
(``grid_rbd.torch``) expose framework-native value, explicit derivative,
integrator and selected plant methods. They are not exact mirrors:
``fk_batched`` and ``plant_step_hessian`` are NumPy-only; ``capture`` is
PyTorch-only. Supported differentiable operations use ``custom_vjp`` /
``autograd.Function``; explicit Hessian outputs do not imply support for
nested higher-order autodiff. Their modules import ``jax`` / ``torch`` at load time, so they are
documented in the guided tour (:doc:`../user_guide/tutorials/python_wrappers`)
rather than autodoc'd here; the docstrings in
``bindings/grid_rbd/jax/__init__.py`` and ``bindings/grid_rbd/torch/__init__.py``
are the authoritative per-method reference.
