grid_rbd (Python bindings)
==========================

The ``grid-rbd`` package is the surface most users actually call: it turns a
URDF into a cached per-robot ``.so`` and hands back a batched handle on the
numpy, JAX, or PyTorch backend.

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
   :exclude-members: __init__

JAX and torch handles
---------------------

``JaxRobotHandle`` (``grid_rbd.jax``) and ``TorchRobotHandle``
(``grid_rbd.torch``) mirror the numpy surface — the same value, gradient,
second-order, integrator, and plant methods with framework-native arrays,
``custom_vjp`` / ``autograd.Function`` wiring, and per-call MuJoCo-convention
views. Their modules import ``jax`` / ``torch`` at load time, so they are
documented in the guided tour (:doc:`../user_guide/tutorials/python_wrappers`)
rather than autodoc'd here; the docstrings in
``bindings/grid_rbd/jax/__init__.py`` and ``bindings/grid_rbd/torch/__init__.py``
are the authoritative per-method reference.
