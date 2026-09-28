API Reference
=============

GRiD exposes the following Python interfaces:

* :doc:`grid_rbd <grid_rbd>` is the Python package users actually call:
  ``register_robot(...)`` → cached per-robot ``.so`` → numpy / JAX / torch
  handles.
* :doc:`RBDReference <rbd>` contains CPU reference rigid-body dynamics and
  kinematics algorithms used for validation.
* :doc:`URDFParser <urdf>` parses robot descriptions into the internal model
  consumed by the reference algorithms and code generator.
* :doc:`GRiD's code generator <gridcodegen>` emits CUDA C++ headers, host wrappers,
  shared-memory layouts, and generated helper APIs.

Start with :doc:`grid_rbd` to call GRiD from Python, :doc:`gridcodegen` when you want to generate CUDA, :doc:`rbd` when
you want Python reference values, and :doc:`urdf` when you need to inspect or
debug parsed robot topology.

.. toctree::
   :maxdepth: 2

   grid_rbd
   rbd
   urdf
   gridcodegen
