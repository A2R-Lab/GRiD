GRiD: GPU-Accelerated Rigid Body Dynamics with Analytical Gradients
===================================================================

GRiD turns URDF robot models into optimized CUDA C++ for rigid-body dynamics,
kinematics, analytical gradients, and validation against Python reference
implementations. It supports fixed- and floating-base robots, shared-memory
fallback paths for larger generated kernels, and benchmark tooling for checking
performance on real GPU targets.

Pick how you want to call GRiD
------------------------------

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: From Python
      :link: user_guide/tutorials/python_wrappers
      :link-type: doc

      ``pip install -e .`` and call the ``RobotHandle`` API — 20+ batched
      methods covering RNEA, FD/ABA, CRBA, Minv, EE pose family, RNEA/FD
      gradients, and second-order ID/FD.

   .. grid-item-card:: From JAX
      :link: jax-ffi-quickstart
      :link-type: ref

      ``pip install -e ".[jax]"`` for a device-resident, ``jax.jit``-compatible
      FFI surface. Same per-robot ``.so`` cache as the plain Python wrapper.

   .. grid-item-card:: From raw CUDA
      :link: user_guide/tutorials/codegen
      :link-type: doc

      Generate a per-robot ``grid.cuh`` with the ``grid-generate`` CLI or the
      ``GRiDCodeGenerator`` Python API, then ``#include`` it in your project.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: API Reference
      :link: api_reference/index
      :link-type: doc

      Browse the Python APIs for URDF parsing, reference algorithms, and CUDA
      code generation.

   .. grid-item-card:: Benchmarks
      :link: user_guide/tutorials/benchmarks
      :link-type: doc

      Where the bench scripts live and how to compare generated kernels on
      target GPU hardware.

.. figure:: user_guide/imgs/benchmark_multi_fd_grad.png
   :alt: GRiD forward-dynamics gradient benchmark performance
   :width: 85%
   :align: center

   Example GRiD benchmark results for batched forward-dynamics gradient
   computation. Use the benchmark and performance-reporting tools to collect
   current numbers on your robot and GPU.

Citation
--------

If you use GRiD in your research, please cite:

.. code-block:: text

   @inproceedings{plancher2022grid,
     title={GRiD: GPU-Accelerated Rigid Body Dynamics with Analytical Gradients},
     author={Brian Plancher and Sabrina M. Neuman and Radhika Ghosal and Scott Kuindersma and Vijay Janapa Reddi},
     booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
     year={2022},
     month={May}
   }

.. toctree::
   :hidden:
   :caption: User Guide

   user_guide/landing_page
   user_guide/getting_started/installation
   user_guide/getting_started/library_overview
   user_guide/getting_started/docker_setup

.. toctree::
   :hidden:
   :caption: Tutorials

   user_guide/tutorials/index

.. toctree::
   :hidden:
   :caption: Concepts

   user_guide/concepts/index

.. toctree::
   :hidden:
   :caption: API Reference

   api_reference/index

.. toctree::
   :hidden:
   :caption: Project info

   contribution_guidelines
   sphinx_edit_guide
