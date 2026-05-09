GRiD: GPU-Accelerated Rigid Body Dynamics Code Generation
==========================================================

GRiD turns URDF robot models into optimized CUDA C++ for rigid-body dynamics,
kinematics, analytical gradients, and validation against Python reference
implementations. It supports fixed- and floating-base robots, shared-memory
fallback paths for larger generated kernels, and benchmark tooling for checking
performance on real GPU targets.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Quick Start
      :link: user_guide/landing_page
      :link-type: doc

      Install GRiD, generate your first CUDA header, and run the core examples.

   .. grid-item-card:: API Reference
      :link: api_reference/index
      :link-type: doc

      Browse the Python APIs for URDF parsing, reference algorithms, and CUDA
      code generation.

   .. grid-item-card:: CUDA Validation
      :link: user_guide/tutorials/cuda_validation
      :link-type: doc

      Run staged fixed/floating correctness checks, shared-memory fallback
      tests, and performance reports.

   .. grid-item-card:: Benchmarks
      :link: user_guide/tutorials/benchmarks
      :link-type: doc

      Learn where benchmark scripts live and how to compare generated kernels
      on target GPU hardware.

.. figure:: user_guide/imgs/benchmark_multi_fd_grad.png
   :alt: GRiD forward-dynamics gradient benchmark performance
   :width: 85%
   :align: center

   Example GRiD benchmark results for batched forward-dynamics gradient
   computation. Use the benchmark and performance-reporting tools to collect
   current numbers on your robot and GPU.

How To Customize This Site
--------------------------

* Homepage text and top-level navigation live in ``docs/source/index.rst``.
* User-facing install, examples, and support notes live under
  ``docs/source/user_guide/``.
* API documentation entry points live under ``docs/source/api_reference/``.
* The A2R Lab logo is configured in ``docs/source/conf.py`` through
  ``html_theme_options["logo"]["image_light"]`` and
  ``html_theme_options["logo"]["image_dark"]``. Replace
  ``docs/source/_static/a2r_lab.jpg`` to update the current logo image.
* Page figures belong in ``docs/source/user_guide/imgs/``; theme images,
  favicon files, and CSS belong in ``docs/source/_static/``.
* Local style overrides live in ``docs/source/_static/custom.css``.

Build And Preview
-----------------

.. code-block:: bash

   .venv/bin/python -m sphinx -W --keep-going -b html docs/source docs/build/html
   .venv/bin/python -m http.server -d docs/build/html 8000

Then open ``http://localhost:8000``.

.. toctree::
   :maxdepth: 3
   :hidden:

   user_guide/landing_page
   api_reference/index
   contribution_guidelines
   sphinx_edit_guide
   faq

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
