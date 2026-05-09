User Guide
==========

GRiD generates optimized CUDA C++ implementations of rigid-body dynamics,
kinematics, and analytical-gradient algorithms from URDF robot models. The
package combines three Python components:

* :doc:`URDFParser <tutorials/urdf_parser>` for reading robot models.
* :doc:`RBDReference <tutorials/python_algorithms>` for CPU reference
  algorithms.
* :doc:`GRiDCodeGenerator <tutorials/codegen>` for CUDA header generation.

.. figure:: imgs/GRiD.png
   :alt: GRiD library ecosystem
   :width: 80%
   :align: center

   A URDF model is parsed, checked against Python reference algorithms, and
   lowered into generated CUDA code that can be validated and benchmarked.

Quick Start
-----------

Install the package in a local virtual environment:

.. code-block:: bash

   bash base_install.sh
   source .venv/bin/activate

For developer tests, robot-description fixtures, CUDA validation, and local
docs builds:

.. code-block:: bash

   bash developer_install.sh

Generate CUDA code for a robot:

.. code-block:: bash

   grid-generate path/to/robot.urdf
   grid-generate path/to/robot.urdf -f        # floating base
   grid-generate path/to/robot.urdf -t ee_jnt # retained fixed-joint target

The default output is ``grid.cuh`` in the current directory.

Examples
--------

The ``examples/`` directory contains small scripts for common workflows:

* ``quickstart_iiwa14.py`` generates a fixed-base iiwa14 CUDA header.
* ``quickstart_go2_floating.py`` generates a floating-base Go2 dynamics header.
* ``print_reference_values.py`` prints Python reference outputs for a URDF.
* ``print_grid.py`` generates, compiles, and runs the CUDA print executable.

Typical commands:

.. code-block:: bash

   .venv/bin/python examples/quickstart_iiwa14.py --output /tmp/grid_iiwa14.cuh
   .venv/bin/python examples/quickstart_go2_floating.py --output /tmp/grid_go2.cuh
   .venv/bin/python examples/print_reference_values.py path/to/robot.urdf
   GRID_CUDA_ARCH=86 .venv/bin/python examples/print_grid.py path/to/robot.urdf

Validation And Performance
--------------------------

Use the staged CUDA checker when changing generated CUDA behavior:

.. code-block:: bash

   .venv/bin/python test/cuda_equivalents/run_staged_cuda_checks.py

See :doc:`tutorials/cuda_validation` for CUDA equivalence tests, artifact
caching, shared-memory fallback controls, L2 persisting options, and
performance-reporting commands.

Current Support
---------------

GRiD supports revolute, prismatic, and fixed-joint robot models without closed
kinematic loops.

Implemented CUDA algorithm families include:

* Inverse dynamics / RNEA.
* Direct inverse mass matrix.
* Forward dynamics via Minv + RNEA.
* ABA and CRBA.
* Inverse- and forward-dynamics gradients.
* End-effector pose, gradient, and Hessian.
* Fixed-base second-order diagnostics for IDSVA-SO/FDSVA-SO.

See :doc:`tutorials/cuda_support_status` for the current fixed/floating support
matrix and known caveats.

Citing GRiD
-----------

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
   :maxdepth: 2
   :caption: User Guide

   getting_started/installation
   getting_started/library_overview
   getting_started/docker_setup

.. toctree::
   :maxdepth: 3
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 3
   :caption: Concepts

   concepts/index
