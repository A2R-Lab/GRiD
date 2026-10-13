Generate CUDA for a robot
==============================

GRiD's code generator (``grid_codegen/`` in this repository) produces a
robot-specific ``grid.cuh`` header; generating the header does not compile or
run CUDA. The ``grid-generate`` CLI ships in the ``grid-rbd`` wheel, so
``pip install grid-rbd`` (:doc:`installation <../getting_started/installation>`)
is enough; only a source checkout additionally needs its Git submodules
populated.

Generate a header
-----------------

From the repository root (the sample URDFs under ``config/robot_assets/`` are
in the checkout, not the wheel — substitute your own URDF path):

.. code-block:: shell

   grid-generate config/robot_assets/iiwa14.urdf --algorithm-list dynamics -o /tmp/iiwa_grid.cuh

``--algorithm-list`` restricts emission to named algorithms or profiles.
Omit it to use the full profile. Add ``-f`` for a floating base; use
``--no-mujoco-kernels`` to omit the additional MuJoCo-convention kernels.
For collision geometry, see :doc:`collisions`.

Generate from Python
--------------------

.. code-block:: python

   from URDFParser import URDFParser
   from grid_codegen import GRiDCodeGenerator

   robot = URDFParser().parse("config/robot_assets/iiwa14.urdf")
   generator = GRiDCodeGenerator(robot)
   generator.gen_all_code(
       algorithm_list=["dynamics"],
       output_path="/tmp/iiwa_grid.cuh",
   )

``GRiDCodeGenerator`` is the Python class name, not a separate repository
or installable module. GRiD's code generator vendors GLASS device-side
linear algebra into the header by default. Set ``vendor_glass=False`` when
using an external ``glass.cuh`` on your C++ include path.

Use the generated code
----------------------

* For Python applications, :doc:`python_wrappers` handles generation,
  compilation and caching through ``grid_rbd.load_robot(...)``.
* For CUDA applications, include the header in a CUDA C++ translation unit.
  See :doc:`../../api_reference/gridcodegen` for the entry-point layers and
  :doc:`../concepts/library_safe_initialization` for allocation and cleanup.
* Use :doc:`../getting_started/fast_robot_setup` to reduce large-robot build
  time and memory, and :doc:`cuda_validation` to check generated results.
