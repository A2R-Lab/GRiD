GRiD's code generator
=====================

The in-tree ``grid_codegen`` package emits robot-specific CUDA C++ headers
and wrapper entry points. Start with the :doc:`generation tutorial
<../user_guide/tutorials/codegen>` for runnable examples.

Python entry points
-------------------

``from grid_codegen import GRiDCodeGenerator`` imports the generator class.
Construct it with the model returned by ``URDFParser().parse(...)``, then
call ``gen_all_code(...)``. Common options are:

* ``output_path``: destination header (default ``grid.cuh``).
* ``codegen_profile`` / ``algorithm_list``: select algorithms to emit.
* ``enable_mujoco_kernels``: include or omit MuJoCo-convention twins.
* ``runtime_inertia``, ``runtime_transform`` and ``runtime_joint_dynamics``:
  enable the corresponding runtime model updates.
* ``collision_spec``: provide collision geometry and tier configuration.
* ``vendor_glass``: embed GLASS (default) or include an external copy.

The constructor also accepts ``dtype`` (``"float"`` by default) and
``FILE_NAMESPACE`` (``"grid"`` by default). The full signatures live in
``grid_codegen/GRiDCodeGenerator.py``. See :doc:`../user_guide/concepts/codegen_architecture`
for emission profiles and dependencies.

Generated CUDA layers
---------------------

Algorithms generally expose four layers; signatures and scratch requirements
vary by operation and resource tier:

* ``*_inner``: device computation with caller-provided operands and scratch.
* ``*_device``: device entry point that arranges the algorithm's working set.
* ``*_kernel``: batched CUDA kernel operating on device buffers.
* Host entry points: launches, synchronization and, where selected, transfers.

Consult the generated header and :doc:`input/output ABI
<../user_guide/concepts/input_output_abi>` before allocating buffers. Public
Python shapes are not the same as the native packed buffer layouts.
:doc:`../user_guide/concepts/resource_tier_system` describes shared-memory
and global-workspace requirements.

Extending and validating the generator
------------------------------------------

Algorithm emitters live under ``grid_codegen/algorithms/``; shared emission
helpers live under ``grid_codegen/helpers/``. Use the
:doc:`algorithm contribution guide <../user_guide/tutorials/adding_an_algorithm>`
for the current emitter conventions and tests.

CPU reference values come from :doc:`RBDReference <rbd>`, not generator
``test_*`` methods. GPU validation is described in
:doc:`../user_guide/tutorials/cuda_validation`.
