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

* ``output_path``: destination header (default ``FILE_NAMESPACE + ".cuh"``,
  i.e. ``grid.cuh`` for the default namespace).
* ``codegen_profile`` / ``algorithm_list``: select algorithms to emit by
  name or by profile; the profile names are tabulated on the
  :doc:`generation tutorial <../user_guide/tutorials/codegen>`.
* ``enable_mujoco_kernels``: include or omit MuJoCo-convention twins.
* ``runtime_inertia``, ``runtime_transform`` and ``runtime_joint_dynamics``:
  enable the corresponding runtime model updates.
* ``collision_spec``: provide collision geometry and tier configuration.
* ``vendor_glass``: embed GLASS (default) or include an external copy.

The constructor also accepts ``dtype`` (``"float"`` by default),
``FILE_NAMESPACE`` (``"grid"`` by default), ``USE_JOINT_DYNAMICS`` (apply the
URDF damping and friction in the dynamics and their derivatives, off by
default), ``FRICTION_MODEL`` (``"sign"`` by default, or ``"tanh"`` for the
smooth Coulomb term ``friction*tanh(qd/w)`` with its first and second
derivatives) and ``FRICTION_WIDTH`` (the tanh width ``w``, 0.03 rad/s by
default, one scalar or one value per velocity slot). The full signatures live in
``grid_codegen/GRiDCodeGenerator.py``. See :doc:`../user_guide/concepts/codegen_architecture`
for emission profiles and dependencies.

Generated CUDA layers
---------------------

Every algorithm is emitted in three layers (the
:doc:`codegen architecture <../user_guide/concepts/codegen_architecture>`
page has the mechanics); signatures and scratch requirements vary by
operation and resource tier:

* ``*_device``: the ``__device__`` entry point that arranges the algorithm's
  working set.
* ``*_kernel``: the batched ``__global__`` kernel operating on device buffers.
* the unsuffixed host function (e.g. ``grid::inverse_dynamics<T, ...>``):
  launches, synchronization and, where selected, transfers.

``*_inner`` helpers (device computation on caller-placed operands and
scratch) exist underneath ``*_device`` for composition; they are callable
but are not a separate public layer.

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
