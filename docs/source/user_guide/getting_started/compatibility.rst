Compatibility and known limitations
===================================

Platforms and toolchain
-----------------------

* **One artifact per GPU architecture.** A robot ``.so`` is compiled for the
  ``sm_XX`` of the GPU that registers it (or ``cuda_arch=``); loading it on a
  different architecture is refused at context creation with a clear error.
  The model, the API and the generated code carry over between machines; the
  binary does not.
* **Tested deployment platform:** Linux x86_64 with CUDA 12+ (the release
  receipt was produced on an RTX 5090). Jetson-class devices are supported by
  design (single-block kernels, resource tiers, runtime memory fitting) but are
  not certified by this receipt.
* **Host compiler:** C++17 for the code generator, benchmarks and a numpy-only
  build. With the ``[torch]`` extra the wrapper follows torch's ATen
  requirement — ``-std=c++20`` from torch 2.14 on — so a torch-enabled build
  needs an nvcc and host compiler accepting C++20. ``GRID_RBD_CXX_STD`` forces
  the standard.
* **Backends:** the ``[jax]`` / ``[torch]`` extras pin the CPU packages; install
  the CUDA wheel yourself (``pip install "jax[cuda12]"`` or ``"jax[cuda13]"``, a
  ``cu1xx`` torch). A CPU-only jax is refused at handle construction with that
  hint. See :doc:`installation` for the PyPI install and its extras.

Runtime contexts
----------------

Every native call resolves a **runtime context** by id; ``handle.context()``
opens an isolated one (own arena, tables, streams, launch overrides) and a
handle is the one owner of its context. Runtime-parameter mutations give the
context a new ``model_version``, and torch/JAX backwards refuse a forward that
ran under a different version. **Limitation:** concurrent asynchronous calls on
ONE context share its scratch buffers — one ordered pipeline per context, so
use a context per pipeline; execution is not fully asynchronous. Ownership,
``workspace_slots``, versioning and the synchronisation points are specified in
:doc:`../concepts/runtime_contexts`.

Captured CUDA graphs (torch)
----------------------------

* ``handle.capture(...)`` replays only on its context at the captured model
  epoch: a replay after a mutation is refused (recapture) and a replay after the
  context was closed is refused (never a launch into freed memory). Replays of
  one graph are serialized; ``static_out`` is overwritten by the next replay.
* Capturing a backward is unsupported. Do not mix graph replay and concurrent
  eager calls on the same context.

Operands
--------

One rule set per operand class is enforced natively on all three surfaces
(``1 <= B <= max_batch``, exact last dim, every operand carrying the leading
batch). Configuration inputs have width ``nq``; velocities, accelerations and
generalized forces have width ``nv`` — floating and spherical quaternions make
these widths different. What each surface coerces, refuses or elides is
specified in :doc:`../concepts/operand_validation`; layouts are in
:doc:`../concepts/input_output_abi`.

Native interface stability
--------------------------

* The generated ``grid.cuh`` keeps its inline entry points
  (``init_gridData_checked`` gained an optional trailing allocator-pool
  argument; the pool-less allocator spellings remain).
* The per-robot wrapper ``.so``'s ``extern "C" grid_rbd_*`` symbols are
  **private to the package**: they are consumed only by the pybind core of the
  same content-keyed build and gained a leading ``ctx_id`` in 0.5.0, the first
  packaged release. Do
  not link against them directly.

Differentiation
---------------

* First-order autograd (torch ``backward``, ``jax.grad``/``vjp``) runs the
  analytic gradient kernels; higher-order differentiation through those
  kernels is not provided. ``*_wrt_params`` is the local sensitivity of the
  compiled model to its inertial parameters (a parameters argument does not
  install a new runtime model).
* Second-order tensors are values (``idsva_so`` / ``fdsva_so``), not
  autograd-composable operators. Per-method and per-joint-type restrictions
  are collected on :doc:`../tutorials/backend_coverage`.

Build cost
----------

* A full-feature humanoid build is expensive (see the cold/warm table on the
  fast-robot-setup page); subset builds (``algorithm_list=``,
  ``enable_mujoco_kernels=False``) are the practical path when compile time or
  memory matters. Generated-source portability is not binary portability.
