Compatibility and known limitations
===================================

Platforms and toolchain
-----------------------

* There is one artifact per GPU architecture. A robot ``.so`` is compiled for the
  ``sm_XX`` of the GPU that registers it (or ``cuda_arch=``); loading it on a
  different architecture is refused at context creation with a clear error.
  The model, the API and the generated code carry over between machines; the
  binary does not.
* The GPU test receipt and the release timing ran on Linux x86-64 with CUDA
  toolkit 13.2 and one RTX 5090 (sm_120); the table below lists that
  environment. CUDA 12 toolkits and other GPUs are untested. Jetson-class
  devices are within the design (single-block kernels, resource tiers, runtime
  memory fitting) but no receipt covers them.
* The host compiler standard is C++17 for the code generator, benchmarks and a
  numpy-only build. With the ``[torch]`` extra the wrapper reads the standard
  that torch's ATen header guard demands and compiles with it. The torch we run
  (2.12.0, cu128) compiles with C++17. A torch that demands C++20 needs an nvcc
  and host compiler accepting C++20, and that path is untested here.
  ``GRID_RBD_CXX_STD`` forces the standard.
* The ``[jax]`` / ``[torch]`` extras pin the CPU packages; install the CUDA
  wheel yourself. We run ``jax[cuda12]`` (``jax-cuda12-plugin`` 0.10.0) and a
  cu128 torch wheel; ``jax[cuda13]`` and other torch wheels are untested. A
  CPU-only jax is refused at handle construction with that hint. See
  :doc:`installation` for the PyPI install and its extras.

What we ran
-----------

One machine produced every piece of GPU evidence in this release: the full
test receipt (``gpu-proof.json``, verified by CI), the release timing on the
:doc:`release measurements <../../release_measurements>` page, and the
installed-wheel check of the release playbook.

.. list-table::
   :header-rows: 1
   :widths: 9 9 9 9 7 12 11 10 12 12

   * - GPU architecture
     - GPU
     - CUDA toolkit (nvcc)
     - Driver
     - Python
     - torch wheel
     - jax plugin
     - OS
     - What ran
     - Date
   * - sm_120 (compute capability 12.0)
     - NVIDIA GeForce RTX 5090, 32 GB
     - 13.2 (V13.2.86)
     - 615.71.09
     - CPython 3.12.3
     - 2.12.0.dev20260408+cu128 (CUDA 12.8 runtime)
     - jax 0.10.0 with jax-cuda12-plugin 0.10.0
     - Ubuntu 24.04.5, Linux 7.0.0-38, x86-64
     - full receipt; release timing; install check (a fresh venv with
       torch 2.11.0+cu128 and jax 0.11.2 from PyPI)
     - receipt 2026-10-15; timing 2026-09-27 to 2026-10-07; install check
       2026-10-15

The toolkit and the framework wheels are separate things. ``nvcc`` 13.2
compiles the generated kernels and the per-robot ``.so``. The torch wheel
carries its own CUDA 12.8 runtime libraries for torch's kernels, and the
``jax-cuda12-plugin`` carries CUDA 12 runtime libraries for XLA; neither uses
``nvcc``. The driver has to satisfy every runtime in the process, and
615.71.09 does. A ``jax[cuda13]`` plugin, a torch wheel other than cu128, a
CUDA 12 toolkit and CPython 3.10 or 3.11 on a GPU are untested; the CI wheel
builds for CPython 3.10 and 3.11 run on CPU only. Jetson Orin (sm_87) and
Xavier (sm_72) rows are planned and will be filled from real runs.

Runtime contexts
----------------

Every native call resolves a runtime context by id; ``handle.context()``
opens an isolated one (own arena, tables, streams, launch overrides) and a
handle is the one owner of its context. Runtime-parameter mutations give the
context a new ``model_version``, and torch/JAX backwards refuse a forward that
ran under a different version. Concurrent asynchronous calls on one context
share its scratch buffers, so each context carries one ordered pipeline; use a
context per pipeline, as execution is not fully asynchronous. Ownership,
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
generalized forces have width ``nv``, and floating and spherical quaternions
make these widths different. What each surface coerces, refuses or elides is
specified in :doc:`../concepts/operand_validation`; layouts are in
:doc:`../concepts/input_output_abi`.

Native interface stability
--------------------------

* The generated ``grid.cuh`` keeps its inline entry points
  (``init_gridData_checked`` gained an optional trailing allocator-pool
  argument; the pool-less allocator spellings remain).
* The per-robot wrapper ``.so``'s ``extern "C" grid_rbd_*`` symbols are
  private to the package: they are consumed only by the pybind core of the
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
