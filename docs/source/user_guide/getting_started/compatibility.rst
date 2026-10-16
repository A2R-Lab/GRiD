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
  memory fitting); the Orin ran the smoke set listed below and the Xavier the
  header-only example, but no receipt covers them yet.
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
installed-wheel check of the release playbook. Everything validated ran in
fp32, the default compute precision; the fp64 tier (``dtype="float64"``) is
an alpha feature outside the receipt.

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
   * - sm_87 (compute capability 8.7)
     - NVIDIA Jetson AGX Orin, 64 GB
     - 13.2 (V13.2.78)
     - L4T R39.2 (JetPack; driver API 13.2)
     - CPython 3.12.3
     - 2.14.0+cu130 (NVIDIA Jetson build)
     - jax 0.11.1 (NVIDIA Jetson build)
     - Ubuntu 24.04.4, Linux 6.8.12-tegra, aarch64
     - smoke runs on the 0.5.1 main checkout, no receipt: the iiwa14 numpy,
       torch and jax wrappers (30, 25 and 60 tests), the CUDA equivalence
       harness on iiwa14 fixed and go2 floating (72 tests), the g1 plant
       Hessian; pinocchio 3.9.0 and mujoco 3.9.0 from PyPI aarch64 wheels
     - 2026-10-16
   * - sm_72 (compute capability 7.2)
     - NVIDIA Jetson AGX Xavier, 32 GB
     - 11.4 (V11.4.315)
     - L4T R35.3.1 (JetPack 5)
     - CPython 3.8.10 and 3.9 only; grid-rbd needs 3.10
     - not applicable
     - not applicable
     - Ubuntu 20.04, Linux 5.10.104-tegra, aarch64
     - header-only CUDA: the fused control step example, built against an
       iiwa14 header generated on the x86-64 machine, compiled with nvcc 11.4
       for sm_72 and validated fused against layered; the generator, the
       bindings and the test suite do not install on this Python
     - 2026-10-16

The toolkit and the framework wheels are separate things. ``nvcc`` 13.2
compiles the generated kernels and the per-robot ``.so``. The torch wheel
carries its own CUDA 12.8 runtime libraries for torch's kernels, and the
``jax-cuda12-plugin`` carries CUDA 12 runtime libraries for XLA; neither uses
``nvcc``. The driver has to satisfy every runtime in the process, and
615.71.09 does. A ``jax[cuda13]`` plugin, a torch wheel other than cu128, a
CUDA 12 toolkit and CPython 3.10, 3.11 or 3.13 on a GPU are untested; the CI
wheel builds for CPython 3.10, 3.11 and 3.13 and the CPU test lane on 3.10 and
3.13 run on CPU only. The Jetson rows record smoke runs, not receipts. The
Orin receipt is recorded under ``receipts/`` once the per-architecture receipt
table is on main; the Xavier stays header-only because JetPack 5 ships CUDA
11.4 and Python 3.8, below the floors of the generator and the bindings.

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
