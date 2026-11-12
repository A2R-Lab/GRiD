Compatibility and known limitations
===================================

Platforms and toolchain
-----------------------

* There is one artifact per GPU architecture unless ``GRID_RBD_CUDA_ARCHS`` names
  several and the build is a fat binary. A robot ``.so`` is otherwise compiled for
  the ``sm_XX`` of the GPU that registers it (or ``cuda_arch=``); loading it on a
  different architecture is refused at context creation with a clear error.
  The model, the API and the generated code carry over between machines; the
  binary does not.
* The 0.7.0 primary GPU test receipt and the release timing ran on Linux x86-64
  with CUDA toolkit 13.2 and one RTX 5090 (sm_120). A secondary receipt of the
  same 2056 tests at the same commit ran on a Jetson AGX Orin (sm_87, aarch64,
  CUDA toolkit 13.2); the table below lists both environments. CUDA 12
  toolkits and other GPUs are untested. The Xavier (sm_72) ran the header-only
  example and has no receipt.
* The host compiler standard is C++17 for the code generator, benchmarks and a
  numpy-only build. With the ``[torch]`` extra the wrapper reads the standard
  that torch's ATen header guard demands and compiles with it. The torch we run
  (2.12.0, cu128) compiles with C++17. A torch that demands C++20 needs an nvcc
  and host compiler accepting C++20; that path ran under the Orin receipt below
  (torch 2.14.0+cu130).
  ``GRID_RBD_CXX_STD`` forces the standard.
* The ``[jax]`` / ``[torch]`` extras pin the CPU packages; install the CUDA
  wheel yourself. We run ``jax[cuda12]`` (``jax-cuda12-plugin`` 0.10.0) and a
  cu128 torch wheel; ``jax[cuda13]`` and other torch wheels are untested. A
  CPU-only jax is refused at handle construction with that hint. See
  :doc:`installation` for the PyPI install and its extras.

What we ran
-----------

The RTX 5090 produced the primary evidence for 0.7.0: the full test
receipt (``gpu-proof.json``, verified by CI), the release timing on the
:doc:`release measurements <../../release_measurements>` page, and the
installed-wheel check of the release playbook. The Jetson AGX Orin recorded
the secondary receipt ``receipts/gpu-proof-sm87.json``, the same collection
at the same commit, which ``make verify-all`` checks against its own
expected-skips baseline. Everything validated ran in
fp32, the default compute precision; the fp64 tier (``dtype="float64"``) is
an alpha feature outside the receipt.

The table below records that historical qualification, not a qualification of
changed 0.7.1 code. The 0.7.1 candidate has additional local correctness and
installed-artifact gates; its fresh full primary receipt is a separate release
gate. The Orin receipt remains tied to the 0.7.0 source and does not attest the
new armature, allocation or free-body repairs. No new Orin qualification is
claimed. Strict verification of a historical receipt against changed inputs
may fail; preserve that evidence and verify it at its recorded source revision,
rather than weakening policy or relabeling it as current.

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
     - receipt 2026-11-01; timing 2026-09-27 to 2026-10-07; install check
       2026-10-15
   * - sm_87 (compute capability 8.7)
     - NVIDIA Jetson AGX Orin, 64 GB
     - 13.2 (V13.2.78)
     - L4T R39.2 (JetPack; driver 595.78)
     - CPython 3.12.3
     - 2.14.0+cu130 (NVIDIA Jetson build)
     - jax 0.11.1 with jax-cuda13-plugin 0.11.1 (NVIDIA Jetson build)
     - Ubuntu 24.04.4, Linux 6.8.12-1021-tegra, aarch64
     - secondary receipt ``receipts/gpu-proof-sm87.json`` at the 0.7.0
       receipt commit 607efc2: 2056 tests, 2055 passed, 1 skipped (the
       compute-sanitizer racecheck cell listed in
       ``test/gpu-proof-expected-skips-sm87.txt``), 82 shards in 24 hours
       with ``GRID_SPLIT_COMPILE_JOBS=2`` and ``GRID_RBD_BUILD_JOBS=4``;
       pinocchio 3.9.0 and mujoco 3.9.0 from PyPI aarch64 wheels
     - receipt 2026-11-01 to 2026-11-02
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
615.71.09 does. On x86-64, a ``jax[cuda13]`` plugin, a torch wheel other
than cu128, a CUDA 12 toolkit and CPython 3.10, 3.11 or 3.13 on a GPU are
untested; the CI wheel builds for CPython 3.10, 3.11 and 3.13 and the CPU test
lane on 3.10 and 3.13 run on CPU only. The Orin receipt is the one run of
``jax-cuda13-plugin`` and a cu130 torch wheel, on aarch64. The Orin row is a
secondary receipt, recorded through the per-architecture table in
``test/gpu-proof-receipts.yaml`` and verified by ``make verify-all``; it
documents the architecture and does not gate CI. The Xavier stays header-only
because JetPack 5 ships CUDA 11.4 and Python 3.8, below the floors of the
generator and the bindings.

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

* Public generated surfaces are the ones documented on the API reference
  pages and the CUDA tutorials: the kernel and ``*_device`` entry points named
  in :doc:`../tutorials/cuda_support_status`, the ``grid::model`` metadata and
  transform helpers of :doc:`../../api_reference/model_coordinates`, the
  configuration primitives of :doc:`../../api_reference/integration_conventions`
  and the ``grid_plant`` workspace views of
  :doc:`../../api_reference/workspace_composition`. Neither an underscore nor
  an ``inline`` spelling decides this: a generated function absent from those
  pages is an implementation detail that a later release may rename or
  remove. (``init_gridData_checked`` gained an optional trailing allocator-pool
  argument; the pool-less allocator spellings remain.)
* Release policy during 0.x: any release may change a documented public
  contract, and a release that does so says so in the changelog with the
  migration step, as :doc:`migrating_to_0_7` does for this one. Correctness
  fixes name the affected operations, the numerical change and any rebuild
  they require. The version number records what shipped; it does not promise
  which kind of change a release may carry, a deprecation period, a support
  window or a response time. What is promised is that a public contract never
  changes silently.
* Source compatibility, descriptor schema compatibility, compiled-artifact
  reuse and tested numerical support are separate promises. A call that still
  compiles does not imply that an old prepared artifact loads: artifact reuse
  is decided by the content, toolchain and descriptor checks described on the
  model-coordinates page, and a schema change requires one rebuild.
* The per-robot wrapper ``.so``'s ``extern "C" grid_rbd_*`` symbols are
  private to the package: they are consumed only by the pybind core of the
  same content-keyed build and gained a leading ``ctx_id`` in 0.5.0, the first
  packaged release. Do not link against them directly.
* The two joint-transform helpers in ``grid::model`` are CUDA device
  functions; host code reads the same placements through the constexpr
  ``joint`` and ``frame`` record accessors.

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

Sanitizer diagnostics with JAX
------------------------------

* Running a process that initializes the JAX CUDA backend under NVIDIA Compute
  Sanitizer reports one CUDA API warning during JAX's device-allocator query
  (``cuDeviceGetAttribute``: fabric handle support requires IMEX channels) and,
  for a standalone JAX reduction, a selective device-code recompilation notice.
  Both come from JAX 0.11 and the CUDA 13 toolchain and reproduce in a process
  that imports no GRiD; the numerical results are correct and the NumPy and
  PyTorch surfaces pass the same sanitizer with zero errors. The first is the
  driver's own behaviour: ``cuDeviceGetAttribute`` reports fabric-handle
  support on a machine that has no IMEX channel node (the driver creates one
  only with ``NVreg_CreateImexChannel0``), so XLA's allocator probe takes the
  fabric path and the API call it makes is refused; the allocator then falls
  back and runs correctly. Nothing in GRiD selects a handle type. We record this as
  an accepted limitation of the combined sanitizer invocation rather than a
  defect of the wrapper: the sanitizer exit code of a JAX-initialized process is
  nonzero, and the release checks state it so instead of suppressing the
  warning.
