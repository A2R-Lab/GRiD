Compatibility and known limitations
===================================

What this release supports, what changed for existing users, and what it does
not do. Everything here is tested at the release tip; the receipt in
``gpu-proof.json`` is the GPU evidence and the CPU-only CI verifies it.

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
  hint. Editable-checkout installation only; no wheels are published.

Runtime contexts (new in this release)
--------------------------------------

* Every native call resolves a **runtime context** by id; ``handle.context()``
  gives an isolated one (own arena, tables, streams, launch overrides). A handle
  is the one owner of its context: dropped handles are finalized at garbage
  collection, ``close()`` is idempotent, views reference the handle.
* ``workspace_slots=N`` on a context is an explicit cap (``0`` = auto-fit;
  explicit cap > ``GRID_WORKSPACE_TIMESTEP_SLOTS`` > auto-fit).
* Runtime-parameter mutations (inertias, transforms, joint dynamics, tool
  attach/detach) are serialized against every in-flight call and give the
  context a new ``model_version`` (an artifact-wide epoch). torch and JAX
  backwards refuse a forward that ran under a different version, including a
  forward that ran on a since-recreated default context.
* **Limitation:** concurrent asynchronous calls on ONE context share its scratch
  buffers — one ordered pipeline per context; use a context per pipeline. Setter
  and numpy fences and the per-backward stamp check remain synchronisation
  points; execution is not fully asynchronous.

Captured CUDA graphs (torch)
----------------------------

* ``handle.capture(...)`` replays only on its context at the captured model
  epoch: a replay after a mutation is refused (recapture) and a replay after the
  context was closed is refused (never a launch into freed memory). Replays of
  one graph are serialized; ``static_out`` is overwritten by the next replay.
* **Limitation:** capturing a backward is unsupported; the per-call lease and
  graph-reservation machinery that would let graphs and eager calls share a
  context safely is a later increment.

Operands
--------

* One rule set per operand class, enforced natively on all three surfaces
  (``1 <= B <= max_batch``, exact last dim, every operand carrying the leading
  batch, ``f_ext`` physically batched, runtime target ids in range). numpy
  coerces layout and dtype; torch refuses CPU, non-contiguous or wrong-dtype
  tensors; JAX checks in Python and again in every FFI handler.
* numpy accepts a per-sample 1D input (batch of one); JAX returns an empty
  result for an empty batch (XLA elides the call); the JAX methods materialize a
  supported ``f_ext`` broadcast before the call. Floating-base inputs are
  ``num_joints``-wide (an ``nv``-wide velocity is refused by name).

Native interface stability
--------------------------

* The generated ``grid.cuh`` keeps its inline entry points
  (``init_gridData_checked`` gained an optional trailing allocator-pool
  argument; the pool-less allocator spellings remain).
* The per-robot wrapper ``.so``'s ``extern "C" grid_rbd_*`` symbols are
  **private to the package**: they are consumed only by the pybind core of the
  same content-keyed build and gained a leading ``ctx_id`` in this release. Do
  not link against them directly.

Differentiation
---------------

* First-order autograd (torch ``backward``, ``jax.grad``/``vjp``) runs the
  analytic gradient kernels; higher-order differentiation through those
  kernels is not provided. ``*_wrt_params`` is the local sensitivity of the
  compiled model to its inertial parameters (a parameters argument does not
  install a new runtime model).
* Second-order tensors are values (``idsva_so`` / ``fdsva_so``), not
  autograd-composable operators.

Build cost
----------

* A full-feature humanoid build is expensive (see the cold/warm table on the
  fast-robot-setup page); subset builds (``algorithm_list=``,
  ``enable_mujoco_kernels=False``) are the practical path when compile time or
  memory matters. Generated-source portability is not binary portability.
