Resource-Tier Design Notes
==========================

Why the tier system looks the way it does — the rationale essays split out of
:doc:`resource_tier_system` (the reference page) on 2026-09-09.

Design philosophy: smart inners, thin wrappers
----------------------------------------------

The organizing principle of the generated code is:

  **Put all the intelligence in the inner functions. Make everything above
  them a thin convenience wrapper.**

Concretely, an emitted algorithm is four layers, and the value is concentrated
entirely in the bottom one:

``<algo>_inner`` — *the engine.*
  A ``__device__`` routine that does the actual rigid-body-dynamics math. It
  is written to use **as much block-wide parallelism as possible** (see
  below), and it is **smart about memory**: it owns the decision of what lives
  in shared memory vs. global memory, how scratch is laid out, what gets
  spilled under resource pressure, and what gets recomputed vs. cached. It
  takes the caller's input/output pointers plus a shared scratch arena
  (``s_temp``) and a global scratch arena (``d_workspace``), and decides
  internally — via a compile-time placement parameter — which buffers go
  where. Nothing above this layer needs to understand the algorithm's memory
  layout.

``<algo>_device`` — *convenience: "call the engine without thinking about arenas."*
  A ``__device__`` wrapper for inline-CUDA users who want a single call rather
  than managing the scratch arena themselves. It declares the shared-memory
  arena (sized for the default placement), loads/updates the per-configuration
  helper tables (``XImats`` etc.), and calls ``_inner``. Use it when you want
  to call a GRiD primitive from your kernel but don't need to micro-manage
  where its scratch lives.

``<algo>_kernel`` — *convenience: "a ready-to-launch batch entry point."*
  A ``__global__`` entry point that loops over a trajectory/batch of inputs,
  loads each timestep's inputs into shared memory, dispatches on
  ``RESOURCE_TIER``, and calls the engine. This is what you launch if you want
  GRiD to own the whole kernel. It is templated on ``<T, RESOURCE_TIER>`` and
  carries the ``__launch_bounds__`` for the tier.

``<algo>`` (host) — *convenience: "I never want to touch device code."*
  A ``__host__`` wrapper that does the host↔device memory transfers, chooses
  block/grid dimensions, sets the kernel's dynamic-shared-memory attribute,
  and launches the kernel. This is what the Python/JAX handles call under the
  hood, and what a C++ host-only user calls.

Why this shape? Because the audience that cares about performance is calling
``_inner`` (or ``_device``) and composing it into a larger fused kernel. For
that user, **the kernel and host layers are noise** — they want the raw
block-parallel routine and full control of the memory hierarchy. The
convenience layers exist so that the *other* 90% of users never have to see
any of it. Keeping the layers thin also means there is exactly one place where
the hard decisions live (the inner), so there is one place to audit, tune, and
get right.


Design choices
---------------

**Why not three completely independent bodies per tier?**
Numerical equivalence: the math is identical at every tier;
``if constexpr`` branches only differ in pointer routing
(s_temp vs d_workspace). One body per algo, with up to two
pointer-routing branches. Less code duplication, fewer drift bugs.

**How does the Python surface pick a tier?** (Earlier revisions of this
doc said Python was "locked to TIER_SHARED" — no longer true.) The Python
wrapper persona is still "sealed product, never touches nvcc": there is no
runtime tier knob. Instead, the tier is a PER-ALGO BAKED choice — the
autotuners write {tier, threads} into ``config/launch_configs/`` and the
codegen bakes it as ``grid::launch_cfg<GRID_ALGO_*>::TIER``, which every
binding launch site (numpy/pybind host-wrapper calls AND the jax/torch
direct kernel launches) instantiates. Divergent-tier instantiations get
their own dynamic-smem registration in ``init_grid_kernel_attrs`` (the B2
fix, 2026-09-05) — a distinct ``__global__`` per tier is the reason that
registration exists. Untuned robots/algos fall back to ``TIER_SHARED``
via the primary ``launch_cfg`` template, which preserves the old behavior.

**The LITE smem target between SHARED and MINIMAL — now landed.**
Early revisions of this design shipped without a distinct LITE smem
target: the per-algo multi-tier spill machinery (``fdsva_so`` 4 levels,
``end_effector_pose_hessian``/``inverse_dynamics_gradient``/
``forward_dynamics_gradient`` 3, ``idsva_so_body_frame`` 2) picked **one**
spill level at codegen time based on ``cuda_target_shared_mem_bytes``,
and LITE collapsed onto SHARED/MINIMAL. That follow-up has since landed:
codegen computes three picks per algo (``select_shared_tier_3way``
against the ~48 KB ``cuda_target_lite_shared_mem_bytes`` target), emits
per-tier ``if constexpr`` bodies where the picks diverge, and the
``gen_declare_shared_arena(tier_workspace_expr=...)`` mechanism in
``grid_codegen/helpers/_code_generation_helpers.py`` supports the
ternary picks. See "Humanoid-scale spill" and "LITE 48 KB smem target"
below for the shipped details.

How it relates to other v2.0 work
----------------------------------

* :doc:`cublasdx_removal_design` — v2.0 set the stage by removing
  cuBLASDx and adding ``set_threads_per_block`` (up to
  MAX_PERF_LEVEL_THREADS). The tier system extends this to **above**
  MAX_PERF_LEVEL_THREADS via TIER_MINIMAL's ``launch_bounds=1024``.
* :doc:`codegen_architecture` — describes the four-layer emission
  (``_inner`` / ``_device`` / ``_kernel`` / host); tier templates
  live at the ``_inner`` / ``_device`` / ``_kernel`` layers.

