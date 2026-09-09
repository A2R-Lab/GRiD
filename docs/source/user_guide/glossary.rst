Glossary
========

The terms of art this repo uses — several are overloaded, and two different
things are both called a "tier". When a doc and this page disagree, fix the
doc.

.. glossary::

   resource tier
      The shared-memory budget a kernel is compiled for: ``TIER_SHARED``
      (everything in smem), then the spill rungs (``TIER_LITE`` /
      ``TIER_MINIMAL``) that move progressively more scratch to the global
      ``d_workspace``. Selected per algorithm per robot at codegen time
      (:doc:`concepts/resource_tier_system`), dispatched via the
      ``RESOURCE_TIER`` template parameter. **This is what "tier" means
      unqualified.**

   precompile variants
      The OTHER historical use of "tier": ``grid_rbd.precompile(...,
      tiers=...)`` builds several pre-compiled ``.so`` option-variants of one
      robot. Unrelated to resource tiers — prefer calling these *variants*.

   arena
      A kernel's flat shared-memory scratch pool (``s_temp``), laid out at
      codegen time with compile-time offsets. The per-tier
      ``*_DYNAMIC_SHARED_MEM_BYTES`` macros size it.

   spill rung
      One step of a tier's shrink ladder: a specific buffer (or group)
      rerouted from the smem arena to ``d_workspace``. Rungs are applied in a
      fixed order until the arena fits the target.

   ``_inner`` / ``_device`` / ``_kernel`` / host
      The four emission layers per algorithm — placement-free computational
      core; the ``__device__`` orchestrator owning scratch placement; the
      ``__global__`` batch entry point; the H↔D-copying host wrapper. See
      :doc:`concepts/codegen_architecture`.

   mjx / twin / pin
      *pin* = Pinocchio-convention (GRiD-native) kernels and I/O. *mjx* =
      MuJoCo-convention (wxyz quaternion, global-linear free-joint velocity).
      A *twin* is the mjx variant of a kernel (``MUJOCO_OUTPUT=true``
      instantiation), emitted only for floating-base non-mimic/skew robots.

   receipt / shard / carry
      The signed ``gpu-proof.json`` at the repo root attests GPU test
      outcomes so CPU-only CI can gate merges. A *shard* is one
      crash-isolated slice of the split suite with its own fingerprint; a
      *carried* shard is one whose files are unchanged since an ancestor
      commit's green run and is reused rather than re-executed (everyday
      policy only — a release receipt refuses carries).

   fingerprint
      The content hash over ``test/cuda_equivalents/`` +
      ``test/python_wrappers/`` files that decides whether a receipt shard is
      stale. Codegen/bindings edits do NOT stale shards (their correctness is
      re-proven by the content-keyed compile caches + the next full pass).

   GLASS
      The first-party GPU linear-algebra header library
      (``external/GLASS``). Extend it rather than working around it.

   ffi_bases
      The per-algo ``{tier, threads}`` block of a robot's tuned launch config
      (``config/launch_configs/<robot>/<gpu>.json``) used by the jax/torch
      FFI launch path; ``pybind_bases`` / ``torch_bases`` are runtime
      overlays over the same bake.

   batch-to-land
      Bench metric: wall time for a batch INCLUDING the device→host copy of
      the result (what a consumer actually waits for), vs. kernel-only time.

   mimic joint
      A URDF joint slaved to another via ``<mimic>`` (``q_child = m·q_parent
      + o``). Collapses ``nv`` below the joint count and makes many
      per-joint kernels take the reduced path; mimic robots never get mjx
      twins.

   subset build / ``algorithm_list``
      Building only named algorithms (plus transitive deps) into a robot's
      ``.so``. Un-built methods raise the clean "not built into this robot
      .so — add to algorithm_list" error (C-ABI rc=3).
