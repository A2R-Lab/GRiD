How do I…?
==========

The task router: find the right entry point for what you are trying to do.
(The same table lives in the repo README for people arriving via GitHub.)

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - I want to…
     - Start here
   * - **Call GRiD from Python** (numpy / JAX / torch)
     - ``grid_rbd.load_robot("robot.urdf", backend=...)`` — one call, no name
       ceremony. Guided tour: :doc:`user_guide/tutorials/python_wrappers`;
       API: :doc:`api_reference/grid_rbd`; agent-facing lifecycle notes:
       ``bindings/examples/AGENT_INTEGRATION_GUIDE.md``.
   * - **Generate CUDA for a new robot**
     - ``grid-generate path/to/robot.urdf [-f]`` (ten sample URDFs in
       ``config/robot_assets/``); walkthrough:
       :doc:`user_guide/tutorials/codegen`.
   * - **Make a big (humanoid) robot build fit in RAM / finish faster**
     - :doc:`user_guide/getting_started/fast_robot_setup` — subset the build
       with ``algorithm_list=`` and/or skip the mjx twins with
       ``enable_mujoco_kernels=False`` (also ``grid-generate
       --algorithm-list ... --no-mujoco-kernels``).
   * - **Use my own top-level GLASS instead of the copy vendored in grid.cuh**
     - ``gen_all_code(..., vendor_glass=False)``: the header ``#include``\ s
       ``glass.cuh`` from your include path (``-I<GLASS root>``) and aliases
       ``grid::glass`` to ``::glass`` — one GLASS per translation unit. The
       default (vendored, self-contained) header is byte-identical. All
       ``*_DYNAMIC_SHARED_MEM_BYTES<T[, TIER]>()`` sizers are ``constexpr``.
   * - **Label the GLASS revision when generating from a source archive (no .git)**
     - ``gen_all_code(..., glass_revision="<sha>")`` or
       ``GRID_GLASS_REVISION=<sha>``: the ``// Pinned commit:`` line carries
       the bare revision, so the header is byte-identical to a git checkout's;
       a live checkout that disagrees is an error, and
       ``gen.glass_revision_source`` reports ``git`` / ``git-verified`` /
       ``supplied-unverified`` / ``unknown``.
   * - **Embed the generated header in a library or interpreter (no exit() on CUDA errors)**
     - ``init_robotModel_checked`` / ``init_joint_limits_checked`` /
       ``free_robotModel_checked`` (or ``robotModel_owner<T>``):
       :doc:`user_guide/concepts/library_safe_initialization`.
   * - **Add a new algorithm to GRiD**
     - :doc:`user_guide/tutorials/adding_an_algorithm` (numpy oracle in
       ``RBDReference`` first, then the codegen emitter, then equivalence
       tests).
   * - **Run / verify the test suites, or fix a red receipt-verify CI job**
     - :doc:`user_guide/tutorials/cuda_validation` — the marker map, the
       split-suite driver, and the two-tier ``gpu-proof.json`` receipt policy
       (a red verify job after touching fingerprinted tests is BY DESIGN; run
       the everyday refresh and commit the receipt).
   * - **Benchmark GRiD (or compare against Pinocchio / MJX / Warp)**
     - :doc:`user_guide/tutorials/benchmarks`.
   * - **Debug a CUDA-vs-numpy mismatch or a weird kernel failure**
     - ``docs/agent_debugging_guide.md`` — the accumulated bug-class bible
       (shared-memory init, output-convention traps, reduction
       nondeterminism, launch-config pitfalls, …).
   * - **Get MuJoCo/mjx-convention inputs & outputs**
     - ``handle.mujoco.<method>(...)`` (per-call, thread-safe) or
       ``output_convention="mujoco"`` — values AND derivative/second-order
       surfaces, floating base; see the conventions section of
       :doc:`user_guide/tutorials/python_wrappers`.
   * - **Understand why something recompiled (or refused to)**
     - Three DIFFERENT "cache keys" exist: (1) the per-robot ``.so`` cache key
       (URDF bytes + codegen options + version + arch — ``register_robot``);
       (2) the test suites' content-keyed nvcc compile caches (header/source
       bytes — byte-identical codegen edits never rebuild); (3) the receipt
       fingerprints over ``test/cuda_equivalents`` + ``test/python_wrappers``
       (what makes shards stale). Details:
       :doc:`user_guide/getting_started/fast_robot_setup` and
       :doc:`user_guide/tutorials/cuda_validation`.
   * - **Tune kernel launch configs for my GPU**
     - ``config/autotune_robot.sh --help`` (writes
       ``config/launch_configs/<robot>/<gpu>.json``, baked at codegen time,
       overlay-able at runtime via ``apply_profile_overlay``).
