Adding a New Algorithm
======================

This is a practical, end-to-end guide for adding a new rigid-body
dynamics algorithm to GRiD. It walks through the canonical pattern using
``fdsva_so`` (second-order forward dynamics) as the worked example.
Pair it with the conceptual docs:

* :doc:`../concepts/design_principles` — *why* the codegen looks the way
  it does (smart inners, thin wrappers, inner-owns-placement, the spill
  ladder).
* :doc:`../concepts/codegen_architecture` — the three emission layers
  (``_host`` / ``_kernel`` / ``_device``) and their composition contract.
* :doc:`../concepts/resource_tier_system` — per-tier dispatch and
  selective spill.

If you internalize those three documents and follow the recipe below,
your new algorithm will plug into the existing tier-spill / equivalence /
bench infrastructure with no special-casing required.

The shape of the work
---------------------

Every algorithm ``X`` ships as one Python file at
``grid_codegen/algorithms/_X.py`` exposing a set of ``gen_*``
emitter functions. The functions are:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Generator function
     - Emits / returns
   * - ``gen_X_inner_temp_mem_size``
     - Python int: number of ``T`` scratch slots needed by the
       single-step computational core (``_inner``). Read by every
       caller that sizes scratch arenas.
   * - ``gen_X_inner_function_call``
     - A string-emitting helper that callers (other algorithms,
       composing kernels) use to invoke ``X_inner`` with the right
       argument list. Hides the helper-pointer plumbing.
   * - ``gen_X_inner``
     - Emits the placement-free C++ ``__device__ X_inner(...)``. This is
       the math. Inputs are already in shared memory; takes ``s_temp``
       (already placed by the caller). Templated on placement flags
       (``SCRATCH_IN_SMEM`` and any surgical sub-flags).
   * - ``gen_X_device``
     - Emits the canonical ``__device__ X_device(...)``. Caller-supplied
       buffers + the placement flags. **Owns scratch placement** via the
       top-of-body ``if constexpr (!SCRATCH_IN_SMEM) { s_temp = d_workspace; }``
       repoint. Calls any sub-step ``_inner`` helpers; for composing
       algorithms also calls other algorithms' ``_inner``.
   * - ``gen_X_kernel``
     - Emits ``__global__ X_kernel(...)``. Allocates ``__shared__`` smem
       from the per-tier ``X_DYNAMIC_SHARED_MEM_BYTES<T, TIER>()``
       macro, runs a grid-stride loop over timesteps, loads inputs,
       calls ``X_device``, saves outputs.
   * - ``gen_X_host``
     - Emits the host launcher ``__host__ X(...)``: wraps ``X_kernel``
       with H↔D copies for inputs and outputs.
   * - ``gen_X``
     - Runs all of the above generators in the right order. The
       top-level driver ``GRiDCodeGenerator.gen_all_code`` calls this
       for every registered algorithm.

Plus, you'll add one ``AlgoDescriptor`` row to ``algo_registry.py`` — the
descriptor table is the single source of truth for per-algo metadata, and
that one row drives the ``GridAlgo`` enum, the launch-config symbol map, and
the ``KERNEL_ATTR_MANIFEST`` / mjx manifest heads (previously these were
scattered hand-maintained dicts). You'll also add a
``gen_X_inner_temp_mem_size`` reference to ``GRiDCodeGenerator.py`` (top-level
imports + per-robot scratch tier selection). See
:doc:`../concepts/codegen_architecture` for the descriptor table.

Step-by-step recipe (worked example: ``fdsva_so``)
--------------------------------------------------

#. **Pick the canonical references**

   The Python reference for the algorithm should already live in our
   ``RBDReference`` submodule (or get added there first; see the
   ``RBDReference`` README). For ``fdsva_so`` that is
   ``RBDReference.fdsva_so(...)`` returning ``(d2tau_dqdq, d2tau_dvdv,
   d2tau_dvdq, dM_dq)``. The CUDA equivalence test compares the
   generated GPU output against this reference numerically.

#. **Lay out the inner's scratch budget**

   Decide what intermediate values you need and how big each is. For
   ``fdsva_so``, the rank-3 contraction sub-step needs a ``4·nv³``
   working buffer. Write the size-getter:

   .. code:: python

      def gen_fdsva_so_contract_temp_mem_size(self):
          return 4 * self.robot.get_num_vel() ** 3

   The orchestrator (``fdsva_so_device``) also embeds ``minv``
   and ``forward_dynamics`` and the IDSVA-SO sub-algorithms. Sum their
   smem footprints (or take the ``max`` for arenas that get reused
   across phases) to size the outer scratch arena.

#. **Write the inner**

   ``gen_fdsva_so_contract`` (the rank-3 contraction sub-step) emits a
   block-cooperative loop over the ``nv²`` output positions, using
   ``glass::gemv`` / ``dot_prod`` via the ``grid_linalg_*`` wrappers and
   the codegen's parallel-loop helper:

   .. code:: python

      def gen_fdsva_so_contract(self, use_thread_group=False):
          n = self.robot.get_num_vel()
          # ... function signature emit ...
          self.gen_add_code_line(
              "template <typename T, bool SCRATCH_IN_SMEM = true>"
          )
          self.gen_add_code_line("__device__")
          self.gen_add_code_line(func_def, True)

          # Inner-owns-placement: the FIRST line of the body.
          self.gen_add_code_line(
              "if constexpr (!SCRATCH_IN_SMEM) { s_temp = d_workspace; }"
              " else { (void)d_workspace; }"
          )
          # ... emit the actual loop using gen_add_parallel_loop /
          # grid_linalg_gemv / dot_prod / etc.
          self.gen_add_end_function()

   The conventions:

   - Inputs that already live in shared memory keep the ``s_`` prefix.
   - Globals that get spilled use the ``d_`` prefix.
   - Templated on ``SCRATCH_IN_SMEM = true`` by default so small
     robots' tiers are byte-identical; the first body line is the
     ``if constexpr`` repoint.
   - Use the existing block-cooperative GLASS primitives
     (``glass::gemv``, ``glass::gemm``, ``glass::invertMatrix_dense``,
     ``glass::cholDecomp_InPlace``, ``glass::trsm``) rather than rolling
     your own. The codegen helpers (``grid_linalg_gemm`` etc.) wrap
     them with a consistent signature.

#. **Write the orchestrator** (``gen_X_device``)

   For a composing algorithm like ``fdsva_so`` that builds on
   ``minv``, ``forward_dynamics``, and the IDSVA-SO sub-inners,
   the device wrapper does ONE ``XImats`` load and then calls each
   sub-algorithm's placement-free ``_inner`` with the SAME ``s_temp``:

   .. code:: python

      # In gen_fdsva_so_device:
      self.gen_add_code_line(
          "if constexpr (!SCRATCH_IN_SMEM) { s_temp = d_workspace; }"
          " else { (void)d_workspace; }"
      )
      self.gen_load_update_XImats_helpers_function_call(use_thread_group)
      self.gen_minv_inner_function_call(use_thread_group, f_in_smem_expr="true")
      self.gen_add_code_line(
          "forward_dynamics_inner<T, true>(s_qdd, s_q, s_qd, s_u, "
          + self.gen_insert_helpers_function_call()
          + "s_temp, nullptr, gravity);"
      )
      # ... IDSVA-SO sub-inners ...
      self.gen_fdsva_so_contract_function_call(use_thread_group)

   The XImats load is paid ONCE here; every sub-inner gets it for free.
   That's the entire reason the ``_inner`` / ``_device`` split exists —
   see :doc:`../concepts/codegen_architecture`.

#. **Write the kernel** (``gen_X_kernel``)

   The kernel is mechanical: allocate ``__shared__`` smem from the
   per-tier ``*_DYNAMIC_SHARED_MEM_BYTES`` macro, run a grid-stride
   loop over timesteps, load inputs to smem, call ``X_device``, store
   outputs:

   .. code:: cuda

      template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
      __global__ void fdsva_so_kernel(/*...*/) {
          __shared__ T s_temp[fdsva_so_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>() / sizeof(T)];
          // ... allocate s_inputs, s_outputs ...
          for (int k = blockIdx.x; k < NUM_TIMESTEPS; k += gridDim.x) {
              // ... load inputs ...
              fdsva_so_device<T, ...>(/* placement flags chosen per RESOURCE_TIER */);
              // ... save outputs ...
          }
      }

   The kernel **never** repoints ``s_temp`` itself — the device owns
   that. The kernel's only job is to size the arena, load/save IO, and
   pass the per-tier placement flags as template arguments.

#. **Register the tier picks**

   Tell ``GRiDCodeGenerator.py`` how to choose a spill rung per tier
   for THIS robot:

   .. code:: python

      # Inside GRiDCodeGenerator.py, in the per-robot tier table build:
      fdsva_so_inner_idsva_so_temp_count = ...   # sub-algorithm scratch
      fdsva_so_contract_temp_count = self.gen_fdsva_so_contract_temp_mem_size()

      # 3-way selection: PERF / LITE / MINIMAL → choose a rung (0..N-1).
      rung_arenas = [...]   # list of (rung_name, bytes_needed, flag_tuple)
      self.fdsva_so_spill_tier_3way = self.select_shared_tier_3way(*rung_arenas)

   ``select_shared_tier_3way`` picks the lowest-byte rung that fits the
   per-tier target. ``PERF`` falls through to the most-spilled rung if
   nothing fits — that's the whole point: big robots spill at PERF too.

#. **Add the host wrapper + descriptor-table row**

   ``gen_X_host`` mirrors any other host wrapper. Add one
   ``AlgoDescriptor`` row (and its ``ALGO_REGISTRY`` entry) in
   ``grid_codegen/algo_registry.py``: the descriptor row carries the
   algorithm's irregular metadata (autotune keys, ``gate_attr``,
   ``bytes_macro`` overrides) and drives the ``GridAlgo`` enum, the
   launch-config symbol map, and the kernel-attr / mjx manifests from a
   single source, while the registry wires the algorithm into the bench,
   equivalence runner, and per-algo TUs. The
   ``test/test_algo_descriptor_parity.py`` net locks the table to the
   generated output.

#. **Add a CUDA equivalence test**

   For a small robot (start with iiwa14), regenerate the header and
   run:

   .. code:: shell

      PATH="/usr/local/cuda/bin:.../bin:$PATH" \
      PYTHONPATH=$REPO \
      GRID_CUDA_RANDOM_SAMPLES=2 \
      pytest -x -q \
        test/cuda_equivalents/test_cuda_executable_equivalence.py \
        -k iiwa14-fixed

   The runner compares your kernel's output against ``RBDReference.X``
   (the Python truth) for several sample states. Match it to
   ``rtol = 2e-4`` for float32 at PERF.

   Also exercise a forced-spill tier to validate the spill path:

   .. code:: shell

      GRID_CUDA_TARGET_SHARED_MEM_BYTES=30000 pytest ...

   PERF validates the math; a forced deep spill validates that the
   spilled rung is byte-identical (only the pointer moves) — otherwise
   that rung is shipped untested and will bite later.

#. **Cross-check shape conventions**

   GRiD outputs are typically column-major (Fortran-order) on
   ``s_temp`` / ``s_M`` / ``s_d2eePos`` etc. The CUDA equivalence
   runner has shape-aware comparators in
   ``test/cuda_equivalents/test_cuda_executable_equivalence.py``; if
   your algorithm has a non-standard output shape, add it there. The
   pre-existing ``end_effector_pose_hessian`` shape is ``6 * NUM_VEL² * NUM_EE``
   per timestep (row-major in the EE / column / row axes), for
   example.

Common pitfalls
---------------

* **Caller-side ``s_temp`` repoint or alias.** Placement is the inner's
  job. A kernel that reassigns ``s_temp`` from outside is the bug class
  the inner-owns-placement design was built to prevent. (See the
  anti-patterns list at the bottom of :doc:`../concepts/design_principles`.)
* **Null ``s_temp`` to a load helper.** In whole-arena spill rungs the
  smem ``s_temp`` slot is ``nullptr``; the ``XImats`` / ``XmatsHom``
  load helper dereferences it for sincos scratch. Always repoint
  ``s_temp`` to ``d_workspace`` BEFORE the first helper call.
* **Single-valued PERF-pick macro as a per-rung flag.** Macros like
  ``GRID_X_USES_SPILL`` equal the *PERF* pick. Using one as the inner's
  template arg under ``if constexpr (RESOURCE_TIER == ...)`` gives the
  non-PERF tier the wrong flag → it tries to write the full band into a
  selective-sized arena → smem OOB on big robots. Pass the actual
  per-rung value as a literal.
* **Single-thread Gauss-Jordan inverse.** This was a real performance
  hotspot in ``aba`` / ``minv`` floating-base root invert. Use
  ``glass::invertMatrix_dense`` (block-cooperative). If your algorithm
  needs an inverse / factor, reach for GLASS first.
* **Spill rung 0..N−1 only relocates code — must be numerically
  identical to the unspilled path.** The only difference is where a
  pointer points. If your spilled tier produces different numbers,
  you've got a real bug.
* **Hardcoded assumptions about block size.** Use the
  ``gen_add_parallel_loop`` helper, which emits a block-stride loop —
  any block size that fits is correct. Don't assume ``blockDim.x ==
  MAX_PERF_LEVEL_THREADS``.

Code-generation helpers (cheat sheet)
-------------------------------------

Most useful helpers (in ``grid_codegen/helpers/``):

* ``gen_add_code_line(line)`` / ``gen_add_code_lines([...])`` — emit
  text into the current function.
* ``gen_add_parallel_loop(var, max_val, use_thread_group=False,
  block_level=False)`` — emit a block-stride
  ``for (i = tid; i < max_val; i += blockDim...)``.
* ``gen_add_sync(use_thread_group=False)`` — emit ``__syncthreads()``.
* ``gen_add_serial_ops(use_thread_group=False)`` — wrap the next block
  in an ``if (threadIdx.x == 0 && threadIdx.y == 0)`` (use sparingly —
  this is the bottleneck class to avoid; see *Single-thread Gauss-
  Jordan inverse* above).
* ``gen_add_func_doc(description, notes, params, return_val)`` — emit a
  Doxygen-style header.
* ``gen_kernel_load_inputs(name, stride, amount, ...)`` and
  ``gen_kernel_save_result(name, stride, amount, ...)`` — boilerplate
  for global ↔ shared memory transfer in the kernel.
* ``grid_linalg_gemm<T, M, N, K>`` / ``grid_linalg_gemv<T, M, N>`` —
  thin wrappers around ``glass::gemm`` / ``glass::gemv``. Always
  prefer these over rolling your own loops.

When you're done
----------------

Open a PR against the codegen submodule with:

#. The new ``_X.py`` algorithm file.
#. The ``algo_registry.py`` entry.
#. Any top-level ``GRiDCodeGenerator.py`` edits (imports, tier selection).
#. A CUDA equivalence test that exercises iiwa14 fixed and floating at
   PERF and at a forced spilled tier.
#. A bench entry in ``run.py`` if you want the algorithm timed in the
   standard sweep.
#. Documentation updates (algorithm description in
   ``docs/source/user_guide/concepts/algorithms/``).

The reviewers will mostly look at: does the inner own its placement?
Does the kernel size match the per-tier macro? Does the CUDA test pass
at PERF *and* at a forced spilled tier? If all three are green, the
algorithm is on the spill ladder for free and tier-aware composing
kernels can build on it directly.
