Fuse dynamics into your own kernel
==================================

The header-only entry point exists so that GRiD's dynamics can run inside a
kernel you own, next to your own control or optimisation math, without a
launch or a host round trip between the stages. This page builds one such
kernel: a computed-torque control step over a batch of joint states, with
GRiD's inverse dynamics called from inside it. The same step is then computed
through the generated host function, one stage at a time, so the two paths can
be checked against each other and timed. The complete source is
``examples/cuda/fused_control_step/`` in the repository: ``fused_control_step.cu``
and ``build_and_time.sh``.

The control step
----------------

One control step for a batch of ``B`` states of a fixed-base arm has three
stages. Stage 1 forms the commanded acceleration from the desired trajectory
and a PD term on the tracking error. Stage 2 is inverse dynamics, the torque
that realises the commanded acceleration. Stage 3 predicts the next state with
one Euler step at the commanded acceleration, which is the state the arm
reaches when the model is exact.

.. math::

   \ddot q_{cmd} &= \ddot q_{des} + k_p (q_{des} - q) + k_d (\dot q_{des} - \dot q) \\
   \tau &= \mathrm{ID}(q, \dot q, \ddot q_{cmd}) \\
   q_{next} &= q + \Delta t\, \dot q, \qquad \dot q_{next} = \dot q + \Delta t\, \ddot q_{cmd}

Stage 2 is GRiD's RNEA. Stages 1 and 3 are elementwise and belong to the
application. In a layered design each stage is a separate launch or a host
loop; in the fused design all three run in one block on shared-memory
operands.

Generate the header
-------------------

The example needs only ``inverse_dynamics``, so it generates a header
restricted to that algorithm (``examples/cuda/gen_iiwa14_header.py``, which
``build_and_time.sh`` runs for you):

.. code-block:: python

   from URDFParser import URDFParser
   from grid_codegen import GRiDCodeGenerator

   robot = URDFParser().parse("grid_assets/robots/iiwa14.urdf", floating_base=False)
   GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(
       algorithm_list=["inverse_dynamics"], output_path="grid.cuh")

The ``grid-generate`` CLI produces the same header outside a checkout
(:doc:`codegen`). The header is self-contained; ``nvcc`` is the only tool the
build needs.

Write the kernel
----------------

Stages 1 and 3 are block-cooperative helpers. Every thread handles a strided
subset of the joints, which is the same loop shape GRiD's own kernels use, so
the helpers are correct at any block size:

.. code-block:: cuda

   template <typename T>
   __device__ __forceinline__ void pd_acceleration(
       T *s_qdd_cmd, const T *s_q, const T *s_qd, const T *s_q_des,
       const T *s_qd_des, const T *s_qdd_des, const T kp, const T kd)
   {
       for (int i = threadIdx.x; i < grid::NUM_JOINTS; i += blockDim.x)
           s_qdd_cmd[i] = s_qdd_des[i] + kp * (s_q_des[i] - s_q[i]) + kd * (s_qd_des[i] - s_qd[i]);
   }

   template <typename T>
   __device__ __forceinline__ void euler_step(
       T *s_q_next, T *s_qd_next, const T *s_q, const T *s_qd, const T *s_qdd_cmd, const T dt)
   {
       for (int i = threadIdx.x; i < grid::NUM_JOINTS; i += blockDim.x) {
           s_q_next[i]  = s_q[i]  + dt * s_qd[i];
           s_qd_next[i] = s_qd[i] + dt * s_qdd_cmd[i];
       }
   }

The kernel follows GRiD's one-block-per-problem layout: block ``k`` handles
state ``k``, with a grid-stride loop so the kernel is correct when the batch
exceeds the grid. Inside the loop the three stages run back to back on
shared-memory vectors, separated by ``__syncthreads()``. Stage 2 is the
generated ``grid::inverse_dynamics_device<T>``, which takes shared-memory
inputs and outputs, allocates its own scratch from the dynamic shared-memory
arena, and returns with the torques in ``s_tau``:

.. code-block:: cuda

   template <typename T>
   __global__ void fused_control_step_kernel(
       T *d_tau, T *d_q_next, T *d_qd_next,
       const T *d_q, const T *d_qd, const T *d_q_des, const T *d_qd_des, const T *d_qdd_des,
       const int B, const grid::robotModel<T> *d_robotModel,
       const T gravity, const T kp, const T kd, const T dt)
   {
       constexpr int n = grid::NUM_JOINTS;
       __shared__ T s_q[n], s_qd[n], s_q_des[n], s_qd_des[n], s_qdd_des[n];
       __shared__ T s_qdd_cmd[n], s_tau[n], s_q_next[n], s_qd_next[n];

       for (int k = blockIdx.x; k < B; k += gridDim.x) {
           for (int i = threadIdx.x; i < n; i += blockDim.x) {
               s_q[i] = d_q[k * n + i];           s_qd[i] = d_qd[k * n + i];
               s_q_des[i] = d_q_des[k * n + i];   s_qd_des[i] = d_qd_des[k * n + i];
               s_qdd_des[i] = d_qdd_des[k * n + i];
           }
           __syncthreads();

           pd_acceleration<T>(s_qdd_cmd, s_q, s_qd, s_q_des, s_qd_des, s_qdd_des, kp, kd);
           __syncthreads();

           grid::inverse_dynamics_device<T>(
               s_tau, s_q, s_qd, s_qdd_cmd, d_robotModel, /*d_f_ext=*/nullptr, gravity);
           __syncthreads();

           euler_step<T>(s_q_next, s_qd_next, s_q, s_qd, s_qdd_cmd, dt);
           __syncthreads();

           for (int i = threadIdx.x; i < n; i += blockDim.x) {
               d_tau[k * n + i] = s_tau[i];
               d_q_next[k * n + i] = s_q_next[i];
               d_qd_next[k * n + i] = s_qd_next[i];
           }
           __syncthreads();
       }
   }

The commanded acceleration never leaves shared memory: stage 1 writes
``s_qdd_cmd``, stage 2 reads it as the RNEA's ``qdd`` input, and stage 3 reads
it again. Only the three results are written to global memory, and they stay
on the device for the next step of whatever pipeline owns the kernel.

Initialise and launch
---------------------

Before the first launch the host creates the header's streams and registers
the dynamic shared-memory attribute of the one generated kernel this program
launches (the host-layer path below). The per-algorithm twin is the right
spelling: a translation unit compiles every kernel it takes the address of,
and ``grid::init_grid<T>()`` takes the address of all of them. The fused
kernel is not in the header, so its arena is registered by hand, sized by the
``_device`` routine's macro:

.. code-block:: cuda

   cudaStream_t *streams = grid::init_grid_streams<T>();
   gpuErrchk(grid::init_grid_kernel_attr_inverse_dynamics_checked<T>());
   grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();

   const size_t smem = grid::INVERSE_DYNAMICS_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>();
   gpuErrchk(cudaFuncSetAttribute(fused_control_step_kernel<T>,
       cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));

   fused_control_step_kernel<T><<<B, threads, smem>>>(
       d_tau, d_q_next, d_qd_next, d_q, d_qd, d_q_des, d_qd_des, d_qdd_des,
       B, d_robotModel, gravity, kp, kd, dt);
   gpuErrchk(cudaPeekAtLastError());

One launch covers the whole batch. ``threads`` is capped at
``grid::MAX_PERF_LEVEL_THREADS``; the example defaults to 128 and accepts
``--threads``.

The layered path
----------------

The same step through the generated host function keeps GRiD's layers intact.
``grid::inverse_dynamics<T, USE_QDD_FLAG, USE_COMPRESSED_MEM>`` reads its
inputs from the host side of a ``gridData``, copies them to the device,
launches the generated ``inverse_dynamics_kernel``, synchronises, and copies
the torques back into ``hd_data->h_c``. Stages 1 and 3 run as host loops
around it:

.. code-block:: cuda

   // stage 1 on the host, packed into the host function's inputs
   for (int k = 0; k < B; ++k)
       for (int i = 0; i < n; ++i) {
           qdd_cmd[k * n + i] = qdd_des[k * n + i] + kp * (q_des[k * n + i] - q[k * n + i])
                              + kd * (qd_des[k * n + i] - qd[k * n + i]);
           hd_data->h_q_qd[k * 2 * n + i]     = q[k * n + i];
           hd_data->h_q_qd[k * 2 * n + n + i] = qd[k * n + i];
           hd_data->h_qdd[k * n + i] = qdd_cmd[k * n + i];
       }
   // stage 2: host-to-device copies, the generated kernel, device-to-host copy
   grid::inverse_dynamics<T, /*USE_QDD_FLAG=*/true, /*USE_COMPRESSED_MEM=*/true>(
       hd_data, d_robotModel, gravity, B, dim3(B), dim3(threads), streams);
   // stage 3 on the host
   for (int e = 0; e < B * n; ++e) {
       tau[e] = hd_data->h_c[e];
       q_next[e] = q[e] + dt * qd[e];
       qd_next[e] = qd[e] + dt * qdd_cmd[e];
   }

``USE_QDD_FLAG`` selects the RNEA variant that takes an acceleration input,
and ``USE_COMPRESSED_MEM`` the ``[q | qd]`` packing of ``h_q_qd`` with the
acceleration in its own ``h_qdd`` slot. The ``gridData`` for ``B`` states
comes from ``grid::init_gridData<T>(B)`` and is released by
``grid::close_grid<T>``.

Why fusion is faster
--------------------

The layered step crosses the host boundary twice per stage boundary. The host
function copies two input arrays in, launches one kernel, synchronises the
device after the copies, after the kernel and after the copy back, and the
host loops before and after it run at host speed on data that was on the
device a moment earlier. Each of those boundaries is a fixed cost that does
not shrink with the batch, so at the batch sizes GRiD targets (tens to
hundreds of states) they dominate a 7-DoF RNEA.

The fused step has one launch and no copies. The intermediate ``s_qdd_cmd``
exists only in shared memory, the inputs were already resident, and the
outputs stay resident. What remains is the kernel's own time, which is where
the batch parallelism applies. This is the payoff of the header-only
entry point, and the reason the generated ``_device`` routines take
shared-memory operands rather than device pointers: they are designed to be
called from inside a kernel that already holds the state.

Build, check and time
---------------------

From the repository root:

.. code-block:: shell

   bash examples/cuda/fused_control_step/build_and_time.sh --batch 256 --repeats 200

The script generates the header, compiles ``fused_control_step.cu`` with
``nvcc -arch=sm_XX -std=c++17 -O2`` (the architecture from ``GRID_CUDA_ARCH``,
else ``nvidia-smi``, else the generator's default; see
``grid_codegen/cuda_env.py``), and runs the binary. The binary first computes
the step both ways on the same deterministic batch and compares the three
outputs state by state; the worst relative error must be below 1e-4 (both
paths run the same single-precision RNEA, so the observed error is at the
rounding level) or the process exits non-zero. It then times each path as the
median of ``--repeats`` cudaEvent-timed runs after a warm-up. The output has
the form:

.. code-block:: text

   correctness: fused vs layered over 256 states, worst rel_err tau=... q_next=... qd_next=... (tol 1e-04) OK
   timing: batch=256 threads=128 repeats=200  fused=... us  layered=... us  (medians; sm_120)

``--json PATH`` writes the same numbers as
``{"example": "fused_control_step", "batch": N, "fused_us": x, "layered_us": y,
"repeats": R, "arch": "sm_XX"}``, which is the record the release measurements
collect. ``--header PATH`` builds against an existing header and skips the
Python generation step.

The measured fused-to-layered ratio on our RTX 5090 is reported on
:doc:`../../release_measurements` once it has been collected on a quiet GPU.
A number taken on a shared or busy device is not comparable to it.

Adapting the kernel
-------------------

A different robot changes the URDF in the generation step and nothing in the
kernel, because every size comes from the header's constants. A floating base
changes the input widths (``q`` is ``NUM_POS`` wide, velocities and torques
``NUM_VEL`` wide) and the load helpers; ``examples/cuda/README.md`` points at
the floating-base pattern. A different algorithm changes the ``_device`` call
and the arena macro, which share the algorithm's name
(``grid::<ALGO>_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>()``); grep the generated
header for the exact signature, since buffer order differs by algorithm. To
share one ``XImats`` load between several algorithms in the same kernel, use
the ``_inner`` layer with caller-placed scratch, described in
``examples/cuda/wrapper_types.md`` and
:doc:`../concepts/codegen_architecture`.
