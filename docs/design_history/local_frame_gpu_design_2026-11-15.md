# Local-frame second-order GPU design

Status: isolated correctness prototype. No production emitter, public ABI,
workspace-size contract, convention, support policy or timing result changes.

## Mathematical prerequisites

The complete CPU candidate now computes symbolic transforms in the selected
dtype, local CRBA and dM/dq, the RNEA bias, an equilibrated Cholesky acceleration,
local first and second ID derivatives at that acceleration, and stable solves
for gradients and second-order composition. It consumes only quantized q/qd/u
and parsed model constants. There are no oracle intermediates or transplanted
device errors. Separate float64 identities establish the convention before
float32 gates. The corrected sweep passes all 192 cases in both accumulation
orders. Earlier CPU outputs used `np.linalg.solve` for a six-dimensional
gravity transform: NumPy internally promoted that solve to double despite
returning float32. Those outputs are retained with this limitation. The final
`codex-local-chain-pure-fp32-20261115` run replaces it with explicit R-transpose
gravity in the selected dtype, as the CUDA probe already did. SciPy Cholesky
dispatch is checked to use single-precision LAPACK for float32 inputs.

Scope is a quaternion floating root, scalar child joints, constant reduced
mimic folds, constant armature and no external forces or joint-dynamics bias.
This finite study does not establish a broader support envelope.
The expanded CPU sweep also passes all 192 cases in both orders. The 128 added
cases pair zero-base-wrench and generalized-wrench inputs at eight additional
deterministic random configurations per robot.

## Bounded-memory prototype

Use one block per problem. Iterate over derivative pairs (j, k), with four
jet components: value, derivative j, derivative k, and ordered derivative jk.
Evaluate the same local RNEA recursion for q-q, v-v, v-q and acceleration-q.
The last block is dM/dq. Do not symmetrize the floating root's Lie Hessian.

Within a pair, threads own distinct spatial rows. A barrier separates dependent
tree bodies and phases. The backward pass visits bodies in a fixed reverse
order. Reduced mimic contributions accumulate in a fixed body order, with one
writer per output. No atomics, inter-block coordination or thread-dependent
reductions are needed. Threads above the six spatial rows are often idle; this
is a correctness baseline, not the proposed optimized schedule.

The prototype stores:

- X, dX and d2X: 108 NB floats, evaluated once per input.
- Velocity, acceleration and force jets: 72 NB floats, reused for every pair.
- Four temporary six-vector jets: 96 floats.
- Configuration: NQ floats.

Total static shared scratch is `(180 NB + NQ + 96) * sizeof(float)`, before
compiler layout padding. For iiwa14/gen3 (NB=8, NQ=14) this is 6,200 bytes.
For g1 (NB=30, NQ=36) it is 22,128 bytes. The four NV-cubed output tensors live
in their output allocation. There is no added global scratch arena. Query the
compiled kernel attributes to verify shared bytes, registers and local storage;
these estimates do not establish occupancy or speed.

Model constants and transform expressions are generated into a private source
file. Runtime quaternion normalization, mimic coordinates and transforms are
float32 on device. No reference transform values are uploaded. The prototype
uses local-frame gravity from the root rotation; it is checked through the
same output gate, not assumed equivalent at the last bit to a CPU matrix solve.

## Staged acceptance

The separate `local_chain_probe.cuh` extends the tensor baseline to an entire
GPU chain. It forms M via local RNEA acceleration columns, not the CPU CRBA
implementation. The bias and first derivative are computed with the same jets.
A deterministic diagonally scaled Cholesky factors M; triangular substitutions
apply the factor rather than multiplying by an explicit inverse. Solved identity
columns are used only where the force-q tensor formula requires them.

Its additional static shared storage is `4 NV^2 + 3 NV` floats: M (later reused
for solved identity), the Cholesky factor, two gradient slabs, scaling,
acceleration and bias. For g1 the total is 42,148 bytes before padding. This
fits the standalone probe's 48 KiB budget but exceeds the old standalone
world-frame inner's 29,088-byte scratch. It is not a drop-in inner replacement.
The combined FDSVA arena needs an explicit lifetime/placement reuse proof before
integration; an increase in a public workspace contract still requires Brian.
The probe keeps ID and FD output tensors in separate output allocations within
one host-owned output buffer, not in an added global scratch arena.

A possible integration layout reuses the existing caller-owned `s_Minv`
(NV squared), `s_df_du` (2 NV squared) and `s_qdd` (NV) buffers instead of
duplicating them in the private shared allocation. That leaves
`180 NB + NQ + 96 + NV^2 + 2 NV` scratch floats. For g1 (NB=30, NV=35,
NQ=36), the arithmetic estimate is 27,308 bytes, below the unsplit world inner's
29,088 bytes. This is a promising storage calculation, not an implemented or
validated alias plan. The surgical cold tier subtracts an 84 NB-float band,
and whole-pool/global and FD-gradient-spill tiers have different live regions.
Check those actual arena maxima and disjoint caller-buffer offsets before
claiming that all existing tiers fit. Preserve public matrix/gradient storage
ordering when reusing these buffers; the probe currently uses row-major slabs.

1. Complete CPU identity and unchanged float32 gates on the 64 saved cases.
2. GPU ID-tensor probe at the candidate CPU acceleration. Compose these tensors
   with the CPU candidate matrix, gradient and solve. Check original full
   acceptance, all output entries written, 1/32/128-thread bit identity, repeat
   determinism and sanitizer results. Stop expansion on a numerical failure.
3. Broaden CPU and device inputs. Preserve all failing probes. No per-robot
   algorithm switches or changed tolerances.
4. Integrate the remaining transform/mass/solve/bias/gradient/contraction stages
   from the private probe only after a complete live-buffer plan passes review. Reuse
   existing output/temporary lifetimes where proven. A required public scratch
   expansion or changed signature stops for Brian.
5. Integrate through the existing placement-aware inner/device/kernel surfaces;
   retain all tiers, non-default threads, batching, runtime model parameters and
   supported consumers. Keep the old implementation available for comparison
   until the full correctness matrix passes. No new support promise follows
   from the isolated probe alone.
6. Timing is a separate coordinated quiet-host experiment. Report compilation
   resource observations separately from robot-runtime performance.

## Optimization questions after correctness

The one-pair sweep repeats value and first-derivative work. Candidate next steps
are a small pair tile, caching baseline vectors, sparse tree dependency masks,
or caching first derivatives when they fit existing storage. Each trades memory
against recomputation. None is justified by speed before measurement. Compare
the complete stable solve plus local tensor path, not just the tensor kernel.

Thread/warp interfaces remain a separate design. They must isolate mutable
model/coupling state and define ownership of scratch, barriers and reductions.
Device-global memory is not pinned host memory. A block prototype fitting
shared memory does not establish that independent warps can safely share it.

## Recorded correctness result and precision terminology

The end-to-end device sweep passes all 64 saved inputs and 128 additional
inputs across eight robots. Every input is launched at 1, 32 and 128 threads,
then repeated at 32: 768 acceptance checks, with bit-identical output across
these launches. These are single-problem kernels, not batched product consumers.
Persistent outputs are `codex-full-gpu-saved-20261115` and
`codex-full-gpu-expanded-20261115` under the repository's ignored float32-study
task folder. No production generated header or receipt is replaced.

Here float32 means the candidate dynamics operands, arrays, accumulation and
solve, with standard float CUDA math APIs. It does NOT mean FP64-free machine
code: SASS inspection found DMUL instructions. A compile-only attribution probe
containing just `sinf(float)` or `cosf(float)` also emits a DMUL; equivalent
division, square root and square-power probes do not. The source introduces no
double dynamics solve, tensor arithmetic, buffer or mixed-precision refinement.
Do not silently replace standard math with fast approximations to remove those
library instructions. If zero FP64 instructions is a requirement, it is an
additional math-library/design question, not a property established here.
No claim is made about whether these library paths execute on the tested inputs
or about their runtime cost; neither tracing nor timing was performed.

The full-GPU binaries also passed memcheck, racecheck and synccheck for iiwa14
high-velocity, g1 random-1 and gen3 conservative at 32 threads: nine checks,
zero reported errors or race hazards/warnings. Sanitized output matched the
unsanitized output byte-for-byte. Logs and binary/input hashes are retained in
`codex-full-gpu-sanitizers-20261115`. This is sentinel coverage, not a claim that
every robot, launch shape or future consumer has been sanitized.
