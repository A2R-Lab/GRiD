# Thread- and warp-scoped dynamics

Status: design only. No new public API, emitter, scratch layout, GPU campaign or
timing run is implemented by this proposal. Block scope remains the supported
core. The proposed order is warp inverse dynamics, thread inverse dynamics,
then Minv, forward dynamics and their first derivatives. Second-order dynamics
and full feature parity are separate decisions.

## Assessment

This is feasible, but not a namespace substitution. GLASS supplies thread/warp
linear-algebra primitives and a rank/size/synchronization policy. GRiD must also
retarget its tree traversal, lane ownership, reductions, scratch ownership and
composition boundaries. A block barrier or block-strided loop hidden in a
generated helper would make an apparently warp-scoped entry point unsafe inside
a consumer kernel. A thread variant also needs a bounded register/live-value
plan; large automatic arrays can spill into local memory.

There is useful precedent: `_eepose_gradient_hessian.py` emits dedicated
`ee_pose_inner_thread` and `ee_pose_inner_warp` bodies; `_collision.py` emits a
warp namespace with per-warp scratch and a whole-warp calling contract. These
are evidence of the integration pattern, not proof that arbitrary dynamics
emitters already support the same scopes. GLASS's `ThreadBarrier` has rank zero,
size one and no synchronization; its warp primitives use explicit lane
communication. Neither supplies GRiD's complete dataflow automatically.

## Proposed caller contract to review before implementation

Each problem owns its input/output slice and non-overlapping scratch slice.
Robot constants may be shared read-only. The API must distinguish immutable
model data, per-sample dynamic intermediates, outputs and temporary workspace.
Publish exact element counts, alignment, strides, alias restrictions and scratch
lifetime. Provide workspace-size queries before a public release; do not require
callers to duplicate generator layout formulas.

For warp scope, initially require a full 32-lane warp, all lanes entering the
function with uniform control flow and the same problem pointers. Several warps
in one block must work independently: no block-wide synchronization or shared
scratch overlap. State explicitly when outputs are visible to the participating
warp and what synchronization a subsequent cross-warp consumer owns. Partial
warps and arbitrary cooperative groups are outside the first prototype.

For thread scope, one lane owns a complete problem. Use explicit global scratch
for large intermediates, with small vectors/scalars retained in registers where
practical. Compare problem-major and component-major batch layouts before
freezing strides: thread-per-problem favors adjacent lanes accessing adjacent
problems, while warp-per-problem favors adjacent lanes accessing a single
problem's components. An internal layout adapter may be preferable to exposing
separate mathematical APIs.

Device global scratch does not require page-locked host memory. If “pinned”
means keeping data in cache, treat that as an optional performance strategy,
never a correctness requirement or residency guarantee. Warp shared-memory
staging should be explicit and limited to measured hot intermediates. No cache
policy or scratch expansion is approved by this note.

## Minimal inverse-dynamics prototype

1. Inventory all helpers reachable from the block ID inner: rank/stride loops,
   barriers, reductions, mutable static/shared state and GLASS calls. Classify
   each as scope-neutral, parameterizable or needing a separate emitter.
2. Emit a private warp ID inner using existing model conventions and unchanged
   block code. Start with a simple fixed-base chain; use it to prove ownership,
   not to define a supported-platform list. Add branched, floating-base and
   mimic/coupled sentinels before recommending a public interface.
3. Exercise two independent warps per block, distinct scratch, nontrivial batch
   strides, consumer composition and output/scratch canaries. CPU source checks
   must reject block barriers and block-rank assumptions in the warp closure.
4. After explicit GPU approval, compare with the unchanged oracle and block ID
   under existing tolerances, then check memory/race behavior. No performance
   claim follows from source generation or CPU checks.
5. Only after warp correctness and a reviewed scratch contract, repeat for the
   private thread ID inner. First-derivative storage scales faster than ID;
   Minv and derivative prototypes need separate scratch/register reviews.

The prototype should expose a private experimental device entry, not a permanent
Python selection parameter. The acceptance report must distinguish supported
features, deliberately unimplemented features, and tested configurations.
Unimplemented features must fail clearly; they must not silently route to a
different mathematical convention. Keep the float32 FDSVA accuracy experiment
separate from this execution-scope experiment.

## Cost and validation scope

The principal work is auditing and refactoring generated cooperation, then
testing composition—not allocating global memory. Cost cannot be estimated
credibly until the ID helper inventory identifies how much can be shared.
If most helpers are scope-neutral, the ID prototype is a bounded change; if
block-local assumptions permeate spatial recursion, it requires new emitters.
Do not extrapolate an ID estimate to gradients or the entire library.

Add tier to compilation and evidence identities. Preserve existing block node
IDs; add scope-specific nodes grouped by robot/base/algorithm compile atoms.
Two new scopes increase both compilation variants and correctness obligations;
thread count within a block is not a substitute for thread-per-problem coverage.
Use the smaller receipt units and measured costs before considering full parity.
Timing is a later, separately approved experiment across batch sizes and robots,
including an embedded-device campaign if requested. No Jetson work is implied.
