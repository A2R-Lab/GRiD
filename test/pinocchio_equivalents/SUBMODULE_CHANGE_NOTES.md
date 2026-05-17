# URDFParser And RBDReference Change Notes

This note captures the functional changes made during the Pinocchio equivalence
work that affect `URDFParser` and `RBDReference`. The main goal is to make it
easy to mirror any relevant fixes or convention updates in the CUDA path later.

## Why This File Exists

The equivalence suite uncovered a mix of:

- parser robustness bugs,
- joint-ordering convention mismatches,
- fixed-joint kinematics bugs,
- continuous-joint support gaps,
- and reference-dynamics derivative bugs.

Some of these are test-only concerns, but the items below are real submodule
changes that affect model structure, kinematics, or dynamics behavior.

## URDFParser Changes

### 1. Whitespace-robust numeric parsing

Problem:
- URDF numeric fields such as `rpy`, `xyz`, and `axis` were split with
  `split(" ")`, which breaks on repeated spaces and can produce empty tokens.
- `to_float()` silently returned raw strings on parse failure, which delayed the
  error until much later in SymPy.

Change:
- Switched tokenization to whitespace-safe splitting.
- Tightened `to_float()` so malformed numeric input raises clearly instead of
  silently propagating string tokens.

Why it matters for CUDA later:
- Any codegen or preprocessing path that assumes raw token arrays may need the
  same normalization behavior to avoid parser divergence.

### 2. Rooted fixed-joint handling

Problem:
- `remove_fixed_joints()` assumed every fixed joint had a parent joint above it.
- Robots like `iiwa14` can have a fixed joint directly from the world/root link
  into the articulated tree.

Change:
- Added a rooted fixed-joint special case.
- Fixed joints attached directly at the root now store `parent_name = -1`
  instead of assuming there is an upstream articulated parent joint.

Why it matters for CUDA later:
- Any retained fixed-joint metadata consumed by generated kinematics code must
  understand the `-1` rooted-fixed-joint convention.

### 3. Missing joint origin defaults to identity

Problem:
- Some URDF joints omit `<origin>`, which is valid URDF and means identity.
- The parser assumed the tag existed and could fail with `NoneType` access.

Change:
- Missing joint origins now default to zero translation and zero rotation.

Why it matters for CUDA later:
- URDF normalization in any CUDA-facing import/codegen path should use the same
  identity-default rule so model transforms stay aligned.

### 4. Inertial-origin parsing now uses `<inertial><origin>`

Problem:
- `parse_links()` was reading the first `<origin>` inside a link instead of the
  inertial origin specifically.
- On robots like Baxter, this caused the spatial inertia COM offset to come from
  visual geometry instead of inertial metadata.

Change:
- Link inertial transforms are now built from `<inertial><origin>`.

Why it matters for CUDA later:
- This directly affects body inertias and therefore all dynamics kernels. Any
  CUDA-side inertia preprocessing must match this corrected interpretation.

### 5. Pinocchio-style joint ordering is now the default

Problem:
- GRiD originally used DFS over raw parser/URDF sibling order.
- Pinocchio uses a different deterministic DFS ordering on branched trees.

Change:
- Added `joint_ordering` support in the parser.
- Default behavior is now `pinocchio_order`, which is DFS with sibling sorting
  by child subtree name.
- Legacy URDF-order behavior remains available as an explicit mode.

Why it matters for CUDA later:
- Joint indices, state-vector layout, and any generated indexing tables must
  follow the new default ordering if CUDA is expected to match the CPU
  reference path.

### 6. Continuous joints are now supported like revolute joints

Problem:
- `continuous` joints were previously unsupported in `Joint.set_type(...)`.

Change:
- `continuous` now uses the same kinematics and motion subspace as `revolute`,
  but with unbounded limits at the parser level.

Why it matters for CUDA later:
- CUDA-side model ingestion should treat `continuous` exactly like `revolute`
  for kinematics/dynamics, while remembering that public `q` conventions may
  still differ from Pinocchio for these joints.

### 7. Signed cardinal joint axes are supported

Problem:
- Axis handling assumed only positive cardinal axes in several joint types.
- Some robots use negative-axis prismatic or revolute joints.

Change:
- Joint axis parsing now supports `+/-x`, `+/-y`, and `+/-z`.

Why it matters for CUDA later:
- Motion-subspace construction in generated kernels must preserve axis sign.

### 8. Fixed-joint homogeneous-transform propagation was corrected

Problem:
- After fixed-joint removal, the parser updated the spatial transforms used by
  dynamics but not the homogeneous transforms used by end-effector pose logic.
- Retained fixed-joint homogeneous chains were also composed in the wrong order.

Change:
- Fixed-joint collapse now updates homogeneous transforms consistently.
- Retained fixed-joint chains compose in the corrected order.
- `Joint` now supports explicitly resetting the homogeneous transform cache.

Why it matters for CUDA later:
- If CUDA or generated code uses retained fixed-joint kinematics, it needs the
  same corrected transform composition rules or pose outputs will diverge from
  CPU reference behavior.

### 9. Floating-base joints now build homogeneous transforms too

Problem:
- Floating-base parse used retained fixed-joint handling that expected
  homogeneous transforms to exist on joints even in floating-base mode.
- `Joint` previously skipped homogeneous-transform initialization whenever
  `Joint.floating_base` was true, which broke floating-base parse before
  numerical tests even started.

Change:
- Homogeneous-transform placeholders now exist regardless of base mode.
- Floating joints now build a homogeneous transform and expose a matching
  homogeneous-transform function in addition to the spatial transform.

Why it matters for CUDA later:
- Any floating-base kinematics or retained-fixed-joint bookkeeping in CUDA-side
  codegen should assume homogeneous transforms are available for the free-flyer
  path too.

### 10. Floating-base quaternion convention now matches Pinocchio

Problem:
- GRiD previously used floating-base quaternion order `wxyz`, while Pinocchio
  uses `xyzw`.
- That required adapter-side quaternion reordering and made floating-base
  equivalence harder to reason about.

Change:
- GRiD floating-base quaternion handling now uses `xyzw` directly.
- The quaternion-to-rotation path in `SpatialAlgebra` no longer reorders the
  quaternion components before building the rotation matrix.

Why it matters for CUDA later:
- Any CUDA or generated floating-base code that assumed `wxyz` must be updated
  to consume and produce `xyzw` instead.

### 11. Joint-type metadata is now stored on `Robot`

Problem:
- The equivalence layer needed to know which joints were `continuous`,
  `prismatic`, `revolute`, etc., but `Robot` did not expose a stable lookup.

Change:
- Added:
  - `joint_type_by_id`
  - `joint_type_by_name`
  - getter helpers for both

Why it matters for CUDA later:
- If CUDA-side equivalence or codegen needs per-joint convention handling, this
  metadata is now part of the CPU-side model contract.

## RBDReference Changes

### 1. Rooted fixed joints are handled in end-effector kinematics

Problem:
- Retained fixed-joint pose logic assumed every fixed joint had an articulated
  parent above it.

Change:
- End-effector kinematics paths now treat `parent_name == -1` as a root/world
  fixed transform with no articulated parent chain above it.

Why it matters for CUDA later:
- Any CUDA-side retained fixed-joint pose logic should honor the same rooted
  fixed-joint convention.

### 2. Fixed-base ABA `pA` bug was fixed

Problem:
- In fixed-base `aba(...)`, the articulated bias force `pA[:, ind]` was being
  collapsed to a scalar by indexing `[0]` after a matrix multiply, then
  broadcast back across the 6D vector.

Change:
- `pA[:, ind]` now keeps the full 6D spatial force vector.

Why it matters for CUDA later:
- This is a real algorithm bug, not a convention issue. Any CUDA ABA
  implementation should be checked for the same mistake.

### 3. Force-cross helper `fxS(...)` was corrected

Problem:
- The helper used in the `dq` backward pass of `rnea_grad(...)` was applying the
  wrong cross action for differentiating transported forces.
- This showed up clearly on Fetch with continuous roll joints upstream of
  prismatic gripper fingers.

Change:
- `fxS(...)` now uses the proper force-space dual operator:
  `dual_cross_operator(S) @ vec`
  instead of the previous incorrect motion-space-derived shortcut.

Why it matters for CUDA later:
- This directly affects inverse-dynamics gradients and forward-dynamics
  gradients. Any CUDA derivative kernels must mirror this corrected force-cross
  term.

### 4. Floating-base RNEA root convention was aligned with Pinocchio

Problem:
- Floating-base inverse dynamics previously disagreed with Pinocchio even after
  quaternion alignment.
- The remaining mismatch came from the floating root using a different
  user-facing velocity/acceleration ordering and the wrong gravity transport at
  the root.

Change:
- Floating-base `rnea(...)` now interprets the root velocity and acceleration
  inputs in Pinocchio-style order:
  `[vx, vy, vz, wx, wy, wz]`.
- The floating joint subspace now encodes the mapping from that user-facing
  Pinocchio order into GRiD's internal spatial-vector order, so the convention
  is native at the API boundary instead of being patched in adapters.
- Root gravity initialization in the floating-base forward pass now uses
  `inv(Xmat) @ gravity_vec` instead of `Xmat @ gravity_vec`.

### 5. Floating second-order dynamics now follow the verified first-order root convention

Problem:
- The top-level second-order helpers `idsva_so_body_frame(...)` and `fdsva_so(...)` only
  assumed fixed-base scalar-joint indexing.
- Floating-base rollout would have diverged immediately on the root
  configuration/velocity convention, and there is no direct Pinocchio
  second-order Python API to compare against.

Change:
- Added floating-base support for `idsva_so_body_frame(...)` and `fdsva_so(...)` using the
  same reduced floating dynamics convention already verified for first-order
  floating derivatives:
  - public floating `q` stays `[x, y, z, qx, qy, qz, qw, ...]`
  - root `qd/qdd` stay `[vx, vy, vz, wx, wy, wz]`
- The smoke-robot test rollout validates these second-order tensors against
  finite differences of the already-verified first-order `rnea_grad(...)`,
  `forward_dynamics_grad(...)`, `crba(...)`, and `minv(...)` paths.
- The current implementation is intentionally hybrid for floating base:
  - `idsva_so_body_frame(...)` keeps the analytic old-style path for the velocity-side and
    mass-matrix-side tensors, while patching `d2tau_dq` from the already-verified
    first-order `rnea_grad(...)` path.
  - `fdsva_so(...)` keeps the analytic old-style composition for the
    velocity-side and torque-side tensors, while patching `daba_dqdq` from the
    already-verified first-order `forward_dynamics_grad(...)` path.
- The public inverse-dynamics second-order name is now `idsva_so_body_frame(...)`, which
  mirrors `fdsva_so(...)` and replaces the older
  `second_order_idsva_parallel(...)` naming in the equivalence harness.

Why it matters for CUDA later:
- Any CUDA-side second-order rollout must mirror the same floating root
  indexing, root gravity transport, and root 6-vector convention as the
  already-verified first-order floating derivatives before tensor-level results
  can line up with the CPU reference path.

Why it matters for CUDA later:
- Any floating-base inverse-dynamics CUDA path must mirror both the root input
  convention encoded in the floating-joint subspace and the corrected root
  gravity transport if it is expected to
  match the CPU reference layer.

### 5. Rooted fixed-joint gradient cleanup

Problem:
- One rooted fixed-joint branch in the pose-gradient path multiplied a zero
  vector by `0` redundantly.

Change:
- Simplified the rooted fixed-joint branch to use the already-zero expression
  directly.

Why it matters for CUDA later:
- Mostly cleanup, but it is worth keeping the same rooted-fixed-joint branch
  structure in any mirrored gradient code.

### 6. Floating-base ABA, CRBA, and pose helpers received first-pass generalization fixes

Problem:
- Broader floating-base robots exposed several floating-only implementation bugs
  even after the Pinocchio-order migration was complete.
- Floating `aba(...)` still had a scalar-versus-vector articulated-bias update
  bug and a root acceleration update that used the wrong root subspace action.
- Floating `crba(...)` still allocated the mass matrix at body-count size
  instead of velocity-count size, so broader floating robots indexed past the
  matrix bounds.
- Floating `end_effector_pose(...)` still tried to evaluate the floating root
  transform with a scalar `q[0]` instead of the full root configuration slice.

Change:
- Floating `aba(...)` now treats the articulated-bias update term as a scalar
  scaling of a 6D vector rather than an invalid matrix product, and the root
  acceleration update now uses the floating joint subspace consistently.
- Floating `crba(...)` now allocates its result at `n x n` velocity size and
  writes the free-flyer root block into `[:6, :6]`.
- Floating `end_effector_pose(...)` now uses `get_joint_index_q(...)` when
  evaluating floating-root transforms in its forward and backward transform
  chains.

Why it matters for CUDA later:
- Any floating-base articulated-body, composite-inertia, or pose helper code in
  CUDA should be checked for these same root-expanded indexing and root-slice
  assumptions before it is trusted against the new CPU reference behavior.

### 6b. Floating-base ABA and CRBA now match Pinocchio on the enabled robot set

Problem:
- After the first floating-base generalization pass, `aba(...)` and `crba(...)`
  still were not trustworthy enough to enforce across the floating-enabled set.
- Floating `aba(...)` still mixed two root-only convention issues:
  the root gravity / acceleration initialization did not match the already-fixed
  floating `rnea(...)` path, and the root final forward update still treated the
  floating root like a transformed child.
- Floating `crba(...)` still mixed conventions inside the root-to-joint
  cross-term blocks: the root block itself was already in the Pinocchio-facing
  ordering, but the off-diagonal root cross terms were still emitted in the
  internal spatial row ordering.

Change:
- Floating `aba(...)` now keeps the root `U` solve in its local form, uses the
  same floating root gravity transport convention as `rnea(...)`, and skips the
  extra root `Xmat` application in the final root forward update.
- Floating `crba(...)` now leaves the root `6 x 6` block untouched and only
  reorders the root-to-joint cross terms into the Pinocchio-facing root order.
- The suite now enforces floating `aba(...)` and floating `crba(...)` across
  `iiwa14`, `go2`, `g1`, `fr3`, `fetch`, `baxter`, and `gen3`, with narrow
  singular-model skips retained for `rizon4`.

Why it matters for CUDA later:
- Any CUDA-side floating articulated-body or composite-inertia path should
  mirror the same root-only distinctions: local root solve quantities stay in
  the floating joint subspace convention, while user-facing mass-matrix
  cross-term blocks must be emitted in Pinocchio order.

### 7. Explicit-`world` URDF roots now convert cleanly into floating bases

Problem:
- Some upstream robot URDFs, including `gen3` and `rizon4`, already include an
  explicit `world` link with a fixed base attachment.
- The previous floating-base adjustment logic always injected a new synthetic
  `world` link and floating joint, which created an invalid `world -> world`
  self-loop for those robots and caused the DFS renumber pass to recurse
  indefinitely.

Change:
- Floating-base adjustment now detects when the URDF root is already `world`.
- In that case, it reuses the existing root child joint as the floating base
  joint instead of creating a second `world` wrapper.
- The converted root joint is renamed to `floating_base_joint`, switched into
  quaternion-based floating mode, and rebuilt as a floating joint in place.

Why it matters for CUDA later:
- Any preprocessing or codegen path that assumes floating-base robots always
  need a synthetic outer `world` wrapper should be updated to recognize explicit
  world-root URDFs and convert the existing root attachment instead.

### 8. Floating-base inverse-dynamics gradients were aligned with Pinocchio

Problem:
- After floating-base `rnea(...)` itself matched Pinocchio, the `dq` block of
  `rnea_grad(...)` still disagreed badly while the `d/dqd` block already
  matched.
- The remaining mismatch came from two floating-root-specific issues:
  the root `dq` backward pass still wrote out raw spatial-order results, and
  the floating `dq` forward pass still differentiated the root gravity term
  using the old `Xmat @ g` convention instead of the corrected
  `inv(Xmat) @ g` root transport.

Change:
- Floating `rnea_grad_bpass_dq(...)` now maps the root block through the
  floating joint subspace, just like the matching `d/dqd` path.
- Floating `rnea_grad_fpass_dq(...)` now uses the corrected root gravity
  transport when differentiating the floating-root acceleration with respect to
  root position.
- The fixed-base path was kept unchanged; the corrected gravity derivative is
  scoped to the floating root only.

Why it matters for CUDA later:
- Any floating-base inverse-dynamics gradient kernels must mirror both the root
  output mapping and the corrected root gravity derivative path to match the
  CPU reference and Pinocchio.

### 9. Floating-base end-effector derivative helpers now use an analytic free-flyer path

Problem:
- `end_effector_pose_gradient(...)` and `end_effector_pose_hessian(...)` were
  still shaped like fixed-base helpers.
- Floating-base calls need the full root configuration slice, and the analytic
  Hessian path also depends on derivative homogeneous transforms that the
  floating joint model does not currently expose.

Change:
- Added block-structured floating-joint homogeneous-transform derivatives for
  the 7D free-flyer configuration.
- Reworked the end-effector gradient/Hessian logic to traverse joint-local
  position blocks instead of assuming one scalar `q[ind]` per joint.
- The floating-base path is now analytic, while the Pinocchio-side suite still
  uses finite differences as the comparison oracle for Hessians.
- The fixed-base analytic behavior was left in place.

Why it matters for CUDA later:
- CUDA-side floating-base pose derivatives now have a concrete CPU-side
  analytic reference implementation to mirror.
- Any generated free-flyer kinematics code should expose the same 7D local
  derivative blocks for homogeneous transforms or provide an equivalent analytic
  chain-rule implementation.

## Behavior And Convention Changes To Remember

These are the highest-value items to keep aligned when updating CUDA:

1. Joint ordering now defaults to Pinocchio-style DFS sibling sorting.
2. Rooted fixed joints are represented with `parent_name = -1`.
3. Inertial origins must come from `<inertial><origin>`.
4. Continuous joints are accepted and behave like revolute joints internally.
5. Signed joint axes must be preserved.
6. Fixed-joint homogeneous transforms must be updated consistently after fixed
   joint removal.
7. ABA must keep full 6D articulated bias forces.
8. Gradient code must use the corrected force-cross helper in the backward pass.
9. Floating pose-derivative helpers now rely on analytic free-flyer transform
   derivatives.

## CUDA Update Checklist

When propagating these changes to CUDA or generated kernels, check:

- parser/codegen preprocessing for whitespace-robust numeric parsing
- body inertia construction against corrected inertial origins
- joint-order index generation against `pinocchio_order`
- retained fixed-joint metadata and rooted fixed-joint handling
- continuous-joint acceptance and signed-axis handling
- homogeneous transform composition for retained fixed-joint kinematics
- ABA articulated-bias-force handling
- inverse-dynamics and forward-dynamics gradient force-cross terms
- floating pose-derivative block semantics if CUDA needs end-effector derivatives

## Non-Submodule Test-Layer Adjustments

These were important for equivalence, but they are not submodule behavior
changes and usually do not need CUDA mirroring directly:

- Pinocchio-side normalization for continuous-joint configuration expansion
- name-based joint alignment in tests
- excluding mimic joints from generic kinematics target selection
- capability-gated skips for singular source models such as the current `rizon4`
  URDF
