# Pinocchio Mapping Inventory

This inventory was created from the checked-out `URDFParser` and `RBDReference`
code in `research/GRiD-A2R`. It is intentionally conservative: only mappings that
appear clear from the current source are considered safe for enforcing tests.

## RBDReference Functions Found In This Checkout

Clear or mostly clear dynamics-facing surface:

- `rnea(q, qd, qdd=None, GRAVITY=-9.81, f_ext=None)`
- `minv(q, output_dense=True)`
- `crba(q)`
- `aba(q, qd, tau, f_ext=[], GRAVITY=-9.81)`
- `forward_dynamics(q, qd, u)`
- `rnea_grad(q, qd, qdd=None, GRAVITY=-9.81, USE_VELOCITY_DAMPING=False)`
- `forward_dynamics_grad(q, qd, u)`
- `idsva_so(q, qd, qdd, GRAVITY=-9.81)`
- `fdsva_so(q, qd, u, GRAVITY=-9.81)`

Kinematics-facing surface:

- `end_effector_pose(...)`
- `end_effector_pose_gradient(...)`
- `end_effector_pose_hessian(...)`

Pass-level helpers and implementation internals:

- `rnea_fpass(...)`, `rnea_bpass(...)`
- `minv_bpass(...)`, `minv_fpass(...)`
- `rnea_grad_fpass_dq(...)`, `rnea_grad_fpass_dqd(...)`
- `rnea_grad_bpass_dq(...)`, `rnea_grad_bpass_dqd(...)`
- several spatial algebra helpers used by the algorithms above

## Clear Pinocchio Mappings Used In V1

- `RBDReference.rnea(...)` maps to `pinocchio.rnea(...)`
  Note: GRiD returns `(c, v, a, f)` while Pinocchio returns the generalized
  torque result directly, so the suite compares `c` to Pinocchio's `tau`. For
  floating-base robots, GRiD now interprets the root velocity and acceleration
  inputs in Pinocchio-style `[vx, vy, vz, wx, wy, wz]` order and returns the
  root generalized-force block in the same order.
- `RBDReference.minv(...)` maps to a Pinocchio mass-matrix path using
  `pinocchio.crba(...)` followed by matrix inversion.
  This is treated as the stable Python-side comparison target for v1,
  including floating-base smoke robots now that GRiD uses Pinocchio-style
  free-flyer ordering natively.
- `RBDReference.crba(q)` maps to `pinocchio.crba(...)`
  This is currently enforced for the verified fixed-base default robots
  `iiwa14`, `go2`, `g1`, `fetch`, `baxter`, `fr3`, `gen3`, and `rizon4`.
  It is also enforced for floating-base `iiwa14`, `go2`, `g1`, `fr3`, `fetch`,
  `baxter`, and `gen3`, with singular-model skips for `rizon4`.
- `RBDReference.aba(q, qd, tau, ...)` maps to `pinocchio.aba(...)`
  This is currently enforced for the verified fixed-base default robots
  `iiwa14`, `go2`, `g1`, `fetch`, `baxter`, `fr3`, and `gen3`.
  It is also enforced for floating-base `iiwa14`, `go2`, `g1`, `fr3`, `fetch`,
  `baxter`, and `gen3`, with singular-model skips for `rizon4`.
- `RBDReference.forward_dynamics(q, qd, u)` maps to the same forward-dynamics
  acceleration computed by `pinocchio.aba(...)`, because the current GRiD
  implementation composes inverse dynamics and inverse mass to recover the ABA
  result. This is currently enforced for the verified fixed-base default robots
  `iiwa14`, `go2`, `g1`, `fetch`, `baxter`, `fr3`, and `gen3`.
  It is also enforced for the floating-enabled robots `iiwa14`, `go2`, `g1`,
  `fr3`, `fetch`, and `baxter`.
- `RBDReference.forward_dynamics_grad(q, qd, u)` maps to
  `pinocchio.computeABADerivatives(...)` for the `ddq_dq` and `ddq_dv` blocks.
  This is currently enforced for the verified fixed-base default robots
  `iiwa14`, `go2`, `g1`, `fetch`, `baxter`, `fr3`, and `gen3`.
  It is also enforced for floating-base `iiwa14`, `go2`, `g1`, `fr3`, `fetch`,
  `baxter`, and `gen3`, with singular-model skips for `rizon4`.
- `RBDReference.rnea_grad(...)` maps to `pinocchio.computeRNEADerivatives(...)`
  for the `dtau_dq` and `dtau_dv` blocks. This is currently enforced for the
  verified fixed-base default robots `iiwa14`, `go2`, `g1`, `fetch`, `baxter`,
  `fr3`, `gen3`, and `rizon4`.
  It is also enforced for floating-base `iiwa14`, `go2`, `g1`, `fr3`, `fetch`,
  `baxter`, `gen3`, and `rizon4`.
- `RBDReference.end_effector_pose(...)` maps to Pinocchio frame placements plus
  local-point offsets. The suite compares translation directly and compares
  orientation through reconstructed rotation matrices to avoid Euler-angle
  singularity artifacts. This is currently enforced for the verified fixed-base
  default robots `iiwa14`, `go2`, `g1`, `fetch`, `baxter`, `fr3`, `gen3`, and
  `rizon4` using model-derived joint and fixed-joint targets that exist on both
  the GRiD and Pinocchio sides. It is also enforced for floating-base
  `iiwa14`, `go2`, `g1`, `fr3`, `fetch`, `baxter`, `gen3`, and `rizon4`.
- `RBDReference.end_effector_pose_gradient(...)` maps to first derivatives of
  the same Pinocchio frame-placement-plus-offset pose target. The current suite
  compares GRiD against a Pinocchio-side finite-difference reference in project
  `q` coordinates and enforces this on the verified fixed-base and
  floating-base default robots using articulated leaf-joint targets.
- `RBDReference.end_effector_pose_hessian(...)` maps to second derivatives of
  the same pose target. The current suite compares GRiD against a Pinocchio-side
  finite-difference reference and enforces this on a focused `iiwa14`
  fixed-base and floating-base slice.
- `RBDReference.idsva_so(...)` does not have a direct
  Pinocchio second-order inverse-dynamics API counterpart in the Python stack
  used here, so the suite validates it against finite differences of the
  already-verified `rnea_grad(...)` and `crba(...)` paths. This is currently
  enforced on a smoke-robot-first rollout: fixed-base `iiwa14` and floating-base
  `iiwa14`, `go2`, and `g1`, with `g1` currently treated as runtime-heavy
  rather than numerically suspect.
- `RBDReference.fdsva_so(...)` likewise does not have a direct Pinocchio
  second-order forward-dynamics API counterpart in the Python stack used here,
  so the suite validates it against finite differences of the already-verified
  `forward_dynamics_grad(...)` and `minv(...)` paths. This is currently
  enforced on the same smoke-robot-first rollout: fixed-base `iiwa14` and
  floating-base `iiwa14`, `go2`, and `g1`, with `g1` currently treated as
  runtime-heavy rather than numerically suspect.

## Ambiguous Or Deferred Mappings

- `forward_dynamics_grad(...)`
  The mapping to Pinocchio ABA derivatives is now explicit for fixed-base
  `iiwa14`, but broader fixed-base coverage and floating-base coverage are still
  deferred.
- Broad end-effector Hessian coverage beyond the focused `iiwa14`
  fixed/floating slice
  The current comparison is intentionally narrow because the Hessian path uses
  higher-runtime finite differences on at least one side, so widening coverage
  should be done deliberately.
- Broad second-order inverse-dynamics / forward-dynamics coverage beyond the
  fixed-base `iiwa14` and floating-base smoke rollout
  These tensor checks are intentionally narrow for now because they are
  validated through higher-runtime finite differences of first-order quantities.

## Normalization Steps Required Today

- GRiD floating-base quaternion order is now `xyzw`, matching Pinocchio's
  free-flyer quaternion convention.
- GRiD floating-base user-facing root 6-vectors now follow Pinocchio order
  `[vx, vy, vz, wx, wy, wz]`. The floating-joint subspace handles the mapping
  into GRiD's internal spatial-vector order.
- GRiD `URDFParser` uses parser-defined DFS joint ordering with optional sibling
  tie-breaking, so joint-name alignment must be explicit.
- GRiD merges fixed joints into retained custom structures, while Pinocchio keeps a
  different model/data view for those URDF elements.
- `RBDReference.rnea(...)` returns more than the primary generalized torque result,
  so tests extract and compare only the torque-like output for v1.

## Relevant Pinocchio Capabilities Not Yet Implemented In RBDReference

The list below is intentionally restricted to capabilities that look relevant from
the current repo layout and documentation, not from guesswork about hidden APIs.

- Pinocchio-style model/data separation as a first-class public Python interface
- A mass-matrix-first public API surface analogous to `crba` plus helper utilities
- More Pinocchio-shaped user-facing wrappers for bias, gravity, and composed
  dynamics quantities
- Clearly documented free-flyer conventions that match Pinocchio terminology
- Broader, clearly exposed kinematics and frame-placement helpers comparable to
  Pinocchio's frame APIs

## V1 Deferrals

- Additional fixed-base algorithms beyond `rnea`, `minv`, `crba`, `aba`,
  `forward_dynamics`, `forward_dynamics_grad`, `rnea_grad`, selected pose
  targets, end-effector pose gradients, and focused end-effector Hessian checks
  on the verified default robots
- Broad end-effector Hessian equivalence beyond the focused `iiwa14`
  fixed/floating slice

## Current Suite Findings

- Fixed-base `iiwa14`: `rnea`, `minv`, `crba`, `aba`, `forward_dynamics`,
  `forward_dynamics_grad`, `rnea_grad`, and selected pose targets match
  Pinocchio in the current suite.
- Fixed-base `go2`: `rnea`, `minv`, `crba`, `aba`, `forward_dynamics`,
  `forward_dynamics_grad`, `rnea_grad`, and selected pose targets match
  Pinocchio in the current suite.
- Fixed-base `g1`: `rnea`, `minv`, `crba`, `aba`, `forward_dynamics`,
  `forward_dynamics_grad`, `rnea_grad`, and selected pose targets match
  Pinocchio in the current suite using a narrowly scoped `g1` tolerance override.
- Fixed-base `fetch`: `rnea`, `minv`, `crba`, `aba`, `forward_dynamics`,
  `forward_dynamics_grad`, `rnea_grad`, and selected pose targets match
  Pinocchio in the current suite after continuous-joint normalization and the
  corrected force-cross derivative transport term.
- Fixed-base `baxter`: `rnea`, `minv`, `crba`, `aba`, `forward_dynamics`,
  `forward_dynamics_grad`, `rnea_grad`, and selected pose targets match
  Pinocchio in the current suite after inertial-origin and fixed-joint
  homogeneous-transform fixes.
- Fixed-base `fr3`: `rnea`, `minv`, `crba`, `aba`, `forward_dynamics`,
  `forward_dynamics_grad`, `rnea_grad`, and selected pose targets match
  Pinocchio in the current suite, with generic pose-target selection excluding
  URDF mimic joints such as `fr3_finger_joint2`.
- Fixed-base `gen3`: `rnea`, `minv`, `crba`, `aba`, `forward_dynamics`,
  `forward_dynamics_grad`, `rnea_grad`, and selected pose targets match
  Pinocchio in the current suite.
- Fixed-base `rizon4`: parse, metadata, `rnea`, `crba`, `rnea_grad`, and
  selected pose targets match Pinocchio in the current suite. `minv`, `aba`,
  and forward-dynamics-family checks are explicitly skipped because the resolved
  source model exposes a singular zero-mass-matrix interpretation on both the
  GRiD and Pinocchio sides.
- Floating-base `iiwa14`, `go2`, and `g1`: parse and metadata match, and
  floating-base `rnea`, `minv`, `crba`, `aba`, `forward_dynamics`,
  `rnea_grad`, and `forward_dynamics_grad`, and selected pose targets match
  Pinocchio in the current suite.
- Floating-base `fr3`, `fetch`, and `baxter`: parse and metadata match, and
  floating-base `rnea`, `minv`, `crba`, `aba`, `forward_dynamics`,
  `rnea_grad`, and `forward_dynamics_grad`, and selected pose targets match
  Pinocchio in the current suite.
- Floating-base `gen3`: parse and metadata match, and floating-base `rnea`,
  `minv`, `crba`, `aba`, `forward_dynamics`, `rnea_grad`, and
  `forward_dynamics_grad` match Pinocchio in the current suite, along with
  selected pose targets.
- Floating-base `rizon4`: parse and metadata match, and floating-base `rnea`
  and `rnea_grad` match Pinocchio in the current suite, along with selected
  pose targets. Floating-base `minv`, `crba`, `aba`, `forward_dynamics`, and
  `forward_dynamics_grad` are explicitly skipped because the resolved source
  model is singular on both sides.
- End-effector pose gradients: the current suite matches Pinocchio across the
  verified fixed-base and floating-base default robots using articulated
  leaf-joint targets shared by both models.
- End-effector pose Hessians: the current suite matches Pinocchio on a focused
  `iiwa14` fixed-base and floating-base slice.
- Second-order inverse dynamics and forward dynamics: the current suite
  validates `idsva_so(...)` and `fdsva_so(...)` against finite differences of
  already-verified first-order quantities on fixed-base `iiwa14` and
  floating-base `iiwa14`, `go2`, and `g1`. In practice, `g1` is currently the
  runtime-heavy member of that rollout rather than the numerically suspicious
  one.
