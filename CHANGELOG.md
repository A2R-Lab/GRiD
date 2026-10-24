# Changelog

All notable changes to GRiD are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) with the pre-1.0 caveat that minor
versions may break APIs.

GRiD was first released alongside the ICRA 2022 paper *GRiD: GPU-Accelerated
Rigid Body Dynamics with Analytical Gradients*. That implementation is preserved
in the archival [robot-acceleration/GRiD](https://github.com/robot-acceleration/GRiD)
repository. Everything below describes the packaged A2R-Lab releases.

## [Unreleased]

- Margin-aware collision checking (HJCD-IK ask G7). `grid_collision::config_free` on the block,
  warp and capsule-row paths and the header drivers `grid_cc_config_free` and
  `grid_cc_config_free_capsule` take a trailing `margin` (metres, default 0): a configuration is a
  hit as soon as any raw clearance is below the margin, robot spheres being tested at `r + margin`
  against the environment and the margin added to one side of each self pair, so exact contact at
  the margin is free and the default verdict is unchanged. `collision_distance`, the normals and
  the cost derivatives keep the geometric signed clearance; no margin-adjusted clearance is
  emitted. Gated warp against block and against the distance surfaces at 0, 5 mm and 2 cm, with a
  sweep from just below to just above each configuration's own threshold.
- Self-collision toggle (HJCD-IK ask G10). The combined checkers take `check_self` (default true);
  false skips the baked self-pair ranges so the verdict reflects the environment alone.

## [0.5.5] — 2026-10-23

- Floating-base cost menu. `tracking_cost_menu` and its gradient and Hessian are
  emitted on a floating base (a free-flyer root with single-axis joints; a spherical
  joint is refused by name) in the tangent layout: the state term is the new
  `state_cost_tangent` on the residual `[difference(q_des, q); qd - qd_des]` under
  any activation, with tangent-indexed bounds whose base-pose entries carry no
  bound; the input terms, the torque barrier and the linear u feature act on the
  `NUM_VEL - 6` actuated inputs; the position barrier bounds the joint positions,
  the velocity barrier the tangent velocities; the linear x feature reads
  `[q[7:]; qd]`; the gradient is `2 NUM_VEL` wide and the Hessian `2 NUM_VEL`
  square, column-major. `GRID_PLANT_MENU_TANGENT`, `GRID_PLANT_MENU_NU` and
  `GRID_PLANT_MENU_NLX` state the layout. Zero base weights under the quadratic
  activation reproduce `quadratic_state_cost_tangent`. The frame terms are
  tangent-sized on both bases (identical to the x layout on a fixed base).
- `frame_velocity_cost` on a floating base: the configuration derivative of the
  frame twist is taken along the tangent, the root entering as six columns under
  the same rigid-motion rules as a joint; oracle and CUDA cell on go2.
- The barrier families take `COUNT` and `VAR_OFFSET` templates (defaults unchanged),
  the input family an `N` template (default `NUM_VEL`) and
  `linear_feature_cost_hessian` a leading-dimension template `LD` (default `N`), so
  the preset can place slices inside a larger gradient or Hessian.
- The cuBLASDx-era linalg scratch is gone from the generated header: the
  `grid_linalg_*` wrappers lose their ignored trailing `glass_nvidia_smem`
  parameter, the inners that carried `unsigned char *s_linalg_smem` as a reserved
  last argument (the end-effector pose inners, `centroidal_inner`, `dccrba_inner`,
  the contact and multi-target inners) drop it, and the zero-returning
  `GRID_LINALG_NVIDIA_MAX_HELPER_BYTES` and `GRID_EE_LINALG_SHARED_BYTES` sizers
  and their arena byte region are removed. A consumer that passed `nullptr` in
  that position removes the argument; no shared-memory size changes (the region
  was zero bytes).
- The orientation residual's SO(3) log map and inverse right Jacobian come from
  the vendored GLASS (`glass::thread::so3_log`, `glass::thread::so3_right_jacobian_inv`)
  instead of a private copy.
- One capability constant per cost-menu piece (`GRID_PLANT_HAS_ACTIVATIONS`,
  `GRID_PLANT_HAS_STATE_COST`, `GRID_PLANT_HAS_INPUT_COST`,
  `GRID_PLANT_HAS_LINEAR_FEATURE_COST`, `GRID_PLANT_HAS_FRAME_EVAL`,
  `GRID_PLANT_HAS_FRAME_POSITION_COST`, `GRID_PLANT_HAS_FRAME_ORIENTATION_COST`,
  next to the existing `GRID_PLANT_HAS_FRAME_VELOCITY_COST`, `GRID_PLANT_HAS_COST_MENU`,
  `GRID_PLANT_HAS_GRAVITY_TORQUE` and `GRID_PLANT_HAS_COST_FRAMES`), so an adapter
  can `static_assert` on what the header emitted instead of reading the skip comments.
- Smooth hinge activation (`GRID_PLANT_ACTIVATION_SMOOTH_HINGE`, id 4) in the cost
  menu: `w (sqrt(d^2 + eps^2) - eps)` on the bound violation `d = max(0, lo - v) +
  max(0, v - hi)`, zero inside the bounds and pseudo-Huber outside, with slope and
  curvature on the diagonal like the other activations; oracle, CUDA cell and docs
  (asked by Felix Mueh on GATO issue #20).
- The Pinocchio oracle's end-effector pose gradient is analytic (frame Jacobian and
  the roll-pitch-yaw rate map) and its pose Hessian is one central difference of
  it. The installed Pinocchio has no kinematic-Hessian binding, so the previous
  path differenced a differenced gradient with a noise floor near 1e-5; the fp64
  pose-Hessian cells now pass at the fp64 tolerance class on every robot. The
  reference drops a duplicated copy of its activation block.
- Legacy code retired. `grid_codegen/_reference_impl.py`, an early numpy reference
  bound into the generator as `test_*` methods, is deleted: its `mxS` dispatched
  on a subspace entry equal to +1 and so dropped the velocity-product term of a
  revolute joint with a negative axis (GATO issue #22, reported by Felix Mueh).
  RBDReference is the one reference; `examples/codegen/print_reference_values.py`
  prints its intermediate passes. The reference manifest gains
  `signed_axis_arm` (+Z, -Z, +X and -Y revolute axes), so the signed-axis property
  is locked against Pinocchio where the reference lives and in the CUDA receipt.
- Clean break on compatibility knobs: `allow_fp64` (the fp32-compute, fp64-copy
  upcast; `dtype="float64"` is the fp64 tier), the `all-dynamics` /
  `dynamics-only` / `kinematics-only` profile aliases, the parser's
  `alpha_tie_breaker` alias of `joint_ordering`, and the parser's
  `floating_base_convention="legacy"` layout switch with its permutation helpers
  (the pinocchio layout is the only one on every surface). With that switch gone
  the identity normalize and denormalize helpers on `Robot` and `RBDReference`
  and the `normalize_input` / `public_output` keywords on the RBDReference
  dynamics entry points are removed as well; every call takes and returns the
  pinocchio layout. RBDReference drops its unused `xfail_if_mimic` helper and
  the convention test.

## [0.5.4] — 2026-10-21

- Docs: the compatibility page records the JAX 0.11 CUDA initialization
  warning under Compute Sanitizer (fabric handle support requires IMEX
  channels; reproduced without GRiD) as an accepted limitation of the
  combined sanitizer invocation, with the NumPy and PyTorch surfaces passing
  at zero errors.
- Cost menu. `grid_plant` gains the activation layer (quadratic, pseudo-Huber
  with a positive `eps`, the relaxed log barrier and the one-sided quadratic
  barrier on bounded feature values), the activated terms `state_cost` and
  `input_cost`, the linear feature residual
  `linear_feature_cost<T, M, N>` (`A v + b - target` over the state or the
  input with a caller matrix), baked cost frames (`cost_frames=` on
  `gen_all_code` and `register_robot`: named joints or fixed joints with their
  offsets, the end-effector target always frame 0) with `frame_eval`,
  `frame_position_cost`, the log-map `frame_orientation_cost` and, on fixed
  base, `frame_velocity_cost` with its analytic configuration derivative, the
  caller-scratch `gravity_torque` (RNEA at zero velocity) for gravity-holding
  torque targets, and `tracking_cost_menu` behind `cost_menu_params<T>`: a
  per-term activation choice and parameter per knot, linear features on `x`
  and `u`, the frame orientation and twist and a second frame position. Every
  existing term keeps its bytes and the quadratic activation reproduces it;
  `RBDReference` carries the same functions, checked against finite
  differences, and the CUDA plant suite compares every term with the
  reference on iiwa14, go2 and fr3 and the menu preset with the composition of
  its terms.
- Topology helpers. The header states the rule for `TOPOLOGY_HELPERS_COUNT`
  (`(not serial) * (5n + 1) + (not identical S) * n`, `n = max(NUM_POS,
  NUM_JOINTS)`: iiwa14 0, a three-joint +Z, -Z, +X arm 3, rizon4 7) in
  `examples/cuda/README.md`, `examples/cuda/wrapper_types.md` and the agent
  integration guide; emits `grid::TOPOLOGY_HELPERS_T_COUNT<T>()`, the helper
  block's footprint in T elements, next to the count so an arena sized in T
  reserves `COUNT + TOPOLOGY_HELPERS_T_COUNT<T>()`; and, in a `DEBUG_MODE`
  header, asserts at the top of every helper-reading `_inner` that the pointer
  is not null when the count is nonzero. Release headers gain the one
  constexpr line only.
- Suggested thread counts. Every header ends with `grid::<ALGO>_SUGGESTED_THREADS`
  for each algorithm it carries, plus `TRACKING_COST_SUGGESTED_THREADS` when the
  plant preset is emitted: the smallest multiple of 32 covering the widest
  block-parallel compute loop reachable from the algorithm's kernel, read back
  from the emitted code at generation time and clamped to [32, 1024]. Copy and
  zero-fill loops do not count. The constants are GPU-independent and untuned;
  a consumer that fuses several algorithms into one kernel takes the max over
  the ones it calls instead of `MAX_PERF_LEVEL_THREADS`, and `launch_cfg`
  stays the tuned override. No kernel body changes; the block reopens the
  namespace after everything else (fragment `suggested_threads`).
- Smooth joint friction. `GRiDCodeGenerator(..., FRICTION_MODEL="tanh",
  FRICTION_WIDTH=w)` and `register_robot(..., use_joint_dynamics=True,
  friction_model="tanh", friction_width=w)` replace the Coulomb term
  `friction*sign(qd)` by `friction*tanh(qd/w)` (w = 0.03 rad/s by default, one
  value for every velocity slot or one per slot). The inverse-dynamics
  gradient carries its slope `(friction/w) sech^2(qd/w)` on the velocity
  diagonal, the second-order kernels carry `-(2 friction/w^2) tanh(qd/w)
  sech^2(qd/w)` on the qd-qd triple diagonal, and the forward-dynamics
  derivatives follow by composition, so a consumer that linearizes at rest
  sees the friction slope the sign model does not have. `RBDReference(...,
  friction_model=, friction_width=)` is the matching reference. On a
  `runtime_joint_dynamics` build the width is the third row of the mutable
  table (`set_joint_dynamics(friction_width=)`, `handle.joint_friction_width`,
  `grid::JOINT_DYNAMICS_PARAMS_COUNT`). The default sign model and every
  friction-free robot emit byte-identical headers. Tests cover fr3 (friction
  and a mimic joint) at rest and inside the width on the CUDA runner, the
  numpy, jax and torch surfaces and the runtime table, and the reference
  against its own finite differences on fr3 and floating iiwa14.
- Release measurements: the core and wrapper dataset is the 21 October 2026
  session at one commit (315 workers, 1,890 measurements at 300 samples per
  repeat), replacing the stitched September and October captures. It adds the
  Hessian wrapper cells and the CPU and GPU baselines at the same protocol, so
  the published selection carries no accepted source drift. The secondary
  table keeps its 30-sample captures.
- Tests: the external-force CUDA modules run their pinocchio leg on every robot.
  gen3 was left out because the modules passed tolerance keys RBDReference did
  not have, which fell back to its 1e-7 default; the mimic robots were left out
  because the pinocchio oracle placed per-body wrenches through a mimic-free
  joint list. RBDReference (external/RBDReference) now refuses unknown tolerance
  keys, carries forward_dynamics and inverse_dynamics_gradient buckets, maps a
  wrench on every body including mimic-driven ones, and checks external forces
  on every robot against pinocchio.
- Tests: test_osc_inertia checks every sample through the normwise backward
  error of the task-space inertia (the check the frame-Jacobian suite uses)
  instead of passing over ill-conditioned samples.
- Tests: g1 floating joins the integrator gradient's float32 norm-bound class
  with a bound measured on the Jetson AGX Orin (sm_87), where the RK4 gradient at
  the most energetic sample rounds three times wider than on the RTX 5090. The
  entrywise check runs first and still passes on the 5090.
- Receipts: compute-sanitizer racecheck is unsupported on the Jetson Orin's
  integrated GPU ("GPU debugging features are disabled"); the racecheck cell
  skips on that exact message and the sm_87 receipt row carries its own
  expected-skips baseline listing that one cell. The primary baseline stays empty.
- Docker: create the sample robot's output directory before generating its
  header. The corrected 0.5.3 image builds and publishes in CI. An RTX 5090
  container smoke test passed on 2026-10-19: precompiled iiwa14 values on all
  three backends, new-robot compilation, autograd and cache-only reload.

## [0.5.3] — 2026-10-18

Published on PyPI and GitHub. Includes the changes listed under the
TestPyPI-only 0.5.2 rehearsal below.

- Fix per-sample external-force indexing in batched inverse/forward dynamics,
  ABA, their gradients, and integrator value/gradient kernels. Previously every
  sample used the first force row. Single-sample and no-force calls are unchanged.
- Add real-robot installed-wheel numerical acceptance checks to the release gate
  and signed suite, including independently varying force rows across NumPy,
  JAX and PyTorch. TestPyPI rehearsals now explicitly require exact artifact-hash
  matching, the numerical matrix, and independently installed source archives.

## [0.5.2] — TestPyPI rehearsal only

Not released on production PyPI; superseded by 0.5.3.

- Benchmark validation checks large Hessians in bounded-memory chunks and releases
  completed snapshots sooner. Checked entries, tolerances and timed calls are unchanged.
- Receipt resumption requires a successful shard and an explicit clean receipt at the
  current commit; missing-tool skips and missing receipts are rerun, not retained as proof.

- URDF parser (external/URDFParser): an `<inertial><origin rpy>` now rotates the inertia
  tensor into the link frame before the parallel-axis shift; it was taken unrotated, which
  put the UR5e's shoulder and wrist inertias 10 percent off pinocchio. A prismatic joint
  whose `<origin>` is rotated now translates through that rotation in the homogeneous
  transform used by the end-effector pose; the 6x6 motion transforms and the dynamics were
  already right. None of the ten shipped robots exercises either case in a form the receipt
  could see, so no generated header changes.
- Parser oracle: `test/test_urdf_oracle.py` checks every shipped robot and every parser
  fixture against pinocchio, body inertias and joint placements at random configurations,
  in the CPU lane; the parser's own suite gains the same oracle over its fixtures and a
  fixture with rotated inertial origins, a rotated prismatic origin, a massive fixed child
  and hyphenated joint names. The parser CI job installs the dev set so the oracle runs.
- Codegen: a fixed-joint target whose name is not a C identifier (`tool0-flange`) now
  emits valid symbols; the raw name still keys the lookup and the string macro.

- Benchmark collector: Frax's floating-base cells validate on the samples whose Euler
  chart is well conditioned (1/|cos pitch| at most 10) and record the excluded count in
  the cell and the report; every sample is still timed. Near the pitch singularity
  Frax's fp32 Cholesky solve of its Euler-chart mass matrix carries 1e-3 to 1e-2
  relative error on the GPU while the transport is exact in float64.
- Python 3.13: the wheel matrix and the classifiers add CPython 3.13; the CPU test lane
  runs on 3.10, 3.12 and 3.13 (pytest-gpu-proof needs 3.11 or newer, so its two scope
  tests skip on 3.10).
- Error messages: the bindings and `grid-generate` now say the cause and the next action
  (missing nvcc or driver, a URDF that does not parse, a missing `[torch]` or `[jax]`
  extra, a robot built without a framework, a stale or damaged cache entry); exit codes
  and exception classes are unchanged. `test/test_error_messages.py` pins the wording.
- Zero expected skips: the floating-base `end_effector_pose_gradient` and `_hessian`
  cells run by default for every robot (`GRID_CUDA_FLOATING_ALGORITHMS` only narrows);
  the plant and centroidal runners launch at a register-safe 256 threads, which brings
  back the six plant cells and go2 centroidal; the six cells that exceed the GPU's
  shared-memory or register limits at every tier are de-collected with the reason at
  the site. `test/gpu-proof-expected-skips.txt` is empty.
- Receipts by architecture: `test/gpu-proof-receipts.yaml` maps the GPU architecture to
  its receipt file; `gpu-proof.json` (sm_120) keeps gating CI, a Jetson records under
  `receipts/`, `make verify-all` checks every receipt on disk and
  `test/receipt_integrity.py --table` prints compatibility rows.
- Precision: the validated path is fp32 on the GPU. The fp64 tier (`dtype="float64"`) is kept as an
  alpha feature: its tests carry the `fp64_alpha` marker, run with `make test-fp64`, and sit outside
  the signed receipt by collection; one numpy iiwa14 parity test stays in the receipt as a canary.
  Registering an fp64 robot warns once. The dccrba smoke runner now runs in float like every other
  runner, which brings the g1-floating and h1_2-fixed cells back.
- Multi-architecture builds: `GRID_RBD_CUDA_ARCHS=86,89,90,120` compiles one robot
  library for several GPUs (cache keys include the list; launch tuning follows the
  `cuda_arch` given); the Docker image carries iiwa14 built that way.
- Release gate: `tools/release_fresh_install.sh` (`make release-check`) builds the sdist
  and wheel, installs the repaired wheel into a throwaway venv outside the checkout and
  exercises both entry points (`load_robot` on iiwa14 against RBDReference; `grid-generate`,
  `nvcc` and `examples/cuda/validate.py`). RELEASING.md requires it before freezing.
- Install recipes: `install/docker/Dockerfile` and `.github/workflows/docker.yml` publish
  `ghcr.io/a2r-lab/grid` (CUDA 13.2 devel base, grid-rbd from PyPI, `jax[cuda12]`, torch
  cu128, baked iiwa14 header; robots compile on first load); `install/environment.yml`
  (conda) and a uv recipe on the installation page. The Docker image builds in
  CI without a GPU; the 0.5.3 image also passed the RTX 5090 smoke test described
  above. The conda recipe remains unvalidated.
- Tutorials: `examples/cuda/fused_control_step/` fuses a computed-torque control step
  around `inverse_dynamics_device` in one kernel and checks and times it against the
  host-function path (`build_and_time.sh --json`); `examples/jax/ik_gradient_descent.py`
  runs `jax.grad` through `end_effector_pose` with a hand-written descent. Both have site
  pages; `docs/test_doc_examples.py` covers them on CPU.
- Docs: platform claims now state what ran (Linux x86-64, Ubuntu 24.04, CPython 3.12, CUDA
  toolkit 13.2, driver 615.71.09, torch 2.12.0 cu128, jax 0.10.0 with the cuda12 plugin, one
  RTX 5090 sm_120); "CUDA 12 or 13", "jax[cuda13]", "sm_8x remain supported" and "cu128 or
  newer" are replaced by "untested". The compatibility page has a "What we ran" table; Jetson
  Orin (sm_87) and Xavier (sm_72) rows are planned.
- Release figures: the unreferenced `stacked_all` and `grid_composition` outputs are removed
  from `docs/plot_release_figures.py`, the manifest and the tracked assets; the release
  measurements page documents how the figure set is regenerated.
- Benchmarks: the release collector covers the Frax floating-base RNEA and FD gradients
  (Euler-coordinate transport in `frax_floating.py`), the GRiD CUDA end-effector pose
  gradient and the BARD autograd end-effector pose gradient; the CUDA bridge registers one
  operation per context through `init_grid_kernel_attr_<op>_checked` and links with
  `--shared` again; `test/benchmarks/timing_night.sh` runs the gap cells, the release
  re-collection and the fused control-step microbench as one resumable overnight job.
- **Checkout layout:** `config/` is now `grid_assets/` (`robots/` for the sample
  URDFs, `launch_configs/` for the tuned per-GPU profiles). Nothing in the wheel
  changes; only paths inside a checkout (`grid_assets/robots/iiwa14.urdf`).
- Examples default to the vendored sample URDFs; `robot_descriptions` is no longer
  needed to run them.
- Developer install: `install/requirements-dev.txt` pins the Pinocchio oracle
  (`pin<4`), `install/developer_install.sh` installs the GPU frameworks (opt out
  with `SKIP_GPU_FRAMEWORKS=1`); the base install stays NumPy-only.
- Documentation consolidated: the README is a router to the site, the site has one
  story per topic, and the generated header is documented with both init paths.
- Maintainers: publishing a tag now requires a fresh release-policy receipt; the
  Pinocchio peer suites run on every PR.

## [0.5.1] — faster per-robot builds + receipt infrastructure (2026-10-11)

- **Per-robot `robot.so` builds are 4–8x faster**: multi-unit build with a
  per-unit object cache kept forever and `GRID_RBD_BUILD_JOBS` (iiwa14 ~100 s,
  was 10 min; go2-floating ~2 min, was 13; g1 pin-only ~6 min, was 48). A
  wrapper-only change rebuilds only the touched units.
- **Compile-on-first-use**: `get_robot(name, urdf_path=...)` compiles and
  registers on a cache miss with a one-line build estimate; `GRID_RBD_QUIET=1`
  silences it.
- **Cache maintenance**: `python -m grid_rbd.gc_cache` (dry run; `--execute`).
- Opt-in `GRID_RBD_SPLIT_COMPILE=N` (nvcc `--split-compile` on the heavy
  unit), content-keyed; off by default.
- `algorithm_list` closure fixes (`forward_dynamics_parameter_gradient` needs
  `forward_dynamics`; the integrator gradients need `integrator`).
- Reproducible sdist/wheel builds in CI (`SOURCE_DATE_EPOCH`); the PyPI
  digests match the CI `SHA256SUMS`.
- Maintainer-side: CUDA test runners register only the kernels they launch and
  the GPU receipt runs two shards side by side, so a full receipt takes hours
  instead of half a day. The dated engineering log is `docs/design_history/engineering_log.md`.

## [0.5.0] — first PyPI release (alpha)

The first release installable as a package. `pip install grid-rbd` gives the
code generator, the NumPy / JAX / PyTorch bindings and the pinned peer
libraries in one distribution; robot CUDA is still compiled on your machine at
`register_robot` time. The tested boundary is Linux x86-64, CPython 3.10–3.12,
CUDA 12+/13 with one RTX 5090 (sm_120). Jetson-class devices are supported by
design but not certified by this release's receipt.

### What is new since the 2022 paper code

The paper covered fixed-base RNEA, forward dynamics and their first-order
gradients as generated CUDA for a handful of arms. This release is one large
update on top of that, organised here by theme:

- **Packaging.** One `grid-rbd` distribution with `jax`, `torch`, `all` and
  `dev` extras; sdist and manylinux_2_34 wheels for CPython 3.10–3.12; a
  prebuilt CPU `_core` extension. Pinned URDFParser and RBDReference sources,
  GLASS headers, launch profiles and peer licenses are bundled with a
  provenance manifest (`grid_codegen.resources.bundled_provenance()`), so an
  installed package needs neither Git nor a retained checkout.
  `tools/check_installed_package.py` validates an installed artifact from
  outside the checkout, including NumPy / JAX / PyTorch values and autograd
  against the oracle on a GPU.
- **Floating-base support.** Free-flyer root with Pinocchio conventions
  (`q = [x, y, z, qx, qy, qz, qw]`, tangent `[linear; angular]`), a `legacy`
  ordering flag, and run-to-run bit-deterministic floating-base reductions.
- **Python bindings, `grid-rbd`.** `register_robot` / `load_robot` return
  NumPy, JAX (`jax.jit`-able FFI) and `torch.autograd`-aware handles over one
  content-addressed per-robot `.so` cache; runtime contexts with their own
  arenas, streams and launch overrides; torch CUDA-Graphs capture;
  allocate-once host round trips on all three surfaces. Cache keys hash the
  installed generator and parser sources and the GLASS header bytes,
  independent of install location.
- **Second-order derivatives.** IDSVA-SO in body and world frame with a
  codegen-time dispatcher, FDSVA-SO, the end-effector pose Hessian and the
  integrator Hessian.
- **The rest of the dynamics stack.** Direct `Minv`, CRBA, ABA, external-force
  gradients (`f_ext_gradient`, `f_ext_gradient_dq`, contact-frame wrench
  mapping via `contact_frames=`), the inertial-parameter regressor family
  (`Y`, ∂Y/∂(q,v), ∂q̈/∂π, kinetic/potential-energy regressors), and the
  centroidal family (CoM + Jacobian, CCRBA, `dccrba`, CMM time variation,
  Coriolis matrix, energy, gravity, nonlinear effects).
- **Kinematics.** End-effector pose / Jacobian / Hessian, general-frame
  `frame_jacobian` (LOCAL / WORLD / LWA) and its time derivative,
  operational-space inertia, runtime multi-target positions and gradients,
  runtime end-effector targets with per-target offsets.
- **Integrators and the `grid_plant` layer.** Euler / semi-implicit Euler
  steps with analytical gradients, quadratic state/input, CoM and momentum
  costs, joint log-barriers, `plant_step` / `plant_step_gradient` /
  `plant_step_hessian` for trajectory-optimization solvers (GATO, MPCGPU).
- **Joint coverage.** Revolute, prismatic, fixed, floating, planar and
  translation (decomposed), spherical (quaternion), arbitrary skew axes, and
  mimic joints on every algorithm including all gradients and second order.
- **Runtime-mutable model parameters.** Inertias, transforms, joint damping
  and friction, tool/payload welding (`attach_tool` / `tool_fext`) without
  regenerating or recompiling the robot.
- **Collision checking.** URDF collision geometry spherized at codegen time
  into a one- or two-tier covering-sphere model (`grid-spherize`,
  `--collision`), block-cooperative `grid_collision::config_free` with a
  race-free shared hit flag, per-pair distances, a collision cost with
  gradient, constant-memory sphere tables, and warp-scoped `config_free` /
  `collision_distance` for multi-candidate solvers (HJCD-IK).
- **MuJoCo / mjx convention twins.** Every kernel can also be emitted in
  MuJoCo's output convention (`handle.mujoco.<method>`), values and
  derivatives, for drop-in comparison against mjx.
- **Resource tiers and spill.** Shared / lite / minimal shared-memory tiers
  with surgical global-workspace spills so humanoid-scale robots (G1, H1-2,
  H2+) fit on desktop and embedded GPUs; per-GPU tuned launch configurations.
- **Peer libraries as pinned products.** [URDFParser](https://github.com/A2R-Lab/URDFParser),
  [RBDReference](https://github.com/A2R-Lab/RBDReference) (NumPy oracle
  validated against Pinocchio) and [GLASS](https://github.com/A2R-Lab/GLASS)
  (GPU small-block linear algebra) each live in their own repository and are
  pinned by commit; GRiD vendors GLASS into the generated header.
- **Validation.** Every algorithm exists on two surfaces that are tested for
  agreement (NumPy oracle vs generated CUDA); GPU results are recorded in a
  signed [pytest-gpu-proof](https://github.com/A2R-Lab/pytest-gpu-proof)
  receipt that CPU-only CI verifies against the source fingerprint. A release
  requires one fresh full run under the strict release policy with no carried
  shards (`test/run_validation.py --full` / `--finalize`); this release ships
  that receipt.
- **Performance.** Branch-major packing, bit-identical barrier fusions in the
  second-order kernels, allocate-once host round trips, and the published
  [release measurements](https://a2r-lab.org/GRiD/release_measurements.html)
  against Pinocchio, MuJoCo, mjx, MuJoCo Warp, BARD and Frax (including
  Frax's floating-base model and end-effector pose gradients for every library
  that exposes a Jacobian), a no-I/O comparison against the CPU libraries and
  a resident-wrapper group.

### Known limitations

- One compiled artifact per GPU architecture; the model and API carry over
  between machines, the binary does not.
- Collision checking is exposed through the generated CUDA interface, not yet
  through the Python handles.
- `plant_step_hessian` is NumPy/CUDA only; `fk_batched` is NumPy only.
- Closed kinematic loops are unsupported.
- No Windows, macOS, Jetson or precompiled-robot wheels.

[Unreleased]: https://github.com/A2R-Lab/GRiD/compare/v0.5.5...HEAD
[0.5.5]: https://github.com/A2R-Lab/GRiD/compare/v0.5.4...v0.5.5
[0.5.4]: https://github.com/A2R-Lab/GRiD/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/A2R-Lab/GRiD/compare/v0.5.1...v0.5.3
[0.5.2]: https://github.com/A2R-Lab/GRiD/compare/v0.5.1...94f4e5c
[0.5.1]: https://github.com/A2R-Lab/GRiD/releases/tag/v0.5.1
[0.5.0]: https://github.com/A2R-Lab/GRiD/releases/tag/v0.5.0
