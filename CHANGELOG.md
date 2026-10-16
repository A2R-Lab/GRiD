# Changelog

All notable changes to GRiD are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) with the pre-1.0 caveat that minor
versions may break APIs.

GRiD was first released alongside the ICRA 2022 paper *GRiD: GPU-Accelerated
Rigid Body Dynamics with Analytical Gradients*. That implementation is preserved
in the archival [robot-acceleration/GRiD](https://github.com/robot-acceleration/GRiD)
repository. Everything below describes the A2R-Lab rebuild that leads to the
first packaged release.

## [Unreleased]

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
  (conda) and a uv recipe on the installation page. Neither recipe is tested in CI.
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

[0.5.1]: https://github.com/A2R-Lab/GRiD/releases/tag/v0.5.1
[0.5.0]: https://github.com/A2R-Lab/GRiD/releases/tag/v0.5.0
