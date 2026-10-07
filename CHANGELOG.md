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

## [0.5.0] - 2026-10 (first PyPI release, alpha)

The first release installable as a package. `pip install grid-rbd` gives the
code generator, the NumPy / JAX / PyTorch bindings and the pinned peer
libraries in one distribution; robot CUDA is still compiled on your machine at
`register_robot` time. The tested boundary is Linux x86-64, CPython 3.10–3.12,
CUDA 12+/13 with one RTX 5090 (sm_120). Jetson-class devices are supported by
design but not certified by this release's receipt.

### Compared with the 2022 paper code

The paper covered fixed-base RNEA, forward dynamics and their first-order
gradients as hand-generated CUDA for a handful of arms. Since then GRiD has
grown into a full dynamics, kinematics and collision stack with a Python
front end. The headline additions, roughly in the order they landed:

- **Floating-base support** (2024–2025): free-flyer root with Pinocchio
  conventions (`q = [x, y, z, qx, qy, qz, qw]`, tangent `[linear; angular]`),
  a `legacy` ordering flag, and run-to-run bit-deterministic floating-base
  reductions.
- **Python bindings, `grid-rbd`** (2025): `register_robot` / `load_robot`
  return NumPy, JAX (`jax.jit`-able FFI) and `torch.autograd`-aware handles
  over one content-addressed per-robot `.so` cache; runtime contexts with
  their own arenas, streams and launch overrides; torch CUDA-Graphs capture.
- **Second-order derivatives** (2025): IDSVA-SO in body and world frame with a
  codegen-time dispatcher, FDSVA-SO, the end-effector pose Hessian and the
  integrator Hessian.
- **The rest of the dynamics stack**: direct `Minv`, CRBA, ABA, external-force
  gradients (`f_ext_gradient`, `f_ext_gradient_dq`, contact-frame wrench
  mapping via `contact_frames=`), the inertial-parameter regressor family
  (`Y`, ∂Y/∂(q,v), ∂q̈/∂π, kinetic/potential-energy regressors), and the
  centroidal family (CoM + Jacobian, CCRBA, `dccrba`, CMM time variation,
  Coriolis matrix, energy, gravity, nonlinear effects).
- **Kinematics**: end-effector pose / Jacobian / Hessian, general-frame
  `frame_jacobian` (LOCAL / WORLD / LWA) and its time derivative,
  operational-space inertia, runtime multi-target positions and gradients,
  runtime end-effector targets with per-target offsets.
- **Integrators and the `grid_plant` layer**: Euler / semi-implicit Euler
  steps with analytical gradients, quadratic state/input, CoM and momentum
  costs, joint log-barriers, `plant_step` / `plant_step_gradient` /
  `plant_step_hessian` for trajectory-optimization solvers (GATO, MPCGPU).
- **Joint coverage**: revolute, prismatic, fixed, floating, planar and
  translation (decomposed), spherical (quaternion), arbitrary skew axes, and
  mimic joints on every algorithm including all gradients and second order.
- **Runtime-mutable model parameters**: inertias, transforms, joint damping
  and friction, tool/payload welding (`attach_tool` / `tool_fext`) without
  regenerating or recompiling the robot.
- **Collision checking** (2026): URDF collision geometry spherized at codegen
  time into a one- or two-tier covering-sphere model (`grid-spherize`,
  `--collision`), block-cooperative `grid_collision::config_free`, per-pair
  distances, a collision cost with gradient, and warp-scoped
  `config_free` / `collision_distance` for multi-candidate solvers (HJCD-IK).
- **MuJoCo / mjx convention twins**: every kernel can also be emitted in
  MuJoCo's output convention (`handle.mujoco.<method>`), values and
  derivatives, for drop-in comparison against mjx.
- **Resource tiers and spill**: shared / lite / minimal shared-memory tiers
  with surgical global-workspace spills so humanoid-scale robots (G1, H1-2,
  H2+) fit on desktop and embedded GPUs; per-GPU tuned launch configurations.
- **Peer libraries as pinned products**: [URDFParser](https://github.com/A2R-Lab/URDFParser),
  [RBDReference](https://github.com/A2R-Lab/RBDReference) (NumPy oracle
  validated against Pinocchio) and [GLASS](https://github.com/A2R-Lab/GLASS)
  (GPU small-block linear algebra) each live in their own repository and are
  pinned by commit; GRiD vendors GLASS into the generated header.
- **Validation**: every algorithm exists on two surfaces that are tested for
  agreement (NumPy oracle vs generated CUDA); GPU results are recorded in a
  signed [pytest-gpu-proof](https://github.com/A2R-Lab/pytest-gpu-proof)
  receipt that CPU-only CI verifies against the source fingerprint. This
  release ships a fresh full receipt: 44 shards, 1,541 tests, 1,456 passed,
  85 expected skips, no carried shards.
- **Performance work** (2026): branch-major packing, bit-identical barrier
  fusions in the second-order kernels (12–33 % faster Hessian cells),
  allocate-once host round trips for all three Python surfaces, and the
  published [release measurements](https://a2r-lab.org/GRiD/release_measurements.html)
  against Pinocchio, MuJoCo, mjx and MuJoCo Warp.

### Added in this release

- `grid-rbd` packaging: one distribution with `jax`, `torch`, `all` and `dev`
  extras; sdist and manylinux_2_34 wheels; a prebuilt CPU `_core` extension.
  Pinned URDFParser and RBDReference sources, GLASS headers, launch profiles
  and peer licenses are bundled with a provenance manifest
  (`grid_codegen.resources.bundled_provenance()`), so an installed package
  needs neither Git nor a retained checkout.
- `tools/check_installed_package.py` validates an installed artifact from
  outside the checkout (imports, provenance hashes, CPU generation, and with
  `--gpu` NumPy / JAX / PyTorch values and autograd against the oracle).
- `test/run_validation.py --full` / `--finalize`: the fresh-full release
  receipt flow, verification-only finalization on the clean descendant.
- Collision `config_free`: shared hit flag written with `atomicExch` and a
  single-writer debug counter; adversarial many-hit tests in the receipt scope.
- Warp-scoped collision entry points and constant-memory sphere tables for
  HJCD-IK (asks G1–G4).
- Figure 2b of the release measurements: GRiD kernel versus Pinocchio and
  MuJoCo CPU with no I/O; resident (no-I/O) wrapper group in the wrapper
  figure.

### Changed

- Registration cache keys now hash the installed generator and parser sources
  and the GLASS header bytes, independent of install location. Existing
  wrapper caches rebuild once after upgrading.
- `pyproject.toml` uses PEP 639 license metadata; the project homepage is
  <https://a2r-lab.org/GRiD/>.

### Known limitations

- One compiled artifact per GPU architecture; the model and API carry over
  between machines, the binary does not.
- Collision checking is exposed through the generated CUDA interface, not yet
  through the Python handles.
- `plant_step_hessian` is NumPy/CUDA only; `fk_batched` is NumPy only.
- Closed kinematic loops are unsupported.
- No Windows, macOS, Jetson or precompiled-robot wheels.

[0.5.0]: https://github.com/A2R-Lab/GRiD/releases/tag/v0.5.0
