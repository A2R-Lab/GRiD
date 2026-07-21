# GRiD
[![CI](https://img.shields.io/github/actions/workflow/status/A2R-Lab/GRiD/verify-gpu-proof.yml?branch=main&style=flat-square&label=CI)](https://github.com/A2R-Lab/GRiD/actions/workflows/verify-gpu-proof.yml)
[![docs](https://img.shields.io/github/actions/workflow/status/A2R-Lab/GRiD/gh-pages.yml?branch=main&style=flat-square&label=docs)](https://a2r-lab.github.io/GRiD/)
[![license](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)](pyproject.toml)
[![agent-ready](https://img.shields.io/badge/agent--ready-CLAUDE.md-8A2BE2?style=flat-square)](CLAUDE.md)
[![All Contributors](https://img.shields.io/github/all-contributors/A2R-Lab/GRiD?color=ee8449&style=flat-square)](#contributors)

A GPU-accelerated library for computing rigid body dynamics with analytical gradients.

GRiD builds on our [URDFParser](https://github.com/A2R-Lab/URDFParser), [RBDReference](https://github.com/A2R-Lab/RBDReference), and [GLASS](https://github.com/A2R-Lab/GLASS) packages (URDF parsing, Pinocchio-validated reference dynamics, and GPU linear algebra), together with its own bundled code generator. Using its scripts, users can easily generate and test optimized rigid body dynamics CUDA C++ code for their URDF files.

For additional information and links to our paper on this work, check out our [project website](https://brianplancher.com/publication/GRiD).

**This package contains submodules make sure to run ```git submodule update --init --recursive```** after cloning!

![The GRiD library package ecosystem, showing how a user's URDF file can be transformed into optimized CUDA C++ code which can then be validated against reference outputs and benchmarked for performance.](docs/imgs/GRiD.png)

## Quick Start

Install (creates a local venv and registers the `grid-generate` CLI):
```shell
bash install/base_install.sh
source .venv/bin/activate
```

Generate CUDA code for your robot:
```shell
# Via the installed CLI:
grid-generate path/to/robot.urdf [-t EE_JOINT_NAME] [-n NAMESPACE] [-f]

# Or via a hardcoded zero-config example:
python examples/codegen/generate_iiwa14.py       # iiwa14 fixed base
python examples/codegen/generate_go2_floating.py # Go2 floating base
```

Validate and debug:
```shell
# Print CPU reference values for all algorithms:
python examples/codegen/print_reference_values.py path/to/robot.urdf

# Compile and run the CUDA print kernel (requires nvcc):
python examples/codegen/print_grid.py path/to/robot.urdf
```

Write your own CUDA kernel against the generated header:
```shell
# Step-by-step walkthrough + compiling/validated example kernels:
#   examples/cuda/README.md   (and examples/cuda/wrapper_types.md)
bash examples/cuda/build_and_validate.sh   # generate → nvcc → run → validate
```

> **Requires a C++17-capable host compiler** (e.g., g++ ≥ 7, clang++ ≥ 5).
> The benchmark and codegen runtime compile with `-std=c++17` — needed for
> inline variables in the bench common header.

## Usage
+ `grid-generate PATH_TO_URDF` — generate `grid.cuh`; add `-d` for full debug mode, `-f` for floating base, `-t JOINT_NAME` to target a specific end-effector joint
+ `python examples/codegen/print_reference_values.py PATH_TO_URDF` — print CPU reference values for all algorithms to validate CUDA output
+ `python examples/codegen/print_grid.py PATH_TO_URDF` — compile and run the CUDA print kernel against the generated header

## Floating-Base Conventions
Floating-base parsing and the Python reference path now accept a public
floating-base convention flag. The default is Pinocchio-compatible:

+ `floating_base_convention="pinocchio"`:
  `q = [x, y, z, qx, qy, qz, qw]`,
  `v = [vx, vy, vz, wx, wy, wz]`
+ `floating_base_convention="legacy"`:
  `q = [x, y, z, qw, qx, qy, qz]`,
  `v = [wx, wy, wz, vx, vy, vz]`

GRiD normalizes both public conventions into one shared internal floating-base
representation, so the code generator and `RBDReference` stay consistent under
the hood while callers can choose the input/output ordering they need.

## Developer Testing
The Pinocchio-side floating convention regression suite exercises both public
floating-base orderings across the current floating robot manifest:

```bash
.venv/bin/python -m pytest RBDReference/tests/test_floating_base_conventions.py -q
```

The CUDA executable equivalence suite still defaults to the Pinocchio-facing
floating convention and is currently being expanded from the first floating-base
smoke slice to broader floating algorithm coverage.

For floating CUDA development, the pytest harness also accepts optional env
overrides:

+ `GRID_CUDA_FLOATING_ALGORITHMS=all` to try the broader floating candidate set
+ `GRID_CUDA_FLOATING_ALGORITHMS=inverse_dynamics,forward_dynamics` to request a subset
+ `GRID_CUDA_FLOATING_SAMPLE_NAMES=all` to run every deterministic/random sample instead of only `zero`

Generated CUDA defaults to a 96 KiB dynamic shared-memory target
(`GRID_CUDA_TARGET_SHARED_MEM_BYTES=98304`) and selects spill fallbacks only
when the generated arena would exceed that target. See
[`docs/source/user_guide/tutorials/cuda_validation.rst`](docs/source/user_guide/tutorials/cuda_validation.rst)
for shared-memory target overrides, L2 controls, and the recommended
ptxas/register-pressure analysis workflow for tuning a specific robot/GPU.

## Current Support
GRiD currently fully supports any robot model consisting of revolute, prismatic, and fixed joints that does not have closed kinematic loops. Arbitrary/skew joint axes (a non-cardinal `<axis>`) are also supported via a dense 6-vector motion subspace — currently for `inverse_dynamics` and `crba` only (cardinal-axis robots stay byte-identical; other algorithms and the helical/planar/spherical joint types are later stages).

GRiD currently implements the following rigid body dynamics algorithms:
+ Inverse Dynamics via the Recursive Newton Euler Algorithm (RNEA) from [Featherstone](https://link.springer.com/book/10.1007/978-1-4899-7560-7)
+ Composite Rigid Body Algorithm (CRBA) for the joint-space mass matrix and the Articulated Body Algorithm (ABA) for forward dynamics, both from [Featherstone](https://link.springer.com/book/10.1007/978-1-4899-7560-7)
+ The Direct Inverse of Mass Matrix from [Carpentier](https://www.researchgate.net/publication/343098270_Analytical_Inverse_of_the_Joint_Space_Inertia_Matrix)
+ Forward Dynamics by combining the above algorithms as qdd = -M^{-1}(u-RNEA(q,qd,0))
+ Analytical Gradients of Inverse Dynamics from [Carpentier](https://hal.archives-ouvertes.fr/hal-01790971)
+ Analytical Gradient of Forward Dynamics from [Carpentier](https://hal.archives-ouvertes.fr/hal-01790971)
+ End-effector pose, pose gradient (Jacobian), and pose Hessian
+ General-frame geometric Jacobian for an arbitrary target frame in any of the three Pinocchio reference frames (`LOCAL`, `WORLD`, `LOCAL_WORLD_ALIGNED`). The numpy reference additionally provides the Jacobian time-variation J̇ and the operational-space (OSC) inertia Λ = (J·M⁻¹·Jᵀ)⁻¹ — all validated against Pinocchio's `getFrameJacobian`/`getJointJacobian`, `computeJointJacobiansTimeVariation`, and `(J·M⁻¹·Jᵀ)⁻¹`. CUDA codegen emits all three as opt-in keys — J (`frame_jacobian`), J̇ (`frame_jacobian_dot`), and Λ (`osc_inertia`) — each validated on-device against the numpy reference across the three frames (Λ is self-contained: it composes M⁻¹ on-device). All three additionally have the full launchable surface (batched `*_kernel` + 3-mode host writing the `gridData` `d_frame_jacobian` / `d_frame_jacobian_dot` / `d_osc_inertia` buffers), so they are benchmarkable + bindable; the launchable surface bakes the leaf-EE target + `LOCAL_WORLD_ALIGNED` frame, while the `*_device` functions stay the arbitrary-target/-frame entry points
+ Second-Order Inverse Dynamics (IDSVA-SO) from [Singh, Russell, & Wensing](https://arxiv.org/abs/2302.06001) — both body-frame and world-frame variants. A codegen-time dispatcher picks body-frame for fixed-base (multi-pass amortizes, ~30× faster) and world-frame for floating-base (single-pass + no gravity shim, 2–4× faster)
+ Second-Order Forward Dynamics (FDSVA-SO) from [Singh, Russell, & Wensing](https://arxiv.org/abs/2302.06001) on both fixed and floating bases
+ A **time-integrator** family: the discrete step `x_{k+1}` plus its gradient `∂x_{k+1}/∂(x,u)` and a fused value-and-gradient variant
+ Optional per-body **external forces** (`f_ext`), threaded through RNEA, forward dynamics, ABA, and the inverse-/forward-dynamics gradients. Opt-in (a `nullptr`/empty default reproduces the no-force path exactly), supplied in the body-local frame (`6*NUM_BODIES`, body-major) and subtracted from the per-body force.
+ **External-force gradients**: `∂tau/∂f_ext = -Jᵀ` and `∂q̈/∂f_ext = M⁻¹Jᵀ`, plus the fixed-base `∂(inverse_dynamics_gradient)/∂f_ext = -∂Jᵀ/∂q`
+ A trajectory-optimization-oriented **`grid_plant` layer** (emitted as a sibling `grid_plant` namespace): a `plant_step` integrator wrapper, quadratic state/input costs, an end-effector position cost (with Gauss-Newton Hessian), and joint position/velocity/torque log-barriers.
+ The **Coriolis matrix** `C(q,q̇)` (with `C·q̇ + g(q) = nonlinear_effects`)
+ **Inertial-parameter energy regressors**: kinetic `y_KE` and potential `y_PE` (each length `10·NB`, with `KE = y_KE·π` and `PE = y_PE·π`)
+ The **centroidal derivatives**: `dccrba` (the ∂A/∂q tensor, 6×NV×NV) and `cmm_time_variation` (the centroidal-momentum-matrix time variation Ȧ)
+ **Runtime arbitrary multi-EE pose / pose-gradient** (`end_effector_pose_runtime` + `_gradient`): the end-effector target joint id and a per-target offset become runtime arguments instead of codegen-baked, so one compiled robot serves any leaf/target frame.
+ **Runtime-mutable inertial parameters** (flag-gated): a `set_inertia_params` device entry mutates an on-device parameter table (sysID / domain randomization) with no recompile; the baked default path is byte-identical.

`RBDReference` additionally provides numpy reference oracles — validated against [Pinocchio](https://github.com/stack-of-tasks/pinocchio) — for generalized gravity, nonlinear effects, kinetic/potential/mechanical energy, the Coriolis matrix, the centroidal quantities (CoM, CoM Jacobian, CCRBA, centroidal momentum) and their derivatives (the analytic `dccrba` ∂A/∂q tensor — replacing the prior finite-difference oracle — and `cmm_time_variation` Ȧ), the inverse-dynamics and kinetic/potential-energy regressors, the general-frame Jacobian / J̇ / OSC inertia described above, and the plant/cost/barrier layer above.

**Dual-surface equivalence.** Every algorithm exists on two surfaces that are tested for numerical agreement: the `RBDReference` numpy implementation (the oracle, checked against Pinocchio) and the generated CUDA C++ kernels (checked against that same numpy reference). This keeps the GPU codegen honest against an independent, Pinocchio-validated baseline.

**Mimic-joint support:** per-robot gating is now essentially eliminated. Non-gradient algorithms (RNEA, forward dynamics, ABA, CRBA, …) work for robots with mimic joints, and every **gradient** emits a correct mimic-reduced result on **both** the fixed and floating base: `inverse_dynamics_gradient`/`forward_dynamics_gradient`, `end_effector_pose_gradient`/`end_effector_pose_hessian`, the second-order `idsva_so`/`fdsva_so`, the external-force gradients (`f_ext_gradient`), and the integrator gradients. The centroidal family — `com`, `ccrba`, `energy`, and the centroidal derivatives `dccrba`/`cmm_time_variation` — now also runs on mimic robots (the per-body Jacobian and per-unit motion columns carry the mimic multiplier α, validated against the mimic-aware reference). `dccrba`/`cmm_time_variation` additionally run on big floating-base robots (e.g. `g1`/`h1_2`-floating) via the sweep-pool spill path. No algorithm raises `NotImplementedError` for mimic robots anymore.

Additional algorithms and features are in development. If you have a particular algorithm or feature in mind please let us know by posting a GitHub issue. We'd also love your collaboration in implementing the Python reference implementation of any algorithm you'd like implemented!

## C++ API
For each algorithm GRiD emits four layers: `*_inner` (core math on
shared-mem inputs), `*_device` (allocates scratch + calls `_inner`),
`*_kernel` (global entry point with batched timestep loop), and the
host wrapper (CPU launcher with H↔D copies). See the
[codegen architecture docs](docs/source/user_guide/concepts/codegen_architecture.rst)
for the rationale and concrete signatures.

## Python API (`grid-rbd`)

For Python users the `grid-rbd` package (in [`bindings/`](bindings/)) wraps
the per-robot codegen behind a register-then-run UX with `numpy`, `jax`,
and `torch` backends. It ships as part of the single repo distribution — a
`pip install -e .` (what `install/base_install.sh` runs) installs the codegen
toolkit *and* the `grid_rbd` wrapper together. The base install is minimal;
pick a backend extra for the surface you want:

```bash
pip install -e "."          # base: numpy handle only
pip install -e ".[jax]"     # + JAX FFI surface
pip install -e ".[torch]"   # + torch backend (CUDA wheel matching your GPU arch)
pip install -e ".[all]"     # jax + torch
```

See the [install matrix in `bindings/README.md`](bindings/README.md#install-editable-from-a-grid-checkout)
for what each extra unlocks (and the torch CUDA-wheel note).

```python
import grid_rbd

# numpy (default), jax, or torch; urdf_string= also accepted instead of urdf_path
handle = grid_rbd.register_robot("iiwa14", urdf_path="iiwa.urdf", backend="torch")

qdd = handle.forward_dynamics(q, qd, u)   # autograd-aware torch.Tensor
qdd.sum().backward()                      # gradients flow to q, qd, u
```

The `torch` backend exposes autograd-aware `inverse_dynamics` / `forward_dynamics` /
`aba` / `integrator` (analytic backward passes) plus CUDA-Graphs capture,
and the handle also surfaces the `grid_plant` cost/barrier methods. `inverse_dynamics`
(alias `rnea`) / `forward_dynamics` (alias `fd`) take an optional `qdd=` (the
autograd gradient is qdd-aware, returning the correct ∂τ/∂(q,q̇) including the
∂(M·q̈)/∂q term), and the numpy surface adds the new value ops `coriolis_matrix`,
`kinetic_energy_regressor`, `potential_energy_regressor`, `dccrba`, and
`cmm_time_variation`. Pass `allow_fp64=True` at `register_robot` for an
fp64-in/fp64-out convenience cast (compute stays fp32). See
[`bindings/README.md`](bindings/README.md) and the
[Python wrappers docs](docs/source/user_guide/tutorials/python_wrappers.rst).

## Citing GRiD
To cite GRiD in your research, please use the following bibtex for our paper ["GRiD: GPU-Accelerated Rigid Body Dynamics with Analytical Gradients"](https://brianplancher.com/publication/grid/):
```
@inproceedings{plancher2022grid,
  title={GRiD: GPU-Accelerated Rigid Body Dynamics with Analytical Gradients}, 
  author={Brian Plancher and Sabrina M. Neuman and Radhika Ghosal and Scott Kuindersma and Vijay Janapa Reddi},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)}, 
  year={2022}, 
  month={May}
}
```

## Performance
When performing multiple computations of rigid body dynamics algorithms, GRiD provides as much as a 7.6x speedup over a state-of-the-art, multi-threaded CPU implementation, and maintains as much as a 2.6x speedup when accounting for I/O overhead. 

![Latency (including GPU I/O overhead) for N = 16, 32, 64, 128, and 256 computations of the gradient of forward dynamics for both the Pinocchio CPU baseline and the GRiD GPU library for various robot models (IIWA, HyQ, and Atlas). Overlayed is the speedup (or slowdown) of GRiD as compared to Pinocchio both in terms of pure computation and including I/O overhead.](docs/imgs/benchmark_multi_fd_grad.png)

To learn more about GRiD's performance results and to run your own benchmark analysis please see [`test/benchmarks/`](test/benchmarks/) and our [paper](https://brianplancher.com/publication/GRiD/).

## Installation
The Quick Start above covers the common-case install. For CUDA Toolkit
setup, developer dependencies (Pinocchio, robot_descriptions, benchmarks),
and Docker, see the full
[installation guide](docs/source/user_guide/getting_started/installation.rst).

## Troubleshooting

### Bench harness `nvcc` hangs on floating-base kernels (sm_8x)

On Ampere (sm_86 / CUDA 12.6) the bench harness can wedge `nvcc` /
`ptxas` at 100 % CPU when compiling heavy floating-base GRiD harnesses.
Pass `--ptxas-opt-level 2` to `test/benchmarks/run_multi_version.py` — it
forwards `-Xptxas -O2` to floating-base compiles only. Blackwell (sm_120) does
not hit this. Typical user code that includes `grid.cuh` and calls the
batch host wrappers (e.g. `grid::forward_dynamics<T>(...)`) does not
trigger the hang — it's specific to the timing-bench template surface.


## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/kawotwi"><img src="https://avatars.githubusercontent.com/kawotwi?s=100" width="100px;" alt="Kwamena A"/><br /><sub><b>Kwamena A</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/plancherb1"><img src="https://avatars.githubusercontent.com/plancherb1?s=100" width="100px;" alt="Brian Plancher"/><br /><sub><b>Brian Plancher</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Z4KH"><img src="https://avatars.githubusercontent.com/Z4KH?s=100" width="100px;" alt="Zachary Pestrikov"/><br /><sub><b>Zachary Pestrikov</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/b619b0ff13333ce2a22bb110eda8f7a9?d=identicon&s=100?s=100" width="100px;" alt="Danelle Tuchman"/><br /><sub><b>Danelle Tuchman</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/anncli"><img src="https://avatars.githubusercontent.com/anncli?s=100" width="100px;" alt="Ann Li"/><br /><sub><b>Ann Li</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/naren-loganathan"><img src="https://avatars.githubusercontent.com/naren-loganathan?s=100" width="100px;" alt="Naren Loganathan"/><br /><sub><b>Naren Loganathan</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/EmreAdabag"><img src="https://avatars.githubusercontent.com/EmreAdabag?s=100" width="100px;" alt="EmreAdabag"/><br /><sub><b>EmreAdabag</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/emilyburnett2003"><img src="https://avatars.githubusercontent.com/emilyburnett2003?s=100" width="100px;" alt="emilyburnett2003"/><br /><sub><b>emilyburnett2003</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/pruyontrarakk"><img src="https://avatars.githubusercontent.com/pruyontrarakk?s=100" width="100px;" alt="pruyontrarakk"/><br /><sub><b>pruyontrarakk</b></sub></a><br /></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
