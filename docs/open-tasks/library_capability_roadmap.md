# Library Capability Roadmap — what GRiD should add next

**Status:** research + planning (read-only survey 2026-05-30, branch `modernizing-tests`).
**Scope:** Survey of mature RBD / robotics-physics libraries, mapped against GRiD's
*current* codegen surface, with a ruthlessly prioritized roadmap for GRiD's niche:
**GPU-batched, codegen'd, differentiable RBD primitives for control / learning / MPC**
(competes with Pinocchio on per-call latency and on batch throughput; integrates via
JAX/pybind, soon PyTorch).

This is the **umbrella roadmap**. It does NOT restate the in-flight specs; it
cross-references them and slots their work into the bigger picture:
- `differentiability_extensions_plan.md` — the `du × {π, f_ext}` differentiability matrix.
- `d4_runtime_inertia_params_plan.md` — runtime inertia (π column) + joint-torque regressor `Y`.
- `d3_pytorch_cudagraphs_plan.md` — PyTorch bindings + CUDA-Graphs callable.
- `urdf_feature_matrix.md` — joint-model / URDF parser gaps (mimic CUDA, planar, skew axes).
- `rbdreference_split_plan.md`, `bc_cleanup_plan.md`, `notebook_examples_plan.md` — hygiene/docs.

---

## 0. GRiD's exact current surface (so nothing below is mis-flagged)

Grounded in `GRiDCodeGenerator/algo_registry.py:41-89`, the
`GRiDCodeGenerator/algorithms/_*.py` file set, and `RBDReference/RBDReference.py`.

**HAS (codegen + reference, benchmarked):**
- Core dynamics: `inverse_dynamics` (RNEA), `forward_dynamics` (Minv+RNEA), `aba`,
  `crba` (M, with floating-base composite-inertia path), `direct_minv` (M⁻¹).
- 1st-order: `inverse_dynamics_gradient` (`id_du` = ∂τ/∂q,q̇),
  `forward_dynamics_gradient` (`fd_du` = ∂q̈/∂q,q̇).
- 2nd-order: `idsva_so` (dispatched body/world), `fdsva_so`.
- Kinematics: `end_effector_pose`, `end_effector_pose_gradient` (**this is already a
  geometric / spatial / LOCAL_WORLD_ALIGNED frame Jacobian** — see
  `algorithms/_eepose_gradient_hessian.py:406-454`, RBD ref `RBDReference.py:1073`),
  `end_effector_pose_hessian` (+ analytic variant, `RBDReference.py:1206/1276`).
- Integrators: `integrator`, `integrator_gradient`, `integrator_with_gradient`
  (Euler / semi-implicit / RK; Butcher table at `RBDReference.py:317`).
- Lie-group pieces already present in the reference: `integrate` / `dIntegrate`
  (`RBDReference.py:235/268`), SE(3)/SO(3) exp + right-Jacobian
  (`RBDReference.py:153-234`), quaternion ops (`RBDReference.py:84-144`).
- Plant (T6, `grid_plant` namespace, `algorithms/_plant.py`): `plant_step` +
  `_gradient`, `quadratic_state_cost` / `quadratic_input_cost` (value/grad/GN-hess),
  `ee_pos_cost` (value/grad/GN-hess = JᵀWJ), `plant_barriers` (log-barrier joint limits).

**PLANNED (specced in `docs/open-tasks/`, mark NOT missing):**
- π column: runtime inertia + **joint-torque regressor `Y = ∂τ/∂π`** and `∂q̈/∂π = −M⁻¹Y`
  — `d4_runtime_inertia_params_plan.md`.
- f_ext column: `ID/FD(...,f_ext)` value + `∂/∂f_ext = ∓Jᵀ / M⁻¹Jᵀ` and `∂(id_du)/∂f_ext`
  — `differentiability_extensions_plan.md §A`.
- `du × π` 2nd-order cell (`∂Y/∂x`) — `differentiability_extensions_plan.md §B`.
- PyTorch custom ops + CUDA-Graphs callable — `d3_pytorch_cudagraphs_plan.md`.
- Mimic-joint CUDA codegen, planar joint, skew axis — `urdf_feature_matrix.md`.

---

## 1. Comparison libraries surveyed

- **Pinocchio 3.x** (primary; CPU-centric, analytic derivatives, the validation oracle).
  Full algorithm families: core dynamics + analytic derivatives; **centroidal**
  (`ccrba`, `computeCentroidalMomentum[TimeVariation]`, `computeCentroidalMap`,
  `computeCentroidalDynamicsDerivatives` / `getCentroidalDynamicsDerivatives` —
  verified signatures, ROS jazzy docs / `expose-centroidal-derivatives.cpp`);
  **frame algorithms** (`getFrameJacobian` LOCAL / WORLD / LOCAL_WORLD_ALIGNED,
  `frameJacobianTimeVariation`, `getFrameAcceleration[Derivatives]`); **CoM**
  (`centerOfMass` + `jacobianCenterOfMass`); **constrained dynamics**
  (`constraintDynamics`, `contactABA`, `impulseDynamics`, the `Delassus` operator
  J M⁻¹ Jᵀ, KKT, and their derivatives — Sathya & Carpentier T-RO 2025);
  `computeCoriolisMatrix`, `computeGeneralizedGravity`, `nonLinearEffects`,
  `computeKineticEnergy` / `computePotentialEnergy` / `computeMechanicalEnergy`;
  **Lie-group config ops** (`integrate`/`difference`/`interpolate`/`distance`/
  `randomConfiguration`/`neutral`/`normalize` + `dIntegrate`/`dDifference`);
  **regressors** (`computeJointTorqueRegressor`, `bodyRegressor`, `computeStaticRegressor`,
  `computeKineticEnergyRegressor`); rich **joint models** (spherical / planar / helical /
  free-flyer / translation / mimic / continuous / composite); `buildReducedModel`;
  geometry/collision via **coal** (ex HPP-FCL).
- **frax — "Fast Robot Kinematics and Dynamics in JAX"** (arXiv 2604.04310; the *most
  direct competitor* — same niche: GPU-batched, autodiff, JAX-native). Exposes RNEA,
  ABA, CRBA, Minv, **centroidal dynamics**, spatial+analytical **Jacobians**, **CoM
  kinematics**, **kinetic/potential/total energy**, SE(3) Lie ops; 1st-order
  derivatives come "for free" via JAX autodiff. Claims low-µs CPU latency and
  >100M dynamics evals/s on GPU at batch. **Does NOT** ship: analytic 2nd-order
  (Hessians), contact/constraint solvers, MPC/control primitives. Differentiates from
  GRiD by relying on XLA autodiff rather than emitting *analytic* gradient kernels.
- **Brax / MJX / MuJoCo-Playground / JaxSim** (JAX differentiable *simulators*,
  SIM-oriented). Differentiable sim **step**, actuator / PD / motor models, **soft
  joint-limit & contact constraints**, batched/vmapped envs, **domain randomization**,
  RL integration. Relevant to GRiD only where its integrator+plant grow toward a
  *minimal differentiable simulator* — GRiD is RBD-primitive-oriented, not a full sim.
- **MuJoCo / Drake / RBDL distinctives** (inform, not core): contact + friction-cone
  solvers, analytic/numeric **IK solvers**, MPC primitives & QP layers, transmissions /
  tendons / actuator dynamics, **sensors** (Drake `MultibodyPlant`, RBDL `Model`).

---

## 2. Capability matrix

Columns: **Pin** = Pinocchio, **JAXphys** = frax / Brax / MJX (frax for RBD prims,
Brax/MJX for sim), **GRiD** = current status, **Val** = value for GRiD's
control/learning/MPC niche (H/M/L), **Reuse** = existing GRiD machinery, **Eff** = rough effort.

| Capability | Pin | JAXphys | GRiD status | Val | Reuse (file:line) | Eff |
|---|---|---|---|---|---|---|
| RNEA / ID | ✓ | ✓ | **HAS** | — | `_inverse_dynamics.py` | — |
| FD (Minv+RNEA) / ABA | ✓ | ✓ | **HAS** | — | `_forward_dynamics.py`, `_aba.py` | — |
| CRBA (M) | ✓ | ✓ | **HAS** | — | `_crba.py` (composite IC) | — |
| Minv (M⁻¹) | ✓ | ✓ | **HAS** | — | `_direct_minv.py` | — |
| ∂ID/∂x, ∂FD/∂x (analytic 1st) | ✓ | autodiff | **HAS** | — | `_inverse_dynamics_gradient.py`, `_forward_dynamics_gradient.py` | — |
| 2nd-order ID/FD (analytic SO) | ✓ | ✗ | **HAS** (GRiD edge) | — | `_idsva_so.py`, `_fdsva_so.py` | — |
| EE pose / fwd-kin frame placement | ✓ | ✓ | **HAS** | — | `_eepose_gradient_hessian.py:34` | — |
| Frame/geometric Jacobian (LWA) | ✓ | ✓ | **HAS** (EE anchor) | — | `_eepose_gradient_hessian.py:406` | — |
| EE-pose Hessian (2nd-order kin) | ~ | ✗ | **HAS** (GRiD edge) | — | `RBDReference.py:1206/1276` | — |
| Integrators + ∂/∂x,u | ✓ | ✓ | **HAS** | — | `_integrator*.py`, Butcher `RBDReference.py:317` | — |
| Lie integrate / dIntegrate | ✓ | ✓ | **HAS (ref)** / codegen partial | M | `RBDReference.py:235/268` | S |
| Cost / barrier / plant-step prims | ✗ | (Brax) | **HAS** (`grid_plant`) | — | `_plant.py` | — |
| Runtime inertia + regressor `Y=∂τ/∂π` | ✓ | ✗ | **PLANNED-D.4** | H | `d4_runtime_inertia_params_plan.md` | — |
| f_ext value + ∂/∂f_ext (=∓Jᵀ, M⁻¹Jᵀ) | ✓ | ~ | **PLANNED-T4/§A** | H | `differentiability_extensions_plan.md §A` | — |
| PyTorch ops + CUDA-Graphs | (ext) | (torch) | **PLANNED-D.3** | H | `d3_pytorch_cudagraphs_plan.md` | — |
| **Generalized gravity `g(q)`** | ✓ | ✓ | **PARTIAL** (RNEA q̇=q̈=0) | H | RNEA gravity sweep `_inverse_dynamics.py:122` | **S** |
| **Coriolis/centrifugal `c(q,q̇)`, nonLinearEffects** | ✓ | ✓ | **PARTIAL** (RNEA q̈=0) | H | RNEA, `vxIv` `RBDReference.py:857` | **S** |
| **Kinetic / potential / mechanical energy** | ✓ | ✓ | **MISSING** | H | M (`crba`) ; gravity sweep | **S** |
| **Coriolis matrix `C(q,q̇)`** | ✓ | ✗ | **MISSING** | M | `id_du` ∂τ/∂q̇ machinery | M |
| **CoM + CoM Jacobian** | ✓ | ✓ | **MISSING** | M | CRBA composite IC `_crba.py:232` | M |
| **Centroidal: CMM `A(q)`, h, ḣ, Ȧq̇** | ✓ | ✓ | **MISSING** | **H** | CRBA composite inertia + `_crba.py:241` recursion | M |
| **Centroidal derivatives** | ✓ | ✗ | **MISSING** | M | RNEA-derivatives (`id_du`) repack, à la Pin | M-L |
| **General frame Jacobian (any link, all 3 conventions)** | ✓ | ✓ | **PARTIAL** (EE-anchored only) | H | generalize `_eepose_gradient_hessian.py:406` | S-M |
| **Frame Jacobian time-variation `J̇`** | ✓ | ~ | **MISSING** | M | spatial-J path + velocity sweep | M |
| **Operational-space inertia `Λ = (J M⁻¹ Jᵀ)⁻¹` / OSC** | ✓ (Delassus) | ✗ | **MISSING** | **H** | `M⁻¹Jᵀ` from f_ext work + `direct_minv` | M |
| **Constraint/contact FD (`constraintDynamics`/`contactABA`)** | ✓ | (sim) | **MISSING** | M | ABA + Λ/Delassus | L |
| **Impulse dynamics** | ✓ | (sim) | **MISSING** | L | Λ/Delassus | M |
| **Lie ops full set (difference/interpolate/distance/randomConfiguration/neutral/normalize, dDifference)** | ✓ | ✓ | **PARTIAL** (integrate/dIntegrate only) | M | `RBDReference.py:153-268` | S-M |
| **Static / kinetic-energy / body regressors** | ✓ | ✗ | **MISSING** (joint-torque `Y` is D.4) | L-M | D.4 `Y` machinery | M |
| **`buildReducedModel` (lock joints)** | ✓ | ~ | **MISSING** | L | mimic reduced-model fallback `RBDReference.py:1907` | M |
| **Mimic-joint CUDA codegen** | ✓ | ~ | **PLANNED** (`urdf_feature_matrix`) | M | RBD mimic mult/offset `RBDReference.py:1687` | M |
| **Planar / helical / spherical joints** | ✓ | ~ | **MISSING** | L | URDFParser `set_type` | M-L |
| **Soft contact / friction-cone solver** | ~ | ✓ | **MISSING** | L (niche) | — | L |
| **IK solver** | ~ | ✗ | **MISSING** | L | EE pose + Jacobian (Gauss-Newton) | M |
| **Collision / geometry (coal/FCL)** | ✓ | ✓ | **OUT** | — | — | XL |
| **MJCF / SDF parsing** | ✓ | ✓ | **OUT (optional)** | L | URDFParser | L |

---

## 3. Prioritized roadmap

### Tier 0 — finish the in-flight specs (highest leverage, already designed)
These are not new asks; they are the committed plans and they unblock several Tier-1
items below. **Land D.4 (regressor `Y`), the f_ext column (incl. `M⁻¹Jᵀ`), and D.3
(PyTorch/CUDA-Graphs) first.** The `M⁻¹Jᵀ` product the f_ext work produces is the
exact kernel OSC / Λ needs (Tier 1, R5); the regressor `Y` is the sysID story the
JAX-physics libs lack.

### Tier 1 — top value-per-effort (the recommended five)

Each: *what it is* · *why it fits the niche* · *reuse* · *GPU-batch angle* · *oracle*.

**R1. Energy + gravity + Coriolis/centrifugal terms** (`kineticEnergy`,
`potentialEnergy`, `mechanicalEnergy`, `generalizedGravity g(q)`, `nonLinearEffects
c(q,q̇)=C q̇ + g`).
- *Math:* KE = ½q̇ᵀM q̇ ; PE = −Σ mᵢ gᵀ pᵢ ; g(q) = RNEA(q, 0, 0) ; c(q,q̇) =
  RNEA(q, q̇, 0).
- *Why:* energy is the cheapest, highest-frequency ask in control/learning
  (Lyapunov/energy-shaping, reward terms, conservation checks for the differentiable
  integrator); g(q)/c(q,q̇) are the canonical `M q̈ + c = τ` split every MPC/ID
  controller wants and every comparison lib exposes. **GRiD is the only one of the
  three (Pin/frax/GRiD) missing the explicit `g`, `c`, energy entry points** even
  though all the arithmetic is already inside RNEA/CRBA.
- *Reuse:* RNEA forward/gravity sweep (`_inverse_dynamics.py:122`), `crba` for M,
  `vxIv` bias (`RBDReference.py:857`). Thin wrappers — mostly arg-pinning (q̇=q̈=0) +
  one reduction.
- *GPU-batch:* trivially batched; energy is a single block-reduction, ideal warp work.
- *Oracle:* `pinocchio.computeKineticEnergy/computePotentialEnergy`,
  `computeGeneralizedGravity`, `nonLinearEffects`.
- *Effort:* **S** (S each). **Highest value-per-effort in the doc — do first in Tier 1.**

**R2. Centroidal momentum matrix `A(q)` + momentum `h`, `ḣ`, bias `Ȧq̇`**
(`ccrba` / `computeCentroidalMap` / `computeCentroidalMomentum[TimeVariation]`).
- *Math:* A(q) = ᶜXₒ Σ (composite spatial inertia projected to CoM frame) ; h = A q̇
  (centroidal momentum: linear + angular about CoM).
- *Why:* centroidal dynamics is **the** model for legged / floating-base balance,
  whole-body MPC, and momentum-based control — GRiD's floating-base CRBA already
  computes the composite inertias this needs, and **no GPU-batched library ships
  batched centroidal derivatives**. Direct differentiator vs frax (which has
  centroidal value but no analytic derivative) and a clean batch-throughput win vs
  Pinocchio (CPU).
- *Reuse:* CRBA composite-inertia recursion (`_crba.py:232` IC init, `:241-243`
  upward composite accumulation, floating path `:198`); CoM (R3) for the ᶜXₒ shift.
- *GPU-batch:* one extra 6×6 projection per body on top of CRBA's existing per-body work.
- *Oracle:* `pinocchio.ccrba` (A, h), `computeCentroidalMomentumTimeVariation` (ḣ),
  `computeCentroidalDynamicsDerivatives` (verified signature) for the gradient variant.
- *Effort:* **M** (value first, derivatives a follow-on once D.4/RNEA-deriv repack lands).

**R3. CoM position + CoM Jacobian** (`centerOfMass`, `jacobianCenterOfMass`).
- *Math:* p_com = (Σ mᵢ pᵢ)/M_total ; J_com = (Σ mᵢ Jᵥ,ᵢ)/M_total.
- *Why:* prerequisite for R2 (centroidal frame) and a standalone control output
  (CoM tracking cost — slots straight into the existing `grid_plant` cost family next
  to `ee_pos_cost`). Cheap and universally used.
- *Reuse:* composite-mass accumulation in CRBA (`_crba.py:232`); the per-joint
  geometric-Jacobian fill already emitted for EE (`_eepose_gradient_hessian.py:406`).
  A CoM cost reuses the `ee_pos_cost` GN-Hessian pattern (`_plant.py:259`).
- *GPU-batch:* a mass-weighted reduction over the same per-body Jacobian columns.
- *Oracle:* `pinocchio.centerOfMass`, `jacobianCenterOfMass`.
- *Effort:* **M** (S-M).

**R4. General frame Jacobian for any link, all 3 conventions** (`getFrameJacobian`
LOCAL / WORLD / LOCAL_WORLD_ALIGNED, + `frameJacobianTimeVariation` as a follow-on).
- *Math:* same geometric Jacobian GRiD already builds for the EE anchor, but (a) for
  an arbitrary requested link, and (b) emitted in all three Pinocchio frame
  conventions instead of only LWA.
- *Why:* the EE-anchored path (`_eepose_gradient_hessian.py:406`) is **already 80% of
  this** — it computes the spatial Jacobian and currently exposes the LWA mapping.
  Generalizing the anchor + offering LOCAL/WORLD closes the single biggest "Pin/frax
  has it, GRiD has a special-case" gap, and is needed for multi-frame tasks
  (hand+foot, multiple contacts).
- *Reuse:* `gen_end_effector_pose_gradient_inner` (`_eepose_gradient_hessian.py:406`),
  the per-ee chain-fill bake (`:365`), world-frame transforms already populated.
- *GPU-batch:* unchanged from EE path (per-chain, already parallel).
- *Oracle:* `pinocchio.getFrameJacobian(..., pin.LOCAL/WORLD/LOCAL_WORLD_ALIGNED)`.
- *Effort:* **S-M** (the generalization is mostly indexing + the LOCAL/WORLD rotation map).

**R5. Operational-space inertia `Λ = (J M⁻¹ Jᵀ)⁻¹` and the OSC/Delassus operator**
(`J M⁻¹ Jᵀ`).
- *Math:* the task-space inertia; OSC torque τ = Jᵀ(Λ ẍ* + …); the same `J M⁻¹ Jᵀ`
  is Pinocchio's Delassus operator and the kernel of constraint dynamics.
- *Why:* operational-space / task-space control is a flagship MPC/whole-body primitive
  that **neither frax nor Brax exposes as an analytic batched op**. The expensive
  piece — `M⁻¹Jᵀ` — is *exactly* what the f_ext gradient work (Tier 0) already emits
  (`∂q̈/∂f_ext = M⁻¹Jᵀ`), so once that lands, Λ is a small 6×6-ish solve on top.
- *Reuse:* `M⁻¹Jᵀ` from `differentiability_extensions_plan.md §A`, `direct_minv`
  (`_direct_minv.py`), the general frame Jacobian from R4.
- *GPU-batch:* `M⁻¹Jᵀ` is already a batched kernel; Λ adds a small dense factor/solve
  (3×3 or 6×6) per task — warp-friendly.
- *Oracle:* `pinocchio` Delassus operator / `computeOperationalSpaceInertiaMatrix`-style
  J M⁻¹ Jᵀ assembled from `crba`+`getFrameJacobian`.
- *Effort:* **M** (gated on Tier 0 f_ext landing).

### Tier 2 — strong, but heavier or narrower
- **Coriolis matrix `C(q,q̇)`** (the full matrix, not just `c`): reuses `id_du`
  ∂τ/∂q̇ structure. Val M, Eff M. (Most users only need `c`; full `C` is for
  passivity-based control.)
- **Centroidal dynamics derivatives** (`computeCentroidalDynamicsDerivatives`):
  follow-on to R2 once the RNEA-derivative repack (Pinocchio's trick:
  `getCentroidalDynamicsDerivatives` reads RNEA derivatives) is wired from `id_du`.
  Val M, Eff M-L.
- **Full Lie-group config-op set** (`difference`, `interpolate`, `distance`,
  `randomConfiguration`, `neutral`, `normalize`, `dDifference`): GRiD has
  `integrate`/`dIntegrate` (`RBDReference.py:235/268`) + the SE(3)/SO(3) primitives
  (`:153-234`); the rest are small host/device helpers that make the JAX/PyTorch
  surface feel complete for trajectory interpolation & sampling-based MPC. Val M, Eff S-M.
- **Frame Jacobian time-variation `J̇`**: R4 + a velocity sweep. Val M, Eff M.

### Tier 3 — defer (real but off the critical path)
- Constraint / contact forward dynamics (`constraintDynamics`, `contactABA`),
  impulse dynamics: large, and overlaps the "minimal differentiable simulator"
  question — revisit only if the integrator+plant deliberately grow into a sim.
  Λ/Delassus (R5) is the useful *primitive* slice; the full solver is Tier 3.
- Static / body / kinetic-energy regressors: niche sysID extensions of D.4's
  joint-torque `Y`.
- `buildReducedModel`, additional joint models (planar/helical/spherical), IK solver
  (Gauss-Newton on R4's Jacobian) — adopt on user demand.

---

## 4. Explicitly out of scope (with reasoning)
- **Full collision / geometry / distance queries** → defer to **coal / HPP-FCL** (and
  to Brax/MJX/MuJoCo for sim-side contact). This is a mature, separable problem an
  order of magnitude larger than GRiD's RBD-primitive niche; GRiD should *consume* a
  collision lib, not reimplement one. (Λ/OSC in R5 gives the contact-relevant *inertia*
  primitive without owning collision.)
- **MJCF / SDF parsing** → optional adoption play, low value for the niche. URDF covers
  the control/learning robots GRiD targets; MJCF/SDF mostly matter for sim
  interoperability (Brax/MuJoCo/Drake territory). Add only if a concrete user needs it.
- **A full differentiable simulator** (Brax/MJX scope: soft contact, domain
  randomization, RL env API) → GRiD is RBD-*primitive*-oriented. The integrator+plant
  may grow toward a *minimal* differentiable step, but owning a competitive sim engine
  is out of scope; GRiD's edge is analytic, batched, codegen'd primitives those
  simulators can be *built on*.

---

## 5. Top-5 (value-per-effort) — quick reference
1. **R1 Energy + g(q) + c(q,q̇)** — H value, S effort, all arithmetic already in RNEA/CRBA. *Do first.*
2. **R2 Centroidal momentum matrix A(q), h, ḣ** — H value, M effort, reuses CRBA composite inertias; unique batched centroidal niche.
3. **R3 CoM + CoM Jacobian** — M-H value, M effort, prereq for R2, drops a CoM cost into `grid_plant`.
4. **R4 General frame Jacobian (3 conventions)** — H value, S-M effort, generalizes the existing EE-anchored geometric Jacobian.
5. **R5 OSC / Λ = (J M⁻¹ Jᵀ)⁻¹** — H value, M effort, *free-rides on the f_ext work's M⁻¹Jᵀ*; flagship whole-body-MPC primitive neither frax nor Brax ships.
