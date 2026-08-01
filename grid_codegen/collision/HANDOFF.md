# grid_collision — automated URDF→sphere collision, and how to integrate it

This is the handoff for wiring GRiD's `grid_collision` into an IK / motion-planning consumer
(e.g. HJCD-IK). It covers the **one-command automated flow** (URDF → spherized URDF →
`grid.cuh` with a `grid_collision` namespace), the **device ABI** you call from a kernel, and
the knobs/caveats. Start by replicating your own robot with the flow below, then call
`grid_collision::config_free` from your solver.

---

## 1. One-command generation

```bash
# any URDF -> grid.cuh with the grid_collision namespace baked in
python -m GRiDCodeGenerator.cli path/to/robot.urdf --collision --collision-res 0.05

# OR multiple densities in ONE header -> a broad->fine cascade (coarsest rejects, finest confirms)
python -m GRiDCodeGenerator.cli path/to/robot.urdf --collision --collision-res 0.10,0.05
```

- `--collision` turns on the collision pipeline (off by default → byte-identical to before).
- `--collision-res R` = sphere **spacing** in meters. Smaller ⇒ finer/more spheres (tighter,
  more spheres to check); larger ⇒ coarser/fewer. `0.05` is a good default arm/quadruped value.
- `--collision-res R1,R2[,…]` = **multiple densities**. They're sorted coarsest→finest and baked as
  named tiers (`broad`/`fine` for two; `tier0…tierK` for more). `config_free` then rejects clear
  configs on the *coarsest* tier and only confirms possible collisions on the *finest* — same
  verdict as fine-only (covering spheres make the coarse reject conservative), fewer checks on the
  common free case. A single value ⇒ single tier ⇒ byte-identical to before.
- `--collision-native` (opt-in, implies `-c`) = **NATIVE capsule rows** instead of covering
  spheres. The URDF's own collision primitives become one row `{a, b, r}` each: sphere → a==b
  degenerate row; cylinder → the containing capsule (same r + axis segment ⇒ conservative);
  box/mesh links keep spherized rows at `--collision-res`. FAR fewer rows (iiwa14: 34 fine
  spheres → 5 capsules) at ~2× the FK transform work per row (two endpoints ride the
  multi_target batch as targets `2i`/`2i+1`). Emits a broad→fine cascade automatically: the
  broad tier is one covering sphere per link **derived from the rows** (encloses the capsule
  caps — a coarser spherizer pass wouldn't). Public constants become `NUM_COLLISION_ROWS` &c.;
  the differentiable API additionally returns the robot-side closest-point parameter `t*` per
  row and composes gradients over BOTH endpoints (envelope theorem:
  `d(d)/dq = nᵀ[(1−t*)·da/dq + t*·db/dq]`). Gate: `test_cuda_collision_native.py`.

Programmatic entry (what the CLI calls):

```python
from URDFParser import URDFParser
from GRiDCodeGenerator import GRiDCodeGenerator
from GRiDCodeGenerator.algorithms._collision import (
    collision_spec_from_urdf, multi_tier_collision_spec_from_urdf)

robot = URDFParser().parse("robot.urdf", floating_base=False)
spec  = collision_spec_from_urdf(robot, "robot.urdf", resolution=0.05)              # single tier
# spec = multi_tier_collision_spec_from_urdf(robot, "robot.urdf", [0.10, 0.05])     # broad->fine
GRiDCodeGenerator(robot).gen_all_code(output_path="grid.cuh", collision_spec=spec)
```

`collision_spec` is either a single-tier `{anchor[N], offset[3N], radius[N], self_cc_ranges[R][3]}`
or a multi-tier `{"tiers": [ {name, anchor, offset, radius, self_cc_ranges}, … ]}` listed
coarsest→finest. One sphere per row: `anchor` = the GRiD movable-joint frame it rides on, `offset` =
its position in that frame (welded-link spheres are pre-folded onto the movable parent),
`self_cc_ranges` = the adjacency-pruned self-collision pair table.

Checked-in examples: `collision/assets/go2_spherized.urdf` (all-primitive, full coverage) and
`collision/assets/iiwa14_spherized.urdf` (arm; the two drake collision meshes need the drake
package to be spherized — see §4). These are the **foam interchange format**, so a foam-produced
spherized URDF is drop-in interchangeable (`parse_spherized_urdf` reads either).

---

## 2. The spherizer (`grid_codegen/algorithms/_spherize.py`)

`spherize_urdf(urdf, resolution)` rewrites each link's `<collision>` geometry as covering
spheres and returns a spherized URDF:

- **Primitives** (`sphere` / `cylinder` / `box`) → covered **analytically**, no external assets.
  A cylinder becomes a line of spheres along its axis; a box a voxel grid. Radii are chosen so
  the union **fully contains** the source surface (conservative — no missed collisions;
  see `test/test_collision_spherize.py`).
- **Meshes** → voxel-filled via `trimesh` (one sphere per interior voxel). If a mesh path can't
  be resolved, that collision is **skipped with a warning** (never silently dropped) and you get
  partial (primitive) coverage rather than an abort. `file://`, absolute, and relative paths
  resolve directly; `package://` is tried relative to the URDF dir.

Coarse `resolution` ⇒ broad tier, fine ⇒ fine tier. Pass several to `--collision-res` (or
`multi_tier_collision_spec_from_urdf`) and each density is spherized independently, then wired into
the header's broad→fine driver `grid_cc_config_free` (see §3).

---

## 3. Device ABI — what you call from a kernel

Single-block, thread-count-invariant (every thread computes the same verdict). fp32 is the
collision change-of-record.

```cpp
#include "grid.cuh"                       // generated (contains the grid_collision namespace)
namespace gc = grid_collision;

// runtime obstacle set — deep-copy each list to device, then patch the pointers:
template <typename T> struct Environment {
    const gc::Sphere<T>  *spheres;  int n_spheres;    // Sphere{ x,y,z,r }
    const gc::Capsule<T> *capsules; int n_capsules;   // Capsule{ ax,ay,az, bx,by,bz, r }
    const gc::Cuboid<T>  *cuboids;  int n_cuboids;    // oriented box: center + 3 (unit axis, half-extent)
};

// SINGLE-tier: returns true iff configuration s_q is collision-free (self + environment):
template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
__device__ bool gc::config_free(
    const T *s_q,                              // joint positions (grid::NUM_POS)
    const grid::robotModel<T> *d_robotModel,   // grid::init_robotModel<T>()
    const gc::Environment<T> &env,
    T *s_sphere_pos,                           // caller scratch, 3*NUM_COLLISION_SPHERES
    T *s_sphere_r,                             // caller scratch,   NUM_COLLISION_SPHERES (filled here)
    T *d_workspace = nullptr);                 // FK spill at TIER_LITE+ (nullptr at TIER_SHARED)

// MULTI-tier (2+ densities): same verdict, broad-reject/fine-confirm. Signature takes BOTH tiers'
// scratch (the coarse tier is named — e.g. _BROAD; the finest keeps the unsuffixed public names):
template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
__device__ bool gc::config_free(
    const T *s_q, const grid::robotModel<T> *d_robotModel, const gc::Environment<T> &env,
    T *s_broad_pos, T *s_broad_r,              // caller scratch, 3*NUM_COLLISION_SPHERES_BROAD / _BROAD
    T *s_fine_pos,  T *s_fine_r,               // caller scratch, 3*NUM_COLLISION_SPHERES / (finest)
    T *d_workspace = nullptr);
```

Sizing:
- `gc::NUM_COLLISION_SPHERES` — finest/public sphere count (== `grid::NUM_MULTI_TARGETS`). Coarse
  tiers are `gc::NUM_COLLISION_SPHERES_BROAD` etc. (== `grid::NUM_MULTI_TARGETS_BROAD`).
- `grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>()` — dynamic smem for the extractor (the
  coarse tier's is `..._BROAD_...`). With multiple tiers set the kernel's dynamic smem to the **max**
  across tiers — the device calls run sequentially and *alias* the same arena; only the caller-owned
  `s_*_pos`/`s_*_r` output buffers must be sized per tier and kept live. Set via
  `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`.
- At `TIER_LITE`/`TIER_MINIMAL` the FK scratch spills to `d_workspace`
  (`MULTI_TARGET_POSITION_DEVICE_INLINE_WORKSPACE_BYTES<T,TIER>()`); at `TIER_SHARED` pass
  `nullptr`. Same tier contract as `forward_dynamics_device`.

Minimal launches: `cuda_collision_config_free_runner.cu` (single-tier) and
`cuda_collision_two_tier_runner.cu` (broad→fine) under `test/cuda_equivalents/`.

The underlying SDFs (squared-gap convention, `<0` = collision) live in
`collision/grid_collision_geometry.cuh` — hand-written, robot-agnostic, peer to the GLASS linalg
headers.

---

## 4. Caveats / knobs

- **Conservative self-collision.** Covering spheres are inflated, so fat-link robots can report
  self-contact between *near-but-not-directly-adjacent* links at bent configs. `self_cc_ranges`
  prunes only same/parent/child pairs. If you see false positives, either raise `--collision-res`
  granularity or (future) supply an allowed-collision matrix that prunes more pairs.
- **Root/pedestal spheres are dropped.** Spheres on the base/world link (anchor = −1) are skipped
  (they can't move); this matches HJCD.
- **Unresolvable meshes are skipped** (loud warning). iiwa14's `package://drake/...` collision
  meshes fall in this bucket unless the drake package is on disk; its cylinder links still cover.
- **q convention** is GRiD's (`grid::NUM_POS`, pinocchio joint order). Feed the same `s_q` you
  feed the rest of GRiD.

---

## 5. Differentiable collision (for GATO/PDDP)

Beyond the boolean `config_free`, the same generated `grid_collision` namespace exposes the smooth
environment-clearance derivatives, all `template <typename T, int RESOURCE_TIER, bool ACCUMULATE>`
and bound to the **finest** sphere tier:

- **Raw primitives** (assemble any objective — hinge, log-barrier, hard constraint):
  - `collision_distance(s_dist, s_normal, s_q, m, env, s_sphere_pos, s_sphere_r, d_workspace)` —
    per-sphere signed clearance `d_i(q)` (min over obstacles; `+1e30` if env empty) + surface normal.
  - `collision_distance_gradient(s_dist, s_ddist, s_q, m, env, s_sphere_pos, s_sphere_r, s_normal,
    s_pos_grad, d_workspace)` — clearance Jacobian `s_ddist[i*NV+vi] = d(d_i)/dq = n̂ᵢᵀ(dpᵢ/dq)`
    (SDF normal ∘ the W2a batched position gradient).
- **Cost** (hinge on a safety margin `viol_i = max(0, margin − d_i)`, `cost = ½·w·Σ violᵢ²`):
  `collision_cost` (value), `collision_cost_gradient` (`Jᵀr`), `collision_cost_hessian`
  (Gauss-Newton, PSD). Env-only (self-collision stays the `config_free` boolean).

Caveats: the per-sphere nearest-obstacle **argmin is non-smooth** where the nearest obstacle
switches — freeze the active obstacle per MPC step for a stable Hessian (the `s_normal` pre-pass is
the seam). The full-Newton (residual-weighted SDF curvature) hessian is a labeled TODO; GN is the
ratified PSD choice. FD-validated in `test/cuda_equivalents/test_cuda_collision_cost.py`.

## 6. What's next (not yet wired)

- **HJCD-IK migration** — you: replace `csrc/collision` + per-robot `.cuh` with this `grid_collision`
  flow (fp64→fp32 is the change-of-record); gate on collision-free-rate.
- **`link_CC` broad-phase mask** — narrow the fine re-check to broad-flagged links (perf; today the
  fine pass re-checks all fine spheres). **k-level (>2) cascade** in `config_free` (today the driver
  uses coarsest+finest; middle tiers are emitted and callable but unused by `config_free`).
- **Full bench registration** of the multi_target extractor (kernels/host wrappers) — orthogonal.
