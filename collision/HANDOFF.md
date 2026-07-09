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
```

- `--collision` turns on the collision pipeline (off by default → byte-identical to before).
- `--collision-res R` = sphere **spacing** in meters. Smaller ⇒ finer/more spheres (tighter,
  more spheres to check); larger ⇒ coarser/fewer. `0.05` is a good default arm/quadruped value.

Programmatic entry (what the CLI calls):

```python
from URDFParser import URDFParser
from GRiDCodeGenerator import GRiDCodeGenerator
from GRiDCodeGenerator.algorithms._collision import collision_spec_from_urdf

robot = URDFParser().parse("robot.urdf", floating_base=False)
spec  = collision_spec_from_urdf(robot, "robot.urdf", resolution=0.05)
GRiDCodeGenerator(robot).gen_all_code(output_path="grid.cuh", collision_spec=spec)
```

`collision_spec` is `{anchor[N], offset[3N], radius[N], self_cc_ranges[R][3]}` — one sphere per
row, `anchor` = the GRiD movable-joint frame it rides on, `offset` = its position in that frame
(welded-link spheres are pre-folded onto the movable parent), `self_cc_ranges` = the
adjacency-pruned self-collision pair table.

Checked-in examples: `collision/assets/go2_spherized.urdf` (all-primitive, full coverage) and
`collision/assets/iiwa14_spherized.urdf` (arm; the two drake collision meshes need the drake
package to be spherized — see §4). These are the **foam interchange format**, so a foam-produced
spherized URDF is drop-in interchangeable (`parse_spherized_urdf` reads either).

---

## 2. The spherizer (`GRiDCodeGenerator/algorithms/_spherize.py`)

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

Coarse `resolution` ⇒ broad tier, fine ⇒ fine tier (the two-tier broad→fine driver
`grid_cc_config_free` in the geometry header is the next increment; `config_free` today is a
single tier).

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

// returns true iff configuration s_q is collision-free (self + environment):
template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>
__device__ bool gc::config_free(
    const T *s_q,                              // joint positions (grid::NUM_POS)
    const grid::robotModel<T> *d_robotModel,   // grid::init_robotModel<T>()
    const gc::Environment<T> &env,
    T *s_sphere_pos,                           // caller scratch, 3*NUM_COLLISION_SPHERES
    T *s_sphere_r,                             // caller scratch,   NUM_COLLISION_SPHERES (filled here)
    T *d_workspace = nullptr);                 // FK spill at TIER_LITE+ (nullptr at TIER_SHARED)
```

Sizing:
- `gc::NUM_COLLISION_SPHERES` — sphere count (== `grid::NUM_MULTI_TARGETS`).
- `grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>()` — dynamic smem for the extractor
  (set via `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`).
- At `TIER_LITE`/`TIER_MINIMAL` the FK scratch spills to `d_workspace`
  (`MULTI_TARGET_POSITION_DEVICE_INLINE_WORKSPACE_BYTES<T,TIER>()`); at `TIER_SHARED` pass
  `nullptr`. Same tier contract as `forward_dynamics_device`.

Minimal launch: see `test/cuda_equivalents/cuda_collision_config_free_runner.cu`.

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

## 5. What's next (not yet wired)

- **broad/fine two tiers** → `grid_cc_config_free` (coarse reject, fine confirm). Header already
  has the driver; the emitter is single-tier today.
- **differentiable collision cost** (value / `Jᵀr` gradient / Gauss-Newton hessian) for GATO/PDDP,
  composed from the W2a batched position gradient + SDF surface normals.
