# Kinematics warp/thread (no-derivative FK) — viability + rename plan

**Status:** IMPLEMENTED 2026-05-30 (branch `g2-warp-fk`). The warp/thread FK
inners were renamed `X_single_thread`/`X_warp` → `ee_pose_inner_thread`/
`ee_pose_inner_warp`, generalized beyond iiwa14 (driven by the robot's symbolic
per-joint Xmats + parent array; serial AND branched trees), and a batched
`ee_pose_fk_batched_kernel`/`ee_pose_fk_batched` host + `grid_rbd.fk_batched`
binding were added ((B×NUM_POS) q → (B×7) pos+quat, one block/warp per sample).
Validated on iiwa14 + gen3 (non-iiwa14 7R) + go2 (branched 12-DoF) at B=64,
both thread+warp variants, vs the RBDReference oracle (max pos/quat err ~1e-7).
`fixed_target` (Caveat B / `mat4_mul`) was dropped from these inners rather than
fixed: the standalone FK inner targets leaf-EE / any-frame via `target_idx`, not
the fixed-flange path. Floating-base / mimic robots are not supported by the
standalone inner (they route through `end_effector_pose`).
**Sequencing gate (historical):** was gated AFTER T2/T3 merge; executed off the
G1-merged tree.

## Customer use case

A real customer runs **fast, no-derivative forward kinematics / end-effector pose**
for **sampling-based algorithms over kinematics** (e.g. sampling-based MPC, motion
planning, IK seeding). The access pattern is throughput-bound batch FK with two
mapping strategies:

- **one thread per sample** (`X_single_thread`) — each CUDA thread walks the whole
  chain serially; maximal occupancy, no intra-sample cooperation.
- **one warp per sample** (`X_warp`) — 32-lane cooperative chain walk (3 lanes do
  the matrix rows, lanes 0..6 build the per-joint `Xhom`), `__syncwarp` between
  levels; better for longer chains / when a sample's working set fits a warp.

No gradients, no hessians. The customer wants (1) confirmation these still work
after recent refactors and (2) clearer names.

## PRIMARY GOAL (user, 2026-05-30): large-batch FK

The point is **throughput over large batches** — the API must take a **vector of B
input configurations** (a batch) plus a **batch-size `B`** parameter, and compute B
end-effector poses in one launch. The two mapping strategies are *per-sample*:
- **single-thread variant** — launch ~B threads, **one thread per sample**, each
  walking its sample's full chain serially. Output: B EE poses.
- **warp variant** — launch ~B warps, **one warp per sample** (32-lane cooperative
  chain walk), `__syncwarp` between levels. Output: B EE poses.

So the deliverable is a **batched kernel + host wrapper + grid_rbd binding** taking
`B` and a `(B × input)` array, returning `(B × 6·N_ee)` — NOT a single-call device
helper. The existing `X_single_thread`/`X_warp` device helpers become the *per-sample
inner* of the batched kernel. Mirror the batch/stride conventions of the existing
`end_effector_pose` batched path (and the generalize-beyond-iiwa14 + `mat4_mul` fixes
still apply to the inner).

## Reference usage: HJCD-IK (https://github.com/a2r-lab/HJCD-IK)

The a2r-lab HJCD-IK project (batched GPU inverse kinematics) is the canonical consumer.
How it actually uses the FK device functions (`src/hjcd_kernel.cu`):
- **Launch:** `forward_kinematics_kernel<T><<<num_configs, 32>>>(d_q, d_pose7, nullptr,
  d_robotModel, num_configs)` — **one block per sample**, 32 threads/block; the batch
  size is the grid dim (`B = gridDim.x`, `b = blockIdx.x`).
- **Input layout:** batch-major, stride N (DoF): `s_q[j] = q[b * N + j]` (cooperative
  load over `threadIdx.x`).
- **Output:** **7-element pose = position + quaternion** per sample, extracted from the
  EE's 4×4 homogeneous transform: `pose7[b*7+0..2] = Cee[12..14]` (translation column),
  `pose7[b*7+3..6]` = quaternion from the rotation block. (NOT the 6-elem rpy that
  `end_effector_pose` emits — consumers want pos+quat.)
- **Device call (the functions to generalize + rename):**
  `grid::X_single_thread<T>(s_jointX, s_XmatsHom, s_q, FLANGE_IDX)` and
  `grid::X_warp<T>(s_jointX, s_XmatsHom, s_x, FLANGE_IDX)` — args
  `(out joint transforms [16/joint = 4×4 homogeneous], XmatsHom scratch, q in,
  target-frame idx)`. The customer reads the **FULL `s_jointX`** (all joint transforms,
  any frame), not just the EE — it reuses intermediates in the IK/collision loop and
  recalls `X_warp` each LM iteration (`warp_id==0`) before `compute_pos_err`/`compute_ori_err`.
- HJCD-IK regenerates its own `grid.cuh` via `scripts/generate_grid.py` → these ARE
  GRiD-emitted device functions today.

**So the G2 deliverable is two levels:**
1. The **robot-general device inner** `ee_pose_inner_{thread,warp}<T>(s_jointX,
   s_XmatsHom, s_q, target_idx)` — generalized beyond hardcoded iiwa14, `mat4_mul`
   fixed, full `s_jointX` output — so consumers like HJCD-IK call it inside their own
   per-sample kernel.
2. A **convenience batched kernel + host wrapper + grid_rbd binding** doing the
   `<<<B, threads>>>` one-block-(or-warp)-per-sample launch, `q[b*N]` in →
   `pose7[b*7]` (position+quaternion) out, for the "just give me batched FK" path.

Keep full `s_jointX` accessible; offer pos+quat (7) as the packaged batched output
(matching HJCD-IK), not only the 6-elem rpy.

---

## Step 1 — What actually exists (current names + locations)

### The two target functions (the warp/thread FK path)

Both are emitter methods on the codegen class, defined in
`GRiDCodeGenerator/algorithms/_eepose_gradient_hessian.py`:

| Emitter (Python) | Emitted CUDA symbol | File:line |
|---|---|---|
| `gen_X_single_thread(self, fixed_target_name="")` | `template<typename T> __device__ void X_single_thread(T *s_jointXforms, T *s_XmatsHom, T *s_q, int tid)` | `_eepose_gradient_hessian.py:2409` (def at :2409, signature emitted at :2432) |
| `gen_X_warp(self, fixed_target_name="")` | `template<typename T> __device__ inline void X_warp(T* s_jointXforms, T* s_XmatsHom, const T* s_q, int tid)` | `_eepose_gradient_hessian.py:2532` (def at :2532, signature emitted at :2554) |

Semantics: given `s_q`, build the per-joint homogeneous transforms `s_XmatsHom`,
then accumulate cumulative joint transforms `s_jointXforms[j]` up to joint `tid`
(the chain walk), so the caller can read the world pose of frame `tid`. `X_warp`
uses `lane = threadIdx.x & 31`, `__syncwarp(mask)`, and 3-lane row partitioning;
`X_single_thread` does the same serially with a `#pragma unroll` loop.

### How they are emitted

- Imported into the codegen class:
  `GRiDCodeGenerator/GRiDCodeGenerator.py:43`
  (`... gen_X_single_thread, gen_X_warp, ...`).
- Emitted unconditionally at the tail of `gen_eepose_and_derivatives(...)`:
  `_eepose_gradient_hessian.py:2737–2739`
  ```python
  if include_pose or include_gradient or include_hessian:
      self.gen_X_single_thread(fixed_target_name = fixed_target_name)
      self.gen_X_warp(fixed_target_name = fixed_target_name)
  ```
- `gen_eepose_and_derivatives` is dispatched from the main pipeline at
  `GRiDCodeGenerator.py:1757` (gated on `include_any_kinematics`). So any robot
  header generated with `ee_pose` (or its gradient/hessian) in `algorithms` also
  gets `X_single_thread` + `X_warp` for free in the header.

### The standard (block-parallel) FK value path — for contrast

The *general* no-derivative EE-pose value path is a **separate, block-parallel**
function family, NOT the warp/thread one:

- `gen_end_effector_pose_inner` → `end_effector_pose_inner<T,TEMP_IN_SMEM>(...)`
  `_eepose_gradient_hessian.py:34` (body at :34–166). This walks the chain with
  BFS-level `gen_add_parallel_loop` blocks over the **whole thread block** and is
  fully topology-general (serial-chain fast path + branched trees + fixed targets).
- Device / kernel / host wrappers:
  `gen_end_effector_pose_device` (:174), `gen_end_effector_pose_kernel` (:201),
  `gen_end_effector_pose_host` (:257) → emitted symbols `end_effector_pose_device`,
  `end_effector_pose_kernel`, host `end_effector_pose<T,USE_COMPRESSED_MEM,KIND>`.

This block-parallel path is the one that is registered, tested, and bound:

- **Registry:** `algo_registry.py:67` `AlgoEntry("ee_pose", "EE_POSE", "Kinematics",
  legacy_labels=("eepos",))` (gradient :69, hessian :71). The `X_*` helpers have
  **no registry entry** (they are not user-selectable algorithms; they ride along).
- **Tests:** `test/cuda_equivalents/test_cuda_executable_equivalence.py` exercises
  `"end_effector_pose"` (:56, :70, :1028, :1078, :1190…) vs the Pinocchio/RBDReference
  oracle, with rpy gimbal-lock handling (:1132–1200). `test_cuda_codegen_layout.py`
  asserts the host signature (:351) and `GRID_DATA_KINEMATICS` calls (:607/631/660).
  `test/python_wrappers/test_any_thread_count.py:106/135` sweeps thread counts on
  `handle.end_effector_pose(q)`. **There is no test that references `X_single_thread`
  or `X_warp` by name** — the warp/thread helpers are currently untested.
- **Bindings (`grid_rbd`):** `python/src/_core.cpp:93` binds `grid_rbd_end_effector_pose`
  (+ gradient :94, hessian :98); the `end_effector_pose(q, batch)` method is at
  `_core.cpp:254–270`. **No binding exists for the warp/thread helpers** — they are
  header-only device functions for the customer to call from their own kernels.

### Callers

Grep across the whole repo (`.py/.cuh/.cu/.cpp/.h`) for `X_single_thread` /
`X_warp` finds **only** the emitter (`_eepose_gradient_hessian.py`) and the import
line (`GRiDCodeGenerator.py:43`). There are **no in-repo callers** of the emitted
CUDA symbols — confirming these are direct customer entry points emitted into
`grid.cuh`, consumed by customer kernels outside this repo.

---

## Step 2 — Viability assessment (read-level)

**Verdict: the default (non-fixed-target) `X_single_thread` / `X_warp` path is
viable and was NOT touched/broken by the recent refactors — with two caveats.**

Reasoning:

1. **Independent of the refactor blast radii.** The recent landed refactors —
   resource-tier spill (per-tier emit), BFS-parallel CRBA (`_crba.py`), idsva_so
   body/world changes (`_idsva_so.py`), and the just-merged T6 `grid_plant`
   (`4c91305`) — do not touch `gen_X_single_thread`/`gen_X_warp`. `git log` on
   `_eepose_gradient_hessian.py` shows no recent commits rewriting these emitters,
   and the functions take raw pointers (`s_jointXforms, s_XmatsHom, s_q, tid`) with
   no `RESOURCE_TIER` template param, no tier-spill smem accounting, and no
   topology-helper / `d_robotModel` dependency — so the tier-spill and
   resource-tier machinery cannot have changed their behavior.

2. **No mimic / α-scaling coupling.** Grep for `alpha|mimic|scal` in
   `_eepose_gradient_hessian.py` finds only the *hessian* d2M scalar-pair comment
   (:1928). The FK value path and the `X_*` helpers have zero mimic logic, so **T3
   (mimic α-scaling codegen) does not touch the FK value path** — confirmed.

3. **T2 targets the gradient, not the value path.** Per HANDOFF F-batch
   (`HANDOFF.md:1180`, `:1206`), T2 = "A.6 ee_pose_gradient perf gaps … loop
   restructuring", changing ZERO signatures. It restructures
   `gen_end_effector_pose_gradient_inner` (`_eepose_gradient_hessian.py:406`), not
   `gen_end_effector_pose_inner` (the value path) nor `gen_X_*`. **T2 does not touch
   the FK value path** — confirmed.

### Caveat A — the `X_*` helpers are hardcoded to a 7-DOF all-revolute serial chain (iiwa14)

Unlike `end_effector_pose_inner` (fully topology-general), `gen_X_single_thread`
and `gen_X_warp` read `n = self.robot.get_num_pos()` but the **emitted body is
hardcoded**: `s_q[0]..s_q[6]` with iiwa14's specific per-joint cos/sin axis pattern
(`_eepose_gradient_hessian.py:2440–2473`), and `X_warp` partitions `lane <= 6`
(:2574). For a non-7-DOF or non-iiwa robot these emit wrong/garbage transforms (or
read `s_q` out of bounds). **They are only correct for iiwa14-class 7R serial
arms today.** This is the single biggest viability limitation and should be called
out to the customer: today this is an iiwa14 fast-path, not a general FK helper.
(The general, correct, but block-parallel alternative is `end_effector_pose_inner`.)

### Caveat B — the `fixed_target_name != ""` branch will not compile

Both helpers, in their `has_fixed_target` branch, call `mat4_mul(Tfl, Xfix, Tee)`
(`_eepose_gradient_hessian.py:2525` and `:2681`). Grep shows **`mat4_mul` is
defined nowhere** in GRiDCodeGenerator, GLASS, or any emitted header. So any header
generated with a fixed EE target (e.g. `fixed_target_name="all"`) that then *uses*
`X_single_thread_<target>` / `X_warp_<target>` would fail to compile. The default
(leaf-EE, no fixed target) path does not hit `mat4_mul` and is fine.

### Smoke test recommendation

A compile + equivalence smoke (**iiwa14, FK-only**, default no-fixed-target) should
be run once the GPU frees up — it is **busy now** with the F-batch:
`ps`/`nvidia-smi` show active `nvcc` compiles under `GRiD-T2-perf-gaps` and a
floating fext runner, plus a `tier_inst_smoke/h1_2_fixed` build. Per the
"no CPU work during a bench" rule, **do not run the smoke now**; queue it after
T2/T3 finish. Suggested smoke: generate the iiwa14 header with `ee_pose`, compile a
tiny kernel that calls `X_single_thread`/`X_warp` for `tid = NUM_POS-1`, and compare
the resulting frame-`tid` translation against `end_effector_pose` (and/or the
RBDReference oracle) for a few random `q`.

---

## Step 3 — Rename + blast radius

### Current → proposed names

| Kind | Current | Proposed |
|---|---|---|
| Python emitter | `gen_X_single_thread` | `gen_ee_pose_inner_thread` |
| Python emitter | `gen_X_warp` | `gen_ee_pose_inner_warp` |
| Emitted CUDA symbol | `X_single_thread` | `ee_pose_inner_thread` |
| Emitted CUDA symbol | `X_warp` | `ee_pose_inner_warp` |
| Fixed-target variants | `X_single_thread_<target>` / `X_warp_<target>` | `ee_pose_inner_thread_<target>` / `ee_pose_inner_warp_<target>` |

Rationale: `ee_pose_inner_{thread,warp}` mirrors the existing
`end_effector_pose_inner` / `*_gradient_inner` naming family (so they read as the
no-derivative siblings), encodes the parallel-mapping (thread-per-sample /
warp-per-sample) the customer cares about, and drops the opaque single-letter `X`.
(The customer's suggested names `ee_pose_inner_warp` / `ee_pose_inner_thread` are
adopted as-is.)

### Full blast radius (what changes when the rename lands)

1. **Emitter defs** — `_eepose_gradient_hessian.py:2409` (`def gen_X_single_thread`)
   and `:2532` (`def gen_X_warp`): rename the `def`.
2. **Emitted symbol strings** inside those defs — `:2432`
   (`void X_single_thread(...)`) and `:2554` (`__device__ inline void X_warp(`):
   rename the literal CUDA function name. Update the func-doc strings (:2420, :2543)
   to say "thread-per-sample" / "warp-per-sample (warp-cooperative)".
3. **Call sites of the emitters** — `_eepose_gradient_hessian.py:2738–2739`:
   `self.gen_X_single_thread(...)` / `self.gen_X_warp(...)`.
4. **Import line** — `GRiDCodeGenerator.py:43`: rename both names in the
   `from ... import (...)` list.
5. **Registry** — no change required (these have no `AlgoEntry`). *Optional:* if we
   want them to be independently selectable/labelled, add Kinematics `AlgoEntry`s
   near `algo_registry.py:67` — but that is a scope-expansion, not part of the
   rename. Recommend NOT adding entries in the rename pass.
6. **Tests** — none reference the old names, so none break. **Add** a new FK-only
   smoke that calls `ee_pose_inner_thread` / `ee_pose_inner_warp` (currently the
   only coverage of this path is indirect via `end_effector_pose`). Put it next to
   `test/cuda_equivalents/test_cuda_executable_equivalence.py`.
7. **`grid_rbd` bindings / handle** — no symbol named `X_*` is bound today
   (`python/src/_core.cpp:93–98` only binds `end_effector_pose[_gradient|_hessian]`).
   So the rename does not touch bindings. *Optional follow-on (separate task):*
   expose a batched `ee_pose_inner_warp`/`_thread` host wrapper + `grid_rbd` method
   if the customer wants a Python surface rather than calling the device fn from
   their own kernel.
8. **Docs** — update `HANDOFF.md:1252` (which already pre-names them
   `ee_pose_inner_{thread,warp}`) and `library_capability_roadmap.md:109/124` (the
   "Frame/geometric Jacobian (LWA) — HAS (EE anchor)" rows that point at this file).
9. **Regenerated headers** — any committed `grid.cuh` fixtures containing
   `X_single_thread`/`X_warp` must be regenerated. (Clear the header cache after
   codegen changes, per the parallel-equivalence-testing note.)

### Consistency with the future general frame-Jacobian (R4)

`library_capability_roadmap.md:124` lists **R4 "General frame Jacobian (any link,
all 3 conventions)"** as "PARTIAL (EE-anchored only)", to be built by generalizing
`gen_end_effector_pose_gradient_inner` (`_eepose_gradient_hessian.py:406`). The
warp/thread FK helpers are the **value-level (no-derivative) analog** of that same
"pose of an arbitrary frame `tid`" capability — `X_*` already takes a `tid`
argument. To avoid a divergent one-off:

- Name them in the same `ee_pose_inner_*` family so R4's frame-value helper (if/when
  it generalizes the chain walk to arbitrary links and gets topology-general like
  `end_effector_pose_inner`) can subsume or sit beside them cleanly.
- When R4 lands, **Caveat A (the iiwa14 hardcoding) should be fixed by routing these
  through the same topology-general chain walk** R4 uses, rather than keeping a
  hand-written 7R body. Flag this as the convergence point so the warp/thread FK
  doesn't fossilize as an iiwa-only fast path.

---

## Sequencing

**Land the rename AFTER T2 and T3 merge.** Both edit
`_eepose_gradient_hessian.py` (T2 = ee_pose_gradient perf gaps / loop restructuring;
T3 = mimic codegen), and both are **currently in flight** in sibling worktrees
`/home/plancher/Desktop/GRiD-T2-perf-gaps` and
`/home/plancher/Desktop/GRiD-T3-mimic-codegen` (T2's `nvcc` was actively compiling
at exploration time). Renaming `X_single_thread`/`X_warp` now would create avoidable
merge conflicts in that file. Per HANDOFF (`:1206`, `:1253`) the planned merge order
is T3 → T5 → T2; do the rename as a small isolated commit once T2 has merged into
`modernizing-tests`.

Recommended order:
1. Wait for T2 + T3 to merge into `modernizing-tests`.
2. Apply the rename (items 1–4, 8 above) in one commit.
3. Run the FK-only iiwa14 compile + equivalence smoke (item, Step 2) on a free GPU.
4. (Optional, separate tasks) decide on Caveat A generalization (tie to R4) and
   Caveat B (`mat4_mul` for fixed targets) — both are pre-existing and out of scope
   for a pure rename, but should be filed.
