# D.4 — Runtime mass/inertia parameters (two-variant emit) — implementation plan

**Status:** planning (read-only exploration done 2026-05-30, branch `modernizing-tests`).
**Goal:** Add a codegen variant that reads each link's mass/inertia as RUNTIME inputs
(so users can do system-ID / domain-randomization / payload changes WITHOUT
regenerating + recompiling the per-robot header), while KEEPING the existing fast
baked-constant path. Selectable by a codegen flag and a kernel-side template bool.

---

## 0. The one load-bearing fact that makes this cheap

The spatial-inertia matrices live INSIDE the same `s_XImats` shared buffer as the
spatial transforms, at a fixed layout that every algorithm already hard-codes:

- Memory order is `X[0..N-1]` then `I[0..N-1]`, each 36 floats (6×6, col-major-ish
  3D index via `gen_static_array_ind_3d`). See `GRiDCodeGenerator/helpers/_topology_helpers.py:88-93`
  (`gen_init_XImats` docstring "Memory order is X[0...N], I[0...N]"), the Imat fill
  loop at `_topology_helpers.py:120-134`, and `gen_get_XI_size` at
  `_topology_helpers.py:4-8` (`base_size = 36*2*n (+36 if include_base_inertia)`).
- Algorithms read the inertia block as `&s_XImats[36*n + 6*jid6 + row]` etc. — the
  offset `36*n` is baked as a Python int at emit time. Examples (all in
  `algorithms/_inverse_dynamics.py`): the I-block dot at `:267`
  (`&s_XImats[36*n + 6*jid6 + row]`), the gravity column reads at `:122`/`:125`
  (`s_XImats[6*jid6 + 30 + row]` = bottom row of an X = mass*I3 region of an I when
  jid is offset), the `printMat` at `:78` (`&s_XImats[36*(i+n)]`).

**Consequence:** the runtime variant does NOT need to change a single algorithm.
It only needs to change how the `I[0..N-1]` region of `s_XImats` gets POPULATED.
Today that region is a verbatim copy from constant global memory; the runtime
variant instead reconstructs each 6×6 from runtime 10-params on-device, writing
into the exact same shared slots. The X region (transforms, q-dependent) is
untouched. This is the whole trick and it is why "two-variant" is tractable.

---

## 1. Inventory of baked inertial constants + exact emit sites

There are exactly TWO places that bake the spatial-inertia numbers into the header,
plus the upstream Python source that computes them:

### 1a. Host init (constant global buffer)
`GRiDCodeGenerator/helpers/_topology_helpers.py:120-134` — inside `gen_init_XImats`:
```
Imats = self.robot.get_Imats_ordered_by_id()        # :121
if not include_base_inertia: Imats = Imats[1:]       # :122-123
for ind ...:                                          # :125
    str_val = str(Imats[ind][row,col])                # :132
    h_XImats[<offset>] = static_cast<T>(str_val)      # :134
```
These 36·N literals are `cudaMemcpy`'d to `d_XImats` at `:179-180` and become
`d_robotModel->d_XImats`.

### 1b. Device load (copy const → shared)
`_topology_helpers.py:280-299` — inside `gen_load_update_XImats_helpers`. The whole
`s_XImats` buffer (X **and** I) is streamed from global to shared:
- trig path: `s_XImats[ind] = d_robotModel->d_XImats[ind];` for `ind` in `[0,XI_size)`
  (`:281-283`).
- non-trig path: `cgrps::memcpy_async(tgrp, s_XImats, d_robotModel->d_XImats, XI_size)`
  (`:296`).
The serial recompute loop at `:301-413` then OVERWRITES only the X region (the I
region is never touched after the bulk copy — that's why inertia is effectively a
compile-time constant baked through `d_XImats`).

### 1c. Upstream source of the numbers (URDFParser; not emitted, but the formula
the device must replicate)
`URDFParser/Link.py:48-65` (`build_spatial_inertia`): the 6×6 is assembled from
`mass`, the 3×3 inertia tensor (`InertiaSet.to_matrix()`, `URDFParser/InertiaSet.py:15-16`),
and the COM translation `c` (`self.origin.translation`, used as the skew `com_trans`):
```
mc      = m * skew(c)              # 3x3
mccT    = mc @ skew(c).T           # 3x3
topLeft = I3x3 + mccT
I6 = [[topLeft, mc],[mc.T, m*eye(3)]]   # Link.py:60-63
```
This is the EXACT on-device reconstruction the runtime path must emit (§2c).
`get_Imats_ordered_by_id` is `Robot.py:818-819`; per-link mass at `Link.py:41`;
3×3 inertia at `Link.py:42`; COM at `Link.py:34/54-55`.

### 1d. Floating-base base inertia
When `include_base_inertia=True`, `Imats[0]` (the root/base link) is also baked
(`_topology_helpers.py:122-129`, `gen_get_XI_size` adds `+36`). Most current
callers pass `include_base_inertia=False` (the base link is the fixed world frame
with zero inertia — `URDFParser/URDFParser.py:78,318`). See §6 risk R1.

**That is the complete inventory.** No other file bakes a spatial-inertia number;
every algorithm consumes them indirectly through `s_XImats[36*n + …]`.

---

## 2. Runtime-param data model

### 2a. Recommended: a 10-param-per-link runtime buffer, NOT a full 6×6
Layout per link (Featherstone "inertial parameters", the system-ID-natural basis):
```
p[10] = [ m, mcx, mcy, mcz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz ]   // I about the link ORIGIN
```
i.e. `[m, mc(3), I_origin(6)]`. Total buffer = `10*N` floats (`10*(N+1)` if base
inertia is included). Choosing `mc` (first moment) rather than `c` keeps the
device math purely affine (no divide by m → robust when m→0 for massless links)
and matches Pinocchio/`pin.Inertia` ordering, which the RBDReference oracle and the
domain-randomization use-cases speak natively.

Rationale for 10-param over full 6×6:
- 10·N vs 36·N device memory + smem rebuild traffic (3.6× smaller).
- It is the actual DoF of a rigid-body inertia (6×6 spatial inertia has only 10
  independent entries); shipping 36 invites inconsistent/non-physical matrices.
- system-ID and domain-randomization perturb exactly these 10 numbers per link.

Provide a "full-6×6 escape hatch" only if a user needs non-physical inertias
(rare); default and documented path is 10-param.

### 2b. Where it lives — extend `robotModel<T>`, plus a per-call override pointer
Two complementary surfaces:

1. **`robotModel<T>` gains `T *d_inertia_params;`** (the default/current values),
   allocated and filled at init.
   - Struct emit: `GRiDCodeGenerator/GRiDCodeGenerator.py:1069-1073` (add the field;
     guard it behind the runtime flag so the baked variant's struct is byte-identical).
   - Init emit: `gen_init_robotModel` at `_topology_helpers.py:889-904` — add
     `h_robotModel.d_inertia_params = init_inertia_params<T>();` next to
     `d_XImats`/`d_topology_helpers` (`:898-899`). A new
     `gen_init_inertia_params` host helper (modeled on `gen_init_XImats`
     `:88-184` but writing the 10-param vector via a new
     `Robot.get_inertia_params_ordered_by_id()` — see §2d) fills `10*N` floats and
     `cudaMemcpy`s them.

2. **A new public mutator** so users can change params without re-init:
   `set_inertia_params<T>(robotModel<T>*, const T* h_params)` — a thin
   `cudaMemcpy` into `d_inertia_params`. This is the system-ID / payload entry
   point. (Optionally also a per-timestep `d_inertia_params` argument threaded
   through the kernels for true per-sample domain randomization — see §3 "stretch".)

Keeping the default values inside `robotModel<T>` means the runtime variant is
turn-key: `init_robotModel` produces a model that reproduces the baked answer
bit-for-bit until the user calls `set_inertia_params`.

### 2c. On-device rebuild of the 6×6 (the new code in `load_update_XImats_helpers`)
In `gen_load_update_XImats_helpers` (`_topology_helpers.py:241-427`), under the
runtime template (§3), REPLACE the I-region of the global→shared copy with an
on-device construction. Concretely:

- The bulk copy at `:281-283` / `:296` currently moves all `XI_size` floats. Split
  it: copy only the X region (`36*n` floats, plus the Xhom tail if present) from
  `d_XImats`, and SKIP the I region (offsets `[36*n, 36*2*n)`).
- Add a parallel loop over links that reads `d_robotModel->d_inertia_params[10*jid
  .. +10]` (or a passed-in override pointer) and writes the 36 floats of
  `s_XImats[36*(n+jid) + …]` using the closed form from `Link.build_spatial_inertia`
  (§1c). Emit it as a small `__device__` helper `build_spatial_inertia_6x6(T* dst,
  const T* p10)`:
  ```
  m = p[0]; mc = {p[1],p[2],p[3]};
  // skew(mc): S = [[0,-mcz,mcy],[mcz,0,-mcx],[-mcy,mcx,0]]
  // topLeft = I3x3(p[4..9]) + skew(mc)@skew(mc).T / m   (== I3x3 + mccT)
  // BR = m*eye(3); TR = skew(mc); BL = skew(mc).T
  ```
  Note: `mccT = m·skew(c)·skew(c)^T = skew(mc)·skew(mc)^T / m`. To avoid the divide
  (massless links), either (a) also ship `c` precomputed, or (b) store the 6 entries
  of `topLeft` directly in the 10-param block as `I_about_origin` (recommended:
  bake `topLeft` rather than `I_com` so the device does `topLeft + 0` and only the
  off-diagonal `mc` skew is assembled — zero divides, and it matches the URDF parser
  which already folds `mccT` into `topLeft` at `Link.py:60`). **Decision: store the
  10-params as `[m, mc(3), topLeft_6]` so the device rebuild is divide-free and is a
  pure scatter+skew, ~20 FMA per link.**

- The TL→BR symmetry copy at `:415-424` already exists for X; the I-block 6×6 is
  fully written by the rebuild so it does not rely on that copy.

### 2d. URDFParser support
Add `Link.get_inertia_params()` returning `[m, m*c, topLeft_6]` (reuse the exact
intermediates already computed in `build_spatial_inertia`, `Link.py:54-63` — store
`mc` and `topLeft` as attributes during that call) and
`Robot.get_inertia_params_ordered_by_id()` mirroring
`get_Imats_ordered_by_id` (`Robot.py:818-819`). Pure additive; no behavior change
to the baked path. Equivalence requirement: `build_spatial_inertia_6x6(params)`
on-device must reproduce `get_spatial_inertia()` to the last bit (§5).

---

## 3. Two-variant selection mechanism

**Codegen flag** (host side): add `runtime_inertia=False` to
`GRiDCodeGenerator.gen_all_code` (`GRiDCodeGenerator.py:1557-1559`) and a CLI flag
`--runtime-inertia` in `GRiDCodeGenerator/cli.py` (`add_argument` block
`:42-49`, plumb through `parseInputs` return `:77` and `main` `:104-107`). Store as
`self.runtime_inertia` (like `self.include_fixed_kinematic_targets` at
`GRiDCodeGenerator.py:1578`).

**Kernel-side template bool** (device side): the flag controls emission, but make
the actual switch a `bool RUNTIME_INERTIA` template parameter on
`load_update_XImats_helpers` (it already has
`template <typename T, bool SKIP_FLOATING_BASE_X = false>` at
`_topology_helpers.py:269` — add `bool RUNTIME_INERTIA = false`). Inside, gate the
I-region copy-vs-rebuild with `if constexpr (RUNTIME_INERTIA)`. The call sites set
it from `self.runtime_inertia` in `gen_load_update_XImats_helpers_function_call`
(`_topology_helpers.py:190-221`, where the `tparams` for `SKIP_FLOATING_BASE_X` are
already assembled). Default `false` guarantees the baked path stays byte-identical.

This two-axis design (host flag selects WHICH header you ship; template bool keeps
the elision a compile-time `if constexpr`) means:
- A baked-only header has the runtime branch fully `if constexpr`-dead → zero perf
  or smem cost, identical SASS (verify per §5 byte-identical gate).
- A runtime header can still default kernels to the baked-fast specialization and
  only opt specific launches into `RUNTIME_INERTIA=true` (useful for a hybrid
  build).

### Perf cost of the runtime path (be honest about it)
- **smem:** none extra — the rebuild writes into the SAME `s_XImats` I-region; no
  new shared buffer (the 10-param source is read straight from global / passed
  pointer per link, into registers).
- **registers:** the `build_spatial_inertia_6x6` helper is ~10 inputs + a 3×3 skew;
  modest, fully inside the existing serial/parallel XImat-load section. Should not
  move occupancy for the dominant robots.
- **global traffic:** runtime path reads `10*N` floats of params instead of `36*N`
  floats of baked I → actually LESS global traffic, but adds ~20 FMA/link of
  reconstruction on the load. Net: a few extra cycles on the once-per-kernel XImat
  load; negligible vs the per-link recursion. The bigger latent cost is if a
  per-timestep override pointer is threaded (loses the `memcpy_async`-from-constant
  coalescing) — keep that behind a separate "stretch" path.
- **The real risk is the floating-base X[0] interaction** (§6 R1) and codegen
  byte-identity of the baked path (§6 R3), NOT steady-state kernel perf.

---

## 4. Algorithms affected

Because the rebuild targets the shared `s_XImats` I-region (§0), **no algorithm
body changes.** The affected set is "everything that reads `s_XImats[36*n + …]`",
which is reached transitively through the one shared helper:
- `crba`, `inverse_dynamics` (RNEA/ID), `aba`, `direct_minv`, `forward_dynamics`,
  `inverse_dynamics_gradient` (id_du), `forward_dynamics_gradient` (fd_du),
  `idsva_so` (body+world), `fdsva_so`, `integrator`/`integrator_gradient`,
  `plant`. (`eepose_gradient_hessian` reads only the X/Xhom region — inertia-free,
  unaffected.)
All of them call `load_update_XImats_helpers` (e.g.
`_inverse_dynamics.py:382`); flipping its `RUNTIME_INERTIA` template parameter at
those call sites is the ONLY per-algorithm touch, and it's mechanical (done in the
shared `gen_load_update_XImats_helpers_function_call`). CRBA's `SKIP_FLOATING_BASE_X`
specialization must compose with `RUNTIME_INERTIA` (both are template bools on the
same function — verify the 2×2 instantiation matrix compiles).

---

## 5. Testing

1. **Byte-identical baked gate (regression):** with `runtime_inertia=False`, the
   generated `grid.cuh` must be diff-clean vs current `HEAD` for iiwa14/go2/g1
   (fixed+floating). This proves the new code is fully `if constexpr`-dead in the
   baked variant. Run codegen for all bench cells and `git diff --exit-code grid.cuh`.

2. **Runtime-reproduces-baked (equivalence, the headline correctness test):**
   generate a `runtime_inertia=True` header, `init_robotModel` (which fills
   `d_inertia_params` from the SAME URDF), run RNEA/ABA/CRBA/Minv/ID-grad/FD-grad/
   IDSVA-SO across random `(q,qd,u)` and assert the runtime-path output matches the
   baked-path output **bit-for-bit** (same FP rounding because the reconstructed
   6×6 must equal the baked 6×6 exactly — this is why §2d demands
   `build_spatial_inertia_6x6` reproduce `get_spatial_inertia()` to the ULP; if it
   can't, store the 6 `topLeft` entries verbatim so only the `mc` skew is assembled,
   guaranteeing exactness). This is a correctness-only, per-(robot,base) parallel
   test — fits the existing CUDA-equivalence harness pattern (see memory
   `feedback_parallel_equivalence_testing`).

3. **Perturbed-params vs RBDReference (the actual feature):** call
   `set_inertia_params` with perturbed inertia (scale a link's mass, shift a COM),
   then compare CUDA output against RBDReference re-run with the SAME perturbed
   inertia (rebuild the Python `Robot` with the new `set_inertia` values, or inject
   the perturbed `Imat`). Covers system-ID / payload semantics end-to-end. Use the
   pinocchio-backed oracle in the RBDReference submodule (memory
   `project_grid_pinocchio_reference_backlog`).

4. **PERF isolation:** keep any timing of the runtime path OUT of the correctness
   sweep and off-box during other benches (memory `feedback_no_cpu_during_bench`).
   Quantify the once-per-kernel XImat-load delta on iiwa14/g1 only.

---

## 6. Open questions / risks

- **R1 — Floating-base base inertia + X[0].** The floating root's X[0] is the heavy
  quat→rot block, gated by `SKIP_FLOATING_BASE_X` (`_topology_helpers.py:262-269,
  309-311`). For floating-base WITH `include_base_inertia=True`, the base spatial
  inertia (`Imats[0]`) would also need a runtime slot (`10*(N+1)` layout, offset
  shift). Decide whether D.4 v1 supports runtime BASE inertia (payload on the
  floating trunk is a real use-case) or restricts runtime params to the articulated
  links (`Imats[1:]`) in v1. Recommend: support base inertia from the start since
  it's just one more 10-param block, but verify the `+36`/`include_base_inertia`
  offset bookkeeping (`gen_get_XI_size` `:7`, the `Imats[1:]` slice at
  `_topology_helpers.py:122-123`) stays consistent in BOTH the param-buffer index
  and the `s_XImats` write offset.

- **R2 — F-batch / T3 (mimic) collision (HIGH coordination risk).** T3
  (`GRiD-T3-mimic-codegen`, branch `mimic-codegen`) edits the XImats q-fold and
  inertia indexing in this very function (`gen_load_update_XImats_helpers`,
  `gen_init_XImats`) — per HANDOFF it constant-folds mimic logic into the q→XImat
  emit. D.4 also rewrites the I-region of this function. **These two MUST be
  sequenced, not merged blind.** Recommendation: land D.4 AFTER T3 merges (T3 is in
  the in-flight F-batch; HANDOFF orders T3 before T5). D.4's change is confined to
  the I-region copy/rebuild; T3's is the X-region q-fold — they touch disjoint
  halves of the same function, so a post-T3 rebase is mechanical IF D.4 keeps its
  edits inside an `if constexpr (RUNTIME_INERTIA)` block that doesn't perturb T3's
  serial X loop. Mimic does NOT change inertia (mimic couples joint velocities, not
  link masses), so the 10-param-per-LINK model is mimic-agnostic — index params by
  LINK id, never by reduced joint/DoF id.

- **R3 — T4 (fext) signature mutation.** T4 (`external-forces`) is the only F-batch
  task that adds a trailing kernel arg (`d_f_ext`). D.4 should mirror that
  convention if it threads a per-call `d_inertia_params` override (add as a trailing
  pointer, default `nullptr` → fall back to `d_robotModel->d_inertia_params`).
  Coordinate ordering so the two signature changes don't both land un-rebased.

- **R4 — T5 (tier enum rename) `TIER_PERF→TIER_SHARED`.** Pure value-rename per
  HANDOFF; D.4 touches no tier names, so no real conflict — just rebase text.

- **R5 — Byte-identity of the baked path.** The single hardest guarantee. The
  runtime branch must be 100% `if constexpr`-dead when `RUNTIME_INERTIA=false`,
  AND the `robotModel<T>` struct must stay identical in the baked-only build (gate
  the new `d_inertia_params` field behind the host flag, or always-emit it but
  prove the struct-size/codegen diff is empty when unused). Enforced by test §5.1.

- **R6 — exact reconstruction.** If `build_spatial_inertia_6x6` does NOT reproduce
  the baked 6×6 to the ULP (e.g. due to `mccT` recomputation order), test §5.2's
  bit-for-bit gate fails. Mitigation already chosen in §2c/§2d: ship the 6 `topLeft`
  entries pre-folded so the device only assembles the `mc` skew blocks (no
  arithmetic on the symmetric core) → exact by construction.

- **R7 — `gridDataKind` / init paths.** `init_robotModel` is independent of
  `gridData`/KIND, so the param buffer init is a clean add; but confirm `close_grid`
  (`GRiDCodeGenerator.py:1419`) frees `d_inertia_params` to avoid a leak.

---

## 7. Suggested implementation order (post-T3 merge)
1. URDFParser: `Link.get_inertia_params` + `Robot.get_inertia_params_ordered_by_id`
   (cache `mc`/`topLeft` in `build_spatial_inertia`). [additive, no risk]
2. Host: `gen_init_inertia_params` + `robotModel` field (flag-gated) + `set_inertia_params`
   mutator + `close_grid` free.
3. Device: `build_spatial_inertia_6x6` helper + `RUNTIME_INERTIA` template branch in
   `gen_load_update_XImats_helpers` (split the X-vs-I copy).
4. Wire `runtime_inertia` flag through `gen_all_code` + `cli.py` + the
   `gen_load_update_XImats_helpers_function_call` template params.
5. Tests §5.1 (byte-identical), §5.2 (runtime==baked bit-for-bit), §5.3 (perturbed
   vs RBDReference).
