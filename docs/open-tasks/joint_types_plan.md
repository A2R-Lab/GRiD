# Additional URDF Joint Types — Support Plan + Ripple Assessment

**Status (2026-05-30):** scoping/planning doc, READ-ONLY exploration. No code
changed. Companion to `urdf_feature_matrix.md` (T1 audit) and
`d2_codegen_mimic_plan.md` (T3 mimic NV≠NQ pattern). Gating criterion (user):
**add joint types that cheaply prevent future breakage; defer ones whose ripple
is huge.**

## Why this doc exists

The P1-C broad sweep ([broad-coverage findings] memory) hit two model-quirk
walls already catalogued: **rizon4** (broken zero-inertia asset, not a joint-type
issue — skip) and **fr3** (`<mimic>` joint, NV≠NQ, handled by the mimic path).
The next class of breakage is **unsupported joint *types*** which today hit a
hard, silent failure:

```
URDFParser/Joint.py:259-263  (set_type else-branch)
    print('Only revolute and fixed joints currently supported (outside of floating base)!')
    exit()
```

A bare `exit()` with no traceback is the worst failure mode — it kills the whole
process. **The single cheapest, highest-value robustness fix in this whole plan
is replacing that `print+exit` with a typed exception** (e.g.
`raise UnsupportedJointTypeError(jtype)`), independent of which types we
actually implement. That alone converts every future "GRiD silently died on a
planar/spherical robot" into a catchable, debuggable error.

## The two reusable NV≠NQ mechanisms (the central insight)

GRiD already supports **two** joints where the velocity DOF count (NV) differs
from the position coordinate count (NQ) and/or the config-space update is not a
plain vector add. Every new multi-DOF / manifold joint should be framed as
"reuse one of these," not "invent new machinery":

1. **Floating base** — `Joint.set_type` `floating` branch
   (`URDFParser/Joint.py:209-258`). NV=6, NQ=7 (quaternion) or 6 (rpy). Has a
   **6×NV motion subspace `S` matrix** (not a single index — see
   `Joint.py:248-258`), a **multi-symbol `position_symbols`** list, and
   `local_q_dim>1`. Config-space update is an **SE(3)/quaternion Lie exp**
   (`RBDReference.integrate`, `RBDReference.py:235-266`; Jacobian
   `dIntegrate`, `:268-312`). DOF accounting already tolerates NQ−NV=1 via
   `Robot.get_num_pos = get_num_vel + (1 if floating&quaternion else 0)`
   (`Robot.py:241`).

2. **Mimic** — `Joint.set_mimic` / `get_num_dof→0` (`Joint.py:375-381`),
   reduced-model NV via `Robot.get_num_vel` summing `get_num_dof()`
   (`Robot.py:251`), `q_for_joint` fold (`Robot.py:210-227`), dense
   `_dense_q_offset_by_id`/`_dense_v_offset_by_id` maps (`Robot.py:170-208`),
   `get_joint_index_v/q`. This is the "NV<NQ-naively, route coordinates through
   a map" machinery. **CUDA codegen for mimic is still PENDING** (D.2 plan,
   4 phases) — so "reuse mimic" on the CUDA side means "reuse the *plan*,"
   which itself isn't landed yet.

### The one piece of genuinely shared NEW machinery every multi-DOF type needs

The CUDA emitter represents each joint's `S` as a **single signed index**:
`Robot.get_S_index_by_id` / `get_S_sign_by_id` (`Robot.py:840-855`) scan the
flat 6-vector for the one entry with `abs==1` and **raise if there isn't exactly
one**. `get_S_inds` (`Robot.py:887`) emits that signed index into
`h_topology_helpers`, and `_topology_helpers.py:758,815,833` consume it as a
scalar `S_ind`/`S_sign`. Floating base is **special-cased around this** (its 6×6
`S` is handled by separate `_inner_floating` codepaths per algorithm, e.g.
`_crba.py:198 gen_crba_inner_floating`), NOT through `get_S_index_by_id`.

**Consequence:** any *non-floating* joint with a multi-column `S`, or even a
single-column `S` with a non-cardinal (skew) axis (two/three nonzero entries),
**breaks `get_S_index_by_id` immediately** (the `raise ValueError("Joint
subspace does not contain a unit axis.")`). This is the single largest shared
ripple. Two ways out, in increasing cost:
- **(a) General-axis-but-still-1-DOF** (continuous w/ skew axis, helical):
  extend the emit to carry a **full 6-vector `S` column** instead of a signed
  index. Medium, one-time, benefits every 1-DOF joint.
- **(b) Multi-column `S`** (planar, spherical): needs per-joint `S` *matrices*
  in the non-floating emit path — i.e. generalize the floating-base
  `_inner_*`-style handling to **any NV-per-joint > 1**, OR (cheaper) decompose
  the multi-DOF joint into a short internal chain of 1-DOF joints + a 0-mass
  dummy link (the classic Featherstone trick). The decomposition route lets
  spherical/planar **reuse the existing single-index `S` machinery entirely**
  at the cost of dummy links and a non-vector-add quaternion update for the
  ball case.

---

## Per-type assessment

For each type: DOF, NQ vs NV, motion subspace `S`, config-space update, ripple
ratings (S/M/L) into the seven surfaces, NEW-vs-reuse tag, validation oracle,
and the robustness note.

Ripple surfaces (columns): **[Parse]** URDFParser/Joint.set_type ·
**[S/XI]** S + XImats emission (`_topology_helpers.py`) · **[CUDA-NV]** every
NV-scaled CUDA output (crba/minv/rnea/fd/grad/SO) · **[Integ]** integrator +
config-space ops (`RBDReference.integrate/dIntegrate`) · **[RBD]** RBDReference
numpy reference.

---

### 1. continuous (1-DOF) — VERIFY, do not re-implement

- **DOF** 1. **NQ=NV=1.** No NV≠NQ. Treated identically to `revolute`
  (`Joint.py:160` `if self.jtype in ('revolute','continuous')`), limits set to
  ±inf. `S` = single cardinal axis (1 nonzero) → fits `get_S_index_by_id`.
- **Config-space update:** plain vector add `q += v*dt`. **Wraparound:**
  Pinocchio models a continuous joint's q as a **2-vector `(cos θ, sin θ)`** on
  SO(2) (NQ=2, NV=1!) and its `integrate` rotates it; GRiD stores a **raw scalar
  angle** and adds. For *dynamics* (M, c, τ, qdd and all derivatives — which
  depend only on `θ mod 2π` through `cos/sin`) the two are **numerically
  identical**; they diverge only if you (a) compare the raw q value to
  Pinocchio's 2-vector q, or (b) integrate over many turns and care about the
  unwrapped value. The equivalence harness should compare *outputs*
  (transforms/dynamics), not raw q, for continuous joints.
- **Ripple:** Parse **S** (already done) · S/XI **S** · CUDA-NV **S** ·
  Integ **S** (scalar add already correct for dynamics) · RBD **S**.
- **NEW vs reuse:** fully **reuse** (it IS revolute). The only action is a
  **test/verification** that wraparound doesn't leak into the comparator, plus a
  doc note that GRiD's continuous-joint q is a raw angle (NQ=1), diverging from
  Pinocchio's SO(2) NQ=2 representation *only* in the raw-q comparison.
- **Oracle:** `pin.JointModelRUBX/Y/Z` (revolute-unbounded) — but compare
  dynamics outputs, not q.
- **Robustness:** already non-fatal (no `exit()`).
- **Recommendation: cheapest. Do first — it's a verification + comparator
  guard, zero new dynamics code.**

---

### 2. helical / screw (1-DOF) — cheap, exposes the general-axis `S` gap

- **DOF** 1. **NQ=NV=1.** No NV≠NQ. One scalar θ drives **coupled** rotation
  *and* translation along the same axis: `X = rot(axis,θ)·xlt(pitch·axis·θ)`.
- **Motion subspace:** `S = [axis_xyz, pitch·axis_xyz]` — a **single column but
  with up to 6 nonzero entries** (rotation + pitch-scaled translation). This is
  the key: even a cardinal-axis screw has `S = [0,0,1, 0,0,pitch]` → **two**
  nonzero entries → **breaks `get_S_index_by_id` `abs==1` scan** (`Robot.py:840`).
- **Config-space update:** plain vector add `θ += v*dt` (1-DOF, no manifold).
- **Ripple:** Parse **S** (one new `set_type` branch, like prismatic+revolute
  fused) · S/XI **M** (forces general-6-vector-`S` emit, route (a) above) ·
  CUDA-NV **M** (every kernel that today does `S_sign * x[S_ind]` must do a
  6-dot-product `Sᵀ·x`; but this is a *localized, mechanical* change shared with
  general-axis revolute) · Integ **S** (scalar add) · RBD **S** (numpy uses the
  full symbolic `S`/`Xmat_sp` already, so reference is ~free).
- **NEW vs reuse:** the dynamics recursion is **reuse** (1-DOF, same as
  revolute); the **NEW** piece is the **general-6-vector-`S` emit path** —
  which is **the same change non-cardinal-axis revolute needs**
  (`urdf_feature_matrix.md` §1). So helical is the natural *forcing function* to
  build route (a) once and get skew-axis revolute/prismatic for free.
- **Oracle:** `pin.JointModelHelicalX/Y/Z` (or `pin.JointModelHelicalUnaligned`).
- **Robustness:** today hits `print+exit`. Typed exception fixes that.
- **Recommendation: do *second*, bundled with the general-axis-`S` work, because
  it converts the existing "skew axis → silent `None` → crash"
  (`urdf_feature_matrix.md` §1) into a solved case.**

---

### 3. planar (3-DOF) — reuse mimic NV-bookkeeping + decompose

- **DOF** 3 (2 in-plane translations + 1 normal-axis rotation).
  **NQ=NV=3** (no quaternion → no NV≠NQ *coordinate* mismatch), **but NV>1 per
  joint**, which the non-floating CUDA emit has never seen.
- **Motion subspace:** 6×3 `S` (two translation columns + one rotation column).
  Multi-column → **route (b)**.
- **Config-space update:** plain **vector add** on all 3 (planar is a *vector*
  group, no Lie exp needed) — so Integ ripple is **S**, unlike spherical.
- **Ripple:** Parse **M** (new multi-DOF branch in `set_type`, mirror the
  `floating` branch's `position_symbols`/`local_q_dim`/`dof` multi-symbol
  pattern, `Joint.py:209-258`) · S/XI **L** (multi-column non-floating `S`;
  either generalize floating `_inner_*` to NV-per-joint=3, or decompose) ·
  CUDA-NV **L** (per-joint q/v block indexing for a 3-wide non-root joint — the
  emitters assume 1 slot/joint outside the root) · Integ **S** (vector add) ·
  RBD **M** (numpy reference can mirror floating's multi-symbol handling).
- **NEW vs reuse:**
  - **reuse** the *DOF-accounting* (`get_num_dof` returning 3,
    `_dense_v_offset_by_id` advancing by `joint.dof` — `Robot.py:190` already
    does `v_cursor += joint.dof`, so multi-DOF non-root joints slot in for free).
  - **reuse** the *mimic dense-offset map* idea: `get_joint_index_v(jid)` already
    returns a block, so q/v routing is in place at the Robot layer.
  - **NEW**: the CUDA emit of a **multi-column `S` for a non-root joint** — the
    single biggest new chunk. **Cheaper alternative that makes it pure reuse:**
    decompose planar into **2 prismatic + 1 revolute** through 2 zero-mass dummy
    links (Featherstone). Then every joint is 1-DOF cardinal-`S`, NV-accounting
    is automatic, and **no S/XI or CUDA-NV change at all** — the ripple collapses
    to Parse-only (emit the dummy chain at parse time). **Strongly prefer
    decomposition.**
- **Oracle:** `pin.JointModelPlanar`.
- **Robustness:** today `print+exit`. Typed exception fixes that.
- **Recommendation: do *third*, via the prismatic/prismatic/revolute
  decomposition so it rides existing 1-DOF machinery (ripple → S).**

---

### 4. spherical / ball (3-DOF) — the one genuinely-new manifold case

- **DOF** 3 (rotation only). **NV=3, NQ=4** (unit quaternion) → **NV≠NQ, exactly
  the floating-base sub-pattern.** This is the type that *most* reuses
  floating-base machinery and *least* reuses mimic.
- **Motion subspace:** 6×3 `S = [[I₃],[0]]` (angular-only) in body frame —
  multi-column → route (b).
- **Config-space update:** **SO(3) quaternion exp** — NOT a vector add. This is
  the floating-base rotation block *minus* the translation:
  `q_quat_new = quat_mul(q_quat, exp_quat(0.5·ω·dt))`,
  `dIntegrate` = SO(3) adjoint / right-Jacobian. `RBDReference.integrate`
  already has every primitive needed (`_quat_exp_from_half_omega`,
  `_quat_mul_xyzw`, `_so3_right_jacobian`, `_so3_exp` — `RBDReference.py:98-233`);
  a spherical joint is the **rotation-only restriction** of the existing
  free-flyer prefix code.
- **Ripple:** Parse **M** (quaternion-parameterized 3-DOF branch; mirror
  `floating` using_quaternion path, but rotation-only) · S/XI **L** (3-column
  angular `S` for a non-root joint) · CUDA-NV **L** (NV≠NQ *and* multi-column
  S for a mid-chain joint — combines planar's multi-col problem with
  floating's NV≠NQ) · Integ **M** (quaternion exp — **reuse** the floating
  primitives, but `integrate`/`dIntegrate` currently hardcode a single 6-wide
  free-flyer *prefix* at q[0:7]/v[0:6]; supporting a spherical joint **mid-chain**
  means generalizing those from "fixed prefix block" to "iterate joints, apply
  the per-joint retract by type" — a real but bounded refactor) · RBD **M**
  (same generalization on the numpy side; primitives exist).
- **NEW vs reuse:**
  - **reuse** floating-base quaternion math (exp/log/Jacobian) and the
    `get_num_pos = nv + (#quaternion joints)` accounting idea — though
    `Robot.py:241` currently only counts **one** quaternion offset (the root);
    **NEW**: generalize that `+1` to `+ (number of quaternion joints)`.
  - **reuse** mimic's per-joint `q_for_joint` block extraction (returns a
    4-vector quaternion block cleanly).
  - **NEW**: `integrate`/`dIntegrate` becoming **per-joint retract dispatch**
    instead of a hardcoded free-flyer prefix; and multi-column non-root `S` emit
    (shared with planar). **Decomposition is NOT clean here** — you cannot
    decompose a ball joint into 3 revolutes without gimbal lock / a different
    q-parameterization, so the quaternion manifold update is unavoidable. This is
    the **most genuinely-new** type.
- **Oracle:** `pin.JointModelSpherical` (and `JointModelSphericalZYX` for the
  Euler-param variant — avoid; use the quaternion `Spherical`).
- **Robustness:** today `print+exit`. Typed exception fixes that.
- **Recommendation: do *fourth/last* of the implemented types. It's the only one
  that forces the `integrate`/`dIntegrate` per-joint-retract generalization and
  multi-quaternion NQ accounting. High value for humanoid/quadruped ball joints,
  but highest ripple — schedule after the cheap wins and the multi-column-S
  groundwork from planar.**

---

### 5. translation / cartesian (3-DOF prismatic) — trivial via decomposition

- **DOF** 3 pure translations. **NQ=NV=3**, vector group (no manifold).
- **Motion subspace:** 6×3, three translation columns. Multi-column, but each
  column is a single cardinal entry.
- **Config-space update:** plain vector add.
- **Ripple:** identical shape to **planar minus the rotation**, and **fully
  decomposes into 3 prismatic joints** through 2 dummy links → **Parse-only S**,
  everything else reuse.
- **NEW vs reuse:** **reuse** entirely via 3-prismatic decomposition.
- **Oracle:** `pin.JointModelTranslation`.
- **Robustness:** typed exception.
- **Recommendation: free rider on the planar decomposition — implement the
  decomposition helper once, get planar + translation + (cartesian) together.**

---

### 6. composite / multi-DOF (`<joint type="..."` chains, URDF has none natively)

- URDF itself is a tree of single joints; "composite" here means our own
  **decomposition output** (the dummy-link chains used for planar / translation /
  helical-if-needed), and the SRDF/xacro-implied multi-DOF constructs.
- **DOF/NQ/NV:** whatever the decomposition sums to. **Vector add** unless a
  ball link is involved.
- **Ripple:** the *enabling* work is a single **parse-time decomposition helper**
  (`URDFParser`) that, on encountering a planar/translation/(optionally screw)
  joint, injects N−1 zero-mass dummy links + N single-DOF joints with the right
  axes/origins. After that helper exists, every decomposable multi-DOF type is
  **Parse-S, everything-else-reuse**. The helper must: (a) give dummy links exact
  zero mass/inertia **and flag them so the rizon4-style degenerate-inertia
  detector does NOT reject them** (cross-ref `urdf_feature_matrix.md` §5 and the
  rizon4 finding — dummy links are *intentionally* massless), (b) preserve joint
  ordering / DFS renumber, (c) compose origins so the chain reproduces the single
  joint's transform.
- **NEW vs reuse:** the helper is **NEW** (one localized chunk); it then makes
  planar/translation **reuse**. Spherical cannot use it (manifold).
- **Recommendation: build the decomposition helper as the vehicle for planar +
  translation; do NOT over-generalize it to spherical.**

---

## Recommended ORDER (cheap-first, ripple-minimizing)

0. **(Robustness, do regardless) Replace `Joint.set_type` `print+exit`
   (`Joint.py:259-263`) with a typed `UnsupportedJointTypeError`.** Converts
   every future unsupported-type robot from a silent process kill into a
   catchable error. Touch the equivalence harness to xfail/skip on it (mirrors
   the rizon4/fr3 skip pattern). **Cost: trivial. Value: stops the worst
   breakage today.**

1. **continuous** — verify-only. Comparator guard so SO(2) wraparound
   (Pinocchio NQ=2 vs GRiD NQ=1) doesn't false-fail; no dynamics code.
   *Ripple: S across the board.*

2. **helical/screw + general-axis `S`** — bundle. Build the **full-6-vector-`S`
   emit** (route a); this simultaneously fixes the existing skew-axis-revolute
   `None`-crash (`urdf_feature_matrix.md` §1). *Ripple: M, localized.*

3. **planar + translation/cartesian** — build the **parse-time decomposition
   helper** (prismatic/prismatic/revolute and 3×prismatic through zero-mass dummy
   links). Reuses 1-DOF cardinal-`S` machinery + existing multi-DOF DOF
   accounting (`v_cursor += joint.dof`). *Ripple: S (everything but parse is
   reuse) — IF decomposed; L if emitted as native multi-column-S.*

4. **spherical/ball** — last. The only genuinely-new manifold type: generalize
   `integrate`/`dIntegrate` from a hardcoded free-flyer prefix to **per-joint
   retract dispatch**, generalize `get_num_pos` to count **multiple** quaternion
   joints, and emit a mid-chain 3-column angular `S`. Reuses the floating-base
   quaternion primitives wholesale. *Ripple: M–L, genuinely new.*

**Cross-cutting dependency:** steps 3 (native-S route) and 4 both want the
multi-column non-root `S` emit; the decomposition route in step 3 lets you defer
that to step 4 (spherical), where it's unavoidable.

## Cross-references

- `urdf_feature_matrix.md` — §1 (skew axis, fixed by step 2), §2 (planar,
  superseded by this doc's decomposition route), mimic status (CUDA pending).
- `d2_codegen_mimic_plan.md` — the canonical NV≠NQ CUDA pattern; spherical's
  NV<NQ emit should mirror its `get_joint_index_v` routing and per-phase
  validation.
- Broad-coverage findings (memory): **rizon4** = degenerate-inertia asset, NOT a
  joint-type issue (skip); the decomposition helper's dummy links must be
  whitelisted past any degenerate-inertia guard. **fr3** = mimic (NV≠NQ via
  reduction), the existing exemplar to copy.
- `get_S_index_by_id` / `get_S_sign_by_id` (`Robot.py:840-855`) and
  `get_S_inds` (`:887`) — the single-axis `S` assumption that steps 2–4 must lift
  or route around.
