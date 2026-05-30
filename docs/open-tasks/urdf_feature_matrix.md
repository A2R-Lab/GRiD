# URDF Feature Support Matrix

Status of URDF features in the GRiD parser (`URDFParser/URDFParser.py`) and joint
model (`URDFParser/Joint.set_type`), and how each propagates to the RBDReference
Python backend and the CUDA codegen.

Grounded in:
- `URDFParser/Joint.py` — `set_type`, `_axis_scale`, `set_mimic`, `set_damping`
- `URDFParser/URDFParser.py` — `parse_links`, `parse_joints`, `resolve_mimic_targets`

Legend: **supported** = parsed and correctly modeled; **partial** = parsed but
incomplete/limited; **missing** = not handled (or hits a hard guard).

| URDF feature | Status | Where | Notes |
|---|---|---|---|
| `revolute` joint | supported | `Joint.set_type` | cardinal axis only (see axis row) |
| `continuous` joint | supported | `Joint.set_type` | treated like revolute; limits set to ±inf |
| `prismatic` joint | supported | `Joint.set_type` | cardinal axis only |
| `fixed` joint | supported | `Joint.set_type` | dof 0, merged downstream |
| `floating` joint | supported | `Joint.set_type` | xyzw quaternion + [v;w] order, Pinocchio-aligned |
| `planar` joint | **missing** | `Joint.set_type` else-branch | hits `print(...) + exit()` guard |
| Cardinal `<axis>` (±x/±y/±z) | supported | `_axis_scale` | `np.isclose(abs(v), 1.0)` |
| Arbitrary / skew `<axis>` | **partial** | `_axis_scale` | returns `None` for non-cardinal → `Xmat_sp_free` left `None` |
| `<mimic>` joint | partial | `set_mimic` / `resolve_mimic_targets` | parsed + RBD-aware; **CUDA codegen pending (T3)** |
| `<dynamics damping=...>` | partial | `parse_joints` → `set_damping` | stored; used only in `rnea_grad` damping term |
| `<dynamics friction=...>` | **missing** | `parse_joints` | `friction` attribute never read |
| Position `<limit lower/upper>` | supported | `parse_joints` | stored as `joint.joint_limits` |
| Velocity / effort `<limit>` | **missing** | `parse_joints` | `velocity` / `effort` attributes never read |
| `<inertial>` present | supported | `parse_links` | mass + 6 inertia terms |
| Missing / degenerate `<inertial>` | **partial** | `parse_links` | absent inertial silently assumed "world base frame"; zero-inertia not flagged (rizon4) |

---

## Mimic joints

Mimic is **parsed and RBD-aware**: `<mimic joint=... multiplier=... offset=...>`
is captured by `Joint.set_mimic`, resolved to a stable jid post-renumber by
`URDFParser.resolve_mimic_targets`, and the RBDReference Python backend honors it
(reduced-model CRBA / RNEA, and `minv` falls back to `inv(crba(q))`).

**CUDA codegen is pending — tracked by T3 in `docs/d2_codegen_mimic_plan.md`.**
The generated kernels do not yet emit the alpha / alpha^2 mimic projections.

---

## Genuinely missing / partial features — proposals

For each: a **Python-first** proposal (land + test in `URDFParser` /
`RBDReference` first) and a **CUDA propagation** note.

### 1. Arbitrary / non-cardinal `<axis>` (partial)

`_axis_scale(axis, index)` returns the signed component only when
`np.isclose(abs(value), 1.0)`, else `None`. For a skew axis (e.g.
`0.577 0.577 0.577`) all three indices return `None`, so neither `Xmat_sp_free`
nor `S` is assigned and `set_type` falls through with `Xmat_sp_free = None`,
breaking downstream `Xmat_sp` construction.

- **Python-first:** build the rotation/translation about a general unit axis.
  For revolute, use the Rodrigues `rot` about the normalized `axis` and set
  `S = [axis_xyz, 0,0,0]` (prismatic: `S = [0,0,0, axis_xyz]`). Replace the
  three-way cardinal cascade with a single general-axis path; cardinal axes fall
  out as a special case. Add a parse test with a skew-axis URDF.
- **CUDA propagation:** the codegen emits per-joint transform/`S` from the same
  symbolic `Xmat_sp` / `S`, so a correct general-axis `set_type` flows through
  automatically — but verify the generated `S` is the full 6-vector (not a
  single-axis index assumption) in the transform/`crba`/`rnea` emitters.

### 2. Planar joint (missing)

`planar` hits the `set_type` else-branch guard (`print + exit`).

- **Python-first:** model planar as a 3-dof joint (2 translation + 1 rotation in
  the joint plane). Add a `planar` branch to `set_type` building `Xmat_sp_free`
  from the two in-plane translation symbols and the normal-axis rotation symbol,
  with the matching 6x3 `S`. Mirror the `floating` branch's multi-dof handling
  (`position_symbols`, `local_q_dim`, `dof`). Add parse + RNEA/CRBA equivalence
  tests.
- **CUDA propagation:** multi-dof non-floating joints are a new shape for the
  emitters; ensure per-joint `q`/`v` indexing and the multi-column `S` are
  handled like floating-base (which is the only current >1-dof joint). Likely the
  larger half of the work.

### 3. `<dynamics damping>` / `friction` (partial / missing)

`damping` is parsed and stored, and is consumed in `rnea_grad`'s `dc_dqd`
velocity-damping term, but **not** added to the inverse/forward-dynamics bias.
`friction` is never parsed.

- **Python-first:** thread stored `damping` into the RNEA/ABA bias
  (`tau += damping * qd`) behind an explicit flag so existing equivalence tests
  (which assume no damping) stay green; parse `friction` into a new field and
  apply Coulomb friction similarly. Add damping/friction-specific tests.
- **CUDA propagation:** add the per-joint damping/friction coefficients to the
  generated robot constants and the bias accumulation in the rnea/aba kernels;
  gate behind the same flag.

### 4. Velocity / effort `<limit>` (missing)

Only `lower`/`upper` position limits are parsed.

- **Python-first:** extend the `<limit>` parse to read `velocity` and `effort`,
  store as `joint.velocity_limit` / `joint.effort_limit`, and expose via
  accessors. No dynamics change — metadata only.
- **CUDA propagation:** optional; emit as robot constants only if a kernel needs
  saturation. Default: Python-side metadata only.

### 5. Degenerate / missing `<inertial>` detection (partial)

A link without `<inertial>` is silently assumed to be the fixed world base
(`URDFParser.py:74`). A link with zero/degenerate inertia (e.g. rizon4) is not
flagged and yields singular dynamics downstream.

- **Python-first:** distinguish "this is the declared root" from "a non-root link
  is missing inertia." For non-root links, raise (or warn via the additive
  strict-parse API in `PINOCCHIO_ALIGNMENT_BACKLOG.md`) on absent/degenerate
  inertia instead of silently zeroing. Add a regression test on a rizon4-like
  URDF.
- **CUDA propagation:** none — pure parse-time validation; failing fast prevents
  emitting a degenerate model.
