# D.2 CUDA Codegen Mimic-Awareness — Implementation Plan

**Status (2026-05-29):** scoping doc, not yet implemented. Authored by the read-only scoping sub-agent; main agent verified the algebraic-decomposition reasoning matches today's RBDReference landings.

## Executive summary

D.2 CUDA codegen mimic-awareness is a **2.5–4 day project** that should be **phased into 4 independently-mergeable PRs**. The codegen currently assumes `NJ == NV` everywhere: it emits `NUM_JOINTS = get_num_pos()` (already reduced) but uses `range(NJ)`-style jid loops and writes `s_qd[jid]`/`s_c[jid]` instead of `s_qd[get_joint_index_v(jid)]` scaled by `α`. The clean path is to (1) introduce per-codegen helpers `_v_slot_cpp(jid)`, `_alpha_cpp(jid)`, and emit a constant-folded `MIMIC_ALPHA[NJ]` array on the C++ side; (2) sweep per-algo emit sites to use `+= α * (...)` into the v-slot; (3) for `Minv`/`ABA`/`FD`/`FD-grad`/`integrator` whose recursions don't fold cleanly, route to algebraic decomposition (`Minv = inv(CRBA)`, `qdd = Minv·(u − c)`, `qdd_dq = −Minv · rnea_grad_dq`) — exactly mirroring the RBDReference fallback. Implementer order: (P1) emit-side macros + ID + CRBA, (P2) Minv-via-CRBA-inv + FD-via-decomp, (P3) ID-du + FD-du + integrator, (P4) ee_pose + idsva_so + fdsva_so. Validation: un-skip fr3/h1_2-fixed (then -floating) per-algo in `test_cuda_executable_equivalence.py` after each phase.

## Background

After the 2026-05-29 batch, `RBDReference` is mimic-aware end-to-end and `URDFParser.Robot` exposes `get_joint_index_v(jid)`, `get_joint_index_q(jid)`, `q_for_joint(jid, q)`, `_dense_v_offset_by_id`, plus `joint.is_mimic` / `joint.get_mimic_multiplier()` / `joint.get_mimic_offset()`. The CUDA codegen has **zero** mimic awareness today: it indexes `s_qd[jid]` and assumes `NJ == NV` throughout. With mimic-aware URDFParser, `NUM_VEL = NV` drops below `NJ`, output buffers shrink, and per-jid emit loops write into the wrong slots or out-of-bounds slots.

**Subtle observation:** `_code_generation_helpers.py` line ~710 emits `const int NUM_JOINTS = get_num_pos()` — so the C++ macro `NUM_JOINTS` is actually `nq` (reduced for mimic). The Python-side `self.robot.get_num_joints()` returns the raw `len(self.joints)` (un-reduced). This naming mismatch is **dangerous** going forward and must be addressed in P1 (rename / dual emit).

## 1. Inventory by file (every codegen file affected)

### Helpers (touched by every algo)
- **`helpers/_code_generation_helpers.py`** — rename emitted `NUM_JOINTS` to `NUM_POS` (the value is `get_num_pos()`); add `const int NUM_LINKS = NJ` for the un-reduced count; emit `__constant__ T MIMIC_ALPHA[NUM_LINKS]` and `MIMIC_V_SLOT[NUM_LINKS]` tables. Add Python-side helpers: `self._v_slot_cpp(jid)`, `self._alpha_cpp(jid)`, `self._emit_v_accumulate(dst, jid, expr)`.
- **`helpers/_topology_helpers.py`** — `gen_load_update_XImats_helpers`: per-joint Xmat update reads `s_q[ind]` (jid-indexed). Rewrite to do the `q_for_joint`-equivalent fold at the top into a per-jid `T s_q_eff[NJ]` scratch, then existing emit stays jid-indexed.
- **`helpers/_lin_alg_helpers.py`** — line 257 `int cur = 36*((index/num)%NUM_JOINTS);` — verify `NUM_JOINTS` here means jid (it does, addresses `s_XImats`) and update to `NUM_LINKS`.

### Per-algo files

| File | Required change |
| --- | --- |
| `algorithms/_inverse_dynamics.py` | `gen_inverse_dynamics_inner`: replace `s_qd[jid]` reads with `α * s_qd[v_slot(jid)]` (constant-folded). compute_c parallel loop must use per-jid bpass that does `s_c[v_slot(jid)] += α * S^T · f[:,jid]` (scalar fold or atomicAdd if two mimic share v-slot). |
| `algorithms/_crba.py` | H assembly: `s_M[v_i, v_j] += α_i α_j (S^T fh)` plus symmetric write. Floating-base root-cross also `α_i`-scales. Mirror RBDReference lines 2403–2424 / 2479–2506. |
| `algorithms/_direct_minv.py` | **Cannot be patched** — U/Dinv/F recursion doesn't superpose. **Fallback:** detect mimic at codegen time, emit a body that calls `crba_inner` + `glass::invertMatrix_dense` on NVxNV reduced M. |
| `algorithms/_aba.py` | **Cannot be patched** (per-body U,d divergence). Apply algebraic-decomposition at kernel level: for mimic, emit thin wrapper `aba_inner` = call `inverse_dynamics_inner` (c) + `direct_minv_inner` (Minv) + GEMV `qdd = Minv · (u − c)`. Non-mimic keeps fast path. |
| `algorithms/_forward_dynamics.py` | Calls aba; mimic case routes through fallback automatically. Confirm `s_qdd` is NV-sized. |
| `algorithms/_inverse_dynamics_gradient.py` | bpass write `s_dc_du[v_i*NV + v_j] += α_i α_j (...)`. Sparsity helpers reduce from `n*NJ` to `n*NV`. |
| `algorithms/_forward_dynamics_gradient.py` | Composes `−Minv · dc_du` (algebraic decomp). Automatic once Minv + dc_du land. Verify FB GEMM dims use NV. |
| `algorithms/_integrator.py` | Mostly transparent — verify mimic q coords reconstructed at consumer side (likely reduced too, no change). |
| `algorithms/_integrator_gradient.py` | Verify per-stage `s_stage_grad_qdd[n*stage + ind]` n is NV. |
| `algorithms/_eepose_gradient_hessian.py` | Already partially mimic-aware (uses `get_joint_index_v(j)` at 387, 1389). Hessian assembly needs `α_i α_j` scaling like RBDReference's analytic Hessian. |
| `algorithms/_idsva_so.py` | Biggest single file (~3589 LOC). Every `qd[jid]` read / `dM_dq[jid][...]` write needs `α`-scaling + `v_slot()` routing. `t_index_map` NJ×NJ already patched. |
| `algorithms/_fdsva_so.py` | Composes from idsva_so + Minv. Verify temp arena sizing `18*get_num_joints() + 2*n*n` is still aligned with vaf (NJ-sized) vs reduced outputs (NV-sized). |

### Test harness
- **`test/cuda_equivalents/test_cuda_executable_equivalence.py`** — fr3 and h1_2 already in matrix via `iter_robot_cases`. After each P-phase, remove skip entries per (robot, algo) and add equivalence assertion.

## 2. Macro / constant emission changes (C++ side)

```cpp
const int NUM_LINKS = <NJ>;             // un-reduced joint count
const int NUM_POS   = <nq>;             // reduced position DoF (was NUM_JOINTS)
const int NUM_VEL   = <nv>;             // reduced velocity DoF
const int NUM_MIMIC = <num_mimic>;
__constant__ T   MIMIC_ALPHA[NUM_LINKS]    = { 1.0, 1.0, 1.6, ... };
__constant__ int MIMIC_V_SLOT[NUM_LINKS]   = {   0,   1,   1, ... };
```
Most usages constant-fold via Python emit (so tables are informational/debug). Keep `NUM_JOINTS` as backward-compat alias for `NUM_POS`.

## 3. Python-side codegen helpers (new)

```python
def _v_slot_cpp(self, jid):
    return self.robot.get_joint_index_v(jid)

def _alpha_cpp(self, jid):
    j = self.robot.get_joint_by_id(jid)
    if getattr(j, "is_mimic", False):
        return f"static_cast<T>({j.get_mimic_multiplier()!r})"
    return None  # caller skips multiplication

def _emit_v_accumulate(self, dst, jid, expr):
    """Emit `dst[v_slot(jid)] += α * (expr);` collapsing α=1 to assign/add."""
    ...
```

## 4. Order of implementation (phased PRs)

**Phase 1 (1.0 day) — Foundation + ID + CRBA**
- Helper plumbing (_v_slot_cpp, _alpha_cpp, MIMIC_ALPHA / MIMIC_V_SLOT emit, NUM_LINKS rename).
- `gen_load_update_XImats_helpers` mimic-aware q-fold.
- ID inner (compute_c bpass per-jid + α-scaled v-slot accumulate).
- CRBA inner (H assembly with α_i α_j).
- **Validate:** fr3-fixed + h1_2-fixed CUDA-equiv for `inverse_dynamics`, `crba`. **Mergeable.**

**Phase 2 (0.5 day) — Minv-via-CRBA + FD-via-decomp**
- `direct_minv_inner` for mimic: call `crba_inner` + `glass::invertMatrix_dense` on NVxNV M.
- `aba_kernel` for mimic: call `inverse_dynamics_inner` (compute_c) + `direct_minv_inner` + GEMV `qdd = Minv·(u-c)`.
- `forward_dynamics_kernel` routes through ABA — no change.
- **Validate:** fr3+h1_2-fixed for `minv`, `aba`, `forward_dynamics`. **Mergeable.**

**Phase 3 (0.75 day) — Gradients**
- ID-du inner: α-scaled v-slot accumulate in bpass; reduce sparsity sizes.
- FD-du: automatic via `−Minv · dc_du` decomposition.
- Integrator + integrator_gradient: explicit fr3 validation.
- **Validate:** `inverse_dynamics_gradient_q/qd`, `forward_dynamics_gradient_q/qd`, `integrator*`. **Mergeable.**

**Phase 4 (1.0 day) — Kinematic + Second-order**
- `ee_pose_gradient_hessian` audit + Hessian α_i α_j folding.
- `idsva_so` body-frame inner mimic-fold (biggest emit).
- `fdsva_so` composes automatically.
- **Validate:** fr3+h1_2 fixed AND floating for `ee_pose*`, `idsva_so`, `fdsva_so`. **Mergeable.**

Total: **3.25 days** for one focused implementer; 2 implementers in parallel could compress to ~2.5 wall days.

## 5. Validation strategy

- Per-algo `test_cuda_executable_equivalence.py::test_fixed[<spec>-fixed]` already parametrized for fr3, h1_2.
- After each phase, per-algo CUDA-vs-RBDReference equivalence goes FAILED → PASSED for fr3-fixed first, then h1_2-fixed, then fr3-floating.
- Defer h1_2-floating until Phase 4 (idsva_so body-frame is on critical path).
- Tolerance: existing per-(robot, algo) overrides at `test_cuda_executable_equivalence.py:145` — reuse for any mimic-induced cancellation.

## 6. Estimated effort per phase

| Phase | Implementer hours | Validation hours |
| --- | --- | --- |
| P1 (foundation + ID + CRBA) | 6–8 | 2 |
| P2 (Minv-via-CRBA + FD-via-decomp) | 3–4 | 1 |
| P3 (gradients + integrator) | 5–7 | 2 |
| P4 (ee_pose + idsva_so + fdsva_so) | 6–9 | 3 |
| **Total** | **20–28 h** | **8 h** |

## 7. Risks and gotchas

1. **ABA divergence.** Confirmed unavoidable. RBDReference parks `aba()` and routes `forward_dynamics()` through CRBA+inv (line 2822). CUDA must do the same.
2. **Direct Minv recursion divergence.** Same reason. `M^{-1} = (G^T M_full G)^{-1} ≠ G^T M_full^{-1} G` — per-body U/Dinv can't be `+=`-patched.
3. **Multi-DoF mimic joints.** Non-existent in current URDF set (URDF spec restricts mimic to single-DoF; floating can't be mimic).
4. **MIMIC_V_SLOT collisions.** When two jids share a v-slot, parallel writes need atomicAdd or serial fold. Serial scalar fold is what RBDReference does — simplest.
5. **Sparsity helpers (`gen_topology_sparsity_helpers_python`).** `NJ * n` tables overestimate for mimic (waste, not correctness) — defer to P3.
6. **`NUM_JOINTS` macro semantic drift.** Keep backward-compat alias; document rename.
7. **`idsva_so_body_frame_inner` t_index_map NJxNJ** — defensively patched 2026-05-29 (codegen b573849); confirm in P4.

## 8. What CANNOT use algebraic-decomposition shortcut

- **`crba_inner`** — must do α_i α_j accumulation; it's the M producer.
- **`inverse_dynamics_inner` bpass** — must compute `c[v_slot] += α S^T f` directly; it's the bias kernel everything else decomposes through.
- **`idsva_so` body-frame inner** — second-order RNEA, heaviest mimic work.
- **`ee_pose_*` kinematics** — pure kinematic, mimic shows up only via Xmat update q-fold in P1.

Algorithms that CAN ride the shortcut:
- `direct_minv` → `inv(CRBA)` (mimic-only path)
- `aba` → `Minv · (u − c)` (mimic-only path)
- `forward_dynamics` → composes through aba
- `forward_dynamics_gradient` → `−Minv · rnea_grad`
- `fdsva_so` → composes through idsva_so + Minv

## Recommended pickup

Start with Phase 1 alone — most user-visible algos, complete shippable unit, sets up the helper plumbing every later phase reuses. Phase 1 implementer-agent prompt should:
- Read this doc + the RBDReference 2026-05-29 commits (`0b1a89d`, `8c351ad`, `bea0ac1`, `aa3eaa1`) to absorb the canonical patterns.
- Land helpers + ID + CRBA + ID/CRBA fr3-fixed equivalence.
- NOT touch Minv/aba/idsva_so (those are Phase 2/4).
