# Extended-session execution plan (2026-05-31, flight delayed to ~9pm)

Authoritative state + plan for the post-compaction continuation. Merged stack tip when
written: **`018d38a`** on `modernizing-tests` (NOT pushed). Tree GREEN.

## What's DONE tonight (closed)
A1, A2, B1 (floating+mimic id_du/fd_du), B2-ee (fixed-base mimic ee), C1 (f_ext −∂Jᵀ/∂q
fixed AND floating), C2 (centroidal ∂h/∂q), C3 (floating Coriolis), D1 (all cuda tests),
E1 (regressor) + FD param-gradient ∂q̈/∂π, E4 (URDFParser typed-exc + continuous + planar/
spherical PARSER), E5/E6 (notebooks + torch f_ext), mimic-Y validated. PERF: crba
dead-buffer −3.3×, idsva_so body output_tp de-alias, **d2ee parallelized (Step2 per-slot +
Step5b per-cell, bit-exact incl mimic — 7a587d4)**, **ABA floating sibling-fusion (018d38a)**,
shared-helper dedup (minv-apply, BFS-index-decode). Floating codegen regression fixed; xdist
test-infra fixed; docs consolidated (HANDOFF 1781→1465, archive/, so_audit_plan.md, G4 flag).

## User decisions (2026-05-31): full SO audit NOW + complete/harden + E2 if feasible-in-parallel; defer E3.

## RUNNING at compaction
- **K-iddu** (`_inverse_dynamics_gradient.py`): id_du WIN-A (branched-fixed dc/du 2NJ→2n²) + WIN-B (mimic single-thread passes → column-parallel). Validate go2-fixed/fr3/h1_2 + Gate-A untouched paths.
- **SO-idsva** (`_idsva_so.py`): so_audit A1 (world-frame 6-vec parallelization, sweep-relevant) + B1 (Xdown dedup) + B2 (reference-order dedup). bit-exact / byte-identical.

## TO LAUNCH (file-isolated; clone each off main, branch, reuse main .venv, clear header cache before validation, NO push, short commits no-Claude-footer, MANDATORY floating+fixed codegen smoke + clean-cache equivalence):
1. **SO-fdsva** (`_fdsva_so.py`): B3 — dedup the duplicated timing/non-timing kernel-body emitters (~L402-456, byte-identical). PLUS a profiling-PLAN note for the Minv-apply hotspot (do NOT blindly change it — needs in-context Nsight + transposed layout; it was twice-rejected; see so_audit_plan A2).
2. **floating-mimic-ee** (`_eepose_gradient_hessian.py`, free post-P-d2ee): B2-ee FLOATING — un-refuse floating+mimic ee_pose_gradient/hessian. The floating root needs a 6-DoF subspace fold (mirror how K-b1 did B1 floating+mimic id_du: free-flyer root S = [[0,I],[I,0]] block-swap, per-root-DoF loop). Validate fr3-floating + h1_2-floating; Gate-A non-mimic byte-identical.
3. **g1-spill** (`_f_ext_gradient.py` + `_regressor.py` + their GCG slices): smem-cap workspace/spill kernel variant so g1-floating runs `f_ext_grad_dq` AND `fd_parameter_gradient` (both currently skip: 35-DoF buffers exceed 101KB/block). Mirror the tier-spill (d_workspace) pattern. Validate g1-floating on-device.
4. **E2** (new `_frame_jacobian.py` or similar + RBDReference numpy ref + GCG additive): general-frame Jacobians J(LOCAL/WORLD/LOCAL_WORLD_ALIGNED) + frame J̇, then OSC operational-space inertia Λ=(J M⁻¹ Jᵀ)⁻¹ (M⁻¹Jᵀ already lands via f_ext_grad). Numpy ref vs pinocchio getFrameJacobian/computeJointJacobiansTimeVariation; CUDA equivalence test. iiwa14 fixed + a floating robot. Defer E3 (contact/constraint) — too large for tonight.

## QUEUED (fire as owners free files)
- **B2-SO mimic** (idsva_so/fdsva_so NB>NV internal-sweep VALUE bug; J-idsva built+reverted, resume hint in `docs/idsva_so_inner_refactor_notes.md`) — AFTER SO-idsva + SO-fdsva land (same files). Hard device debug.
- **E4 CUDA multi-column-S** (planar/spherical emit; needs multi-column S + NV≠NQ codegen).
- **F1** RBDReference file-split (`rbdreference_split_plan.md`) — after E2/floating-Coriolis-area RBDReference quiet.
- **F2** warnings sweep (`mxS` NumPy ndim>0 + nvcc/ptxas) — tree-quiet pass.
- **F3** autotune best-tier defaults; LITE-aliases-SHARED launch_bounds.

## MERGE discipline (hardened-checklist, learned this session)
Per-submodule fetch+merge; AST-dup-check tolerances.py after RBDReference merges; resolve
GCG.py keep-both for additive collisions; **byte-identical diff** for refactors (capture
baseline grid.cuh BEFORE, regen AFTER); **floating+fixed codegen smoke** after any
URDFParser/Robot.py or shared-GCG merge (NOT just py_compile — a floating regression slipped
past py_compile); clear generated-header cache before equivalence (stale = phantom failures);
discard agents' HANDOFF.md edits (main owns it). Agents that end mid-turn without a report
(d2ee class) → inspect clone, commit, validate-yourself before merging.

## CONVERGENCE
After the wave merges + queue drains: run the **pre-sweep consolidated green-gate** —
`bash test/cuda_equivalents/run_parallel.sh -k "iiwa14 or go2 or g1 or h1_2"` (deterministic
thread counts now; fresh-compile the integrated tree incl. parallelized d2ee + fused ABA +
new algos). Re-gate any red surgically. Then **launch the 12h SWEEP ~9pm** (C.7 recipe:
`run_multi_version.py --robots iiwa14 go2 g1 h1_2 --bases fixed floating --columns glass
--tiers perf lite minimal --autotune-threads`) vs the C.7 baseline. NO other GPU/CPU during
the sweep. Push to origin only on explicit approval.
