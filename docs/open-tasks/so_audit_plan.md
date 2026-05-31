# Second-Order (SO) audit — dedicated pass

**Status:** queued (user-requested 2026-05-31). A focused audit of the second-order
algorithms (`idsva_so` body+world, `fdsva_so`) covering parallelism, dedup, the kept
floating-reference fallback, and the unfinished mimic-SO path. Notes below are from the
2026-05-31 parallelism-audit explore pass (read-only analysts) + the J-idsva/K-idsva work.

Files: `GRiDCodeGenerator/algorithms/_idsva_so.py` (~3.6k LOC, body + world + floating-ref),
`GRiDCodeGenerator/algorithms/_fdsva_so.py`, `docs/idsva_so_inner_refactor_notes.md`.

---

## Current state (what's already good)

- **Fixed-base body-frame inner** (`gen_idsva_so_body_frame_inner`, ~1338-2329) is FULLY
  block-parallel at tensor-element granularity — no `threadIdx==0` guards, no serial
  joint loops gating compute. The recursion sweeps are parallelized per-BFS-level
  (siblings concurrent). **This is the reference for "what good looks like."**
- **World-frame inner** (`gen_idsva_so_world_frame_inner`, ~2628-3263, the floating-base
  production path) is mostly fanned — the dominant triple-ancestor walk + IC/BC 36-element
  builds are block-parallel; per-`vel_k` output columns are disjoint/parallel.
- **fdsva_so** is already element-parallel everywhere (4n³ independent (i,j,k) cells, one
  thread each). The orchestration compose is a genuine required data-dependency chain.
- Spill/occupancy: world-frame cold-trio de-alias landed; **body-frame `output_tp` rung**
  landed (K-idsva, 2026-05-31) — g1-fixed body SO ~54KB hot-in-smem vs 8.7KB whole-arena.

## A. Parallelism opportunities (harvestable)

- **A1. World-frame forward-sweep thread-0 6-vector chains** — `_idsva_so.py` ~2907-2942
  (v/a/vJ/aJ/psid/psidd/Sd per body on thread 0), ~2963-2965 (IC_v), ~2981-2987 (f), and
  the Xup forward thread-0 36-matmul ~2787-2841. The outer `for jid` is REQUIRED recursion,
  but the per-velocity psid/psidd/Sd loops are independent across a body's velocity columns
  and the 36-element matmuls are element-independent. **Low-to-moderate yield** (floating-base
  hot path; 6-element vectors, NB× per sweep; needs a sync + scratch already exists). Risk:
  medium (forward sweep is correctness-critical; race-prone if a body's velocities alias).
- **A2. fdsva_so Minv-apply contraction hotspot** — `_fdsva_so.py` ~135-141 (`iL,Ljk->ijk`,
  ~6M of ~9M FMAs on g1_floating). Already 4n³-way parallel at the OUTER element level; the
  serial part is the per-thread length-n L-reduction. **HIGH-impact but HIGH-risk and ALREADY
  TWICE-REJECTED** (in-file comment ~94-134): cuBLASDx-gemm (2.4-5.2× standalone but doesn't
  translate in-kernel — register pressure, non-gemm strides, spill pressure, per-timestep
  sync) and `grid_linalg_dot_strided_coalesced` (the contracted L axis maps to the slowest
  stride-n² index → worse loads, and collapses the 4n³ thread parallelism). **Prerequisite:
  in-context Nsight profile (confirm compute- vs sync/bandwidth-bound) + a TRANSPOSED inner_*
  layout** before any retry. Do not touch blind.

## B. Dedup / DRY (low-risk, byte-identical)

- **B1. Xdown Plücker-block magic-number inverse, 3×** — `_idsva_so.py` ~1632 (fixed body),
  ~1106 (floating-ref), ~2847 (world-frame Step 2). The 1632/1106 copies are byte-identical.
  *(IN PROGRESS — P-dedup agent extracting a shared helper, 2026-05-31; if it lands this is done.)*
- **B2. Reference-order tensor-assembly body, 2×** — block-parallel for branched fixed-base
  (`gen_idsva_so_body_frame_reference_order_output_repair`, ~895-956) vs single-thread in the
  floating-ref (~1245-1321): same `rt1..rt9`/`rp1..rp6` outer products + `d2tau_*`/`dM_dq`
  writes. Share an emitter (only relevant if the floating-ref path is kept — see D).
- **B3. fdsva_so timing/non-timing kernel-body emitters** — `_fdsva_so.py` ~402-431 vs
  ~432-456: near-verbatim workspace-pointer setup + device-call, differ only by the offset
  prefix. Factor into one helper (~20 lines). Pure hygiene.

## C. Mimic SO (correctness completeness)

- **C1. idsva_so/fdsva_so mimic (B2-SO)** — still REFUSED for mimic robots. J-idsva BUILT the
  fixed-base mimic SO path (internal NUM_BODIES-coordinate sweep into a 4·NB³ buffer + alpha-R
  fold to reduced 4·NV³; the fold + Gate-A were proven correct) but the **CUDA internal NB³
  sweep produces WRONG values for NB>NV (mimic) robots** — `dM_dq`/`d2tau_dq2` garbage on fr3,
  while the IDENTICAL machinery is GREEN for the non-mimic branched robot `fetch` (NB=NV=14).
  Localized but not root-caused; J-idsva REVERTED rather than land a wrong path. Precise resume
  hint in `docs/idsva_so_inner_refactor_notes.md`. Floating+mimic SO additionally needs a root-
  DoF loop (not the scalar fold) — see the B1 (id_du) floating-mimic pattern for the template.

## D. Decision: the floating body-frame "reference" fallback (~700 lines)

`gen_idsva_so_body_frame_floating_reference_inner` (~`_idsva_so.py:959-1336`) + its gravity-shim
family (`gen_floating_gravity_d2tau_dq_lie_inline` ~250-747 + the metadata/count helpers).
- **Confirmed 2026-05-31:** NOT emitted in production (the dispatcher routes ALL floating-base
  SO to `world_frame`; the body-frame inner's floating branch at ~1363 is unreachable since
  floating robots never call the body-frame inner) and NOT a live test oracle (only a comment
  in `test_cuda_idsva_so_world_frame.py` mentions it).
- **It is almost entirely single-threaded** (one giant `if(threadIdx==0)` block ~1064-1323 +
  the gravity shim ~285-746) — the largest serial surface in the file, but zero production impact.
- **Decision (user, 2026-05-31): KEEP for now** — it's a deliberately-retained fallback
  ("world-frame co-exists with" it, ~`_idsva_so.py:2606"). The SO audit should DECIDE its fate:
  retire it (if world-frame is confirmed to fully cover floating SO incl. as the analytic
  cross-check) OR keep + document the contract. If retired, it removes ~700 lines + the only
  large serial blocks + the B2 duplication + several now-dead floating-velocity metadata outputs.
  Flagged inline at the def.

## Suggested audit ordering

1. Land B1/B3 dedup (low-risk; B1 may already be done by P-dedup).
2. Resolve D (keep-vs-retire the floating-ref fallback) — biggest cleanup lever; unblocks B2.
3. A1 world-frame 6-vec parallelization (floating-base hot path, moderate yield).
4. C1 mimic-SO root-cause (the NB>NV internal-sweep value bug) — correctness completeness.
5. A2 fdsva_so hotspot ONLY after an in-context Nsight profile + transposed-layout prototype.

Validate everything with the per-tier CUDA equivalence gate (clear the generated-header cache
first — stale headers give phantom failures) + Gate-A byte-identical for untouched paths.
