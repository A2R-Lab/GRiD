# Tier-baseline report v2 (P1)

Each row: one kernel × one robot config. **R** = registers/thread,
**sp** = local-memory spill stores (bytes), **smem** = per-block
dynamic shared memory bytes (computed at codegen via the
`*_DYNAMIC_SHARED_MEM_BYTES<float>()` constexpr; ptxas's static-smem
report is unhelpful here because GRiD uses `extern __shared__`).

**perf** = `__launch_bounds__(MAX_PERF_LEVEL_THREADS)` (the default);
**relax** = `__launch_bounds__(1024)` (what TIER_MINIMAL looks like).

**Decision predicate**:
- `YES (spill)`: relax.sp ≥ 500 AND relax.sp > 2 × max(perf.sp, 100).
  Real perf cliff at relaxed bounds.
- `YES (smem)`: smem ≥ 80 KB (~80% of the sm_120 100 KB per-block cap).
  Outer-kernel inline users hit smem pressure even at perf — they want
  a smem-axis downgrade variant.
- `no`: free-alias TIER_LITE/MINIMAL to TIER_PERF; no body changes needed.

**Per-robot `MAX_PERF_LEVEL_THREADS`**: iiwa14_fixed=352, go2_fixed=288, g1_fixed=512, g1_floating=512, h1_2_fixed=512, h1_2_floating=512

| Robot | Kernel | perf R/sp | relax R/sp | smem | downgrade? |
|---|---|---|---|---:|---|
| iiwa14_fixed | ID | R=73,sp=0 | R=64,sp=0 | 2800 | no |
| iiwa14_fixed | Minv | R=80,sp=0 | R=64,sp=16 | 4912 | no |
| iiwa14_fixed | FD | R=80,sp=64 | R=64,sp=56 | 5696 | no |
| iiwa14_fixed | ABA | R=80,sp=0 | R=64,sp=0 | 6384 | no |
| iiwa14_fixed | CRBA | R=80,sp=0 | R=64,sp=0 | 6192 | no |
| iiwa14_fixed | EE_POSE | R=56,sp=0 | R=64,sp=0 | 640 | no |
| iiwa14_fixed | EE_POSE_GRAD | R=74,sp=0 | R=64,sp=0 | 2896 | no |
| iiwa14_fixed | EE_POSE_HESS | R=77,sp=0 | R=64,sp=0 | 10016 | no |
| iiwa14_fixed | ID_DU | R=167,sp=0 | R=64,sp=48 | 9888 | no |
| iiwa14_fixed | FD_DU | R=168,sp=0 | R=64,sp=232 | 10112 | no |
| iiwa14_fixed | IDSVA_SO_B | R=168,sp=4 | R=64,sp=44 | 22576 | no |
| iiwa14_fixed | IDSVA_SO_W | R=158,sp=0 | R=64,sp=336 | 22576 | no |
| iiwa14_fixed | FDSVA_SO | R=168,sp=24 | R=64,sp=500 | 28704 | YES (spill) |
| go2_fixed | ID | R=72,sp=8 | R=64,sp=8 | 5104 | no |
| go2_fixed | Minv | R=168,sp=8 | R=64,sp=16 | 11056 | no |
| go2_fixed | FD | R=168,sp=12 | R=64,sp=60 | 12400 | no |
| go2_fixed | ABA | R=168,sp=16 | R=64,sp=8 | 11248 | no |
| go2_fixed | CRBA | R=72,sp=0 | R=64,sp=0 | 11152 | no |
| go2_fixed | EE_POSE | R=56,sp=20 | R=64,sp=0 | 1728 | no |
| go2_fixed | EE_POSE_GRAD | R=93,sp=0 | R=64,sp=0 | 15328 | no |
| go2_fixed | EE_POSE_HESS | R=92,sp=0 | R=64,sp=0 | 98016 | YES (smem) |
| go2_fixed | ID_DU | R=72,sp=44 | R=64,sp=8 | 13120 | no |
| go2_fixed | FD_DU | R=168,sp=4 | R=64,sp=76 | 13744 | no |
| go2_fixed | IDSVA_SO_B | R=168,sp=12 | R=64,sp=104 | 53744 | no |
| go2_fixed | IDSVA_SO_W | R=96,sp=324 | R=64,sp=312 | 53744 | no |
| go2_fixed | FDSVA_SO | R=96,sp=336 | R=64,sp=400 | 88672 | YES (smem) |
| g1_fixed | ID | R=128,sp=28 | R=64,sp=36 | 12304 | no |
| g1_fixed | Minv | R=64,sp=848 | R=64,sp=772 | 38864 | no |
| g1_fixed | FD | R=64,sp=916 | R=64,sp=744 | 42112 | no |
| g1_fixed | ABA | R=128,sp=12 | R=64,sp=148 | 27152 | no |
| g1_fixed | CRBA | R=128,sp=24 | R=64,sp=24 | 28896 | no |
| g1_fixed | EE_POSE | R=128,sp=48 | R=64,sp=60 | 3280 | no |
| g1_fixed | EE_POSE_GRAD | R=128,sp=36 | R=64,sp=44 | 37008 | no |
| g1_fixed | EE_POSE_HESS | R=128,sp=4 | R=64,sp=28 | 21744 | no |
| g1_fixed | ID_DU | R=128,sp=32 | R=64,sp=20 | 52512 | no |
| g1_fixed | FD_DU | R=128,sp=24 | R=64,sp=840 | 56000 | YES (spill) |
| g1_fixed | IDSVA_SO_B | R=128,sp=544 | R=64,sp=1120 | 75696 | YES (spill) |
| g1_fixed | IDSVA_SO_W | R=128,sp=40 | R=64,sp=752 | 75696 | YES (spill) |
| g1_fixed | FDSVA_SO | R=128,sp=1324 | R=64,sp=2516 | 86016 | YES (smem) |
| g1_floating | ID | R=128,sp=36 | R=64,sp=52 | 13552 | no |
| g1_floating | Minv | R=64,sp=1200 | R=64,sp=1232 | 51536 | no |
| g1_floating | FD | R=64,sp=1280 | R=64,sp=1288 | 55552 | no |
| g1_floating | ABA | R=128,sp=36 | R=64,sp=108 | 56992 | no |
| g1_floating | CRBA | R=128,sp=32 | R=64,sp=60 | 19328 | no |
| g1_floating | EE_POSE | R=128,sp=0 | R=64,sp=0 | 3552 | no |
| g1_floating | EE_POSE_GRAD | R=128,sp=0 | R=64,sp=0 | 45568 | no |
| g1_floating | EE_POSE_HESS | R=64,sp=16 | R=64,sp=8 | 24192 | no |
| g1_floating | ID_DU | R=128,sp=36 | R=64,sp=88 | 81904 | no |
| g1_floating | FD_DU | R=128,sp=8 | R=64,sp=1172 | 86944 | YES (spill+smem) |
| g1_floating | IDSVA_SO_B | R=128,sp=268 | R=64,sp=992 | 100944 | YES (spill+smem) |
| g1_floating | IDSVA_SO_W | R=128,sp=40 | R=64,sp=744 | 34704 | YES (spill) |
| g1_floating | FDSVA_SO | R=128,sp=664 | R=64,sp=2036 | 96512 | YES (spill+smem) |
| h1_2_fixed | ID | R=64,sp=180 | R=64,sp=20 | 21632 | no |
| h1_2_fixed | Minv | R=64,sp=980 | R=64,sp=1004 | 100608 | YES (smem) |
| h1_2_fixed | FD | R=64,sp=1244 | R=64,sp=1372 | 106320 | YES (smem) |
| h1_2_fixed | ABA | R=64,sp=360 | R=64,sp=300 | 47744 | no |
| h1_2_fixed | CRBA | R=64,sp=172 | R=64,sp=28 | 55296 | no |
| h1_2_fixed | EE_POSE | R=128,sp=48 | R=64,sp=48 | 6528 | no |
| h1_2_fixed | EE_POSE_GRAD | R=128,sp=48 | R=64,sp=48 | 179328 | YES (smem) |
| h1_2_fixed | EE_POSE_HESS | R=128,sp=16 | R=64,sp=40 | 91104 | YES (smem) |
| h1_2_fixed | ID_DU | R=64,sp=40 | R=64,sp=20 | 71232 | no |
| h1_2_fixed | FD_DU | R=64,sp=1696 | R=64,sp=1696 | 51616 | no |
| h1_2_fixed | IDSVA_SO_B | R=128,sp=772 | R=64,sp=1332 | 146368 | YES (smem) |
| h1_2_fixed | IDSVA_SO_W | R=128,sp=44 | R=64,sp=756 | 146368 | YES (spill+smem) |
| h1_2_fixed | FDSVA_SO | R=128,sp=1704 | R=64,sp=2928 | 178000 | YES (smem) |
| h1_2_floating | ID | R=128,sp=0 | R=64,sp=48 | 22880 | no |
| h1_2_floating | Minv | R=64,sp=1372 | R=64,sp=1372 | 120848 | YES (smem) |
| h1_2_floating | FD | R=64,sp=1344 | R=64,sp=1376 | 127328 | YES (smem) |
| h1_2_floating | ABA | R=128,sp=60 | R=64,sp=128 | 129824 | YES (smem) |
| h1_2_floating | CRBA | R=128,sp=0 | R=64,sp=72 | 37632 | no |
| h1_2_floating | EE_POSE | R=128,sp=0 | R=64,sp=0 | 6784 | no |
| h1_2_floating | EE_POSE_GRAD | R=114,sp=0 | R=64,sp=0 | 203552 | YES (smem) |
| h1_2_floating | EE_POSE_HESS | R=64,sp=8 | R=64,sp=8 | 99296 | YES (smem) |
| h1_2_floating | ID_DU | R=128,sp=36 | R=64,sp=148 | 47168 | no |
| h1_2_floating | FD_DU | R=128,sp=100 | R=64,sp=2396 | 60384 | YES (spill) |
| h1_2_floating | IDSVA_SO_B | R=128,sp=264 | R=64,sp=1036 | 168784 | YES (spill+smem) |
| h1_2_floating | IDSVA_SO_W | R=128,sp=44 | R=64,sp=776 | 58208 | YES (spill) |
| h1_2_floating | FDSVA_SO | R=128,sp=1108 | R=64,sp=2528 | 243568 | YES (spill+smem) |

## Cells needing real downgrade

**YES (spill)** (7):
  - iiwa14_fixed/FDSVA_SO
  - g1_fixed/FD_DU
  - g1_fixed/IDSVA_SO_B
  - g1_fixed/IDSVA_SO_W
  - g1_floating/IDSVA_SO_W
  - h1_2_floating/FD_DU
  - h1_2_floating/IDSVA_SO_W

**YES (smem)** (14):
  - go2_fixed/EE_POSE_HESS
  - go2_fixed/FDSVA_SO
  - g1_fixed/FDSVA_SO
  - h1_2_fixed/Minv
  - h1_2_fixed/FD
  - h1_2_fixed/EE_POSE_GRAD
  - h1_2_fixed/EE_POSE_HESS
  - h1_2_fixed/IDSVA_SO_B
  - h1_2_fixed/FDSVA_SO
  - h1_2_floating/Minv
  - h1_2_floating/FD
  - h1_2_floating/ABA
  - h1_2_floating/EE_POSE_GRAD
  - h1_2_floating/EE_POSE_HESS

**YES (spill+smem)** (6):
  - g1_floating/FD_DU
  - g1_floating/IDSVA_SO_B
  - g1_floating/FDSVA_SO
  - h1_2_fixed/IDSVA_SO_W
  - h1_2_floating/IDSVA_SO_B
  - h1_2_floating/FDSVA_SO

