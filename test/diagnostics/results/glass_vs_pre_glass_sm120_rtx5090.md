# ptxas -v diagnostic

Per-kernel register count (R), shared mem (S, bytes), spill stores (sp). Lower R is usually better for occupancy.

| Robot | Kernel | HEAD_glass | pre_glass | Δ regs | Δ smem | Δ spill |
|---|---|---|---|---|---|---|
| go2 | FD | R=168,S=0,sp=12 | R=128,S=0,sp=0 | +40 | +0 | +12 |
| go2 | ABA | R=168,S=0,sp=16 | R=136,S=0,sp=0 | +32 | +0 | +16 |
| go2 | EE_POSE_GRAD | R=93,S=0,sp=0 | R=56,S=0,sp=0 | +37 | +0 | +0 |
| go2 | ID | R=72,S=0,sp=8 | R=136,S=0,sp=0 | -64 | +0 | +8 |
| go2 | CRBA | R=72,S=0,sp=0 | R=72,S=0,sp=0 | +0 | +0 | +0 |
| go2 | Minv | R=168,S=0,sp=8 | R=134,S=0,sp=0 | +34 | +0 | +8 |

| g1 | FD | R=64,S=0,sp=916 | R=255,S=0,sp=20 | -191 | +0 | +896 |
| g1 | ABA | R=128,S=0,sp=12 | R=255,S=0,sp=16 | -127 | +0 | -4 |
| g1 | EE_POSE_GRAD | R=128,S=0,sp=36 | R=74,S=0,sp=0 | +54 | +0 | +36 |
| g1 | ID | R=128,S=0,sp=28 | R=144,S=0,sp=0 | -16 | +0 | +28 |
| g1 | CRBA | R=128,S=0,sp=24 | R=130,S=0,sp=0 | -2 | +0 | +24 |
| g1 | Minv | R=64,S=0,sp=848 | R=255,S=0,sp=8 | -191 | +0 | +840 |

| iiwa14 | FD | R=80,S=0,sp=64 | R=138,S=0,sp=0 | -58 | +0 | +64 |
| iiwa14 | ABA | R=80,S=0,sp=0 | R=62,S=0,sp=0 | +18 | +0 | +0 |
| iiwa14 | EE_POSE_GRAD | R=74,S=0,sp=0 | R=56,S=0,sp=0 | +18 | +0 | +0 |
| iiwa14 | ID | R=73,S=0,sp=0 | R=64,S=0,sp=0 | +9 | +0 | +0 |
| iiwa14 | CRBA | R=80,S=0,sp=0 | R=64,S=0,sp=0 | +16 | +0 | +0 |
| iiwa14 | Minv | R=80,S=0,sp=0 | R=122,S=0,sp=0 | -42 | +0 | +0 |
