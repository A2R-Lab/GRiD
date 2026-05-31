# Archived plans — LANDED

Completed implementer plans, kept for reference (history/rationale). These shipped on
the 2026-05-31 H/I/J/K agent campaign and are no longer active backlog:

- `d3_pytorch_cudagraphs_plan.md` — D.3 PyTorch backend + autograd + CUDA-Graphs (LANDED; `grid_rbd` torch surface).
- `notebook_examples_plan.md` — E5 example notebooks (LANDED; `notebooks/01_quickstart`, `02_autograd_torch`, `03_plant_control`).
- `centroidal_quickwins_plan.md` — R1/R2/R3 energy/g(q)/Coriolis + CCRBA + CoM + C2 derivatives (LANDED; floating Coriolis = open C3).
- `kinematics_warp_thread_plan.md` — batched FK `ee_pose_inner_{thread,warp}` (LANDED + committed test).
- `bc_cleanup_plan.md` — B+C device-wrapper / tier-dispatch consolidation (LANDED, byte-identical).

Active backlog stays in `docs/open-tasks/`; the canonical tracker is `HANDOFF.md` (OPEN ITEMS).
