# Memory-aware contributor validation

Status: contributor reserve implemented November 18; constrained execution
validation remains required before a compatibility claim. Brian asked for
useful validation on a 16 GiB host with about 8 GiB available. The existing
contributor scope and all numerical gates remain unchanged.

## Separate resources from evidence

A resource profile specifies scheduling and conservative admission estimates.
An evidence tier specifies required test IDs and verification policy. Prefer
running the existing contributor smoke with a lower-memory resource profile.
If its indivisible builds cannot fit, define a separately named, smaller
diagnostic tier instead of treating missing mandatory smoke as provisional
success. CPU-only checks remain useful without any GPU receipt. None of these
tiers substitutes for primary or release evidence.

## Target profiles

- CPU-only: no GPU, compiler or signing requirement; existing CPU gates.
- Low-memory GPU: target 8 GiB effective available RAM on a 16 GiB host, one compiler unit,
  one active build or GPU test group, and no prewarm/test overlap. Admission
  uses a 2 GiB reserve plus the 1 GiB allowance for the exact measured four-ID
  smoke. Host MemAvailable and visible ancestor cgroup headroom both bound it.
  This is an enabled admission policy, not a measured hardware guarantee.
- Standard contributor: the reviewed mandatory smoke plus complete optional
  groups. The same 2 GiB contributor reserve applies; more available memory
  permits whole optional groups. The full lab lane keeps its 14 GiB reserve.
  Unmeasured optional small-native groups retain 8 GiB before a 1.3 margin;
  unknown groups retain 16 GiB before margin.
- Full validation/release: exact required scope. Memory may serialize or pause
  work, never remove tests, change tolerances or turn failures into skips.

## Implementation sequence

1. Measure each mandatory smoke build cold using private cache directories.
   Include the pytest parent, compiler descendants, and retained memory across
   tests. Do not use warm cache hits to lower cold-build floors.
2. Establish one-unit class estimates with explicit source/toolchain/options
   identity. Keep larger observed peaks and a conservative unknown fallback.
   Resource estimates never authorize reusing test outcomes or evidence.
3. Prefer separate small sessions if allocator retention makes the combined
   smoke too large. Merge authenticated results only after exact mandatory
   scope and all-passed checks. Preserve compilation groups where beneficial.
4. Select profiles from live available RAM and explicit user choice, not just
   installed capacity. CPU-only mode must not initialize CUDA. Distinguish host
   RAM from VRAM; discrete and unified-memory GPUs need different accounting.
   Effective headroom must also respect an active process/container memory
   limit; host-wide available RAM alone must not admit work beyond that limit.
5. Validate on an actual constrained host or approved memory-limited execution,
   including background memory pressure, refusal before the first job, a
   failed optional job, and clean shutdown. Synthetic accounting tests alone
   cannot establish that the builds fit.

## Failure behavior

Print available RAM, unspent reservations, predicted peak, reserve and exact
pending group when refusing admission. Preserve completed logs. Do not kill
peers, use swap as promised RAM, or launch a heavier group just because a
lighter one finished. Once a job starts, transient external pressure can still
exhaust memory: sampled admission is not an RSS cap. Keep pressure-stop and
resume behavior explicit, with no recording fingerprint changes during a run.

No 16 GiB or 8 GiB-available compatibility claim is made by this document.
Hardware campaigns, hard resource limits and changes to the required evidence
sets need their own validation before enabling the low-memory profile.

## First exact-smoke measurement

On November 15, the four mandatory IDs were run together, without signing a
receipt, on the 64 GiB / RTX 5090 development host with CUDA 13.2. Build,
driver and pytest directories were new and private; no shared cache was purged.
The run used one compiler unit. All four tests passed with no skips or errors.

Across 319 samples at approximately 100 ms intervals, the largest observed
pytest-plus-descendant RSS was 0.374 GiB (rounded upward). The admission
sampler separately observed 0.371 GiB; minimum host MemAvailable was 45.18 GiB.
This is sampled process-tree RSS, not a hard peak bound, total system memory
cost or VRAM measurement. OS file caches were not made cold. Other toolchains,
GPU drivers and optional groups remain unmeasured by this four-test experiment.

Evidence: `docs/open-tasks/codex_contributor_cold_smoke_20261115/` in the root
checkout, including exact IDs, command/toolchain identity, JUnit results,
private build manifests and resource samples. Recorded source: `2d2ccdf7`.
Brian's November 15 measurement-based ruling now sets the exact smoke's
allowance to 1 GiB: twice 0.39 GiB (a conservative rounding of the observed
peak), plus 0.22 GiB absolute headroom. No extra small-machine experiment is
needed to establish that process estimate. Installed RAM does not multiply a
process's footprint. On November 18 the contributor reserve was reduced to
2 GiB, with effective cgroup-aware admission; the full lab reserve stays 14 GiB.
The handoff still requires constrained execution before claiming 16 GiB
machine compatibility. A separately approved Jetson experiment would assess
a different compiler/platform, not establish the desktop process's footprint.
Optional groups and heavy/unknown classes retain conservative estimates; the
four-test measurement is not applied to the whole suite. Existing full-driver
identity-qualified measured history and larger class floors remain unchanged.
