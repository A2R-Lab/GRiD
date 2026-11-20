# Report-only follow-up package

Baseline under review: `91388bef`. This package is independent follow-up work,
not a revision of that review baseline. It does not change generators, bindings,
peers, cache behavior, receipt policy, manifests or numerical tolerances.

## Dependency experiment: implemented boundary

`tools/dependency_report.py` implements a versioned graph, explicit ID export
roots, separate proposed object/evidence keys, edge provenance, and a JSON
snapshot/comparison CLI. It executes no candidate code or compiler. It is not
imported by production code or the receipt driver. Its output always says
`reuse_authorized: false` and `narrowing_ready: false`.

The object roots are `inverse_dynamics_inner` and `inverse_dynamics_inner_vaf`.
They conservatively depend on all tracked generator, binding, model and peer
inputs, plus caller-declared model/build configuration. Unknown paths enter the
same broad fallback. Evidence roots add the complete tracked test/install input
set. This is deliberately not a narrowed C++ closure or a usable cache key.
In particular, a change to another algorithm still invalidates ID in this
prototype. Even RBDReference gitlinks remain broad rather than claiming a proven
compile/oracle separation for all consumers.

The graph hashes edges as well as values. Comparison recomputes keys from nodes,
does not trust the snapshot's cached key fields, and rejects dangling edges or
incompatible schema. Missing/deleted unstaged inputs, untracked non-doc sources
and symlinks are unresolved, not silently absent. Documentation under `docs/`
(except its executable test file) is explicitly excluded. Unknown paths are
not excluded merely because they look irrelevant.

`complete` means the requested tracked file inputs were present. It does NOT
mean the dependency graph is proven complete. Gitlinks record committed peer
identities, not peer working-tree bytes. Compiler, architecture, flags, ABI and
generation context are caller declarations, not verified measurements. These
limitations independently prohibit using the keys to authorize reuse.

CPU mutation fixtures cover leaf/shared/unrelated generator files, models,
wrappers, build settings, oracle/fixture/tolerance/policy/test-list changes,
unknown files, missing inputs, graph cycles, dangling edges, symlinks and
malformed context. They test the graph mechanism, not the correctness of an
inferred generated-code closure. Whole-header regeneration comparisons and
compiler dependency validation are still future work.

Commands (stdout only; choose private output files):

```sh
.venv/bin/python tools/dependency_report.py snapshot --root /path/to/checkout --context /path/to/context.json
.venv/bin/python tools/dependency_report.py compare before.json after.json
.venv/bin/python tools/id_scope_audit.py --root /path/to/checkout
```

Context requires exactly `architecture`, `compiler`, `host_compiler`, `flags`,
`abi`, `generation_options`, and `model`. Use explicit `UNMEASURED` declarations
for a pure CPU inventory; never present them as an actual build identity.

The clean review-baseline inventory produced 680 graph nodes, no missing
tracked inputs, and 63 unclassified paths retained conservatively. These are
inventory counts, not a coverage percentage or a speedup estimate.

## ID cooperation audit: concrete blockers

`tools/id_scope_audit.py` follows direct `self` calls to unique module-level
Python emitters. On the baseline it visits 42 emitters and reports 17 unresolved
method/dynamic-dispatch items. Its textual findings include 39 barrier sites,
18 leader sites, eight stride sites, and three shared-storage sites. These are
source findings, including branch alternatives—not runtime counts or a proof
of generated C++ reachability. Imports, generated call strings and class-method
registration require manual review. An empty audit is never a safety certificate.

Manual source inspection establishes these specific issues:

| Area | Existing assumption | Prototype treatment |
|---|---|---|
| `_code_generation_helpers.py`, `gen_add_sync` | Emits `__syncthreads()` | Separate private scope-aware synchronization; leave block emitter unchanged |
| `gen_add_parallel_loop` / `gen_add_serial_ops` | Whole-block rank/stride and block thread zero | Warp lane/rank and warp leader, not block thread zero |
| `_inverse_dynamics.py`, polynomial follower tables | Function-local mutable `__shared__` alpha/alpha-prime arrays | Per-problem scratch; different warps must never share these arrays |
| `_lin_alg_helpers.py`, `grid_linalg_gemv` | Calls block GLASS GEMV and adds a block barrier | Private scope-specific adapter, including synchronization audit |
| ID backward pass | Sequential child-to-parent force additions, including repeated parents | Preserve deterministic accumulation order; do not parallelize siblings blindly |
| ID mimic torque extraction | Serial fold into shared velocity slots | Warp-leader fold and explicit handoff; preserve projection semantics |
| Device wrapper / XImats arena | Block-owned shared arena with optional caller arena | Review all mutable slices and helper lifetimes; a caller pointer alone does not change cooperation scope |

The polynomial tables are a concrete reason a GLASS namespace swap cannot be
the implementation: even with warp linear algebra, two warps would alias those
mutable tables. Conversely, existing caller-arena plumbing is useful groundwork;
there is no need to invent a second arena system before auditing it.

## Next implementation packages, separated for review

1. **Dependency detector, still report-only.** Add explicit generated-entry
   metadata and capture dependency explanations for one fixed-base ID model.
   Cross-check deliberate emitter/helper/model mutations against whole-header
   regeneration on CPU. Keep broad fallbacks for macros/templates/layouts.
   Do not yet narrow cache keys or receipt carry. Exit: expected/actual affected
   set for each mutation, unresolved-edge list, no missed dependency in the
   declared mutation set. Only then propose an object-cache integration.
2. **Private warp ID prototype, after scope approval.** Start with private
   rank/leader/sync and GEMV adapters, explicit scratch, fixed-base chain ID,
   external force and qdd/no-qdd variants. No changes to existing block emitters
   or public registration. CPU generated-source checks precede an approved GPU
   correctness window. Two independent warps per block, non-overlapping canary
   scratch, repeated-parent tree, floating and mimic/coupled sentinels are
   acceptance requirements before recommending an API. Thread ID follows warp;
   gradients/Minv/FD need their own scratch and feature inventory.
3. **Remaining numerical attribution, CPU only.** Reuse the saved iiwa14
   high-velocity and g1 random-1 captures, with passing control cases. Compare
   per-entry d²tau/dq² error and its propagated output contribution, not just
   tensor norm. Establish float64 algebraic identity before emulating candidate
   float32 local/body-frame accumulation. Test accumulation order and cancellation
   hypotheses without altering input samples, tolerances or product code. Keep
   first-order/bias oracle substitutions visible. Widen to the 64 saved cases
   only after a candidate survives the initial two cases. A CPU pass remains
   insufficient for a device fix or support-envelope claim.

No GPU validation, timing, numerical kernel, mixed precision, scratch ABI,
external write or publication is authorized by this package. The work under
Claude's review is unchanged and can be reviewed independently.
