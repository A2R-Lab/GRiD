# Next phase after the overnight correctness work

Status: local implementation and experiments, not promoted release content.
The November 14 reviewed branches remain separate and unchanged. This follow-up
is `codex/overnight-next-phase-20261115`, based on `fd59c64a`.

## 1. Finish the infrastructure review, then one promotion refresh

The hardware receipt tiers, smaller validation units and contributor lane are
implemented on the reviewed branch stack. This follow-up addresses the review's
RAM admission, shard-name explanation, smaller-plan fallback visibility and
historical-policy documentation findings. The 30-minute partition target remains
provisional: the extra-session overhead is not measured until the refresh.

The contributor's exact measured four-test smoke now needs 15 GiB available
(14 GiB reserve plus 1 GiB allowance: twice 0.39 plus 0.22 GiB headroom).
Unmeasured optional small-native groups still need 24.4 GiB; unknown groups
retain 34.8 GiB. This is an approximate admission estimate, not a universal
memory bound. See the measurement and low-memory plan for platform limits.

Claude ran the adversarial exercise in a private scratch repository using
same-repository branches, not forks. No candidate code executed; deletion and
signature-tampering cases failed correctly. It exposed a verifier import defect,
now locally repaired and covered by isolated no-submodule tests. Claude must
rerun the full-as-provisional and symlink cases plus a genuine signed receipt;
fork-specific event/token protections remain untested. Keep the scratch repo
until the rerun is complete. Push/PR/merge and the single promotion receipt
refresh remain approval gates, not local side effects.

## 2. Make useful validation fit a 16 GiB host

Follow `low_memory_validation_plan_2026-11-15.md`. Target about 8 GiB live
available host RAM, not a machine-size label. Keep resource profiles separate
from evidence tiers. First try the same mandatory contributor scope with one
compiler and smaller retained sessions. CPU-only checks remain a useful tier.
If mandatory units do not fit, define a distinctly named diagnostic tier;
never silently remove required tests from provisional evidence.

Next implementation work is identity-safe cold peak estimation, a conservative
unknown fallback, explicit reserve/admission messages and pressure-stop tests.
Validate with an approved constrained-host experiment before enabling automatic
profile selection. Admission is approximate scheduling, not an OOM guarantee.
Host RAM and device VRAM are separate budgets on this discrete-GPU machine.

## 3. Turn the numerical result into an integration candidate

The new CPU chain removes oracle intermediates. Its precision audit also removes
NumPy's internally-double root-gravity solve, replacing it with the explicit
rotation. The isolated CUDA prototype computes its own transforms, mass, bias,
scaled Cholesky, acceleration, first derivatives, second derivatives and
composition from q/qd/u and parsed model constants. No mixed precision is used
in the authored CUDA dynamics candidate. Standard float CUDA sine/cosine APIs
can still contain internal double instructions; the design note records the
compile-only audit and does not claim FP64-free machine code. Existing tolerances
and root Lie-Hessian ordering remain.

See `local_frame_gpu_design_2026-11-15.md` and the persistent task evidence for
exact acceptance results and limitations. A finite passing sample set is not a
condition-number support rule. It also does not establish full feature parity.

Next integration checkpoints, in order:

Brian's November 15 ruling supersedes autonomous implementation of this list:
Claude first presents a fast local-frame design with structure, pass counts and
scratch to Brian. Slow serial-pair code will not ship. No production emitter
work starts before that design decision; the passing private probe is an
accuracy baseline, not a performance candidate.

1. Review the local recursion and solve as a complete numerical method; retain
   independent float64 identities and adverse float32 cases as regression data.
2. Prove buffer lifetimes for the actual placement-aware FDSVA device interface.
   The standalone probe's static shared allocation is not the product scratch
   contract. Reuse existing output and scratch regions only when alias/lifetime
   proofs cover every resource tier and consumer. Stop on required public ABI
   or workspace expansion for Brian's decision.
3. Make an isolated opt-in production emitter without changing default generated
   code. Wire caller-provided gravity, model tables, runtime armature/inertia,
   batching, strides and the existing output conventions. Explicitly account for
   joint-dynamics terms, nonlinear coupling and non-root multi-DoF joints;
   unsupported private cases must fail clearly, never fake a support promise.
4. Run source/byte gates, the cross-feature matrix, direct-inner composition,
   non-default launch shapes, workspace canaries, sanitizer checks and relevant
   backend consumers under the shared correctness coordinator.
5. Only after correctness, schedule separately approved timing on a quiet host.
   Compare the entire dynamics chain. The directional-pair prototype is a
   deliberately repetitive baseline; no speed claim follows from fitting in
   shared memory or passing correctness. Jetson work needs separate approval.

Default selection and restoration of standalone receipt coverage wait for those
checks and the support decision. A public numerical kernel change is not part
of promoting infrastructure-only work.

## 4. Keep dependency narrowing report-only

The three report tools were read directly in this follow-up, and their focused
CPU tests rerun. They conservatively hash source families, inventory Python
emitter scope assumptions and compare controlled regenerated headers. They are
not C++ dependency resolvers or a security boundary for untrusted generators.

Next: resolve generated includes, macros, templates, overloads, layouts and
configuration dependencies for one inverse-dynamics family, with deliberately
adverse mutations. Keep the whole-family fallback for unresolved inputs.
Compiler-object reuse and signed test-evidence reuse have different keys and
different approval gates. Do not alter either cache or receipt semantics from
an unchanged report-only hash. Decentralized registration and broad helper
splits follow this evidence rather than precede it.

## 5. Scope warp/thread support separately

The existing `thread_warp_scope_proposal_2026-11-14.md` remains the contract
proposal. Start with a private warp inverse-dynamics inner, then thread inverse
dynamics, before Minv, forward dynamics and gradients. This is not the same
thing as the float32 prototype's one-thread-per-RHS solve within one block.

The first deliverable is closure-level synchronization and scratch ownership,
including two independent warps in one block, batch strides and composed
consumers. GLASS's scope primitives do not remove GRiD's tree-traversal and
barrier assumptions. Device-global scratch need not be pinned host memory.
Review the caller layout contract before exposing permanent API choices.

## Deferred and approval-gated work

No timing, pushes, PRs, merges, tags, uploads, environment approvals, receipt
campaigns, mixed precision, public ABI/scratch changes, dependency-based reuse
or embedded-device work is implied by these local commits. Broader coupling,
identification and paper campaigns remain separate. Review can proceed on the
local branches without disturbing the released code or another agent's work.
