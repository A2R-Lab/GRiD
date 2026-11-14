# CPU numerical attribution and generated-dependency checks

This follow-up leaves the review branches at `91388bef` and `146552db`
unchanged. The implementation is local commit `cdaf554b` on
`codex/cpu-attribution-dependencies-20261114`. It adds isolated study tools and
tests, not product algorithms, receipt changes or cache reuse.

## Numerical experiment

The preceding CPU experiment used a float32 local-frame CRBA, an equilibrated
Cholesky solve and a local-frame dM/dq. Keeping the other captured device tensor
errors left two failing cases among 64: iiwa14 `high_velocity_accel` and g1
`cuda_random_1`. This experiment isolates their d²tau/dq² contribution.

`tools/cpu_qq_study.py` differentiates local-frame RNEA twice with explicit
first- and second-derivative arrays. The floating root uses the existing ordered
Lie-tangent convention. Its Hessian is not symmetrized. Scalar joints and constant
mimic folds are supported, including the saved fr3 cases. Nonlinear couplings and
non-root multi-DoF joints are explicitly outside this experimental implementation.

Every case first compares the float64 local recursion with the existing
reference at the same candidate acceleration. All 64 pass at
`atol=2e-9, rtol=2e-10`; the largest absolute discrepancy is `9.641e-11`.
Independent scalar-joint finite differences also cover a synthetic branched tree
with and without a shared mimic coordinate. A root test checks that the ordered
Lie Hessian has not accidentally been symmetrized.

The world-frame comparison executes the existing reference method in an isolated
namespace with dtype-controlled constructors and helpers. It does not patch the
installed reference. Its local floating arrays are checked for the requested
dtype. Both paths receive identical rounded spatial transforms, subspaces and
inertias. The algebraic reassociation variant reverses additive expression order
and changes left-associated matrix chains to right-associated ones. Its float64
identity check also passes all cases; maximum discrepancy is `1.708e-11`.

All composed outputs use the unchanged full FDSVA-SO gate. The table counts
failures, not individual failing entries.

| q-Hessian source, with preceding matrix and dM/dq candidates | Failures / 64 |
|---|---:|
| Captured device error transplanted to candidate acceleration | 2 |
| Exact oracle q-Hessian rounded to float32 | 0 |
| Local-frame RNEA derivative, float32 | 0 |
| Local-frame derivative, reversed summation and sibling order | 0 |
| Existing world-frame algorithm emulated in float32 on CPU | 2 |
| World-frame algorithm with algebraic reassociation | 1 |

Reassociation passes the original two failures but introduces iiwa14
`cuda_random_1`. Its maximum absolute output error is `0.03435` against that
case's `0.03` norm-guard ceiling. This is evidence of rounding-order sensitivity,
not a robust fix. The local-frame formulation survives both tested orders.

| Case | Transplanted q-Hessian: max output error | Local float32: max output error | Case's norm-guard max-error ceiling |
|---|---:|---:|---:|
| iiwa14 high-velocity | 0.26902 | 0.02903 | 0.17138 |
| g1 random-1 | 0.34656 | 0.03088 | 0.33787 |

The ceiling alone is not acceptance. The elementwise test and existing combined
norm/max-error fallback remain in force. Full metrics and tolerance parameters
are in each result row.

### Entrywise attribution

For explanation only, propagate the captured q-Hessian error through the inverse
of the candidate CPU mass matrix in float64. This is a linear error attribution,
not another candidate solve or a claim about exact float32 operation order.
Indices below are zero-based reduced velocity indices.

- g1's worst propagated output is `[32,33,33]`, about `+0.36309`.
  Tensor entry `[32,33,33]` has error `-1.3214e-4` and contributes `+0.36335`.
  Correcting only that entry makes the complete counterfactual gate pass.
- iiwa14's worst propagated output is `[12,11,12]`, about `-0.26902`.
  Torque rows 10, 12 and 8 contribute approximately `-0.15460`, `-0.09342`
  and `-0.02140`. None of the six highest-weight single-entry substitutions
  passes the full gate.

The absolute-sum/net contribution ratios are about 1.019 and 1.010 respectively.
The final propagation of those errors is therefore mostly amplification, not
cancellation between opposing error terms. The arithmetic-order experiment
supports sensitivity in upstream tensor formation. It does not identify one
specific CUDA instruction or prove that frame choice is the only cause.

### Limits and next numerical decision

The result is a promising float32 algorithm candidate, not an end-to-end device
fix. Spatial inputs are rounded reference values, not device-evaluated transforms.
The bias right-hand side and first-order ID derivative still come from the oracle.
Unreplaced tensor blocks retain the earlier device-error transplant, which assumes
their error is unchanged at a different acceleration. The 64 saved captures are
eight samples per robot, not a new random sweep or a user-checkable support envelope.

The experimental derivative arrays scale as O(NB * nv²); this is not an approved
CUDA scratch design or a performance proposal. No mixed-precision product code,
numerical kernel, public signature, tolerance or support policy changed.

After review, the next numerical package should first replace the remaining
CPU oracle substitutions, particularly bias and first-order derivatives, and
validate a complete CPU arithmetic chain. Any GPU implementation or capture must
be separately approved. That device experiment would need actual M, candidate
acceleration, gradients and all four tensors at that acceleration. It must include
the formerly failing cases and unchanged full gates before any support claim.

## Dependency regeneration experiment

`tools/generated_dependency_study.py` generates fixed-base iiwa14 ID-only headers
from private copied fixtures. It uses clean peers at the source's pinned commits.
Deliberately incorrect mutations are text-generation fixtures only. They are never
compiled or executed. The original source, review worktrees and peers are not edited.

The inventory records signatures, line spans and byte hashes for both qdd/no-qdd
overloads of `inverse_dynamics_inner` and `inverse_dynamics_inner_vaf`. This bounded
lexer is not a C++ dependency parser; raw strings and incomplete definitions fail
closed. Observed function-body bytes do not include transitive dependencies.

| Mutation | Whole header changed | ID definitions changed | Broad object key changed | Evidence key changed |
|---|---|---|---|---|
| ID emitter body | yes | yes | yes | yes |
| Shared synchronization emitter | yes | yes | yes | yes |
| Unrequested FD emitter | no | no | yes | yes |
| Out-of-line spatial helper | yes | no | yes | yes |
| Robot mass constant | yes | no | yes | yes |
| Runtime-inertia generation option | yes | no | yes | yes |
| Evidence-policy file bytes | no | no | no | yes |
| README bytes | no | no | no | no |

Baseline generation was repeated byte-identically. Across ten generations
(eight mutations and two baselines), all five header-changing mutations were
detected by the current conservative object identity. The unrequested FD edit is
a visible conservative false positive, not an optimization.

Three mutations change the whole header without changing the ID definitions.
An ID-body-only hash would miss these relevant surrounding inputs. This is why
the report-only analyzer retains broad source and context dependencies. The
policy mutation is a byte-identity check, not a semantic policy change.

No compiler depfile, macro expansion, template instantiation or object comparison
was performed. Compiler/architecture context is explicitly unmeasured. No cache
or receipt reuse, narrowed test selection or performance gain is authorized by
these results. A future narrower graph needs explicit transitive generated helper,
model-table, configuration and external-header edges, plus independent compiler
dependency validation before it can replace the broad fallback.

## Validation and artifacts

Focused CPU suite: 48 passed, comprising 18 new study tests and 30 existing
report-only dependency tests. This is not a full product CPU/GPU rerun. The tools
are not imported by product code or the receipt launcher; the study tests are
invoked explicitly rather than added to the production CPU-lane manifest.

Persistent local evidence under the shared checkout's ignored `docs/open-tasks/`:

- `fdsva_fp32_study_2026-11-12/codex-qq-verified-20261114/`: authoritative post-commit
  numerical provenance, 64 JSON result rows and per-case entry attribution.
- `dependency_regeneration_2026-11-14-verified/`: authoritative post-commit header
  fixtures, logs, snapshots, explicit export metadata and ten comparison rows.
- `codex_qq_verified_2026-11-14.log`, `codex_dependency_verified_2026-11-14.log`
  and `codex_cpu_attribution_dependency_tests_2026-11-14.log`.

Earlier directories are retained as preliminary evidence. The first dependency
attempt failed before producing a header because the private fixture lacked
checkout resource markers. The harness now copies `pyproject.toml` and references
validated pinned peers. The final numerical run matches spatial inputs across
local/world paths; preliminary runs without that explicit matching are superseded.

Reproduce with the shared `.venv/bin/python`, explicit worktree/bindings/peer
`PYTHONPATH`, `CUDA_VISIBLE_DEVICES=''`, `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1`. Choose new output directories; the drivers refuse overwrite:

```sh
python tools/run_cpu_qq_study.py --study-dir /path/to/saved/study --out /new/numerical/output
python tools/generated_dependency_study.py --source /clean/source --peers /pinned/external --out /new/dependency/output
python -m pytest -q test/test_cpu_qq_study.py test/test_generated_dependency_study.py test/test_dependency_report.py
```

No GPU run, timing, product kernel edit, cache purge, peer edit/message, subagent,
push, PR, merge, tag or upload was used. Local commits and ignored evidence are
not a remote backup. Both earlier review branches remain frozen.
