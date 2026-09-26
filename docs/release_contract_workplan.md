# Pre-release contract changes — 2026-09-26

This tracked handoff lets collaborators distinguish the shared checkpoint from
the next implementation batch. The user approved a clean break: **no backward
compatibility requirement**. None of the numerical/API changes below is
implemented by the documentation checkpoint.

## Checkpoint scope

The branch contains the release measurement tooling and captures' derived
website assets, corrected public documentation, a source-checked 35-method
backend inventory, and CPU-tested input/parameter examples. Raw captures and
the longer local audit in `docs/open-tasks/` are not tracked.

The site remains a review preview. A branch push is not a main merge, public
release, or fresh GPU correctness receipt. In particular, documentation of a
restriction is not evidence that the next implementation removes it.

## Approved next batch

1. **Integration semantics.** Replace nonstandard midpoint/RK behavior with
   properly specified full-state schemes, including a defined manifold update
   for quaternion models. Remove unsupported/misleading variants rather than
   preserving aliases. Rename the current one-evaluation `trapezoidal` scheme
   descriptively. Update reference, CUDA, gradients, framework rules and docs
   together. Require independently solvable convergence tests, not only CUDA
   versus our own reference. Do not promise unsupported multi-stage Hessians.
2. **Momentum cost.** Return the full state gradient. Form a clearly identified
   Gauss–Newton Hessian from the complete residual Jacobian, including the
   configuration dependence of the centroidal momentum matrix. An exact cost
   Hessian is distinct and not implied. Test derivatives using the declared
   tangent perturbations and nonzero velocities/residuals.
3. **Public coordinate shapes.** Configuration has NQ entries; physical
   velocities, accelerations and generalized forces have NV entries. Dynamics
   vector outputs are NV-wide; state is NQ+NV. Geometric state derivatives use
   tangent dimensions and must not masquerade as ambient quaternion
   derivatives. Eliminate public padding consistently across NumPy/JAX/Torch;
   explicitly settle/document direct CUDA storage separately. Do not infer
   quaternion ordering or frame convention from width alone.
4. **Benchmark labels.** Label the operation `forward_dynamics` as FD, and its
   derivatives as grad/Hessian FD, rather than naming all implementations ABA.
   Update report labels, website assets, prose and regression checks. Replot
   from the identical existing report and preserve every measured value,
   status, comparison boundary and capture hash.

For ordinary scalar-joint models, fixed-base NQ=NV=n; quaternion floating-base
NQ=n+7 and NV=n+6. Independent spherical joints add one configuration coordinate
relative to tangent width even on a fixed base. NQ is a model-derived count,
not itself a declaration of quaternion representation. Generalized forces have
NV entries including base coordinates, not merely the actuated-joint count.

## Timing and validation boundaries

- Integrators and momentum costs are not operations in the current release
  timing matrix. Their changes do not directly replace any measured cell.
  Check shared generated code, scratch/resource declarations and artifact
  identities before asserting unchanged timing elsewhere.
- Public packing/output changes can affect wrapper and transfer costs,
  especially on floating-base robots. Refresh affected boundary measurements;
  do not attach old wrapper timings to the new interface without evidence.
- Core CUDA timings may remain useful if generated code, build inputs and
  execution path are demonstrably unchanged. Treat reuse as an explicit
  provenance decision, not an assumption based on operation names.
- Label-only plotting needs no new GPU runs. Preserve existing raw captures.
- Run the timing-stability pilot before deciding on a full matched
  re-collection; CPU power policy must match across compared backends.
- Run the final correctness receipt after the implementation batch settles.
  Main merge and publication still require the user's approval.

## Parallel-work guidance

Independent work can proceed on reproducibility packaging, benchmark label
presentation, and docs/navigation. Numerical code, shared ABI definitions,
bindings, reference implementations and their tests overlap heavily; agree on
file ownership and land coordinated patches rather than making independent
shape migrations. Existing scripts at this checkpoint still use the OLD API.

Deferred work remains adapter expansion, collision benchmarking, broad
performance optimization and exhaustive backend/joint coverage beyond the
release's declared supported combinations.
