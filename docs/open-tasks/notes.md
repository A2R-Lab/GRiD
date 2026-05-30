# Relocated Code Notes

This file holds explanatory / TODO comment blocks that were moved out of the
source so the code reads cleanly. Each entry is keyed by `file:line` (line
numbers as of the relocation; treat them as anchors, not guarantees). The
in-code site now carries at most a one-line pointer back here.

These are notes only. None of them describe behavior that changed in this batch.

---

## RBDReference/RBDReference.py

### `RBDReference.py:936` — `end_effector_pose` docstring TODO

> End Effector Positions
>
> `offsets` is an array of np matrices of the form `(offset_x, offset_y, offset_z, 1)`.
>
> TODO: Add and test floating base support.

The end-effector pose path (`end_effector_pose`, and the chain helpers
`forwardChain` / `backwardChain`) is exercised on fixed-base robots. Floating-base
EE pose has not been validated; adding and testing it is open work.

### `RBDReference.py:987` — `eePos_from_Xmat_hom` per-branch offsets

> TODO handle different offsets for different branches

`eePos_from_Xmat_hom` currently applies `ee_offsets[0]` to every end-effector
joint. Supporting a distinct offset per EE branch is open work.

### `RBDReference.py:1907` — `minv_bpass` reduced-model caveat

> Backward pass.
>
> Use `get_joint_index_v(ind)` for matrix indices so the pass works uniformly
> across fixed-base, floating-base, and (the indexing parts of) mimic robots.
> Mimic-joint reduced-model handling itself is done in the public `minv` entry by
> falling back to `inv(crba(q))`; the ABA recursion below assumes per-body
> `(U, d)` which would need projection for true reduced-model `M^{-1}`.

This documents an intentional divergence: for mimic robots, `minv` does not run
the ABA-style backward recursion (which is per-body and not reduced-model aware);
it returns the dense `inv(crba(q))` instead, which matches Pinocchio's reduced
model. See also the alignment-backlog entry recording this as known-correct.

### `RBDReference.py:2140-2144` — mimic fast-path external forces (T4 owns the fix)

> NOTE: external forces (`f_ext`) are not currently threaded through this fast
> path; `rnea` does not apply them either. The downstream test suite does not
> exercise `f_ext` on mimic robots, but a future task should consolidate the
> external-force handling so the mimic path supports it cleanly.

**Status: T4 owns the fix.** The mimic-aware `aba` fast path computes
`qdd = Minv @ (tau - bias)` with `bias = rnea(q, qd, 0)`; neither the bias nor
the solve threads `f_ext` through. This is recorded in
`RBDReference/tests/PINOCCHIO_ALIGNMENT_BACKLOG.md` as a T4 fix item.

### `RBDReference.py:2158` — floating-base `aba` allocation sizing

> allocate memory TODO check NB vs. n

In the floating-base `aba` branch the per-body arrays (`v`, `c`, `a`, `IA`,
`pA`) are sized by `NB` while the joint-space arrays (`f`, `U`, `u`, `qdd`) are
sized by `n = len(qd)`. The TODO asks whether all of these are using the right
dimension; it has not been audited. No observed failure — recorded for the
cleanup pass.

### `RBDReference.py:655-658` — `mxS` NumPy `ndim>0`-to-scalar DeprecationWarning (catalog only)

The `mxS` helper flattens `S` with `np.asarray(S).reshape(-1)` specifically so
each `S[k]` is a Python scalar before being passed as `alpha` to `mx1`..`mx6`.
Without that flatten, a 6x1 subspace column makes `S[k]` a 1-element array, and
passing a `ndim>0` array where a scalar is expected raises NumPy's
"Conversion of an array with ndim > 0 to a scalar" DeprecationWarning (slated to
error in a future NumPy) on every element write.

**Catalog only.** The in-code comment here is an accurate explanation of the
existing mitigation and is intentionally retained. The broader project-wide
DeprecationWarning cleanup is the later warnings sweep, not this batch. Other
`mx1`..`mx6` call sites that pass raw array slices as `alpha` should be audited
in that sweep.

---

## URDFParser/Joint.py

### `Joint.py:38` — `child` field unused

> child link name TODO - currently unused

`Joint.child` is stored at construction but not consumed by the parser or
codegen. Either wire it into the parent/child indexing helpers or drop it.

### `Joint.py:260` — unsupported joint-type guard (guard left in place)

The `else` branch of `Joint.set_type` prints
`Only revolute and fixed joints currently supported (outside of floating base)!`
and calls `exit()`. In practice the parser also handles `continuous`,
`prismatic`, and `floating`; the guard fires for genuinely unsupported types
(notably **`planar`**) and for any other unknown `jtype`.

**The guard is intentionally left in place** (a hard `exit()` on an unsupported
joint type is the current contract). The planar-joint gap and the
arbitrary/skew-axis gap are tracked in
`docs/open-tasks/urdf_feature_matrix.md`.

---

## URDFParser/URDFParser.py

### `URDFParser.py:74` — missing-`<inertial>` "world base frame" assumption

> Link [name] does not have inertial properties. Assuming this is the fixed
> world base frame. Else there is an error with your URDF file.

When a `<link>` has no `<inertial>`, the parser assumes it is the fixed world
base and sets zero origin + zero inertia. A real link that legitimately omits
inertia (or a degenerate/zero-inertia link, e.g. `rizon4`) is silently treated
as the world base, which produces wrong dynamics rather than a diagnostic.
Degenerate-`<inertial>` detection is tracked in
`docs/open-tasks/urdf_feature_matrix.md`.
