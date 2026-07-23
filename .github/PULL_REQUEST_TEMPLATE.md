<!-- Thanks for contributing to GRiD! Please skim CONTRIBUTING.md + CLAUDE.md first. -->

## What this changes

<!-- One or two sentences. Link any related issue. -->

## Checklist

- [ ] Read the conventions in [`CLAUDE.md`](../CLAUDE.md) (single-block /
      thread-invariant kernels, fix-don't-guard, Pinocchio-authoritative).
- [ ] **Codegen discipline:** if this refactor shouldn't change emitted code,
      the generated `grid.cuh` is **byte-identical** (regenerated before/after +
      `diff`). If it does change emission, a CUDA-equivalence sign-off is included.
- [ ] The numpy oracle (`RBDReference`) and generated CUDA still agree (relevant
      `cuda_equivalence` / `pinocchio_equivalence` tests pass).
- [ ] `.venv/bin/python -m pytest -q` passes (GPU markers run on a GPU; the
      signed `gpu-proof.json` receipt is refreshed if GPU tests changed).
- [ ] No new `xfail`/`skip`/defensive guards masking a real bug.
- [ ] Docs / examples / `CLAUDE.md` updated if behavior or the public surface changed.
