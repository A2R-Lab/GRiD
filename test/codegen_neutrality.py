"""Byte-neutrality prover for SPLIT_REFRESH's cuda-shard carry (2026-09-11).

Cuda shards' NARROW fingerprints deliberately cover only test-side files
(member modules + runner .cu + cuda_harness) so test edits never force a
multi-day cuda re-run. The generator inputs — grid_codegen/**, the URDF
assets, and the URDFParser submodule — are pinned at the RECEIPT level
(repo.commit_sha) instead. That leaves refresh with a blind spot: a carried
cuda shard's proof was produced against the OLD tree's emitted headers, and
nothing re-checks that the CURRENT tree still emits the same bytes.

This module closes it the content-key way (user decision 2026-09-11,
option 2): when the generator inputs changed vs the old receipt's commit,
regenerate a covering matrix of headers from BOTH trees (a temporary
worktree at the old sha vs the current working tree) and byte-compare.
All-identical → the carried shards' proofs are proofs of the same bytes
(the nvcc content keys would not even rotate); any difference → the whole
cuda domain is stale and the refresh re-runs it honestly.

Scope note: byte-equality over MATRIX is a *covering* proof, not a per-test
one — the rows are chosen to touch every Python-conditional emission block
(full default set, subset gating, multi-target batch [the 7.z7 class],
floating/quaternion + mjx twins, floating second-order world+body, branching
topology). A test whose exact gen kwargs prune differently could in
principle diverge while MATRIX stays identical; extend MATRIX when adding a
new Python-conditional emission block (the 7.z7 lesson: byte-identity gates
only cover configurations something actually emits).

Escape hatch: GRID_REFRESH_ASSUME_NEUTRAL=1 skips the proof and carries
anyway (loudly) — for a change KNOWN neutral where the ~2×matrix codegen
cost is unwanted. Never the default.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Generator-input paths: a change here (vs the old receipt's commit) is what
# triggers the proof. Submodule pin bumps for URDFParser show up in git diff
# under its path.
CODEGEN_INPUT_PATHS = ("grid_codegen", "config/robot_assets",
                      "external/URDFParser")

# (name, robot_id, floating_base, gen_all_code kwargs) — one row per
# Python-conditional emission family; see the scope note above.
MATRIX = (
    ("iiwa14-fixed-full", "iiwa14", False, {}),
    ("iiwa14-fixed-subset", "iiwa14", False,
     {"algorithm_list": ["inverse_dynamics", "inverse_dynamics_gradient"]}),
    ("iiwa14-fixed-multitarget", "iiwa14", False,
     {"algorithm_list": ["end_effector_pose"],
      "multi_target_batch": [
          {"anchor_jid": 6, "offset": (0.0, 0.0, 0.0)},
          {"anchor_jid": 6, "offset": (0.03, -0.02, 0.05)}]}),
    ("go2-floating-full", "go2", True, {}),
    ("go2-floating-so-world", "go2", True,
     {"algorithm_list": ["idsva_so_body_frame", "fdsva_so"],
      "enable_floating_second_order": True,
      "enable_idsva_so_world_frame": True}),
    ("baxter-fixed-full", "baxter", False, {}),
)

_GEN_SCRIPT = textwrap.dedent("""\
    import contextlib, os, sys
    sys.path.insert(0, "test"); sys.path.insert(0, ".")
    import warnings; warnings.filterwarnings("ignore")
    from config import robot_urdf
    from external.URDFParser.URDFParser import URDFParser
    from grid_codegen.GRiDCodeGenerator import GRiDCodeGenerator
    robot_id, floating, out_path = sys.argv[1], sys.argv[2] == "1", sys.argv[3]
    kwargs = eval(sys.argv[4])
    robot = URDFParser().parse(str(robot_urdf(robot_id)), floating_base=floating)
    g = GRiDCodeGenerator(robot, FILE_NAMESPACE="grid")
    with open(os.devnull, "w") as d, contextlib.redirect_stdout(d), \\
            contextlib.redirect_stderr(d):
        g.gen_all_code(output_path=out_path, **kwargs)
""")


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout


def codegen_inputs_changed(old_sha: str) -> bool:
    """True when any generator-input path differs between the old commit and
    the CURRENT WORKING TREE (tracked diffs and untracked files alike)."""
    diff = subprocess.run(
        ["git", "diff", "--quiet", old_sha, "--", *CODEGEN_INPUT_PATHS],
        cwd=REPO_ROOT)
    if diff.returncode != 0:
        return True
    untracked = _git("status", "--porcelain", "--", *CODEGEN_INPUT_PATHS)
    return any(line.startswith("??") for line in untracked.splitlines())


def _submodule_pins_match(old_sha: str) -> bool:
    """The worktree reconstruction symlinks the CURRENT external/ checkouts —
    only valid when the old commit pinned the same submodule SHAs."""
    old = {}
    for line in _git("ls-tree", old_sha, "external").splitlines():
        meta, name = line.split("\t")
        if meta.split()[1] == "commit":
            old[name.split("/")[-1]] = meta.split()[2]
    cur = {}
    for line in _git("submodule", "status").splitlines():
        sha, path = line.split()[0].lstrip("+-U"), line.split()[1]
        cur[path.split("/")[-1]] = sha
    return all(cur.get(k) == v for k, v in old.items())


def _generate(tree: Path, row, out: Path) -> None:
    name, robot_id, floating, kwargs = row
    subprocess.run(
        [sys.executable, "-c", _GEN_SCRIPT, robot_id,
         "1" if floating else "0", str(out), repr(kwargs)],
        cwd=tree, check=True, capture_output=True, text=True, timeout=1800)


def prove_byte_neutrality(old_sha: str) -> tuple[bool, str]:
    """Generate MATRIX from a worktree at ``old_sha`` and from the current
    tree; return (all byte-identical, detail). Raises on infrastructure
    failure (a row that cannot generate in EITHER tree is a real signal —
    surface it, don't carry)."""
    if not _submodule_pins_match(old_sha):
        return False, "submodule pins differ from the old receipt's commit"
    with tempfile.TemporaryDirectory(prefix="grid_neutrality_") as td:
        tmp = Path(td)
        wt = tmp / "old_tree"
        _git("worktree", "add", "--detach", "-q", str(wt), old_sha)
        try:
            # worktrees do not populate submodules; pins match (checked), so
            # the current checkouts are the old commit's content.
            for sub in (REPO_ROOT / "external").iterdir():
                dst = wt / "external" / sub.name
                if dst.is_dir() and not any(dst.iterdir()):
                    dst.rmdir()
                if not dst.exists():
                    dst.symlink_to(sub)
            diffs = []
            for row in MATRIX:
                old_out = tmp / f"{row[0]}.old.cuh"
                new_out = tmp / f"{row[0]}.new.cuh"
                _generate(wt, row, old_out)
                _generate(REPO_ROOT, row, new_out)
                if old_out.read_bytes() != new_out.read_bytes():
                    diffs.append(row[0])
            if diffs:
                return False, "matrix rows differ: " + ", ".join(diffs)
            return True, f"{len(MATRIX)} matrix rows byte-identical"
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                           cwd=REPO_ROOT, capture_output=True)


def cuda_carry_soundness(old_receipt: dict) -> tuple[bool, str]:
    """The refresh-time gate: may fingerprint-clean cuda shards be carried?

    Returns (carry_ok, reason). Carry is sound when the generator inputs are
    unchanged vs the old receipt's commit; otherwise it must be PROVEN
    byte-neutral (or explicitly assumed via GRID_REFRESH_ASSUME_NEUTRAL=1).
    A dirty old receipt cannot anchor the proof → not carryable once inputs
    changed."""
    repo = old_receipt.get("repo") or {}
    old_sha = repo.get("commit_sha")
    if not old_sha:
        return False, "old receipt records no commit_sha"
    if not codegen_inputs_changed(old_sha):
        return True, "generator inputs unchanged vs old receipt"
    if os.environ.get("GRID_REFRESH_ASSUME_NEUTRAL") == "1":
        return True, ("generator inputs CHANGED — carried WITHOUT proof "
                      "(GRID_REFRESH_ASSUME_NEUTRAL=1)")
    if repo.get("dirty"):
        return False, ("generator inputs changed and the old receipt was "
                       "built from a DIRTY tree — cannot reconstruct it for "
                       "the byte-neutrality proof")
    ok, detail = prove_byte_neutrality(old_sha)
    if ok:
        return True, f"generator inputs changed; PROVEN byte-neutral ({detail})"
    return False, f"generator inputs changed and are NOT byte-neutral ({detail})"
