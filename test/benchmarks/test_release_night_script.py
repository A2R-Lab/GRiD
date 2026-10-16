"""CPU-only gates for test/benchmarks/timing_night.sh: syntax, --help, the dry
run's exit code, stage list and job counts (the collector's dry plans), and the
rejection of unknown stages. Nothing here touches the GPU or writes a run."""
from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "test/benchmarks/timing_night.sh"


def run(*args):
    # The interpreter running this test has the dev stack; the script must not
    # depend on a repository venv (CI has none).
    env = {**os.environ, "PYTHON": sys.executable}
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=ROOT, text=True, capture_output=True, env=env)


def test_syntax_and_help():
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True).returncode == 0
    out = run("--help")
    assert out.returncode == 0
    for word in ("gaps", "release", "tutorial", "report", "--resume", "--dry-run", "--stages"):
        assert word in out.stdout
    assert "a2rlab-timing" not in SCRIPT.read_text()


def test_dry_run_lists_every_stage_with_job_counts(tmp_path):
    out = run("--dry-run", "--out", str(tmp_path / "night"))
    assert out.returncode == 0, out.stderr
    text = out.stdout
    assert "DRY RUN" in text and "maintainer is responsible" in text
    assert not (tmp_path / "night").exists()
    for stage in ("gaps", "release", "tutorial", "report"):
        assert re.search(rf"^stage {stage}", text, re.M), text
    counts = dict(re.findall(r"^  (\S+): collect .*\n    jobs=(\d+)", text, re.M))
    assert set(counts) == {"gap-frax-floating-gradients", "gap-grid-cuda-ee-gradient", "gap-bard-ee-gradient",
                           "release-core", "release-wrappers", "release-table", "release-table-core-baselines"}
    runnable = dict(re.findall(r"^  (\S+): collect .*\n    jobs=\d+ runnable=(\d+)", text, re.M))
    assert (counts["release-core"], counts["release-wrappers"]) == ("45", "48")
    assert runnable["gap-frax-floating-gradients"] == "4"         # 2 floating robots x 2 gradients
    assert runnable["gap-grid-cuda-ee-gradient"] == "3"
    assert runnable["gap-bard-ee-gradient"] == "3"
    assert "iterations=300" in text and "iterations=30" in text


def test_stage_subset_and_unknown_stage(tmp_path):
    out = run("--dry-run", "--stages", "tutorial", "--out", str(tmp_path / "n"))
    assert out.returncode == 0 and "stage tutorial" in out.stdout and "release-core" not in out.stdout
    bad = run("--dry-run", "--stages", "nope")
    assert bad.returncode == 2 and "unknown stage" in bad.stderr
    assert run("--bogus").returncode == 2
    missing = run("--dry-run", "--resume", str(tmp_path / "absent"))
    assert missing.returncode == 2
