#!/usr/bin/env python3
"""Run resumable CUDA equivalence checkpoints.

The stages intentionally stay small enough to produce useful progress and a
clear pass/fail boundary.  They reuse the persistent CUDA artifact cache unless
GRID_CUDA_DISABLE_CACHE=1 is set by the caller.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = ".venv/bin/python"


@dataclass(frozen=True)
class Command:
    label: str
    argv: tuple[str, ...]
    env: dict[str, str]


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    commands: tuple[Command, ...]


def _pytest(*args: str, env: dict[str, str] | None = None, label: str = "pytest") -> Command:
    return Command(
        label=label,
        argv=(PYTHON, "-m", "pytest", *args),
        env={} if env is None else env,
    )


STAGES: tuple[Stage, ...] = (
    Stage(
        name="toolchain",
        description="GPU and CUDA toolchain smoke checks.",
        commands=(
            Command("nvidia-smi", ("nvidia-smi",), {}),
            Command("nvcc --version", ("nvcc", "--version"), {}),
        ),
    ),
    Stage(
        name="layout",
        description="Fast generated-header/layout checks; second-order diagnostic remains skipped by default.",
        commands=(
            _pytest(
                "test/cuda_equivalents/test_cuda_codegen_layout.py",
                "test/cuda_equivalents/test_cuda_second_order_fallback.py",
                "-q",
                "-rs",
                env={"GRID_CUDA_RANDOM_SAMPLES": "0"},
                label="layout/codegen",
            ),
        ),
    ),
    Stage(
        name="fixed",
        description="Normal fixed-base equivalence checks.",
        commands=(
            _pytest(
                "test/cuda_equivalents/test_cuda_executable_equivalence.py::test_fixed_base_generated_cuda_matches_python_reference[fr3-fixed]",
                "-q",
                "-s",
                env={"GRID_CUDA_RANDOM_SAMPLES": "0"},
                label="fr3-fixed",
            ),
            _pytest(
                "test/cuda_equivalents/test_cuda_executable_equivalence.py::test_fixed_base_generated_cuda_matches_python_reference[go2-fixed]",
                "test/cuda_equivalents/test_cuda_executable_equivalence.py::test_fixed_base_generated_cuda_matches_python_reference[g1-fixed]",
                "-q",
                "-s",
                env={"GRID_CUDA_RANDOM_SAMPLES": "0"},
                label="go2/g1 fixed",
            ),
        ),
    ),
    Stage(
        name="fixed-fallback",
        description="Forced low-target fixed-base fallback.",
        commands=(
            _pytest(
                "test/cuda_equivalents/test_cuda_executable_equivalence.py::test_fixed_base_generated_cuda_matches_python_reference[fr3-fixed]",
                "-q",
                "-s",
                env={
                    "GRID_CUDA_RANDOM_SAMPLES": "0",
                    "GRID_CUDA_TARGET_SHARED_MEM_BYTES": "10000",
                },
                label="fr3 fixed forced fallback",
            ),
        ),
    ),
    Stage(
        name="floating",
        description="Floating-base ID, Minv, FD, and ID-gradient equivalence checks.",
        commands=(
            _pytest(
                "test/cuda_equivalents/test_cuda_executable_equivalence.py",
                "-k",
                "(iiwa14 or go2 or g1) and floating",
                "-q",
                "-rs",
                "-s",
                env={
                    "GRID_CUDA_RANDOM_SAMPLES": "0",
                    "GRID_CUDA_SAMPLE_NAMES": "zero,conservative",
                },
                label="iiwa14/go2/g1 floating",
            ),
        ),
    ),
    Stage(
        name="floating-stress",
        description="Floating-base stress and low-target fallback checks.",
        commands=(
            _pytest(
                "test/cuda_equivalents/test_cuda_executable_equivalence.py",
                "-k",
                "g1-floating",
                "-q",
                "-rs",
                "-s",
                env={
                    "GRID_CUDA_RANDOM_SAMPLES": "0",
                    "GRID_CUDA_SAMPLE_NAMES": "zero",
                },
                label="g1-floating selective spill",
            ),
            _pytest(
                "test/cuda_equivalents/test_cuda_executable_equivalence.py::test_floating_base_generated_cuda_matches_python_reference[iiwa14-floating]",
                "-q",
                "-s",
                env={
                    "GRID_CUDA_RANDOM_SAMPLES": "0",
                    "GRID_CUDA_SAMPLE_NAMES": "zero",
                    "GRID_CUDA_TARGET_SHARED_MEM_BYTES": "10000",
                },
                label="iiwa14 floating forced fallback",
            ),
        ),
    ),
    Stage(
        name="l2",
        description="L2 persisting parity checks.",
        commands=(
            _pytest(
                "test/cuda_equivalents/test_cuda_executable_equivalence.py::test_fixed_base_generated_cuda_matches_python_reference[fr3-fixed]",
                "-q",
                "-s",
                env={
                    "GRID_CUDA_RANDOM_SAMPLES": "0",
                    "GRID_CUDA_TARGET_SHARED_MEM_BYTES": "10000",
                    "GRID_CUDA_ENABLE_L2_PERSISTING": "1",
                },
                label="fr3 fixed fallback L2",
            ),
            _pytest(
                "test/cuda_equivalents/test_cuda_executable_equivalence.py",
                "-k",
                "g1-floating",
                "-q",
                "-rs",
                "-s",
                env={
                    "GRID_CUDA_RANDOM_SAMPLES": "0",
                    "GRID_CUDA_SAMPLE_NAMES": "zero",
                    "GRID_CUDA_ENABLE_L2_PERSISTING": "1",
                },
                label="g1 floating L2",
            ),
        ),
    ),
    Stage(
        name="corner-samples",
        description="Explicit positive/negative/tiny/mixed-sign sample coverage.",
        commands=(
            _pytest(
                "test/cuda_equivalents/test_cuda_executable_equivalence.py::test_fixed_base_generated_cuda_matches_python_reference[fr3-fixed]",
                "-q",
                "-s",
                env={
                    "GRID_CUDA_RANDOM_SAMPLES": "0",
                    "GRID_CUDA_SAMPLE_NAMES": "positive,negative,mixed_sign,tiny,velocity_only,accel_or_torque_only,near_limit",
                },
                label="fr3 fixed corner samples",
            ),
            _pytest(
                "test/cuda_equivalents/test_cuda_executable_equivalence.py::test_floating_base_generated_cuda_matches_python_reference[iiwa14-floating]",
                "-q",
                "-s",
                env={
                    "GRID_CUDA_RANDOM_SAMPLES": "0",
                    "GRID_CUDA_SAMPLE_NAMES": "positive,negative,mixed_sign,tiny,velocity_only,accel_or_torque_only,near_limit,floating_quat_identity,floating_quat_positive,floating_quat_mixed",
                },
                label="iiwa14 floating corner samples",
            ),
        ),
    ),
    Stage(
        name="long-random",
        description=(
            "Overnight correctness sweep with deterministic corner samples plus "
            "seeded random samples across the smoke fixed/floating robots. "
            "This intentionally excludes performance timing."
        ),
        commands=(
            _pytest(
                "test/cuda_equivalents/test_cuda_executable_equivalence.py",
                "-k",
                "fixed",
                "-q",
                "-s",
                env={
                    "GRID_CUDA_RANDOM_SAMPLES": "3",
                    "GRID_CUDA_SAMPLE_NAMES": "all",
                },
                label="all fixed smoke robots, deterministic+random",
            ),
            _pytest(
                "test/cuda_equivalents/test_cuda_executable_equivalence.py",
                "-k",
                "floating",
                "-q",
                "-s",
                env={
                    "GRID_CUDA_RANDOM_SAMPLES": "3",
                    "GRID_CUDA_SAMPLE_NAMES": "all",
                },
                label="all floating smoke robots, deterministic+random",
            ),
        ),
    ),
)


def _selected_stages(names: list[str]) -> list[Stage]:
    by_name = {stage.name: stage for stage in STAGES}
    if not names:
        names = ["toolchain", "layout", "fixed", "fixed-fallback", "floating", "floating-stress", "l2"]
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise SystemExit(
            "Unknown stage(s): "
            + ", ".join(unknown)
            + "\nKnown stages: "
            + ", ".join(by_name)
        )
    return [by_name[name] for name in names]


def _run_command(command: Command, timeout: int, dry_run: bool) -> int:
    env = os.environ.copy()
    env.setdefault("GRID_CUDA_PROGRESS", "1")
    env.setdefault("GRID_CUDA_VERBOSE_CACHE", "1")
    env.update(command.env)
    printable = " ".join(command.argv)
    print(f"\n--- {command.label} ---", flush=True)
    print(printable, flush=True)
    if command.env:
        print("env: " + " ".join(f"{key}={value}" for key, value in sorted(command.env.items())), flush=True)
    if dry_run:
        return 0
    try:
        result = subprocess.run(
            command.argv,
            cwd=REPO_ROOT,
            env=env,
            timeout=None if timeout <= 0 else timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"timeout after {timeout}s: {command.label}", flush=True)
        return 124
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help="Stage to run. Repeatable. Defaults to toolchain through l2. Add --stage corner-samples and --stage long-random for overnight coverage.",
    )
    parser.add_argument(
        "--timeout-per-command",
        type=int,
        default=0,
        help="Optional timeout in seconds for each command.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for stage in _selected_stages(args.stage):
        print(f"\n=== Stage: {stage.name} ===", flush=True)
        print(stage.description, flush=True)
        for command in stage.commands:
            returncode = _run_command(command, args.timeout_per_command, args.dry_run)
            if returncode != 0:
                print(f"stage failed: {stage.name} / {command.label} -> {returncode}", flush=True)
                return returncode
    print("\nAll selected CUDA stages passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
