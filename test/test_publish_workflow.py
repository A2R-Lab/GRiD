"""CPU-only guards for receipt ancestry and publication boundaries."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _jobs():
    return yaml.safe_load((ROOT / '.github/workflows/publish.yml').read_text())['jobs']


def test_publish_receipt_checkout_includes_candidate_history():
    receipt = _jobs()['receipt']
    checkout = next(step for step in receipt['steps']
                    if step.get('uses', '').startswith('actions/checkout@'))
    assert checkout['with']['fetch-depth'] == 0
    verification = '\n'.join(step.get('run', '') for step in receipt['steps'])
    assert '--policy test/gpu-proof-policy-release.yaml' in verification
    assert '--expected-skips test/gpu-proof-expected-skips.txt' in verification


def test_manual_rehearsal_cannot_publish_to_production():
    jobs = _jobs()
    assert jobs['pypi']['if'] == "startsWith(github.ref, 'refs/tags/v')"
    assert jobs['pypi']['needs'] == 'testpypi'
    assert jobs['pypi']['environment']['name'] == 'pypi'
    assert jobs['testpypi']['environment']['name'] == 'testpypi'


def test_testpypi_requires_successful_build_without_inheriting_receipt_skip():
    job = _jobs()['testpypi']
    assert job['needs'] == 'build'
    # An explicit status function avoids GitHub's implicit success() across the
    # dependency chain, where the tag-only receipt job is skipped on dispatch.
    assert job['if'] == "${{ !cancelled() && needs.build.result == 'success' }}"


def test_docker_header_generation_creates_output_directory(tmp_path):
    """Run the recipe's shell prelude with a writer that requires its parent."""
    dockerfile = (ROOT / 'install/docker/Dockerfile').read_text().replace('\\\n', '')
    run = next(line.removeprefix('RUN ') for line in dockerfile.splitlines()
               if line.startswith('RUN ') and 'grid-generate ' in line)
    # The following Python import/version print is independent of directory setup.
    command = run.split('&& python -c', 1)[0]
    command = command.replace('/opt/grid', str(tmp_path / 'grid'))
    executable = tmp_path / 'grid-generate'
    executable.write_text(
        f'#!{sys.executable}\n'
        'import pathlib, sys\n'
        'pathlib.Path(sys.argv[sys.argv.index("-o") + 1]).write_text("header\\n")\n'
    )
    executable.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ['PATH'])
    # The original recipe omitted mkdir; the same writer must fail in that case.
    old_command = command[command.index('grid-generate '):]
    result = subprocess.run(['sh', '-ec', old_command], env=env, capture_output=True,
                            text=True)
    assert result.returncode != 0 and 'FileNotFoundError' in result.stderr
    subprocess.run(['sh', '-ec', command], env=env, check=True, capture_output=True)
    assert (tmp_path / 'grid/iiwa14/grid.cuh').read_text() == 'header\n'


@pytest.mark.parametrize('scenario', [
    'success', 'private', 'bad-digest', 'busy-lock', 'gpu-busy',
    'compiler-busy', 'baked-failure', 'reload-failure',
])
def test_container_smoke_protocol(tmp_path, scenario):
    """Exercise the real shell orchestration without Docker, network or GPU."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    calls = tmp_path / 'calls.jsonl'
    stub = (
        f'#!{sys.executable}\n'
        'import json, os, pathlib, sys\n'
        'tool = pathlib.Path(sys.argv[0]).name\n'
        'args = sys.argv[1:]\n'
        'scenario = os.environ["SMOKE_SCENARIO"]\n'
        'with open(os.environ["SMOKE_CALLS"], "a") as f:\n'
        '    f.write(json.dumps([tool, args]) + "\\n")\n'
        'if tool == "pgrep":\n'
        '    sys.exit(0 if scenario == "compiler-busy" else 1)\n'
        'if tool == "nvidia-smi":\n'
        '    if scenario == "gpu-busy": print("12345")\n'
        '    sys.exit(0)\n'
        'assert args[0] == "--config"\n'
        'assert not list(pathlib.Path(args[1]).iterdir()), "Expected empty auth config"\n'
        'args = args[2:]\n'
        'if args[0] == "pull":\n'
        '    sys.exit(1 if scenario == "private" else 0)\n'
        'if args[:2] == ["image", "inspect"]:\n'
        '    if "--format" in args:\n'
        '        print("bad" if scenario == "bad-digest" else "ghcr.io/a2r-lab/grid@sha256:" + "a"*64)\n'
        '    else: print("{}")\n'
        '    sys.exit(0)\n'
        'assert args[0] == "run"\n'
        'if scenario == "baked-failure" and "/checks/baked.py" in args: sys.exit(42)\n'
        'if scenario == "reload-failure" and "--reload" in args: sys.exit(42)\n'
    )
    for name in ('docker', 'nvidia-smi', 'pgrep'):
        executable = bin_dir / name
        executable.write_text(stub)
        executable.chmod(0o755)
    lock = tmp_path / 'shared.lock'
    lock.touch()
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
               TMPDIR=str(tmp_path), GRID_CONTAINER_TIMING_LOCK=str(lock),
               SMOKE_SCENARIO=scenario, SMOKE_CALLS=str(calls))
    with lock.open('r') as held:
        if scenario == 'busy-lock':
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(['bash', str(ROOT / 'tools/check_container_image.sh'), '0.5.3'],
                                env=env, capture_output=True, text=True, timeout=15)
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    runs = [args for tool, args in records if tool == 'docker' and args[2] == 'run']
    outputs = list(tmp_path.glob('grid-container-smoke-*'))
    assert len(outputs) == 1
    assert (outputs[0] / 'PASS').exists() == (scenario == 'success')
    if scenario == 'success':
        assert result.returncode == 0, result.stderr + result.stdout
        assert len(runs) == 4
        for args in runs:
            assert 'ghcr.io/a2r-lab/grid@sha256:' + 'a'*64 in args
            assert args[args.index('--network') + 1] == 'none'
            assert args[args.index('--gpus') + 1] == 'all'
        assert '--expected-version' in runs[1] and runs[1][-1] == '0.5.3'
        assert '--reload' not in runs[2] and '--reload' in runs[3]
        assert runs[2][runs[2].index('--work') + 1] == runs[3][runs[3].index('--work') + 1]
        mounts = [runs[2][i+1] for i, arg in enumerate(runs[2][:-1]) if arg == '--mount']
        assert len(mounts) == 3 and sum(m.endswith(',readonly') for m in mounts) == 2
    else:
        assert result.returncode != 0
        expected_runs = {'baked-failure': 2, 'reload-failure': 4}.get(scenario, 0)
        assert len(runs) == expected_runs
