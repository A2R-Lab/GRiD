"""CPU-only guards for receipt ancestry and publication boundaries."""
from pathlib import Path
import os
import subprocess
import sys

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
