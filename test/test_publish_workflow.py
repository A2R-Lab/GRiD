"""CPU-only guards for receipt ancestry and publication boundaries."""
from pathlib import Path

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
