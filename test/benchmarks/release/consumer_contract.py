"""Versioned dense consumer contract shared by the first native boundary slice."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from .protocol import agreement, accuracy_status, ACCURACY_POLICY_VERSION

VERSION = 1
BATCHES = (1, 4, 16, 64, 256, 1024)
OPERATIONS = ('inverse_dynamics', 'end_effector_pose')
ROBOTS = ('iiwa14', 'go2')


def contract(operation, batch, nq, nv, fixture_sha256):
    if operation not in OPERATIONS or batch not in BATCHES or min(nq, nv) < 1:
        raise ValueError('outside the first consumer slice')
    if len(fixture_sha256) != 64 or any(c not in '0123456789abcdef' for c in fixture_sha256):
        raise ValueError('fixture identity must be SHA256')
    return dict(schema=VERSION, operation=operation, batch=batch, fixture_sha256=fixture_sha256,
                input=dict(dtype='float32', layout='row-major', representation='dense',
                           fields=['q', 'v', 'a'] if operation == 'inverse_dynamics' else ['q'],
                           width=nq+2*nv if operation == 'inverse_dynamics' else nq,
                           bytes=4*batch*(nq+2*nv if operation == 'inverse_dynamics' else nq),
                           convention='project-order; local tangent; quaternion xyzw'),
                output=dict(dtype='float32', layout='row-major', representation='dense',
                            width=nv if operation == 'inverse_dynamics' else 6,
                            bytes=4*batch*(nv if operation == 'inverse_dynamics' else 6),
                            convention='project generalized force' if operation == 'inverse_dynamics'
                            else 'world xyz; roll pitch yaw'),
                boundaries=['host', 'gpu'], allocation='persistent I/O buffers outside samples; backend internals included',
                completion='host: result readable on return; gpu: result ready on device on return',
                staging='page-locked host buffers required; no pageable fallback',
                included='all boundary transfers, input mapping, conversion, compute and output normalization')


def compatible(a, b):
    required = {'schema', 'operation', 'batch', 'fixture_sha256', 'input', 'output',
                'boundaries', 'allocation', 'completion', 'staging', 'included'}
    return (isinstance(a, dict) and isinstance(b, dict)
            and required <= a.keys() and required <= b.keys() and a['schema'] == VERSION and a == b
            and a['boundaries'] == ['host', 'gpu']
            and isinstance(a['batch'], int) and a['batch'] in BATCHES
            and all(isinstance(a[k], dict) and isinstance(a[k].get('width'), int)
                    and a[k]['width'] > 0 and a[k].get('bytes') == 4*a['batch']*a[k]['width']
                    and a[k].get('dtype') == 'float32' and a[k].get('layout') == 'row-major'
                    and a[k].get('representation') == 'dense' for k in ('input', 'output')))


def accept(metadata, row_ids, expected, snapshots, *, policy='strict'):
    """Same rows, both outputs, pre/post oracle and inter-call repeatability.

    Metadata/coverage errors always fail, including under the warning policy.
    Keep the existing entrywise and explicit fp32-warning numerical policies.
    """
    result = dict(status='validation_failed', comparison_eligible=False,
                  accuracy_policy=policy, accuracy_policy_version=ACCURACY_POLICY_VERSION,
                  validated_row_ids=[], fixture_sha256=metadata.get('fixture_sha256') if isinstance(metadata, dict) else None)
    try:
        if not compatible(metadata, metadata):
            raise ValueError('missing or unsupported boundary contract')
        batch, width = metadata['batch'], metadata['output']['width']
        if list(row_ids) != list(range(batch)):
            raise ValueError('validation must cover every fixture row exactly once in order')
        expected = np.asarray(expected)
        if expected.shape != (batch, width):
            raise ValueError('oracle shape differs from contract')
        checks = {}
        for boundary in ('host', 'gpu'):
            for phase in ('before', 'after', 'repeat'):
                value = np.asarray(snapshots[boundary][phase])
                if value.shape != expected.shape:
                    raise ValueError('boundary output shape differs from contract')
                if value.dtype != np.float32:
                    raise ValueError('boundary output dtype differs from contract')
                checks[f'{boundary}_{phase}_oracle'] = agreement(value, expected)
            checks[boundary+'_repeatability'] = agreement(snapshots[boundary]['after'], snapshots[boundary]['repeat'])
            checks[boundary+'_pre_post'] = agreement(snapshots[boundary]['before'], snapshots[boundary]['after'])
        for phase in ('before', 'after', 'repeat'):
            checks['boundary_'+phase] = agreement(snapshots['host'][phase], snapshots['gpu'][phase])
        statuses = [accuracy_status(c, policy, metadata['operation'], 'float32') for c in checks.values()]
        status = ('validation_failed' if 'validation_failed' in statuses else
                  'accuracy_warning' if 'accuracy_warning' in statuses else 'validated')
        result.update(status=status, comparison_eligible=status != 'validation_failed', checks=checks,
                      validated_row_ids=list(row_ids), validated_rows=batch,
                      warning_checks=[k for k, c in checks.items() if not c['passed']])
    except (KeyError, ValueError, TypeError) as error:
        result['reason'] = str(error)
    return result


def summarize_repeats(samples, *, expected_repeats, noise_fraction=0.1):
    if len(samples) != expected_repeats or expected_repeats < 1:
        raise ValueError('incomplete repeats')
    if any(not s or any(not math.isfinite(x) or x < 0 for x in s) for s in samples):
        raise ValueError('invalid sample')
    means = [float(np.mean(s)) for s in samples]
    center, spread = float(np.median(means)), max(means)-min(means)
    return dict(repeat_means_us=means, median_repeat_mean_us=center, spread_us=spread,
                spread_fraction=spread/center if center else 0, noise_fraction=noise_fraction,
                noisy=spread > noise_fraction*center, samples_us=samples)


def build_identity(paths, flags, compiler):
    contents = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in sorted(set(map(Path, paths)))}
    payload = dict(files=contents, flags=flags, compiler=compiler)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
