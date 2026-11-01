"""Descriptor/header/binary consistency for prepared consumer artifacts.

This detects stale or accidentally swapped files, not adversarial modifications
to a whole store. No parser, compiler, source lookup or binary loading is used.
"""
import hashlib
import json
import re

from grid_codegen.model_descriptor import canonical_bytes, validate_descriptor


def check_expected_id(value):
    if value is not None and (type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value)):
        raise ValueError('expected_descriptor_id must be a lowercase SHA256 hex string')


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_descriptor(entry):
    return validate_descriptor(json.loads((entry/'model_descriptor.json').read_text()))


def _header_string(header, name):
    # Read only the emitter's finite literal grammar, never evaluate C++ or
    # invoke a generator on a prepared load. Escapes are three octal digits.
    pattern = (r'^\s*inline constexpr char ' + name +
               r'\[\] = "((?:[^"\\\n]|\\[0-7]{3})*)";[ \t]*$')
    values = re.findall(pattern, header, re.MULTILINE)
    if len(values) != 1:
        raise ValueError('missing or ambiguous generated model literal: ' + name)
    raw, result, i = values[0], bytearray(), 0
    while i < len(raw):
        if raw[i] == '\\':
            result.append(int(raw[i+1:i+4], 8))
            i += 4
        else:
            result.extend(raw[i].encode('ascii'))
            i += 1
    return result.decode('utf-8')


def _association_snapshot(entry, artifact_name):
    descriptor=_read_descriptor(entry)
    header_bytes=(entry/'grid.cuh').read_bytes()
    header=header_bytes.decode('utf-8')
    embedded=validate_descriptor(json.loads(_header_string(header, 'descriptor_json')))
    if (canonical_bytes(embedded) != canonical_bytes(descriptor) or
            _header_string(header, 'descriptor_id') != descriptor['descriptor_id']):
        raise ValueError('model descriptor disagrees with generated header')
    association=dict(schema=1,descriptor_id=descriptor['descriptor_id'],
                     header_sha256=hashlib.sha256(header_bytes).hexdigest(),
                     artifact_sha256=file_digest(entry/artifact_name))
    return descriptor, association


def record_model_association(entry, artifact_name):
    return _association_snapshot(entry, artifact_name)[1]


def validate_model_association(entry, meta, artifact_name, expected=None):
    """Return validated descriptor or raise ValueError with rebuild guidance."""
    check_expected_id(expected)
    try:
        descriptor, actual=_association_snapshot(entry, artifact_name)
        recorded=meta.get('model_association')
        if canonical_bytes(recorded) != canonical_bytes(actual):
            raise ValueError('descriptor/header/binary association mismatch')
        if expected is not None and descriptor['descriptor_id'] != expected:
            raise ValueError('expected_descriptor_id does not match artifact')
        return descriptor
    except (OSError,ValueError,TypeError,KeyError) as exc:
        raise ValueError(f'model descriptor association invalid; rebuild artifact: {exc}') from exc
