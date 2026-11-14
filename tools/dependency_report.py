"""CPU-only ID dependency prototype. Reports hypotheses; NEVER authorizes reuse.

Tracks whole source families conservatively, rather than pretending a Python
emitter call graph resolves generated C++ templates, macros or dispatch.
No imports from the candidate, compiler calls, cache writes or receipt changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SCHEMA = 1
EXPORTS = ('inverse_dynamics_inner', 'inverse_dynamics_inner_vaf')
CONTEXT_KEYS = {'architecture', 'compiler', 'host_compiler', 'flags', 'abi',
                'generation_options', 'model'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def classify(path):
    """Conservative prototype coverage, NOT a kernel-level dependency proof."""
    if path.startswith(('test/', 'install/')) or path == 'docs/test_doc_examples.py':
        return 'evidence'
    if path.startswith(('grid_codegen/', 'bindings/', 'grid_assets/', 'external/')):
        return 'object'
    if path in {'pyproject.toml', 'setup.py', 'setup.cfg', 'CMakeLists.txt', 'Makefile'}:
        return 'object'
    if path.startswith('docs/') or path in {'README.md', 'LICENSE', 'CLAUDE.md',
                                           'CONTRIBUTING.md', 'RELEASING.md'}:
        return 'ignored-documentation'
    return 'unknown'


def closure(nodes, entry):
    """Validate all graph edges; cycles are valid shared dependency components."""
    for name, node in nodes.items():
        if (not isinstance(name, str) or not isinstance(node, dict)
                or not isinstance(node.get('deps'), list)
                or any(not isinstance(dep, str) for dep in node['deps'])):
            raise ValueError(f'invalid node {name}')
        for dep in node['deps']:
            if dep not in nodes:
                raise ValueError(f'unresolved dependency: {name} -> {dep}')
    seen, pending = set(), [entry]
    while pending:
        name = pending.pop()
        if name not in nodes:
            raise ValueError(f'missing entry {name}')
        if name not in seen:
            seen.add(name)
            pending.extend(nodes[name]['deps'])
    return sorted(seen)


def key(nodes, entry):
    return digest({'schema': SCHEMA, 'nodes': {n: nodes[n] for n in closure(nodes, entry)}})


def make_snapshot(files, context, *, missing=(), provenance=None):
    """Pure graph builder used by the deliberate-mutation suite.

    files maps repo-relative paths to {digest, mode}; missing inputs cannot be
    silently dropped. Unclassified files widen both identities. Metadata for
    the exported ID variants is explicit; their generated closure is unresolved.
    """
    if not isinstance(context, dict) or set(context) != CONTEXT_KEYS:
        raise ValueError(f'context needs exactly {sorted(CONTEXT_KEYS)}')
    if any(value is None or value == '' for value in context.values()):
        raise ValueError('context values must be explicit, not missing')
    digest(context)  # reject non-JSON values / NaN
    nodes, objects, evidence, unknown, ignored = {}, [], [], [], []
    for path, value in sorted(files.items()):
        if Path(path).is_absolute() or '..' in Path(path).parts:
            raise ValueError(f'unsafe source path {path}')
        category = classify(path)
        if category == 'ignored-documentation':
            ignored.append(path)
            continue
        if not isinstance(value, dict) or set(value) != {'digest', 'mode'}:
            raise ValueError(f'invalid source identity {path}')
        name = 'file:' + path
        nodes[name] = dict(kind=category, identity=value, deps=[], provenance='source-bytes-or-gitlink')
        if category == 'evidence': evidence.append(name)
        else: objects.append(name)
        if category == 'unknown': unknown.append(path)
    nodes['context'] = dict(kind='object', identity=context, deps=[], provenance='caller-declared-not-measured')
    object_roots, evidence_roots = {}, {}
    for export in EXPORTS:
        obj, ev = 'object:' + export, 'evidence:' + export
        nodes[obj] = dict(kind='export', identity={'symbol': export, 'scope': 'block'},
                          deps=sorted(objects + ['context']), provenance='whole-source-fallback')
        nodes[ev] = dict(kind='acceptance', identity={'symbol': export},
                         deps=sorted(evidence + [obj]), provenance='whole-evidence-fallback')
        object_roots[export], evidence_roots[export] = key(nodes, obj), key(nodes, ev)
    missing = sorted(set(missing))
    return dict(schema=SCHEMA, mode='report-only', reuse_authorized=False, narrowing_ready=False,
                complete=not missing, missing=missing, unclassified=unknown, ignored=ignored,
                unresolved=['Generated C++ calls/templates/macros are not resolved; whole-source fallback.',
                            'Test/fixture/oracle closure is not narrowed; whole-evidence fallback.',
                            'Context is caller-declared; compiler/device identity is not measured.',
                            'Gitlinks record commits, not peer working-tree contents.'],
                provenance=provenance or {}, nodes=nodes,
                proposed_object_keys=object_roots, proposed_evidence_keys=evidence_roots)


def snapshot(root, context):
    root = Path(root).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    tracked = git('ls-files', '--stage', '-z').decode().split('\0')
    files, missing = {}, []
    for line in filter(None, tracked):
        metadata, path = line.split('\t', 1)
        mode, oid, stage = metadata.split()
        if stage != '0':
            raise ValueError('unmerged index cannot be analyzed')
        if classify(path) == 'ignored-documentation':
            files[path] = dict(digest='ignored', mode=mode)
            continue
        target = root / path
        if mode == '160000':
            files[path] = dict(digest=oid, mode=mode)
        elif target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(root):
            missing.append(path)  # no following symlinks into a peer/user directory
        else:
            live_mode = '100755' if target.stat().st_mode & 0o111 else '100644'
            files[path] = dict(digest=hashlib.sha256(target.read_bytes()).hexdigest(), mode=live_mode)
    # New sources are not invisible: report them as unresolved until staged.
    new = git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')
    missing += ['untracked:' + p for p in new if p and classify(p) != 'ignored-documentation']
    return make_snapshot(files, context, missing=missing,
                         provenance={'commit': git('rev-parse', 'HEAD').decode().strip()})


def compare(before, after):
    """Recompute keys, rather than trust cached JSON key fields."""
    for data in (before, after):
        if not isinstance(data, dict) or data.get('schema') != SCHEMA or data.get('mode') != 'report-only':
            raise ValueError('incompatible snapshot schema/mode; invalidate broadly')
        if (not isinstance(data.get('nodes'), dict) or type(data.get('complete')) is not bool
                or not isinstance(data.get('missing'), list)
                or any(not isinstance(p, str) for p in data['missing'])
                or data['complete'] != (not data['missing'])):
            raise ValueError('invalid snapshot completeness; invalidate broadly')
    changed = sorted(n for n in set(before['nodes']) | set(after['nodes'])
                     if before['nodes'].get(n) != after['nodes'].get(n))
    broad = not before['complete'] or not after['complete']
    exports = {}
    for export in EXPORTS:
        objects = key(before['nodes'], 'object:' + export) != key(after['nodes'], 'object:' + export)
        evidence = key(before['nodes'], 'evidence:' + export) != key(after['nodes'], 'evidence:' + export)
        reach = set(closure(before['nodes'], 'evidence:' + export)) | set(closure(after['nodes'], 'evidence:' + export))
        exports[export] = dict(object_changed=broad or objects, evidence_changed=broad or evidence,
                               reasons=[n for n in changed if n in reach])
    return dict(schema=SCHEMA, mode='report-only', reuse_authorized=False, narrowing_ready=False,
                invalidate_broadly=broad, missing=sorted(set(before['missing'] + after['missing'])),
                exports=exports, changed_nodes=changed,
                warning='Unchanged proposed keys do not authorize reuse; generated closure remains unresolved.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    snap = actions.add_parser('snapshot')
    snap.add_argument('--root', type=Path, required=True)
    snap.add_argument('--context', type=Path, required=True)
    diff = actions.add_parser('compare')
    diff.add_argument('before', type=Path)
    diff.add_argument('after', type=Path)
    args = parser.parse_args()
    try:
        if args.action == 'snapshot':
            result = snapshot(args.root, json.loads(args.context.read_text()))
        else:
            result = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f'Unresolved input; invalidate broadly: {exc}\n')
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
