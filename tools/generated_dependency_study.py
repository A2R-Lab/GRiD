"""CPU header-regeneration mutation study; report-only, never authorizes reuse.

Runs trusted local generator source in private copied fixtures. Deliberately
incorrect mutations are generated as text, NEVER compiled or run on a GPU.
Whole-header and ID-definition byte hashes are observations, not compiler
dependency closures. Macros, templates and out-of-line helpers remain unresolved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

try:
    from . import dependency_report as dep
except ImportError:
    import dependency_report as dep


def sha(data):
    return hashlib.sha256(data).hexdigest()


def mask_cpp(text):
    """Keep offsets while masking ordinary strings/comments. Reject raw strings."""
    if re.search(r'\bR"', text):
        raise ValueError('raw strings require a real C++ parser')
    pattern = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    return re.sub(pattern, lambda m: ''.join('\n' if c == '\n' else ' ' for c in m[0]), text)


def definition_inventory(text):
    """Explicit ID root/overload metadata; not transitive dependency inference."""
    masked = mask_cpp(text)
    result = {}
    for name in dep.EXPORTS:
        definitions = []
        for match in re.finditer(r'\bvoid\s+' + re.escape(name) + r'\s*\(', masked):
            start = match.start(); pos = match.end(); parens = 1
            while pos < len(masked) and parens:
                parens += (masked[pos] == '(') - (masked[pos] == ')'); pos += 1
            if parens: raise ValueError('unbalanced signature')
            while pos < len(masked) and masked[pos].isspace(): pos += 1
            if pos == len(masked) or masked[pos] != '{': continue
            body = pos; braces = 1; pos += 1
            while pos < len(masked) and braces:
                braces += (masked[pos] == '{') - (masked[pos] == '}'); pos += 1
            if braces: raise ValueError('unbalanced function')
            definitions.append(dict(signature=' '.join(text[start:body].split()),
                                    start_line=text[:start].count('\n') + 1,
                                    end_line=text[:pos].count('\n') + 1,
                                    sha256=sha(text[start:pos].encode())))
        if not definitions: raise ValueError(f'missing definition: {name}')
        result[name] = definitions
    return result


def mutate_function(text, function, old, new):
    import ast
    tree = ast.parse(text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    lines = text.splitlines(keepends=True)
    body = ''.join(lines[fn.lineno - 1:fn.end_lineno])
    if body.count(old) != 1: raise ValueError('mutation anchor must be unique')
    return ''.join(lines[:fn.lineno - 1]) + body.replace(old, new) + ''.join(lines[fn.end_lineno:])


ID = 'grid_codegen/algorithms/_inverse_dynamics.py'
HELPER = 'grid_codegen/helpers/_code_generation_helpers.py'
FD = 'grid_codegen/algorithms/_forward_dynamics.py'
MODEL = 'grid_assets/robots/iiwa14.urdf'


def fixtures(source):
    """Mechanically mutated test copies only. No checkout file is modified."""
    emit = 'self.gen_add_code_line(func_def, True)'
    variants = [dict(name='baseline', edits={}, expected_header=False, expected_roots=False)]
    for name, path, fn, old, new, expected in (
        ('id_leaf', ID, 'gen_inverse_dynamics_inner', emit,
         emit + '\n    self.gen_add_code_line("s_vaf[0] += static_cast<T>(0.125);")', True),
        ('shared_sync', HELPER, 'gen_add_sync', '"__syncthreads();"', '"__syncwarp();"', True),
        ('unrelated_fd', FD, 'gen_forward_dynamics_finish', emit,
         emit + '\n    self.gen_add_code_line("s_qdd[0] += static_cast<T>(0.125);")', False),
    ):
        changed = mutate_function((source / path).read_text(), fn, old, new)
        variants.append(dict(name=name, edits={path: changed.encode()}, expected_header=expected, expected_roots=expected))
    helper = 'grid_codegen/helpers/_spatial_algebra_helpers.py'
    changed = mutate_function((source / helper).read_text(), 'gen_crm_mul',
                              'return glass::spatial_detail::motion_cross_mul_row<T>',
                              'return static_cast<T>(0.125) + glass::spatial_detail::motion_cross_mul_row<T>')
    variants.append(dict(name='out_of_line_helper', edits={helper: changed.encode()},
                         expected_header=True, expected_roots=False))
    model = (source / MODEL).read_text()
    old = '<mass value="5.76"/>'
    if model.count(old) != 1: raise ValueError('model mutation anchor changed')
    variants.append(dict(name='model_mass', edits={MODEL: model.replace(old, '<mass value="5.77"/>').encode()},
                         expected_header=True, expected_roots=False))
    variants.append(dict(name='runtime_inertia', edits={}, runtime=True, expected_header=True, expected_roots=None))
    for name, path in [('evidence_policy', 'test/gpu-proof-policy.yaml'), ('docs_only', 'README.md')]:
        variants.append(dict(name=name, edits={path: (source / path).read_bytes() + b'\n# dependency-study mutation\n'},
                             expected_header=False, expected_roots=False))
    variants.append(dict(name='baseline_repeat', edits={}, expected_header=False, expected_roots=False))
    return variants


def generate(source, output, runtime):
    sys.path.insert(0, str(source))
    from URDFParser import URDFParser
    import grid_codegen
    if not Path(grid_codegen.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError('wrong generator imported')
    robot = URDFParser().parse(str(source / MODEL), floating_base=False)
    generator = grid_codegen.GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE='grid')
    generator.gen_all_code(output_path=str(output), algorithm_list=['inverse_dynamics'],
                           enable_mujoco_kernels=False, runtime_inertia=runtime)


def run(source, peers, out):
    source = source.resolve(); peers = peers.resolve(); out = out.resolve()
    if out.is_relative_to(source): raise ValueError('output must be outside the source checkout')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--short'], text=True).strip():
        raise ValueError('source checkout must be clean for the controlled fixture study')
    for peer in ('URDFParser', 'RBDReference', 'GLASS'):
        if not (peers / peer).is_dir(): raise ValueError(f'missing peer {peer}')
    out.mkdir(parents=True, exist_ok=False)
    entries = subprocess.check_output(['git', '-C', str(source), 'ls-files', '--stage', '-z']).decode().split('\0')
    contents, files = {}, {}
    for line in filter(None, entries):
        metadata, path = line.split('\t', 1); mode, oid, stage = metadata.split()
        if stage != '0': raise ValueError('unmerged source index')
        if mode == '160000':
            files[path] = dict(digest=oid, mode=mode); continue
        p = source / path
        if p.is_symlink() or not p.is_file(): raise ValueError(f'unresolved source {path}')
        contents[path] = p.read_bytes()
        files[path] = dict(digest=sha(contents[path]), mode=mode)
    peer_state = {}
    for peer in ('URDFParser', 'RBDReference', 'GLASS'):
        def git(*args): return subprocess.check_output(['git', '-C', str(peers / peer), *args], text=True).strip()
        status = git('status', '--short')
        if status: raise ValueError(f'dirty peer {peer}: no unrecorded inputs')
        head = git('rev-parse', 'HEAD')
        if files['external/' + peer]['digest'] != head: raise ValueError(f'peer pin mismatch: {peer}')
        peer_state[peer] = head
    context = dict(architecture='UNMEASURED', compiler='NOT_RUN', host_compiler='NOT_RUN', flags=[],
                   abi='unchanged-baseline', generation_options=dict(algorithm_list=['inverse_dynamics'],
                   enable_mujoco_kernels=False, runtime_inertia=False), model=MODEL)
    provenance = dict(source=str(source), source_head=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                      tool_sha256=sha(Path(__file__).read_bytes()), analyzer_sha256=sha(Path(dep.__file__).read_bytes()),
                      peers=peer_state, compiler_run=False, gpu_run=False)
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    baseline = None; rows = []
    for variant in fixtures(source):
        directory = out / variant['name']; directory.mkdir()
        copied = directory / 'source'; copied.mkdir()
        (copied / 'pyproject.toml').write_bytes(contents['pyproject.toml'])
        (copied / 'external').mkdir()
        for peer in ('URDFParser', 'RBDReference', 'GLASS'):
            # Read-only use of validated, clean pinned peers; no peer edits.
            (copied / 'external' / peer).symlink_to(peers / peer, target_is_directory=True)
        # Fixture materialization: copies only, never source checkout mutation.
        # The generator can read assets/helpers throughout these two packages.
        for path, data in contents.items():
            if path.startswith(('grid_codegen/', 'grid_assets/')):
                p = copied / path; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
        changed_files = copy.deepcopy(files)
        for path, data in variant['edits'].items():
            p = copied / path; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
            changed_files[path]['digest'] = sha(data)
        ctx = copy.deepcopy(context); ctx['generation_options']['runtime_inertia'] = variant.get('runtime', False)
        snapshot = dep.make_snapshot(changed_files, ctx)
        header = directory / 'grid.cuh'
        env = {k: v for k, v in os.environ.items() if not k.startswith('GRID_')}
        env.update(PYTHONPATH=str(copied) + ':' + str(peers), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        command = [sys.executable, str(Path(__file__).resolve()), '--generate', '--source', str(copied), '--out', str(header)]
        if variant.get('runtime'): command.append('--runtime')
        result = subprocess.run(command, cwd=copied, env=env, capture_output=True, text=True)
        (directory / 'generation.log').write_text(result.stdout + result.stderr)
        if result.returncode: raise RuntimeError(f'generation failed: {directory}')
        text = header.read_text(); inventory = definition_inventory(text)
        fingerprint = {k: [v['sha256'] for v in values] for k, values in inventory.items()}
        if baseline is None: baseline = (sha(header.read_bytes()), fingerprint, snapshot)
        report = dep.compare(baseline[2], snapshot)
        row = dict(name=variant['name'], header_changed=sha(header.read_bytes()) != baseline[0],
                   definition_changed={k: fingerprint[k] != baseline[1][k] for k in dep.EXPORTS},
                   inventory=inventory, comparison=report, header_sha256=sha(header.read_bytes()))
        if row['header_changed'] != variant['expected_header']: raise AssertionError(row)
        if variant['expected_roots'] is not None:
            if any(v != variant['expected_roots'] for v in row['definition_changed'].values()): raise AssertionError(row)
        # Whole-header differences require broad object invalidation here even
        # when individual ID definition bytes did not move (e.g. model tables).
        if row['header_changed'] and not all(v['object_changed'] for v in report['exports'].values()):
            raise AssertionError('missed whole-header dependency')
        expected_object = variant['name'] not in {'baseline', 'baseline_repeat', 'evidence_policy', 'docs_only'}
        expected_evidence = expected_object or variant['name'] == 'evidence_policy'
        if any(v['object_changed'] != expected_object or v['evidence_changed'] != expected_evidence
               for v in report['exports'].values()):
            raise AssertionError('object/evidence separation mismatch')
        (directory / 'snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
        (directory / 'result.json').write_text(json.dumps(row, indent=2) + '\n')
        rows.append(row)
        print(variant['name'], 'header', row['header_changed'], 'definitions', row['definition_changed'], flush=True)
    (out / 'results.json').write_text(json.dumps(dict(reuse_authorized=False, narrowing_ready=False, rows=rows), indent=2) + '\n')
    for peer, head in peer_state.items():
        if (subprocess.check_output(['git', '-C', str(peers / peer), 'rev-parse', 'HEAD'], text=True).strip() != head
                or subprocess.check_output(['git', '-C', str(peers / peer), 'status', '--short'], text=True).strip()):
            raise ValueError('peer changed during generation; results must not be accepted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--peers', type=Path)
    parser.add_argument('--generate', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--runtime', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.generate:
        generate(args.source, args.out, args.runtime)
    else:
        if args.peers is None: parser.error('--peers is required')
        run(args.source, args.peers, args.out)


if __name__ == '__main__': main()
