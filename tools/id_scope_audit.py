"""Read-only Python-emitter inventory, not a generated-CUDA reachability proof."""
import argparse
import ast
import json
from pathlib import Path

ROOTS = ('gen_inverse_dynamics_inner', 'gen_inverse_dynamics_device')
SCOPED_CALLS = {'gen_add_sync': 'block-barrier',
                'gen_add_serial_ops': 'block-leader',
                'gen_add_parallel_loop': 'block-stride',
                'gen_device_wrapper': 'device-scratch-wrapper'}
TOKENS = {'__shared__': 'block-shared-storage', '__syncthreads': 'block-barrier',
          'threadIdx': 'block-rank', 'blockDim': 'block-size',
          'grid_linalg_': 'linear-algebra-helper', 'glass::': 'glass-call'}


def audit(root):
    root = Path(root).resolve()
    symbols = {}
    for path in sorted((root / 'grid_codegen').rglob('*.py')):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('source symlinks need explicit review')
        module = ast.parse(path.read_text(), filename=str(path))
        # Only module-level emitters. Class methods / dynamic registration are
        # unresolved rather than assigned a guessed definition.
        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.setdefault(node.name, []).append((path, node))
    pending, seen, rows, unresolved = list(ROOTS), set(), [], set()
    while pending:
        name = pending.pop()
        if name in seen: continue
        seen.add(name)
        choices = symbols.get(name, [])
        if len(choices) != 1:
            unresolved.add(name + (': ambiguous' if choices else ': not a unique module emitter'))
            continue
        path, function = choices[0]
        calls, findings = set(), []
        # Walk body statements, excluding the function's documentation literal.
        body = function.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        for statement in body:
            for node in ast.walk(statement):
                if isinstance(node, ast.Call):
                    f = node.func
                    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == 'self':
                        calls.add(f.attr)
                        if f.attr in SCOPED_CALLS:
                            findings.append(dict(line=node.lineno, kind=SCOPED_CALLS[f.attr], token=f.attr))
                    elif isinstance(f, ast.Name) and f.id in {'getattr', 'setattr'}:
                        unresolved.add(f'{name}: dynamic attribute access at {node.lineno}')
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for token, kind in TOKENS.items():
                        if token in node.value:
                            findings.append(dict(line=node.lineno, kind=kind, token=token))
        pending.extend(calls - seen)
        rows.append(dict(emitter=name, path=path.relative_to(root).as_posix(),
                         line=function.lineno, self_calls=sorted(calls), findings=findings))
    return dict(mode='report-only', scope_safe=False, roots=list(ROOTS),
                emitters=sorted(rows, key=lambda x: (x['path'], x['line'])),
                unresolved=sorted(unresolved), limitations=[
                    'Only direct self calls to unique module-level Python functions are followed.',
                    'Generated C++ calls, imports, dynamic dispatch and constant-built names need manual review.',
                    'Literal matches may be comments or inactive branches; counts are not runtime barrier counts.',
                    'An empty finding list is not a proof of thread/warp safety.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2))


if __name__ == '__main__':
    main()
