"""The permanent release pair must include source-free armature identification."""
import ast
from pathlib import Path

from tools import check_installed_package as checker


def test_release_pair_calls_identification_gate():
    source = Path(checker.__file__).read_text()
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == 'check_identification' for n in ast.walk(main))
    body = ast.get_source_segment(source, next(n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == 'check_identification'))
    for contract in ('assert not source.exists()', 'URDFParser.URDFParser.parse = forbidden',
                     'compiler.find_nvcc = compiler.compile_so = compiler.compile_sources = forbidden',
                     'h.set_armature([.8])', "source.rename(work / 'source-hidden.urdf')",
                     "np.asarray([.4], np.float32)", 'ref.forward_dynamics_armature_gradient',
                     'handle.armature_regressor'):
        assert contract in body
    root = Path(__file__).parents[1]
    assert 'tools/check_installed_package.py' in (root/'test/gpu-proof-policy-release.yaml').read_text()
    assert '--gpu --reload' in (root/'RELEASING.md').read_text()
