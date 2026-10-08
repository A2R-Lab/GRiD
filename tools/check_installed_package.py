"""Run from outside the source checkout to validate an installed release artifact.

CPU by default. --gpu additionally registers a tiny robot and checks the three
backends; --reload requires its existing cache and forbids recompilation.
Use --package-root only for overlaying an unpacked wheel on an existing framework
environment; every GRiD/peer import is asserted to come from that wheel.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

URDF = '''<robot name="package_probe"><link name="base"/>
<link name="arm"><inertial><origin xyz="0.3 0 0"/><mass value="1"/>
<inertia ixx="0.1" iyy="0.2" izz="0.2" ixy="0" ixz="0" iyz="0"/></inertial></link>
<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>
<axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="10"/></joint></robot>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--package-root', type=Path)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()
    if args.reload and not args.gpu:
        parser.error('--reload requires --gpu')
    if args.package_root:
        sys.path.insert(0, str(args.package_root.resolve()))
    import grid_codegen
    import grid_rbd
    import URDFParser
    import RBDReference
    from grid_codegen import resources
    from grid_codegen.helpers._lin_alg_helpers import _glass_commit
    from grid_rbd import _cache
    import numpy as np
    root = Path(grid_codegen.__file__).resolve().parent.parent
    assert resources.checkout_root() is None, 'This checks installed artifacts, not editable installs'
    for module in (grid_rbd, URDFParser, RBDReference):
        assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
    if args.package_root:
        assert root == args.package_root.resolve()
    provenance = resources.bundled_provenance()
    assert provenance and _glass_commit() == provenance['peers']['GLASS']['commit']
    for name, expected in provenance['resources'].items():
        assert hashlib.sha256((resources.data_dir() / name).read_bytes()).hexdigest() == expected, name
    for name, expected in provenance['python_sources'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    assert _cache._glass_content_hash() and _cache._codegen_source_hash()
    assert (resources.collision_include_dir() / 'grid_collision_geometry.cuh').is_file()
    from grid_codegen.launch_config import load_launch_config
    assert load_launch_config('iiwa14', False)
    args.work.mkdir(parents=True, exist_ok=True)
    urdf = args.work / 'probe.urdf'
    urdf.write_text(URDF)
    robot = URDFParser.URDFParser().parse(str(urdf))
    if not args.reload:
        grid_codegen.GRiDCodeGenerator(robot).gen_all_code(
            algorithm_list=['inverse_dynamics', 'inverse_dynamics_gradient'],
            output_path=str(args.work / 'grid.cuh'))
    if args.gpu:
        import jax
        import jax.numpy as jnp
        import torch
        cache = args.work / 'cache'
        if args.reload:
            # Any cache miss is a failure, not an unnoticed warmup/compile.
            def forbidden(*a, **k):
                raise AssertionError('Reload attempted generation or compilation')
            grid_rbd.generate_sources = forbidden
            grid_rbd.compile_sources = forbidden
            grid_codegen.GRiDCodeGenerator.gen_all_code = forbidden
        options = dict(urdf_path=str(urdf), cache_dir=str(cache),
                       algorithm_list=['inverse_dynamics', 'inverse_dynamics_gradient'],
                       enable_mujoco_kernels=False)
        q = np.array([[.2], [.7]], dtype=np.float32)
        qd = np.array([[.1], [-.2]], dtype=np.float32)
        qdd = np.array([[.3], [-.1]], dtype=np.float32)
        ref = RBDReference.RBDReference(robot)
        expected = np.stack([ref.inverse_dynamics(a, b, c)[0] for a,b,c in zip(q,qd,qdd)]).reshape(2,1)
        handles = {b: grid_rbd.register_robot('package_probe', backend=b, **options)
                   for b in ('numpy', 'jax', 'torch')}
        value = handles['numpy'].inverse_dynamics(q, qd, qdd)
        np.testing.assert_allclose(value, expected, rtol=2e-5, atol=2e-6)
        jvalue = handles['jax'].inverse_dynamics(jnp.asarray(q), jnp.asarray(qd), jnp.asarray(qdd))
        targs = [torch.tensor(a, device='cuda', requires_grad=True) for a in (q, qd, qdd)]
        tvalue = handles['torch'].inverse_dynamics(*targs)
        np.testing.assert_allclose(np.asarray(jvalue), value, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(tvalue.detach().cpu().numpy(), value, rtol=2e-5, atol=2e-6)
        jgrad = jax.grad(lambda x: handles['jax'].inverse_dynamics(x, jnp.asarray(qd), jnp.asarray(qdd)).sum())(jnp.asarray(q))
        tvalue.sum().backward()
        np.testing.assert_allclose(np.asarray(jgrad), targs[0].grad.cpu().numpy(), rtol=2e-5, atol=2e-6)
        step = 1e-5
        oracle_grad = np.stack([(ref.inverse_dynamics(a.astype(float)+step, b, c)[0]
                                - ref.inverse_dynamics(a.astype(float)-step, b, c)[0])/(2*step)
                               for a,b,c in zip(q,qd,qdd)]).reshape(2,1)
        np.testing.assert_allclose(np.asarray(jgrad), oracle_grad, rtol=2e-4, atol=2e-5)
        print('PASS installed NumPy/JAX/PyTorch values, autograd parity,', 'warm reload' if args.reload else 'registration')
    print(json.dumps({'result': 'PASS', 'version': grid_rbd.__version__, 'root': str(root),
                      'bundled_peers': provenance['peers'], 'resource_count': len(provenance['resources'])}, indent=2))


if __name__ == '__main__':
    main()
