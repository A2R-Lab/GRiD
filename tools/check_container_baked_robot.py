"""Smoke the image's precompiled iiwa14 on this GPU without rebuilding it."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-version', required=True)
    args = parser.parse_args()
    import numpy as np
    import grid_rbd
    import URDFParser
    import RBDReference

    assert grid_rbd.__version__ == args.expected_version, grid_rbd.__version__
    assert Path('/opt/grid/iiwa14/grid.cuh').stat().st_size > 0
    entries = grid_rbd.list_registered()
    print(json.dumps(entries, indent=2))
    entry = next(e for e in entries if e['name'] == 'iiwa14')
    assert set(map(int, entry['cuda_archs'])) == {86, 89, 90, 120}

    def forbidden(*args, **kwargs):
        raise AssertionError('Baked robot attempted generation or compilation')

    grid_rbd.generate_sources = forbidden
    grid_rbd.compile_sources = forbidden
    robot = URDFParser.URDFParser().parse('/opt/grid/robots/iiwa14.urdf')
    reference = RBDReference.RBDReference(robot)
    q = np.array([[.1, -.2, .3, -.4, .5, -.6, .7],
                  [-.3, .2, -.1, .6, -.5, .4, -.7]], dtype=np.float32)
    qd = q * np.float32(.2)
    qdd = q * np.float32(-.3)
    expected = np.stack([reference.inverse_dynamics(a, b, c)[0]
                         for a, b, c in zip(q, qd, qdd)]).reshape(2, 7)
    import jax
    import jax.numpy as jnp
    import torch
    assert torch.cuda.is_available()
    assert any(d.platform == 'gpu' for d in jax.devices()), jax.devices()
    print(json.dumps({'grid_rbd': grid_rbd.__version__, 'numpy': np.__version__,
                      'jax': jax.__version__, 'torch': torch.__version__}))
    for backend in ('numpy', 'jax', 'torch'):
        handle = grid_rbd.get_robot('iiwa14', backend=backend)
        inputs = (q, qd, qdd)
        if backend == 'jax':
            inputs = tuple(jnp.asarray(a) for a in inputs)
        elif backend == 'torch':
            inputs = tuple(torch.tensor(a, device='cuda') for a in inputs)
        actual = handle.inverse_dynamics(*inputs)
        if backend == 'torch':
            actual = actual.detach().cpu().numpy()
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=5e-4, atol=5e-4)
        print('PASS baked iiwa14 inverse dynamics vs CPU oracle:', backend, flush=True)


if __name__ == '__main__':
    main()
