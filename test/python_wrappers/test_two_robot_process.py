"""Two robot .so's in ONE process (debugging-guide §7.z4 regression).

Inline-function statics emitted into grid.cuh compile to weak default-
visibility symbols; without hidden visibility the dynamic linker unified the
device-pool state across dlopened robot .so's, so the SECOND robot's init
carved from the first robot's exhausted slab and "OOM"d on an empty GPU.
This registers two different iiwa14 builds (different max_batch => different
.so) through the jax and torch surfaces in one process and checks that each
carved its OWN pool and computes correctly.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")
_jax = pytest.importorskip("jax", reason="jax not installed")
_torch = pytest.importorskip("torch", reason="torch not installed")

_URDF = (
    Path.home()
    / ".cache/robot_descriptions/drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
)
if not _URDF.exists():
    pytest.skip(f"iiwa14 URDF fixture not present at {_URDF}", allow_module_level=True)
if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; register_robot requires it", allow_module_level=True)

pytestmark = pytest.mark.python_wrappers


def test_two_so_pools_are_independent():
    import grid_rbd.jax as gj
    import grid_rbd.torch as gt

    # same names/keys as the jax + torch smokes => warm cache hits
    jh = gj.register_robot(name="iiwa14_jax_pytest", urdf_path=str(_URDF),
                           floating_base=False, max_batch_size=8)
    q = np.zeros((2, jh.num_joints), np.float32)
    tau_j = np.asarray(jh.inverse_dynamics(q, q))
    used_a = int(jh._base._runner.device_pool_used())

    th = gt.register_robot(name="iiwa14_torch_smoke", urdf_path=str(_URDF),
                           floating_base=False, max_batch_size=64)
    tq = _torch.zeros((2, th.num_joints), device="cuda")
    tau_t = th.inverse_dynamics(tq, tq).cpu().numpy()
    used_b = int(th._base._runner.device_pool_used())

    assert np.allclose(tau_j, tau_t, atol=1e-6)
    # each .so carved its OWN slab (pool active on both; §7.z4 failure mode was
    # robot B binding to robot A's pool and failing init outright)
    assert used_a > 0 and used_b > 0
    # different max_batch => different arena sizes => provably not one shared pool
    assert used_a != used_b
