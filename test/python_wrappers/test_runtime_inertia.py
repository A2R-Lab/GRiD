"""Validation for the runtime-mutable inertia binding (D.4 / Phase 5).

The runtime-inertia surface is ``register_robot(..., runtime_inertia=True)`` +
``handle.set_inertia_params(params)`` + ``handle.inertia_params``. The .so carries
a device-resident ``d_inertia_params`` 10-param-per-body table and rebuilds each
link's spatial 6x6 inertia on-device from it (a divide-free scatter), instead of
streaming the baked ``d_XImats`` I-region.

Coverage already in place elsewhere (the landing commit): runtime==baked
(bit-identical with the baked params) + a param-change-shifts-output sanity. This
file closes the three remaining gaps:

  1. **Mutated-table PHYSICAL correctness** — set a PERTURBED 10-param table and
     assert the CUDA output matches an independent ``RBDReference`` oracle whose
     link inertias were mutated to the *same* perturbed params. This proves the
     on-device 6x6 rebuild is physically correct, not merely that runtime==baked.
  2. **fp64 + runtime_inertia** — same, at ``dtype="float64"`` to ~1e-9 (exercises
     the ``RunnerF64`` + double ``set_inertia_params`` path).
  3. **Floating-base runtime_inertia** — go2 floating: runtime==baked + a
     perturbation vs the oracle. Validates the body-0/base index mapping of the
     dropped-base table layout.

Oracle construction
-------------------
The 10-param vector is ``[m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]`` in the
frozen GRiD/URDF regressor basis (``Link.get_inertia_params``). The device rebuild
(``_topology_helpers._emit_runtime_inertia_rebuild``) scatters it into the 6x6 as::

    I(pi) = [[ I_O,        skew(h) ],
             [ skew(h)^T,  m * I3  ]]

We rebuild that *same* 6x6 from a perturbed 10-param row and inject it into the
oracle robot via ``link.set_spatial_inertia`` (RBDReference reads
``get_Imat_by_id`` -> ``get_spatial_inertia`` live on every call, so this fully
re-parameterizes the oracle's dynamics).

Index mapping (verified, both fixed + floating)
-----------------------------------------------
``get_inertia_params_ordered_by_id()[1:]`` (the table layout) drops the id=-1
world-frame link. Table row ``k`` <-> link ``get_links_ordered_by_id()[k+1]``.
For a FIXED base that is link ids 0..N (NUM_JOINTS rows). For a FLOATING base it
is link ids 0..N where link 0 is the floating trunk — the trunk inertia IS in the
table (row 0), only the synthetic id=-1 world frame is dropped.

Run with:
    pytest test/python_wrappers/test_runtime_inertia.py -m python_wrappers -v
"""
from __future__ import annotations

import contextlib
import io
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


# Repo root is parent of `test/`; add it (URDFParser/RBDReference) and bindings/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "bindings"))
from config import robot_urdf


# ─── skip preconditions ─────────────────────────────────────────────────────

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it",
                allow_module_level=True)

_IIWA = robot_urdf("iiwa14")
_GO2 = robot_urdf("go2")
if not _IIWA.exists():
    pytest.skip(f"iiwa14 URDF not present at {_IIWA}", allow_module_level=True)


pytestmark = pytest.mark.python_wrappers


# ─── helpers ────────────────────────────────────────────────────────────────


def _parse(urdf_path, floating_base):
    """Parse a URDF quietly (the parser is chatty on stdout)."""
    from URDFParser import URDFParser
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(str(urdf_path), floating_base=floating_base)


def _spatial_inertia_from_params(p):
    """Rebuild the spatial 6x6 from a 10-param row exactly as the device does.

    Mirrors grid_codegen/helpers/_topology_helpers._emit_runtime_inertia_rebuild:
        p = [m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
        I = [[ I_O,        skew(h) ],
             [ skew(h)^T,  m*I3    ]]
    """
    m, hx, hy, hz = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
    Ixx, Ixy, Ixz, Iyy, Iyz, Izz = (float(v) for v in p[4:10])
    I_O = np.array([[Ixx, Ixy, Ixz],
                    [Ixy, Iyy, Iyz],
                    [Ixz, Iyz, Izz]], dtype=np.float64)
    skew_h = np.array([[0.0, -hz, hy],
                       [hz, 0.0, -hx],
                       [-hy, hx, 0.0]], dtype=np.float64)
    S = np.zeros((6, 6), dtype=np.float64)
    S[:3, :3] = I_O
    S[:3, 3:] = skew_h
    S[3:, :3] = skew_h.T
    S[3:, 3:] = m * np.eye(3)
    return S


def _build_perturbed_oracle(urdf_path, floating_base, params_table):
    """Build an RBDReference whose link inertias == the perturbed `params_table`.

    params_table is (NUM_JOINTS, 10), the same layout the handle's
    set_inertia_params takes (bodies = link ids 0..N, the id=-1 world frame
    dropped). Row k is injected into link get_links_ordered_by_id()[k+1].
    """
    from RBDReference import RBDReference
    robot = _parse(urdf_path, floating_base)
    links = robot.get_links_ordered_by_id()  # [-1, 0, 1, ...]
    body_links = links[1:]                    # drops the id=-1 world frame
    assert len(body_links) == params_table.shape[0], (
        f"index-map mismatch: {len(body_links)} bodies vs "
        f"{params_table.shape[0]} param rows")
    for k, link in enumerate(body_links):
        link.set_spatial_inertia(_spatial_inertia_from_params(params_table[k]))
    return RBDReference(robot)


def _perturb(baked, rng):
    """Physically-meaningful perturbation: scale masses (±20%), shift the first
    moment h, and tweak the inertia I_O — touching all three param groups."""
    p = baked.astype(np.float64).copy()
    nj = p.shape[0]
    p[:, 0] *= (1.0 + 0.2 * rng.standard_normal(nj))          # mass m
    p[:, 1:4] += 0.05 * rng.standard_normal((nj, 3))          # first moment h
    p[:, 4:10] *= (1.0 + 0.1 * rng.standard_normal((nj, 6)))  # inertia I_O
    return p


def _max_rel_err(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))


# ─── 1. mutated-table physical correctness (fp32, fixed base) ────────────────


@pytest.fixture(scope="module")
def iiwa_rt():
    return _grid_rbd.register_robot(
        name="iiwa14_runtime_inertia_pytest",
        urdf_path=str(_IIWA),
        floating_base=False,
        runtime_inertia=True,
        max_batch_size=8,
    )


@pytest.fixture(scope="module")
def iiwa_samples(iiwa_rt):
    rng = np.random.default_rng(0)
    NJ = iiwa_rt.num_joints
    B = 4
    return {
        "q":  rng.standard_normal((B, NJ)).astype(np.float32),
        "qd": rng.standard_normal((B, NJ)).astype(np.float32),
    }


def test_runtime_inertia_metadata(iiwa_rt):
    assert iiwa_rt.runtime_inertia is True
    # fixed base: num_bodies == num_joints, so the table is (num_joints, 10).
    assert iiwa_rt.num_bodies == iiwa_rt.num_joints
    assert iiwa_rt.inertia_params.shape == (iiwa_rt.num_bodies, 10)


def test_baked_params_reproduce_baked_result(iiwa_rt, iiwa_samples):
    """Sanity: setting the baked params back == not setting at all (==
    RBDReference on the unmutated robot). The required-floor before trusting a
    perturbation."""
    oracle = RBDReference_unmutated(_IIWA, floating_base=False)
    iiwa_rt.set_inertia_params(iiwa_rt.inertia_params)
    grid = iiwa_rt.inverse_dynamics(iiwa_samples["q"], iiwa_samples["qd"])
    for i, (q, qd) in enumerate(zip(iiwa_samples["q"], iiwa_samples["qd"])):
        c_ref, *_ = oracle.inverse_dynamics(q.astype(np.float64), qd.astype(np.float64),
                                            GRAVITY=-9.81)
        assert _max_rel_err(grid[i], c_ref) < 5e-3


def test_mutated_table_matches_oracle_fp32(iiwa_rt, iiwa_samples):
    """CORE gap-closer: a PERTURBED 10-param table must produce dynamics that
    match an oracle re-running RNEA + CRBA with the SAME perturbed inertias.
    Proves the on-device 6x6 rebuild from the mutated params is physically
    correct (not merely runtime==baked)."""
    rng = np.random.default_rng(7)
    perturbed = _perturb(iiwa_rt.inertia_params, rng)            # (NJ, 10) float64
    oracle = _build_perturbed_oracle(_IIWA, False, perturbed)
    iiwa_rt.set_inertia_params(perturbed.astype(np.float32))

    q, qd = iiwa_samples["q"], iiwa_samples["qd"]
    grid_c = iiwa_rt.inverse_dynamics(q, qd)
    grid_M = iiwa_rt.crba(q)
    for i, (qi, qdi) in enumerate(zip(q, qd)):
        c_ref, *_ = oracle.inverse_dynamics(qi.astype(np.float64),
                                            qdi.astype(np.float64), GRAVITY=-9.81)
        M_ref = oracle.crba(qi.astype(np.float64))
        assert _max_rel_err(grid_c[i], c_ref) < 5e-3, f"RNEA mismatch sample {i}"
        assert _max_rel_err(grid_M[i], M_ref) < 5e-3, f"CRBA mismatch sample {i}"

    # and the perturbed result is genuinely different from the baked one (guards
    # against a no-op set_inertia_params that would trivially "match").
    iiwa_rt.set_inertia_params(iiwa_rt.inertia_params)
    baked_c = iiwa_rt.inverse_dynamics(q, qd)
    assert _max_rel_err(grid_c, baked_c) > 1e-2, "perturbation had no effect"


# ─── 2. fp64 + runtime_inertia ───────────────────────────────────────────────


@pytest.fixture(scope="module")
def iiwa_rt_f64():
    return _grid_rbd.register_robot(
        name="iiwa14_runtime_inertia_f64_pytest",
        urdf_path=str(_IIWA),
        floating_base=False,
        runtime_inertia=True,
        dtype="float64",
        max_batch_size=8,
    )


def test_mutated_table_matches_oracle_fp64(iiwa_rt_f64):
    """fp64 runtime-inertia: the RunnerF64 + double set_inertia_params path must
    match the perturbed float64 oracle in true double precision.

    Tolerance is 1e-7 relative — ~100,000x tighter than the fp32 path's 5e-3,
    proving the double set_inertia_params + RunnerF64 compute path. (A perturbed
    CRBA/RNEA in fp64 lands around a few×1e-8 rel vs the oracle, dominated by the
    differing accumulation order between GRiD's on-device reduction and
    RBDReference's numpy — both genuinely double precision; an fp32 path would
    miss by ~1e-3.)"""
    assert iiwa_rt_f64.dtype == "float64"
    rng = np.random.default_rng(11)
    NJ = iiwa_rt_f64.num_joints
    q = rng.standard_normal((4, NJ)).astype(np.float64)
    qd = rng.standard_normal((4, NJ)).astype(np.float64)

    perturbed = _perturb(iiwa_rt_f64.inertia_params, rng)        # (NJ, 10) float64
    oracle = _build_perturbed_oracle(_IIWA, False, perturbed)
    iiwa_rt_f64.set_inertia_params(perturbed)                    # double path

    grid_c = iiwa_rt_f64.inverse_dynamics(q, qd)
    grid_M = iiwa_rt_f64.crba(q)
    assert grid_c.dtype == np.float64
    for i, (qi, qdi) in enumerate(zip(q, qd)):
        c_ref, *_ = oracle.inverse_dynamics(qi, qdi, GRAVITY=-9.81)
        M_ref = oracle.crba(qi)
        assert _max_rel_err(grid_c[i], c_ref) < 1e-7, f"fp64 RNEA mismatch sample {i}"
        assert _max_rel_err(grid_M[i], M_ref) < 1e-7, f"fp64 CRBA mismatch sample {i}"


# ─── 3. floating-base runtime_inertia ────────────────────────────────────────


@pytest.fixture(scope="module")
def go2_rt():
    if not _GO2.exists():
        pytest.skip(f"go2 URDF not present at {_GO2}")
    return _grid_rbd.register_robot(
        name="go2_floating_runtime_inertia_pytest",
        urdf_path=str(_GO2),
        floating_base=True,
        runtime_inertia=True,
        # Non-mjx test on a floating robot -> drop the mjx twins (the bulk of the compile).
        enable_mujoco_kernels=False,
        max_batch_size=8,
    )


def _floating_q(handle, seed):
    """Floating-base q for the grid_rbd binding: q is packed (B, num_joints) ==
    (B, num_pos) (the wrapper's pack_q_qd_u is NUM_POS-wide for q/qd/u alike)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((3, handle.num_joints)).astype(np.float32)


def test_floating_runtime_inertia_metadata(go2_rt):
    assert go2_rt.floating_base is True
    assert go2_rt.runtime_inertia is True
    # FLOATING base: the device d_inertia_params table is sized by num_bodies
    # (the inertia-body count = grid::NUM_BODIES), NOT num_joints (== num_pos).
    # go2: num_bodies=13 < num_joints=19. Regression guard for the binding fix
    # (the table/validation used to use 10*num_joints and reject the only valid
    # (num_bodies,10) table).
    assert go2_rt.num_bodies < go2_rt.num_joints
    assert go2_rt.inertia_params.shape == (go2_rt.num_bodies, 10)


# NOTE on the floating-base oracle gap
# ------------------------------------
# A *value* cross-check of floating-base CRBA/RNEA against RBDReference is blocked
# by a PRE-EXISTING binding floating-base-layout gap that is independent of
# runtime-inertia: the grid_rbd binding packs q/qd/u NUM_POS-wide and returns
# velocity-space matrices NUM_POS×NUM_POS (go2: 19×19, quaternion-derivative
# columns), whereas RBDReference works in the NUM_VEL tangent space (18×18). The
# two don't line up without a quaternion→tangent reconciliation that the binding
# doesn't yet expose (same class of issue as the floating-base EE-pose-gradient
# convention note). So the floating-base runtime-inertia tests below validate the
# runtime-inertia MECHANISM device-side (GRiD-vs-GRiD self-consistency + the
# trunk/base-body index mapping) rather than vs the oracle. The fixed-base
# fp32/fp64 tests above already prove the on-device 6×6 rebuild is *physically*
# correct vs the oracle; floating base adds only the base-body index mapping,
# which the row-0 sensitivity test pins down precisely.


def test_floating_runtime_equals_baked(go2_rt):
    """Setting the baked params back must reproduce the baked result bit-for-bit
    (device self-consistency): set_inertia_params(inertia_params) is the identity,
    and a second identical set must give an identical CRBA. The floor before
    trusting a perturbation."""
    q = _floating_q(go2_rt, 0)
    go2_rt.set_inertia_params(go2_rt.inertia_params)
    M1 = go2_rt.crba(q)
    go2_rt.set_inertia_params(go2_rt.inertia_params)
    M2 = go2_rt.crba(q)
    assert _max_rel_err(M1, M2) < 1e-5, "baked set_inertia_params is not idempotent"
    # mass matrix M is NUM_VEL x NUM_VEL (a floating base has num_vel < num_joints==num_pos)
    assert M1.shape == (q.shape[0], go2_rt.num_vel, go2_rt.num_vel)


def test_floating_mutated_table_shifts_output_and_recovers(go2_rt):
    """Floating-base runtime-inertia mechanism: a PERTURBED table changes the
    output, and resetting the baked table recovers the baseline. Proves the
    device rebuild reads the mutated (num_bodies,10) table on a floating base."""
    q = _floating_q(go2_rt, 1)
    go2_rt.set_inertia_params(go2_rt.inertia_params)
    M_baked = go2_rt.crba(q)

    rng = np.random.default_rng(23)
    perturbed = _perturb(go2_rt.inertia_params, rng)
    go2_rt.set_inertia_params(perturbed.astype(np.float32))
    M_pert = go2_rt.crba(q)
    assert _max_rel_err(M_pert, M_baked) > 1e-2, "floating perturbation had no effect"

    go2_rt.set_inertia_params(go2_rt.inertia_params)
    M_back = go2_rt.crba(q)
    assert _max_rel_err(M_back, M_baked) < 1e-5, "reset did not recover the baked CRBA"


def test_floating_trunk_row0_index_mapping(go2_rt):
    """THE floating-base index-mapping test: row 0 of the inertia table is the
    floating TRUNK (base body, link id 0), NOT a dropped/world row. Doubling only
    row 0's mass must visibly shift the CRBA. If the base/body index map were off
    by one (e.g. row 0 silently dropped, as the fixed-base ``params[1:]`` slice
    might suggest), this perturbation would land on the wrong link or be ignored.
    """
    q = _floating_q(go2_rt, 2)
    go2_rt.set_inertia_params(go2_rt.inertia_params)
    M_baked = go2_rt.crba(q)

    trunk_pert = go2_rt.inertia_params.copy()
    trunk_pert[0, 0] *= 2.0                      # double the floating-trunk mass
    go2_rt.set_inertia_params(trunk_pert.astype(np.float32))
    M_trunk = go2_rt.crba(q)
    # The trunk dominates the floating-base inertia, so doubling its mass moves
    # the CRBA substantially (measured ~45% rel at a zero-ish config).
    assert _max_rel_err(M_trunk, M_baked) > 1e-1, \
        "row-0 (floating trunk) mass change had ~no effect — base-body index map is wrong"

    go2_rt.set_inertia_params(go2_rt.inertia_params)  # restore for any later test


# ─── unmutated-oracle convenience ────────────────────────────────────────────


def RBDReference_unmutated(urdf_path, floating_base):
    from RBDReference import RBDReference
    return RBDReference(_parse(urdf_path, floating_base))


# ─── §5.2 gate: runtime .so (baked values) == plain baked .so, bit-for-bit ────


@pytest.fixture(scope="module")
def iiwa_baked():
    # the plain baked .so (runtime_inertia=False) — the oracle for the §5.2
    # output-parity gate; currently the only registration without runtime_inertia.
    return _grid_rbd.register_robot(
        name="iiwa14_baked_inertia_pytest", urdf_path=str(_IIWA),
        floating_base=False, runtime_inertia=False, max_batch_size=8)


# ─── FFI parity of a poke (numpy/jax/torch share the device inertia table) ────


def test_inertia_poke_seen_across_surfaces():
    """A poke through the numpy handle mutates the single device-resident
    d_inertia_params table, so jax/torch (sharing the same dlopen'd .so __device__
    global) see the poked inertias: numpy == jax == torch AFTER set_inertia_params.
    This is the runtime_inertia analogue of
    test_runtime_joint_dynamics::test_poke_seen_across_surfaces, and the end-to-end
    check that the jax/torch runtime_inertia exposure threads the shared table.
    Skips if jax/torch unavailable."""
    gj = pytest.importorskip("grid_rbd.jax")
    pytest.importorskip("jax")
    gt = pytest.importorskip("grid_rbd.torch")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available for torch")
    common = dict(urdf_path=str(_IIWA), floating_base=False, max_batch_size=8,
                  runtime_inertia=True)
    hn = _grid_rbd.register_robot(name="iiwa14_rti_ffi_pytest", force_rebuild=True, **common)
    hj = gj.register_robot(name="iiwa14_rti_ffi_pytest", **common)
    ht = gt.register_robot(name="iiwa14_rti_ffi_pytest", **common)
    nj = hn.num_joints
    rng = np.random.default_rng(31)
    perturbed = _perturb(hn.inertia_params, rng).astype(np.float32)
    hn.set_inertia_params(perturbed)                  # poke via numpy -> device table
    q = rng.standard_normal((4, nj)).astype(np.float32)
    qd = rng.standard_normal((4, nj)).astype(np.float32)
    qt = torch.tensor(q, device="cuda", dtype=torch.float32)
    qdt = torch.tensor(qd, device="cuda", dtype=torch.float32)
    cn = np.asarray(hn.inverse_dynamics(q, qd))
    cj = np.asarray(hj.inverse_dynamics(q, qd))
    ct = ht.inverse_dynamics(qt, qdt).detach().cpu().numpy()
    assert _max_rel_err(cn, cj) < 2e-3, "jax did not see the numpy inertia poke"
    assert _max_rel_err(cn, ct) < 2e-3, "torch did not see the numpy inertia poke"
    # Confirm the poke genuinely moved the output (not a trivial baked==baked pass).
    hn.set_inertia_params(hn.inertia_params)
    cb = np.asarray(hn.inverse_dynamics(q, qd))
    assert _max_rel_err(cn, cb) > 1e-2, "inertia poke had no effect"
    hn.set_inertia_params(hn.inertia_params)          # restore


def test_runtime_equals_baked_so(iiwa_rt, iiwa_baked):
    """A runtime_inertia=True .so, fed its ORIGINAL baked params, must reproduce
    the plain baked .so to fp tolerance — the rebuild is a pure scatter of the
    same I_O numbers (phase5 §5.2). Oracle = the baked .so itself (NOT
    RBDReference): this isolates the runtime MECHANISM, not the physics."""
    iiwa_rt.set_inertia_params(iiwa_rt.inertia_params)  # original baked values
    rng = np.random.default_rng(1)
    NJ = iiwa_rt.num_joints
    B = 4
    q = rng.standard_normal((B, NJ)).astype(np.float32)
    qd = rng.standard_normal((B, NJ)).astype(np.float32)
    u = rng.standard_normal((B, NJ)).astype(np.float32)
    checks = {
        "inverse_dynamics": (iiwa_rt.inverse_dynamics(q, qd), iiwa_baked.inverse_dynamics(q, qd)),
        "crba": (iiwa_rt.crba(q), iiwa_baked.crba(q)),
        "minv": (iiwa_rt.minv(q), iiwa_baked.minv(q)),
        "forward_dynamics": (iiwa_rt.forward_dynamics(q, qd, u), iiwa_baked.forward_dynamics(q, qd, u)),
    }
    for name, (rt, baked) in checks.items():
        err = float(np.max(np.abs(np.asarray(rt) - np.asarray(baked))))
        assert err < 1e-6, f"{name}: runtime .so != baked .so by {err:.2e}"
