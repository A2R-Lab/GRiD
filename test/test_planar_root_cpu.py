"""CPU checks for a planar root: the planar_base option and its oracle.

The fixture (test/python_wrappers/planar_root_model.py) is a mobile manipulator with a
continuous wheel, limited arm joints and a mimic finger. Its planar root comes
from a planar joint declared in the URDF or from the planar_base parse option;
the two must be the same model. No GPU and no compiler are used here: the
generated header is only read as text.

The Pinocchio comparison uses a copy without the mimic and Pinocchio's own
JointModelPlanar root. Pinocchio stores the root as [x, y, cos, sin] with the
velocity [vx, vy, omega] in the moving joint frame, while the decomposed chain
uses [px, py, theta] with the linear velocity in the parent frame. With
G = blockdiag(Rz(theta)^T, 1, ..., 1) the two velocities satisfy
v_pin = G v, hence a_pin = G a + dG/dt v, and power invariance gives
tau = G^T tau_pin and M = G^T M_pin G. The test maps inputs with these
relations, runs Pinocchio's rnea, crba and aba, and maps the outputs back.
"""
import contextlib
import io
import os
import re
import sys
import warnings

import numpy as np
import pinocchio as pin
import pytest

import grid_rbd
from grid_codegen import GRiDCodeGenerator
from grid_codegen.model_descriptor import planar_base_block, validate_descriptor
from grid_rbd._model import describe_model, parse_model
from RBDReference import RBDReference
from URDFParser import URDFParser
from URDFParser.errors import URDFParseError
from test.python_wrappers import planar_root_model as model

VARIANTS = ("declared", "option")


def _quiet(function, *args, **kwargs):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    directory = tmp_path_factory.mktemp("planar_root_cpu")
    return {variant: model.write_planar_root_model(directory, variant)
            for variant in ("declared", "option", "fixed")}


@pytest.fixture(scope="module")
def descriptors(paths):
    return {variant: _quiet(describe_model, path, planar_base=option)
            for variant, (path, option) in paths.items()}


@pytest.fixture(scope="module")
def oracle(paths):
    path, option = paths["option"]
    return RBDReference(_quiet(parse_model, path, planar_base=option))


# ---- the option is the declared joint ---------------------------------------

def test_option_and_declared_joint_give_one_descriptor(descriptors):
    declared, option = descriptors["declared"], descriptors["option"]
    assert option == declared
    assert option["descriptor_id"] == declared["descriptor_id"]
    assert option["descriptor_id"] != descriptors["fixed"]["descriptor_id"]
    assert [joint["name"] for joint in option["joints"]] == model.JOINT_ORDER
    assert (option["nq"], option["nv"]) == (model.NQ, model.NV)
    assert option["floating_base"] is False
    owners = [joint["name"] for joint in option["joints"] if joint["owns_coordinates"]]
    assert owners == model.COORDINATE_OWNERS
    follower = option["joints"][model.JOINT_ORDER.index("finger_right")]
    assert follower["mimic"] == {"leader_joint": model.JOINT_ORDER.index("finger_left"),
                                 "multiplier": model.FINGER_MULTIPLIER,
                                 "offset": model.FINGER_OFFSET}


def test_option_and_declared_joint_give_one_reference_model(paths, oracle):
    path, option = paths["declared"]
    declared = RBDReference(_quiet(parse_model, path, planar_base=option))
    assert declared.robot.planar_base is False and oracle.robot.planar_base is True
    assert declared.robot.floating_base is False and oracle.robot.floating_base is False
    q, v, u = model.sample_states(4, 5)
    for k in range(4):
        np.testing.assert_array_equal(oracle.crba(q[k]), declared.crba(q[k]))
        np.testing.assert_array_equal(oracle.inverse_dynamics(q[k], v[k], u[k])[0],
                                      declared.inverse_dynamics(q[k], v[k], u[k])[0])
        np.testing.assert_array_equal(
            oracle.end_effector_pose(q[k], ee_joint_names=[model.TOOL_FRAME])[0],
            declared.end_effector_pose(q[k], ee_joint_names=[model.TOOL_FRAME])[0])


def test_planar_base_excludes_floating_base(paths):
    path = paths["fixed"][0]
    with pytest.raises(URDFParseError, match="mutually exclusive"):
        _quiet(URDFParser().parse, str(path), floating_base=True, planar_base=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_model(path, floating_base=True, planar_base=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        describe_model(path, floating_base=True, planar_base=True)
    with pytest.raises(TypeError, match="planar_base must be bool"):
        parse_model(path, planar_base=1)
    for backend in ("numpy", "jax", "torch"):
        with pytest.raises(ValueError, match="mutually exclusive"):
            grid_rbd.register_robot("planar_and_floating", str(path), floating_base=True,
                                    planar_base=True, backend=backend, cuda_arch=120)
    with pytest.raises(ValueError, match="mutually exclusive"):
        grid_rbd.build_plan("planar_and_floating", str(path), floating_base=True,
                            planar_base=True, cuda_arch=120)


def test_planar_base_rekeys_the_build_and_leaves_other_keys_alone(paths, tmp_path):
    path = paths["fixed"][0]
    common = dict(cuda_arch=120, cache_dir=tmp_path / "cache")
    fixed = grid_rbd.build_plan("planar_key_probe", str(path), **common)
    explicit = grid_rbd.build_plan("planar_key_probe", str(path), planar_base=False, **common)
    planar = grid_rbd.build_plan("planar_key_probe", str(path), planar_base=True, **common)
    assert "planar_base" not in fixed["code_options"]
    assert explicit["input_key"] == fixed["input_key"]
    assert planar["code_options"]["planar_base"] is True
    assert planar["input_key"] != fixed["input_key"]
    from grid_rbd._artifacts import _code_options
    artifact = dict(ee_joint_names=None, dtype="float32", algorithm_list=None,
                    enable_mujoco_kernels=True, cost_frames=None, contact_frames=None,
                    use_joint_dynamics=False, friction_model="sign", friction_width=None,
                    runtime_inertia=False, runtime_transform=False, runtime_joint_dynamics=False,
                    emit_plant=None, collision=False, collision_res=0.05, collision_native=False,
                    spherized_urdf=None, strict_collision=True, launch_config_profile=None)
    plain = _code_options(path, 120, floating_base=False, **artifact)
    keyed = _code_options(path, 120, floating_base=False, planar_base=True, **artifact)
    assert "planar_base" not in plain and keyed == {**plain, "planar_base": True}
    with pytest.raises(ValueError, match="mutually exclusive"):
        _code_options(path, 120, floating_base=True, planar_base=True, **artifact)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_planar_base_reaches_the_backend_registration(monkeypatch, backend):
    class Captured(Exception):
        pass

    def stub(*args, **kwargs):
        raise Captured(kwargs)

    module = pytest.importorskip(f"grid_rbd.{backend}")
    monkeypatch.setattr(module, "register_robot", stub)
    with pytest.raises(Captured) as root:
        grid_rbd.register_robot("planar_probe", "/nonexistent/robot.urdf", backend=backend,
                                planar_base=True)
    assert root.value.args[0]["planar_base"] is True
    monkeypatch.undo()
    pytest.importorskip(backend)
    monkeypatch.setattr(grid_rbd, "register_robot", stub)
    with pytest.raises(Captured) as leaf:
        module.register_robot("planar_probe", "/nonexistent/robot.urdf", planar_base=True)
    assert leaf.value.args[0]["planar_base"] is True


def test_load_robot_names_the_planar_base(monkeypatch, paths):
    seen = []
    monkeypatch.setattr(grid_rbd, "register_robot",
                        lambda name, *args, **kwargs: seen.append((name, kwargs)))
    path = str(paths["fixed"][0])
    grid_rbd.load_robot(path)
    grid_rbd.load_robot(path, planar_base=True)
    assert "_fixed_" in seen[0][0] and "_planar_" in seen[1][0]
    assert seen[1][1]["planar_base"] is True and "planar_base" not in seen[0][1]


# ---- the descriptor helper --------------------------------------------------

def test_planar_base_block_recognises_the_chain(descriptors):
    assert planar_base_block(descriptors["declared"]) == (0, 0, 3)
    assert planar_base_block(descriptors["option"]) == (0, 0, 3)
    assert planar_base_block(descriptors["fixed"]) is None


def test_planar_base_block_rejects_other_models(paths, tmp_path):
    import copy
    path = paths["fixed"][0]
    assert planar_base_block(_quiet(describe_model, path, floating_base=True)) is None
    base = _quiet(describe_model, paths["option"][0], planar_base=True)
    for index, field, value in [(0, "name", "planar_joint__sub0"), (1, "kind", "revolute"),
                                (2, "kind", "revolute"), (2, "parent_body", 0),
                                (1, "q_start", 2)]:
        edited = copy.deepcopy(base)
        edited["joints"][index][field] = value
        assert planar_base_block(edited) is None, (index, field)
    assert planar_base_block({"joints": base["joints"][:2]}) is None
    assert planar_base_block(None) is None
    # A planar joint beside the rest of the tree is not a planar base, whatever its name.
    text = model.urdf_text(declared_planar_root=False).replace(
        '<joint name="wheel_spin" type="continuous">',
        '<joint name="planar_base_joint" type="planar">')
    inner = tmp_path / "inner_planar.urdf"
    inner.write_text(text)
    described = _quiet(describe_model, inner)
    assert "planar_base_joint__sub0" in [joint["name"] for joint in described["joints"]]
    assert planar_base_block(described) is None


def test_descriptor_limits_of_unbounded_coordinates(descriptors):
    joints = {joint["name"]: joint for joint in validate_descriptor(descriptors["option"])["joints"]}
    assert joints["planar_base_joint__sub0"]["limits"]["position"] == [None, None]
    assert joints["planar_base_joint__sub1"]["limits"]["position"] == [None, None]
    assert joints["planar_base_joint__sub2"]["limits"]["position"] is None
    assert joints["wheel_spin"]["limits"]["position"] is None
    assert joints["shoulder"]["limits"]["position"] == [-1.2, 1.5]
    assert joints["torso_lift"]["limits"]["position"] == [0.0, 0.35]


# ---- the generated header ---------------------------------------------------

def _header(path, option, directory):
    robot = _quiet(URDFParser().parse, str(path), planar_base=option)
    out = directory / "grid.cuh"
    generator = GRiDCodeGenerator(robot, FILE_NAMESPACE="grid")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        generator.gen_all_code(codegen_profile="kinematics", output_path=str(out),
                               fixed_target_name=model.TOOL_FRAME)
    return out.read_text()


@pytest.fixture(scope="module")
def headers(paths, tmp_path_factory):
    directory = tmp_path_factory.mktemp("planar_root_headers")
    result = {}
    for variant, (path, option) in paths.items():
        (directory / variant).mkdir()
        result[variant] = _header(path, option, directory / variant)
    return result


def _limit_table(header):
    start = header.index("init_joint_limits_checked(T **out")
    body = header[start:header.index("d_joint_limits", start)]
    entries = dict((int(index), value.strip()) for index, value in
                   re.findall(r"h_joint_limits\[(\d+)\] = static_cast<T>\(([^)]*)\);", body))
    count = len(entries) // 2
    assert sorted(entries) == list(range(2 * count))
    return [(entries[i], entries[i + count]) for i in range(count)]


def test_option_and_declared_joint_give_one_header(headers):
    assert headers["option"] == headers["declared"]
    assert headers["option"] != headers["fixed"]


def test_metadata_marks_only_the_planar_base(headers):
    assert headers["option"].count("inline constexpr bool planar_base = true;") == 1
    assert headers["fixed"].count("inline constexpr bool planar_base = false;") == 1
    assert "planar_base = true" not in headers["fixed"]


def test_generated_limit_table(headers):
    expected = [("-INFINITY" if lower is None else repr(float(lower)),
                 "INFINITY" if upper is None else repr(float(upper)))
                for lower, upper in model.POSITION_LIMITS]
    table = _limit_table(headers["option"])
    assert len(table) == model.NQ
    assert table == expected
    # The same body without the base keeps its limits three slots earlier.
    assert _limit_table(headers["fixed"]) == expected[3:]


# ---- oracle gradients against its own finite differences --------------------

def _central(function, x, step=1e-6):
    columns = []
    for k in range(len(x)):
        e = np.zeros(len(x))
        e[k] = step
        columns.append((np.asarray(function(x + e), float) - np.asarray(function(x - e), float))
                       / (2 * step))
    return np.stack(columns, axis=-1)


@pytest.mark.parametrize("sample", range(6))
def test_oracle_gradients_match_central_differences(oracle, sample):
    q, v, u = [x[sample] for x in model.sample_states(6, 11)]
    nv = model.NV
    scale = lambda x: max(1.0, float(np.abs(x).max()))

    tau = lambda q_, v_: oracle.inverse_dynamics(q_, v_, u)[0]
    analytic = np.asarray(oracle.inverse_dynamics_gradient(q, v, u), float)
    numeric = np.hstack([_central(lambda x: tau(x, v), q), _central(lambda x: tau(q, x), v)])
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-6 * scale(numeric))

    dq, dv = oracle.forward_dynamics_gradient(q, v, u)
    numeric_q = _central(lambda x: oracle.forward_dynamics(x, v, u), q)
    numeric_v = _central(lambda x: oracle.forward_dynamics(q, x, u), v)
    np.testing.assert_allclose(dq, numeric_q, rtol=1e-6, atol=1e-6 * scale(numeric_q))
    np.testing.assert_allclose(dv, numeric_v, rtol=1e-6, atol=1e-6 * scale(numeric_v))

    # Away from pitch = +-pi/2 the rpy rows are smooth; unwrap the yaw so a
    # +-pi crossing between the two stencil points is not differenced.
    pose = lambda x: np.asarray(
        oracle.end_effector_pose(x, ee_joint_names=[model.TOOL_FRAME])[0], float).reshape(6)
    jacobian = np.asarray(
        oracle.end_effector_pose_gradient(q, ee_joint_names=[model.TOOL_FRAME])[0], float)
    assert jacobian.shape == (6, nv)
    step = 1e-6
    numeric = np.zeros((6, nv))
    for k in range(nv):
        e = np.zeros(nv)
        e[k] = step
        delta = pose(q + e) - pose(q - e)
        delta[3:] = (delta[3:] + np.pi) % (2 * np.pi) - np.pi
        numeric[:, k] = delta / (2 * step)
    assert abs(abs(pose(q)[4]) - np.pi / 2) > 0.05
    np.testing.assert_allclose(jacobian, numeric, rtol=1e-6, atol=1e-6 * scale(numeric))


# ---- oracle values against Pinocchio ----------------------------------------

@pytest.fixture(scope="module")
def free_finger(tmp_path_factory):
    """Mimic-free copy: the oracle with the planar base option, and Pinocchio
    with its own planar root joint, plus the coordinate maps between them."""
    directory = tmp_path_factory.mktemp("planar_root_pinocchio")
    path, option = model.write_planar_root_model(directory, "option", mimic=False)
    reference = RBDReference(_quiet(parse_model, path, planar_base=option))
    pin_model = pin.buildModelFromUrdf(str(path), pin.JointModelPlanar())
    names = [joint.get_name() for joint in reference.robot.get_joints_ordered_by_id()]
    assert names[:3] == model.JOINT_ORDER[:3]
    assert reference.robot.get_num_vel() == pin_model.nv == model.NV + 1
    q_index, v_index, circular = [], [], []
    for k, name in enumerate(names[3:], start=3):
        joint = pin_model.joints[pin_model.getJointId(name)]
        assert joint.nv == 1 and joint.nq in (1, 2), name
        q_index.append(joint.idx_q)
        v_index.append(joint.idx_v)
        circular.append(joint.nq == 2)
    root = pin_model.joints[1]
    assert (root.nq, root.nv, root.idx_q, root.idx_v) == (4, 3, 0, 0)
    assert circular[names[3:].index("wheel_spin")]
    return reference, pin_model, q_index, v_index, circular


def _to_pinocchio(mapping, q, v):
    reference, pin_model, q_index, v_index, circular = mapping
    nv = pin_model.nv
    c, s = np.cos(q[2]), np.sin(q[2])
    q_pin = np.zeros(pin_model.nq)
    q_pin[:4] = [q[0], q[1], c, s]
    # P places chain velocity slot k at Pinocchio's slot; G rotates the base pair.
    P = np.zeros((nv, nv))
    P[:3, :3] = [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
    P_dot = np.zeros((nv, nv))
    P_dot[:2, :2] = v[2] * np.array([[-s, c], [-c, -s]])
    for k, (iq, iv, wraps) in enumerate(zip(q_index, v_index, circular), start=3):
        q_pin[iq:iq + (2 if wraps else 1)] = [np.cos(q[k]), np.sin(q[k])] if wraps else [q[k]]
        P[iv, k] = 1.0
    return q_pin, P, P_dot


@pytest.mark.parametrize("sample", range(6))
def test_oracle_matches_pinocchio_planar_root(free_finger, sample):
    reference, pin_model = free_finger[:2]
    nv = pin_model.nv
    rng = np.random.default_rng(100 + sample)
    q8, v8, _ = [x[sample] for x in model.sample_states(6, 23)]
    # insert the freed second finger after finger_left
    q = np.insert(q8, 7, rng.uniform(-0.04, 0.002))
    v = np.insert(v8, 7, rng.uniform(-0.8, 0.8))
    a = rng.uniform(-2.0, 2.0, nv)
    q_pin, G, G_dot = _to_pinocchio(free_finger, q, v)
    data = pin_model.createData()
    assert np.allclose(G.T @ G, np.eye(nv))

    mass = pin.crba(pin_model, data, q_pin)
    mass = np.triu(mass) + np.triu(mass, 1).T
    np.testing.assert_allclose(reference.crba(q), G.T @ mass @ G, rtol=1e-9, atol=1e-9)

    tau_pin = pin.rnea(pin_model, data, q_pin, G @ v, G @ a + G_dot @ v)
    tau = reference.inverse_dynamics(q, v, a)[0]
    np.testing.assert_allclose(tau, G.T @ tau_pin, rtol=1e-9, atol=1e-9)
    # Power is the same number in both velocity coordinates.
    assert np.isclose(tau @ v, tau_pin @ (G @ v), rtol=1e-9, atol=1e-9)

    u = rng.uniform(-3.0, 3.0, nv)
    a_pin = pin.aba(pin_model, data, q_pin, G @ v, G @ u)   # G^-T = G for orthogonal G
    np.testing.assert_allclose(reference.forward_dynamics(q, v, u),
                               G.T @ (a_pin - G_dot @ v), rtol=1e-8, atol=1e-8)
