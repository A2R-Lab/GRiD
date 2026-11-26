"""The public coordinate converters (grid_rbd.conventions) against Pinocchio.

Pinocchio is the judge of its own convention: a GRiD state is converted, moved
by Pinocchio's integrate, and compared with the converted moved GRiD state.
That fixes the configuration map, the velocity frame and the joint pairing at
once, with no reference to GRiD's dynamics.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from grid_assets import robot_urdf  # noqa: E402
from grid_rbd import conventions, describe_model  # noqa: E402
from test.python_wrappers import planar_root_model  # noqa: E402


def _describe(path, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return describe_model(str(path), **options)


def _case(name, tmp_path_factory):
    if name == "planar":
        path, option = planar_root_model.write_planar_root_model(
            tmp_path_factory.mktemp("conventions_planar"), "option", mimic=False)
        return _describe(path, planar_base=option), pin.buildModelFromUrdf(str(path), pin.JointModelPlanar())
    robot, base = name.split("-")
    path = robot_urdf(robot)
    if base == "floating":
        return _describe(path, floating_base=True), pin.buildModelFromUrdf(str(path), pin.JointModelFreeFlyer())
    return _describe(path), pin.buildModelFromUrdf(str(path))


CASES = ["iiwa14-fixed", "gen3-fixed", "fetch-fixed", "fr3-fixed", "go2-floating", "gen3-floating", "planar"]


@pytest.fixture(scope="module", params=CASES)
def case(request, tmp_path_factory):
    descriptor, pin_model = _case(request.param, tmp_path_factory)
    return request.param, descriptor, pin_model, conventions.PinocchioMap(descriptor, pin_model)


def _state(descriptor, seed):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1.0, 1.0, descriptor["nq"])
    for joint in descriptor["joints"]:
        if joint["kind"] in ("floating", "spherical") and joint["q_start"] is not None:
            start = joint["q_start"] + (3 if joint["kind"] == "floating" else 0)
            q[start:start + 4] /= np.linalg.norm(q[start:start + 4])
    return q, rng.uniform(-1.0, 1.0, descriptor["nv"]), rng.uniform(-1.0, 1.0, descriptor["nv"])


def _grid_step(descriptor, q, v, dt):
    """GRiD's configuration after moving along v for dt, by Pinocchio-free rules:
    scalar coordinates add, and the base moves by the documented right increment."""
    out = q + 0.0
    floating = descriptor["floating_base"]
    if floating:
        model = pin.Model()
        model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "root")
        out[:7] = pin.integrate(model, q[:7], v[:6] * dt)
        out[7:] = q[7:] + v[6:] * dt
    else:
        out = q + v * dt
    return out


@pytest.mark.parametrize("seed", range(3))
def test_a_grid_step_is_the_pinocchio_step(case, seed):
    name, descriptor, pin_model, mapping = case
    q, v, _ = _state(descriptor, seed)
    q_pin, v_pin = mapping.q_to_pinocchio(q), mapping.v_to_pinocchio(q, v)
    assert q_pin.shape == (pin_model.nq,) and v_pin.shape == (pin_model.nv,)
    dt = 1e-6
    moved = mapping.q_to_pinocchio(_grid_step(descriptor, q, v, dt))
    expected = pin.integrate(pin_model, q_pin, v_pin * dt)
    # First order in dt: the two agree to O(dt^2).
    np.testing.assert_allclose(moved, expected, rtol=0.0, atol=5e-11, err_msg=name)
    assert np.linalg.norm(moved - q_pin) > 1e-8


@pytest.mark.parametrize("seed", range(3))
def test_round_trips(case, seed):
    name, descriptor, pin_model, mapping = case
    q, v, a = _state(descriptor, seed)
    np.testing.assert_allclose(mapping.q_from_pinocchio(mapping.q_to_pinocchio(q)), q, atol=1e-12)
    np.testing.assert_allclose(mapping.v_from_pinocchio(q, mapping.v_to_pinocchio(q, v)), v, atol=1e-12)
    np.testing.assert_allclose(
        mapping.a_from_pinocchio(q, v, mapping.a_to_pinocchio(q, v, a)), a, atol=1e-12)
    if not mapping.has_mimic:
        tau = a
        np.testing.assert_allclose(
            mapping.tau_from_pinocchio(q, mapping.tau_to_pinocchio(q, tau)), tau, atol=1e-12)
        # Power is the same number in both conventions.
        assert np.isclose(mapping.tau_to_pinocchio(q, tau) @ mapping.v_to_pinocchio(q, v), tau @ v)


@pytest.mark.parametrize("seed", range(2))
def test_acceleration_and_jacobian_match_finite_differences(case, seed):
    name, descriptor, pin_model, mapping = case
    q, v, a = _state(descriptor, seed)
    h = 1e-6
    ahead = mapping.v_to_pinocchio(_grid_step(descriptor, q, v, h), v + a * h)
    behind = mapping.v_to_pinocchio(_grid_step(descriptor, q, v, -h), v - a * h)
    np.testing.assert_allclose(mapping.a_to_pinocchio(q, v, a), (ahead - behind) / (2 * h), atol=1e-7)

    jacobian = mapping.configuration_jacobian(q)
    numeric = np.zeros_like(jacobian)
    for k in range(descriptor["nq"]):
        e = np.zeros(descriptor["nq"])
        e[k] = h
        numeric[:, k] = (mapping.q_to_pinocchio(q + e) - mapping.q_to_pinocchio(q - e)) / (2 * h)
    np.testing.assert_allclose(jacobian, numeric, atol=1e-8)


def test_the_cases_cover_each_difference(tmp_path_factory):
    wraps = lambda m: any(j.nq == 2 and j.nv == 1 for j in m.joints)
    _, gen3 = _case("gen3-fixed", tmp_path_factory)
    assert wraps(gen3), "gen3 no longer has a continuous joint; pick another robot for that case"
    descriptor, fr3 = _case("fr3-fixed", tmp_path_factory)
    mapping = conventions.PinocchioMap(descriptor, fr3)
    assert mapping.has_mimic and fr3.nv == descriptor["nv"] + 1
    q, v, _ = _state(descriptor, 0)
    # The follower's Pinocchio slot carries the leader's value and rate.
    follower = next(j for j in descriptor["joints"] if j["mimic"] is not None)
    leader = descriptor["joints"][follower["mimic"]["leader_joint"]]
    slot = fr3.joints[fr3.getJointId(follower["name"])]
    assert mapping.q_to_pinocchio(q)[slot.idx_q] == pytest.approx(
        follower["mimic"]["multiplier"] * q[leader["q_start"]] + follower["mimic"]["offset"])
    with pytest.raises(ValueError, match="mimic"):
        mapping.tau_to_pinocchio(q, v)
    # A force on the follower lands on the leader.
    tau_pin = np.zeros(fr3.nv)
    tau_pin[slot.idx_v] = 2.0
    assert mapping.tau_from_pinocchio(q, tau_pin)[leader["v_start"]] == pytest.approx(
        2.0 * follower["mimic"]["multiplier"])


def test_models_of_different_robots_are_refused(tmp_path_factory):
    descriptor, _ = _case("iiwa14-fixed", tmp_path_factory)
    _, other = _case("go2-floating", tmp_path_factory)
    with pytest.raises(ValueError):
        conventions.PinocchioMap(descriptor, other)
    floating, _ = _case("go2-floating", tmp_path_factory)
    _, fixed = _case("go2-fixed", tmp_path_factory)
    with pytest.raises(ValueError, match="floating base"):
        conventions.PinocchioMap(floating, fixed)


def test_mujoco_floating_base_maps():
    rng = np.random.default_rng(5)
    q = rng.uniform(-1.0, 1.0, 10)
    q[3:7] /= np.linalg.norm(q[3:7])
    v = rng.uniform(-1.0, 1.0, 9)
    q_mj = conventions.q_to_mujoco(q)
    assert q_mj[3] == q[6] and np.array_equal(q_mj[4:7], q[3:6]) and np.array_equal(q_mj[7:], q[7:])
    np.testing.assert_array_equal(conventions.q_from_mujoco(q_mj), q)
    v_mj = conventions.v_to_mujoco(q, v)
    rotation = pin.Quaternion(q[6], q[3], q[4], q[5]).toRotationMatrix()
    np.testing.assert_allclose(v_mj[:3], rotation @ v[:3], atol=1e-12)
    np.testing.assert_array_equal(v_mj[3:], v[3:])
    np.testing.assert_allclose(conventions.v_from_mujoco(q, v_mj), v, atol=1e-12)
    # World-frame linear velocity is the time derivative of the position.
    model = pin.Model()
    model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "root")
    dt = 1e-6
    moved = pin.integrate(model, q[:7], v[:6] * dt)
    np.testing.assert_allclose((moved[:3] - q[:3]) / dt, v_mj[:3], atol=1e-5)
    # A fixed base needs no conversion.
    np.testing.assert_array_equal(conventions.q_to_mujoco(q, floating_base=False), q)
    np.testing.assert_array_equal(conventions.v_to_mujoco(q, v, floating_base=False), v)
