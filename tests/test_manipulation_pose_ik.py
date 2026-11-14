"""Independent native/hand pose-IK checks; no physical manipulation claim."""

from dataclasses import FrozenInstanceError
from math import atan2, cos, pi, sin

import numpy as np
import pytest
from test_manipulation_pose import binary

from phymeo.control.manipulation_diagnostic import solve_native_point_ik
from phymeo.control.manipulation_pose_ik import NativePoseIKResult, solve_native_pose_ik

NAMES = ("px", "py", "pz", "rx", "ry", "rz")


@pytest.fixture
def mj():
    return pytest.importorskip("mujoco")


def rx(angle):
    return np.array(((1, 0, 0), (0, cos(angle), -sin(angle)), (0, sin(angle), cos(angle))))


def ry(angle):
    return np.array(((cos(angle), 0, sin(angle)), (0, 1, 0), (-sin(angle), 0, cos(angle))))


def rz(angle):
    return np.array(((cos(angle), -sin(angle), 0), (sin(angle), cos(angle), 0), (0, 0, 1)))


def model_fixture(mj):
    content = ('<body name="tool" pos=".12 .03 -.02"><geom size=".025" mass=".1"/>'
               '<body pos="0 .05 0"><joint name="finger" type="slide" axis="0 1 0"/>'
               '<geom type="box" size=".02 .01 .03" mass=".04"/></body></body>')
    for index in reversed(range(6)):
        name = NAMES[index]
        axis = ("1 0 0", "0 1 0", "0 0 1")[index % 3]
        kind, interval = ("slide", "-.2 .2") if index < 3 else ("hinge", "-1.5 1.5")
        base = ' pos=".1 -.2 .7" quat=".7071067811865476 .7071067811865476 0 0"' if index == 0 else ""
        content = (f'<body name="body-{name}"{base}><joint name="{name}" type="{kind}" '
                   f'axis="{axis}" limited="true" range="{interval}"/>'
                   '<geom size=".02" mass=".1"/>' + content + '</body>')
    return mj.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody>'
        '<body pos="-1 0 1" quat=".9238795325 0 0 .3826834324"><freejoint name="object-free"/>'
        '<geom size=".03" mass=".2"/></body>' + content +
        '<body pos="1 0 1" quat=".9659258263 .2588190451 0 0"><freejoint name="other-free"/>'
        '<geom type="box" size=".02 .03 .04" mass=".1"/></body></worldbody></mujoco>'
    )


def hinge_fixture(mj, interval="-3.2 3.2"):
    return mj.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody>'
        '<body pos="1 0 1"><freejoint name="object-free"/><geom size=".03" mass=".1"/></body>'
        '<body name="tool" pos=".1 -.2 .7" quat=".7071067811865476 .7071067811865476 0 0">'
        f'<joint name="turn" type="hinge" axis="0 0 1" limited="true" range="{interval}"/>'
        '<geom type="box" size=".02 .03 .04" mass=".1"/></body></worldbody></mujoco>'
    )


def native(mj, model, q, point=(0, 0, 0)):
    data = mj.MjData(model)
    data.qpos[:] = q
    mj.mj_forward(model, data)
    body = model.body("tool").id
    rotation = data.xmat[body].reshape(3, 3).copy()
    return data.xpos[body] + rotation @ np.asarray(point), rotation


def forbidden(*args, **kwargs):
    raise AssertionError("pose IK must not call native forward/inverse/step")


def test_pose_and_local_tcp_match_native_and_hand_chain_with_shuffled_addresses(mj, monkeypatch):
    model = model_fixture(mj)
    seed = model.qpos0.copy()
    seed[model.jnt_qposadr[model.joint("finger").id]] = .012
    expected_q = seed.copy()
    command = (.04, -.03, .05, .35, -.25, .4)
    for name, value in zip(NAMES, command, strict=True):
        expected_q[model.jnt_qposadr[model.joint(name).id]] = value
    local = np.array((.04, -.02, .07))
    target, target_rotation = native(mj, model, expected_q, local)
    base = rx(pi / 2)
    hand_rotation = base @ rx(command[3]) @ ry(command[4]) @ rz(command[5])
    hand_position = np.array((.1, -.2, .7)) + base @ command[:3] + hand_rotation @ (np.array((.12, .03, -.02)) + local)
    np.testing.assert_allclose(target_rotation, hand_rotation, rtol=0, atol=1e-12)
    np.testing.assert_allclose(target, hand_position, rtol=0, atol=1e-12)
    names = ("rz", "py", "rx", "px", "ry", "pz")
    qa = [int(model.jnt_qposadr[model.joint(name).id]) for name in names]
    assert min(qa) >= 7 and qa != sorted(qa)
    before = (binary(mj, model), seed.tobytes(), target.tobytes(), target_rotation.tobytes(), local.tobytes())
    with monkeypatch.context() as guard:
        for name in ("mj_forward", "mj_inverse", "mj_step", "mj_step1", "mj_step2"):
            guard.setattr(mj, name, forbidden)
        result = solve_native_pose_ik(model, target, target_rotation, seed, names, local_point=local,
                                     tolerance_m=1e-7, tolerance_rad=1e-7)
    assert isinstance(result, NativePoseIKResult) and result.converged and 0 < result.iterations <= 100
    actual_position, actual_rotation = native(mj, model, result.configuration, local)
    np.testing.assert_allclose(actual_position, target, rtol=0, atol=1e-7)
    np.testing.assert_allclose(actual_rotation, target_rotation, rtol=0, atol=1e-7)
    np.testing.assert_allclose(result.position, actual_position, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.rotation, actual_rotation, rtol=0, atol=1e-12)
    assert result.position_error_m == pytest.approx(np.linalg.norm(target - actual_position), abs=1e-12)
    assert result.orientation_error_rad <= 1e-7
    np.testing.assert_array_equal(result.robot_configuration, result.configuration[qa])
    unselected = [address for address in range(model.nq) if address not in qa]
    assert result.configuration[unselected].tobytes() == seed[unselected].tobytes()
    assert before == (binary(mj, model), seed.tobytes(), target.tobytes(), target_rotation.tobytes(), local.tobytes())
    for value in (result.configuration, result.robot_configuration, result.position, result.rotation):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        result.status = "converged"


@pytest.mark.parametrize("angle", (0.0, 1e-8, pi - 1e-8, pi), ids=("zero", "near-zero", "near-pi", "pi"))
def test_world_rotation_log_handles_zero_and_pi_without_false_convergence(mj, angle, monkeypatch):
    model = hinge_fixture(mj)
    seed = model.qpos0.copy()
    target, _ = native(mj, model, seed)
    desired = rx(pi / 2) @ rz(angle)
    # This native joint rotates about world -Y, not local +Z: the error must be spatial.
    with monkeypatch.context() as guard:
        for name in ("mj_forward", "mj_inverse", "mj_step", "mj_step1", "mj_step2"):
            guard.setattr(mj, name, forbidden)
        result = solve_native_pose_ik(model, target, desired, seed, ("turn",), local_point=(0, 0, 0),
                                     tolerance_m=1e-10, tolerance_rad=1e-10)
        short = solve_native_pose_ik(model, target, desired, seed, ("turn",), local_point=(0, 0, 0),
                                    tolerance_m=1e-10, tolerance_rad=1e-10, max_iterations=1)
    assert result.converged
    if angle == 0:
        assert result.iterations == 0
    else:
        assert result.iterations > 0
    qa = int(model.jnt_qposadr[model.joint("turn").id])
    error = abs(atan2(sin(angle - result.configuration[qa]), cos(angle - result.configuration[qa])))
    assert error <= 1e-10
    assert result.orientation_error_rad == pytest.approx(error, abs=1e-12)
    actual_position, actual_rotation = native(mj, model, result.configuration)
    np.testing.assert_allclose(actual_position, target, rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual_rotation, desired, rtol=0, atol=1e-10)
    assert result.configuration[:7].tobytes() == seed[:7].tobytes()
    if angle > 3:
        assert not short.converged and short.status == "iteration-limit" and short.iterations == 1
        assert abs(short.configuration[qa]) <= .15 + 1e-12
        assert short.orientation_error_rad == pytest.approx(angle - .15, abs=1e-9)
        assert short.position_error_m < 1e-12


def test_position_only_ik_is_a_negative_control_for_wrong_orientation(mj):
    model = hinge_fixture(mj)
    seed = model.qpos0.copy()
    target, initial_rotation = native(mj, model, seed)
    desired = rx(pi / 2) @ rz(.8)
    point = solve_native_point_ik(model, target, seed, ("turn",), local_point=(0, 0, 0))
    assert point.converged and point.iterations == 0
    _, point_rotation = native(mj, model, point.configuration)
    np.testing.assert_allclose(point_rotation, initial_rotation, rtol=0, atol=1e-12)
    assert np.linalg.norm(point_rotation - desired) > .5
    pose = solve_native_pose_ik(model, target, desired, seed, ("turn",), local_point=(0, 0, 0))
    assert pose.converged and pose.orientation_error_rad <= 5e-3
    _, rotation = native(mj, model, pose.configuration)
    np.testing.assert_allclose(rotation, desired, rtol=0, atol=5e-3)


def test_limits_and_unreachable_residual_remain_honest_and_bounded(mj):
    model = hinge_fixture(mj, "-.2 .2")
    seed = model.qpos0.copy()
    target, _ = native(mj, model, seed)
    qa = int(model.jnt_qposadr[model.joint("turn").id])
    for position in (target, target + (.2, 0, 0)):
        result = solve_native_pose_ik(model, position, rx(pi / 2) @ rz(1), seed, ("turn",), local_point=(0, 0, 0))
        assert not result.converged and result.status in ("stalled", "iteration-limit")
        assert result.iterations <= 100 and -.2 <= result.configuration[qa] <= .2
        assert result.orientation_error_rad >= .8 - 1e-12
        actual_position, actual_rotation = native(mj, model, result.configuration)
        np.testing.assert_allclose(result.rotation, actual_rotation, rtol=0, atol=1e-12)
        assert result.position_error_m == pytest.approx(np.linalg.norm(position - actual_position), abs=1e-12)
        assert result.configuration[:7].tobytes() == seed[:7].tobytes()
    # Initial projection also respects the original selected range, with no caller write.
    outside = seed.copy()
    outside[qa] = .7
    projected = solve_native_pose_ik(model, target, rx(pi / 2) @ rz(.2), outside, ("turn",), local_point=(0, 0, 0))
    assert projected.converged and projected.iterations == 0
    assert projected.configuration[qa] == .2 and outside[qa] == .7


def test_bad_rotations_inputs_and_loosened_budgets_reject_before_native_work(mj, monkeypatch):
    model = model_fixture(mj)
    seed = model.qpos0.copy()
    target, rotation = native(mj, model, seed)
    request = dict(model=model, target=target, target_rotation=rotation, q_seed=seed,
                   joint_names=NAMES, local_point=(0, 0, 0))
    malformed = (np.eye(2), np.diag((1, 1, -1)), np.eye(3) * 2, np.zeros((3, 3)),
                 np.eye(3) + np.array(((0, .01, 0), (0, 0, 0), (0, 0, 0))),
                 np.full((3, 3), np.nan), np.full((3, 3), np.inf), np.eye(3, dtype=bool))
    bad = [dict(target_rotation=value) for value in malformed]
    bad.extend((dict(max_step=.151), dict(tolerance_m=.000501), dict(tolerance_rad=.005001),
                dict(max_iterations=101), dict(max_iterations=0), dict(max_iterations=True),
                dict(joint_names=()), dict(joint_names=("px", "px")), dict(joint_names=("missing",)),
                dict(joint_names=("object-free",)), dict(joint_names=("finger",)),
                dict(joint_names=iter(NAMES)), dict(body_name="missing"),
                dict(target=(0, np.nan, 0)), dict(local_point=(0, 0)), dict(q_seed=seed[:-1])))
    for name in ("orientation_length_m", "damping", "max_step", "tolerance_m", "tolerance_rad"):
        bad.extend({name: value} for value in (0, -1, np.nan, np.inf, True))
    quaternion, nonfinite = seed.copy(), seed.copy()
    quaternion[3:7], nonfinite[0] = 0, np.nan
    bad.extend((dict(q_seed=quaternion), dict(q_seed=nonfinite)))
    before = binary(mj, model)
    monkeypatch.setattr(mj, "MjData", forbidden)
    for values in bad:
        with pytest.raises(ValueError):
            solve_native_pose_ik(**(request | values))
    assert binary(mj, model) == before


def test_native_dimension_budget_is_checked_before_ik(mj, monkeypatch):
    content = '<body name="tool"><geom size=".01" mass=".1"/></body>'
    for index in range(36):
        content = (f'<body pos=".02 0 0"><joint name="joint-{index}" type="hinge" axis="0 0 1"/>'
                   '<geom size=".01" mass=".1"/>' + content + '</body>')
    model = mj.MjModel.from_xml_string('<mujoco><worldbody>' + content + '</worldbody></mujoco>')
    assert model.nv == 36
    monkeypatch.setattr(mj, "MjData", forbidden)
    with pytest.raises(ValueError, match="dimensions"):
        solve_native_pose_ik(model, (0, 0, 0), np.eye(3), model.qpos0, ("joint-0",))
