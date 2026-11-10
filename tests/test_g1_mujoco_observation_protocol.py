"""Current-state G1 observation axes, independent native sensor oracle.

These tests require only inline native models, not the G1 motion or ONNX asset.
The policy double records its input; no learned policy or physical step is run.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def example():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("yaml")
    path = Path(__file__).resolve().parents[1] / "examples/g1_mujoco_tyx_policy.py"
    spec = importlib.util.spec_from_file_location("g1_observation_example_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gyro_model(sensor="gyro", site_quat="0.9238795325112867 0.3826834323650898 0 0"):
    site = {"world": "world_imu", "child": "child_imu"}.get(sensor, "root_imu")
    sensor_xml = (
        ""
        if sensor == "missing"
        else f'<{("accelerometer" if sensor == "wrong_type" else "gyro")} '
        f'name="imu_ang_vel" site="{site}"/>'
    )
    return mj.MjModel.from_xml_string(
        f"""<mujoco>
          <option gravity="0 0 0"/>
          <worldbody>
            <site name="world_imu" size=".01"/>
            <body name="preceding">
              <joint name="preceding_joint" type="hinge"/>
              <geom type="sphere" size=".05" mass=".2"/>
            </body>
            <body name="floating" pos=".1 -.2 .7">
              <freejoint name="floating_joint"/>
              <inertial pos=".01 .02 -.03" quat=".9238795325112867 0 .3826834323650898 0"
                        mass="2" diaginertia=".07 .08 .09"/>
              <geom type="sphere" size=".05" mass="0"/>
              <site name="root_imu" pos=".03 -.04 .02" quat="{site_quat}" size=".01"/>
              <body name="child" pos=".2 0 0">
                <joint name="child_joint" type="hinge"/>
                <geom type="sphere" size=".04" mass=".3"/>
                <site name="child_imu" size=".01"/>
              </body>
            </body>
          </worldbody>
          <sensor>{sensor_xml}</sensor>
        </mujoco>"""
    )


def _body(model, name):
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)


def _state(model):
    root = _body(model, "floating")
    joint = int(model.body_jntadr[root])
    qa, va = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    data = mj.MjData(model)
    data.qpos[qa + 3 : qa + 7] = np.array([0.7, -0.2, 0.3, 0.4]) / np.sqrt(0.78)
    data.qvel[:] = np.linspace(-0.4, 0.6, model.nv)
    data.qvel[va + 3 : va + 6] = [0.7, -1.1, 0.4]
    mj.mj_forward(model, data)
    return root, joint, data


@pytest.mark.parametrize(
    "site_quat",
    ["1 0 0 0", "0.9238795325112867 0.3826834323650898 0 0"],
    ids=["body_aligned_site", "rotated_site"],
)
def test_current_site_axes_match_independent_native_sensor_and_site_velocity(example, site_quat):
    model = _gyro_model(site_quat=site_quat)
    root, joint, data = _state(model)
    gyro = example.GyroObservation(model, root_body=root)
    legacy = example.GyroObservation(
        model, root_body=root, protocol_id=example.BODY_INERTIAL_PROTOCOL
    )
    sensor = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    site = int(model.sensor_objid[sensor])
    address = int(model.sensor_adr[sensor])
    native_site, native_body = np.zeros(6), np.zeros(6)
    mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_SITE, site, native_site, 1)
    mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, root, native_body, 1)
    assert root != 1 and int(model.jnt_dofadr[joint]) != 0
    np.testing.assert_allclose(gyro.angular_velocity(data.qvel), native_site[:3], atol=2e-14)
    np.testing.assert_allclose(
        gyro.angular_velocity(data.qvel), data.sensordata[address : address + 3], atol=2e-14
    )
    np.testing.assert_allclose(legacy.angular_velocity(data.qvel), native_body[:3], atol=2e-14)
    assert not np.allclose(native_site[:3], native_body[:3], rtol=0, atol=1e-3)
    saved = data.qvel.copy()
    gyro.angular_velocity(data.qvel)
    np.testing.assert_array_equal(data.qvel, saved)


def test_current_state_has_no_cached_sensor_or_native_kinematics_dependency(example, monkeypatch):
    model = _gyro_model()
    root, _, first = _state(model)
    second = mj.MjData(model)
    second.qpos[:] = first.qpos
    second.qvel[:] = first.qvel * -0.75
    mj.mj_forward(model, second)
    sensor = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    address = int(model.sensor_adr[sensor])
    expected = second.sensordata[address : address + 3].copy()
    gyro = example.GyroObservation(model, root_body=root)
    first.sensordata[:] = 999
    second.sensordata[:] = -999

    def forbidden(*args, **kwargs):
        raise AssertionError("observation producer must not query native state caches or FK")

    for name in ("mj_forward", "mj_kinematics", "mj_objectVelocity"):
        monkeypatch.setattr(example.mj, name, forbidden)
    np.testing.assert_allclose(gyro.angular_velocity(second.qvel), expected, atol=2e-14)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "gyro sensor is absent"),
        ("wrong_type", "sensor must be a gyro"),
        ("world", "site must be mounted on the declared free root"),
        ("child", "site must be mounted on the declared free root"),
        ("nonfree", "one free joint"),
        ("bad_object_type", "gyro sensor must reference a site"),
        ("bad_object_id", "gyro site is absent"),
        ("missing_root", "free root body is absent"),
        ("unknown_protocol", "unknown observation protocol"),
    ],
)
def test_invalid_sensor_and_root_bindings_fail_explicitly(example, case, message):
    model = _gyro_model(case)
    root = _body(model, "preceding" if case == "nonfree" else "floating")
    protocol = example.IMU_SITE_PROTOCOL
    if case == "bad_object_type":
        model.sensor_objtype[0] = mj.mjtObj.mjOBJ_BODY
    elif case == "bad_object_id":
        model.sensor_objid[0] = model.nsite
    elif case == "missing_root":
        root = -1
    elif case == "unknown_protocol":
        protocol = "unversioned"
    with pytest.raises(ValueError, match=message):
        example.GyroObservation(model, root_body=root, protocol_id=protocol)


def test_metadata_binds_compiled_sensor_site_root_addresses_and_version(example):
    model = _gyro_model()
    root, joint, _ = _state(model)
    gyro = example.GyroObservation(model, root_body=root)
    metadata = gyro.metadata()
    sensor = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    site = int(model.sensor_objid[sensor])
    assert metadata["protocol_id"] == example.IMU_SITE_PROTOCOL
    assert metadata["sensor"] == {"id": sensor, "name": "imu_ang_vel", "type": "gyro"}
    assert metadata["site"]["id"] == site and metadata["site"]["name"] == "root_imu"
    np.testing.assert_array_equal(metadata["site"]["local_quaternion"], model.site_quat[site])
    assert metadata["body"] == {"id": root, "name": "floating"}
    assert metadata["free_joint"]["id"] == joint
    assert metadata["free_joint"]["dof_address"] == int(model.jnt_dofadr[joint])
    assert metadata["angular_dof_addresses"] == list(
        range(int(model.jnt_dofadr[joint]) + 3, int(model.jnt_dofadr[joint]) + 6)
    )
    assert metadata["timing"] == "current accepted input state before policy inference"
    assert metadata["frame_rule"] == "min(round(stored_time * 50), last_frame)"
    assert metadata["control_period_s"] == 0.02
    metadata["site"]["local_quaternion"][0] = 0
    assert gyro.metadata()["site"]["local_quaternion"][0] == model.site_quat[site, 0]


class _PolicyRecorder:
    def __init__(self):
        self.observations = []

    def run(self, outputs, inputs):
        assert outputs == ["actions"] and set(inputs) == {"obs"}
        self.observations.append(inputs["obs"].copy())
        return [np.linspace(-0.2, 0.3, 29, dtype=np.float32)[None]]


def _controller_protocol(example, protocol_id):
    children = "".join(
        f'<body pos="0 0 .1"><joint name="j{i}" type="hinge"/>'
        '<geom type="sphere" size=".02" mass=".1"/></body>'
        for i in range(29)
    )
    motors = "".join(
        f'<motor joint="j{i}" ctrllimited="true" ctrlrange="-100 100"/>' for i in range(29)
    )
    model = mj.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="base" pos="0 0 .7">'
        '<freejoint/><inertial pos="0 0 0" quat=".9238795325112867 0 .3826834323650898 0" '
        'mass="2" diaginertia=".07 .08 .09"/>'
        '<site name="sensor_frame" quat=".9238795325112867 .3826834323650898 0 0"/>'
        + children
        + '</body></worldbody><actuator>'
        + motors
        + '</actuator><sensor><gyro name="imu_ang_vel" site="sensor_frame"/></sensor></mujoco>'
    )
    protocol = object.__new__(example.Protocol)
    protocol.model = model
    protocol.qa = model.jnt_qposadr[model.actuator_trnid[:, 0]].astype(int)
    protocol.va = model.jnt_dofadr[model.actuator_trnid[:, 0]].astype(int)
    protocol.default = np.linspace(-0.1, 0.1, 29)
    protocol.kp, protocol.kd = np.full(29, 5.0), np.full(29, 0.5)
    protocol.scale, protocol.offset = np.full(29, 0.3), protocol.default.copy()
    protocol.alignment = np.eye(3)
    protocol.motion = {
        "joint_pos": np.stack([np.linspace(-0.2, 0.2, 29) + i * 0.01 for i in range(4)]),
        "joint_vel": np.stack([np.linspace(-0.3, 0.4, 29) - i * 0.02 for i in range(4)]),
        "body_quat_w": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (4, 1, 1)),
    }
    protocol.session = _PolicyRecorder()
    protocol.gyro = example.GyroObservation(
        model, root_body=_body(model, "base"), protocol_id=protocol_id
    )
    q, v = model.qpos0.copy(), np.linspace(-0.4, 0.7, model.nv)
    q[3:7] = np.array([0.7, -0.2, 0.3, 0.4]) / np.sqrt(0.78)
    q[protocol.qa] = np.linspace(-0.15, 0.25, 29)
    return protocol, q, v


def test_version_switch_changes_only_gyro_words_and_preserves_legacy_and_hold(example):
    legacy, q, v = _controller_protocol(example, example.BODY_INERTIAL_PROTOCOL)
    current, q2, v2 = _controller_protocol(example, example.IMU_SITE_PROTOCOL)
    np.testing.assert_array_equal(q, q2)
    np.testing.assert_array_equal(v, v2)
    previous = np.linspace(0.2, -0.3, 29, dtype=np.float32)
    controllers = [example.Controller(p, q, 0.002) for p in (legacy, current)]
    for controller in controllers:
        controller.action = previous.copy()
    proposals = [controller.propose(q, v, 0.02, 10) for controller in controllers]
    old_obs, new_obs = [proposal["observation"][0] for proposal in proposals]
    keep = np.r_[0:64, 67:154]
    np.testing.assert_array_equal(old_obs.view(np.uint32)[keep], new_obs.view(np.uint32)[keep])
    assert not np.array_equal(old_obs[64:67], new_obs[64:67])
    native_inertial = np.zeros(9)
    inertial_q = legacy.model.body_iquat[_body(legacy.model, "base")].copy()
    inertial_q /= np.linalg.norm(inertial_q)
    mj.mju_quat2Mat(native_inertial, inertial_q)
    np.testing.assert_array_equal(
        old_obs[64:67], (native_inertial.reshape(3, 3).T @ v[3:6]).astype(np.float32)
    )
    np.testing.assert_array_equal(new_obs[:29], current.motion["joint_pos"][1].astype(np.float32))
    np.testing.assert_array_equal(new_obs[29:58], current.motion["joint_vel"][1].astype(np.float32))
    np.testing.assert_array_equal(new_obs[125:], previous)
    assert current.frame(0.029) == 1 and current.frame(0.031) == 2
    assert current.frame(100.0) == 3
    for controller, proposal in zip(controllers, proposals, strict=True):
        controller.commit(proposal)
        next_q, next_v = q.copy(), v.copy()
        next_q[controller.p.qa] += 0.001
        next_v[controller.p.va] -= 0.002
        held = controller.propose(next_q, next_v, 0.022, 11)
        assert held["observation"] is None
        np.testing.assert_array_equal(held["action"], proposal["action"])
        np.testing.assert_array_equal(held["target"], proposal["target"])
        expected = controller.p.kp * (held["target"] - next_q[controller.p.qa])
        expected -= controller.p.kd * next_v[controller.p.va]
        np.testing.assert_array_equal(held["ctrl"], np.clip(expected, -100, 100))
        assert len(controller.p.session.observations) == 1


@pytest.mark.parametrize("explicit", [False, True], ids=["default_site_v2", "explicit_legacy_v1"])
def test_cli_selects_explicit_version_without_changing_motion_options(example, monkeypatch, explicit):
    args = ["g1_mujoco_tyx_policy.py", "--assets", ".", "--output", "result.json", "--mode", "cuda"]
    if explicit:
        args += ["--observation-protocol", example.BODY_INERTIAL_PROTOCOL]
    monkeypatch.setattr(sys, "argv", args)
    parsed = example.parse_args()
    assert parsed.observation_protocol == (
        example.BODY_INERTIAL_PROTOCOL if explicit else example.IMU_SITE_PROTOCOL
    )
    assert parsed.h == 0.002 and parsed.duration == 2.0 and parsed.push_sign == 0
