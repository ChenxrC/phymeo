"""AUTHORING ONLY: real free-root two-foot support, history and late-failure cases.

No tests in this file have been collected or executed by the author. The parent
runs the focused HOST nodes serially. G1 and device capacity are separate lanes.
"""

import json
from contextlib import contextmanager
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

import phymeo as ph
from phymeo.experiments.humanoid_support import foot_material_registry, read_foot_support
from phymeo.runtime import _multi_robot_tangent as tangent_adapter
from phymeo.sensors.contact_wrench import measure_contact_wrench

H = 1 / 1024
G = 9.81
ROOT_MASS = 2.0
FOOT_MASS = 1.0
TOTAL_MASS = ROOT_MASS + 2 * FOOT_MASS
K_NORMAL = 20000.0
D_NORMAL = 120.0
FOOT_HALF_HEIGHT = 1 / 32
ROOT_HEIGHT = 0.6
SPAN = 0.25
NORMAL_ATOL = 1e-5
STATE_ATOL = 2e-8


def _inertia(mass, diagonal):
    return (
        f'<inertial><mass value="{mass}"/>'
        f'<inertia ixx="{diagonal}" iyy="{diagonal}" izz="{diagonal}" '
        'ixy="0" ixz="0" iyz="0"/></inertial>'
    )


def _case(tmp_path, *, pitch_torque=0.0, reverse_owners=False, friction=0.0):
    normal = np.array(
        [
            TOTAL_MASS * G / 2 - pitch_torque / (2 * SPAN),
            TOTAL_MASS * G / 2 + pitch_torque / (2 * SPAN),
        ]
    )
    assert np.all(normal > 0)
    common_depth = TOTAL_MASS * G / (2 * K_NORMAL)
    q_feet = common_depth - normal / K_NORMAL
    links = []
    for name, x in (("left", -SPAN), ("right", SPAN)):
        links.append(
            f'<link name="{name}">'
            + _inertia(FOOT_MASS, 1 / 64)
            + f'<collision name="{name}_sole"><geometry>'
            '<box size=".24 .12 .0625"/></geometry></collision></link>'
            f'<joint name="{name}_slide" type="prismatic">'
            f'<parent link="trunk"/><child link="{name}"/>'
            f'<origin xyz="{x} 0 {-ROOT_HEIGHT + FOOT_HALF_HEIGHT}"/>'
            '<axis xyz="0 0 1"/>'
            '<limit lower="-.1" upper=".1" effort="128" velocity="20"/></joint>'
        )
    source = (
        '<robot name="free_two_foot_support"><link name="trunk">'
        + _inertia(ROOT_MASS, 0.5)
        + "</link>"
        + "".join(links)
        + "</robot>"
    )
    path = tmp_path / "two-foot.urdf"
    path.write_text(source, encoding="utf-8")
    robot, floor = (5, 10) if reverse_owners else (10, 5)
    q0 = np.r_[0.0, 0.0, ROOT_HEIGHT - common_depth, 0.0, 0.0, 0.0, 1.0, q_feet]
    state = ph.FloatingRobotState(q0, np.zeros(8))
    builder = ph.ContactSceneBuilder(gravity=(0, 0, -G))
    builder.add_floating_urdf(
        robot, path, robot_state=state, kp=[0, 0], kd=[0, 0], effort_limit=[128, 128]
    )
    builder.add_box(floor, (2.0, 2.0, 0.05), mass=None, position=(0, 0, -0.05))
    builder.pair(robot, floor, ph.RobotContactMaterial(K_NORMAL, D_NORMAL, friction))
    return SimpleNamespace(
        builder=builder,
        robot=robot,
        floor=floor,
        q0=q0,
        q_feet=q_feet,
        expected_normal=normal,
        pitch_torque=pitch_torque,
        equilibrium_effort=FOOT_MASS * G - normal,
        friction=friction,
        source=source,
    )


def _setup(case, scene):
    feet = {
        name: (scene.shape_key(case.robot, name, name + "_sole"),) for name in ("left", "right")
    }
    ground = scene.body_keys[0, case.floor]
    return feet, ground


def _summarize(case, scene, reading):
    feet, ground = _setup(case, scene)
    return read_foot_support(
        reading,
        feet,
        ground,
        floor_rigid_rows=(scene.rigid_rows[0, case.floor],),
        robot_address=(0, case.robot),
        reference_root_position_m=case.q0[:3],
    )


def _external(
    *, pitch_torque=0.0, up_acceleration=None, lateral_force=0.0, application_z=0.0, root_z=0.0
):
    force, torque = np.zeros((3, 3)), np.zeros((3, 3))
    torque[0, 1] = pitch_torque
    if up_acceleration is not None:
        force[:, 2] = np.array([ROOT_MASS, FOOT_MASS, FOOT_MASS]) * (G + up_acceleration)
    force[0, 0] = lateral_force
    torque[0, 1] += (application_z - root_z) * lateral_force
    return ph.FloatingRobotInputs(np.zeros(2), force, torque)


def _command(case, effort, external=None):
    return ph.FloatingRobotCommand(
        ph.RobotCommand(case.q_feet, feedforward=effort),
        external=external,
    )


@contextmanager
def _trace(tmp_path, name, case):
    record = {
        "schema": "humanoid-foot-support-authoring-v1",
        "name": name,
        "source_urdf": case.source,
        "dt_s": H,
        "masses_kg": [ROOT_MASS, FOOT_MASS, FOOT_MASS],
        "gravity_m_s2": [0, 0, -G],
        "normal_stiffness_n_m": K_NORMAL,
        "normal_damping_ns_m": D_NORMAL,
        "friction": case.friction,
        "attempts": [],
        "failure": None,
    }
    try:
        yield record["attempts"]
    except BaseException as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        (tmp_path / (name + ".json")).write_text(
            json.dumps(record, allow_nan=False), encoding="utf-8"
        )


def _advance(case, scene, command, attempts, phase):
    attempt = {
        "phase": phase,
        "entry_time_s": scene.state.time_s,
        "feedforward": command.joints.feedforward.tolist(),
        "external_force_n": None if command.external is None else command.external.force.tolist(),
        "external_torque_nm": None
        if command.external is None
        else command.external.torque.tolist(),
        "returned": False,
    }
    attempts.append(attempt)
    reading = scene.step(H, actions={case.robot: command})
    support = _summarize(case, scene, reading)
    row = reading.robots[0, case.robot]
    attempt.update(
        returned=True,
        time_s=reading.time_s,
        q=row.q.tolist(),
        v=row.v.tolist(),
        entry_q=row.entry_q.tolist(),
        entry_v=row.entry_v.tolist(),
        actual_drive_effort=row.drive_effort.tolist(),
        link_wrenches_world=row.link_wrenches_world.tolist(),
        contact_effort_full=row.contact_effort_full.tolist(),
        support=asdict(support),
        sites=[
            {
                "first": [load.first.world, load.first.body, load.first.shape],
                "second": [load.second.world, load.second.body, load.second.shape],
                "gap_m": load.gap_m,
                "normal": list(load.normal),
                "force_n": load.force_n,
                "material_first": list(site.material.first_point),
                "material_second": list(site.material.second_point),
                "weights_first": list(site.material.first_weights),
                "weights_second": list(site.material.second_weights),
                "fraction": site.fraction,
                "force_on_second_n": list(site.force_on_second_n),
                "application_first": list(site.application_point_first_m),
                "application_second": list(site.application_point_second_m),
            }
            for load in reading.loads.contacts
            for site in load.sites
        ],
    )
    return reading, support


def _station_registry(case, scene):
    feet, ground = _setup(case, scene)
    geometry = ph.floating_kinematics(
        scene.assets[0, case.robot].model, scene.state.robots[0, case.robot]
    )
    asset = scene.assets[0, case.robot]
    frames = {}
    for name, keys in feet.items():
        index = asset.link_names.index(name) + 1
        frames[keys[0]] = (geometry.position[index], geometry.rotation[index])
    shapes = scene.geometry().model.shapes
    vertices = {shape.key: shape.vertex_ids for shape in shapes}
    charts = {shape.key: ordinal for ordinal, shape in enumerate(shapes)}
    return foot_material_registry(
        scene.sample(),
        feet,
        ground,
        frames_by_shape=frames,
        shape_vertex_ids=vertices,
        chart_ids=charts,
    )


@pytest.mark.parametrize(
    "pitch_torque,reverse_owners",
    [(0.0, False), (0.0, True), (1.0, False)],
    ids=("symmetric", "reversed-owner-order", "unequal-load-and-moment"),
)
def test_h1_two_finite_feet_support_the_real_free_root(tmp_path, pitch_torque, reverse_owners):
    case = _case(tmp_path, pitch_torque=pitch_torque, reverse_owners=reverse_owners)
    with _trace(tmp_path, "stance", case) as attempts, case.builder.build(backend="host") as scene:
        for _ in range(128):
            reading, support = _advance(
                case,
                scene,
                _command(case, case.equilibrium_effort, _external(pitch_torque=pitch_torque)),
                attempts,
                "stance",
            )
            row = reading.robots[0, case.robot]
            actual = [foot.normal_force_n for foot in support.feet]
            np.testing.assert_allclose(actual, case.expected_normal, atol=NORMAL_ATOL, rtol=0)
            np.testing.assert_allclose(row.q, case.q0, atol=STATE_ATOL, rtol=0)
            np.testing.assert_allclose(row.v, 0, atol=STATE_ATOL, rtol=0)
            np.testing.assert_allclose(
                support.total_force_n, [0, 0, TOTAL_MASS * G], atol=NORMAL_ATOL, rtol=0
            )
            np.testing.assert_allclose(
                support.selected_ground_force_n,
                -np.array(support.total_force_n),
                atol=1e-10,
                rtol=0,
            )
            np.testing.assert_allclose(
                support.published_floor_force_n, support.selected_ground_force_n, atol=1e-10, rtol=0
            )
            np.testing.assert_allclose(
                support.total_moment_nm, [0, -pitch_torque, 0], atol=NORMAL_ATOL, rtol=0
            )
            assert support.loaded_feet == ("left", "right")
            assert (
                support.sample_time_s == reading.load_time_s
                and support.commit_time_s == reading.time_s
            )
            np.testing.assert_array_equal(row.fixed_root_contact_wrench, np.zeros(6))


def test_h1_liftoff_and_reentry_use_one_continuous_trajectory(tmp_path):
    case = _case(tmp_path)
    with (
        _trace(tmp_path, "liftoff-reentry", case) as attempts,
        case.builder.build(backend="host") as scene,
    ):
        feet, ground = _setup(case, scene)
        saw_open = saw_reentry = False
        highest = case.q0[2]
        for phase, count, acceleration in (
            ("stand", 16, None),
            ("lift", 128, 2.0),
            ("return", 384, None),
        ):
            for _ in range(count):
                # A real load-feedback motor command preserves the two relative
                # vertical coordinates; it never writes root or joint state.
                sample = scene.sample()
                normal = np.array(
                    [
                        sum(
                            measure_contact_wrench(sample, key, counterparts=ground).force_n[2]
                            for key in feet[name]
                        )
                        for name in ("left", "right")
                    ]
                )
                effort = FOOT_MASS * normal.sum() / TOTAL_MASS - normal
                command = _command(case, effort, _external(up_acceleration=acceleration))
                reading, support = _advance(case, scene, command, attempts, phase)
                row = reading.robots[0, case.robot]
                np.testing.assert_allclose(row.q[7:], case.q_feet, atol=2e-6, rtol=0)
                np.testing.assert_allclose(row.q[:2], 0, atol=2e-6, rtol=0)
                highest = max(highest, row.q[2])
                if phase == "lift" and not support.loaded_feet:
                    saw_open = True
                    np.testing.assert_allclose(support.total_force_n, 0, atol=1e-10, rtol=0)
                if phase == "return" and saw_open and support.loaded_feet == ("left", "right"):
                    saw_reentry = True
        assert saw_open and saw_reentry
        assert highest - case.q0[2] > 0.005
        assert support.loaded_feet == ("left", "right")
        assert abs(row.v[2]) < 0.1
        assert row.q[2] > 0.5
        assert all(attempt["returned"] for attempt in attempts)


def test_h1_changed_foot_patch_rejects_without_losing_the_accepted_state(tmp_path):
    case = _case(tmp_path, friction=0.5)
    with (
        _trace(tmp_path, "lateral-load-release", case) as attempts,
        case.builder.build(backend="host") as scene,
    ):
        registry = _station_registry(case, scene)
        assert len(registry.registrations) > 2
        scene.enable_coupled_tangent(registry, stiffness_n_per_m=200.0)

        def command():
            entry = scene.observe().robots[0, case.robot]
            return _command(
                case,
                case.equilibrium_effort,
                _external(
                    lateral_force=2.0,
                    application_z=-TOTAL_MASS * G / (4 * K_NORMAL),
                    root_z=entry.q[2],
                ),
            )

        _, support = _advance(case, scene, command(), attempts, "lateral-load")
        assert scene.tangent_receipt.accepted
        assert support.loaded_feet == ("left", "right")
        before, history, events = scene.observe(), scene.tangent_history, scene.contact_events
        # The original 128-step positive target failed at this second request.
        # Preserve it as an explicit unsupported-feature/transaction regression;
        # this is not a successful lateral-support or rolling-friction claim.
        with pytest.raises(ValueError, match="missing or ambiguous material registration"):
            _advance(case, scene, command(), attempts, "changed-patch-rejection")
        after = scene.observe()
        assert after.time_s == before.time_s
        np.testing.assert_array_equal(after.robots[0, case.robot].q, before.robots[0, case.robot].q)
        np.testing.assert_array_equal(after.robots[0, case.robot].v, before.robots[0, case.robot].v)
        assert scene.tangent_history is history and scene.contact_events is events


def test_h1_late_tangent_failure_keeps_the_published_foot_state(tmp_path, monkeypatch):
    case = _case(tmp_path, friction=0.5)
    with (
        _trace(tmp_path, "late-rejection", case) as attempts,
        case.builder.build(backend="host") as scene,
    ):
        scene.enable_coupled_tangent(_station_registry(case, scene), stiffness_n_per_m=200.0)
        command = _command(case, case.equilibrium_effort)
        _advance(case, scene, command, attempts, "accepted")
        before, history, events, geometry = (
            scene.observe(),
            scene.tangent_history,
            scene.contact_events,
            scene.geometry(),
        )
        original = tangent_adapter.finish_tangent_step

        def reject_after_real_finish(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected after complete material-motion audit")

        monkeypatch.setattr(tangent_adapter, "finish_tangent_step", reject_after_real_finish)
        with pytest.raises(RuntimeError, match="complete material-motion"):
            _advance(case, scene, command, attempts, "late-rejection")
        after = scene.observe()
        assert after.time_s == before.time_s and after.step_number == before.step_number
        np.testing.assert_array_equal(after.robots[0, case.robot].q, before.robots[0, case.robot].q)
        np.testing.assert_array_equal(after.robots[0, case.robot].v, before.robots[0, case.robot].v)
        assert (
            scene.tangent_history is history
            and scene.contact_events is events
            and scene.geometry() is geometry
        )
        monkeypatch.setattr(tangent_adapter, "finish_tangent_step", original)
        resumed, _ = _advance(case, scene, command, attempts, "retry")
        assert resumed.step_number == before.step_number + 1
