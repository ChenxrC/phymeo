"""UNEXECUTED T3 tests-first for one common floating-mimic scene transaction."""

import gzip
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from phymeo.builders.floating_mimic_urdf import load_floating_mimic_urdf
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.dynamics.floating_robot import FloatingRobotState
from phymeo.runtime._floating_robot_participant import FloatingRobotCommand
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand
from phymeo.solvers.floating_tree import FloatingRobotInputs

ASSETS = Path(__file__).parent / "fixtures" / "urdf"
JAWS = ASSETS / "floating_mimic_jaws.urdf"
WORLD, ROBOT, PARTNER, BALL = 23, 17, 42, 91
H = 1 / 1024
STATE_ATOL, FORCE_ATOL = 2e-11, 1e-9
CYCLE = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])


def _example():
    path = Path(__file__).resolve().parents[1] / "examples" / "floating_mimic_contact.py"
    spec = spec_from_file_location("floating_mimic_contact_example", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state():
    return FloatingRobotState(np.array([0.0, 0, 0, 0, 0, 0, 1, 0]), np.zeros(7))


def _add_mimic(builder, *, kp=0, kd=0):
    builder.add_floating_urdf(
        ROBOT,
        JAWS,
        robot_state=_state(),
        kp=[kp],
        kd=[kd],
        effort_limit=[1.5],
        loader=load_floating_mimic_urdf,
        world=WORLD,
    )


def _command(effort=0.0, *, external=None):
    return FloatingRobotCommand(RobotCommand([0.0], feedforward=[effort]), external)


def test_a01_public_builder_accepts_reduced_controls_and_full_physical_owners():
    # T2's real free-root response is a precondition; RED is the builder's
    # inability to register this valid wrapper, not a missing helper import.
    asset = load_floating_mimic_urdf(JAWS, gravity=(0, 0, 0))
    standalone = asset.step(_state(), effort=np.array([1.0]), dt_s=H)
    assert standalone.loads.multiplier_force == pytest.approx(-4 / 7, abs=STATE_ATOL)
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add_mimic(builder)
    with builder.build(backend="host") as scene:
        initial = scene.observe()
        row = initial.robots[WORLD, ROBOT]
        assert row.q.shape == (8,) and row.v.shape == (7,)
        assert row.full_q.shape == (9,) and row.full_v.shape == (8,)
        assert row.drive_effort.shape == (1,)
        assert row.contact_effort.shape == (7,) and row.contact_effort_full.shape == (8,)
        assert row.link_wrenches_world.shape == (3, 6)
        key = scene.shape_key(ROBOT, "right_jaw", "right_pad")
        assert key.world == WORLD
        assert np.max(scene.geometry().poses[key].start[:, 0]) == pytest.approx(
            0.254 + 1 / 32, abs=STATE_ATOL
        )
        events = scene.contact_events
        scene.observe_scene()
        scene.sample()
        assert scene.contact_events is events and events.step_index == 0
        reading = scene.step(H, actions={ROBOT: _command(1)})
        actual = reading.robots[WORLD, ROBOT]
        np.testing.assert_allclose(actual.q, standalone.state.q, atol=STATE_ATOL, rtol=0)
        np.testing.assert_allclose(actual.full_v, standalone.full_state.v, atol=STATE_ATOL, rtol=0)
        np.testing.assert_allclose(
            actual.acceleration_full, standalone.loads.acceleration_full, atol=STATE_ATOL, rtol=0
        )
        np.testing.assert_allclose(actual.fixed_root_contact_wrench, 0, atol=STATE_ATOL, rtol=0)
        assert actual.mimic_loads.multiplier_force == pytest.approx(-4 / 7, abs=STATE_ATOL)
        assert (reading.step_number, reading.load_time_s, reading.time_s) == (1, 0.0, H)
        assert scene.geometry().time_s == H
        assert np.max(scene.geometry().poses[key].start[:, 0]) == pytest.approx(
            0.254 + 1 / 32 - 2 * H * H / 7, abs=STATE_ATOL
        )


@pytest.mark.parametrize(
    "cap,root_force,feedforward",
    [(1.5, 0.0, 0.0), (0.25, 0.0, 0.0), (1.5, 1.0, None), (1.5, 1.0, 0.0)],
    ids=["free-schur", "motor-cap", "external-default-ff", "external-explicit-zero"],
)
def test_a02_pd_uses_projected_free_root_schur_and_preserves_requested_command(
    cap, root_force, feedforward
):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add_mimic(builder, kp=100, kd=2)
    external_force = np.zeros((3, 3))
    external_force[0, 0] = root_force
    external = FloatingRobotInputs(np.zeros(2), external_force, np.zeros((3, 3)))
    requested = RobotCommand(
        [0.01], effort_limit=[cap], feedforward=None if feedforward is None else [feedforward]
    )
    command = FloatingRobotCommand(requested, external)
    h, effective_mass, effective_bias = 1 / 128, 14 / 5, -root_force / 5
    ff = effective_bias if feedforward is None else feedforward
    alpha = h * (100 * h + 2) / effective_mass
    expected_effort = np.clip((1 + ff + alpha * effective_bias) / (1 + alpha), -cap, cap)
    acceleration = np.array(
        [
            (3 * root_force + expected_effort) / 14,
            0,
            0,
            0,
            0,
            0,
            (root_force + 5 * expected_effort) / 14,
            -(root_force + 5 * expected_effort) / 14,
        ]
    )
    with builder.build(backend="host") as scene:
        row = scene.step(h, actions={ROBOT: command}).robots[WORLD, ROBOT]
        np.testing.assert_allclose(row.drive_effort, [expected_effort], atol=1e-10, rtol=0)
        np.testing.assert_allclose(row.acceleration_full, acceleration, atol=1e-10, rtol=0)
        np.testing.assert_allclose(row.full_v, h * acceleration, atol=STATE_ATOL, rtol=0)
        np.testing.assert_allclose(
            row.applied_effort_full,
            [root_force, 0, 0, 0, 0, 0, expected_effort, 0],
            atol=1e-10,
            rtol=0,
        )
        np.testing.assert_array_equal(row.external.force, external_force)
        np.testing.assert_array_equal(row.contact_effort_full, np.zeros(8))
        np.testing.assert_allclose(
            row.mimic_loads.contact_effort_full,
            [root_force, 0, 0, 0, 0, 0, 0, 0],
            atol=STATE_ATOL,
            rtol=0,
        )
        assert row.mimic_loads.multiplier_force == pytest.approx(
            (2 * root_force - 4 * expected_effort) / 7, abs=1e-10
        )
        np.testing.assert_allclose(row.command.joints.feedforward, [ff], atol=STATE_ATOL, rtol=0)
        if feedforward is None:
            assert row.requested_command.joints.feedforward is None
        else:
            np.testing.assert_array_equal(row.requested_command.joints.feedforward, [0.0])
        assert row.q[7] != pytest.approx(0.01, abs=1e-6)


def _ordinary_path(tmp_path):
    center = 0.25 + 0.004 + 1 / 32 + 1 / 16 - 1 / 1024
    path = tmp_path / "ordinary_probe.urdf"
    path.write_text(
        f'''<robot name="ordinary_probe">
<link name="base"/>
<link name="probe"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.015" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial>
<collision name="sphere"><geometry><sphere radius="0.0625"/></geometry></collision></link>
<joint name="slide" type="prismatic"><parent link="base"/><child link="probe"/><origin xyz="{center:.17g} 0 0"/><axis xyz="1 0 0"/><limit lower="-0.01" upper="0.01" effort="4" velocity="1"/></joint>
</robot>''',
        encoding="utf-8",
    )
    return path


def _pair_builder(tmp_path, *, reverse=False, partner_q=0.0, partner_v=0.0, extra_ball=False):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    path = _ordinary_path(tmp_path)

    def mimic():
        _add_mimic(builder)

    def ordinary():
        builder.add_urdf(
            PARTNER,
            path,
            robot_state=RobotState(np.array([partner_q]), np.array([partner_v])),
            kp=[0],
            kd=[0],
            effort_limit=[4],
            world=WORLD,
        )

    for add in (ordinary, mimic) if reverse else (mimic, ordinary):
        add()
    builder.pair(ROBOT, PARTNER, RobotContactMaterial(2048, 0, 0), world=WORLD)
    if extra_ball:
        builder.add_sphere(BALL, 1 / 16, mass=1, position=(3, 0, 0), world=WORLD)
        builder.pair(ROBOT, BALL, None, world=WORLD)
        builder.pair(PARTNER, BALL, None, world=WORLD)
    return builder


def _pair_actions(partner_effort=0.0, *, external=None):
    return {
        ROBOT: _command(external=external),
        PARTNER: RobotCommand([0.0], feedforward=[partner_effort]),
    }


@pytest.mark.parametrize(
    "reverse", [False, True], ids=["forward-registration", "reverse-registration"]
)
def test_b01_real_two_newton_ordinary_robot_contact_preserves_full_owners_and_reactions(
    tmp_path, reverse
):
    with _pair_builder(tmp_path, reverse=reverse).build(backend="host") as scene:
        prior_events = scene.contact_events
        sampled = scene.sample()
        assert sum(
            site.force_n for pair in sampled.contacts for site in pair.sites
        ) == pytest.approx(2.0, abs=FORCE_ATOL)
        assert scene.contact_events is prior_events and prior_events.step_index == 0
        reading = scene.step(H, actions=_pair_actions())
        first, second = reading.robots[WORLD, ROBOT], reading.robots[WORLD, PARTNER]
        expected = np.array([-2 / 7, 0, 0, 0, 0, 0, 4 / 7, -4 / 7])
        np.testing.assert_allclose(first.acceleration_full, expected, atol=FORCE_ATOL, rtol=0)
        np.testing.assert_allclose(first.full_v, H * expected, atol=STATE_ATOL, rtol=0)
        np.testing.assert_allclose(second.v, [2 * H], atol=STATE_ATOL, rtol=0)
        np.testing.assert_allclose(
            first.contact_effort_full, [-2, 0, 0, 0, 0, 0, 0, -2], atol=FORCE_ATOL, rtol=0
        )
        np.testing.assert_allclose(
            first.contact_effort, [-2, 0, 0, 0, 0, 0, 2], atol=FORCE_ATOL, rtol=0
        )
        np.testing.assert_allclose(second.contact_effort_full, [2], atol=FORCE_ATOL, rtol=0)
        np.testing.assert_allclose(
            first.applied_effort_full, first.contact_effort_full, atol=FORCE_ATOL, rtol=0
        )
        assert first.mimic_loads.multiplier_force == pytest.approx(2 / 7, abs=FORCE_ATOL)
        np.testing.assert_allclose(
            first.mimic_loads.constraint_effort_full[:6], 0, atol=STATE_ATOL, rtol=0
        )
        np.testing.assert_allclose(
            first.link_wrenches_world,
            [[0] * 6, [0] * 6, [-2, 0, 0, 0, 0, 0]],
            atol=FORCE_ATOL,
            rtol=0,
        )
        np.testing.assert_allclose(
            second.link_wrenches_world, [[2, 0, 0, 0, 0, 0]], atol=FORCE_ATOL, rtol=0
        )
        np.testing.assert_array_equal(first.drive_effort, [0.0])
        np.testing.assert_array_equal(second.drive_effort, [0.0])
        moving_momentum = 5 * first.full_v[0] + first.full_v[6] + 2 * first.full_v[7] + second.v[0]
        assert moving_momentum == pytest.approx(0.0, abs=STATE_ATOL)
        events = scene.contact_events
        assert events.sample_time_s == 0.0 and events.commit_time_s == H
        assert len(events.active) == 1
        record = events.active[0]
        for owner in (record.first_owner, record.second_owner):
            assert owner.world == WORLD and owner.kind == "robot_link"
            assert owner.link_index == (2 if owner.body == ROBOT else 0)
        centers = {
            ROBOT: np.array([0.254, 0, 0]),
            PARTNER: np.array([0.25 + 0.004 + 1 / 32 + 1 / 16 - 1 / 1024, 0, 0]),
        }
        for component in record.components:
            for material, site in zip(component.material, component.load.sites, strict=True):
                for owner, address, point in (
                    (record.first_owner, material.first, site.material.first_point),
                    (record.second_owner, material.second, site.material.second_point),
                ):
                    np.testing.assert_allclose(
                        centers[owner.body] + address.local_point_m, point, atol=STATE_ATOL, rtol=0
                    )
                np.testing.assert_allclose(
                    site.application_point_first_m,
                    site.application_point_second_m,
                    atol=STATE_ATOL,
                    rtol=0,
                )
        scene.observe_scene()
        scene.sample()
        assert scene.contact_events is events


def _same(scene, old, geometry, events):
    example = _example()
    assert example.plain(scene.observe()) == example.plain(old)
    assert scene.geometry() is geometry
    assert scene.contact_events is events


def test_b02_second_participant_limit_keeps_all_candidates_and_retries(tmp_path):
    builder = _pair_builder(tmp_path, partner_q=0.0099999, partner_v=0.001, extra_ball=True)
    with builder.build(backend="host") as scene, builder.build(backend="host") as fresh:
        old, geometry = scene.observe_scene()
        events = scene.contact_events
        external = scene.inputs()
        external.rigid.force[scene.rigid_rows[WORLD, BALL], 1] = 7.0
        robot_force = np.zeros((3, 3))
        robot_force[0, 0] = 1.0
        com = FloatingRobotInputs(np.zeros(2), robot_force, np.zeros((3, 3)))
        with pytest.raises(ValueError, match="position|interval"):
            scene.step(H, actions=_pair_actions(external=com), external=external)
        _same(scene, old, geometry, events)
        assert scene.last_frame is None
        actions = _pair_actions(-4.0, external=com)
        actual = scene.step(H, actions=actions, external=external)
        expected = fresh.step(H, actions=actions, external=external)
        assert _example().plain(actual) == _example().plain(expected)
        assert scene.contact_events == fresh.contact_events
        assert actual.robots[WORLD, ROBOT].q[0] != 0
        assert actual.rigid.velocity[scene.rigid_rows[WORLD, BALL], 1] == pytest.approx(7 * H)


def test_b03_late_event_failure_keeps_loaded_material_then_retry_matches(tmp_path, monkeypatch):
    builder = _pair_builder(tmp_path)
    with builder.build(backend="host") as scene, builder.build(backend="host") as fresh:
        scene.step(H, actions=_pair_actions())
        fresh.step(H, actions=_pair_actions())
        old, geometry = scene.observe_scene()
        events = scene.contact_events
        assert events.active

        def reject(*args, **kwargs):
            raise RuntimeError("F2 late event failure")

        with monkeypatch.context() as patcher:
            patcher.setattr(scene._event_adapter, "stage", reject)
            with pytest.raises(RuntimeError, match="F2 late event failure"):
                scene.step(H, actions=_pair_actions())
        _same(scene, old, geometry, events)
        actual = scene.step(H, actions=_pair_actions())
        expected = fresh.step(H, actions=_pair_actions())
        assert _example().plain(actual) == _example().plain(expected)
        assert scene.contact_events == fresh.contact_events
        assert events.step_index == 1


def test_b04_owned_readings_and_repeated_observations_do_not_change_episode(tmp_path):
    builder = _pair_builder(tmp_path, extra_ball=True)
    target = np.array([0.0])
    command = FloatingRobotCommand(RobotCommand(target, feedforward=[0.0]))
    with builder.build(backend="host") as scene:
        reading = scene.step(
            H, actions={ROBOT: command, PARTNER: RobotCommand([0.0], feedforward=[0.0])}
        )
        saved = _example().plain(scene.observe())
        events = scene.contact_events
        target[0] = 0.015
        reading.loads.inputs.rigid.force[0, 1] = 99.0
        assert _example().plain(scene.observe()) == saved
        np.testing.assert_array_equal(
            scene.observe().robots[WORLD, ROBOT].requested_command.joints.target_q, [0.0]
        )
        scene.observe_scene()
        scene.sample()
        assert scene.contact_events is events and events.step_index == 1


@pytest.fixture(scope="module")
def contact_windows(tmp_path_factory):
    example = _example()
    directory = tmp_path_factory.mktemp("floating-mimic-full-windows")
    results = {}
    for name, options in (
        ("h", {"dt_s": 1 / 2048}),
        ("half", {"dt_s": 1 / 4096}),
        ("reverse", {"dt_s": 1 / 2048, "reverse": True}),
        ("transform", {"dt_s": 1 / 2048, "transformed": True}),
    ):
        result = example.run_contact_window(fixture_dir=ASSETS, **options)
        path = directory / f"{name}.json.gz"
        with path.open("xb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                compressed.write(
                    json.dumps(result, allow_nan=False, separators=(",", ":")).encode("utf-8")
                )
        print("F03", name, "complete artifact", path, result["summary"], result["failure"])
        results[name] = result
    return results


def test_c01_original_physical_window_keeps_full_material_force_and_mimic_receipts(contact_windows):
    example = _example()
    for name, result in contact_windows.items():
        frames, summary, h = result["frames"], result["summary"], result["protocol"]["dt_s"]
        assert result["failure"] is None, (name, result["failure"])
        assert summary["qualified"], (name, summary)
        assert len(frames) == round((1 / 4) / h)
        assert summary["time_s"] == pytest.approx(1 / 4, abs=STATE_ATOL)
        assert not example.sites(frames[0])
        assert 0 < summary["first_load_time_s"] < summary["last_load_time_s"] < 1 / 4
        assert summary["peak_force_n"] > 0.1
        assert summary["max_linear_residual_ns"] <= 5e-4
        impulse = np.zeros(3)
        for index, frame in enumerate(frames, start=1):
            reading, row = frame["reading"], example.robot_reading(frame)
            entry_v, end_v = np.asarray(frame["entry_full"]["v"]), np.asarray(row["full_v"])
            assert (reading["step_number"], reading["load_time_s"], reading["time_s"]) == (
                index,
                (index - 1) * h,
                index * h,
            )
            assert frame["events"]["sample_time_s"] == reading["load_time_s"]
            assert (
                frame["events"]["commit_time_s"] == frame["geometry"]["time_s"] == reading["time_s"]
            )
            assert len(row["q"]) == 8 and len(row["v"]) == 7
            assert len(row["full_q"]) == 9 and len(row["full_v"]) == 8
            assert len(row["contact_effort"]) == 7 and len(row["contact_effort_full"]) == 8
            assert row["full_q"][-1] == pytest.approx(-row["full_q"][-2] + 0.004, abs=STATE_ATOL)
            assert row["full_v"][-1] == pytest.approx(-row["full_v"][-2], abs=STATE_ATOL)
            receipt = row["mimic_loads"]
            assert receipt is not None
            np.testing.assert_allclose(
                (end_v - entry_v) / h, row["acceleration_full"], atol=1e-9, rtol=0
            )
            np.testing.assert_allclose(
                receipt["acceleration_full"], row["acceleration_full"], atol=1e-9, rtol=0
            )
            np.testing.assert_allclose(frame["full_balance_residual"], 0, atol=1e-8, rtol=0)
            np.testing.assert_allclose(
                receipt["constraint_effort_full"][:6], 0, atol=STATE_ATOL, rtol=0
            )
            assert receipt["constraint_power_W"] == pytest.approx(
                np.dot(receipt["constraint_effort_full"], entry_v), abs=1e-10
            )
            assert abs(receipt["constraint_power_W"]) <= 1e-10
            assert len(receipt["row_forces"]) == 1
            assert abs(receipt["row_forces"][0]["constraint_power_W"]) <= 1e-10
            np.testing.assert_allclose(
                receipt["contact_effort_full"], row["contact_effort_full"], atol=FORCE_ATOL, rtol=0
            )
            np.testing.assert_array_equal(row["drive_effort"], [0.0])
            np.testing.assert_array_equal(receipt["actuator_effort_full"], np.zeros(8))
            full_contact = np.asarray(row["contact_effort_full"])
            np.testing.assert_allclose(
                row["contact_effort"],
                np.r_[full_contact[:6], full_contact[6] - full_contact[7]],
                atol=FORCE_ATOL,
                rtol=0,
            )
            force = np.asarray(reading["loads"]["inputs"]["rigid"]["force"][0])
            torque = np.asarray(reading["loads"]["inputs"]["rigid"]["torque"][0])
            wrenches = np.asarray(row["link_wrenches_world"])
            impulse += h * force
            np.testing.assert_allclose(frame["ball_impulse_ns"], impulse, atol=1e-10, rtol=0)
            np.testing.assert_allclose(reading["rigid"]["velocity"][0], impulse, atol=1e-10, rtol=0)
            np.testing.assert_allclose(
                wrenches[:, :3].sum(axis=0) + force, 0, atol=FORCE_ATOL, rtol=0
            )
            total_moment = np.cross(frame["entry_geometry"]["com_position"], wrenches[:, :3]).sum(
                axis=0
            ) + wrenches[:, 3:].sum(axis=0)
            total_moment += np.cross(reading["entry_rigid"]["position"][0], force) + torque
            np.testing.assert_allclose(total_moment, 0, atol=FORCE_ATOL, rtol=0)
            power = (
                full_contact @ entry_v
                + force @ reading["entry_rigid"]["velocity"][0]
                + torque @ reading["entry_rigid"]["angular_velocity"][0]
            )
            assert power == pytest.approx(
                sum(site["power_w"] for site in example.sites(frame)), abs=1e-9
            )
            for site in example.sites(frame):
                np.testing.assert_allclose(
                    site["application_point_first_m"],
                    site["application_point_second_m"],
                    atol=STATE_ATOL,
                    rtol=0,
                )
            for record in frame["events"]["active"]:
                owners = (record["first_owner"], record["second_owner"])
                assert {owner["body"] for owner in owners} == {ROBOT, BALL}
                assert all(owner["world"] == WORLD for owner in owners)
                for owner in owners:
                    if owner["body"] == ROBOT:
                        assert owner["kind"] == "robot_link" and owner["link_index"] == 2
                for component in record["components"]:
                    assert len(component["material"]) == len(component["load"]["sites"])
        initial_v = np.asarray(example.robot_reading(frames[0])["entry_v"])
        assert (
            max(
                np.linalg.norm(np.asarray(example.robot_reading(frame)["v"])[:6] - initial_v[:6])
                for frame in frames
            )
            > 1e-5
        )
        assert max(abs(example.robot_reading(frame)["v"][-1]) for frame in frames) > 1e-7
        # Force-level common-point moment cancellation is checked above. Actual
        # finite-step angular momentum is retained separately, not declared zero.
        assert np.isfinite(summary["max_angular_residual_nms"])


def test_c02_h_halving_keeps_original_physical_task_with_frozen_engineering_bounds(contact_windows):
    example = _example()
    a, b = contact_windows["h"]["frames"][-1], contact_windows["half"]["frames"][-1]
    for field in ("com_position", "position"):
        assert (
            np.max(np.abs(np.asarray(a["end_geometry"][field]) - b["end_geometry"][field])) <= 5e-4
        )
    assert (
        np.max(
            np.abs(
                np.asarray(a["reading"]["rigid"]["position"]) - b["reading"]["rigid"]["position"]
            )
        )
        <= 5e-4
    )
    assert (
        np.max(
            np.abs(
                np.asarray(a["end_geometry"]["com_velocity"]) - b["end_geometry"]["com_velocity"]
            )
        )
        <= 5e-3
    )
    assert (
        np.max(
            np.abs(
                np.asarray(a["reading"]["rigid"]["velocity"]) - b["reading"]["rigid"]["velocity"]
            )
        )
        <= 5e-3
    )
    assert (
        np.max(
            np.abs(
                np.asarray(example.robot_reading(a)["v"])[3:]
                - np.asarray(example.robot_reading(b)["v"])[3:]
            )
        )
        <= 1e-2
    )
    assert np.linalg.norm(np.asarray(a["ball_impulse_ns"]) - b["ball_impulse_ns"]) <= 5e-3


def test_c03_reverse_registration_and_world_covariance_keep_all_physical_outputs(contact_windows):
    example = _example()
    shift = np.array([1 / 2, -1 / 4, 3 / 4])
    for a, b, c in zip(
        contact_windows["h"]["frames"],
        contact_windows["reverse"]["frames"],
        contact_windows["transform"]["frames"],
        strict=True,
    ):
        ra, rb, rc = (example.robot_reading(frame) for frame in (a, b, c))
        for name in ("q", "v", "full_q", "full_v", "contact_effort_full", "acceleration_full"):
            np.testing.assert_allclose(ra[name], rb[name], atol=1e-11, rtol=0)
        np.testing.assert_allclose(
            np.asarray(a["end_geometry"]["com_position"]) @ CYCLE.T + shift,
            c["end_geometry"]["com_position"],
            atol=1e-10,
            rtol=0,
        )
        np.testing.assert_allclose(
            np.asarray(a["reading"]["rigid"]["position"]) @ CYCLE.T + shift,
            c["reading"]["rigid"]["position"],
            atol=1e-10,
            rtol=0,
        )
        for name in ("com_velocity", "angular_velocity"):
            np.testing.assert_allclose(
                np.asarray(a["end_geometry"][name]) @ CYCLE.T,
                c["end_geometry"][name],
                atol=1e-10,
                rtol=0,
            )
        for name in ("velocity", "angular_velocity"):
            np.testing.assert_allclose(
                np.asarray(a["reading"]["rigid"][name]) @ CYCLE.T,
                c["reading"]["rigid"][name],
                atol=1e-10,
                rtol=0,
            )
        np.testing.assert_allclose(
            np.asarray(ra["full_v"])[6:], np.asarray(rc["full_v"])[6:], atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(
            ra["mimic_loads"]["constraint_effort_full"][6:],
            rc["mimic_loads"]["constraint_effort_full"][6:],
            atol=1e-9,
            rtol=0,
        )
        assert (
            [event["kind"] for event in a["events"]["events"]]
            == [event["kind"] for event in b["events"]["events"]]
            == [event["kind"] for event in c["events"]["events"]]
        )
