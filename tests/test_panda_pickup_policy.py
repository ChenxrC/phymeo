"""Fixed-fork reset geometry, pickup checkpoints and motor-entry regressions."""

from importlib import import_module
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return import_module("_panda_pickup_resets"), import_module("_panda_pickup_policy")


@pytest.fixture
def native(modules):
    mj = pytest.importorskip("mujoco")
    scene = import_module("_panda_grasp_insert_scene").build_scene()
    return mj, scene, mj.MjModel.from_xml_string(scene.xml), import_module("_panda_grasp_insert_control")


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def test_cases_are_reproducible_and_splits_disjoint(modules):
    resets, _ = modules
    train, holdout = resets.reset_bank("train"), resets.reset_bank("holdout")
    assert (len(train), len(holdout)) == (6, 4)
    for split, cases in (("train", train), ("holdout", holdout)):
        assert cases == resets.reset_bank(split)
        assert all(case["split"] == split for case in cases)
        for key in ("cross_m", "along_m", "yaw_deg"):
            assert min(case[key] for case in cases) < 0. < max(case[key] for case in cases)
        assert all(abs(case["along_m"]) > 3.*abs(case["cross_m"]) for case in cases)
    cases = train + holdout
    assert len({case["seed"] for case in cases}) == len(cases)
    assert len({case["id"] for case in cases}) == len(cases)
    assert len({(case["cross_m"], case["along_m"], case["yaw_deg"]) for case in cases}) == len(cases)
    for first in train:
        for second in holdout:
            centers = [a["stratum_center_cross_mm_along_mm_yaw_deg"][2] for a in (first, second)]
            widths = [a["uniform_halfwidth_cross_mm_along_mm_yaw_deg"][2] for a in (first, second)]
            assert abs(centers[0] - centers[1]) > sum(widths)
    for cross, along in ((.003, 0.), (0., -.040)):
        with pytest.raises(ValueError):
            resets.validate_reset(dict(cross_m=cross, along_m=along, yaw_deg=25.))


def test_native_resets_turn_the_free_part_and_support_both_shoulders(native, modules):
    mj, scene, model, _ = native
    resets, _ = modules
    data = mj.MjData(model)
    nominal = scene.initial_state()
    data.qpos[:] = nominal.q
    mj.mj_kinematics(model, data)
    left, right = [model.geom(name).id for name in ("pickup_rail_left", "pickup_rail_right")]
    shoulder, shaft = [model.geom(name).id for name in ("part_shoulder", "assembly_pin")]
    pin = model.body("assembly_part").id
    part_origin, home_rotation = data.xpos[pin].copy(), data.xmat[pin].reshape(3, 3).copy()
    rail_rotation = data.geom_xmat[left].reshape(3, 3).copy()
    rail_origins = data.geom_xpos[[left, right]].copy()
    rail_top = rail_origins[0, 2] + model.geom_size[left, 2]
    np.testing.assert_allclose(model.geom_size[[left, right]], [[.0045, .020, .010]]*2)
    np.testing.assert_allclose(model.geom_size[shoulder], [.015, .015, .003])
    assert model.geom_size[shaft, 0] == pytest.approx(.004)
    assert np.linalg.norm(rail_origins[1] - rail_origins[0]) == pytest.approx(.022)

    for case in resets.reset_bank("train") + resets.reset_bank("holdout"):
        state = resets.initial_state(scene, case)
        np.testing.assert_array_equal(state.q[:16], nominal.q[:16])
        np.testing.assert_array_equal(state.v, nominal.v)
        assert state.time == nominal.time and state.q[18] == nominal.q[18]
        data.qpos[:] = state.q
        mj.mj_kinematics(model, data)
        measured_xyz = data.xpos[pin] - part_origin
        np.testing.assert_allclose(rail_rotation.T @ measured_xyz,
                                   [case["cross_m"], case["along_m"], 0.], atol=2e-14, rtol=0.)
        np.testing.assert_allclose(data.xmat[pin].reshape(3, 3),
                                   _rot_z(np.deg2rad(case["yaw_deg"])) @ home_rotation, atol=2e-14)
        np.testing.assert_array_equal(data.geom_xpos[[left, right]], rail_origins)
        assert data.geom_xpos[shoulder, 2] - model.geom_size[shoulder, 2] == pytest.approx(rail_top, abs=2e-14)

        # Reconstruct the witness corners from actual world transforms and
        # verify containment in both solids. Use actual compiled capsule size.
        center = rail_origins.mean(axis=0)
        rail_x = rail_rotation[:, 0]
        rail_y = rail_rotation[:, 1]
        actual_clearances = []
        for sign, rail in ((-1., left), (1., right)):
            axis_distance = abs(rail_x @ (data.geom_xpos[shaft] - data.geom_xpos[rail]))
            actual_clearances.append(axis_distance - model.geom_size[rail, 0] - model.geom_size[shaft, 0])
            for x in (.007, .010):
                for y in (-.003, .003):
                    corner = center + sign*x*rail_x + (case["along_m"] + y)*rail_y
                    corner[2] = rail_top
                    for geom in (rail, shoulder):
                        local = data.geom_xmat[geom].reshape(3, 3).T @ (corner - data.geom_xpos[geom])
                        assert np.all(np.abs(local[:2]) < model.geom_size[geom, :2])
                        assert abs(local[2]) == pytest.approx(model.geom_size[geom, 2], abs=2e-14)
        assert min(actual_clearances) > .001
        np.testing.assert_allclose(actual_clearances, case["geometry"]["shaft_rail_clearance_left_right_m"],
                                   atol=2e-14, rtol=0.)
        assert case["geometry"]["guaranteed_patch_area_per_rail_m2"] == pytest.approx(.003*.006)


@pytest.mark.parametrize("yaw", [-.63, .63])
def test_full_correction_uses_signed_world_yaw_and_preserves_height_and_tilt(modules, yaw):
    _, policy = modules
    nominal = np.eye(4)
    nominal[:3, :3] = _rot_z(np.deg2rad(172.)) @ np.diag([1., -1., -1.])
    nominal[:3, 3] = [.3, -.2, .4]
    expected = _rot_z(yaw) @ nominal[:3, :3]
    observed = nominal.copy()
    # Roll about the observed local X axis preserves its heading.
    c, s = np.cos(.09), np.sin(.09)
    observed[:3, :3] = expected @ np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    observed[:3, 3] += [.006, -.014, .003]
    assert policy.observed_yaw_delta(nominal, observed) == pytest.approx(yaw)
    result = policy.PickupPolicy((1., 1.)).target(nominal, observed)
    np.testing.assert_allclose(result[:3, :3], expected, atol=2e-14, rtol=0.)
    np.testing.assert_array_equal(result[:2, 3], observed[:2, 3])
    assert result[2, 3] == nominal[2, 3]


def test_checkpoint_roundtrip_rejects_ambiguous_or_changed_contracts(modules, tmp_path):
    _, policy = modules
    original = policy.PickupPolicy((-.125, .8125))
    path = tmp_path / "policy.json"
    original.save(path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt == original.to_checkpoint()
    assert policy.PickupPolicy.load(path) == original
    with pytest.raises(FileExistsError):
        original.save(path)
    for change in (dict(schema="v0"), dict(parameter_names=receipt["parameter_names"][::-1]),
                   dict(observation="reset-label"), dict(target="local-yaw"), dict(extra=True),
                   dict(parameters=[1.]), dict(parameters=[True, 1.]), dict(parameters=["0.5", 1.]),
                   dict(parameters=[0., float("nan")])):
        with pytest.raises((ValueError, TypeError)):
            policy.PickupPolicy.from_checkpoint({**receipt, **change})
    with pytest.raises(ValueError):
        policy.PickupPolicy.from_checkpoint({k: v for k, v in receipt.items() if k != "target"})
    path.write_text('{"schema": "shadowed", ' + json.dumps(receipt)[1:], encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        policy.PickupPolicy.load(path)


def test_zero_policy_replays_default_targets_clock_and_motor_output(native, modules):
    _, scene, model, control = native
    resets, policy = modules
    initial = resets.initial_state(scene, resets.reset_bank("train")[0])
    original = control.GraspInsertController(scene, model)
    zero = control.GraspInsertController(scene, model, policy=policy.PickupPolicy((0., 0.)))
    # Short input tape exercises each distinct phase through the real motor
    # entry point. The off-center start makes accidental policy gains visible.
    tape = [(0., "settle"), (.499, "settle"), (.5, "approach"), (1.25, "approach"),
            (2.5, "descend"), (3.5, "close"), (4.6, "lift"), (6.2, "transfer"),
            (7.95, "insert"), (9.2, "seat"), (10.1, "release"), (11.25, "clear"), (12., "complete")]
    for t, phase in tape:
        old = original.advance(initial.q, initial.v, t)
        new = zero.advance(initial.q, initial.v, t)
        assert old.phase == new.phase == phase
        assert new.diagnostics["time_s"] == old.diagnostics["time_s"] == t
        for key in ("phase_fraction", "planned_tcp_world", "feedback_tcp_world", "arm_target_q", "jaw_target_m"):
            np.testing.assert_array_equal(new.diagnostics[key], old.diagnostics[key])
        assert old.ctrl.shape == new.ctrl.shape == (16,)
        np.testing.assert_array_equal(new.ctrl, old.ctrl)
        if phase == "close":
            assert old.diagnostics["phase_fraction"] == pytest.approx(.5)
            np.testing.assert_allclose(old.diagnostics["jaw_target_m"], [.0175]*2, atol=1e-14)
            np.testing.assert_array_equal(old.ctrl[14:], [-20., -20.])
            np.testing.assert_allclose(old.diagnostics["planned_tcp_world"], original.start_part, atol=1e-14)


def test_policy_latches_first_post_settle_measured_part_without_moving_state(native, modules):
    mj, scene, model, control = native
    resets, policy = modules
    controller = control.GraspInsertController(scene, model, policy=policy.PickupPolicy((1., 1.)))
    cases = resets.reset_bank("train")
    first, later = [resets.initial_state(scene, case) for case in (cases[0], cases[-1])]
    # A synthetic observation with a displacement distinct from both declared
    # resets makes reading the home key or original case metadata detectable.
    q = first.q.copy()
    q[16:19] += [.003, -.004, .001]
    unchanged = q.copy()
    data = mj.MjData(model)
    data.qpos[:] = q
    mj.mj_kinematics(model, data)
    site = model.site("part_grasp_center").id
    pose = np.eye(4)
    pose[:3, :3] = data.site_xmat[site].reshape(3, 3)
    pose[:3, 3] = data.site_xpos[site]
    assert controller.advance(later.q, later.v, .498).diagnostics["pickup_observation"] is None
    obs = controller.advance(q, first.v, .502).diagnostics["pickup_observation"]
    np.testing.assert_array_equal(q, unchanged)
    np.testing.assert_allclose(obs["observed_part_grasp_world"], pose, atol=2e-14, rtol=0.)
    assert obs["time_s"] == .502
    selected = np.array(obs["selected_grasp_world"])
    np.testing.assert_allclose(selected[:3, :3], pose[:3, :3], atol=2e-14, rtol=0.)
    np.testing.assert_array_equal(selected[:2, 3], pose[:2, 3])
    assert selected[2, 3] == pytest.approx(first.q[18], abs=2e-14)
    after = controller.advance(later.q, later.v, 3.5)
    assert after.diagnostics["pickup_observation"] == obs
    np.testing.assert_array_equal(after.diagnostics["planned_tcp_world"], selected)
