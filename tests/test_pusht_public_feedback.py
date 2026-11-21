"""Portable causal checks for the real RR reference driver and public Scene.

These create fresh bounded controller histories from the catalog and shipped
entry vectors. They do not claim to reconstruct the archived route prefix.
Native FK/IK supplies references only; public Scene.step alone advances physics.
"""

import hashlib
import json
from dataclasses import asdict, replace
from shutil import copyfile
from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest
from test_manipulation_push_planner import load_driver

from experiments.area_pressure import pusht_public_feedback as feedback
from experiments.area_pressure.pusht_entry_rows import load_entries
from experiments.area_pressure.pusht_public_scene import (
    H,
    build_scene_spec,
    build_side_asset,
    legal_initial_q,
)
from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.backends.mujoco_tyx_state import MuJoCoTyxIntegrationState
from phymeo.control.manipulation_diagnostic import NativeJointPD
from phymeo.control.manipulation_push import enumerate_native_planar_push_contacts
from phymeo.training.manipulation_tasks import manipulation_scenes


@pytest.fixture
def case(tmp_path, monkeypatch):
    original = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
    model = mj.MjModel.from_xml_string(original.mjcf)
    query = enumerate_native_planar_push_contacts(
        model,
        model.qpos0,
        object_geom_names=original.object_geom_names,
        pusher_geom_name="pusher",
    )
    legacy = load_driver()
    saved = dict(summary=dict(model_sha256=legacy.model_digest(model)), query=asdict(query))

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU MuJoCo stepping is not a feedback or physics oracle")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mj, name, forbidden)
    return SimpleNamespace(
        original=original, model=model, legacy=legacy, saved=saved, path=tmp_path
    )


def _snapshot(model, q, v=None, *, tick=0, time=0.2, world=41):
    q = np.array(q, dtype=float, copy=True)
    v = np.zeros(model.nv) if v is None else np.array(v, dtype=float, copy=True)
    control = MuJoCoControl(
        np.zeros(model.nu), qfrc_applied=np.zeros(model.nv), xfrc_applied=np.zeros((model.nbody, 6))
    )
    return MuJoCoTyxIntegrationState(
        world,
        q,
        v,
        np.zeros(0),
        control.ctrl,
        time,
        tick,
        control.qfrc_applied,
        control.xfrc_applied,
        control,
        None,
    )


def _driver(case, name, state, start_tick, *, route_source=None):
    """Explicit new local controller history; never a physical q/v write."""
    driver = feedback.PushFeedbackDriver(case.original, case.model, case.path / name, case.saved)
    driver.tick = start_tick + state.tick
    driver.last_q = state.q.copy()
    driver.initial_cost = driver.last_cost = driver.cost(state.q)
    driver.monitor.start(driver.tick, driver.last_cost)
    driver.hold = state.q[driver.qa].copy()
    if route_source is not None:
        assert route_source.active is not None and route_source.routes
        driver.active = route_source.active
        driver.routes = [dict(row) for row in route_source.routes]
        driver.completed = route_source.completed
        driver.previous_direction = route_source.previous_direction
        driver.planner.ik_calls = route_source.planner.ik_calls
        driver.planner.path_calls = route_source.planner.path_calls
        # Preserve the genuine planner artifact and its existing SHA. Only the
        # local controller interval starts anew; the route is not fabricated.
        for row in driver.routes:
            source = route_source.output / row["file"]
            assert hashlib.sha256(source.read_bytes()).hexdigest() == row["sha256"]
            destination = driver.output / row["file"]
            copyfile(source, destination)
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == row["sha256"]
    pd = NativeJointPD(case.model, case.original.robot_joint_names, 25.0, 3.0)
    return driver, pd


def _plan(case, name="seed-route", q=None):
    state = _snapshot(case.model, case.model.qpos0 if q is None else q)
    driver, pd = _driver(case, name, state, 200)
    command, diagnostic = feedback.command_from_snapshot(driver, pd, 200, state, start_tick=200)
    assert driver.active is not None and not driver.stopped
    assert driver.active[1].status == "sampled_reference_ready"
    assert driver.planner.ik_calls > 0 and driver.planner.path_calls > 0
    return state, driver, pd, command, diagnostic


def _bias(model, q, v):
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:] = q, v
    mj.mj_forward(model, data)
    return data.qfrc_bias.copy()


def _assert_command(model, pd, command, diagnostic, state, absolute_tick):
    assert diagnostic["input_tick"] == absolute_tick
    assert diagnostic["local_tick"] == state.tick and diagnostic["input_time"] == state.time
    np.testing.assert_array_equal(diagnostic["measurement_q"], state.q)
    np.testing.assert_array_equal(diagnostic["measurement_v"], state.v)
    qa, va = list(pd.qpos_addresses), list(pd.dof_addresses)
    desired, desired_v = np.asarray(diagnostic["target_q"]), np.asarray(diagnostic["target_v"])
    raw = (
        _bias(model, state.q, state.v)[va]
        + 25 * (desired - state.q[qa])
        + 3 * (desired_v - state.v[va])
    )
    expected = np.zeros(model.nu)
    expected[list(pd.actuator_ids)] = np.clip(raw, -19.9, 19.9)
    np.testing.assert_array_equal(command.ctrl, expected.astype(np.float32))
    np.testing.assert_array_equal(command.qfrc_applied, np.zeros(model.nv))
    np.testing.assert_array_equal(command.xfrc_applied, np.zeros((model.nbody, 6)))
    for name in ("ctrl", "qfrc_applied", "xfrc_applied"):
        np.testing.assert_array_equal(
            diagnostic["generated_channels"][name], getattr(command, name)
        )
    assert tuple(diagnostic["pd"]["joint_names"]) == tuple(pd.joint_names)
    assert tuple(diagnostic["pd"]["qpos_addresses"]) == tuple(qa)
    assert tuple(diagnostic["pd"]["dof_addresses"]) == tuple(va)
    assert tuple(diagnostic["pd"]["actuator_ids"]) == tuple(pd.actuator_ids)


def test_feedback_command_uses_current_joint_state_and_rejects_wrong_binding(case):
    state, seed, pd, baseline, diagnostic = _plan(case)
    _assert_command(case.model, pd, baseline, diagnostic, state, 200)
    # The same real route is frozen here: isolate actual q/v feedback from replanning.
    changed_q, changed_v = state.q.copy(), state.v.copy()
    changed_q[pd.qpos_addresses[1]] += 1e-4
    changed_v[pd.dof_addresses[0]] += 1e-3
    changed = replace(state, q=changed_q, v=changed_v)
    driver, changed_pd = _driver(case, "joint-perturbation", changed, 200, route_source=seed)
    command, report = feedback.command_from_snapshot(
        driver, changed_pd, 200, changed, start_tick=200
    )
    _assert_command(case.model, changed_pd, command, report, changed, 200)
    np.testing.assert_array_equal(report["target_q"], diagnostic["target_q"])
    np.testing.assert_array_equal(report["target_v"], diagnostic["target_v"])
    bias_delta = (_bias(case.model, changed.q, changed.v) - _bias(case.model, state.q, state.v))[:2]
    expected_delta = bias_delta - np.array([0.003, 0.0025])
    np.testing.assert_allclose(command.ctrl - baseline.ctrl, expected_delta, rtol=2e-6, atol=2e-9)
    assert np.all(command.ctrl < baseline.ctrl)
    # Object XY cannot change this active route's target, and the free object
    # is a separate dynamic tree. Replanning is the object-pose feedback branch.
    object_q = state.q.copy()
    free = int(case.model.jnt_qposadr[case.model.joint("object-free").id])
    object_q[free + 1] += 0.001
    moved = replace(state, q=object_q)
    object_driver, object_pd = _driver(case, "active-object-probe", moved, 200, route_source=seed)
    object_command, object_report = feedback.command_from_snapshot(
        object_driver, object_pd, 200, moved, start_tick=200
    )
    np.testing.assert_array_equal(object_command.ctrl, baseline.ctrl)
    np.testing.assert_array_equal(object_report["target_q"], diagnostic["target_q"])
    before = (driver.tick, driver.last_q.copy(), driver.monitor.history, tuple(driver.routes))
    for tick, candidate in ((201, changed), (200, replace(changed, tick=1)), (200, state)):
        with pytest.raises(ValueError):
            feedback.command_from_snapshot(driver, changed_pd, tick, candidate, start_tick=200)
        assert driver.tick == before[0] and driver.monitor.history == before[2]
        np.testing.assert_array_equal(driver.last_q, before[1])
        assert tuple(driver.routes) == before[3]


def test_feedback_boundary_transports_current_object_pose_under_the_same_goal(case):
    model = case.model
    free = int(model.jnt_qposadr[model.joint("object-free").id])
    base = model.qpos0.copy()
    yaw, delta = 0.1, 0.001
    base[free + 3 : free + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0])  # Body-local +Y in world axes.
    cases = []
    for sign, name in ((-1, "object-minus"), (1, "object-plus")):
        q = base.copy()
        q[free : free + 3] += sign * delta * axis
        state = _snapshot(model, q)
        driver, pd = _driver(case, name, state, 200)
        seen = []
        original_plan = driver.planner.plan

        def plan(source_q, proposals, *, previous_direction=None):
            result = original_plan(source_q, proposals, previous_direction=previous_direction)
            seen.append((source_q.copy(), proposals, result))
            return result

        driver.planner.plan = plan  # Observe the real planner; never replace its result.
        command, report = feedback.command_from_snapshot(driver, pd, 200, state, start_tick=200)
        assert len(seen) == 1 and seen[0][2].status == "sampled_reference_ready"
        np.testing.assert_array_equal(seen[0][0], state.q)
        _assert_command(model, pd, command, report, state, 200)
        cases.append((state, driver, pd, seen[0][1], seen[0][2], report))
    left, right = cases
    assert left[1].goal == right[1].goal and left[1].target == right[1].target
    assert left[5]["goal"] == right[5]["goal"]
    left_proposals = {(p.index, p.distance_ordinal): p for p in left[3]}
    right_proposals = {(p.index, p.distance_ordinal): p for p in right[3]}
    assert left_proposals.keys() == right_proposals.keys()
    target = np.asarray(left[1].target.position[:2])
    tolerance = left[1].goal.position_tolerance_m
    expected_cost_change = -4 * delta * ((target - base[free : free + 2]) @ axis[:2]) / tolerance**2
    assert expected_cost_change < 0  # This fixed target lies along body-local +Y.
    assert right[1].cost(right[0].q) - left[1].cost(left[0].q) == pytest.approx(
        expected_cost_change, abs=2e-11
    )
    for key, a in left_proposals.items():
        b = right_proposals[key]
        assert a.input_binding != b.input_binding
        np.testing.assert_allclose(
            np.asarray(b.point_world) - a.point_world, 2 * delta * axis, atol=2e-14
        )
        np.testing.assert_allclose(
            np.asarray(b.precontact_center_world) - a.precontact_center_world,
            2 * delta * axis,
            atol=2e-14,
        )
        np.testing.assert_array_equal(a.direction_world, b.direction_world)
        assert b.cost_before - a.cost_before == pytest.approx(expected_cost_change, abs=2e-11)
        expected_improvement = (
            -4 * delta * (axis[:2] @ np.asarray(a.predicted_translation_xy)) / tolerance**2
        )
        assert b.predicted_improvement - a.predicted_improvement == pytest.approx(
            expected_improvement, abs=3e-11
        )
    # At the boundary both routes start from the same robot q. Compare the
    # genuinely replanned references midway through the first .25 s phase.
    controls = []
    for state, driver, pd, _, route, _ in cases:
        desired, desired_v, _ = case.legacy.reference(route.nodes, 0.125)
        control, _ = pd.command(state.q, state.v, desired, desired_v)
        controls.append(control)
    assert np.max(np.abs(controls[1] - controls[0])) > 1e-6


def _accept(driver, start_tick, before, after, model):
    receipt = after.last_step
    (pair,) = receipt.raw_contact_fields["area_friction"]
    assert tuple(pair["body_ids"]) == (model.body("tool").id, model.body("object").id)
    np.testing.assert_array_equal(receipt.input_q, before.q)
    np.testing.assert_array_equal(receipt.input_v, before.v)
    assert receipt.input_time == before.time and receipt.output_time == after.time
    measurement = SimpleNamespace(
        input_tick=start_tick + before.tick,
        output_tick=start_tick + after.tick,
        group_force_world=pair["force_on_b"].sum(axis=0),
    )
    result = driver.accepted_measurement(
        start_tick + after.tick,
        after.q,
        measurement,
        SimpleNamespace(success=False),
        input_q=receipt.input_q,
    )
    np.testing.assert_array_equal(driver.last_q, after.q)
    assert driver.tick == start_tick + after.tick
    assert result["current_cost"] == driver.cost(after.q)
    return result, measurement


@pytest.mark.cuda
def test_public_feedback_reads_the_last_accepted_endpoint_against_held_control(case, tmp_path):
    _, seed, _, _, _ = _plan(case)
    entry = load_entries()["peak"]
    model, start_tick = case.model, entry["tick"]
    q = legal_initial_q(model, entry["q"])
    spec, asset = build_scene_spec(case.original.mjcf), build_side_asset(model)
    keys = (41, 305)
    worlds = tuple(
        MuJoCoTyxWorldSpec(key, spec.key, MuJoCoInitialState(q, entry["v"], time=entry["time"]))
        for key in keys
    )
    records = []
    with MuJoCoTyxHeterogeneousBatch((spec,), worlds, mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (asset,)})
        initial = batch.integration_snapshot()
        controllers = {
            key: _driver(case, f"lane-{key}", initial[key], start_tick, route_source=seed)
            for key in keys
        }
        first_commands = {}
        for key in keys:
            driver, pd = controllers[key]
            command, diagnostic = feedback.command_from_snapshot(
                driver, pd, start_tick, initial[key], start_tick=start_tick
            )
            _assert_command(model, pd, command, diagnostic, initial[key], start_tick)
            first_commands[key] = command
        np.testing.assert_array_equal(first_commands[keys[0]].ctrl, first_commands[keys[1]].ctrl)
        for step in range(3):
            before = batch.integration_snapshot()
            commands, diagnostics = {}, {}
            for key in keys:
                driver, pd = controllers[key]
                current, diagnostic = feedback.command_from_snapshot(
                    driver, pd, start_tick + step, before[key], start_tick=start_tick
                )
                _assert_command(model, pd, current, diagnostic, before[key], start_tick + step)
                commands[key] = current if key == keys[0] else first_commands[key]
                diagnostics[key] = diagnostic
                if step:
                    np.testing.assert_array_equal(diagnostic["measurement_v"], before[key].v)
                    assert not np.array_equal(before[key].v, before[key].last_step.input_v)
            if step == 1:
                assert np.max(np.abs(commands[keys[0]].ctrl - commands[keys[1]].ctrl)) > 1e-7
                np.testing.assert_array_equal(
                    diagnostics[keys[0]]["generated_channels"]["ctrl"],
                    diagnostics[keys[1]]["generated_channels"]["ctrl"],
                )
            batch.set_controls(commands)
            for key in keys:
                np.testing.assert_array_equal(batch.integration_snapshot()[key].q, before[key].q)
            assert scene.step(H, expected_revision=batch.checkpoint_revision) == dict.fromkeys(
                keys, True
            )
            after = batch.integration_snapshot()
            for key in keys:
                driver, _ = controllers[key]
                result, measurement = _accept(driver, start_tick, before[key], after[key], model)
                receipt = after[key].last_step
                (pair,) = receipt.raw_contact_fields["area_friction"]
                assert pair["normal_force"].sum() > 0 and any(
                    row.force > 0 for row in receipt.constraints
                )
                np.testing.assert_array_equal(after[key].held_control.ctrl, commands[key].ctrl)
                assert np.isfinite(after[key].q).all() and np.isfinite(after[key].v).all()
                records.append(
                    dict(
                        world=key,
                        step=step + 1,
                        input_q=before[key].q.tolist(),
                        input_v=before[key].v.tolist(),
                        ctrl=commands[key].ctrl.tolist(),
                        q=after[key].q.tolist(),
                        v=after[key].v.tolist(),
                        cost=result["current_cost"],
                        object_force_world=measurement.group_force_world.tolist(),
                    )
                )
            if step == 0:
                np.testing.assert_array_equal(after[keys[0]].q, after[keys[1]].q)
                np.testing.assert_array_equal(after[keys[0]].v, after[keys[1]].v)
                np.testing.assert_array_equal(
                    after[keys[0]].last_step.qfrc_constraint,
                    after[keys[1]].last_step.qfrc_constraint,
                )
                assert after[keys[0]].time == after[keys[1]].time
            else:
                assert np.max(np.abs(after[keys[0]].v - after[keys[1]].v)) > 1e-10
            (tmp_path / "feedback-short.json").write_text(
                json.dumps(records, indent=2, allow_nan=False), encoding="utf-8"
            )
