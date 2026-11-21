"""Fresh lateral-material episode boundaries; synthetic records are not physics evidence."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.area_pressure.pusht_lateral_route import (
    accumulate_measurement,
    bind_fresh_snapshot,
    bootstrap_driver,
    initial_load_assessment,
)


def source_state():
    q = np.array([0.1, -0.2, 0.52, 0.02, 0.015, 1, 0, 0, 0], np.float32)
    v = np.array([0.03, -0.04, 0.005, 0, 0, 0, 0, 0.02], np.float32)
    return dict(
        local_tick=2750,
        time=4.4500002,
        q=q.tolist(),
        v=v.tolist(),
        object_position_world_m=q[2:5].tolist(),
        object_quaternion_wxyz=q[5:9].tolist(),
    )


def snapshot(state=None):
    state = source_state() if state is None else state
    return SimpleNamespace(
        tick=0,
        time=state["time"],
        q=np.asarray(state["q"], np.float32),
        v=np.asarray(state["v"], np.float32),
        last_step=None,
    )


def test_new_material_owner_preserves_saved_qv_bytes_without_inheriting_receipt():
    state = source_state()
    proof = bind_fresh_snapshot(snapshot(state), state)
    assert proof["q_fp32_bytes_equal"] and proof["v_fp32_bytes_equal"]
    assert proof["source_local_tick"] == 2750 and proof["new_local_tick"] == 0
    assert proof["previous_receipt_inherited"] is False


@pytest.mark.parametrize("fault", ["q", "v", "time", "receipt", "tick"])
def test_new_episode_cannot_silently_change_state_or_reuse_old_receipt(fault):
    saved = source_state()
    value = snapshot(saved)
    if fault in ("q", "v"):
        field = getattr(value, fault)
        field[0] = np.nextafter(field[0], np.float32(np.inf))
    elif fault == "time":
        value.time += 0.001
    elif fault == "tick":
        value.tick = 2750
    else:
        value.last_step = object()
    with pytest.raises(ValueError):
        bind_fresh_snapshot(value, saved)


def fresh_driver():
    starts = []
    monitor = SimpleNamespace(
        started=False, history=(), start=lambda tick, cost: starts.append((tick, cost))
    )
    driver = SimpleNamespace(
        active=None,
        completed=0,
        total_impulse=0.0,
        segment_impulse=0.0,
        routes=[],
        previous_direction=None,
        stopped=False,
        reason="continue",
        tick=0,
        monitor=monitor,
        cost=lambda q: 12.0,
        planner=SimpleNamespace(ik_calls=0, path_calls=0, max_ik=512, max_paths=128),
    )
    return driver, starts


def test_fresh_controller_inherits_only_budget_and_reference_clock_not_old_history():
    driver, starts = fresh_driver()
    state = source_state()
    proof = bootstrap_driver(driver, state, 4450, (27, 12))
    assert starts == [(4450, 12.0)]
    assert driver.tick == 4450 and driver.completed == 0 and driver.routes == []
    np.testing.assert_array_equal(driver.last_q, state["q"])
    assert (driver.planner.ik_calls, driver.planner.path_calls) == (27, 12)
    assert driver.total_impulse == driver.segment_impulse == 0
    assert proof["goal_history_inherited"] is False
    assert proof["source_force_history_inherited"] is False
    assert proof["source_budget_counts"] == [27, 12]


def test_existing_controller_history_cannot_be_cleared_and_rebranded_fresh():
    driver, _ = fresh_driver()
    driver.total_impulse = 0.25
    with pytest.raises(ValueError):
        bootstrap_driver(driver, source_state(), 4450, (27, 12))


def zero_pair(identity):
    return dict(
        identity=identity,
        dA=np.zeros(0),
        normal_force=np.zeros(0),
        force_on_a=np.empty((0, 3)),
        force_on_b=np.empty((0, 3)),
    )


def test_first_real_step_checks_new_identity_and_actual_zero_force_before_selecting():
    identity = ("new", "all-lateral")
    geometry = SimpleNamespace(surface=SimpleNamespace(identity=identity, triangles=()))
    pair = zero_pair(identity)
    result = initial_load_assessment(pair, geometry, identity)
    assert result["safe"] is True
    pair["normal_force"] = np.array([1.0])
    assert initial_load_assessment(pair, geometry, identity)["safe"] is False
    pair = zero_pair(("old", "negative-y"))
    with pytest.raises(ValueError):
        initial_load_assessment(pair, geometry, identity)


def test_angular_impulse_uses_real_accepted_input_moment_and_dt_once():
    before, after = dict(time=4.45), dict(time=4.451)
    measurement = dict(
        input_tick=4450,
        output_tick=4451,
        time_s=4.45,
        phase="accepted-step-input",
        group_force_world=[2.0, 0.0, 0.0],
        group_moment_com_world=[0.0, 0.0, 0.3],
    )
    value = accumulate_measurement({}, before, after, measurement, 4450)
    assert value["object_angular_impulse_z_nms"] == pytest.approx(0.0003)
    assert value["object_force_impulse_world_ns"] == pytest.approx([0.002, 0, 0])
    bad = deepcopy(measurement)
    bad["time_s"] = after["time"]
    with pytest.raises(ValueError):
        accumulate_measurement({}, before, after, bad, 4450)


def test_new_episode_allows_only_public_normalization_of_real_saved_free_quaternion():
    from experiments.area_pressure.pusht_public_scene import legal_initial_q

    saved = source_state()
    saved["q"][5:9] = [
        0.9984166026115417,
        -1.801822264724251e-7,
        1.3197508863527219e-8,
        -0.05625271424651146,
    ]
    model = SimpleNamespace(
        nq=9, jnt_qposadr=np.array([2]), joint=lambda name: SimpleNamespace(id=0)
    )
    declared = legal_initial_q(model, saved["q"])
    actual = snapshot(saved)
    actual.q = declared.astype(np.float32)
    proof = bind_fresh_snapshot(actual, saved, quaternion_address=5)
    assert proof["initialization_policy"] == "new-episode-public-free-quaternion-normalization"
    assert proof["nonrotation_q_fp32_bytes_equal"] and proof["v_fp32_bytes_equal"]
    assert proof["normalized_quaternion_fp32_bytes_equal"]
    assert proof["source_quaternion_norm"] != 1.0
    assert proof["source_q"] == saved["q"]
    assert proof["published_q"] == actual.q.tolist()
    assert proof["quaternion_changed"]
    assert proof["quaternion_angle_change_rad"] >= 0.0


@pytest.mark.parametrize("fault", ["rotation", "sign", "robot", "translation", "velocity", "time"])
def test_new_episode_normalization_does_not_authorize_other_pose_or_state_changes(fault):
    saved = source_state()
    saved["q"][5:9] = [
        0.9984166026115417,
        -1.801822264724251e-7,
        1.3197508863527219e-8,
        -0.05625271424651146,
    ]
    actual = snapshot(saved)
    quat = np.asarray(saved["q"][5:9], np.float64)
    actual.q[5:9] = (quat / np.linalg.norm(quat)).astype(np.float32)
    if fault == "rotation":
        actual.q[8] = np.nextafter(actual.q[8], np.float32(np.inf))
    elif fault == "sign":
        actual.q[5:9] *= -1
    elif fault in ("robot", "translation"):
        index = 0 if fault == "robot" else 2
        actual.q[index] = np.nextafter(actual.q[index], np.float32(np.inf))
    elif fault == "velocity":
        actual.v[0] = np.nextafter(actual.v[0], np.float32(np.inf))
    else:
        actual.time = np.nextafter(actual.time, np.inf)
    with pytest.raises(ValueError):
        bind_fresh_snapshot(actual, saved, quaternion_address=5)


def test_public_fp64_snapshot_container_preserves_exact_fp32_values_and_signed_zero():
    saved = source_state()
    saved["q"][0] = -0.0
    saved["v"][0] = -0.0
    value = snapshot(saved)
    value.q, value.v = value.q.astype(np.float64), value.v.astype(np.float64)
    proof = bind_fresh_snapshot(value, saved, quaternion_address=5)
    assert proof["q_fp32_bytes_equal"] and proof["v_fp32_bytes_equal"]
    assert proof["snapshot_q_dtype"] == proof["snapshot_v_dtype"] == "float64"
    assert proof["snapshot_q_lossless_fp32"] and proof["snapshot_v_lossless_fp32"]
    assert proof["quaternion_changed"] is False


@pytest.mark.parametrize("field", ["q", "v"])
def test_fp64_container_one_ulp_hidden_by_fp32_cast_is_rejected(field):
    saved = source_state()
    value = snapshot(saved)
    value.q, value.v = value.q.astype(np.float64), value.v.astype(np.float64)
    changed = getattr(value, field)
    changed[0] = np.nextafter(changed[0], np.inf)
    with pytest.raises(ValueError, match=field + ".*lossless FP32"):
        bind_fresh_snapshot(value, saved, quaternion_address=5)


@pytest.mark.parametrize("field", ["q", "v"])
def test_public_fp64_container_cannot_flip_source_signed_zero(field):
    saved = source_state()
    saved[field][0] = -0.0
    value = snapshot(saved)
    value.q, value.v = value.q.astype(np.float64), value.v.astype(np.float64)
    getattr(value, field)[0] = 0.0
    with pytest.raises(ValueError, match=field + ".*FP32 bytes"):
        bind_fresh_snapshot(value, saved, quaternion_address=5)


def test_new_episode_measurement_uses_local_goal_clock_and_absolute_force_clock(monkeypatch):
    from experiments.area_pressure import pusht_lateral_route as runner
    from experiments.area_pressure import pusht_public_feedback as feedback

    calls = []
    result = (object(), object(), object())

    def measure(scene, model, world, absolute_tick, before, after, history, *, goal_tick):
        calls.append((absolute_tick, goal_tick, before.time, history))
        return result

    monkeypatch.setattr(feedback, "accepted_measurement_and_goal", measure)
    entry = SimpleNamespace(scene=object(), model=object(), tick=4450)
    before = SimpleNamespace(tick=0, time=4.45)
    after = SimpleNamespace(tick=1, time=4.451)
    assert runner.episode_measurement(entry, 6, before, after, None) is result
    history = object()
    assert runner.episode_measurement(entry, 6, after, SimpleNamespace(tick=2), history) is result
    assert calls == [(4450, 0, 4.45, None), (4451, 1, 4.451, history)]
