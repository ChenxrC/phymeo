"""Route orchestration with native scratch IK/path queries; no physical stepping."""

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest

from phymeo.control.manipulation_push_planner import NativePushRoutePlanner, orbit_waypoints


def model():
    return mj.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
    <worldbody><body name="tool" pos="0 0 1">
    <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
    <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
    <geom name="pusher" type="sphere" size=".02" mass="1"/>
    </body><body name="object" pos=".5 0 1"><freejoint/>
    <geom name="object-shape" type="box" size=".05 .05 .05" mass=".1"/>
    </body></worldbody></mujoco>""")


def proposal(index=0, pre=(0.4, 0, 1)):
    return SimpleNamespace(
        status="PROPOSAL_ONLY",
        reference_usable=True,
        predicted_improvement=1.0,
        index=index,
        input_binding="a" * 64,
        precontact_center_world=pre,
        contact_center_world=(0.43, 0, 1),
        push_center_world=(0.44, 0, 1),
        distance_m=0.01,
        mobility_radius_m=0.08,
        object_com_world=(0.5, 0, 1),
        direction_world=(1.0, 0.0, 0.0),
    )


def planner(m, **kwargs):
    return NativePushRoutePlanner(
        m, ("x", "y"), object_geom_names=("object-shape",), pusher_geom_name="pusher", **kwargs
    )


def test_native_complete_route_preserves_free_state_model_and_input(monkeypatch):
    m = model()
    q = m.qpos0.copy()
    before = q.tobytes()
    binary = np.empty(mj.mj_sizeModel(m), np.uint8)
    mj.mj_saveModel(m, None, binary)

    def forbidden(*args, **kwargs):
        raise AssertionError("no physical stepping")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mj, name, forbidden)
    p = planner(m)
    result = p.plan(q, (proposal(),))
    assert result.status == "sampled_reference_ready"
    assert result.ik_calls == 3 and result.path_calls == 3
    assert result.nodes[0][0] == 0 and result.nodes[-1][0] == pytest.approx(1.5)
    np.testing.assert_allclose(result.nodes[-1][2], (0.44, 0), rtol=0, atol=5e-4)
    assert q.tobytes() == before
    for _, audit in result.attempts[0]["paths"]:
        assert audit.sampled_clear
        for sample in audit.samples:
            assert sample.q[2:].tobytes() == q[2:].tobytes()
    after = np.empty_like(binary)
    mj.mj_saveModel(m, None, after)
    assert after.tobytes() == binary.tobytes()
    with pytest.raises(ValueError):
        result.source_q[0] = 1


def test_budget_charges_failed_attempt_and_never_resets_total():
    m = model()
    p = planner(m, max_ik_per_segment=1, max_ik=2)
    first = p.plan(m.qpos0, (proposal(),))
    assert first.status == "ik_budget" and first.ik_calls == 1 and first.path_calls == 0
    assert first.attempts[-1]["reason"] == "ik_budget"
    second = p.plan(m.qpos0, (proposal(),))
    assert second.status == "ik_budget" and second.ik_calls == 2
    third = p.plan(m.qpos0, (proposal(),))
    assert third.status == "ik_budget" and third.ik_calls == 2
    assert not third.nodes


def test_unreachable_seed_keeps_failure_and_does_not_run_paths():
    m = model()
    result = planner(m).plan(m.qpos0, (proposal(pre=(5, 0, 1)),))
    assert result.status == "no_qualified_route"
    assert result.ik_calls == 1 and result.path_calls == 0
    assert result.attempts[0]["reason"] == "precontact_ik"
    assert not result.attempts[0]["ik"][0][1].converged


def test_projection_changes_reference_only_and_is_explicit():
    m = model()
    q = m.qpos0.copy()
    q[0] = -1.001
    before = q.tobytes()
    result = planner(m, max_ik_per_segment=1).plan(q, (proposal(),))
    assert result.initial_reference_projection == pytest.approx((0.001, 0))
    assert result.source_q.tobytes() == before and q.tobytes() == before


def test_orbits_take_declared_direction_and_bounded_complete_arc():
    ccw = orbit_waypoints((0, 0), (1, 0), (0, 1), 2, False)
    cw = orbit_waypoints((0, 0), (1, 0), (0, 1), 2, True)
    assert len(ccw) == 3 and len(cw) == 7
    np.testing.assert_allclose(ccw, ((2, 0), (math.sqrt(2), math.sqrt(2)), (0, 2)), atol=1e-14)
    assert cw[1][1] < 0 and ccw[1][1] > 0
    for points in (ccw, cw):
        assert len(points) <= 9
        assert all(math.hypot(*p) == pytest.approx(2) for p in points)
        np.testing.assert_allclose(points[-1], (0, 2), atol=1e-14)


@pytest.mark.parametrize(
    "values",
    [
        ((0, 0), (0, 0), (1, 0), 1, False),
        ((0, 0), (1, 0), (0, 1), 0, False),
        ((0, 0), (1, 0), (0, 1), 1, 1),
        ((0, 0), (math.nan, 0), (0, 1), 1, False),
    ],
)
def test_invalid_orbit(values):
    with pytest.raises(ValueError):
        orbit_waypoints(*values)


def test_path_budget_stops_before_an_unchecked_route_can_be_selected():
    m = model()
    result = planner(m, max_paths=1).plan(m.qpos0, (proposal(),))
    assert result.status == "path_budget" and result.path_calls == 1
    assert result.proposal is None and not result.nodes


def load_driver():
    script = Path(__file__).parents[1] / "scripts" / "run_manipulation_push_feedback.py"
    spec = importlib.util.spec_from_file_location("test_feedback_runner", script)
    module = importlib.util.module_from_spec(spec)
    old = sys.path.copy()
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old
    return module


def test_feedback_binds_first_published_fp32_endpoint_and_checks_later_q():
    Driver = load_driver().PushFeedbackDriver
    from phymeo.control.manipulation_push_progress import BoundedPushProgress

    driver = object.__new__(Driver)
    driver.monitor = BoundedPushProgress()
    driver.tick = 0
    driver.last_q = np.array([0.52], np.float64)
    driver.hold = np.array([0.0])
    driver.cost = lambda q: float(q[0] ** 2)
    actual = np.array([0.52], np.float32)
    target, velocity, stage, _ = driver.reference(0, actual, np.zeros(1))
    assert stage == "settle" and target[0] == velocity[0] == 0
    assert np.array_equal(driver.last_q, actual) and driver.monitor.started
    with pytest.raises(ValueError, match="accepted endpoint"):
        driver.reference(0, np.array([0.53], np.float32), np.zeros(1))


@pytest.mark.parametrize(
    "complete,success,cap,expected",
    [
        (True, False, 100, "continue"),
        (False, False, 4, "step_budget"),
        (False, True, 100, "success"),
    ],
)
def test_driver_accepted_force_clock_and_partial_endpoint(
    complete, success, cap, expected, monkeypatch
):
    module = load_driver()
    from phymeo.control.manipulation_push_progress import BoundedPushProgress

    driver = object.__new__(module.PushFeedbackDriver)
    driver.model = SimpleNamespace(nq=1, body_ipos=np.zeros((1, 3)))
    driver.object_body, driver.pusher = 0, 1
    driver.tick = 3
    driver.last_q = np.array([0.52], np.float32)
    driver.pose = lambda q: (np.zeros(3), np.eye(3), np.zeros(3))
    driver.cost = lambda q: 9.0
    driver.monitor = BoundedPushProgress(max_steps=cap)
    driver.monitor.start(0, 10.0)
    driver.active = (
        0,
        SimpleNamespace(
            nodes=((0, "start", ()), ((0.004 if complete else 0.01), "end", ())),
            proposal=SimpleNamespace(direction_world=(1.0, 0.0, 0.0)),
        ),
    )
    driver.completed = 0
    driver.segment_impulse = 0.1
    driver.total_impulse = 0.2
    driver.previous_direction = None
    driver.stopped = False
    driver.reason = "continue"
    calls = []

    def measurement(arrays, row, **kwargs):
        calls.append((arrays, row, kwargs))
        return SimpleNamespace(input_tick=3, output_tick=4, group_force_world=(3.0, 4.0, 0.0))

    monkeypatch.setattr(module, "measure_object_contact_wrench", measurement)
    arrays = {"entry_q": driver.last_q[None, :].copy()}
    endpoint = np.array([0.521], np.float32)
    report = driver.accepted(4, endpoint, arrays, SimpleNamespace(success=success))
    assert calls and report["progress"].end_step == 4
    assert report["progress"].contact_impulse_ns == pytest.approx(0.105)
    assert report["progress"].status == expected
    assert report["completed_route"] == complete
    assert driver.completed == int(complete)
    assert np.array_equal(driver.last_q, endpoint) and driver.last_q is not endpoint
    assert driver.stopped == (expected != "continue")
    assert (driver.active is None) == complete
    assert driver.segment_impulse == pytest.approx(0 if complete else 0.105)
    assert driver.total_impulse == pytest.approx(0.205)
