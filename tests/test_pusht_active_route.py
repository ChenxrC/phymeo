"""An active PushT continuation cannot silently restart or replan its route."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.area_pressure import pusht_active_route as active


def entry_at_phase(*, tick=2014, local_tick=0):
    route = SimpleNamespace(
        nodes=(
            (0.0, "approach", np.array([0.0, 0.5])),
            (0.25, "approach", np.array([0.1, 0.6])),
            (2.25, "push", np.array([0.2, 0.7])),
        )
    )
    q, v = np.array([0.02, 0.51]), np.array([0.1, 0.03])
    driver = SimpleNamespace(
        tick=tick,
        last_q=q.copy(),
        active=(1950, route),
        stopped=False,
        routes=[{"file": "route-02.json.gz"}],
        planner=SimpleNamespace(ik_calls=20, path_calls=7),
    )
    entry = SimpleNamespace(
        tick=tick - local_tick, driver=driver, pd=object(), state={"local_tick": 0}
    )
    snapshot = SimpleNamespace(tick=local_tick, time=tick * 0.001, q=q, v=v)
    return entry, snapshot


def test_active_phase_resumes_the_original_clock_and_stops_at_its_existing_end():
    entry, snapshot = entry_at_phase()
    phase = active.active_phase(entry, snapshot)
    assert phase == dict(
        absolute_tick=2014,
        route_start_tick=1950,
        route_end_tick=4200,
        route_duration_s=2.25,
        elapsed_s=0.064,
        remaining_steps=2186,
    )
    entry, snapshot = entry_at_phase(tick=4199, local_tick=2185)
    assert active.active_phase(entry, snapshot)["remaining_steps"] == 1
    entry, snapshot = entry_at_phase(tick=4200, local_tick=2186)
    with pytest.raises(ValueError, match="completed|ended|end"):
        active.active_phase(entry, snapshot)


@pytest.mark.parametrize(
    "mutation", ("no_route", "stopped", "wrong_tick", "wrong_q", "fractional_end")
)
def test_active_phase_rejects_a_different_or_noncontinuable_controller(mutation):
    entry, snapshot = entry_at_phase()
    if mutation == "no_route":
        entry.driver.active = None
    elif mutation == "stopped":
        entry.driver.stopped = True
    elif mutation == "wrong_tick":
        entry.driver.tick += 1
    elif mutation == "wrong_q":
        entry.driver.last_q[0] += 0.01
    else:
        start, route = entry.driver.active
        route.nodes = (*route.nodes[:-1], (2.2505, "push", np.zeros(2)))
    with pytest.raises(ValueError):
        active.active_phase(entry, snapshot)


def test_active_command_uses_existing_phase_and_actual_transport(monkeypatch):
    from phymeo.backends.mujoco_runtime import MuJoCoControl

    entry, snapshot = entry_at_phase()
    calls = []

    def command(driver, pd, tick, value, *, start_tick):
        assert driver is entry.driver and pd is entry.pd and value is snapshot
        assert (tick, start_tick) == (2014, 2014)
        calls.append("pd")
        return MuJoCoControl(
            [1.0, 2.0], qfrc_applied=np.zeros(2), xfrc_applied=np.zeros((1, 6))
        ), dict(
            measurement_q=value.q.tolist(),
            measurement_v=value.v.tolist(),
            route=dict(route_start_tick=1950, route_elapsed_s=0.064),
            target_q=[0.01, 0.51],
            target_v=[0.1, 0.1],
            pd={},
        )

    def transport(ctrl):
        calls.append("actual_cuda_transport")
        np.testing.assert_array_equal(ctrl, [1.0, 2.0])
        return np.array([1.125, 2.25]), dict(kind="test-actual-transport")

    monkeypatch.setattr(active, "command_from_snapshot", command)
    monkeypatch.setattr(active, "_original_action_transport", transport)
    actual, diagnostic = active.active_command(entry, snapshot)
    assert calls == ["pd", "actual_cuda_transport"]
    np.testing.assert_array_equal(actual.ctrl, [1.125, 2.25])
    assert diagnostic["transport"]["kind"] == "test-actual-transport"
    assert diagnostic["generated_channels"]["ctrl"] == [1.125, 2.25]
    assert diagnostic["active_phase"]["elapsed_s"] == 0.064
    assert entry.driver.planner.ik_calls == 20 and entry.driver.planner.path_calls == 7


@pytest.mark.parametrize("mutation", ("plan", "route", "phase"))
def test_command_rejects_hidden_replan_or_wrong_phase_before_transport(monkeypatch, mutation):
    from phymeo.backends.mujoco_runtime import MuJoCoControl

    entry, snapshot = entry_at_phase()

    def command(driver, pd, tick, value, *, start_tick):
        if mutation == "plan":
            driver.planner.path_calls += 1
        elif mutation == "route":
            driver.active = (1950, SimpleNamespace(nodes=driver.active[1].nodes))
        return MuJoCoControl([1.0, 2.0]), dict(
            route=dict(route_start_tick=1950, route_elapsed_s=0.0 if mutation == "phase" else 0.064)
        )

    monkeypatch.setattr(active, "command_from_snapshot", command)
    monkeypatch.setattr(
        active,
        "_original_action_transport",
        lambda ctrl: pytest.fail("invalid route must not reach transport"),
    )
    with pytest.raises(ValueError):
        active.active_command(entry, snapshot)


def test_checkpoint_origin_314_preserves_the_same_absolute_active_phase():
    entry, snapshot = entry_at_phase()
    entry.state["local_tick"] = 314
    snapshot.tick = 314
    assert active.active_phase(entry, snapshot) == dict(
        absolute_tick=2014,
        route_start_tick=1950,
        route_end_tick=4200,
        route_duration_s=2.25,
        elapsed_s=0.064,
        remaining_steps=2186,
    )
    entry.driver.tick = 4199
    snapshot.tick = 2499
    assert active.active_phase(entry, snapshot)["remaining_steps"] == 1


def test_checkpoint_command_offsets_local_tick_without_restarting_route(monkeypatch):
    from phymeo.backends.mujoco_runtime import MuJoCoControl

    entry, snapshot = entry_at_phase()
    entry.state["local_tick"] = 314
    snapshot.tick = 314

    def command(driver, pd, tick, value, *, start_tick):
        assert tick == 2014 and start_tick == 1700 and value.tick == 314
        return MuJoCoControl([1.0, 2.0]), dict(
            route=dict(route_start_tick=1950, route_elapsed_s=0.064)
        )

    monkeypatch.setattr(active, "command_from_snapshot", command)
    monkeypatch.setattr(
        active,
        "_original_action_transport",
        lambda ctrl: (np.array([1.0, 2.0]), dict(kind="test-actual-transport")),
    )
    actual, diagnostic = active.active_command(entry, snapshot)
    np.testing.assert_array_equal(actual.ctrl, [1.0, 2.0])
    assert diagnostic["active_phase"]["absolute_tick"] == 2014
    assert diagnostic["active_phase"]["remaining_steps"] == 2186
