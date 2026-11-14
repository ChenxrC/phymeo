"""Hand-calculated progress and budget boundaries; no physics or IK oracle."""

from dataclasses import FrozenInstanceError

import pytest

from phymeo.control.manipulation_push_progress import (
    BoundedPushProgress,
    PushProgressPolicy,
)


def snapshot(monitor):
    return monitor.policy, monitor.started, monitor.terminal, monitor.status, monitor.history


def test_initial_state_and_positive_cost_drop():
    monitor = BoundedPushProgress()
    assert monitor.policy == PushProgressPolicy(8, 12000, .005, 2)
    assert not monitor.started and not monitor.terminal
    assert monitor.status == "not_started" and monitor.history == ()
    assert monitor.start(100, 10) is None
    assert monitor.started and monitor.status == "continue"
    result = monitor.observe(120, 9, contact_impulse_ns=.25)
    assert (result.start_step, result.end_step) == (100, 120)
    assert (result.cost_before, result.cost_after) == (10., 9.)
    assert result.relative_improvement == .1
    assert result.contact_impulse_ns == .25
    assert (result.segment, result.stalls, result.status) == (1, 0, "continue")
    assert monitor.history == (result,) and not monitor.terminal


def test_rising_cost_stalls_and_later_contact_progress_clears_stalls():
    monitor = BoundedPushProgress()
    monitor.start(0, 9)
    first = monitor.observe(2, 9.5, contact_impulse_ns=1)
    assert first.relative_improvement == pytest.approx(-1 / 18)
    assert first.stalls == 1 and first.status == "continue"
    second = monitor.observe(9, 9, contact_impulse_ns=.01)
    assert second.relative_improvement == pytest.approx(1 / 19)
    assert (second.start_step, second.end_step, second.stalls) == (2, 9, 0)
    assert second.status == "continue"


def test_contact_absence_does_not_count_as_progress_even_with_cost_drop():
    monitor = BoundedPushProgress()
    monitor.start(0, 10)
    first = monitor.observe(1, 9, contact_impulse_ns=0)
    second = monitor.observe(2, 8, contact_impulse_ns=0)
    assert first.relative_improvement == .1 and first.stalls == 1
    assert second.relative_improvement == pytest.approx(1 / 9)
    assert second.stalls == 2 and second.status == "no_progress"
    assert monitor.terminal


def test_stall_recovery_uses_last_segment_cost_and_inclusive_threshold():
    monitor = BoundedPushProgress(max_stalled_segments=3)
    monitor.start(10, 100)
    first = monitor.observe(11, 100, contact_impulse_ns=1)
    second = monitor.observe(20, 99.5, contact_impulse_ns=0)
    third = monitor.observe(21, 99, contact_impulse_ns=1)
    assert (first.stalls, second.stalls, third.stalls) == (1, 2, 0)
    assert second.relative_improvement == .005
    assert third.relative_improvement == pytest.approx(1 / 199)
    equal = BoundedPushProgress()
    equal.start(0, 100)
    assert equal.observe(1, 99.5, contact_impulse_ns=1).stalls == 0


def test_zero_cost_does_not_infer_goal_success_and_floor_is_explicit():
    monitor = BoundedPushProgress()
    monitor.start(0, 0)
    first = monitor.observe(1, 0, contact_impulse_ns=1)
    assert first.relative_improvement == 0 and first.status == "continue"
    second = monitor.observe(2, 0, contact_impulse_ns=1)
    assert second.status == "no_progress"
    small = BoundedPushProgress()
    small.start(3, 5e-13)
    decreased = small.observe(4, 0, contact_impulse_ns=1)
    increased = small.observe(5, 5e-13, contact_impulse_ns=1)
    assert decreased.relative_improvement == .5
    assert increased.relative_improvement == -.5
    assert increased.stalls == 1


def test_zero_threshold_still_requires_positive_contact_impulse():
    monitor = BoundedPushProgress(min_relative_improvement=0)
    monitor.start(0, 1)
    first = monitor.observe(1, 1, contact_impulse_ns=0)
    second = monitor.observe(2, 1, contact_impulse_ns=1e-300)
    assert first.stalls == 1
    assert second.relative_improvement == 0 and second.stalls == 0
    assert second.status == "continue"


@pytest.mark.parametrize(
    ("options", "success", "expected"),
    (
        (dict(max_segments=1, max_steps=1, max_stalled_segments=1), True, "success"),
        (dict(max_segments=1, max_steps=1, max_stalled_segments=1), False, "step_budget"),
        (dict(max_segments=1, max_steps=2, max_stalled_segments=1), False, "segment_budget"),
        (dict(max_segments=2, max_steps=2, max_stalled_segments=1), False, "no_progress"),
    ),
)
def test_terminal_priority_and_literal_caller_goal_signal(options, success, expected):
    monitor = BoundedPushProgress(**options)
    monitor.start(40, 1)
    result = monitor.observe(41, 2, contact_impulse_ns=0, goal_success=success)
    assert result.status == expected
    assert result.relative_improvement == -1
    assert (result.segment, result.stalls) == (1, 1)
    assert monitor.status == expected and monitor.terminal
    before = snapshot(monitor)
    with pytest.raises(ValueError):
        monitor.observe(42, 0, contact_impulse_ns=1, goal_success=True)
    with pytest.raises(ValueError):
        monitor.start(42, 0)
    assert snapshot(monitor) == before


def test_step_budget_counts_from_original_start_with_variable_segment_lengths():
    monitor = BoundedPushProgress(max_segments=10, max_steps=10)
    monitor.start(100, 10)
    first = monitor.observe(101, 9, contact_impulse_ns=1)
    second = monitor.observe(108, 8, contact_impulse_ns=1)
    third = monitor.observe(110, 7, contact_impulse_ns=1)
    assert [first.status, second.status, third.status] == ["continue", "continue", "step_budget"]
    assert [(r.start_step, r.end_step) for r in monitor.history] == [(100, 101), (101, 108), (108, 110)]
    assert third.segment == 3 and third.stalls == 0


@pytest.mark.parametrize("success", (False, True))
def test_overshooting_observation_is_retained_whole_and_not_rolled_back(success):
    monitor = BoundedPushProgress(max_steps=10)
    monitor.start(7, 10)
    result = monitor.observe(20, 9, contact_impulse_ns=1, goal_success=success)
    assert (result.start_step, result.end_step) == (7, 20)
    assert result.status == ("success" if success else "step_budget")
    assert result.cost_after == 9 and monitor.history == (result,)
    assert monitor.terminal


def test_segment_budget_equality_stops_even_when_every_segment_improves():
    monitor = BoundedPushProgress(max_segments=2)
    monitor.start(0, 4)
    assert monitor.observe(1, 2, contact_impulse_ns=1).status == "continue"
    final = monitor.observe(2, 1, contact_impulse_ns=1)
    assert final.status == "segment_budget" and final.stalls == 0
    assert len(monitor.history) == 2


def test_lifecycle_start_once_and_observe_requires_start():
    monitor = BoundedPushProgress()
    initial = snapshot(monitor)
    with pytest.raises(ValueError):
        monitor.observe(1, 1, contact_impulse_ns=1)
    assert snapshot(monitor) == initial
    monitor.start(10, 3)
    before = snapshot(monitor)
    with pytest.raises(ValueError):
        monitor.start(0, 50)
    assert snapshot(monitor) == before
    result = monitor.observe(11, 2, contact_impulse_ns=1)
    assert result.start_step == 10 and result.cost_before == 3


class IntegerSubclass(int):
    pass


@pytest.mark.parametrize("step", (-1, True, 1., "1", IntegerSubclass(1)))
def test_start_rejects_non_builtin_nonnegative_steps_transactionally(step):
    monitor = BoundedPushProgress()
    before = snapshot(monitor)
    with pytest.raises(ValueError):
        monitor.start(step, 1)
    assert snapshot(monitor) == before
    monitor.start(0, 1)
    assert monitor.started


@pytest.mark.parametrize("cost", (-1, True, float("nan"), float("inf"), 10 ** 400, "1", [1], 1j))
def test_invalid_start_cost_does_not_consume_start(cost):
    monitor = BoundedPushProgress()
    with pytest.raises(ValueError):
        monitor.start(0, cost)
    assert not monitor.started and monitor.history == ()
    monitor.start(0, 1)
    assert monitor.status == "continue"


@pytest.mark.parametrize(
    "change",
    (
        dict(step=20), dict(step=19), dict(step=-1), dict(step=True), dict(step=21.),
        dict(step=IntegerSubclass(21)), dict(cost=-1), dict(cost=True),
        dict(cost=float("nan")), dict(cost=float("inf")), dict(cost=10 ** 400),
        dict(contact_impulse_ns=-1), dict(contact_impulse_ns=True),
        dict(contact_impulse_ns=float("inf")), dict(contact_impulse_ns=float("nan")),
        dict(contact_impulse_ns=[1]), dict(goal_success=1), dict(goal_success=None),
    ),
)
def test_invalid_observation_leaves_all_state_unchanged_and_can_retry(change):
    monitor = BoundedPushProgress()
    monitor.start(10, 100)
    monitor.observe(20, 90, contact_impulse_ns=1)
    before = snapshot(monitor)
    values = dict(step=21, cost=81, contact_impulse_ns=1, goal_success=False)
    values.update(change)
    with pytest.raises(ValueError):
        monitor.observe(**values)
    assert snapshot(monitor) == before
    accepted = monitor.observe(21, 81, contact_impulse_ns=1)
    assert accepted.cost_before == 90 and accepted.segment == 2
    assert accepted.relative_improvement == .1


def test_nonfinite_derived_relative_change_rejects_without_advancing():
    monitor = BoundedPushProgress()
    monitor.start(0, 0)
    before = snapshot(monitor)
    with pytest.raises(ValueError, match="relative"):
        monitor.observe(1, 1e308, contact_impulse_ns=1)
    assert snapshot(monitor) == before
    accepted = monitor.observe(1, 0, contact_impulse_ns=1)
    assert accepted.start_step == 0 and accepted.segment == 1


@pytest.mark.parametrize(
    "options",
    (
        dict(max_segments=0), dict(max_segments=65), dict(max_segments=True),
        dict(max_segments=1.), dict(max_segments=IntegerSubclass(1)),
        dict(max_steps=0), dict(max_steps=1000001), dict(max_steps=True), dict(max_steps=1.),
        dict(max_stalled_segments=0), dict(max_stalled_segments=65),
        dict(max_stalled_segments=True), dict(max_stalled_segments=1.),
        dict(min_relative_improvement=-1), dict(min_relative_improvement=True),
        dict(min_relative_improvement=float("inf")), dict(min_relative_improvement=float("nan")),
    ),
)
def test_policy_rejects_invalid_or_unbounded_parameters(options):
    with pytest.raises(ValueError):
        BoundedPushProgress(**options)
    with pytest.raises(ValueError):
        PushProgressPolicy(**options)


def test_maximum_policy_capacities_keep_history_bounded():
    monitor = BoundedPushProgress(max_segments=64, max_steps=1000000, max_stalled_segments=64)
    monitor.start(123, 1)
    for segment in range(1, 65):
        result = monitor.observe(123 + segment, 1, contact_impulse_ns=0)
        assert result.segment == segment and result.stalls == segment
        assert result.status == ("continue" if segment < 64 else "segment_budget")
    assert monitor.terminal and len(monitor.history) == 64
    with pytest.raises(ValueError):
        monitor.observe(188, 0, contact_impulse_ns=1)


def test_finite_threshold_above_one_is_not_silently_clamped():
    monitor = BoundedPushProgress(min_relative_improvement=2.)
    monitor.start(0, 1)
    result = monitor.observe(1, 0, contact_impulse_ns=1)
    assert result.relative_improvement == 1.
    assert result.stalls == 1 and monitor.policy.min_relative_improvement == 2.


def test_policy_records_history_and_state_are_read_only_snapshots():
    monitor = BoundedPushProgress()
    monitor.start(0, 10)
    first = monitor.observe(1, 9, contact_impulse_ns=1)
    history = monitor.history
    with pytest.raises(FrozenInstanceError):
        monitor.policy.max_steps = 2
    with pytest.raises(FrozenInstanceError):
        first.status = "success"
    with pytest.raises(TypeError):
        history[0] = first
    for name, value in (("policy", None), ("terminal", True), ("status", "success"),
                        ("history", ()), ("started", False), ("_state", None), ("_policy", None)):
        with pytest.raises(AttributeError):
            setattr(monitor, name, value)
    with pytest.raises(AttributeError):
        del monitor._state
    with pytest.raises(ValueError):
        monitor.__init__()
    second = monitor.observe(2, 8, contact_impulse_ns=1)
    assert history == (first,)
    assert monitor.history == (first, second)
