"""Independent 2 kg planar actuator, including a wall beyond the first accepted command."""

import numpy as np
import pytest

from phymeo.control.kinodynamic_approach import ApproachBudget, SegmentCheck, plan_approach
from phymeo.core.capabilities import SolverFailure


def run(start, target=(1.0, 0.0), *, obstacle=None, wall=None, budget=None, plant=None):
    budget = budget or ApproachBudget(0.05, 0.7, 2.0, 10, 16, 1e-9)
    start = np.array(start)
    frozen = start.copy()

    def apply(state, goal, rate, index):
        position, speed = state
        force = 2 * (rate - speed) / budget.dt
        assert np.linalg.norm(force) <= 2 * budget.peak_acceleration + 1e-11
        assert np.linalg.norm(rate) <= budget.peak_speed + 1e-12
        accel = force / 2 if plant is None else plant(force)
        new_speed = speed + budget.dt * accel
        new_position = position + budget.dt * (speed + new_speed) / 2
        return force, np.array([new_position, new_speed])

    def swept(first, last, index):
        a, b = first[0], last[0]
        if wall is not None:
            gap = wall - max(a[0], b[0])
        elif obstacle is None:
            return SegmentCheck()
        else:
            center, radius = obstacle
            chord = b - a
            normalizer = chord @ chord
            t = 0.0 if normalizer == 0 else np.clip((center - a) @ chord / normalizer, 0, 1)
            gap = np.linalg.norm(a + t * chord - center) - radius
        return SegmentCheck(None if gap > 0 else "obstacle", float(gap))

    result = plan_approach(
        start, target, (0, 0), budget, observe=lambda s: s, advance=apply, sweep=swept
    )
    np.testing.assert_array_equal(start, frozen)
    return result


def test_empty_space_integrates_first_drive_and_finishes_checked_brake():
    result = run([[0.0, 0.0], [0.0, 0.0]])
    assert result.admissible and result.selected.name == "seek"
    np.testing.assert_allclose(result.selected.control, [4, 0], atol=1e-12)
    np.testing.assert_allclose(result.selected.first_position, [0.0025, 0], atol=1e-12)
    assert result.selected.brake_steps > 1
    assert np.linalg.norm(result.selected.end_velocity) < 1e-9


def test_mirrored_obstacles_choose_mirrored_detours_while_direct_is_blocked():
    decisions = [
        run([[0.0, 0.0], [0.0, 0.0]], obstacle=(np.array([0.13, 0.026 * sign]), 0.05))
        for sign in (1, -1)
    ]
    for result in decisions:
        assert result.admissible
        assert next(c for c in result.candidates if c.name == "seek").reason == "obstacle"
        assert result.selected.progress > 0.025 and result.selected.minimum_margin > 0
    left, right = (r.selected for r in decisions)
    assert left.control[1] < 0 < right.control[1]
    np.testing.assert_allclose(left.control * [1, -1], right.control, atol=1e-12)


def test_finite_braking_tail_rejects_unavoidable_wall_after_first_clear_segment():
    budget = ApproachBudget(0.05, 0.7, 1.0, 1, 15, 1e-9)
    result = run([[0.0, 0.0], [0.50, 0.0]], wall=0.06, budget=budget)
    assert not result.admissible
    assert all(c.clear_steps >= 1 and c.reason == "obstacle" for c in result.candidates)
    assert result.selected.control is not None


def test_unresponsive_actuator_and_short_budget_do_not_certify_stopping():
    result = run(
        [[0.0, 0.0], [0.2, 0.0]],
        budget=ApproachBudget(0.05, 0.7, 2.0, 2, 2, 0.01),
        plant=lambda force: np.zeros_like(force),
    )
    assert not result.admissible
    assert all(c.reason == "brake_budget" for c in result.candidates)
    np.testing.assert_allclose(result.selected.first_position, [0.01, 0])
    np.testing.assert_allclose(result.selected.end_velocity, [0.2, 0])


def test_speed_overshoot_from_plant_is_not_accepted():
    result = run(
        [[0.0, 0.0], [0.65, 0.0]],
        budget=ApproachBudget(0.05, 0.7, 2.0, 2, 3, 0.005),
        plant=lambda force: np.array([10.0, 0]),
    )
    assert not result.admissible
    assert all(c.clear_steps == 0 and c.reason == "measured_speed" for c in result.candidates)


@pytest.mark.parametrize(
    "args",
    [
        (0, 0.7, 2, 2, 2, 0.01),
        (0.1, 0.7, 2, True, 2, 0.01),
        (0.1, 0.7, 2, 2, 0, 0.01),
        (0.1, 0.7, 2, 2, 2, -0.01),
        (0.1, 0.7, np.inf, 2, 2, 0.01),
    ],
)
def test_invalid_motion_budgets(args):
    with pytest.raises(ValueError):
        ApproachBudget(*args)


def test_kinematic_target_jump_cannot_be_certified_as_physical_motion():
    budget = ApproachBudget(0.01, 0.5, 1.0, 2, 3, 0.01)
    state = np.zeros((2, 2))
    result = plan_approach(
        state,
        (1, 0),
        (0, 0),
        budget,
        observe=lambda s: s,
        advance=lambda s, g, r, i: (g, np.array([[1.0, 0.0], [0.0, 0.0]])),
        sweep=lambda a, b, i: SegmentCheck(),
    )
    assert not result.admissible
    assert all(c.reason == "unreachable_displacement" for c in result.candidates)


def test_goal_outside_single_slide_workspace_can_only_park_or_retreat():
    state = np.zeros((2, 2))
    budget = ApproachBudget(0.05, 0.7, 2.0, 4, 10, 0.01)

    def slide(current, goal, rate, index):
        if abs(goal[1]) > 1e-12 or goal[0] > 1e-12:
            raise SolverFailure("slide_workspace")
        return rate.copy(), np.array([current[0] + budget.dt * (current[1] + rate) / 2, rate])

    result = plan_approach(
        state,
        (1.0, 0.0),
        (0.0, 0.0),
        budget,
        observe=lambda s: s,
        advance=slide,
        sweep=lambda a, b, i: SegmentCheck(),
    )
    assert result.admissible and result.selected.name == "brake"
    assert result.selected.progress == 0
    assert next(c for c in result.candidates if c.name == "seek").reason == "slide_workspace"
    np.testing.assert_array_equal(state, np.zeros((2, 2)))
