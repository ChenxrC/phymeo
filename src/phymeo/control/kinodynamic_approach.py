"""Bounded planar steering and braking trials from the same measured dynamic state."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.validation import finite_real
from .grasp_transfer import bounded_approach


@dataclass(frozen=True)
class ApproachBudget:
    dt: float
    peak_speed: float
    peak_acceleration: float
    approach_steps: int
    brake_steps: int
    stop_speed: float

    def __post_init__(self):
        for name in ("dt", "peak_speed", "peak_acceleration", "stop_speed"):
            value = finite_real(getattr(self, name), name)
            if value < 0 or (name != "stop_speed" and value == 0):
                raise ValueError(
                    "positive motion budgets and a nonnegative stop speed are required"
                )
            object.__setattr__(self, name, value)
        if self.stop_speed >= self.peak_speed:
            raise ValueError("stop speed must be below the speed budget")
        for name in ("approach_steps", "brake_steps"):
            count = getattr(self, name)
            if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
                raise ValueError("positive integer approach and braking horizons are required")


@dataclass(frozen=True)
class SegmentCheck:
    """Optional swept blocker and a metric margin. Positive margins denote separation."""

    blocker: str | None = None
    margin: float | None = None

    def __post_init__(self):
        if self.margin is not None:
            finite_real(self.margin, "segment margin")


@dataclass(frozen=True)
class ApproachTrial:
    name: str
    control: object
    first_position: np.ndarray | None
    first_velocity: np.ndarray | None
    end_velocity: np.ndarray
    clear_steps: int
    brake_steps: int
    progress: float
    minimum_margin: float | None
    reason: str | None


@dataclass(frozen=True)
class ApproachPlan:
    selected: ApproachTrial
    candidates: tuple[ApproachTrial, ...]

    @property
    def admissible(self):
        """The selected forecast passed every sweep and its terminal speed test."""
        return self.selected.reason is None


def _point(value):
    data = np.asarray(value, dtype=float)
    if data.shape != (2,) or not np.isfinite(data).all():
        raise ValueError("expected a finite planar vector")
    return data


def plan_approach(state, target, target_velocity, budget, *, observe, advance, sweep):
    """Return the first actuator control of a finite, deterministic forecast.

    observe(state) reads physical planar position and velocity. The pure callback
    advance(state, goal, rate, index) returns (control, new_state) from its plant.
    sweep(first, last, index) checks the intervening motion. Index zero starts at
    the observed entry. The terminal world speed must reach budget.stop_speed
    in the separately checked braking tail. If all trials fail, their common
    initial state is intact and the selected trial identifies a mitigation only.
    """
    if not isinstance(budget, ApproachBudget):
        raise TypeError("expected an ApproachBudget")
    target, target_velocity = _point(target), _point(target_velocity)
    entry, entry_velocity = map(_point, observe(state))
    direction = target - entry
    if np.linalg.norm(direction) == 0:
        direction = target_velocity - entry_velocity
    length = np.linalg.norm(direction)
    forward = direction / length if length else np.array([1.0, 0.0])
    lateral = np.array([-forward[1], forward[0]])
    modes = (
        ("seek", 0.0, 0.0),
        ("left", 0.0, 1.0),
        ("right", 0.0, -1.0),
        ("forward_left", 1.0, 1.0),
        ("forward_right", 1.0, -1.0),
        ("back_left", -1.0, 1.0),
        ("back_right", -1.0, -1.0),
        ("retreat", -1.0, 0.0),
        ("brake", 0.0, 0.0),
    )
    trials = []
    for name, along, across in modes:
        current = state
        position, velocity = entry, entry_velocity
        control = first_position = first_velocity = None
        reason = None
        margin = None
        clear = brakes = 0
        progress = 0.0
        for index in range(budget.approach_steps + budget.brake_steps):
            tail = index >= budget.approach_steps
            goal = target + index * budget.dt * target_velocity
            rate = target_velocity
            if tail or name == "brake":
                goal, rate = position, np.zeros(2)
            elif name != "seek" and index < (budget.approach_steps + 1) // 2:
                steering = along * forward + across * lateral
                distance = max(np.linalg.norm(goal - position), budget.dt * budget.peak_speed)
                goal, rate = position + distance * steering / np.linalg.norm(steering), np.zeros(2)
            desired, desired_rate = bounded_approach(
                position,
                velocity,
                goal,
                rate,
                budget.dt,
                peak_speed=budget.peak_speed,
                peak_acceleration=budget.peak_acceleration,
            )
            try:
                command, predicted = advance(current, desired, desired_rate, index)
            except SolverFailure as exc:
                reason = str(exc)
                break
            end_position, end_velocity = map(_point, observe(predicted))
            if index == 0:
                control, first_position, first_velocity = (
                    command,
                    end_position.copy(),
                    end_velocity.copy(),
                )
            result = sweep(current, predicted, index)
            if result.margin is not None:
                margin = result.margin if margin is None else min(margin, result.margin)
            if result.blocker is not None:
                reason = result.blocker
                break
            speed = np.linalg.norm(end_velocity)
            if speed > budget.peak_speed * (1 + 64 * np.finfo(float).eps):
                reason = "measured_speed"
                break
            if np.linalg.norm(end_position - position) > budget.dt * budget.peak_speed * (
                1 + 64 * np.finfo(float).eps
            ):
                reason = "unreachable_displacement"
                break
            clear += 1
            current, position, velocity = predicted, end_position, end_velocity
            if tail:
                brakes += 1
                if speed <= budget.stop_speed:
                    break
            else:
                progress = np.linalg.norm(target - entry) - np.linalg.norm(
                    target + (index + 1) * budget.dt * target_velocity - position
                )
        else:
            reason = "brake_budget"
        trials.append(
            ApproachTrial(
                name,
                control,
                first_position,
                first_velocity,
                velocity.copy(),
                clear,
                brakes,
                float(progress),
                margin,
                reason,
            )
        )

    def rank(trial):
        margin = -np.inf if trial.minimum_margin is None else trial.minimum_margin
        if trial.reason is None:
            return (2, trial.progress, margin, -trial.brake_steps)
        return (int(trial.control is not None), trial.clear_steps, margin, trial.name == "brake")

    return ApproachPlan(max(trials, key=rank), tuple(trials))
