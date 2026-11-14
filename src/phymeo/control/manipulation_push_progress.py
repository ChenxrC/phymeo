"""Finite segment progress monitoring for one continuous physical episode.

This is a caller-fed diagnostic, not a Goal evaluator or physical verifier.
It does not execute simulation, inspect contacts, select controls, or run IK.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Real


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"push progress: {name} must be a finite nonnegative real")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"push progress: {name} must be finite") from error
    if not isfinite(result) or result < 0:
        raise ValueError(f"push progress: {name} must be finite and nonnegative")
    return result


def _step(value):
    if type(value) is not int or value < 0:
        raise ValueError("push progress: step must be a nonnegative builtin int, not bool")
    return value


def _budget(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"push progress: {name} must be a builtin int in 1..{maximum}")
    return value


@dataclass(frozen=True, slots=True)
class PushProgressPolicy:
    max_segments: int = 8
    max_steps: int = 12000
    min_relative_improvement: float = 0.005
    max_stalled_segments: int = 2

    def __post_init__(self):
        _budget(self.max_segments, "max_segments", 64)
        _budget(self.max_steps, "max_steps", 1000000)
        _budget(self.max_stalled_segments, "max_stalled_segments", 64)
        threshold = _nonnegative(self.min_relative_improvement, "min_relative_improvement")
        object.__setattr__(self, "min_relative_improvement", threshold)


@dataclass(frozen=True, slots=True)
class PushProgressRecord:
    start_step: int
    end_step: int
    cost_before: float
    cost_after: float
    relative_improvement: float
    contact_impulse_ns: float
    segment: int
    stalls: int
    status: str


@dataclass(frozen=True, slots=True)
class _ProgressState:
    initial_step: int | None
    last_step: int | None
    last_cost: float | None
    stalls: int
    history: tuple[PushProgressRecord, ...]
    status: str


class BoundedPushProgress:
    """Monitor finitely many reported segments without inferring task success.

    start(step, cost) accepts the initial endpoint once and returns None.
    observe accepts strictly later segment endpoints. Its receipt records the
    previous/current endpoint, whereas the step budget always uses the original
    start step. Gaps cannot reset that cumulative budget. Overshooting endpoints
    are retained whole; this monitor neither truncates nor rolls back physics.
    The execution owner must enforce its hard step cap before taking each step.

    Relative improvement is (previous_cost-cost)/max(previous_cost, 1e-12).
    Only positive contact impulse AND improvement >= the declared threshold
    reset the consecutive stall count. Otherwise it increases by one, including
    cost decreases without contact and a zero cost with the default threshold.
    contact_impulse_ns means the caller's integral of the pusher resultant force
    magnitude over this same segment, in N s; it is not a cancelling vector sum.
    This module cannot verify how that scalar or the cost was measured.

    goal_success must be a literal bool supplied by the real task evaluator.
    Terminal priority is success, cumulative step budget, segment budget, then
    stalled-segment budget. Equality reaches a budget. Thus success on a final
    budget endpoint remains success; cost zero alone never declares success.

    Public state is read-only, policy/receipts are frozen, and history is a
    bounded immutable tuple. Every failed call leaves the previous state intact.
    A terminal monitor cannot restart: a new episode needs a new instance.
    Instances have one sequential owner, not a concurrent update protocol.
    """

    __slots__ = ("_policy", "_state")

    def __init__(
        self,
        *,
        max_segments=8,
        max_steps=12000,
        min_relative_improvement=0.005,
        max_stalled_segments=2,
    ):
        if hasattr(self, "_state"):
            raise ValueError("push progress: an initialized monitor cannot restart")
        policy = PushProgressPolicy(
            max_segments, max_steps, min_relative_improvement, max_stalled_segments
        )
        state = _ProgressState(None, None, None, 0, (), "not_started")
        object.__setattr__(self, "_policy", policy)
        object.__setattr__(self, "_state", state)

    def __setattr__(self, name, value):
        raise AttributeError("push progress state is read-only; use start or observe")

    def __delattr__(self, name):
        raise AttributeError("push progress state is read-only")

    @property
    def policy(self):
        return self._policy

    @property
    def started(self):
        return self._state.initial_step is not None

    @property
    def terminal(self):
        return self._state.status not in ("not_started", "continue")

    @property
    def status(self):
        return self._state.status

    @property
    def history(self):
        return self._state.history

    def start(self, step, cost):
        step, cost = _step(step), _nonnegative(cost, "cost")
        if self._state.initial_step is not None:
            raise ValueError("push progress: start is accepted only once")
        state = _ProgressState(step, step, cost, 0, (), "continue")
        object.__setattr__(self, "_state", state)

    def observe(self, step, cost, *, contact_impulse_ns, goal_success=False):
        step, cost = _step(step), _nonnegative(cost, "cost")
        impulse = _nonnegative(contact_impulse_ns, "contact_impulse_ns")
        if type(goal_success) is not bool:
            raise ValueError("push progress: goal_success must be a literal bool")
        previous, policy = self._state, self._policy
        if previous.initial_step is None:
            raise ValueError("push progress: observe requires start")
        if previous.status != "continue":
            raise ValueError("push progress: terminal monitor cannot observe again")
        if step <= previous.last_step:
            raise ValueError("push progress: observation steps must strictly increase")
        try:
            improvement = (previous.last_cost - cost) / max(previous.last_cost, 1e-12)
        except OverflowError as error:
            raise ValueError("push progress: relative improvement is not finite") from error
        if not isfinite(improvement):
            raise ValueError("push progress: relative improvement is not finite")
        segment = len(previous.history) + 1
        stalls = (
            0
            if impulse > 0 and improvement >= policy.min_relative_improvement
            else previous.stalls + 1
        )
        if goal_success:
            status = "success"
        elif step - previous.initial_step >= policy.max_steps:
            status = "step_budget"
        elif segment >= policy.max_segments:
            status = "segment_budget"
        elif stalls >= policy.max_stalled_segments:
            status = "no_progress"
        else:
            status = "continue"
        record = PushProgressRecord(
            previous.last_step,
            step,
            previous.last_cost,
            cost,
            improvement,
            impulse,
            segment,
            stalls,
            status,
        )
        state = _ProgressState(
            previous.initial_step, step, cost, stalls, (*previous.history, record), status
        )
        object.__setattr__(self, "_state", state)
        return record
