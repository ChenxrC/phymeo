"""Bounded diagnostic finger closure from contact slip; no physics execution.

All command positions and corrections are in metres. This feedback only adjusts
two increasing-closure joint commands within their original limits. It is not a
learned policy, a force controller, or a guarantee of grasp success.
"""

from collections.abc import Mapping, Set
from dataclasses import dataclass
from math import isfinite
from numbers import Real


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"slip closure: {name} must be a finite real, not bool")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"slip closure: {name} must be a finite real") from error
    if not isfinite(result):
        raise ValueError(f"slip closure: {name} must be a finite real")
    return result


def _pair(value, name):
    if isinstance(value, (str, bytes, Mapping, Set)):
        raise ValueError(f"slip closure: {name} requires two finite reals")
    try:
        if len(value) != 2:
            raise ValueError(f"slip closure: {name} requires two finite reals")
        values = tuple(value)
    except TypeError as error:
        raise ValueError(f"slip closure: {name} requires two finite reals") from error
    if len(values) != 2:
        raise ValueError(f"slip closure: {name} requires two finite reals")
    return tuple(_finite(item, name) for item in values)


@dataclass(frozen=True)
class SlipClosureReceipt:
    time_s: float
    dt: float
    enabled: bool
    contact_ready: bool
    base_targets: tuple[float, float]
    normal_loads: tuple[float, float]
    slip_speeds: tuple[float, float]
    targets: tuple[float, float]
    correction: tuple[float, float]


class BoundedSlipClosure:
    """Accumulate limited closing travel only while both fingers bear load.

    Inputs follow the same two finger joints; positive command motion must close
    each finger. Slip inputs are supplied by the caller in metres per second,
    with loads in newtons. Values below their thresholds do not drive closure.
    Contact loss holds the correction; disabling clears it. Changed base targets
    may trim a held correction to preserve the original upper joint limits.

    The first accepted update has dt=0. Subsequent accepted timestamps must
    strictly increase by at most max_dt. Invalid calls leave both the timestamp
    and accumulated correction unchanged. Returned tuples own their values.
    Instances are intended for one sequential controller owner.
    """

    def __init__(
        self,
        lower,
        upper,
        *,
        max_correction=0.016,
        slip_threshold=0.015,
        gain=2.0,
        max_rate=0.02,
        min_normal=0.05,
        max_dt=0.01,
    ):
        self._lower = _pair(lower, "lower")
        self._upper = _pair(upper, "upper")
        if any(lo >= hi for lo, hi in zip(self._lower, self._upper, strict=True)):
            raise ValueError("slip closure: lower must be strictly below upper")
        parameters = {
            "max_correction": max_correction,
            "slip_threshold": slip_threshold,
            "gain": gain,
            "max_rate": max_rate,
            "min_normal": min_normal,
            "max_dt": max_dt,
        }
        for name, value in parameters.items():
            checked = _finite(value, name)
            if checked < 0 or (name == "max_dt" and checked == 0):
                raise ValueError(f"slip closure: invalid {name}")
            setattr(self, "_" + name, checked)
        self._time = None
        self._correction = (0.0, 0.0)

    def update(self, time_s, base_targets, normal_loads, slip_speeds, enabled=True):
        """Return a frozen command receipt, committing state only after validation."""
        time = _finite(time_s, "time_s")
        base = _pair(base_targets, "base_targets")
        loads = _pair(normal_loads, "normal_loads")
        slips = _pair(slip_speeds, "slip_speeds")
        if type(enabled) is not bool:
            raise ValueError("slip closure: enabled must be bool")
        if any(
            not lo <= value <= hi
            for lo, value, hi in zip(self._lower, base, self._upper, strict=True)
        ):
            raise ValueError("slip closure: base_targets must lie within original limits")
        dt = 0.0 if self._time is None else time - self._time
        if self._time is not None and (not isfinite(dt) or dt <= 0 or dt > self._max_dt):
            raise ValueError("slip closure: time must increase by at most max_dt")

        ready = all(load >= self._min_normal for load in loads)
        candidate = []
        for index in range(2):
            correction = self._correction[index] if enabled else 0.0
            if enabled and ready and dt > 0:
                rate = min(
                    self._max_rate,
                    self._gain * max(0.0, slips[index] - self._slip_threshold),
                )
                correction += dt * rate
            candidate.append(
                min(correction, self._max_correction, self._upper[index] - base[index])
            )
        corrections = tuple(candidate)
        targets = tuple(
            min(hi, max(lo, value + correction))
            for lo, hi, value, correction in zip(
                self._lower, self._upper, base, corrections, strict=True
            )
        )
        receipt = SlipClosureReceipt(
            time, dt, enabled, ready, base, loads, slips, targets, corrections
        )
        self._time, self._correction = time, corrections
        return receipt
