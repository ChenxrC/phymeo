"""Task-space references for two actuators sharing a transported load."""

import numpy as np

from ..core.validation import finite_real


class SoloLiftHold:
    """Hold the measured tool pose once solo support is observed in consecutive frames."""

    def __init__(self, frames=2):
        if type(frames) is not int or frames < 1:
            raise ValueError("expected a positive number of support frames")
        self.frames = frames
        self._count = 0
        self.position = None

    def observe(self, position, supported):
        point = np.asarray(position, dtype=float)
        if point.ndim != 1 or not point.size or not np.isfinite(point).all():
            raise ValueError("expected a finite tool position")
        if self.position is None:
            self._count = self._count + 1 if supported else 0
            if self._count >= self.frames:
                self.position = point.copy()
                self.position.flags.writeable = False
        return self.position


def bounded_approach(position, velocity, goal, goal_velocity, dt, *, peak_speed, peak_acceleration):
    """One constant-acceleration tool reference from the measured position and velocity."""
    p, v, target, rate = (
        np.asarray(value, dtype=float) for value in (position, velocity, goal, goal_velocity)
    )
    if (
        p.ndim != 1
        or not p.size
        or any(row.shape != p.shape or not np.isfinite(row).all() for row in (p, v, target, rate))
    ):
        raise ValueError("expected matching finite tool coordinates")
    dt = finite_real(dt, "control interval")
    speed = finite_real(peak_speed, "peak tool speed")
    acceleration = finite_real(peak_acceleration, "peak tool acceleration")
    if dt <= 0 or min(speed, acceleration) <= 0:
        raise ValueError("positive approach duration and motion budgets are required")

    def limited(vector, limit):
        length = np.hypot.reduce(vector)
        return vector if length <= limit else vector * (limit / length)

    error = target - p
    braking_speed = np.sqrt(2 * acceleration * np.hypot.reduce(error))
    desired = limited(limited(rate, speed) + limited(error / dt, braking_speed), speed)
    end_velocity = v + limited(desired - v, acceleration * dt)
    return p + 0.5 * dt * (v + end_velocity), end_velocity


class PairedTransport:
    """A common cubic time law starting from each tool's actual acquired pose."""

    def __init__(self, starts, goals, *, peak_speed):
        self.starts = np.array(starts, dtype=float, copy=True)
        self.goals = np.array(goals, dtype=float, copy=True)
        if (
            self.starts.ndim != 2
            or self.starts.shape[0] != 2
            or not self.starts.shape[1]
            or self.starts.shape != self.goals.shape
            or not np.isfinite(self.starts).all()
            or not np.isfinite(self.goals).all()
        ):
            raise ValueError("two finite tool positions and goals are required")
        speed = finite_real(peak_speed, "peak tool speed")
        if speed <= 0:
            raise ValueError("peak tool speed must be positive")
        self.starts.flags.writeable = self.goals.flags.writeable = False
        self.duration = 1.5 * float(np.linalg.norm(self.goals - self.starts, axis=1).max()) / speed

    def sample(self, elapsed):
        elapsed = finite_real(elapsed, "elapsed time")
        if self.duration == 0 or elapsed >= self.duration:
            return self.goals.copy(), np.zeros_like(self.goals)
        if elapsed <= 0:
            return self.starts.copy(), np.zeros_like(self.starts)
        t = elapsed / self.duration
        delta = self.goals - self.starts
        return self.starts + t * t * (3 - 2 * t) * delta, (6 * t * (1 - t) / self.duration) * delta
