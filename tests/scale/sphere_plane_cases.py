"""Predeclared independent material/input rows and event gates; shared only by the observers."""

from dataclasses import dataclass

import numpy as np

H = 1 / 128
WINDOW = 16
MASS = 2.0
RADIUS = 1 / 8
PLANE = 0.0


@dataclass(frozen=True)
class Ball:
    key: int
    role: str
    z: float
    vx: float
    vz: float
    gravity: float
    restitution: float


def cases(name):
    if name == "mixed":
        return (
            Ball(41, "support", RADIUS, 0.25, 0, -8, 0),
            Ball(9, "bounce", RADIUS, -0.25, -1, -8, 0.5),
            Ball(70, "drop", RADIUS + 3 / 128, 0.125, -0.5, -8, 0),
            Ball(20, "clear", RADIUS + 1, 0.375, 0.5, -8, 0),
        )
    if name == "isolated":
        return (Ball(3, "bounce", RADIUS, -0.25, -1, 0, 0.5),)
    if name == "support":
        return tuple(
            Ball(17 + 3 * i, "support", RADIUS, (i - 16) / 32, 0, -8, 0)
            for i in reversed(range(32))
        )
    raise ValueError(name)


def application(previous, xyz_v, worlds):
    actual = np.ascontiguousarray(xyz_v, dtype=np.float32)
    if actual.shape != (len(worlds), 6) or not np.isfinite(actual).all():
        raise ValueError("invalid public COM position/velocity")
    result = np.empty((len(worlds), 8), np.float32)
    result[:, :6] = actual
    result[:, 6] = actual[:, 2].astype(float) - RADIUS - PLANE
    g = np.asarray([w.gravity for w in worlds])
    result[:, 7] = (
        0
        if previous is None
        else MASS * (actual[:, 5].astype(float) - previous[:, 5].astype(float) - g * H)
    )
    return result


def metrics(trajectory, worlds):
    """Report unsupported or missed targets explicitly and preserve every measured pose."""
    counters = []
    times = np.arange(len(trajectory), dtype=np.float64) * H
    for i, w in enumerate(worlds):
        a = trajectory[:, i].astype(float)
        J = a[1:, 7]
        positive = tuple(int(t + 1) for t in np.flatnonzero(J > 1e-5))
        first = positive[0] if positive else None
        gate = dict(
            finite=bool(np.isfinite(a).all()),
            tangent_position_velocity=bool(
                np.max(abs(a[:, 0] - times * w.vx)) <= 2e-6
                and np.max(abs(a[:, 3] - w.vx)) <= 2e-5
                and np.max(abs(a[:, [1, 4]])) <= 2e-6
            ),
        )
        minimum_gap = float(a[:, 6].min())
        net = float(J.sum())
        measure = dict(
            key=w.key,
            role=w.role,
            first_impulse_step=first,
            minimum_gap_m=minimum_gap,
            integrated_normal_impulse_n_s=net,
        )
        if w.role == "support":
            force = float(np.mean(J[-8:]) / H)
            gate.update(
                reaction=abs(force + MASS * w.gravity) <= 0.01 * MASS * abs(w.gravity),
                support_position=bool(np.max(abs(a[:, 6])) <= 2e-5),
                support_velocity=bool(np.max(abs(a[:, 5])) <= 2e-4),
                first_load=first == 1,
            )
            measure["last_eight_mean_support_n"] = force
        elif w.role == "clear":
            want_v = w.vz + times * w.gravity
            want_z = (
                w.z
                + times * w.vz
                + w.gravity * H**2 * np.arange(len(a)) * (np.arange(len(a)) + 1) / 2
            )
            gate.update(
                no_load=not positive,
                free_z=bool(np.max(abs(a[:, 2] - want_z)) <= 2e-6),
                free_v=bool(np.max(abs(a[:, 5] - want_v)) <= 2e-5),
            )
        elif w.role == "bounce":
            e = (
                float(a[first, 5] / -(a[first - 1, 5] + H * w.gravity))
                if first and a[first - 1, 5] + H * w.gravity < 0
                else None
            )
            gate.update(
                first_load=first is not None and abs(first - 1) <= 1,
                restitution=e is not None and abs(e - w.restitution) <= 0.03,
                penetration=minimum_gap >= -2e-5,
            )
            measure.update(
                measured_restitution=e,
                first_normal_impulse_n_s=None if first is None else float(J[first - 1]),
            )
        else:
            force = float(np.mean(J[-8:]) / H)
            through = max(0, (first - 1) if first is not None else len(a) - 1)
            n = np.arange(through + 1)
            z = w.z + n * H * w.vz + w.gravity * H**2 * n * (n + 1) / 2
            gate.update(
                free_until_impact=bool(
                    np.max(abs(a[: through + 1, 2] - z)) <= 2e-6
                    and np.max(abs(a[: through + 1, 5] - (w.vz + n * H * w.gravity))) <= 2e-5
                ),
                first_load=first is not None and abs(first - 5) <= 1,
                penetration=minimum_gap >= -2e-5,
                settled=abs(force + MASS * w.gravity) <= 0.01 * MASS * abs(w.gravity),
            )
            measure["last_eight_mean_support_n"] = force
        counters.append(dict(**measure, gates=gate, passed=all(gate.values())))
    return counters
