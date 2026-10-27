"""Explicit small fixed-base pendulum models for CPU reference validation."""

from numbers import Integral

import numpy as np

from ..core.robot import RobotModel, RobotState


def pendulum_chain(count=2):
    """Half-metre, one-kilogram links; two-link default is the frozen S1 case."""
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 1:
        raise ValueError("count must be a positive integer")
    model = RobotModel(
        parents=np.arange(-1, count - 1, dtype=np.int64),
        joint_types=("revolute",) * count,
        joint_position=np.array([[0.0, 0, 0]] + [[0.0, -0.5, 0]] * (count - 1)),
        joint_orientation=np.tile([0.0, 0, 0, 1], (count, 1)),
        joint_axis=np.tile([0.0, 0, 1], (count, 1)),
        mass=np.ones(count),
        com=np.tile([0.0, -0.25, 0], (count, 1)),
        inertia_diagonal=np.tile([1 / 48, 0.001, 1 / 48], (count, 1)),
        gravity=np.array([0.0, -9.81, 0]),
    )
    q, v = np.zeros(count), np.zeros(count)
    q[0], v[0] = 0.35, 0.1
    if count > 1:
        q[1], v[1] = -0.2, -0.05
    return model, RobotState(q, v)
