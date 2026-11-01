"""Shared physical midpoint state updates for particle/robot compliant programs."""

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.robot import RobotState
from ..core.scene import SceneState
from ..core.state import ParticleState
from ..core.validation import finite_real


def advance_state(state, accel, dt, *, midpoint):
    domains = {}
    for name, s in state.domains.items():
        particle = isinstance(s, ParticleState)
        q, v = (s.position, s.velocity) if particle else (s.q, s.v)
        u = v + dt * accel[name]
        x = q + dt * (v if midpoint else (v + u) / 2)
        domains[name] = ParticleState(x, u) if particle else RobotState(x, u)
    return SceneState(domains)


def optional_budget(value, name):
    if value is None:
        return None
    value = finite_real(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def check_robot_contact(rows, max_penetration, max_violation):
    for row in rows:
        if max_penetration is not None and row["penetration_m"] > max_penetration:
            raise SolverFailure(
                f"sampled robot penetration limit exceeded: {row['penetration_m']:.6g} m"
            )
        if max_violation is not None and row["joint_violation"] > max_violation:
            raise SolverFailure(
                f"sampled joint limit exceeded: {row['joint_violation']:.6g} native coordinates"
            )


def finite_accelerations(accel):
    if not all(np.isfinite(a).all() for a in accel.values()):
        raise SolverFailure("nonfinite coupled acceleration")
