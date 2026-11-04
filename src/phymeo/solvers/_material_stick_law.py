"""Implicit elastic trial traction and Coulomb return map for one tangential material station."""

from math import hypot

import numpy as np

from ..core.capabilities import SolverFailure


def return_traction(
    old, free, inverse, h, normal_force, prior, stiffness, static_friction, kinetic_friction
):
    trial = (old + free) / (1.0 + stiffness * h * h * inverse)
    trial_n = stiffness * hypot(*trial)
    continuing = prior is not None and prior.regime == "slide" and trial @ old >= 0
    limit = normal_force * (kinetic_friction if continuing else static_friction)
    sliding = trial_n > limit
    force = -stiffness * trial
    if sliding:
        force *= kinetic_friction * normal_force / trial_n
    if not np.isfinite((limit, trial_n)).all():
        raise SolverFailure("material tangent prediction exceeds precision")
    return force, sliding
