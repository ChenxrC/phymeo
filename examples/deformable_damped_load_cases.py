"""D6-B one-choice application of four explicit 4096 N s/m radial dampers.

Reuse the unchanged D4 declarations and independent geometry/elastic/rigid
references. Only the optional node dissipation and case identity change. The
continued FP64 reference always advances its own damped state.
"""

from dataclasses import replace
from types import MappingProxyType

import deformable_normal_load_cases as base
import numpy as np

from phymeo.backends.deformable_dissipation import DistanceDissipation

COEFFICIENT = 4096.0
NAMES = (
    "d6b_n1_sphere_c4096",
    "d6b_n2_capsule_c4096",
    "d6b_n3_soft_c4096",
    "d6b_n1_fine_c4096",
)
BASE_NAMES = MappingProxyType(dict(zip(NAMES, base.NAMES, strict=True)))
ph = base.ph
T, BASE_H = base.T, base.BASE_H
RADIUS, WIDTH, STIFFNESS, DAMPING, LOAD = (
    base.RADIUS,
    base.WIDTH,
    base.STIFFNESS,
    base.DAMPING,
    base.LOAD,
)
SHORT_ATOL, SHORT_RTOL = base.SHORT_ATOL, base.SHORT_RTOL
LONG_POSITION, LONG_VELOCITY, LONG_ANGLE = base.LONG_POSITION, base.LONG_VELOCITY, base.LONG_ANGLE
SUPPORT_WINDOWS, EVENT_DURATION = base.SUPPORT_WINDOWS, base.EVENT_DURATION
close, local_scale, rotation = base.close, base.local_scale, base.rotation
root_state, reference_load = base.root_state, base.reference_load
material_step, rigid_step = base.material_step, base.rigid_step
load_at_step, controls, create = base.load_at_step, base.controls, base.create
initial_reference = base.initial_reference


def make_case(name, *, world=7):
    if name not in BASE_NAMES:
        raise ValueError("unknown fixed damped normal-load case")
    original_name = BASE_NAMES[name]
    original = base.make_case(original_name, world=world)
    parameters = DistanceDissipation(
        original["ids"][original["edges"]], np.full(4, COEFFICIENT, np.float64)
    )
    node = replace(original["node"], dissipation=parameters)
    return MappingProxyType(dict(original, name=name, base_name=original_name, node=node))


def manifest(case):
    return dict(
        base.manifest(case),
        base_name=case["base_name"],
        dissipation=case["node"].dissipation,
        application="one ordered axial velocity pass after unchanged elastic reconstruction",
        coefficient_selection="predeclared 4096 N s/m; no trajectory fit or sweep",
    )


def ordered_velocity(spec, x, velocity, h):
    """Independent FP64 split recurrence; never uploaded to a physical owner."""
    result = np.asarray(velocity, float).copy()
    weights = np.asarray(spec.model.inverse_mass, np.float32).astype(float)
    for (i, j), coefficient in zip(
        spec.constraints.indices, spec.dissipation.coefficients_ns_per_m, strict=True
    ):
        if coefficient == 0:
            continue
        separation = np.asarray(x[i], float) - x[j]
        normal = separation / np.linalg.norm(separation)
        rate = float(normal @ (result[i] - result[j]))
        impulse = (
            -h
            * coefficient
            * rate
            / (1 + h * coefficient * (weights[i] + weights[j]) * (normal @ normal))
        )
        result[i] += weights[i] * impulse * normal
        result[j] -= weights[j] * impulse * normal
    return result


def continue_reference(case, state, external):
    x, v, q, speed = state
    force, wrench, rows = reference_load(case, x, v, q, speed)
    y, reconstructed, lam = material_step(case, x, v, force, case["h"])
    velocity = ordered_velocity(case["node"], y, reconstructed, case["h"])
    next_q, next_speed = rigid_step(case, q, speed, wrench, case["h"], external)
    return (y, velocity, next_q, next_speed), dict(
        force=force,
        wrench=wrench,
        rows=rows,
        multipliers=lam,
        reconstructed_velocity=reconstructed,
        dissipated_velocity=velocity,
    )
