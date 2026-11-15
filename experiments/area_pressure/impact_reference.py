"""V06 axial two-finite-mass impact reference, driven by sampled pressure area.

A soft radial-field sphere A is above a translating rigid-pressure target B.
Both bodies have finite inertia; there are no external forces or prescribed
velocities. The centered axial symmetry gives zero torque for this restricted
model. This is not a general rigid-body collision interface or a TYX solver.
Continuous-field potential energy is diagnostic only; it never supplies or
repairs an impulse. Actual elastic forces come from the existing area sampler.
"""

import json
from dataclasses import asdict, dataclass, fields
from math import pi, sqrt

from .reference import PressureIntegral, _real, sphere_plane_case


def _need(condition, message):
    if not condition:
        raise ValueError("axial impact reference: " + message)


def _number(value, name):
    try:
        return _real(value, name)
    except OverflowError as error:
        raise ValueError("axial impact reference: nonfinite " + name) from error


@dataclass(frozen=True)
class AxialImpactConfig:
    mass_a_kg: float = .5
    mass_b_kg: float = 1.
    radius_m: float = .02
    target_half_width_m: float = .04
    target_thickness_m: float = .01
    modulus_pa: float = 1e5
    resolution_m: float = .001
    dissipation_s_per_m: float = 0.
    max_indentation_m: float = .006

    def __post_init__(self):
        for item in fields(self):
            object.__setattr__(self, item.name, _number(getattr(self, item.name), item.name))
        _check_config(self)


@dataclass(frozen=True)
class AxialPairState:
    time_s: float
    position_a_m: float
    position_b_m: float
    velocity_a_m_s: float
    velocity_b_m_s: float

    def __post_init__(self):
        for item in fields(self):
            object.__setattr__(self, item.name, _number(getattr(self, item.name), item.name))
        _check_state(self)


def _check_config(config):
    _need(isinstance(config, AxialImpactConfig), "AxialImpactConfig required")
    for item in fields(config):
        value = _number(getattr(config, item.name), item.name)
        if item.name == "dissipation_s_per_m":
            _need(value >= 0, "dissipation must be nonnegative")
        else:
            _need(value > 0, item.name + " must be positive")
    _need(config.max_indentation_m <= min(config.radius_m, config.target_thickness_m),
          "maximum indentation exceeds the sphere or target depth domain")


def _check_state(state):
    _need(isinstance(state, AxialPairState), "AxialPairState required")
    for item in fields(state):
        _number(getattr(state, item.name), item.name)
    _need(state.time_s >= 0, "state time must be nonnegative")


def _geometry(config, state):
    """Read-only checks shared by pressure sampling and energy diagnostics."""
    _check_config(config)
    _check_state(state)
    top = _number(state.position_b_m + .5 * config.target_thickness_m, "target top")
    depth = _number(config.radius_m + top - state.position_a_m, "signed indentation")
    relative = _number(state.velocity_a_m_s - state.velocity_b_m_s, "relative velocity")
    if depth > 0:
        _need(depth <= config.max_indentation_m, "indentation exceeds the declared positive domain")
        radius_squared = _number(depth * (2 * config.radius_m - depth), "contact radius squared")
        _need(radius_squared > 0, "positive contact disk is not representable")
        radius = _number(sqrt(radius_squared), "contact disk radius")
        _need(radius <= config.target_half_width_m, "contact disk exceeds the finite target top")
    return depth, top, relative


@dataclass(frozen=True)
class ContactSample:
    indentation_m: float
    relative_velocity_m_s: float
    elastic: PressureIntegral
    instantaneous_factor: float
    instantaneous_force_n: float


def sample_contact(config, state):
    """Sample the actual current disk; signed nonpositive depth gives an empty disk.

    elastic is the UNMULTIPLIED pressure-area integral. Its normal points along
    +z on A, and its world plane follows the top of the moving target B. Spatial
    mesh admission and finite-data checks remain those of sphere_plane_case.
    instantaneous_force_n uses the incoming sample velocity, not a kick's future
    midpoint velocity and not an average force over a full dynamics step.
    """
    depth, top, relative = _geometry(config, state)
    elastic = sphere_plane_case(
        radius_m=config.radius_m, indentation_m=max(depth, 0.),
        modulus_pa=config.modulus_pa, resolution_m=config.resolution_m,
        plane_position_world=(0., 0., top), origin_world=(0., 0., 0.),
    )
    elastic_force = _number(elastic.first_force_n[2], "sampled elastic force")
    _need(elastic_force >= 0, "sampled elastic force must be nonnegative")
    factor = max(0., _number(1 - config.dissipation_s_per_m * relative, "instantaneous multiplier"))
    force = _number(elastic_force * factor, "instantaneous force")
    return ContactSample(depth, relative, elastic, factor, force)


@dataclass(frozen=True)
class KickRecord:
    sample: ContactSample
    duration_s: float
    velocity_a_before: float
    velocity_b_before: float
    velocity_a_after: float
    velocity_b_after: float
    relative_mid_velocity_m_s: float
    factor: float
    impulse_n_s: float
    dissipated_j: float


def _kick(config, state, duration):
    duration = _number(duration, "kick duration")
    _need(duration > 0, "positive representable kick duration required")
    sample = sample_contact(config, state)
    elastic_force = sample.elastic.first_force_n[2]
    inverse_mass = _number(1 / config.mass_a_kg + 1 / config.mass_b_kg, "inverse mass sum")
    elastic_impulse = _number(duration * elastic_force, "elastic kick impulse")
    denominator = _number(1 + .5 * elastic_impulse * config.dissipation_s_per_m * inverse_mass,
                          "kick denominator")
    _need(denominator >= 1, "invalid implicit kick denominator")
    impulse = _number(elastic_impulse * sample.instantaneous_factor / denominator, "contact impulse")
    _need(impulse >= 0, "contact impulse must be nonnegative")
    velocity_a = _number(state.velocity_a_m_s + impulse / config.mass_a_kg, "post-kick velocity A")
    velocity_b = _number(state.velocity_b_m_s - impulse / config.mass_b_kg, "post-kick velocity B")
    midpoint = _number(sample.relative_velocity_m_s + .5 * inverse_mass * impulse,
                       "relative midpoint velocity")
    factor = max(0., _number(1 - config.dissipation_s_per_m * midpoint, "midpoint multiplier"))
    # Raw dissipated energy of this fixed-geometry midpoint kick; no energy or
    # velocity correction follows. Preserve any roundoff rather than clamp it.
    dissipated = _number((elastic_impulse - impulse) * midpoint, "kick dissipation")
    return KickRecord(sample, duration, state.velocity_a_m_s, state.velocity_b_m_s,
                      velocity_a, velocity_b, midpoint, factor, impulse, dissipated)


@dataclass(frozen=True)
class ImpactStep:
    entry: AxialPairState
    drifted: AxialPairState
    exit: AxialPairState
    first: KickRecord
    second: KickRecord
    impulse_n_s: float
    average_force_n: float
    dissipated_j: float

    def to_dict(self):
        """Complete owned phase/kick/quadrature record, with finite JSON output."""
        result = asdict(self)
        json.dumps(result, allow_nan=False)
        return result


def step_pair(config, state, dt_s):
    """Commit one local kick-drift-kick result only after BOTH samples succeed.

    entry is the original immutable input. drifted contains first-kick velocities
    and positions advanced by h for both bodies; its time is entry.time+h. The
    second kick changes only velocities. No shared state, partial-step record,
    fallback load, or analytic-force trajectory is published if either kick fails.
    """
    _check_config(config)
    _check_state(state)
    step = _number(dt_s, "step duration")
    _need(0 < step <= .01, "step duration must be in (0, .01] seconds")
    next_time = _number(state.time_s + step, "next time")
    _need(next_time > state.time_s, "step time does not advance representably")
    first = _kick(config, state, .5 * step)
    drifted = AxialPairState(
        next_time,
        state.position_a_m + step * first.velocity_a_after,
        state.position_b_m + step * first.velocity_b_after,
        first.velocity_a_after, first.velocity_b_after,
    )
    second = _kick(config, drifted, .5 * step)
    endpoint = AxialPairState(next_time, drifted.position_a_m, drifted.position_b_m,
                             second.velocity_a_after, second.velocity_b_after)
    impulse = _number(first.impulse_n_s + second.impulse_n_s, "step impulse")
    average = _number(impulse / step, "step average force")
    dissipated = _number(first.dissipated_j + second.dissipated_j, "step dissipation")
    return ImpactStep(state, drifted, endpoint, first, second, impulse, average, dissipated)


def mechanical_energy_j(config, state):
    """Two-body kinetic energy plus the ideal continuous-field potential, diagnostic only.

    This function does not sample pressure, advance state, alter impulses, or
    compensate for the disk's spatial quadrature error. The same positive-depth
    model domain is required as for contact sampling.
    """
    depth, _, _ = _geometry(config, state)
    kinetic = (.5 * config.mass_a_kg * state.velocity_a_m_s * state.velocity_a_m_s
               + .5 * config.mass_b_kg * state.velocity_b_m_s * state.velocity_b_m_s)
    potential = 0.
    if depth > 0:
        potential = pi * config.modulus_pa * depth * depth * depth * (1 / 3 - depth / (6 * config.radius_m))
    return _number(kinetic + potential, "mechanical energy")
