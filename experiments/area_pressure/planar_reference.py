"""Planar extruded pressure-field dynamics against a fixed convex support.

This is a three-DOF NumPy reference, not full 3D contact or a product solver.
Body-local coordinates are relative to its declared COM; geometry is never
recentered. Positive theta_y maps (u,w) to (cos*u+sin*w,-sin*u+cos*w).
The material pressure is kappa times the nearest inward edge distance. Every
support boundary edge can contribute; the target is not an infinite top plane.
"""

import json
from dataclasses import asdict, dataclass, fields
from math import cos, sin, sqrt
from numbers import Integral

import numpy as np

from .finite_support_reference import _area, _clip, _numeric, _polygon
from .reference import _real, _tuples

_LENGTH_TOL = 1e-12
_MAX_SEGMENTS = 256
_MAX_POINTS = 512
_MAX_EDGE_SEGMENTS = 29
_RESIDUAL_TOL = 1e-12
_ANGULAR_LENGTH = 0.02
_BODY = ((-0.02, -0.01), (0.02, -0.01), (0.02, 0.01), (-0.02, 0.01))
_SUPPORT = ((-0.04, -0.01), (0.04, -0.01), (0.04, 0.0), (-0.04, 0.0))
_CONFIG_SCALARS = (
    "width_m",
    "mass_kg",
    "inertia_kg_m2",
    "stiffness_pa_per_m",
    "max_depth_m",
    "dissipation_s_per_m",
    "friction",
    "slip_speed_m_s",
    "gravity_m_s2",
)


def _need(condition, message):
    if not condition:
        raise ValueError("planar pressure reference: " + message)


def _number(value, name):
    try:
        return _real(value, name)
    except OverflowError as error:
        raise ValueError("planar pressure reference: nonfinite " + name) from error


def _finite(*arrays):
    _need(all(np.isfinite(value).all() for value in arrays), "nonfinite numerical result")


def _small_polygon(value, name):
    _need(
        isinstance(value, (tuple, list, np.ndarray))
        and (not isinstance(value, np.ndarray) or value.ndim == 2)
        and 3 <= len(value) <= 8,
        name + " requires 3..8 vertices",
    )
    return _polygon(value, name)


class _JsonRecord:
    def to_dict(self):
        """Detached complete record; JSON must never silently contain nonfinite data."""
        result = asdict(self)
        json.dumps(result, allow_nan=False)
        return result


@dataclass(frozen=True)
class PlanarConfig(_JsonRecord):
    body_polygon_xz: tuple = _BODY
    support_polygon_xz: tuple = _SUPPORT
    width_m: float = 0.04
    mass_kg: float = 0.5
    inertia_kg_m2: float = 1 / 12000
    stiffness_pa_per_m: float = 1e7
    max_depth_m: float = 0.002
    dissipation_s_per_m: float = 20.0
    friction: float = 0.5
    slip_speed_m_s: float = 0.001
    gravity_m_s2: float = 9.81

    def __post_init__(self):
        object.__setattr__(
            self, "body_polygon_xz", _small_polygon(self.body_polygon_xz, "body polygon")
        )
        object.__setattr__(
            self, "support_polygon_xz", _small_polygon(self.support_polygon_xz, "support polygon")
        )
        for name in _CONFIG_SCALARS:
            object.__setattr__(self, name, _number(getattr(self, name), name))
        _check_config(self)


@dataclass(frozen=True)
class PlanarState(_JsonRecord):
    time_s: float
    x_m: float
    z_m: float
    angle_rad: float
    vx_m_s: float
    vz_m_s: float
    omega_rad_s: float

    def __post_init__(self):
        for item in fields(self):
            object.__setattr__(self, item.name, _number(getattr(self, item.name), item.name))
        _check_state(self)


def _check_config(config):
    _need(isinstance(config, PlanarConfig), "PlanarConfig required")
    _small_polygon(config.body_polygon_xz, "body polygon")
    _small_polygon(config.support_polygon_xz, "support polygon")
    for name in _CONFIG_SCALARS:
        value = _number(getattr(config, name), name)
        if name in ("dissipation_s_per_m", "friction", "gravity_m_s2"):
            _need(value >= 0, name + " must be nonnegative")
        else:
            _need(value > 0, name + " must be positive")


def _check_state(state):
    _need(isinstance(state, PlanarState), "PlanarState required")
    for item in fields(state):
        _number(getattr(state, item.name), item.name)
    _need(state.time_s >= 0, "time must be nonnegative")
    _need(abs(state.x_m) <= 1 and abs(state.z_m) <= 1, "COM coordinates exceed 1 m")


def _velocity(state):
    return np.array((state.vx_m_s, state.vz_m_s, state.omega_rad_s), dtype=np.float64)


def _world_body(config, state):
    c, s = cos(state.angle_rad), sin(state.angle_rad)
    rotation = np.array(((c, s), (-s, c)))
    points = np.asarray(config.body_polygon_xz) @ rotation.T + (state.x_m, state.z_m)
    _finite(points)
    return _small_polygon(_tuples(points), "world body polygon")


def _edges(polygon):
    starts = np.asarray(polygon, dtype=np.float64)
    delta = np.roll(starts, -1, axis=0) - starts
    length = np.hypot(delta[:, 0], delta[:, 1])
    _need(np.all(length > _LENGTH_TOL), "edge is degenerate")
    inward = np.column_stack((-delta[:, 1], delta[:, 0])) / length[:, None]
    _finite(delta, length, inward)
    return starts, delta, length, inward


def _distances(points, starts, inward):
    return np.einsum("pij,ij->pi", np.asarray(points)[:, None, :] - starts[None, :, :], inward)


def _overlap(body, support_starts, support_inward):
    polygon = body
    for start, normal in zip(support_starts, support_inward, strict=True):
        polygon = _clip(polygon, lambda point: float(np.dot(normal, np.asarray(point) - start)))
        if not polygon:
            break
    return polygon


def _potential(config, overlap, starts, inward):
    """Affine integration per nearest-edge cell; inspect ALL cell vertices for depth."""
    potential = 0.0
    maximum = 0.0
    for owner in range(len(starts)):
        cell = overlap
        for other in range(len(starts)):
            if owner == other or not cell:
                continue
            cell = _clip(
                cell,
                lambda point: float(
                    np.dot(inward[other], np.asarray(point) - starts[other])
                    - np.dot(inward[owner], np.asarray(point) - starts[owner])
                ),
            )
        if not cell:
            continue
        _need(len(cell) <= 40, "nearest-edge cell exceeds vertex budget")
        vertices = np.asarray(cell)
        depths = _distances(vertices, starts, inward).min(axis=1)
        _finite(depths)
        _need(np.min(depths) >= -_LENGTH_TOL, "cell left the body")
        maximum = max(maximum, float(depths.max()), 0.0)
        anchor = vertices[0]
        for i in range(1, len(vertices) - 1):
            triangle = (tuple(anchor), tuple(vertices[i]), tuple(vertices[i + 1]))
            area = _area(triangle)
            _need(area >= 0, "negative cell triangle area")
            centroid = anchor + ((vertices[i] - anchor) + (vertices[i + 1] - anchor)) / 3
            depth = float(np.dot(inward[owner], centroid - starts[owner]))
            _need(depth >= -_LENGTH_TOL, "negative cell pressure depth")
            potential += config.width_m * config.stiffness_pa_per_m * area * max(0.0, depth)
    maximum = _number(maximum, "maximum overlap depth")
    _need(maximum <= config.max_depth_m, "overlap exceeds maximum pressure depth")
    return _number(potential, "pressure potential"), maximum


def _clipped_interval(start, delta, body_starts, body_inward):
    low, high = 0.0, 1.0
    for origin, normal in zip(body_starts, body_inward, strict=True):
        left = float(np.dot(normal, start + low * delta - origin))
        right = float(np.dot(normal, start + high * delta - origin))
        _finite(left, right)
        if left < 0 and right < 0:
            return None
        if (left < 0) != (right < 0):
            cut = low + (high - low) * left / (left - right)
            if left < 0:
                low = cut
            else:
                high = cut
    return low, high


def _boundary_samples(config, body_starts, body_inward, support_edges):
    starts, deltas, lengths, support_inward = support_edges
    points, normals, areas, pressures, sources = [], [], [], [], []
    segment_count = 0
    for support_id, (start, delta, length) in enumerate(zip(starts, deltas, lengths, strict=True)):
        interval = _clipped_interval(start, delta, body_starts, body_inward)
        if interval is None:
            continue
        low, high = interval
        if (high - low) * length <= _LENGTH_TOL:
            continue
        end_depths = _distances(
            (start + low * delta, start + high * delta), body_starts, body_inward
        )
        roots = [low, high]
        for i in range(len(body_starts)):
            for j in range(i + 1, len(body_starts)):
                a = float(end_depths[0, i] - end_depths[0, j])
                b = float(end_depths[1, i] - end_depths[1, j])
                # Identically tied fields have no isolated root. A line segment
                # is integrated once, not once for each equally near body edge.
                if (a < 0 < b) or (b < 0 < a):
                    root = low + (high - low) * a / (a - b)
                    if low < root < high:
                        _need(len(roots) < _MAX_EDGE_SEGMENTS + 1, "too many distance roots")
                        roots.append(root)
        roots.sort()
        cuts = [roots[0]]
        for root in roots[1:]:
            if (root - cuts[-1]) * length > _LENGTH_TOL:
                cuts.append(root)
        if (high - cuts[-1]) * length <= _LENGTH_TOL:
            cuts[-1] = high
        else:
            cuts.append(high)
        _need(len(cuts) - 1 <= _MAX_EDGE_SEGMENTS, "edge segment budget exceeded")
        for a, b in zip(cuts[:-1], cuts[1:], strict=True):
            endpoint_depths = _distances(
                (start + a * delta, start + b * delta), body_starts, body_inward
            )
            # A coincident body boundary has identically zero pressure. The fixed
            # length cleanup tolerance removes roundoff, not a finite pressure cap.
            if np.any(np.max(np.abs(endpoint_depths), axis=0) <= _LENGTH_TOL):
                continue
            midpoint = 0.5 * (a + b)
            middle_depths = _distances((start + midpoint * delta,), body_starts, body_inward)[0]
            owner = int(np.argmin(middle_depths))
            if middle_depths[owner] <= 0:
                continue
            # np.argmin returns the smallest body edge ID in an exact full-segment tie.
            _need(
                segment_count < _MAX_SEGMENTS and len(points) + 2 <= _MAX_POINTS,
                "contact quadrature budget exceeded",
            )
            segment_count += 1
            half = 0.5 * (b - a)
            weight = _number(config.width_m * length * half, "contact area weight")
            _need(weight > 0, "contact area is not representable")
            for coordinate in (midpoint - half / sqrt(3), midpoint + half / sqrt(3)):
                point = start + coordinate * delta
                depth = float(_distances((point,), body_starts, body_inward).min())
                _need(depth >= -_LENGTH_TOL, "contact point left the body")
                pressure = _number(config.stiffness_pa_per_m * max(0.0, depth), "contact pressure")
                points.append(tuple(map(float, point)))
                normals.append(tuple(map(float, -support_inward[support_id])))
                areas.append(weight)
                pressures.append(pressure)
                sources.append((support_id, owner))
    return tuple(points), tuple(normals), tuple(areas), tuple(pressures), tuple(sources)


def _basis(points, normals, state):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 2)
    arms = points - (state.x_m, state.z_m)
    tangents = np.column_stack((-normals[:, 1], normals[:, 0]))
    bn = np.column_stack((normals, normals[:, 0] * arms[:, 1] - normals[:, 1] * arms[:, 0]))
    bt = np.column_stack((tangents, tangents[:, 0] * arms[:, 1] - tangents[:, 1] * arms[:, 0]))
    _finite(bn, bt)
    return bn, bt


@dataclass(frozen=True)
class PlanarContact(_JsonRecord):
    state: PlanarState
    body_world_xz: tuple
    points_xz: tuple
    normals_xz: tuple
    area_weights_m2: tuple
    pressure_pa: tuple
    source_edges: tuple
    area_m2: float
    potential_j: float
    max_depth_m: float
    elastic_generalized_force: tuple


def sample_contact(config, state):
    """Integrate material pressure on the whole fixed boundary and in overlap cells."""
    _check_config(config)
    _check_state(state)
    body = _world_body(config, state)
    body_starts, _, _, body_inward = _edges(body)
    support_edges = _edges(config.support_polygon_xz)
    overlap = _overlap(body, support_edges[0], support_edges[3])
    potential, maximum = _potential(config, overlap, body_starts, body_inward)
    if overlap:
        points, normals, areas, pressure, sources = _boundary_samples(
            config, body_starts, body_inward, support_edges
        )
    else:
        points = normals = areas = pressure = sources = ()
    bn, _ = _basis(points, normals, state)
    with np.errstate(over="ignore", invalid="ignore"):
        elastic = (np.asarray(areas) * np.asarray(pressure)) @ bn
    _finite(elastic)
    return PlanarContact(
        state,
        body,
        points,
        normals,
        areas,
        pressure,
        sources,
        _number(sum(areas), "contact area"),
        potential,
        maximum,
        tuple(map(float, elastic)),
    )


def _contact_arrays(config, sample):
    _need(isinstance(sample, PlanarContact), "PlanarContact required")
    _check_state(sample.state)
    body = _small_polygon(sample.body_world_xz, "sample body")
    _need(len(body) == len(config.body_polygon_xz), "sample body edge count differs from config")
    _need(
        isinstance(sample.points_xz, (tuple, list, np.ndarray))
        and (not isinstance(sample.points_xz, np.ndarray) or sample.points_xz.ndim == 2)
        and len(sample.points_xz) <= _MAX_POINTS,
        "invalid point capacity",
    )
    count = len(sample.points_xz)
    points = _numeric(sample.points_xz, (count, 2), "sample points")
    normals = _numeric(sample.normals_xz, (count, 2), "sample normals")
    areas = _numeric(sample.area_weights_m2, (count,), "sample areas")
    pressure = _numeric(sample.pressure_pa, (count,), "sample pressure")
    _need(
        np.all(np.abs(points) <= 1) and np.all(areas > 0) and np.all(pressure >= 0),
        "invalid contact point, area or pressure",
    )
    _need(
        np.allclose(np.hypot(normals[:, 0], normals[:, 1]), 1.0, rtol=0, atol=1e-10),
        "sample normals must be unit length",
    )
    _need(
        isinstance(sample.source_edges, (tuple, list)) and len(sample.source_edges) == count,
        "invalid source edge mapping",
    )
    for pair in sample.source_edges:
        _need(
            isinstance(pair, (tuple, list))
            and len(pair) == 2
            and all(
                isinstance(index, Integral) and not isinstance(index, (bool, np.bool_))
                for index in pair
            )
            and 0 <= pair[0] < len(config.support_polygon_xz)
            and 0 <= pair[1] < len(body),
            "invalid source edge ID",
        )
    for name in ("area_m2", "potential_j", "max_depth_m"):
        _need(_number(getattr(sample, name), name) >= 0, name + " must be nonnegative")
    _need(sample.max_depth_m <= config.max_depth_m, "sample depth exceeds config")
    _numeric(sample.elastic_generalized_force, (3,), "elastic generalized force")
    bn, bt = _basis(points, normals, sample.state)
    return bn, bt, areas, pressure


def _law(config, bn, bt, areas, elastic, velocity, *, jacobian=False):
    """Actual pressure law and its analytic velocity derivative at fixed geometry."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        vn, vt = bn @ velocity, bt @ velocity
        unclipped = 1 - config.dissipation_s_per_m * vn
        normal = elastic * np.maximum(unclipped, 0.0)
        speed = np.hypot(vt, config.slip_speed_m_s)
        sliding = vt / speed
        tangent = -config.friction * normal * sliding
        generalized = (areas * normal) @ bn + (areas * tangent) @ bt
        power = float(np.sum(areas * ((elastic - normal) * vn - tangent * vt)))
    _finite(vn, vt, unclipped, normal, speed, sliding, tangent, generalized, power)
    _need(power >= 0, "negative contact dissipation")
    derivative = None
    if jacobian:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            dn = np.where(unclipped > 0, -config.dissipation_s_per_m * elastic, 0.0)
            ds = (config.slip_speed_m_s / speed) ** 2 / speed
            derivative = bn.T @ ((areas * dn)[:, None] * bn)
            # Cross term d(pt)/d(vn) is essential: friction magnitude depends on pn.
            derivative -= config.friction * bt.T @ ((areas * dn * sliding)[:, None] * bn)
            derivative -= config.friction * bt.T @ ((areas * normal * ds)[:, None] * bt)
        _finite(derivative)
    return generalized, normal, tangent, vn, vt, power, derivative


@dataclass(frozen=True)
class PlanarWrench(_JsonRecord):
    generalized_force: tuple
    support_force_xz: tuple
    support_moment_origin_nm: float
    normal_pressure_pa: tuple
    tangent_pressure_pa: tuple
    normal_velocity_m_s: tuple
    tangent_velocity_m_s: tuple
    dissipation_power_w: float


def _wrench_record(sample, data):
    generalized, normal, tangent, vn, vt, power, _ = data
    state = sample.state
    reaction_moment = _number(
        -(generalized[2] + state.z_m * generalized[0] - state.x_m * generalized[1]),
        "support origin moment",
    )
    return PlanarWrench(
        tuple(map(float, generalized)),
        tuple(map(float, -generalized[:2])),
        reaction_moment,
        tuple(map(float, normal)),
        tuple(map(float, tangent)),
        tuple(map(float, vn)),
        tuple(map(float, vt)),
        power,
    )


def contact_wrench(config, sample, velocity=None):
    """Apply the prescribed pressure/friction law to a frozen geometry sample.

    velocity is (vx,vz,omega_y). Body generalized moment is about COM; support
    reaction moment is about the WORLD ORIGIN, including the COM lever arm.
    """
    _check_config(config)
    bn, bt, areas, elastic = _contact_arrays(config, sample)
    velocity = _velocity(sample.state) if velocity is None else _numeric(velocity, (3,), "velocity")
    data = _law(config, bn, bt, areas, elastic, velocity)
    return _wrench_record(sample, data)


def _residual(midpoint, before, duration, inverse_mass, external, generalized):
    with np.errstate(over="ignore", invalid="ignore"):
        residual = midpoint - before - 0.5 * duration * inverse_mass * (external + generalized)
    _finite(residual)
    norm = float(np.max(np.abs(residual) * (1.0, 1.0, _ANGULAR_LENGTH)))
    return residual, norm


@dataclass(frozen=True)
class PlanarKick(_JsonRecord):
    sample: PlanarContact
    duration_s: float
    velocity_before: tuple
    velocity_after: tuple
    mid_velocity: tuple
    contact: PlanarWrench
    external_generalized_force: tuple
    contact_impulse: tuple
    external_impulse: tuple
    dissipated_j: float
    external_work_j: float
    iterations: int
    residual_m_s: float


def _kick(config, state, duration, external):
    """At most 24 Newton updates, each with at most 12 actual residual trials."""
    duration = _number(duration, "kick duration")
    _need(duration > 0, "positive representable kick duration required")
    sample = sample_contact(config, state)
    bn, bt, areas, elastic = _contact_arrays(config, sample)
    before = _velocity(state)
    inverse_mass = np.array((1 / config.mass_kg, 1 / config.mass_kg, 1 / config.inertia_kg_m2))
    _finite(inverse_mass)
    midpoint = before.copy()
    accepted_data = None
    for iteration in range(25):
        data = _law(config, bn, bt, areas, elastic, midpoint, jacobian=True)
        residual, norm = _residual(midpoint, before, duration, inverse_mass, external, data[0])
        if norm <= _RESIDUAL_TOL:
            accepted_data = data
            break
        _need(iteration < 24, "Newton update budget exhausted")
        matrix = np.eye(3) - 0.5 * duration * inverse_mass[:, None] * data[-1]
        _finite(matrix)
        try:
            update = np.linalg.solve(matrix, -residual)
        except np.linalg.LinAlgError as error:
            raise ValueError("planar pressure reference: singular Newton Jacobian") from error
        _finite(update)
        improved = False
        last_error = None
        for backtrack in range(12):
            fraction = 2.0 ** (-backtrack)
            candidate = midpoint + fraction * update
            try:
                _finite(candidate)
                trial = _law(config, bn, bt, areas, elastic, candidate)
                _, candidate_norm = _residual(
                    candidate, before, duration, inverse_mass, external, trial[0]
                )
            except ValueError as error:
                last_error = error
                continue
            if candidate_norm <= max(_RESIDUAL_TOL, (1 - 1e-4 * fraction) * norm):
                midpoint = candidate
                improved = True
                break
        if not improved:
            raise ValueError(
                "planar pressure reference: Newton line search exhausted"
            ) from last_error
    _need(accepted_data is not None, "no accepted midpoint solution")
    contact = _wrench_record(sample, accepted_data)
    with np.errstate(over="ignore", invalid="ignore"):
        after = 2 * midpoint - before
        contact_impulse = duration * accepted_data[0]
        external_impulse = duration * external
        dissipated = duration * contact.dissipation_power_w
        external_work = duration * float(np.dot(external, midpoint))
    _finite(after, contact_impulse, external_impulse, dissipated, external_work)
    return PlanarKick(
        sample,
        duration,
        tuple(map(float, before)),
        tuple(map(float, after)),
        tuple(map(float, midpoint)),
        contact,
        tuple(map(float, external)),
        tuple(map(float, contact_impulse)),
        tuple(map(float, external_impulse)),
        float(dissipated),
        float(external_work),
        iteration,
        norm,
    )


@dataclass(frozen=True)
class PlanarStep(_JsonRecord):
    entry: PlanarState
    drifted: PlanarState
    exit: PlanarState
    first: PlanarKick
    second: PlanarKick
    contact_impulse: tuple
    average_contact_generalized_force: tuple
    dissipated_j: float


def step_body(config, state, dt_s, external_force_xz_n=(0.0, 0.0), external_moment_nm=0.0):
    """One transactional KDK step; external input is constant across BOTH kicks.

    The caller owns the declared time schedule. Gravity is added here at COM.
    external_work_j in each kick includes gravity; a ledger containing mg*z
    must separately compute non-gravity input work from the saved midpoint.
    """
    _check_config(config)
    _check_state(state)
    duration = _number(dt_s, "step duration")
    _need(0 < duration <= 0.001, "step duration must be in (0,.001] seconds")
    next_time = _number(state.time_s + duration, "next time")
    _need(next_time > state.time_s, "time does not advance representably")
    force = _numeric(external_force_xz_n, (2,), "external force")
    moment = _number(external_moment_nm, "external moment")
    external = np.array((force[0], force[1] - config.mass_kg * config.gravity_m_s2, moment))
    _finite(external)
    first = _kick(config, state, 0.5 * duration, external)
    half = first.velocity_after
    drifted = PlanarState(
        next_time,
        state.x_m + duration * half[0],
        state.z_m + duration * half[1],
        state.angle_rad + duration * half[2],
        half[0],
        half[1],
        half[2],
    )
    second = _kick(config, drifted, 0.5 * duration, external)
    endpoint = PlanarState(
        next_time, drifted.x_m, drifted.z_m, drifted.angle_rad, *second.velocity_after
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        impulse = np.asarray(first.contact_impulse) + second.contact_impulse
        average = impulse / duration
        dissipated = first.dissipated_j + second.dissipated_j
    _finite(impulse, average, dissipated)
    return PlanarStep(
        state,
        drifted,
        endpoint,
        first,
        second,
        tuple(map(float, impulse)),
        tuple(map(float, average)),
        float(dissipated),
    )


def mechanical_energy_j(config, state):
    """Kinetic + material overlap potential + gravity potential; never a state repair."""
    sample = sample_contact(config, state)
    with np.errstate(over="ignore", invalid="ignore"):
        velocity = _velocity(state)
        kinetic = 0.5 * config.mass_kg * (velocity[0] * velocity[0] + velocity[1] * velocity[1])
        kinetic += 0.5 * config.inertia_kg_m2 * velocity[2] * velocity[2]
        energy = kinetic + sample.potential_j + config.mass_kg * config.gravity_m_s2 * state.z_m
    return _number(energy, "mechanical energy")
