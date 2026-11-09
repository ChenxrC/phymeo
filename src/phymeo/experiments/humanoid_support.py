"""Read-only finite-foot support diagnostics and explicit initial material stations.

These helpers consume real contact loads/accepted observations. They do not
integrate a body, synthesize an impulse, repair state or match rolling contacts.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..constraints.collision_shapes import CollisionKey
from ..dynamics.rotation import quaternion_matrix
from ..runtime.material_stations import MaterialStationRegistration, MaterialStationRegistry
from ..sensors.contact_wrench import measure_contact_wrench
from ..solvers.tangent_history import StationKey


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector")
    return result


def _feet(foot_shapes, ground_shapes):
    if not isinstance(foot_shapes, Mapping) or not foot_shapes:
        raise ValueError("foot_shapes must name at least one finite foot")
    feet = {}
    for name, raw in foot_shapes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("each foot needs a nonempty name")
        shapes = tuple(raw)
        if not shapes or any(not isinstance(key, CollisionKey) for key in shapes):
            raise ValueError("each foot needs actual CollisionKeys")
        feet[name] = shapes
    ground = tuple(ground_shapes)
    selected = tuple(key for keys in feet.values() for key in keys)
    if not ground or any(not isinstance(key, CollisionKey) for key in ground):
        raise ValueError("ground_shapes must contain actual CollisionKeys")
    if len(set(selected)) != len(selected) or len(set(ground)) != len(ground):
        raise ValueError("foot and ground shape declarations must be unique")
    if set(selected) & set(ground):
        raise ValueError("a shape cannot be both foot and ground")
    return feet, ground


@dataclass(frozen=True)
class FootWrench:
    name: str
    force_n: tuple
    moment_nm: tuple
    friction_force_n: tuple
    normal_force_n: float
    contributing_site_sides: int


@dataclass(frozen=True)
class FootSupportReading:
    sample_time_s: float
    commit_time_s: float
    feet: tuple[FootWrench, ...]
    total_force_n: tuple
    total_moment_nm: tuple
    selected_ground_force_n: tuple
    published_floor_force_n: tuple | None
    loaded_feet: tuple[str, ...]
    root_displacement_m: tuple | None
    root_com_velocity_m_s: tuple | None
    root_up_world: tuple | None
    root_tilt_rad: float | None


def read_foot_support(
    reading,
    foot_shapes,
    ground_shapes,
    *,
    reference_world_m=(0.0, 0.0, 0.0),
    up_world=(0.0, 0.0, 1.0),
    floor_rigid_rows=(),
    robot_address=None,
    reference_root_position_m=None,
    positive_load_threshold_n=0.0,
):
    """Sum actual forces and couples, retaining input and publication times.

    Moments use one caller-declared world origin. Site forces already contain
    quadrature weight, so the fraction is never multiplied a second time.
    floor_rigid_rows optionally checks the separately published rigid load
    channel; that channel may also contain loads from unselected participants.
    """
    feet, ground = _feet(foot_shapes, ground_shapes)
    origin, up = _vector(reference_world_m, "reference"), _vector(up_world, "up")
    if not np.isclose(np.linalg.norm(up), 1.0, atol=1e-12, rtol=0):
        raise ValueError("up_world must be a unit vector")
    threshold = float(positive_load_threshold_n)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("positive load threshold must be finite and nonnegative")
    sample, commit = reading.load_time_s, float(reading.time_s)
    if sample is None or not 0 <= float(sample) < commit or reading.step_number < 1:
        raise ValueError("an accepted reading with an input-state load time is required")
    rows, total_force, total_moment = [], np.zeros(3), np.zeros(3)
    for name, shapes in feet.items():
        force, moment, friction, sites = np.zeros(3), np.zeros(3), np.zeros(3), 0
        for shape in shapes:
            actual = measure_contact_wrench(
                reading.loads, shape, counterparts=ground, reference_world_m=origin
            )
            force += actual.force_n
            moment += actual.torque_nm
            friction += actual.friction_force_n
            sites += actual.site_sides
        rows.append(FootWrench(name, tuple(force), tuple(moment), tuple(friction), float(force @ up), sites))
        total_force += force
        total_moment += moment
    selected = tuple(key for keys in feet.values() for key in keys)
    reaction = np.zeros(3)
    for shape in ground:
        reaction += measure_contact_wrench(
            reading.loads, shape, counterparts=selected, reference_world_m=origin
        ).force_n
    published = None
    floor_rows = tuple(floor_rigid_rows)
    if floor_rows:
        if any(isinstance(row, bool) or not isinstance(row, (int, np.integer)) for row in floor_rows):
            raise ValueError("floor rigid rows must be integer indices")
        if len(set(floor_rows)) != len(floor_rows) or any(row < 0 or row >= len(reading.loads.inputs.rigid.force) for row in floor_rows):
            raise ValueError("floor rigid rows must be distinct and in range")
        published = tuple(reading.loads.inputs.rigid.force[list(floor_rows)].sum(axis=0))
    displacement = velocity = root_up = root_tilt = None
    if robot_address is not None:
        robot = reading.robots[tuple(robot_address)]
        if reference_root_position_m is None:
            raise ValueError("root displacement requires an explicit reference position")
        reference = _vector(reference_root_position_m, "reference root position")
        displacement = tuple(np.asarray(robot.q[:3]) - reference)
        velocity = tuple(robot.v[:3])
        root_axis = quaternion_matrix(np.asarray(robot.q)[None, 3:7])[0, :, 2]
        root_up = tuple(root_axis)
        root_tilt = float(np.arctan2(np.linalg.norm(np.cross(root_axis, up)), root_axis @ up))
    elif reference_root_position_m is not None:
        raise ValueError("reference root position requires a robot address")
    return FootSupportReading(
        float(sample), commit, tuple(rows), tuple(total_force), tuple(total_moment),
        tuple(reaction), published,
        tuple(row.name for row in rows if row.normal_force_n > threshold),
        displacement, velocity, root_up, root_tilt,
    )


def foot_material_registry(
    sampled_loads,
    foot_shapes,
    ground_shapes,
    *,
    frames_by_shape,
    shape_vertex_ids,
    chart_ids,
    tangent_axis_local=(1.0, 0.0, 0.0),
    token_start=0,
):
    """Declare every currently loaded foot-local site before the first step.

    frames_by_shape supplies real body-origin (position, rotation) pairs.
    Full original vertex IDs and explicit chart IDs come from the input model.
    Tokens are assigned once after sorting by pair and local anchor, never from
    force magnitude. Reusing this registry requires the same local sites; it is
    deliberately not an automatic rolling/feature-transition matcher.
    """
    feet, ground = _feet(foot_shapes, ground_shapes)
    selected = {key for keys in feet.values() for key in keys}
    axis = _vector(tangent_axis_local, "tangent axis")
    if isinstance(token_start, bool) or not isinstance(token_start, (int, np.integer)) or token_start < 0:
        raise ValueError("token_start must be a nonnegative integer")
    declarations = []
    for load in sampled_loads.contacts:
        if load.first in selected and load.second in ground:
            foot, side = load.first, "first"
        elif load.second in selected and load.first in ground:
            foot, side = load.second, "second"
        else:
            continue
        position, rotation = frames_by_shape[foot]
        position = _vector(position, "foot origin")
        rotation = np.asarray(rotation, dtype=float)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0):
            raise ValueError("foot rotation must be proper orthonormal")
        for site in load.sites:
            if site.force_n <= 0:
                continue
            point = np.asarray(getattr(site.material, side + "_point"))
            local = tuple((point - position) @ rotation)
            declarations.append((load.first, load.second, side, local))
    declarations.sort(key=lambda row: (row[0], row[1], row[3]))
    if len(set(declarations)) != len(declarations):
        raise ValueError("duplicate local foot station")
    registrations = []
    for ordinal, (first, second, side, local) in enumerate(declarations):
        registrations.append(MaterialStationRegistration(
            StationKey(first.world, int(token_start) + ordinal),
            first, second, side, local, tuple(axis),
            chart_ids[first], chart_ids[second],
            first_feature=tuple(shape_vertex_ids[first]),
            second_feature=tuple(shape_vertex_ids[second]),
        ))
    return MaterialStationRegistry(registrations)
