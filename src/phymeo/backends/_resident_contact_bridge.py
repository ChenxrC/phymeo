"""Owned HOST narrow-phase/normal-load packets for resident fixed-root trees.

This module neither integrates nor publishes a physical state. It consumes the
actual stored entry/candidate frames supplied by the resident plan. All forces
come from the existing finite patch and normal law, with friction explicitly zero.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..constraints.collision_shapes import (
    MAX_SHAPES,
    MAX_VERTICES,
    CollisionKey,
    CollisionMotion,
    CollisionShape,
)
from ..constraints.robot_contact import RobotContactMaterial
from ..geometry._convex_hull import HullShape
from ..geometry.contact_patch_adapter import ContactSite, contact_patch_sample
from ..solvers.contact_patch_loads import sample_patch_loads

_ROTATION_DOMAIN = 5e-5
_CONSTRUCTION_M = 1e-10
# A floating resident tree has up to 32 joints plus its actual root body.
_MAX_TREE_GROUPS = 33


def _integer(value, name, low=0, high=2**31 - 1):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _owned(value, shape, name):
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a finite real array with shape {shape}")
    array = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite FP64")
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(shape)


def _key(value):
    if not isinstance(value, CollisionKey):
        raise TypeError("expected CollisionKey")
    return CollisionKey(value.world, value.body, value.shape)


def _pair(value, world):
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("contact pair must be a canonical two-key tuple")
    first, second = map(_key, value)
    if first.world != world or second.world != world or not first < second:
        raise ValueError("contact pair must be canonical and belong to this world")
    if first.body == second.body:
        raise ValueError("contact pair requires two distinct physical owners")
    return first, second


def _shape(value):
    if not isinstance(value, CollisionShape):
        raise TypeError("expected CollisionShape")
    result = replace(value, key=_key(value.key))
    if result.kind not in ("sphere", "capsule", "hull") or (
        result.kind == "hull" and result.radius
    ):
        raise ValueError("resident contact supports sphere, finite capsule and unrounded hull only")
    return result


def _law(value):
    if not isinstance(value, RobotContactMaterial):
        raise TypeError("expected the original RobotContactMaterial")
    result = RobotContactMaterial(
        value.stiffness, value.damping, value.friction, value.slip_speed_m_s
    )
    if result.friction != 0:
        raise ValueError("resident contact first slice requires zero friction")
    return result


@dataclass(frozen=True, eq=False)
class ContactShapeFrame:
    """One source shape in its actual stored frame; reference is the real COM.

    Static geometry is already in world coordinates: zero origin, identity R,
    zero velocities and a reference equal to the declared vertices' mean. The
    rotation-domain check is an engineering input bound, never a correction.
    """

    shape: CollisionShape
    world_points_m: np.ndarray
    group_index: int | None
    position_m: np.ndarray
    rotation: np.ndarray
    reference_point_m: np.ndarray
    linear_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray
    rotation_orthogonality_error: float = field(init=False, default=0.0)

    def __post_init__(self):
        shape = _shape(self.shape)
        points = _owned(self.world_points_m, (len(shape.vertex_ids), 3), "world material points")
        group = (
            None
            if self.group_index is None
            else _integer(self.group_index, "group index", 0, _MAX_TREE_GROUPS - 1)
        )
        values = {
            name: _owned(getattr(self, name), (3,), name)
            for name in (
                "position_m",
                "reference_point_m",
                "linear_velocity_m_s",
                "angular_velocity_rad_s",
            )
        }
        rotation = _owned(self.rotation, (3, 3), "stored rotation")
        try:
            with np.errstate(over="raise", invalid="raise"):
                error = float(np.max(abs(rotation.T @ rotation - np.eye(3))))
                determinant = float(np.linalg.det(rotation))
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise ValueError("stored rotation is not representable") from exc
        if not np.isfinite(determinant) or determinant <= 0 or error > _ROTATION_DOMAIN:
            raise ValueError(
                "stored rotation lies outside the positive near-rigid engineering domain"
            )
        if group is None:
            with np.errstate(over="ignore", invalid="ignore"):
                center = points.mean(axis=0)
            if (
                np.any(values["position_m"] != 0)
                or not np.array_equal(rotation, np.eye(3))
                or np.any(values["linear_velocity_m_s"] != 0)
                or np.any(values["angular_velocity_rad_s"] != 0)
                or not np.isfinite(center).all()
                or np.max(abs(values["reference_point_m"] - center)) > _CONSTRUCTION_M
            ):
                raise ValueError(
                    "static frame must retain world vertices, mean reference and zero motion"
                )
        if shape.kind == "capsule" and np.array_equal(points[0], points[1]):
            raise ValueError("capsule requires a nonzero finite axis")
        if shape.kind == "hull":
            HullShape(
                shape, CollisionMotion(points)
            )  # Invalid full geometry cannot hide behind no candidates.
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "world_points_m", points)
        object.__setattr__(self, "group_index", group)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "rotation_orthogonality_error", error)
        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, eq=False)
class ResidentContactSite:
    """One exact input-time site, optionally finished with candidate motion/work."""

    pair: tuple[CollisionKey, CollisionKey]
    geometry: ContactSite
    law: RobotContactMaterial
    first_velocity_m_s: np.ndarray
    second_velocity_m_s: np.ndarray
    first_force_n: np.ndarray
    second_force_n: np.ndarray
    first_reference_point_m: np.ndarray
    second_reference_point_m: np.ndarray
    first_material_displacement_m: np.ndarray | None = None
    second_material_displacement_m: np.ndarray | None = None
    first_application_displacement_m: np.ndarray | None = None
    second_application_displacement_m: np.ndarray | None = None
    application_work_j: float | None = None
    material_work_j: float | None = None

    def __post_init__(self):
        if not isinstance(self.geometry, ContactSite):
            raise TypeError("resident site requires the original ContactSite")
        if (
            not isinstance(self.pair, tuple)
            or len(self.pair) != 2
            or not isinstance(self.pair[0], CollisionKey)
        ):
            raise ValueError("resident site requires a canonical two-key pair")
        pair = _pair(self.pair, self.pair[0].world)
        object.__setattr__(self, "pair", pair)
        object.__setattr__(self, "geometry", replace(self.geometry))
        object.__setattr__(self, "law", _law(self.law))
        for name in (
            "first_velocity_m_s",
            "second_velocity_m_s",
            "first_force_n",
            "second_force_n",
            "first_reference_point_m",
            "second_reference_point_m",
        ):
            object.__setattr__(self, name, _owned(getattr(self, name), (3,), name))
        motion_fields = (
            "first_material_displacement_m",
            "second_material_displacement_m",
            "first_application_displacement_m",
            "second_application_displacement_m",
        )
        values = [
            getattr(self, name)
            for name in (*motion_fields, "application_work_j", "material_work_j")
        ]
        if all(value is None for value in values):
            return
        if any(value is None for value in values):
            raise ValueError("candidate motion and both works must be recorded together")
        for name in motion_fields:
            object.__setattr__(self, name, _owned(getattr(self, name), (3,), name))
        for name in ("application_work_j", "material_work_j"):
            value = np.asarray(getattr(self, name))
            if value.shape != () or value.dtype.kind not in "fiu" or not np.isfinite(value):
                raise ValueError("actual contact work must be a finite scalar")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True, eq=False)
class ResidentStaticReaction:
    reference_point_m: np.ndarray
    force_n: np.ndarray
    torque_nm: np.ndarray

    def __post_init__(self):
        for name in ("reference_point_m", "force_n", "torque_nm"):
            object.__setattr__(self, name, _owned(getattr(self, name), (3,), name))


def _frames(values, world, group_count):
    if not isinstance(values, Mapping) or len(values) > MAX_SHAPES:
        raise ValueError("frames must be a bounded shape-key mapping")
    result, groups = {}, {}
    vertex_count = 0
    for raw_key, frame in values.items():
        key = _key(raw_key)
        if not isinstance(frame, ContactShapeFrame) or key != frame.shape.key or key.world != world:
            raise ValueError("frame key, shape source and world must agree")
        vertex_count += len(frame.shape.vertex_ids)
        if vertex_count > MAX_VERTICES:
            raise ValueError("frame material vertex count exceeds the original model bound")
        group = frame.group_index
        if group is not None:
            if group >= group_count:
                raise ValueError("contact group index exceeds the real tree group count")
            previous = groups.setdefault(group, frame)
            for name in (
                "position_m",
                "rotation",
                "reference_point_m",
                "linear_velocity_m_s",
                "angular_velocity_rad_s",
            ):
                if not np.array_equal(getattr(previous, name), getattr(frame, name)):
                    raise ValueError(
                        "shapes in one group must share the same stored FK/COM/velocity source"
                    )
        # ContactShapeFrame owns immutable arrays and a canonical immutable shape;
        # sharing that value cannot alias the caller's engine or source arrays.
        result[key] = frame
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True, eq=False)
class ContactLoads:
    sites: tuple[ResidentContactSite, ...]
    wrench_fp64: np.ndarray
    static_reactions: Mapping[CollisionKey, ResidentStaticReaction]
    frames: Mapping[CollisionKey, ContactShapeFrame]
    world: int
    finalized: bool = False

    def __post_init__(self):
        world = _integer(self.world, "world")
        raw = np.asarray(self.wrench_fp64)
        if raw.ndim != 2 or raw.shape[1:] != (6,):
            raise ValueError("contact wrench must contain six COM components per group")
        count = _integer(len(raw), "group count", 1, _MAX_TREE_GROUPS)
        frames = _frames(self.frames, world, count)
        sites = tuple(self.sites)
        if len(sites) > 4096 or any(not isinstance(site, ResidentContactSite) for site in sites):
            raise ValueError("contact sites must be a bounded immutable site tuple")
        if type(self.finalized) is not bool:
            raise ValueError("finalized must be bool")
        for site in sites:
            if any(key not in frames for key in site.pair):
                raise ValueError("contact site refers to an absent frame source")
            if (site.application_work_j is not None) != self.finalized:
                raise ValueError("site motion fields disagree with the staging phase")
        if not isinstance(self.static_reactions, Mapping):
            raise TypeError("static reactions must be a shape-key mapping")
        reactions = {}
        for raw_key, value in self.static_reactions.items():
            key = _key(raw_key)
            if (
                key not in frames
                or frames[key].group_index is not None
                or not isinstance(value, ResidentStaticReaction)
            ):
                raise ValueError("static reaction must name its actual static frame")
            if not np.array_equal(value.reference_point_m, frames[key].reference_point_m):
                raise ValueError("static reaction reference differs from the declared frame")
            reactions[key] = value
        object.__setattr__(self, "world", world)
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "wrench_fp64", _owned(raw, (count, 6), "contact COM wrench"))
        object.__setattr__(self, "frames", frames)
        object.__setattr__(
            self, "static_reactions", MappingProxyType(dict(sorted(reactions.items())))
        )


class ContactSiteCapacityExceeded(ValueError):
    """Complete geometric site count, including sites whose normal force is zero."""

    def __init__(self, count, limit):
        self.count = int(count)
        self.limit = int(limit)
        super().__init__(f"contact site count {count} exceeds capacity {limit}")


def sample_tree_contacts(
    *,
    world,
    frames,
    materials_by_pair,
    candidates,
    group_count,
    site_capacity,
    tree_dynamic_pairs=False,
):
    """Sample each declared candidate once; return transient HOST loads, never a step.

    Pair ordering is canonicalized for deterministic rows, not persistent identity.
    All patch sites count before any force is evaluated. No global pair table or
    candidate inference is constructed here; the resident plan supplies candidates.
    """
    if type(tree_dynamic_pairs) is not bool:
        raise TypeError("tree_dynamic_pairs must be a bool")
    world = _integer(world, "world")
    count = _integer(group_count, "group count", 1, _MAX_TREE_GROUPS)
    capacity = _integer(site_capacity, "site capacity", 0, 4096)
    frames = _frames(frames, world, count)
    if not isinstance(materials_by_pair, Mapping):
        raise TypeError("materials_by_pair must explicitly map canonical pairs to original laws")
    laws = {}
    for raw_pair, value in materials_by_pair.items():
        pair = _pair(raw_pair, world)
        if any(key not in frames for key in pair):
            raise ValueError("material pair refers to an absent frame source")
        first_group, second_group = (frames[key].group_index for key in pair)
        if first_group is None and second_group is None:
            raise ValueError("contact requires at least one actual dynamic owner")
        if first_group is not None and second_group is not None:
            if not tree_dynamic_pairs:
                raise ValueError(
                    "first contact slice requires exactly one dynamic and one static owner"
                )
            if first_group == second_group or pair[0].body == pair[1].body:
                raise ValueError("dynamic contact requires two distinct physical groups")
        laws[pair] = _law(value)
    maximum_pairs = len(frames) * (len(frames) - 1) // 2
    selected = set()
    for index, value in enumerate(candidates):
        if index >= maximum_pairs:
            raise ValueError("candidate list exceeds distinct declared shape pairs")
        pair = _pair(value, world)
        if pair in selected or pair not in laws:
            raise ValueError("candidate requires one unique pair with an explicit material law")
        selected.add(pair)
    patches, sites_count = [], 0
    for pair in sorted(selected):
        first, second = (frames[key] for key in pair)
        patch = contact_patch_sample(
            first.shape, second.shape, first.world_points_m, second.world_points_m
        )
        sites_count += len(patch.sites)
        if sites_count <= capacity:
            patches.append((pair, patch))
    if sites_count > capacity:
        raise ContactSiteCapacityExceeded(sites_count, capacity)
    wrench = np.zeros((count, 6), dtype=float)
    static = {key: np.zeros(6) for key, frame in frames.items() if frame.group_index is None}
    rows = []
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for pair, patch in patches:
                first, second = (frames[key] for key in pair)
                points = np.array([site.load_point for site in patch.sites])
                velocities = [
                    frame.linear_velocity_m_s
                    + np.cross(frame.angular_velocity_rad_s, points - frame.reference_point_m)
                    for frame in (first, second)
                ]
                loads = sample_patch_loads(patch, laws[pair], *velocities)
                for index, geometry in enumerate(patch.sites):
                    forces = loads["first_force"][index], loads["second_force"][index]
                    for key, frame, force in zip(pair, (first, second), forces, strict=True):
                        moment = np.cross(
                            np.asarray(geometry.load_point) - frame.reference_point_m, force
                        )
                        target = (
                            static[key] if frame.group_index is None else wrench[frame.group_index]
                        )
                        target[:3] += force
                        target[3:] += moment
                    rows.append(
                        ResidentContactSite(
                            pair,
                            geometry,
                            laws[pair],
                            velocities[0][index],
                            velocities[1][index],
                            forces[0],
                            forces[1],
                            first.reference_point_m,
                            second.reference_point_m,
                        )
                    )
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError("contact load assembly is not representable") from exc
    reactions = {
        key: ResidentStaticReaction(frames[key].reference_point_m, value[:3], value[3:])
        for key, value in static.items()
    }
    return ContactLoads(tuple(rows), wrench, reactions, frames, world)


def finish_contact_loads(loads, candidate_frames):
    """Transport original entry anchors with stored candidate frames; never resample.

    1e-10 m validates packet/source reconstruction only. It is neither a physical
    accuracy claim nor a contact energy gate. Candidate velocities cannot alter
    the entry load. Returned frames remain the owned original entry packet.
    """
    if not isinstance(loads, ContactLoads) or loads.finalized:
        raise ValueError("finish requires a transient ContactLoads entry packet")
    candidates = _frames(candidate_frames, loads.world, len(loads.wrench_fp64))
    if candidates.keys() != loads.frames.keys():
        raise ValueError("candidate shape sources differ from the original entry packet")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for key, entry in loads.frames.items():
                end = candidates[key]
                if end.shape != entry.shape or end.group_index != entry.group_index:
                    raise ValueError(
                        "candidate shape/material IDs/radius/group source differs from entry"
                    )
                if entry.group_index is None:
                    if not np.array_equal(entry.world_points_m, end.world_points_m):
                        raise ValueError("static material geometry changed during the candidate")
                    continue
                local = np.linalg.solve(
                    entry.rotation, (entry.world_points_m - entry.position_m).T
                ).T
                reproduced = local @ end.rotation.T + end.position_m
                if np.max(abs(reproduced - end.world_points_m)) > _CONSTRUCTION_M:
                    raise ValueError(
                        "candidate local material source changed beyond construction consistency"
                    )
                # Actual device COM has independent FP32 rounding. The parent
                # proves its model/group source; do not reconstruct or replace it.

            def displacement(key, point):
                entry, end = loads.frames[key], candidates[key]
                if entry.group_index is None:
                    return np.zeros(3)
                local = np.linalg.solve(entry.rotation, np.asarray(point) - entry.position_m)
                return end.position_m + end.rotation @ local - point

            rows = []
            for site in loads.sites:
                first, second = site.pair
                witness = site.geometry.witness
                d1 = displacement(first, witness.first_point)
                d2 = displacement(second, witness.second_point)
                a1 = displacement(first, site.geometry.load_point)
                a2 = displacement(second, site.geometry.load_point)
                rows.append(
                    replace(
                        site,
                        first_material_displacement_m=d1,
                        second_material_displacement_m=d2,
                        first_application_displacement_m=a1,
                        second_application_displacement_m=a2,
                        application_work_j=float(
                            site.first_force_n @ a1 + site.second_force_n @ a2
                        ),
                        material_work_j=float(site.first_force_n @ d1 + site.second_force_n @ d2),
                    )
                )
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError("candidate contact transport is not representable") from exc
    return ContactLoads(
        tuple(rows), loads.wrench_fp64, loads.static_reactions, loads.frames, loads.world, True
    )
