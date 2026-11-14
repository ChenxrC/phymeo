"""Deterministic geometric ranking of recorded planar push candidates.

This heuristic does not predict delivered forces, yaw response, reachability,
collision-free robot motion, or task success. It never reruns the query or FK.
"""

from dataclasses import dataclass
from math import hypot, isfinite, pi

import numpy as np

from .manipulation_diagnostic import _require
from .manipulation_path import _items, _number
from .manipulation_push import (
    NativePlanarPushContactCandidate,
    NativePlanarPushContacts,
    PlanarPushBoundaryEdge,
    PlanarPushRectangle,
)


@dataclass(frozen=True)
class NativePlanarPushRanking:
    index: int
    usable: bool
    translation_alignment: float
    nominal_yaw_moment: float
    sort_key: tuple[int, float, float, int]


def _integer(value, name, minimum, maximum):
    _require(type(value) is int and minimum <= value <= maximum, name + " has an invalid index")
    return value


def _vector(value, width, name):
    _require(type(value) is tuple and len(value) == width, name + " requires an immutable tuple")
    return tuple(_number(item, name) for item in value)


def _array(value, shape, name):
    _require(
        type(value) is np.ndarray
        and value.shape == shape
        and value.dtype == np.dtype(np.float64)
        and np.isfinite(value).all(),
        name + " requires a finite native FP64 array",
    )
    # Producer arrays are byte-backed; a writable owner hidden behind a
    # read-only view is not immutable. No caller data is retained by rankings.
    owner = value
    for _ in range(4):
        if type(owner) is bytes:
            return
        _require(
            type(owner) is np.ndarray and not owner.flags.writeable,
            name + " requires immutable byte-backed storage",
        )
        owner = owner.base
    raise ValueError("native diagnostic: " + name + " has unsupported storage ownership")


def _owners(value, geometry_ids):
    _require(
        type(value) is tuple and 1 <= len(value) <= len(geometry_ids),
        "edge owners require a bounded immutable tuple",
    )
    for item in value:
        _integer(item, "owner geometry", 0, 127)
    _require(
        len(set(value)) == len(value) and set(value) <= geometry_ids,
        "edge owners must identify original query geometry",
    )
    return value


def _validate_query(query):
    _require(type(query) is NativePlanarPushContacts, "a public planar push query is required")
    _require(
        type(query.input_binding) is str
        and len(query.input_binding) == 64
        and all(character in "0123456789abcdef" for character in query.input_binding),
        "query requires its full SHA256 input binding",
    )
    _require(type(query.q) is np.ndarray and query.q.ndim == 1 and 1 <= query.q.size <= 45, "query q shape")
    _array(query.q, query.q.shape, "query q")
    _array(query.plane_axes_world, (3, 3), "plane axes")
    _integer(query.object_body_id, "object body", 1, 127)
    _integer(query.pusher_geom_id, "pusher geometry", 0, 127)
    _vector(query.object_com_world, 3, "object COM")
    _vector(query.plane_origin_world, 3, "plane origin")
    _require(_number(query.pusher_radius_m, "pusher radius") > 0, "pusher radius must be positive")
    _require(_number(query.query_tolerance_m, "query tolerance") == 1e-9, "query tolerance schema")
    _require(type(query.rectangles) is tuple and 1 <= len(query.rectangles) <= 2, "query BOX tuple")
    geometry_ids = set()
    for rectangle in query.rectangles:
        _require(type(rectangle) is PlanarPushRectangle, "query rectangle type")
        geom = _integer(rectangle.geometry_id, "rectangle geometry", 0, 127)
        _require(geom not in geometry_ids and geom != query.pusher_geom_id, "query geometry identity")
        geometry_ids.add(geom)
        _require(type(rectangle.geometry_name) is str and rectangle.geometry_name, "rectangle name")
        lower = _vector(rectangle.lower_xy, 2, "rectangle lower")
        upper = _vector(rectangle.upper_xy, 2, "rectangle upper")
        z = _vector(rectangle.z_interval_world, 2, "rectangle Z interval")
        _require(all(lo < hi for lo, hi in zip(lower, upper, strict=True)) and z[0] < z[1], "rectangle bounds")
    _require(type(query.boundary) is tuple and 1 <= len(query.boundary) <= 16, "query boundary tuple")
    edges = {}
    for edge in query.boundary:
        _require(type(edge) is PlanarPushBoundaryEdge, "query edge type")
        index = _integer(edge.index, "edge", 0, len(query.boundary) - 1)
        _require(index not in edges, "duplicate edge index")
        edges[index] = edge
        _vector(edge.start_world, 3, "edge start")
        _vector(edge.end_world, 3, "edge end")
        _vector(edge.outward_normal_world, 3, "edge normal")
        _owners(edge.owner_geometry_ids, geometry_ids)
    _require(
        type(query.candidates) is tuple and len(query.candidates) <= 128,
        "query candidates require a bounded immutable tuple",
    )
    indices, scoring = set(), []
    for candidate in query.candidates:
        _require(type(candidate) is NativePlanarPushContactCandidate, "query candidate type")
        index = _integer(candidate.index, "candidate", 0, len(query.candidates) - 1)
        _require(index not in indices, "duplicate candidate index")
        indices.add(index)
        edge_index = _integer(candidate.edge_index, "candidate edge", 0, len(edges) - 1)
        owners = _owners(candidate.owner_geometry_ids, geometry_ids)
        _require(owners == edges[edge_index].owner_geometry_ids, "candidate edge owners differ")
        _require(
            0 < _number(candidate.edge_fraction, "edge fraction") < 1
            and abs(_number(candidate.approach_angle_rad, "approach angle")) < pi / 2,
            "candidate edge parameter",
        )
        for name in (
            "point_world", "outward_normal_world", "contact_center_world",
            "precontact_center_world", "approach_direction_world",
        ):
            _vector(getattr(candidate, name), 3, name)
        wrench = _vector(candidate.normal_wrench_com_world, 6, "normal wrench")
        for name in ("contact_distances_m", "entry_distances_m"):
            distances = _vector(getattr(candidate, name), len(query.rectangles), name)
            _require(all(distance >= 0 for distance in distances), "negative analytic distance")
        _require(type(candidate.usable) is bool, "candidate usable must be bool")
        _require(
            type(candidate.reason) is str and candidate.reason
            and (candidate.reason == "clear") == candidate.usable,
            "candidate reason and usability differ",
        )
        scoring.append((index, candidate.usable, wrench[0], wrench[1], wrench[5]))
    return tuple(scoring)


def rank_planar_push_candidates(query, desired_translation_world, desired_yaw_change):
    """Rank every recorded candidate using a fixed lexicographic heuristic.

    Translation is an explicit finite 2- or 3-vector; a third component must be
    exactly zero. Its XY direction is normalized, while zero translation gives
    every candidate alignment zero. The force used for alignment is the recorded
    nominal force, not the possibly angled approach direction.

    Keys are (usable-first, -alignment, yaw-key, original-index). For nonzero yaw,
    yaw-key is -sign(desired_yaw_change)*nominal_Mz; for zero yaw it is abs(Mz).
    nominal_yaw_moment always reports the original signed Mz. A zero translation
    and zero yaw therefore sorts usable candidates by minimum absolute moment,
    then index. Blocked candidates retain scores and are never discarded.

    The public immutable query structure, binding format, finite shapes and
    contiguous unique indices are checked without re-deriving geometry or
    authenticating a binding against unavailable native model inputs. Returned
    frozen records contain only owned Python scalars and immutable sort keys.
    """
    translation = tuple(
        _number(value, "desired translation")
        for value in _items(desired_translation_world, "desired translation", 2, 3)
    )
    _require(len(translation) == 2 or translation[2] == 0, "desired translation must be planar")
    yaw = _number(desired_yaw_change, "desired yaw")
    scoring = _validate_query(query)
    # Normalize after scaling so finite large/subnormal translations keep their
    # direction without overflow in a squared norm or underflow to zero.
    scale = max(abs(translation[0]), abs(translation[1]))
    if scale == 0:
        ux, uy = 0.0, 0.0
    else:
        x, y = translation[0] / scale, translation[1] / scale
        norm = hypot(x, y)
        ux, uy = x / norm, y / norm
    records = []
    for index, usable, fx, fy, moment in scoring:
        alignment = ux * fx + uy * fy
        _require(isfinite(alignment), "translation alignment overflow")
        yaw_key = abs(moment) if yaw == 0 else -moment if yaw > 0 else moment
        key = (0 if usable else 1, -alignment, yaw_key, index)
        records.append(NativePlanarPushRanking(index, usable, alignment, moment, key))
    return tuple(sorted(records, key=lambda record: record.sort_key))
