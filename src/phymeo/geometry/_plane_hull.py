"""Static, one-sided infinite plane versus an unrounded, full-dimensional convex hull."""

from dataclasses import dataclass
from fractions import Fraction as F
from math import isfinite

from ..core.capabilities import SolverFailure
from ._convex_hull import MaterialChart, plane_axes
from ._round_sweep import _unit, _vector, _xyz
from ._triangle_pair_sweep import _cross3
from .collision_witness import (
    CollisionWitness,
    ConvexContactPatch,
    ConvexContactPoint,
    ConvexContactPose,
    ConvexHullTrace,
    MaterialContactPoint,
    ProximityWitness,
)

_MAX_MATERIAL_ERROR_M = F(1, 10**7)


@dataclass(frozen=True)
class PlaneHullPoint(ConvexContactPoint):
    """Each plane foot is addressed by the registered anchor plus a world tangent offset."""

    first_offset: tuple[float, ...] | None
    second_offset: tuple[float, ...] | None


@dataclass(frozen=True)
class _PlaneHullIntersection(MaterialContactPoint):
    first_offset: tuple[float, ...] | None = None
    second_offset: tuple[float, ...] | None = None


class StaticPlaneHull:
    features = ()

    def __init__(self, hull, plane, plane_motion, plane_first, proximity):
        self.hull = hull
        self.plane = plane
        self.anchor = _vector(plane_motion.start[0])
        self.plane_first = plane_first
        self.proximity = proximity

    def result(self):
        n = _vector(self.plane.normal)
        unit, _ = _unit(n)
        direction = _xyz(unit if self.plane_first else -unit)
        xyz = self.hull.points
        heights = {i: n @ (xyz[i] - self.anchor) for i in self.hull.vertices}
        support = min(heights.values())
        extreme = [i for i, height in heights.items() if height == support]
        i, j = plane_axes(n)
        chart = MaterialChart({(xyz[k][i], xyz[k][j]): k for k in extreme})
        nodes = tuple(chart.ids[p] for p in chart.ring)
        dimension = min(2, len(nodes) - 1)
        vector_area = _vector((0, 0, 0))
        base = xyz[nodes[0]]
        for a, b in zip(nodes[1:-1], nodes[2:], strict=True):
            vector_area += _cross3(xyz[a] - base, xyz[b] - base)
        if (vector_area @ n) * (1 if self.plane_first else -1) < 0:
            nodes = nodes[:1] + nodes[:0:-1]
        area = _unit(vector_area)[1] / 2
        separation = (support / (n @ n)) * n
        length = _unit(separation)[1]
        gap = length if support >= 0 else -length
        if (
            not isfinite(gap)
            or not isfinite(area)
            or (support and not gap)
            or (dimension == 2 and not area)
        ):
            raise SolverFailure("plane_hull_support_resolution")
        points = []
        plane_id = self.plane.vertex_ids[0]
        for node in nodes:
            point = xyz[node]
            foot = point - separation
            offset = _xyz(foot - self.anchor)
            world = _xyz(foot)
            if (
                sum(abs(x) for x in _vector(world) - foot) > _MAX_MATERIAL_ERROR_M
                or sum(abs(x) for x in self.anchor + _vector(offset) - foot) > _MAX_MATERIAL_ERROR_M
            ):
                raise SolverFailure("plane_hull_material_address_resolution")
            pw, hw = ((plane_id, 1.0),), ((node, 1.0),)
            if self.plane_first:
                corner = PlaneHullPoint(world, _xyz(point), pw, hw, offset, None)
            else:
                corner = PlaneHullPoint(_xyz(point), world, hw, pw, None, offset)
            points.append(corner)
        hull_ids, plane_ids = tuple(sorted(nodes)), (plane_id,)
        a, b = (plane_ids, hull_ids) if self.plane_first else (hull_ids, plane_ids)
        patch = ConvexContactPatch(direction, a, b, dimension, area, tuple(points))
        first = points[0]
        intersections = (
            tuple(
                _PlaneHullIntersection(
                    p.first_point,
                    p.first_weights,
                    p.second_weights,
                    first_offset=p.first_offset,
                    second_offset=p.second_offset,
                )
                for p in points
            )
            if not support
            else ()
        )
        witness = CollisionWitness(
            gap,
            direction,
            first.first_point,
            first.second_point,
            first.first_weights,
            first.second_weights,
            "host-plane-hull",
            first_offset=first.first_offset,
            second_offset=first.second_offset,
            intersection_points=intersections,
        )
        status = "separated" if support > 0 else "overlap" if support < 0 else "touching"
        if self.proximity and support > 0:
            witness = ProximityWitness(**vars(witness), proximity_patch=patch)
        pose = ConvexContactPose(status, witness, (patch,))
        trace = ConvexHullTrace(pose, pose, None, None, None, None, 1)
        return status, "plane_hull_support", witness, trace
