"""Material quadrature for a moving cuboid against a finite oriented support."""

from fractions import Fraction
from itertools import product
from types import SimpleNamespace

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionShape
from ..geometry._convex_contact import HullPairQuery
from ..geometry._convex_hull import HullShape
from ..geometry._round_sweep import _vector
from ._patch_quadrature import patch_quadrature
from .box_link_contact import BoxLinkContacts
from .sphere_link_contact import _RigidLinkContacts

_BITS = tuple(product((0, 1), repeat=3))
_HULL = CollisionShape(CollisionKey(0, 0, 0), "hull", tuple(range(8)))


def _hull(base, edges):
    # Independently rounded world vertices can incorrectly split a coplanar face.
    points = np.array(
        [base + sum(edge for flag, edge in zip(row, edges, strict=True) if flag) for row in _BITS],
        dtype=object,
    )
    return HullShape(_HULL, SimpleNamespace(start=points, end=None, turn=None))


def _blend(chart, coeff):
    result = np.zeros(8)
    for point, amount in zip(chart, coeff, strict=True):
        for node, weight in point:
            result[node] += amount * weight
    return result


class TableLinkSolids(BoxLinkContacts):
    """Round link witnesses and cuboid patches share the same equal-and-opposite force law."""

    def __init__(self, solids):
        _RigidLinkContacts.__init__(self, solids)
        self._table_hull_key = None
        self._table_hull = None

    def _witnesses(self, solid, positions, rotations, center, geometry):
        if solid.kind != "box":
            yield from super()._witnesses(solid, positions, rotations, center, geometry)
            return
        half, R = geometry
        owner = solid.owner_index
        # Fixed-owner vertices from the asset registry are already in world coordinates.
        link_rotation = rotations[owner] if owner >= 0 else np.eye(3)
        link_position = positions[owner] if owner >= 0 else np.zeros(3)
        world = link_position + solid.vertices @ link_rotation.T
        table_extent = np.abs(R) @ half
        tol = (
            32
            * np.finfo(float).eps
            * max(1.0, float(np.max(np.abs(world))), float(np.max(np.abs(center) + table_extent)))
        )
        if np.any(world.min(axis=0) > center + table_extent + tol) or np.any(
            world.max(axis=0) < center - table_extent - tol
        ):
            return
        key = half.tobytes()
        if self._table_hull_key != key:
            exact_half = _vector(half)
            self._table_hull = _hull(-exact_half, np.diag(2 * exact_half))
            self._table_hull_key = key
        # Work in the fixed support frame. A common rotation cancels in this
        # product. Only entries inside its componentwise dot-product error bound
        # may be restored to an exact coordinate direction.
        relative = R.T @ link_rotation @ solid._axes.T
        rounding = (
            16 * np.finfo(float).eps * (np.abs(R.T) @ np.abs(link_rotation) @ np.abs(solid._axes.T))
        )
        relative[np.abs(relative) <= rounding] = 0.0
        aligned = np.abs(np.abs(relative) - 1) <= rounding
        relative[aligned] = np.sign(relative[aligned])
        local_center = _vector(R.T @ ((link_position - center) + link_rotation @ solid._center))
        B = np.array(
            [
                _vector(axis) * Fraction(float(h))
                for axis, h in zip(relative.T, solid._half, strict=True)
            ]
        )
        link_hull = _hull(local_center - B.sum(axis=0), 2 * B)
        query = HullPairQuery(link_hull, self._table_hull)
        if query._window(None) is None:
            return
        pose = query.sample(Fraction(0))
        if pose.relation == "touching":
            return
        if len(pose.patches) != 1:
            raise ValueError(f"{solid.shape}: no unique box/table penetration axis")
        (patch,) = pose.patches
        normal = R @ np.array(patch.normal)
        gap = pose.witness.gap_m
        if gap >= 0:
            return
        table_points = center + ((np.asarray(_BITS) * 2 - 1) * half) @ R.T
        link_chart = [p.first_weights for p in patch.points]
        table_chart = [p.second_weights for p in patch.points]
        for coeff, weight, area in patch_quadrature(patch):
            wa, wb = _blend(link_chart, coeff), _blend(table_chart, coeff)
            pa, pb = wa @ world, wb @ table_points
            if not np.allclose(pb - pa, gap * normal, atol=tol * 8, rtol=0):
                raise ValueError(f"{solid.shape}: box/table material witness cannot be rebound")
            yield (
                gap,
                normal,
                pa,
                pb,
                ("point", "edge", "face")[patch.dimension],
                float(weight),
                dict(
                    patch_dimension=patch.dimension,
                    patch_area_m2=patch.area_m2,
                    area_weight_m2=float(area),
                    link_material_weights=[(int(i), float(wa[i])) for i in np.flatnonzero(wa)],
                    table_material_weights=[(int(i), float(wb[i])) for i in np.flatnonzero(wb)],
                ),
            )
