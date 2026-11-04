"""Address the material points of a published static rigid support."""

from dataclasses import replace
from math import fsum, isfinite
from numbers import Integral

import numpy as np

from ..constraints.collision_shapes import MAX_CONVEX_VERTICES


def hull_polygon(pair):
    """The unique hull face or half-space support, including the live proximity query."""
    if pair.sweep is not None or pair.convex is None:
        return None
    patches = pair.convex.end.patches
    if not patches:
        close = getattr(pair.witness, "proximity_patch", None)
        patches = () if close is None else (close,)
    kinds = tuple(s.kind for s in pair.witness_shapes)
    halfspace = set(kinds) == {"plane", "hull"}
    if not patches or (not halfspace and not any(p.dimension == 2 for p in patches)):
        return None
    if len(patches) != 1 or pair.witness is None or pair.witness.normal is None:
        raise ValueError("contact polygon requires a single resolved support normal")
    patch = patches[0]
    if (
        tuple(s.key for s in pair.witness_shapes) != (pair.first, pair.second)
        or (not halfspace and kinds != ("hull", "hull"))
        or not 1 <= len(patch.points) <= 2 * MAX_CONVEX_VERTICES
        or patch.dimension != min(2, len(patch.points) - 1)
        or not isfinite(patch.area_m2)
        or (patch.area_m2 <= 0 if patch.dimension == 2 else patch.area_m2 != 0)
        or patch.normal != pair.witness.normal
    ):
        raise ValueError("invalid registered convex contact polygon")
    return patch


def select_static_support(pair, witness, index):
    """Select an ordered clipped hull corner or an endpoint of a cylinder/plane support."""
    if index is None:
        return witness
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
        raise ValueError("a material support index must be an integer")
    patch = hull_polygon(pair)
    if patch is not None:
        if not 0 <= index < len(patch.points):
            raise ValueError("material index outside convex contact polygon")
        corner = patch.points[index]
        return replace(
            witness,
            first_point=corner.first_point,
            second_point=corner.second_point,
            first_weights=corner.first_weights,
            second_weights=corner.second_weights,
            first_offset=getattr(corner, "first_offset", witness.first_offset),
            second_offset=getattr(corner, "second_offset", witness.second_offset),
        )
    if index not in (0, 1):
        raise ValueError("a support segment endpoint index must be integer 0 or 1")
    trace = pair.curved
    shapes = pair.witness_shapes
    if (
        pair.sweep is not None
        or pair.status not in ("touching", "overlap", "separated")
        or trace is None
        or trace.locus != "segment"
        or trace.first_contact is not None
        or trace.last_contact is not None
        or len(trace.points_m) != 2
        or tuple(shape.key for shape in shapes) != (pair.first, pair.second)
        or {shape.kind for shape in shapes} != {"plane", "cylinder"}
    ):
        raise ValueError("endpoint selection requires a static cylinder/plane support segment")
    owner_first = shapes[0].kind == "cylinder"
    if trace.owner != (pair.first if owner_first else pair.second) or witness is None:
        raise ValueError("support segment has no matching curved owner or witness")
    vectors = (
        *trace.points_m,
        witness.first_point,
        witness.second_point,
        witness.first_offset,
        witness.second_offset,
    )
    if any(v is None or len(v) != 3 or not all(isfinite(x) for x in v) for v in vectors):
        raise ValueError("support segment requires finite surface points and material offsets")
    if tuple(trace.points_m[0]) == tuple(trace.points_m[1]):
        raise ValueError("support segment endpoints must be distinct")
    origin = witness.first_point if owner_first else witness.second_point
    endpoint = tuple(trace.points_m[index])

    def shifted(vector):
        return tuple(
            fsum((v, end, -start)) for v, end, start in zip(vector, endpoint, origin, strict=True)
        )

    return replace(
        witness,
        first_point=endpoint if owner_first else shifted(witness.first_point),
        second_point=shifted(witness.second_point) if owner_first else endpoint,
        first_offset=shifted(witness.first_offset),
        second_offset=shifted(witness.second_offset),
    )
