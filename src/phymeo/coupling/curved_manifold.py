"""Select one material endpoint of a published static curved/plane support segment."""

from dataclasses import replace
from math import fsum, isfinite
from numbers import Integral

import numpy as np


def select_static_support(pair, witness, index):
    """Keep the representative, or select one of two static cylinder/plane endpoints."""
    if index is None:
        return witness
    if (
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, Integral)
        or index not in (0, 1)
    ):
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
