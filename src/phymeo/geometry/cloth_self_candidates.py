"""Conservative exact swept AABBs, with no collision or CCD certification."""

from fractions import Fraction

import numpy as np

from ..constraints.cloth_self_contact import ClothSelfContactTopology


def _positions(value, name):
    result = np.asarray(value)
    if result.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must use float64 storage")
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.array(result, copy=True, order="C")


def _swept_bounds(faces, thickness, endpoints):
    result = []
    for face, rho in zip(faces, thickness, strict=True):
        radius = Fraction.from_float(float(rho))
        bounds = []
        for axis in range(3):
            coordinates = tuple(
                Fraction.from_float(float(positions[index, axis]))
                for positions in endpoints
                for index in face
            )
            bounds.append((min(coordinates) - radius, max(coordinates) + radius))
        result.append(tuple(bounds))
    return tuple(result)


def candidate_self_pairs(topology, start_positions, end_positions=None):
    """Return sorted canonical face pairs with inclusively overlapping bounds.

    Nodes follow affine paths between supplied endpoints, or remain static when
    end_positions is omitted. Every nodal component lies between its endpoint
    values; every component of the unit-normal offset lies within [-rho, rho].
    Thus the exact endpoint union expanded by rho encloses the entire rotating
    prism, wherever its normal exists. Fraction.from_float preserves the exact
    supplied binary values without epsilon expansion or floating-point overflow.

    This O(f**2) reference excludes any pair sharing a vertex. An output pair
    is only a broad-phase candidate. No output certifies touching, separation
    throughout a valid step, nondegeneracy, safe CCD, or a contact response.
    """
    if not isinstance(topology, ClothSelfContactTopology):
        raise TypeError("topology must be ClothSelfContactTopology")
    start = _positions(start_positions, "start_positions")
    end = None if end_positions is None else _positions(end_positions, "end_positions")
    if end is not None and end.shape != start.shape:
        raise ValueError("start_positions and end_positions must have the same shape")
    faces = topology.triangles
    if np.any(faces >= len(start)):
        raise ValueError("triangle index outside supplied positions")
    endpoints = (start,) if end is None else (start, end)
    bounds = _swept_bounds(faces, topology.half_thickness_m, endpoints)
    incidence = tuple(frozenset(int(index) for index in face) for face in faces)
    candidates = []
    for left in range(len(faces)):
        for right in range(left + 1, len(faces)):
            if not incidence[left].isdisjoint(incidence[right]):
                continue
            if all(
                bounds[left][axis][0] <= bounds[right][axis][1]
                and bounds[right][axis][0] <= bounds[left][axis][1]
                for axis in range(3)
            ):
                candidates.append((left, right))
    return tuple(candidates)
