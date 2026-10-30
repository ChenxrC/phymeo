"""Immutable face topology for conservative cloth self-contact candidates."""

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np


def _read_only_copy(value):
    result = value.copy(order="C")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, eq=False, init=False)
class ClothSelfContactTopology:
    """Oriented manifold faces and positive prism half thicknesses in metres.

    S1 excludes every pair sharing any vertex, including pairs without a shared
    edge. This incidence policy does not cover all possible adjacent-face
    crossings. Face winding defines orientation; reversing all connected faces
    is valid. No positions, geometry proof, or collision state are stored here.
    """

    domain: str
    _triangles: np.ndarray = field(repr=False)
    _half_thickness_m: np.ndarray = field(repr=False)
    incidence_policy: ClassVar[str] = "exclude_shared_vertex"

    def __init__(self, domain, triangles, half_thickness_m):
        if not isinstance(domain, str) or not domain or domain.strip() != domain:
            raise ValueError("domain must be a nonempty unpadded string")
        faces = np.asarray(triangles)
        if faces.dtype.kind not in "iu":
            raise TypeError("triangles must use integer indices")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("triangles must have shape (f, 3)")
        if np.any(faces < 0) or np.any(faces > np.iinfo(np.int64).max):
            raise ValueError("triangles must contain non-negative int64 indices")
        faces = np.array(faces, dtype=np.int64, order="C", copy=True)
        rho = np.asarray(half_thickness_m)
        if rho.dtype != np.dtype(np.float64):
            raise TypeError("half_thickness_m must use float64 storage")
        if rho.shape != (len(faces),):
            raise ValueError("half_thickness_m must have shape (f,)")
        if not np.isfinite(rho).all() or np.any(rho <= 0):
            raise ValueError("half_thickness_m must be finite and strictly positive")
        rho = np.array(rho, copy=True, order="C")

        seen_faces, edges = set(), {}
        for face in faces:
            a, b, c = (int(index) for index in face)
            if len({a, b, c}) != 3:
                raise ValueError("each triangle must have three distinct vertices")
            identity = tuple(sorted((a, b, c)))
            if identity in seen_faces:
                raise ValueError("duplicate unoriented triangle")
            seen_faces.add(identity)
            for start, end in ((a, b), (b, c), (c, a)):
                edge = (min(start, end), max(start, end))
                edges.setdefault(edge, []).append((start, end))
        for directions in edges.values():
            if len(directions) > 2:
                raise ValueError("nonmanifold edge has more than two incident faces")
            if len(directions) == 2 and directions[0] != directions[1][::-1]:
                raise ValueError("shared edge orientation must be opposite")

        faces.flags.writeable = False
        rho.flags.writeable = False
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "_triangles", faces)
        object.__setattr__(self, "_half_thickness_m", rho)

    @property
    def triangles(self):
        """Owned read-only int64 face indices; modifying a copy cannot alter topology."""
        return _read_only_copy(self._triangles)

    @property
    def half_thickness_m(self):
        """Owned read-only FP64 half thicknesses in face order."""
        return _read_only_copy(self._half_thickness_m)
