"""Per-workspace reuse of a successfully certified translating closed mesh."""

from dataclasses import dataclass
from struct import pack
from types import MappingProxyType

import numpy as np

from ._closed_mesh import ClosedMesh


def _motion_key(path):
    # CollisionMotion owns contiguous FP64 (vertex, 3) arrays. Retain their bytes
    # independently of caller object identity and distinguish every motion form.
    turn = path.turn
    motion = (
        None
        if turn is None
        else pack("<10d", *turn.pivot_start, *turn.pivot_end, *turn.axis, turn.angle)
    )
    return (
        path.start.shape,
        path.start.tobytes(),
        None if path.end is None else path.end.tobytes(),
        motion,
    )


@dataclass(frozen=True, eq=False)
class _PreparedClosedMesh(ClosedMesh):
    """The owned points and facet charts are read-only for all narrow-phase consumers."""

    points: MappingProxyType
    step: np.ndarray
    facets: tuple

    @classmethod
    def seal(cls, validated):
        for point in validated.points.values():
            point.setflags(write=False)
        validated.step.setflags(write=False)
        for facet in validated.facets:
            facet.points = MappingProxyType(facet.points)
            # stationarity and material addressing share these same arrays with the facet.
            for values in (facet.xyz, facet.lo, facet.hi, facet.face.normal):
                values.setflags(write=False)
            facet.face.chart.ids = MappingProxyType(facet.face.chart.ids)
            facet.face.chart.ring = tuple(facet.face.chart.ring)
        return cls(MappingProxyType(validated.points), validated.step, validated.facets)


class ValidatedMeshCache:
    """One successful preparation per bound shape, used under the owning query lock."""

    def __init__(self):
        self._entries = {}
        self._attempts = self._builds = self._hits = 0

    def get(self, shape, path, builder):
        motion = _motion_key(path)
        entry = self._entries.get(shape.key)
        if entry is not None and entry[0] is shape and entry[1] == motion:
            self._hits += 1
            return entry[2]
        # An invalid replacement must be checked on every subsequent attempt.
        self._entries.pop(shape.key, None)
        self._attempts += 1
        prepared = _PreparedClosedMesh.seal(builder(shape, path))
        self._entries[shape.key] = (shape, motion, prepared)
        self._builds += 1
        return prepared

    def stats(self):
        return dict(
            validated_mesh_attempts=self._attempts,
            validated_mesh_builds=self._builds,
            validated_mesh_hits=self._hits,
        )

    def __len__(self):
        return len(self._entries)

    def clear(self):
        self._entries.clear()
