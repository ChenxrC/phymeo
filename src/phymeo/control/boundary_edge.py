"""Material boundary topology and reachable addresses for finite parallel jaws."""

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .material_grasp import MaterialEdgeTarget, ParallelJawWindow


@dataclass(frozen=True)
class BoundaryTrial:
    edge: tuple[int, int]
    fraction: float
    distance_m: float
    obstruction: tuple[str, str] | None


@dataclass(frozen=True)
class BoundarySelection:
    target: MaterialEdgeTarget | None
    axial_candidates: int
    trials: tuple[BoundaryTrial, ...]


def _feasible_address(x, edge, window, clearance):
    address = window.approach_target(x, edge, axial_clearance=clearance)
    if address is None:
        return None

    def admitted(candidate):
        return (
            window.approach_target(x, edge, preferred=candidate, axial_clearance=clearance)
            is candidate
        )

    if admitted(address):
        return address
    # Interpolating a clipped boundary fraction can round to the outside. Move toward the
    # axial middle and recheck the original physical inequality at the actual material point.
    a, b = x[list(edge)]
    axial_span = (b - a) @ window.axis
    if not axial_span:
        return None
    interior = float(np.clip((window.center - a) @ window.axis / axial_span, 0.0, 1.0))
    adjacent = MaterialEdgeTarget(edge, float(np.nextafter(address.fraction, interior)))
    if admitted(adjacent):
        return adjacent
    valid = MaterialEdgeTarget(edge, interior)
    if not admitted(valid):
        return None
    outside = address.fraction
    while True:
        fraction = 0.5 * outside + 0.5 * valid.fraction
        if fraction == outside or fraction == valid.fraction:
            return valid
        probe = MaterialEdgeTarget(edge, fraction)
        if admitted(probe):
            valid = probe
        else:
            outside = fraction


class BoundaryEdges:
    """Manifold topological boundary, independent of face order and winding.

    Each edge contributes the closest material address within the measured axial band.
    Closer addresses are tried first. Numerically tied locations use stable local node IDs.
    The caller certifies the supplied address against its real tool and all finite faces.
    """

    def __init__(self, triangles):
        ids = np.asarray(triangles)
        if (
            ids.ndim != 2
            or ids.shape[1] != 3
            or not len(ids)
            or ids.dtype.kind not in "iu"
            or np.any(ids < 0)
        ):
            raise ValueError("expected triangles with nonnegative material IDs")
        faces = [tuple(sorted(map(int, row))) for row in ids]
        if len(set(faces)) != len(faces) or any(len(set(row)) != 3 for row in faces):
            raise ValueError("duplicate or degenerate material face")
        count = Counter(edge for row in faces for edge in combinations(row, 2))
        self.edges = tuple(sorted(edge for edge, uses in count.items() if uses == 1))
        degree = Counter(node for edge in self.edges for node in edge)
        if not self.edges or max(count.values()) > 2 or any(n != 2 for n in degree.values()):
            raise ValueError("expected manifold sheets with an open material boundary")
        self._largest = int(ids.max())

    def choose(self, positions, window, *, axial_clearance, obstruction):
        x = np.asarray(positions, dtype=float)
        if x.ndim != 2 or x.shape[1] != 3 or len(x) <= self._largest or not np.isfinite(x).all():
            raise ValueError("missing or invalid current material position")
        if not isinstance(window, ParallelJawWindow) or not callable(obstruction):
            raise TypeError("a measured parallel-jaw window and a structural checker are required")
        candidates = []
        for edge in self.edges:
            address = _feasible_address(x, edge, window, axial_clearance)
            if address is not None:
                distance = float(np.linalg.norm(address.evaluate(x) - window.center))
                candidates.append((distance, address))
        candidates.sort(key=lambda row: row[0])
        scale = max(1.0, float(np.linalg.norm(x - window.center, axis=1).max()))
        tolerance = 64 * np.finfo(float).eps * scale
        trials = []
        i = 0
        while i < len(candidates):
            stop = i + 1
            while stop < len(candidates) and candidates[stop][0] - candidates[i][0] <= tolerance:
                stop += 1
            for distance, address in sorted(candidates[i:stop], key=lambda row: row[1].nodes):
                blocked = obstruction(address)
                if blocked is not None and not (
                    isinstance(blocked, tuple)
                    and len(blocked) == 2
                    and all(isinstance(n, str) for n in blocked)
                ):
                    raise TypeError(
                        "structural checker must return None or a named obstruction pair"
                    )
                trials.append(BoundaryTrial(address.nodes, address.fraction, distance, blocked))
                if blocked is None:
                    return BoundarySelection(address, len(candidates), tuple(trials))
            i = stop
        return BoundarySelection(None, len(candidates), tuple(trials))
