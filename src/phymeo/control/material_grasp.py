"""Track a material edge through finite parallel jaws using actual endpoint observations."""

from dataclasses import dataclass

import numpy as np


def _points(values):
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("expected finite world points of shape (n, 3)")
    return points


def _edge_indices(edge_nodes, positions):
    ids = np.asarray(edge_nodes)
    if (
        ids.ndim != 1
        or len(ids) < 2
        or ids.dtype.kind not in "iu"
        or ids.min() < 0
        or ids.max() >= len(positions)
    ):
        raise ValueError("an ordered material edge needs two or more valid node indices")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("repeated material edge node")
    return ids


@dataclass(frozen=True)
class MaterialEdgeTarget:
    """A fixed point on a material segment. Evaluation also interpolates velocities."""

    nodes: tuple[int, int]
    fraction: float

    def __post_init__(self):
        if (
            len(self.nodes) != 2
            or any(
                not isinstance(i, (int, np.integer)) or isinstance(i, bool) or i < 0
                for i in self.nodes
            )
            or self.nodes[0] == self.nodes[1]
        ):
            raise ValueError("a material target needs two distinct nonnegative node indices")
        if not 0 <= self.fraction <= 1:
            raise ValueError("material fraction must be between zero and one")
        object.__setattr__(self, "nodes", tuple(map(int, self.nodes)))

    def evaluate(self, values):
        a, b = _points(values)[list(self.nodes)]
        return a + self.fraction * (b - a)


@dataclass(frozen=True)
class MaterialEdgeStandoff:
    """A boundary address and a fixed exterior approach direction in the tool motion plane."""

    target: MaterialEdgeTarget
    direction: tuple[float, float, float]

    @classmethod
    def from_window(cls, positions, edge_nodes, center, plane_normal):
        x = _points(positions)
        ids = _edge_indices(edge_nodes, x)
        center, normal = _points([center, plane_normal])
        normal_length = np.linalg.norm(normal)
        if normal_length == 0.0:
            raise ValueError("the tool motion plane needs a nonzero normal")
        normal = normal / normal_length
        in_plane = np.eye(3) - np.outer(normal, normal)
        distance = float("inf")
        best = None
        for first, second in zip(ids[:-1], ids[1:], strict=True):
            tangent = in_plane @ (x[second] - x[first])
            delta = in_plane @ (center - x[first])
            length2 = float(tangent @ tangent)
            station = float(delta @ tangent) / length2 if length2 else 0.0
            fraction = float(np.clip(station, 0.0, 1.0))
            residual = delta - fraction * tangent
            if length2:
                perpendicular = np.cross(normal, tangent) / np.sqrt(length2)
                outward = (delta @ perpendicular) * perpendicular
            else:
                outward = delta
            length = np.linalg.norm(outward)
            score = float(residual @ residual)
            if length <= 128 * np.finfo(float).eps * max(1.0, np.linalg.norm(delta)):
                continue
            if score < distance:
                best = cls(MaterialEdgeTarget((first, second), fraction), tuple(outward / length))
                distance = score
        return best

    def point(self, positions, reserve):
        """Keep the pad envelope beyond the support plane of every current cloth vertex."""
        if not np.isfinite(reserve) or reserve < 0:
            raise ValueError("exterior standoff reserve must be finite and nonnegative")
        x = _points(positions)
        p = self.target.evaluate(x)
        direction = np.asarray(self.direction)
        return p + direction * (np.max((x - p) @ direction) + reserve)


class ParallelJawWindow:
    """Observation from two ordered convex planar finger fronts, with the first wound inward."""

    def __init__(self, first, second):
        faces = [_points(face) for face in (first, second)]
        centers = [face.mean(0) for face in faces]
        self.center = (centers[0] + centers[1]) / 2
        planes = []
        margins = []
        normals = []
        for i, face in enumerate(faces):
            if len(face) < 3:
                raise ValueError("a finger front requires a polygon")
            edges = np.roll(face, -1, axis=0) - face
            lengths = np.linalg.norm(edges, axis=1)
            raw = np.cross(face[1] - face[0], face[-1] - face[0])
            length = np.linalg.norm(raw)
            if np.any(lengths == 0) or length == 0:
                raise ValueError("degenerate finger front")
            normal = raw / length
            size = lengths.max()
            tolerance = 128 * np.finfo(float).eps * max(1.0, size, np.max(abs(face)))
            if np.max(abs((face - face[0]) @ normal)) > tolerance:
                raise ValueError("finger front is not planar")
            sides = np.cross(normal, edges) / lengths[:, None]
            if np.any(
                np.einsum("ijk,ik->ij", face[None, :, :] - face[:, None, :], sides) < -tolerance
            ):
                raise ValueError("finger front must be ordered and convex")
            normals.append(normal)
            planes.extend(zip(sides, face, strict=True))
            margins.extend(np.einsum("ij,ij->i", self.center - face, sides))
        self.outer_radius = float(
            max(np.linalg.norm(face - self.center, axis=1).max() for face in faces)
        )
        self.axis = normals[0]
        if np.linalg.norm(np.cross(self.axis, normals[1])) > 1e-10:
            raise ValueError("parallel finger fronts required")
        planes.extend(((self.axis, centers[0]), (-self.axis, centers[1])))
        self._normals = np.array([normal for normal, _ in planes])
        self._offsets = np.array([point for _, point in planes])
        self.radius = float(min(margins))
        if self.radius <= 0:
            raise ValueError("finger fronts lack a common interior around their midpoint")

    def acquire(self, positions, edge_nodes, *, clearance=0.0):
        """Choose the nearest visible point on the supplied material polyline inside both faces."""
        positions = _points(positions)
        ids = _edge_indices(edge_nodes, positions)
        if not np.isfinite(clearance) or not 0 <= clearance < self.radius:
            raise ValueError("clearance must leave an interior on both finger fronts")
        margins = np.full(len(self._normals), clearance)
        margins[-2:] = 0.0
        return self._nearest(positions, ids, self._normals, self._offsets, margins)

    def contains(self, point, *, clearance=0.0):
        """Test one fixed material address against both finite front polygons and the opening."""
        p = _points([point])[0]
        if not np.isfinite(clearance) or not 0 <= clearance < self.radius:
            raise ValueError("clearance must leave an interior on both finger fronts")
        margins = np.full(len(self._normals), clearance)
        margins[-2:] = 0.0
        return bool(np.all(np.einsum("ij,ij->i", p - self._offsets, self._normals) >= margins))

    def approach_target(self, positions, edge_nodes, *, preferred=None, axial_clearance=0.0):
        """Find an edge address between the open jaws. It can still be outside both finite fronts."""
        positions = _points(positions)
        ids = _edge_indices(edge_nodes, positions)
        if not np.isfinite(axial_clearance) or axial_clearance < 0:
            raise ValueError("axial clearance must be finite and nonnegative")
        normals, offsets = self._normals[-2:], self._offsets[-2:]
        if preferred is not None:
            if not isinstance(preferred, MaterialEdgeTarget) or not any(
                (i, j) in (preferred.nodes, preferred.nodes[::-1])
                for i, j in zip(ids[:-1], ids[1:], strict=True)
            ):
                raise ValueError("preferred address must lie on the supplied material edge")
            p = preferred.evaluate(positions)
            if np.all(np.einsum("ij,ij->i", p - offsets, normals) >= axial_clearance):
                return preferred
        return self._nearest(positions, ids, normals, offsets, axial_clearance)

    def _nearest(self, positions, ids, normals, offsets, margins):
        best = None
        distance = float("inf")
        for first, second in zip(ids[:-1], ids[1:], strict=True):
            a, b = positions[[first, second]]
            ab = b - a
            length2 = float(ab @ ab)
            if length2 == 0:
                continue
            start = np.einsum("ij,ij->i", a - offsets, normals) - margins
            end = np.einsum("ij,ij->i", b - offsets, normals) - margins
            if np.any((start < 0) & (end < 0)):
                continue
            enter = start < 0
            leave = end < 0
            lo = max(0.0, np.max(start[enter] / (start[enter] - end[enter]), initial=0.0))
            hi = min(1.0, np.min(start[leave] / (start[leave] - end[leave]), initial=1.0))
            if lo > hi:
                continue
            fraction = float(np.clip((self.center - a) @ ab / length2, lo, hi))
            delta = a + fraction * ab - self.center
            score = float(delta @ delta)
            if score < distance:
                distance = score
                best = MaterialEdgeTarget((int(first), int(second)), fraction)
        return best

    def limit_transport(self, material_point, nominal_center, *, clearance):
        """Limit tangential lead so the observed point fits in the common finger footprint."""
        p, goal = _points([material_point, nominal_center])
        clearance = float(clearance)
        if not np.isfinite(clearance) or not 0 <= clearance < self.radius:
            raise ValueError("clearance must leave an interior on both finger fronts")
        delta = goal - p
        normal = (delta @ self.axis) * self.axis
        tangent = delta - normal
        length = np.linalg.norm(tangent)
        limit = self.radius - clearance
        if length <= limit:
            return goal.copy()
        return p + normal + tangent * (limit / length)
