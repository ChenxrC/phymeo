"""Owned STL triangles and convex collision data in the authored geometry frame."""

import struct
from dataclasses import dataclass

import numpy as np

from ..geometry.visual import VisualMesh
from ._native_stl import binary_stl


@dataclass(frozen=True)
class ConvexSTL:
    visual: VisualMesh
    polygons: tuple
    normals: tuple


def stl_visual(name, content, scale):
    """Decode owned display triangles while preserving scale and winding."""
    data = binary_stl(content)
    if len(data) < 84:
        raise ValueError(f"{name}: truncated STL")
    count = struct.unpack_from("<I", data, 80)[0]
    if not 0 < count <= np.iinfo(np.int32).max // 3 or len(data) != 84 + 50 * count:
        raise ValueError(f"{name}: invalid STL triangle count")
    record = np.dtype([("normal", "<f4", 3), ("points", "<f4", (3, 3)), ("attribute", "<u2")])
    points = np.frombuffer(data, dtype=record, offset=84)["points"].reshape(-1, 3)
    scale = np.asarray(scale, dtype=np.float64)
    if scale.shape != (3,) or not np.isfinite(scale).all() or not np.all(scale != 0):
        raise ValueError(f"{name}: STL scale must contain three nonzero finite values")
    vertices, inverse = np.unique((points * scale).astype(np.float32), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    if np.count_nonzero(scale < 0) % 2:
        faces = faces[:, (0, 2, 1)]
    return VisualMesh(name, vertices, faces)


def compile_stl(name, content, scale):
    """Keep the original triangles and compile the support surface for PhySolve."""
    from scipy.spatial import ConvexHull, QhullError

    visual = stl_visual(name, content, scale)
    try:
        hull = ConvexHull(visual.vertices)
    except QhullError as exc:
        raise ValueError(f"{name}: STL collision mesh needs a three-dimensional convex hull") from exc

    # Qhull triangulates its boundary. Reassemble coplanar source facets to
    # avoid introducing artificial contact edges inside flat faces.
    groups = {}
    for triangle, plane in zip(hull.simplices, hull.equations, strict=True):
        key = tuple(np.round(plane, 12))
        if key not in groups:
            groups[key] = (plane[:3], set())
        groups[key][1].update(map(int, triangle))
    polygons, normals = [], []
    for normal, indices in groups.values():
        ids = np.asarray(sorted(indices))
        points = visual.vertices[ids]
        relative = points - points.mean(axis=0)
        direction = relative[0] / np.linalg.norm(relative[0])
        tangent = np.cross(normal, direction)
        angles = np.arctan2(relative @ tangent, relative @ direction)
        polygons.append(tuple(map(int, ids[np.argsort(angles)])))
        normals.append(tuple(normal))
    return ConvexSTL(visual, tuple(polygons), tuple(normals))
