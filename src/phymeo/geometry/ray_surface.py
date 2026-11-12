"""Immutable body-local ray geometry, independent of a native solver model."""

from dataclasses import dataclass

import numpy as np

from ..sensors.body_site import BodySite


@dataclass(frozen=True)
class RaySurface:
    name: str
    body: str
    kind: str
    size: tuple = ()
    position: tuple = (0.0, 0.0, 0.0)
    rotation_xyzw: tuple = (0.0, 0.0, 0.0, 1.0)
    vertices: tuple = ()
    triangles: tuple = ()

    def __post_init__(self):
        pose = BodySite(self.name, self.body, self.position, self.rotation_xyzw)
        object.__setattr__(self, "position", pose.position)
        object.__setattr__(self, "rotation_xyzw", pose.rotation_xyzw)
        widths = dict(plane=0, sphere=1, capsule=2, cylinder=2, ellipsoid=3, box=3, mesh=0)
        if self.kind not in widths:
            raise ValueError(f"unsupported prepared ray surface: {self.kind}")
        size = np.asarray(self.size, np.float64)
        if (
            size.shape != (widths[self.kind],)
            or not np.isfinite(size).all()
            or np.any(size <= 0)
            or np.any(size > np.finfo(np.float32).max)
            or np.any(size.astype(np.float32) <= 0)
        ):
            raise ValueError("ray size must contain positive finite float32 dimensions")
        object.__setattr__(self, "size", tuple(float(x) for x in size))
        if self.kind == "mesh":
            vertices, triangles = np.asarray(self.vertices, np.float64), np.asarray(self.triangles)
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or len(vertices) < 3
                or not np.isfinite(vertices).all()
                or np.any(np.abs(vertices) > np.finfo(np.float32).max)
                or triangles.ndim != 2
                or triangles.shape[1] != 3
                or not len(triangles)
                or triangles.dtype.kind not in "iu"
                or np.any(triangles < 0)
                or np.any(triangles >= len(vertices))
            ):
                raise ValueError("ray mesh requires finite vertices and valid triangular indices")
            represented = vertices.astype(np.float32).astype(np.float64)
            a, b, c = (represented[triangles[:, i]] for i in range(3))
            if np.any(np.linalg.norm(np.cross(b - a, c - a), axis=1) == 0):
                raise ValueError("ray mesh contains a degenerate triangle")
            object.__setattr__(
                self, "vertices", tuple(tuple(float(v) for v in p) for p in vertices)
            )
            object.__setattr__(
                self, "triangles", tuple(tuple(int(v) for v in p) for p in triangles)
            )
        elif self.vertices or self.triangles:
            raise ValueError("only mesh ray surfaces accept vertices/triangles")
