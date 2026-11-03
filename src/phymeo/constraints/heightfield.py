"""A finite regular triangle height graph with a fixed rigid world pose."""

from dataclasses import KW_ONLY, dataclass, field

import numpy as np


def _real_array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"heightfield {name} must contain real numbers")
    array = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"heightfield {name} must be finite in FP64")
    return array


@dataclass(frozen=True)
class Heightfield:
    """Rows index local +Y and columns index local +X, spaced by (dx, dy).

    The local (0, 0, 0) is placed at `origin`. Rotation maps local column vectors
    to world coordinates. Each cell uses the diagonal (row, col) to (row+1, col+1).
    Both sides of the finite triangles participate, including their outer edges.
    """

    heights_yx: tuple[tuple[float, ...], ...]
    spacing_xy: tuple[float, float]
    _: KW_ONLY
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[tuple[float, ...], ...] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    _vertices: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        # CollisionShape also refers to Heightfield for registration validation.
        from .collision_shapes import MAX_VERTICES

        raw_heights = np.asarray(self.heights_yx)
        if raw_heights.ndim != 2 or min(raw_heights.shape) < 2 or raw_heights.size > MAX_VERTICES:
            raise ValueError("heightfield needs a grid of at least 2x2 within vertex capacity")
        heights = _real_array(raw_heights, "heights")
        spacing = _real_array(self.spacing_xy, "spacing")
        origin = _real_array(self.origin, "origin")
        rotation = _real_array(self.rotation, "rotation")
        if spacing.shape != (2,) or np.any(spacing <= 0) or origin.shape != (3,):
            raise ValueError("heightfield needs positive XY spacing and a 3D origin")
        if rotation.shape != (3, 3) or np.max(np.abs(rotation)) > 1 + 1e-12:
            raise ValueError("heightfield rotation must be a proper orthonormal matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-12) or (
            np.linalg.det(rotation) <= 0
        ):
            raise ValueError("heightfield rotation must be a proper orthonormal matrix")

        rows, columns = heights.shape
        try:
            with np.errstate(over="raise", invalid="raise"):
                x, y = np.meshgrid(np.arange(columns) * spacing[0], np.arange(rows) * spacing[1])
                world = np.stack((x, y, heights), axis=-1) @ rotation.T + origin
                # Guard an otherwise valid local grid collapsing after FP64 world placement.
                diagonal = world[1:, 1:] - world[:-1, :-1]
                for side in (world[:-1, 1:] - world[:-1, :-1], world[1:, :-1] - world[:-1, :-1]):
                    scale = np.maximum(abs(diagonal).max(axis=-1), abs(side).max(axis=-1))
                    if np.any(scale == 0):
                        raise ValueError("degenerate heightfield triangle in world coordinates")
                    cross = np.cross(diagonal / scale[..., None], side / scale[..., None])
                    if np.any(np.all(cross == 0, axis=-1)):
                        raise ValueError("degenerate heightfield triangle in world coordinates")
        except FloatingPointError as exc:
            raise ValueError("heightfield world coordinates are not representable") from exc

        object.__setattr__(self, "heights_yx", tuple(tuple(map(float, row)) for row in heights))
        object.__setattr__(self, "spacing_xy", tuple(map(float, spacing)))
        object.__setattr__(self, "origin", tuple(map(float, origin)))
        object.__setattr__(self, "rotation", tuple(tuple(map(float, row)) for row in rotation))
        immutable = np.frombuffer(world.tobytes(order="C"), dtype=np.float64).reshape(-1, 3)
        object.__setattr__(self, "_vertices", immutable)

    @property
    def vertex_count(self):
        return len(self._vertices)

    def world_vertices(self):
        """Return read-only FP64 positions in row-major material order for CollisionMotion."""
        return self._vertices.view()
