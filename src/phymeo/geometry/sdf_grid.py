"""Dense regular Cartesian signed-distance values and cell-local trilinear queries in SI units."""

from dataclasses import dataclass
from fractions import Fraction as F
from math import inf, isfinite, nextafter, prod, ulp
from numbers import Integral

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.validation import finite_real
from ..dynamics.rotation import quaternion_matrix
from ._prism_pair_exact import sqrt_bounds

_MAX_NODES = 1 << 22
_MAX_BATCH = 1 << 18


def _vector(value, size, name):
    source = np.asarray(value)
    if source.shape != (size,) or source.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain {size} finite real values")
    return tuple(finite_real(x, name) for x in source)


def _layout(shape_xyz, origin_m, spacing_m, band_m):
    if len(shape_xyz) != 3 or any(
        not isinstance(x, Integral) or isinstance(x, (bool, np.bool_)) or x < 2 for x in shape_xyz
    ):
        raise ValueError("shape_xyz must contain three integer node counts >= 2")
    shape = tuple(map(int, shape_xyz))
    if prod(shape) > _MAX_NODES:
        raise ValueError(f"SDF grid capacity exceeded: {prod(shape)} nodes")
    origin = _vector(origin_m, 3, "origin_m")
    spacing = _vector(spacing_m, 3, "spacing_m")
    for n, o, h in zip(shape, origin, spacing, strict=True):
        if h <= 0:
            raise ValueError("grid spacing must be positive")
        try:
            high = float(F(o) + (n - 1) * F(h))
        except OverflowError as exc:
            raise ValueError("grid extent exceeds FP64") from exc
        if not isfinite(high) or h < max(ulp(o), ulp(high)):
            raise ValueError("grid spacing cannot distinguish adjacent positions in FP64")
    if band_m is not None:
        band_m = finite_real(band_m, "band_m")
        if band_m <= 0:
            raise ValueError("band_m must be positive")
    return shape, origin, spacing, band_m


def _points(points, capacity):
    points = np.asarray(points)
    if points.shape == (0,):
        return np.empty((0, 3), np.float64)
    if points.shape == (3,):
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in "fiu":
        raise ValueError("query points must be a real 3-vector or an (n,3) array")
    if len(points) > capacity:
        raise ValueError(f"SDF query capacity exceeded: {len(points)} > {capacity}")
    points = np.asarray(points, dtype=np.float64)
    if not np.isfinite(points).all():
        raise ValueError("query points must be finite")
    return points


def _world(points, translation_m, orientation_xyzw):
    origin = _vector(translation_m, 3, "translation_m")
    q = np.array(_vector(orientation_xyzw, 4, "orientation_xyzw"))
    if abs(np.hypot.reduce(q) - 1) > 1e-7:
        raise ValueError("orientation_xyzw must be a unit body-to-world quaternion")
    rotation = quaternion_matrix(q.reshape(1, 4))[0]
    with np.errstate(over="ignore", invalid="ignore"):
        local = (points - origin) @ rotation
    if not np.isfinite(local).all():
        raise ValueError("SDF pose transform exceeds FP64")
    return local, rotation


@dataclass(frozen=True)
class SdfSample:
    """Value in metres and dimensionless trilinear derivative in the caller's frame. A valid flat cell has a zero derivative."""

    status: str
    distance_m: float | None
    gradient: tuple[float, float, float] | None
    cell_xyz: tuple[int, int, int] | None


@dataclass(frozen=True, init=False, eq=False)
class SignedDistanceGrid:
    """A dense, immutable Cartesian field with values indexed as [z,y,x].

    Distances supplied in metres use negative inside and positive outside. The
    first node is at origin_m in the asset frame, with spacing_m in XYZ order.
    When band_m is set, cells require all eight node magnitudes strictly below
    it. External values need not satisfy the signed-distance equation.
    """

    values_zyx_m: np.ndarray
    origin_m: tuple[float, float, float]
    spacing_m: tuple[float, float, float]
    shape_xyz: tuple[int, int, int]
    band_m: float | None
    geometry_error_bound_m: float | None
    _bake: tuple | None
    _origin: tuple
    _spacing: tuple

    def __init__(self, values_zyx_m, *, origin_m, spacing_m, band_m=None):
        data = np.asarray(values_zyx_m)
        if data.ndim != 3 or data.dtype.kind not in "fiu":
            raise ValueError("values_zyx_m must be a real three-dimensional node array")
        shape, o, h, band = _layout(data.shape[::-1], origin_m, spacing_m, band_m)
        data = np.asarray(data, dtype=np.float64)
        if not np.isfinite(data).all():
            raise ValueError("grid values must be finite")
        if band is not None:
            data = np.clip(data, -band, band)
        raw = np.frombuffer(data.tobytes(order="C"), np.float64).reshape(data.shape)
        for name, value in (
            ("values_zyx_m", raw),
            ("origin_m", o),
            ("spacing_m", h),
            ("shape_xyz", shape),
            ("band_m", band),
            ("geometry_error_bound_m", None),
            ("_bake", None),
            ("_origin", tuple(map(F, o))),
            ("_spacing", tuple(map(F, h))),
        ):
            object.__setattr__(self, name, value)

    @property
    def bake_stats(self):
        return None if self._bake is None else dict(self._bake)

    def _locate(self, points):
        # Decide cells in the stored binary geometry before optional FP32
        # evaluation. A representable rounded knot owns its positive-side
        # cell; the maximum node owns the final cell with fraction one.
        bases = np.full(len(points), -1, np.int32)
        fractions = np.zeros((len(points), 3), np.float64)
        nx, ny, _ = self.shape_xyz
        for i, p in enumerate(points):
            cell = []
            for k, (x, o, h, n) in enumerate(
                zip(p, self._origin, self._spacing, self.shape_xyz, strict=True)
            ):
                g = (F(float(x)) - o) / h
                nearest = round(g)
                if 0 <= nearest < n and float(o + nearest * h) == x:
                    g = F(nearest)
                if not 0 <= g <= n - 1:
                    break
                j = min(int(g), n - 2)
                cell.append(j)
                fractions[i, k] = float(g - j)
            else:
                bases[i] = cell[0] + nx * (cell[1] + ny * cell[2])
        return bases, fractions

    def _results(self, bases, status, raw, rotation):
        nx, ny, _ = self.shape_xyz
        output = []
        for b, code, row in zip(bases, status, raw, strict=True):
            b = int(b)
            if b < 0:
                if code != 1:
                    raise RuntimeError("invalid SDF outside-grid result")
                output.append(SdfSample("outside_grid", None, None, None))
                continue
            cell = (b % nx, (b // nx) % ny, b // (nx * ny))
            if code == 2:
                output.append(SdfSample("outside_band", None, None, cell))
            elif code == 0 and np.isfinite(row).all():
                g = row[1:].astype(float)
                if rotation is not None:
                    g = rotation @ g
                if not np.isfinite(g).all():
                    raise SolverFailure("SDF gradient exceeds frame precision")
                output.append(SdfSample("valid", float(row[0]), tuple(map(float, g)), cell))
            else:
                raise SolverFailure("SDF interpolation outside numeric precision")
        return tuple(output)

    def _sample(self, points, rotation=None):
        bases, fraction = self._locate(points)
        status = np.where(bases < 0, 1, 0).astype(np.int32)
        raw = np.zeros((len(bases), 4))
        valid = np.flatnonzero(bases >= 0)
        nx, ny, _ = self.shape_xyz
        offsets = np.array([0, 1, nx, nx + 1, nx * ny, nx * ny + 1, nx * ny + nx, nx * ny + nx + 1])
        v = self.values_zyx_m.ravel()[bases[valid, None] + offsets]
        if self.band_m is not None:
            inside = np.all(np.abs(v) < self.band_m, axis=1)
            status[valid[~inside]] = 2
            valid, v = valid[inside], v[inside]
        x, y, z = fraction[valid].T

        def mix(a, b, t):
            return a + (b - a) * t

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            a, b, c, d, e, f, g, h = v.T
            ab, cd, ef, gh = mix(a, b, x), mix(c, d, x), mix(e, f, x), mix(g, h, x)
            near, far = mix(ab, cd, y), mix(ef, gh, y)
            raw[valid, 0] = mix(near, far, z)
            raw[valid, 1] = mix(mix(b - a, d - c, y), mix(f - e, h - g, y), z) / self.spacing_m[0]
            raw[valid, 2] = mix(cd - ab, gh - ef, z) / self.spacing_m[1]
            raw[valid, 3] = (far - near) / self.spacing_m[2]
        return self._results(bases, status, raw, rotation)

    def sample_local(self, points_m):
        """Return one SdfSample per asset-local point, in input order."""
        return self._sample(_points(points_m, _MAX_BATCH))

    def sample_world(self, points_m, *, translation_m, orientation_xyzw):
        """Transform world points to the asset and return derivatives in world axes."""
        p, r = _world(_points(points_m, _MAX_BATCH), translation_m, orientation_xyzw)
        return self._sample(p, r)


def bake_mesh_sdf(shape, vertices_local_m, *, shape_xyz, origin_m, spacing_m, band_m=None):
    """Bake finite-triangle distances and closed-solid signs at the local lattice.

    geometry_error_bound_m bounds the valid local scalar interpolation error
    against this mesh, including node storage and HOST evaluation roundoff.
    The gradient is the derivative of the interpolated field inside its chosen
    cell. Surface features, triangle labels and material weights require a
    separate geometry query.
    """
    from ._sdf_mesh_bake import bake_nodes

    dims, origin, spacing, band = _layout(shape_xyz, origin_m, spacing_m, band_m)
    values, node_error, stats = bake_nodes(shape, vertices_local_m, dims, origin, spacing)
    grid = SignedDistanceGrid(values, origin_m=origin, spacing_m=spacing, band_m=band)
    # The signed Euclidean distance is 1-Lipschitz. Trilinear weights define
    # a distribution of corners with variance sum(h_i²*u_i*(1-u_i)).
    # Its expected distance from the query is at most half a cell diagonal.
    half_diag = sqrt_bounds(sum(F(h) ** 2 for h in spacing), 96)[1] / 2
    # Locating a representable knot may snap to its exact lattice position.
    # Bound that positional rounding and the floating-point fraction conversion.
    localization = sum(
        F(max(ulp(o), ulp(float(F(o) + (n - 1) * F(h))))) / 2 + F(h) * F(ulp(1.0))
        for n, o, h in zip(dims, origin, spacing, strict=True)
    )
    # The bounded chain of binary64 lerps evaluates the stored node values.
    # Keeping this margin separate also identifies the TYX FP32 evaluator's
    # additional numerical error when both consume the same grid asset.
    evaluation = 64 * F(ulp(float(np.max(np.abs(grid.values_zyx_m)))))
    total = half_diag + node_error + localization + evaluation
    bound = float(total)
    if F(bound) < total:
        bound = nextafter(bound, inf)
    stats.update(
        localization_roundoff_m=nextafter(float(localization), inf),
        host_evaluation_roundoff_m=nextafter(float(evaluation), inf),
    )
    object.__setattr__(grid, "geometry_error_bound_m", bound)
    object.__setattr__(grid, "_bake", tuple(stats.items()))
    return grid
