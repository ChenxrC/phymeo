"""Owned 3D area-pressure samples for an already clipped compliant/rigid surface.

A is compliant, B is rigid, and triangle winding points from B to A. This
module performs quadrature only: it neither establishes collision validity nor
proves disjoint triangles, field continuity, or a binding to a runtime state.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

MAX_TRIANGLES = 1024
MAX_SAMPLES = 3 * MAX_TRIANGLES


def _identifier(value, name):
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise ValueError(f"{name} must be a nonempty string of at most 128 characters")
    return str(value)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} is not representable in FP64") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sequence(value, cap, name):
    if not isinstance(value, (tuple, list)) or len(value) > cap:
        raise ValueError(f"{name} must be a bounded tuple or list with at most {cap} entries")
    return tuple(value)


def _array(value, shape, name):
    """Check the complete bounded shape before making an owned FP64 copy."""

    def check(item, remaining):
        if isinstance(item, np.ndarray):
            if item.shape != remaining or item.dtype.kind not in "fiu":
                raise ValueError(f"{name} must be a real array with shape {shape}")
        elif remaining:
            if not isinstance(item, (list, tuple)) or len(item) != remaining[0]:
                raise ValueError(f"{name} must have shape {shape}")
            for child in item:
                check(child, remaining[1:])
        elif isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
            raise ValueError(f"{name} must contain real numbers")

    check(value, shape)
    try:
        with np.errstate(over="raise", invalid="raise"):
            array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError, FloatingPointError) as error:
        raise ValueError(f"{name} is not a finite FP64 array") from error
    # An empty nested sequence cannot express trailing dimensions; only the
    # exact zero-size shape checked above is supplied here.
    if array.size == 0:
        array = array.reshape(shape)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(shape)


def _same(actual, expected, name):
    if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
        raise ValueError(f"inconsistent {name}")


@dataclass(frozen=True)
class AreaPressureIdentity:
    compliant_body: str
    rigid_body: str
    compliant_material: str
    rigid_material: str

    def __post_init__(self):
        for name in ("compliant_body", "rigid_body", "compliant_material", "rigid_material"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.compliant_body == self.rigid_body:
            raise ValueError("area pressure requires two distinct bodies")


@dataclass(frozen=True)
class AreaPressureStamp:
    world_id: str
    step_index: int
    sample_time_s: float
    configuration_id: str

    def __post_init__(self):
        object.__setattr__(self, "world_id", _identifier(self.world_id, "world_id"))
        object.__setattr__(self, "step_index", _integer(self.step_index, "step_index"))
        time = _scalar(self.sample_time_s, "sample_time_s")
        if time < 0:
            raise ValueError("sample_time_s must be nonnegative")
        object.__setattr__(self, "sample_time_s", time)
        object.__setattr__(
            self, "configuration_id", _identifier(self.configuration_id, "configuration_id")
        )


def _identity(value):
    if not isinstance(value, AreaPressureIdentity):
        raise ValueError("expected AreaPressureIdentity")
    return AreaPressureIdentity(
        value.compliant_body, value.rigid_body, value.compliant_material, value.rigid_material
    )


def _stamp(value):
    if not isinstance(value, AreaPressureStamp):
        raise ValueError("expected AreaPressureStamp")
    return AreaPressureStamp(
        value.world_id, value.step_index, value.sample_time_s, value.configuration_id
    )


def _triangle_geometry(vertices, pressure, gradient):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            cross = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            length = float(np.linalg.norm(cross))
            if not np.isfinite(length) or length <= 0:
                raise ValueError("triangle must have finite positive area")
            normal = cross / length
            area = length / 2
            effective = float(gradient @ normal)
            if area <= 0 or not np.isfinite(effective) or effective <= 0:
                raise ValueError("triangle requires positive area and grad(pA) dot n")
            for i, j in ((0, 1), (1, 2), (2, 0)):
                change = float(gradient @ (vertices[j] - vertices[i]))
                residual = abs(float(pressure[j] - pressure[i]) - change)
                bound = 1e-10 + 1e-12 * max(abs(pressure[i]), abs(pressure[j]), abs(change))
                if not np.isfinite(change) or not np.isfinite(residual) or residual > bound:
                    raise ValueError("vertex pressure differences disagree with the world gradient")
    except FloatingPointError as error:
        raise ValueError("triangle geometry or pressure arithmetic overflowed") from error
    return area, normal, effective


@dataclass(frozen=True, eq=False)
class AreaPressureTriangle:
    triangle_id: str
    compliant_cell_id: str
    rigid_face_id: str
    vertices: np.ndarray
    vertex_pressure: np.ndarray
    pressure_gradient: np.ndarray

    def __post_init__(self):
        for name in ("triangle_id", "compliant_cell_id", "rigid_face_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        vertices = _array(self.vertices, (3, 3), "vertices")
        pressure = _array(self.vertex_pressure, (3,), "vertex_pressure")
        gradient = _array(self.pressure_gradient, (3,), "pressure_gradient")
        if np.any(pressure < 0):
            raise ValueError("clipped vertex pressures must be nonnegative")
        _triangle_geometry(vertices, pressure, gradient)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "vertex_pressure", pressure)
        object.__setattr__(self, "pressure_gradient", gradient)


@dataclass(frozen=True, eq=False)
class AreaPressureSurface:
    identity: AreaPressureIdentity
    stamp: AreaPressureStamp
    triangles: tuple[AreaPressureTriangle, ...]

    def __post_init__(self):
        object.__setattr__(self, "identity", _identity(self.identity))
        object.__setattr__(self, "stamp", _stamp(self.stamp))
        triangles = _sequence(self.triangles, MAX_TRIANGLES, "triangles")
        checked = []
        seen = set()
        for triangle in triangles:
            if not isinstance(triangle, AreaPressureTriangle):
                raise ValueError("surface entries must be AreaPressureTriangle")
            item = AreaPressureTriangle(
                triangle.triangle_id,
                triangle.compliant_cell_id,
                triangle.rigid_face_id,
                triangle.vertices,
                triangle.vertex_pressure,
                triangle.pressure_gradient,
            )
            if item.triangle_id in seen:
                raise ValueError("duplicate triangle_id")
            seen.add(item.triangle_id)
            checked.append(item)
        object.__setattr__(self, "triangles", tuple(checked))


def _sample_ids(value):
    entries = _sequence(value, MAX_SAMPLES, "sample_ids")
    result = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError("each sample ID must contain triangle_id and local_slot")
        triangle = _identifier(entry[0], "sample triangle_id")
        slot = _integer(entry[1], "local_slot")
        item = (triangle, slot)
        if slot > 2 or item in seen:
            raise ValueError("sample IDs must be unique with local_slot in 0..2")
        seen.add(item)
        result.append(item)
    return tuple(result)


def _source_ids(value, count):
    entries = _sequence(value, MAX_SAMPLES, "source_ids")
    if len(entries) != count:
        raise ValueError("source_ids must match sample count")
    result = []
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError("source ID must contain compliant cell and rigid face")
        result.append(
            (_identifier(entry[0], "compliant_cell_id"), _identifier(entry[1], "rigid_face_id"))
        )
    return tuple(result)


def _triangle_samples(sample_ids, source_ids):
    ids = _sample_ids(sample_ids)
    sources = _source_ids(source_ids, len(ids))
    if len(ids) % 3:
        raise ValueError("quadrature requires three ordered samples per triangle")
    for i in range(0, len(ids), 3):
        expected = tuple((ids[i][0], slot) for slot in range(3))
        if ids[i : i + 3] != expected or sources[i : i + 3] != (sources[i],) * 3:
            raise ValueError("sample/source order differs from triangle slots")
    return ids, sources


def _normals(value, count):
    normals = _array(value, (count, 3), "normals")
    try:
        with np.errstate(over="raise", invalid="raise"):
            squared = np.sum(normals * normals, axis=1)
    except FloatingPointError as error:
        raise ValueError("normal magnitude overflowed") from error
    _same(squared, np.ones(count), "unit normals")
    return normals


@dataclass(frozen=True, eq=False)
class AreaPressureQuadrature:
    identity: AreaPressureIdentity
    stamp: AreaPressureStamp
    sample_ids: tuple[tuple[str, int], ...]
    source_ids: tuple[tuple[str, str], ...]
    points: np.ndarray
    normals: np.ndarray
    dA: np.ndarray
    pressure: np.ndarray
    gradient: np.ndarray
    total_area_m2: float

    def __post_init__(self):
        object.__setattr__(self, "identity", _identity(self.identity))
        object.__setattr__(self, "stamp", _stamp(self.stamp))
        ids, sources = _triangle_samples(self.sample_ids, self.source_ids)
        count = len(ids)
        values = {
            "points": _array(self.points, (count, 3), "points"),
            "normals": _normals(self.normals, count),
            "dA": _array(self.dA, (count,), "dA"),
            "pressure": _array(self.pressure, (count,), "pressure"),
            "gradient": _array(self.gradient, (count,), "gradient"),
        }
        if (
            np.any(values["dA"] <= 0)
            or np.any(values["gradient"] <= 0)
            or np.any(values["pressure"] < 0)
        ):
            raise ValueError("quadrature requires positive dA/gradient and nonnegative pressure")
        area = _scalar(self.total_area_m2, "total_area_m2")
        try:
            with np.errstate(over="raise", invalid="raise"):
                summed = float(values["dA"].sum())
        except FloatingPointError as error:
            raise ValueError("total quadrature area overflowed") from error
        if (
            not np.isfinite(summed)
            or area < 0
            or (count == 0 and area != 0)
            or (count and area <= 0)
        ):
            raise ValueError("invalid total quadrature area")
        if abs(area - summed) > 1e-12 * max(area, summed):
            raise ValueError("inconsistent total quadrature area")
        for i in range(0, count, 3):
            for name in ("normals", "dA", "gradient"):
                if not np.array_equal(values[name][i + 1], values[name][i]) or not np.array_equal(
                    values[name][i + 2], values[name][i]
                ):
                    raise ValueError("triangle samples must retain one normal, dA and gradient")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "sample_ids", ids)
        object.__setattr__(self, "source_ids", sources)
        object.__setattr__(self, "total_area_m2", area)


def _quadrature(value):
    if not isinstance(value, AreaPressureQuadrature):
        raise ValueError("expected AreaPressureQuadrature")
    return AreaPressureQuadrature(
        value.identity,
        value.stamp,
        value.sample_ids,
        value.source_ids,
        value.points,
        value.normals,
        value.dA,
        value.pressure,
        value.gradient,
        value.total_area_m2,
    )


def quadrature_area_pressure(surface):
    """Integrate PL pressure with three degree-two samples and real square metres."""
    if not isinstance(surface, AreaPressureSurface):
        raise ValueError("expected AreaPressureSurface")
    surface = AreaPressureSurface(surface.identity, surface.stamp, surface.triangles)
    count = 3 * len(surface.triangles)
    points, normals = np.empty((count, 3)), np.empty((count, 3))
    areas, pressure, gradient = (np.empty(count) for _ in range(3))
    sample_ids, source_ids = [], []
    barycentric = np.array(((2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for index, triangle in enumerate(surface.triangles):
                area, normal, effective = _triangle_geometry(
                    triangle.vertices, triangle.vertex_pressure, triangle.pressure_gradient
                )
                section = slice(3 * index, 3 * index + 3)
                points[section] = barycentric @ triangle.vertices
                normals[section] = normal
                areas[section] = area / 3
                pressure[section] = barycentric @ triangle.vertex_pressure
                gradient[section] = effective
                sample_ids.extend((triangle.triangle_id, slot) for slot in range(3))
                source_ids.extend(((triangle.compliant_cell_id, triangle.rigid_face_id),) * 3)
            total = float(areas.sum())
    except FloatingPointError as error:
        raise ValueError("area-pressure quadrature arithmetic overflowed") from error
    return AreaPressureQuadrature(
        surface.identity,
        surface.stamp,
        tuple(sample_ids),
        tuple(source_ids),
        points,
        normals,
        areas,
        pressure,
        gradient,
        total,
    )
