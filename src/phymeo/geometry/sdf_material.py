"""Discrete point, offset-particle and finite-face observations in a registered rigid scalar field."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction as F
from math import inf, isfinite, nextafter
from numbers import Integral

import numpy as np

from ..constraints.collision_shapes import (
    CollisionKey,
    CollisionModel,
    CollisionMotion,
    CollisionShape,
)
from ..core.capabilities import SolverFailure
from ._sdf_material_triangle import FieldPlan, affine, plan_material
from ._sheet_pair_bounds import _dot, _sub
from .sdf_grid import _MAX_BATCH, SignedDistanceGrid, _points, _vector, bake_mesh_sdf


def _axes(q):
    # Normalization cancels in the products of quaternion components. The
    # rotation of the stored binary quaternion is an exact rational matrix.
    x, y, z, w = map(F, q)
    n = x * x + y * y + z * z + w * w
    return (
        (1 - 2 * (y * y + z * z) / n, 2 * (x * y - z * w) / n, 2 * (x * z + y * w) / n),
        (2 * (x * y + z * w) / n, 1 - 2 * (x * x + z * z) / n, 2 * (y * z - x * w) / n),
        (2 * (x * z - y * w) / n, 2 * (y * z + x * w) / n, 1 - 2 * (x * x + y * y) / n),
    )


def _fractions(points):
    return tuple(tuple(F(float(x)) for x in p) for p in points)


@dataclass(frozen=True, eq=False)
class SdfFieldPose:
    """A scalar field in a registered rigid frame. The plain constructor binds field values only."""

    key: CollisionKey
    grid: SignedDistanceGrid
    translation_m: tuple = (0.0, 0.0, 0.0)
    orientation_xyzw: tuple = (0.0, 0.0, 0.0, 1.0)
    _source: tuple | None = field(init=False, default=None, repr=False)
    _rotation: tuple = field(init=False, repr=False)
    _translation: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.key, CollisionKey) or not isinstance(self.grid, SignedDistanceGrid):
            raise TypeError("field requires a CollisionKey and SignedDistanceGrid")
        t = _vector(self.translation_m, 3, "translation_m")
        q = _vector(self.orientation_xyzw, 4, "orientation_xyzw")
        if abs(np.hypot.reduce(q) - 1) > 1e-7:
            raise ValueError("field orientation must be a unit quaternion")
        for name, val in (
            ("translation_m", t),
            ("orientation_xyzw", q),
            ("_translation", tuple(map(F, t))),
            ("_rotation", _axes(q)),
        ):
            object.__setattr__(self, name, val)

    @classmethod
    def bake(
        cls,
        shape,
        vertices_local_m,
        *,
        shape_xyz,
        origin_m,
        spacing_m,
        band_m=None,
        translation_m=(0.0, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    ):
        """Bake once and retain immutable source geometry for optional current-pose verification."""
        if not isinstance(shape, CollisionShape) or shape.kind != "mesh":
            raise ValueError("source baking requires a closed mesh registration")
        snapshot = _points(vertices_local_m, len(shape.vertex_ids)).copy()
        grid = bake_mesh_sdf(
            shape,
            snapshot,
            shape_xyz=shape_xyz,
            origin_m=origin_m,
            spacing_m=spacing_m,
            band_m=band_m,
        )
        result = cls(shape.key, grid, translation_m, orientation_xyzw)
        object.__setattr__(result, "_source", (shape, _fractions(snapshot)))
        return result

    def at_pose(self, *, translation_m, orientation_xyzw):
        """Reuse the existing field and source registration at another discrete rigid pose."""
        result = type(self)(self.key, self.grid, translation_m, orientation_xyzw)
        object.__setattr__(result, "_source", self._source)
        return result


_Point = tuple[float, float, float]
_Bounds = tuple[float | None, float | None]


@dataclass(frozen=True)
class SdfMaterialSample:
    """One field evaluation with deformable material weights and a first-order surface estimate."""

    weights: tuple[tuple[int, float], ...]
    world_point_m: _Point
    local_point_m: _Point
    status: str
    cell_xyz: tuple[int, int, int] | None
    field_distance_m: float | None
    field_gap_m: float | None
    exact_field_gap_bounds_m: _Bounds
    gradient_local: _Point | None
    gradient_world: _Point | None
    projected_surface_estimate_world_m: _Point | None


@dataclass(frozen=True)
class SdfMaterialResult:
    """Field and optional source-solid bounds on the complete requested material."""

    material_key: CollisionKey
    field_status: str
    reasons: tuple[str, ...]
    field_gap_bounds_m: _Bounds
    source_geometry_status: str
    source_gap_bounds_m: _Bounds
    samples: tuple[SdfMaterialSample, ...]
    cells_examined: int
    cells_covered: int


@dataclass(frozen=True)
class SdfContactBatch:
    """Material results in CollisionKey order for a single discrete field pose."""

    field_key: CollisionKey
    backend: str
    results: tuple[SdfMaterialResult, ...]
    samples_evaluated: int


@dataclass
class _Work:
    shape: CollisionShape
    ids: tuple = ()
    world: tuple = ()
    local: tuple = ()
    plan: FieldPlan = field(default_factory=FieldPlan)
    state: str | None = None


def _source_error(pose, rigid, vertices):
    if pose._source is None:
        return "unbound", None
    shape, local = pose._source
    if shape != rigid or vertices is None:
        return "pose_unverified", None
    actual = _points(vertices, len(local))
    if len(actual) != len(local):
        return "pose_unverified", None
    delta = F(0)
    for p, w in zip(local, actual, strict=True):
        nominal = tuple(
            _dot(axis, p) + t for axis, t in zip(pose._rotation, pose._translation, strict=True)
        )
        try:
            if tuple(map(float, nominal)) != tuple(w):
                return "pose_unverified", None
        except OverflowError:
            return "pose_unverified", None
        delta = max(delta, sum(abs(x - F(float(y))) for x, y in zip(nominal, w, strict=True)))
    # Corresponding rigid triangle points differ by <= delta. At a sign
    # change both signed distances are within delta of the boundary swept
    # under the vertex correspondence, so twice delta also covers that case.
    return "verified", F(pose.grid.geometry_error_bound_m) + 2 * delta


def _limit(value, maximum, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not 0 <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer between zero and {maximum}")
    return int(value)


def _inside_material(model, rigid, shape):
    if rigid.key.world != shape.key.world:
        return "filtered", "different_world"
    if shape.rigid and rigid.key.body == shape.key.body:
        return "filtered", "same_rigid_body"
    if not (rigid.group & shape.mask and shape.group & rigid.mask):
        return "filtered", "mask"
    if tuple(sorted((rigid.key, shape.key))) in model.excluded:
        return "filtered", "excluded"
    if shape.kind not in ("particle", "rounded_particle", "triangle"):
        return "unsupported", "material_kind"
    if shape.kind == "triangle" and shape.radius:
        return "unsupported", "triangle_radius"
    return None, None


def _prepare(model, rigid, pose, shape, points, cell_budget, sample_budget, depth):
    work = _Work(shape)
    work.state, reason = _inside_material(model, rigid, shape)
    if reason:
        work.plan.reasons.add(reason)
        return work
    if isinstance(points, CollisionMotion):
        raise TypeError("pass a discrete array of current world vertices")
    p = _points(points, len(shape.vertex_ids))
    if len(p) != len(shape.vertex_ids):
        raise ValueError("material positions must follow all registered vertex IDs")
    order = sorted(range(len(p)), key=shape.vertex_ids.__getitem__)
    work.ids = tuple(shape.vertex_ids[i] for i in order)
    work.world = _fractions(p[order])
    inverse = tuple(zip(*pose._rotation, strict=True))
    work.local = tuple(
        tuple(_dot(axis, _sub(x, pose._translation)) for axis in inverse) for x in work.world
    )
    try:
        for point in work.local:
            tuple(map(float, point))
    except OverflowError:
        work.plan.reasons.add("coordinate_precision")
        return work
    work.plan = plan_material(
        pose.grid, work.local, shape.radius, cell_budget, sample_budget, depth
    )
    return work


def _out(x, upper):
    if x is None:
        return None
    try:
        v = float(x)
    except OverflowError:
        return None
    if (upper and F(v) < x) or (not upper and F(v) > x):
        v = nextafter(v, inf if upper else -inf)
    return v if isfinite(v) else None


def _interval(lo, hi):
    return _out(lo, False), _out(hi, True)


def _status(lo, hi):
    if hi is not None and hi <= 0:
        return "contact"
    if lo is not None and lo > 0:
        return "clear"
    return "unresolved"


def _sample(work, weights, point, raw, rotation):
    expected, owner, gap = work.plan.addresses[weights]
    world = tuple(map(float, affine(work.world, weights)))
    address = tuple((i, float(w)) for i, w in zip(work.ids, weights, strict=True) if w)
    if raw.status == "numeric_precision":
        status = "numeric_precision"
    else:
        status = (
            raw.status
            if raw.status == expected and raw.cell_xyz == owner
            else "coordinate_precision"
        )
    local_g = raw.gradient
    world_g = projection = None
    offset = None
    if status == "valid":
        offset = raw.distance_m - work.shape.radius
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            g = rotation @ np.asarray(local_g)
            length = np.hypot.reduce(g)
            if not np.isfinite(g).all() or not isfinite(offset):
                offset = None
                status = "numeric_precision"
            else:
                world_g = tuple(map(float, g))
                if isfinite(length) and length > 0:
                    estimate = np.asarray(world) - (raw.distance_m / length) * (g / length)
                    if np.isfinite(estimate).all():
                        projection = tuple(map(float, estimate))
    if status != expected:
        work.plan.reasons.add(status)
        gap = None
    return SdfMaterialSample(
        address,
        world,
        point,
        status,
        raw.cell_xyz,
        raw.distance_m,
        offset,
        _interval(gap, gap),
        local_g,
        world_g,
        projection,
    ), gap


def _finish(work, observations, source_state, error):
    plan = work.plan
    upper = min((g for _, g in observations if g is not None), default=None)
    lower = None if plan.reasons else plan.lower
    state = work.state or _status(lower, upper)
    if state == "unresolved" and not plan.reasons:
        plan.reasons.add("field_precision")
    source_bounds = (None, None)
    geometry = source_state
    if error is not None:
        lo = lower - error if lower is not None else None
        hi = upper + error if upper is not None else None
        geometry = work.state or _status(lo, hi)
        source_bounds = _interval(lo, hi)
    return SdfMaterialResult(
        work.shape.key,
        state,
        tuple(sorted(plan.reasons)),
        _interval(lower, upper),
        geometry,
        source_bounds,
        tuple(s for s, _ in observations),
        plan.cells_examined,
        plan.cells_covered,
    )


def sample_sdf_contacts(
    model,
    field,
    positions_world_m,
    *,
    rigid_vertices_world_m=None,
    sampler=None,
    max_cells=4096,
    max_samples=16384,
    max_depth=2,
):
    """Observe this discrete pose of the requested deformable shape keys against one field.

    Bounds enclose the minimum of the stored continuous field minus particle
    radius on each entire material primitive. An absent lower bound leaves
    unvisited regions unknown. Source-solid claims additionally require the
    baked registration and its current posed material vertices.
    """
    if (
        not isinstance(model, CollisionModel)
        or not isinstance(field, SdfFieldPose)
        or not isinstance(positions_world_m, Mapping)
    ):
        raise TypeError("expected CollisionModel, SdfFieldPose and a mapping of discrete positions")
    cells = _limit(max_cells, 65536, "max_cells")
    samples = _limit(max_samples, _MAX_BATCH, "max_samples")
    depth = _limit(max_depth, 4, "max_depth")
    shapes = {s.key: s for s in model.shapes}
    rigid = shapes.get(field.key)
    if rigid is None or not rigid.rigid:
        raise ValueError("field key must bind a registered rigid shape")
    if any(k not in shapes or k == field.key for k in positions_world_m):
        raise ValueError("requested materials must have registered keys distinct from the field")
    backend = "host"
    if sampler is not None:
        from ..backends.tyx_sdf import TyxSdfSampler

        if not isinstance(sampler, TyxSdfSampler) or sampler.grid is not field.grid:
            raise ValueError("SDF sampler must bind the same grid asset")
        if sampler.closed:
            raise RuntimeError("SDF sampler is closed")
        backend = sampler.stats()["backend"]
    source_state, error = _source_error(field, rigid, rigid_vertices_world_m)
    works = []
    queue = []
    local_points = []
    for key in sorted(positions_world_m):
        work = _prepare(
            model, rigid, field, shapes[key], positions_world_m[key], cells, samples, depth
        )
        cells -= work.plan.cells_examined
        samples -= len(work.plan.addresses)
        source_index = len(works)
        for weights in work.plan.addresses:
            queue.append((source_index, weights))
            local_points.append(tuple(map(float, affine(work.local, weights))))
        works.append(work)
    rows = [[] for w in works]
    block = sampler.capacity if sampler is not None else _MAX_BATCH
    rotation = np.asarray(field._rotation, dtype=float)
    for start in range(0, len(queue), block):
        points = local_points[start : start + block]
        try:
            read = (field.grid if sampler is None else sampler).sample_local(points)
        except SolverFailure:
            if sampler is not None:
                raise
            # A huge external scalar may have an unrepresentable derivative.
            # Keep well-conditioned rows usable without fabricating a gradient.
            from .sdf_grid import SdfSample

            read = []
            for p in points:
                try:
                    read.extend(field.grid.sample_local([p]))
                except SolverFailure:
                    read.append(SdfSample("numeric_precision", None, None, None))
        for (i, w), p, raw in zip(queue[start : start + block], points, read, strict=True):
            rows[i].append(_sample(works[i], w, p, raw, rotation))
    results = tuple(
        _finish(w, row, source_state, error) for w, row in zip(works, rows, strict=True)
    )
    return SdfContactBatch(field.key, backend, results, len(queue))
