"""Immutable shape registrations and motion paths for collision detection."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..core.validation import finite_real
from .curved_solids import CurvedSolid
from .heightfield import Heightfield

MAX_SHAPES = 256
MAX_VERTICES = 8192
MAX_CONVEX_VERTICES = 64
MAX_MESH_TRIANGLES = 128
_INT_MAX = 2**31 - 1


def _integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not 0 <= value <= _INT_MAX
    ):
        raise ValueError(f"{name} must be a nonnegative i32 integer")
    return int(value)


def _vector(value, name):
    a = np.asarray(value)
    if a.shape != (3,) or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite real 3-vector")
    return tuple(finite_real(v, name) for v in a)


def _points(value, name):
    a = np.asarray(value)
    if (
        a.ndim != 2
        or a.shape[1] != 3
        or not 1 <= len(a) <= MAX_VERTICES
        or a.dtype.kind not in "fiu"
    ):
        raise ValueError(f"{name} must be a bounded nonempty array of 3D vertices")
    data = np.asarray(a, dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError(f"{name} must contain finite FP64 coordinates")
    return np.frombuffer(data.tobytes(order="C"), dtype=np.float64).reshape(-1, 3)


@dataclass(frozen=True, order=True)
class CollisionKey:
    world: int
    body: int
    shape: int

    def __post_init__(self):
        for name in ("world", "body", "shape"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))


@dataclass(frozen=True)
class CollisionShape:
    """Material IDs follow the motion vertices. A plane has one anchor and a fixed normal."""

    key: CollisionKey
    kind: str
    vertex_ids: tuple[int, ...]
    radius: float = 0.0
    group: int = 1
    mask: int = _INT_MAX
    revision: int = 0
    normal: tuple[float, float, float] | None = None
    triangles: tuple[tuple[int, int, int], ...] = ()
    heightfield: Heightfield | None = None
    curved: CurvedSolid | None = None

    def __post_init__(self):
        if not isinstance(self.key, CollisionKey):
            raise TypeError("shape key must be a CollisionKey")
        counts = {
            "sphere": 1,
            "cylinder": 1,
            "cone": 1,
            "ellipsoid": 1,
            "particle": 1,
            "capsule": 2,
            "triangle": 3,
            "tetrahedron": 4,
            "plane": 1,
        }
        if self.kind not in (*counts, "hull", "mesh", "plate", "heightfield"):
            raise ValueError("unsupported collision shape kind")
        if not 1 <= len(self.vertex_ids) <= MAX_VERTICES:
            raise ValueError("vertex count outside limits")
        ids = tuple(_integer(v, "vertex ID") for v in self.vertex_ids)
        if (
            len(ids) != len(set(ids))
            or (self.kind == "hull" and len(ids) < 4)
            or (self.kind == "plate" and not 3 <= len(ids) <= 32)
            or (self.kind in counts and len(ids) != counts[self.kind])
        ):
            raise ValueError("invalid vertex IDs or count for shape kind")
        if self.kind == "mesh":
            if not 4 <= len(self.triangles) <= MAX_MESH_TRIANGLES:
                raise ValueError("closed mesh face count outside limits")
            known = set(ids)
            faces = []
            for face in self.triangles:
                if len(face) != 3:
                    raise ValueError("closed mesh faces must be triangles")
                tri = tuple(_integer(i, "triangle material ID") for i in face)
                if len(set(tri)) != 3 or not set(tri) <= known:
                    raise ValueError("closed mesh triangle has invalid material IDs")
                faces.append(min(tri, tri[1:] + tri[:1], tri[2:] + tri[:2]))
            object.__setattr__(self, "triangles", tuple(sorted(faces)))
        elif len(self.triangles):
            raise ValueError("triangles require a closed mesh shape")
        else:
            object.__setattr__(self, "triangles", ())
        radius = finite_real(self.radius, "radius")
        if self.kind in ("cylinder", "cone", "ellipsoid"):
            if not isinstance(self.curved, CurvedSolid) or self.curved.kind != self.kind:
                raise ValueError("curved shape kind must match its registered geometry")
            if radius:
                raise ValueError("curved solid dimensions specify the entire shape")
        elif self.curved is not None:
            raise ValueError("curved geometry requires a cylinder, cone or ellipsoid")
        if self.kind == "heightfield":
            if (
                not isinstance(self.heightfield, Heightfield)
                or len(ids) != self.heightfield.vertex_count
            ):
                raise ValueError("heightfield shape must bind one material ID per grid sample")
            if radius != 0:
                raise ValueError("heightfield has zero surface offset")
        elif self.heightfield is not None:
            raise ValueError("heightfield data requires a heightfield shape")
        if radius < 0 or (self.kind in ("sphere", "capsule") and radius == 0):
            raise ValueError("invalid shape radius or maximum surface offset")
        if (self.kind == "plate" and radius == 0) or (self.kind == "particle" and radius != 0):
            raise ValueError(
                "a plate has positive rounding and a material particle has zero radius"
            )
        if self.kind == "plane":
            normal = _vector(self.normal, "plane normal")
            if not any(normal) or radius != 0:
                raise ValueError("a half-space requires a nonzero normal and zero radius")
            object.__setattr__(self, "normal", normal)
        elif self.normal is not None:
            raise ValueError("normal is only defined for a plane")
        object.__setattr__(self, "vertex_ids", ids)
        object.__setattr__(self, "radius", radius)
        for name in ("group", "mask", "revision"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))

    @property
    def rigid(self):
        return self.curved is not None or self.kind in (
            "sphere",
            "capsule",
            "hull",
            "mesh",
            "plate",
            "plane",
            "heightfield",
        )


@dataclass(frozen=True)
class CollisionModel:
    shapes: tuple[CollisionShape, ...]
    excluded: tuple[tuple[CollisionKey, CollisionKey], ...] = ()

    def __post_init__(self):
        if len(self.shapes) > MAX_SHAPES or not all(
            isinstance(s, CollisionShape) for s in self.shapes
        ):
            raise ValueError("invalid shape table or shape count")
        shapes = tuple(sorted(self.shapes, key=lambda s: s.key))
        keys = {s.key for s in shapes}
        bodies = {}
        for s in shapes:
            body = (s.key.world, s.key.body)
            if bodies.setdefault(body, s.rigid) != s.rigid:
                raise ValueError("a body cannot mix rigid and deformable shapes")
        if len(keys) != len(shapes) or sum(len(s.vertex_ids) for s in shapes) > MAX_VERTICES:
            raise ValueError("duplicate shape key or model vertex capacity exceeded")
        exclusions = set()
        for a, b in self.excluded:
            if a == b or a not in keys or b not in keys:
                raise ValueError("invalid excluded shape pair")
            pair = tuple(sorted((a, b)))
            if pair in exclusions:
                raise ValueError("duplicate excluded shape pair")
            exclusions.add(pair)
        object.__setattr__(self, "shapes", shapes)
        object.__setattr__(self, "excluded", tuple(sorted(exclusions)))


@dataclass(frozen=True)
class RigidTurn:
    """Linear world pivot path, constant world axis, signed angle in radians."""

    pivot_start: tuple[float, float, float]
    pivot_end: tuple[float, float, float]
    axis: tuple[float, float, float]
    angle: float

    def __post_init__(self):
        for name in ("pivot_start", "pivot_end", "axis"):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        norm = float(np.hypot.reduce(self.axis))
        if abs(norm - 1) > 1e-12:
            raise ValueError("rotation axis must have unit length")
        object.__setattr__(self, "axis", tuple(v / norm for v in self.axis))
        object.__setattr__(self, "angle", finite_real(self.angle, "angle"))


@dataclass(frozen=True, eq=False)
class CollisionMotion:
    """Single pose, affine vertex path, or rigid arc about a moving pivot."""

    start: np.ndarray
    end: np.ndarray | None = None
    turn: RigidTurn | None = None

    def __post_init__(self):
        start = _points(self.start, "start")
        object.__setattr__(self, "start", start)
        if self.turn is not None and (not isinstance(self.turn, RigidTurn) or self.end is not None):
            raise ValueError("a rigid turn supplies its own trajectory; omit end vertices")
        if self.end is not None:
            end = _points(self.end, "end")
            if start.shape != end.shape:
                raise ValueError("motion vertex counts must match")
            object.__setattr__(self, "end", end)

    @property
    def moving(self):
        if self.turn is not None:
            return self.turn.angle != 0 or self.turn.pivot_start != self.turn.pivot_end
        return self.end is not None and not np.array_equal(self.start, self.end)


@dataclass(frozen=True)
class CollisionPolicy:
    """Candidate capacity and additional *pair* detection margin, in metres."""

    candidate_capacity: int = MAX_SHAPES * (MAX_SHAPES - 1) // 2
    margin: float = 0.0
    proximity_patches: bool = False

    def __post_init__(self):
        object.__setattr__(
            self, "candidate_capacity", _integer(self.candidate_capacity, "capacity")
        )
        if not isinstance(self.proximity_patches, bool):
            raise TypeError("proximity_patches must be bool")
        margin = finite_real(self.margin, "margin")
        if margin < 0:
            raise ValueError("detection margin cannot be negative")
        object.__setattr__(self, "margin", margin)
