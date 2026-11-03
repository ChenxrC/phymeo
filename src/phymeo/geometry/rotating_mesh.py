"""Conservative HOST face pairs over the complete rigid paths of two closed triangle meshes."""

from dataclasses import dataclass
from fractions import Fraction as F
from math import ceil, fsum, ulp
from numbers import Integral

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionMotion, CollisionShape
from ..core.capabilities import SolverFailure
from ..core.validation import finite_real
from ._closed_mesh import ClosedMesh
from ._collision_bvh import CollisionTrees
from ._prism_pair_exact import sqrt_bounds
from ._rotation_angle import axis_length_bounds, rotate_enclosures
from ._round_sweep import _vector
from ._triangle_pair_sweep import _cross3

_MAX_WINDOWS = 128
_MAX_CANDIDATES = 65536


@dataclass(frozen=True)
class MeshArcFacePair:
    """Two oriented triangles identified by their owners' original material node IDs."""

    first: tuple[int, int, int]
    second: tuple[int, int, int]

    def material_weights(self, first_barycentric, second_barycentric):
        """Bind two caller-supplied face points to their material IDs, in owner order."""
        result = []
        for face, coordinates in ((self.first, first_barycentric), (self.second, second_barycentric)):
            weights = tuple(finite_real(x, "barycentric weight") for x in coordinates)
            if len(weights) != 3 or min(weights) < 0 or max(weights) > 1 or abs(fsum(weights)-1)>8*ulp(1.0):
                raise ValueError("barycentric coordinates must be nonnegative and sum to one")
            result.append(tuple(zip(face, weights, strict=True)))
        return tuple(result)


@dataclass(frozen=True)
class MeshArcWindow:
    """Closed normalized time interval and retained face pairs; adjacent windows share endpoints."""

    interval: tuple[float, float]
    pairs: tuple[MeshArcFacePair, ...]


@dataclass(frozen=True)
class MeshArcCandidates:
    first: CollisionKey
    second: CollisionKey
    windows: tuple[MeshArcWindow, ...]
    possible_pairs: int
    candidate_count: int
    bvh_node_visits: int
    bvh_leaf_tests: int


class _MeshArc:
    def __init__(self, shape, motion):
        if not isinstance(shape, CollisionShape) or shape.kind != "mesh" or shape.radius:
            raise ValueError("mesh arc candidates require unrounded closed meshes")
        if not isinstance(motion, CollisionMotion) or len(motion.start) != len(shape.vertex_ids):
            raise ValueError("mesh arc motion must contain one start position per registered node")
        mesh = ClosedMesh(shape, motion)
        self.start = np.array([mesh.points[i] for i in shape.vertex_ids])
        self.step = mesh.step
        if motion.end is not None and any(
            not np.array_equal(_vector(end) - start, self.step)
            for end,start in zip(motion.end, self.start, strict=True)
        ):
            raise ValueError("mesh arc candidates require rigid motion")
        by_id = {node: i for i,node in enumerate(shape.vertex_ids)}
        self.rows = np.array([[by_id[node] for node in f.ids] for f in mesh.facets], dtype=np.intp)
        self.faces = tuple(f.ids for f in mesh.facets)
        self.angle = F(0) if motion.turn is None else F(motion.turn.angle)
        self.curvature = np.zeros_like(self.start)
        if not self.angle:
            return
        self.pivot = _vector(motion.turn.pivot_start)
        self.axis = _vector(motion.turn.axis)
        self.length = axis_length_bounds(self.axis)
        self.local = self.start - self.pivot
        axis2 = self.axis @ self.axis
        for i, point in enumerate(self.local):
            perp = point - self.axis * ((self.axis @ point) / axis2)
            cross = _cross3(self.axis, point)
            for k in range(3):
                # The k-th coordinate is affine + B*cos(omega*t) + D*sin(omega*t).
                amplitude2 = perp[k]**2 + cross[k]**2 / axis2
                self.curvature[i,k] = self.angle**2 * sqrt_bounds(amplitude2,80)[1]

    def positions(self, time):
        if not self.angle or not time:
            pts = self.start + time * self.step
            return pts, pts
        shift = self.pivot + time * self.step
        rotated = rotate_enclosures(self.local, self.axis, self.length, time*self.angle)
        points = np.array([p + shift for p,_ in rotated])
        errors = np.array([e for _,e in rotated],dtype=object)[:,None]
        errors = np.where(self.curvature != 0,errors,0)
        return points-errors, points+errors

    def boxes(self, left, right, width, pad):
        # Linear interpolation between the actual endpoints differs from the arc by
        # at most max|x''|*width²/8 on each coordinate. A moving pivot is exactly affine.
        from .collision_pipeline import _outward

        remainder = self.curvature * (width * width / 8) + pad
        lo = np.minimum(left[0], right[0]) - remainder
        hi = np.maximum(left[1], right[1]) + remainder
        try:
            lower = np.array([[_outward(x,False) for x in p] for p in lo],dtype=np.float32)
            upper = np.array([[_outward(x,True) for x in p] for p in hi],dtype=np.float32)
        except ValueError as exc:
            raise SolverFailure("mesh_arc_bound_outside_fp32") from exc
        return np.column_stack((lower[self.rows].min(1), upper[self.rows].max(1)))


def mesh_arc_face_candidates(first, second, first_motion, second_motion, *, margin_m=0.0, max_candidates=_MAX_CANDIDATES):
    """Bound the Cartesian product of the two directed face tables in each time window.

    Both shapes must be unrounded, closed, in the same world and have distinct keys.
    Each path can be static, a rigid translation, or a signed rotation about a fixed
    world axis and linearly moving pivot. A nonnegative Cartesian margin is split
    between the two boxes. Empty windows exclude surface intersections in that
    window. They do not exclude initial or persisting solid containment.
    """
    if first.key.world != second.key.world or first.key == second.key:
        raise ValueError("mesh arc candidates require distinct shape keys in the same world")
    pad = F(finite_real(margin_m,"margin_m")) / 2
    if pad < 0:
        raise ValueError("mesh arc margin must be nonnegative")
    if isinstance(max_candidates,(bool,np.bool_)) or not isinstance(max_candidates,Integral) or not 0 <= max_candidates <= _MAX_CANDIDATES:
        raise ValueError(f"max_candidates must be an integer in [0, {_MAX_CANDIDATES}]")
    a,b = _MeshArc(first,first_motion), _MeshArc(second,second_motion)
    minimum = max(1,ceil(4*max(abs(a.angle),abs(b.angle))))
    if minimum > _MAX_WINDOWS:
        raise SolverFailure(f"mesh_arc_window_capacity required_at_least={minimum} limit={_MAX_WINDOWS}")
    count = 1 << (minimum-1).bit_length()
    width = F(1,count)
    trees, layouts = CollisionTrees(), {}
    key = ("mesh",second.key,second.revision)
    left_a,left_b = a.positions(F(0)),b.positions(F(0))
    windows = []
    retained = 0
    for i in range(count):
        t1 = (i+1)*width
        right_a, right_b = a.positions(t1),b.positions(t1)
        queries = a.boxes(left_a,right_a,width,pad)
        leaves = b.boxes(left_b,right_b,width,pad)
        rows = trees.facets(layouts,key,leaves,queries,None)
        retained += sum(len(row) for row in rows)
        if retained <= max_candidates:
            pairs = tuple(MeshArcFacePair(a.faces[j],b.faces[k]) for j,row in enumerate(rows) for k in row)
            windows.append(MeshArcWindow((float(i*width),float(t1)),pairs))
        left_a,left_b = right_a,right_b
    if retained > max_candidates:
        raise SolverFailure(f"mesh_arc_candidate_capacity required={retained} limit={max_candidates}")
    return MeshArcCandidates(
        first.key,second.key,tuple(windows),count*len(a.faces)*len(b.faces),retained,
        trees.counts["mesh_bvh_node_visits"],trees.counts["mesh_bvh_leaf_tests"],
    )
