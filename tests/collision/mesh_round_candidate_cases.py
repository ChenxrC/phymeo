"""Frozen independent finite-mesh inputs for B3 candidate selection.

The distance oracle enumerates exact finite segment/triangle features in an
older test-only helper, never a production closest-feature or BVH routine.
The producer contract is frozen here before its implementation:

MeshRoundFaceSelector(shape, prepared_mesh, query_boxes)
.sample(start, end, radius=..., vertex_distance2={material_id: Fraction})
.translation(start, end, drift, radius=...)

Both return a sorted unique tuple of original facet-row indices. query_boxes is
called as query_boxes(binding, leaves_fp32, queries_fp32), with (F,6)/(1,6)
arrays, and returns one list of indices. The immutable binding has key,
revision, vertex_ids and faces (directed tuples in prepared_mesh facet order).
The callback is a real workspace BVH in integration tests; no fallback is
allowed for empty sample selection or malformed callback data.
"""

from dataclasses import dataclass
from fractions import Fraction as F

import numpy as np
from rotating_capsule_mesh_cases import (
    box,
    closed_oriented_reference,
    segment_triangle_reference,
    u_mesh,
)

import phymeo as ph


@dataclass(frozen=True)
class Case:
    name: str
    axis: tuple
    radius: float
    relation: str
    axis_gap_m: float | None
    mesh_kind: str = "box"
    mesh_step: tuple = (0.0, 0.0, 0.0)
    round_step: tuple = (0.0, 0.0, 0.0)
    first_t: F | None = None


FACE = Case("face", ((35 / 32, 0, 0),), 1 / 8, "boundary_overlap", -1 / 32)
EDGE = Case("edge", ((35 / 32, 9 / 8, 0),), 3 / 16, "boundary_overlap", -1 / 32)
SHAFT = Case("shaft", ((35 / 32, -1 / 2, 0), (35 / 32, 1 / 2, 0)), 1 / 8,
             "boundary_overlap", -1 / 32)
TIP = Case("tip", ((35 / 32, 0, 0), (51 / 32, 0, 0)), 1 / 8,
           "boundary_overlap", -1 / 32)
FINITE_MISS = Case("finite-miss", ((35 / 32, 2, 0), (35 / 32, 3, 0)), 1 / 8,
                   "separated", None)
CROSSING = Case("axis-crossing", ((-2, 1 / 8, 1 / 4), (2, 1 / 8, 1 / 4)), 1 / 8,
                "boundary_overlap", -1 / 8)
INSIDE = Case("inside", ((0, 0, 0),), 1 / 8, "mesh_contains_round", 7 / 8)
CONTAINS = Case("contains", ((0, 0, 0),), 2.0, "round_contains_mesh", -1.0)
NOTCH = Case("notch", ((0, 0, 0),), 1 / 8, "separated", 7 / 8, mesh_kind="u")
WALL = Case("notch-wall", ((29 / 32, 0, 0),), 1 / 8, "boundary_overlap", -1 / 32,
            mesh_kind="u")
COMMON = (1 / 8, -1 / 16, 3 / 32)
THROUGH = Case("sphere-through", ((-2, 0, 0),), 1 / 8, "separated", 7 / 8,
               mesh_step=COMMON, round_step=(33 / 8, -1 / 16, 3 / 32), first_t=F(7, 32))
THROUGH_SHAFT = Case("shaft-through", ((-2, -1 / 2, 0), (-2, 1 / 2, 0)), 1 / 8,
                     "separated", 7 / 8, mesh_step=COMMON,
                     round_step=(33 / 8, -1 / 16, 3 / 32), first_t=F(7, 32))
CLEAR_SWEEP = Case("clear-sweep", ((-2, 73 / 64, 0),), 1 / 8, "separated", None,
                   round_step=(4, 0, 0))
STATIC_CASES = (FACE, EDGE, SHAFT, TIP, FINITE_MISS, CROSSING, INSIDE, CONTAINS, NOTCH, WALL)
SWEEP_CASES = (THROUGH, THROUGH_SHAFT, CLEAR_SWEEP)


def subdivided_box(divisions=3):
    """Split every authored face into n^2 triangles with exact shared edge nodes."""
    vertices, triangles = box((-1.0,) * 3, (1.0,) * 3)
    points, ids, faces = [], {}, []

    def node(a, b, c, i, j):
        point = tuple(F(a[k]) + F(i, divisions) * (F(b[k]) - F(a[k]))
                      + F(j, divisions) * (F(c[k]) - F(a[k])) for k in range(3))
        if point not in ids:
            ids[point] = len(points)
            points.append(tuple(float(x) for x in point))
        return ids[point]

    for ia, ib, ic in triangles:
        a, b, c = (vertices[k] for k in (ia, ib, ic))
        for i in range(divisions):
            for j in range(divisions - i):
                p, q, r = (node(a, b, c, *ij) for ij in ((i, j), (i + 1, j), (i, j + 1)))
                faces.append((p, q, r))
                if i + j < divisions - 1:
                    faces.append((q, node(a, b, c, i + 1, j + 1), r))
    return tuple(points), tuple(faces)


def build(case, *, dense=False, world=0, reverse=False, reorder=False, variant="identity"):
    vertices, faces = u_mesh() if case.mesh_kind == "u" else (
        subdivided_box() if dense else box((-1.0,) * 3, (1.0,) * 3)
    )
    closed_oriented_reference(vertices, faces)
    matrices = {
        "identity": np.eye(3),
        "rigid": np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0)), dtype=float),
        "mirror": np.diag((-1.0, 1.0, 1.0)),
    }
    matrix = matrices[variant]
    offset = np.zeros(3) if variant == "identity" else np.array((9 / 4, -3 / 2, 3 / 8))
    if variant == "mirror":
        faces = tuple((a, c, b) for a, b, c in faces)
    ids = tuple(101 + 7 * i for i in range(len(vertices)))
    mesh_ids = ids[::-1] if reorder else ids
    mk, rk = ph.CollisionKey(world, 23 if reverse else 11, 0), ph.CollisionKey(
        world, 11 if reverse else 23, 0
    )
    mesh = ph.CollisionShape(mk, "mesh", mesh_ids,
        triangles=tuple(tuple(ids[i] for i in f) for f in faces))
    round_ids = (11,) if len(case.axis) == 1 else ((29, 11) if reorder else (11, 29))
    round_shape = ph.CollisionShape(rk, "sphere" if len(case.axis) == 1 else "capsule",
                                    round_ids, radius=case.radius)
    xyz = np.asarray(vertices, dtype=float) @ matrix.T + offset
    axis = np.asarray(case.axis, dtype=float) @ matrix.T + offset
    if reorder:
        xyz = xyz[::-1]
        if len(axis) == 2:
            axis = axis[::-1]
    paths = {}
    for shape, start, step in ((mesh, xyz, case.mesh_step), (round_shape, axis, case.round_step)):
        delta = matrix @ step
        paths[shape.key] = ph.CollisionMotion(start, start + delta if any(step) else None)
    return ph.CollisionModel((round_shape, mesh)), paths, mesh, round_shape


def at(path, time=0):
    assert path.turn is None, "this oracle is deliberately affine only"
    return path.start if path.end is None else path.start + float(time) * (path.end - path.start)


def vertex_distances(vertices, start, end):
    """Exact distances of stored vertices to a finite segment, including a point."""
    a, b = (tuple(F(float(v)) for v in point) for point in (start, end))
    delta = tuple(y - x for x, y in zip(a, b, strict=True))
    square = sum(x * x for x in delta)
    result = {}
    for node, point in vertices.items():
        p = tuple(F(float(v)) for v in point)
        along = sum((x - y) * d for x, y, d in zip(p, a, delta, strict=True)) / square if square else F(0)
        along = min(F(1), max(F(0), along))
        result[node] = sum((x - y - along * d) ** 2 for x, y, d in zip(p, a, delta, strict=True))
    return result


def face_distances(mesh, round_shape, paths, time=0):
    points = dict(zip(mesh.vertex_ids, at(paths[mesh.key], time), strict=True))
    axis = at(paths[round_shape.key], time)
    ends = (axis[0], axis[-1])
    return tuple(segment_triangle_reference(ends, [points[i] for i in face])[0]
                 for face in mesh.triangles)


def required_faces(mesh, round_shape, paths, time=0):
    squares = face_distances(mesh, round_shape, paths, time)
    best, radius2 = min(squares), F(round_shape.radius) ** 2
    return tuple(i for i, value in enumerate(squares) if value == best or value <= radius2)


def five_fields(pair):
    return pair.status, pair.reason, pair.witness, pair.sweep, pair.mesh_round


def check_materials(pair, pose, mesh, round_shape, paths, time=0):
    """Rebuild both physical skins from original material IDs at the same sample."""
    supports = (*pose.supports, *pose.boundary)
    assert supports
    for support in supports:
        assert support.face in mesh.triangles
        w = support.witness
        points = (w.first_point, w.second_point)
        weights = (w.first_weights, w.second_weights)
        offsets = (w.first_offset, w.second_offset)
        if pair.first != mesh.key:
            points, weights, offsets = points[::-1], weights[::-1], offsets[::-1]
        for index, shape in enumerate((mesh, round_shape)):
            positions = dict(zip(shape.vertex_ids, at(paths[shape.key], time), strict=True))
            assert abs(sum(value for _, value in weights[index]) - 1) <= 1e-12
            assert all(value >= 0 and node in positions for node, value in weights[index])
            material = sum(value * positions[node] for node, value in weights[index])
            np.testing.assert_allclose(material + offsets[index], points[index], rtol=0, atol=2e-9)
        np.testing.assert_array_equal(offsets[0], (0, 0, 0))
        assert abs(np.linalg.norm(offsets[1]) - round_shape.radius) <= 2e-9
        axis = at(paths[round_shape.key], time)
        v = axis[-1] - axis[0]
        s = float(np.clip((np.asarray(points[1]) - axis[0]) @ v / (v @ v), 0, 1)) if np.any(v) else 0
        assert abs(np.linalg.norm(np.asarray(points[1]) - axis[0] - s * v) - round_shape.radius) <= 2e-9
        if w.normal is not None:
            assert abs(np.linalg.norm(w.normal) - 1) <= 2e-9
            assert abs((np.asarray(w.second_point) - w.first_point) @ w.normal - w.gap_m) <= 2e-9
        for endpoint in w.capsule_axis_span or ():
            assert 0 <= endpoint.axis_parameter <= 1
            np.testing.assert_allclose(endpoint.axis_point,
                axis[0] + endpoint.axis_parameter * (axis[-1] - axis[0]), rtol=0, atol=2e-9)


def record_facets(workspace, monkeypatch):
    """Observe the real tree call without substituting its traversal or indices."""
    calls = []
    original = workspace._trees.facets

    def observed(pending, key, leaves, queries, driver):
        rows = original(pending, key, leaves, queries, driver)
        calls.append(dict(key=key, leaves=leaves.copy(), queries=queries.copy(),
                          rows=tuple(tuple(row) for row in rows), driver=driver))
        return rows

    monkeypatch.setattr(workspace._trees, "facets", observed)
    return calls


def prepare_selector(case, callback, *, dense=False, variant="identity", reverse=False,
                     reorder=False):
    """New interface is imported only when its protocol tests are selected."""
    from phymeo.geometry._closed_mesh import ClosedMesh
    from phymeo.geometry._mesh_round_candidates import MeshRoundFaceSelector

    model, paths, mesh, round_shape = build(
        case, dense=dense, variant=variant, reverse=reverse, reorder=reorder
    )
    prepared = ClosedMesh(mesh, paths[mesh.key])
    selector = MeshRoundFaceSelector(mesh, prepared, callback)
    axis = paths[round_shape.key].start
    distances = vertex_distances(prepared.points, axis[0], axis[-1])
    return model, paths, mesh, round_shape, prepared, selector, distances


class BoxOracle:
    """Protocol stand-in only; public/device tests always execute the actual BVH."""

    def __init__(self, *, scrambled=False):
        self.calls = []
        self.scrambled = scrambled

    def __call__(self, binding, leaves, queries):
        assert leaves.dtype == queries.dtype == np.float32
        assert leaves.ndim == 2 and leaves.shape[1] == 6 and queries.shape == (1, 6)
        assert np.isfinite(leaves).all() and np.isfinite(queries).all()
        assert np.all(leaves[:, :3] <= leaves[:, 3:])
        assert np.all(queries[:, :3] <= queries[:, 3:])
        assert len(binding.faces) == len(leaves)
        hits = [i for i, leaf in enumerate(leaves)
                if np.all(leaf[:3] <= queries[0, 3:]) and np.all(queries[0, :3] <= leaf[3:])]
        self.calls.append((binding, leaves.copy(), queries.copy(), tuple(hits)))
        if self.scrambled:
            hits = hits[::-1] + hits[:1]
        return [hits]
