"""Predeclared COL2 literals and independent exact active-face oracle.

No Phymeo/TYX imports. The oracle solves all active faces of [0,1] x a
triangle simplex using Fraction normal equations, not the production feature
enumerator. Gates fixed before any execution: distance1e-4 L, witness2e-4 L,
normal2e-3; no absolute world-coordinate contribution to L.
"""

from fractions import Fraction as Q
from itertools import combinations

import numpy as np

U = 2.0**-24
CONDITION = 32 * U
SOURCE_ERROR = 8 * U
TIE_ERROR = 16 * U
NORMAL_FLOOR = 32 * U  # u=2^-24, half FP32 machine epsilon.
NAMES = (
    "face_above",
    "face_below",
    "edge",
    "vertex",
    "beyond_edge",
    "zero_axis",
    "parallel_face",
    "oblique_endpoint",
    "edge_edge",
    "extension_miss",
    "through_face",
    "parallel_edge",
    "rotation",
    "large_translation",
    "reverse_winding",
    "shared_edge",
    "crease",
    "deformed_snapshot",
    "bvh_pruned",
    "concave_faces",
    "near_parallel_clear",
    "near_parallel_conditioning",
    "zero_area",
    "distinct_segment_owners",
)


def make_case(name):
    if name not in NAMES:
        raise KeyError(name)
    vertices = np.array(((0, 0, 0), (2, 0, 0), (0, 2, 0)), dtype=np.float64)
    faces = np.array(((0, 1, 2),), dtype=np.int64)
    segments = np.array((((0.5, 0.5, 0.25), (0.5, 0.5, 0.25)),), dtype=np.float64)
    origin = np.zeros(3)
    points = {
        "face_below": (0.5, 0.5, -0.25),
        "edge": (1, -0.25, 0.25),
        "vertex": (-0.25, -0.25, 0.25),
        "beyond_edge": (2, 2, 0.25),
        "zero_axis": (0.25, 0.75, 0.125),
    }
    axes = {
        "parallel_face": ((0.25, 0.5, 0.25), (1, 0.5, 0.25)),
        "oblique_endpoint": ((0.5, 0.5, 0.25), (0.75, 0.5, 0.75)),
        "edge_edge": ((1, -0.25, -0.25), (1, -0.25, 0.25)),
        "extension_miss": ((0.5, 0.5, 0.25), (0.5, 0.5, 0.5)),
        "through_face": ((0.5, 0.5, -0.25), (0.5, 0.5, 0.25)),
        "parallel_edge": ((-0.5, -0.25, 0.25), (2.5, -0.25, 0.25)),
        "near_parallel_clear": ((0.25, 0.25, 0.25), (1.25, 0.3125, 0.25)),
        "near_parallel_conditioning": ((0.25, 0.25, 0.25), (1.25, 0.25 + 2**-15, 0.25)),
    }
    if name in points:
        segments[:] = points[name]
    if name in axes:
        segments[0] = axes[name]
    if name == "rotation":
        rotation = np.array(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        vertices = vertices @ rotation.T
        segments = segments @ rotation.T
    if name == "large_translation":
        origin = np.array((2**24, -(2**24), 2**24), dtype=np.float64)
        vertices += origin
        segments += origin
        origin = np.zeros(3)  # Pair frame must handle a distant caller origin too.
    if name == "reverse_winding":
        faces = np.array(((2, 1, 0),), dtype=np.int64)
    if name == "shared_edge":
        vertices = np.vstack((vertices, (2.0, 2.0, 0.0)))
        faces = np.array(((0, 1, 2), (3, 2, 1)), dtype=np.int64)
        segments[:] = (1.0, 1.0, 0.25)
    if name == "crease":
        vertices = np.vstack((vertices, (0.0, 0.0, 2.0)))
        faces = np.array(((0, 1, 2), (0, 3, 1)), dtype=np.int64)
        segments[:] = (1.0, -0.25, 0.25)
    if name == "deformed_snapshot":
        vertices[2, 2] = 0.25
        segments[:] = (0.5, 0.5, 0.5)
    if name == "bvh_pruned":
        vertices = np.vstack(
            [vertices, np.array(((2.0, 2.0, 0.0), (0.0, 2.0, 0.0), (2.0, 0.0, 0.0)))]
            + [vertices + (4 * i, 0, 0) for i in range(1, 7)]
        )
        faces = np.arange(24, dtype=np.int64).reshape(-1, 3)
    if name == "concave_faces":
        vertices = np.array(
            ((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2), (2, 0, 2), (2, 2, 0), (2, 2, 2)),
            dtype=np.float64,
        )
        faces = np.array(((0, 1, 2), (0, 3, 1), (1, 4, 5)), dtype=np.int64)
        segments[:] = (1.0, 0.25, 0.25)
    if name == "zero_area":
        vertices[2] = (1.0, 0.0, 0.0)
    if name == "distinct_segment_owners":
        segments = np.concatenate((segments, segments), axis=0)
    vertex_ids = np.arange(len(vertices), dtype=np.int64) * 7 + 101
    if name == "reverse_winding":
        vertices = vertices[[2, 0, 1]].copy()
        vertex_ids = vertex_ids[[2, 0, 1]].copy()
        faces = np.array(((0, 2, 1),), dtype=np.int64)
    facet_ids = np.arange(len(faces), dtype=np.int64) * 11 + 307
    segment_ids = np.arange(len(segments), dtype=np.int64) * 13 + 509
    candidates = np.array(
        [(int(s), int(f)) for s in segment_ids for f in facet_ids], dtype=np.int64
    )
    if name == "bvh_pruned":
        candidates = candidates[:2].copy()
        segments[:] = (1.0, 1.0, 0.25)
    return dict(
        mesh_key="col2-fixed",
        revision=3,
        vertex_ids=vertex_ids,
        facet_ids=facet_ids,
        faces=faces,
        vertices=vertices,
        segment_ids=segment_ids,
        segments=segments,
        radii=np.full(len(segments), 0.125, dtype=np.float64),
        candidates=candidates,
        origin_m=origin,
        snapshot_id=name,
    )


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def linear(matrix, rhs):
    n = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for k in range(n):
        pivot = next((i for i in range(k, n) if rows[i][k]), None)
        if pivot is None:
            return None
        rows[k], rows[pivot] = rows[pivot], rows[k]
        value = rows[k][k]
        rows[k] = [x / value for x in rows[k]]
        for i in range(n):
            if i != k:
                value = rows[i][k]
                rows[i] = [a - value * b for a, b in zip(rows[i], rows[k], strict=True)]
    return tuple(rows[i][-1] for i in range(n))


def oracle(triangle, segment):
    t = [tuple(Q(float(x)) for x in row) for row in triangle]
    p = [tuple(Q(float(x)) for x in row) for row in segment]
    axis = sub(p[1], p[0])
    records = []
    for count in (1, 2, 3):
        for ids in combinations(range(3), count):
            base = t[ids[0]]
            edges = [sub(t[i], base) for i in ids[1:]]
            for state in (0, 1, 2):
                start = p[state] if state < 2 else p[0]
                columns = ([axis] if state == 2 else []) + [tuple(-x for x in e) for e in edges]
                offset = sub(start, base)
                rhs = [-dot(c, offset) for c in columns]
                gram = [[dot(a, b) for b in columns] for a in columns]
                solution = linear(gram, rhs) if columns else ()
                if solution is None:
                    continue
                s = solution[0] if state == 2 else Q(state)
                ws = solution[1:] if state == 2 else solution
                bary = [Q(0)] * 3
                bary[ids[0]] = 1 - sum(ws)
                for i, value in zip(ids[1:], ws, strict=True):
                    bary[i] = value
                if not (0 <= s <= 1 and min(bary) >= 0):
                    continue
                a = tuple(p[0][k] + s * axis[k] for k in range(3))
                b = tuple(sum(bary[i] * t[i][k] for i in range(3)) for k in range(3))
                d = sub(b, a)
                records.append((dot(d, d), s, tuple(bary), a, b))
    assert records
    best = min(x[0] for x in records)
    tied = [x for x in records if x[0] == best]
    return dict(
        distance2=best, axis_span=(min(x[1] for x in tied), max(x[1] for x in tied)), witnesses=tied
    )


def local_scale(triangle, segment, radius):
    return (
        max(float(np.linalg.norm(triangle[(i + 1) % 3] - triangle[i])) for i in range(3))
        if radius is None
        else max(
            *(float(np.linalg.norm(triangle[(i + 1) % 3] - triangle[i])) for i in range(3)),
            float(np.linalg.norm(segment[1] - segment[0])),
            float(radius),
        )
    )
