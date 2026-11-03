"""Frozen M3 geometry and independent material/occupancy checks for the G4 world batch."""

import json
import sys
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

import numpy as np

import phymeo as ph

# Use the original M3 constructors, including their material IDs and three moving frames.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collision"))
from case_catalog import geometry_for  # noqa: E402
from spatial_index_cases import mesh_scene, sheet  # noqa: E402
from test_spatial_broad_phase import balls  # noqa: E402

NAMES = ("sparse65", "dense34", "sheet18", "sheet72", "mesh28", "mesh112")
WORLD_IDS = (37, 5, 92, 16)
TRUE = {"hit", "touching", "overlap"}


def frozen(name):
    if name in ("sparse65", "dense34"):
        m, p = balls(65 if name == "sparse65" else 34, name == "dense34")
        return m, (p,)
    if name.startswith("sheet"):
        return sheet(3 if name == "sheet18" else 6)
    if name.startswith("mesh"):
        return mesh_scene(name == "mesh112")
    raise ValueError("unknown frozen workload")


def copies(model, frames, ids):
    models, paths = [], [{} for _ in frames]
    for key in ids:
        keys = {s.key: replace(s.key, world=key) for s in model.shapes}
        shapes = tuple(replace(s, key=keys[s.key]) for s in model.shapes)
        excludes = tuple((keys[a], keys[b]) for a, b in model.excluded)
        models.append((key, ph.CollisionModel(shapes, excludes)))
        for dest, frame in zip(paths, frames, strict=True):
            dest[key] = {keys[k]: v for k, v in frame.items()}
    return models, paths


def digest(observed, world):
    record = asdict(replace(observed, backend="canonical"))

    def local(obj):
        if isinstance(obj, dict):
            if "world" in obj:
                assert obj["world"] == world, "public row crosses worlds"
                obj["world"] = 0
            for child in obj.values():
                local(child)
        elif isinstance(obj, (list, tuple)):
            for child in obj:
                local(child)

    local(record)
    return sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def pose(path, t):
    if path.turn is not None:
        assert path.turn.angle == 0
        return path.start + t * (np.asarray(path.turn.pivot_end) - path.turn.pivot_start)
    if path.end is not None:
        return path.start + t * (path.end - path.start)
    return path.start


def material(shape, path, weights, t):
    assert weights
    ids, ws = zip(*weights, strict=True)
    assert len(set(ids)) == len(ids) and set(ids) <= set(shape.vertex_ids)
    assert min(ws) >= -2e-9 and abs(sum(ws) - 1.0) < 2e-8
    xyz = pose(path, t)
    return np.asarray(ws) @ xyz[[shape.vertex_ids.index(i) for i in ids]]


def check_witness(witness, t, first, second, paths):
    if witness is None:
        return 0
    for side, shape, sign in (("first", first, 1.0), ("second", second, -1.0)):
        point = getattr(witness, side + "_point")
        base = material(shape, paths[shape.key], getattr(witness, side + "_weights"), t)
        if point is None:
            assert witness.normal is None
            continue
        offset = getattr(witness, side + "_offset")
        if offset is not None:
            base = base + offset
        elif shape.kind == "sphere":
            base = base + sign * shape.radius * np.asarray(witness.normal)
        np.testing.assert_allclose(base, point, atol=2e-8, rtol=1e-10)
    return 1


def xy_distance2(center, tri):
    edge = np.roll(tri, -1, axis=0) - tri
    delta = center - tri
    cross = edge[:, 0] * delta[:, 1] - edge[:, 1] * delta[:, 0]
    if np.all(cross >= 0) or np.all(cross <= 0):
        return 0.0
    t = np.clip((delta * edge).sum(axis=1) / (edge * edge).sum(axis=1), 0.0, 1.0)
    return float(((delta - edge * t[:, None]) ** 2).sum(axis=1).min())


def first_in_u(mesh, hull, frame):
    _, _, g = geometry_for("RR05", "base")
    xl, il, ir, xr = sorted({p[0] for p in g["A"]["ring_xy"]})
    _, y1, y2 = sorted({p[1] for p in g["A"]["ring_xy"]})
    z0, z1 = g["A"]["z"]
    v0, v1 = pose(frame[hull.key], 0), pose(frame[hull.key], 1)
    assert np.all((v0[:, 1] > y1) & (v0[:, 1] < y2) & (v0[:, 2] > z0) & (v0[:, 2] < z1))
    dx = v1[0, 0] - v0[0, 0]
    assert dx > 0 and np.allclose(v1 - v0, [dx, 0, 0])
    times = []
    for low, high in ((xl, il), (ir, xr)):
        enter = max(0.0, (low - v0[:, 0].max()) / dx)
        leave = min(1.0, (high - v0[:, 0].min()) / dx)
        if enter <= leave:
            times.append(enter)
    return min(times) if times else None


def check_geometry(name, model, paths, result, expected_candidates):
    assert result.status == "ok"
    shapes = {s.key: s for s in model.shapes}
    assert (
        len(result.bounds) == len(shapes)
        and len(result.pairs) == len(shapes) * (len(shapes) - 1) // 2
    )
    assert (
        result.required_candidates == sum(p.candidate for p in result.pairs) == expected_candidates
    )
    for bounds in result.bounds:
        shape = shapes[bounds.key]
        vertices = np.vstack((pose(paths[shape.key], 0.0), pose(paths[shape.key], 1.0)))
        lo, hi = vertices.min(axis=0) - shape.radius, vertices.max(axis=0) + shape.radius
        lower, upper = np.asarray(bounds.lower), np.asarray(bounds.upper)
        # The original paths have no rotations. Roundoff in our FP64 translated
        # endpoint may be one FP64 ulp away from the exact stored displacement.
        slack = 4 * np.finfo(float).eps * np.maximum(1, np.maximum(abs(lo), abs(hi)))
        assert np.all(lower <= lo + slack) and np.all(upper >= hi - slack)
        np.testing.assert_allclose(lower, lo, rtol=2e-7, atol=1e-11)
        np.testing.assert_allclose(upper, hi, rtol=2e-7, atol=1e-11)
    # The sphere sweeps vertically across the full sheet while every face vertex keeps
    # fixed xy. The radius-offset triangle in xy decides which faces can physically touch.
    sheet_hits = set()
    if name.startswith("sheet"):
        ball = next(s for s in model.shapes if s.kind == "sphere")
        center = paths[ball.key].start[0, :2]
        for tri in (s for s in model.shapes if s.kind == "triangle"):
            p = paths[tri.key]
            np.testing.assert_array_equal(p.start[:, :2], p.end[:, :2])
            assert p.start[:, 2].max() + ball.radius < paths[ball.key].start[0, 2]
            assert p.end[:, 2].min() - ball.radius > paths[ball.key].end[0, 2]
            if xy_distance2(center, p.start[:, :2]) <= ball.radius**2:
                sheet_hits.add(tri.key)
    marks, material_pairs, points, examples = [], 0, 0, []
    for row in result.pairs:
        a, b = shapes[row.first], shapes[row.second]
        assert row.first.world == row.second.world and row.status not in (
            "invalid_geometry",
            "unresolved",
            "unsupported",
        )
        hit = row.status in TRUE
        if name in ("sparse65", "dense34"):
            d = np.linalg.norm(paths[b.key].start[0] - paths[a.key].start[0])
            assert hit == (d <= a.radius + b.radius)
            if hit:
                assert (
                    row.witness is not None
                    and abs(row.witness.gap_m - (d - a.radius - b.radius)) < 2e-9
                )
        elif name.startswith("sheet"):
            expected = (a.key in sheet_hits and b.kind == "sphere") or (
                b.key in sheet_hits and a.kind == "sphere"
            )
            assert hit == expected
        elif name.startswith("mesh"):
            t = first_in_u(a, b, paths)
            assert row.mesh is not None and hit == (t is not None)
            if t is not None:
                num, den = row.mesh.first_ratio
                assert abs(num / den - t) < 2e-12
        if not row.candidate:
            continue
        checked = check_witness(row.witness, 1.0, a, b, paths)
        if row.sweep and row.sweep.witness is not None:
            assert row.sweep.witness_time is not None
            checked += check_witness(row.sweep.witness, row.sweep.witness_time, a, b, paths)
            if name.startswith("sheet") and hit:
                assert abs(row.sweep.witness.gap_m) < 2e-8
                np.testing.assert_allclose(
                    row.sweep.witness.first_point, row.sweep.witness.second_point, atol=2e-8, rtol=0
                )
        if row.mesh is not None:
            poses = [(0.0, row.mesh.start), (1.0, row.mesh.end)]
            if row.mesh.entry is not None:
                numerator, denominator = row.mesh.first_ratio
                poses.append((numerator / denominator, row.mesh.entry))
            for time, sample in poses:
                checked += check_witness(sample.witness, time, a, b, paths)
                for point in sample.contacts:
                    for shape, weights in ((a, point.first_weights), (b, point.second_weights)):
                        np.testing.assert_allclose(
                            material(shape, paths[shape.key], weights, time),
                            point.point,
                            atol=2e-8,
                            rtol=1e-10,
                        )
        material_pairs += int(checked > 0)
        points += checked
        if hit:
            marks.append([a.key.body, a.key.shape, b.key.body, b.key.shape])
        if len(examples) < 3:
            examples.append(
                dict(
                    local_pair=[[a.key.body, a.key.shape], [b.key.body, b.key.shape]],
                    status=row.status,
                    endpoint_gap_m=None if row.witness is None else row.witness.gap_m,
                    endpoint=None if row.witness is None else asdict(row.witness),
                    first_t=None if not row.sweep else row.sweep.first_t,
                    entry_witness=None
                    if not row.sweep or not row.sweep.witness
                    else asdict(row.sweep.witness),
                    mesh_entry_boundary_m=None
                    if not row.mesh or not row.mesh.entry
                    else row.mesh.entry.boundary_gap_m,
                    mesh_relations=None
                    if not row.mesh
                    else [row.mesh.start.relation, row.mesh.end.relation],
                )
            )
    return dict(
        pair_rows=len(result.pairs),
        candidates=result.required_candidates,
        hits=len(marks),
        true_pair_sha256=sha256(json.dumps(marks, separators=(",", ":")).encode()).hexdigest(),
        candidate_pairs_with_witness=material_pairs,
        witnesses_checked=points,
        examples=examples,
    )
