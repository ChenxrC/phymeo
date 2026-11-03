"""Closed-solid relations and physical sphere/capsule surface pairs through the public pipeline."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from test_mesh_mesh_contact import box

import phymeo as ph

REF = Path(__file__).resolve().parents[1] / "reference_newton" / "rr"
CASES = json.loads((REF / "mesh_round_cases.json").read_text("utf-8"))["poses"]
CYCLE = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])


def build(case, reverse=False, transformed=False, travel=None):
    if case["mesh"] == "box":
        vertices, triangles = box((-1,) * 3, (1,) * 3, flip=reverse)
    else:
        vertices, triangles = runpy.run_path(str(REF / "baseline.py"))["_u_geometry"](
            [1, 1, 1], reverse
        )
        triangles = triangles.reshape(-1, 3)
    ids = tuple(17 + 5 * i for i in range(len(vertices)))
    mk, rk = (ph.CollisionKey(0, 2 - int(reverse), 1), ph.CollisionKey(0, 1 + int(reverse), 1))
    mesh = ph.CollisionShape(
        mk, "mesh", ids, triangles=tuple(tuple(ids[i] for i in f) for f in triangles)
    )
    axis = case["axis_m"]
    shape = ph.CollisionShape(
        rk,
        "sphere" if len(axis) == 1 else "capsule",
        (17,) if len(axis) == 1 else (92, 17),
        radius=case["radius_m"],
    )
    matrix = CYCLE if transformed else np.eye(3)
    origin = np.array([2.25, -1.5, 0.375]) if transformed else np.zeros(3)
    paths = {}
    for i, (q, xyz) in enumerate(((mesh, vertices), (shape, axis))):
        coords = np.asarray(xyz) @ matrix.T + origin
        end = None if travel is None else coords + matrix @ travel[i]
        paths[q.key] = ph.CollisionMotion(coords, end)
    return ph.CollisionModel([shape, mesh]), paths, mesh, shape, matrix, origin


def query(model, paths):
    with ph.CollisionWorkspace(model) as workspace:
        (pair,) = ph.detect(model, paths, ph.CollisionPolicy(margin=1), workspace).pairs
        return pair


def verify_points(support, pair, mesh, shape, paths, time=0):
    w = support.witness
    mesh_first = pair.first == mesh.key
    assert support.face in mesh.triangles
    points, weights, offsets = (
        (w.first_point, w.second_point),
        (w.first_weights, w.second_weights),
        (w.first_offset, w.second_offset),
    )
    if not mesh_first:
        points, weights, offsets = points[::-1], weights[::-1], offsets[::-1]
    for i, body in enumerate((mesh, shape)):
        path = paths[body.key]
        nodes = path.start if path.end is None else path.start + time * (path.end - path.start)
        mapping = dict(zip(body.vertex_ids, nodes, strict=True))
        assert abs(sum(v for _, v in weights[i]) - 1) < 1e-12 and min(v for _, v in weights[i]) >= 0
        assert {k for k, v in weights[i] if v} <= (
            set(support.face) if i == 0 else set(body.vertex_ids)
        )
        base = sum(v * mapping[k] for k, v in weights[i])
        np.testing.assert_allclose(base + offsets[i], points[i], atol=1e-9, rtol=0)
    assert np.linalg.norm(offsets[0]) == 0
    assert np.linalg.norm(offsets[1]) == pytest.approx(shape.radius, abs=1e-9)
    # Independent projection on the closed finite axis checks that the radial
    # point is on the actual capsule, including at a cap and a zero-length axis.
    axis_vec = nodes[-1] - nodes[0]
    u = (
        np.clip((np.asarray(points[1]) - nodes[0]) @ axis_vec / (axis_vec @ axis_vec), 0, 1)
        if np.any(axis_vec)
        else 0
    )
    assert np.linalg.norm(points[1] - (nodes[0] + u * axis_vec)) == pytest.approx(
        shape.radius, abs=1e-8
    )
    if w.normal is not None:
        assert np.linalg.norm(w.normal) == pytest.approx(1)
        assert np.subtract(w.second_point, w.first_point) @ w.normal == pytest.approx(
            w.gap_m, abs=1e-8
        )
    return points, weights


@pytest.mark.parametrize("reverse,transformed", [(False, False), (True, True)])
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_closed_mesh_round_static_relation_and_real_materials(case, reverse, transformed, tmp_path):
    model, paths, mesh, shape, matrix, origin = build(case, reverse, transformed)
    pair = query(model, paths)
    relation = case["relation"]
    status = {"separated": "separated", "touching": "touching"}.get(relation, "overlap")
    assert pair.status == status and pair.sweep is None
    assert pair.mesh_round is not None
    pose = pair.mesh_round.end
    assert pose.relation == relation and pose is pair.mesh_round.start
    assert pose.boundary_gap_m == pytest.approx(case["boundary_gap_m"], abs=1e-9)
    assert pose.axis_gap_m == pytest.approx(case["axis_gap_m"], abs=1e-9)
    assert bool(pose.boundary) == (case["boundary_gap_m"] == 0)
    assert pose.supports and pose.witness == pair.witness
    for s in pose.boundary:
        pts, _ = verify_points(s, pair, mesh, shape, paths)
        np.testing.assert_allclose(pts[0], pts[1], atol=1e-8, rtol=0)
    world_normals = []
    cap_sites = []
    for s in pose.supports:
        pts, weights = verify_points(s, pair, mesh, shape, paths)
        if "mesh_point_m" in case:
            np.testing.assert_allclose(
                pts[0], origin + matrix @ case["mesh_point_m"], atol=1e-9, rtol=0
            )
        if "normal_mesh_to_round" in case:
            normal = np.asarray(s.witness.normal) * (1 if pair.first == mesh.key else -1)
            np.testing.assert_allclose(
                normal, matrix @ case["normal_mesh_to_round"], atol=1e-9, rtol=0
            )
        if s.witness.normal is not None:
            world_normals.append(s.witness.normal)
        if s.witness.capsule_axis_span is not None:
            cap_sites.extend(p.axis_parameter for p in s.witness.capsule_axis_span)
    if "contains" in relation:
        assert pair.witness.gap_m == pytest.approx(-pose.boundary_gap_m, abs=1e-9)
    elif case["id"] != "capsule-pierces":
        assert pair.witness.gap_m == pytest.approx(pose.axis_gap_m, abs=1e-9)
    if case["id"] == "u-notch":
        assert pair.witness.normal is None and len(set(world_normals)) == 2
    if case["id"] == "capsule-shaft":
        assert min(cap_sites) == 0 and max(cap_sites) == 1
    if case["id"] == "capsule-tip":
        assert min(cap_sites) == max(cap_sites) == 0
    if case["id"] == "capsule-pierces":
        assert pose.axis_location == "on_boundary" and pose.boundary
    (tmp_path / "mesh-round-pose.json").write_text(
        json.dumps(
            dict(
                id=case["id"],
                reverse=reverse,
                relation=pose.relation,
                boundary_gap_m=pose.boundary_gap_m,
                axis_gap_m=pose.axis_gap_m,
                supports=len(pose.supports),
                boundary_faces=len(pose.boundary),
                witnesses=[
                    dict(
                        mesh_face=s.face,
                        first=s.witness.first_point,
                        second=s.witness.second_point,
                        first_weights=s.witness.first_weights,
                        second_weights=s.witness.second_weights,
                        first_offset=s.witness.first_offset,
                        second_offset=s.witness.second_offset,
                    )
                    for s in pose.supports
                ],
            ),
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def test_turning_mesh_sphere_preserves_overlap_and_explicit_pair_filter():
    case = next(c for c in CASES if c["id"] == "sphere-face")
    model, paths, mesh, shape, _, _ = build(case)
    arc = ph.CollisionMotion(
        paths[mesh.key].start, turn=ph.RigidTurn((0, 0, 0), (0, 0, 0), (0, 0, 1), 0.5)
    )
    pair = query(model, {**paths, mesh.key: arc})
    assert pair.status == "hit" and pair.sweep.first_t == (0.0, 0.0)
    assert pair.mesh_round.start.relation == case["relation"]
    assert pair.mesh_round.start.axis_gap_m == case["axis_gap_m"]
    excluded = ph.CollisionModel(model.shapes, excluded=((mesh.key, shape.key),))
    for motion in (paths, {**paths, mesh.key: arc}):
        row = query(excluded, motion)
        assert row.status == "filtered" and row.mesh_round is None


def test_material_node_reordering_preserves_the_same_physical_supports():
    case = next(c for c in CASES if c["id"] == "capsule-shaft")
    model, paths, mesh, shape, _, _ = build(case)
    original = query(model, paths)
    reordered = [
        ph.CollisionShape(mesh.key, "mesh", mesh.vertex_ids[::-1], triangles=mesh.triangles[::-1]),
        ph.CollisionShape(shape.key, "capsule", shape.vertex_ids[::-1], radius=shape.radius),
    ]
    motions = {s.key: ph.CollisionMotion(paths[s.key].start[::-1]) for s in reordered}
    row = query(ph.CollisionModel(reordered), motions)
    assert row.mesh_round.end.relation == original.mesh_round.end.relation
    assert row.mesh_round.end.boundary_gap_m == original.mesh_round.end.boundary_gap_m
    first = {s.face: s.witness for s in original.mesh_round.end.supports}
    last = {s.face: s.witness for s in row.mesh_round.end.supports}
    assert first.keys() == last.keys()
    for face, w in first.items():
        assert (
            w.first_weights == last[face].first_weights
            and w.second_weights == last[face].second_weights
        )
        np.testing.assert_allclose(w.first_point, last[face].first_point, rtol=0, atol=1e-12)
        np.testing.assert_allclose(w.second_offset, last[face].second_offset, rtol=0, atol=1e-12)


@pytest.mark.parametrize("owner", ["mesh", "round"])
def test_internal_tangency_keeps_solid_containment_and_zero_boundary_clearance(owner):
    axes = [[0.75, 0, 0]] if owner == "mesh" else [[0, 0, 0]]
    radius = 0.25 if owner == "mesh" else 0.75
    model, paths, mesh, shape, _, _ = build(dict(mesh="box", axis_m=axes, radius_m=radius))
    if owner == "round":
        # Each box corner is exactly 3/4 m from the sphere center.
        paths[mesh.key] = ph.CollisionMotion(paths[mesh.key].start * [0.25, 0.5, 0.5])
    row = query(model, paths)
    assert row.status == "overlap"
    assert row.mesh_round.end.relation == (
        "mesh_contains_round" if owner == "mesh" else "round_contains_mesh"
    )
    assert row.mesh_round.end.boundary_gap_m == row.witness.gap_m == 0
    assert row.mesh_round.end.boundary
    for s in row.mesh_round.end.supports:
        p, _ = verify_points(s, row, mesh, shape, paths)
        np.testing.assert_allclose(*p, atol=1e-9, rtol=0)
