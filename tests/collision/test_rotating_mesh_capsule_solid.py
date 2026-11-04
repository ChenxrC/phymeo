"""Actual rotating-solid queries against the previously frozen finite-shaft oracles."""

import json
import runpy
from dataclasses import asdict
from math import acos, cos, pi, sin
from pathlib import Path

import numpy as np
import pytest
from test_mesh_mesh_contact import box, detect

import phymeo as ph

REF = Path(__file__).resolve().parents[1] / "reference_newton" / "rr"
CASES = json.loads((REF / "rotating_mesh_capsule_cases.json").read_text("utf8"))
CYCLE = np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0)))


def spin(points, theta):
    c, s = cos(theta), sin(theta)
    return np.asarray(points) @ np.array(((c, s, 0), (-s, c, 0), (0, 0, 1)))


def scene(asset, spine, radius, theta, *, swap=False, moved=False, worlds=(0, 0), drift=(0, 0, 0)):
    local, faces = asset
    ids = tuple(41 + 7 * i for i in range(len(local)))
    mesh = ph.CollisionShape(ph.CollisionKey(worlds[0], int(swap), 0), "mesh", ids,
                             triangles=tuple(tuple(ids[i] for i in f) for f in faces))
    cap = ph.CollisionShape(ph.CollisionKey(worlds[1], int(not swap), 0), "capsule", (41, 93), radius)
    basis = CYCLE if moved else np.eye(3)
    origin = np.array((2.25, -0.75, 0.375)) if moved else np.zeros(3)
    turn = ph.RigidTurn(origin, origin + basis @ drift, basis @ (0, 0, 1), theta)
    paths = {mesh.key: ph.CollisionMotion(np.asarray(local) @ basis.T + origin, turn=turn),
             cap.key: ph.CollisionMotion(np.asarray(spine) @ basis.T + origin)}
    return ph.CollisionModel([mesh, cap]), paths, mesh, cap, basis


def materials(row, pose, mesh, cap, paths, basis, t):
    turn = paths[mesh.key].turn
    pivot = np.asarray(turn.pivot_start)
    travel = np.subtract(turn.pivot_end, pivot)
    local = (paths[mesh.key].start - pivot) @ basis
    xyz = spin(local, t * turn.angle) @ basis.T + pivot + t * travel
    lookup = dict(zip(mesh.vertex_ids, xyz, strict=True))
    axis = paths[cap.key].start
    if paths[cap.key].end is not None:
        axis = axis + t * (paths[cap.key].end - axis)
    spans = []
    for support in (*pose.supports, *pose.boundary):
        w = support.witness
        assert support.face in mesh.triangles
        mp, cp, mw, cw, radial = (w.first_point, w.second_point, w.first_weights, w.second_weights, w.second_offset)
        if row.first != mesh.key:
            mp, cp, mw, cw, radial = w.second_point, w.first_point, w.second_weights, w.first_weights, w.first_offset
        assert set(i for i, x in mw if x) <= set(support.face)
        np.testing.assert_allclose(sum(x * lookup[i] for i, x in mw), mp, atol=2e-9, rtol=0)
        anchor = sum(x * axis[cap.vertex_ids.index(i)] for i, x in cw)
        np.testing.assert_allclose(anchor + radial, cp, atol=2e-9, rtol=0)
        assert np.linalg.norm(radial) == pytest.approx(cap.radius, abs=2e-9)
        if w.normal is not None:
            assert np.dot(np.subtract(w.second_point, w.first_point), w.normal) == pytest.approx(w.gap_m, abs=2e-9)
        if w.capsule_axis_span is None:
            continue
        low, high = w.capsule_axis_span
        for c in (low, high):
            u = c.axis_parameter
            np.testing.assert_allclose(c.axis_point, (1-u)*axis[0]+u*axis[1], atol=2e-9, rtol=0)
            np.testing.assert_allclose(sum(v*lookup[i] for i,v in c.face_weights), c.face_point, atol=2e-9, rtol=0)
        spans.append((low.axis_parameter, high.axis_parameter))
    return sorted(spans)


@pytest.mark.parametrize("case", CASES["cases"], ids=[c["id"] for c in CASES["cases"]])
def test_public_rotating_capsule_against_frozen_midshaft_and_clear_cases(case, tmp_path):
    if case["mesh"] == "bar":
        asset = box(*CASES["bar_bounds_m"])
    else:
        vertices, triangles = runpy.run_path(str(REF / "baseline.py"))["_u_geometry"]([1,1,1], False)
        asset = vertices, triangles.reshape(-1,3)
    spine = [(0,case["axis_y_m"],z) for z in CASES["axis_ends_z_m"]]
    model, paths, mesh, cap, basis = scene(asset,spine,CASES["capsule_radius_m"],case["angle_rad"],
        swap=case["body_order"]=="capsule_first", moved=case["placement"]=="cyclic_translation", worlds=case["worlds"])
    row = detect(model,paths)
    (tmp_path/"rotating-shaft.json").write_text(json.dumps(asdict(row)),encoding="utf8")
    if mesh.key.world != cap.key.world:
        assert row.status=="filtered" and row.reason=="different_world" and not row.candidate
        return
    if CASES["analytic_targets"][case["id"]]["status"]=="clear":
        assert row.candidate and row.status=="separated" and row.sweep.first_t is None
        assert row.mesh_round.start.relation==row.mesh_round.end.relation=="separated"
        return
    assert row.status=="hit",row
    trace=row.mesh_round
    assert trace.start.relation==trace.end.relation=="separated"
    event=trace.first_contact
    root=acos(1/6)/pi
    t=event.witness_time
    assert event.interval[0] <= root <= t <= event.interval[1]
    assert event.interval[1]-event.interval[0] <= 2**-27
    assert abs(case["axis_y_m"])*cos(pi*t)-0.125 <= cap.radius+1e-14
    assert trace.start.axis_gap_m>0 and trace.end.axis_gap_m>0 and event.pose.axis_gap_m<=0
    spans=materials(row,event.pose,mesh,cap,paths,basis,t)
    left,end=spans[0]
    for lo,hi in spans[1:]:
        assert lo <= end + 1e-10
        end=max(end,hi)
    np.testing.assert_allclose([left,end],[3/8,5/8],atol=1e-10,rtol=0)
    for p,m in ((trace.start,0),(trace.end,1)):
        materials(row,p,mesh,cap,paths,basis,m)


def test_oblique_capsule_endcap_on_a_separately_translating_mesh_pivot():
    asset=box((-0.75,-0.75,-0.25),(0.75,0.75,0.25))
    spine=[(0,-0.25,0.625),(0.25,0.25,1.125)]
    model,paths,mesh,cap,basis=scene(asset,spine,0.125,-1.25,drift=(0,0,5/16))
    row=detect(model,paths)
    assert row.status=="hit",row
    event=row.mesh_round.first_contact
    assert event.interval[0] <= 0.8 <= event.witness_time <= event.interval[1]
    spans=materials(row,event.pose,mesh,cap,paths,basis,event.witness_time)
    assert spans and all(abs(s)<1e-10 for pair in spans for s in pair)
    assert event.pose.witness.gap_m <= 0


def test_rotating_mesh_with_separately_rotating_capsule_remains_unavailable():
    asset=box((1,-0.125,-0.25),(2,0.125,0.25))
    model,paths,_,cap,_=scene(asset,[(0,1.5,-1),(0,1.5,1)],0.125,pi)
    paths[cap.key]=ph.CollisionMotion(paths[cap.key].start,turn=ph.RigidTurn((0,0,0),(0,0,0),(1,0,0),0.125))
    row=detect(model,paths)
    assert row.status=="unsupported" and row.sweep is None
