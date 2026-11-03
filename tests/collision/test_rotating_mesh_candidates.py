"""Conservative whole-arc face pairs of two independently owned closed meshes."""

import json
import math
import runpy
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.core.capabilities import SolverFailure

from rotating_mesh_oracles import (
    all_box_pairs,
    exact_face_boxes,
    face_indices,
    harmonics,
    position,
    transverse_crossings,
)
from test_mesh_mesh_contact import box, scene


def candidates(*args, **kwargs):
    from phymeo.geometry.rotating_mesh import mesh_arc_face_candidates

    return mesh_arc_face_candidates(*args, **kwargs)


def mesh_pair(geometry):
    m, paths = scene(*geometry)
    a, b = m.shapes
    return (a, b), [paths[a.key], paths[b.key]]


def set_turn(path, pivot, drift, axis, angle):
    return ph.CollisionMotion(path.start, turn=ph.RigidTurn(pivot, np.add(pivot, drift), axis, angle))


def orbital(*, short=False, dual=False, sign=1, angle=math.pi):
    when = 21.5 / 64 if short else math.pi / (2 * abs(angle))
    angle = sign * (3 if short else angle)
    at = np.array([math.cos(when * angle), math.sin(when * angle), 0])
    radius = (np.array([2, 1, 2]) / 2048) if short else np.array([1/12, 1/24, 1/20])
    target = at + radius * [0.4, 0.2, 0.4]
    first = box(np.array([1,0,0]) - radius, np.array([1,0,0]) + radius)
    second = box(target - radius * [0.8,0.9,0.7], target + radius * [0.8,0.9,0.7], flip=True)
    shapes, paths = mesh_pair((first, second))
    drift = np.array([0.125, -0.0625, 0.25]) if dual else np.zeros(3)
    paths[0] = set_turn(paths[0], (0,0,0), drift, (0,0,1), angle)
    if dual:
        paths[1] = set_turn(paths[1], target, drift + [0.015625,0,0], (2/3,2/3,1/3), -2.4)
    return shapes, paths, when


def double_u(mode):
    adapter = runpy.run_path(str(Path(__file__).resolve().parents[1]/"reference_newton"/"rr"/"baseline.py"))
    geometry = []
    shift = 0.0 if mode == "notch" else (-0.375 if mode == "left" else 0.375)
    for scale, flip, x in (([1,1,1],False,0),([0.25,0.25,0.5],True,shift)):
        points, triangles = adapter["_u_geometry"](scale, flip)
        geometry.append((points + [x,0,0], triangles.reshape(-1,3)))
    shapes, paths = mesh_pair(geometry)
    drift = np.array([0.125,0.0625,0])
    paths[0] = ph.CollisionMotion(paths[0].start, paths[0].start+drift)
    angle = (-1 if mode == "left" else 1) * math.pi / 2
    paths[1] = set_turn(paths[1], (shift,0,0), drift, (0,0,1), angle)
    return shapes, paths


def pairs(window):
    return {(p.first, p.second):p for p in window.pairs}


def audit(shapes, paths, result, tmp_path, name, margin=0.0):
    terms = [harmonics(path) for path in paths]
    indices = [face_indices(s) for s in shapes]
    previous = 0.0
    brute_total = 0
    inventory = []
    for window in result.windows:
        t0, t1 = window.interval
        assert t0 == previous and t0 < t1 <= 1
        previous = t1
        tables = [exact_face_boxes(t, ix, t0, t1) for t, ix in zip(terms, indices, strict=True)]
        brute = {(shapes[0].triangles[i], shapes[1].triangles[j]) for i,j in all_box_pairs(*tables, margin)}
        got = pairs(window)
        assert brute <= got.keys()
        assert len(got) == len(window.pairs)
        assert all(a in shapes[0].triangles and b in shapes[1].triangles for a,b in got)
        brute_total += len(brute)
        inventory.append(dict(interval=window.interval, retained=len(got), brute=len(brute)))
    assert previous == 1 and (result.first, result.second) == tuple(s.key for s in shapes)
    assert result.possible_pairs == len(result.windows) * math.prod(len(s.triangles) for s in shapes)
    assert result.candidate_count == sum(row["retained"] for row in inventory)
    assert result.bvh_node_visits >= result.bvh_leaf_tests
    record = dict(name=name, intervals=len(result.windows), possible=result.possible_pairs,
                  candidates=result.candidate_count, independent_swept_aabb_pairs=brute_total,
                  leaf_tests=result.bvh_leaf_tests, node_visits=result.bvh_node_visits, windows=inventory)
    (tmp_path/(name+".json")).write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
    return terms


def check_material(shapes, terms, when, result):
    crossing = transverse_crossings(shapes, terms, when)
    assert crossing
    relevant = [w for w in result.windows if w.interval[0] <= when <= w.interval[1]]
    assert relevant
    for window in relevant:
        got = pairs(window)
        assert crossing.keys() <= got.keys()
        for pair, (wa,wb,expected) in crossing.items():
            weights = got[pair].material_weights(wa,wb)
            for side, (shape, term, face, material) in enumerate(zip(shapes, terms, pair, weights, strict=True)):
                assert [i for i,_ in material] == list(face)
                np.testing.assert_allclose([v for _,v in material], (wa,wb)[side], atol=1e-14)
                coords = dict(zip(shape.vertex_ids, position(term,when), strict=True))
                np.testing.assert_allclose(sum(v*coords[i] for i,v in material), expected, atol=2e-10, rtol=0)
    return len(crossing)


@pytest.mark.parametrize("kind", ["offcenter","dual","short","reverse","later-revolution"])
def test_offcenter_and_dual_rotors_keep_all_continuous_face_boxes(kind,tmp_path):
    angle = 2*math.pi + math.pi/2 if kind == "later-revolution" else math.pi
    shapes, paths, when = orbital(short=kind=="short", dual=kind=="dual", sign=-1 if kind=="reverse" else 1, angle=angle)
    if kind == "later-revolution":
        # Revisit the original target after one entire revolution, well before t=1.
        angle = 3*math.pi
        shapes, paths, when = orbital(angle=angle)
    result = candidates(*shapes,*paths)
    terms = audit(shapes,paths,result,tmp_path,kind)
    assert result.candidate_count < result.possible_pairs//2
    assert not transverse_crossings(shapes,terms,0) and not transverse_crossings(shapes,terms,1)
    assert check_material(shapes,terms,when,result)
    if kind == "short":
        # Full bounding spheres prove separation at every one of the 65 times.
        radii = [np.linalg.norm(p.start-p.start.mean(0),axis=1).max() for p in paths]
        assert all(np.linalg.norm(position(terms[0],j/64).mean(0)-position(terms[1],j/64).mean(0))>sum(radii) for j in range(65))
    if kind == "later-revolution":
        assert check_material(shapes,terms,when+math.tau/angle,result)
    if kind == "dual":
        swapped = candidates(*shapes[::-1],*paths[::-1])
        for a,b in zip(result.windows,swapped.windows,strict=True):
            assert a.interval==b.interval and set(pairs(a)) == {(j,i) for i,j in pairs(b)}


@pytest.mark.parametrize("mode",["notch","right","left"])
def test_two_concave_u_boundaries_and_open_notch(mode,tmp_path):
    shapes,paths = double_u(mode)
    result = candidates(*shapes,*paths)
    terms = audit(shapes,paths,result,tmp_path,"u-"+mode)
    assert result.candidate_count < result.possible_pairs//4
    if mode=="notch":
        # Every rotating insert point has XY radius <= sqrt(1/2)<1. The outer U
        # occupies x<=-1, x>=1 or y<=-1 in the frame moving with both pivots.
        assert result.candidate_count==0
    else:
        assert not transverse_crossings(shapes,terms,0) and not transverse_crossings(shapes,terms,1)
        assert check_material(shapes,terms,0.5,result)
        # First/last boundary events are at |shift|+0.5*(cos+sin)=1.
        delta = math.acos(1.25/math.sqrt(2))
        for phase in (math.pi/4-delta,math.pi/4+delta):
            t=phase/(math.pi/2)
            support=position(terms[1],t)
            side=1 if mode=="right" else -1
            assert np.max(side*(support[:,0] - 0.125*t)) == pytest.approx(1,abs=1e-12)
            assert any(w.pairs for w in result.windows if w.interval[0]<=t<=w.interval[1])


def test_real_clearance_can_retain_candidates_when_margin_is_requested(tmp_path):
    shapes,paths,_=orbital()
    lifted=paths[1].start+np.array([0,0,0.25])
    paths[1]=ph.CollisionMotion(lifted)
    clear=candidates(*shapes,*paths,max_candidates=0)
    assert clear.candidate_count==0
    result=candidates(*shapes,*paths,margin_m=0.25)
    audit(shapes,paths,result,tmp_path,"clear-margin",0.25)
    assert result.candidate_count>0 and lifted[:,2].min() - paths[0].start[:,2].max()>0.125


def test_invalid_path_and_exhausted_resources_return_no_partial_face_table():
    shapes,paths,_=orbital()
    result=candidates(*shapes,*paths)
    with pytest.raises(SolverFailure,match="candidate_capacity") as err:
        candidates(*shapes,*paths,max_candidates=result.candidate_count-1)
    assert str(result.candidate_count) in str(err.value)
    paths[0]=set_turn(paths[0],(0,0,0),(0,0,0),(0,0,1),1e6)
    with pytest.raises(SolverFailure,match="window_capacity"):
        candidates(*shapes,*paths)
    moved=paths[0].start.copy()
    moved[0,1]+=0.125
    paths[0]=ph.CollisionMotion(paths[0].start,moved)
    with pytest.raises(ValueError,match="rigid"):
        candidates(*shapes,*paths)
    with pytest.raises(ValueError,match="barycentric"):
        next(p for w in result.windows for p in w.pairs).material_weights((2,-1,0),(1,0,0))
