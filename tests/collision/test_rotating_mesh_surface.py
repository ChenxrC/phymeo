"""Independent first-boundary times of closed solids under one relative axial rotation."""

import json
import math
from dataclasses import asdict

import numpy as np
import pytest

import phymeo as ph
from rotating_mesh_oracles import harmonics, position
from test_mesh_mesh_contact import box
from test_rotating_mesh_candidates import double_u, mesh_pair, orbital, set_turn


def first_surface(shapes, paths, **kwargs):
    from phymeo.geometry.mesh_arc_surface import mesh_arc_first_surface
    return mesh_arc_first_surface(*shapes, *paths, **kwargs)


def validate(event, shapes, paths, expected):
    lo,hi = event.interval
    assert 0 <= lo <= expected <= hi <= 1 and hi-lo <= 1e-6
    assert lo <= event.witness_time <= hi
    assert event.supports
    vertices = [dict(zip(s.vertex_ids, position(harmonics(p),event.witness_time), strict=True))
                for s,p in zip(shapes,paths,strict=True)]
    for support in event.supports:
        witness = support.witness
        for s,face,table,weights,point in zip(
            shapes,(support.faces.first,support.faces.second),vertices,
            (witness.first_weights,witness.second_weights),(witness.first_point,witness.second_point),strict=True):
            assert face in s.triangles and {i for i,w in weights if w}<=set(face)
            assert min(w for _,w in weights)>=0
            assert sum(w for _,w in weights)==pytest.approx(1,abs=1e-12)
            np.testing.assert_allclose(sum(w*table[i] for i,w in weights),point,atol=1e-6,rtol=0)
        assert np.linalg.norm(np.subtract(witness.first_point,witness.second_point))<=1e-6
    return {(s.faces.first,s.faces.second):s.witness for s in event.supports}


def bar_and_wall(height,angle=3.0):
    shapes,paths=mesh_pair((box((-0.5,-0.375,-0.125),(0.5,0.375,0.125)),
                           box((-0.875,height,-0.25),(0.875,height+0.25,0.25),flip=True)))
    paths[0]=set_turn(paths[0], (0,0,0),(0,0,0),(0,0,1),angle)
    return shapes,paths


@pytest.mark.parametrize("mode",["right","left","coaxial"])
def test_true_concave_wall_and_matching_material_preimages(mode,tmp_path):
    shapes,paths=double_u("left" if mode=="left" else "right")
    angle=math.pi/2
    if mode=="coaxial":
        pivot=(0.375,0,0)
        drift=(0.125,0.0625,0)
        paths[0]=set_turn(paths[0],np.add(pivot,[0,0,0.5]),drift,(0,0,1),0.25)
        paths[1]=set_turn(paths[1],pivot,drift,(0,0,-1),-(0.25+angle))
        angle=abs(paths[1].turn.angle)-paths[0].turn.angle
    expected=(math.pi/4-math.acos(1.25/math.sqrt(2)))/angle
    row=first_surface(shapes,paths)
    assert row.status=="hit",row.reason
    found=validate(row.event,shapes,paths,expected)
    t=row.event.witness_time
    for w in found.values():
        p=np.array(w.first_point)-t*np.array([0.125,0.0625,0])
        if mode=="coaxial":
            p-=np.array([0.375,0,0])
            c,s=math.cos(-t*0.25),math.sin(-t*0.25)
            p=np.array([[c,-s,0],[s,c,0],[0,0,1]])@p+np.array([0.375,0,0])
        assert p[0]==pytest.approx(-1 if mode=="left" else 1,abs=1e-6)
    swapped=first_surface(shapes[::-1],paths[::-1])
    assert swapped.status=="hit" and swapped.event.interval==row.event.interval
    other=validate(swapped.event,shapes[::-1],paths[::-1],expected)
    assert set(found)=={(b,a) for a,b in other}
    for (a,b),w in found.items():
        np.testing.assert_allclose(w.first_point,other[b,a].second_point,atol=1e-6,rtol=0)
    (tmp_path/"u-surface.json").write_text(json.dumps(asdict(row),allow_nan=False),encoding="utf-8")


@pytest.mark.parametrize("height",[0.625-1/131072,0.625,0.625+1/1024])
def test_thin_angular_window_repeated_tangent_and_finite_miss(height,tmp_path):
    shapes,paths=bar_and_wall(height)
    angle=paths[0].turn.angle
    assert all(0.5*abs(math.sin(angle*j/64))+0.375*abs(math.cos(angle*j/64)) < height for j in range(65))
    row=first_surface(shapes,paths)
    if height>0.625:
        assert row.status=="surface_clear" and row.event is None
    else:
        expected=(math.atan2(0.5,0.375)-math.acos(height/0.625))/angle
        assert row.status=="hit",row.reason
        validate(row.event,shapes,paths,expected)
        for s in row.event.supports:
            assert s.witness.second_point[1]==pytest.approx(height,abs=1e-6)
    (tmp_path/"wall-surface.json").write_text(json.dumps(asdict(row),allow_nan=False),encoding="utf-8")


def test_empty_notch_and_contained_solid_both_have_no_boundary_event():
    shapes,paths=double_u("notch")
    assert first_surface(shapes,paths).status=="surface_clear"
    shapes,paths=mesh_pair((box((-1,)*3,(1,)*3),box((-0.125,)*3,(0.125,)*3)))
    paths[1]=set_turn(paths[1],(0,0,0),(0,0,0),(0,0,1),3)
    row=first_surface(shapes,paths)
    assert row.status=="surface_clear" and row.event is None


def test_starting_transverse_intersection_has_zero_time_and_both_faces():
    shapes,paths=mesh_pair((box((-1,-0.125,-0.25),(1,0.125,0.25)),
                            box((-0.125,-1,-0.125),(0.125,1,0.125))))
    paths[0]=set_turn(paths[0],(0,0,0),(0,0,0),(0,0,1),1)
    row=first_surface(shapes,paths)
    assert row.status=="hit" and row.event.interval==(0.0,0.0)
    validate(row.event,shapes,paths,0)


def test_two_skew_axes_and_differential_linear_drift_are_explicit():
    shapes,paths,_=orbital(dual=True)
    row=first_surface(shapes,paths)
    assert row.status=="unsupported" and row.event is None
    shapes,paths=bar_and_wall(0.625)
    paths[1]=ph.CollisionMotion(paths[1].start,paths[1].start+[0,0.125,0])
    row=first_surface(shapes,paths)
    assert row.status=="unsupported" and row.event is None


def test_exhausted_proof_budget_cannot_publish_a_sampled_hit_or_clear():
    shapes,paths=bar_and_wall(0.625)
    row=first_surface(shapes,paths,max_pair_charts=0)
    assert row.status=="unresolved" and row.event is None
