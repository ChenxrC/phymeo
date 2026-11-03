"""Public closed-solid CCD under coaxial rotations and a shared linear pivot drift."""

import json
import math
from dataclasses import asdict,replace
from fractions import Fraction

import numpy as np
import pytest

import phymeo as ph
from rotating_mesh_oracles import harmonics,position
from test_mesh_mesh_contact import box
from test_rotating_mesh_candidates import double_u,mesh_pair,orbital,set_turn
from test_rotating_mesh_surface import bar_and_wall


def detect(shapes,paths,*,swap=False,excluded=False):
    if swap:
        shapes=[replace(shapes[0],key=shapes[1].key),replace(shapes[1],key=shapes[0].key)]
    mapping={s.key:p for s,p in zip(shapes,paths,strict=True)}
    excluded_pairs=[(shapes[0].key,shapes[1].key)] if excluded else []
    model=ph.CollisionModel(shapes,excluded=excluded_pairs)
    with ph.CollisionWorkspace(model) as work:
        (row,)=ph.detect(model,mapping,ph.CollisionPolicy(margin=0.125),work).pairs
    return row,model.shapes,[mapping[s.key] for s in model.shapes]


def materials(w,shapes,paths,t):
    for shape,path,weights,point in zip(shapes,paths,(w.first_weights,w.second_weights),
            (w.first_point,w.second_point),strict=True):
        assert abs(sum(x for _,x in weights)-1)<1e-10 and min(x for _,x in weights)>=0
        nodes=dict(zip(shape.vertex_ids,position(harmonics(path),t),strict=True))
        np.testing.assert_allclose(sum(x*nodes[k] for k,x in weights),point,atol=1e-6,rtol=0)
    if w.normal is not None:
        assert np.dot(np.subtract(w.second_point,w.first_point),w.normal)==pytest.approx(w.gap_m,abs=1e-6)


def bracket(interval,oracle):
    assert 0<=interval[0]<=oracle<=interval[1]<=1 and interval[1]-interval[0]<=1e-6


@pytest.mark.parametrize("mode",["notch","right","coaxial","partial","tangent","miss"])
@pytest.mark.parametrize("swap",[False,True])
def test_ordinary_detect_rotating_closed_solids(mode,swap,tmp_path):
    if mode in ("tangent","miss"):
        shapes,paths=bar_and_wall(0.625 if mode=="tangent" else 0.625+1/1024)
        expected=math.atan2(0.5,0.375)/3 if mode=="tangent" else None
    else:
        shapes,paths=double_u("notch" if mode=="notch" else "right")
        pivot=(0.375,0,0)
        shift=(0.125,0.0625,0)
        angle=math.pi/4 if mode=="partial" else math.pi/2
        if mode=="coaxial":
            paths[0]=set_turn(paths[0],np.add(pivot,[0,0,0.5]),shift,(0,0,1),0.25)
            paths[1]=set_turn(paths[1],pivot,shift,(0,0,-1),-(0.25+angle))
        elif mode=="partial":
            paths[1]=set_turn(paths[1],pivot,shift,(0,0,1),angle)
        expected=(math.pi/4-math.acos(1.25/math.sqrt(2)))/angle if mode!="notch" else None
    row,ordered,ordered_paths=detect(shapes,paths,swap=swap)
    assert row.status==("hit" if expected is not None else "separated"),row.reason
    trace=row.mesh_mesh
    assert trace.start.relation=="separated" and trace.start.witness.gap_m>0
    assert trace.end.relation==("boundary_overlap" if mode=="partial" else "separated")
    assert trace.end.witness==row.witness
    assert row.sweep.end_gap_m==row.witness.gap_m and row.sweep.start_gap_m==trace.start.witness.gap_m
    materials(row.witness,ordered,ordered_paths,1)
    for support in trace.end.boundary:
        materials(support.witness,ordered,ordered_paths,1)
    if mode=="partial":
        assert trace.end.boundary_gap_m==0 and trace.end.boundary
    if mode in ("right","coaxial"):
        assert trace.end.boundary_gap_m==pytest.approx(0.125,abs=1e-6)
    if expected is None:
        assert row.sweep.first_t is trace.first_boundary is None
    else:
        bracket(row.sweep.first_t,expected)
        event=trace.first_boundary
        assert row.sweep.first_t==event.interval and row.sweep.witness_time==event.witness_time
        assert trace.first_ratio is None
        materials(row.sweep.witness,ordered,ordered_paths,event.witness_time)
        for s in event.supports:
            assert s.faces.first in ordered[0].triangles and s.faces.second in ordered[1].triangles
            materials(s.witness,ordered,ordered_paths,event.witness_time)
    (tmp_path/"solid-arc.json").write_text(json.dumps(dict(mode=mode,swap=swap,
        first=row.sweep.first_t,witness_time=row.sweep.witness_time,start=trace.start.relation,
        end=trace.end.relation,end_gap_m=trace.end.boundary_gap_m,end_certificate=trace.end_certificate,
        end_error_m=trace.end_error_m, boundary=asdict(trace.first_boundary) if trace.first_boundary else None),
        allow_nan=False),encoding="utf-8")


@pytest.mark.parametrize("leaves",[False,True])
def test_contained_start_is_time_zero_even_when_first_surface_is_later_or_absent(leaves):
    eps=0.0625
    center=np.array([0.75,0,0]) if leaves else np.zeros(3)
    shapes,paths=mesh_pair((box((-1,)*3,(1,)*3),box(center-eps,center+eps)))
    angle=math.pi/2 if leaves else 0.7
    pivot=(0.75,1,0) if leaves else (0,0,0)
    paths[1]=set_turn(paths[1],pivot,(0,0,0),(0,0,1),angle)
    row,ordered,ordered_paths=detect(shapes,paths)
    assert row.status=="hit",row.reason
    trace=row.mesh_mesh
    assert trace.start.relation=="first_contains_second" and trace.start.boundary_gap_m>0
    assert not trace.start.boundary and trace.entry is trace.start
    assert row.sweep.first_t==(0.0,0.0) and row.sweep.witness_time==0 and row.sweep.witness==trace.start.witness
    assert trace.first_ratio==(0,1)
    materials(row.sweep.witness,ordered,ordered_paths,0)
    materials(row.witness,ordered,ordered_paths,1)
    if leaves:
        expected=(math.asin(0.25/math.hypot(1+eps,eps))-math.atan2(eps,1+eps))/angle
        assert trace.end.relation=="separated"
        bracket(trace.first_boundary.interval,expected)
        assert trace.first_boundary.interval[0]>0
        assert trace.end.boundary_gap_m==pytest.approx(0.75-eps,abs=1e-6)
    else:
        assert trace.first_boundary is None and trace.end.relation=="first_contains_second"
        assert trace.end.boundary_gap_m==pytest.approx(1-eps*(math.sin(angle)+math.cos(angle)),abs=1e-6)


def test_unsupported_skew_and_drift_filtered_and_adjacent_endpoint_control():
    shapes,paths,_=orbital(dual=True)
    row,*_=detect(shapes,paths)
    assert row.status=="unsupported" and row.sweep is row.mesh_mesh is None
    row,*_=detect(shapes,paths,excluded=True)
    assert row.status=="filtered" and row.sweep is None
    shapes,paths=bar_and_wall(0.625)
    paths[1]=ph.CollisionMotion(paths[1].start,paths[1].start+[0,0.125,0])
    row,*_=detect(shapes,paths)
    assert row.status=="unsupported"
    # This stored angle terminates within a floating-point step of the exact
    # peak. A root ordering or endpoint relation that cannot be certified is explicit.
    shapes,paths=bar_and_wall(0.625,angle=math.atan2(0.5,0.375))
    row,*_=detect(shapes,paths)
    assert row.status=="unresolved" and row.sweep is None


def test_original_linear_mesh_ccd_still_publishes_exact_entry_ratio():
    shapes,paths=mesh_pair((box((0,0,0),(1,1,1)),box((-0.75,0.25,0.25),(-0.5,0.75,0.75))))
    paths[1]=ph.CollisionMotion(paths[1].start,paths[1].start+[2.5,0,0])
    row,*_=detect(shapes,paths)
    assert row.status=="hit" and row.mesh_mesh.first_ratio==(1,5)
    bracket(row.sweep.first_t,Fraction(1,5))
