"""First solid contact of translating closed meshes and finite round primitives."""

import math
from dataclasses import asdict
from fractions import Fraction
import json

import numpy as np
import pytest

from test_mesh_round import build,query,verify_points

COMMON=np.array([0.125,-0.1875,0.0625])


@pytest.mark.parametrize("kind",["through","edge-root","isolated","miss","initial","shaft","tip","notch"])
@pytest.mark.parametrize("reverse",[False,True])
def test_translating_closed_mesh_and_round_first_touch(kind,reverse,tmp_path):
    mesh="box"
    radius=0.25
    delta=np.array([4.0,0.0,0.0])
    start=[[-2,0.125,-0.25]]
    target=3/16
    if kind=="edge-root":
        start=[[-2,1.125,-0.25]]
        target=(1-math.sqrt(3)/8)/4
    elif kind=="isolated":
        radius=5/16
        delta=np.array([0.25,-0.1875,0])
        start=[np.array([1+3/16,1+4/16,0])-3*delta/8]
        target=3/8
    elif kind=="miss":
        start=[[-2,1.375,0]]
        target=None
    elif kind=="initial":
        start=[[0.5,0,0]]
        target=0
    elif kind=="shaft":
        start=[[-2,0.125,-1.5],[-2,0.125,1.5]]
    elif kind=="tip":
        start=[[-2.875,0.125,0],[-2.25,0.125,0]]
        delta=np.array([3,0,0])
        target=Fraction(1,3)
    elif kind=="notch":
        mesh="u"
        start=[[-0.5,0.25,0]]
        delta=np.array([1.375,0,0])
        target=Fraction(10,11)
    case=dict(mesh=mesh,axis_m=start,radius_m=radius)
    model,paths,ms,rs,mat,_=build(case,reverse,reverse,(COMMON,COMMON+delta))
    row=query(model,paths)
    assert row.status==("hit" if target is not None else "separated"),row.reason
    assert row.sweep.start_gap_m==row.mesh_round.start.witness.gap_m
    assert row.sweep.end_gap_m==row.witness.gap_m==row.mesh_round.end.witness.gap_m
    assert row.mesh_round.end.relation==("boundary_overlap" if kind=="notch" else "separated") if kind not in ("tip","initial") else True
    if target is None:
        assert row.sweep.first_t is row.sweep.witness is row.mesh_round.first_contact is None
        return
    event=row.mesh_round.first_contact
    assert row.sweep.first_t==event.interval and row.sweep.witness_time==event.witness_time
    assert row.sweep.witness==event.pose.witness
    lo,hi=event.interval
    assert 0<=lo<=target<=hi<=1 and hi-lo<=1e-6
    assert lo<=event.witness_time<=hi
    for support in event.pose.supports:
        pts,weights=verify_points(support,row,ms,rs,paths,event.witness_time)
        if kind=="initial":
            assert event.pose.relation=="mesh_contains_round" and event.interval==(0,0)
            assert event.pose.boundary_gap_m>0
        else:
            np.testing.assert_allclose(pts[0],pts[1],rtol=0,atol=1e-6)
        if kind=="tip":
            assert dict(weights[1])[rs.vertex_ids[1]]==pytest.approx(1,abs=1e-8)
    if kind=="shaft":
        q=[p.axis_parameter for s in event.pose.supports for p in s.witness.capsule_axis_span]
        assert min(q)==pytest.approx(1/6,abs=1e-9) and max(q)==pytest.approx(5/6,abs=1e-9)
    if kind=="isolated":
        assert row.mesh_round.start.boundary_gap_m>0 and row.mesh_round.end.boundary_gap_m>0
    (tmp_path/"mesh-round-sweep.json").write_text(json.dumps(dict(kind=kind,reverse=reverse,
        expected=float(target),interval=event.interval,witness_time=event.witness_time,
        start=row.mesh_round.start.relation,end=row.mesh_round.end.relation,
        materials=[asdict(s) for s in event.pose.supports]),allow_nan=False),encoding="utf-8")
