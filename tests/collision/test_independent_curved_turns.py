"""Independent orbital and support-function oracles for two material motions."""

import json
from math import acos, asin, cos, pi, sin, sqrt

import numpy as np
import pytest

import phymeo as ph


def query(shapes, paths, backend="host", cache=None):
    model=ph.CollisionModel(shapes[::-1])
    with ph.CollisionWorkspace(model,backend,cache_root=cache) as work:
        (row,)=ph.detect(model,paths,ph.CollisionPolicy(margin=.125),work).pairs
        if backend!="host":
            assert work.stats()["completed_launches"]>0
    assert row.candidate
    return row


def orbit(swapped=False,rotate=False,near=False):
    # Both centres rotate by a half turn about distinct moving axes. Their relative
    # vector is (cos(theta),0,-cos(theta)) minus a constant perpendicular shift.
    frame=np.array([[0,0,1],[1,0,0],[0,1,0.]]) if rotate else np.eye(3)
    base=np.array([3.,-.5,1.])
    drift=frame@np.array([.25,-.125,.375])
    offset=frame@np.array([.25,0,.25]) if near else np.zeros(3)
    ka,kb=[ph.CollisionKey(16,k,0) for k in ((7,2) if swapped else (2,7))]
    a=ph.CollisionShape(ka,"ellipsoid",(9,),curved=ph.CurvedSolid("ellipsoid",(.125,)*3))
    b=ph.CollisionShape(kb,"sphere",(20,),3/16)
    ma=ph.CollisionMotion([base+frame@[1,0,0]],turn=ph.RigidTurn(base,base+drift,frame@[0,0,1],pi))
    mb=ph.CollisionMotion([base+offset+frame@[0,0,1]],turn=ph.RigidTurn(base+offset,base+offset+drift,frame@[1,0,0],-pi))
    return (a,b),{ka:ma,kb:mb},frame,base,drift


def check_cross(swapped=False,rotate=False,backend="host",cache=None):
    shapes,paths,R,base,drift=orbit(swapped,rotate)
    row=query(shapes,paths,backend,cache)
    assert row.status=="hit",row.reason
    sweep=row.sweep
    expected=acos((5/16)/sqrt(2))/pi
    lo,hi=sweep.first_t
    assert lo<=expected<=hi and hi-lo<1/1024
    assert sweep.start_gap_m==pytest.approx(sqrt(2)-5/16,abs=3e-6)
    assert sweep.end_gap_m==pytest.approx(sqrt(2)-5/16,abs=3e-6)
    w=sweep.witness
    t=sweep.witness_time
    assert lo<=t<=hi and w is not None
    two=(w.second_point,w.first_point) if swapped else (w.first_point,w.second_point)
    c1=base+drift*t+R@[cos(pi*t),sin(pi*t),0]
    c2=base+drift*t+R@[0,sin(pi*t),cos(pi*t)]
    assert np.linalg.norm(two[0]-c1)==pytest.approx(1/8,abs=2e-5)
    assert np.linalg.norm(two[1]-c2)==pytest.approx(3/16,abs=2e-5)
    n=(c2-c1)/np.linalg.norm(c2-c1)
    assert np.dot(w.normal,(-1 if swapped else 1)*n)>.999
    return dict(expected=expected,interval=sweep.first_t,end_gap_m=sweep.end_gap_m,
                material_points_m=two,time_cells=row.support_map.time_cells,static_samples=row.support_map.static_samples)


@pytest.mark.parametrize("swapped,rotate",[(False,False),(True,False),(False,True)])
def test_perpendicular_moving_pivots_hit_between_free_endpoints(swapped,rotate,tmp_path):
    observed=check_cross(swapped,rotate)
    (tmp_path/"independent-orbits.json").write_text(json.dumps(observed),encoding="utf-8")


@pytest.mark.tyx
def test_cuda_pair_retains_independent_turn_first_contact(tyx_mode,tmp_path):
    check_cross(True,True,tyx_mode,tmp_path)


def test_two_rotating_centres_miss_inside_their_swept_boxes():
    shapes,paths,*_=orbit(near=True)
    row=query(shapes,paths)
    assert sqrt(1/8)-5/16>0
    assert row.status=="separated" and row.sweep.first_t is None
    assert row.support_map.time_cells>1


@pytest.mark.parametrize("reverse",[False,True])
def test_rotating_anisotropic_ellipsoid_meets_a_deforming_face(reverse):
    ka,kb=[ph.CollisionKey(2,k,0) for k in ((3,1) if reverse else (1,3))]
    solid=ph.CollisionShape(ka,"ellipsoid",(8,),curved=ph.CurvedSolid("ellipsoid",(.5,.125,.125)))
    nodes=(17,3,14)
    face=ph.CollisionShape(kb,"triangle",nodes)
    start=np.array([[-2.,-2.,0],[2.,-2.,0],[0,2.,0]])
    end=start+np.array([[.125,0,0],[0,-.125,0],[-.125,0,0]])
    center=np.array([0.,0.,3/8])
    turn=ph.RigidTurn(center,center,(0,1,0),pi)
    row=query((solid,face),{ka:ph.CollisionMotion([center],turn=turn),kb:ph.CollisionMotion(start,end)})
    assert row.status=="hit",row.reason
    sweep=row.sweep
    lo,hi=sweep.first_t
    assert lo<=asin(sqrt(8/15))/pi<=hi and hi-lo<1/1024
    assert sweep.start_gap_m==pytest.approx(.25,abs=1e-6) and sweep.end_gap_m==pytest.approx(.25,abs=1e-6)
    t=sweep.witness_time
    w=sweep.witness
    point,weights=(w.first_point,w.first_weights) if reverse else (w.second_point,w.second_weights)
    assert set(dict(weights))==set(nodes)
    x=(1-t)*start+t*end
    np.testing.assert_allclose(point,sum(weight*x[nodes.index(i)] for i,weight in weights),atol=2e-8)


def test_grazing_is_never_published_as_free():
    a,b=ph.CollisionKey(2,1,0),ph.CollisionKey(2,3,0)
    left=ph.CollisionShape(a,"ellipsoid",(0,),curved=ph.CurvedSolid("ellipsoid",(.125,)*3))
    right=ph.CollisionShape(b,"sphere",(0,),3/16)
    t=ph.RigidTurn((0,0,0),(0,0,0),(0,0,1),pi)
    row=query((left,right),{a:ph.CollisionMotion([[1,0,0]],turn=t),b:ph.CollisionMotion([[0,1+5/16,0]])})
    assert row.status in ("unresolved","hit"),row.reason
    if row.status=="hit":
        assert row.sweep.first_t[0]<=.5<=row.sweep.first_t[1]
