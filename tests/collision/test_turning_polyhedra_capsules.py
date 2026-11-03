"""Known first contacts for finite boxes and capsules rotating about different axes."""

from itertools import product
from math import acos, asin, atan2, pi, sqrt

import numpy as np
import pytest

import phymeo as ph


def observe(a,b,ma,mb,backend="host",cache=None,patch=False):
    model=ph.CollisionModel((b,a))
    with ph.CollisionWorkspace(model,backend,cache_root=cache) as ws:
        (row,)=ph.detect(model,{a.key:ma,b.key:mb},ph.CollisionPolicy(margin=.2,proximity_patches=patch),ws).pairs
        if backend!="host":
            assert ws.stats()["completed_launches"]>0
    assert row.candidate
    return row


def check_hit(row,first,endpoint_gap):
    assert row.status=="hit",row.reason
    lo,hi=row.sweep.first_t
    assert lo<=first<=hi and hi-lo<1/1024
    assert lo<=row.sweep.witness_time<=hi and row.sweep.witness is not None
    assert row.sweep.start_gap_m==pytest.approx(endpoint_gap,abs=2e-6)
    assert row.sweep.end_gap_m==pytest.approx(endpoint_gap,abs=2e-6)
    assert row.support_map is not None
    return row.sweep.witness


def caps(reverse=False,backend="host",cache=None):
    ka,kb=[ph.CollisionKey(28,k,0) for k in ((7,2) if reverse else (2,7))]
    a=ph.CollisionShape(ka,"capsule",(15,3),1/16)
    b=ph.CollisionShape(kb,"capsule",(23,12),1/8)
    pivot=np.array([1.,-1.,.5]);drift=np.array([-.25,.125,.5])
    ma=ph.CollisionMotion(pivot+[[1.,0,-.25],[1.,0,.25]],turn=ph.RigidTurn(pivot,pivot+drift,(0,0,1),pi))
    mb=ph.CollisionMotion(pivot+[[-.25,0,1.],[.25,0,1.]],turn=ph.RigidTurn(pivot,pivot+drift,(1,0,0),-pi))
    row=observe(a,b,ma,mb,backend,cache)
    w=check_hit(row,acos((1/4+3/(16*sqrt(2))))/pi,sqrt(2)*.75-3/16)
    wa,wb=(w.second_weights,w.first_weights) if reverse else (w.first_weights,w.second_weights)
    assert dict(wa)==pytest.approx({3:1.}) and dict(wb)==pytest.approx({12:1.})
    return row


@pytest.mark.parametrize("reverse",[False,True])
def test_two_perpendicular_capsule_orbits_keep_both_caps(reverse):
    caps(reverse)


def boxes(face=False,reverse=False):
    box_key, other_key=[ph.CollisionKey(28,i,0) for i in ((9,8) if reverse else (8,9))]
    corners=np.array(list(product((-1.,1.),repeat=3)))
    block=ph.CollisionShape(box_key,"hull",tuple(range(8)))
    pivot=np.array([0.,0.,.375]);drift=np.array([.25,-.375,.125])
    block_motion=ph.CollisionMotion(pivot+corners*[.5,.125,.125],turn=ph.RigidTurn(pivot,pivot+drift,(0,1,0),pi))
    if face:
        ground=ph.CollisionShape(other_key,"triangle",(3,7,1))
        top=np.array([[-2.,-2.,0],[2.,-2.,0],[0,2.,0]])
        support=ph.CollisionMotion(top,top+drift+[[.125,0,0],[0,-.125,0],[-.125,0,0]])
    else:
        ground=ph.CollisionShape(other_key,"hull",tuple(range(31,39)))
        center=np.array([0,0,-.25])
        support=ph.CollisionMotion(center+corners*[2.,2.,.25],turn=ph.RigidTurn(center,center+drift,(0,0,1),pi/3))
    return block,ground,block_motion,support


@pytest.mark.parametrize("face,reverse",[(False,False),(False,True),(True,True)])
def test_tipping_box_meets_finite_rotating_slab_or_deforming_cloth(face,reverse):
    a,b,ma,mb=boxes(face,reverse)
    row=observe(a,b,ma,mb)
    # Until pi/2 the bottom support relative to the slab top is 3/8 - (4sin+cos)/8.
    expected=(asin(3/sqrt(17))-atan2(1,4))/pi
    w=check_hit(row,expected,.25)
    ground_point=w.first_point if reverse else w.second_point
    assert abs(ground_point[2]-row.sweep.witness_time*.125)<3e-6
    assert row.family==("RD" if face else "RR")


@pytest.mark.tyx
def test_real_broad_phase_preserves_capsule_time_and_material(tyx_mode,tmp_path):
    caps(True,tyx_mode,tmp_path)
