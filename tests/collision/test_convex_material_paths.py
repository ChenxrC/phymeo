"""Trajectory oracles for typed convex witnesses between two separated endpoint poses."""

from fractions import Fraction

import numpy as np
import pytest

import phymeo as ph


def query(a,b,ma,mb,margin=0.25):
    model=ph.CollisionModel((b,a))
    with ph.CollisionWorkspace(model) as workspace:
        result=ph.detect(model,{a.key:ma,b.key:mb},ph.CollisionPolicy(margin=margin),workspace)
    assert result.status=="ok"
    row,=result.pairs
    assert row.candidate
    return row


def keys(swapped):
    a,b=ph.CollisionKey(3,7,1),ph.CollisionKey(3,22,8)
    return (b,a) if swapped else (a,b)


def hit(row, expected):
    assert row.status=="hit",row.reason
    s=row.sweep
    lo,hi=s.first_t
    assert Fraction.from_float(lo) <= expected <= Fraction.from_float(hi)
    assert hi-lo<1e-4 and lo<=s.witness_time<=hi
    assert s.witness is not None
    return s


@pytest.mark.parametrize("swapped",[False,True])
def test_translating_round_solid_crosses_ellipsoid_and_ends_clear(swapped):
    ka,kb=keys(swapped)
    a=ph.CollisionShape(ka,"ellipsoid",(18,),curved=ph.CurvedSolid("ellipsoid",(.5,.25,.25)))
    b=ph.CollisionShape(kb,"sphere",(4,),.125)
    # Under common translation, separation is set by centers -9/8 + 3 t and 0.
    ba=np.array([-.25,.5,.125]); drift=np.array([.25,-.25,.5])
    ma=ph.CollisionMotion([ba], [ba+drift])
    mb=ph.CollisionMotion([ba+[-1.125,0,0]], [ba+drift+[1.875,0,0]])
    row=query(a,b,ma,mb)
    s=hit(row,Fraction(1,6))
    assert s.start_gap_m==pytest.approx(.5,abs=2e-5)
    assert s.end_gap_m==pytest.approx(1.25,abs=2e-5)
    assert row.witness.gap_m==s.end_gap_m
    t=s.witness_time
    pa,pb=(s.witness.second_point,s.witness.first_point) if swapped else (s.witness.first_point,s.witness.second_point)
    np.testing.assert_allclose(pa,ba+t*drift+[-.5,0,0],rtol=0,atol=2e-4)
    np.testing.assert_allclose(pb,ba+t*drift+[-1.+3*t,0,0],rtol=0,atol=2e-4)


def test_positive_fixed_plane_covers_a_relative_near_miss():
    ka,kb=keys(False)
    a=ph.CollisionShape(ka,"ellipsoid",(12,),curved=ph.CurvedSolid("ellipsoid",(.25,.25,.25)))
    b=ph.CollisionShape(kb,"sphere",(51,),.125)
    row=query(a,b,ph.CollisionMotion([[0,0,0]]),ph.CollisionMotion([[-.8,.4,0]],[[.7,.4,0]]))
    assert row.status=="separated" and row.sweep.first_t is None


@pytest.mark.parametrize("swapped",[False,True])
def test_interior_of_a_deforming_triangle_passes_curved_body_between_clear_endpoints(swapped):
    ka,kb=keys(swapped)
    a=ph.CollisionShape(ka,"ellipsoid",(5,),curved=ph.CurvedSolid("ellipsoid",(.375,.25,.125)))
    ids=(81,42,37)
    b=ph.CollisionShape(kb,"triangle",ids)
    start=np.array([[-2.,-1.,.5],[2.,-1.,.5],[0.,2.,.5]])
    end=start+np.array([[.25,0,-1.5],[-.125,-.125,-1.5],[.125,.25,-1.5]])
    row=query(a,b,ph.CollisionMotion([[0,0,0]]),ph.CollisionMotion(start,end))
    s=hit(row,Fraction(1,4))
    assert row.family=="RD" and s.start_gap_m>0 and s.end_gap_m>0
    w=s.witness
    weights,p=(w.first_weights,w.first_point) if swapped else (w.second_weights,w.second_point)
    x=start+s.witness_time*(end-start)
    assert {k for k,_ in weights}==set(ids)
    np.testing.assert_allclose(sum(v*x[ids.index(k)] for k,v in weights),p,atol=1e-9,rtol=0)
    assert abs(p[2]-.125)<2e-4


def test_material_face_collapsing_midpath_is_reported():
    ka,kb=keys(False)
    a=ph.CollisionShape(ka,"cone",(3,),curved=ph.CurvedSolid("cone",(.375,.25)))
    b=ph.CollisionShape(kb,"triangle",(32,19,23))
    p=np.array([[-1,-1,0],[1,-1,0],[0,1,0]],dtype=float)
    row=query(a,b,ph.CollisionMotion([[0,0,.35]]),ph.CollisionMotion(p,p[[1,0,2]]))
    assert row.status=="invalid_geometry" and row.sweep is None


def test_already_interpenetrating_solids_report_the_start_time():
    ka,kb=keys(False)
    a=ph.CollisionShape(ka,"cylinder",(1,),curved=ph.CurvedSolid("cylinder",(.5,.375)))
    b=ph.CollisionShape(kb,"cylinder",(2,),curved=ph.CurvedSolid("cylinder",(.25,.25)))
    row=query(a,b,ph.CollisionMotion([[0,0,0]]),ph.CollisionMotion([[.71875,0,0]],[[1.5,0,0]]))
    s=hit(row,Fraction(0))
    assert s.first_t==(0.,0.) and s.start_gap_m==pytest.approx(-1/32,abs=2e-5)
    assert s.end_gap_m>0
