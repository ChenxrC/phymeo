"""Independent affine-path geometry, strict positive layer entry and isolated grazing."""
import json
from fractions import Fraction as F
from math import sqrt
from pathlib import Path
import runpy

import numpy as np
import pytest
import phymeo as ph
from test_deformable_features import CAT,DD,fixture,key,query,register


def vt_mesh(start=.25,end=-.75,*,corner=None):
    a=np.array([[-1,-1,0],[1,-1,0],[0,1,0.]]) if corner is None else np.array([[0,0,0.],[1,0,0],[0,1,0]])
    xy=(0.,0.) if corner is None else corner
    b=np.array([[*xy,start],[xy[0]-.125,xy[1]+.125,start+.25],[xy[0]+.125,xy[1]+.125,start+.25]])
    model,paths=fixture([(0,7,0,(11,23,37),a),(0,13,0,(53,67,79),b)])
    paths[model.shapes[1].key]=ph.CollisionMotion(b,b+[0,0,end-start])
    return model,paths,("VT",(53,),(11,23,37))


def ee_mesh():
    a=np.array([[-1,0,0.],[1,0,0],[0,-.75,-.125]])
    b=np.array([[0,-1,.375],[0,1,.375],[.75,0,.5]])
    a1=a+[[0,-.25,0],[0,.25,0],[-.125,0,0]]
    b1=b+[[.125,0,-.75],[-.125,0,-.75],[0,-.125,-.75]]
    model,paths=fixture([(0,7,0,(11,23,37),a),(0,13,0,(53,67,79),b)])
    for s,p,q in zip(model.shapes,(a,b),(a1,b1),strict=True): paths[s.key]=ph.CollisionMotion(p,q)
    return model,paths,("EE",(11,23),(53,67))


def fold_mesh():
    shapes,base=register(CAT["DD-REMOTE-FOLD"])
    model=ph.CollisionModel(shapes)
    pack={}
    for s in shapes:
        p=base[s.key].start.copy()
        e=p.copy()
        ids=np.asarray(s.vertex_ids)
        p[ids>=112,2]=.25
        e[ids>=112,2]=-.25
        pack[s.key]=ph.CollisionMotion(p,e)
    return model,pack,("VT",(114,),(100,101,102))


def sweep(m,p,w,*,search=1/8,force=1/16,**kw):
    return ph.sweep_deformable_features(m,p,w,search_distance_m=search,contact_distance_m=force,**kw)


def lookup(frame,k):
    return {key(row):row for row in frame.paths}[k]


def geometry_at(model,paths,time):
    return {(s.key.world,s.key.body,id_):x+time*(z-x) for s in model.shapes for id_,x,z in zip(
        s.vertex_ids, paths[s.key].start,paths[s.key].start if paths[s.key].end is None else paths[s.key].end,strict=True)}


def check_event(event,truth,model,paths,radius,strict=False):
    assert event.interval[0]<=truth<=event.interval[1] and event.interval[1]-event.interval[0]<=2**-34
    assert 0<=event.sample_time<=1
    row=event.sample
    xyz=geometry_at(model,paths,event.sample_time)
    f=lambda group:tuple(tuple(F(float(v)) for v in xyz[n]) for n in group)
    lib=runpy.run_path(str(DD/"oracles.py"))
    d2=lib["vt"](f(row.first_nodes)[0],f(row.second_nodes)) if row.kind=="VT" else lib["ee"](f(row.first_nodes),f(row.second_nodes))
    assert row.distance_m==pytest.approx(sqrt(d2),abs=2e-12)
    for q in row.closest:
        for nodes,weights,pos in ((row.first_nodes,q.first_weights,q.first_point),(row.second_nodes,q.second_weights,q.second_point)):
            by_id={n[2]:xyz[n] for n in nodes}
            reconstructed=sum(w*by_id[i] for i,w in weights)
            np.testing.assert_allclose(pos,reconstructed,rtol=0,atol=2e-12)
    if strict: assert event.sample_relation=="inside" and d2<F(radius)**2
    else: assert abs(row.distance_m-radius)<=event.distance_error_bound_m+3e-12


def test_transient_vertex_plane_has_exact_first_touch_and_verified_inside_material(tmp_path):
    m,p,k=vt_mesh()
    with ph.CollisionWorkspace(m,"host") as w:
        for t in (0.,1.):
            vals=geometry_at(m,p,t)
            static={s.key:ph.CollisionMotion([vals[s.key.world,s.key.body,n] for n in s.vertex_ids]) for s in m.shapes}
            assert not query(m,static,w,search=.125,force=.0625).contacts
        frame=sweep(m,p,w)
    assert frame.status=="ok"
    row=lookup(frame,k)
    assert row.contact.start_relation==row.contact.end_relation=="outside"
    check_event(row.search.first_touch,1/8,m,p,1/8)
    check_event(row.search.first_entry,1/8,m,p,1/8,strict=True)
    check_event(row.contact.first_touch,3/16,m,p,1/16)
    check_event(row.contact.first_entry,3/16,m,p,1/16,strict=True)
    assert row.contact.first_touch.interval==(3/16,3/16)
    q=row.contact.first_touch.sample.closest[0]
    assert q.first_weights==((53,1.),) and q.second_weights==((11,.25),(23,.25),(37,.5))
    (tmp_path/"vt-layer-entry.json").write_text(json.dumps(dict(first=row.contact.first_touch.interval,
        witness_time=row.contact.first_entry.sample_time,sample_gap=row.contact.first_entry.sample.gap_to_force_m,
        broad=frame.broad_face_pairs,tests=frame.feature_tests,root_nodes=frame.root_nodes)),encoding="utf-8")


def test_independently_deforming_edges_meet_through_positive_distance_and_mirror_swap():
    m,p,k=ee_mesh()
    observations=[]
    for mirror in (False,True):
        if mirror:
            shapes=[];paths={}
            for s in m.shapes:
                key_=ph.CollisionKey(0,20-s.key.body,s.key.shape)
                q=ph.CollisionShape(key_,"triangle",s.vertex_ids[::-1]);shapes.append(q)
                transform=lambda values:values[::-1]*[-1,1,1]+[1/4,-1/2,1/8]
                paths[key_]=ph.CollisionMotion(transform(p[s.key].start),transform(p[s.key].end))
            model=ph.CollisionModel(shapes)
            target=("EE",k[2],k[1])
        else: model,paths,target=m,p,k
        with ph.CollisionWorkspace(model,"host") as w:
            result=sweep(model,paths,w,search=3/16,force=3/32)
        assert result.status=="ok"
        out=lookup(result,target)
        check_event(out.search.first_touch,1/4,model,paths,3/16)
        check_event(out.contact.first_entry,3/8,model,paths,3/32,strict=True)
        value=out.contact.first_touch
        check_event(value,3/8,model,paths,3/32)
        assert [w for _,w in value.sample.closest[0].first_weights]==[.5,.5]
        observations.append(value.interval)
    assert observations[0]==observations[1]


@pytest.mark.parametrize("x,kind",[(-1/16,"grazing"),(-1/16-1/4096,"near_miss")])
def test_irrational_search_boundary_and_isolated_force_tangent(x,kind):
    m,p,k=vt_mesh(-.25,.25,corner=(x,0.))
    with ph.CollisionWorkspace(m,"host") as w: frame=sweep(m,p,w)
    row=lookup(frame,k)
    truth=(.25-sqrt((1/8)**2-x*x))/.5
    check_event(row.search.first_touch,truth,m,p,1/8)
    check_event(row.search.first_entry,truth,m,p,1/8,strict=True)
    assert row.contact.first_entry is None
    if kind=="grazing": check_event(row.contact.first_touch,.5,m,p,1/16)
    else: assert row.contact.first_touch is None


@pytest.mark.parametrize("end,enters",[(.25,False),(.03125,True)])
def test_initial_boundary_opening_or_closing_is_not_initial_force(end,enters):
    m,p,k=vt_mesh(1/16,end)
    with ph.CollisionWorkspace(m,"host") as w: q=lookup(sweep(m,p,w),k).contact
    assert q.start_relation=="boundary" and q.first_touch.interval==(0.,0.)
    assert (q.first_entry is not None)==enters
    if enters:
        assert q.first_entry.interval==(0.,0.) and q.first_entry.sample_time>0
        check_event(q.first_entry,0.,m,p,1/16,strict=True)


def test_subframe_window_missed_by_64_uniform_samples():
    z=17/64+1/2048
    r=1/65536
    m,p,k=vt_mesh(z,z-1)
    assert all(abs(z-i/64)>r for i in range(65))
    with ph.CollisionWorkspace(m,"host") as w: row=lookup(sweep(m,p,w,search=2*r,force=r),k)
    check_event(row.contact.first_touch,z-r,m,p,r)
    check_event(row.contact.first_entry,z-r,m,p,r,strict=True)


def test_connected_fold_trajectory_and_legal_hinge_reuse_material_topology(tmp_path):
    m,p,k=fold_mesh()
    with ph.CollisionWorkspace(m,"host") as w: result=sweep(m,p,w,search=3/32)
    assert result.status=="ok"
    row=lookup(result,k)
    check_event(row.search.first_touch,5/16,m,p,3/32)
    check_event(row.contact.first_entry,3/8,m,p,1/16,strict=True)
    # At t=13/32 these exact moving vertices reproduce the measured Newton free pose.
    original,base=register(CAT["DD-REMOTE-FOLD"])
    xy=geometry_at(m,p,13/32)
    for s in original:
        np.testing.assert_array_equal([xy[s.key.world,s.key.body,n] for n in s.vertex_ids],base[s.key].start)
    (tmp_path/"moving-fold.json").write_text(json.dumps(dict(paths=len(result.paths),broad=result.broad_face_pairs,
         tests=result.feature_tests,roots=result.root_nodes,force_entry=row.contact.first_entry.interval)),encoding="utf-8")
    shapes,h=register(CAT["DD-ADJACENT-EDGE"])
    model=ph.CollisionModel(shapes)
    with ph.CollisionWorkspace(model,"host") as w: legal=sweep(model,h,w)
    assert legal.status=="ok" and legal.paths==()


def test_degenerate_middle_or_endpoint_and_exhausted_proof_do_not_publish_clear():
    m,p,k=vt_mesh()
    a=m.shapes[0].key
    for end in ([[-1,-1,0],[1,-1,0],[0,-3,0]],[[-1,-1,0],[1,-1,0],[0,-1,0]]):
        bad={**p,a:ph.CollisionMotion(p[a].start,end)}
        with ph.CollisionWorkspace(m,"host") as w: out=sweep(m,bad,w)
        assert out.status=="unresolved" and out.reason=="face_degenerate_on_path" and out.paths==() and a in out.invalid_faces
    with ph.CollisionWorkspace(m,"host") as w:
        unknown=sweep(m,p,w,max_root_nodes=0)
        cap=sweep(m,p,w,feature_capacity=0)
    assert unknown.status=="unresolved" and unknown.reason=="root_capacity" and unknown.paths==()
    assert cap.status=="capacity_exceeded" and cap.paths==() and cap.required_features>0
