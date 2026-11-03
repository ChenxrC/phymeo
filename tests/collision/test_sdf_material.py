"""Discrete material observations have independent analytic point, chart and coverage oracles."""
import numpy as np
import pytest

import phymeo as ph


def shape(kind, ids, *, world=0, body=1, index=0, radius=0, mask=2**31-1):
    return ph.CollisionShape(ph.CollisionKey(world,body,index),kind,ids,radius=radius,mask=mask)


def grid(values, *, origin=(0.,0.,0.), step=(1.,1.,1.), dims=(2,2,2), band=None):
    z,y,x=np.indices(dims[::-1], dtype=float)
    v=values(origin[0]+step[0]*x,origin[1]+step[1]*y,origin[2]+step[2]*z)
    return ph.SignedDistanceGrid(np.broadcast_to(v,dims[::-1]),origin_m=origin,spacing_m=step,band_m=band)


def setup(field, *, pose=None, solids=(), materials=(), excluded=()):
    rigid=shape("sphere",(70,),body=0,radius=1)
    model=ph.CollisionModel((rigid,*solids,*materials),excluded)
    frame=ph.SdfFieldPose(rigid.key,field,**(pose or {}))
    return model,frame


def one(batch):
    assert len(batch.results)==1
    return batch.results[0]


def check_material(row,shape,points):
    by_id=dict(zip(shape.vertex_ids,points,strict=True))
    for obs in row.samples:
        assert tuple(sorted(obs.weights))==obs.weights
        assert sum(w for _,w in obs.weights)==pytest.approx(1,abs=2e-14)
        assert all(w>0 for _,w in obs.weights)
        p=sum(w*np.array(by_id[i]) for i,w in obs.weights)
        np.testing.assert_allclose(obs.world_point_m,p,rtol=0,atol=4e-14)
        assert not hasattr(obs,"mesh_face") and not hasattr(obs,"rigid_weights")


def analytic_triangle():
    # On x+y+z=1, xy+xz+yz is at most 1/3. On each edge it is at most 1/4.
    g=grid(lambda x,y,z:5/16-x*y-y*z-z*x)
    s=shape("triangle",(47,5,29))
    p=np.eye(3)
    return g,s,p


def test_nonvertex_interior_contact_with_positive_entire_boundary():
    g,s,p=analytic_triangle()
    model,f=setup(g,materials=(s,))
    row=one(ph.sample_sdf_contacts(model,f,{s.key:p}))
    assert row.field_status=="contact" and row.source_geometry_status=="unbound"
    assert row.field_gap_bounds_m[0]<=-1/48<=row.field_gap_bounds_m[1]+1e-15
    inner=min((v for v in row.samples if v.status=="valid"),key=lambda q:q.field_gap_m)
    assert inner.field_gap_m==pytest.approx(-1/48,abs=2e-15)
    np.testing.assert_allclose(inner.world_point_m,[1/3]*3,atol=1e-15)
    assert [i for i,_ in inner.weights]==[5,29,47]
    np.testing.assert_allclose(inner.gradient_world,[-2/3]*3,atol=2e-15)
    assert inner.projected_surface_estimate_world_m is not None
    check_material(row,s,p)


def test_exact_triangle_clear_requires_a_surface_bound_and_budget_is_explicit():
    s=shape("triangle",(1,5,9))
    p=np.eye(3)
    g=grid(lambda x,y,z:13/32-x*y-y*z-z*x)
    model,f=setup(g,materials=(s,))
    whole=one(ph.sample_sdf_contacts(model,f,{s.key:p},max_depth=3))
    assert whole.field_status=="clear"
    exact=13/32-1/3
    assert 0<whole.field_gap_bounds_m[0]<=exact<=whole.field_gap_bounds_m[1]+1e-15
    loose=one(ph.sample_sdf_contacts(model,f,{s.key:p},max_depth=0))
    assert loose.field_status=="unresolved" and "field_precision" in loose.reasons
    limited=one(ph.sample_sdf_contacts(model,f,{s.key:p},max_samples=1))
    assert limited.field_status=="unresolved" and "sample_capacity" in limited.reasons


def test_discrete_radius_world_filters_masks_and_projection_are_decoupled():
    g=grid(lambda x,y,z:x-0.5)
    a=shape("particle",(7,),index=0)
    b=shape("rounded_particle",(7,),index=1,radius=.25)
    far=shape("rounded_particle",(7,),world=1,radius=.25)
    masked=shape("particle",(7,),body=2,mask=0)
    excluded=shape("particle",(7,),body=3)
    rigid=shape("sphere",(70,),body=0,radius=1)
    model,f=setup(g,materials=(a,b,far,masked,excluded),excluded=((rigid.key,excluded.key),))
    at=[[.625,.5,.5]]
    results=ph.sample_sdf_contacts(model,f,{s.key:at for s in (a,b,far,masked,excluded)})
    rows={r.material_key:r for r in results.results}
    assert rows[a.key].field_status=="clear" and rows[b.key].field_status=="contact"
    q=rows[b.key].samples[0]
    assert q.field_distance_m==.125 and q.field_gap_m==-.125 and q.weights==((7,1.),)
    assert q.projected_surface_estimate_world_m==(.5,.5,.5)
    assert rows[far.key].reasons==("different_world",) and rows[masked.key].reasons==("mask",)
    assert rows[excluded.key].reasons==("excluded",) and not rows[far.key].samples
    assert ph.sample_sdf_contacts(model,f,{}).results==()
    with pytest.raises((TypeError,ValueError),match="discrete"):
        ph.sample_sdf_contacts(model,f,{a.key:ph.CollisionMotion(at,[[.125,.5,.5]])})


def test_partial_domain_and_band_do_not_certify_whole_face_clear():
    a=shape("triangle",(1,5,8))
    p=np.array([[-.5,.5,.5],[.75,.25,.5],[.75,.75,.5]])
    g=grid(lambda x,y,z:x+1)
    model,f=setup(g,materials=(a,))
    partial=one(ph.sample_sdf_contacts(model,f,{a.key:p}))
    assert partial.field_status=="unresolved" and "outside_grid" in partial.reasons
    assert partial.field_gap_bounds_m[0] is None
    _,neg=setup(grid(lambda x,y,z:x-.25),materials=(a,))
    hit=one(ph.sample_sdf_contacts(model,neg,{a.key:p}))
    assert hit.field_status=="contact" and "outside_grid" in hit.reasons
    assert hit.field_gap_bounds_m[0] is None and hit.field_gap_bounds_m[1]<0
    _,band=setup(grid(lambda x,y,z:x-.5,band=.45),materials=(a,))
    cut=one(ph.sample_sdf_contacts(model,band,{a.key:p+.125}))
    assert cut.field_status=="unresolved" and "outside_band" in cut.reasons
    outside=one(ph.sample_sdf_contacts(model,f,{a.key:p+[2,0,0]}))
    assert outside.field_status=="unresolved" and outside.field_gap_bounds_m==(None,None)


def test_multicell_edge_feature_full_pose_and_material_reordering():
    g=grid(lambda x,y,z:np.maximum.reduce([np.abs(x),np.abs(y),np.abs(z)])-.25,
           origin=(-1,-1,-1),step=(.25,.25,.5),dims=(9,9,5))
    s=shape("triangle",(29,3,17))
    p=np.array([[-1,0,0],[1,0,0],[1,1,.5]])
    model,f=setup(g,materials=(s,))
    row=one(ph.sample_sdf_contacts(model,f,{s.key:p}))
    assert row.field_status=="contact" and row.cells_covered>1
    edge=next(q for q in row.samples if np.array_equal(q.local_point_m,(0,0,0)))
    assert edge.weights==((3,.5),(29,.5)) and edge.field_distance_m==-.25
    turned=shape("triangle",(17,3,29))
    shift=np.array([2.,-3.,.5])
    pose=dict(translation_m=shift,orientation_xyzw=(.5,.5,.5,.5))
    m2,f2=setup(g,materials=(turned,),pose=pose)
    wp=np.roll(p[::-1],1,axis=1)+shift
    other=one(ph.sample_sdf_contacts(m2,f2,{turned.key:wp}))
    assert other.field_status=="contact" and other.field_gap_bounds_m==row.field_gap_bounds_m
    same=next(q for q in other.samples if q.weights==edge.weights)
    np.testing.assert_allclose(same.gradient_world,np.roll(edge.gradient_world,1),atol=1e-14)
    check_material(other,turned,wp)
    capped=one(ph.sample_sdf_contacts(model,f,{s.key:p},max_cells=1))
    assert capped.field_status=="unresolved" and "cell_capacity" in capped.reasons


def test_baked_source_requires_registered_topology_vertices_and_pose():
    from test_sdf_grid import asset
    mesh,v=asset("u")
    s=shape("triangle",(33,15,21),body=1)
    model=ph.CollisionModel((mesh,s))
    frame=ph.SdfFieldPose.bake(mesh,v,shape_xyz=(13,13,7),origin_m=(-3,-3,-1.5),spacing_m=(.5,.5,.5),
                               translation_m=(2,-1,.5),orientation_xyzw=(.5,.5,.5,.5))
    tx=lambda p:np.roll(p,1,axis=1)+[2,-1,.5]
    rigid=tx(v)
    p=tx(np.array([[-.125,0,0],[.125,0,0],[0,.25,0.]]))
    row=one(ph.sample_sdf_contacts(model,frame,{s.key:p},rigid_vertices_world_m=rigid))
    assert row.field_status==row.source_geometry_status=="clear" and row.source_gap_bounds_m[0]>0
    no_rigid=one(ph.sample_sdf_contacts(model,frame,{s.key:p}))
    assert no_rigid.field_status=="clear" and no_rigid.source_geometry_status=="pose_unverified"
    alternate=rigid.copy();alternate[0,0]+=.03125
    mismatch=one(ph.sample_sdf_contacts(model,frame,{s.key:p},rigid_vertices_world_m=alternate))
    assert mismatch.source_geometry_status=="pose_unverified" and mismatch.source_gap_bounds_m==(None,None)
    wrong=ph.CollisionShape(mesh.key,"mesh",mesh.vertex_ids,triangles=mesh.triangles,revision=1)
    changed=one(ph.sample_sdf_contacts(ph.CollisionModel((wrong,s)),frame,{s.key:p},rigid_vertices_world_m=rigid))
    assert changed.source_geometry_status=="pose_unverified"
    straight=ph.SdfFieldPose(mesh.key,frame.grid,translation_m=frame.translation_m,orientation_xyzw=frame.orientation_xyzw)
    assert one(ph.sample_sdf_contacts(model,straight,{s.key:p},rigid_vertices_world_m=rigid)).source_geometry_status=="unbound"


def test_coarse_thin_wall_may_be_clear_in_field_and_uncertain_for_source():
    from test_sdf_grid import asset
    mesh,v=asset("thin")
    q=shape("particle",(91,))
    m=ph.CollisionModel((mesh,q))
    f=ph.SdfFieldPose.bake(mesh,v,shape_xyz=(9,9,9),origin_m=(-1,-1,-1),spacing_m=(.25,)*3)
    row=one(ph.sample_sdf_contacts(m,f,{q.key:[[.125,0,0]]},rigid_vertices_world_m=v))
    assert row.field_status=="clear" and row.samples[0].field_distance_m==7/64
    assert row.samples[0].gradient_world==(0,0,0) and row.samples[0].projected_surface_estimate_world_m is None
    assert row.source_geometry_status=="unresolved"
    lo,hi=row.source_gap_bounds_m
    assert lo <= -1/64 <= hi


def test_degenerate_material_and_nondiscrete_or_unknown_keys():
    s=shape("triangle",(1,2,3))
    t=shape("triangle",(1,2,3),body=2,radius=.125)
    g=grid(lambda x,y,z:x)
    m,f=setup(g,materials=(s,t))
    p=np.array([[0,0,0],[0,0,.5],[0,0,1.]])
    result=ph.sample_sdf_contacts(m,f,{s.key:p,t.key:p})
    assert result.results[0].reasons==("degenerate_triangle",)
    assert result.results[1].field_status=="unsupported"
    with pytest.raises(ValueError,match="registered"):
        ph.sample_sdf_contacts(m,f,{ph.CollisionKey(0,9,0):p})
