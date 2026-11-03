"""Actual two-sided particle force/power, canonical sites, and an accepted dynamics step."""
from dataclasses import replace
import json

import numpy as np
import pytest
import phymeo as ph
from test_deformable_features import CAT,key,query,register


def scene(id_,*,dtype=np.float64,reverse=False):
    case=CAT[id_]
    shapes,paths=register(case)
    lookup={}
    for shape in shapes:
        for v,p in zip(shape.vertex_ids,paths[shape.key].start,strict=True):
            lookup[(shape.key.world,shape.key.body,v)]=p
    nodes=sorted(lookup,reverse=reverse)
    offsets={n:i for i,n in enumerate(nodes)}
    bindings={s.key:ph.ParticleAttachment(tuple(offsets[s.key.world,s.key.body,n] for n in s.vertex_ids)) for s in shapes}
    xyz=np.array([lookup[n] for n in nodes],dtype=dtype)
    return ph.CollisionModel(shapes),bindings,ph.ParticleState(xyz,np.zeros_like(xyz)),nodes,paths


def system(model,bindings,**kw):
    return ph.DeformableFeatureLoadSystem(model,bindings,kw.pop("layer",ph.NormalLayer(1/16,128.)),search_distance_m=3/32,**kw)


def force_map(nodes,nodal):
    return {n[2]:f for n,f in zip(nodes,nodal,strict=True)}


def conservation(q,state,tol=1e-11):
    np.testing.assert_allclose(q.nodal_force_n.sum(0),0.,rtol=0,atol=tol)
    np.testing.assert_allclose(np.cross(state.position,q.nodal_force_n).sum(0),0.,rtol=0,atol=tol)
    np.testing.assert_allclose((state.velocity*q.nodal_force_n).sum(),q.power_w,rtol=0,atol=tol)
    assert q.power_w==pytest.approx(sum(x.power_w for x in q.contacts),abs=tol)


@pytest.mark.parametrize("id_",["DD-VT-TWO-SHEETS","DD-EE-TWO-SHEETS","DD-EE-MIRROR-SKEW","DD-VT-DEEP-LAW"])
def test_material_feature_scatter_matches_independent_physical_forces(id_):
    m,b,state,nodes,paths=scene(id_)
    with ph.CollisionWorkspace(m,"host") as work:
        geo=query(m,paths,work)
        s=system(m,b)
        result=s.sample_particles(state,work,features=geo)
    assert s.last_sample is result and result.geometry is geo and len(result.contacts)==1
    exp=CAT[id_]["expected"]
    magnitude=exp["phymeo_unique_force_n"]
    actual=force_map(nodes,result.nodal_force_n)
    expected={n[2]:np.zeros(3) for n in nodes}
    for sign,weights in ((-1,exp["focus_weights_a"]),(1,exp["focus_weights_b"])):
        for i,w in weights: expected[i][2]=sign*w*magnitude
    np.testing.assert_allclose([actual[i] for i in sorted(actual)],[expected[i] for i in sorted(actual)],atol=1e-12)
    assert result.contacts[0].force_n==magnitude and result.power_w==0.
    assert result.potential_j==pytest.approx(.5*128*(1/16-exp["focus_distance_m"])**2,abs=1e-15)
    conservation(result,state)


def test_fold_publishes_si_nodal_force_and_does_not_change_face_pair_calibration(tmp_path):
    m,b,state,nodes,_=scene("DD-REMOTE-FOLD")
    records=[]
    for reverse in (False,True):
        model,bind,st,n,_=scene("DD-REMOTE-FOLD",reverse=reverse)
        s=system(model,bind)
        with ph.CollisionWorkspace(model,"host") as work:
            q=s.sample_particles(st,work)
            layers={(a.key,c.key):ph.NormalLayer(1/16,128.) for i,a in enumerate(model.shapes) for c in model.shapes[i+1:]}
            default=ph.CollisionLoadSystem(model,bind,layers).sample_particles(st,work)
        fm=force_map(n,q.nodal_force_n)
        f0=force_map(n,default.inputs.particles.force)
        assert len(q.contacts)==32 and sum(x.feature.kind=="VT" for x in q.contacts)==8
        assert sum(len(x.sites)==2 for x in q.contacts)==6
        assert all(abs(sum(site.fraction for site in x.sites)-1)<1e-15 for x in q.contacts)
        assert all(x.force_n==2 for x in q.contacts)
        assert sum(fm[i][2] for i in range(100,106))==pytest.approx(-64,abs=1e-12)
        assert sum(fm[i][2] for i in range(112,118))==pytest.approx(64,abs=1e-12)
        assert sum(f0[i][2] for i in range(100,106))==pytest.approx(-18,abs=1e-12)
        assert q.potential_j==.5
        conservation(q,st)
        records.append({str(i):v.tolist() for i,v in fm.items()})
    assert records[0]==records[1]
    with pytest.raises(ValueError):
        q.nodal_force_n.setflags(write=True)
    payload=dict(physical_nodal_n=records[0],default_lower_n=-18.,feature_lower_n=-64.,stored_energy_j=q.potential_j,sites=sum(len(x.sites) for x in q.contacts))
    (tmp_path/"fold-feature-forces.json").write_text(json.dumps(payload,allow_nan=False),encoding="utf-8")


def test_velocity_damping_power_uses_the_two_addressed_material_velocities():
    m,b,state,nodes,paths=scene("DD-VT-TWO-SHEETS")
    vel=state.velocity.copy()
    for i,v in {11:1/8,23:3/8,37:-1/8,53:-3/8}.items(): vel[next(k for k,n in enumerate(nodes) if n[2]==i),2]=v
    vel[:,0]=[1,2,-3,6,7,8]
    current=ph.ParticleState(state.position,vel)
    s=system(m,b,layer=ph.NormalLayer(1/16,128.,damping_s_per_m=.5))
    with ph.CollisionWorkspace(m,"host") as work:
        geo=query(m,paths,work)
        result=s.sample_particles(current,work,features=geo)
    load,=result.contacts
    site,=load.sites
    # VT first is the tip, pointing downward to the surface. The stored normal
    # rate is closing; source velocities are still those of the original nodes.
    assert site.first_velocity_m_s[2]==-3/8 and site.second_velocity_m_s[2]==1/16
    assert site.rate_m_per_s==-7/16
    assert load.force_n==2*(1+7/32)
    assert result.power_w==load.force_n*(-7/16)
    assert result.potential_j==1/64
    np.testing.assert_allclose(site.force_on_first_n,-np.asarray(site.force_on_second_n),atol=0)
    conservation(result,current)


def test_external_feature_binding_rejects_stale_shapes_and_preserves_last_sample():
    m,b,state,nodes,paths=scene("DD-VT-TWO-SHEETS")
    s=system(m,b)
    with ph.CollisionWorkspace(m,"host") as work:
        good=query(m,paths,work)
        published=s.sample_particles(state,work,features=good)
        pose=state.position.copy()
        pose[next(i for i,n in enumerate(nodes) if n[2]==67),1]+=.125
        with pytest.raises(ph.SolverFailure,match="stale"):
            s.sample_particles(ph.ParticleState(pose,state.velocity),work,features=good)
        for invalid in (query(m,paths,work,force=1/32),query(m,paths,work,topology_rings=1),replace(good,contacts=())):
            with pytest.raises(ph.SolverFailure,match="stale"):
                s.sample_particles(state,work,features=invalid)
        newmodel=ph.CollisionModel(m.shapes)
        with ph.CollisionWorkspace(newmodel,"host") as other:
            foreign=query(newmodel,paths,other)
            with pytest.raises(ph.SolverFailure,match="stale"):
                s.sample_particles(state,work,features=foreign)
        failed=query(m,paths,work,feature_capacity=0)
        with pytest.raises(ph.SolverFailure,match="capacity"):
            s.sample_particles(state,work,features=failed)
        coincident=state.position.copy()
        coincident[next(i for i,n in enumerate(nodes) if n[2]==53),2]=0.
        with pytest.raises(ph.SolverFailure,match="normal"):
            s.sample_particles(ph.ParticleState(coincident,state.velocity),work)
        assert s.last_sample is published


def dynamic(s,model,reg,state,nodes,work,solver):
    h=1/256
    pmodel=ph.ParticleModel(np.ones(len(nodes),dtype=state.position.dtype),np.zeros(3,dtype=state.position.dtype))
    first=s.sample_particles(state,work)
    inputs=first.step_inputs()
    assert inputs.force.dtype==state.position.dtype
    step=solver.step(pmodel,state,inputs,h)
    assert step.report.accepted
    v=state.velocity+h*inputs.force
    np.testing.assert_allclose(step.state.velocity,v,rtol=0,atol=2e-7)
    np.testing.assert_allclose(step.state.position,state.position+h*v,rtol=0,atol=2e-7)
    second=s.sample_particles(step.state,work)
    f0,f1=first.contacts[0],second.contacts[0]
    assert 0.046875<f1.feature.distance_m<0.04694 and f1.force_n<f0.force_n
    kinetic=float(np.sum(step.state.velocity.astype(float)**2)/2)
    assert kinetic==pytest.approx(2.75*h*h,abs=2e-11)
    assert 0<second.potential_j<first.potential_j and first.power_w==0
    return dict(backend=step.report.backend,initial_q=state.position.tolist(),final_q=step.state.position.tolist(),final_v=step.state.velocity.tolist(),
                initial_stored_j=first.potential_j,final_stored_j=second.potential_j,kinetic_j=kinetic,external_work_j=0.,
                first_gap_m=f0.feature.distance_m,end_gap_m=f1.feature.distance_m,initial_force_n=f0.force_n,end_force_n=f1.force_n)


def test_original_host_solver_advances_both_sides_then_queries_the_new_pose(tmp_path):
    m,b,state,nodes,_=scene("DD-VT-TWO-SHEETS")
    s=system(m,b)
    with ph.CollisionWorkspace(m,"host") as work:
        obs=dynamic(s,m,b,state,nodes,work,ph.SemiImplicitSolver())
    (tmp_path/"feature-step.json").write_text(json.dumps(obs,allow_nan=False),encoding="utf-8")


def test_capacity_at_new_pose_cannot_replace_previous_clear_observation():
    m,b,state,nodes,_=scene("DD-REMOTE-FOLD")
    s=system(m,b,feature_capacity=31)
    # A simple open, planar strip with the same registered material graph is far
    # from its nonlocal triangles. Restore the original fold to require 32 rows.
    unrolled=np.array([[float((n[2]-100)//2),float(n[2]%2),0.] for n in nodes])
    with ph.CollisionWorkspace(m,"host") as work:
        previous=s.sample_particles(ph.ParticleState(unrolled,state.velocity),work)
        assert not previous.contacts
        with pytest.raises(ph.SolverFailure,match="capacity"):
            s.sample_particles(state,work)
        assert s.last_sample is previous and np.count_nonzero(previous.nodal_force_n)==0
