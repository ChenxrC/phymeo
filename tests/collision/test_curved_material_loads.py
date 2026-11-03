"""Sample and advance real states through a smooth finite rigid contact layer."""

from contextlib import ExitStack
from math import hypot

import numpy as np
import pytest

import phymeo as ph
from phymeo.core.capabilities import SolverFailure
from phymeo.solvers.collision_loads import CollisionLoadSystem, NormalLayer, ParticleAttachment, RigidAttachment


def test_registered_curve_uses_the_body_start_frame_for_its_bounds_and_witness():
    a, b = ph.CollisionKey(6,1,0), ph.CollisionKey(6,2,0)
    # Local axes cycle x -> y -> z; the body also rotates, about y.
    asset=ph.CurvedSolid("ellipsoid",(.375,.25,.125),(.5,.5,.5,.5))
    shape=ph.CollisionShape(a,"ellipsoid",(17,),curved=asset)
    plane=ph.CollisionShape(b,"plane",(33,),normal=[0,0,1])
    model=ph.CollisionModel((shape,plane))
    cy,sy=7/25,24/25
    # World z projected into the authored shape frame is (0,cy,-sy).
    extent=hypot(.25*cy,.125*sy)
    center=np.array([.25,-.5,.375])
    pa=ph.CollisionMotion([center],body_orientation_xyzw=(0,.6,0,.8))
    with ph.CollisionWorkspace(model) as work:
        res=ph.detect(model,{a:pa,b:ph.CollisionMotion([[0,0,0]])},ph.CollisionPolicy(margin=.75),work)
    pair,=res.pairs
    assert pair.witness.gap_m==pytest.approx(center[2]-extent,abs=1e-12)
    assert pair.witness.first_point[2]==pytest.approx(center[2]-extent,abs=1e-12)
    lower=next(q.lower for q in res.bounds if q.key==a)
    assert lower[2]<=center[2]-extent-.375 and lower[2]>center[2]-extent-.375-2e-6
    with pytest.raises(ValueError):
        ph.CollisionMotion([center],body_orientation_xyzw=(0,0,0,2))


def scene(family,mode):
    dtype=np.float64 if mode=="host" else np.float32
    a,b=ph.CollisionKey(6,1,0),ph.CollisionKey(6,2,0)
    law=NormalLayer(1/16,64.)
    if family=="RR":
        solids=[ph.CurvedSolid("ellipsoid", dims) for dims in ((.25,.125,.125),(.125,.25,.125))]
        geom=[ph.CollisionShape(key,"ellipsoid",(17,),curved=solid) for key,solid in zip((a,b),solids,strict=True)]
        bnd={a:RigidAttachment(0,((0,0,0),)),b:RigidAttachment(1,((0,0,0),))}
        middle=[]; positions=[[0,0,0],[.375+1/32,0,0]]; orientation=[[0,0,0,1]]*2
    else:
        sa=ph.CollisionShape(a,"ellipsoid",(17,),curved=ph.CurvedSolid("ellipsoid",(.375,.25,.125)))
        sb=ph.CollisionShape(b,"triangle",(91,71,38))
        geom=[sa,sb]
        bnd={a:RigidAttachment(0,((0,0,0),)),b:ParticleAttachment((0,1,2))}
        middle=[[-2,-1,0],[2,-1,0],[0,2,0]]
        reach=hypot(.375*(24/25),.125*(7/25))
        side=(.375**2-.125**2)*(24/25)*(7/25)/reach
        positions=[[-side,0,reach+1/32]]
        orientation=[[0,.6,0,.8]]
    particles=ph.ParticleState(np.array(middle,dtype=dtype).reshape(-1,3),np.zeros((len(middle),3),dtype=dtype))
    rigid=ph.RigidBodyState(np.array(positions,dtype=dtype),np.array(orientation,dtype=dtype),
                            np.zeros((len(positions),3),dtype=dtype),np.zeros((len(positions),3),dtype=dtype))
    pm=ph.ParticleModel(np.ones(len(middle),dtype=dtype),np.zeros(3,dtype=dtype))
    inv=np.array([2,1] if family=="RR" else [2],dtype=dtype)
    rm=ph.RigidBodyModel(inv,np.asarray([[.25,.125,.25]]*len(inv),dtype=dtype),np.zeros(3,dtype=dtype))
    model=ph.CollisionModel(geom)
    return model,bnd,{(a,b):law},ph.CoupledState(particles,rigid),(pm,rm)


def window(family,mode,cache):
    m,bnd,laws,state,models=scene(family,mode)
    p0,r0=state.particles,state.rigid
    dt=1/256
    npart,nrigid=len(p0.position),len(r0.position)
    external=ph.CoupledInputs.zeros(npart,nrigid,dtype=p0.position.dtype)
    limits=dict(atol=3e-5 if mode!="host" else 5e-6,rtol=0)
    with ExitStack() as stack:
        work=stack.enter_context(ph.CollisionWorkspace(m,mode,cache_root=cache))
        constraints=None
        if npart:
            edges=np.array([[0,1],[1,2],[2,0]])
            rest=np.linalg.norm(p0.position[edges[:,0]]-p0.position[edges[:,1]],axis=1)
            constraints=ph.DistanceConstraints(edges,rest,np.full(3,.01,dtype=p0.position.dtype))
        if mode=="host":
            ris=ph.RigidSemiImplicitSolver()
            ps=ph.XPBDSolver(constraints=constraints,tolerance=5e-5) if npart else None
            layer=None
        else:
            ris=stack.enter_context(ph.TyxRigidSemiImplicitSolver(mode=mode,cache_root=cache))
            ps=stack.enter_context(ph.TyxXPBDSolver(constraints,mode=mode,tolerance=5e-5,cache_root=cache)) if npart else None
            layer=stack.enter_context(ph.TyxNormalLayer(mode,cache_root=cache))
        system=CollisionLoadSystem(m,bnd,laws,layer_operator=layer)
        for i in range(8):
            frame=system.step(state,external,models,(ps,ris),dt,work)
            contact,=frame.loads.contacts
            assert frame.path.pairs[0].status=="separated" and frame.path.pairs[0].sweep.first_t is None
            assert contact.force_n>0 and system.last_frame is frame
            if not i:
                assert contact.force_n==pytest.approx(2.,abs=3e-5)
                if family=="RR":
                    np.testing.assert_allclose(frame.state.rigid.velocity,[[-1/64,0,0],[1/128,0,0]],**limits)
                else:
                    side=-r0.position[0,0]
                    np.testing.assert_allclose(frame.loads.inputs.rigid.torque[0],[0,-2*side,0],atol=5e-4,rtol=0)
                    np.testing.assert_allclose(frame.state.rigid.angular_velocity[0],[0,-16*side*dt,0],atol=4e-5,rtol=0)
                    np.testing.assert_allclose(frame.state.rigid.velocity[0],[0,0,1/64],**limits)
                    np.testing.assert_allclose(frame.state.particles.velocity[:,2],[-2*dt/3]*3,**limits)
            state=frame.state
            total=(state.rigid.velocity/models[1].inverse_mass[:,None]).sum(axis=0)+state.particles.velocity.sum(axis=0)
            np.testing.assert_allclose(total,np.zeros(3),atol=8e-5 if mode!="host" else 1e-9,rtol=0)
        if mode!="host":
            assert work.stats()["completed_launches"]>0 and layer.stats()["sites"]>=8


@pytest.mark.parametrize("family",["RR","RD"])
def test_smooth_rigid_layers_drive_both_original_owners(family,tmp_path):
    window(family,"host",tmp_path)


@pytest.mark.tyx
@pytest.mark.parametrize("family",["RR","RD"])
def test_smooth_rigid_layers_use_real_tyx_force_and_integrators(family,tyx_mode,tmp_path):
    window(family,tyx_mode,tmp_path)


def test_curved_trial_crossing_aborts_and_preserves_last_publication(tmp_path):
    m,bnd,laws,state,models=scene("RR","host")
    system=CollisionLoadSystem(m,bnd,laws)
    ext=ph.CoupledInputs.zeros(0,2)
    with ph.CollisionWorkspace(m) as work:
        good=system.step(state,ext,models,(None,ph.RigidSemiImplicitSolver()),1/256,work)
        before=work.last_result
        bad=ph.RigidBodyState(good.state.rigid.position,good.state.rigid.orientation,np.array([[0.,0.,0.],[-32.,0.,0.]]),
                              good.state.rigid.angular_velocity)
        failed=ph.CoupledState(good.state.particles,bad)
        with pytest.raises(SolverFailure):
            system.step(failed,ext,models,(None,ph.RigidSemiImplicitSolver()),1/32,work)
        assert system.last_frame is good and work.last_result is before
