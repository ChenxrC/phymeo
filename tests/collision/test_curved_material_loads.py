"""Sample and advance real states through a smooth finite rigid contact layer."""

import json
from contextlib import ExitStack
from dataclasses import asdict
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


def test_angular_clearance_covers_small_real_arcs_and_refuses_hidden_half_turn():
    key, cloth=ph.CollisionKey(0,1,0),ph.CollisionKey(0,2,0)
    shape=ph.CollisionShape(key,"ellipsoid",(5,),curved=ph.CurvedSolid("ellipsoid",(.5,.25,.125)))
    sheet=ph.CollisionShape(cloth,"triangle",(4,9,11))
    center=(0,0,.25)
    triangle=ph.CollisionMotion([[-2,-1,0],[2,-1,0],[0,2,0]])
    model=ph.CollisionModel((sheet,shape))
    with ph.CollisionWorkspace(model) as work:
        def sample(angle):
            arc=ph.RigidTurn(center,center,(0,1,0),angle)
            motion=ph.CollisionMotion([center],turn=arc)
            pair,=ph.detect(model,{key:motion,cloth:triangle},ph.CollisionPolicy(margin=.5),work).pairs
            return pair

        clear=sample(.05)
        assert clear.status=="separated" and clear.reason=="rotational_support_clearance"
        assert clear.sweep.first_t is None
        end_gap=.25-hypot(.5*np.sin(.05),.125*np.cos(.05))
        assert clear.sweep.end_gap_m==pytest.approx(end_gap,abs=2e-5)
        hidden=sample(np.pi)
        assert hidden.status in ("unsupported","unresolved","hit")
        if hidden.status=="hit":
            assert hidden.sweep.first_t is not None
        # At half the arc, the long axis is vertical and passes through the sheet.
        assert .25-.5<0 and .25-.125>0


def spring_impact_window(mode, root):
    model, bindings, laws, initial, models = scene("RR", mode)
    p=initial.particles
    rigid=initial.rigid
    dtype=rigid.position.dtype
    pos=rigid.position.copy()
    pos[1,0]=.375+5/64
    velocity=rigid.velocity.copy()
    velocity[0,0]=.5
    state=ph.CoupledState(p,ph.RigidBodyState(pos,rigid.orientation,velocity,rigid.angular_velocity))
    dt=1/256
    phase=2*np.arcsin(np.sqrt(192.)*dt/2)
    history=[]
    with ExitStack() as stack:
        work=stack.enter_context(ph.CollisionWorkspace(model,mode,cache_root=root))
        op=None if mode=="host" else stack.enter_context(ph.TyxNormalLayer(mode,cache_root=root))
        sys=CollisionLoadSystem(model,bindings,laws,layer_operator=op)
        solver=ph.RigidSemiImplicitSolver() if mode=="host" else stack.enter_context(ph.TyxRigidSemiImplicitSolver(mode=mode,cache_root=root))
        ext=ph.CoupledInputs.zeros(0,2,dtype=dtype)
        for n in range(80):
            frame=sys.step(state,ext,models,(None,solver),dt,work)
            sites=frame.loads.contacts
            magnitude=sites[0].force_n if sites else 0.
            # Eight free steps reach the layer. In contact, the scalar relative oscillator
            # has exact discrete solution c[j]=(dt/2) sin(j*phase)/sin(phase).
            j=n-8
            compression=(dt/2)*np.sin(j*phase)/np.sin(phase) if 0<j*phase<np.pi else 0.
            assert magnitude==pytest.approx(64*compression,abs=8e-5)
            state=frame.state
            gap=float(state.rigid.position[1,0]-state.rigid.position[0,0]-.375)
            assert gap>0 and frame.path.pairs[0].status=="separated"
            v=state.rigid.velocity.astype(float)
            momentum=.5*v[0]+v[1]
            np.testing.assert_allclose(momentum,[.25,0,0],atol=5e-6,rtol=0)
            energy=.25*(v[0]@v[0])+.5*(v[1]@v[1])+32*max(1/16-gap,0.)**2
            assert abs(energy-.0625)<.003
            history.append(dict(step=n+1,contacts=[asdict(c) for c in sites],entry_force_n=magnitude,analytic_entry_force_n=64*compression,
                end_gap_m=gap,end_positions_m=state.rigid.position.tolist(),end_velocity_m_s=v.tolist(),
                momentum_kg_m_s=momentum.tolist(),mechanical_energy_j=energy))
        assert history[7]["entry_force_n"]==0 and history[-1]["entry_force_n"]==0
        assert max(row["entry_force_n"] for row in history)>2.2
        assert state.rigid.velocity[1,0] > state.rigid.velocity[0,0]
        result=dict(mode=mode,step_s=dt,family="RR",rows=history,workspace=work.stats(),layer=op.stats() if op else None)
        (root/"curved-impact-window.json").write_text(json.dumps(result,allow_nan=False),encoding="utf8")


def sliding_cloth_window(mode, root):
    model,bindings,laws,initial,models=scene("RD",mode)
    laws=dict.fromkeys(laws,NormalLayer(1/16,64.,friction=.25))
    p0=initial.particles
    v0=p0.velocity.copy(); v0[:,1]=.25
    state=ph.CoupledState(ph.ParticleState(p0.position,v0),initial.rigid)
    dtype=v0.dtype
    dt=1/256
    impulse=np.array((0.,.75,0.))
    edges=np.array([[0,1],[1,2],[2,0]])
    rest=np.linalg.norm(p0.position[edges[:,0]]-p0.position[edges[:,1]],axis=1)
    constraints=ph.DistanceConstraints(edges,rest,np.full(3,.01,dtype=dtype))
    history=[]
    with ExitStack() as stack:
        work=stack.enter_context(ph.CollisionWorkspace(model,mode,cache_root=root))
        op=None if mode=="host" else stack.enter_context(ph.TyxNormalLayer(mode,cache_root=root))
        sys=CollisionLoadSystem(model,bindings,laws,layer_operator=op)
        if mode=="host":
            rigid_solver=ph.RigidSemiImplicitSolver()
            cloth_solver=ph.XPBDSolver(constraints=constraints,tolerance=5e-5)
        else:
            rigid_solver=stack.enter_context(ph.TyxRigidSemiImplicitSolver(mode=mode,cache_root=root))
            cloth_solver=stack.enter_context(ph.TyxXPBDSolver(constraints,mode=mode,tolerance=5e-5,cache_root=root))
        for n in range(48):
            ext=ph.CoupledInputs.zeros(3,1,dtype=dtype)
            if 8<=n<16:
                ext.rigid.force[0,1]=.25
            impulse+=dt*ext.rigid.force[0].astype(float)
            frame=sys.step(state,ext,models,(cloth_solver,rigid_solver),dt,work)
            got=frame.loads.contacts
            force=frame.loads.inputs.rigid.force[0].astype(float)
            if not n:
                contact,=got
                assert contact.force_n==pytest.approx(2,abs=3e-5)
                np.testing.assert_allclose(force,[0,.5,2],atol=1e-4,rtol=0)
                assert contact.power_w==pytest.approx(-.125,abs=2e-5)
                site,=contact.sites
                assert site.application_point_first_m==site.application_point_second_m
                torque=np.cross(np.asarray(site.application_point_first_m)-state.rigid.position[0],force)
                np.testing.assert_allclose(frame.loads.inputs.rigid.torque[0],torque,atol=2e-6)
            state=frame.state
            measured=.5*state.rigid.velocity[0]+state.particles.velocity.sum(axis=0)
            assert frame.path.pairs[0].status=="separated"
            np.testing.assert_allclose(measured,impulse,atol=5e-4 if mode!="host" else 1e-8,rtol=0)
            history.append(dict(step=n+1,contacts=[asdict(c) for c in got],external_force_n=ext.rigid.force.tolist(),entry_normal_n=got[0].force_n if got else 0.,
                contact_power_w=got[0].power_w if got else 0.,rigid_force_n=force.tolist(),
                rigid_torque_nm=frame.loads.inputs.rigid.torque[0].tolist(),end_q_xyzw=state.rigid.orientation.tolist(),
                end_omega_rad_s=state.rigid.angular_velocity.tolist(),cloth_positions_m=state.particles.position.tolist(),
                cloth_velocities_m_s=state.particles.velocity.tolist(),rigid_velocities_m_s=state.rigid.velocity.tolist(),
                end_linear_momentum=measured.tolist(),expected_linear_momentum=impulse.tolist()))
        assert np.linalg.norm(state.rigid.angular_velocity[0])>0.02
        assert history[-1]["entry_normal_n"]==0
        result=dict(mode=mode,step_s=dt,family="RD",rows=history,workspace=work.stats(),layer=op.stats() if op else None)
        (root/"curved-sliding-window.json").write_text(json.dumps(result,allow_nan=False),encoding="utf8")


@pytest.mark.parametrize("run",[spring_impact_window,sliding_cloth_window])
def test_curved_contact_windows_transfer_real_load_and_unload(run,tmp_path):
    run("host",tmp_path)


@pytest.mark.tyx
@pytest.mark.parametrize("run",[spring_impact_window,sliding_cloth_window])
def test_curved_contact_windows_on_native_participants(run,tmp_path,tyx_mode):
    run(tyx_mode,tmp_path)
