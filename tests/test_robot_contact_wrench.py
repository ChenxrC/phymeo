"""Signed per-side force, friction and moment aggregated from real published surface and link loads."""

import json
from dataclasses import replace

import numpy as np
import pytest

import phymeo as ph
from phymeo.sensors.contact_wrench import ContactSelector, measure_cloth_link_wrench, measure_contact_wrench
from phymeo.solvers.collision_loads import CollisionLoadSystem, NormalLayer, ParticleAttachment, RigidAttachment
from test_cloth_link_contact import make_scene


def layer_fixture(kind="sphere",*,reverse=False,worlds=1,friction=0.):
    shapes=[];bindings={};laws={}; rigid=[]; angular=[]; linear=[];nodes=[];keys=[]
    for world in range(worlds):
        z=0.25+world/32
        row=len(rigid)
        rigid.append([0,0,z])
        linear.append([1 if friction and kind=="sphere" else 0,0,0])
        angular.append([0,0,2 if kind=="capsule" else 0])
        sphere_body,cloth_body=(30,20) if reverse else (10,20)
        material=ph.CollisionKey(world,sphere_body,3)
        sheet=ph.CollisionKey(world,cloth_body,5)
        vertices=((-0.5,0.,0.),(0.5,0.,0.)) if kind=="capsule" else ((0.25,-0.125,0.),)
        shapes.append(ph.CollisionShape(material,kind,tuple(range(len(vertices))),radius=0.125))
        bindings[material]=RigidAttachment(row,vertices)
        ids=tuple(range(len(nodes),len(nodes)+3))
        nodes.extend([[-2,-1,0],[2,-1,0],[0,3,0]])
        shapes.append(ph.CollisionShape(sheet,"triangle",tuple(range(10,13))))
        bindings[sheet]=ParticleAttachment(ids)
        laws[material,sheet]=NormalLayer(0.25,16.,friction=friction,slip_speed_m_s=0.0625)
        keys.append((material,sheet))
    system=CollisionLoadSystem(ph.CollisionModel(shapes),bindings,laws)
    state=ph.CoupledState(ph.ParticleState(np.array(nodes,dtype=float),np.zeros((len(nodes),3))),
                          ph.RigidBodyState(np.array(rigid,dtype=float),np.tile([0.,0,0,1],(worlds,1)),
                                            np.array(linear,dtype=float),np.array(angular,dtype=float)))
    models=(ph.ParticleModel(np.ones(len(nodes)),np.zeros(3)),ph.RigidBodyModel(np.ones(worlds),np.ones((worlds,3)),np.zeros(3)))
    return system,state,models,keys


@pytest.mark.parametrize("reverse",[False,True])
def test_real_two_newton_each_side_with_friction_offcenter_moment_and_sensor_frame(reverse):
    system,state,_,keys=layer_fixture(reverse=reverse,friction=0.5)
    body,cloth=keys[0]
    with ph.CollisionWorkspace(system.model) as workspace:
        sampled=system.sample(state,workspace)
    left=measure_contact_wrench(sampled,body,counterparts=[cloth])
    right=measure_contact_wrench(sampled,cloth,counterparts=[body])
    np.testing.assert_allclose(left.force_n,[-1,0,2],atol=1e-13)
    np.testing.assert_allclose(left.friction_force_n,[-1,0,0],atol=1e-13)
    np.testing.assert_allclose(left.torque_nm,[-0.25,-0.5,-0.125],atol=1e-13)
    np.testing.assert_allclose(right.force_n,-left.force_n,atol=1e-13)
    np.testing.assert_allclose(right.torque_nm,-left.torque_nm,atol=1e-13)
    assert left.source=="SurfaceLoads" and left.phase=="sampled-state" and left.site_sides==1
    # Cyclic axes map site x,y,z onto world y,z,x. Shift first, then rotate.
    rotated=measure_contact_wrench(sampled,body,reference_world_m=[0,0,0.25],sensor_to_world_xyzw=[0.5]*4)
    np.testing.assert_allclose(rotated.force_n,[0,2,-1],atol=1e-13)
    np.testing.assert_allclose(rotated.torque_nm,[-0.25,-0.125,-0.25],atol=1e-13)
    np.testing.assert_allclose(rotated.friction_force_n,[0,0,-1],atol=1e-13)


def test_rigid_patch_opposed_friction_has_zero_net_tangent_but_real_torsion():
    system,state,_,keys=layer_fixture("capsule",friction=0.5)
    body,cloth=keys[0]
    with ph.CollisionWorkspace(system.model) as ws:
        loads=system.sample(state,ws)
    result=measure_contact_wrench(loads,ContactSelector(body.world,body.body),counterparts=[cloth])
    assert result.site_sides==2
    np.testing.assert_allclose(result.force_n,[0,0,2],atol=1e-13)
    np.testing.assert_allclose(result.friction_force_n,0,atol=1e-13)
    np.testing.assert_allclose(result.torque_nm,[0,0,-0.5],atol=1e-13)
    np.testing.assert_allclose(result.torque_nm,loads.inputs.rigid.torque[0],atol=1e-13)
    np.testing.assert_allclose(measure_contact_wrench(loads,cloth).torque_nm,-result.torque_nm,atol=1e-13)


def test_two_opposing_normal_forces_on_distinct_shapes_produce_a_pure_couple():
    shapes=[];bindings={}; laws={}; nodes=[]
    for i,(x,z) in enumerate(((0.5,-0.25),(-0.5,0.25))):
        part,face=ph.CollisionKey(0,10,3+i),ph.CollisionKey(0,20+i,5)
        shapes.extend((ph.CollisionShape(part,"sphere",(i,),radius=0.125),ph.CollisionShape(face,"triangle",(10,11,12))))
        bindings[part]=RigidAttachment(0,((x,0,0),))
        bindings[face]=ParticleAttachment(tuple(range(3*i,3*i+3)))
        nodes.extend([[-2,-1,z],[2,-1,z],[0,3,z]])
        laws[part,face]=NormalLayer(0.25,16)
    system=CollisionLoadSystem(ph.CollisionModel(shapes),bindings,laws)
    s=ph.CoupledState(ph.ParticleState(np.array(nodes,dtype=float),np.zeros((6,3))),
                      ph.RigidBodyState(np.zeros((1,3)),np.array([[0.,0,0,1]]),np.zeros((1,3)),np.zeros((1,3))))
    with ph.CollisionWorkspace(system.model) as ws:
        loads=system.sample(s,ws)
    target=ContactSelector(0,10)
    net=measure_contact_wrench(loads,target)
    np.testing.assert_allclose(net.force_n,0,atol=1e-13)
    np.testing.assert_allclose(net.torque_nm,[0,-2,0],atol=1e-13)
    assert net.site_sides==2
    only=measure_contact_wrench(loads,target,counterparts=[ContactSelector(0,20)])
    np.testing.assert_allclose(only.force_n,[0,0,2],atol=1e-13)
    np.testing.assert_allclose(only.torque_nm,[0,-1,0],atol=1e-13)
    empty=measure_contact_wrench(loads,target,counterparts=[])
    np.testing.assert_array_equal(empty.force_n,np.zeros(3))
    assert empty.site_sides==0
    np.testing.assert_allclose(measure_contact_wrench(loads,ph.CollisionKey(0,10,4)).force_n,[0,0,-2],atol=1e-13)


def test_world_isolation_empty_source_and_transactional_last_frame():
    system,state,models,keys=layer_fixture(worlds=2)
    target,other=keys[0]
    with pytest.raises(ValueError,match="publish"):
        measure_contact_wrench(system,target)
    with ph.CollisionWorkspace(system.model) as ws:
        sampled=system.sample(state,ws)
        np.testing.assert_allclose(measure_contact_wrench(sampled,target).force_n,[0,0,2],atol=1e-13)
        np.testing.assert_allclose(measure_contact_wrench(sampled,keys[1][0]).force_n,[0,0,1.5],atol=1e-13)
        with pytest.raises(ValueError,match="world"):
            measure_contact_wrench(sampled,target,counterparts=[keys[1][1]])
        crossed=replace(sampled,contacts=(replace(sampled.contacts[0],second=keys[1][1]),))
        with pytest.raises(ValueError,match="world"):
            measure_contact_wrench(crossed,target)
        solvers=(ph.SemiImplicitSolver(),ph.RigidSemiImplicitSolver())
        zero=ph.CoupledInputs.zeros(6,2)
        accepted=system.step(state,zero,models,solvers,1/256,ws)
        first=measure_contact_wrench(system,target)
        assert first.source=="CollisionLoadStep" and first.phase=="substep-entry"
        np.testing.assert_array_equal(measure_contact_wrench(accepted,target,published_by=system).force_n,first.force_n)
        with pytest.raises(RuntimeError,match="abort"):
            with ws.transaction():
                staged=system.step(accepted.state,zero,models,solvers,1/256,ws)
                with pytest.raises(ValueError,match="publish"):
                    measure_contact_wrench(staged,target,published_by=system)
                np.testing.assert_array_equal(measure_contact_wrench(system,target).force_n,first.force_n)
                raise RuntimeError("abort")
        assert system.last_frame is accepted
        np.testing.assert_array_equal(measure_contact_wrench(system,target).force_n,first.force_n)
        separated=ph.CoupledState(state.particles,state.rigid.copy())
        separated.rigid.position[:,2]+=2
        no_contacts=system.sample(separated,ws)
        result=measure_contact_wrench(no_contacts,other)
        np.testing.assert_array_equal(result.force_n,np.zeros(3))
        np.testing.assert_array_equal(result.torque_nm,np.zeros(3))
        assert result.site_sides==0
    with pytest.raises(ValueError):
        measure_contact_wrench(sampled,target,sensor_to_world_xyzw=[0,0,0,2])


def test_cloth_link_midpoint_window_matches_published_joint_effort_and_real_integrated_impulses():
    program,state,inputs=make_scene(revolute=True,friction=0.5)
    state.domains["arm"].v[0]=0.2
    state.domains["cloth"].velocity[:,0]=0.125
    link,face=ContactSelector(0,("arm",0),"beam"), ContactSelector(0,"cloth","0")
    pivot=np.array([-0.5,0.,0.25])
    initial_arm=state.domains["arm"].v[0]
    initial_cloth=state.domains["cloth"].velocity.sum(axis=0)
    impulse=np.zeros(3);angular_impulse=0.;max_error=0.;peak_friction=0.;reported=0
    settings=ph.StepSettings(substeps=2,position_iterations=1,velocity_iterations=1)
    with program.stepper(settings) as rt:
        for _ in range(8):
            accepted=rt.step(state,inputs,1/128)
            assert accepted.report.accepted, accepted.report.reason
            for i,r in enumerate(accepted.report.substeps):
                sensed=measure_cloth_link_wrench(accepted,program,link,counterparts=[face],substep=i,reference_world_m=pivot)
                cloth=measure_cloth_link_wrench(accepted,program,face,counterparts=[link],substep=i,reference_world_m=pivot)
                assert sensed.source=="cloth_links" and sensed.phase=="substep-midpoint" and sensed.substep_index==r.index and sensed.dt_s==r.dt
                assert sensed.site_sides>0
                np.testing.assert_allclose(sensed.force_n,-cloth.force_n,atol=2e-13)
                np.testing.assert_allclose(sensed.torque_nm,-cloth.torque_nm,atol=2e-13)
                tau=r.diagnostics["robot_contact"]["cloth_link_efforts"]["arm"][0]
                np.testing.assert_allclose(sensed.torque_nm[1],tau,rtol=0,atol=2e-13)
                max_error=max(max_error,abs(sensed.torque_nm[1]-tau));peak_friction=max(peak_friction,np.linalg.norm(sensed.friction_force_n))
                impulse+=r.dt*cloth.force_n;angular_impulse+=r.dt*sensed.torque_nm[1];reported+=1
            state=accepted.state
    delta_p=state.domains["cloth"].velocity.sum(axis=0)-initial_cloth
    delta_h=state.domains["arm"].v[0]-initial_arm
    np.testing.assert_allclose(impulse,delta_p,rtol=0,atol=3e-12)
    np.testing.assert_allclose(angular_impulse,delta_h,rtol=0,atol=3e-12)
    assert peak_friction>0.05 and reported==16
    print(json.dumps(dict(steps=reported,joint_torque_error=max_error,cloth_impulse_ns=impulse.tolist(),angular_impulse_nms=angular_impulse,
                          peak_friction_n=peak_friction)))
    with pytest.raises(ValueError,match="world"):
        measure_cloth_link_wrench(accepted,program,ContactSelector(1,("arm",0)))


def test_genuinely_rejected_scene_cannot_publish_a_cloth_load_reading():
    program,state,inputs=make_scene()
    selector=ContactSelector(0,("arm",0),"beam")
    with program.stepper(ph.StepSettings(substeps=1,position_iterations=1,velocity_iterations=1)) as rt:
        keep=rt.step(state,inputs,1/256)
        assert keep.report.accepted
        previous=measure_cloth_link_wrench(keep,program,selector)
        state.domains["arm"].v[0]=-128.
        rejected=rt.step(state,inputs,1/128)
        assert not rejected.report.accepted
    with pytest.raises(ValueError,match="accepted"):
        measure_cloth_link_wrench(rejected,program,selector)
    np.testing.assert_array_equal(measure_cloth_link_wrench(keep,program,selector).force_n,previous.force_n)
