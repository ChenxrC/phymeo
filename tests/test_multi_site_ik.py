"""Bounded multi-site IK on a shared-trunk robot, independently measured at its tool frames."""

import json

import numpy as np
import pytest
from reference_newton.robot.site_ik_fixture import fixture, qmatrix

from phymeo.constraints.robot_contact import JointLimits
from phymeo.control.point_ik import solve_point_ik
from phymeo.control.site_ik import SiteIKTarget, solve_multi_site_ik
from phymeo.core.robot import RobotModel, RobotState
from phymeo.core.scene import StepSettings
from phymeo.dynamics.robot_kinematics import forward_kinematics
from phymeo.sensors import RobotSite
from phymeo.solvers.driven_robot import DrivenRobotProgram


def double_arm(mirror=False):
    f=fixture(mirror)
    model=RobotModel(f["parents"],f["kinds"],f["anchors"],f["frames"],f["axes"],
                     np.array([0.5,0.25,0.125,0.25,0.125]),np.array([[0.,0,0],[0.125,0,0],[0.0625,0,0],[0.125,0,0],[0.0625,0,0]]),
                     np.tile([0.006,0.008,0.009],(5,1)),np.zeros(3))
    goals=[SiteIKTarget(RobotSite(link,f["local"][k],f["mount"][k]),position_world_m=f["goals"][k],
                        orientation_world_xyzw=f["rotations"][k],position_weight=2+k,rotation_weight=1+k)
           for k,link in enumerate([2,4])]
    bounds=JointLimits(f["lower"],f["upper"],stiffness=200.,damping=1.)
    return f,model,goals,bounds


def measure(model,q,f):
    fk=forward_kinematics(model,q)
    errors=[]
    for k,body in enumerate((2,4)):
        position=fk.position[body]+fk.rotation[body]@f["local"][k]
        rotation=fk.rotation[body]@qmatrix(f["mount"][k])
        delta=rotation.T@qmatrix(f["rotations"][k])
        sine=np.linalg.norm([delta[2,1]-delta[1,2],delta[0,2]-delta[2,0],delta[1,0]-delta[0,1]])/2
        angle=np.arctan2(sine,(np.trace(delta)-1)/2)
        errors.append([np.linalg.norm(position-f["goals"][k]),angle])
    return np.array(errors)


@pytest.mark.parametrize("mirror",[False,True])
def test_branched_two_poses_and_mirror_from_non_solutions(mirror):
    f,m,targets,limits=double_arm(mirror)
    seed=f["seed"].copy()
    assert np.min(measure(m,seed,f)[:,1])>1.
    result=solve_multi_site_ik(m,seed,targets,limits=limits,tolerance_m=2e-9,tolerance_rad=2e-9)
    assert result.converged, result.status
    actual=measure(m,result.configuration,f)
    np.testing.assert_allclose(result.configuration,f["truth"],rtol=0,atol=6e-9)
    np.testing.assert_array_less(actual,2e-9)
    np.testing.assert_allclose([[r.position_m,r.rotation_rad] for r in result.residuals],actual,rtol=0,atol=2e-14)
    assert result.conflicting_pairs==() and result.active_bounds==()
    np.testing.assert_array_equal(seed,f["seed"])
    print(json.dumps(dict(mirror=mirror,iterations=result.iterations,q=result.configuration.tolist(),errors=actual.tolist())))


def chain(kinds,axes):
    n=len(kinds)
    return RobotModel(np.arange(-1,n-1),tuple(kinds),np.zeros((n,3)),np.tile([0.,0,0,1],(n,1)),
                      np.array(axes,dtype=float),np.ones(n),np.zeros((n,3)),np.ones((n,3)),np.zeros(3))


def test_weighted_incompatible_positions_and_rigid_separation_are_reported():
    model=chain(["prismatic"],[[1,0,0]])
    s=RobotSite(0)
    targets=[SiteIKTarget(s,position_world_m=[-1,0,0],position_weight=1),
             SiteIKTarget(s,position_world_m=[1,0,0],position_weight=2)]
    result=solve_multi_site_ik(model,np.array([-0.125]),targets)
    assert not result.converged and result.conflicting_pairs==((0,1),)
    np.testing.assert_allclose(result.configuration,[0.6],rtol=0,atol=2e-7)
    np.testing.assert_allclose([e.position_m for e in result.residuals],[1.6,0.4],rtol=0,atol=2e-7)
    assert result.weighted_error==pytest.approx(np.sqrt(3.2),abs=2e-12)
    # Mounts 1/4 m apart on one rigid link cannot both sit at the world origin.
    conflict=solve_multi_site_ik(model,np.zeros(1),[SiteIKTarget(s,position_world_m=[0,0,0]),
                                SiteIKTarget(RobotSite(0,(0.25,0,0)),position_world_m=[0,0,0])])
    assert conflict.conflicting_pairs==((0,1),)


def test_hard_bounds_reallocate_other_columns_and_never_change_inactive_coordinates():
    model=chain(["prismatic","prismatic"],[[1,0,0],[1,0,0]])
    limits=JointLimits(np.array([-0.25,-0.5]),np.array([0.05,0.4]),1.)
    target=SiteIKTarget(RobotSite(1),position_world_m=[0.43,0,0])
    reachable=solve_multi_site_ik(model,np.zeros(2),[target],limits=limits)
    assert reachable.converged
    np.testing.assert_allclose(reachable.configuration,[0.05,0.38],atol=2e-7)
    blocked=solve_multi_site_ik(model,np.array([0.8,1.]),[SiteIKTarget(RobotSite(1),position_world_m=[1.,0,0])],limits=limits)
    assert blocked.status=="bound_limited" and blocked.active_bounds==(0,1)
    np.testing.assert_array_equal(blocked.configuration,limits.upper)
    assert blocked.residuals[0].position_m==pytest.approx(0.55)
    frozen=solve_multi_site_ik(model,np.array([0.125,0.]),[target],limits=limits,active_joints=[1])
    assert frozen.converged and frozen.configuration[0]==0.125
    assert frozen.configuration[1]==pytest.approx(0.305,abs=2e-7)


def test_antipodal_direction_can_turn_inward_at_bound_and_roll_is_unconstrained():
    model=chain(["revolute","revolute"],[[0,0,1],[1,0,0]])
    limits=JointLimits(np.array([-3.3,-1.]),np.array([0.,1.]),1.)
    start=np.array([0.,0.625])
    target=SiteIKTarget(RobotSite(1),direction_world=[-1.,0,0],local_direction=[1,0,0])
    result=solve_multi_site_ik(model,start,[target],limits=limits,tolerance_rad=1e-9)
    assert result.converged,result.status
    np.testing.assert_allclose(result.configuration,[-np.pi,0.625],rtol=0,atol=2e-9)
    assert result.residuals[0].position_m is None and result.residuals[0].direction_rad<1e-9
    assert result.residuals[0].rotation_rad is None


def test_orientation_only_at_half_turn_and_equivalent_quaternion_signs():
    model=chain(["revolute"],[[0.6,0.8,0]])
    limits=JointLimits(np.array([-3.2]),np.array([0.]),1.)
    q=np.array([0.6,0.8,0.,0.])
    for sign in (-1,1):
        result=solve_multi_site_ik(model,np.zeros(1),[SiteIKTarget(RobotSite(0),orientation_world_xyzw=sign*q)],
                                   limits=limits,tolerance_rad=1e-9)
        assert result.converged,result.status
        np.testing.assert_allclose(result.configuration,[-np.pi],atol=2e-9,rtol=0)


def test_single_point_agrees_with_existing_solver_and_zero_weight_channel_is_diagnostic():
    f,m,targets,bounds=double_arm()
    s=targets[0].site
    start=f["seed"].copy()
    old=solve_point_ik(m,start,s.link_index,np.asarray(s.local_position_m),f["goals"][0],limits=bounds,active_joints=[0,1,2],max_iterations=80)
    new=solve_multi_site_ik(m,start,[SiteIKTarget(s,position_world_m=f["goals"][0])],limits=bounds,active_joints=[0,1,2])
    assert new.converged and old.converged
    np.testing.assert_allclose(new.configuration[:3],old.configuration[:3],atol=6e-7)
    np.testing.assert_array_equal(new.configuration[3:],start[3:])
    ignored=SiteIKTarget(s,position_world_m=[99,0,0],position_weight=0.)
    first=solve_multi_site_ik(m,start,[targets[0],ignored],limits=bounds)
    assert first.converged and first.residuals[1].position_m>98
    assert first.conflicting_pairs==()
    with pytest.raises(ValueError):
        solve_multi_site_ik(m,start,[ignored])
    with pytest.raises(ValueError):
        solve_multi_site_ik(m,start,[targets[0]],active_joints=[1,1])
    with pytest.raises(ValueError):
        SiteIKTarget(s,direction_world=[0.,0,0])
    with pytest.raises(ValueError):
        SiteIKTarget(RobotSite(None),position_world_m=[0,0,0])


def test_two_tool_targets_drive_real_bounded_actuators_over_a_fixed_window():
    f,model,goals,limits=double_arm()
    solved=solve_multi_site_ik(model,f["seed"],goals,limits=limits)
    assert solved.converged
    caps=np.array([12.,1.5,4.,1.5,4.])
    program=DrivenRobotProgram(model,kp=np.array([120.,12,80,12,80]),kd=np.array([18.,1.5,5.,1.5,5.]),
                               effort_limit=caps,limits=limits)
    state=program.initial_state(RobotState(f["seed"],np.zeros(5)))
    command=program.inputs(solved.configuration)
    initial=measure(model,f["seed"],f)
    peak=np.zeros(5); saturated=0; first=None
    with program.stepper(StepSettings(substeps=4,record_substeps=True)) as rt:
        for k in range(48):
            result=rt.step(state,command,1/32)
            assert result.report.accepted,result.report.reason
            for step in result.report.substeps:
                effort=np.abs(step.diagnostics["actuator_effort"])
                assert np.all(effort<=caps+1e-12)
                peak=np.maximum(peak,effort)
                saturated+=bool(np.any(effort>=caps-1e-9))
            state=result.state
            if k==0:
                first=measure(model,state.domains["robot"].q,f)
    physical=measure(model,state.domains["robot"].q,f)
    geometric=measure(model,solved.configuration,f)
    assert saturated and np.max(first[:,1])>1.
    assert np.max(physical[:,0])<0.035 and np.max(physical[:,1])<0.08
    assert np.max(physical)>np.max(geometric)
    print(json.dumps(dict(steps=192,initial=initial.tolist(),first=first.tolist(),final=physical.tolist(),
                          ik=geometric.tolist(),peak_effort=peak.tolist(),saturated_steps=saturated)))
