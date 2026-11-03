"""Physical relative translation between a bounded curved solid and an oriented half-space."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

import phymeo as ph
from phymeo.solvers.driven_robot import DrivenRobotProgram


def scene(kind,dims,center,end,plane,plane_end,normal,reverse=False):
    k1,k2=ph.CollisionKey(3,4,0),ph.CollisionKey(3,8,0)
    ck,pk=(k2,k1) if reverse else (k1,k2)
    a=ph.CollisionShape(ck,kind,(17,),curved=ph.CurvedSolid(kind,dims))
    b=ph.CollisionShape(pk,"plane",(83,),normal=normal)
    model=ph.CollisionModel((b,a))
    paths={ck:ph.CollisionMotion([center],[end]),pk:ph.CollisionMotion([plane],[plane_end])}
    with ph.CollisionWorkspace(model) as work:
        row,=ph.detect(model,paths,ph.CollisionPolicy(),work).pairs
    return row,ck


@pytest.mark.parametrize("kind,dims", [("ellipsoid",(.5,.25,.125)),("cylinder",(.5,.25)),("cone",(.5,.25))])
@pytest.mark.parametrize("swapped",[False,True])
def test_two_moving_owners_cross_at_three_eighths(kind,dims,swapped):
    reach=dims[-1]
    shift=np.array([.5,-.25,.125]); anchor=np.array([1.,.5,-.25])
    start=anchor+[.25,-.5,reach+.375]
    row,ck=scene(kind,dims,start,start+shift-[0,0,1.],anchor,anchor+shift,[0,0,2],swapped)
    assert row.status=="hit",row.reason
    sweep=row.sweep
    assert sweep.start_gap_m==pytest.approx(.375,abs=1e-12)
    assert sweep.end_gap_m==pytest.approx(-.625,abs=1e-12)
    lo,hi=sweep.first_t
    assert lo<=.375<=hi and hi-lo<=1e-12
    assert row.curved.last_contact.interval==(1.,1.)
    w=sweep.witness
    assert abs(w.gap_m)<1e-12
    for x in (w.first_point,w.second_point):
        assert x[2]==pytest.approx((anchor+sweep.witness_time*shift)[2],abs=1e-12)
    assert row.witness.gap_m==sweep.end_gap_m


def test_oblique_cylinder_first_time_encloses_the_irrational_analytic_root():
    row,_=scene("cylinder",(.25,.375),[0,0,1],[0,0,0],[0,0,0],[0,0,0],[1,1,1])
    assert row.status=="hit",row.reason
    with localcontext() as ctx:
        ctx.prec=60
        expected=Decimal(5)/8-Decimal(2).sqrt()/4
        lo,hi=map(Decimal.from_float,row.sweep.first_t)
        assert lo<=expected<=hi and hi-lo<Decimal("1e-12")


@pytest.mark.parametrize("start,end,first,last",[(-.125,.375,0.,.25),(.5,.25,None,None),(0.,0.,0.,1.)])
def test_relative_planeload_separates_or_persists_as_declared(start,end,first,last):
    # Start/end are material gaps. The plane supplies the relative motion.
    ro=np.array([0.,0.,.25]); drift=np.array([.125,-.25,.375])
    row,_=scene("cone",(.375,.25),ro,ro+drift,[0,0,-start],drift+[0,0,-end],[0,0,1])
    if first is None:
        assert row.status=="separated" and row.sweep.first_t is None
    else:
        assert row.status=="hit" and row.sweep.first_t==(first,first)
        a,b=row.curved.last_contact.interval
        assert a<=last<=b


def test_multi_site_command_drives_a_real_tool_and_observes_the_plane_passage():
    # Two sites on the same rail require a consistent translation. The actual bounded
    # actuator supplies -4N. Only the third physical step moves the tool into the plane.
    robot=ph.RobotModel(np.array([-1]),("prismatic",),np.zeros((1,3)),np.array([[0.,0.,0.,1.]]),
        np.array([[0.,0.,1.]]),np.array([2.]),np.zeros((1,3)),np.ones((1,3)),np.zeros(3))
    origin=ph.RobotSite(0); marker=ph.RobotSite(0,(.25,0,.125))
    targets=[ph.SiteIKTarget(origin,position_world_m=(0,0,-.5)),
             ph.SiteIKTarget(marker,position_world_m=(.25,0,-.375))]
    ik=ph.solve_multi_site_ik(robot,np.array([.25]),targets,tolerance_m=1e-8)
    assert ik.converged and abs(ik.configuration[0]+.5)<1e-8
    drive=DrivenRobotProgram(robot,kp=[64],kd=[0],effort_limit=[4])
    state=drive.initial_state(ph.RobotState(np.array([.25]),np.array([0.])))
    packet=drive.inputs(ik.configuration)
    dt=1/8; seen=[]
    with drive.stepper(ph.StepSettings(substeps=1,record_substeps=True)) as rt:
        for n in range(3):
            previous=state.domains["robot"]
            result=rt.step(state,packet,dt)
            assert result.report.accepted
            current=result.state.domains["robot"]
            qdd=(current.v-previous.v)/dt
            imu=ph.measure_robot_imu(robot,previous,qdd,origin,time_s=n*dt)
            np.testing.assert_allclose(imu.specific_force_m_s2,[0,0,-2],atol=1e-9,rtol=0)
            np.testing.assert_allclose(imu.angular_velocity_rad_s,[0,0,0],atol=1e-9,rtol=0)
            row,_=scene("ellipsoid",(.25,.25,.125),[0,0,previous.q[0]],[0,0,current.q[0]],[0,0,0],[0,0,0],[0,0,1])
            seen.append(row)
            state=result.state
    assert [row.status for row in seen]==["separated","separated","hit"]
    assert seen[2].sweep.first_t[0]<=1/3<=seen[2].sweep.first_t[1]
    assert [r.sweep.end_gap_m for r in seen]==pytest.approx([.09375,.03125,-.0625],abs=1e-10)
