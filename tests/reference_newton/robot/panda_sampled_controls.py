"""Analytic free-box slip/torque controls for external Newton wrenches with native contact disabled."""

import argparse
import json
from pathlib import Path

import newton
import numpy as np
import warp as wp

from panda_sampled_loads import box_corners, tabletop_load

from phymeo.constraints.robot_contact import RobotContactMaterial


def check(speed):
    dt = 1 / 128
    half = np.array([1/8, 1/4, 1/16])
    full = 2 * half
    mass = 2.
    inertia = mass / 12 * np.array([full[1]**2 + full[2]**2, full[0]**2 + full[2]**2, full[0]**2 + full[1]**2])
    center = np.array([0., 0., half[2]-1/1024])
    gravity = np.array([0., 0., -2.])
    builder = newton.ModelBuilder(gravity=tuple(gravity), up_axis=newton.Axis.Z)
    body = builder.add_body(xform=wp.transform(wp.vec3(*center), wp.quat_identity()), mass=mass,
                            inertia=wp.mat33(*np.diag(inertia).ravel()), lock_inertia=True, label="free")
    cfg = newton.ModelBuilder.ShapeConfig(density=0, gap=0, margin=0)
    builder.add_shape_box(body,hx=half[0],hy=half[1],hz=half[2],cfg=cfg)
    builder.add_shape_box(-1,xform=wp.transform((0,0,-.125),wp.quat_identity()),hx=1.,hy=1.,hz=.125,cfg=cfg)
    model = builder.finalize(device="cpu")
    solver = newton.solvers.SolverFeatherstone(model,angular_damping=0.)
    table_center, table_half = np.array([0.,0.,-.125]), np.array([1.,1.,.125])
    law = RobotContactMaterial(1024., 0., .5, .04)
    results = {}
    for loaded in (False, True):
        state, after = model.state(), model.state()
        state.joint_q.assign(np.r_[center,[0.,0.,0.,1.]].astype(np.float32))
        state.joint_qd.assign(np.array([speed,0.,0.,0.,0.,0.],dtype=np.float32))
        newton.eval_fk(model,state.joint_q,state.joint_qd,state)
        twist = state.body_qd.numpy()[0].astype(float)
        force, normal, arms = tabletop_load(
            box_corners(half),state.body_q.numpy()[0,:3].astype(float),np.eye(3),twist,
            table_center,table_half,law,
        )
        # Exactly four bottom corners have 1 mm-scale compression, each with 1 N normal load.
        fx = -2 * min(speed/.04, 1.)
        expected = np.array([fx,0.,4.,0.,-half[2]*fx,0.])
        wrench = np.r_[force.sum(axis=0), np.cross(arms,force).sum(axis=0)]
        np.testing.assert_allclose(wrench,expected,rtol=0,atol=2e-7)
        np.testing.assert_allclose(np.sort(normal),[0,0,0,0,1,1,1,1],rtol=0,atol=2e-7)
        assert abs(np.sum(force @ twist[:3])-fx*speed) < 3e-8
        state.clear_forces()
        if loaded:
            state.body_f.assign(wrench[None,:].astype(np.float32))
        solver.step(state,after,model.control(),None,dt)
        actual = after.body_qd.numpy()[0].astype(float)
        target = twist + dt*np.r_[gravity+(expected[:3]/mass if loaded else 0),
                                   expected[3:]/inertia if loaded else np.zeros(3)]
        np.testing.assert_allclose(actual,target,rtol=0,atol=2e-7)
        results["applied" if loaded else "disabled"] = dict(v_after=actual.tolist(),wrench=wrench.tolist() if loaded else [0.]*6)
    return dict(input_speed_m_s=speed,tangent_power_W=fx*speed,dt=dt,results=results)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    result=[check(s) for s in (.02,.08)]
    with args.out.open("x",encoding="utf8") as out:
        json.dump(result,out,allow_nan=False)
    print(json.dumps(result,allow_nan=False))
