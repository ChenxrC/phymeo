"""Equal and opposite traction on a calibrated layer between two free curved bodies."""

from contextlib import ExitStack
from dataclasses import asdict

import json
import numpy as np
import pytest

import phymeo as ph
from phymeo.solvers.collision_loads import CollisionLoadSystem, NormalLayer, RigidAttachment


def sliding_pair(mode, reverse, root):
    dtype=np.float64 if mode=="host" else np.float32
    center=np.array([[0,0,0],[.375+1/32,0,0]],dtype=dtype)
    q=np.tile(np.array([0.,0.,0.,1.],dtype=dtype),(2,1))
    v=np.array([[0,.25,0],[0,0,0]],dtype=dtype)
    state=ph.CoupledState(ph.ParticleState(np.empty((0,3),dtype=dtype),np.empty((0,3),dtype=dtype)),
                          ph.RigidBodyState(center,q,v,np.zeros_like(center)))
    inv=np.array([2.,1.],dtype=dtype)
    pm=ph.ParticleModel(np.empty(0,dtype=dtype),np.zeros(3,dtype=dtype))
    rm=ph.RigidBodyModel(inv,np.full((2,3),.25,dtype=dtype),np.zeros(3,dtype=dtype))
    keys=[ph.CollisionKey(3,k,0) for k in ([9,2] if reverse else [2,9])]
    geoms=[ph.CollisionShape(key,"ellipsoid",(17,),curved=ph.CurvedSolid("ellipsoid",dims))
           for key,dims in zip(keys,((.25,.125,.125),(.125,.25,.125)),strict=True)]
    model=ph.CollisionModel(geoms)
    bnd={key:RigidAttachment(i,((0.,0.,0.),)) for i,key in enumerate(keys)}
    dt=1/256
    with ExitStack() as stack:
        work=stack.enter_context(ph.CollisionWorkspace(model,mode,cache_root=root))
        layer=None if mode=="host" else stack.enter_context(ph.TyxNormalLayer(mode,cache_root=root))
        solver=ph.RigidSemiImplicitSolver() if mode=="host" else stack.enter_context(ph.TyxRigidSemiImplicitSolver(mode=mode,cache_root=root))
        sys=CollisionLoadSystem(model,bnd,{tuple(keys):NormalLayer(1/16,64.,friction=.25)},layer_operator=layer)
        ext=ph.CoupledInputs.zeros(0,2,dtype=dtype)
        states=[]
        for i in range(20):
            frame=sys.step(state,ext,(pm,rm),(None,solver),dt,work)
            contact,=frame.loads.contacts
            site,=contact.sites
            kp=(np.array(site.material.first_point)+np.array(site.material.second_point))*.5
            np.testing.assert_allclose(site.application_point_first_m,kp,atol=1e-9,rtol=0)
            np.testing.assert_allclose(site.application_point_second_m,kp,atol=1e-9,rtol=0)
            loads=frame.loads.inputs.rigid
            np.testing.assert_allclose(loads.force.sum(axis=0),[0,0,0],rtol=0,atol=1e-6)
            np.testing.assert_allclose((np.cross(state.rigid.position,loads.force)+loads.torque).sum(axis=0),
                                       [0,0,0],atol=8e-7,rtol=0)
            if i==0:
                np.testing.assert_allclose(loads.force,[[-2.,-.5,0],[2.,.5,0]],atol=2e-6,rtol=0)
                np.testing.assert_allclose(loads.torque[:,2],[-.1328125,-.0703125],atol=2e-6,rtol=0)
                assert contact.power_w==pytest.approx(-.125,abs=2e-6)
            state=frame.state
            momentum=state.rigid.velocity.astype(float)/inv[:,None]
            np.testing.assert_allclose(momentum.sum(axis=0),[0,.125,0],rtol=0,atol=3e-6)
            ang=np.cross(state.rigid.position,momentum).sum(axis=0)+.25*state.rigid.angular_velocity.sum(axis=0)
            np.testing.assert_allclose(ang,[0,0,0],atol=5e-6,rtol=0)
            assert frame.path.pairs[0].status=="separated"
            states.append(dict(step=i+1,position=state.rigid.position.tolist(),velocity=state.rigid.velocity.tolist(),
                omega=state.rigid.angular_velocity.tolist(),contact=asdict(contact),momentum=momentum.sum(axis=0).tolist(),
                angular_momentum=ang.tolist()))
        assert np.max(abs(state.rigid.angular_velocity))>.002
        return states


def compare(mode, root):
    normal=sliding_pair(mode,False,root)
    swapped=sliding_pair(mode,True,root)
    for a,b in zip(normal,swapped,strict=True):
        for key in ("position","velocity","omega"):
            np.testing.assert_allclose(a[key],b[key],atol=3e-5,rtol=0)
    (root/"rigid-friction-paired.json").write_text(json.dumps(dict(mode=mode,first=normal,swapped=swapped),
                                                              allow_nan=False),encoding="utf8")


def test_smooth_rigid_friction_preserves_pair_momentum_and_swaps(tmp_path):
    compare("host",tmp_path)


@pytest.mark.tyx
def test_smooth_rigid_friction_with_real_integrators(tyx_mode,tmp_path):
    compare(tyx_mode,tmp_path)
