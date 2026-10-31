"""Fixed whole/step/rebuild grid; analytic geometry paths, not dynamics."""
import dataclasses
from fractions import Fraction as F

import numpy as np
import pytest
from test_shell_boundary import VARIANTS,control,payload,save

from phymeo.constraints.conforming_shell import ShellUnresolved
from phymeo.constraints.shell_boundary import BoundaryProofLimits
from phymeo.geometry.shell_boundary_sweep import certify_shell_boundary_sweep
from phymeo.validation.shell_boundary import verify_shell_boundary_sweep


def state(name,ref,x,rotation,time):
    result=x.copy()
    if name=="C1":
        result+=np.array([1/4,1/8,-1/16])@rotation.T*float(time)
    else:
        delta=abs(time-F(1,8))
        result[ref.node_ids>=4]+=np.array([float(delta),0,0])@rotation.T
    return result


@pytest.mark.parametrize("name",("C1","C2"))
@pytest.mark.parametrize("variant",VARIANTS)
@pytest.mark.parametrize("h",(F(1,64),F(1,128),F(1,256)))
def test_full_analytic_grid_and_two_predeclared_rebuilds(name,variant,h,tmp_path):
    # Pair by authored name/variant/h/rebuild/ordinal; retain every payload field.
    snapshots=[]
    for rebuild in range(2):
        ref,x,hint,rotation,_=control(name,variant)
        end=F(1,8) if name=="C1" else F(1,4)
        joins=(F(0),end) if name=="C1" else (F(0),F(1,8),end)
        segments=tuple(np.stack((state(name,ref,x,rotation,a),state(name,ref,x,rotation,b)))
                       for a,b in zip(joins,joins[1:]))
        durations=tuple(float(b-a) for a,b in zip(joins,joins[1:]))
        calls=[("whole",segments,durations)]
        for step in range(int(end/h)):
            a,b=step*h,(step+1)*h
            calls.append((f"step{step:03d}",
                          (np.stack((state(name,ref,x,rotation,a),state(name,ref,x,rotation,b))),),
                          (float(h),)))
        rebuilt=[]
        for label,path,duration in calls:
            receipt=certify_shell_boundary_sweep(ref,path,durations=duration,witness_hint=hint)
            checked=verify_shell_boundary_sweep(receipt,ref,path,durations=duration,
                                                witness_hint=hint,limits=BoundaryProofLimits())
            assert checked
            assert receipt.status=="weak_material_path_certified"
            assert len(receipt.segments)==len(path)
            assert receipt.operations<=2_000_000
            assert receipt.path_kind=="constant-derived-director-v1"
            save(tmp_path/f"{name}-{variant}-h{h.denominator}-r{rebuild}-{label}.json",
                 (ref,path,duration,hint,receipt,checked))
            rebuilt.append(payload(receipt))
        snapshots.append(rebuilt)
    assert snapshots[0]==snapshots[1]


def test_legal_endpoints_cannot_hide_constant_director_component_crossing():
    ref,x,hint,_,_=control("C2")
    x[ref.node_ids>=4,1]-=2
    x[ref.node_ids>=4,0]+=2
    y=x.copy()
    y[ref.node_ids<4,0]+=4
    y[ref.node_ids>=4,0]-=4
    with pytest.raises(ShellUnresolved):
        certify_shell_boundary_sweep(ref,(np.stack((x,y)),),durations=(1.0,),witness_hint=hint)


@pytest.mark.parametrize("kind",("changing_director","zero_star_inside","bad_join","zero_duration"))
def test_actual_path_domain_and_full_closed_join_failures(kind):
    ref,x,hint,_,_=control("C2")
    y=x.copy()
    if kind=="changing_director":
        y[0,2]+=1/8
        paths=(np.stack((x,y)),)
        times=(1/8,)
    elif kind=="zero_star_inside":
        center=x[:4].mean(axis=0)
        y[:4]=2*center-x[:4]
        paths=(np.stack((x,y)),)
        times=(1/8,)
    elif kind=="bad_join":
        paths=(np.stack((x,x)),np.stack((x+1/16,x+1/16)))
        times=(1/8,1/8)
    else:
        paths=(np.stack((x,x)),)
        times=(0.0,)
    with pytest.raises((ShellUnresolved,ValueError,TypeError)):
        certify_shell_boundary_sweep(ref,paths,durations=times,witness_hint=hint)


def test_missing_time_node_or_endpoint_is_not_a_certificate():
    ref,x,hint,_,_=control("C2")
    paths=(np.stack((x,x)),)
    certificate=certify_shell_boundary_sweep(ref,paths,durations=(1/8,),witness_hint=hint)
    forged=dataclasses.replace(certificate,segments=())
    assert not verify_shell_boundary_sweep(forged,ref,paths,durations=(1/8,),
                                          witness_hint=hint,limits=BoundaryProofLimits())