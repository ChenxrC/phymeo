"""Substep inspection retains the actual finite surface, material site and force time."""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

script=Path(__file__).resolve().parents[1]/"examples/export_compliant_replay.py"
spec=spec_from_file_location("physical_substep_replay",script)
replay=module_from_spec(spec)
spec.loader.exec_module(replay)


def source():
    site=dict(face_weights=[[0,.5],[1,.25],[2,.25]],axis_parameter=0.,point_m=[.25,.25,0.],
              force_on_cloth_n=[0.,0.,2.],force_on_link_n=[0.,0.,-2.],
              relative_velocity_m_s=[0.,0.,-.25],normal_force_n=2.,power_w=-.5)
    row=dict(link="shaft",robot="R",face="cloth",face_nodes=[0,1,2],
             face_positions_m=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],
             axis_ends_m=[[.25,.25,-.25],[.25,.25,-.5]],normal=[0.,0.,1.],
             half_thickness_m=0.,gap_m=.125,sites=[site],normal_force_n=2.,power_w=-.5)
    body=[dict(t=.25,cloth_links=[row]),dict(t=.75,cloth_links=[])]
    return dict(cloth_link_material={},robot_contact={"capsules":[dict(key="shaft",domain="R",radius_m=.125)]},
                faces=[[0,1,2]],owners=["cloth/0","cloth/1","cloth/2"],
                substeps=2,dt_s=1.,frames=[dict(t=0.),dict(t=1.,force_sample_s=.75,robot_contact=deepcopy(body[1]))],
                robot_samples=body)


def test_transient_load_inside_one_display_interval_survives_while_frame_is_unloaded():
    trace=source()
    windows=replay.structural_substep_windows(trace)
    assert len(windows)==1 and [s["t"] for s in windows[0]]==[.25,.75]
    assert windows[0][0]["cloth_links"]==trace["robot_samples"][0]["cloth_links"]
    assert windows[0][0]["cloth_links"][0]["sites"][0]["force_on_link_n"]==[0.,0.,-2.]
    assert windows[0][1]["cloth_links"]==trace["frames"][1]["robot_contact"]["cloth_links"]==[]


@pytest.mark.parametrize("fault",["missing","time","force","material","frame"])
def test_invalid_intermediate_record_does_not_produce_inspectable_contact_geometry(fault):
    trace=source()
    if fault=="missing":
        trace["robot_samples"].pop(0)
    elif fault=="time":
        trace["robot_samples"][0]["t"]+=.25
    elif fault=="force":
        trace["robot_samples"][0]["cloth_links"][0]["sites"][0]["force_on_link_n"][2]=1.
    elif fault=="material":
        trace["robot_samples"][0]["cloth_links"][0]["sites"][0]["point_m"][0]+=.1
    else:
        trace["frames"][1]["force_sample_s"]=.25
    with pytest.raises(ValueError):
        replay.structural_substep_windows(trace)
