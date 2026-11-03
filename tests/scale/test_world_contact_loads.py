"""Physical material forces, world identity and midpoint trajectories on the contact batch."""

import json
from importlib import util
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.constraints.cloth_material import ClothMaterial
from phymeo.constraints.robot_contact import LinkCapsule
from phymeo.dynamics.robot_contact import RobotContactSystem


def api():
    assert util.find_spec("phymeo.backends.tyx_world_contact") is not None, "missing keyed TYX world contact batch"
    from phymeo.backends import tyx_world_contact
    return tyx_world_contact


def fixture(*, lift=0.0, reverse=False, shifted=False, rotate=False, moving=False):
    casefile = Path(__file__).resolve().parents[1] / "reference_newton" / "rd" / "cases.json"
    source = json.loads(casefile.read_text(encoding="utf8"))
    cases = source["cases"] if isinstance(source,dict) else source
    p = next(c["inputs"] for c in cases if c["id"]=="RD-PRISMATIC-LINK-STEP")
    R = np.array([[1.,0,0],[0,0.8,-0.6],[0,0.6,0.8]]) if rotate else np.eye(3)
    # Rotate the physical fixture, then translate the shaft towards different material nodes.
    x = np.array(p["vertices_m"],dtype=float) @ R.T
    origin = R @ (np.array(p["link_origin_m"]) + [0.5 if shifted else 0.0, 0.25 if shifted else 0.0, lift])
    ends = np.array(p["link_axis_ends_m"]) @ R.T
    ax = R @ np.array([0.,0.,1.])
    robot = ph.RobotModel(np.array([-1]),("prismatic",),origin[None,:],np.array([[0.,0.,0.,1.]]),
                          ax[None,:],np.array([p["link_mass_kg"]]),np.zeros((1,3)),np.ones((1,3)),np.zeros(3))
    link = LinkCapsule("beam","arm",0,*ends,p["link_radius_m"])
    contacts = RobotContactSystem({"arm":robot},capsules=(link,))
    masses = np.array([1.,1.5,0.75]) if shifted else np.array(p["node_masses_kg"],float)
    cloth = ph.ParticleModel(1 / masses,np.zeros(3))
    scene = ph.SceneModel((ph.DomainBinding("cloth","particles",cloth), ph.DomainBinding("arm","robot",robot)))
    triangle = ph.TrianglePrismSurface("sheet","cloth",(2,0,1) if reverse else (0,1,2),half_thickness_m=p["half_thickness_m"])
    distance = ph.DistanceConstraints(np.array([[0,1]]),np.array([2.0]),np.array([1.0]))
    material = ph.ClothLinkMaterial(p["layer_width_m"],p["stiffness_pa_per_m"],
                                    0.2 if moving else 0.0,0.35 if moving else 0.0,0.05)
    program = ph.CompliantClothContactProgram(scene,"cloth",ClothMaterial(distance=distance),(triangle,),x,(),
                           robot_contact=contacts, max_penetration_m=0.01, cloth_link_material=material)
    v = np.tile(R @ [0.1,0.,-0.02],(3,1)) if moving else np.zeros((3,3))
    state = ph.SceneState({"cloth":ph.ParticleState(x,v),"arm":ph.RobotState(np.zeros(1),np.array([0.04 if moving else 0.0]))})
    loads = ph.SceneInputs({"cloth":ph.StepInputs.zeros(3),"arm":ph.RobotInputs.zeros(1)})
    return program,state,loads


@pytest.mark.parametrize("mode",["interpret","cuda"])
def test_grouped_force_uses_actual_material_and_two_transposes(mode,tmp_path):
    mod = api()
    first, clear = fixture(),fixture(lift=0.5)
    worlds = [(29,mod.ClothLinkWorld(*first[:2])),(3,mod.ClothLinkWorld(*clear[:2]))]
    with mod.TyxWorldContactBatch(worlds,mode,cache_root=tmp_path) as batch:
        before = batch.snapshot()
        result = batch.sample()
        assert tuple(result) == (29,3) and batch.snapshot() == before
        contact, separated = result[29],result[3]
        assert contact.gap_m == pytest.approx(3/32) and contact.compression_m == pytest.approx(1/32)
        assert contact.normal_force_n == pytest.approx(2)
        assert contact.joint_effort_n == pytest.approx(2)
        np.testing.assert_allclose(contact.node_forces_n,np.tile([0,0,-2/3],(3,1)),atol=2e-6)
        np.testing.assert_allclose(contact.link_force_n,[0,0,2],atol=2e-6)
        assert sorted(s.axis_parameter for s in contact.sites) == [0,1]
        for site in contact.sites:
            assert len(site.law_row) == 13 and site.law_row[9] == pytest.approx(0.5)
            np.testing.assert_allclose(site.law_row[6:9], [1/8,3/32,64],atol=2e-6)
            assert sum(w for _,w in site.face_weights) == pytest.approx(1)
        assert separated.sites == () and separated.normal_force_n == 0
        assert separated.gap_m == pytest.approx(3/32+0.5)
        assert batch.stats()["completed_kernels"] == 3



def match_state(point, ref, *, atol=8e-7):
    np.testing.assert_allclose(point.position,ref.domains["cloth"].position,rtol=2e-6,atol=atol)
    np.testing.assert_allclose(point.velocity,ref.domains["cloth"].velocity,rtol=2e-6,atol=atol)
    np.testing.assert_allclose([point.q,point.qd],[ref.domains["arm"].q[0],ref.domains["arm"].v[0]],rtol=2e-6,atol=atol)


@pytest.mark.parametrize("mode",["interpret","cuda"])
def test_two_newtons_drive_both_sides_through_the_midpoint_window(mode,tmp_path):
    mod = api(); program,initial,loads = fixture()
    h = 1/256
    with mod.TyxWorldContactBatch([(11,mod.ClothLinkWorld(program,initial))],mode,cache_root=tmp_path) as batch, program.stepper(ph.StepSettings(substeps=1,position_iterations=1,velocity_iterations=1)) as reference:
        out=batch.step(h,{11:loads})[11]
        assert out.accepted and out.required_sites == 2
        state=batch.snapshot()[11]
        expected=reference.step(initial,loads,h)
        assert expected.report.accepted
        match_state(state,expected.state)
        assert state.tick==1 and state.q==pytest.approx(1/32768,abs=1e-10)
        assert state.qd==pytest.approx(1/64)
        np.testing.assert_allclose(np.array(state.velocity)[:,2],[-1/384]*3,atol=2e-9)
        audit=out.audit
        assert audit.sample.normal_force_n==pytest.approx(2)
        assert audit.sample.joint_effort_n==pytest.approx(2)
        np.testing.assert_allclose(audit.momentum_change_ns, [0,0,0],atol=1e-8)
        assert audit.external_work_j == 0
        assert audit.kinetic_change_j==pytest.approx(7/98304,abs=2e-10)
        assert audit.contact_work_j==pytest.approx(audit.kinetic_change_j,abs=2e-10)
        assert audit.layer_energy_before_j==pytest.approx(1/32)
        assert audit.layer_energy_after_j < audit.layer_energy_before_j
        # The second physical window starts from the actual, displaced face and slider.
        second=batch.step(h,{11:loads})[11]
        again=reference.step(expected.state,loads,h)
        assert again.report.accepted
        match_state(batch.snapshot()[11],again.state)
        assert 0 < second.audit.sample.normal_force_n < 2


@pytest.mark.parametrize("mode",["interpret","cuda"])
def test_asymmetric_external_work_preserves_world_identity_and_reference(mode,tmp_path):
    from contextlib import ExitStack

    mod=api(); h=1/256
    table={31:fixture(shifted=True,rotate=True,reverse=True,moving=True),4:fixture(lift=0.5),19:fixture(),8:fixture(shifted=True)}
    table[31][2].domains["cloth"].force[1]=[1.0, -0.25, 0.5]
    table[31][2].domains["arm"].effort[0]=0.375
    table[4][2].domains["arm"].effort[0]=-0.5
    inputs={key:data[2] for key,data in table.items()}
    with ExitStack() as stack:
        batch=stack.enter_context(mod.TyxWorldContactBatch([(k,mod.ClothLinkWorld(p,s)) for k,(p,s,_) in table.items()],mode,cache_root=tmp_path))
        refs={k:stack.enter_context(p.stepper(ph.StepSettings(substeps=1,position_iterations=1,velocity_iterations=1))) for k,(p,_,_) in table.items()}
        refstates={k:v[1] for k,v in table.items()}
        for _ in range(3):
            old=batch.snapshot()
            results=batch.step(h,inputs)
            for key in table:
                res=refs[key].step(refstates[key],inputs[key],h)
                assert res.report.accepted and results[key].accepted
                refstates[key]=res.state
                observed=batch.snapshot()[key]
                match_state(observed,res.state)
                audit=results[key].audit
                dx=np.array(observed.position)-np.array(old[key].position)
                dq=observed.q-old[key].q
                expected_work=(inputs[key].domains["cloth"].force*dx).sum()+inputs[key].domains["arm"].effort[0]*dq
                assert audit.external_work_j==pytest.approx(expected_work,abs=2e-8)
                assert abs(audit.kinetic_change_j-audit.contact_work_j-audit.external_work_j-audit.elastic_work_j) < 3e-6
                np.testing.assert_allclose(audit.momentum_change_ns,np.array(audit.external_impulse_ns)+audit.support_impulse_ns,atol=2e-6)
            sample=results[31].audit.sample
            assert sample.law_width_m==pytest.approx(1/8)
            assert any(row.law_row[10]>0 and row.law_row[11]>0 for row in sample.sites)
            assert not np.allclose(np.array(sample.node_forces_n)[0],np.array(sample.node_forces_n)[1])
            assert results[4].audit.sample.sites == ()
        assert batch.stats()["completed_kernels"]==18


@pytest.mark.parametrize("mode",["interpret","cuda"])
def test_local_capacity_retries_reset_and_checkpoint_replay(mode,tmp_path):
    mod=api(); h=1/256
    a=fixture(); b=fixture(lift=0.5)
    with mod.TyxWorldContactBatch([(5,mod.ClothLinkWorld(*a[:2])),(41,mod.ClothLinkWorld(*b[:2]))],mode,cache_root=tmp_path) as batch:
        loads={5:a[2],41:b[2]}; loads[41].domains["arm"].effort[0]=0.75
        first=batch.step(h,loads)
        old=batch.snapshot()
        assert all(r.accepted for r in first.values())
        refused=batch.step(h,loads,site_capacity={5:1,41:0})
        assert not refused[5].accepted and refused[5].required_sites==2
        assert refused[5].audit==first[5].audit and batch.snapshot()[5]==old[5]
        assert refused[41].accepted and batch.snapshot()[41].tick==2
        retried=batch.step(h,loads,site_capacity={5:2,41:0})
        after=batch.snapshot()
        assert retried[5].accepted and after[5].tick==2 and after[41].tick==3
        batch.restore({5:old[5]})
        assert batch.snapshot()[5]==old[5] and batch.snapshot()[41]==after[41]
        redo=batch.step(h,loads)
        assert batch.snapshot()[5]==after[5] and redo[5].audit==retried[5].audit
        neighbor=batch.snapshot()[41]
        batch.reset([5])
        assert batch.snapshot()[5].tick==0 and batch.snapshot()[41]==neighbor


def test_real_batched_kernels_and_measured_transfer_at_w2_and_w4(tmp_path):
    import tyx

    mod=api(); evidence={}
    for W in (2,4):
        fixtures=[fixture(lift=0.5 if i==1 else 0.0,shifted=i>=2) for i in range(W)]
        with mod.TyxWorldContactBatch([(17+3*i,mod.ClothLinkWorld(*v[:2])) for i,v in enumerate(fixtures)],"cuda",cache_root=tmp_path / "cache") as batch:
            inputs={17+3*i:v[2] for i,v in enumerate(fixtures)}
            before=tyx.runtime_stats()
            result=batch.step(1/256,inputs)
            after=tyx.runtime_stats()
            assert all(v.accepted for v in result.values())
            launches=after.cuda_kernel_launches-before.cuda_kernel_launches
            assert launches==6
            evidence[W]=dict(launches=launches, h2d=after.cuda_h2d_bytes-before.cuda_h2d_bytes, d2h=after.cuda_d2h_bytes-before.cuda_d2h_bytes, stats=batch.stats(), world_keys=list(result), contacts=[len(v.audit.sample.sites) for v in result.values()])
            assert evidence[W]["h2d"]>0 and evidence[W]["d2h"]>0
    (tmp_path/"world-contact-window.json").write_text(json.dumps(evidence,indent=2),encoding="utf8")
