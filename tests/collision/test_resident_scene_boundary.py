"""S1/S2 tests first: logical ownership over real four-family accepted banks."""
import gzip
import importlib.util
import json
import math
from collections.abc import Mapping
from contextlib import closing,contextmanager
from dataclasses import fields,is_dataclass,replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import phymeo as ph
import test_resident_boundary_query as m1

FAMILIES=("ordinary","affine","floating","floating_affine")
H=1/1024
BUDGET=64*1024*1024


@lru_cache(maxsize=1)
def reference():
    path=Path(__file__).parents[1]/"reference_newton/rr/resident_scene_boundary_reference.py"
    spec=importlib.util.spec_from_file_location("scene_boundary_reference",path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def encode(value):
    if isinstance(value,np.ndarray):
        return dict(dtype=value.dtype.str,shape=value.shape,bytes_hex=value.tobytes().hex())
    if isinstance(value,np.generic): return encode(value.item())
    if isinstance(value,bytes): return dict(bytes_hex=value.hex())
    if type(value) is object: return dict(opaque_identity=id(value))
    if is_dataclass(value): return {f.name:encode(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,Mapping): return dict(mapping_entries=[[encode(k),encode(v)] for k,v in value.items()])
    if isinstance(value,(list,tuple)): return [encode(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value): return dict(nonfinite=repr(value))
    if value is None or isinstance(value,(str,bool,int,float)): return value
    raise TypeError("unrecorded scene evidence "+type(value).__name__)


def canonical(value): return json.dumps(encode(value),sort_keys=True,allow_nan=False)


@contextmanager
def saved(path,identity,mode,**inputs):
    record=dict(schema="scene-boundary-source-v1",identity=identity,mode=mode,inputs=inputs,steps=[],queries=[],complete=False)
    try:
        yield record
        record["complete"]=True
    except BaseException as error:
        record["failure"]=dict(type=type(error).__name__,message=str(error)); raise
    finally:
        with gzip.open(path/(identity+".json.gz"),"wt",encoding="utf-8") as stream:
            json.dump(encode(record),stream,allow_nan=False,sort_keys=True)


def make_asset_tree(path,family,anchor,*,cycle=False,folded=False):
    ref=reference(); floating=family.startswith("floating")
    source,data=ref.write_asset(path,anchor=anchor,floating=floating,cycle=cycle,folded=folded)
    raw=(ph.load_floating_mimic_urdf(source,gravity=np.zeros(3),collapse_fixed_joints=folded)
         if floating else ph.load_mimic_urdf(source,gravity=np.zeros(3)))
    full=raw.tree; asset=raw if "affine" in family else full
    q=np.array((1/512,.004-1/512)); v=np.array((1/1024,-1/1024))
    independent=1 if "affine" in family else 2
    gains=dict(kp=np.zeros(independent),kd=np.zeros(independent),effort_limit=np.ones(independent),delay_steps=0)
    frame=ref.CYCLE_R if cycle else np.eye(3)
    root=np.r_[anchor,(.5,.5,.5,.5) if cycle else (0,0,0,1)]
    root_v=np.r_[frame@np.array((1/128,0,0)),np.zeros(3)]
    if family=="ordinary":
        program=ph.DrivenRobotProgram(full.model,**gains)
        tree=ph.FixedRobotTree(program,program.initial_state(ph.RobotState(q,v)))
    elif family=="affine": tree=ph.AffineRobotTree(asset,ph.RobotState(q[:1],v[:1]),**gains)
    elif family=="floating": tree=ph.FloatingRobotTree(full.model,ph.FloatingRobotState(np.r_[root,q],np.r_[root_v,v]),**gains)
    else: tree=ph.FloatingAffineRobotTree(asset,ph.FloatingRobotState(np.r_[root,q[:1]],np.r_[root_v,v[:1]]),**gains)
    data.update(family=family)
    return asset,tree,data


def build(path,*,families=("ordinary","ordinary"),worlds=(83,),reverse=False,cycle=False,folded=False,asset_mode=True,free_object=False,static=False):
    ref=reference(); participants=[]; shapes=[]; bindings=[]; definitions={}; data={}; static_ids=[]
    classes=(ph.ResidentTreeBinding,ph.ResidentAffineTreeBinding,ph.ResidentFloatingTreeBinding,ph.ResidentFloatingAffineTreeBinding)
    for wi,world in enumerate(worlds):
        frame=ref.CYCLE_R if cycle else np.eye(3)
        base=np.array((1+2*wi,-.75,.125),np.float64)
        for index,family in enumerate(families):
            pid=ref.PIDS[index]; key=ph.ResidentParticipantKey(world,pid)
            anchor=base+frame@np.array((index/8,0,0))
            asset,tree,description=make_asset_tree(path/f"{world}-{pid}",family,anchor,cycle=cycle,folded=folded and index==0)
            full=asset.tree if "affine" in family else asset
            body_count=3 if family.startswith("floating") else 2
            owner=ph.ResidentParticipant(key,tree,tuple(100*pid+i for i in range(body_count)))
            participants.append(owner); data[key]=description
            physical=getattr(owner.compiled,"physical",owner.compiled)
            for name,which in (("surface","driver"),("round","follower")):
                original=next(c.shape for c in full.collisions if c.name==name)
                group=(0 if name=="surface" else 1)+int(family.startswith("floating"))
                shape_key=ph.CollisionKey(world,owner.body_ids[group],0)
                shape=replace(original,key=shape_key)
                local=ref.local_surface(description["folded"]) if name=="surface" else ref.ROUND
                if asset_mode:
                    bound=classes[FAMILIES.index(family)].from_asset(asset,original.key,key=shape_key,world=world)
                else:
                    options=dict(topology=physical.topology)
                    if family=="floating_affine": options["graph_token"]=owner.graph_token
                    bound=classes[FAMILIES.index(family)](shape_key,world,group,local,**options)
                shapes.append(shape); bindings.append(ph.ResidentParticipantBinding(key,bound))
                definitions[shape_key]=dict(owner=key,which=which,local=local,description=description)
            # A second finite primitive belongs to the same actual follower body.
            group=1+int(family.startswith("floating")); shape_key=ph.CollisionKey(world,owner.body_ids[group],1)
            local=np.vstack((ref.ROUND,ref.ROUND+(1/4,0,0)))
            shape=ph.CollisionShape(shape_key,"capsule",(0,1),radius=ref.RADIUS,revision=3)
            options=dict(topology=physical.topology)
            if family=="floating_affine": options["graph_token"]=owner.graph_token
            bound=classes[FAMILIES.index(family)](shape_key,world,group,local,**options)
            shapes.append(shape); bindings.append(ph.ResidentParticipantBinding(key,bound))
            definitions[shape_key]=dict(owner=key,which="follower",local=local,description=description)
        if free_object:
            key=ph.ResidentParticipantKey(world,97)
            position=base+frame@np.array((.125,.5,.5))
            model=ph.FloatingRobotModel(ph.RigidBodyModel(np.array((1.,)),np.full((1,3),1/8),np.zeros(3)),np.zeros(3))
            state=ph.FloatingRobotState(np.r_[position,(.5,.5,.5,.5) if cycle else (0,0,0,1)],np.r_[frame@np.array((-1/64,0,0)),np.zeros(3)])
            tree=ph.FloatingRobotTree(model,state,kp=np.zeros(0),kd=np.zeros(0),effort_limit=np.zeros(0))
            owner=ph.ResidentParticipant(key,tree,(9700,)); participants.append(owner)
            shape_key=ph.CollisionKey(world,9700,0); local=np.zeros((1,3))
            shape=ph.CollisionShape(shape_key,"sphere",(0,),radius=ref.RADIUS)
            bound=ph.ResidentFloatingTreeBinding(shape_key,world,0,local,topology=owner.compiled.topology)
            shapes.append(shape); bindings.append(ph.ResidentParticipantBinding(key,bound))
            description=dict(family="free",anchor=position,cycle=cycle,folded=False)
            data[key]=description; definitions[shape_key]=dict(owner=key,which="root",local=local,description=description)
        if static:
            shape_key=ph.CollisionKey(world,9999,0); static_ids.append((world,9999))
            local=ref.VERTICES+np.array((-.5,0,0)); points=base+local@frame.T
            shape=ph.CollisionShape(shape_key,"mesh",(0,1,2,3),triangles=ref.FACES)
            shapes.append(shape); bindings.append(ph.ResidentStaticBinding(shape_key,ph.CollisionMotion(points)))
            definitions[shape_key]=dict(owner=None,points=points)
    if reverse: participants.reverse(); shapes.reverse(); bindings.reverse()
    return dict(participants=tuple(participants),model=ph.CollisionModel(tuple(shapes)),bindings=tuple(bindings),
                definitions=definitions,data=data,static_body_ids=tuple(static_ids))


def registration_record(built):
    return dict(model=built["model"],bindings=built["bindings"],definitions=built["definitions"],
        data=built["data"],static_body_ids=built["static_body_ids"],participants=tuple(
            dict(key=p.key,body_ids=p.body_ids,family=p.family,compiled=p.compiled,
                 model_token=p.model_token,graph_token=p.graph_token,fingerprint=p.fingerprint)
            for p in built["participants"]))

def expected(states,built):
    result={}
    for key,definition in built["definitions"].items():
        owner=definition["owner"]
        result[key]=(definition["points"] if owner is None else reference().points(
            states[key.world].participants[owner],definition["description"]["family"],
            definition["description"],definition["which"],definition["local"]))
    return result


def step(scene,record):
    attempt=dict(h=H,before=scene.snapshot()); record["steps"].append(attempt)
    try:
        attempt["result"]=scene.step(H)
        assert all(attempt["result"].values())
    finally: attempt["after"]=scene.snapshot()
    return attempt["after"]


def query(scene,plan,record,margin=.5):
    attempt=dict(before=scene.snapshot(),before_stats=plan.stats(),margin=margin)
    record["queries"].append(attempt)
    try:
        result=plan.query(margin=margin); attempt["result"]=result
        return result
    finally:
        attempt.update(after=scene.snapshot(),last_attempt=plan.last_attempt,stats=plan.stats())
        assert canonical(attempt["before"])==canonical(attempt["after"])


def check(scene,plan,packet,built):
    ref=m1.reference(); states=scene.snapshot(); material=expected(states,built)
    assert packet.source_kind=="resident-scene-rigid-device-v1"
    assert packet.world_keys==scene.world_keys
    assert packet.state_epoch==scene.state_epoch
    assert packet.owner is scene._owner and plan.require_current(packet) is packet
    np.testing.assert_array_equal(packet.ticks,[states[w].tick for w in packet.world_keys])
    source=packet.source; assert source["scene_registry_fingerprint"]==scene.layout.fingerprint
    assert source["scene_owner"] is scene._owner
    assert source["participant_keys"]==scene.participant_keys
    assert source["anchor_shifts_exact"].shape==(len(built["model"].shapes),3)
    assert source["uploaded_constants"]["anchor_shifts"].shape==(len(built["model"].shapes),9)
    assert source["stamps"]["native_poses"].shape==(len(built["model"].shapes),12)
    assert len(source["participant_addresses"])==len(scene.participant_keys)
    m1.immutable(source)
    by_key={shape.key:shape for shape in built["model"].shapes}
    for key,points in material.items():
        anchor=np.asarray(source["world_anchors"][key.world])
        length=max(1.,float(np.max(np.linalg.norm(points-points[0],axis=1))))
        np.testing.assert_allclose(source["gathered_points"][key],points-anchor,atol=2e-4*length,rtol=0)
    count=0
    for group in packet.groups:
        shape=by_key[group.surface]; vertices=material[group.surface]
        facets=tuple(tuple(shape.vertex_ids.index(v) for v in f) for f in group.facet_vertices)
        axes=[ref.axis(material[k]) for k in group.round_keys]
        radii=[by_key[k].radius for k in group.round_keys]
        wanted=ref.aabb_candidates(vertices,facets,axes,radii,packet.margin)
        assert tuple(map(tuple,group.candidates.tolist()))==wanted
        f,r=len(facets),len(axes); words=(f+29)//30
        assert group.bvh_raw.shape==(r,words+3)
        assert np.all(group.bvh_raw[:,-1]==2*f-1)
        bits=tuple((s,t) for s in range(r) for t in range(f) if int(group.bvh_raw[s,t//30])&(1<<(t%30)))
        assert bits==wanted
        narrow=group.packet; assert narrow.source_binding==packet.source_binding
        assert len(narrow.rows)==len(wanted)
        for row,(segment,facet) in zip(narrow.rows,wanted,strict=True):
            assert (row.segment_id,row.facet_id)==(segment,facet)
            assert row.vertex_ids==group.facet_vertices[facet]
            ref.assert_row(row,vertices[list(facets[facet])],axes[segment],radii[segment],narrow.origin_m)
        count+=len(wanted)
    assert packet.required_candidates==count
    shapes=built["model"].shapes
    assert len(packet.pairs)==sum(a.key.world==b.key.world for i,a in enumerate(shapes) for b in shapes[i+1:])
    assert all(pair.first.world==pair.second.world for pair in packet.pairs)
    assert plan.stats()["live_bytes"]==plan.stats()["persistent_bytes"]
    return count


def prepare(scene,built,**options):
    return scene.prepare_boundary_query(built["model"],built["bindings"],
        limits=options.get("limits",ph.BoundaryDistanceLimits()),memory_limit_bytes=options.get("memory_limit_bytes",BUDGET))


def run_three(tmp_path,mode,built,identity):
    with saved(tmp_path,identity,mode,registration=registration_record(built)) as record:
        with ph.ResidentSceneBatch(built["participants"],mode=mode,cache_root=tmp_path,static_body_ids=built["static_body_ids"]) as scene:
            with closing(prepare(scene,built)) as plan:
                previous=None
                for tick in (1,2,3):
                    states=step(scene,record)
                    if previous is not None:
                        with pytest.raises(ValueError): plan.require_current(previous)
                    result=query(scene,plan,record)
                    assert check(scene,plan,result,built)>0
                    assert all(state.tick==tick for state in states.values())
                    previous=result
                record["stats"]=dict(scene=scene.stats(),plan=plan.stats())
    return record


@pytest.mark.tyx
@pytest.mark.parametrize("family",FAMILIES)
@pytest.mark.parametrize("asset_mode",(True,False),ids=("asset","direct"))
def test_four_families_cross_participant_three_steps(family,asset_mode,tyx_mode,tmp_path):
    built=build(tmp_path/"assets",families=(family,family),asset_mode=asset_mode)
    run_three(tmp_path,tyx_mode,built,f"two-{family}-{asset_mode}")


@pytest.mark.tyx
@pytest.mark.parametrize("reverse",(False,True))
def test_heterogeneous_sparse_worlds_and_original_local_ids(reverse,tyx_mode,tmp_path):
    built=build(tmp_path/"assets",families=FAMILIES,worlds=(83,17),reverse=reverse)
    run_three(tmp_path,tyx_mode,built,"heterogeneous-"+str(reverse))


@pytest.mark.tyx
@pytest.mark.parametrize("cycle",(False,True))
def test_two_arms_fixed_tool_free_object_and_static_surface(cycle,tyx_mode,tmp_path):
    built=build(tmp_path/"assets",families=("affine","floating_affine"),folded=True,
                free_object=True,static=True,cycle=cycle)
    run_three(tmp_path,tyx_mode,built,"arms-tool-object-"+str(cycle))


@pytest.mark.tyx
def test_registration_order_preserves_logical_geometry(tyx_mode,tmp_path):
    results=[]
    for index,reverse in enumerate((False,True)):
        directory=tmp_path/str(index); directory.mkdir()
        built=build(directory/"assets",families=("ordinary","floating"),reverse=reverse)
        results.append(run_three(directory,tyx_mode,built,"permutation"))
    for left,right in zip(results[0]["queries"],results[1]["queries"],strict=True):
        a,b=left["result"],right["result"]
        assert a.world_keys==b.world_keys and a.required_candidates==b.required_candidates
        for ga,gb in zip(a.groups,b.groups,strict=True):
            assert ga.surface==gb.surface and ga.round_keys==gb.round_keys
            for ra,rb in zip(ga.packet.rows,gb.packet.rows,strict=True):
                assert canonical(ra)==canonical(rb)


@pytest.mark.tyx
def test_partial_reset_restore_and_next_real_step(tyx_mode,tmp_path):
    built=build(tmp_path/"assets",families=("ordinary","floating_affine"),worlds=(83,17))
    with saved(tmp_path,"reset-restore",tyx_mode,registration=registration_record(built)) as record:
        with ph.ResidentSceneBatch(built["participants"],mode=tyx_mode,cache_root=tmp_path) as scene:
            with closing(prepare(scene,built)) as plan:
                step(scene,record); accepted=scene.snapshot(); old=query(scene,plan,record)
                scene.reset((17,)); after=scene.snapshot()
                assert canonical(after[83])==canonical(accepted[83]) and after[17].tick==0
                with pytest.raises(ValueError): plan.require_current(old)
                scene.restore({17:accepted[17]},expected_epoch=scene.state_epoch)
                step(scene,record); result=query(scene,plan,record); check(scene,plan,result,built)
                assert all(value.tick==2 for value in scene.snapshot().values())


@pytest.mark.tyx
def test_empty_geometry_still_binds_all_logical_ticks(tyx_mode,tmp_path):
    built=build(tmp_path/"assets",families=FAMILIES,worlds=(83,17))
    empty=dict(built,model=ph.CollisionModel(()),bindings=(),definitions={})
    with saved(tmp_path,"empty-all-participants",tyx_mode,registration=registration_record(empty)) as record:
        with ph.ResidentSceneBatch(built["participants"],mode=tyx_mode,cache_root=tmp_path) as scene:
            with closing(prepare(scene,empty,memory_limit_bytes=1)) as plan:
                step(scene,record); result=query(scene,plan,record)
                check(scene,plan,result,empty)
                assert result.required_candidates==0 and len(result.groups)==0
                assert plan.stats()["live_bytes"]==0 and result.ticks.tolist()==[1,1]


@pytest.mark.tyx
def test_no_host_current_geometry_as_query_input(tyx_mode,tmp_path,monkeypatch):
    built=build(tmp_path/"assets",families=("ordinary","floating_affine"))
    with saved(tmp_path,"no-host-current-points",tyx_mode,registration=registration_record(built)) as record:
        with ph.ResidentSceneBatch(built["participants"],mode=tyx_mode,cache_root=tmp_path) as scene:
            with closing(prepare(scene,built)) as plan:
                step(scene,record)
                def forbidden(*args,**kwargs): raise AssertionError("HOST geometry cannot feed this source")
                monkeypatch.setattr(scene._service,"geometry",forbidden)
                monkeypatch.setattr(scene._service,"_read_packets",forbidden)
                result=query(scene,plan,record); check(scene,plan,result,built)
