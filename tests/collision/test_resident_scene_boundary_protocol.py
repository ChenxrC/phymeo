"""Scene boundary ownership/capacity/publication; actual FK is never HOST input."""
from contextlib import closing
from dataclasses import replace
from types import SimpleNamespace
import gc
import weakref

import numpy as np
import pytest
import phymeo as ph
import test_resident_scene_boundary as s
import test_resident_boundary_query_protocol as old_protocol


def minimal(built):
    # One mesh from arm 7, one sphere from arm 29; other real bodies stay owned.
    chosen=tuple(shape for shape in built["model"].shapes if
                 (shape.kind=="mesh" and shape.key.body//100==7) or
                 (shape.kind=="sphere" and shape.key.body//100==29))
    keys={shape.key for shape in chosen}
    return dict(built,model=ph.CollisionModel(chosen),bindings=tuple(
        b for b in built["bindings"] if b.binding.key in keys),definitions={
        key:value for key,value in built["definitions"].items() if key in keys})


def unchanged(scene,plan,old,before):
    assert plan.last_result is old
    assert s.canonical(scene.snapshot())==s.canonical(before)
    assert plan.stats()["live_bytes"]==plan.stats()["persistent_bytes"]


@pytest.mark.tyx
@pytest.mark.parametrize("fault,reason",(
    ("missing","exactly one"),("participant","unknown participant"),
    ("body","body owner"),("group","body owner"),("family","physical layout"),
    ("topology","complete physical topology"),("revision","asset source shape"),
    ("static","declared static")))
def test_registration_rejection_precedes_allocation(fault,reason,tyx_mode,tmp_path,monkeypatch):
    from phymeo.backends._resident_boundary_query import ResidentBoundaryQueryPlan
    built=s.build(tmp_path/"assets",families=("ordinary","ordinary"),asset_mode=fault=="revision")
    with ph.ResidentSceneBatch(built["participants"],mode=tyx_mode,cache_root=tmp_path) as scene:
        model=built["model"]; bindings=list(built["bindings"]); wrapper=bindings[0]; bound=wrapper.binding
        if fault=="missing": bindings.pop()
        elif fault=="participant": bindings[0]=ph.ResidentParticipantBinding(ph.ResidentParticipantKey(83,999),bound)
        elif fault=="body":
            key=ph.CollisionKey(83,9911,0)
            model=ph.CollisionModel(tuple(replace(x,key=key) if x.key==bound.key else x for x in model.shapes))
            bindings[0]=ph.ResidentParticipantBinding(wrapper.participant,replace(bound,key=key))
        elif fault=="group": bindings[0]=ph.ResidentParticipantBinding(wrapper.participant,replace(bound,group_index=1))
        elif fault=="family": bindings[0]=ph.ResidentParticipantBinding(wrapper.participant,
            ph.ResidentAffineTreeBinding(bound.key,83,bound.group_index,bound.local_points_m,topology=bound.topology))
        elif fault=="topology": bindings[0]=ph.ResidentParticipantBinding(wrapper.participant,replace(bound,topology=("wrong",)))
        elif fault=="revision": model=ph.CollisionModel(tuple(replace(x,revision=x.revision+1) if x.key==bound.key else x for x in model.shapes))
        else: bindings[0]=ph.ResidentStaticBinding(bound.key,ph.CollisionMotion(s.reference().VERTICES))
        def forbidden(*args,**kwargs): raise AssertionError("invalid source reached allocation")
        monkeypatch.setattr(ResidentBoundaryQueryPlan,"_allocate",forbidden)
        before=scene.snapshot()
        with pytest.raises(ValueError,match=reason):
            scene.prepare_boundary_query(model,tuple(bindings),limits=ph.BoundaryDistanceLimits(),memory_limit_bytes=s.BUDGET)
        assert s.canonical(before)==s.canonical(scene.snapshot())
        assert not scene._service._batch._boundary_plans


@pytest.mark.tyx
@pytest.mark.parametrize("resource",("memory","candidates","groups"))
def test_exact_and_one_less_limits_are_whole_query(resource,tyx_mode,tmp_path):
    built=minimal(s.build(tmp_path/"assets")); expected=s.reference().memory(2,5,((4,1),),(4,))
    with s.saved(tmp_path,"limits-"+resource,tyx_mode,registration=s.registration_record(built),expected=expected) as record:
        with ph.ResidentSceneBatch(built["participants"],mode=tyx_mode,cache_root=tmp_path) as scene:
            limits=ph.BoundaryDistanceLimits(groups=1,candidates=4,vertices=4,faces=4,segments=1)
            with closing(s.prepare(scene,built,limits=limits,memory_limit_bytes=expected["peak_required_bytes"])) as plan:
                old=s.query(scene,plan,record); assert s.check(scene,plan,old,built)==4
                assert all(plan.stats()[key]==value for key,value in expected.items())
                small_limit=replace(limits,**{resource:getattr(limits,resource)-1}) if resource!="memory" else limits
                budget=expected["peak_required_bytes"]-int(resource=="memory")
                with closing(s.prepare(scene,built,limits=small_limit,memory_limit_bytes=budget)) as small:
                    before=scene.snapshot()
                    with pytest.raises((ph.BoundaryDistanceCapacityError,MemoryError)): s.query(scene,small,record)
                    unchanged(scene,small,None,before)
                    assert small.last_attempt.required_capacities["candidates"]==4
                    assert not any(label.endswith("/narrow") for label in small.last_attempt.counters["launch_labels"])
                    assert plan.last_result is old


@pytest.mark.tyx
@pytest.mark.parametrize("fault",("last-bvh","last-narrow","common-frame","packet"))
def test_query_failure_keeps_prior_packet_but_not_prior_physics(fault,tyx_mode,tmp_path,monkeypatch):
    built=s.build(tmp_path/"assets",families=("ordinary","floating_affine"))
    with s.saved(tmp_path,"late-"+fault,tyx_mode,registration=s.registration_record(built)) as record:
        with ph.ResidentSceneBatch(built["participants"],mode=tyx_mode,cache_root=tmp_path) as scene:
            with closing(s.prepare(scene,built)) as plan:
                old=s.query(scene,plan,record); s.step(scene,record); before=scene.snapshot()
                read,build=plan._download,plan._build_packet; fired=[]
                def bad_read(array,label):
                    value=read(array,label)
                    if label=="group:1/bvh" and fault=="last-bvh" or label=="group:1/meta" and fault=="last-narrow":
                        fired.append(label); raise RuntimeError("late query read")
                    if label=="source/native_poses" and fault=="common-frame":
                        fired.append(label); value[-1,0]=np.nan
                    return value
                def bad_build(*args,**kwargs):
                    build(*args,**kwargs); fired.append("packet"); raise RuntimeError("late packet")
                monkeypatch.setattr(plan,"_download",bad_read)
                if fault=="packet": monkeypatch.setattr(plan,"_build_packet",bad_build)
                with pytest.raises((ValueError,RuntimeError)): s.query(scene,plan,record)
                assert fired; unchanged(scene,plan,old,before)
                assert all(point.tick==1 for point in before.values())
                with pytest.raises(ValueError): plan.require_current(old)
                monkeypatch.setattr(plan,"_download",read); monkeypatch.setattr(plan,"_build_packet",build)
                new=s.query(scene,plan,record); s.check(scene,plan,new,built)
                assert new.sequence==old.sequence+1


@pytest.mark.tyx
@pytest.mark.parametrize("operation",("step","reset","restore","controls","close"))
def test_scene_writers_are_blocked_during_actual_source_loan(operation,tyx_mode,tmp_path,monkeypatch):
    built=minimal(s.build(tmp_path/"assets"))
    with s.saved(tmp_path,"reentry-"+operation,tyx_mode,registration=s.registration_record(built)) as record:
        with ph.ResidentSceneBatch(built["participants"],mode=tyx_mode,cache_root=tmp_path) as scene:
            with closing(s.prepare(scene,built)) as plan:
                before=scene.snapshot(); read=plan._download; fired=[]
                operations=dict(step=lambda:scene.step(s.H),reset=lambda:scene.reset((83,)),
                    restore=lambda:scene.restore(before,expected_epoch=scene.state_epoch),
                    controls=lambda:scene.set_controls({}),close=scene.close)
                def callback(array,label):
                    if not fired:
                        fired.append(label)
                        with pytest.raises(RuntimeError,match="boundary source loan"): operations[operation]()
                    return read(array,label)
                monkeypatch.setattr(plan,"_download",callback)
                result=s.query(scene,plan,record); assert fired; s.check(scene,plan,result,built)


@pytest.mark.tyx
def test_participant_without_geometry_cannot_have_a_different_tick(tyx_mode,tmp_path,monkeypatch):
    built=s.build(tmp_path/"assets",families=s.FAMILIES)
    minimal_built=minimal(built)
    with s.saved(tmp_path,"unshaped-tick",tyx_mode,registration=s.registration_record(minimal_built)) as record:
        with ph.ResidentSceneBatch(built["participants"],mode=tyx_mode,cache_root=tmp_path) as scene:
            with closing(s.prepare(scene,minimal_built)) as plan:
                old=s.query(scene,plan,record); read=plan._download; fired=[]
                def bad(array,label):
                    value=read(array,label)
                    if label=="source/accepted_ticks": value[-1]+=1; fired.append(label)
                    return value
                before=scene.snapshot(); monkeypatch.setattr(plan,"_download",bad)
                with pytest.raises(RuntimeError,match="participant ticks"): s.query(scene,plan,record)
                assert fired; unchanged(scene,plan,old,before)
                monkeypatch.setattr(plan,"_download",read); new=s.query(scene,plan,record); s.check(scene,plan,new,minimal_built)


@pytest.mark.tyx
def test_wrong_scene_and_full_provenance_cannot_be_current(tyx_mode,tmp_path):
    built=minimal(s.build(tmp_path/"assets"))
    with ph.ResidentSceneBatch(built["participants"],mode=tyx_mode,cache_root=tmp_path) as a:
        with closing(s.prepare(a,built)) as pa:
            old=pa.query(margin=.5)
            with ph.ResidentSceneBatch(built["participants"],mode=tyx_mode,cache_root=tmp_path) as b:
                with closing(s.prepare(b,built)) as pb:
                    with pytest.raises(ValueError): pb.require_current(old)
            for name in ("participant_addresses","scene_registry_fingerprint","anchor_shifts_exact"):
                source=dict(old.source); source[name]=("wrong",)
                with pytest.raises(ValueError): pa.require_current(replace(old,source=source))
            assert pa.require_current(old) is old


@pytest.mark.tyx
def test_filters_use_validated_bodies_and_remain_complete(tyx_mode,tmp_path):
    built=s.build(tmp_path/"assets")
    shapes=list(built["model"].shapes)
    a=next(x.key for x in shapes if x.kind=="mesh" and x.key.body//100==7)
    b=next(x.key for x in shapes if x.kind=="sphere" and x.key.body//100==29)
    blocked=next(x.key for x in shapes if x.kind=="capsule" and x.key.body//100==29)
    shapes=[replace(x,mask=0) if x.key==blocked else x for x in shapes]
    # Shape masks are authored registration metadata; use direct bindings here.
    direct=s.build(tmp_path/"direct",asset_mode=False)
    built=dict(direct,model=ph.CollisionModel(tuple(shapes),excluded=((a,b),)))
    with s.saved(tmp_path,"filters",tyx_mode,registration=s.registration_record(built)) as record:
        with ph.ResidentSceneBatch(built["participants"],mode=tyx_mode,cache_root=tmp_path) as scene:
            with closing(s.prepare(scene,built)) as plan:
                value=s.query(scene,plan,record); s.check(scene,plan,value,built)
                assert {"same_rigid_body","same_family","mask","excluded"}<={p.reason for p in value.pairs}


@pytest.mark.parametrize("phase",("before","after"))
def test_host_unknown_release_strongly_retains_scene_and_runtime(phase,tmp_path,monkeypatch):
    """Actual scene/plan/ledger graph, fake buffer free; no device geometry."""
    import tyx as tx
    from phymeo.backends.mujoco_tyx._resources import quarantined_resources
    def exercise():
        with monkeypatch.context() as patch:
            events=[]; arrays=[]
            built=s.build(tmp_path/"assets",families=("ordinary",))
            scene=ph.ResidentSceneBatch(built["participants"],mode="interpret",cache_root=tmp_path/"cache")
            plan=scene.prepare_boundary_query(ph.CollisionModel(()),(),limits=ph.BoundaryDistanceLimits(),memory_limit_bytes=1024)
            batch=scene._service._batch
            runtime=old_protocol.FakeRuntime(events); batch._runtime=runtime
            batch._resources.callback(runtime.close); batch._resources.callback(events.append,"parent-resource")
            plan._runtime=runtime; plan._mode="cuda"; plan._tx=tx
            def allocate(shape,dtype):
                del shape,dtype
                value=old_protocol.FakeArray(str(len(arrays)),events,phase if len(arrays)==1 else None)
                arrays.append(value); return value
            patch.setattr(tx,"cuda_empty",allocate)
            old=SimpleNamespace(schema="host-only-retained-sentinel"); plan._last_result=old
            ledger=plan._new_query_resources()
            for _ in range(3): plan._allocate(ledger,shape=(2,3),dtype=np.float32)
            with pytest.raises(BaseExceptionGroup,match="UNKNOWN_RELEASE"): plan._close_query_resources()
            for action in (plan.close,scene.close,scene.close):
                with pytest.raises((RuntimeError,BaseExceptionGroup)): action()
            assert all(value.calls==1 for value in arrays)
            assert arrays[1].released is (phase=="after")
            assert arrays[0].released and arrays[2].released
            assert plan.stats()["live_bytes"]==24 and runtime.calls==0 and "parent-resource" not in events
            assert plan.last_result is old and ledger in quarantined_resources()
            assert ledger._runtime.owner.batch is batch and plan._source_adapter._scene is scene
            return tuple(weakref.ref(x) for x in (scene,batch,runtime,arrays[1],ledger)),dict(events=events,unknown_bytes=24,ledger=ledger.state)
    refs,record=exercise(); gc.collect(); assert all(ref() is not None for ref in refs)
    (tmp_path/"scene-unknown.json").write_text(s.canonical(record),encoding="utf-8")