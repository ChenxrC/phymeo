"""M1-A tests first: original asset materials into the accepted device query.

No product asset FK/collision_motion supplies the independent geometry oracle.
All new cases are predeclared before implementation and initially UNEXECUTED.
"""
import importlib.util
from contextlib import closing
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import test_resident_boundary_query as m1
import test_device_convex_query as k0
import phymeo as ph
from phymeo.dynamics.floating_robot import FloatingRobotState

LAYOUTS=m1.LAYOUTS
H=1/1024
BUDGET=32*1024*1024
ROOT=np.array((2,-3,.25,.5,.5,.5,.5),np.float64)


@lru_cache(maxsize=1)
def reference():
    path=Path(__file__).parents[1]/"reference_newton/rr/resident_boundary_asset_reference.py"
    spec=importlib.util.spec_from_file_location("m1a_reference",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(directory,layout,*,folded=False,triangle=False,mass="1",multiplier="-1"):
    ref=reference()
    path=ref.write_urdf(directory,folded=folded,mass=mass,multiplier=multiplier)
    floating=layout.startswith("floating")
    raw=(ph.load_floating_mimic_urdf(path,gravity=np.zeros(3),collapse_fixed_joints=folded)
         if floating else ph.load_mimic_urdf(path,gravity=np.zeros(3)))
    full=raw.tree
    if triangle:
        original=next(c for c in full.collisions if c.name=="surface")
        shape=ph.CollisionShape(original.shape.key,"triangle",original.shape.vertex_ids[:3],revision=3)
        new=replace(original,shape=shape,local_vertices=original.local_vertices[:3])
        full=replace(full,collisions=tuple(new if c is original else c for c in full.collisions),
                     collision_model=ph.CollisionModel(tuple(shape if s.key==shape.key else s
                                                        for s in full.collision_model.shapes)))
        raw=replace(raw,tree=full)
    asset=raw if "affine" in layout else full
    q=np.array((1/512,.004-1/512)); v=np.array((1/1024,-1/1024))
    gains=dict(kp=np.zeros(1 if "affine" in layout else 2),
               kd=np.zeros(1 if "affine" in layout else 2),
               effort_limit=np.ones(1 if "affine" in layout else 2),delay_steps=0)
    root_v=np.r_[ref.rotation(ROOT[3:])@np.array((1/128,0,0)),np.zeros(3)]
    if layout=="ordinary":
        program=ph.DrivenRobotProgram(full.model,**gains)
        tree=ph.FixedRobotTree(program,program.initial_state(ph.RobotState(q,v)))
    elif layout=="affine":
        tree=ph.AffineRobotTree(asset,ph.RobotState(q[:1],v[:1]),**gains)
    elif layout=="floating":
        tree=ph.FloatingRobotTree(full.model,FloatingRobotState(np.r_[ROOT,q],np.r_[root_v,v]),**gains)
    else:
        tree=ph.FloatingAffineRobotTree(asset,FloatingRobotState(np.r_[ROOT,q[:1]],np.r_[root_v,v[:1]]),**gains)
    return asset,tree,dict(layout=layout,folded=folded,triangle=triangle,path=str(path),
                           xml=path.read_text(),obj=(path.parent/"surface.obj").read_text())


def install(asset,layout,world,*,body_alias=False):
    full=asset.tree if "affine" in layout else asset
    cls=(ph.ResidentTreeBinding,ph.ResidentAffineTreeBinding,
         ph.ResidentFloatingTreeBinding,ph.ResidentFloatingAffineTreeBinding)[LAYOUTS.index(layout)]
    shapes,bindings=[],[]
    for name,body in (("surface",101),("round",101 if body_alias else 202)):
        original=next(c.shape for c in full.collisions if c.name==name)
        key=ph.CollisionKey(world,body,0 if name=="surface" else 1)
        shapes.append(replace(original,key=key))
        bindings.append(cls.from_asset(asset,original.key,key=key,world=world))
    return shapes,bindings


def expected(states,descriptions):
    ref=reference(); out={}
    for world,data in descriptions.items():
        out[ph.CollisionKey(world,101,0)]=ref.world_points(states[world],data["layout"],0,
            ref.material_points(data["folded"],data["triangle"]))
        out[ph.CollisionKey(world,202,1)]=ref.world_points(states[world],data["layout"],1,ref.ROUND)
    return out


def check_provenance(result,bindings):
    definitions={item["key"]:item for item in result.source["bindings"]}
    for binding in bindings:
        data=definitions[binding.key]
        assert data["admission"]=="asset"
        proof=data["provenance"]
        assert proof["schema"]=="phymeo-boundary-asset-admission-v1"
        assert proof["source_shape"]==binding._source_shape
        assert proof["source_shape"].key!=binding.key
        assert proof["model_token"]==binding._model_token
        assert proof["group_index"]==binding.group_index
        assert proof["physical_topology"]==data["topology"] and data["topology"] is not None
        np.testing.assert_array_equal(proof["local_points_m"],binding.local_points_m)
        m1.immutable(proof)
        assert binding.topology is None


@pytest.mark.tyx
@pytest.mark.parametrize("layout",LAYOUTS)
@pytest.mark.parametrize("variant",("plain","fixed-tool","program-triangle"))
def test_asset_surface_round_three_real_steps(layout,variant,tyx_mode,tmp_path):
    asset,tree,data=prepare(tmp_path/"asset",layout,folded=variant=="fixed-tool",
                             triangle=variant=="program-triangle")
    world=83
    shapes,bindings=install(asset,layout,world)
    model=ph.CollisionModel(tuple(shapes))
    with m1.saved(tmp_path,"asset-"+layout+"-"+variant,tyx_mode,inputs=data) as record:
        with ph.TyxWorldBatch([(world,tree)],tyx_mode,cache_root=tmp_path) as batch:
            with closing(batch.prepare_boundary_query(model,bindings,
                limits=ph.BoundaryDistanceLimits(),memory_limit_bytes=BUDGET)) as plan:
                record["registration"]=dict(model=model,bindings=bindings,initial=batch.snapshot())
                previous=None
                for step in range(3):
                    before=batch.snapshot()
                    step_result=batch.step(H)
                    record.setdefault("physics",[]).append(dict(before=before,result=step_result,after=batch.snapshot()))
                    assert all(step_result.values())
                    if previous is not None:
                        with pytest.raises(ValueError): plan.require_current(previous)
                    actual=m1.query(batch,plan,record,.5)
                    m1.assert_packet(batch,plan,actual,model,expected(batch.snapshot(),{world:data}))
                    check_provenance(actual,bindings)
                    assert actual.required_candidates>0
                    assert batch.snapshot()[world].tick==step+1
                    previous=actual


@pytest.mark.tyx
def test_four_asset_families_sparse_worlds_reordered_and_reset(tyx_mode,tmp_path):
    worlds=(83,17,107,31); trees={}; descriptions={}; shapes=[]; bindings=[]
    for world,layout in zip(worlds,LAYOUTS,strict=True):
        asset,trees[world],descriptions[world]=prepare(tmp_path/str(world),layout)
        s,b=install(asset,layout,world); shapes.extend(s); bindings.extend(b)
    model=ph.CollisionModel(tuple(shapes[::-1]))
    with m1.saved(tmp_path,"mixed-asset-reset",tyx_mode,inputs=descriptions) as record:
        with ph.TyxWorldBatch(list(trees.items())[::-1],tyx_mode,cache_root=tmp_path) as batch:
            with closing(batch.prepare_boundary_query(model,bindings[::-1],
                limits=ph.BoundaryDistanceLimits(),memory_limit_bytes=BUDGET)) as plan:
                for _ in range(3):
                    step=batch.step(H); record.setdefault("physics",[]).append(dict(result=step,state=batch.snapshot()))
                    assert all(step.values())
                    result=m1.query(batch,plan,record,.5)
                    m1.assert_packet(batch,plan,result,model,expected(batch.snapshot(),descriptions))
                    check_provenance(result,bindings)
                saved=batch.snapshot(); batch.reset([17])
                with pytest.raises(ValueError): plan.require_current(result)
                for w in (83,107,31): assert k0.canonical(batch.snapshot()[w])==k0.canonical(saved[w])
                batch.restore(saved)
                assert all(batch.step(H).values())
                result=m1.query(batch,plan,record,.5)
                m1.assert_packet(batch,plan,result,model,expected(batch.snapshot(),descriptions))


@pytest.mark.tyx
@pytest.mark.parametrize("floating",(False,True))
def test_real_mjcf_capsule_material(floating,tyx_mode,tmp_path):
    ref=reference(); path=ref.write_mjcf(tmp_path/"mjcf",floating=floating)
    if floating:
        from phymeo.builders.floating_mjcf import load_floating_mjcf
        asset=load_floating_mjcf(path,gravity=np.zeros(3))
        tree=ph.FloatingRobotTree(asset.model,FloatingRobotState(np.r_[ROOT,.01],
            np.r_[ref.rotation(ROOT[3:])@np.array((1/128,0,0)),np.zeros(3),.002]),
            kp=np.zeros(1),kd=np.zeros(1),effort_limit=np.ones(1),delay_steps=0)
        cls=ph.ResidentFloatingTreeBinding
        original=next(c.shape for c in asset.collisions if c.name=="round")
        round_shape=replace(original,key=ph.CollisionKey(7,202,1))
        surface=ph.CollisionShape(ph.CollisionKey(7,101,0),"mesh",(0,1,2,3),triangles=ref.FACES)
        static=(ref.VERTICES+(-.5,0,0))@ref.rotation(ROOT[3:]).T+ROOT[:3]
        bindings=(ph.ResidentStaticBinding(surface.key,ph.CollisionMotion(static)),
                  cls.from_asset(asset,original.key,key=round_shape.key,world=7))
        shapes=(surface,round_shape)
    else:
        asset=ph.load_mjcf(path,gravity=np.zeros(3))
        program=ph.DrivenRobotProgram(asset.model,kp=np.zeros(2),kd=np.zeros(2),effort_limit=np.ones(2),delay_steps=0)
        tree=ph.FixedRobotTree(program,program.initial_state(ph.RobotState([1/512,.004-1/512],[1/1024,-1/1024])))
        shapes,bindings=install(asset,"ordinary",7)
    model=ph.CollisionModel(tuple(shapes))
    with m1.saved(tmp_path,"mjcf-capsule-"+str(floating),tyx_mode,
                  inputs=dict(xml=path.read_text(),obj=(path.parent/"surface.obj").read_text())) as record:
        with ph.TyxWorldBatch([(7,tree)],tyx_mode,cache_root=tmp_path) as batch:
            with closing(batch.prepare_boundary_query(model,bindings,
                limits=ph.BoundaryDistanceLimits(),memory_limit_bytes=BUDGET)) as plan:
                for _ in range(3):
                    step=batch.step(H); record.setdefault("physics",[]).append(dict(result=step,state=batch.snapshot()))
                    assert all(step.values())
                    state=batch.snapshot()[7]
                    if floating:
                        q=np.asarray(state.position); angle=q[-1]
                        rz=np.array(((np.cos(angle),-np.sin(angle),0),(np.sin(angle),np.cos(angle),0),(0,0,1)))
                        truth={surface.key:static,round_shape.key:q[:3]+ref.CAPSULE@(ref.rotation(q[3:7])@rz).T}
                    else:
                        truth={shapes[0].key:ref.world_points(state,"ordinary",0,ref.VERTICES),
                               shapes[1].key:ref.world_points(state,"ordinary",1,ref.CAPSULE)}
                    result=m1.query(batch,plan,record,.5)
                    m1.assert_packet(batch,plan,result,model,truth)
                    check_provenance(result,[b for b in bindings if not isinstance(b,ph.ResidentStaticBinding)])


@pytest.mark.tyx
@pytest.mark.parametrize("layout",LAYOUTS)
def test_wrong_public_asset_model_rejected_before_allocation(layout,tyx_mode,tmp_path,monkeypatch):
    _,tree,_=prepare(tmp_path/"owned",layout)
    wrong,_,_=prepare(tmp_path/"wrong",layout,mass="1.125")
    shapes,bindings=install(wrong,layout,7)
    from phymeo.backends._resident_boundary_query import ResidentBoundaryQueryPlan
    with ph.TyxWorldBatch([(7,tree)],tyx_mode,cache_root=tmp_path) as batch:
        before=batch.snapshot()
        def denied(*args,**kwargs): raise AssertionError("admission must precede owned allocation")
        monkeypatch.setattr(ResidentBoundaryQueryPlan,"_allocate",denied)
        with pytest.raises(ValueError,match="model"):
            batch.prepare_boundary_query(ph.CollisionModel(tuple(shapes)),bindings,
                limits=ph.BoundaryDistanceLimits(),memory_limit_bytes=BUDGET)
        assert k0.canonical(batch.snapshot())==k0.canonical(before)


@pytest.mark.tyx
@pytest.mark.parametrize("layout",("affine","floating_affine"))
def test_wrong_original_graph_rejected(layout,tyx_mode,tmp_path):
    _,tree,_=prepare(tmp_path/"owned",layout)
    wrong,_,_=prepare(tmp_path/"wrong",layout,multiplier="-.5")
    shapes,bindings=install(wrong,layout,7)
    with ph.TyxWorldBatch([(7,tree)],tyx_mode,cache_root=tmp_path) as batch:
        with pytest.raises(ValueError,match="graph"):
            batch.prepare_boundary_query(ph.CollisionModel(tuple(shapes)),bindings,
                limits=ph.BoundaryDistanceLimits(),memory_limit_bytes=BUDGET)


@pytest.mark.tyx
@pytest.mark.parametrize("fault",("revision","radius","ids","winding","missing-face","none-direct","body-alias","source-key","hull"))
def test_asset_material_and_installation_rejections(fault,tyx_mode,tmp_path):
    asset,tree,_=prepare(tmp_path/"asset","ordinary")
    shapes,bindings=install(asset,"ordinary",7,body_alias=fault=="body-alias")
    with ph.TyxWorldBatch([(7,tree)],tyx_mode,cache_root=tmp_path) as batch:
        with pytest.raises(ValueError):
            if fault=="revision": shapes[0]=replace(shapes[0],revision=19)
            elif fault=="radius": shapes[1]=replace(shapes[1],radius=.25)
            elif fault=="ids":
                shapes[0]=replace(shapes[0],vertex_ids=(10,11,12,13),
                    triangles=tuple(tuple(i+10 for i in face) for face in shapes[0].triangles))
            elif fault=="winding":
                shapes[0]=replace(shapes[0],triangles=tuple(tuple(reversed(f)) for f in shapes[0].triangles))
            elif fault=="missing-face": shapes[0]=replace(shapes[0],triangles=shapes[0].triangles[:-1])
            elif fault=="none-direct":
                bindings[1]=ph.ResidentTreeBinding(bindings[1].key,7,1,bindings[1].local_points_m,topology=None)
            elif fault=="source-key":
                ph.ResidentTreeBinding.from_asset(asset,ph.CollisionKey(0,999,0),key=shapes[1].key,world=7)
            elif fault=="hull":
                shapes[0]=ph.CollisionShape(shapes[0].key,"hull",shapes[0].vertex_ids)
            batch.prepare_boundary_query(ph.CollisionModel(tuple(shapes)),bindings,
                limits=ph.BoundaryDistanceLimits(),memory_limit_bytes=BUDGET)


@pytest.mark.tyx
@pytest.mark.parametrize("fault",("capacity","memory","late-meta","late-publication"))
def test_asset_resource_failure_preserves_publication_and_retry(fault,tyx_mode,tmp_path,monkeypatch):
    asset,tree,data=prepare(tmp_path/"asset","ordinary")
    shapes,bindings=install(asset,"ordinary",7); model=ph.CollisionModel(tuple(shapes))
    # One complete tetra, one round. Every facet AABB overlaps this fixed band.
    formula=m1.reference().memory_layout(2,5,((4,1),),(4,))
    limits=ph.BoundaryDistanceLimits(groups=1,candidates=4,vertices=4,faces=4,segments=1)
    budget=formula["peak_required_bytes"]
    with ph.TyxWorldBatch([(7,tree)],tyx_mode,cache_root=tmp_path) as batch:
        with closing(batch.prepare_boundary_query(model,bindings,limits=limits,memory_limit_bytes=budget)) as plan:
            with m1.saved(tmp_path,"asset-fault-"+fault,tyx_mode,inputs=dict(data=data,formula=formula),plan=plan) as record:
                old=m1.query(batch,plan,record,.5)
                assert old.required_candidates==4
                assert plan.stats()["peak_owned_bytes"]==budget
                if fault in ("capacity","memory"):
                    opts=dict(limits=replace(limits,candidates=3) if fault=="capacity" else limits,
                              memory_limit_bytes=budget-1 if fault=="memory" else budget)
                    with closing(batch.prepare_boundary_query(model,bindings,**opts)) as smaller:
                        with pytest.raises((ph.BoundaryDistanceCapacityError,MemoryError)):
                            m1.query(batch,smaller,record,.5)
                        assert smaller.last_result is None
                else:
                    name="_download" if fault=="late-meta" else "_build_packet"
                    real=getattr(plan,name)
                    def failure(*args,**kwargs):
                        result=real(*args,**kwargs)
                        if fault=="late-publication" or args[1]=="group:0/meta":
                            raise RuntimeError("authored late asset failure")
                        return result
                    monkeypatch.setattr(plan,name,failure)
                    with pytest.raises(RuntimeError): m1.query(batch,plan,record,.5)
                    assert plan.last_result is old
                    monkeypatch.setattr(plan,name,real)
                result=m1.query(batch,plan,record,.5)
                m1.assert_packet(batch,plan,result,model,expected(batch.snapshot(),{7:data}))
                check_provenance(result,bindings)


@pytest.mark.parametrize("floating",(False,True))
def test_factory_mesh_does_not_open_old_convex_compiler(floating,tmp_path):
    layout="floating" if floating else "ordinary"
    asset,tree,_=prepare(tmp_path/"asset",layout)
    shapes,bindings=install(asset,layout,7)
    if floating:
        from phymeo.backends._resident_floating_tree import compile_floating_tree_world
        from phymeo.backends._resident_floating_binding import compile_floating_tree_binding
        spec=compile_floating_tree_world(tree)
        with pytest.raises(ValueError,match="convex"):
            compile_floating_tree_binding(shapes[0],bindings[0],spec)
    else:
        from phymeo.backends._resident_tree_world import compile_tree_world
        from phymeo.backends._resident_tree_binding import compile_tree_binding
        spec=compile_tree_world(tree)
        with pytest.raises(ValueError,match="convex"):
            compile_tree_binding(shapes[0],bindings[0],spec)
@pytest.mark.tyx
@pytest.mark.parametrize("filter_kind",("same-body","mask","excluded"))
def test_asset_original_filter_table(filter_kind,tyx_mode,tmp_path):
    asset,tree,data=prepare(tmp_path/"asset","ordinary")
    original=next(c for c in asset.collisions if c.name=="round")
    if filter_kind=="same-body":
        shape=replace(original.shape,key=ph.CollisionKey(0,1,1))
        changed=replace(original,shape=shape,link_index=0,link_name="left_jaw")
    elif filter_kind=="mask":
        shape=replace(original.shape,group=2,mask=2)
        changed=replace(original,shape=shape)
    else:
        changed=original
    if changed is not original:
        asset=replace(asset,collisions=tuple(changed if c is original else c for c in asset.collisions),
            collision_model=ph.CollisionModel(tuple(changed.shape if s.key==original.shape.key else s
                                                   for s in asset.collision_model.shapes)))
    shapes,bindings=install(asset,"ordinary",7,body_alias=filter_kind=="same-body")
    exclusions=(tuple(sorted(s.key for s in shapes)),) if filter_kind=="excluded" else ()
    model=ph.CollisionModel(tuple(shapes),exclusions)
    with m1.saved(tmp_path,"asset-filter-"+filter_kind,tyx_mode,
                  inputs=dict(data=data,model=model,bindings=bindings)) as record:
        with ph.TyxWorldBatch([(7,tree)],tyx_mode,cache_root=tmp_path) as batch:
            with closing(batch.prepare_boundary_query(model,bindings,
                limits=ph.BoundaryDistanceLimits(),memory_limit_bytes=BUDGET)) as plan:
                result=m1.query(batch,plan,record,.5)
                assert result.required_candidates==0 and not result.groups
                assert len(result.pairs)==1 and result.pairs[0].status=="filtered"
                assert result.pairs[0].reason=={"same-body":"same_rigid_body","mask":"mask","excluded":"excluded"}[filter_kind]
                check_provenance(result,bindings)