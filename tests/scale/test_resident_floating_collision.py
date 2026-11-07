"""C9 current accepted floating S/C/H bounds, not CCD or floating contact.

AUTHORING_UNEXECUTED. Independent material vertices use the time-jet reference
and actual public q; no HOST product FK/envelopes construct expected geometry.
"""

from dataclasses import replace
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pytest
from test_resident_floating_step import H, _close, _floating_storage, _json, _mixed_four, _packet, _physical_snapshot, _reference, _save, _tool_data, _tree

import phymeo as ph


def _box(low,high):
    return np.array(list(product(*zip(low,high,strict=True))),float)


def _record(result):
    return {name:_json(getattr(result,name)) for name in ("topology_generation","state_epoch","world_keys","ticks","shape_keys","shape_revisions","bounds","registered_pairs","candidates")}


def _row(result,key):
    return result.bounds[result.shape_keys.index(key)]


def _materials(n):
    return (
        (0,11,"sphere",np.array([[1/4,-1/8,0.]]),1/16),
        (min(1,n),12,"capsule",np.array([[-1/4,0.,0.],[1/4,0.,0.]]),1/32),
        (n,13,"hull",np.array([[-1/8,-1/16,0.],[-1/8,1/16,0.],[1/8,-1/16,0.],[1/8,1/16,0.],[0.,0.,1/8]]),0.),
    )


def _scene(batch,by_world):
    shapes,bindings,exclusions,sources=[],[],[],{}
    states=batch.snapshot()
    for world,data in by_world.items():
        dynamic=[]
        for group,tag,kind,local,radius in _materials(len(data["q"])-7):
            key=ph.CollisionKey(world,100+group,tag)
            shape=ph.CollisionShape(key,kind,tuple(101+17*i+tag for i in range(len(local))),radius,revision=7)
            shapes.append(shape); dynamic.append(key)
            bindings.append(ph.ResidentFloatingTreeBinding(key,world,group,local,topology=states[world].topology))
            sources[key]=(group,local,radius)
        exclusions.extend(combinations(dynamic,2))
        for tag,low,high in ((1,[-8]*3,[8]*3),(2,[30]*3,[31]*3)):
            key=ph.CollisionKey(world,900,tag)
            points=_box(low,high)+data["q"][:3]
            shapes.append(ph.CollisionShape(key,"hull",tuple(range(501,509)),revision=3))
            bindings.append(ph.ResidentStaticBinding(key,ph.CollisionMotion(points)))
            sources[key]=(None,points,0.)
    return ph.CollisionModel(tuple(reversed(shapes)),tuple(exclusions)),tuple(reversed(bindings)),sources


def _audit_bounds(result,states,by_world,sources):
    expected={}
    for key,(group,local,radius) in sources.items():
        if group is None:
            points=local
        else:
            state=states[key.world]
            geom=_reference().jets(by_world[key.world],np.asarray(state.position),np.asarray(state.velocity))
            points=geom["position"][group]+local@geom["rotation"][group].T
        low,high=points.min(0)-radius,points.max(0)+radius
        row=_row(result,key)
        assert np.isfinite(row).all()
        assert np.all(row[:3]<=low) and np.all(row[3:]>=high)
        expected[key]=dict(material_points=points,radius=radius,low=low,high=high)
    candidates=set(result.candidates)
    for pair in result.registered_pairs:
        assert pair[0].world==pair[1].world and pair[0].body!=pair[1].body
        a,b=(expected[key] for key in pair)
        overlaps=bool(np.all(a["low"]<=b["high"]) and np.all(b["low"]<=a["high"]))
        if overlaps:
            assert pair in candidates
        else:
            # These declared far controls have a metre-scale separation, not an
            # ambiguity close to the conservative FP32 padding boundary.
            assert pair not in candidates
    assert candidates <= set(result.registered_pairs)
    assert dict(zip(result.world_keys,result.ticks,strict=True))=={key:states[key].tick for key in by_world}
    return expected


@pytest.mark.tyx
@pytest.mark.parametrize("n,placement",((0,"base"),(1,"base"),(3,"cycle"),(9,"large"),(16,"base")))
def test_p07_all_finite_materials_cover_actual_root_and_child_rotation(n,placement,tyx_mode,tmp_path):
    data=_reference().case("P05",n=n,placement=placement)
    tree=_tree(data)
    rows=[]
    with ph.TyxWorldBatch([(503,tree)],tyx_mode,cache_root=tmp_path) as batch:
        batch.set_controls({503:_packet(tree,data)[0]})
        model,bindings,sources=_scene(batch,{503:data})
        with batch.prepare_collision(model,bindings) as plan:
            for _ in range(40):
                assert batch.step(H)=={503:True}
                state=batch.snapshot(); result=plan.query()
                expected=_audit_bounds(result,state,{503:data},sources)
                assert plan.require_current(result) is result
                rows.append(dict(state=state,geometry=batch.tree_kinematics(),result=_record(result),independent_material_bounds=expected,stats=plan.stats(),batch_stats=batch.stats()))
            assert plan.stats()["floating_bounds_kernel_launches"]>0
            assert plan.stats()["dynamic_envelope_uploaded_bytes"]==0
            assert plan.stats()["state_buffer_directions"]==(0,1)
            _save(tmp_path,"floating-current-bounds",dict(input=data,model=model,bindings=bindings,rows=rows,stats=plan.stats()))


@pytest.mark.tyx
@pytest.mark.parametrize("count",(1,4,32))
def test_p07_warm_bounds_consume_real_banks_without_host_snapshot_or_dynamic_upload(count,tyx_mode,tmp_path,monkeypatch):
    keys=[503,91,1207,6]+[2003+17*i for i in range(28)]
    by_world={key:_reference().case("P05",n=(0,1,9,16)[i%4]) for i,key in enumerate(keys[:count])}
    specs={key:_tree(data) for key,data in by_world.items()}
    with ph.TyxWorldBatch(list(specs.items()),tyx_mode,cache_root=tmp_path) as batch:
        batch.set_controls({key:_packet(specs[key],data)[0] for key,data in by_world.items()})
        model,bindings,sources=_scene(batch,by_world)
        with batch.prepare_collision(model,bindings) as plan:
            bootstrap=[]
            for _ in range(2):
                assert all(batch.step(H).values()); bootstrap.append(_record(plan.query()))
            entry=batch.snapshot(); batch_before,plan_before=batch.stats(),plan.stats()
            held=plan.last_result; held_copy=_record(held); rows=[]
            def forbidden(*args,**kwargs):
                raise AssertionError("floating warm path used HOST snapshots, FK, envelope upload or allocation")
            with monkeypatch.context() as patch:
                for name in ("snapshot","tree_kinematics","tree_diagnostics","kinematics"):
                    patch.setattr(batch,name,forbidden)
                patch.setattr(batch,"_buffer",forbidden)
                patch.setattr(plan._engine,"query",forbidden)
                from phymeo.dynamics import floating_robot
                from phymeo.solvers import floating_tree
                patch.setattr(floating_robot,"floating_kinematics",forbidden)
                patch.setattr(floating_tree,"floating_dynamics_terms",forbidden)
                if tyx_mode=="cuda":
                    import tyx as tx
                    patch.setattr(tx,"cuda_array",forbidden); patch.setattr(tx,"cuda_empty",forbidden)
                for step in range(3,43):
                    assert all(batch.step(H).values())
                    result=plan.query()
                    assert result.ticks.tolist()==[step]*count
                    rows.append(dict(result=_record(result),stats=plan.stats(),batch_stats=batch.stats()))
            final=batch.snapshot(); expected=_audit_bounds(result,final,by_world,sources)
            after,stats=plan.stats(),batch.stats()
            assert after["device_allocations"]==plan_before["device_allocations"]
            assert stats["device_allocations"]==batch_before["device_allocations"]
            assert after["floating_bounds_kernel_launches"]-plan_before["floating_bounds_kernel_launches"]==40
            assert after["dynamic_envelope_uploaded_bytes"]==0 and after["state_buffer_directions"]==(0,1)
            # Account only the warm region, before the explicit final observation.
            last=rows[-1]["batch_stats"]
            for field in ("snapshot_downloaded_bytes","kinematics_downloaded_bytes","tree_diagnostic_reads"):
                assert last[field]==batch_before[field]
            assert _record(held)==held_copy
            _save(tmp_path,"floating-warm-bounds",dict(input=by_world,model=model,bindings=bindings,entry=entry,bootstrap=bootstrap,before=batch_before,plan_before=plan_before,rows=rows,final=final,geometry=batch.tree_kinematics(),independent_end_bounds=expected,stats=stats))


@pytest.mark.tyx
def test_p07_folded_asset_root_and_tool_material_is_transformed_once(tyx_mode,tmp_path):
    asset=ph.load_floating_urdf(Path(__file__).resolve().parents[1]/"fixtures/urdf/floating_fixed_composite.urdf",gravity=(0,0,0),collapse_fixed_joints=True)
    data=_tool_data(); tree=_tree(data,model=asset.model)
    with ph.TyxWorldBatch([(503,tree)],tyx_mode,cache_root=tmp_path) as batch:
        batch.set_controls({503:_packet(tree,data)[0]})
        shapes,bindings,sources=[],[],{}
        for col in asset.collisions:
            key=ph.CollisionKey(503,101,col.shape.key.shape)
            shape=replace(col.shape,key=key)
            binding=ph.ResidentFloatingTreeBinding.from_asset(asset,col.shape.key,key=key,world=503)
            assert binding.group_index==0
            # Independently authored mount, not product's transformed binding.
            local=np.asarray(col.local_vertices)+(np.array([1/4,0,0]) if col.link_name=="tool" else 0)
            _close(binding.local_points_m,local,atol=1e-12,rtol=0)
            shapes.append(shape);bindings.append(binding);sources[key]=(0,local,shape.radius)
        model=ph.CollisionModel(tuple(shapes))
        with batch.prepare_collision(model,tuple(bindings)) as plan:
            rows=[]
            for _ in range(40):
                assert batch.step(H)=={503:True}
                state=batch.snapshot(); result=plan.query()
                expected=_audit_bounds(result,state,{503:data},sources)
                assert result.candidates==()
                rows.append(dict(state=state,result=_record(result),independent_material_bounds=expected))
            _save(tmp_path,"floating-folded-source",dict(asset=asset,model=model,bindings=bindings,input=data,rows=rows,stats=plan.stats()))


@pytest.mark.tyx
@pytest.mark.parametrize("filter_kind",("none","same_body","mask","excluded","parent_mask","parent_excluded"))
def test_p07_same_full_tree_branch_domain_filters_before_adjacency(filter_kind,tyx_mode,tmp_path):
    data=_reference().case("P05",n=3)
    tree=_tree(data)
    with ph.TyxWorldBatch([(503,tree)],tyx_mode,cache_root=tmp_path) as batch:
        groups=(1,2) if not filter_kind.startswith("parent_") else (0,1)
        key1,key2=ph.CollisionKey(503,101,1),ph.CollisionKey(503,102,2)
        if filter_kind=="same_body": key2=ph.CollisionKey(503,101,2);groups=(1,1)
        shapes=[ph.CollisionShape(key1,"sphere",(71,),radius=1/4),ph.CollisionShape(key2,"sphere",(81,),radius=1/4)]
        if filter_kind.endswith("mask"): shapes[1]=replace(shapes[1],mask=0)
        pair=tuple(sorted((key1,key2)))
        excluded=(pair,) if filter_kind.endswith("excluded") else ()
        model=ph.CollisionModel(tuple(shapes),excluded)
        bindings=tuple(ph.ResidentFloatingTreeBinding(key,503,g,np.zeros((1,3)),topology=batch.snapshot()[503].topology) for key,g in zip((key1,key2),groups,strict=True))
        if filter_kind=="none":
            with pytest.raises(ValueError): batch.prepare_collision(model,bindings)
        with batch.prepare_collision(model,bindings,tree_dynamic_pairs=True) as plan:
            result=plan.query()
            if filter_kind=="none": assert result.candidates==(pair,)
            else: assert result.candidates==()
            assert batch.snapshot()[503].tick==0
            _save(tmp_path,"floating-branch-domain",dict(filter_kind=filter_kind,input=data,model=model,bindings=bindings,result=_record(result),stats=plan.stats()))


@pytest.mark.tyx
@pytest.mark.parametrize("problem",("parent","group","world","body_alias","two_bodies_one_group","foreign_topology","source_radius","source_vertices","source_revision","source_model"))
def test_p07_mismatched_owner_material_source_and_parent_reject_before_allocation(problem,tyx_mode,tmp_path):
    data=_reference().case("P03");tree=_tree(data)
    with ph.TyxWorldBatch([(503,tree)],tyx_mode,cache_root=tmp_path) as batch:
        key1,key2=ph.CollisionKey(503,101,1),ph.CollisionKey(503,102,2)
        keys=(key1,key2);shape1=ph.CollisionShape(key1,"sphere",(71,),radius=1/8)
        shape2=ph.CollisionShape(key2,"sphere",(81,),radius=1/8)
        shapes=[shape1,shape2]; groups=[0,1]; world=503; top=batch.snapshot()[503].topology
        if problem=="group": groups[1]=2
        elif problem=="world": world=91
        elif problem=="body_alias": shapes[1]=replace(shape2,key=ph.CollisionKey(503,101,2)); keys=(key1,shapes[1].key)
        elif problem=="two_bodies_one_group": groups=[0,0]
        elif problem=="foreign_topology":
            with ph.TyxWorldBatch([(503,_tree(_reference().case("P04")))],tyx_mode,cache_root=tmp_path/"foreign") as foreign:
                top=foreign.snapshot()[503].topology
        before=batch.snapshot(); allocations=batch.stats()["device_allocations"]
        with pytest.raises((ValueError,TypeError)):
            if problem.startswith("source_"):
                asset=ph.load_floating_urdf(Path(__file__).resolve().parents[1]/"fixtures/urdf/floating_fixed_composite.urdf",gravity=(0,0,0),collapse_fixed_joints=True)
                col=next(c for c in asset.collisions if c.shape.kind=="sphere")
                bound=ph.ResidentFloatingTreeBinding.from_asset(asset,col.shape.key,key=key1,world=503)
                shape=replace(col.shape,key=key1)
                if problem=="source_radius": shape=replace(shape,radius=shape.radius*2)
                elif problem=="source_vertices": shape=replace(shape,vertex_ids=(999,))
                elif problem=="source_revision": shape=replace(shape,revision=shape.revision+1)
                # Even unchanged source_model is a different positive-mass tree.
                batch.prepare_collision(ph.CollisionModel((shape,)),(bound,))
            else:
                bindings=tuple(ph.ResidentFloatingTreeBinding(k,world,g,np.zeros((1,3)),topology=top) for k,g in zip(keys,groups,strict=True))
                batch.prepare_collision(ph.CollisionModel(tuple(shapes)),bindings,tree_dynamic_pairs=True)
        assert batch.snapshot()==before and batch.stats()["device_allocations"]==allocations


@pytest.mark.tyx
def test_p07_floating_contacts_stay_explicitly_unsupported(tyx_mode,tmp_path):
    data=_reference().case("P03");tree=_tree(data)
    with ph.TyxWorldBatch([(503,tree)],tyx_mode,cache_root=tmp_path) as batch:
        model,bindings,_=_scene(batch,{503:data})
        before=batch.snapshot()
        with pytest.raises((ValueError,TypeError),match="floating|unsupported|ordinary|affine|contact"):
            batch.prepare_contacts(model,bindings,{})
        assert batch.snapshot()==before


@pytest.mark.tyx
def test_p07_old_fixed_and_affine_contacts_advance_unbound_floating_world(tyx_mode,tmp_path):
    from test_resident_affine_contact_transactions import _ordinary_geometry
    from test_resident_affine_contacts import _a01_geometry
    from test_resident_affine_step import _ref as affine_reference,_load,_world,_action

    data=_reference().case("P03"); floating=_tree(data)
    from test_resident_affine_contact_transactions import _ordinary
    ordinary,ordinary_control=_ordinary()
    old=affine_reference().case("A01");old["motor"][:]=0
    asset=_load(tmp_path/"asset",old);affine=_world(asset,old)
    controls={503:_packet(floating,data)[0],91:ordinary_control,1207:_action(affine,asset,old),6:(0,0,0)}
    with ph.TyxWorldBatch([(503,floating),(91,ordinary),(1207,affine),(6,ph.FreeSphere(2,1/8,position=(0,0,2),gravity=(0,0,-8)))],tyx_mode,cache_root=tmp_path) as batch:
        batch.set_controls(controls)
        model1,bound1,_,pair1=_ordinary_geometry(batch,91)
        model2,bound2,pair2,_,_=_a01_geometry(batch,asset,world=1207)
        model=ph.CollisionModel(model1.shapes+model2.shapes)
        laws={pair1:ph.RobotContactMaterial(64,0,0),pair2:ph.RobotContactMaterial(64,0,0)}
        with batch.prepare_contacts(model,bound1+bound2,laws) as plan:
            rows=[]
            for step in range(2):
                entry=batch.snapshot(); epoch=batch.stats()["state_epoch"]
                assert batch.step_contacts(plan,H)==dict.fromkeys(controls,True)
                state=batch.snapshot(); receipts=plan.last_accepted
                assert set(receipts)=={91,1207} and batch.stats()["state_epoch"]==epoch+1
                _close(state[503].velocity,H*(step+1)*np.array([-1.5,0,0,0,0,0,4.5]))
                if step==0:
                    _close(receipts[91].contact_generalized,[2.])
                    _close(receipts[1207].contact_generalized,[0.,3.])
                    _close(receipts[1207].diagnostics["affine_diagnostic"].acceleration,[1.])
                rows.append(dict(entry=entry,state=state,receipts=receipts,geometry=batch.tree_kinematics(),diagnostics=batch.tree_diagnostics(),energy=batch.summary(),stats=batch.stats()))
            _save(tmp_path,"floating-unbound-contact-mixed",dict(input=data,affine_input=old,controls=controls,model=model,bindings=bound1+bound2,laws=laws,rows=rows,stats=plan.stats()))