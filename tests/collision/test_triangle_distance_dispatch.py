"""COL3 predeclared public snapshot-dispatch matrix; no narrow-phase oracle reuse."""

import gzip
import json
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from collections.abc import Mapping

import numpy as np
import pytest

from phymeo.constraints.collision_shapes import CollisionKey, CollisionModel, CollisionMotion, CollisionShape, RigidTurn
from phymeo.geometry.collision_pipeline import CollisionWorkspace
from phymeo.geometry._triangle_distance_dispatch import BoundaryDistanceLimits, BoundaryDistanceCapacityError


@pytest.fixture(params=("interpret", "cuda"))
def mode(request):
    return request.param


def encode(value):
    if isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=list(value.shape), data=value.tobytes().hex())
    if is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@contextmanager
def saved(tmp_path, name, ws, motion):
    record = dict(identity=name, motion=motion, complete=False)
    try:
        yield record
        record["complete"] = True
    except BaseException as error:
        record["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        record["last"] = ws.last_boundary_distances
        record["workspace_stats"] = ws.stats()
        with gzip.open(tmp_path/(name+".json.gz"), "wt", encoding="utf-8") as stream:
            json.dump(encode(record), stream, allow_nan=False)


def flat(kind="sphere", case="front", world=7, translation=(0.,0.,0.)):
    key = CollisionKey(world, 101, 0)
    round_key = CollisionKey(world, 202, 0)
    triangle = np.array(((0,0,0),(2,0,0),(0,2,0)), np.float64)
    start = dict(front=(.5,.5,.25), back=(.5,.5,-.25), edge=(1,-.25,.25),
                 vertex=(-.25,-.25,.25), outside=(4,4,.25), through=(.5,.5,0))[case]
    axis = np.array((start,), np.float64)
    if kind == "capsule":
        axis = np.vstack((axis, axis))
        if case == "through":
            axis[:,2] = (-.25,.25)
        else:
            axis[1,2] += -.25 if case == "back" else .25
    shapes = (CollisionShape(key,"triangle",(11,37,91)),
              CollisionShape(round_key,kind,(11,) if kind=="sphere" else (11,37),radius=.125))
    shift = np.asarray(translation)
    return CollisionModel(shapes), {key:CollisionMotion(triangle+shift),round_key:CollisionMotion(axis+shift)}


def assert_group(group):
    packet = group.packet
    assert len(packet.rows) == len(group.candidates)
    np.testing.assert_array_equal(packet.inputs["candidates"], group.candidates)
    assert not group.bvh_raw.flags.writeable
    assert not group.leaf_bounds.flags.writeable
    assert not group.query_bounds.flags.writeable
    assert not group.candidates.flags.writeable
    for row in packet.rows:
        assert row.facet_id < len(group.facet_vertices)
        assert row.vertex_ids == group.facet_vertices[row.facet_id]
        assert group.round_keys[row.segment_id].world == group.surface.world
        assert row.status in ("RESOLVED","CONDITIONING","DEGENERATE")
    words = group.bvh_raw[:,:-2]
    actual = [(s,f) for s in range(len(group.round_keys))
              for f in range(len(group.facet_vertices)) if int(words[s,f//30]) & (1 << (f%30))]
    assert actual == list(map(tuple, group.candidates))


@pytest.mark.parametrize("kind",("sphere","capsule"))
@pytest.mark.parametrize("case",("front","back","edge","vertex","outside","through"))
def test_public_finite_features(mode,kind,case,tmp_path):
    model,motion = flat(kind,case)
    margin = .01 if case=="outside" else .5
    with CollisionWorkspace(model,backend=mode) as ws:
        with saved(tmp_path,kind+"-"+case,ws,motion) as record:
            result = ws.boundary_distances(motion,margin=margin)
            record["result"] = result
            assert ws.last_result is None and ws.last_boundary_distances is result
            ws.require_boundary_distances(result,motion,margin=margin)
            assert result.semantics == "threshold-limited-boundary-distance"
            assert len(result.groups)==1
            group=result.groups[0]
            assert_group(group)
            if case=="outside":
                assert result.required_candidates==0 and not group.packet.rows
                assert result.pairs[0].status=="no_boundary_candidate"
            else:
                assert result.required_candidates==1
                row=group.packet.rows[0]
                assert row.status=="RESOLVED"
                expected = dict(front=.25,back=.25,edge=np.sqrt(.125),vertex=np.sqrt(.1875),through=0.)[case]
                assert abs(row.distance-expected)<=1e-4*np.sqrt(8)
                assert abs(row.gap-(expected-.125))<=1e-4*np.sqrt(8)
                if case=="through":
                    assert row.normal is None
                else:
                    q=np.array(dict(front=(.5,.5,0),back=(.5,.5,0),edge=(1,0,0),vertex=(0,0,0))[case])
                    world=np.asarray(row.triangle_point_relative_m)+group.packet.inputs["origin_m"]
                    np.testing.assert_allclose(world,q,atol=2e-4*np.sqrt(8),rtol=0)
                    axis=np.asarray(row.axis_point_relative_m)+group.packet.inputs["origin_m"]
                    np.testing.assert_allclose(row.normal,(q-axis)/expected,atol=2e-3,rtol=0)


def mesh_scene(concave=False, center=(1,1,.25)):
    if concave:
        outline=np.array(((0,0),(2,0),(2,1),(1,1),(1,2),(0,2)),np.float64)
        n=6
        points=np.array([(x,y,z) for z in (0.,1.) for x,y in outline],np.float64)
        caps=((0,1,3),(1,2,3),(0,3,5),(3,4,5))
        faces=list(caps)+[tuple(v+n for v in face) for face in caps]
    else:
        n=4
        points=np.array(((0,0,0),(2,0,0),(2,2,0),(0,2,0),
                         (0,0,2),(2,0,2),(2,2,2),(0,2,2)),np.float64)
        faces=[(0,1,2),(0,2,3),(4,6,5),(4,7,6)]
    for i in range(n):
        j=(i+1)%n
        faces.extend(((i,j,j+n),(i,j+n,i+n)))
    ids=tuple(11+13*i for i in range(len(points)))
    mesh=CollisionShape(CollisionKey(7,101,0),"mesh",ids,triangles=tuple(tuple(ids[i] for i in face) for face in faces))
    sphere=CollisionShape(CollisionKey(7,202,0),"sphere",(11,),radius=.125)
    return CollisionModel((mesh,sphere)),{mesh.key:CollisionMotion(points),sphere.key:CollisionMotion((center,))}


@pytest.mark.parametrize("case",("shared_pruned","closed_inside","concave"))
def test_mesh_original_facets_and_boundary_semantics(mode,case,tmp_path):
    model,motion=mesh_scene(case=="concave",(1.25,1.25,.5) if case=="concave" else ((1,1,1) if case=="closed_inside" else (1,1,.25)))
    margin=1. if case=="closed_inside" else .2
    with CollisionWorkspace(model,backend=mode) as ws:
        with saved(tmp_path,case,ws,motion) as record:
            result=ws.boundary_distances(motion,margin=margin)
            record["result"]=result
            group=result.groups[0]
            assert_group(group)
            distances=[r.distance for r in group.packet.rows]
            assert all(r.status=="RESOLVED" for r in group.packet.rows)
            assert abs(min(distances)-(1. if case=="closed_inside" else .25))<3e-4
            if case=="shared_pruned":
                assert len(distances)==2 and len(distances)<len(group.facet_vertices)
                assert all(abs(d-.25)<3e-4 for d in distances)
            elif case=="concave":
                assert len(distances)==4 and len(distances)<len(group.facet_vertices)
            else:
                assert len(distances)==12
                empty=ws.boundary_distances(motion,margin=.01)
                record["empty_inside_band"]=empty
                assert empty.required_candidates==0
                assert empty.pairs[0].status=="no_boundary_candidate"
                assert empty.semantics=="threshold-limited-boundary-distance"
            assert ws.stats()["mesh_bvh_refits"]>=1


def two_worlds():
    a,x=flat(world=7)
    b,y=flat(world=4096,translation=(2**20,-2**20,1.))
    return CollisionModel(a.shapes+b.shapes),{**x,**y}


def test_sparse_world_binding_storage_and_changed_snapshot(mode,tmp_path):
    model,motion=two_worlds()
    with CollisionWorkspace(model,backend=mode) as ws:
        with saved(tmp_path,"source",ws,motion) as record:
            first=ws.boundary_distances(motion,margin=.5)
            record["first"]=first
            assert len(first.groups)==2 and first.required_candidates==2
            assert sum(p.status=="filtered" and p.reason=="different_world" for p in first.pairs)==4
            changed=dict(motion)
            key=CollisionKey(4096,101,0)
            points=motion[key].start.copy(); points[2,2]+=.125
            changed[key]=CollisionMotion(points)
            with pytest.raises(ValueError,match="binding"):
                ws.require_boundary_distances(first,changed,margin=.5)
            second=ws.boundary_distances(changed,margin=.5)
            record["second"]=second
            assert second.source_binding!=first.source_binding
            with pytest.raises(ValueError,match="current"):
                ws.require_boundary_distances(first,motion,margin=.5)
    reverse=[]; reordered={}
    for shape in model.shapes:
        permutation=tuple(reversed(range(len(shape.vertex_ids))))
        reverse.append(CollisionShape(shape.key,shape.kind,tuple(shape.vertex_ids[i] for i in permutation),radius=shape.radius))
        reordered[shape.key]=CollisionMotion(motion[shape.key].start[list(permutation)])
    with CollisionWorkspace(CollisionModel(tuple(reversed(reverse))),backend=mode) as ws:
        perm=ws.boundary_distances(reordered,margin=.5)
        np.testing.assert_allclose([g.packet.rows[0].distance for g in perm.groups],
                                   [g.packet.rows[0].distance for g in first.groups], rtol=0, atol=3e-4)
        with saved(tmp_path,"storage-permutation",ws,reordered) as record:
            record["result"]=perm


def test_filter_codes_and_same_family_are_preserved(mode,tmp_path):
    triangle=CollisionShape(CollisionKey(7,10,0),"triangle",(1,2,3))
    rows=[triangle,
          CollisionShape(CollisionKey(7,20,0),"sphere",(1,),radius=.125),
          CollisionShape(CollisionKey(7,20,1),"sphere",(1,),radius=.125),
          CollisionShape(CollisionKey(7,30,0),"sphere",(1,),radius=.125,mask=0),
          CollisionShape(CollisionKey(7,40,0),"capsule",(1,2),radius=.125)]
    model=CollisionModel(tuple(rows),excluded=((triangle.key,rows[1].key),))
    motion={s.key:CollisionMotion(((0,0,0),(2,0,0),(0,2,0)) if s.kind=="triangle" else (((.5,.5,.25),(.5,.5,.5)) if s.kind=="capsule" else ((.5,.5,.25),))) for s in rows}
    with CollisionWorkspace(model,backend=mode) as ws:
        with saved(tmp_path,"filters",ws,motion) as record:
            result=ws.boundary_distances(motion,margin=.5); record["result"]=result
            assert {p.reason for p in result.pairs}>={"excluded","same_rigid_body","mask","same_family"}
            assert len(result.pairs)==10 and result.required_candidates==2


@pytest.mark.parametrize("resource",("groups","candidates","vertices","faces","segments"))
def test_exact_and_one_less_capacity(mode,resource,tmp_path):
    model,motion=two_worlds()
    exact=dict(groups=2,candidates=2,vertices=3,faces=1,segments=1)
    with CollisionWorkspace(model,backend=mode) as ws:
        with saved(tmp_path,"capacity-"+resource,ws,motion) as record:
            old=ws.boundary_distances(motion,margin=.5,limits=BoundaryDistanceLimits(**exact))
            record["old"]=old
            lower=dict(exact); lower[resource]-=1
            with pytest.raises(BoundaryDistanceCapacityError) as error:
                ws.boundary_distances(motion,margin=.5,limits=BoundaryDistanceLimits(**lower))
            assert error.value.resource==resource and error.value.required==exact[resource]
            assert error.value.limit==lower[resource] and ws.last_boundary_distances is old
            record["capacity_error"]=dict(resource=resource,required=error.value.required,limit=error.value.limit)


def test_last_group_failure_and_retry(mode,monkeypatch,tmp_path):
    from phymeo.backends._device_triangle_distance import DeviceTriangleDistanceBatch
    model,motion=two_worlds()
    with CollisionWorkspace(model,backend=mode) as ws:
        with saved(tmp_path,"late-read",ws,motion) as record:
            old=ws.boundary_distances(motion,margin=.5); record["old"]=old
            actual=DeviceTriangleDistanceBatch._download
            calls=[]
            def late(batch,array,label):
                value=actual(batch,array,label)
                calls.append((label,value.copy()))
                if len(calls)==7:
                    raise RuntimeError("authored final group read failure")
                return value
            with monkeypatch.context() as patch:
                patch.setattr(DeviceTriangleDistanceBatch,"_download",late)
                with pytest.raises(RuntimeError,match="final group"):
                    ws.boundary_distances(motion,margin=.5)
            assert ws.last_boundary_distances is old
            record["attempted_reads"]=calls
            fresh=ws.boundary_distances(motion,margin=.5)
            record["retry"]=fresh
            assert fresh is ws.last_boundary_distances and fresh.source_binding==old.source_binding


def test_outer_transaction_commits_only_final_or_retains_old(mode,tmp_path):
    model,motion=flat()
    with CollisionWorkspace(model,backend=mode) as ws:
        with saved(tmp_path,"transaction",ws,motion) as record:
            old=ws.boundary_distances(motion,margin=.5)
            with pytest.raises(RuntimeError,match="outer abort"):
                with ws.transaction():
                    first=ws.boundary_distances(motion,margin=.5)
                    second=ws.boundary_distances(motion,margin=.5)
                    assert second.sequence==first.sequence+1 and ws.last_boundary_distances is old
                    with pytest.raises(ValueError,match="current"):
                        ws.require_boundary_distances(second,motion,margin=.5)
                    raise RuntimeError("outer abort")
            assert ws.last_boundary_distances is old
            with ws.transaction():
                first=ws.boundary_distances(motion,margin=.5)
                final=ws.boundary_distances(motion,margin=.5)
            assert first.sequence==old.sequence+1 and final.sequence==first.sequence+1
            assert ws.last_boundary_distances is final
            record.update(old=old,aborted=second,committed=final)


@pytest.mark.parametrize("bad",("end","turn","skin","shape","margin","keys"))
def test_explicit_domain_rejection(mode,bad):
    model,motion=flat()
    if bad=="skin":
        shape=model.shapes[0]
        model=CollisionModel((CollisionShape(shape.key,"triangle",shape.vertex_ids,radius=.01),model.shapes[1]))
    if bad=="shape":
        shape=model.shapes[1]
        model=CollisionModel((model.shapes[0],CollisionShape(shape.key,"particle",shape.vertex_ids)))
    if bad in ("end","turn"):
        key=model.shapes[1].key; points=motion[key].start
        motion[key]=CollisionMotion(points,end=points+1) if bad=="end" else CollisionMotion(points,turn=RigidTurn((0,0,0),(0,0,0),(0,0,1),.1))
    if bad=="keys":
        motion.pop(next(iter(motion)))
    with CollisionWorkspace(model,backend=mode) as ws:
        with pytest.raises((TypeError,ValueError)):
            ws.boundary_distances(motion,margin=-1. if bad=="margin" else .5)
        assert ws.last_boundary_distances is None and ws.last_result is None


def test_limits_reject_boolean_expansion_and_untyped_mapping():
    for field,value in (("groups",True),("candidates",8193),("vertices",513),("faces",129),("segments",65)):
        with pytest.raises((TypeError,ValueError)):
            BoundaryDistanceLimits(**{field:value})

@pytest.mark.parametrize("status",("CONDITIONING","DEGENERATE"))
def test_unresolved_rows_are_complete_public_boundary_evidence(mode,status,tmp_path):
    model,motion=flat("capsule")
    triangle,round_key=(s.key for s in model.shapes)
    if status=="CONDITIONING":
        motion[round_key]=CollisionMotion(((.25,.25,.25),(1.25,.25+2**-15,.25)))
    else:
        motion[triangle]=CollisionMotion(((0,0,0),(2,0,0),(1,0,0)))
    with CollisionWorkspace(model,backend=mode) as ws:
        with saved(tmp_path,status,ws,motion) as record:
            result=ws.boundary_distances(motion,margin=.5)
            record["result"]=result
            assert result.required_candidates==1
            assert result.groups[0].packet.rows[0].status==status
            assert result.groups[0].packet.rows[0].normal is None
            assert_group(result.groups[0])

def test_owner_and_margin_binding(mode,tmp_path):
    model,motion=flat()
    with CollisionWorkspace(model,backend=mode) as first, CollisionWorkspace(model,backend=mode) as second:
        with saved(tmp_path,"owner",first,motion) as record:
            packet=first.boundary_distances(motion,margin=.5)
            other=second.boundary_distances(motion,margin=.5)
            record.update(packet=packet,other=other)
            with pytest.raises(ValueError,match="owner"):
                second.require_boundary_distances(packet,motion,margin=.5)
            with pytest.raises(ValueError,match="binding"):
                first.require_boundary_distances(packet,motion,margin=.25)
            with pytest.raises(TypeError,match="Limits"):
                first.boundary_distances(motion,margin=.5,limits={})
            assert first.last_boundary_distances is packet