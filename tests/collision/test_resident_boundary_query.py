"""M1 tests first: actual accepted four-tree source, finite boundary, one publication.

The independent oracle consumes accepted q/v and authored material geometry;
it never consumes product FK/packed coordinates to define the expected answer.
All tests are initially UNEXECUTED. Parent owns the only project execution slot.
"""
import gzip
import importlib.util
import json
import sys
from collections.abc import Mapping
from contextlib import closing, contextmanager
from dataclasses import fields, is_dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import test_device_convex_family_sources as fsrc
import test_device_convex_query as k0

import phymeo as ph

H = 1 / 1024
BUDGET = 32 * 1024 * 1024
LAYOUTS = ("ordinary", "affine", "floating", "floating_affine")
WORLDS = (83, 17, 107, 31)


@lru_cache(maxsize=1)
def reference():
    path = Path(__file__).parents[1] / "reference_newton/rr/resident_boundary_source_reference.py"
    spec = importlib.util.spec_from_file_location("m1_independent_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def immutable(value):
    if isinstance(value, np.ndarray):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
    elif is_dataclass(value):
        for field in fields(value):
            immutable(getattr(value,field.name))
    elif isinstance(value, Mapping):
        for child in value.values():
            immutable(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            immutable(child)


@contextmanager
def saved(path, identity, mode, *, inputs=None, plan=None):
    record = dict(schema="M1-source-evidence-v1", identity=identity, mode=mode,
                  inputs=inputs, attempts=[], complete=False)
    try:
        yield record
        record["complete"] = True
    except BaseException as error:
        record["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        if plan is not None:
            record.update(last=plan.last_result, attempt=plan.last_attempt, stats=plan.stats())
        with gzip.open(path / (identity + ".json.gz"), "wt", encoding="utf-8") as stream:
            json.dump(k0.encode(record), stream, sort_keys=True, allow_nan=False)


def query(batch, plan, record, margin):
    before = batch.snapshot()
    row = dict(before=before, margin=margin, before_stats=plan.stats())
    record["attempts"].append(row)
    try:
        value = plan.query(margin=margin)
        row["result"] = value
        return value
    finally:
        row.update(after=batch.snapshot(), last_attempt=plan.last_attempt, stats=plan.stats())
        assert k0.canonical(before) == k0.canonical(row["after"])


def binding(tree, checkpoint, layout, key, group, points):
    cls = (ph.ResidentTreeBinding, ph.ResidentAffineTreeBinding,
           ph.ResidentFloatingTreeBinding, ph.ResidentFloatingAffineTreeBinding)[LAYOUTS.index(layout)]
    kwargs = dict(topology=checkpoint.topology)
    if layout == "floating_affine":
        kwargs["graph_token"] = tree.inputs(np.zeros((1, 3))).graph_token
    return cls(key, key.world, group, np.asarray(points, np.float64), **kwargs)


def moving_scene(batch, trees, layouts, *, static_surface=False, order=False):
    """F0 original driver/follower; body202 owns both round shapes."""
    ref = reference()
    states = batch.snapshot()
    shapes, bindings, data = [], [], {}
    for world in trees:
        tree, layout, state = trees[world], layouts[world], states[world]
        triangle_key = ph.CollisionKey(world, 101, 0)
        shapes.append(ph.CollisionShape(triangle_key, "triangle", ref.IDS, revision=3))
        first = ref.world_points(state, layout, "driver", ref.TRIANGLE)
        _, _, driver = ref.pose(state, layout, "driver")
        _, _, follower = ref.pose(state, layout, "follower")
        bindings.append(ph.ResidentStaticBinding(triangle_key, ph.CollisionMotion(first))
                        if static_surface else
                        binding(tree, state, layout, triangle_key, driver, ref.TRIANGLE))
        data[world] = dict(layout=layout, static_triangle=first if static_surface else None)
        for ordinal, kind in enumerate(("sphere", "capsule")):
            key = ph.CollisionKey(world, 202, ordinal)
            ids = (11,) if kind == "sphere" else (11, 37)
            shapes.append(ph.CollisionShape(key, kind, ids, radius=ref.RADIUS, revision=5))
            bindings.append(binding(tree, state, layout, key, follower, ref.local_round(kind)))
    if order:
        shapes.reverse()
        bindings.reverse()
    return ph.CollisionModel(tuple(shapes)), tuple(bindings), data


def expected_scene(states, declarations):
    out = {}
    for world, data in declarations.items():
        tri, rounds = reference().expected_moving(
            states[world], data["layout"], data["static_triangle"])
        out[ph.CollisionKey(world, 101, 0)] = tri
        for ordinal, kind in enumerate(("sphere", "capsule")):
            out[ph.CollisionKey(world, 202, ordinal)] = rounds[kind]
    return out


def static_scene(batch, tree, *, world=7, triangle=None, round_points=None,
                 mesh_faces=None, vertex_ids=None, kind="sphere", translation=(0, 0, 0)):
    """Static surface and a real ordinary follower installed at specified initial pose."""
    ref = reference()
    state = batch.snapshot()[world]
    shift = np.asarray(translation, np.float64)
    triangle = np.asarray(triangle, np.float64) + shift
    round_points = np.asarray(round_points, np.float64) + shift
    ids = tuple(ref.IDS if vertex_ids is None else vertex_ids)
    surface = ph.CollisionShape(ph.CollisionKey(world, 101, 0),
        "triangle" if mesh_faces is None else "mesh", ids, revision=3,
        **({} if mesh_faces is None else
           dict(triangles=tuple(tuple(ids[int(i)] for i in face) for face in mesh_faces))))
    round_shape = ph.CollisionShape(ph.CollisionKey(world, 202, 0), kind,
                                   (11,) if kind=="sphere" else (11,37),
                                   radius=ref.RADIUS, revision=5)
    position, rotation, group = ref.pose(state, "ordinary", "follower")
    local = (round_points-position) @ rotation
    model = ph.CollisionModel((surface, round_shape))
    bindings = (ph.ResidentStaticBinding(surface.key, ph.CollisionMotion(triangle)),
                binding(tree, state, "ordinary", round_shape.key, group, local))
    return model, bindings, {surface.key: triangle, round_shape.key: round_points}


def assert_packet(batch, plan, result, model, expected, *, statuses=None):
    ref = reference()
    assert plan.last_result is result and plan.require_current(result) is result
    assert result.schema == "phymeo-resident-boundary-distances-v1"
    assert result.source_kind == "resident-rigid-device-v1"
    assert result.semantics == "threshold-limited-boundary-distance"
    assert result.state_epoch == batch._state_epoch
    assert result.active == batch._active
    states = batch.snapshot()
    np.testing.assert_array_equal(result.ticks, [states[w].tick for w in result.world_keys])
    assert set(result.source) == {
        "registration", "model", "bindings", "shape_keys", "shape_revisions", "point_ids",
        "facet_ids", "facet_vertices", "world_layouts", "shape_rows", "shape_groups",
        "static_flags", "world_anchors", "authored_constants", "uploaded_constants",
        "conversion_deltas", "gathered_points", "gathered_axes", "stamps"}
    immutable(result.source)
    immutable(result.ticks)
    by_key = {shape.key: shape for shape in model.shapes}
    assert tuple(result.source["shape_keys"]) == tuple(s.key for s in model.shapes)
    for key, points in expected.items():
        anchor = np.asarray(result.source["world_anchors"][key.world])
        actual = result.source["gathered_points"][key]
        # Source gate uses this shape's local material extent, never translation.
        local = points-anchor
        length = max(1., float(np.max(np.linalg.norm(local-local[0], axis=1))))
        np.testing.assert_allclose(actual, points-anchor, atol=2e-4*length, rtol=0)
    count = 0
    for group in result.groups:
        surface = by_key[group.surface]
        ids = surface.vertex_ids
        facets = ((0,1,2),) if surface.kind=="triangle" else tuple(
            tuple(ids.index(i) for i in face) for face in surface.triangles)
        vertices = expected[group.surface]
        axes = [ref.axis(expected[key]) for key in group.round_keys]
        radii = [by_key[key].radius for key in group.round_keys]
        wanted = ref.aabb_candidates(vertices, facets, axes, radii, result.margin)
        actual = tuple(map(tuple, group.candidates.tolist()))
        assert actual == wanted
        assert group.facet_vertices == tuple(tuple(ids[i] for i in face) for face in facets)
        p, f, r = len(actual), len(facets), len(axes)
        words = (f+29)//30
        assert group.bvh_raw.shape == (r,words+3)
        bits = tuple((s,t) for s in range(r) for t in range(f)
                     if int(group.bvh_raw[s,t//30]) & (1 << (t%30)))
        assert bits == actual
        assert np.all(group.bvh_raw[:,-1] == 2*f-1)
        assert group.leaf_bounds.shape == (f,6) and group.query_bounds.shape == (r,6)
        packet = group.packet
        assert packet.schema == "phymeo-resident-triangle-distance-v1"
        assert packet.source_binding == result.source_binding
        assert packet.surface == group.surface and packet.revision == surface.revision
        np.testing.assert_array_equal(packet.facet_ids, np.arange(f,dtype=np.int64))
        np.testing.assert_array_equal(packet.segment_ids, np.arange(r,dtype=np.int64))
        np.testing.assert_array_equal(packet.candidates, group.candidates)
        assert packet.pair_origins_relative_m.shape == (p,3)
        for key, shape, dtype in (
            ("candidates",(p,18,11),np.float32), ("valid",(p,18),np.int32),
            ("summary",(p,16),np.float32), ("meta",(p,7),np.int32)):
            assert packet.raw[key].shape == shape and packet.raw[key].dtype == dtype
        assert packet.packed["geometry"].shape == (p,5,3)
        assert packet.packed["owners"].shape == (p,2)
        assert packet.packed["owners"].dtype == np.int32
        assert packet.packed["radii"].shape == (p,)
        assert packet.packed["scales"].shape == (p,)
        immutable(packet.raw)
        immutable(packet.packed)
        immutable(packet.candidates)
        assert len(packet.rows)==p
        for index,(row,(segment,facet)) in enumerate(zip(packet.rows,actual,strict=True)):
            assert (row.segment_id,row.facet_id)==(segment,facet)
            assert row.vertex_ids==group.facet_vertices[facet]
            tri = vertices[list(facets[facet])]
            ref.assert_row(row,tri,axes[segment],radii[segment],packet.origin_m,
                state=(statuses or {}).get((group.surface,segment,facet),"RESOLVED"))
            pair_origin = np.asarray(packet.origin_m)+packet.pair_origins_relative_m[index]
            length=ref.distance_reference().local_scale(tri,axes[segment],radii[segment])
            np.testing.assert_allclose(packet.packed["geometry"][index],
                np.vstack((tri,axes[segment]))-pair_origin,atol=2e-4*length,rtol=0)
            assert abs(packet.packed["scales"][index]-length)<=2e-4*length
        count += p
    assert result.required_candidates==count
    assert len(result.pairs)==sum(a.key.world==b.key.world
        for i,a in enumerate(model.shapes) for b in model.shapes[i+1:])
    assert plan.stats()["live_bytes"]==plan.stats()["persistent_bytes"]
    assert plan.stats()["peak_owned_bytes"]<=plan.stats()["memory_limit_bytes"]


@pytest.mark.tyx
@pytest.mark.parametrize("layout",LAYOUTS)
@pytest.mark.parametrize("static_surface",(True,False),ids=("M01-static","M02-double"))
def test_four_layouts_three_real_steps(layout,static_surface,tyx_mode,tmp_path):
    trees={w:fsrc.declaration(layout,i) for i,w in enumerate(WORLDS)}
    with ph.TyxWorldBatch(list(trees.items())[::-1],tyx_mode,cache_root=tmp_path) as batch:
        model,bindings,data=moving_scene(batch,trees,{w:layout for w in trees},
                                        static_surface=static_surface,order=True)
        with closing(batch.prepare_boundary_query(model,bindings,
            limits=ph.BoundaryDistanceLimits(),memory_limit_bytes=BUDGET)) as plan:
            with saved(tmp_path,"moving-"+layout+"-"+str(static_surface),tyx_mode,
                       inputs=dict(model=model,bindings=bindings,declarations=data),plan=plan) as record:
                initial=batch.snapshot()
                last_points=None
                for _ in range(3):
                    assert all(batch.step(H).values())
                    states=batch.snapshot()
                    actual=query(batch,plan,record,.5)
                    expected=expected_scene(states,data)
                    assert_packet(batch,plan,actual,model,expected)
                    for world in trees:
                        assert np.any(np.asarray(initial[world].velocity)!=0)
                        assert states[world].tick>initial[world].tick
                        if last_points is not None:
                            assert not np.array_equal(expected[ph.CollisionKey(world,202,0)],
                                                      last_points[ph.CollisionKey(world,202,0)])
                            if not static_surface:
                                assert not np.array_equal(expected[ph.CollisionKey(world,101,0)],
                                                          last_points[ph.CollisionKey(world,101,0)])
                    last_points=expected


@pytest.mark.tyx
def test_four_layouts_mixed_sparse_worlds(tyx_mode,tmp_path):
    layouts=dict(zip(WORLDS,LAYOUTS,strict=True))
    trees={w:fsrc.declaration(layouts[w],i) for i,w in enumerate(WORLDS)}
    with ph.TyxWorldBatch(list(trees.items())[::-1],tyx_mode,cache_root=tmp_path) as batch:
        model,bindings,data=moving_scene(batch,trees,layouts,order=True)
        with closing(batch.prepare_boundary_query(model,bindings,
            limits=ph.BoundaryDistanceLimits(),memory_limit_bytes=BUDGET)) as plan:
            with saved(tmp_path,"M06-mixed",tyx_mode,inputs=data,plan=plan) as record:
                for _ in range(3):
                    assert all(batch.step(H).values())
                    result=query(batch,plan,record,.5)
                    assert_packet(batch,plan,result,model,expected_scene(batch.snapshot(),data))
                    assert {g.surface.world for g in result.groups}==set(WORLDS)


@pytest.mark.tyx
@pytest.mark.parametrize("case",("front","back","edge","vertex","through","outside"))
def test_finite_features_both_round_kinds(case,tyx_mode,tmp_path):
    tree=fsrc.declaration("ordinary",0)
    with ph.TyxWorldBatch([(7,tree)],tyx_mode,cache_root=tmp_path) as batch:
        with saved(tmp_path,"M04-"+case,tyx_mode,inputs=dict(case=case)) as record:
            for kind in ("sphere","capsule"):
                triangle,points,margin=reference().plain_flat(case,kind)
                model,bindings,expected=static_scene(batch,tree,triangle=triangle,
                                                     round_points=points,kind=kind)
                with closing(batch.prepare_boundary_query(model,bindings,
                    limits=ph.BoundaryDistanceLimits(),memory_limit_bytes=BUDGET)) as plan:
                    value=query(batch,plan,record,margin)
                    assert_packet(batch,plan,value,model,expected)
                    if case=="outside":
                        assert value.required_candidates==0
                        assert value.pairs[0].status=="no_boundary_candidate"
                    else:
                        assert value.required_candidates==1


@pytest.mark.tyx
@pytest.mark.parametrize("case",("shared_pruned","concave","closed_inside"))
def test_original_mesh_faces_shared_edges_and_closed_boundary(case,tyx_mode,tmp_path):
    tree=fsrc.declaration("ordinary",0)
    points,faces,center,margin=reference().mesh(case)
    ids=tuple(11+13*i for i in range(len(points)))
    with ph.TyxWorldBatch([(7,tree)],tyx_mode,cache_root=tmp_path) as batch:
        model,bindings,expected=static_scene(batch,tree,triangle=points,mesh_faces=faces,
                                             vertex_ids=ids,round_points=center[None])
        with closing(batch.prepare_boundary_query(model,bindings,
            limits=ph.BoundaryDistanceLimits(),memory_limit_bytes=BUDGET)) as plan:
            with saved(tmp_path,"M05-"+case,tyx_mode,inputs=dict(model=model,bindings=bindings),
                       plan=plan) as record:
                value=query(batch,plan,record,margin)
                assert_packet(batch,plan,value,model,expected)
                assert value.required_candidates==dict(shared_pruned=2,concave=4,closed_inside=12)[case]
                if case=="closed_inside":
                    empty=query(batch,plan,record,.01)
                    assert_packet(batch,plan,empty,model,expected)
                    assert empty.required_candidates==0
                    assert empty.pairs[0].status=="no_boundary_candidate"
                    assert empty.semantics=="threshold-limited-boundary-distance"


@pytest.mark.tyx
def test_stable_material_ids_survive_vertex_storage_permutation(tyx_mode,tmp_path):
    tree=fsrc.declaration("ordinary",0)
    points,faces,center,margin=reference().mesh("shared_pruned")
    ids=tuple(11+13*i for i in range(len(points)))
    permutation=np.array((7,2,5,0,6,1,4,3))
    inverse=np.argsort(permutation)
    with ph.TyxWorldBatch([(7,tree)],tyx_mode,cache_root=tmp_path) as batch:
        with saved(tmp_path,"M07-storage-permutation",tyx_mode,inputs=dict(ids=ids,
                   permutation=permutation,points=points,faces=faces)) as record:
            packets=[]
            for permuted in (False,True):
                model,bindings,expected=static_scene(batch,tree,
                    triangle=points[permutation] if permuted else points,
                    mesh_faces=inverse[faces] if permuted else faces,
                    vertex_ids=tuple(ids[i] for i in permutation) if permuted else ids,
                    round_points=center[None])
                with closing(batch.prepare_boundary_query(model,bindings,
                    limits=ph.BoundaryDistanceLimits(),memory_limit_bytes=BUDGET)) as plan:
                    value=query(batch,plan,record,margin)
                    assert_packet(batch,plan,value,model,expected)
                    packets.append(value)
            a,b=(p.groups[0].packet for p in packets)
            assert [r.vertex_ids for r in a.rows]==[r.vertex_ids for r in b.rows]
            np.testing.assert_allclose([r.distance for r in a.rows],[r.distance for r in b.rows],
                                       atol=1e-4*np.sqrt(8),rtol=0)


@pytest.mark.tyx
def test_large_world_anchor_keeps_small_local_geometry(tyx_mode,tmp_path):
    from phymeo.dynamics.floating_robot import FloatingRobotState
    tree=fsrc.declaration("floating",1)
    # Original F0 state, only the predeclared M07 common root translation changes.
    state=tree.state
    q=np.asarray(state.q,dtype=np.float64).copy()
    q[:3]=(2**24,-2**24,2**24)
    tree=replace(tree,state=FloatingRobotState(q,state.v))
    with ph.TyxWorldBatch([(17,tree)],tyx_mode,cache_root=tmp_path) as batch:
        model,bindings,data=moving_scene(batch,{17:tree},{17:"floating"})
        with closing(batch.prepare_boundary_query(model,bindings,
            limits=ph.BoundaryDistanceLimits(),memory_limit_bytes=BUDGET)) as plan:
            with saved(tmp_path,"M07-large-anchor",tyx_mode,inputs=data,plan=plan) as record:
                for _ in range(3):
                    assert batch.step(H)=={17:True}
                    result=query(batch,plan,record,.5)
                    assert_packet(batch,plan,result,model,expected_scene(batch.snapshot(),data))
                    assert np.max(result.groups[0].packet.packed["scales"])<4


@lru_cache(maxsize=1)
def rotation_builders():
    """Reuse old physical constructors only; never their product geometry/audit."""
    path=Path(__file__).parents[1]/"scale"
    old=list(sys.path)
    try:
        sys.path.insert(0,str(path))
        spec=importlib.util.spec_from_file_location("m1_rotation_builders",
                                                  path/"test_resident_device_contacts.py")
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:]=old


def rotation_terms(builders,item,layout,checkpoint):
    data=item["data"]
    if layout=="ordinary":
        return builders.dynamic._reference().oracle(data["model"],
            np.asarray(checkpoint.position),np.asarray(checkpoint.velocity))
    if layout=="floating":
        return builders.floating._reference().base().mechanics(data,
            np.asarray(checkpoint.position),np.asarray(checkpoint.velocity))
    ref=builders.floating_affine._reference()
    return ref.mechanics(data,np.asarray(checkpoint.full_position),
                         np.asarray(checkpoint.full_velocity))


@pytest.mark.tyx
@pytest.mark.parametrize("layout",("ordinary","floating","floating_affine"))
def test_true_rotation_drive_and_external_load_source(layout,tyx_mode,tmp_path):
    builders=rotation_builders()
    item,tree,command=builders.fixture(layout,tmp_path/"asset")
    with ph.TyxWorldBatch([(83,tree)],tyx_mode,cache_root=tmp_path) as batch:
        batch.set_controls({83:command})
        state=batch.snapshot()[83]
        initial=rotation_terms(builders,item,layout,state)
        groups=(0,len(initial["position"])-1)
        shapes=[ph.CollisionShape(ph.CollisionKey(83,101,0),"triangle",reference().IDS)]
        bindings=[binding(tree,state,layout,shapes[0].key,groups[0],reference().TRIANGLE)]
        for ordinal,kind in enumerate(("sphere","capsule")):
            shape=ph.CollisionShape(ph.CollisionKey(83,202,ordinal),kind,
                (11,) if kind=="sphere" else (11,37),radius=reference().RADIUS)
            shapes.append(shape)
            bindings.append(binding(tree,state,layout,shape.key,groups[1],reference().local_round(kind)))
        model=ph.CollisionModel(tuple(shapes))
        with closing(batch.prepare_boundary_query(model,tuple(bindings),
            limits=ph.BoundaryDistanceLimits(),memory_limit_bytes=BUDGET)) as plan:
            with saved(tmp_path,"M03-"+layout,tyx_mode,
                       inputs=dict(data=item["data"],model=model,bindings=bindings),plan=plan) as record:
                rotations=[]
                for _ in range(3):
                    assert batch.step(H)=={83:True}
                    current=batch.snapshot()[83]
                    terms=rotation_terms(builders,item,layout,current)
                    expected={}
                    for shape,group,local in (
                        (shapes[0],groups[0],reference().TRIANGLE),
                        (shapes[1],groups[1],reference().local_round("sphere")),
                        (shapes[2],groups[1],reference().local_round("capsule"))):
                        expected[shape.key]=terms["position"][group]+np.asarray(local)@terms["rotation"][group].T
                    value=query(batch,plan,record,1.)
                    assert_packet(batch,plan,value,model,expected)
                    rotations.append(np.asarray(terms["rotation"]).copy())
                assert not np.array_equal(rotations[-1],initial["rotation"])
                assert np.any(np.asarray(current.velocity)!=0)


@pytest.mark.tyx
def test_no_host_current_geometry_or_closest_search(tyx_mode,tmp_path,monkeypatch):
    import phymeo.backends._device_triangle_distance as direct
    import phymeo.geometry._capsule_triangle_sweep as host
    trees={w:fsrc.declaration("ordinary",i) for i,w in enumerate(WORLDS)}
    def forbidden(*args,**kwargs):
        raise AssertionError("M1 used HOST current geometry or closest search")
    with ph.TyxWorldBatch(list(trees.items()),tyx_mode,cache_root=tmp_path) as batch:
        model,bindings,data=moving_scene(batch,trees,{w:"ordinary" for w in trees})
        with closing(batch.prepare_boundary_query(model,bindings,
            limits=ph.BoundaryDistanceLimits(),memory_limit_bytes=BUDGET)) as plan:
            assert all(batch.step(H).values())
            expected=expected_scene(batch.snapshot(),data)
            with saved(tmp_path,"M09-no-host-current",tyx_mode,inputs=data,plan=plan) as record:
                with monkeypatch.context() as patch:
                    patch.setattr(host,"segment_minimum",forbidden)
                    patch.setattr(direct.DeviceTriangleDistanceBatch,"_prepare",forbidden)
                    for name in ("tree_kinematics","affine_tree_kinematics",
                                 "floating_tree_kinematics","floating_affine_tree_kinematics"):
                        if hasattr(batch,name):
                            patch.setattr(batch,name,forbidden)
                    patch.setattr(np.linalg,"solve",forbidden)
                    patch.setattr(np.linalg,"lstsq",forbidden)
                    value=query(batch,plan,record,.5)
                assert_packet(batch,plan,value,model,expected)
                assert plan.stats()["launches"]>0
                assert plan.stats()["downloads"]>0 and plan.stats()["downloaded_bytes"]>0


@pytest.mark.tyx
@pytest.mark.parametrize("case",("near_parallel_conditioning","zero_area","through_face"))
def test_original_conditioning_and_zero_direction_status(case,tyx_mode,tmp_path):
    data=reference().distance_reference().make_case(case)
    tree=fsrc.declaration("ordinary",0)
    with ph.TyxWorldBatch([(7,tree)],tyx_mode,cache_root=tmp_path) as batch:
        model,bindings,expected=static_scene(batch,tree,triangle=data["vertices"],
            vertex_ids=tuple(data["vertex_ids"]),round_points=data["segments"][0],kind="capsule")
        with closing(batch.prepare_boundary_query(model,bindings,
            limits=ph.BoundaryDistanceLimits(),memory_limit_bytes=BUDGET)) as plan:
            with saved(tmp_path,"M14-"+case,tyx_mode,inputs=data,plan=plan) as record:
                value=query(batch,plan,record,.5)
                state=dict(near_parallel_conditioning="CONDITIONING",zero_area="DEGENERATE",
                           through_face="RESOLVED")[case]
                status={(model.shapes[0].key,0,0):state}
                assert_packet(batch,plan,value,model,expected,statuses=status)
                assert value.required_candidates==1
                assert value.groups[0].packet.rows[0].normal is None
