"""COL2 tests first: 24 fixed geometry nodes per actual TYX backend."""
import dataclasses
import gzip
import hashlib
import importlib.util
import json
from collections.abc import Mapping
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import numpy as np
import pytest
from phymeo.backends._device_triangle_distance import DeviceTriangleDistanceBatch


@lru_cache(maxsize=1)
def reference():
    path=Path(__file__).resolve().parents[1]/"reference_newton/rr/device_triangle_distance_reference.py"
    spec=importlib.util.spec_from_file_location("col2_reference",path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(params=("interpret","cuda"))
def mode(request):
    return request.param


def encode(value):
    if isinstance(value,np.ndarray):
        return dict(dtype=value.dtype.str,shape=list(value.shape),bytes=value.tobytes().hex())
    if dataclasses.is_dataclass(value):
        return {f.name:encode(getattr(value,f.name)) for f in dataclasses.fields(value)}
    if isinstance(value,Mapping):
        return {str(k):encode(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [encode(v) for v in value]
    if isinstance(value,np.generic):
        return value.item()
    return value


@contextmanager
def saved(path,name,mode,inputs,batch):
    report=dict(schema="COL2-boundary-distance-test-v1",identity=name,mode=mode,inputs=inputs,
                test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),complete=False)
    try:
        yield report
        report["complete"]=True
    except BaseException as error:
        report["failure"]=dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        report["attempt_raw"]=batch.last_attempt_raw
        report["last_packet"]=batch.last_packet
        report["stats"]=batch.stats
        with gzip.open(path/(name+".json.gz"),"wt",encoding="utf-8") as stream:
            json.dump(encode(report),stream,allow_nan=False)


def new_batch(mode,data,**overrides):
    options=dict(vertex_capacity=len(data["vertices"]),face_capacity=len(data["faces"]),
                 segment_capacity=len(data["segments"]),pair_capacity=len(data["candidates"]))
    options.update(overrides)
    return DeviceTriangleDistanceBatch(mode=mode,**options)


def assert_packet(data,packet):
    ref=reference()
    assert packet.schema=="phymeo-device-triangle-distance-v1"
    assert packet.mesh_key==data["mesh_key"] and packet.revision==data["revision"]
    assert packet.snapshot_id==data["snapshot_id"]
    assert len(packet.rows)==len(data["candidates"])
    assert len(packet.input_binding)==64
    for key in ("vertices","segments","candidates","vertex_ids","facet_ids","faces","segment_ids","origin_m","radii"):
        np.testing.assert_array_equal(packet.inputs[key],data[key])
        assert not packet.inputs[key].flags.writeable
    for name,array in packet.raw.items():
        assert not array.flags.writeable,name
    for ordinal,(sid,fid) in enumerate(data["candidates"]):
        row=packet.rows[ordinal]
        si=int(np.flatnonzero(data["segment_ids"]==sid)[0])
        fi=int(np.flatnonzero(data["facet_ids"]==fid)[0])
        tri=data["vertices"][data["faces"][fi]]-data["origin_m"]
        seg=data["segments"][si]-data["origin_m"]
        length=ref.local_scale(tri,seg,data["radii"][si])
        assert (row.segment_id,row.facet_id)==(int(sid),int(fid))
        assert row.vertex_ids==tuple(map(int,data["vertex_ids"][data["faces"][fi]]))
        if data["snapshot_id"]=="zero_area":
            assert row.status=="DEGENERATE"
            assert row.normal is None
            continue
        if data["snapshot_id"]=="near_parallel_conditioning":
            assert row.status=="CONDITIONING"
            assert row.normal is None
            continue
        assert row.status=="RESOLVED",(data["snapshot_id"],row)
        expected=ref.oracle(tri,seg)
        distance=float(expected["distance2"])**.5
        assert abs(row.distance-distance)<=1e-4*length
        assert abs(row.gap-(distance-data["radii"][si]))<=1e-4*length
        bary=np.array(row.barycentric)
        assert np.min(bary)>=0 and abs(float(sum(bary))-1)<=8*ref.U
        assert 0<=row.axis_parameter<=1
        ap=seg[0]+row.axis_parameter*(seg[1]-seg[0])
        tp=bary@tri
        np.testing.assert_allclose(row.axis_point_relative_m,ap,atol=2e-4*length,rtol=0)
        np.testing.assert_allclose(row.triangle_point_relative_m,tp,atol=2e-4*length,rtol=0)
        assert abs(float(np.linalg.norm(tp-ap))-distance)<=2e-4*length
        if distance==0:
            assert row.normal is None
        else:
            delta=np.array(expected["witnesses"][0][4],float)-np.array(expected["witnesses"][0][3],float)
            assert np.linalg.norm(np.array(row.normal)-delta/distance)<=2e-3
        values=packet.raw["candidates"][ordinal]
        valid=packet.raw["valid"][ordinal].astype(bool)
        assert values.shape==(18,11) and np.any(valid)
        best=float(values[row.selected_candidate,4])
        assert row.selected_candidate in row.tied_candidates
        for index in np.flatnonzero(valid):
            s,b0,b1,b2,d2=map(float,values[index,:5])
            assert 0<=s<=1 and min(b0,b1,b2)>=0
            assert abs(b0+b1+b2-1)<=8*ref.U
            x=seg[0]+s*(seg[1]-seg[0])
            y=np.array((b0,b1,b2))@tri
            np.testing.assert_allclose(values[index,5:8]+packet.pair_origins_relative_m[ordinal],x,atol=2e-4*length,rtol=0)
            np.testing.assert_allclose(values[index,8:11]+packet.pair_origins_relative_m[ordinal],y,atol=2e-4*length,rtol=0)
            assert abs(d2-float((y-x)@(y-x)))<=2e-4*length*length
            assert d2+1e-4*length*length>=best
            if abs(d2-best)<=ref.TIE_ERROR*length*length:
                assert int(index) in row.tied_candidates
        if data["snapshot_id"] in ("parallel_face","parallel_edge"):
            np.testing.assert_allclose(row.axis_span,np.array(expected["axis_span"],float),atol=2e-4,rtol=0)
        assert np.max(np.abs(packet.conversion_deltas["geometry"][ordinal]))<=ref.SOURCE_ERROR*length
        np.testing.assert_array_equal(packet.pair_origins_relative_m[ordinal],tri[0])


@pytest.mark.parametrize("name",reference().NAMES)
def test_fixed_finite_geometry(mode,name,tmp_path):
    data=reference().make_case(name)
    if name=="bvh_pruned":
        from phymeo.backends.tyx_collision import TyxCollisionBatch
        from phymeo.geometry._collision_bvh import CollisionTrees
        triangles=data["vertices"][data["faces"]]
        leaves=np.concatenate((triangles.min(axis=1),triangles.max(axis=1)),axis=1).astype(np.float32)
        driver=TyxCollisionBatch(mode,np.array((0,3),np.int32),np.zeros((1,6),np.int32),np.empty((0,2),np.int32),None)
        try:
            ids=CollisionTrees().facets({},("mesh","COL2"),leaves,np.array(((-1,-1,-.5,3,3,.5),),np.float32),driver)[0]
            assert ids==[0,1] and len(ids)<len(data["faces"])
            assert driver.stats()["completed_launches"]==2
            produced=np.array([(data["segment_ids"][0],data["facet_ids"][i]) for i in ids],np.int64)
            np.testing.assert_array_equal(produced,data["candidates"])
            data["candidates"]=produced
        finally:
            driver.close()
    with new_batch(mode,data) as batch:
        with saved(tmp_path,name,mode,data,batch) as record:
            packet=batch.query(**data)
            record["packet"]=packet
            assert_packet(data,packet)
            assert batch.last_packet is packet
            assert batch.stats["completed_launches"]==1
            assert batch.stats["live_bytes"]==0
            batch.require_current(packet,**data)
