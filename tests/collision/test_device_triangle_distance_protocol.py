"""COL2 protocol: no HOST search, explicit capacities, immutable/atomic output."""
import copy
import numpy as np
import pytest
import tyx as tx
from test_device_triangle_distance import assert_packet, mode, new_batch, reference, saved


@pytest.mark.parametrize("field",("vertex_capacity","face_capacity","segment_capacity","pair_capacity"))
def test_exact_and_one_less_capacity(mode,field,tmp_path):
    data=reference().make_case("face_above")
    with new_batch(mode,data) as batch:
        with saved(tmp_path,field,mode,data,batch):
            assert_packet(data,batch.query(**data))
    count={"vertex_capacity":3,"face_capacity":1,"segment_capacity":1,"pair_capacity":1}[field]
    with pytest.raises(ValueError,match="capacity"):
        with new_batch(mode,data,**{field:count-1}) as batch:
            batch.query(**data)


@pytest.mark.parametrize("fault",("vertex-id","facet-id","segment-id","candidate","candidate-duplicate","radius","finite","dtype"))
def test_preflight_rejects_without_launch(mode,fault):
    data=reference().make_case("shared_edge")
    if fault=="vertex-id": data["vertex_ids"][1]=data["vertex_ids"][0]
    if fault=="facet-id": data["facet_ids"][1]=data["facet_ids"][0]
    if fault=="segment-id": data["segment_ids"]=np.array((-1,),np.int64)
    if fault=="candidate": data["candidates"][0,1]=999
    if fault=="candidate-duplicate": data["candidates"][1]=data["candidates"][0]
    if fault=="radius": data["radii"][0]=-.1
    if fault=="finite": data["vertices"][0,0]=float("nan")
    if fault=="dtype": data["faces"]=data["faces"].astype(np.int32)
    with new_batch(mode,data) as batch:
        with pytest.raises(ValueError): batch.query(**data)
        assert batch.stats["launches"]==0 and batch.last_packet is None


@pytest.mark.parametrize("fault",("download","ownership","barycentric"))
def test_late_failure_preserves_packet_and_real_retry(mode,fault,tmp_path,monkeypatch):
    data=reference().make_case("face_above")
    with new_batch(mode,data) as batch:
        old=batch.query(**data)
        real=batch._download
        def fail(array,label):
            out=real(array,label)
            if label=="meta":
                if fault=="download": raise RuntimeError("late download")
                if fault=="ownership": out[0,5]+=1
            if fault=="barycentric" and label=="summary": out[0,3]=2
            return out
        monkeypatch.setattr(batch,"_download",fail)
        with saved(tmp_path,fault,mode,data,batch) as record:
            with pytest.raises(RuntimeError): batch.query(**data)
            record["rejected_raw"]=batch.last_attempt_raw
            assert batch.last_packet is old and batch.stats["live_bytes"]==0
            assert batch.stats["failed_queries"]==1
            monkeypatch.setattr(batch,"_download",real)
            new=batch.query(**data)
            assert new is not old and new.input_binding==old.input_binding
            assert batch.stats["launches"]==3
            assert_packet(data,new)
            record["retry"]=new


def test_snapshot_and_owner_binding(mode):
    data=reference().make_case("face_above")
    with new_batch(mode,data) as first, new_batch(mode,data) as second:
        packet=first.query(**data)
        with pytest.raises(ValueError,match="owner"): second.require_current(packet,**data)
        changed=copy.deepcopy(data)
        changed["vertices"][0,2]=.125
        with pytest.raises(ValueError,match="binding"): first.require_current(packet,**changed)
        changed=copy.deepcopy(data)
        changed["snapshot_id"]="next"
        with pytest.raises(ValueError,match="binding"): first.require_current(packet,**changed)
        current=first.query(**changed)
        with pytest.raises(ValueError,match="current"): first.require_current(packet,**data)
        assert current.snapshot_id=="next"
        with pytest.raises(ValueError): current.raw["meta"].setflags(write=True)
        with pytest.raises(TypeError): current.inputs["new"]=1


def test_source_conversion_cannot_use_world_translation_as_scale(mode):
    data=reference().make_case("face_above")
    # Pre-execution correction: common translation is representable in a pair
    # frame. A distant small segment instead loses .25 relative to this triangle.
    data["segments"][:,:,0]+=2**24
    with new_batch(mode,data) as batch:
        with pytest.raises(ValueError,match="precision"): batch.query(**data)
        assert batch.stats["launches"]==0


def test_no_host_distance_search(mode,monkeypatch):
    import phymeo.geometry._capsule_triangle_sweep as host
    def forbidden(*args,**kwargs): raise AssertionError("HOST closest-point search")
    monkeypatch.setattr(host,"segment_minimum",forbidden)
    monkeypatch.setattr(np.linalg,"solve",forbidden)
    monkeypatch.setattr(np.linalg,"lstsq",forbidden)
    data=reference().make_case("parallel_face")
    with new_batch(mode,data) as batch:
        packet=batch.query(**data)
        assert packet.rows[0].status=="RESOLVED" and batch.stats["completed_launches"]==1


def test_empty_candidate_packet_has_no_launch(mode):
    data=reference().make_case("face_above")
    data["candidates"]=np.empty((0,2),np.int64)
    with new_batch(mode,data,pair_capacity=1) as batch:
        packet=batch.query(**data)
        assert packet.rows==() and batch.stats["launches"]==0
        assert batch.stats["live_bytes"]==0
        batch.require_current(packet,**data)


def test_borrowed_runtime_survives_batch_close(mode):
    data=reference().make_case("face_above")
    runtime=tx.Runtime(backend="cuda" if mode=="cuda" else "cpu")
    try:
        with new_batch(mode,data,runtime=runtime) as batch:
            assert_packet(data,batch.query(**data))
        with new_batch(mode,data,runtime=runtime) as next_batch:
            assert_packet(data,next_batch.query(**data))
    finally:
        runtime.close()
