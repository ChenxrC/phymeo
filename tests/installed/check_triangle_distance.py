"""Source-external consumer of installed finite triangle distance packets.

Copy this file outside every checkout. The caller configures the installed wheel
and existing TYX dependencies. This script never edits sys.path, imports source
examples/tests, calls an old BVH producer or accesses private device buffers.
"""
import argparse
import base64
import copy
import gzip
import hashlib
import json
import math
import struct
import sys
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
import numpy as np


SCHEMA="phymeo-installed-triangle-distance-v1"
DISTANCE_GATE=1e-4
WITNESS_GATE=2e-4
NORMAL_GATE=2e-3


def require(condition,message):
    if not condition:
        raise AssertionError(message)


def encode(value):
    if isinstance(value,np.ndarray):
        require(not value.dtype.hasobject,"object array in evidence")
        return dict(dtype=value.dtype.str,shape=list(value.shape),bytes_b64=base64.b64encode(value.tobytes()).decode())
    if isinstance(value,np.generic):
        return dict(dtype=value.dtype.str,scalar_b64=base64.b64encode(value.tobytes()).decode())
    if is_dataclass(value):
        return dict(type=type(value).__module__+"."+type(value).__name__,fields={f.name:encode(getattr(value,f.name)) for f in fields(value)})
    if isinstance(value,Mapping):
        return dict(mapping_entries=[[encode(k),encode(v)] for k,v in value.items()])
    if isinstance(value,tuple):
        return dict(tuple=[encode(v) for v in value])
    if isinstance(value,list):
        return [encode(v) for v in value]
    if type(value) is float:
        return dict(float64_bits=struct.pack(">d",value).hex())
    if value is None or isinstance(value,(str,bool,int)):
        return value
    raise TypeError("unhandled evidence value "+type(value).__name__)


class Journal:
    def __init__(self,path):
        self.path=path
        self.stream=gzip.open(path,"xb")
        self.sha=hashlib.sha256()
        self.logical_bytes=self.events=0
    def emit(self,event,**payload):
        record=dict(schema=SCHEMA,ordinal=self.events,event=event,payload=encode(payload))
        line=(json.dumps(record,separators=(",",":"),allow_nan=False)+"\n").encode()
        self.stream.write(line)
        self.stream.flush()
        self.sha.update(line)
        self.logical_bytes+=len(line)
        self.events+=1
    def close(self):
        self.stream.close()
        return dict(path=str(self.path),events=self.events,logical_bytes=self.logical_bytes,
                    logical_sha256=self.sha.hexdigest(),compressed_bytes=self.path.stat().st_size,
                    compressed_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())


def provenance(installed_root,*,preflight=False):
    def source_entry(path):
        return ((path/"pyproject.toml").is_file() and (path/"src"/"phymeo").is_dir()) or (
            path.name.lower()=="src" and (path/"phymeo").is_dir() and (path.parent/"pyproject.toml").is_file())
    require((installed_root/"phymeo"/"__init__.py").is_file(),"installed package root required")
    paths=[Path(entry or Path.cwd()).resolve() for entry in sys.path]
    require(not any(source_entry(path) for path in paths),"checkout source entry in sys.path")
    if preflight:
        require(not any(name=="phymeo" or name.startswith("phymeo.") for name in sys.modules),"Phymeo imported before preflight")
        for origin in (Path.cwd().resolve(),Path(__file__).resolve().parent,installed_root):
            require(not any(source_entry(parent) for parent in (origin,*origin.parents)),"consumer/cwd/install must be outside checkouts")
    import phymeo
    require(Path(phymeo.__file__).resolve().is_relative_to(installed_root),"installed phymeo origin")
    loaded={}
    for name,module in sorted(sys.modules.items()):
        if not (name=="phymeo" or name.startswith("phymeo.")):
            continue
        filename=getattr(module,"__file__",None)
        if filename is not None:
            actual=Path(filename).resolve(strict=True)
            require(actual.is_relative_to(installed_root),"noninstalled product module "+name)
            loaded[name]=dict(path=str(actual),sha256=hashlib.sha256(actual.read_bytes()).hexdigest())
        for directory in getattr(module,"__path__",()):
            require(Path(directory).resolve(strict=True).is_relative_to(installed_root),"noninstalled package search path "+name)
    return dict(installed_root=str(installed_root),cwd=str(Path.cwd().resolve()),consumer=str(Path(__file__).resolve()),
                sys_path=[str(path) for path in paths],loaded=loaded,
                numpy=dict(version=np.__version__,path=str(Path(np.__file__).resolve())))


def inputs():
    origin=np.array((2**20,-2**20,2**20),dtype=np.float64)
    local=np.array(((0,0,0),(2,0,0),(0,2,0),(2,2,0)),dtype=np.float64)
    axes=np.array((((1,1,.25),(1,1,.25)),((.5,1,.5),(1.5,1,.5))),dtype=np.float64)
    return dict(mesh_key="installed-shared-edge",revision=7,snapshot_id="flat-plane",
                vertex_ids=np.array((11,37,91,207),dtype=np.int64),facet_ids=np.array((301,907),dtype=np.int64),
                faces=np.array(((0,1,2),(3,2,1)),dtype=np.int64),vertices=local+origin,
                segment_ids=np.array((7,91),dtype=np.int64),segments=axes+origin,
                radii=np.array((.125,.125),dtype=np.float64),origin_m=origin,
                candidates=np.array(((7,301),(7,907),(91,301),(91,907)),dtype=np.int64))


def verify(data,packet,plane_height):
    require(packet.schema=="phymeo-device-triangle-distance-v1","packet schema")
    require(packet.mesh_key==data["mesh_key"] and packet.revision==7,"mesh binding")
    require(packet.snapshot_id==data["snapshot_id"] and len(packet.input_binding)==64,"snapshot binding")
    require(len(packet.rows)==4,"complete two-segment/two-facet membership")
    for name in ("vertices","faces","vertex_ids","facet_ids","segment_ids","segments","radii","candidates","origin_m"):
        np.testing.assert_array_equal(packet.inputs[name],data[name])
        require(not packet.inputs[name].flags.writeable,"immutable original input")
    require(set(packet.raw)=={"candidates","valid","summary","meta"},"complete raw fields")
    require(packet.raw["candidates"].shape==(4,18,11) and packet.raw["valid"].shape==(4,18),"raw candidate layout")
    require(packet.raw["summary"].shape==(4,16) and packet.raw["meta"].shape==(4,7),"raw row layout")
    for array in packet.raw.values():
        require(not array.flags.writeable and np.all(np.isfinite(array)),"immutable finite raw data")
    require(set(packet.uploaded)=={"geometry","radii","scales","owners"},"complete upload fields")
    require(set(packet.conversion_deltas)=={"geometry","radii","scales","published_axis","published_triangle"},"complete conversion fields")
    for collection in (packet.uploaded,packet.conversion_deltas):
        for array in collection.values():
            require(not array.flags.writeable and np.all(np.isfinite(array)),"immutable finite source conversion")
    require(not packet.pair_origins_relative_m.flags.writeable,"immutable pair frames")
    length=math.sqrt(8.0)  # Each original triangle has a diagonal of length sqrt(8).
    checks=[]
    for ordinal,(sid,fid) in enumerate(data["candidates"]):
        si=0 if sid==7 else 1
        fi=0 if fid==301 else 1
        row=packet.rows[ordinal]
        require(row.status=="RESOLVED","ordinary finite query must resolve")
        require((row.segment_id,row.facet_id)==(int(sid),int(fid)),"original material owners")
        require(row.vertex_ids==tuple(map(int,data["vertex_ids"][data["faces"][fi]])),"original facet vertex ownership")
        bary=np.array(row.barycentric,dtype=np.float64)
        require(0<=row.axis_parameter<=1 and np.min(bary)>=0,"finite address bounds")
        require(abs(float(bary.sum())-1)<=8*2**-24,"barycentric sum")
        triangle=data["vertices"][data["faces"][fi]]-data["origin_m"]
        segment=data["segments"][si]-data["origin_m"]
        axis_point=segment[0]+row.axis_parameter*(segment[1]-segment[0])
        triangle_point=bary@triangle
        expected_projection=axis_point.copy()
        expected_projection[2]=plane_height
        expected_distance=(.25 if si==0 else .5)-plane_height
        require(abs(row.distance-expected_distance)<=DISTANCE_GATE*length,"independent plane distance")
        require(abs(row.gap-(expected_distance-.125))<=DISTANCE_GATE*length,"independent radius gap")
        np.testing.assert_allclose(row.axis_point_relative_m,axis_point,atol=WITNESS_GATE*length,rtol=0)
        np.testing.assert_allclose(row.triangle_point_relative_m,triangle_point,atol=WITNESS_GATE*length,rtol=0)
        np.testing.assert_allclose(triangle_point,expected_projection,atol=WITNESS_GATE*length,rtol=0)
        require(row.normal is not None and np.linalg.norm(np.asarray(row.normal)-(0,0,-1))<=NORMAL_GATE,"independent downward normal")
        require(row.selected_candidate in row.tied_candidates,"selected finite candidate retained")
        expected_span=(0.,1.) if si==0 else ((0.,.5) if fi==0 else (.5,1.))
        np.testing.assert_allclose(row.axis_span,expected_span,atol=2e-4,rtol=0)
        if si==0:
            np.testing.assert_allclose(bary,(0,.5,.5),atol=2e-4,rtol=0)
        np.testing.assert_array_equal(packet.pair_origins_relative_m[ordinal],triangle[0])
        require(np.max(np.abs(packet.conversion_deltas["geometry"][ordinal]))<=8*2**-24*length,"input conversion gate")
        for kind in ("published_axis","published_triangle"):
            require(np.linalg.norm(packet.conversion_deltas[kind][ordinal])<=WITNESS_GATE*length,"publication delta gate")
        raw=packet.raw["candidates"][ordinal]
        flags=packet.raw["valid"][ordinal]
        require(np.all((flags==0)|(flags==1)),"all finite candidate validity flags")
        for index in range(18):
            if not flags[index]:
                require(np.all(raw[index]==0),"invalid slot fully initialized")
                continue
            s,w0,w1,w2,d2=map(float,raw[index,:5])
            require(0<=s<=1 and min(w0,w1,w2)>=0 and d2>=0,"raw finite parameter bounds")
            require(abs(w0+w1+w2-1)<=8*2**-24,"raw barycentric sum")
            a=segment[0]+s*(segment[1]-segment[0])
            b=np.array((w0,w1,w2))@triangle
            np.testing.assert_allclose(raw[index,5:8]+packet.pair_origins_relative_m[ordinal],a,atol=WITNESS_GATE*length,rtol=0)
            np.testing.assert_allclose(raw[index,8:11]+packet.pair_origins_relative_m[ordinal],b,atol=WITNESS_GATE*length,rtol=0)
            require(abs(d2-float((b-a)@(b-a)))<=WITNESS_GATE*length*length,"raw squared distance consistency")
        checks.append(dict(segment_id=int(sid),facet_id=int(fid),expected_distance=expected_distance,
                           expected_normal=(0.,0.,-1.),expected_axis_span=expected_span))
    return checks


def run(mode,journal,installed_root):
    from phymeo.backends._device_triangle_distance import DeviceTriangleDistanceBatch
    first_input=inputs()
    changed=copy.deepcopy(first_input)
    changed["vertices"][:,2]+=1/16
    changed["snapshot_id"]="plane-raised-one-sixteenth"
    batch=DeviceTriangleDistanceBatch(mode=mode,vertex_capacity=4,face_capacity=2,segment_capacity=2,pair_capacity=4)
    attempted=first_input
    try:
        first=batch.query(**first_input)
        journal.emit("query",phase="initial",source=first_input,packet=first,counters=dict(batch.stats))
        first_checks=verify(first_input,first,0.0)
        first_encoded=encode(first)
        batch.require_current(first,**first_input)
        before=dict(batch.stats)
        try:
            batch.require_current(first,**changed)
        except ValueError as error:
            require("binding" in str(error),"changed source rejected by input binding")
            journal.emit("changed-source-rejected",source=changed,old_binding=first.input_binding,error=str(error),counters=dict(batch.stats))
        else:
            raise AssertionError("old packet accepted for changed source")
        require(dict(batch.stats)==before,"binding check must not launch or mutate query counters")
        require(batch.last_packet is first,"binding rejection retains previous packet")
        attempted=changed
        second=batch.query(**changed)
        journal.emit("query",phase="changed-source",source=changed,packet=second,counters=dict(batch.stats))
        second_checks=verify(changed,second,1/16)
        require(second.input_binding!=first.input_binding,"actual source change changes binding")
        require(second.generation==first.generation+1,"accepted generation progression")
        require(batch.last_packet is second,"new query publication")
        require(encode(first)==first_encoded,"old immutable evidence survives next query")
        batch.require_current(second,**changed)
        try:
            batch.require_current(first,**first_input)
        except ValueError as error:
            require("current" in str(error),"old accepted packet becomes stale")
            journal.emit("old-packet-stale",old_binding=first.input_binding,current_binding=second.input_binding,error=str(error))
        else:
            raise AssertionError("superseded packet accepted as current")
        counts=dict(batch.stats)
        require(counts["queries"]==2 and counts["launches"]==2 and counts["completed_launches"]==2,"two real device queries")
        require(counts["failed_queries"]==0 and counts["live_bytes"]==0,"all query buffers released normally")
        if mode=="cuda":
            require(counts["uploads"]==8 and counts["downloads"]==8 and counts["synchronizations"]==2,"actual CUDA transfers and synchronization")
            require(counts["uploaded_bytes"]>0 and counts["downloaded_bytes"]>0,"nonempty CUDA data path")
        modules=provenance(installed_root)
        require("phymeo.backends._device_triangle_distance" in modules["loaded"],"installed packet adapter loaded")
        require("phymeo.backends._tyx_triangle_distance" in modules["loaded"],"installed kernel loaded")
        journal.emit("verified",independent_checks=(first_checks,second_checks),counters=counts,provenance=modules)
        return dict(actual_queries=2,candidate_rows=8,changed_binding_rejected=True,old_packet_stale=True,
                    counters=counts,final_provenance=modules)
    except BaseException:
        journal.emit("attempt-failure",attempted_source=attempted,last_packet=batch.last_packet,
                     last_attempt_raw=batch.last_attempt_raw,counters=dict(batch.stats))
        raise
    finally:
        batch.close()
        journal.emit("closed",counters=dict(batch.stats))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("interpret","cuda"),required=True)
    parser.add_argument("--installed-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    require(args.output.is_absolute(),"output directory must be absolute")
    require(args.installed_root.is_absolute(),"installed-root must be absolute")
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    journal=Journal(output/"triangle-distance.jsonl.gz")
    summary=dict(status="RUNNING",schema=SCHEMA,mode=args.mode,
                 consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 scope="installed current-pose finite segment-triangle boundary packet and changed-input binding",
                 excluded="B-point, load response, closed-solid inside/outside, CCD, BVH rerun, full source matrix",
                 gates=dict(distance_over_local_L=DISTANCE_GATE,witness_over_local_L=WITNESS_GATE,normal=NORMAL_GATE))
    try:
        installed_root=args.installed_root.resolve(strict=True)
        summary["initial_provenance"]=provenance(installed_root,preflight=True)
        journal.emit("provenance",details=summary["initial_provenance"],mode=args.mode)
        summary.update(run(args.mode,journal,installed_root))
        summary["status"]="PASS"
    except BaseException as error:
        summary["status"]="FAIL"
        summary["exception"]=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        journal.emit("failure",summary=summary)
        raise
    finally:
        journal.emit("summary",summary=summary)
        summary["trace"]=journal.close()
        with (output/"summary.json").open("x",encoding="utf-8") as stream:
            json.dump(summary,stream,indent=2,allow_nan=False)
        print(json.dumps(summary,allow_nan=False))


if __name__=="__main__":
    main()
