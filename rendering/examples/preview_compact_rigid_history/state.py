"""Presented-frame binding and pure rigid-coordinate checks; no image history."""
from copy import deepcopy
from dataclasses import asdict,is_dataclass
from hashlib import sha256
import json
import math
import re

from preview_compact_history.state import HistoryKey


SHA=re.compile(r"[0-9a-f]{64}\Z")
PRESENTATION_FIELDS={"source_sha256","index","current_key","current_time_s","current_frame_sha256",
                     "previous_key","previous_time_s","previous_frame_sha256"}
KEY_FIELDS={"scene_key","world_id","epoch","revision","tick","phase"}


def plain(value):return asdict(value) if is_dataclass(value) else deepcopy(value)


def frame_digest(value):
    return sha256(json.dumps(plain(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _sha(value):
    if not isinstance(value,str) or SHA.fullmatch(value) is None:
        raise ValueError("exact lowercase SHA256 required")


def _binding(value):
    if not isinstance(value,dict) or set(value)!=PRESENTATION_FIELDS:
        raise ValueError("complete presentation binding required")
    _sha(value["source_sha256"]);_sha(value["current_frame_sha256"])
    if type(value["index"]) is not int or value["index"]<0:raise ValueError("invalid presentation index")
    if set(value["current_key"])!=KEY_FIELDS or not math.isfinite(value["current_time_s"]):
        raise ValueError("invalid current presentation key/time")
    previous=(value["previous_key"],value["previous_time_s"],value["previous_frame_sha256"])
    if any(v is None for v in previous):
        if not all(v is None for v in previous):raise ValueError("partial predecessor binding")
    else:
        if set(previous[0])!=KEY_FIELDS or not math.isfinite(previous[1]):raise ValueError("invalid predecessor key/time")
        _sha(previous[2])


def validate_presentation(frame,binding):
    value=plain(frame);_binding(binding)
    if value["key"]!=binding["current_key"] or value["time_s"]!=binding["current_time_s"]:
        raise ValueError("current presentation key/time differs")
    if frame_digest(value)!=binding["current_frame_sha256"]:
        raise ValueError("current full frame digest differs")


def bind_presentations(frames,source_sha256,*,start_index=0):
    _sha(source_sha256)
    if type(start_index) is not int or start_index<0:raise ValueError("invalid start index")
    result=[];previous=None
    for index,frame in enumerate(frames,start_index):
        value=plain(frame)
        if previous is not None and (value["key"]["tick"]<=previous["key"]["tick"] or
                                     value["time_s"]<=previous["time_s"]):
            raise ValueError("presented source clock must increase")
        entry=dict(source_sha256=source_sha256,index=index,current_key=value["key"],current_time_s=value["time_s"],
            current_frame_sha256=frame_digest(value),previous_key=None if previous is None else previous["key"],
            previous_time_s=None if previous is None else previous["time_s"],
            previous_frame_sha256=None if previous is None else frame_digest(previous))
        validate_presentation(value,entry);result.append(entry);previous=value
    if not result:raise ValueError("empty presented sequence")
    return tuple(result)


class RigidHistoryKey(HistoryKey):
    def prepare(self,value):
        binding=deepcopy(value["presentation"]);_binding(binding)
        base={k:v for k,v in value.items() if k!="presentation"}
        key=HistoryKey().prepare(base)["key"] # reuse strict metadata validation only
        expected=dict(scene_key=key["scene"],world_id=key["world"],epoch=key["epoch"],revision=key["revision"],
                      tick=key["tick"],phase=key["phase"])
        if binding["current_key"]!=expected or binding["current_time_s"]!=key["time"]:
            raise ValueError("presentation does not bind current metadata")
        key["presentation"]=binding
        if self.key is None:reasons=["explicit" if self.explicit else "new_session"]
        else:
            old=self.key;old_binding=old["presentation"]
            reasons=[name for name in ("scene","world","epoch","phase","camera_id","extent","lens","lights","exposure","signal")
                     if key[name]!=old[name]]
            if binding["source_sha256"]!=old_binding["source_sha256"]:
                reasons.append("presentation_source")
            elif binding["index"]==old_binding["index"]:
                if binding!=old_binding:raise ValueError("repeated presentation changed full frame binding")
            else:
                if binding["index"]!=old_binding["index"]+1:
                    raise ValueError("presentation gap requires explicit reset")
                if (binding["previous_key"]!=old_binding["current_key"] or
                    binding["previous_time_s"]!=old_binding["current_time_s"] or
                    binding["previous_frame_sha256"]!=old_binding["current_frame_sha256"]):
                    raise ValueError("presentation predecessor differs from last accepted full frame")
                dt=key["time"]-old["time"]
                if key["tick"]<=old["tick"] or not 0<dt<=.1:
                    raise ValueError("presentation source clock is discontinuous")
                if key["revision"]<=old["revision"]:
                    raise ValueError("presentation revision does not increase")
            a,b=key["camera_pose"],old["camera_pose"]
            if math.dist(a[:3],b[:3])>.25 or abs(sum(x*y for x,y in zip(a[3:],b[3:])))<math.cos(math.radians(10)):
                reasons.append("camera_cut")
            if tuple(n[0] for n in key["nodes"])!=tuple(n[0] for n in old["nodes"]):reasons.append("node_catalog")
        sequence=0 if reasons else self.sequence+1
        if sequence>=2**32:raise ValueError("sample sequence exhausted")
        return dict(key=key,reasons=reasons,sequence=sequence,generation=self.generation)


def translation_matrix(position):
    return (1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,*position,1.)


def validate_rigid(matrix):
    if len(matrix)!=16 or not all(math.isfinite(v) for v in matrix):raise ValueError("finite rigid matrix required")
    if any(abs(matrix[i]-expected)>2e-5 for i,expected in ((3,0),(7,0),(11,0),(15,1))):raise ValueError("affine rigid matrix required")
    columns=[tuple(matrix[4*j+i] for i in range(3)) for j in range(3)]
    for i in range(3):
        for j in range(3):
            if abs(sum(a*b for a,b in zip(columns[i],columns[j]))-(i==j))>2e-5:raise ValueError("rigid scale/shear unsupported")
    a,b,c=columns
    determinant=a[0]*(b[1]*c[2]-b[2]*c[1])-b[0]*(a[1]*c[2]-a[2]*c[1])+c[0]*(a[1]*b[2]-a[2]*b[1])
    if determinant<=0:raise ValueError("reflected transform unsupported")


def map_rigid_normal(normal,current,previous):
    validate_rigid(current);validate_rigid(previous)
    local=tuple(sum(current[4*j+i]*normal[i] for i in range(3)) for j in range(3))
    return tuple(sum(previous[4*j+i]*local[j] for j in range(3)) for i in range(3))


def map_rigid_point(point,current,previous):
    local=map_rigid_normal(tuple(point[i]-current[12+i] for i in range(3)),current,previous)
    return tuple(local[i]+previous[12+i] for i in range(3))


def guide_matches(previous,current):
    return all(abs(a-b)<=.2*max(a,b)+1e-5 for a,b in zip(previous,current))


def incoming_matches(previous,low,high):
    return all(lo-(.5*(hi-lo)+.1*max(hi,1e-5))<=v<=hi+(.5*(hi-lo)+.1*max(hi,1e-5))
               for v,lo,hi in zip(previous,low,high))
