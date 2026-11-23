"""Transactional HOST metadata only. All image history lives on the GPU."""
from copy import deepcopy
import math


FIELDS={"scene","world","epoch","phase","revision","tick","time","camera_id","extent",
        "lens","camera_pose","nodes","lights","exposure","signal"}
STATIC=("scene","world","epoch","phase","camera_id","extent","lens","nodes","lights","exposure","signal")


def _finite(value):
    if isinstance(value,dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value,(tuple,list)):
        return all(_finite(v) for v in value)
    return not isinstance(value,(int,float)) or (not isinstance(value,bool) and math.isfinite(value))


def sample_seed(base,sequence):
    if any(type(v) is not int or not 0<=v<2**32 for v in (base,sequence)):
        raise ValueError("seed and sequence must be uint32")
    return base ^ ((sequence*0x9E3779B9)&0xffffffff)


def accept_depth(a,b):
    return math.isfinite(a) and math.isfinite(b) and min(a,b)>0 and abs(a-b)<=.003*min(a,b)


def project_previous(pixel,depth,lens,current_minus_previous):
    """Small independent analytic oracle for parallel untranslated camera axes."""
    fx,fy,cx,cy=lens
    x=(pixel[0]-cx)*depth/fx+current_minus_previous[0]
    y=-(pixel[1]-cy)*depth/fy+current_minus_previous[1]
    z=depth-current_minus_previous[2]
    if z<=0:raise ValueError("point behind previous camera")
    return fx*x/z+cx,-fy*y/z+cy,z


class HistoryKey:
    def __init__(self):
        self.key=None
        self.sequence=-1
        self.generation=0
        self.explicit=False

    def reset(self):
        self.key=None
        self.sequence=-1
        self.generation+=1
        self.explicit=True

    def prepare(self,value):
        if set(value)!=FIELDS or not _finite(value):
            raise ValueError("complete finite history declaration required")
        key=deepcopy(value)
        for name in ("epoch","revision","tick"):
            if type(key[name]) is not int or key[name]<0:
                raise ValueError("history integer identity required: "+name)
        for name in ("world","scene","phase","camera_id","signal"):
            if not isinstance(key[name],str) or not key[name]:
                raise ValueError("history string identity required: "+name)
        if len(key["camera_pose"])!=7 or len(key["extent"])!=2 or any(type(n) is not int or n<1 for n in key["extent"]):
            raise ValueError("invalid history camera declaration")
        reasons=[]
        if self.key is None:
            reasons=["explicit" if self.explicit else "new_session"]
        else:
            old=self.key
            reasons=[name for name in STATIC if key[name]!=old[name]]
            same_clock=key["tick"]==old["tick"]
            if same_clock and key["time"]!=old["time"]:
                raise ValueError("same tick cannot declare a different exposure time")
            if key["tick"] not in (old["tick"],old["tick"]+1):reasons.append("tick_gap")
            dt=key["time"]-old["time"]
            if dt<0 or dt>.1 or (not same_clock and dt==0):reasons.append("time_gap")
            if key["revision"]<old["revision"]:reasons.append("revision_backwards")
            a,b=key["camera_pose"],old["camera_pose"]
            translation=math.dist(a[:3],b[:3])
            dot=abs(sum(x*y for x,y in zip(a[3:],b[3:])))
            if translation>.25 or dot<math.cos(math.radians(10)):
                reasons.append("camera_cut")
        sequence=0 if reasons else self.sequence+1
        if sequence>=2**32:raise ValueError("history sequence exhausted; reset required")
        return dict(key=key,reasons=reasons,sequence=sequence,generation=self.generation)

    def commit(self,plan):
        if plan["generation"]!=self.generation:
            raise ValueError("stale history transaction")
        self.key=deepcopy(plan["key"])
        self.sequence=plan["sequence"]
        self.explicit=False
        self.generation+=1
