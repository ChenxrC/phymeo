"""Nominal FP32 audit of source-declared support gates, never a renderer oracle.

The GPU readbacks are inputs. No ROI or material identity changes the gates.
Near-boundary comparisons are flagged instead of claiming exact GPU branching.
"""
from collections import Counter
import math
import numpy as np

F=np.float32
TRACED=1<<20
NORMAL=(1<<20)-1


def code(value):
    if not np.isfinite(value) or value<0 or value>16777215 or value!=math.floor(float(value)):
        raise ValueError("invalid actual packed guide code")
    return int(value)


def decode_normal(value):
    value=code(value)
    v=np.array([value&1023,(value>>10)&1023],np.float32)*F(2/1023)-F(1)
    n=np.array([v[0],v[1],F(1)-abs(v[0])-abs(v[1])],np.float32)
    if n[2]<0:n[:2]=(F(1)-np.abs(n[1::-1]))*np.where(n[:2]>=0,F(1),F(-1))
    return n/F(np.sqrt(np.dot(n,n)))


def depth_gate(a,b):
    a=F(a);b=F(b)
    if not np.isfinite(a) or not np.isfinite(b) or min(a,b)<=0:
        return False,None,False
    difference=abs(F(a-b));limit=F(F(.003)*min(a,b));slack=F(limit-difference)
    uncertain=abs(slack)<=F(8*np.finfo(np.float32).eps*max(a,b))
    return bool(slack>=0),float(slack),bool(uncertain)


def normal_gate(a,b):
    cosine=F(np.dot(a,b));slack=F(cosine-F(.98))
    return bool(slack>=0),float(slack),bool(abs(slack)<=F(8*np.finfo(np.float32).eps))


def validate_planes(*,ids,depth,valid,guide,attributes,low_ids,low_depth,low_valid,compact):
    if ids.ndim!=2 or ids.dtype!=np.uint32 or depth.shape!=ids.shape or depth.dtype!=np.float32 or valid.shape!=ids.shape or valid.dtype!=np.uint8:
        raise ValueError("actual full-rate ID/depth/valid planes required")
    h,w=ids.shape;gh,gw=((h+3)//4,(w+3)//4) if compact else (h,w)
    if guide.shape!=(gh,gw,4) or guide.dtype!=np.float32:
        raise ValueError("guide dimensions must match the actual sparse/compact raster")
    if compact:
        if attributes.shape!=(h,w,4) or attributes.dtype!=np.float32:raise ValueError("actual full-rate SH/Ns attributes required")
        for a,dtype in ((low_ids,np.uint32),(low_depth,np.float32),(low_valid,np.uint8)):
            if a.shape!=(gh,gw) or a.dtype!=dtype:raise ValueError("actual compact sensors required")


def inspect_pixel(x,y,*,ids,depth,valid,guide,attributes=None,low_ids=None,low_depth=None,low_valid=None,compact=True):
    """Audit K25/R2/tile4 support for one actual valid diffuse receiver.

    Caller proves diffuse eligibility from the declared material; this read-only
    routine does not have primary-alpha rho and cannot classify nondiffuse pixels.
    """
    validate_planes(ids=ids,depth=depth,valid=valid,guide=guide,attributes=attributes,
        low_ids=low_ids,low_depth=low_depth,low_valid=low_valid,compact=compact)
    h,w=ids.shape
    if type(x) is not int or type(y) is not int or not(0<=x<w and 0<=y<h):raise ValueError("pixel outside full extent")
    identity=int(ids[y,x]);z=depth[y,x]
    if identity==0 or valid[y,x]==0 or not np.isfinite(z) or z<=0:raise ValueError("valid finite foreground receiver required")
    own=attributes[y,x] if compact else guide[y,x]
    if not np.isfinite(own).all() or np.any(own[:3]<0):raise ValueError("invalid own guide")
    own_code=code(own[3]);normal=decode_normal(own_code)
    if compact and own_code>NORMAL:raise ValueError("full attributes must carry only Ns")
    base_x=math.floor((x-2)/4+.5)-2;base_y=math.floor((y-2)/4+.5)-2
    candidates=[];total=F(0);summed=np.zeros(3,np.float32);uncertain=[]
    for cy in range(max(base_y,0),min(base_y+4,(h-1)//4)+1):
        for cx in range(max(base_x,0),min(base_x+4,(w-1)//4)+1):
            ax,ay=cx*4+2,cy*4+2
            if not compact:ax,ay=min(ax,w-1),min(ay,h-1)
            c=dict(cell=[cx,cy],anchor=[ax,ay],reason=None,gates={});candidates.append(c)
            if ax>=w or ay>=h:c["reason"]="anchor_outside_full_extent";continue
            if compact:
                if not low_valid[cy,cx]:c["reason"]="low_coverage";continue
                if int(low_ids[cy,cx])!=identity:c["reason"]="low_identity";continue
            if not valid[ay,ax]:c["reason"]="full_anchor_coverage";continue
            if int(ids[ay,ax])!=identity:c["reason"]="full_anchor_identity";continue
            hit=guide[cy,cx] if compact else guide[ay,ax]
            if not np.isfinite(hit).all() or np.any(hit[:3]<0):raise ValueError("invalid candidate guide")
            hit_code=code(hit[3]);c["packed_code"]=hit_code;c["incoming_rgb"]=hit[:3].astype(float).tolist()
            if not hit_code&TRACED:c["reason"]="not_traced";continue
            candidate_n=decode_normal(hit_code)
            low_z=low_depth[cy,cx] if compact else depth[ay,ax]
            gates=[]
            if compact:
                gates.extend((("low_full_depth",depth_gate(low_z,depth[ay,ax])),
                    ("low_full_normal",normal_gate(candidate_n,decode_normal(attributes[ay,ax,3])))))
            gates.extend((("receiver_depth",depth_gate(z,low_z)),("receiver_normal",normal_gate(normal,candidate_n))))
            for name,(passed,slack,near) in gates:
                c["gates"][name]=dict(passed=passed,slack=slack,boundary_uncertain=near)
                if near:uncertain.append(dict(cell=[cx,cy],gate=name))
                if not passed:c["reason"]=name;break
            if c["reason"] is not None:continue
            distance=np.array([ax-x,ay-y],np.float32)*F(.25)
            weight=F(F(np.exp(F(-F(np.dot(distance,distance))/F(4.5))))*F(1/32))
            summed=np.asarray(summed+hit[:3]*weight,np.float32);total=F(total+weight)
            c.update(reason="accepted",weight=float(weight),distance_pixels=float(np.hypot(ax-x,ay-y)))
    accepted=[c for c in candidates if c["reason"]=="accepted"]
    incoming=summed/total if total>0 else own[:3]
    for c in accepted:c["normalized_weight"]=float(F(F(c["weight"])/total))
    return dict(pixel=[x,y],instance_id=identity,depth_m=float(z),normal=normal.astype(float).tolist(),
        classification="supported_reconstruction" if accepted else "geometry_fallback",
        accepted_count=len(accepted),weight_sum=float(total),
        own_value_scope="unoccluded SH fallback" if compact or not(own_code&TRACED) else "own real traced anchor, normally retained in support",
        own_incoming=own[:3].astype(float).tolist(),incoming_reconstructed_cpu=incoming.astype(float).tolist(),
        boundary_uncertain=bool(uncertain),boundary_comparisons=uncertain,
        reason_counts=dict(Counter(c["reason"] for c in candidates)),candidates=candidates,
        scope="source-gate replay on actual GPU planes; FP32 nominal, not a per-pixel GPU branch attachment")


def display_linear(rgb):
    x=np.asarray(rgb,np.float64)/255
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def display_error(rgb,reference):
    if rgb.shape!=reference.shape or rgb.dtype!=np.uint8 or reference.dtype!=np.uint8 or rgb.shape[-1]!=3:
        raise ValueError("matching actual RGB8 required")
    if not rgb.size:return None
    a=display_linear(rgb).reshape(-1,3);b=display_linear(reference).reshape(-1,3);d=a-b
    bias=d.mean(axis=0)
    return dict(pixels=len(d),rmse=float(np.sqrt(np.mean(d*d))),signed_rgb=bias.tolist(),
        signed_luminance=float(bias@np.array([.2126,.7152,.0722])),
        maximum_absolute_error=float(np.max(np.abs(d))),
        space="inverse-sRGB of published RGB8 (display-linear), not scene radiance")


def reveal_mask(before,after,*,blocker):
    old=before["depth"];z=after["depth"]
    return ((before["ids"]==blocker)&(before["valid"]!=0)&np.isfinite(old)&(old>0)&
        (after["ids"]!=0)&(after["ids"]!=blocker)&(after["valid"]!=0)&np.isfinite(z)&(z>old))
