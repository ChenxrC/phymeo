"""FP64 offline temporal incoming reference, NOT a renderer or performance tier.

Only actual traced incoming is eligible. Current fallback values are copied but
never acquire GI history. Conservative mode resets on any object transform change;
primary-only mode is exploratory and cannot prove secondary-shadow validity.
"""
from copy import deepcopy
import math
import numpy as np

STATUS={name:i for i,name in enumerate(("not-traced","reset","accepted","outside","identity",
    "depth","normal","domain","roundtrip","clamp-support","unvisited"))}


def require(value,message):
    if not value:raise ValueError(message)


def decode_guide(raw):
    raw=np.asarray(raw)
    require(raw.ndim==3 and raw.shape[2]==4 and np.isfinite(raw).all(),"invalid actual RGBA32F guide")
    code=np.rint(raw[...,3])
    require(np.all(code==raw[...,3]) and np.all((code>=0)&(code<(1<<24))),"guide alpha is not an exact uint24 numeric code")
    code=code.astype(np.uint32)
    xy=np.stack((code&1023,(code>>10)&1023),axis=-1).astype(np.float64)/1023*2-1
    normal=np.concatenate((xy,1-np.abs(xy).sum(axis=-1,keepdims=True)),axis=-1)
    lower=normal[...,2]<0
    sign=np.where(xy>=0,1.,-1.)
    unfolded=(1-np.abs(xy[...,::-1]))*sign
    normal[...,:2]=np.where(lower[...,None],unfolded,xy)
    normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
    return dict(incoming=raw[...,:3].astype(np.float64),normal_world=normal,
                traced=(code&(1<<20))!=0,outcome=(code>>21)&7,packed_code=code)


def rigid(value,label):
    value=np.array(value,dtype=np.float64,copy=True)
    require(value.shape==(4,4) and np.isfinite(value).all(),label+" matrix must be finite4x4")
    require(np.max(np.abs(value[3]-[0,0,0,1]))<=1e-10 and
        np.max(np.abs(value[:3,:3].T@value[:3,:3]-np.eye(3)))<=1e-7 and
        abs(np.linalg.det(value[:3,:3])-1)<=1e-7,label+" matrix must be rigid/right handed")
    return value


def inverse(matrix):
    result=np.eye(4);result[:3,:3]=matrix[:3,:3].T
    result[:3,3]=-result[:3,:3]@matrix[:3,3]
    return result


def validate(value):
    p={k:deepcopy(value[k]) for k in ("camera","key","time_s","lighting_token","material_token","signal_token")}
    if "sequence" in value:
        sequence=deepcopy(value["sequence"])
        require(isinstance(sequence,dict) and type(sequence.get("index")) is int and sequence["index"]>=0
                and isinstance(sequence.get("source_sha256"),str) and len(sequence["source_sha256"])==64,
                "invalid bound presentation sequence")
        require((sequence["index"]==0 and sequence.get("previous_key") is None) or
                (sequence["index"]>0 and isinstance(sequence.get("previous_key"),dict)),
                "presentation predecessor identity missing")
        p["sequence"]=sequence
    h,w=p["camera"]["height"],p["camera"]["width"]
    require(type(h) is int and type(w) is int and h>0 and w>0,"invalid camera extent")
    for field in ("fx_px","fy_px","cx_px","cy_px"):
        require(math.isfinite(p["camera"][field]),"nonfinite camera intrinsics")
    require(p["camera"]["fx_px"]>0 and p["camera"]["fy_px"]>0,"nonpositive focal length")
    require(math.isfinite(p["time_s"]),"nonfinite input time")
    for field in ("epoch","tick","revision"):
        require(type(p["key"][field]) is int and p["key"][field]>=0,"invalid source identity")
    for field in ("scene_key","world_id","phase"):
        require(isinstance(p["key"][field],str) and p["key"][field],"missing source identity")
    for field in ("lighting_token","material_token","signal_token"):
        require(isinstance(p[field],str) and p[field],"missing immutable scope token")
    for field,shape,dtype in (("incoming",(h,w,3),np.float64),("normal_world",(h,w,3),np.float64),
            ("depth",(h,w),np.float64),("instance_id",(h,w),np.uint32),("valid",(h,w),np.uint8),("traced",(h,w),bool)):
        array=np.asarray(value[field]);require(array.shape==shape,"bad packet shape: "+field)
        require((np.isfinite(array)|(np.isposinf(array) if field=="depth" else False)).all(),"nonfinite packet: "+field)
        if field=="instance_id":require(np.all((array>=0)&(array<=2**32-1)&(array==np.floor(array))),"invalid instance ID")
        if field in ("valid","traced"):require(np.all((array==0)|(array==1)),"invalid binary mask")
        p[field]=np.array(array,dtype=dtype,copy=True)
    require(np.all(p["incoming"]>=0),"negative incoming radiance")
    hit=(p["valid"]==1)
    require(np.array_equal(hit,p["instance_id"]>0) and
            np.all(np.isfinite(p["depth"][hit])&(p["depth"][hit]>0)) and
            np.all(np.isposinf(p["depth"][~hit])),"public depth/ID/valid hit-miss identity differs")
    positive=hit&(p["instance_id"]>0)&np.isfinite(p["depth"])&(p["depth"]>0)
    require(np.all(np.abs(np.linalg.norm(p["normal_world"][positive],axis=-1)-1)<=1e-5),"actual valid shading normals must be unit")
    p["camera_matrix"]=rigid(value["camera_matrix"],"camera")
    p["instance_matrices"]={int(k):rigid(v,"instance") for k,v in value["instance_matrices"].items()}
    require(set(map(int,np.unique(p["instance_id"][positive])))<=set(p["instance_matrices"]),"visible identity has no bound rigid transform")
    p["eligible"]=positive&p["traced"]
    return p


def clamp_bounds(p,radius,depth_limit,normal_dot):
    h,w=p["depth"].shape;lower=np.full((h,w,3),np.inf);upper=np.full((h,w,3),-np.inf)
    support=np.zeros((h,w),dtype=np.uint16)
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            ya,yb=max(0,-dy),min(h,h-dy);xa,xb=max(0,-dx),min(w,w-dx)
            a=np.s_[ya:yb,xa:xb];b=np.s_[ya+dy:yb+dy,xa+dx:xb+dx]
            good=p["eligible"][a]&p["eligible"][b]&(p["instance_id"][a]==p["instance_id"][b])
            da=np.where(p["eligible"][a],p["depth"][a],0.)
            db=np.where(p["eligible"][b],p["depth"][b],0.)
            good&=np.abs(da-db)<=depth_limit*np.maximum(da,db)
            good&=np.sum(p["normal_world"][a]*p["normal_world"][b],axis=-1)>=normal_dot
            lower[a]=np.minimum(lower[a],np.where(good[...,None],p["incoming"][b],np.inf))
            upper[a]=np.maximum(upper[a],np.where(good[...,None],p["incoming"][b],-np.inf))
            support[a]+=good
    return lower,upper,support


class TemporalIncoming:
    def __init__(self,*,clamp_radius=2,max_age=8,relative_depth=.003,normal_dot=.995,
                 roundtrip_pixels=.75,min_support=3,scene_motion_policy="reset-any"):
        require(type(clamp_radius) is int and 1<=clamp_radius<=8,"clamp radius outside1..8")
        require(type(max_age) is int and 2<=max_age<=64,"history age outside2..64")
        require(0<relative_depth<=.01 and 0<normal_dot<=1 and 0<roundtrip_pixels<=1.,"invalid geometry gates")
        require(type(min_support) is int and min_support>=1,"invalid clamp support")
        require(scene_motion_policy in ("reset-any","primary-only"),"unknown scene motion policy")
        self.radius,self.max_age=clamp_radius,max_age
        self.depth_limit,self.normal_dot,self.roundtrip,self.min_support=relative_depth,normal_dot,roundtrip_pixels,min_support
        self.policy=scene_motion_policy;self.previous=None;self.color=None;self.age=None

    def update(self,value):
        p=validate(value) # Fully validate before replacing any accepted history.
        old=self.previous;h,w=p["depth"].shape
        reasons=[];transforms_changed=False
        if old is None:reasons=["first_frame"]
        else:
            for field in ("scene_key","world_id","epoch","phase"):
                if p["key"][field]!=old["key"][field]:reasons.append(field)
            for field in ("lighting_token","material_token","signal_token"):
                if p[field]!=old[field]:reasons.append(field)
            if p["depth"].shape!=old["depth"].shape:reasons.append("extent")
            if p["camera"]["id"]!=old["camera"]["id"]:reasons.append("camera_id")
            if any(p["camera"].get(k)!=old["camera"].get(k) for k in
                   ("fx_px","fy_px","cx_px","cy_px","near_m","far_m")):reasons.append("camera_lens")
            if "sequence" in p or "sequence" in old:
                current_sequence,old_sequence=p.get("sequence"),old.get("sequence")
                if current_sequence is None or old_sequence is None:reasons.append("sequence_scope")
                else:
                    if current_sequence["source_sha256"]!=old_sequence["source_sha256"]:reasons.append("sequence_source")
                    if (current_sequence["index"]!=old_sequence["index"]+1 or
                        current_sequence["previous_key"]!=old["key"]):reasons.append("discontinuous_sequence")
                    if p["key"]["tick"]<=old["key"]["tick"]:reasons.append("nonincreasing_source_tick")
            elif p["key"]["tick"]!=old["key"]["tick"]+1:reasons.append("discontinuous_tick")
            if not 0<p["time_s"]-old["time_s"]<=.1:reasons.append("time_gap")
            transforms_changed=set(p["instance_matrices"])!=set(old["instance_matrices"]) or any(
                not np.array_equal(p["instance_matrices"][i],old["instance_matrices"][i])
                for i in set(p["instance_matrices"])&set(old["instance_matrices"]))
            if transforms_changed and self.policy=="reset-any":reasons.append("scene_transforms")
        color=p["incoming"].copy();age=p["eligible"].astype(np.uint16)
        status=np.full((h,w),STATUS["not-traced"],dtype=np.uint8)
        status[p["eligible"]]=STATUS["reset" if reasons else "unvisited"]
        oldx=np.full((h,w),-1,dtype=np.int32);oldy=oldx.copy();clipped=0
        if not reasons:
            lower,upper,support=clamp_bounds(p,self.radius,self.depth_limit,self.normal_dot)
            cc,pc=p["camera"],old["camera"]
            for identity in np.unique(p["instance_id"][p["eligible"]]):
                yy,xx=np.nonzero(p["eligible"]&(p["instance_id"]==identity))
                if int(identity) not in old["instance_matrices"]:
                    status[yy,xx]=STATUS["identity"];continue
                cw,pw=p["instance_matrices"][int(identity)],old["instance_matrices"][int(identity)]
                depth=p["depth"][yy,xx]
                rays=np.stack(((xx+.5-cc["cx_px"])/cc["fx_px"],-(yy+.5-cc["cy_px"])/cc["fy_px"],-np.ones(len(xx))))
                points=inverse(old["camera_matrix"])@pw@inverse(cw)@p["camera_matrix"]@np.vstack((rays*depth,np.ones(len(xx))))
                expected=-points[2];good=np.isfinite(expected)&(expected>0)
                px=np.full(len(xx),-1,dtype=int);py=px.copy()
                px[good]=np.floor(pc["fx_px"]*points[0,good]/expected[good]+pc["cx_px"]).astype(int)
                py[good]=np.floor(-pc["fy_px"]*points[1,good]/expected[good]+pc["cy_px"]).astype(int)
                good&=(px>=0)&(px<w)&(py>=0)&(py<h)
                status[yy,xx]=STATUS["outside"]
                selected=np.nonzero(good)[0]
                def keep(indices,condition,reason):
                    status[yy[indices],xx[indices]]=STATUS[reason]
                    return indices[condition]
                if not len(selected):continue
                oldx[yy[selected],xx[selected]]=px[selected];oldy[yy[selected],xx[selected]]=py[selected]
                selected=keep(selected,old["instance_id"][py[selected],px[selected]]==identity,"identity")
                selected=keep(selected,old["eligible"][py[selected],px[selected]]&(self.age[py[selected],px[selected]]>0),"domain")
                measured=old["depth"][py[selected],px[selected]]
                selected=keep(selected,np.abs(measured-expected[selected])<=self.depth_limit*np.maximum(measured,expected[selected]),"depth")
                rotated=old["normal_world"][py[selected],px[selected]]@(cw[:3,:3]@pw[:3,:3].T).T
                selected=keep(selected,np.sum(rotated*p["normal_world"][yy[selected],xx[selected]],axis=1)>=self.normal_dot,"normal")
                if not len(selected):continue
                measured=old["depth"][py[selected],px[selected]]
                oldrays=np.stack(((px[selected]+.5-pc["cx_px"])/pc["fx_px"],-(py[selected]+.5-pc["cy_px"])/pc["fy_px"],-np.ones(len(selected))))
                back=inverse(p["camera_matrix"])@cw@inverse(pw)@old["camera_matrix"]@np.vstack((oldrays*measured,np.ones(len(selected))))
                z=-back[2];safe=np.where(z>0,z,1.)
                u=cc["fx_px"]*back[0]/safe+cc["cx_px"];v=-cc["fy_px"]*back[1]/safe+cc["cy_px"]
                distance=np.sqrt((u-xx[selected]-.5)**2+(v-yy[selected]-.5)**2)
                good=(z>0)&np.isfinite(distance)&(distance<=self.roundtrip)&(
                    np.abs(z-depth[selected])<=self.depth_limit*np.maximum(z,depth[selected]))
                selected=keep(selected,good,"roundtrip")
                selected=keep(selected,support[yy[selected],xx[selected]]>=self.min_support,"clamp-support")
                if not len(selected):continue
                hist=self.color[py[selected],px[selected]]
                bounded=np.clip(hist,lower[yy[selected],xx[selected]],upper[yy[selected],xx[selected]])
                clipped+=int(np.any(hist!=bounded,axis=1).sum())
                n=np.minimum(self.age[py[selected],px[selected]],self.max_age-1).astype(np.float64)+1
                current=p["incoming"][yy[selected],xx[selected]]
                blended=bounded*(1-1/n[:,None])+current/n[:,None]
                color[yy[selected],xx[selected]]=np.where(np.all(bounded==current,axis=1)[:,None],current,blended)
                age[yy[selected],xx[selected]]=n.astype(np.uint16);status[yy[selected],xx[selected]]=STATUS["accepted"]
        diagnostics=dict(reset_reasons=reasons,previous_tick=None if old is None else old["key"]["tick"],
            current_tick=p["key"]["tick"],source_tick_delta=None if old is None else p["key"]["tick"]-old["key"]["tick"],
            adjacency_scope="bound presented-frame predecessor" if "sequence" in p else "unit source tick",
            sequence=deepcopy(p.get("sequence")),accepted_history_pixels=int((status==STATUS["accepted"]).sum()),
            traced_input_pixels=int(p["eligible"].sum()),clamped_history_pixels=clipped,
            status_counts={name:int((status==code).sum()) for name,code in STATUS.items()},
            max_retained_observation_age=int(age.max()),scene_motion_policy=self.policy,
            unvalidated_secondary_visibility=self.policy=="primary-only" and transforms_changed,
            algorithm="CPU FP64 nearest rigid reprojection + current same-ID normal/depth neighborhood clamp",
            meaning="age counts retained observations, not independent samples; display/direct/specular/sensors are not modified",
            clamp_radius=self.radius,min_support=self.min_support,relative_depth=self.depth_limit,
            normal_dot=self.normal_dot,roundtrip_pixels=self.roundtrip,
            performance_claim=False,gpu_execution=False)
        # Own every accepted input/output; caller mutation cannot alter history.
        self.previous=p;self.color=color.copy();self.age=age.copy()
        return dict(incoming=color,age=age,status=status,previous_pixel_x=oldx,previous_pixel_y=oldy,diagnostics=diagnostics)