"""Bounded real GPU history/control replay. Executed only by the serial owner."""
import argparse
from dataclasses import asdict,replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

from .runtime import HistoryBackend,activate


def digest(data):return sha256(data).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def rejected_witness(previous,current,previous_camera,camera):
    """Independent FP64 primary-visibility witness; excludes quantization edges.

    This checks only definite outside/identity/depth mismatches. It does not
    reenact the shader's normal gate, neighborhood clamp or history filtering.
    """
    import numpy as np
    from phymeo_rendering.outputs import Channel
    def rotation(pose):
        x,y,z,w=pose.orientation_xyzw
        return np.array(((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
                         (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
                         (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y))))
    height,width=camera.height,camera.width
    z=np.frombuffer(current["planes"][Channel.FORWARD_DEPTH],"<f4").reshape(height,width).astype(float)
    ids=np.frombuffer(current["planes"][Channel.INSTANCE_ID],"<u4").reshape(height,width)
    old_ids=np.frombuffer(previous["planes"][Channel.INSTANCE_ID],"<u4").reshape(height,width)
    old_z=np.frombuffer(previous["planes"][Channel.FORWARD_DEPTH],"<f4").reshape(height,width).astype(float)
    incoming=np.frombuffer(current["current"],"<f4").reshape(height,width,4)
    eligible=incoming[...,3]==1
    yy,xx=np.nonzero(eligible)
    rays=np.stack(((xx+.5-camera.cx_px)/camera.fx_px,-(yy+.5-camera.cy_px)/camera.fy_px,-np.ones(len(xx))),axis=1)
    world=(rays*z[yy,xx,None])@rotation(camera.world_pose).T+camera.world_pose.position_m
    old=(world-previous_camera.world_pose.position_m)@rotation(previous_camera.world_pose)
    predicted=-old[:,2]
    u=previous_camera.fx_px*old[:,0]/predicted+previous_camera.cx_px
    v=-previous_camera.fy_px*old[:,1]/predicted+previous_camera.cy_px
    finite=np.isfinite(u)&np.isfinite(v)&(predicted>0)
    # Never round or classify a sample within .002 pixels of a raster-cell edge.
    safe=finite&(np.abs(u-np.rint(u))>.002)&(np.abs(v-np.rint(v))>.002)
    inside=safe&(u>=0)&(u<width)&(v>=0)&(v<height)
    outside=finite&((u<-.002)|(u>width+.002)|(v<-.002)|(v>height+.002))
    witness=outside.copy()
    selected=np.nonzero(inside)[0]
    qx=np.floor(u[selected]).astype(int);qy=np.floor(v[selected]).astype(int)
    different=old_ids[qy,qx]!=ids[yy[selected],xx[selected]]
    expected=predicted[selected];saved=old_z[qy,qx]
    definitely_wrong=np.isfinite(saved)&(np.abs(expected-saved)>.003*np.minimum(expected,saved)+1e-5)
    witness[selected]=different|definitely_wrong
    dtype=np.dtype([("incoming","<f4",4),("geometry","<u4",4)])
    ages=np.frombuffer(current["history"],dtype=dtype).reshape(height,width)["incoming"][...,3]
    bad=int(np.count_nonzero(ages[yy[witness],xx[witness]]!=1))
    if bad:raise ValueError(f"{bad} independently exposed pixels retained stale history")
    return dict(definite_disocclusion_pixels=int(witness.sum()),retained_history_violations=bad,
                exclusion=".002 pixel cell-edge margin and 10um depth-bound margin; only independent rejection witnesses")


def inputs(scene,frame,camera):
    """Static repeated exposures, real camera motion, then explicit invalidations."""
    steps=[]
    for n in range(8):steps.append(("static-"+str(n),frame,camera,None,False))
    for tick,offset in ((1,.02),(2,.08)):
        key=replace(frame.key,tick=frame.key.tick+tick,revision=frame.key.revision+tick)
        f=replace(frame,key=key,time_s=frame.time_s+tick*.01)
        pos=camera.world_pose.position_m
        c=replace(camera,frame_key=key,world_pose=replace(camera.world_pose,position_m=(pos[0]+offset,pos[1],pos[2])))
        steps.append(("camera-"+str(tick),f,c,None,False))
    key=replace(frame.key,tick=frame.key.tick+3,revision=frame.key.revision+3)
    node=frame.nodes[0];pos=node.world_pose.position_m
    moved=replace(node,world_pose=replace(node.world_pose,position_m=(pos[0]+.001,pos[1],pos[2])))
    f=replace(frame,key=key,time_s=frame.time_s+.03,nodes=(moved,*frame.nodes[1:]))
    c=replace(camera,frame_key=key)
    steps.append(("node-reset",f,c,None,False))
    from phymeo_rendering.scene import DirectionalLight
    key=replace(key,tick=key.tick+1,revision=key.revision+1)
    f=replace(f,key=key,time_s=frame.time_s+.04);c=replace(c,frame_key=key)
    steps.append(("light-reset",f,c,(DirectionalLight((0.,0.,-1.),(.1,.1,.1)),),False))
    steps.append(("explicit-reset",f,c,None,True))
    steps.append(("after-sensor-resize",f,c,None,False))
    return steps


def run(args):
    root,manifest=activate(args.prepared)
    import numpy as np
    from PIL import Image
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest,RenderPath
    from preview_compact.run_smoke import cases
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    report=dict(schema="phymeo-preview-compact-history-smoke-v1",status="started",runs={},checks={},
        source_manifest_sha256=digest((root/"source-manifest.json").read_bytes()),
        script_sha256=digest(Path(__file__).read_bytes()),scope="static geometry moving camera; node changes deliberately reset all",
        warning="This bounded correctness test is not a dynamic Panda noise or performance acceptance.")
    dump(output/"report.json",report)
    recorded={}
    try:
        fixtures=cases(output,args.scene=="panda-60")
        name,scene,frame,camera,studio,exposure=next(value for value in fixtures if value[0]==args.scene)
        steps=inputs(scene,frame,camera)
        dump(output/"inputs.json",dict(scene=asdict(scene),studio=asdict(studio),steps=[dict(name=n,frame=asdict(f),camera=asdict(c),
            lights=None if lights is None else [asdict(v) for v in lights],reset=reset) for n,f,c,lights,reset in steps]))
        for label,reuse in (("fresh-sequence",False),("history",True)):
            folder=output/label;folder.mkdir()
            owner=HistoryBackend(root,device_index=args.device_index,studio=studio,seed=args.seed,exposure=exposure,
                reuse_history=reuse,validation=args.validation).open_scene(scene)
            rows=[];report["runs"][label]=dict(captures=rows)
            snapshots=[];recorded[label]=snapshots
            try:
                for index,(name,f,c,lights,reset) in enumerate(steps):
                    if lights is not None:owner.set_directional_lights(lights)
                    if reset:owner.reset_history()
                    if name=="after-sensor-resize":
                        before=owner.diagnostics()["history"]["actual"]["committed_rgb"]
                        sensor_camera=replace(c,width=c.width+2,height=c.height+2)
                        owner.capture(f,CaptureRequest(sensor_camera,RenderPath.RASTER,(Channel.INSTANCE_ID,)))
                        if owner.diagnostics()["history"]["actual"]["committed_rgb"]!=before:
                            raise ValueError("sensor-only capture committed history")
                        try:owner.read_history_incoming()
                        except RuntimeError:pass
                        else:raise ValueError("sensor-only exposed stale history read")
                    if index==4:
                        # Invalid public frame binding must neither launch nor
                        # consume the next successful sample sequence.
                        bad=replace(c,frame_key=replace(c.frame_key,revision=c.frame_key.revision+1000))
                        before=owner.diagnostics()["history"]["actual"]["committed_rgb"]
                        try:owner.capture(f,CaptureRequest(bad,RenderPath.RASTER,(Channel.RGB,)))
                        except (ValueError,RuntimeError):pass
                        else:raise ValueError("mismatched public exposure was accepted")
                        if owner.diagnostics()["history"]["actual"]["committed_rgb"]!=before:
                            raise ValueError("rejected exposure mutated published history")
                    request=CaptureRequest(c,RenderPath.RASTER,tuple(Channel))
                    start=perf_counter();capture=owner.capture(f,request);duration=1000*(perf_counter()-start)
                    if (capture.frame_key,capture.time_s,capture.camera)!=(f.key,f.time_s,c):raise ValueError("stale exposure publication")
                    planes={v.channel:v.data for v in capture.planes}
                    current=owner.read_current_incoming();history=owner.read_history_incoming();linear=owner.read_reconstructed_linear()
                    now=np.frombuffer(current["data"],"<f4").reshape(c.height,c.width,4)
                    cells=np.frombuffer(history["data"],dtype=np.dtype([("incoming","<f4",4),("geometry","<u4",4)])).reshape(c.height,c.width)
                    if not np.isfinite(now).all() or not np.isfinite(cells["incoming"]).all():raise ValueError("nonfinite history data")
                    ages=cells["incoming"][...,3];supported=now[...,3]==1
                    if np.any(ages[~supported]) or np.any(ages[supported]<1) or np.any(ages>8):raise ValueError("unsupported/invalid history age")
                    diag=owner.diagnostics();actual=diag["history"]["actual"]
                    if actual["committed_rgb"]!=index+1 or actual["eligible_pixels"]!=int(supported.sum()):raise ValueError("history accounting differs")
                    partition=sum(actual[k] for k in ("accepted_pixels","rejected_projection","rejected_geometry","rejected_support","reset_pixels"))
                    if partition!=actual["eligible_pixels"] or actual["age_sum"]!=int(ages.sum()):raise ValueError("history rejection/age accounting differs")
                    for suffix,data in (("current.f32",current["data"]),("history.bin",history["data"]),("linear.f32",linear["data"]),
                                        *((ch.value,raw) for ch,raw in planes.items())):
                        (folder/(f"{index:02d}-{name}."+suffix)).write_bytes(data)
                    Image.frombytes("RGB",(c.width,c.height),planes[Channel.RGB]).save(folder/f"{index:02d}-{name}.png")
                    row=dict(name=name,capture_ms=duration,diagnostics=diag,
                        planes={ch.value:digest(raw) for ch,raw in planes.items()},
                        current_sha256=digest(current["data"]),history_sha256=digest(history["data"]),linear_sha256=digest(linear["data"]))
                    rows.append(row);snapshots.append(dict(planes=planes,current=current["data"],history=history["data"],linear=linear["data"]))
                    dump(output/"report.json",report)
            finally:
                original=sys.exc_info()[0] is not None
                try:
                    owner.close();closed=owner.diagnostics();report["runs"][label]["closed"]=closed
                    device=closed["device"]
                    if device.get("closed") is not True or any(device.get(k,1) for k in ("live_contexts","live_frames","live_resources","validation_errors")):
                        raise ValueError("history resource/validation close failed")
                except BaseException as error:
                    report["runs"][label]["close_error"]=str(error)
                    if not original:raise
        fresh=recorded["fresh-sequence"];temporal=recorded["history"]
        for index,(a,b) in enumerate(zip(fresh,temporal)):
            if a["current"]!=b["current"]:raise ValueError("history changed current indirect estimator/seed")
            for channel in Channel:
                if channel is not Channel.RGB and a["planes"][channel]!=b["planes"][channel]:
                    raise ValueError("history changed primary sensor")
            actual=report["runs"]["history"]["captures"][index]["diagnostics"]["history"]["actual"]
            if actual["last_reset"] and (a["planes"]!=b["planes"] or a["linear"]!=b["linear"]):
                raise ValueError("reset did not reproduce same-sequence current exposure")
        if len({s["current"] for s in fresh[:8]})<2:raise ValueError("static current samples did not actually vary")
        if not any(row["diagnostics"]["history"]["actual"]["accepted_pixels"] for row in report["runs"]["history"]["captures"][1:8]):
            raise ValueError("static history never accepted a pixel")
        report["camera_visibility_witnesses"]=[rejected_witness(temporal[i-1],temporal[i],steps[i-1][2],steps[i][2]) for i in (8,9)]
        report["camera_visibility_witness_scope"]="reported zero witnesses means no disocclusion evidence in that view, not a passed disocclusion test"
        report["checks"]=dict(current_estimator_same_bytes=True,sensors_same_bytes=True,reset_same_bytes=True,
            actual_seed_variation=True,static_history_accepted=True,first_and_reset_no_temporal_mix=True,
            sensor_resize_and_invalid_publication_do_not_commit=True)
        report["status"]="complete"
    except BaseException as error:
        report["status"]="partial_failure";report["error"]=dict(type=type(error).__name__,message=str(error));raise
    finally:dump(output/"report.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--scene",choices=("room","odd","thin","panda-60"),default="odd")
    parser.add_argument("--device-index",type=int,default=1)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--validation",action="store_true")
    run(parser.parse_args())


if __name__=="__main__":main()
