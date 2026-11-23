"""Bounded real GPU rigid-history/control test; run by the serial device owner.

The two owners receive identical geometry, presentation records and actual seed
sequences. Only temporal reuse differs. This is correctness/observability
evidence, not a noise-quality or 30 FPS acceptance.
"""
import argparse
from dataclasses import asdict,replace
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

from .runtime import RigidHistoryBackend,activate
from .state import bind_presentations,frame_digest


def digest(data):return sha256(data).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def synthetic_inputs(output):
    from preview_compact.run_smoke import cases
    from phymeo_rendering.scene import DirectionalLight,Pose
    _,scene,original,camera,studio,exposure=next(v for v in cases(output,False) if v[0]=="thin")
    steps=[]
    for i in range(12):
        # Synthetic authored presentation clock deliberately uses the same
        # non-unit tick stride, but is never labelled as the Panda recording.
        tick=round((i+1)*1600/30)
        key=replace(original.key,tick=tick,revision=tick-1)
        nodes=list(original.nodes)
        if i>=2:
            back_x=min(i-1,5)*.008
            baffle_x=(.04,.08,.10,-.08,-.12,-.10,.30,.30,.30,.30)[i-2]
            angle=math.radians(3 if i>=4 else 0)
            for j,node in enumerate(nodes):
                if node.node_id=="back":nodes[j]=replace(node,world_pose=Pose((back_x,0.,0.)))
                if node.node_id=="baffle":nodes[j]=replace(node,world_pose=Pose((baffle_x,0.,0.),(0.,math.sin(angle/2),0.,math.cos(angle/2))))
                if node.node_id=="thin" and i>=6:nodes[j]=replace(node,world_pose=Pose((.06,0.,0.)))
        frame=replace(original,key=key,time_s=tick/1600,nodes=tuple(nodes))
        position=camera.world_pose.position_m
        cam=replace(camera,frame_key=key,world_pose=replace(camera.world_pose,
            position_m=(position[0]+(.015 if i>=4 else 0.),position[1],position[2])))
        lights=(DirectionalLight((0.,0.,-1.),(.15,.15,.15)),) if i==9 else None
        steps.append(dict(name=f"motion-{i:02d}",frame=frame,camera=cam,lights=lights,reset=i==10,sensor_resize=i==11))
    source=frame_digest(dict(scope="synthetic finite room, moving wall/baffle/thin strip and camera",
                             scene=asdict(scene),frames=[asdict(s["frame"]) for s in steps]))
    return scene,studio,exposure,steps,dict(kind="synthetic-authoring",source_sha256=source,
        teleport_step=8,teleport_instance=8,recorded_panda=False)


def panda_inputs(args,output):
    if args.inputs is None:raise ValueError("Panda requires the previously saved full --inputs binding")
    import panda_render_performance as factory
    raw=args.inputs.read_bytes();saved=json.loads(raw)
    scenes,frames,studio,trace=factory.bind_scene(output/"panda-geometry")
    scene=scenes[1]
    if digest(trace.read_bytes())!=saved["trace_sha256"]:raise ValueError("recorded source trace SHA differs")
    records={v["pose_index"]:v for v in saved["actual_full_inputs"]}
    indices=range(args.start,args.start+args.steps)
    if args.steps<2 or args.steps>32:raise ValueError("bounded Panda smoke needs 2..32 presentations")
    if any(i not in records for i in indices):raise ValueError("saved inputs omit requested recorded presentation")
    steps=[]
    for i in indices:
        frame=replace(frames[i],key=replace(frames[i].key,scene_key=scene.key))
        if frame_digest(frame)!=frame_digest(records[i]["frame"]):
            raise ValueError(f"reconstructed full recorded frame differs at index {i}")
        original_camera=factory.film.exposure_camera(SimpleNamespace(camera="detail"),frame,i,len(frames),1280,720)
        if frame_digest(original_camera)!=frame_digest(records[i]["camera"]):
            raise ValueError(f"recorded camera differs at index {i}")
        scale=args.width/1280
        if args.height/720!=scale:raise ValueError("Panda reduced extent must preserve the recorded aspect ratio")
        camera=replace(original_camera,width=args.width,height=args.height,fx_px=original_camera.fx_px*scale,
            fy_px=original_camera.fy_px*scale,cx_px=original_camera.cx_px*scale,cy_px=original_camera.cy_px*scale)
        steps.append(dict(name=f"recorded-{i:03d}",frame=frame,camera=camera,lights=None,reset=False,sensor_resize=False))
    return scene,studio,8.,steps,dict(kind="actual-recorded-Panda",source_sha256=saved["trace_sha256"],
        inputs_path=str(args.inputs.resolve()),inputs_sha256=digest(raw),trace_path=str(trace),indices=list(indices),
        source_keys=[records[i]["frame"]["key"] for i in indices],
        source_times=[records[i]["frame"]["time_s"] for i in indices],
        source_tick_deltas=[steps[i]["frame"].key.tick-steps[i-1]["frame"].key.tick for i in range(1,len(steps))],
        camera_scope="same recorded detail camera pose; explicit proportional test extent, not 720p performance",
        recorded_panda=True)


def pose_matrix(pose,np):
    x,y,z,w=pose.orientation_xyzw
    matrix=np.eye(4,dtype=float)
    matrix[:3,:3]=((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
                  (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
                  (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))
    matrix[:3,3]=pose.position_m
    return matrix


def matrices(scene,frame,np):
    nodes={v.node_id:pose_matrix(v.world_pose,np) for v in frame.nodes}
    return {v.id:nodes[v.node_id]@pose_matrix(v.local_pose,np) for v in scene.instances}


def disocclusion_witness(scene,previous,current,old_step,step):
    """FP64 rigid correspondence checks definite rejection, not shader acceptance.

    Quantization-near raster cell edges and near-threshold depths are excluded.
    This independent test does not reproduce normal/color gates or the filter.
    """
    import numpy as np
    from phymeo_rendering.outputs import Channel
    camera,old_camera=step["camera"],old_step["camera"]
    height,width=camera.height,camera.width
    z=np.frombuffer(current["planes"][Channel.FORWARD_DEPTH],"<f4").reshape(height,width).astype(float)
    ids=np.frombuffer(current["planes"][Channel.INSTANCE_ID],"<u4").reshape(height,width)
    old_ids=np.frombuffer(previous["planes"][Channel.INSTANCE_ID],"<u4").reshape(height,width)
    old_z=np.frombuffer(previous["planes"][Channel.FORWARD_DEPTH],"<f4").reshape(height,width).astype(float)
    now=np.frombuffer(current["current"],"<f4").reshape(height,width,4)
    yy,xx=np.nonzero(now[...,3]==1)
    rays=np.stack(((xx+.5-camera.cx_px)/camera.fx_px,-(yy+.5-camera.cy_px)/camera.fy_px,-np.ones(len(xx))),axis=1)
    cam=pose_matrix(camera.world_pose,np);old_cam=pose_matrix(old_camera.world_pose,np)
    world=(rays*z[yy,xx,None])@cam[:3,:3].T+cam[:3,3]
    new_instances=matrices(scene,step["frame"],np);old_instances=matrices(scene,old_step["frame"],np)
    old_world=world.copy()
    for identity in np.unique(ids[yy,xx]):
        selected=ids[yy,xx]==identity
        transform=old_instances[int(identity)]@np.linalg.inv(new_instances[int(identity)])
        old_world[selected]=world[selected]@transform[:3,:3].T+transform[:3,3]
    points=(old_world-old_cam[:3,3])@old_cam[:3,:3]
    predicted=-points[:,2]
    with np.errstate(divide="ignore",invalid="ignore"):
        u=old_camera.fx_px*points[:,0]/predicted+old_camera.cx_px
        v=-old_camera.fy_px*points[:,1]/predicted+old_camera.cy_px
    finite=np.isfinite(u)&np.isfinite(v)&(predicted>0)
    safe=finite&(np.abs(u-np.rint(u))>.002)&(np.abs(v-np.rint(v))>.002)
    inside=safe&(u>=0)&(u<width)&(v>=0)&(v<height)
    witness=finite&((u<-.002)|(u>width+.002)|(v<-.002)|(v>height+.002))
    selection=np.nonzero(inside)[0]
    qx=np.floor(u[selection]).astype(int);qy=np.floor(v[selection]).astype(int)
    expected,saved=predicted[selection],old_z[qy,qx]
    wrong_depth=np.isfinite(saved)&(np.abs(expected-saved)>.003*np.minimum(expected,saved)+1e-5)
    witness[selection]=(old_ids[qy,qx]!=ids[yy[selection],xx[selection]])|wrong_depth
    cells=np.frombuffer(current["history"],dtype=history_dtype(np)).reshape(height,width)
    bad=int(np.count_nonzero(cells["incoming"][yy[witness],xx[witness],3]!=1))
    if bad:raise ValueError(f"{bad} definite rigid-disocclusion pixels retained history")
    return dict(definite_disocclusion_pixels=int(witness.sum()),retained_history_violations=bad,
        exclusion=".002px raster-cell margin and 10um depth-bound margin; no acceptance oracle")


def history_dtype(np):
    return np.dtype([("incoming","<f4",4),("geometry","<u4",4),("lighting","<f4",4)])


def run(args):
    root,manifest=activate(args.prepared)
    import numpy as np
    from PIL import Image
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest,RenderPath
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    report=dict(schema="phymeo-preview-compact-rigid-history-smoke-v1",status="started",runs={},checks={},
        manifest_sha256=digest((root/"source-manifest.json").read_bytes()),script_sha256=digest(Path(__file__).read_bytes()),
        author_module_sha256={p.name:digest(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*")) if p.is_file()},
        limitation="Finite biased primary rigid history; no proof of unchanged hidden secondary visibility; no performance or quality acceptance")
    dump(output/"report.json",report)
    recorded={}
    try:
        scene,studio,exposure,steps,source=(panda_inputs(args,output) if args.case=="panda" else synthetic_inputs(output))
        bindings=bind_presentations([s["frame"] for s in steps],source["source_sha256"],
            start_index=args.start if args.case=="panda" else 0)
        report["source"]=source
        dump(output/"inputs.json",dict(scene=asdict(scene),studio=asdict(studio),source=source,
            steps=[dict(name=s["name"],frame=asdict(s["frame"]),camera=asdict(s["camera"]),presentation=b,
                        lights=None if s["lights"] is None else [asdict(v) for v in s["lights"]],reset=s["reset"],
                        sensor_resize=s["sensor_resize"]) for s,b in zip(steps,bindings)]))
        for label,reuse in (("fresh-sequence",False),("rigid-history",True)):
            folder=output/label;folder.mkdir();rows=[];snapshots=[];recorded[label]=snapshots
            report["runs"][label]=dict(captures=rows)
            owner=RigidHistoryBackend(root,device_index=args.device_index,studio=studio,seed=args.seed,
                exposure=exposure,reuse_history=reuse,validation=args.validation).open_scene(scene)
            try:
                for index,(step,binding) in enumerate(zip(steps,bindings)):
                    frame,camera=step["frame"],step["camera"]
                    if step["lights"] is not None:owner.set_directional_lights(step["lights"])
                    if step["reset"]:owner.reset_history()
                    if step["sensor_resize"]:
                        before=owner.diagnostics()["history"]["actual"]["committed_rgb"]
                        sensor=replace(camera,width=camera.width+2,height=camera.height+2)
                        owner.capture(frame,CaptureRequest(sensor,RenderPath.RASTER,(Channel.INSTANCE_ID,)))
                        if owner.diagnostics()["history"]["actual"]["committed_rgb"]!=before:
                            raise ValueError("sensor-only capture committed RGB history")
                        try:owner.read_history_incoming()
                        except RuntimeError:pass
                        else:raise ValueError("sensor-only exposed stale RGB history")
                    if index==3:
                        before=owner.diagnostics()["history"]["actual"]["committed_rgb"]
                        wrong=dict(binding,previous_frame_sha256="0"*64)
                        try:owner.capture(frame,CaptureRequest(camera,RenderPath.RASTER,(Channel.RGB,)),presentation=wrong)
                        except ValueError:pass
                        else:raise ValueError("wrong predecessor was accepted")
                        if owner.diagnostics()["history"]["actual"]["committed_rgb"]!=before:
                            raise ValueError("failed predecessor mutated history")
                    request=CaptureRequest(camera,RenderPath.RASTER,tuple(Channel))
                    start=perf_counter();shot=owner.capture(frame,request,presentation=binding);elapsed=1000*(perf_counter()-start)
                    if (shot.frame_key,shot.time_s,shot.camera)!=(frame.key,frame.time_s,camera):raise ValueError("stale captured identity")
                    planes={v.channel:v.data for v in shot.planes}
                    current=owner.read_current_incoming();history=owner.read_history_incoming();linear=owner.read_reconstructed_linear()
                    now=np.frombuffer(current["data"],"<f4").reshape(camera.height,camera.width,4)
                    cells=np.frombuffer(history["data"],dtype=history_dtype(np)).reshape(camera.height,camera.width)
                    if not np.isfinite(now).all() or not np.isfinite(cells["incoming"]).all() or not np.isfinite(cells["lighting"]).all():
                        raise ValueError("nonfinite current/history/confidence values")
                    age=cells["incoming"][...,3];supported=now[...,3]==1
                    if np.any(age[~supported]) or np.any(age[supported]<1) or np.any(age>4):raise ValueError("invalid short-history age")
                    diag=owner.diagnostics();actual=diag["history"]["actual"];rigid=diag["rigid_history"]
                    if actual["committed_rgb"]!=index+1 or rigid["committed_rgb"]!=index+1:raise ValueError("publication count differs")
                    if actual["eligible_pixels"]!=int(supported.sum()) or actual["age_sum"]!=int(age.sum()):raise ValueError("history support/age accounting differs")
                    if sum(actual[k] for k in ("accepted_pixels","rejected_projection","rejected_geometry","rejected_support","reset_pixels"))!=actual["eligible_pixels"]:
                        raise ValueError("history rejection partition differs")
                    if rigid["moving_accepted_pixels"]>actual["accepted_pixels"] or rigid["lookup_misses"]:
                        raise ValueError("rigid lookup/motion accounting differs")
                    if sum(rigid[k] for k in ("rejected_lighting","rejected_incoming"))>actual["rejected_support"]:
                        raise ValueError("confidence rejection accounting differs")
                    for suffix,data in (("current.f32",current["data"]),("history.bin",history["data"]),("linear.f32",linear["data"]),
                                        *((channel.value,data) for channel,data in planes.items())):
                        (folder/f"{index:02d}-{step['name']}.{suffix}").write_bytes(data)
                    Image.frombytes("RGB",(camera.width,camera.height),planes[Channel.RGB]).save(folder/f"{index:02d}-{step['name']}.png")
                    rows.append(dict(name=step["name"],capture_ms=elapsed,diagnostics=diag,
                        planes={ch.value:digest(data) for ch,data in planes.items()},current_sha256=digest(current["data"]),
                        history_sha256=digest(history["data"]),linear_sha256=digest(linear["data"])))
                    snapshots.append(dict(planes=planes,current=current["data"],history=history["data"],linear=linear["data"]))
                    dump(output/"report.json",report)
            finally:
                original_error=sys.exc_info()[0] is not None
                try:
                    owner.close();closed=owner.diagnostics();report["runs"][label]["closed"]=closed
                    device=closed["device"]
                    if device.get("closed") is not True or any(device.get(k,1) for k in ("live_contexts","live_frames","live_resources","validation_errors")):
                        raise ValueError("resource/validation close failed")
                except BaseException as error:
                    report["runs"][label]["close_error"]=str(error)
                    if not original_error:raise
        fresh,temporal=recorded["fresh-sequence"],recorded["rigid-history"]
        rows=report["runs"]["rigid-history"]["captures"]
        for index,(a,b) in enumerate(zip(fresh,temporal)):
            if a["current"]!=b["current"]:raise ValueError("temporal reuse changed current estimator/seed")
            if any(a["planes"][ch]!=b["planes"][ch] for ch in Channel if ch is not Channel.RGB):raise ValueError("temporal reuse changed sensors")
            if rows[index]["diagnostics"]["history"]["actual"]["last_reset"] and (a["planes"]!=b["planes"] or a["linear"]!=b["linear"]):
                raise ValueError("reset failed same-sequence fresh comparison")
        moving=sum(r["diagnostics"]["rigid_history"]["moving_accepted_pixels"] for r in rows)
        if not moving:raise ValueError("no moving-instance history pixel was accepted")
        if any(r["diagnostics"]["history"]["last_reset_reasons"] for r in rows[1:8]):
            raise ValueError("continuous bound rigid motion unexpectedly reset all history")
        witnesses=[dict(index=i,**disocclusion_witness(scene,temporal[i-1],temporal[i],steps[i-1],steps[i]))
                   for i in range(1,len(steps)) if not rows[i]["diagnostics"]["history"]["actual"]["last_reset"]]
        report["disocclusion_witnesses"]=witnesses
        report["disocclusion_evidence"]=sum(v["definite_disocclusion_pixels"] for v in witnesses)>0
        if args.case=="synthetic":
            if not report["disocclusion_evidence"]:raise ValueError("synthetic motion did not expose an independent rejection witness")
            if fresh[0]["current"]==fresh[1]["current"]:raise ValueError("same geometry did not actually vary current sample sequence")
            cells=np.frombuffer(temporal[8]["history"],dtype=history_dtype(np))
            supported=(cells["geometry"][:,2]==8)&(cells["geometry"][:,3]==1)
            if not supported.any():raise ValueError("teleported baffle has no supported visibility witness")
            if np.any(cells["incoming"][supported,3]!=1):raise ValueError("large per-ID motion retained history")
        report["checks"]=dict(current_estimator_same_bytes=True,sensors_same_bytes=True,reset_same_bytes=True,
            actual_rigid_reuse_pixels=moving,bound_predecessor_rejection=True,primary_disocclusion_age_one=True,
            quality_improvement_evaluated=False,dynamic_secondary_visibility_verified=False)
        report["status"]="complete"
    except BaseException as error:
        report["status"]="partial_failure";report["error"]=dict(type=type(error).__name__,message=str(error));raise
    finally:dump(output/"report.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--case",choices=("synthetic","panda"),default="synthetic")
    parser.add_argument("--inputs",type=Path)
    parser.add_argument("--start",type=int,default=54)
    parser.add_argument("--steps",type=int,default=12)
    parser.add_argument("--width",type=int,default=320)
    parser.add_argument("--height",type=int,default=180)
    parser.add_argument("--device-index",type=int,default=1)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--validation",action="store_true")
    run(parser.parse_args())


if __name__=="__main__":main()
