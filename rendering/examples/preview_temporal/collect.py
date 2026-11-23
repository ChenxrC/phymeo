"""Bounded actual raw-guide collection, NOT a temporal GPU implementation.

Uses12 consecutive sealed Panda states by default. No lighting/physics is
synthesized. Extra guide readback and file IO are explicitly outside capture wall
intervals; timings here are diagnostic, not a30FPS benchmark.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import numpy as np

HERE=Path(__file__).resolve().parent
EXAMPLES=HERE.parent
SCHEMA="phymeo-temporal-guide-collection-v1"


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def save(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def require(x,message):
    if not x:raise ValueError(message)


def pose_matrix(pose):
    q=np.asarray(pose["orientation_xyzw"],dtype=float)
    norm=np.linalg.norm(q);require(np.isfinite(norm) and abs(norm-1)<1e-6,"invalid recorded quaternion")
    x,y,z,w=q/norm;m=np.eye(4)
    m[:3,:3]=((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
               (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
               (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))
    m[:3,3]=pose["position_m"];return m


def metadata_for(scene,frame,camera,*,material_token,lighting_token,signal_token):
    nodes={p["node_id"]:pose_matrix(p["world_pose"]) for p in frame["nodes"]}
    return dict(camera=camera,key=frame["key"],time_s=frame["time_s"],
        camera_matrix=pose_matrix(camera["world_pose"]).tolist(),
        instance_matrices={str(i["id"]):(nodes[i["node_id"]]@pose_matrix(i["local_pose"])).tolist()
                           for i in scene["instances"]},
        material_token=material_token,lighting_token=lighting_token,signal_token=signal_token)

def control_camera_position(pose,index):
    require(type(index) is int and 0<=index<32,"bounded authored camera-control index required")
    transform=pose_matrix(pose)
    return (transform[:3,3]+transform[:3,0]*(index*.0001)).tolist()

def load_adapter(path):
    spec=importlib.util.spec_from_file_location("temporal_bound_sparse_adapter",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime",type=Path,required=True)
    ap.add_argument("--prepared",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--device-index",type=int,default=1)
    ap.add_argument("--start",type=int,default=55)
    ap.add_argument("--mode",choices=("recorded","camera-control"),default="recorded")
    ap.add_argument("--control-pose",type=int,default=59)
    ap.add_argument("--count",type=int,default=12)
    ap.add_argument("--camera",choices=("detail","sequence"),default="detail")
    ap.add_argument("--validation",action="store_true")
    args=ap.parse_args()
    require(2<=args.count<=32 and 0<=args.start<=300-args.count,"bounded contiguous source range required")
    require(0<=args.control_pose<300,"source control pose outside original episode")
    adapter=load_adapter(args.runtime.resolve(strict=True));prepared,manifest=adapter.activate(args.prepared)
    require("compact" not in manifest["contract"] and manifest["contract"].get("bounces")==1,
            "this offline study requires the original full-resolution sparse guide, not Compact/history data")
    # Renderer and fixture imports occur only after private-runtime activation.
    sys.path.insert(0,str(EXAMPLES))
    import panda_motion_preview as base
    import panda_render_performance as factory
    from phymeo_rendering.outputs import Channel
    from dataclasses import replace
    from PIL import Image
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(schema=SCHEMA,status="preparing",records=[],source=dict(
        script_sha256=digest(__file__),adapter_file=str(args.runtime.resolve()),adapter_sha256=digest(args.runtime),
        manifest_file=str(prepared/"source-manifest.json"),manifest_sha256=digest(prepared/"source-manifest.json"),
        library_sha256=digest(prepared/"build/phymeo_native.dll"),contract=manifest["contract"]),
        start=args.start if args.mode=="recorded" else 0,count=args.count,camera=args.camera,seed=0,mode=args.mode,
        scope="actual per-frame guide readback; traced-anchor incoming and untraced SH are distinguished; no GPU temporal accumulation",
        timing_scope="diagnostic synchronous capture, separately timed guide read and file output; no throughput claim")
    save(out/"summary.json",report);owner=None;problem=None
    try:
        with TemporaryDirectory(prefix="phymeo-temporal-guide-") as tmp:
            scenes,frames,studio,trace=factory.bind_scene(Path(tmp)/"geometry");scene=scenes[1]
            inputs=base.make_inputs(scene,frames,base.preset("preview720"),args.camera)
            origin=None
            if args.mode=="camera-control":
                frame,request=inputs[args.control_pose]
                origin=dict(pose_index=args.control_pose,frame=asdict(frame),camera=asdict(request.camera),
                    scope="new authored optical control: all original nodes frozen; +0.1mm camera-local X per exposure; not accepted physical trajectory")
                controlled={}
                for i in range(args.count):
                    key=replace(frame.key,world_id="camera-control:"+frame.key.world_id,epoch=0,revision=i,tick=i+1,phase="authored")
                    current=replace(frame,key=key,time_s=i/30)
                    pose=replace(request.camera.world_pose,position_m=tuple(control_camera_position(asdict(request.camera.world_pose),i)))
                    camera=replace(request.camera,world_pose=pose,frame_key=key)
                    controlled[i]=(current,replace(request,camera=camera))
                inputs=controlled
                selected=list(range(args.count))
            else:selected=list(range(args.start,args.start+args.count))
            save(out/"inputs.json",dict(scene=asdict(scene),studio=asdict(studio),trace_sha256=digest(trace),
                mode=args.mode,authored_control_origin=origin,
                frames=[dict(pose_index=i,frame=asdict(inputs[i][0]),camera=asdict(inputs[i][1].camera)) for i in selected]))
            report["inputs_sha256"]=digest(out/"inputs.json")
            owner=adapter.SparseBackend(prepared,device_index=args.device_index,studio=studio,seed=0,
                exposure=8.,display_transform="pbr-neutral",reflection_mode="single_bounce",validation=args.validation).open_scene(scene)
            for index in range(10):
                frame,request=inputs[index%args.count] if args.mode=="camera-control" else inputs[index];request=replace(request,channels=tuple(Channel))
                base.verify_capture(owner.capture(frame,request),frame,request)
            static=owner.diagnostics()
            material_token=sha256(canonical(dict(materials=asdict(scene)["materials"],
                decoded_geometry_sha256=static["decoded_geometry_sha256"]))).hexdigest()
            lighting_token=sha256(canonical(dict(studio=asdict(studio),lights=static["effective_directional_lights"]))).hexdigest()
            signal_token=sha256(canonical(dict(library=static["library_sha256"],contract=manifest["contract"],
                seed=0,signal="actual traced incoming guide; SH outside traced domain"))).hexdigest()
            require(static["library_sha256"]==report["source"]["library_sha256"] and
                    static["contract"]==manifest["contract"] and static["history"]=="none",
                    "loaded original sparse source/history differs")
            report["initial_backend"]=static
            report["tokens"]=dict(material_token=material_token,lighting_token=lighting_token,signal_token=signal_token)
            save(out/"summary.json",report)
            for ordinal,index in enumerate(selected):
                frame,request=inputs[index];request=replace(request,channels=tuple(Channel))
                start=perf_counter();shot=owner.capture(frame,request);capture_ms=1000*(perf_counter()-start)
                base.verify_capture(shot,frame,request);diag=owner.diagnostics()
                before=diag["device"]["resident_captures"]
                start=perf_counter();guide=owner.read_incoming_guide();read_ms=1000*(perf_counter()-start)
                require(owner.diagnostics()["device"]["resident_captures"]==before,"guide read advanced exposure")
                require(guide["format"]=="RGBA32F_LE" and guide["capture_index"]==10+ordinal and
                    guide["width"]==request.camera.width and guide["height"]==request.camera.height,"guide phase/extent differs")
                h,w=request.camera.height,request.camera.width;planes={p.channel:p.data for p in shot.planes}
                raw=np.frombuffer(guide["data"],"<f4").reshape(h,w,4)
                require(np.isfinite(raw).all(),"nonfinite actual guide")
                start=perf_counter();data=out/f"frame-{index:04d}.npz"
                np.savez_compressed(data,guide=raw,depth=np.frombuffer(planes[Channel.FORWARD_DEPTH],"<f4").reshape(h,w),
                    instance_id=np.frombuffer(planes[Channel.INSTANCE_ID],"<u4").reshape(h,w),
                    valid=np.frombuffer(planes[Channel.VALID],"u1").reshape(h,w))
                png=out/f"public-{index:04d}.png";Image.frombytes("RGB",(w,h),planes[Channel.RGB]).save(png)
                metadata=metadata_for(asdict(scene),asdict(frame),asdict(request.camera),**report["tokens"])
                record=dict(pose_index=index,file=data.name,sha256=digest(data),metadata=metadata,
                    frame_sha256=sha256(canonical(asdict(frame))).hexdigest(),camera_sha256=sha256(canonical(asdict(request.camera))).hexdigest(),
                    guide_sha256=sha256(guide["data"]).hexdigest(),guide_space=guide["space"],
                    sensor_sha256={c.value:sha256(v).hexdigest() for c,v in planes.items() if c!=Channel.RGB},
                    public_png=png.name,public_png_sha256=digest(png),public_rgb_sha256=sha256(planes[Channel.RGB]).hexdigest(),
                    capture_ms=capture_ms,guide_read_ms=read_ms,file_write_ms=1000*(perf_counter()-start),actual=diag["actual"])
                report["records"].append(record);save(out/"summary.json",report)
            report["status"]="complete"
    except BaseException as error:
        problem=error;report.update(status="partial_failure",error=dict(type=type(error).__name__,message=str(error)))
    finally:
        if owner is not None:
            try:owner.close();report["closed"]=owner.diagnostics();base.verify_close(report["closed"]["device"])
            except BaseException as error:
                report["close_error"]=str(error)
                if problem is None:problem=error;report["status"]="failed"
        save(out/"summary.json",report)
    if problem is not None:raise problem


if __name__=="__main__":main()