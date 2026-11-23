"""Shared same-model one-hop reference; candidate is never used as an oracle.

Captures an independent stream of full-rate batches, each with 16 proposals per pixel.
The private package is activated only as transport for the unchanged baseline DLL.
"""
import argparse
from dataclasses import asdict,replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import json
import run


def main():
    ap=argparse.ArgumentParser(description=__doc__);run.common(ap)
    ap.add_argument("--seed",type=int,choices=(1,2),default=1)
    ap.add_argument("--batches",type=int,choices=(64,512,4096),default=64,
                    help="fixed reference refinement levels; simulator limit is 4096 batches")
    ap.add_argument("--save-linear",action="store_true")
    args=ap.parse_args()
    run.require(args.camera=="detail" and args.studio_look=="recorded",
        "registered reference captures use recorded/detail; aesthetic films are separate")
    args.baseline_library=args.baseline_library.resolve(strict=True)
    args.baseline_shaders=(args.baseline_shaders or args.baseline_library.parent).resolve(strict=True)
    args.output=args.output.resolve()
    runtime,prepared,manifest=run.load_runtime(args)
    import numpy as np
    from PIL import Image
    import panda_render_performance as factory
    from phymeo_rendering.outputs import Channel
    source=run.source_binding(args,prepared,manifest)
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(schema=run.REFERENCE_SCHEMA,status="preparing",source=source,seed=args.seed,batches=args.batches,
        primary_proposals_per_pixel=16*args.batches,records=[],reference_model="unchanged full-rate one diffuse bounce; final unfiltered GPU mean display",
        sparse_candidate_captured=False)
    run.save(args.output/"summary.json",report)
    owner=None;problem=None
    try:
        with TemporaryDirectory(prefix="phymeo-sparse-reference-") as temp:
            scenes,frames,studio,trace=factory.bind_scene(Path(temp)/"geometry");scene=scenes[1]
            inputs=run.base.make_inputs(scene,frames,run.base.preset(run.SIZES[args.resolution]),"detail")
            run.save(args.output/"inputs.json",dict(scene=asdict(scene),studio=asdict(studio),trace_sha256=run.digest(trace),
                poses=[dict(index=i,frame=asdict(inputs[i][0]),camera=asdict(inputs[i][1].camera)) for i in run.REGISTERED]))
            report["inputs_sha256"]=run.digest(args.output/"inputs.json")
            owner=run.open_owner(args,scene,studio,runtime,prepared,manifest["contract"],"baseline16",seed=args.seed,reference=True)
            for index in range(10):
                frame,request=inputs[index];request=replace(request,channels=tuple(Channel))
                run.base.verify_capture(owner.capture(frame,request),frame,request)
            for index in run.REGISTERED:
                frame,request=inputs[index];request=replace(request,channels=tuple(Channel))
                owner.reset_accumulation();times=[]
                before=owner.diagnostics()["device"]["resident_profile"].copy()
                for batch in range(args.batches):
                    start=perf_counter();shot=owner.capture(frame,request);times.append(1000*(perf_counter()-start))
                    run.base.verify_capture(shot,frame,request)
                diag=owner.diagnostics()
                run.require(diag["progressive"]["batches"]==args.batches and diag["progressive"]["primary_samples_per_pixel"]==16*args.batches
                    and diag["indirect"]["seed"]==args.seed and diag["indirect"]["bounces"]==1
                    and not diag["spatial_filter"]["last_capture_filtered"],"reference actual sampling/display differs")
                planes={p.channel:p.data for p in shot.planes}
                width,height=request.camera.width,request.camera.height
                png=args.output/f"pose-{index:04d}.png";Image.frombytes("RGB",(width,height),planes[Channel.RGB]).save(png)
                geo=args.output/f"pose-{index:04d}.npz"
                np.savez_compressed(geo,instance_id=np.frombuffer(planes[Channel.INSTANCE_ID],"<u4").reshape(height,width),
                    depth=np.frombuffer(planes[Channel.FORWARD_DEPTH],"<f4").reshape(height,width),
                    valid=np.frombuffer(planes[Channel.VALID],"u1").reshape(height,width))
                start=perf_counter();raw=owner.read_linear_mean();read_ms=1000*(perf_counter()-start)
                run.require(raw["format"]=="RGBA32F_LE" and raw["space"]=="scene-linear-before-exposure"
                    and raw["batches"]==args.batches and np.isfinite(np.frombuffer(raw["data"],"<f4")).all(),"reference linear packet invalid")
                if args.save_linear:(args.output/f"pose-{index:04d}.rgba32f").write_bytes(raw["data"])
                record=dict(pose_index=index,frame_key=asdict(frame.key),time_s=frame.time_s,
                    state_sha256=sha256(run.canonical(asdict(frame))).hexdigest(),
                    camera_sha256=sha256(run.canonical(asdict(request.camera))).hexdigest(),
                    capture_ms=times,profile=run.base.profile_delta(before,diag["device"]["resident_profile"],args.batches),
                    png=png.name,png_sha256=run.digest(png),rgb_sha256=sha256(planes[Channel.RGB]).hexdigest(),
                    sensors={c.value:sha256(v).hexdigest() for c,v in planes.items() if c!=Channel.RGB},
                    geometry_file=geo.name,geometry_sha256=run.digest(geo),
                    raw_mean_sha256=sha256(raw["data"]).hexdigest(),raw_read_ms=read_ms,backend=diag)
                report["records"].append(record);run.save(args.output/"summary.json",report)
            report["status"]="complete"
    except BaseException as error:
        problem=error;report.update(status="partial_failure",error=dict(type=type(error).__name__,message=str(error)))
    finally:
        if owner is not None:
            try:
                owner.close();report["closed"]=owner.diagnostics();run.base.verify_close(report["closed"]["device"])
            except BaseException as error:
                report["close_error"]=str(error)
                if problem is None:problem=error;report["status"]="failed"
        run.save(args.output/"summary.json",report)
    if problem is not None:raise problem
    print(json.dumps(dict(status=report["status"],seed=args.seed,poses=list(run.REGISTERED))))


if __name__=="__main__":main()
