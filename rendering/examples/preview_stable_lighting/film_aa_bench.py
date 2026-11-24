"""Panda movie and analytic area/temporal controls for the opt-in film capture."""
import argparse
from contextlib import closing, ExitStack
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
import subprocess
from time import perf_counter
import json

import numpy as np
from PIL import Image, ImageDraw

import msaa_calibration as cal
from preview_compact_rigid_history.state import bind_presentations, frame_digest
from preview_lowres_filter._helpers import digest, save
from .film_aa import block_mean, film_capture, linear_exposure, scaled_camera
from .runtime import activate, open_session


def centered(owner, frame, request, binding):
    shot = owner.capture(frame, request, presentation=binding)
    if shot.camera != request.camera or shot.frame_key != frame.key or shot.time_s != frame.time_s:
        raise ValueError("center camera publication differs from the source")
    raw = owner.read_reconstructed_linear()
    return shot, np.frombuffer(raw["data"], "<f4").reshape(request.camera.height,request.camera.width,4)[...,:3]


def sensor_hashes(shot):
    from phymeo_rendering.outputs import Channel
    return {p.channel.value:sha256(p.data).hexdigest() for p in shot.planes if p.channel != Channel.RGB}


def calibration(args):
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath
    parent = args.output/"geometry"
    parent.mkdir()
    scene, models = cal._fixture(parent)
    cases = [("solid", ("solid",), 0.), ("edge", ("edge",), 0.)]
    cases.extend((f"rod-{n:02d}", ("rod",), n/16) for n in range(16))
    cases.append(("empty", (), 0.))
    work = [cal._exposure(scene,models,case,i) for i,case in enumerate(cases)]
    bindings = bind_presentations([r[0] for r in work],frame_digest(asdict(scene)))
    ideal = np.asarray([r[3] for r in work],dtype=float)
    edge_mask = (ideal[1]>1e-12) & (ideal[1]<1-1e-12)
    motion = ideal[2:-1]
    rod_mask = np.any((motion>1e-12) & (motion<1-1e-12),axis=0)
    reports, evidence = {}, {"reference":ideal}
    controls, previous = {}, None
    for name in ("realtime","film"):
        images, sensors, checks = [], [], []
        with ExitStack() as stack:
            owner = stack.enter_context(closing(open_session(args.package,scene,None,args.device_index,validation=True)))
            geometry = stack.enter_context(closing(open_session(args.package,scene,None,args.device_index,validation=True))) if name=="film" else None
            for (frame,camera,layers,_),binding in zip(work,bindings):
                request=CaptureRequest(camera,RenderPath.RASTER,tuple(Channel))
                if name=="film":
                    shot, hdr=film_capture(owner,frame,request,presentation=binding,sensor_owner=geometry,samples=args.samples)
                else:
                    shot, hdr=centered(owner,frame,request,binding)
                planes={p.channel:p.data for p in shot.planes}
                checks.append(cal._sensor_check(planes,layers,camera.width,camera.height))
                images.append(hdr.mean(axis=2))
                sensors.append(sensor_hashes(shot))
                if len(images) == 2:
                    Image.frombytes("RGB",(camera.width,camera.height),planes[Channel.RGB]).save(args.output/(name+"-edge.png"))
        values=np.stack(images)
        contrast=values[0]-values[-1]
        if np.any(contrast<=0):raise ValueError("white and empty calibration frames have insufficient contrast")
        normalized=(values-values[-1])/contrast
        edge=float(np.abs(normalized[1]-ideal[1])[edge_mask].mean())
        rod=np.abs(np.diff(normalized[2:-1],axis=0)-np.diff(motion,axis=0))[:,rod_mask].mean(axis=1)
        vacated=(motion[:-1]>1e-12)&(motion[1:]<=1e-12)
        reports[name]=dict(edge_mae=edge, rod_temporal_mae=float(rod.mean()), transition_maes=rod.tolist(),
                           vacated_observations=int(vacated.sum()),vacated_max_abs=float(np.abs(normalized[3:-1][vacated]).max()),
                           analytic_sensor_checks=checks, sensor_sha256=sensors, cleanup=cal._cleanup(owner.diagnostics()))
        evidence[name]=normalized
        if name=="realtime":
            previous=sensors
        else:
            controls["output_center_sensors_equal"]=(previous==sensors)
    np.savez_compressed(args.output/"analytic.npz",**evidence)
    report=dict(method=args.samples,case_names=[x[0] for x in cases],edge_pixels=int(edge_mask.sum()),
                registered_motion_pixels=int(rod_mask.sum()),modes=reports,**controls)
    report["improvement_vs_realtime"]={k:1-reports["film"][k]/reports["realtime"][k] for k in ("edge_mae","rod_temporal_mae")}
    save(args.output/"report.json",report)
    if not controls["output_center_sensors_equal"] or not all(r["cleanup"] and all(c["passed"] for c in r["analytic_sensor_checks"]) for r in reports.values()):
        raise ValueError("sensor or teardown control failed")


def movie(raw, output, n, other=None):
    common=["ffmpeg","-v","error","-nostdin","-n","-filter_complex_threads","1","-filter_threads","1"]
    inputs=["-f","rawvideo","-pixel_format","rgb24","-video_size","1280x720","-framerate","30","-i"]
    cmd=common+inputs+[str(raw)]
    if other is not None:
        cmd+=inputs+[str(other),"-filter_complex","[0:v][1:v]hstack=inputs=2[v]","-map","[v]"]
    cmd+=["-frames:v",str(n),"-c:v","libx264","-threads","2","-x264-params","rc-lookahead=6:sync-lookahead=0",
          "-crf","17","-pix_fmt","yuv420p","-movflags","+faststart",str(output)]
    subprocess.run(cmd,check=True)
    probe=subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries",
                          "stream=nb_read_frames,width,height,avg_frame_rate","-of","json",str(output)],check=True,capture_output=True,text=True)
    stream=json.loads(probe.stdout)["streams"][0]
    if (stream["nb_read_frames"] != str(n) or stream["avg_frame_rate"]!="30/1" or
            (stream["width"],stream["height"])!=(2560 if other else 1280,720)):
        raise ValueError("encoded movie did not preserve all presented frames")
    return dict(sha256=digest(output),stream=stream)


def finger_window(ids):
    finger=np.isin(ids,(155,156,157,158))
    y,x=np.where(finger)
    if not y.size:raise ValueError("no registered finger geometry in this exposure")
    pad=4
    x0=max(0,int(x.min())-pad); y0=max(0,int(y.min())-pad)
    x1=min(ids.shape[1],int(x.max())+pad+1); y1=min(ids.shape[0],int(y.max())+pad+1)
    return x0,y0,x1,y1


def reference_crop(owner,frame,request,binding,box,scale=8):
    x0,y0,x1,y1=box
    result=np.empty((y1-y0,x1-x0,3),dtype=np.float32)
    # Keep the ray lattice consistent across tiles and limit the size of the
    # compact transport's full-rate attachments and history allocations.
    for y in range(y0,y1,96):
        for x in range(x0,x1,96):
            right,bottom=min(x+96,x1),min(y+96,y1)
            # The fine-pixel contour resolver needs a one-fine-pixel neighbor
            # on the other side of a tile boundary. Render a one-output-pixel
            # guard and discard it before assembling the disjoint interiors.
            gx,gy=max(0,x-1),max(0,y-1)
            padded=(gx,gy,min(request.camera.width,right+1),min(request.camera.height,bottom+1))
            camera=scaled_camera(request.camera,scale,padded)
            high=linear_exposure(owner,frame,replace(request,camera=camera),presentation=binding)
            result[y-y0:bottom-y0,x-x0:right-x0]=block_mean(high,scale)[y-gy:bottom-gy,x-gx:right-gx]
    return result


def inspect_panda(args, scene, studio, inputs, bindings, index):
    from phymeo_rendering.backends._sapien_pixels import encode_linear_image
    from phymeo_rendering.outputs import Channel
    frame,request=inputs[index]
    request=replace(request,channels=tuple(Channel)); binding=bindings[index]
    with closing(open_session(args.package,scene,studio,args.device_index)) as base:
        shot,center=centered(base,frame,request,binding)
    ids=np.frombuffer(next(p.data for p in shot.planes if p.channel==Channel.INSTANCE_ID),"<u4").reshape(720,1280)
    box=finger_window(ids)
    with ExitStack() as stack:
        film=stack.enter_context(closing(open_session(args.package,scene,studio,args.device_index)))
        geometry=stack.enter_context(closing(open_session(args.package,scene,studio,args.device_index)))
        fine,film_hdr=film_capture(film,frame,request,presentation=binding,sensor_owner=geometry,samples=args.samples)
    with closing(open_session(args.package,scene,studio,args.device_index)) as target:
        ref=reference_crop(target,frame,request,binding,box)
    x0,y0,x1,y1=box
    sl=np.s_[y0:y1,x0:x1]
    mask=np.isin(ids,(155,156,157,158))
    padded=np.pad(mask,1,mode="edge")
    boundary=np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            boundary |= mask != padded[dy:dy+720,dx:dx+1280]
    boundary=boundary[sl]
    images={"realtime":center[sl],"film":film_hdr[sl],"reference":ref}
    metrics={}
    views={}
    ref_codes=np.frombuffer(encode_linear_image(ref,exposure=8.,display_transform="pbr-neutral"),"u1").reshape(ref.shape)
    for name,hdr in images.items():
        data=encode_linear_image(hdr,exposure=8.,display_transform="pbr-neutral")
        views[name]=Image.frombytes("RGB",(x1-x0,y1-y0),data)
        views[name].save(args.output/f"finger-{name}.png")
        errors=np.abs(hdr-ref)
        codes=np.frombuffer(data,"u1").reshape(ref.shape)
        metrics[name]=dict(crop_linear_rgb_mae=float(errors.mean()),
                           boundary_linear_rgb_mae=float(errors[boundary].mean()),
                           boundary_rgb8_mae=float(np.abs(codes[boundary].astype(float)-ref_codes[boundary]).mean()))
    w,h=(x1-x0)*3,(y1-y0)*3
    board=Image.new("RGB",(w*3,h+26),"#18222b")
    draw=ImageDraw.Draw(board)
    for i,(label,name) in enumerate((("Realtime 720p","realtime"),("Film 2x2" if args.samples=="grid2" else "Film rotated 4", "film"),
                                     ("Local 8x8 reference","reference"))):
        board.paste(views[name].resize((w,h),Image.Resampling.NEAREST),(i*w,26))
        draw.text((i*w+4,7),label,fill="white")
    board.save(args.output/"finger-comparison.png")
    np.savez_compressed(args.output/"finger-reference.npz",reference=ref,film=film_hdr[sl],realtime=center[sl],boundary=boundary)
    return dict(frame=index,box=box,reference="same scene and transport, 8x8 independently shaded fine pixels per output pixel before display",
                foreground_ids=[155,156,157,158],boundary_pixels=int(boundary.sum()),actual=metrics,
                center_sensor_equality=sensor_hashes(shot)==sensor_hashes(fine),
                center_sensor_hashes=sensor_hashes(fine))


def panda(args):
    from phymeo_rendering.outputs import Channel
    from .scene import prepare_scene
    if not (0 <= args.start < 300 and 1 <= args.frames <= 300-args.start):raise ValueError("frame range outside recorded motion")
    scene,studio,inputs,bindings,_=prepare_scene(args.output/"scene",args.source_inputs)
    end=args.start+args.frames
    index=60 if args.start<=60<end else (args.start+end-1)//2
    test=inspect_panda(args,scene,studio,inputs,bindings,index)
    report=dict(method=args.samples,registered_fingers=test,source_sha256=digest(args.source_inputs),
                package_seal_sha256=digest(args.package/"stable-seal.json"),
                sampling="true sensor/visibility/shading samples, linear area average, exposure 8, PBR Neutral and sRGB",
                first=args.start,last=end-1,frames=args.frames,timing={})
    save(args.output/"report.json",report)
    if not test["center_sensor_equality"]:raise ValueError("film changed center sensors")
    for name in ("realtime","film"):
        raw=args.output/(name+".rgb24")
        times=[]
        with closing(open_session(args.package,scene,studio,args.device_index)) as owner,raw.open("xb") as stream:
            def take(i):
                frame,request=inputs[i]
                request=replace(request,channels=(Channel.RGB,))
                if name=="film":
                    return film_capture(owner,frame,request,presentation=bindings[i],samples=args.samples)[0]
                return owner.capture(frame,request,presentation=bindings[i])
            for i in range(args.start,min(args.start+10,end)):take(i)
            owner.reset_history()
            for i in range(args.start,end):
                started=perf_counter(); shot=take(i); times.append((perf_counter()-started)*1000)
                stream.write(shot.planes[0].data)
        mean=float(np.mean(times))
        report["timing"][name]=dict(frame_ms=times,mean_ms=mean,fps=1000/mean,p95_ms=sorted(times)[int(np.ceil(.95*len(times)))-1],
                                     cleanup=cal._cleanup(owner.diagnostics()),device=owner.diagnostics()["device"]["device_name"])
        save(args.output/"report.json",report)
    report["movies"]={}
    for name in ("film","realtime"):
        report["movies"][name]=movie(args.output/(name+".rgb24"),args.output/(name+".mp4"),args.frames)
    report["movies"]["realtime_left_film_right"]=movie(args.output/"realtime.rgb24",args.output/"comparison.mp4",args.frames,args.output/"film.rgb24")
    save(args.output/"report.json",report)
    for name in ("realtime","film"):(args.output/(name+".rgb24")).unlink()


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    modes=cli.add_subparsers(dest="mode",required=True)
    for mode in ("panda","calibration"):
        p=modes.add_parser(mode)
        p.add_argument("--package",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
        p.add_argument("--samples",choices=("grid2","rotated4"),default="grid2")
        p.add_argument("--device-index",type=int,default=1)
        if mode=="panda":
            p.add_argument("--source-inputs",type=Path,required=True)
            p.add_argument("--start",type=int,default=0);p.add_argument("--frames",type=int,default=300)
    args=cli.parse_args();activate(args.package)
    args.output.mkdir(parents=True,exist_ok=False)
    (panda if args.mode=="panda" else calibration)(args)
