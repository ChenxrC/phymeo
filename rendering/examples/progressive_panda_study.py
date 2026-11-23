"""Unfiltered progressive replay of the original backlit Panda/red-board still.

The original meshes, camera, light, reflectance, exposure and dark-face ROI are
reused. RGB is published by the product's GPU scene-linear mean/display path;
this driver never averages images for presentation. Four independent seeds
measure convergence, and changed-state captures must equal fresh sessions.
"""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

import diffuse_bounce_study as original
from _panda_scene import look_at
from normal_map_film import _json, compare_sensors, verify_close, SENSORS
from panda_msaa_compare import _hash, _implementation_files, _timing
from phymeo_rendering.outputs import Channel
from phymeo_rendering.scene import DirectionalLight, NodePose, Pose

ROOT=Path(__file__).resolve().parent
STAGES=(1,4,16,64)
SEEDS=(0,1,2,3)


def frozen_profile():
    return dict(original_profile=original.frozen_profile(),bounces=1,milestones=list(STAGES),
        seeds=list(SEEDS),warmup_batches=0,primary_samples_per_batch=16,
        noise_space="scene-linear-before-exposure",spatial_filter=None,legacy_max_rgb8=1,
        rms_ratio_batch16_max=.55,rms_ratio_batch64_max=.35,mean_relative_change_16_to_64_max=.10,
        mean_red_excess_min=.01,restart_initial_batches=4,restart_new_batches=[1,4],
        changes=dict(camera_eye_m=[.42,.02,.24],blocker_centre_m=[0.,0.,.055],
                     board_centre_m=[.06,0.,.037],light_direction=[-2**-.5,0.,-2**-.5]),
        reset_fresh_comparison="all four public planes and real GPU linear mean exactly equal",
        timing="each public four-channel capture; no hidden warmup; linear readback separately timed",
        limits="one diffuse bounce; 16 primary paths per batch; no denoising, post-image blur, physical dynamics or two-bounce claim")


def _plain(value):
    return json.loads(json.dumps(value,allow_nan=False))


def build_inputs(directory):
    base=original.build_cases(directory)["red"]
    scene,frame,camera=base
    reference=original.register_reference(*base)
    key=replace(frame.key,tick=1,revision=1)
    moved_frame=replace(frame,key=key,time_s=.1)
    moved_camera=replace(camera,frame_key=key)
    variants={"camera":(scene,moved_frame,replace(moved_camera,
        world_pose=look_at((.42,.02,.24),(0.,0.,.09),(0,0,1))))}
    for name,position in (("blocker",(0.,0.,.055)),("board",(.06,0.,.037))):
        nodes=tuple(NodePose(n.node_id,Pose(position,n.world_pose.orientation_xyzw))
                    if n.node_id==name else n for n in frame.nodes)
        variants[name]=(scene,replace(moved_frame,nodes=nodes),moved_camera)
    light=DirectionalLight((-2**-.5,0.,-2**-.5),(3.,3.,3.))
    fresh_scene=replace(scene,key=scene.key+"-changed-light",lights=(light,))
    fresh_key=replace(key,scene_key=fresh_scene.key)
    variants["lights"]=(fresh_scene,replace(moved_frame,key=fresh_key),replace(moved_camera,frame_key=fresh_key))
    return base,reference,variants


def reason_for(name):
    return {"camera":"camera","board":"transforms","blocker":"transforms","lights":"lights"}[name]


def verify_history(diagnostic,*,batches,reason=None,previous_resets=None):
    data=diagnostic.get("progressive",{})
    expected=dict(enabled=True,bounces=1,batches=batches,primary_samples_per_pixel=16*batches,
        batch_index=batches-1,accumulation_space="scene-linear-before-exposure",storage="GPU RGBA32F mean",max_batches=4096)
    if any(data.get(k)!=v for k,v in expected.items()):
        raise ValueError("actual progressive sample/batch/linear-storage receipt differs")
    if reason is not None and reason not in data.get("last_reset_reasons",[]):
        raise ValueError("actual history reset reason differs")
    if previous_resets is not None and data.get("reset_count")!=previous_resets+1:
        raise ValueError("actual changed-state history was not reset exactly once")
    return True


def decode_linear(packet,*,width,height,batches):
    expected=dict(width=width,height=height,batches=batches,primary_samples_per_pixel=16*batches,
                  format="RGBA32F_LE",space="scene-linear-before-exposure")
    if (any(packet.get(k)!=v for k,v in expected.items()) or type(packet.get("data")) is not bytes
        or len(packet["data"])!=width*height*16):
        raise ValueError("public GPU linear mean layout/stage/owned-byte contract differs")
    value=np.frombuffer(packet["data"],"<f4").reshape(height,width,4)
    if not np.isfinite(value).all() or np.any(value[:,:,:3]<0):
        raise ValueError("GPU linear mean contains nonfinite or negative RGB")
    return value


def noise_report(values,mask):
    if set(values)!=set(SEEDS) or any(set(values[s])!=set(STAGES) for s in SEEDS):
        raise ValueError("all four independent seeds and all registered batch stages required")
    mask=np.asarray(mask,bool)
    if not mask.any():
        raise ValueError("registered pixels required")
    rows={};means={};baseline=None
    for batch in STAGES:
        arrays=[np.asarray(values[s][batch],float) for s in SEEDS]
        if any(a.shape!=mask.shape+(3,) or not np.isfinite(a).all() or np.any(a<0) for a in arrays):
            raise ValueError("finite scene-linear RGB required for every seed/stage; no sample deletion")
        samples=np.stack([a[mask] for a in arrays])
        variance=np.var(samples,axis=0,ddof=1)
        rms=float(np.sqrt(np.mean(variance)))
        if baseline is None:
            baseline=rms
            if baseline<=0:
                raise ValueError("zero independent-seed variance cannot substantiate noise reduction")
        mean=samples.mean(axis=(0,1));means[batch]=mean
        rows[str(batch)]=dict(batches=batch,primary_spp=16*batch,independent_seeds=4,
            rms_scene_linear=rms,rms_ratio_to_batch1=rms/baseline,mean_rgb=mean.tolist(),
            mean_red_excess=float(mean@np.array((1.,-.5,-.5))),
            per_seed_mean_rgb=samples.mean(axis=1).tolist(),
            per_seed_zero_rgb_pixels=[int(np.sum(np.all(a==0,axis=1))) for a in samples])
    relative=float(np.linalg.norm(means[16]-means[64])/max(np.linalg.norm(means[64]),1e-30))
    stable=relative<=.10
    noise=rows["16"]["rms_ratio_to_batch1"]<=.55 and rows["64"]["rms_ratio_to_batch1"]<=.35
    color=rows["64"]["mean_red_excess"]>=.01
    return dict(passed=bool(stable and noise and color),pixels=int(mask.sum()),stages=rows,
        mean_relative_change_16_to_64=relative,mean_stability_passed=bool(stable),
        noise_passed=bool(noise),red_propagation_passed=bool(color),
        scope="sample variance across four independent seed sequences at identical pre-registered pixels; scene-linear RGB, ddof=1; all zero/noisy pixels retained",
        red_scope="raw mean R-(G+B)/2 on the original red-board scene; not a repetition of the previous red/grey difference-of-differences test")


def compare_restart(before,current,fresh,*,geometry_changed):
    if any(set(p)!=set(Channel) for p in (before,current,fresh)):
        raise ValueError("complete four-channel restart comparison required")
    if any(current[ch]!=fresh[ch] for ch in Channel):
        raise ValueError("changed-state frame differs from fresh same-state/seed frame: stale history or input mismatch")
    rgb_changed=before[Channel.RGB]!=current[Channel.RGB]
    sensor_changed=any(before[ch]!=current[ch] for ch in SENSORS)
    if not rgb_changed or sensor_changed!=geometry_changed:
        raise ValueError("old frame did not show the registered real light-path/geometry change")
    return dict(fresh_identical=True,old_frame_rejected=True,rgb_changed=True,
                geometry_changed=sensor_changed,
                actual_sha256={ch.value:sha256(current[ch]).hexdigest() for ch in Channel})


def compare_legacy(legacy,first):
    a,b=np.asarray(legacy),np.asarray(first)
    if a.dtype!=np.uint8 or b.dtype!=np.uint8 or a.shape!=b.shape or a.ndim!=3 or a.shape[-1]!=3:
        raise ValueError("same-size public RGB8 images required")
    difference=np.abs(a.astype(np.int16)-b.astype(np.int16))
    maximum=int(difference.max())
    if maximum>1:
        raise ValueError("legacy to first progressive batch exceeds registered one-RGB8 quantization bound")
    return dict(max_rgb8=maximum,mean_abs_rgb8=float(difference.mean()),
                scope="same original input and seed directions; new linear mean/display quantization may differ by one code")


def _light_binding(diagnostic,lights):
    declared=[asdict(light) for light in lights]
    if _plain(diagnostic.get("effective_directional_lights"))!=_plain(declared):
        raise ValueError("current declared directional light differs")
    submitted=[]
    for light in lights:
        direction=np.asarray(light.direction,float)/math.hypot(*light.direction)
        submitted.append(dict(direction=direction.astype(np.float32).astype(float).tolist(),
                              illuminance=np.asarray(light.illuminance,np.float32).astype(float).tolist()))
    if diagnostic.get("last_request",{}).get("effective_directional_lights")!=submitted:
        raise ValueError("actual submitted directional light differs from normalized FP32 input")
    return True


def _backend(library,shaders,device_index,seed,*,progressive=True):
    from phymeo_rendering.backends.native import NativeBackend
    return NativeBackend(library_path=library,shader_directory=shaders,device_index=device_index,
        resident=True,shadow_mode="ray_query",studio=None,msaa_samples=1,validation=False,
        exposure=1.,display_transform="pbr-neutral",indirect_mode="diffuse_bounce",
        indirect_seed=seed,indirect_bounces=1,progressive=progressive)


def _shot(session,case):
    _,frame,camera=case
    started=perf_counter();capture=session.capture(frame,original.capture_request(camera));elapsed=(perf_counter()-started)*1000
    if (capture.frame_key!=frame.key or capture.time_s!=frame.time_s or capture.camera!=camera
        or tuple(p.channel for p in capture.planes)!=tuple(Channel)):
        raise ValueError("actual capture publication differs from request")
    planes={p.channel:p.data for p in capture.planes}
    diagnostic=session.diagnostics()
    return dict(capture_ms=elapsed,planes=planes,diagnostic=diagnostic,
                frame=asdict(frame),camera=asdict(camera))


def _small(shot):
    return dict(capture_ms=shot["capture_ms"],frame=shot["frame"],camera=shot["camera"],
        progressive=shot["diagnostic"].get("progressive"),
        indirect=shot["diagnostic"].get("indirect"),
        effective_directional_lights=shot["diagnostic"].get("effective_directional_lights"),
        submitted_directional_lights=shot["diagnostic"].get("last_request",{}).get("effective_directional_lights"),
        plane_sha256={ch.value:sha256(raw).hexdigest() for ch,raw in shot["planes"].items()})


def _save(root,name,shot,session=None,*,batches=None):
    from PIL import Image
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    planes=shot["planes"]
    rgb=np.frombuffer(planes[Channel.RGB],np.uint8).reshape(960,1280,3)
    Image.fromarray(rgb).save(path.with_suffix(".png"))
    data=dict(RGB=rgb,
        FORWARD_DEPTH=np.frombuffer(planes[Channel.FORWARD_DEPTH],"<f4").reshape(960,1280),
        INSTANCE_ID=np.frombuffer(planes[Channel.INSTANCE_ID],"<u4").reshape(960,1280),
        VALID=np.frombuffer(planes[Channel.VALID],np.uint8).reshape(960,1280))
    record=_small(shot)
    if session is not None:
        before=session.diagnostics();started=perf_counter();packet=session.read_linear_mean();elapsed=(perf_counter()-started)*1000
        rgba=decode_linear(packet,width=1280,height=960,batches=batches)
        after=session.diagnostics()
        if (before["progressive"]!=after["progressive"]
            or before["device"]["resident_captures"]!=after["device"]["resident_captures"]):
            raise ValueError("read-only linear mean unexpectedly advanced accumulation/capture")
        data["RGBA32F"]=rgba
        record["linear_readback"]=dict(capture_advancement=0,readback_ms=elapsed,
            metadata={k:v for k,v in packet.items() if k!="data"},sha256=sha256(packet["data"]).hexdigest(),
            device_before=before["device"].get("resident_progressive_info"),
            device_after=after["device"].get("resident_progressive_info"))
    np.savez_compressed(path.with_suffix(".npz"),**data)
    record.update(png=str(path.with_suffix(".png").relative_to(root)).replace("\\","/"),
        png_sha256=_hash(path.with_suffix(".png")),data=str(path.with_suffix(".npz").relative_to(root)).replace("\\","/"),
        data_sha256=_hash(path.with_suffix(".npz")))
    _json(path.with_suffix(".json"),record)
    return data,record


def _close(session,record,path):
    original_error=sys.exc_info()[0] is not None
    close_error=None
    if session is not None:
        try:
            record["before_close"]=session.diagnostics();session.close()
            record["backend"]=session.diagnostics();record["close"]=verify_close(record["backend"])
        except Exception as error:
            close_error=error;record["close_error"]=dict(type=type(error).__name__,message=str(error))
    _json(path,record)
    if close_error is not None and not original_error:
        raise close_error


def _require_owner_counts(record,captures,reads):
    device=record["backend"]["device"]
    if device.get("resident_captures")!=captures or device.get("resident_upload_calls")!=1:
        raise ValueError("actual resident capture/upload counts differ")
    info=device.get("resident_progressive_info")
    if reads:
        if (not isinstance(info,dict) or info.get("mean_readbacks")!=reads
            or info.get("mean_readback_bytes")!=reads*1280*960*16):
            raise ValueError("actual public linear readback counts/bytes differ")


def _bind_original(directory,base,reference):
    if directory is None:
        return dict(verified=False,reason="original saved capture not supplied; source fixture still exactly reused"),None
    directory=Path(directory).resolve()
    if directory.is_file():
        directory=directory.parent
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8-sig"))
    protocol=json.loads((directory/"protocol.json").read_text(encoding="utf-8-sig"))
    if summary.get("status")!="complete" or summary.get("protocol_sha256")!=_hash(directory/"protocol.json"):
        raise ValueError("original capture summary/protocol binding differs")
    scene,frame,camera=base
    old=protocol["cases"]["red"]
    actual=_plain(asdict(scene))
    if (protocol["profile"]!=original.frozen_profile() or old["frame"]!=_plain(asdict(frame))
        or old["camera"]!=_plain(asdict(camera))
        or any(old["scene"][k]!=actual[k] for k in ("key","instances","materials","lights"))
        or protocol["mesh_sha256"]!={m.id:original.load_mesh(m).source_sha256 for m in scene.meshes}):
        raise ValueError("original full camera/frame/material/light/mesh input differs")
    path=directory/"reference.npz"
    if _hash(path)!=protocol["reference_sha256"]:
        raise ValueError("original registered reference archive changed")
    with np.load(path,allow_pickle=False) as stored:
        if any(not np.array_equal(stored[k],v) for k,v in reference.items()):
            raise ValueError("original pre-registered ROI differs")
    receipt_row=summary["receipts"]["red"]["diffuse_bounce"]
    receipt_path=directory/receipt_row["file"]
    if _hash(receipt_path)!=receipt_row["sha256"]:
        raise ValueError("original red image receipt changed")
    receipt=json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    rgb,sensors=original._load(directory,receipt)
    return dict(verified=True,directory=str(directory),summary_sha256=_hash(directory/"summary.json"),
        protocol_sha256=_hash(directory/"protocol.json"),reference_sha256=_hash(path),
        receipt_sha256=_hash(receipt_path),png=receipt["png"],png_sha256=receipt["png_sha256"],
        scope="entire original camera/frame/illumination/material/mesh and predeclared ROI bound; mesh URI roots may relocate"),(rgb,sensors)


def _static_sequence(root,base,reference,library,shaders,device_index,seed,baseline):
    folder=root/f"seed-{seed}";folder.mkdir()
    record=dict(seed=seed,status="opening",captures=[],snapshots={})
    session=None;noise={};first_planes=None
    try:
        start=perf_counter();session=_backend(library,shaders,device_index,seed).open_scene(base[0])
        record["open_ms"]=(perf_counter()-start)*1000
        for batch in range(1,65):
            shot=_shot(session,base)
            verify_history(shot["diagnostic"],batches=batch,reason="new_session" if batch==1 else None,previous_resets=0)
            _light_binding(shot["diagnostic"],base[0].lights)
            compare_sensors({c:baseline[c] for c in SENSORS},{c:shot["planes"][c] for c in SENSORS})
            ids=np.frombuffer(baseline[Channel.INSTANCE_ID],"<u4").reshape(960,1280)
            a=np.frombuffer(baseline[Channel.RGB],np.uint8).reshape(960,1280,3)
            b=np.frombuffer(shot["planes"][Channel.RGB],np.uint8).reshape(960,1280,3)
            if not np.array_equal(a[ids==0],b[ids==0]):
                raise ValueError("progressive sampling changed background outside foreground geometry")
            record["captures"].append(_small(shot))
            if batch in STAGES:
                data,snapshot=_save(root,f"seed-{seed}/batch-{batch:04d}",shot,session,batches=batch)
                record["snapshots"][str(batch)]=snapshot
                noise[batch]=data["RGBA32F"][:,:,:3][reference["ids"]!=0].astype(float).reshape(-1,1,3)
                original.verify_roi(reference,data["INSTANCE_ID"],data["FORWARD_DEPTH"],data["VALID"])
            if batch==1:
                first_planes=shot["planes"]
        record.update(status="complete",timing_all_batches=_timing([r["capture_ms"] for r in record["captures"]]),
            timing_after_first=_timing([r["capture_ms"] for r in record["captures"]][1:]),
            geometry_full_frame_identical=True,background_identical=True)
    except BaseException as error:
        record.update(status="failed",error=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        _close(session,record,folder/"receipt.json")
    _require_owner_counts(record,64,4)
    return record,noise,first_planes


def _restart(root,name,base,variant,library,shaders,device_index):
    folder=root/f"restart-{name}";folder.mkdir()
    record=dict(name=name,status="opening",same_owner_change=True,initial=[],changed=[],fresh=[])
    session=None;before=None;changed={};fresh={};linear_changed={};linear_fresh={}
    try:
        session=_backend(library,shaders,device_index,0).open_scene(base[0])
        for batch in range(1,5):
            shot=_shot(session,base);verify_history(shot["diagnostic"],batches=batch,previous_resets=0)
            _light_binding(shot["diagnostic"],base[0].lights)
            record["initial"].append(_small(shot))
        before=shot["planes"]
        _,record["before_snapshot"]=_save(root,f"restart-{name}/before",shot,session,batches=4)
        previous=shot["diagnostic"]["progressive"]["reset_count"]
        actual_case=variant
        if name=="lights":
            # This modifies live uniforms in this owner; it is NOT replaced with a new session.
            session.set_directional_lights(variant[0].lights)
            actual_key=replace(variant[1].key,scene_key=base[0].key)
            actual_case=(base[0],replace(variant[1],key=actual_key),replace(variant[2],frame_key=actual_key))
            record["light_update"]=dict(api="NativeRenderSession.set_directional_lights",
                same_session=True,declared_before=[asdict(x) for x in base[0].lights],
                declared_after=[asdict(x) for x in variant[0].lights])
        for batch in range(1,5):
            shot=_shot(session,actual_case)
            verify_history(shot["diagnostic"],batches=batch,
                reason=reason_for(name) if batch==1 else None,previous_resets=previous)
            _light_binding(shot["diagnostic"],variant[0].lights)
            record["changed"].append(_small(shot))
            if batch in (1,4):
                data,saved=_save(root,f"restart-{name}/changed-b{batch:04d}",shot,session,batches=batch)
                changed[batch]=shot["planes"];linear_changed[batch]=data["RGBA32F"].copy()
                record.setdefault("changed_snapshots",{})[str(batch)]=saved
        record["status"]="changed_complete"
    except BaseException as error:
        record.update(status="failed",error=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        _close(session,record,folder/"changed-owner.json")
    _require_owner_counts(record,8,3)
    # Close the changed owner before fresh creation, so global resource counts are honest.
    fresh_record=dict(status="opening",captures=[]);session=None
    try:
        session=_backend(library,shaders,device_index,0).open_scene(variant[0])
        for batch in range(1,5):
            shot=_shot(session,variant)
            verify_history(shot["diagnostic"],batches=batch,reason="new_session" if batch==1 else None,previous_resets=0)
            _light_binding(shot["diagnostic"],variant[0].lights)
            fresh_record["captures"].append(_small(shot))
            if batch in (1,4):
                data,saved=_save(root,f"restart-{name}/fresh-b{batch:04d}",shot,session,batches=batch)
                fresh[batch]=shot["planes"];linear_fresh[batch]=data["RGBA32F"].copy()
                record.setdefault("fresh_snapshots",{})[str(batch)]=saved
        fresh_record["status"]="complete"
    except BaseException as error:
        fresh_record.update(status="failed",error=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        _close(session,fresh_record,folder/"fresh-owner.json")
    _require_owner_counts(fresh_record,4,2)
    comparisons={}
    for batch in (1,4):
        comparisons[str(batch)]=compare_restart(before,changed[batch],fresh[batch],geometry_changed=name!="lights")
        if linear_changed[batch].tobytes()!=linear_fresh[batch].tobytes():
            raise ValueError("changed-state real GPU linear mean differs from fresh same state")
        comparisons[str(batch)]["linear_mean_identical"]=True
        # Exercise the stale-history negative against the actual captured fresh image.
        try:
            compare_restart(before,before,fresh[batch],geometry_changed=name!="lights")
        except ValueError:
            comparisons[str(batch)]["actual_stale_negative_rejected"]=True
        else:
            raise ValueError("actual old frame accepted as changed-state evidence")
    result=dict(name=name,status="complete",same_owner_change=True,reason=reason_for(name),
        comparisons=comparisons,changed_owner=dict(file=f"restart-{name}/changed-owner.json",sha256=_hash(folder/"changed-owner.json")),
        fresh_owner=dict(file=f"restart-{name}/fresh-owner.json",sha256=_hash(folder/"fresh-owner.json")),
        semantics="fresh owner is only a replay reference; the tested mutation happened after four accumulated batches in the original owner")
    _json(folder/"comparison.json",result)
    return result


def _panels(root,converged):
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.load_default(size=22)
    def panel(items,path,columns,crop=None):
        width,height=(1280,960) if crop is None else (crop[2]-crop[0],crop[3]-crop[1])
        rows=math.ceil(len(items)/columns)
        canvas=Image.new("RGB",(columns*width+(columns-1)*2,rows*(height+48)+(rows-1)*2),(22,24,29))
        for i,(file,label) in enumerate(items):
            with Image.open(root/file) as source:
                image=source.convert("RGB");image=image if crop is None else image.crop(crop)
            x=(i%columns)*(width+2);y=(i//columns)*(height+50)
            canvas.paste(image,(x,y+48));ImageDraw.Draw(canvas).text((x+10,y+15),label,font=font,fill="white")
        canvas.save(root/path)
    suffix="" if converged else " | CONVERGENCE GATE FAILED"
    items=[(f"seed-0/batch-{n:04d}.png",f"{n} BATCHES / {n*16} SPP | SAME EXPOSURE"+suffix) for n in STAGES]
    panel(items,"progressive-full.png",2)
    panel(items,"progressive-roi.png",4,original.CROP)
    panel([("legacy.png","LEGACY 16 SPP"),("seed-0/batch-0001.png","PROGRESSIVE FIRST 16 SPP")],"legacy-first.png",2)
    for name in ("camera","blocker","board","lights"):
        items=[(f"restart-{name}/{file}.png",f"{name.upper()} | {label}") for file,label in
               (("before","BEFORE / 64 SPP"),("changed-b0001","MOVED / FIRST 16 SPP"),("fresh-b0001","FRESH / FIRST 16 SPP"))]
        panel(items,f"restart-{name}-first.png",3)
        panel(items,f"restart-{name}-first-roi.png",3,original.CROP)
        panel([(f"restart-{name}/changed-b0004.png",f"{name.upper()} | MOVED / 64 SPP"),
               (f"restart-{name}/fresh-b0004.png",f"{name.upper()} | FRESH / 64 SPP")],f"restart-{name}-later.png",2)


def run(output,library,*,device_index=1,shader_directory=None,original_directory=None):
    output,library=Path(output).resolve(),Path(library).resolve()
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema="phymeo-progressive-panda-study-v1",status="preparing",physics_steps=0)
    try:
        base,reference,variants=build_inputs(output/"geometry")
        np.savez_compressed(output/"reference.npz",**reference)
        binding,stored_original=_bind_original(original_directory,base,reference)
        shaders=library.parent if shader_directory is None else Path(shader_directory).resolve()
        protocol=dict(schema="phymeo-progressive-panda-protocol-v1",profile=frozen_profile(),
            input=dict(scene=asdict(base[0]),frame=asdict(base[1]),camera=asdict(base[2])),
            variants={k:dict(scene=asdict(s),frame=asdict(f),camera=asdict(c)) for k,(s,f,c) in variants.items()},
            original_capture=binding,reference_sha256=_hash(output/"reference.npz"),
            roi_pixels={str(i):int(np.sum(reference["ids"]==i)) for i in original.RECEIVERS},
            script_sha256=_hash(__file__),implementation=_implementation_files(),
            helpers={name:_hash(ROOT/name) for name in ("diffuse_bounce_study.py","_panda_scene.py","normal_map_film.py","panda_msaa_compare.py")},
            library=dict(file=str(library),sha256=_hash(library)),
            shaders={p.name:_hash(p) for p in sorted(shaders.glob("*.spv"))},device_index=device_index,
            mesh_sha256={m.id:original.load_mesh(m).source_sha256 for m in base[0].meshes},
            panda_provenance=original._panda_provenance(base[0]))
        _json(output/"protocol.json",protocol)
        report["stage"]="legacy"
        legacy_record=dict(status="opening");session=None
        try:
            session=_backend(library,shaders,device_index,0,progressive=False).open_scene(base[0])
            shot=_shot(session,base);legacy=shot["planes"]
            data,legacy_record["snapshot"]=_save(output,"legacy",shot)
            original.verify_roi(reference,data["INSTANCE_ID"],data["FORWARD_DEPTH"],data["VALID"])
            legacy_record.update(status="complete",capture_ms=shot["capture_ms"])
        except BaseException as error:
            legacy_record.update(status="failed",error=dict(type=type(error).__name__,message=str(error)))
            raise
        finally:
            _close(session,legacy_record,output/"legacy-owner.json")
        _require_owner_counts(legacy_record,1,0)
        if stored_original is not None:
            compare_sensors(stored_original[1],{c:legacy[c] for c in SENSORS})
            report["original_image_comparison"]=compare_legacy(stored_original[0],data["RGB"])
        sequence_records={};noise={};first=None
        for seed in SEEDS:
            report["stage"]=f"static-seed-{seed}"
            record,values,planes=_static_sequence(output,base,reference,library,shaders,device_index,seed,legacy)
            sequence_records[str(seed)]=dict(file=f"seed-{seed}/receipt.json",sha256=_hash(output/f"seed-{seed}/receipt.json"),
                timing_all_batches=record["timing_all_batches"],timing_after_first=record["timing_after_first"],close=record["close"])
            noise[seed]=values
            if seed==0:
                first=planes
        old_rgb=np.frombuffer(legacy[Channel.RGB],np.uint8).reshape(960,1280,3)
        first_rgb=np.frombuffer(first[Channel.RGB],np.uint8).reshape(960,1280,3)
        report["legacy_first"]=compare_legacy(old_rgb,first_rgb)
        metrics=noise_report(noise,np.ones((int(np.sum(reference["ids"]!=0)),1),bool))
        _json(output/"noise.json",metrics)
        restarts={}
        for name,variant in variants.items():
            report["stage"]=f"restart-{name}"
            restarts[name]=_restart(output,name,base,variant,library,shaders,device_index)
        if (protocol["script_sha256"]!=_hash(__file__) or protocol["implementation"]!=_implementation_files()
            or protocol["library"]["sha256"]!=_hash(library)
            or protocol["shaders"]!={p.name:_hash(p) for p in sorted(shaders.glob("*.spv"))}
            or protocol["mesh_sha256"]!={m.id:original.load_mesh(m).source_sha256 for m in base[0].meshes}
            or any(_hash(ROOT/name)!=digest for name,digest in protocol["helpers"].items())):
            raise ValueError("registered source/device/mesh bindings changed during capture")
        _panels(output,metrics["passed"])
        report.update(status="complete" if metrics["passed"] else "convergence_gate_failed",
            stage="complete",protocol_sha256=_hash(output/"protocol.json"),noise=metrics,noise_sha256=_hash(output/"noise.json"),
            static_sequences=sequence_records,restarts=restarts,actual_captures=305,actual_linear_readbacks=36,
            note="raw original-size product images; panels add only exterior labels. Static noise gates are separate from exact reset/state/sensor gates.")
    except BaseException as error:
        report.update(status="failed",error=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        _json(output/"summary.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--shader-directory",type=Path)
    parser.add_argument("--device-index",type=int,default=1)
    parser.add_argument("--original-directory",type=Path)
    args=parser.parse_args()
    result=run(args.output,args.library,device_index=args.device_index,
               shader_directory=args.shader_directory,original_directory=args.original_directory)
    print(json.dumps(dict(status=result["status"],output=str(args.output.resolve())),ensure_ascii=False))
    if result["status"]!="complete":
        raise SystemExit(2)


if __name__=="__main__":
    main()
