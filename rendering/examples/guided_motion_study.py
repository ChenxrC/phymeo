"""Registered Panda camera/foreground motion through the public guided renderer.

Images are direct public results. The motion is an authored optical fixture,
not robot dynamics. Independent RGB error statistics live in guided_motion_metrics.
The higher-sample seed is a finite reference, never an exact-radiance claim.
"""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import platform
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

import diffuse_bounce_study as original
import progressive_panda_study as progressive
import guided_motion_metrics as metrics
from _panda_scene import box, look_at
from normal_map_film import _json, SENSORS
from panda_msaa_compare import _hash, _implementation_files, _timing
from phymeo_rendering.outputs import Channel
from phymeo_rendering.scene import FrameKey, NodePose, Pose

ROOT=Path(__file__).resolve().parent
POSES=tuple(range(25))
REFERENCES=(0,15,24)
SAVED=(1,2,4)
BLOCKER=202


def frozen_profile():
    return dict(frames=25,fps=12,time_s=[0.,2.],saved_batches=list(SAVED),batches_per_pose=4,
        reference_poses=list(REFERENCES),required_reveal_poses=[15],seed=0,reference_seed=1,
        reference_batches=16,primary_samples_per_batch=16,expected_public_captures=272,
        msaa_samples=1,bounces=1,size=[1280,960],exposure=1.,display_transform="pbr-neutral",
        shadow_mode="ray_query",environment=None,display_error_space="inverse-sRGB display-linear",
        mode_order=["none","guided","reference","fresh-none-then-guided-per-reference"],
        camera_eye_start_m=[.42,-.012,.24],camera_eye_end_m=[.42,.012,.24],camera_target_m=[0.,0.,.09],
        blocker_id=BLOCKER,blocker_node="blocker",blocker_size_m=[.004,.060,.180],
        blocker_start_m=[.08,-.18,.14],blocker_end_m=[.08,.18,.14],blocker_path="smoothstep(0,1) in world Y",
        receiver_ids={"156":"panda_leftfinger","158":"panda_rightfinger"},right_finger_id=158,
        reference_registration="indices 0,15,24 fixed before capture; 15 must expose actual right-finger pixels or evidence is insufficient",
        masks="actual current ID/depth/VALID; screen-space reveal requires previous blocker and current farther finite valid foreground; new background excluded",
        filter_scope="public guided mode only first/reset RGB batch; batches 2 and 4 must exactly equal none",
        timing="request-to-return four-channel wall time including product guide/filter; explicit audit guide/raw reads and PNG encoding timed separately; no warmup",
        playback="12 Hz viewing schedule, not a measured throughput claim",
        original_profile=original.frozen_profile(),limits=["authored optical motion, zero physics steps","one diffuse bounce","single independent seed finite 256-spp reference","no image-space smoothing in this driver"])


def build_motion(directory):
    directory=Path(directory)
    scene,frame,camera=original.build_cases(directory)["red"]
    mesh=box(directory,"motion-foreground-blocker",(.004,.060,.180))
    old_mesh=next(i.mesh_id for i in scene.instances if i.id==BLOCKER)
    scene=replace(scene,key="panda-guided-motion-v1",
        meshes=tuple(m for m in scene.meshes if m.id!=old_mesh)+(mesh,),
        instances=tuple(replace(i,mesh_id=mesh.id) if i.id==BLOCKER else i for i in scene.instances))
    result=[]
    for i in POSES:
        t=i/24.;s=t*t*(3-2*t)
        # Endpoint expressions are exact declared decimal inputs, not accumulated motion.
        y=-.18 if i==0 else .18 if i==24 else -.18+.36*s
        eye_y=-.012 if i==0 else .012 if i==24 else -.012+.024*t
        key=FrameKey(scene.key,"guided-motion",0,i,i,"authored")
        nodes=tuple(NodePose(n.node_id,Pose((.08,y,.14))) if n.node_id=="blocker" else n for n in frame.nodes)
        current=replace(frame,key=key,time_s=i/12,nodes=nodes)
        cam=replace(camera,id="motion-gripper-camera",frame_key=key,
                    world_pose=look_at((.42,eye_y,.24),(0.,0.,.09),(0,0,1)))
        result.append((scene,current,cam))
    return tuple(result)


def image_name(mode,pose,batch):
    if (mode not in ("none","guided","reference") or type(pose) is not int or pose not in POSES
        or (mode=="reference" and (pose not in REFERENCES or batch!=16))
        or (mode!="reference" and batch not in SAVED)):
        raise ValueError("registered mode/pose/batch image required")
    return f"{mode}/frame-{pose:04d}-b{batch:04d}.png"


def _geometry_equal(a,b):
    keys={c.value for c in Channel}
    if set(a.get("plane_sha256",{}))!=keys or set(b.get("plane_sha256",{}))!=keys:
        raise ValueError("complete public four-plane evidence required")
    if any(a["plane_sha256"][c.value]!=b["plane_sha256"][c.value] for c in SENSORS):
        raise ValueError("actual depth/ID/VALID differs")


def verify_pair(raw,filtered):
    if any(raw.get(k)!=filtered.get(k) for k in ("frame","camera","batch")):
        raise ValueError("paired physical exposure or accumulation phase differs")
    _geometry_equal(raw,filtered)
    if not isinstance(raw.get("raw_mean_sha256"),str) or raw["raw_mean_sha256"]!=filtered.get("raw_mean_sha256"):
        raise ValueError("display filter changed the raw GPU scene-linear mean")
    identical=raw["plane_sha256"][Channel.RGB.value]==filtered["plane_sha256"][Channel.RGB.value]
    if raw["batch"]>1 and not identical:
        raise ValueError("later unfiltered batch must recover the exact same public RGB")
    return dict(raw_mean_identical=True,geometry_identical=True,rgb_identical=identical)


def verify_reference(raw,reference):
    if any(raw.get(k)!=reference.get(k) for k in ("frame","camera")) or reference.get("batch")!=16:
        raise ValueError("independent reference has a different actual pose or sample stage")
    _geometry_equal(raw,reference)
    return True


def verify_filter(diagnostic,*,mode,batch,filtered_count):
    d=diagnostic.get("spatial_filter",{})
    if (d.get("mode")!=mode or d.get("last_capture_filtered")!=(mode=="guided" and batch==1)
        or d.get("filtered_captures")!=filtered_count):
        raise ValueError("actual spatial filter phase/count differs")
    if mode=="guided" and (d.get("scope")!="first-rgb-batch-only"
                            or d.get("guide_source")!="same-exposure-gpu-direct-pass"):
        raise ValueError("guided path is not the declared first/reset separate GPU direct-pass path")
    return True


def visibility_counts(previous,current):
    ids=current["INSTANCE_ID"];z=current["FORWARD_DEPTH"];valid=current["VALID"]==1
    reveal=np.zeros(ids.shape,bool);background=np.zeros(ids.shape,bool)
    if previous is not None:
        before=previous["INSTANCE_ID"]==BLOCKER
        background=before & (ids==0)
        reveal=(before & (previous["VALID"]==1) & valid & (ids!=0) & (ids!=BLOCKER)
                & np.isfinite(previous["FORWARD_DEPTH"]) & (previous["FORWARD_DEPTH"]>0)
                & np.isfinite(z) & (z>0) & (z>previous["FORWARD_DEPTH"]))
    receivers={str(i):int(np.sum(reveal & (ids==i))) for i in (156,158)}
    return dict(revealed_foreground=int(reveal.sum()),revealed_receivers=receivers,
                revealed_right_finger=receivers["158"],new_background_excluded=int(background.sum()))


def _arrays(planes):
    return dict(INSTANCE_ID=np.frombuffer(planes[Channel.INSTANCE_ID],"<u4").reshape(960,1280),
        FORWARD_DEPTH=np.frombuffer(planes[Channel.FORWARD_DEPTH],"<f4").reshape(960,1280),
        VALID=np.frombuffer(planes[Channel.VALID],np.uint8).reshape(960,1280))


def _load_geometry(root,pose):
    with np.load(root/f"geometry/frame-{pose:04d}.npz",allow_pickle=False) as archive:
        return {k:archive[k] for k in ("INSTANCE_ID","FORWARD_DEPTH","VALID")}


def _backend(library,shaders,index,mode,seed):
    from phymeo_rendering.backends.native import NativeBackend
    return NativeBackend(library_path=library,shader_directory=shaders,device_index=index,
        resident=True,shadow_mode="ray_query",validation=False,exposure=1.,display_transform="pbr-neutral",
        msaa_samples=1,indirect_mode="diffuse_bounce",indirect_seed=seed,indirect_bounces=1,
        progressive=True,spatial_filter=mode)


def _record(shot,batch):
    row=progressive._small(shot)
    row.update(batch=batch,spatial_filter=shot["diagnostic"].get("spatial_filter"),
        spatial_device=shot["diagnostic"]["device"].get("resident_spatial_info"))
    return row


def _raw(session,record):
    before=session.diagnostics();start=perf_counter();packet=session.read_linear_mean();elapsed=(perf_counter()-start)*1000
    progressive.decode_linear(packet,width=1280,height=960,batches=record["batch"])
    after=session.diagnostics()
    if (before["progressive"]!=after["progressive"]
        or before["device"]["resident_captures"]!=after["device"]["resident_captures"]):
        raise ValueError("audit raw readback changed history")
    record.update(raw_mean_sha256=sha256(packet["data"]).hexdigest(),raw_readback_ms=elapsed,
                  raw_metadata={k:v for k,v in packet.items() if k!="data"})


def _guide(root,stem,session,record):
    from PIL import Image
    before=session.diagnostics();start=perf_counter();packet=session.read_spatial_guide();elapsed=(perf_counter()-start)*1000
    expected=dict(width=1280,height=960,format="RGBA8_UNORM",space="display-srgb-direct")
    if (any(packet.get(k)!=v for k,v in expected.items()) or type(packet.get("data")) is not bytes
        or len(packet["data"])!=1280*960*4):
        raise ValueError("public same-exposure spatial guide packet differs")
    after=session.diagnostics()
    if before["progressive"]!=after["progressive"] or before["device"]["resident_captures"]!=after["device"]["resident_captures"]:
        raise ValueError("audit guide readback changed history")
    path=root/(stem+"-guide.png");path.parent.mkdir(parents=True,exist_ok=True)
    start=perf_counter();Image.frombytes("RGBA",(1280,960),packet["data"]).save(path);save_ms=(perf_counter()-start)*1000
    record["guide"]=dict(file=path.relative_to(root).as_posix(),sha256=_hash(path),
        raw_sha256=sha256(packet["data"]).hexdigest(),readback_ms=elapsed,save_ms=save_ms,
        metadata={k:v for k,v in packet.items() if k!="data"},scope="actual same-exposure separate GPU direct guide; RGB meaningful on valid foreground only")


def _save_rgb(root,name,shot,record):
    from PIL import Image
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    start=perf_counter();Image.frombytes("RGB",(1280,960),shot["planes"][Channel.RGB]).save(path)
    record.update(png=name,png_sha256=_hash(path),png_save_ms=(perf_counter()-start)*1000)


def _owner_check(record,captures,raw_reads,guide_reads):
    d=record["backend"]["device"]
    if d.get("resident_captures")!=captures or d.get("resident_upload_calls")!=1:
        raise ValueError("actual owner capture/upload counts differ")
    p=d.get("resident_progressive_info",{})
    if p.get("mean_readbacks")!=raw_reads or p.get("mean_readback_bytes")!=raw_reads*1280*960*16:
        raise ValueError("actual audit raw readback counts differ")
    s=d.get("resident_spatial_info")
    if guide_reads and (not isinstance(s,dict) or s.get("guide_readbacks")!=guide_reads
                       or s.get("guide_readback_bytes")!=guide_reads*1280*960*4):
        raise ValueError("actual audit guide readback counts differ")


def _sequence(root,cases,library,shaders,index,mode,raw_records=None):
    folder=root/mode;folder.mkdir()
    owner=dict(mode=mode,status="opening",captures=[],sequence_order=0 if mode=="none" else 1)
    session=None;records={};visibility=[];previous=None;pairs=[]
    try:
        start=perf_counter();session=_backend(library,shaders,index,mode,0).open_scene(cases[0][0]);owner["open_ms"]=(perf_counter()-start)*1000
        for pose,case in enumerate(cases):
            first=None
            for batch in range(1,5):
                shot=progressive._shot(session,case)
                progressive.verify_history(shot["diagnostic"],batches=batch,previous_resets=pose,
                    reason="new_session" if pose==0 else "camera")
                if pose and "transforms" not in shot["diagnostic"]["progressive"]["last_reset_reasons"]:
                    raise ValueError("actual moving foreground transform did not reset history")
                progressive._light_binding(shot["diagnostic"],case[0].lights)
                verify_filter(shot["diagnostic"],mode=mode,batch=batch,filtered_count=pose+1 if mode=="guided" else 0)
                row=_record(shot,batch);row["pose_id"]=pose
                if first is None:first=row
                _geometry_equal(first,row)
                if raw_records is not None:
                    old=raw_records[(pose,batch)];_geometry_equal(old,row)
                    if any(old[k]!=row[k] for k in ("frame","camera")):
                        raise ValueError("two mode input trajectories differ")
                    if batch>1 and old["plane_sha256"][Channel.RGB.value]!=row["plane_sha256"][Channel.RGB.value]:
                        raise ValueError("later public RGB was still filtered")
                if batch in SAVED:
                    _raw(session,row)
                    _save_rgb(root,image_name(mode,pose,batch),shot,row)
                    if raw_records is not None:pairs.append(dict(pose_id=pose,batch=batch,**verify_pair(old,row)))
                if batch==1:
                    if mode=="none":
                        current=_arrays(shot["planes"])
                        path=root/f"geometry/frame-{pose:04d}.npz";np.savez_compressed(path,**current)
                        visibility.append(dict(pose_id=pose,geometry_file=path.relative_to(root).as_posix(),
                            geometry_sha256=_hash(path),**visibility_counts(previous,current)))
                        previous={k:v.copy() for k,v in current.items()}
                    else:_guide(root,f"guided/frame-{pose:04d}",session,row)
                records[(pose,batch)]=row;owner["captures"].append(row)
        owner.update(status="complete",timing_all=_timing([r["capture_ms"] for r in owner["captures"]]),
            timing_motion_first_batches=_timing([r["capture_ms"] for r in owner["captures"] if r["batch"]==1]),
            timing_motion_resets_after_initial=_timing([r["capture_ms"] for r in owner["captures"] if r["batch"]==1 and r["pose_id"]>0]),
            timing_settled_batches_2_to_4=_timing([r["capture_ms"] for r in owner["captures"] if r["batch"]>1]),
            initial_first_capture_ms=owner["captures"][0]["capture_ms"],
            audit_raw_readback_ms=sum(r.get("raw_readback_ms",0.) for r in owner["captures"]),
            audit_guide_readback_ms=sum(r.get("guide",{}).get("readback_ms",0.) for r in owner["captures"]),
            png_save_ms=sum(r.get("png_save_ms",0.)+r.get("guide",{}).get("save_ms",0.) for r in owner["captures"]),
            visibility=visibility,pairs=pairs)
    except BaseException as error:
        owner.update(status="failed",error=dict(type=type(error).__name__,message=str(error)));raise
    finally:progressive._close(session,owner,folder/"receipt.json")
    _owner_check(owner,100,75,25 if mode=="guided" else 0)
    return records,owner


def _references(root,cases,raw_records,library,shaders,index):
    folder=root/"reference";folder.mkdir()
    owner=dict(status="opening",seed=1,captures=[]);session=None;result={}
    try:
        session=_backend(library,shaders,index,"none",1).open_scene(cases[0][0])
        for ordinal,pose in enumerate(REFERENCES):
            for batch in range(1,17):
                shot=progressive._shot(session,cases[pose])
                progressive.verify_history(shot["diagnostic"],batches=batch,previous_resets=ordinal)
                verify_filter(shot["diagnostic"],mode="none",batch=batch,filtered_count=0)
                row=_record(shot,batch);row["pose_id"]=pose
                _geometry_equal(raw_records[(pose,1)],row)
                if batch==16:
                    _raw(session,row);_save_rgb(root,image_name("reference",pose,16),shot,row)
                    verify_reference(raw_records[(pose,1)],row);result[pose]=row
                owner["captures"].append(row)
        owner["status"]="complete"
    except BaseException as error:
        owner.update(status="failed",error=dict(type=type(error).__name__,message=str(error)));raise
    finally:progressive._close(session,owner,folder/"receipt.json")
    _owner_check(owner,48,3,0)
    return result


def _fresh(root,cases,source,library,shaders,index,pose,mode):
    folder=root/f"fresh/pose-{pose:04d}/{mode}";folder.mkdir(parents=True)
    owner=dict(status="opening",pose_id=pose,mode=mode,captures=[]);session=None
    try:
        session=_backend(library,shaders,index,mode,0).open_scene(cases[pose][0])
        for batch in range(1,5):
            shot=progressive._shot(session,cases[pose])
            progressive.verify_history(shot["diagnostic"],batches=batch,previous_resets=0,reason="new_session")
            verify_filter(shot["diagnostic"],mode=mode,batch=batch,filtered_count=1 if mode=="guided" else 0)
            row=_record(shot,batch);row["pose_id"]=pose;old=source[(pose,batch)]
            if any(row[k]!=old[k] for k in ("frame","camera","plane_sha256")):
                raise ValueError("continuous history reset differs from fresh same-state publication")
            if batch in SAVED:
                _raw(session,row)
                if row["raw_mean_sha256"]!=old["raw_mean_sha256"]:raise ValueError("fresh raw mean differs")
                name=f"fresh/pose-{pose:04d}/{mode}/b{batch:04d}.png";_save_rgb(root,name,shot,row)
            if batch==1 and mode=="guided":_guide(root,f"fresh/pose-{pose:04d}/{mode}/first",session,row)
            owner["captures"].append(row)
        owner.update(status="complete",all_planes_identical=True,raw_mean_identical=True)
    except BaseException as error:
        owner.update(status="failed",error=dict(type=type(error).__name__,message=str(error)));raise
    finally:progressive._close(session,owner,folder/"receipt.json")
    _owner_check(owner,4,3,1 if mode=="guided" else 0)
    return dict(pose_id=pose,mode=mode,file=(folder/"receipt.json").relative_to(root).as_posix(),
                sha256=_hash(folder/"receipt.json"),all_planes_identical=True,raw_mean_identical=True,close=owner["close"])


def _mask_inputs(current,previous):
    result=dict(instance_id=current["INSTANCE_ID"],depth=current["FORWARD_DEPTH"],valid=current["VALID"],blocker_ids=(BLOCKER,))
    if previous is not None:result.update(previous_instance_id=previous["INSTANCE_ID"],previous_depth=previous["FORWARD_DEPTH"],previous_valid=previous["VALID"])
    return result


def _scores(root):
    from PIL import Image
    def rgb(name):
        with Image.open(root/name) as im:return np.asarray(im.convert("RGB"))
    results={};scores={str(b):[] for b in SAVED};visibility={}
    (root/"masks").mkdir()
    for pose in REFERENCES:
        current=_load_geometry(root,pose);previous=None if pose==0 else _load_geometry(root,pose-1)
        inputs=_mask_inputs(current,previous);masks=metrics.build_masks(**inputs)
        visibility[str(pose)]=visibility_counts(previous,current)
        if int(masks["reveal"].sum())!=visibility[str(pose)]["revealed_foreground"]:
            raise ValueError("independent reveal definition differs from actual-motion ledger")
        for name,mask in dict(reveal_all=masks["reveal"],reveal_right_158=masks["reveal"] & (current["INSTANCE_ID"]==158),edge=masks["edge"]).items():
            Image.fromarray(mask.astype(np.uint8)*255).save(root/f"masks/pose-{pose:04d}-{name}.png")
        for batch in SAVED:
            scores[str(batch)].append(metrics.score_pose(pose_id=pose,
                raw_rgb8=rgb(image_name("none",pose,batch)),filtered_rgb8=rgb(image_name("guided",pose,batch)),
                reference_rgb8=rgb(image_name("reference",pose,16)),**inputs))
    for batch in SAVED:
        report=metrics.summarize_scores(scores[str(batch)],registered_pose_ids=list(REFERENCES),required_reveal_pose_ids=[15])
        report["interpretation"]=("first/reset displayed filter quality gate" if batch==1 else
            "unfiltered settled control: ratio 1 is expected and does not meet the separate 0.8 first-batch improvement gate")
        results[str(batch)]=report
    target_supported=visibility["15"]["revealed_right_finger"]>=16
    result=dict(batches=results,registered_visibility=visibility,right_finger_reveal_supported=target_supported,
        status="passed" if results["1"]["status"]=="passed" and target_supported else
            "insufficient_evidence" if not target_supported or results["1"]["status"]=="insufficient_evidence" else "failed",
        scope="only the three predeclared poses have a matched independent finite reference; other 22 poses are not scored against a fabricated target")
    _json(root/"metrics.json",result)
    return result


def webp_available(features):
    # Pillow 12 removed webp_anim; WebP is the installed codec module.
    return features.check_module("webp")


def _media(root,destination=None):
    from PIL import Image,ImageDraw,ImageFont,features
    destination=root if destination is None else destination
    font=ImageFont.load_default(size=20);outputs=[]
    width,height=512,384
    for pose in REFERENCES:
        for batch in SAVED:
            canvas=Image.new("RGB",(3*width+4,height+48),(22,24,29))
            items=[(image_name("none",pose,batch),f"RAW | {16*batch} SPP"),
                   (image_name("guided",pose,batch),f"GUIDED MODE | {16*batch} SPP"),
                   (image_name("reference",pose,16),"INDEPENDENT SEED | 256 SPP")]
            for i,(name,label) in enumerate(items):
                with Image.open(root/name) as im:canvas.paste(im.crop(original.CROP),(i*(width+2),48))
                ImageDraw.Draw(canvas).text((i*(width+2)+8,15),label,font=font,fill="white")
            name=f"pose-{pose:04d}-b{batch:04d}-comparison.png";canvas.save(destination/name);outputs.append(name)
    animations={}
    if webp_available(features):
        durations=[round((i+1)*1000/12)-round(i*1000/12) for i in POSES]
        for mode in ("none","guided"):
            frames=[]
            for pose in POSES:
                with Image.open(root/image_name(mode,pose,1)) as im:frames.append(im.convert("RGB").copy())
            name=f"motion-{mode}.webp"
            frames[0].save(destination/name,save_all=True,append_images=frames[1:],duration=durations,loop=0,lossless=True,method=6)
            with Image.open(destination/name) as animation:
                if animation.n_frames!=25:raise ValueError("lossless animation lost motion poses")
                actual=[]
                for i,source in enumerate(frames):
                    animation.seek(i);decoded=animation.convert("RGB")
                    if decoded.tobytes()!=source.tobytes():raise ValueError("lossless animation changed public RGB")
                    actual.append(animation.info["duration"])
                if actual!=durations:raise ValueError("animation viewing timing differs")
            animations[mode]=dict(file=name,sha256=_hash(destination/name),frames=25,durations_ms=durations,
                size=[1280,960],lossless_decode_exact=True,scope="12-Hz viewing rhythm, not measured renderer throughput")
    else:animations["not_verified"]="This Pillow build has no animated WebP; all original lossless PNG frames remain available"
    return dict(comparisons=outputs,animations=animations,crop=list(original.CROP),resized=False)


def rebuild_media(source,destination):
    """Re-encode bound saved PNGs without opening a backend or editing originals."""
    from PIL import Image
    source=Path(source).resolve();destination=Path(destination).resolve()
    summary=json.loads((source/"summary.json").read_text(encoding="utf-8-sig"))
    if summary.get("status") not in ("complete","quality_gate_failed") or summary.get("protocol_sha256")!=_hash(source/"protocol.json"):
        raise ValueError("complete bound capture required for media-only export")
    bindings={}
    for mode in ("none","guided","reference"):
        receipt=source/mode/"receipt.json"
        owner=json.loads(receipt.read_text(encoding="utf-8-sig"))
        if owner.get("status")!="complete":raise ValueError("incomplete capture owner")
        for row in owner["captures"]:
            if "png" not in row:continue
            path=source/row["png"]
            if _hash(path)!=row["png_sha256"]:raise ValueError("original PNG bytes changed")
            with Image.open(path) as im:
                if im.size!=(1280,960) or sha256(im.convert("RGB").tobytes()).hexdigest()!=row["plane_sha256"][Channel.RGB.value]:
                    raise ValueError("saved PNG no longer equals the actual public RGB")
            bindings[row["png"]]=row["png_sha256"]
    destination.mkdir(parents=True,exist_ok=False)
    report=dict(schema="phymeo-guided-motion-media-v1",status="encoding",capture_directory=str(source),
        capture_summary_sha256=_hash(source/"summary.json"),capture_protocol_sha256=_hash(source/"protocol.json"),
        script_sha256=_hash(__file__),source_png_sha256=bindings,new_public_captures=0)
    try:
        report["media"]=_media(source,destination)
        if any(_hash(source/name)!=digest for name,digest in bindings.items()):raise ValueError("original PNG changed during encoding")
        report["status"]="complete" if set(report["media"]["animations"])=={"none","guided"} else "media_not_verified"
    except BaseException as error:
        report.update(status="failed",error=dict(type=type(error).__name__,message=str(error)));raise
    finally:_json(destination/"receipt.json",report)
    return report

def run(output,library,*,device_index=1,shader_directory=None):
    output=Path(output).resolve();library=Path(library).resolve();output.mkdir(parents=True,exist_ok=False)
    shaders=library.parent if shader_directory is None else Path(shader_directory).resolve()
    report=dict(schema="phymeo-guided-motion-study-v1",status="preparing",physics_steps=0)
    try:
        cases=build_motion(output/"geometry")
        helpers=("diffuse_bounce_study.py","progressive_panda_study.py","guided_motion_metrics.py","_panda_scene.py","normal_map_film.py","panda_msaa_compare.py")
        protocol=dict(schema="phymeo-guided-motion-protocol-v1",profile=frozen_profile(),scene=asdict(cases[0][0]),
            runtime=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__),
            poses=[dict(pose_id=i,frame=asdict(f),camera=asdict(c)) for i,(_,f,c) in enumerate(cases)],
            source=dict(script_sha256=_hash(__file__),helpers={n:_hash(ROOT/n) for n in helpers},implementation=_implementation_files()),
            library=dict(file=str(library),sha256=_hash(library)),shaders={p.name:_hash(p) for p in sorted(shaders.glob("*.spv"))},
            device_index=device_index,mesh_sha256={m.id:original.load_mesh(m).source_sha256 for m in cases[0][0].meshes},
            panda_provenance=original._panda_provenance(cases[0][0]))
        _json(output/"protocol.json",protocol)
        report["stage"]="none";raw,raw_owner=_sequence(output,cases,library,shaders,device_index,"none")
        report["stage"]="guided";guided,guided_owner=_sequence(output,cases,library,shaders,device_index,"guided",raw)
        report["stage"]="reference";_references(output,cases,raw,library,shaders,device_index)
        fresh=[]
        for pose in REFERENCES:
            for mode,source in (("none",raw),("guided",guided)):
                report["stage"]=f"fresh-{pose}-{mode}"
                fresh.append(_fresh(output,cases,source,library,shaders,device_index,pose,mode))
        report["stage"]="analysis";quality=_scores(output)
        media=_media(output)
        if (protocol["source"]["script_sha256"]!=_hash(__file__) or protocol["source"]["implementation"]!=_implementation_files()
            or any(_hash(ROOT/n)!=h for n,h in protocol["source"]["helpers"].items())
            or protocol["mesh_sha256"]!={m.id:original.load_mesh(m).source_sha256 for m in cases[0][0].meshes}
            or protocol["library"]["sha256"]!=_hash(library)
            or protocol["shaders"]!={p.name:_hash(p) for p in sorted(shaders.glob("*.spv"))}):
            raise ValueError("bound source or device build changed during execution")
        owners={mode:dict(file=f"{mode}/receipt.json",sha256=_hash(output/f"{mode}/receipt.json"),
            open_ms=owner["open_ms"],initial_first_capture_ms=owner["initial_first_capture_ms"],
            timing_motion_first_batches=owner["timing_motion_first_batches"],
            timing_motion_resets_after_initial=owner["timing_motion_resets_after_initial"],
            timing_settled_batches_2_to_4=owner["timing_settled_batches_2_to_4"],
            audit_raw_readback_ms=owner["audit_raw_readback_ms"],audit_guide_readback_ms=owner["audit_guide_readback_ms"],
            png_save_ms=owner["png_save_ms"],close=owner["close"]) for mode,owner in (("none",raw_owner),("guided",guided_owner))}
        report.update(status="complete" if quality["status"]=="passed" else "quality_gate_failed",stage="complete",
            protocol_sha256=_hash(output/"protocol.json"),quality=quality,metrics_sha256=_hash(output/"metrics.json"),
            visibility=raw_owner["visibility"],mode_order=frozen_profile()["mode_order"],owners=owners,fresh=fresh,media=media,
            actual_public_captures=272,audit_raw_readbacks=171,audit_guide_readbacks=28,
            counters_scope="nine owners independently checked against exact capture/readback totals; first guided batch adds a separate direct-guide draw in the same public exposure; later batches omit that draw",
            geometry_and_raw_mean_identical=True,settled_public_rgb_identical=True,
            limitations=frozen_profile()["limits"],unsupported="only this progressive one-bounce, MSAA1, native ray_query configuration is exercised")
    except BaseException as error:
        report.update(status="failed",error=dict(type=type(error).__name__,message=str(error)));raise
    finally:_json(output/"summary.json",report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--library",type=Path)
    p.add_argument("--media-only",type=Path,help="Existing completed capture; encode into a fresh output without rendering")
    p.add_argument("--device-index",type=int,default=1);p.add_argument("--shader-directory",type=Path)
    args=p.parse_args()
    if args.media_only is not None:
        if args.library is not None:p.error("media-only does not open a rendering library")
        result=rebuild_media(args.media_only,args.output)
    else:
        if args.library is None:p.error("capture requires --library")
        result=run(args.output,args.library,device_index=args.device_index,shader_directory=args.shader_directory)
    print(json.dumps(dict(status=result["status"],output=str(args.output.resolve()))))
    if result["status"]!="complete":raise SystemExit(2)


if __name__=="__main__":main()