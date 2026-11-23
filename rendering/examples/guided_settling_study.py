"""Matched settling checkpoints on three entered states of the original Panda input.

This is a finite optical experiment, not playback of the intervening 25-pose
motion. Every public image is retained at its original resolution; the driver
never filters images. Independent error/drift scoring is in settling_scores.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

import guided_motion_study as motion
import guided_motion_metrics as geometry_metrics
from normal_map_film import _json
from panda_msaa_compare import _hash, _implementation_files, _timing
from phymeo_rendering.outputs import Channel

ROOT=Path(__file__).resolve().parent
POSES=(0,15,24)
SAVED=(1,2,4,8,16,17)
SCORED=(1,2,4,8,16)
NEEDED=(0,14,15,23,24)


def frozen_profile():
    return dict(poses=list(POSES),saved_batches=list(SAVED),scored_batches=list(SCORED),
        full_batches_per_pose=17,reference_batches=64,seed=0,reference_seed=1,fresh_pose=15,
        expected_public_captures=379,expected_raw_reads=190,expected_guide_reads=20,
        mode_order=["raw","old","settling","reference","fresh-old","fresh-settling"],
        public_modes={"raw":"none","old":"guided","settling":"guided_settling"},
        trajectory_scope="three entered states from the registered 25-pose input; not replay of the intervening motion",
        size=[1280,960],msaa_samples=1,bounces=1,exposure=1.,display_transform="pbr-neutral",
        shadow_mode="ray_query",primary_samples_per_batch=16,
        schedule="w=1-sqrt((n-1)/15), one-based n<16; n>=16 exactly zero",
        native_weight_arithmetic="float32 divide, float32 sqrt, float32 subtraction; scheduled_filtered_fraction retains the separate FP64 formula",
        source_geometry="current 0/15/24 and predecessor 14/23 from bound actual 4c evidence; full current arrays must match before masks are used",
        guide_source="same-exposure-gpu-direct-pass",guide_check_batches=[1,2,4,8],
        reference_scope="new same-state native seed1 64 batches=1024 spp; a finite reference, not exact radiance",
        display_error_space="inverse-sRGB display-linear",warmup=0,
        timing="mode order fixed; open, state-reset first exposure, later captures and explicit debug reads/saves reported separately; request-to-return wall time, not pure GPU time")


def build_inputs(directory):
    return motion.build_motion(directory)


def image_name(label,pose,n):
    if label not in ("raw","old","settling","reference") or type(pose) is not int or pose not in POSES:
        raise ValueError("registered label and pose required")
    if label=="reference":
        if n!=64:raise ValueError("reference must be the complete independent 1024-spp capture")
        return f"reference/pose-{pose:04d}.png"
    if n not in SAVED:raise ValueError("unregistered display checkpoint")
    return f"{label}/pose-{pose:04d}-b{n:04d}.png"


def verify_settling(diagnostic,*,n,ordinal):
    if type(n) is not int or not 1<=n<=17 or type(ordinal) is not int or ordinal<0:
        raise ValueError("registered actual batch/state ordinal required")
    k=min(n,15);w=1-math.sqrt((n-1)/15) if n<16 else 0.
    fp=float(np.float32(1)-np.sqrt(np.float32(n-1)/np.float32(15))) if n<16 else 0.
    count=ordinal*15+k;represented=ordinal*1920+8*k*(k+1)
    top=dict(mode="guided_settling",scope="settling-first-16-rgb-batches",last_capture_filtered=n<16,
        filtered_captures=count,last_filtered_fraction=fp,scheduled_filtered_fraction=w,last_batch=n,
        filtered_primary_samples=represented,last_filtered_primary_samples=16*n if n<16 else 0,
        guide_source="same-exposure-gpu-direct-pass")
    actual=diagnostic.get("spatial_filter",{})
    if any(actual.get(name)!=value for name,value in top.items()):raise ValueError("actual settling weight/phase/cumulative exposure receipt differs")
    expected=dict(settling_abi_version=1,enabled=1,last_batch=n,last_capture_filtered=int(n<16),
        last_filtered_fraction=fp,last_filtered_primary_samples=16*n if n<16 else 0,guide_valid=1,
        guide_draws=ordinal+1,filtered_rgb_captures=count,filtered_primary_samples=represented,filter_dispatches=count)
    device=diagnostic.get("device",{}).get("resident_settling_info",{})
    if any(device.get(name)!=value for name,value in expected.items()):raise ValueError("actual guide reuse/settling device counters differ")
    return True


def compare_stage(raw,old,new):
    for other in (old,new):
        if any(other.get(k)!=raw.get(k) for k in ("frame","camera","batch")):
            raise ValueError("matched settling inputs or accumulation phase differ")
        motion._geometry_equal(raw,other)
        if not isinstance(raw.get("raw_mean_sha256"),str) or raw["raw_mean_sha256"]!=other.get("raw_mean_sha256"):
            raise ValueError("display settling altered actual raw mean")
    n=raw["batch"];key=Channel.RGB.value
    old_same=old["plane_sha256"][key]==raw["plane_sha256"][key]
    new_same=new["plane_sha256"][key]==raw["plane_sha256"][key]
    pair_same=old["plane_sha256"][key]==new["plane_sha256"][key]
    if (n==1 and not pair_same) or (n>1 and not old_same) or (n>=16 and not new_same):
        raise ValueError("first or zero-tail public RGB identity differs")
    return dict(raw_mean_identical=True,geometry_identical=True,rgb_old_new_identical=pair_same,
                old_equals_raw=old_same,new_equals_raw=new_same)


def verify_cached_guide(first,current):
    if any(first.get(k)!=current.get(k) for k in ("raw_sha256","metadata")):
        raise ValueError("same-state cached guide bytes changed after first batch")
    return True


def _plain(value):
    return json.loads(json.dumps(value,allow_nan=False))


def _scene_binding(scene,hashes):
    result=deepcopy(scene)
    for mesh in result["meshes"]:mesh["uri"]="sha256:"+hashes[mesh["id"]]
    return _plain(result)


def bind_motion_source(cases,source):
    source=Path(source).resolve()
    summary=json.loads((source/"summary.json").read_text(encoding="utf-8-sig"))
    protocol=json.loads((source/"protocol.json").read_text(encoding="utf-8-sig"))
    if summary.get("status")!="complete" or summary.get("protocol_sha256")!=_hash(source/"protocol.json"):
        raise ValueError("complete bound original motion source required")
    hashes={m.id:motion.original.load_mesh(m).source_sha256 for m in cases[0][0].meshes}
    actual=[dict(pose_id=i,frame=asdict(f),camera=asdict(c)) for i,(_,f,c) in enumerate(cases)]
    if (len(cases)!=25 or protocol["poses"]!=_plain(actual) or hashes!=protocol["mesh_sha256"]
        or _scene_binding(protocol["scene"],protocol["mesh_sha256"])!=_scene_binding(asdict(cases[0][0]),hashes)):
        raise ValueError("original full motion input, camera, light, material or mesh bytes differ")
    rows={r["pose_id"]:r for r in summary["visibility"]};geometry={};files={}
    for pose in NEEDED:
        name=f"geometry/frame-{pose:04d}.npz"
        if pose not in rows or rows[pose].get("geometry_file")!=name or rows[pose].get("geometry_sha256")!=_hash(source/name):
            raise ValueError("actual previous/current geometry bytes are not bound to source motion")
        data=motion._load_geometry(source,pose)
        for key,dtype in (("INSTANCE_ID",np.dtype("<u4")),("FORWARD_DEPTH",np.dtype("<f4")),("VALID",np.dtype("u1"))):
            if data[key].shape!=(960,1280) or data[key].dtype!=dtype:raise ValueError("source geometry layout differs")
        geometry[pose]=data;files[str(pose)]=dict(file=name,sha256=_hash(source/name))
    return dict(source_directory=str(source),summary_sha256=_hash(source/"summary.json"),
        protocol_sha256=_hash(source/"protocol.json"),full_input_equal=True,predecessor_for_pose15=14,
        geometry=files,uri_relocation="mesh URI roots may change; every mesh byte hash and other scene field is identical"),geometry


def _geometry_check(row,geometry):
    keys={Channel.INSTANCE_ID:"INSTANCE_ID",Channel.FORWARD_DEPTH:"FORWARD_DEPTH",Channel.VALID:"VALID"}
    if any(row["plane_sha256"][channel.value]!=sha256(geometry[name].tobytes()).hexdigest() for channel,name in keys.items()):
        raise ValueError("current actual native geometry differs from original matched pose; prior disocclusion cannot be reused")


def _take_sequence(root,cases,prior,library,shaders,index,label,*,seed=0,poses=POSES,batches=17,source=None,raw_source=None,old_source=None):
    mode={"raw":"none","old":"guided","settling":"guided_settling","reference":"none"}[label.split('-')[-1] if label.startswith('fresh-') else label]
    folder=root/label;folder.mkdir()
    owner=dict(status="opening",label=label,mode=mode,seed=seed,captures=[],pairs=[]);session=None;records={}
    try:
        start=perf_counter();session=motion._backend(library,shaders,index,mode,seed).open_scene(cases[0][0]);owner["open_ms"]=(perf_counter()-start)*1000
        for ordinal,pose in enumerate(poses):
            first_guide=None
            for n in range(1,batches+1):
                shot=motion.progressive._shot(session,cases[pose])
                motion.progressive.verify_history(shot["diagnostic"],batches=n,previous_resets=ordinal,
                    reason="new_session" if ordinal==0 else "camera")
                motion.progressive._light_binding(shot["diagnostic"],cases[pose][0].lights)
                if mode=="guided_settling":verify_settling(shot["diagnostic"],n=n,ordinal=ordinal)
                else:motion.verify_filter(shot["diagnostic"],mode=mode,batch=n,filtered_count=ordinal+1 if mode=="guided" else 0)
                row=motion._record(shot,n);row.update(pose_id=pose,settling_device=shot["diagnostic"]["device"].get("resident_settling_info"))
                _geometry_check(row,prior[pose])
                # Every main/fresh batch hashes its actual raw mean; reference only needs its complete endpoint.
                if label!="reference" or n==64:motion._raw(session,row)
                if source is not None:
                    old=source[(pose,n)]
                    if any(old[k]!=row[k] for k in ("frame","camera","plane_sha256","raw_mean_sha256")):
                        raise ValueError("state entered after history differs from fresh same-state/seed capture")
                if raw_source is not None:
                    raw=raw_source[(pose,n)]
                    motion._geometry_equal(raw,row)
                    if any(raw[k]!=row[k] for k in ("frame","camera","raw_mean_sha256")):raise ValueError("actual modes have different raw/input state")
                    if label=="old" and n>1 and row["plane_sha256"][Channel.RGB.value]!=raw["plane_sha256"][Channel.RGB.value]:raise ValueError("old guided cutoff changed")
                    if old_source is not None:owner["pairs"].append(dict(pose_id=pose,batch=n,**compare_stage(raw,old_source[(pose,n)],row)))
                if (mode=="guided" and n==1) or (mode=="guided_settling" and n in (1,2,4,8)):
                    motion._guide(root,f"{label}/pose-{pose:04d}-b{n:04d}",session,row)
                    if first_guide is None:first_guide=row["guide"]
                    else:verify_cached_guide(first_guide,row["guide"])
                save=n==64 if label=="reference" else n in SAVED
                if save:
                    name=(f"{label}/pose-{pose:04d}-b{n:04d}.png" if label.startswith("fresh-") else image_name(label,pose,n))
                    motion._save_rgb(root,name,shot,row)
                if n==1 and label=="raw":
                    path=root/f"geometry/current-pose-{pose:04d}.npz"
                    np.savez_compressed(path,**motion._arrays(shot["planes"]))
                    row["geometry_file"]=path.relative_to(root).as_posix();row["geometry_sha256"]=_hash(path)
                records[(pose,n)]=row;owner["captures"].append(row)
        owner.update(status="complete",timing_all=_timing([r["capture_ms"] for r in owner["captures"]]),
            timing_first_batches=_timing([r["capture_ms"] for r in owner["captures"] if r["batch"]==1]),
            timing_later_batches=_timing([r["capture_ms"] for r in owner["captures"] if r["batch"]>1]),
            initial_first_capture_ms=owner["captures"][0]["capture_ms"],
            stage_timings={str(n):_timing([r["capture_ms"] for r in owner["captures"] if r["batch"]==n]) for n in (64,) if label=="reference"} if label=="reference" else {str(n):_timing([r["capture_ms"] for r in owner["captures"] if r["batch"]==n]) for n in SAVED},
            audit_raw_readback_ms=sum(r.get("raw_readback_ms",0.) for r in owner["captures"]),
            audit_guide_readback_ms=sum(r.get("guide",{}).get("readback_ms",0.) for r in owner["captures"]),
            png_save_ms=sum(r.get("png_save_ms",0.)+r.get("guide",{}).get("save_ms",0.) for r in owner["captures"]))
    except BaseException as error:
        owner.update(status="failed",error=dict(type=type(error).__name__,message=str(error)));raise
    finally:motion.progressive._close(session,owner,folder/"receipt.json")
    reads=len(poses) if label=="reference" else len(poses)*batches
    guides=len(poses)*(4 if mode=="guided_settling" else 1 if mode=="guided" else 0)
    motion._owner_check(owner,len(poses)*batches,reads,guides)
    return records,owner


def _score(root,prior):
    from PIL import Image
    from settling_scores import score_settling
    def rgb(path):
        with Image.open(path) as im:return np.asarray(im.convert("RGB"))
    old={p:{n:rgb(root/image_name("old",p,n)) for n in SCORED} for p in POSES}
    new={p:{n:rgb(root/image_name("settling",p,n)) for n in SCORED} for p in POSES}
    refs={p:rgb(root/image_name("reference",p,64)) for p in POSES}
    geometry={p:motion._mask_inputs(prior[p],None if p==0 else prior[p-1]) for p in POSES}
    result=score_settling(old_rgb8=old,new_rgb8=new,reference_rgb8=refs,geometry=geometry)
    (root/"masks").mkdir()
    masks=geometry_metrics.build_masks(**geometry[15])
    for name,mask in dict(reveal_all=masks["reveal"],reveal_right_158=masks["reveal"] & (prior[15]["INSTANCE_ID"]==158),edge=masks["edge"]).items():
        Image.fromarray(mask.astype(np.uint8)*255).save(root/f"masks/pose-0015-{name}.png")
    result["source_visibility"]=motion.visibility_counts(prior[14],prior[15])
    result["reference_samples_per_pixel"]=1024
    _json(root/"scores.json",result)
    return result


def _media(root):
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.load_default(size=20);width,height=512,384;names=[]
    for pose in POSES:
        for n in SAVED:
            canvas=Image.new("RGB",(1540,432),(22,24,29))
            items=[(image_name("old",pose,n),f"OLD GUIDED | BATCH {n}"),
                   (image_name("settling",pose,n),f"SETTLING | BATCH {n}"),
                   (image_name("reference",pose,64),"INDEPENDENT SEED | 1024 SPP")]
            for column,(file,label) in enumerate(items):
                with Image.open(root/file) as im:canvas.paste(im.crop(motion.original.CROP),(column*(width+2),48))
                ImageDraw.Draw(canvas).text((column*(width+2)+8,15),label,font=font,fill="white")
            name=f"pose-{pose:04d}-b{n:04d}-comparison.png";canvas.save(root/name);names.append(name)
    return dict(comparisons=names,comparison_size=[1540,432],original_size=[1280,960],crop=list(motion.original.CROP),
                scope="original pixel crops and exterior labels only; n17 is a preserved unfiltered anchor, not a scored checkpoint; no motion video")


def run(output,library,motion_source,*,device_index=1,shader_directory=None):
    output=Path(output).resolve();library=Path(library).resolve();source=Path(motion_source).resolve()
    output.mkdir(parents=True,exist_ok=False);shaders=library.parent if shader_directory is None else Path(shader_directory).resolve()
    report=dict(schema="phymeo-guided-settling-study-v1",status="preparing",physics_steps=0);started=perf_counter()
    try:
        cases=build_inputs(output/"geometry");binding,prior=bind_motion_source(cases,source)
        (output/"source/geometry").mkdir(parents=True)
        for name in ("summary.json","protocol.json"):(output/"source"/name).write_bytes((source/name).read_bytes())
        for pose in NEEDED:(output/"source/geometry"/f"frame-{pose:04d}.npz").write_bytes((source/f"geometry/frame-{pose:04d}.npz").read_bytes())
        helpers=("guided_motion_study.py","guided_motion_metrics.py","settling_scores.py","progressive_panda_study.py","diffuse_bounce_study.py","_panda_scene.py","normal_map_film.py","panda_msaa_compare.py")
        protocol=dict(schema="phymeo-guided-settling-protocol-v1",profile=frozen_profile(),motion_binding=binding,
            scene=asdict(cases[0][0]),poses=[dict(pose_id=p,frame=asdict(cases[p][1]),camera=asdict(cases[p][2])) for p in POSES],
            runtime=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__),
            script_sha256=_hash(__file__),helpers={n:_hash(ROOT/n) for n in helpers},implementation=_implementation_files(),
            library=dict(file=str(library),sha256=_hash(library)),shaders={p.name:_hash(p) for p in sorted(shaders.glob("*.spv"))},device_index=device_index)
        _json(output/"protocol.json",protocol);report["preparation_ms"]=(perf_counter()-started)*1000
        owners={};report["stage"]="raw"
        raw,owners["raw"]=_take_sequence(output,cases,prior,library,shaders,device_index,"raw")
        report["stage"]="old"
        old,owners["old"]=_take_sequence(output,cases,prior,library,shaders,device_index,"old",raw_source=raw)
        report["stage"]="settling"
        new,owners["settling"]=_take_sequence(output,cases,prior,library,shaders,device_index,"settling",raw_source=raw,old_source=old)
        report["stage"]="reference"
        _,owners["reference"]=_take_sequence(output,cases,prior,library,shaders,device_index,"reference",seed=1,batches=64)
        for label,records in (("old",old),("settling",new)):
            name="fresh-"+label;report["stage"]=name
            _,owners[name]=_take_sequence(output,cases,prior,library,shaders,device_index,name,poses=(15,),source=records)
        report["stage"]="scores";scores=_score(output,prior);media=_media(output)
        if (protocol["script_sha256"]!=_hash(__file__) or protocol["implementation"]!=_implementation_files()
            or any(_hash(ROOT/n)!=h for n,h in protocol["helpers"].items())
            or protocol["library"]["sha256"]!=_hash(library)
            or protocol["shaders"]!={p.name:_hash(p) for p in sorted(shaders.glob("*.spv"))}):raise ValueError("bound source/device changed")
        report.update(status="complete" if scores["status"]=="passed" else "quality_gate_failed",stage="complete",
            protocol_sha256=_hash(output/"protocol.json"),scores_file="scores.json",scores_sha256=_hash(output/"scores.json"),scores=scores,
            owners={k:dict(file=f"{k}/receipt.json",sha256=_hash(output/f"{k}/receipt.json"),
                open_ms=v["open_ms"],initial_first_capture_ms=v["initial_first_capture_ms"],
                timing_first_batches=v["timing_first_batches"],timing_later_batches=v["timing_later_batches"],stage_timings=v["stage_timings"],
                audit_raw_readback_ms=v["audit_raw_readback_ms"],audit_guide_readback_ms=v["audit_guide_readback_ms"],png_save_ms=v["png_save_ms"],close=v["close"]) for k,v in owners.items()},
            actual_captures=379,actual_raw_reads=190,actual_guide_reads=20,raw_and_geometry_exact=True,first_and_tail_exact=True,
            guide_reused_exactly=True,fresh_after_prior_history_exact=True,media=media,mode_order=frozen_profile()["mode_order"],
            scope="all registered poses/checkpoints retained; query counters include full17 batches; filtered sample counters represent reused cumulative exposures, not independent new paths",
            supported_scope="actual native resident ray_query/progressive/MSAA1/one-bounce; other paths not exercised")
    except BaseException as error:
        report.update(status="failed",error=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        report["total_wall_ms"]=(perf_counter()-started)*1000
        _json(output/"summary.json",report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--library",type=Path,required=True)
    p.add_argument("--motion-source",type=Path,required=True);p.add_argument("--device-index",type=int,default=1)
    p.add_argument("--shader-directory",type=Path)
    a=p.parse_args();result=run(a.output,a.library,a.motion_source,device_index=a.device_index,shader_directory=a.shader_directory)
    print(json.dumps(dict(status=result["status"],output=str(a.output.resolve()))))
    if result["status"]!="complete":raise SystemExit(2)


if __name__=="__main__":main()