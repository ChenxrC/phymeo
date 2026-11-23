"""Real public sparse capture delivery; no renderer/shader changes.

One invocation activates exactly one prepared private runtime. Full-rate baseline
and sparse candidate have separate owners, DLL/SPV identities and profile deltas.
"""
import argparse
from dataclasses import asdict,replace
from hashlib import sha256
import candidate_adapter as adapter
import json
import math
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

EXAMPLES=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXAMPLES))
import panda_motion_preview as base

SCHEMA="phymeo-preview-sparse-motion-v1"
REFERENCE_SCHEMA="phymeo-preview-sparse-reference-v1"
REGISTERED=(59,60,149)
SIZES={"720p":"preview720","540p":"fast540","360p":"fast360"}
IDENTITY=("pose_index","frame_key","time_s","state_sha256","camera_sha256")
require=base.require
digest=base.digest
save=base.save
canonical=base.canonical


def indices(quick,poses):
    require(all(type(p) is int and 0<=p<300 for p in poses),"pose must be an episode index0..299")
    return tuple(sorted(set(base.pose_indices(quick))|set(poses)))


def jobs(channels,round_number,*,candidate_only=False):
    rows=[(name,c) for name in (("sparse",) if candidate_only else ("baseline16","sparse")) for c in channels]
    return rows if round_number==1 else list(reversed(rows))


def statistics(values,*,full):
    require(values and all(math.isfinite(x) and x>0 for x in values),"invalid completed capture duration")
    require(not full or len(values)==300,"full run needs300 completed states")
    ordered=sorted(values);n=len(values);mean=sum(values)/n
    p95=ordered[math.ceil(.95*n)-1]
    hits=sum(x<=33.3 for x in values)
    return dict(count=n,mean_ms=mean,p50_ms=ordered[math.ceil(.5*n)-1],p95_ms=p95,max_ms=ordered[-1],
        completed_fps_from_mean=1000/mean,within_33_3_ms=hits,within_33_3_fraction=hits/n,
        latency_gate_passed=p95<=33.3 if full else None,
        scope="completed synchronous public capture including CPU-owned requested planes; not presentation/GPU-only")


def identity_audit(frames,expected_indices):
    require([f["pose_index"] for f in frames]==list(expected_indices),"source index order/count differs")
    keys=[canonical(f["frame_key"]) for f in frames]
    require(len(set(keys))==len(keys),"duplicate published frame identity")
    require(all(a["time_s"]<b["time_s"] for a,b in zip(frames,frames[1:])),"published source time is not strictly increasing")
    return dict(count=len(frames),unique_frame_keys=len(set(keys)),unique_source_indices=len(set(expected_indices)),
        first_pose=frames[0]["pose_index"],last_pose=frames[-1]["pose_index"],
        scope="unique source/frame identities, not a claim that all physical joint configurations differ")


def phase_statistics(frames):
    phases=(("before_finger_window",lambda t:t<.5),
            ("finger_view_window",lambda t:.5<=t<2.3),
            ("after_finger_window",lambda t:t>=2.3))
    result={}
    for label,predicate in phases:
        rows=[f for f in frames if predicate(f["time_s"])]
        if not rows:
            result[label]=dict(count=0,timing=None);continue
        timing=statistics([f["capture_ms"] for f in rows],full=False)
        keys=set.intersection(*(set(f["native_profile"]["total_ns"]) for f in rows))
        result[label]=dict(count=len(rows),source_pose_indices=[f["pose_index"] for f in rows],timing=timing,
            native_mean_ms={k:sum(f["native_profile"]["total_ns"][k] for f in rows)/len(rows)/1e6 for k in sorted(keys)})
    return dict(scope="registered movie-view clock windows [0,.5),[.5,2.3),[2.3,10]; not inferred controller phases",
                phases=result)

def profile_delta(before,after):
    return base.profile_delta(before,after,1)


def profile_name(policy):
    name=(adapter.kind(policy)+"-t{tile}-p{paths_per_anchor}-g{area_light_grid}-k{neighbor_candidates}").format(**policy)+f"-r{policy.get('reconstruction_revision',1)}"
    return name if policy.get("estimator","radiance")=="radiance" else name+"-"+policy["estimator"]


def check_actual(actual,policy,expected_captures):
    require(all(actual[k]==policy[k] for k in ("tile","paths_per_anchor","area_light_grid","neighbor_candidates")),
            "sparse actual ABI profile differs from source contract")
    require(actual.get("reconstruction_revision",1)==policy.get("reconstruction_revision",1),
            "sparse reconstruction revision differs")
    require(actual.get("estimator","radiance")==policy.get("estimator","radiance"),"sparse estimator differs")
    require(actual["invalid"]==0 and actual["last_capture_rgb"] and
            actual["rgb_captures"]==expected_captures and actual["sensor_only_captures"]==0,
            "sparse publication/capture counters differ")
    parts=("reconstructed_pixels","fallback_pixels","nondiffuse_pixels")
    if adapter.kind(policy)=="sparse":parts=("traced_anchor_pixels",)+parts
    require(sum(actual[k] for k in parts)==actual["visible_pixels"],"candidate visible pixel partition differs")
    require(actual["primary_proposals"]==actual["traced_anchor_pixels"]*actual["paths_per_anchor"],
            "sparse actual primary proposal counter differs")


def compare(rows):
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            require(a["extent"]==b["extent"] and len(a["frames"])==len(b["frames"]),"paired extent/count differs")
            if "backend" in a and "backend" in b:
                require(a["backend"]["device"]["device_uuid"]==b["backend"]["device"]["device_uuid"] and
                        a["backend"]["decoded_geometry_sha256"]==b["backend"]["decoded_geometry_sha256"],
                        "paired actual device/decoded geometry differs")
                require(a["backend"]["effective_directional_lights"]==b["backend"]["effective_directional_lights"],
                        "paired actual directional light submission differs")
            for x,y in zip(a["frames"],b["frames"],strict=True):
                require(all(x[k]==y[k] for k in IDENTITY),"paired actual frame/camera differs")
                if a["name"]==b["name"]:
                    require(x["rgb_sha256"]==y["rgb_sha256"],"same-mode RGB/all pixels differ")
                if a["channels"]==b["channels"]=="all":
                    require(x["sensors"]==y["sensors"],"full-frame depth/ID/valid differ")


def compare_repeat(a,b):
    compare([a,b])
    require(a["name"]==b["name"] and a["channels"]==b["channels"],"different repeated configuration")
    for x,y in zip(a["frames"],b["frames"],strict=True):
        require(x["rgb_sha256"]==y["rgb_sha256"] and x["sensors"]==y["sensors"],"repeat bytes differ")
        if "actual" in x:require(x["actual"]==y["actual"],"repeat sparse actual counters differ")
        if "compact" in x:require(x["compact"]==y["compact"],"repeat compact actual counters differ")


def apply_look(scene,studio,look):
    require(look in ("recorded","soft-studio"),"unknown studio look")
    if look=="recorded":return scene,studio
    # A separate authored lighting presentation: same HDR/material/exposure,
    # no independent directional source; larger panels at the same total power.
    return replace(scene,key=scene.key+":soft-studio",lights=()),replace(studio,panel_size_scale=2.)

def load_runtime(args):
    module=adapter.load_runtime(args.runtime)
    prepared,manifest=module.activate(args.prepared)
    adapter.require_kind(manifest["contract"],getattr(args,"candidate_backend","sparse"))
    # Runtime import dependencies are frozen before any renderer/device call.
    args.runtime_dependencies=module.motion_adapter_dependencies
    return module,prepared,manifest


def source_binding(args,prepared,manifest):
    from phymeo_rendering.backends import native
    import panda_render_performance as factory
    root=prepared/"build"
    return dict(wrapper_file=str(Path(__file__).resolve()),wrapper_sha256=digest(__file__),
        runtime_adapter_file=str(args.runtime.resolve()),runtime_adapter_sha256=digest(args.runtime),
        delivery_adapter_file=str(Path(adapter.__file__).resolve()),delivery_adapter_sha256=digest(adapter.__file__),
        runtime_dependencies=getattr(args,"runtime_dependencies",[]),candidate_backend=adapter.kind(manifest["contract"]),
        private_source_manifest_file=str(prepared/"source-manifest.json"),
        private_source_manifest_sha256=digest(prepared/"source-manifest.json"),contract=manifest["contract"],
        private_package_file=str(Path(native.__file__).resolve()),private_package_native_sha256=digest(native.__file__),
        capture_helper_sha256=digest(base.__file__),fixture_builder_sha256=digest(factory.__file__),
        baseline=dict(library=str(args.baseline_library),library_sha256=digest(args.baseline_library),
            shaders={p.name:digest(p) for p in sorted(args.baseline_shaders.glob("*.spv"))}),
        sparse=dict(library=str(root/"phymeo_native.dll"),library_sha256=digest(root/"phymeo_native.dll"),
            shaders={p.name:digest(p) for p in sorted(root.glob("*.spv"))}))


def open_owner(args,scene,studio,runtime,prepared,policy,name,*,seed=0,reference=False):
    if name=="sparse":
        return adapter.backend_class(runtime,policy)(prepared,device_index=args.device_index,studio=studio,seed=seed,
            exposure=8.,display_transform="pbr-neutral",reflection_mode="single_bounce",validation=args.validation).open_scene(scene)
    from phymeo_rendering.backends.native import NativeBackend
    settings=base.preset(SIZES[args.resolution])["settings"]
    settings.update(area_light_grid=policy["area_light_grid"],indirect_seed=seed,
                    spatial_filter="none" if reference else "guided_settling")
    return NativeBackend(library_path=args.baseline_library,shader_directory=args.baseline_shaders,
        device_index=args.device_index,studio=studio,exposure=8.,display_transform="pbr-neutral",
        validation=args.validation,**settings).open_scene(scene)


def check_backend(diag,args,policy,name,source,*,expected_captures):
    require(diag["device"]["resident_upload_calls"]==1 and
            diag["device"]["resident_captures"]==expected_captures,"owner upload/capture count differs")
    bound=source["baseline" if name=="baseline16" else "sparse"]
    require(diag["library_sha256"]==bound["library_sha256"],"loaded DLL is outside bound profile")
    require(diag["shader_sha256"] and all(v in bound["shaders"].values() for v in diag["shader_sha256"].values()),
            "loaded shader is outside bound profile")
    if name=="sparse":
        require(diag["backend"]==adapter.backend_name(policy) and diag["history"]=="none"
                and diag["contract"]==policy and diag["library_sha256"]==bound["library_sha256"],
                "candidate public runtime identity/source contract differs")
        check_actual(diag["actual"],policy,expected_captures)
        if adapter.kind(policy)=="compact":
            adapter.check_compact(diag["compact"],base.preset(SIZES[args.resolution])["extent"],
                                  expected_captures,diag["actual"])
    else:
        p=base.preset(SIZES[args.resolution]);p["settings"]["area_light_grid"]=policy["area_light_grid"]
        base.verify_backend(diag,p)
        require(diag["progressive"]["batches"]==1,"moving baseline did not reset to one batch")


def write_buffers(folder,extent,pngs,geometries,records):
    import numpy as np
    from PIL import Image
    started=perf_counter();width,height=extent
    for ordinal,data in pngs:
        path=folder/"frames"/f"{ordinal:04d}.png";path.parent.mkdir(exist_ok=True)
        Image.frombytes("RGB",(width,height),data).save(path)
        records[ordinal].update(png=str(path.relative_to(folder)),png_sha256=digest(path))
    for ordinal,planes in geometries:
        from phymeo_rendering.outputs import Channel
        path=folder/f"pose-{records[ordinal]['pose_index']:04d}.npz"
        np.savez_compressed(path,instance_id=np.frombuffer(planes[Channel.INSTANCE_ID],"<u4").reshape(height,width),
            depth=np.frombuffer(planes[Channel.FORWARD_DEPTH],"<f4").reshape(height,width),
            valid=np.frombuffer(planes[Channel.VALID],"u1").reshape(height,width))
        records[ordinal].update(geometry_file=path.name,geometry_sha256=digest(path))
    return 1000*(perf_counter()-started)


def capture_owner(args,scene,studio,inputs,runtime,prepared,policy,source,name,channels,folder,round_number):
    from phymeo_rendering.outputs import Channel
    folder.mkdir(parents=True)
    selected=indices(args.quick,args.poses)
    requested=(Channel.RGB,) if channels=="rgb" else tuple(Channel)
    extent=base.preset(SIZES[args.resolution])["extent"]
    row=dict(name=name,profile="baseline16" if name=="baseline16" else profile_name(policy),channels=channels,
        extent=extent,round=round_number,status="opening",frames=[],warmup_ms=[],source_key="baseline" if name=="baseline16" else "sparse",
        candidate_backend=adapter.kind(policy) if name=="sparse" else "baseline16",
        profile_label="baseline16" if name=="baseline16" else profile_name(policy),
        capture_scope="actual public capture; candidate reconstruction is not an accumulated16spp image")
    owner=None;problem=None;pngs=[];geometries=[]
    try:
        start=perf_counter();owner=open_owner(args,scene,studio,runtime,prepared,policy,name)
        row["open_ms"]=1000*(perf_counter()-start)
        for index in range(10):
            frame,request=inputs[index];request=replace(request,channels=requested)
            start=perf_counter();shot=owner.capture(frame,request)
            row["warmup_ms"].append(1000*(perf_counter()-start));base.verify_capture(shot,frame,request)
        if name=="baseline16":owner.reset_accumulation()
        before=owner.diagnostics()["device"]["resident_profile"].copy();previous=before
        loop_start=perf_counter()
        for ordinal,index in enumerate(selected):
            frame,request=inputs[index];request=replace(request,channels=requested)
            start=perf_counter();shot=owner.capture(frame,request)
            elapsed=1000*(perf_counter()-start)
            base.verify_capture(shot,frame,request)
            diag=owner.diagnostics();check_backend(diag,args,policy,name,source,expected_captures=11+ordinal)
            planes={p.channel:p.data for p in shot.planes}
            profile=diag["device"]["resident_profile"]
            item=dict(pose_index=index,frame_key=asdict(frame.key),time_s=frame.time_s,
                state_sha256=sha256(canonical(asdict(frame))).hexdigest(),
                camera_sha256=sha256(canonical(asdict(request.camera))).hexdigest(),capture_ms=elapsed,
                rgb_sha256=sha256(planes[Channel.RGB]).hexdigest(),
                sensors={c.value:sha256(v).hexdigest() for c,v in planes.items() if c!=Channel.RGB},
                public_bytes=sum(map(len,planes.values())),native_profile=profile_delta(previous,profile))
            previous=profile.copy()
            if name=="sparse":
                item["actual"]=diag["actual"]
                if adapter.kind(policy)=="compact":item["compact"]=diag["compact"]
            else:item["baseline_history"]=base.compact_history(diag)
            row["frames"].append(item)
            film=round_number==1 and args.film==name and (channels=="rgb" or "rgb" not in args.channels)
            if film or index in args.poses:pngs.append((ordinal,planes[Channel.RGB]))
            if round_number==1 and channels=="all" and index in args.poses:geometries.append((ordinal,planes))
        row["consumer_loop_ms"]=1000*(perf_counter()-loop_start)
        row["consumer_loop_scope"]="capture, hashes and diagnostics; PNG/geometry encoding occurs after owner close"
        row["identity_audit"]=identity_audit(row["frames"],selected)
        row["phase_timing"]=phase_statistics(row["frames"])
        row["timing"]=statistics([f["capture_ms"] for f in row["frames"]],full=not args.quick)
        row["profile"]=base.profile_delta(before,previous,len(selected))
        row["backend"]=owner.diagnostics();row["status"]="complete"
    except BaseException as error:
        problem=error;row.update(status="failed",error=dict(type=type(error).__name__,message=str(error)))
    finally:
        if owner is not None:
            try:
                owner.close();row["closed"]=owner.diagnostics();base.verify_close(row["closed"]["device"])
            except BaseException as error:
                row["close_error"]=str(error)
                if problem is None:problem=error;row["status"]="failed"
        row["encoding_ms"]=write_buffers(folder,extent,pngs,geometries,row["frames"])
        save(folder/"receipt.json",row)
    if problem is not None:raise problem
    return row


def capture(args):
    runtime,prepared,manifest=load_runtime(args)
    import panda_render_performance as factory
    policy=manifest["contract"]
    source=source_binding(args,prepared,manifest)
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(schema=SCHEMA,status="preparing",source=source,contract=policy,resolution=args.resolution,
        platform=dict(system=platform.platform(),python=sys.version,device_index=args.device_index),
        camera=args.camera,studio_look=args.studio_look,quick=args.quick,poses=args.poses,rounds=[],film=args.film,
        candidate_backend=adapter.kind(policy),paired_baseline=not args.candidate_only,
        candidate_group_key="sparse (legacy generic candidate slot); actual kind is candidate_backend and contract",
        registered_quality_configuration=args.camera=="detail" and args.studio_look=="recorded",
        quality_claim=False,realtime_claim=False,
        timing_scope="serial public owner groups; paired_baseline declares whether baseline was measured; completed synchronous wall times and separate GPU timestamps",
        sparse_scope="one independent reconstructed exposure per state; final visible-guide anchors times configured paths is a proposal budget, not all fragment/helper/overdraw invocations or hardware ray queries; no full-image spp/history claim")
    save(args.output/"summary.json",report)
    try:
        with TemporaryDirectory(prefix="phymeo-sparse-motion-") as temp:
            scenes,frames,studio,trace=factory.bind_scene(Path(temp)/"geometry")
            scene,studio=apply_look(scenes[1],studio,args.studio_look)
            inputs=base.make_inputs(scene,frames,base.preset(SIZES[args.resolution]),args.camera)
            save(args.output/"inputs.json",dict(scene=asdict(scene),studio=asdict(studio),trace_sha256=digest(trace),
                actual_full_inputs=[dict(pose_index=i,frame=asdict(f),camera=asdict(q.camera)) for i,(f,q) in inputs.items()],
                declaration=base.preset(SIZES[args.resolution]),area_light_grid=policy["area_light_grid"],
                camera=args.camera,studio_look=args.studio_look,
                look_scope="recorded lighting unchanged" if args.studio_look=="recorded" else
                    "authored presentation: direction lights removed, panel_size_scale2 with same power, original HDR/material/exposure"))
            report["inputs_sha256"]=digest(args.output/"inputs.json");save(args.output/"summary.json",report)
            previous={};all_rows=[]
            for number in range(1,args.rounds+1):
                entry=dict(index=number,order=jobs(args.channels,number,candidate_only=args.candidate_only),runs=[])
                report["rounds"].append(entry)
                for name,channels in jobs(args.channels,number,candidate_only=args.candidate_only):
                    folder=args.output/f"round-{number}"/name/channels
                    row=capture_owner(args,scene,studio,inputs,runtime,prepared,policy,source,name,channels,folder,number)
                    row["receipt"]=str((folder/"receipt.json").relative_to(args.output));row["receipt_sha256"]=digest(folder/"receipt.json")
                    key=(name,channels)
                    if key in previous:compare_repeat(previous[key],row)
                    previous[key]=row;entry["runs"].append(row);all_rows.append(row)
                    save(args.output/"summary.json",report)
                    print(json.dumps(dict(round=number,profile=row["profile_label"],channels=channels,timing=row["timing"])),flush=True)
                compare(entry["runs"])
                entry["same_mode_rgb_all_exact"]="rgb" in args.channels and "all" in args.channels
                entry["baseline_sparse_sensor_bytes_exact"]=("all" in args.channels) if not args.candidate_only else None
                save(args.output/"summary.json",report)
            report["pooled"]={}
            for name,channels in jobs(args.channels,1,candidate_only=args.candidate_only):
                rows=[r for r in all_rows if r["name"]==name and r["channels"]==channels]
                times=[f["capture_ms"] for r in rows for f in r["frames"]]
                stats=statistics(times,full=False)
                stats["latency_gate_passed"]=stats["p95_ms"]<=33.3 if not args.quick else None
                stats["real_groups"]=len(rows);stats["group_mean_ms"]=[r["timing"]["mean_ms"] for r in rows]
                stats["scope"]="distinct measured groups pooled; completed synchronous capture"
                report["pooled"][name+"/"+channels]=stats
            report["status"]="complete"
    except BaseException as error:
        report.update(status="partial_failure",error=dict(type=type(error).__name__,message=str(error)));raise
    finally:save(args.output/"summary.json",report)


def media_profile_name(row,policy,*,camera="detail",look="recorded"):
    """Derive a safe label from actual contract; historical row.profile is timing data."""
    require(row["name"] in ("baseline16","sparse"),"unknown captured model")
    require(camera in ("detail","sequence") and look in ("recorded","soft-studio"),"unknown captured presentation")
    if row["name"]=="baseline16":name="baseline16"
    else:
        for key,allowed in (("tile",(1,2,4)),("paths_per_anchor",(2,4,16)),
                ("area_light_grid",(2,4,8)),("neighbor_candidates",(4,25))):
            require(type(policy[key]) is int and policy[key] in allowed,"invalid sparse filename contract: "+key)
        revision=policy.get("reconstruction_revision",1)
        require(type(revision) is int and 1<=revision<=65535,"invalid reconstruction revision")
        require(policy.get("estimator","radiance") in ("radiance","visibility-ratio"),"unknown estimator")
        require(row["backend"]["contract"]==policy,"saved backend/source sparse contracts differ")
        actual=row["backend"]["actual"]
        check_actual(actual,policy,actual["rgb_captures"])
        if adapter.kind(policy)=="compact":
            require(row["backend"]["backend"]==adapter.backend_name(policy) and row["backend"]["history"]=="none",
                    "compact media runtime identity differs")
            adapter.check_compact(row["backend"]["compact"],row["extent"],actual["rgb_captures"],actual)
        name=profile_name(policy)
    if camera!="detail" or look!="recorded":name+="-"+camera+"-"+look
    require(0<len(name)<=128 and all(c.isascii() and (c.isalnum() or c=="-") for c in name),"unsafe media filename")
    return name

def encode(args):
    root=args.result.resolve(strict=True);summary=json.loads((root/"summary.json").read_text(encoding="utf-8-sig"))
    require(summary["schema"]==SCHEMA and summary["status"] in ("complete","partial_failure"),
            "sparse capture has no finalized status")
    rows=[r for r in summary["rounds"][0]["runs"] if r["name"]==summary["film"]]
    require(rows,"no designated film; capture --film sparse or --film baseline16")
    row=next((r for r in rows if r["channels"]=="rgb"),rows[0])
    require(row["status"]=="complete" and all("png" in f for f in row["frames"]),"designated film is incomplete or missing original PNGs")
    identity_audit(row["frames"],indices(summary["quick"],summary["poses"]))
    require(digest(root/"inputs.json")==summary["inputs_sha256"] and digest(root/row["receipt"])==row["receipt_sha256"],
            "capture inputs/receipt changed")
    # Adapt the genuine measured receipt to the frozen B media reader, in a NEW
    # view directory. Original records/PNGs are untouched; hardlinks save space.
    import os
    import shutil
    view=args.media_output.resolve().with_name(args.media_output.name+"-source-view")
    movie_name=media_profile_name(row,summary["contract"],camera=summary.get("camera","detail"),
                                  look=summary.get("studio_look","recorded"))
    view.mkdir(parents=True,exist_ok=False);folder=view/movie_name;folder.mkdir();(folder/"frames").mkdir()
    borrowed=dict(row);borrowed["preset"]=movie_name
    for f in row["frames"]:
        source=root/row["receipt"];source=source.parent/f["png"]
        require(digest(source)==f["png_sha256"],"source film PNG changed")
        target=folder/f["png"]
        try:os.link(source,target)
        except OSError:shutil.copyfile(source,target)
    save(folder/"receipt.json",borrowed)
    borrowed["receipt"]=str((folder/"receipt.json").relative_to(view));borrowed["receipt_sha256"]=digest(folder/"receipt.json")
    shutil.copyfile(root/"inputs.json",view/"inputs.json")
    save(view/"summary.json",dict(schema=base.SCHEMA,status="complete",quick=summary["quick"],
        inputs_sha256=summary["inputs_sha256"],requested_presets=[movie_name],runs=[borrowed],
        trajectory_scope="registered sparse-pose diagnostic montage" if summary["quick"] else "all300 registered original episode states",
        source_sparse_summary_sha256=digest(root/"summary.json"),source_run_status=summary["status"],
        scope="one complete designated capture group; unrelated failed groups are not claimed; media adapter is not a baseline capture record"))
    return base.encode(SimpleNamespace(result=view,media_output=args.media_output))


def common(parser):
    parser.add_argument("--runtime",type=Path,required=True,help="actual preview_sparse/runtime.py or preview_compact/runtime.py")
    parser.add_argument("--candidate-backend",choices=("sparse","compact"),default="sparse",
                        help="explicit prepared/runtime kind; no cross-backend fallback")
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--baseline-library",type=Path,required=True)
    parser.add_argument("--baseline-shaders",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device-index",type=int,default=1)
    parser.add_argument("--validation",action="store_true")
    parser.add_argument("--resolution",choices=tuple(SIZES),default="720p")
    parser.add_argument("--camera",choices=("detail","sequence"),default="detail")
    parser.add_argument("--studio-look",choices=("recorded","soft-studio"),default="recorded")


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest="command",required=True)
    cap=sub.add_parser("capture");common(cap)
    cap.add_argument("--quick",action="store_true")
    cap.add_argument("--candidate-only",action="store_true",help="presentation/diagnostic only; no paired baseline or registered quality acceptance")
    cap.add_argument("--poses",type=int,nargs="+",default=list(REGISTERED),help="always retain registered images; appended to diagnostic states without duplicates")
    cap.add_argument("--channels",choices=("rgb","all"),nargs="+",default=["rgb","all"])
    cap.add_argument("--rounds",type=int,choices=(1,2),default=2)
    cap.add_argument("--film",choices=("none","sparse","baseline16"),default="none")
    enc=sub.add_parser("encode");enc.add_argument("result",type=Path);enc.add_argument("--media-output",type=Path,required=True)
    score=sub.add_parser("score");score.add_argument("result",type=Path)
    score.add_argument("--references",type=Path,nargs="+",required=True)
    score.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.command=="encode":return encode(args)
    if args.command=="score":
        import assess
        return assess.main(["--result",str(args.result),"--references",*[str(p) for p in args.references],"--output",str(args.output)])
    args.baseline_library=args.baseline_library.resolve(strict=True)
    args.baseline_shaders=(args.baseline_shaders or args.baseline_library.parent).resolve(strict=True)
    args.output=args.output.resolve();args.poses=sorted(set(REGISTERED)|set(args.poses))
    require(len(set(args.channels))==len(args.channels),"duplicate channel request")
    require(not args.candidate_only or args.film!="baseline16","candidate-only has no baseline film")
    indices(args.quick,args.poses)
    return capture(args)


if __name__=="__main__":main()