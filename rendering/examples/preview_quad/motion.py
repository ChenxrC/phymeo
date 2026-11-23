"""Bound 2/4-proposal Panda motion/film wrapper; frozen B capture/encode stay unchanged."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import panda_motion_preview as preview
from build_overlay import TARGET, digest, require, save, verify_overlay
from validate import subgroup_capabilities

SCHEMA="phymeo-preview-quad-motion-v1"
RESOLUTIONS={"720p":"preview720","540p":"fast540","360p":"fast360"}


def clean_uris(value):
    if isinstance(value,dict):
        return {k:clean_uris(v) for k,v in value.items() if k!="uri"}
    if isinstance(value,(list,tuple)):
        return [clean_uris(v) for v in value]
    return value


def descriptor(model,resolution):
    require(resolution in RESOLUTIONS,"unknown preview extent")
    p=preview.preset(RESOLUTIONS[resolution])
    p["name"]=f"{model}-{resolution}"
    p["label"]=f"{model} / {resolution} / 单跳运动预览"
    return p


def actual_budget(row,samples,manifest):
    row["experiment"]=dict(actual_primary_proposals_per_eligible_pixel_per_batch=samples,
        runtime_nominal_primary_proposals=16,overlay_manifest_sha256=manifest,
        scope="shader proposal budget, not a measured ray/hit counter; neighbors are not rays from this pixel")
    for f in row["frames"]:
        h=f["history"]["progressive"]
        f["actual_primary_proposals_per_eligible_pixel_per_batch"]=samples
        f["actual_accumulated_proposal_budget"]=samples*h["batches"]
        f["legacy_nominal_primary_samples"]=h["primary_samples_per_pixel"]
    for f in row.get("stopped",{}).get("samples",[]):
        f["actual_accumulated_proposal_budget"]=samples*f["batch"]
        f["legacy_nominal_primary_samples"]=f["history"]["progressive"]["primary_samples_per_pixel"]


def same_repeat(a,b):
    require(a["extent"]==b["extent"] and len(a["frames"])==len(b["frames"]),
            "repeat extent/frame count differs")
    for x,y in zip(a["frames"],b["frames"],strict=True):
        require(all(x[k]==y[k] for k in ("pose_index","frame_key","time_s","state_sha256",
                    "camera_sha256","rgb_sha256","sensors")), "repeat input/RGB/sensor bytes differ")
        require(x["history"]["progressive"]["batches"]==y["history"]["progressive"]["batches"],
                "repeat accumulation history differs")


def aggregate_measurements(rows, *, quick):
    """Pool distinct real groups; a borrowed receipt referenced twice counts once."""
    import math
    groups={}
    for row in rows:
        key=row["preset"]+"/"+row["channels"]
        measurement=row.get("cache",{}).get("receipt_sha256",row.get("receipt_sha256"))
        require(measurement is not None,"measurement has no immutable receipt identity")
        groups.setdefault(key,{})[measurement]=row
    result={}
    for key,unique in groups.items():
        values=list(unique.values())
        times=[f["capture_ms"] for r in values for f in r["frames"]]
        require(times and all(math.isfinite(t) and t>0 for t in times),"invalid pooled capture durations")
        ranked=sorted(times)
        n=len(times)
        mean=sum(times)/n
        p95=ranked[math.ceil(.95*n)-1]
        result[key]=dict(distinct_measured_groups=len(values),completed_captures=n,
            p50_ms=ranked[math.ceil(.5*n)-1],p95_ms=p95,max_ms=ranked[-1],mean_ms=mean,
            completed_fps_from_mean=1000/mean,within_33_3_ms=sum(t<=33.3 for t in times),
            group_mean_ms=[sum(f["capture_ms"] for f in r["frames"])/len(r["frames"]) for r in values],
            contains_borrowed_measurement=any("cache" in r for r in values),
            latency_gate_passed=(p95<=33.3) if not quick and n>=300 else None,
            scope="distinct completed capture intervals only; cached groups are not newly interleaved measurements")
    return result


def source_equivalence(path, expected_hash):
    """Allow exact bytes or explicitly reconstructed terminal blank-line variants only."""
    data=Path(path).read_bytes()
    current=sha256(data).hexdigest()
    if expected_hash==current:
        return dict(current_sha256=current,original_sha256=expected_hash,exact_bytes=True,
                    equivalence="exact bytes")
    body=data.rstrip(b"\r\n")
    for newline in (b"\n",b"\r\n"):
        for count in range(1,9):
            historical=body+newline*count
            if sha256(historical).hexdigest()==expected_hash:
                return dict(current_sha256=current,original_sha256=expected_hash,exact_bytes=False,
                            equivalence="reconstructed terminal empty-line-only difference",
                            terminal_newline="LF" if newline==b"\n" else "CRLF",terminal_newline_count=count)
    raise ValueError("cached capture script is neither exact nor a reconstructed terminal empty-line variant")

def read_cache(directory,expected_source,args,scene,studio,inputs,decoded_geometry):
    """Admit saved B measurements; never call a renderer or invent a second timing round."""
    directory=Path(directory).resolve(strict=True)
    raw=(directory/"summary.json").read_bytes()
    summary=json.loads(raw.decode("utf-8-sig"))
    require(summary["schema"]==preview.SCHEMA and summary["status"]=="complete","baseline cache is incomplete/wrong schema")
    require(summary["quick"]==args.quick and summary["camera"]==args.camera and
            summary["validation"]==args.validation,"baseline cache protocol differs")
    require(digest(directory/"inputs.json")==summary["inputs_sha256"],"baseline input manifest changed")
    binding=json.loads((directory/"inputs.json").read_text(encoding="utf-8-sig"))
    require(clean_uris(binding["scene"])==clean_uris(asdict(scene)) and
            clean_uris(binding["studio"])==clean_uris(asdict(studio)),"cached scene/material/light declaration differs")
    for key in ("library_sha256","shaders","native_python_sha256","fixture_builder_sha256"):
        require(summary["source"][key]==expected_source[key],"baseline runtime identity differs: "+key)
    script_binding=source_equivalence(preview.__file__,summary["source"]["sha256"])
    admitted={}
    for row in summary["runs"]:
        resolution=next((k for k,v in RESOLUTIONS.items() if v==row["preset"]),None)
        if resolution is None or resolution not in args.resolutions or row["channels"] not in args.channels:
            continue
        declaration=preview.preset(RESOLUTIONS[resolution])
        require(row["declaration"]==declaration and row["status"]=="complete","baseline is not the same one-hop preset")
        path=directory/row["receipt"]
        require(path.resolve().is_relative_to(directory) and digest(path)==row["receipt_sha256"],
                "baseline receipt path/hash differs")
        stored=json.loads(path.read_text(encoding="utf-8-sig"))
        require({k:v for k,v in row.items() if k not in ("receipt","receipt_sha256")}==stored,
                "baseline summary does not match its receipt")
        require(row["backend"]["decoded_geometry_sha256"]==decoded_geometry,"baseline decoded mesh content differs")
        current=inputs[resolution]
        require([f["pose_index"] for f in row["frames"]]==list(preview.pose_indices(args.quick)),
                "baseline ordered pose list differs")
        for f in row["frames"]:
            frame,req=current[f["pose_index"]]
            require(f["frame_key"]==asdict(frame.key) and f["time_s"]==frame.time_s and
                    f["state_sha256"]==sha256(preview.canonical(asdict(frame))).hexdigest() and
                    f["camera_sha256"]==sha256(preview.canonical(asdict(req.camera))).hexdigest(),
                    "baseline actual state/camera differs")
            if "png" in f:
                png=path.parent/f["png"]
                require(png.resolve().is_relative_to(directory) and digest(png)==f["png_sha256"],
                        "baseline saved PNG bytes differ")
        preview.verify_close(row["closed"])
        copied=deepcopy(row)
        copied["preset"]=f"baseline16-{resolution}"
        copied["cache"]=dict(summary_file=str(directory/"summary.json"),summary_sha256=sha256(raw).hexdigest(),
                            receipt_file=str(path),receipt_sha256=row["receipt_sha256"],
                            original_capture_script_sha256=summary["source"]["sha256"],
                            current_capture_script_sha256=digest(preview.__file__),
                            exact_capture_script_bytes_equal=summary["source"]["sha256"]==digest(preview.__file__),
                            capture_script_binding=script_binding,
                            timing_reused=True,scope="one original measured group; repeated references are NOT new timing samples")
        admitted[(resolution,row["channels"])]=copied
    return admitted


def validation_proofs(paths,library_hash,uuid):
    accepted=set()
    for path in paths:
        value=json.loads(path.read_text(encoding="utf-8-sig"))
        require(value["schema"]=="phymeo-preview-quad-validation-v1" and value["status"]=="complete",
                "prototype validation proof is not complete")
        require(value["library"]["sha256"]==library_hash and value["subgroup"]["device_uuid"]==uuid,
                "prototype validation used another DLL/device")
        quad=next(r for r in value["runs"] if r["name"]=="quad")
        require(quad["status"]=="complete" and quad["records"] and
                all(r["backend"]["device"]["validation_enabled"] for r in quad["records"]),
                "prototype proof did not enable actual Vulkan validation layers")
        preview.verify_close(quad["closed"])
        accepted.add(value["overlay_manifest_sha256"])
    return accepted


def geometry_packets(args,scene,studio,inputs,runs):
    """Small separate sensor-only exposures; never part of motion latency or GI sampling."""
    import numpy as np
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.outputs import Channel
    result=[]
    for resolution in args.resolutions:
        folder=args.output/"geometry"/resolution
        folder.mkdir(parents=True)
        p=preview.preset(RESOLUTIONS[resolution])
        owner=NativeBackend(library_path=args.library,shader_directory=args.base_shaders,
            device_index=args.device_index,studio=studio,exposure=8.,display_transform="pbr-neutral",
            validation=args.validation,**p["settings"]).open_scene(scene)
        try:
            for index in (59,60,149):
                f,req=inputs[resolution][index]
                req=replace(req,channels=tuple(c for c in Channel if c!=Channel.RGB))
                start=perf_counter()
                shot=owner.capture(f,req)
                elapsed=(perf_counter()-start)*1000
                preview.verify_capture(shot,f,req)
                planes={v.channel:v.data for v in shot.planes}
                hashes={c.value:sha256(v).hexdigest() for c,v in planes.items()}
                matches=[r for r in runs if r["extent"]==p["extent"] and r["channels"]=="all"]
                for r in matches:
                    hit=next((z for z in r["frames"] if z["pose_index"]==index),None)
                    if hit:
                        require(hit["sensors"]==hashes,"separate identity packet differs from measured four-channel capture")
                width,height=p["extent"]
                path=folder/f"pose-{index:04d}.npz"
                np.savez_compressed(path,instance_id=np.frombuffer(planes[Channel.INSTANCE_ID],"<u4").reshape(height,width),
                    depth=np.frombuffer(planes[Channel.FORWARD_DEPTH],"<f4").reshape(height,width),
                    valid=np.frombuffer(planes[Channel.VALID],np.uint8).reshape(height,width))
                result.append(dict(resolution=resolution,pose_index=index,file=str(path.relative_to(args.output)),
                    sha256=digest(path),sensors=hashes,capture_ms=elapsed,frame_key=asdict(f.key),time_s=f.time_s,
                    state_sha256=sha256(preview.canonical(asdict(f))).hexdigest(),
                    camera_sha256=sha256(preview.canonical(asdict(req.camera))).hexdigest()))
            require(owner.diagnostics()["progressive"]["batches"]==0,"sensor-only pass advanced GI history")
        finally:
            owner.close()
            preview.verify_close(owner.diagnostics()["device"])
    return result


def capture(args):
    import panda_render_performance as bench
    from phymeo_rendering.backends import native
    from phymeo_rendering.mesh_data import load_mesh
    args.library=args.library.resolve(strict=True)
    args.base_shaders=args.base_shaders.resolve(strict=True)
    args.output=args.output.resolve()
    require(len(set(args.resolutions))==len(args.resolutions) and len(set(args.channels))==len(args.channels),
            "duplicate resolution/channel requested")
    source=dict(script=str(Path(__file__).resolve()),sha256=digest(__file__),
                library=str(args.library),library_sha256=digest(args.library),
                shaders={p.name:digest(p) for p in sorted(args.base_shaders.glob("*.spv"))},
                native_python=str(Path(native.__file__).resolve()),native_python_sha256=digest(native.__file__),
                fixture_builder_sha256=digest(bench.__file__),
                capture_script=str(Path(preview.__file__).resolve()),capture_script_sha256=digest(preview.__file__))
    variants=[dict(name="baseline16",samples=16,shaders=args.base_shaders,manifest=None)]
    seen=set()
    for directory in args.overlays:
        directory=directory.resolve(strict=True)
        manifest=verify_overlay(directory)
        require(manifest["base_shaders"]==source["shaders"],"overlay was compiled over a different complete base shader set")
        name=("quad" if manifest["sharing"] else "local")+str(manifest["samples_per_eligible_pixel"])
        require(name not in seen,"duplicate model/sample overlay")
        seen.add(name)
        variants.append(dict(name=name,samples=manifest["samples_per_eligible_pixel"],shaders=directory,
                             manifest=manifest,manifest_sha256=digest(directory/"overlay.json")))
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(schema=SCHEMA,status="preparing",source=source,quick=args.quick,
                rounds=[],cached_groups=[],quality_claim=False,realtime_claim=False,
                actual_path_scope="single diffuse bounce; actual2/4 proposals are named/hash-bound; legacy diagnostics16 retained separately",
                camera=args.camera,resolutions=args.resolutions,channels=args.channels,
                ordering="round1 forward complete configuration/channel jobs; round2 reversed jobs; pose order always chronological",
                reference_scope="Panda workshop only; 25-pose blocker/69-pixel reveal is a different fixture",
                overlays=[dict(directory=str(v["shaders"]),manifest_sha256=v["manifest_sha256"],
                    manifest=v["manifest"]) for v in variants[1:]])
    save(args.output/"experiment.json",report)
    try:
        capability=subgroup_capabilities(args.device_index)
        report["subgroup"]=capability
        proofs=validation_proofs(args.validation_proof,source["library_sha256"],capability["device_uuid"])
        for v in variants[1:]:
            require(args.validation or v["manifest"]["standalone_spirv_validation"]["status"]=="passed"
                    or v["manifest_sha256"] in proofs,
                    "unvalidated prototype needs --validation or a bound --validation-proof")
        with TemporaryDirectory(prefix="phymeo-quad-motion-") as temp:
            scenes,frames,studio,trace=bench.bind_scene(Path(temp)/"geometry")
            scene=scenes[1]
            inputs={r:preview.make_inputs(scene,frames,preview.preset(RESOLUTIONS[r]),args.camera) for r in args.resolutions}
            decoded={m.id:load_mesh(m) for m in scene.meshes}
            geometry=sha256(preview.canonical({name:[m.positions,m.normals,m.triangles] for name,m in decoded.items()})).hexdigest()
            cache={}
            for path in args.baseline_cache:
                for key,row in read_cache(path,source,args,scene,studio,inputs,geometry).items():
                    require(key not in cache,"ambiguous duplicate cached baseline group")
                    cache[key]=row
            report["cached_groups"]=[row["cache"] for row in cache.values()]
            jobs=[(resolution,v,c) for resolution in args.resolutions for v in variants for c in args.channels]
            report["orders"]=[[dict(resolution=r,variant=v["name"],channels=c) for r,v,c in order]
                              for order in (jobs,list(reversed(jobs)))[:args.rounds]]
            save(args.output/"experiment.json",report)
            previous={}
            all_rows=[]
            for round_no in range(1,args.rounds+1):
                root=args.output/f"round-{round_no}"
                root.mkdir()
                ordered=jobs if round_no==1 else list(reversed(jobs))
                round_report=dict(schema=preview.SCHEMA,status="capturing",quick=args.quick,
                    source=source,camera=args.camera,validation=args.validation,nominal_playback_fps=30,
                    trajectory_scope="six registered diagnostic poses; montage" if args.quick else "all300 original source states",
                    requested_presets=[],requested_channels=args.channels,runs=[],cached=[],realtime_claim=False,
                    quality_acceptance="not verified by timing or video; see separately bound ROI assessment")
                binding=dict(scene=asdict(scene),studio=asdict(studio),trace_sha256=digest(trace),
                    actual_full_inputs={r:[dict(pose_index=i,frame=asdict(f),camera=asdict(q.camera))
                        for i,(f,q) in rows.items()] for r,rows in inputs.items()},
                    cases=[dict(name=v["name"],samples=v["samples"],overlay_manifest=v.get("manifest_sha256")) for v in variants])
                save(root/"inputs.json",binding)
                round_report["inputs_sha256"]=digest(root/"inputs.json")
                save(root/"summary.json",round_report)
                round_rows=[]
                for ordinal,(resolution,v,channels) in enumerate(ordered):
                    key=(resolution,v["name"],channels)
                    if v["name"]=="baseline16" and (resolution,channels) in cache:
                        row=deepcopy(cache[(resolution,channels)])
                        round_report["cached"].append(row["cache"])
                        round_rows.append(row)
                        continue
                    p=descriptor(v["name"],resolution)
                    call=SimpleNamespace(library=args.library,shader_directory=v["shaders"],
                        device_index=args.device_index,validation=args.validation,quick=args.quick,
                        settle_batches=0,order_index=ordinal)
                    folder=root/p["name"]/channels
                    row=preview.capture_run(call,scene,studio,inputs[resolution],p,channels,folder,
                        save_all=(round_no==1 or args.keep_second_round) and (channels=="rgb" or "rgb" not in args.channels))
                    expected_frag=source["shaders"][TARGET] if v["manifest"] is None else v["manifest"]["shaders"][TARGET]
                    require(row["backend"]["shader_sha256"]["frag"]==expected_frag and
                            row["backend"]["device"]["device_uuid"]==capability["device_uuid"],
                            "actual shader/device differs from admitted experiment")
                    actual_budget(row,v["samples"],v.get("manifest_sha256"))
                    save(folder/"receipt.json",row)
                    row["receipt"]=str((folder/"receipt.json").relative_to(root))
                    row["receipt_sha256"]=digest(folder/"receipt.json")
                    if key in previous:
                        same_repeat(previous[key],row)
                    else:
                        previous[key]=row
                    round_report["runs"].append(row)
                    if p["name"] not in round_report["requested_presets"]:
                        round_report["requested_presets"].append(p["name"])
                    round_rows.append(row)
                    save(root/"summary.json",round_report)
                    print(json.dumps(dict(round=round_no,preset=p["name"],channels=channels,timing=row["timing"],
                                          actual_proposals=v["samples"]),ensure_ascii=False),flush=True)
                round_report["comparisons"]=preview.compare_runs(round_rows)
                round_report["status"]="complete"
                save(root/"summary.json",round_report)
                all_rows.extend(round_rows)
                report["rounds"].append(dict(index=round_no,file=str((root/"summary.json").relative_to(args.output)),
                    sha256=digest(root/"summary.json"),new_measured_groups=len(round_report["runs"]),
                    reused_groups=len(round_report["cached"])))
                save(args.output/"experiment.json",report)
            if not args.no_geometry:
                report["geometry"]=geometry_packets(args,scene,studio,inputs,all_rows)
            report["pooled_timings"]=aggregate_measurements(all_rows,quick=args.quick)
            report["repeat_rgb_sensors_exact"]=args.rounds==2
            report["repeat_check_scope"]="fresh groups only; repeated cached references are not fresh captures"
            report["status"]="complete"
    except BaseException as error:
        report.update(status="failed",error=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        save(args.output/"experiment.json",report)


def encode(args):
    root=args.result.resolve(strict=True)
    report=json.loads((root/"experiment.json").read_text(encoding="utf-8-sig"))
    require(report["schema"]==SCHEMA and report["status"]=="complete","quad motion capture incomplete")
    entry=next(r for r in report["rounds"] if r["index"]==args.round)
    path=root/entry["file"]
    require(digest(path)==entry["sha256"],"round summary changed")
    round_report=json.loads(path.read_text(encoding="utf-8-sig"))
    for name in round_report["requested_presets"]:
        rows=[r for r in round_report["runs"] if r["preset"]==name]
        row=next((r for r in rows if r["channels"]=="rgb"),rows[0])
        require(all("png" in f for f in row["frames"]),
                "selected round lacks full PNGs; encode round1, or capture with --keep-second-round")
    # Frozen B encoder validates receipts, every PNG, PTS, complete decode, RGB/YUV metadata.
    return preview.encode(SimpleNamespace(result=path.parent,media_output=args.media_output))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    c=sub.add_parser("capture")
    c.add_argument("--library",type=Path,required=True)
    c.add_argument("--base-shaders",type=Path,required=True)
    c.add_argument("--overlays",type=Path,nargs="+",required=True)
    c.add_argument("--output",type=Path,required=True)
    c.add_argument("--device-index",type=int,default=1)
    c.add_argument("--resolutions",nargs="+",choices=tuple(RESOLUTIONS),default=["720p"])
    c.add_argument("--channels",nargs="+",choices=("rgb","all"),default=["rgb","all"])
    c.add_argument("--rounds",type=int,choices=(1,2),default=2)
    c.add_argument("--quick",action="store_true")
    c.add_argument("--camera",choices=("fixed","wide","detail","sequence"),default="detail")
    c.add_argument("--validation",action="store_true")
    c.add_argument("--validation-proof",type=Path,nargs="*",default=[])
    c.add_argument("--baseline-cache",type=Path,nargs="*",default=[])
    c.add_argument("--keep-second-round",action="store_true")
    c.add_argument("--no-geometry",action="store_true",help="omit separate3-pose sensor packets; ROI scoring then needs other bound geometry")
    e=sub.add_parser("encode")
    e.add_argument("result",type=Path)
    e.add_argument("--round",type=int,choices=(1,2),default=1)
    e.add_argument("--media-output",type=Path,required=True)
    args=p.parse_args()
    capture(args) if args.command=="capture" else encode(args)


if __name__=="__main__":
    main()

