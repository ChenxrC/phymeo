"""Registered 300-pose, 720p current/rigid-history preview over the same method.

Capture does not run physics. Both owners receive ten warmups, an explicit
history reset, then every original presented state 0..299. PNG encoding and
diagnostic reads are outside the reported synchronous capture times.
"""
import argparse
from dataclasses import asdict,replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

from .runtime import RigidHistoryBackend,activate
from .state import bind_presentations,frame_digest

SCHEMA="phymeo-preview-rigid-history-motion-v1"
POSES=tuple(range(300))
WARMUP=tuple(range(10))
REGISTERED=(59,60,149)
IDENTITY=("pose_index","frame_key","time_s","state_sha256","camera_sha256")


def require(value,message):
    if not value:raise ValueError(message)


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def clean_source(value):
    if isinstance(value,dict):return {k:clean_source(v) for k,v in value.items() if k!="uri"}
    if isinstance(value,(tuple,list)):return [clean_source(v) for v in value]
    return value


def presentation_options(args):
    camera=getattr(args,"camera","detail")
    look=getattr(args,"studio_look","recorded")
    require(camera in ("detail","sequence") and look in ("recorded","soft-studio"),
            "unsupported authored camera/studio presentation")
    return camera,look


def build_presentation(factory,shared,directory,args):
    """Use the frozen helpers; no replacement animation, lighting or camera math."""
    camera,look=presentation_options(args)
    scenes,frames,studio,trace=factory.bind_scene(directory)
    scene,studio=shared.apply_look(scenes[1],studio,look)
    declaration=shared.base.preset("preview720")
    inputs=shared.base.make_inputs(scene,frames,declaration,camera)
    return scene,studio,trace,inputs,declaration


def verify_presentation_source(source,camera,look,declaration):
    require(source.get("camera","detail")==camera and source.get("studio_look","recorded")==look,
            "saved camera/studio declaration differs")
    if camera!="detail" or look!="recorded":
        require(all(key in source for key in ("camera","studio_look","declaration","area_light_grid")),
                "nondefault presentation needs an explicit frozen source declaration")
    if "declaration" in source:
        require(canonical(source["declaration"])==canonical(declaration),"saved render/exposure declaration differs")
    if "area_light_grid" in source:
        require(source["area_light_grid"]==declaration["settings"]["area_light_grid"],"saved area-light grid differs")
    require(declaration["exposure"]==8.,"this bound capture uses the declared exposure8")


def media_profile_name(policy,summary):
    camera=summary.get("camera","detail");look=summary.get("studio_look","recorded")
    require(camera in ("detail","sequence") and look in ("recorded","soft-studio"),"unknown captured presentation")
    name=f"rigid-t4-p{policy['paths_per_anchor']}-g{policy['area_light_grid']}-k25-r2-age4"
    if policy.get("estimator")=="visibility-ratio":name+="-visibility-ratio"
    if camera!="detail" or look!="recorded":name+="-"+camera+"-"+look
    return name


def registered_source(value):
    records=value["actual_full_inputs"]
    require([r["pose_index"] for r in records]==list(POSES),"source needs exactly the ordered300 presented records")
    return {r["pose_index"]:r for r in records}


def verify_reference_header(value):
    require(value.get("schema")=="phymeo-preview-sparse-reference-v1" and value.get("status")=="complete",
            "completed same-model reference required")
    require(type(value.get("seed")) is int and value["seed"]>0 and value.get("batches",0)>=512 and
            value.get("primary_proposals_per_pixel")==16*value["batches"],"independent8192-or-more-proposal reference required")


def verify_chain(records,*,expected_indices=POSES):
    require([r["pose_index"] for r in records]==list(expected_indices),"presented predecessor sequence is incomplete")
    for i,row in enumerate(records):
        actual=row["history_actual"]
        expected=0 if actual["last_reset"] else records[i-1]["history_actual"]["sequence"]+1 if i else 0
        require(actual["sequence"]==expected,"actual history sequence differs from reset/presentation order")
        if i:require(row["time_s"]>records[i-1]["time_s"],"presented clock did not advance")


def verify_pair(current,history,*,require_reset_rgb=False):
    require(len(current)==len(history),"current/history count differs")
    for a,b in zip(current,history):
        require(all(a[k]==b[k] for k in IDENTITY),"current/history source or camera differs")
        require(a["effective_seed"]==b["effective_seed"],"current/history actual seed differs")
        require(a["sensors"]==b["sensors"],"current/history geometry bytes differ")
        require(a.get("current_sha256")==b.get("current_sha256"),"current/history current estimator differs")
        if require_reset_rgb and b["history_actual"]["last_reset"]:
            require(a["rgb_sha256"]==b["rgb_sha256"],"reset RGB differs from same-sequence current")


def load_delivery(directory):
    """Reuse the frozen delivery helpers, without editing their files."""
    directory=Path(directory).resolve(strict=True)
    expected=directory/"candidate_adapter.py"
    imported=sys.modules.get("candidate_adapter")
    if imported is not None:
        require(Path(imported.__file__).resolve()==expected,"another delivery adapter is already loaded")
    prior=list(sys.path)
    try:
        sys.path.insert(0,str(directory))
        spec=importlib.util.spec_from_file_location("rigid_shared_delivery",directory/"run.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:sys.path[:]=prior
    return module


def prepare_inputs(args,output,shared,policy=None):
    import panda_render_performance as factory
    source=read(args.source_inputs);by=registered_source(source)
    camera,look=presentation_options(args)
    scene,studio,trace,inputs,declaration=build_presentation(factory,shared,output/"geometry",args)
    verify_presentation_source(source,camera,look,declaration)
    if policy is not None:
        require(policy["area_light_grid"]==declaration["settings"]["area_light_grid"],"prepared light sampling differs from frozen declaration")
    require(digest(trace)==source["trace_sha256"],"original trace SHA differs")
    require(clean_source(asdict(scene))==clean_source(source["scene"]) and
            clean_source(asdict(studio))==clean_source(source["studio"]),"registered scene/material/light differs")
    for i,(frame,request) in inputs.items():
        require(frame_digest(frame)==frame_digest(by[i]["frame"]),f"original full frame differs at {i}")
        require(frame_digest(request.camera)==frame_digest(by[i]["camera"]),f"original camera differs at {i}")
    bindings=bind_presentations([inputs[i][0] for i in POSES],source["trace_sha256"])
    save(output/"inputs.json",dict(scene=asdict(scene),studio=asdict(studio),trace_sha256=digest(trace),
        actual_full_inputs=[dict(pose_index=i,frame=asdict(f),camera=asdict(q.camera),presentation=bindings[i]) for i,(f,q) in inputs.items()],
        camera=camera,studio_look=look,declaration=declaration,area_light_grid=declaration["settings"]["area_light_grid"],
        look_scope="recorded lighting unchanged" if look=="recorded" else
            "same frozen soft-studio presentation: no directional source, panel_size_scale2 at same power, original HDR/material/exposure",
        source_inputs_path=str(args.source_inputs.resolve()),source_inputs_sha256=digest(args.source_inputs)))
    return scene,studio,inputs,bindings


def capture_owner(args,root,policy,shared,scene,studio,inputs,bindings,output,name,row):
    import numpy as np
    from phymeo_rendering.outputs import Channel
    from .run_smoke import history_dtype
    reuse=name=="rigid-history";folder=output/name;folder.mkdir()
    requested=tuple(Channel) if args.channels=="all" else (Channel.RGB,)
    row.update(name=name,status="opening",channels=args.channels,extent=[1280,720],frames=[],warmup_ms=[],
        history_policy="per-ID rigid cap4" if reuse else "same-sequence current, no history reuse",
        capture_scope="synchronous public capture; audit readback, hashing and PNG encoding excluded")
    owner=None;problem=None;pngs=[];geometries=[];previous=None
    library_sha=digest(root/"build/phymeo_native.dll")
    shader_shas={digest(p) for p in (root/"build").glob("*.spv")}
    try:
        start=perf_counter()
        owner=RigidHistoryBackend(root,device_index=args.device_index,studio=studio,seed=args.seed,exposure=8.,
            validation=args.validation,reuse_history=reuse).open_scene(scene)
        row["open_ms"]=1000*(perf_counter()-start)
        for i in WARMUP:
            frame,request=inputs[i];request=replace(request,channels=requested)
            start=perf_counter();shot=owner.capture(frame,request,presentation=bindings[i])
            row["warmup_ms"].append(1000*(perf_counter()-start));shared.base.verify_capture(shot,frame,request)
        owner.reset_history()
        before=owner.diagnostics()["device"]["resident_profile"].copy();previous=before
        with (folder/"frames.jsonl").open("x",encoding="utf-8") as journal:
            for index in POSES:
                frame,request=inputs[index];request=replace(request,channels=requested)
                start=perf_counter();shot=owner.capture(frame,request,presentation=bindings[index]);elapsed=1000*(perf_counter()-start)
                shared.base.verify_capture(shot,frame,request)
                diag=owner.diagnostics();history=diag["history"];actual=history["actual"];rigid=diag["rigid_history"]
                require(diag["backend"]=="experimental-native-compact-rigid-history-v1" and rigid["marker"]==97 and
                    rigid["age_cap"]==4 and rigid["lookup_misses"]==0,"actual rigid backend differs")
                require(diag["library_sha256"]==library_sha and diag["rigid_shader_sha256"] in shader_shas and
                    all(v in shader_shas for v in diag["shader_sha256"].values()),"loaded binary differs from prepared binding")
                require(diag["device"]["resident_captures"]==11+index and actual["committed_rgb"]==11+index and
                    diag["actual"]["rgb_captures"]==11+index,"actual public/guide/history capture counts differ")
                require(diag["contract"]==policy and diag["device"]["resident_upload_calls"]==1,"scene upload/contract differs")
                shared.adapter.check_compact(diag["compact"],[1280,720],11+index,diag["actual"])
                require(sum(diag["actual"][k] for k in ("reconstructed_pixels","fallback_pixels","nondiffuse_pixels"))==diag["actual"]["visible_pixels"],"full-rate partition differs")
                require(diag["actual"]["primary_proposals"]==diag["actual"]["traced_anchor_pixels"]*policy["paths_per_anchor"],"actual GI proposal count differs")
                planes={v.channel:v.data for v in shot.planes};profile=diag["device"]["resident_profile"]
                record=dict(pose_index=index,frame_key=asdict(frame.key),time_s=frame.time_s,
                    state_sha256=digest_bytes(canonical(asdict(frame))),camera_sha256=digest_bytes(canonical(asdict(request.camera))),
                    rgb_sha256=digest_bytes(planes[Channel.RGB]),sensors={k.value:digest_bytes(v) for k,v in planes.items() if k!=Channel.RGB},
                    capture_ms=elapsed,public_bytes=sum(map(len,planes.values())),native_profile=shared.profile_delta(previous,profile),
                    effective_seed=history["effective_seed"],history_actual=actual,history_reset_reasons=history["last_reset_reasons"],
                    rigid=rigid,actual=diag["actual"],compact=diag["compact"],presentation=diag["presentation"])
                previous=profile.copy()
                if index in REGISTERED:
                    audit=perf_counter();current=owner.read_current_incoming();hist=owner.read_history_incoming();linear=owner.read_reconstructed_linear()
                    record["audit_read_ms"]=1000*(perf_counter()-audit)
                    cells=np.frombuffer(hist["data"],dtype=history_dtype(np))
                    now=np.frombuffer(current["data"],"<f4").reshape(-1,4)
                    require(np.isfinite(now).all() and np.isfinite(cells["incoming"]).all(),"nonfinite debug incoming")
                    support=now[:,3]==1;ages=cells["incoming"][:,3]
                    require(np.all(ages[~support]==0) and np.all(ages[support]>=1) and np.all(ages<=4),"invalid actual age/support")
                    require(int(ages.sum())==actual["age_sum"],"actual age counter differs from GPU buffer")
                    record["age_histogram"]={str(n):int(np.count_nonzero(ages==n)) for n in range(5)}
                    for suffix,packet in (("current",current),("history",hist),("linear",linear)):
                        path=folder/f"pose-{index:04d}.{suffix}.bin";path.write_bytes(packet["data"])
                        record[suffix+"_file"]=path.name;record[suffix+"_sha256"]=digest(path)
                    if args.channels=="all":geometries.append((index,planes))
                    record["backend"]=diag
                    # A following capture delta must not include this separate
                    # diagnostic transfer. Whole-owner totals still include it.
                    previous=owner.diagnostics()["device"]["resident_profile"].copy()
                if reuse or index in REGISTERED or args.current_film:pngs.append((index,planes[Channel.RGB]))
                row["frames"].append(record)
                journal.write(json.dumps(record,allow_nan=False,separators=(",",":"))+"\n");journal.flush()
        if reuse:verify_chain(row["frames"])
        row["timing"]=shared.statistics([r["capture_ms"] for r in row["frames"]],full=True)
        row["phase_timing"]=shared.phase_statistics(row["frames"])
        row["profile"]=shared.base.profile_delta(before,previous,300)
        row["profile_scope"]="whole measured owner interval includes the three registered debug reads; per-frame deltas and capture timing exclude them"
        row["observed_history_resets"]=[dict(pose_index=r["pose_index"],frame_key=r["frame_key"],
            camera_sha256=r["camera_sha256"],sequence=r["history_actual"]["sequence"],reasons=r["history_reset_reasons"])
            for r in row["frames"] if r["history_actual"]["last_reset"]]
        row["backend"]=owner.diagnostics();row["status"]="complete"
    except BaseException as error:
        problem=error;row.update(status="partial_failure",error=dict(type=type(error).__name__,message=str(error)))
    finally:
        if owner is not None:
            try:
                owner.close();row["closed"]=owner.diagnostics();shared.base.verify_close(row["closed"]["device"])
            except BaseException as error:
                row["close_error"]=str(error)
                if problem is None:problem=error;row["status"]="partial_failure"
        row["encoding_ms"]=shared.write_buffers(folder,[1280,720],pngs,geometries,row["frames"])
        journal=folder/"frames.jsonl"
        if journal.exists():row.update(journal=journal.name,journal_sha256=digest(journal))
        save(folder/"receipt.json",row)
    if problem is not None:raise problem


def digest_bytes(value):return sha256(value).hexdigest()


def capture(args):
    root,manifest=activate(args.prepared)
    shared=load_delivery(args.delivery)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    camera,look=presentation_options(args)
    dependencies=[Path(__file__),Path(shared.__file__),Path(shared.base.__file__),Path(shared.adapter.__file__)]
    report=dict(schema=SCHEMA,status="preparing",resolution="720p",camera=camera,studio_look=look,
        registered_quality_configuration=camera=="detail" and look=="recorded",quality_claim=False,
        presentation_scope="registered numeric configuration" if camera=="detail" and look=="recorded" else
            "authored viewing presentation; original numeric-quality acceptance does not transfer",
        contract=manifest["contract"],seed=args.seed,channels=args.channels,runs=[],
        source=dict(prepared=str(root),manifest_sha256=digest(root/"source-manifest.json"),
            library_sha256=digest(root/"build/phymeo_native.dll"),shaders={p.name:digest(p) for p in sorted((root/"build").glob("*.spv"))},
            dependencies=[dict(path=str(p.resolve()),sha256=digest(p)) for p in dependencies]),
        warmup=list(WARMUP),warmup_history_discarded=True,presentations=list(POSES),
        limitation="Single serial order; temporal local confidence cannot prove hidden secondary visibility. No quality claim before independent assessment.")
    save(output/"summary.json",report)
    try:
        scene,studio,inputs,bindings=prepare_inputs(args,output,shared,manifest["contract"])
        report["inputs_sha256"]=digest(output/"inputs.json")
        producer_names=("preview_compact_rigid_history.runtime","preview_compact_rigid_history.state",
            "preview_compact_history.runtime","preview_compact_history.state","preview_compact.runtime",
            "preview_sparse.runtime","panda_render_performance","panda_realtime_film")
        report["source"]["producer_modules"]=[dict(module=name,path=str(Path(sys.modules[name].__file__).resolve()),
            sha256=digest(sys.modules[name].__file__)) for name in producer_names]
        modes=("rigid-history","current") if args.reverse_order else ("current","rigid-history")
        for mode in modes:
            row={};report["runs"].append(row)
            capture_owner(args,root,manifest["contract"],shared,scene,studio,inputs,bindings,output,mode,row)
            row.update(receipt=f"{mode}/receipt.json",receipt_sha256=digest(output/mode/"receipt.json"))
            save(output/"summary.json",report)
        by={r["name"]:r for r in report["runs"]}
        verify_pair(by["current"]["frames"],by["rigid-history"]["frames"],require_reset_rgb=True)
        report["checks"]=dict(all300_source_and_seed_pairs=True,geometry_same_bytes=args.channels=="all",
            registered_current_estimator_same_bytes=True,first_reset_rgb_same_bytes=True,
            rigid_reuse_pixels=sum(r["rigid"]["moving_accepted_pixels"] for r in by["rigid-history"]["frames"]))
        report["status"]="complete"
    except BaseException as error:
        report.update(status="partial_failure",error=dict(type=type(error).__name__,message=str(error)));raise
    finally:save(output/"summary.json",report)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--source-inputs",type=Path,required=True)
    parser.add_argument("--delivery",type=Path,required=True,help="frozen parent's preview_sparse_motion directory")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--channels",choices=("rgb","all"),default="all")
    parser.add_argument("--camera",choices=("detail","sequence"),default="detail")
    parser.add_argument("--studio-look",choices=("recorded","soft-studio"),default="recorded")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--device-index",type=int,default=1)
    parser.add_argument("--validation",action="store_true")
    parser.add_argument("--reverse-order",action="store_true")
    parser.add_argument("--current-film",action="store_true")
    return parser.parse_args(argv)


def main():
    capture(parse_args())


if __name__=="__main__":main()
