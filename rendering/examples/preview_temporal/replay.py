"""Offline audit of actual original sparse-guide sequences; no GPU or RGB editing.

Current and retained values are demodulated incoming at actual traced anchors.
This does not turn those sparse anchors into a full-rate renderer or claim a
quality/speed improvement without independent incoming-radiance references.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image
import collect
from incoming import TemporalIncoming, STATUS, decode_guide

require=collect.require
SCHEMA="phymeo-offline-temporal-incoming-audit-v1"
TOKENS=("material_token","lighting_token","signal_token")


def read_json(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def original_file(root,name,expected):
    p=(root/name).resolve(strict=True)
    require(p.is_relative_to(root.resolve()) and collect.digest(p)==expected,"saved packet/PNG bytes changed or escaped source")
    return p


def audit_collection(report):
    require(report["schema"]==collect.SCHEMA and report["status"]=="complete","requires a complete actual guide collection")
    require("compact" not in report["source"]["contract"] and report["source"]["contract"].get("bounces")==1,
            "this anchor-only offline protocol cannot consume compact or two-hop guides")
    require(type(report["count"]) is int and 2<=report["count"]<=32,"bounded source count required")
    require([r["pose_index"] for r in report["records"]]==list(range(report["start"],report["start"]+report["count"])),
            "saved original-state sequence has a missing/repeated/reordered frame")


def load_packet(root,binding,row):
    entries=[f for f in binding["frames"] if f["pose_index"]==row["pose_index"]]
    require(len(entries)==1,"packet has no unique original bound frame")
    entry=entries[0];frame,camera=entry["frame"],entry["camera"]
    require(sha256(collect.canonical(frame)).hexdigest()==row["frame_sha256"] and
            sha256(collect.canonical(camera)).hexdigest()==row["camera_sha256"],"bound physical frame/camera changed")
    meta=row["metadata"]
    expected=collect.metadata_for(binding["scene"],frame,camera,**{k:meta[k] for k in TOKENS})
    require(collect.canonical(meta)==collect.canonical(expected),"metadata source identity/camera/rigid mapping differs")
    path=original_file(root,row["file"],row["sha256"])
    with np.load(path,allow_pickle=False) as data:
        require(set(data.files)=={"guide","depth","instance_id","valid"},"unexpected actual packet layout")
        planes={k:data[k].copy() for k in data.files}
    h,w=camera["height"],camera["width"]
    for name,shape,dtype in (("guide",(h,w,4),"<f4"),("depth",(h,w),"<f4"),
                             ("instance_id",(h,w),"<u4"),("valid",(h,w),"u1")):
        require(planes[name].shape==shape and planes[name].dtype==np.dtype(dtype),"actual public plane shape/type differs: "+name)
    require(sha256(planes["guide"].tobytes()).hexdigest()==row["guide_sha256"],"actual incoming-guide bytes changed")
    hashes={public:sha256(planes[name].tobytes()).hexdigest() for name,public in (
        ("depth","forward_depth_m_f32"),("instance_id","instance_id_u32"),("valid","valid_u8"))}
    require(hashes==row["sensor_sha256"],"original public sensor bytes changed")
    path=original_file(root,row["public_png"],row["public_png_sha256"])
    with Image.open(path) as png:
        require(png.mode=="RGB" and png.size==(w,h) and sha256(png.tobytes()).hexdigest()==row["public_rgb_sha256"],
                "original public RGB PNG changed")
    decoded=decode_guide(planes["guide"])
    return dict(meta,**{k:decoded[k] for k in ("incoming","normal_world","traced")},
                **{k:planes[k] for k in ("depth","instance_id","valid")})


def statistics(delta):
    if not len(delta):return dict(pixels=0,rms=None,signed_rgb=None,signed_luminance=None,max_absolute=None)
    return dict(pixels=len(delta),rms=float(np.sqrt(np.mean(delta**2))),signed_rgb=delta.mean(axis=0).tolist(),
                signed_luminance=float((delta@np.array([.2126,.7152,.0722])).mean()),max_absolute=float(np.abs(delta).max()))


def audit_fallback_exact(current,result):
    accepted=result["status"]==STATUS["accepted"]
    require(np.array_equal(result["incoming"][~accepted],current[~accepted]),"rejected/untraced pixel consumed stale history")
    require(np.all(result["age"][accepted]>=2),"accepted history has no prior observation")
    return dict(accepted_pixels=int(accepted.sum()),rejected_and_untraced_current_exact=True,
                changed_from_current=statistics((result["incoming"]-current)[accepted]),
                interpretation="incoming change only, not RMSE to truth, noise reduction or rendered RGB quality")


def stale_screen_counterexample(previous,current):
    if previous is None:return None
    if previous["depth"].shape!=current["depth"].shape:return dict(comparable=False,reason="extent changed")
    eligible=current["traced"]&(current["valid"]==1)&np.isfinite(current["depth"])
    olddepth=np.where(np.isfinite(previous["depth"]),previous["depth"],0.)
    nowdepth=np.where(np.isfinite(current["depth"]),current["depth"],0.)
    unsafe=eligible&((previous["valid"]==0)|(previous["instance_id"]!=current["instance_id"])|
        (np.abs(nowdepth-olddepth)>.003*np.maximum(nowdepth,olddepth)))
    return dict(comparable=True,unsafe_current_pixels=int(unsafe.sum()),
        unguarded_half_blend_change=statistics(((previous["incoming"]-current["incoming"])*.5)[unsafe]),
        interpretation="deliberately stale screen-space negative control, not a physically valid motion correspondence")


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source",type=Path)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--policies",choices=("reset-any","primary-only"),nargs="+",default=["reset-any","primary-only"])
    ap.add_argument("--mask-poses",nargs="+",type=int,default=[59,60])
    args=ap.parse_args(argv);root=args.source.resolve(strict=True)
    report=read_json(root/"summary.json");audit_collection(report)
    require(collect.digest(root/"inputs.json")==report["inputs_sha256"],"actual input declarations changed")
    binding=read_json(root/"inputs.json")
    require([x["pose_index"] for x in binding["frames"]]==[x["pose_index"] for x in report["records"]],"source input sequence differs")
    if report.get("mode","recorded")=="camera-control":
        origin=binding["authored_control_origin"]
        require(origin is not None and binding["mode"]=="camera-control","authored control source missing")
        for i,entry in enumerate(binding["frames"]):
            frame,camera=entry["frame"],entry["camera"]
            require(frame["nodes"]==origin["frame"]["nodes"] and frame["key"]["phase"]=="authored" and
                    frame["key"]["tick"]==i+1 and frame["key"]["revision"]==i and frame["time_s"]==i/30,
                    "static optical control changed original geometry or relabelled physical time")
            require(camera["world_pose"]["position_m"]==collect.control_camera_position(origin["camera"]["world_pose"],i)
                    and camera["world_pose"]["orientation_xyzw"]==origin["camera"]["world_pose"]["orientation_xyzw"],
                    "authored camera differs from preregistered real0.1mm motion")
        require(binding["frames"][0]["camera"]["world_pose"]!=binding["frames"][-1]["camera"]["world_pose"],
                "camera positive control did not actually move")
    require(len(set(args.policies))==len(args.policies),"duplicate policy would double-count evidence")
    radius=report["source"]["contract"]["tile"]
    require(type(radius) is int and radius in (1,2,4),"unregistered sparse lattice")
    # Before evaluating data: one full lattice spacing reaches adjacent anchors.
    # Keep min_support3; missing sparse support rejects, never relax after capture.
    settings=dict(clamp_radius=radius,min_support=3,max_age=8,relative_depth=.003,normal_dot=.995,roundtrip_pixels=.75)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    result=dict(schema=SCHEMA,status="reviewing",source=dict(file=str(root/"summary.json"),
        sha256=collect.digest(root/"summary.json"),inputs_sha256=report["inputs_sha256"],
        script_sha256=collect.digest(__file__),algorithm_sha256=collect.digest(Path(__file__).with_name("incoming.py")),
        original_runtime=report["source"]),settings=settings,
        protocol="anchor-only domain; clamp radius fixed to declared tile spacing before observing data; current fallback unchanged",
        quality_claim=False,performance_claim=False,gpu_temporal_execution=False,
        policies={},public_rgb_scope="original PNGs are never filtered, recolored, blended or replaced",
        limitations=["No independent incoming-radiance reference is present; no quality/noise claim.",
                    "Sparse anchors are not full-rate reconstructed indirect radiance.",
                    "primary-only is exploratory: moving geometry may change secondary visibility outside the receiver.",
                    "Conservative reset-any may legitimately reuse zero history throughout articulated motion.",
                    "CPU offline elapsed time is not GPU rendering performance."])
    collect.save(out/"summary.json",result)
    try:
        for policy in args.policies:
            engine=TemporalIncoming(scene_motion_policy=policy,**settings);rows=[];previous=None
            for ordinal,record in enumerate(report["records"]):
                require(all(record["metadata"][k]==report["tokens"][k] for k in TOKENS),"immutable signal/light/material token changed")
                packet=load_packet(root,binding,record)
                # The registered movie subsamples physics ticks (53/54 ticks per
                # presented frame). Keep the original key and certify adjacency
                # separately from the exact complete source record sequence.
                packet["sequence"]=dict(index=ordinal,source_sha256=result["source"]["sha256"],
                    previous_key=None if ordinal==0 else report["records"][ordinal-1]["metadata"]["key"])
                start=perf_counter();value=engine.update(packet);elapsed=1000*(perf_counter()-start)
                evidence=audit_fallback_exact(packet["incoming"],value)
                mask=value["status"]==STATUS["accepted"]
                identities,counts=np.unique(packet["instance_id"][mask],return_counts=True)
                evidence["accepted_by_actual_instance_id"]={str(int(i)):int(n) for i,n in zip(identities,counts,strict=True)}
                row=dict(pose_index=record["pose_index"],frame_sha256=record["frame_sha256"],
                    input_guide_sha256=record["guide_sha256"],source_packet_sha256=record["sha256"],
                    public_png=record["public_png"],public_png_sha256=record["public_png_sha256"],
                    algorithm_ms=elapsed,diagnostics=value["diagnostics"],audit=evidence,
                    stale_screen_negative_control=stale_screen_counterexample(previous,packet))
                if record["pose_index"] in args.mask_poses:
                    folder=out/policy;folder.mkdir(exist_ok=True)
                    for name,array in (("accepted",(value["status"]==STATUS["accepted"]).astype("u1")*255),
                                       ("traced",packet["traced"].astype("u1")*255),
                                       ("age",value["age"].astype("u1"))):
                        path=folder/f"pose-{record['pose_index']:04d}-{name}.png"
                        Image.fromarray(array).save(path)
                        row.setdefault("diagnostic_masks",{})[name]=dict(file=str(path.relative_to(out)),sha256=collect.digest(path),
                            meaning="exact diagnostic mask/count; not an altered rendered image")
                rows.append(row);previous=packet
            total=sum(r["audit"]["accepted_pixels"] for r in rows)
            result["policies"][policy]=dict(status="bindings_and_guards_passed",records=rows,
                total_accepted_history_pixels=total,reuse_observed=total>0,
                secondary_motion_guard="all object/light changes reset" if policy=="reset-any" else "not established",
                all_rejected_current_exact=all(r["audit"]["rejected_and_untraced_current_exact"] for r in rows),
                algorithm_total_ms=sum(r["algorithm_ms"] for r in rows))
            collect.save(out/"summary.json",result)
        if report.get("mode","recorded")=="camera-control":
            result["authored_camera_control"]=dict(minimum_accepted_pixels=16,
                all_static_scene_nodes_verified=True,actual_camera_motion_verified=True,
                original_pose=binding["authored_control_origin"]["pose_index"],camera_step_m=.0001,
                positive_history_observed=all(v["total_accepted_history_pixels"]>=16 for v in result["policies"].values()),
                scope="authored optical control in original recorded scene; not a continuation of physics")
            require(result["authored_camera_control"]["positive_history_observed"],"camera positive control has insufficient accepted history")
        result["status"]="bindings_and_guards_passed"
    except BaseException as error:
        result.update(status="failed",error=dict(type=type(error).__name__,message=str(error)));raise
    finally:collect.save(out/"summary.json",result)
    print(json.dumps(dict(status=result["status"],policies={k:v["total_accepted_history_pixels"] for k,v in result["policies"].items()})))
    return result


if __name__=="__main__":main()