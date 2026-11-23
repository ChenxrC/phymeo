"""Offline five identity ROIs / rigid finger-time scoring for actual sparse records.

Uses the already registered metric implementation, never copies product rendering
or turns the sparse image/full-rate16-proposal image into a truth reference.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np
# Reuse only the registered numeric helpers. Their legacy modules temporarily
# prepend example directories; restore the complete search path after import so
# unittest discovery and other callers cannot resolve a different test_motion.
_IMPORT_PATH=list(sys.path)
try:
    import run
    QUAD=Path(__file__).resolve().parents[1]/"preview_quad"
    sys.path.insert(0,str(QUAD))
    import score as metrics
finally:
    sys.path[:]=_IMPORT_PATH
    del _IMPORT_PATH


def read_json(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def require(value,message):return run.require(value,message)


def read_row(root,row):
    path=root/row["receipt"]
    require(path.resolve().is_relative_to(root) and run.digest(path)==row["receipt_sha256"],"motion receipt changed")
    actual=read_json(path)
    require(actual=={k:v for k,v in row.items() if k not in ("receipt","receipt_sha256")},"summary/receipt payload differ")
    require(actual["status"]=="complete","motion group incomplete")
    return path.parent,actual


def verify_reference(diag,*,seed,batches,baseline,grid):
    require(type(seed) is int and seed>0 and batches>=64,"reference must use independent nonzero seed and >=1024 proposals")
    require(diag["library_sha256"]==baseline["library_sha256"] and
        diag["indirect"]["mode"]=="diffuse_bounce" and diag["indirect"]["bounces"]==1 and
        diag["indirect"]["seed"]==seed and diag["indirect"]["samples_per_pixel"]==16 and
        diag["progressive"]["batches"]==batches and diag["progressive"]["primary_samples_per_pixel"]==16*batches and
        diag["area_light_grid"]==grid and diag["exposure"]==8. and diag["display_transform"]=="pbr-neutral" and
        diag["reflection"]["mode"]=="single_bounce" and not diag["spatial_filter"]["last_capture_filtered"],
        "reference actual model/seed/batches/exposure/filter differs")
    require(all(v in baseline["shaders"].values() for v in diag["shader_sha256"].values()),"reference loaded a different shader model")


def references(paths,report,binding,baseline_rows,candidate_rows,geometry):
    result=[]
    for directory in paths:
        directory=directory.resolve(strict=True)
        file=directory/"summary.json";value=read_json(file)
        require(value["status"]=="complete","reference source is incomplete")
        require(run.digest(directory/"inputs.json")==value["inputs_sha256"],"reference input binding changed")
        rb=read_json(directory/"inputs.json")
        require(run_source_clean(rb["scene"])==run_source_clean(binding["scene"]) and
                run_source_clean(rb["studio"])==run_source_clean(binding["studio"]),"reference scene/material/light declaration differs")
        if value["schema"]==run.REFERENCE_SCHEMA:
            native=value["source"]["baseline"];seed=value["seed"];batches=value["batches"]
            records=value["records"];image_root=directory
        elif value["schema"]=="phymeo-preview-quad-validation-v1":
            native=dict(library_sha256=value["library"]["sha256"],shaders=value["overlay"]["base_shaders"])
            rr=next(r for r in value["runs"] if r["name"]=="reference")
            require(rr["status"]=="complete" and rr["proposal_budget"]==16,"saved reference group invalid")
            seed=rr["seed"];batches=rr["batches_per_pose"];records=rr["records"];image_root=directory/"reference"
        else:raise ValueError("unsupported genuine reference schema; previous two-hop appearance archives are not one-hop truth")
        baseline=report["source"]["baseline"]
        require(native["library_sha256"]==baseline["library_sha256"] and native["shaders"]==baseline["shaders"],
                "reference baseline DLL or complete SPV set differs")
        by={r["pose_index"]:r for r in records};images={}
        for pose in run.REGISTERED:
            f=by[pose];b=baseline_rows[pose];c=candidate_rows[pose];g=geometry[pose][0]
            require(all(f[k]==b[k]==c[k]==g[k] for k in run.IDENTITY),"reference/current physical frame/camera binding differs")
            sensor=f["sensors"] if "sensors" in f else f["sensor_sha256"]
            require(sensor==g["sensors"],"reference actual geometry bytes differ")
            verify_reference(f["backend"],seed=seed,batches=batches,baseline=baseline,grid=report["contract"]["area_light_grid"])
            images[pose]=metrics.image(image_root/f["png"],f["png_sha256"],f["rgb_sha256"])
        result.append(dict(seed=seed,batches=batches,images=images,file=str(file),sha256=run.digest(file)))
    require(len(result)==len({r["seed"] for r in result}),"duplicate seed cannot establish independent reference resolution")
    return result


def run_source_clean(value):
    if isinstance(value,dict):return {k:run_source_clean(v) for k,v in value.items() if k!="uri"}
    if isinstance(value,(tuple,list)):return [run_source_clean(v) for v in value]
    return value


def geometry_packet(path,row):
    file=path/row["geometry_file"]
    require(run.digest(file)==row["geometry_sha256"],"geometry packet changed")
    with np.load(file,allow_pickle=False) as arrays:
        data={k:arrays[k].copy() for k in ("instance_id","depth","valid")}
    hashes={"instance_id_u32":sha256(data["instance_id"].astype("<u4",copy=False).tobytes()).hexdigest(),
            "forward_depth_m_f32":sha256(data["depth"].astype("<f4",copy=False).tobytes()).hexdigest(),
            "valid_u8":sha256(data["valid"].astype("u1",copy=False).tobytes()).hexdigest()}
    require(hashes==row["sensors"],"geometry typed bytes differ from capture")
    return row,data


def audit_configuration(report):
    require(report["resolution"]=="720p" and report.get("camera","detail")=="detail"
            and report.get("studio_look","recorded")=="recorded",
            "registered quality gates require720p recorded/detail; aesthetic sequence/lighting is separate")

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result",type=Path,required=True)
    ap.add_argument("--references",type=Path,nargs="+",required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args(argv);root=args.result.resolve(strict=True)
    require(not args.output.exists(),"assessment output must be new")
    report=read_json(root/"summary.json")
    require(report["schema"]==run.SCHEMA and report["status"] in ("complete","partial_failure"),"motion capture not finalized")
    audit_configuration(report)
    candidate_kind=run.adapter.kind(report["contract"])
    require(report.get("candidate_backend","sparse")==candidate_kind,"saved candidate kind/contract differs")
    require(run.digest(root/"inputs.json")==report["inputs_sha256"],"motion inputs changed")
    binding=read_json(root/"inputs.json")
    rows=report["rounds"][0]["runs"]
    selected={}
    for name in ("baseline16","sparse"):
        candidates=[r for r in rows if r["name"]==name]
        row=next((r for r in candidates if r["channels"]=="rgb"),candidates[0])
        directory,value=read_row(root,row)
        if name=="sparse":
            run.media_profile_name(value,report["contract"])
            for ordinal,f in enumerate(value["frames"]):
                run.check_actual(f["actual"],report["contract"],11+ordinal)
                if candidate_kind=="compact":
                    run.adapter.check_compact(f["compact"],value["extent"],11+ordinal,f["actual"])
        selected[name]=(directory,{f["pose_index"]:f for f in value["frames"]})
    all_rows=[r for r in rows if r["channels"]=="all"]
    require(len(all_rows)==2,"independent geometry audit needs baseline and candidate all-channel groups")
    for row in all_rows:
        _,value=read_row(root,row)
        if value["name"]=="sparse":run.media_profile_name(value,report["contract"])
    run.compare(all_rows)
    geometry={}
    baseline_all=next(r for r in all_rows if r["name"]=="baseline16")
    directory,value=read_row(root,baseline_all)
    for f in value["frames"]:
        if f["pose_index"] in run.REGISTERED:geometry[f["pose_index"]]=geometry_packet(directory,f)
    require(set(geometry)==set(run.REGISTERED),"missing registered actual ID/depth/valid packet")
    refs=references(args.references,report,binding,selected["baseline16"][1],selected["sparse"][1],geometry)
    images={pose:{name:metrics.image(directory/by[pose]["png"],by[pose]["png_sha256"],by[pose]["rgb_sha256"])
        for name,(directory,by) in selected.items()} for pose in run.REGISTERED}
    scores={};missing=[];failed=False
    for pose in run.REGISTERED:
        subsets={};ids=geometry[pose][1]["instance_id"]
        for name,identities in metrics.ROI_IDS.items():
            mask=np.isin(ids,identities);count=int(mask.sum());estimates=[]
            if count<metrics.MIN_PIXELS:missing.append(f"{pose}:{name}:insufficient_pixels")
            for ref in refs:
                values={}
                for mode in ("baseline16","sparse"):
                    error=(images[pose][mode]-ref["images"][pose])[mask]
                    values[mode]=dict(rmse=float(np.sqrt(np.mean(error**2))) if count else None,
                        signed_bias_rgb=error.mean(axis=0).tolist() if count else None,
                        signed_luminance_bias=float((error@np.array([.2126,.7152,.0722])).mean()) if count else None)
                passed=(values["sparse"]["rmse"]<=values["baseline16"]["rmse"]+metrics.RMSE_ALLOWANCE) if count>=metrics.MIN_PIXELS else None
                failed|=passed is False;estimates.append(dict(seed=ref["seed"],**values,observed_gate=passed))
            disagreement=None
            if len(refs)>1 and count:
                disagreement=max(float(np.sqrt(np.mean((a["images"][pose][mask]-b["images"][pose][mask])**2)))
                    for i,a in enumerate(refs) for b in refs[i+1:])
            resolved=disagreement is not None and disagreement<=metrics.REFERENCE_DISAGREEMENT_LIMIT
            if not resolved:missing.append(f"{pose}:{name}:reference_noise_unresolved")
            subsets[name]=dict(ids=list(identities),pixels=count,estimates=estimates,
                reference_disagreement_rmse=disagreement,reference_resolution_screen=resolved,
                passed=all(x["observed_gate"] is True for x in estimates) if resolved and count>=metrics.MIN_PIXELS else None)
        scores[str(pose)]=subsets
    by={x["pose_index"]:x for x in binding["actual_full_inputs"]}
    pairs,correspondence=metrics.rigid_correspondences(binding["scene"],by[59],by[60],geometry[59][1],geometry[60][1])
    temporal=[]
    if len(pairs)>=metrics.MIN_PIXELS:
        for ref in refs:
            values={name:metrics.temporal_p95(images[59][name],images[60][name],ref["images"][59],ref["images"][60],pairs)
                for name in ("baseline16","sparse")}
            passed=values["sparse"]["p95"]<=values["baseline16"]["p95"];failed|=not passed
            temporal.append(dict(seed=ref["seed"],**values,observed_gate=passed))
    else:missing.append("59->60:insufficient_reliable_finger_correspondences")
    result=dict(schema="phymeo-preview-sparse-motion-assessment-v1",
        status="failed_observed_gate" if failed else "insufficient_evidence" if missing else "passed_registered_gates",
        source_summary_sha256=run.digest(root/"summary.json"),metric_implementation_sha256=run.digest(metrics.__file__),
        space="inverse IEC sRGB display-linear; not scene radiance",poses=scores,
        roi_definition=dict(identities=metrics.ROI_IDS,source_sha256=metrics.ROI_SOURCE_SHA256,intensity_selection=False),
        temporal=dict(pair=[59,60],correspondence=correspondence,estimates=temporal),missing_evidence=missing,
        gates=dict(rmse_allowance=metrics.RMSE_ALLOWANCE,reference_resolution_screen=metrics.REFERENCE_DISAGREEMENT_LIMIT,
            temporal_p95_lte_baseline=True,min_pixels=metrics.MIN_PIXELS),
        references=[{k:v for k,v in r.items() if k!="images"} for r in refs],
        source_run_status=report["status"],resolution=report["resolution"],camera=report.get("camera","detail"),
        candidate_backend=candidate_kind,candidate_contract=report["contract"],
        candidate_group_key="sparse (legacy generic slot; actual kind is candidate_backend)",
        per_round_timing=[dict(index=r["index"],groups=[dict(name=x["name"],channels=x["channels"],
            timing=x["timing"],phase_timing=x.get("phase_timing"),identity_audit=x.get("identity_audit"))
            for x in r["runs"] if x["status"]=="complete"]) for r in report["rounds"]],
        scope="three registered real poses and one adjacent reliable pair only; finite noisy references; not300-frame quality proof",
        thin_board_scope="separate25-pose fixture and69 right-finger reveal pixels NOT measured by this studio run")
    args.output.parent.mkdir(parents=True,exist_ok=True);run.save(args.output,result)
    print(json.dumps(dict(status=result["status"],missing_evidence=missing)))


if __name__=="__main__":main()