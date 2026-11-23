"""Offline same-method A/B and sealed independent 8192-proposal references.

Uses the existing registered identity/rigid-correspondence metrics. No render,
physics, denoiser, exposure fit, or candidate-intensity-based ROI selection.
"""
import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

from . import preview as p


def load_metrics(delivery):
    folder=Path(delivery).resolve().parent/"preview_quad"
    prior=list(sys.path)
    try:
        sys.path.insert(0,str(folder))
        spec=importlib.util.spec_from_file_location("rigid_registered_metrics",folder/"score.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:sys.path[:]=prior
    return module


def load_row(root,row):
    receipt=root/row["receipt"]
    p.require(receipt.resolve().is_relative_to(root) and p.digest(receipt)==row["receipt_sha256"],"motion receipt changed")
    value=p.read(receipt)
    p.require(value=={k:v for k,v in row.items() if k not in ("receipt","receipt_sha256")},"summary/receipt differ")
    p.require(value["status"]=="complete","motion owner did not complete")
    return receipt.parent,value


def geometry(folder,row,np):
    path=folder/row["geometry_file"]
    p.require(p.digest(path)==row["geometry_sha256"],"geometry packet changed")
    with np.load(path,allow_pickle=False) as source:data={k:source[k].copy() for k in ("instance_id","depth","valid")}
    hashes={"instance_id_u32":p.digest_bytes(data["instance_id"].astype("<u4",copy=False).tobytes()),
            "forward_depth_m_f32":p.digest_bytes(data["depth"].astype("<f4",copy=False).tobytes()),
            "valid_u8":p.digest_bytes(data["valid"].astype("u1",copy=False).tobytes())}
    p.require(hashes==row["sensors"],"geometry bytes differ from capture")
    p.require(data["instance_id"].shape==(720,1280),"registered geometry must be native720p")
    return data


def check_identity(a,b):p.require(all(a[k]==b[k] for k in p.IDENTITY),"saved actual frame/camera differs")


def references(paths,binding,rows,grid,metrics):
    result=[];model=None
    for directory in paths:
        directory=Path(directory).resolve(strict=True);value=p.read(directory/"summary.json")
        p.verify_reference_header(value)
        p.require(p.digest(directory/"inputs.json")==value["inputs_sha256"],"reference input changed")
        source=p.read(directory/"inputs.json")
        p.require(source["trace_sha256"]==binding["trace_sha256"] and
            p.clean_source(source["scene"])==p.clean_source(binding["scene"]) and
            p.clean_source(source["studio"])==p.clean_source(binding["studio"]),"reference scene/light/material/source differs")
        native=value["source"]["baseline"]
        fingerprint={k:native[k] for k in ("library_sha256","shaders")}
        if model is not None:p.require(model==fingerprint,"independent references used different native models")
        model=fingerprint;by={r["pose_index"]:r for r in value["records"]};images={}
        for pose in p.REGISTERED:
            row=by[pose];current=rows["current"][pose];history=rows["rigid-history"][pose]
            check_identity(row,current);check_identity(row,history)
            p.require(row["sensors"]==current["sensors"]==history["sensors"],"reference actual sensor bytes differ")
            diag=row["backend"];ind=diag["indirect"];progressive=diag["progressive"]
            p.require(diag["library_sha256"]==native["library_sha256"] and
                all(v in native["shaders"].values() for v in diag["shader_sha256"].values()),"reference loaded unbound shader/DLL")
            p.require(ind["mode"]=="diffuse_bounce" and ind["bounces"]==1 and ind["seed"]==value["seed"] and
                ind["samples_per_pixel"]==16 and progressive["batches"]==value["batches"] and
                progressive["primary_samples_per_pixel"]==value["primary_proposals_per_pixel"] and
                diag["area_light_grid"]==grid and diag["exposure"]==8. and diag["display_transform"]=="pbr-neutral" and
                diag["reflection"]["mode"]=="single_bounce" and not diag["spatial_filter"]["last_capture_filtered"],
                "reference actual lighting/sample/display contract differs")
            p.require(diag["decoded_geometry_sha256"]==history["backend"]["decoded_geometry_sha256"],"decoded geometry differs")
            images[pose]=metrics.image(directory/row["png"],row["png_sha256"],row["rgb_sha256"])
        result.append(dict(seed=value["seed"],batches=value["batches"],file=str(directory/"summary.json"),
            sha256=p.digest(directory/"summary.json"),images=images,model=fingerprint))
    p.require(len(result)>=2 and len({r["seed"] for r in result})==len(result),"two genuinely independent reference seeds required")
    return result


def historical_baseline(root,binding,rows,model,metrics):
    root=Path(root).resolve(strict=True);summary=p.read(root/"summary.json")
    p.require(summary["schema"]=="phymeo-preview-sparse-motion-v1" and summary["status"]=="complete","sealed baseline is incomplete")
    p.require(p.digest(root/"inputs.json")==summary["inputs_sha256"],"sealed baseline input changed")
    source=p.read(root/"inputs.json")
    p.require(p.clean_source(source["scene"])==p.clean_source(binding["scene"]) and
              p.clean_source(source["studio"])==p.clean_source(binding["studio"]) and source["trace_sha256"]==binding["trace_sha256"],
              "sealed baseline source differs")
    p.require(all(summary["source"]["baseline"][k]==model[k] for k in model),"sealed baseline and high-reference model differ")
    choices=summary["rounds"][0]["runs"]
    rgb=next(r for r in choices if r["name"]=="baseline16" and r["channels"]=="rgb")
    sensors=next(r for r in choices if r["name"]=="baseline16" and r["channels"]=="all")
    folder,value=load_row(root,rgb);_,all_value=load_row(root,sensors)
    by={r["pose_index"]:r for r in value["frames"]};all_by={r["pose_index"]:r for r in all_value["frames"]}
    images={}
    for pose in p.REGISTERED:
        check_identity(by[pose],rows["current"][pose]);check_identity(all_by[pose],rows["current"][pose])
        p.require(all_by[pose]["sensors"]==rows["current"][pose]["sensors"],"sealed baseline sensors differ")
        images[pose]=metrics.image(folder/by[pose]["png"],by[pose]["png_sha256"],by[pose]["rgb_sha256"])
    return dict(images=images,summary=str(root/"summary.json"),sha256=p.digest(root/"summary.json"),
        timing=value["timing"],timing_scope="previously recorded different DLL/full-rate algorithm; not this paired timing experiment")


def crop_panels(destination,geometry_by,groups,root,reference_paths,rows):
    """Native-pixel crops, bounded only by fixed geometric ID masks."""
    import numpy as np
    from PIL import Image,ImageDraw
    destination.mkdir(parents=True,exist_ok=False)
    pose=60;ids=geometry_by[pose]["instance_id"];receipts=[]
    candidates=[]
    for mode in ("current","rigid-history"):
        row=rows[mode][pose];folder=(root/next(v["receipt"] for v in groups if v["name"]==mode)).parent
        candidates.append((mode,folder/row["png"],row["png_sha256"]))
    for path in reference_paths:
        value=p.read(path/"summary.json");row=next(r for r in value["records"] if r["pose_index"]==pose)
        candidates.append((f"reference-seed{value['seed']}",path/row["png"],row["png_sha256"]))
    for name,identities in (("left-arm",(112,113,115,117)),("right-finger",(158,)),("white-shell",(150,151,154))):
        yy,xx=np.nonzero(np.isin(ids,identities))
        if not len(xx):continue
        box=(max(0,int(xx.min())-8),max(0,int(yy.min())-8),min(1280,int(xx.max())+9),min(720,int(yy.max())+9))
        width,height=box[2]-box[0],box[3]-box[1];cell=max(width,240)
        panel=Image.new("RGB",(len(candidates)*cell,height+44),(245,245,245));draw=ImageDraw.Draw(panel)
        for index,(label,file,expected) in enumerate(candidates):
            p.require(p.digest(file)==expected,"crop source changed")
            with Image.open(file) as image:panel.paste(image.crop(box),(index*cell+(cell-width)//2,44))
            draw.text((index*cell+5,5),label,fill=(20,20,20))
        draw.text((5,25),f"pose60 / {name} / original pixels / geometry-selected crop",fill=(20,20,20))
        path=destination/(name+".png");panel.save(path)
        receipts.append(dict(file=path.name,sha256=p.digest(path),pose=pose,ids=list(identities),box=list(box),
            sources=[dict(label=n,path=str(f.resolve()),sha256=h) for n,f,h in candidates],
            scope="unchanged native-pixel rectangular crops; not ROI-masked statistics or new render"))
    p.save(destination/"panels.json",receipts)
    return dict(path=str(destination.resolve()),sha256=p.digest(destination/"panels.json"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result",type=Path,required=True)
    parser.add_argument("--delivery",type=Path,required=True)
    parser.add_argument("--references",type=Path,nargs="+",required=True)
    parser.add_argument("--baseline",type=Path)
    parser.add_argument("--panels",type=Path,help="optional NEW directory of unchanged native-pixel pose60 crops")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    root=args.result.resolve(strict=True);report=p.read(root/"summary.json")
    p.require(not args.output.exists(),"assessment output must be new")
    p.require(report["schema"]==p.SCHEMA and report["status"]=="complete" and report["channels"]=="all" and
        report["resolution"]=="720p" and report["camera"]=="detail" and report["studio_look"]=="recorded",
        "quality assessment requires completed registered720p all-plane run")
    p.require(p.digest(root/"inputs.json")==report["inputs_sha256"],"motion inputs changed")
    binding=p.read(root/"inputs.json");inputs=p.registered_source(binding)
    metrics=load_metrics(args.delivery)
    groups={};rows={};images={pose:{} for pose in p.REGISTERED};geometry_by={}
    for row in report["runs"]:
        folder,value=load_row(root,row);name=value["name"];groups[name]=value
        p.require(p.digest(folder/value["journal"])==value["journal_sha256"],"incremental motion journal changed")
        rows[name]={f["pose_index"]:f for f in value["frames"]}
        p.require(tuple(rows[name])==p.POSES,"full motion predecessor chain missing")
        for index,record in rows[name].items():
            source=inputs[index]
            p.require(record["frame_key"]==source["frame"]["key"] and record["time_s"]==source["frame"]["time_s"] and
                record["state_sha256"]==p.digest_bytes(p.canonical(source["frame"])) and
                record["camera_sha256"]==p.digest_bytes(p.canonical(source["camera"])) and
                record["presentation"]==source["presentation"],"presented source binding differs")
            actual=record["history_actual"]
            expected_seed=report["seed"]^((actual["sequence"]*0x9e3779b9)&0xffffffff)
            p.require(record["effective_seed"]==expected_seed and actual["committed_rgb"]==index+11,
                      "actual sample sequence or accepted capture count differs")
        for pose in p.REGISTERED:
            record=rows[name][pose]
            images[pose][name]=metrics.image(folder/record["png"],record["png_sha256"],record["rgb_sha256"])
            data=geometry(folder,record,np)
            if pose in geometry_by:
                p.require(all(np.array_equal(data[k],geometry_by[pose][k]) for k in data),"paired actual geometry packet differs")
            else:geometry_by[pose]=data
            for signal in ("current","history","linear"):
                p.require(p.digest(folder/record[signal+"_file"])==record[signal+"_sha256"],"registered debug signal changed")
    p.verify_pair(groups["current"]["frames"],groups["rigid-history"]["frames"],require_reset_rgb=True)
    p.verify_chain(groups["rigid-history"]["frames"])
    p.require(rows["rigid-history"][0]["history_actual"]["last_reset"] and
              rows["rigid-history"][0]["history_actual"]["sequence"]==0,"warmup history was not discarded")
    refs=references(args.references,binding,rows,report["contract"]["area_light_grid"],metrics)
    p.require(all(r["seed"]!=report["seed"] for r in refs),"candidate seed cannot masquerade as an independent reference")
    baseline=None
    if args.baseline:
        baseline=historical_baseline(args.baseline,binding,rows,refs[0]["model"],metrics)
        for pose in p.REGISTERED:images[pose]["archived-full-rate16"]=baseline["images"][pose]
    roi_ids=dict(metrics.ROI_IDS,right_finger=(158,))
    scores={};missing=[];observed_failure=False
    for pose in p.REGISTERED:
        ids=geometry_by[pose]["instance_id"];subsets={}
        for name,identities in roi_ids.items():
            mask=np.isin(ids,identities);count=int(mask.sum());estimates=[]
            for reference in refs:
                values={}
                for mode,image in images[pose].items():
                    error=(image-reference["images"][pose])[mask]
                    values[mode]=dict(rmse=float(np.sqrt(np.mean(error**2))) if count else None,
                        signed_bias_rgb=error.mean(axis=0).tolist() if count else None)
                nonworse=count>=metrics.MIN_PIXELS and values["rigid-history"]["rmse"]<=values["current"]["rmse"]+metrics.RMSE_ALLOWANCE
                if count>=metrics.MIN_PIXELS:observed_failure|=not nonworse
                estimates.append(dict(seed=reference["seed"],errors=values,history_nonworse_than_current=nonworse if count>=metrics.MIN_PIXELS else None))
            disagreement=max(float(np.sqrt(np.mean((a["images"][pose]-b["images"][pose])[mask]**2)))
                for i,a in enumerate(refs) for b in refs[i+1:]) if count else None
            resolved=count>=metrics.MIN_PIXELS and disagreement<=metrics.REFERENCE_DISAGREEMENT_LIMIT
            if not resolved:missing.append(f"{pose}:{name}:reference_or_roi_unresolved")
            subsets[name]=dict(ids=list(identities),pixels=count,estimates=estimates,reference_disagreement_rmse=disagreement,
                reference_resolution_screen=resolved,passed=all(e["history_nonworse_than_current"] for e in estimates) if resolved else None)
        scores[str(pose)]=subsets
    pairs,correspondence=metrics.rigid_correspondences(binding["scene"],inputs[59],inputs[60],geometry_by[59],geometry_by[60])
    temporal=[]
    if len(pairs)>=metrics.MIN_PIXELS:
        for ref in refs:
            values={name:metrics.temporal_p95(images[59][name],images[60][name],ref["images"][59],ref["images"][60],pairs) for name in images[59]}
            good=values["rigid-history"]["p95"]<=values["current"]["p95"]
            observed_failure|=not good;temporal.append(dict(seed=ref["seed"],values=values,history_nonworse_than_current=good))
    else:missing.append("59->60:insufficient_reliable_correspondence")
    result=dict(schema="phymeo-preview-rigid-history-assessment-v1",
        status="failed_observed_gate" if observed_failure else "insufficient_evidence" if missing else "passed_registered_gates",
        source_summary_sha256=p.digest(root/"summary.json"),metric_source=str(Path(metrics.__file__).resolve()),metric_sha256=p.digest(metrics.__file__),
        references=[{k:v for k,v in r.items() if k not in ("images","model")} for r in refs],
        space="inverse IEC sRGB display-linear; not raw HDR",poses=scores,
        roi_definition=dict(ids=roi_ids,source_sha256=metrics.ROI_SOURCE_SHA256,
            selection="fixed geometric instance IDs; right_finger is a fixed subset158; no candidate intensity selection"),
        temporal=dict(pair=[59,60],correspondence=correspondence,estimates=temporal),missing_evidence=missing,
        timing={name:value["timing"] for name,value in groups.items()},channels=report["channels"],
        actual_rigid_reuse_pixels=report["checks"]["rigid_reuse_pixels"],
        registered_history={str(i):dict(age=rows["rigid-history"][i]["age_histogram"],
            actual=rows["rigid-history"][i]["history_actual"],rigid=rows["rigid-history"][i]["rigid"]) for i in p.REGISTERED},
        gates=dict(rmse_allowance=metrics.RMSE_ALLOWANCE,reference_disagreement=metrics.REFERENCE_DISAGREEMENT_LIMIT,
            temporal_p95_lte_same_method_current=True,min_pixels=metrics.MIN_PIXELS),
        baseline=None if baseline is None else {k:v for k,v in baseline.items() if k!="images"},
        scope="300 real moving poses captured; quality limited to59/60/149 and reliable59->60 finger correspondence. Hidden secondary visibility remains unproven.")
    if args.panels:
        result["panels"]=crop_panels(args.panels,geometry_by,report["runs"],root,args.references,rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);p.save(args.output,result)
    print(result["status"])


if __name__=="__main__":main()
