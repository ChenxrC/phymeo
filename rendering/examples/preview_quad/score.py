"""Offline same-model Panda five-identity-ROI and registered finger-time audit.

Input images remain untouched. This does not run a renderer or invent blocker
evidence. A single noisy reference reports comparisons but cannot sign quality.
"""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from build_overlay import digest, require, save
from motion import SCHEMA, clean_uris

ROI_IDS={"white-shell":(150,151,154),"fingers":(156,158),"rubber":(2,),
         "left-arm":(112,113,115,117),"wrist-shell":(125,142,149)}
ROI_SOURCE_SHA256="4875708e5abe4e34bea83377c09e7d6447e0b1efecbf93566c25771d587e92b0"
RMSE_ALLOWANCE=.005
REFERENCE_DISAGREEMENT_LIMIT=.0025
MIN_PIXELS=16
TEMPORAL_PAIR=(59,60)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def linear(rgb):
    x=rgb.astype(np.float64)/255.
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def image(path,expected_png,expected_rgb):
    require(digest(path)==expected_png,"PNG bytes changed")
    with Image.open(path) as im:
        require(im.mode=="RGB","expected original RGB PNG")
        value=np.asarray(im).copy()
    require(sha256(value.tobytes()).hexdigest()==expected_rgb,"decoded image differs from published RGB")
    return linear(value)


def pose_matrix(value):
    x,y,z,w=value["orientation_xyzw"]
    q=np.array((x,y,z,w),dtype=np.float64)
    norm=float(np.linalg.norm(q))
    require(math.isfinite(norm) and abs(norm-1)<=1e-6,"invalid registered quaternion")
    x,y,z,w=q/norm
    m=np.eye(4)
    m[:3,:3]=((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
               (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
               (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))
    m[:3,3]=value["position_m"]
    return m


def interior(mask):
    h,w=mask.shape
    padded=np.pad(mask,1,constant_values=False)
    return np.logical_and.reduce([padded[y:y+h,x:x+w] for y in range(3) for x in range(3)])


def rigid_correspondences(scene,previous,current,old_geometry,new_geometry):
    """Current pixel -> same-instance previous pixel, no image optical-flow fitting."""
    def state(record):
        camera=record["camera"]
        cam=pose_matrix(camera["world_pose"])
        nodes={n["node_id"]:pose_matrix(n["world_pose"]) for n in record["frame"]["nodes"]}
        return camera,cam,nodes
    pc,pm,pnodes=state(previous)
    cc,cm,cnodes=state(current)
    ids,z,valid=(new_geometry[k] for k in ("instance_id","depth","valid"))
    oldids,oldz,oldvalid=(old_geometry[k] for k in ("instance_id","depth","valid"))
    h,w=ids.shape
    require(oldids.shape==(h,w) and pc["width"]==cc["width"]==w and
            pc["height"]==cc["height"]==h,"temporal geometry extent differs")
    transforms={i["id"]:i for i in scene["instances"]}
    rows=[]
    proposed=0
    for identity in ROI_IDS["fingers"]:
        good=interior((ids==identity)&(valid==1)&np.isfinite(z)&(z>0))
        oldgood=interior((oldids==identity)&(oldvalid==1)&np.isfinite(oldz)&(oldz>0))
        yy,xx=np.nonzero(good)
        proposed+=len(xx)
        if not len(xx):
            continue
        instance=transforms[identity]
        local=pose_matrix(instance["local_pose"])
        cw=cnodes[instance["node_id"]]@local
        pw=pnodes[instance["node_id"]]@local
        depth=z[yy,xx].astype(float)
        rays=np.stack(((xx+.5-cc["cx_px"])/cc["fx_px"],
                       -(yy+.5-cc["cy_px"])/cc["fy_px"],-np.ones(len(xx))))
        points=np.vstack((rays*depth,np.ones(len(xx))))
        previous_camera=np.linalg.inv(pm)@pw@np.linalg.inv(cw)@cm@points
        expected=-previous_camera[2]
        ok=np.isfinite(expected)&(expected>0)
        px=np.full(len(xx),-1,dtype=int)
        py=np.full(len(xx),-1,dtype=int)
        px[ok]=np.floor(pc["fx_px"]*previous_camera[0,ok]/expected[ok]+pc["cx_px"]).astype(int)
        py[ok]=np.floor(-pc["fy_px"]*previous_camera[1,ok]/expected[ok]+pc["cy_px"]).astype(int)
        ok&=(px>=0)&(px<w)&(py>=0)&(py<h)
        candidate=np.nonzero(ok)[0]
        if not len(candidate):
            continue
        candidate=candidate[oldgood[py[candidate],px[candidate]]]
        measured=oldz[py[candidate],px[candidate]].astype(float)
        candidate=candidate[np.abs(measured-expected[candidate])<=.003*np.maximum(measured,expected[candidate])]
        if not len(candidate):
            continue
        measured=oldz[py[candidate],px[candidate]].astype(float)
        oldrays=np.stack(((px[candidate]+.5-pc["cx_px"])/pc["fx_px"],
                          -(py[candidate]+.5-pc["cy_px"])/pc["fy_px"],-np.ones(len(candidate))))
        back=np.linalg.inv(cm)@cw@np.linalg.inv(pw)@pm@np.vstack((oldrays*measured,np.ones(len(candidate))))
        backz=-back[2]
        uv=np.stack((cc["fx_px"]*back[0]/backz+cc["cx_px"],
                    -cc["fy_px"]*back[1]/backz+cc["cy_px"]))
        dist=np.sqrt((uv[0]-(xx[candidate]+.5))**2+(uv[1]-(yy[candidate]+.5))**2)
        reliable=(backz>0)&np.isfinite(dist)&(dist<=.75)&(
            np.abs(backz-depth[candidate])<=.003*np.maximum(backz,depth[candidate]))
        candidate=candidate[reliable]
        rows.extend(zip((yy[candidate]*w+xx[candidate]).tolist(),
                        (py[candidate]*w+px[candidate]).tolist()))
    pairs=np.array(rows,dtype=np.int64).reshape(-1,2)
    return pairs,dict(proposed_pixels=proposed,accepted_pixels=len(pairs),
        identity_ids=list(ROI_IDS["fingers"]),relative_depth_limit=.003,roundtrip_pixel_limit=.75,
        method="registered rigid instance/camera transforms and actual depth; nearest pixel, eroded same-ID endpoints",
        new_background_excluded=True,scope="screen resampling introduces finite correspondence error; same mapping for candidate/baseline/reference")


def temporal_p95(previous,current,ref_previous,ref_current,pairs):
    require(len(pairs)>0,"no temporal correspondences")
    now,old=pairs[:,0],pairs[:,1]
    residual=(current.reshape(-1,3)[now]-previous.reshape(-1,3)[old]
              -ref_current.reshape(-1,3)[now]+ref_previous.reshape(-1,3)[old])
    per_pixel=np.sqrt(np.mean(residual**2,axis=1))
    ranked=np.sort(per_pixel)
    return dict(count=len(pairs),p95=float(ranked[math.ceil(.95*len(ranked))-1]),
                mean_bias_rgb=residual.mean(axis=0).tolist(),
                scope="reference-compensated adjacent-frame residual; same geometric correspondence")


def read_run(round_root,row):
    path=round_root/row["receipt"]
    require(digest(path)==row["receipt_sha256"],"motion receipt changed")
    value=json.loads(path.read_text(encoding="utf-8-sig"))
    require(value["status"]=="complete","incomplete motion receipt")
    return path.parent,value


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion",type=Path,required=True)
    ap.add_argument("--references",type=Path,nargs="+",required=True,
                    help="completed validate.py directories with an actual seed1 64-batch reference; independent seeds may be added")
    ap.add_argument("--variant",required=True,help="e.g. quad2-720p; must be RGB at native720p")
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    root=args.motion.resolve(strict=True)
    report=json.loads((root/"experiment.json").read_text(encoding="utf-8-sig"))
    require(report["schema"]==SCHEMA and report["status"]=="complete","motion run incomplete")
    round_info=report["rounds"][0]
    round_file=root/round_info["file"]
    require(digest(round_file)==round_info["sha256"],"round summary changed")
    round_root=round_file.parent
    round_report=json.loads(round_file.read_text(encoding="utf-8-sig"))
    require(digest(round_root/"inputs.json")==round_report["inputs_sha256"],"motion inputs changed")
    binding=json.loads((round_root/"inputs.json").read_text(encoding="utf-8-sig"))
    candidate_row=next(r for r in round_report["runs"] if r["preset"]==args.variant and r["channels"]=="rgb")
    require(candidate_row["extent"]==[1280,720],"five-ROI720p gate cannot be applied to upsampled low-resolution frames")
    candidate_dir,candidate=read_run(round_root,candidate_row)
    baseline_row=next((r for r in round_report["runs"] if r["preset"]=="baseline16-720p" and r["channels"]=="rgb"),None)
    if baseline_row:
        baseline_dir,baseline=read_run(round_root,baseline_row)
    else:
        found=[]
        for cached in round_report["cached"]:
            path=Path(cached["receipt_file"])
            require(digest(path)==cached["receipt_sha256"],"cached baseline changed")
            value=json.loads(path.read_text(encoding="utf-8-sig"))
            if value["extent"]==[1280,720] and value["channels"]=="rgb":
                found.append((path.parent,value))
        require(len(found)==1,"need one unambiguous baseline16 RGB group")
        baseline_dir,baseline=found[0]
    baseline_by={x["pose_index"]:x for x in baseline["frames"]}
    candidate_by={x["pose_index"]:x for x in candidate["frames"]}
    poses=(59,60,149)
    geometry={}
    for g in report.get("geometry",[]):
        if g["resolution"]!="720p" or g["pose_index"] not in poses:
            continue
        path=root/g["file"]
        require(digest(path)==g["sha256"],"identity packet changed")
        with np.load(path,allow_pickle=False) as arrays:
            data={k:arrays[k].copy() for k in ("instance_id","depth","valid")}
        hashes={"instance_id_u32":sha256(data["instance_id"].astype("<u4",copy=False).tobytes()).hexdigest(),
                "forward_depth_m_f32":sha256(data["depth"].astype("<f4",copy=False).tobytes()).hexdigest(),
                "valid_u8":sha256(data["valid"].astype("u1",copy=False).tobytes()).hexdigest()}
        require(hashes==g["sensors"],"identity packet bytes mismatch")
        geometry[g["pose_index"]]=(g,data)
    require(set(geometry)==set(poses),"need all registered59/60/149 geometry packets")
    references=[]
    for directory in args.references:
        summary_file=directory/"summary.json"
        value=json.loads(summary_file.read_text(encoding="utf-8-sig"))
        require(value["schema"]=="phymeo-preview-quad-validation-v1" and value["status"]=="complete",
                "reference is not a completed actual quad-validation capture")
        require(digest(directory/"inputs.json")==value["inputs_sha256"] and
                value["library"]["sha256"]==report["source"]["library_sha256"] and
                value["overlay"]["base_shaders"]==report["source"]["shaders"],
                "reference native model/runtime differs")
        rb=json.loads((directory/"inputs.json").read_text(encoding="utf-8-sig"))
        require(clean_uris(rb["scene"])==clean_uris(binding["scene"]) and
                clean_uris(rb["studio"])==clean_uris(binding["studio"]),"reference assets/lighting differ")
        rr=next(r for r in value["runs"] if r["name"]=="reference")
        require(rr["status"]=="complete" and rr["proposal_budget"]==16 and rr["batches_per_pose"]>=64
                and rr["seed"]!=0,"need a nonzero independent seed and at least1024 same-model proposals")
        rows={x["pose_index"]:x for x in rr["records"]}
        images={}
        for pose in poses:
            f=rows[pose]
            b=baseline_by[pose]
            c=candidate_by[pose]
            g=geometry[pose][0]
            for key in ("frame_key","time_s","state_sha256","camera_sha256"):
                require(f[key]==b[key]==c[key]==g[key],"reference/candidate actual pose or camera differs")
            diag=f["backend"]
            require(f["sensor_sha256"]==g["sensors"] and diag["indirect"]["bounces"]==1
                    and diag["indirect"]["seed"]==rr["seed"]
                    and diag["progressive"]["batches"]==rr["batches_per_pose"]
                    and diag["progressive"]["primary_samples_per_pixel"]==16*rr["batches_per_pose"]
                    and diag["shader_sha256"]["frag"]==report["source"]["shaders"][
                        "resident_rayquery_normal_reflection_indirect_progressive.frag.spv"]
                    and diag["exposure"]==8. and diag["display_transform"]=="pbr-neutral"
                    and diag["area_light_grid"]==2 and diag["reflection"]["mode"]=="single_bounce"
                    and diag["spatial_filter"]["last_capture_filtered"] is False,
                    "reference geometry, independent sampling, final unfiltered display or diffuse model differs")
            images[pose]=image(directory/"reference"/f["png"],f["png_sha256"],f["rgb_sha256"])
        references.append(dict(seed=rr["seed"],images=images,summary_sha256=digest(summary_file),
                               source=str(summary_file.resolve())))
    require(len({r["seed"] for r in references})==len(references),"duplicate reference seeds cannot establish uncertainty")
    actual={}
    missing=[]
    for pose in poses:
        actual[pose]={}
        for mode,path,rows in (("baseline",baseline_dir,baseline_by),("candidate",candidate_dir,candidate_by)):
            f=rows[pose]
            actual[pose][mode]=image(path/f["png"],f["png_sha256"],f["rgb_sha256"])
        ids=geometry[pose][1]["instance_id"]
        subsets={}
        for name,identities in ROI_IDS.items():
            mask=np.isin(ids,identities)
            count=int(mask.sum())
            if count<MIN_PIXELS:
                missing.append(f"{pose}:{name}:too_few_pixels")
            estimates=[]
            for r in references:
                def stats(mode):
                    error=(actual[pose][mode]-r["images"][pose])[mask]
                    return dict(rmse=float(np.sqrt(np.mean(error**2))) if count else None,
                                signed_bias_rgb=error.mean(axis=0).tolist() if count else None)
                b,c=stats("baseline"),stats("candidate")
                estimates.append(dict(seed=r["seed"],baseline=b,candidate=c,
                    observed_gate=(c["rmse"]<=b["rmse"]+RMSE_ALLOWANCE) if count>=MIN_PIXELS else None))
            disagreement=None
            if len(references)>=2 and count:
                disagreement=max(float(np.sqrt(np.mean((a["images"][pose][mask]-b["images"][pose][mask])**2)))
                    for i,a in enumerate(references) for b in references[i+1:])
            resolved=disagreement is not None and disagreement<=REFERENCE_DISAGREEMENT_LIMIT
            if not resolved:
                missing.append(f"{pose}:{name}:reference_noise_unresolved")
            subsets[name]=dict(ids=list(identities),pixels=count,estimates=estimates,
                independent_reference_disagreement_rmse=disagreement,reference_resolution_screen=resolved,
                passed=all(x["observed_gate"] is True for x in estimates) if resolved and count>=MIN_PIXELS else None)
        actual[pose]["roi"]=subsets
    full_inputs={x["pose_index"]:x for x in binding["actual_full_inputs"]["720p"]}
    pairs,correspondence=rigid_correspondences(binding["scene"],full_inputs[59],full_inputs[60],
                                              geometry[59][1],geometry[60][1])
    temporal=[]
    if len(pairs)>=MIN_PIXELS:
        for r in references:
            values={mode:temporal_p95(actual[59][mode],actual[60][mode],
                    r["images"][59],r["images"][60],pairs) for mode in ("baseline","candidate")}
            temporal.append(dict(seed=r["seed"],**values,
                observed_gate=values["candidate"]["p95"]<=values["baseline"]["p95"]))
    else:
        missing.append("59->60:insufficient_reliable_finger_correspondences")
    output=dict(schema="phymeo-preview-quad-roi-v1",motion_sha256=digest(root/"experiment.json"),
                variant=args.variant,space="inverse IEC sRGB display-linear; not scene radiance",
                roi_definition=dict(ids=ROI_IDS,source_file="panda_appearance_diagnosis.py",
                                    source_sha256=ROI_SOURCE_SHA256,intensity_selection=False),
                poses={str(p):actual[p]["roi"] for p in poses},
                gates=dict(static_rmse_allowance=RMSE_ALLOWANCE,
                           reference_disagreement_screen=REFERENCE_DISAGREEMENT_LIMIT,
                           temporal_candidate_p95_lte_baseline=True,min_pixels=MIN_PIXELS),
                temporal=dict(pair=list(TEMPORAL_PAIR),correspondence=correspondence,estimates=temporal),
                reference_sources=[{k:v for k,v in r.items() if k!="images"} for r in references],
                missing_evidence=missing,
                reference_scope="finite independent samples; seed disagreement is a resolution screen, not a confidence interval",
                blocker_scope="not measured here: separate guided-motion fixture's69 newly-visible right-finger pixels are not this scene",
                full_trajectory_quality_scope="only registered three poses and one true adjacent pair; does not sign all300 frames")
    failures=any(x["observed_gate"] is False for p in actual.values() for r in p["roi"].values()
                 for x in r["estimates"]) or any(t["observed_gate"] is False for t in temporal)
    output["status"]="failed_observed_gate" if failures else "insufficient_evidence" if missing else "passed_registered_gates"
    require(not args.output.exists(),"score output must be new")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save(args.output,output)
    print(json.dumps(dict(status=output["status"],missing=missing),ensure_ascii=False))


if __name__=="__main__":
    main()

