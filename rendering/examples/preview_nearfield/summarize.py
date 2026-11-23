"""Extract portable, auditable numerical results from the registered experiment."""
import argparse
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
from statistics import fmean


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def timing(path):
    receipt = load(path)
    frames = receipt["frames"]
    if receipt["status"] != "complete" or [r["pose_index"] for r in frames] != list(range(300)):
        raise ValueError(f"incomplete measurement: {path}")
    chronological = [r["capture_ms"] for r in frames]
    values = sorted(chronological)
    def percentile(q):
        return round(values[ceil(len(values) * q) - 1], 3)
    telemetry = receipt["backend"]["device"]["resident_telemetry"]
    return dict(fps=round(1000/fmean(values), 3), mean_ms=round(fmean(values),3),
                p50_ms=percentile(.50), p95_ms=percentile(.95), p99_ms=percentile(.99),
                worst_ms=percentile(1), over_33_3_ms=sum(v>33.3 for v in values),
                open_ms=round(receipt["open_ms"], 2),
                device_owned_peak_bytes=telemetry["device_local_bytes_peak"],
                public_capture_ms=[round(t,4) for t in chronological],
                source=path.as_posix(), sha256=digest(path))


def export(root, dest, final, quality_run, publication_abba):
    if dest.exists():
        raise FileExistsError(dest)
    def measure(rel):
        item=timing(root/rel)
        item["source"]=str(Path(rel).as_posix())
        return item
    groups={}
    for preset in ("direct720","preview720","quality720"):
        groups[preset]=[measure(f"film-01/{preset}/rgb/receipt.json")]
    groups["local-soft-camera-sequence"]=[measure("film-near-studio/round-1/sparse/rgb/receipt.json")]
    for run in dict.fromkeys(["film-near",final]):
        groups[run]={ch:[measure(f"{run}/round-{r}/sparse/{ch}/receipt.json") for r in (1,2)] for ch in ("rgb","all")}
    original_model=load(root/"film-near/summary.json")
    final_model=load(root/final/"summary.json")
    def lighting_contract(model):
        # The newer generator records hit/miss diagnostic bits in the private
        # guide. Capture hashes below still cover every public output byte.
        return {k:v for k,v in model["contract"].items() if k!="guide_outcome_flags"}
    if lighting_contract(original_model) != lighting_contract(final_model):
        raise ValueError("different lighting models")
    compared=0
    for ch in ("rgb","all"):
        for round_id in (1,2):
            name=Path(f"round-{round_id}/sparse/{ch}/receipt.json")
            old=load(root/"film-near"/name)
            new=load(root/final/name)
            for left,right in zip(old["frames"],new["frames"],strict=True):
                if any(left[k]!=right[k] for k in ("frame_key","time_s","pose_index","state_sha256",
                                                   "camera_sha256","rgb_sha256","sensors","public_bytes")):
                    raise ValueError(f"capture changed at {name} pose {left['pose_index']}")
                compared+=1
    quality_file=root/quality_run
    quality=load(quality_file)
    result=dict(schema="phymeo-motion-preview-measurements-v1",
                device="NVIDIA GeForce RTX 4060 Laptop GPU", resolution=[1280,720],
                timing_scope="synchronous public capture until requested immutable host planes are available",
                speed_gates={"mean_fps_min":30,"p95_ms_max":33.3}, measurements=groups,
                matched_output=dict(comparisons=compared,original="film-near",final=final,
                                    original_guide_outcomes=original_model["contract"].get("guide_outcome_flags",False),
                                    final_guide_outcomes=final_model["contract"].get("guide_outcome_flags",False),
                                    includes="frame key, time, source state, camera, RGB and all requested sensor hashes"),
                matched_pose_quality=dict(source=quality_file.name,sha256=digest(quality_file),
                    scope=quality["space"],gates=quality["gates"],poses=quality["poses"],temporal=quality["temporal"]))
    thin_file=root/"thin3-near/summary.json"
    thin=load(thin_file)
    result["moving_thin_edge"]=dict(sha256=digest(thin_file),pose=15,
        fresh_and_moving_identical=thin["fresh_and_moving_rgb_identical_at_pose15"],
        two_1024_sample_references=thin["scores"]["15"]["reveal_right"])
    result["sealed_room_model"]=load(root/"near-model-exposures/summary.json")
    if publication_abba:
        result["publication_abba"]=[
            dict(transport=name,measurement=measure(f"owned-abba-{i}/round-1/sparse/all/receipt.json"))
            for i,name in enumerate(("packed","owned","owned","packed"))
        ]
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence",type=Path,required=True)
    p.add_argument("--result",type=Path,required=True)
    p.add_argument("--final-timing-directory",default="film-near")
    p.add_argument("--quality-run",default="score-near-8192.json")
    p.add_argument("--publication-abba",action="store_true")
    a=p.parse_args()
    export(a.evidence,a.result,a.final_timing_directory,a.quality_run,a.publication_abba)
