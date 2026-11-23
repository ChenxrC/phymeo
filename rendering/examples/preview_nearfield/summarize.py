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
    values = sorted(r["capture_ms"] for r in frames)
    def percentile(q):
        return round(values[ceil(len(values) * q) - 1], 3)
    telemetry = receipt["backend"]["device"]["resident_telemetry"]
    return dict(fps=round(1000/fmean(values), 3), mean_ms=round(fmean(values),3),
                p50_ms=percentile(.50), p95_ms=percentile(.95), p99_ms=percentile(.99),
                worst_ms=percentile(1), over_33_3_ms=sum(v>33.3 for v in values),
                open_ms=round(receipt["open_ms"], 2),
                device_owned_peak_bytes=telemetry["device_local_bytes_peak"],
                source=path.as_posix(), sha256=digest(path))


def export(root, dest, final):
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
    quality_file=root/"score-near-8192.json"
    quality=load(quality_file)
    result=dict(schema="phymeo-motion-preview-measurements-v1",
                device="NVIDIA GeForce RTX 4060 Laptop GPU", resolution=[1280,720],
                timing_scope="synchronous public capture until requested immutable host planes are available",
                speed_gates={"mean_fps_min":30,"p95_ms_max":33.3}, measurements=groups,
                matched_pose_quality=dict(source=quality_file.name,sha256=digest(quality_file),
                    scope=quality["space"],gates=quality["gates"],poses=quality["poses"],temporal=quality["temporal"]))
    thin_file=root/"thin3-near/summary.json"
    thin=load(thin_file)
    result["moving_thin_edge"]=dict(sha256=digest(thin_file),pose=15,
        fresh_and_moving_identical=thin["fresh_and_moving_rgb_identical_at_pose15"],
        two_1024_sample_references=thin["scores"]["15"]["reveal_right"])
    result["sealed_room_model"]=load(root/"near-model-exposures/summary.json")
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence",type=Path,required=True)
    p.add_argument("--result",type=Path,required=True)
    p.add_argument("--final-timing-directory",default="film-near")
    a=p.parse_args()
    export(a.evidence,a.result,a.final_timing_directory)
