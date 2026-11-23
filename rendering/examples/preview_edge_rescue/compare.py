"""Compare saved edge diagnostics only; no renderer, resampling, or image edits."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path


def read(path):return json.loads(path.read_text(encoding="utf-8-sig"))
def digest(path):return sha256(path.read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def model_key(value):
    policy=deepcopy(value["source"]["contract"])
    policy.pop("local_diffuse",None)
    return policy


def compare(paths):
    runs=[];reference=None;identities=None
    for directory in paths:
        directory=Path(directory).resolve(strict=True);row=read(directory/"summary.json")
        require(row["schema"]=="phymeo-edge-support-diagnostic-v1" and row["status"]=="actual_readbacks_bound", "incomplete edge diagnostic")
        require(digest(directory/"pixels.json")==row["pixels_sha256"] and digest(directory/"evidence.npz")==row["evidence_sha256"],"edge payload changed")
        pixels=read(directory/"pixels.json");keys=[(p["pixel"],p["instance_id"]) for p in pixels]
        binding=(row["source"]["fixture_sha256"],row["source"]["reference_summary_sha256"],row["frames"]["15"]["key"],row["frames"]["15"]["sensors"])
        if reference is None:reference=binding;identities=keys
        require(reference==binding and identities==keys and len(keys)==69,"registered source, same actual geometry or69-pixel identities differ")
        runs.append((directory,row,pixels))
    pairs=[]
    for i in range(len(runs)):
        for j in range(i+1,len(runs)):
            pa,a,aa=runs[i];pb,b,bb=runs[j]
            transitions={}
            for x,y in zip(aa,bb,strict=True):
                key=x["classification"]+" -> "+y["classification"]
                transitions[key]=transitions.get(key,0)+1
            pairs.append(dict(a=str(pa),b=str(pb),same_base_transport_contract=model_key(a)==model_key(b),
                local_a=a["source"]["contract"].get("local_diffuse"),local_b=b["source"]["contract"].get("local_diffuse"),
                support_class_transitions=transitions,identical_actual_rgb=a["frames"]["15"]["rgb_sha256"]==b["frames"]["15"]["rgb_sha256"],
                scope="same-geometry paired observation; source contracts disclose sampling/raster/model confounds; no automatically inferred cause"))
    return dict(schema="phymeo-edge-support-comparison-v1",status="saved_evidence_bound",pairs=pairs,
        runs=[dict(directory=str(p),summary_sha256=digest(p/"summary.json"),contract=r["source"]["contract"],coverage=r["coverage"],
            groups=r["groups"],reference_disagreement=r["reference_disagreement"],timing={k:v["capture_ms"] for k,v in r["frames"].items()},
            quality_claim=False) for p,r,_ in runs],
        interpretation="Signed bias distinguishes bright/dark; no-support differs from supported geometric sharing. Two1024spp references are finite. Candidate cost here is un-warmed diagnostic capture, not measured steady throughput.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories",nargs="+",type=Path);parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args();require(len(args.directories)>=2,"at least two saved candidates required")
    value=compare(args.directories)
    with args.output.open("x",encoding="utf-8") as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(args.output)


if __name__=="__main__":main()
