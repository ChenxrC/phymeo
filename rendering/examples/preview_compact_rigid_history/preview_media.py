"""Encode the captured rigid-history PNG sequence with the shared frozen encoder."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

from . import preview as p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result",type=Path,required=True)
    parser.add_argument("--delivery",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();root=args.result.resolve(strict=True)
    summary=p.read(root/"summary.json")
    p.require(summary["schema"]==p.SCHEMA and summary["status"]=="complete","complete measured rigid movie required")
    p.require(p.digest(root/"inputs.json")==summary["inputs_sha256"],"movie inputs changed")
    row=next(v for v in summary["runs"] if v["name"]=="rigid-history")
    p.require(p.digest(root/row["receipt"])==row["receipt_sha256"],"movie receipt changed")
    p.verify_chain(row["frames"])
    shared=p.load_delivery(args.delivery)
    policy=summary["contract"]
    name=p.media_profile_name(policy,summary)
    view=args.output.resolve().with_name(args.output.name+"-source-view")
    view.mkdir(parents=True,exist_ok=False);folder=view/name;(folder/"frames").mkdir(parents=True)
    for frame in row["frames"]:
        source=(root/row["receipt"]).parent/frame["png"]
        p.require(p.digest(source)==frame["png_sha256"],"published PNG changed")
        target=folder/frame["png"]
        try:os.link(source,target)
        except OSError:shutil.copyfile(source,target)
    borrowed=deepcopy({k:v for k,v in row.items() if k not in ("receipt","receipt_sha256")})
    borrowed["preset"]=name;p.save(folder/"receipt.json",borrowed)
    borrowed.update(receipt=str((folder/"receipt.json").relative_to(view)),receipt_sha256=p.digest(folder/"receipt.json"))
    shutil.copyfile(root/"inputs.json",view/"inputs.json")
    p.save(view/"summary.json",dict(schema=shared.base.SCHEMA,status="complete",quick=False,
        camera=summary.get("camera","detail"),studio_look=summary.get("studio_look","recorded"),
        registered_quality_configuration=summary.get("registered_quality_configuration",True),quality_claim=False,
        inputs_sha256=summary["inputs_sha256"],requested_presets=[name],runs=[borrowed],
        source_rigid_summary=str(root/"summary.json"),source_rigid_summary_sha256=p.digest(root/"summary.json"),
        trajectory_scope="all300 original recorded Panda poses, per-ID short history; no new physics",
        scope="media-only adapter over genuine measured receipt; neither a new capture nor a baseline result"))
    shared.base.encode(SimpleNamespace(result=view,media_output=args.output))


if __name__=="__main__":main()
