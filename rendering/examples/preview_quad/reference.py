"""Capture only a shared same-model one-hop 1024-proposal reference, once per seed."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_overlay import digest,require,save,verify_overlay
from validate import run_owner,subgroup_capabilities


def main():
    import panda_render_performance as bench
    import panda_motion_preview as preview
    from phymeo_rendering.outputs import Channel
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library",type=Path,required=True)
    ap.add_argument("--base-shaders",type=Path,required=True)
    ap.add_argument("--overlay",type=Path,required=True,help="bind the complete same base SPV set; overlay is NOT rendered")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seed",type=int,choices=(1,2),default=1)
    ap.add_argument("--device-index",type=int,default=1)
    ap.add_argument("--validation",action="store_true")
    ap.add_argument("--save-linear",action="store_true")
    args=ap.parse_args()
    args.output=args.output.resolve()
    args.library=args.library.resolve(strict=True)
    args.base_shaders=args.base_shaders.resolve(strict=True)
    args.overlay=args.overlay.resolve(strict=True)
    args.poses=[59,60,149]
    args.width,args.height=1280,720
    args.filter="none"
    manifest=verify_overlay(args.overlay)
    require({p.name:digest(p) for p in args.base_shaders.glob("*.spv")}==manifest["base_shaders"],
            "baseline shader set changed")
    args.output.mkdir(parents=True,exist_ok=False)
    value=dict(schema="phymeo-preview-quad-validation-v1",role="reference-only",status="preparing",
        overlay=manifest,overlay_manifest_sha256=digest(args.overlay/"overlay.json"),
        library=dict(file=str(args.library),sha256=digest(args.library)),runs=[],
        capture_scope="one shared same-model reference seed, three poses x64batches; no quad/baseline motion recapture")
    save(args.output/"summary.json",value)
    try:
        value["subgroup"]=subgroup_capabilities(args.device_index)
        with TemporaryDirectory(prefix="phymeo-quad-reference-") as temp:
            scenes,frames,studio,trace=bench.bind_scene(Path(temp)/"geometry")
            inputs=preview.make_inputs(scenes[1],frames,preview.preset("preview720"),"detail")
            inputs={i:(f,replace(req,channels=tuple(Channel))) for i,(f,req) in inputs.items()}
            save(args.output/"inputs.json",dict(scene=asdict(scenes[1]),studio=asdict(studio),
                trace_sha256=digest(trace),poses=[dict(index=i,frame=asdict(inputs[i][0]),
                camera=asdict(inputs[i][1].camera)) for i in args.poses]))
            value["inputs_sha256"]=digest(args.output/"inputs.json")
            row,_=run_owner(args,scenes[1],studio,inputs,args.base_shaders,"reference",
                samples=16,seed=args.seed,batches=64,expected_frag=manifest["base_shaders"][
                "resident_rayquery_normal_reflection_indirect_progressive.frag.spv"],
                device_uuid=value["subgroup"]["device_uuid"])
            value["runs"].append(row)
            value["status"]="complete"
    except BaseException as error:
        value.update(status="failed",error=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        save(args.output/"summary.json",value)


if __name__=="__main__":
    main()

