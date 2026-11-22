"""Make a two-view video source with one presentation timestamp per pose."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image


def compose(native, reference, output):
    sources = (native,reference)
    logs = [json.loads((s/"capture.json").read_text()) for s in sources]
    first,second = logs
    for field in ("source_excerpt_sha256","source_indices","source_times_s","frames","fps",
                  "camera_setup","lighting","photographic_worktop","scene_radiance_unit_scale"):
        if first[field]!=second[field]:
            raise ValueError(f"exposures disagree at {field}")
    for field,default in (("scanned_linear_roughness",False),("display_transform","clip")):
        if first.get(field,default)!=second.get(field,default):
            raise ValueError(f"materials or camera display differ at {field}")
    if first["mode"]!="native" or second["mode"]!="studio":
        raise ValueError("expected one native and one reference journal")
    if ("cameras" in first)!=("cameras" in second):
        raise ValueError("only one journal includes per-frame camera bindings")
    if "cameras" in first:
        for a,b in zip(first["cameras"],second["cameras"],strict=True):
            if any(a[k]!=b[k] for k in ("id","position_m","orientation_xyzw","clip_m")):
                raise ValueError("camera pose or clipping differs")
            for va,vb in zip(a["intrinsics"],b["intrinsics"],strict=True):
                if abs(va/first["size"][1]-vb/second["size"][1])>1e-12:
                    raise ValueError("normalized camera optics differ")
    width,height = second["size"]
    if any(log["size"][0]*height != log["size"][1]*width for log in logs):
        raise ValueError("camera aspect ratios differ")
    output.mkdir(parents=True,exist_ok=False)
    frames=output/"frames"
    frames.mkdir()
    hashes=[]
    for i in range(first["frames"]):
        sheet=Image.new("RGB",(width*2,height))
        for side,(source,log) in enumerate(zip(sources,logs,strict=True)):
            with Image.open(source/"frames"/f"{i:04d}.png") as view:
                if list(view.size)!=log["size"] or view.mode!="RGB" or sha256(view.tobytes()).hexdigest()!=log["frame_sha256"][i]:
                    raise ValueError(f"unbound frame {i} on side {side}")
                display=view.resize((width,height),Image.Resampling.LANCZOS) if view.size!=(width,height) else view
                sheet.paste(display,(side*width,0))
        hashes.append(sha256(sheet.tobytes()).hexdigest())
        sheet.save(frames/f"{i:04d}.png",compress_level=2)
    keys=("frames","fps","source_excerpt_sha256","source_indices","source_times_s","camera_setup")
    report={k:first[k] for k in keys}
    report.update(size=[width*2,height],frame_sha256=hashes,
                  source_journals=[sha256((s/"capture.json").read_bytes()).hexdigest() for s in sources],
                  original_sizes=[log["size"] for log in logs],
                  images=["native camera resized with Lanczos","reference camera in its original resolution"])
    (output/"capture.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("native","reference","output"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    compose(args.native,args.reference,args.output)
