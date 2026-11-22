"""Compose two complete camera recordings, matching every source step before encoding."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size):
    for file in ("segoeui.ttf","DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(file,size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose(left, right, output):
    paths = (left,right)
    journals = [json.loads((p/"capture.json").read_text(encoding="utf-8")) for p in paths]
    a,b = journals
    for key in ("source_excerpt_sha256","source_indices","source_times_s","fps","frames","size"):
        if a[key]!=b[key]:
            raise ValueError(f"two camera sequences disagree at {key}")
    if [entry.get("camera_setup","fixed") for entry in journals] != ["fixed","fixed"]:
        raise ValueError("use the same fixed optics for the lighting comparison")
    if any(e.get("workshop",False) or e.get("photographic_worktop",False) for e in journals):
        raise ValueError("materials must match the archived neutral scene")
    output.mkdir(parents=True,exist_ok=False)
    images = output/"frames"
    images.mkdir()
    hashes = []
    label = font(22)
    clock = font(18)
    for i,t in enumerate(a["source_times_s"]):
        sheet = Image.new("RGB",(1536,484),(24,29,33))
        pen = ImageDraw.Draw(sheet)
        for j,(path,journal,name) in enumerate(zip(paths,journals,
                ["Directional light","Two area lights + environment"],strict=True)):
            with Image.open(path/"frames"/f"{i:04d}.png") as im:
                if sha256(im.tobytes()).hexdigest()!=journal["frame_sha256"][i]:
                    raise ValueError(f"capture checksum differs at {i}, view {j}")
                sheet.paste(im.resize((768,432),Image.Resampling.LANCZOS),(768*j,52))
            pen.text((768*j+15,12),name,font=label,fill=(241,242,238))
        pen.text((1390,15),f"t = {t:.3f} s",font=clock,fill=(195,206,208))
        hashes.append(sha256(sheet.tobytes()).hexdigest())
        sheet.save(images/f"{i:04d}.png",compress_level=2)
    report = dict(frames=a["frames"],size=[1536,484],fps=a["fps"],frame_sha256=hashes,
                  source_excerpt_sha256=a["source_excerpt_sha256"],source_indices=a["source_indices"],
                  source_times_s=a["source_times_s"],inputs=[sha256((p/"capture.json").read_bytes()).hexdigest() for p in paths])
    (output/"capture.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("key",type=Path)
    cli.add_argument("studio",type=Path)
    cli.add_argument("output",type=Path)
    args=cli.parse_args()
    compose(args.key,args.studio,args.output)
