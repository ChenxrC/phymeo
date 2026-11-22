"""Present eight verified GPU front cameras with a fixed crop per world."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size):
    for name in ("segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name,size)
        except OSError:
            pass
    return ImageFont.load_default()


def run(source, destination):
    report = json.loads((source/"summary.json").read_text(encoding="utf-8"))
    if report["scales"]["8"]["failed_views"] or report["scales"]["8"]["timing"]["completed_batches"] != 300:
        raise ValueError("expected the completed eight-world run")
    journal = (source/"batches.jsonl").read_bytes()
    digest = sha256(journal).hexdigest()
    if digest != report["journal"]["sha256"]:
        raise ValueError("R4 journal checksum mismatch")
    fronts = {}
    for line in journal.splitlines():
        row = json.loads(line)
        if row["phase"] == "batch" and row["worlds"] == 8:
            for v in row["views"]:
                if v["identity"]["camera"]["id"] != "front":
                    continue
                if not v["comparison"]["passed"] or not v["comparison"]["rgb_exact"]:
                    raise ValueError("expected matching independently replayed RGB")
                fronts[v["world"],row["batch"]] = v
    if len(fronts) != 2400:
        raise ValueError("incomplete front camera set")
    width,height = 320,240
    def read(w,i):
        meta = fronts[w,i]
        raw = zlib.decompress((source/"references"/str(w)/f"{i:04d}-0.zlib").read_bytes())
        count = width*height
        if len(raw)!=count*12:
            raise ValueError("unexpected plane layout")
        rgb,ids = raw[:3*count],raw[7*count:11*count]
        for key,plane in [("srgb8_rgb",rgb),("instance_id_u32",ids)]:
            if sha256(plane).hexdigest() != meta["plane_sha256"][key]:
                raise ValueError("reference bytes differ from batched camera")
        return rgb,np.frombuffer(ids,dtype="<u4").reshape(height,width)

    destination.mkdir(parents=True,exist_ok=False)
    output = destination/"frames"
    output.mkdir()
    boxes = []
    for w,assets in enumerate(report["source"]["assets"]):
        subjects = [geom["id"] for geom in assets if geom["kind"] != "plane"]
        bounds = [width,height,0,0]
        for i in range(300):
            _,ids = read(w,i)
            y,x = np.nonzero(np.isin(ids,subjects))
            if len(x):
                bounds = [min(bounds[0],int(x.min())), min(bounds[1],int(y.min())),
                          max(bounds[2],int(x.max())+1),max(bounds[3],int(y.max())+1)]
        x0,y0,x1,y1 = bounds
        if x1 <= x0 or y1 <= y0:
            raise ValueError("world has no visible subject")
        crop_width = min(width,max(86,(x1-x0)*1.18,(y1-y0)*1.18*1.5))
        crop_height = crop_width/1.5
        left = round(np.clip((x0+x1-crop_width)/2,0,width-crop_width))
        top = round(np.clip((y0+y1-crop_height)/2,0,height-crop_height))
        boxes.append((left,top,round(left+crop_width),round(top+crop_height)))
    hashes = []
    title,small = font(22),font(15)
    for i in range(300):
        canvas = Image.new("RGB",(1280,560),(23,27,30))
        draw = ImageDraw.Draw(canvas)
        for w,world in enumerate(report["source"]["worlds"]):
            x,y = w%4*320, w//4*280
            rgb,_ = read(w,i)
            view = Image.frombytes("RGB",(width,height),rgb).crop(boxes[w]).resize((316,210),Image.Resampling.BILINEAR)
            canvas.paste(view,(x+2,y+38))
            draw.text((x+8,y+4),world["key"],font=title,fill=(239,241,234))
            meta=fronts[w,i]
            epoch=meta["epoch"]
            clock=meta["identity"]["time_s"]
            draw.text((x+8,y+253),f"W{w}   {clock:.2f} s   |   epoch {epoch}",font=small,
                      fill=(253,193,91) if epoch else (181,193,196))
        hashes.append(sha256(canvas.tobytes()).hexdigest())
        canvas.save(output/f"{i:04d}.png",compress_level=2)
    receipt = dict(frames=300,size=[1280,560],fps=30,frame_sha256=hashes,source_excerpt_sha256=digest,
                   crop_boxes=boxes, verified_batched_front_exposures=2400,
                   source="R4 archived simulation poses; front view only; fixed crop per world; no new simulation")
    (destination/"capture.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print("Prepared 300 mosaics from 2400 matching front-camera exposures.")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source",type=Path)
    cli.add_argument("out",type=Path)
    args=cli.parse_args()
    run(args.source,args.out)
