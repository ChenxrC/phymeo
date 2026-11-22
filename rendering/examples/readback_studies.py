"""Measure material, display and tile changes in registered Panda camera readbacks."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read(directory):
    journal = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
    with Image.open(directory / "frames/0000.png") as image:
        data = np.asarray(image).copy()
    if journal["frames"] != 1 or sha256(data.tobytes()).hexdigest() != journal["frame_sha256"][0]:
        raise ValueError(f"readback differs from capture journal: {directory}")
    return journal, data


def compare(directories, tile):
    loaded = [read(directory) for directory in directories]
    journals, frames = zip(*loaded, strict=True)
    j = journals[0]
    common = ("source_excerpt_sha256", "source_indices", "source_times_s", "camera_setup", "size", "lighting")
    if any(other[key] != j[key] for other in journals for key in common):
        raise ValueError("camera series do not share source, time, size and lighting")
    if tile and journals[0]["cameras"] != journals[1]["cameras"]:
        raise ValueError("tiled camera calibration differs from full raster")
    w, h = j["size"]
    rois = {"whole": (0, 0, w, h)}
    if tile:
        if (w, h) != (1920,1080) or [v.get("tiles_per_axis", 1) for v in journals] != [1, 2, 1]:
            raise ValueError("unexpected HD tile study inputs")
        if [v["backend"]["samples_per_pixel"] for v in journals] != [512, 512, 1024]:
            raise ValueError("unexpected HD sample counts")
        for offset in (0, -16, 16):
            rois[f"vertical_{offset}"] = (w//2 + offset - 2, 0, w//2 + offset + 2, h)
            rois[f"horizontal_{offset}"] = (0, h//2 + offset - 2, w, h//2 + offset + 2)
        pairs = ((0,1,"full512_to_tiled512"), (0,2,"full512_to_full1024"), (1,2,"tiled512_to_full1024"))
    else:
        if (w,h) != (960,540) or [v.get("scanned_linear_roughness",False) for v in journals] != [False,True,True]:
            raise ValueError("unexpected roughness study inputs")
        if [v.get("display_transform","clip") for v in journals] != ["clip","clip","pbr-neutral"]:
            raise ValueError("unexpected display transforms")
        rois.update(shell=(410,70,490,90), wood=(690,465,820,500))
        pairs = ((0,1,"scanned_roughness"), (1,2,"neutral_display"))
    stats = {}
    for a,b,label in pairs:
        stats[label] = {}
        for name,(x0,y0,x1,y1) in rois.items():
            first, second = (f[y0:y1, x0:x1].astype(np.float64) for f in (frames[a],frames[b]))
            delta = second - first
            stats[label][name] = dict(pixels=len(delta.reshape(-1,3)), rmse_rgb8=float(np.sqrt(np.mean(delta**2))),
                    signed_mean_rgb8=np.mean(delta, axis=(0,1)).tolist(),
                    fully_white=[int(np.all(f==255,axis=2).sum()) for f in (first,second)])
    return dict(registered_source={k:j[k] for k in common},rois=rois,metrics=stats,
                exposures=[dict(image=v["frame_sha256"][0],ms=v["capture_ms"][0],
                    samples=v["backend"]["samples_per_pixel"],tiles=v.get("tiles_per_axis",1)) for v in journals])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",choices=("tiles","materials"))
    parser.add_argument("output",type=Path)
    parser.add_argument("sources",type=Path,nargs=3)
    args=parser.parse_args()
    result=compare(args.sources, args.study=="tiles")
    with args.output.open("x",encoding="utf-8") as target:
        json.dump(result,target,indent=2,allow_nan=False)
