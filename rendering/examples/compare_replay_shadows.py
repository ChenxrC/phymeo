"""Compare two public camera journals and the decoded lossless source frames."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image


def compare(first, second, output):
    runs = [json.loads((directory / "capture.json").read_text()) for directory in (first, second)]
    for field in ("frames", "size", "source_excerpt_sha256", "source_indices", "source_times_s",
                  "camera_setup", "scene_radiance_unit_scale", "lighting", "photographic_worktop"):
        if runs[0][field] != runs[1][field]:
            raise ValueError(f"incompatible camera journals: {field}")
    width, height = runs[0]["size"]
    rows = []
    stable_max = peak = different = total_error = 0
    worst = None
    for i in range(runs[0]["frames"]):
        images = []
        for directory, run in zip((first, second), runs, strict=True):
            with Image.open(directory / f"frames/{i:04d}.png") as im:
                if im.mode != "RGB" or im.size != (width, height):
                    raise ValueError(f"unexpected image layout in frame {i}")
                pixels = np.asarray(im)
                if sha256(pixels.tobytes()).hexdigest() != run["frame_sha256"][i]:
                    raise ValueError(f"image differs from journal in frame {i}")
                images.append(pixels)
        error = np.abs(images[0].astype(np.int16) - images[1])
        mx = error.max(axis=2)
        y, x = np.nonzero(mx)
        count = len(y)
        different += count
        total_error += int(error.sum())
        maximum = int(mx.max())
        if maximum > peak:
            peak = maximum
            pos = np.unravel_index(int(mx.argmax()), mx.shape)
            worst = [i, int(pos[1]), int(pos[0])]
        # A changed central sample belongs to a smooth image interior only
        # when BOTH images vary by <=2 RGB8 within a full 3x3 neighbourhood.
        interior = 0
        for py, px in zip(y, x, strict=True):
            if 0 < px < width-1 and 0 < py < height-1 and all(
                np.ptp(im[py-1:py+2, px-1:px+2], axis=(0,1)).max() <= 2 for im in images
            ):
                interior += 1
                stable_max = max(stable_max, int(mx[py, px]))
        rows.append(dict(frame=i, changed=count, smooth_interior_changed=interior,
                         max_rgb8=maximum, changed_over_one=int((mx>1).sum())))
    data = dict(source=runs[0]["source_excerpt_sha256"], resolution=[width,height],
                frames=len(rows), exact_frames=sum(row["changed"]==0 for row in rows),
                changed_pixel_exposures=different, pixel_exposures=len(rows)*width*height,
                mean_absolute_rgb8=total_error/(len(rows)*width*height*3),
                max_rgb8=peak, worst_frame_x_y=worst, smooth_interior_max_rgb8=stable_max,
                smooth_definition="Both full 3x3 RGB windows have per-channel span <=2",
                modes=[r["shadow_traversal"] for r in runs], per_frame=rows)
    with output.open("x",encoding="utf-8") as stream:
        json.dump(data,stream,indent=2)
    print({k:v for k,v in data.items() if k!="per_frame"})
    if stable_max>1:
        raise ValueError("smooth-interior error exceeds 1 RGB8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("first","second","output"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    compare(args.first,args.second,args.output)
