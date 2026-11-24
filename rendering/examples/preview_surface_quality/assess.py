"""Compare captured Panda frame 60 against its registered white-shell reference.

All coordinates are top-left pixel coordinates. The penumbra polygons sit on
the two white faces below the wrist, clear of the molded lip and silhouette.
Metrics are in 8-bit display code values, measured before any board enlargement.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SIZE = (1280, 720)
BOX = (457, 110, 795, 355)
SHELL_SIZE = (BOX[2] - BOX[0], BOX[3] - BOX[1])
SHELL_SLICE = np.s_[BOX[1]:BOX[3], BOX[0]:BOX[2]]
# Coordinates below are relative to the white-shell crop.
SHADOW_POLYGONS = (
    ((138, 115), (214, 100), (216, 115), (140, 134)),
    ((142, 144), (210, 126), (218, 137), (149, 158)),
)
SHADOW_DETAIL = (123, 86, 235, 170)
CONTOUR_DETAIL = (129, 177, 225, 240)


def read_rgb(path, sizes):
    with Image.open(path) as img:
        if img.size not in sizes or img.mode != "RGB":
            raise ValueError(f"{path}: expected RGB image with size in {sizes}, got {img.mode} {img.size}")
        return np.array(img)


def read_sensors(directory):
    arrays, hashes = {}, {}
    for name, dtype in (("id", "<u4"), ("depth", "<f4"), ("valid", "u1")):
        data = (directory / f"frame-060-{name}.bin").read_bytes()
        arrays[name] = np.frombuffer(data, dtype).reshape(SIZE[::-1])
        hashes[name] = hashlib.sha256(data).hexdigest()
    ids, depth, valid = (arrays[key] for key in ("id", "depth", "valid"))
    if (not np.array_equal(valid, ids != 0) or not np.array_equal(valid, np.isfinite(depth))
            or not (depth > 0).all()):
        raise ValueError(f"{directory}: inconsistent center visibility, ID or forward depth")
    return ids, hashes


def edge_masks(ids):
    """Apply the shader's 3x3 candidate test; image borders have no wrapped neighbors."""
    h, w = ids.shape
    padded = np.pad(ids, 1, mode="edge")
    hit_miss = np.zeros_like(ids, dtype=bool)
    object_object = hit_miss.copy()
    for y in range(3):
        for x in range(3):
            neighbor = padded[y:y+h, x:x+w]
            hit_miss |= (ids == 0) != (neighbor == 0)
            object_object |= (ids != 0) & (neighbor != 0) & (ids != neighbor)
    return hit_miss, object_object


def polygon_mask(polygons):
    canvas = Image.new("1", SHELL_SIZE)
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        draw.polygon(polygon, fill=1)
    return np.array(canvas)


def luma(rgb):
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def masked_error(rgb, ref, mask):
    result = {"pixels": int(mask.sum())}
    if ref is not None and result["pixels"]:
        residual = rgb[mask].astype(float) - ref[mask]
        result["rgb_mae"] = float(np.abs(residual).mean())
        result["rgb_rmse"] = float(np.sqrt(np.mean(residual**2)))
    return result


def penumbra_measure(rgb, ref, mask):
    result = masked_error(rgb, ref, mask)
    # Vertical differences cross the sloping shadow bands on these faces.
    # Subtracting a 4-pixel local slope suppresses a smooth brightness ramp.
    support = mask[2:-2] & mask[1:-3] & mask[:-4] & mask[3:-1] & mask[4:]
    if not support.any():
        raise ValueError("penumbra mask has no five-pixel interior")

    def excess_slope(image):
        y = luma(image)
        return ((y[2:-2] - y[1:-3]) - (y[4:] - y[:-4]) / 4)[support]

    slopes = excess_slope(rgb)
    result.update(vertical_samples=len(slopes), local_slope_residual_p95=float(np.quantile(np.abs(slopes), .95)))
    if ref is not None:
        ref_slopes = excess_slope(ref)
        result["reference_local_slope_residual_p95"] = float(np.quantile(np.abs(ref_slopes), .95))
        result["local_slope_residual_mae_vs_reference"] = float(np.abs(slopes - ref_slopes).mean())
    return result


def make_board(path, images, title, detail, shadow):
    width, height = SHELL_SIZE
    gap, heading = 14, 54
    zoom = 3
    zw, zh = (detail[2] - detail[0])*zoom, (detail[3] - detail[1])*zoom
    board = Image.new("RGB", (3*width + 4*gap, heading + height + zh + 3*gap), "#18222b")
    draw = ImageDraw.Draw(board)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    draw.text((gap, 5), title, fill="white", font=font)
    labels = ("Baseline", "Shadow only", "Shadow + contour AA")
    for index, (rgb, label) in enumerate(zip(images, labels)):
        left = gap + index*(width + gap)
        draw.text((left, 31), label, fill="white", font=font)
        image = Image.fromarray(rgb)
        board.paste(image, (left, heading))
        x0, y0, x1, y1 = detail
        draw.rectangle((left+x0, heading+y0, left+x1-1, heading+y1-1), outline="#13d9f2")
        if shadow:
            for polygon in SHADOW_POLYGONS:
                draw.line([(left+x, heading+y) for x, y in (*polygon, polygon[0])], fill="#e47d22", width=1)
        board.paste(image.crop(detail).resize((zw, zh), Image.Resampling.NEAREST),
                    (left + (width-zw)//2, heading + height + gap))
    board.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "shadow", "combined", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Matched RGB PNG, full 1280x720 frame or 338x245 white-shell crop")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ref = None
    report = dict(frame=60, size=SIZE, white_shell_box=BOX, local_penumbra_polygons=SHADOW_POLYGONS,
                  local_lower_diagonal_detail=CONTOUR_DETAIL, variants={})
    if args.reference:
        ref = read_rgb(args.reference, [SHELL_SIZE, SIZE])
        if ref.shape[:2] == SIZE[::-1]:
            ref = ref[SHELL_SLICE]
        report["reference_sha256"] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    centers = {}
    for name in ("baseline", "shadow", "combined"):
        path = getattr(args, name)
        frames = [read_rgb(path / f"frame-{i:03}.png", [SIZE]) for i in (59, 60, 61)]
        centers[name] = frames[1]
        ids, hashes = read_sensors(path)
        if name == "baseline":
            baseline_ids, baseline_hashes = ids, hashes
        elif hashes != baseline_hashes:
            raise ValueError(f"{name}: frame-060 center sensors differ from baseline")
        # These fixed-coordinate deltas include motion and shading changes.
        # They are not motion-compensated temporal quality measurements.
        timing = path / "timings.json"
        json.loads(timing.read_text(encoding="utf-8-sig"))
        report["variants"][name] = dict(center_sha256=hashes,
            timings_sha256=hashlib.sha256(timing.read_bytes()).hexdigest(),
            successive_frame_changed_pixels=[int((a != b).any(axis=2).sum()) for a, b in zip(frames, frames[1:])])

    bg, obj = edge_masks(baseline_ids)
    candidate = bg | obj
    penumbra = polygon_mask(SHADOW_POLYGONS) & (baseline_ids[SHELL_SLICE] != 0) & ~candidate[SHELL_SLICE]
    # Third mask restricts all ID transitions to the isolated diagonal shell
    # against the studio. The studio may itself carry a nonzero instance ID.
    x0, y0, x1, y1 = CONTOUR_DETAIL
    diagonal = np.zeros(SHELL_SIZE[::-1], bool)
    diagonal[y0:y1, x0:x1] = True
    masks = {"hit_miss": bg[SHELL_SLICE], "distinct_hit_ids": obj[SHELL_SLICE],
             "lower_diagonal": diagonal & candidate[SHELL_SLICE]}
    if not masks["lower_diagonal"].any():
        raise ValueError("no visible ID contour in the registered lower diagonal")
    for name, rgb in centers.items():
        shell = rgb[SHELL_SLICE]
        variant = report["variants"][name]
        variant["penumbra"] = penumbra_measure(shell, ref, penumbra)
        variant["white_shell_id_contours"] = {k: masked_error(shell, ref, m) for k, m in masks.items()}

    changed = (centers["shadow"] != centers["combined"]).any(axis=2)
    report["frame60_contour_isolation"] = {
        label: {"pixels": int(mask.sum()), "changed_rgb_pixels": int((changed & mask).sum())}
        for label, mask in (("adjacent_id_transition", candidate), ("no_adjacent_id_transition", ~candidate),
                            ("visible_body_interior", ~candidate & (baseline_ids != 0)))}
    report["center_sensors_identical"] = True
    imgs = [rgb[SHELL_SLICE] for rgb in centers.values()]
    make_board(args.output / "shadow-comparison.png", imgs, "Frame 60   |   Penumbra samples in orange   |   Detail at 3x",
               SHADOW_DETAIL, True)
    make_board(args.output / "contour-comparison.png", imgs, "Frame 60   |   Lower shell contour   |   Detail at 3x",
               CONTOUR_DETAIL, False)
    (args.output / "assessment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if (changed & ~candidate).any():
        raise RuntimeError("shadow and combined differ away from center ID transitions; see assessment.json")


if __name__ == "__main__":
    main()
