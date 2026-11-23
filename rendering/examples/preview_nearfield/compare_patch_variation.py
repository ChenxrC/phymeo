"""Compare broad luminance errors on registered material interiors in matched motion captures."""
import argparse
import json
from pathlib import Path

import numpy as np

from preview_compact_rigid_history import preview, preview_assess


def block_errors(image, reference, labels, identities):
    """Keep whole 8x8 tiles on one object, using the same grid in both exposures."""
    if image.shape != reference.shape or image.shape != (*labels.shape, 3):
        raise ValueError("image and geometry extent differ")
    height, width = labels.shape
    if height % 8 or width % 8:
        raise ValueError("this registered measurement uses complete 8x8 tiles")
    tile_shape = (height // 8, 8, width // 8, 8)
    tiled = labels.reshape(tile_shape)
    tile_id = tiled[:, 0, :, 0]
    eligible = np.isin(tile_id, identities) & (tiled.min(axis=(1, 3)) == tiled.max(axis=(1, 3)))
    if not eligible.any():
        raise ValueError("no complete interiors for this registered region")
    luminance_error = (image - reference) @ np.array([.2126, .7152, .0722])
    tile_error = luminance_error.reshape(tile_shape).mean(axis=(1, 3))[eligible]
    return dict(tiles=int(tile_error.size), rms=float(np.sqrt(np.mean(tile_error ** 2))),
                mean_bias=float(tile_error.mean()), spatial_std=float(tile_error.std()))


def exposure(directory, metrics):
    metadata = preview.read(directory / "summary.json")
    if metadata["status"] != "complete" or metadata["channels"] != "all":
        raise ValueError("a complete four-channel dataset is required")
    row, = (r for r in metadata["runs"] if r["name"] == "rigid-history")
    folder, data = preview_assess.load_row(directory, row)
    records = {r["pose_index"]: r for r in data["frames"]}
    if tuple(records) != preview.POSES:
        raise ValueError("incomplete motion dataset")
    frames = {}
    for key in preview.REGISTERED:
        r = records[key]
        picture = metrics.image(folder / r["png"], r["png_sha256"], r["rgb_sha256"])
        frames[key] = (r, picture, preview_assess.geometry(folder, r, np)["instance_id"])
    return frames


def main(args):
    metrics = preview_assess.load_metrics(args.delivery)
    pair = [exposure(d.resolve(), metrics) for d in (args.original, args.filtered)]
    result = {}
    seen_seeds = set()
    for directory in args.references:
        receipt = preview.read(directory / "summary.json")
        if (receipt["status"] != "complete" or receipt["batches"] != 4096
                or receipt["seed"] in seen_seeds):
            raise ValueError("requires distinct completed 65536-sample reference streams")
        seen_seeds.add(receipt["seed"])
        for row in receipt["records"]:
            pose = row["pose_index"]
            old, new = (f[pose] for f in pair)
            for record in (old[0], new[0]):
                preview_assess.check_identity(row, record)
                if row["sensors"] != record["sensors"]:
                    raise ValueError("reference and motion geometry differ")
            ref = metrics.image(directory / row["png"], row["png_sha256"], row["rgb_sha256"])
            for region in ("left-arm", "white-shell", "fingers", "rubber", "wrist-shell"):
                ids = metrics.ROI_IDS[region]
                result.setdefault(region, []).append(dict(
                    pose=pose, seed=receipt["seed"], reference_receipt_sha256=preview.digest(directory / "summary.json"),
                    original=block_errors(old[1], ref, old[2], ids),
                    filtered=block_errors(new[1], ref, new[2], ids)))
    if len(result["left-arm"]) != 6 or seen_seeds != {1, 2}:
        raise ValueError("incomplete registered comparisons")
    arms = result["left-arm"]
    verdict = dict(left_arm_rms_down_10pct_every_comparison=all(
        r["filtered"]["rms"] <= .9 * r["original"]["rms"] for r in arms),
        left_arm_absolute_bias_increase_at_most_005=all(
            abs(r["filtered"]["mean_bias"]) <= abs(r["original"]["mean_bias"]) + .005 for r in arms))
    output = dict(schema="phymeo-motion-8px-luminance-v1", units="inverse-sRGB display-linear",
                  scope="8x8 disjoint same-object interiors; compare with per-pixel and temporal metrics separately",
                  original_sha256=preview.digest(args.original / "summary.json"),
                  filtered_sha256=preview.digest(args.filtered / "summary.json"), verdict=verdict, regions=result)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "filtered", "delivery", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--references", nargs=2, type=Path, required=True)
    main(p.parse_args())
