"""Evaluate a spatial preview of one linear RGB exposure against a held-out sequence.

The guide is a separate direct-only render at the same view and illumination.
This bounded experiment measures filtering bias as well as noise reduction.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from phymeo_rendering.backends._sapien_neutral import encode_neutral


KERNEL = (1, 4, 6, 4, 1)
CROP = (384, 288, 896, 672)


def neighbors(shape, dy, dx):
    height, width = shape
    target = (slice(max(0, -dy), min(height, height - dy)),
              slice(max(0, -dx), min(width, width - dx)))
    source = (slice(max(0, dy), min(height, height + dy)),
              slice(max(0, dx), min(width, width + dx)))
    return target, source


def reconstruct(rgb, depth, ids, valid, guide):
    """Reconstruct only from nearby visible pixels with matching surface guides."""
    if (rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.shape != guide.shape
            or any(plane.shape != rgb.shape[:2] for plane in (depth, ids, valid))):
        raise ValueError("linear image and guides must share one camera extent")
    if (not np.issubdtype(rgb.dtype, np.floating) or guide.dtype != np.uint8
            or not np.isfinite(rgb).all()):
        raise ValueError("expected finite linear RGB and a uint8 direct-light guide")
    visible = valid.astype(bool) & (ids != 0)
    if np.any(~np.isfinite(depth[visible]) | (depth[visible] <= 0)):
        raise ValueError("visible forward depth must be finite and positive")

    color_sum = np.zeros_like(rgb, dtype=np.float64)
    weight_sum = np.zeros(visible.shape, dtype=np.float64)
    direct = guide.astype(np.int16)
    for y, wy in enumerate(KERNEL):
        for x, wx in enumerate(KERNEL):
            dy, dx = y - 2, x - 2
            if abs(dy) >= depth.shape[0] or abs(dx) >= depth.shape[1]:
                continue
            p, q = neighbors(depth.shape, dy, dx)
            allowed = visible[p] & visible[q] & (ids[p] == ids[q])
            difference = np.zeros_like(depth[p])
            np.subtract(depth[p], depth[q], out=difference, where=allowed)
            allowed &= np.abs(difference) <= .003 * np.minimum(depth[p], depth[q])
            allowed &= np.max(np.abs(direct[p] - direct[q]), axis=-1) <= 2
            weights = allowed.astype(np.float64) * (wy * wx)
            color_sum[p] += rgb[q] * weights[..., None]
            weight_sum[p] += weights

    result = rgb.copy()
    result[visible] = color_sum[visible] / weight_sum[visible, None]
    return result


def edge_band(ids, valid, guide):
    """Visible one-pixel boundary at a silhouette, ID edge or direct-light edge."""
    edge = np.zeros_like(valid, dtype=bool)
    guide = guide.astype(np.int16)
    for dy, dx in ((0, 1), (1, 0)):
        p, q = neighbors(ids.shape, dy, dx)
        change = ((ids[p] != ids[q]) | (valid[p] != valid[q])
                  | (np.max(np.abs(guide[p] - guide[q]), axis=-1) > 2))
        edge[p] |= change
        edge[q] |= change
    return edge & valid.astype(bool)


def measure(raw, preview, target, mask):
    a, b, ref = (value[mask].astype(np.float64) for value in (raw, preview, target))
    if not len(ref):
        raise ValueError("empty registered region")
    raw_rmse = float(np.sqrt(np.mean((a - ref) ** 2)))
    preview_rmse = float(np.sqrt(np.mean((b - ref) ** 2)))
    if raw_rmse <= 0:
        raise ValueError("cannot measure relative improvement on a zero-error baseline")
    return dict(pixels=len(ref), raw_rmse=raw_rmse, preview_rmse=preview_rmse,
                rmse_ratio=preview_rmse / raw_rmse, target_mean=ref.mean(axis=0).tolist(),
                max_mean_bias=float(np.max(np.abs((b - ref).mean(axis=0)))))


def run(first, reference, original_study, output):
    output.mkdir(parents=True, exist_ok=False)
    direct_file = original_study / "red" / "none.npz"
    with (np.load(first, allow_pickle=False) as low,
          np.load(reference, allow_pickle=False) as high,
          np.load(direct_file, allow_pickle=False) as direct):
        ids, depth, valid = (low[name] for name in ("INSTANCE_ID", "FORWARD_DEPTH", "VALID"))
        for name, plane in (("INSTANCE_ID", ids), ("FORWARD_DEPTH", depth), ("VALID", valid)):
            if plane.tobytes() != high[name].tobytes():
                raise ValueError(f"independent sequence geometry differs: {name}")
            if direct[name].tobytes() != plane.tobytes():
                raise ValueError(f"direct-light guide has different geometry: {name}")
        a, ref = low["RGBA32F"][..., :3], high["RGBA32F"][..., :3]
        public_first, public_reference = low["RGB"], high["RGB"]
    with np.load(original_study / "reference.npz", allow_pickle=False) as registration:
        fingers = registration["ids"]
    with Image.open(original_study / "red" / "none.png") as image:
        guide = np.asarray(image.convert("RGB"))
    if not np.isfinite(ref).all() or np.any((fingers != 0) & (ids != fingers)):
        raise ValueError("reference or registered receiver differs")

    start = perf_counter()
    reconstructed = reconstruct(a, depth, ids, valid, guide)
    seconds = perf_counter() - start
    encoded = np.frombuffer(encode_neutral(reconstructed, 1.), np.uint8).reshape(a.shape)
    cpu_first = np.frombuffer(encode_neutral(a, 1.), np.uint8).reshape(a.shape)
    max_display_error = int(np.max(np.abs(cpu_first.astype(np.int16) - public_first)))
    if max_display_error > 2:
        raise ValueError("CPU display does not match the public first exposure")
    # Copy unchanged display pixels on the background, including exact UNORM tie cases.
    background = valid == 0
    preview = encoded.copy()
    preview[background] = public_first[background]

    finger_results = {}
    for identity, count in ((156, 2058), (158, 2062)):
        metrics = measure(a, reconstructed, ref, fingers == identity)
        if metrics["pixels"] != count:
            raise ValueError("registered selection changed")
        metrics["bias_limit"] = max(.0025, .10 * max(metrics["target_mean"]))
        metrics["passed"] = metrics["rmse_ratio"] <= .70 and metrics["max_mean_bias"] <= metrics["bias_limit"]
        finger_results[str(identity)] = metrics
    edges = measure(a, reconstructed, ref, edge_band(ids, valid, guide))
    passed = (all(m["passed"] for m in finger_results.values()) and edges["rmse_ratio"] <= 1.05
              and np.array_equal(a[background], reconstructed[background]))
    width, height = CROP[2] - CROP[0], CROP[3] - CROP[1]
    sheet = Image.new("RGB", (3 * width + 4, height))
    for i, pixels in enumerate((public_first, preview, public_reference)):
        sheet.paste(Image.fromarray(pixels).crop(CROP), (i * (width + 2), 0))
    sheet.save(output / "first-preview-reference.png")
    Image.fromarray(preview).save(output / "preview.png")
    paths = [first, reference, original_study / "red" / "none.png", original_study / "reference.npz"]
    report = dict(passed=passed, receivers=finger_results, edges=edges, edge_ratio_limit=1.05,
                  max_display_error_codes=max_display_error, filter_cpu_wall_s=seconds,
                  source_sha256={str(p):sha256(p.read_bytes()).hexdigest() for p in paths},
                  scope="One-frame offline reconstruction with separate current direct-light guide. "
                        "No temporal history. Held-out 1024-spp target is a finite reference.",
                  kernel=list(KERNEL), depth_relative_limit=.003, guide_code_limit=2)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "reference", "original-study", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.first, args.reference, args.original_study, args.output) else 2)
