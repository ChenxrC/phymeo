"""Compare the saved R0 device readbacks without running the renderers again."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def read_plane(folder, sample, channel, pixel_size):
    name = f"{sample['lighting']}-{channel}.raw"
    data = (folder / name).read_bytes()
    if (
        len(data) != 640 * 480 * pixel_size
        or hashlib.sha256(data).hexdigest() != sample["planes"][name]
    ):
        raise ValueError(f"Invalid saved plane: {folder / name}")
    return data


def aluminium_strips(rgb):
    # Two equal strips inside the exposed horizontal top of instance 11.
    # The overhead light reflects toward camera x=0; inclining its incoming
    # ray to +x puts the mirror point at x=-0.75 m on this z plane.
    # For a fixed camera this reverses the brightness slope on the left block.
    values = []
    for left in (174, 204):
        total = 0
        for y in range(232, 248):
            offset = (640 * y + left) * 3
            total += sum(rgb[offset : offset + 12 * 3])
        values.append(total / (16 * 12 * 3))
    return values


def compare(root):
    strips = {}
    raster = {}
    binding = {}
    for backend in ("native-raster", "sapien-raster", "sapien-path_trace"):
        folder = root / backend
        report = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        samples = {sample["lighting"]: sample for sample in report["samples"]}
        strips[backend] = {}
        for light in ("overhead", "inclined"):
            sample = samples[light]
            identity = (
                sample["snapshot_sha256"],
                sample["camera"],
                sample["light"],
                sample["time_s"],
            )
            if binding.setdefault(light, identity) != identity or sample["source_steps"] != 0:
                raise ValueError(f"Different scene or exposure: {backend}/{light}")
            rgb = read_plane(folder, sample, "srgb8_rgb", 3)
            strips[backend][light] = aluminium_strips(rgb)
        if backend.endswith("-raster"):
            raster[backend] = (
                read_plane(folder, samples["overhead"], "instance_id_u32", 4),
                read_plane(folder, samples["overhead"], "forward_depth_m_f32", 4),
            )

    (native_ids, native_z), (ref_ids, ref_z) = raster.values()
    changed = foreground = 0
    max_error = 0.0
    for (i,), (j,), (z,), (r,) in zip(
        struct.iter_unpack("<I", native_ids),
        struct.iter_unpack("<I", ref_ids),
        struct.iter_unpack("<f", native_z),
        struct.iter_unpack("<f", ref_z),
        strict=True,
    ):
        changed += i != j
        if i == j and i:
            if not (math.isfinite(z) and math.isfinite(r) and min(z, r) > 0):
                raise ValueError("Invalid foreground depth")
            foreground += 1
            max_error = max(max_error, abs(z - r))
        elif i == j and (z != math.inf or r != math.inf):
            raise ValueError("Invalid background depth")
    result = {
        "pixels": 640 * 480,
        "different_instance_id": changed,
        "matching_foreground": foreground,
        "maximum_depth_difference_m": max_error,
        "aluminium_srgb8_left_right": strips,
    }
    print(json.dumps(result, indent=2))
    for backend, pair in strips.items():
        left, right = pair["overhead"]
        tilted_left, tilted_right = pair["inclined"]
        if right <= left or tilted_left <= tilted_right:
            raise ValueError(f"Expected the reflected highlight gradient to reverse: {backend}")
    if changed or not foreground or max_error > 0.002:
        raise ValueError("The two raster sensor images disagree on the calibration table")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing the three saved R0 paths")
    compare(parser.parse_args().root)
