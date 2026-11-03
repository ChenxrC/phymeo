"""Read five bounded convex primitive pairs from the installed Newton 1.7 pipeline."""

import argparse
import json
from importlib.metadata import version
from math import dist, hypot
from pathlib import Path

import warp as wp
from baseline import _newton_samples


def cases():
    z, x = (0, 0, 1), (1, 0, 0)
    cylinder = {"kind": "cylinder", "radius_m": 0.5, "half_height_m": 0.375}
    cone = {"kind": "cone", "radius_m": 0.5, "half_height_m": 0.375}
    ell = {"kind": "ellipsoid", "radii_xyz_m": [0.5, 0.375, 0.25]}
    yield "two-cylinders", (cylinder, dict(cylinder, radius_m=0.25, half_height_m=0.25)), x, (0.75, 0, 0)
    yield "ellipsoid-sphere", (ell, {"kind": "sphere", "radius_m": 0.125}), z, (0, 0, 0.375)
    yield "cone-capsule", (cone, {"kind": "capsule", "radius_m": 0.125, "half_height_m": 0.125}), z, (0, 0, 0.625)
    yield "cone-cylinder", (cone, dict(cylinder, radius_m=0.75, half_height_m=0.25)), z, (0, 0, 0.625)
    sine, cosine = 24 / 25, 7 / 25
    reach = hypot(0.375 * sine, 0.125 * cosine)
    side = (0.375**2 - 0.125**2) * sine * cosine / reach
    box = {"kind": "box", "half_extents_xyz_m": [0.5, 0.5, 0.25]}
    tilted = dict(ell, radii_xyz_m=[0.375, 0.25, 0.125], local_quat_xyzw=[0, 0.6, 0, 0.8])
    yield "oblique-box", (box, tilted), z, (-side, 0.125, 0.25 + reach)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if version("newton") != "1.7.0" or args.out.exists():
        p.error("use the pinned reference and a new output path")
    wp.init()
    report = dict(
        newton=version("newton"), warp=version("warp-lang"), device=args.device,
        gap_tolerance_m=0.002, normal_error_tolerance=0.02,
        scope="Discrete states. Native anchor readbacks retained separately from physical signed gaps and normals.",
        records=[], differences=[],
    )
    for name, assets, normal, center in cases():
        positions = [[c + gap * axis for c, axis in zip(center, normal, strict=True)]
                     for gap in (1 / 16, -1 / 32)]
        inputs = dict(a_path_m=[[0, 0, 0], [0, 0, 0]], b_path_m=positions,
                      duration_s=1, sample_fractions=[0, 1])
        result = _newton_samples(inputs, assets, args.device, keep_all=True)
        start, end = result["samples"]
        observed = dict(case=name, inputs=inputs, assets=assets, actual=result)
        if start["contact"] or not end["contact"]:
            report["differences"].append([name, "contact membership"])
        else:
            deepest = min(end["contacts"], key=lambda row: row["gap_m"])
            observed["observed_gap_m"] = deepest["gap_m"]
            observed["gap_error_m"] = abs(deepest["gap_m"] + 1 / 32)
            observed["normal_error"] = dist(deepest["normal_a_to_b"], normal)
            if observed["gap_error_m"] > report["gap_tolerance_m"] or observed["normal_error"] > report["normal_error_tolerance"]:
                report["differences"].append([name, "deepest signed gap or normal"])
        report["records"].append(observed)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, allow_nan=False, indent=2)
    print(json.dumps({"cases": len(report["records"]), "differences": report["differences"]}))
    return int(bool(report["differences"]))


if __name__ == "__main__":
    raise SystemExit(main())
