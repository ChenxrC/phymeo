"""Discrete Newton 1.7 rigid/soft observations with native default and full-surface buffers."""

import argparse
import json
import warnings
from importlib.metadata import version
from math import dist
from pathlib import Path

import newton
import warp as wp
from baseline import _newton_model, _read_soft


def cases():
    for name, dims, rotation in (
        ("ellipsoid", [0.375, 0.25, 0.125], [0, 0, 0, 1]),
        ("cylinder", [0.25, 0.375], [0, 0, 0, 1]),
        ("cone", [0.25, 0.375], [1, 0, 0, 0]),
    ):
        for gap in (1 / 16, -1 / 32):
            yield (
                name,
                gap,
                [0, 0, -1],
                dict(
                    kind=name,
                    dimensions_m=dims,
                    orientation_xyzw=rotation,
                    center_m=[0.0625, 0.125, dims[-1] + gap],
                ),
            )
    for feature, point, n in (
        ("edge", [0, -1, 0], [0, -0.6, 0.8]),
        ("vertex", [0, 2, 0], [0, 0.6, 0.8]),
    ):
        for gap in (1 / 16, -1 / 32):
            yield (
                feature,
                gap,
                [-v for v in n],
                dict(
                    kind="ellipsoid",
                    dimensions_m=[0.25, 0.25, 0.25],
                    orientation_xyzw=[0, 0, 0, 1],
                    center_m=[x + v * (0.25 + gap) for x, v in zip(point, n, strict=True)],
                ),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or version("newton") != "1.7.0":
        parser.error("use a new output and pinned Newton")
    report = dict(
        newton=version("newton"),
        warp=version("warp-lang"),
        device=args.device,
        gap_tolerance_m=0.002,
        normal_tolerance=0.03,
        rows=[],
        differences=[],
    )
    for name, gap, normal, asset in cases():
        case = dict(
            family="curved_triangle",
            inputs=asset
            | dict(
                vertices_m=[[-2, -1, 0], [2, -1, 0], [0, 2, 0]],
                particle_radius_m=0.0,
                shape_margin_m=0.0,
            ),
        )
        model = _newton_model(case, newton, wp, args.device)
        state = model.state()
        assert str(state.particle_q.device) == args.device
        positions = state.particle_q.numpy().astype(float)
        options = dict(
            broad_phase="nxn", soft_contact_gap=0.0, rigid_contact_max=1, max_triangle_pairs=32
        )
        obs = dict(
            id=name,
            inputs=case["inputs"],
            ideal_surface_gap_m=gap,
            ideal_rigid_to_soft_normal=normal,
        )
        for mode in ("particle_default", "full_surface"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipeline = newton.CollisionPipeline(
                    model, enable_rigid_soft_full_surface_contact=mode == "full_surface", **options
                )
                rows = _read_soft(pipeline, state, positions, args.device)
            obs[mode] = rows
            obs[mode + "_warnings"] = [str(w.message) for w in caught]
            expected = gap < 0 and (mode == "full_surface" or name == "vertex")
            failure = bool(rows["records"]) != expected
            if expected and rows["records"]:
                deepest = min(rows["records"], key=lambda entry: entry["gap_m"])
                gerr = abs(deepest["gap_m"] - gap)
                nerr = dist(deepest["normal"], normal)
                obs[mode + "_errors"] = {"gap_m": gerr, "normal_l2": nerr}
                failure = (
                    failure or gerr > report["gap_tolerance_m"] or nerr > report["normal_tolerance"]
                )
            if failure:
                report["differences"].append([name, gap, mode])
        report["rows"].append(obs)
    with args.out.open("x", encoding="utf8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(
        json.dumps(
            dict(device=args.device, poses=len(report["rows"]), differences=report["differences"])
        )
    )
    return int(bool(report["differences"]))


if __name__ == "__main__":
    raise SystemExit(main())
