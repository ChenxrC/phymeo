"""Run fixed V00-V02 static reference cases; no physics engine or training."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.area_pressure.reference import rectangle_case, sphere_plane_case

MODEL = "declared-area-pressure-static-v1"


def cases():
    identity = np.eye(3)
    transforms = [
        ("identity", identity, np.zeros(3)),
        (
            "translated-rz90",
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            np.array([0.03, -0.02, 0.01]),
        ),
        (
            "tilted-rx90",
            np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
            np.array([-0.01, 0.02, 0.03]),
        ),
    ]
    for width, height in ((0.02, 0.02), (0.04, 0.02), (0.04, 0.04)):
        for n in (1, 2, 4):
            yield dict(
                id=f"V00-{width:g}x{height:g}-n{n}",
                family="V00",
                width_m=width,
                height_m=height,
                subdivisions=[n, n],
                pressure_pa=2000.0,
                pressure_gradient_pa_per_m=[0.0, 0.0],
                position_world=[0.0] * 3,
                rotation_world=identity.tolist(),
            )
    for n in (1, 2, 4):
        for name, rotation, position in transforms:
            yield dict(
                id=f"V01-{name}-n{n}",
                family="V01",
                width_m=0.02,
                height_m=0.02,
                subdivisions=[n, n],
                pressure_pa=2000.0,
                pressure_gradient_pa_per_m=[100000.0, 0.0],
                position_world=position.tolist(),
                rotation_world=rotation.tolist(),
            )
    for depth in (0.0, 0.0001, 0.0002, 0.0004):
        for resolution in (0.002, 0.001, 0.0005):
            yield dict(
                id=f"V02-d{depth:g}-h{resolution:g}",
                family="V02",
                radius_m=0.02,
                indentation_m=depth,
                modulus_pa=100000.0,
                resolution_m=resolution,
            )


def oracle(case):
    if case["family"] == "V02":
        radius, depth, modulus = (case[k] for k in ("radius_m", "indentation_m", "modulus_pa"))
        area = math.pi * (2 * radius * depth - depth * depth)
        force = math.pi * modulus * depth * depth * (1 - 2 * depth / (3 * radius))
        return dict(
            area_m2=area,
            first_force_n=[0, 0, force],
            first_moment_nm=[0, 0, 0],
            cop_world=None if depth == 0 else [0, 0, 0],
        )
    width, height, p0 = (case[k] for k in ("width_m", "height_m", "pressure_pa"))
    gx, gy = case["pressure_gradient_pa_per_m"]
    area = width * height
    force = area * p0
    local_cop = np.array([gx * width * width / (12 * p0), gy * height * height / (12 * p0), 0])
    rotation = np.array(case["rotation_world"], dtype=float)
    position = np.array(case["position_world"], dtype=float)
    cop = position + rotation @ local_cop
    world_force = rotation @ np.array([0, 0, force])
    return dict(
        area_m2=area,
        first_force_n=world_force.tolist(),
        first_moment_nm=np.cross(cop, world_force).tolist(),
        cop_world=cop.tolist(),
    )


def evaluate(case):
    kw = {k: v for k, v in case.items() if k not in ("id", "family")}
    result = sphere_plane_case(**kw) if case["family"] == "V02" else rectangle_case(**kw)
    expected = oracle(case)
    force_scale = float(np.linalg.norm(expected["first_force_n"]))
    force_error = float(np.linalg.norm(np.array(result.first_force_n) - expected["first_force_n"]))
    moment_error = float(
        np.linalg.norm(np.array(result.first_moment_nm) - expected["first_moment_nm"])
    )
    area_error = abs(result.area_m2 - expected["area_m2"])
    cop_error = (
        None
        if expected["cop_world"] is None
        else float(np.linalg.norm(np.array(result.cop_world) - expected["cop_world"]))
    )
    balance = float(np.linalg.norm(np.array(result.first_force_n) + result.second_force_n))
    couple = float(np.linalg.norm(np.array(result.first_moment_nm) + result.second_moment_nm))
    error = dict(
        area_relative=None if expected["area_m2"] == 0 else area_error / expected["area_m2"],
        force_relative=None if force_scale == 0 else force_error / force_scale,
        force_absolute_n=force_error,
        moment_absolute_nm=moment_error,
        cop_absolute_m=cop_error,
        action_reaction_n=balance,
        common_origin_couple_nm=couple,
    )
    if force_scale == 0:
        passed = (
            result.area_m2 == 0
            and result.pressure_load_n == 0
            and not result.quadrature.points_world
        )
        gate = "EMPTY_DOMAIN"
    elif case["family"] != "V02":
        passed = (
            max(error["area_relative"], error["force_relative"]) <= 1e-6
            and moment_error <= 1e-9
            and cop_error <= 1e-6
        )
        gate = "ANALYTIC_AREA_AND_MOMENT"
    elif case["resolution_m"] == 0.0005:
        passed = (
            max(error["area_relative"], error["force_relative"]) <= 0.05 and cop_error <= 0.0002
        )
        gate = "FINEST_DECLARED_FIELD"
    else:
        passed = True
        gate = "COARSE_OBSERVATION_NOT_FINEST_GATE"
    passed = bool(passed and balance <= 1e-12 and couple <= 1e-12)
    return dict(
        case=case, expected=expected, error=error, gate=gate, passed=passed, result=result.to_dict()
    )


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("A fresh output directory is required")
    output.mkdir(parents=True)
    manifest = dict(
        schema="area-pressure-cases-v1",
        model=MODEL,
        status="PRESCRIBED_STATIC_REFERENCE_NOT_DYNAMICS",
        engine="NumPy; no Drake/Newton/TYX execution",
        cases=list(cases()),
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    records = [evaluate(case) for case in manifest["cases"]]
    convergence = []
    for depth in (0.0001, 0.0002, 0.0004):
        pair = [
            r
            for r in records
            if r["case"]["family"] == "V02"
            and r["case"]["indentation_m"] == depth
            and r["case"]["resolution_m"] in (0.001, 0.0005)
        ]
        coarse, fine = pair
        area_change = (
            abs(coarse["result"]["area_m2"] - fine["result"]["area_m2"]) / fine["result"]["area_m2"]
        )
        load_change = (
            abs(coarse["result"]["pressure_load_n"] - fine["result"]["pressure_load_n"])
            / fine["result"]["pressure_load_n"]
        )
        convergence.append(
            dict(
                depth_m=depth,
                area_change=area_change,
                load_change=load_change,
                passed=max(area_change, load_change) <= 0.05,
            )
        )
    passed = all(r["passed"] for r in records) and all(r["passed"] for r in convergence)
    payload = dict(
        manifest=manifest,
        records=records,
        convergence=convergence,
        status="STATIC_REFERENCE_PASS" if passed else "STATIC_REFERENCE_FAIL",
        caveat="Manufactured/radial pressure-field experiment only; no free-body dynamics, Hydroelastic integration or training.",
    )
    data = json.dumps(payload, indent=2, allow_nan=False)
    (output / "results.json").write_text(data, encoding="utf-8")
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case",
                "area_m2",
                "load_n",
                "area_rel_error",
                "force_rel_error",
                "moment_abs_nm",
                "gate",
                "passed",
            ]
        )
        for row in records:
            writer.writerow(
                [
                    row["case"]["id"],
                    row["result"]["area_m2"],
                    row["result"]["pressure_load_n"],
                    row["error"]["area_relative"],
                    row["error"]["force_relative"],
                    row["error"]["moment_absolute_nm"],
                    row["gate"],
                    row["passed"],
                ]
            )
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    (output / "index.html").write_text(
        template.replace("__RESULTS_JSON__", encoded), encoding="utf-8"
    )
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()
    }
    (output / "sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(
        json.dumps(
            dict(
                status=payload["status"],
                cases=len(records),
                failed=[r["case"]["id"] for r in records if not r["passed"]],
                convergence=convergence,
                output=str(output),
            ),
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.output))


if __name__ == "__main__":
    main()
