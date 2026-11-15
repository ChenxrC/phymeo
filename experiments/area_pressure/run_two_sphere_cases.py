"""Fixed, separately recorded V03 NumPy and Newton collision-only comparisons."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

MODEL = "two-radial-spheres-v1"


def fixed_cases():
    result = []
    for variant, ea, eb in (("soft-A", 1e5, 3e5), ("soft-B", 3e5, 1e5)):
        for h in (0.002, 0.001, 0.0005):
            result.append(
                dict(
                    id=f"{variant}-h{h:g}",
                    variant=variant,
                    modulus_a_pa=ea,
                    modulus_b_pa=eb,
                    overlap_m=0.0004,
                    resolution_m=h,
                )
            )
    result.append(
        dict(
            id="zero-overlap",
            variant="zero",
            modulus_a_pa=1e5,
            modulus_b_pa=3e5,
            overlap_m=0.0,
            resolution_m=0.0005,
        )
    )
    return result


def pressure_parameter_oracle(case):
    radius = 0.02
    depth = case["overlap_m"]
    if depth == 0:
        return dict(
            area_m2=0.0,
            projected_area_m2=0.0,
            axial_force_n=0.0,
            pressure_load_n=0.0,
            pressure_centroid_world=None,
            axis_height_m=None,
            axis_pressure_pa=None,
        )
    ea, eb = case["modulus_a_pa"], case["modulus_b_pa"]
    distance = 2 * radius - depth
    p0 = depth / (radius / ea + radius / eb)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    pressure, weights = (nodes + 1) * p0 / 2, weights * p0 / 2
    ra, rb = radius * (1 - pressure / ea), radius * (1 - pressure / eb)
    dra, drb = -radius / ea, -radius / eb
    z = (distance**2 + ra**2 - rb**2) / (2 * distance)
    dz = (ra * dra - rb * drb) / distance
    s = ra**2 - z**2
    ds = 2 * ra * dra - 2 * z * dz
    jac = np.sqrt(ds**2 + 4 * s * dz**2)
    area = np.pi * (weights @ jac)
    load = np.pi * (weights @ (pressure * jac))
    force = np.pi * (weights @ s)
    center_z = np.pi * (weights @ (z * pressure * jac)) / load
    return dict(
        area_m2=float(area),
        projected_area_m2=float(np.pi * (radius**2 - (distance / 2) ** 2)),
        axial_force_n=float(force),
        pressure_load_n=float(load),
        pressure_centroid_world=[0.0, 0.0, float(center_z)],
        axis_height_m=radius * (1 - p0 / ea),
        axis_pressure_pa=p0,
    )


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def fresh_directory(path):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    return path


def execute_reference(output):
    from experiments.area_pressure.two_sphere_reference import two_sphere_case

    output = fresh_directory(output)
    cases = fixed_cases()
    write_json(
        output / "manifest.json",
        dict(
            model=MODEL,
            phase="PRESCRIBED_STATIC_REFERENCE",
            radius_m=0.02,
            cases=cases,
            oracle="independent pressure parameterization, 64 Gauss nodes",
        ),
    )
    records = []
    for case in cases:
        result = two_sphere_case(
            modulus_a_pa=case["modulus_a_pa"],
            modulus_b_pa=case["modulus_b_pa"],
            overlap_m=case["overlap_m"],
            resolution_m=case["resolution_m"],
        )
        record = dict(case=case, reference=result.to_dict(), oracle=pressure_parameter_oracle(case))
        records.append(record)
        write_json(output / (case["id"] + ".json"), record)
    payload = dict(model=MODEL, records=records, phase="PRESCRIBED_STATIC_REFERENCE")
    write_json(output / "reference.json", payload)
    print(json.dumps(dict(status="REFERENCE_CAPTURED", cases=len(records), output=str(output))))
    return 0


def execute_newton(output):
    from experiments.area_pressure.newton_reference import newton_two_sphere_case

    output = fresh_directory(output)
    cases = fixed_cases()
    write_json(
        output / "manifest.json",
        dict(
            model=MODEL,
            phase="PRESCRIBED_POSE_COLLISION_ONLY",
            radius_m=0.02,
            cases=cases,
            device="cuda:0",
            reduction=False,
            texture_format="float32",
            margin=0.0,
            gap=0.0,
            mc_edge_clamp_min=0.0,
            rigid_contact_max=16384,
        ),
    )
    records = []
    for case in cases:
        print("START " + case["id"], flush=True)
        try:
            result = newton_two_sphere_case(
                modulus_a_pa=case["modulus_a_pa"],
                modulus_b_pa=case["modulus_b_pa"],
                overlap_m=case["overlap_m"],
                voxel_size_m=case["resolution_m"],
                device="cuda:0",
            )
        except Exception as exc:
            write_json(
                output / "failure.json",
                dict(
                    case=case,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    completed=[r["case"]["id"] for r in records],
                ),
            )
            raise
        record = dict(case=case, newton=result)
        records.append(record)
        write_json(output / (case["id"] + ".json"), record)
        print("CAPTURED " + case["id"], flush=True)
    write_json(
        output / "newton.json",
        dict(model=MODEL, records=records, phase="PRESCRIBED_POSE_COLLISION_ONLY"),
    )
    print(
        json.dumps(
            dict(status="NEWTON_CAPTURED_NOT_YET_COMPARED", cases=len(records), output=str(output))
        )
    )
    return 0


def errors(result, oracle):
    area = result["area_m2"]
    force = abs(result["first_force_n"][2])
    center = result["pressure_centroid_world"]
    if oracle["area_m2"] == 0:
        return dict(
            area_relative=None,
            force_relative=None,
            center_error_m=None,
            zero_domain=area == 0 and result["pressure_load_n"] == 0 and force == 0,
        )
    return dict(
        area_relative=abs(area / oracle["area_m2"] - 1),
        force_relative=abs(force / oracle["axial_force_n"] - 1),
        center_error_m=(
            None
            if center is None
            else float(np.linalg.norm(np.asarray(center) - oracle["pressure_centroid_world"]))
        ),
        zero_domain=None,
    )


def relative_change(a, b):
    return None if b == 0 else abs(a / b - 1)


def within(value, maximum):
    return value is not None and value <= maximum


def compare(reference_path, newton_path, output):
    output = fresh_directory(output)
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    newton = json.loads(Path(newton_path).read_text(encoding="utf-8"))
    cases = fixed_cases()
    if [r["case"] for r in reference["records"]] != cases or [
        r["case"] for r in newton["records"]
    ] != cases:
        raise ValueError("Saved membership differs from preregistration")
    records = []
    for ref, native in zip(reference["records"], newton["records"], strict=True):
        case, nr, oracle = ref["case"], native["newton"], ref["oracle"]
        re = errors(ref["reference"]["integral"], oracle)
        ne = errors(nr, oracle)
        force_balance = float(
            np.linalg.norm(np.asarray(nr["first_force_n"]) + nr["second_force_n"])
        )
        moment_balance = float(
            np.linalg.norm(np.asarray(nr["first_moment_nm"]) + nr["second_moment_nm"])
        )
        if case["overlap_m"] == 0:
            passes = re["zero_domain"] and ne["zero_domain"]
            gate = "ZERO_DOMAIN"
        elif case["resolution_m"] == 0.0005:
            passes = all(
                within(e["area_relative"], 0.05)
                and within(e["force_relative"], 0.05)
                and within(e["center_error_m"], 0.0002)
                for e in (re, ne)
            )
            gate = "FINEST_5_PERCENT"
        else:
            passes, gate = True, "COARSE_OBSERVATION"
        passes = bool(passes and force_balance <= 1e-10 and moment_balance <= 1e-10)
        records.append(
            dict(
                case=case,
                reference=ref["reference"],
                oracle=oracle,
                newton=nr,
                reference_error=re,
                newton_error=ne,
                force_balance_n=force_balance,
                moment_balance_nm=moment_balance,
                gate=gate,
                passed=passes,
            )
        )
    convergence = []
    for variant in ("soft-A", "soft-B"):
        pair = [
            r
            for r in records
            if r["case"]["variant"] == variant and r["case"]["resolution_m"] in (0.001, 0.0005)
        ]
        coarse, fine = pair
        for engine in ("reference", "newton"):
            a = coarse[engine]["integral"] if engine == "reference" else coarse[engine]
            b = fine[engine]["integral"] if engine == "reference" else fine[engine]
            area_change = relative_change(a["area_m2"], b["area_m2"])
            force_change = relative_change(a["first_force_n"][2], b["first_force_n"][2])
            convergence.append(
                dict(
                    variant=variant,
                    engine=engine,
                    area_change=area_change,
                    force_change=force_change,
                    passed=within(area_change, 0.05) and within(force_change, 0.05),
                )
            )
    a, b = records[2]["newton"], records[5]["newton"]
    exchange = dict(
        area_relative=relative_change(a["area_m2"], b["area_m2"]),
        force_relative=relative_change(a["first_force_n"][2], b["first_force_n"][2]),
        mirrored_center_error_m=(
            None
            if a["pressure_centroid_world"] is None or b["pressure_centroid_world"] is None
            else float(
                np.linalg.norm(
                    np.asarray(a["pressure_centroid_world"])
                    - np.asarray(
                        [
                            b["pressure_centroid_world"][0],
                            b["pressure_centroid_world"][1],
                            0.0396 - b["pressure_centroid_world"][2],
                        ]
                    )
                )
            )
        ),
    )
    exchange["passed"] = (
        within(exchange["area_relative"], 0.05)
        and within(exchange["force_relative"], 0.05)
        and within(exchange["mirrored_center_error_m"], 0.0002)
    )
    passed = (
        all(r["passed"] for r in records)
        and all(c["passed"] for c in convergence)
        and exchange["passed"]
    )
    payload = dict(
        model=MODEL,
        status="STATIC_CROSS_REFERENCE_PASS" if passed else "STATIC_CROSS_REFERENCE_FAIL",
        phase="PRESCRIBED_POSE_COLLISION_ONLY",
        cases=records,
        convergence=convergence,
        exchange=exchange,
        sources={
            str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (reference_path, newton_path)
        },
        caveat="Same continuous radial pressure law; different discretizations. No dynamics, training or Phymeo integration.",
    )
    write_json(output / "comparison.json", payload)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case",
                "area_m2",
                "axial_force_n",
                "area_relative_error",
                "force_relative_error",
                "gate",
                "passed",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r["case"]["id"],
                    r["newton"]["area_m2"],
                    abs(r["newton"]["first_force_n"][2]),
                    r["newton_error"]["area_relative"],
                    r["newton_error"]["force_relative"],
                    r["gate"],
                    r["passed"],
                ]
            )
    template = Path(__file__).with_name("two_sphere_viewer.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(
        template.replace(
            "__RESULTS_JSON__", json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            dict(
                status=payload["status"],
                cases=len(records),
                failed=[r["case"]["id"] for r in records if not r["passed"]],
                convergence=convergence,
                exchange=exchange,
            ),
            indent=2,
        )
    )
    return 0 if passed else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("reference", "newton", "compare"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--newton", type=Path)
    args = p.parse_args()
    if args.mode == "reference":
        code = execute_reference(args.output)
    elif args.mode == "newton":
        code = execute_newton(args.output)
    else:
        if args.reference is None or args.newton is None:
            p.error("compare requires --reference and --newton")
        code = compare(args.reference, args.newton, args.output)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
