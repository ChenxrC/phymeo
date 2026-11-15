"""Fixed V05 finite-support pose samples; no state evolution or engine."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .finite_support_reference import build_finite_patch
from .friction_reference import integrate_friction

MASS = 0.5
GRAVITY = 9.81
KAPPA = 1e7
MU = 0.5
PRESSURE = MASS * GRAVITY / 0.04**2
DEPTH = PRESSURE / KAPPA
PHASE = "PRESCRIBED_POSE"
SUBDIVISIONS = (1, 4, 8)


def rectangle(cx, cy, hx, hy):
    return ((cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy))


def cases():
    fixed = []
    for n in SUBDIVISIONS:
        for j in range(31):
            fixed.append(
                dict(id=f"slide-n{n}-j{j:02d}", kind="slide", subdivisions=n, x_m=0.002 * j)
            )
    for n in SUBDIVISIONS:
        for kind, x in (("outside", 0.061), ("lift", 0.0), ("tilt", 0.0)):
            fixed.append(dict(id=f"{kind}-n{n}", kind=kind, subdivisions=n, x_m=x))
    return fixed


def geometry(case):
    x = case["x_m"]
    gap = (
        (0.02, 0.0, -0.0001)
        if case["kind"] == "tilt"
        else (0.0, 0.0, 0.0001 if case["kind"] == "lift" else -DEPTH)
    )
    com = (x, 0.0, 0.01 + gap[2])
    return dict(
        footprint_xy=rectangle(x, 0.0, 0.02, 0.02),
        support_xy=rectangle(0.0, 0.0, 0.04, 0.04),
        gap_coefficients=gap,
        com_world=com,
        velocity_a=(0.02, 0.0, 0.0)
        if case["kind"] in ("slide", "outside") and x > 0
        else (0.0, 0.0, 0.0),
        velocity_b=(0.0, 0.0, 0.0),
        body_a="prescribed-block",
        body_b="finite-support",
    )


def oracle(case, inputs):
    kind, x = case["kind"], case["x_m"]
    if kind == "tilt":
        overlap, area, fn, cop = 0.0016, 0.001, 2.5, (-0.035 / 3, 0.0, 0.0)
    else:
        width = 0.04 if x <= 0.02 else max(0.0, 0.06 - x)
        overlap = 0.04 * width
        area = 0.0 if kind == "lift" else overlap
        fn = PRESSURE * area
        cop = ((x if x <= 0.02 else (x + 0.02) / 2), 0.0, 0.0) if area > 0 else None
    fx = -MU * fn if inputs["velocity_a"][0] > 0 else 0.0
    normal = np.array((0.0, 0.0, fn))
    friction = np.array((fx, 0.0, 0.0))
    world_n = np.cross(cop, normal) if cop is not None else np.zeros(3)
    world_t = np.cross(cop, friction) if cop is not None else np.zeros(3)
    com_n = world_n - np.cross(inputs["com_world"], normal)
    com_t = world_t - np.cross(inputs["com_world"], friction)
    return dict(
        overlap_area_m2=overlap,
        active_area_m2=area,
        normal_force_n=normal.tolist(),
        friction_force_n=friction.tolist(),
        normal_world_moment_nm=world_n.tolist(),
        total_world_moment_nm=(world_n + world_t).tolist(),
        cop_world=cop,
        normal_com_moment_nm=com_n.tolist(),
        friction_com_moment_nm=com_t.tolist(),
        total_com_moment_nm=(com_n + com_t).tolist(),
        friction_power_w=fx * inputs["velocity_a"][0],
    )


def close(actual, expected):
    return bool(np.allclose(actual, expected, rtol=1e-8, atol=1e-10))


def evaluate(case):
    inputs = geometry(case)
    patch = build_finite_patch(
        inputs["footprint_xy"],
        inputs["support_xy"],
        gap_coefficients=inputs["gap_coefficients"],
        stiffness_pa_per_m=KAPPA,
        layer_depth_m=0.002,
        subdivisions=case["subdivisions"],
    )
    count = len(patch.quadrature.points_world)
    result = integrate_friction(
        patch.quadrature,
        patch.pressure_pa,
        velocity_a_world=np.tile(inputs["velocity_a"], (count, 1)),
        velocity_b_world=np.tile(inputs["velocity_b"], (count, 1)),
        law="coulomb",
        mu=MU,
        slip_speed_m_s=0.001,
        origin_world=(0.0, 0.0, 0.0),
    )
    expected = oracle(case, inputs)
    com = np.asarray(inputs["com_world"])
    com_normal = np.asarray(result.normal.first_moment_nm) - np.cross(
        com, result.normal.first_force_n
    )
    com_friction = np.asarray(result.first_friction_moment_nm) - np.cross(
        com, result.first_friction_force_n
    )
    com_total = np.asarray(result.first_total_moment_nm) - np.cross(com, result.first_total_force_n)
    values = dict(
        overlap_area_m2=patch.overlap_area_m2,
        active_area_m2=patch.active_area_m2,
        normal_force_n=result.normal.first_force_n,
        friction_force_n=result.first_friction_force_n,
        normal_world_moment_nm=result.normal.first_moment_nm,
        total_world_moment_nm=result.first_total_moment_nm,
        cop_world=result.normal.cop_world,
        normal_com_moment_nm=com_normal.tolist(),
        friction_com_moment_nm=com_friction.tolist(),
        total_com_moment_nm=com_total.tolist(),
        friction_power_w=result.friction_power_w,
    )
    errors = {}
    passed = True
    for key, wanted in expected.items():
        actual = values[key]
        if key == "cop_world":
            ok = (
                actual is None
                if wanted is None
                else actual is not None and np.max(np.abs(np.asarray(actual) - wanted)) <= 1e-9
            )
            errors[key] = (
                None
                if wanted is None or actual is None
                else float(np.max(np.abs(np.asarray(actual) - wanted)))
            )
        else:
            ok = close(actual, wanted)
            errors[key] = float(np.max(np.abs(np.asarray(actual) - wanted)))
        passed = passed and bool(ok)
    balance_force = float(
        np.max(np.abs(np.asarray(result.first_total_force_n) + result.second_total_force_n))
    )
    balance_moment = float(
        np.max(np.abs(np.asarray(result.first_total_moment_nm) + result.second_total_moment_nm))
    )
    errors.update(force_balance_n=balance_force, moment_balance_nm=balance_moment)
    passed = (
        passed
        and balance_force <= 1e-12
        and balance_moment <= 1e-12
        and result.friction_power_w <= 1e-12
    )
    if expected["active_area_m2"] == 0:
        passed = (
            passed
            and count == 0
            and not patch.active_polygon_xy
            and result.normal.cop_world is None
        )
        passed = (
            passed and not any(result.first_total_force_n) and not any(result.first_total_moment_nm)
        )
    return dict(
        case=case,
        phase=PHASE,
        inputs=inputs,
        patch=patch.to_dict(),
        loads=result.to_dict(),
        values=values,
        expected=expected,
        absolute_errors=errors,
        passed=bool(passed),
    )


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    fixed = cases()
    manifest = dict(
        phase=PHASE,
        qualification="NUMPY_FINITE_AFFINE_LAYER_REFERENCE",
        mass_kg=MASS,
        gravity_m_s2=GRAVITY,
        mu=MU,
        stiffness_pa_per_m=KAPPA,
        layer_depth_m=0.002,
        fixed_pressure_pa=PRESSURE,
        fixed_indent_m=DEPTH,
        cases=fixed,
        note="One centered calibration; unchanged indentation during prescribed displacement, no equilibrium or time integration.",
    )
    write_json(output / "manifest.json", manifest)
    records = []
    for case in fixed:
        record = evaluate(case)
        records.append(record)
        write_json(output / (case["id"] + ".json"), record)
    convergence = []
    for kind, j in [("slide", i) for i in range(31)] + [
        ("outside", None),
        ("lift", None),
        ("tilt", None),
    ]:
        group = [
            r
            for r in records
            if r["case"]["kind"] == kind and (j is None or r["case"]["x_m"] == 0.002 * j)
        ]
        mid, fine = group[-2:]
        diff = {
            key: float(np.max(np.abs(np.asarray(mid["values"][key]) - fine["values"][key])))
            for key in ("active_area_m2", "normal_force_n", "total_com_moment_nm")
        }
        ok = all(close(mid["values"][key], fine["values"][key]) for key in diff)
        convergence.append(dict(kind=kind, j=j, absolute_changes=diff, passed=ok))
    passed = all(r["passed"] for r in records) and all(c["passed"] for c in convergence)
    summary = dict(
        manifest=manifest,
        status="V05_REFERENCE_PASS" if passed else "V05_REFERENCE_FAIL",
        convergence=convergence,
        cases=[
            dict(
                case=r["case"],
                inputs=r["inputs"],
                values=r["values"],
                expected=r["expected"],
                absolute_errors=r["absolute_errors"],
                passed=r["passed"],
                points=len(r["patch"]["pressure_pa"]),
                data_file=r["case"]["id"] + ".json",
            )
            for r in records
        ],
    )
    write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "case",
                "x_m",
                "subdivisions",
                "overlap_area_m2",
                "active_area_m2",
                "Fn_N",
                "Fx_N",
                "COPx_m",
                "normal_My_COM_Nm",
                "total_My_COM_Nm",
                "passed",
            )
        )
        for row in summary["cases"]:
            v = row["values"]
            writer.writerow(
                (
                    row["case"]["id"],
                    row["case"]["x_m"],
                    row["case"]["subdivisions"],
                    v["overlap_area_m2"],
                    v["active_area_m2"],
                    v["normal_force_n"][2],
                    v["friction_force_n"][0],
                    None if v["cop_world"] is None else v["cop_world"][0],
                    v["normal_com_moment_nm"][1],
                    v["total_com_moment_nm"][1],
                    row["passed"],
                )
            )
    viewer = dict(
        summary,
        cases=[
            dict(
                s,
                overlap_polygon_xy=r["patch"]["overlap_polygon_xy"],
                active_polygon_xy=r["patch"]["active_polygon_xy"],
                vertices_world=r["patch"]["quadrature"]["vertices_world"],
                triangles=r["patch"]["quadrature"]["triangles"],
                pressure_pa=r["patch"]["pressure_pa"],
            )
            for s, r in zip(summary["cases"], records, strict=True)
        ],
    )
    template = Path(__file__).with_name("finite_support_viewer.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(
        template.replace(
            "__RESULTS_JSON__", json.dumps(viewer, allow_nan=False).replace("<", "\\u003c")
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            dict(
                status=summary["status"],
                cases=len(records),
                failed=[r["case"]["id"] for r in records if not r["passed"]],
                max_force_error_n=max(r["absolute_errors"]["normal_force_n"] for r in records),
                max_moment_error_nm=max(
                    r["absolute_errors"]["total_com_moment_nm"] for r in records
                ),
                max_area_error_m2=max(r["absolute_errors"]["active_area_m2"] for r in records),
                max_cop_error_m=max(r["absolute_errors"]["cop_world"] or 0 for r in records),
                convergence_pass=all(c["passed"] for c in convergence),
            ),
            indent=2,
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
