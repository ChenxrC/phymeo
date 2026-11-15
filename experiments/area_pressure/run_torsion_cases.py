"""Fixed V04 traction/torsion experiments; prescribed velocity, no dynamics."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .friction_reference import disk_quadrature, integrate_friction

PHASE = "PRESCRIBED_VELOCITY_NO_DYNAMICS"
RADIUS, PRESSURE, MU, SLIP_SPEED = 0.01, 2000.0, 0.5, 0.001
MESHES = ((2, 16), (4, 32), (8, 64))


def cases():
    result = []
    for layout in ("single", "double"):
        for law in ("coulomb", "regularized"):
            for omega in (1.0, -1.0):
                for rings, sectors in MESHES:
                    result.append(
                        dict(
                            id=f"{layout}-{law}-w{omega:g}-n{rings}",
                            layout=layout,
                            law=law,
                            omega_rad_s=omega,
                            rings=rings,
                            sectors=sectors,
                        )
                    )
    for layout in ("single", "double"):
        result.append(
            dict(
                id=f"{layout}-rest",
                layout=layout,
                law="regularized",
                omega_rad_s=0.0,
                rings=8,
                sectors=64,
            )
        )
    return result


def centers(case):
    return ((0.0, 0.0),) if case["layout"] == "single" else ((-0.015, 0.0), (0.015, 0.0))


def oracle(case):
    omega = case["omega_rad_s"]
    count = len(centers(case))
    fn = count * np.pi * RADIUS**2 * PRESSURE
    if omega == 0:
        return dict(
            normal_force_n=fn,
            friction_moment_z_nm=0.0,
            friction_power_w=0.0,
            method="zero prescribed slip",
        )
    if case["layout"] == "single" and case["law"] == "coulomb":
        torque = -np.sign(omega) * 2 / 3 * MU * fn * RADIUS
        method = "single disk closed form"
    else:
        roots, weights = np.polynomial.legendre.leggauss(64)
        r = (roots + 1) * RADIUS / 2
        radial_weights = weights * RADIUS / 2
        smooth = SLIP_SPEED if case["law"] == "regularized" else 0.0
        if case["layout"] == "single":
            torque = (
                -2
                * np.pi
                * MU
                * PRESSURE
                * omega
                * np.sum(radial_weights * r**3 / np.sqrt((omega * r) ** 2 + smooth**2))
            )
            method = "independent 64-node radial Gauss"
        else:
            theta = (np.arange(256) + 0.5) * 2 * np.pi / 256
            torque = 0.0
            for cx, cy in centers(case):
                x = cx + r[:, None] * np.cos(theta)
                y = cy + r[:, None] * np.sin(theta)
                rho2 = x * x + y * y
                lever = -MU * PRESSURE * omega * rho2 / np.sqrt(omega**2 * rho2 + smooth**2)
                torque += np.sum(lever * (r * radial_weights)[:, None]) * (2 * np.pi / 256)
            method = "independent per-disk 64 Gauss x 256 midpoint angles, common rotation axis"
    return dict(
        normal_force_n=float(fn),
        friction_moment_z_nm=float(torque),
        friction_power_w=float(omega * torque),
        method=method,
    )


def evaluate(case):
    quad = disk_quadrature(RADIUS, centers(case), rings=case["rings"], sectors=case["sectors"])
    points = np.asarray(quad.points_world)
    velocities = np.cross(np.array([0.0, 0.0, case["omega_rad_s"]]), points)
    p = np.full(len(points), PRESSURE)
    result = integrate_friction(
        quad,
        p,
        velocity_a_world=velocities,
        velocity_b_world=np.zeros_like(velocities),
        mu=MU,
        law=case["law"],
        slip_speed_m_s=SLIP_SPEED,
    ).to_dict()
    expected = oracle(case)
    torque = result["first_friction_moment_nm"][2]
    normal = result["normal"]["first_force_n"][2]
    traction = np.asarray(result["tangential_traction_pa"])
    norms = np.asarray(result["normal"]["quadrature"]["normals_world"])
    point_force = np.asarray(result["first_friction_force_per_point_n"])
    point_power = np.asarray(result["friction_power_per_point_w"])
    force_balance = float(
        np.linalg.norm(np.asarray(result["first_total_force_n"]) + result["second_total_force_n"])
    )
    moment_balance = float(
        np.linalg.norm(
            np.asarray(result["first_total_moment_nm"]) + result["second_total_moment_nm"]
        )
    )
    errors = dict(
        normal_relative=abs(normal / expected["normal_force_n"] - 1),
        torque_relative=None
        if expected["friction_moment_z_nm"] == 0
        else abs(torque / expected["friction_moment_z_nm"] - 1),
        force_balance_n=force_balance,
        moment_balance_nm=moment_balance,
        tangential_force_n=float(np.linalg.norm(result["first_friction_force_n"])),
        traction_normal_pa=float(np.max(np.abs(np.sum(traction * norms, axis=1)))),
        coulomb_excess_pa=float(max(0.0, np.max(np.linalg.norm(traction, axis=1) - MU * p))),
        positive_power_w=float(max(0.0, np.max(point_power), result["friction_power_w"])),
        power_moment_error_w=abs(result["friction_power_w"] - case["omega_rad_s"] * torque),
    )
    elementary = (
        force_balance <= 1e-12
        and moment_balance <= 1e-12
        and errors["tangential_force_n"] <= 1e-10
        and errors["traction_normal_pa"] <= 1e-10
        and errors["coulomb_excess_pa"] <= 1e-10
        and errors["positive_power_w"] <= 1e-10
        and errors["power_moment_error_w"] <= 1e-12
    )
    if case["omega_rad_s"] == 0:
        gate = "ZERO_SLIP_NOT_STICTION"
        passed = (
            elementary and torque == 0 and result["friction_power_w"] == 0 and not np.any(traction)
        )
        passed = passed and errors["normal_relative"] <= 0.05
    elif case["rings"] == 8:
        gate = "FINEST_5_PERCENT"
        passed = (
            elementary and errors["normal_relative"] <= 0.05 and errors["torque_relative"] <= 0.05
        )
    else:
        gate = "COARSE_OBSERVATION"
        passed = elementary
    patches = []
    for cx, cy in centers(case):
        selected = (
            np.ones(len(points), dtype=bool)
            if case["layout"] == "single"
            else (points[:, 0] < 0 if cx < 0 else points[:, 0] > 0)
        )
        force = point_force[selected].sum(axis=0)
        full = np.cross(points[selected], point_force[selected]).sum(axis=0)
        local = np.cross(points[selected] - [cx, cy, 0.0], point_force[selected]).sum(axis=0)
        orbital = np.cross([cx, cy, 0.0], force)
        patches.append(
            dict(
                center_world=[cx, cy, 0.0],
                friction_force_n=force.tolist(),
                common_origin_moment_nm=full.tolist(),
                own_center_moment_nm=local.tolist(),
                lever_arm_moment_nm=orbital.tolist(),
            )
        )
        passed = passed and bool(np.linalg.norm(full - local - orbital) <= 1e-12)
    return dict(
        case=case,
        expected=expected,
        result=result,
        errors=errors,
        patches=patches,
        gate=gate,
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
        radius_m=RADIUS,
        pressure_pa=PRESSURE,
        mu=MU,
        slip_speed_m_s=SLIP_SPEED,
        cases=fixed,
        note="Fixed pressure and point velocities; no automatic contact geometry or simulation.",
    )
    write_json(output / "manifest.json", manifest)
    records = []
    for case in fixed:
        record = evaluate(case)
        records.append(record)
        write_json(output / (case["id"] + ".json"), record)
    convergence = []
    reversal = []
    holdout = []
    for layout in ("single", "double"):
        for law in ("coulomb", "regularized"):
            groups = {}
            for omega in (1.0, -1.0):
                group = [
                    r
                    for r in records
                    if r["case"]["layout"] == layout
                    and r["case"]["law"] == law
                    and r["case"]["omega_rad_s"] == omega
                ]
                before, fine = group[-2:]
                a, b = (
                    before["result"]["first_friction_moment_nm"][2],
                    fine["result"]["first_friction_moment_nm"][2],
                )
                delta = None if b == 0 else abs(a / b - 1)
                convergence.append(
                    dict(
                        layout=layout,
                        law=law,
                        omega=omega,
                        relative_change=delta,
                        passed=delta is not None and delta <= 0.05,
                    )
                )
                groups[omega] = group
            maximum = 0.0
            for pos, neg in zip(groups[1.0], groups[-1.0], strict=True):
                maximum = max(
                    maximum,
                    float(
                        np.max(
                            np.abs(
                                np.asarray(pos["result"]["first_friction_force_per_point_n"])
                                + np.asarray(neg["result"]["first_friction_force_per_point_n"])
                            )
                        )
                    ),
                )
            reversal.append(
                dict(layout=layout, law=law, max_point_force_sum_n=maximum, passed=maximum <= 1e-12)
            )
    for law in ("coulomb", "regularized"):
        pair = [
            next(
                r
                for r in records
                if r["case"]["layout"] == layout
                and r["case"]["law"] == law
                and r["case"]["omega_rad_s"] == 1
                and r["case"]["rings"] == 8
            )
            for layout in ("single", "double")
        ]
        single, double = (abs(r["result"]["first_friction_moment_nm"][2]) for r in pair)
        holdout.append(
            dict(
                law=law,
                double_common_axis_torque_nm=double,
                two_single_spin_torque_nm=2 * single,
                passed=double > 2 * single,
            )
        )
    passed = all(r["passed"] for r in records) and all(
        r["passed"] for r in convergence + reversal + holdout
    )
    summary = dict(
        phase=PHASE,
        status="V04_REFERENCE_PASS" if passed else "V04_REFERENCE_FAIL",
        manifest=manifest,
        convergence=convergence,
        reversal=reversal,
        holdout=holdout,
        cases=[
            dict(
                case=r["case"],
                expected=r["expected"],
                errors=r["errors"],
                patches=r["patches"],
                gate=r["gate"],
                passed=r["passed"],
                normal_force_n=r["result"]["normal"]["first_force_n"],
                friction_force_n=r["result"]["first_friction_force_n"],
                friction_moment_nm=r["result"]["first_friction_moment_nm"],
                friction_power_w=r["result"]["friction_power_w"],
                points=len(r["result"]["friction_power_per_point_w"]),
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
                "Fn_N",
                "Mz_Nm",
                "oracle_Mz_Nm",
                "torque_relative_error",
                "power_W",
                "gate",
                "passed",
            )
        )
        for r in summary["cases"]:
            writer.writerow(
                (
                    r["case"]["id"],
                    r["normal_force_n"][2],
                    r["friction_moment_nm"][2],
                    r["expected"]["friction_moment_z_nm"],
                    r["errors"]["torque_relative"],
                    r["friction_power_w"],
                    r["gate"],
                    r["passed"],
                )
            )
    # UI gets the complete point traction/velocity arrays, but not duplicated report fields.
    view = dict(summary)
    view["cases"] = [
        dict(
            s,
            points_world=r["result"]["normal"]["quadrature"]["points_world"],
            traction_pa=r["result"]["tangential_traction_pa"],
            slip_velocity_m_s=r["result"]["tangential_velocity_world"],
            area_weights_m2=r["result"]["normal"]["quadrature"]["area_weights_m2"],
        )
        for s, r in zip(summary["cases"], records, strict=True)
    ]
    template = Path(__file__).with_name("torsion_viewer.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(
        template.replace(
            "__RESULTS_JSON__", json.dumps(view, allow_nan=False).replace("<", "\\u003c")
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            dict(
                status=summary["status"],
                cases=len(records),
                failed=[r["case"]["id"] for r in records if not r["passed"]],
                convergence=convergence,
                holdout=holdout,
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
