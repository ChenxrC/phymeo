"""Frozen V05 dynamic support/push/edge windows, using planar PFC reference."""

import argparse
import csv
import gzip
import json
import math
from dataclasses import asdict
from pathlib import Path

from experiments.area_pressure.planar_reference import (
    PlanarConfig,
    PlanarState,
    sample_contact,
    step_body,
)


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
    )


def kinetic(c, s):
    return 0.5 * c.mass_kg * (s.vx_m_s**2 + s.vz_m_s**2) + 0.5 * c.inertia_kg_m2 * s.omega_rad_s**2


def energy(c, s, sample):
    return kinetic(c, s) + sample.potential_j + c.mass_kg * c.gravity_m_s2 * s.z_m


def inspect(c, r, h):
    e = []
    reaction = []
    v0 = (r.entry.vx_m_s, r.entry.vz_m_s, r.entry.omega_rad_s)
    masses = (c.mass_kg, c.mass_kg, c.inertia_kg_m2)
    for k in (r.first, r.second):
        e.extend(
            abs(m * (b - a) - ic - ie)
            for m, a, b, ic, ie in zip(
                masses, k.velocity_before, k.velocity_after, k.contact_impulse, k.external_impulse
            )
        )
        e.extend(
            abs(w - (a + b) / 2)
            for w, a, b in zip(k.mid_velocity, k.velocity_before, k.velocity_after)
        )
        e.extend(
            abs(j - k.duration_s * f)
            for j, f in zip(k.contact_impulse, k.contact.generalized_force)
        )
        e.extend(
            abs(j - k.duration_s * f)
            for j, f in zip(k.external_impulse, k.external_generalized_force)
        )
        fx, fz, my = k.contact.generalized_force
        reaction.extend(
            (
                abs(fx + k.contact.support_force_xz[0]),
                abs(fz + k.contact.support_force_xz[1]),
                abs(
                    my
                    + k.sample.state.z_m * fx
                    - k.sample.state.x_m * fz
                    + k.contact.support_moment_origin_nm
                ),
            )
        )
        e.extend(
            (
                abs(k.dissipated_j - k.duration_s * k.contact.dissipation_power_w),
                abs(
                    k.external_work_j
                    - k.duration_s
                    * sum(f * v for f, v in zip(k.external_generalized_force, k.mid_velocity))
                ),
            )
        )
        if k.dissipated_j < -1e-10:
            return False, abs(k.dissipated_j)
    e.extend(abs(a - b) for a, b in zip(r.first.velocity_before, v0))
    e.extend(abs(a - b) for a, b in zip(r.second.velocity_before, r.first.velocity_after))
    e.extend(
        abs(j - a - b)
        for j, a, b in zip(r.contact_impulse, r.first.contact_impulse, r.second.contact_impulse)
    )
    e.extend(abs(f * h - j) for f, j in zip(r.average_contact_generalized_force, r.contact_impulse))
    drift = [
        r.exit.x_m - r.entry.x_m - h * r.first.velocity_after[0],
        r.exit.z_m - r.entry.z_m - h * r.first.velocity_after[1],
        r.exit.angle_rad - r.entry.angle_rad - h * r.first.velocity_after[2],
    ]
    e.extend(abs(v) for v in drift)
    e.extend(
        abs(a - b)
        for a, b in zip((r.exit.vx_m_s, r.exit.vz_m_s, r.exit.omega_rad_s), r.second.velocity_after)
    )
    e.extend(
        (
            abs(r.exit.time_s - r.entry.time_s - h),
            abs(r.dissipated_j - r.first.dissipated_j - r.second.dissipated_j),
        )
    )
    phase_ok = r.first.sample.state == r.entry and r.second.sample.state == r.drifted
    phase_ok &= r.drifted.time_s == r.exit.time_s and (
        r.drifted.x_m,
        r.drifted.z_m,
        r.drifted.angle_rad,
    ) == (r.exit.x_m, r.exit.z_m, r.exit.angle_rad)
    phase_ok &= (
        r.drifted.vx_m_s,
        r.drifted.vz_m_s,
        r.drifted.omega_rad_s,
    ) == r.first.velocity_after
    phase_ok &= r.first.duration_s == r.second.duration_s == h / 2
    mean_error = max(
        abs(f - j / h) for f, j in zip(r.average_contact_generalized_force, r.contact_impulse)
    )
    return phase_ok and max(e, default=0) <= 1e-9 and max(
        reaction, default=0
    ) <= 1e-10 and mean_error <= 1e-12 and max(map(abs, drift)) <= 1e-12, max(
        e + reaction + [mean_error], default=0
    )


def frame(s, sample, contact, force, mid_velocity=None):
    return dict(
        state=asdict(s),
        body=sample.body_world_xz,
        points=sample.points_xz,
        normals=sample.normals_xz,
        pressure=(contact.normal_pressure_pa if contact else sample.pressure_pa),
        force=force,
        area=sample.area_m2,
        geometry_phase="accepted position",
        pressure_phase="second kick midpoint" if contact else "initial elastic",
        pressure_mid_velocity=mid_velocity,
        force_phase="whole step average" if contact else "initial elastic",
    )


def run_case(spec, out):
    directory = out / spec["id"]
    directory.mkdir()
    c = PlanarConfig()
    h = spec["dt_s"]
    s = PlanarState(0, 0, 0.01 - 0.0003065625, 0, 0, 0, 0)
    initial = s
    initial_sample = sample_contact(c, s)
    e0 = energy(c, s, initial_sample)
    write_json(directory / "input.json", dict(spec=spec, config=asdict(c), state=asdict(s)))
    frames = [frame(s, initial_sample, None, initial_sample.elastic_generalized_force)]
    rows = []
    events = {}
    peak = -1.0
    depth = -1.0
    work = 0.0
    dissipated = 0.0
    normal_impulse = 0.0
    max_error = 0.0
    max_ledger = 0.0
    checks = True
    last_contact = None
    last_force_positive = None
    side_seen = False
    fail = None
    with gzip.open(directory / "steps.jsonl.gz", "wt", encoding="utf-8") as stream:
        for i in range(spec["steps"]):
            fx = 4.0 if 0.2 <= (i + 0.5) * h < 0.4 else 0.0
            try:
                r = step_body(c, s, h, external_force_xz_n=(fx, 0))
            except Exception as exc:
                fail = dict(
                    step=i,
                    entry=asdict(s),
                    external_force_xz_n=[fx, 0],
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                write_json(directory / "failure.json", fail)
                break
            ok, err = inspect(c, r, h)
            checks &= ok
            max_error = max(max_error, err)
            push_work = sum(k.duration_s * fx * k.mid_velocity[0] for k in (r.first, r.second))
            work += push_work
            dissipated += r.dissipated_j
            normal_impulse += sum(
                k.duration_s
                * sum(a * p for a, p in zip(k.sample.area_weights_m2, k.contact.normal_pressure_pa))
                for k in (r.first, r.second)
            )
            s = r.exit
            e = energy(c, s, r.second.sample)
            ledger = e + dissipated - work - e0
            max_ledger = max(max_ledger, abs(ledger))
            force = r.average_contact_generalized_force
            norm = math.hypot(force[0], force[1])
            if norm > 0:
                last_force_positive = s.time_s
            d = max(r.first.sample.max_depth_m, r.second.sample.max_depth_m)
            row = dict(
                time_s=s.time_s,
                x_m=s.x_m,
                z_m=s.z_m,
                angle_rad=s.angle_rad,
                vx_m_s=s.vx_m_s,
                vz_m_s=s.vz_m_s,
                omega_rad_s=s.omega_rad_s,
                force_x_n=force[0],
                force_z_n=force[1],
                moment_y_nm=force[2],
                force_norm_n=norm,
                area_m2=r.second.sample.area_m2,
                first_area_m2=r.first.sample.area_m2,
                max_depth_m=d,
                energy_j=e,
                push_work_j=work,
                dissipated_j=dissipated,
                energy_ledger_j=ledger,
                max_equation_error=err,
                newton_iterations=max(r.first.iterations, r.second.iterations),
            )
            rows.append(row)
            payload = dict(index=i, step=r.to_dict(), observations=row)
            stream.write(json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n")
            if norm > peak:
                peak = norm
                events["peak_force"] = payload
            if d > depth:
                depth = d
                events["maximum_depth"] = payload
            if r.first.sample.area_m2 > 0 or r.second.sample.area_m2 > 0:
                last_contact = max(
                    k.sample.state.time_s for k in (r.first, r.second) if k.sample.area_m2 > 0
                )
                events["last_contact"] = payload
            if not side_seen and any(
                abs(n[0]) > 0.5
                for k in (r.first, r.second)
                for n, p in zip(k.sample.normals_xz, k.sample.pressure_pa)
                if p > 0
            ):
                side_seen = True
                events["first_side_contact"] = payload
            if i + 1 in (round(0.2 / h), round(0.4 / h)):
                events[f"boundary_{round((i + 1) * h, 1)}"] = payload
            if (i + 1) % round(0.005 / h) == 0 or i + 1 == spec["steps"]:
                frames.append(
                    frame(s, r.second.sample, r.second.contact, force, r.second.mid_velocity)
                )
    if rows:
        with (directory / "trace.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(directory / "frames.json", frames)
    for name, value in events.items():
        write_json(directory / (name + ".json"), value)
    settle = [r for r in rows if 0.15 - 1e-12 <= r["time_s"] <= 0.2 + 1e-12]
    tail = [r for r in rows if r["time_s"] >= 0.55 - 1e-12]
    mg = c.mass_kg * c.gravity_m_s2
    work_scale = max(0.01, max((abs(r["push_work_j"]) for r in rows), default=0))
    gates = dict(
        complete=fail is None and len(rows) == spec["steps"],
        equations=checks,
        depth=max((r["max_depth_m"] for r in rows), default=0) <= c.max_depth_m,
        support=bool(settle)
        and abs(sum(r["force_z_n"] for r in settle) / len(settle) - mg) <= 0.1 * mg,
        settled=bool(settle)
        and max(r["z_m"] for r in settle) - min(r["z_m"] for r in settle) < 0.0001
        and abs(settle[-1]["vz_m_s"]) < 0.001,
        moved=max((r["x_m"] for r in rows), default=0) > 0.06,
        pitched=max((abs(r["angle_rad"]) for r in rows), default=0) > math.pi / 180,
        unloaded=len(tail) >= round(0.05 / h)
        and all(
            r["area_m2"] == 0 and r["first_area_m2"] == 0 and r["force_norm_n"] == 0 for r in tail
        ),
        fell=s.z_m < initial.z_m - 0.01,
        energy=max_ledger <= 0.05 * work_scale,
        positive_contact=peak > 0,
    )
    result = dict(
        **spec,
        status="PASS" if all(gates.values()) else "FAIL_RECORDED",
        gates=gates,
        completed_steps=len(rows),
        failure=fail,
        initial_state=asdict(initial),
        final_state=asdict(s),
        peak_force_n=peak,
        total_normal_impulse_n_s=normal_impulse,
        maximum_depth_m=depth,
        last_contact_s=last_contact,
        last_force_positive_s=last_force_positive,
        side_contact_seen=side_seen,
        max_equation_error=max_error,
        max_energy_ledger_j=max_ledger,
        energy_scale_j=work_scale,
        push_work_j=work,
        dissipated_j=dissipated,
        initial_energy_j=e0,
        settled_mean_force_n=sum(r["force_z_n"] for r in settle) / len(settle) if settle else None,
        maximum_newton_iterations=max((r["newton_iterations"] for r in rows), default=0),
    )
    write_json(directory / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError("Fresh absent results directory required")
    out.mkdir(parents=True)
    specs = [dict(id="h1ms", dt_s=0.001, steps=600), dict(id="h0p5ms", dt_s=0.0005, steps=1200)]
    write_json(
        out / "predeclared-manifest.json",
        dict(
            qualification="PLANAR_EXTRUDED_PRESSURE_FIELD_REFERENCE",
            cases=specs,
            force=dict(start_s=0.2, end_s=0.4, force_x_n=4),
            events=[
                "peak_force",
                "maximum_depth",
                "last_contact",
                "first_side_contact",
                "boundary_0.2",
                "boundary_0.4",
            ],
        ),
    )
    results = [run_case(s, out) for s in specs]
    coarse, fine = results
    compare = {
        k: abs(coarse[k] - fine[k]) / abs(fine[k]) if fine[k] else (0.0 if coarse[k] == 0 else None)
        for k in ("peak_force_n", "total_normal_impulse_n_s")
    }
    times = [r["last_contact_s"] for r in results]
    compare["last_contact_difference_s"] = (
        abs(times[0] - times[1]) if all(t is not None for t in times) else None
    )
    valid = (
        all(r["status"] == "PASS" for r in results)
        and all(
            compare[k] is not None and compare[k] <= 0.1
            for k in ("peak_force_n", "total_normal_impulse_n_s")
        )
        and compare["last_contact_difference_s"] is not None
        and compare["last_contact_difference_s"] <= 0.01
    )
    summary = dict(
        status="PASS" if valid else "FAIL_RECORDED",
        cases=results,
        comparison=compare,
        qualification="2D extruded planar reference, fixed finite support, not TYX or full 3D",
    )
    write_json(out / "summary.json", summary)
    template = Path(__file__).with_name("planar_viewer.html")
    viewer = dict(summary=summary, cases={})
    for spec in specs:
        directory = out / spec["id"]
        with (directory / "trace.csv").open(encoding="utf-8", newline="") as f:
            trace = list(csv.DictReader(f))
        viewer["cases"][spec["id"]] = dict(
            frames=json.loads((directory / "frames.json").read_text(encoding="utf-8")), rows=trace
        )
    payload = json.dumps(
        viewer, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).replace("</", "<\\/")
    (out / "index.html").write_text(
        template.read_text(encoding="utf-8").replace("__RESULTS_JSON__", payload), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
