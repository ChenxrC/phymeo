"""Run the four frozen V06 axial dynamic trajectories; no product solver."""

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

from experiments.area_pressure.impact_reference import (
    AxialImpactConfig,
    AxialPairState,
    mechanical_energy_j,
    step_pair,
)


def cases():
    return [
        dict(id=f"{name}-{suffix}", dissipation_s_per_m=d, dt_s=h, steps=n)
        for name, d in (("elastic", 0.0), ("damped", 20.0))
        for suffix, h, n in (("h1ms", 0.001, 500), ("h0p5ms", 0.0005, 1000))
    ]


def write_json(path, data):
    path.write_text(json.dumps(data, allow_nan=False, indent=2), encoding="utf-8")


def energy(config, state):
    delta = max(
        0.0,
        config.radius_m + state.position_b_m + config.target_thickness_m / 2 - state.position_a_m,
    )
    potential = math.pi * config.modulus_pa * (delta**3 / 3 - delta**4 / (6 * config.radius_m))
    kinetic = 0.5 * (
        config.mass_a_kg * state.velocity_a_m_s**2 + config.mass_b_kg * state.velocity_b_m_s**2
    )
    return kinetic + potential


def sample_summary(sample):
    e = sample.elastic
    return dict(
        indentation_m=sample.indentation_m,
        relative_velocity_m_s=sample.relative_velocity_m_s,
        instantaneous_factor=sample.instantaneous_factor,
        instantaneous_force_n=sample.instantaneous_force_n,
        elastic_force_n=e.first_force_n[2],
        elastic_area_m2=e.area_m2,
        cop_world=e.cop_world,
        points=len(e.pressure_pa),
        triangles=len(e.quadrature.triangles),
        maximum_elastic_pressure_pa=max(e.pressure_pa, default=0.0),
        first_force_n=e.first_force_n,
        second_force_n=e.second_force_n,
        first_moment_nm=e.first_moment_nm,
        second_moment_nm=e.second_moment_nm,
    )


def kick_summary(k, config):
    return dict(
        sample=sample_summary(k.sample),
        requested_impulse_n_s=[k.impulse_n_s, -k.impulse_n_s],
        observed_momentum_change_n_s=[
            config.mass_a_kg * (k.velocity_a_after - k.velocity_a_before),
            config.mass_b_kg * (k.velocity_b_after - k.velocity_b_before),
        ],
        **{
            name: getattr(k, name)
            for name in (
                "duration_s",
                "velocity_a_before",
                "velocity_b_before",
                "velocity_a_after",
                "velocity_b_after",
                "relative_mid_velocity_m_s",
                "factor",
                "impulse_n_s",
                "dissipated_j",
            )
        },
    )


def inspect_step(c, r, h):
    errors = [
        abs(c.mass_a_kg * (r.exit.velocity_a_m_s - r.entry.velocity_a_m_s) - r.impulse_n_s),
        abs(c.mass_b_kg * (r.exit.velocity_b_m_s - r.entry.velocity_b_m_s) + r.impulse_n_s),
        abs(r.impulse_n_s - r.first.impulse_n_s - r.second.impulse_n_s),
        abs(r.average_force_n * h - r.impulse_n_s),
        abs(r.drifted.position_a_m - r.entry.position_a_m - h * r.first.velocity_a_after),
        abs(r.drifted.position_b_m - r.entry.position_b_m - h * r.first.velocity_b_after),
        abs(r.drifted.velocity_a_m_s - r.first.velocity_a_after),
        abs(r.drifted.velocity_b_m_s - r.first.velocity_b_after),
        abs(r.exit.position_a_m - r.drifted.position_a_m),
        abs(r.exit.position_b_m - r.drifted.position_b_m),
        abs(r.exit.velocity_a_m_s - r.second.velocity_a_after),
        abs(r.exit.velocity_b_m_s - r.second.velocity_b_after),
        abs(r.exit.time_s - r.entry.time_s - h),
        abs(r.dissipated_j - r.first.dissipated_j - r.second.dissipated_j),
        abs(r.first.duration_s - h / 2),
        abs(r.second.duration_s - h / 2),
        abs(r.first.velocity_a_before - r.entry.velocity_a_m_s),
        abs(r.first.velocity_b_before - r.entry.velocity_b_m_s),
        abs(r.second.velocity_a_before - r.drifted.velocity_a_m_s),
        abs(r.second.velocity_b_before - r.drifted.velocity_b_m_s),
        abs(
            r.first.sample.indentation_m
            - (c.radius_m + r.entry.position_b_m + c.target_thickness_m / 2 - r.entry.position_a_m)
        ),
        abs(
            r.second.sample.indentation_m
            - (
                c.radius_m
                + r.drifted.position_b_m
                + c.target_thickness_m / 2
                - r.drifted.position_a_m
            )
        ),
        abs(
            r.first.sample.relative_velocity_m_s - (r.entry.velocity_a_m_s - r.entry.velocity_b_m_s)
        ),
        abs(
            r.second.sample.relative_velocity_m_s
            - (r.drifted.velocity_a_m_s - r.drifted.velocity_b_m_s)
        ),
        abs(r.drifted.time_s - r.exit.time_s),
        abs(energy(c, r.exit) - mechanical_energy_j(c, r.exit)),
    ]
    for k in (r.first, r.second):
        fe = k.sample.elastic.first_force_n[2]
        errors.extend(
            [
                abs(c.mass_a_kg * (k.velocity_a_after - k.velocity_a_before) - k.impulse_n_s),
                abs(c.mass_b_kg * (k.velocity_b_after - k.velocity_b_before) + k.impulse_n_s),
                abs(k.impulse_n_s - k.duration_s * fe * k.factor),
                abs(k.factor - max(0.0, 1 - c.dissipation_s_per_m * k.relative_mid_velocity_m_s)),
                abs(
                    k.relative_mid_velocity_m_s
                    - 0.5
                    * (
                        k.velocity_a_before
                        - k.velocity_b_before
                        + k.velocity_a_after
                        - k.velocity_b_after
                    )
                ),
            ]
        )
        errors.extend(
            abs(a + b)
            for a, b in zip(
                k.sample.elastic.first_force_n, k.sample.elastic.second_force_n, strict=True
            )
        )
        errors.extend(
            abs(a + b)
            for a, b in zip(
                k.sample.elastic.first_moment_nm, k.sample.elastic.second_moment_nm, strict=True
            )
        )
        if k.impulse_n_s < 0 or k.dissipated_j < -1e-12:
            return False, max(errors)
    return max(errors) <= 1e-12, max(errors)


def run_case(spec, output):
    c = AxialImpactConfig(dissipation_s_per_m=spec["dissipation_s_per_m"])
    state = AxialPairState(0.0, 0.027, 0.0, -0.2, 0.0)
    initial = state
    h = spec["dt_s"]
    e0 = energy(c, state)
    p0 = c.mass_a_kg * state.velocity_a_m_s + c.mass_b_kg * state.velocity_b_m_s
    trace = []
    events = {}
    total_j = 0.0
    total_d = 0.0
    checks = True
    max_binding = 0.0
    max_p = 0.0
    max_e = 0.0
    max_ledger = 0.0
    peak = -1.0
    max_depth = -math.inf
    unloading = False
    late_empty = True
    max_points = 0
    max_triangles = 0
    try:
        for i in range(spec["steps"]):
            r = step_pair(c, state, h)
            ok, err = inspect_step(c, r, h)
            checks = checks and ok
            max_binding = max(max_binding, err)
            total_j += r.impulse_n_s
            total_d += r.dissipated_j
            state = r.exit
            momentum = c.mass_a_kg * state.velocity_a_m_s + c.mass_b_kg * state.velocity_b_m_s
            en = energy(c, state)
            max_p = max(max_p, abs(momentum - p0))
            max_e = max(max_e, abs(en - e0))
            max_ledger = max(max_ledger, abs(en + total_d - e0))
            depth = max(r.first.sample.indentation_m, r.second.sample.indentation_m)
            if r.impulse_n_s > 0:
                events.setdefault("first_positive", (i, r))
                events["last_positive"] = (i, r)
                unloading = unloading or state.velocity_a_m_s - state.velocity_b_m_s > 0
            if r.average_force_n > peak:
                peak = r.average_force_n
                events["peak_average_force"] = (i, r)
            if depth > max_depth:
                max_depth = depth
                events["maximum_indentation"] = (i, r)
            if r.entry.time_s >= 0.4 - 1e-12:
                late_empty = late_empty and all(
                    k.sample.elastic.area_m2 == 0 and k.impulse_n_s == 0
                    for k in (r.first, r.second)
                )
            for k in (r.first, r.second):
                max_points = max(max_points, len(k.sample.elastic.pressure_pa))
                max_triangles = max(max_triangles, len(k.sample.elastic.quadrature.triangles))
            trace.append(
                dict(
                    index=i,
                    entry=asdict(r.entry),
                    drifted=asdict(r.drifted),
                    exit=asdict(r.exit),
                    first=kick_summary(r.first, c),
                    second=kick_summary(r.second, c),
                    requested_impulse_n_s=[r.impulse_n_s, -r.impulse_n_s],
                    observed_momentum_change_n_s=[
                        c.mass_a_kg * (r.exit.velocity_a_m_s - r.entry.velocity_a_m_s),
                        c.mass_b_kg * (r.exit.velocity_b_m_s - r.entry.velocity_b_m_s),
                    ],
                    impulse_n_s=r.impulse_n_s,
                    average_force_n=r.average_force_n,
                    dissipated_j=r.dissipated_j,
                    cumulative_impulse_n_s=total_j,
                    cumulative_dissipated_j=total_d,
                    momentum_kg_m_s=momentum,
                    mechanical_energy_j=en,
                    energy_plus_dissipation_j=en + total_d,
                    binding_pass=ok,
                    binding_error=err,
                )
            )
    except Exception as exc:
        write_json(
            output / (spec["id"] + "-failure.json"),
            dict(
                case=spec,
                error_type=type(exc).__name__,
                message=str(exc),
                completed_steps=len(trace),
                state=asdict(state),
                trace=trace,
            ),
        )
        raise
    relative = state.velocity_a_m_s - state.velocity_b_m_s
    gap = state.position_a_m - state.position_b_m - c.radius_m - c.target_thickness_m / 2
    gates = dict(
        complete=len(trace) == spec["steps"],
        binding=checks,
        depth_domain=0 < max_depth <= c.max_indentation_m and max_depth < c.target_thickness_m,
        momentum=max_p <= 0.001,
        compression_and_unloading="first_positive" in events and unloading,
        separated=gap > 0 and relative > 0 and late_empty,
        energy_ledger=max_ledger <= 0.05 * e0,
        nonnegative_dissipation=total_d >= -1e-12,
    )
    if c.dissipation_s_per_m == 0:
        gates.update(
            elastic_energy=max_e <= 0.05 * e0,
            elastic_impulse=abs(total_j - 2 / 15) / (2 / 15) <= 0.05,
            elastic_velocity_a=abs(state.velocity_a_m_s - 1 / 15) / (1 / 15) <= 0.05,
            elastic_velocity_b=abs(state.velocity_b_m_s + 2 / 15) / (2 / 15) <= 0.05,
        )
    else:
        gates.update(dissipative=en < e0 and total_d > 0, relative_rebound=0 < relative < 0.2)
    event_manifest = {}
    for name, (i, r) in events.items():
        filename = f"{spec['id']}-event-{name}.json"
        write_json(
            output / filename, dict(case=spec, config=asdict(c), index=i, record=r.to_dict())
        )
        event_manifest[name] = dict(index=i, file=filename, time_s=r.exit.time_s)
    metrics = dict(
        peak_average_force_n=peak,
        total_impulse_n_s=total_j,
        max_indentation_m=max_depth,
        max_momentum_error_kg_m_s=max_p,
        max_energy_error_j=max_e,
        max_energy_ledger_error_j=max_ledger,
        cumulative_dissipated_j=total_d,
        final_energy_j=en,
        initial_energy_j=e0,
        final_relative_velocity_m_s=relative,
        final_gap_m=gap,
        max_binding_error=max_binding,
        max_points=max_points,
        max_triangles=max_triangles,
    )
    result = dict(
        case=spec,
        config=asdict(c),
        initial=asdict(initial),
        final=asdict(state),
        metrics=metrics,
        gates=gates,
        passed=all(gates.values()),
        events=event_manifest,
        trace_file=spec["id"] + "-trace.json",
    )
    write_json(
        output / result["trace_file"],
        dict(case=spec, config=asdict(c), initial=asdict(initial), steps=trace),
    )
    write_json(output / (spec["id"] + "-summary.json"), result)
    return result, trace


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    fixed = cases()
    manifest = dict(
        phase="DYNAMIC",
        qualification="NUMPY_AXIAL_TWO_BODY_REFERENCE",
        model="V02 radial pressure + two finite masses + KDK midpoint HC kicks",
        cases=fixed,
        event_rules=[
            "first_positive",
            "peak_average_force",
            "maximum_indentation",
            "last_positive",
        ],
        bounds=dict(
            window_s=0.5,
            dt_s=[0.001, 0.0005],
            momentum_error=0.001,
            energy_fraction=0.05,
            step_convergence_fraction=0.1,
            binding_abs=1e-12,
        ),
        note="Symmetric centered normal impact, zero torque; no general 6DOF/runtime/TYX qualification.",
    )
    write_json(output / "manifest.json", manifest)
    records = []
    viewer_cases = []
    for spec in fixed:
        result, trace = run_case(spec, output)
        records.append(result)
        viewer_cases.append(
            dict(
                summary=result,
                frames=[
                    dict(
                        time_s=0.0,
                        position_a_m=0.027,
                        position_b_m=0.0,
                        velocity_a_m_s=-0.2,
                        velocity_b_m_s=0.0,
                        force_n=0.0,
                        area_m2=0.0,
                        indentation_m=-0.002,
                        impulse_n_s=0.0,
                        energy_j=0.01,
                        ledger_j=0.01,
                        momentum_error=0.0,
                    )
                ]
                + [
                    dict(
                        time_s=s["exit"]["time_s"],
                        position_a_m=s["exit"]["position_a_m"],
                        position_b_m=s["exit"]["position_b_m"],
                        velocity_a_m_s=s["exit"]["velocity_a_m_s"],
                        velocity_b_m_s=s["exit"]["velocity_b_m_s"],
                        force_n=s["average_force_n"],
                        area_m2=s["second"]["sample"]["elastic_area_m2"],
                        indentation_m=s["second"]["sample"]["indentation_m"],
                        impulse_n_s=s["cumulative_impulse_n_s"],
                        energy_j=s["mechanical_energy_j"],
                        ledger_j=s["energy_plus_dissipation_j"],
                        momentum_error=s["momentum_kg_m_s"] + 0.1,
                    )
                    for s in trace
                ],
            )
        )
        print(
            json.dumps(
                dict(case=spec["id"], passed=result["passed"], metrics=result["metrics"]),
                allow_nan=False,
            ),
            flush=True,
        )
    comparisons = []
    for i in (0, 2):
        coarse, fine = records[i : i + 2]
        diffs = {
            k: abs(coarse["metrics"][k] - fine["metrics"][k]) / abs(fine["metrics"][k])
            for k in ("peak_average_force_n", "total_impulse_n_s")
        }
        comparisons.append(
            dict(
                dissipation_s_per_m=fine["case"]["dissipation_s_per_m"],
                relative_changes=diffs,
                passed=max(diffs.values()) <= 0.1,
            )
        )
    passed = all(r["passed"] for r in records) and all(c["passed"] for c in comparisons)
    summary = dict(
        manifest=manifest,
        status="V06_DYNAMIC_PASS" if passed else "V06_DYNAMIC_FAIL",
        cases=records,
        step_comparisons=comparisons,
    )
    write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "passed"] + list(records[0]["metrics"]))
        writer.writeheader()
        for r in records:
            writer.writerow(dict(case=r["case"]["id"], passed=r["passed"], **r["metrics"]))
    payload = dict(summary=summary, cases=viewer_cases)
    template = Path(__file__).with_name("impact_viewer.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(
        template.replace(
            "__RESULTS_JSON__", json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(dict(status=summary["status"], steps=3000, step_comparisons=comparisons)),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    raise SystemExit(run(parser.parse_args().output))
