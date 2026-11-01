"""Check the preregistered normal-layer coupon with exact bounds and scalar reference traces."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
from dataclasses import asdict, dataclass
from fractions import Fraction as Q
from pathlib import Path
from typing import NamedTuple


INPUT = Path(__file__).resolve().parents[1] / "docs/benchmarks/p65-contact-layer-screen-input.json"
INPUT_SHA = "689fccf60a5142cb4b171e4a15434bc93b0ecea4f3db5174dd17d6987229bc58"


def sha(path, *, normalize=False):
    data = path.read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if normalize else data).hexdigest()


@dataclass(frozen=True)
class Layer:
    k: Q
    delta: Q
    d: Q

    def static_pressure(self, gap):
        return self.k * max(self.delta - gap, 0)


def fit(rows, pressures):
    first, second, dynamic = rows
    g1, g2, gd = (Q(row["gap_m"]) for row in rows)
    p1, p2, pd = pressures
    rate = Q(dynamic["rate_m_s"])
    if not (0 <= g1 < gd < g2 and p1 > p2 > 0 and rate > 0):
        raise ValueError("invalid ordered calibration inputs")
    k = (p1 - p2) / (g2 - g1)
    delta = g1 + p1 / k
    d = (pd / (k * (delta - gd)) - 1) / rate
    if d < 0 or any(Q(row["rate_m_s"]) != 0 for row in (first, second)):
        raise ValueError("invalid rate factor or static calibration row")
    return Layer(k, delta, d)


def holdout_range(corners, gap, rate):
    if gap < 0:
        raise ValueError("hard-core violation")
    elastic = [layer.static_pressure(gap) for layer in corners]
    rate_factor = [max(0, 1 + layer.d * rate) for layer in corners]
    pressure = (min(elastic) * min(rate_factor), max(elastic) * max(rate_factor))
    energy = (
        min(elastic) ** 2 / (2 * max(layer.k for layer in corners)),
        max(elastic) ** 2 / (2 * min(layer.k for layer in corners)),
    )
    return pressure, energy


def enclosed(prediction, declared):
    low, high = map(Q, declared)
    return low <= prediction[0] <= prediction[1] <= high


def calibration(fixture):
    rows = [*fixture["fit"]["static"], fixture["fit"]["compression"]]
    nominal = fit(rows, [Q(row["pressure_pa"]) for row in rows])
    corners = [
        fit(rows, pressures)
        for pressures in itertools.product(*(list(map(Q, row["band_pa"])) for row in rows))
    ]
    held = []
    for row in fixture["holdout"]:
        pressure, energy = holdout_range(corners, Q(row["gap_m"]), Q(row["rate_m_s"]))
        ok = enclosed(pressure, row["pressure_band_pa"])
        if "energy_band_j_m2" in row:
            ok = ok and enclosed(energy, row["energy_band_j_m2"])
        held.append(dict(id=row["id"], pressure_pa=pressure, energy_j_m2=energy, passed=ok))
    try:
        holdout_range(corners, Q(fixture["geometry_negative"]["gap_m"]), Q(0))
    except ValueError as error:
        hard_reject = str(error) == "hard-core violation"
    else:
        hard_reject = False
    bands = {}
    for key in ("k", "delta", "d"):
        values = [getattr(layer, key) for layer in corners]
        bands[key] = (min(values), max(values))
    return nominal, corners, bands, held, hard_reject


def sqrt_upper(value):
    """Exact rational upper bound with a picometre grid for reporting a length."""
    scale = 10**12
    scaled = value * scale**2
    lo = math.isqrt(scaled.numerator // scaled.denominator)
    return Q(lo if lo**2 == scaled else lo + 1, scale)


def screen(fixture, corners, bands):
    c = fixture["controls"]
    mlo, mhi = map(Q, c["mass_range_kg"])
    area, window = Q(c["area_m2"]), Q(c["window_s"])
    u, g0 = Q(c["sc2"]["closing_speed_m_s"]), Q(c["sc2"]["gap_initial_m"])
    klo, _ = bands["k"]
    dlo, _ = bands["d"]
    wlo, whi = bands["delta"]
    full_load = [area * layer.k * layer.delta for layer in corners]
    least_compression = wlo - window / dlo
    support_fraction = 1 - window / (dlo * wlo)
    speed_range = list(map(Q, fixture["rate_range_m_s"]))
    min_node_acceleration = min(
        -Q(1, 8) + (5 * min(full_load) / 12 - Q(rest)) / m
        for rest in c["sc1"]["four_node_rest_rad"] for m in (mlo, mhi)
    )
    # Log(1+du)>0 in the compression first integral gives a strict upper bound.
    peak = sqrt_upper(2 * (2 * mhi) * u / (area * klo * dlo))
    latest_entry = (g0 - wlo) / u
    impulse_deadline = latest_entry + 2 * peak / u
    elastic_capacity = max(full_load) * whi / 2
    least_incident_energy = (2 * mlo) * u**2 / 2
    gates = {
        "support_starts_outward": min(full_load) > mhi / 4,
        "support_stays_compressed": least_compression > 0 and support_fraction > 0,
        "support_only_unloads_in_window": area * klo * least_compression > mhi / 4,
        "four_node_initial_acceleration_positive": min_node_acceleration > 0,
        "free_entry_and_core_clearance": 0 < wlo <= whi < g0 and peak < wlo,
        "positive_impact_impulse_in_window": impulse_deadline < window,
        "certified_rates_in_declared_domain": speed_range[0] <= -1 / dlo and u <= speed_range[1],
        "zero_damping_negative_rejects_all_masses": elastic_capacity < least_incident_energy,
    }
    return {
        "passed": all(gates.values()), "gates": gates,
        "total_initial_load_n": (min(full_load), max(full_load)),
        "support_max_hard_gap_m": window / dlo,
        "support_min_compression_m": least_compression,
        "support_min_force_fraction_of_applied_load": support_fraction,
        "sc1_min_initial_relative_node_acceleration_m_s2": min_node_acceleration,
        "impact_max_compression_m": peak, "impact_min_core_clearance_m": wlo - peak,
        "impact_entry_time_s": ((g0 - whi) / u, latest_entry),
        "impact_half_initial_momentum_delivered_by_s": impulse_deadline,
        "impact_impulse_at_T_lower_ns_for_lightest_mass": mlo * u,
        "zero_damping_max_capacity_j": elastic_capacity,
        "min_incident_kinetic_j": least_incident_energy,
    }


class Frame(NamedTuple):
    t: float
    g: float
    v: float
    force: float
    impulse: float
    dissipation: float
    energy_defect: float
    momentum_defect: float


def rk4(state, h, rhs):
    def shifted(slope, interval):
        return tuple(x + interval * rate for x, rate in zip(state, slope, strict=True))

    a = rhs(state)
    b = rhs(shifted(a, h / 2))
    c = rhs(shifted(b, h / 2))
    d = rhs(shifted(c, h))
    return tuple(x + h * (i + 2*j + 2*k + l) / 6
                 for x, i, j, k, l in zip(state, a, b, c, d, strict=True))


def reference(control, mass, case, layer, dt):
    M = float(2 * mass)
    K = float(Q(control["area_m2"]) * layer.k)
    width, damping = float(layer.delta), float(layer.d)
    if case == "support":
        initial_z, initial_v, applied, entry = width, 0.0, float(mass / 4), 0.0
    else:
        initial_z = width - float(Q(control["sc2"]["gap_initial_m"]))
        initial_v = float(Q(control["sc2"]["closing_speed_m_s"]))
        applied, entry = 0.0, -initial_z / initial_v

    # z=delta-g is signed. y=1+d*v resolves the slow unloading tail without
    # subtracting two nearly equal floats in the pressure factor.
    state = (initial_z, 1 + damping * initial_v, 0.0, 0.0)
    initial_energy = M * initial_v**2 / 2 + K * max(0.0, initial_z)**2 / 2

    def rhs(s):
        z, y, _, _ = s
        velocity = (y - 1) / damping
        elastic = K * max(z, 0.0)
        force = elastic * max(y, 0.0)
        return velocity, damping * (applied-force) / M, force, (force-elastic) * velocity

    frames = []
    ticks = Q(control["window_s"]) / dt
    if ticks.denominator != 1:
        raise ValueError("reference grid must cover the full window")
    for tick in range(int(ticks) + 1):
        t = float(tick * dt)
        if tick:
            prev = float((tick-1) * dt)
            if t <= entry:
                state = (initial_z + t*initial_v, 1+damping*initial_v, 0.0, 0.0)
            else:
                if prev < entry:
                    state = (0.0, 1+damping*initial_v, 0.0, 0.0)
                state = rk4(state, t-max(prev, entry), rhs)
        z, y, impulse, dissipation = state
        v = (y - 1) / damping
        force = K * max(z, 0.0) * max(y, 0.0)
        energy = M*v*v/2 + K*max(z, 0.0)**2/2
        frames.append(Frame(t, width-z, v, force, impulse, dissipation,
                            energy-initial_energy-applied*(z-initial_z)+dissipation,
                            M*(v-initial_v)-applied*t+impulse))
    return frames


def analytical_impact_peaks(K, damping, mass, speed):
    def primitive(y):
        return y - math.log1p(y)

    target = primitive(damping * speed)
    compression = math.sqrt(2 * mass * target / K) / damping
    low, high = 0.0, damping * speed
    for _ in range(64):
        mid = (low + high) / 2
        if mid / 2 + primitive(mid) < target:
            low = mid
        else:
            high = mid
    y = (low + high) / 2
    force = math.sqrt(K * mass * y) * (1+y) / damping
    return compression, force


def extrema(frames):
    return {
        "g_min_m": min(f.g for f in frames), "g_max_m": max(f.g for f in frames),
        "min_force_n": min(f.force for f in frames), "peak_force_n": max(f.force for f in frames),
        "peak_force_time_s": max(frames, key=lambda f: f.force).t,
        "max_energy_balance_error_j": max(abs(f.energy_defect) for f in frames),
        "max_momentum_balance_error_ns": max(abs(f.momentum_defect) for f in frames),
        "last": frames[-1]._asdict(),
    }


def check_reference(fixture, layer, mass, case, output):
    config, control = fixture["diagnostics"], fixture["controls"]
    coarse_dt, fine_dt = map(Q, config["reference_dt_s"])
    stride = coarse_dt / fine_dt
    if stride.denominator != 1:
        raise ValueError("reference grids are not nested")
    coarse = reference(control, mass, case, layer, coarse_dt)
    fine = reference(control, mass, case, layer, fine_dt)
    common = list(zip(coarse, fine[::int(stride)], strict=True))
    checks, errors = {}, {}
    for field, label in (("g","gap_error_m"), ("v","speed_error_m_s"),
                         ("force","force_error_n"), ("impulse","impulse_error_ns")):
        key = "max_common_" + label
        errors[key] = max(abs(getattr(a, field)-getattr(b, field)) for a, b in common)
        checks[key] = errors[key] <= float(Q(config[key]))
    coarse_stats, fine_stats = extrema(coarse), extrema(fine)
    for key in ("max_energy_balance_error_j", "max_momentum_balance_error_ns"):
        checks[key] = max(coarse_stats[key], fine_stats[key]) <= float(Q(config[key]))
    checks["no_sample_penetration"] = min(coarse_stats["g_min_m"], fine_stats["g_min_m"]) >= 0
    checks["dissipation_monotone"] = all(
        b.dissipation >= a.dissipation for a, b in zip(fine, fine[1:])
    )
    peaks = {}
    if case == "impact":
        u = float(Q(control["sc2"]["closing_speed_m_s"]))
        x_peak, force_peak = analytical_impact_peaks(
            float(Q(control["area_m2"])*layer.k), float(layer.d), 2*float(mass), u
        )
        peaks = {"compression_m": x_peak, "force_n": force_peak}
        checks["sampled_peak_compression"] = (
            abs(float(layer.delta)-fine_stats["g_min_m"]-x_peak)
            <= float(Q(config["max_fine_peak_compression_error_m"]))
        )
        checks["sampled_peak_force"] = (
            abs(fine_stats["peak_force_n"]-force_peak)
            <= float(Q(config["max_fine_peak_force_error_n"]))
        )
        checks["window_impulse"] = fine[-1].impulse >= float(mass)*u
    else:
        checks["load_stays_positive"] = fine_stats["min_force_n"] > 0
        checks["stays_inside_layer"] = fine_stats["g_max_m"] < float(layer.delta)

    filename = f"{case}-m{mass.numerator}over{mass.denominator}-fine.csv"
    with (output/filename).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s","hard_gap_m","closing_speed_m_s","normal_force_n",
                         "impulse_ns","dissipation_j","energy_defect_j","momentum_defect_ns"))
        writer.writerows(fine)
    grid_samples = {}
    for h in control["report_h_s"]:
        ratio = Q(h)/fine_dt
        if ratio.denominator != 1:
            raise ValueError("original sample time is absent from the reference grid")
        grid_samples[h] = [
            (f.t, f.g, f.force, f.impulse) for f in fine[::int(ratio)]
        ]
    return {
        "mass_kg": mass, "case": case, "passed": all(checks.values()),
        "checks": checks, "convergence": errors, "analytical_impact_peaks": peaks,
        "coarse": coarse_stats, "fine": fine_stats, "fine_csv": filename,
        "fine_csv_sha256": sha(output/filename), "fine_frame_count": len(fine),
        "original_grid_t_g_force_impulse": grid_samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if sha(INPUT, normalize=True) != INPUT_SHA:
        raise ValueError("preregistered input changed")
    fixture = json.loads(INPUT.read_bytes())
    args.output_dir.mkdir(exist_ok=False)
    nominal, corners, bands, held, hard_reject = calibration(fixture)
    certificate = screen(fixture, corners, bands)
    traces = [
        check_reference(fixture, nominal, Q(mass), case, args.output_dir)
        for mass in fixture["controls"]["trace_masses_kg"] for case in ("support", "impact")
    ]
    ok = (hard_reject and certificate["passed"] and all(row["passed"] for row in held)
          and all(row["passed"] for row in traces))
    result = {
        "schema": "p65-1d-contact-layer-screen-result-v1", "passed": ok,
        "scope": fixture["scope"], "material": fixture["material"],
        "fixture_sha256_lf": INPUT_SHA, "script_sha256_lf": sha(Path(__file__), normalize=True),
        "python": platform.python_version(), "nominal": asdict(nominal), "parameter_bands": bands,
        "held_out": held, "negative_gap_rejected": hard_reject,
        "continuous_certificate": certificate, "reference_traces": traces,
    }
    with (args.output_dir/"result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, default=str, allow_nan=False)
        stream.write("\n")
    for row in traces:
        print(row["case"], row["mass_kg"], "PASS" if row["passed"] else row["checks"])
    print("PASS" if ok else "FAIL", "preregistered one-dimensional material screen")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
