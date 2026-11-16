"""Run the fixed offline calibration/qualification experiment; never solve FFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from .autoselect import (
    ALGORITHMS,
    CALIBRATION_DELTA_M,
    DOMAIN_M,
    MODULUS_PA,
    POWERS,
    PROFILES,
    RADIUS_M,
    SUPPORTS,
    VALIDATED_DELTAS_M,
    Calibration,
    Selector,
    predict,
    seal_registry,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def fit_case(case, source_sha):
    """Only this one calibration case can affect the fitted coefficients."""
    c = case["config"]
    if (case["id"], case["grid"], case["delta"]) != ("n512-d100um", 512, CALIBRATION_DELTA_M):
        raise ValueError("wrong calibration case")
    if c["radius_m"] != RADIUS_M or c["contact_modulus_pa"] != MODULUS_PA:
        raise ValueError("material mismatch")
    if c["physical_sizes_m"] != [DOMAIN_M, DOMAIN_M] or c["periodic"]:
        raise ValueError("domain mismatch")
    if case["metrics"]["kkt"]["passed"] is not True:
        raise ValueError("unusable calibration solve")
    return Calibration(case["metrics"]["force_n"], source_sha)


def oracle(delta):
    """Independent dimensional Hertz formulas; not the candidate evaluator."""
    m = MODULUS_PA * math.sqrt(RADIUS_M)
    force = (4 / 3) * m * delta * math.sqrt(delta)
    area = math.pi * RADIUS_M * delta
    return {
        "force": force,
        "tangent": 2 * m * math.sqrt(delta),
        "energy": (8 / 15) * m * delta * delta * math.sqrt(delta),
        "area": area,
        "peak": 3 * force / (2 * area),
    }


def load_reference_rows(historical, prospective):
    old_cases = [c for c in historical["cases"] if c["grid"] == 512]
    if len(old_cases) != 3 or len(prospective["cases"]) != 2:
        raise ValueError("duplicate or incomplete reference cases")
    old = {c["delta"]: c for c in old_cases}
    new = {c["delta_m"]: c for c in prospective["cases"]}
    if set(old) != {50e-6, 100e-6, 200e-6} or set(new) != {75e-6, 150e-6}:
        raise ValueError("incomplete fixed reference grid")
    rows = []
    for d in VALIDATED_DELTAS_M:
        c = old[d] if d in old else new[d]
        if c["grid"] != 512:
            raise ValueError("wrong reference grid")
        cfg = c["config"]
        if cfg["radius_m"] != RADIUS_M or cfg["contact_modulus_pa"] != MODULUS_PA:
            raise ValueError("reference material mismatch")
        if cfg["physical_sizes_m"] != [DOMAIN_M, DOMAIN_M] or cfg["periodic"]:
            raise ValueError("reference domain mismatch")
        met = c["metrics"]
        ok = met["kkt"]["passed"] if d in old else met["kkt_pass"]
        if ok is not True:
            raise ValueError("incomplete usable discrete reference evidence")
        rows.append(
            {
                "delta_m": d,
                "role": "calibration"
                if d == CALIBRATION_DELTA_M
                else "retrospective"
                if d in old
                else "prospective-qualification",
                "id": c["id"],
                "source_status": c["status"],
                "fft": {
                    "force": met["force_n"],
                    "area": met["area_m2"],
                    "peak": met["peak_pressure_pa"],
                    "tangent": None,
                    "energy": None,
                },
                "oracle": oracle(d),
                "solver_s": c["timing"]["solver_s"],
            }
        )
    return rows


def qualify(calibration, references):
    validation = {}
    for algorithm in ALGORITHMS:
        rows = []
        for ref in references:
            d = ref["delta_m"]
            value = predict(algorithm, d, calibration, SUPPORTS[algorithm])
            truth = ref["oracle"]
            rows.append(
                {
                    "delta_m": d,
                    "role": ref["role"],
                    "prediction": value,
                    "relative_errors": {k: abs(v - truth[k]) / truth[k] for k, v in value.items()},
                    "relative_to_fft": {
                        k: abs(v - ref["fft"][k]) / ref["fft"][k]
                        if ref["fft"].get(k) is not None
                        else None
                        for k, v in value.items()
                    },
                }
            )
        validation[algorithm] = rows
    return validation


def benchmark(calibration):
    """Fixed measured callable cost, same output dict per comparison group."""
    timings = {a: {} for a in ALGORITHMS}
    raw = []
    checksum = 0.0
    for profile, outputs in PROFILES.items():
        allowed = [a for a in ALGORITHMS if set(outputs) <= set(SUPPORTS[a])]
        per_algorithm = {a: [] for a in allowed}
        for a in allowed:
            for i in range(32):
                predict(a, VALIDATED_DELTAS_M[i % 5], calibration, outputs)
        for batch in range(7):
            order = allowed[batch % len(allowed) :] + allowed[: batch % len(allowed)]
            for algorithm in order:
                start = time.perf_counter_ns()
                total = 0.0
                for i in range(1000):
                    total += predict(algorithm, VALIDATED_DELTAS_M[i % 5], calibration, outputs)[
                        "force"
                    ]
                elapsed = (time.perf_counter_ns() - start) * 1e-9
                checksum += total
                per_algorithm[algorithm].append(elapsed / 1000)
                raw.append(
                    {
                        "profile": profile,
                        "algorithm": algorithm,
                        "batch": batch,
                        "elapsed_s": elapsed,
                        "calls": 1000,
                    }
                )
        for algorithm, values in per_algorithm.items():
            timings[algorithm][profile] = {
                "median_s": statistics.median(values),
                "min_s": min(values),
                "max_s": max(values),
                "batches": 7,
                "calls_per_batch": 1000,
                "note": "batch-average callable time including identical dict output; not whole physics step",
            }
    return timings, {"batches": raw, "checksum": checksum, "warmup_calls_per_profile": 32}


def demo_requests(selector):
    rows = []
    for d in VALIDATED_DELTAS_M:
        for profile in (*PROFILES, "field"):
            for tolerance in (0.05, 0.01, 0.001):
                for mode in ("all", "reduced"):
                    for budget in (1e-3, 1e-9):
                        request = {
                            "delta_m": d,
                            "profile": profile,
                            "tolerances": {m: tolerance for m in PROFILES.get(profile, ("field",))},
                            "allowed": list(ALGORITHMS) if mode == "all" else list(POWERS),
                            "budget_s": budget,
                        }
                        before = time.perf_counter_ns()
                        result = selector.select(**request)
                        elapsed = (time.perf_counter_ns() - before) * 1e-9
                        rows.append(
                            {
                                "id": f"{round(d * 1e6)}-{profile}-{tolerance}-{mode}-{budget}",
                                "request": request,
                                "tolerance": tolerance,
                                "mode": mode,
                                "decision": result,
                                "selection_and_evaluation_s": elapsed,
                            }
                        )
    for ident, request in [
        ("intermediate-125um", {"delta_m": 125e-6}),
        ("finite-layer", {"delta_m": 100e-6, "physics": "finite-elastic-layer"}),
        ("different-material", {"delta_m": 100e-6, "modulus_pa": 2e6}),
        ("no-candidates", {"delta_m": 100e-6, "allowed": []}),
    ]:
        rows.append({"id": ident, "request": request, "decision": selector.select(**request)})
    return rows


def run(historical, prospective, output):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    calibration_path = historical / "n512-d100um/case.json"
    # Commit the one-point parameters before reading any validation point.
    calibration = fit_case(read(calibration_path), sha(calibration_path))
    write(
        output / "calibration.json",
        {
            "parameters": asdict(calibration),
            "coefficients_n_per_m_power": {
                a: calibration.force_n / calibration.delta_m**p for a, p in POWERS.items()
            },
            "exponents": POWERS,
            "fit_points": [CALIBRATION_DELTA_M],
            "frozen_before_validation_read": True,
            "note": "The exponent 1.5 is a declared Hertz prior, not learned from one point.",
        },
    )
    refs = load_reference_rows(
        read(historical / "summary.json"), read(prospective / "summary.json")
    )
    validation = qualify(calibration, refs)
    costs, timing = benchmark(calibration)
    registry = seal_registry(
        calibration,
        validation,
        costs,
        {
            "calibration": sha(calibration_path),
            "historical": sha(historical / "summary.json"),
            "prospective": sha(prospective / "summary.json"),
        },
    )
    write(output / "registry.json", registry)
    # Exercise the deserialized exported artifact, not an in-memory special path.
    selector = Selector(read(output / "registry.json"))
    requests = demo_requests(selector)
    for row in requests:
        decision = row["decision"]
        if decision["status"] == "SELECTED":
            expected = predict(
                decision["algorithm"],
                row["request"]["delta_m"],
                calibration,
                decision["required_outputs"],
            )
            if decision["prediction"] != expected:
                raise ValueError("selected output mismatch")
        elif decision["prediction"] is not None or decision["algorithm"] is not None:
            raise ValueError("refusal contains fabricated output")
    summary = {
        "schema": "phymeo-contact-autoselection-experiment-v1",
        "scope": "one material/geometry, five checked static points, empirical offline qualification",
        "references": refs,
        "registry": registry,
        "requests": requests,
        "selected": sum(r["decision"]["status"] == "SELECTED" for r in requests),
        "refused": sum(r["decision"]["status"] == "REFUSED" for r in requests),
        "timings": timing,
        "total_offline_s": time.perf_counter() - start,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "fft_solves_in_this_runner": 0,
        "continuous_interval_guarantee": False,
        "independent_post_selection_generalization_test": False,
    }
    write(output / "summary.json", summary)
    view = output / "viewer"
    view.mkdir()
    (view / "index.html").write_bytes(Path(__file__).with_name("auto_viewer.html").read_bytes())
    (view / "data.js").write_text(
        "window.AUTO_DATA="
        + json.dumps(summary, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + ";",
        encoding="utf-8",
    )
    write(
        output / "inputs.json",
        {
            "historical": str(historical.resolve()),
            "prospective": str(prospective.resolve()),
            "sources": {
                str(Path(__file__).name): sha(__file__),
                "autoselect.py": sha(Path(__file__).with_name("autoselect.py")),
            },
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "selected": summary["selected"],
                "refused": summary["refused"],
                "fft_solves": 0,
            }
        )
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--prospective", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.historical, args.prospective, args.output)
