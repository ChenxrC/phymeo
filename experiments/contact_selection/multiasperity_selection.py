"""Finite-point force surrogate, fitted offline; never a pressure-field solver."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

GEOMETRY = "compact-double-asperity-v1"
POINTS = (20e-6, 40e-6, 60e-6, 80e-6, 100e-6)
KNOTS = (0.0, 20e-6, 60e-6, 100e-6)
VALIDATION = (40e-6, 80e-6)


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("expected a finite number, not a flag")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite value")
    return value


def coefficients(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("four descending polynomial coefficients required")
    if any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in value):
        raise ValueError("three polynomial intervals required")
    return tuple(tuple(finite(v) for v in row) for row in value)


def evaluate(coeffs, delta):
    """Horner evaluation of scipy PCHIP's saved coefficients; no fitting here."""
    index = 0 if delta < KNOTS[1] else 1 if delta < KNOTS[2] else 2
    x = delta - KNOTS[index]
    return ((coeffs[0][index] * x + coeffs[1][index]) * x + coeffs[2][index]) * x + coeffs[3][index]


def qualification(records, validation):
    """Separate reference consistency from hold-out agreement; no retuning."""
    if set(records) != set(POINTS) or set(validation) != set(VALIDATION):
        raise ValueError("exact frozen reference and validation membership required")
    checks = []
    for delta in POINTS:
        row = records[delta]
        if type(row["kkt_256"]) is not bool or type(row["kkt_512"]) is not bool:
            raise ValueError("KKT must be a boolean")
        error = finite(row["force_grid_relative"])
        if error < 0:
            raise ValueError("negative error")
        checks.append(row["kkt_256"] and row["kkt_512"] and error <= 0.02)
    for delta in VALIDATION:
        error = finite(validation[delta]["relative_error"])
        if error < 0:
            raise ValueError("negative error")
        checks.append(error <= 0.05)
    return all(checks)


@dataclass(frozen=True)
class ForcePlan:
    """An already selected force-only plan. Evidence covers five points only."""

    polynomial: tuple

    def __post_init__(self):
        object.__setattr__(self, "polynomial", coefficients(self.polynomial))

    def predict(self, delta_m):
        delta = finite(delta_m)
        if delta not in POINTS:
            raise ValueError("outside the five qualified points; no interpolation qualification")
        result = evaluate(self.polynomial, delta)
        if not math.isfinite(result) or result <= 0:
            raise ValueError("invalid positive force prediction")
        return {"force_n": result}


def choose_plan(model, *, outputs=("force_n",), geometry=GEOMETRY, modulus_pa=1e6):
    """Initialization admission, not a per-step race between unrelated solvers."""
    if geometry != GEOMETRY or finite(modulus_pa) != 1e6:
        raise ValueError("geometry or material is outside the calibrated domain")
    if tuple(outputs) != ("force_n",):
        raise ValueError(
            "force-only model: area, pressure, wrench, stiffness and energy unsupported"
        )
    if model.get("geometry") != GEOMETRY or model.get("modulus_pa") != 1e6:
        raise ValueError("model domain mismatch")
    if model.get("qualified_points_m") != list(POINTS):
        raise ValueError("model has a different evidence domain")
    if (
        model.get("schema") != "multi-asperity-force-plan-v1"
        or model.get("knots_m") != list(KNOTS)
        or model.get("capabilities") != ["force_n"]
    ):
        raise ValueError("schema, interpolation knots or capability binding mismatch")
    coeffs = coefficients(model["coefficients"])
    forces = tuple(finite(v) for v in model["calibration_force_n"])
    if len(forces) != 4 or forces[0] != 0 or not all(b > a for a, b in zip(forces, forces[1:])):
        raise ValueError("invalid calibration forces")
    for delta, force in zip(KNOTS, forces):
        if not math.isclose(evaluate(coeffs, delta), force, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("polynomial no longer matches the calibration knots")
    for row in model["validation"]:
        delta, truth = finite(row["delta_m"]), finite(row["force_n"])
        if delta not in VALIDATION or truth <= 0:
            raise ValueError("invalid validation point or force")
        prediction = evaluate(coeffs, delta)
        error = abs(prediction - truth) / truth
        if not math.isclose(
            prediction, finite(row["predicted_n"]), rel_tol=1e-12, abs_tol=1e-14
        ) or not math.isclose(error, finite(row["relative_error"]), rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("polynomial and validation evidence no longer match")
    records = {float(row["delta_m"]): row for row in model["reference_checks"]}
    validations = {float(row["delta_m"]): row for row in model["validation"]}
    if len(model["reference_checks"]) != 5 or len(model["validation"]) != 2:
        raise ValueError("duplicate or missing qualification records")
    if not qualification(records, validations):
        raise ValueError("not qualified: numerical reference or held-out force target failed")
    return ForcePlan(model["coefficients"])


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_bound(path):
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def bench(plan):
    for _ in range(32):
        plan.predict(60e-6)
    batches, checksum = [], 0.0
    for _ in range(7):
        started = time.perf_counter()
        for index in range(1000):
            checksum += plan.predict(POINTS[index % 5])["force_n"]
        batches.append((time.perf_counter() - started) / 1000)
    return {
        "batch_seconds_per_call": batches,
        "median_s": statistics.median(batches),
        "min_s": min(batches),
        "max_s": max(batches),
        "checksum": checksum,
        "scope": "checked force evaluation and output dict, after initialization; no FFT",
    }


def _run_in_directory(reference, output):
    """Fit frozen knots, then read the held-out reference records."""
    from experiments.contact_selection.multiasperity_reference import configuration

    reference, output = Path(reference), Path(output)
    if not output.is_absolute():
        raise ValueError("absolute fresh output required")
    start = time.perf_counter()
    bindings, read_order = [], []

    def load(n, delta):
        path = reference / f"c-n{n}-d{round(delta * 1e6):03d}um" / "case.json"
        item, binding = read_bound(path)
        if (
            item["grid"] != n
            or item["delta_m"] != delta
            or item["mode"] != "coupled"
            or item["config"] != configuration(n, delta, "coupled")
        ):
            raise ValueError("reference geometry/material/phase binding mismatch")
        metrics = item.get("metrics")
        if (
            not isinstance(metrics, dict)
            or not isinstance(metrics.get("force_n"), (int, float))
            or not math.isfinite(metrics["force_n"])
            or metrics["force_n"] <= 0
        ):
            raise ValueError(f"unavailable numerical force reference: {path}")
        bindings.append(binding)
        read_order.append(str(path))
        return item

    calibration = [load(512, delta) for delta in KNOTS[1:]]
    forces = [0.0] + [item["metrics"]["force_n"] for item in calibration]
    if not all(math.isfinite(f) for f in forces) or not all(
        b > a for a, b in zip(forces, forces[1:])
    ):
        raise ValueError("calibration forces must be finite and strictly increasing")
    from scipy.interpolate import PchipInterpolator

    interpolator = PchipInterpolator(KNOTS, forces, extrapolate=False)
    model = {
        "schema": "multi-asperity-force-plan-v1",
        "geometry": GEOMETRY,
        "modulus_pa": 1e6,
        "knots_m": list(KNOTS),
        "calibration_force_n": forces,
        "coefficients": interpolator.c.tolist(),
        "coefficient_order": "descending powers of local delta_m in each interval; F in N",
        "fitter": {
            "scipy": importlib.metadata.version("scipy"),
            "method": "PchipInterpolator, extrapolate=False",
        },
        "author_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "qualified_points_m": list(POINTS),
        "capabilities": ["force_n"],
        "zero_point": "analytic no-contact boundary, not a fourth numerical solve",
        "source_bindings": bindings.copy(),
    }
    write_json(output / "calibration-before-validation.json", model)
    read_order.append("WROTE_CALIBRATION_BEFORE_VALIDATION")
    all_cases = {float(item["delta_m"]): item for item in calibration}
    for delta in VALIDATION:
        all_cases[delta] = load(512, delta)
    validation = []
    for delta in VALIDATION:
        truth = all_cases[delta]["metrics"]["force_n"]
        predicted = evaluate(coefficients(model["coefficients"]), delta)
        validation.append(
            {
                "delta_m": delta,
                "force_n": truth,
                "predicted_n": predicted,
                "relative_error": abs(predicted - truth) / truth,
            }
        )
    reference_checks = []
    for delta in POINTS:
        coarse = load(256, delta)
        fine = all_cases[delta]
        reference_checks.append(
            {
                "delta_m": delta,
                "kkt_256": coarse["metrics"]["kkt_pass"],
                "kkt_512": fine["metrics"]["kkt_pass"],
                "force_grid_relative": abs(
                    coarse["metrics"]["force_n"] - fine["metrics"]["force_n"]
                )
                / fine["metrics"]["force_n"],
            }
        )
    model.update(
        validation=validation, reference_checks=reference_checks, source_bindings=bindings.copy()
    )
    decisions = []
    plan = None
    begin = time.perf_counter()
    try:
        plan = choose_plan(model)
        decisions.append(
            {
                "outputs": ["force_n"],
                "status": "QUALIFIED_FIVE_POINTS",
                "predictions": [plan.predict(d) for d in POINTS],
            }
        )
    except ValueError as exc:
        decisions.append({"outputs": ["force_n"], "status": "REFUSED", "reason": str(exc)})
    init_s = time.perf_counter() - begin
    for request in (
        {"outputs": ("pressure_pa",)},
        {"outputs": ("wrench",)},
        {"geometry": "hertz-paraboloid"},
        {"modulus_pa": 2e6},
    ):
        try:
            choose_plan(model, **request)
        except ValueError as exc:
            decisions.append({"request": request, "status": "REFUSED", "reason": str(exc)})
        else:
            raise AssertionError("unqualified request accepted")
    if plan is not None:
        try:
            plan.predict(55e-6)
        except ValueError as exc:
            decisions.append({"delta_m": 55e-6, "status": "REFUSED", "reason": str(exc)})
        else:
            raise AssertionError("unvalidated interpolation accepted")
    timing = bench(plan) if plan is not None else None
    model["admission"] = decisions[0]["status"]
    write_json(output / "model.json", model)
    result = {
        "model": model,
        "decisions": decisions,
        "read_order": read_order,
        "initialization_s": init_s,
        "evaluation": timing,
        "offline_excluding_fft_s": time.perf_counter() - start,
        "research_target_passed": plan is not None,
        "limitations": "five-point scalar evidence only; no pressure/CoP/moment/energy or runtime dynamics qualification",
    }
    write_json(output / "summary.json", result)
    return result


def run(reference, output):
    output = Path(output)
    if not output.is_absolute():
        raise ValueError("absolute fresh output required")
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _run_in_directory(reference, output)
    except (ValueError, KeyError, TypeError, ZeroDivisionError, FileNotFoundError) as exc:
        result = {
            "model": {"admission": "REFUSED_UNAVAILABLE_REFERENCE", "validation": []},
            "research_target_passed": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "reference": str(Path(reference).resolve()),
            "evaluation": None,
            "new_solves": 0,
        }
        write_json(output / "summary.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.reference, args.output)
    print(
        json.dumps(
            {
                "status": result["model"]["admission"],
                "validation": result["model"]["validation"],
                "evaluation": result["evaluation"],
            }
        )
    )
    return 0 if result["research_target_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
