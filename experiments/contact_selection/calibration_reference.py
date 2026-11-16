"""Two new, fixed FFT512 validation points; no calibration fit or new solver.

Only ContactMechanics performs the elastic solve. The older nine-case runner is
never invoked or patched. Its unbounded analytic and metadata helpers are reused.
Every raw optimizer array is copied before metadata decoding, including when the
public system raises during its post-solve boundary check.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from experiments.contact_selection.fft_reference import (
    _external_api,
    _field,
    _process_memory,
    _source,
    hertz_fields,
    hertz_reference,
    preview_fields,
)

SCHEMA = "phymeo-calibration-reference-v1"
GRID = 512
DELTAS_M = (75e-6, 150e-6)
RADIUS_M = 0.02
MODULUS_PA = 1e6
DOMAIN_M = 0.16
PENTOL_M = 1e-10
FORCETOL_N = 1e-10
THERMOTOL = 1e-8
MAXITER = 10000


def configuration(delta_m):
    if (
        isinstance(delta_m, (bool, np.bool_))
        or not isinstance(delta_m, (int, float, np.integer, np.floating))
        or float(delta_m) not in DELTAS_M
    ):
        raise ValueError("only the new 75 and 150 micrometre validation points are allowed")
    delta_m = float(delta_m)
    return {
        "grid": GRID,
        "delta_m": delta_m,
        "radius_m": RADIUS_M,
        "contact_modulus_pa": MODULUS_PA,
        "physical_sizes_m": [DOMAIN_M, DOMAIN_M],
        "computational_grid": [2 * GRID, 2 * GRID],
        "padding_factor": 2,
        "periodic": False,
        "check_boundaries": True,
        "external_force": None,
        "hardness": None,
        "initial_displacements": None,
        "initial_forces": None,
        "geometry": "full-domain negative Hertz paraboloid; small-deformation approximation",
        "coordinates": "x_i=(i-n//2)*L/n; first axis x, second y; origin node",
        "solver": {
            "pentol_m": PENTOL_M,
            "forcetol_n": FORCETOL_N,
            "thermotol": THERMOTOL,
            "maxiter": MAXITER,
        },
        "kkt_limits": {
            "gap_min_m": -PENTOL_M,
            "tensile_pixel_force_n": FORCETOL_N,
            "active_gap_abs_m": 10 * PENTOL_M,
            "complementarity_normalized": 1e-5,
            "reconstructed_pixel_force_error_n": 10 * FORCETOL_N,
            "padding_pixel_force_n": 10 * FORCETOL_N,
            "physical_outer_ring_force_n": 0.0,
        },
        "hertz_relative_target": 0.05,
        "contact_area_rule": "count(raw f > 0) * dA, without fitted thresholds",
    }


def analytic_reference(delta_m):
    configuration(delta_m)
    reference = hertz_reference(delta_m, RADIUS_M, MODULUS_PA)
    return {
        "radius_m": reference["radius_m"],
        "force_n": reference["force_n"],
        "area_m2": reference["area_m2"],
        "peak_pressure_pa": reference["peak_pressure_pa"],
        "tangent_n_m": 2 * MODULUS_PA * math.sqrt(RADIUS_M * delta_m),
    }


def input_arrays(delta_m):
    configuration(delta_m)
    x = (np.arange(GRID, dtype=np.float64) - GRID // 2) * (DOMAIN_M / GRID)
    heights = -(x[:, None] ** 2 + x[None, :] ** 2) / (2 * RADIUS_M)
    return {"x": x, "heights": heights, "delta_m": np.array(delta_m, dtype=np.float64)}


def json_value(value):
    """Owned JSON tree, preserving nested arrays and nonfinite failure evidence."""
    if isinstance(value, np.ndarray):
        return {
            "ndarray_data": json_value(value.tolist()),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    raise TypeError(f"unsupported optimizer metadata: {type(value).__name__}")


def optimizer_metadata(result, arrays):
    # Two passes: a nested diagnostic must never prevent copying x/jac first.
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            arrays[f"result_{key}"] = value.copy()
    return {
        key: (
            {"npz_key": f"result_{key}", "shape": list(value.shape), "dtype": str(value.dtype)}
            if isinstance(value, np.ndarray)
            else json_value(value)
        )
        for key, value in result.items()
    }


def extract_fields(result, substrate, system, delta_m):
    configuration(delta_m)
    inputs = input_arrays(delta_m)
    area = (DOMAIN_M / GRID) ** 2
    if not math.isclose(float(substrate.area_per_pt), area, rel_tol=1e-14, abs_tol=0):
        raise ValueError("pixel area does not match the frozen physical domain")
    slices = substrate.local_topography_subdomain_slices
    if (
        not isinstance(slices, tuple)
        or len(slices) != 2
        or any(
            not isinstance(s, slice) or s.start != 0 or s.stop != GRID or s.step not in (None, 1)
            for s in slices
        )
    ):
        raise ValueError("physical-domain slices are not the registered serial slices")
    if isinstance(result.offset, (bool, np.bool_)) or float(result.offset) != delta_m:
        raise ValueError("prescribed displacement changed")
    u_full = _field(result.x, (2 * GRID, 2 * GRID), "result.x")
    force = _field(result.jac, (GRID, GRID), "result.jac")
    gap = u_full[slices] - (inputs["heights"] + delta_m)
    public_gap = _field(system.compute_gap(u_full, result.offset), (GRID, GRID), "public gap")
    if not np.allclose(gap, public_gap, rtol=0, atol=1e-14):
        raise ValueError("public gap differs from the saved state and height")
    hp, hu = hertz_fields(inputs["x"], delta_m)
    return {
        **inputs,
        "u_full": u_full,
        "u": u_full[slices].copy(),
        "f": force,
        "p": force / area,
        "gap": gap,
        "public_gap": public_gap,
        "contact_mask": force > 0,
        "hertz_pressure": hp,
        "hertz_displacement": hu,
        "grid": GRID,
        "delta": delta_m,
        "pixel_area_m2": area,
    }


def assess_fields(fields, reaction_full, *, solver_success):
    """Same frozen KKT gates as the original reference, with separate Hertz errors."""
    delta_m = fields["delta"]
    reference = analytic_reference(delta_m)
    force, pressure, gap = (_field(fields[k], (GRID, GRID), k) for k in ("f", "p", "gap"))
    reaction = _field(reaction_full, (2 * GRID, 2 * GRID), "positive reconstructed reaction")
    area = (DOMAIN_M / GRID) ** 2
    active = force > 0
    padding = np.ones(reaction.shape, dtype=bool)
    padding[:GRID, :GRID] = False
    boundary = np.concatenate((force[0], force[-1], force[1:-1, 0], force[1:-1, -1]))
    kkt = {
        "minimum_pressure_pa": float(np.min(pressure)),
        "minimum_gap_m": float(np.min(gap)),
        "active_gap_max_abs_m": float(np.max(np.abs(gap[active]))) if np.any(active) else 0.0,
        "complementarity_abs_work_j": float(np.sum(np.abs(force * gap))),
        "reconstructed_force_max_error_n": float(np.max(np.abs(reaction[:GRID, :GRID] - force))),
        "padding_force_max_abs_n": float(np.max(np.abs(reaction[padding]))),
        "outer_ring_force_max_abs_n": float(np.max(np.abs(boundary))),
    }
    kkt["complementarity_normalized"] = kkt["complementarity_abs_work_j"] / (
        reference["force_n"] * delta_m
    )
    kkt["checks"] = {
        "optimizer": bool(solver_success),
        "pressure": kkt["minimum_pressure_pa"] >= -FORCETOL_N / area,
        "nonpenetration": kkt["minimum_gap_m"] >= -PENTOL_M,
        "active_gap": kkt["active_gap_max_abs_m"] <= 10 * PENTOL_M,
        "complementarity": kkt["complementarity_normalized"] <= 1e-5,
        "elastic_consistency": kkt["reconstructed_force_max_error_n"] <= 10 * FORCETOL_N,
        "zero_padding_traction": kkt["padding_force_max_abs_n"] <= 10 * FORCETOL_N,
        "free_physical_boundary": kkt["outer_ring_force_max_abs_n"] == 0.0,
    }
    measured = {
        "force_n": float(np.sum(force)),
        "area_m2": int(np.count_nonzero(active)) * area,
        "peak_pressure_pa": float(np.max(pressure)),
    }
    errors = {
        name: abs(measured[key] - reference[key]) / reference[key]
        for name, key in (("force", "force_n"), ("area", "area_m2"), ("peak", "peak_pressure_pa"))
    }
    return {
        **measured,
        "contact_pixels": int(np.count_nonzero(active)),
        "pixel_area_m2": area,
        "kkt_pass": all(kkt["checks"].values()),
        "kkt": kkt,
        "hertz": reference,
        "relative_errors": errors,
        "hertz_target_pass": all(value <= 0.05 for value in errors.values()),
        "tangent_note": "analytic Hertz tangent only; no measured FFT tangent or fitted law",
    }


def _load_external():
    api, binding = _external_api()
    from ContactMechanics.Optimization import constrained_conjugate_gradients

    return (*api, constrained_conjugate_gradients), binding


def _write_json(path, data):
    path.write_text(
        json.dumps(json_value(data), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _failure(exc, phase):
    return {
        "phase": phase,
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _run_case(directory, delta_m, api, binding):
    started = time.perf_counter()
    case_id = "n512-d075um" if delta_m == DELTAS_M[0] else "n512-d150um"
    case = {
        "id": case_id,
        "delta_m": delta_m,
        "grid": GRID,
        "config": configuration(delta_m),
        "status": "ERROR",
        "return_code": 1,
        "metrics": None,
        "usable_discrete_solution": False,
        "timing": {"solver_s": None},
        "files": {},
        "binding": binding,
        "failures": [],
    }
    folder = directory / case_id
    folder.mkdir()
    arrays = input_arrays(delta_m)
    raw_result = None
    solve_returned = False
    system = substrate = None
    memory_before = _process_memory()
    phase = "construction"
    try:
        Topography, HalfSpace, ContactSystem, public_solver = api
        begin = time.perf_counter()
        surface = Topography(arrays["heights"], (DOMAIN_M, DOMAIN_M), periodic=False, unit="m")
        substrate = HalfSpace(
            (GRID, GRID),
            young=MODULUS_PA,
            physical_sizes=(DOMAIN_M, DOMAIN_M),
            fft="serial",
            check_boundaries=True,
        )
        system = ContactSystem(substrate, surface)
        case["timing"]["construction_including_first_fft_s"] = time.perf_counter() - begin

        def retain_result(*args, **kwargs):
            nonlocal raw_result
            raw_result = public_solver(*args, **kwargs)
            # Capture before NonSmoothContactSystem's later boundary check.
            for key, value in raw_result.items():
                if isinstance(value, np.ndarray):
                    arrays[f"result_{key}"] = value.copy()
            return raw_result

        phase = "solve"
        begin = time.perf_counter()
        try:
            system.minimize_proxy(
                solver=retain_result,
                offset=delta_m,
                external_force=None,
                hardness=None,
                pentol=PENTOL_M,
                forcetol=FORCETOL_N,
                thermotol=THERMOTOL,
                maxiter=MAXITER,
            )
            if raw_result is None:
                raise ValueError("public system returned without the registered solver result")
            solve_returned = True
        finally:
            case["timing"]["solver_s"] = time.perf_counter() - begin
    except Exception as exc:
        case["failures"].append(_failure(exc, phase))

    if raw_result is not None:
        phase = "optimizer-export"
        try:
            case["optimizer"] = optimizer_metadata(raw_result, arrays)
            case["counters"] = {
                "solver_iterations": int(raw_result.nit),
                "solver_function_evaluations": int(raw_result.nfev),
                "extra_elastic_force_evaluations": 0,
            }
            phase = "field-export"
            fields = extract_fields(raw_result, substrate, system, delta_m)
            arrays.update(
                {key: value for key, value in fields.items() if isinstance(value, np.ndarray)}
            )
            case["preview"] = preview_fields(fields)
            if solve_returned:
                phase = "elastic-consistency"
                begin = time.perf_counter()
                reaction = -substrate.evaluate_force(fields["u_full"])
                case["timing"]["consistency_force_s"] = time.perf_counter() - begin
                case["counters"]["extra_elastic_force_evaluations"] = 1
                arrays["reconstructed_reaction_full"] = np.array(reaction, copy=True)
                phase = "assessment"
                case["metrics"] = assess_fields(fields, reaction, solver_success=raw_result.success)
                metrics = case["metrics"]
                case["usable_discrete_solution"] = metrics["kkt_pass"]
                case["status"] = (
                    "SOLVER_FAIL"
                    if not raw_result.success
                    else "KKT_FAIL"
                    if not metrics["kkt_pass"]
                    else "HERTZ_TARGET_FAIL"
                    if not metrics["hertz_target_pass"]
                    else "PASS"
                )
        except Exception as exc:
            case["failures"].append(_failure(exc, phase))
            case["status"] = "ERROR"
            case["usable_discrete_solution"] = False
    case["return_code"] = int(case["status"] != "PASS")
    case["memory"] = {
        "before_case": memory_before,
        "after_case": _process_memory(),
        "saved_array_bytes_uncompressed": sum(value.nbytes for value in arrays.values()),
        "note": "OS peak is process lifetime, not isolated solver peak",
    }
    begin = time.perf_counter()
    np.savez_compressed(folder / "fields.npz", **arrays)
    case["timing"]["array_save_s"] = time.perf_counter() - begin
    case["files"] = {"arrays": f"{case_id}/fields.npz", "case": f"{case_id}/case.json"}
    # The case's own JSON write cannot be timed inside its already-written bytes.
    case["timing"]["total_case_s"] = time.perf_counter() - started
    case["timing"]["total_case_scope"] = "through NPZ save, excluding this case JSON write"
    begin = time.perf_counter()
    _write_json(folder / "case.json", case)
    case["timing"]["case_json_save_s"] = time.perf_counter() - begin
    case["timing"]["total_with_metadata_s"] = time.perf_counter() - started
    return case


def run(output):
    directory = Path(output)
    if not directory.is_absolute():
        raise ValueError("--output must be an absolute fresh directory")
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    summary = {
        "schema": SCHEMA,
        "cases": [],
        "return_code": 1,
        "qualification": "EXTERNAL_CPU_STATIC_ELASTIC_REFERENCE",
        "adapter_source": _source(__file__),
        "python": sys.version,
        "platform": platform.platform(),
        "existing_nine_cases_rerun": False,
        "reference_helper_source": _source(sys.modules[hertz_reference.__module__].__file__),
    }
    begin = time.perf_counter()
    try:
        api, binding = _load_external()
    except Exception as exc:
        failure = _failure(exc, "dependency-import")
        summary["dependency_failure"] = failure
        for delta in DELTAS_M:
            case_id = "n512-d075um" if delta == DELTAS_M[0] else "n512-d150um"
            folder = directory / case_id
            folder.mkdir()
            np.savez_compressed(folder / "fields.npz", **input_arrays(delta))
            case = {
                "id": case_id,
                "grid": GRID,
                "delta_m": delta,
                "status": "UNSUPPORTED",
                "return_code": 1,
                "metrics": None,
                "usable_discrete_solution": False,
                "config": configuration(delta),
                "timing": {"solver_s": None, "total_case_s": None},
                "files": {"case": f"{case_id}/case.json", "arrays": f"{case_id}/fields.npz"},
                "failures": [failure],
            }
            _write_json(folder / "case.json", case)
            summary["cases"].append(case)
    else:
        summary["binding"] = binding
        summary["external_import_s"] = time.perf_counter() - begin
        for delta in DELTAS_M:
            summary["cases"].append(_run_case(directory, delta, api, binding))
            _write_json(directory / "summary.json", summary)
    summary["return_code"] = int(any(case["return_code"] != 0 for case in summary["cases"]))
    summary["total_before_final_summary_save_s"] = time.perf_counter() - started
    _write_json(directory / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    summary = run(args.output)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "return_code": summary["return_code"],
                "cases": [
                    {"id": case["id"], "status": case["status"]} for case in summary["cases"]
                ],
            }
        )
    )
    return summary["return_code"]


if __name__ == "__main__":
    raise SystemExit(main())
