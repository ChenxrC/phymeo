"""Frozen, smooth two-peak topography against an external elastic half-space.

This is a static, frictionless CPU reference, not a contact dynamics solver.
Twelve cold public ContactMechanics solves are the entire numerical budget.
Grid agreement is reported per output, not presented as exact physical truth.
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

from experiments.contact_selection.calibration_reference import json_value, optimizer_metadata
from experiments.contact_selection.fft_reference import (
    _external_api,
    _field,
    _process_memory,
    _source,
)

SCHEMA = "phymeo-multiasperity-reference-v1"
GRIDS = (256, 512)
DELTAS_M = (20e-6, 40e-6, 60e-6, 80e-6, 100e-6)
MODES = ("coupled", "isolated_1", "isolated_2")
DOMAIN_M = 0.024
MODULUS_PA = 1e6
PENTOL_M = 1e-10
FORCETOL_N = 1e-10
THERMOTOL = 1e-8
MAXITER = 10000
GRID_LIMITS = {
    "force_relative": 0.02,
    "area_relative": 0.08,
    "peak_relative": 0.05,
    "cop_distance_m": 50e-6,
}


def _grid_mode(grid, mode):
    if type(grid) is not int or grid not in GRIDS or not isinstance(mode, str) or mode not in MODES:
        raise ValueError("unregistered grid or topography mode")


def configuration(grid, delta_m, mode="coupled"):
    _grid_mode(grid, mode)
    if (
        isinstance(delta_m, (bool, np.bool_))
        or not isinstance(delta_m, (int, float, np.integer, np.floating))
        or float(delta_m) not in DELTAS_M
    ):
        raise ValueError("indentation must be one of the five registered SI values")
    delta_m = float(delta_m)
    if mode != "coupled" and (grid != 512 or delta_m != 100e-6):
        raise ValueError("isolated peaks are registered only at n512, 100 micrometres")
    return {
        "grid": grid,
        "delta_m": delta_m,
        "mode": mode,
        "physical_sizes_m": [DOMAIN_M, DOMAIN_M],
        "contact_modulus_pa": MODULUS_PA,
        "computational_grid": [2 * grid, 2 * grid],
        "padding_factor": 2,
        "periodic": False,
        "check_boundaries": True,
        "baseline_m": -200e-6,
        "geometry": "C2 compact two-peak height field; no maximum normalization",
        "peaks": [
            {
                "center_m": [-0.003, -0.000375],
                "semiaxes_m": [0.003, 0.002],
                "height_term_m": 200e-6,
            },
            {
                "center_m": [0.003, 0.000375],
                "semiaxes_m": [0.0025, 0.0035],
                "height_term_m": 160e-6,
            },
        ],
        "profile": "max(1-r_squared,0)^3; isolated mode zeros only the other height term",
        "coordinates": "x_i=(i-n//2)*L/n; first array axis x, second y",
        "wrench_convention": "positive Fz; origin fixed at (0,0,0); Mx=sum(y*f), My=-sum(x*f)",
        "solver": {
            "pentol_m": PENTOL_M,
            "forcetol_n": FORCETOL_N,
            "thermotol": THERMOTOL,
            "maxiter": MAXITER,
            "external_force": None,
            "hardness": None,
            "initial_displacements": None,
            "initial_forces": None,
        },
        "kkt_limits": {
            "minimum_gap_m": -PENTOL_M,
            "tensile_pixel_force_n": FORCETOL_N,
            "active_gap_max_abs_m": 10 * PENTOL_M,
            "normalized_complementarity": 1e-5,
            "normalization": "sum(abs(f*gap))/(raw_total_force*delta), total_force>0",
            "reconstructed_force_error_n": 10 * FORCETOL_N,
            "padding_force_n": 10 * FORCETOL_N,
            "physical_boundary_force_n": 0.0,
        },
        "area_rule": "count(raw f > 0)*dA; support membership uses r_squared<1",
    }


def case_specs():
    return tuple((n, delta, "coupled") for delta in DELTAS_M for n in GRIDS) + (
        (512, 100e-6, "isolated_1"),
        (512, 100e-6, "isolated_2"),
    )


def case_id(grid, delta_m, mode="coupled"):
    configuration(grid, delta_m, mode)
    prefix = {"coupled": "c", "isolated_1": "i1", "isolated_2": "i2"}[mode]
    return f"{prefix}-n{grid}-d{round(delta_m * 1e6):03d}um"


def geometry(grid, mode="coupled"):
    _grid_mode(grid, mode)
    x = (np.arange(grid, dtype=np.float64) - grid // 2) * (DOMAIN_M / grid)
    y = x.copy()
    r1 = ((x[:, None] + 0.003) / 0.003) ** 2 + ((y[None, :] + 0.000375) / 0.002) ** 2
    r2 = ((x[:, None] - 0.003) / 0.0025) ** 2 + ((y[None, :] - 0.000375) / 0.0035) ** 2
    term1 = 200e-6 * np.maximum(1 - r1, 0) ** 3
    term2 = 160e-6 * np.maximum(1 - r2, 0) ** 3
    if mode == "isolated_1":
        term2 = np.zeros_like(term2)
    elif mode == "isolated_2":
        term1 = np.zeros_like(term1)
    return {
        "x_m": x,
        "y_m": y,
        "heights": term1 + term2 - 200e-6,
        "peak1_term": term1,
        "peak2_term": term2,
        "r1_squared": r1,
        "r2_squared": r2,
        "support1": r1 < 1,
        "support2": r2 < 1,
    }


def integrate_loads(force_n, x_m, y_m, pixel_area_m2, support1, support2):
    """Integrate raw pixel forces at their positions; never average two CoPs."""
    x, y = (np.asarray(value) for value in (x_m, y_m))
    for coordinate in (x, y):
        if (
            coordinate.ndim != 1
            or not 2 <= coordinate.size <= 512
            or coordinate.dtype.kind != "f"
            or not np.isfinite(coordinate).all()
            or not np.all(np.diff(coordinate) > 0)
            or not np.allclose(
                np.diff(coordinate), coordinate[1] - coordinate[0], rtol=1e-12, atol=0
            )
        ):
            raise ValueError("coordinates must be finite, increasing, uniform metre vectors")
    if (
        isinstance(pixel_area_m2, (bool, np.bool_))
        or not math.isfinite(float(pixel_area_m2))
        or pixel_area_m2 <= 0
    ):
        raise ValueError("pixel area must be finite positive square metres")
    area = float(pixel_area_m2)
    if not math.isclose(area, float((x[1] - x[0]) * (y[1] - y[0])), rel_tol=1e-12, abs_tol=0):
        raise ValueError("pixel area disagrees with the metre grid")
    shape = (x.size, y.size)
    force = _field(force_n, shape, "pixel force in newtons")
    masks = tuple(np.asarray(mask) for mask in (support1, support2))
    if any(mask.shape != shape or mask.dtype.kind != "b" for mask in masks) or np.any(
        masks[0] & masks[1]
    ):
        raise ValueError("peak supports must be disjoint boolean masks on the force grid")
    total = float(np.sum(force))
    mx = float(np.sum(force * y[None, :]))
    my = -float(np.sum(force * x[:, None]))
    pressure_peak = float(np.max(force) / area)
    if not all(math.isfinite(v) for v in (total, mx, my, pressure_peak)):
        raise ValueError("integrated load is not representable")
    active = force > 0
    outside = ~(masks[0] | masks[1])
    per_peak = [
        {
            "peak": i + 1,
            "force_n": float(np.sum(force[mask])),
            "area_m2": int(np.count_nonzero(active & mask)) * area,
            "contact_pixels": int(np.count_nonzero(active & mask)),
        }
        for i, mask in enumerate(masks)
    ]
    return {
        "force_n": total,
        "area_m2": int(np.count_nonzero(active)) * area,
        "peak_pressure_pa": pressure_peak,
        "cop_xy_m": [-my / total, mx / total] if total > 0 else None,
        "wrench_world_origin": [0.0, 0.0, total, mx, my, 0.0],
        "contact_pixels": int(np.count_nonzero(active)),
        "pixel_area_m2": area,
        "per_peak": per_peak,
        "outside_support": {
            "force_n": float(np.sum(force[outside])),
            "absolute_force_n": float(np.sum(np.abs(force[outside]))),
            "area_m2": int(np.count_nonzero(active & outside)) * area,
            "contact_pixels": int(np.count_nonzero(active & outside)),
        },
    }


def extract_fields(result, substrate, system, grid, delta_m, mode="coupled"):
    configuration(grid, delta_m, mode)
    inputs = geometry(grid, mode)
    area = (DOMAIN_M / grid) ** 2
    physical_sizes = np.asarray(substrate.physical_sizes)
    if (
        tuple(substrate.nb_grid_pts) != (grid, grid)
        or tuple(substrate.nb_domain_grid_pts) != (2 * grid, 2 * grid)
        or physical_sizes.shape != (2,)
        or not np.allclose(physical_sizes, (DOMAIN_M, DOMAIN_M), rtol=1e-14, atol=0)
        or not math.isclose(float(substrate.area_per_pt), area, rel_tol=1e-14, abs_tol=0)
    ):
        raise ValueError("external domain, grid or pixel area differs from the frozen SI inputs")
    slices = substrate.local_topography_subdomain_slices
    if (
        not isinstance(slices, tuple)
        or len(slices) != 2
        or any(
            not isinstance(s, slice) or s.start != 0 or s.stop != grid or s.step not in (None, 1)
            for s in slices
        )
    ):
        raise ValueError("invalid physical-domain slices")
    if isinstance(result.offset, (bool, np.bool_)) or float(result.offset) != delta_m:
        raise ValueError("prescribed displacement changed")
    u_full = _field(result.x, (2 * grid, 2 * grid), "result.x")
    force = _field(result.jac, (grid, grid), "result.jac")
    gap = u_full[slices] - (inputs["heights"] + delta_m)
    public_gap = _field(system.compute_gap(u_full, result.offset), (grid, grid), "public gap")
    if not np.allclose(public_gap, gap, rtol=0, atol=1e-14):
        raise ValueError("public gap differs from the frozen height and saved state")
    return {
        **inputs,
        "u_full": u_full,
        "u": u_full[slices].copy(),
        "f": force,
        "p": force / area,
        "gap": gap,
        "public_gap": public_gap,
        "contact_mask": force > 0,
        "grid": grid,
        "delta_m": delta_m,
        "mode": mode,
    }


def assess_fields(fields, reaction_full, *, solver_success):
    n, delta, mode = fields["grid"], fields["delta_m"], fields["mode"]
    configuration(n, delta, mode)
    force, pressure, gap = (_field(fields[k], (n, n), k) for k in ("f", "p", "gap"))
    reaction = _field(reaction_full, (2 * n, 2 * n), "positive reconstructed reaction")
    area = (DOMAIN_M / n) ** 2
    if not np.allclose(pressure, force / area, rtol=1e-14, atol=0):
        raise ValueError("pressure is not raw pixel force divided by the SI pixel area")
    metrics = integrate_loads(
        force, fields["x_m"], fields["y_m"], area, fields["support1"], fields["support2"]
    )
    active = force > 0
    pad = np.ones(reaction.shape, dtype=bool)
    pad[:n, :n] = False
    boundary = np.concatenate((force[0], force[-1], force[1:-1, 0], force[1:-1, -1]))
    work = float(np.sum(np.abs(force * gap)))
    denominator = metrics["force_n"] * delta
    normalized = work / denominator if denominator > 0 and math.isfinite(denominator) else None
    kkt = {
        "minimum_pressure_pa": float(np.min(pressure)),
        "minimum_gap_m": float(np.min(gap)),
        "active_gap_max_abs_m": float(np.max(np.abs(gap[active]))) if np.any(active) else 0.0,
        "complementarity_abs_work_j": work,
        "complementarity_normalized": normalized,
        "normalization_force_n": metrics["force_n"],
        "normalization_delta_m": delta,
        "reconstructed_force_max_error_n": float(np.max(np.abs(reaction[:n, :n] - force))),
        "padding_force_max_abs_n": float(np.max(np.abs(reaction[pad]))),
        "outer_ring_force_max_abs_n": float(np.max(np.abs(boundary))),
    }
    kkt["checks"] = {
        "optimizer": bool(solver_success),
        "positive_total_force": metrics["force_n"] > 0,
        "pressure": kkt["minimum_pressure_pa"] >= -FORCETOL_N / area,
        "nonpenetration": kkt["minimum_gap_m"] >= -PENTOL_M,
        "active_gap": kkt["active_gap_max_abs_m"] <= 10 * PENTOL_M,
        "complementarity": normalized is not None
        and math.isfinite(normalized)
        and normalized <= 1e-5,
        "elastic_consistency": kkt["reconstructed_force_max_error_n"] <= 10 * FORCETOL_N,
        "zero_padding_traction": kkt["padding_force_max_abs_n"] <= 10 * FORCETOL_N,
        "free_physical_boundary": kkt["outer_ring_force_max_abs_n"] == 0.0,
    }
    return {**metrics, "kkt": kkt, "kkt_pass": all(kkt["checks"].values())}


def _bound_case(case, grid, delta, mode):
    if (
        case["grid"] != grid
        or case["delta_m"] != delta
        or case["mode"] != mode
        or case["config"] != configuration(grid, delta, mode)
    ):
        raise ValueError("comparison cases have mismatched physical bindings")


def _usable(case):
    return (
        case["status"] == "KKT_PASS"
        and case["usable_discrete_solution"] is True
        and case["metrics"] is not None
        and case["metrics"]["kkt_pass"] is True
    )


def compare_grids(coarse, fine):
    delta = fine["delta_m"]
    _bound_case(coarse, 256, delta, "coupled")
    _bound_case(fine, 512, delta, "coupled")
    base = {
        "delta_m": delta,
        "coarse_id": coarse["id"],
        "fine_id": fine["id"],
        "limits": dict(GRID_LIMITS),
        "denominator": "n512",
        "exact_truth": False,
    }
    if not (_usable(coarse) and _usable(fine)):
        return {**base, "status": "UNAVAILABLE", "errors": None, "checks": None, "all_pass": False}
    a, b = coarse["metrics"], fine["metrics"]
    errors = {}
    for name, key in (
        ("force_relative", "force_n"),
        ("area_relative", "area_m2"),
        ("peak_relative", "peak_pressure_pa"),
    ):
        if not all(math.isfinite(float(value)) and value > 0 for value in (a[key], b[key])):
            raise ValueError("grid comparison requires finite positive measured outputs")
        errors[name] = abs(a[key] - b[key]) / b[key]
    ca, cb = (np.asarray(m["cop_xy_m"], dtype=np.float64) for m in (a, b))
    if (
        ca.shape != (2,)
        or cb.shape != (2,)
        or not np.isfinite(ca).all()
        or not np.isfinite(cb).all()
    ):
        raise ValueError("grid comparison requires finite two-dimensional CoPs")
    errors["cop_distance_m"] = float(np.linalg.norm(ca - cb))
    checks = {key: value <= GRID_LIMITS[key] for key, value in errors.items()}
    all_pass = all(checks.values())
    return {
        **base,
        "status": "PASS" if all_pass else "GRID_LIMIT_FAIL",
        "errors": errors,
        "checks": checks,
        "all_pass": all_pass,
    }


def compare_coupling(joint, isolated1, isolated2):
    for case, mode in ((joint, "coupled"), (isolated1, "isolated_1"), (isolated2, "isolated_2")):
        _bound_case(case, 512, 100e-6, mode)
    base = {
        "joint_id": joint["id"],
        "isolated_ids": [isolated1["id"], isolated2["id"]],
        "denominator": "joint n512 d100um",
        "is_accuracy_gate": False,
    }
    if not all(_usable(case) for case in (joint, isolated1, isolated2)):
        return {**base, "status": "UNAVAILABLE"}
    wj, w1, w2 = (
        np.asarray(case["metrics"]["wrench_world_origin"], dtype=np.float64)
        for case in (joint, isolated1, isolated2)
    )
    if any(w.shape != (6,) or not np.isfinite(w).all() or w[2] <= 0 for w in (wj, w1, w2)):
        raise ValueError("coupling comparison requires finite wrenches with positive Fz")
    summed = w1 + w2
    cop_sum = np.array([-summed[4], summed[3]]) / summed[2]
    cop_joint = np.array([-wj[4], wj[3]]) / wj[2]
    difference = summed - wj
    moment_norm = float(np.linalg.norm(wj[3:]))
    return {
        **base,
        "status": "MEASURED",
        "joint_wrench_world_origin": wj.tolist(),
        "isolated_sum_wrench_world_origin": summed.tolist(),
        "wrench_difference": difference.tolist(),
        "joint_cop_xy_m": cop_joint.tolist(),
        "isolated_sum_cop_xy_m": cop_sum.tolist(),
        "force_absolute_error_n": abs(float(difference[2])),
        "force_relative_error": abs(float(difference[2])) / wj[2],
        "cop_distance_m": float(np.linalg.norm(cop_sum - cop_joint)),
        "moment_absolute_error_nm": np.abs(difference[3:]).tolist(),
        "moment_norm_error_nm": float(np.linalg.norm(difference[3:])),
        "moment_relative_components": [
            abs(float(difference[i])) / abs(float(wj[i])) if wj[i] != 0 else None
            for i in range(3, 6)
        ],
        "moment_relative_norm": float(np.linalg.norm(difference[3:])) / moment_norm
        if moment_norm > 0
        else None,
    }


def preview_fields(fields):
    n = fields["grid"]
    uniform = np.linspace(0, n - 1, min(n, 128), dtype=int)
    local = np.flatnonzero(np.abs(fields["x_m"]) <= 0.007)

    def image(indices):
        return {
            "x_m": fields["x_m"][indices].tolist(),
            "y_m": fields["y_m"][indices].tolist(),
            **{
                name: fields[key][np.ix_(indices, indices)].tolist()
                for name, key in (
                    ("height_m", "heights"),
                    ("pressure_pa", "p"),
                    ("displacement_m", "u"),
                    ("gap_m", "gap"),
                )
            },
            "interpolated": False,
        }

    lines = []
    for y in (-0.000375, 0.0, 0.000375):
        index = int(round(y * n / DOMAIN_M)) + n // 2
        lines.append(
            {
                "y_m": float(fields["y_m"][index]),
                "x_m": fields["x_m"].tolist(),
                **{
                    name: fields[key][:, index].tolist()
                    for name, key in (
                        ("height_m", "heights"),
                        ("pressure_pa", "p"),
                        ("displacement_m", "u"),
                        ("gap_m", "gap"),
                    )
                },
            }
        )
    return {
        "axis_order": "first x, second y; transpose for row-y raster",
        "full_domain": image(uniform),
        "peak_zoom": image(local),
        "zoom_half_width_m": 0.007,
        "native_lines": lines,
    }


def _load_external():
    api, binding = _external_api()
    from ContactMechanics.Optimization import constrained_conjugate_gradients

    return (*api, constrained_conjugate_gradients), binding


def _write_json(path, value):
    path.write_text(
        json.dumps(json_value(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _failure(exc, phase):
    return {
        "phase": phase,
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _run_case(directory, grid, delta, mode, api, binding, dependency_failure=None):
    started = time.perf_counter()
    identity = case_id(grid, delta, mode)
    case = {
        "id": identity,
        "grid": grid,
        "delta_m": delta,
        "mode": mode,
        "config": configuration(grid, delta, mode),
        "status": "ERROR",
        "return_code": 1,
        "usable_discrete_solution": False,
        "metrics": None,
        "optimizer": None,
        "timing": {"solver_s": None},
        "files": {},
        "binding": binding,
        "failures": [],
        "counters": {"solve_calls": 0, "extra_elastic_force_evaluations": 0},
    }
    folder = directory / identity
    folder.mkdir()
    arrays = {**geometry(grid, mode), "delta_m": np.array(delta, dtype=np.float64)}
    raw_result = None
    solve_returned = False
    system = substrate = None
    memory_before = _process_memory()
    phase = "construction"
    if dependency_failure is not None:
        case["status"] = "UNSUPPORTED"
        case["failures"].append(dependency_failure)
    else:
        try:
            Topography, HalfSpace, ContactSystem, public_solver = api
            begin = time.perf_counter()
            surface = Topography(arrays["heights"], (DOMAIN_M, DOMAIN_M), periodic=False, unit="m")
            substrate = HalfSpace(
                (grid, grid),
                young=MODULUS_PA,
                physical_sizes=(DOMAIN_M, DOMAIN_M),
                fft="serial",
                check_boundaries=True,
            )
            system = ContactSystem(substrate, surface)
            case["timing"]["construction_including_first_fft_s"] = time.perf_counter() - begin

            def retain_result(*args, **kwargs):
                nonlocal raw_result
                case["counters"]["solve_calls"] += 1
                start = time.perf_counter()
                try:
                    raw_result = public_solver(*args, **kwargs)
                finally:
                    case["timing"]["optimizer_s"] = time.perf_counter() - start
                start = time.perf_counter()
                for key, value in raw_result.items():
                    if isinstance(value, np.ndarray):
                        arrays[f"result_{key}"] = value.copy()
                case["timing"]["raw_result_copy_s"] = time.perf_counter() - start
                return raw_result

            phase = "solve"
            begin = time.perf_counter()
            try:
                system.minimize_proxy(
                    solver=retain_result,
                    offset=delta,
                    external_force=None,
                    hardness=None,
                    initial_displacements=None,
                    initial_forces=None,
                    pentol=PENTOL_M,
                    forcetol=FORCETOL_N,
                    thermotol=THERMOTOL,
                    maxiter=MAXITER,
                )
                if raw_result is None:
                    raise ValueError("public system did not invoke the registered solver")
                solve_returned = True
            finally:
                case["timing"]["solver_s"] = time.perf_counter() - begin
        except Exception as exc:
            case["failures"].append(_failure(exc, phase))
    if raw_result is not None:
        phase = "optimizer-export"
        try:
            case["optimizer"] = optimizer_metadata(raw_result, arrays)
            case["counters"].update(
                solver_iterations=int(raw_result.nit),
                solver_function_evaluations=int(raw_result.nfev),
            )
            phase = "field-export"
            fields = extract_fields(raw_result, substrate, system, grid, delta, mode)
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
                case["usable_discrete_solution"] = case["metrics"]["kkt_pass"]
                case["status"] = (
                    "SOLVER_FAIL"
                    if not raw_result.success
                    else "KKT_PASS"
                    if case["metrics"]["kkt_pass"]
                    else "KKT_FAIL"
                )
        except Exception as exc:
            case["failures"].append(_failure(exc, phase))
            case["status"] = "ERROR"
            case["usable_discrete_solution"] = False
    case["return_code"] = int(case["status"] != "KKT_PASS")
    case["memory"] = {
        "before_case": memory_before,
        "after_case": _process_memory(),
        "saved_array_bytes_uncompressed": sum(v.nbytes for v in arrays.values()),
        "note": "OS peak is process lifetime, not an isolated solver allocation peak",
    }
    begin = time.perf_counter()
    np.savez_compressed(folder / "fields.npz", **arrays)
    case["timing"]["array_save_s"] = time.perf_counter() - begin
    case["files"] = {"arrays": f"{identity}/fields.npz", "case": f"{identity}/case.json"}
    case["timing"]["total_case_s"] = time.perf_counter() - started
    case["timing"]["total_case_scope"] = "through NPZ save, excluding own JSON write"
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
        "status": "INCOMPLETE",
        "return_code": 1,
        "qualification": "EXTERNAL_CPU_STATIC_ELASTIC_REFERENCE",
        "maximum_solve_calls": 12,
        "old_cases_rerun": False,
        "adapter_source": _source(__file__),
        "python": sys.version,
        "platform": platform.platform(),
        "helper_sources": [
            _source(sys.modules[f.__module__].__file__) for f in (_external_api, json_value)
        ],
        "comparison_note": "two-grid output agreement, not an exact solution or rigorous error bound",
    }
    api = binding = dependency_failure = None
    begin = time.perf_counter()
    try:
        api, binding = _load_external()
    except Exception as exc:
        dependency_failure = _failure(exc, "dependency-import")
        summary["dependency_failure"] = dependency_failure
    summary["external_import_s"] = time.perf_counter() - begin
    summary["binding"] = binding
    for n, delta, mode in case_specs():
        case = _run_case(directory, n, delta, mode, api, binding, dependency_failure)
        # Full native-derived previews remain in case.json; do not rewrite large
        # images into the cumulative summary after every new solve.
        summary["cases"].append({key: value for key, value in case.items() if key != "preview"})
        _write_json(directory / "summary.json", summary)
    indexed = {case["id"]: case for case in summary["cases"]}
    summary["grid_comparisons"] = [
        compare_grids(indexed[case_id(256, d)], indexed[case_id(512, d)]) for d in DELTAS_M
    ]
    summary["coupling_comparison"] = compare_coupling(
        indexed[case_id(512, 100e-6)],
        indexed[case_id(512, 100e-6, "isolated_1")],
        indexed[case_id(512, 100e-6, "isolated_2")],
    )
    passed = all(case["return_code"] == 0 for case in summary["cases"]) and all(
        c["all_pass"] for c in summary["grid_comparisons"]
    )
    summary["return_code"] = int(not passed)
    summary["status"] = "PASS" if passed else "FAIL_RECORDED"
    summary["actual_solve_calls"] = sum(
        case["counters"]["solve_calls"] for case in summary["cases"]
    )
    summary["total_before_final_summary_save_s"] = time.perf_counter() - started
    _write_json(directory / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    summary = run(parser.parse_args(argv).output)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": summary["status"],
                "return_code": summary["return_code"],
                "cases": [{"id": c["id"], "status": c["status"]} for c in summary["cases"]],
            }
        )
    )
    return summary["return_code"]


if __name__ == "__main__":
    raise SystemExit(main())
