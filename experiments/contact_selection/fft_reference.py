"""Fixed ContactMechanics CPU references; this module contains no FFT solver.

The full physical domain carries a Hertz paraboloid. FreeFFT owns the padded
half-space Green function and its zero mode. Positive output forces are pixel
forces in N; pressure is force / pixel area. External imports are lazy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SCHEMA = "phymeo-fft-selection-v1"
GRIDS = (128, 256, 512)
DELTAS_M = (0.00005, 0.0001, 0.0002)
RADIUS_M = 0.02
CONTACT_MODULUS_PA = 1e6
DOMAIN_M = 0.16
PENTOL_M = 1e-10
FORCETOL_N = 1e-10
THERMOTOL = 1e-8
MAXITER = 10000


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def frozen_config(grid, delta):
    """Owned metadata for one preregistered case; no parameter overrides."""
    if type(grid) is not int or grid not in GRIDS:
        raise ValueError("grid must be one of 128, 256, 512")
    delta = _positive(delta, "delta")
    if delta not in DELTAS_M:
        raise ValueError("delta is not a preregistered indentation")
    return {
        "grid": grid, "delta_m": delta, "radius_m": RADIUS_M,
        "contact_modulus_pa": CONTACT_MODULUS_PA,
        "physical_sizes_m": [DOMAIN_M, DOMAIN_M],
        "computational_grid": [2 * grid, 2 * grid],
        "geometry": "full-domain negative Hertz paraboloid; small-deformation sphere approximation",
        "coordinates": "x_i=(i-n//2)*L/n; first axis x, second y; center node",
        "backend": "ContactMechanics 1.8.3 FreeFFTElasticHalfSpace, CPU serial",
        "periodic": False, "check_boundaries": True,
        "external_force": None, "hardness": None,
        "initial_displacements": None, "initial_forces": None,
        "solver": {"pentol_m": PENTOL_M, "forcetol_n": FORCETOL_N,
                   "thermotol": THERMOTOL, "maxiter": MAXITER},
        "kkt_limits": {"gap_min_m": -PENTOL_M,
                       "tensile_pixel_force_n": FORCETOL_N,
                       "active_gap_abs_m": 10 * PENTOL_M,
                       "complementarity_normalized": 1e-5,
                       "reconstructed_pixel_force_error_n": 10 * FORCETOL_N,
                       "padding_pixel_force_n": 10 * FORCETOL_N,
                       "physical_outer_ring_force_n": 0.0},
        "hertz_relative_target": 0.05, "fine_grid_target": grid == 512,
        "contact_area_rule": "count(raw positive pixel force > 0) * pixel_area",
    }


def hertz_reference(delta_m, radius_m=RADIUS_M, contact_modulus_pa=CONTACT_MODULUS_PA):
    """Analytic half-space/paraboloid oracle in SI units."""
    delta = _positive(delta_m, "delta_m")
    radius = _positive(radius_m, "radius_m")
    modulus = _positive(contact_modulus_pa, "contact_modulus_pa")
    a = math.sqrt(radius * delta)
    result = {"radius_m": a, "area_m2": math.pi * a * a,
              "force_n": (4 / 3) * modulus * math.sqrt(radius) * delta**1.5,
              "peak_pressure_pa": 2 * modulus * a / (math.pi * radius)}
    if not all(math.isfinite(v) and v > 0 for v in result.values()):
        raise ValueError("Hertz scales are not representable")
    return result


def physical_grid(grid):
    frozen_config(grid, DELTAS_M[0])
    x = (np.arange(grid, dtype=np.float64) - grid // 2) * (DOMAIN_M / grid)
    heights = -(x[:, None] ** 2 + x[None, :] ** 2) / (2 * RADIUS_M)
    return x, heights, (DOMAIN_M / grid) ** 2


def hertz_fields(x, delta):
    """Analytic pressure and displacement, including the exterior elastic tail."""
    ref = hertz_reference(delta)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not 1 <= x.size <= 512 or not np.isfinite(x).all():
        raise ValueError("x must be a finite vector of at most 512 coordinates")
    radius = np.sqrt(x[:, None] ** 2 + x[None, :] ** 2)
    a = ref["radius_m"]
    inside = radius <= a
    pressure = np.zeros(radius.shape)
    pressure[inside] = ref["peak_pressure_pa"] * np.sqrt(np.maximum(0.0, 1 - (radius[inside] / a) ** 2))
    displacement = np.empty(radius.shape)
    displacement[inside] = delta - radius[inside] ** 2 / (2 * RADIUS_M)
    r = radius[~inside]
    displacement[~inside] = ((2 * a * a - r * r) * np.arcsin(a / r)
                             + a * np.sqrt(r * r - a * a)) / (math.pi * RADIUS_M)
    return pressure, displacement


def _field(value, shape, name):
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{name} must not be masked")
    array = np.array(value, copy=True)
    if array.shape != shape or array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite floating array with shape {shape}")
    return array


def extract_fields(result, substrate, system, grid, delta):
    """Decode public external results without a solve or an elastic call."""
    frozen_config(grid, delta)
    x, heights, area = physical_grid(grid)
    if not math.isclose(float(substrate.area_per_pt), area, rel_tol=1e-14, abs_tol=0):
        raise ValueError("external pixel area differs from the frozen physical grid")
    slices = substrate.local_topography_subdomain_slices
    if not isinstance(slices, tuple) or len(slices) != 2 or any(
        not isinstance(s, slice) or s.start != 0 or s.stop != grid or s.step not in (None, 1)
        for s in slices
    ):
        raise ValueError("external serial physical-domain slices are inconsistent")
    if isinstance(result.offset, (bool, np.bool_)) or float(result.offset) != delta:
        raise ValueError("the prescribed displacement was changed")
    u_full = _field(result.x, (2 * grid, 2 * grid), "result.x")
    force = _field(result.jac, (grid, grid), "result.jac")
    gap = _field(system.compute_gap(u_full, result.offset), (grid, grid), "gap")
    own_gap = u_full[slices] - (heights + delta)
    if not np.allclose(gap, own_gap, rtol=0, atol=1e-14):
        raise ValueError("public gap does not match the saved displacement/height/offset")
    pressure_hertz, displacement_hertz = hertz_fields(x, delta)
    return {"x": x, "heights": heights, "u_full": u_full,
            "u": u_full[slices].copy(), "f": force, "p": force / area,
            "gap": gap, "contact_mask": force > 0,
            "hertz_pressure": pressure_hertz, "hertz_displacement": displacement_hertz,
            "grid": grid, "delta": delta, "pixel_area_m2": area}


def assess_fields(fields, reconstructed_force_full, *, solver_success):
    """Separate KKT from Hertz accuracy; never repair raw exported fields.

    reconstructed_force_full is POSITIVE reaction: the negative of the
    backend's evaluate_force(u_full) restoring force, including padding.
    """
    n, delta = fields["grid"], fields["delta"]
    frozen_config(n, delta)
    reaction = _field(reconstructed_force_full, (2 * n, 2 * n), "reconstructed reaction")
    f, p, gap = (fields[k] for k in ("f", "p", "gap"))
    for key in ("f", "p", "gap"):
        _field(fields[key], (n, n), key)
    active = f > 0
    reference = hertz_reference(delta)
    area = fields["pixel_area_m2"]
    padding = np.ones(reaction.shape, dtype=bool)
    padding[:n, :n] = False
    boundary = np.concatenate((f[0], f[-1], f[1:-1, 0], f[1:-1, -1]))
    kkt = {
        "solver_success": bool(solver_success),
        "minimum_pressure_pa": float(np.min(p)),
        "minimum_gap_m": float(np.min(gap)),
        "active_gap_max_abs_m": float(np.max(np.abs(gap[active]))) if np.any(active) else 0.0,
        "complementarity_abs_work_j": float(np.sum(np.abs(f * gap))),
        "reconstructed_force_max_error_n": float(np.max(np.abs(reaction[:n, :n] - f))),
        "padding_force_max_abs_n": float(np.max(np.abs(reaction[padding]))),
        "outer_ring_force_max_abs_n": float(np.max(np.abs(boundary))),
    }
    kkt["complementarity_normalized"] = kkt["complementarity_abs_work_j"] / (reference["force_n"] * delta)
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
    kkt["passed"] = all(kkt["checks"].values())
    force = float(np.sum(f))
    contact_area = int(np.count_nonzero(active)) * area
    peak = float(np.max(p))
    hertz = {"reference": reference,
             "force_relative_error": abs(force - reference["force_n"]) / reference["force_n"],
             "area_relative_error": abs(contact_area - reference["area_m2"]) / reference["area_m2"],
             "peak_pressure_relative_error": abs(peak - reference["peak_pressure_pa"]) / reference["peak_pressure_pa"],
             "center_pressure_relative_error": abs(float(p[n // 2, n // 2]) - reference["peak_pressure_pa"]) / reference["peak_pressure_pa"],
             "relative_target": 0.05, "fine_grid_target": n == 512}
    hertz["target_passed"] = all(hertz[k] <= 0.05 for k in (
        "force_relative_error", "area_relative_error", "peak_pressure_relative_error"))
    return {"force_n": force, "area_m2": contact_area, "peak_pressure_pa": peak,
            "center_pressure_pa": float(p[n // 2, n // 2]),
            "contact_pixels": int(np.count_nonzero(active)), "pixel_area_m2": area,
            "kkt": kkt, "hertz": hertz}


def preview_fields(fields):
    n = fields["grid"]
    uniform = np.linspace(0, n - 1, min(n, 128), dtype=int)
    local = np.flatnonzero(abs(fields["x"]) <= 0.004)

    def image(indices):
        return {"x_m": fields["x"][indices].tolist(),
                "y_m": fields["x"][indices].tolist(),
                "pressure_pa": fields["p"][np.ix_(indices, indices)].tolist(),
                "displacement_m": fields["u"][np.ix_(indices, indices)].tolist(),
                "gap_m": fields["gap"][np.ix_(indices, indices)].tolist(),
                "hertz_pressure_pa": fields["hertz_pressure"][np.ix_(indices, indices)].tolist(),
                "interpolated": False}

    return {"axis_order": "first x, second y; transpose for row-y screen raster",
            "contact_radius_hertz_m": hertz_reference(fields["delta"])["radius_m"],
            "full_domain": image(uniform), "contact_zoom": image(local),
            "zoom_half_width_m": 0.004,
            "center_line": {"y_m": 0.0, "x_m": fields["x"].tolist(),
                            "pressure_pa": fields["p"][:, n // 2].tolist(),
                            "displacement_m": fields["u"][:, n // 2].tolist(),
                            "gap_m": fields["gap"][:, n // 2].tolist(),
                            "hertz_pressure_pa": fields["hertz_pressure"][:, n // 2].tolist(),
                            "hertz_displacement_m": fields["hertz_displacement"][:, n // 2].tolist()}}


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    raise TypeError(f"unsupported result metadata type: {type(value).__name__}")


def _source(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _process_memory():
    """OS process observations, explicitly not an isolated solver allocation count."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counts = Counters()
            counts.cb = ctypes.sizeof(counts)
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counts), counts.cb):
                raise ctypes.WinError(ctypes.get_last_error())
            return {"working_set_bytes": int(counts.WorkingSetSize),
                    "process_lifetime_peak_working_set_bytes": int(counts.PeakWorkingSetSize)}
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"working_set_bytes": None,
                "process_lifetime_peak_working_set_bytes": int(peak if sys.platform == "darwin" else peak * 1024)}
    except Exception as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def _external_api():
    # No Newton, Warp, TYX, GPU initialization or package installation here.
    from ContactMechanics import FreeFFTElasticHalfSpace
    from ContactMechanics.Optimization import constrained_conjugate_gradients
    from ContactMechanics.Systems import NonSmoothContactSystem
    from SurfaceTopography import Topography

    if importlib.metadata.version("ContactMechanics") != "1.8.3":
        raise RuntimeError("this experiment requires exactly ContactMechanics 1.8.3")
    versions = {}
    for name in ("ContactMechanics", "SurfaceTopography", "muGrid", "NuMPI", "numpy", "scipy", "newton", "warp-lang"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = {name: _source(inspect.getsourcefile(obj)) for name, obj in (
        ("FreeFFTElasticHalfSpace", FreeFFTElasticHalfSpace),
        ("NonSmoothContactSystem", NonSmoothContactSystem),
        ("constrained_conjugate_gradients", constrained_conjugate_gradients),
        ("Topography", Topography))}
    return (Topography, FreeFFTElasticHalfSpace, NonSmoothContactSystem), {"versions": versions, "sources": sources}


def _write_json(path, value):
    path.write_text(json.dumps(_json_value(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _run_case(directory, grid, delta, api, binding):
    started = time.perf_counter()
    case = {"id": f"n{grid}-d{round(delta * 1e6):03d}um", "grid": grid,
            "delta": delta, "status": "ERROR", "config": frozen_config(grid, delta),
            "metrics": None, "timing": {}, "files": {}, "binding": binding}
    arrays, system, result = {}, None, None
    memory_before = _process_memory()
    folder = directory / case["id"]
    folder.mkdir()
    try:
        Topography, HalfSpace, ContactSystem = api
        x, heights, _ = physical_grid(grid)
        arrays.update(x=x, heights=heights)
        start = time.perf_counter()
        surface = Topography(heights, (DOMAIN_M, DOMAIN_M), periodic=False, unit="m")
        case["timing"]["topography_creation_s"] = time.perf_counter() - start
        start = time.perf_counter()
        substrate = HalfSpace((grid, grid), young=CONTACT_MODULUS_PA,
                              physical_sizes=(DOMAIN_M, DOMAIN_M), fft="serial",
                              check_boundaries=True)
        case["timing"]["substrate_creation_including_first_fft_s"] = time.perf_counter() - start
        system = ContactSystem(substrate, surface)
        start = time.perf_counter()
        try:
            result = system.minimize_proxy(offset=delta, external_force=None, hardness=None,
                                           pentol=PENTOL_M, forcetol=FORCETOL_N,
                                           thermotol=THERMOTOL, maxiter=MAXITER)
        finally:
            case["timing"]["solver_s"] = time.perf_counter() - start
        metadata = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                arrays[f"result_{key}"] = value.copy()
                metadata[key] = {"npz_key": f"result_{key}", "shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                metadata[key] = _json_value(value)
        case["optimizer"] = metadata
        case["counters"] = {"solver_iterations": int(result.nit),
                            "solver_function_evaluations": int(result.nfev),
                            "extra_elastic_force_evaluations": 0}
        start = time.perf_counter()
        fields = extract_fields(result, substrate, system, grid, delta)
        arrays.update({k: v for k, v in fields.items() if isinstance(v, np.ndarray)})
        case["timing"]["extract_s"] = time.perf_counter() - start
        start = time.perf_counter()
        # Exactly one extra elastic application for the saved same-run KKT check.
        reaction = -substrate.evaluate_force(fields["u_full"])
        case["counters"]["extra_elastic_force_evaluations"] = 1
        case["timing"]["consistency_force_s"] = time.perf_counter() - start
        arrays["reconstructed_reaction_full"] = np.array(reaction, copy=True)
        start = time.perf_counter()
        case["metrics"] = assess_fields(fields, reaction, solver_success=result.success)
        case["preview"] = preview_fields(fields)
        case["timing"]["assessment_and_preview_s"] = time.perf_counter() - start
        if not result.success:
            case["status"] = "SOLVER_FAIL"
        elif not case["metrics"]["kkt"]["passed"]:
            case["status"] = "KKT_FAIL"
        elif not case["metrics"]["hertz"]["target_passed"]:
            case["status"] = "HERTZ_TARGET_FAIL"
        else:
            case["status"] = "PASS"
    except Exception as exc:
        case["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        # minimize_proxy publishes these before checking the free boundary. Keep
        # them if that check raises; do not invent unavailable result metadata.
        if result is None and system is not None:
            for name in ("disp", "force"):
                value = getattr(system, name, None)
                if isinstance(value, np.ndarray):
                    arrays[f"system_{name}_after_exception"] = value.copy()
    case["memory"] = {"saved_array_bytes_uncompressed": sum(v.nbytes for v in arrays.values()),
                      "padded_float64_array_bytes": 8 * (2 * grid) ** 2,
                      "before_case": memory_before, "after_case_before_save": _process_memory(),
                      "note": "OS peak is process lifetime, not reset between cases. Saved-array bytes are not solver peak working memory."}
    start = time.perf_counter()
    if arrays:
        np.savez_compressed(folder / "fields.npz", **arrays)
        case["files"]["arrays"] = f"{case['id']}/fields.npz"
    case["timing"]["array_save_s"] = time.perf_counter() - start
    case["files"]["case"] = f"{case['id']}/case.json"
    case["timing"]["case_before_metadata_save_s"] = time.perf_counter() - started
    start = time.perf_counter()
    _write_json(folder / "case.json", case)
    case["timing"]["metadata_save_s"] = time.perf_counter() - start
    case["timing"]["case_total_s"] = time.perf_counter() - started
    return case


def run(output):
    """Run each frozen case once and preserve all outcomes in a new directory."""
    directory = Path(output)
    if not directory.is_absolute():
        raise ValueError("--output must be an absolute new directory")
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    summary = {"schema": SCHEMA, "qualification": "EXTERNAL_CPU_STATIC_ELASTIC_REFERENCE",
               "cases": [], "python": sys.version, "platform": platform.platform(),
               "adapter_source": _source(__file__),
               "timing_note": "CPU wall time; no reruns for timing. Constructor includes FFT kernel construction; postprocessing/saving separate.",
               "limits": "Hertz paraboloid, elastic half-space; no finite thickness, adhesion, large deformation or dynamics."}
    import_start = time.perf_counter()
    try:
        api, binding = _external_api()
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        summary["dependency_failure"] = failure
        for delta in DELTAS_M:
            for grid in GRIDS:
                summary["cases"].append({"id": f"n{grid}-d{round(delta * 1e6):03d}um",
                                         "grid": grid, "delta": delta, "status": "UNSUPPORTED",
                                         "config": frozen_config(grid, delta), "metrics": None,
                                         "timing": {}, "files": {}, "failure": failure})
    else:
        summary["binding"] = binding
        summary["external_import_s"] = time.perf_counter() - import_start
        for delta in DELTAS_M:
            for grid in GRIDS:
                case = _run_case(directory, grid, delta, api, binding)
                summary["cases"].append(case)
                _write_json(directory / "summary.json", summary)
    summary["total_before_final_summary_save_s"] = time.perf_counter() - started
    _write_json(directory / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    summary = run(args.output)
    print(json.dumps({"schema": SCHEMA, "cases": [{"id": c["id"], "status": c["status"]} for c in summary["cases"]]}))
    # A coarse-grid Hertz miss is data, not a hidden solver failure. Fine-grid
    # accuracy and all solver/KKT failures remain an explicit nonzero outcome.
    return int(any(c["status"] in ("ERROR", "UNSUPPORTED", "SOLVER_FAIL", "KKT_FAIL")
                   or (c["grid"] == 512 and c["status"] != "PASS") for c in summary["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
