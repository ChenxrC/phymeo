"""One installed-wheel CUDA solve of the accepted I3u 128-point input.

Copy this file outside the checkout and run Python -I. Existing dependencies
remain in use; all Phymeo code must come from --site. This verifies a standalone
frozen-J backend, not automatic scene friction or native contact integration.
"""

import argparse
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
import traceback
import zipfile
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from time import perf_counter

LAW_NAME = "final_normal_smooth_v1"
RESULT_SHAPES = {
    "velocity": (6,),
    "free_velocity": (6,),
    "normal_force": (128,),
    "tangential_force": (128, 3),
    "force_on_a": (128, 3),
    "force_on_b": (128, 3),
    "torque_on_a_world_origin": (128, 3),
    "torque_on_b_world_origin": (128, 3),
    "generalized_force": (6,),
    "residual": (6,),
    "residual_jacobian": (6, 6),
    "contact_data": (128, 17),
    "response_matrix": (6, 6),
}
PROBLEM_FIELDS = {"mass_matrix", "damping", "input_velocity", "applied_force", "h_s", "samples"}
SAMPLE_FIELDS = {
    "point_world",
    "normal_world",
    "jacobian_a",
    "jacobian_b",
    "stiffness_n_m",
    "phi_m",
    "mu",
    "slip_speed_m_s",
}
MAX_INPUT_BYTES = 64 * 1024 * 1024
np = None  # No third-party/package imports before the isolation checks.


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _under(path, root):
    return path.resolve().is_relative_to(root.resolve())


def _checkout(path):
    return (path / "pyproject.toml").is_file() and (path / "src" / "phymeo").is_dir()


def _guard(site, input_path, output, forbidden):
    _require(
        site.is_absolute() and site.is_dir(),
        "--site must be an existing absolute installation target",
    )
    _require(
        input_path.is_absolute() and input_path.is_file(),
        "--input must be an existing absolute saved gzip",
    )
    _require(
        output.is_absolute() and not output.exists(),
        "--output must be an absent absolute directory",
    )
    _require(all(p.is_absolute() for p in forbidden), "--forbid-root must be absolute")
    _require(
        sys.flags.isolated == 1 and not os.environ.get("PYTHONPATH"),
        "use Python -I with PYTHONPATH cleared",
    )
    for path in (
        site.resolve(),
        output.resolve(),
        Path.cwd().resolve(),
        Path(__file__).resolve().parent,
    ):
        _require(
            not any(_checkout(p) for p in (path, *path.parents)),
            "consumer path is inside a Phymeo checkout",
        )
        _require(
            not any(_under(path, root) for root in forbidden),
            "consumer path is under --forbid-root",
        )
    _require(
        not any(n == "phymeo" or n.startswith("phymeo.") for n in sys.modules),
        "Phymeo was preloaded",
    )
    _require((site / "phymeo" / "__init__.py").is_file(), "target lacks installed Phymeo")
    before, pth = tuple(sys.path), {}
    for entry in before:
        path = Path(entry or os.getcwd()).resolve()
        _require(
            not _checkout(path) and not (_checkout(path.parent) and path.name == "src"),
            "source import root on sys.path",
        )
        # Dependencies may legitimately reside in a venv under a forbidden
        # checkout. Only a source/package candidate there is forbidden.
        _require(
            not any(
                path == root.resolve() or path == (root / "src").resolve() for root in forbidden
            ),
            "forbidden source root on sys.path",
        )
        candidate = (path / "phymeo").exists() or (path / "phymeo.py").exists()
        if path.is_file() and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                candidate = any(
                    n == "phymeo.py" or n.startswith("phymeo/") for n in archive.namelist()
                )
        _require(not candidate or path == site.resolve(), "another Phymeo candidate: " + str(path))
        if path.is_dir():
            for file in sorted(path.glob("*.pth")):
                pth[str(file)] = dict(
                    sha256=_sha(file), text=file.read_text(encoding="utf-8", errors="replace")
                )
    sys.path.insert(0, str(site.resolve()))
    spec = importlib.util.find_spec("phymeo")
    _require(
        spec is not None and spec.origin is not None and _under(Path(spec.origin), site),
        "Phymeo import spec outside target",
    )
    return dict(
        site=str(site.resolve()),
        input=str(input_path.resolve()),
        output=str(output.resolve()),
        forbidden_roots=tuple(map(str, forbidden)),
        executable=sys.executable,
        python=sys.version,
        isolated=sys.flags.isolated,
        cwd=str(Path.cwd()),
        script=str(Path(__file__).resolve()),
        script_sha256=_sha(Path(__file__)),
        sys_path_before=before,
        sys_path_after=tuple(sys.path),
        dependency_pth=pth,
    )


def _inventory(site, forbidden):
    modules = {}
    for name, module in sorted(tuple(sys.modules.items())):
        if name == "phymeo" or name.startswith("phymeo."):
            origin = getattr(module, "__file__", None)
            _require(
                origin is not None and Path(origin).is_file(),
                "missing loaded module origin: " + name,
            )
            path = Path(origin).resolve()
            _require(
                _under(path, site) and not any(_under(path, root) for root in forbidden),
                "loaded module outside target: " + name,
            )
            modules[name] = dict(path=str(path), sha256=_sha(path))
    _require(bool(modules), "no installed Phymeo modules loaded")
    dependencies = {}
    for name, distribution_name in (("numpy", "numpy"), ("tyx", "tyx-compute")):
        module = sys.modules.get(name)
        path = getattr(module, "__file__", None)
        distribution = importlib.metadata.distribution(distribution_name)
        dependencies[name] = dict(
            version=distribution.version,
            path=path,
            sha256=_sha(Path(path)) if path and Path(path).is_file() else None,
            direct_url_json=distribution.read_text("direct_url.json"),
        )
    return dict(
        phymeo_version=importlib.metadata.version("phymeo"),
        modules=modules,
        dependencies=dependencies,
        sys_path=tuple(sys.path),
    )


def _plain(value):
    if np is not None and isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=list(value.shape), hex=value.tobytes().hex())
    if np is not None and isinstance(value, np.generic):
        return _plain(np.asarray(value))
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("nonfinite or unsupported saved object: " + type(value).__name__)


def _decode(value, depth=0):
    _require(depth <= 16, "saved input nesting limit")
    if isinstance(value, dict):
        if {"dtype", "shape", "hex"}.intersection(value):
            _require(set(value) == {"dtype", "shape", "hex"}, "saved array fields")
            _require(value["dtype"] in ("<f4", "<f8", "<i4", "<i8"), "saved array dtype")
            shape = value["shape"]
            _require(
                type(shape) is list
                and len(shape) <= 3
                and all(type(v) is int and v >= 0 for v in shape),
                "saved array shape",
            )
            count = math.prod(shape)
            _require(count <= 65536, "saved result array capacity")
            dtype = np.dtype(value["dtype"])
            raw = value["hex"]
            _require(
                type(raw) is str and len(raw) == 2 * count * dtype.itemsize,
                "saved array byte count",
            )
            data = bytes.fromhex(raw)
            _require(len(data) == count * dtype.itemsize, "saved array extent")
            array = np.frombuffer(data, dtype=dtype).reshape(shape)
            _require(np.isfinite(array).all(), "nonfinite accepted input/result")
            return array
        return {k: _decode(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        _require(len(value) <= 4096, "saved sequence limit")
        return tuple(_decode(v, depth + 1) for v in value)
    _require(value is None or type(value) in (str, bool, int, float), "saved scalar type")
    _require(not isinstance(value, float) or math.isfinite(value), "saved nonfinite scalar")
    return value


def _unique_object(entries):
    result = {}
    for key, value in entries:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _load(compressed):
    _require(len(compressed) <= MAX_INPUT_BYTES, "compressed input limit")
    # Limit decompression before JSON/array allocation. The original complete
    # W32-capacity diagnostic file remains bound by its compressed SHA256.
    import io

    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    _require(len(raw) <= MAX_INPUT_BYTES, "uncompressed input limit")
    value = json.loads(raw, object_pairs_hook=_unique_object)
    _require(
        set(value)
        == {"originals", "kwargs", "stats_before", "raw", "results", "stats_after", "elapsed_s"},
        "accepted evidence schema differs",
    )
    _require(
        value["kwargs"] == {} and len(value["originals"]) == len(value["results"]) == 1,
        "expected one accepted default-budget W1 solve",
    )
    original, result = value["originals"][0], value["results"][0]
    _require(set(original) == PROBLEM_FIELDS, "original problem fields differ")
    _require(
        type(original["samples"]) is list and len(original["samples"]) == 128,
        "original sample count differs",
    )
    _require(
        all(set(s) == SAMPLE_FIELDS for s in original["samples"]), "original sample fields differ"
    )
    _require(
        set(result)
        == set(RESULT_SHAPES)
        | {"original_problem", "packed_problem", "trace", "diagnostics", "energy"},
        "result fields differ",
    )
    _require(result["original_problem"] == original, "saved result is bound to another original")
    decoded = _decode(original)
    expected = _decode(result)
    _require(expected["diagnostics"]["law"] == LAW_NAME, "saved law differs")
    for name, shape in RESULT_SHAPES.items():
        array = expected[name]
        _require(
            isinstance(array, np.ndarray)
            and array.dtype == np.dtype("<f4")
            and array.shape == shape,
            "saved result shape/dtype: " + name,
        )
    _require(
        decoded["mass_matrix"].shape == (6, 6) and len(decoded["samples"]) == 128,
        "expected nv6/ns128",
    )
    return value, decoded, expected


def _equations(problem, velocity):
    """Only substitute actual v; no nonlinear or linear HOST solver is called."""
    v = np.asarray(velocity, dtype=np.float64)
    mass = np.asarray(problem.mass_matrix, dtype=np.float64)
    h = float(problem.h_s)
    response = mass + h * np.diag(np.asarray(problem.damping, dtype=np.float64))
    diagonal = np.diag(response)
    _require(
        np.array_equal(response, np.diag(diagonal)) and np.all(diagonal > 0),
        "registered input must have positive diagonal M+hD",
    )
    rhs = mass @ np.asarray(problem.input_velocity, dtype=np.float64) + h * np.asarray(
        problem.applied_force, dtype=np.float64
    )
    free = rhs / diagonal
    force, moment, tangent, normal, contact, tangent_maps, normal_maps = [], [], [], [], [], [], []
    total, jacobian = np.zeros(6), response.copy()
    for sample in problem.samples:
        n = np.asarray(sample.normal_world, dtype=np.float64)
        g = np.asarray(sample.jacobian_a, dtype=np.float64) - np.asarray(
            sample.jacobian_b, dtype=np.float64
        )
        project = np.eye(3) - np.outer(n, n) / (n @ n)
        gn, gt = n @ g, project @ g
        vn, vt = float(gn @ v), gt @ v
        k, mu = float(sample.stiffness_n_m), float(sample.mu)
        gap = float(sample.phi_m) + h * vn
        fn = max(0.0, -k * gap)
        speed = np.sqrt(vt @ vt + float(sample.slip_speed_m_s) ** 2)
        ft = -mu * fn * vt / speed
        f = fn * n + ft
        m = np.cross(np.asarray(sample.point_world, dtype=np.float64), f)
        dfn = -k * h * gn if fn > 0 else np.zeros(6)
        df = (
            np.outer(n - mu * vt / speed, dfn)
            - mu * fn * (np.eye(3) / speed - np.outer(vt, vt) / speed**3) @ gt
        )
        total += g.T @ f
        jacobian -= h * g.T @ df
        normal.append(fn)
        tangent.append(ft)
        force.append(f)
        moment.append(m)
        normal_maps.append(gn)
        tangent_maps.append(gt)
        contact.append(np.r_[fn, vn, gap, vt, ft, f, m, fn * vn, ft @ vt])
    residual = response @ v - rhs - h * total
    norm = float(np.sqrt(np.sum(residual * residual / diagonal)))
    scale = 1 + float(np.sqrt(np.sum(diagonal * free * free)))
    return dict(
        response=response,
        free=free,
        normal=np.array(normal),
        tangent=np.array(tangent),
        force=np.array(force),
        moment=np.array(moment),
        generalized=total,
        residual=residual,
        residual_norm=norm,
        scale=scale,
        jacobian=jacobian,
        contact=np.array(contact),
        normal_maps=np.array(normal_maps),
        tangent_maps=np.array(tangent_maps),
    )


def _engineering(actual, expected, bound, name):
    delta = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64))
    _require(np.all(delta <= bound), "engineering comparison failed: " + name)
    return dict(
        max_abs_error=float(np.max(delta, initial=0)),
        byte_equal=actual.tobytes() == expected.tobytes(),
    )


def _compare(result, expected):
    report = {}
    for name, shape in RESULT_SHAPES.items():
        actual = getattr(result, name)
        _require(
            actual.dtype == np.dtype("<f4") and actual.shape == shape and np.isfinite(actual).all(),
            "actual result shape/dtype/finite: " + name,
        )
        _require(not actual.flags.writeable, "mutable installed result: " + name)
    for name in ("velocity", "free_velocity"):
        report[name] = _engineering(
            getattr(result, name), expected[name], 5e-5 * (1 + np.abs(expected[name])), name
        )
    for name in (
        "normal_force",
        "tangential_force",
        "force_on_a",
        "force_on_b",
        "generalized_force",
    ):
        report[name] = _engineering(
            getattr(result, name), expected[name], 0.002 + 0.002 * np.abs(expected[name]), name
        )
    for name in ("torque_on_a_world_origin", "torque_on_b_world_origin"):
        report[name] = _engineering(
            getattr(result, name), expected[name], 1e-5 + 0.002 * np.abs(expected[name]), name
        )
    report["residual_jacobian"] = _engineering(
        result.residual_jacobian,
        expected["residual_jacobian"],
        2e-7 + 2e-5 * np.abs(expected["residual_jacobian"]),
        "residual_jacobian",
    )
    report["response_matrix"] = _engineering(
        result.response_matrix,
        expected["response_matrix"],
        1e-9 + 2e-6 * np.abs(expected["response_matrix"]),
        "response_matrix",
    )
    equations = _equations(result.packed_problem, result.velocity)
    diagonal = np.diag(equations["response"])
    residual_difference = np.asarray(result.residual, dtype=np.float64) - expected["residual"]
    residual_norm = float(np.sqrt(np.sum(residual_difference**2 / diagonal)))
    _require(
        residual_norm <= 2e-5 * equations["scale"],
        "raw residual difference exceeds mass-scaled gate",
    )
    report["residual"] = dict(
        mass_scaled_difference=residual_norm,
        max_abs_error=float(np.max(np.abs(residual_difference))),
        byte_equal=result.residual.tobytes() == expected["residual"].tobytes(),
    )
    # Derived contact columns use the registered velocity/force/moment bounds,
    # propagated through their linear maps and products; no new physics gate.
    old = expected["contact_data"].astype(np.float64)
    dv = 5e-5 * (1 + np.abs(expected["velocity"].astype(np.float64)))
    dn = np.abs(equations["normal_maps"]) @ dv
    dt = np.einsum("sij,j->si", np.abs(equations["tangent_maps"]), dv)
    df = 0.002 + 0.002 * np.abs(old[:, 6:9])
    dfn = 0.002 + 0.002 * np.abs(old[:, 0])
    bound = np.zeros_like(old)
    bound[:, 0], bound[:, 1] = dfn, dn
    bound[:, 2], bound[:, 3:6] = float(result.packed_problem.h_s) * dn, dt
    bound[:, 6:12] = 0.002 + 0.002 * np.abs(old[:, 6:12])
    bound[:, 12:15] = 1e-5 + 0.002 * np.abs(old[:, 12:15])
    bound[:, 15] = np.abs(old[:, 1]) * dfn + np.abs(old[:, 0]) * dn + dfn * dn
    bound[:, 16] = np.sum(np.abs(old[:, 3:6]) * df + np.abs(old[:, 6:9]) * dt + df * dt, axis=1)
    bound[:, (1, 2, 3, 4, 5, 15, 16)] += (
        8 * np.finfo(np.float32).eps * np.abs(old[:, (1, 2, 3, 4, 5, 15, 16)])
    )
    report["contact_data"] = _engineering(
        result.contact_data, expected["contact_data"], bound, "all 17 contact columns"
    )
    _require(set(report) == set(RESULT_SHAPES), "result comparison omitted an array")
    return report, equations


def _qualify(result, equations):
    _require(
        equations["residual_norm"] <= 2e-5 * equations["scale"],
        "actual FP32-input/output mass residual failed",
    )
    for got, expected in (
        (result.normal_force, equations["normal"]),
        (result.tangential_force, equations["tangent"]),
        (result.force_on_a, equations["force"]),
        (result.generalized_force, equations["generalized"]),
    ):
        _require(
            np.all(np.abs(got - expected) <= 0.002 + 0.002 * np.abs(expected)),
            "independent full force law failed",
        )
    _require(
        np.all(
            np.abs(result.torque_on_a_world_origin - equations["moment"])
            <= 1e-5 + 0.002 * np.abs(equations["moment"])
        ),
        "independent world moment failed",
    )
    np.testing.assert_allclose(
        result.residual_jacobian, equations["jacobian"], rtol=2e-5, atol=2e-7
    )
    _require(
        np.max(np.abs(equations["jacobian"] - equations["jacobian"].T)) > 1e-6,
        "manufactured non-symmetric cross terms lost",
    )
    np.testing.assert_array_equal(result.force_on_b, -result.force_on_a)
    np.testing.assert_array_equal(result.torque_on_b_world_origin, -result.torque_on_a_world_origin)
    # Independently use the stored device forces as well as the law evaluated
    # above. Neither set of values replaces any actual result buffer.
    stored_generalized = np.zeros(6)
    stored_moments = []
    for sample, force in zip(result.packed_problem.samples, result.force_on_a, strict=True):
        g = np.asarray(sample.jacobian_a, dtype=np.float64) - np.asarray(
            sample.jacobian_b, dtype=np.float64
        )
        stored_generalized += g.T @ np.asarray(force, dtype=np.float64)
        stored_moments.append(np.cross(np.asarray(sample.point_world, dtype=np.float64), force))
    h = float(result.packed_problem.h_s)
    stored_residual = (
        equations["response"] @ (result.velocity.astype(np.float64) - equations["free"])
        - h * stored_generalized
    )
    stored_norm = float(np.sqrt(np.sum(stored_residual**2 / np.diag(equations["response"]))))
    _require(stored_norm <= 2e-5 * equations["scale"], "stored-device-force mass residual failed")
    _require(
        np.all(
            np.abs(result.generalized_force - stored_generalized)
            <= 0.002 + 0.002 * np.abs(stored_generalized)
        ),
        "stored generalized force differs from both point Jacobians",
    )
    stored_moments = np.asarray(stored_moments)
    _require(
        np.all(
            np.abs(result.torque_on_a_world_origin - stored_moments)
            <= 1e-5 + 0.002 * np.abs(stored_moments)
        ),
        "stored world-origin moments differ",
    )
    equations.update(
        stored_generalized=stored_generalized,
        stored_residual=stored_residual,
        stored_residual_norm=stored_norm,
    )
    mu = np.array([s.mu for s in result.packed_problem.samples])
    _require(
        np.all(result.normal_force >= 0)
        and np.all(
            np.linalg.norm(result.tangential_force.astype(np.float64), axis=1)
            <= mu * result.normal_force + 1e-5
        ),
        "final normal friction capacity failed",
    )
    diagnostics = result.diagnostics
    _require(diagnostics["law"] == LAW_NAME, "result diagnostic law differs")
    _require(
        0 <= diagnostics["device_residual_norm"] <= diagnostics["device_tolerance"],
        "device residual gate failed",
    )
    np.testing.assert_allclose(
        diagnostics["device_tolerance"], 1e-5 * equations["scale"], rtol=2e-6, atol=0
    )
    np.testing.assert_allclose(
        diagnostics["host_tolerance"], 2e-5 * equations["scale"], rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        diagnostics["host_residual_norm"], equations["residual_norm"], rtol=1e-6, atol=1e-12
    )
    _require(
        bool(diagnostics["host_reference_gate"])
        == (equations["residual_norm"] <= 1e-10 * equations["scale"]),
        "strict HOST gate must remain a separate report",
    )
    _require(
        0 <= diagnostics["iterations"] <= 24 and 0 <= diagnostics["line_search_trials"] <= 288,
        "iteration bounds differ",
    )
    _require(
        type(result.trace) is tuple
        and len(result.trace) == diagnostics["line_search_trials"] + 2
        and 2 <= len(result.trace) <= 290,
        "trace count differs",
    )
    _require(
        result.trace[0].phase == "initial" and result.trace[-1].phase == "final",
        "trace phases differ",
    )
    np.testing.assert_array_equal(result.trace[-1].velocity, result.velocity)


def run(site, input_path, output, forbidden=()):
    global np
    site, input_path, output = Path(site), Path(input_path), Path(output)
    forbidden = tuple(Path(p) for p in forbidden)
    provenance = _guard(site, input_path, output, forbidden)
    output.mkdir(parents=True, exist_ok=False)
    record = dict(
        schema="phymeo-area-friction-wheel-v1",
        status="started",
        provenance=provenance,
        law=LAW_NAME,
        capacity=1,
        worlds=1,
        nv=6,
        samples=128,
        max_iterations=24,
        max_backtracks=12,
        scope="installed standalone backend; no q update, scene, type7 or native-constraint integration",
        raw={},
    )
    start, primary, inventory_failure, workspace = perf_counter(), None, None, None
    try:
        import numpy as numpy

        from phymeo.backends.tyx_area_friction import (
            TyxAreaFrictionResult,
            TyxAreaFrictionWorkspace,
        )
        from phymeo.solvers.area_friction import LAW, AreaFrictionProblem, AreaFrictionSample

        np = numpy
        record["initial_inventory"] = _inventory(site, forbidden)
        _require(LAW == LAW_NAME, "installed law differs")
        _require(input_path.stat().st_size <= MAX_INPUT_BYTES, "compressed input limit")
        compressed = input_path.read_bytes()
        record["input_sha256"] = hashlib.sha256(compressed).hexdigest()
        record["input_bytes"] = len(compressed)
        (output / "accepted-input.json.gz").write_bytes(compressed)
        saved, decoded, expected = _load(compressed)
        problem = AreaFrictionProblem(
            decoded["mass_matrix"],
            decoded["damping"],
            decoded["input_velocity"],
            decoded["applied_force"],
            decoded["h_s"],
            tuple(AreaFrictionSample(**s) for s in decoded["samples"]),
        )
        _require(
            problem.law == LAW_NAME and _plain(problem) == saved["originals"][0],
            "installed input reconstruction changed bytes",
        )
        _require(
            len(problem.samples) == 128
            and len({tuple(s.point_world) for s in problem.samples}) == 128,
            "samples truncated or repeated",
        )
        _require(
            all(
                s.law == LAW_NAME
                and s.jacobian_a.shape == s.jacobian_b.shape == (3, 6)
                and np.any(s.jacobian_b)
                for s in problem.samples
            ),
            "full two-body six-DOF Jacobians required",
        )
        record["problem"] = problem
        with TyxAreaFrictionWorkspace(capacity=1, mode="cuda") as workspace:
            _require(workspace.law == LAW_NAME, "workspace law differs")
            record["stats_before"] = workspace.stats
            _require(
                workspace.stats["live_bytes"] == workspace.stats["peak_bytes"] == 23076 + 252 * 128,
                "capacity1 allocation budget differs",
            )
            get = workspace._download

            def capture(name):
                value = get(name)
                record["raw"][name] = value.copy()
                return value

            workspace._download = capture
            try:
                # The only actual solve in this standalone process.
                results = workspace.solve_many((problem,), max_iterations=24, max_backtracks=12)
            finally:
                workspace._download = get
                record["stats_after_solve"] = workspace.stats
            record["results"] = results
            _require(
                type(results) is tuple
                and len(results) == 1
                and type(results[0]) is TyxAreaFrictionResult,
                "installed result schema/count differs",
            )
            result = results[0]
            _require(result.law == LAW_NAME, "result law differs")
            _require(
                _plain(result.original_problem) == saved["originals"][0],
                "actual original input differs",
            )
            _require(
                _plain(result.packed_problem) == saved["results"][0]["packed_problem"],
                "actual uploaded FP32 input differs",
            )
            _require(
                set(f.name for f in fields(result)) == set(saved["results"][0]),
                "installed result field set differs",
            )
            comparison, equations = _compare(result, expected)
            record["comparisons"], record["independent_equations"] = comparison, equations
            _qualify(result, equations)
            _require(
                set(record["raw"])
                == {"meta", "scalars", "vectors", "matrices", "contact", "trace_meta", "trace"},
                "raw download field set differs",
            )
            _require(
                workspace.stats["launches"] == workspace.stats["batch_calls"] == 1,
                "expected exactly one actual solve",
            )
            _require(
                workspace.stats["uploaded_bytes"] == 23788
                and workspace.stats["downloaded_bytes"] == 31544,
                "capacity1 transfer budget differs",
            )
            record["qualification"] = dict(
                fp32_device=True,
                independent_actual_input=True,
                strict_host_reference=bool(result.diagnostics["host_reference_gate"]),
                all_saved_result_arrays=True,
            )
        _require(workspace.closed, "workspace did not close")
        record["status"] = "passed"
    except BaseException as error:
        primary = error
        record["status"] = "failed"
        record["error"] = dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
        )
        if hasattr(error, "diagnostics"):
            record["rejected_diagnostics"] = error.diagnostics
    finally:
        if workspace is not None:
            record["stats_after_close"] = workspace.stats
        try:
            record["final_inventory"] = _inventory(site, forbidden)
        except BaseException as error:
            inventory_failure = error
            record["status"] = "failed"
            record["inventory_failure"] = dict(type=type(error).__name__, message=str(error))
        record["total_s"] = perf_counter() - start
        with gzip.open(output / "actual.json.gz", "xt", encoding="utf-8") as stream:
            json.dump(_plain(record), stream, ensure_ascii=True, allow_nan=False)
        summary = {
            name: record[name]
            for name in (
                "schema",
                "status",
                "law",
                "capacity",
                "worlds",
                "nv",
                "samples",
                "scope",
                "total_s",
            )
        }
        for name in (
            "input_sha256",
            "qualification",
            "error",
            "inventory_failure",
            "comparisons",
            "stats_after_solve",
            "final_inventory",
        ):
            if name in record:
                summary[name] = record[name]
        summary["files"] = {
            p.name: dict(bytes=p.stat().st_size, sha256=_sha(p))
            for p in sorted(output.iterdir())
            if p.is_file()
        }
        with (output / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(_plain(summary), stream, ensure_ascii=True, indent=2, allow_nan=False)
    if primary is not None:
        raise primary
    if inventory_failure is not None:
        raise inventory_failure
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forbid-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = run(args.site, args.input, args.output, args.forbid_root)
    print(
        json.dumps(
            dict(
                status=result["status"],
                output=str(args.output),
                input_sha256=result.get("input_sha256"),
            )
        )
    )


if __name__ == "__main__":
    main()
