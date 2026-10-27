"""Run and save the fixed benchmark with an explicit one-way counterfactual."""

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..builders.rigid_cloth import rigid_cloth_scene
from ..core.coupled import CoupledState
from ..core.rigid import RigidBodyModel
from ..core.validation import validate_dt
from ..solvers.colored_coupled import ColoredCoupledXPBDSolver
from ..solvers.coupled_xpbd import CoupledXPBDSolver
from ..solvers.rigid_semi_implicit import RigidSemiImplicitSolver
from ..validation.rigid_cloth import audit_substep


class SceneRunFailure(RuntimeError):
    pass


@dataclass(frozen=True, eq=False)
class SceneRun:
    config: dict
    metrics: dict
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True, eq=False)
class _PrescribedSphereSolver(CoupledXPBDSolver):
    """Experiment-only driver: finite-mass free prediction, zero contact response.

    This is deliberately not a physical configuration of the public solver.
    The driver supplies all momentum needed to cancel the cloth's reaction.
    """

    driver_model: RigidBodyModel | None = None

    def _predict_rigid(self, state, inputs, h):
        return RigidSemiImplicitSolver().step(self.driver_model, state, inputs, h)


def trajectory_digest(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def run_scene(
    *,
    frames=240,
    dt=1 / 120,
    substeps=4,
    mode="numpy",
    reaction=True,
    dtype=np.float64,
    iterations=200,
    tolerance=1e-7,
    sphere_mass=0.05,
    progress=None,
    cache_root=None,
):
    """Return numeric data and audits; rejected frames never enter a success trace."""
    if (
        isinstance(frames, (bool, np.bool_))
        or not isinstance(frames, (int, np.integer))
        or frames < 1
    ):
        raise ValueError("frames must be a positive integer")
    if mode not in ("numpy", "numpy_colored", "interpret", "cpu_cpp", "cuda"):
        raise ValueError("unsupported scene execution mode")
    if not isinstance(reaction, (bool, np.bool_)):
        raise TypeError("reaction must be a boolean")
    if not reaction and mode != "numpy":
        raise ValueError("the prescribed one-way experiment currently requires mode='numpy'")
    scene = rigid_cloth_scene(dtype, sphere_mass=sphere_mass)
    bound_model = (
        scene.rigid_model
        if reaction
        else RigidBodyModel(
            np.zeros_like(scene.rigid_model.inverse_mass),
            scene.rigid_model.inertia_diagonal,
            scene.rigid_model.gravity,
        )
    )
    arguments = dict(
        particle_model=scene.particle_model,
        rigid_model=bound_model,
        constraints=scene.constraints,
        spheres=scene.spheres,
        substeps=substeps,
        iterations=iterations,
        tolerance=tolerance,
        record_substeps=True,
    )
    if mode == "numpy":
        solver = (
            CoupledXPBDSolver(**arguments)
            if reaction
            else _PrescribedSphereSolver(**arguments, driver_model=scene.rigid_model)
        )
    elif mode == "numpy_colored":
        solver = ColoredCoupledXPBDSolver(**arguments)
    else:
        from ..backends.tyx_coupled import TyxCoupledXPBDSolver

        solver = TyxCoupledXPBDSolver(**arguments, mode=mode, cache_root=cache_root)
    try:
        return _record_scene(
            scene,
            solver,
            frames=frames,
            dt=dt,
            substeps=substeps,
            mode=mode,
            reaction=reaction,
            dtype=dtype,
            iterations=iterations,
            tolerance=tolerance,
            progress=progress,
        )
    finally:
        if hasattr(solver, "close"):
            solver.close()


def _record_scene(
    scene, solver, *, frames, dt, substeps, mode, reaction, dtype, iterations, tolerance, progress
):
    h = float(validate_dt(float(validate_dt(dt, np.dtype(dtype))) / substeps, np.dtype(dtype)))
    config = dict(
        scene.config,
        frames=int(frames),
        requested_frame_dt=float(dt),
        effective_substep_dt=h,
        substeps=int(substeps),
        duration_seconds=int(frames) * int(substeps) * h,
        iterations=int(iterations),
        tolerance=float(tolerance),
        backend=mode,
        distance_order="original" if mode == "numpy" else "stable_colors",
        projection_arithmetic="fp64" if mode in ("numpy", "numpy_colored") else "fp32",
        interaction="two_way_dynamic" if reaction else "one_way_prescribed_sphere",
    )
    current = CoupledState(scene.initial_state.particles, scene.initial_state.rigid)
    trajectory = {
        name: []
        for name in (
            "particle_position",
            "particle_velocity",
            "rigid_position",
            "rigid_velocity",
            "rigid_orientation",
            "rigid_angular_velocity",
        )
    }

    def record(state):
        for name in trajectory:
            family, field = name.split("_", 1)
            owner = state.particles if family == "particle" else state.rigid
            trajectory[name].append(getattr(owner, field).copy())

    record(current)
    audits = []
    started = time.perf_counter()
    for frame in range(int(frames)):
        result = solver.step(current, scene.inputs, dt)
        if mode in ("interpret", "cpu_cpp", "cuda") and result.report.backend != f"tyx:{mode}":
            raise SceneRunFailure("requested device backend did not execute")
        if not result.report.accepted:
            raise SceneRunFailure(
                f"frame {frame + 1}: {result.report.reason}; residual {result.report.max_residual}"
            )
        previous = current
        for snapshot, report in zip(result.substep_states, result.report.substeps, strict=True):
            try:
                audits.append(
                    audit_substep(
                        scene, previous, snapshot, report, h, tolerance=tolerance, reaction=reaction
                    )
                )
            except AssertionError as error:
                raise SceneRunFailure(f"frame {frame + 1}: independent audit: {error}") from error
            previous = snapshot
        if len(result.substep_states) != substeps:
            raise SceneRunFailure(f"frame {frame + 1}: missing substep evidence")
        current = result.state
        record(current)
        if progress is not None:
            progress(frame + 1, result.report)
    elapsed = time.perf_counter() - started
    arrays = {name: np.stack(rows) for name, rows in trajectory.items()}
    arrays.update(
        time=np.arange(int(frames) + 1) * int(substeps) * h,
        triangles=scene.cloth.triangles.copy(),
        edges=scene.constraints.indices.copy(),
        pinned=scene.pinned.copy(),
        vertex_mass=scene.cloth.vertex_mass.copy(),
    )
    for name in audits[0]:
        arrays["substep_" + name] = np.array([row[name] for row in audits])
    metrics = {
        "execution_backend": result.report.backend,
        "accepted_frames": int(frames),
        "audited_substeps": len(audits),
        "contact_substeps": sum(row["contact"] for row in audits),
        "runtime_seconds_including_audits": elapsed,
        "max_position_iterations": max(row["position_iterations"] for row in audits),
        "max_velocity_iterations": max(row["velocity_iterations"] for row in audits),
    }
    for name in audits[0]:
        if name not in ("contact", "position_iterations", "velocity_iterations"):
            metrics["max_" + name] = max(row[name] for row in audits)
    metrics["trajectory_sha256"] = trajectory_digest(arrays)
    return SceneRun(config, metrics, arrays)


def compare_runs(coupled, prescribed):
    """Prove feedback changes both trajectories under the same experiment controls."""
    if (
        coupled.config["interaction"] != "two_way_dynamic"
        or prescribed.config["interaction"] != "one_way_prescribed_sphere"
    ):
        raise ValueError("comparison requires two-way and one-way runs")
    if {k: v for k, v in coupled.config.items() if k != "interaction"} != {
        k: v for k, v in prescribed.config.items() if k != "interaction"
    }:
        raise ValueError("comparison configurations must otherwise match")
    np.testing.assert_array_equal(coupled.arrays["time"], prescribed.arrays["time"])
    sphere = float(
        np.max(
            np.linalg.norm(
                coupled.arrays["rigid_position"] - prescribed.arrays["rigid_position"], axis=-1
            )
        )
    )
    cloth = float(
        np.max(
            np.linalg.norm(
                coupled.arrays["particle_position"] - prescribed.arrays["particle_position"],
                axis=-1,
            )
        )
    )
    if (
        sphere < 1e-3
        or cloth < 1e-6
        or min(coupled.metrics["contact_substeps"], prescribed.metrics["contact_substeps"]) < 1
    ):
        raise AssertionError("comparison does not establish two-way interaction")
    return {"max_sphere_position_difference_m": sphere, "max_cloth_position_difference_m": cloth}


def free_fall_convergence(*, frames=240, dt=1 / 120, substeps=4):
    """Independent analytic test: semi-implicit position error must halve with h."""
    scene = rigid_cloth_scene()
    solver = RigidSemiImplicitSolver()
    h = float(validate_dt(dt, np.dtype(np.float64))) / substeps
    count = frames * substeps
    duration = count * h
    exact_position = (
        scene.initial_state.rigid.position
        + duration * scene.initial_state.rigid.velocity
        + 0.5 * duration**2 * scene.rigid_model.gravity
    )
    exact_velocity = scene.initial_state.rigid.velocity + duration * scene.rigid_model.gravity
    errors = []
    for scale in (1, 2):
        state = scene.initial_state.rigid
        for _ in range(count * scale):
            result = solver.step(scene.rigid_model, state, scene.inputs.rigid, h / scale)
            if not result.report.accepted:
                raise SceneRunFailure("free-fall convergence prediction rejected")
            state = result.state
        errors.append(
            (
                float(np.max(np.abs(state.position - exact_position))),
                float(np.max(np.abs(state.velocity - exact_velocity))),
            )
        )
    ratio = errors[0][0] / errors[1][0]
    if not np.isclose(ratio, 2.0, rtol=1e-8, atol=0):
        raise SceneRunFailure("free-fall position error did not halve with the step size")
    return {
        "coarse_position_error_m": errors[0][0],
        "fine_position_error_m": errors[1][0],
        "error_ratio": ratio,
        "fine_velocity_error_mps": errors[1][1],
        "duration_seconds": duration,
    }


def run_suite(destination, *, progress=None, **arguments):
    """Run the FP64 acceptance controls and save each measured trajectory.

    A suite is successful only when every run, deterministic repetition and the
    one-way difference pass. Step-halving reports trajectory sensitivity; it does
    not pretend a contact trajectory has an analytic solution.
    """
    if "reaction" in arguments or np.dtype(arguments.get("dtype", np.float64)) != np.dtype(
        np.float64
    ):
        raise ValueError("the reference suite requires FP64 and chooses its own reaction controls")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    source = provenance()
    frames, dt = arguments.get("frames", 240), arguments.get("dt", 1 / 120)
    mass = arguments.get("sphere_mass", 0.05)
    variants = (
        ("two_way", {}),
        ("repeat", {}),
        ("one_way", {"reaction": False}),
        ("half_step", {"frames": 2 * frames, "dt": dt / 2}),
        ("mass_half", {"sphere_mass": mass / 2}),
        ("mass_double", {"sphere_mass": mass * 2}),
    )
    runs = {}
    try:
        for name, changes in variants:
            callback = (
                None
                if progress is None
                else lambda frame, report, name=name: progress(name, frame, report)
            )
            runs[name] = run_scene(**(arguments | changes), progress=callback)
            save_run(runs[name], destination / name, source=source)
        baseline, repeated = runs["two_way"], runs["repeat"]
        for name in baseline.arrays:
            np.testing.assert_array_equal(baseline.arrays[name], repeated.arrays[name])
        if baseline.metrics["trajectory_sha256"] != repeated.metrics["trajectory_sha256"]:
            raise SceneRunFailure("repeated trajectory digest differs")
        comparison = compare_runs(baseline, runs["one_way"])
        prescribed = runs["one_way"]
        h = prescribed.config["effective_substep_dt"]
        n = np.arange(frames + 1) * prescribed.config["substeps"]
        expected_y = 0.3 + n * h * 2 - 9.81 * h**2 * n * (n + 1) / 2
        expected_vy = 2 - n * h * 9.81
        analytic_error = float(
            np.max(np.abs(prescribed.arrays["rigid_position"][:, 0, 1] - expected_y))
        )
        np.testing.assert_allclose(
            prescribed.arrays["rigid_position"][:, 0, 1], expected_y, atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(
            prescribed.arrays["rigid_velocity"][:, 0, 1], expected_vy, atol=1e-10, rtol=0
        )
        comparison["one_way_discrete_free_fall_error_m"] = analytic_error
        mass_sensitivity = {}
        for name in ("mass_half", "mass_double"):
            variation = runs[name]
            difference = float(
                np.max(
                    np.linalg.norm(
                        baseline.arrays["rigid_position"] - variation.arrays["rigid_position"],
                        axis=-1,
                    )
                )
            )
            if variation.metrics["contact_substeps"] == 0 or difference < 1e-6:
                raise SceneRunFailure(
                    "mass variation did not produce a measurable contact response"
                )
            mass_sensitivity[name] = {
                "sphere_mass_kg": variation.config["sphere_mass_kg"],
                "max_sphere_position_difference_m": difference,
            }
        np.testing.assert_array_equal(
            baseline.arrays["time"], runs["half_step"].arrays["time"][::2]
        )
        half_step = {
            "duration_seconds": runs["half_step"].config["duration_seconds"],
            "max_sphere_position_difference_m": float(
                np.max(
                    np.linalg.norm(
                        baseline.arrays["rigid_position"]
                        - runs["half_step"].arrays["rigid_position"][::2],
                        axis=-1,
                    )
                )
            ),
            "max_cloth_position_difference_m": float(
                np.max(
                    np.linalg.norm(
                        baseline.arrays["particle_position"]
                        - runs["half_step"].arrays["particle_position"][::2],
                        axis=-1,
                    )
                )
            ),
        }
        record = {
            "status": "complete",
            "provenance": source,
            "deterministic_repetition": True,
            "comparison": comparison,
            "mass_sensitivity": mass_sensitivity,
            "step_halving": half_step,
            "analytic_free_fall": free_fall_convergence(
                frames=frames, dt=dt, substeps=arguments.get("substeps", 4)
            ),
            "runs": {
                name: {"config": run.config, "metrics": run.metrics} for name, run in runs.items()
            },
        }
        (destination / "suite.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return record
    except (ValueError, TypeError, RuntimeError, AssertionError) as error:
        (destination / "failure.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(error),
                    "completed_runs": list(runs),
                    "provenance": source,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        raise


def provenance():
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for source in sorted(package.rglob("*.py")):
        digest.update(source.relative_to(package).as_posix().encode())
        digest.update(source.read_bytes())
    root = Path(__file__).resolve().parents[3]
    revision, dirty = None, None
    if (root / ".git").exists():
        try:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=root, text=True
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            pass
    return {
        "git_revision": revision,
        "git_dirty": dirty,
        "source_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def save_run(run, destination, *, source=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / "trajectory.npz", **run.arrays)
    record = {
        "status": "complete",
        "config": run.config,
        "metrics": run.metrics,
        "provenance": provenance() if source is None else source,
    }
    (destination / "run.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def load_run(directory):
    """Load a completed numeric trace and reject corruption before comparison."""
    directory = Path(directory)
    record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if record.get("status") != "complete":
        raise ValueError("trace record is not complete")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        arrays = dict(archive)
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("trace contains non-finite numeric data")
    if trajectory_digest(arrays) != record["metrics"]["trajectory_sha256"]:
        raise ValueError("trace digest does not match the recorded data")
    return SceneRun(record["config"], record["metrics"], arrays)
