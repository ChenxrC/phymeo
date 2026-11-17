"""Bounded HOST reference for final-normal smooth friction on shared DOFs.

Geometry and both point Jacobians are frozen for one backward-Euler step.
This is not TYX, a general contact detector, or a production contact law.
The energy ledger uses the linear proxy phi + h * Jn v, not a trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from numbers import Real
from pathlib import Path

import numpy as np

MAX_DOF = 6
MAX_SAMPLES = 8
MAX_ITERATIONS = 24
MAX_BACKTRACKS = 12


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _owned(value):
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("nonfinite numerical value")
    # The backing bytes, not merely the array flag, are immutable.
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _array(value, shape, name):
    if isinstance(value, np.ndarray):
        if value.shape != shape or value.dtype.kind not in "iuf":
            raise ValueError(f"{name} has invalid shape or numeric dtype")
    else:

        def check(node, remaining):
            if not remaining:
                _number(node, name)
                return
            if type(node) not in (tuple, list) or len(node) != remaining[0]:
                raise ValueError(f"{name} must have shape {shape}")
            for item in node:
                check(item, remaining[1:])

        check(value, shape)
    try:
        return _owned(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite FP64 values") from error


def _dimension(value, matrix=False):
    if isinstance(value, np.ndarray):
        if value.ndim != 2:
            raise ValueError("matrix rank must be two")
        count = value.shape[0] if matrix else value.shape[1]
    elif type(value) in (list, tuple):
        if matrix:
            count = len(value)
        else:
            if len(value) != 3 or type(value[0]) not in (list, tuple):
                raise ValueError("point Jacobian must have three rows")
            count = len(value[0])
    else:
        raise ValueError("matrix must be an array or bounded nested sequence")
    if not 1 <= count <= MAX_DOF:
        raise ValueError("DOF count must be 1..6")
    return count


@dataclass(frozen=True)
class CoupledFrictionSample:
    point_world: np.ndarray
    normal_world: np.ndarray
    jacobian_a: np.ndarray
    jacobian_b: np.ndarray
    stiffness_n_m: float
    phi_m: float
    mu: float
    slip_speed_m_s: float

    def __post_init__(self):
        count = _dimension(self.jacobian_a)
        point = _array(self.point_world, (3,), "point_world")
        normal = _array(self.normal_world, (3,), "normal_world")
        length = math.hypot(*normal)
        if abs(length - 1.0) > 1e-12:
            raise ValueError("normal_world must be near unit length within 1e-12")
        normal = _owned(normal / length)
        ja = _array(self.jacobian_a, (3, count), "jacobian_a")
        jb = _array(self.jacobian_b, (3, count), "jacobian_b")
        k = _number(self.stiffness_n_m, "stiffness_n_m")
        phi = _number(self.phi_m, "phi_m")
        mu = _number(self.mu, "mu")
        slip = _number(self.slip_speed_m_s, "slip_speed_m_s")
        if k <= 0 or phi > 0 or mu < 0 or slip <= 0:
            raise ValueError("require k>0, phi<=0, mu>=0 and slip_speed>0")
        for name, value in (
            ("point_world", point),
            ("normal_world", normal),
            ("jacobian_a", ja),
            ("jacobian_b", jb),
            ("stiffness_n_m", k),
            ("phi_m", phi),
            ("mu", mu),
            ("slip_speed_m_s", slip),
        ):
            object.__setattr__(self, name, value)


def _copy_sample(sample):
    if type(sample) is not CoupledFrictionSample:
        raise ValueError("samples must be CoupledFrictionSample records")
    return CoupledFrictionSample(*(getattr(sample, f.name) for f in fields(sample)))


@dataclass(frozen=True)
class CoupledFrictionProblem:
    mass_matrix: np.ndarray
    damping: np.ndarray
    input_velocity: np.ndarray
    applied_force: np.ndarray
    h_s: float
    samples: tuple[CoupledFrictionSample, ...]

    def __post_init__(self):
        count = _dimension(self.mass_matrix, matrix=True)
        if type(self.samples) not in (tuple, list) or len(self.samples) > MAX_SAMPLES:
            raise ValueError("samples must be a bounded sequence of at most eight")
        mass = _array(self.mass_matrix, (count, count), "mass_matrix")
        damping = _array(self.damping, (count,), "damping")
        velocity = _array(self.input_velocity, (count,), "input_velocity")
        force = _array(self.applied_force, (count,), "applied_force")
        h = _number(self.h_s, "h_s")
        if h <= 0 or np.any(damping < 0):
            raise ValueError("h must be positive and diagonal damping nonnegative")
        if not np.array_equal(mass, mass.T):
            raise ValueError("mass_matrix must be symmetric; no symmetrization is applied")
        try:
            factor = np.linalg.cholesky(mass)
        except np.linalg.LinAlgError as error:
            raise ValueError("mass_matrix must be positive definite") from error
        if not np.all(np.isfinite(factor)) or np.any(np.diag(factor) <= 0):
            raise ValueError("mass_matrix has no finite positive Cholesky factor")
        samples = tuple(_copy_sample(sample) for sample in self.samples)
        if any(sample.jacobian_a.shape != (3, count) for sample in samples):
            raise ValueError("all samples must use the same full DOF space")
        for name, value in (
            ("mass_matrix", mass),
            ("damping", damping),
            ("input_velocity", velocity),
            ("applied_force", force),
            ("h_s", h),
            ("samples", samples),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class CoupledSampleResult:
    sample_index: int
    normal_velocity_m_s: float
    tangent_velocity_world: np.ndarray
    linear_gap_m: float
    normal_force_n: float
    tangent_force_world: np.ndarray
    force_on_a_world: np.ndarray
    force_on_b_world: np.ndarray
    moment_on_a_world: np.ndarray
    moment_on_b_world: np.ndarray
    impulse_on_a_world: np.ndarray
    impulse_on_b_world: np.ndarray
    generalized_force: np.ndarray
    normal_power_w: float
    tangent_power_w: float


@dataclass(frozen=True)
class CoupledEnergy:
    kinetic_entry_j: float
    kinetic_exit_j: float
    spring_entry_j: float
    spring_exit_j: float
    external_work_j: float
    damping_loss_j: float
    normal_work_j: float
    tangent_work_j: float
    kinetic_numerical_loss_j: float
    kinetic_balance_residual_j: float
    normal_spring_balance_j: float
    mechanical_balance_j: float


@dataclass(frozen=True)
class CoupledAttempt:
    iteration: int
    backtrack: int | None
    alpha: float
    status: str
    velocity: tuple[float, ...] | None
    residual: tuple[float, ...] | None
    residual_norm: float | None
    normal_forces_n: tuple[float, ...]
    tangent_forces_world: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class CoupledFrictionResult:
    problem: CoupledFrictionProblem
    response_matrix: np.ndarray
    free_velocity: np.ndarray
    velocity: np.ndarray
    residual: np.ndarray
    residual_jacobian: np.ndarray
    generalized_contact_force: np.ndarray
    samples: tuple[CoupledSampleResult, ...]
    residual_norm: float
    tolerance: float
    iterations: int
    attempts: tuple[CoupledAttempt, ...]
    energy: CoupledEnergy

    def to_dict(self):
        return _plain(self)


class CoupledFrictionFailure(RuntimeError):
    """Owned failure diagnostics; deliberately has no partial result field."""

    def __init__(self, reason, problem, attempts, tolerance, velocity):
        super().__init__(reason)
        self._reason = reason
        self._problem = problem
        self._attempts = tuple(attempts)
        self._tolerance = tolerance
        self._velocity = None if velocity is None else _owned(velocity)

    @property
    def attempts(self):
        return self._attempts

    @property
    def tolerance(self):
        return self._tolerance

    def to_dict(self):
        return {
            "reason": self._reason,
            "problem": _plain(self._problem),
            "attempts": _plain(self._attempts),
            "tolerance": self._tolerance,
            "last_accepted_iterate": _plain(self._velocity),
        }


def _evaluate(problem, velocity, response, free_velocity):
    """Full residual and its nonsymmetric derivative at this actual iterate."""
    count = len(velocity)
    total = np.zeros(count)
    derivative = np.zeros((count, count))
    samples = []
    identity = np.eye(3)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for index, sample in enumerate(problem.samples):
            g = sample.jacobian_a - sample.jacobian_b
            n = sample.normal_world
            nn = float(n @ n)
            relative = g @ velocity
            vn = _number(n @ relative, "normal velocity")
            tangent = np.cross(n, np.cross(relative, n)) / nn
            gap = _number(sample.phi_m + problem.h_s * vn, "linear gap")
            fn = _number(max(0.0, -sample.stiffness_n_m * gap), "normal force")
            speed = _number(math.hypot(*tangent, sample.slip_speed_m_s), "smoothed slip speed")
            unit = tangent / speed
            ft = -sample.mu * fn * unit
            force = fn * n + ft
            moment = np.cross(sample.point_world, force)
            generalized = g.T @ force
            total += generalized
            # At the max() kink, use the inactive one-sided derivative.
            dfn = -sample.stiffness_n_m * problem.h_s * (n @ g) if gap < 0 else np.zeros(count)
            projection = identity - np.outer(n, n) / nn
            dt = ((identity - np.outer(unit, unit)) / speed) @ projection @ g
            force_derivative = np.outer(n - sample.mu * unit, dfn) - sample.mu * fn * dt
            derivative += g.T @ force_derivative
            normal_power = _number(fn * vn, "normal power")
            tangent_power = _number(ft @ tangent, "tangent power")
            if tangent_power > 1e-14:
                raise ValueError("friction power is positive")
            samples.append(
                CoupledSampleResult(
                    index,
                    vn,
                    _owned(tangent),
                    gap,
                    fn,
                    _owned(ft),
                    _owned(force),
                    _owned(-force),
                    _owned(moment),
                    _owned(-moment),
                    _owned(problem.h_s * force),
                    _owned(-problem.h_s * force),
                    _owned(generalized),
                    normal_power,
                    tangent_power,
                )
            )
        residual = response @ (velocity - free_velocity) - problem.h_s * total
        jacobian = response - problem.h_s * derivative
    return _owned(residual), _owned(jacobian), _owned(total), tuple(samples)


def _metric(residual, factor):
    transformed = np.linalg.solve(factor, residual)
    return _number(math.hypot(*transformed), "mass-metric residual norm")


def _attempt(iteration, backtrack, alpha, status, velocity, evaluation, norm):
    return CoupledAttempt(
        iteration,
        backtrack,
        alpha,
        status,
        None if velocity is None else tuple(float(x) for x in velocity),
        None if evaluation is None else tuple(float(x) for x in evaluation[0]),
        norm,
        () if evaluation is None else tuple(sample.normal_force_n for sample in evaluation[3]),
        ()
        if evaluation is None
        else tuple(tuple(float(x) for x in sample.tangent_force_world) for sample in evaluation[3]),
    )


def _energy(problem, velocity, samples):
    v0, h, mass = problem.input_velocity, problem.h_s, problem.mass_matrix
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        kinetic0 = 0.5 * float(v0 @ mass @ v0)
        kinetic1 = 0.5 * float(velocity @ mass @ velocity)
        spring0 = sum(0.5 * s.stiffness_n_m * s.phi_m**2 for s in problem.samples)
        spring1 = sum(
            0.5 * source.stiffness_n_m * min(actual.linear_gap_m, 0.0) ** 2
            for source, actual in zip(problem.samples, samples, strict=True)
        )
        external = h * float(problem.applied_force @ velocity)
        damping = h * float(np.dot(problem.damping * velocity, velocity))
        normal = h * sum(s.normal_power_w for s in samples)
        tangent = h * sum(s.tangent_power_w for s in samples)
        dv = velocity - v0
        numerical_loss = 0.5 * float(dv @ mass @ dv)
        kinetic_balance = (
            kinetic1 - kinetic0 - external + damping - normal - tangent + numerical_loss
        )
        normal_balance = spring1 - spring0 + normal
        mechanical_balance = kinetic_balance + normal_balance
    values = (
        kinetic0,
        kinetic1,
        spring0,
        spring1,
        external,
        damping,
        normal,
        tangent,
        numerical_loss,
        kinetic_balance,
        normal_balance,
        mechanical_balance,
    )
    return CoupledEnergy(*(_number(value, "energy ledger") for value in values))


def solve(problem, *, max_iterations=MAX_ITERATIONS, max_backtracks=MAX_BACKTRACKS):
    """Solve one bounded frozen-geometry step or raise owned failure diagnostics.

    No global convergence guarantee is implied. Each trial uses the full
    actual residual. A Cholesky measures residuals, never solves Newton's
    generally nonsymmetric matrix. Zero budgets are useful failure probes.
    """
    if type(problem) is not CoupledFrictionProblem:
        raise ValueError("problem must be CoupledFrictionProblem")
    if type(max_iterations) is not int or not 0 <= max_iterations <= MAX_ITERATIONS:
        raise ValueError("max_iterations must be an integer in 0..24")
    if type(max_backtracks) is not int or not 0 <= max_backtracks <= MAX_BACKTRACKS:
        raise ValueError("max_backtracks must be an integer in 0..12")
    # Rebuild all public data: frozen records can still be maliciously replaced.
    problem = CoupledFrictionProblem(*(getattr(problem, f.name) for f in fields(problem)))
    attempts = []
    velocity = None
    tolerance = None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            response = _owned(problem.mass_matrix + problem.h_s * np.diag(problem.damping))
            factor = np.linalg.cholesky(response)
            free = _owned(
                np.linalg.solve(
                    response,
                    problem.mass_matrix @ problem.input_velocity
                    + problem.h_s * problem.applied_force,
                )
            )
            tolerance = _number(1e-10 * (1.0 + math.hypot(*(factor.T @ free))), "tolerance")
        velocity = free
        evaluation = _evaluate(problem, velocity, response, free)
        norm = _metric(evaluation[0], factor)
        attempts.append(_attempt(0, None, 0.0, "initial", velocity, evaluation, norm))
        iterations = 0
        while norm > tolerance:
            if iterations >= max_iterations:
                raise CoupledFrictionFailure(
                    "Newton update budget exhausted", problem, attempts, tolerance, velocity
                )
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                direction = _owned(np.linalg.solve(evaluation[1], -evaluation[0]))
            accepted = False
            for backtrack in range(max_backtracks):
                alpha = 2.0 ** (-backtrack)
                candidate = None
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        candidate = _owned(velocity + alpha * direction)
                    trial = _evaluate(problem, candidate, response, free)
                    trial_norm = _metric(trial[0], factor)
                except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
                    attempts.append(
                        _attempt(
                            iterations + 1,
                            backtrack,
                            alpha,
                            "nonfinite-or-singular",
                            candidate,
                            None,
                            None,
                        )
                    )
                    continue
                accepted = trial_norm <= max(tolerance, (1.0 - 1e-4 * alpha) * norm)
                attempts.append(
                    _attempt(
                        iterations + 1,
                        backtrack,
                        alpha,
                        "accepted" if accepted else "rejected",
                        candidate,
                        trial,
                        trial_norm,
                    )
                )
                if accepted:
                    velocity, evaluation, norm = candidate, trial, trial_norm
                    iterations += 1
                    break
            if not accepted:
                raise CoupledFrictionFailure(
                    "backtracking budget exhausted", problem, attempts, tolerance, velocity
                )
        # Re-evaluate the same returned velocity, including its force and ledger.
        evaluation = _evaluate(problem, velocity, response, free)
        norm = _metric(evaluation[0], factor)
        attempts.append(_attempt(iterations, None, 0.0, "final", velocity, evaluation, norm))
        if norm > tolerance:
            raise CoupledFrictionFailure(
                "final residual verification failed", problem, attempts, tolerance, velocity
            )
        energy = _energy(problem, velocity, evaluation[3])
        return CoupledFrictionResult(
            problem,
            response,
            free,
            _owned(velocity),
            evaluation[0],
            evaluation[1],
            evaluation[2],
            evaluation[3],
            norm,
            tolerance,
            iterations,
            tuple(attempts),
            energy,
        )
    except CoupledFrictionFailure:
        raise
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        raise CoupledFrictionFailure(
            f"numerical failure: {error}", problem, attempts, tolerance, velocity
        ) from error


def fixed_problems():
    """The six preregistered inputs; no target is computed by the force kernel."""
    ja = ((1, 0), (0, 0), (0, 1))
    jb = ((0, 0), (0, 0), (0, 0))
    support_sample = CoupledFrictionSample((0, 0, 0), (0, 0, 1), ja, jb, 4000, -0.001, 0.5, 0.001)
    release_sample = CoupledFrictionSample((0, 0, 0), (0, 0, 1), ja, jb, 4000, -0.0001, 0.5, 0.001)
    mass2 = ((0.5, 0), (0, 0.5))
    support = CoupledFrictionProblem(mass2, (0, 0), (0.02, 0), (0, -4), 0.001, (support_sample,))
    release = CoupledFrictionProblem(mass2, (0, 0), (0.02, 0.3), (0, 0), 0.001, (release_sample,))
    shared_sample = CoupledFrictionSample(
        (0, 0, 0), (0, 0, 1), ((1,), (0,), (1,)), ((0,), (0,), (0,)), 4000, -0.001, 0.5, 0.001
    )
    shared = CoupledFrictionProblem(((0.5,),), (0,), (0.02,), (0,), 0.001, (shared_sample,))
    points = ((-0.02, 2000, -0.001, 0.4), (0.02, 3000, -0.0008, 0.6))

    def two_samples(frictionless):
        return tuple(
            CoupledFrictionSample(
                (x, 0, -0.01),
                (0, 0, 1),
                ((1, 0, -0.01), (0, 0, 0), (0, 1, -x)),
                ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
                k,
                phi,
                0.0 if frictionless else mu,
                0.001,
            )
            for x, k, phi, mu in points
        )

    mass3 = ((0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.0004))
    # Hand calculation at (.015,-.002,.3): vt=.012, Fn=(1.992,2.424).
    # Qx=-27.0144/sqrt(145), Qz=4.416, Qpitch=-.01*Qx-.00864.
    root = math.sqrt(145)
    force = (-2.485 + 27.0144 / root, -0.42, 0.04894 - 0.270144 / root)
    manufactured = CoupledFrictionProblem(
        mass3, (1, 2, 0.001), (0.02, -0.01, 0.2), force, 0.001, two_samples(False)
    )
    frictionless = CoupledFrictionProblem(
        mass3, (0, 0, 0), (0, -0.1, 0.2), (0, 0, 0), 0.001, two_samples(True)
    )
    empty = CoupledFrictionProblem(mass2, (0, 0), (0.02, 0), (0, -4), 0.001, ())
    return (
        ("support", support),
        ("release", release),
        ("shared-one-dof", shared),
        ("two-point-manufactured", manufactured),
        ("two-point-frictionless", frictionless),
        ("empty", empty),
    )


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite JSON value")
    return value


def write_reference(output):
    """Run these six independent steps once and retain complete diagnostics."""
    directory = Path(output)
    if not directory.is_absolute() or directory.exists():
        raise ValueError("output must be an absolute fresh directory")
    directory.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema": "phymeo-coupled-friction-reference-v1",
        "scope": "frozen-jacobian-shared-dof-host-reference",
        "is_tyx_runtime": False,
        "law": "final-normal-smooth",
        "status": "running",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "budgets": {"newton_updates": MAX_ITERATIONS, "backtracks": MAX_BACKTRACKS},
        "records": [],
    }
    destination = directory / "reference.json"
    try:
        for name, problem in fixed_problems():
            try:
                result = solve(problem)
            except CoupledFrictionFailure as error:
                payload["records"].append({"case": name, "failure": error.to_dict()})
                raise
            payload["records"].append({"case": name, "result": result.to_dict()})
        payload["status"] = "complete"
    except Exception as error:
        payload["status"] = "failed"
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
        )
    lines = [
        "# 共享自由度摩擦参考",
        "",
        "六个独立单步，固定 Jacobian 和线性间隙；不是 TYX 或连续物理轨迹。",
        "平滑摩擦使用同一步最终法向承载，允许微滑，不声称精确静摩擦或一般收敛。",
        "",
        "| case | velocity | normal forces (N) | residual / tolerance |",
        "| --- | --- | --- | --- |",
    ]
    for record in payload["records"]:
        result = record["result"]
        loads = [sample["normal_force_n"] for sample in result["samples"]]
        lines.append(
            f"| {record['case']} | {result['velocity']} | {loads} | {result['residual_norm']:.6g} / {result['tolerance']:.6g} |"
        )
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(write_reference(arguments.output))


if __name__ == "__main__":
    main()
