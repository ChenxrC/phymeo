"""HOST implicit tangential spring return with full reciprocal station mobility.

The caller supplies a frozen tangent chart and free displacement. This module
neither detects contact nor publishes physical state or contact identity history.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..constraints.friction import solve_friction_disk
from ..core.capabilities import SolverFailure
from ..core.validation import finite_real


def _array(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
        raise ValueError(f"{name} requires finite FP64 shape {shape}")
    return np.array(a, copy=True)


def _owned(value):
    a = np.asarray(value, dtype=np.float64)
    return np.frombuffer(a.tobytes(), dtype=np.float64).reshape(a.shape)


def _norm(a):
    result = np.hypot.reduce(a, axis=-1)
    if not np.isfinite(result).all():
        raise SolverFailure("non-finite coupled tangent norm")
    return result


@dataclass(frozen=True, eq=False)
class TangentMotionVerification:
    """Actual-displacement audit; an accepted flag is not a scene publication."""

    accepted: bool
    actual_displacement_m: np.ndarray
    discrepancy_m: np.ndarray
    maximum_error_m: float
    work_j: np.ndarray
    energy_residual_j: np.ndarray

    def __post_init__(self):
        for name in ("actual_displacement_m", "discrepancy_m", "work_j", "energy_residual_j"):
            object.__setattr__(self, name, _owned(getattr(self, name)))


@dataclass(frozen=True, eq=False)
class CoupledTangentResult:
    """Forces act on the positive side of each supplied relative tangent coordinate.

    Work and energies are per station, in joules. ``iterations`` counts bounded
    block sweeps (zero for a feasible full unconstrained solve). ``residual`` is
    the full projected KKT residual divided by each station's force scale;
    ``kkt_residual_n`` retains its unscaled force magnitude in newtons.
    No numerical objective from the block optimizer is reported as kinetic energy.
    """

    force_n: np.ndarray
    predicted_displacement_m: np.ndarray
    elastic_displacement_m: np.ndarray
    plastic_increment_m: np.ndarray
    work_j: np.ndarray
    stored_energy_j: np.ndarray
    plastic_dissipation_j: np.ndarray
    implicit_dissipation_j: np.ndarray
    released_energy_j: np.ndarray
    energy_residual_j: np.ndarray
    regime: tuple[str, ...]
    iterations: int
    residual: float
    kkt_residual_n: np.ndarray
    entry_stored_energy_j: np.ndarray

    def __post_init__(self):
        for name in (
            "force_n",
            "predicted_displacement_m",
            "elastic_displacement_m",
            "plastic_increment_m",
            "work_j",
            "stored_energy_j",
            "plastic_dissipation_j",
            "implicit_dissipation_j",
            "released_energy_j",
            "energy_residual_j",
            "kkt_residual_n",
            "entry_stored_energy_j",
        ):
            object.__setattr__(self, name, _owned(getattr(self, name)))

    def verify_motion(self, actual_displacement_m, *, tolerance_m):
        """Compare actual material increments with the linear prediction, without mutation.

        The absolute tolerance in metres is mandatory, finite and positive.
        Finite rigid rotations generally do not realize h*J*v exactly. An adapter
        must check this receipt before publishing its physical and history candidate.
        """
        bound = finite_real(tolerance_m, "tolerance_m")
        if bound <= 0:
            raise ValueError("tolerance_m must be finite and positive")
        actual = _array(
            actual_displacement_m, self.predicted_displacement_m.shape, "actual displacement"
        )
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                error = actual - self.predicted_displacement_m
                maximum = float(np.max(_norm(error)))
                work = np.einsum("ij,ij->i", self.force_n, actual)
                energy = (
                    work
                    + self.stored_energy_j
                    - self.entry_stored_energy_j
                    + self.plastic_dissipation_j
                    + self.implicit_dissipation_j
                    + self.released_energy_j
                )
        except (FloatingPointError, OverflowError) as exc:
            raise SolverFailure(f"material motion receipt is not representable: {exc}") from exc
        return TangentMotionVerification(maximum <= bound, actual, error, maximum, work, energy)


def _project(force, capacity):
    lengths = _norm(force)
    scale = np.ones(len(force))
    outside = lengths > capacity
    scale[outside] = capacity[outside] / lengths[outside]
    return force * scale[:, None]


def _residual(A, rhs, force, stiffness, capacity):
    gradient = (A @ force.ravel() + rhs).reshape(-1, 2)
    projected = _project(force - stiffness[:, None] * gradient, capacity)
    raw = _norm(force - projected)
    # This scale is a constitutive force scale, not a convergence value substituted
    # for a measured residual. Keep the raw N-valued discrepancy in the receipt.
    scale = np.maximum.reduce(
        (
            np.minimum(stiffness * _norm(rhs.reshape(-1, 2)), capacity),
            _norm(force),
            np.full(len(force), 1e-12),
        )
    )
    return float(np.max(raw / scale)), raw, scale, gradient


def _solve(A, rhs, stiffness, capacity, tolerance, max_iterations):
    count = len(stiffness)
    active = np.flatnonzero(capacity > 0)
    rows = np.ravel(np.column_stack((2 * active, 2 * active + 1)))
    force = np.zeros((count, 2))
    if len(rows):
        full = np.linalg.solve(A[np.ix_(rows, rows)], -rhs[rows]).reshape(-1, 2)
        force[active] = full
        if np.all(_norm(full) <= capacity[active]):
            residual, raw, scale, gradient = _residual(A, rhs, force, stiffness, capacity)
            if residual <= tolerance:
                return force, 0, residual, raw, scale, gradient
        force = _project(force, capacity)
    else:
        residual, raw, scale, gradient = _residual(A, rhs, force, stiffness, capacity)
        return force, 0, residual, raw, scale, gradient
    for iteration in range(1, max_iterations + 1):
        for station in active:
            at = slice(2 * station, 2 * station + 2)
            block = A[at, at]
            local = rhs[at] + A[at] @ force.ravel() - block @ force[station]
            # Algebraic disk optimizer: response has m/N, local has m and output
            # has N here. Its energy_change is a numerical objective difference,
            # not the kinetic-energy channel of the physical mass response.
            optimized = solve_friction_disk(
                block,
                local,
                float(capacity[station]),
                mu_static=1.0,
                mu_dynamic=1.0,
                tolerance=min(tolerance * 0.05, 1e-12),
                max_iterations=96,
            )
            force[station] = optimized.impulse
        residual, raw, scale, gradient = _residual(A, rhs, force, stiffness, capacity)
        if residual <= tolerance:
            return force, iteration, residual, raw, scale, gradient
    raise SolverFailure(
        f"coupled tangent iteration limit reached with full KKT residual {residual:g}"
    )


def solve_coupled_tangent(
    elastic_displacement_m,
    free_displacement_m,
    mobility_per_kg,
    dt_s,
    normal_force_n,
    *,
    stiffness_n_per_m,
    friction,
    tolerance=1e-10,
    max_iterations=256,
):
    """Solve all equal-mu tangential spring/Coulomb stations as one convex problem.

    Displacements are (n,2) metres in caller-supplied tangent coordinates, W is
    (2n,2n) kg^-1 including every cross block, N is (n,) newtons, kt is (n,) N/m,
    and friction is (n,) dimensionless. Inputs use FP64. 1<=n<=128 and a bounded
    iteration budget apply. W may be PSD and redundant; kt must be positive.

    The normal loads and free motion are prescribed by the caller at one physical
    step. No new rigid/robot trajectory is computed here. Normal unloading and
    zero friction clear strain and book old energy exactly once as released energy.
    Old smooth-friction materials and the scalar MaterialStick law are unchanged.
    """
    raw_old = np.asarray(elastic_displacement_m)
    if raw_old.ndim != 2 or raw_old.shape[1:] != (2,) or not 1 <= len(raw_old) <= 128:
        raise ValueError("coupled tangent capacity is 1..128 two-component stations")
    n = len(raw_old)
    old = _array(raw_old, (n, 2), "elastic displacement")
    free = _array(free_displacement_m, (n, 2), "free displacement")
    W = _array(mobility_per_kg, (2 * n, 2 * n), "mobility")
    normal = _array(normal_force_n, (n,), "normal force")
    stiffness = _array(stiffness_n_per_m, (n,), "tangent stiffness")
    mu = _array(friction, (n,), "friction")
    h, tolerance = finite_real(dt_s, "dt_s"), finite_real(tolerance, "tolerance")
    if h <= 0 or not 0 < tolerance <= 1e-3:
        raise ValueError("require positive dt and tolerance in (0,1e-3]")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, Integral)
        or not 1 <= max_iterations <= 4096
    ):
        raise ValueError("max_iterations must be an integer in 1..4096")
    if np.any(normal < 0) or np.any(mu < 0) or np.any(stiffness <= 0):
        raise ValueError("require N>=0, mu>=0 and kt>0")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            scale = float(np.max(np.abs(W)))
            if scale and np.max(np.abs(W / scale - W.T / scale)) > 1e-12:
                raise ValueError("mobility must be symmetric to relative 1e-12")
            W = 0.5 * W + 0.5 * W.T
            eps = np.finfo(float).eps
            if scale and np.linalg.eigvalsh(W / scale)[0] < -32 * eps * (2 * n):
                raise ValueError("mobility must be positive semidefinite within roundoff")
            # No eigenvalue clipping: retain the supplied physical response.
            A = np.diag(np.repeat(1 / stiffness, 2)) + h * h * W
            np.linalg.cholesky(A)
            capacity = mu * normal
            rhs = (old + free).ravel()
            force, iterations, residual, raw, force_scale, gradient = _solve(
                A,
                rhs,
                stiffness,
                capacity,
                tolerance,
                max_iterations,
            )
            delta = free + (h * h * (W @ force.ravel())).reshape(n, 2)
            active = capacity > 0
            elastic = np.zeros((n, 2))
            elastic[active] = -force[active] / stiffness[active, None]
            plastic = np.zeros((n, 2))
            plastic[active] = old[active] + delta[active] - elastic[active]
            prior_energy = 0.5 * stiffness * np.einsum("ij,ij->i", old, old)
            stored = 0.5 * stiffness * np.einsum("ij,ij->i", elastic, elastic)
            work = np.einsum("ij,ij->i", force, delta)
            loss = -np.einsum("ij,ij->i", force, plastic)
            implicit = np.zeros(n)
            increment = elastic[active] - old[active]
            implicit[active] = 0.5 * stiffness[active] * np.einsum("ij,ij->i", increment, increment)
            released = np.where(active, 0.0, prior_energy)
            budget = (
                128 * eps * (abs(work) + stored + prior_energy + abs(loss) + implicit + released)
            )
            budget += tolerance * force_scale * _norm(plastic)
            if np.any(loss < -budget):
                raise SolverFailure("coupled tangent return has negative plastic dissipation")
            loss = np.maximum(loss, 0.0)
            energy_residual = work + stored - prior_energy + loss + implicit + released
            if np.any(abs(energy_residual) > budget):
                raise SolverFailure("coupled tangent work and spring energy do not balance")
            regimes = []
            for i in range(n):
                if normal[i] == 0:
                    regimes.append("open")
                elif mu[i] == 0:
                    regimes.append("frictionless")
                elif _norm(force[i]) >= capacity[i] * (1 - max(tolerance, 32 * eps)) and (
                    stiffness[i] * _norm(gradient[i]) > tolerance * force_scale[i]
                ):
                    regimes.append("slide")
                else:
                    regimes.append("stick")
            result = CoupledTangentResult(
                force,
                delta,
                elastic,
                plastic,
                work,
                stored,
                loss,
                implicit,
                released,
                energy_residual,
                tuple(regimes),
                iterations,
                residual,
                raw,
                prior_energy,
            )
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise SolverFailure(f"coupled tangent solve is not representable: {exc}") from exc
    return result
