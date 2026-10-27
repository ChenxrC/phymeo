"""Fixed-configuration closed-contact impulses with whole-graph physical gates."""

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..core._arrays import float_array
from ..core.capabilities import SolverFailure
from ..core.validation import finite_real
from .friction import _norm, _numerical, solve_friction_disk


def _array(value, name, shape):
    result = float_array(value, name, ndim=len(shape))
    if result.shape != shape or result.dtype != np.float64:
        raise ValueError(f"{name} must be float64 with shape {shape}")
    return result


def _actual(value, shape):
    if not np.isfinite(value).all():
        raise SolverFailure("non-finite actual graph response")
    return _array(value, "graph result", shape)


@dataclass(frozen=True, eq=False)
class ContactImpulseResult:
    states: object
    world_impulses: np.ndarray
    relative_velocity_after: np.ndarray
    contact_basis: np.ndarray
    normal_impulses: np.ndarray
    tangent_impulses: np.ndarray
    regimes: tuple[str, ...]
    iterations: int
    residual_mps: float
    energy_change: float
    response_asymmetry: float
    response_tolerance: float

    def __post_init__(self):
        object.__setattr__(self, "states", MappingProxyType(dict(self.states)))
        for name in (
            "world_impulses",
            "relative_velocity_after",
            "contact_basis",
            "normal_impulses",
            "tangent_impulses",
        ):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if not np.isfinite(value).all():
                raise SolverFailure("non-finite contact result")
            value.flags.writeable = False
            object.__setattr__(self, name, value)


def _basis(normals):
    basis = np.empty((len(normals), 3, 3))
    for index, normal in enumerate(normals):
        length = _norm(normal)
        if abs(length - 1) > 1e-12:
            raise ValueError("contact normals must have unit length")
        normal = normal / length
        axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
        first = np.cross(normal, axis)
        first /= _norm(first)
        basis[index] = np.column_stack((normal, first, np.cross(normal, first)))
    return basis


def _local(basis, values):
    return np.einsum("kji,kj->ki", basis, values)


def _world(basis, values):
    return np.einsum("kij,kj->ki", basis, values)


def _audit(velocity, impulse, normal_response, tangent_scale, static, dynamic, tolerance):
    """Check laws directly at the final velocity, independently of the GS updates."""
    if not np.isfinite(velocity).all() or not np.isfinite(impulse).all():
        raise SolverFailure("non-finite contact audit input")
    residual, regimes = 0.0, []
    for index, (u, p) in enumerate(zip(velocity, impulse, strict=True)):
        kn, kt = normal_response[index], tangent_scale[index]
        projected = max(0.0, p[0] - u[0] / kn)
        residual = max(residual, abs(p[0] - projected) * kn, -u[0], -p[0] * kn)
        speed, tangent = _norm(u[1:]), _norm(p[1:])
        if speed <= tolerance:
            residual = max(residual, (tangent - static[index] * p[0]) * kt)
            regimes.append("open" if p[0] == 0 else "sticking")
        else:
            radius = dynamic[index] * p[0]
            residual = max(residual, abs(tangent - radius) * kt)
            if radius > 0:
                if tangent == 0:
                    residual = max(residual, speed)
                else:
                    residual = max(residual, _norm(u[1:] + speed * (p[1:] / tangent)))
            regimes.append("open" if p[0] == 0 else "sliding")
    return float(residual), tuple(regimes)


@_numerical
def solve_contact_impulses(
    graph,
    normals,
    mu_static,
    mu_dynamic,
    *,
    velocity_tolerance=1e-10,
    response_tolerance=1e-12,
    position_tolerance=1e-9,
    max_iterations=200,
):
    """Solve a zero-restitution impulse problem; no positions or history advance.

    Normals point from the right endpoint toward the left endpoint. Closed
    geometry is required. Final physical gates use a fresh actual port query,
    never the symmetrized response prediction in place of that query.
    """
    count = len(graph.left)
    normals = _array(normals, "normals", (count, 3))
    static = _array(mu_static, "mu_static", (count,))
    dynamic = _array(mu_dynamic, "mu_dynamic", (count,))
    if np.any(dynamic < 0) or np.any(static < dynamic):
        raise ValueError("require 0 <= mu_dynamic <= mu_static")
    velocity_tolerance = finite_real(velocity_tolerance, "velocity_tolerance")
    response_tolerance = finite_real(response_tolerance, "response_tolerance")
    position_tolerance = finite_real(position_tolerance, "position_tolerance")
    if not 0 < velocity_tolerance <= 1e-3 or not 0 < response_tolerance <= 1e-3:
        raise ValueError("velocity/response tolerances must be in (0,1e-3]")
    if position_tolerance < 0:
        raise ValueError("position_tolerance must be non-negative")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, Integral):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    positions = _actual(graph.relative_positions(), (count, 3))
    if any(_norm(value) > position_tolerance for value in positions):
        raise ValueError("contact endpoints must be closed at the same world point")
    basis = _basis(normals)
    free_world = _actual(graph.relative_velocities(), (count, 3))
    if count == 0:
        states = graph.apply_impulses(np.empty((0, 3)))
        audit = graph.audit_candidates(
            states, impulses=np.empty((0, 3)), velocity_tolerance=velocity_tolerance
        )
        if abs(audit.energy_change) > 1e-12:
            raise SolverFailure("empty contact candidate changes kinetic energy")
        return ContactImpulseResult(
            states,
            np.empty((0, 3)),
            free_world,
            basis,
            np.empty(0),
            np.empty((0, 2)),
            (),
            0,
            0.0,
            0.0,
            0.0,
            response_tolerance,
        )

    raw = np.empty((3 * count, 3 * count))
    for column in range(3 * count):
        edge, axis = divmod(column, 3)
        probe = np.zeros((count, 3))
        probe[edge] = basis[edge, :, axis]
        raw[:, column] = _local(basis, _actual(graph.response(probe), (count, 3))).reshape(-1)
    scale = float(np.max(np.abs(raw)))
    if not np.isfinite(raw).all() or scale == 0:
        raise SolverFailure("contact graph has no finite dynamic response")
    asymmetry = float(np.max(np.abs(raw / scale - raw.T / scale)))
    if asymmetry > response_tolerance:
        raise SolverFailure("contact response asymmetry exceeds its declared tolerance")
    matrix = raw * 0.5 + raw.T * 0.5
    spectrum = np.linalg.eigvalsh(matrix / scale)
    if spectrum[0] < -response_tolerance * np.max(np.abs(spectrum)):
        raise SolverFailure("contact response is not positive semidefinite")
    normal_response, tangent_scale, tangents = [], [], []
    for edge in range(count):
        row = 3 * edge
        kn = float(matrix[row, row])
        tangent = matrix[row + 1 : row + 3, row + 1 : row + 3].copy()
        values = np.linalg.eigvalsh(tangent / scale)
        if kn <= 1e-12 * scale or values[-1] <= 0 or values[0] <= 1e-12 * values[-1]:
            raise SolverFailure("contact effective response is singular or ill-conditioned")
        normal_response.append(kn)
        tangent_scale.append(float(values[-1] * scale))
        tangents.append(tangent)
    free = _local(basis, free_world).reshape(-1)
    impulse, velocity = np.zeros(3 * count), free.copy()
    for iteration in range(1, max_iterations + 1):
        for edge in range(count):
            row = 3 * edge
            new_normal = max(0.0, impulse[row] - velocity[row] / normal_response[edge])
            velocity += matrix[:, row] * (new_normal - impulse[row])
            impulse[row] = new_normal
            current = impulse[row + 1 : row + 3].copy()
            free_tangent = velocity[row + 1 : row + 3] - tangents[edge] @ current
            local_scale = max(tangent_scale[edge] * max(static[edge] * new_normal, 1e-12), 1e-12)
            local_tolerance = min(1e-12, (velocity_tolerance / float(local_scale)) / 10)
            solved = solve_friction_disk(
                tangents[edge],
                free_tangent,
                new_normal,
                mu_static=static[edge],
                mu_dynamic=dynamic[edge],
                tolerance=local_tolerance,
            )
            velocity += matrix[:, row + 1 : row + 3] @ (solved.impulse - current)
            impulse[row + 1 : row + 3] = solved.impulse
        velocity = free + matrix @ impulse
        values = impulse.reshape(count, 3)
        residual, _ = _audit(
            velocity.reshape(count, 3),
            values,
            normal_response,
            tangent_scale,
            static,
            dynamic,
            velocity_tolerance,
        )
        if residual > velocity_tolerance:
            continue
        world_impulse = _world(basis, values)
        actual_delta = _actual(graph.response(world_impulse), (count, 3))
        actual_velocity = free_world + actual_delta
        residual, regimes = _audit(
            _local(basis, actual_velocity),
            values,
            normal_response,
            tangent_scale,
            static,
            dynamic,
            velocity_tolerance,
        )
        if residual > velocity_tolerance:
            continue
        energy = float(np.sum(world_impulse * (free_world + actual_delta * 0.5)))
        budget = velocity_tolerance * sum(_norm(row) for row in world_impulse) + 1e-12
        if not np.isfinite(energy) or not np.isfinite(budget) or energy > budget:
            raise SolverFailure("contact impulses increase kinetic energy")
        states = graph.apply_impulses(world_impulse)
        audit = graph.audit_candidates(
            states, impulses=world_impulse, velocity_tolerance=velocity_tolerance
        )
        if any(
            _norm(row) > velocity_tolerance for row in audit.initial_velocities - free_world
        ) or any(
            _norm(row) > velocity_tolerance for row in audit.candidate_velocities - actual_velocity
        ):
            raise SolverFailure("actual candidate velocity disagrees with impulse response")
        if abs(audit.energy_change - energy) > budget or audit.energy_change > budget:
            raise SolverFailure("actual candidate energy disagrees with impulse work")
        actual_velocity, energy = audit.candidate_velocities, audit.energy_change
        residual, regimes = _audit(
            _local(basis, actual_velocity),
            values,
            normal_response,
            tangent_scale,
            static,
            dynamic,
            velocity_tolerance,
        )
        if residual > velocity_tolerance:
            raise SolverFailure("actual candidate violates contact laws")
        return ContactImpulseResult(
            states,
            world_impulse,
            actual_velocity,
            basis,
            values[:, 0],
            values[:, 1:],
            regimes,
            iteration,
            residual,
            energy,
            asymmetry,
            response_tolerance,
        )
    raise SolverFailure("contact impulse graph did not converge within iteration budget")
