"""World point motion and matrix-free impulse response across all attached points."""

from dataclasses import dataclass
from functools import wraps
from types import MappingProxyType
from typing import Protocol

import numpy as np

from ..core._arrays import float_array
from ..core.capabilities import SolverFailure
from ..core.model import ParticleModel
from ..core.rigid import RigidBodyInputs, RigidBodyModel, RigidBodyState
from ..core.scene import BackendContext, _name
from ..core.state import ParticleState, StepInputs
from ..core.validation import finite_real
from ..dynamics.rotation import quaternion_matrix
from ..runtime.registry import StateRegistry
from .point_port import _group_sum


def _numerical(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
                result = function(*args, **kwargs)
                # einsum/BLAS need not raise NumPy floating-point warnings.
                if isinstance(result, np.ndarray) and not np.isfinite(result).all():
                    raise FloatingPointError("non-finite point result")
                return result
        except (FloatingPointError, OverflowError) as error:
            raise SolverFailure(f"point response is not representable: {error}") from error

    return checked


class PointPort(Protocol):
    """Trusted extension over a snapshot; response includes shared-object cross terms.

    Geometry and point velocities use world coordinates. response returns m/s for
    supplied world impulses in kg m/s. apply_impulses returns an owned state and
    must not mutate the port, its input, or any external history.
    Certified contact solving additionally requires candidate_audit(state),
    returning PointCandidateAudit from the actual candidate and full-domain KE.
    """

    @property
    def domain(self) -> str: ...

    @property
    def dtype(self) -> np.dtype: ...

    def positions(self) -> np.ndarray: ...
    def velocities(self) -> np.ndarray: ...
    def response(self, impulses: np.ndarray) -> np.ndarray: ...
    def apply_impulses(self, impulses: np.ndarray) -> object: ...
    def snapshot(self) -> object: ...
    def validate_candidate(self, state: object) -> object: ...


class _PointPortBase:
    def _setup(self, family, model, state, inputs, indices, domain):
        self._domain = _name(domain, "port domain")
        self._family = StateRegistry().get(family)
        self._context = BackendContext("numpy", model.inverse_mass.dtype)
        self._state, _ = self._family.snapshot(model, state, inputs, self._context)
        self._family.candidate(model, self._state, self._state, self._context)
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise TypeError("port indices must be a rank-one integer array")
        if np.any(indices < 0) or np.any(indices >= len(model.inverse_mass)):
            raise ValueError("port index outside model")
        self._indices = np.array(indices, dtype=np.int64, copy=True)
        self._indices.flags.writeable = False
        self._model = model

    @property
    def domain(self):
        return self._domain

    @property
    def dtype(self):
        return self._context.dtype

    @property
    def indices(self):
        return self._indices.copy()

    def snapshot(self):
        return self._family.copy_state(self._state, self._context)

    def validate_candidate(self, state):
        candidate = self._family.candidate(self._model, self._state, state, self._context)
        for field in self.configuration_fields:
            if not np.array_equal(getattr(candidate, field), getattr(self._state, field)):
                raise ValueError("instantaneous impulses must preserve configuration")
        return candidate

    def _impulses(self, impulses):
        impulses = float_array(impulses, "impulses", ndim=2, columns=3)
        if len(impulses) != len(self._indices) or impulses.dtype != self.dtype:
            raise ValueError("impulse shape/precision must match the port")
        return impulses.astype(np.float64)


class ParticlePointPort(_PointPortBase):
    configuration_fields = ("position",)

    def __init__(self, model, state, indices, *, domain="particles"):
        if not isinstance(model, ParticleModel):
            raise TypeError("particle port requires ParticleModel")
        model = ParticleModel(model.inverse_mass, model.gravity)
        self._setup(
            "particles",
            model,
            state,
            StepInputs.zeros(model.particle_count, dtype=model.inverse_mass.dtype),
            indices,
            domain,
        )

    def positions(self):
        return self._state.position[self._indices].astype(np.float64)

    def velocities(self):
        return self._state.velocity[self._indices].astype(np.float64)

    def candidate_audit(self, state):
        from .candidate_audit import particle_audit

        return particle_audit(
            self._model, self._state, self.validate_candidate(state), self._indices
        )

    def _delta(self, impulses):
        loads = _group_sum(self._indices, self._impulses(impulses), self._model.particle_count)
        delta = np.zeros_like(loads)
        dynamic = self._model.inverse_mass > 0
        delta[dynamic] = self._model.inverse_mass[dynamic, None].astype(np.float64) * loads[dynamic]
        return delta

    @_numerical
    def response(self, impulses):
        return self._delta(impulses)[self._indices]

    @_numerical
    def apply_impulses(self, impulses):
        velocity = (self._state.velocity.astype(np.float64) + self._delta(impulses)).astype(
            self.dtype
        )
        return ParticleState(self._state.position, velocity)


class RigidPointPort(_PointPortBase):
    """Fixed body-frame offsets; their world point velocities include angular motion."""

    configuration_fields = ("position", "orientation")

    def __init__(self, model, state, indices, local_points, *, domain="rigid_bodies"):
        if not isinstance(model, RigidBodyModel):
            raise TypeError("rigid port requires RigidBodyModel")
        model = RigidBodyModel(model.inverse_mass, model.inertia_diagonal, model.gravity)
        self._setup(
            "rigid_bodies",
            model,
            state,
            RigidBodyInputs.zeros(model.body_count, dtype=model.inverse_mass.dtype),
            indices,
            domain,
        )
        points = float_array(local_points, "local_points", ndim=2, columns=3)
        if len(points) != len(self._indices) or points.dtype != self.dtype:
            raise ValueError("attachment shape/precision must match the port")
        points.flags.writeable = False
        self._local = points

    def _arms(self):
        return np.einsum(
            "nij,nj->ni",
            quaternion_matrix(self._state.orientation[self._indices]),
            self._local.astype(np.float64),
        )

    def candidate_audit(self, state):
        from .candidate_audit import rigid_audit

        return rigid_audit(
            self._model, self._state, self.validate_candidate(state), self._indices, self._local
        )

    @_numerical
    def positions(self):
        return self._state.position[self._indices].astype(np.float64) + self._arms()

    @_numerical
    def velocities(self):
        return self._state.velocity[self._indices].astype(np.float64) + np.cross(
            self._state.angular_velocity[self._indices].astype(np.float64), self._arms()
        )

    def _delta(self, impulses):
        impulses = self._impulses(impulses)
        count, indices = self._model.body_count, self._indices
        # Fixed supports absorb assigned loads; avoid irrelevant zero-inertia arithmetic.
        active = (self._model.inverse_mass[indices] > 0) & np.any(impulses != 0, axis=1)
        torque = np.zeros_like(impulses)
        if np.any(active):
            rotation = quaternion_matrix(self._state.orientation[indices[active]])
            arms = np.einsum("nij,nj->ni", rotation, self._local[active].astype(np.float64))
            torque[active] = np.cross(arms, impulses[active])
        linear_load = _group_sum(indices, impulses, count)
        angular_load = _group_sum(indices, torque, count)
        linear, angular = np.zeros((count, 3)), np.zeros((count, 3))
        dynamic = self._model.inverse_mass > 0
        linear[dynamic] = (
            self._model.inverse_mass[dynamic, None].astype(np.float64) * linear_load[dynamic]
        )
        changed = dynamic & np.any(angular_load != 0, axis=1)
        if np.any(changed):
            rotation = quaternion_matrix(self._state.orientation[changed])
            body_load = np.einsum("nji,nj->ni", rotation, angular_load[changed])
            body_delta = body_load / self._model.inertia_diagonal[changed].astype(np.float64)
            angular[changed] = np.einsum("nij,nj->ni", rotation, body_delta)
        if not np.isfinite(linear).all() or not np.isfinite(angular).all():
            raise FloatingPointError("non-finite body impulse response")
        return linear, angular

    @_numerical
    def response(self, impulses):
        linear, angular = self._delta(impulses)
        response = linear[self._indices].copy()
        active = np.any(angular[self._indices] != 0, axis=1)
        if np.any(active):
            rotation = quaternion_matrix(self._state.orientation[self._indices[active]])
            arms = np.einsum("nij,nj->ni", rotation, self._local[active].astype(np.float64))
            response[active] += np.cross(angular[self._indices[active]], arms)
        return response

    @_numerical
    def apply_impulses(self, impulses):
        linear, angular = self._delta(impulses)
        return RigidBodyState(
            self._state.position,
            self._state.orientation,
            (self._state.velocity.astype(np.float64) + linear).astype(self.dtype),
            (self._state.angular_velocity.astype(np.float64) + angular).astype(self.dtype),
        )


@dataclass(frozen=True)
class PortExchangeResult:
    """Producer-owned states and per-point world load evidence, not a time step."""

    states: object
    impulses: object
    points: object

    def __post_init__(self):
        object.__setattr__(self, "states", MappingProxyType(dict(self.states)))
        for name in ("impulses", "points"):
            values = {}
            for key, value in getattr(self, name).items():
                owned = float_array(value, name, ndim=2, columns=3)
                owned.flags.writeable = False
                values[key] = owned
            object.__setattr__(self, name, MappingProxyType(values))


@_numerical
def exchange_impulses(left: PointPort, right: PointPort, impulses, *, position_tolerance=1e-9):
    """Produce both states atomically for p/-p at matching world points.

    Ports must identify distinct state domains. Internal contacts within a domain
    require one combined response, not two independent state snapshots.
    """
    tolerance = finite_real(position_tolerance, "position_tolerance")
    if tolerance < 0:
        raise ValueError("position_tolerance must be non-negative")
    if left.domain == right.domain or left.dtype != right.dtype:
        raise ValueError("exchange requires distinct domains with matching precision")
    _name(left.domain, "left domain")
    _name(right.domain, "right domain")
    a, b = (
        float_array(port.positions(), "world_points", ndim=2, columns=3) for port in (left, right)
    )
    impulses = float_array(impulses, "impulses", ndim=2, columns=3)
    if a.shape != b.shape or impulses.shape != a.shape or impulses.dtype != left.dtype:
        raise ValueError("port counts and impulse precision must match")
    if np.any(np.hypot.reduce(a.astype(np.float64) - b.astype(np.float64), axis=1) > tolerance):
        raise ValueError("opposite impulses must act at the same world point")
    left_state = left.validate_candidate(left.apply_impulses(impulses.copy()))
    right_state = right.validate_candidate(right.apply_impulses(-impulses))
    return PortExchangeResult(
        {left.domain: left_state, right.domain: right_state},
        {left.domain: impulses, right.domain: -impulses},
        {left.domain: a, right.domain: b},
    )
