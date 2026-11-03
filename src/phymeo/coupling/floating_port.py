"""An immutable floating articulated tree sampled at multiple physical point attachments."""

from math import fsum

import numpy as np

from ..core.robot import robot_array
from ..core.scene import _name
from ..dynamics.floating_robot import FloatingRobotState, floating_kinematics, floating_point_jacobians
from ..dynamics.robot_dynamics import _finite, _numerical, _solve_mass_matrix
from ..solvers.floating_tree import floating_mass_matrix
from .candidate_audit import PointCandidateAudit


class FloatingRobotPointPort:
    """PointPort with one full mass solve for all signed impulses on the same floating system."""

    def __init__(self, model, state, indices, local_points, *, domain="floating_robot"):
        if not isinstance(state, FloatingRobotState):
            raise TypeError("expected a floating robot state")
        self._domain = _name(domain, "port domain")
        state = FloatingRobotState(state.q, state.v)
        self._state = state
        # The public Jacobian validates material body indices, local point count and precision.
        self._linear, _ = floating_point_jacobians(model, state, indices, local_points)
        indices = np.array(indices, dtype=np.int64, copy=True)
        local = robot_array(local_points, "local_points", ndim=2, columns=3)
        geometry = floating_kinematics(model, state)
        self._positions = _finite(geometry.position[indices] + np.einsum("nij,nj->ni", geometry.rotation[indices], local))
        self._indices = indices
        self._mass = floating_mass_matrix(model, state)

    @property
    def domain(self):
        return self._domain

    @property
    def dtype(self):
        return self._state.v.dtype

    @property
    def indices(self):
        return self._indices.copy()

    def snapshot(self):
        return FloatingRobotState(self._state.q, self._state.v)

    def validate_candidate(self, state):
        if not isinstance(state, FloatingRobotState) or not np.array_equal(state.q, self._state.q):
            raise ValueError("an instantaneous floating impulse must preserve the selected configuration")
        return FloatingRobotState(state.q, state.v)

    def positions(self):
        return self._positions.copy()

    @_numerical
    def velocities(self):
        return _finite(self._linear @ self._state.v)

    def _delta(self, impulses):
        impulses = robot_array(impulses, "impulses", ndim=2, columns=3)
        if impulses.shape != self._positions.shape:
            raise ValueError("impulses must contain one world vector per attachment")
        parts = _finite(self._linear * impulses[:, :, None])
        load = np.array([fsum(parts[:, :, i].flat) for i in range(len(self._state.v))])
        return _solve_mass_matrix(self._mass, _finite(load))

    @_numerical
    def response(self, impulses):
        return _finite(self._linear @ self._delta(impulses))

    @_numerical
    def apply_impulses(self, impulses):
        return FloatingRobotState(self._state.q, _finite(self._state.v + self._delta(impulses)))

    @_numerical
    def candidate_audit(self, state):
        candidate = self.validate_candidate(state)
        before, after = self._state.v, candidate.v
        energy = float((after - before) @ self._mass @ (0.5 * before + 0.5 * after))
        return PointCandidateAudit(self.velocities(), _finite(self._linear @ after), energy)
