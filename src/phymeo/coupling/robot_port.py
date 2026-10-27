"""Owned joint-tree point snapshots with full shared-ancestor impulse response."""

from dataclasses import replace
from math import fsum

import numpy as np

from ..core.robot import RobotInputs, RobotModel, RobotState, robot_array
from ..core.scene import BackendContext, _name
from ..dynamics.robot_dynamics import _finite, _numerical, solve_mass
from ..dynamics.robot_kinematics import forward_kinematics, point_jacobians
from ..runtime.robot_family import robot_family


class RobotPointPort:
    def __init__(self, model, state, indices, local_points, *, domain="robot"):
        if not isinstance(model, RobotModel):
            raise TypeError("robot port requires RobotModel")
        self._model = replace(model)
        self._domain = _name(domain, "port domain")
        self._family = robot_family()
        self._context = BackendContext("numpy", np.float64)
        self._state, _ = self._family.snapshot(
            self._model, state, RobotInputs.zeros(model.dof_count), self._context
        )
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise TypeError("robot port indices must be rank-one integers")
        if np.any(indices < 0) or np.any(indices >= model.dof_count):
            raise ValueError("robot port index outside model")
        self._indices = np.array(indices, dtype=np.int64, copy=True)
        self._local = robot_array(local_points, "local_points", ndim=2, columns=3)
        if len(self._indices) != len(self._local):
            raise ValueError("each attachment must have a link index")
        self._indices.flags.writeable = self._local.flags.writeable = False

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
        if not np.array_equal(candidate.q, self._state.q):
            raise ValueError("instantaneous robot impulses must preserve configuration")
        return candidate

    def _jacobian(self):
        return point_jacobians(self._model, self._state.q, self._indices, self._local)[0]

    def candidate_audit(self, state):
        from .candidate_audit import robot_audit

        return robot_audit(
            self._model, self._state, self.validate_candidate(state), self._indices, self._local
        )

    @_numerical
    def positions(self):
        geometry = forward_kinematics(self._model, self._state.q)
        return geometry.position[self._indices] + np.einsum(
            "nij,nj->ni", geometry.rotation[self._indices], self._local
        )

    @_numerical
    def velocities(self):
        return np.einsum("nki,i->nk", self._jacobian(), self._state.v)

    def _delta(self, impulses):
        impulses = robot_array(impulses, "impulses", ndim=2, columns=3)
        if impulses.shape != self._local.shape:
            raise ValueError("impulse shape must match the robot port")
        jacobian = self._jacobian()
        contributions = _finite(jacobian * impulses[:, :, None])
        # Preserve cancellation across repeated points/components before solving
        # once for the whole tree. Per-point independent solves lose cross terms.
        load = np.array([fsum(contributions[:, :, i].flat) for i in range(self._model.dof_count)])
        return solve_mass(self._model, self._state.q, _finite(load))

    @_numerical
    def response(self, impulses):
        return np.einsum("nki,i->nk", self._jacobian(), self._delta(impulses))

    @_numerical
    def apply_impulses(self, impulses):
        velocity = _finite(self._state.v + self._delta(impulses))
        return RobotState(self._state.q, velocity)
