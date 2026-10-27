"""Native solver stages with distinct configuration-load and impulse operations.

Factories are explicit trusted extensions. Their signature binds all physical
configuration; persistent physical memory must live in registered scene state.
Configuration loads are kg m, instantaneous impulses are kg m/s.
"""

from math import fsum
from typing import Protocol

import numpy as np

from ..core.robot import robot_array
from ..dynamics.robot_dynamics import _finite, _numerical, solve_mass
from ..dynamics.robot_kinematics import forward_kinematics, point_jacobians
from ..runtime.registry import StateRegistry
from ..runtime.robot_family import robot_family
from ..runtime.tetrahedral_family import tetrahedral_family
from ..solvers.cloth import ClothProgram, ClothWorkspace
from ..solvers.robot import RobotProgram, RobotWorkspace
from ..solvers.tetrahedral import TetrahedralProgram, TetrahedralWorkspace
from .point_port import _group_sum
from .ports import ParticlePointPort
from .robot_port import RobotPointPort
from .tetrahedral_port import TetrahedralPointPort

_PARTICLES = StateRegistry().get("particles")


class ParticipantWorkspace(Protocol):
    """One owned substep; no adapter may advance time in an iterative sweep."""

    def predict(self): ...
    def position_sweep(self): ...
    def position_metrics(self): ...
    def point_port(self, indices, local_points=None): ...
    def apply_position_load(self, indices, local_points, loads): ...
    def configuration(self): ...
    def velocity(self): ...
    def prepare_velocity(self): ...
    def apply_velocity_impulses(self, indices, local_points, impulses): ...
    def velocity_sweep(self): ...
    def velocity_metrics(self): ...
    def result(self): ...
    def diagnostics(self): ...
    def close(self): ...


class _Participant:
    def problem_signature(self):
        return self.program.problem_signature()

    @property
    def model(self):
        return self.program.model

    @property
    def capabilities(self):
        return self.program.capabilities

    def _snapshot(self, state, inputs, context):
        if context.backend != "numpy" or context.dtype != np.float64:
            raise ValueError("participant backend or precision unsupported")
        return self.family.snapshot(self.model, state, inputs, context)


class RobotParticipant(_Participant):
    point_role = "articulated"

    def __init__(self, model):
        self.program, self.family = RobotProgram(model), robot_family()

    def create_workspace(self, domain, state, inputs, h, settings, context):
        state, inputs = self._snapshot(state, inputs, context)
        return RobotParticipantWorkspace(RobotWorkspace(self.model, domain, state, inputs, h))


class ClothParticipant(_Participant):
    point_role = "node"

    def __init__(self, model, constraints):
        self.program, self.family = ClothProgram(model, constraints), _PARTICLES

    def create_workspace(self, domain, state, inputs, h, settings, context):
        state, inputs = self._snapshot(state, inputs, context)
        return NodeParticipantWorkspace(
            ClothWorkspace(
                self.model, self.program.constraints, domain, state, inputs, h, settings
            ),
            ParticlePointPort,
        )


class TetrahedralParticipant(_Participant):
    point_role = "node"

    def __init__(self, model):
        self.program, self.family = TetrahedralProgram(model), tetrahedral_family()

    def create_workspace(self, domain, state, inputs, h, settings, context):
        state, inputs = self._snapshot(state, inputs, context)
        return NodeParticipantWorkspace(
            TetrahedralWorkspace(self.model, domain, state, inputs, h, settings),
            TetrahedralPointPort,
        )


class _Workspace:
    def __init__(self, native):
        self.native = native

    def predict(self):
        self.native.predict()

    def position_sweep(self):
        self.native.position_sweep()

    def position_metrics(self):
        return self.native.position_metrics()

    def prepare_velocity(self):
        self.native.prepare_velocity()

    def apply_velocity_impulses(self, indices, local_points, impulses):
        port = self.point_port(indices, local_points)
        self.native.candidate = port.validate_candidate(port.apply_impulses(impulses))

    def velocity_sweep(self):
        self.native.velocity_sweep()

    def velocity_metrics(self):
        return self.native.velocity_metrics()

    def result(self):
        return self.native.result().domains[self.native.domain]

    def diagnostics(self):
        return self.native.diagnostics()

    def close(self):
        self.native.close()


class RobotParticipantWorkspace(_Workspace):
    def __init__(self, native):
        super().__init__(native)
        self.velocity_ready = False

    def predict(self):
        super().predict()
        self.predicted_q = self.configuration()
        self.predicted_v = self.velocity()

    def configuration(self):
        return self.native.candidate.q.copy()

    def velocity(self):
        return self.native.candidate.v.copy()

    def point_port(self, indices, local_points=None):
        return RobotPointPort(
            self.native.model,
            self.native.candidate,
            indices,
            local_points,
            domain=self.native.domain,
        )

    @_numerical
    def world_to_local(self, indices, world_points):
        geometry = forward_kinematics(self.native.model, self.native.candidate.q)
        points = robot_array(world_points, "world_points", ndim=2, columns=3)
        # Validate attachments before indexing geometry.
        self.point_port(indices, points)
        return np.einsum(
            "nji,nj->ni", geometry.rotation[indices], points - geometry.position[indices]
        )

    @_numerical
    def generalized_load(self, indices, local_points, loads):
        port = self.point_port(indices, local_points)
        loads = robot_array(loads, "loads", ndim=2, columns=3)
        if loads.shape != (len(port.indices), 3):
            raise ValueError("load shape must match attachments")
        jacobian = point_jacobians(
            self.native.model, self.native.candidate.q, port.indices, local_points
        )[0]
        contribution = _finite(jacobian * loads[:, :, None])
        return np.array(
            [fsum(contribution[:, :, i].flat) for i in range(self.native.model.dof_count)]
        )

    @_numerical
    def apply_position_load(self, indices, local_points, loads):
        q = self.native.candidate.q
        delta = solve_mass(
            self.native.model, q, self.generalized_load(indices, local_points, loads)
        )
        self.native.candidate.q = _finite(q + delta)

    @_numerical
    def prepare_velocity(self):
        if self.velocity_ready:
            raise RuntimeError("robot participant velocity already prepared")
        self.native.candidate.v = _finite(
            self.predicted_v + (self.native.candidate.q - self.predicted_q) / self.native.h
        )
        self.velocity_ready = True

    def diagnostics(self):
        return {
            **super().diagnostics(),
            "predicted_q": self.predicted_q,
            "predicted_v": self.predicted_v,
        }


class NodeParticipantWorkspace(_Workspace):
    def __init__(self, native, port_type):
        super().__init__(native)
        self.port_type = port_type

    def configuration(self):
        return self.native.candidate.position.copy()

    def velocity(self):
        return self.native.candidate.velocity.copy()

    def point_port(self, indices, local_points=None):
        if local_points is not None:
            points = robot_array(local_points, "local_points", ndim=2, columns=3)
            if points.shape != (len(indices), 3) or np.any(points != 0):
                raise ValueError("node attachments cannot have local offsets")
        return self.port_type(
            self.native.model, self.native.candidate, indices, domain=self.native.domain
        )

    @_numerical
    def apply_position_load(self, indices, local_points, loads):
        port = self.point_port(indices, local_points)
        loads = robot_array(loads, "loads", ndim=2, columns=3)
        if loads.shape != (len(port.indices), 3):
            raise ValueError("load shape must match node attachments")
        grouped = _group_sum(port.indices, loads, self.native.model.particle_count)
        delta = _finite(grouped * self.native.model.inverse_mass[:, None])
        self.native.candidate.position = _finite(self.native.candidate.position + delta)
