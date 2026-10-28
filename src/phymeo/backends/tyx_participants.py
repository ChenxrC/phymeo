"""Owned tyx participants and snapshot ports for the unified scene contract."""

from dataclasses import replace

import numpy as np

from ..constraints.distance import distance_residual
from ..constraints.distance_trace import path_diagnostics
from ..core.robot import RobotState, robot_array
from ..core.scene import StageMetrics, _name
from ..core.state import ParticleState
from ..core.validation import finite_real
from ..materials.stvk import deformation_gradient
from ..runtime.program import ProgramCapabilities, own_diagnostics
from ..runtime.registry import StateRegistry
from ..runtime.robot_family import robot_family
from ..runtime.tetrahedral_family import tetrahedral_family
from ..solvers.cloth import ClothProgram
from ..solvers.robot import RobotProgram
from ..solvers.tetrahedral import TetrahedralProgram
from .tyx_math import TyxMath, as_compute, as_index
from .tyx_node_math import TyxNodeMath
from .tyx_robot_math import TyxRobotMath

PRECISION = dict(
    storage_dtype="float64", compute_dtype="float32", host_validation="float64", ordered_serial=True
)
_PARTICLES = StateRegistry().get("particles")


class _Participant:
    def _setup(self, program, family, mode):
        if mode not in TyxMath.MODES:
            raise ValueError("participant requires an explicit tyx mode")
        self.program, self.family, self.mode = program, family, mode
        self.capabilities = ProgramCapabilities(
            "tyx:" + mode, {"float64"}, {family.kind}, program.capabilities.features
        )

    @property
    def model(self):
        return self.program.model

    def problem_signature(self):
        return dict(self.program.problem_signature(), **PRECISION, backend="tyx:" + self.mode)

    def create_workspace(self, domain, state, inputs, h, settings, context):
        if context.backend != "tyx:" + self.mode or context.dtype != np.float64:
            raise ValueError("participant backend or storage precision unsupported")
        self.program.scene.assert_unchanged()
        state, inputs = self.family.snapshot(self.model, state, inputs, context)
        h = finite_real(h, "substep")
        if h <= 0:
            raise ValueError("substep must be positive")
        _name(domain, "domain")
        return self.workspace_type(self, domain, state, inputs, h, settings, context)


class TyxRobotParticipant(_Participant):
    point_role = "articulated"

    def __init__(self, model, mode):
        self._setup(RobotProgram(model), robot_family(), mode)

    @property
    def workspace_type(self):
        return _RobotWorkspace


class TyxClothParticipant(_Participant):
    point_role = "node"

    def __init__(self, model, constraints, mode, *, record_material_path=False):
        self._setup(
            ClothProgram(model, constraints, record_material_path=record_material_path),
            _PARTICLES,
            mode,
        )

    @property
    def workspace_type(self):
        return _NodeWorkspace


class TyxTetrahedralParticipant(_Participant):
    point_role = "node"

    def __init__(self, model, mode):
        self._setup(TetrahedralProgram(model), tetrahedral_family(), mode)

    @property
    def workspace_type(self):
        return _NodeWorkspace


class _Workspace:
    def _setup(self, factory, domain, state, inputs, h, settings, context):
        self.model, self.family, self.context = replace(factory.model), factory.family, context
        self.domain, self.initial, self.inputs = domain, state, inputs
        self.h, self.settings = h, settings
        self.candidate, self.predicted, self.velocity_ready = None, None, False

    def _ready(self):
        self.engine._check_open()
        if self.candidate is None:
            raise RuntimeError("participant prediction has not been performed")

    def result(self):
        self._ready()
        return self.family.candidate(self.model, self.initial, self.candidate, self.context)

    def apply_velocity_impulses(self, indices, local_points, impulses):
        self._ready()
        port = self.point_port(indices, local_points)
        self.candidate = port.validate_candidate(port.apply_impulses(impulses))

    def velocity_sweep(self):
        self._ready()

    def close(self):
        self.engine.close()


class _RobotWorkspace(_Workspace):
    def __init__(self, factory, domain, state, inputs, h, settings, context):
        self._setup(factory, domain, state, inputs, h, settings, context)
        self.engine = TyxRobotMath(self.model, factory.mode)

    def predict(self):
        self.engine._check_open()
        if self.candidate is not None:
            raise RuntimeError("robot prediction already performed")
        engine, state = self.engine, self.initial
        self.applied = engine.generalized(
            state.q, self.inputs.effort, self.inputs.force, self.inputs.torque
        )
        self.bias = engine.bias(state.q, state.v)
        rhs = engine.axpy(self.applied, self.bias, -1.0)
        mass = engine.mass(state.q)
        self.acceleration = engine.solve_spd(mass, rhs).astype(np.float64)
        # Independent equation audit; no host replacement answer enters the step.
        scale = max(
            1.0,
            np.max(np.abs(rhs)),
            np.linalg.norm(mass, np.inf) * np.max(np.abs(self.acceleration)),
        )
        self.equation_residual = float(np.max(np.abs(mass @ self.acceleration - rhs)) / scale)
        v = engine.axpy(state.v, self.acceleration, self.h).astype(np.float64)
        q = engine.axpy(state.q, v, self.h).astype(np.float64)
        self.candidate = RobotState(q, v)
        self.predicted = RobotState(q, v)

    def configuration(self):
        self._ready()
        return self.candidate.q.copy()

    def velocity(self):
        self._ready()
        return self.candidate.v.copy()

    def position_sweep(self):
        self._ready()

    def position_metrics(self):
        self._ready()
        return StageMetrics()

    def velocity_metrics(self):
        self._ready()
        return StageMetrics({"equation": self.equation_residual})

    def point_port(self, indices, local_points=None):
        self._ready()
        return _RobotPort(self, indices, local_points)

    def world_to_local(self, indices, world_points):
        port = self.point_port(indices, world_points)
        return port._transform(world_points, inverse=True)

    def generalized_load(self, indices, local_points, loads):
        return self.point_port(indices, local_points)._generalized(loads)

    def apply_position_load(self, indices, local_points, loads):
        self._ready()
        delta = self.engine.solve(
            self.candidate.q, self.generalized_load(indices, local_points, loads)
        )
        self.candidate.q = self.engine.axpy(self.candidate.q, delta, 1.0).astype(np.float64)

    def prepare_velocity(self):
        self._ready()
        if self.velocity_ready:
            raise RuntimeError("robot velocity already prepared")
        delta = self.engine.axpy(self.candidate.q, self.predicted.q, -1.0)
        self.candidate.v = self.engine.axpy(self.predicted.v, delta, 1 / self.h).astype(np.float64)
        self.velocity_ready = True

    def diagnostics(self):
        self._ready()
        mass = self.engine.mass(self.candidate.q)
        geometry = self.engine.geometry(self.candidate.q)
        return own_diagnostics(
            dict(
                PRECISION,
                acceleration=self.acceleration,
                applied=self.applied,
                bias=self.bias,
                equation_residual=self.equation_residual,
                predicted_q=self.predicted.q,
                predicted_v=self.predicted.v,
                kinetic_energy=float(self.candidate.v @ mass @ self.candidate.v / 2),
                potential_energy=float(
                    -np.sum(self.model.mass[:, None] * geometry.com_position * self.model.gravity)
                ),
                execution=self.engine.stats(),
            )
        )


class _NodeWorkspace(_Workspace):
    def __init__(self, factory, domain, state, inputs, h, settings, context):
        self._setup(factory, domain, state, inputs, h, settings, context)
        self.is_solid = factory.family.kind == "tetrahedra"
        self.record_material_path = not self.is_solid and factory.program.record_material_path
        self.constraints = None if self.is_solid else factory.program.constraints
        self.engine = TyxNodeMath(factory.mode)

    def predict(self):
        self.engine._check_open()
        if self.candidate is not None:
            raise RuntimeError("node prediction already performed")
        total = self.inputs.force
        if self.is_solid:
            self.internal_force, _ = self.engine.material(self.model, self.initial.position)
            total = self.engine.axpy(total, self.internal_force, 1.0).astype(np.float64)
        x, v = self.engine.predict(
            self.initial.position,
            self.initial.velocity,
            total,
            self.model.inverse_mass,
            self.model.gravity,
            self.h,
        )
        self.candidate, self.predicted = ParticleState(x, v), ParticleState(x, v)
        count = 0 if self.constraints is None else len(self.constraints.indices)
        self.alpha, self.multipliers = np.zeros(count), np.zeros(count)
        self.material_frames = []
        if self.constraints is not None:
            self.alpha = self.engine.axpy(self.alpha, self.constraints.compliance, 1 / self.h)
            self.alpha = self.engine.axpy(np.zeros(count), self.alpha, 1 / self.h).astype(
                np.float64
            )
        if self.is_solid:
            # Diagnostic support and equation checks cannot feed predicted state.
            dynamic = self.model.inverse_mass > 0
            self.acceleration = np.zeros_like(total)
            self.acceleration[dynamic] = (
                self.model.gravity + self.model.inverse_mass[dynamic, None] * total[dynamic]
            )
            self.support_force = np.zeros_like(total)
            self.support_force[~dynamic] = (
                -total[~dynamic] - self.model.mass[~dynamic, None] * self.model.gravity
            )
            self.integration_residual = float(
                max(
                    np.max(
                        np.abs((v - self.initial.velocity) - self.h * self.acceleration), initial=0
                    )
                    / self.settings.reference_speed,
                    np.max(np.abs((x - self.initial.position) - self.h * v), initial=0)
                    / self.settings.reference_length,
                )
            )

    def configuration(self):
        self._ready()
        return self.candidate.position.copy()

    def velocity(self):
        self._ready()
        return self.candidate.velocity.copy()

    def point_port(self, indices, local_points=None):
        self._ready()
        return _NodePort(self, indices, local_points)

    def apply_position_load(self, indices, local_points, loads):
        port = self.point_port(indices, local_points)
        delta = port._delta(loads)
        position = self.engine.axpy(self.candidate.position, delta, 1.0).astype(np.float64)
        fixed = self.model.inverse_mass == 0
        position[fixed] = self.initial.position[fixed]
        self.candidate.position = position

    def position_sweep(self):
        self._ready()
        frame = None
        if self.record_material_path:
            frame = np.empty((0 if self.constraints is None else len(self.constraints.indices), 18))
        if self.constraints is not None:
            self.candidate.position, self.multipliers = self.engine.project_distances(
                self.candidate.position,
                self.model.inverse_mass,
                self.constraints,
                self.alpha,
                self.multipliers,
                trace=frame,
            )
        if frame is not None:
            self.material_frames.append(frame)

    def position_metrics(self):
        self._ready()
        if self.is_solid:
            deformation_gradient(self.model, self.candidate.position)
            return StageMetrics({"prediction_equation": self.integration_residual})
        if self.constraints is None:
            return StageMetrics()
        residual, geometric = distance_residual(
            self.candidate.position,
            self.constraints,
            self.alpha,
            self.multipliers,
            self.settings.reference_length,
        )
        return StageMetrics({"distance": residual}, geometric_error=geometric)

    def prepare_velocity(self):
        self._ready()
        if self.velocity_ready:
            raise RuntimeError("node velocity already prepared")
        self.candidate.velocity = self.engine.reconstruct(
            self.predicted.position,
            self.predicted.velocity,
            self.candidate.position,
            self.model.inverse_mass,
            self.h,
        )
        self.velocity_ready = True

    def velocity_metrics(self):
        self._ready()
        return StageMetrics()

    def diagnostics(self):
        self._ready()
        data = dict(PRECISION, distance_multipliers=self.multipliers)
        if self.record_material_path:
            data["material_path"] = path_diagnostics(
                self.material_frames, self.predicted, self.alpha, self.h
            )
        if self.is_solid:
            gradient = deformation_gradient(self.model, self.candidate.position)
            singular = np.linalg.svd(gradient, compute_uv=False)
            _, energy = self.engine.material(self.model, self.candidate.position)
            data.update(
                internal_force=self.internal_force,
                support_force=self.support_force,
                acceleration=self.acceleration,
                integration_residual=self.integration_residual,
                elastic_energy=energy,
                min_det_f=float(np.min(np.linalg.det(gradient))),
                min_singular_value=float(singular.min()),
                max_singular_value=float(singular.max()),
                kinetic_energy=float(
                    np.sum(self.model.mass[:, None] * self.candidate.velocity**2) / 2
                ),
                potential_energy=float(
                    -np.sum(self.model.mass[:, None] * self.candidate.position * self.model.gravity)
                ),
            )
        data["execution"] = self.engine.stats()
        return own_diagnostics(data)


class _Port:
    def _setup(self, workspace, indices, count):
        self._engine, self._model = workspace.engine, workspace.model
        self._family, self._context, self._domain = (
            workspace.family,
            workspace.context,
            workspace.domain,
        )
        self._state = self._family.copy_state(workspace.candidate, self._context)
        self._indices = as_index(indices, "point indices")
        if self._indices.ndim != 1 or np.any(self._indices < 0) or np.any(self._indices >= count):
            raise ValueError("point index outside model")

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

    def _loads(self, loads):
        result = robot_array(loads, "loads", ndim=2, columns=3)
        if result.shape != (len(self._indices), 3):
            raise ValueError("load shape must match attachments")
        return result

    def validate_candidate(self, state):
        candidate = self._family.candidate(self._model, self._state, state, self._context)
        if not np.array_equal(
            getattr(candidate, self.configuration_field),
            getattr(self._state, self.configuration_field),
        ):
            raise ValueError("instantaneous impulses must preserve configuration")
        return candidate


class _RobotPort(_Port):
    configuration_field = "q"

    def candidate_audit(self, state):
        from ..coupling.candidate_audit import robot_audit

        self._engine._check_open()
        return robot_audit(
            self._model, self._state, self.validate_candidate(state), self._indices, self._local
        )

    def __init__(self, workspace, indices, local_points):
        self._setup(workspace, indices, workspace.model.dof_count)
        self._local = robot_array(local_points, "local_points", ndim=2, columns=3)
        if self._local.shape != (len(self._indices), 3):
            raise ValueError("each link index requires a local point")

    def _transform(self, points, inverse=False):
        from ._tyx_point_kernels import transform_points

        self._engine._check_open()
        if not len(self._indices):
            return np.zeros((0, 3))
        geometry = self._engine.geometry(self._state.q)
        (output,) = self._engine.call(
            transform_points,
            len(self._indices),
            int(inverse),
            self._indices,
            as_compute(geometry.position, "origins"),
            as_compute(geometry.rotation, "rotations"),
            as_compute(points, "points"),
            np.zeros_like(self._local, dtype=np.float32),
            outputs=(6,),
        )
        return output.astype(np.float64)

    def positions(self):
        return self._transform(self._local)

    def _jacobian(self):
        return self._engine.jacobians(self._state.q, self._indices, self._local)[0].reshape(
            -1, self._model.dof_count
        )

    def velocities(self):
        return (
            self._engine.matmul(self._jacobian(), self._state.v[:, None])
            .reshape(-1, 3)
            .astype(np.float64)
        )

    def _generalized(self, impulses):
        loads = self._loads(impulses)
        return (
            self._engine.matmul(self._jacobian().T, loads.reshape(-1, 1))
            .reshape(-1)
            .astype(np.float64)
        )

    def _delta(self, impulses):
        return self._engine.solve(self._state.q, self._generalized(impulses))

    def response(self, impulses):
        return (
            self._engine.matmul(self._jacobian(), self._delta(impulses)[:, None])
            .reshape(-1, 3)
            .astype(np.float64)
        )

    def apply_impulses(self, impulses):
        return RobotState(
            self._state.q,
            self._engine.axpy(self._state.v, self._delta(impulses), 1.0).astype(np.float64),
        )


class _NodePort(_Port):
    configuration_field = "position"

    def candidate_audit(self, state):
        from ..coupling.candidate_audit import particle_audit

        self._engine._check_open()
        return particle_audit(
            self._model, self._state, self.validate_candidate(state), self._indices
        )

    def __init__(self, workspace, indices, local_points):
        self._setup(workspace, indices, workspace.model.particle_count)
        if local_points is not None:
            points = robot_array(local_points, "local_points", ndim=2, columns=3)
            if points.shape != (len(self._indices), 3) or np.any(points != 0):
                raise ValueError("node attachments cannot have local offsets")

    def positions(self):
        self._engine._check_open()
        return self._state.position[self._indices].copy()

    def velocities(self):
        self._engine._check_open()
        return self._state.velocity[self._indices].copy()

    def _delta(self, impulses):
        return self._engine.apply_load(
            np.zeros_like(self._state.position),
            self._indices,
            self._loads(impulses),
            self._model.inverse_mass,
        )

    def response(self, impulses):
        return self._delta(impulses)[self._indices].copy()

    def apply_impulses(self, impulses):
        velocity = self._engine.axpy(self._state.velocity, self._delta(impulses), 1.0).astype(
            np.float64
        )
        return ParticleState(self._state.position, velocity)
