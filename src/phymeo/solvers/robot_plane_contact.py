"""Final-configuration robot/plane impulses through complete native replay."""

from dataclasses import fields, replace
from fractions import Fraction

import numpy as np

from ..constraints.contact_impulse import _basis
from ..constraints.friction import _norm
from ..constraints.robot_plane import RobotPlaneContacts
from ..core.capabilities import SolverFailure
from ..core.robot import RobotModel
from ..core.scene import SceneModel, StageMetrics, StepSettings
from ..core.validation import finite_real
from ..dynamics.robot_kinematics import forward_kinematics, point_jacobians
from ..runtime.load_replay import SceneLoadReplay
from ..runtime.program import ProgramCapabilities
from ..runtime.registry import StateRegistry
from ..runtime.robot_family import robot_family
from ._robot_replay_ledger import (
    FrozenRobotLedger,
    audit_affine_geometry,
    dot,
    exact_float,
    rational,
    upper_float,
)
from .particle_plane_contact import _BorrowedProgram, _disk, _gap
from .robot import robot_energy


class RobotPlaneContactProgram:
    def __init__(
        self,
        scene,
        base_program,
        contacts,
        *,
        robot_domain="robot",
        registry=None,
        inner_settings=None,
        relaxation=1.0,
        reference_angular_speed=1.0,
    ):
        if not isinstance(scene, SceneModel):
            raise TypeError("expected SceneModel")
        caps = base_program.capabilities
        if not isinstance(caps, ProgramCapabilities) or caps.provider_kind != "mathematical":
            raise TypeError("robot replay requires a mathematical program")
        if "float64" not in caps.dtypes or caps.backend not in (
            "numpy",
            "tyx:interpret",
            "tyx:cpu_cpp",
            "tyx:cuda",
        ):
            raise ValueError("robot replay requires supported float64 storage and compute backend")
        if registry is None:
            registry = StateRegistry()
            registry.register(robot_family())
        elif not isinstance(registry, StateRegistry):
            raise TypeError("expected StateRegistry")
        registry = registry.copy()
        bindings = {d.name: d for d in scene.domains}
        if robot_domain not in bindings:
            raise ValueError("robot domain is absent")
        binding = bindings[robot_domain]
        if binding.family != "robot" or not isinstance(binding.model, RobotModel):
            raise ValueError("robot domain requires RobotModel and robot family")
        family = registry.get(binding.family)
        if not {"q", "v"} <= set(family.state_fields) or not {"effort", "force", "torque"} <= set(
            family.input_fields
        ):
            raise ValueError("robot q/v and effort/force/torque must be registered")
        contacts = tuple(contacts)
        for contact in contacts:
            if not isinstance(contact, RobotPlaneContacts):
                raise TypeError("expected RobotPlaneContacts")
            if contact.domain != robot_domain or np.any(contact.indices >= binding.model.dof_count):
                raise ValueError("contact requires valid links in the selected robot")
        if inner_settings is not None and (
            not isinstance(inner_settings, StepSettings) or inner_settings.substeps != 1
        ):
            raise ValueError("inner settings require one substep")
        relaxation = finite_real(relaxation, "relaxation")
        reference_angular_speed = finite_real(reference_angular_speed, "reference_angular_speed")
        if not 0 < relaxation <= 1 or reference_angular_speed <= 0:
            raise ValueError("invalid relaxation or angular speed scale")
        self.scene, self.base, self.contacts = scene, base_program, contacts
        self.robot_domain, self.model, self.registry = robot_domain, binding.model, registry
        self.inner_settings, self.relaxation = inner_settings, relaxation
        self.reference_angular_speed, self._closed = reference_angular_speed, False

    @property
    def capabilities(self):
        return replace(
            self.base.capabilities,
            features=self.base.capabilities.features | {"robot_plane_replay"},
            dtypes=frozenset({"float64"}),
        )

    @property
    def history_fields(self):
        return self.base.history_fields

    def problem_signature(self):
        caps = self.base.capabilities
        capability = {
            f.name: tuple(sorted(v)) if isinstance(v, frozenset) else v
            for f in fields(caps)
            for v in (getattr(caps, f.name),)
        }
        return dict(
            law="frozen-mass-final-jacobian-robot-plane-replay-v1",
            scene=self.scene.identity,
            base=self.base.problem_signature(),
            base_capabilities=capability,
            robot_domain=self.robot_domain,
            contacts=self.contacts,
            inner_settings=self.inner_settings,
            relaxation=self.relaxation,
            reference_angular_speed=self.reference_angular_speed,
        )

    def create_workspace(self, scene, state, inputs, h, settings, context):
        if self._closed:
            raise RuntimeError("robot plane program is closed")
        self.scene.assert_unchanged()
        if scene.identity != self.scene.identity or settings.tolerance <= 0:
            raise ValueError("bound scene and positive tolerance are required")
        inner = (
            replace(settings, substeps=1, record_substeps=False)
            if self.inner_settings is None
            else self.inner_settings
        )
        if any(
            getattr(inner, k) != getattr(settings, k)
            for k in ("tolerance", "reference_length", "reference_speed")
        ):
            raise ValueError("inner tolerance and scales must match outer")
        return _RobotPlaneWorkspace(self, state, inputs, h, settings, inner, context)

    def close(self):
        if not self._closed:
            self._closed = True
            self.base.close()


class _RobotPlaneWorkspace:
    def __init__(self, program, state, inputs, h, settings, inner, context):
        self.program, self.initial, self.h, self.settings = program, state, h, settings
        self.domain, self.model = program.robot_domain, program.model
        self.indices = (
            np.concatenate([c.indices for c in program.contacts])
            if program.contacts
            else np.empty(0, int)
        )
        self.local = (
            np.concatenate([c.local_points for c in program.contacts])
            if program.contacts
            else np.empty((0, 3))
        )
        self.normals = (
            np.concatenate([c.normals for c in program.contacts])
            if program.contacts
            else np.empty((0, 3))
        )
        self.offsets = (
            np.concatenate([c.offsets for c in program.contacts])
            if program.contacts
            else np.empty(0)
        )
        self.friction = (
            np.concatenate([c.friction for c in program.contacts])
            if program.contacts
            else np.empty(0)
        )
        self.basis = _basis(self.normals)
        self.basis[:, :, 0] = self.normals
        self.impulses = np.zeros((len(self.indices), 3))
        self._geometry(state)  # Audit coefficient geometry before freezing M0/b0.
        self.ledger = FrozenRobotLedger(
            self.model,
            state.domains[self.domain],
            inputs.domains[self.domain],
            h,
            settings,
            program.reference_angular_speed,
        )
        self.positions0 = self._geometry(state)[0]
        self.gap_scale = min(settings.reference_length, h * settings.reference_speed)
        if np.any(self._gaps(self.positions0) < -settings.tolerance * self.gap_scale):
            raise SolverFailure("initial robot-plane penetration")
        self.trial, self.free_velocity, self.velocity_ready = None, None, False
        self.replays, self.native_audit = 0, {}
        self.execution = dict(launches=0, completed_launches=0, uploads=0, downloads=0)
        self.replay = SceneLoadReplay(
            program.scene,
            _BorrowedProgram(program.base),
            state,
            inputs,
            h,
            channels={self.domain: ("effort",)},
            settings=inner,
            context=context,
            registry=program.registry,
        )

    def _geometry(self, state):
        robot = state.domains[self.domain]
        audit_affine_geometry(self.model, robot, self.indices, self.local, self.h, self.settings)
        geometry = forward_kinematics(self.model, robot.q)
        positions = geometry.position[self.indices] + np.sum(
            geometry.rotation[self.indices] * self.local[:, None, :], axis=2
        )
        jacobian = point_jacobians(self.model, robot.q, self.indices, self.local)[0]
        velocities = np.array(
            [[exact_float(dot(row, robot.v)) for row in jac] for jac in jacobian]
        ).reshape(-1, 3)
        if not np.isfinite(positions).all():
            raise SolverFailure("non-finite robot contact geometry")
        return positions, jacobian, velocities

    def _gaps(self, positions):
        return np.array(
            [_gap(n, p, b) for n, p, b in zip(self.normals, positions, self.offsets, strict=True)]
        )

    def _world(self):
        return np.sum(self.basis * self.impulses[:, None, :], axis=2)

    def _generalized(self, jacobian):
        world = self._world()
        return [
            sum((dot(jacobian[k, :, j], world[k]) for k in range(len(world))), Fraction(0))
            for j in range(self.model.dof_count)
        ]

    def _evaluate(self, mapping_state):
        expected = self._generalized(self._geometry(mapping_state)[1])
        loads = {self.domain: {"effort": np.array([exact_float(v) for v in expected])}}
        trial = self.replay.evaluate(loads)
        if not trial.result.report.accepted:
            raise SolverFailure("native robot replay failed: " + str(trial.result.report.reason))
        self.native_audit = self.ledger.trial_audit(
            trial.result.state.domains[self.domain],
            trial.effective_inputs.domains[self.domain],
            expected,
        )
        if max(self.native_audit.values()) > rational(self.settings.tolerance):
            raise SolverFailure(
                "native robot dynamics, input or configuration exceeds absolute tolerance"
            )
        self.trial = trial
        if self.free_velocity is None:
            self.free_velocity = trial.result.state.domains[self.domain].v.copy()
        self.replays += 1
        for report in trial.result.report.substeps:
            execution = report.diagnostics.get("execution", {})
            for key in self.execution:
                self.execution[key] += execution.get(key, 0)

    def _mobility(self, jacobian, basis):
        response = jacobian @ self.ledger.inverse @ jacobian.T
        normal = float(basis[:, 0] @ response @ basis[:, 0])
        tangent = basis[:, 1:].T @ response @ basis[:, 1:]
        tangential = float(np.linalg.eigvalsh((tangent + tangent.T) * 0.5)[-1])
        if not np.isfinite(normal + tangential) or normal < 0 or tangential < 0:
            raise SolverFailure("invalid robot point mobility")
        return normal, tangential

    def predict(self):
        if self.trial is not None:
            raise RuntimeError("robot plane prediction already performed")
        self._evaluate(self.initial)

    def position_sweep(self):
        for i in range(len(self.indices)):
            state = self.trial.result.state
            positions, jacobian, velocities = self._geometry(state)
            gap = self._gaps(positions)[i]
            kn, kt = self._mobility(jacobian[i], self.basis[i])
            local_v = self.basis[i].T @ velocities[i]
            old = self.impulses[i].copy()
            if kn == 0:
                if gap < -self.settings.tolerance * self.gap_scale:
                    raise SolverFailure("robot normal contact cannot respond")
                pn = 0.0
            else:
                pn = max(0.0, old[0] - gap / (self.h * kn))
            if kt == 0:
                if _norm(local_v[1:]) > self.settings.tolerance * self.settings.reference_speed:
                    raise SolverFailure("robot tangent contact cannot respond")
                pt = np.zeros(2)
            else:
                pt = _disk(old[1:] - local_v[1:] / kt, self.friction[i] * pn)
            target = np.r_[pn, pt]
            self.impulses[i] = old + self.program.relaxation * (target - old)
            # Refresh even unchanged p: its final-configuration Jacobian can change.
            if not np.array_equal(old, self.impulses[i]) or np.any(self.impulses):
                self._evaluate(state)

    def _audit(self):
        state = self.trial.result.state
        robot = state.domains[self.domain]
        positions, jacobian, velocities = self._geometry(state)
        gaps = self._gaps(positions)
        generalized = self._generalized(jacobian)
        scale = self.settings.reference_speed
        residual = {k: upper_float(v) for k, v in self.native_audit.items()}
        residual.update(
            mapping=upper_float(
                self.ledger.mapping_error(
                    self.trial.effective_inputs.domains[self.domain], generalized
                )
            ),
            native=self.trial.result.report.max_residual,
            normal=0.0,
            friction=0.0,
        )
        for i, (p, v, gap, mu) in enumerate(
            zip(self.impulses, velocities, gaps, self.friction, strict=True)
        ):
            kn, kt = self._mobility(jacobian[i], self.basis[i])
            local_v = self.basis[i].T @ v
            rho = max(kn, kt)
            normal = max(0.0, p[0] - gap / (self.h * kn)) if kn else 0.0
            residual["normal"] = max(
                residual["normal"],
                abs(p[0] - normal) * kn / scale,
                max(-p[0], 0.0) * rho / scale,
                (abs(gap) if p[0] > 0 else max(-gap, 0.0)) / self.gap_scale,
            )
            tangent = _disk(p[1:] - local_v[1:] / kt, mu * p[0]) if kt else np.zeros(2)
            residual["friction"] = max(
                residual["friction"],
                _norm(p[1:] - tangent) * rho / scale,
                max(0.0, _norm(p[1:]) - mu * p[0]) * rho / scale,
            )
            speed = _norm(local_v[1:])
            if speed > self.settings.tolerance * scale:
                residual["friction"] = max(
                    residual["friction"],
                    _norm(p[1:] + mu * p[0] * local_v[1:] / speed) * rho / scale,
                )
        work = sum(
            (g * rational(v) for g, v in zip(generalized, robot.v, strict=True)), Fraction(0)
        )
        energy = self.ledger.kinetic(robot.v) - self.ledger.kinetic(self.free_velocity)
        magnitude = sum((abs(rational(v)) for v in self._world().flat), Fraction(0))
        if magnitude:
            residual["contact_work"] = upper_float(
                max(work, Fraction(0)) / (magnitude * rational(scale))
            )
            residual["frozen_energy"] = upper_float(
                max(energy, Fraction(0)) / (magnitude * rational(scale))
            )
        elif not np.array_equal(robot.v, self.free_velocity):
            raise SolverFailure("zero contact changed the native free velocity")
        penetration = max((max(-g, 0.0) for g in gaps), default=0.0)
        closure = max(
            (abs(g) for g, p in zip(gaps, self.impulses, strict=True) if p[0] > 0), default=0.0
        )
        return (
            StageMetrics(residual, geometric_error=closure, penetration=penetration),
            positions,
            jacobian,
            velocities,
            gaps,
            generalized,
            work,
            energy,
        )

    def position_metrics(self):
        return self._audit()[0]

    def prepare_velocity(self):
        if self.velocity_ready:
            raise RuntimeError("robot plane velocity already prepared")
        if self._audit()[0].max_residual > self.settings.tolerance:
            raise SolverFailure("robot plane contact has not converged")
        self.velocity_ready = True

    def velocity_sweep(self):
        pass

    def velocity_metrics(self):
        return self._audit()[0]

    def result(self):
        if not self.velocity_ready:
            raise RuntimeError("robot plane velocity not prepared")
        return self.trial.result.state

    def diagnostics(self):
        _, positions, jacobian, velocities, gaps, generalized, work, energy = self._audit()
        world = self._world()
        actual_energy = robot_energy(self.model, self.trial.result.state.domains[self.domain])
        return dict(
            robot_plane_contact=dict(
                positions=positions,
                jacobians=jacobian,
                material_velocities=velocities,
                chord_velocities=(positions - self.positions0) / self.h,
                gaps=gaps,
                world_impulses=world,
                world_forces=world / self.h,
                environment_impulses=-world,
                environment_moments=-np.cross(positions, world),
                generalized_integral=np.array([exact_float(v) for v in generalized]),
                input_integrals=self.trial.integrated_loads,
                contact_work_j=exact_float(work),
                frozen_kinetic_change_from_free=exact_float(energy),
                kinetic_at_final_configuration=float(actual_energy[0]),
                potential_at_final_configuration=float(actual_energy[1]),
                replays=self.replays,
                inner_report=self.trial.result.report,
                coefficient_backend="canonical_host_float64",
                native_backend=self.program.capabilities.backend,
            ),
            execution=self.execution,
        )

    def close(self):
        self.replay.close()
