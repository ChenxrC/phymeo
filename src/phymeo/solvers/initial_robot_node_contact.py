"""Initial common-point contact quadrature with paired native load replay."""

from dataclasses import replace
from fractions import Fraction

import numpy as np

from ..constraints.contact_impulse import _basis
from ..constraints.friction import _norm
from ..constraints.robot_node_plane import RobotNodePlaneContacts
from ..core.capabilities import SolverFailure
from ..core.model import ParticleModel
from ..core.scene import StageMetrics
from ..dynamics.robot_kinematics import forward_kinematics, point_jacobians
from ..runtime.load_replay import SceneLoadReplay
from ._particle_replay_ledger import FreeParticleLedger
from ._robot_replay_ledger import (
    FrozenRobotLedger,
    audit_affine_geometry,
    dot,
    exact_float,
    rational,
    upper_float,
)
from .particle_plane_contact import _BorrowedProgram, _disk
from .robot_plane_contact import RobotPlaneContactProgram


class InitialRobotNodeContactProgram(RobotPlaneContactProgram):
    """Reuse the robot replay lifecycle/settings contract, with a distinct law.

    Selected node domains must obey free-particle equations. This admits only
    contacts closed at the beginning of each substep; endpoint checks are not CCD.
    """

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
        super().__init__(
            scene,
            base_program,
            (),
            robot_domain=robot_domain,
            registry=registry,
            inner_settings=inner_settings,
            relaxation=relaxation,
            reference_angular_speed=reference_angular_speed,
        )
        domains = {d.name: d for d in scene.domains}
        self.contacts = tuple(contacts)
        for binding in self.contacts:
            if not isinstance(binding, RobotNodePlaneContacts):
                raise TypeError("expected RobotNodePlaneContacts")
            if binding.node_domain not in domains:
                raise ValueError("node contact domain is absent")
            domain = domains[binding.node_domain]
            if domain.family != "particles" or not isinstance(domain.model, ParticleModel):
                raise ValueError("node contacts require a particles domain")
            family = self.registry.get(domain.family)
            if (
                not {"position", "velocity"} <= set(family.state_fields)
                or "force" not in family.input_fields
            ):
                raise ValueError("node position/velocity/force must be registered")
            if (
                np.any(binding.node_indices >= domain.model.particle_count)
                or np.any(binding.robot_indices >= self.model.dof_count)
                or np.any(domain.model.inverse_mass <= 0)
            ):
                raise ValueError("contact indices and positive free-node masses are required")

    @property
    def capabilities(self):
        return replace(
            self.base.capabilities,
            features=self.base.capabilities.features | {"initial_robot_node_replay"},
            dtypes=frozenset({"float64"}),
        )

    def problem_signature(self):
        result = super().problem_signature()
        result["law"] = "initial-common-point-robot-free-node-replay-v1"
        return result

    def create_workspace(self, scene, state, inputs, h, settings, context):
        if self._closed:
            raise RuntimeError("initial robot-node program is closed")
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
        return _InitialPairWorkspace(self, state, inputs, h, settings, inner, context)


class _InitialPairWorkspace:
    def __init__(self, program, state, inputs, h, settings, inner, context):
        self.program, self.initial, self.h, self.settings = program, state, h, settings
        self.domain, self.model = program.robot_domain, program.model
        self.rows = [
            (c.node_domain, int(node), int(link), point, normal, mu)
            for c in program.contacts
            for node, link, point, normal, mu in zip(
                c.node_indices,
                c.robot_indices,
                c.local_points,
                c.local_normals,
                c.friction,
                strict=True,
            )
        ]
        self.indices = np.array([row[2] for row in self.rows], int)
        self.anchors = np.array([row[3] for row in self.rows]).reshape(-1, 3)
        self.local_normals = np.array([row[4] for row in self.rows]).reshape(-1, 3)
        self.mu = np.array([row[5] for row in self.rows])
        self.models = {d.name: d.model for d in program.scene.domains}
        self.node_domains = tuple(sorted({row[0] for row in self.rows}))
        self.node_ledgers = {
            name: FreeParticleLedger(
                self.models[name], state.domains[name], inputs.domains[name], h, settings
            )
            for name in self.node_domains
        }
        self.gap_scale = min(settings.reference_length, h * settings.reference_speed)
        robot = state.domains[self.domain]
        geometry = forward_kinematics(self.model, robot.q)
        audit_affine_geometry(self.model, robot, self.indices, self.anchors, h, settings)
        self.local = []
        for name, node, link, anchor, normal, _ in self.rows:
            local = geometry.rotation[link].T @ (
                state.domains[name].position[node] - geometry.position[link]
            )
            self.local.append(local - normal * float(normal @ (local - anchor)))
        self.local = np.array(self.local).reshape(-1, 3)
        self.robot_points0, self.j0, self.normals0, self.gaps0 = self._geometry(state)
        self.node_points0 = self._node_points(state)[0]
        length = settings.tolerance * self.gap_scale
        if np.any(np.abs(self.gaps0) > length) or np.any(
            np.sum(np.abs(self.node_points0 - self.robot_points0), axis=1) > length
        ):
            raise SolverFailure("initial robot-node contact must be closed")
        self.basis = _basis(self.normals0)
        self.basis[:, :, 0] = self.normals0
        self.ledger = FrozenRobotLedger(
            self.model,
            robot,
            inputs.domains[self.domain],
            h,
            settings,
            program.reference_angular_speed,
        )
        self.responses = []
        for row, jac, basis in zip(self.rows, self.j0, self.basis, strict=True):
            response = jac @ self.ledger.inverse @ jac.T + self.models[row[0]].inverse_mass[
                row[1]
            ] * np.eye(3)
            normal = float(basis[:, 0] @ response @ basis[:, 0])
            tangent = basis[:, 1:].T @ response @ basis[:, 1:]
            rate = float(np.linalg.eigvalsh((tangent + tangent.T) * 0.5)[-1])
            if not np.isfinite(normal + rate) or min(normal, rate) <= 0:
                raise SolverFailure("invalid paired contact response")
            self.responses.append((normal, rate))
        self.impulses = np.zeros((len(self.rows), 3))
        self.trial, self.free_energy, self.velocity_ready = None, None, False
        self.replays, self.native_audit = 0, {}
        self.execution = dict(launches=0, completed_launches=0, uploads=0, downloads=0)
        self.replay = SceneLoadReplay(
            program.scene,
            _BorrowedProgram(program.base),
            state,
            inputs,
            h,
            channels={self.domain: ("effort",), **{name: ("force",) for name in self.node_domains}},
            settings=inner,
            context=context,
            registry=program.registry,
        )

    def _node_points(self, state):
        return (
            np.array([state.domains[r[0]].position[r[1]] for r in self.rows]).reshape(-1, 3),
            np.array([state.domains[r[0]].velocity[r[1]] for r in self.rows]).reshape(-1, 3),
        )

    def _geometry(self, state):
        robot = state.domains[self.domain]
        audit_affine_geometry(
            self.model,
            robot,
            np.r_[self.indices, self.indices],
            np.concatenate((self.local, self.anchors)),
            self.h,
            self.settings,
        )
        geometry = forward_kinematics(self.model, robot.q)
        positions = geometry.position[self.indices] + np.sum(
            geometry.rotation[self.indices] * self.local[:, None, :], axis=2
        )
        anchors = geometry.position[self.indices] + np.sum(
            geometry.rotation[self.indices] * self.anchors[:, None, :], axis=2
        )
        normals = np.sum(geometry.rotation[self.indices] * self.local_normals[:, None, :], axis=2)
        if len(normals):
            normals /= np.hypot.reduce(normals, axis=1)[:, None]
        jac = point_jacobians(self.model, robot.q, self.indices, self.local)[0]
        nodes = self._node_points(state)[0]
        gaps = np.array(
            [
                exact_float(
                    sum(
                        (rational(n[k]) * (rational(x[k]) - rational(a[k])) for k in range(3)),
                        Fraction(0),
                    )
                )
                for n, x, a in zip(normals, nodes, anchors, strict=True)
            ]
        )
        return positions, jac, normals, gaps

    def _world(self):
        return np.sum(self.basis * self.impulses[:, None, :], axis=2)

    def _loads(self):
        world = self._world()
        node = {
            name: [[Fraction(0)] * 3 for _ in range(self.models[name].particle_count)]
            for name in self.node_domains
        }
        generalized = [Fraction(0)] * self.model.dof_count
        for row, jac, p in zip(self.rows, self.j0, world, strict=True):
            for k in range(3):
                node[row[0]][row[1]][k] += rational(p[k])
            for j in range(self.model.dof_count):
                generalized[j] -= dot(jac[:, j], p)
        loads = {self.domain: {"effort": np.array([exact_float(x) for x in generalized])}}
        loads.update(
            {
                name: {"force": np.array([[exact_float(x) for x in row] for row in values])}
                for name, values in node.items()
            }
        )
        return loads, generalized, node

    def _kinetic(self, state):
        return self.ledger.kinetic(state.domains[self.domain].v) + sum(
            (ledger.kinetic(state.domains[name]) for name, ledger in self.node_ledgers.items()),
            Fraction(0),
        )

    def _evaluate(self):
        loads, generalized, node = self._loads()
        trial = self.replay.evaluate(loads)
        if not trial.result.report.accepted:
            raise SolverFailure("native paired replay failed: " + str(trial.result.report.reason))
        state, effective = trial.result.state, trial.effective_inputs
        audit = self.ledger.trial_audit(
            state.domains[self.domain], effective.domains[self.domain], generalized
        )
        if max(audit.values()) > rational(self.settings.tolerance):
            raise SolverFailure("native robot dynamics/input/configuration exceeds tolerance")
        for name, ledger in self.node_ledgers.items():
            for key, value in ledger.require(
                state.domains[name], effective.domains[name], node[name], self.settings.tolerance
            ).items():
                audit[key] = max(audit.get(key, Fraction(0)), value)
        self.native_audit = audit
        self.trial = trial
        if self.free_energy is None:
            self.free_energy = self._kinetic(state)
        self.replays += 1
        for report in trial.result.report.substeps:
            execution = report.diagnostics.get("execution", {})
            for key in self.execution:
                self.execution[key] += execution.get(key, 0)

    def _relative(self):
        state = self.trial.result.state
        robot = state.domains[self.domain]
        node = self._node_points(state)[1]
        return np.array(
            [
                [exact_float(rational(v[k]) - dot(jac[k], robot.v)) for k in range(3)]
                for v, jac in zip(node, self.j0, strict=True)
            ]
        ).reshape(-1, 3)

    def predict(self):
        if self.trial is not None:
            raise RuntimeError("paired contact already predicted")
        self._evaluate()

    def position_sweep(self):
        for i, (kn, kt) in enumerate(self.responses):
            u = self.basis[i].T @ self._relative()[i]
            old = self.impulses[i].copy()
            normal = max(0.0, old[0] - u[0] / kn)
            tangent = _disk(old[1:] - u[1:] / kt, self.mu[i] * normal)
            self.impulses[i] += self.program.relaxation * (np.r_[normal, tangent] - old)
            if not np.array_equal(old, self.impulses[i]):
                self._evaluate()

    def _audit(self):
        state = self.trial.result.state
        positions, jac, normals, gaps = self._geometry(state)
        relative, world = self._relative(), self._world()
        scale = self.settings.reference_speed
        residual = {key: upper_float(value) for key, value in self.native_audit.items()}
        residual.update(
            native=self.trial.result.report.max_residual,
            normal=0.0,
            friction=0.0,
            final_penetration=max((max(-g, 0.0) for g in gaps), default=0.0) / self.gap_scale,
        )
        for i, ((kn, kt), p, basis, v, mu) in enumerate(
            zip(self.responses, self.impulses, self.basis, relative, self.mu, strict=True)
        ):
            u = basis.T @ v
            normal = max(0.0, p[0] - u[0] / kn)
            tangent = _disk(p[1:] - u[1:] / kt, mu * p[0])
            residual["normal"] = max(
                residual["normal"],
                abs(p[0] - normal) * kn / scale,
                max(-p[0], 0.0) * kn / scale,
                max(-u[0], 0.0) / scale,
            )
            residual["friction"] = max(
                residual["friction"],
                _norm(p[1:] - tangent) * kt / scale,
                max(_norm(p[1:]) - mu * p[0], 0.0) * kt / scale,
            )
            speed = _norm(u[1:])
            if speed > self.settings.tolerance * scale:
                residual["friction"] = max(
                    residual["friction"], _norm(p[1:] + mu * p[0] * u[1:] / speed) * kt / scale
                )
        work = sum(
            (
                rational(p[k])
                * (
                    rational(state.domains[row[0]].velocity[row[1], k])
                    - dot(j[k], state.domains[self.domain].v)
                )
                for row, j, p in zip(self.rows, self.j0, world, strict=True)
                for k in range(3)
            ),
            Fraction(0),
        )
        energy = self._kinetic(state) - self.free_energy
        magnitude = sum((abs(rational(v)) for v in world.flat), Fraction(0))
        moment = Fraction(0)
        for node, point, p in zip(self.node_points0, self.robot_points0, world, strict=True):
            delta = [rational(node[k]) - rational(point[k]) for k in range(3)]
            p = list(map(rational, p))
            moment += sum(
                map(
                    abs,
                    (
                        delta[1] * p[2] - delta[2] * p[1],
                        delta[2] * p[0] - delta[0] * p[2],
                        delta[0] * p[1] - delta[1] * p[0],
                    ),
                )
            )
        if magnitude:
            residual["contact_work"] = upper_float(
                max(work, Fraction(0)) / (magnitude * rational(scale))
            )
            residual["frozen_energy"] = upper_float(
                max(energy, Fraction(0)) / (magnitude * rational(scale))
            )
            residual["paired_moment"] = upper_float(moment / (magnitude * rational(self.gap_scale)))
        elif energy != 0:
            raise SolverFailure("zero contact changed native free kinetic energy")
        return (
            StageMetrics(residual, penetration=max((max(-g, 0.0) for g in gaps), default=0.0)),
            positions,
            jac,
            normals,
            gaps,
            relative,
            work,
            energy,
            moment,
        )

    def position_metrics(self):
        return self._audit()[0]

    def prepare_velocity(self):
        if self.velocity_ready or self._audit()[0].max_residual > self.settings.tolerance:
            raise SolverFailure("paired contact is not ready for velocity")
        self.velocity_ready = True

    def velocity_sweep(self):
        pass

    def velocity_metrics(self):
        return self._audit()[0]

    def result(self):
        if not self.velocity_ready:
            raise RuntimeError("paired contact velocity is not prepared")
        return self.trial.result.state

    def diagnostics(self):
        _, positions, jac, normals, gaps, relative, work, energy, moment = self._audit()
        robot_v = self.trial.result.state.domains[self.domain].v
        return dict(
            initial_robot_node_contact=dict(
                initial_robot_points=self.robot_points0,
                initial_node_points=self.node_points0,
                initial_jacobians=self.j0,
                initial_normals=self.normals0,
                initial_gaps=self.gaps0,
                final_robot_points=positions,
                final_jacobians=jac,
                final_normals=normals,
                final_gaps=gaps,
                initial_configuration_relative_velocity_after=relative,
                final_robot_material_velocities=np.array(
                    [[exact_float(dot(row, robot_v)) for row in j] for j in jac]
                ).reshape(-1, 3),
                world_impulses=self._world(),
                world_forces=self._world() / self.h,
                robot_generalized_integral=np.array([exact_float(x) for x in self._loads()[1]]),
                input_integrals=self.trial.integrated_loads,
                paired_moment_error=exact_float(moment),
                contact_work_j=exact_float(work),
                frozen_total_kinetic_change_from_free=exact_float(energy),
                replays=self.replays,
                inner_report=self.trial.result.report,
                coefficient_backend="canonical_host_float64",
                native_backend=self.program.capabilities.backend,
            ),
            execution=self.execution,
        )

    def close(self):
        self.replay.close()
