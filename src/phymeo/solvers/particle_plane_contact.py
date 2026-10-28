"""Dynamic particle-plane contact by whole-interval native-solver replay.

The outer solver is host FP64; every candidate comes from the original backend.
Position-level normal complementarity is not a continuous collision certificate.
"""

from dataclasses import fields, replace
from fractions import Fraction
from math import fsum

import numpy as np

from ..constraints.contact_impulse import _basis
from ..constraints.friction import _norm
from ..constraints.particle_plane import ParticlePlaneContacts
from ..core.capabilities import SolverFailure
from ..core.model import ParticleModel
from ..core.scene import SceneModel, StageMetrics, StepSettings
from ..core.validation import finite_real
from ..runtime.load_replay import SceneLoadReplay
from ..runtime.program import ProgramCapabilities
from ..runtime.registry import StateRegistry


def _disk(value, radius):
    length = _norm(value)
    if radius == 0:
        return np.zeros(2)
    return value * (radius / length) if length > radius else value.copy()


def _gap(normal, point, offset):
    # The plane expression is exact for the stored coefficients and coordinates.
    exact = sum(
        (Fraction(float(a)) * Fraction(float(b)) for a, b in zip(normal, point, strict=True)),
        -Fraction(float(offset)),
    )
    result = float(exact)
    if exact != 0 and result == 0:
        raise SolverFailure("plane gap underflow")
    return result


def _input_velocity_error(before, after, h, impulses, inverse_mass):
    # Audit stored inputs and each reported contact impulse before any FP64
    # subtraction, time multiplication or same-node aggregation can hide error.
    error = Fraction(0)
    for axis in range(3):
        delivered = (Fraction(float(after[axis])) - Fraction(float(before[axis]))) * Fraction(h)
        requested = sum((Fraction(float(p[axis])) for p in impulses), Fraction(0))
        error += abs(delivered - requested)
    return Fraction(float(inverse_mass)) * error


def _prediction_roundoff(velocity, gravity, force, inverse_mass, h, unit_roundoff):
    # Conservative supported-range restriction for the canonical free velocity
    # predictor, including input casts; not an error estimate for material solves.
    d, dt = Fraction(float(inverse_mass)), Fraction(h)
    size = sum(
        (
            abs(Fraction(float(v)))
            + abs(dt * Fraction(float(g)))
            + abs(dt * d * Fraction(float(f)))
            for v, g, f in zip(velocity, gravity, force, strict=True)
        ),
        Fraction(0),
    )
    gamma = 16 * unit_roundoff / (1 - 16 * unit_roundoff)
    return gamma * size


class _BorrowedProgram:
    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def close(self):
        # Original per-trial workspaces close normally; the outer factory owns base.
        pass


class ParticlePlaneContactProgram:
    """Static-plane contact on selected particle nodes, wrapping a native program."""

    def __init__(
        self, scene, base_program, contacts, *, registry=None, inner_settings=None, relaxation=1.0
    ):
        if not isinstance(scene, SceneModel):
            raise TypeError("expected SceneModel")
        caps = base_program.capabilities
        if not isinstance(caps, ProgramCapabilities) or caps.provider_kind != "mathematical":
            raise TypeError("plane replay requires a mathematical base program")
        if "float64" not in caps.dtypes:
            raise ValueError("plane replay requires float64 storage")
        precision_bits = {"numpy": 53, "tyx:interpret": 24, "tyx:cpu_cpp": 24, "tyx:cuda": 24}
        if caps.backend not in precision_bits:
            raise ValueError("plane replay requires a registered compute precision")
        self.unit_roundoff = Fraction(1, 2 ** precision_bits[caps.backend])
        if registry is not None and not isinstance(registry, StateRegistry):
            raise TypeError("expected StateRegistry")
        registry = StateRegistry() if registry is None else registry.copy()
        if inner_settings is not None:
            if not isinstance(inner_settings, StepSettings) or inner_settings.substeps != 1:
                raise ValueError("inner settings require exactly one substep")
        relaxation = finite_real(relaxation, "relaxation")
        if not 0 < relaxation <= 1:
            raise ValueError("relaxation must be in (0,1]")
        contacts = tuple(contacts)
        bindings = {d.name: d for d in scene.domains}
        for contact in contacts:
            if not isinstance(contact, ParticlePlaneContacts):
                raise TypeError("expected ParticlePlaneContacts")
            if contact.domain not in bindings:
                raise ValueError("contact domain is not in scene")
            binding = bindings[contact.domain]
            family = registry.get(binding.family)
            if (
                binding.family != "particles"
                or not isinstance(binding.model, ParticleModel)
                or not {"position", "velocity"} <= set(family.state_fields)
                or "force" not in family.input_fields
            ):
                raise ValueError(
                    "plane contact requires registered particle position/velocity/force"
                )
            if np.any(contact.indices >= binding.model.particle_count):
                raise ValueError("contact node outside model")
            if np.any(binding.model.inverse_mass[contact.indices] <= 0):
                raise ValueError("plane contact requires dynamic nodes")
        self.scene, self.base, self.contacts = scene, base_program, contacts
        self.registry, self.inner_settings, self.relaxation = registry, inner_settings, relaxation
        self._closed = False

    @property
    def capabilities(self):
        return replace(
            self.base.capabilities,
            features=self.base.capabilities.features | {"particle_plane_replay"},
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
            law="particle-plane-position-coulomb-replay-v1",
            arithmetic="exact-load-l1-and-gamma16-predictor-v1",
            scene=self.scene.identity,
            base=self.base.problem_signature(),
            base_capabilities=capability,
            contacts=self.contacts,
            inner_settings=self.inner_settings,
            relaxation=self.relaxation,
        )

    def create_workspace(self, scene, state, inputs, h, settings, context):
        if self._closed:
            raise RuntimeError("plane contact program is closed")
        self.scene.assert_unchanged()
        if scene.identity != self.scene.identity:
            raise ValueError("plane contact must use its bound scene")
        if settings.tolerance <= 0:
            raise ValueError("plane contact requires a positive tolerance")
        inner = (
            replace(settings, substeps=1, record_substeps=False)
            if self.inner_settings is None
            else self.inner_settings
        )
        if any(
            getattr(inner, name) != getattr(settings, name)
            for name in ("tolerance", "reference_length", "reference_speed")
        ):
            raise ValueError("inner tolerance and reference scales must match outer settings")
        return _PlaneWorkspace(self, state, inputs, h, settings, inner, context)

    def close(self):
        if not self._closed:
            self._closed = True
            self.base.close()


class _PlaneWorkspace:
    def __init__(self, program, state, inputs, h, settings, inner, context):
        self.program, self.initial, self.h, self.settings = program, state, h, settings
        self.domains, self.indices, normals, offsets, friction, mass = [], [], [], [], [], []
        self.models = {d.name: d.model for d in program.scene.domains}
        for binding in program.contacts:
            for index, normal, offset, mu in zip(
                binding.indices, binding.normals, binding.offsets, binding.friction, strict=True
            ):
                self.domains.append(binding.domain)
                self.indices.append(int(index))
                normals.append(normal)
                offsets.append(offset)
                friction.append(mu)
                mass.append(self.models[binding.domain].inverse_mass[index])
        normal_array = np.array(normals).reshape(-1, 3)
        self.basis = _basis(normal_array)
        # The tangent construction must not renormalize the stored plane equation.
        self.basis[:, :, 0] = normal_array
        self.offsets, self.friction, self.inverse_mass = (
            np.array(offsets),
            np.array(friction),
            np.array(mass),
        )
        self.impulses = np.zeros((len(mass), 3))
        self.positions0, _ = self._points(state)
        self.gap_scale = min(settings.reference_length, h * settings.reference_speed)
        initial_gaps = self._gaps(self.positions0)
        if np.any(initial_gaps < -self.gap_scale * settings.tolerance):
            raise SolverFailure("initial particle-plane penetration")
        self.replays = 0
        self.execution = dict(launches=0, completed_launches=0, uploads=0, downloads=0)
        self.trial, self.velocity_ready = None, False
        self.ledger_residual = self.prediction_residual = 0.0
        self.original_force = {d: inputs.domains[d].force.copy() for d in set(self.domains)}
        self.replay = SceneLoadReplay(
            program.scene,
            _BorrowedProgram(program.base),
            state,
            inputs,
            h,
            channels={d: ("force",) for d in set(self.domains)},
            settings=inner,
            context=context,
            registry=program.registry,
        )

    def _points(self, state):
        positions = [
            state.domains[d].position[i] for d, i in zip(self.domains, self.indices, strict=True)
        ]
        velocities = [
            state.domains[d].velocity[i] for d, i in zip(self.domains, self.indices, strict=True)
        ]
        return np.array(positions).reshape(-1, 3), np.array(velocities).reshape(-1, 3)

    def _gaps(self, positions):
        return np.array(
            [
                _gap(basis[:, 0], point, offset)
                for basis, point, offset in zip(self.basis, positions, self.offsets, strict=True)
            ]
        )

    def _world(self):
        return np.sum(self.basis * self.impulses[:, None, :], axis=2)

    def _local(self, values):
        return np.sum(np.transpose(self.basis, (0, 2, 1)) * values[:, None, :], axis=2)

    def _integrals(self):
        world = self._world()
        loads = {}
        for domain in sorted(set(self.domains)):
            force = np.zeros((self.models[domain].particle_count, 3))
            for node in sorted(
                {i for d, i in zip(self.domains, self.indices, strict=True) if d == domain}
            ):
                rows = [
                    k
                    for k, (d, i) in enumerate(zip(self.domains, self.indices, strict=True))
                    if d == domain and i == node
                ]
                force[node] = [fsum(float(world[k, axis]) for k in rows) for axis in range(3)]
            loads[domain] = {"force": force}
        return loads

    def _evaluate(self):
        requested = self._integrals()
        trial = self.replay.evaluate(requested)
        if not trial.result.report.accepted:
            raise SolverFailure("native replay failed: " + str(trial.result.report.reason))
        self.ledger_residual = self.prediction_residual = 0.0
        world = self._world()
        scale = Fraction(self.settings.reference_speed)
        tolerance = Fraction(self.settings.tolerance)
        for domain in requested:
            model = self.models[domain]
            effective = trial.effective_inputs.domains[domain].force
            for node, d in enumerate(model.inverse_mass):
                if d == 0:
                    continue
                rows = [
                    k
                    for k, (name, index) in enumerate(zip(self.domains, self.indices, strict=True))
                    if name == domain and index == node
                ]
                error = (
                    _input_velocity_error(
                        self.original_force[domain][node], effective[node], self.h, world[rows], d
                    )
                    / scale
                )
                if error > tolerance:
                    raise SolverFailure(
                        "actual contact input integral exceeds absolute load tolerance"
                    )
                self.ledger_residual = max(self.ledger_residual, float(error))
                budget = (
                    _prediction_roundoff(
                        self.initial.domains[domain].velocity[node],
                        model.gravity,
                        effective[node],
                        d,
                        self.h,
                        self.program.unit_roundoff,
                    )
                    / scale
                )
                if budget > tolerance:
                    raise SolverFailure(
                        "native prediction roundoff budget exceeds absolute tolerance"
                    )
                self.prediction_residual = max(self.prediction_residual, float(budget))
        self.trial = trial
        self.replays += 1
        for report in trial.result.report.substeps:
            execution = report.diagnostics.get("execution", {})
            for name in self.execution:
                self.execution[name] += execution.get(name, 0)

    def predict(self):
        if self.trial is not None:
            raise RuntimeError("plane prediction already performed")
        self._evaluate()

    def position_sweep(self):
        for index, d in enumerate(self.inverse_mass):
            positions, velocities = self._points(self.trial.result.state)
            gap = self._gaps(positions)[index]
            local = self._local(velocities)[index]
            old = self.impulses[index].copy()
            normal = max(0.0, old[0] - gap / (self.h * d))
            tangent = _disk(old[1:] - local[1:] / d, self.friction[index] * normal)
            target = np.r_[normal, tangent]
            proposed = old + self.program.relaxation * (target - old)
            if not np.array_equal(old, proposed):
                self.impulses[index] = proposed
                self._evaluate()

    def _audit(self):
        positions, velocities = self._points(self.trial.result.state)
        gaps, local = self._gaps(positions), self._local(velocities)
        displacement_rate = (positions - self.positions0) / self.h
        scale, tolerance = self.settings.reference_speed, self.settings.tolerance
        residual = dict(
            normal=0.0,
            friction=0.0,
            motion=0.0,
            contact_work=0.0,
            input_integral=self.ledger_residual,
            prediction_roundoff=self.prediction_residual,
            native=self.trial.result.report.max_residual,
        )
        work = np.sum(self._world() * displacement_rate, axis=1)
        for i, (p, u, gap, d, mu) in enumerate(
            zip(self.impulses, local, gaps, self.inverse_mass, self.friction, strict=True)
        ):
            normal = max(0.0, p[0] - gap / (self.h * d))
            residual["normal"] = max(
                residual["normal"],
                abs(p[0] - normal) * d / scale,
                max(-p[0], 0.0) * d / scale,
                (abs(gap) if p[0] > 0 else max(-gap, 0.0)) / self.gap_scale,
            )
            projected = _disk(p[1:] - u[1:] / d, mu * p[0])
            residual["friction"] = max(
                residual["friction"],
                _norm(p[1:] - projected) * d / scale,
                max(0.0, _norm(p[1:]) - mu * p[0]) * d / scale,
            )
            speed = _norm(u[1:])
            if speed > tolerance * scale:
                residual["friction"] = max(
                    residual["friction"], _norm(p[1:] + mu * p[0] * u[1:] / speed) * d / scale
                )
            residual["motion"] = max(
                residual["motion"], _norm(velocities[i] - displacement_rate[i]) / scale
            )
            magnitude = _norm(self._world()[i])
            if magnitude > 0:
                residual["contact_work"] = max(
                    residual["contact_work"], max(0.0, work[i]) / magnitude / scale
                )
        penetration = max((max(-g, 0.0) for g in gaps), default=0.0)
        closure = max(
            (abs(g) for g, p in zip(gaps, self.impulses, strict=True) if p[0] > 0), default=0.0
        )
        return (
            StageMetrics(residual, geometric_error=closure, penetration=penetration),
            gaps,
            velocities,
            work,
        )

    def position_metrics(self):
        return self._audit()[0]

    def prepare_velocity(self):
        if self.velocity_ready:
            raise RuntimeError("plane velocity already prepared")
        if self._audit()[0].max_residual > self.settings.tolerance:
            raise SolverFailure("plane contact has not converged")
        self.velocity_ready = True

    def velocity_sweep(self):
        pass

    def velocity_metrics(self):
        return self._audit()[0]

    def result(self):
        if not self.velocity_ready:
            raise RuntimeError("plane velocity not prepared")
        return self.trial.result.state

    def diagnostics(self):
        _, gaps, velocities, work = self._audit()
        world = self._world()
        return dict(
            plane_contact=dict(
                world_impulses=world,
                world_forces=world / self.h,
                environment_impulses=-world,
                input_integrals=self.trial.integrated_loads,
                gaps=gaps,
                relative_velocities=velocities,
                contact_work_j=fsum(float(w) for w in work),
                replays=self.replays,
                inner_report=self.trial.result.report,
            ),
            execution=self.execution,
        )

    def close(self):
        self.replay.close()
