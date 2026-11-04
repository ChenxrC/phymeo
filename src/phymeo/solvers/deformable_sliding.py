"""Regularized kinetic sliding and committed material-address histories for discrete VT/EE sites."""

from dataclasses import dataclass, field, replace
from math import hypot
from typing import ClassVar

import numpy as np

from ..constraints.collision_shapes import CollisionModel
from ..core.capabilities import SolverFailure
from ..core.model import ParticleModel
from ..core.result import StepReport
from ..core.state import ParticleState, StepInputs
from ..core.validation import validate_step
from ..geometry.collision_witness import ConvexContactPoint
from ..geometry.deformable_features import DeformableFeatureContact
from .collision_loads import NormalLayer
from .deformable_feature_loads import DeformableFeatureLoads, DeformableFeatureLoadSystem
from .semi_implicit import SemiImplicitSolver


def _owned_readonly(values):
    values = np.ascontiguousarray(values)
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


def _tuple(values):
    return tuple(map(float, values))


@dataclass(frozen=True)
class MaterialSlipHistory:
    """One uninterrupted episode at a canonical pair of material features."""

    key: tuple
    start_tick: int
    active_steps: int
    slip_distance_m: float
    normal_impulse_ns: float
    tangent_impulse_on_second_ns: tuple
    tangent_work_j: float


@dataclass(frozen=True, eq=False)
class MaterialSlidingState:
    """An owned checkpoint. Reading particles gives a detached mutable ParticleState."""

    tick: int
    active: tuple[MaterialSlipHistory, ...]
    _position: np.ndarray = field(repr=False)
    _velocity: np.ndarray = field(repr=False)
    _owner: object = field(repr=False)

    @property
    def particles(self):
        return ParticleState(self._position, self._velocity)


@dataclass(frozen=True)
class MaterialSlidingContact:
    feature: DeformableFeatureContact
    material: ConvexContactPoint
    phase: str
    normal_force_n: float
    force_on_second_n: tuple
    tangent_force_on_second_n: tuple
    relative_tangent_velocity_m_s: tuple
    tangent_displacement_m: tuple
    entry_tangent_power_w: float
    tangent_work_j: float
    history: MaterialSlipHistory

    @property
    def force_on_first_n(self):
        return tuple(-value for value in self.force_on_second_n)


@dataclass(frozen=True, eq=False)
class MaterialSlidingStep:
    tangent_backend: ClassVar[str] = "host"
    state: MaterialSlidingState
    report: StepReport
    normal_loads: DeformableFeatureLoads | None
    nodal_tangent_force_n: np.ndarray | None
    contacts: tuple[MaterialSlidingContact, ...]
    ended: tuple[MaterialSlipHistory, ...]


class MaterialSlidingStepper:
    """Advance independent particle states with finite-range normal and tangential tractions.

    A unique VT or EE site uses f_t = -mu * f_n * v_t / max(|v_t|, slip_speed).
    Forces act at both original material addresses. A finite gap can therefore
    give the tangential pair a moment. This kinetic law has no static anchoring.
    A history ends when its loaded key disappears from a sampled entry state.
    """

    def __init__(
        self,
        particle_model,
        contact_model,
        bindings,
        layer,
        *,
        search_distance_m=None,
        topology_rings=2,
        layer_operator=None,
        max_feature_tests=65536,
        feature_capacity=32768,
    ):
        if not isinstance(particle_model, ParticleModel) or not isinstance(
            contact_model, CollisionModel
        ):
            raise TypeError("expected ParticleModel and CollisionModel")
        if not isinstance(layer, NormalLayer):
            raise TypeError("expected a calibrated NormalLayer")
        self._model = particle_model
        self._layer = layer
        self._owner = object()
        self._solver = SemiImplicitSolver()
        self._normal = DeformableFeatureLoadSystem(
            contact_model,
            bindings,
            replace(layer, friction=0.0),
            search_distance_m=search_distance_m,
            topology_rings=topology_rings,
            layer_operator=layer_operator,
            max_feature_tests=max_feature_tests,
            feature_capacity=feature_capacity,
        )

    def _checkpoint(self, particles, tick, active):
        return MaterialSlidingState(
            tick,
            tuple(active),
            _owned_readonly(particles.position),
            _owned_readonly(particles.velocity),
            self._owner,
        )

    def initial_state(self, particles):
        zeros = StepInputs.zeros(self._model.particle_count, dtype=self._model.inverse_mass.dtype)
        initial, _, _ = validate_step(self._model, particles, zeros, 1.0)
        return self._checkpoint(initial, 0, ())

    def _tractions(self, normal_loads):
        nodal = np.zeros_like(normal_loads.nodal_force_n)
        prepared = []
        for load in normal_loads.contacts:
            if load.force_n == 0:
                continue
            if len(load.sites) != 1:
                raise SolverFailure("material sliding requires unique point sites")
            site = load.sites[0]
            feature = load.feature
            n = np.asarray(feature.normal)
            relative = np.asarray(site.second_velocity_m_s) - site.first_velocity_m_s
            tangent = relative - n * ((n @ relative) / (n @ n))
            force = (
                -self._layer.friction
                * site.force_n
                * tangent
                / max(hypot(*tangent), self._layer.slip_speed_m_s)
            )
            first = self._normal._site_rows(feature.first_nodes, site.material.first_weights)
            second = self._normal._site_rows(feature.second_nodes, site.material.second_weights)
            for i, w in first:
                nodal[i] -= w * force
            for i, w in second:
                nodal[i] += w * force
            prepared.append((feature, site, n, tangent, force, first, second))
        if not np.isfinite(nodal).all():
            raise SolverFailure("tangential nodal load exceeds precision")
        return nodal, prepared

    def step(self, state, inputs, dt, workspace, *, particle_solver=None, features=None):
        """Return an accepted new checkpoint or the original on a rejected particle step.

        A geometry, arithmetic or dissipativity error raises without advancing the input.
        Work and slip use actual accepted nodal displacements, interpolated with this
        step's entry material weights on both sides. They exclude changes of closest-point address.
        """
        if not isinstance(state, MaterialSlidingState) or state._owner is not self._owner:
            raise ValueError("sliding checkpoint belongs to a different stepper")
        initial, external, h = validate_step(self._model, state.particles, inputs, dt)
        normal = self._normal.sample_particles(initial, workspace, features=features)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            nodal, prepared = self._tractions(normal)
            loads = (
                normal
                if self._layer.friction == 0
                else replace(normal, nodal_force_n=normal.nodal_force_n + nodal)
            )
            total = loads.step_inputs(StepInputs(external))
            solver = self._solver if particle_solver is None else particle_solver
            result = solver.step(self._model, initial, total, h)
            if not result.report.accepted:
                return MaterialSlidingStep(state, result.report, None, None, (), ())
            dx = result.state.position.astype(float) - state._position.astype(float)
            previous = {record.key: record for record in state.active}
            current = []
            contacts = []
            for feature, site, n, tangent, force, first, second in prepared:
                delta = sum(w * dx[i] for i, w in second) - sum(w * dx[i] for i, w in first)
                slip = delta - n * ((n @ delta) / (n @ n))
                work = float(force @ slip)
                tolerance = (
                    32 * np.finfo(initial.position.dtype).eps * hypot(*force) * hypot(*delta)
                )
                if work > tolerance:
                    raise SolverFailure(
                        "kinetic friction would add work on the accepted slip; reduce dt"
                    )
                key = (feature.kind, feature.first_nodes, feature.second_nodes)
                prior = previous.pop(key, None)
                impulse = force * float(h)
                if prior is not None:
                    impulse += prior.tangent_impulse_on_second_ns
                history = MaterialSlipHistory(
                    key,
                    prior.start_tick if prior else state.tick,
                    (prior.active_steps if prior else 0) + 1,
                    (prior.slip_distance_m if prior else 0.0) + hypot(*slip),
                    (prior.normal_impulse_ns if prior else 0.0) + site.force_n * float(h),
                    _tuple(impulse),
                    (prior.tangent_work_j if prior else 0.0) + work,
                )
                if not np.isfinite(
                    (
                        history.slip_distance_m,
                        history.normal_impulse_ns,
                        *impulse,
                        history.tangent_work_j,
                    )
                ).all():
                    raise SolverFailure("material sliding history exceeds precision")
                current.append(history)
                contacts.append(
                    MaterialSlidingContact(
                        feature,
                        site.material,
                        "continued" if prior else "entered",
                        site.force_n,
                        _tuple(np.asarray(site.force_on_second_n) + force),
                        _tuple(force),
                        _tuple(tangent),
                        _tuple(slip),
                        float(force @ tangent),
                        work,
                        history,
                    )
                )
        return MaterialSlidingStep(
            self._checkpoint(result.state, state.tick + 1, current),
            result.report,
            normal,
            _owned_readonly(nodal),
            tuple(contacts),
            tuple(previous.values()),
        )
