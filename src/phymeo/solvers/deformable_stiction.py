"""Compliant material stiction for free vertices against fixed finite triangle interiors."""

from dataclasses import dataclass, replace
from math import hypot
from typing import ClassVar

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.result import StepReport
from ..core.state import StepInputs
from ..core.validation import finite_real, validate_step
from ..geometry.collision_witness import ConvexContactPoint
from ..geometry.deformable_features import DeformableFeatureContact
from ._material_stick_law import return_traction
from .deformable_feature_loads import DeformableFeatureLoads, DeformableFeatureSite
from .deformable_sliding import (
    MaterialSlidingState,
    MaterialSlidingStepper,
    _owned_readonly,
    _tuple,
)


@dataclass(frozen=True)
class MaterialStickHistory:
    key: tuple
    start_tick: int
    active_steps: int
    regime: str
    outward_normal: tuple
    elastic_displacement_m: tuple
    stored_energy_j: float
    plastic_slip_m: float
    tangent_work_j: float


@dataclass(frozen=True, eq=False)
class MaterialStickState(MaterialSlidingState):
    active: tuple[MaterialStickHistory, ...]


@dataclass(frozen=True)
class MaterialStickContact:
    feature: DeformableFeatureContact
    material: ConvexContactPoint
    normal_force_n: float
    tangent_force_on_vertex_n: tuple
    material_displacement_m: tuple
    plastic_increment_m: tuple
    tangent_work_j: float
    plastic_dissipation_j: float
    implicit_dissipation_j: float
    history: MaterialStickHistory


@dataclass(frozen=True, eq=False)
class MaterialStickStep:
    tangent_backend: ClassVar[str] = "host"
    state: MaterialStickState
    report: StepReport
    normal_loads: DeformableFeatureLoads | None
    nodal_tangent_force_n: np.ndarray | None
    contacts: tuple[MaterialStickContact, ...]
    ended: tuple[MaterialStickHistory, ...]

    @property
    def released_spring_energy_j(self):
        """Spring energy discarded when a loaded material episode disappears."""
        return sum((row.stored_energy_j for row in self.ended), 0.0)


def _in_plane(vector, normal):
    return vector - normal * (vector @ normal)


@dataclass(frozen=True)
class _Trial:
    feature: DeformableFeatureContact
    site: DeformableFeatureSite
    vertex: int
    face: tuple
    normal: np.ndarray
    prior: MaterialStickHistory | None
    old_elastic: np.ndarray
    force: np.ndarray
    predicted_delta: np.ndarray
    predicted_velocity: np.ndarray
    sliding: bool


class MaterialStickSlipStepper(MaterialSlidingStepper):
    """Use an implicit tangent spring and a Coulomb yield surface on each loaded VT.

    The free vertex carries the spring displacement relative to its fixed material face.
    Static breakaway uses static_friction*N. Continued forward slip uses the kinetic
    coefficient in NormalLayer.friction. An unloading spring may do positive work.
    Contacts with shared free vertices or ambiguous supports require a coupled solve.
    """

    def __init__(
        self,
        particle_model,
        contact_model,
        bindings,
        layer,
        *,
        static_friction,
        tangent_stiffness_n_per_m,
        **feature_options,
    ):
        super().__init__(particle_model, contact_model, bindings, layer, **feature_options)
        self._static = finite_real(static_friction, "static_friction")
        self._kt = finite_real(tangent_stiffness_n_per_m, "tangent_stiffness_n_per_m")
        if self._static < layer.friction or self._kt <= 0:
            raise ValueError(
                "require static_friction >= kinetic friction and positive tangent stiffness"
            )

    def _checkpoint(self, particles, tick, active):
        return MaterialStickState(
            tick,
            tuple(active),
            _owned_readonly(particles.position),
            _owned_readonly(particles.velocity),
            self._owner,
        )

    def _return_map(self, old, free, inverse, h, normal_force, prior):
        return return_traction(
            old, free, inverse, h, normal_force, prior, self._kt, self._static, self._layer.friction
        )

    def _dynamic_rows(self, trial):
        return (trial.vertex,)

    def _trial(self, load, state, base_force, previous, h):
        feature = load.feature
        if feature.kind != "VT" or len(load.sites) != 1:
            raise SolverFailure("stiction requires a unique vertex and fixed face interior")
        site = load.sites[0]
        ((vertex, _),) = self._normal._site_rows(feature.first_nodes, site.material.first_weights)
        face = self._normal._site_rows(feature.second_nodes, site.material.second_weights)
        inverse = float(self._model.inverse_mass[vertex])
        if (
            inverse <= 0
            or len(face) != 3
            or any(self._model.inverse_mass[row] or np.any(state.velocity[row]) for row, _ in face)
        ):
            raise SolverFailure(
                "stiction requires a free vertex and a fixed, stationary triangle interior"
            )
        normal = -np.asarray(feature.normal)
        normal /= hypot(*normal)
        prior = previous.pop((feature.kind, feature.first_nodes, feature.second_nodes), None)
        old = np.zeros(3) if prior is None else np.asarray(prior.elastic_displacement_m)
        if prior is not None and np.dot(normal, prior.outward_normal) < 1 - 1e-12:
            raise SolverFailure("loaded fixed material face changed side or normal")
        acceleration = self._model.gravity.astype(float) + inverse * base_force[vertex].astype(
            float
        )
        free_velocity = state.velocity[vertex].astype(float) + h * acceleration
        free = h * _in_plane(free_velocity, normal)
        # At a fixed face, d = d_free + h^2/m * f. In the elastic regime f = -k*(old+d).
        force, sliding = self._return_map(old, free, inverse, h, site.force_n, prior)
        predicted = free + h * h * inverse * force
        predicted_velocity = free_velocity + h * inverse * force
        if not np.isfinite([*force, *predicted, *predicted_velocity]).all():
            raise SolverFailure("material tangent prediction exceeds precision")
        return _Trial(
            feature,
            site,
            vertex,
            face,
            normal,
            prior,
            old,
            force,
            predicted,
            predicted_velocity,
            sliding,
        )

    def _accept(self, trial, initial, final, tick):
        vertex, n = trial.vertex, trial.normal
        before, after = initial.position[vertex].astype(float), final.position[vertex].astype(float)
        if any(
            not np.array_equal(final.position[i], initial.position[i]) or np.any(final.velocity[i])
            for i, _ in trial.face
        ):
            raise SolverFailure("particle solver moved the fixed support")
        displacement = after - before
        if trial.feature.distance_m + displacement @ n <= 0:
            raise SolverFailure("loaded material vertex would cross its fixed face plane")
        delta = _in_plane(displacement, n)
        eps = float(np.finfo(initial.position.dtype).eps)
        tolerance = 32 * eps * (hypot(*before) + hypot(*after) + hypot(*trial.predicted_delta))
        velocity = final.velocity[vertex].astype(float)
        # A large absolute position must not hide a stale or inconsistent momentum update.
        vtol = (
            32
            * eps
            * (
                hypot(*initial.velocity[vertex])
                + hypot(*trial.predicted_velocity)
                + hypot(*velocity)
            )
        )
        if (
            hypot(*(velocity - trial.predicted_velocity)) > vtol
            or hypot(*(delta - trial.predicted_delta)) > tolerance
        ):
            raise SolverFailure("particle solver did not realize the tangential mass response")
        return self._record(trial, delta, eps, tolerance, tick)

    def _record(self, trial, delta, eps, tolerance, tick):
        force, n = trial.force, trial.normal
        old = trial.old_elastic
        if trial.sliding:
            elastic = -force / self._kt
            plastic = old + delta - elastic
        else:
            elastic = old + delta
            plastic = np.zeros(3)
        stored = 0.5 * self._kt * float(elastic @ elastic)
        work = float(force @ delta)
        dissipation = -float(force @ plastic)
        increment = elastic - old
        implicit = 0.5 * self._kt * float(increment @ increment)
        prior = trial.prior
        prior_energy = prior.stored_energy_j if prior is not None else 0.0
        residual = work + stored - prior_energy + dissipation + implicit
        energy_tolerance = tolerance * (hypot(*force) + self._kt * hypot(*increment)) + (
            64 * eps * (stored + prior_energy + abs(work) + abs(dissipation) + implicit)
        )
        if dissipation < -energy_tolerance or abs(residual) > energy_tolerance:
            raise SolverFailure(
                "tangent spring work, stored energy and plastic loss do not balance"
            )
        history = MaterialStickHistory(
            (trial.feature.kind, trial.feature.first_nodes, trial.feature.second_nodes),
            prior.start_tick if prior else tick,
            (prior.active_steps if prior else 0) + 1,
            "slide" if trial.sliding else "stick",
            _tuple(n),
            _tuple(elastic),
            stored,
            (prior.plastic_slip_m if prior else 0.0) + hypot(*plastic),
            (prior.tangent_work_j if prior else 0.0) + work,
        )
        if not np.isfinite(
            [
                stored,
                work,
                dissipation,
                implicit,
                history.plastic_slip_m,
                history.tangent_work_j,
                residual,
            ]
        ).all():
            raise SolverFailure("material stiction history exceeds precision")
        return MaterialStickContact(
            trial.feature,
            trial.site.material,
            trial.site.force_n,
            _tuple(force),
            _tuple(delta),
            _tuple(plastic),
            work,
            max(0.0, dissipation),
            implicit,
            history,
        )

    def step(self, state, inputs, dt, workspace, *, particle_solver=None, features=None):
        """Sample entry loads, advance the actual particles and validate before publishing history."""
        if not isinstance(state, MaterialStickState) or state._owner is not self._owner:
            raise ValueError("material stick checkpoint belongs to a different stepper")
        initial, external, h = validate_step(self._model, state.particles, inputs, dt)
        normal = self._normal.sample_particles(initial, workspace, features=features)
        applied = StepInputs(external)
        base = normal.step_inputs(applied)
        previous = {row.key: row for row in state.active}
        nodal = np.zeros_like(normal.nodal_force_n)
        trials, used = [], set()
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for load in normal.contacts:
                if load.force_n == 0:
                    continue
                trial = self._trial(load, initial, base.force, previous, float(h))
                rows = self._dynamic_rows(trial)
                if not used.isdisjoint(rows):
                    raise SolverFailure(
                        "multiple contacts share a free material vertex or support node"
                    )
                used.update(rows)
                nodal[trial.vertex] += trial.force
                for row, weight in trial.face:
                    nodal[row] -= weight * trial.force
                trials.append(trial)
            total = replace(normal, nodal_force_n=normal.nodal_force_n + nodal).step_inputs(applied)
            solver = self._solver if particle_solver is None else particle_solver
            result = solver.step(self._model, initial, total, h)
            if not result.report.accepted:
                return MaterialStickStep(state, result.report, None, None, (), ())
            contacts = tuple(self._accept(t, initial, result.state, state.tick) for t in trials)
            closed = self._checkpoint(result.state, state.tick + 1, (c.history for c in contacts))
        return MaterialStickStep(
            closed,
            result.report,
            normal,
            _owned_readonly(nodal),
            contacts,
            tuple(previous.values()),
        )
