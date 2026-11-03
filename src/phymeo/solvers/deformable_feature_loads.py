"""Frictionless material-feature loads for an existing discrete particle solver."""

from dataclasses import dataclass
from threading import RLock

import numpy as np

from ..backends.tyx_normal_layer import TyxNormalLayer
from ..constraints.collision_shapes import CollisionModel
from ..core.capabilities import SolverFailure
from ..core.state import ParticleState, StepInputs
from ..core.validation import finite_real
from ..geometry.collision_pipeline import CollisionWorkspace
from ..geometry.deformable_features import (
    DeformableFeatureContact,
    DeformableFeatureContacts,
    _capacity,
    _matches_feature_input,
    query_deformable_features,
)
from .collision_loads import (
    CollisionLoadSystem,
    LoadSite,
    NormalLayer,
    ParticleAttachment,
    _host_law,
    _quadrature,
)

_Point = tuple[float, float, float]


@dataclass(frozen=True)
class DeformableFeatureSite(LoadSite):
    first_velocity_m_s: _Point
    second_velocity_m_s: _Point

    @property
    def force_on_first_n(self):
        return tuple(-v for v in self.force_on_second_n)


@dataclass(frozen=True)
class DeformableFeatureLoad:
    feature: DeformableFeatureContact
    force_n: float
    potential_j: float
    power_w: float
    sites: tuple[DeformableFeatureSite, ...]


@dataclass(frozen=True, eq=False)
class DeformableFeatureLoads:
    """Entry-state contact forces, applied mechanical power, and stored layer energy."""

    geometry: DeformableFeatureContacts
    force_backend: str
    contacts: tuple[DeformableFeatureLoad, ...]
    nodal_force_n: np.ndarray
    potential_j: float
    power_w: float
    particle_dtype: np.dtype

    def step_inputs(self, external=None):
        """Add any external forces to a new StepInputs in the original particle precision."""
        total = self.nodal_force_n
        if external is not None:
            if not isinstance(external, StepInputs) or external.force.shape != total.shape:
                raise ValueError("external particle forces must match the feature sample")
            with np.errstate(over="ignore", invalid="ignore"):
                total = total + external.force
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            data = total.astype(self.particle_dtype)
        if not np.isfinite(data).all() or np.any((data == 0) & (total != 0)):
            raise SolverFailure("feature and external forces exceed particle precision")
        return StepInputs(data)


class DeformableFeatureLoadSystem:
    """Sample a calibrated NormalLayer once per material VT or canonical EE.

    A closest segment has half of the feature measure at each end. Both
    endpoint velocities and both nodal force maps use the same material weights.
    Only a complete sample changes last_sample.
    """

    def __init__(
        self,
        model,
        bindings,
        layer,
        *,
        search_distance_m=None,
        topology_rings=2,
        layer_operator=None,
        max_feature_tests=65536,
        feature_capacity=32768,
    ):
        if not isinstance(model, CollisionModel) or any(
            s.kind != "triangle" or s.radius for s in model.shapes
        ):
            raise ValueError("feature loads require registered zero-thickness material triangles")
        if not isinstance(layer, NormalLayer) or layer.friction:
            raise ValueError("feature loads require a frictionless NormalLayer")
        if layer_operator is not None and not isinstance(layer_operator, TyxNormalLayer):
            raise TypeError("expected a TyxNormalLayer")
        search = (
            layer.width_m
            if search_distance_m is None
            else finite_real(search_distance_m, "search_distance_m")
        )
        if search < layer.width_m:
            raise ValueError("search distance must cover the contact layer")
        self._rings = _capacity(topology_rings, 1, 8, "topology_rings")
        self._test_capacity = _capacity(max_feature_tests, 0, 1_000_000, "max_feature_tests")
        self._capacity = _capacity(feature_capacity, 0, 262144, "feature_capacity")
        # Reuse the ordinary system's globally one-to-one particle/material
        # binding validation and its conversion from ParticleState to face poses.
        self._registered = CollisionLoadSystem(model, bindings, {})
        self._rows = {}
        for shape in model.shapes:
            binding = self._registered.bindings[shape.key]
            if not isinstance(binding, ParticleAttachment):
                raise ValueError("feature loads require particle attachments")
            for n, r in zip(shape.vertex_ids, binding.rows, strict=True):
                self._rows[shape.key.world, shape.key.body, n] = r
        self._layer = layer
        self._search = search
        self._operator = layer_operator
        self._lock = RLock()
        self._last = None

    @property
    def last_sample(self):
        with self._lock:
            return self._last

    def _geometry(self, particles, work, given):
        model = self._registered.model
        if not isinstance(work, CollisionWorkspace) or work._model is not model:
            raise ValueError("workspace belongs to a different collision model")
        with work._lock:
            if work._closed:
                raise RuntimeError("collision workspace is closed")
            paths = self._registered._paths(particles)
            if given is None:
                given = query_deformable_features(
                    model,
                    paths,
                    work,
                    search_distance_m=self._search,
                    contact_distance_m=self._layer.width_m,
                    topology_rings=self._rings,
                    max_feature_tests=self._test_capacity,
                    feature_capacity=self._capacity,
                )
            if not isinstance(given, DeformableFeatureContacts):
                raise TypeError("expected a material feature query")
            if given.status != "ok":
                raise SolverFailure("feature geometry " + given.reason)
            if not _matches_feature_input(
                given, model, paths, self._search, self._layer.width_m, self._rings
            ):
                raise SolverFailure("stale feature binding or changed geometry query")
            if (
                len(given.contacts) > self._capacity
                or given.candidate_features > self._test_capacity
            ):
                raise SolverFailure("feature geometry exceeds force-system capacity")
        return given

    def _site_rows(self, nodes, weights):
        owner = nodes[0][:2]
        return tuple((self._rows[(*owner, n)], w) for n, w in weights if w)

    def _evaluate(self, state, geo):
        layer = self._layer
        groups = []
        sites = []
        data = []
        for feature in geo.contacts:
            if not feature.inside_force_distance:
                continue
            if not feature.force_eligible or feature.normal is None:
                raise SolverFailure("zero or unavailable material separation normal")
            begin = len(sites)
            for point, frac in zip(feature.closest, _quadrature(feature.closest), strict=True):
                first = self._site_rows(feature.first_nodes, point.first_weights)
                second = self._site_rows(feature.second_nodes, point.second_weights)
                va = sum(w * state.velocity[i].astype(float) for i, w in first)
                vb = sum(w * state.velocity[i].astype(float) for i, w in second)
                sites.append((point, frac, va, vb, first, second))
                data.append(
                    (
                        *feature.normal,
                        *(vb - va),
                        layer.width_m,
                        feature.distance_m,
                        layer.stiffness_n_per_m,
                        frac,
                        layer.damping_s_per_m,
                        0.0,
                        layer.slip_speed_m_s,
                    )
                )
            groups.append((feature, begin, len(sites)))
        backend = self._operator.mode if self._operator is not None else "host"
        if data:
            packed = np.asarray(data, float)
            site_precision = np.float32 if self._operator is not None else np.float64
            reference = packed[:, 6:8].astype(site_precision)
            if not np.isfinite(reference).all():
                raise SolverFailure("feature layer values exceed force precision")
            if np.any(reference[:, 0] <= reference[:, 1]):
                raise SolverFailure("feature compression below force precision")
            evaluate = _host_law if self._operator is None else self._operator.evaluate
            values = np.asarray(evaluate(packed), dtype=float)
            if (
                values.shape != (len(sites), 5)
                or not np.isfinite(values).all()
                or np.any(values[:, 1] < 0)
            ):
                raise SolverFailure("invalid material feature force result")
            # Check the directional contract of the returned normal tractions.
            # A missing zeroed device row cannot replace a positive site force.
            precision = np.finfo(site_precision)
            expected_vector = values[:, 1, None] * packed[:, :3]
            bound = 16 * float(precision.eps) * values[:, 1, None] + 16 * float(precision.tiny)
            if np.any(abs(values[:, 2:] - expected_vector) > bound):
                raise SolverFailure("invalid direction of material feature force")
            engaged = (1 - packed[:, 10] * values[:, 0]) > 0
            if np.any(engaged & (values[:, 1] == 0)):
                raise SolverFailure("positive material feature force below output precision")
        forces = np.zeros(state.position.shape, dtype=float)
        contacts = []
        for feature, begin, end in groups:
            rows = []
            for site, (rate, magnitude, *f) in zip(
                sites[begin:end], values[begin:end], strict=True
            ):
                point, frac, va, vb, first, second = site
                force = np.asarray(f)
                for i, w in first:
                    forces[i] -= w * force
                for i, w in second:
                    forces[i] += w * force
                power = float(force @ (vb - va))
                rows.append(
                    DeformableFeatureSite(
                        point,
                        frac,
                        float(rate),
                        float(magnitude),
                        point.first_point,
                        point.second_point,
                        tuple(map(float, force)),
                        power,
                        tuple(map(float, va)),
                        tuple(map(float, vb)),
                    )
                )
            energy = 0.5 * layer.stiffness_n_per_m * (-feature.gap_to_force_m) ** 2
            contacts.append(
                DeformableFeatureLoad(
                    feature,
                    sum(r.force_n for r in rows),
                    energy,
                    sum(r.power_w for r in rows),
                    tuple(rows),
                )
            )
        energy = sum((q.potential_j for q in contacts), 0.0)
        power = sum((q.power_w for q in contacts), 0.0)
        if not np.isfinite(forces).all() or not np.isfinite((energy, power)).all():
            raise SolverFailure("feature nodal force, power or stored energy exceeds precision")
        immutable = np.frombuffer(forces.tobytes(order="C"), dtype=float).reshape(forces.shape)
        return DeformableFeatureLoads(
            geo, backend, tuple(contacts), immutable, energy, power, state.position.dtype
        )

    def sample_particles(self, state, workspace, *, features=None):
        """Publish complete nodal forces at the supplied pose and current velocities."""
        if not isinstance(state, ParticleState):
            raise TypeError("expected a ParticleState")
        with self._lock:
            snapshot = state.copy()
            geometry = self._geometry(snapshot, workspace, features)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                result = self._evaluate(snapshot, geometry)
            self._last = result
            return result
