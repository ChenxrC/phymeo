"""Explicit material-station identity and retryable tangential-state transactions.

The caller supplies stable tokens, actual bilateral addresses, and a material
frame. This module neither matches nearest contacts nor computes traction or
material displacement. In particular, geometric witness drift is not slip.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import atan2, fsum, hypot, isclose, pi
from numbers import Integral

import numpy as np

from ..constraints.collision_shapes import CollisionKey
from ..core.validation import finite_real
from ..runtime.contact_events import ContactOwner


@dataclass(frozen=True, order=True)
class StationKey:
    """A caller-declared station token, scoped to one world and one history owner."""

    world: int
    token: int


@dataclass(frozen=True)
class MaterialAddress:
    """An actual surface address, independent of transient sampler row order.

    feature contains the canonical material vertex IDs of the declared feature.
    Rigid/link points additionally retain their actual local surface coordinate.
    A non-None chart_id explicitly asserts a validated common material chart;
    it does not prove continuity or supply the displacement needed by a solver.
    The producer must change topology_revision when that material chart changes.
    """

    owner: ContactOwner
    shape: CollisionKey
    topology_revision: int
    feature: tuple[int, ...]
    weights: tuple[tuple[int, float], ...]
    local_point_m: tuple[float, float, float] | None
    chart_id: int | None = None


@dataclass(frozen=True)
class StationSample:
    """Input-state material sample. frame_world columns are t1, t2, normal.

    Inputs are copied and validated by prepare, not trusted merely because this
    declaration is frozen. A committed record's frame is its load-sample frame,
    not an assertion that the body remained in that frame throughout the step.
    """

    key: StationKey
    first: MaterialAddress
    second: MaterialAddress
    normal: tuple[float, float, float]
    frame_world: tuple
    normal_force_n: float


@dataclass(frozen=True)
class TangentUpdate:
    """Solver-produced state in the current trial's world material frame.

    Plastic slip and tangent work are cumulative within this episode; work may
    be positive on spring unloading. B validates storage, not physical balance.
    Supplying stiffness also checks E = k_t * |elastic|^2 / 2.
    """

    elastic_displacement_m: tuple[float, float, float]
    stored_energy_j: float
    plastic_slip_m: float
    tangent_work_j: float
    regime: str
    stiffness_n_per_m: float | None = None


@dataclass(frozen=True, order=True)
class StationIdentity:
    key: StationKey
    generation: int


@dataclass(frozen=True)
class TangentRecord:
    identity: StationIdentity
    sample: StationSample
    update: TangentUpdate


@dataclass(frozen=True)
class EndedStation:
    identity: StationIdentity
    reason: str
    released_energy_j: float


@dataclass(frozen=True)
class PreparedStation:
    identity: StationIdentity
    sample: StationSample
    continuity: str
    prior: TangentUpdate | None


@dataclass(frozen=True)
class TangentHistoryState:
    _owner: object = field(repr=False)
    step_index: int = 0
    sample_time_s: float = 0.0
    commit_time_s: float = 0.0
    active: tuple[TangentRecord, ...] = ()
    generations: tuple[StationIdentity, ...] = ()
    ended: tuple[EndedStation, ...] = ()
    dropped_spring_energy_j: float = 0.0


@dataclass(frozen=True, eq=False)
class TangentTrial:
    _owner: object = field(repr=False)
    _base: TangentHistoryState = field(repr=False)
    _times: tuple = field(repr=False)
    step_index: int
    sample_time_s: float
    commit_time_s: float
    stations: tuple[PreparedStation, ...]
    generations: tuple[StationIdentity, ...]
    ended: tuple[EndedStation, ...]
    dropped_spring_energy_j: float
    keys: tuple[StationKey, ...]
    frames_world: np.ndarray
    normal_force_n: np.ndarray
    elastic_displacement_m: np.ndarray
    elastic_tangent_m: np.ndarray


def _index(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError("invalid " + name)
    return int(value)


def _vector(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("invalid " + name)
    return tuple(map(float, a))


def _readonly(value, shape):
    a = np.asarray(value, dtype=np.float64).reshape(shape)
    return np.frombuffer(a.tobytes(), dtype=np.float64).reshape(shape)


def _address(value, world):
    if not isinstance(value, MaterialAddress) or not isinstance(value.owner, ContactOwner):
        raise TypeError("expected MaterialAddress with an actual ContactOwner")
    owner, shape = value.owner, value.shape
    if not isinstance(shape, CollisionKey) or shape.world != world or owner.world != world:
        raise ValueError("station and bilateral material owners must share a world")
    _index(owner.world, "owner world")
    _index(owner.body, "physical owner")
    frames = {
        "rigid": "rigid_com", "particle": "material_vertices",
        "robot_link": "robot_model_link", "robot_root": "robot_root",
    }
    if owner.kind not in frames or owner.local_frame != frames[owner.kind]:
        raise ValueError("unsupported physical owner material frame")
    if owner.kind in {"rigid", "particle"} and owner.body != shape.body:
        raise ValueError("material shape differs from its physical owner")
    if owner.kind == "robot_link":
        _index(owner.link_index, "robot link")
        if not isinstance(owner.link_name, str) or not owner.link_name:
            raise ValueError("missing computational link name")
    if owner.kind == "robot_root" and (owner.link_index != -1 or not owner.link_name):
        raise ValueError("invalid fixed root owner")
    feature = tuple(sorted(_index(i, "material feature vertex") for i in value.feature))
    if not feature or len(set(feature)) != len(feature):
        raise ValueError("feature requires distinct material vertex IDs")
    weights = tuple(sorted(
        (_index(i, "material weight vertex"), finite_real(w, "material weight"))
        for i, w in value.weights
    ))
    if (not weights or len({i for i, _ in weights}) != len(weights)
        or any(i not in feature or w < 0 for i, w in weights)
        or not isclose(fsum(w for _, w in weights), 1.0, rel_tol=0, abs_tol=1e-10)):
        raise ValueError("material weights require a valid partition of the declared feature")
    local = value.local_point_m
    if owner.kind == "particle":
        if local is not None:
            raise ValueError("particle material addresses use vertex weights, not a rigid frame")
    else:
        local = _vector(local, "local material point")
    chart = None if value.chart_id is None else _index(value.chart_id, "material chart")
    return MaterialAddress(
        owner, shape, _index(value.topology_revision, "topology revision"),
        feature, weights, local, chart,
    )


def _sample(value):
    if not isinstance(value, StationSample) or not isinstance(value.key, StationKey):
        raise TypeError("expected StationSample with a declared StationKey")
    key = StationKey(_index(value.key.world, "world"), _index(value.key.token, "station token"))
    first, second = _address(value.first, key.world), _address(value.second, key.world)
    if not first.shape < second.shape:
        raise ValueError("bilateral station shapes must be in canonical order")
    n = np.asarray(_vector(value.normal, "station normal"))
    frame = np.asarray(value.frame_world, dtype=float)
    if (frame.shape != (3, 3) or not np.isfinite(frame).all()
        or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-10, rtol=0)
        or not isclose(float(np.linalg.det(frame)), 1.0, rel_tol=0, abs_tol=1e-10)
        or not np.allclose(frame[:, 2], n, atol=1e-10, rtol=0)):
        raise ValueError("material frame must be proper orthonormal and normal-aligned")
    force = finite_real(value.normal_force_n, "normal force")
    if force < 0:
        raise ValueError("normal force must be nonnegative")
    return StationSample(key, first, second, tuple(n), tuple(map(tuple, frame)), force)


def _update(value, normal):
    if not isinstance(value, TangentUpdate):
        raise TypeError("expected a solver-produced TangentUpdate")
    elastic = _vector(value.elastic_displacement_m, "elastic displacement")
    length = hypot(*elastic)
    if not np.isfinite(length):
        raise ValueError("elastic displacement magnitude is not representable")
    if abs(float(np.dot(elastic, normal))) > 1e-10 * max(length, 1e-300):
        raise ValueError("elastic update must be tangent to the material normal")
    energy = finite_real(value.stored_energy_j, "stored energy")
    plastic = finite_real(value.plastic_slip_m, "plastic slip")
    work = finite_real(value.tangent_work_j, "tangent work")
    if energy < 0 or plastic < 0 or value.regime not in {"stick", "slide", "frictionless"}:
        raise ValueError("invalid tangent update energy, plastic slip or regime")
    stiffness = value.stiffness_n_per_m
    if stiffness is not None:
        stiffness = finite_real(stiffness, "tangent stiffness")
        expected = 0.5 * stiffness * float(np.dot(elastic, elastic))
        if stiffness <= 0 or not np.isfinite(expected) or not isclose(energy, expected, rel_tol=1e-10, abs_tol=1e-300):
            raise ValueError("stored energy differs from the supplied tangent stiffness")
    if value.regime == "frictionless" and (length != 0 or energy != 0):
        raise ValueError("frictionless history must have zero elastic strain and energy")
    return TangentUpdate(elastic, energy, plastic, work, value.regime, stiffness)


def _continuity(old, new):
    sides = ((old.first, new.first), (old.second, new.second))
    if any(a.owner != b.owner for a, b in sides):
        return "owner_changed"
    if any(a.topology_revision != b.topology_revision for a, b in sides):
        return "topology_changed"
    if any(a.shape != b.shape or a.feature != b.feature for a, b in sides):
        return "feature_changed"
    if any(a.chart_id != b.chart_id for a, b in sides):
        return "chart_changed"
    moved = [(a, b) for a, b in sides if a.weights != b.weights or a.local_point_m != b.local_point_m]
    if moved:
        return "same_chart" if all(a.chart_id is not None for a, _ in moved) else "material_changed"
    return "persist"


class TangentHistory:
    """A bounded namespace producing immutable candidates, with no internal clock.

    capacity bounds both current samples and the number of distinct ever-loaded
    tokens, including inactive generation tombstones. No implicit eviction or
    resize can resurrect a stale spring. New independent experiments use a new
    owner. The caller publishes the returned candidate with its physical state.
    """

    def __init__(self, capacity, *, max_frame_turn_rad=pi / 12):
        self._capacity = _index(capacity, "station capacity")
        self._max_frame_turn_rad = finite_real(max_frame_turn_rad, "maximum frame turn")
        if not 0 <= self.max_frame_turn_rad <= pi:
            raise ValueError("maximum frame turn must lie in [0, pi]")
        self._owner = object()

    @property
    def capacity(self):
        return self._capacity

    @property
    def max_frame_turn_rad(self):
        return self._max_frame_turn_rad

    def initial_state(self):
        return TangentHistoryState(self._owner)

    def _check_state(self, state):
        if not isinstance(state, TangentHistoryState) or state._owner is not self._owner:
            raise ValueError("history base belongs to another owner")

    def prepare(self, state, samples, *, sample_time_s, dt_s):
        """Stage continuity and full material-frame transport; never consume state.

        The solver supplies real material displacement separately. Same-chart
        witness drift only permits continuity; it does not increment strain,
        slip or work. Unsupported changes explicitly release the old spring.
        """
        self._check_state(state)
        time = finite_real(sample_time_s, "sample time")
        dt = finite_real(dt_s, "step time")
        end = time + dt
        if time != state.commit_time_s or dt <= 0 or not np.isfinite(end) or end <= time:
            raise ValueError("sample time and positive step time must follow the base state")
        current = {}
        for i, value in enumerate(samples):
            if i >= self.capacity:
                raise ValueError("station sample capacity exceeded")
            owned = _sample(value)
            if owned.key in current:
                raise ValueError("duplicate explicit station key")
            current[owned.key] = owned
        generations = {g.key: g for g in state.generations}
        loaded_keys = {key for key, s in current.items() if s.normal_force_n > 0}
        if len(generations.keys() | loaded_keys) > self.capacity:
            raise ValueError("station generation capacity exceeded")
        old = {r.identity.key: r for r in state.active}
        prepared, ended = [], []
        for key in sorted(old.keys() | current.keys()):
            prior, sample = old.get(key), current.get(key)
            if sample is None or sample.normal_force_n == 0:
                if prior is not None:
                    ended.append(EndedStation(
                        prior.identity, "absent" if sample is None else "zero_load",
                        prior.update.stored_energy_j,
                    ))
                continue
            status = "begin" if prior is None else _continuity(prior.sample, sample)
            update = None
            if status in {"persist", "same_chart"}:
                rotation = np.asarray(sample.frame_world) @ np.asarray(prior.sample.frame_world).T
                sine = np.linalg.norm([rotation[2, 1] - rotation[1, 2],
                                       rotation[0, 2] - rotation[2, 0],
                                       rotation[1, 0] - rotation[0, 1]]) / 2
                angle = atan2(float(sine), float((np.trace(rotation) - 1) / 2))
                if angle > self.max_frame_turn_rad + 1e-12:
                    raise ValueError("material frame turn exceeds configured limit")
                update = _update(replace(
                    prior.update,
                    elastic_displacement_m=rotation @ prior.update.elastic_displacement_m,
                ), sample.normal)
                identity = prior.identity
            else:
                if prior is not None:
                    ended.append(EndedStation(prior.identity, status, prior.update.stored_energy_j))
                    status = "reset"
                previous = generations.get(key)
                identity = StationIdentity(key, 1 if previous is None else previous.generation + 1)
                generations[key] = identity
            prepared.append(PreparedStation(identity, sample, status, update))
        energy = finite_real(fsum(e.released_energy_j for e in ended), "released spring energy")
        count = len(prepared)
        frames = _readonly([p.sample.frame_world for p in prepared], (count, 3, 3))
        elastic = _readonly([
            (0.0, 0.0, 0.0) if p.prior is None else p.prior.elastic_displacement_m for p in prepared
        ], (count, 3))
        tangent = _readonly([F[:, :2].T @ e for F, e in zip(frames, elastic, strict=True)], (count, 2))
        return TangentTrial(
            self._owner, state, (state.step_index + 1, time, end),
            state.step_index + 1, time, end, tuple(prepared),
            tuple(generations[key] for key in sorted(generations)), tuple(ended), energy,
            tuple(p.identity.key for p in prepared), frames,
            _readonly([p.sample.normal_force_n for p in prepared], (count,)), elastic, tangent,
        )

    def commit(self, state, trial, updates):
        """Return an owned candidate bound to exactly this base/owner/trial.

        Repeating on the unchanged base is valid. Using a trial with a later
        state is stale. Nothing is published here: failed allocation or update
        validation leaves the physical caller free to retry the same proposal.
        """
        self._check_state(state)
        if not isinstance(trial, TangentTrial) or trial._owner is not self._owner or trial._base is not state:
            raise ValueError("stale trial or different history owner/base")
        if (trial._times != (trial.step_index, trial.sample_time_s, trial.commit_time_s)
            or trial.step_index != state.step_index + 1
            or trial.sample_time_s != state.commit_time_s):
            raise ValueError("trial time differs from its prepared base/time")
        if not isinstance(updates, Mapping) or set(updates) != set(trial.keys):
            raise ValueError("updates must cover exactly the prepared station keys")
        records = []
        for prepared in trial.stations:
            update = _update(updates[prepared.identity.key], prepared.sample.normal)
            if prepared.prior is not None and update.plastic_slip_m < prepared.prior.plastic_slip_m:
                raise ValueError("cumulative plastic slip cannot decrease within an episode")
            records.append(TangentRecord(prepared.identity, prepared.sample, update))
        return TangentHistoryState(
            self._owner, trial.step_index, trial.sample_time_s, trial.commit_time_s,
            tuple(records), trial.generations, trial.ended, trial.dropped_spring_energy_j,
        )
