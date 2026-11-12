"""Persistent, transactional FP32 node banks using the existing ordered TYX solver.

D1 publishes real changing vertex states; an opt-in adapter borrows them for
finite boundary queries. No self contact, contact response, cloth bending, or
general soft-body material is implied.
"""

import json
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from phymeo.constraints.distance import DistanceConstraints, distance_residual
from phymeo.core.model import ParticleModel
from phymeo.core.scene import StepSettings
from phymeo.core.state import ParticleState, StepInputs

from ._tyx_deformable_buffers import DeformableBuffers

if TYPE_CHECKING:
    from .deformable_dissipation import DistanceDissipation, DistanceDissipationStep

_DISSIPATION_ARRAYS = (
    "ordered_point_ids",
    "coefficients_ns_per_m",
    "reconstructed_velocity",
    "dissipated_velocity",
    "trace",
    "node_impulse",
    "node_velocity_impulse",
    "kinetic_change",
    "ideal_work",
    "rounding_work",
)

_BOUNDARY_QUARANTINE = {}
_BOUNDARY_QUARANTINE_LOCK = RLock()


def _owned(value, dtype=None):
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or value > np.iinfo(np.int64).max:
        raise ValueError(f"{name} is outside the stable int64 ID domain")
    return value


def _native(value, shape, name, *, exact=False):
    source = np.asarray(value)
    if source.shape != shape or source.dtype.kind != "f":
        raise ValueError(f"{name} has an invalid floating shape")
    if exact and source.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} must already be native float32")
    if not np.all(np.isfinite(source)):
        raise ValueError(f"{name} must be finite")
    with np.errstate(over="ignore", under="ignore"):
        result = source.astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} is not representable in float32")
    tiny = np.finfo(np.float32).tiny
    if np.any((result != 0) & (np.abs(result) < tiny)) or np.any((source != 0) & (result == 0)):
        raise ValueError(f"{name} contains unsupported subnormal/underflow values")
    return _owned(result)


def _ids(value, shape, name):
    array = np.asarray(value)
    if array.dtype != np.dtype(np.int64) or array.shape != shape or np.any(array < 0):
        raise ValueError(f"{name} must be a nonnegative int64 array of shape {shape}")
    return _owned(array)


@dataclass(frozen=True, eq=False)
class DeformableSurface:
    key: int
    facet_ids: np.ndarray
    oriented_point_ids: np.ndarray


@dataclass(frozen=True, eq=False)
class DeformableNodeWorld:
    key: int
    model: ParticleModel
    initial_state: ParticleState
    constraints: DistanceConstraints
    point_ids: np.ndarray
    surfaces: tuple[DeformableSurface, ...] = ()
    dissipation: "DistanceDissipation | None" = field(default=None, kw_only=True)


@dataclass(frozen=True, eq=False)
class DeformableSurfaceBinding:
    key: int
    facet_ids: np.ndarray
    oriented_point_ids: np.ndarray
    native_indices: np.ndarray


@dataclass(frozen=True)
class DeformableMemoryRequirements:
    device_bytes: int
    host_peak_bytes: int
    total_bytes: int
    # This is a budget of owned array payloads, not a Python/driver/RSS limit.
    accounting: str = "owned-array-reservation-v1"


@dataclass(frozen=True, eq=False)
class DeformableWorldStep:
    world: int
    input_position: np.ndarray
    input_velocity: np.ndarray
    input_clock: np.ndarray
    input_tick: int
    force: np.ndarray
    h: float
    predicted_position: np.ndarray
    predicted_velocity: np.ndarray
    alpha: np.ndarray
    multipliers: np.ndarray
    sweeps: int
    residual: float
    max_constraint: float
    trace: tuple[np.ndarray, ...]


@dataclass(frozen=True, eq=False)
class DeformableDissipatedWorldStep(DeformableWorldStep):
    """Explicit velocity phase after the original elastic reconstruction."""

    dissipation: "DistanceDissipationStep"


@dataclass(frozen=True, eq=False)
class DeformableAcceptedState:
    world: int
    position: np.ndarray
    velocity: np.ndarray
    clock: np.ndarray
    tick: int
    time: float
    last_step: DeformableWorldStep | None


@dataclass(frozen=True, eq=False)
class DeformableStepReceipt:
    schema: str
    binding_sha256: str
    source_revision: int
    destination_revision: int
    active_bank: int
    accepted: bool
    worlds: Mapping
    states: Mapping


@dataclass(frozen=True, eq=False)
class DeformableCheckpoint:
    schema: str
    binding_sha256: str
    owner_id: object
    active_bank: int
    worlds: Mapping
    last_step: DeformableStepReceipt | None
    content_sha256: str


@dataclass(frozen=True, eq=False)
class _DeformableReplacement:
    source_revision: int
    destination_revision: int
    active_bank: int
    states: Mapping
    last_step: object
    replacements: int


@dataclass(frozen=True, eq=False)
class _World:
    key: int
    n: int
    e: int
    p: int
    initial_position: np.ndarray
    initial_velocity: np.ndarray
    weights: np.ndarray
    gravity: np.ndarray
    edges: np.ndarray
    rest: np.ndarray
    compliance: np.ndarray
    point_ids: np.ndarray
    surfaces: tuple[DeformableSurfaceBinding, ...]
    dissipation: "DistanceDissipation | None"
    damping_coefficients: np.ndarray | None

    @property
    def dissipative(self):
        return self.damping_coefficients is not None

    @property
    def indices(self):
        return self.edges[: self.e]

    @property
    def rest_length(self):
        return self.rest[: self.e]


def _geometry(model, position):
    # Validation only; these HOST calculations never generate candidate vertices.
    for surface in model.surfaces:
        tri = position[surface.native_indices].astype(np.float64)
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if not np.all(np.isfinite(cross)) or np.any(np.max(np.abs(cross), axis=1) == 0):
            raise ValueError("surface contains a degenerate/nonfinite triangle")


def _compile(worlds, settings, trace):
    if not isinstance(settings, StepSettings) or settings.substeps != 1:
        raise ValueError("D1 requires one physics substep per explicit step()")
    if settings.position_iterations > 200:
        raise ValueError("D1 position iteration hard ceiling is 200")
    if not isinstance(trace, bool):
        raise TypeError("trace must be bool")
    bounded_worlds = []
    for world in worlds:
        bounded_worlds.append(world)
        if len(bounded_worlds) > 8:
            raise ValueError("D1 world capacity is 1..8")
    worlds = tuple(bounded_worlds)
    if not 1 <= len(worlds) <= 8:
        raise ValueError("D1 world capacity is 1..8")
    compiled = {}
    device = constant = receipt = 0
    for source in worlds:
        if not isinstance(source, DeformableNodeWorld):
            raise TypeError("expected DeformableNodeWorld")
        key = _integer(source.key, "world")
        if key in compiled:
            raise ValueError("duplicate world identity")
        if not isinstance(source.model, ParticleModel) or not isinstance(
            source.initial_state, ParticleState
        ):
            raise TypeError("expected ParticleModel and ParticleState")
        if not isinstance(source.constraints, DistanceConstraints):
            raise TypeError("D1 accepts DistanceConstraints only")
        n = source.model.particle_count
        if not 1 <= n <= 512:
            raise ValueError("D1 node capacity is 1..512 per world")
        e = len(source.constraints.indices)
        if e > 1536:
            raise ValueError("D1 distance capacity is 1536 per world")
        p = max(1, e)
        x = _native(source.initial_state.position, (n, 3), "initial position", exact=True)
        v = _native(source.initial_state.velocity, (n, 3), "initial velocity", exact=True)
        weights = _native(source.model.inverse_mass, (n,), "inverse mass")
        gravity = _native(source.model.gravity, (3,), "gravity")
        if np.any(weights < 0) or np.any(v[weights == 0] != 0):
            raise ValueError("fixed nodes require exactly zero initial velocity")
        point_ids = _ids(source.point_ids, (n,), "point IDs")
        if len(np.unique(point_ids)) != n:
            raise ValueError("point IDs must be unique")
        source_edges = np.asarray(source.constraints.indices)
        if source_edges.shape != (e, 2) or np.any(source_edges < 0) or np.any(source_edges >= n):
            raise ValueError("distance endpoint outside the registered node domain")
        edges = np.zeros((p, 2), np.int32)
        edges[:e] = source_edges
        rest = np.zeros(p, np.float32)
        rest[:e] = _native(source.constraints.rest_length, (e,), "rest lengths")
        compliance = np.zeros(p, np.float32)
        compliance[:e] = _native(source.constraints.compliance, (e,), "compliance")
        if np.any(rest < 0) or np.any(compliance < 0):
            raise ValueError("rest/compliance must be nonnegative")
        damping = source.dissipation
        coefficients = None
        if damping is not None:
            from .deformable_dissipation import DistanceDissipation

            if type(damping) is not DistanceDissipation:
                raise TypeError("expected DistanceDissipation")
            # Bind all original ordered edges, including zero coefficients.
            if not np.array_equal(damping.ordered_point_ids, point_ids[source_edges]):
                raise ValueError("dissipation ordered point IDs differ from material edges")
            native = _native(damping.coefficients_ns_per_m, (e,), "dissipation coefficient")
            if np.any(native < 0):
                raise ValueError("dissipation coefficients must be nonnegative")
            if np.any(native):
                padded = np.zeros(p, np.float32)
                padded[:e] = native
                coefficients = _owned(padded)
                active_edges = source_edges[native > 0]
                if np.any(np.all(x[active_edges[:, 0]] == x[active_edges[:, 1]], axis=1)):
                    raise ValueError("nonzero dissipation requires nonzero edge length")
        lookup = {int(stable): row for row, stable in enumerate(point_ids)}
        surfaces = []
        surface_keys = set()
        total_facets = 0
        for surface in source.surfaces:
            if not isinstance(surface, DeformableSurface):
                raise TypeError("expected DeformableSurface")
            surface_key = _integer(surface.key, "surface")
            if surface_key in surface_keys:
                raise ValueError("duplicate surface key")
            surface_keys.add(surface_key)
            count = len(surface.facet_ids)
            if count < 1:
                raise ValueError("registered surfaces must contain at least one facet")
            total_facets += count
            if total_facets > 128:
                raise ValueError("D1 facet capacity is 128 per world")
            ids = _ids(surface.facet_ids, (count,), "facet IDs")
            points = _ids(surface.oriented_point_ids, (count, 3), "oriented point IDs")
            if len(np.unique(ids)) != count:
                raise ValueError("duplicate facet ID")
            if any(len(set(map(int, row))) != 3 for row in points):
                raise ValueError("a facet must reference three distinct points")
            try:
                native = np.asarray([[lookup[int(i)] for i in row] for row in points], np.int32)
            except KeyError as error:
                raise ValueError("unknown stable point in facet") from error
            canonical = [tuple(sorted(map(int, row))) for row in points]
            if len(set(canonical)) != len(canonical):
                raise ValueError("duplicate facet topology")
            surfaces.append(DeformableSurfaceBinding(surface_key, ids, points, _owned(native)))
        model = _World(
            key,
            n,
            e,
            p,
            x,
            v,
            weights,
            gravity,
            _owned(edges),
            _owned(rest),
            _owned(compliance),
            point_ids,
            tuple(surfaces),
            damping,
            coefficients,
        )
        _geometry(model, x)
        compiled[key] = model
        device += 88 * n + 32 * p + 40 + (72 * p if trace else 0)
        constant += 36 * n + 16 * p + 44 * total_facets + 12
        receipt += 84 * n + 8 * e + 16 + (72 * e * settings.position_iterations if trace else 0)
        if damping is not None:
            constant += 24 * e
        if coefficients is not None:
            device += 12 * n + 132 * p
            constant += 4 * p
            # Two velocities, requested/actual impulses, full trace, energies,
            # original parameters; reserve observer scratch without using RSS.
            receipt += 72 * n + 176 * e
            constant += 128 * n + 1024 * e + 1024
    host = constant + 3 * receipt + 2 * device
    return compiled, DeformableMemoryRequirements(device, host, device + host)


def _binding(worlds, settings, trace, mode):
    digest = sha256()
    digest.update(
        json.dumps(
            {
                "schema": "tyx-deformable-bank-v1",
                "mode": mode,
                "settings": vars(settings),
                "trace": trace,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for key, world in worlds.items():
        digest.update(str(key).encode() + b":")
        for array in (
            world.initial_position,
            world.initial_velocity,
            world.weights,
            world.gravity,
            world.edges,
            world.rest,
            world.compliance,
            world.point_ids,
        ):
            digest.update(str((array.dtype.str, array.shape)).encode())
            digest.update(array.tobytes())
        for surface in world.surfaces:
            digest.update(str(surface.key).encode() + b":")
            for array in (surface.facet_ids, surface.oriented_point_ids, surface.native_indices):
                digest.update(str((array.dtype.str, array.shape)).encode())
                digest.update(array.tobytes())
        if world.dissipation is not None:
            digest.update(b"distance-dissipation-v1")
            for array in (
                world.dissipation.ordered_point_ids,
                world.dissipation.coefficients_ns_per_m,
            ):
                digest.update(str((array.dtype.str, array.shape)).encode())
                digest.update(array.tobytes())
    return digest.hexdigest()


def _checkpoint_hash(binding, states, last_step):
    digest = sha256(binding.encode())

    def array_bytes(array):
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())

    if last_step is not None:
        digest.update(
            str(
                (
                    last_step.schema,
                    last_step.binding_sha256,
                    last_step.source_revision,
                    last_step.destination_revision,
                    last_step.active_bank,
                    last_step.accepted,
                    tuple(last_step.worlds),
                    tuple(last_step.states),
                )
            ).encode()
        )
    else:
        digest.update(b"no-batch-receipt")
    for key, state in states.items():
        digest.update(str((key, state.world, state.tick, state.time)).encode())
        for array in (state.position, state.velocity, state.clock):
            array_bytes(array)
        row = state.last_step
        digest.update(b"none" if row is None else b"step")
        if row is not None:
            digest.update(
                str(
                    (row.world, row.input_tick, row.h, row.sweeps, row.residual, row.max_constraint)
                ).encode()
            )
            for array in (
                row.input_position,
                row.input_velocity,
                row.input_clock,
                row.force,
                row.predicted_position,
                row.predicted_velocity,
                row.alpha,
                row.multipliers,
            ):
                array_bytes(array)
            for frame in row.trace:
                array_bytes(frame)
            if type(row) is DeformableDissipatedWorldStep:
                digest.update(row.dissipation.schema.encode())
                for name in _DISSIPATION_ARRAYS:
                    array_bytes(getattr(row.dissipation, name))
    return digest.hexdigest()


class _VertexLoan:
    """Scoped source descriptor, protocol-read-only (not hardware write protection)."""

    def __init__(self, owner):
        self._owner = owner
        self._live = True
        self._revision = owner.revision
        self._bank = owner._active

    def _check(self):
        if not self._live or self._owner._loan is not self:
            raise RuntimeError("vertex loan is expired")
        self._owner._require_open()
        if self._revision != self._owner.revision or self._bank != self._owner._active:
            raise RuntimeError("vertex loan no longer names the accepted bank")

    @property
    def revision(self):
        self._check()
        return self._revision

    @property
    def active_bank(self):
        self._check()
        return self._bank

    @property
    def world_keys(self):
        self._check()
        return self._owner.world_keys

    @property
    def binding_sha256(self):
        self._check()
        return self._owner.binding_sha256

    @property
    def mode(self):
        self._check()
        return self._owner.mode

    @property
    def runtime(self):
        self._check()
        return self._owner._buffers.runtime

    def position(self, world):
        self._check()
        return self._owner._buffers.worlds[world][f"x{self._bank}"]

    def velocity(self, world):
        self._check()
        return self._owner._buffers.worlds[world][f"v{self._bank}"]

    def tick(self, world):
        self._check()
        return self._owner._buffers.worlds[world][f"tick{self._bank}"]

    def point_ids(self, world):
        self._check()
        return self._owner._models[world].point_ids

    def surfaces(self, world):
        self._check()
        return self._owner._models[world].surfaces


class TyxDeformableBatch:
    """A bounded, all-world atomic device producer with explicit CAS operations."""

    def __init__(
        self,
        worlds,
        *,
        mode="interpret",
        settings=None,
        trace=False,
        memory_limit_bytes=256 * 1024 * 1024,
        cache_root=None,
        _runtime_lease=None,
    ):
        self._lock = RLock()
        self._busy = False
        self._pending_stage = None
        self._loan = None
        self._close_pending = False
        self._poisoned = False
        self._revision = 0
        self._owner_id = object()
        self._collision_owner = self._owner_id
        self._active = 0
        self._topology_generation = 0
        self._boundary_plans = set()
        self._boundary_runtime_leases = {}
        self._boundary_unknown = {}
        self._boundary_closing = False
        self._boundary_close_pending = False
        if mode not in ("interpret", "cuda"):
            raise ValueError("D1 mode must be interpret or cuda")
        settings = StepSettings(substeps=1) if settings is None else settings
        self._models, self.memory_requirements = _compile(worlds, settings, trace)
        limit = _integer(memory_limit_bytes, "memory_limit_bytes")
        if self.memory_requirements.total_bytes > limit:
            raise MemoryError(
                "D1 owned device plus HOST array reservation exceeds memory_limit_bytes"
            )
        self.mode, self.settings, self.trace = mode, settings, trace
        self.world_keys = tuple(self._models)
        self.binding_sha256 = _binding(self._models, settings, trace, mode)
        self._states = MappingProxyType(
            {key: self._initial_state(model) for key, model in self._models.items()}
        )
        self.last_step = None
        self._buffers = DeformableBuffers(
            self._models, mode, trace, cache_root, runtime_lease=_runtime_lease
        )
        self._mode = mode
        self._tx = self._buffers.tx
        self._runtime = self._buffers.runtime
        self._state_uploaded_bytes = sum(2 * (24 * model.n + 12) for model in self._models.values())
        self._force_uploaded_bytes = 0
        self._successful_steps = 0
        self._replacements = 0
        if self._buffers.native_bytes != self.memory_requirements.device_bytes:
            primary = RuntimeError("D1 buffer allocation differs from its declared resource table")
            try:
                self._buffers.close()
            except BaseException as cleanup:
                error = BaseExceptionGroup(
                    "D1 allocation table and cleanup failed", [primary, cleanup]
                )
                error.resource_owner = self
                raise error
            raise primary

    @staticmethod
    def requirements(worlds, *, settings=None, trace=False):
        return _compile(worlds, StepSettings(substeps=1) if settings is None else settings, trace)[
            1
        ]

    @property
    def revision(self):
        return self._revision

    @property
    def _state_epoch(self):
        return self._revision

    @property
    def resource_state(self):
        state = self._buffers.resources.state
        if state == "OPEN" and (self._close_pending or self._boundary_close_pending):
            return "PENDING_CLOSE"
        return state

    @staticmethod
    def _initial_state(model):
        return DeformableAcceptedState(
            model.key,
            model.initial_position,
            model.initial_velocity,
            _owned(np.zeros(2, np.float32)),
            0,
            0.0,
            None,
        )

    def _require_open(self):
        if self.resource_state != "OPEN" or self._poisoned:
            raise RuntimeError(
                "deformable owner is closed, close-only, or synchronization-poisoned"
            )

    def _ready(self, expected_revision):
        self._require_open()
        if self._busy or self._loan is not None:
            raise RuntimeError("deformable mutation cannot reenter a transaction or vertex loan")
        if _integer(expected_revision, "expected_revision") != self._revision:
            raise ValueError("stale deformable revision")

    def _check_open(self):
        self._require_open()
        if self._busy or self._loan is not None:
            raise RuntimeError("boundary query cannot reenter an operation or vertex loan")

    def _device(self):
        return self._buffers.context()

    def _register_boundary_runtime_lease(self, lease):
        with self._lock:
            self._require_open()
            if self._busy or getattr(lease, "runtime", None) is not self._runtime:
                raise ValueError("boundary lease must borrow this idle node runtime")
            if any(value is lease for value in self._boundary_runtime_leases.values()):
                raise ValueError("boundary runtime lease is already registered")
            token = object()
            self._boundary_runtime_leases[token] = lease
            return token

    def _release_boundary_runtime_lease(self, token):
        with self._lock:
            if token in self._boundary_unknown:
                raise RuntimeError("boundary UNKNOWN_RELEASE token cannot be unregistered")
            if token not in self._boundary_runtime_leases:
                raise ValueError("unknown boundary runtime lease token")
            del self._boundary_runtime_leases[token]

    def _quarantine_boundary_runtime_lease(self, token, ledger):
        with self._lock:
            if token not in self._boundary_runtime_leases:
                raise ValueError("unknown boundary runtime lease token")
            if ledger.state != "UNKNOWN_RELEASE":
                raise ValueError("only UNKNOWN_RELEASE may quarantine the node runtime")
            previous = self._boundary_unknown.get(token)
            if previous is not None and previous is not ledger:
                raise ValueError("boundary token already belongs to another unknown ledger")
            self._boundary_unknown[token] = ledger
            self._poisoned = True
            with _BOUNDARY_QUARANTINE_LOCK:
                _BOUNDARY_QUARANTINE[id(self)] = self

    def _close_boundary_resources(self):
        if self._boundary_closing:
            raise RuntimeError("recursive boundary cleanup")
        self._boundary_closing = True
        self._boundary_close_pending = True
        try:
            errors = []
            for plan in tuple(self._boundary_plans):
                try:
                    plan.close()
                except BaseException as error:
                    errors.append(error)
            if errors:
                raise BaseExceptionGroup(
                    "boundary cleanup incomplete; node runtime retained", errors
                )
            if self._boundary_unknown or self._boundary_runtime_leases:
                raise RuntimeError("boundary runtime dependencies retained; owner cannot close")
            self._boundary_plans.clear()
        finally:
            self._boundary_closing = False

    def prepare_boundary_query(
        self, model, bindings=(), *, limits=None, memory_limit_bytes=64 * 1024 * 1024
    ):
        """Prepare finite static round-shape queries over real accepted node surfaces."""
        from ..geometry._triangle_distance_dispatch import BoundaryDistanceLimits
        from ._deformable_boundary_source import DeformableBoundaryAdapter
        from ._resident_boundary_query import ResidentBoundaryQueryPlan

        with self._lock:
            self._check_open()
            return ResidentBoundaryQueryPlan(
                self,
                model,
                bindings,
                limits=BoundaryDistanceLimits() if limits is None else limits,
                memory_limit_bytes=memory_limit_bytes,
                source_adapter=DeformableBoundaryAdapter(self),
            )

    @contextmanager
    def _transaction(self, expected_revision):
        with self._lock:
            self._ready(expected_revision)
            self._busy = True
            try:
                with self._buffers.context():
                    yield
            except BaseException as primary:
                try:
                    self._buffers.sync()
                except BaseException as synchronization:
                    self._poisoned = True
                    raise BaseExceptionGroup(
                        "deformable operation and recovery synchronization failed",
                        [primary, synchronization],
                    )
                raise
            finally:
                self._pending_stage = None
                self._busy = False

    def snapshot(self):
        with self._lock:
            self._require_open()
            if self._busy:
                raise RuntimeError("snapshot cannot reenter a pending transaction")
            return self._states

    def stats(self):
        with self._lock:
            return MappingProxyType(
                {
                    "allocations": self._buffers.allocations,
                    "native_buffer_bytes": self._buffers.native_bytes,
                    "host_reserved_bytes": self.memory_requirements.host_peak_bytes,
                    "state_uploaded_bytes": self._state_uploaded_bytes,
                    "force_uploaded_bytes": self._force_uploaded_bytes,
                    "uploaded_bytes": self._buffers.uploaded_bytes,
                    "downloaded_bytes": self._buffers.downloaded_bytes,
                    "kernels": self._buffers.kernels,
                    "synchronizations": self._buffers.syncs,
                    "successful_steps": self._successful_steps,
                    "replacements": self._replacements,
                    "boundary_plans": len(self._boundary_plans),
                    "boundary_runtime_leases": len(self._boundary_runtime_leases),
                    "boundary_unknown_releases": len(self._boundary_unknown),
                }
            )

    def _read_candidate(self, key, bank):
        model, arrays = self._models[key], self._buffers.worlds[key]
        x = _native(
            self._buffers.read(arrays[f"x{bank}"]), (model.n, 3), "candidate position", exact=True
        )
        v = _native(
            self._buffers.read(arrays[f"v{bank}"]), (model.n, 3), "candidate velocity", exact=True
        )
        clock = _native(
            self._buffers.read(arrays[f"clock{bank}"]), (2,), "candidate clock", exact=True
        )
        tick_array = self._buffers.read(arrays[f"tick{bank}"])
        if tick_array.dtype != np.dtype(np.int32) or tick_array.shape != (1,) or tick_array[0] < 0:
            raise ValueError("candidate tick is invalid")
        _geometry(model, x)
        if np.any(v[model.weights == 0] != 0):
            raise ValueError("candidate moved a fixed velocity")
        if not np.array_equal(x[model.weights == 0], model.initial_position[model.weights == 0]):
            raise ValueError("candidate moved a fixed point")
        return x, v, clock, int(tick_array[0])

    def _make_receipt(self, rows, states, bank):
        return DeformableStepReceipt(
            "tyx-deformable-step-v1",
            self.binding_sha256,
            self._revision,
            self._revision + 1,
            bank,
            True,
            MappingProxyType(dict(rows)),
            MappingProxyType(dict(states)),
        )

    def step(self, inputs, h, *, expected_revision):
        with self._transaction(expected_revision):
            receipt = self._stage_step(inputs, h)
            self._validate_staged_step(receipt)
            self._publish_staged_step(receipt)
            return receipt

    def _validate_staged_step(self, receipt):
        if (
            not self._busy
            or receipt is not self._pending_stage
            or type(receipt) is not DeformableStepReceipt
            or receipt.source_revision != self._revision
            or receipt.active_bank != 1 - self._active
        ):
            raise ValueError("node step draft is stale or outside its owner transaction")

    def _publish_staged_step(self, receipt):
        self._active, self._states, self.last_step = (receipt.active_bank, receipt.states, receipt)
        self._revision = receipt.destination_revision
        self._successful_steps += 1
        self._pending_stage = None

    def _stage_step(self, inputs, h):
        """Run the original integrator into standby inside an owned transaction."""
        from ._tyx_deformable_state import advance_clock, begin_candidate
        from ._tyx_kernels import integrate_particles
        from ._tyx_math_kernels import axpy_kernel
        from ._tyx_node_kernels import project_node_distances, reconstruct_nodes

        if not self._busy or self._pending_stage is not None:
            raise RuntimeError("node staging requires one active unpublished transaction")
        with self._buffers.context():
            if not isinstance(inputs, Mapping) or set(inputs) != set(self.world_keys):
                raise ValueError("one explicit StepInputs force packet is required for every world")
            if isinstance(h, (bool, np.bool_)) or not np.isscalar(h):
                raise TypeError("h must be a positive scalar")
            native_h = _native(np.asarray([h], dtype=np.float64), (1,), "h")[0]
            if native_h <= 0:
                raise ValueError("h must be positive")
            factor = _native(np.asarray([1.0 / float(native_h)]), (1,), "inverse h")[0]
            packets = {}
            for key, model in self._models.items():
                if not isinstance(inputs[key], StepInputs):
                    raise TypeError("expected StepInputs")
                packets[key] = _native(inputs[key].force, (model.n, 3), "force", exact=True)
                if self._states[key].tick >= np.iinfo(np.int32).max:
                    raise ValueError("D1 device tick capacity exceeded")
            bank = 1 - self._active
            rows, states = {}, {}
            for key, model in self._models.items():
                a = self._buffers.worlds[key]
                entry = self._states[key]
                self._buffers.upload(a["force"], packets[key])
                self._force_uploaded_bytes += packets[key].nbytes
                self._buffers.call(
                    integrate_particles,
                    model.n,
                    native_h,
                    a[f"x{self._active}"],
                    a[f"v{self._active}"],
                    a["force"],
                    a["weights"],
                    a["gravity"],
                    a["predicted_x"],
                    a["predicted_v"],
                )
                self._buffers.call(
                    begin_candidate,
                    max(model.n, model.p),
                    model.n,
                    model.p,
                    a["predicted_x"],
                    a[f"x{bank}"],
                    a["multipliers"],
                    a["status"],
                )
                # Preserve the original two native axpy evaluations of compliance / h / h.
                self._buffers.call(
                    axpy_kernel, model.p, factor, a["zero"], a["compliance"], a["alpha_tmp"]
                )
                self._buffers.call(
                    axpy_kernel, model.p, factor, a["zero"], a["alpha_tmp"], a["alpha"]
                )
                self._buffers.sync()
                px = _native(
                    self._buffers.read(a["predicted_x"]), (model.n, 3), "predictor", exact=True
                )
                pv = _native(
                    self._buffers.read(a["predicted_v"]),
                    (model.n, 3),
                    "predictor velocity",
                    exact=True,
                )
                alpha = _native(
                    self._buffers.read(a["alpha"])[: model.e], (model.e,), "alpha", exact=True
                )
                frames = []
                residual = maximum = 0.0
                sweeps = 0
                multipliers = _owned(np.zeros(model.e, np.float32))
                if model.e:
                    for sweep in range(self.settings.position_iterations):
                        if self.trace:
                            from ._tyx_distance_trace import project_node_distances_traced

                            self._buffers.call(
                                project_node_distances_traced,
                                model.e,
                                a["edges"],
                                a["rest"],
                                a["alpha"],
                                a["weights"],
                                a[f"x{bank}"],
                                a["multipliers"],
                                a["trace"],
                            )
                        else:
                            self._buffers.call(
                                project_node_distances,
                                model.e,
                                a["edges"],
                                a["rest"],
                                a["alpha"],
                                a["weights"],
                                a[f"x{bank}"],
                                a["multipliers"],
                            )
                        self._buffers.sync()
                        candidate_x = _native(
                            self._buffers.read(a[f"x{bank}"]),
                            (model.n, 3),
                            "projected position",
                            exact=True,
                        )
                        multipliers = _native(
                            self._buffers.read(a["multipliers"])[: model.e],
                            (model.e,),
                            "multipliers",
                            exact=True,
                        )
                        if self.trace:
                            frames.append(
                                _native(
                                    self._buffers.read(a["trace"])[: model.e],
                                    (model.e, 18),
                                    "ordered trace",
                                    exact=True,
                                )
                            )
                        residual, maximum = distance_residual(
                            candidate_x, model, alpha, multipliers, self.settings.reference_length
                        )
                        sweeps = sweep + 1
                        if not np.isfinite(residual) or not np.isfinite(maximum):
                            raise ValueError("nonfinite distance audit")
                        if residual <= self.settings.tolerance:
                            break
                    if residual > self.settings.tolerance:
                        raise RuntimeError(
                            "D1 ordered projection exhausted the original iteration budget"
                        )
                self._buffers.call(
                    reconstruct_nodes,
                    model.n,
                    native_h,
                    a["predicted_x"],
                    a["predicted_v"],
                    a[f"x{bank}"],
                    a["weights"],
                    a[f"v{bank}"],
                )
                if model.dissipative:
                    from ._tyx_distance_dissipation import (
                        copy_dissipation_velocity,
                        dissipate_distances,
                    )

                    self._buffers.call(
                        copy_dissipation_velocity,
                        model.n,
                        a[f"v{bank}"],
                        a["damping_reconstructed_v"],
                    )
                    self._buffers.upload(a["damping_trace"], np.zeros((model.p, 32), np.float32))
                    self._buffers.call(
                        dissipate_distances,
                        model.e,
                        native_h,
                        a["edges"],
                        a["damping_coefficients"],
                        a["weights"],
                        a[f"x{bank}"],
                        a[f"v{bank}"],
                        a["damping_trace"],
                        a["status"],
                    )
                    self._buffers.sync()
                    if np.any(self._buffers.read(a["status"])):
                        raise ValueError(
                            "D6 axial dissipation rejected candidate geometry/arithmetic"
                        )
                self._buffers.call(
                    advance_clock,
                    native_h,
                    a[f"clock{self._active}"],
                    a[f"tick{self._active}"],
                    a[f"clock{bank}"],
                    a[f"tick{bank}"],
                    a["status"],
                )
                self._buffers.sync()
                if np.any(self._buffers.read(a["status"])):
                    raise ValueError("D1 clock/tick publication failed")
                x, v, clock, tick = self._read_candidate(key, bank)
                if tick != entry.tick + 1:
                    raise ValueError("candidate tick is not the immediate successor")
                row_values = (
                    key,
                    entry.position,
                    entry.velocity,
                    entry.clock,
                    entry.tick,
                    packets[key],
                    float(native_h),
                    px,
                    pv,
                    alpha,
                    multipliers,
                    sweeps,
                    float(residual),
                    float(maximum),
                    tuple(frames),
                )
                if model.dissipative:
                    from .deformable_dissipation import make_dissipation_step

                    phase = make_dissipation_step(
                        model.dissipation,
                        model.edges[: model.e],
                        model.weights,
                        self._buffers.read(a["damping_reconstructed_v"]),
                        v,
                        self._buffers.read(a["damping_trace"])[: model.e],
                    )
                    row = DeformableDissipatedWorldStep(*row_values, phase)
                else:
                    row = DeformableWorldStep(*row_values)
                rows[key] = row
                states[key] = DeformableAcceptedState(key, x, v, clock, tick, float(clock[0]), row)
            receipt = self._make_receipt(rows, states, bank)
            # The ordinary step or joint coordinator publishes only after all
            # of its participants and aggregate records have been validated.
            self._pending_stage = receipt
            return receipt

    def checkpoint(self):
        with self._lock:
            self._require_open()
            if self._busy or self._loan is not None:
                raise RuntimeError("checkpoint cannot reenter a transaction or vertex loan")
            return DeformableCheckpoint(
                "tyx-deformable-checkpoint-v1",
                self.binding_sha256,
                self._owner_id,
                self._active,
                self._states,
                self.last_step,
                _checkpoint_hash(self.binding_sha256, self._states, self.last_step),
            )

    @staticmethod
    def _is_owned(array):
        base = array
        while isinstance(base, np.ndarray):
            if base.flags.writeable:
                return False
            base = base.base
        return isinstance(base, bytes)

    def _checkpoint_state(self, key, state):
        model = self._models[key]
        if not isinstance(state, DeformableAcceptedState) or state.world != key:
            raise ValueError("checkpoint state identity mismatch")
        frozen = []
        unchanged = True
        for value, shape, name in (
            (state.position, (model.n, 3), "position"),
            (state.velocity, (model.n, 3), "velocity"),
            (state.clock, (2,), "clock"),
        ):
            safe = _native(value, shape, "checkpoint " + name, exact=True)
            trusted = self._is_owned(value)
            frozen.append(value if trusted else safe)
            unchanged = unchanged and trusted
        if _integer(state.tick, "checkpoint tick") > np.iinfo(np.int32).max or state.time != float(
            frozen[2][0]
        ):
            raise ValueError("checkpoint clock/tick mismatch")
        row = state.last_step
        if row is not None:
            expected_type = (
                DeformableDissipatedWorldStep if model.dissipative else DeformableWorldStep
            )
            if type(row) is not expected_type or row.world != key:
                raise ValueError("checkpoint step identity mismatch")
            if (
                row.input_tick + 1 != state.tick
                or not 0 <= row.sweeps <= self.settings.position_iterations
            ):
                raise ValueError("checkpoint step/tick/count mismatch")
            if len(row.trace) != (row.sweeps if self.trace else 0):
                raise ValueError("checkpoint trace count mismatch")
            if not np.isfinite([row.h, row.residual, row.max_constraint]).all() or row.h <= 0:
                raise ValueError("checkpoint step scalars invalid")
            values = []
            row_owned = True
            for name, shape in (
                ("input_position", (model.n, 3)),
                ("input_velocity", (model.n, 3)),
                ("input_clock", (2,)),
                ("force", (model.n, 3)),
                ("predicted_position", (model.n, 3)),
                ("predicted_velocity", (model.n, 3)),
                ("alpha", (model.e,)),
                ("multipliers", (model.e,)),
            ):
                value = getattr(row, name)
                safe = _native(value, shape, "checkpoint " + name, exact=True)
                trusted = self._is_owned(value)
                values.append(value if trusted else safe)
                row_owned = row_owned and trusted
            frames = []
            for frame in row.trace:
                safe = _native(frame, (model.e, 18), "checkpoint trace", exact=True)
                trusted = self._is_owned(frame)
                frames.append(frame if trusted else safe)
                row_owned = row_owned and trusted
            if model.dissipative:
                from .deformable_dissipation import DistanceDissipationStep

                phase = row.dissipation
                if (
                    type(phase) is not DistanceDissipationStep
                    or phase.schema != "distance-dissipation-step-v1"
                ):
                    raise ValueError("checkpoint dissipation phase identity mismatch")
                layouts = (
                    ("ordered_point_ids", (model.e, 2), np.int64),
                    ("coefficients_ns_per_m", (model.e,), np.float64),
                    ("reconstructed_velocity", (model.n, 3), np.float32),
                    ("dissipated_velocity", (model.n, 3), np.float32),
                    ("trace", (model.e, 32), np.float32),
                    ("node_impulse", (model.n, 3), np.float64),
                    ("node_velocity_impulse", (model.n, 3), np.float64),
                    ("kinetic_change", (model.e,), np.float64),
                    ("ideal_work", (model.e,), np.float64),
                    ("rounding_work", (model.e,), np.float64),
                )
                for name, shape, dtype in layouts:
                    value = getattr(phase, name)
                    if (
                        not isinstance(value, np.ndarray)
                        or value.shape != shape
                        or value.dtype != np.dtype(dtype)
                        or not np.all(np.isfinite(value))
                        or not self._is_owned(value)
                    ):
                        raise ValueError("checkpoint dissipation array invalid: " + name)
                for name in ("ordered_point_ids", "coefficients_ns_per_m"):
                    if getattr(phase, name).tobytes() != getattr(model.dissipation, name).tobytes():
                        raise ValueError("checkpoint dissipation parameter binding mismatch")
                if phase.dissipated_velocity.tobytes() != state.velocity.tobytes():
                    raise ValueError("checkpoint dissipation does not bind accepted velocity")
            if not row_owned:
                row = DeformableWorldStep(
                    key,
                    values[0],
                    values[1],
                    values[2],
                    row.input_tick,
                    values[3],
                    row.h,
                    values[4],
                    values[5],
                    values[6],
                    values[7],
                    row.sweeps,
                    row.residual,
                    row.max_constraint,
                    tuple(frames),
                )
            unchanged = unchanged and row_owned
        if not unchanged:
            raise ValueError("D1 checkpoints require their original immutable array payloads")
        return state

    def _validate_checkpoint(self, checkpoint):
        if (
            not isinstance(checkpoint, DeformableCheckpoint)
            or checkpoint.schema != "tyx-deformable-checkpoint-v1"
        ):
            raise TypeError("expected a D1 in-memory checkpoint")
        if checkpoint.owner_id is not self._owner_id:
            raise ValueError("checkpoint belongs to another owner")
        if (
            checkpoint.binding_sha256 != self.binding_sha256
            or tuple(checkpoint.worlds) != self.world_keys
        ):
            raise ValueError("checkpoint model/layout/backend binding mismatch")
        if checkpoint.active_bank not in (0, 1):
            raise ValueError("invalid checkpoint bank")
        states = {
            key: self._checkpoint_state(key, state) for key, state in checkpoint.worlds.items()
        }
        receipt = checkpoint.last_step
        if receipt is not None:
            if (
                not isinstance(receipt, DeformableStepReceipt)
                or receipt.schema != "tyx-deformable-step-v1"
                or receipt.binding_sha256 != self.binding_sha256
                or receipt.accepted is not True
                or tuple(receipt.worlds) != self.world_keys
                or tuple(receipt.states) != self.world_keys
                or receipt.destination_revision != receipt.source_revision + 1
                or receipt.active_bank not in (0, 1)
            ):
                raise ValueError("checkpoint batch receipt binding mismatch")
            for key, state in checkpoint.worlds.items():
                if receipt.states[key] is not state or receipt.worlds[key] is not state.last_step:
                    raise ValueError("checkpoint receipt does not bind the complete saved state")
        if _checkpoint_hash(self.binding_sha256, states, receipt) != checkpoint.content_sha256:
            raise ValueError("checkpoint content hash mismatch")
        return MappingProxyType(states)

    def _verify_replacement(self, expected, bank):
        for key, state in expected.items():
            x, v, clock, tick = self._read_candidate(key, bank)
            if tick != state.tick:
                raise ValueError("replacement device tick differs")
            for actual, wanted in ((x, state.position), (v, state.velocity), (clock, state.clock)):
                if actual.tobytes() != wanted.tobytes():
                    raise ValueError(
                        "replacement device state differs from its exact requested bytes"
                    )

    def _replace(self, selected, expected, last_step=None):
        draft = self._stage_replace(selected, expected, last_step)
        self._validate_staged_replace(draft)
        self._publish_staged_replace(draft)

    def _validate_staged_replace(self, draft):
        if (
            not self._busy
            or type(draft) is not _DeformableReplacement
            or draft is not self._pending_stage
            or draft.source_revision != self._revision
            or draft.active_bank != 1 - self._active
        ):
            raise ValueError("node replacement is stale or outside its transaction")

    def _publish_staged_replace(self, draft):
        self._active, self._states, self.last_step = (
            draft.active_bank,
            draft.states,
            draft.last_step,
        )
        self._revision, self._replacements = draft.destination_revision, draft.replacements
        self._pending_stage = None

    def _stage_replace(self, selected, expected, last_step=None):
        from ._tyx_deformable_state import copy_state

        if not self._busy or self._pending_stage is not None:
            raise RuntimeError("replacement requires one unpublished node transaction")
        bank = 1 - self._active
        for key, model in self._models.items():
            a = self._buffers.worlds[key]
            if key in selected:
                state = expected[key]
                for name, value in (
                    (f"x{bank}", state.position),
                    (f"v{bank}", state.velocity),
                    (f"clock{bank}", state.clock),
                    (f"tick{bank}", np.asarray([state.tick], np.int32)),
                ):
                    self._buffers.upload(a[name], value)
                    self._state_uploaded_bytes += value.nbytes
            else:
                self._buffers.call(
                    copy_state,
                    model.n,
                    a[f"x{self._active}"],
                    a[f"v{self._active}"],
                    a[f"clock{self._active}"],
                    a[f"tick{self._active}"],
                    a[f"x{bank}"],
                    a[f"v{bank}"],
                    a[f"clock{bank}"],
                    a[f"tick{bank}"],
                )
        self._buffers.sync()
        self._verify_replacement(expected, bank)
        publication = MappingProxyType(dict(expected))
        draft = _DeformableReplacement(
            self._revision,
            self._revision + 1,
            bank,
            publication,
            last_step,
            self._replacements + 1,
        )
        self._pending_stage = draft
        return draft

    def reset(self, worlds, *, expected_revision):
        with self._transaction(expected_revision):
            keys = tuple(_integer(key, "reset world") for key in worlds)
            if len(set(keys)) != len(keys) or any(key not in self._models for key in keys):
                raise ValueError("reset world identities must be unique and registered")
            if not keys:
                return
            expected = dict(self._states)
            for key in keys:
                expected[key] = self._initial_state(self._models[key])
            self._replace(set(keys), expected)

    def restore(self, checkpoint, *, expected_revision):
        with self._transaction(expected_revision):
            states = self._validate_checkpoint(checkpoint)
            self._replace(set(self.world_keys), states, checkpoint.last_step)

    @contextmanager
    def borrow_vertices(self, *, expected_revision):
        with self._lock:
            self._ready(expected_revision)
            loan = _VertexLoan(self)
            self._loan = loan
            try:
                try:
                    self._buffers.sync()
                except BaseException:
                    self._poisoned = True
                    raise
                primary = None
                try:
                    yield loan
                except BaseException as error:
                    primary = error
                    raise
                finally:
                    try:
                        self._buffers.sync()
                    except BaseException as synchronization:
                        self._poisoned = True
                        if primary is not None:
                            raise BaseExceptionGroup(
                                "vertex consumer and completion synchronization failed",
                                [primary, synchronization],
                            )
                        raise
            finally:
                loan._live = False
                self._loan = None

    def close(self):
        with self._lock:
            if self._busy or self._loan is not None:
                raise RuntimeError("close cannot reenter an operation or active vertex loan")
            if self.resource_state == "CLOSED":
                return
            self._close_pending = True
            self._close_boundary_resources()
            self._buffers.close()

    def __enter__(self):
        with self._lock:
            self._require_open()
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if value is not None:
                raise BaseExceptionGroup(
                    "deformable operation and cleanup both failed", [value, cleanup]
                )
            raise
        return False
