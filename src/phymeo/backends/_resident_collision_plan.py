"""Versioned broad phase from a resident batch's accepted FreeSphere poses.

The batch owns the runtime, lock and accepted/candidate transaction. This module
owns only immutable registrations, collision scratch and complete query results.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral

import numpy as np

from ..constraints.collision_shapes import (
    MAX_SHAPES,
    CollisionKey,
    CollisionModel,
    CollisionMotion,
)
from ..geometry._collision_bvh import BoundsTree, CollisionTrees
from ..geometry._collision_spatial import SpatialPairs
from ..geometry.collision_pipeline import _envelopes, _host_bounds
from ._world_step_layout import world_key
from .tyx_collision import TyxCollisionBatch


@dataclass(frozen=True)
class AcceptedCollisionState:
    """Private scoped loan constructed only while the batch lock/runtime is held.

    states/ticks refer to one current accepted buffer, never the standby buffer.
    Consumers must not retain this view. Empty batches have no device objects.
    """

    owner: object
    topology_generation: int
    state_epoch: int
    active: int
    world_keys: tuple
    bindings: tuple
    slots: Mapping
    states: object
    ticks: object
    props: object
    runtime: object
    mode: str
    tx: object


@dataclass(frozen=True)
class ResidentSphereBinding:
    key: CollisionKey
    world: int

    def __post_init__(self):
        if not isinstance(self.key, CollisionKey):
            raise TypeError("resident sphere key must be a CollisionKey")
        object.__setattr__(self, "world", world_key(self.world))


@dataclass(frozen=True)
class ResidentStaticBinding:
    key: CollisionKey
    motion: CollisionMotion

    def __post_init__(self):
        if not isinstance(self.key, CollisionKey):
            raise TypeError("static key must be a CollisionKey")
        if not isinstance(self.motion, CollisionMotion):
            raise TypeError("static binding requires a CollisionMotion")


def _owned_readonly(value, dtype, shape):
    data = np.asarray(value, dtype=dtype)
    if data.shape != shape:
        raise ValueError("invalid resident collision result array shape")
    return np.frombuffer(data.tobytes(order="C"), dtype=dtype).reshape(shape)


@dataclass(frozen=True, eq=False)
class ResidentCollisionResult:
    """Current-pose broad candidates, with no claim of narrow contact or CCD."""

    owner: object
    registration: object
    topology_generation: int
    state_epoch: int
    world_keys: tuple
    ticks: np.ndarray
    shape_keys: tuple
    shape_revisions: tuple
    bounds: np.ndarray
    registered_pairs: tuple
    candidates: tuple

    def __post_init__(self):
        for name in (
            "world_keys",
            "shape_keys",
            "shape_revisions",
            "registered_pairs",
            "candidates",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(
            self, "ticks", _owned_readonly(self.ticks, np.int32, (len(self.world_keys),))
        )
        object.__setattr__(
            self, "bounds", _owned_readonly(self.bounds, np.float32, (len(self.shape_keys), 6))
        )


class _ResidentCollisionBatch(TyxCollisionBatch):
    """Account for every HOST copy, including interpretation and failed queries."""

    def __init__(self, *args, **kwargs):
        self.copied_to_host_bytes = 0
        self.host_read_calls = 0
        self.resident_device_bytes = 0
        super().__init__(*args, **kwargs)

    def _static(self, array):
        result = super()._static(array)
        if self._mode == "cuda":
            self.resident_device_bytes += int(result.nbytes)
        return result

    def _scratch(self, shape, dtype):
        result = super()._scratch(shape, dtype)
        if self._mode == "cuda":
            self.resident_device_bytes += int(result.nbytes)
        return result

    def _read(self, array):
        result = super()._read(array)
        self.copied_to_host_bytes += result.nbytes
        self.host_read_calls += 1
        return result

    def close(self):
        try:
            super().close()
        finally:
            self.resident_device_bytes = 0


def _capacity(value):
    maximum = MAX_SHAPES * (MAX_SHAPES - 1) // 2
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not 0 <= value <= maximum
    ):
        raise ValueError(f"candidate capacity must be an integer from 0 to {maximum}")
    return int(value)


class ResidentCollisionPlan:
    """One batch-owned, fixed-topology FreeSphere/static-geometry query plan.

    Each model key has one explicit binding. The only queried pair domain is a
    sphere and a static shape in the same world. Model masks, revisions and
    exclusions remain authoritative. Static/static pairs are outside that domain.
    """

    def __init__(self, batch, model, bindings, *, candidate_capacity=4096):
        from .tyx_world_batch import TyxWorldBatch

        if not isinstance(batch, TyxWorldBatch):
            raise TypeError("resident collision requires a TyxWorldBatch owner")
        if not isinstance(model, CollisionModel):
            raise TypeError("resident collision requires a CollisionModel")
        self._candidate_capacity = _capacity(candidate_capacity)
        self._batch = batch
        self._registration = object()
        self._model = model
        self._engine = None
        self._closed = False
        self._last = None
        self._kernels = None
        self._slots_device = self._static_device = self._stamp_device = None
        self._queries = self._failed_queries = self._bounds_launches = 0
        self._directions = set()
        self._trees = CollisionTrees()
        self._shape_keys = tuple(s.key for s in model.shapes)
        self._shape_revisions = tuple(s.revision for s in model.shapes)
        with batch._lock:
            batch._check_open()
            try:
                with batch._borrow_collision_state() as view:
                    self._owner = view.owner
                    self._topology_generation = view.topology_generation
                    slots, paths = self._validate_bindings(view, bindings)
                    self._world_keys = tuple(sorted(self._world_shape_rows))
                    self._tick_rows = tuple(self._world_shape_rows[w] for w in self._world_keys)
                    self._prepare_topology(slots)
                    if model.shapes:
                        self._prepare_device(view, slots, paths)
                batch._collision_plans.add(self)
            except BaseException:
                self.close()
                raise

    def _validate_bindings(self, view, bindings):
        try:
            rows = tuple(bindings)
        except TypeError as error:
            raise TypeError("collision bindings must be a finite sequence") from error
        if len(rows) != len(self._shape_keys):
            raise ValueError("bindings must cover exactly every collision model key")
        if not all(isinstance(b, (ResidentSphereBinding, ResidentStaticBinding)) for b in rows):
            raise TypeError("unsupported resident collision binding")
        by_key = {b.key: b for b in rows}
        if len(by_key) != len(rows) or set(by_key) != set(self._shape_keys):
            raise ValueError("bindings have duplicate or missing collision keys")
        slots = np.full(len(rows), -1, dtype=np.int32)
        paths = {}
        self._world_shape_rows = {}
        for row, shape in enumerate(self._model.shapes):
            binding = by_key[shape.key]
            if isinstance(binding, ResidentSphereBinding):
                if binding.world != shape.key.world or binding.world not in view.slots:
                    raise ValueError("sphere binding world must match its collision key and owner")
                if binding.world in self._world_shape_rows:
                    raise ValueError("at most one resident sphere may be bound in each world")
                slot = view.slots[binding.world]
                physical = view.bindings[slot]
                if physical.kind != 0 or shape.kind != "sphere":
                    raise ValueError(
                        "C1 requires an actual FreeSphere and a sphere collision shape"
                    )
                radius = physical.topology[3][7]
                if shape.radius != radius:
                    raise ValueError(
                        "collision sphere radius must equal the actual resident FP32 radius"
                    )
                slots[row] = slot
                self._world_shape_rows[binding.world] = row
                # Immutable initial metadata seeds only the tree's split order.
                # Every query refits from the actual accepted DEVICE buffer.
                initial = self._batch._initial[binding.world]
                paths[shape.key] = CollisionMotion([initial.position])
            else:
                if shape.kind == "plane":
                    raise ValueError(
                        "C1 static geometry must be finite; half-spaces are unsupported"
                    )
                if binding.motion.end is not None or binding.motion.turn is not None:
                    raise ValueError("static binding requires one pose, not a moving interval")
                paths[shape.key] = binding.motion
        if {key.world for key in self._shape_keys} != set(self._world_shape_rows):
            raise ValueError(
                "each static geometry world needs its explicitly bound resident sphere"
            )
        return slots, paths

    def _prepare_topology(self, slots):
        shapes = self._model.shapes
        self._tags = np.asarray(
            [(s.key.world, s.key.body, int(s.rigid), s.group, s.mask, 0) for s in shapes],
            dtype=np.int32,
        ).reshape(-1, 6)
        self._offsets = np.asarray(
            [0, *np.cumsum([len(s.vertex_ids) for s in shapes], dtype=np.int32)], dtype=np.int32
        )
        excluded = set(self._model.excluded)
        pairs, registered = [], []
        for i, j in combinations(range(len(shapes)), 2):
            keys = (shapes[i].key, shapes[j].key)
            in_domain = keys[0].world == keys[1].world and ((slots[i] >= 0) != (slots[j] >= 0))
            if in_domain:
                registered.append(keys)
            pairs.append((i, j, int(not in_domain or keys in excluded)))
        self._pairs = np.asarray(pairs, dtype=np.int32).reshape(-1, 3)
        self._registered_pairs = tuple(registered)
        self._space = SpatialPairs(self._tags, self._pairs, "bvh") if len(self._pairs) else None

    def _prepare_device(self, view, slots, paths):
        from ._tyx_resident_bounds import make_resident_bounds_kernels

        envelopes = _envelopes(self._model, paths, 0)
        seed_bounds = _host_bounds(self._offsets, envelopes)
        self._kernels = make_resident_bounds_kernels()
        self._engine = _ResidentCollisionBatch(
            view.mode,
            self._offsets,
            self._tags,
            self._pairs,
            None,
            spatial=self._space,
            runtime=view.runtime,
        )
        self._slots_device = self._engine._static(slots)
        self._static_device = self._engine._static(seed_bounds)
        self._stamp_device = self._engine._scratch((len(slots), 2), np.int32)
        if self._space is not None:
            tree = BoundsTree(seed_bounds, self._space.worlds, self._space.tags[:, 5])
            self._trees.layouts[("shape",)] = tree
            self._engine.prepare_shape_bvh(tree)

    @property
    def last_result(self):
        with self._batch._lock:
            return self._last

    def _check_open(self):
        if self._closed:
            raise RuntimeError("resident collision plan is closed")
        self._batch._check_open()

    def require_current(self, result):
        """Reject another owner, changed topology, or any earlier accepted epoch."""
        with self._batch._lock:
            self._check_open()
            if not isinstance(result, ResidentCollisionResult):
                raise TypeError("expected a ResidentCollisionResult")
            if result.registration is not self._registration:
                raise ValueError("collision result belongs to a different plan registration")
            if result.owner is not self._owner:
                raise ValueError("collision result belongs to a different owner")
            if (
                result.topology_generation != self._topology_generation
                or result.shape_keys != self._shape_keys
                or result.shape_revisions != self._shape_revisions
                or result.registered_pairs != self._registered_pairs
                or result.world_keys != self._world_keys
            ):
                raise ValueError("collision result belongs to a different topology")
            with self._batch._borrow_collision_state(
                owner=self._owner, topology_generation=self._topology_generation
            ) as view:
                if result.state_epoch != view.state_epoch:
                    raise ValueError("stale resident collision state epoch")
            return result

    def _read_bounds(self, view):
        engine = self._engine
        self._batch._launch(
            self._kernels[view.active],
            len(self._shape_keys),
            self._slots_device,
            self._static_device,
            view.states,
            view.ticks,
            view.props,
            engine._bounds,
            self._stamp_device,
        )
        self._bounds_launches += 1
        self._directions.add(view.active)
        self._batch._sync()
        stamp = engine._read(self._stamp_device)
        if stamp.shape != (len(self._shape_keys), 2) or stamp.dtype != np.int32:
            raise ValueError("invalid resident bounds status and tick readback")
        if np.any(stamp[:, 0] != 1) or any(stamp[row, 1] < 0 for row in self._tick_rows):
            raise ValueError("resident sphere bound overflow or non-finite accepted state")
        ticks = stamp[np.asarray(self._tick_rows, dtype=np.int32), 1]
        pending = dict(self._trees.layouts)
        bounds, flags = engine.query_device_bounds(trees=self._trees, pending=pending)
        if (
            bounds.shape != (len(self._shape_keys), 6)
            or bounds.dtype != np.float32
            or not np.isfinite(bounds).all()
            or np.any(bounds[:, :3] > bounds[:, 3:])
        ):
            raise ValueError("device collision bounds must be finite ordered FP32 intervals")
        if flags.shape != (len(self._pairs),) or flags.dtype != np.int32:
            raise ValueError("invalid resident collision candidate readback")
        if self._space is not None:
            defaults = self._space.defaults
            if np.any((flags < 0) | (flags > 5)) or np.any((defaults != 5) & (flags != defaults)):
                raise ValueError("device candidate readback violated model filters")
        candidates = tuple(
            (self._shape_keys[i], self._shape_keys[j])
            for (i, j, _), flag in zip(self._pairs, flags, strict=True)
            if flag == 0
        )
        if not set(candidates) <= set(self._registered_pairs):
            raise ValueError("device candidate escaped registered dynamic/static pair domain")
        if len(candidates) > self._candidate_capacity:
            raise ValueError(
                f"resident collision candidate capacity exceeded: {len(candidates)} > {self._candidate_capacity}"
            )
        return bounds, ticks, candidates, pending

    def query(self):
        """Publish a whole result only after genuine device bounds/BVH validation."""
        with self._batch._lock:
            self._check_open()
            try:
                with self._batch._borrow_collision_state(
                    owner=self._owner, topology_generation=self._topology_generation
                ) as view:
                    if self._shape_keys:
                        bounds, ticks, candidates, pending = self._read_bounds(view)
                    else:
                        bounds = np.empty((0, 6), dtype=np.float32)
                        ticks = np.empty(0, dtype=np.int32)
                        candidates, pending = (), dict(self._trees.layouts)
                    result = ResidentCollisionResult(
                        view.owner,
                        self._registration,
                        view.topology_generation,
                        view.state_epoch,
                        self._world_keys,
                        ticks,
                        self._shape_keys,
                        self._shape_revisions,
                        bounds,
                        self._registered_pairs,
                        candidates,
                    )
                # No borrowed current-buffer view is retained in the result or plan.
                self._trees.layouts = pending
                self._last = result
                self._queries += 1
                return result
            except Exception:
                self._failed_queries += 1
                raise

    def stats(self):
        with self._batch._lock:
            engine = self._engine
            counts = engine.stats() if engine is not None else {}
            return dict(
                mode=self._batch._mode,
                closed=self._closed,
                shape_count=len(self._shape_keys),
                registered_pair_count=len(self._registered_pairs),
                queries=self._queries,
                failed_queries=self._failed_queries,
                bounds_kernel_launches=self._bounds_launches,
                state_buffer_directions=tuple(sorted(self._directions)),
                device_allocations=counts.get("allocations", 0)
                if self._batch._mode == "cuda"
                else 0,
                device_owned_bytes=engine.resident_device_bytes if engine is not None else 0,
                static_uploads=counts.get("static_uploads", 0),
                uploaded_bytes=counts.get("uploaded_bytes", 0),
                metadata_uploaded_bytes=counts.get("spatial_uploaded_bytes", 0),
                # No call on this path uploads current dynamic envelopes.
                dynamic_envelope_uploaded_bytes=0,
                downloaded_bytes=counts.get("downloaded_bytes", 0),
                copied_to_host_bytes=engine.copied_to_host_bytes if engine is not None else 0,
                host_read_calls=engine.host_read_calls if engine is not None else 0,
                bvh_completed_launches=counts.get("spatial_completed_launches", 0),
                **self._trees.counts,
            )

    def close(self):
        with self._batch._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._engine is not None:
                    self._engine.close()
            finally:
                self._batch._collision_plans.discard(self)
                self._kernels = self._slots_device = self._static_device = self._stamp_device = None

    def __enter__(self):
        with self._batch._lock:
            self._check_open()
        return self

    def __exit__(self, *_):
        self.close()


def collision_for(batch, plan):
    """The batch-facing owner gate precedes any device borrow or query."""
    if not isinstance(plan, ResidentCollisionPlan):
        raise TypeError("expected a ResidentCollisionPlan")
    with batch._lock:
        batch._check_open()
        if plan._batch is not batch:
            raise ValueError("collision plan belongs to a different batch owner")
        return plan.query()
