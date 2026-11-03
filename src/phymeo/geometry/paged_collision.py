"""Batched collision diagnostics for populations that span several bounded whole-world pages."""

from collections import Counter, OrderedDict
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from numbers import Integral
from threading import RLock
from time import perf_counter_ns

import numpy as np

from ..constraints.collision_shapes import (
    MAX_SHAPES,
    MAX_VERTICES,
    CollisionModel,
    CollisionMotion,
    CollisionPolicy,
    _integer,
)
from .batched_collision import WorldCollisionWorkspace
from .compact_contacts import CompactWorldContactWorkspace, _contact_limits

__all__ = [
    "CollisionPage",
    "CollisionPageCapacityError",
    "CollisionPageLimits",
    "PagedWorldCollisionWorkspace",
]

_MAX_PAIRS = MAX_SHAPES * (MAX_SHAPES - 1) // 2
_COUNTERS = (
    "allocations",
    "static_uploads",
    "dynamic_uploads",
    "downloads",
    "uploaded_bytes",
    "downloaded_bytes",
    "launches",
    "completed_launches",
    "synchronizations",
    "engine_builds",
    "envelope_ns",
    "broad_ns",
    "narrow_ns",
    "engine_setup_ns",
    "sweep_allocations",
    "sweep_completed_launches",
    "sweep_uploaded_bytes",
    "sweep_downloaded_bytes",
    "hull_completed_launches",
    "hull_uploaded_bytes",
    "hull_downloaded_bytes",
    "validated_mesh_attempts",
    "validated_mesh_builds",
    "validated_mesh_hits",
    "compact_selected_slots",
    "compact_validation_slots",
    "compact_skipped_slots",
)


def _broad_bytes(shapes, vertices, pairs):
    # For P>0, the existing TYX all-pairs engine owns six arrays of 4-byte words:
    # offsets (S+1), tags (6S), per-vertex envelopes (6V), bounds (6S), pairs (3P), flags (P).
    return 4 * (1 + 13 * shapes + 6 * vertices + 4 * pairs) if pairs else 0


class CollisionPageCapacityError(ValueError):
    """An entire world, or the requested complete result, exceeds a declared budget."""

    def __init__(self, resource, required, limit, world=None):
        self.resource, self.required, self.limit, self.world = resource, required, limit, world
        owner = "population" if world is None else f"world {world}"
        super().__init__(f"{owner} requires {required} {resource}, budget {limit}")


@dataclass(frozen=True)
class CollisionPageLimits:
    """Per-page geometry limits, complete-query pair-row budget and workspace residency.

    resident_broad_bytes limits the six base CUDA broad-phase arrays across residents.
    Shape-specific narrow-phase scratch and caller-retained output have separate lifetimes.
    """

    shapes: int = MAX_SHAPES
    vertices: int = MAX_VERTICES
    pairs: int = _MAX_PAIRS
    result_pairs: int = _MAX_PAIRS
    resident_pages: int = 2
    resident_broad_bytes: int | None = None

    def __post_init__(self):
        for name, low, high in (
            ("shapes", 1, MAX_SHAPES),
            ("vertices", 1, MAX_VERTICES),
            ("pairs", 0, _MAX_PAIRS),
            ("result_pairs", 0, None),
            ("resident_pages", 1, None),
            ("resident_broad_bytes", 0, None),
        ):
            value = getattr(self, name)
            if name == "resident_broad_bytes" and value is None:
                continue
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or value < low
                or (high is not None and value > high)
            ):
                raise ValueError(f"invalid collision page budget {name}")
            object.__setattr__(self, name, int(value))

    @property
    def broad_byte_budget(self):
        if self.resident_broad_bytes is not None:
            return self.resident_broad_bytes
        return self.resident_pages * _broad_bytes(self.shapes, self.vertices, self.pairs)


@dataclass(frozen=True)
class CollisionPage:
    index: int
    world_keys: tuple[int, ...]
    shapes: int
    vertices: int
    pair_slots: int
    planned_broad_bytes: int


def _overflow(limits, s, v, p):
    for name, count, cap in (
        ("shapes", s, limits.shapes),
        ("vertices", v, limits.vertices),
        ("pairs", p, limits.pairs),
        ("resident_broad_bytes", _broad_bytes(s, v, p), limits.broad_byte_budget),
    ):
        if count > cap:
            return name, count, cap
    return None


def _plan(worlds, limits):
    models, keys, pages = {}, {}, []
    batch = []
    s = v = p = total = 0

    def finish():
        pages.append(CollisionPage(len(pages), tuple(batch), s, v, p, _broad_bytes(s, v, p)))

    for key, model in worlds:
        key = _integer(key, "world")
        if key in models or not isinstance(model, CollisionModel):
            raise ValueError("register each world once with a CollisionModel")
        if any(shape.key.world != key for shape in model.shapes):
            raise ValueError("shape world must match its registration")
        ns = len(model.shapes)
        nv = sum(len(shape.vertex_ids) for shape in model.shapes)
        npairs = ns * (ns - 1) // 2
        failure = _overflow(limits, ns, nv, npairs)
        if failure:
            raise CollisionPageCapacityError(*failure, world=key)
        if total + npairs > limits.result_pairs:
            raise CollisionPageCapacityError("result_pairs", total + npairs, limits.result_pairs)
        if batch and _overflow(limits, s + ns, v + nv, p + npairs):
            finish()
            s = v = p = 0
            batch = []
        batch.append(key)
        s += ns
        v += nv
        p += npairs
        total += npairs
        models[key] = model
        keys[key] = frozenset(shape.key for shape in model.shapes)
    if batch:
        finish()
    return models, keys, tuple(pages), total


def _actual_broad_bytes(work):
    leaf = work._work
    engine = None if leaf is None else leaf._engine
    if engine is None or work._backend != "cuda":
        return 0
    return sum(
        getattr(engine, name).nbytes
        for name in ("_offsets", "_tags", "_envelopes", "_bounds", "_pairs", "_flags")
    )


class PagedWorldCollisionWorkspace:
    """Run one WorldCollisionWorkspace per resident page and commit one complete query.

    Worlds stay whole and follow registration order. Resident leaves are acquired lazily
    and evicted in least-recently-used order to satisfy the explicit page and byte budgets.
    Only this object's observations are public. A later page failure publishes no earlier
    page results, even when an earlier leaf has already been evicted. Use result_format='contacts'
    to retain contact and diagnostic rows on each page. The page budgets still bound pair slots.
    """

    def __init__(self, worlds, backend="host", *, limits=None, cache_root=None, result_format="full"):
        if result_format not in ("full", "contacts"):
            raise ValueError("result_format must be full or contacts")
        self._result_format = result_format
        if backend not in ("host", "interpret", "cuda"):
            raise ValueError("collision backend must be host, interpret or cuda")
        self._limits = CollisionPageLimits() if limits is None else limits
        if not isinstance(self._limits, CollisionPageLimits):
            raise TypeError("limits must be CollisionPageLimits")
        self._models, self._keys, self._layout, self._total_pairs = _plan(worlds, self._limits)
        self._backend, self._cache = backend, cache_root
        self._resident = OrderedDict()
        self._retired = [Counter() for _ in self._layout]
        self._metrics = [Counter() for _ in self._layout]
        self._actual = [0] * len(self._layout)
        self._published = dict.fromkeys(self._models)
        self._lock = RLock()
        self._closed = False
        self._counts = Counter()
        self._last_failure = None
        self._planned = 0
        self._live_actual = 0

    @property
    def layout(self):
        return self._layout

    @property
    def limits(self):
        return self._limits

    def _inputs(self, motion, policies):
        for mapping in (motion, policies):
            if not isinstance(mapping, Mapping):
                raise TypeError("motion and policies must be world-keyed mappings")
            if {_integer(k, "world") for k in mapping} != self._models.keys():
                raise ValueError("inputs must cover exactly the registered worlds")
        packed = {}
        rules = dict(policies)
        for k, model in self._models.items():
            if not isinstance(rules[k], CollisionPolicy):
                raise TypeError("every world requires a CollisionPolicy")
            paths = motion[k]
            if not isinstance(paths, Mapping) or paths.keys() != self._keys[k]:
                raise ValueError("motion keys must cover the world's registered shapes")
            local = dict(paths)
            for shape in model.shapes:
                path = local[shape.key]
                if not isinstance(path, CollisionMotion) or path.start.shape != (
                    len(shape.vertex_ids),
                    3,
                ):
                    raise ValueError("motion must supply all registered vertices in material order")
            packed[k] = local
        return packed, rules

    def _release(self, index):
        work = self._resident.pop(index)
        self._planned -= self._layout[index].planned_broad_bytes
        self._live_actual -= self._actual[index]
        self._actual[index] = 0
        try:
            stats = work.stats()
            self._retired[index].update({k: stats.get(k, 0) for k in _COUNTERS})
        finally:
            work.close()

    def _release_all(self):
        # ExitStack still disposes later resources if a driver cleanup itself fails.
        with ExitStack() as stack:
            for index in tuple(self._resident):
                stack.callback(self._release, index)

    def _acquire(self, page):
        i = page.index
        if i in self._resident:
            self._resident.move_to_end(i)
            return self._resident[i]
        while self._resident and (
            len(self._resident) >= self._limits.resident_pages
            or self._planned + page.planned_broad_bytes > self._limits.broad_byte_budget
        ):
            oldest = next(iter(self._resident))
            self._metrics[oldest]["evictions"] += 1
            self._release(oldest)
        t0 = perf_counter_ns()
        factory = CompactWorldContactWorkspace if self._result_format == "contacts" else WorldCollisionWorkspace
        work = factory(
            [(k, self._models[k]) for k in page.world_keys], self._backend, cache_root=self._cache
        )
        self._resident[i] = work
        self._planned += page.planned_broad_bytes
        self._metrics[i]["build_ns"] += perf_counter_ns() - t0
        self._metrics[i]["builds"] += 1
        self._counts["peak_resident_pages"] = max(
            self._counts["peak_resident_pages"], len(self._resident)
        )
        self._counts["peak_planned_broad_bytes"] = max(
            self._counts["peak_planned_broad_bytes"], self._planned
        )
        return work

    def detect(self, motion_by_world, policy_by_world, *, limits_by_world=None):
        """Query all pages. Optional per-world contact/diagnostic budgets apply to contacts output."""
        with self._lock:
            if self._closed:
                raise RuntimeError("paged collision workspace is closed")
            try:
                motion, rules = self._inputs(motion_by_world, policy_by_world)
                if limits_by_world is not None and self._result_format != "contacts":
                    raise ValueError("limits_by_world requires contacts output")
                overrides = _contact_limits(limits_by_world, self._models)
            except (TypeError, ValueError, KeyError):
                self._counts["input_rejections"] += 1
                raise
            attempts = {}
            i = None
            try:
                for page in self._layout:
                    i = page.index
                    work = self._acquire(page)
                    metric = self._metrics[i]
                    local_motion = {k: motion[k] for k in page.world_keys}
                    local_rules = {k: rules[k] for k in page.world_keys}
                    metric["dispatches"] += 1
                    t0 = perf_counter_ns()
                    try:
                        if overrides:
                            local_limits = {k: overrides[k] for k in page.world_keys if k in overrides}
                            output = work.detect(local_motion, local_rules, limits_by_world=local_limits)
                        else:
                            output = work.detect(local_motion, local_rules)
                    finally:
                        metric["dispatch_ns"] += perf_counter_ns() - t0
                    metric["completed_dispatches"] += 1
                    allocated = _actual_broad_bytes(work)
                    self._live_actual += allocated - self._actual[i]
                    self._actual[i] = allocated
                    if allocated > page.planned_broad_bytes:
                        raise RuntimeError("allocated broad-phase storage exceeded the page plan")
                    self._counts["peak_broad_device_bytes"] = max(
                        self._counts["peak_broad_device_bytes"], self._live_actual
                    )
                    metric["peak_broad_device_bytes"] = max(
                        metric["peak_broad_device_bytes"], self._actual[i]
                    )
                    attempts.update(output)
                i = None
                published = self._published.copy()
                published.update((k, r) for k, r in attempts.items() if r.status == "ok")
            except BaseException as error:
                self._counts["failed_queries"] += 1
                self._last_failure = dict(page=i, type=type(error).__name__, message=str(error))
                try:
                    self._release_all()
                except Exception as cleanup:
                    error.add_note(f"page cleanup failed with {cleanup!r}")
                raise
            self._published = published
            self._counts["completed_queries"] += 1
            return attempts

    def last_result(self, world_id):
        with self._lock:
            return self._published[_integer(world_id, "world")]

    def stats(self):
        with self._lock:
            pages = []
            summary = Counter()
            for page in self._layout:
                i = page.index
                work = self._resident.get(i)
                current = {} if work is None else work.stats()
                cumulative = {k: self._retired[i][k] + current.get(k, 0) for k in _COUNTERS}
                summary.update(cumulative)
                metric = self._metrics[i]
                pages.append(
                    dict(
                        **asdict(page),
                        resident=work is not None,
                        actual_broad_device_bytes=self._actual[i],
                        peak_broad_device_bytes=metric["peak_broad_device_bytes"],
                        builds=metric["builds"],
                        dispatches=metric["dispatches"],
                        completed_dispatches=metric["completed_dispatches"],
                        evictions=metric["evictions"],
                        build_ns=metric["build_ns"],
                        dispatch_ns=metric["dispatch_ns"],
                        host_pair_table_slots=current.get("host_pair_table_slots", 0),
                        host_pair_table_bytes=current.get("host_pair_table_bytes", 0),
                        device_pair_table_slots=current.get("device_pair_table_slots", 0),
                        device_pair_table_bytes=current.get("device_pair_table_bytes", 0),
                        device_flag_bytes=current.get("device_flag_bytes", 0),
                        backend_counts=cumulative,
                    )
                )
            return dict(
                backend=self._backend,
                result_format=self._result_format,
                world_count=len(self._models),
                result_pair_slots=self._total_pairs,
                pages=pages,
                resident_page_limit=self._limits.resident_pages,
                broad_byte_budget=self._limits.broad_byte_budget,
                resident_pages=len(self._resident),
                peak_resident_pages=self._counts["peak_resident_pages"],
                resident_device_pair_slots=sum(p["device_pair_table_slots"] for p in pages),
                resident_host_pair_slots=sum(p["host_pair_table_slots"] for p in pages),
                resident_broad_device_bytes=self._live_actual,
                peak_broad_device_bytes=self._counts["peak_broad_device_bytes"],
                peak_planned_broad_bytes=self._counts["peak_planned_broad_bytes"],
                page_builds=sum(p["builds"] for p in pages),
                page_dispatches=sum(p["dispatches"] for p in pages),
                evictions=sum(p["evictions"] for p in pages),
                completed_queries=self._counts["completed_queries"],
                failed_queries=self._counts["failed_queries"],
                input_rejections=self._counts["input_rejections"],
                completed_launches=summary["completed_launches"],
                backend_counts=dict(summary),
                last_failure=None if self._last_failure is None else dict(self._last_failure),
            )

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._release_all()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("paged collision workspace is closed")
        return self

    def __exit__(self, *args):
        self.close()
