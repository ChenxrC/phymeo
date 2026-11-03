"""Collision diagnostics with one packed all-pairs broad phase across disjoint worlds."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import groupby
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
from .collision_pipeline import (
    CollisionDetection,
    CollisionWorkspace,
    _detect_pairs,
    _detection_result,
    _envelopes,
    _host_bounds,
)


@dataclass(frozen=True)
class _World:
    key: int
    model: CollisionModel
    keys: frozenset
    shapes: slice
    vertices: slice
    pairs: slice


class WorldCollisionWorkspace:
    """One same-world diagnostic table, shared by one HOST, interpret or CUDA engine.

    Register (world_id, CollisionModel) pairs. Each shape's existing world key
    must match its registration. detect() takes mappings for *all* registrations,
    including empty models. Each returned CollisionDetection uses the original
    keys, order of local shapes, witnesses and geometry conventions of detect().

    A locally over-capacity attempt returns the measured requirement and no rows.
    Its previous successful result stays accessible via last_result(). A backend
    fault discards the entire attempt. The next valid call rebuilds that engine.
    """

    def __init__(self, worlds, backend="host", *, cache_root=None):
        if backend not in ("host", "interpret", "cuda"):
            raise ValueError("collision backend must be host, interpret or cuda")
        models = {}
        shape_count = vertex_count = 0
        for world, model in worlds:
            key = _integer(world, "world")
            if key in models or not isinstance(model, CollisionModel):
                raise ValueError("register each world once with a CollisionModel")
            if any(s.key.world != key for s in model.shapes):
                raise ValueError("shape world must match its registration")
            shape_count += len(model.shapes)
            vertex_count += sum(len(s.vertex_ids) for s in model.shapes)
            if shape_count > MAX_SHAPES or vertex_count > MAX_VERTICES:
                raise ValueError("aggregate collision shape or vertex count outside limits")
            models[key] = model
        self._order = tuple(models)
        self._keys = frozenset(models)
        # CollisionModel keeps the existing aggregate entity limits and sorts full
        # keys, placing each world's shapes in a contiguous slice.
        self._model = CollisionModel(
            tuple(s for m in models.values() for s in m.shapes),
            tuple(pair for m in models.values() for pair in m.excluded),
        )
        self._worlds = []
        i = v = p = 0
        for key in sorted(models):
            m = models[key]
            n = len(m.shapes)
            next_v = v + sum(len(s.vertex_ids) for s in m.shapes)
            next_p = p + n * (n - 1) // 2
            self._worlds.append(
                _World(
                    key,
                    m,
                    frozenset(s.key for s in m.shapes),
                    slice(i, i + n),
                    slice(v, next_v),
                    slice(p, next_p),
                )
            )
            i, v, p = i + n, next_v, next_p
        self._pairs = np.empty((p, 3), dtype=np.int32)
        excluded = set(self._model.excluded)
        shapes = self._model.shapes
        for part in self._worlds:
            slot = part.pairs.start
            for first in range(part.shapes.start, part.shapes.stop):
                for second in range(first + 1, part.shapes.stop):
                    self._pairs[slot] = (
                        first,
                        second,
                        (shapes[first].key, shapes[second].key) in excluded,
                    )
                    slot += 1
        self._pairs.flags.writeable = False
        self._vertex_count = v
        self._unpaired_offsets = (
            np.asarray(
                [0, *np.cumsum([len(s.vertex_ids) for s in shapes], dtype=np.int32)], dtype=np.int32
            )
            if not p
            else None
        )
        self._backend, self._cache_root = backend, cache_root
        self._lock = RLock()
        self._work = None
        self._closed = False
        self._last = dict.fromkeys(self._order)
        self._times = dict(envelope_ns=0, broad_ns=0, narrow_ns=0, engine_setup_ns=0)
        self._retired = Counter()
        self._builds = 0

    def _inputs(self, motion, policies):
        for value, name in ((motion, "motion"), (policies, "policy")):
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a world-keyed mapping")
            if frozenset(_integer(k, "world") for k in value) != self._keys:
                raise ValueError(f"{name} must cover exactly the registered worlds")
        rules, flat = dict(policies), {}
        for part in self._worlds:
            if not isinstance(rules[part.key], CollisionPolicy):
                raise TypeError("each world requires a CollisionPolicy")
            paths = motion[part.key]
            if not isinstance(paths, Mapping) or paths.keys() != part.keys:
                raise ValueError("motion keys must exactly cover each world's registered shapes")
            for shape in part.model.shapes:
                path = paths[shape.key]
                if not isinstance(path, CollisionMotion) or path.start.shape != (
                    len(shape.vertex_ids),
                    3,
                ):
                    raise ValueError("motion must supply all registered vertices in material order")
                flat[shape.key] = path
        return rules, flat

    def _packed_envelopes(self, paths, rules):
        margins = {rule.margin for rule in rules.values()}
        if len(margins) == 1:
            return _envelopes(self._model, paths, margins.pop())
        envelopes = np.empty((self._vertex_count, 6), np.float32)
        for part in self._worlds:
            envelopes[part.vertices] = _envelopes(part.model, paths, rules[part.key].margin)
        return envelopes

    def _results(self, work, paths, rules, bounds, flags):
        counts = {part.key: int(np.count_nonzero(flags[part.pairs] == 0)) for part in self._worlds}

        def route(part):
            rule = rules[part.key]
            return None if counts[part.key] > rule.candidate_capacity else rule.proximity_patches

        attempts = {}
        for proximity, members in groupby(self._worlds, route):
            group = tuple(members)
            if proximity is None:
                for part in group:
                    attempts[part.key] = CollisionDetection(
                        "capacity_exceeded", self._backend, counts[part.key], (), ()
                    )
                continue
            start, end = group[0].pairs.start, group[-1].pairs.stop
            rows = (
                _detect_pairs(work, paths, flags[start:end], {}, proximity, pair_start=start)
                if end > start
                else ()
            )
            for part in group:
                local_rows = rows[part.pairs.start - start : part.pairs.stop - start]
                attempts[part.key] = _detection_result(
                    part.model,
                    paths,
                    self._backend,
                    counts[part.key],
                    local_rows,
                    bounds[part.shapes],
                )
        return {key: attempts[key] for key in self._order}

    def _empty_results(self, paths, rules, bounds):
        results = {
            part.key: _detection_result(part.model, paths, self._backend, 0, (), bounds[part.shapes])
            for part in self._worlds
        }
        return {world: results[world] for world in self._order}

    def detect(self, motion_by_world, policy_by_world):
        """Classify the packed input once, then publish successful world observations together."""
        with self._lock:
            if self._closed:
                raise RuntimeError("world collision workspace is closed")
            rules, paths = self._inputs(motion_by_world, policy_by_world)
            if not self._model.shapes:
                empty = self._empty_results(paths, rules, np.empty((0, 6), dtype=np.float32))
                self._last = empty.copy()
                return empty
            t0 = perf_counter_ns()
            envelopes = self._packed_envelopes(paths, rules)
            self._times["envelope_ns"] += perf_counter_ns() - t0
            if not len(self._pairs):
                bounds = _host_bounds(self._unpaired_offsets, envelopes)
                ordered = self._empty_results(paths, rules, bounds)
                self._last = ordered.copy()
                return ordered
            if self._work is None:
                t0 = perf_counter_ns()
                self._work = CollisionWorkspace._from_world_pairs(
                    self._model, self._backend, self._pairs, cache_root=self._cache_root
                )
                self._times["engine_setup_ns"] += perf_counter_ns() - t0
                self._builds += 1
            try:
                t0 = perf_counter_ns()
                bounds, flags = self._work._classify_world_envelopes(envelopes)
                t1 = perf_counter_ns()
                self._times["broad_ns"] += t1 - t0
                attempts = self._results(self._work, paths, rules, bounds, flags)
                self._times["narrow_ns"] += perf_counter_ns() - t1
            except BaseException:
                self._retire_engine()
                raise
            published = self._last.copy()
            published.update(
                (key, result) for key, result in attempts.items() if result.status == "ok"
            )
            self._last = published
            return attempts

    def last_result(self, world_id):
        """Most recent accepted immutable observation for a registered world, or None."""
        with self._lock:
            return self._last[_integer(world_id, "world")]

    def _retire_engine(self):
        work, self._work = self._work, None
        if work is not None:
            self._retired.update({k: v for k, v in work.stats().items() if type(v) is int})
            work.close()

    def stats(self):
        """Actual pair/flag allocations, cumulative backend counters and HOST wall clocks."""
        with self._lock:
            fields = {} if self._work is None else self._work.stats()
            base = dict(
                completed_launches=0,
                launches=0,
                synchronizations=0,
                uploaded_bytes=0,
                downloaded_bytes=0,
                allocations=0,
                static_uploads=0,
                dynamic_uploads=0,
            )
            base.update(fields)
            for k, v in self._retired.items():
                base[k] = base.get(k, 0) + v
            engine = (
                self._work._engine if self._work is not None and self._backend == "cuda" else None
            )
            device_pairs = getattr(engine, "_pairs", None)
            device_flags = getattr(engine, "_flags", None)
            return dict(
                base,
                **self._times,
                backend=self._backend,
                world_count=len(self._order),
                shape_count=len(self._model.shapes),
                vertex_count=self._vertex_count,
                engine_builds=self._builds,
                host_pair_table_slots=len(self._pairs),
                host_pair_table_bytes=self._pairs.nbytes,
                device_pair_table_slots=device_pairs.shape[0] if device_pairs is not None else 0,
                device_pair_table_bytes=device_pairs.nbytes if device_pairs is not None else 0,
                device_flag_slots=device_flags.shape[0] if device_flags is not None else 0,
                device_flag_bytes=device_flags.nbytes if device_flags is not None else 0,
            )

    def close(self):
        with self._lock:
            self._closed = True
            self._retire_engine()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("world collision workspace is closed")
        return self

    def __exit__(self, *args):
        self.close()
