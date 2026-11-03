"""Batched finite cloth-face / prismatic-capsule contact forces and physical substeps."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from threading import RLock

import numpy as np

from ..core.capabilities import SolverFailure
from ._tyx_world_contact_engine import WorldContactEngine
from ._world_contact_sites import (
    ClothLinkWorld,
    WorldContactResult,
    WorldContactState,
    WorldMaterial,
)
from ._world_step_layout import as_fp32, world_key

__all__ = ["ClothLinkWorld", "TyxWorldContactBatch", "WorldContactResult", "WorldContactState"]


@dataclass
class _SamplePair:
    state: object
    mid: object
    entry_face: object
    mid_face: object
    external: np.ndarray
    spring: np.ndarray
    required: int


class TyxWorldContactBatch:
    """One finite face, one capsule on a scalar sliding link and optional distance springs per world.

    Original FP64 material geometry, path acceptance and publication belong to HOST. A shared
    TYX runtime gathers relative material velocities, evaluates the calibrated force, transposes
    it to both domains and integrates FP32 motion increments for the midpoint physical step.
    """

    def __init__(self, worlds, mode="cuda", *, cache_root=None):
        if mode not in ("cuda", "interpret"):
            raise ValueError("world contact batch requires cuda or interpret")
        self._mode, self._cache = mode, cache_root
        self._lock = RLock()
        self._closed = False
        self._worlds = {}
        for key, spec in worlds:
            key = world_key(key)
            if key in self._worlds:
                raise ValueError("duplicate world key")
            if len(self._worlds) >= np.iinfo(np.int32).max // 2:
                raise ValueError("material sites exceed TYX i32 capacity")
            self._worlds[key] = WorldMaterial(key, spec)
        self._points = {k: w.initial for k, w in self._worlds.items()}
        self._engine = None
        self._retired = {}
        self._builds = 0

    def _open(self):
        if self._closed:
            raise RuntimeError("world contact batch is closed")

    def _keys(self, values):
        keys = list(self._worlds) if values is None else [world_key(v) for v in values]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate selected world")
        for key in keys:
            if key not in self._worlds:
                raise KeyError(key)
        return keys

    def snapshot(self, keys=None):
        with self._lock:
            self._open()
            return {k: self._points[k] for k in self._keys(keys)}

    def restore(self, by_world):
        with self._lock:
            self._publish_points(self._stage_restore(by_world))

    def _stage_restore(self, by_world):
        self._open()
        if not isinstance(by_world, Mapping):
            raise TypeError("checkpoints must be keyed by world")
        points = dict(self._points)
        for k in self._keys(by_world):
            points[k] = self._worlds[k].validate(by_world[k])
        return points

    def reset(self, keys=None):
        with self._lock:
            self._open()
            self.restore({k: self._worlds[k].initial for k in self._keys(keys)})

    def _limits(self, capacities):
        caps = dict.fromkeys(self._worlds, 2)
        if capacities is not None:
            if not isinstance(capacities, Mapping):
                raise TypeError("site capacities must be a world mapping")
            for k in self._keys(capacities):
                c = capacities[k]
                if isinstance(c, (bool, np.bool_)) or not isinstance(c, Integral) or c < 0:
                    raise ValueError("site capacity must be a nonnegative integer")
                caps[k] = int(c)
        return caps

    def _retire(self):
        if self._engine is not None:
            engine, self._engine = self._engine, None
            for name, count in engine.counts.items():
                self._retired[name] = self._retired.get(name, 0) + count
            engine.close()

    def _evaluate(self, velocity, stages, enabled, h):
        if self._engine is None:
            self._engine = WorldContactEngine(tuple(self._worlds.values()), self._mode, self._cache)
            self._builds += 1
        try:
            return self._engine.evaluate(velocity, stages, enabled, h)
        except Exception:
            self._retire()
            raise

    def _frames(self, h, inputs, caps):
        n = len(self._worlds)
        enabled = np.zeros(n, np.int32)
        velocity = np.zeros((n, 4, 3), np.float32)
        count = 1 if inputs is None else 2
        stages = [
            (np.zeros((2 * n, 13), np.float32), np.zeros((n, 4, 3), np.float32))
            for _ in range(count)
        ]
        frames, failures = {}, {}
        for i, (key, world) in enumerate(self._worlds.items()):
            old = self._points[key]
            state = world.scene(old.position, old.velocity, old.q, old.qd)
            external = world.loads(state, inputs[key]) if inputs is not None else np.zeros((4, 3))
            velocity[i, :3] = as_fp32(old.velocity, "accepted nodal velocity", (3, 3))
            velocity[i, 3, 0] = old.qd
            mid = state
            if inputs is not None:
                mid = world.scene(
                    np.asarray(old.position) + 0.5 * h * np.asarray(old.velocity),
                    old.velocity,
                    old.q + 0.5 * h * old.qd,
                    old.qd,
                )
            try:
                entry_face = world.geometry(state)
                mid_face = world.geometry(mid) if inputs is not None else entry_face
                required = max(len(entry_face.sites), len(mid_face.sites))
                if required > caps[key]:
                    failures[key] = WorldContactResult(
                        False, required, "site_capacity", old.last_audit
                    )
                    continue
                states = (state, mid)
                geometries = (entry_face, mid_face)
                spring = np.zeros((3, 3))
                for index, (table, other) in enumerate(stages):
                    geom = geometries[index]
                    table[2 * i : 2 * i + 2] = as_fp32(
                        world.site_rows(geom), "normal layer geometry", (2, 13)
                    )
                    spring = world.elasticity(states[index])
                    load = np.zeros((4, 3))
                    load[:3] = external[:3] + spring
                    load[3, 0] = external[3] @ world.axis
                    other[i] = as_fp32(load, "external and elastic load", (4, 3))
            except (SolverFailure, FloatingPointError, ValueError) as error:
                if inputs is None:
                    raise
                failures[key] = WorldContactResult(False, 0, str(error), old.last_audit)
                continue
            enabled[i] = 1
            frames[key] = _SamplePair(state, mid, entry_face, mid_face, external, spring, required)
        return frames, failures, velocity, stages, enabled

    def sample(self):
        """Return the physical load at each accepted state without advancing either domain."""
        with self._lock:
            self._open()
            if not self._worlds:
                return {}
            frames, _, v, stages, enabled = self._frames(0.0, None, self._limits(None))
            rows, forces, mapped, status = self._evaluate(v, stages, enabled, 0.0)
            if not np.all(status) or not all(np.isfinite(a).all() for a in (rows, forces, mapped)):
                raise SolverFailure("nonfinite TYX material force")
            return {
                k: w.sample(frames[k].entry_face, rows[i], forces[i], mapped[i])
                for i, (k, w) in enumerate(self._worlds.items())
            }

    def step(self, h, loads, *, site_capacity=None):
        """Advance co-located worlds; rejected worlds retain their last accepted state and audit."""
        with self._lock:
            points, results = self._stage_step(h, loads, site_capacity=site_capacity)
            self._publish_points(points)
            return results

    def _publish_points(self, points):
        # The physical checkpoint and its audit are immutable. All device work is done.
        self._points = points

    def _stage_step(self, h, loads, *, site_capacity=None):
        """Prepare a complete material result; the owning stepper controls publication."""
        self._open()
        h = float(as_fp32(h, "physical substep", ()))
        if h <= 0:
            raise ValueError("physical substep must be positive")
        if not isinstance(loads, Mapping) or set(self._keys(loads)) != set(self._worlds):
            raise ValueError("every world requires physical loads")
        caps = self._limits(site_capacity)
        if not self._worlds:
            return self._points, {}
        frames, failures, v, stages, enabled = self._frames(h, loads, caps)
        if not frames:
            return self._points, {key: failures[key] for key in self._worlds}
        rows, forces, mapped, status, dx, dv = self._evaluate(v, stages, enabled, h)
        points = dict(self._points)
        results = {}
        for i, (key, world) in enumerate(self._worlds.items()):
            old = self._points[key]
            if key in failures:
                results[key] = failures[key]
                continue
            f = frames[key]
            try:
                if not status[i] or not all(
                    np.isfinite(a[i]).all() for a in (rows, forces, mapped, dx, dv)
                ):
                    raise SolverFailure("nonfinite candidate from TYX contact operator")
                # Add device-integrated increments to the original FP64 material coordinates.
                end = world.scene(
                    np.asarray(old.position) + dx[i, :3],
                    np.asarray(old.velocity) + dv[i, :3],
                    old.q + float(dx[i, 3, 0]),
                    old.qd + float(dv[i, 3, 0]),
                )
                final_face = world.geometry(end)
                max_depth = world.p.max_penetration_m
                if (
                    max_depth is not None
                    and min(f.entry_face.gap, f.mid_face.gap, final_face.gap) < -max_depth
                ):
                    raise SolverFailure("sampled cloth-link penetration exceeds limit")
                world.p._check_link_motion(
                    f.state, f.mid, end, None, world.p._link_geometry(end)
                )
                sample = world.sample(f.mid_face, rows[i], forces[i], mapped[i])
                audit = world.audit(
                    old, end, f.entry_face, final_face, sample, f.external, f.spring, h
                )
                cloth = end.domains[world.cloth]
                arm = end.domains[world.robot]
                points[key] = world.point(
                    cloth.position, cloth.velocity, arm.q[0], arm.v[0], old.tick + 1, audit
                )
                results[key] = WorldContactResult(True, f.required, None, audit)
            except (SolverFailure, FloatingPointError, ValueError) as error:
                results[key] = WorldContactResult(False, f.required, str(error), old.last_audit)
        return points, results

    def stats(self):
        with self._lock:
            acc = dict(self._retired)
            if self._engine is not None:
                for name, count in self._engine.counts.items():
                    acc[name] = acc.get(name, 0) + count
            return dict(
                acc,
                backend=self._mode,
                worlds=len(self._worlds),
                site_slots=2 * len(self._worlds),
                engine_builds=self._builds,
            )

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._retire()
