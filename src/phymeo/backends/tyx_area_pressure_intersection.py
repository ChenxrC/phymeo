"""Persistent TYX clipping proposals with exact, complete boundary refinement.

HOST validation, coefficient construction and publication remain explicit. Empty
or unproven output uses the chosen exact-host/reject policy, never a silent miss.
"""

from fractions import Fraction
from itertools import combinations
from threading import RLock
from time import perf_counter
from types import MappingProxyType

import numpy as np

from ..geometry import _area_pressure_clipping as exact
from ..geometry.area_pressure import MAX_TRIANGLES, AreaPressureSurface, _identity, _stamp
from ..geometry.area_pressure_intersection import (
    MAX_CANDIDATE_PAIRS,
    AreaPressureIntersection,
    _meshes,
    _owns_polygon,
    _publish_triangle,
)
from .mujoco_tyx._resources import DeviceResources


class _Unresolved(ValueError):
    pass


def _cross2(a, b):
    return exact.subtract(exact.multiply(a[0], b[1]), exact.multiply(a[1], b[0]))


def _refine_polygon(coefficients, face, active):
    """Positive polygon certificate: feasible vertices + actual supporting edges."""
    if not 3 <= len(active) <= 7 or active.shape != (len(active), 7):
        raise _Unresolved("vertex_count")
    if not np.isin(active, (0, 1)).all():
        raise _Unresolved("active_flags")
    constraints = (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
        *coefficients,
    )
    lines = tuple(
        (exact.subtract(c[1], c[0]), exact.subtract(c[2], c[0]), c[0]) for c in constraints
    )
    barycentric, masks = [], []
    for row in active:
        ids = tuple(int(i) for i in np.flatnonzero(row))
        bary = None
        for i, j in combinations(ids, 2):
            a, b = lines[i], lines[j]
            det = _cross2(a, b)
            if det:
                u = exact.divide(
                    exact.subtract(exact.multiply(a[1], b[2]), exact.multiply(a[2], b[1])), det
                )
                v = exact.divide(
                    exact.subtract(exact.multiply(a[2], b[0]), exact.multiply(a[0], b[2])), det
                )
                bary = (exact.subtract(Fraction(1), exact.add(u, v)), u, v)
                break
        if bary is None:
            raise _Unresolved("active_rank")
        values = tuple(exact.dot(c, bary) for c in constraints)
        if any(value < 0 for value in values) or any(values[i] != 0 for i in ids):
            raise _Unresolved("infeasible_or_false_active")
        if bary in barycentric:
            raise _Unresolved("repeated_vertex")
        barycentric.append(bary)
        masks.append(tuple(i for i, value in enumerate(values) if value == 0))
    for index, a in enumerate(barycentric):
        nxt = (index + 1) % len(barycentric)
        b = barycentric[nxt]
        edge = exact.difference(b[1:], a[1:])
        turns = tuple(_cross2(edge, exact.difference(p[1:], a[1:])) for p in barycentric)
        if any(t < 0 for t in turns) or not any(t > 0 for t in turns):
            raise _Unresolved("nonconvex_or_zero_area")
        # A constant-zero coplanar tetra face is NOT a boundary certificate.
        if not any(
            i in masks[nxt] and (lines[i][0] != 0 or lines[i][1] != 0) for i in masks[index]
        ):
            raise _Unresolved("edge_has_no_support_line")
    points = tuple(
        tuple(
            exact.total(exact.multiply(bary[j], face.points[j][axis]) for j in range(3))
            for axis in range(3)
        )
        for bary in barycentric
    )
    result = exact._canonical_polygon(points)
    if not result:
        raise _Unresolved("zero_area")
    return result


class TyxAreaPressureIntersector:
    """Caller-owned bounded workspace; full exact output, mixed computation."""

    def __init__(self, *, mode="interpret", capacity=MAX_CANDIDATE_PAIRS, fallback="exact_host"):
        if mode not in ("interpret", "cuda"):
            raise ValueError("intersector mode must be interpret or cuda")
        if type(capacity) is not int or not 1 <= capacity <= MAX_CANDIDATE_PAIRS:
            raise ValueError("intersector capacity must be in 1..2048")
        if fallback not in ("exact_host", "reject"):
            raise ValueError("fallback must be exact_host or reject")
        import tyx as tx

        from ._tyx_area_pressure_clip_kernels import clip_topology

        self.mode, self.capacity, self.fallback = mode, capacity, fallback
        self._tx, self._kernel = tx, clip_topology
        self._lock, self._resources, self._buffers = RLock(), DeviceResources(), {}
        self._last_result, self._last_routes = None, ()
        self._stats = dict.fromkeys(
            (
                "attempts",
                "launches",
                "device_polygons",
                "fallback_pairs",
                "uploaded_bytes",
                "downloaded_bytes",
                "live_bytes",
                "peak_bytes",
            ),
            0,
        )
        self._stats.update(
            dict.fromkeys(
                (
                    "prepare_seconds",
                    "device_seconds",
                    "refine_seconds",
                    "fallback_seconds",
                    "total_seconds",
                ),
                0.0,
            )
        )
        layouts = {
            "coefficients": ((capacity, 4, 3), np.float32),
            "work": ((capacity, 2, 8, 7), np.float32),
            "tight": ((capacity, 2, 8, 7), np.int32),
            "meta": ((capacity, 2), np.int32),
            "active": ((capacity, 8, 7), np.int32),
        }
        try:
            self._runtime = tx.Runtime(backend="cuda" if mode == "cuda" else "cpu")
            self._resources.set_runtime(self._runtime)
            with tx.use_runtime(self._runtime):
                for name, (shape, dtype) in layouts.items():
                    buffer = (
                        tx.cuda_empty(shape, dtype) if mode == "cuda" else np.empty(shape, dtype)
                    )
                    if mode == "cuda":
                        self._resources.add_buffer(buffer)
                    self._buffers[name] = buffer
                    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
                    self._stats["live_bytes"] += size
                    self._stats["peak_bytes"] += size
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "intersector initialization and cleanup failed", [error, cleanup]
                ) from None
            raise

    @property
    def stats(self):
        with self._lock:
            return MappingProxyType(dict(self._stats, mode=self.mode, closed=self.closed))

    @property
    def closed(self):
        return self._resources.closed

    @property
    def last_result(self):
        return self._last_result

    @property
    def last_routes(self):
        return self._last_routes

    def _device_clip(self, coefficients, count):
        tx, buffers = self._tx, self._buffers
        start = perf_counter()
        try:
            with tx.use_runtime(self._runtime):
                if self.mode == "cuda":
                    buffers["coefficients"].copy_from(coefficients)
                    self._stats["uploaded_bytes"] += coefficients.nbytes
                else:
                    np.copyto(buffers["coefficients"], coefficients)
                self._stats["launches"] += 1
                self._kernel(
                    count,
                    *(
                        buffers[name]
                        for name in (
                            "coefficients",
                            "work",
                            "tight",
                            "meta",
                            "active",
                        )
                    ),
                    mode=self.mode,
                )
                self._runtime.sync()
                if self.mode == "cuda":
                    meta, active = buffers["meta"].to_numpy(), buffers["active"].to_numpy()
                    self._stats["downloaded_bytes"] += meta.nbytes + active.nbytes
                else:
                    meta, active = buffers["meta"].copy(), buffers["active"].copy()
                return meta, active
        finally:
            self._stats["device_seconds"] += perf_counter() - start

    def intersect(self, pressure_mesh, rigid_mesh, *, identity, stamp):
        with self._lock:
            if self._resources.state != "OPEN":
                raise RuntimeError("intersector is closed or cleanup is unresolved")
            start = perf_counter()
            self._stats["attempts"] += 1
            try:
                return self._intersect(pressure_mesh, rigid_mesh, identity, stamp)
            finally:
                self._stats["total_seconds"] += perf_counter() - start

    def _intersect(self, pressure_mesh, rigid_mesh, identity, stamp):
        start = perf_counter()
        identity, stamp = _identity(identity), _stamp(stamp)
        pressure, rigid = _meshes(pressure_mesh, rigid_mesh)
        count = len(pressure.cells) * len(rigid.faces)
        if count > self.capacity:
            raise exact.AreaPressureIntersectionError(
                "resource_limit", "device candidate capacity exceeded"
            )
        rows, coefficients = [], np.zeros((self.capacity, 4, 3), np.float32)
        for ci in sorted(range(len(pressure.cells)), key=lambda i: pressure.cell_ids[i]):
            for fi in sorted(range(len(rigid.faces)), key=lambda i: rigid.face_ids[i]):
                cell, face = pressure._exact_cells[ci], rigid._exact_faces[fi]
                weights = tuple(cell.weights(p) for p in face.points)
                original = tuple(tuple(w[plane] for w in weights) for plane in range(4))
                index = len(rows)
                for plane, values in enumerate(original):
                    scale = max(abs(value) for value in values)
                    if scale:
                        coefficients[index, plane] = tuple(
                            float(exact.divide(v, scale)) for v in values
                        )
                rows.append((ci, fi, cell, face, original))
        self._stats["prepare_seconds"] += perf_counter() - start
        meta, active = self._device_clip(coefficients, count)
        if (
            meta.dtype != np.int32
            or meta.shape != (self.capacity, 2)
            or active.dtype != np.int32
            or active.shape != (self.capacity, 8, 7)
        ):
            raise RuntimeError("invalid device clipping readback layout")
        triangles, routes = [], []
        polygons = duplicates = 0
        for index, (ci, fi, cell, face, original) in enumerate(rows):
            status, size = map(int, meta[index])
            if status not in (1, 2, 3, 4, 5):
                raise RuntimeError("invalid device clipping status")
            if status == 4:
                raise exact.AreaPressureIntersectionError(
                    "resource_limit", "device polygon capacity exceeded"
                )
            if not 0 <= size <= 8 or (status == 1 and not 3 <= size <= 7):
                raise RuntimeError("invalid device clipping vertex count")
            reason = {2: "empty_or_low_dimension", 3: "ambiguous_f32", 5: "nonfinite_f32"}.get(
                status
            )
            if status == 1:
                start = perf_counter()
                try:
                    polygon = _refine_polygon(original, face, active[index, :size])
                except _Unresolved as error:
                    reason = str(error)
                finally:
                    self._stats["refine_seconds"] += perf_counter() - start
            if reason is not None:
                if self.fallback == "reject":
                    raise exact.AreaPressureIntersectionError("device_unresolved", reason)
                start = perf_counter()
                try:
                    polygon = exact.clip_triangle(cell, face)
                finally:
                    self._stats["fallback_seconds"] += perf_counter() - start
                routes.append("exact_host:" + reason)
                self._stats["fallback_pairs"] += 1
            else:
                routes.append("device")
                self._stats["device_polygons"] += 1
            if not polygon:
                continue
            if not _owns_polygon(pressure, ci, face, polygon):
                duplicates += 1
                continue
            if len(triangles) + len(polygon) - 2 > MAX_TRIANGLES:
                raise exact.AreaPressureIntersectionError(
                    "resource_limit", "intersection exceeds I0 triangle capacity"
                )
            polygons += 1
            for fan in range(len(polygon) - 2):
                triangles.append(
                    _publish_triangle(cell, polygon, pressure.cell_ids[ci], rigid.face_ids[fi], fan)
                )
        result = AreaPressureIntersection(
            AreaPressureSurface(identity, stamp, tuple(triangles)),
            pressure.pressure_domain,
            count,
            polygons,
            duplicates,
        )
        publication = tuple(routes)
        self._last_result, self._last_routes = result, publication
        return result

    def close(self):
        with self._lock:
            self._resources.close()
            self._buffers.clear()
            self._stats["live_bytes"] = 0

    def __enter__(self):
        if self._resources.state != "OPEN":
            raise RuntimeError("intersector is closed or cleanup is unresolved")
        return self

    def __exit__(self, *_):
        self.close()
