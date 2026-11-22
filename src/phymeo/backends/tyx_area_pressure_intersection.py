"""Persistent TYX clipping proposals with exact, complete boundary refinement.

HOST validation, coefficient construction and publication remain explicit. Empty
or unproven output uses the chosen exact-host/reject policy, never a silent miss.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from threading import RLock
from time import perf_counter
from types import MappingProxyType

import numpy as np

from ..geometry import _area_pressure_clipping as exact
from ..geometry.area_pressure import (
    MAX_TRIANGLES,
    AreaPressureIdentity,
    AreaPressureStamp,
    AreaPressureSurface,
    _identity,
    _origin_world,
    _stamp,
)
from ..geometry.area_pressure_intersection import (
    MAX_CANDIDATE_PAIRS,
    AreaPressureIntersection,
    RigidPressureMesh,
    TetraPressureMesh,
    _AreaPressureGeometryInput,
    _build_geometry_inputs,
    _meshes,
    _owns_polygon,
    _publish_triangle,
)
from .mujoco_tyx._resources import DeviceResources


@dataclass(frozen=True)
class AreaPressureIntersectionRequest:
    """A query description, checked together with every other batch request."""

    pressure_mesh: TetraPressureMesh
    rigid_mesh: RigidPressureMesh
    identity: AreaPressureIdentity
    stamp: AreaPressureStamp
    origin_world: np.ndarray | None = field(default=None, kw_only=True)

    def __post_init__(self):
        object.__setattr__(self, "origin_world", _origin_world(self.origin_world))


@dataclass
class _Pair:
    ci: int
    fi: int
    cell: object
    face: object
    coefficients: tuple = ()
    polygon: tuple = ()
    route: str = ""


def _strict_aabb_separation(cell, face):
    """Stored coordinates are exact FP64 values; strict bounds cannot hide touch."""
    return any(
        max(p[axis] for p in cell.points) < min(p[axis] for p in face.points)
        or max(p[axis] for p in face.points) < min(p[axis] for p in cell.points)
        for axis in range(3)
    )


def _checked_mesh_requests(requests):
    for request in requests:
        identity, stamp = _identity(request.identity), _stamp(request.stamp)
        pressure, rigid = _meshes(request.pressure_mesh, request.rigid_mesh)
        yield pressure, rigid, identity, stamp


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
        self._publication = ((), ())
        self._stats = dict.fromkeys(
            (
                "attempts",
                "batch_calls",
                "candidate_pairs",
                "culled_pairs",
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
        results = self._publication[0]
        return results[-1] if results else None

    @property
    def last_routes(self):
        routes = self._publication[1]
        return routes[-1] if routes else ()

    @property
    def last_results(self):
        return self._publication[0]

    @property
    def last_batch_routes(self):
        return self._publication[1]

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

    def intersect(self, pressure_mesh, rigid_mesh, *, identity, stamp, origin_world=None):
        return self.intersect_many(
            (
                AreaPressureIntersectionRequest(
                    pressure_mesh, rigid_mesh, identity, stamp, origin_world=origin_world
                ),
            )
        )[0]

    def intersect_many(self, requests):
        """Validate all queries, batch active candidates, then publish all results.

        Capacity bounds each query and each device launch. The complete batch
        has a separate 2048-pair bound; empty witnesses do not bypass either cap.
        Counters describe attempted work, including failed work. Published query
        results change only after all chunks, exact checks and outputs succeed.
        """
        return self._run_many(requests, AreaPressureIntersectionRequest, _checked_mesh_requests)

    def _intersect_geometry_inputs(self, requests):
        # Raw declarations still undergo every current-geometry check. This is
        # not a "trusted mesh" switch and cannot admit a caller's private cache.
        return self._run_many(requests, _AreaPressureGeometryInput, _build_geometry_inputs)

    def _run_many(self, requests, request_type, build):
        with self._lock:
            if self._resources.state != "OPEN":
                raise RuntimeError("intersector is closed or cleanup is unresolved")
            if type(requests) is not tuple:
                raise ValueError("requests must be a tuple")
            if len(requests) > MAX_CANDIDATE_PAIRS:
                raise exact.AreaPressureIntersectionError("resource_limit", "too many queries")
            if any(type(request) is not request_type for request in requests):
                raise ValueError(f"expected {request_type.__name__}")
            start = perf_counter()
            self._stats["batch_calls"] += 1
            self._stats["attempts"] += len(requests)
            try:
                prepared, active_rows = self._prepare_many(requests, build)
                for offset in range(0, len(active_rows), self.capacity):
                    rows = active_rows[offset : offset + self.capacity]
                    start_pack = perf_counter()
                    try:
                        coefficients = np.zeros((self.capacity, 4, 3), np.float32)
                        for index, row in enumerate(rows):
                            for plane, values in enumerate(row.coefficients):
                                scale = max(abs(value) for value in values)
                                if scale:
                                    coefficients[index, plane] = tuple(
                                        float(exact.divide(v, scale)) for v in values
                                    )
                    finally:
                        self._stats["prepare_seconds"] += perf_counter() - start_pack
                    meta, active = self._device_clip(coefficients, len(rows))
                    if (
                        meta.dtype != np.int32
                        or meta.shape != (self.capacity, 2)
                        or active.dtype != np.int32
                        or active.shape != (self.capacity, 8, 7)
                    ):
                        raise RuntimeError("invalid device clipping readback layout")
                    for index, row in enumerate(rows):
                        row.polygon, row.route = self._resolve(row, meta[index], active[index])
                results = tuple(
                    self._publish_query(*query, origin_world=request.origin_world)
                    for query, request in zip(prepared, requests, strict=True)
                )
                routes = tuple(tuple(row.route for row in query[-1]) for query in prepared)
                self._publication = (results, routes)
                return results
            finally:
                self._stats["total_seconds"] += perf_counter() - start

    def _prepare_many(self, requests, build):
        start = perf_counter()
        try:
            checked, count = [], 0
            # Mesh validation is bounded and complete before the first device launch.
            for pressure, rigid, identity, stamp in build(requests):
                size = len(pressure.cells) * len(rigid.faces)
                count += size
                if size > self.capacity or count > MAX_CANDIDATE_PAIRS:
                    raise exact.AreaPressureIntersectionError(
                        "resource_limit", "device candidate capacity exceeded"
                    )
                checked.append((pressure, rigid, identity, stamp))
            self._stats["candidate_pairs"] += count
            prepared, active_rows = [], []
            for pressure, rigid, identity, stamp in checked:
                rows = []
                for ci in sorted(range(len(pressure.cells)), key=lambda i: pressure.cell_ids[i]):
                    for fi in sorted(range(len(rigid.faces)), key=lambda i: rigid.face_ids[i]):
                        cell, face = pressure._exact_cells[ci], rigid._exact_faces[fi]
                        row = _Pair(ci, fi, cell, face)
                        if _strict_aabb_separation(cell, face):
                            row.route = "exact_empty:aabb"
                        else:
                            weights = tuple(cell.weights(p) for p in face.points)
                            row.coefficients = tuple(
                                tuple(w[plane] for w in weights) for plane in range(4)
                            )
                            if any(all(v < 0 for v in values) for values in row.coefficients):
                                row.route = "exact_empty:plane"
                        if row.route:
                            self._stats["culled_pairs"] += 1
                        else:
                            active_rows.append(row)
                        rows.append(row)
                prepared.append((pressure, rigid, identity, stamp, tuple(rows)))
            return tuple(prepared), active_rows
        finally:
            self._stats["prepare_seconds"] += perf_counter() - start

    def _resolve(self, row, metadata, active):
        status, size = map(int, metadata)
        if status not in (1, 2, 3, 4, 5):
            raise RuntimeError("invalid device clipping status")
        if status == 4:
            raise exact.AreaPressureIntersectionError(
                "resource_limit", "device polygon capacity exceeded"
            )
        if not 0 <= size <= 8 or (status == 1 and not 3 <= size <= 7):
            raise RuntimeError("invalid device clipping vertex count")
        reason = {2: "empty_or_low_dimension", 3: "ambiguous_f32", 5: "nonfinite_f32"}.get(status)
        if status == 1:
            start = perf_counter()
            try:
                polygon = _refine_polygon(row.coefficients, row.face, active[:size])
            except _Unresolved as error:
                reason = str(error)
            finally:
                self._stats["refine_seconds"] += perf_counter() - start
        if reason is not None:
            if self.fallback == "reject":
                raise exact.AreaPressureIntersectionError("device_unresolved", reason)
            start = perf_counter()
            try:
                polygon = exact.clip_triangle(row.cell, row.face)
            finally:
                self._stats["fallback_seconds"] += perf_counter() - start
            self._stats["fallback_pairs"] += 1
            return polygon, "exact_host:" + reason
        self._stats["device_polygons"] += 1
        return polygon, "device"

    def _publish_query(self, pressure, rigid, identity, stamp, rows, *, origin_world=None):
        triangles = []
        polygons = duplicates = 0
        for row in rows:
            polygon, ci, fi, cell, face = row.polygon, row.ci, row.fi, row.cell, row.face
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
                    _publish_triangle(
                        cell,
                        polygon,
                        pressure.cell_ids[ci],
                        rigid.face_ids[fi],
                        fan,
                        origin_world=origin_world,
                    )
                )
        return AreaPressureIntersection(
            AreaPressureSurface(identity, stamp, tuple(triangles)),
            pressure.pressure_domain,
            len(rows),
            polygons,
            duplicates,
        )

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
