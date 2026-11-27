"""Opt-in batched mesh geometry queries, independent of the dynamics compiler.

No contact forces, integration, CPU narrow-phase fallback or training admission.
Static support vertices stay resident. Device poses may be supplied directly;
the host convenience entry uploads poses once. Only candidate counts are read
between broad and narrow phase. UNKNOWN is retained, never treated as separated.
RESOLVED is the original kernel outcome, not a certificate of a requested
physical accuracy: callers must inspect the returned lower/upper gap bracket.
This experimental interface is not yet a contact response/learning backend.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType

import numpy as np

from ._mesh_catalog import MeshQueryCatalog

WIRE = {
    "meta": (np.int32, (16,)),
    "gap": (np.float32, (4,)),
    "normal": (np.float32, (3,)),
    "surface": (np.float32, (2, 3)),
    "material_ids": (np.int32, (2, 4)),
    "barycentric": (np.float32, (2, 4)),
    "skin_offset": (np.float32, (2, 3)),
}


def _positive(value, name, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not 1 <= value <= high:
        raise ValueError(f"{name} must be in 1..{high}")
    return int(value)


@dataclass(frozen=True)
class MeshQueryResult:
    workspace: object
    generation: int
    counts: tuple

    def read(self):
        """Explicit detached diagnostic download, not required by device consumers."""
        return self.workspace._read(self)

    def device_view(self):
        """Borrow validated device outputs for a synchronous consumer scope.

        The caller must treat arrays as read-only and not retain them beyond
        the scope. Query/close is prohibited while the scope owns the workspace.
        """
        return self.workspace._device_view(self)


class MeshQueryWorkspace:
    """One owned, serial query workspace with bounded reusable GPU buffers.

    Results expire at the next attempt (including a failed attempt) or close.
    Candidate order is unspecified. Consumers must use original geometry IDs,
    not interpret atomic compaction order as a stable contact solver order.
    """

    def __init__(
        self,
        catalog,
        *,
        worlds=1,
        capacity=4096,
        tile=128,
        mode="cuda",
        cache_root=None,
        memory_limit_bytes=256 * 1024 * 1024,
    ):
        if not isinstance(catalog, MeshQueryCatalog):
            raise TypeError("a compiled MeshQueryCatalog is required")
        if mode not in ("cuda", "interpret"):
            raise ValueError("mode must be cuda or interpret; no automatic fallback")
        self.worlds = w = _positive(worlds, "worlds", 256)
        self.capacity = c = _positive(capacity, "capacity", 65536)
        self.tile = t = _positive(tile, "tile", 4096)
        limit = _positive(memory_limit_bytes, "memory_limit_bytes", 2**63 - 1)
        self.catalog, self.mode = catalog, mode
        self._lock, self._generation, self.closed = RLock(), 0, False
        self._borrowed = 0
        self._buffers, self._runtime = {}, None
        self.stats = dict(launches=0, uploaded_bytes=0, downloaded_bytes=0, queries=0)
        g, v, p = catalog.ngeom, len(catalog.points), w * c
        if w * max(g, v, len(catalog.pairs), c) >= 2**31:
            raise ValueError("flattened layout exceeds int32")
        self._max_vertices = maximum = int(np.max(np.diff(catalog.offsets)))
        if 2 * t * maximum >= 2**31:
            raise ValueError("pair-local support layout exceeds int32")
        static = dict(
            geom_ids=catalog.geom_ids,
            body_ids=catalog.body_ids,
            offsets=catalog.offsets,
            local=catalog.points,
            allowed=catalog.pairs,
            margins=catalog.margins,
            radius=np.zeros(2 * t, np.float32),
            precision=np.zeros(2 * t, np.int32),
            dimensions=np.zeros((2 * t, 3), np.float32),
        )
        layouts = {name: (value.shape, value.dtype) for name, value in static.items()}
        layouts.update(
            {
                "positions": ((w, g, 3), np.float32),
                "rotations": ((w, g, 3, 3), np.float32),
                "points": ((w * v, 3), np.float32),
                "bounds": ((w * g, 6), np.float32),
                "counts": ((w + 1,), np.int32),
                "pairs": ((p, 2), np.int32),
                "packed_pairs": ((p, 2), np.int32),
                "packed_points": ((2 * t * maximum, 3), np.float32),
                "ids": ((2 * t * maximum,), np.int32),
                "tags": ((2 * t, 4), np.int32),
                "frames": ((t, 4), np.float32),
                "shape_rotation": ((2 * t, 3, 3), np.float32),
                "support_points": ((t, 2, 3), np.float32),
                "vertices": ((t, 64, 9), np.float32),
                "vertex_slots": ((t, 64, 2), np.int32),
                "faces": ((t, 128, 3), np.int32),
                "face_geometry": ((t, 128, 4), np.float32),
                "edges": ((t, 384, 2), np.int32),
                "weights": ((t, 4), np.float32),
                "indices": ((t, 4), np.int32),
                "simplex_copy": ((t, 4, 9), np.float32),
                "simplex_slots": ((t, 4, 2), np.int32),
            }
        )
        layouts.update({name: ((p, *tail), dtype) for name, (dtype, tail) in WIRE.items()})
        required = sum(
            max(1, int(np.prod(shape))) * np.dtype(dtype).itemsize
            for shape, dtype in layouts.values()
        )
        if required > limit:
            raise MemoryError(f"mesh query requires {required} bytes, budget {limit}")
        self.stats["owned_bytes"] = required
        import tyx as tx

        from .._tyx_convex_query_kernels import make_convex_query_kernels
        from . import _mesh_feature_refinement
        from . import _mesh_query_kernels as kernels

        self._runtime = tx.Runtime(
            backend="cuda" if mode == "cuda" else "cpu", cache_root=cache_root
        )
        self._geometry, self._convex = kernels, make_convex_query_kernels()
        self._refine = _mesh_feature_refinement.refine
        try:
            with self._runtime.activate():
                for name, (shape, dtype) in layouts.items():
                    # TYX accepts zero-sized arrays, but native kernels must never dereference them.
                    value = (
                        np.array(static[name], copy=True)
                        if name in static
                        else np.zeros(shape, dtype)
                    )
                    self._buffers[name] = tx.cuda.array(value) if mode == "cuda" else value
                    self.stats["uploaded_bytes"] += value.nbytes if mode == "cuda" else 0
        except BaseException:
            self.close()
            raise

    def _launch(self, kernel, *args):
        kernel(*args, mode=self.mode)
        self.stats["launches"] += 1

    def _download(self, name):
        value = self._buffers[name]
        result = value.to_numpy() if self.mode == "cuda" else value.copy()
        self.stats["downloaded_bytes"] += result.nbytes if self.mode == "cuda" else 0
        return result

    def query(self, positions, rotations):
        """Host pose convenience boundary. Never used as hidden CUDA fallback."""
        with self._lock:
            self._ensure_open()
            self._ensure_idle()
            self._generation += 1
            for name, source, shape in (
                ("positions", positions, (self.worlds, self.catalog.ngeom, 3)),
                ("rotations", rotations, (self.worlds, self.catalog.ngeom, 3, 3)),
            ):
                value = np.asarray(source)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"{name} requires finite shape {shape}")
                value = np.ascontiguousarray(value, dtype=np.float32)
                if not np.isfinite(value).all():
                    raise ValueError(f"{name} overflows FP32")
                if self.mode == "cuda":
                    self._buffers[name].copy_from(value)
                    self.stats["uploaded_bytes"] += value.nbytes
                else:
                    self._buffers[name][...] = value
            return self.query_device(self._buffers["positions"], self._buffers["rotations"])

    def query_device(self, positions, rotations):
        """Borrow same-device TYX poses without state readback or copying.

        Caller retains buffers and synchronizes any external producer before
        entry. This method completes device work before returning the loan.
        A small (worlds+1) int32 count/error readback currently schedules GJK.
        """
        with self._lock:
            self._ensure_open()
            self._ensure_idle()
            self._generation += 1
            for value, shape in (
                (positions, (self.worlds, self.catalog.ngeom, 3)),
                (rotations, (self.worlds, self.catalog.ngeom, 3, 3)),
            ):
                if self.mode == "cuda":
                    import tyx as tx

                    if not isinstance(value, tx.cuda.Array) or value.device_ordinal != 0:
                        raise ValueError("poses must be TYX CUDA arrays on device 0")
                elif not isinstance(value, np.ndarray):
                    raise TypeError("interpret poses must be ndarrays")
                if tuple(value.shape) != shape or np.dtype(value.dtype) != np.dtype(np.float32):
                    raise ValueError("device pose shape/dtype mismatch")
            b, cat, k = self._buffers, self.catalog, self._geometry
            with self._runtime.activate():
                self._launch(k.clear_counts, self.worlds, b["counts"])
                self._launch(
                    k.transform_meshes,
                    self.worlds * cat.ngeom,
                    cat.ngeom,
                    len(cat.points),
                    self.worlds,
                    b["offsets"],
                    b["local"],
                    positions,
                    rotations,
                    b["points"],
                    b["bounds"],
                    b["counts"],
                )
                if len(cat.pairs):
                    self._launch(
                        k.compact_mesh_pairs,
                        self.worlds * len(cat.pairs),
                        len(cat.pairs),
                        cat.ngeom,
                        self.capacity,
                        b["allowed"],
                        b["margins"],
                        b["bounds"],
                        b["counts"],
                        b["pairs"],
                    )
                counts = self._download("counts")
                if counts[-1]:
                    raise ValueError("invalid finite rigid geometry pose on device")
                if np.any(counts[:-1] > self.capacity):
                    raise OverflowError(
                        f"candidate capacity {self.capacity} exceeded: {counts[:-1].tolist()}"
                    )
                self._launch(
                    self._convex["clear"], self.worlds * self.capacity, *(b[name] for name in WIRE)
                )
                for world, count in enumerate(counts[:-1]):
                    for first in range(0, int(count), self.tile):
                        active = min(self.tile, int(count) - first)
                        offset = world * self.capacity + first
                        self._launch(
                            k.pack_local_pairs,
                            active,
                            offset,
                            cat.ngeom,
                            len(cat.points),
                            self._max_vertices,
                            b["offsets"],
                            b["points"],
                            b["bounds"],
                            b["pairs"],
                            b["tags"],
                            b["ids"],
                            b["packed_points"],
                            b["packed_pairs"],
                            b["frames"],
                        )
                        self._launch(
                            self._convex["query"],
                            active,
                            offset,
                            self._generation,
                            64,
                            192,
                            64,
                            128,
                            b["tags"],
                            b["ids"],
                            b["radius"],
                            b["precision"],
                            b["packed_points"],
                            b["dimensions"],
                            b["shape_rotation"],
                            b["support_points"],
                            b["packed_pairs"],
                            b["vertices"],
                            b["vertex_slots"],
                            b["faces"],
                            b["face_geometry"],
                            b["edges"],
                            b["weights"],
                            b["indices"],
                            b["simplex_copy"],
                            b["simplex_slots"],
                            *(b[name] for name in WIRE),
                            0,
                            b["pairs"],
                            b["vertices"],
                            b["vertex_slots"],
                        )
                        self._launch(
                            self._refine,
                            active,
                            offset,
                            5e-6,
                            b["tags"],
                            b["packed_points"],
                            b["ids"],
                            b["frames"],
                            b["meta"],
                            b["gap"],
                            b["normal"],
                            b["surface"],
                            b["material_ids"],
                            b["barycentric"],
                            b["skin_offset"],
                        )
                        self._launch(
                            k.unpack_local_pairs,
                            active,
                            offset,
                            b["frames"],
                            b["meta"],
                            b["gap"],
                            b["surface"],
                            b["skin_offset"],
                        )
                self._runtime.sync()
            self.stats["queries"] += 1
            self._poses = (positions, rotations)
            return MeshQueryResult(self, self._generation, tuple(map(int, counts[:-1])))

    def _read(self, result):
        with self._lock:
            self._ensure_open()
            if result.workspace is not self or result.generation != self._generation:
                raise RuntimeError("mesh query result has expired")
            indices = np.concatenate(
                [
                    np.arange(w * self.capacity, w * self.capacity + n)
                    for w, n in enumerate(result.counts)
                ]
            )
            raw = {name: self._download(name)[indices] for name in ("pairs", *WIRE)}
            raw["world"] = raw["pairs"][:, 0] // self.catalog.ngeom
            raw["geom_ids"] = self.catalog.geom_ids[raw["pairs"] % self.catalog.ngeom]
            for value in raw.values():
                value.setflags(write=False)
            return raw

    @contextmanager
    def _device_view(self, result):
        with self._lock:
            self._ensure_open()
            self._ensure_idle()
            if result.workspace is not self or result.generation != self._generation:
                raise RuntimeError("mesh query result has expired")
            b = self._buffers
            with self._runtime.activate():
                self._launch(
                    self._geometry.validate_response_input,
                    self.worlds,
                    self.capacity,
                    5e-6,
                    b["counts"],
                    b["meta"],
                    b["gap"],
                    b["normal"],
                    b["surface"],
                )
                if self._download("counts")[-1]:
                    raise ValueError("uncertain or invalid geometry cannot enter contact response")
                self._borrowed += 1
                try:
                    yield MappingProxyType(
                        {
                            "positions": self._poses[0],
                            "rotations": self._poses[1],
                            **{
                                name: b[name]
                                for name in ("geom_ids", "body_ids", "counts", "pairs", *WIRE)
                            },
                        }
                    )
                finally:
                    try:
                        self._runtime.sync()
                    finally:
                        self._borrowed -= 1

    def _ensure_idle(self):
        if self._borrowed:
            raise RuntimeError("device geometry is borrowed by a consumer")

    def _ensure_open(self):
        if self.closed:
            raise RuntimeError("mesh query workspace is closed")

    def close(self):
        with self._lock:
            if not self.closed:
                self._ensure_idle()
                self.closed = True
                try:
                    if self._runtime is not None:
                        self._runtime.close()
                finally:
                    for value in self._buffers.values():
                        if hasattr(value, "free"):
                            value.free()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *exc):
        self.close()
