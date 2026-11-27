"""Experimental geometry-to-elliptic-row boundary, not a full dynamics backend.

The original M16 contact domain has zero margin/gap, implicit condim3 elliptic
pairs, and MULTICCD disabled. Native box-box multiple contacts are retained.
Body COM/Jacobians/velocity are supplied by the caller at the query pose epoch;
this module does not yet provide full-model kinematics or integration.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType

import numpy as np
import tyx as tx

from . import _mesh_contact_kernels as k
from ._mesh_catalog import MeshQueryCatalog
from ._normal_extended_geometry import extended_contacts


@dataclass(frozen=True)
class MeshContactResult:
    workspace: object
    generation: int

    def read(self):
        return self.workspace._read(self)

    def device_view(self):
        return self.workspace._device_view(self)


class MeshContactWorkspace:
    """Bounded reusable device manifold/constraint buffers with scoped loans.

    Query and motion must describe one coherent pose. No CPU narrow phase or
    state readback in assemble_device; only world status is read to admit a
    result loan. Results expire on the next attempt, even if that attempt fails.
    Arrays borrowed from a result must not be retained/mutated beyond its scope.
    """

    OUTPUTS = (
        "counts",
        "pairs",
        "geometry",
        "point_low",
        "frame",
        "jac",
        "aref",
        "D",
        "R",
        "friction",
        "mu",
        "eligible",
        "law",
        "status",
    )
    MOTION = ("com", "body_jac", "qvel")

    def __init__(
        self,
        mj,
        model,
        catalog,
        *,
        worlds=1,
        capacity=2048,
        mode="cuda",
        cache_root=None,
        memory_limit_bytes=256 * 1024**2,
    ):
        if not isinstance(catalog, MeshQueryCatalog):
            raise TypeError("compiled mesh catalog required")
        if not np.array_equal(catalog.geom_ids, np.arange(model.ngeom)):
            raise ValueError("full ordered original geometry catalog required")
        if not np.array_equal(catalog.body_ids, model.geom_bodyid):
            raise ValueError("catalog/body identity mismatch")
        if (
            model.npair
            or model.nflex
            or int(model.opt.enableflags)
            or int(model.opt.disableflags)
            or int(model.opt.cone) != 1
            or np.any(model.geom_condim != 3)
            or np.any(model.geom_margin != 0)
            or np.any(model.geom_gap != 0)
        ):
            raise ValueError(
                "requires implicit zero-margin/gap condim3 elliptic contacts and default flags"
            )
        if not np.isin(model.geom_type, [6, 7]).all():
            raise ValueError("only original box/mesh geometry supported")
        for value, high in ((worlds, 256), (capacity, 8192)):
            if type(value) is not int or not 1 <= value <= high:
                raise ValueError("bounded positive integer dimensions required")
        if mode not in ("cuda", "interpret"):
            raise ValueError("explicit cuda or interpret mode required")
        if not 1 <= model.nv <= 35:
            raise ValueError("1..35 original velocity coordinates required")
        self.catalog, self.worlds, self.capacity, self.mode = catalog, worlds, capacity, mode
        self.nv, self.nbody = int(model.nv), int(model.nbody)
        self.h, self.impratio = float(model.opt.timestep), float(model.opt.impratio)
        if not np.isfinite([self.h, self.impratio]).all() or min(self.h, self.impratio) <= 0:
            raise ValueError("finite positive timestep/impratio required")
        self.constants = tuple(map(float, (mj.mjMINVAL, mj.mjMINIMP, mj.mjMAXIMP, mj.mjMINMU)))
        kinds = np.stack((model.geom_type, model.geom_priority, model.geom_bodyid), axis=1).astype(
            np.int32
        )
        params = np.column_stack(
            (
                model.geom_solmix,
                model.geom_solref,
                model.geom_solimp,
                model.geom_friction,
                model.body_invweight0[model.geom_bodyid, 0],
            )
        ).astype(np.float32)
        if not np.isfinite(params).all() or np.any(params[:, 0] < 0) or np.any(params[:, 11] < 0):
            raise ValueError("invalid static contact parameters")
        box_map = np.flatnonzero(model.geom_type == 6).astype(np.int32)
        box_index = {int(g): INT for INT, g in enumerate(box_map)}
        box_pairs = [
            (int(a), int(b)) for a, b in catalog.pairs if a in box_index and b in box_index
        ]
        self.nb, self.nbp, self.ns = len(box_map), len(box_pairs), 24 * len(box_pairs)
        # A dummy allocation avoids native zero-byte ownership quirks; loop bounds stay exact.
        nb, ns, npair = max(1, self.nb), max(1, self.ns), max(1, self.nbp)
        box_ids = np.column_stack((np.arange(nb), np.full(nb, 6))).astype(np.int32)
        box_data = np.zeros((nb, 15), np.float32)
        box_data[:, 3:12] = np.eye(3).reshape(9)
        box_data[: self.nb, 12:15] = model.geom_size[box_map]
        bp = np.zeros((npair, 4), np.int32)
        slot_pairs = np.zeros((ns, 2), np.int32)
        for INT, (a, b) in enumerate(box_pairs):
            bp[INT] = box_index[a], box_index[b], 24 * INT, 24
            slot_pairs[24 * INT : 24 * (INT + 1)] = a, b
        box_law = np.zeros((npair, 16), np.float32)
        box_law[:, 10] = mj.mjMINVAL
        static = dict(
            kinds=kinds,
            params=params,
            box_map=box_map if self.nb else np.zeros(1, np.int32),
            box_ids=box_ids,
            box_data=box_data,
            box_pairs=bp,
            box_law=box_law,
            box_slot_pairs=slot_pairs,
        )
        w, c, n = worlds, worlds * capacity, self.nv
        F, INT = np.float32, np.int32
        self.layouts = {name: (v.shape, v.dtype) for name, v in static.items()}
        self.layouts.update(
            {
                "counts": ((w,), INT),
                "pairs": ((c, 2), INT),
                "geometry": ((c, 7), F),
                "point_low": ((c, 3), F),
                "frame": ((c, 3, 3), F),
                "jac": ((c, 3, n), F),
                "aref": ((c, 3), F),
                "D": ((c, 3), F),
                "R": ((c, 3), F),
                "friction": ((c, 2), F),
                "mu": ((c,), F),
                "eligible": ((c,), INT),
                "law": ((c, 10), F),
                "status": ((w,), INT),
                "com": ((w, self.nbody, 3), F),
                "body_jac": ((w, self.nbody, 6, n), F),
                "qvel": ((w, n), F),
                "box_motion": ((w, nb, 1, 3), F),
                "box_rotation": ((w, nb, 3, 3), F),
                "box_world": ((w, nb, 12), F),
                "box_raw": ((w, ns, 11), F),
                "box_scratch": ((w, 64, 3), F),
            }
        )
        self.owned_bytes = sum(
            int(np.prod(shape)) * np.dtype(dtype).itemsize for shape, dtype in self.layouts.values()
        )
        if self.owned_bytes > memory_limit_bytes:
            raise MemoryError("contact workspace exceeds memory budget")
        self._lock, self._generation, self._borrowed, self.closed = RLock(), 0, 0, False
        self._buffers = {}
        self.runtime = tx.Runtime(
            backend="cuda" if mode == "cuda" else "cpu", cache_root=cache_root
        )
        self.extended = extended_contacts
        try:
            with self.runtime.activate():
                for name, (shape, dtype) in self.layouts.items():
                    v = static[name].copy() if name in static else np.zeros(shape, dtype)
                    self._buffers[name] = tx.cuda.array(v) if mode == "cuda" else v
        except BaseException:
            self.close()
            raise

    def _check(self, result=None):
        if self.closed:
            raise RuntimeError("contact workspace closed")
        if result is not None and (
            result.workspace is not self or result.generation != self._generation
        ):
            raise RuntimeError("contact result expired")

    def _idle(self):
        if self._borrowed:
            raise RuntimeError("contact output is borrowed")

    def assemble(self, query, motion):
        """Explicit host diagnostic convenience, never a hot-path fallback."""
        with self._lock:
            self._check()
            self._idle()
            self._generation += 1
            if set(motion) != set(self.MOTION):
                raise ValueError("exact motion schema required")
            for name, v in motion.items():
                shape, dtype = self.layouts[name]
                if not isinstance(v, np.ndarray) or v.shape != shape or v.dtype != dtype:
                    raise ValueError(f"{name}: shape/dtype mismatch")
                if self.mode == "cuda":
                    self._buffers[name].copy_from(v)
                else:
                    self._buffers[name][...] = v
            return self.assemble_device(query, {name: self._buffers[name] for name in self.MOTION})

    def assemble_device(self, query, motion):
        with self._lock:
            self._check()
            self._idle()
            self._generation += 1
            if (
                query.workspace.catalog is not self.catalog
                or query.workspace.mode != self.mode
                or query.workspace.worlds != self.worlds
            ):
                raise ValueError("same catalog/worlds/device mode required")
            if set(motion) != set(self.MOTION):
                raise ValueError("exact motion schema required")
            for name, v in motion.items():
                shape, dtype = self.layouts[name]
                if self.mode == "cuda" and (
                    not isinstance(v, tx.cuda.Array) or v.device_ordinal != 0
                ):
                    raise ValueError("same-device TYX CUDA arrays required")
                if tuple(v.shape) != shape or np.dtype(v.dtype) != np.dtype(dtype):
                    raise ValueError(f"{name}: shape/dtype mismatch")
            b = {**self._buffers, **motion}

            def args(*names):
                return tuple(b[name] for name in names)

            with query.device_view() as q, self.runtime.activate():
                k.prepare_boxes(
                    self.worlds,
                    self.nb,
                    self.capacity,
                    b["box_map"],
                    q["positions"],
                    q["rotations"],
                    *args("box_motion", "box_rotation", "counts", "eligible", "status"),
                    mode=self.mode,
                )
                if self.nbp:
                    self.extended(
                        self.worlds,
                        self.nb,
                        self.nbp,
                        self.ns,
                        *args(
                            "box_ids",
                            "box_data",
                            "box_pairs",
                            "box_law",
                            "box_motion",
                            "box_rotation",
                            "box_world",
                            "box_raw",
                            "box_scratch",
                            "status",
                        ),
                        mode=self.mode,
                    )
                k.compact_contacts(
                    self.worlds,
                    self.catalog.ngeom,
                    query.workspace.capacity,
                    self.capacity,
                    self.ns,
                    b["kinds"],
                    q["counts"],
                    q["pairs"],
                    q["gap"],
                    q["normal"],
                    q["surface"],
                    q["surface_low"],
                    *args(
                        "box_slot_pairs",
                        "box_raw",
                        "pairs",
                        "geometry",
                        "point_low",
                        "counts",
                        "status",
                    ),
                    mode=self.mode,
                )
                k.check_motion(
                    self.worlds,
                    self.nbody,
                    self.nv,
                    *args("com", "body_jac", "qvel", "status"),
                    mode=self.mode,
                )
                k.assemble_contacts(
                    self.worlds * self.capacity,
                    self.capacity,
                    self.nv,
                    self.h,
                    self.impratio,
                    *self.constants,
                    *args(
                        "counts",
                        "kinds",
                        "params",
                        "pairs",
                        "geometry",
                        "point_low",
                        "com",
                        "body_jac",
                        "qvel",
                        "frame",
                        "jac",
                        "aref",
                        "D",
                        "R",
                        "friction",
                        "mu",
                        "eligible",
                        "law",
                        "status",
                    ),
                    mode=self.mode,
                )
                self.runtime.sync()
            return MeshContactResult(self, self._generation)

    def _download(self, name):
        v = self._buffers[name]
        return v.to_numpy() if self.mode == "cuda" else v.copy()

    def _read(self, result):
        with self._lock:
            self._check(result)
            return {name: self._download(name) for name in self.OUTPUTS}

    @contextmanager
    def _device_view(self, result):
        with self._lock:
            self._check(result)
            self._idle()
            with self.runtime.activate():
                if np.any(self._download("status") != 0):
                    raise ValueError("invalid or overflowing contact batch")
                self._borrowed += 1
                try:
                    yield MappingProxyType({name: self._buffers[name] for name in self.OUTPUTS})
                finally:
                    try:
                        self.runtime.sync()
                    finally:
                        self._borrowed -= 1

    def close(self):
        with self._lock:
            if not self.closed:
                self._idle()
                self.closed = True
                try:
                    self.runtime.close()
                finally:
                    for v in self._buffers.values():
                        if hasattr(v, "free"):
                            v.free()

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *exc):
        self.close()
