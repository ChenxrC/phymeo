"""TYX buffers for complete per-world pair tables and staged sphere states."""

from contextlib import ExitStack

import numpy as np


class SphereIslandDevice:
    def __init__(self, mode, cache, props, bodies, offsets, sides, restitution):
        import tyx as tx
        from . import _tyx_sphere_island_kernels as kernels

        self.tx, self.k, self.mode = tx, kernels, mode
        self.n, self.w, self.p = len(props), len(restitution), int(offsets[-1])
        self.resources = ExitStack()
        self.runtime = tx.Runtime(backend="cuda" if mode == "cuda" else "cpu", cache_root=cache)
        self.resources.callback(self.runtime.close)
        self.counts = dict(
            completed_kernels=0,
            synchronizations=0,
            device_allocations=0,
            static_uploaded_bytes=0,
            uploaded_bytes=0,
            downloaded_bytes=0,
        )
        try:
            with tx.use_runtime(self.runtime):
                self.props = self.buffer(props.shape, props.dtype, props)
                self.bodies = self.buffer(bodies.shape, bodies.dtype, bodies)
                self.offsets = self.buffer(offsets.shape, offsets.dtype, offsets)
                self.sides = self.buffer(sides.shape, sides.dtype, sides)
                self.restitution = self.buffer(restitution.shape, restitution.dtype, restitution)
                self.original = self.buffer((self.n, 6), np.float32)
                self.forces = self.buffer((self.n, 3), np.float32)
                self.capacity = self.buffer((self.w,), np.int32)
                self.staged = self.buffer((self.n, 6), np.float32)
                self.rows = self.buffer((max(self.p, 1), 9), np.float32)
                self.issues = self.buffer((max(self.p, 1),), np.int32)
                self.status = self.buffer((self.w, 2), np.int32)
                self.residual = self.buffer((self.w, 2), np.float32)
        except BaseException:
            self.close()
            raise

    def buffer(self, shape, dtype, values=None):
        if self.mode == "interpret":
            return np.empty(shape, dtype) if values is None else values.copy()
        result = self.tx.cuda_empty(shape, dtype) if values is None else self.tx.cuda_array(values)
        self.resources.callback(result.free)
        self.counts["device_allocations"] += 1
        if values is not None:
            self.counts["static_uploaded_bytes"] += values.nbytes
        return result

    def _put(self, buffer, values):
        if self.mode == "cuda":
            buffer.copy_from(values)
            self.counts["uploaded_bytes"] += values.nbytes
        else:
            np.copyto(buffer, values)

    def _get(self, buffer):
        if self.mode == "interpret":
            return buffer.copy()
        result = buffer.to_numpy()
        self.counts["downloaded_bytes"] += result.nbytes
        return result

    def step(self, initial, forces, capacity, h, iterations, ptol, vtol):
        with self.tx.use_runtime(self.runtime), np.errstate(all="ignore"):
            self._put(self.original, initial)
            self._put(self.forces, forces)
            self._put(self.capacity, capacity)
            self.k.free_motion(
                self.n, h, self.props, self.original, self.forces, self.staged, mode=self.mode
            )
            self.counts["completed_kernels"] += 1
            if self.p:
                self.k.candidate_rows(
                    self.p,
                    self.sides,
                    self.props,
                    self.original,
                    self.staged,
                    self.rows,
                    self.issues,
                    mode=self.mode,
                )
                self.counts["completed_kernels"] += 1
            self.k.solve_worlds(
                self.w,
                iterations,
                ptol,
                vtol,
                self.bodies,
                self.offsets,
                self.sides,
                self.props,
                self.restitution,
                self.capacity,
                self.original,
                self.staged,
                self.rows,
                self.issues,
                self.status,
                self.residual,
                mode=self.mode,
            )
            self.counts["completed_kernels"] += 1
            self.runtime.sync()
            self.counts["synchronizations"] += 1
            return (
                self._get(self.staged),
                self._get(self.rows)[: self.p],
                self._get(self.status),
                self._get(self.residual),
                self._get(self.issues)[: self.p],
            )

    def close(self):
        self.resources.close()
