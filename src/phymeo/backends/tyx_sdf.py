"""Explicit TYX batch sampler with one persistent SDF node table per workspace."""

from contextlib import ExitStack
from numbers import Integral
from threading import RLock

import numpy as np

from ..geometry.sdf_grid import _MAX_BATCH, SignedDistanceGrid, _points, _world


def _fp32(values, name):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        raw = np.asarray(values, dtype=np.float32)
    tiny = np.finfo(np.float32).tiny
    if not np.isfinite(raw).all() or np.any((values != 0) & (np.abs(raw) < tiny)):
        raise ValueError(f"{name} outside normal FP32 range")
    return raw


class TyxSdfSampler:
    """HOST locates cells in the asset and rotates gradients; TYX reads and interpolates eight nodes."""

    def __init__(self, grid, mode="cuda", *, capacity=4096, cache_root=None):
        if not isinstance(grid, SignedDistanceGrid):
            raise TypeError("expected SignedDistanceGrid")
        if mode not in ("cuda", "interpret"):
            raise ValueError("SDF sampler mode must be cuda or interpret")
        if (
            isinstance(capacity, (bool, np.bool_))
            or not isinstance(capacity, Integral)
            or not 1 <= capacity <= _MAX_BATCH
        ):
            raise ValueError("invalid SDF query capacity")
        self._grid = grid
        self._capacity = int(capacity)
        self._mode = mode
        self._lock = RLock()
        self._closed = False
        nodes = grid.values_zyx_m.ravel()
        numeric = _fp32(nodes, "SDF nodes")
        keep = (
            np.ones(len(nodes), np.float32) if grid.band_m is None else np.abs(nodes) < grid.band_m
        )
        packed = np.column_stack((numeric, keep)).astype(np.float32, copy=False)
        with np.errstate(over="ignore", divide="ignore", under="ignore"):
            self._inv = _fp32(1 / np.array(grid.spacing_m), "inverse spacing")
        if np.any(self._inv == 0):
            raise ValueError("inverse spacing outside normal FP32 range")
        self._counts = dict(
            grid_uploads=0,
            grid_uploaded_bytes=0,
            query_uploads=0,
            query_uploaded_bytes=0,
            downloads=0,
            downloaded_bytes=0,
            launches=0,
            completed_launches=0,
            synchronizations=0,
            samples=0,
        )
        self._quantization = float(np.max(np.abs(nodes - numeric.astype(float))))
        peak = float(np.max(np.abs(numeric)))
        f32 = np.finfo(np.float32)
        # Covers fraction rounding and the fixed scalar lerp chain on a
        # successful finite FP32 result, including possible denormal flushing.
        self._value_allowance = float(
            np.nextafter(
                self._quantization + 128 * float(f32.eps) * peak + 64 * float(f32.tiny), np.inf
            )
        )
        import tyx as tx

        from ._tyx_sdf_kernels import sample_sdf_nodes

        self._tx = tx
        self._launch = sample_sdf_nodes
        self._resources = ExitStack()
        self._runtime = tx.Runtime(
            backend="cuda" if mode == "cuda" else "cpu", cache_root=cache_root
        )
        self._resources.callback(self._runtime.close)
        try:
            with tx.use_runtime(self._runtime):
                if mode == "cuda":
                    self._nodes = tx.cuda_array(packed)
                    self._resources.callback(self._nodes.free)
                    self._counts["grid_uploads"] = 1
                    self._counts["grid_uploaded_bytes"] = packed.nbytes
                else:
                    self._nodes = packed
        except BaseException:
            self.close()
            raise

    @property
    def grid(self):
        return self._grid

    @property
    def capacity(self):
        return self._capacity

    @property
    def closed(self):
        return self._closed

    def stats(self):
        with self._lock:
            return dict(
                self._counts,
                backend=self._mode,
                grid_quantization_error_m=self._quantization,
                value_precision_allowance_m=self._value_allowance,
                geometry_error_bound_m=(
                    None
                    if self.grid.geometry_error_bound_m is None
                    else float(
                        np.nextafter(
                            self.grid.geometry_error_bound_m + self._value_allowance, np.inf
                        )
                    )
                ),
            )

    def __enter__(self):
        if self.closed:
            raise RuntimeError("SDF workspace closed")
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        with self._lock:
            if not self.closed:
                self._closed = True
                self._resources.close()

    def _sample(self, points, rotation=None):
        with self._lock:
            if self.closed:
                raise RuntimeError("SDF workspace closed")
            if not len(points):
                return ()
            bases, fraction = self.grid._locate(points)
            generation = self._counts["launches"] + 1
            if generation > (np.iinfo(np.int32).max - 3) // 4:
                self.close()
                raise RuntimeError("SDF launch generation exhausted")
            nx, ny, _ = self.grid.shape_xyz
            fractions = fraction.astype(np.float32)
            try:
                with self._tx.use_runtime(self._runtime), ExitStack() as scope:
                    if self._mode == "cuda":
                        args = []
                        for data in (bases, fractions):
                            arr = self._tx.cuda_array(data)
                            scope.callback(arr.free)
                            self._counts["query_uploads"] += 1
                            self._counts["query_uploaded_bytes"] += data.nbytes
                            args.append(arr)
                        addr, frac = args
                        out = self._tx.cuda_empty((len(points), 4), np.float32)
                        scope.callback(out.free)
                    else:
                        addr, frac = bases.copy(), fractions
                        out = np.empty((len(points), 4), np.float32)
                    self._counts["launches"] += 1
                    self._launch(
                        len(points),
                        nx,
                        nx * ny,
                        generation,
                        *map(float, self._inv),
                        self._nodes,
                        addr,
                        frac,
                        out,
                        mode=self._mode,
                    )
                    if self._mode == "cuda":
                        self._runtime.sync()
                        self._counts["synchronizations"] += 1
                        result = []
                        for arr in (addr, out):
                            host = arr.to_numpy()
                            self._counts["downloads"] += 1
                            self._counts["downloaded_bytes"] += host.nbytes
                            result.append(host)
                        addr, out = result
                    if (
                        addr.shape != (len(points),)
                        or addr.dtype != np.int32
                        or out.shape != (len(points), 4)
                        or out.dtype != np.float32
                    ):
                        raise RuntimeError("incomplete SDF device output")
                    code = -addr.astype(np.int64) - 4 * generation - 1
                    if np.any((code < 0) | (code > 2)) or np.any((bases < 0) != (code == 1)):
                        raise RuntimeError("incomplete SDF device output")
                    samples = self.grid._results(bases, code, out, rotation)
            except BaseException:
                self.close()
                raise
            self._counts["completed_launches"] += 1
            self._counts["samples"] += len(samples)
            return samples

    def sample_local(self, points_m):
        return self._sample(_points(points_m, self.capacity))

    def sample_world(self, points_m, *, translation_m, orientation_xyzw):
        p, r = _world(_points(points_m, self.capacity), translation_m, orientation_xyzw)
        return self._sample(p, r)
