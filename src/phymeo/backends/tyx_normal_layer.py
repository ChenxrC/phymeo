"""Paged FP32 evaluation of the calibrated contact law on an explicit tyx runtime."""

from contextlib import ExitStack
from threading import RLock

import numpy as np

from ..core.capabilities import SolverFailure


class TyxNormalLayer:
    """Input rows are (normal[3], relative velocity[3], width, gap, k, fraction, damping).

    Optional final columns supply friction and slip speed. Results contain gap rate,
    scalar normal magnitude and the total world force on side two.
    Material addressing and the two transposed maps remain on the host.
    """

    def __init__(self, mode, *, cache_root=None):
        if mode not in ("interpret", "cuda"):
            raise ValueError("normal layer requires interpret or cuda")
        import tyx as tx

        from ._tyx_normal_layer import normal_layer_rows

        self.mode = mode
        self._tx, self._kernel = tx, normal_layer_rows
        self._lock = RLock()
        self._closed = False
        self._stats = dict(
            completed_launches=0,
            sites=0,
            uploads=0,
            downloads=0,
            uploaded_bytes=0,
            downloaded_bytes=0,
        )
        self._resources = ExitStack()
        self._runtime = tx.Runtime(
            backend="cuda" if mode == "cuda" else "cpu", cache_root=cache_root
        )
        self._resources.callback(self._runtime.close)
        try:
            self._host = np.zeros((128, 13), np.float32)
            with tx.use_runtime(self._runtime):
                self._in = self._buffer(self._host.shape)
                self._out = self._buffer((len(self._host), 5))
        except BaseException:
            self.close()
            raise

    def _buffer(self, shape):
        if self.mode == "interpret":
            return np.empty(shape, np.float32)
        array = self._tx.cuda_empty(shape, np.float32)
        self._resources.callback(array.free)
        return array

    def evaluate(self, rows):
        rows = np.asarray(rows, dtype=float)
        with self._lock:
            if self._closed:
                raise RuntimeError("normal layer runtime is closed")
            if rows.ndim != 2 or rows.shape[1] not in (11, 13) or not np.isfinite(rows).all():
                raise ValueError("invalid normal layer site rows")
            if rows.shape[1] == 11:
                rows = np.column_stack((rows, np.zeros(len(rows)), np.ones(len(rows))))
            if (
                np.any(abs(np.linalg.norm(rows[:, :3], axis=1) - 1) > 2e-6)
                or np.any(rows[:, 6] <= 0)
                or np.any(rows[:, 8] <= 0)
                or np.any((rows[:, 9] < 0) | (rows[:, 9] > 1))
                or np.any(rows[:, 10:12] < 0)
                or np.any(rows[:, 12] <= 0)
            ):
                raise ValueError("invalid site normal or calibration")
            with np.errstate(over="ignore", under="ignore"):
                packed = rows.astype(np.float32)
            if not np.isfinite(packed).all() or np.any(
                (rows != 0) & (np.abs(packed) < np.finfo(np.float32).tiny)
            ):
                raise SolverFailure("contact layer input exceeds normal FP32 range")
            result = np.empty((len(rows), 5), np.float32)
            try:
                with self._tx.use_runtime(self._runtime):
                    for start in range(0, len(rows), len(self._host)):
                        chunk = packed[start : start + len(self._host)]
                        count = len(chunk)
                        self._host[:count] = chunk
                        if self.mode == "cuda":
                            self._in.copy_from(self._host)
                            self._stats["uploads"] += 1
                            self._stats["uploaded_bytes"] += self._host.nbytes
                        else:
                            self._in[:count] = chunk
                        self._kernel(count, self._in, self._out, mode=self.mode)
                        if self.mode == "cuda":
                            self._runtime.sync()
                            data = self._out.to_numpy()
                            self._stats["downloads"] += 1
                            self._stats["downloaded_bytes"] += data.nbytes
                        else:
                            data = self._out
                        result[start : start + count] = data[:count]
                        self._stats["completed_launches"] += 1
                        self._stats["sites"] += count
                if not np.isfinite(result).all() or np.any(result[:, 1] < 0):
                    raise SolverFailure("invalid computed normal layer force")
            except BaseException:
                self.close()
                raise
            return result

    def stats(self):
        with self._lock:
            return dict(self._stats, backend=self.mode)

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._resources.close()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("normal layer runtime is closed")
        return self

    def __exit__(self, *args):
        self.close()
