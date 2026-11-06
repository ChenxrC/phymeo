"""Explicit selected-world readback resources, owned by one resident world batch."""

from contextlib import ExitStack

import numpy as np


class WorldReadbackPlan:
    """Read an immutable selection from the current accepted state under its batch lock.

    Only gather scratch is persistent. Each public reading owns its HOST arrays or
    immutable checkpoint tuples. Kernel compilation caches remain runtime-owned.
    """

    __slots__ = (
        "_batch", "_keys", "_slots", "_kind", "_width", "_resources", "_closed",
        "_indices", "_rows", "_ticks", "_kernels", "_cuda_bytes", "_host_bytes",
        "_allocations", "_index_bytes", "_reads", "_failed_reads", "_downloaded_bytes",
        "_host_read_bytes",
    )

    def __init__(self, batch, keys, *, kind):
        from .tyx_world_batch import TyxWorldBatch

        if not isinstance(batch, TyxWorldBatch):
            raise TypeError("a readback plan requires a TyxWorldBatch owner")
        with batch._lock:
            batch._check_open()
            if type(kind) is not str or kind not in ("kinematics", "snapshot"):
                raise ValueError("readback kind must be kinematics or snapshot")
            selected = tuple(batch._keys(keys))
            self._batch = batch
            self._keys = selected
            self._slots = tuple(batch._slots[key] for key in selected)
            self._kind = kind
            self._width = 6 if kind == "kinematics" else 12
            self._resources = ExitStack()
            self._closed = False
            self._indices = self._rows = self._ticks = self._kernels = None
            self._cuda_bytes = self._host_bytes = self._allocations = self._index_bytes = 0
            self._reads = self._failed_reads = self._downloaded_bytes = self._host_read_bytes = 0
            try:
                if selected:
                    from ._tyx_world_readback_kernels import make_gather_kernels

                    self._kernels = make_gather_kernels()
                    with batch._device():
                        slots = np.asarray(self._slots, dtype=np.int32)
                        self._indices = self._allocate(slots.shape, slots.dtype, slots)
                        self._rows = self._allocate((len(selected), self._width), np.float32)
                        self._ticks = self._allocate((len(selected),), np.int32)
                batch._readback_plans.add(self)
            except BaseException:
                self.close()
                raise

    @property
    def keys(self):
        return self._keys

    @property
    def kind(self):
        return self._kind

    def _free_device(self, buffer, byte_count):
        buffer.free()
        self._cuda_bytes -= byte_count
        self._batch._counts["device_owned_bytes"] -= byte_count
        self._batch._counts["readback_plan_owned_bytes"] -= byte_count

    def _allocate(self, shape, dtype, initial=None):
        batch = self._batch
        if batch._mode == "interpret":
            result = np.empty(shape, dtype=dtype) if initial is None else initial.copy()
            self._host_bytes += result.nbytes
            return result
        result = batch._tx.cuda_empty(shape, dtype) if initial is None else batch._tx.cuda_array(initial)
        nbytes = int(result.nbytes)
        self._resources.callback(self._free_device, result, nbytes)
        self._allocations += 1
        self._cuda_bytes += nbytes
        batch._counts["device_allocations"] += 1
        batch._counts["device_owned_bytes"] += nbytes
        batch._counts["readback_plan_owned_bytes"] += nbytes
        if initial is not None:
            self._index_bytes += initial.nbytes
            batch._counts["readback_index_uploaded_bytes"] += initial.nbytes
        return result

    def _read_buffer(self, buffer):
        result = self._batch._read(buffer, self._kind)
        self._host_read_bytes += result.nbytes
        if self._batch._mode == "cuda":
            self._downloaded_bytes += result.nbytes
        return result

    def _publish(self, rows, ticks):
        if self._kind == "snapshot":
            return {
                key: self._batch._bindings[slot].checkpoint(
                    rows[i], ticks[i], self._batch._held[slot],
                )
                for i, (key, slot) in enumerate(zip(self._keys, self._slots, strict=True))
            }
        from ._world_kinematics import WorldKinematics

        rows.setflags(write=False)
        ticks.setflags(write=False)
        return WorldKinematics(
            self._keys,
            tuple(self._batch._bindings[slot].width for slot in self._slots),
            ticks,
            rows,
        )

    def read(self):
        batch = self._batch
        with batch._lock:
            if self._closed:
                raise RuntimeError("readback plan is closed")
            batch._check_open()
            try:
                if not self._keys:
                    result = self._publish(
                        np.empty((0, self._width), np.float32), np.empty(0, np.int32),
                    )
                else:
                    with batch._device():
                        source = batch._active
                        batch._launch(
                            self._kernels[source], len(self._keys), self._width,
                            self._indices, batch._states[source], batch._ticks[source],
                            self._rows, self._ticks,
                        )
                        batch._sync()
                        rows = self._read_buffer(self._rows)
                        ticks = self._read_buffer(self._ticks)
                        result = self._publish(rows, ticks)
            except Exception:
                self._failed_reads += 1
                batch._counts["readback_plan_failed_reads"] += 1
                raise
            self._reads += 1
            batch._counts["readback_plan_reads"] += 1
            return result

    def stats(self):
        with self._batch._lock:
            return dict(
                keys=self._keys, kind=self._kind, closed=self._closed,
                mode=self._batch._mode, selected_worlds=len(self._keys),
                device_allocations=self._allocations, device_owned_bytes=self._cuda_bytes,
                host_workspace_bytes=self._host_bytes,
                index_uploaded_bytes=self._index_bytes, reads=self._reads,
                failed_reads=self._failed_reads, downloaded_bytes=self._downloaded_bytes,
                copied_to_host_bytes=self._host_read_bytes,
            )

    def close(self):
        with self._batch._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._resources.close()
            finally:
                self._batch._readback_plans.discard(self)
                self._indices = self._rows = self._ticks = self._kernels = None
                self._host_bytes = 0

    def __enter__(self):
        with self._batch._lock:
            if self._closed:
                raise RuntimeError("readback plan is closed")
            self._batch._check_open()
        return self

    def __exit__(self, *_):
        self.close()


def readback_for(batch, plan):
    """Public batch.readback(plan) owner gate; identity is never inferred from equal keys."""
    if not isinstance(plan, WorldReadbackPlan):
        raise TypeError("expected a WorldReadbackPlan")
    with batch._lock:
        if plan._batch is not batch:
            raise ValueError("readback plan belongs to a different batch owner")
        return plan.read()
