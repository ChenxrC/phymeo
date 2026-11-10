"""Owned TYX resident buffers and atomic candidate bank; no host physics step."""

from contextlib import ExitStack

import numpy as np


class TyxDevice:
    def __init__(self, plan, mode, cache_root, q, v, time, nbody, h, capacities):
        import tyx as tx

        from . import _kernels

        self.tx, self.kernels = tx, (_kernels.advance, _kernels.advance_other)
        self.plan, self.mode, self.h = plan, mode, float(h)
        self.nbody, self.capacities = nbody, capacities
        self.nworld = len(q)
        self.resources = ExitStack()
        self.runtime = tx.Runtime(
            backend="cuda" if mode == "cuda" else "cpu", cache_root=cache_root
        )
        self.resources.callback(self.runtime.close)
        self.active = 0
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
                self.matrices = self._buffer(plan.matrices)
                self.vectors = self._buffer(plan.vectors)
                self.body_jac = self._buffer(plan.body_jacobian)
                self.point_jac = self._buffer(plan.point_jacobian)
                self.law = self._buffer(plan.contact)
                self.act_ids = self._buffer(plan.actuator_ids)
                self.act_values = self._buffer(plan.actuator_values)
                self.q = (self._buffer(q), self._buffer(q))
                self.v = (self._buffer(v), self._buffer(v))
                self.time = (self._buffer(time), self._buffer(time))
                self.controls = self._buffer(np.zeros((self.nworld, max(plan.nu, 1)), np.float32))
                self.applied = self._buffer(np.zeros((self.nworld, 8), np.float32))
                self.body_force = self._buffer(np.zeros((self.nworld, nbody, 6), np.float32))
                self.force = self._buffer(np.zeros((self.nworld, 8), np.float32))
                self.smooth_acc = self._buffer(np.zeros((self.nworld, 8), np.float32))
                self.acceleration = self._buffer(np.zeros((self.nworld, 8), np.float32))
                self.integration_acc = self._buffer(np.zeros((self.nworld, 8), np.float32))
                self.actuator = self._buffer(np.zeros((self.nworld, 8), np.float32))
                self.constraint = self._buffer(np.zeros((self.nworld, 8), np.float32))
                self.rows = self._buffer(np.zeros((self.nworld, 20), np.float32))
                self.status = self._buffer(np.zeros(self.nworld, np.int32))
        except BaseException:
            self.close()
            raise

    def _buffer(self, value):
        if self.mode == "interpret":
            return value.copy()
        result = self.tx.cuda_array(value)
        self.resources.callback(result.free)
        self.counts["device_allocations"] += 1
        self.counts["static_uploaded_bytes"] += value.nbytes
        return result

    def _put(self, buffer, value):
        if self.mode == "interpret":
            np.copyto(buffer, value)
        else:
            buffer.copy_from(value)
            self.counts["uploaded_bytes"] += value.nbytes

    def _get(self, buffer):
        if self.mode == "interpret":
            return buffer.copy()
        result = buffer.to_numpy()
        self.counts["downloaded_bytes"] += result.nbytes
        return result

    def advance(self, controls, applied, body_force):
        plan = self.plan
        old, new = self.active, 1 - self.active
        with self.tx.use_runtime(self.runtime):
            self._put(self.controls, controls)
            self._put(self.applied, applied)
            self._put(self.body_force, body_force)
            self.kernels[old](
                self.nworld,
                plan.kind,
                plan.nq,
                plan.nv,
                plan.nu,
                self.nbody,
                plan.body,
                int(plan.pair is not None),
                *self.capacities,
                self.h,
                self.matrices,
                self.vectors,
                self.body_jac,
                self.point_jac,
                self.law,
                self.act_ids,
                self.act_values,
                self.controls,
                self.applied,
                self.body_force,
                self.q[old],
                self.v[old],
                self.time[old],
                self.q[new],
                self.v[new],
                self.time[new],
                self.force,
                self.smooth_acc,
                self.acceleration,
                self.integration_acc,
                self.actuator,
                self.constraint,
                self.rows,
                self.status,
                mode=self.mode,
            )
            self.counts["completed_kernels"] += 1
            self.runtime.sync()
            self.counts["synchronizations"] += 1

    def download(self):
        target = 1 - self.active
        with self.tx.use_runtime(self.runtime):
            return {
                name: self._get(value)
                for name, value in (
                    ("q", self.q[target]),
                    ("v", self.v[target]),
                    ("time", self.time[target]),
                    ("qacc", self.acceleration),
                    ("integration_acc", self.integration_acc),
                    ("actuator", self.actuator),
                    ("constraint", self.constraint),
                    ("rows", self.rows),
                    ("status", self.status),
                )
            }

    def publish(self):
        self.active = 1 - self.active

    def close(self):
        self.resources.close()
