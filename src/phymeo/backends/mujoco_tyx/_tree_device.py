"""Resident T1 scratch/state banks; same original atomic publication boundary."""

from contextlib import ExitStack

import numpy as np

from ._device import TyxDevice


class TyxTreeDevice(TyxDevice):
    def __init__(self, plan, mode, cache_root, q, v, time, nbody, h, capacities):
        import tyx as tx
        from . import _tree_kernels as kernels

        self.tx, self.mode, self.plan = tx, mode, plan
        self.nbody, self.nworld, self.h = nbody, len(q), float(h)
        self.active = 0
        self.resources = ExitStack()
        self.runtime = tx.Runtime(backend="cuda" if mode == "cuda" else "cpu", cache_root=cache_root)
        self.resources.callback(self.runtime.close)
        self.counts = dict(completed_kernels=0, synchronizations=0, device_allocations=0,
                           static_uploaded_bytes=0, uploaded_bytes=0, downloaded_bytes=0)
        self.geometry = (kernels.geometry, kernels.geometry_other)
        self.mechanics = (kernels.mechanics, kernels.mechanics_other)
        self.solve = (kernels.solve, kernels.solve_other)
        try:
            with tx.use_runtime(self.runtime):
                self.parents = self._buffer(plan.parents)
                self.ranges = self._buffer(plan.body_joint_range)
                self.body_data = self._buffer(plan.body_data)
                self.joints = self._buffer(plan.joint_ids)
                self.joint_data = self._buffer(plan.joint_data)
                self.ancestors = self._buffer(plan.ancestors)
                self.gravity = self._buffer(plan.gravity)
                self.vectors = self._buffer(plan.vectors)
                self.act_ids = self._buffer(plan.actuator_ids)
                self.act_values = self._buffer(plan.actuator_values)
                self.q = (self._buffer(q), self._buffer(q))
                self.v = (self._buffer(v), self._buffer(v))
                self.time = (self._buffer(time), self._buffer(time))
                self.controls = self._buffer(np.zeros((self.nworld, max(plan.nu, 1)), np.float32))
                self.applied_input = self._buffer(np.zeros((self.nworld, 8), np.float32))
                self.body_force = self._buffer(np.zeros((self.nworld, nbody, 6), np.float32))
                self.motion = self._buffer(np.zeros((self.nworld, nbody, 7, 3), np.float32))
                self.rotation = self._buffer(np.zeros((self.nworld, nbody, 3, 3), np.float32))
                self.axes = self._buffer(np.zeros((self.nworld, 8, 3), np.float32))
                self.anchors = self._buffer(np.zeros((self.nworld, 8, 3), np.float32))
                self.work = self._buffer(np.zeros((self.nworld, 8, 3), np.float32))
                self.work_rotation = self._buffer(np.zeros((self.nworld, 3, 3), np.float32))
                self.inertia = self._buffer(np.zeros((self.nworld, nbody, 3, 3), np.float32))
                self.jacobian = self._buffer(np.zeros((self.nworld, nbody, 6, 8), np.float32))
                self.mass = self._buffer(np.zeros((self.nworld, 8, 8), np.float32))
                self.factor = self._buffer(np.zeros((self.nworld, 8, 8), np.float32))
                self.bias = self._buffer(np.zeros((self.nworld, 8), np.float32))
                self.passive = self._buffer(np.zeros((self.nworld, 8), np.float32))
                self.applied = self._buffer(np.zeros((self.nworld, 8), np.float32))
                self.actuator = self._buffer(np.zeros((self.nworld, 8), np.float32))
                self.constraint = self._buffer(np.zeros((self.nworld, 8), np.float32))
                self.rhs = self._buffer(np.zeros((self.nworld, 8), np.float32))
                self.forward = self._buffer(np.zeros((self.nworld, 8), np.float32))
                self.acceleration = self._buffer(np.zeros((self.nworld, 8), np.float32))
                self.integration_acc = self._buffer(np.zeros((self.nworld, 8), np.float32))
                self.rows = self._buffer(np.zeros((self.nworld, 20), np.float32))
                self.status = self._buffer(np.zeros(self.nworld, np.int32))
        except BaseException:
            self.close()
            raise

    def advance(self, controls, applied, body_force):
        old, new = self.active, 1-self.active
        with self.tx.use_runtime(self.runtime):
            self._put(self.controls, controls)
            self._put(self.applied_input, applied)
            self._put(self.body_force, body_force)
            self.geometry[old](
                self.nworld, self.nbody, self.plan.nv, self.parents, self.ranges,
                self.body_data, self.joints, self.joint_data, self.vectors,
                self.q[old], self.v[old], self.motion, self.rotation,
                self.axes, self.anchors, self.work, self.work_rotation, self.status, mode=self.mode,
            )
            self.counts["completed_kernels"] += 1
            self.mechanics[old](
                self.nworld, self.nbody, self.plan.nv, self.plan.nu,
                self.body_data, self.joints, self.ancestors, self.vectors, self.gravity,
                self.act_ids, self.act_values, self.controls, self.applied_input,
                self.body_force, self.q[old], self.v[old], self.motion, self.rotation,
                self.axes, self.anchors, self.inertia, self.jacobian,
                self.mass, self.bias, self.passive, self.applied, self.actuator,
                self.work, self.status, mode=self.mode,
            )
            self.counts["completed_kernels"] += 1
            self.solve[old](
                self.nworld, self.plan.nv, self.h, self.vectors,
                self.mass, self.bias, self.passive, self.applied,
                self.q[old], self.v[old], self.time[old], self.q[new], self.v[new], self.time[new],
                self.factor, self.rhs, self.forward, self.acceleration, self.integration_acc,
                self.constraint, self.rows, self.status, mode=self.mode,
            )
            self.counts["completed_kernels"] += 1
            self.runtime.sync()
            self.counts["synchronizations"] += 1

    def download(self):
        old, target = self.active, 1-self.active
        with self.tx.use_runtime(self.runtime):
            motion = self._get(self.motion)
            result = {name: self._get(buffer) for name, buffer in (
                ("q", self.q[target]), ("v", self.v[target]), ("time", self.time[target]),
                ("qacc", self.acceleration), ("integration_acc", self.integration_acc),
                ("actuator", self.actuator), ("constraint", self.constraint),
                ("rows", self.rows), ("status", self.status),
                ("body_rotation", self.rotation), ("jacobian_com", self.jacobian),
                ("mass_matrix", self.mass), ("bias", self.bias), ("passive", self.passive),
                ("applied", self.applied), ("input_q", self.q[old]),
                ("input_v", self.v[old]), ("input_time", self.time[old]),
            )}
            result["body_position"] = motion[:, :, 0, :]
            result["body_com"] = motion[:, :, 5, :]
            result["smooth_acceleration"] = result["qacc"]
            return result
