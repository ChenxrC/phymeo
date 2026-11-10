"""T3 resident glue: unchanged T1/T2 dynamics plus two normal-contact kernels."""

import numpy as np

from ._device import TyxDevice
from ._floating_device import TyxFloatingDevice
from ._tree_device import TyxTreeDevice


class TyxNormalDevice(TyxDevice):
    """Cold-start normal QP, five/six launches, one publication boundary."""

    def __init__(self, plan, mode, cache_root, q, v, time, nbody, h, capacities):
        from . import _normal_geometry, _normal_solver

        extended_contacts = None
        if plan.extended_contacts:
            from . import _normal_extended_geometry

            extended_contacts = _normal_extended_geometry.extended_contacts
        self._base_type = TyxFloatingDevice if plan.base.kind == 3 else TyxTreeDevice
        self._base_type.__init__(self, plan, mode, cache_root, q, v, time, nbody, h, capacities)
        self.capacities = tuple(map(int, capacities))
        self.contacts = (_normal_geometry.contacts, _normal_geometry.contacts_other)
        self.normal_solve = _normal_solver.solve
        self.extended_contacts = extended_contacts
        self.ngeom = len(plan.geom_ids)
        self.npair = len(plan.pair_ids)
        self.nslot = len(plan.slot_ids)
        self.width = max(8, plan.nv)
        try:
            with self.tx.use_runtime(self.runtime):
                self.geom_ids = self._buffer(plan.geom_ids)
                self.geom_data = self._buffer(plan.geom_data)
                self.pair_ids = self._buffer(plan.pair_ids)
                self.pair_law = self._buffer(plan.pair_law)
                self.geom_world = self._buffer(np.zeros((self.nworld, self.ngeom, 12), np.float32))
                self.normal_rows = self._buffer(np.zeros((self.nworld, self.nslot, 32), np.float32))
                self.normal_jacobian = self._buffer(
                    np.zeros((self.nworld, self.nslot, self.width), np.float32)
                )
                self.contact_work = self._buffer(np.zeros((self.nworld, 8, 3), np.float32))
                self.raw_contacts = self._buffer(
                    np.zeros((self.nworld, self.nslot, 11), np.float32)
                )
                self.extended_work = self._buffer(np.zeros((self.nworld, 64, 3), np.float32))
                self.solver_diagnostics = self._buffer(np.zeros((self.nworld, 8), np.float32))
                self.normal_factor = self._buffer(
                    np.zeros((self.nworld, self.width, self.width), np.float32)
                )
                self.normal_mass_work = self._buffer(
                    np.zeros((self.nworld, self.width), np.float32)
                )
                self.smooth_acc = self._buffer(np.zeros((self.nworld, self.width), np.float32))
                self.response = self._buffer(
                    np.zeros((self.nworld, self.nslot, self.width), np.float32)
                )
                self.dual_matrix = self._buffer(
                    np.zeros((self.nworld, self.nslot, self.nslot), np.float32)
                )
                self.normal_linear = self._buffer(np.zeros((self.nworld, self.nslot), np.float32))
                self.normal_force = self._buffer(np.zeros((self.nworld, self.nslot), np.float32))
                self.normal_gradient = self._buffer(np.zeros((self.nworld, self.nslot), np.float32))
                self.normal_working = self._buffer(np.zeros((self.nworld, self.nslot), np.int32))
                self.face_factor = self._buffer(
                    np.zeros((self.nworld, self.nslot, self.nslot), np.float32)
                )
                self.face_forward = self._buffer(np.zeros((self.nworld, self.nslot), np.float32))
                self.normal_trial = self._buffer(np.zeros((self.nworld, self.nslot), np.float32))
                self.normal_constraint = self._buffer(
                    np.zeros((self.nworld, self.width), np.float32)
                )
                self.total_applied = self._buffer(np.zeros((self.nworld, self.width), np.float32))
        except BaseException:
            self.close()
            raise

    def advance(self, controls, applied, body_force):
        old, new = self.active, 1 - self.active
        with self.tx.use_runtime(self.runtime):
            self._put(self.controls, controls)
            self._put(self.applied_input, applied)
            self._put(self.body_force, body_force)
            geometry_args = [self.nworld, self.nbody, self.plan.nv]
            if self.plan.base.kind == 3:
                geometry_args.append(self.plan.root_body)
            self.geometry[old](
                *geometry_args,
                self.parents,
                self.ranges,
                self.body_data,
                self.joints,
                self.joint_data,
                self.vectors,
                self.q[old],
                self.v[old],
                self.motion,
                self.rotation,
                self.axes,
                self.anchors,
                self.work,
                self.work_rotation,
                self.status,
                mode=self.mode,
            )
            self.counts["completed_kernels"] += 1
            self.mechanics[old](
                self.nworld,
                self.nbody,
                self.plan.nv,
                self.plan.nu,
                self.body_data,
                self.joints,
                self.ancestors,
                self.vectors,
                self.gravity,
                self.act_ids,
                self.act_values,
                self.controls,
                self.applied_input,
                self.body_force,
                self.q[old],
                self.v[old],
                self.motion,
                self.rotation,
                self.axes,
                self.anchors,
                self.inertia,
                self.jacobian,
                self.mass,
                self.bias,
                self.passive,
                self.applied,
                self.actuator,
                self.work,
                self.status,
                mode=self.mode,
            )
            self.counts["completed_kernels"] += 1
            if self.extended_contacts is not None:
                self.extended_contacts(
                    self.nworld,
                    self.ngeom,
                    self.npair,
                    self.nslot,
                    self.geom_ids,
                    self.geom_data,
                    self.pair_ids,
                    self.pair_law,
                    self.motion,
                    self.rotation,
                    self.geom_world,
                    self.raw_contacts,
                    self.extended_work,
                    self.status,
                    mode=self.mode,
                )
                self.counts["completed_kernels"] += 1
            self.contacts[old](
                self.nworld,
                self.plan.nv,
                self.ngeom,
                self.npair,
                self.nslot,
                *self.capacities,
                self.h,
                self.geom_ids,
                self.geom_data,
                self.pair_ids,
                self.pair_law,
                self.motion,
                self.rotation,
                self.jacobian,
                self.v[old],
                self.geom_world,
                self.normal_rows,
                self.normal_jacobian,
                self.contact_work,
                self.raw_contacts,
                self.solver_diagnostics,
                self.status,
                mode=self.mode,
            )
            self.counts["completed_kernels"] += 1
            self.normal_solve(
                self.nworld,
                self.plan.nv,
                self.nslot,
                self.plan.solver_iterations,
                self.plan.solver_tolerance,
                self.mass,
                self.bias,
                self.passive,
                self.applied,
                self.normal_jacobian,
                self.normal_rows,
                self.normal_factor,
                self.normal_mass_work,
                self.smooth_acc,
                self.response,
                self.dual_matrix,
                self.normal_linear,
                self.normal_force,
                self.normal_gradient,
                self.normal_working,
                self.face_factor,
                self.face_forward,
                self.normal_trial,
                self.normal_constraint,
                self.total_applied,
                self.solver_diagnostics,
                self.status,
                mode=self.mode,
            )
            self.counts["completed_kernels"] += 1
            # Existing solve computes M^-1(total RHS), then (M+hD)^-1(total RHS),
            # and integrates the original native coordinates. Its cleared
            # constraint/rows buffers are deliberately disposable here.
            self.solve[old](
                self.nworld,
                self.plan.nv,
                self.h,
                self.vectors,
                self.mass,
                self.bias,
                self.passive,
                self.total_applied,
                self.q[old],
                self.v[old],
                self.time[old],
                self.q[new],
                self.v[new],
                self.time[new],
                self.factor,
                self.rhs,
                self.forward,
                self.acceleration,
                self.integration_acc,
                self.constraint,
                self.rows,
                self.status,
                mode=self.mode,
            )
            self.counts["completed_kernels"] += 1
            self.runtime.sync()
            self.counts["synchronizations"] += 1

    def download(self):
        result = self._base_type.download(self)
        with self.tx.use_runtime(self.runtime):
            for name, buffer in (
                ("normal_rows", self.normal_rows),
                ("normal_jacobian", self.normal_jacobian),
                ("dual_matrix", self.dual_matrix),
                ("solver_diagnostics", self.solver_diagnostics),
                ("smooth_acceleration", self.smooth_acc),
                ("constraint", self.normal_constraint),
            ):
                result[name] = self._get(buffer)
        return result
