"""Resident mixed constraint assembly, bounded face solve and atomic integration."""

import numpy as np

from ._device import TyxDevice
from ._floating_device import TyxFloatingDevice
from ._tree_device import TyxTreeDevice


class TyxConstraintDevice(TyxDevice):
    def __init__(self, plan, mode, cache_root, q, v, time, nbody, h, capacities):
        from . import _constraint_rows, _constraint_solver, _normal_geometry

        self._base_type = TyxFloatingDevice if plan.base.kind == 3 else TyxTreeDevice
        self._base_type.__init__(self, plan, mode, cache_root, q, v, time, nbody, h, capacities)
        self.capacities = tuple(map(int, capacities))
        self.contacts = (_normal_geometry.contacts, _normal_geometry.contacts_other)
        self.assemble = (_constraint_rows.assemble, _constraint_rows.assemble_other)
        self.constraint_solve = _constraint_solver.solve
        self.decode = _constraint_rows.decode
        self.extended_contacts = None
        if plan.extended_contacts:
            from . import _normal_extended_geometry

            self.extended_contacts = _normal_extended_geometry.extended_contacts
        self.ngeom, self.npair, self.nslot = plan.ngeom, plan.npair, plan.nslot
        self.width = max(8, plan.nv)
        w, vsize, slots = self.nworld, self.width, max(1, plan.nslot)
        contact_slots, efc_slots = (max(1, cap) for cap in self.capacities)
        try:
            with self.tx.use_runtime(self.runtime):
                for name in (
                    "geom_ids",
                    "geom_data",
                    "pair_ids",
                    "pair_law",
                    "pair_condim",
                    "pair_friction",
                    "slot_ids",
                    "friction_ids",
                    "friction_law",
                    "limit_ids",
                    "limit_law",
                ):
                    setattr(self, name, self._buffer(getattr(plan, name)))
                arrays = {
                    "geom_world": (w, max(1, plan.ngeom), 12),
                    "normal_rows": (w, slots, 32),
                    "normal_jacobian": (w, slots, vsize),
                    "contact_work": (w, 8, 3),
                    "raw_contacts": (w, slots, 11),
                    "extended_work": (w, 64, 3),
                    "geometry_diagnostics": (w, 8),
                    "solver_diagnostics": (w, 10),
                    "efc_data": (w, efc_slots, 14),
                    "efc_jacobian": (w, efc_slots, vsize),
                    "contact_jacobian": (w, contact_slots, 6, vsize),
                    "contact_wrench": (w, contact_slots, 6),
                    "physical_factor": (w, vsize, vsize),
                    "mass_work": (w, vsize),
                    "smooth_acc": (w, vsize),
                    "face_matrix": (w, vsize, vsize),
                    "face_factor": (w, vsize, vsize),
                    "face_rhs": (w, vsize),
                    "face_forward": (w, vsize),
                    "trial_acc": (w, vsize),
                    "trial_force": (w, efc_slots),
                    "previous_force": (w, efc_slots),
                    "mixed_constraint": (w, vsize),
                    "total_applied": (w, vsize),
                    "physical_acc": (w, vsize),
                }
                for name, shape in arrays.items():
                    setattr(self, name, self._buffer(np.zeros(shape, np.float32)))
                for name, shape in {
                    "efc_ids": (w, efc_slots, 5),
                    "contact_map": (w, contact_slots, 3),
                    "constraint_counts": (w, 3),
                    "working": (w, efc_slots),
                    "previous_working": (w, efc_slots),
                }.items():
                    setattr(self, name, self._buffer(np.zeros(shape, np.int32)))
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
                self.geometry_diagnostics,
                self.status,
                mode=self.mode,
            )
            self.counts["completed_kernels"] += 1
            self.assemble[old](
                self.nworld,
                self.plan.nv,
                self.nslot,
                self.plan.nfriction,
                self.plan.nlimit,
                *self.capacities,
                self.h,
                self.plan.impratio,
                self.slot_ids,
                self.geom_ids,
                self.pair_condim,
                self.pair_friction,
                self.pair_law,
                self.friction_ids,
                self.friction_law,
                self.limit_ids,
                self.limit_law,
                self.q[old],
                self.v[old],
                self.motion,
                self.jacobian,
                self.normal_rows,
                self.efc_ids,
                self.efc_data,
                self.efc_jacobian,
                self.contact_map,
                self.contact_jacobian,
                self.constraint_counts,
                self.status,
                mode=self.mode,
            )
            self.counts["completed_kernels"] += 1
            self.constraint_solve(
                self.nworld,
                self.plan.nv,
                self.capacities[1],
                self.plan.solver_iterations,
                self.plan.solver_tolerance,
                self.mass,
                self.bias,
                self.passive,
                self.applied,
                self.efc_data,
                self.efc_jacobian,
                self.constraint_counts,
                self.physical_factor,
                self.mass_work,
                self.smooth_acc,
                self.face_matrix,
                self.face_factor,
                self.face_rhs,
                self.face_forward,
                self.trial_acc,
                self.trial_force,
                self.working,
                self.previous_working,
                self.previous_force,
                self.mixed_constraint,
                self.total_applied,
                self.physical_acc,
                self.solver_diagnostics,
                self.status,
                mode=self.mode,
            )
            self.counts["completed_kernels"] += 1
            self.decode(
                self.nworld,
                self.plan.nv,
                *self.capacities,
                self.slot_ids,
                self.pair_condim,
                self.pair_friction,
                self.contact_map,
                self.constraint_counts,
                self.efc_data,
                self.contact_wrench,
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
                ("solver_diagnostics", self.solver_diagnostics),
                ("smooth_acceleration", self.smooth_acc),
                ("constraint", self.mixed_constraint),
                ("qacc", self.physical_acc),
                ("efc_ids", self.efc_ids),
                ("efc_data", self.efc_data),
                ("efc_J", self.efc_jacobian),
                ("counts", self.constraint_counts),
                ("working_set", self.working),
                ("contact_map", self.contact_map),
                ("contact_J", self.contact_jacobian),
                ("contact_wrench", self.contact_wrench),
            ):
                result[name] = self._get(buffer)
        return result
