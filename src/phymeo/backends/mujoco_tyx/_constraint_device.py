"""Resident mixed constraint assembly, bounded face solve and atomic integration."""

from . import _actuator_device
from ._device import TyxDevice
from ._floating_device import TyxFloatingDevice
from ._tree_device import TyxTreeDevice


class TyxConstraintDevice(TyxDevice):
    def __init__(
        self,
        plan,
        mode,
        cache_root,
        q,
        v,
        time,
        nbody,
        h,
        capacities,
        *,
        activation=None,
        resource_budget=None,
    ):
        from . import _constraint_rows, _constraint_solver, _normal_geometry

        self._base_type = TyxFloatingDevice if plan.base.kind == 3 else TyxTreeDevice
        self._base_type.__init__(
            self,
            plan,
            mode,
            cache_root,
            q,
            v,
            time,
            nbody,
            h,
            capacities,
            activation=activation,
            resource_budget=resource_budget,
        )
        try:
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
            with self.tx.use_runtime(self.runtime):
                # The complete final-route manifest was reserved by the base.
                # Only this extension's remaining allocations are materialized.
                for entry in self.allocation_manifest.entries:
                    if entry.name not in self._allocated_names:
                        setattr(self, entry.name, self._allocate(entry.name))
                self._complete_allocations()
        except BaseException as primary:
            self._close_after_construction_failure(primary)
            raise

    def advance(self, controls, applied, body_force):
        old, new = self.active, 1 - self.active
        with self.tx.use_runtime(self.runtime):
            self._put(self.controls, controls)
            self._put(self.applied_input, applied)
            self._put(self.body_force, body_force)
            geometry_args = [self.nworld, self.nbody, self.plan.nv]
            if self.plan.base.kind == 3:
                geometry_args.append(len(self.plan.joint_ids))
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
                0 if self.activation_enabled else self.plan.nu,
                self.body_data,
                self.dofs,
                self.ancestors,
                self.vectors,
                self.gravity,
                self.act_ids,
                self.act_values,
                int(self.transmission_enabled),
                self.transmission_offsets,
                self.transmission_dofs,
                self.transmission_moment,
                self.transmission_fields,
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
            _actuator_device.advance(self, old, new)
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
                self.free_ids,
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
            solve_args = [self.nworld, self.plan.nv]
            if self.plan.base.kind == 2:
                solve_args.extend((self.plan.nq, len(self.plan.joint_ids)))
            solve_args.append(self.h)
            if self.plan.base.kind == 2:
                solve_args.append(self.joints)
            self.solve[old](
                *solve_args,
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
                *(
                    (
                        self.plan.nq,
                        len(self.plan.joint_ids),
                        self.joints,
                    )
                    if self.plan.base.kind == 3
                    else ()
                ),
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
