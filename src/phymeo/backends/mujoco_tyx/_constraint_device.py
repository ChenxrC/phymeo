"""Resident mixed constraint assembly, bounded face solve and atomic integration."""

from contextlib import contextmanager

import numpy as np

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
        entry_body_ids=None,
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
            entry_body_ids=entry_body_ids,
        )
        try:
            self.capacities = tuple(map(int, capacities))
            self.contacts = (_normal_geometry.contacts, _normal_geometry.contacts_other)
            self.assemble = (_constraint_rows.assemble, _constraint_rows.assemble_other)
            self.constraint_solve = _constraint_solver.solve
            self._trial_scope = None
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

    def _assemble_rows(self, old):
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

    def _prepare_entry(self, old, controls, applied, body_force, *, assemble_contacts):
        new = 1 - old
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
        if assemble_contacts:
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
            self._assemble_rows(old)
        elif self.plan.nfriction or self.plan.nlimit:
            self._assemble_rows(old)

    def _solve_rows(self, mass, applied):
        self.constraint_solve(
            self.nworld,
            self.plan.nv,
            self.capacities[1],
            self.plan.solver_iterations,
            self.plan.solver_tolerance,
            mass,
            self.bias,
            self.passive,
            applied,
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

    def _integrate_candidate(self, old, new):
        # Shared native tail retains physical M acceleration. The live mode
        # consumes the common velocity; the old mode solves M+hD as before.
        # Both integrate native q/clock once; constraint/rows are disposable.
        solve_args = [self.nworld, self.plan.nv]
        if self.plan.base.kind == 2:
            solve_args.extend((self.plan.nq, len(self.plan.joint_ids)))
        solve_args.append(self.h)
        if self.plan.base.kind == 2:
            solve_args.append(self.joints)
        integrator = self.solve[old]
        if getattr(self.plan, "area_friction_laws", ()):
            if self.plan.base.kind == 3:
                from ._floating_kernels import solve_live, solve_live_other
            else:
                from ._tree_kernels import solve_live, solve_live_other
            integrator = (solve_live, solve_live_other)[old]
        integrator(
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

    @contextmanager
    def prepare_native_rows(self, controls, applied, body_force):
        """Assemble the active entry once and borrow its CUDA working set.

        The owner holds exclusive access until return. Each evaluation accepts
        an additional resident generalized force, or None for the original
        load. Results and per-world status stay in the existing solve buffers.
        A failed world remains failed until a new entry is prepared. Pass
        integrate=True on the last load to write a one-step candidate into
        the inactive physical bank. That call checks all working-set statuses
        before integration. The caller must still check the integrator status
        and validate the candidate to publish it. This scope never publishes.
        """
        from ._constraint_solver import add_trial_load

        if self.resources.state != "OPEN" or self._trial_scope is not None:
            raise RuntimeError("native row workspace is closed or in use")
        if self.mode != "cuda" or self.activation_enabled or hasattr(self.plan, "area_pair_names"):
            raise ValueError("native row trials require a stateless native CUDA constraint plan")
        old = self.active

        finalized = False

        def evaluate(extra_load=None, *, integrate=False):
            nonlocal finalized
            if type(integrate) is not bool:
                raise ValueError("integrate must be a boolean")
            if finalized:
                raise RuntimeError("native trial entry already finalized")
            if (
                self._trial_scope is not evaluate
                or self.active != old
                or self.resources.state != "OPEN"
            ):
                raise RuntimeError("native row trial entry has expired")
            with self.tx.use_runtime(self.runtime):
                trial = self.applied
                if extra_load is not None:
                    if (
                        not isinstance(extra_load, self.tx.CudaArray)
                        or extra_load.shape != self.applied.shape
                        or np.dtype(extra_load.host_dtype) != np.float32
                        or extra_load.device_ordinal != self.applied.device_ordinal
                    ):
                        raise ValueError("expected a same-device padded FP32 generalized load")
                    extra_load._ensure_alive()
                    # The integrator RHS is idle until the physical step tail.
                    # The working-set kernel clears its own outputs at each solve.
                    trial = self.rhs
                    start, dest = int(extra_load.device_ptr), int(trial.device_ptr)
                    if start < dest + trial.nbytes and dest < start + extra_load.nbytes:
                        raise ValueError("extra load overlaps native trial scratch")
                    add_trial_load(
                        self.nworld,
                        self.plan.nv,
                        self.width,
                        self.applied,
                        extra_load,
                        trial,
                        self.status,
                        mode=self.mode,
                    )
                    self.counts["completed_kernels"] += 1
                self._solve_rows(self.mass, trial)
                if integrate:
                    finalized = True
                    if np.any(self._get(self.status)):
                        raise RuntimeError("native trial rejected before candidate integration")
                    self._integrate_candidate(old, 1 - old)

        self._trial_scope = evaluate
        try:
            with self.tx.use_runtime(self.runtime):
                self._prepare_entry(old, controls, applied, body_force, assemble_contacts=True)
            yield evaluate
        finally:
            self._trial_scope = None

    def publish(self):
        if self._trial_scope is not None:
            raise RuntimeError("cannot publish during native row trials")
        super().publish()

    def advance(self, controls, applied, body_force):
        if self._trial_scope is not None:
            raise RuntimeError("cannot advance during native row trials")
        old, new = self.active, 1 - self.active
        with self.tx.use_runtime(self.runtime):
            live_friction = bool(getattr(self.plan, "area_friction_laws", ()))
            mixed_support = live_friction and self.plan.damped_contact_dofs > 0
            self._prepare_entry(
                old,
                controls,
                applied,
                body_force,
                assemble_contacts=not live_friction or mixed_support,
            )
            if live_friction:
                from ._area_friction_device import solve_contact

                solve_contact(self, old)
            else:
                response_mass = self.mass
                if hasattr(self.plan, "area_pair_names"):
                    from ._area_pressure_kernels import append_rows

                    packet = self.area_packet
                    self._put(self.area_input, packet[0])
                    self._put(self.area_measure_input, packet[4])
                    self._put(self.area_pair_index, packet[1])
                    self._put(self.area_sizes, packet[2])
                    append_rows(
                        self.nworld,
                        self.plan.nv,
                        self.capacities[1],
                        self.h,
                        int(self.plan.scalar_damped_contacts),
                        self.area_body_ids,
                        self.area_input,
                        self.area_pair_index,
                        self.area_sizes,
                        self.motion,
                        self.jacobian,
                        self.v[old],
                        self.mass,
                        self.vectors,
                        self.area_response_mass,
                        self.efc_ids,
                        self.efc_data,
                        self.efc_jacobian,
                        self.constraint_counts,
                        self.status,
                        mode=self.mode,
                    )
                    self.counts["completed_kernels"] += 1
                    if not self.plan.scalar_damped_contacts:
                        response_mass = self.area_response_mass
                self._solve_rows(response_mass, self.applied)
            if not live_friction or mixed_support:
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
            self._integrate_candidate(old, new)
            if live_friction and (mixed_support or self.plan.nfriction or self.plan.nlimit):
                from ._area_friction_live_kernels import store_native_gradients

                store_native_gradients(
                    self.nworld,
                    self.plan.nv,
                    self.capacities[1],
                    self.constraint_counts,
                    self.acceleration,
                    self.efc_data,
                    self.efc_jacobian,
                    mode=self.mode,
                )
                self.counts["completed_kernels"] += 1
            if not live_friction and getattr(self.plan, "scalar_damped_contacts", False):
                from ._area_pressure_kernels import publish_scalar_area_rows

                publish_scalar_area_rows(
                    self.nworld,
                    self.h,
                    self.area_input,
                    self.area_sizes,
                    self.constraint_counts,
                    self.efc_jacobian,
                    self.efc_data,
                    self.integration_acc,
                    self.status,
                    mode=self.mode,
                )
                self.counts["completed_kernels"] += 1
            self.runtime.sync()
            self.counts["synchronizations"] += 1

    def download(self):
        result = self._base_type.download(self)
        if getattr(self.plan, "area_friction_laws", ()):
            from ._area_friction_device import download

            return download(self, result)
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
                if getattr(self.plan, "scalar_damped_contacts", False) and name == "qacc":
                    result["area_response_acceleration"] = result["integration_acc"]
                    # Keep the physical acceleration from the independent integration tail.
                elif hasattr(self.plan, "area_pair_names") and name == "qacc":
                    # The base integrator's qacc uses physical M; the constraint
                    # solve uses the actual velocity response M+hD.
                    result["area_response_acceleration"] = self._get(buffer)
                elif (
                    hasattr(self.plan, "area_pair_names")
                    and not self.plan.scalar_damped_contacts
                    and name == "smooth_acceleration"
                ):
                    result["area_smooth_response_acceleration"] = self._get(buffer)
                    result["smooth_acceleration"] = result["area_smooth_response_acceleration"]
                else:
                    result[name] = self._get(buffer)
        return result
