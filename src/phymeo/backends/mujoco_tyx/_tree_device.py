"""Resident T1 scratch/state banks; same original atomic publication boundary."""

from . import _actuator_device
from ._device import TyxDevice


class TyxTreeDevice(TyxDevice):
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
        import tyx as tx

        from . import _tree_kernels as kernels

        if plan.dof_ids is None:
            raise ValueError("fixed-tree native DOF mapping is required")
        self.activation_enabled = _actuator_device.validate(plan, activation, len(q))
        self.transmission_enabled = plan.joint_transmission is not None
        self.tx, self.mode, self.plan = tx, mode, plan
        self.nbody, self.nworld, self.h = nbody, len(q), float(h)
        self.active = 0
        self.capacities = tuple(map(int, capacities))
        self.counts = dict(
            completed_kernels=0,
            synchronizations=0,
            device_allocations=0,
            static_uploaded_bytes=0,
            uploaded_bytes=0,
            downloaded_bytes=0,
        )
        self.geometry = (kernels.geometry, kernels.geometry_other)
        self.mechanics = (kernels.mechanics, kernels.mechanics_other)
        self.solve = (kernels.solve, kernels.solve_other)
        self.native_integrators = kernels.native_integrators
        self._prepare_allocations(plan, self.nworld, nbody, capacities, resource_budget)
        try:
            self._open_runtime(cache_root)
            with self.runtime_scope():
                for name in (
                    "parents",
                    "ranges",
                    "body_data",
                    "joints",
                    "dofs",
                    "joint_data",
                    "ancestors",
                    "gravity",
                    "vectors",
                    "act_ids",
                    "act_values",
                ):
                    setattr(self, name, self._allocate(name))
                self.q = (self._allocate("q.0", q), self._allocate("q.1", q))
                self.v = (self._allocate("v.0", v), self._allocate("v.1", v))
                self.time = (self._allocate("time.0", time), self._allocate("time.1", time))
                _actuator_device.allocate(self, activation)
                for name in ("controls", "applied_input", "body_force"):
                    setattr(self, name, self._allocate(name))
                for alias, target in self.allocation_manifest.aliases:
                    setattr(self, alias, getattr(self, target))
                if self.transmission_enabled:
                    for name in (
                        "transmission_offsets",
                        "transmission_dofs",
                        "transmission_moment",
                        "transmission_fields",
                    ):
                        setattr(self, name, self._allocate(name))
                for name in (
                    "motion",
                    "rotation",
                    "axes",
                    "anchors",
                    "work",
                    "work_rotation",
                    "inertia",
                    "jacobian",
                    "mass",
                    "factor",
                    "bias",
                    "passive",
                    "applied",
                    "actuator",
                    "constraint",
                    "rhs",
                    "forward",
                    "acceleration",
                    "integration_acc",
                    "rows",
                    "status",
                ):
                    setattr(self, name, self._allocate(name))
                self._allocate_entry(entry_body_ids)
                if not getattr(plan, "normal_contact", False) and not getattr(
                    plan, "constraint_contact", False
                ):
                    self._complete_allocations()
        except BaseException as primary:
            self._close_after_construction_failure(primary)
            raise

    def advance(self, controls, applied, body_force):
        old, new = self.active, 1 - self.active
        with self.runtime_scope():
            self._put(self.controls, controls)
            self._put(self.applied_input, applied)
            self._put(self.body_force, body_force)
            self.launch_kernel(
                self.geometry[old],
                self.nworld,
                self.nbody,
                self.plan.nv,
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
            )
            self.counts["completed_kernels"] += 1
            self.launch_kernel(
                self.mechanics[old],
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
            )
            self.counts["completed_kernels"] += 1
            _actuator_device.advance(self, old, new)
            self.launch_kernel(
                self.native_integrators[old],
                self.nworld,
                self.plan.nv,
                self.plan.nq,
                len(self.plan.joint_ids),
                self.h,
                self.joints,
                self.vectors,
                self.mass,
                self.bias,
                self.passive,
                self.applied,
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
                0,  # Ordinary tree step uses the native damped integration velocity.
            )
            self.counts["completed_kernels"] += 1
            self.runtime.sync()
            self.counts["synchronizations"] += 1

    def download(self):
        old, target = self.active, 1 - self.active
        with self.runtime_scope():
            motion = self._get(self.motion)
            result = {
                name: self._get(buffer)
                for name, buffer in (
                    ("q", self.q[target]),
                    ("v", self.v[target]),
                    ("time", self.time[target]),
                    ("qacc", self.acceleration),
                    ("integration_acc", self.integration_acc),
                    ("actuator", self.actuator),
                    ("constraint", self.constraint),
                    ("rows", self.rows),
                    ("status", self.status),
                    ("body_rotation", self.rotation),
                    ("jacobian_com", self.jacobian),
                    ("mass_matrix", self.mass),
                    ("bias", self.bias),
                    ("passive", self.passive),
                    ("applied", self.applied),
                    ("input_q", self.q[old]),
                    ("input_v", self.v[old]),
                    ("input_time", self.time[old]),
                )
            }
            _actuator_device.download(self, result, old, target)
            if self.transmission_enabled:
                result["joint_transmission_fields"] = self._get(self.transmission_fields)
            result["body_position"] = motion[:, :, 0, :]
            result["body_com"] = motion[:, :, 5, :]
            result["smooth_acceleration"] = result["qacc"]
            return result
