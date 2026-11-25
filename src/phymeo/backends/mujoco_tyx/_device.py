"""Owned TYX resident buffers and atomic candidate bank; no host physics step."""

from contextlib import nullcontext
from types import MappingProxyType

import numpy as np

from ..mujoco_resource_budget import DEVICE_PAYLOAD_EXCLUSIONS, MuJoCoTyxResourceBudget
from ._allocation_manifest import allocation_manifest
from ._resources import DeviceResources


class TyxDevice:
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
        resource_budget=None,
        entry_body_ids=None,
    ):
        import tyx as tx

        from . import _kernels

        self.tx, self.kernels = tx, (_kernels.advance, _kernels.advance_other)
        self.plan, self.mode, self.h = plan, mode, float(h)
        self.nbody, self.capacities = nbody, capacities
        self.nworld = len(q)
        self.active = 0
        self.counts = dict(
            completed_kernels=0,
            synchronizations=0,
            device_allocations=0,
            static_uploaded_bytes=0,
            uploaded_bytes=0,
            downloaded_bytes=0,
        )
        self._prepare_allocations(plan, self.nworld, nbody, capacities, resource_budget)
        try:
            self._open_runtime(cache_root)
            if mode == "musa":
                from . import _entry_kernels

                # Preserve construction-time validation for the original slider route.
                for kernel in (*self.kernels, _entry_kernels.state):
                    self._musa_kernel(kernel)
                if plan.pair is not None:
                    self._musa_kernel(_entry_kernels.legacy)
            with self.runtime_scope():
                for name in (
                    "matrices",
                    "vectors",
                    "body_jac",
                    "point_jac",
                    "law",
                    "act_ids",
                    "act_values",
                ):
                    setattr(self, name, self._allocate(name))
                self.q = (self._allocate("q.0", q), self._allocate("q.1", q))
                self.v = (self._allocate("v.0", v), self._allocate("v.1", v))
                self.time = (self._allocate("time.0", time), self._allocate("time.1", time))
                for name in (
                    "controls",
                    "applied",
                    "body_force",
                    "force",
                    "smooth_acc",
                    "acceleration",
                    "integration_acc",
                    "actuator",
                    "constraint",
                    "rows",
                    "status",
                ):
                    setattr(self, name, self._allocate(name))
                self._allocate_entry(entry_body_ids)
                self._complete_allocations()
        except BaseException as primary:
            self._close_after_construction_failure(primary)
            raise

    def _prepare_allocations(self, plan, nworld, nbody, capacities, resource_budget):
        if resource_budget is not None and type(resource_budget) is not MuJoCoTyxResourceBudget:
            raise TypeError("resource_budget must be MuJoCoTyxResourceBudget or None")
        self.allocation_manifest = allocation_manifest(plan, nworld, nbody, capacities)
        self._allocation_entries = {entry.name: entry for entry in self.allocation_manifest.entries}
        self._allocated_names = set()
        self._allocation_entry = None
        self._shared_resource_budget = resource_budget is not None
        self._allocation_budget = resource_budget
        self._allocation_reservation = None
        if self.mode in ("cuda", "musa"):
            if self._allocation_budget is None:
                self._allocation_budget = MuJoCoTyxResourceBudget(
                    self.allocation_manifest.payload_bytes
                )
            self._allocation_reservation = self._allocation_budget.reserve(
                self.allocation_manifest.payload_bytes, backend=self.mode
            )
        try:
            self.resources = DeviceResources(self._allocation_reservation)
        except BaseException:
            if self._allocation_reservation is not None:
                self._allocation_reservation.finish()
            raise

    def _allocate(self, name, value=None):
        if name not in self._allocation_entries or name in self._allocated_names:
            raise RuntimeError("undeclared or duplicate TYX allocation")
        entry = self._allocation_entries[name]
        if value is None:
            if entry.source is None:
                value = np.zeros(entry.shape, dtype=np.dtype(entry.dtype))
            elif entry.source.startswith("input:"):
                raise ValueError("manifest allocation requires its actual packed input")
            else:
                value = self.plan
                for part in entry.source.split("."):
                    value = getattr(value, part)
        if (
            not isinstance(value, np.ndarray)
            or value.shape != entry.shape
            or value.dtype != np.dtype(entry.dtype)
            or value.nbytes != entry.nbytes
        ):
            raise ValueError(f"TYX allocation input differs from manifest: {name}")
        self._allocation_entry = entry
        self.resources.describe_buffer(entry)
        try:
            result = self._buffer(value)
            self._allocated_names.add(name)
            return result
        finally:
            self._allocation_entry = None
            # This is metadata cleanup, including when the allocator returned no owner.
            self.resources._allocation_entry = None

    def _allocate_entry(self, body_ids):
        from ._entry_storage import allocate_entry

        allocate_entry(self, body_ids=body_ids)

    def _complete_allocations(self):
        if self._allocated_names != set(self._allocation_entries):
            raise RuntimeError("TYX constructor omitted manifest allocations")
        self.resources.finish_reservation()

    def _close_after_construction_failure(self, primary):
        try:
            self.close()
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "TYX construction and cleanup failed", [primary, cleanup]
            ) from None

    def allocation_snapshot(self):
        budget = None if self._allocation_budget is None else self._allocation_budget.snapshot()
        identity = (
            None if self._allocation_reservation is None else self._allocation_reservation.owner_id
        )
        owner = (
            None
            if identity is None
            else next(row for row in budget["owners"] if row["owner_id"] == identity)
        )
        return MappingProxyType(
            dict(
                schema="mujoco-tyx-device-allocation-v1",
                scope=(
                    "registered-musa-array-payload-only"
                    if self.mode == "musa"
                    else "registered-cuda-array-payload-only"
                ),
                mode=self.mode,
                manifest=self.allocation_manifest,
                manifest_payload_bytes=self.allocation_manifest.payload_bytes,
                allocated_names=tuple(
                    entry.name
                    for entry in self.allocation_manifest.entries
                    if entry.name in self._allocated_names
                ),
                owner_id=identity,
                owner=owner,
                current_bytes=0 if owner is None else owner["current_bytes"],
                reserved_bytes=0 if owner is None else owner["reserved_bytes"],
                unknown_retained_bytes=0 if owner is None else owner["unknown_retained_bytes"],
                shared_budget=self._shared_resource_budget,
                budget=budget,
                resources_state=self.resources.state,
                excluded=DEVICE_PAYLOAD_EXCLUSIONS,
                host_live_bytes=None,
                host_peak_bytes=None,
                total_device_live_bytes=None,
                total_device_peak_bytes=None,
            )
        )

    def _buffer(self, value):
        if self._allocation_entry is None:
            raise RuntimeError("TYX buffers require a declared allocation")
        if self.mode == "interpret":
            return value.copy()
        result = self.runtime.from_numpy(value) if self.mode == "musa" else self.tx.cuda_array(value)
        self.resources.add_buffer(result)
        entry = self._allocation_entry
        if (
            tuple(result.shape) != entry.shape
            or np.dtype(result.dtype) != np.dtype(entry.dtype)
            or result.nbytes != entry.nbytes
        ):
            raise RuntimeError(f"{self.mode.upper()} allocation differs from its manifest")
        self.counts["device_allocations"] += 1
        self.counts["static_uploaded_bytes"] += value.nbytes
        return result

    def _put(self, buffer, value):
        if self.mode == "interpret":
            np.copyto(buffer, value)
        elif self.mode == "cuda" and isinstance(value, self.tx.CudaArray):
            buffer.copy_from_device(value)
            self.counts["device_control_d2d_bytes"] = (
                self.counts.get("device_control_d2d_bytes", 0) + value.nbytes
            )
        else:
            buffer.copy_from(value)
            self.counts["uploaded_bytes"] += value.nbytes
            self.counts["control_host_uploaded_bytes"] = (
                self.counts.get("control_host_uploaded_bytes", 0) + value.nbytes
            )

    def _get(self, buffer):
        if self.mode == "interpret":
            return buffer.copy()
        result = buffer.to_numpy()
        self.counts["downloaded_bytes"] += result.nbytes
        return result

    def _open_runtime(self, cache_root):
        if self.mode == "musa":
            from ..tyx_musa import MusaRuntime

            self.runtime = MusaRuntime()
        else:
            self.runtime = self.tx.Runtime(
                backend="cuda" if self.mode == "cuda" else "cpu", cache_root=cache_root
            )
        self.resources.set_runtime(self.runtime)
        if self.mode == "musa":
            self._musa_arch = self.runtime.report()["arch"]
            self._musa_cache_root = cache_root
            self._musa_kernels = {}

    def _musa_kernel(self, kernel):
        from ..tyx_musa.compiler import MusaKernel

        # TYX decorates the same function twice for its ping-pong bank handles.
        # Its frontend fingerprint includes the typed body and inline dependencies.
        key = (kernel.name, kernel.cache_fingerprint, kernel.gpu_schedule)
        if key not in self._musa_kernels:
            self._musa_kernels[key] = MusaKernel(
                kernel, cache_root=self._musa_cache_root, arch=self._musa_arch
            )
        return self._musa_kernels[key]

    def runtime_scope(self):
        # MUSA owns an explicit stream. TYX interpret and CUDA use the current runtime.
        return nullcontext() if self.mode == "musa" else self.tx.use_runtime(self.runtime)

    def launch_kernel(self, kernel, *args):
        if self.mode == "musa":
            self.runtime.launch(self._musa_kernel(kernel), *args)
        else:
            kernel(*args, mode=self.mode)

    def advance(self, controls, applied, body_force):
        plan = self.plan
        old, new = self.active, 1 - self.active
        with self.runtime_scope():
            self._put(self.controls, controls)
            self._put(self.applied, applied)
            self._put(self.body_force, body_force)
            self.launch_kernel(
                self.kernels[old],
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
            )
            self.counts["completed_kernels"] += 1
            self.runtime.sync()
            self.counts["synchronizations"] += 1

    def download(self):
        target = 1 - self.active
        with self.runtime_scope():
            return {
                name: self._get(value)
                for name, value in (
                    ("q", self.q[target]),
                    ("v", self.v[target]),
                    ("time", self.time[target]),
                    ("input_time", self.time[self.active]),
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
