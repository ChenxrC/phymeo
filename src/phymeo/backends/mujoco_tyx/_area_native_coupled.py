"""Bounded HOST control of resident area/native load evaluations.

Only status and convergence metadata cross to HOST inside the outer loop.
Geometry, M/J, native laws, all forces and candidate velocities stay in the
existing device owner. The caller integrates and publishes once afterward.
"""

import numpy as np


def solve_contact(device, old):
    from . import _area_native_coupled_kernels as kernels
    from ._area_friction_device import pack_contact

    pack_contact(device, old)

    def out(name):
        return getattr(device, "friction_" + name)

    packed_width = out("mass").shape[1]
    kernels.initialize(
        device.nworld,
        device.plan.nv,
        device.width,
        packed_width,
        out("samples").shape[1],
        device.capacities[1],
        device.h,
        out("pack_status"),
        out("sizes"),
        device.constraint_counts,
        device.mass,
        out("inputs"),
        device.applied,
        device.passive,
        device.bias,
        out("matrices"),
        out("vectors"),
        out("contact"),
        out("meta"),
        out("scalars"),
        out("trace_meta"),
        out("trace"),
        device.area_native_applied,
        device.area_native_meta,
        device.area_native_trace_meta,
        device.area_native_trace,
        device.area_native_publication,
        device.status,
        mode=device.mode,
    )
    device.counts["completed_kernels"] += 1
    for iteration in range(25):
        # Every call is the existing cold physical-M working set. The first
        # receives the original applied load; later calls add a frozen trial Q.
        # The original applied/passive/bias and native J/R/aref remain inputs.
        device._solve_rows(device.mass, device.area_native_applied)
        kernels.evaluate(
            device.nworld,
            device.plan.nv,
            packed_width,
            device.h,
            iteration,
            out("sizes"),
            out("inputs"),
            out("samples"),
            out("ja"),
            out("jb"),
            out("matrices"),
            out("vectors"),
            out("contact"),
            device.applied,
            device.passive,
            device.bias,
            device.total_applied,
            device.solver_diagnostics,
            device.area_native_applied,
            device.area_native_meta,
            device.area_native_trace_meta,
            device.area_native_trace,
            device.status,
            mode=device.mode,
        )
        device.counts["completed_kernels"] += 1
        kernels.qualify_publication(
            device.nworld,
            device.plan.nv,
            packed_width,
            iteration,
            device.plan.solver_tolerance,
            device.constraint_counts,
            device.physical_factor,
            device.mixed_constraint,
            device.efc_data,
            device.efc_jacobian,
            device.applied,
            device.passive,
            device.bias,
            out("vectors"),
            device.area_native_applied,
            device.area_native_meta,
            device.area_native_trace_meta,
            device.area_native_trace,
            device.area_native_publication,
            device.status,
            mode=device.mode,
        )
        device.counts["completed_kernels"] += 1
        device.runtime.sync()
        device.counts["synchronizations"] += 1
        status = device._get(device.status)
        metadata = device._get(device.area_native_meta)
        if np.any(status) or np.all(metadata[:, 3] == 1):
            break
    kernels.finish(
        device.nworld,
        device.plan.nv,
        device.width,
        device.h,
        out("inputs"),
        out("matrices"),
        out("vectors"),
        device.v[old],
        device.applied,
        device.passive,
        device.bias,
        device.mixed_constraint,
        device.total_applied,
        device.smooth_acc,
        device.v[1 - old],
        device.area_native_meta,
        device.status,
        mode=device.mode,
    )
    device.counts["completed_kernels"] += 1
