"""Connect the existing resident pack/solver to one native integration.

No independent runtime, HOST M/J repack, state publication, or q integration
lives here. The caller still owns the original inactive physical/entry bank.
"""

from ._area_friction_layout import FRICTION_FIELDS


def pack_contact(device, old):
    """Pack the same frozen entry for either admitted live friction solver."""
    from .._tyx_area_friction_entry import pack_entry

    packet = device.area_packet
    sample_capacity = device.friction_samples.shape[1]
    for name, value in (
        ("area_input", packet[0]),
        ("area_pair_index", packet[1]),
        ("area_sizes", packet[2]),
        ("area_measure_input", packet[4]),
    ):
        device._put(getattr(device, name), value)

    def out(name):
        return getattr(device, "friction_" + name)

    pack_entry(
        device.nworld,
        device.nworld,
        device.plan.nv,
        device.width,
        device.nbody,
        len(device.plan.area_pair_names),
        max(1, device.capacities[1]),
        device.h,
        device.status,
        device.mass,
        device.vectors,
        device.v[old],
        device.bias,
        device.passive,
        device.applied,
        device.motion,
        device.jacobian,
        device.area_input,
        device.area_pair_index,
        device.area_sizes,
        device.area_body_ids,
        device.area_friction_parameters,
        *(out(name) for name in FRICTION_FIELDS[:7]),
        out("pack_status"),
        sample_capacity=sample_capacity,
        mode=device.mode,
    )
    device.counts["completed_kernels"] += 1


def solve_contact(device, old):
    from .._tyx_area_friction_kernels import solve_batch
    from ._area_friction_live_kernels import apply_load

    pack_contact(device, old)
    sample_capacity = device.friction_samples.shape[1]

    def out(name):
        return getattr(device, "friction_" + name)

    coupled_limit = device.plan.coupled_damped_contacts and device.plan.nlimit > 0
    coupled_supports = device.plan.two_native_supports
    native_mode = device.plan.damped_contact_dofs
    if coupled_limit:
        native_mode = 3
    elif coupled_supports:
        native_mode = 4
    solve_batch(
        device.nworld,
        device.nworld,
        *(out(name) for name in FRICTION_FIELDS),
        require_velocity_agreement=True,
        sample_capacity=sample_capacity,
        native_rows=(device.efc_data, device.efc_jacobian, device.constraint_counts)
        if device.plan.nfriction or device.plan.nlimit or device.plan.damped_contact_dofs
        else None,
        native_source_forces=(device.applied, device.passive, device.bias)
        if coupled_limit or coupled_supports
        else None,
        native_physical_acceleration=native_mode,
        native_limit_count=device.plan.nlimit if coupled_limit else 1,
        mode=device.mode,
    )
    device.counts["completed_kernels"] += 1
    apply_load(
        device.nworld,
        device.plan.nv,
        device.width,
        device.h,
        out("pack_status"),
        out("meta"),
        out("vectors"),
        device.v[old],
        device.applied,
        device.mixed_constraint,
        device.total_applied,
        device.smooth_acc,
        device.v[1 - old],
        int(coupled_supports),
        device.status,
        mode=device.mode,
    )
    device.counts["completed_kernels"] += 1


def stage_entry(device, new):
    from ._area_friction_live_kernels import stage

    stage(
        device.nworld,
        device.plan.nv,
        max(1, device.capacities[1]),
        len(device.plan.area_pair_names),
        device.area_sizes,
        device.area_pair_index,
        device.area_body_ids,
        device.area_measure_input,
        device.friction_samples,
        device.friction_ja,
        device.friction_jb,
        device.friction_contact,
        device.v[new],
        device.entry_meta[new],
        device.area_meta[new],
        device.area_ids[new],
        device.area_data[new],
        device.area_measure_words[new],
        device.status,
        mode=device.mode,
    )
    device.counts["completed_kernels"] += 1


def download(device, result):
    resident_native = bool(getattr(device.plan, "resident_native_contacts", False))
    with device.tx.use_runtime(device.runtime):
        for name in FRICTION_FIELDS + ("pack_status",):
            result["friction_" + name] = device._get(getattr(device, "friction_" + name))
        if (
            device.plan.nfriction
            or device.plan.nlimit
            or device.plan.damped_contact_dofs
            or resident_native
        ):
            for name in ("efc_ids", "efc_data", "efc_jacobian", "constraint_counts"):
                result[name] = device._get(getattr(device, name))
        if device.plan.damped_contact_dofs or resident_native:
            for name in ("normal_rows", "normal_jacobian", "contact_map", "contact_wrench"):
                result[name] = device._get(getattr(device, name))
            result["contact_J"] = device._get(device.contact_jacobian)
        if device.plan.two_native_supports or resident_native:
            result["applied_with_contact"] = device._get(device.total_applied)
        if resident_native:
            for name in (
                "area_native_applied",
                "area_native_meta",
                "area_native_trace_meta",
                "area_native_trace",
                "area_native_publication",
                "solver_diagnostics",
                "working",
            ):
                result["working_set" if name == "working" else name] = device._get(
                    getattr(device, name)
                )
            result["efc_J"] = result["efc_jacobian"]
            result["counts"] = result["constraint_counts"]
            # The common native tail recomputes physical M acceleration from
            # the final published loads; the last trial's a is not a receipt.
            result["qacc"] = device._get(device.acceleration)
        result["constraint"] = device._get(device.mixed_constraint)
        result["smooth_acceleration"] = device._get(device.smooth_acc)
    return result
