"""Connect the existing resident pack/solver to one native integration.

No independent runtime, HOST M/J repack, state publication, or q integration
lives here. The caller still owns the original inactive physical/entry bank.
"""

from ._area_friction_layout import FRICTION_FIELDS


def solve_contact(device, old):
    from .._tyx_area_friction_entry import pack_entry
    from .._tyx_area_friction_kernels import solve_batch
    from ._area_friction_live_kernels import apply_load

    packet = device.area_packet
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
        mode=device.mode,
    )
    device.counts["completed_kernels"] += 1
    solve_batch(
        device.nworld,
        device.nworld,
        *(out(name) for name in FRICTION_FIELDS),
        require_velocity_agreement=True,
        guide=device.efc_data if device.plan.nfriction else None,
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
    with device.tx.use_runtime(device.runtime):
        for name in FRICTION_FIELDS + ("pack_status",):
            result["friction_" + name] = device._get(getattr(device, "friction_" + name))
        if device.plan.nfriction:
            for name in ("efc_ids", "efc_data", "efc_jacobian", "constraint_counts"):
                result[name] = device._get(getattr(device, name))
        result["constraint"] = device._get(device.mixed_constraint)
        result["smooth_acceleration"] = device._get(device.smooth_acc)
    return result
