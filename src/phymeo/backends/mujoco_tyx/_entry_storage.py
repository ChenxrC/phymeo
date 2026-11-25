"""Persistent accepted-entry storage owned by the existing physical device.

The seven ABI fields have two independent banks, indexed by device.active.
Preparation writes only the inactive bank. The runtime owns synchronization,
validation and the single physics/entry publication; this module never publishes.
"""

import numpy as np

ENTRY_FIELDS = (
    "entry_q",
    "entry_v",
    "entry_act",
    "entry_clock",
    "entry_meta",
    "entry_contact_ids",
    "entry_contact_data",
)


def _route(plan):
    if getattr(plan, "constraint_contact", False):
        return "constraint"
    if getattr(plan, "normal_contact", False):
        return "normal"
    return "legacy" if plan.pair is not None else "none"


def _na(plan):
    return int(plan.activation.na) if hasattr(plan, "activation") else 0


def entry_layout(plan, nworld, capacities):
    """Return single-bank shape/dtype metadata without allocating any arrays."""
    if type(nworld) is not int or nworld < 1:
        raise ValueError("entry storage requires a positive builtin world count")
    if (
        type(capacities) is not tuple
        or len(capacities) != 2
        or any(type(cap) is not int or cap < 0 for cap in capacities)
    ):
        raise ValueError("entry storage requires two nonnegative builtin capacities")
    cap = max(1, capacities[0])
    f32, i32 = np.dtype("float32"), np.dtype("int32")
    layout = {
        "entry_q": ((nworld, max(8, int(plan.nq))), f32),
        "entry_v": ((nworld, max(8, int(plan.nv))), f32),
        "entry_act": ((nworld, max(1, _na(plan))), f32),
        "entry_clock": ((nworld, 4), f32),
        "entry_meta": ((nworld, 6), i32),
        "entry_contact_ids": ((nworld, cap, 8), i32),
        "entry_contact_data": ((nworld, cap, 20), f32),
    }
    from ..mujoco_area_pressure_device import area_layout

    layout.update(area_layout(plan, nworld, capacities))
    return layout


def entry_static_layout(plan):
    """Describe only extra static arrays; safe during budget preflight."""
    route = _route(plan)
    if route == "normal":
        return {"entry_slot_ids": (plan.slot_ids.shape, np.dtype("int32"))}
    if route == "legacy":
        return {
            "entry_legacy_ids": ((4,), np.dtype("int32")),
            "entry_legacy_frame": ((3, 3), np.dtype("float32")),
        }
    return {}


def entry_static(plan, body_ids=None):
    """Supply constructor-only values, preserving the producer's geom order.

    A legacy pair's two real model.geom_bodyid entries are required. The plane
    may belong to a fixed child body rather than body zero. Neither geom sorting
    nor plan.body can recover this mapping. T4 reuses its existing static arrays.
    """
    route = _route(plan)
    if route == "normal":
        return {"entry_slot_ids": plan.slot_ids}
    if route == "legacy":
        bodies = np.asarray(body_ids)
        if (
            bodies.shape != (2,)
            or bodies.dtype.kind not in "iu"
            or np.any(bodies < 0)
            or np.any(bodies > np.iinfo(np.int32).max)
        ):
            raise ValueError("legacy entry requires two real nonnegative body IDs")
        return {
            "entry_legacy_ids": np.asarray((*plan.pair, *bodies), dtype=np.int32),
            "entry_legacy_frame": plan.frame,
        }
    return {}


def allocate_entry(device, body_ids=None):
    """Allocate every entry through the device's existing manifest and ledger.

    Base constructors call this before their completion check or extension
    allocation loop. Route selection depends only on plan, not on extension
    attributes which may not have been allocated yet.
    """
    static = entry_static(device.plan, body_ids)
    if "entry_legacy_ids" in static and np.any(static["entry_legacy_ids"][2:] >= device.nbody):
        raise ValueError("legacy entry body ID exceeds the model body count")
    device.entry = {}
    for name in entry_layout(device.plan, device.nworld, device.capacities):
        banks = tuple(device._allocate(f"{name}.{bank}") for bank in (0, 1))
        setattr(device, name, banks)
        device.entry[name] = banks
    for name, value in static.items():
        setattr(device, name, device._allocate(name, value))


def prepare_entry(device):
    """Stage accepted-input native state and this candidate's real contacts.

    Call after advance and before download, while active still selects the old
    accepted physics state. No HOST state/receipt is uploaded, no synchronization
    or publication occurs, and a prior solver error is never cleared. The caller
    seeds entry_meta[:, 2] when restoring historical state without an entry.
    """
    from . import _entry_kernels as kernels

    old, new = device.active, 1 - device.active
    plan, cap = device.plan, device.capacities[0]
    na = _na(plan)
    with device.runtime_scope():
        device.launch_kernel(
            device.musa_entry_state if device.mode == "musa" else kernels.state,
            device.nworld,
            max(8, int(plan.nq)),
            max(8, int(plan.nv)),
            na,
            max(1, cap),
            device.q[old],
            device.v[old],
            device.act[old] if na else device.q[old],
            device.time[old],
            device.time[new],
            device.entry_meta[old],
            *(getattr(device, name)[new] for name in ENTRY_FIELDS),
            device.status,
        )
        device.counts["completed_kernels"] += 1
        if getattr(plan, "area_friction_laws", ()):
            from ._area_friction_device import stage_entry

            if getattr(plan, "damped_contact_dofs", 0) or getattr(
                plan, "resident_native_contacts", False
            ):
                # Preserve native geometry in its existing seven-field ABI.
                # Area staging follows in the same inactive bank and does not
                # synthesize or clear these native contact records.
                kernels.constraint(
                    device.nworld,
                    int(plan.nslot),
                    int(plan.ngeom),
                    int(plan.npair),
                    cap,
                    device.slot_ids,
                    device.geom_ids,
                    device.pair_condim,
                    device.normal_rows,
                    device.constraint_counts,
                    device.contact_map,
                    device.contact_wrench,
                    device.entry_meta[new],
                    device.entry_contact_ids[new],
                    device.entry_contact_data[new],
                    device.status,
                    mode=device.mode,
                )
                device.counts["completed_kernels"] += 1
            stage_entry(device, new)
            return new
        route = _route(plan)
        outputs = (
            device.entry_meta[new],
            device.entry_contact_ids[new],
            device.entry_contact_data[new],
            device.status,
        )
        if route == "legacy":
            kernels.legacy(
                device.nworld,
                cap,
                device.rows,
                device.law,
                device.entry_legacy_ids,
                device.entry_legacy_frame,
                *outputs,
                mode=device.mode,
            )
        elif route == "normal":
            kernels.normal(
                device.nworld,
                len(plan.slot_ids),
                len(plan.geom_ids),
                cap,
                device.entry_slot_ids,
                device.geom_ids,
                device.normal_rows,
                *outputs,
                mode=device.mode,
            )
        elif route == "constraint":
            kernels.constraint(
                device.nworld,
                int(plan.nslot),
                int(plan.ngeom),
                int(plan.npair),
                cap,
                device.slot_ids,
                device.geom_ids,
                device.pair_condim,
                device.normal_rows,
                device.constraint_counts,
                device.contact_map,
                device.contact_wrench,
                *outputs,
                mode=device.mode,
            )
        if route != "none":
            device.counts["completed_kernels"] += 1
        if hasattr(plan, "area_pair_names"):
            from ._area_entry_kernels import stage

            stage(
                device.nworld,
                max(1, device.capacities[1]),
                len(plan.area_pair_names),
                device.area_sizes,
                device.area_pair_index,
                device.area_body_ids,
                device.area_input,
                device.area_measure_input,
                device.constraint_counts,
                device.efc_ids,
                device.efc_data,
                device.entry_meta[new],
                device.area_meta[new],
                device.area_ids[new],
                device.area_data[new],
                device.area_measure_words[new],
                device.status,
                mode=device.mode,
            )
            device.counts["completed_kernels"] += 1
    return new
