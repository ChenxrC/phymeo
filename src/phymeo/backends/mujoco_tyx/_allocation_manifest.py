"""Single immutable shape/source table for all existing TYX leaf allocations."""

import sys
from dataclasses import dataclass
from math import prod

import numpy as np


@dataclass(frozen=True, slots=True)
class AllocationEntry:
    name: str
    shape: tuple[int, ...]
    dtype: str
    role: str
    source: str | None = None

    @property
    def nbytes(self):
        return prod(self.shape) * np.dtype(self.dtype).itemsize


@dataclass(frozen=True, slots=True)
class AllocationManifest:
    entries: tuple[AllocationEntry, ...]
    aliases: tuple[tuple[str, str], ...]
    schema: str = "mujoco-tyx-allocation-manifest-v1"

    @property
    def payload_bytes(self):
        return sum(entry.nbytes for entry in self.entries)


def allocation_manifest(plan, nworld, nbody, capacities):
    """Describe the final route once, including its base, padding and extensions.

    This builds metadata only: no zeros, device context or CUDA allocation.
    Source paths name original plan arrays; input:* requires the actual packed
    input. A source-less entry is lazily zero-filled by the owned device.
    """
    if type(nworld) is not int or nworld < 1 or type(nbody) is not int or nbody < 1:
        raise ValueError("manifest requires positive builtin world/body counts")
    if (
        type(capacities) is not tuple
        or len(capacities) != 2
        or any(type(cap) is not int or cap < 0 for cap in capacities)
    ):
        raise ValueError("manifest requires two nonnegative builtin capacities")
    entries, aliases = [], []

    def add(name, shape, dtype="float32", role="scratch", source=None):
        shape = tuple(int(dim) for dim in shape)
        if any(dim < 0 for dim in shape):
            raise ValueError("negative allocation dimension")
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype("float32"), np.dtype("int32")):
            raise ValueError("manifest supports only original FP32/i32 buffers")
        entry = AllocationEntry(name, shape, dtype.str, role, source)
        if entry.nbytes > sys.maxsize:
            raise ValueError("allocation payload exceeds addressable size")
        if any(item.name == name for item in entries):
            raise ValueError("duplicate manifest allocation name")
        entries.append(entry)

    def static(name, source=None):
        key = source or name
        value = plan
        for part in key.split("."):
            value = getattr(value, part)
        if not isinstance(value, np.ndarray):
            raise TypeError("manifest static source must be an ndarray")
        add(name, value.shape, value.dtype, "static", key)

    kind, w = int(plan.kind), nworld
    tree = kind in (2, 3)
    normal = bool(getattr(plan, "normal_contact", False))
    constraint = bool(getattr(plan, "constraint_contact", False))
    if normal and constraint:
        raise ValueError("one leaf cannot select both contact allocation routes")
    width = max(8, int(plan.nv)) if kind == 3 else 8
    if tree:
        for name, source in (
            ("parents", "parents"),
            ("ranges", "body_joint_range"),
            ("body_data", "body_data"),
            ("joints", "joint_ids"),
            ("dofs", "dof_ids"),
            ("joint_data", "joint_data"),
            ("ancestors", "ancestors"),
            ("gravity", "gravity"),
            ("vectors", "vectors"),
            ("act_ids", "actuator_ids"),
            ("act_values", "actuator_values"),
        ):
            static(name, source)
    else:
        for name, source in (
            ("matrices", "matrices"),
            ("vectors", "vectors"),
            ("body_jac", "body_jacobian"),
            ("point_jac", "point_jacobian"),
            ("law", "contact"),
            ("act_ids", "actuator_ids"),
            ("act_values", "actuator_values"),
        ):
            static(name, source)
    for bank in (0, 1):
        for name, size in (("q", max(8, int(plan.nq))), ("v", max(8, int(plan.nv))), ("time", 2)):
            add(f"{name}.{bank}", (w, size), role=f"state-bank-{bank}", source=f"input:{name}")
    if hasattr(plan, "activation"):
        if not tree:
            raise ValueError("activation allocation requires an original tree base")
        for bank in (0, 1):
            add(
                f"act.{bank}",
                (w, int(plan.activation.na)),
                role=f"state-bank-{bank}",
                source="input:act",
            )
        static("activation_ids", "activation.ids")
        static("activation_values", "activation.values")
        for name in ("activation_derivative", "integration_derivative"):
            add(name, (w, int(plan.activation.na)))
        add("actuator_fields", (w, int(plan.nu), 4))
    add("controls", (w, max(int(plan.nu), 1)), role="control")
    add("applied_input" if tree else "applied", (w, width), role="control")
    add("body_force", (w, nbody, 6), role="control")
    if tree:
        if getattr(plan, "joint_transmission", None) is not None:
            for name, source in (
                ("transmission_offsets", "joint_transmission.row_offsets"),
                ("transmission_dofs", "joint_transmission.dof_indices"),
                ("transmission_moment", "joint_transmission.moment"),
            ):
                static(name, source)
            add("transmission_fields", (w, int(plan.nu), 3))
        else:
            aliases.extend(
                (
                    ("transmission_offsets", "parents"),
                    ("transmission_dofs", "parents"),
                    ("transmission_moment", "gravity"),
                    ("transmission_fields", "body_force"),
                )
            )
        for name, shape in (
            ("motion", (w, nbody, 7, 3)),
            ("rotation", (w, nbody, 3, 3)),
            ("axes", (w, width, 3)),
            ("anchors", (w, width, 3)),
            ("work", (w, 8, 3)),
            ("work_rotation", (w, 3, 3)),
            ("inertia", (w, nbody, 3, 3)),
            ("jacobian", (w, nbody, 6, width)),
            ("mass", (w, width, width)),
            ("factor", (w, width, width)),
        ):
            add(name, shape)
        for name in (
            "bias",
            "passive",
            "applied",
            "actuator",
            "constraint",
            "rhs",
            "forward",
            "acceleration",
            "integration_acc",
        ):
            add(name, (w, width))
    else:
        for name in (
            "force",
            "smooth_acc",
            "acceleration",
            "integration_acc",
            "actuator",
            "constraint",
        ):
            add(name, (w, width))
    add("rows", (w, 20))
    add("status", (w,), "int32")
    if normal:
        for name in ("geom_ids", "geom_data", "pair_ids", "pair_law"):
            static(name)
        s, g = len(plan.slot_ids), len(plan.geom_ids)
        for name, shape in (
            ("geom_world", (w, g, 12)),
            ("normal_rows", (w, s, 32)),
            ("normal_jacobian", (w, s, width)),
            ("contact_work", (w, 8, 3)),
            ("raw_contacts", (w, s, 11)),
            ("extended_work", (w, 64, 3)),
            ("solver_diagnostics", (w, 8)),
            ("normal_factor", (w, width, width)),
            ("normal_mass_work", (w, width)),
            ("smooth_acc", (w, width)),
            ("response", (w, s, width)),
            ("dual_matrix", (w, s, s)),
            ("normal_linear", (w, s)),
            ("normal_force", (w, s)),
            ("normal_gradient", (w, s)),
            ("normal_working", (w, s)),
            ("face_factor", (w, s, s)),
            ("face_forward", (w, s)),
            ("normal_trial", (w, s)),
            ("normal_constraint", (w, width)),
            ("total_applied", (w, width)),
        ):
            add(name, shape, "int32" if name == "normal_working" else "float32")
    if constraint:
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
            static(name)
        s = max(1, int(plan.nslot))
        c, e = (max(1, cap) for cap in capacities)
        for name, shape in (
            ("geom_world", (w, max(1, int(plan.ngeom)), 12)),
            ("normal_rows", (w, s, 32)),
            ("normal_jacobian", (w, s, width)),
            ("contact_work", (w, 8, 3)),
            ("raw_contacts", (w, s, 11)),
            ("extended_work", (w, 64, 3)),
            ("geometry_diagnostics", (w, 8)),
            ("solver_diagnostics", (w, 10)),
            ("efc_data", (w, e, 14)),
            ("efc_jacobian", (w, e, width)),
            ("contact_jacobian", (w, c, 6, width)),
            ("contact_wrench", (w, c, 6)),
            ("physical_factor", (w, width, width)),
            ("mass_work", (w, width)),
            ("smooth_acc", (w, width)),
            ("face_matrix", (w, width, width)),
            ("face_factor", (w, width, width)),
            ("face_rhs", (w, width)),
            ("face_forward", (w, width)),
            ("trial_acc", (w, width)),
            ("trial_force", (w, e)),
            ("previous_force", (w, e)),
            ("mixed_constraint", (w, width)),
            ("total_applied", (w, width)),
            ("physical_acc", (w, width)),
        ):
            add(name, shape)
        for name, shape in (
            ("efc_ids", (w, e, 5)),
            ("contact_map", (w, c, 3)),
            ("constraint_counts", (w, 3)),
            ("working", (w, e)),
            ("previous_working", (w, e)),
            ("free_ids", (w, width)),
        ):
            add(name, shape, "int32")
    from ._entry_storage import entry_layout, entry_static_layout

    for name, (shape, dtype) in entry_layout(plan, w, capacities).items():
        for bank in (0, 1):
            add(f"{name}.{bank}", shape, dtype, f"accepted-entry-bank-{bank}")
    for name, (shape, dtype) in entry_static_layout(plan).items():
        add(name, shape, dtype, "accepted-entry-static")
    manifest = AllocationManifest(tuple(entries), tuple(aliases))
    if manifest.payload_bytes > sys.maxsize:
        raise ValueError("leaf allocation payload exceeds addressable size")
    return manifest
