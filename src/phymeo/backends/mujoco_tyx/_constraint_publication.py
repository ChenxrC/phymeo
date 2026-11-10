"""Immutable mixed-constraint receipts and checked host publication.

Only validation and coordinate presentation occur here; dynamics and the
coupled solve are supplied by the device transaction.
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..mujoco_runtime import _frozen


@dataclass(frozen=True)
class MuJoCoTyxConstraintRow:
    schema: str
    constraint_type: int
    source_id: int
    side_or_edge: int
    geometry_slot: int
    dof: int
    jacobian: np.ndarray
    position: float
    margin: float
    velocity: float
    reference_acceleration: float
    regularization: float
    inverse_regularization: float
    impedance: float
    diag_approx: float
    lower: float
    upper: float | None
    force: float
    gradient: float


DIAGNOSTICS = (
    "face_solves",
    "normalized_kkt",
    "absolute_kkt",
    "bound_violation",
    "actual_efc",
    "free_rows",
    "converged",
    "budget",
    "normalized_mass_residual",
    "kkt_scale",
)


def publish(runtime, result, index):
    from .._mujoco_tyx_runtime import MuJoCoTyxNormalContact

    plan = runtime._plan
    capc, cape = runtime._contact_capacity, runtime._constraint_capacity
    csize, esize, width = max(1, capc), max(1, cape), max(8, plan.nv)
    expected = {
        "efc_ids": (esize, 5),
        "efc_data": (esize, 14),
        "efc_J": (esize, width),
        "counts": (3,),
        "working_set": (esize,),
        "solver_diagnostics": (10,),
        "contact_map": (csize, 3),
        "contact_J": (csize, 6, width),
        "contact_wrench": (csize, 6),
        "normal_rows": (max(1, plan.nslot), 32),
        "normal_jacobian": (max(1, plan.nslot), width),
    }
    fields = {name: result[name][index] for name in expected}
    for name, shape in expected.items():
        if fields[name].shape != shape or not np.all(np.isfinite(fields[name])):
            raise RuntimeError(f"invalid mixed constraint field: {name}")
    ncontact, nactive, nefc = map(int, fields["counts"])
    if not 0 <= nactive <= ncontact <= capc or not 0 <= nefc <= cape:
        raise RuntimeError("invalid mixed capacity counts")
    diag = fields["solver_diagnostics"]
    if (
        diag[6] != 1
        or diag[4] != nefc
        or diag[7] != plan.solver_iterations
        or not 0 <= diag[0] <= plan.solver_iterations
        or diag[0] != int(diag[0])
        or not 0 <= diag[5] <= nefc
        or diag[5] != int(diag[5])
        or np.any(diag < 0)
        or diag[9] < 1
        or diag[1] > plan.solver_tolerance
        or diag[8] > plan.solver_tolerance
        or diag[3] > plan.solver_tolerance
    ):
        raise RuntimeError("invalid mixed convergence receipt")
    ids, data, jac = fields["efc_ids"], fields["efc_data"], fields["efc_J"]
    normal, cmap = fields["normal_rows"], fields["contact_map"]
    if np.any(~np.isin(normal[: plan.nslot, :2], (0, 1))) or np.any(normal[:, 1] > normal[:, 0]):
        raise RuntimeError("invalid mixed geometry flags")
    detected = np.flatnonzero(normal[: plan.nslot, 0])
    if len(detected) != ncontact or int(np.sum(normal[: plan.nslot, 1])) != nactive:
        raise RuntimeError("mixed geometry count mismatch")
    if np.any(~np.isin(fields["working_set"][:nefc], (0, 1, 2))):
        raise RuntimeError("invalid mixed working set")
    expected_ids = []
    for dof in plan.friction_ids[: plan.nfriction]:
        expected_ids.append((1, int(dof), -1, -1, int(dof)))
    input_q = result["input_q"][index]
    for (joint, qa, dof), law in zip(
        plan.limit_ids[: plan.nlimit], plan.limit_law[: plan.nlimit], strict=True
    ):
        for side, distance in ((-1, input_q[qa] - law[0]), (1, law[1] - input_q[qa])):
            if distance < law[2]:
                expected_ids.append((3, int(joint), side, -1, int(dof)))
    contacts = []
    for ci, slot in enumerate(detected):
        pair, feature, geom1, geom2 = map(int, plan.slot_ids[slot])
        dim = int(plan.pair_condim[pair])
        count = (1 if dim == 1 else 2 * (dim - 1)) if normal[slot, 1] else 0
        first = len(expected_ids) if count else -1
        if tuple(cmap[ci]) != (slot, first, count):
            raise RuntimeError("mixed contact map does not cover exact active rows")
        expected_ids.extend(
            (5 if dim == 1 else 6, ci, edge, int(slot), -1) for edge in range(count)
        )
        row, law = normal[slot], plan.pair_law[pair]
        frame, point = row[12:21].reshape(3, 3), row[9:12]
        wrench = fields["contact_wrench"][ci]
        if (not count and np.any(wrench)) or wrench[0] < 0:
            raise RuntimeError("invalid decoded contact force")
        force, couple = (
            frame.T.astype(np.float64) @ wrench[:3],
            frame.T.astype(np.float64) @ wrench[3:],
        )
        body1, body2 = map(int, plan.geom_ids[[geom1, geom2], 0])
        com1, com2 = result["body_com"][index, [body1, body2]]
        contacts.append(
            MuJoCoTyxNormalContact(
                schema="mujoco-tyx-pyramidal-contact-v1",
                geometry_ids=(geom1, geom2),
                body_ids=(body1, body2),
                feature_index=feature,
                dimension=dim,
                distance=float(row[2]),
                position=_frozen(point),
                frame=_frozen(frame),
                include_margin=float(row[21]),
                friction=_frozen(plan.pair_friction[pair]),
                solref=_frozen(law[3:5]),
                solimp=_frozen(law[5:10]),
                active=bool(count),
                wrench_contact_frame=_frozen(wrench),
                force_on_geometry1_world=_frozen(-force),
                force_on_geometry2_world=_frozen(force),
                wrench_on_body1_world=_frozen(
                    np.r_[-force, -couple + np.cross(point - com1, -force)]
                ),
                wrench_on_body2_world=_frozen(np.r_[force, couple + np.cross(point - com2, force)]),
                impedance=float(data[first, 6]) if count else 0.0,
                regularization=float(data[first, 4]) if count else 0.0,
                reference_acceleration=float(data[first, 3]) if count else 0.0,
                normal_velocity=float(row[7]),
                jacobian=_frozen(fields["contact_J"][ci, :, : plan.nv]),
                normal_force=float(wrench[0]),
            )
        )
    if len(expected_ids) != nefc or not np.array_equal(
        ids[:nefc], np.asarray(expected_ids, dtype=np.int32).reshape(-1, 5)
    ):
        raise RuntimeError("mixed rows omit, reorder or replace source constraints")
    constraints = []
    for i in range(nefc):
        row = data[i]
        if (
            row[11] != 1
            or row[12] not in (0, 1)
            or row[4] <= 0
            or row[5] <= 0
            or not 0 < row[6] < 1
            or row[13] < 0
            or row[9] < row[7]
            or (row[12] and (row[8] < row[7] or row[9] > row[8]))
            or (not row[12] and row[8] != 0)
        ):
            raise RuntimeError("invalid mixed force bounds or regularization")
        constraints.append(
            MuJoCoTyxConstraintRow(
                "mujoco-tyx-constraint-row-v1",
                *map(int, ids[i]),
                _frozen(jac[i, : plan.nv]),
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                float(row[6]),
                float(row[13]),
                float(row[7]),
                float(row[8]) if row[12] else None,
                float(row[9]),
                float(row[10]),
            )
        )
    raw = {
        name: _frozen(value, dtype=value.dtype)
        for name, value in fields.items()
        if name not in ("solver_diagnostics", "normal_rows", "normal_jacobian")
    }
    raw["candidate_rows"] = _frozen(normal)
    raw["candidate_jacobian"] = _frozen(fields["normal_jacobian"])
    raw["slot_ids"] = _frozen(plan.slot_ids[: plan.nslot], dtype=np.int32)
    raw["solver_diagnostics"] = MappingProxyType(
        dict(zip(DIAGNOSTICS, map(float, diag), strict=True))
    )
    return tuple(contacts), tuple(constraints), MappingProxyType(raw)
