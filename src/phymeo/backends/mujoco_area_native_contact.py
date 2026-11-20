"""Native geometry receipts beside final-normal area friction.

The common device solve supplies every force. This module only decodes and
substitutes saved values, including during restore; it performs no FK, contact
solve, device allocation, staging or state mutation.
"""

from collections.abc import Mapping
from dataclasses import fields as dataclass_fields

import numpy as np

from .mujoco_tyx._constraint_publication import publish as publish_constraints

RAW_FIELDS = frozenset(
    (
        "efc_ids",
        "efc_data",
        "efc_J",
        "counts",
        "contact_map",
        "contact_J",
        "contact_wrench",
        "candidate_rows",
        "candidate_jacobian",
        "slot_ids",
    )
)


def enabled(plan):
    return bool(getattr(plan, "damped_contact_dofs", 0) and plan.area_friction_laws)


def _require(ok, message):
    if not ok:
        raise ValueError("area-friction native contact: " + message)


def _close(actual, expected, name, *, absolute=2e-6, relative=2e-5):
    _require(
        np.isfinite(actual).all()
        and np.isfinite(expected).all()
        and np.all(np.abs(actual - expected) <= absolute + relative * np.abs(expected)),
        name,
    )


def publish(runtime, result, index):
    """Adapt the live download names to the shared native contact decoder."""
    _require(enabled(runtime._plan), "explicit damped native contact scope required")
    adapted = dict(result)
    adapted["efc_J"] = result["efc_jacobian"]
    adapted["counts"] = result["constraint_counts"]
    return publish_constraints(runtime, adapted, index, native_only=True)


def _same_records(actual, expected):
    _require(type(actual) is tuple and len(actual) == len(expected), "complete decoded records")
    for got, want in zip(actual, expected, strict=True):
        _require(type(got) is type(want), "decoded record type")
        for field in dataclass_fields(want):
            left, right = getattr(got, field.name), getattr(want, field.name)
            if isinstance(right, np.ndarray):
                equal = (
                    isinstance(left, np.ndarray)
                    and left.dtype == right.dtype
                    and left.shape == right.shape
                    and left.tobytes() == right.tobytes()
                )
            else:
                equal = type(left) is type(right) and left == right
            _require(equal, "raw/decoded record binding: " + field.name)


def _source_geometry(plan, row, dynamics):
    """Bind the sole plane/sphere candidate to saved entry body poses.

    This is bounded primitive substitution, not native FK. The device composes
    packed local geoms with the already saved body poses; FP64 substitution is
    bounded for ordinary FP32 accumulation/FMA, including detection boundaries.
    Geometry IDs need not be numerically ordered: slot order is native type order.
    """
    _, feature, plane, sphere = map(int, plan.slot_ids[0])
    _require(
        feature == 0 and tuple(plan.geom_ids[[plane, sphere], 1]) == (0, 2),
        "source plane/sphere slot",
    )
    eps = np.finfo(np.float32).eps
    tiny = np.finfo(np.float32).tiny

    def pose(geom):
        body = int(plan.geom_ids[geom, 0])
        local = plan.geom_data[geom].astype(np.float64)
        rotation = dynamics.body_rotation[body]
        origin = dynamics.body_position[body]
        position = origin + rotation @ local[:3]
        orientation = rotation @ local[3:12].reshape(3, 3)
        position_scale = np.abs(origin) + np.abs(rotation) @ np.abs(local[:3])
        orientation_scale = np.abs(rotation) @ np.abs(local[3:12].reshape(3, 3))
        return position, orientation, position_scale, orientation_scale

    pp, pr, ps, rs = pose(plane)
    center, _, cs, _ = pose(sphere)
    radius = float(plan.geom_data[sphere, 12])
    normal = pr[:, 2]
    delta = center - pp
    distance = float(delta @ normal - radius)
    # Include cancellation in world-coordinate subtraction and pose composition,
    # not only the possibly tiny resulting signed gap.
    distance_scale = radius + np.abs(normal) @ (ps + cs) + np.abs(delta) @ rs[:, 2]
    gap_error = 32 * eps * distance_scale + tiny
    _require(np.isfinite(distance) and np.isfinite(gap_error), "source geometric range")
    margin, gap = map(float, plan.pair_law[0, :2])
    include = float(np.float32(plan.pair_law[0, 0] - plan.pair_law[0, 1]))
    detected = bool(row[0])
    # The true device comparison can lie on either side only within the forward
    # error envelope. Never require FP64 and FP32 branch choices to agree there.
    _require(
        distance <= margin + gap_error if detected else distance >= margin - gap_error,
        "source native detection",
    )
    if not detected:
        return
    _require(abs(float(row[2]) - distance) <= gap_error, "source native distance")
    _require(row[21] == include and row[22] == margin and row[23] == gap, "source native margins")
    _require(
        distance <= include + gap_error if row[1] else distance >= include - gap_error,
        "source native activity",
    )
    point = center - (radius + 0.5 * distance) * normal
    point_error = (
        32 * eps * (cs + (radius + 0.5 * abs(distance)) * rs[:, 2])
        + 0.5 * gap_error * np.abs(normal)
        + tiny
    )
    _require(
        np.isfinite(point).all() and np.all(np.abs(row[9:12] - point) <= point_error),
        "source native point",
    )
    length = float(np.linalg.norm(normal))
    _require(np.isfinite(length) and length > 0, "source native normal length")
    normal = normal / length
    frame = row[12:21].reshape(3, 3)
    _close(frame[0], normal, "source native normal", absolute=32 * eps, relative=32 * eps)
    # Plane/sphere has no capsule tangent seed. Match the native deterministic
    # Y/Z seed, allowing either branch only at its FP32 +/-0.5 boundary.
    seed_axes = [1 if -0.5 < normal[1] < 0.5 else 2]
    if abs(abs(normal[1]) - 0.5) <= 32 * eps:
        seed_axes = [1, 2]
    matches = False
    for axis in seed_axes:
        seed = np.eye(3)[axis]
        tangent = seed - normal[axis] * normal
        tangent /= np.linalg.norm(tangent)
        expected = np.vstack((normal, tangent, np.cross(normal, tangent)))
        matches |= bool(np.all(np.abs(frame - expected) <= 64 * eps))
    _require(matches, "source native tangent frame")


def _source_law(plan, row, h):
    """Reconstruct the native pyramidal coefficients from compiled pair law."""
    law = plan.pair_law[0].astype(np.float64)
    _, _, invweight, ref0, ref1, raw_min, raw_max, width, midpoint, power = law[:10]
    minimum, minimp, maximp = law[10:13]
    dmin, dmax = np.clip((raw_min, raw_max), minimp, maximp)
    width, midpoint, power = max(0.0, width), np.clip(midpoint, minimp, maximp), max(1.0, power)
    position = row.position - row.margin
    impedance = 0.5 * (dmin + dmax)
    if dmin != dmax and width > minimum:
        x = abs(position / width)
        y = min(1.0, x)
        if 0 < x < 1 and power != 1:
            y = (
                x**power / midpoint ** (power - 1)
                if x <= midpoint
                else 1 - (1 - x) ** power / (1 - midpoint) ** (power - 1)
            )
        impedance = dmin + y * (dmax - dmin)
    ref0 = max(ref0, 2 * h) if ref0 > 0 else ref0
    stiffness = (
        1 / max(minimum, dmax * dmax * ref0 * ref0 * ref1 * ref1)
        if ref0 > 0
        else -ref0 / max(minimum, dmax * dmax)
    )
    damping = 2 / max(minimum, dmax * ref0) if ref1 > 0 else -ref1 / max(minimum, dmax)
    mu = float(plan.pair_friction[0, 0])
    regularization = max(minimum, (1 - impedance) * invweight * (1 + mu * mu) / impedance)
    tangent_r = regularization / max(minimum, plan.impratio)
    scaled_mu = mu * np.sqrt(tangent_r / regularization)
    regularization = 2 * scaled_mu * scaled_mu * regularization
    aref = -damping * row.velocity - stiffness * impedance * position
    for actual, expected, name in (
        (row.impedance, impedance, "source impedance"),
        (row.regularization, regularization, "source regularization"),
        (row.inverse_regularization, 1 / regularization, "source inverse regularization"),
        (row.diag_approx, regularization * impedance / (1 - impedance), "source diagonal"),
        (row.reference_acceleration, aref, "source reference acceleration"),
    ):
        magnitude = (
            abs(damping * row.velocity) + abs(stiffness * impedance * position)
            if name == "source reference acceleration"
            else abs(expected)
        )
        _require(
            np.isfinite(expected)
            and abs(actual - expected)
            <= 64 * np.finfo(np.float32).eps * magnitude + np.finfo(np.float32).tiny,
            name,
        )
    return float(aref), float(regularization)


def validate(runtime, raw, contacts, constraints, dynamics, qacc, h, *, output_v=None):
    """Return stored and independent-law native generalized loads.

    Scope is one plane/sphere condim=3 pair on one damped scalar joint
    or the explicitly admitted two-scalar coupled route, optionally with one
    or two distinct finite joint limits. No native pair is
    inferred from an area pair or from receipt fields.
    The physical-M row law intentionally uses qacc, not integration acceleration.
    """
    plan = runtime._plan
    _require(
        enabled(plan)
        and plan.damped_contact_dofs in (1, 2)
        and plan.nv == plan.nq == plan.damped_contact_dofs
        and plan.npair == plan.nslot == 1
        and plan.nfriction == 0
        and (plan.nlimit == 0 or (plan.damped_contact_dofs == 2 and plan.nlimit in (1, 2)))
        and int(plan.pair_condim[0]) == 3,
        "unsupported native geometry scope",
    )
    _require(isinstance(raw, Mapping) and set(raw) == RAW_FIELDS, "raw native fields")
    integer = {"efc_ids", "counts", "contact_map", "slot_ids"}
    for name, value in raw.items():
        dtype = (
            np.int32
            if name in integer
            else np.float64
            if name in ("candidate_rows", "candidate_jacobian")
            else np.float32
        )
        _require(
            isinstance(value, np.ndarray) and value.dtype == dtype and np.isfinite(value).all(),
            "raw field dtype/finite: " + name,
        )
        if dtype == np.float64:
            _require(
                np.array_equal(value, value.astype(np.float32).astype(np.float64)),
                "decoded native FP32 field: " + name,
            )
    _require(np.array_equal(raw["slot_ids"], plan.slot_ids[:1]), "source slot identity")
    names = {"candidate_rows": "normal_rows", "candidate_jacobian": "normal_jacobian"}
    replay = {
        names.get(name, name): (value.astype(np.float32) if name in names else value)[None]
        for name, value in raw.items()
        if name != "slot_ids"
    }
    # Reproduce the decoder's device-precision point-minus-COM arithmetic;
    # the independent point-J/source-law audit below stays in FP64.
    replay.update(
        input_q=dynamics.input_q.astype(np.float32)[None],
        body_com=dynamics.body_com.astype(np.float32)[None],
        qacc=qacc.astype(np.float32)[None],
    )
    limit_rows = tuple(row for row in constraints if row.constraint_type == 3)
    ground_rows = tuple(row for row in constraints if row.constraint_type == 6)
    _require(len(limit_rows) + len(ground_rows) == len(constraints), "native row groups")
    if plan.nlimit:
        from .mujoco_area_limit import expected_limit_sides, validate_limit_row

        expected = expected_limit_sides(plan, dynamics.input_q)
        _require(
            len(expected) <= plan.nlimit
            and len({joint for joint, _, _ in expected}) == len(expected)
            and tuple((row.source_id, row.dof, row.side_or_edge) for row in limit_rows) == expected,
            "complete nonoverlapping source limit rows",
        )
    decoded_contacts, decoded_rows, _ = publish_constraints(runtime, replay, 0, native_only=True)
    _source_geometry(plan, raw["candidate_rows"][0], dynamics)
    _same_records(contacts, decoded_contacts)
    _same_records(constraints, decoded_rows)
    _require(np.all(raw["efc_J"][:, plan.nv :] == 0), "native Jacobian padding")
    _require(np.all(raw["candidate_jacobian"][:, plan.nv :] == 0), "candidate Jacobian padding")
    _require(np.all(raw["contact_J"][:, :, plan.nv :] == 0), "contact Jacobian padding")
    total, predicted_total = np.zeros(plan.nv), np.zeros(plan.nv)
    if plan.nlimit:
        for row in constraints:
            gradient = (
                float(row.jacobian @ qacc)
                + row.regularization * row.force
                - row.reference_acceleration
            )
            _require(
                gradient >= -2e-5 * (1.0 + abs(row.reference_acceleration)),
                "unilateral gradient at published physical acceleration",
            )
        for row in limit_rows:
            predicted = validate_limit_row(
                plan, row, dynamics.input_q, dynamics.input_v, output_v, h, qacc
            )
            total += row.jacobian * row.force
            predicted_total += row.jacobian * predicted
    else:
        _require(not limit_rows, "no undeclared limit rows")
    for ci, contact in enumerate(contacts):
        ba, bb = contact.body_ids
        point, frame = contact.position, contact.frame
        _close(frame @ frame.T, np.eye(3), "native contact frame")
        ja = (
            dynamics.jacobian_com[ba, :3]
            + np.cross(dynamics.jacobian_com[ba, 3:].T, point - dynamics.body_com[ba]).T
        )
        jb = (
            dynamics.jacobian_com[bb, :3]
            + np.cross(dynamics.jacobian_com[bb, 3:].T, point - dynamics.body_com[bb]).T
        )
        expected_j = np.vstack(
            (
                frame @ (jb - ja),
                frame @ (dynamics.jacobian_com[bb, 3:] - dynamics.jacobian_com[ba, 3:]),
            )
        )
        _close(contact.jacobian, expected_j, "same-entry native point Jacobian")
        _close(
            raw["candidate_jacobian"][0, : plan.nv],
            contact.jacobian[0],
            "candidate normal Jacobian",
        )
        _close(
            contact.normal_velocity,
            float(contact.jacobian[0] @ dynamics.input_v),
            "native normal entry velocity",
        )
        include = float(np.float32(plan.pair_law[0, 0] - plan.pair_law[0, 1]))
        _require(contact.include_margin == include, "source include margin")
        _require(contact.active == (contact.distance < include), "native activity")
        _require(contact.distance <= float(plan.pair_law[0, 0]), "native detection margin")
        # Native joint rows precede ground rows; they never contribute to the
        # contact wrench or consume a pyramid edge index.
        rows = tuple(row for row in ground_rows if row.source_id == ci)
        _require(len(rows) == (4 if contact.active else 0), "complete ground pyramid")
        loads = np.zeros(4)
        for edge, row in enumerate(rows):
            _require(
                row.side_or_edge == edge and row.geometry_slot == 0 and row.dof == -1,
                "source ground edge identity",
            )
            component, sign = edge // 2 + 1, 1 if edge % 2 == 0 else -1
            jacobian = (
                contact.jacobian[0]
                + sign * float(plan.pair_friction[0, component - 1]) * contact.jacobian[component]
            )
            _close(row.jacobian, jacobian, "pyramid edge Jacobian")
            _require(
                row.position == contact.distance and row.margin == include, "native row geometry"
            )
            _require(row.lower == 0 and row.upper is None, "native unilateral bound")
            _close(row.velocity, float(row.jacobian @ dynamics.input_v), "native entry velocity")
            aref, regularization = _source_law(plan, row, h)
            predicted = max(0.0, (aref - float(row.jacobian @ qacc)) / regularization)
            _close(
                row.force, predicted, "native physical-M force law", absolute=0.002, relative=0.002
            )
            total += row.jacobian * row.force
            predicted_total += row.jacobian * predicted
            loads[edge] = row.force
        wrench = np.zeros(6)
        wrench[0] = loads.sum()
        wrench[1:3] = plan.pair_friction[0, :2] * (loads[::2] - loads[1::2])
        _close(contact.wrench_contact_frame, wrench, "pyramid decoded wrench")
    return total, predicted_total
