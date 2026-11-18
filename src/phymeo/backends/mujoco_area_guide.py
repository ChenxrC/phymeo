"""Independent audit of one scalar native dry-friction row beside area contact.

The caller supplies the source-compiled plan and same-entry/accepted-velocity
receipt data. This helper does not authenticate their source, solve contact,
modify state or infer a body contact wrench for a joint load. The original
native gradient uses physical qacc; common-velocity KKT is checked separately.
Linear coordinates use metres and newtons. Angular coordinates use radians
and newton-metres. Coefficients come from the plan's actual FP32 native friction table,
not from the row being checked.
"""

import numpy as np

from .mujoco_tyx._constraint_publication import MuJoCoTyxConstraintRow

_EPS = float(np.finfo(np.float32).eps)
_TINY = float(np.finfo(np.float32).tiny)


def _require(ok, message):
    if not ok:
        raise ValueError("MuJoCo area guide row: " + message)


def _scalar(value, name):
    _require(type(value) is float and np.isfinite(value), name + " finite builtin float")
    return value


def _vector(value, name, size):
    _require(
        isinstance(value, np.ndarray)
        and value.shape == (size,)
        and value.dtype == np.float64
        and np.isfinite(value).all(),
        name + " finite FP64 vector of length plan.nv",
    )
    return value


def _rounded(actual, expected, name, scale=None):
    _require(np.isfinite(expected), name + " finite reconstruction")
    magnitude = abs(expected) if scale is None else float(scale)
    _require(np.isfinite(magnitude) and magnitude >= 0.0, name + " finite rounding scale")
    _require(
        abs(actual - expected) <= 32 * _EPS * magnitude + _TINY,
        name + " differs from native FP32 coefficients/algebra",
    )


def _kkt(gradient, force, loss, tolerance, scale, regularization, name):
    _require(
        np.isfinite(gradient) and np.isfinite(scale) and np.isfinite(regularization * force),
        name + " finite KKT algebra",
    )
    violation = abs(gradient)
    at_bound = False
    if force == -loss:
        violation = max(0.0, -gradient)
        at_bound = True
    elif force == loss:
        violation = max(0.0, gradient)
        at_bound = True
    _require(violation <= tolerance * max(1.0, scale), name + " normalized KKT")
    if at_bound:
        _require(
            violation <= tolerance * max(1.0, abs(regularization * force)),
            name + " bound KKT",
        )


def has_scalar_joint_layout(joints, nv):
    """One native qpos and DOF address per fixed-tree slide (0) or hinge (1)."""
    return (
        isinstance(joints, np.ndarray)
        and joints.dtype == np.int32
        and joints.shape == (nv, 4)
        and all(
            kind in (0, 1) and q == d == j and body > 0
            for j, (kind, q, d, body) in enumerate(joints)
        )
    )


def expected_scalar_rows(plan, input_q):
    """Native assembly order and complete owner/side identity at entry q."""
    rows = [(1, int(dof), -1, -1, int(dof)) for dof in plan.friction_ids[: plan.nfriction]]
    if plan.nlimit:
        from .mujoco_area_limit import expected_limit_sides

        rows.extend(
            (3, joint, side, -1, dof) for joint, dof, side in expected_limit_sides(plan, input_q)
        )
    return tuple(rows)


# Keep the original public helper available to slide-only callers.
expected_slide_rows = expected_scalar_rows


def validate_guide_row(plan, row, input_velocity, output_velocity, h, physical_acceleration):
    """Validate one bounded native row and return independent predicted lambda.

    The admitted plan has one or two undamped scalar joints, one dry-friction
    row on either DOF, no native geometry pairs and at most one limited joint. All components
    of velocity/acceleration/Jacobian inputs are checked as finite FP64 receipt
    arrays of shape ``(plan.nv,)``; ``h`` is a finite builtin float whose
    actual integration value is RN32(h). Source/phase authentication, the area
    law, total generalized force and physical-M equation remain the enclosing
    receipt auditor's responsibility. This function only reads its arguments.
    """
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _validate(plan, row, input_velocity, output_velocity, h, physical_acceleration)
    except (AttributeError, TypeError, IndexError, FloatingPointError, OverflowError) as error:
        raise ValueError(
            "MuJoCo area guide row: invalid layout or nonfinite audit algebra"
        ) from error


def _validate(plan, row, input_velocity, output_velocity, h, physical_acceleration):
    _require(type(row) is MuJoCoTyxConstraintRow, "native typed constraint row")
    nv = plan.nv
    _require(type(nv) is int and nv in (1, 2), "one or two scalar DOFs")
    for name, expected in (
        ("kind", 2),
        ("nq", nv),
        ("nfriction", 1),
        ("npair", 0),
        ("nslot", 0),
    ):
        value = getattr(plan, name)
        _require(type(value) is int and value == expected, "unsupported plan " + name)
    _require(type(plan.nlimit) is int and plan.nlimit in (0, 1), "unsupported limit count")
    joints, ids, law, vectors = plan.joint_ids, plan.friction_ids, plan.friction_law, plan.vectors
    _require(
        has_scalar_joint_layout(joints, nv),
        "complete native-addressed scalar joints",
    )
    _require(
        isinstance(ids, np.ndarray)
        and ids.dtype == np.int32
        and ids.shape == (1,)
        and 0 <= ids[0] < nv,
        "one dry row at an admitted DOF",
    )
    # Native assembly writes friction_ids[f] to both source_id and dof. It is
    # the native velocity address, not the ordinal f in the friction table.
    dof = int(ids[0])
    _require(
        isinstance(law, np.ndarray)
        and law.dtype == np.float32
        and law.shape == (1, 9)
        and np.isfinite(law).all(),
        "finite source friction law with nine FP32 coefficients",
    )
    _require(
        isinstance(vectors, np.ndarray)
        and vectors.dtype == np.float32
        and vectors.ndim == 2
        and vectors.shape[0] == 9
        and 8 <= vectors.shape[1] <= 45
        and np.all(vectors[1, :nv] == 0.0),
        "undamped native scalar joints",
    )
    vin = _vector(input_velocity, "input velocity", nv)
    vout = _vector(output_velocity, "output velocity", nv)
    physical = _vector(physical_acceleration, "physical acceleration", nv)
    step = float(np.float32(_scalar(h, "h")))
    _require(np.isfinite(step) and step >= _TINY, "positive normal FP32 h")
    tolerance = _scalar(plan.solver_tolerance, "source solver tolerance")
    _require(tolerance >= 2e-5 * (1.0 - _EPS), "native solver tolerance floor")
    for name, expected in (
        ("constraint_type", 1),
        ("source_id", dof),
        ("side_or_edge", -1),
        ("geometry_slot", -1),
        ("dof", dof),
    ):
        value = getattr(row, name)
        _require(type(value) is int and value == expected, "native row " + name)
    _require(
        type(row.schema) is str and row.schema == "mujoco-tyx-constraint-row-v1",
        "native row schema",
    )
    jacobian = _vector(row.jacobian, "row Jacobian", nv)
    _require(
        all(jacobian[d] == (1.0 if d == dof else 0.0) for d in range(nv)),
        "dry scalar Jacobian is the exact native DOF basis row",
    )
    for name in (
        "position",
        "margin",
        "velocity",
        "reference_acceleration",
        "regularization",
        "inverse_regularization",
        "impedance",
        "diag_approx",
        "lower",
        "upper",
        "force",
        "gradient",
    ):
        _scalar(getattr(row, name), "row " + name)
    _require(row.position == 0.0 and row.margin == 0.0, "zero dry position and margin")
    _require(row.velocity == float(vin[dof]), "entry velocity binding")

    loss, invweight, ref0, ref1, raw_min, raw_max, width, _, _ = map(float, law[0])
    _require(loss > 0.0 and invweight > 0.0, "positive source loss and inverse weight")
    _require((ref0 > 0.0) == (ref1 > 0.0), "native positive/direct solref agreement")
    minimum = float(np.float32(1e-15))
    minimp, maximp = float(np.float32(0.0001)), float(np.float32(0.9999))
    dmin = min(maximp, max(minimp, raw_min))
    dmax = min(maximp, max(minimp, raw_max))
    impedance = dmin if dmin != dmax and max(0.0, width) > minimum else 0.5 * (dmin + dmax)
    # Position is zero: midpoint/power cannot affect impedance, but both source
    # coefficients were still checked for finiteness with the complete table.
    reference_time = max(ref0, 2.0 * step) if ref0 > 0.0 else ref0
    damping = (
        2.0 / max(minimum, dmax * reference_time) if ref1 > 0.0 else -ref1 / max(minimum, dmax)
    )
    aref = -damping * float(vin[dof])
    regularization = max(minimum, (1.0 - impedance) * invweight / impedance)
    _rounded(row.impedance, impedance, "impedance")
    _rounded(row.reference_acceleration, aref, "reference acceleration")
    _rounded(row.regularization, regularization, "regularization")
    _rounded(row.inverse_regularization, 1.0 / regularization, "inverse regularization")
    _rounded(row.diag_approx, regularization * impedance / (1.0 - impedance), "diag approximation")
    _require(row.lower == -loss and row.upper == loss, "source friction bounds")
    force = row.force
    _require(-loss <= force <= loss, "force inside exact box bounds")

    common = (vout - vin) / step
    _require(np.isfinite(common).all(), "finite common-velocity acceleration on every DOF")
    _require(
        np.all(np.abs(vout - (vin + step * physical)) <= 5e-5 * (1.0 + np.abs(vout))),
        "physical acceleration and common velocity disagree on a DOF",
    )
    # The joint row acts on one DOF, but coupled area response may accelerate
    # both. Only after checking the complete vectors may KKT use this projection.
    physical_axis, common_axis = float(physical[dof]), float(common[dof])
    physical_gradient = physical_axis + row.regularization * force - row.reference_acceleration
    physical_scale = (
        abs(physical_axis) + abs(row.regularization * force) + abs(row.reference_acceleration)
    )
    _rounded(row.gradient, physical_gradient, "physical-qacc gradient", physical_scale)
    _kkt(
        physical_gradient,
        force,
        loss,
        tolerance,
        physical_scale,
        row.regularization,
        "physical-qacc",
    )
    common_gradient = common_axis + regularization * force - aref
    common_scale = abs(common_axis) + abs(regularization * force) + abs(aref)
    _kkt(common_gradient, force, loss, tolerance, common_scale, regularization, "common-velocity")
    predicted = min(loss, max(-loss, (aref - common_axis) / regularization))
    _require(np.isfinite(predicted), "finite predicted force")
    _require(abs(force - predicted) <= 0.002 + 0.002 * abs(predicted), "predicted native force")
    return float(predicted)
