"""Independent audit of one native dry-friction SLIDE row beside area contact.

The caller supplies the source-compiled plan and same-entry/accepted-velocity
receipt data. This helper does not authenticate their source, solve contact,
modify state or infer a body contact wrench for a joint load. The original
native gradient uses physical qacc; common-velocity KKT is checked separately.
All quantities use SI (SLIDE loss/force in N, velocity in m/s, acceleration in
m/s**2). Coefficients come from the plan's actual FP32 native friction table,
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


def _vector(value, name):
    _require(
        isinstance(value, np.ndarray)
        and value.shape == (1,)
        and value.dtype == np.float64
        and np.isfinite(value).all(),
        name + " finite FP64 vector of length one",
    )
    return float(value[0])


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


def validate_guide_row(plan, row, input_velocity, output_velocity, h, physical_acceleration):
    """Validate one bounded native row and return independent predicted lambda.

    The admitted plan has one undamped scalar SLIDE, one dry-friction row, no
    native geometry pairs and no limits. Velocity/acceleration inputs are finite
    FP64 receipt arrays of shape ``(1,)``; ``h`` is a finite builtin float whose
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
    for name, expected in (
        ("kind", 2),
        ("nv", 1),
        ("nq", 1),
        ("nfriction", 1),
        ("nlimit", 0),
        ("npair", 0),
        ("nslot", 0),
    ):
        value = getattr(plan, name)
        _require(type(value) is int and value == expected, "unsupported plan " + name)
    joints, ids, law, vectors = plan.joint_ids, plan.friction_ids, plan.friction_law, plan.vectors
    _require(
        isinstance(joints, np.ndarray)
        and joints.dtype == np.int32
        and joints.shape == (1, 4)
        and tuple(joints[0, :3]) == (0, 0, 0)
        and joints[0, 3] > 0,
        "one native-addressed SLIDE joint",
    )
    _require(
        isinstance(ids, np.ndarray) and ids.dtype == np.int32 and ids.shape == (1,) and ids[0] == 0,
        "one dry row at DOF zero",
    )
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
        and vectors[1, 0] == 0.0,
        "undamped native SLIDE vectors",
    )
    vin = _vector(input_velocity, "input velocity")
    vout = _vector(output_velocity, "output velocity")
    physical = _vector(physical_acceleration, "physical acceleration")
    step = float(np.float32(_scalar(h, "h")))
    _require(np.isfinite(step) and step >= _TINY, "positive normal FP32 h")
    tolerance = _scalar(plan.solver_tolerance, "source solver tolerance")
    _require(tolerance >= 2e-5 * (1.0 - _EPS), "native solver tolerance floor")
    for name, expected in (
        ("constraint_type", 1),
        ("source_id", 0),
        ("side_or_edge", -1),
        ("geometry_slot", -1),
        ("dof", 0),
    ):
        value = getattr(row, name)
        _require(type(value) is int and value == expected, "native row " + name)
    _require(
        type(row.schema) is str and row.schema == "mujoco-tyx-constraint-row-v1",
        "native row schema",
    )
    _require(_vector(row.jacobian, "row Jacobian") == 1.0, "SLIDE Jacobian equals one")
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
    _require(row.velocity == vin, "entry velocity binding")

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
    aref = -damping * vin
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
    _require(np.isfinite(common), "finite common-velocity acceleration")
    _require(
        abs(vout - (vin + step * physical)) <= 5e-5 * (1.0 + abs(vout)),
        "physical acceleration and common velocity disagree",
    )
    physical_gradient = physical + row.regularization * force - row.reference_acceleration
    physical_scale = (
        abs(physical) + abs(row.regularization * force) + abs(row.reference_acceleration)
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
    common_gradient = common + regularization * force - aref
    common_scale = abs(common) + abs(regularization * force) + abs(aref)
    _kkt(common_gradient, force, loss, tolerance, common_scale, regularization, "common-velocity")
    predicted = min(loss, max(-loss, (aref - common) / regularization))
    _require(np.isfinite(predicted), "finite predicted force")
    _require(abs(force - predicted) <= 0.002 + 0.002 * abs(predicted), "predicted native force")
    return float(predicted)
