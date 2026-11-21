"""Independent native scalar joint-limit receipt audit beside area contact.

The source-compiled plan and same-entry receipt are trusted inputs whose source
and phase the enclosing auditor binds. This module neither solves contact nor
changes state. It supports one or two undamped scalar joints, one declared limited
joint and zero or one dry-friction row. The optional dry row is audited separately.
The explicit damped two-DOF ground/area route admits one or two distinct limited
joints and uses physical-M acceleration for each native limit law; its enclosing
auditor binds M+hD integration separately.
The active-row list is derived from actual FP32 q/range/margin; typed position
is distance, while the impedance and spring position is distance minus margin.
"""

import numpy as np

from .mujoco_area_guide import _EPS, _TINY, _rounded, _scalar, _vector, has_scalar_joint_layout
from .mujoco_tyx._constraint_publication import MuJoCoTyxConstraintRow


def _require(ok, message):
    if not ok:
        raise ValueError("MuJoCo area limit row: " + message)


def _rn32(value, name):
    value = float(np.float32(value))
    _require(np.isfinite(value), name + " finite FP32 value")
    return value


def _intermediate32(value):
    # Some native intermediates may saturate to infinity while the row stays
    # finite: |position/width| selects the plateau, or 1/overflowed denominator
    # becomes zero stiffness/damping. Do not add a stricter finite-source gate.
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.float32(value))


def _damped_native_limit(plan):
    if getattr(plan, "resident_native_contacts", False):
        return True
    return bool(
        getattr(plan, "damped_contact_dofs", 0) == 2
        and plan.nv == plan.nq == 2
        and plan.nlimit in (1, 2)
        and plan.npair == plan.nslot == 1
        and plan.nfriction == 0
        and getattr(plan, "area_friction_laws", ())
    )


def _layout(plan, input_q):
    if getattr(plan, "resident_native_contacts", False):
        return _resident_layout(plan, input_q)
    nv = plan.nv
    _require(type(nv) is int and nv in (1, 2), "one or two scalar DOFs")
    damped_native = _damped_native_limit(plan)
    pairs = int(damped_native)
    nlimit = plan.nlimit
    _require(
        type(nlimit) is int and (nlimit in (1, 2) if damped_native else nlimit == 1),
        "declared limit count in the admitted layout",
    )
    for name, expected in (
        ("kind", 2),
        ("nq", nv),
        ("npair", pairs),
        ("nslot", pairs),
    ):
        value = getattr(plan, name)
        _require(type(value) is int and value == expected, "unsupported plan " + name)
    _require(type(plan.nfriction) is int and plan.nfriction in (0, 1), "at most one dry row")
    joints, vectors = plan.joint_ids, plan.vectors
    _require(
        has_scalar_joint_layout(joints, nv),
        "complete native-addressed scalar joints",
    )
    _require(
        isinstance(vectors, np.ndarray)
        and vectors.dtype == np.float32
        and vectors.ndim == 2
        and vectors.shape[0] == 9
        and 8 <= vectors.shape[1] <= 45
        and np.isfinite(vectors[1, :nv]).all()
        and np.all(vectors[1, :nv] >= 0.0)
        and (damped_native or np.all(vectors[1, :nv] == 0.0)),
        "source viscous damping in the admitted layout",
    )
    ids, laws = plan.limit_ids, plan.limit_law
    _require(
        isinstance(ids, np.ndarray) and ids.dtype == np.int32 and ids.shape == (nlimit, 3),
        "complete declared limit identities",
    )
    _require(
        isinstance(laws, np.ndarray)
        and laws.dtype == np.float32
        and laws.shape == (nlimit, 11)
        and np.isfinite(laws).all(),
        "complete finite source limit laws",
    )
    entries = []
    previous_joint = -1
    for identity, law in zip(ids, laws, strict=True):
        joint, qa, dof = map(int, identity)
        _require(
            previous_joint < joint < nv
            and 0 <= qa < nv
            and 0 <= dof < nv
            and qa == int(joints[joint, 1])
            and dof == int(joints[joint, 2]),
            "ordered unique limit joint/qpos/DOF mapping",
        )
        previous_joint = joint
        _require(law[3] > 0.0, "positive source inverse weight")
        _require((law[4] > 0.0) == (law[5] > 0.0), "native positive/direct solref agreement")
        entries.append((joint, qa, dof, tuple(map(float, law))))
    q = _vector(input_q, "limit input q", nv)
    q32 = q.astype(np.float32)
    _require(np.isfinite(q32).all(), "all input q representable as finite FP32")
    return nv, tuple(entries), q32


def _distances(q32, qa, law):
    value = float(q32[qa])
    # Both operands are device FP32 values; the subtraction itself rounds before
    # the strict activation comparison. Equality with margin is not active.
    return (
        (-1, _rn32(value - law[0], "lower distance")),
        (1, _rn32(law[1] - value, "upper distance")),
    )


def expected_limit_sides(plan, input_q):
    """Return all active ``(native_joint_id, dof, side)`` in native row order.

    Lower side is -1, upper side +1. The legacy undamped contract retains both
    sides when their margins overlap. The damped native route admits at most one
    side per joint. This function derives every row from its own source q/range;
    callers compare the complete returned order to their receipts.
    """
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            _, entries, q32 = _layout(plan, input_q)
            damped_native = _damped_native_limit(plan)
            expected = []
            for joint, qa, dof, law in entries:
                active = tuple(
                    (joint, dof, side)
                    for side, distance in _distances(q32, qa, law)
                    if distance < law[2]
                )
                _require(
                    not damped_native or len(active) <= 1,
                    "at most one active side per damped limited joint",
                )
                expected.extend(active)
            return tuple(expected)
    except (AttributeError, TypeError, IndexError, FloatingPointError, OverflowError) as error:
        raise ValueError("MuJoCo area limit row: invalid layout or nonfinite activation") from error


def _coefficients(law, distance, velocity, step):
    _, _, margin, invweight, ref0, ref1, raw_min, raw_max, width, raw_mid, raw_power = law
    minimum = _rn32(1e-15, "native minimum")
    minimp, maximp = _rn32(0.0001, "minimum impedance"), _rn32(0.9999, "maximum impedance")
    dmin, dmax = min(maximp, max(minimp, raw_min)), min(maximp, max(minimp, raw_max))
    width, midpoint, power = max(0.0, width), min(maximp, max(minimp, raw_mid)), max(1.0, raw_power)
    position = _rn32(distance - margin, "spring position")
    impedance = _rn32(0.5 * _rn32(dmin + dmax, "impedance sum"), "mean impedance")
    if dmin != dmax and width > minimum:
        x = abs(_intermediate32(position / width))
        impedance = dmin
        if x >= 1.0:
            impedance = dmax
        elif x > 0.0:
            y = x
            if power != 1.0:
                base, numerator_base = midpoint, x
                if x > midpoint:
                    base = _rn32(1.0 - midpoint, "upper midpoint")
                    numerator_base = _rn32(1.0 - x, "upper impedance ratio")
                denominator = base
                if power == 3.0:
                    denominator = _rn32(base * base, "impedance denominator")
                elif power != 2.0:
                    denominator = _rn32(
                        np.power(
                            np.float32(base), np.float32(_rn32(power - 1.0, "power exponent"))
                        ),
                        "impedance denominator",
                    )
                numerator = _rn32(numerator_base * numerator_base, "impedance numerator")
                if power != 2.0:
                    numerator = _rn32(
                        np.power(np.float32(numerator_base), np.float32(power)),
                        "impedance numerator",
                    )
                _require(denominator > 0.0, "positive native impedance denominator")
                y = _rn32(
                    _rn32(1.0 / denominator, "impedance reciprocal") * numerator, "impedance blend"
                )
                if x > midpoint:
                    y = _rn32(1.0 - y, "upper impedance blend")
            impedance = _rn32(dmin + y * _rn32(dmax - dmin, "impedance range"), "impedance")
    _require(0.0 < impedance < 1.0, "native impedance domain")
    if ref0 > 0.0:
        ref0 = max(ref0, _intermediate32(2.0 * step))
        denominator = dmax
        for factor in (dmax, ref0, ref0, ref1, ref1):
            denominator = _intermediate32(denominator * factor)
        stiffness = _rn32(1.0 / max(minimum, denominator), "stiffness")
    else:
        stiffness = _rn32(
            -ref0 / max(minimum, _rn32(dmax * dmax, "direct denominator")), "stiffness"
        )
    damping = (
        _rn32(2.0 / max(minimum, _intermediate32(dmax * ref0)), "damping")
        if ref1 > 0.0
        else _rn32(-ref1 / max(minimum, dmax), "damping")
    )
    numerator = _rn32(_rn32(1.0 - impedance, "impedance complement") * invweight, "R numerator")
    regularization = max(minimum, _rn32(numerator / impedance, "regularization"))
    inverse = _rn32(1.0 / regularization, "inverse regularization")
    diag = _rn32(
        _rn32(regularization * impedance, "diagonal numerator")
        / _rn32(1.0 - impedance, "diagonal denominator"),
        "diagonal approximation",
    )
    spring_factor = _rn32(stiffness * impedance, "spring factor")
    # Keep a cancellation-aware independent reference. The device may fuse the
    # final multiply/subtract; the existing 32-epsilon gate uses both term sizes.
    damping_term, spring_term = damping * velocity, spring_factor * position
    aref = -damping_term - spring_term
    scale = abs(damping_term) + abs(spring_term)
    _require(np.isfinite(aref) and np.isfinite(scale), "finite reference acceleration")
    return impedance, regularization, inverse, diag, aref, scale


def _lower_kkt(gradient, force, tolerance, scale, label):
    _require(np.isfinite(gradient) and np.isfinite(scale), label + " finite KKT algebra")
    violation = max(0.0, -gradient) if force == 0.0 else abs(gradient)
    _require(violation <= tolerance * max(1.0, scale), label + " normalized KKT")
    if force == 0.0:
        # Native boundary normalization is max(1, |R*lambda|), here exactly one.
        _require(violation <= tolerance, label + " lower-bound KKT")


def validate_limit_row(plan, row, input_q, input_v, output_v, h, physical_qacc):
    """Audit an active typed native limit row; return predicted nonnegative lambda.

    All q/v/qacc/J vectors have complete ``plan.nv`` FP64 shape. h is RN32 for
    actual integration. The full velocity/physical-acceleration agreement is
    checked for the undamped route before projecting onto the signed unit row.
    In the explicit damped native route, only physical qacc enters the limit law;
    the enclosing auditor checks the distinct M+hD integration law. Physical-M,
    total generalized force, endpoint q integration, phase and complete row-list
    binding remain the caller's responsibility. Typed rows have no raw flags:
    the caller must also bind raw active=1, upper_mask=0 and unused upper=0.
    """
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _validate(plan, row, input_q, input_v, output_v, h, physical_qacc)
    except (AttributeError, TypeError, IndexError, FloatingPointError, OverflowError) as error:
        raise ValueError(
            "MuJoCo area limit row: invalid layout or nonfinite audit algebra"
        ) from error


def _validate(plan, row, input_q, input_v, output_v, h, physical_qacc):
    nv, entries, q32 = _layout(plan, input_q)
    _require(type(row) is MuJoCoTyxConstraintRow, "native typed constraint row")
    _require(type(row.source_id) is int, "native limit source identity")
    matches = tuple(entry for entry in entries if entry[0] == row.source_id)
    _require(len(matches) == 1, "row belongs to exactly one declared limited joint")
    joint, qa, dof, law = matches[0]
    _require(type(row.side_or_edge) is int and row.side_or_edge in (-1, 1), "native limit side")
    side = row.side_or_edge
    distance = dict(_distances(q32, qa, law))[side]
    _require(distance < law[2], "row side must be strictly active at the input q")
    for name, expected in (
        ("constraint_type", 3),
        ("source_id", joint),
        ("geometry_slot", -1),
        ("dof", dof),
    ):
        value = getattr(row, name)
        _require(type(value) is int and value == expected, "native row " + name)
    _require(
        type(row.schema) is str and row.schema == "mujoco-tyx-constraint-row-v1",
        "native row schema",
    )
    sign = -float(side)
    jacobian = _vector(row.jacobian, "limit Jacobian", nv)
    _require(
        all(jacobian[d] == (sign if d == dof else 0.0) for d in range(nv)),
        "exact native signed DOF Jacobian",
    )
    vin = _vector(input_v, "limit input velocity", nv)
    vout = _vector(output_v, "limit output velocity", nv)
    physical = _vector(physical_qacc, "limit physical acceleration", nv)
    step = _rn32(_scalar(h, "limit h"), "integration h")
    _require(step >= _TINY, "positive normal FP32 h")
    tolerance = _scalar(plan.solver_tolerance, "source solver tolerance")
    _require(tolerance >= 2e-5 * (1.0 - _EPS), "native solver tolerance floor")
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
        "force",
        "gradient",
    ):
        _scalar(getattr(row, name), "limit row " + name)
    _require(
        row.upper is None and row.lower == 0.0 and row.force >= 0.0, "unilateral native bounds"
    )
    velocity = sign * float(vin[dof])
    _require(row.position == distance and row.margin == law[2], "input distance/margin binding")
    _require(row.velocity == velocity, "signed entry velocity binding")
    impedance, regularization, inverse, diag, aref, aref_scale = _coefficients(
        law, distance, velocity, step
    )
    _rounded(row.impedance, impedance, "limit impedance")
    _rounded(row.regularization, regularization, "limit regularization")
    _rounded(row.inverse_regularization, inverse, "limit inverse regularization")
    _rounded(row.diag_approx, diag, "limit diagonal approximation")
    _rounded(row.reference_acceleration, aref, "limit reference acceleration", aref_scale)

    common = (vout - vin) / step
    _require(np.isfinite(common).all(), "finite common acceleration on every DOF")
    damped_native = _damped_native_limit(plan)
    if not damped_native:
        _require(
            np.all(np.abs(vout - (vin + step * physical)) <= 5e-5 * (1.0 + np.abs(vout))),
            "physical acceleration and common velocity disagree on a DOF",
        )
    force = row.force
    physical_axis, common_axis = sign * float(physical[dof]), sign * float(common[dof])
    physical_gradient = physical_axis + row.regularization * force - row.reference_acceleration
    physical_scale = (
        abs(physical_axis) + abs(row.regularization * force) + abs(row.reference_acceleration)
    )
    _rounded(row.gradient, physical_gradient, "limit physical gradient", physical_scale)
    _lower_kkt(physical_gradient, force, tolerance, physical_scale, "physical-qacc")
    law_axis = physical_axis if damped_native else common_axis
    law_gradient = law_axis + regularization * force - aref
    law_scale = abs(law_axis) + abs(regularization * force) + abs(aref)
    _lower_kkt(law_gradient, force, tolerance, law_scale, "source-law")
    predicted = max(0.0, (aref - law_axis) / regularization)
    _require(np.isfinite(predicted), "finite predicted force")
    _require(abs(force - predicted) <= 0.002 + 0.002 * abs(predicted), "predicted native force")
    return float(predicted)


def _resident_layout(plan, input_q):
    """Source-addressed FREE/scalar limit layout, independent of contact count."""
    nv, nq = plan.nv, plan.nq
    _require(
        getattr(plan, "resident_native_contacts", False)
        and plan.kind == 3
        and 6 <= nv <= 8
        and nq == nv + 1
        and plan.nfriction == 0
        and 0 <= plan.nlimit <= 2,
        "resident FREE/scalar source layout",
    )
    joints = plan.joint_ids
    _require(
        isinstance(joints, np.ndarray)
        and joints.dtype == np.int32
        and joints.shape == (nv - 5, 4)
        and np.count_nonzero(joints[:, 0] == 3) == 1
        and np.isin(joints[:, 0], (0, 1, 3)).all(),
        "complete resident native joint mapping",
    )
    q = _vector(input_q, "resident limit input q", nq)
    q32 = q.astype(np.float32)
    _require(np.isfinite(q32).all(), "finite packed resident q")
    entries, seen_qa, seen_dof = [], set(), set()
    previous = -1
    _require(
        plan.limit_ids.dtype == np.int32
        and plan.limit_ids.shape == (max(1, plan.nlimit), 3)
        and plan.limit_law.dtype == np.float32
        and plan.limit_law.shape == (max(1, plan.nlimit), 11),
        "resident source limit arrays",
    )
    for identity, law in zip(
        plan.limit_ids[: plan.nlimit], plan.limit_law[: plan.nlimit], strict=True
    ):
        joint, qa, dof = map(int, identity)
        _require(
            previous < joint < len(joints)
            and 0 <= qa < nq
            and 0 <= dof < nv
            and joints[joint, 0] in (0, 1)
            and (qa, dof) == tuple(map(int, joints[joint, 1:3]))
            and qa not in seen_qa
            and dof not in seen_dof,
            "resident ordered unique joint/qpos/DOF identities",
        )
        _require(
            np.isfinite(law).all() and law[3] > 0 and (law[4] > 0) == (law[5] > 0),
            "resident finite source limit law",
        )
        previous = joint
        seen_qa.add(qa)
        seen_dof.add(dof)
        entries.append((joint, qa, dof, tuple(map(float, law))))
    return nv, tuple(entries), q32


def expected_resident_limit_rows(plan, input_q):
    """All active (joint, DOF, side) rows in original native ordering.

    The compiled source owns margins and address layout. Both sides are retained
    if an admitted source has overlapping margins, exactly as T4 assembles them.
    """
    _, entries, q32 = _resident_layout(plan, input_q)
    return tuple(
        (joint, dof, side)
        for joint, qa, dof, law in entries
        for side, distance in _distances(q32, qa, law)
        if distance < law[2]
    )
