"""Independent HOST audit of accepted live area-friction receipts.

This validates trusted owned publication/checkpoint data before state replacement.
It performs bounded algebraic substitution only: no contact solve, native FK,
allocation of device resources, or runtime mutation. Historical configuration IDs
are provenance, not capabilities of a restored destination owner.
"""

from collections.abc import Mapping

import numpy as np

from ..geometry.area_pressure import (
    AreaPressureIdentity,
    AreaPressureQuadrature,
    AreaPressureStamp,
    _quadrature,
)
from .area_friction_law import LAW, PRECISION, capture_pair_laws, solver_schema
from .mujoco_area_guide import expected_scalar_rows, validate_guide_row
from .mujoco_area_native_contact import enabled as native_contact_enabled
from .mujoco_tyx._area_friction_layout import MAX_LIVE_DOF, friction_width
from .mujoco_tyx._constraint_publication import MuJoCoTyxConstraintRow
from .mujoco_tyx._model import fp32

SCHEMA = "mujoco-tyx-area-friction-live-v1"
TOLERANCE = 2e-5
PAIR_FIELDS = frozenset(
    (
        "source_quadrature",
        "identity",
        "stamp",
        "sample_ids",
        "source_ids",
        "body_ids",
        "dA",
        "points",
        "normals",
        "stiffness",
        "surrogate_gap",
        "jacobian_a",
        "jacobian_b",
        "law",
        "mu",
        "slip_speed_m_s",
        "normal_force",
        "tangential_force",
        "relative_velocity",
        "force_on_a",
        "force_on_b",
        "torque_on_a_world_origin",
        "torque_on_b_world_origin",
    )
)
SOLVER_FIELDS = frozenset(
    (
        "schema",
        "law",
        "precision",
        "mass_matrix",
        "damping",
        "input_velocity",
        "applied_force",
        "h_s",
        "solver_velocity",
        "output_velocity",
        "generalized_force",
        "residual",
        "normalized_residual",
        "tolerance",
        "solver_status",
        "raw_diagnostics",
    )
)


def _require(condition, message):
    if not condition:
        raise ValueError("MuJoCo TYX area-friction state: " + message)


def _array(value, shape, name, dtype=np.float64):
    _require(
        isinstance(value, np.ndarray)
        and value.dtype == dtype
        and value.shape == shape
        and np.isfinite(value).all(),
        name + " layout/finite",
    )
    return value


def _tag(value, expected):
    return type(value) is str and value == expected


def _scalar(value, name):
    _require(type(value) is float and np.isfinite(value), name + " finite float")
    return value


def _same(left, right, name):
    _require(
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C"),
        name + " differs",
    )


def _close(got, expected, name, *, absolute=0.002, relative=0.002):
    _require(
        np.isfinite(expected).all()
        and np.all(np.abs(got - expected) <= absolute + relative * np.abs(expected)),
        name,
    )


def _packed(value, shape, name):
    value = _array(value, shape, name)
    _same(value, value.astype(np.float32).astype(np.float64), name + " FP32 decoding")
    return value


def _diagnostics(raw, solver_velocity, nv):
    _require(
        isinstance(raw, Mapping)
        and len(raw) == 4
        and set(raw) == {"meta", "scalars", "trace_meta", "trace"},
        "raw diagnostics fields",
    )
    meta = _array(raw["meta"], (5,), "diagnostic meta", np.int32)
    scalars = _array(raw["scalars"], (3,), "diagnostic scalars", np.float32)
    tm, trace = raw["trace_meta"], raw["trace"]
    _require(
        isinstance(tm, np.ndarray) and tm.ndim == 2 and 2 <= tm.shape[0] <= 290, "trace extent"
    )
    tm = _array(tm, (tm.shape[0], 4), "trace meta", np.int32)
    width = friction_width(nv)
    residual_start = 3 + width
    trace = _array(trace, (tm.shape[0], residual_start + width), "trace", np.float32)
    _require(sum(x.size for x in (meta, scalars, tm, trace)) <= 8192, "diagnostic capacity")
    count = int(meta[2])
    _require(
        meta[0] == 0
        and 0 <= meta[1] <= 24
        and 2 <= count <= len(tm)
        and 0 <= meta[3] <= 288
        and 0 <= meta[4] <= 24 * width,
        "solver metadata",
    )
    _require(
        scalars[0] >= 0 and scalars[1] > 0 and scalars[2] >= 0 and scalars[0] <= scalars[1],
        "device residual gate",
    )
    _require(
        np.all(tm[count:] == 0)
        and np.all(trace[count:] == 0)
        and np.all(trace[:count, 3 + nv : residual_start] == 0)
        and np.all(trace[:count, residual_start + nv :] == 0),
        "trace padding",
    )
    _require(
        np.all((tm[:count, 2] >= 0) & (tm[:count, 2] <= 2))
        and np.all((tm[:count, 3] >= 0) & (tm[:count, 3] <= 2))
        and tm[0, 2] == 0
        and tuple(tm[count - 1, 2:]) == (2, 2),
        "trace phases",
    )
    _same(
        trace[count - 1, 3 : 3 + nv].astype(np.float64),
        solver_velocity,
        "final trace/solver velocity",
    )
    _require(
        trace[count - 1, 1] == scalars[0] and trace[count - 1, 2] == scalars[1],
        "final trace diagnostics",
    )


def _validate(runtime, key, receipt, state):
    from ._mujoco_tyx_runtime import MuJoCoTyxDynamics, MuJoCoTyxStepReceipt

    laws = capture_pair_laws(runtime._plan.area_pair_names, runtime._area_friction_laws)
    _require(bool(laws), "missing opt-in laws")
    if receipt is None:
        return None
    _require(isinstance(receipt, MuJoCoTyxStepReceipt), "expected TYX receipt")
    _require(
        type(receipt.world) is int
        and receipt.world == key
        and type(receipt.tick) is int
        and 1 <= receipt.tick <= 2147483647,
        "world/tick",
    )
    _require(
        type(receipt.contact_law) is str
        and receipt.contact_law == LAW
        and _tag(receipt.numerical_solver, "area-friction-newton-v1")
        and _tag(receipt.source_solver, runtime._plan.source_solver)
        and _tag(receipt.backend, "mujoco-tyx-" + runtime._mode)
        and _tag(receipt.state_precision, "fp32-device-integration"),
        "solver/source binding",
    )
    time = _scalar(receipt.input_time, "input time")
    end = _scalar(receipt.output_time, "output time")
    _require(
        time >= 0
        and end > time
        and end == state.time
        and _scalar(receipt.contact_time, "contact time") == time
        and _tag(receipt.contact_phase, "tyx-euler-input-constraint-evaluation"),
        "receipt phase",
    )
    _require(_scalar(receipt.h, "h") == runtime.timestep and receipt.h > 0, "timestep binding")
    h = float(fp32(receipt.h, "historical friction h").item())
    native_geometry = native_contact_enabled(runtime._plan)
    _require(
        h > 0
        and type(receipt.contacts) is tuple
        and (native_geometry or not receipt.contacts)
        and type(receipt.constraints) is tuple,
        "native contacts and constraint rows",
    )
    fields = receipt.raw_contact_fields
    expected_fields = {"area_friction", "area_friction_solver", "response_operator"}
    if native_geometry:
        expected_fields |= {"native_contacts", "native_contact_operator"}
    _require(
        isinstance(fields, Mapping)
        and set(fields) == expected_fields
        and _tag(fields["response_operator"], "M+h*joint_damping"),
        "raw receipt fields",
    )
    if native_geometry:
        _require(_tag(fields["native_contact_operator"], "M"), "native physical-M operator")
    pairs, solver = fields["area_friction"], fields["area_friction_solver"]
    _require(type(pairs) is tuple and len(pairs) == len(laws), "complete pair records")
    solver_fields = SOLVER_FIELDS
    if runtime._plan.two_native_supports:
        solver_fields = solver_fields | {"applied_with_contact"}
    _require(
        isinstance(solver, Mapping) and set(solver) == solver_fields,
        "solver record fields",
    )
    _require(
        _tag(solver["schema"], solver_schema(runtime._plan))
        and _tag(solver["law"], LAW)
        and _tag(solver["precision"], PRECISION)
        and type(solver["solver_status"]) is int
        and solver["solver_status"] == 0,
        "solver schema/status",
    )
    nv, nq, nbody = runtime._plan.nv, runtime._plan.nq, runtime._model.nbody
    _require(1 <= nv <= MAX_LIVE_DOF and 1 <= nq <= 45 and 1 <= nbody <= 128, "model capacity")
    ids = _array(runtime._plan.area_body_ids, (len(laws), 2), "declared body IDs", np.int32)
    _require(
        np.all(ids >= 0) and np.all(ids < nbody) and np.all(ids[:, 0] != ids[:, 1]),
        "declared body indices",
    )
    dyn = receipt.dynamics
    _require(
        type(dyn) is MuJoCoTyxDynamics
        and _tag(dyn.phase, "tyx-euler-input-dynamics")
        and _scalar(dyn.time, "dynamics time") == time,
        "same-entry dynamics",
    )
    shapes = dict(
        body_position=(nbody, 3),
        body_rotation=(nbody, 3, 3),
        body_com=(nbody, 3),
        jacobian_com=(nbody, 6, nv),
        mass_matrix=(nv, nv),
        bias=(nv,),
        passive=(nv,),
        applied=(nv,),
        smooth_acceleration=(nv,),
        input_q=(nq,),
        input_v=(nv,),
    )
    data = {name: _packed(getattr(dyn, name), shape, name) for name, shape in shapes.items()}
    _same(data["input_q"], _array(receipt.input_q, (nq,), "receipt q"), "entry q")
    _same(data["input_v"], _array(receipt.input_v, (nv,), "receipt v"), "entry v")
    _array(state.qpos, (nq,), "endpoint q")
    output = _packed(solver["output_velocity"], (nv,), "output velocity")
    _same(output, _array(state.qvel, (nv,), "endpoint v"), "endpoint velocity")
    velocity = _packed(solver["solver_velocity"], (nv,), "solver velocity")
    _close(output, velocity, "actual/solver velocity", absolute=5e-5, relative=5e-5)
    mass = _packed(solver["mass_matrix"], (nv, nv), "solver M")
    _same(mass, data["mass_matrix"], "physical M")
    _require(np.array_equal(mass, mass.T), "physical M symmetry")
    np.linalg.cholesky(mass)
    damping = _packed(solver["damping"], (nv,), "damping")
    expected_damping = fp32(runtime._model.dof_damping, "joint damping").astype(np.float64)
    _same(damping, expected_damping, "source damping")
    _require(np.all(damping >= 0), "negative damping")
    v0 = _packed(solver["input_velocity"], (nv,), "solver input velocity")
    _same(v0, data["input_v"], "solver entry velocity")
    _require(_scalar(solver["h_s"], "solver h") == h, "actual packed h")
    base = data["applied"] + data["passive"] - data["bias"]
    # Allow ordinary FP32 roundoff/FMA, while binding the actual stored input.
    expected_f = base + damping * v0
    f_bound = (
        8
        * np.finfo(np.float32).eps
        * (
            np.abs(data["applied"])
            + np.abs(data["passive"])
            + np.abs(data["bias"])
            + np.abs(damping * v0)
        )
        + np.finfo(np.float32).tiny
    )
    external = _packed(solver["applied_force"], (nv,), "solver applied force")
    _require(np.all(np.abs(external - expected_f) <= f_bound), "packed applied+passive-bias+Dv0")
    response = mass + h * np.diag(damping)
    factor = np.linalg.cholesky(response)
    free = np.linalg.solve(response, mass @ v0 + h * external)
    scale = 1.0 + float(np.linalg.norm(factor.T @ free))
    _close(
        h * data["smooth_acceleration"],
        free - v0,
        "free acceleration response",
        absolute=5e-5,
        relative=5e-5,
    )
    acceleration = _packed(receipt.integration_acceleration, (nv,), "integration acceleration")
    _close(
        output, v0 + h * acceleration, "actual velocity integration", absolute=5e-5, relative=5e-5
    )
    total = _packed(solver["generalized_force"], (nv,), "stored generalized force")
    _same(total, _array(receipt.qfrc_constraint, (nv,), "receipt constraint force"), "receipt Q")
    _same(total, _array(state.qfrc_constraint, (nv,), "state constraint force"), "state Q")
    qacc = _packed(receipt.qacc, (nv,), "physical acceleration")
    _same(qacc, _array(state.qacc, (nv,), "state acceleration"), "physical qacc")
    _close(mass @ qacc, base + total, "physical M acceleration")
    _close(response @ acceleration, base + total, "M+hD integration acceleration")
    if getattr(runtime._plan, "coupled_damped_contacts", False):
        # Every DOF participates in the complete M and A products above. In this
        # scalar-joint-only route q advances once in native coordinate order.
        _require(nq == nv == 2, "coupled native scalar coordinate layout")
        expected_q = (
            data["input_q"].astype(np.float32) + np.float32(h) * output.astype(np.float32)
        ).astype(np.float64)
        # Device FMA may differ by one rounding from separate HOST operations.
        q_scale = np.abs(data["input_q"]) + h * np.abs(output)
        q_bound = 4 * np.finfo(np.float32).eps * q_scale + np.finfo(np.float32).tiny
        _require(
            np.all(np.abs(state.qpos - expected_q) <= q_bound),
            "coupled native endpoint coordinate integration",
        )
    _same(
        _array(receipt.qfrc_actuator, (nv,), "receipt actuator force"),
        _array(state.qfrc_actuator, (nv,), "state actuator force"),
        "actuator publication",
    )
    _diagnostics(solver["raw_diagnostics"], velocity, nv)
    count, common_stamp = 0, None
    summed, law_total = np.zeros(nv), np.zeros(nv)
    if native_geometry:
        from .mujoco_area_native_contact import validate as validate_native

        summed, law_total = validate_native(
            runtime,
            fields["native_contacts"],
            receipt.contacts,
            receipt.constraints,
            dyn,
            qacc,
            h,
            output_v=output,
        )
        owners, joint_rows = (), ()
    else:
        owners = expected_scalar_rows(runtime._plan, data["input_q"])
        joint_rows = receipt.constraints
        _require(len(joint_rows) == len(owners), "complete native scalar row count")
    for owner, row in zip(owners, joint_rows, strict=True):
        _require(type(row) is MuJoCoTyxConstraintRow, "native scalar row type")
        _require(
            (row.constraint_type, row.source_id, row.side_or_edge, row.geometry_slot, row.dof)
            == owner,
            "native scalar row identity or ordering",
        )
        if owner[0] == 1:
            predicted = validate_guide_row(runtime._plan, row, v0, output, h, qacc)
        else:
            from .mujoco_area_limit import validate_limit_row

            predicted = validate_limit_row(runtime._plan, row, data["input_q"], v0, output, h, qacc)
        summed += row.jacobian * row.force
        law_total += row.jacobian * predicted
    for raw, law, body_ids in zip(pairs, laws, ids, strict=True):
        _require(
            isinstance(raw, Mapping) and len(raw) == len(PAIR_FIELDS) and set(raw) == PAIR_FIELDS,
            "pair record fields",
        )
        original = raw["source_quadrature"]
        _require(
            type(original) is AreaPressureQuadrature and type(original.sample_ids) is tuple,
            "source quadrature",
        )
        n = len(original.sample_ids)
        _require(
            type(original.source_ids) is tuple and len(original.source_ids) == n,
            "source identity extent",
        )
        count += n
        _require(
            count + len(receipt.constraints) <= runtime._constraint_capacity,
            "sample capacity",
        )
        for name, shape in (
            ("points", (n, 3)),
            ("normals", (n, 3)),
            ("dA", (n,)),
            ("pressure", (n,)),
            ("gradient", (n,)),
        ):
            _array(getattr(original, name), shape, "source " + name)
        quad = _quadrature(original)
        if common_stamp is None:
            common_stamp = quad.stamp
        _require(
            type(raw["identity"]) is AreaPressureIdentity
            and type(raw["stamp"]) is AreaPressureStamp
            and quad.identity == law.identity
            and raw["identity"] == quad.identity
            and raw["stamp"] == quad.stamp
            and quad.stamp == common_stamp,
            "complete identity/shared historical stamp",
        )
        _require(
            quad.stamp.world_id == str(key)
            and quad.stamp.step_index == receipt.tick - 1
            and quad.stamp.sample_time_s == time,
            "historical sample phase",
        )
        _require(
            type(raw["sample_ids"]) is tuple
            and type(raw["source_ids"]) is tuple
            and raw["sample_ids"] == quad.sample_ids
            and raw["source_ids"] == quad.source_ids
            and type(raw["body_ids"]) is tuple
            and all(type(x) is int for x in raw["body_ids"])
            and raw["body_ids"] == tuple(map(int, body_ids)),
            "sample/body identities",
        )
        _require(_tag(raw["law"], LAW), "pair law")
        mu, slip = (
            float(fp32(law.mu, "mu").item()),
            float(fp32(law.slip_speed_m_s, "slip speed").item()),
        )
        _require(
            _scalar(raw["mu"], "mu") == mu
            and _scalar(raw["slip_speed_m_s"], "slip speed") == slip
            and mu >= 0
            and slip > 0,
            "packed pair parameters",
        )
        converted = fp32(
            np.column_stack(
                (quad.points, quad.normals, quad.dA * quad.gradient, -quad.pressure / quad.gradient)
            ),
            "source samples",
        ).astype(np.float64)
        _require(np.all(converted[:, 6] > 0) and np.all(converted[:, 7] <= 0), "sample law domain")
        _same(_array(raw["dA"], (n,), "dA"), quad.dA, "original dA")
        for name, expected in (
            ("points", converted[:, :3]),
            ("normals", converted[:, 3:6]),
            ("stiffness", converted[:, 6]),
            ("surrogate_gap", converted[:, 7]),
        ):
            _same(_array(raw[name], expected.shape, name), expected, "packed source " + name)
        points, normals = raw["points"], raw["normals"]
        norm2 = np.sum(normals * normals, axis=1)
        _require(np.all(np.abs(norm2 - 1.0) <= 2e-6), "packed normal length")
        ja = _packed(raw["jacobian_a"], (n, 3, nv), "point JA")
        jb = _packed(raw["jacobian_b"], (n, 3, nv), "point JB")
        for actual, body in ((ja, int(body_ids[0])), (jb, int(body_ids[1]))):
            expected = np.asarray(
                [
                    data["jacobian_com"][body, :3]
                    + np.cross(data["jacobian_com"][body, 3:].T, p - data["body_com"][body]).T
                    for p in points
                ]
            ).reshape(n, 3, nv)
            _close(actual, expected, "same-entry point Jacobian", absolute=2e-6, relative=2e-5)
        relative = np.einsum("sij,j->si", ja - jb, output)
        _close(
            _array(raw["relative_velocity"], (n, 3), "relative velocity"),
            relative,
            "actual relative velocity",
            absolute=5e-5,
            relative=5e-5,
        )
        vn = np.sum(normals * relative, axis=1)
        tangent = np.cross(normals, np.cross(relative, normals)) / norm2[:, None]
        predicted_fn = np.maximum(0.0, -raw["stiffness"] * (raw["surrogate_gap"] + h * vn))
        predicted_ft = (
            -mu
            * predicted_fn[:, None]
            * tangent
            / np.sqrt(np.sum(tangent * tangent, axis=1) + slip * slip)[:, None]
        )
        predicted_force = predicted_fn[:, None] * normals + predicted_ft
        fn = _packed(raw["normal_force"], (n,), "normal force")
        ft = _packed(raw["tangential_force"], (n, 3), "tangential force")
        fa = _packed(raw["force_on_a"], (n, 3), "force A")
        fb = _packed(raw["force_on_b"], (n, 3), "force B")
        _require(np.all(fn >= 0), "negative normal force")
        _close(fn, predicted_fn, "final-normal force law")
        _close(ft, predicted_ft, "final-normal tangential law")
        _close(fa, predicted_force, "final-velocity total force")
        _close(fa, fn[:, None] * normals + ft, "normal/tangent force sum")
        _same(fb, -fa, "action/reaction force")
        ma = _packed(raw["torque_on_a_world_origin"], (n, 3), "moment A")
        mb = _packed(raw["torque_on_b_world_origin"], (n, 3), "moment B")
        _close(ma, np.cross(points, fa), "stored force/world moment", absolute=1e-5)
        _close(ma, np.cross(points, predicted_force), "final law/world moment", absolute=1e-5)
        _same(mb, -ma, "action/reaction moment")
        # A declared pair with no quadrature samples contributes exactly zero.
        if n:
            summed += np.einsum("sij,si->j", ja - jb, fa)
            law_total += np.einsum("sij,si->j", ja - jb, predicted_force)
    _close(total, summed, "complete two-sided generalized force")
    if runtime._plan.two_native_supports:
        net = _packed(solver["applied_with_contact"], (nv,), "integrator applied/contact load")
        _close(
            net,
            data["applied"] + summed,
            "physical load before cancellation",
            absolute=2e-5,
            relative=2e-5,
        )
        _close(
            mass @ qacc,
            net + data["passive"] - data["bias"],
            "actual integrated net load",
            absolute=2e-5,
            relative=2e-5,
        )
    residual = response @ output - mass @ v0 - h * (external + total)
    saved_residual = _array(solver["residual"], (nv,), "actual residual")
    _close(saved_residual, residual, "stored-Q residual binding", absolute=1e-12, relative=1e-10)
    norm = float(np.linalg.norm(np.linalg.solve(factor, residual)))
    normalized = norm / scale
    _require(_scalar(solver["tolerance"], "tolerance") == TOLERANCE, "fixed FP32 audit gate")
    saved_normalized = _scalar(solver["normalized_residual"], "normalized residual")
    _require(saved_normalized >= 0, "negative normalized residual")
    _close(
        saved_normalized, normalized, "normalized residual binding", absolute=1e-12, relative=1e-10
    )
    law_residual = response @ output - mass @ v0 - h * (external + law_total)
    law_norm = float(np.linalg.norm(np.linalg.solve(factor, law_residual)))
    _require(
        normalized <= TOLERANCE and law_norm / scale <= TOLERANCE,
        "actual-velocity stored-force/law residual gate",
    )
    return data["body_position"], data["body_com"]


def validate_friction_receipt(runtime, key, receipt, state):
    """Audit one complete historical receipt without modifying any owner.

    Returns the original entry body origins and COM arrays, or ``None`` when no
    accepted receipt exists. The engineering qualification is FP32 (2e-5); no
    strict 1e-10 contact-solver qualification is inferred from this audit.
    """
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _validate(runtime, key, receipt, state)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        raise ValueError(
            "MuJoCo TYX area-friction state: nonfinite/invalid audit algebra"
        ) from error
