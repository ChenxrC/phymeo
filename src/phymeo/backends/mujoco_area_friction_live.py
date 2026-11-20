"""Versioned live friction publication: inspect actual device results, never solve."""

from types import MappingProxyType

import numpy as np

from ..solvers.area_friction import LAW
from .area_friction_law import PRECISION, solver_schema
from .mujoco_area_guide import expected_scalar_rows
from .mujoco_area_native_contact import enabled as native_contact_enabled
from .mujoco_runtime import _frozen
from .mujoco_tyx._constraint_publication import decode_row
from .tyx_area_friction import PackedFrictionProblem, PackedFrictionSample, _certificate

SCHEMA = "mujoco-tyx-area-friction-live-v1"
DEVICE_SCHEMA = "mujoco-tyx-area-device-entry-v2"
TOLERANCE = 2e-5


def _require(ok, message):
    if not ok:
        raise ValueError("live area friction: " + message)


def _owned(value):
    value = np.asarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def publish(runtime, result, index):
    n = runtime._plan.nv
    if getattr(runtime._plan, "coupled_damped_contacts", False):
        _require(n == runtime._plan.nq == 2, "coupled native scalar coordinate layout")
    packet = runtime._area_packet
    ns = int(packet[2][index])
    sizes = result["friction_sizes"][index]
    meta = result["friction_meta"][index]
    _require(np.array_equal(sizes, (n, ns, 24, 12)), "solver sizes")
    _require(result["friction_pack_status"][index] == 0 and meta[0] == 0, "solver status")
    _require(0 <= ns <= 128, "sample count")
    mass = result["friction_mass"][index, :n, :n]
    inputs = result["friction_inputs"][index, :, :n]
    h = result["friction_steps"][index]
    samples = result["friction_samples"][index, :ns]
    ja = result["friction_ja"][index, :ns, :, :n]
    jb = result["friction_jb"][index, :ns, :, :n]
    contact = result["friction_contact"][index, :ns]
    vectors = result["friction_vectors"][index]
    output = result["v"][index, :n].astype(np.float64)
    qforce = result["constraint"][index, :n].astype(np.float64)
    for value in (mass, inputs, h, samples, ja, jb, contact, vectors, output, qforce):
        _require(np.isfinite(value).all(), "nonfinite candidate")
    _require(np.array_equal(mass, result["mass_matrix"][index, :n, :n]), "mass input binding")
    _require(np.array_equal(inputs[1], result["input_v"][index, :n]), "velocity input binding")
    _require(float(h) == float(np.float32(runtime.timestep)), "timestep binding")
    _require(np.array_equal(samples[:, :8], packet[0][index, :ns]), "surface input bytes")
    _require(
        np.allclose(vectors[1, :n], output, atol=5e-5, rtol=5e-5), "native integration velocity"
    )
    problem = PackedFrictionProblem(
        mass,
        inputs[0],
        inputs[1],
        inputs[2],
        h,
        tuple(
            PackedFrictionSample(
                samples[s, :3],
                samples[s, 3:6],
                ja[s],
                jb[s],
                samples[s, 6],
                samples[s, 7],
                samples[s, 8],
                samples[s, 9],
            )
            for s in range(ns)
        ),
    )
    law_norm, scale, expected, law_residual, _, _ = _certificate(problem, output)
    contacts, constraints, native_raw = (), (), None
    aggregate = np.zeros(n)
    if native_contact_enabled(runtime._plan):
        from .mujoco_area_native_contact import publish as publish_native

        contacts, constraints, native_raw = publish_native(runtime, result, index)
        _require(ns + len(constraints) <= runtime._constraint_capacity, "combined row capacity")
        aggregate = sum((row.jacobian * row.force for row in constraints), np.zeros(n))
        response = mass.astype(float) + float(h) * np.diag(inputs[0].astype(float))
        law_norm = float(
            np.linalg.norm(
                np.linalg.solve(np.linalg.cholesky(response), law_residual - float(h) * aggregate)
            )
        )
    elif runtime._plan.nfriction or runtime._plan.nlimit:
        owners = expected_scalar_rows(runtime._plan, result["input_q"][index, :n].astype(float))
        _require(
            np.array_equal(result["constraint_counts"][index], (0, 0, len(owners))),
            "native row count",
        )
        decoded = []
        for r, owner in enumerate(owners):
            ids = result["efc_ids"][index, r]
            data = result["efc_data"][index, r]
            jacobian = result["efc_jacobian"][index, r]
            _require(np.array_equal(ids, owner), "native scalar row identity")
            is_dry = owner[0] == 1
            _require(
                np.isfinite(data).all()
                and np.array_equal(data[11:13], (1, int(is_dry)))
                and (is_dry or data[8] == 0),
                "native bound flags",
            )
            sign = 1 if is_dry else -owner[2]
            _require(
                jacobian[owner[4]] == sign and np.count_nonzero(jacobian) == 1,
                "signed scalar jacobian",
            )
            row = decode_row(ids, data, jacobian[:n])
            aggregate += row.jacobian * row.force
            decoded.append(row)
        constraints = tuple(decoded)
        # D=0 for this branch. The complete receipt audit below publication
        # also substitutes the independent source-derived guide law.
        law_norm = float(
            np.linalg.norm(
                np.linalg.solve(
                    np.linalg.cholesky(mass.astype(float)), law_residual - float(h) * aggregate
                )
            )
        )
    _require(law_norm <= TOLERANCE * scale, "actual velocity constitutive equation")
    for s, predicted in enumerate(expected):
        fn, _, ft, force, moment = predicted
        _require(contact[s, 0] >= 0, "negative normal load")
        _require(np.allclose(contact[s, 0], fn, atol=0.002, rtol=0.002), "final normal force")
        _require(np.allclose(contact[s, 6:9], ft, atol=0.002, rtol=0.002), "final tangent force")
        _require(np.allclose(contact[s, 9:12], force, atol=0.002, rtol=0.002), "actual total force")
        _require(np.allclose(contact[s, 12:15], moment, atol=1e-5, rtol=0.002), "actual moment")
        aggregate += (ja[s].astype(float) - jb[s].astype(float)).T @ contact[s, 9:12].astype(float)
    _require(np.allclose(qforce, aggregate, atol=2e-5, rtol=2e-5), "stored generalized load")
    a = mass.astype(float) + float(h) * np.diag(inputs[0].astype(float))
    residual = (
        a @ output
        - mass.astype(float) @ inputs[1].astype(float)
        - float(h) * (inputs[2].astype(float) + qforce)
    )
    factor = np.linalg.cholesky(a)
    norm = float(np.linalg.norm(np.linalg.solve(factor, residual)))
    _require(norm <= TOLERANCE * scale, "actual stored-force equation")
    data = result["area_data"][index]
    _require(np.array_equal(data[:ns, :8], samples[:, :8]), "staged sample bytes")
    _require(np.array_equal(data[:ns, 8], contact[:, 0]), "staged Fn bytes")
    _require(np.array_equal(data[:ns, 9:15], contact[:, 9:15]), "staged force/moment bytes")
    _require(np.array_equal(data[:ns, 15:18], contact[:, 6:9]), "staged Ft bytes")
    _require(np.array_equal(data[:ns, 21:23], samples[:, 8:10]), "staged law bytes")
    pairs = []
    for pi, (quad, offset, count) in enumerate(packet[3][index]):
        sl = slice(offset, offset + count)
        row = data[sl]
        law = runtime._area_friction_laws[pi]
        params = runtime._plan.area_friction_parameters[pi].astype(float)
        _require(quad.identity == law.identity, "material identity")
        pairs.append(
            MappingProxyType(
                dict(
                    source_quadrature=quad,
                    identity=quad.identity,
                    stamp=quad.stamp,
                    sample_ids=quad.sample_ids,
                    source_ids=quad.source_ids,
                    body_ids=tuple(map(int, runtime._plan.area_body_ids[pi])),
                    dA=quad.dA,
                    points=_frozen(row[:, :3]),
                    normals=_frozen(row[:, 3:6]),
                    stiffness=_frozen(row[:, 6]),
                    surrogate_gap=_frozen(row[:, 7]),
                    normal_force=_frozen(row[:, 8]),
                    force_on_a=_frozen(row[:, 9:12]),
                    force_on_b=_frozen(-row[:, 9:12]),
                    torque_on_a_world_origin=_frozen(row[:, 12:15]),
                    torque_on_b_world_origin=_frozen(-row[:, 12:15]),
                    tangential_force=_frozen(row[:, 15:18]),
                    relative_velocity=_frozen(row[:, 18:21]),
                    law=LAW,
                    mu=float(params[0]),
                    slip_speed_m_s=float(params[1]),
                    jacobian_a=_frozen(ja[sl]),
                    jacobian_b=_frozen(jb[sl]),
                )
            )
        )
    solver = MappingProxyType(
        dict(
            schema=solver_schema(runtime._plan),
            law=LAW,
            precision=PRECISION,
            mass_matrix=_frozen(mass),
            damping=_frozen(inputs[0]),
            input_velocity=_frozen(inputs[1]),
            applied_force=_frozen(inputs[2]),
            h_s=float(h),
            solver_velocity=_frozen(vectors[1, :n]),
            output_velocity=_frozen(output),
            generalized_force=_frozen(qforce),
            residual=_frozen(residual),
            normalized_residual=norm / scale,
            tolerance=TOLERANCE,
            solver_status=int(meta[0]),
            raw_diagnostics=MappingProxyType(
                {
                    name: _owned(result["friction_" + name][index])
                    for name in ("meta", "scalars", "trace_meta", "trace")
                }
            ),
        )
    )
    return (
        contacts,
        constraints,
        MappingProxyType(
            dict(
                area_friction=tuple(pairs),
                area_friction_solver=solver,
                response_operator="M+h*joint_damping",
                **(
                    {"native_contacts": native_raw, "native_contact_operator": "M"}
                    if native_raw is not None
                    else {}
                ),
            )
        ),
    )


def validate_device_row(record, index, receipt):
    meta, ids, data, words = (
        record[k][index] for k in ("area_meta", "area_ids", "area_data", "area_measure_words")
    )
    if receipt is None:
        _require(not any(np.any(x) for x in (meta, ids, data, words)), "invalid entry history")
        return
    pairs = receipt.raw_contact_fields["area_friction"]
    count = sum(len(p["normal_force"]) for p in pairs)
    active = sum(int(np.count_nonzero(p["normal_force"] > 0)) for p in pairs)
    _require(np.array_equal(meta, (2, count, active, 0)), "version-2 entry metadata")
    _require(data.shape == (len(ids), 23) and count <= len(ids), "version-2 entry layout")
    _require(
        not np.any(ids[count:]) and not np.any(data[count:]) and not np.any(words[count:]),
        "entry padding",
    )
    offset = 0
    for pi, pair in enumerate(pairs):
        ns = len(pair["normal_force"])
        sl = slice(offset, offset + ns)
        expected_ids = np.zeros((ns, 6), np.int32)
        expected_ids[:, 0] = 1
        expected_ids[:, 1] = pi
        expected_ids[:, 2] = np.arange(offset, offset + ns)
        expected_ids[:, 3:5] = pair["body_ids"]
        expected_ids[:, 5] = 1
        expected = np.column_stack(
            (
                pair["points"],
                pair["normals"],
                pair["stiffness"],
                pair["surrogate_gap"],
                pair["normal_force"],
                pair["force_on_a"],
                pair["torque_on_a_world_origin"],
                pair["tangential_force"],
                pair["relative_velocity"],
                np.full(ns, pair["mu"]),
                np.full(ns, pair["slip_speed_m_s"]),
            )
        ).astype(np.float32)
        _require(np.array_equal(ids[sl], expected_ids), "entry IDs")
        _require(np.array_equal(data[sl], expected), "entry load bytes")
        exact = np.asarray(pair["dA"], dtype="<f8").view("<i4").reshape(ns, 2)
        _require(np.array_equal(words[sl], exact), "entry area measure bytes")
        offset += ns
