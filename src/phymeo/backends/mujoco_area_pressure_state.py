"""HOST owned area observations and asset-bound scene checkpoint envelopes.

Accepted device q/v and native entry banks remain owned by the standard runtime
checkpoint. Area records here are HOST publication data, never device tokens.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..geometry.area_pressure import AreaPressureQuadrature
from .mujoco_runtime import _frozen

AREA_SCHEMA = "mujoco-tyx-area-receipt-v1"
SCENE_MAGIC = b"PHYMEOTYXAREASCENE1\n"


def _require(ok, message):
    if not ok:
        raise ValueError("MuJoCo TYX area state: " + message)


def area_binding(runtime):
    if runtime._area_friction_laws:
        from .area_friction_law import law_binding

        return dict(
            schema="mujoco-tyx-area-friction-receipt-v1",
            pairs=runtime._plan.area_pair_names,
            body_ids=tuple(tuple(map(int, pair)) for pair in runtime._plan.area_body_ids),
            parameters=law_binding(runtime._area_friction_laws),
            response_operator="M+h*joint_damping",
            **(
                {
                    "native_contact_operator": "M",
                    "native_contact_scope": (
                        f"coupled-damped-geometry-{runtime._plan.coupled_contact_variant}"
                        if getattr(runtime._plan, "coupled_damped_contacts", False)
                        else "scalar-damped-geometry-v1"
                    ),
                }
                if runtime._plan.damped_contact_dofs
                else {}
            ),
            storage="host-owned-accepted-receipt; area-entry-v2",
        )
    return dict(
        schema=AREA_SCHEMA,
        pairs=runtime._plan.area_pair_names,
        body_ids=tuple(tuple(map(int, pair)) for pair in runtime._plan.area_body_ids),
        law=runtime.capabilities["area_pressure_law"],
        response_operator="M+h*joint_damping",
        **({"native_contact_operator": "M"} if runtime._plan.scalar_damped_contacts else {}),
        storage="host-owned-accepted-receipt; native-entry-layout-unchanged",
    )


def _array(value, shape, name):
    _require(
        isinstance(value, np.ndarray)
        and value.dtype == np.float64
        and value.shape == shape
        and np.isfinite(value).all(),
        name + " array",
    )
    return value


def validate_area_receipt(runtime, key, receipt, state):
    """Check redundant historical publication before any replacement allocation."""
    if runtime._area_friction_laws:
        from .mujoco_area_friction_state import validate_friction_receipt

        return validate_friction_receipt(runtime, key, receipt, state)
    if not runtime._area_pressure_pairs:
        return
    if receipt is None:
        return
    _require(
        receipt.world == key
        and receipt.output_time == state.time
        and receipt.tick > 0
        and receipt.h == runtime.timestep,
        "receipt endpoint",
    )
    _require(receipt.dynamics is not None, "missing input dynamics")
    _require(
        receipt.contact_time == receipt.input_time
        and receipt.contact_phase == "tyx-euler-input-constraint-evaluation"
        and receipt.dynamics.phase == "tyx-euler-input-dynamics",
        "input phase",
    )
    fields = receipt.raw_contact_fields
    _require(isinstance(fields, Mapping) and "area_pressure" in fields, "missing area records")
    pairs = fields["area_pressure"]
    _require(
        type(pairs) is tuple and len(pairs) == len(runtime._plan.area_pair_names), "area pair count"
    )
    nbody, nv = runtime._model.nbody, runtime._plan.nv
    dynamics = receipt.dynamics
    origins = _array(dynamics.body_position, (nbody, 3), "input body origins")
    com = _array(dynamics.body_com, (nbody, 3), "input COM")
    body_j = _array(dynamics.jacobian_com, (nbody, 6, nv), "input Jacobians")
    _require(
        np.array_equal(dynamics.input_q, receipt.input_q)
        and np.array_equal(dynamics.input_v, receipt.input_v)
        and dynamics.time == receipt.input_time,
        "dynamics phase",
    )
    _array(receipt.input_v, (nv,), "input velocity")
    _array(receipt.integration_acceleration, (nv,), "integration acceleration")
    h = float(np.float32(runtime.timestep))
    _require(
        np.allclose(
            state.qvel, receipt.input_v + h * receipt.integration_acceleration, rtol=2e-6, atol=2e-7
        ),
        "motion publication",
    )
    _require(fields.get("response_operator") == "M+h*joint_damping", "response operator")
    if runtime._plan.scalar_damped_contacts:
        _require(
            fields.get("native_contact_operator") == "M"
            and fields.get("solver_diagnostics_basis") == "physical-acceleration-scaled-area-rows",
            "scalar mixed contact operators",
        )
        mass = float(dynamics.mass_matrix[0, 0])
        effective = mass + h * float(runtime._model.dof_damping[0])
        _require(
            np.allclose(
                mass * receipt.qacc,
                effective * receipt.integration_acceleration,
                rtol=2e-5,
                atol=2e-6,
            ),
            "scalar mixed acceleration",
        )
    response = _array(fields.get("area_response_acceleration"), (nv,), "response acceleration")
    _require(
        np.allclose(response, receipt.integration_acceleration, rtol=2e-5, atol=2e-6),
        "response/integration acceleration",
    )
    cap, width = max(1, runtime._constraint_capacity), max(8, nv)
    for name, shape, dtype in (
        ("efc_ids", (cap, 5), np.int32),
        ("efc_data", (cap, 14), np.float32),
        ("efc_J", (cap, width), np.float32),
        ("counts", (3,), np.int32),
    ):
        value = fields.get(name)
        _require(
            isinstance(value, np.ndarray)
            and value.dtype == dtype
            and value.shape == shape
            and np.isfinite(value).all(),
            "raw " + name,
        )
    _require(fields["counts"][2] == len(receipt.constraints), "raw constraint count")
    for i, row in enumerate(receipt.constraints):
        _require(
            tuple(fields["efc_ids"][i])
            == (row.constraint_type, row.source_id, row.side_or_edge, row.geometry_slot, row.dof),
            "raw constraint IDs",
        )
        _require(np.array_equal(fields["efc_J"][i, :nv], row.jacobian), "raw constraint Jacobian")
        expected = (
            row.position,
            row.margin,
            row.velocity,
            row.reference_acceleration,
            row.regularization,
            row.inverse_regularization,
            row.impedance,
            row.lower,
            0.0 if row.upper is None else row.upper,
            row.force,
            row.gradient,
            1.0,
            float(row.upper is not None),
            row.diag_approx,
        )
        _require(
            np.array_equal(fields["efc_data"][i].astype(np.float64), expected),
            "raw constraint values",
        )
    area_rows = tuple(row for row in receipt.constraints if row.constraint_type == 7)
    cursor = 0
    keys = {
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
        "normal_jacobian",
        "normal_force",
        "force_on_a",
        "force_on_b",
        "torque_on_a_world_origin",
        "torque_on_b_world_origin",
        "source_quadrature",
    }
    for raw, names, ids in zip(
        pairs, runtime._plan.area_pair_names, runtime._plan.area_body_ids, strict=True
    ):
        _require(isinstance(raw, Mapping) and set(raw) == keys, "area record fields")
        quad = raw["source_quadrature"]
        _require(type(quad) is AreaPressureQuadrature, "original quadrature missing")
        _require(
            raw["identity"] == quad.identity and raw["stamp"] == quad.stamp, "source identity/stamp"
        )
        _require(
            (quad.identity.compliant_body, quad.identity.rigid_body) == names,
            "ordered pair binding",
        )
        _require(
            quad.stamp.world_id == str(key)
            and quad.stamp.step_index == receipt.tick - 1
            and quad.stamp.sample_time_s == receipt.input_time,
            "historical stamp",
        )
        # configuration_id is retained provenance, not a current destination capability.
        _require(
            raw["sample_ids"] == quad.sample_ids and raw["source_ids"] == quad.source_ids,
            "source sample identities",
        )
        ba, bb = map(int, ids)
        _require(raw["body_ids"] == (ba, bb), "body IDs")
        n = len(quad.dA)
        _require(cursor + n <= runtime._constraint_capacity, "sample capacity")
        from .mujoco_tyx._model import fp32

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            converted = fp32(
                np.column_stack(
                    (
                        quad.points,
                        quad.normals,
                        quad.dA * quad.gradient,
                        -quad.pressure / quad.gradient,
                    )
                ),
                "historical area samples",
            ).astype(np.float64)
        _require(np.all(converted[:, 6] > 0), "historical stiffness")
        expected = {
            "dA": quad.dA,
            "points": converted[:, :3],
            "normals": converted[:, 3:6],
            "stiffness": converted[:, 6],
            "surrogate_gap": converted[:, 7],
        }
        for name, value in expected.items():
            got = _array(raw[name], value.shape, name)
            _require(np.array_equal(got, value), "quadrature conversion: " + name)
        point, normal = raw["points"], raw["normals"]
        j = _array(raw["normal_jacobian"], (n, nv), "normal Jacobian")
        force = _array(raw["normal_force"], (n,), "normal force")
        _require(np.all(force >= 0), "negative normal force")
        wanted_j = []
        for p, direction in zip(point, normal, strict=True):
            ja = body_j[ba, :3] + np.cross(body_j[ba, 3:].T, p - com[ba]).T
            jb = body_j[bb, :3] + np.cross(body_j[bb, 3:].T, p - com[bb]).T
            wanted_j.append(direction @ (ja - jb))
        _require(
            np.allclose(j, np.asarray(wanted_j).reshape(n, nv), rtol=2e-5, atol=2e-6),
            "body point Jacobian",
        )
        predicted = np.maximum(0, -raw["stiffness"] * (raw["surrogate_gap"] + h * (j @ state.qvel)))
        _require(np.allclose(force, predicted, rtol=5e-4, atol=5e-4), "force/motion law")
        fa = normal * force[:, None]
        for name, wanted in (
            ("force_on_a", fa),
            ("force_on_b", -fa),
            ("torque_on_a_world_origin", np.cross(point, fa)),
            ("torque_on_b_world_origin", -np.cross(point, fa)),
        ):
            _require(np.array_equal(_array(raw[name], (n, 3), name), wanted), name)
        _require(cursor + n <= len(area_rows), "missing area constraint rows")
        for local, row in enumerate(area_rows[cursor : cursor + n]):
            _require(
                row.source_id == cursor + local
                and np.array_equal(row.jacobian, j[local])
                and row.position == raw["surrogate_gap"][local]
                and row.force == force[local],
                "constraint row binding",
            )
        cursor += n
    _require(cursor == len(area_rows), "extra area constraint rows")
    if cursor:
        _require(
            all(row.constraint_type == 7 for row in receipt.constraints[-cursor:]),
            "area rows must follow native rows",
        )
    total = sum((np.asarray(row.jacobian) * row.force for row in receipt.constraints), np.zeros(nv))
    _require(
        np.allclose(total, receipt.qfrc_constraint, rtol=3e-5, atol=3e-5),
        "complete generalized constraint load",
    )
    return origins, com


def observe(runtime, keys=None):
    """No launch or physics: own the last accepted entry and current endpoint."""
    with runtime._lock:
        runtime._check_open()
        _require(bool(runtime._area_pressure_pairs), "runtime has no area declaration")
        result = {}
        for key in runtime._select(keys):
            state, receipt = runtime._data[key], runtime._receipts[key]
            frame = dict(
                schema="mujoco-tyx-area-friction-observation-v1"
                if runtime._area_friction_laws
                else "mujoco-tyx-area-observation-v1",
                storage="host-owned-accepted-receipt",
                world=key,
                source_revision=runtime._checkpoint_revision,
                valid=receipt is not None,
                current_control=runtime._controls[key],
                endpoint_tick=runtime._ticks[key],
                endpoint_time=state.time,
                endpoint_q=_frozen(state.qpos),
                endpoint_v=_frozen(state.qvel),
            )
            if receipt is not None:
                origins, com = validate_area_receipt(runtime, key, receipt, state)
                pairs = []
                for raw in receipt.raw_contact_fields[
                    "area_friction" if runtime._area_friction_laws else "area_pressure"
                ]:
                    ba, bb = raw["body_ids"]
                    owned = dict(raw)
                    for side, body in (("a", ba), ("b", bb)):
                        force = raw["force_on_" + side]
                        owned["force_sum_" + side] = _frozen(force.sum(axis=0))
                        for label, origin in (
                            ("world", np.zeros(3)),
                            ("body", origins[body]),
                            ("com", com[body]),
                        ):
                            owned["moment_" + label + "_" + side] = _frozen(
                                np.cross(raw["points"] - origin, force).sum(axis=0)
                            )
                    pairs.append(MappingProxyType(owned))
                frame.update(
                    input_tick=receipt.tick - 1,
                    input_time=receipt.input_time,
                    input_q=_frozen(receipt.input_q),
                    input_v=_frozen(receipt.input_v),
                    output_tick=receipt.tick,
                    output_time=receipt.output_time,
                    applied_control=receipt.stored_control,
                    pairs=tuple(pairs),
                )
            else:
                frame["pairs"] = ()
            result[key] = MappingProxyType(frame)
        return MappingProxyType(result)


def asset_digest(assets):
    import json

    from .mujoco_checkpoint import _encode

    entries = []
    for item in assets:
        a, b = item.pressure_local, item.rigid_local
        entries.append(
            dict(
                identity=item.identity,
                pressure_domain=a.pressure_domain,
                nodes_a=a.node_ids,
                positions_a=a.world_vertices,
                cells=a.cells,
                nodal_pressure=a.nodal_pressure,
                cell_ids=a.cell_ids,
                nodes_b=b.node_ids,
                positions_b=b.world_vertices,
                faces=b.faces,
                face_ids=b.face_ids,
            )
        )
    raw = json.dumps(
        _encode(tuple(entries)), sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, init=False)
class MuJoCoTyxAreaSceneCheckpoint:
    """Trusted immutable envelope: local asset digest plus a runtime capsule."""

    _bytes: bytes

    def __init__(self, raw):
        from .mujoco_checkpoint import MAX_BYTES, MuJoCoTyxCheckpoint

        _require(isinstance(raw, (bytes, bytearray, memoryview)), "expected scene capsule bytes")
        size = raw.nbytes if isinstance(raw, memoryview) else len(raw)
        _require(size <= MAX_BYTES + 256, "scene capsule size")
        raw = bytes(raw)
        _require(raw.startswith(SCENE_MAGIC), "scene capsule header")
        digest, sep, payload = raw[len(SCENE_MAGIC) :].partition(b"\n")
        _require(
            sep and digest == hashlib.sha256(payload).hexdigest().encode("ascii"),
            "scene capsule checksum",
        )
        asset, sep, capsule = payload.partition(b"\n")
        _require(
            sep and len(asset) == 64 and all(c in b"0123456789abcdef" for c in asset),
            "scene asset digest",
        )
        MuJoCoTyxCheckpoint.from_bytes(capsule)
        object.__setattr__(self, "_bytes", raw)

    @classmethod
    def from_bytes(cls, raw):
        return cls(raw)

    @classmethod
    def make(cls, assets, checkpoint):
        payload = asset_digest(assets).encode("ascii") + b"\n" + checkpoint.to_bytes()
        return cls(
            SCENE_MAGIC + hashlib.sha256(payload).hexdigest().encode("ascii") + b"\n" + payload
        )

    def to_bytes(self):
        return self._bytes

    @property
    def asset_sha256(self):
        return self._bytes[len(SCENE_MAGIC) + 65 :].split(b"\n", 1)[0].decode("ascii")

    @property
    def runtime_checkpoint(self):
        from .mujoco_checkpoint import MuJoCoTyxCheckpoint

        return MuJoCoTyxCheckpoint.from_bytes(
            self._bytes[len(SCENE_MAGIC) + 65 :].split(b"\n", 1)[1]
        )
