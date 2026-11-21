"""Per-face evidence for the declared finite PL pressure/rigid-side material.

This is a HOST geometry/receipt adapter, not a dynamics or force predictor.
The first returned row audits the whole material; subsequent rows partition it
by the asset's stable rigid face IDs. Do not add that whole row to its parts.
Unknown archived face forces remain None. Archived whole forces retain their
original accepted step's input geometry and are never moved to its endpoint.
"""

from collections.abc import Mapping
from dataclasses import replace

import mujoco as mj
import numpy as np

from phymeo.backends.mujoco_area_pressure_scene import BodyAreaPressureAsset
from phymeo.control.manipulation_release_monitor import ReleasePairEvidence, ReleasePairKey
from phymeo.geometry.area_pressure import AreaPressureStamp, quadrature_area_pressure
from phymeo.geometry.area_pressure_intersection import (
    AreaPressureIntersection,
    _intersect_checked,
)

from .pusht_public_separation import _pair_witness, _world_vertices

DEPTH_DEFINITION = (
    "maximum p/(grad(p) dot outward rigid normal) over the vertices of each "
    "finite clipped PL contact polygon reconstructed by the same independent "
    "CPU FK at both endpoints; zero when no positive-area patch; "
    "the material's normal surrogate depth, not native capsule distance"
)
SPEED_DEFINITION = (
    "minimum n dot (J_a-J_b) v at CPU-reconstructed clipped polygon vertices; CPU point "
    "Jacobians and the specified phase's q/v, not implicit final contact velocity"
)
EVIDENCE_SCOPE = (
    "whole audit plus a fixed face partition (not additive across both levels); "
    "all monitor geometry at both endpoints uses independent CPU FK and "
    "exact-from-FP64 material clipping; live forces retain their receipt-bound "
    "device input geometry/source IDs, with CPU/device input face-contact "
    "disagreement explicitly rejected; device input depths/speeds are a separate "
    "diagnostic channel and never replace either CPU endpoint; strict separation "
    "requires positive outward-rounded SAT witnesses for every tetrahedron "
    "against each finite face; unknown is false, empty area alone is insufficient; "
    "CPU-pose witnesses do not bound CPU/device FK differences or establish CCD, "
    "undeclared material contact, physical calibration or task success"
)


def _require(condition, message):
    if not condition:
        raise ValueError("release material: " + message)


def _array(value, shape, label):
    raw = np.asarray(value)
    _require(raw.shape == shape and raw.dtype.kind in "fiu", label + " shape/type")
    result = np.asarray(raw, dtype=np.float64)
    _require(np.isfinite(result).all(), label + " finite values")
    return result


def _state(model, state):
    _require(isinstance(state, Mapping), "state mapping")
    q = _array(state["q"], (model.nq,), "state q")
    v = _array(state["v"], (model.nv,), "state v")
    time = state["time"]
    _require(
        not isinstance(time, (bool, np.bool_))
        and np.ndim(time) == 0
        and np.isfinite(time)
        and time >= 0,
        "state time",
    )
    return dict(q=q, v=v, time=float(time))


def _same_state(a, b):
    return (
        a["time"] == b["time"] and np.array_equal(a["q"], b["q"]) and np.array_equal(a["v"], b["v"])
    )


def _data(model, state):
    data = mj.MjData(model)
    data.qpos[:] = state["q"]
    data.qvel[:] = state["v"]
    data.time = state["time"]
    mj.mj_kinematics(model, data)
    mj.mj_comPos(model, data)
    _require(np.array_equal(data.qpos, state["q"]), "CPU FK changed supplied q")
    return data


def _world(asset, data, bodies):
    return tuple(
        _world_vertices(mesh.world_vertices, data.xmat[body].reshape(3, 3), data.xpos[body])
        for mesh, body in zip((asset.pressure_local, asset.rigid_local), bodies, strict=True)
    )


def _cpu_geometry(asset, data, bodies):
    transformed = _world(asset, data, bodies)
    pressure = replace(asset.pressure_local, world_vertices=transformed[0][0])
    rigid = replace(asset.rigid_local, world_vertices=transformed[1][0])
    # Both meshes were just constructed/validated; avoid validating them twice.
    geometry = _intersect_checked(
        pressure,
        rigid,
        identity=asset.identity,
        stamp=AreaPressureStamp("release-cpu", 0, float(data.time), "independent-FK"),
    )
    return geometry, transformed


def _phase(model, asset, data, bodies, geometry, *, transformed=None):
    _require(
        isinstance(geometry, AreaPressureIntersection)
        and geometry.surface.identity == asset.identity,
        "geometry identity/type",
    )
    faces = asset.rigid_local.face_ids
    values = {face: dict(depth=0.0, speed=None, contact=False, strict=False) for face in faces}
    ja, jb, angular = (np.empty((3, model.nv)) for _ in range(3))
    for triangle in geometry.surface.triangles:
        _require(
            triangle.rigid_face_id in values
            and triangle.compliant_cell_id in asset.pressure_local.cell_ids,
            "foreign material face/cell",
        )
        normal = np.cross(
            triangle.vertices[1] - triangle.vertices[0], triangle.vertices[2] - triangle.vertices[0]
        )
        magnitude = float(np.linalg.norm(normal))
        _require(np.isfinite(magnitude) and magnitude > 0, "positive contact patch area")
        normal /= magnitude
        gradient = float(triangle.pressure_gradient @ normal)
        _require(np.isfinite(gradient) and gradient > 0, "positive material normal gradient")
        depth = float(np.max(triangle.vertex_pressure) / gradient)
        _require(np.isfinite(depth) and depth >= 0, "finite nonnegative material depth")
        row = values[triangle.rigid_face_id]
        row["depth"] = max(row["depth"], depth)
        row["contact"] = True
        for point in triangle.vertices:
            mj.mj_jac(model, data, ja, angular, point, bodies[0])
            mj.mj_jac(model, data, jb, angular, point, bodies[1])
            speed = float(normal @ ((ja - jb) @ data.qvel))
            _require(np.isfinite(speed), "finite contact-point normal speed")
            row["speed"] = speed if row["speed"] is None else min(row["speed"], speed)
    if transformed is not None:
        # The tested SAT helper encloses transformations/projections of the given
        # CPU FP64 poses. A failed/zero gap is unknown, never a separation proof.
        cells, indices = asset.pressure_local.cells, asset.rigid_local.faces
        tet, tet_lo, tet_hi = (v[cells] for v in transformed[0])
        tri, tri_lo, tri_hi = (v[indices] for v in transformed[1])
        for j, face in enumerate(faces):
            strict = True
            for i in range(len(cells)):
                witness = _pair_witness(tet[i], tri[j], tet_lo[i], tet_hi[i], tri_lo[j], tri_hi[j])
                if witness is None or witness["gap_lower_m"] <= 0:
                    strict = False
                    break
            values[face]["strict"] = strict
            _require(
                not strict or not values[face]["contact"],
                "strict CPU separation contradicts its clipped contact",
            )
    speeds = [v["speed"] for v in values.values() if v["speed"] is not None]
    whole = dict(
        depth=max(v["depth"] for v in values.values()),
        speed=min(speeds) if speeds else None,
        contact=any(v["contact"] for v in values.values()),
        strict=all(v["strict"] for v in values.values()),
    )
    return whole, values


def _live_forces(asset, bodies, before, after, geometry, receipt):
    _require(
        np.array_equal(receipt.input_q, before["q"])
        and np.array_equal(receipt.input_v, before["v"])
        and receipt.input_time == before["time"],
        "receipt input state binding",
    )
    if hasattr(receipt, "output_time"):
        _require(receipt.output_time == after["time"], "receipt output time binding")
    pairs = receipt.raw_contact_fields["area_friction"]
    matches = [p for p in pairs if p["identity"] == asset.identity]
    _require(len(matches) == 1, "exactly one matching material receipt")
    pair = matches[0]
    _require(tuple(pair["body_ids"]) == bodies, "receipt material body order")
    quad = pair["source_quadrature"]
    actual = quadrature_area_pressure(geometry.surface)
    _require(
        quad.identity == asset.identity
        and quad.stamp == actual.stamp == pair["stamp"]
        and quad.stamp.sample_time_s == before["time"],
        "source geometry stamp",
    )
    _require(
        quad.sample_ids == actual.sample_ids == tuple(pair["sample_ids"])
        and quad.source_ids == actual.source_ids == tuple(pair["source_ids"]),
        "source face/sample mapping",
    )
    for name in ("points", "normals", "dA", "pressure", "gradient"):
        _require(
            np.array_equal(getattr(quad, name), getattr(actual, name)), "source quadrature " + name
        )
    _require(quad.total_area_m2 == actual.total_area_m2, "source quadrature area")
    count = len(quad.sample_ids)
    load = _array(pair["normal_force"], (count,), "actual normal loads")
    fa = _array(pair["force_on_a"], (count, 3), "actual force on A")
    fb = _array(pair["force_on_b"], (count, 3), "actual force on B")
    _require(np.all(load >= 0), "nonnegative actual normal loads")
    values = {}
    for face in asset.rigid_local.face_ids:
        mask = np.asarray([source[1] == face for source in quad.source_ids], dtype=bool)
        values[face] = (
            float(load[mask].sum()),
            tuple(fa[mask].sum(axis=0)),
            tuple(fb[mask].sum(axis=0)),
        )
    return (float(load.sum()), tuple(fa.sum(axis=0)), tuple(fb.sum(axis=0))), values


def _seed_forces(model, before, after, previous):
    _require(isinstance(previous, Mapping), "previous accepted seed record required")
    _require(
        _same_state(_state(model, previous["input_state"]), before)
        and _same_state(_state(model, previous["output_state"]), after),
        "previous seed state binding; old loads cannot be shifted to the next input",
    )
    measurement = previous["measurement"]
    _require(
        measurement["phase"] == "accepted-step-input"
        and measurement["time_s"] == before["time"]
        and measurement["native_contacts"] == 0,
        "previous area-only force phase",
    )
    area = previous["area"]
    load = float(_array(area["normal_load_n"], (), "previous whole normal load"))
    _require(load >= 0, "previous normal load must be nonnegative")
    fa = _array(area["total_force_on_pusher_world_n"], (3,), "previous force on A")
    fb = _array(measurement["group_force_world"], (3,), "previous force on B")
    return load, tuple(fa), tuple(fb)


def _geometry_report(asset, first, inputs, last, outputs, device):
    """Keep both input geometry definitions visible, including on disagreement."""

    def phase(values, *, device=False):
        if values is None:
            return None
        return dict(
            depth_m=values["depth"],
            normal_speed_m_s=values["speed"],
            contact=values["contact"],
            strictly_separated=None if device else values["strict"],
        )

    device_first, device_inputs = (None, {}) if device is None else device
    faces = asset.rigid_local.face_ids
    cpu_only = (
        []
        if device is None
        else [f for f in faces if inputs[f]["contact"] and not device_inputs[f]["contact"]]
    )
    device_only = (
        []
        if device is None
        else [f for f in faces if device_inputs[f]["contact"] and not inputs[f]["contact"]]
    )
    agreement = None if device is None else not (cpu_only or device_only)
    return dict(
        schema="release-material-geometries-v1",
        monitor_geometry="cpu-fk-exact-clip",
        device_geometry=None if device is None else "receipt-bound-input",
        device_speed_evaluation=(
            None
            if device is None
            else "CPU point Jacobians at receipt-bound device polygon vertices"
        ),
        status="seed_geometry"
        if device is None
        else "face_contact_agreement"
        if agreement
        else "face_contact_disagreement",
        input_face_contact_agreement=agreement,
        cpu_only_input_faces=cpu_only,
        device_only_input_faces=device_only,
        whole=dict(
            cpu_input=phase(first),
            cpu_output=phase(last),
            device_input=phase(device_first, device=True),
        ),
        faces=[
            dict(
                rigid_face_id=f,
                cpu_input=phase(inputs[f]),
                cpu_output=phase(outputs[f]),
                device_input=phase(device_inputs.get(f), device=True),
            )
            for f in faces
        ],
    )


def material_evidence(
    model,
    asset,
    before_state,
    after_state,
    *,
    geometry=None,
    receipt=None,
    previous=None,
    diagnostics=None,
):
    """Return one whole audit plus every actual rigid face in asset order.

    Live calls require both this step's geometry and receipt. A seed instead
    requires the previous saved *whole lane row*, with before/after matching
    that row's input/output; its unknown face loads are None, even for clear
    faces. Monitor depths, contact flags and normal speeds use CPU FK at both
    ends, so one accepted endpoint has one definition across consecutive calls.
    Actual forces still bind to device input geometry. An input face-contact
    disagreement raises, after filling optional ``diagnostics`` with both
    definitions; it cannot silently hide a new device contact or discard force.
    Apart from that optional output dict, inputs are not mutated. No dynamics
    stepping, q normalization or positive numerical tolerance is introduced.
    """
    _require(diagnostics is None or type(diagnostics) is dict, "diagnostics output must be a dict")
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            schema="release-material-geometries-v1",
            monitor_geometry="cpu-fk-exact-clip",
            status="not_evaluated",
        )
    _require(
        isinstance(model, mj.MjModel) and isinstance(asset, BodyAreaPressureAsset),
        "actual MuJoCo model and declared body material required",
    )
    _require(len(asset.rigid_local.face_ids) > 0, "nonempty fixed rigid face catalogue")
    bodies = tuple(
        int(model.body(name).id)
        for name in (asset.identity.compliant_body, asset.identity.rigid_body)
    )
    _require(bodies[0] != bodies[1], "distinct material bodies")
    before, after = _state(model, before_state), _state(model, after_state)
    _require(after["time"] > before["time"], "positive accepted step clock")
    live = geometry is not None or receipt is not None
    _require(
        (live and geometry is not None and receipt is not None and previous is None)
        or (not live and previous is not None),
        "one live or previous-seed force phase",
    )
    input_data, output_data = _data(model, before), _data(model, after)
    if live:
        whole_force, forces = _live_forces(asset, bodies, before, after, geometry, receipt)
    else:
        whole_force = _seed_forces(model, before, after, previous)
        forces = {face: (None, None, None) for face in asset.rigid_local.face_ids}
    input_geometry, input_transformed = _cpu_geometry(asset, input_data, bodies)
    first, inputs = _phase(
        model, asset, input_data, bodies, input_geometry, transformed=input_transformed
    )
    output_geometry, transformed = _cpu_geometry(asset, output_data, bodies)
    last, outputs = _phase(
        model, asset, output_data, bodies, output_geometry, transformed=transformed
    )
    device = _phase(model, asset, input_data, bodies, geometry) if live else None
    report = _geometry_report(asset, first, inputs, last, outputs, device)
    if diagnostics is not None:
        diagnostics.update(report)
    _require(
        report["input_face_contact_agreement"] is not False,
        "CPU/device input face contact disagreement: cpu_only="
        + repr(report["cpu_only_input_faces"])
        + ", device_only="
        + repr(report["device_only_input_faces"]),
    )
    material = (asset.identity.compliant_material, asset.identity.rigid_material)

    def evidence(identity, entry, endpoint, loads):
        return ReleasePairEvidence(
            ReleasePairKey("area-material", identity, bodies),
            entry["depth"],
            endpoint["depth"],
            loads[0],
            entry["speed"],
            endpoint["speed"],
            entry["contact"],
            endpoint["contact"],
            endpoint["strict"],
            loads[1],
            loads[2],
        )

    return (evidence((*material, "whole"), first, last, whole_force),) + tuple(
        evidence((*material, "face", face), inputs[face], outputs[face], forces[face])
        for face in asset.rigid_local.face_ids
    )
