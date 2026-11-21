"""Original RR states through public current-mesh Scene.step.

The XML, five q/v/control inputs and native floor law are original RR fixtures.
The explicitly declared octagonal-prism PL pressure tool is a NEW material
geometry, not the capsule-distance midpoint material from the entry experiment.
Only the production scene advances states; CPU MuJoCo supplies entry coefficients.
"""

import json

import numpy as np
import pytest
from test_damped_coupled_mixed_area_friction import _value
from test_mujoco_area_friction_lifecycle import _same
from test_mujoco_area_friction_live import _surface
from test_mujoco_tyx_constraints import ROW_TOL, match_contacts

from experiments.area_pressure.capsule_box_reference import axis_box_clearance
from experiments.area_pressure.pusht_entry_rows import LABELS, assemble_entry, load_entries
from experiments.area_pressure.pusht_public_scene import IDENTITY, MU, SLIP, H, build_scene_spec
from experiments.area_pressure.pusht_public_scene import build_side_asset as _asset
from experiments.area_pressure.pusht_public_scene import legal_initial_q as _legal_q
from phymeo.backends.area_friction_readback import AreaFrictionEntryReader
from phymeo.backends.mujoco_area_pressure import area_collision_ownership
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.geometry.area_pressure_intersection import RigidPressureMesh
from phymeo.solvers.area_friction import LAW
from phymeo.training.manipulation_tasks import manipulation_scenes
from phymeo.training.resident_stepper import PhysicsStepFailure

pytestmark = pytest.mark.cuda
KEYS = (91, 7, 41, 11, 305)


def _declaration():
    import mujoco as mj

    original = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
    model = mj.MjModel.from_xml_string(original.mjcf)
    assert (model.nq, model.nv, model.nu) == (9, 8, 2)
    entries = load_entries()
    spec = build_scene_spec(original.mjcf)
    worlds = tuple(
        MuJoCoTyxWorldSpec(
            key,
            spec.key,
            MuJoCoInitialState(
                _legal_q(model, entries[label]["q"]),
                entries[label]["v"],
                time=entries[label]["time"],
            ),
        )
        for key, label in zip(KEYS, LABELS, strict=True)
    )
    return model, spec, worlds, entries


def _controls(entries):
    return {
        key: MuJoCoControl(
            entries[label]["ctrl"],
            qfrc_applied=entries[label]["qfrc_applied"],
            xfrc_applied=entries[label]["xfrc_applied"],
        )
        for key, label in zip(KEYS, LABELS, strict=True)
    }


def _native(model, before):
    packet = before.held_control
    return assemble_entry(
        model,
        dict(
            q=before.q,
            v=before.v,
            time=before.time,
            ctrl=packet.ctrl,
            qfrc_applied=packet.qfrc_applied,
            xfrc_applied=packet.xfrc_applied,
        ),
    )


def _separated_other_candidates(model, data):
    """Independent segment/box distance for all retained non-floor candidates."""
    gaps = []
    import mujoco as mj

    ownership = area_collision_ownership(mj, model, (("tool", "object"),))
    for first, second in ownership.native_geometry_pairs:
        types = tuple(model.geom_type[[first, second]])
        if types[0] == 0:
            continue
        assert set(types) == {3, 6}, types  # Original RR capsule/box candidates.
        capsule, box = (first, second) if types[0] == 3 else (second, first)
        rotation = data.geom_xmat[capsule].reshape(3, 3)
        offset = model.geom_size[capsule, 1] * rotation[:, 2]
        ends = data.geom_xpos[capsule] + np.array([-offset, offset])
        local = (ends - data.geom_xpos[box]) @ data.geom_xmat[box].reshape(3, 3)
        size = model.geom_size[box]
        gap = axis_box_clearance(local, np.asarray([[-size, size]]), model.geom_size[capsule, 0])
        assert gap > 0, (model.geom(capsule).name, model.geom(box).name, gap)
        gaps.append(gap)
    assert gaps
    return min(gaps)


def _match_rows(model, receipt, source):
    """A complete geometry bijection; original compact contact IDs may differ."""
    data = source["data"]
    expected = [
        dict(c, native_index=c["contact_index"]) for c in source["contacts"] if c["retained"]
    ]
    matches = match_contacts(receipt.contacts, expected)
    mapping = []
    for actual, contact in zip(receipt.contacts, matches, strict=True):
        native = data.contact[contact["native_index"]]
        assert actual.dimension == native.dim == 3
        assert actual.active == (native.efc_address >= 0)
        np.testing.assert_allclose(actual.frame, contact["frame"], rtol=3e-5, atol=4e-5)
        # Every active retained RR candidate must really be a floor support.
        if actual.active:
            types = tuple(model.geom_type[list(actual.geometry_ids)])
            assert types[0] == 0 and types[1] in (2, 3, 6), types
    limits = {
        (r["joint_id"], r["side"]): r["source_row"]
        for r in source["owners"]
        if r["kind"] == "limit"
    }
    for row in receipt.constraints:
        if row.constraint_type == 3:
            index = limits[row.source_id, row.side_or_edge]
            assert row.geometry_slot == -1
        else:
            assert row.constraint_type == 6 and 0 <= row.side_or_edge < 4
            contact = matches[row.source_id]
            index = contact["efc_address"] + row.side_or_edge
            assert contact["efc_address"] >= 0
        mapping.append(index)
        for name, array in (
            ("position", data.efc_pos),
            ("margin", data.efc_margin),
            ("velocity", data.efc_vel),
            ("reference_acceleration", data.efc_aref),
            ("regularization", data.efc_R),
            ("inverse_regularization", data.efc_D),
            ("diag_approx", data.efc_diagApprox),
        ):
            np.testing.assert_allclose(getattr(row, name), array[index], **ROW_TOL, err_msg=name)
        np.testing.assert_allclose(
            row.jacobian, data.efc_J.reshape(-1, 8)[index], rtol=2e-4, atol=5e-5
        )
        assert row.lower == 0 and row.upper is None and row.force >= 0
    assert len(mapping) == len(set(mapping)) == len(source["retained_indices"])
    assert set(mapping) == set(source["retained_indices"])
    return np.asarray(mapping, dtype=int), matches


def _root(problem):
    """Independent eight-variable checked Newton, never seeded by device output."""
    velocity = np.linalg.solve(problem["response"], problem["rhs"])
    factor = np.linalg.cholesky(problem["response"])
    scale = 1 + np.linalg.norm(np.linalg.solve(factor, problem["rhs"]))

    def norm(residual):
        return float(np.linalg.norm(np.linalg.solve(factor, residual)))

    for _ in range(80):
        result = _value(velocity, problem)
        residual = result["residual"]
        error = norm(residual)
        if error <= 1e-11 * scale:
            return velocity, result
        jac = np.empty((8, 8))
        for j in range(8):
            delta = np.zeros(8)
            delta[j] = 1e-6 * max(1.0, abs(velocity[j]))
            jac[:, j] = (
                _value(velocity + delta, problem)["residual"]
                - _value(velocity - delta, problem)["residual"]
            ) / (2 * delta[j])
        direction = np.linalg.solve(jac, -residual)
        for power in range(24):
            candidate = velocity + direction * 0.5**power
            if norm(_value(candidate, problem)["residual"]) < error:
                velocity = candidate
                break
        else:
            raise AssertionError(("independent RR Newton stalled", velocity, residual))
    raise AssertionError("independent RR Newton exceeded its iteration budget")


def _material(asset, receipt, geometry, pair):
    """Check each live point against the declared current PL cell and rigid face."""
    audit = _surface(pair, geometry, receipt)
    dyn = receipt.dynamics
    ba, bb = pair["body_ids"]
    soft, rigid = asset.pressure_local, asset.rigid_local
    vertices = soft.world_vertices @ dyn.body_rotation[ba].T + dyn.body_position[ba]
    faces = rigid.world_vertices @ dyn.body_rotation[bb].T + dyn.body_position[bb]
    for triangle in geometry.surface.triangles:
        cell = soft.cells[soft.cell_ids.index(triangle.compliant_cell_id)]
        tetra = vertices[cell]
        affine = np.column_stack((tetra, np.ones(4)))
        field = np.linalg.solve(affine, soft.nodal_pressure[cell])
        bary = np.linalg.solve(affine.T, np.column_stack((triangle.vertices, np.ones(3))).T).T
        assert np.min(bary) >= -2e-10 and np.max(bary) <= 1 + 2e-10
        np.testing.assert_allclose(
            triangle.vertex_pressure,
            np.column_stack((triangle.vertices, np.ones(3))) @ field,
            rtol=1e-8,
            atol=2e-7,
        )
        np.testing.assert_allclose(triangle.pressure_gradient, field[:3], rtol=1e-9, atol=2e-5)
        face = faces[rigid.faces[rigid.face_ids.index(triangle.rigid_face_id)]]
        normal = np.cross(face[1] - face[0], face[2] - face[0])
        normal /= np.linalg.norm(normal)
        np.testing.assert_allclose((triangle.vertices - face[0]) @ normal, 0, atol=2e-12)
        assert normal @ triangle.pressure_gradient > 0
    triangles = geometry.surface.triangles
    normals, areas = [], []
    for triangle in triangles:
        cross = np.cross(
            triangle.vertices[1] - triangle.vertices[0], triangle.vertices[2] - triangle.vertices[0]
        )
        magnitude = np.linalg.norm(cross)
        normals.extend((cross / magnitude,) * 3)
        areas.extend((magnitude / 6,) * 3)
    audit.update(
        normal=np.asarray(normals).reshape(-1, 3),
        stiffness=np.asarray(areas) * audit["source_gradient"],
        gap=-audit["source_pressure"] / audit["source_gradient"],
    )
    return audit


def _audit(model, asset, before, after, geometry):
    import mujoco as mj

    receipt, source = after.last_step, _native(model, before)
    data, dyn = source["data"], receipt.dynamics
    assert after.tick == receipt.tick == before.tick + 1
    np.testing.assert_array_equal(receipt.input_q, before.q)
    np.testing.assert_array_equal(receipt.input_v, before.v)
    assert receipt.input_time == receipt.contact_time == dyn.time == before.time
    assert receipt.contact_phase == "tyx-euler-input-constraint-evaluation"
    assert after.time - before.time == pytest.approx(H, abs=2e-9)
    (pair,) = receipt.raw_contact_fields["area_friction"]
    solver = receipt.raw_contact_fields["area_friction_solver"]
    assert pair["identity"] == IDENTITY and pair["law"] == solver["law"] == LAW
    assert solver["solver_status"] == 0
    assert tuple(pair["body_ids"]) == (model.body("tool").id, model.body("object").id)
    assert pair["mu"] == pytest.approx(MU, abs=5e-8)
    assert pair["slip_speed_m_s"] == pytest.approx(SLIP, abs=1e-9)
    np.testing.assert_array_equal(solver["input_velocity"], before.v)
    np.testing.assert_array_equal(solver["output_velocity"], after.v)
    for actual, expected in (
        (dyn.mass_matrix, source["mass"]),
        (dyn.bias, data.qfrc_bias),
        (dyn.passive, data.qfrc_passive),
        (dyn.applied, source["base_load"] - data.qfrc_passive + data.qfrc_bias),
        (dyn.body_position, data.xpos),
        (dyn.body_rotation, data.xmat.reshape(model.nbody, 3, 3)),
    ):
        np.testing.assert_allclose(actual, expected, rtol=4e-4, atol=5e-5)
    separation = _separated_other_candidates(model, data)
    mapping, matches = _match_rows(model, receipt, source)
    material = _material(asset, receipt, geometry, pair)
    points = pair["points"].astype(float)
    ja, jb = (np.empty((len(points), 3, 8)) for _ in range(2))
    angular = np.empty((3, 8))
    for i, point in enumerate(points):
        mj.mj_jac(model, data, ja[i], angular, point, model.body("tool").id)
        mj.mj_jac(model, data, jb[i], angular, point, model.body("object").id)
    np.testing.assert_allclose(pair["jacobian_a"], ja, rtol=2e-4, atol=5e-5)
    np.testing.assert_allclose(pair["jacobian_b"], jb, rtol=2e-4, atol=5e-5)
    native_j = np.array([r.jacobian for r in receipt.constraints]).reshape(-1, 8)
    reg = np.array([r.regularization for r in receipt.constraints])
    aref = np.array([r.reference_acceleration for r in receipt.constraints])
    native_force = np.array([r.force for r in receipt.constraints])
    mass = solver["mass_matrix"].astype(float)
    damping = solver["damping"].astype(float)
    h = solver["h_s"]
    assert h == float(np.float32(H))
    np.testing.assert_allclose(damping, model.dof_damping, rtol=1e-7, atol=1e-9)
    np.testing.assert_array_equal(mass, dyn.mass_matrix.astype(np.float32))
    base = dyn.applied + dyn.passive - dyn.bias
    np.testing.assert_allclose(
        solver["applied_force"], base + damping * before.v, rtol=2e-5, atol=2e-5
    )
    p = dict(
        mass=mass,
        response=mass + h * np.diag(damping),
        h=h,
        v0=before.v.astype(float),
        rhs=mass @ before.v + h * (base + damping * before.v),
        native_j=native_j,
        reg=reg,
        aref=aref,
        point_j=pair["jacobian_a"].astype(float) - pair["jacobian_b"].astype(float),
        normal=pair["normals"].astype(float),
        stiffness=pair["stiffness"].astype(float),
        gap=pair["surrogate_gap"].astype(float),
        mu=pair["mu"],
        slip=pair["slip_speed_m_s"],
    )
    expected_v, reference = _root(p)
    np.testing.assert_allclose(after.v, expected_v, rtol=5e-5, atol=5e-5)
    actual = _value(after.v.astype(float), p)
    for observed, expected in (
        (native_force, reference["native"]),
        (pair["normal_force"], reference["normal"]),
        (pair["tangential_force"], reference["tangent"]),
        (pair["force_on_a"], reference["force"]),
    ):
        np.testing.assert_allclose(observed, expected, rtol=0.003, atol=0.002)
    np.testing.assert_allclose(pair["relative_velocity"], actual["relative"], rtol=5e-5, atol=2e-5)
    force = pair["force_on_a"].astype(float)
    tangent, normal = pair["tangential_force"].astype(float), pair["normal_force"].astype(float)
    assert np.all(normal >= 0)
    assert np.all(np.linalg.norm(tangent, axis=1) <= pair["mu"] * normal + 2e-6)
    assert np.all(np.sum(tangent * actual["tangent_velocity"], axis=1) <= 2e-10)
    np.testing.assert_array_equal(pair["force_on_b"], -pair["force_on_a"])
    np.testing.assert_allclose(
        pair["torque_on_a_world_origin"], np.cross(points, force), rtol=3e-4, atol=2e-6
    )
    np.testing.assert_array_equal(
        pair["torque_on_b_world_origin"], -pair["torque_on_a_world_origin"]
    )
    tool_q, object_q = np.einsum("sij,si->j", ja, force), np.einsum("sij,si->j", jb, -force)
    area_q = tool_q + object_q
    ground_q = native_j.T @ native_force
    np.testing.assert_allclose(receipt.qfrc_constraint, ground_q + area_q, rtol=5e-5, atol=2e-5)
    np.testing.assert_array_equal(receipt.qfrc_constraint, solver["generalized_force"])
    np.testing.assert_allclose(
        h * receipt.qacc, h * reference["acceleration"], rtol=5e-5, atol=5e-5
    )
    physical_rhs = base + receipt.qfrc_constraint
    np.testing.assert_allclose(mass @ receipt.qacc, physical_rhs, rtol=5e-5, atol=2e-5)
    np.testing.assert_allclose(
        p["response"] @ receipt.integration_acceleration, physical_rhs, rtol=5e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        after.v, before.v + h * receipt.integration_acceleration, rtol=5e-5, atol=5e-5
    )
    next_q = before.q.astype(float).copy()
    mj.mj_integratePos(model, next_q, expected_v, h)
    np.testing.assert_allclose(after.q, next_q, rtol=5e-6, atol=2e-7)
    assert np.linalg.norm(after.q[5:9]) == pytest.approx(1, abs=2e-7)
    point_power = float(np.sum(force * actual["relative"]))
    assert area_q @ after.v == pytest.approx(point_power, rel=5e-5, abs=2e-5)
    support = 0.0
    for ci, (contact, original) in enumerate(zip(receipt.contacts, matches, strict=True)):
        edge = [
            r.force for r in receipt.constraints if r.constraint_type == 6 and r.source_id == ci
        ]
        if not contact.active:
            assert not edge
            continue
        assert len(edge) == 4
        mu = data.contact[original["native_index"]].friction
        local = np.array([sum(edge), mu[0] * (edge[0] - edge[1]), mu[1] * (edge[2] - edge[3])])
        world = contact.frame.T @ local
        np.testing.assert_allclose(contact.force_on_geometry2_world, world, rtol=3e-4, atol=2e-5)
        np.testing.assert_array_equal(
            contact.force_on_geometry1_world, -contact.force_on_geometry2_world
        )
        if model.body("object").id in contact.body_ids:
            support += world[2] if contact.body_ids[1] == model.body("object").id else -world[2]
    # Second independently solved system uses original FP64 MuJoCo M/J/R/aref
    # and FP64 triangle quadrature; no GPU force is an input to this reference.
    cpu = dict(
        p,
        mass=source["mass"],
        response=source["mass"] + h * np.diag(model.dof_damping),
        rhs=source["mass"] @ before.v + h * (source["base_load"] + model.dof_damping * before.v),
        native_j=data.efc_J.reshape(-1, 8)[mapping],
        reg=data.efc_R[mapping],
        aref=data.efc_aref[mapping],
        point_j=ja - jb,
        normal=material["normal"],
        stiffness=material["stiffness"],
        gap=material["gap"],
        mu=MU,
        slip=SLIP,
    )
    cpu_v, cpu_reference = _root(cpu)
    np.testing.assert_allclose(after.v, cpu_v, rtol=5e-4, atol=5e-5)
    np.testing.assert_allclose(native_force, cpu_reference["native"], rtol=0.003, atol=0.002)
    return dict(
        samples=len(points),
        normal=float(normal.sum()),
        tangent=float(np.linalg.norm(tangent)),
        support=support,
        tool_q=tool_q,
        object_q=object_q,
        native_rows=len(native_force),
        contacts=len(matches),
        other_candidate_min_clearance_m=separation,
        max_native_force_error=float(
            np.max(np.abs(native_force - cpu_reference["native"]), initial=0)
        ),
        max_velocity_error=float(np.max(np.abs(after.v - cpu_v))),
        max_joint_torque_error=float(
            np.max(
                np.abs(
                    receipt.qfrc_constraint[:2]
                    - (cpu_reference["area_q"] + cpu_reference["native_q"])[:2]
                )
            )
        ),
        max_quaternion_error=float(np.max(np.abs(after.q[5:9] - next_q[5:9]))),
    )


def _observe(workspace, batch, spec, states):
    frame = workspace.refresh(expected_revision=batch.checkpoint_revision)
    arrays = workspace.readback(frame)
    decoded = AreaFrictionEntryReader(workspace, {spec.key: (IDENTITY,)}).validate(arrays, frame)
    assert workspace.binding["area_schema"] == "mujoco-tyx-area-device-entry-v2"
    for key, state in states.items():
        (pair,) = state.last_step.raw_contact_fields["area_friction"]
        samples = decoded[key].samples
        assert len(samples) == len(pair["sample_ids"])
        for values, expected in (
            ([s.normal_force_n for s in samples], pair["normal_force"]),
            ([s.tangential_force_on_a_world for s in samples], pair["tangential_force"]),
            ([s.relative_velocity_world for s in samples], pair["relative_velocity"]),
        ):
            np.testing.assert_array_equal(np.asarray(values).reshape(expected.shape), expected)
    return frame, arrays


def test_rr_five_public_worlds_commit_mixed_entry_and_replay(
    record_property, monkeypatch, tmp_path
):
    import mujoco as mj

    model, spec, worlds, entries = _declaration()
    asset = _asset(model)
    ownership = area_collision_ownership(mj, model, spec.area_pressure_pairs)
    replaced = {frozenset(p) for p in ownership.replaced_geometry_pairs[0]}
    assert replaced == {
        frozenset((model.geom("pusher").id, model.geom(name).id))
        for name in ("object-0", "object-1")
    }
    assert len(ownership.native_geometry_pairs) == 9
    record_property(
        "public_quaternion_normalization_max_abs",
        {
            label: float(np.max(np.abs(world.initial_state.q - entries[label]["q"])))
            for world, label in zip(worlds, LABELS, strict=True)
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU MuJoCo must not advance the public production scene")

    monkeypatch.setattr(mj, "mj_step", forbidden)
    with MuJoCoTyxHeterogeneousBatch((spec,), worlds, mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (asset,)})
        batch.set_controls(_controls(entries))
        with batch.prepare_device_workspace(
            include_bodies=True, include_contacts=True, include_area_pressure=True
        ) as workspace:
            summary = []

            def advance(phase="original"):
                before = batch.integration_snapshot()
                assert scene.step(H, expected_revision=batch.checkpoint_revision) == dict.fromkeys(
                    KEYS, True
                )
                after = batch.integration_snapshot()
                audits = {
                    key: _audit(model, asset, before[key], after[key], scene.last_geometry[key][0])
                    for key in KEYS
                }
                frame, arrays = _observe(workspace, batch, spec, after)
                for key, label in zip(KEYS, LABELS, strict=True):
                    summary.append(
                        dict(
                            label=label,
                            phase=phase,
                            tick=after[key].tick,
                            **{
                                name: value.tolist() if isinstance(value, np.ndarray) else value
                                for name, value in audits[key].items()
                            },
                        )
                    )
                (tmp_path / "summary.json").write_text(
                    json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
                )
                return after, audits, frame, arrays

            first, audits, _, _ = advance()
            for key in (KEYS[0], KEYS[4]):
                assert audits[key]["samples"] == 0 and audits[key]["normal"] == 0
                np.testing.assert_array_equal(audits[key]["tool_q"], 0)
                np.testing.assert_array_equal(audits[key]["object_q"], 0)
            assert audits[KEYS[2]]["samples"] > 0 and audits[KEYS[2]]["normal"] > 1e-8
            assert audits[KEYS[2]]["tangent"] > 1e-9
            assert np.linalg.norm(audits[KEYS[2]]["tool_q"][:2]) > 1e-9
            assert np.linalg.norm(audits[KEYS[2]]["object_q"][2:]) > 1e-9
            assert all(audits[key]["support"] > 1.0 for key in KEYS[1:])
            record_property("first_step_mesh_samples", [audits[key]["samples"] for key in KEYS])
            checkpoint = scene.checkpoint()
            second, _, _, second_arrays = advance()  # Same accepted q/v; held real motors.
            assert all(second[key].tick == first[key].tick + 1 for key in KEYS)
            scene.restore(checkpoint, expected_revision=batch.checkpoint_revision)
            replay, _, _, replay_arrays = advance("replay")
            _same(replay, second, fresh_stamp=True)
            _same(replay_arrays, second_arrays)


def test_rr_public_failure_after_candidate_preserves_entry_and_retries(monkeypatch):
    model, spec, worlds, entries = _declaration()
    asset = _asset(model)
    with MuJoCoTyxHeterogeneousBatch((spec,), worlds, mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (asset,)})
        batch.set_controls(_controls(entries))
        with batch.prepare_device_workspace(
            include_bodies=True, include_contacts=True, include_area_pressure=True
        ) as workspace:
            assert scene.step(H, expected_revision=batch.checkpoint_revision) == dict.fromkeys(
                KEYS, True
            )
            before = batch.integration_snapshot()
            geometry, checkpoint, revision = (
                scene.last_geometry,
                scene.checkpoint(),
                batch.checkpoint_revision,
            )
            frame, arrays = _observe(workspace, batch, spec, before)
            attempts = []

            def reject(stage, publication):
                assert stage == "step"
                attempts.append(stage)
                raise RuntimeError("test failure after real RR mixed candidate, before publication")

            with monkeypatch.context() as patch:
                patch.setattr(batch, "_before_commit", reject)
                with pytest.raises(PhysicsStepFailure):
                    scene.step(H, expected_revision=revision)
            assert attempts == ["step"] and batch.checkpoint_revision == revision
            _same(batch.integration_snapshot(), before)
            assert scene.last_geometry is geometry
            assert scene.checkpoint().to_bytes() == checkpoint.to_bytes()
            _same(workspace.readback(frame), arrays)
            _, fresh_arrays = _observe(workspace, batch, spec, before)
            _same(fresh_arrays, arrays)
            assert scene.step(H, expected_revision=revision) == dict.fromkeys(KEYS, True)
            retry = batch.integration_snapshot()
            for key in KEYS:
                _audit(model, asset, before[key], retry[key], scene.last_geometry[key][0])
            _, retry_arrays = _observe(workspace, batch, spec, retry)
            scene.restore(checkpoint, expected_revision=batch.checkpoint_revision)
            assert scene.step(H, expected_revision=batch.checkpoint_revision) == dict.fromkeys(
                KEYS, True
            )
            _same(batch.integration_snapshot(), retry, fresh_stamp=True)
            _, replay_arrays = _observe(workspace, batch, spec, batch.integration_snapshot())
            _same(replay_arrays, retry_arrays)


def _refined_side_asset(asset):
    """Split each real face around an interior point; keep its complete boundary."""
    from dataclasses import replace

    rigid = asset.rigid_local
    vertices = list(rigid.world_vertices)
    faces, names = [], []
    for face, name in zip(rigid.faces, rigid.face_ids, strict=True):
        center = len(vertices)
        point = rigid.world_vertices[face].mean(axis=0)
        point[1] = rigid.world_vertices[face[0], 1]  # Preserve the exact stored plane.
        vertices.append(point)
        for edge in range(3):
            faces.append((int(face[edge]), int(face[(edge + 1) % 3]), center))
            names.append(f"{name}-part-{edge}")
    refined = RigidPressureMesh(
        (*rigid.node_ids, *(f"interior-{i}" for i in range(len(rigid.faces)))),
        np.asarray(vertices),
        tuple(faces),
        tuple(names),
    )
    # The original edges remain whole, so no rounded edge midpoint changes a
    # shared boundary. All new points lie in the same planar finite interface.
    old_tri = rigid.world_vertices[rigid.faces]
    new_tri = refined.world_vertices[refined.faces].reshape(-1, 3, 3, 3)
    old_cross = np.cross(old_tri[:, 1] - old_tri[:, 0], old_tri[:, 2] - old_tri[:, 0])
    new_cross = np.cross(new_tri[:, :, 1] - new_tri[:, :, 0], new_tri[:, :, 2] - new_tri[:, :, 0])
    assert np.all(np.sum(new_cross * old_cross[:, None], axis=2) > 0)
    np.testing.assert_allclose(new_cross.sum(axis=1), old_cross, rtol=1e-14, atol=1e-18)
    return replace(asset, rigid_local=refined)


def test_rr_only_loaded_lane_exceeds_combined_capacity_and_same_owner_retries(record_property):
    from dataclasses import replace

    from phymeo.backends.mujoco_area_pressure_scene import _geometry_requests, _intersect_requests

    model, original_spec, all_worlds, entries = _declaration()
    bad, good = KEYS[2], KEYS[4]  # Peak and truly withdrawn; one shared model/owner.
    keys, capacity = (bad, good), 56
    spec = replace(original_spec, constraint_capacity=capacity)
    worlds = tuple(world for world in all_worlds if world.world in keys)
    asset = _asset(model)
    refined = _refined_side_asset(asset)
    with MuJoCoTyxHeterogeneousBatch((spec,), worlds, mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (asset,)})
        dense_scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (refined,)})
        batch.set_controls(
            {key: control for key, control in _controls(entries).items() if key in keys}
        )
        assert len(batch._buckets) == 1
        owner = batch._buckets[0]
        with batch.prepare_device_workspace(
            include_bodies=True, include_contacts=True, include_area_pressure=True
        ) as workspace:
            assert scene.step(H, expected_revision=batch.checkpoint_revision) == dict.fromkeys(
                keys, True
            )
            before = batch.integration_snapshot()
            native = {key: len(_native(model, before[key])["retained_indices"]) for key in keys}

            def sample_counts(candidate):
                # Read the actual current resident poses without a trial step,
                # then generate genuine bounded finite-mesh intersections.
                with batch._transaction():
                    requests = _geometry_requests(owner, {key: (candidate,) for key in keys})
                    generated, _ = _intersect_requests(requests, None)
                return {key: 3 * len(generated[key][0].surface.triangles) for key in keys}

            coarse, dense = sample_counts(asset), sample_counts(refined)
            assert all(coarse[key] + native[key] <= capacity for key in keys)
            assert dense[good] == 0 and native[good] <= capacity
            assert 0 < coarse[bad] < dense[bad] <= capacity
            assert native[bad] + dense[bad] > capacity
            # This must reach the combined device gate, not HOST sample-only
            # packing or geometry resource limits; the unchanged limits are idle.
            assert all(
                not any(row["kind"] == "limit" for row in _native(model, before[key])["owners"])
                for key in keys
            )
            record_property(
                "combined_capacity_case",
                dict(capacity=capacity, coarse=coarse, refined=dense, native=native),
            )
            revision, publication = batch.checkpoint_revision, batch.last_step
            geometry = scene.last_geometry
            checkpoint = scene.checkpoint()
            dense_checkpoint = dense_scene.checkpoint().to_bytes()
            frame, arrays = _observe(workspace, batch, spec, before)
            with pytest.raises(PhysicsStepFailure):
                dense_scene.step(H, expected_revision=revision)
            device = owner._device
            with device.tx.use_runtime(device.runtime):
                device.runtime.sync()
                status = device._get(device.status)
                counts = device._get(device.constraint_counts)
                sizes = device._get(device.friction_sizes)
            assert batch._buckets[0] is owner
            assert batch.checkpoint_revision == revision and batch.last_step is publication
            assert scene.last_geometry is geometry and not dense_scene.last_geometry
            assert scene.checkpoint().to_bytes() == checkpoint.to_bytes()
            assert dense_scene.checkpoint().to_bytes() == dense_checkpoint
            _same(batch.integration_snapshot(), before)
            _same(workspace.readback(frame), arrays)  # Includes native and area wrenches.
            _, unchanged = _observe(workspace, batch, spec, before)
            _same(unchanged, arrays)
            lanes = {key: index for index, key in enumerate(owner.world_keys)}
            for key in keys:
                lane = lanes[key]
                assert counts[lane, 2] == native[key]
                assert sizes[lane, 1] == dense[key]
            assert status[lanes[bad]] == 8 and status[lanes[good]] == 0
            # No reset, state injection, new model, owner or control packet.
            assert scene.step(H, expected_revision=revision) == dict.fromkeys(keys, True)
            assert batch._buckets[0] is owner
            retry = batch.integration_snapshot()
            for key in keys:
                _audit(model, asset, before[key], retry[key], scene.last_geometry[key][0])
            _, retried_arrays = _observe(workspace, batch, spec, retry)
            scene.restore(checkpoint, expected_revision=batch.checkpoint_revision)
            assert scene.step(H, expected_revision=batch.checkpoint_revision) == dict.fromkeys(
                keys, True
            )
            replay = batch.integration_snapshot()
            _same(replay, retry, fresh_stamp=True)
            _, replay_arrays = _observe(workspace, batch, spec, replay)
            _same(replay_arrays, retried_arrays)
