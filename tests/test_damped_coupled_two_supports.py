"""Two fixed support identities share a damped slide/hinge velocity solve."""

import xml.etree.ElementTree as ET
from dataclasses import replace
from time import perf_counter_ns

import numpy as np
import pytest
import test_damped_coupled_articulated_contact as articulated
from test_damped_coupled_mixed_area_friction import _observe, _root, _value
from test_damped_hinge_mixed_area_friction import _side_asset
from test_mujoco_area_friction_lifecycle import _evidence, _same
from test_mujoco_area_friction_live import _surface as _audit_surface

from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch
from phymeo.backends.mujoco_heterogeneous_area_scene import (
    MuJoCoTyxHeterogeneousAreaPressureScene,
    MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

H, KEY = articulated.H, articulated.KEY
SLIDE, AXIS, DAMPING = articulated.SLIDE, articulated.AXIS, articulated.DAMPING
FEET = ("left-foot", "right-foot")
SCHEMA = "mujoco-tyx-area-friction-coupled-contact-two-supports-v1"


def _declaration(*, q=(0.001, 0.0), v=(0.04, 0.15)):
    base, world = articulated._declaration()
    tree = ET.fromstring(base.compilation.source.primary_xml)
    body = tree.find(".//body[@name='compliant']")
    left = body.find("geom[@name='foot']")
    left.attrib.update(name=FEET[0], pos="-.12 0 -.0502")
    right = ET.SubElement(body, "geom", left.attrib.copy())
    right.attrib.update(
        name=FEET[1],
        pos=".12 0 -.0502",
        priority="1",
        friction=".45 .01 .001",
        solref=".025 1",
        solimp=".8 .9 .001 .5 2",
    )
    spec = replace(
        base,
        key="damped-slide-hinge-two-supports",
        contact_capacity=2,
        compilation=compile_mujoco_xml(ET.tostring(tree, encoding="unicode"), require_asset=False),
    )
    return spec, replace(world, model_key=spec.key, initial_state=MuJoCoInitialState(q, v))


def _source(spec, state):
    import mujoco as mj

    model = spec.compilation.fresh_native_model()
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = state.q, state.v, state.time
    data.ctrl[:] = state.held_control.ctrl
    for field in ("qfrc_applied", "xfrc_applied"):
        value = getattr(state.held_control, field)
        if value is not None:
            getattr(data, field)[:] = value
    mj.mj_forward(model, data)
    assert model.nv == model.nq == 2 and model.neq == model.ntendon == 0
    assert not np.any(model.jnt_limited) and not np.any(model.dof_frictionloss)
    np.testing.assert_array_equal(model.dof_damping, DAMPING)
    theta, omega = state.q[1], state.v[1]
    coupling = 0.04 * np.cos(theta) - 0.03 * np.sin(theta)
    mass = np.empty((2, 2))
    mj.mj_fullM(model, mass, data.qM)
    np.testing.assert_allclose(mass, [[2.0, coupling], [coupling, 0.0325]], rtol=0, atol=1e-13)
    bias = [(-0.04 * np.sin(theta) - 0.03 * np.cos(theta)) * omega**2, 0.0]
    np.testing.assert_allclose(data.qfrc_bias, bias, rtol=1e-12, atol=1e-13)
    assert data.ncon in (0, 1, 2) and data.nefc == 4 * data.ncon
    pairs = tuple((model.geom("floor").id, model.geom(name).id) for name in FEET)
    contacts = {tuple(map(int, contact.geom)): contact for contact in data.contact}
    assert len(contacts) == data.ncon and set(contacts).issubset(pairs)
    assert all(contact.dim == 3 for contact in contacts.values())
    return model, data, mass, pairs, contacts


def _drive(batch, spec, target):
    # Source-only native inversion selects finite external force/torque.
    # The real area pressure/friction remains in the independent step oracle.
    state = batch.integration_snapshot()[KEY]
    _model, data, mass, _pairs, _contacts = _source(spec, state)
    h = float(np.float32(H))
    acceleration = (
        np.linalg.solve(mass, (mass + h * np.diag(DAMPING)) @ (np.asarray(target) - state.v)) / h
    )
    jac = data.efc_J.reshape(data.nefc, 2)
    loads = np.maximum(0.0, (data.efc_aref - jac @ acceleration) / data.efc_R)
    applied = mass @ acceleration - (data.qfrc_passive - data.qfrc_bias) - jac.T @ loads
    assert np.isfinite(applied).all()
    batch.set_controls({KEY: MuJoCoControl((), qfrc_applied=applied)})
    return applied


def _audit(spec, before, after, geometry):
    model, data, source_mass, pairs, sources = _source(spec, before)
    receipt = after.last_step
    np.testing.assert_array_equal(receipt.input_q, before.q)
    np.testing.assert_array_equal(receipt.input_v, before.v)
    assert receipt.tick == after.tick == before.tick + 1
    assert receipt.input_time == receipt.contact_time == receipt.dynamics.time == before.time
    raw = receipt.raw_contact_fields
    (pair,) = raw["area_friction"]
    solver, native = raw["area_friction_solver"], raw["native_contacts"]
    assert solver["schema"] == SCHEMA and solver["solver_status"] == 0
    assert raw["native_contact_operator"] == "M"
    np.testing.assert_array_equal(native["slot_ids"][:, 2:], pairs)
    np.testing.assert_array_equal(native["slot_ids"][:, :2], [[0, 0], [1, 0]])
    _audit_surface(pair, geometry, receipt)
    assert len(receipt.contacts) == len(sources) and len(receipt.constraints) == data.nefc
    assert {contact.geometry_ids for contact in receipt.contacts} == set(sources)
    rows = receipt.constraints
    jac, reg, aref, loads = (
        np.array([getattr(row, field) for row in rows])
        for field in ("jacobian", "regularization", "reference_acceleration", "force")
    )
    jac = jac.reshape(-1, 2)
    source_j = data.efc_J.reshape(data.nefc, 2)
    source_indices = []
    for row in rows:
        assert row.constraint_type == 6 and row.dof == -1
        contact = receipt.contacts[row.source_id]
        assert row.geometry_slot == pairs.index(contact.geometry_ids)
        assert row.lower == 0 and row.upper is None and row.force >= 0
        source_indices.append(sources[contact.geometry_ids].efc_address + row.side_or_edge)
    assert len(set(source_indices)) == data.nefc
    for actual, wanted in (
        (jac, source_j[source_indices]),
        (reg, data.efc_R[source_indices]),
        (aref, data.efc_aref[source_indices]),
        (solver["mass_matrix"], source_mass),
    ):
        np.testing.assert_allclose(actual, wanted, rtol=4e-4, atol=2e-5)
    dyn = receipt.dynamics
    for actual, wanted in (
        (dyn.bias, data.qfrc_bias),
        (dyn.passive, data.qfrc_passive),
        (dyn.applied, data.qfrc_applied + data.qfrc_actuator),
    ):
        np.testing.assert_allclose(actual, wanted, rtol=4e-4, atol=2e-5)
    pivot = articulated.PIVOT0 + SLIDE * before.q[0]
    ja, jb = articulated._point_jacobians(model, data, pair["points"].astype(float), pivot)
    np.testing.assert_allclose(pair["jacobian_a"], ja, rtol=2e-5, atol=2e-7)
    np.testing.assert_array_equal(pair["jacobian_b"], jb)
    h, mass = solver["h_s"], solver["mass_matrix"].astype(float)
    np.testing.assert_array_equal(solver["damping"], DAMPING)
    response, applied = mass + h * np.diag(DAMPING), solver["applied_force"].astype(float)
    np.testing.assert_allclose(
        applied, dyn.applied + dyn.passive - dyn.bias + DAMPING * before.v, rtol=2e-5, atol=2e-5
    )
    p = dict(
        mass=mass,
        response=response,
        h=h,
        v0=before.v.astype(float),
        rhs=mass @ before.v + h * applied,
        native_j=jac,
        reg=reg,
        aref=aref,
        point_j=ja - jb,
        normal=pair["normals"].astype(float),
        stiffness=pair["stiffness"].astype(float),
        gap=pair["surrogate_gap"].astype(float),
        mu=pair["mu"],
        slip=pair["slip_speed_m_s"],
    )
    expected_v, expected, iterations = _root(p)
    actual = _value(after.v.astype(float), p)
    np.testing.assert_allclose(after.v, expected_v, rtol=5e-5, atol=5e-5)
    for got, wanted in (
        (loads, expected["native"]),
        (loads, actual["native"]),
        (pair["force_on_a"], expected["force"]),
        (pair["force_on_a"], actual["force"]),
        (pair["normal_force"], actual["normal"]),
        (pair["tangential_force"], actual["tangent"]),
    ):
        np.testing.assert_allclose(got, wanted, rtol=0.002, atol=0.002)
    assert pair["normal_force"].sum() > 0.1 and np.linalg.norm(pair["tangential_force"]) > 0.001
    ground, ground_q = {}, np.zeros(2)
    for ci, contact in enumerate(receipt.contacts):
        slot = pairs.index(contact.geometry_ids)
        source = sources[contact.geometry_ids]
        edge_rows = sorted(
            (row for row in rows if row.source_id == ci), key=lambda row: row.side_or_edge
        )
        assert [row.side_or_edge for row in edge_rows] == [0, 1, 2, 3]
        edge_loads = np.array([row.force for row in edge_rows])
        force = source.frame.reshape(3, 3).T @ np.array(
            [
                edge_loads.sum(),
                source.friction[0] * (edge_loads[0] - edge_loads[1]),
                source.friction[1] * (edge_loads[2] - edge_loads[3]),
            ]
        )
        moment = np.cross(source.pos - pivot, force)
        point_j = np.column_stack((SLIDE, np.cross(AXIS, source.pos - pivot)))
        source_contact_j = np.vstack(
            (
                source.frame.reshape(3, 3) @ point_j,
                source.frame.reshape(3, 3) @ np.column_stack((np.zeros(3), AXIS)),
            )
        )
        np.testing.assert_allclose(contact.jacobian, source_contact_j, rtol=2e-5, atol=2e-7)
        np.testing.assert_allclose(contact.force_on_geometry2_world, force, rtol=0.002, atol=0.002)
        np.testing.assert_array_equal(
            contact.force_on_geometry1_world, -contact.force_on_geometry2_world
        )
        moment_com = np.cross(source.pos - data.xipos[contact.body_ids[1]], force)
        np.testing.assert_allclose(
            contact.wrench_on_body2_world, np.r_[force, moment_com], rtol=0.002, atol=0.002
        )
        generalized = np.array([SLIDE @ force, AXIS @ moment])
        np.testing.assert_allclose(
            generalized,
            sum((r.jacobian * r.force for r in edge_rows), np.zeros(2)),
            rtol=2e-5,
            atol=2e-5,
        )
        ground_q += generalized
        ground[FEET[slot]] = dict(
            slot=slot,
            ci=ci,
            normal=force[2],
            force=force,
            moment_about_pivot=moment,
            generalized=generalized,
        )
    area_force = pair["force_on_a"].astype(float)
    area_moment = np.cross(pair["points"] - pivot, area_force).sum(axis=0)
    area_q = np.array([SLIDE @ area_force.sum(axis=0), AXIS @ area_moment])
    total = ground_q + area_q
    stored_area = np.einsum(
        "sij,si->j", pair["jacobian_a"].astype(float) - pair["jacobian_b"].astype(float), area_force
    )
    net = dyn.applied.astype(float) + jac.T @ loads + stored_area
    np.testing.assert_allclose(solver["applied_with_contact"], net, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(receipt.qfrc_constraint, total, rtol=2e-5, atol=2e-5)
    rhs = dyn.applied + dyn.passive - dyn.bias + total
    np.testing.assert_allclose(
        h * receipt.qacc, h * np.linalg.solve(mass, rhs), rtol=5e-5, atol=5e-5
    )
    np.testing.assert_allclose(h * receipt.qacc, h * expected["acceleration"], rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(
        after.v, before.v + h * np.linalg.solve(response, rhs), rtol=5e-5, atol=5e-5
    )
    np.testing.assert_allclose(after.q, before.q + h * after.v, rtol=0, atol=2e-7)
    factor = np.linalg.cholesky(response)
    scale = 1 + np.linalg.norm(np.linalg.solve(factor, p["rhs"]))
    for residual in (actual["residual"], response @ after.v - p["rhs"] - h * total):
        assert np.linalg.norm(np.linalg.solve(factor, residual)) <= 2e-5 * scale
    assert solver["normalized_residual"] <= solver["tolerance"] == 2e-5
    return dict(
        ground=ground,
        active_ground=sum(c.active for c in receipt.contacts),
        ns=len(pair["sample_ids"]),
        area_normal=float(pair["normal_force"].sum()),
        area_tangent_norm=float(np.linalg.norm(pair["tangential_force"])),
        area_q=area_q,
        area_moment_about_pivot=area_moment,
        ground_q=ground_q,
        source_mass=source_mass,
        source_bias=data.qfrc_bias.copy(),
        expected_v=expected_v,
        native_loads=loads,
        newton_iterations=iterations,
    )


@pytest.mark.cuda
def test_two_supports_contribute_distinct_levers_to_one_coupled_velocity(tmp_path):
    spec, world = _declaration()
    with _evidence(tmp_path / "two-support-one-step.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (_side_asset(),)})
            applied = _drive(batch, spec, (0.04, 0.15))
            before = batch.integration_snapshot()[KEY]
            start = perf_counter_ns()
            success = scene.step(H, expected_revision=batch.checkpoint_revision)
            elapsed = perf_counter_ns() - start
            assert success == {KEY: True}
            after = batch.integration_snapshot()[KEY]
            audit = _audit(spec, before, after, scene.last_geometry[KEY][0])
            evidence.update(q=after.q, v=after.v, applied=applied, step_ns=elapsed, audit=audit)
            assert set(audit["ground"]) == set(FEET)
            assert all(item["normal"] > 0.1 for item in audit["ground"].values())
            assert audit["ground"][FEET[0]]["ci"] == 0 and audit["ground"][FEET[1]]["ci"] == 1


@pytest.mark.cuda
def test_two_supports_respond_to_contact_without_an_external_drive(tmp_path):
    spec, world = _declaration(q=(0.0, 0.0), v=(0.002, 0.005))
    with _evidence(tmp_path / "two-support-unforced.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(
                batch, {spec.key: (_side_asset(depth=0.0001),)}
            )
            before = batch.integration_snapshot()[KEY]
            assert scene.step(H, expected_revision=batch.checkpoint_revision) == {KEY: True}
            after = batch.integration_snapshot()[KEY]
            audit = _audit(spec, before, after, scene.last_geometry[KEY][0])
            np.testing.assert_array_equal(after.last_step.dynamics.applied, np.zeros(2))
            assert set(audit["ground"]) == set(FEET)
            assert all(foot["normal"] > 0 for foot in audit["ground"].values())
            assert not np.array_equal(before.v, after.v)
            evidence.update(input_v=before.v, output_v=after.v, audit=audit)


def _timing_summary(records):
    """Report measured whole-step costs; make no timing acceptance threshold."""
    groups = {}
    for record in records:
        category = (
            "replay" if record["replay"] else "first" if record["index"] == 0 else "original-warm"
        )
        key = f"{category}:ground={record['active_ground']}:ns={record['ns']}"
        groups.setdefault(key, []).append(record["step_ns"])
    return {
        key: dict(samples_ns=values, count=len(values), median_ns=float(np.median(values)))
        for key, values in groups.items()
    }


@pytest.mark.cuda
def test_two_supports_touch_share_load_then_first_lifts_and_second_remains_ci_zero(tmp_path):
    spec, world = _declaration(q=(0.0, -0.006), v=(0.08, 0.1))
    with _evidence(tmp_path / "two-support-window.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (_side_asset(),)})
            with batch.prepare_device_workspace(
                include_bodies=True, include_area_pressure=True
            ) as ws:
                history, commands = [], []

                def advance(target, index, *, replay=False):
                    previous = batch.integration_snapshot()[KEY]
                    applied = _drive(batch, spec, target)
                    before = batch.integration_snapshot()[KEY]
                    np.testing.assert_array_equal(before.q, previous.q)
                    np.testing.assert_array_equal(before.v, previous.v)
                    assert before.time == previous.time and before.tick == previous.tick
                    start = perf_counter_ns()
                    success = scene.step(H, expected_revision=batch.checkpoint_revision)
                    elapsed = perf_counter_ns() - start
                    assert success == {KEY: True}
                    after, geometry = batch.integration_snapshot()[KEY], scene.last_geometry[KEY][0]
                    record = dict(
                        index=index,
                        replay=replay,
                        step_ns=elapsed,
                        active_ground=sum(c.active for c in after.last_step.contacts),
                        ns=len(
                            after.last_step.raw_contact_fields["area_friction"][0]["sample_ids"]
                        ),
                        input_q=before.q,
                        input_v=before.v,
                        q=after.q,
                        v=after.v,
                        qacc=after.last_step.qacc,
                        applied=applied,
                    )
                    evidence["records"].append(record)
                    evidence["timings"] = _timing_summary(evidence["records"])
                    audit = record["audit"] = _audit(spec, before, after, geometry)
                    _model, data, _mass, pairs, sources = _source(spec, before)
                    for contact in after.last_step.contacts:
                        source = sources[contact.geometry_ids]
                        np.testing.assert_allclose(
                            contact.position, source.pos, rtol=2e-5, atol=2e-7
                        )
                        np.testing.assert_allclose(
                            contact.friction, source.friction, rtol=2e-5, atol=2e-7
                        )
                        np.testing.assert_allclose(
                            contact.solref, source.solref, rtol=2e-5, atol=2e-7
                        )
                        slot = pairs.index(contact.geometry_ids)
                        assert contact.friction[0] == pytest.approx((0.7, 0.45)[slot], abs=2e-7)
                        total_moment = np.zeros(3)
                        for body, wrench, signed_force in (
                            (
                                contact.body_ids[0],
                                contact.wrench_on_body1_world,
                                -contact.force_on_geometry2_world,
                            ),
                            (
                                contact.body_ids[1],
                                contact.wrench_on_body2_world,
                                contact.force_on_geometry2_world,
                            ),
                        ):
                            moment = np.cross(source.pos - data.xipos[body], signed_force)
                            np.testing.assert_allclose(
                                wrench, np.r_[signed_force, moment], rtol=0.002, atol=0.002
                            )
                            published_com = after.last_step.dynamics.body_com[body]
                            np.testing.assert_allclose(
                                published_com, data.xipos[body], rtol=2e-5, atol=2e-7
                            )
                            total_moment += wrench[3:] + np.cross(published_com, wrench[:3])
                        np.testing.assert_allclose(total_moment, 0.0, rtol=0, atol=2e-5)
                    frame, arrays = _observe(ws, batch, (spec,), {KEY: after})
                    return after, geometry, frame, arrays, audit

                # First press down until both distinct native pairs carry load.
                # Then pitch right-down/left-up through finite external loads.
                # These bounds limit real motion; they do not prescribe event ticks.
                phases = (((0.08, 0.0), 16, set(FEET)), ((-0.02, 0.8), 24, {FEET[1]}))
                for phase, (target, bound, wanted) in enumerate(phases):
                    consecutive = 0
                    for _ in range(bound):
                        index = len(history)
                        commands.append(target)
                        history.append(advance(target, index))
                        ground = history[-1][4]["ground"]
                        loaded = set(ground) == wanted and all(
                            item["normal"] > 0.1 for item in ground.values()
                        )
                        consecutive = consecutive + 1 if loaded else 0
                        if consecutive == 2:
                            break
                    else:
                        raise AssertionError(
                            ("bounded physical drive did not complete support phase", phase)
                        )
                    if phase == 0:
                        saved_index, saved = len(history) - 1, history[-1]
                        capsule = MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(
                            scene.checkpoint().to_bytes()
                        )
                assert set(history[0][4]["ground"]) == {FEET[0]}
                assert history[0][4]["ground"][FEET[0]]["normal"] > 0.1
                runs = []
                for item in history:
                    names = tuple(sorted(item[4]["ground"]))
                    if not runs or runs[-1] != names:
                        runs.append(names)
                assert runs == [(FEET[0],), tuple(sorted(FEET)), (FEET[1],)]
                release = [item[4]["ground"] for item in history[saved_index + 1 :]]
                assert all(ground[FEET[1]]["normal"] > 0.1 for ground in release)
                assert any(
                    FEET[0] in ground and ground[FEET[0]]["normal"] <= 0.002 for ground in release
                )
                for item in history[-2:]:
                    right = item[4]["ground"][FEET[1]]
                    assert right["slot"] == 1 and right["ci"] == 0
                    assert len(item[0].last_step.constraints) == 4
                    assert all(
                        row.source_id == 0 and row.geometry_slot == 1
                        for row in item[0].last_step.constraints
                    )
                scene.restore(capsule, expected_revision=batch.checkpoint_revision)
                _same(batch.integration_snapshot()[KEY], saved[0])
                with pytest.raises(ValueError):
                    ws.readback(history[-1][2])
                restored = ws.refresh(expected_revision=batch.checkpoint_revision)
                _same(ws.readback(restored), saved[3])
                for index in range(saved_index + 1, len(history)):
                    replay = advance(commands[index], index, replay=True)
                    _same(replay[0], history[index][0], fresh_stamp=True)
                    _same(replay[1], history[index][1], fresh_stamp=True)
                    _same(replay[3], history[index][3])
                evidence["timings"] = _timing_summary(evidence["records"])
