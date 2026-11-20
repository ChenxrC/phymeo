"""Two named stops join ground and live area friction in one coupled solve.

The child reaches its upper stop before the parent reaches its lower stop.
Finite applied forces then release the child, followed by the parent. Only
the real CUDA owner evolves state; native FP64 forward supplies coefficients
to the independent two-variable root already used by the mixed-contact tests.
"""

import xml.etree.ElementTree as ET
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest
from test_damped_coupled_mixed_area_friction import (
    DAMPING,
    MASS,
    H,
    U,
    W,
    _observe,
    _root,
    _value,
)
from test_damped_coupled_mixed_area_friction import (
    _declaration as _unlimited,
)
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

KEY = 41
NAMES = ("parent-slide", "child-slide")
SCHEMA = "mujoco-tyx-area-friction-coupled-contact-two-limits-v1"


def _declaration(*, q=(0.0, 0.0)):
    base, world = _unlimited()
    tree = ET.fromstring(base.compilation.source.primary_xml)
    for name, bounds, solref, solimp in (
        ("parent-slide", "-.00035 .02", ".02 1", ".5 .5 .001 .5 2"),
        ("child-slide", "-.02 .0005", ".025 1", ".55 .55 .001 .5 2"),
    ):
        tree.find(f".//joint[@name='{name}']").attrib.update(
            limited="true",
            range=bounds,
            margin="0",
            solreflimit=solref,
            solimplimit=solimp,
        )
    spec = replace(
        base,
        key="two-coupled-stops",
        compilation=compile_mujoco_xml(ET.tostring(tree, encoding="unicode"), require_asset=False),
    )
    return spec, replace(
        world, model_key=spec.key, initial_state=MuJoCoInitialState(q, (-0.03, 0.1))
    )


def _source(spec, state):
    import mujoco as mj

    model = spec.compilation.fresh_native_model()
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = state.q, state.v, state.time
    data.ctrl[:] = state.held_control.ctrl
    for name in ("qfrc_applied", "xfrc_applied"):
        value = getattr(state.held_control, name)
        if value is not None:
            getattr(data, name)[:] = value
    mj.mj_forward(model, data)
    np.testing.assert_array_equal(model.jnt_limited, [1, 1])
    np.testing.assert_array_equal(model.dof_damping, DAMPING)
    assert not np.any(model.dof_frictionloss) and model.neq == model.ntendon == 0
    mass = np.empty((2, 2))
    mj.mj_fullM(model, mass, data.qM)
    np.testing.assert_allclose(mass, MASS, rtol=0, atol=1e-13)
    assert data.ncon == 1 and data.contact[0].dim == 3
    jac = data.efc_J.reshape(data.nefc, 2)
    keys, edge = [], 0
    for index, (kind, identity) in enumerate(zip(data.efc_type, data.efc_id, strict=True)):
        kind, identity = int(kind), int(identity)
        if kind == 3:
            dof = int(model.jnt_dofadr[identity])
            side = -1 if jac[index, dof] > 0 else 1
            np.testing.assert_array_equal(jac[index], -side * np.eye(2)[dof])
        else:
            assert kind == 6 and identity == 0
            side, edge = edge, edge + 1
        keys.append((kind, identity, side))
    wanted = {
        (3, joint, side)
        for joint in range(2)
        for side, distance in (
            (-1, state.q[joint] - model.jnt_range[joint, 0]),
            (1, model.jnt_range[joint, 1] - state.q[joint]),
        )
        if distance < model.jnt_margin[joint]
    }
    assert edge == 4 and len(keys) == len(set(keys))
    assert {key for key in keys if key[0] == 3} == wanted
    return model, data, jac, keys


def _drive(batch, spec, target):
    # Native-only inverse loads choose a finite drive, not an expected output.
    # The omitted live area force is included in the independent post-step root.
    state = batch.integration_snapshot()[KEY]
    _model, data, jac, _keys = _source(spec, state)
    h = float(np.float32(H))
    acceleration = np.linalg.solve(MASS, (MASS + h * np.diag(DAMPING)) @ (target - state.v)) / h
    loads = np.maximum(0.0, (data.efc_aref - jac @ acceleration) / data.efc_R)
    applied = MASS @ acceleration - (data.qfrc_passive - data.qfrc_bias) - jac.T @ loads
    assert np.isfinite(applied).all()
    batch.set_controls({KEY: MuJoCoControl((), qfrc_applied=applied)})
    return applied


def _audit(spec, before, after, geometry):
    model, data, source_j, keys = _source(spec, before)
    receipt = after.last_step
    np.testing.assert_array_equal(receipt.input_q, before.q)
    np.testing.assert_array_equal(receipt.input_v, before.v)
    assert receipt.tick == after.tick == before.tick + 1
    assert receipt.input_time == receipt.contact_time == receipt.dynamics.time == before.time
    raw = receipt.raw_contact_fields
    (pair,) = raw["area_friction"]
    solver = raw["area_friction_solver"]
    assert solver["schema"] == SCHEMA and solver["solver_status"] == 0
    assert raw["native_contact_operator"] == "M"
    _audit_surface(pair, geometry, receipt)
    rows = {
        (row.constraint_type, row.source_id, row.side_or_edge): row for row in receipt.constraints
    }
    assert len(rows) == len(receipt.constraints) == len(keys) and set(rows) == set(keys)
    ordered = [rows[key] for key in keys]
    jac, reg, aref, loads = (
        np.array([getattr(row, name) for row in ordered])
        for name in ("jacobian", "regularization", "reference_acceleration", "force")
    )
    for actual, wanted in ((jac, source_j), (reg, data.efc_R), (aref, data.efc_aref)):
        np.testing.assert_allclose(actual, wanted, rtol=4e-4, atol=2e-5)
    named, limit_q = {}, np.zeros(2)
    for index, row in enumerate(ordered):
        assert row.lower == 0 and row.upper is None and row.force >= 0
        for field, source in (
            ("position", data.efc_pos),
            ("margin", data.efc_margin),
            ("velocity", data.efc_vel),
        ):
            assert getattr(row, field) == pytest.approx(source[index], rel=4e-4, abs=2e-7)
        if row.constraint_type == 3:
            dof = int(model.jnt_dofadr[row.source_id])
            name = model.joint(row.source_id).name
            assert name == NAMES[dof] and row.dof == dof and row.geometry_slot == -1
            np.testing.assert_array_equal(row.jacobian, -row.side_or_edge * np.eye(2)[dof])
            projected = max(
                0.0, (data.efc_aref[index] - source_j[index] @ receipt.qacc) / data.efc_R[index]
            )
            assert row.force == pytest.approx(projected, rel=0.002, abs=0.002)
            named[name] = row.force
            limit_q[dof] = -row.side_or_edge * row.force
    h, mass = solver["h_s"], solver["mass_matrix"].astype(float)
    np.testing.assert_allclose(mass, MASS, rtol=4e-4, atol=2e-5)
    np.testing.assert_array_equal(solver["damping"], DAMPING)
    dyn = receipt.dynamics
    for actual, wanted in (
        (dyn.bias, data.qfrc_bias),
        (dyn.passive, data.qfrc_passive),
        (dyn.applied, data.qfrc_applied + data.qfrc_actuator),
    ):
        np.testing.assert_allclose(actual, wanted, rtol=4e-4, atol=2e-5)
    applied = solver["applied_force"].astype(float)
    np.testing.assert_allclose(
        applied, dyn.applied + dyn.passive - dyn.bias + DAMPING * before.v, rtol=2e-5, atol=2e-5
    )
    point_j = np.zeros((len(pair["sample_ids"]), 3, 2))
    point_j[:, :, 0] = U
    np.testing.assert_allclose(pair["jacobian_a"], point_j, rtol=0, atol=2e-7)
    np.testing.assert_array_equal(pair["jacobian_b"], 0)
    response = mass + h * np.diag(DAMPING)
    p = dict(
        mass=mass,
        response=response,
        h=h,
        v0=before.v.astype(float),
        rhs=mass @ before.v + h * applied,
        native_j=jac,
        reg=reg,
        aref=aref,
        point_j=point_j,
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
    if geometry.surface.triangles:
        assert pair["normal_force"].sum() > 0.1 and np.linalg.norm(pair["tangential_force"]) > 0.001
    else:
        assert pair["sample_ids"] == ()
    edges = np.array([rows[(6, 0, edge)].force for edge in range(4)])
    contact = data.contact[0]
    ground_force = contact.frame.reshape(3, 3).T @ np.array(
        [
            edges.sum(),
            contact.friction[0] * (edges[0] - edges[1]),
            contact.friction[1] * (edges[2] - edges[3]),
        ]
    )
    assert ground_force[2] > 0.1 and len(receipt.contacts) == 1
    np.testing.assert_allclose(
        receipt.contacts[0].force_on_geometry2_world, ground_force, rtol=0.002, atol=0.002
    )
    ground_q = np.array([U @ ground_force, W @ ground_force])
    area_q = np.array([U @ pair["force_on_a"].astype(float).sum(axis=0), 0.0])
    np.testing.assert_allclose(ground_q + limit_q, jac.T @ loads, rtol=2e-5, atol=2e-5)
    total = ground_q + area_q + limit_q
    np.testing.assert_allclose(receipt.qfrc_constraint, total, rtol=2e-5, atol=2e-5)
    physical_rhs = dyn.applied + dyn.passive - dyn.bias + total
    np.testing.assert_allclose(
        h * receipt.qacc, h * np.linalg.solve(mass, physical_rhs), rtol=5e-5, atol=5e-5
    )
    np.testing.assert_allclose(h * receipt.qacc, h * expected["acceleration"], rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(
        after.v, before.v + h * np.linalg.solve(response, physical_rhs), rtol=5e-5, atol=5e-5
    )
    np.testing.assert_allclose(after.q, before.q + h * after.v, rtol=0, atol=2e-7)
    gradients = jac @ receipt.qacc + reg * loads - aref
    np.testing.assert_allclose([row.gradient for row in ordered], gradients, rtol=2e-5, atol=2e-5)
    assert np.all(gradients >= -2e-5 * (1 + np.abs(aref)))
    factor = np.linalg.cholesky(response)
    scale = 1 + np.linalg.norm(np.linalg.solve(factor, p["rhs"]))
    for residual in (actual["residual"], response @ after.v - p["rhs"] - h * total):
        assert np.linalg.norm(np.linalg.solve(factor, residual)) <= 2e-5 * scale
    assert solver["normalized_residual"] <= solver["tolerance"] == 2e-5
    return dict(
        named_limit_loads=named,
        limit_q=limit_q,
        ground_q=ground_q,
        area_q=area_q,
        source_keys=keys,
        oracle=expected,
        newton_iterations=iterations,
    )


@pytest.mark.cuda
def test_two_named_limits_enter_together_release_independently_and_replay(tmp_path):
    spec, world = _declaration()
    with _evidence(tmp_path / "two-limit-window.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            caps = batch.capabilities["models"][spec.key]
            assert caps["domain"] == "area-friction-cuda-damped-coupled-support-two-limits-v1"
            assert caps["native_scalar_limit_joint_ids"] == (0, 1)
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (_side_asset(),)})
            with batch.prepare_device_workspace(
                include_bodies=True, include_area_pressure=True
            ) as ws:
                history, commands = [], []

                def advance(target):
                    previous = batch.integration_snapshot()[KEY]
                    applied = _drive(batch, spec, np.asarray(target))
                    before = batch.integration_snapshot()[KEY]
                    np.testing.assert_array_equal(before.q, previous.q)
                    np.testing.assert_array_equal(before.v, previous.v)
                    assert before.time == previous.time and before.tick == previous.tick
                    assert scene.step(H, expected_revision=batch.checkpoint_revision) == {KEY: True}
                    after, geometry = batch.integration_snapshot()[KEY], scene.last_geometry[KEY][0]
                    record = dict(before=before, after=after, geometry=geometry, applied=applied)
                    evidence["records"].append(record)
                    record["audit"] = _audit(spec, before, after, geometry)
                    frame, arrays = _observe(ws, batch, (spec,), {KEY: after})
                    record["arrays"] = arrays
                    return after, geometry, frame, arrays, record["audit"]

                # Bounds are safety limits for real motion, not an expected event tick.
                phases = (
                    ((-0.03, 0.1), 16, set(NAMES)),
                    ((-0.03, -0.2), 12, {NAMES[0]}),
                    ((0.2, -0.02), 8, set()),
                )
                for phase, (target, bound, wanted) in enumerate(phases):
                    consecutive = 0
                    for _ in range(bound):
                        commands.append(target)
                        history.append(advance(target))
                        named = history[-1][4]["named_limit_loads"]
                        matches = set(named) == wanted and all(
                            load > 0.01 for load in named.values()
                        )
                        consecutive = consecutive + 1 if matches else 0
                        if consecutive == 2:
                            break
                    else:
                        raise AssertionError(
                            ("finite drive did not complete phase", phase, history[-1])
                        )
                    if phase == 0:
                        saved_index, saved = len(history) - 1, history[-1]
                        capsule = MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(
                            scene.checkpoint().to_bytes()
                        )
                runs = []
                for item in history:
                    present = tuple(sorted(item[4]["named_limit_loads"]))
                    if not runs or runs[-1] != present:
                        runs.append(present)
                assert runs == [(), (NAMES[1],), tuple(sorted(NAMES)), (NAMES[0],), ()]
                scene.restore(capsule, expected_revision=batch.checkpoint_revision)
                _same(batch.integration_snapshot()[KEY], saved[0])
                with pytest.raises(ValueError):
                    ws.readback(history[-1][2])
                restored = ws.refresh(expected_revision=batch.checkpoint_revision)
                _same(ws.readback(restored), saved[3])
                for index in range(saved_index + 1, len(history)):
                    replay = advance(commands[index])
                    _same(replay[0], history[index][0], fresh_stamp=True)
                    _same(replay[1], history[index][1], fresh_stamp=True)
                    _same(replay[3], history[index][3])


@pytest.mark.cuda
def test_restore_rejects_second_limit_impersonating_first_joint_before_device_write(monkeypatch):
    from phymeo.backends import mujoco_checkpoint as codec

    spec, world = _declaration(q=(-0.00045, 0.0007))
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (_side_asset(),)})
        _drive(batch, spec, np.array([-0.03, 0.1]))
        before = batch.integration_snapshot()[KEY]
        assert scene.step(H, expected_revision=batch.checkpoint_revision) == {KEY: True}
        after = batch.integration_snapshot()[KEY]
        audit = _audit(spec, before, after, scene.last_geometry[KEY][0])
        assert set(audit["named_limit_loads"]) == set(NAMES)
        assert all(force > 0.01 for force in audit["named_limit_loads"].values())
        owner = batch._buckets[batch.layout[KEY].bucket]
        capsule, revision = owner.checkpoint(), owner.checkpoint_revision
        state = dict(capsule._read(capsule.binding))
        world_state = dict(state["worlds"][0])
        receipt = world_state["receipt"]
        index = next(
            i
            for i, row in enumerate(receipt.constraints)
            if row.constraint_type == 3 and row.source_id == 1
        )
        typed, raw = list(receipt.constraints), dict(receipt.raw_contact_fields)
        native = dict(raw["native_contacts"])
        ids = native["efc_ids"].copy()
        ids[index, 1] = 0
        typed[index] = replace(typed[index], source_id=0)
        native["efc_ids"] = ids
        raw["native_contacts"] = MappingProxyType(native)
        world_state["receipt"] = replace(
            receipt, constraints=tuple(typed), raw_contact_fields=MappingProxyType(raw)
        )
        state["worlds"] = (world_state,)
        forged = codec._make_checkpoint(capsule.binding, capsule.source_revision, state)
        monkeypatch.setattr(
            owner,
            "_build_device",
            lambda *a, **kw: pytest.fail("forged second limit reached device"),
        )
        with pytest.raises(ValueError, match="complete nonoverlapping source limit rows"):
            owner.restore(forged, expected_revision=revision)
        assert owner.checkpoint_revision == revision
        assert owner.checkpoint().to_bytes() == capsule.to_bytes()
        _same(batch.integration_snapshot()[KEY], after)


@pytest.mark.cuda
def test_second_limit_exceeds_eight_combined_rows_atomically_and_empty_area_retry_recovers(
    tmp_path,
):
    from types import SimpleNamespace

    from test_damped_coupled_mixed_area_friction import _prescribed

    from phymeo.training.resident_stepper import PhysicsStepFailure

    spec, world = _declaration(q=(0.0, 0.0006))
    spec = replace(spec, constraint_capacity=8)
    target = np.array([-0.03, 0.1])
    with _evidence(tmp_path / "two-limit-capacity.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            accepted = 0
            # One declared stop is already active. Accepted driven states alone
            # bring the other stop into range; no q/v injection or model change.
            for _ in range(16):
                current = batch.integration_snapshot()[KEY]
                keys = _source(spec, current)[3]
                limits = {key for key in keys if key[0] == 3}
                if limits == {(3, 0, -1), (3, 1, 1)}:
                    break
                assert limits == {(3, 1, 1)} and len(keys) == 5
                applied = _drive(batch, spec, target)
                before = batch.integration_snapshot()[KEY]
                surface = _prescribed(batch, KEY)
                assert batch.step_area_pressure(
                    H, {KEY: (surface,)}, expected_revision=batch.checkpoint_revision
                ) == {KEY: True}
                after = batch.integration_snapshot()[KEY]
                (pair,) = after.last_step.raw_contact_fields["area_friction"]
                assert len(after.last_step.constraints) == 5 and len(pair["sample_ids"]) == 3
                audit = _audit(spec, before, after, SimpleNamespace(surface=surface))
                assert audit["named_limit_loads"][NAMES[1]] > 0.01
                evidence["records"].append(
                    dict(before=before, after=after, applied=applied, audit=audit)
                )
                accepted += 1
            else:
                raise AssertionError("finite accepted motion did not activate the second stop")
            assert accepted >= 1
            _drive(batch, spec, target)
            before = batch.integration_snapshot()[KEY]
            source_keys = _source(spec, before)[3]
            assert len(source_keys) == 6
            with batch.prepare_device_workspace(
                include_bodies=True, include_area_pressure=True
            ) as ws:
                frame, arrays = _observe(ws, batch, (spec,), {KEY: before})
                revision, publication = batch.checkpoint_revision, batch.last_step
                checkpoint = batch.checkpoint().to_bytes()
                surface = _prescribed(batch, KEY)
                assert len(surface.triangles) == 1
                evidence.update(rejected_input=surface, before_rejection=before, arrays=arrays)
                # 2 limits + 4 native edges + 3 area samples exceeds the same
                # eight slots that just accepted 1 + 4 + 3.
                with pytest.raises((ValueError, PhysicsStepFailure)) as error:
                    batch.step_area_pressure(H, {KEY: (surface,)}, expected_revision=revision)
                evidence["rejection"] = str(error.value)
                assert batch.checkpoint_revision == revision and batch.last_step is publication
                assert batch.checkpoint().to_bytes() == checkpoint
                _same(batch.integration_snapshot()[KEY], before)
                _same(ws.readback(frame), arrays)

                # Capacity belongs to the model. Change only this step's area
                # input: six native rows and zero area samples fit unchanged.
                empty = _prescribed(batch, KEY, empty=True)
                assert batch.step_area_pressure(H, {KEY: (empty,)}, expected_revision=revision) == {
                    KEY: True
                }
                after = batch.integration_snapshot()[KEY]
                receipt = after.last_step
                assert after.tick == before.tick + 1
                np.testing.assert_array_equal(receipt.input_q, before.q)
                np.testing.assert_array_equal(receipt.input_v, before.v)
                assert receipt.input_time == before.time
                rows = {
                    (r.constraint_type, r.source_id, r.side_or_edge): r for r in receipt.constraints
                }
                assert len(rows) == 6 and set(rows) == set(source_keys)
                assert rows[(3, 0, -1)].force > 0.01 and rows[(3, 1, 1)].force > 0.01
                raw = receipt.raw_contact_fields
                (pair,) = raw["area_friction"]
                assert pair["sample_ids"] == ()
                solver = raw["area_friction_solver"]
                assert solver["schema"] == SCHEMA and solver["solver_status"] == 0
                retry_audit = _audit(spec, before, after, SimpleNamespace(surface=empty))
                _, retry_arrays = _observe(ws, batch, (spec,), {KEY: after})
                evidence.update(retry=after, retry_arrays=retry_arrays, retry_audit=retry_audit)
