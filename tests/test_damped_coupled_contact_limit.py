"""One finite slide limit alongside damped coupled ground and side friction.

The two-slide geometry and independent two-variable root are reused unchanged.
Native forward provides current input row identities and coefficients. A fixed
six-step approach/hold and four-step reverse schedule selects finite applied
forces; only the live owner evolves q/v. Area forces are deliberately omitted
when selecting the drive and included in every independent physical audit.
"""

import xml.etree.ElementTree as ET
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from test_damped_coupled_mixed_area_friction import (
    DAMPING,
    MASS,
    H,
    U,
    W,
    _observe,
    _prescribed,
    _root,
    _value,
)
from test_damped_coupled_mixed_area_friction import (
    _declaration as _unlimited,
)
from test_damped_hinge_mixed_area_friction import _side_asset
from test_mujoco_area_friction_lifecycle import _evidence, _same
from test_mujoco_area_friction_live import _surface as _audit_surface

from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
from phymeo.backends.mujoco_heterogeneous_area_scene import (
    MuJoCoTyxHeterogeneousAreaPressureScene,
    MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.solvers.area_friction import LAW
from phymeo.training.resident_stepper import PhysicsStepFailure

KEY = 41
SCHEMA = "mujoco-tyx-area-friction-coupled-contact-limit-v1"
LOAD_STEPS, STEPS = 6, 10


def _declaration(side=1, *, capacity=64):
    base, _ = _unlimited(capacity=capacity)
    tree = ET.fromstring(base.compilation.source.primary_xml)
    joint = tree.find(".//joint[@name='child-slide']")
    joint.attrib.update(
        limited="true",
        range="-.02 .0005" if side == 1 else "-.0005 .02",
        margin="0",
        solreflimit=".02 1",
        solimplimit=".5 .5 .001 .5 2",
    )
    spec = replace(
        base,
        key=f"damped-ground-side-limit-{side}",
        compilation=compile_mujoco_xml(ET.tostring(tree, encoding="unicode"), require_asset=False),
    )
    world = MuJoCoTyxWorldSpec(KEY, spec.key, MuJoCoInitialState((0.0, 0.0), (0.02, side * 0.1)))
    return spec, world


def _native(spec, state, *, ground=True):
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
    assert model.nq == model.nv == 2 and model.neq == model.ntendon == 0
    assert np.count_nonzero(model.jnt_limited) == 1 and model.jnt_limited[1]
    assert not np.any(model.dof_frictionloss)
    np.testing.assert_array_equal(model.dof_damping, DAMPING)
    mass = np.empty((2, 2))
    mj.mj_fullM(model, mass, data.qM)
    np.testing.assert_allclose(mass, MASS, rtol=0, atol=1e-13)
    assert data.ncon == int(ground)
    if ground:
        assert data.contact[0].dim == 3
    jac = data.efc_J.reshape(data.nefc, 2)
    sources, edge = [], 0
    for index in range(data.nefc):
        kind, identity = int(data.efc_type[index]), int(data.efc_id[index])
        assert kind in (3, 6)
        if kind == 3:
            dof = int(model.jnt_dofadr[identity])
            side = -1 if jac[index, dof] > 0 else 1
            assert identity == 1 and dof == 1
            np.testing.assert_array_equal(jac[index], (0, -side))
        else:
            assert identity == 0
            side, dof = edge, -1
            edge += 1
        sources.append(
            dict(
                index=index,
                key=(kind, identity, side),
                dof=dof,
                j=jac[index].copy(),
                r=float(data.efc_R[index]),
                aref=float(data.efc_aref[index]),
                position=float(data.efc_pos[index]),
                margin=float(data.efc_margin[index]),
                velocity=float(data.efc_vel[index]),
            )
        )
    assert edge == 4 * int(ground) and sum(row["key"][0] == 3 for row in sources) <= 1
    lower, upper = model.jnt_range[1]
    margin = model.jnt_margin[1]
    active = [
        side
        for side, distance in ((-1, state.q[1] - lower), (1, upper - state.q[1]))
        if distance < margin
    ]
    assert [row["key"][2] for row in sources if row["key"][0] == 3] == active
    return model, data, sources


def _drive(batch, spec, side, step):
    # Invert only current source native rows to select finite motor loads.
    # The production step also sees the genuine side load, so this is not a
    # state assignment or an assertion that actual v exactly tracks target v.
    state = batch.integration_snapshot()[KEY]
    _model, data, sources = _native(spec, state)
    target = np.array([0.02, side * (0.1 if step < LOAD_STEPS else -0.2)])
    h = float(np.float32(H))
    response = MASS + h * np.diag(DAMPING)
    acceleration = np.linalg.solve(MASS, response @ (target - state.v)) / h
    jac = np.array([row["j"] for row in sources])
    loads = np.maximum(0.0, (data.efc_aref - jac @ acceleration) / data.efc_R)
    applied = MASS @ acceleration - (data.qfrc_passive - data.qfrc_bias) - jac.T @ loads
    assert np.isfinite(applied).all()
    batch.set_controls({KEY: MuJoCoControl((), qfrc_applied=applied)})
    return dict(
        target=target,
        applied=applied,
        native_only_force=loads,
        source_q=state.q,
        source_v=state.v,
        source_keys=tuple(row["key"] for row in sources),
    )


def _audit(spec, side, before, after, geometry, *, require_area=True, ground_present=True):
    _model, data, sources = _native(spec, before, ground=ground_present)
    receipt = after.last_step
    np.testing.assert_array_equal(receipt.input_q, before.q)
    np.testing.assert_array_equal(receipt.input_v, before.v)
    assert receipt.tick == after.tick == before.tick + 1
    assert receipt.input_time == receipt.contact_time == receipt.dynamics.time == before.time
    raw = receipt.raw_contact_fields
    (pair,) = raw["area_friction"]
    solver = raw["area_friction_solver"]
    assert solver["schema"] == SCHEMA and solver["law"] == pair["law"] == LAW
    assert solver["solver_status"] == 0 and raw["native_contact_operator"] == "M"
    _audit_surface(pair, geometry, receipt)
    rows = {
        (row.constraint_type, row.source_id, row.side_or_edge): row for row in receipt.constraints
    }
    assert len(rows) == len(receipt.constraints) == len(sources)
    assert set(rows) == {source["key"] for source in sources}
    ordered = [rows[source["key"]] for source in sources]
    jac = np.array([row.jacobian for row in ordered])
    reg = np.array([row.regularization for row in ordered])
    aref = np.array([row.reference_acceleration for row in ordered])
    forces = np.array([row.force for row in ordered])
    for source, row in zip(sources, ordered, strict=True):
        np.testing.assert_allclose(row.jacobian, source["j"], rtol=2e-5, atol=2e-7)
        assert row.regularization == pytest.approx(source["r"], abs=2e-5, rel=4e-4)
        assert row.reference_acceleration == pytest.approx(source["aref"], abs=2e-5, rel=4e-4)
        for name in ("position", "margin", "velocity"):
            assert getattr(row, name) == pytest.approx(source[name], abs=2e-7, rel=4e-4)
        assert row.lower == 0 and row.upper is None and row.force >= 0
        if source["key"][0] == 3:
            assert row.side_or_edge == side and row.dof == 1 and row.geometry_slot == -1
            np.testing.assert_array_equal(row.jacobian, (0, -side))
    limit = [row for row in ordered if row.constraint_type == 3]
    ground = [rows[(6, 0, edge)] for edge in range(4)] if ground_present else []
    h, mass = solver["h_s"], solver["mass_matrix"].astype(float)
    assert h == float(np.float32(H))
    np.testing.assert_allclose(mass, MASS, rtol=4e-4, atol=2e-5)
    np.testing.assert_array_equal(solver["damping"], DAMPING)
    np.testing.assert_array_equal(solver["input_velocity"], before.v)
    np.testing.assert_array_equal(solver["output_velocity"], after.v)
    dyn = receipt.dynamics
    for actual, expected in (
        (dyn.bias, data.qfrc_bias),
        (dyn.passive, data.qfrc_passive),
        (dyn.applied, data.qfrc_applied + data.qfrc_actuator),
    ):
        np.testing.assert_allclose(actual, expected, rtol=4e-4, atol=2e-5)
    response = mass + h * np.diag(DAMPING)
    applied = solver["applied_force"].astype(float)
    np.testing.assert_allclose(
        applied, dyn.applied + dyn.passive - dyn.bias + DAMPING * before.v, rtol=2e-5, atol=2e-5
    )
    ns = len(pair["sample_ids"])
    point_j = np.zeros((ns, 3, 2))
    point_j[:, :, 0] = U
    np.testing.assert_allclose(pair["jacobian_a"], point_j, rtol=0, atol=2e-7)
    np.testing.assert_array_equal(pair["jacobian_b"], 0)
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
    np.testing.assert_allclose(after.v, expected_v, rtol=5e-5, atol=5e-5)
    actual = _value(after.v.astype(float), p)
    for got, wanted in (
        (forces, expected["native"]),
        (forces, actual["native"]),
        (pair["normal_force"], expected["normal"]),
        (pair["tangential_force"], expected["tangent"]),
        (pair["force_on_a"], expected["force"]),
        (pair["normal_force"], actual["normal"]),
        (pair["tangential_force"], actual["tangent"]),
        (pair["force_on_a"], actual["force"]),
    ):
        np.testing.assert_allclose(got, wanted, rtol=0.002, atol=0.002)
    np.testing.assert_allclose(pair["relative_velocity"], actual["relative"], rtol=2e-5, atol=2e-5)
    ft, fn, force = (
        pair[name].astype(float) for name in ("tangential_force", "normal_force", "force_on_a")
    )
    if require_area:
        assert fn.sum() > 0.1 and np.linalg.norm(ft) > 0.001
    assert np.all(np.linalg.norm(ft, axis=1) <= pair["mu"] * fn + 1e-5)
    assert np.all(np.sum(ft * actual["tangent_velocity"], axis=1) <= 1e-14)
    np.testing.assert_array_equal(pair["force_on_b"], -pair["force_on_a"])
    np.testing.assert_array_equal(
        pair["torque_on_b_world_origin"], -pair["torque_on_a_world_origin"]
    )
    np.testing.assert_allclose(
        pair["torque_on_a_world_origin"], np.cross(pair["points"], force), rtol=0.002, atol=1e-5
    )
    ground_force = np.zeros(3)
    assert len(receipt.contacts) == int(ground_present)
    if ground_present:
        edge_loads = np.array([row.force for row in ground])
        contact = data.contact[0]
        local = np.array(
            [
                edge_loads.sum(),
                contact.friction[0] * (edge_loads[0] - edge_loads[1]),
                contact.friction[1] * (edge_loads[2] - edge_loads[3]),
            ]
        )
        ground_force = contact.frame.reshape(3, 3).T @ local
        assert ground_force[2] > 0.1
        np.testing.assert_allclose(
            receipt.contacts[0].force_on_geometry2_world, ground_force, rtol=0.002, atol=0.002
        )
        np.testing.assert_array_equal(
            receipt.contacts[0].force_on_geometry1_world,
            -receipt.contacts[0].force_on_geometry2_world,
        )
    ground_q = np.array([U @ ground_force, W @ ground_force])
    area_q = np.array([U @ force.sum(axis=0), 0.0])
    limit_q = np.array([0.0, -side * sum(row.force for row in limit)])
    assert side * limit_q[1] <= 0
    np.testing.assert_allclose(
        ground_q,
        sum((row.jacobian * row.force for row in ground), np.zeros(2)),
        rtol=2e-5,
        atol=2e-5,
    )
    total_q = ground_q + area_q + limit_q
    np.testing.assert_allclose(receipt.qfrc_constraint, total_q, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(receipt.qfrc_constraint, solver["generalized_force"])
    rhs = dyn.applied + dyn.passive - dyn.bias + total_q
    physical, integration = np.linalg.solve(mass, rhs), np.linalg.solve(response, rhs)
    np.testing.assert_allclose(h * receipt.qacc, h * physical, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(
        h * receipt.integration_acceleration, h * integration, rtol=5e-5, atol=5e-5
    )
    np.testing.assert_allclose(after.v, before.v + h * integration, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(after.q, before.q + h * after.v, rtol=0, atol=2e-7)
    power = (
        np.sum(force * actual["relative"])
        + ground_force @ (U * after.v[0] + W * after.v[1])
        + limit_q @ after.v
    )
    assert total_q @ after.v == pytest.approx(power, abs=2e-5, rel=2e-5)
    gradients = jac @ receipt.qacc + reg * forces - aref
    np.testing.assert_allclose([row.gradient for row in ordered], gradients, rtol=2e-5, atol=2e-5)
    assert np.all(gradients >= -2e-5 * (1 + np.abs(aref)))
    factor = np.linalg.cholesky(response)
    scale = 1 + np.linalg.norm(np.linalg.solve(factor, p["rhs"]))
    for residual in (actual["residual"], response @ after.v - p["rhs"] - h * total_q):
        assert np.linalg.norm(np.linalg.solve(factor, residual)) <= 2e-5 * scale
    assert solver["normalized_residual"] <= solver["tolerance"] == 2e-5
    return dict(
        limit_present=bool(limit),
        limit_force=sum(row.force for row in limit),
        limit_q=limit_q,
        ground_q=ground_q,
        area_q=area_q,
        source_rows=sources,
        oracle=expected,
        newton_iterations=iterations,
    )


@pytest.mark.cuda
@pytest.mark.parametrize("side", [1, -1], ids=["upper", "lower"])
def test_contact_limit_enters_holds_releases_and_replays_with_both_contacts_loaded(side, tmp_path):
    spec, world = _declaration(side)
    with _evidence(tmp_path / "contact-limit-window.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            caps = batch.capabilities["models"][spec.key]
            assert caps["domain"] == "area-friction-cuda-damped-coupled-support-limit-v1"
            assert caps["native_scalar_limit_joint_ids"] == (1,)
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (_side_asset(),)})
            with batch.prepare_device_workspace(
                include_bodies=True, include_area_pressure=True
            ) as ws:

                def advance(step):
                    previous = batch.integration_snapshot()[KEY]
                    drive = _drive(batch, spec, side, step)
                    before = batch.integration_snapshot()[KEY]
                    np.testing.assert_array_equal(before.q, previous.q)
                    np.testing.assert_array_equal(before.v, previous.v)
                    assert before.time == previous.time and before.tick == previous.tick
                    assert scene.step(H, expected_revision=batch.checkpoint_revision) == {KEY: True}
                    after, geometry = batch.integration_snapshot()[KEY], scene.last_geometry[KEY][0]
                    record = dict(
                        step=step, before=before, after=after, geometry=geometry, drive=drive
                    )
                    evidence["records"].append(record)
                    record["audit"] = _audit(spec, side, before, after, geometry)
                    frame, arrays = _observe(ws, batch, (spec,), {KEY: after})
                    record["arrays"] = arrays
                    return after, geometry, frame, arrays, record["audit"]

                history = []
                for step in range(STEPS):
                    history.append(advance(step))
                    if step == 4:
                        saved = history[-1]
                        capsule = MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(
                            scene.checkpoint().to_bytes()
                        )
                phases = [item[4] for item in history]
                assert not phases[0]["limit_present"]
                loaded = [i for i in range(LOAD_STEPS) if phases[i]["limit_force"] > 0.01]
                assert len(loaded) >= 2 and any(b == a + 1 for a, b in zip(loaded, loaded[1:]))
                assert all(side * history[i][0].v[1] > 0 for i in loaded)
                assert any(
                    phases[i]["limit_present"] and phases[i]["limit_force"] <= 0.002
                    for i in range(LOAD_STEPS, STEPS)
                )
                assert all(side * history[i][0].v[1] < 0 for i in range(LOAD_STEPS, STEPS))
                assert not phases[-1]["limit_present"] and not phases[-2]["limit_present"]
                assert side * history[-1][0].q[1] < 0.0005
                scene.restore(capsule, expected_revision=batch.checkpoint_revision)
                _same(batch.integration_snapshot()[KEY], saved[0])
                with pytest.raises(ValueError):
                    ws.readback(history[-1][2])
                restored_frame = ws.refresh(expected_revision=batch.checkpoint_revision)
                _same(ws.readback(restored_frame), saved[3])
                for step in range(5, STEPS):
                    replay = advance(step)
                    _same(replay[0], history[step][0], fresh_stamp=True)
                    _same(replay[1], history[step][1], fresh_stamp=True)
                    _same(replay[3], history[step][3])


def _approach_active_limit(batch, spec):
    # Accepted approaches use four ground rows + one triangle (seven slots).
    # Stop at the first genuine current limit input before attempting its step.
    for step in range(6):
        state = batch.integration_snapshot()[KEY]
        if any(row["key"][0] == 3 for row in _native(spec, state)[2]):
            return state
        _drive(batch, spec, 1, step)
        surface = _prescribed(batch, KEY)
        assert batch.step_area_pressure(
            H, {KEY: (surface,)}, expected_revision=batch.checkpoint_revision
        ) == {KEY: True}
    raise AssertionError("driven accepted states did not reach the finite upper limit")


@pytest.mark.cuda
def test_five_native_rows_plus_area_capacity_failure_is_atomic_and_native_retry_recovers(tmp_path):
    spec, world = _declaration(capacity=7)
    with _evidence(tmp_path / "contact-limit-capacity.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            _approach_active_limit(batch, spec)
            _drive(batch, spec, 1, 4)
            before = batch.integration_snapshot()[KEY]
            assert len(_native(spec, before)[2]) == 5
            with batch.prepare_device_workspace(
                include_bodies=True, include_area_pressure=True
            ) as ws:
                frame, arrays = _observe(ws, batch, (spec,), {KEY: before})
                revision, publication = batch.checkpoint_revision, batch.last_step
                checkpoint = batch.checkpoint().to_bytes()
                surface = _prescribed(batch, KEY)
                evidence.update(before=before, surface=surface, arrays=arrays)
                with pytest.raises((ValueError, PhysicsStepFailure)):
                    batch.step_area_pressure(H, {KEY: (surface,)}, expected_revision=revision)
                assert batch.checkpoint_revision == revision and batch.last_step is publication
                assert batch.checkpoint().to_bytes() == checkpoint
                _same(batch.integration_snapshot()[KEY], before)
                _same(ws.readback(frame), arrays)
                empty = _prescribed(batch, KEY, empty=True)
                assert batch.step_area_pressure(H, {KEY: (empty,)}, expected_revision=revision) == {
                    KEY: True
                }
                after = batch.integration_snapshot()[KEY]
                evidence["retry"] = after
                audit = _audit(
                    spec, 1, before, after, SimpleNamespace(surface=empty), require_area=False
                )
                assert audit["limit_present"] and len(after.last_step.constraints) == 5
                np.testing.assert_array_equal(audit["area_q"], 0)
                assert after.tick == before.tick + 1


@pytest.mark.cuda
@pytest.mark.parametrize("fault", ["side", "source-id"])
def test_restore_rejects_forged_limit_identity_before_device_writes(monkeypatch, fault):
    from phymeo.backends import mujoco_checkpoint as codec

    spec, world = _declaration()
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        _approach_active_limit(batch, spec)
        _drive(batch, spec, 1, 4)
        surface = _prescribed(batch, KEY)
        assert batch.step_area_pressure(
            H, {KEY: (surface,)}, expected_revision=batch.checkpoint_revision
        ) == {KEY: True}
        owner = batch._buckets[batch.layout[KEY].bucket]
        capsule, revision = owner.checkpoint(), owner.checkpoint_revision
        original = batch.integration_snapshot()
        state = dict(capsule._read(capsule.binding))
        world_state = dict(state["worlds"][0])
        receipt = world_state["receipt"]
        index = next(i for i, row in enumerate(receipt.constraints) if row.constraint_type == 3)
        typed = list(receipt.constraints)
        raw = dict(receipt.raw_contact_fields)
        native = dict(raw["native_contacts"])
        ids = native["efc_ids"].copy()
        if fault == "side":
            ids[index, 2] *= -1
            typed[index] = replace(typed[index], side_or_edge=int(ids[index, 2]))
        else:
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
            owner, "_build_device", lambda *a, **kw: pytest.fail("forged limit reached device")
        )
        with pytest.raises(ValueError, match="complete nonoverlapping source limit rows"):
            owner.restore(forged, expected_revision=revision)
        assert owner.checkpoint_revision == revision
        assert owner.checkpoint().to_bytes() == capsule.to_bytes()
        _same(batch.integration_snapshot(), original)


@pytest.mark.cuda
def test_positive_margin_limit_carries_area_load_with_separated_ground_and_restores(tmp_path):
    base, world = _declaration()
    tree = ET.fromstring(base.compilation.source.primary_xml)
    tree.find(".//geom[@name='foot']").set("pos", ".12 0 .15")
    tree.find(".//joint[@name='child-slide']").set("margin", ".001")
    spec = replace(
        base,
        compilation=compile_mujoco_xml(ET.tostring(tree, encoding="unicode"), require_asset=False),
    )
    with _evidence(tmp_path / "positive-margin-separated-ground.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (_side_asset(),)})

            def advance():
                before = batch.integration_snapshot()[KEY]
                assert scene.step(H, expected_revision=batch.checkpoint_revision) == {KEY: True}
                after = batch.integration_snapshot()[KEY]
                geometry = scene.last_geometry[KEY][0]
                evidence["records"].append(dict(before=before, after=after, geometry=geometry))
                audit = _audit(spec, 1, before, after, geometry, ground_present=False)
                assert audit["limit_force"] > 0.01
                assert len(after.last_step.constraints) == 1
                if before.tick == 0:
                    row = after.last_step.constraints[0]
                    assert 0 < row.position < row.margin
                return after, geometry

            first, _ = advance()
            capsule = MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(
                scene.checkpoint().to_bytes()
            )
            expected = advance()
            scene.restore(capsule, expected_revision=batch.checkpoint_revision)
            _same(batch.integration_snapshot()[KEY], first)
            replay = advance()
            _same(replay, expected, fresh_stamp=True)


@pytest.mark.cuda
@pytest.mark.parametrize("slide_limit", [False, True], ids=["hinge-only", "slide-and-hinge"])
def test_margin_hinge_limit_delivers_signed_torque_with_both_contacts_loaded(tmp_path, slide_limit):
    """The second generalized load is torque about the hinge axis."""
    import mujoco as mj
    import test_damped_coupled_articulated_contact as articulated

    base, original_world = articulated._declaration()
    tree = ET.fromstring(base.compilation.source.primary_xml)
    # Keep the original q/v, foot and area asset. The lower stop's +y reaction
    # presses the foot down; its positive soft margin reaches the initial q=0.
    tree.find(".//joint[@name='rotation']").attrib.update(
        limited="true",
        range="-.001 .5",
        margin=".02",
        solreflimit=".02 1",
        solimplimit=".5 .5 .001 .5 2",
    )
    if slide_limit:
        tree.find(".//joint[@name='translation']").attrib.update(
            limited="true",
            range="-.001 .2",
            margin=".004",
            solreflimit=".02 1",
            solimplimit=".5 .5 .001 .5 2",
        )
    spec = replace(
        base,
        key="damped-slide-hinge-lower-margin",
        compilation=compile_mujoco_xml(ET.tostring(tree, encoding="unicode"), require_asset=False),
    )
    world = replace(original_world, model_key=spec.key)
    with _evidence(tmp_path / "hinge-limit-torque.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (_side_asset(),)})
            before = batch.integration_snapshot()[KEY]
            model = spec.compilation.fresh_native_model()
            source = mj.MjData(model)
            source.qpos[:], source.qvel[:], source.time = before.q, before.v, before.time
            mj.mj_forward(model, source)
            assert model.jnt_type[1] == mj.mjtJoint.mjJNT_HINGE
            assert tuple(model.jnt_limited) == (slide_limit, True)
            assert (
                source.ncon == 1
                and source.nefc == 5 + int(slide_limit)
                and source.contact[0].dim == 3
            )
            (limit_index,) = np.flatnonzero((source.efc_type == 3) & (source.efc_id == 1))
            source_j = source.efc_J.reshape(source.nefc, 2)
            assert source.efc_id[limit_index] == 1
            assert 0 < source.efc_pos[limit_index] < source.efc_margin[limit_index]
            np.testing.assert_array_equal(source_j[limit_index], [0, 1])
            np.testing.assert_array_equal(source.xaxis[1], articulated.AXIS)
            mass = np.empty((2, 2))
            mj.mj_fullM(model, mass, source.qM)
            np.testing.assert_allclose(mass, [[2, 0.04], [0.04, 0.0325]], rtol=0, atol=1e-13)
            assert scene.step(H, expected_revision=batch.checkpoint_revision) == {KEY: True}
            after = batch.integration_snapshot()[KEY]
            receipt = after.last_step
            (pair,) = receipt.raw_contact_fields["area_friction"]
            solver = receipt.raw_contact_fields["area_friction_solver"]
            expected_schema = (
                "mujoco-tyx-area-friction-coupled-contact-two-limits-v1" if slide_limit else SCHEMA
            )
            assert solver["schema"] == expected_schema and solver["solver_status"] == 0
            rows = receipt.constraints
            np.testing.assert_array_equal([row.constraint_type for row in rows], source.efc_type)
            np.testing.assert_array_equal([row.source_id for row in rows], source.efc_id)
            jac, reg, aref, loads = (
                np.array([getattr(row, field) for row in rows])
                for field in ("jacobian", "regularization", "reference_acceleration", "force")
            )
            for actual, expected in (
                (jac, source_j),
                (reg, source.efc_R),
                (aref, source.efc_aref),
                (solver["mass_matrix"], mass),
            ):
                np.testing.assert_allclose(actual, expected, rtol=4e-4, atol=2e-5)
            limit = rows[limit_index]
            assert (limit.dof, limit.side_or_edge, limit.geometry_slot) == (1, -1, -1)
            assert limit.lower == 0 and limit.upper is None and limit.force > 0.01
            np.testing.assert_array_equal(limit.jacobian, [0, 1])
            pivot = articulated.PIVOT0 + articulated.SLIDE * before.q[0]
            ja, jb = articulated._point_jacobians(
                model, source, pair["points"].astype(float), pivot
            )
            np.testing.assert_allclose(pair["jacobian_a"], ja, rtol=2e-5, atol=2e-7)
            np.testing.assert_array_equal(pair["jacobian_b"], jb)
            h, packed_mass = solver["h_s"], solver["mass_matrix"].astype(float)
            np.testing.assert_array_equal(solver["damping"], articulated.DAMPING)
            response = packed_mass + h * np.diag(articulated.DAMPING)
            p = dict(
                mass=packed_mass,
                response=response,
                h=h,
                v0=before.v.astype(float),
                rhs=packed_mass @ before.v + h * solver["applied_force"],
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
            expected_v, expected, _ = _root(p)
            np.testing.assert_allclose(after.v, expected_v, rtol=5e-5, atol=5e-5)
            np.testing.assert_allclose(
                h * receipt.qacc, h * expected["acceleration"], rtol=5e-5, atol=5e-5
            )
            for actual, wanted in (
                (loads, expected["native"]),
                (pair["force_on_a"], expected["force"]),
            ):
                np.testing.assert_allclose(actual, wanted, rtol=0.002, atol=0.002)
            projected = max(
                0.0,
                (source.efc_aref[limit_index] - source_j[limit_index] @ receipt.qacc)
                / source.efc_R[limit_index],
            )
            assert limit.force == pytest.approx(projected, rel=0.002, abs=0.002)
            assert (
                pair["normal_force"].sum() > 0.1
                and np.linalg.norm(pair["tangential_force"]) > 0.001
            )
            edges = sorted(
                (row for row in rows if row.constraint_type == 6), key=lambda row: row.side_or_edge
            )
            assert len(edges) == 4 and len(receipt.contacts) == 1
            edge_loads = np.array([row.force for row in edges])
            contact = source.contact[0]
            ground_force = contact.frame.reshape(3, 3).T @ np.array(
                [
                    edge_loads.sum(),
                    contact.friction[0] * (edge_loads[0] - edge_loads[1]),
                    contact.friction[1] * (edge_loads[2] - edge_loads[3]),
                ]
            )
            assert ground_force[2] > 0.1
            np.testing.assert_allclose(
                receipt.contacts[0].force_on_geometry2_world, ground_force, rtol=0.002, atol=0.002
            )
            force = pair["force_on_a"].astype(float)
            area_moment = np.cross(pair["points"] - pivot, force).sum(axis=0)
            ground_moment = np.cross(contact.pos - pivot, ground_force)
            limit_moment = limit.force * source.xaxis[1]
            assert articulated.AXIS @ limit_moment > 0.01
            slide_limit_force = 0.0
            if slide_limit:
                (slide,) = [row for row in rows if row.constraint_type == 3 and row.source_id == 0]
                assert (slide.dof, slide.side_or_edge) == (0, -1)
                np.testing.assert_array_equal(slide.jacobian, [1, 0])
                assert slide.force > 0.01
                slide_limit_force = slide.force
            total = np.array(
                [
                    articulated.SLIDE @ (force.sum(axis=0) + ground_force) + slide_limit_force,
                    articulated.AXIS @ (area_moment + ground_moment + limit_moment),
                ]
            )
            np.testing.assert_allclose(receipt.qfrc_constraint, total, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(
                after.v, before.v + h * receipt.integration_acceleration, rtol=2e-6, atol=2e-7
            )
            np.testing.assert_allclose(after.q, before.q + h * after.v, rtol=0, atol=2e-7)
            evidence.update(
                before=before,
                after=after,
                oracle=expected,
                slide_limit_force_n=slide_limit_force,
                limit_torque_world_nm=limit_moment,
                area_torque_world_nm=area_moment,
                ground_torque_world_nm=ground_moment,
            )
