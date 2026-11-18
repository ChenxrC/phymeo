"""A real one-slider Scene couples area load to native guide dry friction.

Native forward supplies independent entry M/J and dry-row R/aref, not the
mixed solution. The one-dimensional oracle solves the joint equation afresh.
Only Scene.step may advance the physical owner.
"""

import gzip
import json
import xml.etree.ElementTree as ET
from dataclasses import replace

import numpy as np
import pytest
from test_mujoco_area_friction import _close, _plain
from test_mujoco_area_friction_lifecycle import _same
from test_mujoco_area_friction_live import _surface

from experiments.area_pressure.free_body_support import case_declarations
from phymeo.backends.mujoco_area_friction import AreaFrictionPairLaw
from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
from phymeo.backends.mujoco_heterogeneous_area_scene import (
    MuJoCoTyxHeterogeneousAreaPressureScene,
    MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
)
from phymeo.backends.mujoco_runtime import MuJoCoInitialState
from phymeo.backends.tyx_area_friction import TyxAreaFrictionWorkspace
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.solvers.area_friction import LAW
from phymeo.training.resident_stepper import PhysicsStepFailure

H = 0.01
LOSSES = (2.0, 0.25)


def _declarations(loss, key):
    original, _, original_assets, _ = case_declarations()
    base = original[0]
    tree = ET.fromstring(base.compilation.source.primary_xml)
    option = tree.find("option")
    option.set("gravity", "0 0 0")
    option.set("timestep", str(H))
    body = tree.find("./worldbody/body[@name='pressure']")
    body.remove(body.find("freejoint"))
    body.insert(
        0,
        ET.Element(
            "joint",
            name="guide",
            type="slide",
            axis="0 0 1",
            limited="false",
            damping="0",
            frictionloss=str(loss),
            armature="0",
            solimpfriction=".5 .5 .001 .5 2",
        ),
    )
    body.find("inertial").set("mass", "1")
    body.find("inertial").set(
        "diaginertia", ".00016666666666666666 .00016666666666666666 .0002666666666666667"
    )
    xml = ET.tostring(tree, encoding="unicode")
    original_asset = original_assets[base.key][0]
    pressure = replace(
        original_asset.pressure_local,
        nodal_pressure=original_asset.pressure_local.nodal_pressure * 0.25,
    )
    asset = replace(original_asset, pressure_local=pressure)
    spec = replace(
        base,
        key=f"area-guide-{loss}",
        compilation=compile_mujoco_xml(xml, require_asset=False),
        area_friction_laws=(AreaFrictionPairLaw(asset.identity, 0.5, 0.001),),
    )
    world = MuJoCoTyxWorldSpec(key, spec.key, MuJoCoInitialState((-0.002,), (0.0,)))
    return spec, world, {spec.key: (asset,)}


def _native(spec, before, pairs):
    import mujoco as mj

    model = spec.compilation.fresh_native_model()
    assert model.nq == model.nv == 1 and model.neq == model.ntendon == 0
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = before.q, before.v, before.time
    control = before.held_control
    data.ctrl[:] = control.ctrl
    if control.qfrc_applied is not None:
        data.qfrc_applied[:] = control.qfrc_applied
    if control.xfrc_applied is not None:
        data.xfrc_applied[:] = control.xfrc_applied
    mj.mj_forward(model, data)
    assert data.ncon == 0 and data.nefc == 1
    assert int(data.efc_type[0]) == int(mj.mjtConstraint.mjCNSTR_FRICTION_DOF)
    assert int(data.efc_id[0]) == 0 and not np.any(model.dof_damping)
    mass = np.empty((1, 1))
    mj.mj_fullM(model, mass, data.qM)
    ja, jb = [], []
    for pair in pairs:
        ids = tuple(
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            for name in (pair["identity"].compliant_body, pair["identity"].rigid_body)
        )
        assert ids == tuple(pair["body_ids"])
        for point in pair["points"]:
            left, right, angular = (np.empty((3, 1)) for _ in range(3))
            mj.mj_jac(model, data, left, angular, point, ids[0])
            mj.mj_jac(model, data, right, angular, point, ids[1])
            ja.append(left.copy())
            jb.append(right.copy())
    return dict(
        mass=mass,
        bias=data.qfrc_bias.copy(),
        passive=data.qfrc_passive.copy(),
        applied=data.qfrc_applied.copy() + data.qfrc_actuator.copy(),
        ja=np.asarray(ja).reshape(-1, 3, 1),
        jb=np.asarray(jb).reshape(-1, 3, 1),
        body_position=data.xpos.copy(),
        body_rotation=data.xmat.reshape(-1, 3, 3).copy(),
        dry_jacobian=data.efc_J.reshape(1, 1).copy(),
        dry_regularization=float(data.efc_R[0]),
        dry_reference_acceleration=float(data.efc_aref[0]),
        dry_limit=float(data.efc_frictionloss[0]),
        native_only_force=float(data.efc_force[0]),
    )


def _scalar_value(
    velocity,
    *,
    mass,
    input_velocity,
    applied,
    h,
    normal_rows,
    stiffness,
    gap,
    dry_jacobian,
    regularization,
    reference,
    limit,
):
    """Acceleration-level MuJoCo dry-row law on the same new velocity as area."""
    acceleration = (velocity - input_velocity) / h
    normal = np.maximum(0.0, -stiffness * (gap + h * normal_rows * velocity))
    dry = float(np.clip((reference - dry_jacobian * acceleration) / regularization, -limit, limit))
    residual = mass * acceleration - applied - float(normal_rows @ normal) - dry_jacobian * dry
    return residual, normal, dry, acceleration


def _scalar_root(**parameters):
    # Independent fixed monotone bracket, no production Newton or contact solver.
    assert parameters["mass"] > 0 and parameters["regularization"] > 0
    lower, upper = -1.0, 1.0
    assert _scalar_value(lower, **parameters)[0] < 0 < _scalar_value(upper, **parameters)[0]
    for _ in range(96):
        middle = (lower + upper) / 2
        if _scalar_value(middle, **parameters)[0] > 0:
            upper = middle
        else:
            lower = middle
    velocity = (lower + upper) / 2
    residual, normal, dry, acceleration = _scalar_value(velocity, **parameters)
    return dict(
        velocity=velocity,
        residual=residual,
        normal=normal,
        dry=dry,
        acceleration=acceleration,
        parameters=parameters,
    )


def test_report_constant_rows_have_the_two_hand_solutions():
    common = dict(
        mass=1.0,
        input_velocity=0.0,
        applied=0.0,
        h=0.01,
        normal_rows=np.array([1.0]),
        stiffness=np.array([1000.0]),
        gap=np.array([-0.001]),
        dry_jacobian=1.0,
        regularization=1.0,
        reference=0.0,
    )
    for limit, acceleration, normal, dry in (
        (2.0, 10 / 21, 20 / 21, -10 / 21),
        (0.25, 15 / 22, 41 / 44, -0.25),
    ):
        answer = _scalar_root(**common, limit=limit)
        assert answer["acceleration"] == pytest.approx(acceleration, rel=0, abs=1e-13)
        assert answer["normal"][0] == pytest.approx(normal, rel=0, abs=1e-13)
        assert answer["dry"] == pytest.approx(dry, rel=0, abs=1e-13)


def _published(receipt):
    raw = receipt.raw_contact_fields
    assert "area_pressure" not in raw and receipt.contacts == ()
    assert len(receipt.constraints) == 1
    dry = receipt.constraints[0]
    assert dry.constraint_type == 1 and dry.source_id == dry.dof == 0
    assert dry.jacobian.shape == (1,)
    pairs, solver = raw["area_friction"], raw["area_friction_solver"]
    assert len(pairs) == 1
    assert solver["schema"] == "mujoco-tyx-area-friction-guide-v1"
    assert solver["law"] == LAW and solver["solver_status"] == 0
    assert solver["precision"] == "actual-packed-point-fp32"
    assert pairs[0]["law"] == LAW
    assert solver["mass_matrix"].shape == (1, 1)
    np.testing.assert_array_equal(solver["generalized_force"], receipt.qfrc_constraint)
    return pairs, dry, solver


def _audit(spec, loss, before, after, geometry, evidence):
    import mujoco as mj

    receipt = after.last_step
    pairs, dry, solver = _published(receipt)
    pair = pairs[0]
    assert receipt.world == before.world == after.world
    assert receipt.tick == after.tick == before.tick + 1
    assert receipt.input_time == before.time == receipt.contact_time == receipt.dynamics.time
    assert receipt.output_time == after.time
    assert receipt.contact_phase == "tyx-euler-input-constraint-evaluation"
    assert receipt.dynamics.phase == "tyx-euler-input-dynamics"
    np.testing.assert_array_equal(receipt.input_q, before.q)
    np.testing.assert_array_equal(receipt.input_v, before.v)
    evidence["surface"] = _surface(pair, geometry[0], receipt)
    assert len(geometry) == 1 and len(pair["sample_ids"]) > 0
    native = evidence["native"] = _native(spec, before, pairs)
    _close(solver["mass_matrix"], native["mass"], 2e-6, 3e-4)
    _close(pair["jacobian_a"], native["ja"], 2e-6, 2e-5)
    _close(pair["jacobian_b"], native["jb"], 2e-6, 2e-5)
    np.testing.assert_array_equal(
        native["ja"][:, :, 0], np.tile((0.0, 0.0, 1.0), (len(pair["sample_ids"]), 1))
    )
    np.testing.assert_array_equal(native["jb"], 0)
    assert native["dry_regularization"] == pytest.approx(1.0, abs=1e-12)
    assert native["dry_limit"] == loss
    _close(dry.jacobian, native["dry_jacobian"][0], 2e-6, 2e-5)
    assert dry.regularization == pytest.approx(native["dry_regularization"], rel=3e-4, abs=2e-6)
    assert dry.reference_acceleration == pytest.approx(
        native["dry_reference_acceleration"], rel=3e-4, abs=2e-5
    )
    assert (dry.lower, dry.upper) == (-loss, loss)
    for name in ("bias", "passive", "applied"):
        _close(getattr(receipt.dynamics, name), native[name], 2e-5, 3e-4)
    np.testing.assert_array_equal(solver["damping"], 0)
    np.testing.assert_array_equal(solver["input_velocity"], before.v)
    np.testing.assert_array_equal(solver["output_velocity"], after.v)
    _close(solver["solver_velocity"], after.v, 5e-5, 5e-5)
    actual_f = receipt.dynamics.applied + receipt.dynamics.passive - receipt.dynamics.bias
    _close(solver["applied_force"], actual_f, 2e-7, 2e-5)
    h = float(solver["h_s"])
    assert h == float(np.float32(H))
    g = (pair["jacobian_a"] - pair["jacobian_b"])[:, :, 0]
    n = pair["normals"]
    normal_rows = np.sum(n * g, axis=1)
    parameters = dict(
        mass=float(native["mass"][0, 0]),
        input_velocity=float(before.v[0]),
        applied=float(native["applied"][0] + native["passive"][0] - native["bias"][0]),
        h=h,
        normal_rows=normal_rows,
        stiffness=pair["stiffness"],
        gap=pair["surrogate_gap"],
        dry_jacobian=float(native["dry_jacobian"][0, 0]),
        regularization=native["dry_regularization"],
        reference=native["dry_reference_acceleration"],
        limit=loss,
    )
    oracle = evidence["oracle"] = _scalar_root(**parameters)
    velocity = float(after.v[0])
    _close(np.array([velocity]), np.array([oracle["velocity"]]), 5e-5, 5e-5)
    _close(pair["normal_force"], oracle["normal"], 0.002, 0.002)
    assert dry.force == pytest.approx(oracle["dry"], abs=0.002, rel=0.002)
    # Independently substitute actual accepted v into both force laws as well.
    joint_residual, final_normal, final_dry, acceleration = _scalar_value(velocity, **parameters)
    _close(pair["normal_force"], final_normal, 0.002, 0.002)
    assert dry.force == pytest.approx(final_dry, abs=0.002, rel=0.002)
    relative = g * velocity
    np.testing.assert_allclose(pair["relative_velocity"], relative, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(pair["tangential_force"], 0)
    _close(pair["force_on_a"], pair["normal_force"][:, None] * n, 0.002, 0.002)
    np.testing.assert_array_equal(pair["force_on_b"], -pair["force_on_a"])
    _close(
        pair["torque_on_a_world_origin"], np.cross(pair["points"], pair["force_on_a"]), 1e-5, 0.002
    )
    np.testing.assert_array_equal(
        pair["torque_on_b_world_origin"], -pair["torque_on_a_world_origin"]
    )
    area_q = float(np.sum(g * pair["force_on_a"]))
    total_q = area_q + float(dry.jacobian[0]) * float(dry.force)
    _close(receipt.qfrc_constraint, np.array([total_q]), 2e-5, 2e-5)
    mass = float(solver["mass_matrix"][0, 0])
    physical = (float(actual_f[0]) + total_q) / mass
    _close(h * receipt.qacc, np.array([h * physical]), 5e-5, 5e-5)
    _close(h * receipt.integration_acceleration, np.array([h * physical]), 5e-5, 5e-5)
    np.testing.assert_allclose(
        after.v, before.v + h * receipt.integration_acceleration, rtol=2e-6, atol=2e-7
    )
    gradient = (
        float(dry.jacobian[0]) * physical
        + dry.regularization * dry.force
        - dry.reference_acceleration
    )
    assert dry.gradient == pytest.approx(gradient, abs=2e-5, rel=2e-5)
    if loss == LOSSES[0]:
        assert abs(dry.force) < loss - 0.1
        assert abs(gradient) <= 2e-5 * (1 + abs(physical) + abs(dry.reference_acceleration))
    else:
        assert dry.force == pytest.approx(-loss, abs=2e-6)
        assert gradient > 0.1
    momentum_residual = mass * (velocity - float(before.v[0])) - h * (float(actual_f[0]) + total_q)
    free_velocity = float(before.v[0]) + h * float(actual_f[0]) / mass
    scale = 1 + abs(free_velocity) * np.sqrt(mass)
    assert abs(momentum_residual) / np.sqrt(mass) <= 2e-5 * scale
    assert solver["normalized_residual"] <= solver["tolerance"] == 2e-5
    expected_q = before.q.copy()
    mj.mj_integratePos(spec.compilation.fresh_native_model(), expected_q, after.v, h)
    np.testing.assert_allclose(after.q, expected_q, rtol=0, atol=2e-7)
    assert after.q[0] != before.q[0]
    if before.tick == 0:
        expected_a, expected_f, expected_dry = (
            (10 / 21, 20 / 21, -10 / 21) if loss == 2.0 else (15 / 22, 41 / 44, -0.25)
        )
        _close(np.array([velocity]), np.array([H * expected_a]), 5e-5, 5e-5)
        assert np.sum(pair["normal_force"]) == pytest.approx(expected_f, abs=0.002, rel=0.002)
        assert dry.force == pytest.approx(expected_dry, abs=0.002, rel=0.002)
    evidence.update(
        actual_velocity=velocity,
        actual_acceleration=acceleration,
        actual_area_generalized=area_q,
        actual_native_generalized=float(dry.jacobian[0]) * dry.force,
        actual_total_generalized=total_q,
        actual_joint_residual=joint_residual,
        stored_momentum_residual=momentum_residual,
        actual_gradient=gradient,
        integrated_q=expected_q,
    )


@pytest.mark.cuda
def test_real_area_and_guide_share_velocity_restore_and_late_failure(tmp_path, monkeypatch):
    import mujoco as mj

    def forbidden(*args, **kwargs):
        raise AssertionError("only the production Scene may step or assemble a solve")

    monkeypatch.setattr(mj, "mj_step", forbidden)
    monkeypatch.setattr(TyxAreaFrictionWorkspace, "solve_many", forbidden)
    evidence = dict(
        schema="mujoco-area-friction-guide-test-v1",
        records=[],
        h=H,
        losses=LOSSES,
        status="IN_PROGRESS",
        scope="one native dry row plus current full pressure surface per world",
    )
    try:
        declared = [_declarations(loss, key) for loss, key in zip(LOSSES, (41, 7), strict=True)]
        specs, worlds = tuple(d[0] for d in declared), tuple(d[1] for d in declared)
        assets = {key: value for d in declared for key, value in d[2].items()}
        evidence.update(
            models={
                s.key: dict(
                    source=s.compilation.source,
                    pairs=s.area_pressure_pairs,
                    laws=s.area_friction_laws,
                    contact_capacity=s.contact_capacity,
                    constraint_capacity=s.constraint_capacity,
                )
                for s in specs
            },
            worlds=worlds,
            assets=assets,
        )
        with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda") as batch:
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
            with batch.prepare_device_workspace(
                include_bodies=True, include_area_pressure=True
            ) as ws:
                evidence["initial"] = batch.integration_snapshot()

                def step(label):
                    before = batch.integration_snapshot()
                    configs = {
                        key: batch.area_pressure_configuration_id(key) for key in batch.world_keys
                    }
                    record = dict(label=label, before=before, state_accepted=False)
                    evidence["records"].append(record)
                    result = scene.step(H, expected_revision=batch.checkpoint_revision)
                    after = batch.last_step.states
                    record.update(
                        state_accepted=True,
                        accepted=result,
                        after=after,
                        geometry=scene.last_geometry,
                    )
                    assert result == dict.fromkeys(batch.world_keys, True)
                    frame = ws.refresh(expected_revision=batch.checkpoint_revision)
                    record.update(frame=frame, arrays=ws.readback(frame), audits={})
                    for spec, world, loss in zip(specs, worlds, LOSSES, strict=True):
                        key = world.world
                        assert (
                            record["geometry"][key][0].surface.stamp.configuration_id
                            == configs[key]
                        )
                        audit = record["audits"][key] = {}
                        _audit(spec, loss, before[key], after[key], record["geometry"][key], audit)
                    return record

                saved = step("first")
                capsule = scene.checkpoint()
                restored_capsule = MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(
                    capsule.to_bytes()
                )
                assert restored_capsule.to_bytes() == capsule.to_bytes()
                evidence["checkpoint"] = capsule.to_bytes()
                revision, geometry = batch.checkpoint_revision, scene.last_geometry
                owner = batch._buckets[-1]
                original_download, touched = owner._device.download, []
                with monkeypatch.context() as patch:

                    def fail_after_download():
                        original_download()
                        touched.append(True)
                        raise RuntimeError("late guide candidate download")

                    patch.setattr(owner._device, "download", fail_after_download)
                    with pytest.raises(PhysicsStepFailure) as failed:
                        scene.step(H, expected_revision=revision)
                evidence["late_failure"] = dict(
                    type=type(failed.value).__name__,
                    message=str(failed.value),
                    accepted_worlds=failed.value.accepted_worlds,
                    downloads=len(touched),
                )
                assert touched == [True] and failed.value.accepted_worlds == dict.fromkeys(
                    batch.world_keys, False
                )
                assert batch.checkpoint_revision == revision and scene.last_geometry is geometry
                _same(batch.integration_snapshot(), saved["after"])
                _same(ws.readback(saved["frame"]), saved["arrays"])
                assert scene.checkpoint().to_bytes() == capsule.to_bytes()
                expected = step("retry")
                scene.restore(restored_capsule, expected_revision=batch.checkpoint_revision)
                assert not scene.last_geometry
                _same(batch.integration_snapshot(), saved["after"])
                with pytest.raises(ValueError):
                    ws.readback(expected["frame"])
                restored_frame = ws.refresh(expected_revision=batch.checkpoint_revision)
                _same(ws.readback(restored_frame), saved["arrays"])
                replay = step("replay")
                for key in batch.world_keys:
                    assert (
                        replay["geometry"][key][0].surface.stamp.configuration_id
                        != expected["geometry"][key][0].surface.stamp.configuration_id
                    )
                _same(replay["after"], expected["after"], fresh_stamp=True)
                _same(replay["geometry"], expected["geometry"], fresh_stamp=True)
                _same(replay["arrays"], expected["arrays"])
                evidence.update(
                    final=batch.integration_snapshot(), resources=batch.resource_snapshot()
                )
        evidence["status"] = "PASS"
    except BaseException as error:
        evidence.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        with gzip.open(tmp_path / "guide-live.json.gz", "xt", encoding="utf-8") as stream:
            json.dump(_plain(evidence), stream, allow_nan=False)


@pytest.mark.cuda
def test_separated_slider_keeps_guide_force_and_rejects_forged_row(monkeypatch):
    import phymeo.backends.mujoco_checkpoint as codec

    spec, world, assets = _declarations(0.25, 41)
    world = replace(world, initial_state=MuJoCoInitialState((0.01,), (-0.01,)))
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
        assert scene.step(H, expected_revision=batch.checkpoint_revision) == {41: True}
        after = batch.integration_snapshot()[41]
        receipt = after.last_step
        pairs, dry, _ = _published(receipt)
        assert len(pairs[0]["sample_ids"]) == len(pairs[0]["normal_force"]) == 0
        assert dry.force == pytest.approx(0.25, abs=2e-6)
        assert dry.gradient < 0
        np.testing.assert_allclose(receipt.qfrc_constraint, [0.25], atol=2e-6)
        np.testing.assert_allclose(after.v, [-0.0075], atol=2e-7)
        np.testing.assert_allclose(after.q, [0.01 - H * 0.0075], atol=2e-7)

        owner = batch._buckets[batch.layout[41].bucket]
        leaf = owner.checkpoint()
        revision, original = owner.checkpoint_revision, leaf.to_bytes()
        for mutation in (dict(regularization=0.5), dict(force=-0.25), dict(gradient=1.0)):
            state = dict(leaf._read(leaf.binding))
            item = dict(state["worlds"][0])
            item["receipt"] = replace(item["receipt"], constraints=(replace(dry, **mutation),))
            state["worlds"] = (item,)
            forged = codec._make_checkpoint(leaf.binding, leaf.source_revision, state)
            with monkeypatch.context() as patch:

                def forbidden(*args, **kwargs):
                    pytest.fail("a forged guide row reached replacement device allocation")

                patch.setattr(owner, "_build_device", forbidden)
                with pytest.raises(ValueError):
                    owner.restore(forged, expected_revision=revision)
            assert owner.checkpoint_revision == revision
            assert owner.checkpoint().to_bytes() == original


@pytest.mark.cuda
def test_six_area_samples_and_one_native_row_exceed_six_slots_atomically():
    spec, world, assets = _declarations(0.25, 41)
    spec = replace(spec, constraint_capacity=6)
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
        original = scene.checkpoint().to_bytes()
        revision, geometry = batch.checkpoint_revision, scene.last_geometry
        with pytest.raises(ValueError, match="capacity"):
            scene.step(H, expected_revision=revision)
        assert scene.checkpoint().to_bytes() == original
        assert batch.checkpoint_revision == revision and scene.last_geometry is geometry
