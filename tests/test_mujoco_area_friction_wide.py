"""Seven/eight-DOF live Scene coverage, including the actual final DOF.

The original I3v pressure volume, COM, pose and six velocities are retained.
Only the finite plate gains independent Z or X/Z translations. Native MuJoCo
evaluates current M/J/bias and integrates an observed velocity; it never steps.
No supplied-surface path, HOST friction solve, or old trajectory is replayed.
"""

import gzip
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_mujoco_area_friction import H, _close, _declarations, _equations, _native, _plain
from test_mujoco_area_friction_lifecycle import _same
from test_mujoco_area_friction_live import _published, _surface

from phymeo.backends.mujoco_area_friction import AreaFrictionPairLaw
from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
from phymeo.backends.mujoco_heterogeneous_area_scene import (
    MuJoCoTyxHeterogeneousAreaPressureScene,
    MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
)
from phymeo.backends.mujoco_runtime import MuJoCoInitialState
from phymeo.backends.tyx_area_friction import TyxAreaFrictionWorkspace
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

PLATE_MASS = 0.75
WORLD = 41


def _wide(nv):
    specs, worlds, original_assets = _declarations()
    base, original = specs[0], worlds[0].initial_state
    xml = base.compilation.source.primary_xml.decode("utf-8")
    marker = '<body name="plate" pos="0 0 -.0025">'
    assert xml.count(marker) == 1
    axes = ((0, 0, 1),) if nv == 7 else ((1, 0, 0), (0, 0, 1))
    joints = "".join(
        f'<joint name="plate-{index}" type="slide" axis="{x} {y} {z}" '
        'limited="false" damping="0" frictionloss="0" armature="0"/>'
        for index, (x, y, z) in enumerate(axes)
    )
    # The finite 20 x 20 x 5 mm plate receives its own physical inertia.
    inertia = '<inertial pos="0 0 0" mass=".75" diaginertia=".0000265625 .0000265625 .00005"/>'
    compilation = compile_mujoco_xml(
        xml.replace(marker, marker + joints + inertia), require_asset=False
    )
    asset = original_assets[base.key][0]
    spec = replace(
        base,
        key=f"live-wide-{nv}",
        compilation=compilation,
        area_friction_laws=(AreaFrictionPairLaw(asset.identity, 0.5, 0.001),),
    )
    state = MuJoCoInitialState(
        tuple(original.q) + (0.0,) * (nv - 6), tuple(original.v) + (0.0,) * (nv - 6)
    )
    world = MuJoCoTyxWorldSpec(WORLD, spec.key, state)
    return spec, world, {spec.key: (asset,)}, np.asarray(axes, dtype=float).T


def _audit(spec, before, after, geometry, nv, plate_axes, output):
    import mujoco as mj

    receipt = after.last_step
    rows, solver, problem = _published(receipt)
    assert len(rows) == len(geometry) == 1
    row = rows[0]
    output["surface"] = _surface(row, geometry[0], receipt)
    assert len(row["sample_ids"]) > 0 and np.any(row["normal_force"] > 0)
    assert before.q.shape == after.q.shape == (nv + 1,)
    assert before.v.shape == after.v.shape == (nv,)
    assert receipt.tick == after.tick == before.tick + 1
    assert receipt.world == before.world == after.world == WORLD
    assert receipt.input_time == before.time == receipt.contact_time == receipt.dynamics.time
    assert receipt.output_time == after.time > before.time
    assert receipt.contact_phase == "tyx-euler-input-constraint-evaluation"
    assert receipt.dynamics.phase == "tyx-euler-input-dynamics"
    np.testing.assert_array_equal(receipt.input_q, before.q)
    np.testing.assert_array_equal(receipt.input_v, before.v)
    np.testing.assert_array_equal(solver["output_velocity"], after.v)
    native = output["native"] = _native(
        spec, receipt, (SimpleNamespace(identity=row["identity"], points=row["points"]),)
    )
    mass = np.asarray(problem.mass_matrix, dtype=float)
    linear = np.array([0, 1, 2, *range(6, nv)])
    angular = np.array([3, 4, 5])
    for left, right, absolute in (
        (linear, linear, 2e-6),
        (angular, angular, 2e-9),
        (linear, angular, 2e-8),
        (angular, linear, 2e-8),
    ):
        _close(mass[np.ix_(left, right)], native["mass"][np.ix_(left, right)], absolute, 3e-4)
    assert mass.shape == (nv, nv) and np.max(np.abs(native["mass"][:3, 3:6])) > 1e-4
    np.testing.assert_allclose(
        native["mass"][6:, 6:], PLATE_MASS * np.eye(nv - 6), rtol=0, atol=1e-14
    )
    np.testing.assert_array_equal(native["mass"][:6, 6:], 0)
    dyn = receipt.dynamics
    np.testing.assert_array_equal(mass, dyn.mass_matrix)
    np.testing.assert_array_equal(problem.input_velocity, before.v)
    np.testing.assert_array_equal(problem.damping, native["damping"].astype(np.float32))
    assert problem.h_s == float(np.float32(H))
    for name in ("bias", "passive", "applied"):
        _close(getattr(dyn, name), native[name], 2e-5, 3e-4)
    _close(dyn.jacobian_com, native["com_j"], 2e-6, 2e-5)
    _close(dyn.body_com, native["body_com"], 2e-6, 2e-5)
    _close(dyn.body_position, native["body_position"], 2e-6, 2e-5)
    _close(dyn.body_rotation, native["body_rotation"], 2e-6, 2e-5)
    f = dyn.applied + dyn.passive - dyn.bias + problem.damping * before.v
    bound = (
        8
        * np.finfo(np.float32).eps
        * (
            np.abs(dyn.applied)
            + np.abs(dyn.passive)
            + np.abs(dyn.bias)
            + np.abs(problem.damping * before.v)
        )
        + np.finfo(np.float32).tiny
    )
    assert np.all(np.abs(problem.applied_force - f) <= bound)
    ja, jb = row["jacobian_a"], row["jacobian_b"]
    assert ja.shape == jb.shape == (len(row["sample_ids"]), 3, nv)
    _close(ja, native["ja"], 2e-6, 2e-5)
    _close(jb, native["jb"], 2e-6, 2e-5)
    np.testing.assert_array_equal(ja[:, :, 6:], 0)
    np.testing.assert_array_equal(jb[:, :, :6], 0)
    np.testing.assert_allclose(
        jb[:, :, 6:], np.broadcast_to(plate_axes, jb[:, :, 6:].shape), rtol=0, atol=2e-6
    )
    assert np.all(np.max(np.abs(jb[:, :, 6:]), axis=(0, 1)) > 0)
    # An independent joint on B must contribute to G and Q, rather than being
    # silently dropped after a six-column prefix. Shared ancestor columns would
    # cancel at a common contact point and would not serve as this positive control.
    velocity = after.v.astype(float)
    equations = output["equations"] = _equations(problem, velocity)
    assert equations["residual_norm"] <= 2e-5 * equations["scale"]
    assert solver["normalized_residual"] <= solver["tolerance"] == 2e-5
    _close(velocity, solver["solver_velocity"], 5e-5, 5e-5)
    _close(row["normal_force"], equations["normal"], 0.002, 0.002)
    _close(row["tangential_force"], equations["tangent"], 0.002, 0.002)
    _close(row["force_on_a"], equations["force"], 0.002, 0.002)
    _close(row["torque_on_a_world_origin"], equations["moment"], 1e-5, 0.002)
    np.testing.assert_array_equal(row["force_on_b"], -row["force_on_a"])
    np.testing.assert_array_equal(row["torque_on_b_world_origin"], -row["torque_on_a_world_origin"])
    actual_q = np.asarray(solver["generalized_force"], dtype=float)
    assembled = np.einsum("sij,si->j", ja - jb, row["force_on_a"])
    native_q = np.einsum("sij,si->j", native["ja"] - native["jb"], row["force_on_a"])
    _close(actual_q, assembled, 2e-5, 2e-5)
    _close(actual_q[linear], native_q[linear], 0.002, 0.002)
    _close(actual_q[angular], native_q[angular], 1e-5, 0.002)
    np.testing.assert_array_equal(receipt.qfrc_constraint, actual_q)
    plate_force = np.sum(row["force_on_b"], axis=0)
    _close(actual_q[6:], plate_axes.T @ plate_force, 2e-5, 2e-5)
    assert np.all(np.abs(actual_q[6:]) > 1e-5)
    assert np.all(np.abs(velocity[6:] - equations["free"][6:]) > 1e-7)
    assert np.all(after.q[7:] != before.q[7:])
    for sample_index, sample in enumerate(problem.samples):
        g = sample.jacobian_a - sample.jacobian_b
        relative = g @ velocity
        _close(row["relative_velocity"][sample_index], relative, 2e-5, 2e-5)
        normal = sample.normal_world
        tangent = relative - normal * (normal @ relative) / (normal @ normal)
        assert row["tangential_force"][sample_index] @ tangent <= 1e-14
    rhs = dyn.applied + dyn.passive - dyn.bias + actual_q
    physical = np.linalg.solve(mass, rhs)
    integration = np.linalg.solve(equations["response"], rhs)
    h = problem.h_s
    _close(h * receipt.qacc, h * physical, 5e-5, 5e-5)
    _close(h * receipt.integration_acceleration, h * integration, 5e-5, 5e-5)
    np.testing.assert_allclose(
        velocity, before.v + h * receipt.integration_acceleration, rtol=2e-6, atol=2e-7
    )
    residual = equations["response"] @ velocity - equations["rhs"] - h * actual_q
    stored_norm = np.sqrt(
        max(0.0, float(residual @ np.linalg.solve(equations["response"], residual)))
    )
    assert stored_norm <= 2e-5 * equations["scale"]
    model = spec.compilation.fresh_native_model()
    assert (model.nq, model.nv) == (nv + 1, nv)
    for index in range(nv - 6):
        joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"plate-{index}")
        assert model.jnt_qposadr[joint] == 7 + index and model.jnt_dofadr[joint] == 6 + index
    expected_q = before.q.copy()
    mj.mj_integratePos(model, expected_q, velocity, h)
    np.testing.assert_allclose(after.q[:3], expected_q[:3], rtol=0, atol=2e-7)
    sign = 1 if after.q[3:7] @ expected_q[3:7] >= 0 else -1
    np.testing.assert_allclose(sign * after.q[3:7], expected_q[3:7], rtol=0, atol=3e-7)
    np.testing.assert_allclose(after.q[7:], expected_q[7:], rtol=0, atol=2e-7)
    assert abs(float(after.q[3:7] @ after.q[3:7]) - 1) <= 2e-6
    output.update(
        actual_generalized=actual_q,
        native_generalized=native_q,
        stored_residual=residual,
        stored_norm=stored_norm,
        integrated_q=expected_q,
        physical_acceleration=physical,
        integration_acceleration=integration,
        strict_host_gate=equations["residual_norm"] <= 1e-10 * equations["scale"],
    )


@pytest.mark.cuda
@pytest.mark.parametrize("nv", (7, 8))
def test_current_wide_scene_and_checkpoint_replay(nv, tmp_path, monkeypatch):
    import mujoco as mj

    def forbidden(*args, **kwargs):
        raise AssertionError("wide Scene must not CPU-step or use HOST-packed solve_many")

    monkeypatch.setattr(mj, "mj_step", forbidden)
    monkeypatch.setattr(TyxAreaFrictionWorkspace, "solve_many", forbidden)
    evidence = dict(
        schema="mujoco-area-friction-wide-test-v1",
        nv=nv,
        h=H,
        records=[],
        status="IN_PROGRESS",
        scope="current full geometry; independent FREE A and guided moving B",
    )
    try:
        spec, world, assets, plate_axes = _wide(nv)
        evidence.update(
            model=dict(
                key=spec.key,
                source=spec.compilation.source,
                area_pressure_pairs=spec.area_pressure_pairs,
                area_friction_laws=spec.area_friction_laws,
                contact_capacity=spec.contact_capacity,
                constraint_capacity=spec.constraint_capacity,
            ),
            world=world,
            assets=assets,
            plate_axes=plate_axes,
        )
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
            with batch.prepare_device_workspace(
                include_bodies=True, include_area_pressure=True
            ) as ws:
                evidence["initial"] = batch.integration_snapshot()

                def step(label):
                    before = batch.integration_snapshot()[WORLD]
                    config = batch.area_pressure_configuration_id(WORLD)
                    record = dict(label=label, before=before, state_accepted=False)
                    evidence["records"].append(record)
                    accepted = scene.step(H, expected_revision=batch.checkpoint_revision)
                    after = batch.last_step.states[WORLD]
                    record.update(
                        state_accepted=True,
                        accepted=accepted,
                        after=after,
                        geometry=scene.last_geometry[WORLD],
                    )
                    assert accepted == {WORLD: True}
                    assert record["geometry"][0].surface.stamp.configuration_id == config
                    frame = ws.refresh(expected_revision=batch.checkpoint_revision)
                    arrays = ws.readback(frame)
                    record.update(frame=frame, arrays=arrays, audit={})
                    np.testing.assert_array_equal(arrays["q"][0, : nv + 1], after.q)
                    np.testing.assert_array_equal(arrays["v"][0, :nv], after.v)
                    np.testing.assert_array_equal(arrays["entry_q"][0, : nv + 1], before.q)
                    np.testing.assert_array_equal(arrays["entry_v"][0, :nv], before.v)
                    assert arrays["area_meta"][0, 0] == 2 and arrays["area_meta"][0, 1] > 0
                    _audit(spec, before, after, record["geometry"], nv, plate_axes, record["audit"])
                    return record

                saved = step("initial")
                checkpoint = scene.checkpoint()
                capsule = MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(
                    checkpoint.to_bytes()
                )
                assert capsule.to_bytes() == checkpoint.to_bytes()
                evidence["checkpoint"] = capsule.to_bytes()
                expected = step("expected")
                scene.restore(capsule, expected_revision=batch.checkpoint_revision)
                assert not scene.last_geometry
                restored = batch.integration_snapshot()[WORLD]
                evidence["restored"] = restored
                _same(restored, saved["after"])
                with pytest.raises(ValueError):
                    ws.readback(expected["frame"])
                restored_frame = ws.refresh(expected_revision=batch.checkpoint_revision)
                _same(ws.readback(restored_frame), saved["arrays"])
                replay = step("replay")
                old_stamp = expected["geometry"][0].surface.stamp
                new_stamp = replay["geometry"][0].surface.stamp
                assert new_stamp.configuration_id != old_stamp.configuration_id
                _same(replay["after"], expected["after"], fresh_stamp=True)
                _same(replay["geometry"], expected["geometry"], fresh_stamp=True)
                _same(replay["arrays"], expected["arrays"])
                assert replay["after"].tick == 2
                evidence.update(
                    final=batch.integration_snapshot(), resources=batch.resource_snapshot()
                )
        evidence["status"] = "PASS"
    except BaseException as error:
        evidence.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        with gzip.open(tmp_path / f"wide-{nv}.json.gz", "xt", encoding="utf-8") as stream:
            json.dump(_plain(evidence), stream, allow_nan=False)


@pytest.mark.cuda
def test_six_seven_eight_dof_share_a_scene(tmp_path):
    specs, worlds, assets = _declarations()
    base = specs[0]
    law = AreaFrictionPairLaw(assets[base.key][0].identity, 0.5, 0.001)
    models = [replace(base, area_friction_laws=(law,))]
    lanes = [worlds[0]]
    geometry = {base.key: assets[base.key]}
    for nv in (7, 8):
        spec, world, extra_assets, _ = _wide(nv)
        models.append(spec)
        lanes.append(replace(world, world=100 + nv))
        geometry.update(extra_assets)
    # World order deliberately differs from model declaration order.
    lanes = (lanes[2], lanes[0], lanes[1])
    with MuJoCoTyxHeterogeneousBatch(tuple(models), lanes, mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, geometry)
        with batch.prepare_device_workspace(include_area_pressure=True) as ws:
            before = batch.integration_snapshot()
            assert scene.step(H, expected_revision=batch.checkpoint_revision) == dict.fromkeys(
                batch.world_keys, True
            )
            states = batch.integration_snapshot()
            frame = ws.refresh(expected_revision=batch.checkpoint_revision)
            arrays = ws.readback(frame)
            with gzip.open(tmp_path / "mixed-width.json.gz", "xt", encoding="utf-8") as stream:
                json.dump(
                    _plain(
                        dict(
                            before=before,
                            after=states,
                            frame=frame,
                            arrays=arrays,
                            geometry=scene.last_geometry,
                        )
                    ),
                    stream,
                    allow_nan=False,
                )
            for i, key in enumerate(batch.world_keys):
                state = states[key]
                nv = len(before[key].v)
                assert nv == {108: 8, 7: 6, 107: 7}[key]
                solver = state.last_step.raw_contact_fields["area_friction_solver"]
                assert solver["raw_diagnostics"]["trace"].shape[1] == (15 if nv == 6 else 19)
                np.testing.assert_array_equal(arrays["entry_v"][i, :nv], before[key].v)
                np.testing.assert_array_equal(arrays["v"][i, :nv], state.v)
                np.testing.assert_array_equal(arrays["v"][i, nv:], 0)
                np.testing.assert_array_equal(arrays["q"][i, : nv + 1], state.q)
                np.testing.assert_array_equal(arrays["q"][i, nv + 1 :], 0)
                assert arrays["area_meta"][i, 2] > 0
                assert np.all(np.abs(solver["generalized_force"][6:]) > 1e-5)
