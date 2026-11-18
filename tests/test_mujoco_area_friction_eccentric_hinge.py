"""Serial carriage/hinge: eccentric pressure, native joint loads and link motion."""

import json
import xml.etree.ElementTree as ET
from dataclasses import replace

import numpy as np
import pytest
from test_mujoco_area_friction_guide import H
from test_mujoco_area_friction_guide import _declarations as _slider
from test_mujoco_area_friction_live import _surface

from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.backends.mujoco_runtime import MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

X0, INERTIA = 0.25, 0.002


def _model(key, arm, angular_stop):
    base, _, assets = _slider(0.01, key)
    tree = ET.fromstring(base.compilation.source.primary_xml)
    worldbody = tree.find("worldbody")
    pad = worldbody.find("body[@name='pressure']")
    worldbody.remove(pad)
    carriage = ET.SubElement(worldbody, "body", name="carriage", pos=f"{X0} 0 .011")
    slide = pad.find("joint")
    pad.remove(slide)
    carriage.append(slide)
    ET.SubElement(carriage, "inertial", pos="0 0 0", mass="1", diaginertia=".002 .002 .002")
    carriage.append(pad)
    pad.set("pos", "0 0 0")
    pad.find("inertial").set("diaginertia", ".002 .002 .002")
    hinge = ET.SubElement(
        pad,
        "joint",
        type="hinge",
        axis="0 1 0",
        name="pivot",
        pos=f"{arm} 0 0",
        limited="false",
        frictionloss="0",
    )
    worldbody.find("body[@name='plate']").set("pos", f"{X0} 0 -.0025")
    theta = omega = 0.0
    mu = 0.0
    if angular_stop:
        side = np.sign(arm)
        slide.set("frictionloss", "0")
        hinge.attrib.update(
            limited="true",
            range="-.5 0" if arm > 0 else "0 .5",
            solreflimit=".02 1",
            solimplimit=".5 .5 .001 .5 2",
            frictionloss=".001",
            solimpfriction=".5 .5 .001 .5 2",
        )
        theta, omega, mu = side * 0.002, side * 0.05, 0.4
    spec = replace(
        base,
        key=f"eccentric-{key}",
        compilation=compile_mujoco_xml(ET.tostring(tree, encoding="unicode"), require_asset=False),
        area_friction_laws=(replace(base.area_friction_laws[0], mu=mu),),
    )
    world = MuJoCoTyxWorldSpec(key, spec.key, MuJoCoInitialState((-0.002, theta), (0.0, omega)))
    return spec, world, assets[base.key]


def _flat_reference(arm):
    # Uniform square pressure patch, width 2 cm. Integrate [1, arm-x]^T[1, arm-x]
    # exactly about its centre, including the nonzero second moment of area.
    mass = np.array(((2.0, arm), (arm, INERTIA + arm**2)))
    stiffness = 1000 * np.array(((1.0, arm), (arm, arm**2 + 0.01**2 / 3)))
    unloaded = np.array((1.0, arm))
    acc = np.linalg.solve(mass + H**2 * stiffness, unloaded - (0.01, 0.0))
    # The slider dry row saturates at -0.01 N; all normal samples stay loaded.
    assert acc[0] > 0.01 * np.linalg.inv(mass)[0, 0]
    assert H**2 * (abs(acc[0]) + (abs(arm) + 0.01) * abs(acc[1])) < 0.001
    return acc, unloaded - H**2 * stiffness @ acc


def _audit(model, before, after, intersection, device, index, arm, angular_stop):
    import mujoco as mj

    receipt = after.last_step
    (pair,) = receipt.raw_contact_fields["area_friction"]
    _surface(pair, intersection, receipt)
    points, forces = pair["points"].astype(float), pair["force_on_a"].astype(float)
    assert len(points) > 0 and np.sum(pair["normal_force"]) > 0
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:] = before.q, before.v
    mj.mj_forward(model, data)
    pivot = model.joint("pivot").id
    pad = model.body("pressure").id
    mass = np.empty((2, 2))
    mj.mj_fullM(model, mass, data.qM)
    np.testing.assert_allclose(mass[0, 1], arm * np.cos(before.q[1]), atol=1e-12)
    solver = receipt.raw_contact_fields["area_friction_solver"]
    np.testing.assert_allclose(solver["mass_matrix"], mass, rtol=2e-5, atol=2e-7)
    axis, anchor = data.xaxis[pivot], data.xanchor[pivot]
    np.testing.assert_allclose(anchor, (X0 + arm, 0.0, 0.011 + before.q[0]), atol=1e-12)
    # Each material point moves with the carriage plus rotation around the hinge.
    arms = points - anchor
    rot = np.cross(axis, arms)
    jp = np.zeros((len(points), 3, 2))
    jp[:, 2, 0], jp[:, :, 1] = 1.0, rot
    np.testing.assert_allclose(pair["jacobian_a"], jp, atol=2e-7)
    np.testing.assert_array_equal(pair["jacobian_b"], 0)
    normals = pair["normals"].astype(float)
    vel = jp @ after.v
    vn = np.einsum("ij,ij->i", normals, vel)
    vt = vel - vn[:, None] * normals
    fn = np.maximum(0.0, -pair["stiffness"] * (pair["surrogate_gap"] + H * vn))
    mu = 0.4 if angular_stop else 0.0
    tangent = -mu * fn[:, None] * vt / np.sqrt(np.sum(vt * vt, axis=1) + 0.001**2)[:, None]
    np.testing.assert_allclose(forces, fn[:, None] * normals + tangent, rtol=0.002, atol=2e-5)
    np.testing.assert_array_equal(pair["force_on_b"], -pair["force_on_a"])
    world_moment = np.cross(points, forces).sum(axis=0)
    np.testing.assert_allclose(
        pair["torque_on_a_world_origin"].sum(axis=0), world_moment, atol=2e-7
    )
    np.testing.assert_array_equal(
        pair["torque_on_b_world_origin"], -pair["torque_on_a_world_origin"]
    )
    force = forces.sum(axis=0)
    torque = float(axis @ np.cross(arms, forces).sum(axis=0))
    np.testing.assert_allclose(torque, axis @ (world_moment - np.cross(anchor, force)), atol=1e-12)
    area = np.array((force[2], torque))
    total = area.copy()
    assert len(receipt.constraints) == data.nefc
    for i, row in enumerate(receipt.constraints):
        jac = data.efc_J.reshape(-1, 2)[i]
        assert (row.constraint_type, row.source_id) == (data.efc_type[i], data.efc_id[i])
        np.testing.assert_array_equal(row.jacobian, jac)
        # Reconstruct each native reaction from independent source coefficients.
        reaction = (data.efc_aref[i] - jac @ receipt.qacc) / data.efc_R[i]
        if row.constraint_type == 1:
            loss = data.efc_frictionloss[i]
            reaction = np.clip(reaction, -loss, loss)
        else:
            assert row.constraint_type == 3
            reaction = max(0.0, reaction)
        np.testing.assert_allclose(row.force, reaction, atol=2e-6, rtol=0.001)
        total += jac * row.force
    np.testing.assert_allclose(receipt.qfrc_constraint, total, atol=2e-6, rtol=2e-5)
    acceleration = np.linalg.solve(mass, total + data.qfrc_passive - data.qfrc_bias)
    np.testing.assert_allclose(receipt.qacc, acceleration, atol=3e-5, rtol=3e-4)
    np.testing.assert_allclose(after.v, before.v + H * acceleration, atol=5e-6)
    np.testing.assert_allclose(after.q, before.q + H * after.v, atol=2e-7)
    np.testing.assert_allclose(
        device["body_angular_velocity"][index][pad], axis * after.v[1], atol=2e-6
    )
    theta = after.q[1]
    expected_pad = (X0 + arm * (1 - np.cos(theta)), 0.0, 0.011 + after.q[0] + arm * np.sin(theta))
    np.testing.assert_allclose(device["body_position"][index][pad], expected_pad, atol=2e-7)
    if before.tick == 0 and not angular_stop:
        acc, reference_area = _flat_reference(arm)
        np.testing.assert_allclose(receipt.qacc, acc, atol=8e-5, rtol=3e-4)
        np.testing.assert_allclose(area, reference_area, atol=3e-6, rtol=3e-4)
    if before.tick == 0 and angular_stop:
        dry, limit = receipt.constraints
        assert dry.dof == limit.dof == 1 and limit.side_or_edge == np.sign(arm)
        assert limit.force > 1e-4 and abs(dry.force) > 1e-5
        assert np.linalg.norm(tangent) > 1e-5 and np.sum(vt * tangent) < 0
    return dict(
        force_N=float(force[2]),
        torque_Nm=torque,
        omega_rad_s=float(after.v[1]),
        angle_rad=float(theta),
        native=[float(row.force) for row in receipt.constraints],
        world_origin_My_Nm=float(world_moment[1]),
    )


@pytest.mark.cuda
@pytest.mark.parametrize("angular_stop", (False, True), ids=("slider-drag", "hinge-drag-and-stop"))
def test_serial_hinge_torque_and_link_motion(angular_stop, tmp_path, monkeypatch):
    import mujoco as mj

    def forbid_cpu_step(*args, **kwargs):
        pytest.fail("the CUDA scene must own the state update")

    monkeypatch.setattr(mj, "mj_step", forbid_cpu_step)
    cases = ((41, 0.03), (7, -0.03)) if angular_stop else ((41, 0.03), (7, -0.03), (305, 0.0))
    cases_by_key = dict(cases)
    declarations = [_model(key, arm, angular_stop) for key, arm in cases]
    specs, worlds, entries = zip(*declarations, strict=True)
    assets = {spec.key: asset for spec, asset in zip(specs, entries, strict=True)}
    models = {
        world.world: spec.compilation.fresh_native_model()
        for spec, world in zip(specs, worlds, strict=True)
    }
    history = []
    with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
        with batch.prepare_device_workspace(include_bodies=True, include_area_pressure=True) as ws:
            for _ in range(3):
                before = batch.integration_snapshot()
                assert scene.step(H, expected_revision=batch.checkpoint_revision) == dict.fromkeys(
                    batch.world_keys, True
                )
                frame = ws.refresh(expected_revision=batch.checkpoint_revision)
                device = ws.readback(frame)
                record = {}
                for index, key in enumerate(batch.world_keys):
                    caps = batch._buckets[batch.layout[key].bucket].capabilities
                    assert caps["domain"] == "area-friction-cuda-scalar-hinge-v1"
                    actual = batch.last_step.states[key]
                    record[key] = _audit(
                        models[key],
                        before[key],
                        actual,
                        scene.last_geometry[key][0],
                        device,
                        index,
                        cases_by_key[key],
                        angular_stop,
                    )
                history.append(record)
    first = history[0]
    assert first[41]["torque_Nm"] > 0.01 and first[7]["torque_Nm"] < -0.01
    if not angular_stop:
        assert first[41]["omega_rad_s"] > 0.01 and first[7]["omega_rad_s"] < -0.01
        assert abs(first[305]["torque_Nm"]) < 1e-7 and abs(first[305]["omega_rad_s"]) < 1e-6
        assert first[41]["torque_Nm"] == pytest.approx(-first[7]["torque_Nm"], abs=2e-6)
        assert first[41]["force_N"] == pytest.approx(first[7]["force_N"], abs=2e-5)
    (tmp_path / "hinge-trajectory.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


@pytest.mark.cuda
def test_loaded_angular_rows_survive_checkpoint_and_replay():
    from test_mujoco_area_friction_lifecycle import _same

    key = 41
    spec, world, asset = _model(key, 0.03, True)
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: asset})

        def advance():
            assert scene.step(H, expected_revision=batch.checkpoint_revision) == {key: True}
            return batch.last_step.states[key]

        loaded = advance()
        dry, stop = loaded.last_step.constraints
        assert dry.dof == stop.dof == 1 and abs(dry.force) > 0 and stop.force > 0
        capsule = scene.checkpoint()
        expected = advance()
        scene.restore(
            type(capsule).from_bytes(capsule.to_bytes()),
            expected_revision=batch.checkpoint_revision,
        )
        _same(batch.integration_snapshot()[key], loaded)
        actual = advance()
        _same(actual, expected, fresh_stamp=True)


@pytest.mark.cuda
@pytest.mark.parametrize("hinges", (1, 2))
def test_all_rotary_scalar_layout_transmits_area_moments(hinges):
    import mujoco as mj

    original, _, assets = _model(41, 0.03, False)
    tree = ET.fromstring(original.compilation.source.primary_xml)
    carriage = tree.find(".//body[@name='carriage']")
    carriage.set("pos", f"{X0} 0 .009")
    guide = carriage.find("joint")
    if hinges == 1:
        carriage.remove(guide)
        arms = np.array([0.03])
    else:
        guide.attrib.update(type="hinge", axis="0 1 0", pos="-.02 0 0", frictionloss="0")
        arms = np.array([-0.02, 0.03])
    tree.find(".//joint[@name='pivot']").set("frictionloss", ".001")
    spec = replace(
        original,
        compilation=compile_mujoco_xml(ET.tostring(tree, encoding="unicode"), require_asset=False),
    )
    world = MuJoCoTyxWorldSpec(41, spec.key, MuJoCoInitialState((0.0,) * hinges, (0.0,) * hinges))
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: assets})
        scene.step(H, expected_revision=batch.checkpoint_revision)
        state = batch.last_step.states[41]
        receipt = state.last_step
        (pair,) = receipt.raw_contact_fields["area_friction"]
        (dry,) = receipt.constraints
        assert dry.dof == hinges - 1 and dry.force < 0
        model = spec.compilation.fresh_native_model()
        data = mj.MjData(model)
        mj.mj_forward(model, data)
        M = np.empty((hinges, hinges))
        mj.mj_fullM(model, M, data.qM)
        # All axes are parallel to +y. A unit vertical force at local x acts on
        # every ancestor hinge with signed arm (axis_x - x), measured in metres.
        K = 1000 * (np.outer(arms, arms) + 0.01**2 / 3)
        jdry = np.eye(hinges)[-1]
        acc = np.linalg.solve(M + H**2 * K, arms - 0.001 * jdry)
        assert acc[-1] > 0.001 * data.efc_R[0]
        assert dry.force == pytest.approx(-0.001, abs=1e-7)
        np.testing.assert_allclose(receipt.qacc, acc, atol=1e-4, rtol=3e-4)
        np.testing.assert_allclose(state.v, H * acc, atol=3e-6)
        levers = (X0 + arms)[None, :] - pair["points"][:, 0, None]
        np.testing.assert_allclose(
            levers.T @ pair["normal_force"], arms - H**2 * K @ acc, atol=2e-6
        )
