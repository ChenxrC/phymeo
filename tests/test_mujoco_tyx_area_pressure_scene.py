"""Continuous actual-state I1 -> I2a tests; fixed in the I2b plan before running."""

import json

import numpy as np
import pytest
import test_mujoco_tyx_normal_contacts as normal

from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime
from phymeo.backends.mujoco_area_pressure_scene import (
    BodyAreaPressureAsset,
    MuJoCoTyxAreaPressureScene,
)
from phymeo.backends.mujoco_runtime import MuJoCoInitialState
from phymeo.geometry.area_pressure import AreaPressureIdentity
from phymeo.geometry.area_pressure_intersection import RigidPressureMesh, TetraPressureMesh

MODES = normal.MODES
PAIR = ("pressure", "plate")
MASS = np.array([0.5, 1.25])
AREA = 0.00045
K = 9000.0
BASE = np.array([[-0.12, -0.12, 0], [0.12, -0.12, 0], [0, 0.12, 0], [0, 0, 0.08]])
RIGID = np.array([[-0.015, -0.045, 0], [0.015, -0.045, 0], [0, -0.015, 0]])
CELLS = ((0, 1, 2, 4), (0, 2, 3, 4), (0, 3, 1, 4), (1, 3, 2, 4))


def asset(rotating=False):
    vertices = np.vstack((BASE, BASE.mean(axis=0)))
    if rotating:
        vertices = vertices + [0.06, 0, -0.025]
    rigid = RIGID + ([0.06, 0, 0] if rotating else [0, 0, 0])
    return BodyAreaPressureAsset(
        AreaPressureIdentity(*PAIR, "linear-pressure-full", "finite-rigid-face"),
        TetraPressureMesh(
            ("a", "b", "c", "d", "center"),
            vertices,
            CELLS,
            (0, 0, 0, 0, 400000),
            ("bottom", "side-a", "side-b", "slant"),
        ),
        RigidPressureMesh(("r0", "r1", "r2"), rigid, ((0, 1, 2),), ("face",)),
    )


def runtime_for(mode, h, rotating=False):
    mj = normal._deps()
    kind, axis = ("hinge", "0 1 0") if rotating else ("slide", "0 0 1")
    height = 0.027 if rotating else 0.002
    xml = f'''<mujoco model="current-area-surface">
    <compiler angle="radian" fusestatic="false"/>
    <option timestep="{h}" gravity="0 0 0" integrator="Euler"
      iterations="128" tolerance="1e-10" cone="pyramidal" jacobian="dense"/>
    <worldbody>
      <body name="pressure" pos="0 0 {height}">
        <joint name="a" type="{kind}" axis="{axis}" damping="0" armature="0"/>
        <inertial pos="0 0 0" mass=".5" diaginertia=".02 .02 .02"/>
        <geom type="sphere" size=".01" contype="0" conaffinity="0"/>
      </body>
      <body name="plate">
        <joint name="b" type="slide" axis="0 0 1" damping="0" armature="0"/>
        <inertial pos="0 0 0" mass="1.25" diaginertia=".03 .03 .03"/>
        <geom type="sphere" size=".01" contype="0" conaffinity="0"/>
      </body>
    </worldbody></mujoco>'''
    model = mj.MjModel.from_xml_string(xml)
    runtime = MuJoCoTyxRuntime(
        normal._compile(xml),
        world_keys=(13,),
        mode=mode,
        initial_states={13: MuJoCoInitialState(model.qpos0, [2 if rotating else -0.2, 0])},
        area_pressure_pairs=(PAIR,),
        constraint_capacity=64,
    )
    return runtime, MuJoCoTyxAreaPressureScene(runtime, (asset(rotating),))


def _save(path, kind, mode, h, records, summary):
    payload = dict(
        schema="current-area-trajectory-v1",
        case=kind,
        mode=mode,
        h=h,
        pressure_domain="full",
        records=records,
        summary=summary,
    )
    path.write_text(json.dumps(payload, allow_nan=False, indent=2), encoding="utf-8")


def _record(before, after, geometry, extra):
    receipt = after.last_step
    area = receipt.raw_contact_fields["area_pressure"][0]
    result = dict(
        input_tick=before.tick,
        output_tick=after.tick,
        input_time=before.time,
        output_time=after.time,
        input_q=before.q.tolist(),
        input_v=before.v.tolist(),
        output_q=after.q.tolist(),
        output_v=after.v.tolist(),
        input_configuration=geometry.surface.stamp.configuration_id,
        pressure_area=float(np.sum(area["dA"])),
        points=area["points"].tolist(),
        normals=area["normals"].tolist(),
        pressure_source_ids=area["source_ids"],
        sample_ids=area["sample_ids"],
        normal_forces=area["normal_force"].tolist(),
        forces_a=area["force_on_a"].tolist(),
        torques_a=area["torque_on_a_world_origin"].tolist(),
        generalized_force=receipt.qfrc_constraint.tolist(),
        integration_acceleration=receipt.integration_acceleration.tolist(),
        polygons=geometry.polygon_count,
        candidate_pairs=geometry.candidate_pairs,
    )
    result.update(extra)
    return result


def _depth(q):
    # Independent scalar geometry from the two accepted joint coordinates.
    # Body offsets are stored FP32 by this runtime, as are the FK additions.
    a = float(np.float32(np.float32(0.002) + np.float32(q[0])))
    return float(q[1]) - a


@pytest.mark.parametrize("mode", MODES)
def test_two_moving_bodies_enter_compress_unload_on_one_continuous_trajectory(mode, tmp_path):
    summaries = []
    for h in (0.0005, 0.00025):
        runtime, scene = runtime_for(mode, h)
        records, energies, loads, depths, momenta = [], [], [], [], []
        with runtime:
            initial = runtime.integration_snapshot()[13]
            reduced = 1 / np.sum(1 / MASS)
            energy0 = 0.5 * reduced * (initial.v[0] - initial.v[1]) ** 2
            momentum0 = float(MASS @ initial.v)
            for _ in range(round(0.08 / h)):
                before = runtime.integration_snapshot()[13]
                depth = _depth(before.q)
                relative = before.v[0] - before.v[1]
                force = (
                    max(0.0, K * (depth - h * relative) / (1 + h * h * K / reduced))
                    if depth >= 0
                    else 0.0
                )
                expected_v = before.v + h * force * np.array([1 / MASS[0], -1 / MASS[1]])
                scene.step(h, expected_revision=runtime.checkpoint_revision)
                after = runtime.integration_snapshot()[13]
                geometry = scene.last_geometry[13][0]
                row = after.last_step.raw_contact_fields["area_pressure"][0]
                actual_force = float(np.sum(row["normal_force"]))
                if len(row["normal_force"]):
                    np.testing.assert_allclose(
                        row["normal_force"], actual_force / 3, rtol=2e-5, atol=1e-5
                    )
                    np.testing.assert_allclose(
                        np.sum(row["torque_on_a_world_origin"], axis=0),
                        [-0.035 * actual_force, 0, 0],
                        rtol=2e-5,
                        atol=1e-6,
                    )
                assert abs(actual_force - force) <= 0.001 + 0.001 * abs(force)
                np.testing.assert_allclose(after.v, expected_v, rtol=0, atol=3e-5)
                assert geometry.surface.stamp.step_index == before.tick
                assert geometry.surface.stamp.sample_time_s == before.time
                expected_area = AREA if depth >= 0 else 0.0
                assert np.sum(row["dA"]) == pytest.approx(expected_area, rel=1e-10, abs=1e-12)
                if row["points"].size:
                    np.testing.assert_allclose(row["points"][:, 2], before.q[1], rtol=0, atol=2e-8)
                    assert {source[0] for source in row["source_ids"]} == {"bottom"}
                output_depth = _depth(after.q)
                assert max(depth, output_depth) < 0.004
                energy = (
                    0.5 * reduced * (after.v[0] - after.v[1]) ** 2
                    + 0.5 * K * max(output_depth, 0) ** 2
                )
                momentum = float(MASS @ after.v)
                energies.append(energy)
                momenta.append(momentum)
                loads.append(actual_force)
                depths.append(output_depth)
                records.append(
                    _record(
                        before,
                        after,
                        geometry,
                        dict(
                            input_depth=depth,
                            output_depth=output_depth,
                            expected_force=force,
                            relative_energy=energy,
                            momentum=momentum,
                        ),
                    )
                )
            active = [i for i, force in enumerate(loads) if force > 1e-4]
            summary = dict(
                peak_force=max(loads),
                peak_depth=max(depths),
                first_contact=records[active[0]]["input_time"] if active else None,
                last_contact=records[active[-1]]["output_time"] if active else None,
                final_relative_speed=float(after.v[0] - after.v[1]),
                relative_energy_initial=energy0,
                relative_energy_final=energies[-1],
                energy_gain_fraction=max(0, max(energies) / energy0 - 1),
                energy_loss_fraction=1 - energies[-1] / energy0,
                max_momentum_error=max(abs(p - momentum0) for p in momenta),
            )
            _save(tmp_path / f"sliders-{mode}-{h}.json", "sliders", mode, h, records, summary)
            assert active and active[0] > 0 and active[-1] < len(loads) - 1
            assert all(force == 0 for force in loads[: active[0]])
            assert all(force == 0 for force in loads[-10:])
            assert after.v[0] - after.v[1] > 0.1
            assert summary["energy_gain_fraction"] <= 0.02
            assert summary["energy_loss_fraction"] <= (0.30 if h == 0.0005 else 0.18)
            assert summary["max_momentum_error"] <= 1e-5
            summaries.append(summary)
    a, b = summaries
    for field in ("peak_force", "peak_depth"):
        assert abs(a[field] - b[field]) / b[field] <= 0.08
    for field in ("first_contact", "last_contact"):
        assert abs(a[field] - b[field]) <= 0.001


@pytest.mark.parametrize("mode", MODES)
def test_eccentric_hinge_refreshes_rotation_and_contact_area(mode, tmp_path):
    h = 0.0005
    runtime, scene = runtime_for(mode, h, rotating=True)
    records, loads, areas, velocities = [], [], [], []
    with runtime:
        for _ in range(round(0.08 / h)):
            before = runtime.integration_snapshot()[13]
            scene.step(h, expected_revision=runtime.checkpoint_revision)
            after = runtime.integration_snapshot()[13]
            geometry = scene.last_geometry[13][0]
            row = after.last_step.raw_contact_fields["area_pressure"][0]
            points = row["points"]
            loads.append(float(np.sum(row["normal_force"])))
            areas.append(float(np.sum(row["dA"])))
            velocities.append(float(after.v[0]))
            if len(points):
                # A y hinge about (0,0,.027) and B z slider, all normals +z.
                j = np.column_stack((-points[:, 0], -np.ones(len(points))))
                np.testing.assert_allclose(row["normal_jacobian"], j, rtol=2e-5, atol=2e-6)
                force = row["normal_force"]
                np.testing.assert_allclose(
                    after.last_step.qfrc_constraint, j.T @ force, rtol=5e-4, atol=5e-5
                )
                np.testing.assert_allclose(points[:, 2], before.q[1], rtol=0, atol=2e-8)
                # Independent inverse rigid transform verifies the pressure:
                # bottom-cell p=g*(local_z+.025), with stored TYX pose.
                angle = before.q[0]
                origin = np.array([0, 0, float(np.float32(0.027))])
                local_z = np.sin(angle) * (points[:, 0] - origin[0]) + np.cos(angle) * (
                    points[:, 2] - origin[2]
                )
                pressure_depth = local_z + 0.025
                np.testing.assert_allclose(
                    -row["surrogate_gap"], pressure_depth / np.cos(angle), rtol=3e-4, atol=2e-8
                )
                expected_force = np.maximum(
                    0, row["stiffness"] * (-row["surrogate_gap"] - h * (j @ after.v))
                )
                np.testing.assert_allclose(force, expected_force, rtol=5e-4, atol=5e-4)
            assert abs(after.q[0]) < 0.2
            records.append(_record(before, after, geometry, {}))
        summary = dict(
            peak_force=max(loads),
            maximum_area=max(areas),
            initial_speed=2.0,
            final_speed=velocities[-1],
            minimum_speed=min(velocities),
            changing_area=bool(any(0 < a < AREA * (1 - 1e-6) for a in areas)),
        )
        _save(tmp_path / f"hinge-{mode}.json", "hinge", mode, h, records, summary)
        assert max(loads) > 0.1 and min(velocities) < 1.95
        assert summary["changing_area"]
        assert areas[0] == 0
        assert max(areas) <= AREA * (1 + 1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_current_device_pose_tamper_refuses_before_physics_and_retry(mode, monkeypatch):
    runtime, scene = runtime_for(mode, 0.0005)
    with runtime:
        scene.step(0.0005, expected_revision=runtime.checkpoint_revision)
        before = runtime.integration_snapshot()[13]
        geometry = scene.last_geometry
        revision = runtime.checkpoint_revision
        original = runtime._device._get

        def corrupted(buffer):
            result = original(buffer)
            if buffer is runtime._device.q[runtime._device.active]:
                result[0, 0] += 0.01
            return result

        with monkeypatch.context() as patch:
            patch.setattr(runtime._device, "_get", corrupted)
            with pytest.raises(RuntimeError, match="accepted state"):
                scene.step(0.0005, expected_revision=revision)
        after = runtime.integration_snapshot()[13]
        np.testing.assert_array_equal(before.q, after.q)
        np.testing.assert_array_equal(before.v, after.v)
        assert before.last_step is after.last_step
        assert scene.last_geometry is geometry
        assert runtime.checkpoint_revision == revision
        scene.step(0.0005, expected_revision=revision)
