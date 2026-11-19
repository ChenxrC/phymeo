"""Physical native friction and integrated area pressure on the same damped hinge."""

import json

import numpy as np
import pytest
import test_mujoco_tyx_area_pressure as area
from test_mujoco_tyx_area_pressure import MODES, TOL, H

from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime
from phymeo.backends.mujoco_runtime import MuJoCoInitialState

COM_X = 0.05
MASS_MOMENT = 0.03 + 0.6 * COM_X**2
DAMPING = 5.0
V0 = 0.35
PIVOT = np.array([0.0, 0.0, 0.1])
VERTICES = np.array([[0, -0.05, 0.15], [0, 0.05, 0.15], [0, 0, 0.25]])


def _hinge_xml():
    friction = 'condim="3" friction=".7 .01 .001"'
    return area._xml(
        f"""<body name="compliant" pos="0 0 .1">
          <joint name="turn" type="hinge" axis="0 1 0" damping="5"/>
          <inertial mass=".6" pos=".05 0 0" diaginertia=".02 .03 .04"/>
          <geom name="foot" type="sphere" size=".05" pos=".12 0 -.052" {friction}/>
        </body><body name="rigid"><geom type="box" size=".01 .06 .05"
          pos="-.01 0 .2" contype="0" conaffinity="0"/></body>""",
        static=f'<geom name="floor" type="plane" size="2 2 .1" {friction}/>',
    )


def _reference(mj, model, include_area):
    # MuJoCo supplies only entry geometry and coefficients of its own native rows.
    data = mj.MjData(model)
    data.qvel[0] = V0
    mj.mj_forward(model, data)
    assert data.ncon == 1 and data.nefc == 4 and data.contact[0].dim == 3
    native_j = data.efc_J.reshape(-1)[:4].copy()
    reg = data.efc_R[:4].copy()
    aref = data.efc_aref[:4].copy()
    points = area.BARYCENTRIC @ VERTICES if include_area else np.empty((0, 3))
    levers = points[:, 2] - PIVOT[2]
    gain = MASS_MOMENT / (MASS_MOMENT + H * DAMPING)
    k = area.AREA * area.GRADIENT / 3

    def forces(acceleration):
        native = np.maximum(0, (aref - native_j * acceleration) / reg)
        velocity = V0 + H * gain * acceleration
        pressure = k * np.maximum(0, area.DEPTH - H * levers * velocity)
        return native, pressure

    def balance(acceleration):
        native, pressure = forces(acceleration)
        return MASS_MOMENT * acceleration + DAMPING * V0 - native_j @ native - levers @ pressure

    lo, hi = -1e4, 1e4
    assert balance(lo) < 0 < balance(hi)
    for _ in range(64):
        mid = (lo + hi) / 2
        if balance(mid) < 0:
            lo = mid
        else:
            hi = mid
    acceleration = (lo + hi) / 2
    native, pressure = forces(acceleration)
    contact = data.contact[0]
    fn = native.sum()
    local = np.array(
        [
            fn,
            contact.friction[0] * (native[0] - native[1]),
            contact.friction[1] * (native[2] - native[3]),
        ]
    )
    force_world = contact.frame.reshape(3, 3).T @ local
    native_torque = np.cross(contact.pos - PIVOT, force_world)[1]
    assert native_torque == pytest.approx(native_j @ native, abs=1e-10)
    assert abs(balance(acceleration)) < 1e-10
    return (
        acceleration,
        gain * acceleration,
        native_j,
        reg,
        aref,
        native,
        pressure,
        levers,
        force_world,
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("include_area", [True, False], ids=["shared", "empty"])
def test_friction_and_pressure_share_one_damped_hinge(mode, include_area, tmp_path):
    xml = _hinge_xml()
    mj, model, runtime = area._runtime(xml, mode, velocity=(V0,))
    with runtime:
        a, a_int, native_j, reg, aref, native, pressure, levers, force_world = _reference(
            mj, model, include_area
        )
        state = area._step(runtime, area._mapping(runtime, empty=not include_area))[11]
        step = state.last_step
        rows = step.constraints
        assert len(rows) == 4 + len(pressure)
        assert len(step.contacts) == 1 and step.contacts[0].dimension == 3
        assert step.raw_contact_fields["native_contact_operator"] == "M"
        native_rows = rows[:4]
        for actual, expected in (
            ([row.jacobian[0] for row in native_rows], native_j),
            ([row.regularization for row in native_rows], reg),
            ([row.reference_acceleration for row in native_rows], aref),
            ([row.force for row in native_rows], native),
            ([row.force for row in rows[4:]], pressure),
            ([row.jacobian[0] for row in rows[4:]], levers),
            (step.contacts[0].force_on_geometry2_world, force_world),
            (step.qacc, [a]),
            (step.integration_acceleration, [a_int]),
            (state.v, [V0 + H * a_int]),
            (state.q, [H * (V0 + H * a_int)]),
            (step.qfrc_constraint, [native_j @ native + levers @ pressure]),
        ):
            np.testing.assert_allclose(actual, expected, **TOL)
        record = step.raw_contact_fields["area_pressure"][0]
        np.testing.assert_allclose(record["normal_force"], pressure, **TOL)
        np.testing.assert_allclose(record["force_on_a"], -record["force_on_b"], **TOL)
        assert np.linalg.norm(force_world[:2]) > 0.01 and force_world[2] > 1.0
        assert abs(native_j @ native) > 0.05 and abs(a - a_int) > 1.0
        if include_area:
            assert levers @ pressure > 0.05
            inv_kh2 = 3 / (area.AREA * area.GRADIENT * H**2)
            for row, lever in zip(rows[4:], levers, strict=True):
                assert row.regularization == pytest.approx(inv_kh2, rel=2e-6)
                assert row.reference_acceleration == pytest.approx(
                    area.DEPTH / H**2 - lever * V0 / H, rel=2e-6
                )
                assert abs(row.gradient) < 2e-5 * max(1.0, row.regularization * row.force)
        else:
            with MuJoCoTyxRuntime(
                area.normal._compile(xml),
                mode=mode,
                world_keys=(11,),
                initial_states={11: MuJoCoInitialState(model.qpos0, [V0])},
            ) as plain:
                plain.step(H)
                native_state = plain.integration_snapshot()[11]
                for left, right in (
                    (state.q, native_state.q),
                    (state.v, native_state.v),
                    (step.qacc, native_state.last_step.qacc),
                    (step.qfrc_constraint, native_state.last_step.qfrc_constraint),
                    (
                        [r.force for r in rows],
                        [r.force for r in native_state.last_step.constraints],
                    ),
                ):
                    np.testing.assert_array_equal(left, right)
        output = dict(
            mode=mode,
            area=include_area,
            q=float(state.q[0]),
            v=float(state.v[0]),
            physical_acc=float(step.qacc[0]),
            integrated_acc=float(step.integration_acceleration[0]),
            support_force=step.contacts[0].force_on_geometry2_world.tolist(),
            support_torque=float(
                step.contacts[0].wrench_on_body2_world[4]
                - COM_X * step.contacts[0].force_on_geometry2_world[2]
            ),
            area_load=float(sum(row.force for row in rows[4:])),
            area_torque=float(sum(row.jacobian[0] * row.force for row in rows[4:])),
            row_forces=[row.force for row in rows],
        )
        (tmp_path / "hinge.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


def _pressure_asset():
    from test_mujoco_tyx_area_pressure_scene import BASE, CELLS, RIGID

    from phymeo.backends.mujoco_area_pressure_scene import BodyAreaPressureAsset
    from phymeo.geometry.area_pressure_intersection import RigidPressureMesh, TetraPressureMesh

    # A four-tet pressure volume on the rotating link and an eccentric fixed face.
    return BodyAreaPressureAsset(
        area.AreaPressureIdentity(*area.PAIR, "elastic", "plate"),
        TetraPressureMesh(
            ("a", "b", "c", "d", "center"),
            np.vstack([BASE, BASE.mean(axis=0)]) + [-0.06, 0, -0.002],
            CELLS,
            (0, 0, 0, 0, 400000),
            ("bottom", "side-a", "side-b", "slant"),
        ),
        RigidPressureMesh(("r0", "r1", "r2"), RIGID + [-0.06, 0, 0.1], ((0, 1, 2),), ("face",)),
    )


@pytest.mark.parametrize("mode", MODES)
def test_current_surface_mixed_support_replays_from_moving_state(mode, tmp_path):
    from phymeo.backends.mujoco_area_pressure_scene import MuJoCoTyxAreaPressureScene

    _, _, runtime = area._runtime(_hinge_xml(), mode, velocity=(V0,), constraint_capacity=64)
    with runtime:
        scene = MuJoCoTyxAreaPressureScene(runtime, (_pressure_asset(),))

        def advance():
            before = runtime.integration_snapshot()[11]
            scene.step(H, expected_revision=runtime.checkpoint_revision)
            state = runtime.integration_snapshot()[11]
            step = state.last_step
            record = step.raw_contact_fields["area_pressure"][0]
            assert record["stamp"] == scene.last_geometry[11][0].surface.stamp
            assert record["stamp"].step_index == before.tick
            np.testing.assert_array_equal(step.input_v, before.v)
            force = record["force_on_a"]
            np.testing.assert_array_equal(force, -record["force_on_b"])
            support = step.contacts[0]
            np.testing.assert_array_equal(
                support.force_on_geometry1_world, -support.force_on_geometry2_world
            )
            support_torque = np.cross(support.position - PIVOT, support.force_on_geometry2_world)[1]
            pressure_torque = np.cross(record["points"] - PIVOT, force).sum(axis=0)[1]
            assert abs(pressure_torque) > 0.01 and abs(support_torque) > 0.01
            np.testing.assert_allclose(
                step.qfrc_constraint, [support_torque + pressure_torque], **TOL
            )
            np.testing.assert_allclose(
                state.v,
                before.v
                + H * (step.qfrc_constraint - DAMPING * before.v) / (MASS_MOMENT + H * DAMPING),
                **TOL,
            )
            return [
                state.tick,
                float(state.q[0]),
                float(state.v[0]),
                support.normal_force,
                float(record["normal_force"].sum()),
                float(pressure_torque),
                float(support_torque),
            ]

        first = advance()
        checkpoint = scene.checkpoint()
        tail = [advance(), advance()]
        assert len({row[4] for row in [first, *tail]}) == 3
        scene.restore(checkpoint, expected_revision=runtime.checkpoint_revision)
        replay = [advance(), advance()]
        np.testing.assert_array_equal(replay, tail)
        (tmp_path / "trajectory.json").write_text(json.dumps([first, *tail]), encoding="utf-8")


@pytest.mark.parametrize("mode", MODES)
def test_combined_capacity_failure_preserves_state_then_allows_native_only_retry(mode):
    _, _, runtime = area._runtime(_hinge_xml(), mode, velocity=(V0,), constraint_capacity=4)
    with runtime:
        before = area._retention_input(runtime)
        with pytest.raises(area.PhysicsStepFailure):
            area._step(runtime, area._mapping(runtime))
        area._retained(runtime, *before)
        state = area._step(runtime, area._mapping(runtime, empty=True))[11]
        assert state.tick == 1 and state.last_step.contacts[0].normal_force > 0


@pytest.mark.parametrize("mode", MODES)
def test_inclined_damped_slide_uses_physical_native_and_integrated_pressure_laws(mode):
    xml = _hinge_xml().replace('type="hinge" axis="0 1 0"', 'type="slide" axis=".6 0 -.8"')
    mj, model, runtime = area._runtime(xml, mode, velocity=(V0,))
    with runtime:
        cpu = mj.MjData(model)
        cpu.qvel[0] = V0
        mj.mj_forward(model, cpu)
        assert cpu.nefc == 4
        state = area._step(runtime, area._mapping(runtime))[11]
        step = state.last_step
        native_j = cpu.efc_J.reshape(-1)
        native_f = np.maximum(0, (cpu.efc_aref - native_j * step.qacc[0]) / cpu.efc_R)
        pressure_f = area.AREA * area.GRADIENT * max(0, area.DEPTH - H * 0.6 * state.v[0])
        assert native_f.sum() > 0 and pressure_f > 0
        np.testing.assert_allclose([r.force for r in step.constraints[:4]], native_f, **TOL)
        np.testing.assert_allclose(sum(r.force for r in step.constraints[4:]), pressure_f, **TOL)
        force = native_j @ native_f + 0.6 * pressure_f - DAMPING * V0
        np.testing.assert_allclose(0.6 * step.qacc, force, **TOL)
        np.testing.assert_allclose(state.v, V0 + H * force / (0.6 + H * DAMPING), **TOL)
