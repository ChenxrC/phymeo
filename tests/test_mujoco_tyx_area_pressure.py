"""Independent real-step oracles for the opt-in area-pressure normal route.

The triangles are prescribed input surfaces, not collision certificates. The
oracles below use hand-specified masses, joint axes and pressure geometry; no
area-row assembler or native contact solve supplies the expected motion.
Authored without execution; the integration owner runs interpret and CUDA.
"""

from dataclasses import replace

import numpy as np
import pytest
import test_mujoco_tyx_normal_contacts as normal
from test_mujoco_tyx_floating import _same

from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime, MuJoCoTyxUnsupported
from phymeo.backends.mujoco_runtime import MuJoCoInitialState
from phymeo.geometry.area_pressure import (
    AreaPressureIdentity,
    AreaPressureStamp,
    AreaPressureSurface,
    AreaPressureTriangle,
)
from phymeo.training.resident_stepper import PhysicsStepFailure

MODES = normal.MODES
H = 0.002
PAIR = ("compliant", "rigid")
MASS = np.array([0.5, 1.25])
DAMPING = np.array([3.0, 7.0])
GRADIENT = 2e6
DEPTH = 0.001
AREA = 0.005
BARYCENTRIC = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
TOL = dict(rtol=4e-4, atol=2e-5)
APPROX = dict(rel=4e-4, abs=2e-5)


def _xml(body, *, static=""):
    return f'''<mujoco model="area-pressure-port">
      <compiler angle="radian" autolimits="true" fusestatic="false"/>
      <option timestep="{H}" gravity="0 0 0" integrator="Euler"
        solver="Newton" iterations="128" tolerance="1e-10" cone="pyramidal" jacobian="dense"/>
      <default><geom condim="1" friction="0 0 0" solref=".02 1"
        solimp=".8 .9 .002 .5 2"/><joint armature="0" damping="0"/></default>
      <worldbody>{static}{body}</worldbody></mujoco>'''


def _sliders_xml(*, damping=(3.0, 7.0), native=False, limit=False, dry=False):
    bodies = []
    for index, (name, x, mass, damp) in enumerate(
        zip(PAIR, (0.099, -0.099), MASS, damping, strict=True)
    ):
        extra = ' limited="true" range="-.1 .1" margin=".001"' if limit and index == 0 else ""
        loss = 0.1 if dry and index == 0 else 0.0
        # All four possible geom pairs between these two bodies belong to the
        # area route, including those that are not currently penetrating.
        bodies.append(f'''<body name="{name}" pos="{x} 0 .2">
          <joint name="slide{index}" type="slide" axis="1 0 0" damping="{damp}"
            frictionloss="{loss}"{extra}/>
          <inertial mass="{mass}" pos="0 .01 -.02" diaginertia=".02 .03 .04"/>
          <geom name="{name}-near" type="sphere" size=".1" pos="0 -.15 0" contype="1" conaffinity="1"/>
          <geom name="{name}-far" type="sphere" size=".1" pos="0 .15 0" contype="1" conaffinity="1"/>
        </body>''')
    static = ""
    if native:
        static = (
            '<geom name="native-floor" type="plane" size="2 2 .1" contype="2" conaffinity="2"/>'
        )
        bodies.append("""<body name="native-slider" pos=".6 0 .098">
          <joint name="native-z" type="slide" axis="0 0 1"/>
          <inertial mass=".7" pos="0 0 0" diaginertia=".02 .03 .04"/>
          <geom name="native-ball" type="sphere" size=".1" contype="2" conaffinity="2"/>
        </body>""")
    return _xml("".join(bodies), static=static)


def _hinge_slider_xml():
    return _xml("""<body name="compliant" pos="0 0 .3">
      <joint name="turn" type="hinge" axis="0 1 0" damping="3"/>
      <inertial mass=".5" pos=".03 0 -.02" diaginertia=".02 .03 .04"/>
      <geom name="hinge-surface" type="sphere" size=".06" pos=".1 0 -.18"/>
      </body><body name="rigid" pos="0 0 .12">
      <joint name="slide" type="slide" axis="1 0 0" damping="7"/>
      <inertial mass="1.25" pos="-.01 .02 0" diaginertia=".02 .03 .04"/>
      <geom name="slider-surface" type="sphere" size=".06"/>
      </body>""")


def _runtime(xml, mode, *, keys=(11,), velocity=(-0.2, 0.1), **kwargs):
    mj = normal._deps()
    model = mj.MjModel.from_xml_string(xml)
    velocities = np.asarray(velocity, dtype=np.float64)
    assert velocities.shape == (model.nv,)
    runtime = MuJoCoTyxRuntime(
        normal._compile(xml),
        mode=mode,
        world_keys=keys,
        initial_states={key: MuJoCoInitialState(model.qpos0, velocities) for key in keys},
        area_pressure_pairs=(PAIR,),
        **kwargs,
    )
    return mj, model, runtime


def _surface(runtime, key, *, vertices=None, depth=DEPTH, empty=False, pair=PAIR):
    state = runtime.integration_snapshot()[key]
    stamp = AreaPressureStamp(
        str(key), state.tick, state.time, runtime.area_pressure_configuration_id(key)
    )
    if vertices is None:
        vertices = np.array([[0, -0.05, 0.15], [0, 0.05, 0.15], [0, 0, 0.25]])
    triangles = (
        ()
        if empty
        else (
            AreaPressureTriangle(
                "triangle-0",
                "cell-0",
                "face-0",
                vertices,
                np.full(3, GRADIENT * depth),
                np.array([GRADIENT, 0, 0]),
            ),
        )
    )
    return AreaPressureSurface(
        AreaPressureIdentity(*pair, "pressure-material", "rigid-material"), stamp, triangles
    )


def _mapping(runtime, **kwargs):
    return {key: (_surface(runtime, key, **kwargs),) for key in runtime.world_keys}


def _step(runtime, mapping):
    accepted = runtime.step_area_pressure(H, mapping, expected_revision=runtime.checkpoint_revision)
    assert accepted == dict.fromkeys(runtime.world_keys, True)
    return runtime.integration_snapshot()


def _retained(runtime, before, revision, configurations):
    actual = runtime.integration_snapshot()
    _same(before, actual)
    assert runtime.checkpoint_revision == revision
    for key in runtime.world_keys:
        assert actual[key].last_step is before[key].last_step
        assert runtime.area_pressure_configuration_id(key) == configurations[key]
        for field in ("qfrc_applied", "xfrc_applied"):
            np.testing.assert_array_equal(getattr(actual[key], field), getattr(before[key], field))


def _retention_input(runtime):
    return (
        runtime.integration_snapshot(),
        runtime.checkpoint_revision,
        {key: runtime.area_pressure_configuration_id(key) for key in runtime.world_keys},
    )


def _slider_solution(velocity, damping):
    """Two-body BE closed form, summing three identical-J area springs."""
    effective = MASS + H * np.asarray(damping)
    free = MASS * velocity / effective
    jacobian = np.array([1.0, -1.0])
    stiffness = AREA * GRADIENT
    force = (
        stiffness
        * (DEPTH - H * (jacobian @ free))
        / (1 + H * H * stiffness * np.sum(1 / effective))
    )
    assert force > 0  # This particular oracle stays on the compressive branch.
    updated = free + H * jacobian * force / effective
    return updated, force


def _area_rows(receipt):
    return tuple(row for row in receipt.constraints if row.constraint_type == 7)


def _check_area_export(receipt, surface, body_ids, points, area, jacobian, force):
    records = receipt.raw_contact_fields["area_pressure"]
    assert isinstance(records, tuple) and len(records) == 1
    record = records[0]
    assert set(record) == {
        "source_quadrature",
        "identity",
        "stamp",
        "sample_ids",
        "source_ids",
        "body_ids",
        "dA",
        "points",
        "normals",
        "stiffness",
        "surrogate_gap",
        "normal_jacobian",
        "normal_force",
        "force_on_a",
        "force_on_b",
        "torque_on_a_world_origin",
        "torque_on_b_world_origin",
    }
    assert record["identity"] == surface.identity
    assert record["stamp"] == surface.stamp
    assert record["sample_ids"] == tuple(("triangle-0", i) for i in range(3))
    assert record["source_ids"] == (("cell-0", "face-0"),) * 3
    assert record["body_ids"] == body_ids
    np.testing.assert_allclose(record["points"], points, **TOL)
    np.testing.assert_allclose(record["dA"], area, rtol=0, atol=1e-15)
    np.testing.assert_allclose(record["normals"], np.tile([1.0, 0.0, 0.0], (3, 1)), **TOL)
    np.testing.assert_allclose(record["normal_jacobian"], jacobian, **TOL)
    np.testing.assert_allclose(record["normal_force"], force, **TOL)
    np.testing.assert_allclose(record["stiffness"], area * GRADIENT, **TOL)
    np.testing.assert_allclose(record["surrogate_gap"], -DEPTH, **TOL)
    force_a = np.column_stack((force, np.zeros((3, 2))))
    moment_a = np.cross(points, force_a)
    np.testing.assert_allclose(record["force_on_a"], force_a, **TOL)
    np.testing.assert_allclose(record["force_on_b"], -force_a, **TOL)
    np.testing.assert_allclose(record["torque_on_a_world_origin"], moment_a, **TOL)
    np.testing.assert_allclose(record["torque_on_b_world_origin"], -moment_a, **TOL)
    np.testing.assert_array_equal(record["force_on_a"] + record["force_on_b"], 0)
    np.testing.assert_array_equal(
        record["torque_on_a_world_origin"] + record["torque_on_b_world_origin"], 0
    )
    with pytest.raises(TypeError):
        record["stamp"] = surface.stamp
    for value in record.values():
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
            with pytest.raises(ValueError):
                value.setflags(write=True)


def _check_row_law(rows, jacobian, pressure, gradient, area, velocity, updated):
    assert len(rows) == len(jacobian)
    stiffness = area * gradient
    depth = pressure / gradient
    force = np.array([row.force for row in rows])
    np.testing.assert_allclose(force, stiffness * (depth - H * (jacobian @ updated)), **TOL)
    for index, row in enumerate(rows):
        np.testing.assert_allclose(row.jacobian, jacobian[index], **TOL)
        assert row.position == pytest.approx(-depth[index], **APPROX)
        assert row.velocity == pytest.approx(jacobian[index] @ velocity, **APPROX)
        assert row.regularization == pytest.approx(1 / (H * H * stiffness[index]), **APPROX)
        assert row.reference_acceleration == pytest.approx(
            depth[index] / (H * H) - (jacobian[index] @ velocity) / H, **APPROX
        )
        assert row.force >= 0


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "damping", [(0.0, 0.0), tuple(DAMPING)], ids=["physical-mass", "implicit-damping"]
)
def test_two_sliders_actual_backward_euler_uses_both_masses_and_damping(mode, damping):
    mj, model, runtime = _runtime(_sliders_xml(damping=damping), mode)
    with runtime:
        before = runtime.integration_snapshot()[11]
        mapping = _mapping(runtime)
        expected_v, expected_force = _slider_solution(before.v, damping)
        after = _step(runtime, mapping)[11]
        row = after.last_step
        np.testing.assert_allclose(after.v, expected_v, **TOL)
        np.testing.assert_allclose(after.q, before.q + H * expected_v, rtol=4e-4, atol=1e-7)
        rows = _area_rows(row)
        assert len(rows) == len(row.constraints) == 3
        assert not row.contacts
        assert sum(item.force for item in rows) == pytest.approx(expected_force, **APPROX)
        np.testing.assert_allclose(row.qfrc_constraint, [expected_force, -expected_force], **TOL)
        np.testing.assert_allclose(row.integration_acceleration, (expected_v - before.v) / H, **TOL)
        np.testing.assert_allclose(
            row.qacc,
            (np.array([expected_force, -expected_force]) - np.asarray(damping) * before.v) / MASS,
            **TOL,
        )
        np.testing.assert_allclose(
            MASS * (after.v - before.v) + H * np.asarray(damping) * after.v,
            H * np.array([expected_force, -expected_force]),
            rtol=4e-4,
            atol=1e-7,
        )
        compression = max(0.0, DEPTH - H * (after.v[0] - after.v[1]))
        old_energy = 0.5 * np.sum(MASS * before.v**2) + 0.5 * AREA * GRADIENT * DEPTH**2
        new_energy = 0.5 * np.sum(MASS * after.v**2) + 0.5 * AREA * GRADIENT * compression**2
        be_loss = (
            0.5 * np.sum(MASS * (after.v - before.v) ** 2)
            + H * np.sum(np.asarray(damping) * after.v**2)
            + 0.5 * AREA * GRADIENT * (compression - DEPTH) ** 2
        )
        assert new_energy <= old_energy
        assert new_energy + be_loss == pytest.approx(old_energy, rel=4e-4, abs=1e-7)
        _check_row_law(
            rows,
            np.tile([1.0, -1.0], (3, 1)),
            np.full(3, GRADIENT * DEPTH),
            np.full(3, GRADIENT),
            np.full(3, AREA / 3),
            before.v,
            after.v,
        )
        assert after.tick == before.tick + 1
        assert row.contact_time == before.time
        assert row.input_time == before.time
        assert row.output_time == after.time
        np.testing.assert_array_equal(row.input_q, before.q)
        np.testing.assert_array_equal(row.input_v, before.v)
        _check_area_export(
            row,
            mapping[11][0],
            tuple(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name) for name in PAIR),
            BARYCENTRIC @ mapping[11][0].triangles[0].vertices,
            np.full(3, AREA / 3),
            np.tile([1.0, -1.0], (3, 1)),
            np.full(3, expected_force / 3),
        )


@pytest.mark.parametrize("mode", MODES)
def test_hinge_offset_com_and_three_distinct_point_levers_drive_real_motion(mode):
    mj, model, runtime = _runtime(_hinge_slider_xml(), mode, velocity=(0.0, 0.0))
    with runtime:
        vertices = np.array([[0.1, -0.04, 0.1], [0.1, 0.04, 0.1], [0.1, 0, 0.16]])
        points = BARYCENTRIC @ vertices
        weights = np.full(3, 0.0024 / 3)
        # y-axis at world z=.3: (axis x r).x = r.z. The inertia is
        # about that joint, not the offset COM's own diagonal inertia.
        jacobian = np.column_stack((points[:, 2] - 0.3, -np.ones(3)))
        mass = np.array([0.03 + 0.5 * (0.03**2 + 0.02**2), 1.25])
        effective = mass + H * DAMPING
        stiffness = weights * GRADIENT
        velocity = np.linalg.solve(
            np.diag(effective) + H * H * jacobian.T @ (stiffness[:, None] * jacobian),
            H * jacobian.T @ (stiffness * DEPTH),
        )
        force = stiffness * (DEPTH - H * (jacobian @ velocity))
        assert np.all(force > 0)
        before = runtime.integration_snapshot()[11]
        mapping = _mapping(runtime, vertices=vertices)
        after = _step(runtime, mapping)[11]
        row = after.last_step
        np.testing.assert_allclose(after.v, velocity, **TOL)
        np.testing.assert_allclose(after.q, before.q + H * velocity, rtol=4e-4, atol=1e-7)
        np.testing.assert_allclose(row.qfrc_constraint, jacobian.T @ force, **TOL)
        assert after.v[0] < 0 and after.v[1] < 0
        np.testing.assert_allclose([item.force for item in _area_rows(row)], force, **TOL)
        _check_row_law(
            _area_rows(row),
            jacobian,
            np.full(3, GRADIENT * DEPTH),
            np.full(3, GRADIENT),
            weights,
            before.v,
            after.v,
        )
        _check_area_export(
            row,
            mapping[11][0],
            tuple(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name) for name in PAIR),
            points,
            weights,
            jacobian,
            force,
        )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("remaining", ["native-pair", "limit", "dry-friction"])
def test_nonzero_damping_refuses_possible_native_constraints_before_execution(mode, remaining):
    xml = _sliders_xml(
        native=remaining == "native-pair",
        limit=remaining == "limit",
        dry=remaining == "dry-friction",
    )
    normal._deps()
    with pytest.raises(ValueError, match="damped area route"):
        MuJoCoTyxRuntime(normal._compile(xml), mode=mode, area_pressure_pairs=(PAIR,))


@pytest.mark.parametrize("mode", MODES)
def test_empty_declared_surface_advances_only_real_joint_damping(mode):
    _, _, runtime = _runtime(_sliders_xml(), mode)
    with runtime:
        before = runtime.integration_snapshot()[11]
        expected = MASS * before.v / (MASS + H * DAMPING)
        after = _step(runtime, _mapping(runtime, empty=True))[11]
        np.testing.assert_allclose(after.v, expected, **TOL)
        np.testing.assert_allclose(after.q, before.q + H * expected, rtol=4e-4, atol=1e-7)
        assert not after.last_step.constraints
        assert not after.last_step.contacts
        record = after.last_step.raw_contact_fields["area_pressure"][0]
        assert record["sample_ids"] == ()
        assert record["points"].shape == (0, 3)
        assert record["normal_jacobian"].shape == (0, 2)
        assert record["normal_force"].shape == (0,)


@pytest.mark.parametrize("mode", MODES)
def test_zero_pressure_nonempty_surface_and_separating_motion_have_no_attraction(mode):
    _, _, runtime = _runtime(_sliders_xml(), mode, velocity=(0.2, -0.1))
    with runtime:
        before = runtime.integration_snapshot()[11]
        packet = _mapping(runtime, depth=0.0)
        assert len(packet[11][0].triangles) == 1
        after = _step(runtime, packet)[11]
        expected = MASS * before.v / (MASS + H * DAMPING)
        np.testing.assert_allclose(after.v, expected, **TOL)
        np.testing.assert_allclose(after.q, before.q + H * expected, rtol=4e-4, atol=1e-7)
        rows = _area_rows(after.last_step)
        assert len(rows) == 3
        assert all(row.position == 0 and row.velocity > 0 for row in rows)
        assert all(row.force == 0 and row.reference_acceleration < 0 for row in rows)
        np.testing.assert_array_equal(after.last_step.qfrc_constraint, 0)
        record = after.last_step.raw_contact_fields["area_pressure"][0]
        assert record["points"].shape == (3, 3)
        for field in (
            "normal_force",
            "force_on_a",
            "force_on_b",
            "torque_on_a_world_origin",
            "torque_on_b_world_origin",
        ):
            np.testing.assert_array_equal(record[field], 0)


@pytest.mark.parametrize("mode", MODES)
def test_undamped_other_native_pair_survives_area_pair_exclusion(mode):
    xml = _sliders_xml(damping=(0.0, 0.0), native=True)
    mj, model, runtime = _runtime(xml, mode, velocity=(-0.2, 0.1, 0.0))
    plain = MuJoCoTyxRuntime(
        normal._compile(xml),
        mode=mode,
        world_keys=(11,),
        initial_states={11: MuJoCoInitialState(model.qpos0, [-0.2, 0.1, 0.0])},
    )
    with runtime, plain:
        before = runtime.integration_snapshot()[11]
        expected_v, expected_force = _slider_solution(before.v[:2], (0.0, 0.0))
        after = _step(runtime, _mapping(runtime))[11]
        plain.step(H)
        native = plain.integration_snapshot()[11]
        np.testing.assert_allclose(after.v[:2], expected_v, **TOL)
        np.testing.assert_allclose(after.v[2:], native.v[2:], **TOL)
        np.testing.assert_allclose(after.q[2:], native.q[2:], rtol=4e-4, atol=1e-7)
        assert after.v[2] > 0
        assert sum(row.force for row in _area_rows(after.last_step)) == pytest.approx(
            expected_force, **APPROX
        )
        assert any(row.constraint_type == 5 for row in after.last_step.constraints)
        assert len(after.last_step.contacts) == 1
        contact = after.last_step.contacts[0]
        expected_pair = {
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
            for name in ("native-floor", "native-ball")
        }
        assert set(contact.geometry_ids) == expected_pair
        assert contact.normal_force > 0
        # The undeclared runtime demonstrates that the A/B native pair was
        # actually penetrating; absence on the area route is not a no-contact fixture.
        area_bodies = {mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name) for name in PAIR}
        assert any(set(item.body_ids) == area_bodies for item in native.last_step.contacts)


@pytest.mark.parametrize("mode", MODES)
def test_surface_binding_and_revision_preflight_is_transactional(mode):
    _, _, runtime = _runtime(_sliders_xml(), mode, keys=(11, 37))
    with runtime:
        good = _mapping(runtime)
        surface = good[37][0]
        wrong_surfaces = [
            replace(
                surface,
                identity=AreaPressureIdentity(*PAIR[::-1], "pressure-material", "rigid-material"),
            ),
            replace(surface, stamp=replace(surface.stamp, world_id="11")),
            replace(surface, stamp=replace(surface.stamp, step_index=1)),
            replace(surface, stamp=replace(surface.stamp, sample_time_s=H)),
            replace(surface, stamp=replace(surface.stamp, configuration_id="0" * 64)),
        ]
        packets = [
            {},
            {11: good[11]},
            {**good, 99: good[37]},
            {**good, 37: ()},
            {**good, 37: (surface, surface)},
            *({**good, 37: (bad,)} for bad in wrong_surfaces),
        ]
        before = _retention_input(runtime)
        for packet in packets:
            with pytest.raises(ValueError):
                runtime.step_area_pressure(H, packet, expected_revision=runtime.checkpoint_revision)
            _retained(runtime, *before)
        for revision in (
            runtime.checkpoint_revision + 1,
            float(runtime.checkpoint_revision),
            False,
        ):
            with pytest.raises(ValueError):
                runtime.step_area_pressure(H, good, expected_revision=revision)
            _retained(runtime, *before)
        with pytest.raises(ValueError):
            runtime.step(H)
        _retained(runtime, *before)
        with pytest.raises(ValueError):
            runtime.step_area_pressure(H / 2, good, expected_revision=runtime.checkpoint_revision)
        _retained(runtime, *before)
        _step(runtime, good)
        accepted = _retention_input(runtime)
        with pytest.raises(ValueError):
            runtime.step_area_pressure(H, good, expected_revision=runtime.checkpoint_revision)
        _retained(runtime, *accepted)
        _step(runtime, _mapping(runtime))


@pytest.mark.parametrize("mode", MODES)
def test_three_area_samples_cannot_fit_two_constraint_rows(mode):
    _, _, runtime = _runtime(_sliders_xml(), mode, constraint_capacity=2)
    with runtime:
        before = _retention_input(runtime)
        with pytest.raises(ValueError, match="capacity"):
            runtime.step_area_pressure(
                H, _mapping(runtime), expected_revision=runtime.checkpoint_revision
            )
        _retained(runtime, *before)
        _step(runtime, _mapping(runtime, empty=True))


@pytest.mark.parametrize("mode", MODES)
def test_late_second_world_publication_failure_retains_all_and_retry_is_real(mode, monkeypatch):
    import phymeo.backends.mujoco_tyx._constraint_publication as publication

    _, _, runtime = _runtime(_sliders_xml(), mode, keys=(37, 11))
    with runtime:
        _step(runtime, _mapping(runtime))
        before = _retention_input(runtime)
        packet = _mapping(runtime)
        seen = []
        original = publication.publish

        def refuse_last(*args, **kwargs):
            result = original(*args, **kwargs)
            seen.append(args[2])
            assert len(result[2]["area_pressure"]) == 1
            if len(seen) == 2:
                raise RuntimeError("deliberate last-world area publication failure")
            return result

        with monkeypatch.context() as change:
            change.setattr(publication, "publish", refuse_last)
            with pytest.raises(PhysicsStepFailure) as caught:
                runtime.step_area_pressure(H, packet, expected_revision=runtime.checkpoint_revision)
        assert seen == [0, 1]
        assert caught.value.acceptance_known and not caught.value.physics_accepted
        assert caught.value.accepted_worlds == {37: False, 11: False}
        _retained(runtime, *before)
        after = _step(runtime, packet)
        for key in runtime.world_keys:
            expected_v, expected_force = _slider_solution(before[0][key].v, DAMPING)
            np.testing.assert_allclose(after[key].v, expected_v, **TOL)
            np.testing.assert_allclose(
                after[key].q, before[0][key].q + H * expected_v, rtol=4e-4, atol=1e-7
            )
            assert sum(row.force for row in _area_rows(after[key].last_step)) == pytest.approx(
                expected_force, **APPROX
            )


@pytest.mark.parametrize("mode", MODES)
def test_reset_requires_new_stamp_and_checkpoint_keeps_history(mode):
    _, _, runtime = _runtime(_sliders_xml(), mode, keys=(11, 37))
    with runtime:
        initial = runtime.integration_snapshot()
        _step(runtime, _mapping(runtime))
        old = _mapping(runtime)
        before = _retention_input(runtime)
        assert runtime.capabilities["area_pressure_checkpoint"]
        checkpoint = runtime.checkpoint()
        assert checkpoint.source_revision == runtime.checkpoint_revision
        _retained(runtime, *before)
        with pytest.raises(ValueError, match="expected MuJoCoTyxCheckpoint"):
            runtime.restore(None, expected_revision=runtime.checkpoint_revision)
        _retained(runtime, *before)
        runtime.reset((11,))
        reset = runtime.integration_snapshot()
        _same({11: initial[11]}, {11: reset[11]})
        _same({37: before[0][37]}, {37: reset[37]})
        assert reset[11].last_step is None
        assert reset[37].last_step is before[0][37].last_step
        retained = _retention_input(runtime)
        with pytest.raises(ValueError):
            runtime.step_area_pressure(H, old, expected_revision=runtime.checkpoint_revision)
        _retained(runtime, *retained)
        after = _step(runtime, _mapping(runtime))
        assert (after[11].tick, after[37].tick) == (1, 2)


def test_area_table_pair_does_not_remove_wall_with_same_world_weld():
    from phymeo.backends.mujoco_area_pressure import compile_area_plan

    mj = pytest.importorskip("mujoco")
    xml = _xml(
        """<body name="object" pos=".1 0 .099">
        <joint name="object-z" type="slide" axis="0 0 1"/>
        <inertial mass=".5" pos="0 0 0" diaginertia=".02 .03 .04"/>
        <geom name="object-geom" type="sphere" size=".1"/></body>""",
        static="""<body name="table"><geom name="table-geom" type="plane" size="2 2 .1"/></body>
        <body name="wall" pos="-.1 0 0" quat=".7071067811865476 0 .7071067811865476 0">
        <geom name="wall-geom" type="plane" size="2 2 .1"/></body>""",
    )
    model = mj.MjModel.from_xml_string(xml)
    body_ids = {
        name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        for name in ("object", "table", "wall")
    }
    assert body_ids["table"] != body_ids["wall"]
    assert model.body_weldid[body_ids["table"]] == model.body_weldid[body_ids["wall"]] == 0
    plan = compile_area_plan(mj, model, (("object", "table"),))
    remaining = {
        frozenset(mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(geom)) for geom in pair[:2])
        for pair in plan.pair_ids[: plan.npair]
    }
    assert remaining == {frozenset(("wall-geom", "object-geom"))}
    np.testing.assert_array_equal(plan.area_body_ids, [[body_ids["object"], body_ids["table"]]])


@pytest.mark.parametrize("mode", MODES)
def test_nonzero_gap_underflow_cannot_silently_remove_contact(mode):
    _, _, runtime = _runtime(_sliders_xml(), mode)
    with runtime:
        before = _retention_input(runtime)
        packet = _mapping(runtime, depth=1e-42)
        with pytest.raises(MuJoCoTyxUnsupported, match="FP32"):
            runtime.step_area_pressure(H, packet, expected_revision=runtime.checkpoint_revision)
        _retained(runtime, *before)


@pytest.mark.parametrize("mode", MODES)
def test_actual_velocity_readback_is_bound_to_area_force(mode, monkeypatch):
    _, _, runtime = _runtime(_sliders_xml(), mode)
    with runtime:
        _step(runtime, _mapping(runtime))
        before = _retention_input(runtime)
        packet = _mapping(runtime)
        original = runtime._device.download

        def corrupt():
            result = original()
            result["v"][0, 0] += 0.01
            return result

        with monkeypatch.context() as change:
            change.setattr(runtime._device, "download", corrupt)
            with pytest.raises(PhysicsStepFailure):
                runtime.step_area_pressure(H, packet, expected_revision=runtime.checkpoint_revision)
        _retained(runtime, *before)
        _step(runtime, packet)
