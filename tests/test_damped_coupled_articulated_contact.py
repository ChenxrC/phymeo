"""A translating carriage and eccentric rotating link share two live contacts.

For c=.05 m, Iy=.03 kg m^2, m_parent=m_child=1 kg and slide axis (.6,0,-.8),
M00=2, M11=.0325, M01=.04*cos(theta)-.03*sin(theta). The input bias is
((- .04*sin(theta)-.03*cos(theta))*omega**2, 0). Both joints are damped.

The rotating child owns one native ground sphere and the existing finite side
pressure asset. Native forward/Jacobians provide input coefficients, never the
expected solution; the independent two-variable root from the slide tests does.
"""

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest
from test_damped_coupled_mixed_area_friction import _observe, _root, _value
from test_damped_hinge_mixed_area_friction import IDENTITY, _side_asset
from test_mujoco_area_friction_lifecycle import _evidence, _same
from test_mujoco_area_friction_live import _surface as _audit_surface
from test_mujoco_tyx_area_pressure import _xml

from phymeo.backends.mujoco_area_friction import AreaFrictionPairLaw
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.backends.mujoco_runtime import MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.solvers.area_friction import LAW

H = 0.002
KEY = 41
SLIDE = np.array([0.6, 0.0, -0.8])
AXIS = np.array([0.0, 1.0, 0.0])
PIVOT0 = np.array([0.0, 0.0, 0.1])
DAMPING = np.array([3.0, 5.0])
SCHEMA = "mujoco-tyx-area-friction-coupled-contact-v1"


def _declaration():
    friction = 'condim="3" friction=".7 .01 .001"'
    body = f"""<body name="carriage" pos="0 0 .1">
      <joint name="translation" type="slide" axis=".6 0 -.8" damping="3"/>
      <inertial mass="1" pos="0 0 0" diaginertia=".02 .03 .04"/>
      <body name="compliant">
        <joint name="rotation" type="hinge" axis="0 1 0" damping="5"/>
        <inertial mass="1" pos=".05 0 0" diaginertia=".02 .03 .04"/>
        <geom name="foot" type="sphere" size=".05" pos=".12 0 -.052" {friction}/>
      </body>
    </body><body name="rigid"><geom type="box" size=".01 .06 .05"
      pos="-.01 0 .2" contype="0" conaffinity="0"/></body>"""
    xml = _xml(body, static=f'<geom name="floor" type="plane" size="2 2 .1" {friction}/>')
    spec = MuJoCoTyxModelSpec(
        "damped-slide-eccentric-hinge",
        compile_mujoco_xml(xml, require_asset=False),
        contact_capacity=1,
        constraint_capacity=64,
        area_pressure_pairs=(("compliant", "rigid"),),
        area_friction_laws=(AreaFrictionPairLaw(IDENTITY, 0.5, 0.001),),
    )
    world = MuJoCoTyxWorldSpec(KEY, spec.key, MuJoCoInitialState((0.0, 0.0), (0.04, 0.35)))
    return spec, world


def _native(spec, before):
    import mujoco as mj

    model = spec.compilation.fresh_native_model()
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = before.q, before.v, before.time
    data.ctrl[:] = before.held_control.ctrl
    for name in ("qfrc_applied", "xfrc_applied"):
        held = getattr(before.held_control, name)
        if held is not None:
            getattr(data, name)[:] = held
    mj.mj_forward(model, data)
    assert model.nq == model.nv == 2 and model.neq == model.ntendon == 0
    assert tuple(model.jnt_type) == (mj.mjtJoint.mjJNT_SLIDE, mj.mjtJoint.mjJNT_HINGE)
    assert not np.any(model.jnt_limited) and not np.any(model.dof_frictionloss)
    np.testing.assert_array_equal(model.dof_damping, DAMPING)
    theta, omega = before.q[1], before.v[1]
    sine, cosine = np.sin(theta), np.cos(theta)
    coupling = 0.04 * cosine - 0.03 * sine
    analytic_mass = np.array([[2.0, coupling], [coupling, 0.0325]])
    analytic_bias = np.array([(-0.04 * sine - 0.03 * cosine) * omega**2, 0.0])
    mass = np.empty((2, 2))
    mj.mj_fullM(model, mass, data.qM)
    np.testing.assert_allclose(mass, analytic_mass, rtol=0, atol=1e-13)
    np.testing.assert_allclose(data.qfrc_bias, analytic_bias, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(data.qfrc_passive, -DAMPING * before.v, rtol=0, atol=1e-12)
    assert abs(mass[0, 1]) > 0.01
    pivot = PIVOT0 + SLIDE * before.q[0]
    rotation = np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
    child = model.body("compliant").id
    np.testing.assert_allclose(data.xpos[child], pivot, rtol=0, atol=1e-13)
    np.testing.assert_allclose(data.xmat[child].reshape(3, 3), rotation, rtol=0, atol=1e-13)
    np.testing.assert_allclose(
        data.xipos[child], pivot + rotation @ np.array([0.05, 0, 0]), rtol=0, atol=1e-13
    )
    assert data.ncon == 1 and data.nefc == 4 and data.contact[0].dim == 3
    contact = data.contact[0]
    assert (int(contact.geom1), int(contact.geom2)) == (
        model.geom("floor").id,
        model.geom("foot").id,
    )
    return model, data, mass, pivot


def _point_jacobians(model, data, points, pivot):
    import mujoco as mj

    child, plate = model.body("compliant").id, model.body("rigid").id
    ja, jb = [], []
    for point in points:
        left, right, angular = (np.empty((3, 2)) for _ in range(3))
        mj.mj_jac(model, data, left, angular, point, child)
        mj.mj_jac(model, data, right, angular, point, plate)
        analytic = np.column_stack((SLIDE, np.cross(AXIS, point - pivot)))
        np.testing.assert_allclose(left, analytic, rtol=0, atol=1e-13)
        np.testing.assert_array_equal(right, 0)
        ja.append(left.copy())
        jb.append(right.copy())
    return np.asarray(ja).reshape(-1, 3, 2), np.asarray(jb).reshape(-1, 3, 2)


def _audit(spec, before, after, geometry):
    model, data, source_mass, pivot = _native(spec, before)
    receipt = after.last_step
    assert receipt.tick == after.tick == before.tick + 1
    np.testing.assert_array_equal(receipt.input_q, before.q)
    np.testing.assert_array_equal(receipt.input_v, before.v)
    assert receipt.input_time == receipt.contact_time == receipt.dynamics.time == before.time
    assert receipt.contact_phase == "tyx-euler-input-constraint-evaluation"
    assert receipt.dynamics.phase == "tyx-euler-input-dynamics"
    raw = receipt.raw_contact_fields
    assert raw["native_contact_operator"] == "M" and raw["response_operator"] == "M+h*joint_damping"
    (pair,) = raw["area_friction"]
    solver = raw["area_friction_solver"]
    assert solver["schema"] == SCHEMA and solver["law"] == pair["law"] == LAW
    assert solver["solver_status"] == 0 and solver["precision"] == "actual-packed-point-fp32"
    _audit_surface(pair, geometry, receipt)
    assert len(receipt.contacts) == 1 and len(receipt.constraints) == 4
    rows = receipt.constraints
    jac = np.array([row.jacobian for row in rows])
    reg = np.array([row.regularization for row in rows])
    aref = np.array([row.reference_acceleration for row in rows])
    native_force = np.array([row.force for row in rows])
    for i, row in enumerate(rows):
        assert row.constraint_type == int(data.efc_type[i]) and row.source_id == int(data.efc_id[i])
    dyn = receipt.dynamics
    for actual, expected in (
        (jac, data.efc_J.reshape(4, 2)),
        (reg, data.efc_R),
        (aref, data.efc_aref),
        (solver["mass_matrix"], source_mass),
        (dyn.mass_matrix, source_mass),
        (solver["damping"], DAMPING),
        (dyn.bias, data.qfrc_bias),
        (dyn.passive, data.qfrc_passive),
        (dyn.applied, data.qfrc_applied + data.qfrc_actuator),
    ):
        np.testing.assert_allclose(actual, expected, rtol=4e-4, atol=2e-5)
    points = pair["points"].astype(float)
    ja, jb = _point_jacobians(model, data, points, pivot)
    np.testing.assert_allclose(pair["jacobian_a"], ja, rtol=2e-5, atol=2e-7)
    np.testing.assert_array_equal(pair["jacobian_b"], jb)
    assert len(points) > 0 and np.linalg.norm(ja[:, :, 1]) > 0.01
    assert tuple(pair["body_ids"]) == (model.body("compliant").id, model.body("rigid").id)
    source_contact = data.contact[0]
    ground_j = np.column_stack((SLIDE, np.cross(AXIS, source_contact.pos - pivot)))
    angular_j = np.column_stack((np.zeros(3), AXIS))
    frame = source_contact.frame.reshape(3, 3)
    source_contact_j = np.vstack((frame @ ground_j, frame @ angular_j))
    np.testing.assert_allclose(receipt.contacts[0].jacobian, source_contact_j, rtol=2e-5, atol=2e-7)
    # This source Jacobian has two real columns. Neither may be replaced by
    # the other, a scalar shortcut, or the previous entry's lever.
    assert np.linalg.norm(source_contact_j[:, 0]) > 0.1
    assert np.linalg.norm(source_contact_j[:, 1]) > 0.1
    h = solver["h_s"]
    assert h == float(np.float32(H))
    mass = solver["mass_matrix"].astype(float)
    response = mass + h * np.diag(solver["damping"].astype(float))
    applied = solver["applied_force"].astype(float)
    np.testing.assert_array_equal(solver["input_velocity"], before.v)
    np.testing.assert_array_equal(solver["output_velocity"], after.v)
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
    np.testing.assert_allclose(after.v, expected_v, rtol=5e-5, atol=5e-5)
    actual = _value(after.v.astype(float), p)
    for actual_value, reference in (
        (native_force, expected["native"]),
        (pair["normal_force"], expected["normal"]),
        (pair["tangential_force"], expected["tangent"]),
        (pair["force_on_a"], expected["force"]),
        (native_force, actual["native"]),
        (pair["normal_force"], actual["normal"]),
        (pair["tangential_force"], actual["tangent"]),
        (pair["force_on_a"], actual["force"]),
    ):
        np.testing.assert_allclose(actual_value, reference, rtol=0.002, atol=0.002)
    np.testing.assert_allclose(pair["relative_velocity"], actual["relative"], rtol=2e-5, atol=2e-5)
    force = pair["force_on_a"].astype(float)
    tangent = pair["tangential_force"].astype(float)
    normal = pair["normal_force"].astype(float)
    assert normal.sum() > 0.1 and np.linalg.norm(tangent) > 0.001
    assert np.all(normal >= 0)
    assert np.all(np.linalg.norm(tangent, axis=1) <= pair["mu"] * normal + 1e-5)
    assert np.all(np.sum(tangent * actual["tangent_velocity"], axis=1) <= 1e-14)
    np.testing.assert_allclose(np.sum(tangent * p["normal"], axis=1), 0, rtol=0, atol=2e-6)
    np.testing.assert_array_equal(pair["force_on_b"], -pair["force_on_a"])
    np.testing.assert_array_equal(
        pair["torque_on_b_world_origin"], -pair["torque_on_a_world_origin"]
    )
    np.testing.assert_allclose(
        pair["torque_on_a_world_origin"], np.cross(points, force), rtol=0.002, atol=1e-5
    )
    area_q = np.array(
        [SLIDE @ force.sum(axis=0), AXIS @ np.cross(points - pivot, force).sum(axis=0)]
    )
    np.testing.assert_allclose(area_q, np.einsum("sij,si->j", ja - jb, force), rtol=0, atol=1e-12)
    assert np.all(np.abs(area_q) > 0.001)
    local = np.array(
        [
            native_force.sum(),
            source_contact.friction[0] * (native_force[0] - native_force[1]),
            source_contact.friction[1] * (native_force[2] - native_force[3]),
        ]
    )
    ground_force = frame.T @ local
    contact = receipt.contacts[0]
    assert ground_force[2] > 0.1
    np.testing.assert_allclose(
        contact.force_on_geometry2_world, ground_force, rtol=0.002, atol=0.002
    )
    np.testing.assert_array_equal(
        contact.force_on_geometry1_world, -contact.force_on_geometry2_world
    )
    total_moment = np.zeros(3)
    for body, wrench, signed_force in (
        (contact.body_ids[0], contact.wrench_on_body1_world, -ground_force),
        (contact.body_ids[1], contact.wrench_on_body2_world, ground_force),
    ):
        expected_moment = np.cross(source_contact.pos - data.xipos[body], signed_force)
        np.testing.assert_allclose(
            wrench, np.r_[signed_force, expected_moment], rtol=0.002, atol=0.002
        )
        total_moment += wrench[3:] + np.cross(data.xipos[body], wrench[:3])
    np.testing.assert_allclose(total_moment, 0, rtol=0, atol=2e-5)
    ground_q = np.array(
        [SLIDE @ ground_force, AXIS @ np.cross(source_contact.pos - pivot, ground_force)]
    )
    np.testing.assert_allclose(ground_q, jac.T @ native_force, rtol=2e-5, atol=2e-5)
    total_q = area_q + ground_q
    np.testing.assert_allclose(receipt.qfrc_constraint, total_q, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(receipt.qfrc_constraint, solver["generalized_force"])
    point_power = float(np.sum(force * actual["relative"]) + ground_force @ (ground_j @ after.v))
    assert total_q @ after.v == pytest.approx(point_power, abs=2e-5, rel=2e-5)
    physical_rhs = dyn.applied + dyn.passive - dyn.bias + total_q
    acceleration = np.linalg.solve(mass, physical_rhs)
    integration = np.linalg.solve(response, physical_rhs)
    np.testing.assert_allclose(h * receipt.qacc, h * acceleration, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(
        h * receipt.integration_acceleration, h * integration, rtol=5e-5, atol=5e-5
    )
    np.testing.assert_allclose(after.v, before.v + h * integration, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(after.q, before.q + h * after.v, rtol=0, atol=2e-7)
    factor = np.linalg.cholesky(response)
    scale = 1 + np.linalg.norm(np.linalg.solve(factor, p["rhs"]))
    stored = response @ after.v - p["rhs"] - h * total_q
    assert np.linalg.norm(np.linalg.solve(factor, stored)) <= 2e-5 * scale
    assert np.linalg.norm(np.linalg.solve(factor, actual["residual"])) <= 2e-5 * scale
    assert solver["normalized_residual"] <= solver["tolerance"] == 2e-5
    gradients = jac @ receipt.qacc + reg * native_force - aref
    np.testing.assert_allclose([row.gradient for row in rows], gradients, rtol=2e-5, atol=2e-5)
    assert np.all(gradients >= -2e-5 * (1 + np.abs(aref)))
    return dict(
        source_mass=source_mass,
        packed_mass=mass,
        source_bias=data.qfrc_bias.copy(),
        area_j=ja,
        ground_j=source_contact_j,
        area_q=area_q,
        ground_q=ground_q,
        oracle=expected,
        expected_v=expected_v,
        newton_iterations=iterations,
        physical_acceleration=acceleration,
        integration_acceleration=integration,
    )


@pytest.mark.cuda
def test_damped_slide_hinge_updates_configuration_dependent_mass_bias_and_contact_jacobians(
    tmp_path,
):
    # A frozen mass, omitted bias, stale COM/lever or scalar-only integration
    # must fail on the same body's three successive accepted physical states.
    spec, world = _declaration()
    with _evidence(tmp_path / "articulated-current-three-step.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (_side_asset(),)})
            with batch.prepare_device_workspace(
                include_bodies=True, include_area_pressure=True
            ) as ws:
                previous = None
                for _ in range(3):
                    before = batch.integration_snapshot()[KEY]
                    if previous is not None:
                        _same(before, previous)
                    assert scene.step(H, expected_revision=batch.checkpoint_revision) == {KEY: True}
                    after = batch.integration_snapshot()[KEY]
                    geometry = scene.last_geometry[KEY][0]
                    record = dict(before=before, after=after, geometry=geometry)
                    evidence["records"].append(record)
                    record["audit"] = _audit(spec, before, after, geometry)
                    _, record["arrays"] = _observe(ws, batch, (spec,), {KEY: after})
                    previous = after
                records = evidence["records"]
                assert abs(records[0]["audit"]["source_bias"][0]) > 1e-4
                angles = [item["before"].q[1] for item in records]
                assert np.ptp(angles) > 1e-5
                coupling = [item["audit"]["source_mass"][0, 1] for item in records]
                packed_coupling = [item["audit"]["packed_mass"][0, 1] for item in records]
                assert np.ptp(coupling) > 1e-7 and np.ptp(packed_coupling) > 1e-7
                # Current native contact lever changes with genuine rotation,
                # even when the finite rigid area face stays geometrically fixed.
                assert (
                    max(
                        np.linalg.norm(
                            item["audit"]["ground_j"][:, 1] - records[0]["audit"]["ground_j"][:, 1]
                        )
                        for item in records[1:]
                    )
                    > 1e-7
                )


@pytest.mark.cuda
@pytest.mark.parametrize("column", [0, 1], ids=["slide-column", "hinge-column"])
def test_articulated_restore_rejects_changed_contact_jacobian_column_before_device_writes(
    monkeypatch, column
):
    from phymeo.backends import mujoco_checkpoint as codec

    spec, world = _declaration()
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (_side_asset(),)})
        assert scene.step(H, expected_revision=batch.checkpoint_revision) == {KEY: True}
        before, batch_revision, geometry = (
            batch.integration_snapshot(),
            batch.checkpoint_revision,
            scene.last_geometry,
        )
        owner = batch._buckets[batch.layout[KEY].bucket]
        capsule, revision = owner.checkpoint(), owner.checkpoint_revision
        state = dict(capsule._read(capsule.binding))
        saved_world = dict(state["worlds"][0])
        receipt = saved_world["receipt"]
        raw = dict(receipt.raw_contact_fields)
        native = dict(raw["native_contacts"])
        contact_j = native["contact_J"].copy()
        contact_j[0, 0, column] += np.float32(0.125)
        native["contact_J"] = contact_j
        (contact,) = receipt.contacts
        changed_j = contact.jacobian.copy()
        changed_j[0, column] = float(contact_j[0, 0, column])
        # Change both stored representations coherently. Raw/decoded equality
        # alone cannot reject this; the saved input body kinematics must do so.
        raw["native_contacts"] = MappingProxyType(native)
        saved_world["receipt"] = replace(
            receipt,
            contacts=(replace(contact, jacobian=changed_j),),
            raw_contact_fields=MappingProxyType(raw),
        )
        state["worlds"] = (saved_world,)
        forged = codec._make_checkpoint(capsule.binding, capsule.source_revision, state)
        monkeypatch.setattr(
            owner, "_build_device", lambda *a, **kw: pytest.fail("restore touched device")
        )
        with pytest.raises(ValueError, match="point Jacobian"):
            owner.restore(forged, expected_revision=revision)
        assert owner.checkpoint_revision == revision and batch.checkpoint_revision == batch_revision
        assert owner.checkpoint().to_bytes() == capsule.to_bytes()
        assert scene.last_geometry is geometry
        _same(batch.integration_snapshot(), before)
