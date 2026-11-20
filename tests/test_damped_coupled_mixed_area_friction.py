"""Two damped serial slides share distal ground and proximal live side friction.

Each body is 1 kg. Axes u=(.6,0,-.8), w=(1,0,0) give the hand-derived mass
[[2,.6],[.6,1]]. Area force acts only on the parent, yet changes the child's
motion through this mass and the common native solve. This is a two-slide
fixture, not a claim about arbitrary articulated robots.

Only the production CUDA owner steps. Native forward supplies input M/J/R/aref;
an independent checked two-variable Newton solve supplies expected velocities.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from test_damped_hinge_mixed_area_friction import IDENTITY, _side_asset
from test_mujoco_area_friction_lifecycle import _evidence, _same
from test_mujoco_area_friction_live import _surface as _audit_surface
from test_mujoco_tyx_area_pressure import _xml

from phymeo.backends.area_friction_readback import AreaFrictionEntryReader
from phymeo.backends.mujoco_area_friction import AreaFrictionPairLaw
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_heterogeneous_area_scene import (
    MuJoCoTyxHeterogeneousAreaPressureScene,
    MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
)
from phymeo.backends.mujoco_runtime import MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.geometry.area_pressure import (
    AreaPressureStamp,
    AreaPressureSurface,
    AreaPressureTriangle,
)
from phymeo.solvers.area_friction import LAW
from phymeo.training.resident_stepper import PhysicsStepFailure

H = 0.002
U = np.array([0.6, 0.0, -0.8])
W = np.array([1.0, 0.0, 0.0])
MASS = np.array([[2.0, 0.6], [0.6, 1.0]])
DAMPING = np.array([3.0, 7.0])
V0 = np.array([0.12, 0.08])
SIDE = np.array([[0.06, -0.05, 0.15], [0.06, 0.05, 0.15], [0.06, 0.0, 0.25]])
SCHEMA = "mujoco-tyx-area-friction-coupled-contact-v1"


def _declaration(mu=0.5, key=41, *, capacity=64, ground=True):
    law = 'condim="3" friction=".7 .01 .001"'
    # Preserve the established floor and foot location. Only the dynamic tree
    # changes: the sphere is on the child, the pressure asset on the parent.
    foot_z = "-.052" if ground else ".15"
    body = f"""<body name="compliant" pos="0 0 .1">
      <joint name="parent-slide" type="slide" axis=".6 0 -.8" damping="3"/>
      <inertial mass="1" pos="0 0 0" diaginertia=".02 .03 .04"/>
      <body name="distal">
        <joint name="child-slide" type="slide" axis="1 0 0" damping="7"/>
        <inertial mass="1" pos="0 0 0" diaginertia=".02 .03 .04"/>
        <geom name="foot" type="sphere" size=".05" pos=".12 0 {foot_z}" {law}/>
      </body>
    </body><body name="rigid"><geom type="box" size=".01 .06 .05"
      pos="-.01 0 .2" contype="0" conaffinity="0"/></body>"""
    xml = _xml(body, static=f'<geom name="floor" type="plane" size="2 2 .1" {law}/>')
    spec = MuJoCoTyxModelSpec(
        f"coupled-mixed-{mu}-{ground}",
        compile_mujoco_xml(xml, require_asset=False),
        contact_capacity=1,
        constraint_capacity=capacity,
        area_pressure_pairs=(("compliant", "rigid"),),
        area_friction_laws=(AreaFrictionPairLaw(IDENTITY, mu, 0.001),),
    )
    world = MuJoCoTyxWorldSpec(key, spec.key, MuJoCoInitialState((0.0, 0.0), V0))
    return spec, world


@pytest.mark.parametrize(
    "restriction",
    [
        "normal-only",
        "overlapping-second-limit",
        "two-limits-undamped",
        "limit-with-dry-friction",
        "condim-1",
        "three-supports",
        "second-support-condim-1",
        "two-supports-with-limit",
    ],
)
def test_damped_coupled_contact_admission_limits(restriction):
    import mujoco as mj

    from phymeo.backends.mujoco_area_pressure import compile_area_plan

    spec, _ = _declaration()
    xml = spec.compilation.source.primary_xml.decode()
    laws = spec.area_friction_laws
    expected = "damped area route cannot mix native constraints"
    if restriction == "normal-only":
        laws = ()
    elif restriction in (
        "overlapping-second-limit",
        "two-limits-undamped",
        "limit-with-dry-friction",
    ):
        child_margin = ".21" if restriction == "overlapping-second-limit" else "0"
        xml = xml.replace(
            'name="child-slide"',
            f'name="child-slide" limited="true" range="-.2 .2" margin="{child_margin}"',
        )
        attribute = (
            'frictionloss=".1"'
            if restriction == "limit-with-dry-friction"
            else 'limited="true" range="-.2 .2"'
        )
        xml = xml.replace('name="parent-slide"', f'name="parent-slide" {attribute}')
        if restriction == "overlapping-second-limit":
            expected = "nonoverlapping margins"
        elif restriction == "two-limits-undamped":
            xml = xml.replace('damping="3"', 'damping="0"').replace('damping="7"', 'damping="0"')
            expected = "area-only constraints or supported scalar joint rows"
    elif restriction == "condim-1":
        xml = xml.replace('condim="3"', 'condim="1"')
        expected = "native condim-3 plane/sphere"
    else:
        foot = '<geom name="foot"'
        if restriction == "second-support-condim-1":
            other = '<geom name="second-foot" type="sphere" size=".03" priority="2" condim="1"/>'
        else:
            other = '<geom name="second-foot" type="sphere" size=".03" condim="3"/>'
        if restriction == "three-supports":
            other += '<geom name="third-foot" type="sphere" size=".03" condim="3"/>'
        if restriction == "two-supports-with-limit":
            xml = xml.replace(
                'name="child-slide"', 'name="child-slide" limited="true" range="-.1 .1"'
            )
        xml = xml.replace(foot, other + foot)
        expected = "native condim-3 plane/sphere"
    model = mj.MjModel.from_xml_string(xml)
    with pytest.raises(ValueError, match=expected):
        compile_area_plan(mj, model, spec.area_pressure_pairs, laws)


def test_damped_ground_and_area_compile_two_supports():
    import mujoco as mj

    from phymeo.backends.mujoco_area_pressure import compile_area_plan

    spec, _ = _declaration()
    xml = spec.compilation.source.primary_xml.decode().replace(
        '<geom name="foot"', '<geom name="heel" type="sphere" size=".03"/><geom name="foot"'
    )
    model = mj.MjModel.from_xml_string(xml)
    plan = compile_area_plan(mj, model, spec.area_pressure_pairs, spec.area_friction_laws)
    assert plan.coupled_contact_variant == "two-supports-v1"
    assert plan.npair == plan.nslot == 2
    floor, heel, foot = (model.geom(name).id for name in ("floor", "heel", "foot"))
    np.testing.assert_array_equal(plan.slot_ids[:, 2:], [(floor, heel), (floor, foot)])


def test_damped_ground_and_area_compile_each_joint_limit():
    import mujoco as mj

    from phymeo.backends.mujoco_area_pressure import compile_area_plan

    spec, _ = _declaration()
    xml = spec.compilation.source.primary_xml.decode()
    for name, interval in (("parent-slide", "-.25 .5"), ("child-slide", "-.75 1")):
        xml = xml.replace(f'name="{name}"', f'name="{name}" limited="true" range="{interval}"')
    model = mj.MjModel.from_xml_string(xml)
    plan = compile_area_plan(mj, model, spec.area_pressure_pairs, spec.area_friction_laws)
    assert plan.coupled_damped_contacts and plan.nlimit == 2
    np.testing.assert_array_equal(plan.limit_ids, ((0, 0, 0), (1, 1, 1)))
    np.testing.assert_array_equal(plan.limit_law[:, :2], ((-0.25, 0.5), (-0.75, 1.0)))


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
    np.testing.assert_array_equal(model.dof_damping, DAMPING)
    assert not np.any(model.jnt_limited) and not np.any(model.dof_frictionloss)
    mass = np.empty((2, 2))
    mj.mj_fullM(model, mass, data.qM)
    # T = |u*v0|^2/2 + |u*v0+w*v1|^2/2, so the off-diagonal is u.w=.6.
    np.testing.assert_allclose(mass, MASS, rtol=0, atol=1e-13)
    np.testing.assert_allclose(data.qfrc_bias, 0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(data.qfrc_passive, -DAMPING * before.v, rtol=0, atol=1e-12)
    assert data.ncon in (0, 1) and data.nefc == 4 * data.ncon
    if data.ncon:
        contact = data.contact[0]
        assert contact.dim == 3
        assert (int(contact.geom1), int(contact.geom2)) == (
            model.geom("floor").id,
            model.geom("foot").id,
        )
    return model, data, mass


def _value(velocity, p):
    """Original physical native law and final-velocity area law, independently."""
    mass, response, h = p["mass"], p["response"], p["h"]
    acceleration = np.linalg.solve(mass, response @ (velocity - p["v0"])) / h
    native = np.maximum(0.0, (p["aref"] - p["native_j"] @ acceleration) / p["reg"])
    relative = np.einsum("sij,j->si", p["point_j"], velocity)
    n = p["normal"]
    vn = np.sum(n * relative, axis=1)
    vt = relative - n * (vn / np.sum(n * n, axis=1))[:, None]
    fn = np.maximum(0.0, -p["stiffness"] * (p["gap"] + h * vn))
    ft = -p["mu"] * fn[:, None] * vt / np.sqrt(np.sum(vt * vt, axis=1) + p["slip"] ** 2)[:, None]
    force = fn[:, None] * n + ft
    area_q = np.einsum("sij,si->j", p["point_j"], force)
    native_q = p["native_j"].T @ native
    residual = response @ velocity - p["rhs"] - h * (area_q + native_q)
    return dict(
        residual=residual,
        acceleration=acceleration,
        native=native,
        normal=fn,
        tangent=ft,
        force=force,
        relative=relative,
        tangent_velocity=vt,
        area_q=area_q,
        native_q=native_q,
    )


def _root(p):
    """Checked 2x2 central-difference Newton, seeded without accepted output."""
    velocity = np.linalg.solve(p["response"], p["rhs"])
    factor = np.linalg.cholesky(p["response"])
    scale = 1 + np.linalg.norm(np.linalg.solve(factor, p["rhs"]))

    def norm(value):
        return float(np.linalg.norm(np.linalg.solve(factor, value)))

    for iteration in range(80):
        result = _value(velocity, p)
        residual = result["residual"]
        error = norm(residual)
        if error <= 1e-11 * scale:
            return velocity, result, iteration
        derivative = np.empty((2, 2))
        for j in range(2):
            delta = np.zeros(2)
            delta[j] = 1e-6 * max(1.0, abs(velocity[j]))
            derivative[:, j] = (
                _value(velocity + delta, p)["residual"] - _value(velocity - delta, p)["residual"]
            ) / (2 * delta[j])
        direction = np.linalg.solve(derivative, -residual)
        assert np.isfinite(direction).all()
        for backtrack in range(24):
            candidate = velocity + direction * 0.5**backtrack
            if norm(_value(candidate, p)["residual"]) < error:
                velocity = candidate
                break
        else:
            raise AssertionError(("independent Newton line search", velocity, residual))
    raise AssertionError(("independent Newton convergence", velocity, _value(velocity, p)))


def _audit(spec, before, after, geometry):
    model, data, source_mass = _native(spec, before)
    receipt = after.last_step
    assert receipt.tick == after.tick == before.tick + 1
    np.testing.assert_array_equal(receipt.input_q, before.q)
    np.testing.assert_array_equal(receipt.input_v, before.v)
    assert receipt.input_time == before.time == receipt.contact_time == receipt.dynamics.time
    assert receipt.contact_phase == "tyx-euler-input-constraint-evaluation"
    assert receipt.dynamics.phase == "tyx-euler-input-dynamics"
    fields = receipt.raw_contact_fields
    assert fields["native_contact_operator"] == "M"
    assert fields["response_operator"] == "M+h*joint_damping"
    assert "area_pressure" not in fields
    (pair,) = fields["area_friction"]
    solver = fields["area_friction_solver"]
    assert solver["schema"] == SCHEMA and solver["law"] == pair["law"] == LAW
    assert solver["solver_status"] == 0 and solver["precision"] == "actual-packed-point-fp32"
    _audit_surface(pair, geometry, receipt)
    assert len(receipt.contacts) == data.ncon and len(receipt.constraints) == data.nefc
    native_j = np.asarray([row.jacobian for row in receipt.constraints]).reshape(-1, 2)
    reg = np.array([row.regularization for row in receipt.constraints])
    aref = np.array([row.reference_acceleration for row in receipt.constraints])
    native_force = np.array([row.force for row in receipt.constraints])
    for i, row in enumerate(receipt.constraints):
        assert row.constraint_type == int(data.efc_type[i])
        assert row.source_id == int(data.efc_id[i])
    for actual, expected in (
        (native_j, data.efc_J.reshape(-1, 2)),
        (reg, data.efc_R),
        (aref, data.efc_aref),
        (solver["mass_matrix"], source_mass),
        (solver["damping"], DAMPING),
        (receipt.dynamics.bias, data.qfrc_bias),
        (receipt.dynamics.passive, data.qfrc_passive),
        (receipt.dynamics.applied, data.qfrc_applied + data.qfrc_actuator),
    ):
        np.testing.assert_allclose(actual, expected, rtol=4e-4, atol=2e-5)
    assert np.all(reg > 0)
    np.testing.assert_array_equal(solver["input_velocity"], before.v)
    np.testing.assert_array_equal(solver["output_velocity"], after.v)
    h = solver["h_s"]
    assert h == float(np.float32(H))
    mass = solver["mass_matrix"].astype(float)
    response = mass + h * np.diag(solver["damping"].astype(float))
    dyn = receipt.dynamics
    applied = solver["applied_force"].astype(float)
    np.testing.assert_allclose(
        applied,
        dyn.applied + dyn.passive - dyn.bias + DAMPING * before.v,
        rtol=2e-5,
        atol=2e-5,
    )
    # Proximal material points translate with u*q0 only. A child joint cannot
    # enter this Jacobian even though its velocity is affected dynamically.
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
        native_j=native_j,
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
    np.testing.assert_allclose(pair["relative_velocity"], actual["relative"], atol=2e-5, rtol=2e-5)
    force = pair["force_on_a"].astype(float)
    ft = pair["tangential_force"].astype(float)
    fn = pair["normal_force"].astype(float)
    points = pair["points"].astype(float)
    np.testing.assert_array_equal(pair["force_on_b"], -pair["force_on_a"])
    np.testing.assert_array_equal(
        pair["torque_on_b_world_origin"], -pair["torque_on_a_world_origin"]
    )
    np.testing.assert_allclose(
        pair["torque_on_a_world_origin"], np.cross(points, force), rtol=0.002, atol=1e-5
    )
    assert np.all(fn >= 0)
    assert np.all(np.linalg.norm(ft, axis=1) <= pair["mu"] * fn + 1e-5)
    assert np.all(np.sum(ft * actual["tangent_velocity"], axis=1) <= 1e-14)
    np.testing.assert_allclose(np.sum(ft * p["normal"], axis=1), 0, rtol=0, atol=2e-6)
    area_q = np.array([U @ force.sum(axis=0), 0.0])
    area_tangent_q = np.array([U @ ft.sum(axis=0), 0.0])
    assert area_tangent_q @ after.v <= 1e-12
    ground_force = np.zeros(3)
    ground_q = np.zeros(2)
    total_moment = pair["torque_on_a_world_origin"].sum(axis=0) + pair[
        "torque_on_b_world_origin"
    ].sum(axis=0)
    if data.ncon:
        source = data.contact[0]
        contact = receipt.contacts[0]
        assert contact.dimension == 3
        local = np.array(
            [
                native_force.sum(),
                source.friction[0] * (native_force[0] - native_force[1]),
                source.friction[1] * (native_force[2] - native_force[3]),
            ]
        )
        ground_force = source.frame.reshape(3, 3).T @ local
        np.testing.assert_allclose(
            contact.force_on_geometry2_world, ground_force, rtol=0.002, atol=0.002
        )
        np.testing.assert_array_equal(
            contact.force_on_geometry1_world, -contact.force_on_geometry2_world
        )
        body1, body2 = model.geom_bodyid[[source.geom1, source.geom2]]
        for wrench, body, signed_force in (
            (contact.wrench_on_body1_world, body1, -ground_force),
            (contact.wrench_on_body2_world, body2, ground_force),
        ):
            moment = np.cross(source.pos - data.xipos[body], signed_force)
            np.testing.assert_allclose(wrench, np.r_[signed_force, moment], rtol=0.002, atol=0.002)
            total_moment = total_moment + wrench[3:] + np.cross(data.xipos[body], wrench[:3])
        ground_q = np.array([U @ ground_force, W @ ground_force])
        np.testing.assert_allclose(ground_q, native_j.T @ native_force, rtol=2e-5, atol=2e-5)
    # Every contact action/reaction pair has zero total moment about the same
    # world origin. Guide and damping reactions are not asserted to vanish.
    np.testing.assert_allclose(total_moment, 0, rtol=0, atol=2e-5)
    total_q = area_q + ground_q
    np.testing.assert_allclose(receipt.qfrc_constraint, total_q, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(receipt.qfrc_constraint, solver["generalized_force"])
    generalized_power = float(total_q @ after.v)
    cartesian_power = float(
        np.sum(force * actual["relative"]) + ground_force @ (U * after.v[0] + W * after.v[1])
    )
    assert generalized_power == pytest.approx(cartesian_power, abs=2e-5, rel=2e-5)
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
    gradients = native_j @ receipt.qacc + reg * native_force - aref
    np.testing.assert_allclose(
        [row.gradient for row in receipt.constraints], gradients, rtol=2e-5, atol=2e-5
    )
    assert np.all(gradients >= -2e-5 * (1 + np.abs(aref)))
    return dict(
        area_q=area_q,
        area_tangent_q=area_tangent_q,
        ground_q=ground_q,
        ground_force=ground_force,
        native_force=native_force,
        expected_v=expected_v,
        oracle=expected,
        newton_iterations=iterations,
        pair=pair,
        physical_acceleration=acceleration,
        integration_acceleration=integration,
        generalized_power=generalized_power,
        cartesian_power=cartesian_power,
    )


def _observe(ws, batch, specs, states):
    frame = ws.refresh(expected_revision=batch.checkpoint_revision)
    arrays = ws.readback(frame)
    decoded = AreaFrictionEntryReader(ws, {spec.key: (IDENTITY,) for spec in specs}).validate(
        arrays, frame
    )
    for key, state in states.items():
        (pair,) = state.last_step.raw_contact_fields["area_friction"]
        samples = decoded[key].samples
        assert len(samples) == len(pair["sample_ids"])
        assert all(
            sample.identity == IDENTITY and sample.stamp == pair["stamp"] for sample in samples
        )
        for values, expected in (
            ([sample.normal_force_n for sample in samples], pair["normal_force"]),
            ([sample.tangential_force_on_a_world for sample in samples], pair["tangential_force"]),
            ([sample.relative_velocity_world for sample in samples], pair["relative_velocity"]),
        ):
            np.testing.assert_array_equal(np.asarray(values).reshape(expected.shape), expected)
    return frame, arrays


@pytest.mark.cuda
def test_damped_two_slide_ground_and_proximal_friction_couple_three_steps_and_replay(tmp_path):
    # A diagonal response, distal application of parent area force, independent
    # native/area solves, or damping applied twice must violate this oracle.
    declarations = [_declaration(mu, key) for mu, key in ((0.5, 41), (0.0, 7))]
    specs, worlds = tuple(item[0] for item in declarations), tuple(item[1] for item in declarations)
    with _evidence(tmp_path / "coupled-three-step-replay.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda") as batch:
            assets = {spec.key: (_side_asset(),) for spec in specs}
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
            with batch.prepare_device_workspace(
                include_bodies=True, include_area_pressure=True
            ) as ws:

                def advance():
                    before = batch.integration_snapshot()
                    assert scene.step(H, expected_revision=batch.checkpoint_revision) == {
                        41: True,
                        7: True,
                    }
                    after, geometry = batch.integration_snapshot(), scene.last_geometry
                    record = dict(before=before, after=after, geometry=geometry, audits={})
                    evidence["records"].append(record)
                    for spec, world in declarations:
                        key = world.world
                        audit = record["audits"][key] = _audit(
                            spec, before[key], after[key], geometry[key][0]
                        )
                        assert audit["ground_force"][2] > 0.1
                        assert audit["pair"]["normal_force"].sum() > 0.1
                        assert abs(audit["area_q"][0]) > 0.001
                        assert audit["area_q"][1] == 0
                        assert abs(audit["ground_q"][1]) > 0.001
                        if key == 41:
                            assert np.linalg.norm(audit["pair"]["tangential_force"]) > 0.001
                            assert audit["area_tangent_q"][0] * after[key].v[0] < 0
                        else:
                            np.testing.assert_array_equal(audit["pair"]["tangential_force"], 0)
                    frame, arrays = _observe(ws, batch, specs, after)
                    record["arrays"] = arrays
                    return after, geometry, frame, arrays, record["audits"]

                first = advance()
                capsule = MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(
                    scene.checkpoint().to_bytes()
                )
                tail = [advance(), advance()]
                history = [first, *tail]
                # Both worlds start with identical q/v and native laws. Only mu
                # changes; parent friction must reach the distal generalized DOF.
                assert max(abs(item[0][41].v[1] - item[0][7].v[1]) for item in history) > 1e-6
                assert (
                    max(
                        np.linalg.norm(item[4][41]["native_force"] - item[4][7]["native_force"])
                        for item in history
                    )
                    > 1e-5
                )
                for key in (41, 7):
                    assert not np.array_equal(first[0][key].q, tail[-1][0][key].q)
                    assert not np.array_equal(
                        first[4][key]["pair"]["normal_force"],
                        tail[-1][4][key]["pair"]["normal_force"],
                    )
                scene.restore(capsule, expected_revision=batch.checkpoint_revision)
                _same(batch.integration_snapshot(), first[0])
                with pytest.raises(ValueError):
                    ws.readback(tail[-1][2])
                frame = ws.refresh(expected_revision=batch.checkpoint_revision)
                _same(ws.readback(frame), first[3])
                for expected in tail:
                    replay = advance()
                    _same(replay[0], expected[0], fresh_stamp=True)
                    _same(replay[1], expected[1], fresh_stamp=True)
                    _same(replay[3], expected[3])


def _prescribed(batch, key, *, empty=False, copies=1):
    state = batch.integration_snapshot()[key]
    stamp = AreaPressureStamp(
        str(key), state.tick, state.time, batch.area_pressure_configuration_id(key)
    )
    triangles = (
        ()
        if empty
        else tuple(
            AreaPressureTriangle(
                f"side-{i}",
                "cell",
                f"face-{i}",
                SIDE,
                np.full(3, 2000.0),
                np.array([2e6, 0.0, 0.0]),
            )
            for i in range(copies)
        )
    )
    return AreaPressureSurface(IDENTITY, stamp, triangles)


def _prescribed_step(batch, *, empty=False):
    before = batch.integration_snapshot()
    surfaces = {key: (_prescribed(batch, key, empty=empty),) for key in batch.world_keys}
    assert batch.step_area_pressure(
        H, surfaces, expected_revision=batch.checkpoint_revision
    ) == dict.fromkeys(batch.world_keys, True)
    return before, batch.integration_snapshot(), surfaces


@pytest.mark.cuda
@pytest.mark.parametrize(
    "ground,empty",
    [(True, True), (False, False), (False, True)],
    ids=["native-only", "area-only", "free-damped"],
)
def test_coupled_mixed_route_preserves_absent_ground_and_empty_area_limits(ground, empty):
    spec, world = _declaration(ground=ground)
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        before, after, surfaces = _prescribed_step(batch, empty=empty)
        audit = _audit(spec, before[41], after[41], SimpleNamespace(surface=surfaces[41][0]))
        assert len(after[41].last_step.contacts) == int(ground)
        if empty:
            assert audit["pair"]["sample_ids"] == ()
            np.testing.assert_array_equal(audit["area_q"], 0)
        else:
            assert audit["pair"]["normal_force"].sum() > 0.1
            assert np.linalg.norm(audit["pair"]["tangential_force"]) > 0.001
        if not ground:
            np.testing.assert_array_equal(audit["ground_q"], 0)
        if empty and not ground:
            expected = np.linalg.solve(MASS + float(np.float32(H)) * np.diag(DAMPING), MASS @ V0)
            np.testing.assert_allclose(after[41].v, expected, rtol=2e-6, atol=2e-7)


@pytest.mark.cuda
def test_coupled_native_and_area_capacity_failure_preserves_both_dofs_and_retry(tmp_path):
    spec, world = _declaration(capacity=7)
    with _evidence(tmp_path / "coupled-capacity.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            with batch.prepare_device_workspace(
                include_bodies=True, include_area_pressure=True
            ) as ws:
                before, accepted, surfaces = _prescribed_step(batch)
                _audit(spec, before[41], accepted[41], SimpleNamespace(surface=surfaces[41][0]))
                frame, arrays = _observe(ws, batch, (spec,), accepted)
                revision, publication = batch.checkpoint_revision, batch.last_step
                checkpoint = batch.checkpoint().to_bytes()
                oversized = _prescribed(batch, 41, copies=2)
                evidence.update(before=accepted, arrays=arrays, oversized=oversized)
                with pytest.raises((ValueError, PhysicsStepFailure)):
                    batch.step_area_pressure(H, {41: (oversized,)}, expected_revision=revision)
                assert batch.checkpoint_revision == revision and batch.last_step is publication
                assert batch.checkpoint().to_bytes() == checkpoint
                _same(batch.integration_snapshot(), accepted)
                _same(ws.readback(frame), arrays)
                before, after, surfaces = _prescribed_step(batch)
                evidence["retry"] = after
                _audit(spec, before[41], after[41], SimpleNamespace(surface=surfaces[41][0]))
                assert after[41].tick == accepted[41].tick + 1
