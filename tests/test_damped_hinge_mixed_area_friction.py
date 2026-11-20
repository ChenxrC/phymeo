"""One damped hinge shares actual native ground and live regularized side loads.

Only the production CUDA owner advances physics. MuJoCo forward supplies the
native input coefficients; a scalar bisection independently solves the mixed
constitutive equation.
"""

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from test_mujoco_area_friction_lifecycle import _evidence, _same
from test_mujoco_area_friction_live import _surface as _audit_surface
from test_mujoco_tyx_damped_hinge_support import DAMPING, MASS_MOMENT, PIVOT, _hinge_xml

from phymeo.backends.area_friction_readback import AreaFrictionEntryReader
from phymeo.backends.mujoco_area_friction import AreaFrictionPairLaw
from phymeo.backends.mujoco_area_pressure_scene import BodyAreaPressureAsset
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_heterogeneous_area_scene import (
    MuJoCoTyxHeterogeneousAreaPressureScene,
    MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.geometry.area_pressure import (
    AreaPressureIdentity,
    AreaPressureStamp,
    AreaPressureSurface,
    AreaPressureTriangle,
)
from phymeo.geometry.area_pressure_intersection import RigidPressureMesh, TetraPressureMesh
from phymeo.solvers.area_friction import LAW
from phymeo.training.resident_stepper import PhysicsStepFailure

H = 0.002
KEY = 11
IDENTITY = AreaPressureIdentity("compliant", "rigid", "hinge-elastic", "side-plate")
VERTICES = np.array([[0.06, -0.05, 0.15], [0.06, 0.05, 0.15], [0.06, 0, 0.25]])
GRADIENT = 2e6
SLIP = 0.001


def _case(*, velocity=0.35, mu=0.5, capacity=64, xml=None):
    spec = MuJoCoTyxModelSpec(
        "damped-hinge-mixed",
        compile_mujoco_xml(_hinge_xml() if xml is None else xml, require_asset=False),
        contact_capacity=1,
        constraint_capacity=capacity,
        area_pressure_pairs=(("compliant", "rigid"),),
        area_friction_laws=(AreaFrictionPairLaw(IDENTITY, mu, SLIP),),
    )
    world = MuJoCoTyxWorldSpec(KEY, spec.key, MuJoCoInitialState((0.0,), (velocity,)))
    return spec, world


def _surface(batch, *, depth=0.001, empty=False, copies=1, vertices=VERTICES):
    state = batch.integration_snapshot()[KEY]
    stamp = AreaPressureStamp(
        str(KEY), state.tick, state.time, batch.area_pressure_configuration_id(KEY)
    )
    triangles = (
        ()
        if empty
        else tuple(
            AreaPressureTriangle(
                f"side-{i}",
                "cell",
                f"face-{i}",
                vertices,
                np.full(3, GRADIENT * depth),
                np.array([GRADIENT, 0.0, 0.0]),
            )
            for i in range(copies)
        )
    )
    return AreaPressureSurface(IDENTITY, stamp, triangles)


def _advance(batch, **surface_options):
    before = batch.integration_snapshot()[KEY]
    surface = _surface(batch, **surface_options)
    assert batch.step_area_pressure(
        H, {KEY: (surface,)}, expected_revision=batch.checkpoint_revision
    ) == {KEY: True}
    return before, batch.integration_snapshot()[KEY], SimpleNamespace(surface=surface)


def _native(spec, before):
    import mujoco as mj

    model = spec.compilation.fresh_native_model()
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = before.q, before.v, before.time
    control = before.held_control
    data.ctrl[:] = control.ctrl
    if control.qfrc_applied is not None:
        data.qfrc_applied[:] = control.qfrc_applied
    if control.xfrc_applied is not None:
        data.xfrc_applied[:] = control.xfrc_applied
    mj.mj_forward(model, data)
    assert model.nq == model.nv == 1
    assert data.ncon == 1 and data.nefc == 4 and data.contact[0].dim == 3
    contact = data.contact[0]
    assert {int(contact.geom1), int(contact.geom2)} == {
        model.geom("floor").id,
        model.geom("foot").id,
    }
    mass = np.empty((1, 1))
    mj.mj_fullM(model, mass, data.qM)
    assert mass[0, 0] == pytest.approx(MASS_MOMENT, abs=1e-12)
    assert model.dof_damping[0] == DAMPING
    return model, data, mass


def _drive_samples(*, depth=0.001, empty=False):
    """Independent literal-triangle samples used only to select motor torque."""
    bary = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
    triangles = []
    if not empty:
        triangles.append((VERTICES, np.full(3, GRADIENT * depth), GRADIENT))
    points, stiffness, gap = [], [], []
    for triangle, pressure, normal_gradient in triangles:
        area = np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) / 2
        if area == 0:
            continue
        points.extend(bary @ triangle)
        stiffness.extend([area * normal_gradient / 3] * 3)
        gap.extend(-(bary @ pressure) / normal_gradient)
    # Surface packing is a public FP32 boundary. These are independently
    # derived inputs, not a readback of accepted forces or candidate velocity.
    return (
        np.asarray(points, dtype=np.float32).astype(float).reshape(-1, 3),
        np.asarray(stiffness, dtype=np.float32).astype(float),
        np.asarray(gap, dtype=np.float32).astype(float),
    )


def _drive_velocity(batch, spec, target, *, depth=0.001, empty=False):
    """Apply finite inverse-designed torque, leaving all state evolution native.

    A stationary compliant floor can oppose much more than an arbitrary fixed
    motor torque. Specify the desired endpoint and compute the external motor
    load from source entry J/R/aref and the prescribed side law. The subsequent
    audit independently solves the complete mixed equation by bisection.
    """
    before = batch.integration_snapshot()[KEY]
    model, data, matrix = _native(spec, before)
    h = float(np.float32(H))
    mass, damping = float(matrix[0, 0]), float(model.dof_damping[0])
    v0 = float(before.v[0])
    acceleration = (mass + h * damping) * (target - v0) / (h * mass)
    jac = data.efc_J.reshape(4).copy()
    native = np.maximum(0.0, (data.efc_aref - jac * acceleration) / data.efc_R)
    points, stiffness, gap = _drive_samples(depth=depth, empty=empty)
    point_jac = np.cross([0.0, 1.0, 0.0], points - PIVOT)
    relative = target * point_jac
    normal = np.maximum(0.0, -stiffness * (gap + h * relative[:, 0]))
    tangent_velocity = relative.copy()
    tangent_velocity[:, 0] = 0
    law = spec.area_friction_laws[0]
    mu, slip = float(np.float32(law.mu)), float(np.float32(law.slip_speed_m_s))
    tangent = (
        -mu
        * normal[:, None]
        * tangent_velocity
        / np.sqrt(np.sum(tangent_velocity**2, axis=1) + slip**2)[:, None]
    )
    force = tangent.copy()
    force[:, 0] += normal
    area_torque = float(np.sum(point_jac * force))
    torque = (
        mass * acceleration
        - float(data.qfrc_passive[0] - data.qfrc_bias[0])
        - float(jac @ native)
        - area_torque
    )
    assert np.isfinite(torque)
    batch.set_controls({KEY: MuJoCoControl((), qfrc_applied=(torque,))})
    return dict(
        target_velocity=target,
        torque=torque,
        target_acceleration=acceleration,
        target_native_force=native,
        target_area_force=force,
        target_area_torque=area_torque,
        source_q=before.q,
        source_v=before.v,
    )


def _scalar(
    velocity,
    *,
    mass,
    damping,
    v0,
    applied,
    h,
    jac,
    reg,
    aref,
    point_jac,
    normals,
    stiffness,
    gap,
    mu,
    slip,
):
    # Native R/aref acts on physical acceleration M^-1 f; the area law acts
    # on the accepted velocity after implicit Euler damping integration.
    acceleration = (mass + h * damping) * (velocity - v0) / (h * mass)
    native = np.maximum(0.0, (aref - jac * acceleration) / reg)
    relative = point_jac * velocity
    vn = np.einsum("ij,ij->i", normals, relative)
    tangent_velocity = relative - normals * (vn / np.sum(normals * normals, axis=1))[:, None]
    normal = np.maximum(0.0, -stiffness * (gap + h * vn))
    tangent = (
        -mu
        * normal[:, None]
        * tangent_velocity
        / np.sqrt(np.sum(tangent_velocity * tangent_velocity, axis=1) + slip**2)[:, None]
    )
    force = normal[:, None] * normals + tangent
    area_torque = float(np.sum(point_jac * force))
    total = float(jac @ native) + area_torque
    residual = (mass + h * damping) * velocity - mass * v0 - h * (applied + total)
    return dict(
        residual=residual,
        native=native,
        normal=normal,
        tangent=tangent,
        force=force,
        relative=relative,
        tangent_velocity=tangent_velocity,
        area_torque=area_torque,
        total=total,
        acceleration=acceleration,
    )


def _root(parameters):
    lo, hi = -100.0, 100.0
    assert _scalar(lo, **parameters)["residual"] < 0
    assert _scalar(hi, **parameters)["residual"] > 0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _scalar(mid, **parameters)["residual"] < 0:
            lo = mid
        else:
            hi = mid
    velocity = (lo + hi) / 2
    return velocity, _scalar(velocity, **parameters)


def _audit(spec, before, after, intersection):
    model, data, source_mass = _native(spec, before)
    receipt = after.last_step
    assert receipt.tick == before.tick + 1 == after.tick
    np.testing.assert_array_equal(receipt.input_q, before.q)
    np.testing.assert_array_equal(receipt.input_v, before.v)
    assert receipt.input_time == before.time
    assert receipt.contact_phase == "tyx-euler-input-constraint-evaluation"
    assert len(receipt.contacts) == 1 and receipt.contacts[0].dimension == 3
    assert len(receipt.constraints) == 4
    fields = receipt.raw_contact_fields
    assert "area_pressure" not in fields
    (pair,) = fields["area_friction"]
    solver = fields["area_friction_solver"]
    assert solver["schema"] == "mujoco-tyx-area-friction-scalar-contact-v1"
    assert solver["law"] == pair["law"] == LAW
    assert solver["precision"] == "actual-packed-point-fp32"
    assert solver["solver_status"] == 0
    _audit_surface(pair, intersection, receipt)

    rows = receipt.constraints
    jac = np.array([row.jacobian[0] for row in rows])
    reg = np.array([row.regularization for row in rows])
    aref = np.array([row.reference_acceleration for row in rows])
    native_force = np.array([row.force for row in rows])
    for actual, expected in (
        (jac, data.efc_J.reshape(4)),
        (reg, data.efc_R),
        (aref, data.efc_aref),
        (solver["mass_matrix"], source_mass),
        (solver["damping"], model.dof_damping),
        (receipt.dynamics.bias, data.qfrc_bias),
        (receipt.dynamics.passive, data.qfrc_passive),
        (receipt.dynamics.applied, data.qfrc_applied + data.qfrc_actuator),
    ):
        np.testing.assert_allclose(actual, expected, rtol=4e-4, atol=2e-5)
    for i, row in enumerate(rows):
        assert row.constraint_type == int(data.efc_type[i])
        assert row.source_id == int(data.efc_id[i])
    assert np.all(reg > 0)
    np.testing.assert_array_equal(solver["input_velocity"], before.v)
    np.testing.assert_array_equal(solver["output_velocity"], after.v)
    assert solver["h_s"] == float(np.float32(H))

    # The y axis is unchanged by rotation around itself; compute the material
    # point velocity directly from its world lever, not the published Jacobian.
    points = pair["points"].astype(float)
    axis = np.array([0.0, 1.0, 0.0])
    point_jac = np.cross(axis, points - PIVOT)
    np.testing.assert_allclose(pair["jacobian_a"][:, :, 0], point_jac, atol=2e-7)
    np.testing.assert_array_equal(pair["jacobian_b"], 0)
    normals = pair["normals"].astype(float)
    tangent_jac = (
        point_jac
        - normals
        * (np.sum(point_jac * normals, axis=1) / np.sum(normals * normals, axis=1))[:, None]
    )
    if len(points):
        assert np.min(np.linalg.norm(tangent_jac, axis=1)) > 0.04
    mass, damping = float(solver["mass_matrix"][0, 0]), float(solver["damping"][0])
    v0, h = float(before.v[0]), solver["h_s"]
    applied = float(solver["applied_force"][0])
    dyn = receipt.dynamics
    expected_applied = float(dyn.applied[0] + dyn.passive[0] - dyn.bias[0] + damping * v0)
    assert applied == pytest.approx(expected_applied, abs=2e-5, rel=2e-5)
    parameters = dict(
        mass=mass,
        damping=damping,
        v0=v0,
        applied=applied,
        h=h,
        jac=jac,
        reg=reg,
        aref=aref,
        point_jac=point_jac,
        normals=normals,
        stiffness=pair["stiffness"].astype(float),
        gap=pair["surrogate_gap"].astype(float),
        mu=pair["mu"],
        slip=pair["slip_speed_m_s"],
    )
    expected_v, expected = _root(parameters)
    velocity = float(after.v[0])
    assert velocity == pytest.approx(expected_v, abs=5e-5, rel=5e-5)
    actual = _scalar(velocity, **parameters)
    for values, independent in (
        (native_force, expected["native"]),
        (pair["normal_force"], expected["normal"]),
        (pair["tangential_force"], expected["tangent"]),
        (pair["force_on_a"], expected["force"]),
        (native_force, actual["native"]),
        (pair["normal_force"], actual["normal"]),
        (pair["tangential_force"], actual["tangent"]),
        (pair["force_on_a"], actual["force"]),
    ):
        np.testing.assert_allclose(values, independent, rtol=0.002, atol=0.002)
    np.testing.assert_allclose(pair["relative_velocity"], actual["relative"], atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(pair["force_on_b"], -pair["force_on_a"])
    np.testing.assert_array_equal(
        pair["torque_on_b_world_origin"], -pair["torque_on_a_world_origin"]
    )
    force = pair["force_on_a"].astype(float)
    ft = pair["tangential_force"].astype(float)
    fn = pair["normal_force"].astype(float)
    np.testing.assert_allclose(
        pair["torque_on_a_world_origin"], np.cross(points, force), atol=1e-5, rtol=0.002
    )
    assert np.all(fn >= 0)
    assert np.all(np.linalg.norm(ft, axis=1) <= pair["mu"] * fn + 1e-5)
    assert np.all(np.sum(ft * actual["tangent_velocity"], axis=1) <= 1e-14)
    tangent_torque = float(np.sum(point_jac * ft))
    assert tangent_torque * velocity <= 1e-12
    np.testing.assert_allclose(np.sum(ft * normals, axis=1), 0, atol=2e-6)

    # Independently decode the four pyramidal rows into the real ground wrench.
    contact = data.contact[0]
    local_force = np.array(
        [
            native_force.sum(),
            contact.friction[0] * (native_force[0] - native_force[1]),
            contact.friction[1] * (native_force[2] - native_force[3]),
        ]
    )
    world_force = contact.frame.reshape(3, 3).T @ local_force
    ground = receipt.contacts[0]
    np.testing.assert_allclose(ground.force_on_geometry2_world, world_force, rtol=0.002, atol=0.002)
    np.testing.assert_array_equal(ground.force_on_geometry1_world, -ground.force_on_geometry2_world)
    ground_torque = float(np.cross(contact.pos - PIVOT, world_force)[1])
    assert ground_torque == pytest.approx(jac @ native_force, abs=2e-5, rel=2e-5)
    area_torque = float(np.cross(points - PIVOT, force).sum(axis=0)[1])
    total = ground_torque + area_torque
    np.testing.assert_allclose(receipt.qfrc_constraint, [total], rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(receipt.qfrc_constraint, solver["generalized_force"])
    rhs = float(dyn.applied[0] + dyn.passive[0] - dyn.bias[0]) + total
    np.testing.assert_allclose(h * receipt.qacc, [h * rhs / mass], rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(
        h * receipt.integration_acceleration,
        [h * rhs / (mass + h * damping)],
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        after.v, before.v + h * receipt.integration_acceleration, rtol=2e-6, atol=2e-7
    )
    np.testing.assert_allclose(after.q, before.q + h * after.v, rtol=0, atol=2e-7)
    scale = 1 + abs((mass * v0 + h * applied) / (mass + h * damping)) * np.sqrt(mass + h * damping)
    stored_residual = (mass + h * damping) * velocity - mass * v0 - h * (applied + total)
    assert abs(stored_residual) / np.sqrt(mass + h * damping) <= 2e-5 * scale
    assert abs(actual["residual"]) / np.sqrt(mass + h * damping) <= 2e-5 * scale
    assert solver["normalized_residual"] <= solver["tolerance"] == 2e-5
    gradients = jac * float(receipt.qacc[0]) + reg * native_force - aref
    np.testing.assert_allclose([row.gradient for row in rows], gradients, rtol=2e-5, atol=2e-5)
    assert np.all(gradients >= -2e-5 * (1 + np.abs(aref)))
    return dict(
        velocity=velocity,
        oracle=expected,
        tangent_torque=tangent_torque,
        ground_force=world_force,
        area_torque=area_torque,
        pair=pair,
    )


@pytest.mark.cuda
@pytest.mark.parametrize("velocity", [0.35, -0.35], ids=["positive-slip", "negative-slip"])
@pytest.mark.parametrize("mu", [0.5, 0.0], ids=["friction", "normal-only-limit"])
def test_native_ground_and_side_friction_share_damped_hinge_velocity(velocity, mu, tmp_path):
    # Catches separate native/area solves, missing eccentric tangential torque,
    # duplicated damping, and native R/aref evaluated on integration acceleration.
    spec, world = _case(velocity=velocity, mu=mu)
    with _evidence(tmp_path / "mixed-scalar.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            before, after, geometry = _advance(batch)
            evidence.update(before=before, after=after, geometry=geometry.surface)
            audit = _audit(spec, before, after, geometry)
            evidence["audit"] = audit
            assert np.sum(audit["pair"]["normal_force"]) > 0.1
            assert audit["ground_force"][2] > 0.1
            assert abs(audit["area_torque"]) > 0.01
            assert audit["velocity"] * velocity > 0
            assert abs(after.last_step.qacc[0] - after.last_step.integration_acceleration[0]) > 0.1
            if mu:
                assert abs(audit["tangent_torque"]) > 0.001
                assert audit["tangent_torque"] * audit["velocity"] < 0
            else:
                np.testing.assert_array_equal(audit["pair"]["tangential_force"], 0)


@pytest.mark.cuda
def test_inclined_slide_native_ground_and_area_friction():
    import mujoco as mj

    axis = np.array([0.6, 0.0, -0.8])
    xml = _hinge_xml().replace('type="hinge" axis="0 1 0"', 'type="slide" axis=".6 0 -.8"')
    spec, world = _case(xml=xml)
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        before, after, geometry = _advance(batch)
        step = after.last_step
        (pair,) = step.raw_contact_fields["area_friction"]
        _audit_surface(pair, geometry, step)
        model = spec.compilation.fresh_native_model()
        cpu = mj.MjData(model)
        cpu.qvel[:] = before.v
        mj.mj_forward(model, cpu)
        assert cpu.nefc == 4
        points, stiffness, gap = _drive_samples()
        velocity, reference = _root(
            dict(
                mass=0.6,
                damping=DAMPING,
                v0=float(before.v[0]),
                applied=0.0,
                h=float(np.float32(H)),
                jac=cpu.efc_J.reshape(4),
                reg=cpu.efc_R,
                aref=cpu.efc_aref,
                point_jac=np.tile(axis, (len(points), 1)),
                normals=np.tile([1.0, 0.0, 0.0], (len(points), 1)),
                stiffness=stiffness,
                gap=gap,
                mu=0.5,
                slip=float(np.float32(SLIP)),
            )
        )
        assert reference["normal"].sum() > 0 and reference["native"].sum() > 0
        assert np.sum(reference["tangent"] * axis) * velocity < 0
        np.testing.assert_allclose(after.v, [velocity], atol=5e-5, rtol=5e-5)
        np.testing.assert_allclose(
            [r.force for r in step.constraints], reference["native"], atol=0.002, rtol=0.002
        )
        np.testing.assert_allclose(pair["force_on_a"], reference["force"], atol=0.002, rtol=0.002)
        np.testing.assert_allclose(
            step.qfrc_constraint, [reference["total"]], atol=0.002, rtol=0.002
        )


@pytest.mark.cuda
@pytest.mark.parametrize("direction", [1, -1], ids=["right-foot", "left-foot"])
def test_side_friction_reverses_with_accepted_motion_not_previous_slip(direction, tmp_path):
    xml = _hinge_xml()
    if direction == -1:
        xml = xml.replace('pos=".12 0 -.052"', 'pos="-.12 0 -.052"')
    spec, world = _case(velocity=direction * 0.35, xml=xml)
    with _evidence(tmp_path / "mixed-passive-reversal.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            # Ground support turns the unpowered hinge through zero while the
            # side is still carrying load. No target velocity enters the solve.
            for _ in range(12):
                before, after, geometry = _advance(batch)
                audit = _audit(spec, before, after, geometry)
                evidence["records"].append(dict(before=before, after=after, audit=audit))
                assert audit["ground_force"][2] > 0.1
                assert audit["pair"]["normal_force"].sum() > 0.1
                assert audit["tangent_torque"] * after.v[0] < 0
                if before.v[0] * after.v[0] < 0:
                    assert direction * after.v[0] < 0
                    break
            else:
                pytest.fail("the loaded hinge did not reverse its slip")


@pytest.mark.cuda
def test_extreme_driven_reversal_has_certified_output_or_atomic_refusal(tmp_path):
    # The second command opposes several kN m of native reaction. At this
    # scale a packed FP32 force may fail the physical-acceleration KKT check.
    spec, world = _case()
    with _evidence(tmp_path / "mixed-reversal.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            for target in (-0.35, 0.35):
                design = _drive_velocity(batch, spec, target)
                checkpoint = batch.checkpoint().to_bytes()
                revision = batch.checkpoint_revision
                try:
                    before, after, geometry = _advance(batch)
                except PhysicsStepFailure as failure:
                    # Only the extreme second drive may hit this numerical boundary.
                    assert target > 0 and design["torque"] > 7000
                    assert "area_friction_solver_or_publication" in str(failure.__cause__.__cause__)
                    assert batch.checkpoint_revision == revision
                    assert batch.checkpoint().to_bytes() == checkpoint
                    evidence["stress_outcome"] = "candidate rejected before publication"
                    evidence["rejected_drive"] = design
                    break
                record = dict(before=before, after=after, geometry=geometry.surface, drive=design)
                evidence["records"].append(record)
                audit = record["audit"] = _audit(spec, before, after, geometry)
                # Load design uses the FP64 source native coefficients while
                # the audited solve uses actual packed coefficients. This
                # endpoint allowance never changes the physical residual gate.
                assert after.v[0] == pytest.approx(target, abs=5e-4, rel=5e-4)
                assert before.v[0] * after.v[0] < 0
                assert np.sum(audit["pair"]["normal_force"]) > 0.1
                assert np.sign(target) * audit["tangent_torque"] < -0.001


@pytest.mark.cuda
def test_unloading_side_clears_all_area_force_while_ground_support_persists(tmp_path):
    # Current separating velocity must clear old normal and tangential loads;
    # clearing the side route must not erase the simultaneous native ground.
    spec, world = _case()
    with _evidence(tmp_path / "mixed-unload.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            for label, options in (
                ("loaded", {}),
                ("separating", {"depth": 0.0}),
                ("empty", {"empty": True}),
            ):
                before, after, geometry = _advance(
                    batch, vertices=VERTICES - [0, 0, 0.2], **options
                )
                record = dict(label=label, before=before, after=after, geometry=geometry.surface)
                evidence["records"].append(record)
                audit = record["audit"] = _audit(spec, before, after, geometry)
                assert abs(after.v[0]) > 0.01
                assert audit["ground_force"][2] > 0.1
                pair = audit["pair"]
                if label == "loaded":
                    assert pair["normal_force"].sum() > 0.1
                    assert np.linalg.norm(pair["tangential_force"]) > 0.01
                else:
                    if label == "separating":
                        assert len(pair["sample_ids"]) == 3
                        assert np.all(
                            np.sum(pair["normals"] * pair["relative_velocity"], axis=1) > 0
                        )
                    else:
                        assert pair["sample_ids"] == ()
                    for name in (
                        "normal_force",
                        "tangential_force",
                        "force_on_a",
                        "force_on_b",
                        "torque_on_a_world_origin",
                        "torque_on_b_world_origin",
                    ):
                        np.testing.assert_array_equal(pair[name], 0)


def _side_asset(*, depth=0.002, lever_height=0.1):
    # Rotate the established four-tetrahedron pressure volume: its +z face
    # becomes a +x side face, 6 cm from the hinge. By default the face cuts 2 mm
    # into the volume; both normal and tangential hinge levers are nonzero.
    from test_mujoco_tyx_area_pressure_scene import BASE, CELLS, RIGID

    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    vertices = np.vstack((BASE, BASE.mean(axis=0))) @ rotation.T + [0.06 - depth, 0.0, lever_height]
    face = RIGID @ rotation.T + [0.06, 0.0, PIVOT[2] + lever_height]
    return BodyAreaPressureAsset(
        IDENTITY,
        TetraPressureMesh(
            ("a", "b", "c", "d", "center"),
            vertices,
            CELLS,
            (0, 0, 0, 0, 400000),
            ("bottom", "side-a", "side-b", "slant"),
        ),
        RigidPressureMesh(("r0", "r1", "r2"), face, ((0, 1, 2),), ("face",)),
    )


def _observation(workspace, batch, spec, pair):
    frame = workspace.refresh(expected_revision=batch.checkpoint_revision)
    arrays = workspace.readback(frame)
    decoded = AreaFrictionEntryReader(workspace, {spec.key: (IDENTITY,)}).validate(arrays, frame)
    samples = decoded[KEY].samples
    assert len(samples) == len(pair["sample_ids"])
    assert all(sample.identity == IDENTITY and sample.stamp == pair["stamp"] for sample in samples)
    for values, expected in (
        ([sample.normal_force_n for sample in samples], pair["normal_force"]),
        ([sample.tangential_force_on_a_world for sample in samples], pair["tangential_force"]),
        ([sample.relative_velocity_world for sample in samples], pair["relative_velocity"]),
    ):
        np.testing.assert_array_equal(values, expected)
    return frame, arrays


@pytest.mark.cuda
def test_continuing_hinge_motion_leaves_finite_side_pressure_volume(tmp_path):
    # A shallow side contact below the pivot unloads as the native ground
    # reaction turns the hinge. The motor is unpowered throughout.
    spec, world = _case()
    with _evidence(tmp_path / "mixed-finite-release.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(
                batch, {spec.key: (_side_asset(depth=0.0002, lever_height=-0.06),)}
            )
            loaded = False
            released = False
            for _ in range(12):
                before = batch.integration_snapshot()[KEY]
                assert scene.step(H, expected_revision=batch.checkpoint_revision) == {KEY: True}
                after = batch.integration_snapshot()[KEY]
                geometry = scene.last_geometry[KEY][0]
                record = dict(before=before, after=after, geometry=geometry)
                evidence["records"].append(record)
                audit = record["audit"] = _audit(spec, before, after, geometry)
                pair = audit["pair"]
                assert audit["ground_force"][2] > 0.1
                if pair["normal_force"].sum() > 0.1:
                    loaded = True
                    assert audit["tangent_torque"] * after.v[0] < 0
                if not pair["sample_ids"]:
                    assert loaded
                    assert geometry.surface.triangles == ()
                    for name in ("normal_force", "tangential_force", "force_on_a", "force_on_b"):
                        np.testing.assert_array_equal(pair[name], 0)
                    released = True
                    break
            assert released, "the finite side pressure volume must unload during this window"


@pytest.mark.cuda
def test_automatic_side_surface_mixed_support_checkpoint_and_v2_observation(tmp_path):
    # Current geometry, physical native contacts and law-aware v2 records must
    # remain bound to one accepted step across restore and replay.
    spec, world = _case()
    with _evidence(tmp_path / "mixed-scene-replay.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (_side_asset(),)})
            with batch.prepare_device_workspace(
                include_bodies=True, include_area_pressure=True
            ) as ws:

                def advance():
                    before = batch.integration_snapshot()[KEY]
                    assert scene.step(H, expected_revision=batch.checkpoint_revision) == {KEY: True}
                    after = batch.integration_snapshot()[KEY]
                    geometry = scene.last_geometry[KEY][0]
                    record = dict(before=before, after=after, geometry=geometry)
                    evidence["records"].append(record)
                    audit = record["audit"] = _audit(spec, before, after, geometry)
                    assert audit["ground_force"][2] > 0.1
                    assert np.sum(audit["pair"]["normal_force"]) > 0.1
                    assert abs(audit["tangent_torque"]) > 0.001
                    frame, arrays = _observation(ws, batch, spec, audit["pair"])
                    record["arrays"] = arrays
                    return after, geometry, frame, arrays

                saved, _, _, saved_arrays = advance()
                capsule = MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(
                    scene.checkpoint().to_bytes()
                )
                tail = [advance(), advance()]
                assert not np.array_equal(saved.q, tail[-1][0].q)
                # The finite rigid face may remain wholly inside one pressure
                # cell, so its vertices need not move; the sampled pressure
                # must nevertheless follow the moving compliant body.
                assert not np.array_equal(
                    tail[0][0].last_step.raw_contact_fields["area_friction"][0]["normal_force"],
                    tail[1][0].last_step.raw_contact_fields["area_friction"][0]["normal_force"],
                )
                scene.restore(capsule, expected_revision=batch.checkpoint_revision)
                _same(batch.integration_snapshot()[KEY], saved)
                with pytest.raises(ValueError):
                    ws.readback(tail[-1][2])
                frame = ws.refresh(expected_revision=batch.checkpoint_revision)
                _same(ws.readback(frame), saved_arrays)
                for expected in tail:
                    replay = advance()
                    _same(replay[0], expected[0], fresh_stamp=True)
                    _same(replay[1], expected[1], fresh_stamp=True)
                    _same(replay[3], expected[3])


@pytest.mark.cuda
@pytest.mark.parametrize("fault", ["wrench", "inactive-distance"])
def test_mixed_restore_rejects_changed_native_contact_before_device_writes(monkeypatch, fault):
    from phymeo.backends import mujoco_checkpoint as codec

    xml = _hinge_xml()
    if fault == "inactive-distance":
        xml = xml.replace('pos=".12 0 -.052"', 'pos=".12 0 -.042"')
        xml = xml.replace('condim="3"', 'margin=".016" gap=".012" condim="3"')
    spec, world = _case(xml=xml)
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        _advance(batch)
        owner = batch._buckets[batch.layout[KEY].bucket]
        capsule = owner.checkpoint()
        revision = owner.checkpoint_revision
        state = dict(capsule._read(capsule.binding))
        world_state = dict(state["worlds"][0])
        receipt = world_state["receipt"]
        raw_fields = dict(receipt.raw_contact_fields)
        native = dict(raw_fields["native_contacts"])
        if fault == "wrench":
            wrench = native["contact_wrench"].copy()
            wrench.flat[0] += 1.0
            native["contact_wrench"] = wrench
        else:
            (ground,) = receipt.contacts
            assert not ground.active and not receipt.constraints
            rows = native["candidate_rows"].copy()
            rows[0, 2] = float(np.float32(0.009))
            native["candidate_rows"] = rows
            receipt = replace(receipt, contacts=(replace(ground, distance=rows[0, 2].item()),))
        raw_fields["native_contacts"] = MappingProxyType(native)
        world_state["receipt"] = replace(receipt, raw_contact_fields=MappingProxyType(raw_fields))
        state["worlds"] = (world_state,)
        forged = codec._make_checkpoint(capsule.binding, capsule.source_revision, state)
        monkeypatch.setattr(
            owner, "_build_device", lambda *a, **kw: pytest.fail("restore wrote a device")
        )
        match = "source native distance" if fault == "inactive-distance" else None
        with pytest.raises(ValueError, match=match):
            owner.restore(forged, expected_revision=revision)
        assert owner.checkpoint_revision == revision
        assert owner.checkpoint().to_bytes() == capsule.to_bytes()


@pytest.mark.cuda
@pytest.mark.parametrize("foot_z, detected", [("-.042", True), ("-.025", False)])
def test_loaded_side_with_inactive_or_absent_ground(foot_z, detected):
    # Distances are +8 mm and +25 mm. Both exceed the +4 mm activation gap.
    xml = _hinge_xml().replace('pos=".12 0 -.052"', f'pos=".12 0 {foot_z}"')
    xml = xml.replace('condim="3"', 'margin=".016" gap=".012" condim="3"')
    spec, world = _case(xml=xml)
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        before, after, geometry = _advance(batch)
        receipt = after.last_step
        assert len(receipt.contacts) == int(detected) and not receipt.constraints
        if detected:
            ground = receipt.contacts[0]
            assert not ground.active and ground.distance > ground.include_margin
            np.testing.assert_array_equal(ground.wrench_contact_frame, 0)
        (pair,) = receipt.raw_contact_fields["area_friction"]
        _audit_surface(pair, geometry, receipt)
        assert pair["normal_force"].sum() > 0.1
        torque = float(np.cross(pair["points"] - PIVOT, pair["force_on_a"]).sum(axis=0)[1])
        np.testing.assert_allclose(receipt.qfrc_constraint, [torque], rtol=2e-5, atol=2e-5)
        expected_velocity = (MASS_MOMENT * before.v[0] + H * torque) / (MASS_MOMENT + H * DAMPING)
        assert after.v[0] == pytest.approx(expected_velocity, abs=2e-6)


@pytest.mark.cuda
def test_native_rows_and_area_samples_share_capacity_with_atomic_retry(tmp_path):
    # Four native rows plus three side samples fit; another triangle must not
    # overwrite rows, publish partial forces or poison the next valid step.
    spec, world = _case(capacity=7)
    with _evidence(tmp_path / "mixed-capacity.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            with batch.prepare_device_workspace(
                include_bodies=True, include_area_pressure=True
            ) as ws:
                before, accepted, geometry = _advance(batch)
                audit = _audit(spec, before, accepted, geometry)
                frame, arrays = _observation(ws, batch, spec, audit["pair"])
                revision, publication = batch.checkpoint_revision, batch.last_step
                checkpoint = batch.checkpoint().to_bytes()
                oversized = _surface(batch, copies=2)
                evidence.update(before=accepted, oversized=oversized, arrays=arrays)
                with pytest.raises((ValueError, PhysicsStepFailure)):
                    batch.step_area_pressure(H, {KEY: (oversized,)}, expected_revision=revision)
                assert batch.checkpoint_revision == revision and batch.last_step is publication
                assert batch.checkpoint().to_bytes() == checkpoint
                _same(batch.integration_snapshot()[KEY], accepted)
                _same(ws.readback(frame), arrays)
                before, after, geometry = _advance(batch)
                evidence["retry"] = after
                _audit(spec, before, after, geometry)
                assert after.tick == accepted.tick + 1
