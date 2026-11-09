"""AUTHORING_ONLY: real last-body contact on bounded 29/32-joint free trees.

Parent owns all import/test/static/device execution. Requires the parent's
33-group bridge correction; this file neither edits products nor trusts a source
flag as an oracle. Original MAX_SHAPES, mu=0, caps and motion tolerances remain.

The chain is a declared concentric/coaxial mechanical model, not the G1 plant.
Root mass4/I=.5, N children mass1/I=.25, zero COM/offsets, revolute-z ancestors.
Only physical body N has collision geometry: a sphere at local x=.25. Its
initial floor depth1/32 and k64 give Fz=2N and COM torque_y=-.5Nm. A fresh
physical contact step must propagate BOTH into the free root through full M/J.
"""

import traceback
from itertools import product

import numpy as np
import pytest
from test_resident_humanoid_capacity import H, _save

import phymeo as ph
from phymeo.core.rigid import RigidBodyModel
from phymeo.core.robot import RobotModel
from phymeo.dynamics.floating_robot import FloatingRobotModel, FloatingRobotState

WORLD = 41
RTOL, ATOL = 5e-5, 5e-6


def _chain(n):
    children = RobotModel(
        np.arange(-1, n - 1, dtype=np.int64),
        ("revolute",) * n,
        np.zeros((n, 3)),
        np.tile([0.0, 0, 0, 1], (n, 1)),
        np.tile([0.0, 0, 1], (n, 1)),
        np.ones(n),
        np.zeros((n, 3)),
        np.full((n, 3), 0.25),
        np.zeros(3),
    )
    model = FloatingRobotModel(
        RigidBodyModel(np.array([0.25]), np.full((1, 3), 0.5), np.zeros(3)),
        np.zeros(3),
        children,
    )
    state = FloatingRobotState(
        np.r_[np.array([0.0, 0, 3 / 32, 0, 0, 0, 1]), np.zeros(n)], np.zeros(n + 6)
    )
    return (
        model,
        state,
        ph.FloatingRobotTree(
            model, state, kp=np.zeros(n), kd=np.zeros(n), effort_limit=np.full(n, 8.0)
        ),
    )


def _declared_oracle(n):
    # Child angular speed is Omega + e_z*(L*qdot), L lower triangular with ones.
    # K = .5*(N+4)|v|^2 + .5*.5|Omega|^2
    #     + .5*.25*sum_b |Omega + e_z*(L*qdot)_b|^2.
    # This proves SPD and supplies all coupled blocks without a product solve.
    d, b = n + 6, n + 1
    incidence = np.tril(np.ones((n, n)))
    mass = np.zeros((d, d))
    mass[:3, :3] = (n + 4) * np.eye(3)
    mass[3:6, 3:6] = (0.5 + n / 4) * np.eye(3)
    mass[5, 6:] = 0.25 * np.sum(incidence, axis=0)
    mass[6:, 5] = mass[5, 6:]
    mass[6:, 6:] = 0.25 * incidence.T @ incidence
    linear, angular = np.zeros((b, 3, d)), np.zeros((b, 3, d))
    linear[:, :, :3] = np.eye(3)
    angular[:, :, 3:6] = np.eye(3)
    for body in range(1, b):
        angular[body, 2, 6 : 6 + body] = 1
    wrench = np.zeros((b, 6))
    wrench[-1] = [0, 0, 2, 0, -0.5, 0]
    generalized = np.zeros(d)
    generalized[2], generalized[4] = 2, -0.5
    acceleration = np.zeros(d)
    acceleration[2] = 2 / (n + 4)
    acceleration[4] = -0.5 / (0.5 + n / 4)
    np.testing.assert_allclose(mass @ acceleration, generalized, rtol=0, atol=1e-12)
    return dict(
        incidence=incidence,
        mass=mass,
        linear_jacobian=linear,
        angular_jacobian=angular,
        wrench=wrench,
        generalized=generalized,
        acceleration=acceleration,
    )


@pytest.mark.tyx
@pytest.mark.parametrize("n", (29, 32), ids=("g29-last-body", "bound32-last-body"))
def test_actual_last_body_contact_reaches_full_floating_dynamics(n, tyx_mode, tmp_path):
    model, initial, tree = _chain(n)
    expected = _declared_oracle(n)
    command = tree.inputs(np.zeros((n, 3)))
    moving, floor = ph.CollisionKey(WORLD, 101, 1), ph.CollisionKey(WORLD, 900, 2)
    sphere = ph.CollisionShape(moving, "sphere", (701,), radius=1 / 8)
    ground = ph.CollisionShape(floor, "hull", tuple(range(801, 809)))
    ground_points = np.array(list(product((-1.0, 1.0), (-1.0, 1.0), (-0.25, 0.0))))
    local_sphere = np.array([[0.25, 0, 0]])
    collision = ph.CollisionModel((sphere, ground))
    law = ph.RobotContactMaterial(64, 0, 0)
    record = dict(
        schema="humanoid-last-body-contact-capacity-v1",
        mode=tyx_mode,
        n=n,
        physical_model=model,
        initial_state=initial,
        command=command,
        h_s=H,
        collision=collision,
        local_sphere=local_sphere,
        ground_points=ground_points,
        law=law,
        expected=expected,
        requested_physical_steps=1,
    )
    batch = None
    try:
        with ph.TyxWorldBatch([(WORLD, tree)], tyx_mode, cache_root=tmp_path) as batch:
            batch.set_controls({WORLD: command})
            before = batch.snapshot()[WORLD]
            bindings = (
                ph.ResidentFloatingTreeBinding(
                    moving, WORLD, n, local_sphere, topology=before.topology
                ),
                ph.ResidentStaticBinding(floor, ph.CollisionMotion(ground_points)),
            )
            record.update(
                before=before, bindings=bindings, entry_geometry=batch.tree_kinematics()[WORLD]
            )
            with batch.prepare_contacts(collision, bindings, {(moving, floor): law}) as plan:
                try:
                    accepted_flags = batch.step_contacts(plan, H)
                    record["accepted_flags"] = accepted_flags
                    after = batch.snapshot()[WORLD]
                    record.update(
                        after=after,
                        geometry=batch.tree_kinematics()[WORLD],
                        diagnostic=batch.tree_diagnostics()[WORLD],
                        summary=batch.summary()[WORLD],
                        plan_stats=plan.stats(),
                        batch_stats=batch.stats(),
                    )
                    assert accepted_flags == {WORLD: True}
                    receipt = plan.last_accepted[WORLD]
                    record["receipt"] = receipt
                    diagnostic = receipt.diagnostics["floating_diagnostic"]
                    geometry = record["geometry"]
                    assert before.tick == 0 and after.tick == 1
                    assert receipt.entry_state == before and receipt.accepted_state == after
                    assert (receipt.entry_tick, receipt.exit_tick, receipt.dt_s) == (0, 1, H)
                    assert receipt.diagnostics["entry_frames"][moving].group_index == n
                    assert receipt.diagnostics["accepted_frames"][moving].group_index == n
                    assert (
                        geometry.joint_count,
                        geometry.body_count,
                        geometry.velocity_count,
                        geometry.configuration_count,
                    ) == (n, n + 1, n + 6, n + 7)
                    assert receipt.law_wrench_fp64.shape == (n + 1, 6)
                    np.testing.assert_allclose(
                        receipt.law_wrench_fp64, expected["wrench"], rtol=RTOL, atol=ATOL
                    )
                    np.testing.assert_allclose(
                        receipt.uploaded_wrench_fp32, expected["wrench"], rtol=RTOL, atol=ATOL
                    )
                    np.testing.assert_allclose(
                        receipt.contact_generalized, expected["generalized"], rtol=RTOL, atol=ATOL
                    )
                    np.testing.assert_allclose(
                        receipt.input_external, np.zeros(n + 6), rtol=0, atol=ATOL
                    )
                    np.testing.assert_allclose(receipt.actual_drive, np.zeros(n), rtol=0, atol=ATOL)
                    np.testing.assert_allclose(
                        diagnostic.mass, expected["mass"], rtol=RTOL, atol=ATOL
                    )
                    np.testing.assert_allclose(
                        diagnostic.linear_jacobian,
                        expected["linear_jacobian"],
                        rtol=RTOL,
                        atol=ATOL,
                    )
                    np.testing.assert_allclose(
                        diagnostic.angular_jacobian,
                        expected["angular_jacobian"],
                        rtol=RTOL,
                        atol=ATOL,
                    )
                    np.testing.assert_allclose(diagnostic.bias, np.zeros(n + 6), rtol=0, atol=ATOL)
                    np.testing.assert_allclose(
                        diagnostic.external, expected["generalized"], rtol=RTOL, atol=ATOL
                    )
                    np.testing.assert_allclose(
                        diagnostic.acceleration, expected["acceleration"], rtol=RTOL, atol=ATOL
                    )
                    expected_velocity = H * expected["acceleration"]
                    np.testing.assert_allclose(
                        after.velocity, expected_velocity, rtol=RTOL, atol=ATOL
                    )
                    np.testing.assert_allclose(
                        geometry.linear_velocity_m_s,
                        np.tile(expected_velocity[:3], (n + 1, 1)),
                        rtol=RTOL,
                        atol=ATOL,
                    )
                    np.testing.assert_allclose(
                        geometry.angular_velocity_rad_s,
                        np.tile(expected_velocity[3:6], (n + 1, 1)),
                        rtol=RTOL,
                        atol=ATOL,
                    )
                    np.testing.assert_allclose(
                        after.generalized_applied, expected["generalized"], rtol=RTOL, atol=ATOL
                    )
                    # Compare drift per h with the original velocity budget so a
                    # zero tiny position/quaternion update cannot pass by atol.
                    displacement_rate = (np.array(after.position[:3]) - initial.q[:3]) / H
                    np.testing.assert_allclose(
                        displacement_rate, expected_velocity[:3], rtol=RTOL, atol=ATOL
                    )
                    angle = H * expected_velocity[4]
                    expected_quaternion = np.array([0.0, np.sin(angle / 2), 0, np.cos(angle / 2)])
                    quaternion_rate = (np.array(after.position[3:7]) - initial.q[3:7]) / H
                    np.testing.assert_allclose(
                        quaternion_rate,
                        (expected_quaternion - initial.q[3:7]) / H,
                        rtol=RTOL,
                        atol=ATOL,
                    )
                    np.testing.assert_allclose(after.position[7:], np.zeros(n), rtol=0, atol=ATOL)
                    np.testing.assert_allclose(
                        receipt.static_reactions[floor].force_n, [0, 0, -2], rtol=RTOL, atol=ATOL
                    )
                    np.testing.assert_allclose(
                        receipt.static_reactions[floor].torque_nm, [0, 0.5, 0], rtol=RTOL, atol=ATOL
                    )
                    energy = (
                        0.5 * (n + 4) * expected_velocity[2] ** 2
                        + 0.5 * (0.5 + n / 4) * expected_velocity[4] ** 2
                    )
                    np.testing.assert_allclose(
                        record["summary"]["kinetic_energy_j"], energy, rtol=RTOL, atol=ATOL
                    )
                    assert record["summary"]["kinetic_energy_j"] > 0
                    assert after.control == before.control and after.history == before.history == ()
                    assert diagnostic.equation_residual <= 2e-5
                    assert batch.stats()["accepted_world_steps"] == 1
                    record["status"] = "passed"
                except Exception:
                    record.update(
                        retained=batch.snapshot(),
                        plan_stats=plan.stats(),
                        batch_stats=batch.stats(),
                    )
                    raise
    except Exception as error:
        record.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        if batch is not None:
            record["last_actual_statistics"] = batch.stats()
        _save(tmp_path, "humanoid-last-body-contact-" + str(n), record)
