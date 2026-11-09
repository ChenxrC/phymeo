"""Original velocity friction through actual resident full-M mechanics.

AUTHORING_ONLY / UNEXECUTED. Eight continuous accepted steps per physical case,
not a G1 window, unload/recontact proof, static-friction history or device force kernel.
"""

import gzip
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

import phymeo as ph
from phymeo.core.rigid import RigidBodyModel
from phymeo.dynamics.floating_robot import FloatingRobotModel, FloatingRobotState

H = 1 / 1024
LAW = ph.RobotContactMaterial(64, 4, 1 / 2, 1 / 8)
ATOL, RTOL = 5e-6, 5e-5


def _json(value):
    if is_dataclass(value):
        return {field.name: _json(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def _save(path, record):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(_json(record), stream, allow_nan=False)


def _close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)


def _box(lo, hi):
    return np.array(list(product(*zip(lo, hi, strict=True))), dtype=float)


def _force(depth, normal, velocity):
    """Independent literal law, not a call to product geometry or sample_loads."""
    vn = float(np.dot(normal, velocity))
    tangent = velocity - vn * normal
    fn = max(0.0, 64 * depth - 4 * vn) if depth > 0 else 0.0
    return fn * normal - (fn / 2) * tangent / max(np.linalg.norm(tangent), 1 / 8)


def _world_moment(receipt, com):
    """Check full-world balance including the separately published fixed reaction."""
    wrench = receipt.law_wrench_fp64[0]
    reaction = next(iter(receipt.static_reactions.values()))
    _close(wrench[:3] + reaction.force_n, np.zeros(3))
    _close(
        np.cross(com, wrench[:3])
        + wrench[3:]
        + np.cross(reaction.reference_point_m, reaction.force_n)
        + reaction.torque_nm,
        np.zeros(3),
    )


def _works(receipt, force, velocity, material, point, p0, r0, p1, r1):
    """Independent rigid transport using the analytic entry/exit poses."""
    assert len(receipt.sites) == 1
    site = receipt.sites[0]
    material_motion = p1 + r1 @ np.linalg.solve(r0, material - p0) - material
    application_motion = p1 + r1 @ np.linalg.solve(r0, point - p0) - point
    expected_material, expected_application = force @ material_motion, force @ application_motion
    _close(site.first_material_displacement_m, material_motion)
    _close(site.first_application_displacement_m, application_motion)
    _close(site.second_material_displacement_m, np.zeros(3))
    _close(site.second_application_displacement_m, np.zeros(3))
    _close(site.material_work_j, expected_material)
    _close(site.application_work_j, expected_application)
    _close(receipt.diagnostics["material_work_j"], expected_material)
    _close(receipt.diagnostics["application_work_j"], expected_application)
    _close(receipt.diagnostics["entry_power_times_h_j"], (force @ velocity) * H)


def _bind(batch, world, local, support, *, floating):
    first, second = ph.CollisionKey(world, 101, 7), ph.CollisionKey(world, 909, 2)
    shapes = (
        ph.CollisionShape(first, "sphere", (111,), radius=1 / 8),
        ph.CollisionShape(second, "hull", tuple(range(301, 309))),
    )
    topology = batch.snapshot()[world].topology
    binding_type = ph.ResidentFloatingTreeBinding if floating else ph.ResidentTreeBinding
    bindings = (
        binding_type(first, world, 0, np.array([local]), topology=topology),
        ph.ResidentStaticBinding(second, ph.CollisionMotion(support)),
    )
    return ph.CollisionModel(shapes), bindings, (first, second)


def _ry_from_quaternion(q):
    # Independent scalar quaternion-to-Y-rotation identity, no product FK.
    y, w = q[4], q[6]
    scale = y * y + w * w
    c, s = (w * w - y * y) / scale, 2 * y * w / scale
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _drift_y_quaternion(q, omega):
    y, w = q[4], q[6]
    length = np.sqrt(y * y + w * w)
    y, w = y / length, w / length
    sine, cosine = np.sin(H * omega / 2), np.cos(H * omega / 2)
    return np.array([0, cosine * y + sine * w, 0, cosine * w - sine * y])


@pytest.mark.tyx
def test_floating_offset_sphere_friction_has_real_six_coordinate_response(tyx_mode, tmp_path):
    world = 137
    root = RigidBodyModel(np.array([1 / 2]), np.full((1, 3), 1 / 2), np.zeros(3))
    model = FloatingRobotModel(root, np.zeros(3), None)
    state = FloatingRobotState(
        np.array([0.0, 0, 3 / 32, 0, 0, 0, 1]),
        np.array([1 / 4, 0, -1 / 8, 0, 0, 0]),
    )
    tree = ph.FloatingRobotTree(
        model, state, kp=np.zeros(0), kd=np.zeros(0), effort_limit=np.zeros(0)
    )
    local = np.array([1 / 4, 0, 0])
    support = _box([-1, -1, -1 / 4], [1, 1, 0])
    mass = np.diag([2.0, 2, 2, 1 / 2, 1 / 2, 1 / 2])
    record = dict(
        mode=tyx_mode,
        h=H,
        model=model,
        initial=state,
        local_sphere=local,
        static_points=support,
        law=LAW,
        literal_mass=mass,
        steps=[],
    )
    try:
        with ph.TyxWorldBatch([(world, tree)], tyx_mode, cache_root=tmp_path) as batch:
            command = tree.inputs(np.zeros((0, 3)))
            record["command"] = command
            batch.set_controls({world: command})
            collision, bindings, pair = _bind(batch, world, local, support, floating=True)
            record.update(collision_model=collision, bindings=bindings, registered_pair=pair)
            before = batch.snapshot()[world]
            # A real rejected physical attempt must not repair/advance q or v.
            with batch.prepare_contacts(
                collision, bindings, {pair: LAW}, site_capacity=0
            ) as limited:
                assert batch.step_contacts(limited, H) == {world: False}
                assert batch.snapshot()[world] == before and not limited.last_accepted
                record["capacity_rejection"] = dict(
                    retained=batch.snapshot(), status=limited.last_status
                )
            with batch.prepare_contacts(collision, bindings, {pair: LAW}) as plan:
                # The public wrong-owner guard must fire before another batch advances.
                with ph.TyxWorldBatch(
                    [(world + 1, tree)], tyx_mode, cache_root=tmp_path / "other"
                ) as other:
                    retained = other.snapshot()
                    with pytest.raises(ValueError, match="different owner"):
                        other.step_contacts(plan, H)
                    assert other.snapshot() == retained
                for step in range(8):
                    before = batch.snapshot()[world]
                    q, v = np.asarray(before.position), np.asarray(before.velocity)
                    center = q[:3]
                    axis = center + _ry_from_quaternion(q) @ local
                    gap = axis[2] - 1 / 8
                    point = np.array([axis[0], axis[1], gap / 2])
                    contact_velocity = v[:3] + np.cross(v[3:6], point - center)
                    force = _force(-gap, np.array([0.0, 0, 1]), contact_velocity)
                    torque = np.cross(point - center, force)
                    generalized = np.r_[force, torque]
                    acceleration = generalized / np.diag(mass)
                    expected_v = v + H * acceleration
                    expected_q = np.r_[
                        center + H * expected_v[:3],
                        _drift_y_quaternion(q, expected_v[4]),
                    ]
                    if step == 0:
                        _close(force, [-5 / 4, 0, 5 / 2])
                        _close(torque, [0, -125 / 256, 0])
                    assert batch.step_contacts(plan, H) == {world: True}
                    after, receipt = batch.snapshot()[world], plan.last_accepted[world]
                    record["steps"].append(
                        dict(
                            entry=before,
                            accepted=after,
                            receipt=receipt,
                            oracle_gap=gap,
                            oracle_point=point,
                            oracle_velocity=contact_velocity,
                            oracle_force=force,
                            oracle_torque=torque,
                            oracle_acceleration=acceleration,
                            oracle_q=expected_q,
                            oracle_v=expected_v,
                        )
                    )
                    assert receipt.entry_state == before and receipt.accepted_state == after
                    assert (receipt.entry_tick, receipt.exit_tick) == (step, step + 1)
                    _close(receipt.diagnostics["mass"], mass)
                    linear, angular = np.zeros((1, 3, 6)), np.zeros((1, 3, 6))
                    linear[0, :, :3], angular[0, :, 3:] = np.eye(3), np.eye(3)
                    _close(receipt.diagnostics["linear_jacobian"], linear)
                    _close(receipt.diagnostics["angular_jacobian"], angular)
                    _close(receipt.law_wrench_fp64[0], generalized)
                    _close(receipt.contact_generalized, generalized)
                    _close(receipt.diagnostics["bias"], np.zeros(6))
                    _close(receipt.diagnostics["acceleration"], acceleration)
                    _close(after.velocity, expected_v)
                    _close(after.position, expected_q)
                    assert len(receipt.sites) == 1
                    _close(receipt.sites[0].first_velocity_m_s, contact_velocity)
                    _close(receipt.sites[0].geometry.load_point, point)
                    _world_moment(receipt, center)
                    _works(
                        receipt,
                        force,
                        contact_velocity,
                        axis - [0, 0, 1 / 8],
                        point,
                        center,
                        _ry_from_quaternion(q),
                        expected_q[:3],
                        _ry_from_quaternion(expected_q),
                    )
                record.update(stats=plan.stats(), batch_stats=batch.stats())
    except Exception as error:
        record["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        _save(tmp_path / "floating-friction-window.json.gz", record)


@pytest.mark.tyx
def test_fixed_rotor_friction_uses_arc_velocity_and_true_joint_inertia(tyx_mode, tmp_path):
    world = 149
    model = ph.RobotModel(
        np.array([-1]),
        ("revolute",),
        np.zeros((1, 3)),
        np.array([[0.0, 0, 0, 1]]),
        np.array([[0.0, 0, 1]]),
        np.array([1.0]),
        np.zeros((1, 3)),
        np.array([[1 / 8, 1 / 8, 1 / 4]]),
        np.zeros(3),
    )
    program = ph.DrivenRobotProgram(
        model, kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.array([8.0])
    )
    state = program.initial_state(ph.RobotState(np.zeros(1), np.array([-1 / 2])))
    tree = ph.FixedRobotTree(program, state, control_max_iterations=512)
    command = program.inputs(
        np.zeros(1),
        np.zeros(1),
        feedforward=np.zeros(1),
        loads=ph.RobotInputs(np.zeros(1), np.zeros((1, 3)), np.zeros((1, 3))),
    )
    local = np.array([1 / 2, 3 / 32, 0])
    support = _box([-1, -1 / 4, -1 / 2], [1, 0, 1 / 2])
    record = dict(
        mode=tyx_mode,
        h=H,
        model=model,
        initial=state,
        command=command,
        local_sphere=local,
        static_points=support,
        law=LAW,
        steps=[],
    )
    try:
        with ph.TyxWorldBatch([(world, tree)], tyx_mode, cache_root=tmp_path) as batch:
            batch.set_controls({world: command})
            collision, bindings, pair = _bind(batch, world, local, support, floating=False)
            record.update(collision_model=collision, bindings=bindings, registered_pair=pair)
            with batch.prepare_contacts(collision, bindings, {pair: LAW}) as plan:
                for step in range(8):
                    before = batch.snapshot()[world]
                    theta, omega = before.position[0], before.velocity[0]
                    c, s = np.cos(theta), np.sin(theta)
                    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    axis = rotation @ local
                    gap = axis[1] - 1 / 8
                    point = np.array([axis[0], gap / 2, 0])
                    contact_velocity = np.cross([0, 0, omega], point)
                    force = _force(-gap, np.array([0.0, 1, 0]), contact_velocity)
                    torque = np.cross(point, force)
                    acceleration = torque[2] / (1 / 4)
                    expected_v = omega + H * acceleration
                    expected_q = theta + H * expected_v
                    if step == 0:
                        _close(force, [3 / 32, 3, 0])
                        _close(torque, [0, 0, 3075 / 2048])
                    assert batch.step_contacts(plan, H) == {world: True}
                    after, receipt = batch.snapshot()[world], plan.last_accepted[world]
                    record["steps"].append(
                        dict(
                            entry=before,
                            accepted=after,
                            receipt=receipt,
                            oracle_gap=gap,
                            oracle_point=point,
                            oracle_velocity=contact_velocity,
                            oracle_force=force,
                            oracle_torque=torque,
                            oracle_acceleration=acceleration,
                            oracle_q=expected_q,
                            oracle_v=expected_v,
                        )
                    )
                    assert receipt.entry_state == before and receipt.accepted_state == after
                    assert (receipt.entry_tick, receipt.exit_tick) == (step, step + 1)
                    _close(receipt.diagnostics["mass"], [[1 / 4]])
                    _close(receipt.diagnostics["linear_jacobian"], np.zeros((1, 3, 1)))
                    _close(receipt.diagnostics["angular_jacobian"], np.array([[[0], [0], [1]]]))
                    _close(receipt.law_wrench_fp64[0], np.r_[force, torque])
                    _close(receipt.contact_generalized, [torque[2]])
                    _close(receipt.diagnostics["acceleration"], [acceleration])
                    _close(after.position, [expected_q])
                    _close(after.velocity, [expected_v])
                    _close(receipt.sites[0].first_velocity_m_s, contact_velocity)
                    _close(receipt.sites[0].geometry.load_point, point)
                    _world_moment(receipt, np.zeros(3))
                    c1, s1 = np.cos(expected_q), np.sin(expected_q)
                    exit_rotation = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
                    _works(
                        receipt,
                        force,
                        contact_velocity,
                        axis - [0, 1 / 8, 0],
                        point,
                        np.zeros(3),
                        rotation,
                        np.zeros(3),
                        exit_rotation,
                    )
                record.update(stats=plan.stats(), batch_stats=batch.stats())
    except Exception as error:
        record["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        _save(tmp_path / "rotor-friction-window.json.gz", record)
