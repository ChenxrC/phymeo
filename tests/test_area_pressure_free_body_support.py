"""I3s: one real FREE trajectory, audited from independent native entry terms.

MuJoCo supplies only current-state M/J/bias and a position integrator. It never
steps the contact problem. The cube pressure, load law, virtual work and BE
update below do not call a production geometry, row-assembly or solve helper.
"""

import gzip
import json

import numpy as np
import pytest

import experiments.area_pressure.free_body_support as experiment

H, STEPS, MASS, GRAVITY = 0.001, 150, 0.30, 9.81
KEYS = (41, 7, 305)
MODEL = "free-cube-support"
HALF = np.array((0.02, 0.02, 0.01))
INERTIA = np.array((MASS / 6000, MASS / 6000, MASS / 3750))
PRESSURE, GRADIENT = 100000.0, 10000000.0
BARYCENTRIC = np.array(((2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)))


def _rotation(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=np.float64) / np.linalg.norm(quaternion)
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        )
    )


def _tilted_quaternion():
    # Rz(.35) Ry(.02), WXYZ; multiplication is independent of product helpers.
    cy, sy, ct, st = np.cos(0.35 / 2), np.sin(0.35 / 2), np.cos(0.02 / 2), np.sin(0.02 / 2)
    return np.array((cy * ct, -sy * st, cy * st, sy * ct))


def _initial_q(key):
    if key == 41:
        return np.array((0, 0, 0.011, 1, 0, 0, 0), dtype=np.float64)
    if key == 7:
        return np.r_[0, 0, (0.01 - 0.00073575) / np.cos(0.02), _tilted_quaternion()]
    return np.array((0.001, 0, 0.00926425, 1, 0, 0, 0), dtype=np.float64)


def _tent(points, position, rotation):
    local = (np.asarray(points, dtype=np.float64).reshape(-1, 3) - position) @ rotation
    levels = 1 - np.abs(local) / HALF
    pressure = PRESSURE * np.maximum(0, np.min(levels, axis=1))
    bottom = 1 + local[:, 2] / HALF[2]
    margin = np.min(levels[:, :2], axis=1) - bottom
    return pressure, local, margin


def _max_error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)), initial=0))


def _angle(left, right):
    left, right = np.asarray(left), np.asarray(right)
    cosine = abs(float(left @ right)) / float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(2 * np.arccos(np.clip(cosine, 0, 1)))


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _all_finite(value):
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item) for item in value)
    return not isinstance(value, (int, float)) or bool(np.isfinite(value))


def _native(mj, model, state, control):
    """Fresh independent native kinematics/dynamics, never native contact step."""
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = state["q"], state["v"], state["time"]
    data.ctrl[:] = control["ctrl"]
    data.qfrc_applied[:] = control["qfrc_applied"]
    data.xfrc_applied[:] = control["xfrc_applied"]
    mj.mj_forward(model, data)
    assert data.ncon == data.nefc == 0
    mass = np.empty((6, 6))
    mj.mj_fullM(model, mass, data.qM)
    jacobian = np.zeros((model.nbody, 6, 6))
    applied = data.qfrc_actuator.copy() + data.qfrc_applied
    for body in range(model.nbody):
        mj.mj_jac(model, data, jacobian[body, :3], jacobian[body, 3:], data.xipos[body], body)
        applied += jacobian[body].T @ data.xfrc_applied[body]
    return data, dict(
        mass_matrix=mass,
        jacobian_com=jacobian,
        body_position=data.xpos.copy(),
        body_rotation=data.xmat.reshape(model.nbody, 3, 3).copy(),
        body_com=data.xipos.copy(),
        bias=data.qfrc_bias.copy(),
        passive=data.qfrc_passive.copy(),
        applied=applied,
    )


def _point_jacobians(mj, model, data, points, body_a, body_b):
    ja, jb = np.zeros((len(points), 3, 6)), np.zeros((len(points), 3, 6))
    angular = np.zeros((3, 6))
    for index, point in enumerate(points):
        mj.mj_jac(model, data, ja[index], angular, point, body_a)
        mj.mj_jac(model, data, jb[index], angular, point, body_b)
    return ja, jb


def _all_active_velocity(mass, velocity, rhs, jacobian, stiffness, gap, h):
    system = mass + h * h * jacobian.T @ (stiffness[:, None] * jacobian)
    load = mass @ velocity + h * rhs - h * jacobian.T @ (stiffness * gap)
    result = np.linalg.solve(system, load)
    forces = -stiffness * (gap + h * (jacobian @ result))
    assert np.all(forces > 0), "the registered first-step all-active premise failed"
    return result, forces


def _assert_model(mj, model):
    assert (model.nq, model.nv, model.njnt, model.nu, model.na) == (7, 6, 1, 0, 0)
    assert model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE
    body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pressure")
    plate = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "plate")
    assert body > 0 and plate > 0 and model.body_parentid[body] == 0
    assert model.body_mass[body] == pytest.approx(MASS, rel=0, abs=1e-15)
    np.testing.assert_allclose(model.body_inertia[body], INERTIA, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(model.body_ipos[body], np.zeros(3))
    np.testing.assert_array_equal(model.dof_damping, np.zeros(6))
    np.testing.assert_array_equal(model.dof_frictionloss, np.zeros(6))
    np.testing.assert_array_equal(model.dof_armature, np.zeros(6))
    np.testing.assert_array_equal(model.opt.gravity, (0, 0, -GRAVITY))
    assert model.opt.timestep == H and model.opt.iterations == 128 and model.opt.tolerance == 1e-10
    assert not np.any(model.geom_contype) and not np.any(model.geom_conaffinity)
    assert not np.any(model.jnt_limited)
    np.testing.assert_array_equal(model.body_pos[plate], (0, 0, -0.0025))
    geoms = np.flatnonzero(model.geom_bodyid == body)
    assert len(geoms) == 1 and model.geom_type[geoms[0]] == mj.mjtGeom.mjGEOM_BOX
    np.testing.assert_array_equal(model.geom_size[geoms[0]], HALF)
    geoms = np.flatnonzero(model.geom_bodyid == plate)
    assert len(geoms) == 1 and model.geom_type[geoms[0]] == mj.mjtGeom.mjGEOM_BOX
    np.testing.assert_array_equal(model.geom_size[geoms[0]], (0.01, 0.01, 0.0025))
    return body, plate


def test_registered_free_fixture_preserves_full_pressure_and_unconstrained_motion():
    import mujoco as mj

    specs, worlds, assets, manifest = experiment.case_declarations()
    assert len(specs) == 1 and len(worlds) == 3 and set(assets) == {MODEL}
    assert specs[0].key == MODEL and specs[0].area_pressure_pairs == (("pressure", "plate"),)
    assert specs[0].contact_capacity == 16 and specs[0].constraint_capacity == 64
    assert tuple(world.world for world in worlds) == KEYS
    assert all(world.model_key == MODEL for world in worlds)
    assert manifest["schema"] == "free-body-area-support-v1"
    assert manifest["world_order"] == list(KEYS)
    assert manifest["h"] == H and manifest["steps"] == STEPS and manifest["duration"] == 0.15
    model = mj.MjModel.from_xml_string(manifest["xml"])
    _assert_model(mj, model)
    for key in KEYS:
        np.testing.assert_allclose(
            manifest["worlds"][str(key)]["initial_q"], _initial_q(key), rtol=0, atol=1e-15
        )
        np.testing.assert_array_equal(manifest["worlds"][str(key)]["initial_v"], np.zeros(6))
    for world in worlds:
        np.testing.assert_allclose(
            world.initial_state.q, _initial_q(world.world), rtol=0, atol=1e-15
        )
        np.testing.assert_array_equal(world.initial_state.v, np.zeros(6))
    (asset,) = assets[MODEL]
    assert asset.identity.compliant_body == "pressure" and asset.identity.rigid_body == "plate"
    pressure, rigid = asset.pressure_local, asset.rigid_local
    assert pressure.pressure_domain == "full" and pressure.world_vertices.shape == (9, 3)
    assert len(pressure.cells) == 12 and len(rigid.faces) == 2
    expected = {(x, y, z) for x in (-0.02, 0.02) for y in (-0.02, 0.02) for z in (-0.01, 0.01)}
    assert set(map(tuple, pressure.world_vertices[:8])) == expected
    np.testing.assert_array_equal(pressure.world_vertices[8], (0, 0, 0))
    np.testing.assert_array_equal(pressure.nodal_pressure, [0] * 8 + [PRESSURE])
    assert all(8 in cell for cell in pressure.cells)
    boundary = [tuple(node for node in cell if node != 8) for cell in pressure.cells]
    counts = {node: sum(node in face for face in boundary) for node in range(8)}
    assert all(count > 0 for count in counts.values())
    assert set(map(tuple, rigid.world_vertices)) == {
        (x, y, 0.0025) for x in (-0.01, 0.01) for y in (-0.01, 0.01)
    }
    for value in (pressure.world_vertices, pressure.nodal_pressure, rigid.world_vertices):
        assert not value.flags.writeable


def test_independent_tent_full_jacobian_and_all_active_oracles_have_hand_checks():
    q = _initial_q(7)
    rotation = _rotation(q[3:])
    cy, sy, ct, st = np.cos(0.35), np.sin(0.35), np.cos(0.02), np.sin(0.02)
    expected_rotation = np.array(((cy * ct, -sy, cy * st), (sy * ct, cy, sy * st), (-st, 0, ct)))
    np.testing.assert_allclose(rotation, expected_rotation, rtol=0, atol=2e-14)
    local = np.array(((0, 0, -0.009), (0.004, -0.003, -0.0095)))
    points = q[:3] + local @ rotation.T
    pressure, recovered, margin = _tent(points, q[:3], rotation)
    np.testing.assert_allclose(recovered, local, rtol=0, atol=1e-17)
    np.testing.assert_allclose(pressure, (10000, 5000), rtol=0, atol=1e-9)
    assert np.all(margin > 0.4)
    # A scalar z response embedded in all six native DOFs has a hand BE force.
    mass = np.diag(np.r_[np.full(3, MASS), INERTIA])
    jacobian = np.array(((0, 0, 1, 0, 0, 0),), dtype=np.float64)
    result, force = _all_active_velocity(
        mass,
        np.zeros(6),
        np.array((0, 0, -MASS * GRAVITY, 0, 0, 0)),
        jacobian,
        np.array((4000.0,)),
        np.array((-0.001,)),
        H,
    )
    hand_force = 4000 * (0.001 + H * H * GRAVITY) / (1 + H * H * 4000 / MASS)
    assert force[0] == pytest.approx(hand_force, rel=1e-14)
    assert result[2] == pytest.approx(H * (hand_force / MASS - GRAVITY), rel=1e-13)
    np.testing.assert_array_equal(result[[0, 1, 3, 4, 5]], np.zeros(5))
    # Local y and world y are different in the registered tilted world.
    assert np.linalg.norm(rotation @ np.array((0, 1, 0)) - (0, 1, 0)) > 0.3
    r = np.array((0.003, -0.002, -0.01))
    ja = np.column_stack(
        (np.eye(3), -np.array(((0, -r[2], r[1]), (r[2], 0, -r[0]), (-r[1], r[0], 0))) @ rotation)
    )
    force_vector = np.array((0.0, 0.0, 3.0))
    np.testing.assert_allclose(
        ja.T @ force_vector,
        np.r_[force_vector, rotation.T @ np.cross(r, force_vector)],
        rtol=0,
        atol=1e-17,
    )


def _surface_audit(geometry, pair, native, body, plate_plane_z):
    surface, quad = geometry["surface"], pair["source_quadrature"]
    assert geometry["pressure_domain"] == "full"
    assert surface["identity"] == pair["identity"] == quad["identity"]
    assert surface["stamp"] == pair["stamp"] == quad["stamp"]
    for field in ("sample_ids", "source_ids", "dA"):
        assert quad[field] == pair[field]
    triangles = {triangle["triangle_id"]: triangle for triangle in surface["triangles"]}
    assert len(triangles) == len(surface["triangles"])
    assert len(pair["normal_force"]) == 3 * len(triangles)
    sample_points, measures, vertex_errors = [], [], []
    rotation, position = native["body_rotation"][body], native["body_position"][body]
    gradient = GRADIENT * rotation[:, 2]
    for index, (sample_id, source_id) in enumerate(
        zip(pair["sample_ids"], pair["source_ids"], strict=True)
    ):
        triangle_id, local_index = sample_id
        assert local_index == index % 3
        triangle = triangles[triangle_id]
        assert source_id == [triangle["compliant_cell_id"], triangle["rigid_face_id"]]
        assert triangle["compliant_cell_id"] in ("bottom-0", "bottom-1")
        vertices = np.asarray(triangle["vertices"], dtype=np.float64)
        np.testing.assert_allclose(vertices[:, 2], plate_plane_z, rtol=0, atol=1e-12)
        assert np.max(np.abs(vertices[:, :2])) <= 0.01 + 1e-12
        pressure, local, margin = _tent(vertices, position, rotation)
        assert np.all(local[:, 2] < 0) and np.all(margin > 1e-6)
        vertex_errors.append(_max_error(triangle["vertex_pressure"], pressure))
        np.testing.assert_allclose(triangle["vertex_pressure"], pressure, rtol=3e-4, atol=0.2)
        np.testing.assert_allclose(triangle["pressure_gradient"], gradient, rtol=3e-4, atol=2e-5)
        cross = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        assert cross[2] > 0
        area = float(np.linalg.norm(cross) / 2)
        sample_points.append(BARYCENTRIC[local_index] @ vertices)
        measures.append(area / 3)
    points = np.asarray(sample_points).reshape(-1, 3)
    measures = np.asarray(measures)
    np.testing.assert_allclose(
        np.asarray(quad["points"]).reshape(-1, 3), points, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(
        np.asarray(quad["points"], dtype=np.float32).reshape(-1, 3),
        np.asarray(pair["points"]).reshape(-1, 3),
    )
    quad_pressure, _, _ = _tent(points, position, rotation)
    np.testing.assert_allclose(quad["pressure"], quad_pressure, rtol=3e-4, atol=0.2)
    np.testing.assert_allclose(
        quad["gradient"], np.full(len(points), gradient[2]), rtol=3e-4, atol=2e-5
    )
    np.testing.assert_allclose(quad["dA"], measures, rtol=1e-10, atol=1e-14)
    assert quad["total_area_m2"] == pytest.approx(float(np.sum(measures)), rel=1e-10, abs=1e-14)
    assert np.sum(measures) <= 4e-4 * (1 + 1e-10)
    return dict(
        vertex_pressure_max_error_pa=max(vertex_errors, default=0),
        total_area_m2=float(np.sum(measures)),
    )


def _wrench_audit(world, pair, points, forces, before, step, native, body):
    vectors = forces[:, None] * np.array((0.0, 0.0, 1.0))
    total_force = np.sum(vectors, axis=0)
    expected_references = dict(
        world=np.zeros(3), body_origin=native["body_position"][body], com=native["body_com"][body]
    )
    maximum_force_error, maximum_moment_error = 0.0, 0.0
    for name, reference in expected_references.items():
        np.testing.assert_allclose(
            world["reference_points_world"][name], reference, rtol=0, atol=2e-7
        )
        stored_reference = np.asarray(world["reference_points_world"][name])
        expected_moment = np.sum(np.cross(points - stored_reference, vectors), axis=0)
        a, b = world["body_wrenches"][name]["pressure"], world["body_wrenches"][name]["plate"]
        for side, sign in ((a, 1), (b, -1)):
            assert side["input_tick"] == step and side["output_tick"] == step + 1
            assert side["time_s"] == before["time"]
            np.testing.assert_array_equal(side["reference_point_world"], stored_reference)
            assert len(side["contacts"]) == len(points)
            np.testing.assert_array_equal(side["native_force_world"], np.zeros(3))
            np.testing.assert_array_equal(side["native_moment_world"], np.zeros(3))
            np.testing.assert_allclose(
                side["total_force_world"], sign * total_force, rtol=0.002, atol=0.002
            )
            np.testing.assert_allclose(
                side["total_moment_world"], sign * expected_moment, rtol=0.002, atol=1e-5
            )
            maximum_force_error = max(
                maximum_force_error, _max_error(side["total_force_world"], sign * total_force)
            )
            maximum_moment_error = max(
                maximum_moment_error, _max_error(side["total_moment_world"], sign * expected_moment)
            )
            for i, contact in enumerate(side["contacts"]):
                assert contact["source"] == "area" and contact["identity"] == pair["identity"]
                assert (
                    contact["sample_id"] == pair["sample_ids"][i]
                    and contact["source_id"] == pair["source_ids"][i]
                )
                assert contact["area_m2"] == pair["dA"][i]
                assert contact["active"] == (pair["normal_force"][i] > 0)
                np.testing.assert_array_equal(contact["couple_world"], np.zeros(3))
        np.testing.assert_allclose(
            np.asarray(a["total_force_world"]) + b["total_force_world"], 0, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(a["total_moment_world"]) + b["total_moment_world"], 0, rtol=0, atol=1e-12
        )
    return maximum_force_error, maximum_moment_error


def _array_audit(arrays, row, before, after, pair, step, native, body, plate, receipt):
    assert arrays["entry_meta"][row] == [1, step, step + 1, 0, 0, 0]
    assert arrays["area_declared_mask"][row] == [1]
    assert not np.any(arrays["entry_contact_ids"][row]) and not np.any(
        arrays["entry_contact_data"][row]
    )
    np.testing.assert_array_equal(arrays["entry_q"][row][:7], before["q"])
    np.testing.assert_array_equal(arrays["entry_v"][row][:6], before["v"])
    np.testing.assert_array_equal(arrays["q"][row][:7], after["q"])
    np.testing.assert_array_equal(arrays["v"][row][:6], after["v"])
    clock = arrays["entry_clock"][row]
    assert clock[0] - clock[1] == before["time"] and clock[2] - clock[3] == after["time"]
    np.testing.assert_allclose(
        arrays["entry_body_position"][row], native["body_position"], rtol=0, atol=2e-7
    )
    np.testing.assert_allclose(
        arrays["entry_body_rotation"][row], native["body_rotation"], rtol=3e-4, atol=2e-6
    )
    vin, vout = np.asarray(before["v"]), np.asarray(after["v"])
    np.testing.assert_allclose(
        arrays["entry_body_linear_velocity"][row][body], vin[:3], rtol=0, atol=2e-7
    )
    np.testing.assert_allclose(
        arrays["entry_body_angular_velocity"][row][body],
        native["body_rotation"][body] @ vin[3:],
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        arrays["body_position"][row][body], after["q"][:3], rtol=0, atol=2e-7
    )
    np.testing.assert_allclose(
        arrays["body_angular_velocity"][row][body],
        _rotation(after["q"][3:]) @ vout[3:],
        rtol=2e-5,
        atol=2e-6,
    )
    for phase in ("entry_", ""):
        np.testing.assert_array_equal(
            arrays[phase + "body_linear_velocity"][row][plate], np.zeros(3)
        )
        np.testing.assert_array_equal(
            arrays[phase + "body_angular_velocity"][row][plate], np.zeros(3)
        )
    count = len(pair["normal_force"])
    assert arrays["area_meta"][row] == [
        1,
        count,
        sum(value > 0 for value in pair["normal_force"]),
        0,
    ]
    for i in range(count):
        assert arrays["area_ids"][row][i] == [1, 0, i, body, plate, 1]
    assert not np.any(arrays["area_ids"][row][count:])
    assert not np.any(arrays["area_data"][row][count:])
    assert not np.any(arrays["area_measure_words"][row][count:])
    words = np.asarray(arrays["area_measure_words"][row][:count], dtype="<i4")
    np.testing.assert_array_equal(np.frombuffer(words.tobytes(), dtype="<f8"), pair["dA"])
    raw = np.asarray(arrays["area_data"][row][:count], dtype=np.float32).reshape(count, 15)
    expected = np.column_stack(
        (
            np.asarray(pair["points"]).reshape(count, 3),
            np.asarray(pair["normals"]).reshape(count, 3),
            pair["stiffness"],
            pair["surrogate_gap"],
            pair["normal_force"],
            np.asarray(pair["force_on_a"]).reshape(count, 3),
            np.asarray(pair["torque_on_a_world_origin"]).reshape(count, 3),
        )
    )
    np.testing.assert_allclose(raw, expected, rtol=2e-5, atol=2e-7)
    assert not receipt["contacts"]
    assert len(receipt["constraints"]) == count
    assert all(constraint["constraint_type"] == 7 for constraint in receipt["constraints"])


def _world_audit(mj, model, body, plate, world, arrays, catalogs, row, step, audit):
    before, after, receipt = world["input"], world["output"], world["receipt"]
    assert before["tick"] == step and after["tick"] == step + 1
    assert receipt["tick"] == step + 1 and receipt["input_time"] == before["time"]
    assert receipt["output_time"] == after["time"] and receipt["contact_time"] == before["time"]
    assert receipt["contact_phase"] == "tyx-euler-input-constraint-evaluation"
    assert after["time"] > before["time"]
    np.testing.assert_array_equal(receipt["input_q"], before["q"])
    np.testing.assert_array_equal(receipt["input_v"], before["v"])
    control = receipt["stored_control"]
    assert not len(control["ctrl"])
    np.testing.assert_array_equal(control["qfrc_applied"], np.zeros(6))
    np.testing.assert_array_equal(control["xfrc_applied"], np.zeros((model.nbody, 6)))
    data, native = _native(mj, model, before, control)
    audit["native"] = native
    mass, dyn = native["mass_matrix"], receipt["dynamics"]
    assert np.all(np.linalg.eigvalsh(mass) > 0)
    np.testing.assert_allclose(
        np.asarray(dyn["mass_matrix"])[:3, :3], mass[:3, :3], rtol=3e-4, atol=2e-6
    )
    np.testing.assert_allclose(
        np.asarray(dyn["mass_matrix"])[3:, 3:], mass[3:, 3:], rtol=3e-4, atol=2e-9
    )
    for rows, cols in ((slice(0, 3), slice(3, 6)), (slice(3, 6), slice(0, 3))):
        np.testing.assert_allclose(
            np.asarray(dyn["mass_matrix"])[rows, cols], mass[rows, cols], rtol=3e-4, atol=2e-8
        )
    for name in ("bias", "passive", "applied"):
        np.testing.assert_allclose(dyn[name], native[name], rtol=3e-4, atol=2e-5)
    np.testing.assert_allclose(dyn["jacobian_com"], native["jacobian_com"], rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(dyn["input_q"], before["q"])
    np.testing.assert_array_equal(dyn["input_v"], before["v"])
    assert dyn["time"] == before["time"]
    pairs = receipt["raw_contact_fields"]["area_pressure"]
    assert len(pairs) == len(world["geometry"]) == len(catalogs) == 1
    pair, catalog, geometry = pairs[0], catalogs[0], world["geometry"][0]
    assert pair["body_ids"] == [body, plate]
    assert (
        pair["identity"]["compliant_body"] == "pressure"
        and pair["identity"]["rigid_body"] == "plate"
    )
    for name in ("identity", "stamp", "sample_ids", "source_ids"):
        assert pair[name] == catalog[name]
    assert pair["stamp"]["world_id"] == str(KEYS[row])
    assert pair["stamp"]["step_index"] == step and pair["stamp"]["sample_time_s"] == before["time"]
    # Geometry transforms the original FP64 local asset by the actual stored
    # entry pose. The static -0.0025 origin is FP32; its top is not exact z=0.
    # Keep the exact geometric plane check, and separately check that pose
    # against the independent native model with the frozen dynamics tolerance.
    plate_position = np.asarray(dyn["body_position"])[plate]
    plate_rotation = np.asarray(dyn["body_rotation"])[plate]
    np.testing.assert_allclose(plate_position, native["body_position"][plate], rtol=0, atol=2e-7)
    np.testing.assert_array_equal(plate_rotation, np.eye(3))
    plane_point = plate_position + plate_rotation @ np.array((0.0, 0.0, 0.0025))
    audit["geometry"] = _surface_audit(geometry, pair, native, body, plane_point[2])
    points = np.asarray(pair["points"], dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(pair["normals"], dtype=np.float64).reshape(-1, 3)
    count = len(points)
    np.testing.assert_allclose(normals, np.tile((0, 0, 1), (count, 1)), rtol=0, atol=1e-12)
    pressure, local, margin = _tent(
        points, native["body_position"][body], native["body_rotation"][body]
    )
    assert np.all(local[:, 2] < 0) and np.all(margin > 1e-6)
    gradient = GRADIENT * native["body_rotation"][body, :, 2]
    g = normals @ gradient
    assert np.all(g > 0)
    stiffness, gap = np.asarray(pair["dA"]) * g, -pressure / g
    np.testing.assert_allclose(pair["stiffness"], stiffness, rtol=3e-4, atol=2e-5)
    np.testing.assert_allclose(pair["surrogate_gap"], gap, rtol=3e-4, atol=2e-8)
    ja, jb = _point_jacobians(mj, model, data, points, body, plate)
    jacobian = np.einsum("ni,nij->nj", normals, ja - jb)
    np.testing.assert_allclose(
        np.asarray(pair["normal_jacobian"]).reshape(count, 6), jacobian, rtol=2e-5, atol=2e-6
    )
    vin, vout = np.asarray(before["v"]), np.asarray(after["v"])
    h = float(np.float32(receipt["h"]))
    assert h == float(np.float32(H))
    force = np.maximum(0, -stiffness * (gap + h * (jacobian @ vout)))
    actual_force = np.asarray(pair["normal_force"])
    assert np.all(actual_force >= 0)
    np.testing.assert_allclose(actual_force, force, rtol=0.001, atol=0.001)
    generalized = jacobian.T @ force
    rhs = native["applied"] + native["passive"] - native["bias"]
    acceleration = np.linalg.solve(mass, rhs + generalized)
    predicted_velocity = vin + h * acceleration
    np.testing.assert_allclose(vout[:3], predicted_velocity[:3], rtol=5e-4, atol=3e-5)
    np.testing.assert_allclose(vout[3:], predicted_velocity[3:], rtol=5e-4, atol=3e-4)
    np.testing.assert_allclose(
        receipt["qfrc_constraint"][:3], generalized[:3], rtol=0.002, atol=0.002
    )
    np.testing.assert_allclose(
        receipt["qfrc_constraint"][3:], generalized[3:], rtol=0.002, atol=1e-5
    )
    # Express the already-registered velocity gate in acceleration units.
    for field in ("qacc", "integration_acceleration"):
        np.testing.assert_allclose(
            np.asarray(receipt[field])[:3] * h, acceleration[:3] * h, rtol=5e-4, atol=3e-5
        )
        np.testing.assert_allclose(
            np.asarray(receipt[field])[3:] * h, acceleration[3:] * h, rtol=5e-4, atol=3e-4
        )
    np.testing.assert_allclose(
        vout, vin + h * np.asarray(receipt["integration_acceleration"]), rtol=2e-6, atol=2e-7
    )
    expected_q = np.array(before["q"], dtype=np.float64)
    mj.mj_integratePos(model, expected_q, np.array(vout, dtype=np.float64), h)
    actual_q = np.asarray(after["q"])
    np.testing.assert_allclose(actual_q[:3], expected_q[:3], rtol=0, atol=2e-7)
    sign = 1 if actual_q[3:] @ expected_q[3:] >= 0 else -1
    np.testing.assert_allclose(sign * actual_q[3:], expected_q[3:], rtol=0, atol=3e-7)
    assert abs(actual_q[3:] @ actual_q[3:] - 1) <= 2e-6
    _array_audit(arrays, row, before, after, pair, step, native, body, plate, receipt)
    fw_error, mw_error = _wrench_audit(world, pair, points, force, before, step, native, body)
    vectors = normals * actual_force[:, None]
    np.testing.assert_allclose(
        np.asarray(pair["force_on_a"]).reshape(-1, 3), vectors, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(pair["force_on_b"]).reshape(-1, 3), -vectors, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(pair["torque_on_a_world_origin"]).reshape(-1, 3),
        np.cross(points, vectors),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(pair["torque_on_b_world_origin"]).reshape(-1, 3),
        -np.cross(points, vectors),
        rtol=0,
        atol=1e-12,
    )
    moment = np.sum(np.cross(points - native["body_position"][body], vectors), axis=0)
    if step == 0 and KEYS[row] in (7, 305):
        assert count > 0
        closed_v, closed_f = _all_active_velocity(mass, vin, rhs, jacobian, stiffness, gap, h)
        np.testing.assert_allclose(vout[:3], closed_v[:3], rtol=5e-4, atol=3e-5)
        np.testing.assert_allclose(vout[3:], closed_v[3:], rtol=5e-4, atol=3e-4)
        np.testing.assert_allclose(actual_force, closed_f, rtol=0.001, atol=0.001)
        audit["first_all_active"] = dict(velocity=closed_v, force=closed_f)
        if KEYS[row] == 305:
            assert moment[1] > 0 and vout[4] > 0
        else:
            direction = np.array((-np.sin(0.35), np.cos(0.35), 0))
            assert moment @ direction < 0
            assert (native["body_rotation"][body] @ vout[3:]) @ direction < 0
    kinetic_in, kinetic_out = 0.5 * vin @ mass @ vin, 0.5 * vout @ mass @ vout
    potential_in = 0.5 * np.sum(stiffness * np.maximum(-gap, 0) ** 2)
    potential_out = 0.5 * np.sum(stiffness * np.maximum(-gap - h * (jacobian @ vout), 0) ** 2)
    audit.update(
        ja=ja,
        jb=jb,
        normal_jacobian=jacobian,
        pressure=pressure,
        gradient=gradient,
        stiffness=stiffness,
        gap=gap,
        force=force,
        generalized_force=generalized,
        acceleration=acceleration,
        predicted_velocity=predicted_velocity,
        integrated_q=expected_q,
        stored_h=h,
        moment_body_origin=moment,
        errors=dict(
            mass_translation=_max_error(np.asarray(dyn["mass_matrix"])[:3, :3], mass[:3, :3]),
            mass_rotation=_max_error(np.asarray(dyn["mass_matrix"])[3:, 3:], mass[3:, 3:]),
            jacobian_com=_max_error(dyn["jacobian_com"], native["jacobian_com"]),
            normal_jacobian=_max_error(
                np.asarray(pair["normal_jacobian"]).reshape(count, 6), jacobian
            ),
            bias=_max_error(dyn["bias"], native["bias"]),
            passive=_max_error(dyn["passive"], native["passive"]),
            applied=_max_error(dyn["applied"], native["applied"]),
            force=_max_error(actual_force, force),
            generalized_force=_max_error(receipt["qfrc_constraint"], generalized),
            linear_velocity=_max_error(vout[:3], predicted_velocity[:3]),
            angular_velocity=_max_error(vout[3:], predicted_velocity[3:]),
            position=_max_error(actual_q[:3], expected_q[:3]),
            quaternion=_max_error(sign * actual_q[3:], expected_q[3:]),
            public_force=fw_error,
            public_moment=mw_error,
        ),
        energy=dict(
            kinetic_input_j=float(kinetic_in),
            kinetic_output_j=float(kinetic_out),
            gravity_input_j=float(MASS * GRAVITY * before["q"][2]),
            gravity_output_j=float(MASS * GRAVITY * after["q"][2]),
            frozen_proxy_input_j=float(potential_in),
            frozen_proxy_output_j=float(potential_out),
            frozen_be_balance_j=float(
                kinetic_out - kinetic_in + potential_out - potential_in - h * (rhs @ vout)
            ),
            qualification="per-entry frozen-J/phi proxy; not global rotating-body potential",
        ),
    )
    return float(np.sum(actual_force)), count


def _audit(payload, summary):
    import mujoco as mj

    assert payload["schema"] == "free-body-area-support-v1"
    assert payload["mode"] == "cuda" and payload["status"] == "complete"
    assert payload["h"] == H and payload["steps"] == STEPS and payload["duration"] == 0.15
    assert payload["world_order"] == list(KEYS)
    assert (
        payload["accepted_steps"] == payload["complete_records"] == len(payload["records"]) == STEPS
    )
    assert _all_finite(payload)
    model = mj.MjModel.from_xml_string(payload["xml"])
    body, plate = _assert_model(mj, model)
    names = tuple(payload["body_names"])
    assert names.index("pressure") == body and names.index("plate") == plate
    previous = payload["initial"]
    corners = np.array([(x, y, -0.01) for x in (-0.02, 0.02) for y in (-0.02, 0.02)])
    metrics = {
        str(key): dict(
            forces=[], counts=[], angles=[], depths=[], linear_speeds=[], angular_speeds=[]
        )
        for key in KEYS
    }
    summary["world_metrics"] = metrics
    initial_q = {}
    for key in KEYS:
        state = previous[str(key)]
        initial_q[key] = np.asarray(state["q"])
        np.testing.assert_allclose(
            initial_q[key], _initial_q(key).astype(np.float32), rtol=0, atol=2e-7
        )
        np.testing.assert_array_equal(state["v"], np.zeros(6))
        assert state["tick"] == 0 and state["time"] == 0
    for step, record in enumerate(payload["records"]):
        assert (
            record["step"] == step and record["state_accepted"] and record["observation_complete"]
        )
        assert record["readback_calls"] == 1
        assert record["accepted"] == {str(key): True for key in KEYS}
        assert set(record["worlds"]) == set(map(str, KEYS))
        checked = dict(step=step, worlds={})
        summary["native_oracles"].append(checked)
        for row, key in enumerate(KEYS):
            world = record["worlds"][str(key)]
            assert world["input"] == previous[str(key)]
            audit = {}
            checked["worlds"][str(key)] = audit
            total, count = _world_audit(
                mj,
                model,
                body,
                plate,
                world,
                record["arrays"],
                record["catalogs"][str(key)],
                row,
                step,
                audit,
            )
            values = metrics[str(key)]
            values["forces"].append(total)
            values["counts"].append(count)
            for state in (world["input"], world["output"]):
                q, v = np.asarray(state["q"]), np.asarray(state["v"])
                rotation = _rotation(q[3:])
                bottom = q[:3] + corners @ rotation.T
                depth = max(0.0, -float(np.min(bottom[:, 2])))
                angle = _angle(q[3:], initial_q[key][3:])
                values["angles"].append(angle)
                values["depths"].append(depth)
                values["linear_speeds"].append(float(np.linalg.norm(v[:3])))
                values["angular_speeds"].append(float(np.linalg.norm(v[3:])))
                assert 0.003 < q[2] < 0.03
                assert np.linalg.norm(v[:3]) < 0.5 and np.linalg.norm(v[3:]) < 20
                assert angle < 0.15 and depth < 0.003
                np.testing.assert_allclose(q[:2], initial_q[key][:2], rtol=0, atol=5e-6)
            if step == 0 and key == 41:
                assert count == 0 and total == 0
        previous = {str(key): record["worlds"][str(key)]["output"] for key in KEYS}
        summary["audited_steps"] = step + 1
    assert payload["final"] == previous
    assert all(state["tick"] == STEPS for state in previous.values())
    assert metrics["41"]["counts"][0] == 0 and max(metrics["41"]["forces"]) > 0
    assert max(metrics["41"]["angles"]) <= 2e-4
    assert max(metrics["7"]["angles"]) > 1e-4
    assert max(metrics["305"]["angles"]) > 1e-3
    summary["max_errors"] = {
        name: max(
            world["errors"][name]
            for entry in summary["native_oracles"]
            for world in entry["worlds"].values()
        )
        for name in (
            "mass_translation",
            "mass_rotation",
            "jacobian_com",
            "normal_jacobian",
            "bias",
            "passive",
            "applied",
            "force",
            "generalized_force",
            "linear_velocity",
            "angular_velocity",
            "position",
            "quaternion",
            "public_force",
            "public_moment",
        )
    }
    summary["checks"] = dict(
        full_window=True,
        real_center_contact_birth=True,
        full_six_dof_native_terms=True,
        independently_reconstructed_pressure_force=True,
        shared_velocity_equation=True,
        native_position_integration=True,
        world_and_local_velocity_frames=True,
        both_bodies_three_reference_wrenches=True,
        same_phase_catalog_and_device_arrays=True,
        bounded_unconstrained_motion=True,
        actual_tilt_and_offset_rotation=True,
    )


@pytest.mark.cuda
def test_one_saved_free_body_window_has_independent_native_dynamics_and_wrenches(tmp_path):
    trajectory = tmp_path / "free-body-support.json.gz"
    checked_path = tmp_path / "checked-summary.json"
    summary = dict(
        schema="free-body-area-support-independent-checks-v1",
        status="started",
        trajectory=str(trajectory),
        audited_steps=0,
        native_oracles=[],
        qualification="real CUDA normal contact; HOST current geometry; no friction or static equilibrium claim",
        energy_qualification="frozen-J/phi balances only; no global rotating-body elastic potential claim",
    )
    try:
        returned = experiment.run_case(trajectory, mode="cuda")
        assert trajectory.is_file(), "save the one actual trajectory before its physics audit"
        with gzip.open(trajectory, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        assert payload == returned
        _audit(payload, summary)
        summary["status"] = "passed"
    except BaseException as error:
        summary["status"] = "failed"
        summary["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        with checked_path.open("x", encoding="utf-8") as stream:
            json.dump(
                _plain(summary), stream, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
