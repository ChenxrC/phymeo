"""Tests first for opt-in fixed tooling on a real free-root robot.

Only the existing public loader is imported at collection time. The first RED
must be its missing collapse_fixed_joints behavior, after the old explicit
fixed-joint rejection succeeds. The fixture constants are independent inputs.
"""

from pathlib import Path

import numpy as np
import pytest

from phymeo.builders.floating_urdf import load_floating_urdf
from phymeo.dynamics.floating_robot import FloatingRobotState, floating_kinematics
from phymeo.solvers.floating_tree import FloatingRobotInputs, step_floating_robot

ASSETS = Path(__file__).parent / "fixtures" / "urdf"
COMPOSITE = ASSETS / "floating_fixed_composite.urdf"
PROBE = ASSETS / "floating_fixed_probe.urdf"
H = 1 / 1024
ATOL = 2e-12


def _root_state():
    return FloatingRobotState(
        np.array([0, 0, 0, 0, 0, 0, 1], dtype=float),
        np.zeros(6),
    )


def test_a01_public_opt_in_keeps_real_root_mass_and_point_load_response():
    with pytest.raises(ValueError, match="fixed"):
        load_floating_urdf(COMPOSITE, gravity=(0, 0, 0))

    asset = load_floating_urdf(COMPOSITE, gravity=(0, 0, 0), collapse_fixed_joints=True)
    assert asset.joint_names == () and asset.link_names == ()
    assert asset.model.children is None and asset.model.body_count == 1
    np.testing.assert_allclose(asset.model.root.inverse_mass, [1 / 3], atol=ATOL, rtol=0)
    np.testing.assert_allclose(asset.model.root_com_m, [1 / 12, 0, 0], atol=ATOL, rtol=0)
    np.testing.assert_allclose(
        asset.model.root.inertia_diagonal,
        [[3 / 10, 53 / 120, 13 / 24]],
        atol=ATOL,
        rtol=0,
    )

    initial = _root_state()
    # F=(0,3,0) at the authored tool origin x=1/4. Relative to
    # Cx=1/12 this has torque_z=1/2; a_y=1, alpha_z=12/13.
    applied = FloatingRobotInputs(
        np.zeros(0), np.array([[0.0, 3.0, 0.0]]), np.array([[0.0, 0.0, 0.5]])
    )
    result = step_floating_robot(asset.model, initial, applied, H)
    assert result.report.accepted
    np.testing.assert_allclose(result.state.v, [0, H, 0, 0, 0, H * 12 / 13], atol=ATOL, rtol=0)
    theta = H * H * 12 / 13
    np.testing.assert_allclose(
        result.state.q,
        [
            (1 - np.cos(theta)) / 12,
            H * H - np.sin(theta) / 12,
            0,
            0,
            0,
            np.sin(theta / 2),
            np.cos(theta / 2),
        ],
        atol=ATOL,
        rtol=0,
    )
    geometry = floating_kinematics(asset.model, result.state)
    np.testing.assert_allclose(geometry.com_position[0], [1 / 12, H * H, 0], atol=ATOL, rtol=0)
    np.testing.assert_array_equal(initial.q, _root_state().q)
    np.testing.assert_array_equal(initial.v, np.zeros(6))


# F02 constants come from the five original components, never from loaded model
# arrays. F03 is a passive elastic contact window, not a prescribed withdrawal.
WINDOW = {
    "duration_s": 1 / 4,
    "dt_s": (1 / 2048, 1 / 4096),
    "joint_q": (0.0, 1 / 8),
    "root_velocity_m_s": (0.0, 1 / 4, 0.0),
    "sphere_mass_kg": 1.0,
    "sphere_radius_m": 1 / 16,
    "initial_gap_m": 1 / 1024,
    "stiffness_n_per_m": 2048.0,
}
CYCLE = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])


def _rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _skew(p):
    x, y, z = p
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _probe_oracle(joints, position, rotation):
    """Five independent masses, direct serial transforms, world COM basis."""
    theta, extension = joints
    root = np.asarray(position)
    arm_r = rotation @ _rz(theta)
    tool_r = arm_r @ _rz(np.pi / 2)
    pivot = root + rotation @ np.array([1 / 4, 0, 0])
    carriage = pivot + arm_r @ np.array([1 / 2 + extension, 0, 0])
    tool = carriage + arm_r @ np.array([1 / 8, 0, 0])
    frames = {
        "base": (root, rotation, 0),
        "mount": (pivot, rotation, 0),
        "arm": (pivot, arm_r, 1),
        "carriage": (carriage, arm_r, 2),
        "tool": (tool, tool_r, 2),
        "witness": (tool + tool_r @ np.array([1 / 32, 0, 0]), tool_r, 2),
    }
    root_com = root + rotation @ np.array([1 / 20, 0, 0])
    hinge_axis = rotation[:, 2]
    slide_axis = arm_r[:, 0]

    def jacobian(point, body):
        linear, angular = np.zeros((3, 8)), np.zeros((3, 8))
        linear[:, :3] = np.eye(3)
        linear[:, 3:6] = -_skew(point - root_com)
        angular[:, 3:6] = np.eye(3)
        if body > 0:
            linear[:, 6] = np.cross(hinge_axis, point - pivot)
            angular[:, 6] = hinge_axis
        if body == 2:
            linear[:, 7] = slide_axis
        return linear, angular

    definitions = (
        ("base", 2.0, (0, 0, 0), (1 / 5, 3 / 10, 2 / 5)),
        ("mount", 1 / 2, (0, 0, 0), (1 / 50, 3 / 100, 1 / 25)),
        ("arm", 1.0, (1 / 4, 0, 0), (1 / 20, 3 / 40, 1 / 10)),
        ("carriage", 1 / 2, (0, 0, 0), (1 / 40, 1 / 40, 1 / 40)),
        ("tool", 1 / 4, (0, 0, 0), (1 / 100, 1 / 50, 3 / 100)),
    )
    parts, mass = [], np.zeros((8, 8))
    for name, weight, local_com, diagonal in definitions:
        origin, orient, body = frames[name]
        center = origin + orient @ np.array(local_com)
        inertia = orient @ np.diag(diagonal) @ orient.T
        linear, angular = jacobian(center, body)
        mass += weight * linear.T @ linear + angular.T @ inertia @ angular
        parts.append((body, weight, center, inertia, linear, angular))
    groups = []
    for body in range(3):
        entries = [p for p in parts if p[0] == body]
        weight = sum(p[1] for p in entries)
        center = sum(p[1] * p[2] for p in entries) / weight
        inertia = np.zeros((3, 3))
        for _, m, point, tensor, _, _ in entries:
            arm = point - center
            inertia += tensor + m * ((arm @ arm) * np.eye(3) - np.outer(arm, arm))
        groups.append((weight, center, inertia))
    return frames, tuple(parts), tuple(groups), mass, jacobian


@pytest.mark.parametrize(
    "joints,position,xyzw,rotation",
    [
        ((0.0, 1 / 8), (0, 0, 0), (0, 0, 0, 1), np.eye(3)),
        ((0.37, -1 / 16), (1 / 2, -1 / 4, 3 / 4), (0.5, 0.5, 0.5, 0.5), CYCLE),
        ((-0.51, 3 / 16), (-1 / 4, 1 / 2, 0), (0, 0, 0, 1), np.eye(3)),
    ],
)
def test_a02_probe_preserves_five_mass_geometry_full_inertia_and_tool_work(
    joints, position, xyzw, rotation
):
    from phymeo.dynamics.floating_robot import floating_point_jacobians
    from phymeo.solvers.floating_tree import floating_dynamics_terms

    asset = load_floating_urdf(PROBE, gravity=(0, 0, 0), collapse_fixed_joints=True)
    assert asset.link_names == ("arm", "carriage")
    assert asset.joint_names == ("hinge", "extension")
    assert asset.lower == (-1.5, -1 / 8) and asset.upper == (1.5, 1 / 4)
    assert asset.effort_limit == (16, 32) and asset.velocity_limit == (4, 2)
    assert asset.model.body_count == 3
    state = FloatingRobotState(
        np.array((*position, *xyzw, *joints), dtype=float),
        np.array([0.1, -0.2, 0.3, -0.1, 0.2, 0.1, 0.4, -0.3]),
    )
    frames, parts, groups, expected_mass, point_j = _probe_oracle(joints, position, rotation)
    geometry = floating_kinematics(asset.model, state)
    np.testing.assert_allclose(asset.model.root_com_m, [1 / 20, 0, 0], atol=ATOL, rtol=0)
    np.testing.assert_allclose(
        asset.model.root.inertia_diagonal, [[11 / 50, 71 / 200, 93 / 200]], atol=ATOL, rtol=0
    )
    observed_mass = np.r_[1 / asset.model.root.inverse_mass, asset.model.children.mass]
    observed_inertia = np.vstack(
        (asset.model.root.inertia_diagonal, asset.model.children.inertia_diagonal)
    )
    for i, (mass, center, tensor) in enumerate(groups):
        assert observed_mass[i] == pytest.approx(mass, abs=ATOL)
        np.testing.assert_allclose(geometry.com_position[i], center, atol=ATOL, rtol=0)
        np.testing.assert_allclose(
            geometry.rotation[i] @ np.diag(observed_inertia[i]) @ geometry.rotation[i].T,
            tensor,
            atol=ATOL,
            rtol=0,
        )
    for name, (point, orient, owner) in frames.items():
        assert asset.sites[name].body_index == owner
        p, r = asset.authored_link_pose(name, state.q)
        np.testing.assert_allclose(p, point, atol=ATOL, rtol=0)
        np.testing.assert_allclose(r, orient, atol=ATOL, rtol=0)
        mapped = asset.authored_to_model_points(name, [1 / 32, -1 / 64, 1 / 128])
        np.testing.assert_allclose(
            geometry.position[owner] + geometry.rotation[owner] @ mapped,
            point + orient @ np.array([1 / 32, -1 / 64, 1 / 128]),
            atol=ATOL,
            rtol=0,
        )
    terms = floating_dynamics_terms(asset.model, state, FloatingRobotInputs.zeros(2))
    np.testing.assert_allclose(terms.mass_matrix, expected_mass, atol=1e-11, rtol=0)
    expected_energy = sum(
        0.5 * m * np.dot(j @ state.v, j @ state.v) + 0.5 * (a @ state.v) @ tensor @ (a @ state.v)
        for _, m, _, tensor, j, a in parts
    )
    assert 0.5 * state.v @ terms.mass_matrix @ state.v == pytest.approx(expected_energy, abs=1e-11)
    local = np.array([[1 / 32, -1 / 64, 0.0]])
    point = frames["tool"][0] + frames["tool"][1] @ local[0]
    expected_j, expected_a = point_j(point, 2)
    linear, angular = floating_point_jacobians(
        asset.model, state, [2], asset.authored_to_model_points("tool", local)
    )
    np.testing.assert_allclose(linear[0], expected_j, atol=ATOL, rtol=0)
    np.testing.assert_allclose(angular[0], expected_a, atol=ATOL, rtol=0)
    force = rotation @ np.array([-1.0, 2.0, 0.5])
    assert float(force @ (linear[0] @ state.v)) == pytest.approx(
        float((expected_j.T @ force) @ state.v), abs=ATOL
    )
    # At zero velocity/gravity the independently assembled M and applied point
    # wrench determine acceleration without using any product-generated bias.
    rest = FloatingRobotState(state.q, np.zeros(8))
    loads = FloatingRobotInputs(
        np.array([1 / 4, -1 / 2]),
        np.vstack((np.zeros((2, 3)), force)),
        np.vstack((np.zeros((2, 3)), np.cross(point - groups[2][1], force))),
    )
    loaded = floating_dynamics_terms(asset.model, rest, loads)
    expected_applied = expected_j.T @ force + np.r_[np.zeros(6), [1 / 4, -1 / 2]]
    np.testing.assert_allclose(loaded.applied, expected_applied, atol=ATOL, rtol=0)
    np.testing.assert_allclose(
        loaded.acceleration, np.linalg.solve(expected_mass, expected_applied), atol=1e-11, rtol=0
    )
    np.testing.assert_allclose(loaded.bias, 0, atol=ATOL, rtol=0)


def test_a04_asset_constructor_rejects_duplicate_shape_keys_and_fake_root_frame():
    from dataclasses import replace

    from phymeo.builders._floating_fixed_asset import FloatingFixedSite

    asset = load_floating_urdf(COMPOSITE, gravity=(0, 0, 0), collapse_fixed_joints=True)
    with pytest.raises(ValueError, match="unique|keys"):
        replace(asset, collisions=(asset.collisions[0], asset.collisions[0]))
    for site in (
        FloatingFixedSite(0, (1 / 64, 0, 0)),
        FloatingFixedSite(0, (0, 0, 0), (0, 0, 1, 0)),
    ):
        sites = dict(asset.sites)
        sites[asset.root_link] = site
        with pytest.raises(ValueError, match="root.*origin|root.*frame"):
            replace(asset, sites=sites)
    with pytest.raises(ValueError, match="body|joint|metadata"):
        replace(asset, link_names=("invented",), joint_names=("dummy",))
    sites = dict(asset.sites)
    sites[asset.root_link] = FloatingFixedSite(0, (0, 0, 0), (0, 0, 0, -1))
    equivalent = replace(asset, sites=sites)
    for name in sites:
        for actual, expected in zip(
            equivalent.authored_link_pose(name, _root_state().q),
            asset.authored_link_pose(name, _root_state().q),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("transformed", [False, True])
def test_a01_composite_quarter_second_point_force_follows_independent_recurrence(transformed):
    asset = load_floating_urdf(COMPOSITE, gravity=(0, 0, 0), collapse_fixed_joints=True)
    rotation = CYCLE if transformed else np.eye(3)
    shift = np.array([1 / 2, -1 / 4, 3 / 4]) if transformed else np.zeros(3)
    quaternion = (0.5, 0.5, 0.5, 0.5) if transformed else (0, 0, 0, 1)
    state = FloatingRobotState(np.r_[shift, quaternion], np.zeros(6))
    force = rotation @ np.array([0.0, 3.0, 0.0])
    spin, angle = 0.0, 0.0
    for step in range(1, 257):
        geometry = floating_kinematics(asset.model, state)
        point, _ = asset.authored_link_pose("tool", state.q)
        torque = np.cross(point - geometry.com_position[0], force)
        load = FloatingRobotInputs(np.zeros(0), force[None], torque[None])
        out = step_floating_robot(asset.model, state, load, H)
        assert out.report.accepted
        spin += H * (12 / 13) * np.cos(angle)
        angle += H * spin
        expected_rotation = rotation @ _rz(angle)
        expected_com = shift + rotation @ np.array([1 / 12, step * (step + 1) * H * H / 2, 0])
        current = floating_kinematics(asset.model, out.state)
        np.testing.assert_allclose(current.rotation[0], expected_rotation, atol=1e-11, rtol=0)
        np.testing.assert_allclose(current.com_position[0], expected_com, atol=1e-11, rtol=0)
        np.testing.assert_allclose(
            out.state.q[:3],
            expected_com - expected_rotation @ [1 / 12, 0, 0],
            atol=1e-11,
            rtol=0,
        )
        np.testing.assert_allclose(out.state.v[:3], rotation @ [0, step * H, 0], atol=ATOL, rtol=0)
        np.testing.assert_allclose(out.state.v[3:6], rotation @ [0, 0, spin], atol=1e-11, rtol=0)
        state = out.state
    np.testing.assert_allclose(3 * state.v[:3], (1 / 4) * force, atol=ATOL, rtol=0)
    from phymeo.dynamics.floating_robot import floating_point_jacobians

    linear, _ = floating_point_jacobians(
        asset.model, state, [0], asset.authored_to_model_points("tool", [[0, 0, 0]])
    )
    expected_point_v = rotation @ (
        np.array([0, 1 / 4, 0]) + np.cross([0, 0, spin], _rz(angle) @ [1 / 6, 0, 0])
    )
    np.testing.assert_allclose(linear[0] @ state.v, expected_point_v, atol=1e-11, rtol=0)
    assert float((linear[0].T @ force) @ state.v) == pytest.approx(
        float(force @ expected_point_v), abs=1e-11
    )
    print("F01 actual quarter-second", transformed, "q", state.q.tolist(), "v", state.v.tolist())


@pytest.mark.parametrize(
    "bad,pattern",
    [
        ("massless_group", "positive mass"),
        ("off_axis_root", "non-diagonal"),
        ("invalid_inertia", "inertia"),
        ("fixed_control", "unsupported|dynamics"),
        ("mimic", "mimic"),
        ("massless_child", "positive mass"),
    ],
)
def test_a04_invalid_mass_inertia_and_control_are_explicit(tmp_path, bad, pattern):
    from xml.etree import ElementTree as ET

    doc = ET.parse(PROBE if bad == "massless_child" else COMPOSITE)
    if bad == "massless_group":
        for link in doc.findall("./link"):
            inertial = link.find("inertial")
            if inertial is not None:
                link.remove(inertial)
    elif bad == "off_axis_root":
        doc.find(".//joint/origin").set("xyz", "0.25 0.125 0")
    elif bad == "invalid_inertia":
        doc.find(".//link[@name='tool']/inertial/inertia").set("ixx", "1")
    elif bad == "fixed_control":
        ET.SubElement(doc.find(".//joint"), "dynamics", damping="0.1")
    elif bad == "mimic":
        ET.SubElement(doc.find(".//joint"), "mimic", joint="not_a_physical_dof")
    else:
        link = doc.find(".//link[@name='arm']")
        link.remove(link.find("inertial"))
    path = tmp_path / f"{bad}.urdf"
    doc.write(path, encoding="unicode")
    with pytest.raises(ValueError, match=pattern):
        load_floating_urdf(path, gravity=(0, 0, 0), collapse_fixed_joints=True)


@pytest.mark.parametrize("bad", [None, 0, 1, "true", [], np.array([True])])
def test_a04_opt_in_requires_an_actual_boolean(bad):
    with pytest.raises((ValueError, TypeError), match="bool|collapse"):
        load_floating_urdf(COMPOSITE, collapse_fixed_joints=bad)


def test_a04_massless_original_root_is_valid_when_its_fixed_group_has_mass(tmp_path):
    from xml.etree import ElementTree as ET

    doc = ET.parse(COMPOSITE)
    link = doc.find(".//link[@name='base']")
    link.remove(link.find("inertial"))
    path = tmp_path / "mass_in_tool.urdf"
    doc.write(path, encoding="unicode")
    asset = load_floating_urdf(path, gravity=(0, 0, 0), collapse_fixed_joints=np.bool_(True))
    np.testing.assert_allclose(asset.model.root.inverse_mass, [1], atol=ATOL, rtol=0)
    np.testing.assert_allclose(asset.model.root_com_m, [1 / 4, 0, 0], atol=ATOL, rtol=0)
    np.testing.assert_allclose(asset.model.root.inertia_diagonal, [[0.1] * 3], atol=ATOL, rtol=0)
    assert asset.model.children is None and asset.sites["base"].body_index == 0


def test_a04_owned_sites_outputs_and_source_arrays_cannot_move_an_asset():
    from dataclasses import replace

    from phymeo.builders._floating_fixed_asset import FloatingFixedSite

    asset = load_floating_urdf(COMPOSITE, gravity=(0, 0, 0), collapse_fixed_joints=True)
    position, quaternion = np.array([1 / 4, 0, 0]), np.array([0.0, 0, 0, 1])
    site = FloatingFixedSite(0, position, quaternion)
    mapping = dict(asset.sites)
    mapping["tool"] = site
    copied = replace(asset, sites=mapping)
    position[:] = 99
    quaternion[:] = 0
    mapping.clear()
    assert copied.sites["tool"].local_position_m == (1 / 4, 0, 0)
    with pytest.raises(TypeError):
        copied.sites["tool"] = copied.sites["base"]
    result = copied.authored_to_model_points("tool", np.zeros((2, 3)))
    result[:] = 12
    p, r = copied.authored_link_pose("tool", _root_state().q)
    p[:] = 99
    r[:] = 0
    np.testing.assert_allclose(
        copied.authored_to_model_points("tool", [0, 0, 0]), [1 / 4, 0, 0], atol=ATOL, rtol=0
    )
    with pytest.raises(ValueError, match="owner"):
        bad_sites = dict(asset.sites)
        bad_sites["tool"] = FloatingFixedSite(1)
        replace(asset, sites=bad_sites)
    for points in ([np.nan, 0, 0], [1, 2], ["a", "b", "c"]):
        with pytest.raises(ValueError):
            asset.authored_to_model_points("tool", points)
    with pytest.raises(ValueError, match="interval"):
        asset.collision_motion(_root_state().q, _root_state().q)


def test_a02_curved_tool_world_orientation_uses_its_authored_fixed_frame(tmp_path):
    from xml.etree import ElementTree as ET

    from phymeo.dynamics.rotation import quaternion_matrix

    doc = ET.parse(COMPOSITE)
    geometry = doc.find(".//link[@name='tool']/collision/geometry")
    geometry.remove(geometry.find("box"))
    ET.SubElement(geometry, "cylinder", radius="0.0625", length="0.25")
    doc.find(".//joint/origin").set("rpy", "1.5707963267948966 0 0")
    path = tmp_path / "rotated_cylinder.urdf"
    doc.write(path, encoding="unicode")
    asset = load_floating_urdf(path, gravity=(0, 0, 0), collapse_fixed_joints=True)
    q = np.array([1 / 2, -1 / 4, 3 / 4, 0.5, 0.5, 0.5, 0.5])
    collider = next(col for col in asset.collisions if col.link_name == "tool")
    pose = asset.collision_motion(q)[collider.shape.key]
    fixed = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    np.testing.assert_allclose(
        quaternion_matrix(np.array([pose.body_orientation_xyzw]))[0],
        CYCLE @ fixed,
        atol=ATOL,
        rtol=0,
    )
    np.testing.assert_allclose(pose.start[0], q[:3] + CYCLE @ [1 / 4, 0, 0], atol=ATOL, rtol=0)


def _example():
    from importlib.util import module_from_spec, spec_from_file_location

    path = Path(__file__).resolve().parents[1] / "examples" / "floating_fixed_tools.py"
    spec = spec_from_file_location("floating_fixed_tools_example", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def contact_windows():
    example = _example()
    return {
        "h": example.run_contact_window(dt_s=1 / 2048),
        "half": example.run_contact_window(dt_s=1 / 4096),
        "reverse": example.run_contact_window(dt_s=1 / 2048, reverse=True),
        "transform": example.run_contact_window(dt_s=1 / 2048, transformed=True),
    }


def test_a03_actual_entry_load_separation_impulse_and_common_point_power(contact_windows):
    for label, result in contact_windows.items():
        frames, summary = result["frames"], result["summary"]
        h = result["protocol"]["dt_s"]
        assert len(frames) == round((1 / 4) / h)
        assert summary["time_s"] == pytest.approx(1 / 4, abs=ATOL)
        assert not frames[0]["sites"] and summary["has_begin"] and summary["has_end"]
        assert 0 < summary["first_positive_load_time_s"] < summary["last_positive_load_time_s"]
        assert summary["last_positive_load_time_s"] < 1 / 4 - 0.01
        assert summary["final_unloaded"] and summary["peak_force_n"] > 0.1
        assert summary["max_linear_momentum_residual_ns"] <= 5e-4
        impulse = np.zeros(3)
        for frame in frames:
            force = np.asarray(frame["ball_force_n"])
            torque = np.asarray(frame["ball_torque_nm"])
            wrenches = np.asarray(frame["link_wrenches_world"])
            impulse += h * force
            np.testing.assert_allclose(frame["ball_impulse_ns"], impulse, atol=1e-10, rtol=0)
            np.testing.assert_allclose(frame["ball_velocity_m_s"], impulse, atol=1e-10, rtol=0)
            np.testing.assert_allclose(wrenches[:, :3].sum(axis=0) + force, 0, atol=1e-10, rtol=0)
            moment = np.cross(np.asarray(frame["entry_body_com_m"]), wrenches[:, :3]).sum(axis=0)
            moment += wrenches[:, 3:].sum(axis=0)
            moment += np.cross(frame["entry_ball_position_m"], force) + torque
            np.testing.assert_allclose(moment, 0, atol=1e-10, rtol=0)
            actual_power = (
                np.dot(frame["contact_effort_full"], frame["entry_v"])
                + np.dot(force, frame["entry_ball_velocity_m_s"])
                + np.dot(torque, frame["entry_ball_omega_rad_s"])
            )
            assert actual_power == pytest.approx(
                sum(site["power_w"] for site in frame["sites"]), abs=1e-10
            )
            for site in frame["sites"]:
                np.testing.assert_allclose(
                    site["application_first_m"], site["application_second_m"], atol=1e-10, rtol=0
                )
                assert site["first"][0] == site["second"][0] == 7
            for material in frame["materials"]:
                assert {material["first_owner"][1], material["second_owner"][1]} == {101, 205}
                for name in ("first_owner", "second_owner"):
                    if material[name][1] == 101:
                        assert material[name][2] == 2
            np.testing.assert_allclose(frame["drive_effort"], [0, 0], atol=ATOL, rtol=0)
            assert frame["event_sample_time_s"] == frame["load_time_s"]
            assert frame["event_commit_time_s"] == frame["time_s"]
        original = np.asarray(frames[0]["entry_v"])
        assert max(np.linalg.norm(np.asarray(f["v"])[:6] - original[:6]) for f in frames) > 1e-5
        assert max(np.linalg.norm(np.asarray(f["v"])[6:]) for f in frames) > 1e-5
        print("F03", label, summary)


def test_a03_h_and_half_h_keep_the_same_physical_task_with_bounded_error(contact_windows):
    a, b = contact_windows["h"]["frames"][-1], contact_windows["half"]["frames"][-1]
    for field in ("body_com_m", "ball_position_m"):
        assert np.max(np.abs(np.asarray(a[field]) - b[field])) <= 5e-4
    for field in ("body_com_velocity_m_s", "ball_velocity_m_s"):
        assert np.max(np.abs(np.asarray(a[field]) - b[field])) <= 5e-3
    assert np.max(np.abs(np.asarray(a["v"])[3:] - np.asarray(b["v"])[3:])) <= 1e-2
    assert np.linalg.norm(np.asarray(a["ball_impulse_ns"]) - b["ball_impulse_ns"]) <= 5e-3


def test_a03_reversed_registration_and_world_covariance_preserve_real_response(contact_windows):
    original = contact_windows["h"]["frames"]
    reverse = contact_windows["reverse"]["frames"]
    moved = contact_windows["transform"]["frames"]
    shift = np.array([1 / 2, -1 / 4, 3 / 4])
    for a, b, c in zip(original, reverse, moved, strict=True):
        for field in ("q", "v", "ball_position_m", "ball_velocity_m_s", "ball_force_n"):
            np.testing.assert_allclose(a[field], b[field], atol=1e-11, rtol=0)
        for field in ("body_com_m", "ball_position_m"):
            np.testing.assert_allclose(
                np.asarray(a[field]) @ CYCLE.T + shift, c[field], atol=1e-10, rtol=0
            )
        for field in (
            "body_com_velocity_m_s",
            "ball_velocity_m_s",
            "ball_force_n",
            "ball_impulse_ns",
        ):
            np.testing.assert_allclose(np.asarray(a[field]) @ CYCLE.T, c[field], atol=1e-10, rtol=0)
        np.testing.assert_allclose(
            np.asarray(a["v"])[6:], np.asarray(c["v"])[6:], atol=1e-10, rtol=0
        )
        assert a["events"] == b["events"] == c["events"]


def _unchanged(scene, reading, geometry, events):
    current, current_geometry = scene.observe_scene()
    assert (current.step_number, current.time_s, current.load_time_s) == (
        reading.step_number,
        reading.time_s,
        reading.load_time_s,
    )
    assert scene.contact_events is events
    for address in reading.robots:
        np.testing.assert_array_equal(current.robots[address].q, reading.robots[address].q)
        np.testing.assert_array_equal(current.robots[address].v, reading.robots[address].v)
    for field in ("position", "orientation", "velocity", "angular_velocity"):
        np.testing.assert_array_equal(getattr(current.rigid, field), getattr(reading.rigid, field))
    assert current_geometry.time_s == geometry.time_s
    for key in geometry.poses:
        np.testing.assert_array_equal(current_geometry.poses[key].start, geometry.poses[key].start)


def test_a04_late_material_failure_has_no_state_event_or_clock_commit(monkeypatch):
    example = _example()
    with (
        example.build_contact_scene(gap_m=-1 / 1024) as scene,
        example.build_contact_scene(gap_m=-1 / 1024) as fresh,
    ):
        old, geometry = scene.observe_scene()
        events = scene.contact_events

        def fail(*args, **kwargs):
            raise RuntimeError("fixed-tool material staging fault")

        with monkeypatch.context() as patcher:
            patcher.setattr(scene._event_adapter, "stage", fail)
            with pytest.raises(RuntimeError, match="material staging fault"):
                scene.step(H, actions={example.ROBOT: example.zero_drive()})
        _unchanged(scene, old, geometry, events)
        assert scene.last_frame is None
        actual = scene.step(H, actions={example.ROBOT: example.zero_drive()})
        expected = fresh.step(H, actions={example.ROBOT: example.zero_drive()})
        assert actual.loads.contacts and actual.loads.contacts[0].force_n > 0
        for address in actual.robots:
            np.testing.assert_array_equal(actual.robots[address].q, expected.robots[address].q)
            np.testing.assert_array_equal(actual.robots[address].v, expected.robots[address].v)
        np.testing.assert_array_equal(actual.rigid.velocity, expected.rigid.velocity)
        assert scene.contact_events == fresh.contact_events
        a, b = scene.assets[7, 101], fresh.assets[7, 101]
        assert not np.shares_memory(a.model.root_com_m, b.model.root_com_m)
        assert a.sites is not b.sites


def test_a04_second_participant_limit_failure_rolls_back_first_loaded_tool():
    from phymeo.runtime._floating_robot_participant import FloatingRobotCommand
    from phymeo.runtime.contact_scene import ContactSceneBuilder
    from phymeo.runtime.robot_scene import RobotCommand

    example = _example()
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    for owner, position, extension, rate in (
        (10, (0, 0, 0), 1 / 8, 0),
        (20, (3, 0, 0), 1 / 4 - 1e-6, 0.05),
    ):
        state = FloatingRobotState(
            np.array((*position, 0, 0, 0, 1, 0, extension), dtype=float),
            np.array([0, 0, 0, 0, 0, 0, 0, rate], dtype=float),
        )
        builder.add_floating_urdf(
            owner,
            PROBE,
            robot_state=state,
            kp=[0, 0],
            kd=[0, 0],
            loader=example.load_tools,
        )
    builder.pair(10, 20, None)
    loads = FloatingRobotInputs(
        np.zeros(2), np.array([[0, 1.0, 0], [0, 0, 0], [0, 0, 0]]), np.zeros((3, 3))
    )
    actions = {
        10: FloatingRobotCommand(RobotCommand([0, 1 / 8], feedforward=[0, 0]), external=loads),
        20: FloatingRobotCommand(RobotCommand([0, 1 / 8], feedforward=[0, 0])),
    }
    with builder.build() as scene, builder.build() as fresh:
        old, geometry = scene.observe_scene()
        events = scene.contact_events
        with pytest.raises(ValueError, match="joint position"):
            scene.step(H, actions=actions)
        _unchanged(scene, old, geometry, events)
        actions[20] = FloatingRobotCommand(RobotCommand([0, 1 / 8], feedforward=[0, -32]))
        actual, expected = scene.step(H, actions=actions), fresh.step(H, actions=actions)
        for address in actual.robots:
            np.testing.assert_array_equal(actual.robots[address].q, expected.robots[address].q)
            np.testing.assert_array_equal(actual.robots[address].v, expected.robots[address].v)
        assert scene.contact_events == fresh.contact_events


@pytest.mark.parametrize("transformed", [False, True])
def test_a01_root_fixed_tool_contact_has_real_root_reaction_and_material_owner(transformed):
    from phymeo.constraints.robot_contact import RobotContactMaterial
    from phymeo.runtime._floating_robot_participant import FloatingRobotCommand
    from phymeo.runtime.contact_scene import ContactSceneBuilder

    example = _example()
    rotation = CYCLE if transformed else np.eye(3)
    shift = np.array([1 / 2, -1 / 4, 3 / 4]) if transformed else np.zeros(3)
    quaternion = (0.5, 0.5, 0.5, 0.5) if transformed else (0, 0, 0, 1)
    state = FloatingRobotState(np.r_[shift, quaternion], np.zeros(6))
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    builder.add_floating_urdf(
        10,
        COMPOSITE,
        robot_state=state,
        kp=[],
        kd=[],
        loader=example.load_tools,
    )
    builder.add_sphere(
        20,
        1 / 16,
        mass=1,
        position=shift + rotation @ [1 / 4, 1 / 8 - 1 / 1024, 0],
    )
    builder.pair(10, 20, RobotContactMaterial(2048, 0, 0))
    with builder.build() as scene:
        out = scene.step(H, actions={10: FloatingRobotCommand(None)})
        row = out.robots[0, 10]
        assert len(row.q) == 7 and len(row.v) == 6
        assert len(out.loads.contacts) == 1
        assert out.loads.contacts[0].force_n == pytest.approx(2, abs=1e-10)
        np.testing.assert_allclose(
            row.acceleration_full,
            np.r_[rotation @ [0, -2 / 3, 0], rotation @ [0, 0, -8 / 13]],
            atol=1e-10,
            rtol=0,
        )
        np.testing.assert_allclose(
            out.rigid.velocity[0], H * (rotation @ [0, 2, 0]), atol=ATOL, rtol=0
        )
        np.testing.assert_allclose(3 * row.v[:3] + out.rigid.velocity[0], 0, atol=ATOL, rtol=0)
        np.testing.assert_array_equal(row.fixed_root_contact_wrench, np.zeros(6))
        events = scene.contact_events
        assert len(events.active) == 1
        record = events.active[0]
        owner = record.first_owner if record.first_owner.body == 10 else record.second_owner
        assert (owner.kind, owner.link_index, owner.link_name) == ("robot_root", 0, "base")
        for component in record.components:
            for site, material in zip(component.load.sites, component.material, strict=True):
                local = material.first if record.first_owner.body == 10 else material.second
                world = (
                    site.material.first_point
                    if record.first_owner.body == 10
                    else site.material.second_point
                )
                np.testing.assert_allclose(
                    shift + rotation @ local.local_point_m, world, atol=ATOL, rtol=0
                )
                assert local.local_point_m[0] == pytest.approx(1 / 4, abs=ATOL)


def test_a04_fixed_descendant_cannot_be_rebound_to_another_valid_body():
    from dataclasses import replace

    from phymeo.builders._floating_fixed_asset import FloatingFixedSite

    asset = load_floating_urdf(PROBE, gravity=(0, 0, 0), collapse_fixed_joints=True)
    sites = dict(asset.sites)
    sites["tool"] = FloatingFixedSite(1, (0, 0, 0))
    with pytest.raises(ValueError, match="different owners"):
        replace(asset, sites=sites)
