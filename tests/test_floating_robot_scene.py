"""Tests first: one physical floating root and joint tree in the shared HOST scene.

The public first RED uses only existing imports and a valid ordinary URDF before
calling the missing builder entry. No free/dummy body substitutes for the root.
"""

from itertools import combinations

import numpy as np
import pytest

from phymeo.builders.mimic_urdf import load_mimic_urdf
from phymeo.builders.mjcf import load_mjcf
from phymeo.builders.urdf import load_urdf
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.dynamics.floating_robot import FloatingRobotState, floating_kinematics
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand
from phymeo.solvers.floating_tree import FloatingRobotInputs, floating_dynamics_terms

RADIUS = 1 / 16
H = 0.001
FORCE_ATOL = 1e-10
STATE_ATOL = 1e-12
LAW = RobotContactMaterial(2000, 0, 0)


def _xyz(values):
    return " ".join(str(float(value)) for value in values)


def _sphere(name="root_sphere", center=(0, 0, 0)):
    return (
        f'<collision name="{name}"><origin xyz="{_xyz(center)}"/>'
        f'<geometry><sphere radius="{RADIUS}"/></geometry></collision>'
    )


def _inertia(mass, com, diagonal):
    return (
        f'<inertial><origin xyz="{_xyz(com)}"/><mass value="{mass}"/>'
        f'<inertia ixx="{diagonal[0]}" iyy="{diagonal[1]}" izz="{diagonal[2]}" '
        'ixy="0" ixz="0" iyz="0"/></inertial>'
    )


def _asset(
    path, *, root_mass=2, root_com=(0, 0, 0), root_inertia=(0.25, 0.5, 0.75),
    root_collision=None, child=True, child_origin=(0, 0, 0), child_com=(0, 0, 0),
    child_mass=1, axis=(1, 0, 0), child_collision="", lower=-0.1, upper=0.1,
):
    root_collision = _sphere() if root_collision is None else root_collision
    source = (
        '<robot name="floating_mechanism"><link name="base">'
        + _inertia(root_mass, root_com, root_inertia) + root_collision + '</link>'
    )
    if child:
        source += (
            '<link name="tool">' + _inertia(child_mass, child_com, (0.25, 0.25, 0.25))
            + child_collision + '</link><joint name="slide" type="prismatic">'
            '<parent link="base"/><child link="tool"/>'
            f'<origin xyz="{_xyz(child_origin)}"/><axis xyz="{_xyz(axis)}"/>'
            f'<limit lower="{lower}" upper="{upper}" effort="128" velocity="10"/>'
            '</joint>'
        )
    path.write_text(source + '</robot>\n', encoding="utf-8")
    return path


def _state(
    *, q=(0.0,), v=(0.0,), position=(0, 0, 0), xyzw=(0, 0, 0, 1),
    com_velocity=(0, 0, 0), omega=(0, 0, 0),
):
    return FloatingRobotState(
        np.array((*position, *xyzw, *q), dtype=float),
        np.array((*com_velocity, *omega, *v), dtype=float),
    )


def _add(builder, owner, path, state=None, *, kp=0, kd=0, cap=128, **kwargs):
    state = _state() if state is None else state
    n = len(state.v) - 6
    builder.add_floating_urdf(
        owner, path, robot_state=state, kp=[kp] * n, kd=[kd] * n,
        effort_limit=[cap] * n, **kwargs,
    )


def _command(effort=0.0, *, root_only=False, external=None, target=0.0):
    # Keep the real public missing-method RED before importing this NEW type.
    from phymeo.runtime._floating_robot_participant import FloatingRobotCommand

    joints = None if root_only else RobotCommand([target], feedforward=[effort])
    return FloatingRobotCommand(joints, external=external)


def test_a01_real_root_contact_accelerates_full_tree_and_free_body(tmp_path):
    path = _asset(tmp_path / "root-and-slide.urdf")
    old_asset = load_urdf(path, gravity=(0, 0, 0))
    assert old_asset.root_link == "base" and old_asset.joint_names == ("slide",)
    np.testing.assert_array_equal(old_asset.model.mass, [1])
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add(builder, 10, path)
    builder.add_sphere(20, RADIUS, mass=4, position=(0.124, 0, 0))
    builder.pair(10, 20, LAW)
    with builder.build(backend="host") as scene:
        assert scene.rigid_rows == {(0, 20): 0}
        assert scene.state.rigid.body_count == 1
        asset = scene.assets[0, 10]
        np.testing.assert_array_equal(asset.model.root.inverse_mass, [0.5])
        terms = floating_dynamics_terms(asset.model, _state(), FloatingRobotInputs.zeros(1))
        np.testing.assert_array_equal(terms.mass_matrix[np.ix_([0, 6], [0, 6])], [[3, 1], [1, 1]])
        out = scene.step(H, actions={10: _command()})
        row = out.robots[0, 10]
        assert len(row.q) == 8 and len(row.v) == 7 and row.link_wrenches_world.shape == (2, 6)
        np.testing.assert_allclose(row.acceleration_full, [-1, 0, 0, 0, 0, 0, 1], atol=STATE_ATOL)
        np.testing.assert_allclose(row.contact_effort_full, [-2, 0, 0, 0, 0, 0, 0], atol=FORCE_ATOL)
        np.testing.assert_allclose(row.applied_effort_full, row.contact_effort_full, atol=STATE_ATOL)
        np.testing.assert_allclose(row.v, [-H, 0, 0, 0, 0, 0, H], atol=STATE_ATOL)
        np.testing.assert_allclose(row.q, [-H * H, 0, 0, 0, 0, 0, 1, H * H], atol=STATE_ATOL)
        np.testing.assert_array_equal(row.full_q, row.q)
        np.testing.assert_array_equal(row.full_v, row.v)
        np.testing.assert_allclose(row.link_wrenches_world[0], [-2, 0, 0, 0, 0, 0], atol=FORCE_ATOL)
        np.testing.assert_array_equal(row.link_wrenches_world[1], np.zeros(6))
        measured = floating_kinematics(asset.model, FloatingRobotState(row.q, row.v))
        np.testing.assert_allclose(measured.com_position[1], [0, 0, 0], atol=STATE_ATOL)
        np.testing.assert_allclose(measured.com_velocity, [[-H, 0, 0], [0, 0, 0]], atol=STATE_ATOL)
        np.testing.assert_allclose(out.rigid.velocity[0], [H / 2, 0, 0], atol=STATE_ATOL)
        np.testing.assert_allclose(out.rigid.position[0], [0.124 + H * H / 2, 0, 0], atol=STATE_ATOL)
        np.testing.assert_allclose(
            2 * measured.com_velocity[0] + measured.com_velocity[1] + 4 * out.rigid.velocity[0],
            0, atol=STATE_ATOL,
        )
        assert out.step_number == 1 and out.load_time_s == 0 and out.time_s == H
        assert len(out.loads.contacts) == 1 and out.loads.contacts[0].force_n == pytest.approx(2, abs=FORCE_ATOL)
        event = scene.contact_events.active[0]
        root = next(owner for owner in (event.first_owner, event.second_owner) if owner.body == 10)
        assert (root.kind, root.local_frame, root.link_index, root.link_name) == (
            "robot_root", "robot_model_link", 0, "base",
        )
