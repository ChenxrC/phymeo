"""A2 direct robot contact: fixed analytical inputs, parent-only execution.

Only existing public imports are used. The first RED must reach the original
robot-pair build rejection, with no free or dummy rigid body.
"""

from itertools import combinations

import numpy as np
import pytest

from phymeo.builders.mimic_urdf import load_mimic_urdf
from phymeo.builders.mjcf import load_mjcf
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.dynamics.robot_kinematics import forward_kinematics
from phymeo.dynamics.rotation import quaternion_matrix
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_payload_scene import PadFace
from phymeo.runtime.robot_scene import RobotCommand

RADIUS = 1 / 16
LAW = RobotContactMaterial(2000.0, 0.0, 0.0)
H = 0.001
FORCE_ATOL = 1e-10
STATE_ATOL = 1e-12


def _xyz(values):
    return " ".join(str(float(value)) for value in values)


def _collision(name="round_tool", *, kind="sphere", xyz=(0, 0, 0), rpy=(0, 0, 0)):
    geometry = (
        f'<sphere radius="{RADIUS}"/>'
        if kind == "sphere"
        else '<box size="0.2 0.4 0.6"/>'
    )
    return (
        f'<collision name="{name}"><origin xyz="{_xyz(xyz)}" rpy="{_xyz(rpy)}"/>'
        f'<geometry>{geometry}</geometry></collision>'
    )


def _urdf(
    path, *, mass=2, center=(0, 0, 0), axis=(1, 0, 0), rpy=(0, 0, 0),
    kind="prismatic", inertia=0.01, collisions=None, root_collisions="",
):
    collision = _collision() if collisions is None else collisions
    source = f'''<robot name="direct_contact_mechanism">
<link name="base">{root_collisions}</link>
<link name="pusher"><inertial><mass value="{mass}"/>
<inertia ixx="{inertia}" iyy="{inertia}" izz="{inertia}" ixy="0" ixz="0" iyz="0"/>
</inertial>{collision}</link>
<joint name="joint" type="{kind}"><parent link="base"/><child link="pusher"/>
<origin xyz="{_xyz(center)}" rpy="{_xyz(rpy)}"/><axis xyz="{_xyz(axis)}"/>
<limit lower="-0.1" upper="0.1" effort="128" velocity="10"/>
</joint></robot>
'''
    path.write_text(source, encoding="utf-8")
    return path


def _add(builder, owner, path, *, q=0.0, v=0.0, loader=None, pads=()):
    builder.add_urdf(
        owner, path, robot_state=RobotState(np.array([q]), np.array([v])),
        kp=[0], kd=[0], effort_limit=[128], loader=loader, pads=pads,
    )


def _actions(left=0.0, right=0.0):
    return {
        10: RobotCommand([0], feedforward=[left]),
        20: RobotCommand([0], feedforward=[right]),
    }


def _spheres(tmp_path, *, masses=(2, 2), reverse=False, transformed=False, law=LAW):
    rotation = (
        np.array([[0.6, 0, -0.8], [0, 1, 0], [0.8, 0, 0.6]])
        if transformed else np.eye(3)
    )
    shift = np.array([3, -2, 0.75]) if transformed else np.zeros(3)
    angle = -np.arctan2(0.8, 0.6) if transformed else 0.0
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    entries = ((10, masses[0], 0.0), (20, masses[1], 0.124))
    for owner, mass, distance in entries[::-1] if reverse else entries:
        _add(
            builder, owner,
            _urdf(
                tmp_path / f"sphere-{owner}.urdf", mass=mass,
                center=shift + distance * rotation[:, 0], rpy=(0, angle, 0),
            ),
        )
    builder.pair(10, 20, law)
    return builder, rotation, shift


def _bilateral(scene, out, left=10, right=20):
    keys_a, keys_b = set(scene.body_keys[0, left]), set(scene.body_keys[0, right])
    rows = [
        pair for pair in out.loads.contacts
        if (pair.first in keys_a and pair.second in keys_b)
        or (pair.first in keys_b and pair.second in keys_a)
    ]
    assert len(rows) == 1
    (pair,) = rows
    assert len(pair.sites) == 1
    (site,) = pair.sites
    ordered = pair.first in keys_a
    pa = site.material.first_point if ordered else site.material.second_point
    pb = site.material.second_point if ordered else site.material.first_point
    normal = np.asarray(pair.normal) * (1 if ordered else -1)
    force_b = np.asarray(site.force_on_second_n) * (1 if ordered else -1)
    np.testing.assert_allclose(np.asarray(pb) - pa, pair.gap_m * normal, atol=STATE_ATOL, rtol=0)
    np.testing.assert_allclose(site.application_point_first_m, (np.asarray(pa) + pb) / 2, atol=STATE_ATOL)
    np.testing.assert_array_equal(site.application_point_first_m, site.application_point_second_m)
    return pair, site, normal, force_b, np.asarray(pa), np.asarray(pb)


def test_a01_direct_spheres_use_two_real_trees_without_a_free_body(tmp_path):
    builder, _, _ = _spheres(tmp_path)
    # Existing imports/valid URDFs must reach the original direct-pair build guard.
    with builder.build(backend="host") as scene:
        assert scene.rigid_rows == {} and scene.state.rigid.body_count == 0
        assert len(scene.geometry().model.shapes) == 2
        initial, geometry = scene.observe_scene()
        out = scene.step(H, actions=_actions())
        left, right = out.robots[0, 10], out.robots[0, 20]
        assert out.step_number == 1 and out.time_s == H and out.load_time_s == 0
        assert out.loads.inputs.rigid.force.shape == (0, 3)
        for row, direction in ((left, -1), (right, 1)):
            np.testing.assert_allclose(row.contact_effort_full, [2 * direction], atol=FORCE_ATOL, rtol=0)
            np.testing.assert_allclose(row.v, [direction * H], atol=STATE_ATOL, rtol=0)
            np.testing.assert_allclose(row.q, [direction * H * H], atol=STATE_ATOL, rtol=0)
        np.testing.assert_allclose(2 * left.v + 2 * right.v, 0, atol=STATE_ATOL, rtol=0)
        pair, site, normal, force, pa, pb = _bilateral(scene, out)
        assert pair.force_n == pytest.approx(2, abs=FORCE_ATOL)
        assert pair.gap_m == pytest.approx(-0.001, abs=STATE_ATOL)
        assert pair.potential_j == pytest.approx(0.001, abs=STATE_ATOL)
        np.testing.assert_allclose(normal, [1, 0, 0], atol=STATE_ATOL)
        np.testing.assert_allclose(force, [2, 0, 0], atol=FORCE_ATOL)
        np.testing.assert_allclose(pa, [0.0625, 0, 0], atol=STATE_ATOL)
        np.testing.assert_allclose(pb, [0.0615, 0, 0], atol=STATE_ATOL)
        np.testing.assert_allclose(site.application_point_first_m, [0.062, 0, 0], atol=STATE_ATOL)
        assert len(scene.contact_events.active) == 1
        record = scene.contact_events.active[0]
        assert {record.first_owner.body, record.second_owner.body} == {10, 20}
        assert {record.first_owner.kind, record.second_owner.kind} == {"robot_link"}
        assert initial.step_number == 0 and geometry.time_s == 0
        np.testing.assert_array_equal(initial.robots[0, 10].q, [0])
