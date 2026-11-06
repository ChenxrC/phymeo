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
    geometry = f'<sphere radius="{RADIUS}"/>' if kind == "sphere" else '<box size="0.2 0.4 0.6"/>'
    return (
        f'<collision name="{name}"><origin xyz="{_xyz(xyz)}" rpy="{_xyz(rpy)}"/>'
        f"<geometry>{geometry}</geometry></collision>"
    )


def _urdf(
    path,
    *,
    mass=2,
    center=(0, 0, 0),
    axis=(1, 0, 0),
    rpy=(0, 0, 0),
    kind="prismatic",
    inertia=0.01,
    collisions=None,
    root_collisions="",
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
        owner,
        path,
        robot_state=RobotState(np.array([q]), np.array([v])),
        kp=[0],
        kd=[0],
        effort_limit=[128],
        loader=loader,
        pads=pads,
    )


def _actions(left=0.0, right=0.0):
    return {
        10: RobotCommand([0], feedforward=[left]),
        20: RobotCommand([0], feedforward=[right]),
    }


def _spheres(tmp_path, *, masses=(2, 2), reverse=False, transformed=False, law=LAW):
    rotation = np.array([[0.6, 0, -0.8], [0, 1, 0], [0.8, 0, 0.6]]) if transformed else np.eye(3)
    shift = np.array([3, -2, 0.75]) if transformed else np.zeros(3)
    angle = -np.arctan2(0.8, 0.6) if transformed else 0.0
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    entries = ((10, masses[0], 0.0), (20, masses[1], 0.124))
    for owner, mass, distance in entries[::-1] if reverse else entries:
        _add(
            builder,
            owner,
            _urdf(
                tmp_path / f"sphere-{owner}.urdf",
                mass=mass,
                center=shift + distance * rotation[:, 0],
                rpy=(0, angle, 0),
            ),
        )
    builder.pair(10, 20, law)
    return builder, rotation, shift


def _bilateral(scene, out, left=10, right=20):
    keys_a, keys_b = set(scene.body_keys[0, left]), set(scene.body_keys[0, right])
    rows = [
        pair
        for pair in out.loads.contacts
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
    np.testing.assert_allclose(
        site.application_point_first_m, (np.asarray(pa) + pb) / 2, atol=STATE_ATOL
    )
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
            np.testing.assert_allclose(
                row.contact_effort_full, [2 * direction], atol=FORCE_ATOL, rtol=0
            )
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


def _material_from_owner(scene, observation, owner, address):
    asset = scene.assets[owner.world, owner.body]
    tree = asset.tree if hasattr(asset, "tree") else asset
    if owner.kind == "robot_root":
        position = np.asarray(tree.root_xyz)
        rotation = quaternion_matrix(np.asarray([tree.root_xyzw]))[0]
    else:
        joint = observation.robots[owner.world, owner.body]
        full_entry = (
            joint.entry_q
            if not hasattr(asset, "tree")
            else asset.matrix @ joint.entry_q + asset.offset_vector
        )
        geo = forward_kinematics(tree.model, full_entry)
        position, rotation = geo.position[owner.link_index], geo.rotation[owner.link_index]
    # Weights locate sphere centres/capsule spines, not the complete skin point.
    return position + rotation @ np.asarray(address.local_point_m)


def _assert_event_materials(scene, observation):
    events = scene.contact_events
    assert events.sample_time_s == observation.load_time_s
    assert events.commit_time_s == observation.time_s
    for record in events.active:
        for component in record.components:
            for site, material in zip(component.load.sites, component.material, strict=True):
                for owner, address, expected in (
                    (record.first_owner, material.first, site.material.first_point),
                    (record.second_owner, material.second, site.material.second_point),
                ):
                    assert sum(weight for _, weight in address.weights) == pytest.approx(1)
                    np.testing.assert_allclose(
                        _material_from_owner(scene, observation, owner, address),
                        expected,
                        atol=STATE_ATOL,
                        rtol=0,
                    )


@pytest.mark.parametrize("reverse,transformed", [(False, False), (True, False), (True, True)])
def test_a02_mass_registration_and_rigid_transform_preserve_actual_reaction(
    tmp_path,
    reverse,
    transformed,
):
    builder, rotation, shift = _spheres(
        tmp_path,
        masses=(2, 4),
        reverse=reverse,
        transformed=transformed,
    )
    with builder.build() as scene:
        out = scene.step(H, actions=_actions())
        left, right = out.robots[0, 10], out.robots[0, 20]
        np.testing.assert_allclose(left.v, [-H], atol=STATE_ATOL, rtol=0)
        np.testing.assert_allclose(right.v, [H / 2], atol=STATE_ATOL, rtol=0)
        np.testing.assert_allclose(left.q, [-H * H], atol=STATE_ATOL, rtol=0)
        np.testing.assert_allclose(right.q, [H * H / 2], atol=STATE_ATOL, rtol=0)
        np.testing.assert_allclose(2 * left.v + 4 * right.v, 0, atol=STATE_ATOL, rtol=0)
        _, _, normal, force, pa, pb = _bilateral(scene, out)
        np.testing.assert_allclose(normal, rotation[:, 0], atol=STATE_ATOL, rtol=0)
        np.testing.assert_allclose(force, 2 * rotation[:, 0], atol=FORCE_ATOL, rtol=0)
        np.testing.assert_allclose(pa, shift + rotation @ [0.0625, 0, 0], atol=STATE_ATOL)
        np.testing.assert_allclose(pb, shift + rotation @ [0.0615, 0, 0], atol=STATE_ATOL)
        np.testing.assert_allclose(left.link_wrenches_world[0, :3], -force, atol=FORCE_ATOL)
        np.testing.assert_allclose(right.link_wrenches_world[0, :3], force, atol=FORCE_ATOL)
        _assert_event_materials(scene, out)


def _capsule_mjcf(path, *, name="rail_tool", axis=(1, 0, 0), center=(0, 0, 0)):
    # Capsule is an actual MJCF geom, not an invented URDF tag or replaced asset.
    source = f'''<mujoco model="finite_rail">
<compiler angle="radian" inertiafromgeom="false"/>
<worldbody><body name="base"><body name="pusher" pos="{_xyz(center)}">
<joint name="joint" type="slide" axis="{_xyz(axis)}" range="-0.1 0.1"/>
<inertial pos="0 0 0" mass="2" diaginertia="0.01 0.01 0.01"/>
<geom name="{name}" type="capsule" fromto="0 0 -0.25 0 0 0.25" size="{RADIUS}"/>
</body></body></worldbody></mujoco>
'''
    path.write_text(source, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "center,axis,feature,force",
    [
        ((0.124, 0, 0), (1, 0, 0), "shaft", 2),
        ((0, 0, 0.374), (0, 0, 1), "end0", 2),
        ((0.124, 0, 0.375), (1, 0, 0), None, 0),
    ],
)
def test_a02_finite_capsule_shaft_end_and_axis_extension_miss(
    tmp_path,
    center,
    axis,
    feature,
    force,
):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add(builder, 10, _capsule_mjcf(tmp_path / "rail.xml", axis=axis), loader=load_mjcf)
    _add(builder, 20, _urdf(tmp_path / "ball.urdf", mass=4, center=center, axis=axis))
    builder.pair(10, 20, LAW)
    with builder.build() as scene:
        out = scene.step(H, actions=_actions())
        collider = scene.assets[0, 10].collisions[0]
        assert collider.shape.kind == "capsule"
        # MJCF fromto points along (start-end); make_collider stores -axis first.
        # Thus the positive world-Z tip in this unchanged input is endpoint 0.
        np.testing.assert_array_equal(collider.local_vertices, [[0, 0, 0.25], [0, 0, -0.25]])
        assert len(out.loads.contacts) == (1 if force else 0)
        np.testing.assert_allclose(out.robots[0, 10].v, [-force * H / 2], atol=STATE_ATOL)
        np.testing.assert_allclose(out.robots[0, 20].v, [force * H / 4], atol=STATE_ATOL)
        if force:
            assert out.contacts[0]["feature"] == feature
            pair, _, normal, actual, pa, pb = _bilateral(scene, out)
            assert pair.force_n == pytest.approx(force, abs=FORCE_ATOL)
            np.testing.assert_allclose(normal, axis, atol=STATE_ATOL)
            np.testing.assert_allclose(actual, force * np.asarray(axis), atol=FORCE_ATOL)
            tip = np.array([0, 0, 0.25]) if feature == "end0" else np.zeros(3)
            np.testing.assert_allclose(pa, tip + RADIUS * np.asarray(axis), atol=STATE_ATOL)
            np.testing.assert_allclose(
                pb, np.asarray(center) - RADIUS * np.asarray(axis), atol=STATE_ATOL
            )
            _assert_event_materials(scene, out)
        else:
            assert not scene.contact_events.active


@pytest.mark.parametrize(
    "feature,pads",
    [("face", False), ("face", True), ("edge", False), ("outside", False)],
)
def test_a02_local_rotated_box_witnesses_and_pad_registration_do_not_double_force(
    tmp_path,
    feature,
    pads,
):
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    offset = np.array([0.125, -0.25, 0.0625])
    boundary = offset + rotation @ ([0.1, 0, 0] if feature == "face" else [0.1, 0.2, 0])
    normal = rotation @ ([1, 0, 0] if feature == "face" else [0.6, 0.8, 0])
    distance = RADIUS + (0.001 if feature == "outside" else -0.001)
    center = boundary + distance * normal
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    box = _urdf(
        tmp_path / "box.urdf",
        axis=normal,
        collisions=_collision("box_tool", kind="box", xyz=offset, rpy=(0, 0, np.pi / 2)),
    )
    pad = (PadFace.named("pusher", "box_tool", 0, 1, ((0, 0),)),) if pads else ()
    _add(builder, 10, box, pads=pad)
    _add(builder, 20, _urdf(tmp_path / "sphere.urdf", mass=4, center=center, axis=normal))
    builder.pair(10, 20, LAW)
    with builder.build() as scene:
        out = scene.step(H, actions=_actions())
        if feature == "outside":
            assert not out.loads.contacts and not scene.contact_events.active
            np.testing.assert_array_equal(out.robots[0, 10].v, [0])
            np.testing.assert_array_equal(out.robots[0, 20].v, [0])
        else:
            assert len(out.loads.contacts) == 1 and len(out.contacts) == 1
            assert out.contacts[0]["feature"] == feature
            _, _, n, force, pa, pb = _bilateral(scene, out)
            np.testing.assert_allclose(n, normal, atol=STATE_ATOL)
            np.testing.assert_allclose(force, 2 * normal, atol=FORCE_ATOL)
            np.testing.assert_allclose(pa, boundary, atol=STATE_ATOL)
            np.testing.assert_allclose(pb, boundary - 0.001 * normal, atol=STATE_ATOL)
            point = boundary - 0.0005 * normal
            np.testing.assert_allclose(
                out.robots[0, 10].link_wrenches_world[0, 3:],
                np.cross(point, -force),
                atol=FORCE_ATOL,
            )
            np.testing.assert_allclose(out.robots[0, 10].v, [-H], atol=STATE_ATOL)
            np.testing.assert_allclose(out.robots[0, 20].v, [H / 2], atol=STATE_ATOL)
            _assert_event_materials(scene, out)


@pytest.mark.parametrize(
    "friction,velocity,torque,alpha",
    [(0.0, 0.0, 0.5, 4.0), (0.25, 0.02, 0.531, 4.248)],
)
def test_a03_offset_rotor_and_sliding_partner_use_both_com_moments_and_point_power(
    tmp_path,
    friction,
    velocity,
    torque,
    alpha,
):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add(
        builder,
        10,
        _urdf(
            tmp_path / "rotor.urdf",
            kind="revolute",
            axis=(0, 0, 1),
            inertia=1 / 8,
            collisions=_collision(xyz=(0, 0.25, 0)),
        ),
    )
    _add(
        builder,
        20,
        _urdf(tmp_path / "slide.urdf", mass=4, center=(0.124, 0.25, 0), axis=(0, 1, 0)),
        v=velocity,
    )
    builder.pair(10, 20, RobotContactMaterial(2000, 0, friction, 0.01))
    with builder.build() as scene:
        out = scene.step(H, actions=_actions())
        rotor, slide = out.robots[0, 10], out.robots[0, 20]
        tangent = -0.5 if friction else 0.0
        _, site, _, force, _, _ = _bilateral(scene, out)
        np.testing.assert_allclose(force, [2, tangent, 0], atol=FORCE_ATOL)
        np.testing.assert_allclose(
            rotor.link_wrenches_world[0],
            [-2, -tangent, 0, 0, 0, torque],
            atol=FORCE_ATOL,
        )
        np.testing.assert_allclose(
            slide.link_wrenches_world[0],
            [2, tangent, 0, 0, 0, -0.062 * tangent],
            atol=FORCE_ATOL,
        )
        np.testing.assert_allclose(rotor.contact_effort_full, [torque], atol=FORCE_ATOL)
        np.testing.assert_allclose(rotor.v, [alpha * H], atol=STATE_ATOL)
        np.testing.assert_allclose(rotor.q, [alpha * H * H], atol=STATE_ATOL)
        np.testing.assert_allclose(slide.v, [velocity + H * tangent / 4], atol=STATE_ATOL)
        np.testing.assert_allclose(slide.q, [H * (velocity + H * tangent / 4)], atol=STATE_ATOL)
        assert site.power_w == pytest.approx(-0.01 if friction else 0, abs=STATE_ATOL)
        power = sum(float(j.contact_effort_full @ j.entry_v) for j in out.robots.values())
        assert power == pytest.approx(site.power_w, abs=STATE_ATOL)
        np.testing.assert_allclose(
            rotor.link_wrenches_world[0, 3:]
            + slide.link_wrenches_world[0, 3:]
            + np.cross([0.124, 0.25, 0], force),
            0,
            atol=FORCE_ATOL,
        )
        _assert_event_materials(scene, out)


def test_a03_normal_damping_uses_actual_closing_speed_and_dissipates(tmp_path):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add(builder, 10, _urdf(tmp_path / "left.urdf"), v=1 / 16)
    _add(builder, 20, _urdf(tmp_path / "right.urdf", center=(0.124, 0, 0)), v=-1 / 16)
    builder.pair(10, 20, RobotContactMaterial(2000, 8, 0))
    with builder.build() as scene:
        out = scene.step(H, actions=_actions())
        pair, site, _, force, _, _ = _bilateral(scene, out)
        assert pair.force_n == pytest.approx(3, abs=FORCE_ATOL)
        assert site.rate_m_per_s == pytest.approx(-1 / 8, abs=STATE_ATOL)
        assert site.power_w == pytest.approx(-3 / 8, abs=STATE_ATOL)
        # Spring energy rate k*delta*(-vn) = 1/4 W; dashpot loss = 1/8 W.
        assert site.power_w + 1 / 4 == pytest.approx(-1 / 8, abs=STATE_ATOL)
        np.testing.assert_allclose(force, [3, 0, 0], atol=FORCE_ATOL)
        np.testing.assert_allclose(out.robots[0, 10].v, [1 / 16 - 1.5 * H], atol=STATE_ATOL)
        np.testing.assert_allclose(out.robots[0, 20].v, [-1 / 16 + 1.5 * H], atol=STATE_ATOL)


def _mimic(path):
    path.write_text(
        f"""<robot name="negative_transmission"><link name="base"/>
<link name="driver"><inertial><mass value="2"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
<link name="follower"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>{_collision()}</link>
<joint name="drive" type="prismatic"><parent link="base"/><child link="driver"/><origin xyz="0 2 0"/><axis xyz="1 0 0"/><limit lower="-0.1" upper="0.1" effort="128" velocity="10"/></joint>
<joint name="follow" type="prismatic"><parent link="base"/><child link="follower"/><origin xyz="0.121 0 0"/><axis xyz="1 0 0"/><limit lower="-0.1" upper="0.1" effort="128" velocity="10"/><mimic joint="drive" multiplier="-2" offset="0.003"/></joint>
</robot>
""",
        encoding="utf-8",
    )
    return path


def test_a04_mimic_receives_full_contact_and_constraint_reaction_in_one_coordinate_map(tmp_path):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add(builder, 10, _urdf(tmp_path / "ordinary.urdf"))
    _add(builder, 20, _mimic(tmp_path / "mimic.urdf"), loader=load_mimic_urdf)
    builder.pair(10, 20, LAW)
    with builder.build() as scene:
        mechanism = scene.assets[0, 20]
        np.testing.assert_array_equal(mechanism.matrix, [[1], [-2]])
        np.testing.assert_allclose(mechanism.offset_vector, [0, 0.003], atol=STATE_ATOL)
        out = scene.step(H, actions=_actions())
        ordinary, linked = out.robots[0, 10], out.robots[0, 20]
        receipt = linked.mimic_loads
        np.testing.assert_allclose(ordinary.v, [-H], atol=STATE_ATOL)
        np.testing.assert_allclose(linked.contact_effort_full, [0, 2], atol=FORCE_ATOL)
        np.testing.assert_allclose(linked.contact_effort, [-4], atol=FORCE_ATOL)
        np.testing.assert_allclose(linked.v, [-2 * H / 3], atol=STATE_ATOL)
        np.testing.assert_allclose(linked.full_v, [-2 * H / 3, 4 * H / 3], atol=STATE_ATOL)
        np.testing.assert_allclose(
            linked.full_q,
            [-2 * H * H / 3, 0.003 + 4 * H * H / 3],
            atol=STATE_ATOL,
        )
        np.testing.assert_allclose(receipt.acceleration_full, [-2 / 3, 4 / 3], atol=FORCE_ATOL)
        np.testing.assert_allclose(
            receipt.constraint_effort_full, [-4 / 3, -2 / 3], atol=FORCE_ATOL
        )
        np.testing.assert_allclose(
            mechanism.matrix.T @ receipt.constraint_effort_full, 0, atol=STATE_ATOL
        )
        np.testing.assert_allclose(
            np.array([2, 1]) * linked.full_v / H,
            receipt.actuator_effort_full
            + receipt.contact_effort_full
            + receipt.constraint_effort_full,
            atol=FORCE_ATOL,
        )
        assert len(receipt.row_forces) == 1
        assert receipt.multiplier_force == pytest.approx(-2 / 3, abs=FORCE_ATOL)
        assert receipt.constraint_power_W == pytest.approx(0, abs=STATE_ATOL)
        np.testing.assert_allclose(
            receipt.constraint_effort_full @ linked.full_v, 0, atol=STATE_ATOL
        )
        _assert_event_materials(scene, out)


def test_a05_actual_direct_preload_hold_and_withdraw_has_one_force_and_one_drift_per_tree(tmp_path):
    builder, _, _ = _spheres(tmp_path)
    with builder.build() as scene:
        first_events, hold_reading = None, None
        withdrawn = []
        observed_end = False
        for step in range(40):
            holding = step < 32
            dt = 1 / 8192 if holding else 1 / 256
            old = scene.observe()
            out = scene.step(dt, actions=_actions(2 if holding else -8, -2 if holding else 8))
            assert out.load_time_s == old.time_s and out.step_number == step + 1
            for owner in (10, 20):
                row, prior = out.robots[0, owner], old.robots[0, owner]
                total = row.drive_effort + row.contact_effort
                np.testing.assert_allclose(
                    2 * (row.v - prior.v), dt * total, atol=STATE_ATOL, rtol=0
                )
                np.testing.assert_allclose(row.q, prior.q + dt * row.v, atol=STATE_ATOL, rtol=0)
                # Exact constant-force substep kinetic identity for mass 2 kg.
                kinetic = float(row.v @ row.v - prior.v @ prior.v)
                work = float(dt * (total @ prior.v) + dt * dt * (total @ total) / 4)
                assert kinetic == pytest.approx(work, abs=STATE_ATOL)
            left, right = out.robots[0, 10], out.robots[0, 20]
            np.testing.assert_allclose(2 * left.v + 2 * right.v, 0, atol=STATE_ATOL)
            if holding:
                assert len(out.loads.contacts) == 1
                np.testing.assert_allclose(left.contact_effort, [-2], atol=FORCE_ATOL)
                np.testing.assert_allclose(right.contact_effort, [2], atol=FORCE_ATOL)
                np.testing.assert_allclose([left.q[0], right.q[0]], 0, atol=STATE_ATOL)
                first_events = scene.contact_events if first_events is None else first_events
                hold_reading = out
            else:
                withdrawn.append(len(out.loads.contacts))
                observed_end |= any(event.kind == "end" for event in scene.contact_events.events)
        assert out.step_number == 40 and out.time_s == 32 / 8192 + 8 / 256
        assert withdrawn[-4:] == [0, 0, 0, 0] and observed_end
        assert not scene.contact_events.active
        assert left.q[0] < -0.001 and right.q[0] > 0.001
        assert left.v[0] < 0 and right.v[0] > 0
        assert len(first_events.active) == 1 and first_events.step_index == 1
        assert hold_reading.step_number == 32 and len(hold_reading.loads.contacts) == 1
        assert first_events.active[0].identity == scene.contact_events.generations[0]


def _with_rigid_pair(builder):
    builder.add_sphere(30, RADIUS, mass=1, position=(3, 0, RADIUS - 0.001))
    builder.add_box(40, (0.25, 0.25, 0.03125), mass=None, position=(3, 0, -0.03125))
    for a, b in combinations((10, 20, 30, 40), 2):
        if (a, b) != (10, 20):
            builder.pair(a, b, LAW if (a, b) == (30, 40) else None)
    return builder


def test_a06_independent_rigid_pair_and_direct_pair_share_one_transaction_without_double_force(
    tmp_path,
):
    builder, _, _ = _spheres(tmp_path)
    with _with_rigid_pair(builder).build() as scene:
        out = scene.step(H, actions=_actions())
        ball, table = (scene.rigid_rows[0, key] for key in (30, 40))
        assert len(out.loads.contacts) == 2 and len(scene.contact_events.active) == 2
        np.testing.assert_allclose(out.robots[0, 10].v, [-H], atol=STATE_ATOL)
        np.testing.assert_allclose(out.robots[0, 20].v, [H], atol=STATE_ATOL)
        np.testing.assert_allclose(out.loads.inputs.rigid.force[ball], [0, 0, 2], atol=FORCE_ATOL)
        np.testing.assert_allclose(out.loads.inputs.rigid.force[table], [0, 0, -2], atol=FORCE_ATOL)
        np.testing.assert_allclose(out.rigid.velocity[ball], [0, 0, 2 * H], atol=STATE_ATOL)
        np.testing.assert_array_equal(out.rigid.velocity[table], [0, 0, 0])
        assert scene.observe_scene()[1].time_s == out.time_s
        owner_pairs = {
            frozenset((row.first_owner.body, row.second_owner.body))
            for row in scene.contact_events.active
        }
        assert owner_pairs == {frozenset((10, 20)), frozenset((30, 40))}


def test_a06_fixed_root_collider_has_real_root_frame_and_observable_reaction(tmp_path):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    root = _collision("root_tool", xyz=(0, 0.25, 0))
    _add(
        builder,
        10,
        _urdf(tmp_path / "root.urdf", center=(0, 3, 0), collisions="", root_collisions=root),
    )
    _add(builder, 20, _urdf(tmp_path / "moving.urdf", center=(0.124, 0.25, 0)))
    builder.pair(10, 20, LAW)
    with builder.build() as scene:
        out = scene.step(H, actions=_actions())
        fixed, moving = out.robots[0, 10], out.robots[0, 20]
        np.testing.assert_array_equal(fixed.contact_effort_full, [0])
        np.testing.assert_array_equal(fixed.link_wrenches_world, np.zeros((1, 6)))
        np.testing.assert_array_equal(fixed.q, [0])
        # This is only the contact wrench on the actual root collider, not the
        # total dynamical reaction transmitted by every constrained child joint.
        np.testing.assert_allclose(
            fixed.fixed_root_contact_wrench,
            [-2, 0, 0, 0, 0, 0.5],
            atol=FORCE_ATOL,
        )
        np.testing.assert_allclose(moving.v, [H], atol=STATE_ATOL)
        record = scene.contact_events.active[0]
        root_owner = next(
            owner for owner in (record.first_owner, record.second_owner) if owner.body == 10
        )
        assert root_owner.kind == "robot_root" and root_owner.link_index == -1
        assert root_owner.link_name == "base"
        _assert_event_materials(scene, out)


def _assert_unchanged(scene, old, geometry, events):
    actual, actual_geometry = scene.observe_scene()
    assert (actual.step_number, actual.time_s, actual.load_time_s) == (
        old.step_number,
        old.time_s,
        old.load_time_s,
    )
    assert scene.contact_events is events and actual_geometry.time_s == geometry.time_s
    for name in ("position", "orientation", "velocity", "angular_velocity"):
        np.testing.assert_array_equal(getattr(actual.rigid, name), getattr(old.rigid, name))
    np.testing.assert_array_equal(actual.loads.inputs.rigid.force, old.loads.inputs.rigid.force)
    np.testing.assert_array_equal(actual.loads.inputs.rigid.torque, old.loads.inputs.rigid.torque)
    assert actual.contacts == old.contacts and actual.loads.contacts == old.loads.contacts
    for owner in old.robots:
        for name in (
            "q",
            "v",
            "full_q",
            "full_v",
            "drive_effort",
            "contact_effort_full",
            "link_wrenches_world",
            "fixed_root_contact_wrench",
        ):
            np.testing.assert_array_equal(
                getattr(actual.robots[owner], name), getattr(old.robots[owner], name)
            )
    assert set(actual_geometry.poses) == set(geometry.poses)
    for key in geometry.poses:
        np.testing.assert_array_equal(actual_geometry.poses[key].start, geometry.poses[key].start)


def test_a06_second_tree_limit_failure_and_real_rigid_input_roll_back_then_retry(tmp_path):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    _add(builder, 10, _urdf(tmp_path / "left.urdf"))
    _add(
        builder,
        20,
        _urdf(tmp_path / "right.urdf", mass=4, center=(0.024005, 0, 0)),
        q=0.099995,
        v=0.01,
    )
    builder.pair(10, 20, LAW)
    _with_rigid_pair(builder)
    with builder.build() as scene, builder.build() as fresh:
        old, geometry = scene.observe_scene()
        events = scene.contact_events
        external = scene.inputs()
        external.rigid.force[scene.rigid_rows[0, 30], 1] = 7
        with pytest.raises(ValueError, match="joint position"):
            scene.step(H, actions=_actions(), external=external)
        _assert_unchanged(scene, old, geometry, events)
        assert scene.last_frame is None
        # Include the actual +2 N contact: (-42 + 2)/4 stops +.01 m/s in 1 ms.
        actions = _actions(0, -42)
        actual = scene.step(H, actions=actions, external=external)
        expected = fresh.step(H, actions=actions, external=external)
        for owner in actual.robots:
            np.testing.assert_array_equal(actual.robots[owner].q, expected.robots[owner].q)
            np.testing.assert_array_equal(actual.robots[owner].v, expected.robots[owner].v)
        np.testing.assert_allclose(actual.robots[0, 20].v, [0], atol=STATE_ATOL)
        np.testing.assert_array_equal(actual.rigid.position, expected.rigid.position)
        np.testing.assert_array_equal(actual.rigid.velocity, expected.rigid.velocity)
        assert actual.loads.contacts == expected.loads.contacts
        assert scene.contact_events == fresh.contact_events


def test_a06_late_event_failure_preserves_accepted_material_and_geometry_then_retries(
    tmp_path, monkeypatch
):
    builder, _, _ = _spheres(tmp_path)
    with _with_rigid_pair(builder).build() as scene, builder.build() as fresh:
        scene.step(H, actions=_actions(2, -2))
        fresh.step(H, actions=_actions(2, -2))
        old, geometry = scene.observe_scene()
        events = scene.contact_events
        original = scene._event_adapter.stage

        def reject(*args, **kwargs):
            raise RuntimeError("A2 late event failure")

        with monkeypatch.context() as patcher:
            patcher.setattr(scene._event_adapter, "stage", reject)
            with pytest.raises(RuntimeError, match="A2 late event failure"):
                scene.step(H, actions=_actions(-8, 8))
        assert scene._event_adapter.stage == original
        _assert_unchanged(scene, old, geometry, events)
        actual = scene.step(H, actions=_actions(-8, 8))
        expected = fresh.step(H, actions=_actions(-8, 8))
        for owner in actual.robots:
            np.testing.assert_array_equal(actual.robots[owner].q, expected.robots[owner].q)
            np.testing.assert_array_equal(actual.robots[owner].v, expected.robots[owner].v)
        np.testing.assert_array_equal(actual.rigid.position, expected.rigid.position)
        assert scene.contact_events == fresh.contact_events
        assert events.step_index == 1 and events.sample_time_s == 0


def test_a06_duplicate_local_keys_owned_snapshots_and_old_event_material_do_not_alias(tmp_path):
    builder, _, _ = _spheres(tmp_path)
    with _with_rigid_pair(builder).build() as scene, builder.build() as other:
        key_a = scene.shape_key(10, "pusher", "round_tool")
        key_b = scene.shape_key(20, "pusher", "round_tool")
        assert key_a != key_b
        local_a, local_b = (scene.assets[0, owner].collisions[0].shape.key for owner in (10, 20))
        assert local_a == local_b
        original_events = scene.contact_events
        sample = scene.sample()
        assert len(sample.contacts) == 2 and scene.contact_events is original_events
        assert scene.observe().step_number == 0
        out = scene.step(H, actions=_actions())
        events = scene.contact_events
        robot_record = next(
            row
            for row in events.active
            if {row.first_owner.body, row.second_owner.body} == {10, 20}
        )
        material = robot_record.components[0].material
        assert material[0].first.local_point_m is not None
        state = scene.state
        state.robots[0, 10].q[:] = 9
        state.robots[0, 20].v[:] = 9
        state.rigid.position[:] = 9
        sample.inputs.rigid.force[:] = 9
        out.external.rigid.force[:] = 9
        out.loads.inputs.rigid.force[:] = 9
        with pytest.raises(ValueError):
            out.robots[0, 10].q.setflags(write=True)
        with pytest.raises(AttributeError):
            robot_record.first_owner.body = 99
        with pytest.raises(TypeError):
            material[0].first.local_point_m[0] = 9
        current = scene.observe()
        np.testing.assert_allclose(current.robots[0, 10].q, [-H * H], atol=STATE_ATOL)
        np.testing.assert_allclose(current.robots[0, 20].v, [H], atol=STATE_ATOL)
        np.testing.assert_array_equal(current.external.rigid.force, 0)
        np.testing.assert_allclose(
            current.loads.inputs.rigid.force[scene.rigid_rows[0, 30]],
            [0, 0, 2],
            atol=FORCE_ATOL,
        )
        assert other.observe().step_number == 0 and other.contact_events.step_index == 0
        scene.step(H, actions=_actions())
        assert robot_record.components[0].material == material and events.step_index == 1
        assert len({shape.key for shape in scene.geometry().model.shapes}) == 4


@pytest.mark.parametrize("enabled", [False, True])
def test_a06_every_enabled_collider_combination_is_validated_before_build(tmp_path, enabled):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    # A distant unsupported pair cannot disappear behind the near supported pair.
    mixed = _collision() + _collision("far_box", kind="box", xyz=(0, 5, 0))
    _add(builder, 10, _urdf(tmp_path / "mixed.urdf", collisions=mixed))
    _add(
        builder,
        20,
        _urdf(
            tmp_path / "box.urdf",
            center=(0.16, 0, 0),
            collisions=_collision("other_box", kind="box"),
        ),
    )
    builder.pair(10, 20, LAW if enabled else None)
    if enabled:
        with pytest.raises(ValueError) as failure:
            builder.build()
        message = str(failure.value)
        assert all(text in message for text in ("10", "20", "far_box", "other_box"))
    else:
        with builder.build() as scene:
            out = scene.step(H, actions=_actions())
            assert not out.loads.contacts and not scene.contact_events.active
            for joint in out.robots.values():
                np.testing.assert_array_equal(joint.contact_effort, [0])
                np.testing.assert_array_equal(joint.q, [0])


def test_a06_capsule_capsule_pair_is_named_rejection_not_a_hidden_exclusion(tmp_path):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    for owner, name, center in ((10, "left_rail", (0, 0, 0)), (20, "right_rail", (0.124, 0, 0))):
        _add(
            builder,
            owner,
            _capsule_mjcf(tmp_path / f"rail-{owner}.xml", name=name, center=center),
            loader=load_mjcf,
        )
    builder.pair(10, 20, LAW)
    with pytest.raises(ValueError) as failure:
        builder.build()
    assert all(
        text in str(failure.value) for text in ("10", "20", "left_rail", "right_rail", "capsule")
    )


def test_a06_ambiguous_coincident_normal_rejects_before_any_publication(tmp_path):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    for owner in (10, 20):
        _add(builder, owner, _urdf(tmp_path / f"coincident-{owner}.urdf"))
    builder.pair(10, 20, LAW)
    with builder.build() as scene:
        old, geometry = scene.observe_scene()
        events = scene.contact_events
        with pytest.raises(ValueError, match="normal|coincident"):
            scene.step(H, actions=_actions())
        _assert_unchanged(scene, old, geometry, events)
        assert scene.last_frame is None
