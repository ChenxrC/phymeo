"""Actual sampled-pad loads retain rotated payload faces through event publication.

The two cases change only registration IDs so _Records must reverse the pair.
All geometry, material weights and the one-step force oracle are fixed here;
no private pad-weight helper supplies expected values.
"""

from itertools import product
from math import sqrt

import numpy as np
import pytest

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_payload_scene import PadFace
from phymeo.runtime.robot_scene import RobotCommand

H = 1 / 1024
DEPTH = 1 / 1024
GEOMETRY_ATOL = 5e-14
FORCE_ATOL = 2e-11
ROBOT_ID = 10
PAYLOAD_CENTER = np.array([1 / 2, -1 / 4, 3 / 4])
PAYLOAD_HALF = np.array([1 / 8, 3 / 16, 1 / 4])
PAYLOAD_ROTATION = np.array([[3 / 5, -4 / 5, 0], [4 / 5, 3 / 5, 0], [0, 0, 1]])
PAD_ROTATION = np.array([[4 / 5, 3 / 5, 0], [-3 / 5, 4 / 5, 0], [0, 0, 1]])
PAD_CENTER_LOCAL = np.array([1 / 64, -1 / 32, 1 / 128])
PAD_HALF = np.array([1 / 8, 1 / 16, 1 / 8])
LINK_ORIGIN = np.array([2797 / 5120, -41 / 1280, 91 / 128])
PAD_POINT_LOCAL = np.array([1 / 320, -1 / 10, -3 / 128])
APPLICATION_POINT = np.array([2813 / 5120, -169 / 1280, 11 / 16])
PAYLOAD_SURFACE = np.array([11 / 20, -21 / 160, 11 / 16])
PAYLOAD_SURFACE_LOCAL = np.array([1 / 8, 1 / 32, -1 / 16])
AXIS = np.array([3 / 5, 4 / 5, 0])
NORMAL_ROBOT_TO_PAYLOAD = -AXIS
FORCE_ON_PAYLOAD = np.array([-6 / 5, -8 / 5, 0])
TORQUE_ON_PAYLOAD = np.array([-1 / 10, 3 / 40, 1 / 16])
TORQUE_ON_LINK = np.array([3 / 80, -9 / 320, 1 / 8])
PAD_WEIGHTS = {0: 15 / 64, 1: 9 / 64, 4: 25 / 64, 5: 15 / 64}
PAYLOAD_WEIGHTS = {4: 25 / 96, 5: 15 / 96, 6: 35 / 96, 7: 21 / 96}


def _scene(tmp_path, payload_id):
    path = tmp_path / "rotated_sampled_pad.urdf"
    origin = " ".join(f"{value:.17g}" for value in LINK_ORIGIN)
    path.write_text(
        '<robot name="rotated_sampled_pad"><link name="base"/><link name="pad_link">'
        '<inertial><mass value="2"/><inertia ixx=".125" iyy=".125" izz=".125" '
        'ixy="0" ixz="0" iyz="0"/></inertial>'
        '<collision name="pad"><origin xyz=".015625 -.03125 .0078125" '
        'rpy="0 0 -.6435011087932844"/>'
        '<geometry><box size=".25 .125 .25"/></geometry></collision></link>'
        '<joint name="slide" type="prismatic"><parent link="base"/>'
        f'<child link="pad_link"/><origin xyz="{origin}"/>'
        '<axis xyz=".6 .8 0"/>'
        '<limit lower="-1" upper="1" effort="16" velocity="16"/></joint></robot>',
        encoding="utf-8",
    )
    make = ContactSceneBuilder(gravity=(0, 0, 0))
    make.add_urdf(
        ROBOT_ID,
        path,
        robot_state=RobotState(np.zeros(1), np.zeros(1)),
        kp=[0],
        kd=[0],
        pads=[PadFace.named("pad_link", "pad", 1, -1, ((1 / 32, -1 / 32),))],
    )
    make.add_box(
        payload_id,
        PAYLOAD_HALF,
        mass=3,
        position=PAYLOAD_CENTER,
        orientation_xyzw=(0, 0, 1 / sqrt(5), 2 / sqrt(5)),
    )
    law = RobotContactMaterial(2048, 0, 0)
    make.pair(ROBOT_ID, payload_id, law)
    return make, law


def _assert_weights(actual, expected, vertices, surface):
    # Exact support matters: even a tiny positive weight on the opposite face
    # changes the declared material feature. Values use independent rational data.
    assert len(actual) == len(expected) == 4
    assert {node for node, _ in actual} == set(expected)
    assert all(weight > 0 for _, weight in actual)
    assert sum(weight for _, weight in actual) == pytest.approx(1, rel=0, abs=GEOMETRY_ATOL)
    for node, weight in actual:
        assert weight == pytest.approx(expected[node], rel=0, abs=GEOMETRY_ATOL)
    rebuilt = sum(weight * vertices[node] for node, weight in actual)
    np.testing.assert_allclose(rebuilt, surface, atol=GEOMETRY_ATOL, rtol=0)


@pytest.mark.parametrize(
    "payload_id,robot_first", [(20, True), (0, False)], ids=["robot-first", "payload-first"]
)
def test_rotated_payload_sample_publishes_both_material_faces_and_reversed_keys(
    tmp_path, payload_id, robot_first
):
    make, law = _scene(tmp_path, payload_id)
    # Independent authored corner order and transforms, including the pad's
    # nonzero collision offset and local rotation. No runtime helper is the oracle.
    pad_box_vertices = np.array(list(product(*[(-h, h) for h in PAD_HALF])))
    pad_local_vertices = PAD_CENTER_LOCAL + pad_box_vertices @ PAD_ROTATION.T
    pad_world_vertices = LINK_ORIGIN + pad_local_vertices
    payload_local_vertices = np.array(list(product(*[(-h, h) for h in PAYLOAD_HALF])))
    payload_world_vertices = PAYLOAD_CENTER + payload_local_vertices @ PAYLOAD_ROTATION.T
    # At the sample, the pad is inside the payload but has a unique +X exit face.
    interior_local = np.array([127 / 1024, 1 / 32, -1 / 16])
    np.testing.assert_allclose(
        PAYLOAD_CENTER + PAYLOAD_ROTATION @ interior_local,
        APPLICATION_POINT,
        atol=GEOMETRY_ATOL,
        rtol=0,
    )
    np.testing.assert_allclose(
        PAYLOAD_HALF - np.abs(interior_local), [DEPTH, 5 / 32, 3 / 16], atol=0, rtol=0
    )
    np.testing.assert_allclose(
        LINK_ORIGIN + PAD_POINT_LOCAL, APPLICATION_POINT, atol=GEOMETRY_ATOL, rtol=0
    )

    with make.build() as scene:
        robot_key = scene.shape_key(ROBOT_ID, "pad_link", "pad")
        payload_key = scene.body_keys[0, payload_id][0]
        assert robot_key.body == (0 if robot_first else 1)
        assert payload_key.body == payload_id
        assert (robot_key < payload_key) is robot_first
        first, second = (robot_key, payload_key) if robot_first else (payload_key, robot_key)
        entry = scene.geometry()
        shapes = {shape.key: shape for shape in entry.model.shapes}
        for key in (robot_key, payload_key):
            assert shapes[key].kind == "hull"
            assert shapes[key].vertex_ids == tuple(range(8))
        np.testing.assert_allclose(
            entry.poses[robot_key].start, pad_world_vertices, atol=GEOMETRY_ATOL, rtol=0
        )
        np.testing.assert_allclose(
            entry.poses[payload_key].start, payload_world_vertices, atol=GEOMETRY_ATOL, rtol=0
        )
        initial_events = scene.contact_events
        assert initial_events.active == initial_events.events == ()
        reading = scene.step(H, actions={ROBOT_ID: RobotCommand([0], feedforward=[0])})
        view = scene.contact_events
        assert view.step_index == reading.step_number == 1
        assert view.sample_time_s == reading.load_time_s == 0
        assert view.commit_time_s == reading.time_s == H
        (event,) = view.events
        (record,) = view.active
        assert event.kind == "begin" and event.record is record
        assert record.identity.first == first and record.identity.second == second
        assert record.identity.generation == 1
        (component,) = record.components
        assert component.material_kind == "robot_contact_material" and component.law == law
        assert len(reading.loads.contacts) == len(reading.contacts) == 1
        assert component.load == reading.loads.contacts[0]
        load = component.load
        (site,) = load.sites
        (material,) = component.material
        detail = reading.contacts[0]
        assert detail["channel"] == "robot-pad" and detail["feature"] == "sampled-pad"
        assert detail["first"] == first and detail["second"] == second
        assert load.first == first and load.second == second
        sign = 1 if robot_first else -1
        expected_normal = sign * NORMAL_ROBOT_TO_PAYLOAD
        expected_force = sign * FORCE_ON_PAYLOAD
        np.testing.assert_allclose(load.normal, expected_normal, atol=GEOMETRY_ATOL, rtol=0)
        np.testing.assert_allclose(
            detail["normal_world"], expected_normal, atol=GEOMETRY_ATOL, rtol=0
        )
        np.testing.assert_allclose(site.force_on_second_n, expected_force, atol=FORCE_ATOL, rtol=0)
        np.testing.assert_allclose(
            detail["force_on_second_N"], expected_force, atol=FORCE_ATOL, rtol=0
        )
        np.testing.assert_allclose(
            detail["force_on_first_N"], -expected_force, atol=FORCE_ATOL, rtol=0
        )
        assert load.force_n == pytest.approx(2, rel=0, abs=FORCE_ATOL)
        assert site.force_n == pytest.approx(2, rel=0, abs=FORCE_ATOL)
        assert load.gap_m == pytest.approx(-DEPTH, rel=0, abs=GEOMETRY_ATOL)
        assert material.fraction == site.fraction == 1
        assert site.rate_m_per_s == 0 and load.power_w == 0
        for side in ("first", "second"):
            np.testing.assert_allclose(
                getattr(site, f"application_point_{side}_m"),
                APPLICATION_POINT,
                atol=GEOMETRY_ATOL,
                rtol=0,
            )
            np.testing.assert_allclose(
                detail[f"application_point_{side}_m"],
                APPLICATION_POINT,
                atol=GEOMETRY_ATOL,
                rtol=0,
            )

        # Each event address must preserve the complete corresponding _Records
        # material, not just a new local point fitted from the accepted end pose.
        by_key = {}
        for key, side, owner in (
            (first, "first", record.first_owner),
            (second, "second", record.second_owner),
        ):
            address = getattr(material, side)
            surface = np.asarray(getattr(site.material, f"{side}_point"))
            weights = getattr(site.material, f"{side}_weights")
            assert address.weights == weights == detail[f"{side}_weights"]
            assert owner.world == 0
            np.testing.assert_allclose(
                detail[f"{side}_surface_m"], surface, atol=GEOMETRY_ATOL, rtol=0
            )
            by_key[key] = (address, surface, owner)
        robot_address, robot_surface, robot_owner = by_key[robot_key]
        payload_address, payload_surface, payload_owner = by_key[payload_key]
        assert robot_owner.body == ROBOT_ID and robot_owner.kind == "robot_link"
        assert robot_owner.link_name == "pad_link" and robot_owner.link_index == 0
        assert robot_owner.local_frame == "robot_model_link"
        assert payload_owner.body == payload_id and payload_owner.kind == "rigid"
        assert payload_owner.local_frame == "rigid_com"
        assert robot_address.kind == "robot_link" and payload_address.kind == "rigid"
        np.testing.assert_allclose(robot_surface, APPLICATION_POINT, atol=GEOMETRY_ATOL, rtol=0)
        np.testing.assert_allclose(payload_surface, PAYLOAD_SURFACE, atol=GEOMETRY_ATOL, rtol=0)
        np.testing.assert_allclose(
            robot_address.local_point_m, PAD_POINT_LOCAL, atol=GEOMETRY_ATOL, rtol=0
        )
        np.testing.assert_allclose(
            payload_address.local_point_m, PAYLOAD_SURFACE_LOCAL, atol=GEOMETRY_ATOL, rtol=0
        )
        _assert_weights(robot_address.weights, PAD_WEIGHTS, pad_world_vertices, APPLICATION_POINT)
        _assert_weights(
            payload_address.weights, PAYLOAD_WEIGHTS, payload_world_vertices, PAYLOAD_SURFACE
        )
        np.testing.assert_allclose(
            sum(weight * pad_local_vertices[node] for node, weight in robot_address.weights),
            robot_address.local_point_m,
            atol=GEOMETRY_ATOL,
            rtol=0,
        )
        np.testing.assert_allclose(
            sum(weight * payload_local_vertices[node] for node, weight in payload_address.weights),
            payload_address.local_point_m,
            atol=GEOMETRY_ATOL,
            rtol=0,
        )
        assert float((payload_surface - robot_surface) @ NORMAL_ROBOT_TO_PAYLOAD) == pytest.approx(
            -DEPTH, rel=0, abs=GEOMETRY_ATOL
        )
        # The material surface is on +X; the common force point remains inside it.
        assert np.asarray(payload_address.local_point_m)[0] == pytest.approx(
            1 / 8, rel=0, abs=GEOMETRY_ATOL
        )
        assert float(np.linalg.norm(payload_surface - APPLICATION_POINT)) == pytest.approx(
            DEPTH, rel=0, abs=GEOMETRY_ATOL
        )

        # A real accepted step proves the force was applied once, using the same
        # off-centre application point for both owners rather than either skin.
        rigid_row = scene.rigid_rows[0, payload_id]
        np.testing.assert_allclose(
            reading.loads.inputs.rigid.force[rigid_row], FORCE_ON_PAYLOAD, atol=FORCE_ATOL, rtol=0
        )
        np.testing.assert_allclose(
            reading.loads.inputs.rigid.torque[rigid_row], TORQUE_ON_PAYLOAD, atol=FORCE_ATOL, rtol=0
        )
        robot = reading.robots[0, ROBOT_ID]
        np.testing.assert_allclose(
            robot.link_wrenches_world[0],
            np.r_[-FORCE_ON_PAYLOAD, TORQUE_ON_LINK],
            atol=FORCE_ATOL,
            rtol=0,
        )
        np.testing.assert_allclose(robot.entry_q, [0], atol=0, rtol=0)
        np.testing.assert_allclose(robot.entry_v, [0], atol=0, rtol=0)
        np.testing.assert_allclose(robot.drive_effort, [0], atol=0, rtol=0)
        np.testing.assert_allclose(robot.contact_effort, [2], atol=FORCE_ATOL, rtol=0)
        np.testing.assert_allclose(robot.v, [H], atol=GEOMETRY_ATOL, rtol=0)
        np.testing.assert_allclose(robot.q, [H * H], atol=GEOMETRY_ATOL, rtol=0)
        np.testing.assert_allclose(
            reading.rigid.velocity[rigid_row], H * FORCE_ON_PAYLOAD / 3, atol=GEOMETRY_ATOL, rtol=0
        )
        inertia = np.array([25 / 256, 5 / 64, 13 / 256])
        expected_omega = H * PAYLOAD_ROTATION @ (np.array([0, 1 / 8, 1 / 16]) / inertia)
        np.testing.assert_allclose(
            reading.rigid.angular_velocity[rigid_row], expected_omega, atol=GEOMETRY_ATOL, rtol=0
        )
        np.testing.assert_allclose(
            2 * robot.v[0] * AXIS + 3 * reading.rigid.velocity[rigid_row],
            0,
            atol=GEOMETRY_ATOL,
            rtol=0,
        )
        assert scene.contact_events is view
