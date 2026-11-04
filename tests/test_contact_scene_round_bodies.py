"""Round rigid materials through the same scene and force ports as tools and sheets."""

import numpy as np
import pytest

import phymeo as ph


def resting_box_ball(backend, cache):
    b = ph.ContactSceneBuilder(gravity=[0, 0, 0])
    b.add_sphere(9, 1 / 8, mass=1, position=(1 / 16, 0, 9 / 32))
    b.add_box(2, (1 / 2, 3 / 8, 1 / 8), mass=2)
    b.pair(9, 2, ph.NormalLayer(1 / 16, 128))
    return b.build(backend=backend, cache_root=cache)


def check_force_pair(backend, cache):
    with resting_box_ball(backend, cache) as scene:
        reading, geometry = scene.observe_scene()
        assert geometry.time_s == 0.0 and [s.kind for s in geometry.model.shapes] == [
            "hull",
            "sphere",
        ]
        box, ball = scene.rigid_rows[0, 2], scene.rigid_rows[0, 9]
        (contact,) = reading.loads.contacts
        assert contact.gap_m == pytest.approx(1 / 32, abs=2e-7)
        assert contact.force_n == pytest.approx(4, abs=3e-5)
        assert contact.normal == (0.0, 0.0, 1.0)
        assert reading.loads.inputs.rigid.torque[ball].tolist() == [0.0, 0.0, 0.0]
        np.testing.assert_allclose(reading.loads.inputs.rigid.force[box], [0, 0, -4], atol=3e-5)
        np.testing.assert_allclose(reading.loads.inputs.rigid.torque[box], [0, 1 / 4, 0], atol=3e-5)
        h = 1 / 64
        f = scene.step(h)
        np.testing.assert_allclose(f.state.rigid.velocity[ball], [0, 0, 4 * h], atol=2e-6)
        np.testing.assert_allclose(
            2 * f.state.rigid.velocity[box] + f.state.rigid.velocity[ball], 0, atol=2e-6
        )
        iy = 2 / 3 * ((1 / 2) ** 2 + (1 / 8) ** 2)
        assert f.state.rigid.angular_velocity[box, 1] == pytest.approx(h / (4 * iy), abs=2e-5)
        assert scene.geometry().time_s == h


@pytest.mark.parametrize(
    "normal, point",
    [
        ((0.6, 0.8, 0), (0.5, 0.375, 0)),
        ((1 / 3, 2 / 3, 2 / 3), (0.5, 0.375, 0.125)),
    ],
)
def test_finite_edge_and_vertex_publish_original_material_point(normal, point):
    b = ph.ContactSceneBuilder(gravity=[0, 0, 0])
    center = np.asarray(point) + (5 / 32) * np.asarray(normal)
    b.add_sphere(1, 1 / 8, mass=1, position=center)
    b.add_box(8, (1 / 2, 3 / 8, 1 / 8), mass=None)
    b.pair(8, 1, ph.NormalLayer(1 / 16, 128))
    with b.build() as scene:
        (contact,) = scene.sample().loads.contacts
        (site,) = contact.sites
        assert contact.gap_m == pytest.approx(1 / 32, abs=2e-7)
        np.testing.assert_allclose(contact.normal, -np.asarray(normal), atol=5e-4)
        np.testing.assert_allclose(site.application_point_second_m, point, atol=3e-5)


def test_published_force_and_motion_survive_edits_to_returned_native_arrays():
    with resting_box_ball("host", None) as scene:
        external = scene.inputs()
        external.rigid.force[1, 1] = 1 / 16
        f = scene.step(1 / 64, external)
        original = f.state.rigid.position.copy()
        f.state.rigid.position[:] = 40
        f.entry.rigid.position[:] = -30
        f.external.rigid.force[:] = 80
        f.loads.inputs.rigid.force[:] = 0
        saved = scene.last_frame
        np.testing.assert_array_equal(saved.state.rigid.position, original)
        assert saved.entry.rigid.position[0, 0] == 0
        assert saved.external.rigid.force[1, 1] == 1 / 16
        assert saved.loads.inputs.rigid.force[1, 2] == 4
        saved.loads.inputs.rigid.torque[:] = 24
        assert scene.last_frame.loads.inputs.rigid.torque[0, 1] == 1 / 4
        assert scene.last_frame.path is f.path
        np.testing.assert_array_equal(scene.state.rigid.position, original)


@pytest.mark.parametrize("scale", [1 + 2**-24, -1 - 2**-24])
def test_fixed_box_drift_compares_the_rotation_represented_by_its_quaternion(scale):
    b = ph.ContactSceneBuilder(gravity=[0, 0, 0])
    b.add_box(0, (0.5, 0.375, 0.125), mass=None, orientation_xyzw=(0, 0, 0, scale))
    b.add_sphere(1, 0.125, mass=1, position=(0, 0, 9 / 32))
    b.pair(0, 1, ph.NormalLayer(1 / 16, 128))
    with b.build() as scene:
        assert scene.sample().loads.contacts[0].force_n == 4
        frame = scene.step(1 / 64)
        assert frame.state.rigid.velocity[1, 2] == pytest.approx(1 / 16)
        assert frame.state.rigid.orientation[0, 3] == scale


@pytest.mark.tyx
def test_oblique_corner_advances_the_real_sphere(tyx_mode, tmp_path):
    b = ph.ContactSceneBuilder(gravity=[0, 0, 0])
    unit = np.array([1, 2, 2]) / 3
    b.add_box(0, (0.5, 0.375, 0.125), mass=None)
    b.add_sphere(1, 0.125, mass=1, position=np.array([0.5, 0.375, 0.125]) + 5 / 32 * unit)
    b.pair(0, 1, ph.NormalLayer(1 / 16, 128))
    with b.build(backend=tyx_mode, cache_root=tmp_path) as scene:
        f = scene.step(1 / 128)
        np.testing.assert_allclose(f.state.rigid.velocity[1], 4 / 128 * unit, atol=1e-6)
        np.testing.assert_allclose(f.state.rigid.angular_velocity[1], 0, atol=1e-6)


def test_box_ball_actual_reaction():
    check_force_pair("host", None)


@pytest.mark.tyx
def test_box_ball_actual_device_reaction(tyx_mode, tmp_path):
    check_force_pair(tyx_mode, tmp_path)


def test_sliding_spheres_use_equal_opposite_tractions_and_shared_application_point():
    b = ph.ContactSceneBuilder(gravity=[0, 0, 0])
    b.add_sphere(31, 1 / 8, mass=2)
    b.add_sphere(33, 1 / 8, mass=1, position=(9 / 32, 0, 0), velocity=(0, 0.5, 0))
    b.pair(31, 33, ph.NormalLayer(1 / 16, 128, friction=0.5))
    with b.build() as scene:
        first = scene.step(1 / 128)
    (contact,) = first.loads.contacts
    (site,) = contact.sites
    assert contact.force_n == pytest.approx(4)
    assert site.application_point_first_m == site.application_point_second_m
    f = first.loads.inputs.rigid.force
    np.testing.assert_allclose(f[0] + f[1], 0, atol=1e-12)
    assert f[1, 0] == pytest.approx(4) and -2.0 - 1e-12 <= f[1, 1] < -0.5
    np.testing.assert_allclose(
        first.loads.inputs.rigid.torque.sum(axis=0)
        + np.cross(first.entry.rigid.position, f).sum(axis=0),
        0,
        atol=1e-12,
    )
    assert (
        first.state.rigid.angular_velocity[0, 2] > 0
        and first.state.rigid.angular_velocity[1, 2] > 0
    )


def test_spatially_rotated_box_ball_scene_keeps_the_same_wrench():
    b = ph.ContactSceneBuilder(gravity=[0, 0, 0])
    origin = np.array([3.0, -2.0, 1.0])
    b.add_box(5, (1 / 2, 3 / 8, 1 / 8), mass=2, position=origin, orientation_xyzw=(0.5,) * 4)
    b.add_sphere(2, 1 / 8, mass=1, position=origin + [9 / 32, 1 / 16, 0])
    b.pair(2, 5, ph.NormalLayer(1 / 16, 128))
    with b.build() as scene:
        f = scene.step(1 / 64)
        row = scene.rigid_rows[0, 5]
        np.testing.assert_allclose(f.loads.inputs.rigid.force[row], [-4, 0, 0], atol=3e-5)
        np.testing.assert_allclose(f.loads.inputs.rigid.torque[row], [0, 0, 1 / 4], atol=3e-5)


def test_rotated_ellipsoid_inertia_and_depth_are_from_the_same_solid():
    b = ph.ContactSceneBuilder(gravity=[0, 0, 0])
    b.add_ellipsoid(
        9, (1 / 2, 1 / 4, 1 / 8), mass=3, position=(1, 2, 3), orientation_xyzw=(0.5,) * 4
    )
    with b.build() as scene:
        geometry = scene.geometry()
        rays = ph.raycast_scene(geometry.model, geometry.poses, [[1, 4, 3]], [[0, -1, 0]])
        assert rays.distance_m[0] == pytest.approx(1.5)
        loads = scene.inputs()
        loads.rigid.torque[0, 1] = 1
        h = 1 / 64
        f = scene.step(h, loads)
        ix = 3 / 5 * ((1 / 4) ** 2 + (1 / 8) ** 2)
        np.testing.assert_allclose(f.state.rigid.angular_velocity[0], [0, h / ix, 0], atol=1e-12)


def test_sphere_and_rotated_ellipsoid_share_the_material_law():
    b = ph.ContactSceneBuilder(gravity=[0, 0, 0])
    b.add_ellipsoid(19, (1 / 2, 1 / 4, 1 / 8), mass=2, orientation_xyzw=(0.5,) * 4)
    b.add_sphere(2, 1 / 8, mass=1, position=(0, 21 / 32, 0))
    b.pair(2, 19, ph.NormalLayer(1 / 16, 128))
    with b.build() as scene:
        f = scene.step(1 / 128)
        (contact,) = f.loads.contacts
        assert contact.gap_m == pytest.approx(1 / 32, abs=2e-7)
        np.testing.assert_allclose(f.loads.inputs.rigid.force[:, 1], [4, -4], atol=3e-5)
        np.testing.assert_allclose(f.state.rigid.velocity[:, 1], [4 / 128, -2 / 128], atol=2e-6)


def test_round_registration_failure_leaves_its_body_address_available():
    b = ph.ContactSceneBuilder()
    with pytest.raises(ValueError):
        b.add_ellipsoid(0, (0.1, 0.2, 0), mass=1)
    with pytest.raises(ValueError):
        b.add_sphere(0, -1, mass=1)
    b.add_sphere(0, 0.125, mass=1)
    with pytest.raises(ValueError, match="duplicate"):
        b.add_ellipsoid(0, (0.1, 0.2, 0.3), mass=1)


def test_ball_crossing_a_finite_box_rejects_without_publishing():
    b = ph.ContactSceneBuilder(gravity=[0, 0, 0])
    b.add_box(0, (0.125, 0.25, 0.375), mass=None)
    b.add_sphere(1, 0.0625, mass=1, position=(-1, 0, 0), velocity=(4, 0, 0))
    b.pair(0, 1, ph.NormalLayer(1 / 32, 8))
    with b.build() as scene:
        assert not scene.sample().loads.contacts
        with pytest.raises(ph.SolverFailure, match="core path rejected"):
            scene.step(0.5)
        assert scene.geometry().time_s == 0 and scene.last_frame is None
        assert scene.state.rigid.position[1, 0] == -1
