"""Finite beams from different origins share the actual collision geometry used by a camera."""

import numpy as np
import pytest

import phymeo as ph


def curved(kind, dims, *, rotation=(0.0, 0.0, 0.0, 1.0), body=None):
    key = ph.CollisionKey(0, 4, 1)
    shape = ph.CollisionShape(key, kind, (91,), curved=ph.CurvedSolid(kind, dims, rotation))
    return ph.CollisionModel([shape]), {
        key: ph.CollisionMotion([[0.0, 0.0, 0.0]], body_orientation_xyzw=body)
    }


def test_ellipsoid_rotation_and_independent_unscaled_beams():
    turn = (0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))
    model, poses = curved("ellipsoid", (2.0, 1.0, 0.5), body=turn)
    read = ph.raycast_scene(
        model,
        poses,
        [[0, 4, 0], [3, 0, 0], [0, 0, 0], [2, 4, 0]],
        [[0, -3, 0], [-2, 0, 0], [0, 0, 2], [0, 1, 0]],
    )
    np.testing.assert_allclose(read.distance_m[:3], [2, 2, 0.5], atol=1e-14)
    np.testing.assert_allclose(
        read.world_normals[:3], [[0, 1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-14
    )
    assert np.isinf(read.distance_m[3]) and read.shape_index[3] == -1
    assert read.keys[read.shape_index[0]] == model.shapes[0].key


def test_cylinder_barrel_caps_and_exit_after_ray_clip():
    model, poses = curved("cylinder", (0.5, 1.0))
    origins = [[0, 0, 3], [0, 2, 0], [0, 0, -3], [2, 2, 0]]
    rays = [[0, 0, -1], [0, -1, 0], [0, 0, 1], [1, 0, 0]]
    hit = ph.raycast_scene(model, poses, origins, rays)
    np.testing.assert_allclose(hit.distance_m[:3], [2, 1.5, 2])
    np.testing.assert_array_equal(hit.world_normals[:3], [[0, 0, 1], [0, 1, 0], [0, 0, -1]])
    assert hit.shape_index[3] == -1
    clipped = ph.raycast_scene(model, poses, [2, 0, 0], [[-1, 0, 0]], near_m=2)
    assert clipped.distance_m[0] == pytest.approx(2.5)
    assert clipped.world_normals[0, 0] == -1


def test_cone_tip_base_and_finite_slant_contact():
    model, poses = curved("cone", (1.0, 1.0))
    hit = ph.raycast_scene(
        model,
        poses,
        [[0, 0, 3], [2, 0, 0], [0, 0, -3], [1.01, 0, 3]],
        [[0, 0, -2], [-1, 0, 0], [0, 0, 1], [0, 0, -1]],
    )
    np.testing.assert_allclose(hit.distance_m[:3], [2, 1.5, 2], atol=1e-14)
    # At the apex no surface normal is unique.
    np.testing.assert_array_equal(hit.world_normals[0], [0, 0, 0])
    np.testing.assert_allclose(hit.world_normals[1], [2 / np.sqrt(5), 0, 1 / np.sqrt(5)])
    assert hit.shape_index[3] == -1


def test_heightfield_uses_registered_finite_diagonal_and_hides_other_world():
    grid = ph.Heightfield(((0.0, 0.0), (0.0, 1.0)), (1.0, 1.0))
    key = ph.CollisionKey(0, 2, 0)
    s = ph.CollisionShape(key, "heightfield", (3, 7, 11, 15), heightfield=grid)
    other = ph.CollisionShape(ph.CollisionKey(1, 2, 0), "sphere", (0,), radius=0.3)
    poses = {
        key: ph.CollisionMotion(grid.world_vertices()),
        other.key: ph.CollisionMotion([[0.2, 0.7, 1.5]]),
    }
    model = ph.CollisionModel([s, other])
    origins = np.array([[0.2, 0.7, 2], [0.7, 0.2, 2], [1.001, 0.5, 2]])
    read = ph.raycast_scene(model, poses, origins, [[0, 0, -1]] * 3)
    np.testing.assert_allclose(read.distance_m[:2], [1.8, 1.8], atol=1e-14)
    np.testing.assert_allclose(
        read.world_normals[:2], np.array([[-1, 0, 1], [0, -1, 1]]) / np.sqrt(2)
    )
    assert np.isinf(read.distance_m[-1])
    np.testing.assert_allclose(
        ph.raycast_scene(model, poses, origins[:1], [[0, 0, -1]], world=1).distance_m, [0.2]
    )
    poses[key] = ph.CollisionMotion(grid.world_vertices() + 0.01)
    with pytest.raises(ValueError, match="registered"):
        ph.raycast_scene(model, poses, origins[:1], [[0, 0, -1]])


def test_empty_selection_and_bad_unit_ray_data():
    model, poses = curved("ellipsoid", (1.0, 1.0, 1.0))
    assert ph.raycast_scene(model, poses, [0, 0, 0], np.empty((0, 3))).shape_index.size == 0
    for origins, directions in (
        ([0, 0, 0], [[0, 0, 0]]),
        ([[0, 0, 0]] * 2, [[1, 0, 0]]),
        ([0, float("nan"), 0], [[0, 1, 0]]),
    ):
        with pytest.raises(ValueError):
            ph.raycast_scene(model, poses, origins, directions)


def test_generator_is_a_surface_interval_and_respects_the_near_clip():
    model, poses = curved("cylinder", (1.0, 1.0))
    hit = ph.raycast_scene(model, poses, [1.0, 0.0, 2.0], [[0.0, 0.0, -1.0]], near_m=1.5)
    assert hit.distance_m.item() == 1.5
    np.testing.assert_array_equal(hit.world_normals, [[1.0, 0.0, 0.0]])


@pytest.mark.parametrize("magnitude", [7.0, 1e308])
def test_plane_normal_magnitude_does_not_change_distance_or_unit_normal(magnitude):
    plane = ph.CollisionShape(ph.CollisionKey(0, 1, 0), "plane", (2,), normal=(0, 0, magnitude))
    actual = ph.raycast_scene(
        ph.CollisionModel([plane]),
        {plane.key: ph.CollisionMotion([[0, 0, 0]])},
        [[0, 0, 2.0], [0, 0, -2.0]],
        [[0, 0, -4], [0, 0, 3]],
    )
    np.testing.assert_array_equal(actual.distance_m, [2, 2])
    np.testing.assert_array_equal(actual.world_normals, [[0, 0, 1], [0, 0, -1]])
