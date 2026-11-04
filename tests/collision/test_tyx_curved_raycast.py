"""Independent SI roots and physical normals for three finite analytic solids."""

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends.tyx_raycast import TyxRaycaster


def scene(kind, dims, *, material=(0, 0, 0, 1), body=None, center=(0, 0, 0)):
    key = ph.CollisionKey(2, 9, 1)
    shape = ph.CollisionShape(key, kind, (8,), curved=ph.CurvedSolid(kind, dims, material))
    return ph.CollisionModel([shape]), {key: ph.CollisionMotion([center], body_orientation_xyzw=body)}


def check(cast, model, poses, origins, rays, distances, normals, **clip):
    data = cast.query(model, poses, origins, rays, world=2, **clip)
    gold = ph.raycast_scene(model, poses, origins, rays, world=2, **clip)
    labels = np.where(np.isfinite(distances), 0, -1)
    for src in (data, gold):
        np.testing.assert_array_equal(src.shape_index, labels)
        np.testing.assert_allclose(src.distance_m, distances, atol=4e-6, rtol=3e-6)
        np.testing.assert_allclose(src.world_normals, normals, atol=5e-6, rtol=0)
    return data


def test_rotated_triaxial_ellipsoid_physical_normal_clipped_exit_and_tangent(tyx_mode, tmp_path):
    model, poses = scene("ellipsoid", (2, 1, 0.5), center=(4, -2, 1))
    c = np.array([4, -2, 1])
    x, y = 1.2, 0.8
    n = np.array([0.3, 0.8, 0]) / np.hypot(0.3, 0.8)
    origins = c + [[x + 2, y, 0], [0, 0, 0], [2, 0, 2], [2.125, 0, 2]]
    rays = [[-5, 0, 0], [0, 0, -7], [0, 0, -2], [0, 0, -2]]
    with TyxRaycaster(tyx_mode, cache_root=tmp_path) as cast:
        check(cast, model, poses, origins, rays, [2, 0.5, 2, np.inf],
              [n, [0, 0, -1], [1, 0, 0], [0, 0, 0]])
        check(cast, model, poses, c + [4, 0, 0], [[-8, 0, 0]], [6], [[-1, 0, 0]], near_m=3, far_m=6)
        # Independent 3/5,4/5 unit quaternion about +Z rotates the X axis to (7,24,0)/25.
        rot = np.array([[7, -24, 0], [24, 7, 0], [0, 0, 25]]) / 25
        model, poses = scene("ellipsoid", (2, 1, 0.5), body=(0, 0, 0.6, 0.8), center=c)
        check(cast, model, poses, c + np.array([x + 2, y, 0]) @ rot.T, [np.array([-5, 0, 0]) @ rot.T],
              [2], [n @ rot.T])


def test_cylinder_side_caps_generator_interval_rim_inside_and_near_miss(tyx_mode, tmp_path):
    model, poses = scene("cylinder", (0.5, 1))
    with TyxRaycaster(tyx_mode, cache_root=tmp_path) as cast:
        check(cast, model, poses,
              [[0, 2, 0], [0.25, 0, 3], [0, 0, -2], [0, 0, 0], [0.625, 0, 3], [2, 0, 1.125]],
              [[0, -4, 0], [0, 0, -3], [0, 0, 2], [0, 3, 0], [0, 0, -1], [-1, 0, 0]],
              [1.5, 2, 1, 0.5, np.inf, np.inf],
              [[0, 1, 0], [0, 0, 1], [0, 0, -1], [0, 1, 0], [0, 0, 0], [0, 0, 0]])
        # Side precedes cap on the rim; a clipped generator returns its actual near point.
        check(cast, model, poses, [0.5, 0, 2], [[0, 0, -8]], [1.25], [[1, 0, 0]], near_m=1.25)
        check(cast, model, poses, [0, 0, 3], [[0, 0, -2]], [4], [[0, 0, -1]], near_m=3)


def test_cone_linear_root_apex_base_and_finite_slant(tyx_mode, tmp_path):
    model, poses = scene("cone", (1, 0.5))
    n = np.array([1, 0, 1]) / np.sqrt(2)
    with TyxRaycaster(tyx_mode, cache_root=tmp_path) as cast:
        check(cast, model, poses, [[2, 0, 0], [0.25, 0, -2], [0, 0, 3], [0, 0, 0], [1.125, 0, 3]],
              [[-3, 0, 0], [0, 0, 4], [0, 0, -2], [2, 0, 0], [0, 0, -1]],
              [1.5, 1.5, 2.5, 0.5, np.inf], [n, [0, 0, -1], [0, 0, 0], n, [0, 0, 0]])
        # A ray parallel to the opposite generator leaves a linear equation for this side.
        check(cast, model, poses, [0.5, 0, 0.25], [[-1, 0, -1]], [np.sqrt(2) / 8], [n])
        # At the near plane this beam already lies on the positive-X generator.
        check(cast, model, poses, [0, 0, 0.5], [[1, 0, -1]], [0.25], [n], near_m=0.25)


def test_occlusion_competes_with_sphere_flat_and_disjoint_world_at_each_pose(tyx_mode, tmp_path):
    ell = ph.CollisionShape(ph.CollisionKey(2, 8, 1), "ellipsoid", (0,),
                           curved=ph.CurvedSolid("ellipsoid", (0.5, 0.8, 0.25)))
    ball = ph.CollisionShape(ph.CollisionKey(2, 1, 1), "sphere", (0,), 0.5)
    plane = ph.CollisionShape(ph.CollisionKey(2, 6, 0), "plane", (8,), normal=(0, 0, 7))
    hidden = ph.CollisionShape(ph.CollisionKey(1, 8, 1), "sphere", (0,), 1)
    model = ph.CollisionModel([ell, ball, plane, hidden])
    pose = {ell.key: ph.CollisionMotion([[0, 0, 0]]), ball.key: ph.CollisionMotion([[0, 0, -0.25]]),
            plane.key: ph.CollisionMotion([[0, 0, -3]])}
    rays = [[0, 0, -3]] * 3
    origins = [[0, 0, 2], [0, 0.625, 2], [2, 0, 2]]
    with TyxRaycaster(tyx_mode, cache_root=tmp_path) as cast:
        first = cast.query(model, pose, origins, rays, world=2)
        np.testing.assert_array_equal(first.shape_index, [1, 3, 2])
        assert first.distance_m[0] == 1.75 and first.distance_m[2] == 5
        pose[ell.key] = ph.CollisionMotion([[0, 0, 1]])
        advanced = cast.query(model, pose, origins, rays, world=2)
        np.testing.assert_array_equal(advanced.shape_index, [3, 3, 2])
        assert advanced.distance_m[0] == 0.75
        np.testing.assert_array_equal(first.shape_index, [1, 3, 2])


def test_curved_surface_resolution_fails_before_dispatch(tyx_mode, tmp_path):
    model, poses = scene("ellipsoid", (1, 1, 1e-15))
    with TyxRaycaster(tyx_mode, cache_root=tmp_path) as cast:
        with pytest.raises(ValueError, match="FP32"):
            cast.query(model, poses, [0, 0, 2], [[0, 0, -1]], world=2)
        assert cast.stats()["completed_launches"] == 0
