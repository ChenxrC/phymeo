"""Independent metric and material observations from arbitrary finite beams."""

from itertools import product

import numpy as np
import pytest

import phymeo as ph


def operator(mode, cache, **limits):
    from phymeo.backends.tyx_raycast import TyxRaycaster

    return TyxRaycaster(mode=mode, cache_root=cache, **limits)


def scene(*entries):
    return ph.CollisionModel([s for s, _ in entries]), {
        s.key: ph.CollisionMotion(p) for s, p in entries
    }


def verify(hit, distance, labels, normals):
    np.testing.assert_array_equal(hit.shape_index, labels)
    np.testing.assert_array_equal(np.isposinf(hit.distance_m), np.isposinf(distance))
    np.testing.assert_allclose(hit.distance_m, distance, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(hit.world_normals, normals, atol=2e-6, rtol=0)
    for value in (hit.distance_m, hit.shape_index, hit.world_normals):
        with pytest.raises(ValueError):
            value.setflags(write=True)


def test_metric_beams_clip_to_second_root_and_preserve_global_shape_ties(tyx_mode, tmp_path):
    other = ph.CollisionShape(ph.CollisionKey(0, 1, 3), "sphere", (42,), 1)
    second = ph.CollisionShape(ph.CollisionKey(6, 7, 2), "sphere", (1,), 1)
    first = ph.CollisionShape(ph.CollisionKey(6, 1, 3), "rounded_particle", (13,), 1)
    model, poses = scene((second, [[0, 0, 0]]), (other, [[2, 0, 0]]), (first, [[0, 0, 0]]))
    origins = [[3, 0, 0], [0, 0, 0], [0, 1, 3], [2, 2, 2]]
    rays = [[-20, 0, 0], [0, -2, 0], [0, 0, -3], [1, 0, 0]]
    with operator(tyx_mode, tmp_path) as cast:
        hit = cast.query(model, poses, origins, rays, world=6, far_m=3)
        assert hit.keys == (other.key, first.key, second.key)
        verify(hit, [2, 1, 3, np.inf], [1, 1, 1, -1],
               [[1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, 0]])
        exit_hit = cast.query(model, poses, origins[:1], rays[:1], world=6, near_m=2.5, far_m=4)
        verify(exit_hit, [4], [1], [[-1, 0, 0]])
        near_hit = cast.query(model, poses, origins[:1], rays[:1], world=6, near_m=2, far_m=2.5)
        verify(near_hit, [2], [1], [[1, 0, 0]])
        before = cast.stats()
        assert before["completed_launches"] == 3
        if tyx_mode == "cuda":
            assert before["uploads"] > 0 and before["downloads"] > 0
            assert before["uploaded_bytes"] > before["downloaded_bytes"] > 0
        else:
            assert before["uploads"] == before["downloads"] == before["downloaded_bytes"] == 0
        empty = cast.query(model, {}, origins, rays, world=5)
        verify(empty, [np.inf] * 4, [-1] * 4, [[0, 0, 0]] * 4)
        assert cast.stats() == before


def test_finite_capsule_barrel_hemispheres_no_internal_discs_or_generator_extension(tyx_mode, tmp_path):
    s = ph.CollisionShape(ph.CollisionKey(0, 7, 2), "capsule", (89, 7), 0.25)
    model, poses = scene((s, [[-1, 0, 0], [1, 0, 0]]))
    origins = [[0, 2, 0], [3, 0, 0], [1.15, 1, 0], [1.3125, 2, 0], [0, 0, 0]]
    rays = [[0, -4, 0], [-3, 0, 0], [0, -1, 0], [0, -1, 0], [-1, 0, 0]]
    with operator(tyx_mode, tmp_path) as cast:
        hit = cast.query(model, poses, origins, rays)
        verify(hit, [1.75, 1.75, 0.8, np.inf, 1.25], [0, 0, 0, -1, 0],
               [[0, 1, 0], [1, 0, 0], [0.6, 0.8, 0], [0, 0, 0], [-1, 0, 0]])
        # Changes at both ends are read from the new pose.
        poses[s.key] = ph.CollisionMotion([[0, 4, 0], [0, 4, 0]])
        verify(cast.query(model, poses, [[0, 2, 0]], [[0, 7, 0]]), [1.75], [0], [[0, -1, 0]])


def test_flat_face_occlusion_both_sides_hull_and_pose_translation(tyx_mode, tmp_path):
    tri = ph.CollisionShape(ph.CollisionKey(1, 3, 0), "triangle", (91, 4, 27))
    hull = ph.CollisionShape(ph.CollisionKey(1, 9, 0), "hull", tuple(range(8)))
    plane = ph.CollisionShape(ph.CollisionKey(1, 4, 9), "plane", (0,), normal=(0, 0, 7))
    vertices = np.asarray(list(product([3.5, 4.5], [-0.5, 0.5], [0, 0.5])))
    shift = np.array([2.0**30, -2.0**30, 1.0])
    model, poses = scene(
        (tri, shift + [(0, 0, 1), (2, 0, 1), (0, 2, 1)]),
        (hull, shift + vertices),
        (plane, shift + [[0, 0, -1]]),
    )
    origins = shift + [[0.25, 0.25, 2], [0.25, 0.25, 0], [1.25, 1.25, 2],
                       [4, 0.125, 2], [8, 2, -3], [0, 0, 8]]
    rays = [[0, 0, -4], [0, 0, 1], [0, 0, -1], [0, 0, -1], [0, 0, 1], [1, 0, 0]]
    with operator(tyx_mode, tmp_path) as cast:
        hit = cast.query(model, poses, origins, rays, world=1)
        verify(hit, [1, 1, 3, 1.5, 2, np.inf], [0, 0, 1, 2, 1, -1],
               [[0, 0, 1], [0, 0, -1], [0, 0, 1], [0, 0, 1], [0, 0, -1], [0, 0, 0]])
        baseline = ph.raycast_scene(model, poses, origins, rays, world=1)
        np.testing.assert_array_equal(hit.shape_index, baseline.shape_index)
        np.testing.assert_allclose(hit.distance_m, baseline.distance_m, rtol=2e-6, atol=2e-6)


def test_heightfield_finite_diagonal_face_normals_and_other_world_rejection(tyx_mode, tmp_path):
    field = ph.Heightfield(((0, 0), (0, 1)), (1, 1))
    terrain = ph.CollisionShape(ph.CollisionKey(3, 2, 1), "heightfield", (9, 13, 5, 7), heightfield=field)
    curve = ph.CollisionShape(ph.CollisionKey(1, 2, 1), "ellipsoid", (8,),
                              curved=ph.CurvedSolid("ellipsoid", (1, 2, 3)))
    model, poses = scene((curve, [[0, 0, 0]]), (terrain, field.world_vertices()))
    origins = [[0.25, 0.75, 2], [0.75, 0.25, 2], [1.125, 0.5, 2]]
    rays = [[0, 0, -1]] * 3
    with operator(tyx_mode, tmp_path) as cast:
        hit = cast.query(model, {terrain.key: poses[terrain.key]}, origins, rays, world=3)
        n = 1 / np.sqrt(2)
        verify(hit, [1.75, 1.75, np.inf], [1, 1, -1], [[-n, 0, n], [0, -n, n], [0, 0, 0]])
        before = cast.stats()
        for directions in (rays, np.empty((0, 3))):
            with pytest.raises(NotImplementedError, match="ellipsoid"):
                cast.query(model, poses, [0, 0, 2], directions, world=1)
        with pytest.raises(ValueError, match="pose"):
            cast.query(model, {}, origins, rays, world=3)
        assert cast.stats() == before
    with pytest.raises(RuntimeError, match="closed"):
        cast.query(model, poses, origins, rays, world=3)


def test_unrepresentable_size_and_bounded_pair_work_fail_before_launch(tyx_mode, tmp_path):
    s = ph.CollisionShape(ph.CollisionKey(0, 3, 1), "sphere", (0,), 1e-90)
    model, poses = scene((s, [[1, 0, 0]]))
    with operator(tyx_mode, tmp_path) as cast:
        with pytest.raises(ValueError, match="FP32"):
            cast.query(model, poses, [0, 0, 0], [[1, 0, 0]])
        assert cast.stats()["completed_launches"] == 0
        normal = ph.CollisionShape(s.key, "sphere", (0,), 0.25)
        baseline, valid = scene((normal, [[1, 0, 0]]))
        with pytest.raises(ValueError, match="capacity"):
            with operator(tyx_mode, tmp_path, max_ray_patch_tests=2) as limited:
                limited.query(baseline, valid, [0, 0, 0], [[1, 0, 0]] * 3)
        assert cast.query(baseline, valid, [0, 0, 0], [[1, 0, 0]]).distance_m.item() == 0.75


def test_rotated_independent_origins_over_one_block_match_analytic_ranges_and_normals(tyx_mode, tmp_path):
    count = 269
    center = np.array([2.25, -1.0, 1.5])
    rotate = np.array([[0.8, 0, 0.6], [0, 1, 0], [-0.6, 0, 0.8]])
    i = np.arange(count)
    x = np.where(i < 243, ((17 * i) % 61 - 30) / 32, 1.25)
    y = ((7 * i) % 15 - 7) / 64
    z = 2 + (i % 11) / 16
    origins = np.column_stack((x, y, z)) @ rotate.T + center
    rays = -(1 + i % 21)[:, None] * rotate[:, 2]
    visible = i < 243
    radial = np.sqrt(1 - x[visible] ** 2 - y[visible] ** 2)
    target = np.full(count, np.inf)
    target[visible] = z[visible] - radial
    normals = np.zeros((count, 3))
    normals[visible] = np.column_stack((x[visible], y[visible], radial)) @ rotate.T
    ball = ph.CollisionShape(ph.CollisionKey(0, 8, 6), "sphere", (9,), 1)
    model, poses = scene((ball, center[None, :]))
    with operator(tyx_mode, tmp_path) as cast:
        hit = cast.query(model, poses, origins, rays)
        verify(hit, target, np.where(visible, 0, -1), normals)
        print("rays", count, "hits", int(visible.sum()),
              "max_range_error_m", np.max(abs(hit.distance_m[visible] - target[visible])),
              "max_normal_component_error", np.max(abs(hit.world_normals - normals)), cast.stats())


def test_failure_propagates_and_following_observation_uses_current_inputs(tyx_mode, tmp_path, monkeypatch):
    from phymeo.backends.tyx_math import TyxMath

    plane = ph.CollisionShape(ph.CollisionKey(0, 6, 3), "plane", (7,), normal=(0, 0, 1e290))
    model, poses = scene((plane, [[0, 0, -2]]))
    with operator(tyx_mode, tmp_path) as cast:
        def fail(*_args, **_kwargs):
            raise RuntimeError("injected ray execution failure")

        with monkeypatch.context() as patch:
            patch.setattr(TyxMath, "call", fail)
            with pytest.raises(RuntimeError, match="injected ray execution failure"):
                cast.query(model, poses, [0, 0, 0], [[0, 0, -1]])
        assert cast.stats()["completed_launches"] == 0
        poses[plane.key] = ph.CollisionMotion([[0, 0, -3]])
        hit = cast.query(model, poses, [0, 0, 0], [[0, 0, -1]])
        verify(hit, [3], [0], [[0, 0, 1]])


def test_closed_mesh_keeps_sloping_finite_face_and_two_sided_exit(tyx_mode, tmp_path):
    key = ph.CollisionKey(0, 3, 9)
    mesh = ph.CollisionShape(key, "mesh", (9, 2, 5, 17),
                             triangles=((9, 5, 2), (9, 2, 17), (9, 17, 5), (2, 5, 17)))
    model, poses = scene((mesh, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    origins = [[0.25, 0.25, 2], [0.75, 0.75, 2], [0.25, 0.25, 0.125], [0.25, 0.25, 0]]
    with operator(tyx_mode, tmp_path) as cast:
        hit = cast.query(model, poses, origins, [[0, 0, -1]] * 4)
        n = 1 / np.sqrt(3)
        verify(hit, [1.5, np.inf, 0.125, 0], [0, -1, 0, 0],
               [[n, n, n], [0, 0, 0], [0, 0, 1], [0, 0, 1]])
