"""Metric pinhole observations of real registered shapes rather than viewer pixels."""

import numpy as np
import pytest

import phymeo as ph


def make_scene():
    ball = ph.CollisionKey(0, 0, 9)
    plane = ph.CollisionKey(0, 1, 2)
    other = ph.CollisionKey(1, 0, 9)
    shapes = [
        ph.CollisionShape(ball, "sphere", (3,), radius=0.5),
        ph.CollisionShape(plane, "plane", (4,), normal=(0, 0, 1)),
        ph.CollisionShape(other, "sphere", (3,), radius=0.5),
    ]
    poses = {
        ball: ph.CollisionMotion(np.array([[0.0, 0.0, -2.0]])),
        plane: ph.CollisionMotion(np.array([[0.0, 0.0, -4.0]])),
        other: ph.CollisionMotion(np.array([[0.0, 0.0, -1.0]])),
    }
    return ph.CollisionModel(shapes), poses, ball, plane, other


def test_depth_uses_camera_geometry_and_frame_world_not_local_ids():
    model, poses, ball, plane, other = make_scene()
    cam = ph.DepthCamera(3, 3, np.pi / 2)
    result = cam.capture(model, poses, world=0)
    assert result.keys[result.shape_index[1, 1]] == ball
    assert result.keys[result.shape_index[0, 0]] == plane
    assert result.ray_depth_m[1, 1] == 1.5
    np.testing.assert_allclose(result.forward_depth_m[0], [4] * 3)
    assert result.ray_depth_m[0, 0] == pytest.approx(4 * np.sqrt(1 + 2 * (2 / 3) ** 2))
    np.testing.assert_allclose(result.world_normals[1, 1], [0, 0, 1])
    result1 = cam.capture(model, poses, world=1)
    assert result1.keys[result1.shape_index[1, 1]] == other and result1.ray_depth_m[1, 1] == 0.5
    assert np.isinf(result1.ray_depth_m[0, 0]) and result1.shape_index[0, 0] == -1
    np.testing.assert_array_equal(result1.world_normals[0, 0], np.zeros(3))
    with pytest.raises(ValueError):
        result1.ray_depth_m.setflags(write=True)
    assert not result1.ray_depth_m.flags.writeable


def test_finite_capsule_endcap_triangles_occlude_and_reveal():
    key = ph.CollisionKey(0, 1, 0)
    cap = ph.CollisionShape(key, "capsule", (40, 10), radius=0.25)
    k = ph.CollisionKey(0, 2, 8)
    triangle = ph.CollisionShape(k, "triangle", (0, 1, 2))
    model = ph.CollisionModel([cap, triangle])
    poses = {
        key: ph.CollisionMotion(np.array([[-1.0, 0.0, -2.0], [1.0, 0.0, -2.0]])),
        k: ph.CollisionMotion(np.array([[-2.0, -1.0, -3.0], [2.0, -1.0, -3.0], [0.0, 2.0, -3.0]])),
    }
    cam = ph.DepthCamera(1, 1, 1.0)
    assert cam.capture(model, poses, world=0).ray_depth_m.item() == 1.75
    assert cam.capture(
        model, poses, world=0, position_m=[1.15, 0.0, 0.0]
    ).ray_depth_m.item() == pytest.approx(1.8)
    poses[key] = ph.CollisionMotion(np.array([[3.0, 0.0, -2.0], [4.0, 0.0, -2.0]]))
    frame = cam.capture(model, poses, world=0)
    assert frame.ray_depth_m.item() == 3 and frame.keys[frame.shape_index.item()] == k


def test_camera_rotation_clipping_and_invalid_motion_have_explicit_behavior():
    model, poses, ball, _, _ = make_scene()
    k = np.sqrt(0.5)
    # Camera local -Z points toward world -X, looking horizontally past the sphere.
    cam = ph.DepthCamera(1, 1, 1.0)
    assert np.isinf(
        cam.capture(model, poses, world=0, orientation_xyzw=[0, k, 0, k]).ray_depth_m.item()
    )
    assert (
        ph.DepthCamera(1, 1, 1.0, near_m=2).capture(model, poses, world=0).ray_depth_m.item() == 2.5
    )
    assert np.isinf(
        ph.DepthCamera(1, 1, 1.0, far_m=1).capture(model, poses, world=0).ray_depth_m.item()
    )
    poses[ball] = ph.CollisionMotion(poses[ball].start, poses[ball].start + 1)
    with pytest.raises(ValueError, match="pose"):
        cam.capture(model, poses, world=0)
    for bad_limit in ("2.5", False, np.nan, -np.inf, 0):
        with pytest.raises(ValueError):
            ph.DepthCamera(1, 1, 1, far_m=bad_limit)
    for dims in ((True, 1), (0, 2), (10**8, 2)):
        with pytest.raises(ValueError):
            ph.DepthCamera(*dims, 1.0)


def test_hull_mesh_inside_and_tie_order_at_sampled_pose():
    from itertools import product

    verts = np.array(list(product((-0.5, 0.5), (-0.4, 0.4), (-0.25, 0.25))))
    key = ph.CollisionKey(0, 9, 2)
    hull = ph.CollisionShape(key, "hull", tuple(range(8)))
    model = ph.CollisionModel((hull,))
    cam = ph.DepthCamera(1, 1, 0.8)
    pose = ph.CollisionMotion(verts + [0, 0, -2])
    assert cam.capture(model, {key: pose}).ray_depth_m.item() == 1.75
    assert cam.capture(model, {key: pose}, position_m=(0, 0, -2)).ray_depth_m.item() == 0.25
    closed = ph.CollisionKey(0, 10, 1)
    points = np.array([[0.0, 0.0, -1], [1.0, 0.0, -2], [-1.0, 0.0, -2], [0.0, 1.0, -2]])
    mesh = ph.CollisionShape(
        closed, "mesh", (0, 1, 2, 3), triangles=((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    )
    hit = cam.capture(ph.CollisionModel((mesh,)), {closed: ph.CollisionMotion(points)})
    assert hit.ray_depth_m.item() == pytest.approx(1)
    front = ph.CollisionKey(0, 8, 0)
    duplicate = ph.CollisionShape(front, "hull", tuple(range(8)))
    tied = cam.capture(ph.CollisionModel((hull, duplicate)), {key: pose, front: pose})
    assert tied.keys[tied.shape_index.item()] == front


def test_backproject_metric_depth_preserves_the_rotated_world_hit_and_missing_pixels():
    model, poses, ball, plane, _ = make_scene()
    camera = ph.DepthCamera(5, 3, 0.9)
    eye = (0.04, -0.07, 0.3)
    turn = (0.0, np.sin(0.18), 0.0, np.cos(0.18))
    frame = camera.capture(model, poses, position_m=eye, orientation_xyzw=turn)
    xyz = camera.unproject(frame.ray_depth_m, position_m=eye, orientation_xyzw=turn)
    labels = {key: i for i, key in enumerate(frame.keys)}
    floor_hits = frame.shape_index == labels[plane]
    sphere_hits = frame.shape_index == labels[ball]
    assert floor_hits.any() and sphere_hits.any()
    np.testing.assert_allclose(xyz[floor_hits, 2], -4, atol=3e-14, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(xyz[sphere_hits] - poses[ball].start[0], axis=1), 0.5)
    empty = np.full((3, 5), np.inf)
    assert np.isnan(camera.unproject(empty)).all()
    for bad in (np.ones((5, 3)), np.full((3, 5), -1), np.full((3, 5), np.nan)):
        with pytest.raises(ValueError):
            camera.unproject(bad)


def test_explicit_camera_operator_receives_all_rays_and_never_hides_query_failure():
    class OnlyOnce:
        def __init__(self):
            self.calls = 0

        def query(self, model, poses, origins, directions, **kwargs):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("device unavailable")
            assert directions.shape == (9, 3) and origins.shape == (3,)
            return ph.raycast_scene(model, poses, origins, directions, **kwargs)

    model, poses, *_ = make_scene()
    camera, op = ph.DepthCamera(3, 3, 0.8), OnlyOnce()
    frame = camera.capture(model, poses, operator=op)
    np.testing.assert_allclose(frame.ray_depth_m, camera.capture(model, poses).ray_depth_m)
    with pytest.raises(RuntimeError, match="unavailable"):
        camera.capture(model, poses, operator=op)
    assert op.calls == 2
