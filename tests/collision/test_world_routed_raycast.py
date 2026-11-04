"""Independent worlds may overlap geometrically; beam ownership is never spatial."""

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends.tyx_raycast import TyxRaycaster


def scene():
    def k(w):
        return ph.CollisionKey(w, 1, 0)

    plate = ph.CollisionShape(k(1), "plate", (1, 2, 3), radius=0.1)
    large = ph.CollisionShape(k(3), "sphere", (9,), radius=1)
    small = ph.CollisionShape(k(7), "sphere", (9,), radius=0.5)
    rod = ph.CollisionShape(
        k(11), "ellipsoid", (9,), curved=ph.CurvedSolid("ellipsoid", (1, 0.5, 0.25))
    )
    # Rotate 180 degrees about (1,0,1)/sqrt(2) so the long local X axis becomes world Z.
    rot = (np.sqrt(0.5), 0, np.sqrt(0.5), 0)
    return ph.CollisionModel([rod, small, large, plate]), {
        large.key: ph.CollisionMotion([[0, 0, -3]]),
        small.key: ph.CollisionMotion([[0.3, 0, -2]]),
        rod.key: ph.CollisionMotion([[0, 0, -5]], body_orientation_xyzw=rot),
    }


def test_interleaved_matrix_routes_and_retains_full_global_keys(tyx_mode, tmp_path):
    import tyx as tx

    model, poses = scene()
    routed = np.array([[7, 3, 27], [11, 3, 7]], dtype=np.uint64)
    rays = np.zeros((2, 3, 3))
    rays[..., 2] = -np.arange(1, 7).reshape(2, 3)
    origins = np.zeros_like(rays)
    origins[1, :, 2] = [1, -3, 1]
    target = [[1.6, 2, np.inf], [5, 1, 2.6]]
    normals = [[[-0.6, 0, 0.8], [0, 0, 1], [0, 0, 0]], [[0, 0, 1], [0, 0, -1], [-0.6, 0, 0.8]]]
    ids = [[2, 1, -1], [3, 1, 2]]
    with TyxRaycaster(tyx_mode, cache_root=tmp_path) as op:
        before = tx.runtime_stats()
        hit = op.query(model, poses, origins, rays, world=routed)
        raw = tx.runtime_stats() - before
        assert raw.cuda_kernel_launches == int(tyx_mode == "cuda")
        assert op.stats()["completed_launches"] == 1
        host = ph.raycast_scene(model, poses, origins, rays, world=routed)
        for out in (hit, host):
            assert out.keys == tuple(s.key for s in model.shapes)
            np.testing.assert_array_equal(out.shape_index, ids)
            np.testing.assert_allclose(out.distance_m, target, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(out.world_normals, normals, rtol=0, atol=2e-6)
            with pytest.raises(ValueError):
                out.shape_index.setflags(write=True)
        flat = op.query(
            model, poses, origins.reshape(-1, 3), rays.reshape(-1, 3), world=routed.ravel()
        )
        np.testing.assert_array_equal(flat.distance_m, hit.distance_m.ravel())
        np.testing.assert_array_equal(flat.world_normals, hit.world_normals.reshape(-1, 3))
        # Compare every row with its original scalar-world call, preserving input order.
        for w in (7, 3, 27, 11):
            mask = routed == w
            ref = op.query(model, poses, origins[mask], rays[mask], world=w)
            np.testing.assert_array_equal(ref.shape_index, hit.shape_index[mask])
            np.testing.assert_array_equal(ref.distance_m, hit.distance_m[mask])
            np.testing.assert_array_equal(ref.world_normals, hit.world_normals[mask])
        print("matrix world ownership", hit.shape_index.tolist(), raw)


def test_independent_metric_scales_and_clips_survive_one_launch(tyx_mode, tmp_path):
    shapes, poses = [], {}
    for key, center, radius in [(3, 2**35, 0.5), (7, 0, 2**-35)]:
        s = ph.CollisionShape(ph.CollisionKey(key, 8, 1), "sphere", (2,), radius)
        shapes.append(s)
        poses[s.key] = ph.CollisionMotion([[center, 0, 0]])
    model = ph.CollisionModel(shapes)
    origins = [[2**35 + 2, 0, 0], [3 * 2**-35, 0, 0], [2**35, 0, 0], [0, 0, 0]]
    rays = [[-4, 0, 0]] * 4
    with TyxRaycaster(tyx_mode, cache_root=tmp_path) as op:
        for fn in (ph.raycast_scene, op.query):
            data = fn(model, poses, origins, rays, world=[3, 7, 3, 7], near_m=2**-35, far_m=1.5)
            np.testing.assert_array_equal(data.distance_m, [1.5, 2 * 2**-35, 0.5, 2**-35])
            np.testing.assert_array_equal(data.shape_index, [0, 1, 0, 1])
            np.testing.assert_array_equal(data.world_normals[:, 0], [1, 1, -1, -1])


def test_zero_beams_missing_world_and_unselected_unsupported_shapes(tyx_mode, tmp_path):
    model, poses = scene()
    with TyxRaycaster(tyx_mode, cache_root=tmp_path) as op:
        for fn in (ph.raycast_scene, op.query):
            hit = fn(model, {}, [0, 0, 0], np.empty((2, 0, 3)), world=np.empty((2, 0), np.int32))
            assert hit.distance_m.shape == (2, 0) and hit.world_normals.shape == (2, 0, 3)
            clear = fn(model, {}, [0, 0, 0], [[0, 0, -1]], world=[999])
            assert clear.shape_index.item() == -1 and np.isposinf(clear.distance_m.item())
            bad = dict(poses)
            bad[model.shapes[0].key] = ph.CollisionMotion([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
            with pytest.raises(ValueError, match="plate"):
                fn(model, bad, [0, 0, 0], [[0, 0, -1]] * 2, world=[7, 1])
        assert op.stats()["completed_launches"] == 0


def test_world_address_shape_type_and_broadcast_fail_before_launch(tyx_mode, tmp_path):
    model, poses = scene()
    with TyxRaycaster(tyx_mode, cache_root=tmp_path) as op:
        for fn in (ph.raycast_scene, op.query):
            dirs = np.full((2, 2, 3), 1.0)
            for invalid in (
                [[3, True], [7, 3]],
                [[3, -1], [7, 3]],
                [[3.0, 7], [7, 3]],
                [3, 7],
                [[3], [7]],
                np.ones((2, 2, 1), np.int64),
                True,
            ):
                with pytest.raises(ValueError, match="world"):
                    fn(model, poses, [0, 0, 0], dirs, world=invalid)
            with pytest.raises(ValueError):
                fn(model, poses, np.zeros((2, 1, 3)), dirs, world=[[3, 3], [7, 7]])
        assert op.stats()["completed_launches"] == 0


def test_capacity_counts_only_local_pairs_and_retry_keeps_previous_reading(tyx_mode, tmp_path):
    model, poses = scene()
    with TyxRaycaster(tyx_mode, max_ray_patch_tests=4, cache_root=tmp_path) as op:

        def read(ids):
            return op.query(model, poses, [0, 0, 0], [[0, 0, -1]] * len(ids), world=ids)

        good = read([3, 7, 3, 7, 99])
        # 4 real tests, not 5×2. Another routed ray in an empty world costs zero surfaces.
        np.testing.assert_array_equal(good.shape_index, [1, 2, 1, 2, -1])
        before = op.stats()
        with pytest.raises(ValueError, match="capacity"):
            read([3, 7, 3, 7, 11])
        assert op.stats() == before
        again = read([7, 3, 7, 3, 99])
        np.testing.assert_array_equal(again.shape_index, [2, 1, 2, 1, -1])
        np.testing.assert_array_equal(good.shape_index, [1, 2, 1, 2, -1])


def test_32_overlapping_worlds_compare_every_beam_with_32_scalar_queries(tyx_mode, tmp_path):
    import tyx as tx

    worlds = np.arange(32) * 3 + 4
    shapes, poses = [], {}
    for i, w in enumerate(worlds):
        ball = ph.CollisionShape(ph.CollisionKey(w, 1, 0), "sphere", (3,), radius=0.3 + i / 512)
        floor = ph.CollisionShape(ph.CollisionKey(w, 2, 0), "plane", (3,), normal=(0, 0, 1))
        shapes.extend((ball, floor))
        poses[ball.key] = ph.CollisionMotion([[(i % 3 - 1) / 32, 0, -1.5 - i / 32]])
        poses[floor.key] = ph.CollisionMotion([[0, 0, -3 - i / 32]])
    model = ph.CollisionModel(tuple(reversed(shapes)))
    addresses = np.tile(worlds, 25)
    x, y = np.meshgrid(np.linspace(-0.25, 0.25, 5), np.linspace(-0.25, 0.25, 5))
    rays = np.repeat(np.column_stack((x.ravel(), y.ravel(), -np.ones(25))), 32, axis=0)
    with TyxRaycaster(tyx_mode, cache_root=tmp_path) as op:
        start = tx.runtime_stats()
        combined = op.query(model, poses, [0, 0, 0], rays, world=addresses)
        united = tx.runtime_stats() - start
        start = tx.runtime_stats()
        for w in worlds:
            mask = addresses == w
            direct = op.query(model, poses, [0, 0, 0], rays[mask], world=w)
            assert {direct.keys[i].world for i in direct.shape_index} == {w}
            assert {direct.keys[i].body for i in direct.shape_index} == {1, 2}
            for attr in ("distance_m", "shape_index", "world_normals"):
                np.testing.assert_array_equal(getattr(combined, attr)[mask], getattr(direct, attr))
        separate = tx.runtime_stats() - start
        assert united.cuda_kernel_launches == (1 if tyx_mode == "cuda" else 0)
        assert separate.cuda_kernel_launches == (32 if tyx_mode == "cuda" else 0)
        print("W32×25 equal outputs; batched", united, "scalar", separate)


def test_late_backend_failure_does_not_mix_routed_geometry_on_retry(
    tyx_mode, tmp_path, monkeypatch
):
    from phymeo.backends.tyx_math import TyxMath

    model, poses = scene()
    rays = [[0, 0, -1], [0, 0, -1]]
    with TyxRaycaster(tyx_mode, cache_root=tmp_path) as op:
        first = op.query(model, poses, [0, 0, 0], rays, world=[3, 7])
        actual = TyxMath.call

        def fail(self, *args, **kwargs):
            actual(self, *args, **kwargs)
            raise RuntimeError("late routed device read")

        with monkeypatch.context() as patch:
            patch.setattr(TyxMath, "call", fail)
            with pytest.raises(RuntimeError, match="late routed device read"):
                op.query(model, poses, [0, 0, 0], rays, world=[7, 3])
        again = op.query(model, poses, [0, 0, 0], rays, world=[7, 3])
        np.testing.assert_array_equal(again.distance_m, first.distance_m[::-1])
        np.testing.assert_array_equal(again.shape_index, [2, 1])
        np.testing.assert_array_equal(first.shape_index, [1, 2])
