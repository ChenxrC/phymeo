"""Sensor-ready material geometry at a successful variable-duration contact step."""

import numpy as np
import pytest

import phymeo as ph
from phymeo.runtime.contact_scene import ContactSceneBuilder


def optical_case():
    b = ContactSceneBuilder(gravity=(0, 0, 0))
    q = (0, np.sqrt(0.5), 0, np.sqrt(0.5))
    b.add_box(
        4,
        [0.5, 0.25, 0.125],
        mass=1,
        position=(0, 0, -2),
        orientation_xyzw=q,
        velocity=(0.125, 0, -0.25),
        angular_velocity=(0, 0, 0.25),
        world=7,
    )
    nodes = np.array([[1, -0.5, -3], [2, -0.5, -3], [2, 0.5, -3], [1, 0.5, -3]], dtype=float)
    speeds = np.zeros_like(nodes)
    speeds[:, 2] = [0.5, 0, -0.5, 0.125]
    b.add_sheet(
        9,
        nodes,
        [(0, 1, 2), (0, 2, 3)],
        node_masses=(1, 2, 3, 4),
        velocities=speeds,
        edge_compliance=None,
        world=7,
    )
    b.pair(4, 9, ph.NormalLayer(1 / 32, 32), world=7)
    b.pair(9, 9, None, world=7)
    b.add_box(4, [0.5, 0.25, 0.125], mass=None, position=(0, 0, -5), world=42)
    return b, nodes, speeds


def check_optics(snapshot, operator=None, atol=1e-10):
    t = snapshot.time_s
    camera = ph.DepthCamera(1, 1, 0.8)
    for world, unshifted, origin, expected in (
        (7, 4, (0, 0, 0), 1.5 + 0.25 * t),
        (7, 9, (1.625, -0.25, 0), 3.0 - t / 16),
        (42, 4, (0, 0, 0), 4.875),
    ):
        image = camera.capture(
            snapshot.model, snapshot.poses, world=world, position_m=origin, operator=operator
        )
        assert image.keys[image.shape_index.item()] == ph.CollisionKey(world, unshifted, 0)
        assert image.ray_depth_m.item() == pytest.approx(expected, abs=atol, rel=0)
        assert image.forward_depth_m.item() == pytest.approx(expected, abs=atol, rel=0)
    ray = ph.raycast_scene(
        snapshot.model, snapshot.poses, [[1.625, -0.25, 0]], [[0, 0, -1]], world=7
    )
    assert ray.distance_m.item() == pytest.approx(3.0 - t / 16, abs=atol, rel=0)


def test_frozen_material_snapshot_stays_usable_after_steps_close_and_external_mutation():
    setup, x, v = optical_case()
    with setup.build() as scene:
        old = scene.geometry()
        assert old.time_s == 0 and scene.last_frame is None
        check_optics(old)
        x[:] = 400
        v[:] = -800
        sheet = ph.CollisionKey(7, 9, 0)
        tool = ph.CollisionKey(7, 4, 0)
        with pytest.raises(TypeError):
            old.poses[sheet] = old.poses[tool]
        with pytest.raises(ValueError):
            old.poses[sheet].start.setflags(write=True)
        with pytest.raises(ValueError):
            old.poses[tool].start[0, 0] = 20
        pose = scene.state
        pose.particles.position[:] = 300
        pose.rigid.position[:] = 200
        for h in (1 / 16, 1 / 8):
            frame = scene.step(h)
        end = scene.geometry()
        assert end.time_s == 3 / 16 and scene.last_frame.path is frame.path
        assert end.model is old.model
        check_optics(end)
        xyz = old.poses[sheet].start.copy()
        xyz[:, 2] += end.time_s * np.array([0.5, 0, -0.5])
        np.testing.assert_allclose(end.poses[sheet].start, xyz, atol=1e-13)
        angle = 0.25 * end.time_s
        dx, dy = -0.125, -0.25
        corner = [
            0.125 * end.time_s + np.cos(angle) * dx - np.sin(angle) * dy,
            np.sin(angle) * dx + np.cos(angle) * dy,
            -1.5 - 0.25 * end.time_s,
        ]
        np.testing.assert_allclose(end.poses[tool].start[0], corner, atol=1e-12)
        frame.state.rigid.position[:] = 600
        frame.state.particles.position[:] = 900
        check_optics(old)
        check_optics(scene.geometry())
    with pytest.raises(RuntimeError, match="closed"):
        scene.geometry()
    check_optics(end)


def test_fresh_observation_is_end_state_while_original_applied_force_remains_at_entry():
    b = ContactSceneBuilder(gravity=(0, 0, 0))
    b.add_box(4, [0.125] * 3, mass=None)
    b.add_box(7, [0.125] * 3, mass=2, position=(0, 0, 9 / 32))
    b.pair(4, 7, ph.NormalLayer(1 / 16, 128))
    lower, upper = ph.CollisionKey(0, 4, 0), ph.CollisionKey(0, 7, 0)
    with b.build() as scene:
        reading, entry = scene.observe_scene()
        assert entry.time_s == 0 and reading.loads.contacts[0].force_n == pytest.approx(4)
        frame = scene.step(1 / 32)
        reading, end = scene.observe_scene()
        assert end.time_s == 1 / 32 and scene.last_frame.path is frame.path
        gap = end.poses[upper].start[:, 2].min() - end.poses[lower].start[:, 2].max()
        assert gap == pytest.approx(1 / 32 + 2 / 32**2)
        assert reading.loads.contacts[0].force_n == pytest.approx(128 * (1 / 16 - gap))
        assert frame.loads.contacts[0].force_n == pytest.approx(4)
        current = scene.state
        with pytest.raises(ph.SolverFailure, match="capacity"):
            scene.step(1 / 64, capacity=0)
        with pytest.raises(ValueError, match="time"):
            scene.step(np.nextafter(0.0, 1.0))
        assert scene.last_frame.path is frame.path and scene.geometry().time_s == end.time_s
        np.testing.assert_array_equal(scene.state.rigid.position, current.rigid.position)
        scene.step(1 / 64)
        assert scene.geometry().time_s == 3 / 64


def test_actual_backend_snapshot_feeds_optics_and_uses_physical_dt(tyx_mode, tmp_path):
    setup, _, _ = optical_case()
    from phymeo.backends.tyx_raycast import TyxRaycaster

    with (
        setup.build(backend=tyx_mode, cache_root=tmp_path) as scene,
        TyxRaycaster(mode=tyx_mode, cache_root=tmp_path) as caster,
    ):
        dt = 0.1
        frame = scene.step(dt)
        reading, snapshot = scene.observe_scene()
        assert (
            snapshot.time_s == float(np.float32(dt))
            and frame.rigid_report.backend == "tyx:" + tyx_mode
        )
        assert (
            reading.geometry.status == "ok" and frame.particle_report.backend == "tyx:" + tyx_mode
        )
        check_optics(snapshot, caster, 2e-5)
        assert caster.stats()["completed_launches"] > 0
