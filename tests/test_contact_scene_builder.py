"""Ordinary model declarations feed the same finite contact sampler and physical step."""

from itertools import product

import numpy as np
import pytest

from phymeo.core.capabilities import SolverFailure
from phymeo.solvers.collision_loads import NormalLayer


def builder(gravity=(0.0, 0.0, 0.0)):
    from phymeo.runtime.contact_scene import ContactSceneBuilder

    return ContactSceneBuilder(gravity=gravity)


def stack(b, world):
    for body, z, mass in ((10, 0.0, None), (20, 19 / 64, 1.0), (30, 77 / 128, 1.0)):
        b.add_box(body, [1 / 8] * 3, mass=mass, position=(0, 0, z), world=world)
    law = NormalLayer(1 / 16, 128.0, 0.25)
    b.pair(10, 20, law, world=world)
    b.pair(20, 30, law, world=world)
    b.pair(10, 30, None, world=world)


def panel(b, *, flipped=False, world=3):
    # One apex above the interior; all three upper vertices are separated.
    vertices = (
        (1 / 16, -1 / 8, -1 / 8),
        (-1 / 8, -1 / 16, 1 / 8),
        (1 / 8, -1 / 16, 1 / 8),
        (0, 1 / 8, 1 / 8),
    )
    b.add_convex(
        7,
        vertices,
        mass=2.0,
        inertia_diagonal=(1 / 16,) * 3,
        position=(0, 0, 5 / 32),
        orientation_xyzw=(0, 0, 1, 0) if flipped else (0, 0, 0, 1),
        world=world,
    )
    x = [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]]
    b.add_sheet(
        13,
        x,
        [(0, 1, 2), (0, 2, 3)],
        node_masses=[1.0, 2.0, 3.0, 4.0],
        edge_compliance=1 / 256,
        world=world,
    )
    b.pair(7, 13, NormalLayer(1 / 16, 128.0, 0.0), world=world)
    b.pair(13, 13, None, world=world)


def test_declared_boxes_equilibrate_then_only_the_driven_world_unloads():
    b = builder((0.0, 0.0, -1.0))
    stack(b, 7)
    stack(b, 42)
    with b.build() as scene:
        entry = scene.sample()
        assert [row.force_n for row in entry.loads.contacts] == pytest.approx([2, 1, 2, 1])
        assert [len(c.sites) for c in entry.loads.contacts] == [4] * 4
        assert scene.stats()["particle_count"] == 0 and scene.stats()["rigid_count"] == 6
        first = scene.step(1 / 64)
        assert first.particle_report is None and first.rigid_report.accepted
        np.testing.assert_allclose(first.state.rigid.velocity, 0, atol=1e-14)
        external = scene.inputs()
        external.rigid.force[scene.rigid_rows[42, 30], 2] = 4.0
        for _ in range(12):
            f = scene.step(1 / 64, external)
        isolated = [scene.rigid_rows[7, n] for n in (10, 20, 30)]
        np.testing.assert_allclose(
            f.state.rigid.position[isolated], first.state.rigid.position[isolated], atol=2e-13
        )
        assert f.state.rigid.position[scene.rigid_rows[42, 30], 2] > 77 / 128 + 1 / 64
        assert all(c.first.world == c.second.world for c in f.loads.contacts)
        assert not any(c.second.world == 42 and c.second.body == 30 for c in f.loads.contacts)
        assert any(c.second.world == 7 and c.second.body == 30 for c in f.loads.contacts)


@pytest.mark.parametrize("flipped", [False, True])
def test_convex_on_connected_sheet_has_material_force_torque_and_free_nodes(flipped):
    b = builder()
    panel(b, flipped=flipped)
    with b.build() as scene:
        check_panel(scene, flipped, 1e-10)


def check_panel(scene, flipped, tol):
    sample = scene.sample()
    (c,) = sample.loads.contacts
    assert (c.first.body, c.second.body, c.second.shape) == (7, 13, int(flipped))
    assert c.gap_m == pytest.approx(1 / 32, abs=tol)
    assert c.force_n == pytest.approx(4, abs=tol)
    np.testing.assert_allclose(sample.loads.inputs.rigid.force[0], [0, 0, 4], atol=tol)
    sign = -1 if flipped else 1
    np.testing.assert_allclose(
        sample.loads.inputs.rigid.torque[0], [-0.5 * sign, -0.25 * sign, 0], atol=tol
    )
    weights = [7 / 16, 3 / 16, 6 / 16, 0] if not flipped else [6 / 16, 0, 7 / 16, 3 / 16]
    np.testing.assert_allclose(
        sample.loads.inputs.particles.force[:, 2], -4 * np.array(weights), atol=tol
    )
    assert scene.stats()["distance_edges"] == 5
    before = scene.state
    for _ in range(4):
        frame = scene.step(1 / 512)
    assert (
        frame.path.status == "ok" and frame.rigid_report.accepted and frame.particle_report.accepted
    )
    assert frame.state.rigid.position[0, 2] > before.rigid.position[0, 2]
    assert np.ptp(frame.state.particles.position[:, 2]) > 1e-6
    assert np.linalg.norm(frame.state.rigid.angular_velocity[0]) > 1e-3
    np.testing.assert_allclose(
        2 * frame.state.rigid.velocity[0] + np.array([1, 2, 3, 4]) @ frame.state.particles.velocity,
        0,
        atol=tol * 10,
    )
    np.testing.assert_array_equal(before.particles.velocity, 0)
    return frame


def test_registration_snapshot_rollback_and_repeated_builds_own_state():
    b = builder()
    box = np.array([1 / 8] * 3)
    b.add_box(4, box, mass=None)
    box[:] = 100
    b.add_box(5, [1 / 8] * 3, mass=2.0, position=(0, 0, 9 / 32))
    with pytest.raises(ValueError, match="material|pair"):
        b.build()
    b.pair(4, 5, NormalLayer(1 / 16, 128.0, 0.0))
    with b.build() as a, b.build() as other:
        good = a.step(1 / 512)
        expected = a.state
        assert other.last_frame is None
        with pytest.raises(SolverFailure, match="capacity"):
            a.step(1 / 512, capacity=0)
        assert a.last_frame is good
        np.testing.assert_array_equal(a.state.rigid.position, expected.rigid.position)
        state = a.state
        state.rigid.position[:] = 23
        good.state.rigid.position[:] = 40
        np.testing.assert_array_equal(a.state.rigid.position, expected.rigid.position)
        again = other.step(1 / 512)
        np.testing.assert_array_equal(again.state.rigid.position, expected.rigid.position)
        assert a.step(1 / 512).state.rigid.position[1, 2] > expected.rigid.position[1, 2]
        b.add_box(0, [0.2] * 3, mass=None, world=99)
        assert a.stats()["rigid_count"] == 2
        with b.build() as later:
            assert later.stats()["rigid_count"] == 3
    with pytest.raises(RuntimeError, match="closed"):
        a.sample()
    with b.build() as reopened:
        assert reopened.sample().loads.contacts


def test_invalid_geometry_mass_and_unregistered_ownership_are_explicit():
    b = builder()
    with pytest.raises(ValueError):
        b.add_box(2, [1.0, 0.0, 1.0], mass=1.0)
    with pytest.raises(ValueError):
        b.add_convex(
            2, list(product((-1.0, 1.0), repeat=3)), mass=1.0, inertia_diagonal=[1.0, 1.0, 3.0]
        )
    with pytest.raises(ValueError):
        b.add_sheet(9, [(0, 0, 0), (1, 0, 0), (2, 0, 0)], [(0, 1, 2)], node_masses=1.0)
    b.add_box(2, [1.0, 1.0, 1.0], mass=None)
    with pytest.raises(ValueError):
        b.add_box(2, [1.0, 1.0, 1.0], mass=None)
    with pytest.raises(ValueError):
        b.pair(2, 19, NormalLayer(0.1, 1.0, 0.0))


def test_real_backend_runs_registered_participants_and_finite_layer(tyx_mode, tmp_path):
    backend = tyx_mode
    b = builder()
    panel(b)
    with b.build(backend=backend, cache_root=tmp_path) as scene:
        f = check_panel(scene, False, 2e-5)
        assert f.rigid_report.backend == f.particle_report.backend == "tyx:" + backend
        stats = scene.stats()
        assert stats["geometry"]["completed_launches"] > 0
        assert stats["normal_layer"]["completed_launches"] > 0


def test_declarations_reject_unsupported_materials_and_precision_loss_before_stepping():
    b = builder()
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b.add_sheet(0, x, [[0, 1, 2]], node_masses=1.0)
    b.add_sheet(9, x + [0.0, 0.0, 0.1], [[0, 1, 2]], node_masses=1.0)
    b.pair(0, 9, NormalLayer(0.15, 20.0, 0.0, friction=0.3))
    with pytest.raises(ValueError, match="sliding layer"):
        b.build()
    huge = builder()
    huge.add_convex(
        1, list(product((-1.0, 1.0), repeat=3)), mass=1e100, inertia_diagonal=[1.0, 1.0, 1.0]
    )
    with pytest.raises(ValueError, match="precision"):
        huge.build(backend="interpret")
    thin = builder()
    thin.add_sheet(
        3, [[1.0, 0.0, 0.0], [1.0 + 1e-9, 0.0, 0.0], [1.0, 1.0, 0.0]], [[0, 1, 2]], node_masses=1.0
    )
    with pytest.raises(ValueError, match="precision"):
        thin.build(backend="interpret")


def test_true_swept_core_crossing_is_rejected_and_the_same_scene_can_continue():
    b = builder()
    b.add_box(1, [1 / 8] * 3, mass=None)
    b.add_box(2, [1 / 8] * 3, mass=1.0, position=(0, 0, 3 / 8))
    b.pair(1, 2, NormalLayer(1 / 16, 128.0))
    with b.build() as sc:
        first = sc.step(1 / 64)
        fast = sc.inputs()
        fast.rigid.force[sc.rigid_rows[0, 2], 2] = -4096
        with pytest.raises(SolverFailure, match="core"):
            sc.step(1 / 64, fast)
        assert sc.last_frame is first
        np.testing.assert_array_equal(sc.state.rigid.position, first.state.rigid.position)
        retried = sc.step(1 / 64)
        np.testing.assert_array_equal(retried.state.rigid.position, first.state.rigid.position)
