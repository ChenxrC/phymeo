import math
from dataclasses import replace

import numpy as np
import pytest

import phymeo as ph

H = 1 / 128


def api():
    from phymeo.backends.tyx_sphere_island import SphereIslandWorld, TyxSphereIslandBatch

    return SphereIslandWorld, TyxSphereIslandBatch


def ball(m, x, v=(0, 0, 0), g=(0, 0, 0), r=0.25):
    return ph.FreeSphere(m, r, x, v, g)


def test_unequal_masses_oblique_impact_uses_two_material_surfaces(tyx_mode, tmp_path):
    World, Batch = api()
    world = World(
        (
            (81, ball(3, (0.375, 0.5, 0), r=0.3125)),
            (4, ball(1, (0, 0, 0), (1.2, 1.6, 0), r=0.3125)),
        ),
        restitution=0.5,
    )
    # The independent unit direction is (3,4,0)/5. COM speed is 0.5 m/s.
    with Batch([(17, world)], tyx_mode, cache_root=tmp_path) as batch:
        result = batch.step(H)[17]
        assert result.accepted, result
        state = result.state
        assert state.tick == 1 and tuple(state.bodies) == (4, 81) and len(result.contacts) == 1
        c = result.contacts[0]
        assert (c.first, c.second) == (4, 81)
        n = np.array([0.6, 0.8, 0])
        np.testing.assert_allclose(c.normal, n, atol=2e-6)
        assert c.normal_impulse_n_s == pytest.approx(2.25, abs=8e-6)
        np.testing.assert_allclose(state.bodies[4].velocity, -0.25 * n, atol=5e-6)
        np.testing.assert_allclose(state.bodies[81].velocity, 0.75 * n, atol=5e-6)
        np.testing.assert_allclose(
            c.point_first, np.asarray(state.bodies[4].position) + 0.3125 * n, atol=2e-6
        )
        np.testing.assert_allclose(
            c.point_second, np.asarray(state.bodies[81].position) - 0.3125 * n, atol=2e-6
        )
        np.testing.assert_allclose(c.point_first, c.point_second, atol=2e-6)
        np.testing.assert_allclose(
            np.asarray(state.bodies[4].velocity) + 3 * np.asarray(state.bodies[81].velocity),
            2 * n,
            atol=8e-6,
        )
        assert result.contact_kinetic_change_j == pytest.approx(-1.125, abs=1e-5)
        assert result.momentum_residual_n_s < 1e-5
        shapes = [
            ph.CollisionShape(ph.CollisionKey(17, k, 0), "sphere", (0,), 0.3125) for k in (4, 81)
        ]
        model = ph.CollisionModel(shapes)
        motion = {
            shape.key: ph.CollisionMotion([state.bodies[shape.key.body].position])
            for shape in shapes
        }
        with ph.CollisionWorkspace(model, "host") as work:
            (pair,) = ph.detect(model, motion, ph.CollisionPolicy(margin=1e-4), work).pairs
        np.testing.assert_allclose(pair.witness.normal, c.normal, rtol=0, atol=2e-7)
        np.testing.assert_allclose(pair.witness.first_point, c.point_first, rtol=0, atol=2e-7)
        np.testing.assert_allclose(pair.witness.second_point, c.point_second, rtol=0, atol=2e-7)
        assert pair.witness.gap_m == pytest.approx(c.gap_m, abs=1e-8)
        assert batch.step(H)[17].accepted


def stack_world(world_type):
    return world_type(
        tuple(
            (local, ball(m, (0, 0, z), g=(0, 0, -8)))
            for local, m, z in ((8, 1, 0.25), (5, 2, 0.75), (2, 1, 1.25))
        ),
        plane_height=0,
    )


def test_three_balls_support_each_other_and_repeated_ids_stay_world_local(tyx_mode, tmp_path):
    World, Batch = api()
    stacked = stack_world(World)
    clear = World(
        (
            (5, ball(2, (0, 0, 3), (1, 0, 0), g=(0, 0, -8))),
            (8, ball(1, (7, 0, 3), (0, 1, 0), g=(0, 0, -8))),
        ),
        plane_height=0,
    )
    with Batch([(23, stacked), (4, clear)], tyx_mode, cache_root=tmp_path) as batch:
        outputs = []
        initial = batch.snapshot()
        for tick in range(1, 6):
            rows = batch.step(H)
            r = rows[23]
            assert r.accepted and rows[4].accepted, rows
            assert r.state.tick == tick and not rows[4].contacts
            assert (
                r.required_contacts == 3
                and r.position_residual_m <= 1e-5
                and r.velocity_residual_m_s <= 2e-5
            )
            impulse = {(c.first, c.second): c.normal_impulse_n_s / H for c in r.contacts}
            assert impulse == pytest.approx({(None, 8): 32, (5, 8): 24, (2, 5): 8}, abs=0.02)
            for local, z in ((8, 0.25), (5, 0.75), (2, 1.25)):
                assert r.state.bodies[local].position[2] == pytest.approx(z, abs=1e-5)
                assert abs(r.state.bodies[local].velocity[2]) < 3e-5
            outputs.append(r)
        assert batch.snapshot()[4].bodies[5].position[0] == pytest.approx(5 * H)
        kept = batch.snapshot()[4]
        batch.reset([23])
        assert batch.snapshot() == {23: initial[23], 4: kept}
        first = batch.step(H)[23]
        assert first == outputs[0]


def test_capacity_and_iterations_reject_without_truncating_a_dynamic_island(tyx_mode, tmp_path):
    World, Batch = api()
    spec = stack_world(World)
    with Batch([(9, spec), (6, spec)], tyx_mode, cache_root=tmp_path) as batch:
        old = batch.snapshot()
        rows = batch.step(H, contact_capacity={9: 2})
        assert (
            not rows[9].accepted
            and rows[9].reason == "contact_capacity"
            and rows[9].required_contacts == 3
        )
        assert rows[9].state == old[9] and not rows[9].contacts and rows[6].accepted
        good = rows[6]
        failed = batch.step(H, iterations=1)
        assert not failed[9].accepted and failed[9].reason in {
            "position_residual",
            "velocity_residual",
        }
        assert batch.snapshot()[9] == old[9]
        fixed = batch.step(H, contact_capacity={9: 3})[9]
        assert (
            fixed.accepted
            and fixed.contacts == tuple(replace(c, world=9) for c in good.contacts)
            and fixed.state.bodies == good.state.bodies
        )
        with pytest.raises(ValueError):
            batch.step(H, contact_capacity={9: -1})


def test_crossing_coincident_and_finite_force_controls(tyx_mode, tmp_path):
    World, Batch = api()
    crossing = World(((1, ball(1, (-1, 0, 0), (4, 0, 0))), (2, ball(1, (1, 0, 0), (-4, 0, 0)))))
    coincident = World(((1, ball(1, (0, 0, 0))), (2, ball(1, (0, 0, 0)))))
    clear = World(((1, ball(2, (0, 0, 4))),), plane_height=0)
    with Batch(
        [(33, crossing), (5, coincident), (9, clear)], tyx_mode, cache_root=tmp_path
    ) as batch:
        initial = batch.snapshot()
        rows = batch.step(0.5, forces={9: {1: (2, 0, 0)}})
        assert rows[33].reason == "intermediate_crossing" and rows[5].reason == "undefined_normal"
        assert rows[33].required_contacts is None
        assert (rows[33].issues[0].first, rows[33].issues[0].second) == (1, 2)
        assert (
            rows[33].issues[0].closest_free_fraction == 0.5
            and rows[33].issues[0].closest_free_gap_m == -0.5
        )
        assert rows[33].state == initial[33] and rows[5].state == initial[5]
        assert rows[9].accepted and rows[9].state.bodies[1].velocity == (0.5, 0, 0)
        old = batch.snapshot()
        with pytest.raises(ValueError):
            batch.step(H, forces={9: {1: (math.nan, 0, 0)}})
        assert batch.snapshot() == old
        assert batch.stats()["completed_kernels"] > 0


def test_invalid_or_oversized_topology_is_rejected_before_allocation(tmp_path):
    World, Batch = api()
    body = ball(1, (0, 0, 0))
    with pytest.raises(ValueError, match="duplicate"):
        World(((3, body), (3, body)))
    with pytest.raises(ValueError, match="8 spheres"):
        World(tuple((i, body) for i in range(9)))
    with pytest.raises(ValueError):
        World(((0, body),), restitution=1.001)
    with pytest.raises(ValueError, match="duplicate"):
        Batch([(4, World(((0, body),))), (4, World(((0, body),)))], "cuda", cache_root=tmp_path)


def test_newly_engaged_elastic_contact_may_not_create_net_kinetic_energy(tyx_mode, tmp_path):
    World, Batch = api()
    # AB closes at 2 m/s while BC is opening at 0.5 m/s. Solving the two
    # row targets requires (-0.5,1.5,1.5); this adds 0.25 J to equal masses.
    w = World(
        tuple((i, ball(1, (i / 2, 0, 0), (speed, 0, 0))) for i, speed in enumerate((2, 0, 0.5))),
        restitution=1,
    )
    with Batch([(47, w)], tyx_mode, cache_root=tmp_path) as batch:
        old = batch.snapshot()[47]
        out = batch.step(H)[47]
        assert not out.accepted and out.reason == "contact_energy_gain"
        assert out.contact_kinetic_change_j == pytest.approx(0.25, abs=1e-5)
        assert out.required_contacts == 2 and batch.snapshot()[47] == old


def test_late_device_failure_retries_with_original_world_checkpoints(
    tyx_mode, tmp_path, monkeypatch
):
    World, Batch = api()
    w = stack_world(World)
    rev = World(tuple(reversed(w.bodies)), plane_height=0)
    with (
        Batch([(4, w), (1, w)], tyx_mode, cache_root=tmp_path) as batch,
        Batch([(4, rev), (1, rev)], tyx_mode, cache_root=tmp_path) as oracle,
    ):
        before = batch.step(H)
        assert before == oracle.step(H)
        saved = batch.snapshot()
        engine = batch._engine
        read = engine._get
        count = 0

        def fail_after_positions(buffer):
            nonlocal count
            count += 1
            if count == 3:
                raise RuntimeError("injected status read failure")
            return read(buffer)

        with monkeypatch.context() as patch:
            patch.setattr(engine, "_get", fail_after_positions)
            with pytest.raises(RuntimeError, match="status read"):
                batch.step(H)
        assert batch.snapshot() == saved and batch.last_result(4) is before[4]
        assert batch.step(H) == oracle.step(H)


@pytest.mark.parametrize("configuration", ["distant", "boosted", "dissipating"])
def test_passivity_is_local_to_the_colliding_component(tyx_mode, tmp_path, configuration):
    World, Batch = api()
    boost = 300.0 if configuration == "boosted" else 0.0
    coefficient = 15 / 16 if configuration == "dissipating" else 1
    third = 1.0 if configuration == "dissipating" else 0.5
    bodies = [
        (i, ball(1, (i / 2, 0, 0), (speed, boost, 0))) for i, speed in enumerate((2, 0, third))
    ]
    if configuration == "distant":
        bodies.append((8, ball(100, (20, 20, 0), (0, 10, 0))))
    elif configuration == "dissipating":
        bodies.extend([(7, ball(1, (0, 20, 0), (2, 0, 0))), (8, ball(1, (0.5, 20, 0), (-2, 0, 0)))])
    with Batch(
        [(47, World(tuple(bodies), restitution=coefficient))], tyx_mode, cache_root=tmp_path
    ) as batch:
        saved = batch.snapshot()
        row = batch.step(H)[47]
        assert not row.accepted and row.reason == "contact_energy_gain"
        assert batch.snapshot() == saved
        tri = next(group for group in row.energy_components if group.body_ids == (0, 1, 2))
        expected = 11 / 64 if configuration == "dissipating" else 0.25
        assert tri.normal_work_j == pytest.approx(expected, abs=2e-5)
        assert tri.relative_kinetic_change_j == pytest.approx(expected, abs=2e-5)
        if configuration == "dissipating":
            assert row.contact_kinetic_change_j < 0


def test_initial_closed_pair_that_crosses_and_separates_is_not_published_as_empty(
    tyx_mode, tmp_path
):
    World, Batch = api()
    shared = World(((1, ball(1, (0, 0, 0))), (2, ball(1, (0.5, 0, 0), (-1, 2, 0)))))
    neighbor = World(((1, ball(1, (0, 4, 0), (0, 0, 1))),))
    with Batch([(11, shared), (12, neighbor)], tyx_mode, cache_root=tmp_path) as batch:
        saved = batch.snapshot()
        hit = batch.step(0.25)
        assert not hit[11].accepted and hit[11].reason == "intermediate_crossing"
        assert hit[12].accepted and batch.snapshot()[12].tick == 1
        assert batch.snapshot()[11] == saved[11]
        (problem,) = hit[11].issues
        assert problem.initial_gap_m == 0 and problem.free_gap_m > 0
        assert problem.closest_free_gap_m == pytest.approx(np.sqrt(0.2) - 0.5, abs=1e-7)
