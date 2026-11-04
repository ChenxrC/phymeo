"""Macrostep publication, sampled material times and independent sphere impact oracles."""

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends.tyx_sphere_island import SphereIslandWorld, TyxSphereIslandBatch


def sphere(x, v=(0, 0, 0), *, mass=1, g=(0, 0, 0)):
    return ph.FreeSphere(mass, 0.25, x, v, g)


def glancing():
    return SphereIslandWorld(((1, sphere((0, 0, 0))), (2, sphere((0.5, 0, 0), (-1, 2, 0)))))


def crossing(*, start=1.0, speed=4.0, offset=0.0):
    return SphereIslandWorld(((1, sphere((-start, 0, 0), (speed, 0, 0))),
                              (2, sphere((start, offset, 0), (-speed, 0, 0)))), restitution=1)


def loaded(row):
    return [(s, c) for s in row.stages for c in s.report.contacts if c.normal_impulse_n_s > 0]


def test_original_oblique_closed_pair_gets_an_initial_fixed_pose_impulse(tyx_mode, tmp_path):
    with TyxSphereIslandBatch([(11, glancing())], tyx_mode, cache_root=tmp_path) as batch:
        old = batch.snapshot()[11]
        assert batch.step(0.25)[11].reason == "intermediate_crossing"
        assert batch.snapshot()[11] is old
        out = batch.step_substepped(0.25, substeps=4)[11]
        assert out.accepted and out.state is batch.snapshot()[11] and out.state.tick == 1
        assert batch.last_result(11) is out and len(out.stages) == 5
        [(stage, c)] = loaded(out)
        assert stage.start_s == stage.end_s == 0 and stage.kind == "initial_impulse"
        assert [(s.start_s, s.end_s) for s in out.stages[1:]] == [(i / 16, (i + 1) / 16) for i in range(4)]
        for i in (1, 2):
            assert stage.report.state.bodies[i].position == old.bodies[i].position
        assert (c.first, c.second) == (1, 2)
        assert c.normal_impulse_n_s == pytest.approx(0.5)
        np.testing.assert_allclose(c.normal, [1, 0, 0], atol=1e-7)
        np.testing.assert_allclose([c.point_first, c.point_second], [[0.25, 0, 0]] * 2)
        assert stage.report.contact_kinetic_change_j == pytest.approx(-0.25, abs=1e-7)
        a, b = out.state.bodies.values()
        np.testing.assert_allclose([a.velocity, b.velocity], [[-0.5, 0, 0], [-0.5, 2, 0]], atol=2e-6)
        np.testing.assert_allclose([a.position, b.position], [[-0.125, 0, 0], [0.375, 0.5, 0]], atol=2e-6)
        print("glance", stage.start_s, c.normal_impulse_n_s, a, b)


@pytest.mark.parametrize("start,speed,toi", [(1.0, 4.0, 3 / 16), (0.875, 3.0, 5 / 24)])
def test_large_step_impacts_bound_true_entry_and_retain_material_points(tyx_mode, tmp_path, start, speed, toi):
    with TyxSphereIslandBatch([(5, crossing(start=start, speed=speed))], tyx_mode, cache_root=tmp_path) as batch:
        prior = batch.snapshot()[5]
        insufficient = batch.step_substepped(0.5, substeps=1)[5]
        assert not insufficient.accepted and insufficient.reason == "intermediate_crossing"
        assert insufficient.state is prior and insufficient.stages == () and batch.last_result(5) is None
        assert insufficient.failure.report.issues[0].closest_free_gap_m == pytest.approx(-0.5)
        accepted = batch.step_substepped(0.5, substeps=8)[5]
        [(s, c)] = loaded(accepted)
        assert accepted.accepted and accepted.state.tick == 1 and accepted.failure is None
        assert s.kind == "advance" and s.start_s <= toi <= s.end_s and s.end_s - s.start_s == 1 / 16
        assert c.normal_impulse_n_s == pytest.approx(2 * speed, abs=1e-6)
        assert c.normal == (1, 0, 0)
        assert c.point_first == c.point_second == (0, 0, 0)
        a, b = accepted.state.bodies.values()
        assert a.velocity == (-speed, 0, 0) and b.velocity == (speed, 0, 0)
        ideal = 0.25 + (0.5 - toi) * speed
        assert 0 <= ideal - b.position[0] <= speed / 16 + 1e-7
        assert a.position[0] == -b.position[0]
        print("impact", toi, (s.start_s, s.end_s), c.normal_impulse_n_s, b.position[0], ideal)


def test_disjoint_paths_external_force_support_and_repeated_ids(tyx_mode, tmp_path):
    clear = crossing(offset=0.5625)
    free = SphereIslandWorld(((1, sphere((0, 0, 4), mass=2, g=(0, 0, -8))),))
    support = SphereIslandWorld(((1, sphere((0, 0, 0.25), mass=2, g=(0, 0, -8))),), plane_height=0)
    with TyxSphereIslandBatch([(8, clear), (4, free), (3, support)], tyx_mode, cache_root=tmp_path) as batch:
        out = batch.step_substepped(0.5, substeps=8, forces={4: {1: (2, 0, 0)}})
        assert all(v.accepted and v.state.tick == 1 for v in out.values())
        assert not loaded(out[8]) and not loaded(out[4])
        assert out[8].state.bodies[1].position == (1, 0, 0)
        b = out[4].state.bodies[1]
        np.testing.assert_allclose(b.velocity, [0.5, 0, -4], atol=1e-6)
        # Semi-implicit integration of a constant acceleration across eight positive intervals.
        np.testing.assert_allclose(b.position, [9 / 64, 0, 4 - 9 / 8], atol=1e-6)
        assert out[3].state.bodies[1].position == (0, 0, 0.25)
        assert out[3].state.bodies[1].velocity == (0, 0, 0)
        terms = loaded(out[3])
        assert len(terms) == 8 and all(c.first is None for _, c in terms)
        assert sum(c.normal_impulse_n_s for _, c in terms) == pytest.approx(8)
        old = out[4].state
        batch.reset([8])
        assert batch.snapshot()[8].tick == 0 and batch.snapshot()[4] is old


def test_rejected_world_drops_prior_stages_and_retries_with_no_contamination(tyx_mode, tmp_path):
    broken = SphereIslandWorld(((1, sphere((0, 0, 0))), (2, sphere((0, 0, 0)))))
    specs = [(1, crossing()), (2, broken), (3, crossing(offset=0.5625))]
    with TyxSphereIslandBatch(specs, tyx_mode, cache_root=tmp_path) as batch:
        before = batch.snapshot()
        out = batch.step_substepped(0.5, substeps=8, contact_capacity={1: 0})
        assert not out[1].accepted and out[1].reason == "contact_capacity"
        assert out[1].failure.start_s == 2 / 16 and out[1].completed_advances == 2
        assert out[1].stages == () and out[1].state is before[1] and batch.last_result(1) is None
        assert out[2].reason == "undefined_normal" and out[2].state is before[2]
        assert out[3].accepted and out[3].state.tick == 1
        neighbor = batch.last_result(3)
        with pytest.raises(ValueError, match="unselected"):
            batch.step_substepped(0.5, keys=[1], forces={3: {1: (1, 0, 0)}})
        retry = batch.step_substepped(0.5, substeps=8, keys=[1])
        assert tuple(retry) == (1,) and retry[1].accepted and retry[1].state.tick == 1
        assert batch.snapshot()[2] is before[2]
        assert batch.snapshot()[3] is neighbor.state and batch.last_result(3) is neighbor
        counts = batch.stats()
        assert not batch.step_substepped(0.5, keys=[]) and batch.stats() == counts


def test_late_driver_failure_has_no_visible_partial_macrostep(tyx_mode, tmp_path, monkeypatch):
    spec = [(1, crossing()), (3, glancing())]
    with TyxSphereIslandBatch(spec, tyx_mode, cache_root=tmp_path) as batch, TyxSphereIslandBatch(
        spec, tyx_mode, cache_root=tmp_path) as oracle:
        previous = batch.step_substepped(0.5, substeps=8)
        assert previous == oracle.step_substepped(0.5, substeps=8)
        saved = batch.snapshot()
        original = batch._engine.step
        calls = 0

        def late(*args):
            nonlocal calls
            result = original(*args)
            calls += 1
            assert batch.snapshot() == saved and batch.last_result(1) is previous[1]
            if calls == 3:
                raise RuntimeError("late device response")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(batch._engine, "step", late)
            with pytest.raises(RuntimeError, match="late device response"):
                batch.step_substepped(0.5, substeps=4)
        assert calls == 3 and batch.snapshot() == saved and batch.last_result(3) is previous[3]
        assert batch.step_substepped(0.5, substeps=4) == oracle.step_substepped(0.5, substeps=4)


def test_energy_gate_and_resource_limits_remain_effective(tyx_mode, tmp_path):
    bodies = tuple((i, sphere((i / 2, 0, 0), (v, 0, 0))) for i, v in enumerate((2, 0, 0.5)))
    with TyxSphereIslandBatch([(7, SphereIslandWorld(bodies, restitution=1))], tyx_mode, cache_root=tmp_path) as batch:
        saved = batch.snapshot()
        for budget in (0, 3, 65, True):
            with pytest.raises(ValueError, match="substeps"):
                batch.step_substepped(0.5, substeps=budget)
        assert batch.stats()["completed_kernels"] == 0
        out = batch.step_substepped(0.125, substeps=4)[7]
        assert out.reason == "contact_energy_gain" and out.state is saved[7]
        assert out.failure.kind == "initial_impulse" and out.stages == ()
        assert out.failure.report.energy_components[0].normal_work_j == pytest.approx(0.25, abs=1e-5)


def test_initial_impulse_adds_no_extra_duration_or_external_impulse(tyx_mode, tmp_path):
    base = glancing()
    g = (0, 0, -4)
    spec = SphereIslandWorld(tuple((i, sphere(s.position, s.velocity, g=g)) for i, s in base.bodies))
    with TyxSphereIslandBatch([(5, spec)], tyx_mode, cache_root=tmp_path) as batch:
        out = batch.step_substepped(0.25, substeps=4, forces={5: {1: (0, 0, 8), 2: (0, 0, 8)}})[5]
        assert out.accepted and sum(s.end_s - s.start_s for s in out.stages) == 0.25
        first = out.stages[0]
        assert first.kind == "initial_impulse" and first.start_s == first.end_s == 0
        assert all(b.velocity[2] == 0 for b in first.report.state.bodies.values())
        for b in out.state.bodies.values():
            # For each 1 kg sphere, integral of (8 N − 4 N) dt = 1 N s.
            assert b.velocity[2] == 1
            assert b.position[2] == 5 / 32
        assert batch.last_result(5) is out
        assert batch.step(1 / 64)[5].state.tick == 2


def test_unrepresentable_time_partition_rejects_before_device_allocation(tyx_mode, tmp_path):
    with TyxSphereIslandBatch([(11, glancing())], tyx_mode, cache_root=tmp_path) as batch:
        saved = batch.snapshot()
        with pytest.raises(ValueError, match="FP32"):
            batch.step_substepped(np.finfo(np.float32).tiny, substeps=64)
        assert batch.snapshot() == saved and batch.stats()["device_allocations"] == 0
