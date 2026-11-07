"""Accepted resident sphere poses feed genuine device broad phase, never CCD."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends.tyx_world_batch import FreeSphere, TyxWorldBatch


def _api():
    from phymeo.backends._resident_collision_plan import (
        ResidentSphereBinding,
        ResidentStaticBinding,
    )

    return ResidentSphereBinding, ResidentStaticBinding


def _box_points(low=(0, -0.25, -0.25), high=(0.5, 0.25, 0.25)):
    return np.array(
        [
            (x, y, z)
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=np.float64,
    )


def _one_model(world=7, *, radius=0.125, points=None, revision=0):
    ball = ph.CollisionKey(world, 3, 11)
    wall = ph.CollisionKey(world, 9, 4)
    model = ph.CollisionModel(
        (
            ph.CollisionShape(wall, "hull", tuple(range(8)), revision=revision),
            ph.CollisionShape(ball, "sphere", (37,), radius, revision=revision),
        )
    )
    static = ph.CollisionMotion(_box_points() if points is None else points)
    return model, ball, wall, static


def _bindings(ball, wall, static):
    sphere_binding, static_binding = _api()
    return (static_binding(wall, static), sphere_binding(ball, ball.world))


def _contains_sphere(bounds, center, radius):
    for axis in range(3):
        exact = Fraction(float(center[axis]))
        r = Fraction(float(radius))
        assert Fraction(float(bounds[axis])) <= exact - r
        assert Fraction(float(bounds[axis + 3])) >= exact + r


@pytest.mark.tyx
def test_c01_real_accepted_step_feeds_public_device_broad_phase(tyx_mode, tmp_path):
    model, ball, wall, static = _one_model()
    sphere = FreeSphere(2, 0.125, (-0.5, 0, 0), (1, 0, 0), (0, 0, 0))
    with TyxWorldBatch([(7, sphere)], tyx_mode, cache_root=tmp_path / "runtime") as batch:
        assert batch.step(0.25) == {7: True}
        # The real old runtime advances first: the frozen RED is this absent API,
        # not a fixture/import failure in the new plan implementation.
        prepare = batch.prepare_collision
        with prepare(model, _bindings(ball, wall, static)) as plan:
            first = plan.query()
            assert first.world_keys == (7,)
            assert first.ticks.tolist() == [1]
            assert first.shape_keys == (ball, wall)
            assert first.registered_pairs == ((ball, wall),)
            assert first.shape_revisions == (0, 0)
            assert first.candidates == ()
            _contains_sphere(first.bounds[0], (-0.25, 0, 0), 0.125)
            assert first.bounds[0, 3] < -0.124999
            assert plan.require_current(first) is first
            assert batch.step(0.125) == {7: True}
            touched = batch.collision(plan)
            assert touched.ticks.tolist() == [2]
            assert touched.state_epoch > first.state_epoch
            assert touched.candidates == ((ball, wall),)
            _contains_sphere(touched.bounds[0], (-0.125, 0, 0), 0.125)
            assert plan.last_result is touched
            with pytest.raises(ValueError, match="stale|epoch"):
                plan.require_current(first)


def _many_worlds(count):
    worlds, shapes, bindings = [], [], []
    sphere_binding, static_binding = _api()
    keys = tuple(11 + 17 * i for i in reversed(range(count)))
    for i, world in enumerate(keys):
        center = (-0.125 if i % 2 == 0 else -1.0, 0, 0)
        sphere = FreeSphere(2, 0.125, center, (0.25, 0, 0), (0, 0, 0))
        model, ball, wall, static = _one_model(world)
        worlds.append((world, sphere))
        shapes.extend(model.shapes)
        bindings.extend((sphere_binding(ball, world), static_binding(wall, static)))
    return worlds, ph.CollisionModel(tuple(reversed(shapes))), tuple(reversed(bindings))


@pytest.mark.tyx
@pytest.mark.parametrize("count", (1, 4, 32))
def test_c02_world_local_full_results_and_warm_direct_buffers(
    count, tyx_mode, tmp_path, monkeypatch
):
    worlds, model, bindings = _many_worlds(count)
    with TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "runtime") as batch:
        with batch.prepare_collision(model, bindings) as plan:
            # Exercise both true state buffers before the warm resource assertion.
            plan.query()
            assert all(batch.step(0.125).values())
            plan.query()
            before = plan.stats()
            batch_before = batch.stats()
            engine = plan._engine
            assert engine._runtime is batch._runtime

            def forbidden(*args, **kwargs):
                raise AssertionError(
                    "warm query must not snapshot state, upload envelopes or allocate"
                )

            monkeypatch.setattr(batch, "snapshot", forbidden)
            monkeypatch.setattr(batch, "kinematics", forbidden)
            monkeypatch.setattr(engine, "query", forbidden)
            if tyx_mode == "cuda":
                import tyx as tx

                monkeypatch.setattr(tx, "cuda_array", forbidden)
                monkeypatch.setattr(tx, "cuda_empty", forbidden)
                native = tx.runtime_stats()
            frozen = None
            for step in range(2, 6):
                assert all(batch.step(0.125).values())
                actual = plan.query()
                assert actual.world_keys == tuple(sorted(w for w, _ in worlds))
                assert actual.ticks.tolist() == [step] * count
                expected = set()
                for world, initial in worlds:
                    ball = ph.CollisionKey(world, 3, 11)
                    wall = ph.CollisionKey(world, 9, 4)
                    center = (initial.position[0] + step / 32, 0, 0)
                    row = actual.shape_keys.index(ball)
                    _contains_sphere(actual.bounds[row], center, 0.125)
                    if center[0] + 0.125 >= 0:
                        expected.add((ball, wall))
                assert set(actual.candidates) == expected
                assert len(actual.registered_pairs) == count
                assert all(a.world == b.world for a, b in actual.candidates)
                if frozen is None:
                    frozen = actual
                    old_bounds = actual.bounds.copy()
                np.testing.assert_array_equal(frozen.bounds, old_bounds)
            after = plan.stats()
            assert after["device_allocations"] == before["device_allocations"]
            assert after["dynamic_envelope_uploaded_bytes"] == 0
            assert after["bounds_kernel_launches"] - before["bounds_kernel_launches"] == 4
            assert after["copied_to_host_bytes"] > before["copied_to_host_bytes"]
            assert (
                batch.stats()["snapshot_downloaded_bytes"]
                == batch_before["snapshot_downloaded_bytes"]
            )
            assert (
                batch.stats()["kinematics_downloaded_bytes"]
                == batch_before["kinematics_downloaded_bytes"]
            )
            if tyx_mode == "cuda":
                delta = tx.runtime_stats() - native
                assert delta.cuda_kernel_launches >= 16  # advance, bounds, refit, candidates
                assert delta.cuda_d2h_bytes > 0
                assert after["downloaded_bytes"] > before["downloaded_bytes"]
            for array in (actual.bounds, actual.ticks):
                with pytest.raises(ValueError):
                    array.setflags(write=True)


@pytest.mark.tyx
@pytest.mark.parametrize("center,radius", ((0.1, 0.2), (-0.1, 0.2), (2**24, 16.0)))
def test_c03_outward_fp32_touch_and_resolved_near_miss(center, radius, tyx_mode, tmp_path):
    sphere = FreeSphere(2, radius, (center, 0, 0), gravity=(0, 0, 0))
    edge = float(Fraction(sphere.position[0]) + Fraction(sphere.radius))
    clearance = 128.0 if abs(center) > 1 else 2**-14
    near = _box_points((edge, -1, -1), (edge + 2 * clearance, 1, 1))
    far = _box_points((edge + clearance, -1, -1), (edge + 3 * clearance, 1, 1))
    model, ball, wall, static = _one_model(radius=sphere.radius, points=near)
    with TyxWorldBatch([(7, sphere)], tyx_mode, cache_root=tmp_path / "runtime") as batch:
        with batch.prepare_collision(model, _bindings(ball, wall, static)) as plan:
            touched = plan.query()
            _contains_sphere(touched.bounds[0], sphere.position, sphere.radius)
            assert touched.candidates == ((ball, wall),)
        with batch.prepare_collision(model, _bindings(ball, wall, ph.CollisionMotion(far))) as plan:
            assert plan.query().candidates == ()


@pytest.mark.tyx
@pytest.mark.parametrize("filter_kind", ("mask", "excluded", "same_body"))
def test_c04_model_masks_exclusions_and_same_body_are_preserved(filter_kind, tyx_mode, tmp_path):
    model, ball, wall, static = _one_model()
    if filter_kind == "mask":
        model = ph.CollisionModel(
            tuple(replace(s, mask=0) if s.key == ball else s for s in model.shapes)
        )
    elif filter_kind == "excluded":
        model = ph.CollisionModel(model.shapes, ((ball, wall),))
    else:
        replacement = ph.CollisionKey(wall.world, ball.body, wall.shape)
        model = ph.CollisionModel(
            tuple(replace(s, key=replacement) if s.key == wall else s for s in model.shapes)
        )
        wall = replacement
    with TyxWorldBatch(
        [(7, FreeSphere(2, 0.125, gravity=(0, 0, 0)))], tyx_mode, cache_root=tmp_path / "runtime"
    ) as batch:
        with batch.prepare_collision(model, _bindings(ball, wall, static)) as plan:
            assert plan.query().candidates == ()
            assert plan.last_result.registered_pairs == (tuple(sorted((ball, wall))),)


@pytest.mark.tyx
def test_c05_same_tick_restore_reset_and_old_owner_cannot_alias(tyx_mode, tmp_path):
    model, ball, wall, static = _one_model(revision=5)
    sphere = FreeSphere(2, 0.125, (-0.5, 0, 0), gravity=(0, 0, 0))
    with TyxWorldBatch([(7, sphere)], tyx_mode, cache_root=tmp_path / "first") as batch:
        initial = batch.snapshot()[7]
        with batch.prepare_collision(model, _bindings(ball, wall, static)) as plan:
            before = plan.query()
            batch.restore({7: replace(initial, position=(0, 0, 0))})
            moved = plan.query()
            assert moved.ticks.tolist() == before.ticks.tolist() == [0]
            assert moved.state_epoch > before.state_epoch
            assert moved.shape_revisions == (5, 5)
            assert moved.candidates == ((ball, wall),)
            with pytest.raises(ValueError, match="stale|epoch"):
                plan.require_current(before)
            with pytest.raises(ValueError, match="topology"):
                plan.require_current(
                    replace(moved, topology_generation=moved.topology_generation + 1)
                )
            batch.reset([7])
            reset = plan.query()
            assert reset.state_epoch > moved.state_epoch and reset.candidates == ()
            with TyxWorldBatch([(7, sphere)], tyx_mode, cache_root=tmp_path / "second") as other:
                with pytest.raises(ValueError, match="owner|different"):
                    other.collision(plan)
                with other.prepare_collision(model, _bindings(ball, wall, static)) as other_plan:
                    with pytest.raises(ValueError, match="owner|different"):
                        other_plan.require_current(reset)


@pytest.mark.tyx
def test_c06_numeric_reject_exposes_only_each_world_accepted_pose(tyx_mode, tmp_path):
    worlds, model, bindings = _many_worlds(4)
    with TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "runtime") as batch:
        saved = batch.snapshot()
        rejected_world = worlds[0][0]
        batch.restore({rejected_world: replace(saved[rejected_world], tick=2**31 - 1)})
        with batch.prepare_collision(model, bindings) as plan:
            before = plan.query()
            accepted = batch.step(0.125)
            assert accepted[rejected_world] is False
            assert sum(accepted.values()) == 3
            result = plan.query()
            assert result.state_epoch > before.state_epoch
            for world, initial in worlds:
                row = result.shape_keys.index(ph.CollisionKey(world, 3, 11))
                if world == rejected_world:
                    np.testing.assert_array_equal(result.bounds[row], before.bounds[row])
                    assert result.ticks[result.world_keys.index(world)] == 2**31 - 1
                else:
                    _contains_sphere(
                        result.bounds[row], (initial.position[0] + 1 / 32, 0, 0), 0.125
                    )
                    assert result.ticks[result.world_keys.index(world)] == 1


@pytest.mark.tyx
def test_c07_query_failure_does_not_publish_or_change_physics(tyx_mode, tmp_path, monkeypatch):
    model, ball, wall, static = _one_model()
    sphere = FreeSphere(2, 0.125, gravity=(0, 0, 0))
    with TyxWorldBatch([(7, sphere)], tyx_mode, cache_root=tmp_path / "runtime") as batch:
        with batch.prepare_collision(model, _bindings(ball, wall, static)) as plan:
            before = plan.query()
            physical = batch.snapshot()
            original = plan._engine.query_device_bounds

            def failed_submission(**kwargs):
                # Bounds have been submitted from the real accepted buffer first.
                raise RuntimeError("injected collision device failure")

            monkeypatch.setattr(plan._engine, "query_device_bounds", failed_submission)
            with pytest.raises(RuntimeError, match="injected collision"):
                plan.query()
            assert plan.last_result is before
            assert batch.snapshot() == physical
            assert plan.require_current(before) is before
            monkeypatch.setattr(plan._engine, "query_device_bounds", original)
            retry = plan.query()
            assert retry.state_epoch == before.state_epoch
            assert retry.candidates == before.candidates
            np.testing.assert_array_equal(retry.bounds, before.bounds)
            assert plan.stats()["failed_queries"] == 1


@pytest.mark.tyx
def test_c08_step_failure_leaves_accepted_epoch_and_broad_phase(tyx_mode, tmp_path, monkeypatch):
    model, ball, wall, static = _one_model()
    sphere = FreeSphere(2, 0.125, (-0.5, 0, 0), (1, 0, 0), (0, 0, 0))
    with TyxWorldBatch([(7, sphere)], tyx_mode, cache_root=tmp_path / "runtime") as batch:
        with batch.prepare_collision(model, _bindings(ball, wall, static)) as plan:
            before = plan.query()
            original = batch._read_status

            def bad_status():
                raise RuntimeError("injected acceptance read failure")

            monkeypatch.setattr(batch, "_read_status", bad_status)
            with pytest.raises(RuntimeError, match="acceptance read"):
                batch.step(0.5)
            assert plan.require_current(before) is before
            after = plan.query()
            assert after.state_epoch == before.state_epoch
            np.testing.assert_array_equal(after.bounds, before.bounds)
            assert after.ticks.tolist() == [0]
            monkeypatch.setattr(batch, "_read_status", original)
            assert batch.step(0.5) == {7: True}
            assert plan.query().candidates == ((ball, wall),)


@pytest.mark.tyx
def test_c09_capacity_never_publishes_partial_candidates(tyx_mode, tmp_path):
    model, ball, wall, static = _one_model()
    sphere = FreeSphere(2, 0.125, (-0.5, 0, 0), (1, 0, 0), (0, 0, 0))
    with TyxWorldBatch([(7, sphere)], tyx_mode, cache_root=tmp_path / "runtime") as batch:
        with batch.prepare_collision(
            model, _bindings(ball, wall, static), candidate_capacity=0
        ) as small:
            first = small.query()
            assert first.candidates == ()
            assert batch.step(0.5) == {7: True}
            with pytest.raises(ValueError, match="capacity"):
                small.query()
            assert small.last_result is first
            with batch.prepare_collision(
                model, _bindings(ball, wall, static), candidate_capacity=1
            ) as full:
                assert full.query().candidates == ((ball, wall),)


@pytest.mark.tyx
def test_c10_plan_close_frees_own_resources_and_batch_close_closes_all(tyx_mode, tmp_path):
    model, ball, wall, static = _one_model()
    batch = TyxWorldBatch([(7, FreeSphere(2, 0.125))], tyx_mode, cache_root=tmp_path / "runtime")
    try:
        first = batch.prepare_collision(model, _bindings(ball, wall, static))
        second = batch.prepare_collision(model, _bindings(ball, wall, static))
        first.query()
        first.close()
        first.close()
        assert first.stats()["closed"] is True
        with pytest.raises(RuntimeError, match="closed"):
            first.query()
        # A plan borrows the owner's runtime. Closing it cannot kill world stepping.
        assert batch.step(0.125) == {7: True}
        second.query()
        batch.close()
        assert second.stats()["closed"] is True
        with pytest.raises(RuntimeError, match="closed"):
            second.query()
    finally:
        batch.close()


def test_c11_empty_plan_needs_no_runtime_and_keeps_owned_empty_result():
    with TyxWorldBatch([], "interpret") as batch:
        with batch.prepare_collision(ph.CollisionModel(()), ()) as plan:
            result = plan.query()
            assert result.world_keys == result.shape_keys == result.candidates == ()
            assert result.registered_pairs == result.shape_revisions == ()
            assert result.bounds.shape == (0, 6)
            assert result.ticks.shape == (0,)
            assert plan.require_current(result) is result
            assert plan.stats()["device_allocations"] == 0
            assert plan.stats()["bounds_kernel_launches"] == 0
            assert batch.step(0.125) == {}
            assert plan.query().candidates == ()


@pytest.mark.tyx
@pytest.mark.parametrize(
    "problem",
    ("missing", "duplicate", "wrong_world", "radius", "moving", "unbounded", "body_alias"),
)
def test_c12_invalid_binding_rejected_before_device_allocation(problem, tyx_mode, tmp_path):
    sphere_binding, static_binding = _api()
    model, ball, wall, static = _one_model()
    bindings = list(_bindings(ball, wall, static))
    if problem == "missing":
        bindings.pop()
    elif problem == "duplicate":
        bindings.append(bindings[0])
    elif problem == "wrong_world":
        bindings[-1] = sphere_binding(ball, 8)
    elif problem == "radius":
        model = ph.CollisionModel(
            tuple(replace(s, radius=0.25) if s.key == ball else s for s in model.shapes)
        )
    elif problem == "moving":
        bindings[0] = static_binding(
            wall, ph.CollisionMotion(static.start, static.start + [1, 0, 0])
        )
    elif problem == "unbounded":
        model = ph.CollisionModel(
            (model.shapes[0], ph.CollisionShape(wall, "plane", (0,), normal=(0, 0, 1)))
        )
        bindings[0] = static_binding(wall, ph.CollisionMotion([[0, 0, 0]]))
    else:
        other = ph.CollisionKey(7, ball.body, 99)
        model = ph.CollisionModel((*model.shapes, ph.CollisionShape(other, "sphere", (5,), 0.125)))
        bindings.append(sphere_binding(other, 7))
    with TyxWorldBatch(
        [(7, FreeSphere(2, 0.125))], tyx_mode, cache_root=tmp_path / "runtime"
    ) as batch:
        before = batch.stats()
        with pytest.raises((ValueError, TypeError)):
            batch.prepare_collision(model, bindings)
        assert batch.stats()["device_allocations"] == before["device_allocations"]
        assert not batch._collision_plans


@pytest.mark.tyx
def test_c13_slider_cannot_be_inferred_from_a_scalar_row(tyx_mode, tmp_path):
    from test_resident_step import slider

    from phymeo.backends.tyx_world_batch import PrismaticSlider

    program, state = slider(delay=0)
    model, ball, wall, static = _one_model()
    with TyxWorldBatch(
        [(7, PrismaticSlider(program, state))], tyx_mode, cache_root=tmp_path / "runtime"
    ) as batch:
        with pytest.raises(ValueError, match="sphere|FreeSphere"):
            batch.prepare_collision(model, _bindings(ball, wall, static))


@pytest.mark.tyx
def test_c14_static_mesh_uses_registered_material_vertices(tyx_mode, tmp_path):
    sphere_binding, static_binding = _api()
    ball, solid = ph.CollisionKey(7, 3, 11), ph.CollisionKey(7, 9, 4)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    ids = (18, 5, 91, 2)
    faces = ((18, 91, 5), (18, 5, 2), (18, 2, 91), (5, 91, 2))
    model = ph.CollisionModel(
        (
            ph.CollisionShape(ball, "sphere", (37,), 0.125),
            ph.CollisionShape(solid, "mesh", ids, triangles=faces),
        )
    )
    bindings = (static_binding(solid, ph.CollisionMotion(vertices)), sphere_binding(ball, 7))
    with TyxWorldBatch(
        [(7, FreeSphere(2, 0.125, (-0.125, 0, 0), gravity=(0, 0, 0)))],
        tyx_mode,
        cache_root=tmp_path / "runtime",
    ) as batch:
        with batch.prepare_collision(model, bindings) as plan:
            result = plan.query()
            assert result.candidates == ((ball, solid),)
            np.testing.assert_array_equal(result.bounds[1], [0, 0, 0, 1, 1, 1])


@pytest.mark.tyx
def test_c15_overflow_bounds_fail_without_replacing_valid_result(tyx_mode, tmp_path):
    model, ball, wall, static = _one_model()
    with TyxWorldBatch(
        [(7, FreeSphere(2, 0.125, gravity=(0, 0, 0)))], tyx_mode, cache_root=tmp_path / "runtime"
    ) as batch:
        saved = batch.snapshot()[7]
        with batch.prepare_collision(model, _bindings(ball, wall, static)) as plan:
            first = plan.query()
            batch.restore({7: replace(saved, position=(float(np.finfo(np.float32).max), 0, 0))})
            with pytest.raises(ValueError, match="finite|bound|overflow"):
                plan.query()
            assert plan.last_result is first
            batch.restore({7: saved})
            assert plan.query().candidates == first.candidates


@pytest.mark.tyx
@pytest.mark.parametrize("count", (1, 4, 32))
def test_c16_same_task_existing_envelope_route_has_same_candidates(count, tyx_mode, tmp_path):
    from phymeo.geometry.collision_pipeline import _envelopes

    worlds, model, bindings = _many_worlds(count)
    static_paths = {b.key: b.motion for b in bindings if hasattr(b, "motion")}
    with TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "world") as batch:
        with (
            batch.prepare_collision(model, bindings) as plan,
            ph.CollisionWorkspace(
                model, tyx_mode, cache_root=tmp_path / "old", broad_phase="bvh"
            ) as old,
        ):
            for _ in range(2):
                assert all(batch.step(0.125).values())
                actual = plan.query()
                # This explicit old route intentionally pays the snapshot and HOST
                # envelopes; it is a comparison, never an implementation of plan.query.
                checkpoints = batch.snapshot()
                paths = dict(static_paths)
                for world, state in checkpoints.items():
                    paths[ph.CollisionKey(world, 3, 11)] = ph.CollisionMotion([state.position])
                envelopes = _envelopes(model, paths, 0)
                before = old._engine.stats()
                bounds, flags = old._engine.query(envelopes, trees=old._trees, pending={})
                pairs = tuple(
                    (model.shapes[i].key, model.shapes[j].key)
                    for (i, j, _), flag in zip(old._pairs, flags, strict=True)
                    if flag == 0
                )
                assert actual.candidates == pairs
                assert np.all(actual.bounds[:, :3] <= bounds[:, :3])
                assert np.all(actual.bounds[:, 3:] >= bounds[:, 3:])
                if tyx_mode == "cuda":
                    assert (
                        old._engine.stats()["uploaded_bytes"] - before["uploaded_bytes"]
                        >= envelopes.nbytes
                    )
                assert plan.stats()["dynamic_envelope_uploaded_bytes"] == 0


@pytest.mark.parametrize("capacity", (-1, True, 0.5, 32641))
def test_c17_invalid_capacity_is_host_validation(capacity):
    with TyxWorldBatch([], "interpret") as batch:
        with pytest.raises(ValueError, match="capacity"):
            batch.prepare_collision(ph.CollisionModel(()), (), candidate_capacity=capacity)


@pytest.mark.tyx
def test_c18_close_callback_failure_still_closes_every_plan(tyx_mode, tmp_path, monkeypatch):
    model, ball, wall, static = _one_model()
    batch = TyxWorldBatch([(7, FreeSphere(2, 0.125))], tyx_mode, cache_root=tmp_path / "runtime")
    plans = [batch.prepare_collision(model, _bindings(ball, wall, static)) for _ in range(2)]
    original = plans[0].close

    def failed_callback():
        original()
        raise RuntimeError("injected collision plan close callback")

    monkeypatch.setattr(plans[0], "close", failed_callback)
    with pytest.raises(RuntimeError, match="close callback"):
        batch.close()
    assert all(p.stats()["closed"] for p in plans)
    assert batch.stats()["device_owned_bytes"] == 0


@pytest.mark.tyx
def test_c19_same_batch_different_static_registration_cannot_alias(tyx_mode, tmp_path):
    model, ball, wall, near = _one_model()
    far = ph.CollisionMotion(_box_points((10, -0.25, -0.25), (10.5, 0.25, 0.25)))
    with TyxWorldBatch(
        [(7, FreeSphere(2, 0.125, gravity=(0, 0, 0)))], tyx_mode, cache_root=tmp_path / "runtime"
    ) as batch:
        with batch.prepare_collision(model, _bindings(ball, wall, near)) as a:
            with batch.prepare_collision(model, _bindings(ball, wall, far)) as b:
                first, second = a.query(), b.query()
                assert first.state_epoch == second.state_epoch
                assert first.shape_keys == second.shape_keys
                assert first.candidates == ((ball, wall),)
                assert second.candidates == ()
                assert a.require_current(first) is first
                with pytest.raises(ValueError, match="registration|plan"):
                    b.require_current(first)
