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
        [(x, y, z) for x in (low[0], high[0]) for y in (low[1], high[1])
         for z in (low[2], high[2])],
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
