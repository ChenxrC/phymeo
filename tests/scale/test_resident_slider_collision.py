"""Real accepted single-slider poses feed the existing resident device broad phase."""

from dataclasses import replace
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends.tyx_world_batch import FreeSphere, PrismaticSlider, TyxWorldBatch


def _api():
    from phymeo.backends._resident_slider_binding import (
        ResidentSliderBinding,
        SliderGeometryFrame,
        compile_slider_binding,
    )

    return ResidentSliderBinding, SliderGeometryFrame, compile_slider_binding


def _static(key, motion):
    from phymeo.backends._resident_collision_plan import ResidentStaticBinding

    return ResidentStaticBinding(key, motion)


def _sphere(key):
    from phymeo.backends._resident_collision_plan import ResidentSphereBinding

    return ResidentSphereBinding(key, key.world)


def _box(low, high):
    return np.asarray(list(product(*zip(low, high, strict=True))), dtype=float)


def _program(
    *,
    origin=(0, 0, 0),
    orientation=(0, 0, 0, 1),
    axis=(1, 0, 0),
    mass=2.0,
    delay=0,
    q=0.0,
    v=0.0,
):
    model = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.array([origin], dtype=float),
        np.array([orientation], dtype=float),
        np.array([axis], dtype=float),
        np.array([mass], dtype=float),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )
    drive = ph.DrivenRobotProgram(model, kp=[0.0], kd=[0.0], effort_limit=[4.0], delay_steps=delay)
    initial = drive.initial_state(ph.RobotState(np.array([q]), np.array([v])))
    return drive, initial


def _frame_args(program):
    model = program.model
    return {
        "origin_m": model.joint_position[0],
        "orientation_xyzw": model.joint_orientation[0],
        "axis_local": model.joint_axis[0],
    }


def _binding(key, program, zero_motion):
    binding, _, _ = _api()
    return binding(key, key.world, zero_motion, **_frame_args(program))


def _zero_motion(program, local_vertices):
    geometry = ph.forward_kinematics(program.model, np.zeros(1))
    points = geometry.position[0] + np.asarray(local_vertices) @ geometry.rotation[0].T
    return ph.CollisionMotion(points)


def _contains(bounds, low, high):
    for i in range(3):
        assert Fraction(float(bounds[i])) <= Fraction(low[i])
        assert Fraction(float(bounds[i + 3])) >= Fraction(high[i])


def _row(result, key):
    return result.bounds[result.shape_keys.index(key)]


def _s01(world=7):
    s = np.sqrt(0.5)
    drive, initial = _program(
        origin=(1, -0.5, 0.25),
        orientation=(0, 0, s, s),
        axis=(1, 0, 0),
        q=0.125,
    )
    moving = ph.CollisionKey(world, 3, 11)
    wall = ph.CollisionKey(world, 9, 4)
    local = _box((1 / 8, -1 / 16, -1 / 32), (3 / 8, 1 / 16, 1 / 32))
    zero = _zero_motion(drive, local)
    fixed = ph.CollisionMotion(
        _box((15 / 16, 1 / 16, 7 / 32), (17 / 16, 3 / 16, 9 / 32))
    )
    model = ph.CollisionModel(
        (
            ph.CollisionShape(wall, "hull", tuple(range(8))),
            ph.CollisionShape(moving, "hull", tuple(range(101, 109))),
        )
    )
    return drive, initial, model, moving, wall, zero, fixed


@pytest.mark.tyx
def test_s01_actual_slider_step_changes_public_device_bounds(tyx_mode, tmp_path):
    drive, initial, model, moving, wall, zero, fixed = _s01()
    with TyxWorldBatch([(7, PrismaticSlider(drive, initial))], tyx_mode, cache_root=tmp_path) as batch:
        command = drive.inputs(np.zeros(1), feedforward=np.array([2.0]))
        batch.set_controls({7: command})
        assert batch.step(1 / 4) == {7: True}
        actual = batch.snapshot()[7]
        assert actual.position == (3 / 16,)
        assert actual.velocity == (1 / 4,)
        # The pre-existing physical program must advance first. The valid RED uses
        # the candidate binding module with the old plan's unsupported-slider gate,
        # never a missing-module/import failure as proof of the public gap.
        bindings = (_binding(moving, drive, zero), _static(wall, fixed))
        with batch.prepare_collision(model, bindings) as plan:
            result = batch.collision(plan)
            assert result.world_keys == (7,)
            assert result.ticks.tolist() == [1]
            assert result.registered_pairs == ((moving, wall),)
            assert result.candidates == ((moving, wall),)
            _contains(
                _row(result, moving),
                (Fraction(15, 16), Fraction(-3, 16), Fraction(7, 32)),
                (Fraction(17, 16), Fraction(1, 16), Fraction(9, 32)),
            )
            assert plan.last_result is result
            assert plan.require_current(result) is result
            np.testing.assert_array_equal(bindings[0].zero_motion.start, zero.start)
