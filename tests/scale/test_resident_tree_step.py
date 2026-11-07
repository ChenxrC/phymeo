"""T1 fixed-tree resident dynamics: frozen inputs, authoring only / UNEXECUTED.

The public RED constructs a valid existing RobotModel/DrivenRobotProgram first.
No expected FK, mass, bias, or load projection comes from Phymeo mechanics.
"""

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from math import cos, pi, sin

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends import tyx_world_batch as resident


RTOL = 5e-5
ATOL = 5e-6
WINDOW_ATOL = 2e-4


def _json(value):
    if is_dataclass(value):
        return {f.name: _json(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    return value


def _save(tmp_path, name, value):
    (tmp_path / (name + ".json")).write_text(
        json.dumps(_json(value), indent=2, allow_nan=False), encoding="utf-8"
    )


def _program(model, *, q=None, v=None, kp=None, kd=None, cap=None, delay=0):
    n = model.dof_count
    program = ph.DrivenRobotProgram(
        model, kp=np.zeros(n) if kp is None else np.asarray(kp, dtype=float),
        kd=np.zeros(n) if kd is None else np.asarray(kd, dtype=float),
        effort_limit=np.full(n, 8.0) if cap is None else np.asarray(cap, dtype=float),
        delay_steps=delay,
    )
    initial = program.initial_state(ph.RobotState(
        np.zeros(n) if q is None else np.asarray(q, dtype=float),
        np.zeros(n) if v is None else np.asarray(v, dtype=float),
    ))
    return program, initial


def _rr_model():
    return ph.RobotModel(
        np.array([-1, 0]), ("revolute", "revolute"),
        np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
        np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        np.tile([0.0, 0.0, 1.0], (2, 1)), np.ones(2),
        np.tile([0.5, 0.0, 0.0], (2, 1)),
        np.tile([1 / 24, 1 / 24, 1 / 12], (2, 1)), np.zeros(3),
    )


def _action(program, *, target=None, rate=None, ff=None, effort=None, force=None, torque=None):
    n = program.model.dof_count
    vector = lambda value: np.zeros(n) if value is None else np.asarray(value, dtype=float)
    field = lambda value: np.zeros((n, 3)) if value is None else np.asarray(value, dtype=float)
    return program.inputs(vector(target), vector(rate), feedforward=vector(ff),
                          loads=ph.RobotInputs(vector(effort), field(force), field(torque)))


def _tree(program, initial, *, iterations=512):
    return resident.FixedRobotTree(program, initial, control_max_iterations=iterations)


def _close(actual, expected, *, atol=ATOL):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=atol)


@pytest.mark.tyx
def test_t01_public_tree_step_has_real_coupled_two_revolute_response(tyx_mode, tmp_path):
    model = _rr_model()
    assert model.dof_count == 2 and tuple(model.parents) == (-1, 0)
    program, initial = _program(model)
    assert initial.domains[program.domain].q.tolist() == [0.0, 0.0]
    command = _action(program, effort=(1, 1 / 6))
    _save(tmp_path, "t01-public-input", {"model": model, "initial": initial,
                                        "command": command, "h_s": 1 / 128})
    # First missing capability occurs here, not during collection/import.
    world = _tree(program, initial)
    with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: command})
        accepted = batch.step(1 / 128)
        actual = batch.snapshot()[7]
        _save(tmp_path, f"t01-public-{tyx_mode}", {
            "accepted": accepted, "actual": actual, "stats": batch.stats(),
            "oracle_mass": [[8 / 3, 5 / 6], [5 / 6, 1 / 3]],
            "oracle_acceleration": [1, -2],
        })
        assert accepted == {7: True} and actual.tick == 1
        _close(actual.velocity, [1 / 128, -2 / 128])
        _close(actual.position, [1 / 128**2, -2 / 128**2])
        np.testing.assert_array_equal(actual.applied_effort, [0, 0])
        _close(actual.generalized_applied, [1, 1 / 6])
        read = batch.tree_kinematics([7])
        assert tuple(read) == (7,) and read[7].tick == 1
        assert read[7].joint_types == ("revolute", "revolute")
        _close(read[7].position, actual.position)
        _close(read[7].velocity, actual.velocity)
