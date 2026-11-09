"""AUTHORING_ONLY: bounded W256 connection, not GPU contact or PPO throughput.

One real eight-step mixed resident scene uses independent mass/inertia
recurrences, policy hold clocks and heterogeneous horizon resets. The second
node only proves the existing 256-shape contact registration boundary; it has
zero physical steps and must not be counted as loaded W256 support.
"""

import gzip
import json
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.constraints.collision_shapes import MAX_SHAPES, MAX_VERTICES
from phymeo.control.policy_runner import PolicyRunner
from phymeo.training import ResidentVectorEnv

WORLDS, STEPS, DECIMATION, H = 256, 8, 3, 1 / 128
MAX_SAVED_BYTES = 64 * 1024 * 1024


def _encode(value):
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": value.shape,
                "bytes_hex": value.tobytes().hex(), "values": _encode(value.tolist())}
    if isinstance(value, np.generic):
        return _encode(value.item())
    if is_dataclass(value):
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_ieee_value": repr(value)}
    if isinstance(value, Path):
        return str(value)
    return value


def _save(path, name, record):
    payload = json.dumps(_encode(record), allow_nan=False).encode("utf-8")
    if len(payload) > MAX_SAVED_BYTES:
        raise OSError("complete W256 evidence exceeds its fixed 64 MiB bound")
    with gzip.open(path / (name + ".json.gz"), "xb") as stream:
        stream.write(payload)


def _initial_data():
    row = np.arange(WORLDS)
    rotor = row % 2 == 1
    keys = tuple(1003 + 7 * int(i) for i in row)
    q, v = np.zeros((WORLDS, 3)), np.zeros((WORLDS, 3))
    q[~rotor, 0] = (row[~rotor] % 8) / 16
    q[~rotor, 1] = (row[~rotor] % 3 - 1) / 32
    q[~rotor, 2] = .5
    q[rotor, 0] = (row[rotor] % 3 - 1) / 32
    v[:, 0] = (row % 5 - 2) / 64
    v[~rotor, 1] = (row[~rotor] % 4 - 1) / 128
    horizon = 3 + 2 * ((row // 2) % 4)
    base = np.column_stack(((row % 5 - 2) / 8, (row % 3 - 1) / 16, np.full(WORLDS, 1 / 8)))
    base[rotor, 1:] = 0
    return keys, rotor, q, v, horizon, base


def _program():
    model = ph.RobotModel(np.array([-1]), ("revolute",), np.zeros((1, 3)),
                         np.array([[0., 0, 0, 1]]), np.array([[0., 0, 1]]),
                         np.array([2.]), np.zeros((1, 3)), np.full((1, 3), .5), np.zeros(3))
    return ph.DrivenRobotProgram(model, kp=np.zeros(1), kd=np.zeros(1),
                                 effort_limit=np.array([8.]), delay_steps=0)


class MixedTask:
    def __init__(self, keys, rotor, program):
        self.keys, self.rotor, self.program = keys, rotor, program
        self.last_controls = None
        self.last_before = self.last_after = None
        self.evaluations = 0

    def controls(self, actions, keys):
        commands = {}
        for row, key in enumerate(keys):
            if self.rotor[row]:
                commands[key] = self.program.inputs(np.zeros(1), np.zeros(1), feedforward=actions[row, :1])
            else:
                commands[key] = np.array(actions[row], copy=True)
        self.last_controls = commands
        return commands

    def observe(self, states, keys):
        result = np.zeros((len(keys), 6))
        for row, key in enumerate(keys):
            state = states[key]
            if self.rotor[row]:
                result[row, 0], result[row, 3] = state.position[0], state.velocity[0]
            else:
                result[row, :3], result[row, 3:] = state.position, state.velocity
        return result

    def evaluate(self, before, after, actions, keys):
        self.last_before, self.last_after = before, after
        self.evaluations += 1
        observation = self.observe(after, keys)
        return -np.sum(observation**2, axis=1), np.zeros(len(keys), dtype=bool)


class HeldFeedback:
    def __init__(self, rotor, base):
        self.rotor, self.base = rotor, base
        self.calls = []

    def __call__(self, observation, due_mask):
        assert not observation.flags.writeable and not due_mask.flags.writeable
        result = self.base - observation[:, :3] * [1 / 4, 1 / 8, 1 / 8] - observation[:, 3:] * [1 / 8, 1 / 16, 1 / 16]
        result[self.rotor, 1:] = 0
        self.calls.append(dict(observation=observation, due_mask=due_mask, proposed=result.copy()))
        return result


def _physical_state_assert(states, keys, rotor, q, v, ticks, held, just_reset):
    for row, key in enumerate(keys):
        state = states[key]
        width = 1 if rotor[row] else 3
        assert state.world == key and state.tick == int(ticks[row])
        np.testing.assert_allclose(state.position, q[row, :width], rtol=5e-5, atol=5e-6)
        np.testing.assert_allclose(state.velocity, v[row, :width], rtol=5e-5, atol=5e-6)
        applied = np.zeros(width) if just_reset[row] else held[row, :width]
        np.testing.assert_allclose(state.applied_effort, applied, rtol=5e-5, atol=5e-6)


@pytest.mark.tyx
def test_connection_w256_mixed_motion_policy_hold_and_independent_resets(tyx_mode, tmp_path):
    keys, rotor, initial_q, initial_v, horizons, base = _initial_data()
    program = _program()
    worlds = []
    for row, key in enumerate(keys):
        if rotor[row]:
            state = program.initial_state(ph.RobotState(initial_q[row, :1], initial_v[row, :1]))
            worlds.append((key, ph.RevoluteRotor(program, state)))
        else:
            worlds.append((key, ph.FreeSphere(2., 1 / 16, position=tuple(initial_q[row]),
                                             velocity=tuple(initial_v[row]), gravity=(0, 0, 0))))
    record = dict(mode=tyx_mode, world_keys=keys, rotor_mask=rotor, initial_q=initial_q,
                  initial_v=initial_v, horizons=horizons, policy_base=base,
                  sphere_mass_kg=2., rotor_axis_inertia=.5, rotor_model=program.model,
                  h_s=H, requested_physical_steps=STEPS, control_decimation=DECIMATION, events=[])
    batch = None
    try:
        with ph.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path) as batch:
            task = MixedTask(keys, rotor, program)
            env = ResidentVectorEnv(batch, task, keys, action_size=3, dt=H, horizon=horizons)
            policy = HeldFeedback(rotor, base)
            runner = PolicyRunner(env, policy, control_decimation=DECIMATION,
                                  initial_observation=task.observe(batch.snapshot(), keys))
            q, v = initial_q.copy(), initial_v.copy()
            ticks, lengths, episodes = (np.zeros(WORLDS, dtype=np.int64) for _ in range(3))
            refresh, held = np.ones(WORLDS, dtype=bool), np.zeros((WORLDS, 3))
            expected_policy_calls = 0
            for step in range(STEPS):
                event = dict(step=step, before=batch.snapshot(), runner_before=runner.snapshot())
                record["events"].append(event)
                # Independent exact-mass/inertia semi-implicit recurrence. No HOST
                # product M/FK/dynamics is used to generate these expectations.
                due = refresh | (ticks % DECIMATION == 0)
                expected_observation = np.hstack((q, v))
                proposed = base - q * [1 / 4, 1 / 8, 1 / 8] - v * [1 / 8, 1 / 16, 1 / 16]
                proposed[rotor, 1:] = 0
                held[due] = proposed[due]
                expected_policy_calls += int(np.any(due))
                acceleration = held / 2
                acceleration[rotor] = 0
                acceleration[rotor, 0] = 2 * held[rotor, 0]
                v = v + H * acceleration
                q = q + H * v
                final = np.hstack((q, v))
                reward = -np.sum(final**2, axis=1)
                lengths += 1
                completed = lengths >= horizons
                episodes += completed
                entry_ticks = ticks.copy()
                ticks += 1
                ticks[completed], lengths[completed] = 0, 0
                q[completed], v[completed] = initial_q[completed], initial_v[completed]
                refresh = completed.copy()
                event.update(expected_due=due, expected_entry_ticks=entry_ticks,
                             expected_policy_observation=expected_observation,
                             expected_actions=held.copy(), expected_final_observation=final,
                             expected_after_reset=np.hstack((q, v)), expected_reward=reward,
                             expected_completed=completed, expected_episodes=episodes.copy(),
                             expected_lengths=lengths.copy())
                try:
                    result = runner.step()
                except Exception as error:
                    event.update(error_type=type(error).__name__, error=str(error),
                                 retained=batch.snapshot(), runner_after_failure=runner.snapshot(),
                                 commands=task.last_controls)
                    raise
                actual, snapshot = result.transition, batch.snapshot()
                event.update(result=result, commands=task.last_controls, actual_after=snapshot,
                             actual_load_time_states=task.last_after, runner_after=runner.snapshot(),
                             policy_calls=tuple(policy.calls), actual_summary=batch.summary(), stats=batch.stats())
                assert len(policy.calls) == expected_policy_calls
                np.testing.assert_array_equal(result.due_mask, due)
                np.testing.assert_array_equal(result.physical_ticks, entry_ticks)
                np.testing.assert_allclose(result.actions, held, rtol=5e-5, atol=5e-6)
                np.testing.assert_allclose(actual.final_observation, final, rtol=5e-5, atol=5e-6)
                np.testing.assert_allclose(actual.observation, np.hstack((q, v)), rtol=5e-5, atol=5e-6)
                np.testing.assert_allclose(actual.reward, reward, rtol=5e-5, atol=5e-6)
                np.testing.assert_array_equal(actual.terminated, np.zeros(WORLDS, dtype=bool))
                np.testing.assert_array_equal(actual.truncated, completed)
                np.testing.assert_array_equal(actual.final_mask, completed)
                np.testing.assert_array_equal(actual.episode_id, episodes)
                np.testing.assert_array_equal(actual.episode_length, lengths)
                np.testing.assert_array_equal(runner.snapshot().physical_ticks, ticks)
                _physical_state_assert(snapshot, keys, rotor, q, v, ticks, held, completed)
                _physical_state_assert(task.last_after, keys, rotor, final[:, :3], final[:, 3:],
                                       entry_ticks + 1, held, np.zeros(WORLDS, dtype=bool))
                energy = np.sum(v**2, axis=1)
                energy[rotor] = .25 * v[rotor, 0]**2
                np.testing.assert_allclose([event["actual_summary"][key]["kinetic_energy_j"] for key in keys],
                                           energy, rtol=5e-5, atol=5e-6)
                assert not runner.needs_reset
            assert task.evaluations == STEPS and 1 < len(policy.calls) < STEPS
            assert not np.array_equal(record["events"][0]["result"].actions, record["events"][-1]["result"].actions)
            assert any(np.any(call["due_mask"]) and not np.all(call["due_mask"]) for call in policy.calls)
            assert np.any(episodes == 0) and np.any(episodes >= 2)
            assert batch.stats()["accepted_world_steps"] == WORLDS * STEPS
            # A sparse explicit reset after the window must not alter any other
            # accepted checkpoint or reschedule its held policy action.
            selected = np.array([0, 127, 255])
            before_reset, before_runner = batch.snapshot(), runner.snapshot()
            runner.reset(selected)
            after_reset, after_runner = batch.snapshot(), runner.snapshot()
            for row, key in enumerate(keys):
                if row not in selected:
                    assert after_reset[key] == before_reset[key]
                    assert after_runner.physical_ticks[row] == before_runner.physical_ticks[row]
                    np.testing.assert_array_equal(after_runner.held_actions[row], before_runner.held_actions[row])
                else:
                    assert after_reset[key].tick == 0 and after_runner.physical_ticks[row] == 0
                    assert after_runner.refresh_mask[row]
                    width = 1 if rotor[row] else 3
                    np.testing.assert_allclose(after_reset[key].position, initial_q[row, :width], rtol=0, atol=5e-6)
                    np.testing.assert_allclose(after_reset[key].velocity, initial_v[row, :width], rtol=0, atol=5e-6)
            record.update(status="passed", explicit_reset_indices=selected,
                          before_explicit_reset=before_reset, after_explicit_reset=after_reset,
                          runner_before_explicit_reset=before_runner, runner_after_explicit_reset=after_runner,
                          policy_calls=policy.calls, final_stats=batch.stats())
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        if batch is not None:
            record["last_actual_statistics"] = batch.stats()
        _save(tmp_path, "connection-w256-mixed", record)


def test_connection_w256_contact_registration_rejects_existing_shape_capacity(tmp_path):
    # No batch, contact plan, runtime or physical step is created in this node.
    keys = tuple(1003 + 7 * row for row in range(WORLDS))
    shapes = tuple(shape for world in keys for shape in (
        ph.CollisionShape(ph.CollisionKey(world, 100, 1), "hull", tuple(range(101, 109))),
        ph.CollisionShape(ph.CollisionKey(world, 900, 2), "hull", tuple(range(201, 209))),
    ))
    assert MAX_SHAPES == 256 and len(shapes) == 512
    assert sum(len(shape.vertex_ids) for shape in shapes) == 4096 <= MAX_VERTICES
    with pytest.raises(ValueError, match="shape") as caught:
        ph.CollisionModel(shapes)
    _save(tmp_path, "connection-w256-contact-boundary", dict(
        world_keys=keys, shapes=shapes, declared_shape_count=512, existing_shape_cap=MAX_SHAPES,
        declared_vertex_count=4096, existing_vertex_cap=MAX_VERTICES,
        physical_steps=0, status="expected_registration_rejection",
        error_type=type(caught.value).__name__, error=str(caught.value),
        scope="negative declaration only; not loaded W256 contact support",
    ))
