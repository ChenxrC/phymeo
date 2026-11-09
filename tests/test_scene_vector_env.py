"""AUTHORING_ONLY: original loaded HOST scenes through the vector-env boundary.

Every numerical result must come from the parent serial run. These fixtures do
not substitute a fake dynamics batch, write q/v, pin a free root or import G1.
"""

import json
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, fields, is_dataclass

import numpy as np
import pytest

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand, RobotSceneObservation
from phymeo.training import ResidentVectorEnv, VectorEnvStepError
from phymeo.training.resident_stepper import PhysicsStepFailure
from phymeo.training.scene_backend import SceneBatchBackend, SceneControl

H = 1 / 1024
G = 9.81
K = 1000.0
RADIUS = 1 / 16
ROBOT = 10
FLOOR = 20
SPECS = {
    7: dict(mass=2.0, q=0.0, v=0.02, depth=0.003),
    19: dict(mass=3.0, q=0.001, v=-0.01, depth=0.005),
    23: dict(mass=1.5, q=-0.001, v=0.005, depth=0.004),
}


def _serial(value):
    """Retain every dataclass field, array dtype/shape and original mapping key."""
    if isinstance(value, SceneControl):
        return {"actions": _serial(value.actions), "external": _serial(value.external)}
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": list(value.shape), "data": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__module__ + "." + type(value).__qualname__,
            "fields": {field.name: _serial(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, Mapping):
        return {"mapping": [[_serial(key), _serial(item)] for key, item in value.items()]}
    if isinstance(value, (tuple, list)):
        return [_serial(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("unhandled evidence value: " + type(value).__name__)


class _Factory:
    """Each call builds a real original scene; only close ownership is observed."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp_path, self.monkeypatch = tmp_path, monkeypatch
        self.created = {key: [] for key in SPECS}
        self.close_counts, self.calls, self.sources = {}, [], {}
        self.fail_world = None
        self.alias_world = None

    def __call__(self, world):
        generation = len(self.created[world])
        self.calls.append((world, generation))
        if world == self.fail_world:
            raise RuntimeError("injected factory construction failure")
        if world == self.alias_world:
            return self.created[world][-1]
        spec = SPECS[world]
        source = f'''<robot name="loaded_slider_{world}"><link name="base"/>
<link name="foot"><inertial><mass value="{spec["mass"]}"/>
<inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/>
</inertial><collision name="foot_ball"><geometry>
<sphere radius="{RADIUS}"/></geometry></collision></link>
<joint name="slide" type="prismatic"><parent link="base"/><child link="foot"/>
<origin xyz="0 0 {RADIUS - spec["depth"] - spec["q"]}"/><axis xyz="0 0 1"/>
<limit lower="-.2" upper=".2" effort="128" velocity="10"/></joint></robot>'''
        self.sources[world] = source
        path = self.tmp_path / f"world-{world}-generation-{generation}.urdf"
        path.write_text(source, encoding="utf-8")
        builder = ContactSceneBuilder(gravity=(0, 0, -G))
        builder.add_urdf(
            ROBOT,
            path,
            robot_state=RobotState([spec["q"]], [spec["v"]]),
            kp=[0],
            kd=[0],
            effort_limit=[128],
        )
        builder.add_box(FLOOR, (0.5, 0.5, 1 / 32), mass=None, position=(0, 0, -1 / 32))
        builder.pair(ROBOT, FLOOR, RobotContactMaterial(K, 0.0, 0.0))
        scene = builder.build(backend="host")
        original_close = scene.close
        self.close_counts[world, generation] = 0

        def close():
            self.close_counts[world, generation] += 1
            original_close()

        self.monkeypatch.setattr(scene, "close", close)
        self.created[world].append(scene)
        return scene


@contextmanager
def _trace(tmp_path, name, factory):
    record = {
        "schema": "original-host-scene-vector-env-v1",
        "h_s": H,
        "specs": SPECS,
        "normal_law": {"stiffness_n_m": K, "damping_ns_m": 0, "mu": 0},
        "events": [],
        "failure": None,
    }
    try:
        yield record["events"]
    except BaseException as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        record.update(
            sources=factory.sources, factory_calls=factory.calls, close_counts=factory.close_counts
        )
        (tmp_path / (name + ".json")).write_text(
            json.dumps(_serial(record), allow_nan=False), encoding="utf-8"
        )


def _packets(keys, efforts=None):
    efforts = {} if efforts is None else efforts
    return {
        key: SceneControl(
            {ROBOT: RobotCommand([SPECS[key]["q"]], feedforward=[efforts.get(key, 0)])}
        )
        for key in keys
    }


def _step(backend, events, controls):
    event = {"kind": "backend-step", "before": backend.snapshot(), "controls": controls}
    events.append(event)
    backend.set_controls(controls)
    try:
        result = backend.step(H)
    except Exception as error:
        event.update(
            error={"type": type(error).__name__, "message": str(error)},
            accepted_worlds=getattr(error, "accepted_worlds", None),
            acceptance_known=getattr(error, "acceptance_known", None),
            after=backend.snapshot(),
        )
        raise
    event.update(accepted_worlds=result, after=backend.snapshot())
    return result


class _Task:
    def __init__(self, events, *, terminating_worlds=()):
        self.events, self.terminating_worlds = events, tuple(terminating_worlds)
        self.bad_world, self.evaluations = None, 0

    def controls(self, actions, keys):
        result = {
            key: SceneControl(
                {
                    ROBOT: RobotCommand(
                        [2.0 if key == self.bad_world else SPECS[key]["q"]],
                        feedforward=[actions[index, 0]],
                    )
                }
            )
            for index, key in enumerate(keys)
        }
        self.events.append({"kind": "task-controls", "actions": actions, "controls": result})
        return result

    def observe(self, states, keys):
        self.events.append({"kind": "task-observe", "states": states})
        return np.array(
            [
                [
                    states[key].robots[0, ROBOT].q[0],
                    states[key].robots[0, ROBOT].v[0],
                    states[key].robots[0, ROBOT].contact_effort[0],
                    states[key].loads.inputs.rigid.force[0, 2],
                    states[key].time_s,
                ]
                for key in keys
            ]
        )

    def evaluate(self, before, after, actions, keys):
        self.evaluations += 1
        q = np.array([after[key].robots[0, ROBOT].q[0] for key in keys])
        v = np.array([after[key].robots[0, ROBOT].v[0] for key in keys])
        self.events.append(
            {"kind": "task-evaluate", "before": before, "after": after, "actions": actions}
        )
        terminated = np.array(
            [
                key in self.terminating_worlds and q[index] - SPECS[key]["q"] > 1e-5
                for index, key in enumerate(keys)
            ],
            dtype=bool,
        )
        return -(q**2 + 0.01 * v**2), terminated


def _env_step(env, backend, events, actions):
    event = {"kind": "environment-step", "before": backend.snapshot(), "actions": np.array(actions)}
    events.append(event)
    try:
        result = env.step(actions)
    except Exception as error:
        event.update(
            error={"type": type(error).__name__, "message": str(error)},
            accepted_worlds=getattr(error, "accepted_worlds", None),
            acceptance_known=getattr(error, "acceptance_known", None),
            after=backend.snapshot(),
        )
        raise
    event.update(result=result, after=backend.snapshot())
    return result


def test_controls_are_owned_and_only_real_scene_step_advances(tmp_path, monkeypatch):
    factory = _Factory(tmp_path, monkeypatch)
    with (
        _trace(tmp_path, "owned-controls", factory) as events,
        SceneBatchBackend((7, 19), factory) as backend,
    ):
        scene = factory.created[7][0]
        external = scene.inputs()
        external.rigid.force[0, 0] = 0.25
        actions = {ROBOT: RobotCommand([0], feedforward=[20.0])}
        packet = SceneControl(actions, external)
        actions[ROBOT] = RobotCommand([0], feedforward=[99.0])
        external.rigid.force[0, 0] = 99
        returned_external = packet.external
        returned_external.rigid.force[0, 0] = 100
        with pytest.raises(FrozenInstanceError):
            packet._external = None
        controls = _packets((7, 19))
        controls[7] = packet
        backend.set_controls(controls)
        controls[7] = SceneControl(actions)
        initial = backend.snapshot()
        events.append({"kind": "staged", "states": initial, "packet": packet})
        assert all(row.time_s == 0 and row.step_number == 0 for row in initial.values())
        assert isinstance(initial[7], RobotSceneObservation)
        with pytest.raises(ValueError):
            initial[7].robots[0, ROBOT].q.flags.writeable = True
        with pytest.raises(ValueError, match="exactly"):
            backend.set_controls({7: packet})
        result = backend.step(H)
        states = backend.snapshot()
        events.append({"kind": "accepted", "states": states, "accepted_worlds": result})
        assert dict(result) == {7: True, 19: True}
        assert all(type(value) is bool for value in result.values())
        assert states[7].robots[0, ROBOT].drive_effort[0] == 20
        assert states[7].external.rigid.force[0, 0] == 0.25
        assert all(row.time_s == H and row.step_number == 1 for row in states.values())


def test_real_loaded_worlds_feed_actions_observation_reward_and_selective_done(
    tmp_path, monkeypatch
):
    factory = _Factory(tmp_path, monkeypatch)
    with (
        _trace(tmp_path, "environment-loaded-step", factory) as events,
        SceneBatchBackend((7, 19), factory) as backend,
    ):
        task = _Task(events, terminating_worlds=(7,))
        env = ResidentVectorEnv(backend, task, (7, 19), action_size=1, dt=H, horizon=10)
        result = _env_step(env, backend, events, np.array([[20.0], [-0.5]]))
        expected_q, expected_v = [], []
        for key, effort in ((7, 20), (19, -0.5)):
            spec = SPECS[key]
            normal = K * spec["depth"]
            v = spec["v"] + H * ((effort + normal) / spec["mass"] - G)
            q = spec["q"] + H * v
            expected_q.append(q)
            expected_v.append(v)
            index = (7, 19).index(key)
            np.testing.assert_allclose(
                result.final_observation[index], [q, v, normal, -normal, H], atol=1e-10, rtol=0
            )
        np.testing.assert_allclose(
            result.reward,
            -(np.square(expected_q) + 0.01 * np.square(expected_v)),
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_array_equal(result.terminated, [True, False])
        np.testing.assert_array_equal(result.truncated, [False, False])
        np.testing.assert_array_equal(result.final_mask, [True, False])
        np.testing.assert_array_equal(result.episode_id, [1, 0])
        np.testing.assert_array_equal(result.episode_length, [0, 1])
        assert result.observation[0, 0] == SPECS[7]["q"] and result.observation[0, 4] == 0
        np.testing.assert_array_equal(result.observation[1], result.final_observation[1])
        assert factory.close_counts[7, 0] == 1 and factory.close_counts[19, 0] == 0


def test_selected_reset_preserves_other_physics_and_requires_new_controls(tmp_path, monkeypatch):
    factory = _Factory(tmp_path, monkeypatch)
    with (
        _trace(tmp_path, "selected-reset", factory) as events,
        SceneBatchBackend((7, 19), factory) as backend,
    ):
        for _ in range(2):
            _step(backend, events, _packets((7, 19), {7: 1, 19: -1}))
        before = backend.snapshot()
        backend.reset((7,))
        after = backend.snapshot()
        events.append({"kind": "selected-reset", "before": before, "after": after})
        assert after[7].time_s == 0 and after[7].step_number == 0
        np.testing.assert_array_equal(after[7].robots[0, ROBOT].q, [SPECS[7]["q"]])
        np.testing.assert_array_equal(after[7].robots[0, ROBOT].v, [SPECS[7]["v"]])
        np.testing.assert_array_equal(after[19].robots[0, ROBOT].q, before[19].robots[0, ROBOT].q)
        np.testing.assert_array_equal(after[19].robots[0, ROBOT].v, before[19].robots[0, ROBOT].v)
        assert after[19].time_s == before[19].time_s == 2 * H
        assert len(factory.created[7]) == 2 and len(factory.created[19]) == 1
        with pytest.raises(RuntimeError, match="controls"):
            backend.step(H)
        with pytest.raises(ValueError, match="unknown"):
            backend.snapshot((99,))
        with pytest.raises(ValueError, match="unique"):
            backend.reset((7, 7))
        assert backend.snapshot()[19].time_s == 2 * H


def test_partial_rejection_poison_has_complete_mask_without_reward_or_episode(
    tmp_path, monkeypatch
):
    factory = _Factory(tmp_path, monkeypatch)
    keys = (7, 19, 23)
    with (
        _trace(tmp_path, "partial-rejection", factory) as events,
        SceneBatchBackend(keys, factory) as backend,
    ):
        task = _Task(events)
        env = ResidentVectorEnv(backend, task, keys, action_size=1, dt=H, horizon=10)
        task.bad_world = 19
        with pytest.raises(VectorEnvStepError) as rejected:
            _env_step(env, backend, events, np.zeros((3, 1)))
        assert dict(rejected.value.accepted_worlds) == {7: True, 19: False, 23: False}
        assert (
            rejected.value.acceptance_known and rejected.value.physics_accepted and env.needs_reset
        )
        assert isinstance(rejected.value.__cause__, PhysicsStepFailure)
        assert isinstance(rejected.value.__cause__.__cause__, ValueError)
        assert task.evaluations == 0
        states = backend.snapshot()
        assert states[7].time_s == H and states[19].time_s == states[23].time_s == 0
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros((3, 1)))
        with pytest.raises(RuntimeError, match="all"):
            env.reset((0,))
        task.bad_world = None
        env.reset()
        result = _env_step(env, backend, events, np.zeros((3, 1)))
        np.testing.assert_array_equal(result.episode_id, [1, 1, 1])
        np.testing.assert_array_equal(result.episode_length, [1, 1, 1])
        assert task.evaluations == 1 and not env.needs_reset


def test_all_rejected_worlds_are_retryable_without_false_transition(tmp_path, monkeypatch):
    factory = _Factory(tmp_path, monkeypatch)
    keys = (7, 19)
    with (
        _trace(tmp_path, "first-world-rejection", factory) as events,
        SceneBatchBackend(keys, factory) as backend,
    ):
        task = _Task(events)
        env = ResidentVectorEnv(backend, task, keys, action_size=1, dt=H, horizon=10)
        task.bad_world = 7
        with pytest.raises(VectorEnvStepError) as rejected:
            _env_step(env, backend, events, np.zeros((2, 1)))
        assert dict(rejected.value.accepted_worlds) == {7: False, 19: False}
        assert (
            rejected.value.acceptance_known
            and not rejected.value.physics_accepted
            and not env.needs_reset
        )
        assert task.evaluations == 0 and all(row.time_s == 0 for row in backend.snapshot().values())
        task.bad_world = None
        result = _env_step(env, backend, events, np.zeros((2, 1)))
        np.testing.assert_array_equal(result.episode_id, [0, 0])
        np.testing.assert_array_equal(result.episode_length, [1, 1])


def test_scene_error_after_publication_retains_that_world_as_accepted(tmp_path, monkeypatch):
    factory = _Factory(tmp_path, monkeypatch)
    with (
        _trace(tmp_path, "post-publication-error", factory) as events,
        SceneBatchBackend((7, 19), factory) as backend,
    ):
        scene = factory.created[7][0]
        original = scene.step
        failure = RuntimeError("injected caller error after actual public scene step")

        def fail_after_step(*args, **kwargs):
            original(*args, **kwargs)
            raise failure

        monkeypatch.setattr(scene, "step", fail_after_step)
        with pytest.raises(PhysicsStepFailure) as rejected:
            _step(backend, events, _packets((7, 19)))
        assert rejected.value.__cause__ is failure
        assert dict(rejected.value.accepted_worlds) == {7: True, 19: False}
        assert rejected.value.acceptance_known
        assert backend.snapshot()[7].time_s == H and backend.snapshot()[19].time_s == 0


def test_factory_failure_and_alias_cannot_destroy_prior_live_worlds(tmp_path, monkeypatch):
    factory = _Factory(tmp_path, monkeypatch)
    with (
        _trace(tmp_path, "factory-reset-failure", factory) as events,
        SceneBatchBackend((7, 19), factory) as backend,
    ):
        _step(backend, events, _packets((7, 19)))
        before = backend.snapshot()
        factory.fail_world = 19
        with pytest.raises(RuntimeError, match="factory construction"):
            backend.reset((7, 19))
        after = backend.snapshot()
        events.append({"kind": "failed-reset", "before": before, "after": after})
        assert factory.close_counts[7, 1] == 1
        assert factory.close_counts[7, 0] == factory.close_counts[19, 0] == 0
        for key in (7, 19):
            np.testing.assert_array_equal(
                after[key].robots[0, ROBOT].q, before[key].robots[0, ROBOT].q
            )
            np.testing.assert_array_equal(
                after[key].robots[0, ROBOT].v, before[key].robots[0, ROBOT].v
            )
            assert after[key].time_s == H
        factory.fail_world = None
        factory.alias_world = 19
        with pytest.raises(ValueError, match="fresh independent"):
            backend.reset((19,))
        assert backend.snapshot()[19].time_s == H and factory.close_counts[19, 0] == 0
        factory.alias_world = None
        backend.reset((7, 19))
        assert all(row.time_s == 0 for row in backend.snapshot().values())
        assert factory.close_counts[7, 0] == factory.close_counts[19, 0] == 1


def test_constructor_failure_cleans_created_scenes_and_close_owns_current_scenes(
    tmp_path, monkeypatch
):
    factory = _Factory(tmp_path, monkeypatch)
    with _trace(tmp_path, "construction-close-ownership", factory) as events:
        factory.fail_world = 19
        with pytest.raises(RuntimeError, match="factory construction"):
            SceneBatchBackend((7, 19), factory)
        assert factory.close_counts[7, 0] == 1
        factory.fail_world = None
        backend = SceneBatchBackend((7, 19), factory)
        _step(backend, events, _packets((7, 19)))
        backend.close()
        backend.close()
        assert factory.close_counts[7, 1] == factory.close_counts[19, 0] == 1
        with pytest.raises(RuntimeError, match="closed"):
            backend.snapshot()
        for key in (7, 19):
            with pytest.raises(RuntimeError, match="closed"):
                factory.created[key][-1].observe()
