"""HOST-only collector failure orchestration, not a physical/GPU oracle.

All computation owners and tensors are fakes. Only the real collector control
flow and its temporary JSON/gzip files are exercised. Its production 'cuda'
metadata is not evidence of a physical CUDA step in these tests.
"""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class Record(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class HostTensor:
    def __init__(self, value):
        self.value = np.array(value, copy=True)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value.copy()

    def any(self):
        return np.asarray(self.value.any())


def forbidden(*args, **kwargs):
    raise AssertionError("HOST orchestration test must not invoke physical owners")


def read_lines(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def load_runner():
    pytest.importorskip("mujoco")
    pytest.importorskip("torch")
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_manipulation_control.py"
    spec = importlib.util.spec_from_file_location("feedback_failure_collector_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("summary_raises", (False, True), ids=("callback-only", "callback-and-summary"))
def test_accepted_callback_failure_keeps_current_entry_and_original_error(
    tmp_path, monkeypatch, summary_raises,
):
    runner = load_runner()
    world = 5
    initial_q = np.array((.25, -.5), dtype=np.float32)
    endpoint_q = np.array((1.25, 2.5), dtype=np.float32)
    initial_v = np.zeros(2, dtype=np.float32)
    endpoint_v = np.array((.125, -.25), dtype=np.float32)
    current_entry = {
        "entry_q": initial_q[None, :],
        "entry_v": initial_v[None, :],
        "entry_act": np.zeros((1, 0), dtype=np.float32),
        "entry_clock": np.array(((0, 0, runner.H, 0),), dtype=np.float32),
        "entry_meta": np.array(((1, 0, 1, 1, 1, 0),), dtype=np.int32),
        "entry_contact_ids": np.array((((1, 1, 1, 2, 1, 2, 0, 3),),), dtype=np.int32),
        "entry_contact_data": np.zeros((1, 1, 20), dtype=np.float32),
        "entry_body_mask": np.ones((1, 3), dtype=np.int32),
        "entry_body_position": np.array((((0, 0, 0), (.1, .2, .3), (.4, .5, .6)),), dtype=np.float32),
        "entry_body_rotation": np.tile(np.eye(3, dtype=np.float32), (1, 3, 1, 1)),
        "entry_body_linear_velocity": np.zeros((1, 3, 3), dtype=np.float32),
        "entry_body_angular_velocity": np.zeros((1, 3, 3), dtype=np.float32),
        "entry_contact_wrench_world": np.array((((2, .5, 0, 0, 0, 0),),), dtype=np.float32),
        "entry_contact_body_wrench": np.array(((((-2, -.5, 0, 0, 0, 0), (2, .5, 0, 0, 0, 0)),),), dtype=np.float32),
    }
    current_entry["entry_contact_data"][0, 0, 0] = -.001
    current_entry["entry_contact_data"][0, 0, 1:4] = (.1, .2, .3)
    current_entry["entry_contact_data"][0, 0, 4:13] = np.eye(3).reshape(-1)
    current_entry["entry_contact_data"][0, 0, 13:19] = (2, .5, 0, 0, 0, 0)
    empty_entry = {name: np.zeros_like(value) for name, value in current_entry.items()}
    expected_entry = {name: value.tolist() for name, value in current_entry.items()}
    calls = dict(step=0, accepted=0, summary=0, reference=0, command=0, batch_exit=0, env_exit=0)
    captured = {}
    callback_error = ValueError("injected accepted callback failure")
    summary_error = LookupError("injected secondary summary failure")
    model = SimpleNamespace(nq=2, nv=2, na=0, nu=1)
    goal = SimpleNamespace(kind="pusht", instructions_en=("HOST failure fixture only",))
    scene = SimpleNamespace(
        key="host-failure-fixture", robot_kinds="R", mjcf='<mujoco model="host-fake"/>',
        goals=(goal,), default_goal=0, finger_geom_names=(),
    )
    compilation = SimpleNamespace(fresh_native_model=lambda: model)

    def compile_fake(xml, *, require_asset):
        assert xml == scene.mjcf and require_asset is False
        return compilation

    def observation(accepted):
        arrays = dict(current_entry if accepted else empty_entry)
        arrays.update(
            q=(endpoint_q if accepted else initial_q)[None, :],
            v=(endpoint_v if accepted else initial_v)[None, :],
            clock=np.array(((runner.H if accepted else 0, 0),), dtype=np.float32),
        )
        return {name: HostTensor(value) for name, value in arrays.items()}

    class FakeBatch:
        def __init__(self, specs, rows, *, mode):
            assert specs == ((scene.key, compilation),) and rows == ((world, scene.key),)
            assert mode == "cuda"  # Production call is intercepted; no backend is created.
            self.tick = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls["batch_exit"] += 1
            return False

        def integration_snapshot(self):
            return {world: SimpleNamespace(
                q=(endpoint_q if self.tick else initial_q).copy(),
                v=(endpoint_v if self.tick else initial_v).copy(),
                act=np.empty(0, dtype=np.float32),
                time=float(np.float32(self.tick * runner.H)), tick=self.tick,
            )}

    class FakeEnv:
        def __init__(self, batch, scenes, *, goals, action_specs, horizons, auto_reset):
            assert scenes == {world: scene} and goals == {world: goal}
            assert set(action_specs) == {world} and horizons == {world: 4}
            assert auto_reset is False
            self.batch = batch
            self.world_keys, self.num_actions, self.device = (world,), 1, "host-only"
            self.current_frame = object()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls["env_exit"] += 1
            return False

        @property
        def observation(self):
            return observation(bool(self.batch.tick))

        def checkpoint(self):
            forbidden()

        def step(self, action, *, world_keys, expected_frame):
            calls["step"] += 1
            assert calls["step"] == 1 and self.batch.tick == 0
            assert world_keys == (world,) and expected_frame is self.current_frame
            captured["action"] = action.numpy()
            self.batch.tick = 1  # Synthetic owner publication, not a numerical integration.
            frame = Record(
                world=world, episode=0, tick=0, time_s=0., phase="accepted-entry",
                object_key="object", finger_normal_forces_n=(0., 0.), support_up_force_n=0.,
            )
            result = Record(
                success=False, metrics=dict(clearance_m=0., speed_m_s=0.),
                history=Record(lifted=False, transported=False),
            )
            captured["result"] = result
            return SimpleNamespace(
                observation=observation(True), action=action,
                native_channels={"ctrl": HostTensor(((2.,),))},
                action_receipt={"host_fake": True}, accepted_worlds={world: True},
                success_frames={world: frame}, results={world: result},
                final_mask=HostTensor((False,)),
            )

    class FakeController:
        def command(self, q, v, desired, velocity):
            calls["command"] += 1
            np.testing.assert_array_equal(q, initial_q)
            np.testing.assert_array_equal(v, initial_v)
            return np.array((2.,), dtype=np.float32), {"host_fake": True}

    def plan_builder(actual_scene, actual_model):
        assert actual_scene is scene and actual_model is model
        return FakeController(), (), {"valid": True, "host_fake": True}

    class FakeDriver:
        stopped, reason = False, "continue"

        def reference(self, tick, q, v):
            calls["reference"] += 1
            assert tick == 0
            np.testing.assert_array_equal(q, initial_q)
            np.testing.assert_array_equal(v, initial_v)
            return np.zeros(1), np.zeros(1), "host-synthetic", {"host_fake": True}

        def accepted(self, tick, q, entry, result):
            calls["accepted"] += 1
            assert tick == 1 and result is captured["result"]
            np.testing.assert_array_equal(q, endpoint_q)
            assert set(entry) == set(current_entry)
            for name, expected in current_entry.items():
                np.testing.assert_array_equal(entry[name], expected)
            raise callback_error

        def summary(self):
            calls["summary"] += 1
            if summary_raises:
                raise summary_error
            return {"status": "host-callback-failed", "tick": 1}

    def driver_factory(actual_scene, actual_model, output):
        assert actual_scene is scene and actual_model is model
        output.mkdir()
        return FakeDriver()

    def host_tensor(value, *, device):
        assert device == "host-only"
        return HostTensor(value)

    monkeypatch.setattr(runner, "torch", SimpleNamespace(Tensor=HostTensor, tensor=host_tensor))
    monkeypatch.setattr(runner, "mj", SimpleNamespace(mj_step=forbidden, mj_step1=forbidden, mj_step2=forbidden))
    monkeypatch.setattr(runner, "manipulation_scenes", lambda: (scene,) * 8)
    monkeypatch.setattr(runner, "compile_mujoco_xml", compile_fake)
    monkeypatch.setattr(runner, "MuJoCoTyxModelSpec", lambda key, compiled: (key, compiled))
    monkeypatch.setattr(runner, "MuJoCoTyxWorldSpec", lambda key, model_key: (key, model_key))
    monkeypatch.setattr(runner, "MuJoCoTyxActionSpec", lambda *args, **kwargs: {"host_fake": True})
    monkeypatch.setattr(runner, "MuJoCoTyxHeterogeneousBatch", FakeBatch)
    monkeypatch.setattr(runner, "HeterogeneousManipulationEnv", FakeEnv)
    monkeypatch.setattr(runner, "NativeJointPD", forbidden)
    monkeypatch.setattr(runner, "solve_native_point_ik", forbidden)
    output = tmp_path.resolve() / "failed-host-prefix"
    with pytest.raises(RuntimeError, match="reference feedback failed after accepted") as caught:
        runner.run(
            output, 3, (world,), plan_builder=plan_builder, record_entry=True,
            reference_driver_factory=driver_factory,
        )
    error = caught.value
    assert error.__cause__ is callback_error
    assert error.phase == "reference-feedback-accepted"
    assert error.action_accepted is True and error.state_accepted is True
    assert calls == dict(step=1, accepted=1, summary=1, reference=1, command=1, batch_exit=1, env_exit=1)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED_PREFIX"
    assert manifest["steps"] == manifest["recorded_steps"] == 1
    assert manifest["requested_steps"] == 3 and manifest["saved_frames"] == 2
    assert manifest["failure"] == dict(
        type="RuntimeError", message=str(error), phase="reference-feedback-accepted",
        action_accepted=True, state_accepted=True,
    )
    assert manifest["record_entry"] is True and manifest["reference_feedback"] is True
    assert manifest["worlds"][0]["world"] == world
    assert manifest["worlds"][0]["task_success"] is False
    if summary_raises:
        assert manifest["driver_summary_failure"] == dict(type="LookupError", message=str(summary_error))
        assert not (output / "reference-feedback-summary.json").exists()
    else:
        assert "driver_summary_failure" not in manifest
        summary = json.loads((output / "reference-feedback-summary.json").read_text(encoding="utf-8"))
        assert summary == {str(world): {"status": "host-callback-failed", "tick": 1}}

    (trace,) = read_lines(output / "trace.jsonl.gz")
    assert trace["step"] == 1 and trace["accepted"] == {str(world): True}
    assert trace["entry"] == expected_entry
    assert trace["entry"]["entry_q"] == [initial_q.tolist()]
    assert trace["q"] == [endpoint_q.tolist()]
    assert trace["feedback_source"]["entry_meta"] == [[0, 0, 0, 0, 0, 0]]
    assert trace["contact_ids"] == expected_entry["entry_contact_ids"]
    assert trace["contact_data"] == expected_entry["entry_contact_data"]
    assert trace["contact_world_wrench"] == expected_entry["entry_contact_wrench_world"]
    assert trace["contact_body_wrench"] == expected_entry["entry_contact_body_wrench"]
    assert trace["raw_action"] == captured["action"].tolist()
    assert trace["native_channels"] == {"ctrl": [[2.]]}
    assert trace["action_receipt"] == {"host_fake": True}
    assert trace["success_frames"][str(world)]["phase"] == "accepted-entry"
    assert trace["results"][str(world)]["success"] is False
    assert trace["reference_feedback"][str(world)] == dict(
        host_fake=True, callback_error=dict(type="ValueError", message=str(callback_error)),
    )
    states = read_lines(output / "states.jsonl.gz")
    assert [state["step"] for state in states] == [0, 1]
    assert states[0]["worlds"][0]["q"] == initial_q.tolist()
    assert states[1]["worlds"][0]["q"] == endpoint_q.tolist()
    assert states[1]["worlds"][0]["tick"] == 1
    assert not (output / "final.checkpoint").exists()
    for field, filename in (("trace_sha256", "trace.jsonl.gz"), ("state_sha256", "states.jsonl.gz")):
        assert manifest[field] == hashlib.sha256((output / filename).read_bytes()).hexdigest()
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["status"] == "FAILED_PREFIX"
