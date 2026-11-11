"""HOST-only reference arithmetic and failure records; no CUDA physical rerun."""

import importlib.util
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxStepReceipt
from phymeo.backends.mujoco_device_controls import MuJoCoTyxDeviceControlReceipt
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoState
from phymeo.backends.mujoco_task_workspace import MuJoCoTyxTaskFrame
from phymeo.training.mujoco_tyx_env import MuJoCoTyxTransition


@pytest.fixture(scope="module")
def api():
    path = Path(__file__).resolve().parents[2] / "examples/diagnose_mujoco_tyx_reachability.py"
    spec = importlib.util.spec_from_file_location("n4e_public_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(api):
    rows = api.learning_api().tasks().templates("J", held_out=True)
    q = np.asarray([x[0] for x in rows], np.float32)
    v = np.asarray([x[1] for x in rows], np.float32)
    return q, v


def test_original_protocol_and_fixed_window(api):
    protocol = api.protocol_binding()
    assert protocol == api.learning_api().protocol_binding("J")
    assert protocol["xml_sha256"] == api.XML_SHA256
    assert protocol["seeds"] == [14011, 14012, 14013]
    assert api.WORLD_KEYS == tuple(5 + 36 * i for i in range(8))
    assert (api.STEPS, api.SWITCH, api.H) == (256, 128, 0.002)
    assert (api.Q_ATOL, api.V_ATOL) == (2e-4, 3e-3)


def test_schedule_terminal_and_strict_force_bounds(api):
    q, v = inputs(api)
    schedule = api.make_schedule(q, v)
    assert schedule["raw_action"].shape == (256, 8, 2)
    assert schedule["raw_action"].dtype == np.float32
    assert np.max(np.abs(schedule["force"])) < 2
    assert np.array_equal(
        schedule["force"][:128], np.broadcast_to(schedule["force"][0], (128, 8, 2))
    )
    assert np.array_equal(
        schedule["force"][128:], np.broadcast_to(schedule["force"][128], (128, 8, 2))
    )
    # Independent scalar Euler loop, not the example's A/B or advance helper.
    x, velocity = q.astype(np.float64), v.astype(np.float64)
    h = float(np.float32(0.002))
    for force in schedule["force"]:
        velocity = velocity + h * (force - 0.1 * velocity) / (1.01 + h * 0.1)
        x = x + h * velocity
    np.testing.assert_allclose(x, 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(velocity, 0, atol=1e-12, rtol=0)
    assert not schedule["force"].flags.writeable
    assert not schedule["raw_action"].flags.writeable
    q[:] = 0
    assert np.any(schedule["initial_q"] != 0)


def test_zero_and_sign_symmetry(api):
    zero = api.make_schedule(np.zeros((8, 2), np.float32), np.zeros((8, 2), np.float32))
    assert not zero["force"].any()
    q, v = inputs(api)
    positive, negative = api.make_schedule(q, v), api.make_schedule(-q, -v)
    np.testing.assert_array_equal(negative["force"], -positive["force"])
    np.testing.assert_array_equal(negative["raw_action"], -positive["raw_action"])


def test_inverse_tanh_mapping_is_not_linear_raw(api):
    schedule = api.make_schedule(*inputs(api))
    recovered = 2 * np.tanh(schedule["raw_action"].astype(np.float64))
    np.testing.assert_allclose(recovered, schedule["force"], atol=2e-7, rtol=0)
    assert np.max(np.abs(schedule["raw_action"] - schedule["force"] / 2)) > 0.01
    assert np.max(np.abs(schedule["mapped_reference_q"][-1])) < api.Q_ATOL
    assert np.max(np.abs(schedule["mapped_reference_v"][-1])) < api.V_ATOL


@pytest.mark.parametrize("fault", ("shape", "nonfinite", "force", "bounds", "not-fp32"))
def test_infeasible_schedule_never_clips_or_changes_switch(api, fault):
    q, v = inputs(api)
    if fault == "shape":
        q = q[:7]
    elif fault == "nonfinite":
        v[0, 0] = np.inf
    elif fault == "force":
        q[0, 0] = 0.9
    elif fault == "not-fp32":
        q = q.astype(np.float64)
        q[0, 0] = 0.04
    else:
        q[0, 0] = 1.1
    with pytest.raises(api.ReferenceInfeasible):
        api.make_schedule(q, v)


def test_history_read_is_hash_bound_and_bounded(api, tmp_path):
    path = tmp_path / "summary.json"
    raw = b'{"value": 1}'
    path.write_bytes(raw)
    value, digest = api.read_metadata(path, sha256(raw).hexdigest())
    assert value == {"value": 1} and digest == sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="hash"):
        api.read_metadata(path, "0" * 64)
    path.write_bytes(b'{"value": 1, "value": 2}')
    with pytest.raises(ValueError, match="duplicate"):
        api.read_metadata(path)
    path.write_bytes(b" " * (api.MAX_METADATA_BYTES + 1))
    with pytest.raises(ValueError, match="bounded"):
        api.read_metadata(path)


class Writer:
    def __init__(self, fail_kind=None):
        self.records, self.fail_kind = [], fail_kind

    def write(self, kind, value):
        if kind == self.fail_kind:
            self.fail_kind = None
            raise OSError("injected evidence write failure")
        self.records.append((kind, value))


class FakeEnv:
    """A scalar independent HOST oracle with actual public publication dataclasses."""

    def __init__(self, api, *, fail_step=None, fault=None):
        self.api, self.fail_step, self.fault = api, fail_step, fault
        self.q, self.v = inputs(api)
        self.initial_q, self.initial_v = self.q.copy(), self.v.copy()
        self.world_keys, self.num_envs = api.WORLD_KEYS, 8
        self.h, self.num_actions, self.device = 0.002, 2, "synthetic-HOST-only"
        self.current_frame = self.frame(0)
        self.episode_length = np.zeros(8, np.int64)
        self.episode_id = np.zeros(8, np.int64)
        self.calls = 0

    def frame(self, tick):
        return MuJoCoTyxTaskFrame(
            {"test": "owned"}, tick * 2, tick, (tick,) * 8, (tick * float(np.float32(0.002)),) * 8
        )

    def step(self, raw, **kwargs):
        assert kwargs["world_keys"] == self.world_keys
        assert kwargs["expected_frame"] is self.current_frame
        self.calls += 1
        if self.calls == self.fail_step:
            raise RuntimeError("injected second-step physical refusal")
        tick, before = self.calls, self.current_frame
        q0, v0 = self.q.copy(), self.v.copy()
        ctrl = (np.float32(2) * np.tanh(raw)).astype(np.float32)
        h = float(np.float32(0.002))
        # Deliberately distinct from production reference implementation.
        dv = h * (ctrl.astype(np.float64) - 0.1 * v0) / (1.01 + h * 0.1)
        self.v = (v0 + dv).astype(np.float32)
        self.q = (q0 + h * self.v).astype(np.float32)
        qfrc, xfrc = np.zeros((8, 2), np.float32), np.zeros((8, 3, 6), np.float32)
        states, controls = {}, {}
        for row, key in enumerate(self.world_keys):
            control = MuJoCoControl(ctrl[row], qfrc_applied=qfrc[row], xfrc_applied=xfrc[row])
            controls[key] = control
            receipt = MuJoCoTyxStepReceipt(
                world=key,
                tick=tick,
                input_time=before.times[row],
                output_time=tick * h,
                h=0.002,
                control=control,
                stored_control=control,
                contacts=(),
                input_q=q0[row].astype(float),
                input_v=v0[row].astype(float),
                qacc=(ctrl[row] - 0.1 * v0[row]) / 1.01,
                integration_acceleration=dv[row] / h,
                qfrc_actuator=ctrl[row].astype(float),
                qfrc_constraint=np.zeros(2),
                raw_contact_fields={},
                source_solver="Newton",
                numerical_solver="single-normal-direct-qp",
                contact_time=before.times[row],
                backend="mujoco-tyx-cuda",
            )
            states[key] = MuJoCoState(
                world=key,
                q=self.q[row].astype(float),
                v=self.v[row].astype(float),
                act=np.zeros(0),
                ctrl=ctrl[row].astype(float),
                time=tick * h,
                tick=tick,
                body_positions=np.zeros((3, 3)),
                body_quaternions=np.zeros((3, 4)),
                geometry_positions=np.zeros((2, 3)),
                geometry_matrices=np.zeros((2, 3, 3)),
                qacc=receipt.qacc,
                qfrc_actuator=receipt.qfrc_actuator,
                qfrc_passive=-0.1 * self.v[row],
                qfrc_constraint=np.zeros(2),
                qfrc_applied=qfrc[row],
                xfrc_applied=xfrc[row],
                sensordata=np.zeros(0),
                energy=np.zeros(2),
                contacts=(),
                contact_time=tick * h,
                held_control=control,
                last_step=receipt,
            )
        cost = np.mean((self.q.astype(float) / 0.2) ** 2, axis=1) + 0.01 * np.mean(
            self.v.astype(float) ** 2, axis=1
        )
        timeout = np.full(8, tick == 256)
        final_obs = np.concatenate((self.q * 5, self.v), axis=1)
        if tick == 256:
            self.q, self.v = self.initial_q.copy(), self.initial_v.copy()
            self.episode_length[:], self.episode_id[:] = 0, 1
            self.current_frame = self.frame(0)
        else:
            self.episode_length[:] = tick
            self.current_frame = self.frame(tick)
        action_receipt = MuJoCoTyxDeviceControlReceipt(
            {}, before.revision, before.revision + 1, tick, controls
        )
        value = MuJoCoTyxTransition(
            tick,
            self.world_keys,
            before,
            self.current_frame,
            np.concatenate((self.q * 5, self.v), axis=1),
            -cost,
            np.zeros(8, bool),
            timeout,
            final_obs,
            timeout.copy(),
            np.zeros(8, np.int64),
            np.full(8, tick),
            -cost,
            raw.copy(),
            (ctrl, qfrc, xfrc),
            action_receipt,
            dict.fromkeys(self.world_keys, True),
            states,
        )
        fault_tick = 256 if self.fault == "final-timeout" else 2
        if tick == fault_tick:
            key = self.world_keys[0]
            if self.fault == "wrong-world":
                states[key] = replace(states[key], world=99)
            elif self.fault == "stale-tick":
                states[key] = replace(
                    states[key], last_step=replace(states[key].last_step, tick=tick - 1)
                )
            elif self.fault == "stored-control":
                states[key] = replace(
                    states[key],
                    last_step=replace(
                        states[key].last_step,
                        stored_control=MuJoCoControl(
                            ctrl[0] + 1, qfrc_applied=qfrc[0], xfrc_applied=xfrc[0]
                        ),
                    ),
                )
            elif self.fault == "requested-action":
                value = replace(value, action=raw + 1)
            elif self.fault == "early-timeout":
                value = replace(value, time_outs=np.ones(8, bool))
            elif self.fault == "final-timeout":
                value = replace(value, time_outs=np.zeros(8, bool))
            elif self.fault == "nonzero-qfrc":
                qfrc[0, 0] = 1
            elif self.fault == "prediction":
                states[key] = replace(states[key], q=states[key].q + 0.1)
            elif self.fault == "reward":
                value = replace(value, reward=-cost + 1)
        return value


def submit(env, raw):
    return env.step(raw, world_keys=env.world_keys, expected_frame=env.current_frame)


def test_complete_window_keeps_actual_last_pre_reset_state(api):
    env, writer = FakeEnv(api), Writer()
    result = api.run_window(env, api.make_schedule(env.q, env.v), writer, submit=submit)
    assert env.calls == 256 and result["world_steps"] == 2048
    assert result["classification"] == "PASS_FIXED_REFERENCE_DIAGNOSTIC"
    assert result["cost"] > 0 and len(result["per_world_cost"]) == 8
    assert np.max(np.abs(result["terminal_q"])) < api.Q_ATOL
    assert np.array_equal(env.q, env.initial_q) and np.any(env.q != result["terminal_q"])
    transitions = [v for k, v in writer.records if k == "transition"]
    assert len(transitions) == 256
    assert np.all(transitions[-1].time_outs)
    np.testing.assert_array_equal(transitions[-1].states[5].q, result["terminal_q"][0])
    # Independent recurrence consumed receipt controls, not ideal schedule forces.
    np.testing.assert_allclose(
        result["terminal_v"], result["reference_terminal_v"], atol=api.V_ATOL, rtol=0
    )


def test_second_step_refusal_preserves_complete_prefix_and_request(api):
    env, writer = FakeEnv(api, fail_step=2), Writer()
    with pytest.raises(api.WindowFailure) as caught:
        api.run_window(env, api.make_schedule(env.q, env.v), writer, submit=submit)
    assert caught.value.detail["cost"] is None
    assert caught.value.detail["transitions_saved"] == 1
    assert caught.value.detail["validated_steps"] == 1
    assert [k for k, _ in writer.records].count("requested_action") == 2
    assert [k for k, _ in writer.records].count("transition") == 1


@pytest.mark.parametrize(
    "fault",
    (
        "wrong-world",
        "stale-tick",
        "stored-control",
        "requested-action",
        "early-timeout",
        "final-timeout",
        "nonzero-qfrc",
        "prediction",
        "reward",
    ),
)
def test_wrong_publication_or_failed_gate_never_produces_short_cost(api, fault):
    env, writer = FakeEnv(api, fault=fault), Writer()
    with pytest.raises(api.WindowFailure) as caught:
        api.run_window(env, api.make_schedule(env.q, env.v), writer, submit=submit)
    detail = caught.value.detail
    assert detail["cost"] is None
    assert detail["transitions_saved"] == (256 if fault == "final-timeout" else 2)
    assert detail["validated_steps"] == detail["transitions_saved"] - 1


def test_post_step_evidence_failure_keeps_transition_and_no_cost(api):
    env, writer = FakeEnv(api), Writer("independent")
    with pytest.raises(api.WindowFailure) as caught:
        api.run_window(env, api.make_schedule(env.q, env.v), writer, submit=submit)
    assert caught.value.detail["cost"] is None
    assert caught.value.detail["transitions_saved"] == 1
    assert [k for k, _ in writer.records].count("transition") == 1


def synthetic_history(api, root, monkeypatch):
    protocol = api.protocol_binding()
    seeds = [14011, 14012, 14013]
    original_binding = {"anchor_sha256": "a" * 64, "source_bindings": {}}
    zero = {"status": "COMPLETE_DIAGNOSTIC", "world_steps": 2048, "cost": 0.5}
    originals, rows, bindings = {}, [], {}

    def write(path, value):
        raw = json.dumps(value, sort_keys=True).encode()
        path.write_bytes(raw)
        return sha256(raw).hexdigest()

    for seed in seeds:
        directory = root / ("J-" + str(seed))
        directory.mkdir()
        original = {
            "protocol_binding": protocol,
            "baseline": {"cost": 1.2},
            "learned": {"cost": 1.1},
            "source_files": {"original.py": "b" * 64},
            "dependencies": {"mujoco": "3.3.7"},
            "zero_action_diagnostic": {"cost": 0.6},
        }
        originals[seed] = original
        saved = {
            "task": "J",
            "seed": seed,
            "status": "COMPLETE_N4A_SEED",
            "updates": 128,
            "worlds": 8,
            "accepted_world_steps": 32768,
            "evidence": {"logical_sha256": "c" * 64},
            "protocol_binding": protocol,
            "ppo_config": {"fixed": True},
            "source_files": {"variant.py": "d" * 64},
            "dependencies": original["dependencies"],
            "baseline": {"cost": 1.0, "world_steps": 2048},
            "learned": {"cost": 1.0, "world_steps": 2048},
            "zero_action_diagnostic": zero,
        }
        digest = write(directory / "summary.json", saved)
        write(directory / "publication.json", {"summary_sha256": digest})
        write(
            directory / "manifest.json",
            {
                name: saved[name]
                for name in (
                    "task",
                    "seed",
                    "protocol_binding",
                    "ppo_config",
                    "source_files",
                    "dependencies",
                )
            },
        )
        bindings[str(seed)] = {
            "directory": str(directory),
            "summary_sha256": digest,
            "evidence": saved["evidence"],
        }
        rows.append(
            {
                "seed": seed,
                "directory": str(directory),
                "status": "COMPLETE",
                "baseline_cost": 1.0,
                "final_cost": 1.0,
                "original_baseline_cost": 1.2,
                "original_final_cost": 1.1,
            }
        )
    variant = {
        "schema": "n4d-j-ablation-comparison-v1",
        "classification": "FIXED_BUDGET_NOT_LEARNED",
        "original_classification": "FIXED_BUDGET_NOT_LEARNED",
        "original_bindings": original_binding,
        "rows": rows,
        "variant_three_seed": {
            "task": "J",
            "seeds": seeds,
            "protocol_binding": protocol,
            "outcome": {"classification": "FIXED_BUDGET_NOT_LEARNED"},
            "source_bindings": bindings,
            "zero_action_diagnostic": zero,
        },
    }
    monkeypatch.setattr(api, "N4D_SHA256", write(root / "summary.json", variant))
    original_loader = api._example
    monkeypatch.setattr(
        api,
        "_example",
        lambda name: (
            SimpleNamespace(load_originals=lambda path: (original_binding, originals))
            if name == "evaluate_mujoco_tyx_initialization_ablation.py"
            else original_loader(name)
        ),
    )
    return protocol


def test_history_keeps_distinct_source_and_zero_comparisons(api, tmp_path, monkeypatch):
    protocol = synthetic_history(api, tmp_path, monkeypatch)
    result = api.load_history(tmp_path, tmp_path, protocol)
    assert len(result["rows"]) == 3
    assert result["original_zero_cost"] == 0.6 and result["centered_zero_cost"] == 0.5
    assert result["rows"][0]["original_source_files"] != result["rows"][0]["variant_source_files"]


@pytest.mark.parametrize("fault", ("manifest", "publication", "protocol"))
def test_history_binding_failure_prevents_a_window(api, tmp_path, monkeypatch, fault):
    protocol = synthetic_history(api, tmp_path, monkeypatch)
    if fault == "protocol":
        protocol = {**protocol, "h": 0.001}
    else:
        path = tmp_path / "J-14012" / (fault + ".json")
        value = json.loads(path.read_text())
        if fault == "manifest":
            value["source_files"] = {"changed.py": "e" * 64}
        else:
            value["summary_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        api.load_history(tmp_path, tmp_path, protocol)
