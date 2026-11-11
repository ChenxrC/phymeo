"""N6 bounded control windows; no policy/physics fixture tuning."""

import importlib.util
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
KEYS = (7, 31, 103, 4096)


def api():
    spec = importlib.util.spec_from_file_location(
        "n6_control_window_example", ROOT / "examples/mujoco_tyx_control_window.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records():
    from phymeo.training.mujoco_tyx_control_window import (
        MuJoCoTyxControlWindowRecord,
        MuJoCoTyxControlWindowSpec,
        MuJoCoTyxSubstepRecord,
    )

    return MuJoCoTyxControlWindowSpec, MuJoCoTyxSubstepRecord, MuJoCoTyxControlWindowRecord


def test_spec_is_immutable_and_complete_binding():
    Spec, _, _ = records()
    a, b = Spec(), Spec(4)
    assert a.physics_steps == 1 and b.physics_steps == 4
    assert a.sha256 != b.sha256 and b.sha256 == Spec(4).sha256
    binding = b.binding(0.002)
    assert binding["physics_dt_s"] == 0.002 and binding["policy_dt_s"] == 0.008
    assert binding["reward"] == binding["done"] == "endpoint"
    assert binding["horizon_units"] == "policy_steps"
    assert binding["action_hold"] == "native_channels"
    assert binding != b.binding(0.001)
    binding["physics_steps"] = 99
    assert b.binding(0.002)["physics_steps"] == 4
    with pytest.raises(FrozenInstanceError):
        b.physics_steps = 2


@pytest.mark.parametrize("value", (0, -1, 9, True, 2.0, "2", None))
def test_spec_rejects_unsupported_decimation(value):
    with pytest.raises((TypeError, ValueError)):
        records()[0](value)


@pytest.mark.parametrize("value", (0, -0.001, float("nan"), float("inf"), True, "0.002"))
def test_binding_rejects_bad_timestep(value):
    with pytest.raises((TypeError, ValueError)):
        records()[0](2).binding(value)


@dataclass(frozen=True)
class DetachedState:
    tick: int
    q: np.ndarray
    last_step: object


def test_record_owns_nested_complete_states_and_mappings():
    Spec, Substep, Window = records()
    array = np.array([1.0, 2.0])
    source = {7: DetachedState(1, array, {"input_q": array, "contacts": [array]})}
    accepted = {7: True}
    sub = Substep(0, 1, 2, accepted, source, True)
    window = Window(Spec().sha256, 1, 1, 1, [sub], {"ctrl": array}, "complete")
    array[:] = 99
    accepted[7] = False
    source.clear()
    assert window.substeps[0].accepted_worlds[7]
    np.testing.assert_array_equal(sub.states[7].q, [1, 2])
    np.testing.assert_array_equal(sub.states[7].last_step["contacts"][0], [1, 2])
    assert isinstance(sub.states, Mapping) and isinstance(window.substeps, tuple)
    with pytest.raises(TypeError):
        sub.states[31] = sub.states[7]
    with pytest.raises(ValueError):
        sub.states[7].q.setflags(write=True)
    with pytest.raises(ValueError):
        window.action_receipt["ctrl"][0] = 0
    with pytest.raises(FrozenInstanceError):
        sub.ordinal = 2


def test_accepted_step_can_have_missing_failed_diagnostics():
    Spec, Substep, Window = records()
    sub = Substep(0, 1, 2, {7: True}, None, False, "snapshot failed")
    window = Window(Spec(2).sha256, 2, 1, 1, (sub,), None, "physical-diagnostics", 0)
    assert window.completed == 1 and window.substeps[0].states is None
    assert not window.substeps[0].diagnostics_complete


@pytest.mark.parametrize("fault", ("ordinal", "revision", "complete", "counts", "failed-index"))
def test_record_rejects_inconsistent_evidence(fault):
    Spec, Substep, Window = records()
    with pytest.raises((ValueError, TypeError)):
        sub = Substep(
            1 if fault == "ordinal" else 0,
            2,
            1 if fault == "revision" else 3,
            {7: True},
            None,
            fault == "complete",
        )
        Window(
            Spec(2).sha256, 2, 1, 0 if fault == "counts" else 1,
            (sub,), None, "physics", 1 if fault == "failed-index" else 0,
        )


@pytest.mark.cuda
@pytest.mark.tyx
@pytest.mark.parametrize("case,d", (("J", 1), ("J", 2), ("J", 4), ("S", 4)))
def test_actual_windows_match_held_host_route_and_independent_task(case, d, tmp_path):
    result = api().run_case(case, d, tmp_path)
    assert result["status"] == "PASS_CONTROL_WINDOW"
    assert result["policy_steps"] == 8 and result["physical_steps"] == 8 * d
    assert result["compared_world_substeps"] == 4 * 8 * d
    assert result["action_submissions"] == 8
    assert result["random_reset_consumed"]
    if case == "S":
        assert result["contact_world_substeps"] == 4 * 8 * d


@pytest.mark.cuda
@pytest.mark.tyx
def test_default_transition_and_binding_stay_unchanged():
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxTransition

    helper = api()
    torch = helper.require_cuda()
    with helper.make_environment("J", None) as default:
        assert "control_window" not in default.training_binding()
        assert default.decimation == 1 and default.policy_dt_s == default.h
        action = torch.zeros((4, 2), device=default.device)
        tr = default.step(action, world_keys=KEYS, expected_frame=default.current_frame)
        assert type(tr) is MuJoCoTyxTransition and not hasattr(tr, "control_window")
        assert all(tr.accepted_worlds.values())
        with helper.make_environment("J", 1) as explicit:
            other = explicit.step(action, world_keys=KEYS, expected_frame=explicit.current_frame)
            assert type(other).__name__ == "MuJoCoTyxWindowTransition"
            assert helper.json_text(tr.states) == helper.json_text(other.states)
            for name in ("observation", "reward", "terminated", "time_outs", "final_observation"):
                assert torch.equal(getattr(tr, name), getattr(other, name))
            assert explicit.training_binding()["control_window"] == explicit.control_window.binding(explicit.h)


@pytest.mark.cuda
@pytest.mark.tyx
def test_endpoint_termination_precedes_timeout_and_final_obs_precedes_reset(tmp_path):
    from phymeo.backends.mujoco_task_workspace import MuJoCoTyxTaskSpec

    helper = api()
    torch = helper.require_cuda()
    # J first four velocities have signs +,-,+,-. Large preserved margin separates them.
    task = MuJoCoTyxTaskSpec((), (0,), lower=(-1,), upper=(0,), weight=(1,))
    with helper.make_environment("J", 4, task_spec=task, horizons=(1, 1, 1, 1)) as env:
        tr = env.step(torch.zeros((4, 2), device=env.device), world_keys=KEYS, expected_frame=env.current_frame)
        assert tr.terminated.cpu().tolist() == [True, False, True, False]
        assert tr.time_outs.cpu().tolist() == [False, True, False, True]
        assert tr.final_mask.all().item()
        assert np.all(tr.episode_length == 1) and np.all(env.episode_length == 0)
        assert all(s.tick == 4 for s in tr.states.values())
        assert all(s.tick == 0 for s in env.runtime.snapshot().values())
        assert dict(env.reset_indices) == dict.fromkeys(KEYS, 1)
        assert torch.allclose(tr.final_observation[:, 0], torch.tensor([s.v[0] for s in tr.states.values()], device=env.device, dtype=torch.float32))
        assert not torch.equal(tr.final_observation, tr.observation)
        (tmp_path / "endpoint.json").write_text(helper.json_text(tr), encoding="utf-8")


@pytest.mark.cuda
@pytest.mark.tyx
@pytest.mark.parametrize("fault", ("physics-second", "snapshot-second", "endpoint-export", "reset-export"))
def test_late_failure_keeps_complete_actual_prefix(fault, monkeypatch, tmp_path):
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxEnvError

    helper = api()
    torch = helper.require_cuda()
    horizons = (1,) * 4 if fault == "reset-export" else (100,) * 4
    with helper.make_environment("J", 4, horizons=horizons) as env:
        original_step, original_snapshot, original_export = env.runtime.step, env.runtime.snapshot, env._export
        calls = {"step": 0, "snapshot": 0, "export": 0}

        def step(h):
            calls["step"] += 1
            if fault == "physics-second" and calls["step"] == 2:
                raise RuntimeError("fixed second physical request rejected")
            return original_step(h)

        def snapshot():
            calls["snapshot"] += 1
            if fault == "snapshot-second" and calls["snapshot"] == 2:
                raise RuntimeError("fixed second accepted-state observation failed")
            return original_snapshot()

        def export():
            calls["export"] += 1
            if (fault == "endpoint-export" and calls["export"] == 1) or (fault == "reset-export" and calls["export"] == 2):
                raise RuntimeError("fixed task publication failed")
            return original_export()

        monkeypatch.setattr(env.runtime, "step", step)
        monkeypatch.setattr(env.runtime, "snapshot", snapshot)
        monkeypatch.setattr(env, "_export", export)
        with pytest.raises(MuJoCoTyxEnvError) as caught:
            env.step(torch.full((4, 2), 0.1, device=env.device), world_keys=KEYS, expected_frame=env.current_frame)
        failure = caught.value
        window = failure.control_window
        expected = 1 if fault == "physics-second" else 2 if fault == "snapshot-second" else 4
        assert window.completed == expected
        assert window.attempted == (2 if fault in ("physics-second", "snapshot-second") else 4)
        assert calls["step"] == window.attempted
        assert failure.action_accepted and failure.state_accepted and env.needs_recovery
        assert env.last_transition is None and env.counters["transitions"] == 0
        assert window.action_receipt is not None
        for sub in window.substeps[:1]:
            assert sub.ordinal == 0 and sub.diagnostics_complete
            assert all(s.tick == 1 and s.last_step.tick == 1 for s in sub.states.values())
        if fault == "snapshot-second":
            last = window.substeps[-1]
            assert all(last.accepted_worlds.values()) and last.states is None
            assert not last.diagnostics_complete and window.failed_substep == 1
        if fault == "physics-second":
            assert not any(window.substeps[-1].accepted_worlds.values())
            assert window.failed_substep == 1
        if fault == "reset-export":
            assert failure.reset_accepted and dict(env.reset_indices) == dict.fromkeys(KEYS, 1)
        else:
            assert dict(env.reset_indices) == dict.fromkeys(KEYS, 0)
        assert all(s.tick == (0 if fault == "reset-export" else expected) for s in original_snapshot().values())
        (tmp_path / (fault + ".json")).write_text(helper.json_text({"window": window, "state": original_snapshot()}), encoding="utf-8")
        with pytest.raises(RuntimeError):
            env.reset((7,))


@pytest.mark.cuda
@pytest.mark.tyx
def test_bad_action_does_not_commit_any_substep():
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxEnvError

    helper = api()
    torch = helper.require_cuda()
    with helper.make_environment("J", 2) as env:
        before = env.runtime.checkpoint().to_bytes()
        action = torch.zeros((4, 2), device=env.device)
        action[0, 0] = float("nan")
        with pytest.raises(MuJoCoTyxEnvError) as caught:
            env.step(action, world_keys=KEYS, expected_frame=env.current_frame)
        assert not caught.value.action_accepted and not env.needs_recovery
        assert env.runtime.checkpoint().to_bytes() == before
        record = caught.value.control_window
        assert record is None or (record.attempted == record.completed == 0)


@pytest.mark.cuda
@pytest.mark.tyx
@pytest.mark.parametrize("operation", ("step", "reset", "capture", "restore"))
def test_same_thread_reentrancy_is_rejected_inside_window(operation, monkeypatch):
    helper = api()
    torch = helper.require_cuda()
    with helper.make_environment("J", 2) as env:
        capsule, state = env.runtime.checkpoint(), env.capture_training_state()
        original, checked = env.runtime.step, []
        action = torch.zeros((4, 2), device=env.device)

        def guarded(h):
            if not checked:
                with pytest.raises(RuntimeError, match="progress|window|boundary"):
                    if operation == "step":
                        env.step(action, world_keys=KEYS, expected_frame=env.current_frame)
                    elif operation == "reset":
                        env.reset()
                    elif operation == "capture":
                        env.capture_training_state()
                    else:
                        env.restore_training_state(capsule, state, expected_revision=env.runtime.checkpoint_revision)
                checked.append(True)
            return original(h)

        monkeypatch.setattr(env.runtime, "step", guarded)
        tr = env.step(action, world_keys=KEYS, expected_frame=env.current_frame)
        assert checked == [True] and tr.control_window.completed == 2
        assert all(tr.accepted_worlds.values()) and not env.needs_recovery


@pytest.mark.cuda
@pytest.mark.tyx
def test_actual_gru_update_restore_and_cross_decimation_rejection(tmp_path):
    result = api().continuation(tmp_path)
    assert result["status"] == "PASS_CONTROL_WINDOW_CONTINUATION"
    assert result["nonzero_hidden"] and result["partial_done_masks"]
    assert result["cross_decimation_refused"] and result["next_random_reset_equal"]
    assert result["policy_transitions"] == 32 and result["physical_world_steps"] == 64


@pytest.mark.cuda
@pytest.mark.tyx
def test_endpoint_mode_does_not_latch_an_intermediate_task_bound():
    from phymeo.backends.mujoco_task_workspace import MuJoCoTyxTaskSpec

    helper = api()
    torch = helper.require_cuda()
    # Original J: v(k)=.04*(1+.1*.002/1.01)^(-k). This fixed bound
    # lies below v(1) and above v(4); no mass/action/initial-state change.
    task = MuJoCoTyxTaskSpec((), (0,), lower=(-1,), upper=(0.03998,), weight=(1,))
    with helper.make_environment("J", 4, task_spec=task, horizons=(100,) * 4) as env:
        tr = env.step(torch.zeros((4, 2), device=env.device), world_keys=KEYS, expected_frame=env.current_frame)
        assert tr.control_window.substeps[0].states[7].v[0] > task.upper[0]
        assert tr.control_window.substeps[-1].states[7].v[0] < task.upper[0]
        assert not tr.final_mask.any().item()
        assert dict(env.reset_indices) == dict.fromkeys(KEYS, 0)
        assert all(s.tick == 4 for s in env.runtime.snapshot().values())
