"""N2 protocol tests; real CUDA physics, no old test-helper imports."""
import importlib.util
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.cuda, pytest.mark.tyx]
ROOT = Path(__file__).resolve().parents[2]
HORIZONS = (3, 5, 9, 11, 13, 15, 17, 19)


@pytest.fixture
def api():
    spec = importlib.util.spec_from_file_location(
        "n2_training_example", ROOT / "examples" / "train_mujoco_tyx_ppo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.require_cuda()
    return module


def zeros(api, env):
    return api.require_cuda().zeros((len(env.world_keys), env.num_actions), device=env.device)


@pytest.mark.parametrize("case", ("hinge-slide", "support-slide"))
def test_continuous_device_rollout_matches_independent_host_entry(case, api, tmp_path):
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime

    n1 = api.load_n1()
    with ExitStack() as owners:
        env = owners.enter_context(api.make_environment(case, horizons=HORIZONS))
        source = n1.fixture_source(case)
        model = source.fresh_native_model()
        reference = owners.enter_context(MuJoCoRuntime(
            source, mode="cuda", world_keys=env.world_keys,
            initial_states=n1.initial_states(case, model, env.world_keys),
        ))
        previous = env.get_observations()
        previous_copy = previous.clone()
        ended = []
        with (tmp_path / "protocol.jsonl").open("x", encoding="utf-8") as log:
            for tick in range(8):
                actions = zeros(api, env) + (tick + 1) / 100
                transition = env.step(
                    actions, world_keys=env.world_keys, expected_frame=env.current_frame
                )
                assert transition.sequence == tick + 1
                assert all(transition.accepted_worlds.values())
                reference.set_controls(transition.action_receipt.controls)
                assert all(reference.step(env.h).values())
                n1.assert_exact(transition.states, reference.snapshot())
                mask = transition.final_mask.cpu().tolist()
                keys = tuple(key for key, done in zip(env.world_keys, mask, strict=True) if done)
                if keys:
                    ended.append((tick + 1, keys))
                    reference.reset(keys)
                n1.assert_exact(env.runtime.snapshot(), reference.snapshot())
                actual = (
                    transition.observation.cpu().numpy(),
                    transition.reward.cpu().numpy(),
                    transition.terminated.cpu().numpy(),
                )
                # Observation is post-reset; reward/termination are pre-reset.
                n1.task_oracle(
                    transition.states, env.task_spec,
                    (transition.final_observation.cpu().numpy(), actual[1], actual[2]),
                )
                snapshot = env.runtime.snapshot()
                raw = np.array([
                    np.r_[state.q[list(env.task_spec.q_indices)],
                          state.v[list(env.task_spec.v_indices)]]
                    for state in snapshot.values()
                ], np.float32)
                np.testing.assert_allclose(actual[0], raw * env.task_spec.scale + env.task_spec.bias,
                                           rtol=2e-5, atol=2e-6)
                n1.append(log, {"event": "transition", **api.transition_record(transition)})
        assert (3, (env.world_keys[0],)) in ended
        assert (5, (env.world_keys[1],)) in ended
        assert api.require_cuda().equal(previous, previous_copy)
        assert env.counters["transitions"] == 8
        assert env.counters["termination_readback_bytes"] == 8 * len(env.world_keys)
        if case == "support-slide":
            assert all(any(c.normal_force > 0 for c in state.last_step.contacts)
                       for state in transition.states.values())


def test_final_observation_and_terminated_over_timeout(api):
    from phymeo.backends.mujoco_task_workspace import MuJoCoTyxTaskSpec

    spec = MuJoCoTyxTaskSpec((), (1,), weight=(1,), lower=(-1,), upper=(0.05,))
    with api.make_environment("hinge-slide", task_spec=spec,
                              horizons=(1, 1, 9, 11, 13, 15, 17, 19)) as env:
        start = env.get_observations()
        transition = env.step(zeros(api, env), world_keys=env.world_keys,
                              expected_frame=env.current_frame)
        assert transition.time_outs[0].item() and not transition.terminated[0].item()
        assert transition.terminated[1].item() and not transition.time_outs[1].item()
        assert transition.terminated.cpu().tolist() == [False] + [True] * 7
        assert transition.time_outs.cpu().tolist() == [True] + [False] * 7
        assert transition.final_mask.all().item()
        assert not api.require_cuda().equal(transition.final_observation, transition.observation)
        assert api.require_cuda().equal(transition.observation, start)
        assert np.all(transition.episode_id == 0)
        assert np.all(transition.episode_length == 1)
        assert np.all(env.episode_length == 0)
        assert np.all(env.episode_id == 1)


@pytest.mark.parametrize("bad", ("cpu", "dtype", "shape", "stride", "nan", "inf"))
def test_action_preflight_does_not_commit(api, bad):
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxEnvError

    torch = api.require_cuda()
    with api.make_environment("hinge-slide") as env:
        value = zeros(api, env)
        if bad == "cpu":
            value = value.cpu()
        elif bad == "dtype":
            value = value.double()
        elif bad == "shape":
            value = value[:, :-1].contiguous()
        elif bad == "stride":
            value = torch.zeros((len(env.world_keys), 2 * env.num_actions),
                                device=env.device)[:, ::2]
        elif bad == "nan":
            value[0, 0] = float("nan")
        else:
            value[0, 0] = float("inf")
        before = env.runtime.checkpoint().to_bytes()
        revision = env.runtime.checkpoint_revision
        with pytest.raises(MuJoCoTyxEnvError) as caught:
            env.step(value, world_keys=env.world_keys, expected_frame=env.current_frame)
        assert caught.value.phase == "action-preflight"
        assert not caught.value.action_accepted and not caught.value.state_accepted
        assert not env.needs_recovery and env.last_transition is None
        assert env.runtime.checkpoint_revision == revision
        assert env.runtime.checkpoint().to_bytes() == before
        assert all(env.step(zeros(api, env), world_keys=env.world_keys,
                            expected_frame=env.current_frame).accepted_worlds.values())


@pytest.mark.parametrize("bad", ("world-order", "old-frame"))
def test_label_and_frame_identity_refused(api, bad):
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxEnvError

    with api.make_environment("hinge-slide") as env:
        old = env.current_frame
        assert all(env.step(zeros(api, env), world_keys=env.world_keys,
                            expected_frame=old).accepted_worlds.values())
        before = env.runtime.checkpoint().to_bytes()
        with pytest.raises(MuJoCoTyxEnvError) as caught:
            env.step(zeros(api, env),
                     world_keys=env.world_keys[::-1] if bad == "world-order" else env.world_keys,
                     expected_frame=old if bad == "old-frame" else env.current_frame)
        assert not caught.value.action_accepted
        assert env.runtime.checkpoint().to_bytes() == before
        assert env.last_transition.sequence == 1


def test_physics_failure_retains_new_action_but_no_transition(api, monkeypatch):
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxEnvError

    with api.make_environment("hinge-slide") as env:
        before = env.runtime.snapshot()
        original = env.runtime.step
        def fail(h):
            raise RuntimeError("injected public physical failure")
        monkeypatch.setattr(env.runtime, "step", fail)
        with pytest.raises(MuJoCoTyxEnvError) as caught:
            env.step(zeros(api, env), world_keys=env.world_keys,
                     expected_frame=env.current_frame)
        failure = caught.value
        assert failure.phase == "physics" and failure.action_accepted
        assert not failure.state_accepted and env.needs_recovery
        assert failure.action_receipt is not None and env.last_transition is None
        after = env.runtime.snapshot()
        for key in env.world_keys:
            assert after[key].tick == before[key].tick == 0
            np.testing.assert_array_equal(after[key].q, before[key].q)
            np.testing.assert_array_equal(after[key].v, before[key].v)
        with pytest.raises(RuntimeError, match="recovery"):
            env.get_observations()
        monkeypatch.setattr(env.runtime, "step", original)
        env.reset()
        assert all(env.step(zeros(api, env), world_keys=env.world_keys,
                            expected_frame=env.current_frame).accepted_worlds.values())


def test_post_physics_export_failure_is_not_rollback(api, monkeypatch):
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxEnvError

    with api.make_environment("hinge-slide") as env:
        original = env.workspace.to_torch
        def fail(frame):
            raise RuntimeError("injected task export failure")
        monkeypatch.setattr(env.workspace, "to_torch", fail)
        with pytest.raises(MuJoCoTyxEnvError) as caught:
            env.step(zeros(api, env), world_keys=env.world_keys,
                     expected_frame=env.current_frame)
        assert caught.value.phase == "task-export"
        assert caught.value.action_accepted and caught.value.state_accepted
        assert env.needs_recovery and env.last_transition is None
        assert all(state.tick == 1 for state in env.runtime.snapshot().values())
        monkeypatch.setattr(env.workspace, "to_torch", original)
        with pytest.raises(RuntimeError, match="all"):
            env.reset((env.world_keys[0],))
        env.reset()
        assert not env.needs_recovery


def test_late_subset_reset_failure_reports_accepted_reset(api, monkeypatch):
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxEnvError

    with api.make_environment("hinge-slide", horizons=(1, 9, 9, 9, 9, 9, 9, 9)) as env:
        original = env.runtime.reset
        def late(keys=None):
            original(keys)
            raise RuntimeError("injected post-commit reset cleanup failure")
        monkeypatch.setattr(env.runtime, "reset", late)
        with pytest.raises(MuJoCoTyxEnvError) as caught:
            env.step(zeros(api, env), world_keys=env.world_keys,
                     expected_frame=env.current_frame)
        assert caught.value.phase == "subset-reset"
        assert caught.value.state_accepted and caught.value.reset_accepted
        assert env.last_transition is None and env.needs_recovery
        states = env.runtime.snapshot()
        assert states[env.world_keys[0]].tick == 0
        assert all(states[key].tick == 1 for key in env.world_keys[1:])


def test_close_borrows_runtime_and_keeps_exported_tensor_alive(api):
    torch = api.require_cuda()
    with api.make_environment("hinge-slide") as env:
        observation = env.get_observations()
        saved = observation.clone()
        runtime = env.runtime
        env.close()
        assert torch.equal(observation, saved)
        assert all(runtime.step(env.h).values())
        with pytest.raises(RuntimeError, match="closed"):
            env.get_observations()


def test_action_borrow_cleanup_failure_reports_real_commit(api, monkeypatch):
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxEnvError
    import tyx
    import tyx.torch as bridge

    with api.make_environment("hinge-slide") as env:
        original_borrow, original_free = bridge.from_torch, tyx.CudaArray.free
        borrowed_ids, failed = set(), []
        def borrow(*args, **kwargs):
            value = original_borrow(*args, **kwargs)
            borrowed_ids.add(id(value))
            return value
        def close_once(value):
            original_free(value)
            if id(value) in borrowed_ids and not failed:
                failed.append(True)
                raise RuntimeError("injected post-submit borrowed-view cleanup failure")
        monkeypatch.setattr(bridge, "from_torch", borrow)
        monkeypatch.setattr(tyx.CudaArray, "free", close_once)
        with pytest.raises(MuJoCoTyxEnvError) as caught:
            env.step(zeros(api, env), world_keys=env.world_keys,
                     expected_frame=env.current_frame)
        assert failed and caught.value.action_accepted
        assert caught.value.phase == "action-publication"
        assert caught.value.action_receipt is not None
        assert not caught.value.state_accepted
        assert env.needs_recovery and env.last_transition is None
        assert all(state.tick == 0 for state in env.runtime.snapshot().values())
        assert env.runtime.counters["device_control_submissions"] == 1


def test_external_runtime_mutation_cannot_be_adopted_by_subset_reset(api):
    with api.make_environment("hinge-slide") as env:
        initial = env.runtime.snapshot()
        assert all(env.runtime.step(env.h).values())
        before = env.runtime.checkpoint().to_bytes()
        revision = env.runtime.checkpoint_revision
        with pytest.raises(RuntimeError, match="all"):
            env.reset((env.world_keys[0],))
        assert env.needs_recovery
        assert env.runtime.checkpoint_revision == revision
        assert env.runtime.checkpoint().to_bytes() == before
        env.reset()
        assert not env.needs_recovery
        api.load_n1().assert_exact(initial, env.runtime.snapshot())
        assert all(env.step(zeros(api, env), world_keys=env.world_keys,
                            expected_frame=env.current_frame).accepted_worlds.values())
