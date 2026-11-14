"""B3 CUDA lifecycle contracts; author lane never executes project code.

Real M0 scenes and accepted-entry loads are used throughout. The one explicitly
synthetic success flag tests termination scheduling, not manipulation skill.
"""

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace

import numpy as np
import pytest
from test_manipulation_observation import loads
from test_mujoco_tyx_activation import save
from test_mujoco_tyx_body_observation import POSE_TOL, TWIST_TOL, native_bodies
from test_mujoco_tyx_heterogeneous_io import exact

from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.manipulation_checkpoint import (
    ManipulationEnvCheckpoint,
    capture_manipulation_checkpoint,
)
from phymeo.training.manipulation_env import HeterogeneousManipulationEnv, ManipulationTransition
from phymeo.training.manipulation_tasks import H, manipulation_scenes
from phymeo.training.mujoco_tyx_env import MuJoCoTyxActionSpec, MuJoCoTyxEnvError

pytestmark = pytest.mark.cuda
KEYS = (41, 7, 305, 11, 203, 59, 91, 2)
PHYSICAL_FIELDS = ("q", "v", "act", "ctrl", "qfrc_applied", "xfrc_applied", "time", "tick")


def owned_host(value):
    """Lossless evidence copies, including Torch tensors and source bytes."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if is_dataclass(value):
        return {field.name: owned_host(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: owned_host(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(owned_host(item) for item in value)
    return value


@contextmanager
def evidence(path):
    records = []
    try:
        yield records
    except BaseException as error:
        records.append({"failure": repr(error), "phase": getattr(error, "phase", None),
                        "needs_recovery": getattr(error, "needs_recovery", None)})
        raise
    finally:
        save(path, owned_host(records))


def declarations(keys):
    catalog = dict(zip(KEYS, manipulation_scenes(), strict=True))
    scenes = {key: catalog[key] for key in keys}
    # Model order is deliberately independent of a reversed world order.
    specs = tuple(
        MuJoCoTyxModelSpec(catalog[key].key, compile_mujoco_xml(catalog[key].mjcf, require_asset=False))
        for key in KEYS if key in scenes
    )
    models = {key: next(spec for spec in specs if spec.key == scene.key).compilation.fresh_native_model()
              for key, scene in scenes.items()}
    return scenes, specs, models


def make_env(stack, keys=KEYS, *, assets=None, horizons=None, auto_reset=True, channels=False):
    scenes, specs, models = declarations(keys) if assets is None else assets
    selected = {key: scenes[key] for key in keys}
    needed = {scene.key for scene in selected.values()}
    batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(
        tuple(spec for spec in specs if spec.key in needed),
        tuple(MuJoCoTyxWorldSpec(key, selected[key].key) for key in keys), mode="cuda",
    ))
    action_specs = {}
    for row, key in enumerate(keys):
        model = models[key]
        if channels:
            joint = int(model.joint(selected[key].robot_joint_names[0]).id)
            dof, body = int(model.jnt_dofadr[joint]), int(model.jnt_bodyid[joint])
            ctrl = (0,) if row == 0 else (0, int(model.nu) - 1)
            width = len(ctrl) + 3
            action_specs[key] = MuJoCoTyxActionSpec(
                ctrl, (dof,), ((body, 0), (body, 5)),
                scale=np.linspace(0.01, 0.04, width, dtype=np.float32),
                bias=np.linspace(-0.02, 0.015, width, dtype=np.float32),
            )
        else:
            action_specs[key] = MuJoCoTyxActionSpec(
                tuple(range(model.nu)), scale=np.full(model.nu, 0.1, np.float32),
                bias=np.full(model.nu, 0.025, np.float32),
            )
    env = stack.enter_context(HeterogeneousManipulationEnv(
        batch, selected, goals={key: scene.goals[scene.default_goal] for key, scene in selected.items()},
        action_specs=action_specs, horizons=horizons or dict.fromkeys(keys, 64), auto_reset=auto_reset,
    ))
    return env, batch, selected, models, action_specs


def actions(env, *, padding=float("nan")):
    import torch

    result = torch.zeros((env.num_envs, env.num_actions), device=env.device, dtype=torch.float32)
    result[~env.action_mask] = padding
    return result


def step(env, value=None, **kwargs):
    return env.step(actions(env) if value is None else value, world_keys=env.world_keys,
                    expected_frame=env.current_frame, **kwargs)


def metadata(env):
    return {name: dict(getattr(env, name)) for name in
            ("episode_ids", "episode_lengths", "returns", "history", "goals", "pending_reset")}


def physical(batch):
    return {key: {name: getattr(state, name) for name in PHYSICAL_FIELDS}
            for key, state in batch.integration_snapshot().items()}


def record(records, env, batch, transition=None, **extra):
    records.append(owned_host(dict(
        transition=transition, metadata=metadata(env), physical=physical(batch),
        last_step=batch.last_step, frame=env.current_frame, **extra,
    )))


def semantic_transition(transition):
    # CAS revisions, workspace UUIDs and reset-boundary serials are destination-local.
    return owned_host({name: getattr(transition, name) for name in (
        "sequence", "world_keys", "observation", "final_observation", "reward", "terminated",
        "truncated", "final_mask", "episode_id", "episode_length", "episode_return", "action",
        "native_channels", "accepted_worlds", "success_frames", "results", "goal_bindings",
    )})


def same_row(wide, row, narrow):
    for name, tensor in narrow.items():
        expected = owned_host(tensor)[0]
        actual = owned_host(wide[name])[row]
        slices = tuple(slice(0, size) for size in expected.shape)
        assert exact(actual[slices]) == exact(expected), name


def assert_error(error, phase, *, accepted=False):
    assert error.phase == phase
    assert error.needs_recovery is accepted
    assert error.action_accepted is accepted and error.state_accepted is accepted
    assert error.transition is None


@pytest.mark.parametrize("reverse", (False, True), ids=("w8-32-support", "w8-reordered"))
def test_eight_scenes_match_independent_w1_and_same_entry_loads(reverse, tmp_path):
    import mujoco as mj
    import torch

    keys = KEYS[::-1] if reverse else KEYS
    assets = declarations(keys)
    with ExitStack() as stack, evidence(tmp_path / "w8-w1-entry.json.gz") as records:
        env, batch, scenes, models, _ = make_env(stack, keys, assets=assets)
        singles = {key: make_env(stack, (key,), assets=assets)[:2] for key in keys}
        records.append({"sources": {key: scene.mjcf for key, scene in scenes.items()}})
        assert batch.capabilities["bucket_count"] == 8 and env.h == H == 0.001
        assert env.world_keys == keys and env.num_envs == 8
        assert env.workspace.binding["contact_tokens"] is True
        assert env.workspace.binding["endpoint_body_kinematics"] is True
        assert env.workspace.binding["entry_phase"] == "accepted-step-input"
        initial = owned_host(env.observation)
        assert not initial["entry_meta"][:, 0].any()
        assert not {"task_id", "goal", "task_kind"}.intersection(initial)
        for key, context in env.contexts.items():
            assert context["instructions_zh"] == env.goals[key].instructions_zh
            assert context["instructions_en"] == env.goals[key].instructions_en
            assert len(context["instructions_zh"]) >= 2 and len(context["instructions_en"]) >= 2
            assert context["targets"] == scenes[key].targets
            with pytest.raises(TypeError):
                context["instructions_en"] = ()
        supported, changed = set(), set()
        for tick in range(4 if reverse else 32):
            before = batch.integration_snapshot()
            input_frame = env.current_frame
            transition = step(env, actions(env, padding=float("inf") if tick % 2 else float("nan")))
            single_transitions = {key: step(owner) for key, (owner, _) in singles.items()}
            record(records, env, batch, transition, w1=single_transitions)
            assert isinstance(transition, ManipulationTransition)
            assert transition.sequence == tick + 1 and transition.input_frame is input_frame
            assert transition.evaluation_frame is transition.output_frame is env.current_frame
            assert transition.input_frame.ticks == (tick,) * 8
            assert transition.output_frame.ticks == (tick + 1,) * 8
            assert dict(transition.accepted_worlds) == dict.fromkeys(keys, True)
            assert not transition.reward.any().item() and not transition.final_mask.any().item()
            assert transition.episode_id.dtype == transition.episode_length.dtype == np.int64
            assert not transition.episode_id.flags.writeable and not transition.episode_length.flags.writeable
            assert transition.reward.dtype == torch.float32 and transition.reward.is_cuda
            assert transition.terminated.dtype == transition.truncated.dtype == torch.bool
            for tensor in transition.observation.values():
                assert tensor.is_cuda and torch.isfinite(tensor).all().item()
            for row, key in enumerate(keys):
                owner, single_batch = singles[key]
                assert exact(physical(batch)[key]) == exact(physical(single_batch)[key])
                same_row(transition.observation, row, single_transitions[key].observation)
                assert transition.success_frames[key] == single_transitions[key].success_frames[key]
                assert transition.results[key] == single_transitions[key].results[key]
                assert env.history[key] == transition.results[key].history
                assert env.episode_lengths[key] == tick + 1
                observed, model = transition.success_frames[key], models[key]
                receipt = batch.last_step.receipts[key]
                qa = int(model.jnt_qposadr[model.joint("object-free").id])
                assert qa > 0, "M0 object FREE is after robot scalar addresses"
                assert observed.phase == "accepted-entry" and observed.tick == tick
                assert observed.time_s == receipt.input_time and observed.episode == 0
                np.testing.assert_allclose(observed.object_position, receipt.input_q[qa:qa + 3], **POSE_TOL)
                np.testing.assert_array_equal(observed.object_wxyz, receipt.input_q[qa + 3:qa + 7])
                fingers, support = loads(scenes[key], model, receipt)
                np.testing.assert_allclose(observed.finger_normal_forces_n, fingers, rtol=3e-3, atol=3e-3)
                np.testing.assert_allclose(observed.support_up_force_n, support, rtol=3e-3, atol=3e-3)
                if support > 0.01:
                    supported.add(key)
                endpoint = owned_host(transition.observation["q"])[row, qa:qa + 3]
                if not np.array_equal(endpoint, receipt.input_q[qa:qa + 3]):
                    changed.add(key)
                if tick in (0, 16, 31):
                    oracle = native_bodies(mj, model, before[key])
                    records.append({"world": key, "tick": tick, "native_input_bodies": oracle})
                    body = int(model.body("object").id)
                    for field, source, tolerance in (
                        ("object_position", "body_position", POSE_TOL),
                        ("linear_velocity", "body_linear_velocity", TWIST_TOL),
                        ("angular_velocity", "body_angular_velocity", TWIST_TOL),
                    ):
                        np.testing.assert_allclose(getattr(observed, field), oracle[source][body], **tolerance)
                assert not transition.results[key].success  # No learned grasp/place/PushT claim.
        assert changed == set(keys), "entry and endpoint must be observably distinct"
        if not reverse:
            assert supported == set(keys), "all eight real FREE objects reach support in 32 steps"


def test_action_native_channels_padding_and_copy_ownership(tmp_path):
    import torch

    with ExitStack() as stack, evidence(tmp_path / "actions-ownership.json.gz") as records:
        env, batch, _, _, specs = make_env(stack, (KEYS[0], KEYS[6]), channels=True)
        producer = torch.cuda.Stream(device=env.device)
        with torch.cuda.stream(producer):
            raw = actions(env, padding=float("inf"))
            for row, key in enumerate(env.world_keys):
                raw[row, :specs[key].width] = torch.linspace(-0.4, 0.3, specs[key].width, device=env.device)
        transition = step(env, raw, producer_stream=producer)
        raw_host = owned_host(raw)
        record(records, env, batch, transition, raw=raw_host, specs=specs)
        mask = owned_host(env.action_mask)
        assert (~mask).any() and mask.dtype == np.bool_
        assert exact(owned_host(transition.action)) == exact(np.where(mask, raw_host, np.float32(0)))
        assert transition.action_receipt.world_keys == env.world_keys
        expected = [np.zeros_like(owned_host(value)) for value in transition.native_channels]
        for row, key in enumerate(env.world_keys):
            spec = specs[key]
            mapped = spec.bias + spec.scale * np.tanh(raw_host[row, :spec.width])
            offset = 0
            for indices, channel in ((spec.ctrl_indices, expected[0]), (spec.qfrc_indices, expected[1])):
                for index in indices:
                    channel[row, index] = mapped[offset]
                    offset += 1
            for body, component in spec.xfrc_indices:
                expected[2][row, body, component] = mapped[offset]
                offset += 1
            control, layout = batch.last_step.receipts[key].control, batch.layout[key]
            for name, reference in (
                ("ctrl", expected[0][row, :layout.nu]),
                ("qfrc_applied", expected[1][row, :layout.nv]),
                ("xfrc_applied", expected[2][row, :layout.nbody]),
            ):
                np.testing.assert_allclose(getattr(control, name), reference, rtol=2e-6, atol=2e-7)
        for actual, reference in zip(transition.native_channels, expected, strict=True):
            np.testing.assert_allclose(owned_host(actual), reference, rtol=2e-6, atol=2e-7)
        snapshot, observation = physical(batch), owned_host(env.observation)
        held = env.observation
        fresh = env.observation
        for name in held:
            assert held[name].data_ptr() != fresh[name].data_ptr()
            assert held[name].data_ptr() != transition.observation[name].data_ptr()
        for mapping in (env.episode_ids, env.episode_lengths, env.returns, env.history, env.goals):
            with pytest.raises(TypeError):
                mapping[env.world_keys[0]] = None
        binding = env.binding
        binding["h"] = 99.0
        assert env.binding["h"] == H
        raw.fill_(17)
        transition.action.fill_(23)
        for channel in transition.native_channels:
            channel.fill_(31)
        for tensor in transition.observation.values():
            tensor.zero_()
        env.action_mask.zero_()
        for tensor in fresh.values():
            tensor.zero_()
        assert exact(owned_host(env.observation)) == exact(observation)
        assert exact(physical(batch)) == exact(snapshot)
        assert exact(owned_host(env.action_mask)) == exact(mask)
        env.close()
        assert exact(owned_host(held)) == exact(observation)
        # The wrapper owns only its workspace; the borrowed batch still works.
        batch.set_controls({key: MuJoCoControl(np.zeros(row.nu)) for key, row in batch.layout.items()})
        assert batch.step(H) == dict.fromkeys(batch.world_keys, True)
        records.append(owned_host(dict(after_env_close=batch.last_step, held=held)))


@pytest.mark.parametrize("auto_reset", (True, False), ids=("automatic", "explicit"))
def test_independent_horizons_final_observation_and_selected_goal_reset(auto_reset, tmp_path):
    keys = (KEYS[0], KEYS[2], KEYS[7])
    horizons = dict(zip(keys, (2, 3, 5), strict=True))
    with ExitStack() as stack, evidence(tmp_path / "horizons.json.gz") as records:
        env, batch, scenes, _, _ = make_env(stack, keys, horizons=horizons, auto_reset=auto_reset)
        initial = owned_host(env.observation)
        episodes, lengths = dict.fromkeys(keys, 0), dict.fromkeys(keys, 0)
        for sequence in range(1, 7):
            before_frame, before_metadata = env.current_frame, metadata(env)
            transition = step(env)
            record(records, env, batch, transition)
            lengths = {key: lengths[key] + 1 for key in keys}
            selected = tuple(key for key in keys if lengths[key] == horizons[key])
            done = np.array([key in selected for key in keys])
            np.testing.assert_array_equal(transition.episode_id, [episodes[key] for key in keys])
            np.testing.assert_array_equal(transition.episode_length, [lengths[key] for key in keys])
            np.testing.assert_array_equal(owned_host(transition.final_mask), done)
            np.testing.assert_array_equal(owned_host(transition.truncated), done)
            assert not transition.terminated.any().item() and not transition.reward.any().item()
            assert not transition.episode_return.any().item()
            assert transition.sequence == sequence and transition.input_frame is before_frame
            assert transition.evaluation_frame.ticks == tuple(lengths[key] for key in keys)
            final, output = owned_host(transition.final_observation), owned_host(transition.observation)
            assert np.all(final["entry_meta"][:, 0] == 1)
            for row, key in enumerate(keys):
                assert transition.success_frames[key].tick == lengths[key] - 1
                assert transition.success_frames[key].episode == episodes[key]
                assert transition.goal_bindings[key] == before_metadata["goals"][key].binding
                if auto_reset and key in selected:
                    assert env.history[key] is None and env.episode_lengths[key] == 0
                    assert env.episode_ids[key] == episodes[key] + 1
                    assert output["entry_meta"][row, 0] == 0
                    for name in ("q", "v", "act", "clock"):
                        assert exact(output[name][row]) == exact(initial[name][row]), (key, name)
                    assert transition.output_frame.world_boundaries[row] != before_frame.world_boundaries[row]
                else:
                    for name in final:
                        assert exact(output[name][row]) == exact(final[name][row]), (key, name)
                    assert transition.output_frame.world_boundaries[row] == before_frame.world_boundaries[row]
                assert env.goals[key] is before_metadata["goals"][key]
            if selected and not auto_reset:
                assert transition.output_frame is transition.evaluation_frame
                assert dict(env.pending_reset) == dict(zip(keys, done.tolist(), strict=True))
                checkpoint = env.checkpoint().to_bytes()
                with pytest.raises(MuJoCoTyxEnvError) as raised:
                    step(env)
                assert_error(raised.value, "action-preflight")
                assert env.checkpoint().to_bytes() == checkpoint
                assert env.last_transition is transition
                new_goals = {key: scenes[key].goals[(scenes[key].default_goal + 1) % len(scenes[key].goals)]
                             for key in selected}
                before_reset, revision = physical(batch), batch.checkpoint_revision
                old_frame = env.current_frame
                reset_observation = env.reset(selected, goals=new_goals, expected_frame=old_frame)
                record(records, env, batch, reset_observation=reset_observation)
                assert batch.checkpoint_revision > revision
                for row, key in enumerate(keys):
                    if key in selected:
                        assert env.goals[key] is new_goals[key]
                        assert env.contexts[key]["instructions_en"] == new_goals[key].instructions_en
                        assert env.history[key] is None and env.episode_lengths[key] == 0
                        assert env.current_frame.world_boundaries[row] != old_frame.world_boundaries[row]
                    else:
                        assert exact(physical(batch)[key]) == exact(before_reset[key])
                        assert env.current_frame.world_boundaries[row] == old_frame.world_boundaries[row]
            for key in selected:
                episodes[key] += 1
                lengths[key] = 0
            assert dict(env.episode_ids) == episodes and dict(env.episode_lengths) == lengths
            assert not any(env.pending_reset.values())
        # Empty reset is a no-op; identical tick-zero resets still start new episodes.
        before, frame, checkpoint = metadata(env), env.current_frame, env.checkpoint().to_bytes()
        env.reset((), goals={}, expected_frame=frame)
        assert env.current_frame is frame and metadata(env) == before
        assert env.checkpoint().to_bytes() == checkpoint
        key = keys[-1]
        env.reset((key,), expected_frame=env.current_frame)
        first, frame, episode = physical(batch)[key], env.current_frame, env.episode_ids[key]
        env.reset((key,), expected_frame=frame)
        assert exact(physical(batch)[key]) == exact(first)
        assert env.episode_ids[key] == episode + 1 and env.history[key] is None
        assert env.current_frame.world_boundaries[-1] != frame.world_boundaries[-1]
        transition = step(env)
        record(records, env, batch, transition)
        assert transition.success_frames[key].tick == 0
        assert transition.success_frames[key].episode == episode + 1


def test_synthetic_success_only_checks_termination_precedence(tmp_path, monkeypatch):
    import phymeo.training.manipulation_env as module

    keys = KEYS[:2]
    with ExitStack() as stack, evidence(tmp_path / "synthetic-scheduler-only.json.gz") as records:
        env, batch, _, _, _ = make_env(stack, keys, horizons=dict.fromkeys(keys, 1))
        original = module.evaluate_goal
        actual_results = {}

        def scheduling_override(scene, goal, frame, history):
            result = original(scene, goal, frame, history)
            actual_results[frame.world] = result
            return replace(result, success=frame.world == keys[0])

        monkeypatch.setattr(module, "evaluate_goal", scheduling_override)
        transition = step(env)
        record(records, env, batch, transition, synthetic_scheduler_only=True, real_results=actual_results)
        assert not any(result.success for result in actual_results.values())
        np.testing.assert_array_equal(owned_host(transition.terminated), (True, False))
        np.testing.assert_array_equal(owned_host(transition.truncated), (False, True))
        np.testing.assert_array_equal(owned_host(transition.reward), (1.0, 0.0))
        np.testing.assert_array_equal(owned_host(transition.episode_return), (1.0, 0.0))
        np.testing.assert_array_equal(transition.episode_id, (0, 0))
        np.testing.assert_array_equal(transition.episode_length, (1, 1))
        assert transition.final_mask.all().item()
        assert transition.evaluation_frame.ticks == (1, 1) and transition.output_frame.ticks == (0, 0)
        assert dict(env.episode_ids) == dict.fromkeys(keys, 1)
        assert dict(env.returns) == dict.fromkeys(keys, 0.0)
        assert dict(env.history) == dict.fromkeys(keys, None)


def test_invalid_actions_goals_stale_frame_and_external_mutation(tmp_path):
    import torch

    keys = (KEYS[0], KEYS[7])
    with ExitStack() as stack, evidence(tmp_path / "preflight-recovery.json.gz") as records:
        env, batch, scenes, _, _ = make_env(stack, keys)
        initial_frame = env.current_frame
        transition = step(env)
        baseline = env.checkpoint()
        before, observation, revision = physical(batch), owned_host(env.observation), batch.checkpoint_revision
        valid = actions(env)
        nan, inf = valid.clone(), valid.clone()
        nan[0, 0], inf[0, 0] = float("nan"), float("inf")
        noncontiguous = torch.zeros((env.num_envs, 2 * env.num_actions), device=env.device)[:, ::2]
        assert not noncontiguous.is_contiguous()
        invalid = (
            (valid.cpu(), keys), (valid.double(), keys), (valid[:, :-1].contiguous(), keys),
            (noncontiguous, keys), (nan, keys), (inf, keys), (valid, keys[::-1]),
        )
        for value, order in invalid:
            with pytest.raises(MuJoCoTyxEnvError) as raised:
                env.step(value, world_keys=order, expected_frame=env.current_frame)
            records.append(owned_host(dict(input=value, order=order, error=repr(raised.value))))
            assert_error(raised.value, "action-preflight")
            assert not env.needs_recovery and env.last_transition is transition
            assert batch.checkpoint_revision == revision
            assert env.checkpoint().to_bytes() == baseline.to_bytes()
            assert exact(physical(batch)) == exact(before)
            assert exact(owned_host(env.observation)) == exact(observation)
        bad_resets = (
            ((keys[0], keys[0]), None), ((999999,), None),
            ((keys[0],), {keys[1]: env.goals[keys[1]]}),
            ((keys[0],), {keys[0]: env.goals[keys[1]]}),
            ((keys[0],), {keys[0]: replace(env.goals[keys[0]])}),
        )
        for selected, goals in bad_resets:
            with pytest.raises(ValueError):
                env.reset(selected, goals=goals, expected_frame=env.current_frame)
            assert env.checkpoint().to_bytes() == baseline.to_bytes()
            assert not env.needs_recovery
        with pytest.raises(MuJoCoTyxEnvError) as raised:
            env.step(valid, world_keys=keys, expected_frame=initial_frame)
        assert raised.value.phase == "frame-preflight" and raised.value.needs_recovery
        assert not raised.value.action_accepted and not raised.value.state_accepted
        assert exact(physical(batch)) == exact(before)
        env.restore(baseline, expected_revision=batch.checkpoint_revision)
        assert not env.needs_recovery and env.last_failure is None
        batch.set_controls({keys[0]: MuJoCoControl(np.zeros(batch.layout[keys[0]].nu))})
        external_revision = batch.checkpoint_revision
        with pytest.raises(MuJoCoTyxEnvError) as raised:
            step(env)
        assert raised.value.phase == "frame-preflight" and env.needs_recovery
        assert batch.checkpoint_revision == external_revision
        env.restore(baseline, expected_revision=external_revision)
        assert exact(physical(batch)) == exact(before)
        recovered = step(env)
        record(records, env, batch, recovered)
        assert recovered.sequence == transition.sequence + 1
        assert all(frame.tick == 1 for frame in recovered.success_frames.values())
        assert all(env.goals[key] is scenes[key].goals[scenes[key].default_goal] for key in keys)


def test_checkpoint_pairing_fresh_replay_and_semantic_rejection(tmp_path, monkeypatch):
    keys = (KEYS[0], KEYS[6])
    assets = declarations(keys)
    with ExitStack() as stack, evidence(tmp_path / "paired-checkpoint.json.gz") as records:
        env, batch, scenes, _, _ = make_env(stack, keys, assets=assets)
        step(env)
        step(env)
        old = env.checkpoint()
        changed_goal = scenes[keys[0]].goals[(scenes[keys[0]].default_goal + 1) % len(scenes[keys[0]].goals)]
        env.reset((keys[0],), goals={keys[0]: changed_goal}, expected_frame=env.current_frame)
        step(env)
        checkpoint = env.checkpoint()
        raw = checkpoint.to_bytes()
        (tmp_path / "paired-env.checkpoint").write_bytes(raw)
        decoded = ManipulationEnvCheckpoint.from_bytes(raw)
        assert decoded.to_bytes() == raw
        saved_metadata, saved_physics = metadata(env), physical(batch)
        saved_observation = owned_host(env.observation)
        expected = step(env)
        expected_semantics, expected_physics = semantic_transition(expected), physical(batch)
        record(records, env, batch, expected, checkpoint=raw)

        fresh, fresh_batch, _, _, _ = make_env(stack, keys, assets=assets)
        receipt = fresh.restore(decoded, expected_revision=fresh_batch.checkpoint_revision)
        record(records, fresh, fresh_batch, restore_receipt=receipt)
        assert exact(metadata(fresh)) == exact(saved_metadata)
        assert exact(physical(fresh_batch)) == exact(saved_physics)
        assert exact(owned_host(fresh.observation)) == exact(saved_observation)
        replay = step(fresh)
        record(records, fresh, fresh_batch, replay)
        assert exact(semantic_transition(replay)) == exact(expected_semantics)
        assert exact(physical(fresh_batch)) == exact(expected_physics)

        physics_capsule, state = decoded.read(env.binding)
        # Legitimate checksums with invalid semantic bindings must fail before physical CAS.
        broken = []
        bad = deepcopy(state)
        bad["goals"][0] = "0" * 64
        broken.append(bad)
        bad = deepcopy(state)
        bad["episode_ids"][0] += 1
        broken.append(bad)
        bad = deepcopy(state)
        bad["histories"][0]["tick"] += 1
        broken.append(bad)
        bad = deepcopy(state)
        bad["cursor_records"][0][1] += 1
        broken.append(bad)
        before_metadata, before_frame = metadata(env), env.current_frame
        before_checkpoint = env.checkpoint().to_bytes()
        revision = batch.checkpoint_revision

        def forbidden_restore(*args, **kwargs):
            raise AssertionError("malformed semantic state reached physical restore")

        for bad in broken:
            invalid = capture_manipulation_checkpoint(env.binding, physics_capsule, bad)
            records.append({"bad_metadata": bad, "capsule": invalid.to_bytes()})
            with monkeypatch.context() as guard:
                guard.setattr(batch, "restore", forbidden_restore)
                with pytest.raises(ValueError):
                    env.restore(invalid, expected_revision=revision)
            assert env.current_frame is before_frame and not env.needs_recovery
            assert exact(metadata(env)) == exact(before_metadata)
            assert batch.checkpoint_revision == revision
            assert env.checkpoint().to_bytes() == before_checkpoint
        with pytest.raises(ValueError):
            env.restore(decoded, expected_revision=revision - 1)
        assert env.current_frame is before_frame and batch.checkpoint_revision == revision

        # A coherent new cursor paired with old physics can only be rejected after
        # public restored entry is available. Physical acceptance is not rolled back.
        old_physics, _ = old.read(env.binding)
        mismatch = capture_manipulation_checkpoint(env.binding, old_physics, state)
        with pytest.raises(MuJoCoTyxEnvError) as raised:
            env.restore(mismatch, expected_revision=revision)
        error = raised.value
        records.append(owned_host(dict(mismatched_capsule=mismatch.to_bytes(), error=repr(error),
                                       accepted_physics=physical(batch))))
        assert error.phase == "training-restore" and error.state_accepted and error.needs_recovery
        assert not error.action_accepted and error.transition is None
        assert batch.checkpoint_revision > revision and env.needs_recovery
        assert exact(metadata(env)) == exact(before_metadata)
        assert env.current_frame is before_frame
        with pytest.raises(MuJoCoTyxEnvError):
            env.checkpoint()
        env.restore(decoded, expected_revision=batch.checkpoint_revision)
        assert not env.needs_recovery and env.last_failure is None and env.last_transition is None
        replay = step(env)
        record(records, env, batch, replay)
        assert exact(semantic_transition(replay)) == exact(expected_semantics)
        assert exact(physical(batch)) == exact(expected_physics)


@pytest.mark.parametrize("phase", ("success-evaluation", "task-export"))
def test_late_failure_poison_and_paired_recovery(phase, tmp_path, monkeypatch):
    import phymeo.training.manipulation_env as module

    keys = (KEYS[0], KEYS[2], KEYS[7])
    with ExitStack() as stack, evidence(tmp_path / "late-failure.json.gz") as records:
        env, batch, _, _, _ = make_env(stack, keys)
        first = step(env)
        checkpoint = env.checkpoint()
        expected = step(env)
        expected_semantics, expected_physics = semantic_transition(expected), physical(batch)
        record(records, env, batch, expected)
        env.restore(checkpoint, expected_revision=batch.checkpoint_revision)
        before_frame, before_metadata = env.current_frame, metadata(env)
        revision = batch.checkpoint_revision
        last_transition = env.last_transition
        calls = []
        original_evaluate, original_refresh = module.evaluate_goal, env.workspace.refresh

        def fail_last_world(scene, goal, frame, history):
            result = original_evaluate(scene, goal, frame, history)
            calls.append(frame.world)
            if frame.world == keys[-1]:
                raise RuntimeError("injected failure after last world's real evaluation")
            return result

        def fail_after_public_refresh(*args, **kwargs):
            frame = original_refresh(*args, **kwargs)
            calls.append(frame)
            raise RuntimeError("injected failure after full accepted task export")

        with monkeypatch.context() as guard:
            if phase == "success-evaluation":
                guard.setattr(module, "evaluate_goal", fail_last_world)
            else:
                guard.setattr(env.workspace, "refresh", fail_after_public_refresh)
            with pytest.raises(MuJoCoTyxEnvError) as raised:
                step(env)
        error = raised.value
        record(records, env, batch, failure=repr(error), phase=error.phase, calls=calls,
               action_receipt=error.action_receipt)
        assert_error(error, phase, accepted=True)
        assert dict(error.accepted_worlds) == dict.fromkeys(keys, True)
        assert error.action_receipt is not None and not error.reset_accepted
        assert env.last_transition is last_transition and env.current_frame is before_frame
        assert env.needs_recovery and env.last_failure is error
        assert exact(metadata(env)) == exact(before_metadata)
        assert exact(physical(batch)) == exact(expected_physics)
        assert batch.checkpoint_revision > revision
        if phase == "success-evaluation":
            assert calls == list(keys)
        else:
            assert len(calls) == 1 and calls[0].ticks == (2,) * len(keys)
        for operation in (
            lambda: step(env),
            lambda: env.reset(expected_frame=env.current_frame),
            env.checkpoint,
            lambda: env.observation,
        ):
            with pytest.raises(MuJoCoTyxEnvError):
                operation()
            assert env.last_failure is error, "recovery guards preserve the original accepted failure"
        env.restore(checkpoint, expected_revision=batch.checkpoint_revision)
        assert not env.needs_recovery and env.last_failure is None
        replay = step(env)
        record(records, env, batch, replay)
        assert replay.sequence == first.sequence + 1
        assert exact(semantic_transition(replay)) == exact(expected_semantics)
        assert exact(physical(batch)) == exact(expected_physics)
