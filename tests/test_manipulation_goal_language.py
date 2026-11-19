"""Check real text features, public target geometry and PPO row ownership on CUDA."""

import json
import os
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.area_pressure.train_goal_language import make_env
from phymeo.training.goal_language import FrozenMiniLM, GoalLanguageTable
from phymeo.training.manipulation_tasks import Goal, Target


@pytest.fixture(scope="module")
def encoder():
    path = os.environ.get("PHYMEO_MINILM_DIR")
    if path is None:
        path = (
            Path("artifacts/models/paraphrase-multilingual-MiniLM-L12-v2") / FrozenMiniLM.revision
        )
        if not path.is_dir():
            pytest.skip("optional pinned MiniLM model has not been provisioned")
    return FrozenMiniLM(path)


def test_real_multilingual_encoder(encoder, tmp_path):
    source = encoder.encode("把方块推到近处的蓝区")
    translation = encoder.encode("Push the box into the near blue zone")
    unrelated = encoder.encode("Heavy rain is expected this evening")
    assert source.dtype == np.float32 and source.shape == (384,)
    assert np.linalg.norm(source) == pytest.approx(1.0, abs=2e-6)
    assert source @ translation > source @ unrelated
    with pytest.raises(ValueError, match="128-token"):
        encoder.encode("word " * 200)
    with pytest.raises(ValueError, match="nonempty"):
        encoder.encode("  ")
    (tmp_path / "cosines.json").write_text(
        json.dumps(
            dict(translation=float(source @ translation), unrelated=float(source @ unrelated))
        ),
        encoding="utf8",
    )


def test_unequal_english_catalogs(encoder):
    goal = Goal(
        "pusht", "object", "blue", ("推到蓝区", "移进蓝色区域"), ("Push to blue", "Move to blue")
    )
    blue = Target("blue", "蓝区", "blue zone", (0.01, 0, 0), 0, (0.02, 0.02))
    orange = Target("orange", "橙区", "orange zone", (0.02, 0, 0), 0, (0.02, 0.02))
    scenes = {
        7: SimpleNamespace(goals=(goal,), targets=(blue,)),
        41: SimpleNamespace(goals=(goal,), targets=(orange, blue)),
    }
    table = GoalLanguageTable(encoder, scenes, (41, 7), "cpu", "en")
    obs = table.observe(dict.fromkeys(scenes, goal), {41: 1, 7: 0})
    assert obs["visible_target_mask"].tolist() == [[True, True], [True, False]]
    assert not obs["visible_targets"][1, 1].any()
    np.testing.assert_array_equal(obs["visible_targets"][0, 0, :384], encoder.encode("orange zone"))
    np.testing.assert_array_equal(obs["visible_targets"][0, 1], obs["visible_targets"][1, 0])
    for row, variant in enumerate(("Move to blue", "Push to blue")):
        np.testing.assert_array_equal(obs["instruction"][row], encoder.encode(variant))


@pytest.mark.cuda
def test_goal_conditioned_rollout(encoder, monkeypatch, tmp_path):
    import mujoco
    import torch

    from phymeo.training.rsl_manipulation import ManipulationPpo
    from phymeo.training.rsl_mujoco_tyx import optimization_evidence

    torch.manual_seed(720)
    monkeypatch.setattr(mujoco, "mj_step", lambda *a: pytest.fail("expected TYX CUDA stepping"))
    encoded = {}
    original = encoder.encode

    def record(text):
        assert text not in encoded
        encoded[text] = original(text)
        return encoded[text]

    monkeypatch.setattr(encoder, "encode", record)
    with ExitStack() as stack:
        env = make_env(stack, encoder)
        keys = env.world_keys
        first, second = keys
        initial = env.observation
        goals = env.goals
        near, far = goals[first], goals[second]
        catalog = env.contexts[first]["targets"]
        assert len(catalog) == 2 and env.contexts[second]["targets"] == catalog
        assert set(encoded) == {
            *near.instructions_zh,
            *far.instructions_zh,
            *(t.label_zh for t in catalog),
        }
        assert env.batch.layout[first].model_sha256 == env.batch.layout[second].model_sha256
        for name in (
            "q",
            "v",
            "action_mask",
            "area_object_contact",
            "visible_targets",
            "visible_target_mask",
        ):
            torch.testing.assert_close(initial[name][0], initial[name][1], rtol=0, atol=0)
        assert initial["visible_target_mask"].tolist() == [[True, True], [True, True]]

        def check_instruction(row, text):
            np.testing.assert_array_equal(row.cpu().numpy(), encoded[text])

        for i, target in enumerate(catalog):
            # The PushT oracle places the actual box footprint at each candidate pose.
            assert target.half_extents == (0.02, 0.02)
            token = initial["visible_targets"][0, i]
            check_instruction(token[:384], target.label_zh)
            np.testing.assert_allclose(
                token[384:].cpu(),
                [*target.position, np.sin(target.yaw), np.cos(target.yaw), *target.half_extents],
                rtol=1e-6,
                atol=1e-8,
            )
        check_instruction(initial["instruction"][0], near.instructions_zh[0])
        check_instruction(initial["instruction"][1], far.instructions_zh[0])
        assert not torch.equal(*initial["instruction"].unbind())

        checkpoint = env.checkpoint()
        switched = env.reset((first,), goals={first: far}, expected_frame=env.current_frame)
        check_instruction(switched["instruction"][0], far.instructions_zh[1])
        check_instruction(switched["instruction"][1], far.instructions_zh[0])
        torch.testing.assert_close(
            switched["visible_targets"], initial["visible_targets"], atol=0, rtol=0
        )
        env.restore(checkpoint, expected_revision=env.batch.checkpoint_revision)
        for k in ("instruction", "visible_targets", "visible_target_mask", "q", "v"):
            torch.testing.assert_close(env.observation[k], initial[k], rtol=0, atol=0)

        session = ManipulationPpo(env, steps=12)
        algo = session.algorithm
        with torch.no_grad():
            padded = dict(initial)
            padded["visible_target_mask"] = initial["visible_target_mask"].clone()
            padded["visible_target_mask"][1, 1] = False
            padded["visible_targets"] = initial["visible_targets"].clone()
            padded["visible_targets"][1, 1] = float("nan")
            assert not session.pack(padded)["visible_targets"][1].reshape(2, -1)[1].any()
            obs = session.pack(initial)
            mean = algo.actor(obs).clone()
            value = algo.critic(obs).clone()
            assert not torch.equal(mean[0], mean[1]) and not torch.equal(value[0], value[1])
            swap = obs.clone()
            swap["instruction"] = obs["instruction"].flip(0)
            torch.testing.assert_close(algo.actor(swap), mean.flip(0), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(algo.critic(swap), value.flip(0), atol=1e-6, rtol=1e-6)
            # Move both visible destinations on just one world, keeping language fixed.
            pose = obs.clone()
            pose["visible_targets"][0].view(2, -1)[:, 384] += 0.01
            assert not torch.equal(algo.actor(pose)[0], mean[0]) and not torch.equal(
                algo.critic(pose)[0], value[0]
            )
        before = [n.mlp[0].weight.detach().clone() for n in (algo.actor, algo.critic)]
        samples, timeouts, reward_sum = [], [], 0.0
        for tick in range(1, 13):
            start = env.observation
            episodes = env.episode_ids
            tr = session.collect_step()
            assert tr.accepted_worlds == dict.fromkeys(keys, True)
            for i, k in enumerate(keys):
                variants = goals[k].instructions_zh
                check_instruction(start["instruction"][i], variants[episodes[k] % len(variants)])
                torch.testing.assert_close(
                    tr.final_observation["instruction"][i], start["instruction"][i], atol=0, rtol=0
                )
                check_instruction(
                    tr.observation["instruction"][i], variants[env.episode_ids[k] % len(variants)]
                )
            for frame in (start, tr.final_observation, tr.observation):
                torch.testing.assert_close(
                    frame["visible_targets"], initial["visible_targets"], rtol=0, atol=0
                )
            assert tr.truncated.tolist() == [tick == 7, tick == 10]
            reward_sum += tr.reward.sum().item()
            if tr.truncated.any():
                owner = tr.truncated.nonzero().item()
                with torch.no_grad():
                    terminal = algo.critic(session.pack(tr.final_observation))[owner, 0]
                    following = algo.critic(session.pack(tr.observation))[owner, 0]
                assert abs(terminal - following) > 1e-6
                stored = algo.storage.rewards[tick - 1, owner, 0]
                torch.testing.assert_close(stored, tr.reward[owner] + algo.gamma * terminal)
                timeouts.append(
                    dict(
                        world=keys[owner],
                        tick=tick,
                        terminal=terminal.item(),
                        reset=following.item(),
                    )
                )
            samples.append((tr.action.clone(), session.pack(start)))

        generate = algo.storage.mini_batch_generator
        src_actions = torch.cat([a for a, _ in samples])
        rows = {name: torch.cat([row[name] for _, row in samples]) for name in samples[0][1].keys()}
        seen = []

        def checked_batches(*args, **kwargs):
            for batch in generate(*args, **kwargs):
                pairs = (batch.actions[:, None, :] == src_actions[None, :, :]).all(-1)
                assert pairs.sum(0).eq(1).all() and pairs.sum(1).eq(1).all()
                order = pairs.long().argmax(1)
                for name, expected in rows.items():
                    torch.testing.assert_close(
                        batch.observations[name], expected[order], rtol=0, atol=0
                    )
                seen.append(batch.observations["instruction"].clone())
                yield batch

        monkeypatch.setattr(algo.storage, "mini_batch_generator", checked_batches)
        inputs = []

        def capture(net, args):
            if torch.is_grad_enabled():
                inputs.append(args[0]["instruction"].clone())

        with (
            algo.actor.register_forward_pre_hook(capture),
            algo.critic.register_forward_pre_hook(capture),
        ):
            losses = session.update()
        assert len(inputs) == 2 and len(seen) == 1
        for features in inputs:
            torch.testing.assert_close(features, seen[0], atol=0, rtol=0)
        start = sum(rows[k].shape[1] for k in ("state", "area_contact", "action_mask"))
        deltas = {}
        for name, net, old in zip(
            ("actor", "critic"), (algo.actor, algo.critic), before, strict=True
        ):
            delta = (net.mlp[0].weight - old).abs()
            deltas[name] = [
                delta[:, start : start + 384].max().item(),
                delta[:, start + 2 * 384 : start + 2 * 384 + 7].max().item(),
            ]
            assert min(deltas[name]) > 0
        evidence = optimization_evidence(algo)
        assert evidence["optimizer_step"] == 1
        (tmp_path / "semantic-ppo.json").write_text(
            json.dumps(
                dict(
                    ticks=12,
                    worlds=keys,
                    distinct_texts=list(encoded),
                    timeouts=timeouts,
                    initial_mean=mean.flatten().tolist(),
                    initial_value=value.flatten().tolist(),
                    max_language_and_pose_deltas=deltas,
                    raw_reward_sum=reward_sum,
                    losses=losses,
                    optimizer=evidence,
                    language=env.binding["goal_language"],
                ),
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf8",
        )
