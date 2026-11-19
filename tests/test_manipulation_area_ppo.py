"""Verify the actual upstream PPO minibatch against a two-world CUDA rollout."""

import json
import math
from contextlib import ExitStack

import pytest

from experiments.area_pressure.train_gripper import make_env, training_actions


@pytest.mark.cuda
def test_shared_manipulation_ppo_updates_from_real_contact(tmp_path, monkeypatch):
    import mujoco
    import torch
    from torch.distributions import Normal

    from phymeo.training.rsl_manipulation import ManipulationPpo
    from phymeo.training.rsl_mujoco_tyx import optimization_evidence

    def no_native_step(*args, **kwargs):
        pytest.fail("this rollout requires TYX CUDA")

    monkeypatch.setattr(mujoco, "mj_step", no_native_step)
    torch.manual_seed(720)
    steps = 24
    with ExitStack() as stack:
        env = make_env(stack, action_specs=training_actions())
        session = ManipulationPpo(env, steps=steps)
        ppo = session.algorithm
        storage = ppo.storage
        reset, retained = env.world_keys.index(41), env.world_keys.index(7)
        actionspecs = training_actions()
        with pytest.raises(RuntimeError, match="complete rollout"):
            session.update()
        before = {
            tag: v.detach().clone()
            for tag, v in (
                ("actor_trunk", ppo.actor.mlp[0].weight),
                ("actor_head", ppo.actor.mlp[-1].weight),
                ("critic_trunk", ppo.critic.mlp[0].weight),
                ("critic_head", ppo.critic.mlp[-1].weight),
            )
        }
        assert not ppo.optimizer.state
        frames = []
        for t in range(steps):
            obs = env.observation
            tr = session.collect_step()
            entry = storage.observations[t]
            assert storage.step == t + 1
            torch.testing.assert_close(
                entry["area_contact"], obs["area_object_contact"].flatten(1), atol=0, rtol=0
            )
            state = torch.cat(
                [
                    value
                    for k in ("q", "v", "act")
                    for value in (
                        torch.where(obs[k + "_mask"].bool(), obs[k], 0.0),
                        obs[k + "_mask"].float(),
                    )
                ],
                dim=1,
            )
            torch.testing.assert_close(entry["state"], state, atol=0, rtol=0)
            mask = entry["action_mask"]
            assert mask[reset].tolist() == [True, True, False, True]
            assert mask[retained].tolist() == [True, True, False, False]
            a = storage.actions[t]
            torch.testing.assert_close(a, tr.action, atol=0, rtol=0)
            assert not a[~mask].any()
            for i, k in enumerate(env.world_keys):
                spec = actionspecs[k]
                physical = (
                    torch.tensor(spec.bias.copy(), device=env.device)
                    + torch.tensor(spec.scale.copy(), device=env.device) * a[i, : spec.width].tanh()
                )
                torch.testing.assert_close(physical[:2], tr.native_channels[0][i], atol=0, rtol=0)
                torch.testing.assert_close(
                    physical[2:], tr.native_channels[1][i, list(spec.qfrc_indices)], atol=0, rtol=0
                )
            mean, std = (p[t] for p in storage.distribution_params)
            old_logp = torch.where(mask, Normal(mean, std).log_prob(a), 0.0).sum(-1)
            torch.testing.assert_close(storage.actions_log_prob[t, :, 0], old_logp)
            torch.testing.assert_close(storage.dones[t, :, 0].bool(), tr.terminated | tr.truncated)
            with torch.no_grad():
                endpoint = ppo.critic(session.pack(tr.final_observation)).flatten()
            raw_reward = tr.reward
            torch.testing.assert_close(
                storage.rewards[t, :, 0],
                raw_reward + ppo.gamma * torch.where(tr.truncated, endpoint, 0.0),
                atol=1e-7,
                rtol=0,
            )
            frame = dict(
                tick=t + 1,
                fn=obs["area_object_contact"][:, 0, 4].tolist(),
                ft_z=obs["area_object_contact"][:, 0, 7].tolist(),
                truncated=tr.truncated.tolist(),
                reward=raw_reward.tolist(),
            )
            if t == 18:
                assert tr.truncated[reset] and not tr.final_mask[retained]
                f = tr.final_observation["area_object_contact"]
                following = tr.observation["area_object_contact"]
                assert f[reset, 0, 4] > 0.01 and f[retained, 0, 4] > 0.01
                assert not following[reset, :, 2:].any()
                torch.testing.assert_close(following[retained], f[retained], atol=0, rtol=0)
                with torch.no_grad():
                    reset_v = ppo.critic(session.pack(tr.observation))[reset, 0]
                assert abs(endpoint[reset] - reset_v) > 1e-6
                assert abs(endpoint[reset] - storage.values[t, reset, 0]) > 1e-6
                frame.update(
                    terminal_value=endpoint[reset].item(),
                    reset_value=reset_v.item(),
                    entry_value=storage.values[t, reset, 0].item(),
                    stored_reward=storage.rewards[t, reset, 0].item(),
                )
            if t == 19:
                contact = entry["area_contact"].reshape(2, 4, 8)
                assert not contact[reset, :, 2:].any() and contact[retained, 0, 4] > 0.01
            frames.append(frame)
        with pytest.raises(RuntimeError, match="rollout is full"):
            session.collect_step()
        assert all(max(abs(r["ft_z"][i]) for r in frames) > 0.01 for i in range(2))
        # Inspect the same shuffled minibatch passed on to upstream update().
        # Full raw actions distinguish all samples independently of their order.
        flat = storage.observations.flatten(0, 1)
        actions = storage.actions.flatten(0, 1)
        origin = storage.mini_batch_generator
        seen = []
        actor_inputs, critic_inputs = [], []

        def capture(dest):
            def hook(module, args):
                if torch.is_grad_enabled():
                    dest.append(args[0]["area_contact"].detach().clone())

            return hook

        ah = ppo.actor.register_forward_pre_hook(capture(actor_inputs))
        ch = ppo.critic.register_forward_pre_hook(capture(critic_inputs))

        def audit(*args, **kwargs):
            for batch in origin(*args, **kwargs):
                matches = (batch.actions[:, None, :] == actions[None, :, :]).all(dim=-1)
                assert matches.sum(dim=0).eq(1).all() and matches.sum(dim=1).eq(1).all()
                indices = matches.long().argmax(dim=1)
                for name in flat.keys():
                    torch.testing.assert_close(
                        batch.observations[name], flat[name][indices], atol=0, rtol=0
                    )
                with torch.no_grad():
                    ppo.actor(batch.observations, stochastic_output=True)
                    valid = batch.observations["action_mask"]
                    altered = torch.where(valid, batch.actions, 1000.0)
                    torch.testing.assert_close(
                        ppo.actor.get_output_log_prob(altered),
                        batch.old_actions_log_prob.flatten(),
                        atol=1e-5,
                        rtol=1e-5,
                    )
                    m, s = ppo.actor.output_distribution_params
                    torch.testing.assert_close(
                        ppo.actor.output_entropy,
                        torch.where(valid, Normal(m, s).entropy(), 0.0).sum(-1),
                    )
                seen.append(batch.observations["area_contact"].clone())
                yield batch

        monkeypatch.setattr(storage, "mini_batch_generator", audit)
        try:
            losses = session.update()
        finally:
            ah.remove()
            ch.remove()
        assert len(seen) == len(actor_inputs) == len(critic_inputs) == 1
        for actual in (actor_inputs[0], critic_inputs[0]):
            torch.testing.assert_close(actual, seen[0], atol=0, rtol=0)
        assert storage.step == 0 and all(math.isfinite(v) for v in losses.values())
        evidence = optimization_evidence(ppo)
        assert evidence["optimizer_step"] == 1
        after = {
            "actor_trunk": ppo.actor.mlp[0].weight,
            "actor_head": ppo.actor.mlp[-1].weight,
            "critic_trunk": ppo.critic.mlp[0].weight,
            "critic_head": ppo.critic.mlp[-1].weight,
        }
        deltas = {key: (param - before[key]).abs().max().item() for key, param in after.items()}
        assert all(v > 0 for v in deltas.values())
        contact_start = flat["state"].size(-1)
        # Direct gradient to first-finger scalar Fn and world-z Ft input weights.
        for net in (ppo.actor, ppo.critic):
            gradient = net.mlp[0].weight.grad
            assert all(gradient[:, contact_start + j].abs().sum() > 0 for j in (4, 7))
        (tmp_path / "shared-ppo.json").write_text(
            json.dumps(
                dict(
                    worlds=env.world_keys,
                    steps=steps,
                    frames=frames,
                    losses=losses,
                    max_weight_change=deltas,
                    optimizer=evidence,
                ),
                allow_nan=False,
            ),
            encoding="utf-8",
        )


@pytest.mark.cuda
@pytest.mark.parametrize("failure", ["after_physics", "external_reset"])
def test_incomplete_or_stale_learner_cannot_update(failure, monkeypatch):
    from phymeo.training.rsl_manipulation import ManipulationPpo

    with ExitStack() as stack:
        env = make_env(stack, action_specs=training_actions())
        session = ManipulationPpo(env, steps=2)
        session.collect_step()
        if failure == "external_reset":
            env.reset((41,), expected_frame=env.current_frame)
        else:

            def reject(*args, **kwargs):
                raise RuntimeError("injected storage failure")

            monkeypatch.setattr(session.algorithm, "process_env_step", reject)
            with pytest.raises(RuntimeError, match="injected storage failure"):
                session.collect_step()
            assert env.last_transition.sequence == 2
        for run in (session.collect_step, session.update):
            with pytest.raises(RuntimeError, match="stale or interrupted"):
                run()
        assert session.algorithm.storage.step == 1
        assert not session.algorithm.optimizer.state


@pytest.mark.cuda
@pytest.mark.parametrize("mutation", ["direct_batch_reset", "close"])
def test_full_rollout_rechecks_live_env_before_updating(mutation):
    from phymeo.training.rsl_manipulation import ManipulationPpo

    with ExitStack() as stack:
        env = make_env(stack, action_specs=training_actions())
        session = ManipulationPpo(env, steps=1)
        session.collect_step()
        if mutation == "close":
            env.close()
        else:
            env.batch.reset((41,))
        with pytest.raises(RuntimeError, match="closed|external batch mutation"):
            session.update()
        assert session.algorithm.storage.step == 1 and not session.algorithm.optimizer.state


@pytest.mark.cuda
def test_nonfinite_optimizer_commit_invalidates_learner(monkeypatch):
    import torch

    from phymeo.training.rsl_manipulation import ManipulationPpo

    with ExitStack() as stack:
        env = make_env(stack, action_specs=training_actions())
        session = ManipulationPpo(env, steps=1)
        session.collect_step()
        update = session.algorithm.update

        def corrupt_state():
            losses = update()
            with torch.no_grad():
                next(iter(session.algorithm.optimizer.state.values()))["exp_avg"].fill_(
                    float("nan")
                )
            return losses

        monkeypatch.setattr(session.algorithm, "update", corrupt_state)
        with pytest.raises(FloatingPointError, match="nonfinite PPO"):
            session.update()
        with pytest.raises(RuntimeError, match="interrupted"):
            session.collect_step()


def test_gaussian_mask_excludes_padding_from_gradients_and_kl():
    import torch
    from tensordict import TensorDict
    from torch.distributions import Normal, kl_divergence

    from phymeo.training.rsl_manipulation import MaskedGaussianActor
    from phymeo.training.rsl_mujoco_tyx import ppo_config

    valid = torch.tensor([[True, False, False], [False, True, False]])
    obs = TensorDict({"state": torch.tensor([[0.4], [0.7]]), "action_mask": valid}, batch_size=[2])
    cfg = ppo_config()["actor"]
    cfg.pop("class_name")
    actor = MaskedGaussianActor(obs, {"actor": ["state", "action_mask"]}, "actor", 3, **cfg)
    torch.testing.assert_close(actor(obs)[~valid], torch.zeros(4))
    actor(obs, stochastic_output=True)
    old = tuple(p.detach().clone() for p in actor.output_distribution_params)
    with torch.no_grad():
        actor.mlp[-1].bias.add_(torch.tensor([0.03, -0.05, 100.0]))
        actor.distribution.log_std_param.add_(torch.tensor([0.01, 0.02, 1.0]))
    actor(obs, stochastic_output=True)
    new = actor.output_distribution_params
    per_dim = kl_divergence(Normal(*old), Normal(*new))
    assert per_dim[~valid].eq(0).all() and per_dim[valid].gt(0).all()
    torch.testing.assert_close(actor.get_kl_divergence(old, new), per_dim.sum(-1))
    actions = torch.tensor([[0.5, float("nan"), 1e10], [-1e10, 0.5, float("nan")]])
    loss = -(actor.get_output_log_prob(actions) + 0.05 * actor.output_entropy).sum()
    loss.backward()
    grad = actor.mlp[-1].bias.grad
    std_grad = actor.distribution.log_std_param.grad
    assert torch.isfinite(loss) and torch.isfinite(grad).all() and torch.isfinite(std_grad).all()
    assert grad[2] == 0 and std_grad[2] == 0 and torch.count_nonzero(grad[:2]) == 2
