"""Resume a language-conditioned CUDA manipulation rollout across a fresh session."""

import json
import random
from contextlib import ExitStack
from copy import deepcopy

import numpy as np
import pytest
from test_manipulation_goal_language import encoder as encoder

from experiments.area_pressure.train_goal_language import make_env

pytestmark = pytest.mark.cuda


def _epoch(session):
    import torch

    algo = session.algorithm
    trace = []
    for tick in range(12):
        entry = session.pack(session.env.observation)
        transition = session.collect_step()
        with torch.no_grad():
            final = session.pack(transition.final_observation)
            terminal_v = algo.critic(final)
        # Keep independently owned observations, including both sides of each reset.
        trace.append(
            {
                "entry": {k: v.cpu().clone() for k, v in entry.items()},
                "final": {k: v.cpu().clone() for k, v in final.items()},
                "next": {
                    k: v.cpu().clone() for k, v in session.pack(transition.observation).items()
                },
                "action": transition.action.cpu().clone(),
                "terminal_v": terminal_v.cpu().clone(),
                "reward": transition.reward.cpu().clone(),
                "timeout": transition.truncated.cpu().clone(),
                "value": algo.storage.values[tick].cpu().clone(),
                "logp": algo.storage.actions_log_prob[tick].cpu().clone(),
                "credited_reward": algo.storage.rewards[tick].cpu().clone(),
            }
        )
        torch.testing.assert_close(
            algo.storage.rewards[tick].flatten(),
            transition.reward
            + algo.gamma * torch.where(transition.truncated, terminal_v.flatten(), 0.0),
            rtol=0,
            atol=0,
        )
    return trace, session.update()


@pytest.fixture(scope="module")
def reference(encoder, tmp_path_factory):
    import torch

    from phymeo.training.rsl_manipulation import ManipulationPpo

    filename = tmp_path_factory.mktemp("language-resume") / "first.ppo"
    torch.manual_seed(720)
    with ExitStack() as stack:
        env = make_env(stack, encoder)
        trainer = ManipulationPpo(env, steps=12)
        with pytest.raises(RuntimeError, match="completed"):
            trainer.save_checkpoint(filename)
        for i in range(12):
            trainer.collect_step()
            if i == 0:
                with pytest.raises(RuntimeError, match="nonempty"):
                    trainer.save_checkpoint(filename)
        trainer.update()
        trainer.save_checkpoint(filename)
        with pytest.raises(FileExistsError):
            trainer.save_checkpoint(filename)
        sentinels = random.random(), np.random.random(), torch.rand(3)
        trace, losses = _epoch(trainer)
        saved = deepcopy(trainer.algorithm.save())
        rng_end = (torch.get_rng_state(), torch.cuda.get_rng_state(env.device))
    return filename, sentinels, trace, losses, saved, rng_end


def test_fresh_session_matches_uninterrupted_update(encoder, reference, tmp_path):
    import torch

    from phymeo.training.rsl_manipulation import ManipulationPpo

    filename, sentinels, expected, expected_losses, params, rng_end = reference
    torch.manual_seed(913)
    with ExitStack() as stack:
        env = make_env(stack, encoder)
        trainer = ManipulationPpo(env, steps=12)
        old_frame = env.current_frame
        trainer.restore_checkpoint(filename, expected_revision=env.batch.checkpoint_revision)
        assert trainer.completed_updates == 1 and not env.is_current_frame(old_frame)
        assert random.random() == sentinels[0] and np.random.random() == sentinels[1]
        assert torch.equal(torch.rand(3), sentinels[2])
        actual, losses = _epoch(trainer)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert losses == expected_losses and trainer.completed_updates == 2
        reset_rows = [
            (i + 13, row["timeout"].nonzero().flatten().tolist())
            for i, row in enumerate(actual)
            if row["timeout"].any()
        ]
        assert reset_rows == [(14, [0]), (20, [1]), (21, [0])]
        assert any(row["entry"]["area_contact"].abs().max().item() > 0 for row in actual)
        assert not torch.equal(
            actual[1]["final"]["instruction"][0], actual[1]["next"]["instruction"][0]
        )
        result = trainer.algorithm.save()
        for key in ("actor_state_dict", "critic_state_dict"):
            torch.testing.assert_close(result[key], params[key], atol=0, rtol=0)
        adam = result["optimizer_state_dict"]
        assert adam["param_groups"] == params["optimizer_state_dict"]["param_groups"]
        torch.testing.assert_close(
            adam["state"], params["optimizer_state_dict"]["state"], rtol=0, atol=0
        )
        assert all(s["step"].item() == 2 for s in adam["state"].values())
        assert torch.equal(rng_end[0], torch.get_rng_state()) and torch.equal(
            rng_end[1], torch.cuda.get_rng_state(env.device)
        )
        (tmp_path / "resume-summary.json").write_text(
            json.dumps(
                dict(
                    samples=2 * len(actual),
                    resets=reset_rows,
                    losses=losses,
                    adam_states=len(adam["state"]),
                    checkpoint_bytes=filename.stat().st_size,
                    last_actions=actual[-1]["action"].tolist(),
                )
            ),
            encoding="utf-8",
        )


def test_restore_failure_boundaries(encoder, reference, tmp_path, monkeypatch):
    import torch

    from phymeo.training.manipulation_ppo_checkpoint import read_checkpoint, write_checkpoint
    from phymeo.training.mujoco_tyx_checkpoint import MuJoCoTyxTrainingRestoreError
    from phymeo.training.rsl_manipulation import ManipulationPpo

    filename = reference[0]
    physical, state = read_checkpoint(filename)
    bad = tmp_path / "bad-adam.ppo"
    del state["learner"]["optimizer_state_dict"]["state"][0]["exp_avg"]
    write_checkpoint(bad, physical, state)
    broken = tmp_path / "broken.ppo"
    data = bytearray(filename.read_bytes())
    data[-1] ^= 1
    broken.write_bytes(data)
    with ExitStack() as stack:
        env = make_env(stack, encoder)
        trainer = ManipulationPpo(env, steps=12)
        revision, frame, algo = env.batch.checkpoint_revision, env.current_frame, trainer.algorithm
        before_rng = (torch.get_rng_state(), torch.cuda.get_rng_state(env.device))
        for source, rev, why in (
            (bad, revision, "Adam"),
            (broken, revision, "checksum"),
            (filename, revision + 1, "stale"),
        ):
            with pytest.raises(ValueError, match=why):
                trainer.restore_checkpoint(source, expected_revision=rev)
            assert (
                env.is_current_frame(frame)
                and trainer.algorithm is algo
                and trainer.completed_updates == 0
            )
            assert torch.equal(torch.get_rng_state(), before_rng[0]) and torch.equal(
                torch.cuda.get_rng_state(env.device), before_rng[1]
            )

        def late(_state):
            torch.rand(1)
            torch.rand(1, device=env.device)
            raise RuntimeError("injected RNG publication failure")

        monkeypatch.setattr(trainer, "_install_rng", late)
        with pytest.raises(MuJoCoTyxTrainingRestoreError) as failed:
            trainer.restore_checkpoint(filename, expected_revision=revision)
        assert failed.value.physics_restored and "injected" in str(failed.value.__cause__)
        assert torch.equal(torch.get_rng_state(), before_rng[0])
        assert torch.equal(torch.cuda.get_rng_state(env.device), before_rng[1])
        assert env.batch.checkpoint_revision > revision and not env.is_current_frame(frame)
        with pytest.raises(RuntimeError, match="stale or interrupted"):
            trainer.collect_step()
    with ExitStack() as stack:
        env = make_env(stack, encoder, "en")
        trainer = ManipulationPpo(env, steps=12)
        current = env.current_frame
        with pytest.raises(ValueError, match="binding"):
            trainer.restore_checkpoint(filename, expected_revision=env.batch.checkpoint_revision)
        assert env.is_current_frame(current)
