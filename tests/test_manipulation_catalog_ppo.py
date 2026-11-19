"""One shuffled CUDA optimizer update from eight original, differently actuated robots."""

import json
from contextlib import ExitStack
from types import SimpleNamespace

import pytest
from test_manipulation_goal_language import encoder as encoder

from experiments.area_pressure.train_manipulation_catalog import STEPS, make_env, train_epoch
from phymeo.training.native_contact_policy import NativeContactPolicyInput


def test_native_object_sides_padding_and_inactive():
    import torch

    geometry = ("floor", "piece", "tip", "hand", "piece2")
    scene = SimpleNamespace(
        object_geom_names=("piece", "piece2"), finger_geom_names=("tip",), support_geom_name="floor"
    )
    workspace = SimpleNamespace(
        binding={"source": {"worlds": ((7, "arm", 0, 0),)}, "world_keys": (7,)},
        model_metadata={"arm": {"geom_names": geometry}},
    )
    policy = NativeContactPolicyInput(workspace, {7: scene}, "cpu")
    pairs = [[1, 1, 0, 1], [1, 1, 4, 0], [1, 0, 2, 1], [1, 1, 1, 4], [1, 1, 2, 3], [0, 1, -1, 99]]
    ids = torch.tensor([pairs])
    data = torch.full((1, 6, 14), float("nan"))
    data[0, :2, 13] = torch.tensor([2.0, 3.0])
    force = torch.full((1, 6, 6), float("nan"))
    force[0, 0, :3] = torch.tensor([1, 0, 2])
    force[0, 1, :3] = torch.tensor([0, 2, -3])
    values = dict(
        entry_meta=torch.ones(1, 1),
        entry_contact_ids=ids,
        entry_contact_data=data,
        entry_contact_wrench_world=force,
    )
    result = policy.encode(values)[0].tolist()
    assert result == [
        [1, 1, 1, 0, 0, 0, 0, 0],
        [0] * 8,
        [1, 1, 2, 2, 5, 1, -2, 5],
        [1, 1, 0, 0, 0, 0, 0, 0],
    ]
    values["entry_meta"].zero_()
    assert policy.encode(values)[0, :, 1:].count_nonzero() == 0


@pytest.mark.cuda
def test_catalog_cuda_update(encoder, monkeypatch, tmp_path):
    import torch

    from phymeo.training.rsl_manipulation import ManipulationPpo

    torch.manual_seed(720)
    with ExitStack() as stack:
        env, scenes = make_env(stack, encoder, min_horizon=24)
        session = ManipulationPpo(env, steps=STEPS)
        algo = session.algorithm
        assert "native_policy_input" in env.binding and "area_policy_input" not in env.binding
        assert len(set(scene.source_sha256 for scene in scenes.values())) == 8
        assert env.controllable_action_mask.sum(1).tolist() == [6, 6, 7, 8, 9, 2, 3, 4]
        assert env.observation["visible_target_mask"].sum(1).tolist() == [3] * 8
        inputs, actions, checked = [], [], []
        original_step = session.collect_step

        def checked_step():
            entry = session.pack(env.observation)
            inputs.append(entry.clone())
            tr = original_step()
            actions.append(tr.action.clone())
            assert all(tr.accepted_worlds.values())
            ctrl, qfrc, xfrc = tr.native_channels
            mask = entry["action_mask"]
            torch.testing.assert_close(ctrl[mask], 0.025 + 0.1 * tr.action[mask].tanh())
            assert not ctrl[~mask].any() and not qfrc.any() and not xfrc.any()
            features = tr.final_observation["native_object_contact"].tolist()
            for i, key in enumerate(env.world_keys):
                frame = tr.success_frames[key]
                assert features[i][2][7] == pytest.approx(frame.support_up_force_n, abs=1e-5)
                for j, load in enumerate(frame.finger_normal_forces_n):
                    assert features[i][j][4] == pytest.approx(load, abs=1e-5)
                if tr.final_mask[i]:
                    assert tr.observation["native_object_contact"][i, :, 1:].count_nonzero() == 0
            return tr

        generate = algo.storage.mini_batch_generator

        def checked_minibatches(*args, **kwargs):
            source = torch.cat(actions)
            expected = torch.cat(inputs)
            for batch in generate(*args, **kwargs):
                matches = (batch.actions[:, None, :] == source[None, :, :]).all(-1)
                assert matches.sum(0).eq(1).all() and matches.sum(1).eq(1).all()
                order = matches.long().argmax(1)
                for name in expected.keys():
                    torch.testing.assert_close(
                        batch.observations[name], expected[name][order], atol=0, rtol=0
                    )
                checked.append(len(order))
                yield batch

        monkeypatch.setattr(session, "collect_step", checked_step)
        monkeypatch.setattr(algo.storage, "mini_batch_generator", checked_minibatches)
        networks = (algo.actor, algo.critic)
        weights = [next(p for p in net.parameters() if p.ndim == 2) for net in networks]
        before = [p.detach().clone() for p in weights]
        report = train_epoch(session, scenes)
        assert checked == [256]
        assert report["model_buckets"] == 8 and report["optimizer"]["optimizer_step"] == 1
        assert report["optimizer"]["finite_gradient"]
        assert all((a != b).any() for a, b in zip(weights, before, strict=True))
        for row in report["worlds"]:
            assert row["accepted"] == STEPS and row["timeouts"] + row["successes"] >= 1
            assert row["support_steps"] > 0 and row["peak_support_n"] > 0
        (tmp_path / "catalog-update.json").write_text(json.dumps(report, allow_nan=False), "utf-8")
