"""Two kinematic layouts share one optimizer and preserve model-local contact ownership."""

import json
from contextlib import ExitStack

import numpy as np
import pytest

from experiments.area_pressure.train_gripper import make_env


def _check_published_forces(state, contact, world):
    # Independent pair/body lookup from each model's accepted solve, in newtons.
    roles = {"left": 0, "right": 1, "table": 2, "paddle": 3}
    pairs = state.last_step.raw_contact_fields["area_friction"]
    names = []
    torque = 0.0
    for pair in pairs:
        identity = pair["identity"]
        object_on_a = identity.compliant_body == "object"
        counterpart = identity.rigid_body if object_on_a else identity.compliant_body
        names.append(counterpart)
        samples = len(pair["sample_ids"])
        tangents = np.asarray(pair["tangential_force"]).reshape(-1, 3)
        force = tangents.sum(axis=0) * (1 if object_on_a else -1)
        expected = [1, 1, 1, bool(samples), np.sum(pair["normal_force"]), *force]
        np.testing.assert_allclose(contact[roles[counterpart]], expected, rtol=2e-6, atol=2e-6)
        if counterpart == "paddle" and samples:
            anchor = np.array((-0.02225, 0.03, 0))
            arms = np.asarray(pair["points"]) - anchor
            jac = np.cross((0, 0, 1), arms)
            np.testing.assert_allclose(pair["jacobian_b"][:, :, 0], jac, atol=2e-6)
            torque = np.cross(arms, pair["force_on_b"]).sum(axis=0)[2]
            np.testing.assert_allclose(
                torque, state.last_step.qfrc_constraint[0], rtol=2e-5, atol=2e-6
            )
    assert names == (["left", "right", "table"] if world == 41 else ["table", "paddle"])
    return torque


@pytest.mark.cuda
def test_two_kinematic_models_train_together(tmp_path, monkeypatch):
    import mujoco
    import torch

    from phymeo.training.rsl_manipulation import ManipulationPpo
    from phymeo.training.rsl_mujoco_tyx import optimization_evidence

    monkeypatch.setattr(mujoco, "mj_step", lambda *args: pytest.fail("expected TYX CUDA"))
    torch.manual_seed(720)
    expected_rows, frames, timeouts = [], [], []
    with ExitStack() as stack:
        env = make_env(stack, mixed_tools=True)
        keys = env.world_keys
        assert keys == (41, 7)
        g, p = (env.batch.layout[k] for k in keys)
        assert (g.nq, g.nv, g.nu, p.nq, p.nv, p.nu) == (9, 8, 2, 8, 7, 1)
        assert g.bucket != p.bucket and g.primary_sha256 != p.primary_sha256
        meta = env.workspace.model_metadata
        assert list(meta[g.model_key]["jnt_type"]) == [0, 2, 2]
        assert list(meta[p.model_key]["jnt_type"]) == [3, 0]
        assert list(meta[p.model_key]["jnt_qposadr"]) == [0, 1]
        assert list(meta[p.model_key]["jnt_dofadr"]) == [0, 1]
        session = ManipulationPpo(env, steps=24)
        algo = session.algorithm
        before = [net.mlp[0].weight.detach().clone() for net in (algo.actor, algo.critic)]
        for tick in range(1, 25):
            entry = env.observation
            tr = session.collect_step()
            assert all(tr.accepted_worlds.values()) and list(tr.accepted_worlds) == list(keys)
            assert tr.input_frame.ticks == tuple(tr.episode_length - 1)
            q_mask, v_mask = entry["q_mask"].bool(), entry["v_mask"].bool()
            assert q_mask.sum(1).tolist() == [9, 8] and v_mask.sum(1).tolist() == [8, 7]
            assert not q_mask[1, -1] and not v_mask[1, -1] and not entry["act_mask"].any()
            mask = algo.storage.observations[tick - 1]["action_mask"]
            assert mask.tolist() == [[True, True, False, True], [True, False, False, False]]
            raw = tr.action
            assert not raw[~mask].any()
            ctrl, force, wrench = tr.native_channels
            torch.testing.assert_close(ctrl[0], 0.48 + raw[0, :2].tanh())
            torch.testing.assert_close(ctrl[1, 0], 0.04 + 0.08 * raw[1, 0].tanh())
            assert ctrl[1, 1] == 0 and not force[1].any() and not wrench.any()
            expected_force = torch.zeros_like(force[0])
            expected_force[2], expected_force[0] = 0.22, 0.01 * raw[0, 3].tanh()
            torch.testing.assert_close(force[0], expected_force)
            parts = []
            for name in ("q", "v", "act"):
                valid = entry[name + "_mask"].bool()
                parts.extend((torch.where(valid, entry[name], 0), valid.float()))
            expected_rows.append(
                (
                    raw.clone(),
                    torch.cat(parts, 1),
                    entry["area_object_contact"].flatten(1),
                    mask.clone(),
                )
            )
            final = tr.final_observation
            roles = final["area_object_contact"]
            assert roles[:, :, 0].tolist() == [[1, 1, 1, 0], [0, 0, 1, 1]]
            assert not roles[1, :2].any() and not roles[0, 3].any()
            snapshot = env.batch.integration_snapshot()
            torque = None
            for i, key in enumerate(keys):
                if not tr.final_mask[i]:
                    moment = _check_published_forces(snapshot[key], roles[i].cpu().numpy(), key)
                    if key == 7:
                        torque = float(moment)
                np.testing.assert_allclose(
                    tr.success_frames[key].finger_normal_forces_n,
                    roles[i, :2, 4].cpu().numpy(),
                    atol=2e-6,
                )
            assert tr.truncated.tolist() == [tick == 19, tick == 13] and not tr.terminated.any()
            with torch.no_grad():
                terminal = algo.critic(session.pack(final)).squeeze(-1)
            stored = algo.storage.rewards[tick - 1, :, 0]
            torch.testing.assert_close(
                stored, tr.reward + 0.99 * torch.where(tr.truncated, terminal, 0.0)
            )
            if tr.truncated.any():
                owner = int(tr.truncated.nonzero().item())
                other = 1 - owner
                with torch.no_grad():
                    reset_value = algo.critic(session.pack(tr.observation))[owner, 0]
                assert abs(reset_value - terminal[owner]) > 1e-6
                for name in ("q", "v", "area_object_contact"):
                    torch.testing.assert_close(
                        tr.observation[name][other], final[name][other], atol=0, rtol=0
                    )
                assert not tr.observation["area_object_contact"][owner, :, 2:].any()
                assert (
                    tr.output_frame.ticks[owner] == 0
                    and tr.output_frame.ticks[other] == tr.evaluation_frame.ticks[other]
                )
                timeouts.append(
                    dict(
                        tick=tick,
                        world=keys[owner],
                        terminal=terminal[owner].item(),
                        reset=reset_value.item(),
                        stored=stored[owner].item(),
                    )
                )
            frames.append(
                dict(
                    tick=tick,
                    fn=roles[:, :, 4].tolist(),
                    ft=roles[:, :, 5:].tolist(),
                    motor=ctrl[1, 0].item(),
                    hinge=final["q"][1, 0].item(),
                    box_x=final["q"][1, 1].item(),
                    moment=torque,
                    reward=tr.reward.tolist(),
                )
            )

        assert max(f["fn"][0][0] for f in frames) > 0.01
        assert max(f["fn"][1][2] for f in frames) > 0.01
        assert max(f["fn"][1][3] for f in frames) > 0.01
        assert max(abs(f["moment"]) for f in frames if f["moment"] is not None) > 1e-5
        assert max(f["hinge"] for f in frames) > 1e-4 and max(f["box_x"] for f in frames) > 1e-5
        # Reconstruct the expected shuffled minibatch from the live observations
        # recorded above. A distinct action identifies each source world/tick.
        actions, states, contacts, masks = [
            torch.cat(column) for column in zip(*expected_rows, strict=True)
        ]
        generate = algo.storage.mini_batch_generator
        seen = []

        def checked_batches(*args, **kwargs):
            for batch in generate(*args, **kwargs):
                matches = (batch.actions[:, None, :] == actions[None, :, :]).all(-1)
                assert matches.sum(0).eq(1).all() and matches.sum(1).eq(1).all()
                order = matches.long().argmax(1)
                for key, expected in (
                    ("state", states),
                    ("area_contact", contacts),
                    ("action_mask", masks),
                ):
                    torch.testing.assert_close(
                        batch.observations[key], expected[order], rtol=0, atol=0
                    )
                seen.append(batch.observations["area_contact"].clone())
                yield batch

        monkeypatch.setattr(algo.storage, "mini_batch_generator", checked_batches)
        inputs = []

        def capture(net, args):
            inputs.append(args[0]["area_contact"].clone())

        with (
            algo.actor.register_forward_pre_hook(capture),
            algo.critic.register_forward_pre_hook(
                lambda net, args: capture(net, args) if torch.is_grad_enabled() else None
            ),
        ):
            losses = session.update()
        assert len(seen) == 1 and len(inputs) == 2
        for actual in inputs:
            torch.testing.assert_close(actual, seen[0], atol=0, rtol=0)
        changes = [
            (net.mlp[0].weight.detach() - initial).abs().max().item()
            for net, initial in zip((algo.actor, algo.critic), before, strict=True)
        ]
        evidence = optimization_evidence(algo)
        assert min(changes) > 0 and evidence["optimizer_step"] == 1
        (tmp_path / "mixed-ppo.json").write_text(
            json.dumps(
                dict(
                    world_keys=keys,
                    models=[g.model_key, p.model_key],
                    frames=frames,
                    timeouts=timeouts,
                    losses=losses,
                    weight_changes=changes,
                    optimizer=evidence,
                ),
                allow_nan=False,
            ),
            encoding="utf-8",
        )
