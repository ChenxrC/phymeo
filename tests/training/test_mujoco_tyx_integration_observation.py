"""Explicit reduced observation preserves accepted physics and training semantics."""

import importlib.util
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def example():
    spec = importlib.util.spec_from_file_location(
        "integration_observation_example", ROOT / "examples/evaluate_mujoco_tyx_learning.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mapping_order_independent(value):
    """Checkpoint JSON sorts map keys; mapping order is not physical state."""
    if isinstance(value, dict):
        if set(value) == {"mapping"}:
            return {
                "mapping": sorted(
                    [
                        [mapping_order_independent(k), mapping_order_independent(v)]
                        for k, v in value["mapping"]
                    ],
                    key=lambda pair: repr(pair[0]),
                )
            }
        return {key: mapping_order_independent(item) for key, item in value.items()}
    if isinstance(value, list):
        return [mapping_order_independent(item) for item in value]
    return value


@pytest.mark.parametrize("name", ("J", "S", "R"))
def test_complete_transition_equivalence_with_partial_reset(name, tmp_path, monkeypatch):
    from phymeo.backends.mujoco_tyx_state import MuJoCoTyxIntegrationState
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxVectorEnv
    from phymeo.training.mujoco_tyx_evaluation import EvidenceWriter, encode

    api = example()
    torch = api.tasks().require_cuda()
    with api.tasks().make_environment(name, count=8, scale=True) as full:
        with full.runtime.clone() as runtime:
            with MuJoCoTyxVectorEnv(
                runtime,
                full.task_spec,
                full.action_spec,
                h=full.h,
                horizons=tuple(int(x) for x in full.horizons),
                observation_mode="integration",
            ) as fast:
                assert full.observation_mode == "full"
                assert "state_observation" not in full.training_binding()
                assert fast.training_binding()["state_observation"] == "integration"
                with pytest.raises(AttributeError):
                    fast.observation_mode = "full"
                assert all(s.last_step is None for s in runtime.integration_snapshot().values())
                with pytest.raises(KeyError):
                    runtime.integration_snapshot((99999,))

                original = runtime._mj.mj_forward
                calls = []

                def observed(model, data):
                    calls.append(model)
                    return original(model, data)

                monkeypatch.setattr(runtime._mj, "mj_forward", observed)

                def forbidden(*args, **kwargs):
                    raise AssertionError("integration mode invoked full snapshot")

                monkeypatch.setattr(runtime, "snapshot", forbidden)
                retained = None
                writer = EvidenceWriter(tmp_path / "equivalence.jsonl.gz")
                try:
                    for tick in range(66):
                        action = torch.full(
                            (8, full.num_actions), 0.02 * ((tick % 5) - 2), device=full.device
                        )
                        a = full.step(
                            action, world_keys=full.world_keys, expected_frame=full.current_frame
                        )
                        before = len(calls)
                        b = fast.step(
                            action, world_keys=fast.world_keys, expected_frame=fast.current_frame
                        )
                        # Reset initialization still legitimately uses CPU forward.
                        assert len(calls) - before == int(b.final_mask.sum().item())
                        for field in (
                            "observation",
                            "reward",
                            "terminated",
                            "time_outs",
                            "final_observation",
                            "final_mask",
                            "episode_return",
                            "action",
                        ):
                            assert torch.equal(getattr(a, field), getattr(b, field)), field
                        for field in ("episode_id", "episode_length"):
                            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
                        for key in full.world_keys:
                            x, y = a.states[key], b.states[key]
                            assert type(y) is MuJoCoTyxIntegrationState
                            for field in (
                                "q",
                                "v",
                                "act",
                                "ctrl",
                                "time",
                                "tick",
                                "qfrc_applied",
                                "xfrc_applied",
                                "held_control",
                                "last_step",
                            ):
                                assert encode(getattr(x, field)) == encode(getattr(y, field)), field
                            assert not hasattr(y, "contacts") and not hasattr(y, "energy")
                            with pytest.raises(ValueError):
                                y.q.setflags(write=True)
                        if tick == 0:
                            retained = (b.states, encode(b.states))
                            with pytest.raises(FrozenInstanceError):
                                next(iter(b.states.values())).tick = 99
                        assert encode(retained[0]) == retained[1]
                        writer.write(
                            "paired",
                            {
                                "full_states": a.states,
                                "integration_states": b.states,
                                "reward": a.reward,
                                "reset_mask": a.final_mask,
                            },
                        )
                    assert fast.counters["resets"] == full.counters["resets"] == 1
                    saved = runtime.checkpoint()
                    before = encode(runtime.integration_snapshot())
                    fast.step(action, world_keys=fast.world_keys, expected_frame=fast.current_frame)
                    runtime.restore(saved, expected_revision=runtime.checkpoint_revision)
                    after = encode(runtime.integration_snapshot())
                    writer.write("restored", {"before": before, "after": after})
                    assert mapping_order_independent(after) == mapping_order_independent(before)
                finally:
                    writer.close()
                (tmp_path / "summary.json").write_text(
                    json.dumps(
                        {
                            "status": "PASS_PAIRED_OBSERVATION",
                            "task": name,
                            "worlds": 8,
                            "paired_steps": 66,
                            "trace": writer.receipt(),
                        }
                    ),
                    encoding="utf-8",
                )
        with pytest.raises(RuntimeError, match="closed"):
            runtime.integration_snapshot()


def test_mode_contract_and_post_accept_failure(tmp_path, monkeypatch):
    from phymeo.training.mujoco_tyx_checkpoint import MuJoCoTyxPpoSession
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxEnvError, MuJoCoTyxVectorEnv
    from phymeo.training.rsl_mujoco_tyx import RslMuJoCoTyxVecEnv

    api = example()
    torch = api.tasks().require_cuda()
    with api.tasks().make_environment("J") as full:
        with pytest.raises(ValueError, match="observation_mode"):
            MuJoCoTyxVectorEnv(
                full.runtime,
                full.task_spec,
                full.action_spec,
                h=full.h,
                horizons=(8,) * 8,
                observation_mode="silent",
            )
        with full.runtime.clone() as runtime:
            with MuJoCoTyxVectorEnv(
                runtime,
                full.task_spec,
                full.action_spec,
                h=full.h,
                horizons=(256,) * 8,
                observation_mode="integration",
            ) as fast:
                api.tasks().seed_all(14011)
                old = MuJoCoTyxPpoSession(RslMuJoCoTyxVecEnv(full), steps=4)
                old.update()
                old.save_checkpoint(tmp_path / "full-checkpoint")
                new = MuJoCoTyxPpoSession(RslMuJoCoTyxVecEnv(fast), steps=4)
                before = runtime.checkpoint_revision
                with pytest.raises(ValueError, match="binding"):
                    new.restore_checkpoint(tmp_path / "full-checkpoint", expected_revision=before)
                assert runtime.checkpoint_revision == before
                new.update()
                before = runtime.checkpoint_revision
                new.save_checkpoint(tmp_path / "integration-checkpoint")
                new.restore_checkpoint(
                    tmp_path / "integration-checkpoint", expected_revision=before
                )

                def fail(*args):
                    raise RuntimeError("observation failure after accepted physics")

                monkeypatch.setattr(runtime, "integration_snapshot", fail)
                with pytest.raises(MuJoCoTyxEnvError) as caught:
                    fast.step(
                        torch.zeros((8, 2), device=fast.device),
                        world_keys=fast.world_keys,
                        expected_frame=fast.current_frame,
                    )
                assert caught.value.phase == "physical-diagnostics"
                assert caught.value.state_accepted and caught.value.needs_recovery
                assert all(caught.value.accepted_worlds.values())


@pytest.mark.parametrize("worlds", (8, 128))
def test_fixed_snapshot_timing(worlds, tmp_path):
    """Same accepted support state; no threshold and no PPO throughput claim."""
    from phymeo.training.mujoco_tyx_evaluation import encode

    api = example()
    torch = api.tasks().require_cuda()
    with api.tasks().make_environment("S", count=worlds, scale=True) as env:
        env.step(
            torch.zeros((worlds, 1), device=env.device),
            world_keys=env.world_keys,
            expected_frame=env.current_frame,
        )
        env.runtime.snapshot()
        env.runtime.integration_snapshot()
        result = {
            "worlds": worlds,
            "scope": "same accepted state observation only",
            "samples": [],
            "newton_comparison": False,
        }
        for ordinal in range(6):
            row = {}
            names = ("full", "integration") if ordinal % 2 == 0 else ("integration", "full")
            for name in names:
                start = perf_counter()
                states = (
                    env.runtime.snapshot() if name == "full" else env.runtime.integration_snapshot()
                )
                row[name + "_seconds"] = perf_counter() - start
                row[name + "_state"] = states
            for key in env.world_keys:
                for field in ("q", "v", "ctrl", "time", "tick", "last_step"):
                    assert encode(getattr(row["full_state"][key], field)) == encode(
                        getattr(row["integration_state"][key], field)
                    )
            result["samples"].append({k: v for k, v in row.items() if k.endswith("_seconds")})
        (tmp_path / "timing.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
